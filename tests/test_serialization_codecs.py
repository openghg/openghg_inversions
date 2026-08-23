"""Tests for the private serialization codecs shared by artifact schemas."""

import json

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions._serialization_codecs import (
    _TAGGED_JSON_VALUE_ENCODING,
    _decode_serialized_bool,
    _decode_tagged_json_value,
    _encode_tagged_json_value,
    _numpy_scalar_json_default,
)
from openghg_inversions.source_covariance import IndependentSourceCovariance


def _source_covariance() -> IndependentSourceCovariance:
    """Return a small two-source covariance for dispatch tests."""
    latitude = xr.DataArray([50.0, 51.0], dims="latitude")
    longitude = xr.DataArray([-2.0, -1.0], dims="longitude")
    return IndependentSourceCovariance(
        {
            "inventory-b": SeparableExponentialCovariance(latitude, longitude, sigma=1.5),
            "inventory-a": SeparableExponentialCovariance(latitude, longitude, sigma=0.5),
        }
    )


def _source_rhs(source_labels: list[object]) -> xr.DataArray:
    """Return a labelled source right-hand side with non-native dimension order."""
    return xr.DataArray(
        np.arange(3 * 2 * len(source_labels) * 2, dtype=float).reshape(3, 2, len(source_labels), 2),
        dims=("observation", "longitude", "native_source", "latitude"),
        coords={
            "observation": ["obs-1", "obs-2", "obs-3"],
            "longitude": [-2.0, -1.0],
            "native_source": xr.DataArray(
                source_labels,
                dims="native_source",
                attrs={"long_name": "emissions inventory"},
            ),
            "latitude": [50.0, 51.0],
            "inventory_kind": ("native_source", [f"kind-{index}" for index in range(len(source_labels))]),
            "quality": (
                ("observation", "native_source"),
                np.arange(len(source_labels) * 3).reshape(3, len(source_labels)),
            ),
            "mixed_order": (
                ("longitude", "native_source", "observation"),
                np.arange(2 * len(source_labels) * 3).reshape(2, len(source_labels), 3),
            ),
        },
        name="native_rhs",
        attrs={"units": "arbitrary"},
    )


def test_tagged_json_scalar_encoding_name_is_versioned() -> None:
    """The persisted tagged-scalar codec name remains explicitly versioned."""
    assert _TAGGED_JSON_VALUE_ENCODING == "tagged_json_v1"


@pytest.mark.parametrize(
    ("value", "encoded"),
    [
        ("land", '["str","land"]'),
        (True, '["bool",true]'),
        (False, '["bool",false]'),
        (7, '["int",7]'),
        (-2, '["int",-2]'),
        (1.25, '["float",1.25]'),
        (
            ("outer", (np.int64(3), np.bool_(False)), 2.5),
            r'["tuple",["[\"str\",\"outer\"]","[\"tuple\",[\"[\\\"int\\\",3]\",'
            r'\"[\\\"bool\\\",false]\"]]","[\"float\",2.5]"]]',
        ),
    ],
)
def test_tagged_json_scalar_has_stable_bytes_and_roundtrips(value: object, encoded: str) -> None:
    """Supported Python scalars and nested tuples retain type and stable bytes."""
    assert _encode_tagged_json_value(value) == encoded
    assert _decode_tagged_json_value(encoded) == value


@pytest.mark.parametrize(
    ("value", "encoded", "expected"),
    [
        (np.str_("sea"), '["str","sea"]', "sea"),
        (np.bool_(True), '["bool",true]', True),
        (np.int32(4), '["int",4]', 4),
        (np.float32(1.5), '["float",1.5]', 1.5),
    ],
)
def test_tagged_json_scalar_normalises_numpy_scalars(
    value: np.generic,
    encoded: str,
    expected: object,
) -> None:
    """NumPy scalar labels serialize as their equivalent built-in types."""
    actual_encoded = _encode_tagged_json_value(value)

    assert actual_encoded == encoded
    decoded = _decode_tagged_json_value(actual_encoded)
    assert decoded == expected
    assert type(decoded) is type(expected)


@pytest.mark.parametrize("value", [None, [1, 2], {"label": 1}, 1 + 2j, np.array(1)])
def test_tagged_json_scalar_rejects_unsupported_values(value: object) -> None:
    """The tagged codec rejects values outside its documented scalar algebra."""
    with pytest.raises(TypeError, match="Unsupported"):
        _encode_tagged_json_value(value)


@pytest.mark.parametrize(
    "encoded",
    [
        "not JSON",
        "null",
        "{}",
        "[]",
        '["int"]',
        '["int",1,2]',
        "[1,1]",
        '["future",1]',
        '["bool",0]',
        '["int",true]',
        '["int",1.0]',
        '["float","1.0"]',
        '["str",1]',
        '["tuple","not-a-list"]',
        '["tuple",[["int",1]]]',
    ],
)
def test_tagged_json_scalar_rejects_malformed_json_tags_and_payloads(encoded: str) -> None:
    """Malformed JSON, envelopes, tags, and tag-specific payloads fail closed."""
    with pytest.raises(ValueError):
        _decode_tagged_json_value(encoded)


@pytest.mark.parametrize("encoded", [None, 1, b'["int",1]', ["int", 1]])
def test_tagged_json_scalar_requires_serialized_text(encoded: object) -> None:
    """The decoder does not coerce arbitrary objects into serialized text."""
    with pytest.raises((TypeError, ValueError)):
        _decode_tagged_json_value(encoded)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.bool_(True), True),
        (np.int16(5), 5),
        (np.float32(2.5), 2.5),
        (np.str_("coast"), "coast"),
    ],
)
def test_numpy_scalar_json_default_returns_builtin_scalar(value: np.generic, expected: object) -> None:
    """The JSON default hook converts NumPy scalars into built-in JSON values."""
    actual = _numpy_scalar_json_default(value)

    assert actual == expected
    assert type(actual) is type(expected)
    assert json.loads(json.dumps({"value": value}, default=_numpy_scalar_json_default)) == {"value": expected}


@pytest.mark.parametrize("value", [object(), np.array([1]), [np.int64(1)]])
def test_numpy_scalar_json_default_rejects_other_values(value: object) -> None:
    """The NumPy JSON hook refuses non-scalar values instead of guessing."""
    with pytest.raises(TypeError, match="not JSON serializable"):
        _numpy_scalar_json_default(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (False, False),
        (True, True),
        (0, False),
        (1, True),
        (np.bool_(False), False),
        (np.int8(0), False),
        (np.uint64(1), True),
    ],
)
def test_serialized_bool_accepts_only_booleans_and_integer_bits(value: object, expected: bool) -> None:
    """Serialized Boolean decoding accepts Boolean values and integer zero or one."""
    actual = _decode_serialized_bool(value, "enabled")

    assert actual is expected


@pytest.mark.parametrize("value", [-1, 2, 0.0, 1.0, "0", "true", None, np.float64(1.0)])
def test_serialized_bool_rejects_coercible_but_noncanonical_values(value: object) -> None:
    """Serialized Boolean decoding rejects values that merely coerce to Boolean."""
    with pytest.raises(ValueError, match="Serialized enabled must be Boolean or integer 0/1"):
        _decode_serialized_bool(value, "enabled")


@pytest.mark.parametrize(
    "source_labels",
    [
        ["inventory-a", "inventory-b"],
        ["inventory-b"],
        ["inventory-b", "inventory-a", "inventory-extra"],
        ["inventory-b", "inventory-b"],
        [1, 2],
    ],
    ids=["reordered", "missing", "extra", "duplicate", "wrong-type"],
)
def test_source_dispatch_requires_the_exact_configured_index(source_labels: list[object]) -> None:
    """Xarray exact alignment rejects reordered, missing, extra, duplicate, or mistyped labels."""
    covariance = _source_covariance()

    with pytest.raises(ValueError, match=r"join='exact'|align|native_source"):
        covariance.apply(_source_rhs(source_labels))


def test_source_dispatch_preserves_labelled_array_identity() -> None:
    """Per-source selection and concat preserve dimensions, metadata, and auxiliary coordinates."""
    covariance = _source_covariance()
    rhs = _source_rhs(list(covariance.source_labels))

    result = covariance.apply(rhs)

    assert result.dims == rhs.dims
    assert result.name == rhs.name
    assert result.attrs == rhs.attrs
    xr.testing.assert_identical(result.coords.to_dataset(), rhs.coords.to_dataset())


def test_source_dispatch_does_not_compute_lazy_source_coordinates() -> None:
    """Concat bypasses comparisons of source-dependent auxiliary coordinates."""
    covariance = _source_covariance()
    rhs = _source_rhs(list(covariance.source_labels)).assign_coords(
        lazy_equal=(
            ("native_source", "observation"),
            da.from_array(np.ones((2, 3)), chunks=(1, 2)),
        )
    )

    result = covariance.apply(rhs)

    assert isinstance(result.coords["lazy_equal"].data, da.Array)
    assert result.coords["lazy_equal"].data is rhs.coords["lazy_equal"].data
    assert result.coords["lazy_equal"].dims == ("native_source", "observation")
