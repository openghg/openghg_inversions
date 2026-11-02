"""Tests for the private serialization codecs shared by artifact schemas."""

import json
from pathlib import Path

import dask.array as da
from dask import delayed
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numcodecs import Blosc
from openghg.storage._encoding import get_zarr_encoding

from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions._serialization_codecs import (
    _TAGGED_JSON_VALUE_ENCODING,
    _decode_serialized_bool,
    _decode_tagged_json_value,
    _encode_tagged_json_value,
    _numpy_scalar_json_default,
)
from openghg_inversions.source_covariance import IndependentSourceCovariance
from openghg_inversions.serialization import decode_cf_multiindexes, encode_cf_multiindexes, save_datatree


def test_zarr_writer_preserves_format_two_compression_and_borrowed_encoding(tmp_path: Path) -> None:
    """Stored OpenGHG compression remains usable without changing borrowed metadata."""
    dataset = xr.Dataset({"value": ("cell", da.arange(4, chunks=2))})
    compressor = Blosc(cname="zstd", clevel=5)
    encoding = get_zarr_encoding(dataset.data_vars, compressor=compressor)["value"]
    dataset["value"].encoding = encoding.copy()
    tree = xr.DataTree(dataset)
    path = tmp_path / "compressed.zarr"

    save_datatree(tree, path)

    assert json.loads((path / ".zgroup").read_text())["zarr_format"] == 2
    with xr.open_datatree(path, engine="zarr") as stored:
        xr.testing.assert_equal(stored.to_dataset().compute(), dataset.compute())
    assert json.loads((path / "value" / ".zarray").read_text())["compressor"] == compressor.get_config()
    assert tree["value"].encoding == encoding


@pytest.mark.parametrize("storage_chunks", [None, 2, (2,)])
def test_zarr_writer_rechunks_shared_ragged_payloads_lazily(
    tmp_path: Path, storage_chunks: int | tuple[int, ...] | None
) -> None:
    """Normalize ragged data and coordinate chunks without executing shared work twice."""
    executions = []

    @delayed
    def payload() -> np.ndarray:
        executions.append("payload")
        return np.arange(5, dtype=float)

    data = da.from_delayed(payload(), shape=(5,), dtype=float).rechunk(((2, 3),))
    dataset = xr.Dataset({"value": ("cell", data)}, coords={"quality": ("cell", data + 1)})
    dataset["value"].attrs = {"units": "ppm"}
    dataset["value"].encoding = {"dtype": np.dtype("float64")}
    if storage_chunks is not None:
        dataset["value"].encoding["chunks"] = storage_chunks
    tree = xr.DataTree.from_dict({"/": xr.Dataset(attrs={"title": "chunked artifact"}), "inputs": dataset})
    path = tmp_path / "chunks.nc"
    save_datatree(tree, path, "zarr")
    assert executions == ["payload"]
    assert tree["inputs"]["value"].data is data
    assert tree["inputs"]["quality"].data.chunks == ((2, 3),)
    assert tree["inputs"]["value"].encoding == dataset["value"].encoding
    with xr.open_datatree(path.with_suffix(".zarr"), engine="zarr", chunks={}) as stored:
        assert stored["inputs"]["value"].data.chunks == (
            ((3, 2) if storage_chunks is None else (2, 2, 1)),
        )
        assert stored["inputs"]["quality"].data.chunks == ((3, 2),)
        xr.testing.assert_identical(stored["inputs"].to_dataset().compute(), dataset.compute())
        assert stored.attrs == tree.attrs


@pytest.mark.parametrize("suffix,chunks", [(".nc", None), (".zarr", None), (".zarr", (2, 0)), (".zarr", (2, 1)), (".zarr", 2), (".zarr", np.int64(2))])
def test_empty_lazy_payload_preserves_shape_graph_and_borrowed_encoding(tmp_path, monkeypatch, suffix, chunks):
    data = da.empty((3, 0), chunks=(2, 0), dtype=np.float64)
    dataset = xr.Dataset({"pattern": (("row", "term"), data)},
                         coords={"row": [0, 1, 2], "term": np.array([], dtype=np.int64)},
                         attrs={"meaning": "level-only boundary has no pattern coefficients"})
    dataset.pattern.attrs["units"] = "ppm"
    if chunks is not None:
        dataset.pattern.encoding["chunks"] = chunks
    tree = xr.DataTree(dataset)
    graph, original_chunks = data.__dask_graph__(), data.chunks
    expected_encoding = chunks if isinstance(chunks, (int, np.integer)) else (2, 1)
    encoding = dict(tree.pattern.encoding)
    if suffix == ".zarr":
        writer = xr.DataTree.to_zarr
        def check_boundary(stored, *args, **kwargs):
            assert stored.pattern.data is data
            assert stored.pattern.data.__dask_graph__() is graph
            assert stored.pattern.data.chunks == original_chunks
            assert stored.pattern.encoding["chunks"] == expected_encoding
            return writer(stored, *args, **kwargs)
        monkeypatch.setattr(xr.DataTree, "to_zarr", check_boundary)
    path = tmp_path / f"empty{suffix}"
    save_datatree(tree, path)
    assert tree.pattern.data is data
    assert data.__dask_graph__() is graph
    assert data.chunks == original_chunks
    assert tree.pattern.encoding == encoding
    with xr.open_datatree(path, engine="zarr" if suffix == ".zarr" else None) as restored:
        xr.testing.assert_identical(restored.to_dataset().compute(), dataset.compute())
    if suffix == ".zarr":
        assert json.loads((path / "pattern" / ".zarray").read_text())["shape"] == [3, 0]
        expected_chunks = [int(chunks), int(chunks)] if isinstance(chunks, (int, np.integer)) else [2, 1]
        assert json.loads((path / "pattern" / ".zarray").read_text())["chunks"] == expected_chunks


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_cf_mixed_level_labels_roundtrip_real_file(tmp_path: Path, suffix: str) -> None:
    """Real storage preserves integer/string identity alongside ordinary levels."""
    times = pd.date_range("2026-01-01", periods=4)
    expected_index = pd.MultiIndex.from_tuples(
        [("co2", 1, times[0]), ("co2", "1", times[1]), ("o2", 2, times[2]), ("o2", "2", times[3])],
        names=["species", "native_label", "time"],
    )
    original = xr.Dataset(
        {"sensitivity": ("native", da.arange(4, chunks=2))},
        coords=xr.Coordinates.from_pandas_multiindex(expected_index, "native"),
        attrs={"title": "mixed native labels"},
    )
    original["native_label"].attrs = {"long_name": "native cell identifier"}
    original["time"].attrs = {"standard_name": "time"}
    encoded = encode_cf_multiindexes(original, "native")
    assert encoded["native_label"].dtype.kind in "OU"
    assert all(isinstance(value, str) for value in encoded["native_label"].values)
    assert encoded["time"].dtype.kind == "M"
    assert encoded["species"].attrs == original["species"].attrs
    assert isinstance(encoded["sensitivity"].data, da.Array)
    destination = tmp_path / f"labels{suffix}"
    if suffix == ".nc":
        encoded.to_netcdf(destination)
        with xr.open_dataset(destination) as stored:
            loaded = stored.load()
    else:
        encoded.to_zarr(destination, mode="w")
        with xr.open_dataset(destination, engine="zarr") as stored:
            loaded = stored.load()
    actual = decode_cf_multiindexes(loaded, "native")
    pd.testing.assert_index_equal(actual.indexes["native"], expected_index)
    assert type(actual.indexes["native"][0][1]) is int
    assert type(actual.indexes["native"][1][1]) is str
    xr.testing.assert_identical(actual, original.compute())
    assert original["native_label"].attrs == {"long_name": "native cell identifier"}
    assert isinstance(original["sensitivity"].data, da.Array)


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


@pytest.mark.parametrize("suffix", ["nc", "zarr"])
def test_trace_multiindex_coordinate_attributes_survive_storage(tmp_path: Path, suffix: str) -> None:
    """Boundary ownership metadata survives persisted MultiIndex coordinates."""
    from openghg_inversions.serialization import load_trace, save_trace

    index = pd.MultiIndex.from_tuples([("ocean", "a"), ("land", "b")],
                                     names=["bc_source", "bc_region"])
    original = xr.Dataset(
        {"x": ("bc_state", [1., 2.])},
        coords=xr.Coordinates.from_pandas_multiindex(index, "bc_state"),
    )
    for name, local in [("bc_state", "state"),
                        ("bc_source", "source"), ("bc_region", "region")]:
        original[name].attrs = {"rhime_component": "bc", "rhime_local_name": local}
    before = original.copy(deep=True)
    path = tmp_path / f"owned-trace.{suffix}"
    save_trace(xr.DataTree.from_dict({"posterior": original}), path)
    restored = load_trace(path).posterior.to_dataset()
    xr.testing.assert_identical(restored, original)
    xr.testing.assert_identical(original, before)


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
@pytest.mark.parametrize("empty_domains", [False, True])
def test_cf_empty_gather_repeated_file_roundtrip(tmp_path, suffix, empty_domains):
    levels = [pd.Index([], dtype=str), pd.Index([], dtype=int)] if empty_domains else [
        pd.Index(["gpp", "ocean"]), pd.Index([0, 1]),
    ]
    index = pd.MultiIndex(levels=levels, codes=[[], []], names=["source", "region"])
    original = xr.Dataset({"h": (("row", "outer"), np.empty((2, 0)))},
                          coords=xr.Coordinates.from_pandas_multiindex(index, "outer"))
    original.outer.attrs["long_name"] = "selected outer state"
    original.source.attrs["long_name"] = "physical source"
    original.region.attrs["long_name"] = "native region"
    actual = original
    for generation in range(3):
        encoded = encode_cf_multiindexes(actual, "outer")
        path = tmp_path / f"empty-{generation}{suffix}"
        if suffix == ".nc":
            encoded.to_netcdf(path)
            actual = xr.load_dataset(path)
        else:
            encoded.to_zarr(path, mode="w")
            actual = xr.load_dataset(path, engine="zarr")
        actual = decode_cf_multiindexes(actual, "outer")
        xr.testing.assert_identical(actual, original)
        for expected_level, level in zip(index.levels, actual.indexes["outer"].levels, strict=True):
            np.testing.assert_array_equal(level, expected_level)
        assert actual.indexes["outer"].names == ["source", "region"]


def test_cf_nonempty_gather_rejects_empty_level_domain():
    encoded = xr.Dataset(coords={"state": ("state", [0], {"compress": "source region"}),
                                 "source": ("source", np.array([], dtype=str)), "region": [0]})
    with pytest.raises(ValueError, match="CF level coordinate 'source' cannot be empty"):
        decode_cf_multiindexes(encoded, "state")
