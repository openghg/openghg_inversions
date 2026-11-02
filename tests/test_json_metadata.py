"""Shared metadata semantics at stage and output JSON boundaries."""

from datetime import date, time, timedelta
import json
from pathlib import Path

import dask.array as da
from dask import delayed
import numpy as np
import pytest
import xarray as xr

from openghg_inversions._serialization_codecs import _json_value
from openghg_inversions.rhime._stage_artifacts import json_value, write_json
from openghg_inversions.rhime.outputs import _structured_metadata


@pytest.mark.parametrize("convert", [_json_value, json_value, _structured_metadata])
def test_json_metadata_preserves_types_and_temporal_meaning(convert):
    """Both artifact paths preserve dates even inside nanosecond NumPy arrays."""
    timestamp = np.datetime64("2024-02-03T04:05:06.123456789")
    duration = np.timedelta64(17, "ns")
    values = {
        "bool": np.bool_(True),
        "integer": np.int64(7),
        "float": np.float32(1.5),
        "nested": (None, np.array([[False, True]])),
        "date": timestamp,
        "dates": np.array([[timestamp]]),
        "scalar_date": np.array(timestamp),
        "duration": duration,
        "durations": np.array([duration]),
        "slice": slice(date(2024, 2, 3), None, 2),
        "path": Path("inputs.nc"),
        "time": time(4, 5),
        "delta": timedelta(seconds=3),
    }
    actual = json.loads(json.dumps(convert(values), allow_nan=False))
    assert actual == {
        "bool": True,
        "integer": 7,
        "float": 1.5,
        "nested": [None, [[False, True]]],
        "date": "2024-02-03T04:05:06.123456789",
        "dates": [["2024-02-03T04:05:06.123456789"]],
        "scalar_date": "2024-02-03T04:05:06.123456789",
        "duration": "17 nanoseconds",
        "durations": ["17 nanoseconds"],
        "slice": {"type": "slice", "start": "2024-02-03", "stop": None, "step": 2},
        "path": "inputs.nc",
        "time": "04:05:00",
        "delta": "0:00:03",
    }
    assert type(actual["bool"]) is bool
    assert type(actual["integer"]) is int
    assert type(actual["float"]) is float


@pytest.mark.parametrize("convert", [_json_value, json_value, _structured_metadata])
def test_json_metadata_rejects_unsupported_values_without_computation(convert):
    """Lazy arrays and arbitrary objects cannot silently become graph strings."""

    @delayed
    def forbidden():
        raise AssertionError("JSON metadata must not compute a bare Dask array")

    lazy = da.from_delayed(forbidden(), shape=(1,), dtype=float)
    for value in [lazy, object(), complex(1, 2), np.complex128(1 + 2j)]:
        with pytest.raises(TypeError, match="not JSON serializable"):
            convert({"nested": [value]})
    with pytest.raises(TypeError, match="not JSON serializable"):
        _json_value(xr.DataArray(lazy, dims="time"))


@pytest.mark.parametrize("convert", [json_value, _structured_metadata])
def test_json_metadata_dataarray_boundary_retains_dates_and_borrowed_storage(convert, tmp_path):
    """The explicit array boundary computes once and preserves coordinate dates."""
    executions = []

    @delayed
    def payload():
        executions.append(True)
        return np.array([3.0])

    lazy = da.from_delayed(payload(), shape=(1,), dtype=float)
    data = xr.DataArray(lazy, dims="time", coords={"time": np.array(["2024-02-03"], dtype="datetime64[ns]")})
    expected = {
        "dims": ["time"],
        "coords": {"time": ["2024-02-03T00:00:00.000000000"]},
        "values": [3.0],
    }
    assert convert(data) == expected
    assert executions == [True]
    assert data.data is lazy
    written = write_json(tmp_path / "metadata.json", {"array": data})
    assert json.loads(written.read_text()) == {"array": expected}
    assert data.data is lazy


def test_json_mapping_copy_keeps_strict_input_policy_and_independence():
    """Provenance copying accepts only the existing finite JSON input types."""
    from openghg_inversions._serialization_codecs import _copy_json_mapping

    source = {"values": [{"flag": True}]}
    copied = _copy_json_mapping(source, "provenance")
    copied["values"][0]["flag"] = False
    assert source["values"][0]["flag"] is True
    for value in [None, [], {"x": float("nan")}, {"x": np.int64(2)}, {"x": Path("file")}]:
        with pytest.raises(ValueError, match="provenance"):
            _copy_json_mapping(value, "provenance")


@pytest.mark.parametrize("raw", [None, {}, "[]", "null", "{", '{"x": NaN}', '{"x": Infinity}'])
def test_json_mapping_decoder_rejects_invalid_objects(raw):
    """Attribute decoding requires a finite JSON object string."""
    from openghg_inversions._serialization_codecs import _decode_json_mapping

    with pytest.raises(ValueError, match="metadata"):
        _decode_json_mapping(raw, "metadata")
    assert _decode_json_mapping('{"flag": false, "missing": null}', "metadata") == {
        "flag": False,
        "missing": None,
    }
