"""One Pydantic flux record owns construction, schema and xarray attribute I/O."""

import json

import dask.array as da
from dask import delayed
from dask.callbacks import Callback
from jsonschema import Draft202012Validator
import numpy as np
from pydantic import ValidationError
import pytest
import xarray as xr

from openghg_inversions.flux_sanitization import (
    NONFINITE_METADATA_ATTR, FluxNonFiniteMetadata, sanitize_flux_nonfinite,
)


@pytest.mark.parametrize("format", ["netcdf", "zarr"])
def test_public_flux_writer_matches_schema_and_round_trips(tmp_path, format):
    original = xr.DataArray([1., np.nan], dims="cell", name="flux", attrs={"units": "mol/s"})
    saved = sanitize_flux_nonfinite(original, context="schema pilot", check="count")
    raw = json.loads(saved.attrs[NONFINITE_METADATA_ATTR])
    schema = FluxNonFiniteMetadata.model_json_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(raw)
    metadata = FluxNonFiniteMetadata.from_attrs(saved.attrs)
    assert metadata.count == 1 and metadata.total == 2 and metadata.fraction == .5
    assert FluxNonFiniteMetadata.from_json(metadata.to_json()) == metadata
    assert NONFINITE_METADATA_ATTR not in original.attrs
    if format == "netcdf":
        path = tmp_path / "flux.nc"
        saved.to_dataset().to_netcdf(path, engine="scipy")
        loaded = xr.open_dataset(path, engine="scipy")
    else:
        path = tmp_path / "flux.zarr"
        saved.to_dataset().to_zarr(path)
        loaded = xr.open_zarr(path)
    with loaded:
        assert FluxNonFiniteMetadata.from_attrs(loaded.flux.attrs) == metadata


@pytest.mark.parametrize("change,path", [
    ({"schema_version": True}, "schema_version"),
    ({"schema_version": 2}, "schema_version"),
    ({"count": "3"}, "count"),
    ({"count": -1}, "count"),
    ({"fraction": float("nan")}, "fraction"),
    ({"fraction": float("inf")}, "fraction"),
    ({"fraction": 2.}, "fraction"),
    ({"extra": "ignored before"}, "extra"),
])
def test_flux_metadata_boundary_reports_bad_fields(change, path):
    raw = FluxNonFiniteMetadata(schema_version=1).to_json_dict() | change
    with pytest.raises(ValidationError) as error:
        FluxNonFiniteMetadata.from_json(raw)
    assert error.value.errors()[0]["loc"] == (path,)


def test_absence_is_distinct_from_malformed_present_and_lazy_values_never_run():
    assert FluxNonFiniteMetadata.from_attrs({}) is None
    for malformed in (None, "{", "{}", [], 1):
        with pytest.raises(ValidationError):
            FluxNonFiniteMetadata.from_attrs({NONFINITE_METADATA_ATTR: malformed})

    @delayed
    def forbidden():
        raise AssertionError("Metadata validation computed an array")

    lazy = da.from_delayed(forbidden(), shape=(1,), dtype=float)
    executed = []
    with Callback(pretask=lambda key, *_: executed.append(key)):
        with pytest.raises(ValidationError):
            FluxNonFiniteMetadata.from_json({"schema_version": 1, "fraction": lazy})
    assert executed == []
