"""Tests for standalone trace DataTree migration and loading."""

from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.serialization import inferencedata_to_datatree, load_trace


def _trace() -> xr.DataTree:
    """Build a trace with root metadata and a serialisation-sensitive index."""
    state = pd.MultiIndex.from_tuples(
        [("energy", 1), ("transport", 2)],
        names=("sector", "region"),
    )
    state_coords = xr.Coordinates.from_pandas_multiindex(state, "state")
    posterior = xr.Dataset(
        {"x": (("chain", "draw", "state"), [[[1.0, 2.0]]])},
        coords={"chain": [0], "draw": [0], **state_coords},
        attrs={"created_at": "test"},
    )
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"title": "legacy trace"}),
            "posterior": posterior,
        }
    )


def test_load_trace_restores_legacy_group_layout(tmp_path: Path) -> None:
    """The loader preserves trace metadata and restores expanded MultiIndexes."""
    expected = _trace()
    path = tmp_path / "trace.nc"
    inferencedata_to_datatree(expected).to_netcdf(path, engine="netcdf4")

    actual = load_trace(path)

    assert actual.attrs == expected.attrs
    assert tuple(actual.children) == ("posterior",)
    assert actual["posterior"].attrs == expected["posterior"].attrs
    assert isinstance(actual["posterior"].indexes["state"], pd.MultiIndex)
    xr.testing.assert_identical(actual["posterior"].to_dataset(), expected["posterior"].to_dataset())


def test_load_trace_rejects_complete_inversion_output(tmp_path: Path) -> None:
    """Complete inversion artifacts direct callers to their owning loader."""
    path = tmp_path / "inversion-output.nc"
    artifact = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"schema": "openghg_inversions.InversionOutput"}),
            "trace/posterior": xr.Dataset({"x": ("draw", [1.0])}),
            "inv_inputs": xr.Dataset(),
            "basis_functions": xr.Dataset(),
        }
    )
    artifact.to_netcdf(path, engine="netcdf4")

    with pytest.raises(ValueError, match="InversionOutput.load"):
        load_trace(path)
