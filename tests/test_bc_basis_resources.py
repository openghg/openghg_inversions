"""Explicit BC resources and resource-free sensitivity construction."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.basis.support import load_bc_basis


def _contributions():
    coords = {"time": pd.date_range("2020-01-01", periods=2), "lat": [50.0], "lon": [0.0], "height": [10.0]}
    return xr.Dataset({
        f"bc_{direction}": xr.DataArray(
            da.full((2, 1, 1, 1), value, chunks=(1, 1, 1, 1)),
            dims=("time", "lat", "lon", "height"), coords=coords,
        ) for value, direction in enumerate("nesw", start=1)
    })


def _basis(times, weights):
    return xr.Dataset({
        f"basis_bc_{direction}": xr.DataArray(
            np.array(weights).reshape(len(times), 1, 1, 1, 1),
            dims=("time", "region", "lat", "lon", "height"),
            coords={"time": times, "region": ["all"], "lat": [50.0], "lon": [0.0], "height": [10.0]},
        ) for direction in "nesw"
    })


def test_nesw_uses_acquired_contributions_without_io_or_mutation(monkeypatch):
    def no_read(*args, **kwargs):
        pytest.fail("NESW must not read a resource")

    monkeypatch.setattr(xr, "open_dataset", no_read)
    site = _contributions()
    inputs = {"SITE": site, ".metadata": object()}
    result = bc_sensitivity(inputs)
    assert result is not inputs
    assert result["SITE"] is not site
    assert result[".metadata"] is inputs[".metadata"]
    assert "H_bc" not in site
    assert isinstance(result["SITE"].H_bc.data, da.Array)
    np.testing.assert_array_equal(result["SITE"].H_bc.compute(), [[1, 1], [2, 2], [3, 3], [4, 4]])


def test_explicit_multifile_basis_preserves_time_alignment_and_grid_tolerance(tmp_path):
    times = pd.date_range("2020-01-01", periods=2)
    # Filenames deliberately reverse chronological order; scientific alignment
    # must follow time labels, not file order.
    late = tmp_path / "a.nc"
    early = tmp_path / "z.nc"
    _basis(times[1:], [3]).to_netcdf(late)
    _basis(times[:1], [2]).assign_coords(lat=[50.0 + 1e-6]).to_netcdf(early)
    basis = load_bc_basis([early, late])
    snapshot = basis.copy(deep=True)
    result = bc_sensitivity({"SITE": _contributions()}, bc_basis=basis)
    np.testing.assert_array_equal(result["SITE"].H_bc.compute(), [[20], [30]])
    xr.testing.assert_identical(basis, snapshot)


def test_single_basis_time_applies_to_all_observation_times(tmp_path):
    path = tmp_path / "basis.nc"
    _basis(pd.date_range("2010-01-01", periods=1), [2]).to_netcdf(path)
    basis = load_bc_basis([path])
    result = bc_sensitivity({"SITE": _contributions()}, bc_basis=basis)
    np.testing.assert_array_equal(result["SITE"].H_bc.compute(), [[20], [20]])


@pytest.mark.parametrize("invalid", ["missing.nc", "unreadable.nc"])
def test_selected_invalid_bc_file_is_not_silently_dropped(tmp_path, invalid):
    good = tmp_path / "good.nc"
    _basis(pd.date_range("2020-01-01", periods=1), [1]).to_netcdf(good)
    bad = tmp_path / invalid
    if invalid == "unreadable.nc":
        bad.write_text("not a NetCDF file")
    with pytest.raises((OSError, ValueError)):
        load_bc_basis([good, bad])


def test_empty_explicit_bc_selection_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        load_bc_basis([])
