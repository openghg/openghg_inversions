"""Project a shared time-dependent basis after each site's observation selection."""

from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.basis._helpers import bc_sensitivity


def _inputs():
    times = pd.date_range("2020-01-01", periods=4, freq="h")
    grid = {"lat": [0.], "lon": [0.], "height": [10.]}
    site = xr.Dataset(coords={**grid, "time": times}, attrs={"boundary_representation": "centered-v1"})
    basis = xr.Dataset(coords={**grid, "time": times, "region": ["shared"]})
    for index, direction in enumerate("nesw", start=1):
        edge = "lon" if direction in "ns" else "lat"
        dims = (edge, "height", "time")
        site[f"bc_anomaly_{direction}"] = (dims, np.full((1, 1, 4), float(index)), {"units": "ppm"})
        site[f"bc_mass_{direction}"] = (dims, np.full((1, 1, 4), .1), {"units": "1"})
        basis[f"bc_basis_{direction}"] = (
            (edge, "height", "region", "time"), np.arange(1., 5.).reshape(1, 1, 1, 4),
        )
    site["bc_reference"] = xr.DataArray(400., attrs={"units": "ppm"})
    return site, basis


@pytest.mark.parametrize("lazy", [False, True])
def test_each_site_selects_and_orders_shared_basis_without_mutation_or_execution(lazy):
    site, basis = _inputs()
    sites = {"AAA": site.isel(time=[2, 0]), "BBB": site.isel(time=[3, 1])}
    basis = basis.isel(time=[1, 3, 0, 2])
    if lazy:
        sites = {name: data.chunk({"time": 1}) for name, data in sites.items()}
        basis = basis.chunk({"time": 2})
    original_basis = basis.copy(deep=True)
    original_sites = {name: data.copy(deep=True) for name, data in sites.items()}
    with Callback(pretask=lambda *args: pytest.fail("Projection executed a borrowed payload")):
        projected = bc_sensitivity(sites, basis)
    for name, weights in {"AAA": [3., 1.], "BBB": [4., 2.]}.items():
        result = projected[name]
        xr.testing.assert_identical(result.time, sites[name].time)
        np.testing.assert_allclose(result.H_bc_anomaly.transpose("bc_region", "time"), [np.array(weights) * 10.])
        np.testing.assert_allclose(result.H_bc_mass.transpose("bc_region", "time"), [np.array(weights) * .4])
        xr.testing.assert_identical(result.bc_reference, sites[name].bc_reference)
        xr.testing.assert_identical(sites[name], original_sites[name])
        if lazy:
            assert result.H_bc_anomaly.chunks is not None
            assert result.H_bc_mass.chunks is not None
    xr.testing.assert_identical(basis, original_basis)


@pytest.mark.parametrize("time_axis", ["absent", "singleton"])
def test_time_independent_basis_still_broadcasts_to_retained_site_times(time_axis):
    site, basis = _inputs()
    basis = basis.isel(time=slice(0, 1))
    if time_axis == "absent":
        basis = basis.squeeze("time", drop=True)
    selected = site.isel(time=[3, 1])
    result = bc_sensitivity({"AAA": selected}, basis)["AAA"]
    xr.testing.assert_identical(result.time, selected.time)
    np.testing.assert_allclose(result.H_bc_anomaly.transpose("bc_region", "time"), [[10., 10.]])
    np.testing.assert_allclose(result.H_bc_mass.transpose("bc_region", "time"), [[.4, .4]])


def test_missing_retained_timestamp_is_rejected_without_nearest_or_forward_fill():
    site, basis = _inputs()
    basis = basis.isel(time=[0, 2, 3])
    with pytest.raises(ValueError, match="cover every retained site timestamp"):
        bc_sensitivity({"AAA": site.isel(time=[1, 3])}, basis)


def test_temporal_selection_keeps_strict_spatial_alignment():
    site, basis = _inputs()
    basis = basis.assign_coords(lat=[1.])
    with pytest.raises(ValueError, match="align.*exact"):
        bc_sensitivity({"AAA": site.isel(time=[3, 1])}, basis)
