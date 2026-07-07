import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.basis._functions import basis, _flux_fp_from_fp_all, _mean_fp_times_mean_flux
from openghg_inversions.basis import bucketbasisfunction, quadtreebasisfunction, fixed_outer_regions_basis
from openghg_inversions.basis.algorithms import weighted_algorithm
from openghg_inversions.basis._wrapper import (
    _auto_distribute_nested_nbasis,
    basis_functions,
    basis_functions_wrapper,
)
from openghg_inversions.basis._helpers import apply_fp_basis_functions, fp_sensitivity, bc_sensitivity
from openghg_inversions.inversion_data import data_processing_surface_notracer

from helpers import basis_function, footprint

def test_fp_x_flux(tac_ch4_data_args):
    fp_all, *_ = data_processing_surface_notracer(**tac_ch4_data_args)
    emissions_name = [next(iter(fp_all[".flux"].keys()))]
    tac_ds = fp_all["TAC"]["standard"].ds if isinstance(fp_all["TAC"], xr.DataTree) else fp_all["TAC"]

    flux1, fp1 = _flux_fp_from_fp_all(fp_all, emissions_name)
    mean_fp_flux1 = _mean_fp_times_mean_flux(flux1, fp1)

    # add new site with same footprint -- this should not change the mean over time
    fp_all["ABC"] = fp_all["TAC"]

    flux2, fp2 = _flux_fp_from_fp_all(fp_all, emissions_name)
    mean_fp_flux2 = _mean_fp_times_mean_flux(flux2, fp2)

    xr.testing.assert_allclose(mean_fp_flux1, mean_fp_flux2)

    # shift time of second site -- this should not change the mean over time
    max_time = pd.Timedelta(tac_ds.time.max().values - tac_ds.time.min().values)
    new_time = tac_ds.time + max_time
    if isinstance(fp_all["ABC"], xr.DataTree):
        abc_ds = fp_all["ABC"]["standard"].ds.assign_coords(time=new_time)
        fp_all["ABC"] = xr.DataTree.from_dict({"/standard": abc_ds})
    else:
        fp_all["ABC"] = fp_all["ABC"].assign_coords(time=new_time)

    flux3, fp3 = _flux_fp_from_fp_all(fp_all, emissions_name)
    mean_fp_flux3 = _mean_fp_times_mean_flux(flux3, fp3)

    xr.testing.assert_allclose(mean_fp_flux1, mean_fp_flux3)


def test_inner_basis_input_prefers_raw_footprint():
    class FluxEntry:
        def __init__(self, flux):
            self.data = xr.Dataset({"flux": flux})

    time = pd.date_range("2019-01-01", periods=1)
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])

    flux = xr.DataArray(
        np.ones((1, 2, 2), dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        name="flux",
    )
    inner_fp = xr.DataArray(
        np.full((1, 2, 2), 2.0, dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        name="fp",
    )
    inner_fp_x_flux = xr.DataArray(
        np.full((1, 2, 2), 9.0, dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        name="fp_x_flux",
    )
    fp_all = {
        ".inner_flux": {"EDGAR": FluxEntry(flux)},
        "TAC": xr.DataTree.from_dict(
            {
                "/standard": xr.Dataset({"fp": inner_fp}),
                "/inner": xr.Dataset({"fp": inner_fp, "fp_x_flux": inner_fp_x_flux}),
            }
        ),
    }

    _, footprints = _flux_fp_from_fp_all(fp_all, ["EDGAR"], scenario="inner")

    xr.testing.assert_identical(footprints[0], inner_fp)


def test_standard_basis_input_prefers_raw_footprint():
    class FluxEntry:
        def __init__(self, flux):
            self.data = xr.Dataset({"flux": flux})

    time = pd.date_range("2019-01-01", periods=1)
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])

    flux = xr.DataArray(
        np.ones((1, 2, 2), dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        name="flux",
    )
    standard_fp = xr.DataArray(
        np.full((1, 2, 2), 3.0, dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        name="fp",
    )
    standard_fp_x_flux = xr.DataArray(
        np.full((1, 2, 2), 11.0, dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        name="fp_x_flux",
    )
    fp_all = {
        ".flux": {"EDGAR": FluxEntry(flux)},
        "TAC": xr.Dataset({"fp": standard_fp, "fp_x_flux": standard_fp_x_flux}),
    }

    _, footprints = _flux_fp_from_fp_all(fp_all, ["EDGAR"])

    xr.testing.assert_identical(footprints[0], standard_fp)



def test_quadtree_basis_function(tac_ch4_data_args, raw_data_path):
    """Check if quadtree basis created with seed 42 and TAC CH4 args matches
    a basis created with the same arguments and saved to file.

    This is to check against changes in the code from when this test was made
    (13 Feb 2024)
    """
    fp_all, *_ = data_processing_surface_notracer(**tac_ch4_data_args)
    emissions_name = next(iter(fp_all[".flux"].keys()))
    basis_func = quadtreebasisfunction(
        emissions_name=[emissions_name],
        fp_all=fp_all,
        start_date="2019-01-01",
        seed=42,
        domain="EUROPE"
    )

    basis_func_reloaded = basis(
        domain="EUROPE", basis_case="quadtree_ch4-test_basis", basis_directory=raw_data_path / "basis"
    )

    # TODO: create new "fixed" basis function file, since we've switched basis functions from
    # dataset to data array
    xr.testing.assert_allclose(basis_func, basis_func_reloaded.basis)


def test_bucket_basis_function(tac_ch4_data_args, raw_data_path):
    """Check if quadtree basis created with seed 42 and TAC CH4 args matches
    a basis created with the same arguments and saved to file.

    This is to check against changes in the code from when this test was made
    (13 Feb 2024)
    """
    fp_all, *_ = data_processing_surface_notracer(**tac_ch4_data_args)
    emissions_name = next(iter(fp_all[".flux"].keys()))
    basis_func = bucketbasisfunction(
        emissions_name=[emissions_name],
        fp_all=fp_all,
        start_date="2019-01-01",
        domain="EUROPE",
        nbasis=98,
    )


    basis_func_reloaded = basis(
        domain="EUROPE", basis_case="bucket_ch4-test_basis", basis_directory=raw_data_path / "basis"
    )

    # TODO: create new "fixed" basis function file, since we've switched basis functions from
    # dataset to data array
    xr.testing.assert_allclose(basis_func, basis_func_reloaded.basis)

def test_fixed_outer_region_basis_function(tac_ch4_data_args, raw_data_path):
    """Check if fixed outer region basis created wtih seed 42 and TAC CH4 args matches 
    a basis created with the same argumenst and saved to file.
    
    This is to check against changes in the code from when this test was made 
    (2 Sep 2024)
    """
    fp_all, *_ = data_processing_surface_notracer(**tac_ch4_data_args)
    emissions_name = next(iter(fp_all[".flux"].keys()))
    basis_func = fixed_outer_regions_basis(
        emissions_name=[emissions_name],
        fp_all=fp_all,
        start_date="2019-01-01",
        domain="EUROPE",
        basis_algorithm='weighted'
    )

    basis_func_reloaded = basis(
        domain="EUROPE", basis_case="fixed_outer_region_ch4-test_basis", basis_directory=raw_data_path / "basis"
    )

    # TODO: create new "fixed" basis function file, since we've switched basis functions from
    # dataset to data array
    xr.testing.assert_allclose(basis_func, basis_func_reloaded.basis)


def test_inner_domain_fixed_outer_region_uses_explicit_definition_file(monkeypatch):
    fixed_calls = []
    algorithm_calls = []

    def fake_fixed_outer_regions_basis(
        fp_all,
        start_date,
        basis_algorithm,
        domain,
        emissions_name=None,
        nbasis=100,
        country_directory=None,
        abs_flux=False,
        scenario="standard",
        region_definition_file=None,
    ):
        fixed_calls.append(
            {
                "domain": domain,
                "nbasis": nbasis,
                "scenario": scenario,
                "region_definition_file": region_definition_file,
            }
        )
        lat = np.array([0.0])
        lon = np.array([0.0])
        return xr.DataArray(
            [[[1]]],
            dims=["time", "lat", "lon"],
            coords={"time": [pd.Timestamp(start_date)], "lat": lat, "lon": lon},
            name="basis",
        )

    def fake_inner_algorithm(
        fp_all,
        start_date,
        domain,
        emissions_name=None,
        nbasis=100,
        country_directory=None,
        scenario="standard",
    ):
        algorithm_calls.append(
            {
                "domain": domain,
                "nbasis": nbasis,
                "scenario": scenario,
            }
        )
        lat = np.array([0.0])
        lon = np.array([0.0])
        return xr.DataArray(
            [[[1]]],
            dims=["time", "lat", "lon"],
            coords={"time": [pd.Timestamp(start_date)], "lat": lat, "lon": lon},
            name="basis",
        )

    def fake_fp_sensitivity(fp_all, basis_func, inner_basis_func=None):
        return {
            ".basis": basis_func,
            ".basis_inner": inner_basis_func,
        }

    monkeypatch.setattr(
        "openghg_inversions.basis._wrapper.fixed_outer_regions_basis",
        fake_fixed_outer_regions_basis,
    )
    monkeypatch.setitem(
        basis_functions,
        "quadtree",
        basis_functions["quadtree"]._replace(algorithm=fake_inner_algorithm),
    )
    monkeypatch.setattr("openghg_inversions.basis._wrapper.fp_sensitivity", fake_fp_sensitivity)

    fp_all = {"TAC": xr.DataTree.from_dict({"/standard": xr.Dataset(), "/inner": xr.Dataset()})}

    result = basis_functions_wrapper(
        fp_all=fp_all,
        species="ch4",
        domain="EUROPE",
        start_date="2019-01-01",
        emissions_name=["edgar"],
        nbasis=4,
        inner_nbasis=2,
        use_bc=False,
        basis_algorithm="quadtree",
        fix_outer_regions=True,
        outer_region_definition_file="intem_region_definition_EUHROB.nc",
        inner_domain="6km",
    )

    assert result[".basis_inner"] is not None
    assert algorithm_calls == [
        {
            "domain": "EUROPE-6km",
            "nbasis": 2,
            "scenario": "inner",
        }
    ]
    assert fixed_calls == [
        {
            "domain": "EUROPE",
            "nbasis": 4,
            "scenario": "standard",
            "region_definition_file": "intem_region_definition_EUHROB.nc",
        },
    ]


def test_fp_sensitivity_one_flux():
    """Test fp_sensitivity with one flux sector."""
    nlat, nlon = 10, 12
    nbasis = 3
    basis_func = basis_function(nlat, nlon, nbasis)
    fp = footprint(nlat, nlon, "2019-01-01", "2019-01-02", 2)

    fp_and_data = {"TAC": xr.Dataset({"fp_x_flux": fp}), ".flux": {"a": 1}}

    fp_and_data = fp_sensitivity(fp_and_data, basis_func)

    h = fp_and_data["TAC"].H

    # the footprint values at time 0 are 1, and at time 1 are 2
    np.testing.assert_allclose(2 * h.isel(time=0), h.isel(time=1))


def test_auto_distribute_nested_nbasis_uses_inner_outer_sensitivity():
    time = pd.date_range("2019-01-01", periods=2)
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])

    outer_fp = xr.DataArray(
        np.ones((2, 2, 2), dtype=float),
        coords=[lat, lon, time],
        dims=["lat", "lon", "time"],
    )
    inner_fp = xr.DataArray(
        np.ones((2, 2, 2), dtype=float) * 3.0,
        coords=[lat, lon, time],
        dims=["lat", "lon", "time"],
    )
    fp_all = {
        "TAC": xr.DataTree.from_dict(
            {
                "/standard": xr.Dataset({"fp_x_flux": outer_fp}),
                "/inner": xr.Dataset({"fp_x_flux": inner_fp}),
            }
        )
    }

    outer_nbasis, inner_nbasis = _auto_distribute_nested_nbasis(fp_all, total_nbasis=100)

    assert outer_nbasis == 40
    assert inner_nbasis == 60


def test_fp_sensitivity_two_flux_sectors():
    """Check that we can apply a common basis function to two separate sources."""
    nlat, nlon = 10, 12
    nbasis = 3
    basis_func = basis_function(nlat, nlon, nbasis)

    fp1 = footprint(nlat, nlon, "2019-01-01", "2019-01-02", 2)
    fp2 = footprint(nlat, nlon, "2019-01-01", "2019-01-02", 2)
    fp = xr.concat([fp1.expand_dims({"source": ["a"]}), fp2.expand_dims({"source": ["b"]})], dim="source")
    fp_and_data = {"TAC": xr.Dataset({"fp_x_flux_sectoral": fp}), ".flux": {"a": 1, "b": 2}}

    fp_and_data = fp_sensitivity(fp_and_data, basis_func)

    for source in ["a", "b"]:
        h = fp_and_data["TAC"].H.sel(source=source).dropna("region")

        # the footprint values at time 0 are 1, and at time 1 are 2
        np.testing.assert_allclose(2 * h.isel(time=0), h.isel(time=1))


def test_fp_sensitivity_two_flux_sectors_two_basis_funcs():
    """Check that we can apply separate basis functions to separate sources."""
    nlat, nlon = 10, 12
    nbasis1 = 3
    nbasis2 = 4
    basis_func1 = basis_function(nlat, nlon, nbasis1)
    basis_func2 = basis_function(nlat, nlon, nbasis2)
    basis_func = {"a": basis_func1, "b": basis_func2}

    fp1 = footprint(nlat, nlon, "2019-01-01", "2019-01-02", 2)
    fp2 = footprint(nlat, nlon, "2019-01-01", "2019-01-02", 2)
    fp = xr.concat([fp1.expand_dims({"source": ["a"]}), fp2.expand_dims({"source": ["b"]})], dim="source")
    fp_and_data = {"TAC": xr.Dataset({"fp_x_flux_sectoral": fp}), ".flux": {"a": 1, "b": 2}}

    fp_and_data = fp_sensitivity(fp_and_data, basis_func)

    for source in ["a", "b"]:
        h = fp_and_data["TAC"].H.sel(source=source).dropna("region")

        # the footprint values at time 0 are 1, and at time 1 are 2
        np.testing.assert_allclose(2 * h.isel(time=0), h.isel(time=1))


def test_fp_sensitivity_inner_requires_datatree_entry():
    nlat, nlon = 4, 4
    basis_func = basis_function(nlat, nlon, 2)
    inner_basis_func = basis_function(nlat, nlon, 2)
    fp = footprint(nlat, nlon, "2019-01-01", "2019-01-02", 2)

    fp_and_data = {"TAC": xr.Dataset({"fp_x_flux": fp}), ".flux": {"a": 1}}

    with np.testing.assert_raises(ValueError):
        fp_sensitivity(fp_and_data, basis_func, inner_basis_func=inner_basis_func)


def test_fp_sensitivity_masks_outer_where_inner_has_extent():
    time = pd.date_range("2019-01-01", periods=2)
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])

    outer_vals = np.ones((2, 2, 2), dtype=float)
    inner_vals = np.zeros((2, 2, 2), dtype=float)
    inner_vals[0, 0, :] = 2.0

    outer_fp = xr.DataArray(outer_vals, coords=[lat, lon, time], dims=["lat", "lon", "time"])
    inner_fp = xr.DataArray(inner_vals, coords=[lat, lon, time], dims=["lat", "lon", "time"], name="fp_x_flux")

    basis = xr.DataArray(np.ones((2, 2), dtype=int), coords=[lat, lon], dims=["lat", "lon"])
    inner_basis = xr.DataArray(np.ones((2, 2), dtype=int), coords=[lat, lon], dims=["lat", "lon"])

    fp_and_data = {
        "TAC": xr.DataTree.from_dict(
            {
                "/standard": xr.Dataset({"fp_x_flux": outer_fp}),
                "/inner": xr.Dataset({"fp_x_flux": inner_fp}),
            }
        ),
        ".flux": {"a": 1},
    }

    result = fp_sensitivity(fp_and_data, basis, inner_basis_func=inner_basis)

    h_outer = result["TAC"]["standard"].ds["H"].squeeze("region")
    inner_after = result["TAC"]["inner"].ds["fp_x_flux"]
    h_inner = result["TAC"]["inner"].ds["H_inner"].squeeze("region")

    # The full inner-domain extent is masked out of the outer contribution,
    # even when only part of the inner footprint has non-zero sensitivity.
    np.testing.assert_allclose(h_outer.values, np.array([0.0, 0.0]))
    xr.testing.assert_identical(inner_after, inner_fp)
    np.testing.assert_allclose(h_inner.values, np.array([2.0, 2.0]))


def test_fp_sensitivity_rejects_active_cells_outside_basis():
    time = pd.date_range("2019-01-01", periods=2)
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])

    fp = xr.DataArray(np.ones((2, 2, 2)), coords=[lat, lon, time], dims=["lat", "lon", "time"])
    basis = xr.DataArray([[0, 1], [1, 1]], coords=[lat, lon], dims=["lat", "lon"])
    fp_and_data = {"TAC": xr.Dataset({"fp_x_flux": fp}), ".flux": {"a": 1}}

    with pytest.raises(ValueError, match="basis has value 0 in cells with non-zero footprint sensitivity"):
        fp_sensitivity(fp_and_data, basis)


def test_weighted_basis_rejects_missing_inner_landsea_file():
    grid = np.ones((2, 2))

    with pytest.raises(FileNotFoundError, match="No default land-sea file found for domain EUROPE-6km"):
        weighted_algorithm(grid=grid, nregion=2, domain="EUROPE-6km", country_directory=None)


def test_apply_fp_basis_functions_rejects_missing_time_slice():
    time = pd.date_range("2019-01-01", periods=2)
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])
    fp_values = np.ones((2, 2, 2), dtype=float)
    fp_values[:, :, 1] = np.nan

    fp = xr.DataArray(fp_values, coords=[lat, lon, time], dims=["lat", "lon", "time"])
    basis = xr.DataArray(np.ones((2, 2), dtype=int), coords=[lat, lon], dims=["lat", "lon"])

    with pytest.raises(ValueError, match="Refusing to convert missing footprint data to zero sensitivity"):
        apply_fp_basis_functions(fp_x_flux=fp, basis_func=basis)


def test_fp_sensitivity_preserves_empty_root_standard_child():
    nlat, nlon = 4, 4
    basis_func = basis_function(nlat, nlon, 2)
    fp = footprint(nlat, nlon, "2019-01-01", "2019-01-02", 2)

    fp_and_data = {
        "TAC": xr.DataTree.from_dict({"/standard": xr.Dataset({"fp_x_flux": fp})}),
        ".flux": {"a": 1},
    }

    result = fp_sensitivity(fp_and_data, basis_func)

    assert isinstance(result["TAC"], xr.DataTree)
    assert "standard" in result["TAC"].children
    assert len(result["TAC"].ds.data_vars) == 0
    assert "H" in result["TAC"]["standard"].ds


def test_bc_sensitivity_writes_hbc_to_standard_child():
    time = pd.date_range("2019-01-01", periods=2)
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])
    height = np.array([100.0])

    base = xr.Dataset(
        {
            "bc_n": xr.DataArray(np.ones((2, 2, 1, 2)), coords=[lat, lon, height, time], dims=["lat", "lon", "height", "time"]),
            "bc_e": xr.DataArray(np.ones((2, 2, 1, 2)), coords=[lat, lon, height, time], dims=["lat", "lon", "height", "time"]),
            "bc_s": xr.DataArray(np.ones((2, 2, 1, 2)), coords=[lat, lon, height, time], dims=["lat", "lon", "height", "time"]),
            "bc_w": xr.DataArray(np.ones((2, 2, 1, 2)), coords=[lat, lon, height, time], dims=["lat", "lon", "height", "time"]),
        }
    )

    fp_and_data = {
        "TAC": xr.DataTree.from_dict({"/standard": base}),
        ".flux": {"a": 1},
    }

    result = bc_sensitivity(fp_and_data, domain="EUROPE", basis_case="NESW")

    assert isinstance(result["TAC"], xr.DataTree)
    assert "H_bc" in result["TAC"]["standard"].ds
    assert "H_bc" not in result["TAC"].ds
