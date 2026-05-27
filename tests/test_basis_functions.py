import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.basis._functions import basis, _flux_fp_from_fp_all, _mean_fp_times_mean_flux
from openghg_inversions.basis import bucketbasisfunction, quadtreebasisfunction, fixed_outer_regions_basis
from openghg_inversions.basis._helpers import fp_sensitivity, bc_sensitivity
from openghg_inversions.inversion_data import data_processing_surface_notracer

from helpers import basis_function, footprint

def test_fp_x_flux(tac_ch4_data_args):
    fp_all, *_ = data_processing_surface_notracer(**tac_ch4_data_args)
    emissions_name = [next(iter(fp_all[".flux"].keys()))]

    flux1, fp1 = _flux_fp_from_fp_all(fp_all, emissions_name)
    mean_fp_flux1 = _mean_fp_times_mean_flux(flux1, fp1)

    # add new site with same footprint -- this should not change the mean over time
    fp_all["ABC"] = fp_all["TAC"]

    flux2, fp2 = _flux_fp_from_fp_all(fp_all, emissions_name)
    mean_fp_flux2 = _mean_fp_times_mean_flux(flux2, fp2)

    xr.testing.assert_allclose(mean_fp_flux1, mean_fp_flux2)

    # shift time of second site -- this should not change the mean over time
    max_time = pd.Timedelta(fp_all["TAC"].time.max().values - fp_all["TAC"].time.min().values)
    new_time = (fp_all["TAC"].time + max_time)
    fp_all["ABC"] = fp_all["ABC"].assign_coords(time=new_time)

    flux3, fp3 = _flux_fp_from_fp_all(fp_all, emissions_name)
    mean_fp_flux3 = _mean_fp_times_mean_flux(flux3, fp3)

    xr.testing.assert_allclose(mean_fp_flux1, mean_fp_flux3)



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


def test_fp_sensitivity_masks_outer_where_inner_has_coverage():
    time = pd.date_range("2019-01-01", periods=2)
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])

    outer_vals = np.ones((2, 2, 2), dtype=float)
    inner_vals = np.zeros((2, 2, 2), dtype=float)
    inner_vals[0, 0, :] = 2.0

    outer_fp = xr.DataArray(outer_vals, coords=[lat, lon, time], dims=["lat", "lon", "time"])
    inner_fp = xr.DataArray(inner_vals, coords=[lat, lon, time], dims=["lat", "lon", "time"])

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

    # Total outer contribution is 4 cells before masking. One cell overlaps
    # with inner and is forced to zero, so the expected outer sum is 3.
    np.testing.assert_allclose(h_outer.values, np.array([3.0, 3.0]))


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
