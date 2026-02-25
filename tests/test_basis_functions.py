import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.basis._functions import (
    basis,
    basis_functions,
    _flux_fp_from_fp_all,
    _mean_fp_times_mean_flux,
)
from openghg_inversions.basis import (
    basis_functions_wrapper,
    bucketbasisfunction,
    quadtreebasisfunction,
    fixed_outer_regions_basis,
)
from openghg_inversions.basis.basis_functions import BasisFunctions, MultiSectorBasisFunctions
from openghg_inversions.basis._helpers import apply_fp_basis_functions, fp_sensitivity
from openghg_inversions.inversion_data import data_processing_surface_notracer

from helpers import basis_function, footprint
from helpers import (
    convert_old_multisector_H_to_gathered,
    make_basis_flat_from_blocks,
    make_fp_x_flux,
    make_fp_x_flux_sectoral,
    expected_H_from_basis_sum,
    old_apply_fp_basis_functions_like,
)


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
    new_time = fp_all["TAC"].time + max_time
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
        emissions_name=[emissions_name], fp_all=fp_all, start_date="2019-01-01", seed=42, domain="EUROPE"
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
        basis_algorithm="weighted",
    )

    basis_func_reloaded = basis(
        domain="EUROPE",
        basis_case="fixed_outer_region_ch4-test_basis",
        basis_directory=raw_data_path / "basis",
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


def test_basisfunctions_sensitivity_synthetic_matches_explicit_sum():
    # 2x2 grid, 3 times, basis has 2 regions: top row region 1, bottom row region 2
    basis = make_basis_flat_from_blocks([[1, 1], [2, 2]])

    fp_x_flux = make_fp_x_flux(nlat=2, nlon=2, ntime=3)

    # flux is only used for interpolation_matrix etc; sensitivity uses basis_matrix only
    # but BasisFunctions requires flux; use ones
    flux = xr.ones_like(basis, dtype=float)

    bf = BasisFunctions(basis_flat=basis, flux=flux)

    H_new = bf.sensitivity(fp_x_flux)

    H_expected = expected_H_from_basis_sum(fp_x_flux, basis)

    xr.testing.assert_allclose(H_new, H_expected)


def test_basisfunctions_sensitivity_synthetic_multisector_shared_basis():
    sources = ["A", "B"]
    basis = make_basis_flat_from_blocks([[1, 1], [2, 2]])

    fp_x_flux_sectoral = make_fp_x_flux_sectoral(sources=sources, nlat=2, nlon=2, ntime=3)

    flux = xr.ones_like(basis, dtype=float)
    bf = BasisFunctions(basis_flat=basis, flux=flux)

    H = bf.sensitivity(fp_x_flux_sectoral)

    # Expected: same as single-sector, but computed per source and carried through source dim.
    # bf.sensitivity preserves non-(lat,lon) dims, so expect (source, time, region) or (source, region, time)
    # We enforce canonical order via transpose below.
    H = H.transpose("region", "time", "source")

    # Construct expected by explicit summation for each source
    expected_pieces = []
    for s in sources:
        Hs = expected_H_from_basis_sum(fp_x_flux_sectoral.sel(source=s), basis)  # (region, time)
        expected_pieces.append(Hs.expand_dims(source=[s]))
    H_expected = xr.concat(expected_pieces, dim="source").transpose("region", "time", "source")

    xr.testing.assert_allclose(H, H_expected)


def test_basisfunctions_sensitivity_regression_matches_legacy_like():
    basis = make_basis_flat_from_blocks([[1, 1], [2, 2]])
    fp_x_flux = make_fp_x_flux(nlat=2, nlon=2, ntime=4)

    flux = xr.ones_like(basis, dtype=float)
    bf = BasisFunctions(basis_flat=basis, flux=flux)

    H_new = bf.sensitivity(fp_x_flux)  # (region, time)
    H_old = old_apply_fp_basis_functions_like(fp_x_flux, basis)  # (region, time)

    xr.testing.assert_allclose(H_new, H_old)


def test_synthetic_no_all_zero_state_rows_when_fp_positive_everywhere():
    basis = make_basis_flat_from_blocks([[1, 1], [2, 2]])
    fp_x_flux = make_fp_x_flux(nlat=2, nlon=2, ntime=3, values=np.ones((2, 2, 3)))

    flux = xr.ones_like(basis, dtype=float)
    bf = BasisFunctions(basis_flat=basis, flux=flux)

    H = bf.sensitivity(fp_x_flux)  # (region, time)

    # For strictly positive fp_x_flux and nonempty basis regions, each region row should be > 0 somewhere
    assert (H > 0).any(dim="time").all().item()


# @pytest.mark.slow
def test_basisfunctions_sensitivity_matches_apply_fp_basis_functions_real_data():
    data_args = {
        "species": "ch4",
        "sites": ["MHD", "TAC"],
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "bc_store": "inversions_tests",
        "obs_store": "inversions_tests",
        "footprint_store": "inversions_tests",
        "emissions_store": "inversions_tests",
        "inlet": ["10m", "185m"],
        "instrument": ["gcmd", "picarro"],
        "domain": "EUROPE",
        "fp_height": ["10m", "185m"],
        "fp_model": "NAME",
        "emissions_name": ["total-ukghg-edgar7"],
        "averaging_period": ["1h", "1h"],
    }

    fp_all, *_ = data_processing_surface_notracer(**data_args)

    basis_args = {
        "species": "ch4",
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "emissions_name": ["total-ukghg-edgar7"],
        "nbasis": 20,
        "use_bc": True,
        "basis_algorithm": "weighted",
        "bc_basis_case": "NESW",
    }

    fp_all_with_basis = basis_functions_wrapper(fp_all, **basis_args)

    site = "MHD"
    ds = fp_all_with_basis[site]

    # old sensitivity (already computed by wrapper) is ds["H"]; but we want to call the legacy fn directly too
    H_old = apply_fp_basis_functions(ds.fp_x_flux, fp_all_with_basis[".basis"])

    # new sensitivity
    # Need a flux field; the wrapper has fp_all_with_basis[".flux"][source].data.flux
    flux_source = next(iter(fp_all_with_basis[".flux"].keys()))
    flux = fp_all_with_basis[".flux"][flux_source].data.flux

    bf = BasisFunctions(basis_flat=fp_all_with_basis[".basis"], flux=flux)
    H_new = bf.sensitivity(ds.fp_x_flux)

    # Ensure same dim order for comparison
    if H_new.dims != H_old.dims:
        H_new = H_new.transpose(*H_old.dims)

    xr.testing.assert_allclose(H_new, H_old)

    # now test vs. result of basis_functions_wrapper
    H_old = ds.H

    if H_new.dims != H_old.dims:
        H_new = H_new.transpose(*H_old.dims)

    xr.testing.assert_allclose(H_new, H_old)


# @pytest.mark.slow
def test_multisector_ragged_new_matches_old_after_conversion():
    # --- Load test-suite data (same as notebook) ---
    data_args = {
        "species": "ch4",
        "sites": ["MHD", "TAC"],
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "bc_store": "inversions_tests",
        "obs_store": "inversions_tests",
        "footprint_store": "inversions_tests",
        "emissions_store": "inversions_tests",
        "inlet": ["10m", "185m"],
        "instrument": ["gcmd", "picarro"],
        "domain": "EUROPE",
        "fp_height": ["10m", "185m"],
        "fp_model": "NAME",
        "emissions_name": ["total-ukghg-edgar7"],
        "averaging_period": ["1h", "1h"],
    }
    fp_all, *_ = data_processing_surface_notracer(**data_args)

    # --- Make a "sectoral" fp_x_flux like your notebook did ---
    fp_all_sectoral = fp_all.copy()
    fp_all_sectoral[".flux"] = fp_all[".flux"].copy()
    fp_all_sectoral[".flux"]["sector2"] = fp_all_sectoral[".flux"]["total-ukghg-edgar7"]

    for k, v in fp_all_sectoral.items():
        if str(k).startswith("."):
            continue
        ds = v["fp_x_flux"]
        to_concat = [
            ds.expand_dims({"source": ["total-ukghg-edgar7"]}),
            ds.expand_dims({"source": ["sector2"]}),
        ]
        v["fp_x_flux_sectoral"] = xr.concat(to_concat, dim="source")

    # --- Build two different basis partitions (ragged region counts) ---
    weighted_basis_args_1 = {
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "emissions_name": ["total-ukghg-edgar7"],
        "nbasis": 20,
    }
    weighted_basis_args_2 = {
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "emissions_name": ["sector2"],
        "nbasis": 30,
    }

    basis1 = basis_functions["weighted"].algorithm(fp_all_sectoral, **weighted_basis_args_1)
    basis2 = basis_functions["weighted"].algorithm(fp_all_sectoral, **weighted_basis_args_2)

    basis_dict = {"total-ukghg-edgar7": basis1, "sector2": basis2}

    # --- Old behaviour: fp_sensitivity pads to max(region) and introduces zero rows for missing regions ---
    fp_old = fp_sensitivity(fp_all_sectoral.copy(), basis_func=basis_dict)
    site = "MHD"
    H_old = fp_old[site]["H"]  # (region=max, time, source)

    # Convert old padded to gathered multiindex region and drop all-zero rows
    H_old_gathered = convert_old_multisector_H_to_gathered(H_old)

    # --- New behaviour
    flux_dict = {k: v.data.flux for k, v in fp_all_sectoral[".flux"].items()}
    multisector_bf = MultiSectorBasisFunctions(basis_flat=basis_dict, flux=flux_dict)

    H_new_gathered = multisector_bf.sensitivity(fp_all_sectoral[site].fp_x_flux_sectoral)
    xr.testing.assert_allclose(H_new_gathered, H_old_gathered)
