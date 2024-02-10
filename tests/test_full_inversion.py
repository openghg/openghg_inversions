import pytest

from openghg_inversions.hbmcmc.hbmcmc import fixedbasisMCMC


def test_full_inversion(tmp_path):
    mcmc_args = {
        "species": "ch4",
        "sites": ["TAC"],
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "bc_store": "inversions_tests",
        "obs_store": "inversions_tests",
        "footprint_store": "inversions_tests",
        "emissions_store": "inversions_tests",
        "inlet": ["185m"],
        "instrument": ["picarro"],
        "domain": "EUROPE",
        "fp_height": ["185m"],
        "fp_model": "NAME",
        "emissions_name": ["total-ukghg-edgar7"],
        "met_model": "ukv",
        "averaging_period": ["1H"],
        "outputname": "test_run",
        "outputpath": str(tmp_path),
        "basis_algorithm": "quadtree",
        "basis_output_path": str(tmp_path),
        "nbasis": 4,
        "nit": 1,
        "burn": 0,
        "tune": 0,
        "nchain": 1,
    }
    fixedbasisMCMC(**mcmc_args)


def test_full_inversion_min_error(tmp_path):
    mcmc_args = {
        "species": "ch4",
        "sites": ["TAC"],
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "bc_store": "inversions_tests",
        "obs_store": "inversions_tests",
        "footprint_store": "inversions_tests",
        "emissions_store": "inversions_tests",
        "inlet": ["185m"],
        "instrument": ["picarro"],
        "domain": "EUROPE",
        "fp_height": ["185m"],
        "fp_model": "NAME",
        "emissions_name": ["total-ukghg-edgar7"],
        "met_model": "ukv",
        "averaging_period": ["1H"],
        "outputname": "test_run",
        "outputpath": str(tmp_path),
        "basis_algorithm": "quadtree",
        "basis_output_path": str(tmp_path),
        "nbasis": 4,
        "nit": 4,
        "burn": 2,
        "tune": 0,
        "nchain": 1,
        "min_error": 20.0,
    }
    fixedbasisMCMC(**mcmc_args)


@pytest.mark.slow
def test_full_inversion_long(tmp_path):
    mcmc_args = {
        "species": "ch4",
        "sites": ["TAC"],
        "start_date": "2019-01-01",
        "end_date": "2019-01-02",
        "bc_store": "inversions_tests",
        "obs_store": "inversions_tests",
        "footprint_store": "inversions_tests",
        "emissions_store": "inversions_tests",
        "inlet": ["185m"],
        "instrument": ["picarro"],
        "domain": "EUROPE",
        "fp_height": ["185m"],
        "fp_model": "NAME",
        "emissions_name": ["total-ukghg-edgar7"],
        "met_model": "ukv",
        "averaging_period": ["1H"],
        "outputname": "test_run",
        "outputpath": str(tmp_path),
        "basis_algorithm": "quadtree",
        "basis_output_path": str(tmp_path),
        "nbasis": 50,
        "nit": 5000,
        "burn": 2000,
        "tune": 1000,
        "nchain": 4,
    }
    fixedbasisMCMC(**mcmc_args)
