import pytest

from openghg_inversions.hbmcmc.hbmcmc import fixedbasisMCMC


def test_multi_sector_inversion(tmp_path):
    """This tests an inversion with three "sectors" labelled "a",
    "b", "c".

    Sectors "a" and "c" are sampled, and sector "b" is held constant.

    This test should run, and if you run pytest without capturing output,
    it should print traces for "a" and "c", but not for "b".
    """
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
        "mock_multi_sector": True,
    }
    fixedbasisMCMC(**mcmc_args)
