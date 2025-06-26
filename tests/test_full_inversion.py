import numpy as np
import pytest

from openghg_inversions.hbmcmc.hbmcmc import fixedbasisMCMC
from tests.conftest import raw_data_path


@pytest.fixture
def mcmc_args(tmp_path, tac_ch4_data_args, merged_data_dir, merged_data_file_name):
    mcmc_args = tac_ch4_data_args.copy()
    mcmc_args.update(
        {
            "outputname": "test_run",
            "outputpath": str(tmp_path),
            "basis_algorithm": "quadtree",
            "basis_output_path": str(tmp_path),
            "nbasis": 4,
            "nit": 1,
            "burn": 0,
            "tune": 0,
            "nchain": 1,
            "reload_merged_data": True,
            "merged_data_dir": merged_data_dir,
            "merged_data_name": merged_data_file_name,
        }
    )
    return mcmc_args

@pytest.fixture
def satellite_mcmc_args(tmp_path,satellite_ch4_data_args, southamerica_country_file, merged_data_dir,raw_data_path):
    mcmc_args = satellite_ch4_data_args.copy()
    mcmc_args.update(
        {
            "outputname": "satellite_test_run",
            "outputpath": str(tmp_path),
            "basis_algorithm": "quadtree",
            "basis_output_path": str(tmp_path),
            "nbasis": 4,
            "nit": 1,
            "burn": 0,
            "tune": 0,
            "nchain": 1,
            "reload_merged_data": True,
            "merged_data_dir": merged_data_dir,
            "xprior"   : {"pdf" : "normal", "mu" : 1.0, "sigma" : 1.0},
            "bcprior"  : {"pdf" : "normal", "mu" : 1.0, "sigma" : 1.0},
            "sigprior" : {"pdf" : "uniform", "lower" : 0.1, "upper" : 10.0},
            "bc_freq" : "monthly",
            "sigma_freq" : '5D',
            "sigma_per_site" : True,
            "averaging_error" :False,
            "min_error" :0.0,
            "fix_basis_outer_regions" :False,
            "use_bc" :True   ,                 
            "nuts_sampler" :"numpyro",
            "save_trace" :True,
            "min_error_options" :{"by_site": True},
            "pollution_events_from_obs" :True,
            "no_model_error" :False,
            "reparameterise_log_normal" :False,
            "bc_basis_directory" : raw_data_path/"satellite"/"bc_basis_directory",
            "output_format":"hbmcmc",
            "country_file": southamerica_country_file
        }
    )
    return mcmc_args

def test_full_satellite_inversion(satellite_mcmc_args):
    satellite_mcmc_args["reload_merged_data"] = False
    out = fixedbasisMCMC(**satellite_mcmc_args)

    assert "Yerror_repeatability" in out
    assert "Yerror_variability" in out

    # sanity check for modelled values to make sure baseline has correct order of magnitude
    assert np.mean(np.abs(out.Yobs.values - out.Yapriori.values)) < 0.5 * np.mean(out.Yobs.values)

def test_full_inversion(mcmc_args):
    mcmc_args["reload_merged_data"] = False
    out = fixedbasisMCMC(**mcmc_args)

    assert "Yerror_repeatability" in out
    assert "Yerror_variability" in out

    # sanity check for modelled values to make sure baseline has correct order of magnitude
    assert np.mean(np.abs(out.Yobs.values - out.Yapriori.values)) < 0.5 * np.mean(out.Yobs.values)

def test_full_inversion_no_model_error(mcmc_args):
    mcmc_args["no_model_error"] = True
    fixedbasisMCMC(**mcmc_args)


def test_full_inversion_flux_dim_shuffled(mcmc_args):
    mcmc_args["emissions_name"] = ["total-ukghg-edgar7-shuffled"]
    mcmc_args["reload_merged_data"] = False
    fixedbasisMCMC(**mcmc_args)


def test_full_inversion_with_min_error_calc(mcmc_args):
    mcmc_args["min_error"] = "residual"
    out = fixedbasisMCMC(**mcmc_args)

    assert "min_model_error" in out.attrs

    mcmc_args["min_error"] = "percentile"
    out = fixedbasisMCMC(**mcmc_args)

    assert "min_model_error" in out.attrs


def test_full_inversion_with_min_error_calc_no_bc(mcmc_args):
    mcmc_args["min_error"] = "residual"
    mcmc_args["use_bc"] = False
    out = fixedbasisMCMC(**mcmc_args)

    assert "min_model_error" in out.attrs


def test_full_inversion_with_min_error_by_site(mcmc_args):
    mcmc_args["min_error"] = "residual"
    mcmc_args["min_error_options"] = {"by_site": True}
    out = fixedbasisMCMC(**mcmc_args)

    assert "min_model_error" in out.attrs


def test_full_inversion_lognormal_infer(mcmc_args):
    mcmc_args["xprior"] = {"pdf": "lognormal", "stdev": 2.0}
    out = fixedbasisMCMC(**mcmc_args)

    expected_sigma = str(np.sqrt(np.log(5)))

    # look for a few decimal places of expected sigma in output attributes
    assert expected_sigma[:4] in out.attrs["Emissions Prior"]


def test_full_inversion_lognormal_reparam(mcmc_args):
    mcmc_args["reparameterise_log_normal"] = True
    mcmc_args["xprior"] = {"pdf": "lognormal", "mu": 1.0, "sigma": 1.0}
    fixedbasisMCMC(**mcmc_args)


def test_full_inversion_min_error(mcmc_args):
    mcmc_args["min_error"] = 20.0
    fixedbasisMCMC(**mcmc_args)


def test_full_inversion_min_error_numpyro(mcmc_args):
    mcmc_args["min_error"] = 20.0
    mcmc_args["nuts_sampler"] = "numpyro"
    fixedbasisMCMC(**mcmc_args)


def test_inversion_if_merged_data_does_not_exist(mcmc_args):
    """Test that inversion runs if reload_merged_data is True, but
    no merged data exists under the default merged data name.
    """
    mcmc_args["merged_data_name"] = None
    fixedbasisMCMC(**mcmc_args)


def test_full_inversion_pollution_events_from_obs(mcmc_args):
    mcmc_args["pollution_events_from_obs"] = True
    fixedbasisMCMC(**mcmc_args)


def test_full_inversion_min_error_no_bc(mcmc_args):
    """Test inversion without boundary conditions."""
    mcmc_args["use_bc"] = False
    fixedbasisMCMC(**mcmc_args)


def test_full_inversion_pollution_events_from_obs_no_bc(mcmc_args):
    mcmc_args["pollution_events_from_obs"] = True
    mcmc_args["use_bc"] = False
    fixedbasisMCMC(**mcmc_args)


def test_full_inversion_two_sites(mcmc_args, mhd_and_tac_ch4_data_args):
    mcmc_args.update(mhd_and_tac_ch4_data_args)
    mcmc_args["reload_merged_data"] = False
    mcmc_args["add_offset"] = True
    mcmc_args["offset_args"] = {"drop_first": True}
    fixedbasisMCMC(**mcmc_args)


@pytest.mark.slow
def test_full_inversion_long(mcmc_args):
    mcmc_args.update(
        {
            "nbasis": 50,
            "nit": 5000,
            "burn": 2000,
            "tune": 1000,
            "nchain": 4,
        }
    )
    fixedbasisMCMC(**mcmc_args)
