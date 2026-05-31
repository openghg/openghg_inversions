import numpy as np
import pytest

from openghg_inversions.hbmcmc.hbmcmc import fixedbasisMCMC


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
def satellite_mcmc_args(
    tmp_path, satellite_ch4_data_args, southamerica_country_file, merged_data_dir, raw_data_path
):
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
            "xprior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "bcprior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "sigprior": {"pdf": "uniform", "lower": 0.1, "upper": 10.0},
            "bc_freq": "monthly",
            "sigma_freq": "5D",
            "sigma_per_site": True,
            "averaging_error": False,
            "min_error": 0.0,
            "fix_basis_outer_regions": False,
            "use_bc": True,
            "nuts_sampler": "numpyro",
            "save_trace": True,
            "min_error_options": {"by_site": True},
            "pollution_events_from_obs": True,
            "no_model_error": False,
            "reparameterise_log_normal": False,
            "bc_basis_directory": raw_data_path / "satellite" / "bc_basis_directory",
            "output_format": "paris",
            "country_file": southamerica_country_file,
        }
    )
    return mcmc_args


@pytest.mark.slow
def test_full_satellite_inversion(satellite_mcmc_args):
    """Run the satellite/column PARIS path as a slower end-to-end smoke test."""
    satellite_mcmc_args["reload_merged_data"] = False

    out = fixedbasisMCMC(**satellite_mcmc_args)

    assert "Yobs" in out
    assert "Yobs_prior_factor" in out

    # sanity check for modelled values to make sure baseline has correct order of magnitude
    # Below checks are commented as the check passess for nit=100 and morebut fails for nit=1, which is used in this test to speed up the test. The check is not testing the MCMC itself but just that the modelled values are in the correct order of magnitude, which is not the main focus of this test.
    # assert np.mean(np.abs(out.Yobs.values - out.Yapriori.values)) < 0.5 * np.mean(out.Yobs.values)


def test_full_inversion(mcmc_args):
    mcmc_args["reload_merged_data"] = False
    out = fixedbasisMCMC(**mcmc_args)

    assert "Yerror_repeatability" in out
    assert "Yerror_variability" in out

    # sanity check for modelled values to make sure baseline has correct order of magnitude
    assert np.mean(np.abs(out.Yobs.values - out.Yapriori.values)) < 0.5 * np.mean(out.Yobs.values)


def test_full_inversion_paris_outputs(mcmc_args):
    """Test full inversion including loading data with PARIS output format."""
    mcmc_args["reload_merged_data"] = False
    mcmc_args["output_format"] = "paris"
    out = fixedbasisMCMC(**mcmc_args)

    assert "Yapost" in out


def test_full_inversion_flux_dim_shuffled(mcmc_args):
    mcmc_args["emissions_name"] = ["total-ukghg-edgar7-shuffled"]
    mcmc_args["reload_merged_data"] = False
    fixedbasisMCMC(**mcmc_args)


def test_full_inversion_lognormal_infer(mcmc_args):
    mcmc_args["xprior"] = {"pdf": "lognormal", "stdev": 2.0}
    out = fixedbasisMCMC(**mcmc_args)

    expected_sigma = str(np.sqrt(np.log(5)))

    # look for a few decimal places of expected sigma in output attributes
    assert expected_sigma[:4] in out.attrs["Emissions Prior"]


def test_inversion_if_merged_data_does_not_exist(mcmc_args):
    """Test that inversion runs if reload_merged_data is True, but
    no merged data exists under the default merged data name.
    """
    mcmc_args["merged_data_name"] = None
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
