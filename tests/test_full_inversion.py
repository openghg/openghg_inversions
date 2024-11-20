import numpy as np
import pymc as pm
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


def test_full_inversion(mcmc_args):
    #mcmc_args["reload_merged_data"] = False
    out = fixedbasisMCMC(**mcmc_args)

    assert "Yerror_repeatability" in out
    assert "Yerror_variability" in out


def test_full_inversion_no_model_error(mcmc_args):
    mcmc_args["no_model_error"] = True
    fixedbasisMCMC(**mcmc_args)


def test_full_inversion_flux_dim_shuffled(mcmc_args):
    mcmc_args["emissions_name"] = ["total-ukghg-edgar7-shuffled"]
    mcmc_args["reload_merged_data"] = False
    fixedbasisMCMC(**mcmc_args)


def test_full_inversion_with_min_error_calc(mcmc_args):
    mcmc_args["calculate_min_error"] = "residual"
    out = fixedbasisMCMC(**mcmc_args)

    assert "min_model_error" in out.attrs

    mcmc_args["calculate_min_error"] = "percentile"
    out = fixedbasisMCMC(**mcmc_args)

    assert "min_model_error" in out.attrs


def test_full_inversion_with_min_error_calc_no_bc(mcmc_args):
    mcmc_args["calculate_min_error"] = "residual"
    mcmc_args["use_bc"] = False
    out = fixedbasisMCMC(**mcmc_args)

    assert "min_model_error" in out.attrs


def test_full_inversion_with_min_error_by_site(mcmc_args):
    mcmc_args["calculate_min_error"] = "residual"
    mcmc_args["min_error_options"] = {"by_site": True}
    out = fixedbasisMCMC(**mcmc_args)

    assert "min_model_error" in out.attrs


def test_full_inversion_lognormal_infer(mcmc_args):
    mcmc_args["xprior"] = {"pdf": "lognormal", "stdev": 2.0, "reparameterise": True}
    mcmc_outs = fixedbasisMCMC(**mcmc_args, skip_postprocessing=True)

    trace = mcmc_outs["trace"]
    trace.extend(pm.sample_prior_predictive(10000, mcmc_outs["model"], random_seed=196883))

    prior_scaling_stdev = trace.prior["forward::flux::x"].std("draw").values

    # check if computed prior stdev is somewhat close to 2.0...
    # the tolerance is needed because the sample stdev seems to converge very slowly
    np.testing.assert_allclose(prior_scaling_stdev, 2.0, atol=0.2)


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
    fixedbasisMCMC(**mcmc_args)

def test_full_inversion_two_sites_with_offset(mcmc_args, mhd_and_tac_ch4_data_args):
    mcmc_args.update(mhd_and_tac_ch4_data_args)
    mcmc_args["reload_merged_data"] = False
    mcmc_args["add_offset"] = True
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
