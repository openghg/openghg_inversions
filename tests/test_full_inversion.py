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


def test_full_inversion(mcmc_args):
    fixedbasisMCMC(**mcmc_args)


def test_full_inversion_with_min_error_calc(mcmc_args):
    mcmc_args["calculate_min_error"] = True
    out = fixedbasisMCMC(**mcmc_args)

    assert "min_model_error" in out.attrs


def test_full_inversion_pblh_filter(mcmc_args):
    mcmc_args["filters"] = ["pblh"]


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


def test_full_inversion_min_error_no_bc(mcmc_args):
    """Test inversion without boundary conditions."""
    mcmc_args["use_bc"] = False
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
