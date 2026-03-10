import numpy as np
import pytest

from openghg_inversions.hbmcmc.hbmcmc import make_inv_inputs
from openghg_inversions.hbmcmc.inversion_pymc import inferpymc


@pytest.fixture
def inferpymc_args(mhd_and_tac_fp_data) -> dict:
    """Create direct inputs for inferpymc using existing inversion input machinery."""
    mcmc_args, _ = make_inv_inputs(
        fp_data=mhd_and_tac_fp_data,
        sites=["MHD", "TAC"],
        start_date="2019-01-01",
        use_bc=True,
        bc_freq="3h",
        sigma_freq="3h",
        min_error="percentile",
        calculate_min_error=None,
        min_error_options={},
    )

    mcmc_args.update(
        {
            "nit": 1,
            "burn": 0,
            "tune": 0,
            "nchain": 1,
            "verbose": False,
            "sampler_kwargs": {"random_seed": 123},
        }
    )
    return mcmc_args


def test_inferpymc_runs_on_inversion_inputs(inferpymc_args: dict) -> None:
    """Smoke test inferpymc directly on prepared inversion inputs."""
    result = inferpymc(**inferpymc_args)

    expected_keys = {
        "xouts",
        "sigouts",
        "Ytrace",
        "OFFSETtrace",
        "convergence",
        "step1",
        "step2",
        "model",
        "trace",
        "bcouts",
        "YBCtrace",
    }
    assert expected_keys.issubset(result.keys())

    y = np.asarray(inferpymc_args["Y"])
    hx = np.asarray(inferpymc_args["Hx"])
    siteindicator = np.asarray(inferpymc_args["siteindicator"])
    sigma_freq_index = np.asarray(inferpymc_args["sigma_freq_index"])

    assert result["xouts"].sizes["nx"] == hx.shape[0]
    assert result["sigouts"].sizes["nsigma_time"] == len(np.unique(sigma_freq_index))
    assert result["sigouts"].sizes["nsigma_site"] == len(np.unique(siteindicator))

    assert y.size in result["Ytrace"].shape
    assert result["OFFSETtrace"].shape == result["Ytrace"].shape
    assert result["YBCtrace"].shape == result["Ytrace"].shape

    assert np.isfinite(result["Ytrace"]).all()
    assert np.isfinite(result["OFFSETtrace"]).all()
    assert np.isfinite(result["YBCtrace"]).all()


def test_inferpymc_model_contains_expected_variables(inferpymc_args: dict) -> None:
    """Check that inferpymc builds the expected PyMC model structure."""
    result = inferpymc(**inferpymc_args)
    model = result["model"]

    expected_named_vars = {
        "x",
        "bc",
        "sigma",
        "hx",
        "hbc",
        "Y",
        "error",
        "min_error",
        "mu",
        "mu_bc",
        "epsilon",
        "y",
    }

    assert expected_named_vars.issubset(model.named_vars)

    coords = model.coords
    y = np.asarray(inferpymc_args["Y"])
    hx = np.asarray(inferpymc_args["Hx"])
    hbc = np.asarray(inferpymc_args["Hbc"])
    siteindicator = np.asarray(inferpymc_args["siteindicator"])
    sigma_freq_index = np.asarray(inferpymc_args["sigma_freq_index"])

    assert len(coords["nmeasure"]) == y.size
    assert len(coords["nx"]) == hx.shape[0]
    assert len(coords["nbc"]) == hbc.shape[0]
    assert len(coords["nsigma_site"]) == len(np.unique(siteindicator))
    assert len(coords["nsigma_time"]) == len(np.unique(sigma_freq_index))


def test_inferpymc_runs_without_boundary_conditions(inferpymc_args: dict) -> None:
    """Check inferpymc direct call when boundary conditions are disabled."""
    inferpymc_args = inferpymc_args.copy()
    inferpymc_args["use_bc"] = False
    inferpymc_args["Hbc"] = None

    result = inferpymc(**inferpymc_args)

    assert "bcouts" not in result
    assert "YBCtrace" not in result

    model = result["model"]
    assert "bc" not in model.named_vars
    assert "hbc" not in model.named_vars
    assert "mu_bc" not in model.named_vars
