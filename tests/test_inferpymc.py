from types import MappingProxyType

import numpy as np
import pymc as pm
import pytest

from openghg_inversions.hbmcmc.hbmcmc import make_inv_inputs
from openghg_inversions.hbmcmc.inversion_pymc import InferPyMCModelSetup, build_inferpymc_model, inferpymc


@pytest.fixture(scope="module")
def inferpymc_args(mhd_and_tac_fp_data) -> MappingProxyType:
    """Create direct inputs for inferpymc using existing inversion input machinery.

    NOTE: the module level scope allows us to define other module scoped fixtures
    using this fixture, but it means we need to be careful about not mutating the
    return of the fixture.

    The result is wrapped in a MappingProxyType as a precaution.
    """
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
            "xprior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "bcprior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            "sigprior": {"pdf": "uniform", "lower": 0.1, "upper": 10.0},
            "sigma_per_site": True,
            "offsetprior": {"pdf": "normal", "mu": 0, "sigma": 1},
            "add_offset": False,
            "min_error": 0.0,
            "use_bc": True,
            "reparameterise_log_normal": False,
            "pollution_events_from_obs": True,
            "no_model_error": False,
            "offset_args": {"drop_first": False, "offset_freq": "D"},
            "power": 1.99,
            "verbose": False,
            "sampler_kwargs": {"random_seed": 123, "compute_convergence_checks": False},
        }
    )
    return MappingProxyType(mcmc_args)


def test_build_inferpymc_model_returns_setup_for_pymc(inferpymc_args: dict) -> None:
    """Check extracted model builder returns PyMC step methods for the pymc sampler."""
    setup = build_inferpymc_model(
        Hx=inferpymc_args["Hx"],
        Y=inferpymc_args["Y"],
        error=inferpymc_args["error"],
        siteindicator=inferpymc_args["siteindicator"],
        sigma_freq_index=inferpymc_args["sigma_freq_index"],
        Hbc=inferpymc_args["Hbc"],
        xprior=inferpymc_args["xprior"],
        bcprior=inferpymc_args["bcprior"],
        sigprior=inferpymc_args["sigprior"],
        sigma_per_site=inferpymc_args["sigma_per_site"],
        offsetprior=inferpymc_args["offsetprior"],
        add_offset=inferpymc_args["add_offset"],
        min_error=inferpymc_args["min_error"],
        use_bc=inferpymc_args["use_bc"],
        reparameterise_log_normal=inferpymc_args["reparameterise_log_normal"],
        pollution_events_from_obs=inferpymc_args["pollution_events_from_obs"],
        no_model_error=inferpymc_args["no_model_error"],
        offset_args=inferpymc_args["offset_args"],
        power=inferpymc_args["power"],
        nuts_sampler="pymc",
    )

    assert isinstance(setup, InferPyMCModelSetup)
    assert isinstance(setup.model, pm.Model)
    assert isinstance(setup.step1, pm.NUTS)
    assert isinstance(setup.step2, pm.Slice)
    assert setup.sample_kwargs["step"] == [setup.step1, setup.step2]


def test_build_inferpymc_model_returns_no_steps_for_numpyro(inferpymc_args: dict) -> None:
    """Check model setup does not pass steps them via sample_kwargs['step'] for the numpyro sampler.

    The steps are still created by build_inferpymc_model and stored in the dataclass. This matches
    the pre-existing behaviour, but perhaps should be changed.
    """
    setup = build_inferpymc_model(
        Hx=inferpymc_args["Hx"],
        Y=inferpymc_args["Y"],
        error=inferpymc_args["error"],
        siteindicator=inferpymc_args["siteindicator"],
        sigma_freq_index=inferpymc_args["sigma_freq_index"],
        Hbc=inferpymc_args["Hbc"],
        xprior=inferpymc_args["xprior"],
        bcprior=inferpymc_args["bcprior"],
        sigprior=inferpymc_args["sigprior"],
        sigma_per_site=inferpymc_args["sigma_per_site"],
        offsetprior=inferpymc_args["offsetprior"],
        add_offset=inferpymc_args["add_offset"],
        min_error=inferpymc_args["min_error"],
        use_bc=inferpymc_args["use_bc"],
        reparameterise_log_normal=inferpymc_args["reparameterise_log_normal"],
        pollution_events_from_obs=inferpymc_args["pollution_events_from_obs"],
        no_model_error=inferpymc_args["no_model_error"],
        offset_args=inferpymc_args["offset_args"],
        power=inferpymc_args["power"],
        nuts_sampler="numpyro",
    )

    assert isinstance(setup, InferPyMCModelSetup)
    assert isinstance(setup.model, pm.Model)
    assert isinstance(setup.step1, pm.NUTS)
    assert isinstance(setup.step2, pm.Slice)
    assert setup.sample_kwargs["step"] is None


@pytest.fixture(scope="module")
def inferpymc_with_bc_result(inferpymc_args: dict):
    """Run inferpymc with inferpymc_args, including use_bc=True.

    The results from `inferpymc(**inferpymc_args)` are used at least
    twice in the tests, so this fixture saves some computation.
    """
    return inferpymc(**inferpymc_args)


def test_inferpymc_runs_on_inversion_inputs(inferpymc_args: dict, inferpymc_with_bc_result) -> None:
    """Smoke test inferpymc directly on prepared inversion inputs."""
    result = inferpymc_with_bc_result

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


def test_inferpymc_model_contains_expected_variables(inferpymc_args: dict, inferpymc_with_bc_result) -> None:
    """Check that inferpymc builds the expected PyMC model structure."""
    result = inferpymc_with_bc_result
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
    inferpymc_args = dict(inferpymc_args)
    inferpymc_args["use_bc"] = False
    inferpymc_args["Hbc"] = None

    result = inferpymc(**inferpymc_args)

    assert "bcouts" not in result
    assert "YBCtrace" not in result

    model = result["model"]
    assert "bc" not in model.named_vars
    assert "hbc" not in model.named_vars
    assert "mu_bc" not in model.named_vars
