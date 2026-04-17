import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.hbmcmc.inversion_pymc import (
    build_inferpymc_model,
    inferpymc,
    sample,
)
from openghg_inversions.models.coords import restore_inferencedata_coords


@pytest.fixture(scope="module")
def inv_inputs(mhd_and_tac_fp_data) -> xr.Dataset:
    return make_inv_inputs(
        mhd_and_tac_fp_data,
        sites=["MHD", "TAC"],
        bc_freq="3h",
        sigma_freq="3h",
        min_error="percentile",
        min_error_per_site=False,
        start_date="2019-01-01",
    )


@pytest.fixture
def model_args() -> dict:
    return {
        "xprior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
        "bcprior": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
        "sigprior": {"pdf": "uniform", "lower": 0.1, "upper": 10.0},
        "sigma_per_site": True,
        "offsetprior": {"pdf": "normal", "mu": 0, "sigma": 1},
        "add_offset": False,
        "use_bc": True,
        "reparameterise_log_normal": False,
        "pollution_events_from_obs": True,
        "no_model_error": False,
        "offset_args": {"drop_first": False, "offset_freq": "D"},
        "power": 1.99,
    }


@pytest.fixture
def sample_args() -> dict:
    return {"draws": 1, "tune": 0, "chains": 1, "random_seed": 123, "compute_convergence_checks": False}


def test_build_inferpymc_model_returns_model(inv_inputs: xr.Dataset, model_args: dict) -> None:
    """Building the modern inferpymc model returns a PyMC model with canonical coords."""
    model = build_inferpymc_model(inv_inputs, **model_args)

    assert isinstance(model, pm.Model)
    assert len(model.coords["nmeasure"]) == inv_inputs.sizes["nmeasure"]
    assert len(model.coords["region"]) == inv_inputs.sizes["region"]
    assert len(model.coords["bc_region"]) == inv_inputs.sizes["bc_region"]
    assert len(model.coords["nsigma_site"]) == len(np.unique(inv_inputs["site_indicator"].values))
    assert len(model.coords["nsigma_time"]) == len(np.unique(inv_inputs["sigma_freq_index"].values))


def test_build_inferpymc_model_warns_for_deprecated_reparameterise_flag(
    inv_inputs: xr.Dataset, model_args: dict
) -> None:
    """The transitional lognormal reparameterisation flag still warns."""
    args = dict(model_args)
    args["xprior"] = {"pdf": "lognormal", "mu": 1.0, "sigma": 1.0}
    args["reparameterise_log_normal"] = True

    with pytest.warns(DeprecationWarning, match="reparameterise=True"):
        build_inferpymc_model(inv_inputs, **args)


def test_build_inferpymc_model_requires_inv_inputs() -> None:
    """The model builder requires canonical inversion inputs."""
    with pytest.raises(TypeError):
        build_inferpymc_model()  # type: ignore[call-arg]


def test_sample_returns_burned_modern_result(
    inv_inputs: xr.Dataset, model_args: dict, sample_args: dict
) -> None:
    """Modern sampling returns burn-sliced inference data with canonical dims."""
    model = build_inferpymc_model(inv_inputs, **model_args)
    args = dict(sample_args)
    args.update({"draws": 2, "burn": 1})

    modern_result = sample(model, **args)

    assert isinstance(modern_result, az.InferenceData)
    assert modern_result.posterior.sizes["draw"] == 1
    assert "region" in modern_result.posterior["x"].dims
    assert "bc_region" in modern_result.posterior["bc"].dims


def test_sample_does_not_add_predictive_groups_by_default(
    inv_inputs: xr.Dataset, model_args: dict, sample_args: dict
) -> None:
    """Modern sampling leaves predictive groups out unless requested."""
    model = build_inferpymc_model(inv_inputs, **model_args)
    modern_result = sample(model, **sample_args)

    assert "prior" not in modern_result
    assert "prior_predictive" not in modern_result
    assert "posterior_predictive" not in modern_result


def test_sample_accepts_plain_model_and_predictive_options(
    inv_inputs: xr.Dataset, model_args: dict, sample_args: dict
) -> None:
    """Modern sampling adds predictive groups only when explicitly requested."""
    model = build_inferpymc_model(inv_inputs, **model_args)
    args = dict(sample_args)
    args.update({"sample_prior_predictive": True, "sample_posterior_predictive": ["y"]})

    modern_result = sample(model, **args)

    assert "prior" in modern_result
    assert "prior_predictive" in modern_result
    assert "posterior_predictive" in modern_result


def test_sample_always_returns_inferencedata(
    inv_inputs: xr.Dataset, model_args: dict, sample_args: dict
) -> None:
    """Modern sampling always returns InferenceData regardless of caller kwargs."""
    model = build_inferpymc_model(inv_inputs, **model_args)
    args = dict(sample_args)
    args["return_inferencedata"] = False

    modern_result = sample(model, **args)

    assert isinstance(modern_result, az.InferenceData)


def test_sample_preserves_log_likelihood(inv_inputs: xr.Dataset, model_args: dict, sample_args: dict) -> None:
    """Modern sampling keeps log likelihood data in the trace."""
    model = build_inferpymc_model(inv_inputs, **model_args)
    modern_result = sample(model, **sample_args)

    assert "log_likelihood" in modern_result


def test_inferpymc_preserves_legacy_compatibility_outputs(inv_inputs: xr.Dataset, model_args: dict) -> None:
    """The public inferpymc wrapper still returns the legacy-shaped outputs."""
    result = inferpymc(
        inv_inputs=inv_inputs,
        nit=1,
        burn=0,
        tune=0,
        nchain=1,
        verbose=False,
        sampler_kwargs={"random_seed": 123, "compute_convergence_checks": False},
        **model_args,
    )

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

    assert "prior_predictive" in result["trace"]
    assert "posterior_predictive" in result["trace"]
    assert "nx" in result["trace"].posterior["x"].dims
    assert "nbc" in result["trace"].posterior["bc"].dims
    assert "region" not in result["trace"].posterior["x"].dims
    assert "bc_region" not in result["trace"].posterior["bc"].dims

    assert result["xouts"].sizes["nx"] == inv_inputs.sizes["region"]
    assert result["sigouts"].sizes["nsigma_time"] == len(np.unique(inv_inputs["sigma_freq_index"].values))
    assert result["sigouts"].sizes["nsigma_site"] == len(np.unique(inv_inputs["site_indicator"].values))

    y = inv_inputs["mf"].values
    assert y.size in result["Ytrace"].shape
    assert result["OFFSETtrace"].shape == result["Ytrace"].shape
    assert result["YBCtrace"].shape == result["Ytrace"].shape

    assert np.isfinite(result["Ytrace"]).all()
    assert np.isfinite(result["OFFSETtrace"]).all()
    assert np.isfinite(result["YBCtrace"]).all()

    assert "step1" in result
    assert "step2" in result


def test_build_inferpymc_model_contains_expected_variables(inv_inputs: xr.Dataset, model_args: dict) -> None:
    """The builder adds the core named variables expected by downstream code."""
    model = build_inferpymc_model(inv_inputs, **model_args)

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


def test_restore_inferencedata_coords_helper_restores_multiindex() -> None:
    """Coordinate restoration recreates the original MultiIndex on InferenceData."""
    multi_index = pd.MultiIndex.from_arrays(
        [["MHD", "MHD", "TAC"], pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-01"])],
        names=["site", "time"],
    )
    posterior = xr.Dataset(
        data_vars={"x": (("chain", "draw", "nmeasure"), np.zeros((1, 1, 3)))},
        coords={"chain": [0], "draw": [0], "nmeasure": np.arange(3)},
    )
    idata = az.InferenceData(posterior=posterior)

    restored = restore_inferencedata_coords(idata, {"nmeasure": multi_index})

    assert "nmeasure" in restored.posterior.indexes
    assert restored.posterior.indexes["nmeasure"].equals(multi_index)


def test_build_inferpymc_model_without_boundary_conditions_omits_bc_vars(
    inv_inputs: xr.Dataset, model_args: dict
) -> None:
    """Disabling boundary conditions removes BC variables at model-build time."""
    args = dict(model_args)
    args["use_bc"] = False

    model = build_inferpymc_model(inv_inputs, **args)
    assert "bc" not in model.named_vars
    assert "hbc" not in model.named_vars
    assert "mu_bc" not in model.named_vars
