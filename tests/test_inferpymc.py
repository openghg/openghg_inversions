from typing import Any, cast

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import openghg_inversions.hbmcmc.inversion_pymc as inversion_pymc_module
from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.hbmcmc.inversion_pymc import (
    build_inferpymc_model,
    inferpymc,
    sample,
)
from openghg_inversions.models.coords import restore_inferencedata_coords
from openghg_inversions.postprocessing.inversion_output import convert_idata_to_dataset
from openghg_inversions.sigma import SigmaAlignment


@pytest.fixture(scope="module")
def inv_inputs(mhd_and_tac_fp_data) -> xr.Dataset:
    inputs = make_inv_inputs(
        mhd_and_tac_fp_data,
        sites=["MHD", "TAC"],
        bc_freq="3h",
        min_error="percentile",
        min_error_per_site=False,
        start_date="2019-01-01",
    )
    alignment = SigmaAlignment.from_frequency(
        inputs["site_indicator"],
        frequency="3h",
        anchor_time="2019-01-01",
    )
    return inputs.assign(sigma_freq_index=alignment.period_index.rename("sigma_freq_index"))


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


def test_build_inferpymc_model_requires_legacy_sigma_index(
    inv_inputs: xr.Dataset,
    model_args: dict,
) -> None:
    """The hbmcmc adapter requires its explicit legacy sigma compatibility data."""
    with pytest.raises(KeyError, match="sigma_freq_index"):
        build_inferpymc_model(inv_inputs.drop_vars("sigma_freq_index"), **model_args)


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


def test_sample_resets_burned_draws_before_extending_predictive_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy sampling aligns every retained and predictive group after burn-in."""
    raw_trace = az.InferenceData(
        posterior=xr.Dataset(
            {"x": (("chain", "draw"), np.arange(6, dtype=float)[None, :])},
            coords={"chain": [0], "draw": np.arange(6)},
        ),
        sample_stats=xr.Dataset(
            {"diverging": (("chain", "draw"), np.zeros((1, 6), dtype=bool))},
            coords={"chain": [0], "draw": np.arange(6)},
        ),
        log_likelihood=xr.Dataset(
            {"y": (("chain", "draw"), np.zeros((1, 6)))},
            coords={"chain": [0], "draw": np.arange(6)},
        ),
    )

    def fake_sample(**kwargs: Any) -> az.InferenceData:
        """Return a deterministic raw trace without running MCMC."""
        return raw_trace

    def fake_prior_predictive(draws: int, model: pm.Model) -> az.InferenceData:
        """Return zero-based prior groups matching the retained draw count."""
        coords = {"chain": [0], "draw": np.arange(draws)}
        return az.InferenceData(
            prior=xr.Dataset({"x": (("chain", "draw"), np.ones((1, draws)))}, coords=coords),
            prior_predictive=xr.Dataset(
                {"y": (("chain", "draw"), np.ones((1, draws)))},
                coords=coords,
            ),
        )

    def fake_posterior_predictive(
        trace: az.InferenceData,
        **kwargs: Any,
    ) -> az.InferenceData:
        """Return a zero-based posterior-predictive group for the retained trace."""
        draws = cast(Any, trace).posterior.sizes["draw"]
        return az.InferenceData(
            posterior_predictive=xr.Dataset(
                {"y": (("chain", "draw"), np.ones((1, draws)))},
                coords={"chain": [0], "draw": np.arange(draws)},
            )
        )

    monkeypatch.setattr(inversion_pymc_module.pm, "sample", fake_sample)
    monkeypatch.setattr(inversion_pymc_module.pm, "sample_prior_predictive", fake_prior_predictive)
    monkeypatch.setattr(
        inversion_pymc_module.pm,
        "sample_posterior_predictive",
        fake_posterior_predictive,
    )

    result = cast(
        Any,
        sample(
            pm.Model(),
            draws=6,
            burn=3,
            tune=0,
            chains=1,
            sample_prior_predictive=True,
            sample_posterior_predictive=["y"],
        ),
    )
    merged = convert_idata_to_dataset(result)

    expected_draws = np.arange(3)
    for group_name in (
        "posterior",
        "sample_stats",
        "log_likelihood",
        "prior",
        "prior_predictive",
        "posterior_predictive",
    ):
        np.testing.assert_array_equal(result[group_name].draw.values, expected_draws)
    assert result.attrs["burn"] == 3
    assert result.posterior.attrs["burn"] == 3
    assert merged.sizes["draw"] == 3
    np.testing.assert_array_equal(merged.draw.values, expected_draws)
    assert not merged.to_array().isnull().any()


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


def test_inferpymc_forwards_numpyro_sampler_without_pymc_step(
    inv_inputs: xr.Dataset, model_args: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy inferpymc adapter does not inject PyMC step methods for numpyro."""
    captured = {}

    def fake_sample(model: pm.Model, **kwargs):
        captured["model"] = model
        captured["sample_kwargs"] = kwargs
        return object()

    def fake_adapt_legacy_inferpymc_results(**kwargs):
        captured["adapter_kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(inversion_pymc_module, "sample", fake_sample)
    monkeypatch.setattr(
        inversion_pymc_module,
        "_adapt_legacy_inferpymc_results",
        fake_adapt_legacy_inferpymc_results,
    )

    result = inferpymc(
        inv_inputs=inv_inputs,
        nuts_sampler="numpyro",
        nit=1,
        burn=0,
        tune=0,
        nchain=1,
        sampler_kwargs={"random_seed": 123, "compute_convergence_checks": False},
        **model_args,
    )

    assert result == {"ok": True}
    assert captured["sample_kwargs"]["nuts_sampler"] == "numpyro"
    assert "step" not in captured["sample_kwargs"]
    assert "step" not in captured["adapter_kwargs"]["sample_kwargs"]


@pytest.mark.parametrize(
    "option_name",
    [
        "pollution_events_from_obs_one_sided",
        "pollution_events_from_obs_johnson_su",
    ],
)
def test_inferpymc_forwards_pollution_event_options(
    inv_inputs: xr.Dataset,
    model_args: dict,
    monkeypatch: pytest.MonkeyPatch,
    option_name: str,
) -> None:
    """The legacy adapter forwards each opt-in PEFO mode to model construction."""
    captured: dict[str, Any] = {}
    sentinel_model = pm.Model()

    def fake_build_inferpymc_model(dataset: xr.Dataset, **kwargs: Any) -> pm.Model:
        captured["inv_inputs"] = dataset
        captured["model_kwargs"] = kwargs
        return sentinel_model

    def fake_sample(model: pm.Model, **kwargs: Any) -> object:
        captured["sample_model"] = model
        return object()

    def fake_adapt_legacy_inferpymc_results(**kwargs: Any) -> dict[str, bool]:
        return {"ok": True}

    monkeypatch.setattr(
        inversion_pymc_module,
        "build_inferpymc_model",
        fake_build_inferpymc_model,
    )
    monkeypatch.setattr(inversion_pymc_module, "sample", fake_sample)
    monkeypatch.setattr(
        inversion_pymc_module,
        "_adapt_legacy_inferpymc_results",
        fake_adapt_legacy_inferpymc_results,
    )

    result = inferpymc(
        inv_inputs=inv_inputs,
        nuts_sampler="numpyro",
        nit=1,
        burn=0,
        tune=0,
        nchain=1,
        **{option_name: True},
        **model_args,
    )

    assert result == {"ok": True}
    assert captured["inv_inputs"] is inv_inputs
    assert captured["sample_model"] is sentinel_model
    assert captured["model_kwargs"]["pollution_events_from_obs"] is True
    assert captured["model_kwargs"][option_name] is True


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


def test_build_inferpymc_model_with_offset_args_adds_offset_terms(
    inv_inputs: xr.Dataset, model_args: dict
) -> None:
    """Offset args are handled by model construction without a full pipeline run."""
    args = dict(model_args)
    args["add_offset"] = True
    args["offset_args"] = {"drop_first": False, "offset_freq": "D"}

    model = build_inferpymc_model(inv_inputs, **args)

    assert {"offset", "offset_latent", "offset_design", "offset_freq_indicator"}.issubset(model.named_vars)


def test_build_inferpymc_model_with_pollution_events_and_no_bc_builds_likelihood(
    inv_inputs: xr.Dataset, model_args: dict
) -> None:
    """Obs-derived pollution-event scaling works without boundary conditions."""
    args = dict(model_args)
    args["use_bc"] = False
    args["pollution_events_from_obs"] = True

    model = build_inferpymc_model(inv_inputs, **args)

    assert {"mu", "sigma", "epsilon", "y"}.issubset(model.named_vars)
    assert "bc" not in model.named_vars
    assert "hbc" not in model.named_vars
    assert "mu_bc" not in model.named_vars


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
