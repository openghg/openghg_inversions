from types import MappingProxyType

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.hbmcmc.inversion_pymc import (
    _adapt_legacy_inferpymc_results,
    build_inferpymc_model,
    inferpymc,
    sample,
)
from openghg_inversions.models.coords import restore_inferencedata_coords


@pytest.fixture(scope="module")
def inferpymc_inputs_dataset(mhd_and_tac_fp_data) -> xr.Dataset:
    return make_inv_inputs(
        mhd_and_tac_fp_data,
        sites=["MHD", "TAC"],
        bc_freq="3h",
        sigma_freq="3h",
        min_error="percentile",
        min_error_per_site=False,
        start_date="2019-01-01",
    )


@pytest.fixture(scope="module")
def inferpymc_args(inferpymc_inputs_dataset: xr.Dataset) -> MappingProxyType:
    args = {
        "inv_inputs": inferpymc_inputs_dataset,
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
        "use_bc": True,
        "reparameterise_log_normal": False,
        "pollution_events_from_obs": True,
        "no_model_error": False,
        "offset_args": {"drop_first": False, "offset_freq": "D"},
        "power": 1.99,
        "verbose": False,
        "sampler_kwargs": {"random_seed": 123, "compute_convergence_checks": False},
    }
    return MappingProxyType(args)


def test_build_inferpymc_model_returns_model(inferpymc_args: dict) -> None:
    model = build_inferpymc_model(
        inferpymc_args["inv_inputs"],
        xprior=inferpymc_args["xprior"],
        bcprior=inferpymc_args["bcprior"],
        sigprior=inferpymc_args["sigprior"],
        sigma_per_site=inferpymc_args["sigma_per_site"],
        offsetprior=inferpymc_args["offsetprior"],
        add_offset=inferpymc_args["add_offset"],
        use_bc=inferpymc_args["use_bc"],
        reparameterise_log_normal=inferpymc_args["reparameterise_log_normal"],
        pollution_events_from_obs=inferpymc_args["pollution_events_from_obs"],
        no_model_error=inferpymc_args["no_model_error"],
        offset_args=inferpymc_args["offset_args"],
        power=inferpymc_args["power"],
    )

    assert isinstance(model, pm.Model)


def test_build_inferpymc_model_warns_for_deprecated_reparameterise_flag(inferpymc_args: dict) -> None:
    with pytest.warns(DeprecationWarning, match="reparameterise=True"):
        build_inferpymc_model(
            inferpymc_args["inv_inputs"],
            xprior={"pdf": "lognormal", "mu": 1.0, "sigma": 1.0},
            bcprior=inferpymc_args["bcprior"],
            sigprior=inferpymc_args["sigprior"],
            sigma_per_site=inferpymc_args["sigma_per_site"],
            offsetprior=inferpymc_args["offsetprior"],
            add_offset=inferpymc_args["add_offset"],
            use_bc=inferpymc_args["use_bc"],
            reparameterise_log_normal=True,
            pollution_events_from_obs=inferpymc_args["pollution_events_from_obs"],
            no_model_error=inferpymc_args["no_model_error"],
            offset_args=inferpymc_args["offset_args"],
            power=inferpymc_args["power"],
        )


def test_build_inferpymc_model_requires_inv_inputs() -> None:
    with pytest.raises(TypeError):
        build_inferpymc_model()  # type: ignore[call-arg]


def test_sample_returns_burned_modern_result(inferpymc_args: dict) -> None:
    model = build_inferpymc_model(
        inferpymc_args["inv_inputs"],
        xprior=inferpymc_args["xprior"],
        bcprior=inferpymc_args["bcprior"],
        sigprior=inferpymc_args["sigprior"],
        sigma_per_site=inferpymc_args["sigma_per_site"],
        offsetprior=inferpymc_args["offsetprior"],
        add_offset=inferpymc_args["add_offset"],
        use_bc=inferpymc_args["use_bc"],
        reparameterise_log_normal=inferpymc_args["reparameterise_log_normal"],
        pollution_events_from_obs=inferpymc_args["pollution_events_from_obs"],
        no_model_error=inferpymc_args["no_model_error"],
        offset_args=inferpymc_args["offset_args"],
        power=inferpymc_args["power"],
    )

    modern_result = sample(
        model,
        draws=2,
        burn=1,
        tune=0,
        chains=1,
        random_seed=123,
        compute_convergence_checks=False,
    )

    assert isinstance(modern_result, az.InferenceData)
    assert modern_result.posterior.sizes["draw"] == 1
    assert "region" in modern_result.posterior["x"].dims
    assert "bc_region" in modern_result.posterior["bc"].dims


def test_sample_does_not_add_predictive_groups_by_default(inferpymc_args: dict) -> None:
    model = build_inferpymc_model(
        inferpymc_args["inv_inputs"],
        xprior=inferpymc_args["xprior"],
        bcprior=inferpymc_args["bcprior"],
        sigprior=inferpymc_args["sigprior"],
        sigma_per_site=inferpymc_args["sigma_per_site"],
        offsetprior=inferpymc_args["offsetprior"],
        add_offset=inferpymc_args["add_offset"],
        use_bc=inferpymc_args["use_bc"],
        reparameterise_log_normal=inferpymc_args["reparameterise_log_normal"],
        pollution_events_from_obs=inferpymc_args["pollution_events_from_obs"],
        no_model_error=inferpymc_args["no_model_error"],
        offset_args=inferpymc_args["offset_args"],
        power=inferpymc_args["power"],
    )

    modern_result = sample(
        model,
        draws=1,
        tune=0,
        chains=1,
        random_seed=123,
        compute_convergence_checks=False,
    )

    assert "prior" not in modern_result
    assert "prior_predictive" not in modern_result
    assert "posterior_predictive" not in modern_result


def test_sample_accepts_plain_model_and_predictive_options(inferpymc_args: dict) -> None:
    model = build_inferpymc_model(
        inferpymc_args["inv_inputs"],
        xprior=inferpymc_args["xprior"],
        bcprior=inferpymc_args["bcprior"],
        sigprior=inferpymc_args["sigprior"],
        sigma_per_site=inferpymc_args["sigma_per_site"],
        offsetprior=inferpymc_args["offsetprior"],
        add_offset=inferpymc_args["add_offset"],
        use_bc=inferpymc_args["use_bc"],
        reparameterise_log_normal=inferpymc_args["reparameterise_log_normal"],
        pollution_events_from_obs=inferpymc_args["pollution_events_from_obs"],
        no_model_error=inferpymc_args["no_model_error"],
        offset_args=inferpymc_args["offset_args"],
        power=inferpymc_args["power"],
    )

    modern_result = sample(
        model,
        draws=1,
        tune=0,
        chains=1,
        random_seed=123,
        compute_convergence_checks=False,
        sample_prior_predictive=True,
        sample_posterior_predictive=["y"],
    )

    assert "prior" in modern_result
    assert "prior_predictive" in modern_result
    assert "posterior_predictive" in modern_result


def test_sample_always_returns_inferencedata(inferpymc_args: dict) -> None:
    model = build_inferpymc_model(
        inferpymc_args["inv_inputs"],
        xprior=inferpymc_args["xprior"],
        bcprior=inferpymc_args["bcprior"],
        sigprior=inferpymc_args["sigprior"],
        sigma_per_site=inferpymc_args["sigma_per_site"],
        offsetprior=inferpymc_args["offsetprior"],
        add_offset=inferpymc_args["add_offset"],
        use_bc=inferpymc_args["use_bc"],
        reparameterise_log_normal=inferpymc_args["reparameterise_log_normal"],
        pollution_events_from_obs=inferpymc_args["pollution_events_from_obs"],
        no_model_error=inferpymc_args["no_model_error"],
        offset_args=inferpymc_args["offset_args"],
        power=inferpymc_args["power"],
    )

    modern_result = sample(
        model,
        draws=1,
        tune=0,
        chains=1,
        random_seed=123,
        compute_convergence_checks=False,
        return_inferencedata=False,
    )

    assert isinstance(modern_result, az.InferenceData)


def test_sample_preserves_log_likelihood(inferpymc_args: dict) -> None:
    model = build_inferpymc_model(
        inferpymc_args["inv_inputs"],
        xprior=inferpymc_args["xprior"],
        bcprior=inferpymc_args["bcprior"],
        sigprior=inferpymc_args["sigprior"],
        sigma_per_site=inferpymc_args["sigma_per_site"],
        offsetprior=inferpymc_args["offsetprior"],
        add_offset=inferpymc_args["add_offset"],
        use_bc=inferpymc_args["use_bc"],
        reparameterise_log_normal=inferpymc_args["reparameterise_log_normal"],
        pollution_events_from_obs=inferpymc_args["pollution_events_from_obs"],
        no_model_error=inferpymc_args["no_model_error"],
        offset_args=inferpymc_args["offset_args"],
        power=inferpymc_args["power"],
    )

    modern_result = sample(
        model,
        draws=1,
        tune=0,
        chains=1,
        random_seed=123,
        compute_convergence_checks=False,
    )

    assert "log_likelihood" in modern_result


def test_legacy_inferpymc_adapter_preserves_compatibility_keys(inferpymc_args: dict) -> None:
    model = build_inferpymc_model(
        inferpymc_args["inv_inputs"],
        xprior=inferpymc_args["xprior"],
        bcprior=inferpymc_args["bcprior"],
        sigprior=inferpymc_args["sigprior"],
        sigma_per_site=inferpymc_args["sigma_per_site"],
        offsetprior=inferpymc_args["offsetprior"],
        add_offset=inferpymc_args["add_offset"],
        use_bc=inferpymc_args["use_bc"],
        reparameterise_log_normal=inferpymc_args["reparameterise_log_normal"],
        pollution_events_from_obs=inferpymc_args["pollution_events_from_obs"],
        no_model_error=inferpymc_args["no_model_error"],
        offset_args=inferpymc_args["offset_args"],
        power=inferpymc_args["power"],
    )
    modern_result = sample(
        model,
        draws=1,
        burn=0,
        tune=0,
        chains=1,
        random_seed=123,
        compute_convergence_checks=False,
        sample_prior_predictive=True,
        sample_posterior_predictive=["y"],
    )

    legacy_result = _adapt_legacy_inferpymc_results(
        trace=modern_result,
        model=model,
        use_bc=True,
        add_offset=False,
        sample_kwargs={"step": pm.Slice(vars=[model.named_vars["sigma"]], model=model)},
    )

    assert {
        "xouts",
        "sigouts",
        "Ytrace",
        "OFFSETtrace",
        "trace",
        "model",
        "step1",
        "step2",
        "convergence",
    }.issubset(legacy_result)
    assert "bcouts" in legacy_result
    assert "YBCtrace" in legacy_result
    assert "prior_predictive" in legacy_result["trace"]
    assert "posterior_predictive" in legacy_result["trace"]
    assert "nx" in legacy_result["trace"].posterior["x"].dims
    assert "nbc" in legacy_result["trace"].posterior["bc"].dims
    assert "region" not in legacy_result["trace"].posterior["x"].dims
    assert "bc_region" not in legacy_result["trace"].posterior["bc"].dims
    assert legacy_result["step1"] is None
    assert isinstance(legacy_result["step2"], pm.Slice)


@pytest.fixture(scope="module")
def inferpymc_with_bc_result(inferpymc_args: dict):
    return inferpymc(**inferpymc_args)


def test_inferpymc_runs_on_inversion_inputs(
    inferpymc_inputs_dataset: xr.Dataset, inferpymc_with_bc_result
) -> None:
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

    assert result["xouts"].sizes["nx"] == inferpymc_inputs_dataset.sizes["region"]
    assert result["sigouts"].sizes["nsigma_time"] == len(
        np.unique(inferpymc_inputs_dataset["sigma_freq_index"].values)
    )
    assert result["sigouts"].sizes["nsigma_site"] == len(
        np.unique(inferpymc_inputs_dataset["site_indicator"].values)
    )

    y = inferpymc_inputs_dataset["mf"].values
    assert y.size in result["Ytrace"].shape
    assert result["OFFSETtrace"].shape == result["Ytrace"].shape
    assert result["YBCtrace"].shape == result["Ytrace"].shape

    assert np.isfinite(result["Ytrace"]).all()
    assert np.isfinite(result["OFFSETtrace"]).all()
    assert np.isfinite(result["YBCtrace"]).all()


def test_inferpymc_model_contains_expected_variables(
    inferpymc_inputs_dataset: xr.Dataset, inferpymc_with_bc_result
) -> None:
    model = inferpymc_with_bc_result["model"]

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

    assert len(model.coords["nmeasure"]) == inferpymc_inputs_dataset.sizes["nmeasure"]
    assert len(model.coords["region"]) == inferpymc_inputs_dataset.sizes["region"]
    assert len(model.coords["bc_region"]) == inferpymc_inputs_dataset.sizes["bc_region"]
    assert len(model.coords["nsigma_site"]) == len(
        np.unique(inferpymc_inputs_dataset["site_indicator"].values)
    )
    assert len(model.coords["nsigma_time"]) == len(
        np.unique(inferpymc_inputs_dataset["sigma_freq_index"].values)
    )


def test_build_inferpymc_model_accepts_dataset(
    inferpymc_args: dict, inferpymc_inputs_dataset: xr.Dataset
) -> None:
    model = build_inferpymc_model(
        inferpymc_inputs_dataset,
        xprior=inferpymc_args["xprior"],
        bcprior=inferpymc_args["bcprior"],
        sigprior=inferpymc_args["sigprior"],
        sigma_per_site=inferpymc_args["sigma_per_site"],
        offsetprior=inferpymc_args["offsetprior"],
        add_offset=inferpymc_args["add_offset"],
        use_bc=inferpymc_args["use_bc"],
        reparameterise_log_normal=inferpymc_args["reparameterise_log_normal"],
        pollution_events_from_obs=inferpymc_args["pollution_events_from_obs"],
        no_model_error=inferpymc_args["no_model_error"],
        offset_args=inferpymc_args["offset_args"],
        power=inferpymc_args["power"],
    )

    assert isinstance(model, pm.Model)
    assert len(model.coords["nmeasure"]) == inferpymc_inputs_dataset.sizes["nmeasure"]
    assert len(model.coords["region"]) == inferpymc_inputs_dataset.sizes["region"]


def test_restore_inferencedata_coords_helper_restores_multiindex() -> None:
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


def test_inferpymc_runs_without_boundary_conditions(inferpymc_args: dict) -> None:
    args = dict(inferpymc_args)
    args["use_bc"] = False

    result = inferpymc(**args)

    assert "bcouts" not in result
    assert "YBCtrace" not in result

    model = result["model"]
    assert "bc" not in model.named_vars
    assert "hbc" not in model.named_vars
    assert "mu_bc" not in model.named_vars
