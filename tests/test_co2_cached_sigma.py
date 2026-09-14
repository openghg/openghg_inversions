"""Named CO2 cached fixed-OU model and joint-output checks."""

from __future__ import annotations

import json
from typing import Any, cast

import arviz as az
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import (
    build_co2_cached_sigma_model,
    run_rhime_co2_cached_sigma,
)
from openghg_inversions.rhime.co2 import co2_cached_sigma_runner
from openghg_inversions.rhime.co2 import co2_cached_sigma_model
from openghg_inversions.rhime.sampling import RhimeSampler


def _inputs() -> xr.Dataset:
    nmeasure = np.arange(4)
    state = np.asarray(["biosphere", "fossil"])
    design = np.asarray(
        [
            [0.8, 0.1],
            [0.4, 0.3],
            [0.2, 0.7],
            [0.5, 0.2],
        ]
    )
    fixed = np.asarray([0.1, 0.2, 0.1, 0.2])
    observations = fixed + design @ np.ones(2) + np.asarray([0.05, -0.04, 0.02, -0.03])
    result = xr.Dataset(
        {
            "H": (("nmeasure", "region"), design),
            "alpha_prior_mean": (("region",), np.ones(2)),
            "alpha_prior_covariance": (
                ("region", "region_cov"),
                np.asarray([[0.08, 0.01], [0.01, 0.06]]),
            ),
            "fixed_prior_contribution": (("nmeasure",), fixed),
            "aggregation_error_covariance": (
                ("nmeasure", "nmeasure_cov"),
                np.diag([0.02, 0.03, 0.02, 0.03]),
            ),
            "mf": (("nmeasure",), observations),
            "mf_error": (("nmeasure",), np.full(4, 0.1)),
        },
        coords={
            "nmeasure": nmeasure,
            "region": state,
            "site": ("nmeasure", ["AAA", "AAA", "BBB", "BBB"]),
            "time": (
                "nmeasure",
                np.asarray(
                    [
                        "2021-01-01T00",
                        "2021-01-01T03",
                        "2021-01-01T01",
                        "2021-01-01T05",
                    ],
                    dtype="datetime64[h]",
                ),
            ),
        },
    )
    for name in ("fixed_prior_contribution", "mf", "mf_error"):
        result[name].attrs["units"] = "ppm"
    return result


def _build(*, state_activity: StateActivity | None = None) -> Any:
    inputs = _inputs()
    prior = CorrelatedLognormalPrior(
        inputs["alpha_prior_mean"],
        inputs["alpha_prior_covariance"],
        covariance_dim="region_cov",
    )
    return build_co2_cached_sigma_model(
        inputs["H"],
        retained_prior=prior,
        fixed_prior_contribution=inputs["fixed_prior_contribution"],
        observations=inputs["mf"],
        observation_error=inputs["mf_error"],
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        tau_hours={"AAA": 3.0, "BBB": 7.0},
        site_amplitude_prior_scale=0.75,
        initial_site_amplitudes={"BBB": 0.4, "unused": 9.0, "AAA": 0.3},
        state_activity=state_activity,
    )


def test_named_cached_co2_model_owns_normalized_potential_and_sampler_inputs() -> None:
    cached = _build()

    assert {
        "flux_scaling_latent",
        "flux_scaling",
        "modelled_concentration",
        "ou_site_amplitude",
        "ou_site_index",
        "cached_fixed_ou_likelihood",
    } <= set(cached.model.named_vars)
    assert "y" not in cached.model.named_vars
    assert cached.state_value_name == "flux_scaling_latent"
    assert cached.target.n_state == 2
    assert cached.target.n_group == 2
    assert cached.target.prepared.site_labels == ("AAA", "BBB")
    np.testing.assert_allclose(cached.target.prepared.tau_hours_by_site, [3.0, 7.0])

    initial_state = np.exp(cached.active_state_prior.latent_mean.values)
    expected = cached.target.log_likelihood(initial_state, [0.3, 0.4])
    actual = cached.shared_cache.log_likelihood(pm.floatX(initial_state)).eval()
    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-7)


def test_active_prior_is_prepared_once_for_graph_and_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_prior: dict[str, CorrelatedLognormalPrior] = {}
    original_add_state = (
        co2_cached_sigma_model._add_prepared_correlated_lognormal_state_with_activity
    )

    def capture_graph_prior(*args: Any, **kwargs: Any) -> Any:
        graph_prior["value"] = args[1]
        return original_add_state(*args, **kwargs)

    monkeypatch.setattr(
        co2_cached_sigma_model,
        "_add_prepared_correlated_lognormal_state_with_activity",
        capture_graph_prior,
    )
    cached = _build(
        state_activity=StateActivity(
            active=xr.DataArray(
                [True, False],
                dims="region",
                coords={"region": ["biosphere", "fossil"]},
            )
        )
    )
    step_args: dict[str, Any] = {}
    monkeypatch.setattr(
        co2_cached_sigma_runner,
        "make_cached_sigma_compound_step",
        lambda **kwargs: step_args.update(kwargs) or object(),
    )

    co2_cached_sigma_runner._sampler_for_cached_graph(
        RhimeSampler(chains=1),
        cached_model=cached,
        sigma_target_accept=0.81,
        state_target_accept=0.92,
    )

    assert graph_prior["value"] is cached.active_state_prior
    assert step_args["initial_cache"] is cached.initial_cache
    assert step_args["sigma_target_accept"] == 0.81
    assert step_args["state_target_accept"] == 0.92
    np.testing.assert_array_equal(
        step_args["state_location"],
        cached.active_state_prior.latent_mean.values,
    )
    np.testing.assert_array_equal(
        step_args["state_cholesky"],
        cached.active_state_prior.latent_cholesky.values,
    )


def test_joint_outputs_are_exact_and_predict_complete_correlated_vectors() -> None:
    cached = _build()
    draws = 6_000
    state = np.broadcast_to(np.asarray([1.0, 1.0]), (1, draws, 2)).copy()
    sigma = np.broadcast_to(np.asarray([0.5, 0.8]), (1, draws, 2)).copy()
    trace = az.from_dict(
        posterior={"flux_scaling": state, "ou_site_amplitude": sigma},
        dims={"flux_scaling": ["region"], "ou_site_amplitude": ["ou_site"]},
        coords={"region": ["biosphere", "fossil"], "ou_site": ["AAA", "BBB"]},
    )
    inputs = _inputs()

    result = co2_cached_sigma_runner._append_joint_outputs(
        trace,
        cached_model=cached,
        observations=inputs["mf"],
        posterior_predictive=True,
        random_seed=42,
    )

    expected_logp = cached.target.log_likelihood([1.0, 1.0], [0.5, 0.8])
    np.testing.assert_allclose(result.log_likelihood["y"], expected_logp)
    assert result.log_likelihood["y"].dims == ("chain", "draw")
    assert result.log_likelihood["y"].attrs["rhime_likelihood_scope"] == (
        "joint_observation_vector"
    )
    predictive = np.asarray(result.posterior_predictive["y"]).reshape(draws, 4)
    expected_covariance = np.diag([0.03, 0.04, 0.03, 0.04])
    expected_covariance[:2, :2] += 0.5**2 * np.asarray(
        [[1.0, np.exp(-1.0)], [np.exp(-1.0), 1.0]]
    )
    expected_covariance[2:, 2:] += 0.8**2 * np.asarray(
        [[1.0, np.exp(-4.0 / 7.0)], [np.exp(-4.0 / 7.0), 1.0]]
    )
    np.testing.assert_allclose(
        np.cov(predictive, rowvar=False),
        expected_covariance,
        atol=0.035,
    )
    assert result.posterior_predictive["y"].attrs["rhime_predictive_scope"] == (
        "joint_observation_vector"
    )


def test_named_runner_samples_real_graph_and_labels_cached_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    step_settings: dict[str, float] = {}
    original_make_step = co2_cached_sigma_runner.make_cached_sigma_compound_step

    def capture_step_settings(**kwargs: Any) -> pm.CompoundStep:
        step_settings["sigma_target_accept"] = kwargs["sigma_target_accept"]
        step_settings["state_target_accept"] = kwargs["state_target_accept"]
        return original_make_step(**kwargs)

    class PreparedInputsStub:
        inv_inputs = inputs

        def validated(self) -> "PreparedInputsStub":
            return self

    monkeypatch.setattr(
        co2_cached_sigma_runner,
        "materialize_pymc_inputs",
        lambda *_args, **_kwargs: inputs,
    )
    monkeypatch.setattr(
        co2_cached_sigma_runner,
        "make_cached_sigma_compound_step",
        capture_step_settings,
    )
    sampler = RhimeSampler(
        draws=2,
        tune=2,
        chains=1,
        progressbar=False,
        sample_kwargs={
            "random_seed": 147,
            "cores": 1,
            "compute_convergence_checks": False,
        },
        sample_prior_predictive=False,
        posterior_predictive_kwargs={"random_seed": 148},
    )

    result = run_rhime_co2_cached_sigma(
        prepared_inputs=cast(Any, PreparedInputsStub()),
        tau_hours={"AAA": 3.0, "BBB": 7.0},
        site_amplitude_prior_scale=0.75,
        initial_site_amplitudes=0.4,
        sampler=sampler,
        sigma_target_accept=0.82,
        state_target_accept=0.93,
        aggregation_error_mode="dense",
    )

    assert step_settings == {
        "sigma_target_accept": 0.82,
        "state_target_accept": 0.93,
    }
    assert result.posterior["flux_scaling"].shape == (1, 2, 2)
    assert result.posterior_predictive["y"].shape == (1, 2, 4)
    assert result.log_likelihood["y"].shape == (1, 2)
    metadata = json.loads(result.attrs["rhime_model_metadata"])
    assert metadata["recipe"] == "co2_cached_sigma_fixed_ou"
    assert "sampler" not in metadata
    assert "numerical_preparation" not in metadata
    assert "rhime_sampler_provenance" not in result.attrs
    assert set(result.posterior["ou_site_amplitude"].coords["ou_site"].values) == {
        "AAA",
        "BBB",
    }
    assert result.posterior["ou_site_amplitude"].attrs["units"] == "ppm"
    assert result.posterior_predictive["y"].attrs["units"] == "ppm"
    assert json.loads(result.posterior_predictive["y"].attrs["rhime_scientific_roles"]) == [
        "concentration"
    ]
    assert "units" not in result.log_likelihood["y"].attrs
    assert json.loads(result.log_likelihood["y"].attrs["rhime_scientific_roles"]) == [
        "joint_log_likelihood"
    ]
    assert result.posterior.attrs["rhime_recipe"] == "co2_cached_sigma_fixed_ou"


def test_cached_runner_rejects_generic_target_accept() -> None:
    class PreparedInputsStub:
        inv_inputs = _inputs()

        def validated(self) -> "PreparedInputsStub":
            return self

    with pytest.raises(
        ValueError,
        match="sigma_target_accept.*state_target_accept",
    ):
        run_rhime_co2_cached_sigma(
            prepared_inputs=cast(Any, PreparedInputsStub()),
            tau_hours={"AAA": 3.0, "BBB": 7.0},
            site_amplitude_prior_scale=0.75,
            sampler=RhimeSampler(sample_kwargs={"target_accept": 0.95}),
            aggregation_error_mode="dense",
        )
