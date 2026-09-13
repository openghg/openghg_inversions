"""Named CO2 cached fixed-OU model, output, and provenance checks."""

from __future__ import annotations

import json
from typing import Any, cast

import arviz as az
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import (
    build_co2_cached_sigma_model,
    run_rhime_co2_cached_sigma,
)
from openghg_inversions.rhime.co2 import co2_cached_sigma_runner
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


def _build() -> Any:
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
        sigma_prior_scale=0.75,
        initial_site_sigma={"AAA": 0.3, "BBB": 0.4},
    )


def test_named_cached_co2_model_owns_normalized_potential_and_sampler_inputs() -> None:
    cached = _build()

    assert {
        "flux_scaling_latent",
        "flux_scaling",
        "modelled_concentration",
        "sigma_site",
        "sigma_observation",
        "cached_fixed_ou_likelihood",
    } <= set(cached.model.named_vars)
    assert "y" not in cached.model.named_vars
    assert cached.state_value_name == "flux_scaling_latent"
    assert cached.target.n_state == 2
    assert cached.target.n_group == 2
    assert cached.covariance.site_labels == ("AAA", "BBB")
    np.testing.assert_allclose(cached.covariance.tau_hours_by_site, [3.0, 7.0])

    initial_state = np.exp(cached.state_location)
    expected = cached.target.log_likelihood(initial_state, [0.3, 0.4])
    actual = cached.shared_cache.log_likelihood(pm.floatX(initial_state)).eval()
    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-7)


def test_joint_outputs_are_exact_and_predict_complete_correlated_vectors() -> None:
    cached = _build()
    draws = 6_000
    state = np.broadcast_to(np.asarray([1.0, 1.0]), (1, draws, 2)).copy()
    sigma = np.broadcast_to(np.asarray([0.5, 0.8]), (1, draws, 2)).copy()
    trace = az.from_dict(
        posterior={"flux_scaling": state, "sigma_site": sigma},
        dims={"flux_scaling": ["region"], "sigma_site": ["sigma_site_dim"]},
        coords={"region": ["biosphere", "fossil"], "sigma_site_dim": ["AAA", "BBB"]},
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
    expected_covariance = cached.covariance.covariance_dense([0.5, 0.8])
    np.testing.assert_allclose(
        np.cov(predictive, rowvar=False),
        expected_covariance,
        atol=0.035,
    )
    assert result.posterior_predictive["y"].attrs["rhime_predictive_scope"] == (
        "joint_observation_vector"
    )


def test_named_runner_samples_real_graph_and_records_sampler_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()

    class PreparedInputsStub:
        inv_inputs = inputs

        def validated(self) -> "PreparedInputsStub":
            return self

    monkeypatch.setattr(
        co2_cached_sigma_runner,
        "materialize_pymc_inputs",
        lambda *_args, **_kwargs: inputs,
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
        sigma_prior_scale=0.75,
        initial_site_sigma=0.4,
        sampler=sampler,
        aggregation_error_mode="dense",
    )

    assert result.posterior["flux_scaling"].shape == (1, 2, 2)
    assert result.posterior_predictive["y"].shape == (1, 2, 4)
    assert result.log_likelihood["y"].shape == (1, 2)
    provenance = json.loads(result.attrs["rhime_sampler_provenance"])
    assert provenance["step_order"] == ["sigma_site_nuts", "state_nuts"]
    metadata = json.loads(result.attrs["rhime_model_metadata"])
    assert metadata["recipe"] == "co2_cached_sigma_fixed_ou"
    assert metadata["sampler"]["verification_games_source"].endswith(
        "@51aaeb101a5d8daf57b1dcf43ea1716c850fc21c"
    )
    assert metadata["sampler"]["random_seed"] == 147
    assert metadata["sampler"]["posterior_predictive_random_seed"] == 148
    assert metadata["numerical_preparation"] == {
        "accepted_quadratic": (
            "chain_local_runtime_state_not_an_external_artifact"
        ),
        "aggregation_error_mode": "dense",
        "fixed_ou_eigenbasis": "derived_at_model_build_from_prepared_inputs",
        "fixed_covariance_rank": 4,
        "initial_site_sigma": {"AAA": 0.4, "BBB": 0.4},
        "site_labels": ["AAA", "BBB"],
        "target": (
            "openghg_inversions.models.cached_sigma.FixedOuCachedSigmaTarget"
        ),
        "tau_hours_by_site": [3.0, 7.0],
    }
    assert set(result.posterior["sigma_site"].coords["sigma_site_dim"].values) == {
        "AAA",
        "BBB",
    }
    assert result.posterior["sigma_site"].attrs["units"] == "ppm"
    assert result.posterior_predictive["y"].attrs["units"] == "ppm"
    assert result.posterior.attrs["rhime_recipe"] == "co2_cached_sigma_fixed_ou"
