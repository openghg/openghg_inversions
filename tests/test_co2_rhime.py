"""Focused equation and graph tests for the CO2 RHIME recipe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytest
import xarray as xr

from openghg_inversions.models.coords import get_coord_registry
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import (
    build_co2_rhime_model,
    co2_prior_forward_mean,
    run_rhime_co2,
)
from openghg_inversions.rhime.co2 import co2_runner
from openghg_inversions.serialization import load_inferencedata, save_inferencedata


FIXTURE = Path(__file__).parent / "data" / "co2_only_golden.json"


def _golden_inputs() -> xr.Dataset:
    fixture = json.loads(FIXTURE.read_text())
    operator = np.asarray(fixture["operator"])
    prior_mean = np.ones(operator.shape[1])
    fixed = np.asarray(fixture["prior_forward_mf"]) - operator @ prior_mean
    labels = fixture["state_labels"]
    inputs = xr.Dataset(
        {
            "H": (("nmeasure", "region"), operator),
            "alpha_prior_mean": (("region",), prior_mean),
            "alpha_prior_covariance": (
                ("region", "region_cov"),
                fixture["prior_covariance"],
            ),
            "fixed_prior_contribution": (("nmeasure",), fixed),
            "aggregation_error_covariance": (
                ("nmeasure", "nmeasure_cov"),
                fixture["aggregation_error_covariance"],
            ),
            "mf": (("nmeasure",), fixture["mf"]),
            "mf_error": (("nmeasure",), np.zeros(operator.shape[0])),
            "min_error": (("nmeasure",), np.zeros(operator.shape[0])),
        },
        coords={"region": labels, "nmeasure": np.arange(operator.shape[0])},
    )
    for name in ("mf", "mf_error", "min_error", "fixed_prior_contribution"):
        inputs[name].attrs["units"] = "ppm"
    return inputs


def _empty_sampled_trace(inputs: xr.Dataset) -> az.InferenceData:
    """Return a small trace with representative CO2 sample groups."""
    return az.from_dict(
        posterior={
            "x": np.ones((1, 2, inputs.sizes["region"])),
            "mu": np.ones((1, 2, inputs.sizes["nmeasure"])),
            "epsilon": np.ones((1, 2, inputs.sizes["nmeasure"])),
        },
        posterior_predictive={"y": np.ones((1, 2, inputs.sizes["nmeasure"]))},
        constant_data={
            "fixed_model_mismatch": np.ones(inputs.sizes["nmeasure"]),
            "fixed_prior_contribution": inputs["fixed_prior_contribution"].values,
            "hx": inputs["H"].values,
        },
        dims={
            "x": ["region"],
            "mu": ["nmeasure"],
            "epsilon": ["nmeasure"],
            "y": ["nmeasure"],
            "fixed_model_mismatch": ["nmeasure"],
            "fixed_prior_contribution": ["nmeasure"],
            "hx": ["nmeasure", "region"],
        },
    )


def _build_model(inputs: xr.Dataset, **kwargs: Any) -> pm.Model:
    return build_co2_rhime_model(
        inputs["H"],
        prior_mean=inputs["alpha_prior_mean"],
        prior_covariance=inputs["alpha_prior_covariance"],
        fixed_prior_contribution=inputs["fixed_prior_contribution"],
        observations=inputs["mf"],
        observation_error=inputs["mf_error"],
        minimum_error=inputs["min_error"],
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        no_model_error=True,
        **kwargs,
    )


def test_co2_prior_forward_mean_matches_ope74_golden_contract() -> None:
    inputs = _golden_inputs()
    fixture = json.loads(FIXTURE.read_text())
    xr.testing.assert_allclose(
        co2_prior_forward_mean(
            inputs["H"],
            prior_mean=inputs["alpha_prior_mean"],
            fixed_prior_contribution=inputs["fixed_prior_contribution"],
        ),
        xr.DataArray(
            fixture["prior_forward_mf"],
            dims="nmeasure",
            coords={"nmeasure": inputs["nmeasure"]},
            name="prior_forward_mean",
        ),
    )


def test_co2_prior_forward_mean_uses_fixed_values_for_inactive_states() -> None:
    inputs = _golden_inputs()
    nstate = inputs.sizes["region"]
    prior_mean = xr.DataArray(
        np.arange(nstate, dtype=float) + 2.0,
        dims="region",
        coords={"region": inputs["region"]},
    )
    is_active = xr.DataArray(
        [True, False, True, True],
        dims="region",
        coords={"region": inputs["region"]},
    )
    fixed_value = xr.DataArray(
        np.full(nstate, 9.0),
        dims="region",
        coords={"region": inputs["region"]},
    )
    activity = StateActivity(active=is_active, fixed_value=fixed_value)

    expected_state = np.where(is_active, prior_mean, fixed_value)
    expected = inputs["fixed_prior_contribution"].values + inputs["H"].values @ expected_state
    actual = co2_prior_forward_mean(
        inputs["H"],
        prior_mean=prior_mean,
        fixed_prior_contribution=inputs["fixed_prior_contribution"],
        state_activity=activity,
    )

    np.testing.assert_allclose(actual, expected)


def test_co2_prior_forward_mean_rejects_mismatched_labels() -> None:
    inputs = _golden_inputs()
    reordered_state = inputs["alpha_prior_mean"].isel(region=[1, 0, 2, 3])
    reordered_observations = inputs["fixed_prior_contribution"].isel(nmeasure=slice(None, None, -1))

    with pytest.raises(ValueError, match="cannot align.*region"):
        co2_prior_forward_mean(
            inputs["H"],
            prior_mean=reordered_state,
            fixed_prior_contribution=inputs["fixed_prior_contribution"],
        )
    with pytest.raises(ValueError, match="cannot align.*nmeasure"):
        co2_prior_forward_mean(
            inputs["H"],
            prior_mean=inputs["alpha_prior_mean"],
            fixed_prior_contribution=reordered_observations,
        )


def test_co2_model_exposes_affine_correlated_dense_covariance_graph() -> None:
    model = _build_model(_golden_inputs(), fixed_model_mismatch=1.0)

    assert {
        "hx",
        "x_latent",
        "x",
        "mu_pollution",
        "fixed_prior_contribution",
        "mu",
        "Y",
        "error",
        "fixed_model_mismatch",
        "min_error",
        "epsilon",
        "y",
    } <= set(model.named_vars)
    assert model.named_vars_to_dims["x"] == ("region",)
    assert model.named_vars_to_dims["mu"] == ("nmeasure",)
    assert isinstance(model["y"].owner.op, pm.MvNormal.rv_op.__class__)


def test_co2_fixed_mismatch_completes_dense_observation_covariance() -> None:
    """The fixed likelihood uses R = A + diag(error²) + diag(mismatch²)."""
    inputs = _golden_inputs()
    nmeasure = inputs.sizes["nmeasure"]
    observation_error = np.linspace(0.2, 0.4, nmeasure)
    fixed_mismatch = xr.DataArray(
        np.linspace(0.75, 1.25, nmeasure),
        dims="nmeasure",
        coords={"nmeasure": inputs["nmeasure"]},
    )
    inputs["mf_error"] = ("nmeasure", observation_error)
    model = _build_model(inputs, fixed_model_mismatch=fixed_mismatch)

    expected_covariance = (
        inputs["aggregation_error_covariance"].values
        + np.diag(observation_error**2)
        + np.diag(fixed_mismatch.values**2)
    )

    np.testing.assert_allclose(model["y"].owner.inputs[-1].eval(), expected_covariance)
    np.testing.assert_allclose(model["fixed_model_mismatch"].eval(), fixed_mismatch)
    np.testing.assert_allclose(model["epsilon"].eval() ** 2, np.diag(expected_covariance))


def test_co2_partial_activity_preserves_full_gathered_multiindex_state() -> None:
    inputs = _golden_inputs().drop_vars("region")
    state_index = pd.MultiIndex.from_tuples(
        [
            ("ff", "north"),
            ("ff", "south"),
            ("ocean", "atlantic"),
            ("biosphere", "temperate"),
        ],
        names=("source", "region_in_source"),
    )
    inputs = inputs.assign_coords(xr.Coordinates.from_pandas_multiindex(state_index, "region"))
    is_active = xr.DataArray(
        [True, False, True, False],
        dims="region",
        coords={"region": inputs["region"]},
    )
    fixed_value = xr.DataArray(
        [101.0, 12.0, 103.0, 14.0],
        dims="region",
        coords={"region": inputs["region"]},
    )

    model = _build_model(
        inputs,
        fixed_model_mismatch=1.0,
        state_activity=StateActivity(
            active=is_active,
            fixed_value=fixed_value,
        ),
    )
    full_state, active_state, forward = pm.draw(
        [model["x"], model["x_active"], model["mu_pollution"]],
        random_seed=42,
    )
    registry = get_coord_registry(model)

    assert registry is not None
    assert full_state.shape == (4,)
    assert active_state.shape == (2,)
    np.testing.assert_array_equal(full_state[~is_active.values], fixed_value.values[~is_active.values])
    np.testing.assert_allclose(forward, inputs["H"].values @ full_state)
    assert registry.original_coords["region"].equals(state_index)
    assert registry.original_coords["region_x_active"].tolist() == [
        ("ff", "north"),
        ("ocean", "atlantic"),
    ]
    assert registry.auxiliary_coords["source"].values.tolist() == [
        "ff",
        "ff",
        "ocean",
        "biosphere",
    ]


def test_public_co2_runner_persists_fixed_mismatch_manifest(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    inputs = _golden_inputs()

    class PreparedInputsStub:
        inv_inputs = inputs

        def validated(self) -> "PreparedInputsStub":
            return self

    monkeypatch.setattr(co2_runner, "materialize_pymc_inputs", lambda *_args, **_kwargs: inputs)
    sampled_models: list[pm.Model] = []

    def sample_model(built: Any, _sampler: Any) -> az.InferenceData:
        sampled_models.append(built.model)
        return _empty_sampled_trace(inputs)

    monkeypatch.setattr(co2_runner, "sample_rhime_model", sample_model)
    result = run_rhime_co2(
        prepared_inputs=cast(Any, PreparedInputsStub()),
        fixed_model_mismatch=1.0,
        no_model_error=True,
    )

    np.testing.assert_allclose(sampled_models[0]["fixed_model_mismatch"].eval(), 1.0)
    roles = json.loads(result.attrs["rhime_variable_roles"])
    metadata = json.loads(result.attrs["rhime_model_metadata"])
    assert roles["coherent_prior_contribution"] == "fixed_prior_contribution"
    assert metadata["recipe"] == "co2"
    assert metadata["basis_artifact_source"] == "unknown"
    assert result.posterior["x"].attrs["units"] == "1"
    assert result.posterior_predictive["y"].attrs["units"] == "ppm"
    assert json.loads(result.constant_data["fixed_model_mismatch"].attrs["rhime_scientific_roles"]) == [
        "fixed_model_mismatch"
    ]

    path = tmp_path / "co2-trace.nc"
    save_inferencedata(result, path)
    restored = load_inferencedata(path)
    assert json.loads(restored.attrs["rhime_model_metadata"])["recipe"] == "co2"
    assert restored.posterior["x"].attrs["units"] == "1"
    assert restored.constant_data["fixed_prior_contribution"].attrs["units"] == "ppm"


def test_public_co2_runner_derives_default_model_error_alignment(monkeypatch: Any) -> None:
    inputs = _golden_inputs()
    inputs["site_indicator"] = xr.DataArray(
        np.arange(inputs.sizes["nmeasure"]),
        dims="nmeasure",
        coords={"nmeasure": inputs["nmeasure"]},
    )

    class PreparedInputsStub:
        inv_inputs = inputs

        def validated(self) -> "PreparedInputsStub":
            return self

    materialized_names: list[tuple[str, ...]] = []

    def materialize(_prepared: Any, *, variable_names: tuple[str, ...]) -> xr.Dataset:
        materialized_names.append(variable_names)
        return inputs

    sampled_models: list[pm.Model] = []

    def sample_model(built: Any, _sampler: Any) -> az.InferenceData:
        sampled_models.append(built.model)
        return _empty_sampled_trace(inputs)

    monkeypatch.setattr(co2_runner, "materialize_pymc_inputs", materialize)
    monkeypatch.setattr(co2_runner, "sample_rhime_model", sample_model)

    run_rhime_co2(prepared_inputs=cast(Any, PreparedInputsStub()))

    assert "site_indicator" in materialized_names[0]
    np.testing.assert_array_equal(
        sampled_models[0]["sigma_site_index"].eval(),
        np.arange(inputs.sizes["nmeasure"]),
    )
    np.testing.assert_array_equal(
        sampled_models[0]["sigma_period_index"].eval(),
        np.zeros(inputs.sizes["nmeasure"]),
    )
    assert "sigma" in sampled_models[0].named_vars


def test_public_co2_runner_preserves_materialized_fixed_mismatch(monkeypatch: Any) -> None:
    inputs = _golden_inputs()
    inputs["fixed_model_mismatch"] = xr.full_like(inputs["mf"], 0.75)

    class PreparedInputsStub:
        inv_inputs = inputs

        def validated(self) -> "PreparedInputsStub":
            return self

    monkeypatch.setattr(co2_runner, "materialize_pymc_inputs", lambda *_args, **_kwargs: inputs)
    sampled_models: list[pm.Model] = []

    def sample_model(built: Any, _sampler: Any) -> az.InferenceData:
        sampled_models.append(built.model)
        return _empty_sampled_trace(inputs)

    monkeypatch.setattr(co2_runner, "sample_rhime_model", sample_model)

    run_rhime_co2(
        prepared_inputs=cast(Any, PreparedInputsStub()),
        no_model_error=True,
    )

    np.testing.assert_allclose(sampled_models[0]["fixed_model_mismatch"].eval(), 0.75)
