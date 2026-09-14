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
from scipy.stats import multivariate_normal
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.coords import get_coord_registry
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.models.site_sigma import add_site_sigma_gaussian_likelihood
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import (
    build_co2_model,
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
        },
        coords={"region": labels, "nmeasure": np.arange(operator.shape[0])},
    )
    for name in ("mf", "mf_error", "fixed_prior_contribution"):
        inputs[name].attrs["units"] = "ppm"
    return inputs


def _production_boundary_inputs() -> xr.Dataset:
    """Small VG-shaped inner/outer flux and monthly NESW boundary input."""
    inputs = _golden_inputs().assign_coords(
        basis_group=("region", ["inner", "outer", "inner", "outer"]),
        site=("nmeasure", ["MHD", "TAC"]),
        time=(
            "nmeasure",
            np.asarray(["2021-01-15", "2021-02-15"], dtype="datetime64[D]"),
        ),
    )
    bc_index = pd.MultiIndex.from_product(
        [["north", "east", "south", "west"], ["2021-01", "2021-02"]],
        names=("bc_curtain", "bc_period"),
    )
    bc_coords = xr.Coordinates.from_pandas_multiindex(bc_index, "bc_region")
    inputs["H_bc"] = xr.DataArray(
        np.arange(16, dtype=float).reshape(2, 8) / 20.0,
        dims=("nmeasure", "bc_region"),
        coords={"nmeasure": inputs["nmeasure"], **bc_coords},
        attrs={"units": "ppm"},
    )
    inputs["aggregation_error_covariance"][:] = [[0.08, 0.02], [0.02, 0.12]]
    return inputs


def _empty_sampled_trace(inputs: xr.Dataset) -> az.InferenceData:
    """Return a small trace with representative CO2 sample groups."""
    return az.from_dict(
        posterior={
            "flux_scaling": np.ones((1, 2, inputs.sizes["region"])),
            "co2_flux_contribution": np.ones((1, 2, inputs.sizes["nmeasure"])),
            "modelled_concentration": np.ones((1, 2, inputs.sizes["nmeasure"])),
            "epsilon": np.ones((1, 2, inputs.sizes["nmeasure"])),
        },
        posterior_predictive={"y": np.ones((1, 2, inputs.sizes["nmeasure"]))},
        constant_data={
            "error": inputs["mf_error"].values,
            "fixed_model_mismatch": np.ones(inputs.sizes["nmeasure"]),
            "fixed_prior_contribution": inputs["fixed_prior_contribution"].values,
            "co2_sensitivity": inputs["H"].values,
        },
        dims={
            "flux_scaling": ["region"],
            "co2_flux_contribution": ["nmeasure"],
            "modelled_concentration": ["nmeasure"],
            "epsilon": ["nmeasure"],
            "y": ["nmeasure"],
            "error": ["nmeasure"],
            "fixed_model_mismatch": ["nmeasure"],
            "fixed_prior_contribution": ["nmeasure"],
            "co2_sensitivity": ["nmeasure", "region"],
        },
    )


def _build_model(inputs: xr.Dataset, **kwargs: Any) -> pm.Model:
    retained_prior = CorrelatedLognormalPrior(
        inputs["alpha_prior_mean"],
        inputs["alpha_prior_covariance"],
        covariance_dim="region_cov",
    )
    return build_co2_model(
        inputs["H"],
        retained_prior=retained_prior,
        fixed_prior_contribution=inputs["fixed_prior_contribution"],
        observations=inputs["mf"],
        observation_error=inputs["mf_error"],
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        **kwargs,
    )


def test_co2_model_exposes_affine_correlated_dense_covariance_graph() -> None:
    model = _build_model(_golden_inputs(), fixed_model_mismatch=1.0)

    assert {
        "co2_sensitivity",
        "flux_scaling_latent",
        "flux_scaling",
        "co2_flux_contribution",
        "fixed_prior_contribution",
        "modelled_concentration",
        "error",
        "fixed_model_mismatch",
        "epsilon",
        "y",
    } <= set(model.named_vars)
    assert "outer_flux_contribution" not in model.named_vars
    assert model.named_vars_to_dims["flux_scaling"] == ("region",)
    assert model.named_vars_to_dims["modelled_concentration"] == ("nmeasure",)
    assert isinstance(model["y"].owner.op, pm.MvNormal.rv_op.__class__)


def test_co2_model_rejects_sigma_prior_without_alignment() -> None:
    with pytest.raises(ValueError, match="requires `sigma_alignment`"):
        _build_model(
            _golden_inputs(),
            sigma_prior={"pdf": "halfnormal", "sigma": 1.0},
        )


@pytest.mark.parametrize(
    "model_error_option",
    [
        {"sigma_alignment": cast(Any, object())},
        {"sigma_prior": {"pdf": "halfnormal", "sigma": 1.0}},
    ],
)
def test_public_co2_runner_rejects_model_error_options_when_disabled(
    model_error_option: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        run_rhime_co2(
            prepared_inputs=cast(Any, object()),
            no_model_error=True,
            **model_error_option,
        )


def test_public_co2_runner_rejects_default_model_error_with_selected_likelihood() -> None:
    with pytest.raises(ValueError, match="replaces the default model-error options"):
        run_rhime_co2(
            prepared_inputs=cast(Any, object()),
            likelihood_builder=add_site_sigma_gaussian_likelihood,
            no_model_error=True,
        )
    with pytest.raises(ValueError, match="likelihood_kwargs require likelihood_builder"):
        run_rhime_co2(
            prepared_inputs=cast(Any, object()),
            likelihood_kwargs={"fixed_site_amplitudes": {"MHD": 0.5}},
        )


@pytest.mark.parametrize(
    ("offset_args", "error", "message"),
    [
        ({"var_name": "custom_offset"}, ValueError, "Unsupported offset_args"),
        ({"output_name": "custom_output"}, ValueError, "Unsupported offset_args"),
        (
            {
                "offset_freq": "monthly",
                "offset_freq_indicator": np.asarray([0, 1]),
            },
            ValueError,
            "Specify only one",
        ),
        ({"per_site": 1}, TypeError, "must be booleans"),
    ],
)
def test_co2_offset_args_are_normalised_explicitly(
    offset_args: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    """Reject unsupported or internally inconsistent CO2 offset options."""
    with pytest.raises(error, match=message):
        co2_runner._normalise_offset_args(offset_args)


def test_build_co2_model_preserves_offset_args_compatibility() -> None:
    """Direct builder callers can still select a global offset via offset_args."""
    model = _build_model(
        _production_boundary_inputs(),
        offset_prior={"pdf": "normal", "mu": 0.2, "sigma": 0.1},
        offset_args={"per_site": False},
    )

    assert model["offset_latent"].ndim == 0
    assert "offset" in model.named_vars
    assert "offset_design" not in model.named_vars
    assert "site_indicator" not in model.named_vars


def test_co2_structural_zero_is_fixed_at_one_and_pruned_only_from_operator() -> None:
    inputs = _golden_inputs()
    inputs["H"][:, 1] = 0.0

    model = _build_model(inputs)
    full_state, active_state, forward = pm.draw(
        [
            model["flux_scaling"],
            model["flux_scaling_active"],
            model["co2_flux_contribution"],
        ],
        random_seed=42,
    )

    assert model.named_vars_to_dims["flux_scaling"] == ("region",)
    assert model.named_vars_to_dims["co2_sensitivity"] == ("nmeasure", "region_retained")
    assert full_state.shape == (inputs.sizes["region"],)
    assert active_state.shape == (inputs.sizes["region"] - 1,)
    assert full_state[1] == 1.0
    np.testing.assert_allclose(model["co2_sensitivity"].eval(), inputs["H"].values[:, [0, 2, 3]])
    np.testing.assert_allclose(
        forward,
        inputs["fixed_prior_contribution"].values + inputs["H"].values @ full_state,
    )


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
        [
            model["flux_scaling"],
            model["flux_scaling_active"],
            model["co2_flux_contribution"],
        ],
        random_seed=42,
    )
    registry = get_coord_registry(model)

    assert registry is not None
    assert full_state.shape == (4,)
    assert active_state.shape == (2,)
    np.testing.assert_array_equal(full_state[~is_active.values], fixed_value.values[~is_active.values])
    np.testing.assert_allclose(
        forward,
        inputs["fixed_prior_contribution"].values + inputs["H"].values @ full_state,
    )
    assert registry.original_coords["region"].equals(state_index)
    assert registry.original_coords["region_flux_scaling_active"].tolist() == [
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
    builder_kwargs: dict[str, Any] = {}
    original_builder = co2_runner.build_co2_model

    def build_model(flux_sensitivity: xr.DataArray, **kwargs: Any) -> pm.Model:
        builder_kwargs.update(kwargs)
        return original_builder(flux_sensitivity, **kwargs)

    monkeypatch.setattr(co2_runner, "build_co2_model", build_model)
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

    assert "sigma" not in sampled_models[0].named_vars
    assert builder_kwargs["sigma_alignment"] is None
    assert builder_kwargs["sigma_prior"] is None
    assert isinstance(builder_kwargs["retained_prior"], CorrelatedLognormalPrior)
    expected_prior = CorrelatedLognormalPrior(
        inputs["alpha_prior_mean"],
        inputs["alpha_prior_covariance"],
        covariance_dim="region_cov",
    )
    xr.testing.assert_equal(
        builder_kwargs["retained_prior"].mean,
        expected_prior.mean,
    )
    xr.testing.assert_equal(
        builder_kwargs["retained_prior"].arithmetic_covariance,
        expected_prior.arithmetic_covariance,
    )
    np.testing.assert_allclose(sampled_models[0]["fixed_model_mismatch"].eval(), 1.0)
    roles = json.loads(result.attrs["rhime_variable_roles"])
    metadata = json.loads(result.attrs["rhime_model_metadata"])
    assert roles["coherent_prior_contribution"] == "fixed_prior_contribution"
    assert roles["emissions_sensitivity"] == "co2_sensitivity"
    assert roles["flux_scale"] == "flux_scaling"
    assert roles["model_mean"] == "modelled_concentration"
    assert roles["pollution_concentration"] == "co2_flux_contribution"
    assert roles["observation_error"] == "error"
    trace_variables = {
        name
        for group_name in result.groups()
        for name in getattr(result, group_name).data_vars
    }
    assert set(roles.values()) <= trace_variables
    assert metadata["recipe"] == "co2"
    assert metadata["basis_artifact_source"] == "unknown"
    assert result.posterior["flux_scaling"].attrs["units"] == "1"
    assert result.posterior["modelled_concentration"].attrs["units"] == "ppm"
    assert result.posterior_predictive["y"].attrs["units"] == "ppm"
    assert json.loads(result.constant_data["fixed_model_mismatch"].attrs["rhime_scientific_roles"]) == [
        "fixed_model_mismatch"
    ]

    path = tmp_path / "co2-trace.nc"
    save_inferencedata(result, path)
    restored = load_inferencedata(path)
    assert json.loads(restored.attrs["rhime_model_metadata"])["recipe"] == "co2"
    assert restored.posterior["flux_scaling"].attrs["units"] == "1"
    assert restored.constant_data["fixed_prior_contribution"].attrs["units"] == "ppm"


def test_public_co2_runner_derives_default_model_error_alignment(monkeypatch: Any) -> None:
    inputs = _golden_inputs()
    inputs = inputs.assign_coords(
        site=("nmeasure", [f"site-{index}" for index in range(inputs.sizes["nmeasure"])])
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

    assert "site_indicator" not in materialized_names[0]
    np.testing.assert_array_equal(
        sampled_models[0]["sigma_site_index"].eval(),
        np.arange(inputs.sizes["nmeasure"]),
    )
    np.testing.assert_array_equal(
        sampled_models[0]["sigma_period_index"].eval(),
        np.zeros(inputs.sizes["nmeasure"]),
    )
    assert "sigma" in sampled_models[0].named_vars


def test_public_co2_runner_selects_boundary_and_offset_once(monkeypatch: Any) -> None:
    """The public runner materializes and composes each selected baseline once."""
    inputs = _production_boundary_inputs()

    class PreparedInputsStub:
        inv_inputs = inputs

        def validated(self) -> "PreparedInputsStub":
            return self

    materialized_names: list[tuple[str, ...]] = []
    built_models: list[pm.Model] = []

    def materialize(_prepared: Any, *, variable_names: tuple[str, ...]) -> xr.Dataset:
        materialized_names.append(variable_names)
        return inputs

    def sample_model(built: Any, _sampler: Any) -> az.InferenceData:
        built_models.append(built.model)
        return _empty_sampled_trace(inputs)

    monkeypatch.setattr(co2_runner, "materialize_pymc_inputs", materialize)
    monkeypatch.setattr(co2_runner, "sample_rhime_model", sample_model)
    bc_active = xr.DataArray(
        [True, False, True, False, True, False, True, False],
        dims="bc_region",
        coords={"bc_region": inputs["bc_region"]},
    )
    bc_fixed = xr.DataArray(
        np.linspace(0.8, 1.5, 8),
        dims="bc_region",
        coords={"bc_region": inputs["bc_region"]},
    )

    run_rhime_co2(
        prepared_inputs=cast(Any, PreparedInputsStub()),
        fixed_model_mismatch=1.0,
        no_model_error=True,
        use_bc=True,
        bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
        bc_state_activity=StateActivity(active=bc_active, fixed_value=bc_fixed),
        offset_prior={"pdf": "normal", "mu": 0.2, "sigma": 0.1},
        offset_args={"per_site": False},
    )

    model = built_models[0]
    assert len(materialized_names) == 1
    assert materialized_names[0].count("H_bc") == 1
    assert {"hbc", "bc", "mu_bc", "offset", "offset_latent"} <= set(model.named_vars)
    registry = get_coord_registry(model)
    assert registry is not None
    assert registry.original_coords["bc_region"].equals(inputs.indexes["bc_region"])
    np.testing.assert_array_equal(model["bc_is_active"].eval(), bc_active)

    variables = [
        model[name]
        for name in (
            "modelled_concentration",
            "fixed_prior_contribution",
            "co2_flux_contribution",
            "mu_bc",
            "offset",
            "flux_scaling",
        )
    ]
    values = model.compile_fn(
        model.replace_rvs_by_values(variables),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )(model.initial_point())
    total, fixed, flux, boundary, offset, flux_scaling = map(np.asarray, values)
    np.testing.assert_allclose(total, flux + boundary + offset)
    np.testing.assert_allclose(
        flux,
        fixed + inputs["H"].values @ flux_scaling,
    )


def test_public_co2_runner_does_not_auto_select_prepared_baseline(monkeypatch: Any) -> None:
    """A prepared boundary remains unused unless the runner explicitly selects it."""
    inputs = _production_boundary_inputs()

    class PreparedInputsStub:
        inv_inputs = inputs

        def validated(self) -> "PreparedInputsStub":
            return self

    selected: list[tuple[str, ...]] = []
    built_models: list[pm.Model] = []

    def materialize(_prepared: Any, *, variable_names: tuple[str, ...]) -> xr.Dataset:
        selected.append(variable_names)
        return inputs

    def sample_model(built: Any, _sampler: Any) -> az.InferenceData:
        built_models.append(built.model)
        return _empty_sampled_trace(inputs)

    monkeypatch.setattr(co2_runner, "materialize_pymc_inputs", materialize)
    monkeypatch.setattr(co2_runner, "sample_rhime_model", sample_model)
    run_rhime_co2(
        prepared_inputs=cast(Any, PreparedInputsStub()),
        fixed_model_mismatch=1.0,
        no_model_error=True,
    )

    assert "H_bc" not in selected[0]
    assert {"hbc", "bc", "mu_bc", "offset"}.isdisjoint(built_models[0].named_vars)


def _run_selected_co2_likelihood(
    monkeypatch: Any,
    likelihood_builder: Any,
    likelihood_kwargs: dict[str, Any],
) -> tuple[xr.Dataset, Any, dict[str, Any], az.InferenceData]:
    """Run one selected likelihood through the public CO2 route."""
    inputs = _golden_inputs().assign_coords(
        site=("nmeasure", ["MHD", "MHD"]),
        time=("nmeasure", np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")),
    )
    inputs["mf_error"] = xr.DataArray(
        [0.2, 0.3],
        dims="nmeasure",
        coords={"nmeasure": inputs["nmeasure"]},
        attrs={"units": "ppm"},
    )

    class PreparedInputsStub:
        inv_inputs = inputs

        def validated(self) -> "PreparedInputsStub":
            return self

    monkeypatch.setattr(co2_runner, "materialize_pymc_inputs", lambda *_args, **_kwargs: inputs)
    built_results: list[Any] = []

    def sample_model(built: Any, _sampler: Any) -> az.InferenceData:
        built_results.append(built)
        return _empty_sampled_trace(inputs)

    received: dict[str, Any] = {}

    def recording_likelihood(**kwargs: Any) -> Any:
        received.update(kwargs)
        return likelihood_builder(**kwargs)

    monkeypatch.setattr(co2_runner, "sample_rhime_model", sample_model)
    result = run_rhime_co2(
        prepared_inputs=cast(Any, PreparedInputsStub()),
        likelihood_builder=recording_likelihood,
        likelihood_kwargs=likelihood_kwargs,
    )
    return inputs, built_results[0], received, result


def _initial_mean_and_observed_logp(model: pm.Model) -> tuple[np.ndarray, float]:
    """Evaluate the completed mean and likelihood at the model initial point."""
    point = model.initial_point()
    mean_function = model.compile_fn(
        model.replace_rvs_by_values([model["modelled_concentration"]])[0],
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    mean = np.asarray(mean_function(point))
    logp = float(model.compile_logp(vars=model.observed_RVs)(point))
    return mean, logp


def test_public_co2_runner_selects_fixed_ou_likelihood(monkeypatch: Any) -> None:
    """Fixed OU keeps its labelled temporal covariance equation."""
    likelihood_kwargs = {
        "tau_hours": 5.0,
        "fixed_site_amplitudes": {"MHD": 0.5},
    }
    inputs, built, received, result = _run_selected_co2_likelihood(
        monkeypatch,
        add_fixed_ou_gaussian_likelihood,
        likelihood_kwargs,
    )
    model = built.model
    assert "min_error" not in inputs
    assert received["mean"] is model["modelled_concentration"]
    aggregation_error = received["aggregation_error"]
    assert aggregation_error.mode == "dense"
    assert aggregation_error.covariance is not None
    np.testing.assert_allclose(
        aggregation_error.covariance.values,
        inputs["aggregation_error_covariance"].values,
    )
    assert aggregation_error.covariance.values[0, 1] != 0.0
    assert {"modelled_concentration", "y", "epsilon", "ou_site_amplitude"} <= set(
        model.named_vars
    )
    assert set(built.variable_roles.values()) <= set(model.named_vars)
    assert "sigma" not in model.named_vars

    mean, actual_logp = _initial_mean_and_observed_logp(model)
    lag_hours = np.abs(
        (inputs["time"].values[:, None] - inputs["time"].values[None, :])
        / np.timedelta64(1, "h")
    )
    ou_covariance = 0.5**2 * np.exp(-lag_hours / 5.0)
    expected_logp = multivariate_normal.logpdf(
        inputs["mf"].values,
        mean=mean,
        cov=(
            inputs["aggregation_error_covariance"].values
            + np.diag(inputs["mf_error"].values ** 2)
            + ou_covariance
        ),
    )
    assert actual_logp == pytest.approx(expected_logp, rel=1.0e-10)
    assert json.loads(result.attrs["rhime_likelihood_kwargs"]) == likelihood_kwargs
    assert json.loads(result.attrs["rhime_likelihood_builder"])["qualname"].endswith(
        "recording_likelihood"
    )


def test_public_co2_runner_selects_iid_site_sigma_likelihood(monkeypatch: Any) -> None:
    """IID site sigma keeps its labelled diagonal mismatch equation."""
    likelihood_kwargs = {"fixed_site_amplitudes": {"MHD": 0.5}}
    inputs, built, received, result = _run_selected_co2_likelihood(
        monkeypatch,
        add_site_sigma_gaussian_likelihood,
        likelihood_kwargs,
    )
    model = built.model
    assert "min_error" not in inputs
    assert received["mean"] is model["modelled_concentration"]
    aggregation_error = received["aggregation_error"]
    assert aggregation_error.mode == "dense"
    assert aggregation_error.covariance is not None
    np.testing.assert_allclose(
        aggregation_error.covariance.values,
        inputs["aggregation_error_covariance"].values,
    )
    assert aggregation_error.covariance.values[0, 1] != 0.0
    assert {"modelled_concentration", "y", "epsilon", "sigma_site"} <= set(model.named_vars)
    assert set(built.variable_roles.values()) <= set(model.named_vars)
    assert "sigma" not in model.named_vars

    mean, actual_logp = _initial_mean_and_observed_logp(model)
    iid_covariance = np.diag(inputs["mf_error"].values ** 2 + 0.5**2)
    expected_logp = multivariate_normal.logpdf(
        inputs["mf"].values,
        mean=mean,
        cov=inputs["aggregation_error_covariance"].values + iid_covariance,
    )
    assert actual_logp == pytest.approx(expected_logp, rel=1.0e-10)
    assert json.loads(result.attrs["rhime_likelihood_kwargs"]) == likelihood_kwargs
    assert json.loads(result.attrs["rhime_likelihood_builder"])["qualname"].endswith(
        "recording_likelihood"
    )


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
