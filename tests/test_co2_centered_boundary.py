"""Scientific invariants of additive, centred boundary corrections."""

from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
from scipy.stats import multivariate_normal
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import build_co2_cached_sigma_model, build_co2_model


def _inputs(n_boundary: int = 3) -> xr.Dataset:
    response = np.array([[0.2, 0.3, 0.1], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.3, 0.0, 0.5]])
    response = response[:, :n_boundary]
    boundary = response * np.array([400.0, 410.0, 390.0])[:n_boundary]
    inputs = xr.Dataset(
        {
            "H": (("nmeasure", "region"), np.ones((4, 1))),
            "alpha_prior_mean": ("region", [1.0]),
            "alpha_prior_covariance": (("region", "region_cov"), [[0.04]]),
            "fixed_prior_contribution": ("nmeasure", np.full(4, 0.1)),
            "mf": ("nmeasure", boundary.sum(axis=1) + np.array([1.15, 1.06, 1.12, 1.07])),
            "mf_error": ("nmeasure", np.full(4, 0.1)),
            "aggregation_error_covariance": (("nmeasure", "nmeasure_cov"), np.eye(4) * 0.02),
            "H_bc": (("nmeasure", "bc_region"), boundary),
            "bc_correction_sensitivity": (("nmeasure", "bc_region"), response),
            "bc_centering_weights": ("bc_region", np.array([1.0, 2.0, 4.0])[:n_boundary]),
        },
        coords={
            "nmeasure": np.arange(4),
            "region": ["flux"],
            "bc_region": ["north", "east", "west"][:n_boundary],
            "site": ("nmeasure", ["AAA", "AAA", "BBB", "BBB"]),
            "time": (
                "nmeasure",
                np.array(
                    ["2021-01-01T00", "2021-01-01T03", "2021-01-01T01", "2021-01-01T05"],
                    dtype="datetime64[h]",
                ),
            ),
        },
    )
    for name in ("mf", "mf_error", "H_bc", "fixed_prior_contribution"):
        inputs[name].attrs["units"] = "ppm"
    for name in ("bc_correction_sensitivity", "bc_centering_weights"):
        inputs[name].attrs["units"] = "1"
    return inputs


def _kwargs(inputs: xr.Dataset) -> dict[str, Any]:
    return dict(
        retained_prior=CorrelatedLognormalPrior(
            inputs.alpha_prior_mean, inputs.alpha_prior_covariance, covariance_dim="region_cov"
        ),
        fixed_prior_contribution=inputs.fixed_prior_contribution,
        observations=inputs.mf,
        observation_error=inputs.mf_error,
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        state_activity=StateActivity(active=np.zeros(1, dtype=bool), fixed_value=1.0),
        boundary_sensitivity=inputs.H_bc,
        boundary_correction_sensitivity=inputs.bc_correction_sensitivity,
        bc_centering_weights=inputs.bc_centering_weights,
        bc_mean_shift_prior={"pdf": "normal", "mu": 0.0, "sigma": 0.5},
        bc_anomaly_scale=0.3,
    )


def _anomaly_rv(model: pm.Model) -> Any:
    return next(rv for rv in model.free_RVs if rv.name != "bc_mean_shift" and "sigma" not in rv.name)


def _anomaly_design(model: pm.Model) -> np.ndarray:
    anomaly = model.replace_rvs_by_values([model["bc_anomaly"]])[0]
    coefficients = model.rvs_to_values[_anomaly_rv(model)]
    jacobian = pt.jacobian(anomaly, coefficients)
    evaluate = model.compile_fn(jacobian, inputs=model.value_vars, on_unused_input="ignore")
    return np.asarray(evaluate(model.initial_point()))


def test_weighted_anomaly_prior_preserves_contrasts_and_permutation() -> None:
    inputs = _inputs()
    model = build_co2_model(inputs.H, **_kwargs(inputs))
    assert "bc" not in model.named_vars
    assert len(model.free_RVs) == 2
    design = _anomaly_design(model)
    latent = _anomaly_rv(model)
    point = model.initial_point()
    point[model.rvs_to_values[latent].name] = np.array([0.2, -0.4])
    prior_logp = model.compile_fn(
        model.logp(vars=[latent]), inputs=model.value_vars, on_unused_input="ignore"
    )
    assert prior_logp(point) == pytest.approx(
        multivariate_normal.logpdf([0.2, -0.4], mean=np.zeros(2), cov=np.eye(2) * 0.3**2)
    )
    weights = inputs.bc_centering_weights.values / inputs.bc_centering_weights.sum().item()
    projection = np.eye(3) - np.ones((3, 1)) * weights
    np.testing.assert_allclose(weights @ design, 0.0, atol=1e-15)
    np.testing.assert_allclose(design @ design.T, projection @ projection.T, atol=1e-14)
    contrasts = design[[0, 0, 1]] - design[[1, 2, 2]]
    np.testing.assert_allclose(np.sum(contrasts**2, axis=1), 2.0)

    order = np.array([2, 0, 1])
    permuted = inputs.isel(bc_region=order)
    other = build_co2_model(permuted.H, **_kwargs(permuted))
    other_design = _anomaly_design(other)
    np.testing.assert_allclose(
        other_design @ other_design.T, (design @ design.T)[np.ix_(order, order)], atol=1e-14
    )


@pytest.mark.parametrize("n_boundary", [1, 3])
def test_forward_equations_and_cached_likelihood_gradient(n_boundary: int) -> None:
    inputs = _inputs(n_boundary)
    ordinary = build_co2_model(inputs.H, **_kwargs(inputs))
    cached = build_co2_cached_sigma_model(
        inputs.H,
        **_kwargs(inputs),
        tau_hours={"AAA": 3.0, "BBB": 7.0},
        site_amplitude_prior_scale=0.75,
        initial_site_amplitudes=0.4,
    )
    covariance = np.eye(4) * 0.03
    for rows, lag, tau in (([0, 1], 3.0, 3.0), ([2, 3], 4.0, 7.0)):
        covariance[np.ix_(rows, rows)] += 0.16 * np.array(
            [[1.0, np.exp(-lag / tau)], [np.exp(-lag / tau), 1.0]]
        )
    predictions = []
    for model in (ordinary, cached.model):
        assert "bc" not in model.named_vars
        if n_boundary == 1:
            assert not any("anomaly" in rv.name for rv in model.free_RVs)
        point = model.initial_point()
        point["bc_mean_shift"] = np.asarray(0.7)
        if n_boundary > 1:
            rv = _anomaly_rv(model)
            point[model.rvs_to_values[rv].name] = np.array([0.2, -0.4])
        outputs = model.replace_rvs_by_values(
            [
                model[name]
                for name in (
                    "bc_anomaly",
                    "bc_correction",
                    "mu_bc_reference",
                    "mu_bc_mean_shift",
                    "mu_bc_anomaly",
                    "mu_bc",
                    "modelled_concentration",
                )
            ]
        )
        if model is cached.model:
            likelihood = model.replace_rvs_by_values([model["cached_fixed_ou_likelihood"]])[0]
            state_values = [model.rvs_to_values[model["bc_mean_shift"]]]
            if n_boundary > 1:
                state_values.append(model.rvs_to_values[_anomaly_rv(model)])
            outputs.extend([likelihood, *pt.grad(likelihood, state_values)])
        evaluate = model.compile_fn(outputs, inputs=model.value_vars, on_unused_input="ignore")
        values = evaluate(point)
        anomaly, correction, reference, shift, variation, boundary, mean = values[:7]
        response = inputs.bc_correction_sensitivity.values
        weights = inputs.bc_centering_weights.values / inputs.bc_centering_weights.sum().item()
        np.testing.assert_allclose(weights @ anomaly, 0.0, atol=1e-14)
        np.testing.assert_allclose(weights @ correction, 0.7)
        np.testing.assert_allclose(reference, inputs.H_bc.sum("bc_region"))
        np.testing.assert_allclose(shift, response.sum(axis=1) * 0.7)
        np.testing.assert_allclose(variation, response @ anomaly)
        np.testing.assert_allclose(boundary, reference + response @ correction)
        np.testing.assert_allclose(mean, boundary + 1.1)
        predictions.append(mean)
        if model is cached.model:
            np.testing.assert_allclose(
                values[7], multivariate_normal.logpdf(inputs.mf, mean=mean, cov=covariance)
            )
            residual_precision = np.linalg.solve(covariance, inputs.mf.values - mean)
            np.testing.assert_allclose(
                values[8], response.sum(axis=1) @ residual_precision, rtol=1e-6, atol=1e-8
            )
            if n_boundary > 1:
                np.testing.assert_allclose(
                    values[9],
                    (response @ _anomaly_design(model)).T @ residual_precision,
                    rtol=1e-6,
                    atol=1e-8,
                )
    np.testing.assert_allclose(*predictions)


@pytest.mark.parametrize(
    "override",
    [
        {"bc_anomaly_scale": None},
        {"bc_anomaly_scale": 0.0},
        {"bc_anomaly_scale": np.nan},
        {"bc_centering_weights": None},
        {"boundary_correction_sensitivity": None},
        {"bc_prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.1}},
        {"bc_state_activity": StateActivity()},
    ],
)
def test_centered_boundary_rejects_incomplete_or_competing_options(override: dict[str, Any]) -> None:
    inputs = _inputs()
    kwargs = _kwargs(inputs)
    kwargs.update(override)
    with pytest.raises((ValueError, TypeError)):
        build_co2_model(inputs.H, **kwargs)


@pytest.mark.parametrize(
    "invalid",
    [
        "negative_weight",
        "zero_weight",
        "nan_weight",
        "negative_response",
        "zero_response",
        "dimensional_response",
        "misaligned_labels",
    ],
)
def test_centered_boundary_validates_scientific_inputs(invalid: str) -> None:
    inputs = _inputs()
    kwargs = _kwargs(inputs)
    if invalid.endswith("weight"):
        weights = inputs.bc_centering_weights.copy(deep=True)
        weights[0] = {"negative_weight": -1.0, "zero_weight": 0.0, "nan_weight": np.nan}[invalid]
        kwargs["bc_centering_weights"] = weights
    elif invalid == "misaligned_labels":
        kwargs["bc_centering_weights"] = inputs.bc_centering_weights.assign_coords(
            bc_region=["wrong", "east", "west"]
        )
    else:
        response = inputs.bc_correction_sensitivity.copy(deep=True)
        if invalid == "negative_response":
            response[0, 0] = -0.1
        elif invalid == "zero_response":
            response[:] = 0.0
        else:
            response.attrs["units"] = "ppm"
        kwargs["boundary_correction_sensitivity"] = response
    with pytest.raises(ValueError):
        build_co2_model(inputs.H, **kwargs)
