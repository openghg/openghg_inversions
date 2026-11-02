"""Equation checks for OPE-194, without posterior sampling or external data."""

import numpy as np
import pytest
import pytensor
from scipy.stats import multivariate_normal
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.coords import registered_model
from openghg_inversions.observation_error import resolve_aggregation_error
from prototypes.biosphere_controls import (
    add_biosphere_state,
    build_biosphere_model,
    likelihood_rotation,
    linear_gaussian_posterior,
    net_gross_coordinates,
    net_shape_matrix,
)


@pytest.fixture(autouse=True)
def double_precision_equations():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _case(cross_source=False):
    covariance = np.array([[0.09, 0.025, 0.0], [0.025, 0.16, 0.0], [0.0, 0.0, 0.04]])
    if cross_source:
        covariance[0, 2] = covariance[2, 0] = 0.01
    prior = CorrelatedLognormalPrior(
        xr.DataArray(np.ones(3), dims="source", coords={"source": ["GPP", "TER", "FF"]}),
        covariance,
    )
    inputs = xr.Dataset(
        {
            "H": (
                ("nmeasure", "source"),
                [[-3.0, 2.8, 0.2], [-1.0, 2.0, 0.7], [-4.0, 2.0, 0.3]],
            ),
            "mf": ("nmeasure", [0.2, 1.3, -1.5]),
            "mf_error": ("nmeasure", [0.3, 0.2, 0.25]),
            "fixed_prior_contribution": ("nmeasure", [0.1, -0.2, 0.05]),
        },
        coords={"nmeasure": np.arange(3), "source": ["GPP", "TER", "FF"]},
    )
    arguments = dict(
        retained_prior=prior,
        fixed_prior_contribution=inputs.fixed_prior_contribution,
        observations=inputs.mf,
        observation_error=inputs.mf_error,
        aggregation_error=resolve_aggregation_error(inputs, "none"),
        gross_flux_weights=(100.0, 90.0),
    )
    return inputs, prior, arguments


def test_exact_lognormal_coordinates_preserve_density_and_prediction():
    """The Jacobian is checked numerically, independently of its coded formula."""
    inputs, prior, arguments = _case(cross_source=True)
    baseline = build_biosphere_model(inputs.H, **arguments)
    rotation = likelihood_rotation(inputs.H, prior, np.diag(inputs.mf_error.values**2))
    rotated = build_biosphere_model(
        inputs.H, **arguments, parameterization="rotated-lognormal", rotation=rotation
    )
    net = build_biosphere_model(inputs.H, **arguments, parameterization="net-gross-lognormal")
    baseline_logp = baseline.compile_logp()
    rotated_logp = rotated.compile_logp()
    net_logp = net.compile_logp()
    for scales in (np.ones(3), np.array([1.2, 0.8, 1.1]), np.array([0.8, 1.4, 0.9])):
        z = np.linalg.solve(prior.latent_cholesky.values, np.log(scales) - prior.latent_mean.values)
        coordinate = net_gross_coordinates(prior, scales, pair=(0, 1), gross_flux_weights=(100.0, 90.0))

        # Differentiate the explicit forward change z -> net/gross coordinates.
        def forward(value):
            return net_gross_coordinates(
                prior,
                np.exp(prior.latent_mean.values + prior.latent_cholesky.values @ value),
                pair=(0, 1),
                gross_flux_weights=(100.0, 90.0),
            )

        offsets = np.eye(3) * 1e-5
        jacobian = np.column_stack([(forward(z + d) - forward(z - d)) / 2e-5 for d in offsets])
        expected = baseline_logp({"flux_scaling_latent": z}) - np.linalg.slogdet(jacobian)[1]
        np.testing.assert_allclose(net_logp({"net_gross_coordinates": coordinate}), expected, atol=2e-8)
        np.testing.assert_allclose(
            rotated_logp({"flux_scaling_latent": rotation.T @ z}),
            baseline_logp({"flux_scaling_latent": z}),
            atol=1e-10,
        )
        for model, name, value in (
            (baseline, "flux_scaling_latent", z),
            (rotated, "flux_scaling_latent", rotation.T @ z),
            (net, "net_gross_coordinates", coordinate),
        ):
            state, prediction = model.compile_fn(
                [model.flux_scaling, model.modelled_concentration],
                inputs=[model[name]],
                point_fn=False,
            )(value)
            np.testing.assert_allclose(state, scales, atol=1e-12)
            np.testing.assert_allclose(prediction, inputs.fixed_prior_contribution + inputs.H.values @ scales)


def test_gaussian_net_shape_retains_templates_and_ff_lognormal():
    inputs, prior, arguments = _case()
    model = build_biosphere_model(inputs.H, **arguments, parameterization="gaussian-net-shape")
    transform = net_shape_matrix(100.0, 90.0)
    pair_covariance = prior.arithmetic_covariance.values[:2, :2]
    control_covariance = transform @ pair_covariance @ transform.T
    z = np.array([0.4, -0.7, 0.5])
    expected_control = transform @ np.ones(2) + np.linalg.cholesky(control_covariance) @ z[:2]
    state, prediction = model.compile_fn(
        [model.flux_scaling, model.modelled_concentration],
        inputs=[model.flux_scaling_latent],
        point_fn=False,
    )(z)
    np.testing.assert_allclose(state[:2], np.linalg.solve(transform, expected_control))
    np.testing.assert_allclose(
        state[2],
        np.exp(prior.latent_mean.values[2] + prior.latent_cholesky.values[2, 2] * z[2]),
    )
    expected_prediction = inputs.fixed_prior_contribution + inputs.H.values[:, :2] @ np.linalg.solve(
        transform, expected_control
    )
    expected_prediction = expected_prediction + inputs.H.values[:, 2] * state[2]
    np.testing.assert_allclose(prediction, expected_prediction)
    # Two templates are retained even when their integrated net is held fixed.
    assert np.linalg.matrix_rank(inputs.H.values[:, :2] @ np.linalg.inv(transform)) == 2
    original = linear_gaussian_posterior(
        inputs.H.values[:, :2],
        inputs.mf.values - inputs.fixed_prior_contribution.values - inputs.H.values[:, 2],
        np.diag(inputs.mf_error.values**2),
        np.ones(2),
        pair_covariance,
    )
    transformed = linear_gaussian_posterior(
        inputs.H.values[:, :2] @ np.linalg.inv(transform),
        inputs.mf.values - inputs.fixed_prior_contribution.values - inputs.H.values[:, 2],
        np.diag(inputs.mf_error.values**2),
        transform @ np.ones(2),
        control_covariance,
    )
    np.testing.assert_allclose(transformed[0], transform @ original[0], atol=1e-11)
    np.testing.assert_allclose(transformed[1], transform @ original[1] @ transform.T, atol=1e-10)
    # Independent precision-form Gaussian oracle verifies the conditioning result.
    h = inputs.H.values[:, :2]
    precision = np.linalg.inv(pair_covariance) + h.T @ np.diag(1.0 / inputs.mf_error.values**2) @ h
    np.testing.assert_allclose(original[1], np.linalg.inv(precision), atol=1e-12)


def test_gaussian_cross_source_covariance_is_rejected():
    inputs, _, arguments = _case(cross_source=True)
    with pytest.raises(ValueError, match="prior independence"):
        build_biosphere_model(inputs.H, **arguments, parameterization="gaussian-net-shape")


def test_rotation_supports_arbitrary_state_size_and_rejects_scaling():
    prior = CorrelatedLognormalPrior(
        xr.DataArray(np.ones(5), dims="region", coords={"region": np.arange(5)}),
        np.eye(5) * 0.1,
    )
    rotation = np.linalg.qr(np.arange(25).reshape(5, 5) + np.eye(5))[0]
    with registered_model() as model:
        add_biosphere_state(prior, parameterization="rotated-lognormal", rotation=rotation)
    z = np.linspace(-0.4, 0.4, 5)
    actual = model.compile_logp()({"flux_scaling_latent": z})
    np.testing.assert_allclose(actual, multivariate_normal.logpdf(z, mean=np.zeros(5), cov=np.eye(5)))
    with registered_model(), pytest.raises(AssertionError):
        add_biosphere_state(prior, parameterization="rotated-lognormal", rotation=2.0 * rotation)
