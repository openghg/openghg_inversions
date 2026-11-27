"""Synthetic numerical oracles for the experimental Gaussian collapse."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import approx_fprime
from scipy.stats import multivariate_normal, norm

from openghg_inversions.experimental.gaussian_collapse import (
    GaussianCollapse,
    PersistentPairOU,
    fit_log_density,
    laplace_importance,
    marginal_posterior,
)


def test_persistent_ou_matches_stationary_state_model():
    """Check irregular row order, two sites, and cross-gas lags."""
    hours = [4, 0, 1.5, 1.5, 0, 4]
    site = ["B", "A", "A", "A", "A", "B"]
    species = ["o2", "co2", "co2", "o2", "o2", "co2"]
    rho, tau = -0.8, 2.7
    amplitude = np.array([1.2, 1.4, 2.0, 3.0])
    covariance = PersistentPairOU(hours, site, species)
    fitted = covariance(np.r_[np.log(amplitude), np.arctanh(rho), np.log(tau)])

    expected = np.zeros((6, 6))
    stationary = np.array([[1.2**2, rho * 1.2 * 2], [rho * 1.2 * 2, 2.0**2]])
    transition = np.exp(-1.5 / tau) * np.eye(2)
    joint = np.block([[stationary, stationary @ transition.T], [transition @ stationary, stationary]])
    rows = [1, 4, 2, 3]
    expected[np.ix_(rows, rows)] = joint
    expected[np.ix_([0, 5], [0, 5])] = [[3**2, rho * 3 * 1.4], [rho * 3 * 1.4, 1.4**2]]

    np.testing.assert_allclose(fitted, expected)
    fixed = PersistentPairOU(hours, site, species, rho=rho, timescale=tau)
    np.testing.assert_allclose(fixed(np.log(amplitude)), expected)


def test_collapse_matches_joint_normal_oracle():
    """Check evidence, mean and covariance against a joint normal density."""
    prior = np.array([[1.6, 0.27], [0.27, 0.9]])
    projection = np.array([[1.0, -1.0], [0.8, 0.2]])
    observation_operator = np.array([[1.5, 0.1], [0.9, -0.3], [0.2, 0.6]])
    prior_mean = np.array([0.4, 0.1])
    fixed = observation_operator @ prior @ observation_operator.T + np.diag([0.3, 0.5, 0.7])
    cross = projection @ prior @ observation_operator.T
    projected_prior = projection @ prior @ projection.T
    observation_mean = observation_operator @ prior_mean
    observations = np.array([2.2, 0.5, -0.1])
    added = np.array([[0.2, 0.03, -0.05], [0.03, 0.35, 0.06], [-0.05, 0.06, 0.4]])

    model = GaussianCollapse(
        observations - observation_mean, fixed, projection @ prior_mean, projected_prior, cross
    )
    np.testing.assert_allclose(
        model.log_evidence(added), multivariate_normal.logpdf(observations, observation_mean, fixed + added)
    )
    mean, covariance = model.posterior(added)
    joint = multivariate_normal(
        mean=np.r_[projection @ prior_mean, observation_mean],
        cov=np.block([[projected_prior, cross], [cross.T, fixed + added]]),
    )
    conditional = multivariate_normal(mean, covariance)
    for projected in ([0.1, -0.5], [2.0, 0.7], [-1.1, 0.0]):
        np.testing.assert_allclose(
            conditional.logpdf(projected),
            joint.logpdf(np.r_[projected, observations]) - model.log_evidence(added),
        )


def test_type_ii_fit_matches_closed_form_variance():
    """Check that marginal likelihood estimates the remaining observation variance."""
    innovation = np.array([2.0, -3.0, 1.5])
    model = GaussianCollapse(innovation, np.eye(3), [0.0], [[1.0]], [[0.0, 0.0, 0.0]])
    result = fit_log_density(
        lambda theta: model.log_evidence(np.exp(theta[0]) * np.eye(3)), [0.0], [(-10.0, 10.0)]
    )
    assert result.success
    np.testing.assert_allclose(np.exp(result.x[0]), np.mean(innovation**2) - 1, rtol=2e-5)


def test_laplace_importance_mixture_matches_quadrature():
    """Check corrected posterior moments against independent scalar integration."""
    observed = 2.5
    model = GaussianCollapse([observed], [[1.0]], [0.0], [[1.0]], [[1.0]])

    def log_density(theta):
        """Return the collapsed log density with a bounded transformed prior."""
        if abs(theta[0]) > 7:
            return -np.inf
        return model.log_evidence([[np.exp(theta[0])]]) + norm.logpdf(theta[0])

    result = fit_log_density(log_density, [0.0], [(-7.0, 7.0)])
    assert result.success
    samples = laplace_importance(log_density, result.x, 6000, np.random.default_rng(321))
    mean, covariance = marginal_posterior(lambda theta: model.posterior([[np.exp(theta[0])]]), samples)

    def integrand(t, power):
        """Integrate one unnormalized conditional moment over scalar variance."""
        variance = 1 + np.exp(t)
        likelihood_prior = norm.pdf(observed, scale=np.sqrt(variance)) * norm.pdf(t)
        conditional_mean = observed / variance
        moment = [1, conditional_mean, (1 - 1 / variance) + conditional_mean**2][power]
        return likelihood_prior * moment

    total, first, second = [quad(integrand, -7, 7, args=(power,))[0] for power in range(3)]
    assert samples.effective_sample_size > 3000
    assert samples.weights.sum() == pytest.approx(1)
    np.testing.assert_allclose(mean, first / total, atol=0.015)
    np.testing.assert_allclose(covariance, second / total - (first / total) ** 2, atol=0.015)


def test_laplace_requires_positive_curvature():
    """Reject a stationary point when it cannot define a Laplace scale."""
    with pytest.raises(ValueError, match="positive definite curvature"):
        laplace_importance(lambda theta: theta[0] ** 2, [0.0], 10, np.random.default_rng(0))


@pytest.mark.parametrize("fixed_rho, fixed_tau", [(False, False), (True, False), (False, True), (True, True)])
def test_persistent_ou_score_and_prior_match_finite_differences(fixed_rho, fixed_tau):
    """Verify exact scores in each transformed parameter layout."""
    covariance = PersistentPairOU(
        [0.1, 2.2, 4.8, 6.0, 3.1, 7.0],
        ["A", "B", "A", "B", "A", "B"],
        ["co2", "o2", "o2", "co2", "o2", "o2"],
        rho=-0.4 if fixed_rho else None,
        timescale=1.8 if fixed_tau else None,
    )
    assert covariance.sites == ("A", "B")
    model = GaussianCollapse(np.arange(0.5, 3.5, 0.5), np.eye(6), [0.0], [[1.0]], np.zeros((1, 6)))
    theta = np.array([0.1, -0.4, 0.3, 0.8] + ([] if fixed_rho else [-0.42]) + ([] if fixed_tau else [0.7]))

    def log_density_and_gradient(theta):
        """Combine exact evidence and prior with their scores."""
        evidence, score = model.log_evidence_and_gradient(
            covariance(theta), lambda derivative: covariance.contract_gradient(theta, derivative)
        )
        prior, prior_score = covariance.reference_log_prior_and_gradient(theta)
        return evidence + prior, score + prior_score

    exact = log_density_and_gradient(theta)[1]
    finite_difference = approx_fprime(theta, lambda params: log_density_and_gradient(params)[0], 1e-7)
    np.testing.assert_allclose(exact, finite_difference, atol=2e-7)


def test_persistent_ou_hyperparameters_end_to_end():
    """Fit a persistent OU MAP and integrate projected flux moments."""
    hours = np.tile([0, 1, 3, 6, 9, 13], 2)
    species = np.repeat(["co2", "o2"], 6)
    covariance = PersistentPairOU(hours, np.repeat("A", 12), species)
    rng = np.random.default_rng(828)
    cross = np.full((1, 12), 0.15)
    fixed = np.eye(12) + cross.T @ cross
    theta_truth = [np.log(2.5), np.log(3), np.arctanh(-0.6), np.log(5)]
    observations = rng.multivariate_normal(np.zeros(12), fixed + covariance(theta_truth))
    model = GaussianCollapse(observations, fixed, [0.0], [[1.0]], cross)
    bounds = [(-4, 4), (-4, 4), (-3.5, 3.5), (-2, 6)]

    def log_density(theta):
        """Combine evidence and prior without computing gradients."""
        if any(not lo <= value <= hi for value, (lo, hi) in zip(theta, bounds, strict=True)):
            return -np.inf
        return model.log_evidence(covariance(theta)) + covariance.reference_log_prior_and_gradient(theta)[0]

    def log_density_and_gradient(theta):
        """Combine exact evidence and prior with their scores."""
        evidence, gradient = model.log_evidence_and_gradient(
            covariance(theta), lambda derivative: covariance.contract_gradient(theta, derivative)
        )
        prior, prior_gradient = covariance.reference_log_prior_and_gradient(theta)
        return evidence + prior, gradient + prior_gradient

    map_fit = fit_log_density(
        log_density, [1.0, 1.0, 0.0, 1.5], bounds, log_density_and_gradient=log_density_and_gradient
    )
    assert map_fit.success
    samples = laplace_importance(
        log_density, map_fit.x, 512, rng, log_density_and_gradient=log_density_and_gradient
    )
    mean, variance = marginal_posterior(lambda theta: model.posterior(covariance(theta)), samples)
    assert samples.effective_sample_size > 150
    assert np.isfinite(mean).all()
    assert 0 < variance.item() <= 1.05
