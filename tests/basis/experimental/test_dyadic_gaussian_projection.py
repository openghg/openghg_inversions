"""Tests for the dense exact Bocquet Gaussian projection oracle."""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal, localcontext

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.gaussian_projection import (
    build_bocquet_projection,
    equation_45_objective,
    gaussian_projection_oracle,
    native_gaussian_posterior,
    projected_bayesian_information_gain,
    projected_bayesian_kl,
    projected_dfs,
    projected_fisher_aggregation_aware,
    projected_fisher_base_r,
    reduced_gaussian_posterior,
    restriction_for_prolongation,
)


@pytest.fixture
def dense_gaussian_problem() -> tuple[np.ndarray, ...]:
    """Build a deterministic correlated Gaussian projection problem.

    Returns:
        Observation design, dense native covariance, dense observation
        covariance, nonzero nonuniform native mean, and observations.
    """
    prior_factor = np.array(
        [
            [1.2, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.9, 0.0, 0.0, 0.0],
            [-0.2, 0.4, 1.1, 0.0, 0.0],
            [0.1, -0.3, 0.2, 0.8, 0.0],
            [0.4, 0.1, -0.2, 0.3, 0.7],
        ]
    )
    observation_factor = np.array(
        [
            [0.9, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [-0.1, 0.3, 0.7, 0.0],
            [0.25, -0.15, 0.2, 0.6],
        ]
    )
    B = prior_factor @ prior_factor.T + 0.25 * np.eye(5)
    R = observation_factor @ observation_factor.T + 0.35 * np.eye(4)
    H = np.array(
        [
            [0.8, -0.2, 0.4, 0.1, -0.3],
            [0.1, 0.7, -0.5, 0.6, 0.2],
            [-0.4, 0.3, 0.9, -0.2, 0.5],
            [0.5, 0.2, -0.1, 0.8, -0.6],
        ]
    )
    mu = np.array([1.4, -0.6, 2.1, 0.35, -1.2])
    y = np.array([1.7, -0.3, 2.4, 0.8])
    return H, B, R, mu, y


def _restrictions() -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint aggregation and overlapping restriction cases."""
    aggregation = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    overlapping = np.array(
        [
            [1.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.4, 1.0, 1.0],
        ]
    )
    return aggregation, overlapping


def _high_precision_diagonal_covariance_kl(scale: float, *, dimension: int) -> float:
    """Return the analytic covariance KL for ``H = scale I`` and unit covariances."""
    with localcontext() as context:
        context.prec = 80
        signal_ratio = Decimal(str(scale)) ** 2
        mode = (Decimal(1) + signal_ratio).ln() - signal_ratio / (Decimal(1) + signal_ratio)
        return float(Decimal(dimension) * mode / Decimal(2))


@pytest.mark.parametrize("restriction", _restrictions(), ids=("aggregation", "overlapping"))
def test_direct_reduced_posterior_matches_restricted_native_posterior(
    dense_gaussian_problem: tuple[np.ndarray, ...],
    restriction: np.ndarray,
) -> None:
    """Aggregation and overlapping summaries should have exact posterior parity."""
    H, B, R, mu, y = dense_gaussian_problem

    analysis = gaussian_projection_oracle(H, B, R, restriction, mu, y)

    np.testing.assert_allclose(
        analysis.reduced_posterior.mean,
        restriction @ analysis.native_posterior.mean,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        analysis.reduced_posterior.covariance,
        restriction @ analysis.native_posterior.covariance @ restriction.T,
        rtol=3e-12,
        atol=3e-12,
    )


@pytest.mark.parametrize("error_variance", [1e-8, 1e-12, 1e-16])
def test_posterior_covariances_remain_positive_under_strong_information(
    error_variance: float,
) -> None:
    """Native and reduced covariance factors should retain tiny valid variances."""
    dimension = 3
    identity = np.eye(dimension)
    observation_covariance = error_variance * identity
    prior_mean = np.array([0.2, -0.4, 0.7])
    observations = np.array([1.0, -1.5, 0.3])

    native = native_gaussian_posterior(
        identity,
        identity,
        observation_covariance,
        prior_mean,
        observations,
    )
    projection = build_bocquet_projection(
        identity,
        identity,
        observation_covariance,
        identity,
        prior_mean,
    )
    reduced = reduced_gaussian_posterior(projection, observations)
    expected = (error_variance / (1.0 + error_variance)) * identity

    np.testing.assert_allclose(native.covariance, expected, rtol=5e-15, atol=0.0)
    np.testing.assert_allclose(reduced.covariance, expected, rtol=5e-15, atol=0.0)
    assert np.linalg.eigvalsh(native.covariance).min() > 0.0
    assert np.linalg.eigvalsh(reduced.covariance).min() > 0.0


def test_reduced_likelihood_uses_nonzero_prior_mean_offset(
    dense_gaussian_problem: tuple[np.ndarray, ...],
) -> None:
    """Direct conditioning should retain ``H(mu - Lambda Gamma mu)``."""
    H, B, R, mu, y = dense_gaussian_problem
    restriction = _restrictions()[0]
    projection = build_bocquet_projection(H, B, R, restriction, mu)

    offset = H @ (mu - projection.conditional_prolongation @ projection.projected_prior_mean)
    np.testing.assert_allclose(projection.reduced_likelihood_offset, offset, rtol=1e-13, atol=1e-13)
    reduced_prior_prediction = (
        projection.reduced_likelihood_offset + projection.reduced_design @ projection.projected_prior_mean
    )
    np.testing.assert_allclose(reduced_prior_prediction, H @ mu, rtol=1e-13, atol=1e-13)
    assert not np.allclose(projection.reduced_design @ projection.projected_prior_mean, H @ mu)

    cross_covariance = projection.projected_prior_covariance @ projection.reduced_design.T
    expected_mean = projection.projected_prior_mean + cross_covariance @ np.linalg.solve(
        projection.innovation_covariance,
        y - reduced_prior_prediction,
    )
    direct = reduced_gaussian_posterior(projection, y)
    np.testing.assert_allclose(direct.mean, expected_mean, rtol=2e-12, atol=2e-12)


def test_prior_weighted_restriction_preserves_fixed_regional_prolongation(
    dense_gaussian_problem: tuple[np.ndarray, ...],
) -> None:
    """A fixed regional amplitude map should become the conditional prolongation."""
    H, B, R, mu, _ = dense_gaussian_problem
    prolongation = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    restriction = restriction_for_prolongation(B, prolongation)
    projection = build_bocquet_projection(H, B, R, restriction, mu)

    np.testing.assert_allclose(restriction @ prolongation, np.eye(2), atol=2e-12)
    np.testing.assert_allclose(
        projection.conditional_prolongation,
        prolongation,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(projection.reduced_design, H @ prolongation, atol=2e-12)
    assert not restriction.flags.writeable


def test_projection_decomposition_and_innovation_are_exact(
    dense_gaussian_problem: tuple[np.ndarray, ...],
) -> None:
    """The conditional prior decomposition should be PSD and preserve innovation."""
    H, B, R, mu, _ = dense_gaussian_problem
    restriction = _restrictions()[1]
    projection = build_bocquet_projection(H, B, R, restriction, mu)

    np.testing.assert_allclose(
        restriction @ projection.conditional_prolongation,
        np.eye(restriction.shape[0]),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        B,
        projection.conditional_prolongation
        @ projection.projected_prior_covariance
        @ projection.conditional_prolongation.T
        + projection.unresolved_covariance,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        restriction @ projection.unresolved_covariance,
        np.zeros((restriction.shape[0], B.shape[0])),
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.linalg.eigvalsh(projection.unresolved_covariance).min() >= -1e-12
    np.testing.assert_allclose(
        projection.effective_observation_covariance + projection.resolved_signal_covariance,
        R + H @ B @ H.T,
        rtol=2e-12,
        atol=2e-12,
    )


def test_projection_objectives_match_dense_formulas(
    dense_gaussian_problem: tuple[np.ndarray, ...],
) -> None:
    """DFS, both Fisher scores, Equation 45, and KL should match direct formulas."""
    H, B, R, mu, y = dense_gaussian_problem
    restriction = _restrictions()[1]
    projection = build_bocquet_projection(H, B, R, restriction, mu)
    posterior = reduced_gaussian_posterior(projection, y)
    signal = projection.resolved_signal_covariance

    expected_dfs = np.trace(np.linalg.solve(projection.innovation_covariance, signal))
    expected_base_fisher = np.trace(np.linalg.solve(R, signal))
    expected_aware_fisher = np.trace(np.linalg.solve(projection.effective_observation_covariance, signal))
    update = posterior.mean - projection.projected_prior_mean
    expected_equation_45 = update @ np.linalg.solve(
        projection.projected_prior_covariance,
        update,
    )
    sign_prior, logdet_prior = np.linalg.slogdet(projection.projected_prior_covariance)
    sign_posterior, logdet_posterior = np.linalg.slogdet(posterior.covariance)
    assert sign_prior == sign_posterior == 1.0
    expected_kl = 0.5 * (
        np.trace(np.linalg.solve(projection.projected_prior_covariance, posterior.covariance))
        + expected_equation_45
        - restriction.shape[0]
        + logdet_prior
        - logdet_posterior
    )

    assert projected_dfs(projection) == pytest.approx(expected_dfs, rel=2e-12, abs=2e-12)
    assert projected_fisher_base_r(projection) == pytest.approx(
        expected_base_fisher,
        rel=2e-12,
        abs=2e-12,
    )
    assert projected_fisher_aggregation_aware(projection) == pytest.approx(
        expected_aware_fisher,
        rel=2e-12,
        abs=2e-12,
    )
    assert equation_45_objective(projection, y) == pytest.approx(
        expected_equation_45,
        rel=2e-12,
        abs=2e-12,
    )
    assert projected_bayesian_kl(projection, y) == pytest.approx(expected_kl, rel=2e-12, abs=2e-12)
    assert projected_bayesian_information_gain(projection, y) == pytest.approx(expected_kl)


def test_projected_bayesian_kl_retains_weak_covariance_information() -> None:
    """Eigenwise KL should preserve weak covariance information down to tiny designs."""
    dimension = 2
    identity = np.eye(dimension)
    scales = np.array([1e-4, 1e-6, 1e-8, 1e-10])
    actual = []
    for scale in scales:
        projection = build_bocquet_projection(
            scale * identity,
            identity,
            identity,
            identity,
            np.zeros(dimension),
        )
        actual.append(projected_bayesian_kl(projection, np.zeros(dimension)))

    expected = np.array(
        [_high_precision_diagonal_covariance_kl(scale, dimension=dimension) for scale in scales]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0.0)
    assert np.all(np.asarray(actual) > 0.0)
    assert np.all(np.diff(actual) < 0.0)


def test_prior_predictive_expected_equation_45_equals_dfs() -> None:
    """Deterministic prior-predictive covariance directions should recover DFS exactly."""
    identity = np.eye(3)
    projection = build_bocquet_projection(
        identity,
        identity,
        np.diag([0.3, 0.7, 1.1]),
        identity,
        np.zeros(3),
    )
    predictive_factor = np.linalg.cholesky(projection.innovation_covariance)
    expected_equation_45 = sum(
        equation_45_objective(projection, predictive_factor[:, index]) for index in range(3)
    )

    assert expected_equation_45 == pytest.approx(projected_dfs(projection), rel=2e-14, abs=2e-14)


def test_result_arrays_are_read_only_and_detached_from_inputs(
    dense_gaussian_problem: tuple[np.ndarray, ...],
) -> None:
    """Returned arrays should be immutable copies rather than input views."""
    H, B, R, mu, y = dense_gaussian_problem
    restriction = _restrictions()[0]
    expected_restriction = restriction.copy()
    analysis = gaussian_projection_oracle(H, B, R, restriction, mu, y)
    restriction[0, 0] = 99.0

    np.testing.assert_array_equal(analysis.projection.restriction, expected_restriction)
    for result in (analysis.native_posterior, analysis.reduced_posterior):
        assert not result.mean.flags.writeable
        assert not result.covariance.flags.writeable
        assert not result.innovation.flags.writeable
        assert not result.innovation_covariance.flags.writeable
    for result_field in fields(analysis.projection):
        value = getattr(analysis.projection, result_field.name)
        assert isinstance(value, np.ndarray)
        assert not value.flags.writeable


@pytest.mark.parametrize(
    ("restriction", "message"),
    [
        (np.ones(5), "two-dimensional"),
        (np.ones((2, 4)), "5 columns"),
        (np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0]]), "full row rank"),
    ],
)
def test_projection_rejects_invalid_restriction_rank_and_dimensions(
    dense_gaussian_problem: tuple[np.ndarray, ...],
    restriction: np.ndarray,
    message: str,
) -> None:
    """Restrictions must be two-dimensional, compatible, and full row rank."""
    H, B, R, mu, _ = dense_gaussian_problem
    with pytest.raises(ValueError, match=message):
        build_bocquet_projection(H, B, R, restriction, mu)


def test_oracle_rejects_invalid_model_dimensions_and_covariances(
    dense_gaussian_problem: tuple[np.ndarray, ...],
) -> None:
    """The oracle should reject incompatible, asymmetric, and non-SPD model inputs."""
    H, B, R, mu, y = dense_gaussian_problem
    restriction = _restrictions()[0]

    with pytest.raises(ValueError, match="H must be two-dimensional"):
        gaussian_projection_oracle(H[0], B, R, restriction, mu, y)
    with pytest.raises(ValueError, match="mu must have shape"):
        gaussian_projection_oracle(H, B, R, restriction, mu[:-1], y)
    with pytest.raises(ValueError, match="y must have shape"):
        gaussian_projection_oracle(H, B, R, restriction, mu, y[:-1])
    asymmetric_B = B.copy()
    asymmetric_B[0, 1] += 0.5
    with pytest.raises(ValueError, match="B must be symmetric"):
        gaussian_projection_oracle(H, asymmetric_B, R, restriction, mu, y)
    indefinite_R = R.copy()
    indefinite_R[0, 0] = -10.0
    with pytest.raises(ValueError, match="R must be positive definite"):
        gaussian_projection_oracle(H, B, indefinite_R, restriction, mu, y)


def test_native_posterior_rejects_nonfinite_observations(
    dense_gaussian_problem: tuple[np.ndarray, ...],
) -> None:
    """Native conditioning should reject non-finite observation values."""
    H, B, R, mu, y = dense_gaussian_problem
    invalid_y = y.copy()
    invalid_y[1] = np.nan

    with pytest.raises(ValueError, match="finite"):
        native_gaussian_posterior(H, B, R, mu, invalid_y)


def test_objectives_require_a_projection() -> None:
    """Projection-only objectives should reject unrelated objects clearly."""
    with pytest.raises(TypeError, match="BocquetProjection"):
        projected_dfs(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BocquetProjection"):
        projected_fisher_base_r(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BocquetProjection"):
        projected_fisher_aggregation_aware(object())  # type: ignore[arg-type]
