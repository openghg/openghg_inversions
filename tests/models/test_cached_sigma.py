import numpy as np
import pytest
from scipy.stats import multivariate_normal

from openghg_inversions.models.cached_sigma import FixedOuCachedSigmaTarget
from openghg_inversions.models.fixed_ou import prepare_fixed_ou_low_rank


def _target() -> tuple[FixedOuCachedSigmaTarget, np.ndarray, np.ndarray]:
    factor = np.array(
        [
            [0.8, -0.2],
            [0.3, 0.5],
            [-0.1, 0.7],
            [0.6, 0.1],
            [0.2, -0.4],
            [0.1, 0.2],
        ]
    )
    diagonal = np.array([0.16, 0.25, 0.09, 0.36, 0.20, 0.11])
    times = np.array([0.0, 0.25, 1.5, 2.0, 7.0, 8.5])
    sites = np.array([0, 1, 0, 1, 0, 1])
    prepared = prepare_fixed_ou_low_rank(
        factor,
        diagonal,
        times,
        sites,
        5.5,
        site_labels=("MHD", "TAC"),
    )
    observations = np.array([1.2, -0.4, 0.5, 1.6, -0.2, 0.8])
    fixed = np.array([0.1, -0.1, 0.2, 0.3, -0.2, 0.0])
    design = np.array(
        [
            [0.2, -0.1],
            [0.3, 0.4],
            [-0.5, 0.2],
            [0.1, 0.6],
            [0.4, -0.2],
            [-0.1, 0.3],
        ]
    )
    return (
        FixedOuCachedSigmaTarget(
            prepared=prepared,
            observations=observations,
            fixed_contribution=fixed,
            design=design,
        ),
        factor,
        diagonal,
    )


def test_cached_quadratic_value_and_gradient_match_dense_oracle() -> None:
    target, factor, diagonal = _target()
    sigma = np.array([0.23, 0.41])
    state = np.array([0.3, -0.7])

    cache = target.refresh(sigma)
    covariance = factor @ factor.T + np.diag(diagonal)
    for block in target.prepared.site_blocks:
        indices = block.observation_indices
        covariance[np.ix_(indices, indices)] += (
            np.square(sigma[block.site])
            * block.correlation_cholesky
            @ block.correlation_cholesky.T
        )
    mean = target.fixed_contribution + target.design @ state
    expected_value = multivariate_normal.logpdf(
        target.observations,
        mean=mean,
        cov=covariance,
    )
    expected_gradient = target.design.T @ np.linalg.solve(
        covariance,
        target.observations - mean,
    )

    assert cache.log_likelihood(state) == pytest.approx(expected_value, rel=2e-12)
    np.testing.assert_allclose(cache.gradient(state), expected_gradient, rtol=2e-12)
    np.testing.assert_allclose(cache.sigma, sigma)


def test_refresh_uses_one_rank_space_svd_and_state_evaluations_do_not_refactorize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _, _ = _target()
    qr_calls = 0
    svd_calls = 0
    cholesky_calls = 0
    original_qr = np.linalg.qr
    original_svd = np.linalg.svd
    original_cholesky = np.linalg.cholesky

    def counted_qr(*args: object, **kwargs: object) -> tuple[np.ndarray, ...]:
        nonlocal qr_calls
        qr_calls += 1
        return original_qr(*args, **kwargs)

    def counted_svd(*args: object, **kwargs: object) -> tuple[np.ndarray, ...]:
        nonlocal svd_calls
        svd_calls += 1
        return original_svd(*args, **kwargs)

    def counted_cholesky(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal cholesky_calls
        cholesky_calls += 1
        return original_cholesky(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "qr", counted_qr)
    monkeypatch.setattr(np.linalg, "svd", counted_svd)
    monkeypatch.setattr(np.linalg, "cholesky", counted_cholesky)

    cache = target.refresh(np.array([0.23, 0.41]))
    assert qr_calls == 1
    assert svd_calls == 1
    assert cholesky_calls == 0

    for state in (np.zeros(2), np.ones(2), np.array([-0.5, 0.25])):
        assert np.isfinite(cache.log_likelihood(state))
        assert np.isfinite(cache.gradient(state)).all()
    assert qr_calls == 1
    assert svd_calls == 1
    assert cholesky_calls == 0


def test_exact_sigma_conditional_reuses_fixed_ou_target() -> None:
    target, _, _ = _target()
    sigma = np.array([0.23, 0.41])
    state = np.array([0.3, -0.7])
    residual = (
        target.observations
        - target.fixed_contribution
        - target.design @ state
    )

    evaluation = target.evaluate_from_residual(residual, sigma)

    assert evaluation.log_likelihood == pytest.approx(
        target.log_likelihood(state, sigma), rel=1e-15
    )
    step = 1.0e-6
    finite_difference = np.empty(2)
    for site in range(2):
        direction = np.zeros(2)
        direction[site] = step
        finite_difference[site] = (
            target.evaluate_from_residual(residual, sigma + direction).log_likelihood
            - target.evaluate_from_residual(residual, sigma - direction).log_likelihood
        ) / (2.0 * step)
    np.testing.assert_allclose(
        evaluation.gradient_site_amplitude,
        finite_difference,
        rtol=2e-8,
        atol=2e-8,
    )


def test_target_copies_borrowed_inputs() -> None:
    target, _, _ = _target()
    observations = target.observations.copy()
    fixed = target.fixed_contribution.copy()
    design = target.design.copy()
    replacement = FixedOuCachedSigmaTarget(
        prepared=target.prepared,
        observations=observations,
        fixed_contribution=fixed,
        design=design,
    )

    observations[:] = 0.0
    fixed[:] = 0.0
    design[:] = 0.0

    assert np.any(replacement.observations != 0.0)
    assert np.any(replacement.fixed_contribution != 0.0)
    assert np.any(replacement.design != 0.0)


def test_cached_target_avoids_cancellation_for_dominant_low_rank_factor() -> None:
    factor_scale = 1.0e10
    prepared = prepare_fixed_ou_low_rank(
        np.full((2, 1), factor_scale),
        np.ones(2),
        np.array([0.0, 0.0]),
        np.array([0, 1]),
        1.0,
        site_labels=("MHD", "TAC"),
    )
    observations = np.full(2, factor_scale)
    target = FixedOuCachedSigmaTarget(
        prepared=prepared,
        observations=observations,
        fixed_contribution=np.zeros(2),
        design=np.zeros((2, 1)),
    )

    cache = target.refresh(np.ones(2))
    expected = prepared.evaluate(observations, np.ones(2)).log_likelihood

    assert cache.log_likelihood([0.0]) == pytest.approx(expected, rel=1.0e-12)
