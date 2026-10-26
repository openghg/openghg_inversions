import numpy as np
import pytest
from scipy.stats import multivariate_normal

from openghg_inversions.models import cached_sigma
from openghg_inversions.models.cached_sigma import FixedOuCachedSigmaTarget, _sum_squares_gram
from openghg_inversions.models.fixed_ou import FixedOuLowRank, prepare_fixed_ou_low_rank


def _target(*, full_width: bool = False) -> tuple[FixedOuCachedSigmaTarget, np.ndarray, np.ndarray]:
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
    if full_width:
        factor = np.column_stack((factor, 0.3 * np.eye(6)))
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


@pytest.mark.parametrize("full_width", [False, True])
@pytest.mark.parametrize("sigma_values", [[0.23, 0.41], [0.0, 0.0], [0.12, 0.83]])
def test_cached_quadratic_value_and_gradient_match_dense_oracle(
    full_width: bool, sigma_values: list[float]
) -> None:
    target, factor, diagonal = _target(full_width=full_width)
    sigma = np.asarray(sigma_values)
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
    rhs = np.column_stack((target.observations - target.fixed_contribution, target.design))
    expected_gram = rhs.T @ np.linalg.solve(covariance, rhs)
    expected_logdet = np.linalg.slogdet(covariance)[1]

    assert cache.constant == pytest.approx(
        -0.5 * (target.n_obs * np.log(2.0 * np.pi) + expected_logdet + expected_gram[0, 0]),
        rel=2e-12,
    )
    np.testing.assert_allclose(cache.linear, expected_gram[1:, 0], rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(cache.precision, expected_gram[1:, 1:], rtol=2e-12, atol=1e-14)
    assert cache.log_likelihood(state) == pytest.approx(expected_value, rel=2e-12)
    assert cache.log_likelihood(state) == pytest.approx(target.log_likelihood(state, sigma), rel=2e-12)
    np.testing.assert_allclose(cache.gradient(state), expected_gradient, rtol=2e-12)
    np.testing.assert_allclose(cache.sigma, sigma)


@pytest.mark.parametrize("full_width", [False, True])
def test_refresh_factorizes_once_and_state_evaluations_do_not_refactorize(
    full_width: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _, _ = _target(full_width=full_width)
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

    def unexpected_transform(*args: object, **kwargs: object) -> None:
        pytest.fail("Refresh must reuse transformed fixed inputs and keep its quadratic in mode coordinates.")

    for matrix_type in {type(target.prepared.mode_transform), type(target.prepared.mode_transform.T)}:
        monkeypatch.setattr(matrix_type, "__matmul__", unexpected_transform)

    cache = target.refresh(np.array([0.23, 0.41]))
    expected_calls = (0, 0, 1)
    assert (qr_calls, svd_calls, cholesky_calls) == expected_calls

    for state in (np.zeros(2), np.ones(2), np.array([-0.5, 0.25])):
        assert np.isfinite(cache.log_likelihood(state))
        assert np.isfinite(cache.gradient(state)).all()
    assert (qr_calls, svd_calls, cholesky_calls) == expected_calls


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

    sigma = np.array([0.23, 0.41])
    transformed_rhs = replacement._transformed_rhs.copy()
    transformed_factor = replacement.prepared.transformed_factor.copy()
    cache = replacement.refresh(sigma)
    np.testing.assert_array_equal(replacement._transformed_rhs, transformed_rhs)
    np.testing.assert_array_equal(replacement.prepared.transformed_factor, transformed_factor)
    np.testing.assert_array_equal(sigma, [0.23, 0.41])
    sigma[:] = 0.0
    np.testing.assert_array_equal(cache.sigma, [0.23, 0.41])


@pytest.mark.parametrize("full_width", [False, True])
def test_refresh_scalar_amplitude_matches_vector(full_width: bool) -> None:
    target, _, _ = _target(full_width=full_width)
    scalar = target.refresh(0.3)
    vector = target.refresh(np.full(target.n_group, 0.3))

    assert scalar.constant == vector.constant
    np.testing.assert_array_equal(scalar.linear, vector.linear)
    np.testing.assert_array_equal(scalar.precision, vector.precision)
    np.testing.assert_array_equal(scalar.sigma, vector.sigma)


@pytest.mark.parametrize("full_width", [False, True])
@pytest.mark.parametrize("sigma", [-0.1, np.nan, np.inf, [0.1, -0.1], [0.1, np.nan], [0.1], [[0.1, 0.2]]])
def test_refresh_rejects_invalid_amplitudes(full_width: bool, sigma: object) -> None:
    target, _, _ = _target(full_width=full_width)
    with pytest.raises(ValueError):
        target.refresh(sigma)


def test_sum_squares_accepts_zero_rhs(monkeypatch: pytest.MonkeyPatch) -> None:
    target, _, _ = _target()
    target = FixedOuCachedSigmaTarget(
        prepared=target.prepared,
        observations=np.zeros(target.n_obs),
        fixed_contribution=np.zeros(target.n_obs),
        design=np.zeros((target.n_obs, 2)),
    )
    sigma = np.array([0.23, 0.41])
    expected_logdet = target.prepared.solve(np.zeros(target.n_obs), sigma).logdet

    def unexpected_fallback(*args: object, **kwargs: object) -> None:
        pytest.fail("Zero RHS has zero residual rounding error and should be accepted.")

    monkeypatch.setattr(FixedOuLowRank, "_solve_transformed", unexpected_fallback)
    cache = target.refresh(sigma)

    assert cache.constant == pytest.approx(-0.5 * (target.n_obs * np.log(2 * np.pi) + expected_logdet))
    np.testing.assert_array_equal(cache.linear, np.zeros(2))
    np.testing.assert_array_equal(cache.precision, np.zeros((2, 2)))


@pytest.mark.parametrize("rank", [0, 3, 5])
def test_non_low_rank_cache_keeps_existing_solver(rank: int, monkeypatch: pytest.MonkeyPatch) -> None:
    factor = np.arange(3 * rank, dtype=float).reshape(3, rank) / 10
    prepared = prepare_fixed_ou_low_rank(factor, np.ones(3), np.zeros(3), np.arange(3), 1.0)
    target = FixedOuCachedSigmaTarget(
        prepared=prepared,
        observations=np.array([1.0, 2.0, -1.0]),
        fixed_contribution=np.zeros(3),
        design=np.array([[0.1], [0.2], [0.3]]),
    )

    def unexpected_sum_squares(*args: object, **kwargs: object) -> None:
        pytest.fail("Only a nonzero factor narrower than the observations uses sum of squares.")

    monkeypatch.setattr(cached_sigma, "_sum_squares_gram", unexpected_sum_squares)
    cache = target.refresh(0.2)
    assert cache.log_likelihood([0.3]) == pytest.approx(
        target.log_likelihood([0.3], np.full(3, 0.2)), rel=2e-12
    )


@pytest.mark.parametrize("duplicate_columns", [False, True])
def test_sum_squares_rejection_uses_existing_solver(
    duplicate_columns: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The scalar core is well conditioned but the residual cancels; duplicates
    # make the core ill conditioned even before the residual screen is reached.
    factor = np.full((3, 2 if duplicate_columns else 1), 1e5 if duplicate_columns else 1e10)
    observations = factor[:, 0].copy()
    design = np.array([[1.0, 2.0], [-1.0, 1.0], [0.0, -1.0]])
    assert _sum_squares_gram(factor, np.column_stack((observations, design))) is None
    prepared = prepare_fixed_ou_low_rank(factor, np.ones(3), np.zeros(3), np.arange(3), 1.0)
    target = FixedOuCachedSigmaTarget(
        prepared=prepared, observations=observations, fixed_contribution=np.zeros(3), design=design,
    )
    sigma = np.zeros(3)
    reference = prepared._solve_transformed(target._transformed_rhs, sigma)
    expected_gram = target._transformed_rhs.T @ reference.solution
    calls = 0
    original = FixedOuLowRank._solve_transformed

    def counted_solve(self: FixedOuLowRank, *args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FixedOuLowRank, "_solve_transformed", counted_solve)
    cache = target.refresh(sigma)

    assert calls == 1
    assert cache.constant == -0.5 * (3 * np.log(2 * np.pi) + reference.logdet + expected_gram[0, 0])
    np.testing.assert_array_equal(cache.linear, expected_gram[1:, 0])
    np.testing.assert_array_equal(cache.precision, (expected_gram[1:, 1:] + expected_gram[1:, 1:].T) * 0.5)


@pytest.mark.parametrize("factor_scale", [1e10, 1e20])
def test_residual_screen_rejects_cancellation_despite_scalar_core(factor_scale: float) -> None:
    factor = np.full((3, 1), factor_scale)
    rhs = np.column_stack((factor[:, 0], np.array([1.0, -1.0, 0.0])))
    assert _sum_squares_gram(factor, rhs) is None


def test_sum_squares_rejects_nonfinite_roundoff_estimate() -> None:
    assert _sum_squares_gram(np.full((3, 1), 0.2), np.full((3, 2), 1e200)) is None


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
