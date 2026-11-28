"""Paige generalised least squares against dense and extended-precision references."""

import mpmath
import numpy as np
import pytest
import scipy.linalg as sl

from openghg_inversions.inference.gls import paige_gls, psd_factor


def _problem(seed=0, nobs=30, n=5, noise_columns=8):
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(nobs, n))
    low_rank = rng.normal(size=(nobs, noise_columns))
    diagonal = rng.uniform(0.2, 1.0, nobs)
    noise_factor = np.hstack([low_rank, np.diag(np.sqrt(diagonal))])
    prior_factor = rng.normal(size=(n, n)) + 2 * np.eye(n)
    prior_mean = rng.normal(size=n)
    y = rng.normal(size=nobs)
    return y, design, noise_factor, prior_mean, prior_factor


def _kalman(y, design, noise_covariance, prior_mean, prior_covariance):
    innovation = design @ prior_covariance @ design.T + noise_covariance
    gain = sl.solve(innovation, design @ prior_covariance, assume_a="pos").T
    return prior_mean + gain @ (y - design @ prior_mean), prior_covariance - gain @ design @ prior_covariance


def _precision(y, design, noise_covariance, prior_mean, prior_covariance):
    noise = sl.cho_factor(noise_covariance, lower=True)
    precision = design.T @ sl.cho_solve(noise, design) + np.linalg.inv(prior_covariance)
    covariance = np.linalg.inv(precision)
    information = design.T @ sl.cho_solve(noise, y) + sl.solve(prior_covariance, prior_mean, assume_a="pos")
    return covariance @ information, covariance


@pytest.mark.parametrize("reference", [_kalman, _precision], ids=["kalman", "precision"])
def test_well_conditioned_posterior_matches_dense_forms(reference):
    y, design, noise_factor, prior_mean, prior_factor = _problem()
    posterior = paige_gls(y, design, noise_factor, prior_mean, prior_factor)
    mean, covariance = reference(
        y, design, noise_factor @ noise_factor.T, prior_mean, prior_factor @ prior_factor.T,
    )
    np.testing.assert_allclose(posterior.mean, mean, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(posterior.covariance, covariance, rtol=1e-10, atol=1e-12)
    assert posterior.cov_factor.shape == (5, 5)
    diagnostics = posterior.diagnostics
    assert (diagnostics.noise_rank, diagnostics.prior_rank, diagnostics.constraint_rank) == (30, 5, 30)
    assert 0 < diagnostics.rcond_r1 <= 1 and 0 < diagnostics.rcond_s2 <= 1
    np.testing.assert_allclose(diagnostics.observation_residual_norm, np.linalg.norm(y - design @ mean))


def test_rectangular_noise_factor_matches_its_square_cholesky_factor():
    y, design, noise_factor, prior_mean, prior_factor = _problem(seed=1)
    square = np.linalg.cholesky(noise_factor @ noise_factor.T)
    rectangular = paige_gls(y, design, noise_factor, prior_mean, prior_factor)
    cholesky = paige_gls(y, design, square, prior_mean, prior_factor)
    np.testing.assert_allclose(rectangular.mean, cholesky.mean, rtol=1e-11)
    np.testing.assert_allclose(rectangular.covariance, cholesky.covariance, rtol=1e-10, atol=1e-13)
    assert rectangular.diagnostics.noise_factor_columns == 38
    assert cholesky.diagnostics.noise_factor_columns == 30


def test_rank_deficient_noise_and_prior_factors_match_kalman_form():
    rng = np.random.default_rng(2)
    nobs, n = 9, 6
    design = rng.normal(size=(nobs, n))
    # Rank-4 noise and rank-5 prior; their sum still spans the observations.
    noise_factor = rng.normal(size=(nobs, 4)) @ rng.normal(size=(4, 7))
    prior_factor = rng.normal(size=(n, 5))
    prior_mean = rng.normal(size=n)
    y = design @ (prior_mean + prior_factor @ rng.normal(size=5)) + noise_factor @ rng.normal(size=7)
    posterior = paige_gls(y, design, noise_factor, prior_mean, prior_factor)
    mean, covariance = _kalman(y, design, noise_factor @ noise_factor.T, prior_mean, prior_factor @ prior_factor.T)
    np.testing.assert_allclose(posterior.mean, mean, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(posterior.covariance, covariance, rtol=1e-8, atol=1e-10)
    assert posterior.diagnostics.noise_rank == 4
    assert posterior.diagnostics.prior_rank == 5
    assert posterior.diagnostics.constraint_rank == nobs


def test_inconsistent_data_for_rank_deficient_noise_are_rejected():
    rng = np.random.default_rng(3)
    design = np.zeros((4, 1))
    noise_factor = rng.normal(size=(4, 2))
    with pytest.raises(ValueError, match="inconsistent"):
        paige_gls(rng.normal(size=4), design, noise_factor, np.zeros(1), np.eye(1))


def test_ill_conditioned_noise_keeps_digits_that_dense_forms_lose():
    """R is 1e-12 of a rank-3 aggregation covariance, so cond(A + R) is about 1e13."""
    rng = np.random.default_rng(1)
    nobs, n = 24, 4
    design = rng.normal(size=(nobs, n))
    aggregation = rng.normal(size=(nobs, 3))
    mismatch = np.full(nobs, 1e-12)
    noise_factor = np.hstack([aggregation, np.diag(np.sqrt(mismatch))])
    prior_factor = np.diag([1.0, 2.0, 0.5, 1.5])
    prior_mean = rng.normal(size=n)
    y = design @ (prior_mean + prior_factor @ rng.normal(size=n)) + noise_factor @ rng.normal(size=nobs + 3)

    mpmath.mp.dps = 60
    to_mp = lambda array: mpmath.matrix(array.tolist())  # noqa: E731
    noise = to_mp(aggregation) * to_mp(aggregation).T + mpmath.diag([mpmath.mpf(value) for value in mismatch])
    prior_precision = to_mp(prior_factor @ prior_factor.T) ** -1
    weighted = to_mp(design).T * noise ** -1
    reference_covariance = (weighted * to_mp(design) + prior_precision) ** -1
    reference_mean = reference_covariance * (weighted * to_mp(y[:, None]) + prior_precision * to_mp(prior_mean[:, None]))
    reference_mean = np.array([float(value) for value in reference_mean])
    reference_covariance = np.array(reference_covariance.tolist(), dtype=float)

    def error(value, reference):
        return np.abs(value - reference).max() / np.abs(reference).max()

    posterior = paige_gls(y, design, noise_factor, prior_mean, prior_factor)
    assert error(posterior.mean, reference_mean) < 1e-12
    assert error(posterior.covariance, reference_covariance) < 1e-8
    assert posterior.diagnostics.noise_covariance_condition_estimate > 1e12

    covariance = noise_factor @ noise_factor.T
    precision_mean, precision_covariance = _precision(
        y, design, covariance, prior_mean, prior_factor @ prior_factor.T,
    )
    _, kalman_covariance = _kalman(y, design, covariance, prior_mean, prior_factor @ prior_factor.T)
    # The dense forms lose at least four digits of the posterior covariance.
    assert error(precision_covariance, reference_covariance) > 1e-6
    assert error(kalman_covariance, reference_covariance) > 1e-6
    assert error(precision_mean, reference_mean) > 1e3 * error(posterior.mean, reference_mean)


def test_posterior_draws_use_the_square_factor():
    y, design, noise_factor, prior_mean, prior_factor = _problem(seed=4)
    posterior = paige_gls(y, design, noise_factor, prior_mean, prior_factor)
    draws = posterior.draw(3, np.random.default_rng(9))
    standard = np.random.default_rng(9).standard_normal((3, 5))
    np.testing.assert_allclose(draws, posterior.mean + standard @ posterior.cov_factor.T)


def test_psd_factor_keeps_positive_modes_and_records_clipping():
    vectors = np.linalg.qr(np.random.default_rng(5).normal(size=(4, 4)))[0]
    matrix = vectors @ np.diag([2.0, 1.0, 1e-20, -1e-14]) @ vectors.T
    result = psd_factor(matrix)
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    assert result.diagnostics["rank"] == result.factor.shape[1] == int((eigenvalues > 0).sum())
    np.testing.assert_allclose(result.factor @ result.factor.T, matrix, atol=2e-14)
    assert result.diagnostics["dropped_positive_max"] == 0.0
    assert result.diagnostics["negative_count"] >= 1
    assert result.diagnostics["negative_clipped_max"] == pytest.approx(1e-14, rel=1e-2)
    assert result.diagnostics["negative_clipped_sum"] >= result.diagnostics["negative_clipped_max"]
    truncated = psd_factor(matrix, rtol=1e-10)
    assert truncated.factor.shape == (4, 2)
    assert truncated.diagnostics["rank"] == 2
    with pytest.raises(ValueError, match="symmetric"):
        psd_factor(np.array([[1.0, 0.5], [0.0, 1.0]]))


@pytest.mark.parametrize("prior_sd", [1e8, 1e12])
def test_prior_and_noise_scales_are_balanced(prior_sd):
    """A diffuse prior against small noise is equilibrated, not refused."""
    rng = np.random.default_rng(6)
    nobs, n = 12, 3
    design = rng.normal(size=(nobs, n))
    noise_sd = 1e-3 * rng.uniform(0.5, 2.0, nobs)
    prior_mean = rng.normal(size=n)
    y = design @ rng.normal(size=n) + noise_sd * rng.normal(size=nobs)
    posterior = paige_gls(y, design, np.diag(noise_sd), prior_mean, prior_sd * np.eye(n))
    weighted = design / noise_sd[:, None]
    precision = weighted.T @ weighted + np.eye(n) / prior_sd**2
    covariance = np.linalg.inv(precision)
    mean = covariance @ (weighted.T @ (y / noise_sd) + prior_mean / prior_sd**2)
    np.testing.assert_allclose(posterior.covariance, covariance, rtol=1e-10, atol=0)
    np.testing.assert_allclose(posterior.mean, mean, rtol=1e-11)
