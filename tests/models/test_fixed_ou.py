import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.stats import multivariate_normal

from openghg_inversions.models.fixed_ou import prepare_fixed_ou_low_rank


def _prepared(*, rank: int = 2):
    factor = np.array(
        [
            [0.30, 0.00],
            [0.10, 0.20],
            [0.00, 0.25],
            [0.15, 0.10],
            [0.05, 0.00],
        ]
    )[:, :rank]
    diagonal = np.array([0.00, 0.20, 0.10, 0.00, 0.15])
    times = np.array(
        [
            "2021-01-01T06:00",
            "2021-01-01T03:00",
            "2021-01-01T00:00",
            "2021-01-01T09:00",
            "2021-01-01T12:00",
        ],
        dtype="datetime64[m]",
    )
    sites = np.array([0, 1, 0, 1, 0])
    prepared = prepare_fixed_ou_low_rank(
        factor,
        diagonal,
        times,
        sites,
        {"MHD": 4.0, "TAC": 8.0},
        site_labels=("MHD", "TAC"),
    )
    return prepared, factor, diagonal, times, sites


def _direct_covariance(
    factor: np.ndarray,
    diagonal: np.ndarray,
    times: np.ndarray,
    sites: np.ndarray,
    amplitude: np.ndarray,
) -> np.ndarray:
    hours = times.astype("datetime64[m]").astype(np.int64) / 60.0
    tau = np.array([4.0, 8.0])
    lag = np.abs(hours[:, None] - hours[None, :])
    correlation = np.exp(-lag / tau[sites, None])
    correlation[sites[:, None] != sites[None, :]] = 0.0
    return (
        factor @ factor.T
        + np.diag(diagonal)
        + amplitude[sites, None] * amplitude[sites[None, :]] * correlation
    )


def test_covariance_preserves_interleaved_rows_and_labelled_tau() -> None:
    prepared, factor, diagonal, times, sites = _prepared()
    amplitude = np.array([0.7, 1.2])

    expected = _direct_covariance(factor, diagonal, times, sites, amplitude)

    np.testing.assert_allclose(prepared.covariance_dense(amplitude), expected)
    assert prepared.site_labels == ("MHD", "TAC")
    np.testing.assert_allclose(prepared.tau_hours_by_site, [4.0, 8.0])
    ou_covariance = expected - factor @ factor.T - np.diag(diagonal)
    assert ou_covariance[0, 1] == 0.0


@pytest.mark.parametrize("rank", [0, 2])
def test_pytensor_logp_matches_dense_oracle_with_zero_diagonal_tail(rank: int) -> None:
    prepared, factor, diagonal, times, sites = _prepared(rank=rank)
    observed = np.array([0.5, -0.1, 0.2, 0.7, -0.4])
    mean = np.array([0.1, 0.0, 0.3, 0.2, -0.2])
    amplitude = np.array([0.7, 1.2])
    value_symbol = pt.dvector("value")
    mean_symbol = pt.dvector("mean")
    amplitude_symbol = pt.dvector("amplitude")
    compiled = pytensor.function(
        [value_symbol, mean_symbol, amplitude_symbol],
        prepared.logp(value_symbol, mean_symbol, amplitude_symbol),
    )

    covariance = _direct_covariance(factor, diagonal, times, sites, amplitude)
    expected = multivariate_normal.logpdf(observed, mean=mean, cov=covariance)

    assert float(compiled(observed, mean, amplitude)) == pytest.approx(expected, rel=1e-10)


def test_logp_is_differentiable_for_inferred_site_amplitudes() -> None:
    prepared, factor, diagonal, times, sites = _prepared()
    observed = np.array([0.5, -0.1, 0.2, 0.7, -0.4])
    mean = np.array([0.1, 0.0, 0.3, 0.2, -0.2])
    amplitude = np.array([0.7, 1.2])
    amplitude_symbol = pt.dvector("amplitude")
    logp = prepared.logp(pt.as_tensor(observed), pt.as_tensor(mean), amplitude_symbol)
    gradient = pytensor.function([amplitude_symbol], pt.grad(logp, amplitude_symbol))

    step = 1.0e-5
    expected = np.empty(2)
    for site in range(2):
        upper = amplitude.copy()
        lower = amplitude.copy()
        upper[site] += step
        lower[site] -= step
        expected[site] = (
            multivariate_normal.logpdf(
                observed,
                mean=mean,
                cov=_direct_covariance(factor, diagonal, times, sites, upper),
            )
            - multivariate_normal.logpdf(
                observed,
                mean=mean,
                cov=_direct_covariance(factor, diagonal, times, sites, lower),
            )
        ) / (2.0 * step)

    np.testing.assert_allclose(gradient(amplitude), expected, rtol=1e-5)


def test_analytic_value_mean_and_amplitude_gradients_match_dense_finite_difference() -> None:
    prepared, factor, diagonal, times, sites = _prepared()
    observed = np.array([0.5, -0.1, 0.2, 0.7, -0.4])
    mean = np.array([0.1, 0.0, 0.3, 0.2, -0.2])
    amplitude = np.array([0.7, 1.2])
    value_symbol = pt.dvector("value")
    mean_symbol = pt.dvector("mean")
    amplitude_symbol = pt.dvector("amplitude")
    logp = prepared.logp(value_symbol, mean_symbol, amplitude_symbol)
    compiled = pytensor.function(
        [value_symbol, mean_symbol, amplitude_symbol],
        [
            logp,
            pt.grad(logp, value_symbol),
            pt.grad(logp, mean_symbol),
            pt.grad(logp, amplitude_symbol),
        ],
    )
    _, value_gradient, mean_gradient, amplitude_gradient = compiled(
        observed, mean, amplitude
    )

    def dense_logp(value: np.ndarray, location: np.ndarray, scale: np.ndarray) -> float:
        return float(
            multivariate_normal.logpdf(
                value,
                mean=location,
                cov=_direct_covariance(factor, diagonal, times, sites, scale),
            )
        )

    step = 1.0e-6
    expected_value = np.empty(observed.size)
    expected_mean = np.empty(mean.size)
    expected_amplitude = np.empty(amplitude.size)
    for index in range(observed.size):
        direction = np.zeros(observed.size)
        direction[index] = step
        expected_value[index] = (
            dense_logp(observed + direction, mean, amplitude)
            - dense_logp(observed - direction, mean, amplitude)
        ) / (2.0 * step)
        expected_mean[index] = (
            dense_logp(observed, mean + direction, amplitude)
            - dense_logp(observed, mean - direction, amplitude)
        ) / (2.0 * step)
    for index in range(amplitude.size):
        direction = np.zeros(amplitude.size)
        direction[index] = step
        expected_amplitude[index] = (
            dense_logp(observed, mean, amplitude + direction)
            - dense_logp(observed, mean, amplitude - direction)
        ) / (2.0 * step)

    evaluation = prepared.evaluate(observed - mean, amplitude)
    np.testing.assert_allclose(value_gradient, expected_value, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(mean_gradient, expected_mean, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(
        amplitude_gradient, expected_amplitude, rtol=2e-8, atol=2e-8
    )
    np.testing.assert_allclose(evaluation.gradient_residual, expected_value, rtol=2e-8)
    np.testing.assert_allclose(
        evaluation.gradient_site_amplitude, expected_amplitude, rtol=2e-8
    )


def test_logp_matches_frozen_verification_games_fixture() -> None:
    """Match verification-games ``_inputs()`` and its fixed-OU evaluator."""
    observed = np.array([1.2, -0.4, 0.5, 1.6, -0.2, 0.8])
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
    prepared = prepare_fixed_ou_low_rank(
        factor,
        np.array([0.16, 0.25, 0.09, 0.36, 0.20, 0.11]),
        np.array([0.0, 0.25, 1.5, 2.0, 7.0, 8.5]),
        np.array([0, 1, 0, 1, 0, 1]),
        5.5,
    )
    state = np.array([0.3, -0.7])
    amplitude = np.array([0.23, 0.41])
    expression = prepared.logp(
        pt.as_tensor(observed),
        pt.as_tensor(fixed + design @ state),
        pt.as_tensor(amplitude),
    )

    # Frozen from FixedOuCachedMarginalQuadraticTarget.log_likelihood.
    assert float(expression.eval()) == pytest.approx(-7.781189009752531, rel=2e-12)


def test_singular_zero_amplitude_returns_negative_infinity_without_jitter() -> None:
    prepared = prepare_fixed_ou_low_rank(
        np.empty((2, 0)),
        np.zeros(2),
        np.array([0.0, 1.0]),
        np.zeros(2, dtype=int),
        5.0,
    )
    expression = prepared.logp(
        pt.as_tensor(np.zeros(2)),
        pt.as_tensor(np.zeros(2)),
        pt.as_tensor(np.array([0.0])),
    )

    assert float(expression.eval()) == -np.inf


def test_overflowing_log_amplitude_rejects_without_nan_gradient() -> None:
    prepared, *_ = _prepared()
    log_amplitude = pt.dvector("log_amplitude")
    amplitude = pt.exp(log_amplitude)
    logp = prepared.logp(
        pt.as_tensor(np.zeros(prepared.n_observation)),
        pt.as_tensor(np.zeros(prepared.n_observation)),
        amplitude,
    )
    compiled = pytensor.function(
        [log_amplitude],
        [logp, pt.grad(logp, log_amplitude)],
    )

    with np.errstate(over="ignore", invalid="ignore"):
        value, gradient = compiled(np.full(prepared.n_site, 1_000.0))

    assert float(value) == -np.inf
    assert not np.isnan(gradient).any()


def test_custom_dist_supports_fixed_and_inferred_amplitudes() -> None:
    prepared, *_ = _prepared()
    observed = np.array([0.5, -0.1, 0.2, 0.7, -0.4])

    with pm.Model(coords={"observation": np.arange(5), "site": prepared.site_labels}) as model:
        amplitude = pm.HalfNormal("ou_site_amplitude", 1.0, dims="site")
        pm.CustomDist(
            "y",
            np.zeros(5),
            amplitude,
            logp=prepared.logp,
            random=prepared.random,
            signature="(n),(s)->(n)",
            observed=observed,
            dims="observation",
        )

    assert np.isfinite(model.compile_logp()(model.initial_point()))
    with model:
        predictive = pm.sample_prior_predictive(draws=2, var_names=["y"])
    assert predictive.prior_predictive["y"].shape == (1, 2, 5)


def test_marginal_variance_includes_all_three_components_once() -> None:
    prepared, factor, diagonal, *_ = _prepared()
    amplitude = pt.dvector("amplitude")
    compiled = pytensor.function([amplitude], prepared.marginal_variance(amplitude))

    expected = diagonal + np.square(factor).sum(axis=1) + np.square([0.7, 1.2])[prepared.site_index]

    np.testing.assert_allclose(compiled([0.7, 1.2]), expected)


def test_random_draws_reproduce_dense_covariance() -> None:
    prepared, *_ = _prepared()
    amplitude = np.array([0.7, 1.2])
    draws = prepared.random(
        np.zeros(prepared.n_observation),
        amplitude,
        rng=np.random.default_rng(4),
        size=40_000,
    )

    np.testing.assert_allclose(
        np.cov(draws, rowvar=False),
        prepared.covariance_dense(amplitude),
        rtol=0.04,
        atol=0.015,
    )


def test_duplicate_times_within_site_are_rejected_without_jitter() -> None:
    with pytest.raises(ValueError, match="unique within site 'MHD'"):
        prepare_fixed_ou_low_rank(
            np.empty((3, 0)),
            np.zeros(3),
            np.array([0.0, 1.0, 0.0]),
            np.array([0, 1, 0]),
            5.0,
            site_labels=("MHD", "TAC"),
        )


def test_labelled_tau_requires_exact_site_coverage() -> None:
    with pytest.raises(ValueError, match="exactly match site_labels"):
        prepare_fixed_ou_low_rank(
            np.empty((2, 0)),
            np.ones(2),
            np.array([0.0, 1.0]),
            np.array([0, 1]),
            {"MHD": 5.0},
            site_labels=("MHD", "TAC"),
        )


@pytest.mark.parametrize("tau_hours", [0.0, -1.0, np.inf, np.nan])
def test_tau_must_be_finite_and_strictly_positive(tau_hours: float) -> None:
    with pytest.raises(ValueError, match="finite, strictly positive"):
        prepare_fixed_ou_low_rank(
            np.empty((2, 0)),
            np.ones(2),
            np.array([0.0, 1.0]),
            np.array([0, 1]),
            tau_hours,
            site_labels=("MHD", "TAC"),
        )


def test_small_tau_reproduces_iid_site_amplitudes() -> None:
    prepared, factor, diagonal, *_ = _prepared()
    iid = prepare_fixed_ou_low_rank(
        factor,
        diagonal,
        prepared.observation_time_hours,
        prepared.site_index,
        1.0e-12,
        site_labels=prepared.site_labels,
    )
    amplitude = np.array([0.7, 1.2])
    expected = (
        factor @ factor.T
        + np.diag(diagonal)
        + np.diag(np.square(amplitude[prepared.site_index]))
    )

    np.testing.assert_allclose(iid.covariance_dense(amplitude), expected, atol=0.0)
