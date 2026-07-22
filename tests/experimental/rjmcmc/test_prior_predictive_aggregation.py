"""Prior-predictive contracts for lognormal regional scaling factors.

These deterministic tests document the distinction between log-space and
arithmetic lognormal parameters, and how an IID regional prior propagates to a
weighted country total.  The aggregation formula is analytic, so the tests do
not depend on Monte Carlo convergence.
"""

from __future__ import annotations

from math import exp, log, sqrt

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.hierarchy import (
    arithmetic_moments_to_lognormal_parameters,
)


def _lognormal_moments(mu_log: float, sigma_log: float) -> tuple[float, float]:
    """Return exact arithmetic mean and variance from lognormal parameters."""
    mean = exp(mu_log + 0.5 * sigma_log**2)
    variance = (exp(sigma_log**2) - 1.0) * exp(2.0 * mu_log + sigma_log**2)
    return mean, variance


def _weighted_total_moments(
    weights: np.ndarray,
    *,
    multiplier_mean: float,
    multiplier_variance: float,
    baseline_total: float = 1.0,
) -> tuple[float, float]:
    """Return moments of a baseline total scaled by IID regional factors."""
    normalized_weights = weights / weights.sum()
    mean = baseline_total * multiplier_mean
    variance = baseline_total**2 * multiplier_variance * float(np.dot(normalized_weights, normalized_weights))
    return mean, variance


def test_arithmetic_mean_and_sd_one_have_shifted_log_location() -> None:
    """Arithmetic moments (1, 1) should map to the corrected log parameters."""
    mu_log, sigma_log = arithmetic_moments_to_lognormal_parameters(1.0, 1.0)

    assert mu_log == pytest.approx(-0.5 * log(2.0), rel=0.0, abs=1e-15)
    assert sigma_log == pytest.approx(sqrt(log(2.0)), rel=0.0, abs=1e-15)
    assert _lognormal_moments(mu_log, sigma_log) == pytest.approx((1.0, 1.0))


def test_historical_zero_one_log_parameters_have_upward_biased_mean() -> None:
    """Log parameters (0, 1) should imply mean exp(1/2), not arithmetic mean one."""
    mean, variance = _lognormal_moments(0.0, 1.0)

    assert mean == pytest.approx(exp(0.5), rel=0.0, abs=1e-15)
    assert variance == pytest.approx((exp(1.0) - 1.0) * exp(1.0), rel=0.0, abs=1e-14)
    assert mean > 1.0


@pytest.mark.parametrize(
    "weights",
    [
        np.array([1.0]),
        np.array([1.0, 3.0]),
        np.array([0.5, 1.0, 2.0, 4.0]),
    ],
)
def test_normalized_country_weights_preserve_mean_and_control_variance(
    weights: np.ndarray,
) -> None:
    """Normalized weights should preserve the mean and set variance via sum(w^2)."""
    multiplier_mean = 1.25
    multiplier_variance = 2.5
    baseline_total = 40.0
    normalized_weights = weights / weights.sum()

    mean, variance = _weighted_total_moments(
        weights,
        multiplier_mean=multiplier_mean,
        multiplier_variance=multiplier_variance,
        baseline_total=baseline_total,
    )

    assert mean == pytest.approx(baseline_total * multiplier_mean)
    assert variance == pytest.approx(
        baseline_total**2 * multiplier_variance * np.square(normalized_weights).sum()
    )
    assert variance == pytest.approx(
        baseline_total**2 * multiplier_variance / (1.0 / np.square(normalized_weights).sum())
    )


def test_more_equal_regions_concentrate_around_the_same_biased_mean() -> None:
    """Increasing fixed k should shrink spread without removing a prior mean bias."""
    historical_mean, historical_variance = _lognormal_moments(0.0, 1.0)
    small_k = 4
    large_k = 64

    small_mean, small_variance = _weighted_total_moments(
        np.ones(small_k),
        multiplier_mean=historical_mean,
        multiplier_variance=historical_variance,
    )
    large_mean, large_variance = _weighted_total_moments(
        np.ones(large_k),
        multiplier_mean=historical_mean,
        multiplier_variance=historical_variance,
    )

    assert small_mean == pytest.approx(exp(0.5))
    assert large_mean == pytest.approx(small_mean)
    assert small_variance == pytest.approx(historical_variance / small_k)
    assert large_variance == pytest.approx(historical_variance / large_k)
    assert sqrt(large_variance) == pytest.approx(sqrt(small_variance) / 4.0)
    assert large_mean > 1.0
