"""Tests for the shared partially pooled dynamic-coefficient hierarchy.

The tests cover arithmetic-moment conversion, normalized conditional and
hyperprior log densities, positive-support and immutable-configuration
validation, independence of the one-pool hyperprior from the active dimension,
and NumPy/Numba parity.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import log, pi, sqrt

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.hierarchy import (
    SharedLognormalHierarchy,
    arithmetic_moments_to_log_state,
    arithmetic_moments_to_lognormal_parameters,
    log_moments_to_lognormal_parameters,
    shared_coefficient_log_prior_numba,
    shared_coefficient_log_prior_numpy,
    shared_hyperprior_log_density_numba,
    shared_hyperprior_log_density_numpy,
)


def _hierarchy() -> SharedLognormalHierarchy:
    """Return a non-symmetric configuration that exposes parameter swaps."""
    return SharedLognormalHierarchy(
        mean_hyperprior_median=1.2,
        mean_hyperprior_log_sd=0.4,
        sd_hyperprior_median=0.7,
        sd_hyperprior_log_sd=0.9,
    )


def _normal_log_density(value: float, mean: float, standard_deviation: float) -> float:
    """Evaluate a normalized Normal log density independently."""
    standardized = (value - mean) / standard_deviation
    return -0.5 * standardized**2 - log(standard_deviation) - 0.5 * log(2.0 * pi)


def _lognormal_log_density(value: float, mu: float, sigma: float) -> float:
    """Evaluate a normalized lognormal log density independently."""
    standardized = (log(value) - mu) / sigma
    return -0.5 * standardized**2 - log(value) - log(sigma) - 0.5 * log(2.0 * pi)


def test_mean_one_sd_one_convert_to_expected_lognormal_parameters() -> None:
    """Arithmetic mean one and SD one should not be confused with log parameters."""
    eta, zeta = arithmetic_moments_to_log_state(1.0, 1.0)
    mu, sigma = arithmetic_moments_to_lognormal_parameters(1.0, 1.0)

    assert eta == 0.0
    assert zeta == 0.0
    assert mu == pytest.approx(-0.5 * log(2.0), abs=1e-15)
    assert mu == pytest.approx(-0.34657359027997265, abs=1e-15)
    assert sigma == pytest.approx(sqrt(log(2.0)), abs=1e-15)
    assert log_moments_to_lognormal_parameters(eta, zeta) == pytest.approx((mu, sigma))


def test_conditional_prior_matches_independent_normalized_formula() -> None:
    """The active shared coefficient prior should include every normalization term."""
    coefficients = np.array([0.4, 1.1, 2.3, 99.0])
    eta, zeta = arithmetic_moments_to_log_state(1.4, 0.8)
    mu, sigma = arithmetic_moments_to_lognormal_parameters(1.4, 0.8)
    expected = sum(_lognormal_log_density(value, mu, sigma) for value in coefficients[:3])

    actual = shared_coefficient_log_prior_numpy(coefficients, 3, eta, zeta)

    assert actual == pytest.approx(expected, rel=0.0, abs=1e-14)


def test_shared_hyperprior_matches_two_normals_and_is_independent_of_k() -> None:
    """The single shared hyperprior term should be independent of active k."""
    hierarchy = _hierarchy()
    eta = log(1.6)
    zeta = log(0.5)
    expected = _normal_log_density(
        eta,
        log(hierarchy.mean_hyperprior_median),
        hierarchy.mean_hyperprior_log_sd,
    ) + _normal_log_density(
        zeta,
        log(hierarchy.sd_hyperprior_median),
        hierarchy.sd_hyperprior_log_sd,
    )
    hyperprior = shared_hyperprior_log_density_numpy(eta, zeta, hierarchy)

    assert hyperprior == pytest.approx(expected, rel=0.0, abs=1e-15)
    for k in (1, 4, 17):
        coefficients = np.asarray(np.linspace(0.5, 1.5, k), dtype=np.float64)
        total = shared_coefficient_log_prior_numpy(coefficients, k, eta, zeta) + hyperprior
        coefficient_only = shared_coefficient_log_prior_numpy(coefficients, k, eta, zeta)
        assert total - coefficient_only == pytest.approx(hyperprior, rel=0.0, abs=1e-13)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mean_hyperprior_median", 0.0),
        ("mean_hyperprior_log_sd", -1.0),
        ("sd_hyperprior_median", np.inf),
        ("sd_hyperprior_log_sd", np.nan),
    ],
)
def test_hierarchy_configuration_validates_positive_finite_values(field: str, value: float) -> None:
    """Every median and log-space SD should require positive finite support."""
    values = {
        "mean_hyperprior_median": 1.2,
        "mean_hyperprior_log_sd": 0.4,
        "sd_hyperprior_median": 0.7,
        "sd_hyperprior_log_sd": 0.9,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        SharedLognormalHierarchy(**values)


def test_hierarchy_configuration_is_immutable() -> None:
    """A validated hierarchy configuration should not mutate during sampling."""
    hierarchy = _hierarchy()

    with pytest.raises(FrozenInstanceError):
        hierarchy.mean_hyperprior_median = 2.0  # type: ignore[misc]


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_coefficient_prior_rejects_values_outside_positive_finite_support(value: float) -> None:
    """Both kernels should return negative-infinite log density outside support."""
    coefficients = np.array([1.0, value, 3.0])

    numpy_result = shared_coefficient_log_prior_numpy(coefficients, 2, 0.0, 0.0)
    numba_result = shared_coefficient_log_prior_numba(coefficients, 2, 0.0, 0.0)

    assert numpy_result == numba_result == -np.inf


@pytest.mark.parametrize("k", [-1, 0, 4])
def test_coefficient_prior_rejects_invalid_active_count(k: int) -> None:
    """An unsupported active count should have negative-infinite log density."""
    coefficients = np.ones(3)

    assert shared_coefficient_log_prior_numpy(coefficients, k, 0.0, 0.0) == -np.inf
    assert shared_coefficient_log_prior_numba(coefficients, k, 0.0, 0.0) == -np.inf


@pytest.mark.parametrize(("eta", "zeta"), [(np.nan, 0.0), (0.0, np.inf)])
def test_hierarchy_state_requires_finite_log_moments(eta: float, zeta: float) -> None:
    """Nonfinite log moments should have zero conditional and hyperprior density."""
    coefficients = np.ones(2)
    hierarchy = _hierarchy()

    assert shared_coefficient_log_prior_numpy(coefficients, 2, eta, zeta) == -np.inf
    assert shared_coefficient_log_prior_numba(coefficients, 2, eta, zeta) == -np.inf
    assert shared_hyperprior_log_density_numpy(eta, zeta, hierarchy) == -np.inf


@pytest.mark.parametrize("seed", [1, 5, 17])
def test_numpy_and_numba_kernels_have_seeded_parity(seed: int) -> None:
    """NumPy and Numba kernels should agree for varied valid hierarchy states."""
    rng = np.random.default_rng(seed)
    coefficients = rng.lognormal(mean=-0.2, sigma=0.8, size=9)
    k = seed % coefficients.size + 1
    eta = float(rng.normal(0.0, 0.5))
    zeta = float(rng.normal(-0.2, 0.7))
    hierarchy = _hierarchy()

    numpy_coefficient = shared_coefficient_log_prior_numpy(coefficients, k, eta, zeta)
    numba_coefficient = shared_coefficient_log_prior_numba(coefficients, k, eta, zeta)
    numpy_hyperprior = shared_hyperprior_log_density_numpy(eta, zeta, hierarchy)
    numba_hyperprior = shared_hyperprior_log_density_numba(
        eta,
        zeta,
        hierarchy.mean_hyperprior_median,
        hierarchy.mean_hyperprior_log_sd,
        hierarchy.sd_hyperprior_median,
        hierarchy.sd_hyperprior_log_sd,
    )

    assert numba_coefficient == pytest.approx(numpy_coefficient, rel=0.0, abs=1e-12)
    assert numba_hyperprior == pytest.approx(numpy_hyperprior, rel=0.0, abs=1e-15)
