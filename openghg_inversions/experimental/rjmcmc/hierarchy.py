"""Shared lognormal hierarchy for experimental trans-dimensional MCMC.

This module implements a fixed-dimensional hyperparameter layer for a
trans-dimensional coefficient state.  All active dynamic coefficients share
one arithmetic lognormal mean ``M`` and standard deviation ``S``.  This is
complete pooling of the hyperparameters and induces partial pooling of the
conditionally independent coefficients.  The sampler state stores
``eta = log(M)`` and ``zeta = log(S)``, so the hyperprior is evaluated as a
pair of normalized Normal log densities in those state coordinates.

Callers must place only active dynamic coefficients in the first ``k`` slots;
fixed outer-region coefficients are outside this hierarchy.  The numerical
kernels are independent of the structural dimension except for the number of
active coefficients whose conditional prior is evaluated.  The shared
hyperprior term must be added exactly once by the posterior assembly caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, log1p, pi, sqrt

from numba import njit
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_LOG_TWO_PI = log(2.0 * pi)


def _require_positive_finite(value: float, *, name: str) -> float:
    """Return ``value`` as a float after validating positive finite support."""
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return result


@dataclass(frozen=True)
class SharedLognormalHierarchy:
    """Immutable configuration for one shared dynamic-coefficient prior pool.

    Each hyperprior is specified as a lognormal distribution on a positive
    arithmetic moment.  Because the MCMC state is ``eta = log(M)`` and
    ``zeta = log(S)``, its median determines the corresponding Normal location
    and ``*_log_sd`` is the corresponding Normal standard deviation.

    Args:
        mean_hyperprior_median: Median of the lognormal hyperprior for the
            shared coefficient arithmetic mean ``M``.
        mean_hyperprior_log_sd: Standard deviation of ``eta`` under its Normal
            hyperprior.
        sd_hyperprior_median: Median of the lognormal hyperprior for the shared
            coefficient arithmetic standard deviation ``S``.
        sd_hyperprior_log_sd: Standard deviation of ``zeta`` under its Normal
            hyperprior.

    Raises:
        ValueError: If any configured median or log-space standard deviation
            is nonfinite or not strictly positive.
    """

    mean_hyperprior_median: float
    mean_hyperprior_log_sd: float
    sd_hyperprior_median: float
    sd_hyperprior_log_sd: float

    def __post_init__(self) -> None:
        """Validate and normalize all scalar configuration values."""
        for name in (
            "mean_hyperprior_median",
            "mean_hyperprior_log_sd",
            "sd_hyperprior_median",
            "sd_hyperprior_log_sd",
        ):
            object.__setattr__(
                self,
                name,
                _require_positive_finite(getattr(self, name), name=name),
            )

    @property
    def eta_hyperprior_mean(self) -> float:
        """Return the Normal location of the shared log-mean state.

        Returns:
            Normal location of ``eta`` in log arithmetic-mean coordinates.
        """
        return log(self.mean_hyperprior_median)

    @property
    def zeta_hyperprior_mean(self) -> float:
        """Return the Normal location of the shared log-SD state.

        Returns:
            Normal location of ``zeta`` in log arithmetic-SD coordinates.
        """
        return log(self.sd_hyperprior_median)


def arithmetic_moments_to_log_state(mean: float, standard_deviation: float) -> tuple[float, float]:
    """Convert positive arithmetic moments ``(M, S)`` to ``(eta, zeta)``.

    Args:
        mean: Arithmetic mean ``M`` of the conditional coefficient prior.
        standard_deviation: Arithmetic standard deviation ``S`` of that prior.

    Returns:
        ``(log(M), log(S))`` for use as hierarchy state variables.

    Raises:
        ValueError: If either arithmetic moment is nonfinite or nonpositive.
    """
    valid_mean = _require_positive_finite(mean, name="mean")
    valid_sd = _require_positive_finite(standard_deviation, name="standard_deviation")
    return log(valid_mean), log(valid_sd)


def arithmetic_moments_to_lognormal_parameters(
    mean: float,
    standard_deviation: float,
) -> tuple[float, float]:
    """Convert arithmetic moments to lognormal log-location and log-scale.

    Args:
        mean: Positive arithmetic mean ``M``.
        standard_deviation: Positive arithmetic standard deviation ``S``.

    Returns:
        The exact ``(mu_log, sigma_log)`` parameters of the corresponding
        lognormal distribution.

    Raises:
        ValueError: If either arithmetic moment is nonfinite or nonpositive.
    """
    eta, zeta = arithmetic_moments_to_log_state(mean, standard_deviation)
    return log_moments_to_lognormal_parameters(eta, zeta)


def _softplus(value: float) -> float:
    """Evaluate ``log(1 + exp(value))`` without avoidable overflow."""
    if value > 0.0:
        return value + log1p(exp(-value))
    return log1p(exp(value))


def log_moments_to_lognormal_parameters(eta: float, zeta: float) -> tuple[float, float]:
    """Convert log arithmetic moments to lognormal distribution parameters.

    The stable calculation is algebraically identical to
    ``sigma_log**2 = log(1 + (S / M)**2)`` and
    ``mu_log = log(M) - sigma_log**2 / 2``.

    Args:
        eta: Logarithm of the arithmetic mean ``M``.
        zeta: Logarithm of the arithmetic standard deviation ``S``.

    Returns:
        The exact ``(mu_log, sigma_log)`` parameters of the conditional
        lognormal coefficient distribution.

    Raises:
        ValueError: If either state coordinate is nonfinite or if the implied
            log-scale is not representable as a positive finite float.
    """
    eta_float = float(eta)
    zeta_float = float(zeta)
    if not isfinite(eta_float) or not isfinite(zeta_float):
        raise ValueError("eta and zeta must be finite.")
    sigma_squared = _softplus(2.0 * (zeta_float - eta_float))
    sigma = sqrt(sigma_squared)
    mu = eta_float - 0.5 * sigma_squared
    if not isfinite(mu) or not isfinite(sigma) or sigma <= 0.0:
        raise ValueError("eta and zeta must imply representable lognormal parameters.")
    return mu, sigma


def shared_coefficient_log_prior_numpy(
    coefficients: FloatArray,
    k: int,
    eta: float,
    zeta: float,
) -> float:
    """Return the normalized conditional log-prior for active coefficients.

    Args:
        coefficients: One-dimensional padded coefficient vector whose first
            ``k`` entries are active dynamic coefficients.
        k: Number of active dynamic coefficients, satisfying
            ``1 <= k <= coefficients.size``.
        eta: Logarithm of their shared arithmetic prior mean.
        zeta: Logarithm of their shared arithmetic prior standard deviation.

    Returns:
        Sum of normalized lognormal log densities, or negative infinity when
        the active count, coefficient support, or hierarchy state is invalid.
    """
    if coefficients.ndim != 1 or k < 1 or k > coefficients.size:
        return -np.inf
    active = coefficients[:k]
    if np.any(active <= 0.0) or not np.all(np.isfinite(active)):
        return -np.inf
    try:
        mu, sigma = log_moments_to_lognormal_parameters(eta, zeta)
    except ValueError:
        return -np.inf
    standardized = (np.log(active) - mu) / sigma
    return float(
        -0.5 * np.dot(standardized, standardized)
        - np.log(active).sum()
        - k * log(sigma)
        - 0.5 * k * _LOG_TWO_PI
    )


@njit(cache=True)
def shared_coefficient_log_prior_numba(
    coefficients: FloatArray,
    k: int,
    eta: float,
    zeta: float,
) -> float:
    """Return the normalized shared conditional coefficient log-prior with Numba.

    Args:
        coefficients: One-dimensional padded coefficient vector whose first
            ``k`` entries are active dynamic coefficients.
        k: Number of active dynamic coefficients, satisfying
            ``1 <= k <= coefficients.size``.
        eta: Logarithm of their shared arithmetic prior mean.
        zeta: Logarithm of their shared arithmetic prior standard deviation.

    Returns:
        Sum of normalized lognormal log densities, or negative infinity for an
        invalid active count, coefficient, or hierarchy state.
    """
    if (
        coefficients.ndim != 1
        or k < 1
        or k > coefficients.size
        or not np.isfinite(eta)
        or not np.isfinite(zeta)
    ):
        return -np.inf
    twice_log_ratio = 2.0 * (zeta - eta)
    if twice_log_ratio > 0.0:
        sigma_squared = twice_log_ratio + np.log1p(np.exp(-twice_log_ratio))
    else:
        sigma_squared = np.log1p(np.exp(twice_log_ratio))
    sigma = np.sqrt(sigma_squared)
    mu = eta - 0.5 * sigma_squared
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0.0:
        return -np.inf
    result = -k * np.log(sigma) - 0.5 * k * np.log(2.0 * np.pi)
    for index in range(k):
        value = coefficients[index]
        if value <= 0.0 or not np.isfinite(value):
            return -np.inf
        standardized = (np.log(value) - mu) / sigma
        result -= 0.5 * standardized * standardized + np.log(value)
    return result


def shared_hyperprior_log_density_numpy(
    eta: float,
    zeta: float,
    hierarchy: SharedLognormalHierarchy,
) -> float:
    """Return the normalized shared hyperprior log density in state coordinates.

    Args:
        eta: Logarithm of the shared arithmetic coefficient-prior mean.
        zeta: Logarithm of the shared arithmetic coefficient-prior SD.
        hierarchy: Validated one-pool hyperprior configuration.

    Returns:
        The sum of the two normalized Normal log densities in ``eta`` and
        ``zeta``, or negative infinity outside finite state support.
    """
    if not isfinite(eta) or not isfinite(zeta):
        return -np.inf
    eta_standardized = (eta - hierarchy.eta_hyperprior_mean) / hierarchy.mean_hyperprior_log_sd
    zeta_standardized = (zeta - hierarchy.zeta_hyperprior_mean) / hierarchy.sd_hyperprior_log_sd
    return float(
        -0.5 * eta_standardized**2
        - log(hierarchy.mean_hyperprior_log_sd)
        - 0.5 * _LOG_TWO_PI
        - 0.5 * zeta_standardized**2
        - log(hierarchy.sd_hyperprior_log_sd)
        - 0.5 * _LOG_TWO_PI
    )


@njit(cache=True)
def shared_hyperprior_log_density_numba(
    eta: float,
    zeta: float,
    mean_hyperprior_median: float,
    mean_hyperprior_log_sd: float,
    sd_hyperprior_median: float,
    sd_hyperprior_log_sd: float,
) -> float:
    """Return the normalized shared hyperprior log density with Numba.

    Args:
        eta: Logarithm of the shared arithmetic coefficient-prior mean.
        zeta: Logarithm of the shared arithmetic coefficient-prior SD.
        mean_hyperprior_median: Median of the positive mean hyperprior.
        mean_hyperprior_log_sd: Normal SD of ``eta``.
        sd_hyperprior_median: Median of the positive SD hyperprior.
        sd_hyperprior_log_sd: Normal SD of ``zeta``.

    Returns:
        The sum of normalized Normal log densities in ``eta`` and ``zeta``, or
        negative infinity for invalid inputs.  No Jacobian term is required
        because these logarithms, rather than ``M`` and ``S``, are the state.
    """
    if (
        not np.isfinite(eta)
        or not np.isfinite(zeta)
        or not np.isfinite(mean_hyperprior_median)
        or mean_hyperprior_median <= 0.0
        or not np.isfinite(mean_hyperprior_log_sd)
        or mean_hyperprior_log_sd <= 0.0
        or not np.isfinite(sd_hyperprior_median)
        or sd_hyperprior_median <= 0.0
        or not np.isfinite(sd_hyperprior_log_sd)
        or sd_hyperprior_log_sd <= 0.0
    ):
        return -np.inf
    eta_standardized = (eta - np.log(mean_hyperprior_median)) / mean_hyperprior_log_sd
    zeta_standardized = (zeta - np.log(sd_hyperprior_median)) / sd_hyperprior_log_sd
    return (
        -0.5 * eta_standardized * eta_standardized
        - np.log(mean_hyperprior_log_sd)
        - 0.5 * np.log(2.0 * np.pi)
        - 0.5 * zeta_standardized * zeta_standardized
        - np.log(sd_hyperprior_log_sd)
        - 0.5 * np.log(2.0 * np.pi)
    )
