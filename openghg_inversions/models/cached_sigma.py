"""Accepted-site-sigma state quadratics for the fixed-OU likelihood."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from openghg_inversions.models.fixed_ou import (
    FixedOuLikelihoodEvaluation,
    FixedOuLowRank,
)


FloatArray = NDArray[np.float64]


def _vector(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return cast(FloatArray, array.copy())


def _matrix(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return cast(FloatArray, array.copy())


@dataclass(frozen=True)
class MarginalQuadraticCache:
    """Normalized state likelihood coefficients at one accepted site sigma."""

    constant: float
    linear: FloatArray
    precision: FloatArray
    sigma: FloatArray
    factor_cholesky_seconds: float
    factor_cholesky_operations: int
    refresh_seconds: float

    def __post_init__(self) -> None:
        linear = _vector(self.linear, "linear")
        precision = _matrix(self.precision, "precision")
        sigma = _vector(self.sigma, "sigma")
        if precision.shape != (linear.size, linear.size):
            raise ValueError(
                "precision must be square with one row per state; "
                f"got {precision.shape} for {linear.size} states."
            )
        if not np.allclose(precision, precision.T, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("precision must be symmetric.")
        if np.any(sigma < 0.0):
            raise ValueError("sigma must be non-negative.")
        if not np.isfinite(self.constant):
            raise ValueError("constant must be finite.")
        if self.factor_cholesky_seconds < 0.0 or self.refresh_seconds < 0.0:
            raise ValueError("cache timings must be non-negative.")
        if self.factor_cholesky_operations not in (0, 1):
            raise ValueError("a cache refresh must perform zero or one factor Cholesky.")
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "sigma", sigma)

    @property
    def n_state(self) -> int:
        """Number of state variables represented by the quadratic."""
        return int(self.linear.size)

    def log_likelihood(self, state: ArrayLike) -> float:
        """Evaluate the cached normalized likelihood without a factorization."""
        value = _vector(state, "state")
        if value.shape != (self.n_state,):
            raise ValueError(f"state has shape {value.shape}, expected {(self.n_state,)}.")
        return float(
            self.constant
            + self.linear @ value
            - 0.5 * value @ self.precision @ value
        )

    def gradient(self, state: ArrayLike) -> FloatArray:
        """Evaluate the exact cached gradient with respect to state."""
        value = _vector(state, "state")
        if value.shape != (self.n_state,):
            raise ValueError(f"state has shape {value.shape}, expected {(self.n_state,)}.")
        return cast(FloatArray, self.linear - self.precision @ value)


class FixedOuCachedSigmaTarget:
    """Build state quadratics from the OGI fixed-OU covariance preparation.

    ``refresh`` performs one rank-space factorization for an accepted sigma and
    solves the zero-state residual and all state-design columns together. State
    evaluations on the returned cache are then pure dense quadratic algebra.
    """

    def __init__(
        self,
        *,
        prepared: FixedOuLowRank,
        observations: ArrayLike,
        fixed_contribution: ArrayLike,
        design: ArrayLike,
    ) -> None:
        self.prepared = prepared
        self.observations = _vector(observations, "observations")
        self.fixed_contribution = _vector(fixed_contribution, "fixed_contribution")
        self.design = _matrix(design, "design")
        expected = (prepared.n_observation,)
        if self.observations.shape != expected:
            raise ValueError(
                f"observations has shape {self.observations.shape}, expected {expected}."
            )
        if self.fixed_contribution.shape != expected:
            raise ValueError(
                "fixed_contribution has shape "
                f"{self.fixed_contribution.shape}, expected {expected}."
            )
        if self.design.shape[0] != prepared.n_observation:
            raise ValueError("design must have one row per observation.")
        self._residual_at_zero = self.observations - self.fixed_contribution

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self.prepared.n_observation

    @property
    def n_state(self) -> int:
        """Number of state-design columns."""
        return int(self.design.shape[1])

    @property
    def n_group(self) -> int:
        """Number of independently inferred site amplitudes."""
        return self.prepared.n_site

    def refresh(self, sigma: ArrayLike | float) -> MarginalQuadraticCache:
        """Build one exact float64 quadratic for an accepted sigma."""
        start = time.perf_counter()
        sigma_value = np.asarray(sigma, dtype=np.float64)
        if sigma_value.ndim == 0:
            sigma_value = np.full(self.n_group, sigma_value.item(), dtype=np.float64)
        elif sigma_value.shape != (self.n_group,):
            raise ValueError(
                f"sigma must be scalar or have shape {(self.n_group,)}, got {sigma_value.shape}."
            )
        rhs = np.column_stack((self._residual_at_zero, self.design))
        covariance_solve = self.prepared.solve(rhs, sigma_value)
        solved_residual = covariance_solve.solution[:, 0]
        solved_design = covariance_solve.solution[:, 1:]
        precision = self.design.T @ solved_design
        precision = (precision + precision.T) * 0.5
        linear = self.design.T @ solved_residual
        constant = -0.5 * (
            self.n_obs * math.log(2.0 * math.pi)
            + covariance_solve.logdet
            + float(self._residual_at_zero @ solved_residual)
        )
        return MarginalQuadraticCache(
            constant=constant,
            linear=cast(FloatArray, linear),
            precision=cast(FloatArray, precision),
            sigma=cast(FloatArray, sigma_value),
            factor_cholesky_seconds=covariance_solve.factor_cholesky_seconds,
            factor_cholesky_operations=covariance_solve.factor_cholesky_operations,
            refresh_seconds=time.perf_counter() - start,
        )

    def evaluate_from_residual(
        self,
        residual: ArrayLike,
        sigma: ArrayLike | float,
    ) -> FixedOuLikelihoodEvaluation:
        """Evaluate the exact sigma conditional for a fixed state residual."""
        return self.prepared.evaluate(residual, sigma)

    def log_likelihood(self, state: ArrayLike, sigma: ArrayLike | float) -> float:
        """Evaluate the uncached exact likelihood for output construction."""
        state_value = _vector(state, "state")
        if state_value.shape != (self.n_state,):
            raise ValueError(f"state has shape {state_value.shape}, expected {(self.n_state,)}.")
        residual = self._residual_at_zero - self.design @ state_value
        return self.evaluate_from_residual(residual, sigma).log_likelihood

    def random(
        self,
        state: ArrayLike,
        sigma: ArrayLike | float,
        *,
        rng: np.random.Generator,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Draw a joint observation vector at one state and sigma."""
        state_value = _vector(state, "state")
        if state_value.shape != (self.n_state,):
            raise ValueError(f"state has shape {state_value.shape}, expected {(self.n_state,)}.")
        mean = self.fixed_contribution + self.design @ state_value
        return self.prepared.random(mean, np.asarray(sigma), rng=rng, size=size)
