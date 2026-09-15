"""State-likelihood quadratics for supplied site amplitudes in the fixed-OU likelihood."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    """Normalized state-likelihood coefficients at one site-amplitude vector."""

    constant: float
    linear: FloatArray
    precision: FloatArray
    sigma: FloatArray

    @property
    def n_state(self) -> int:
        """Number of state variables represented by the quadratic."""
        return int(self.linear.size)

    def log_likelihood(self, state: ArrayLike) -> float:
        """Evaluate the cached normalized likelihood without a factorization."""
        value = np.asarray(state)
        return float(
            self.constant
            + self.linear @ value
            - 0.5 * value @ self.precision @ value
        )

    def gradient(self, state: ArrayLike) -> FloatArray:
        """Evaluate the exact cached gradient with respect to state."""
        value = np.asarray(state)
        return cast(FloatArray, self.linear - self.precision @ value)


class FixedOuCachedSigmaTarget:
    """Build state quadratics from the OGI fixed-OU covariance preparation.

    ``refresh`` performs one rank-space factorization for the supplied site
    amplitudes and solves the zero-state residual and all state-design columns
    together. State evaluations on the returned cache are then pure dense
    quadratic algebra.
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
        """Build one exact float64 quadratic for the supplied site amplitudes."""
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
            sigma=cast(FloatArray, sigma_value.copy()),
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

    def log_likelihood_from_mean(
        self,
        mean: ArrayLike,
        sigma: ArrayLike | float,
    ) -> float:
        """Evaluate the exact likelihood from a completed observation mean."""
        mean_value = _vector(mean, "mean")
        if mean_value.shape != (self.n_obs,):
            raise ValueError(
                f"mean has shape {mean_value.shape}, expected {(self.n_obs,)}."
            )
        return self.evaluate_from_residual(
            self.observations - mean_value,
            sigma,
        ).log_likelihood

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

    def random_from_mean(
        self,
        mean: ArrayLike,
        sigma: ArrayLike | float,
        *,
        rng: np.random.Generator,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Draw a joint observation vector from a completed observation mean."""
        mean_value = _vector(mean, "mean")
        if mean_value.shape != (self.n_obs,):
            raise ValueError(
                f"mean has shape {mean_value.shape}, expected {(self.n_obs,)}."
            )
        return self.prepared.random(
            mean_value,
            np.asarray(sigma),
            rng=rng,
            size=size,
        )
