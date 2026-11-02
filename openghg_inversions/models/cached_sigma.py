"""State-likelihood quadratics for supplied site amplitudes in the fixed-OU likelihood."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_solve
from scipy.linalg.lapack import dpocon  # type: ignore[attr-defined]  # Generated SciPy wrapper.

from openghg_inversions.models.fixed_ou import (
    FixedOuLikelihoodEvaluation,
    FixedOuLowRank,
    _site_values,
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


def _sum_squares_gram(factor: FloatArray, rhs: FloatArray) -> tuple[FloatArray, float] | None:
    """Return a whitened covariance Gram matrix, or reject its numerical screens.

    The ridge solution ``Z`` gives ``G = (B - W Z).T (B - W Z) + Z.T Z``.
    Screen core conditioning and residual rounding before accepting this form.
    The rounding budget bounds Gram errors relative to column energies, not
    relative to individual cross coefficients or the total latent-solve error.
    """
    core = factor.T @ factor
    core.flat[:: core.shape[0] + 1] += 1.0
    core = (core + core.T) * 0.5
    if not np.isfinite(core).all():
        return None
    try:
        cholesky = np.linalg.cholesky(core)
    except np.linalg.LinAlgError:
        return None
    eps = np.finfo(np.float64).eps
    budget = np.sqrt(eps)
    reciprocal_condition, info = dpocon(cholesky, np.linalg.norm(core, 1), uplo="L")
    if info or not reciprocal_condition > budget:
        return None
    latent = cho_solve((cholesky, True), factor.T @ rhs, check_finite=False)
    conditional = rhs - factor @ latent
    gamma = (factor.shape[1] + 1) * eps / (1.0 - (factor.shape[1] + 1) * eps)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        factor_norm = np.sqrt(np.einsum("ij,ij->", factor, factor))
        rhs_norm = np.sqrt(np.einsum("ij,ij->j", rhs, rhs))
        latent_norm = np.sqrt(np.einsum("ij,ij->j", latent, latent))
        conditional_norm = np.sqrt(np.einsum("ij,ij->j", conditional, conditional))
        root_energy = np.hypot(conditional_norm, latent_norm)
        delta = gamma * (rhs_norm + factor_norm * latent_norm)
        relative = np.divide(delta, root_energy, out=np.zeros_like(delta), where=root_energy > 0)
        relative[
            ((root_energy == 0) & (delta > 0))
            | ~np.isfinite(delta)
            | ~np.isfinite(root_energy)
        ] = np.inf
        worst_relative = float(np.max(relative, initial=0.0))
        gram_roundoff_bound = 2.0 * worst_relative + worst_relative * worst_relative
    # shortcut: these screens do not certify latent-solve error; revisit for new unstable cases.
    if not np.isfinite(gram_roundoff_bound) or gram_roundoff_bound > budget:
        return None
    gram = conditional.T @ conditional + latent.T @ latent
    logdet = float(2.0 * np.log(np.diag(cholesky)).sum())
    if not np.isfinite(gram).all() or not np.isfinite(logdet):
        return None
    return cast(FloatArray, gram), logdet


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

    Construction transforms the fixed residual and design into prepared mode
    coordinates once.
    ``refresh`` forms their joint Gram matrix for the supplied site amplitudes,
    using a screened sum of squares for low-rank factors and the fixed-OU
    covariance solver otherwise. State evaluations on the returned cache are
    then pure dense quadratic algebra.
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
        self._residual_at_zero = _vector(
            self.observations - self.fixed_contribution, "zero-state residual"
        )
        self._transformed_rhs = cast(
            FloatArray,
            prepared.mode_transform @ np.column_stack((self._residual_at_zero, self.design)),
        )

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
        gram_result = None
        if 0 < self.prepared.rank < self.n_obs:
            amplitude = _site_values(sigma_value, self.n_group, "site_amplitude", positive=False)
            weights, logdet = self.prepared._mode_weights_and_logdet(amplitude)
            scale = np.sqrt(weights)[:, None]
            gram_result = _sum_squares_gram(
                scale * self.prepared.transformed_factor,
                scale * self._transformed_rhs,
            )
        if gram_result is None:
            covariance_solve = self.prepared._solve_transformed(self._transformed_rhs, sigma_value)
            quadratic = self._transformed_rhs.T @ covariance_solve.solution
            logdet = covariance_solve.logdet
        else:
            quadratic, logdet_update = gram_result
            logdet += logdet_update
        precision = quadratic[1:, 1:]
        precision = (precision + precision.T) * 0.5
        linear = quadratic[1:, 0].copy()
        constant = -0.5 * (
            self.n_obs * math.log(2.0 * math.pi)
            + logdet
            + float(quadratic[0, 0])
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

    def _evaluate_from_transformed_residual(
        self,
        transformed_residual: FloatArray,
        sigma: ArrayLike | float,
    ) -> tuple[float, FloatArray]:
        """Evaluate only the amplitude conditional from an installed mode residual."""
        log_likelihood, _, gradient_amplitude = self.prepared._evaluate(
            transformed_residual, sigma, amplitude_only=True,
        )
        return log_likelihood, gradient_amplitude

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
