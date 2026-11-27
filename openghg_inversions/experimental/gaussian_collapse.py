"""Dense reference for Gaussian collapse and covariance hyperparameter inference.

This module ports the numerical core of the OPE-188 linked CO₂–O₂ experiment.
The fixed observation covariance contains all integrated Gaussian terms,
including observation error. A separate covariance supplies the term being
fitted. All covariance and cross-covariance matrices use a common observation
row order. This reference factors the full observation covariance on each
evaluation, with cubic time and quadratic storage in the number of observations.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp
from scipy.stats import multivariate_t

FloatArray = NDArray[np.float64]
_LogDensity = Callable[[FloatArray], float]
_LogDensityAndGradient = Callable[[FloatArray], tuple[float, FloatArray]]


def _symmetric(matrix: FloatArray, size: int, name: str) -> None:
    """Check dimensions and symmetry at the dense input boundary."""
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite matrix of shape ({size}, {size})")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")


class GaussianCollapse:
    """Conditional Gaussian for user-selected state or flux quantities.

    Args:
        innovation: Observations minus their complete prior mean.
        fixed_covariance: Observation covariance excluding the term being fitted.
        projected_mean: Prior mean of the quantities to reconstruct.
        projected_covariance: Prior covariance of those quantities.
        cross_covariance: Prior covariance between those quantities and the
            observations, with observations on the second axis.
    """

    def __init__(
        self,
        innovation: ArrayLike,
        fixed_covariance: ArrayLike,
        projected_mean: ArrayLike,
        projected_covariance: ArrayLike,
        cross_covariance: ArrayLike,
    ) -> None:
        self._innovation = np.array(innovation, dtype=float)
        self._fixed = np.array(fixed_covariance, dtype=float)
        self._mean = np.array(projected_mean, dtype=float)
        self._covariance = np.array(projected_covariance, dtype=float)
        self._cross = np.array(cross_covariance, dtype=float)
        n, p = self._innovation.size, self._mean.size
        if n == 0 or self._innovation.shape != (n,) or self._mean.shape != (p,):
            raise ValueError("innovation and projected_mean must be vectors with observations present")
        if self._cross.shape != (p, n):
            raise ValueError(f"cross_covariance must have shape ({p}, {n})")
        if not all(np.isfinite(value).all() for value in (self._innovation, self._mean, self._cross)):
            raise ValueError("innovation, projected_mean and cross_covariance must be finite")
        _symmetric(self._fixed, n, "fixed_covariance")
        _symmetric(self._covariance, p, "projected_covariance")

    def _factor(self, covariance: ArrayLike) -> tuple[tuple[FloatArray, bool], FloatArray, float]:
        """Factor the observation covariance and evaluate the normalized evidence."""
        covariance = np.asarray(covariance, dtype=float)
        n = self._innovation.size
        _symmetric(covariance, n, "covariance")
        factor = cho_factor(self._fixed + covariance, lower=True)
        solved = cho_solve(factor, self._innovation)
        log_det = 2 * np.log(np.diag(factor[0])).sum()
        log_evidence = -0.5 * (self._innovation @ solved + log_det + n * np.log(2 * np.pi))
        return factor, solved, float(log_evidence)

    def log_evidence(self, covariance: ArrayLike) -> float:
        """Evaluate the exact Gaussian marginal log likelihood.

        Args:
            covariance: Added observation covariance. The sum with the fixed
                covariance must be positive definite.

        Returns:
            Log density of the observed innovation.
        """
        return self._factor(covariance)[2]

    def log_evidence_and_gradient(
        self, covariance: ArrayLike, contract_gradient: Callable[[FloatArray], FloatArray]
    ) -> tuple[float, FloatArray]:
        """Evaluate the exact marginal likelihood and its covariance gradient.

        Args:
            covariance: Added observation covariance.
            contract_gradient: Function contracting the symmetric derivative
                with respect to covariance entries against each parameter's
                covariance derivative.

        Returns:
            Log evidence and gradient in fitted parameter coordinates.
        """
        factor, solved, evidence = self._factor(covariance)
        precision = cho_solve(factor, np.eye(self._innovation.size))
        covariance_score = 0.5 * (np.outer(solved, solved) - precision)
        return evidence, contract_gradient(covariance_score)

    def posterior(self, covariance: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Condition the projected quantities on the observations.

        Args:
            covariance: Added observation covariance. The sum with the fixed
                covariance must be positive definite.

        Returns:
            Conditional mean and covariance.
        """
        mean, posterior_covariance, _ = self.posterior_and_log_evidence(covariance)
        return mean, posterior_covariance

    def posterior_and_log_evidence(self, covariance: ArrayLike) -> tuple[FloatArray, FloatArray, float]:
        """Evaluate the conditional Gaussian and evidence with a single factorization.

        Args:
            covariance: Added observation covariance. The sum with the fixed
                covariance must be positive definite.

        Returns:
            Conditional mean, conditional covariance and log evidence.
        """
        factor, solved, evidence = self._factor(covariance)
        mean = self._mean + self._cross @ solved
        posterior_covariance = self._covariance - self._cross @ cho_solve(factor, self._cross.T)
        return mean, (posterior_covariance + posterior_covariance.T) / 2, evidence


class PersistentPairOU:
    """Dense independent-site covariance with persistent CO₂–O₂ coupling.

    Each covariance entry within a site is the product of its two amplitudes
    and exp(-absolute time lag / timescale), multiplied by rho for cross-gas
    entries. The processes have a shared decay and stationary correlation,
    including across different observation times.

    Args:
        hours: Observation times in hours, in the supplied observation row order.
        site: Site labels in observation row order. Amplitude parameters use
            sorted unique site labels within each gas.
        species: Gas labels in observation row order, each 'co2' or 'o2'.
            Each site must contain both gases.
        rho: Fixed correlation, or None to fit atanh(rho).
        timescale: Fixed decay in hours, or None to fit log(timescale).
    """

    def __init__(
        self,
        hours: ArrayLike,
        site: ArrayLike,
        species: ArrayLike,
        *,
        rho: float | None = None,
        timescale: float | None = None,
    ) -> None:
        hours = np.asarray(hours, dtype=float)
        site = np.asarray(site, dtype=str)
        species = np.asarray(species, dtype=str)
        if hours.ndim != 1 or hours.size == 0 or not np.isfinite(hours).all():
            raise ValueError("hours must be a nonempty finite vector")
        if site.shape != hours.shape or species.shape != hours.shape:
            raise ValueError("site and species must match hours")
        if not np.isin(species, ("co2", "o2")).all():
            raise ValueError("species must be 'co2' or 'o2'")
        if rho is not None and not -1 < rho < 1:
            raise ValueError("rho must be strictly between -1 and 1")
        if timescale is not None and not (np.isfinite(timescale) and timescale > 0):
            raise ValueError("timescale must be finite and positive")
        sites, site_index = np.unique(site, return_inverse=True)
        self._sites = tuple(sites.tolist())
        self._n_sites = sites.size
        self._n_rows = hours.size
        self._rho = rho
        self._timescale = timescale
        self._blocks = []
        for index in range(self._n_sites):
            rows = np.flatnonzero(site_index == index)
            oxygen = species[rows] == "o2"
            if oxygen.all() or not oxygen.any():
                raise ValueError(f"site {sites[index]!r} must contain both co2 and o2 observations")
            self._blocks.append(
                (
                    rows,
                    index + self._n_sites * oxygen,
                    oxygen[:, None] != oxygen,
                    np.abs(hours[rows, None] - hours[rows]),
                )
            )

    @property
    def sites(self) -> tuple[str, ...]:
        """Sorted site labels in amplitude parameter order for each gas."""
        return self._sites

    def reference_log_prior_and_gradient(self, theta: ArrayLike) -> tuple[float, FloatArray]:
        """Evaluate the OPE-188 hyperpriors and their transformed gradient.

        Log amplitudes have independent Normal(log(2.5), 1) priors. If free,
        log timescale has a Normal(log(5), 1) prior and (1 + rho) / 2 has a
        Beta(2, 2) prior with the Fisher-z change of variables.

        Args:
            theta: Transformed parameters in the same order as __call__.

        Returns:
            Normalized log prior and gradient in fitted parameter coordinates.
        """
        theta = self._parameters(theta)
        amplitude_count = 2 * self._n_sites
        gradient = np.zeros_like(theta)
        delta = theta[:amplitude_count] - np.log(2.5)
        log_prior = -0.5 * (delta @ delta + delta.size * np.log(2 * np.pi))
        gradient[:amplitude_count] = -delta
        if self._rho is None:
            eta = theta[amplitude_count]
            log_u = -np.logaddexp(0, -2 * eta)
            log_one_minus_u = -np.logaddexp(0, 2 * eta)
            log_prior += np.log(12) + 2 * (log_u + log_one_minus_u)
            gradient[amplitude_count] = -4 * np.tanh(eta)
        if self._timescale is None:
            delta_time = theta[-1] - np.log(5)
            log_prior -= 0.5 * (delta_time**2 + np.log(2 * np.pi))
            gradient[-1] = -delta_time
        return float(log_prior), gradient

    def _parameters(self, theta: ArrayLike) -> FloatArray:
        """Check the transformed vector layout and finite coordinates."""
        theta = np.asarray(theta, dtype=float)
        n_parameters = 2 * self._n_sites + int(self._rho is None) + int(self._timescale is None)
        if theta.shape != (n_parameters,) or not np.isfinite(theta).all():
            raise ValueError(f"theta must be a finite vector of length {n_parameters}")
        return theta

    def _unpack(self, theta: ArrayLike) -> tuple[FloatArray, float, float]:
        """Resolve log amplitudes, Fisher-z correlation and log decay."""
        theta = self._parameters(theta)
        n_amplitude = 2 * self._n_sites
        rho = np.tanh(theta[n_amplitude]) if self._rho is None else self._rho
        with np.errstate(over="ignore", under="ignore"):
            amplitude = np.exp(theta[:n_amplitude])
            timescale = np.exp(theta[-1]) if self._timescale is None else self._timescale
        if not (
            np.all((amplitude > 0) & (amplitude <= np.sqrt(np.finfo(float).max))) and 0 < timescale < np.inf
        ):
            raise ValueError("amplitudes and timescale must be representable in the dense covariance")
        return amplitude, float(rho), float(timescale)

    def __call__(self, theta: ArrayLike) -> FloatArray:
        """Build the covariance at the given transformed parameters.

        Args:
            theta: Log CO₂ amplitudes by site, then log O₂ amplitudes by site,
                then atanh(rho) and log(timescale) when free, in that order.

        Returns:
            Covariance in the original observation row order.
        """
        amplitude, rho, timescale = self._unpack(theta)
        covariance = np.zeros((self._n_rows, self._n_rows))
        for rows, parameters, cross_gas, lag in self._blocks:
            scale = amplitude[parameters]
            covariance[np.ix_(rows, rows)] = (
                np.outer(scale, scale) * np.where(cross_gas, rho, 1.0) * np.exp(-lag / timescale)
            )
        return covariance

    def contract_gradient(self, theta: ArrayLike, covariance_score: FloatArray) -> FloatArray:
        """Contract a symmetric covariance score with exact parameter derivatives.

        Args:
            theta: Transformed parameters in the same order as __call__.
            covariance_score: Symmetric score in observation row order, supplied
                by GaussianCollapse.log_evidence_and_gradient.

        Returns:
            One contracted derivative per element of theta.
        """
        amplitude, rho, timescale = self._unpack(theta)
        n_amplitude = amplitude.size
        gradient = np.zeros(n_amplitude + int(self._rho is None) + int(self._timescale is None))
        for rows, parameters, cross_gas, lag in self._blocks:
            scale = amplitude[parameters]
            decay_lags = lag / timescale
            kernel = np.outer(scale, scale) * np.exp(-decay_lags)
            score_kernel = covariance_score[np.ix_(rows, rows)] * kernel
            score_covariance = score_kernel * np.where(cross_gas, rho, 1.0)
            # Each log amplitude contributes once per covariance axis.
            np.add.at(gradient, parameters, 2 * score_covariance.sum(axis=1))
            if self._rho is None:
                gradient[n_amplitude] += (1 - rho**2) * score_kernel[cross_gas].sum()
            if self._timescale is None:
                gradient[-1] += np.sum(score_covariance * decay_lags)
        return gradient


def fit_log_density(
    log_density: _LogDensity,
    initial: ArrayLike,
    bounds: Sequence[tuple[float, float]],
    *,
    log_density_and_gradient: _LogDensityAndGradient | None = None,
) -> OptimizeResult:
    """Fit covariance parameters by type-II ML or MAP.

    Args:
        log_density: Collapsed log likelihood for ML, plus log prior in the
            fitted parameter coordinates for MAP.
        initial: Starting point in fitted parameter coordinates.
        bounds: Lower and upper numerical optimization bounds for each parameter.
        log_density_and_gradient: Optional joint evaluator for the same log
            density and its analytic gradient.

    Returns:
        SciPy optimizer result, including the mode and convergence status.
        The minimized objective is the negative log density.
    """
    if log_density_and_gradient is None:
        return minimize(lambda theta: -log_density(theta), initial, method="L-BFGS-B", bounds=bounds)

    def objective(theta: FloatArray) -> tuple[float, FloatArray]:
        """Return the negative objective and gradient for minimization."""
        value, gradient = log_density_and_gradient(theta)
        return -value, -gradient

    return minimize(objective, initial, method="L-BFGS-B", bounds=bounds, jac=True)


@dataclass(frozen=True)
class CovarianceImportanceSamples:
    """Weighted draws from a covariance hyperparameter posterior.

    Attributes:
        theta: Proposal draws in fitted parameter coordinates.
        weights: Normalized importance weights in draw order.
        effective_sample_size: Inverse sum of squared normalized weights.
    """

    theta: FloatArray
    weights: FloatArray
    effective_sample_size: float


def laplace_importance(
    log_density: _LogDensity,
    mode: ArrayLike,
    draws: int,
    rng: np.random.Generator,
    *,
    step: float = 1e-3,
    log_density_and_gradient: _LogDensityAndGradient | None = None,
) -> CovarianceImportanceSamples:
    """Correct a local Laplace approximation with importance sampling.

    A Student-t proposal with five degrees of freedom and scale 1.2² times the
    inverse observed curvature has heavier tails than the Gaussian approximation.

    Args:
        log_density: Collapsed log likelihood plus log prior on the fitted
            scale. Return negative infinity for samples outside prior support.
        mode: Converged interior MAP estimate.
        draws: Number of independent proposal draws, at least two.
        rng: Random number generator.
        step: Central-difference step in fitted parameter coordinates.
        log_density_and_gradient: Optional joint evaluator for the same log
            density and its gradient. With a gradient, curvature requires only
            two evaluations per parameter.

    Returns:
        Weighted draws and the importance effective sample size.

    Raises:
        ValueError: If the inputs are invalid, the density is nonfinite near the
            mode, curvature is not positive definite, or all weights are zero.
    """
    mode = np.asarray(mode, dtype=float)
    if mode.ndim != 1 or not mode.size or not np.isfinite(mode).all():
        raise ValueError("mode must be a nonempty finite vector")
    if draws < 2 or not (np.isfinite(step) and step > 0):
        raise ValueError("at least two draws and a positive finite step are required")
    offsets = np.eye(mode.size) * step
    if log_density_and_gradient is not None:
        columns = []
        for offset in offsets:
            lower, lower_gradient = log_density_and_gradient(mode - offset)
            upper, upper_gradient = log_density_and_gradient(mode + offset)
            if not np.isfinite([lower, upper]).all():
                raise ValueError("log density must be finite around an interior mode")
            columns.append((lower_gradient - upper_gradient) / (2 * step))
        curvature = np.column_stack(columns)
        curvature = (curvature + curvature.T) / 2
    else:
        curvature = _numerical_curvature(log_density, mode, offsets, step)
    if not np.isfinite(curvature).all():
        raise ValueError("log density must be finite around an interior mode")
    try:
        factor = cho_factor(curvature, lower=True)
    except np.linalg.LinAlgError as error:
        raise ValueError("Laplace approximation requires positive definite curvature") from error
    scale = 1.2**2 * cho_solve(factor, np.eye(mode.size))
    proposal = multivariate_t(loc=mode, shape=scale, df=5)
    theta = np.asarray(proposal.rvs(size=draws, random_state=rng)).reshape(draws, mode.size)
    log_weights = np.array([log_density(draw) for draw in theta]) - proposal.logpdf(theta)
    if np.isnan(log_weights).any() or np.isposinf(log_weights).any():
        raise ValueError("log importance weights must be finite or negative infinity")
    normalization = logsumexp(log_weights)
    if not np.isfinite(normalization):
        raise ValueError("all importance weights are zero")
    weights = np.exp(log_weights - normalization)
    return CovarianceImportanceSamples(theta, weights, float(1 / (weights @ weights)))


def _numerical_curvature(
    log_density: _LogDensity, mode: FloatArray, offsets: FloatArray, step: float
) -> FloatArray:
    """Evaluate negative log-density curvature without analytic gradients."""
    centre = log_density(mode)
    curvature = np.empty((mode.size, mode.size))
    for i, di in enumerate(offsets):
        curvature[i, i] = (2 * centre - log_density(mode + di) - log_density(mode - di)) / step**2
        for j, dj in enumerate(offsets[:i]):
            mixed = (
                log_density(mode + di + dj)
                - log_density(mode + di - dj)
                - log_density(mode - di + dj)
                + log_density(mode - di - dj)
            )
            curvature[i, j] = curvature[j, i] = -mixed / (4 * step**2)
    return curvature


def marginal_posterior(
    posterior: Callable[[FloatArray], tuple[FloatArray, FloatArray]], samples: CovarianceImportanceSamples
) -> tuple[FloatArray, FloatArray]:
    """Integrate conditional Gaussian moments over covariance parameters.

    Args:
        posterior: Function returning conditional mean and covariance at a
            single transformed covariance parameter vector.
        samples: Normalized weighted parameter draws from laplace_importance.

    Returns:
        Mixture mean and covariance including the variation between conditional
        means.
    """
    means, covariances = zip(*(posterior(theta) for theta in samples.theta[samples.weights > 0]), strict=True)
    weights = samples.weights[samples.weights > 0]
    mean = weights @ np.asarray(means)
    centred = np.asarray(means) - mean
    covariance = np.einsum("s,sij->ij", weights, np.asarray(covariances))
    return mean, covariance + (centred.T * weights) @ centred
