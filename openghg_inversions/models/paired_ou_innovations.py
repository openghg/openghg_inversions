"""Experimental innovations likelihood for inferred paired OU amplitudes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import ArrayLike
from scipy.linalg import cho_solve
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from openghg_inversions._linalg import cholesky_logdet, conditioned_cholesky
from openghg_inversions.models.cached_sigma import MarginalQuadraticCache, _sum_squares_gram
from openghg_inversions.models.fixed_ou import (
    FloatArray,
    IntArray,
    _evaluate_whitened,
    _group_index,
    _matrix,
    _site_values,
    _solve_whitened,
    _time_hours,
    _vector,
)
from openghg_inversions.models.paired_error import PairedErrorCorrelation


@njit
def _whiten_block(
    values, previous, subtract, loading, noise, decay, innovation, stationary, amplitudes, gradients
):
    """Filter a site's standardized OU states and differentiate their whitening.

    Args:
        values: Observation rows in time order, with one column per right-hand side.
        previous: Earlier observation to subtract for a same-time pair, or -1.
        subtract: Measurement regression coefficient for each row.
        loading: Amplitude-free state loadings after the same-time row operation.
        noise: Measurement variance after the same-time row operation.
        decay: Per-row transition coefficient for each state.
        innovation: Per-row unit process variance for each state.
        stationary: Stationary covariance of the standardized states.
        amplitudes: Physical state loadings before the same-time row operation.
        gradients: Propagate amplitude sensitivities.

    Returns:
        Whitened values, their amplitude sensitivities, log determinant, and
        its amplitude sensitivities. A nonfinite determinant signals rejection.
    """
    rows, width = values.shape
    dimension = len(amplitudes)
    white = np.empty_like(values)
    derivative = np.empty((dimension if gradients else 0, rows, width))
    logdet = 0.0
    dlogdet = np.zeros(dimension)
    mean = np.zeros((dimension, width))
    covariance = stationary.copy()
    dmean = np.zeros((dimension, dimension, width))
    dcov = np.zeros((dimension, dimension, dimension))
    h = np.empty(dimension)
    projected = np.empty(dimension)
    gain = np.empty(dimension)
    dprojected = np.empty(dimension)
    for row in range(rows):
        variance = noise[row]
        for i in range(dimension):
            h[i] = amplitudes[i] * loading[row, i]
            for j in range(width):
                mean[i, j] *= decay[row, i]
            for j in range(dimension):
                multiplier = decay[row, i] * decay[row, j]
                covariance[i, j] *= multiplier
                for k in range(dimension if gradients else 0):
                    dcov[k, i, j] *= multiplier
                # An off-diagonal stationary entry requires identical group decays.
                covariance[i, j] += innovation[row, i] * stationary[i, j]
            for k in range(dimension if gradients else 0):
                for j in range(width):
                    dmean[k, i, j] *= decay[row, i]
        for i in range(dimension):
            projected[i] = 0.0
            for j in range(dimension):
                projected[i] += covariance[i, j] * h[j]
            variance += h[i] * projected[i]
        if not np.isfinite(variance) or variance <= 0.0:
            return white, derivative, -np.inf, dlogdet
        sd = np.sqrt(variance)
        logdet += np.log(variance)
        for i in range(dimension):
            gain[i] = projected[i] / sd
        for j in range(width):
            residual = values[row, j]
            if previous[row] >= 0:
                residual -= subtract[row] * values[previous[row], j]
            for i in range(dimension):
                residual -= h[i] * mean[i, j]
            white[row, j] = residual / sd

        # Differentiate each scalar innovation before replacing the predicted state.
        for k in range(dimension if gradients else 0):
            dvariance = loading[row, k] * projected[k]
            for i in range(dimension):
                dprojected[i] = covariance[i, k] * loading[row, k]
                for j in range(dimension):
                    dprojected[i] += dcov[k, i, j] * h[j]
                dvariance += h[i] * dprojected[i]
            dlogdet[k] += dvariance / variance
            relative_sd = 0.5 * dvariance / variance
            dgain = dprojected / sd - gain * relative_sd
            for j in range(width):
                dresidual = -loading[row, k] * mean[k, j]
                for i in range(dimension):
                    dresidual -= h[i] * dmean[k, i, j]
                dwhite = dresidual / sd - white[row, j] * relative_sd
                derivative[k, row, j] = dwhite
                for i in range(dimension):
                    dmean[k, i, j] += dgain[i] * white[row, j] + gain[i] * dwhite
            for i in range(dimension):
                for j in range(dimension):
                    dcov[k, i, j] -= dgain[i] * gain[j] + gain[i] * dgain[j]

        for i in range(dimension):
            for j in range(width):
                mean[i, j] += gain[i] * white[row, j]
            for j in range(dimension):
                covariance[i, j] -= gain[i] * gain[j]
    return white, derivative, logdet, dlogdet


@dataclass(frozen=True)
class _SiteBlock:
    """Sorted observation rows and fixed filtering inputs for a site."""

    rows: IntArray
    groups: IntArray
    previous: IntArray
    subtract: FloatArray
    loading: FloatArray
    noise: FloatArray
    decay: FloatArray
    innovation: FloatArray
    stationary: FloatArray

    def whiten(
        self, values: FloatArray, amplitudes: FloatArray, gradients: bool
    ) -> tuple[FloatArray, FloatArray, float, FloatArray]:
        """Run the compiled filter and optionally propagate amplitude sensitivities."""
        return _whiten_block(
            values[self.rows],
            self.previous,
            self.subtract,
            self.loading,
            self.noise,
            self.decay,
            self.innovation,
            self.stationary,
            amplitudes[self.groups],
            gradients,
        )


@dataclass(frozen=True)
class PairedOuInnovations:
    """Prepared observation covariance for repeated inferred-amplitude evaluations.

    Attributes:
        factor: Coherent aggregation factor in observation order.
    """

    factor: FloatArray
    _blocks: tuple[_SiteBlock, ...]

    @property
    def n_observation(self) -> int:
        """Number of observation rows."""
        return self.factor.shape[0]

    @property
    def n_site(self) -> int:
        """Number of distinct OU amplitude groups."""
        return sum(len(block.groups) for block in self._blocks)

    def _whiten(
        self, rhs: FloatArray, amplitudes: FloatArray, *, gradients: bool = False
    ) -> tuple[FloatArray, FloatArray, float, list[tuple[_SiteBlock, FloatArray, FloatArray]]]:
        """Whiten the factor and right-hand sides in the same filter pass."""
        values = np.column_stack((self.factor, rhs))
        white = np.empty_like(values)
        logdet = 0.0
        sensitivities = []
        for site in self._blocks:
            block, derivative, log_block, dlog = site.whiten(values, amplitudes, gradients)
            if not np.isfinite(log_block):
                return white[:, :0], white[:, :0], -np.inf, []
            white[site.rows] = block
            logdet += log_block
            if gradients:
                sensitivities.append((site, derivative, dlog))
        rank = self.factor.shape[1]
        return white[:, :rank], white[:, rank:], logdet, sensitivities

    def likelihood(
        self, residual: ArrayLike, amplitudes: ArrayLike | float, *, gradients: bool = True
    ) -> tuple[float, FloatArray]:
        """Evaluate the normalized Gaussian log density.

        Args:
            residual: Observation minus completed model mean in observation order.
            amplitudes: Nonnegative OU standard deviation per group, or one common value.
            gradients: Include the derivative with respect to each physical amplitude.

        Returns:
            Log density and one derivative per group, or zeros when gradients are disabled.
            The log density is negative infinity for nonfinite residuals or invalid proposals.

        Raises:
            ValueError: Residual or amplitude has an incompatible shape.
            FloatingPointError: The low-rank gradient solve fails its conditioning check.
        """
        y = np.asarray(residual, dtype=np.float64)
        if y.shape != (self.n_observation,):
            raise ValueError("residual must have one value per observation.")
        a = np.asarray(amplitudes, dtype=np.float64)
        if a.ndim == 0:
            a = np.full(self.n_site, a.item())
        elif a.shape != (self.n_site,):
            raise ValueError("amplitudes must be scalar or have one value per group.")
        gradient = np.zeros_like(a)
        if not np.isfinite(y).all() or not np.isfinite(a).all() or np.any(a < 0.0):
            return -np.inf, gradient
        factor, values, logdet, blocks = self._whiten(y[:, None], a, gradients=gradients)
        if not np.isfinite(logdet):
            return -np.inf, gradient
        if not gradients:
            quadratic, logdet, _, _ = _evaluate_whitened(
                factor,
                values[:, 0],
                logdet,
                amplitude_gradient=False,
            )
            return _gaussian_logp(self.n_observation, logdet, quadratic), gradient
        y = values[:, 0]
        rank = factor.shape[1]
        if rank:
            core = factor.T @ factor
            core = (core + core.T) * 0.5
            core.flat[:: rank + 1] += 1.0
            chol = conditioned_cholesky(core, np.sqrt(np.finfo(float).eps))
            if chol is None:
                raise FloatingPointError("The innovations Woodbury core failed its conditioning check.")
            z = cho_solve((chol, True), factor.T @ y, check_finite=False)
            error = y - factor @ z
            logdet += cholesky_logdet(chol)
            inverse_weighted = cho_solve((chol, True), factor.T, check_finite=False).T
        else:
            z = np.empty(0)
            error = y
            inverse_weighted = factor
        # Stationarity of the ridge solve cancels its derivative in the quadratic.
        for site, derivative, dlog in blocks:
            for i, group in enumerate(site.groups):
                d_factor, d_y = derivative[i, :, :-1], derivative[i, :, -1]
                gradient[group] = (
                    -0.5 * dlog[i]
                    - error[site.rows] @ (d_y - d_factor @ z)
                    - np.einsum("ij,ij->", inverse_weighted[site.rows], d_factor)
                )
        return _gaussian_logp(self.n_observation, logdet, float(error @ error + z @ z)), gradient

    def state_cache(
        self, residual_at_zero: ArrayLike, design: ArrayLike, amplitudes: ArrayLike | float
    ) -> MarginalQuadraticCache:
        """Build the state likelihood quadratic for one amplitude proposal.

        Args:
            residual_at_zero: Observations minus the fixed mean contribution.
            design: Observation-major matrix mapping the state to the variable mean.
            amplitudes: Nonnegative OU standard deviation per group, or one common value.

        Returns:
            Normalized likelihood coefficients for residual_at_zero minus design times state.

        Raises:
            ValueError: Inputs have incompatible shapes or invalid values.
            FloatingPointError: A base covariance innovation is singular or nonfinite.
        """
        residual = _vector(residual_at_zero, "residual_at_zero")
        design_value = _matrix(design, "design")
        if residual.size != self.n_observation or design_value.shape[0] != self.n_observation:
            raise ValueError("residual_at_zero and design must have one row per observation.")
        amplitude = _site_values(amplitudes, self.n_site, "amplitudes", positive=False)
        rhs = np.column_stack((residual, design_value))
        factor, white, logdet, _ = self._whiten(rhs, amplitude)
        if not np.isfinite(logdet):
            raise FloatingPointError("Invalid base covariance innovation.")
        gram_result = None
        if 0 < factor.shape[1] < self.n_observation:
            gram_result = _sum_squares_gram(factor, white)
        if gram_result is None:
            solved, logdet = _solve_whitened(factor, white, logdet)
            gram = white.T @ solved
        else:
            gram, update = gram_result
            logdet += update
        precision = gram[1:, 1:]
        return MarginalQuadraticCache(
            _gaussian_logp(self.n_observation, logdet, float(gram[0, 0])),
            gram[1:, 0].copy(),
            (precision + precision.T) * 0.5,
            amplitude,
        )


def _gaussian_logp(n: int, logdet: float, quadratic: float) -> float:
    """Evaluate the normalized Gaussian log density from covariance statistics."""
    return float(-0.5 * (n * np.log(2.0 * np.pi) + logdet + quadratic))


def prepare_paired_ou_innovations(
    factor: ArrayLike,
    diagonal: ArrayLike,
    site_index: ArrayLike,
    observation_times: ArrayLike,
    tau_hours: ArrayLike | float,
    *,
    paired_error: PairedErrorCorrelation | None = None,
    model_error_sd: ArrayLike | None = None,
    persistent_correlation: Mapping[tuple[int, int], float] | None = None,
) -> PairedOuInnovations:
    """Prepare exact innovations for independent or paired stationary OU processes.

    The covariance is the sum of the coherent aggregation covariance, the
    measurement and model-error covariance, and the OU covariance. For a
    persistent pair of groups i and j with common decay tau, cross-group OU
    covariance at lag dt is rho * amplitude[i] * amplitude[j] * exp(-abs(dt) / tau).

    Args:
        factor: Observation-major coherent aggregation factor, possibly with zero columns.
        diagonal: Strictly positive independent variance including the model-error variance.
        site_index: One contiguous zero-based tracer group index per observation.
        observation_times: One datetime or numeric hour per observation, in any order.
        tau_hours: Strictly positive decay in hours per group, or one common value.
        paired_error: Existing same-time cross-tracer model-error pairs.
        model_error_sd: Model-error standard deviations, required with paired_error.
            Their squares are already included in diagonal.
        persistent_correlation: Optional fixed stationary latent correlations for
            disjoint pairs of group indices. Each pair must share one decay.
            The model-error and persistent pairs can coexist within a site.

    Returns:
        An experimental Python likelihood and state-quadratic evaluator.

    Raises:
        ValueError: Inputs violate shape, pairing, positive variance, or shared decay requirements.
    """
    factor_value = _matrix(factor, "factor")
    n = factor_value.shape[0]
    if n == 0:
        raise ValueError("At least one observation is required.")
    diagonal_value = _vector(diagonal, "diagonal")
    times = _time_hours(observation_times)
    if diagonal_value.size != n or times.size != n or np.any(diagonal_value <= 0.0):
        raise ValueError("One time and one strictly positive diagonal variance per observation are required.")
    group = _group_index(site_index, n)
    n_group = int(group.max()) + 1
    tau = _site_values(tau_hours, n_group, "tau_hours", positive=True)

    partner = np.full(n, -1, dtype=np.int64)
    subtract = np.zeros(n)
    conditioning_variance = diagonal_value.copy()
    model_edges = np.empty((0, 2), dtype=np.int64)
    if paired_error is not None:
        if model_error_sd is None:
            raise ValueError("model_error_sd is required with paired_error.")
        sd = _vector(model_error_sd, "model_error_sd")
        if sd.shape != (n,) or np.any(sd < 0.0) or np.any(sd**2 > diagonal_value):
            raise ValueError("model_error_sd must be nonnegative and included in the diagonal variance.")
        paired_error._check_size(n)
        use = paired_error.correlated()
        first, second = paired_error.first[use], paired_error.second[use]
        cov = paired_error.correlation[use] * sd[first] * sd[second]
        left, right = np.minimum(first, second), np.maximum(first, second)
        if np.any(times[left] != times[right]) or np.any(group[left] == group[right]):
            raise ValueError("Model-error pairs must connect distinct groups at the same time.")
        model_edges = np.column_stack((group[left], group[right]))
        partner[right] = left
        subtract[right] = cov / diagonal_value[left]
        conditioning_variance[right] -= subtract[right] * cov
        if np.any(conditioning_variance[right] <= 0.0):
            raise ValueError("Model-error pairs must have positive-definite measurement blocks.")

    correlations: dict[tuple[int, int], float] = {}
    for (i, j), correlation in (persistent_correlation or {}).items():
        if not (
            isinstance(i, (int, np.integer))
            and isinstance(j, (int, np.integer))
            and 0 <= i < n_group
            and 0 <= j < n_group
            and i != j
        ):
            raise ValueError("Persistent pairs must connect distinct valid group indices.")
        key = (min(i, j), max(i, j))
        if key in correlations or not np.isfinite(correlation) or abs(correlation) >= 1:
            raise ValueError("Each persistent pair needs one finite correlation strictly between -1 and 1.")
        if tau[i] != tau[j]:
            raise ValueError("Persistent pairs require a common decay.")
        correlations[key] = correlation

    persistent_edges = np.asarray(list(correlations), dtype=np.int64).reshape(-1, 2)
    edges = np.concatenate((model_edges, persistent_edges))
    graph = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_group, n_group))
    n_site, labels = connected_components(graph, directed=False)
    blocks = []
    for site in range(n_site):
        groups = np.flatnonzero(labels == site)
        if groups.size > 2:
            raise ValueError("Each connected site can contain at most two tracer groups.")
        rows = np.flatnonzero(labels[group] == site)
        rows = rows[np.argsort(times[rows], kind="stable")]
        local_group = np.searchsorted(groups, group[rows])
        loading = np.eye(groups.size)[local_group]
        previous = np.full(len(rows), -1, dtype=np.int64)
        paired = np.flatnonzero(partner[rows] >= 0)
        local_rows = {row: pos for pos, row in enumerate(rows)}
        for pos in paired:
            left = local_rows[partner[rows[pos]]]
            previous[pos] = left
            loading[pos, local_group[left]] -= subtract[rows[pos]]
        steps = np.diff(times[rows], prepend=times[rows[0]])
        rate = steps[:, None] / tau[groups]
        stationary = np.eye(groups.size)
        if groups.size == 2:
            stationary[0, 1] = stationary[1, 0] = correlations.get(tuple(groups), 0.0)
        blocks.append(
            _SiteBlock(
                rows,
                groups,
                previous,
                subtract[rows],
                loading,
                conditioning_variance[rows],
                np.exp(-rate),
                -np.expm1(-2.0 * rate),
                stationary,
            )
        )
    return PairedOuInnovations(factor_value, tuple(blocks))
