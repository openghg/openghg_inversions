"""Posterior summaries on the native fine grid for spatial TD-MCMC.

Lunt et al. (2016), Sect. 2.3.5 maps every retained Voronoi state back to
the native grid before calculating posterior summaries. This module follows
that prescription directly: each saved trace row is reconstructed from its
own active nuclei and coefficients, then cell-wise means and quantiles are
calculated across the retained rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from openghg_inversions.tdmcmc.core import (
    Backend,
    TransDimensionalProblem,
    assign_cells_numba,
    assign_cells_numpy,
)
from openghg_inversions.tdmcmc.sampling import SamplingTrace

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
DEFAULT_QUANTILES = (0.05, 0.5, 0.95)


def _readonly_float_array(values: ArrayLike, *, name: str) -> FloatArray:
    """Return a finite, read-only float64 copy of an array."""
    try:
        array = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values.") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _readonly_int_array(values: ArrayLike, *, name: str) -> IntArray:
    """Return a read-only int64 copy without silently truncating values."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers.")
    result = np.array(array, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _validated_quantile_levels(values: ArrayLike) -> FloatArray:
    """Return ordered, unique quantile probabilities on the closed unit interval."""
    quantile_levels = _readonly_float_array(values, name="quantiles")
    if quantile_levels.ndim != 1 or quantile_levels.size == 0:
        raise ValueError("quantiles must be a non-empty one-dimensional sequence.")
    if np.any((quantile_levels < 0.0) | (quantile_levels > 1.0)):
        raise ValueError("quantiles must lie within [0, 1].")
    if np.any(np.diff(quantile_levels) <= 0.0):
        raise ValueError("quantiles must be strictly increasing and unique.")
    return quantile_levels


def _selected_rows(n_states: int, *, start: int, thin: int) -> IntArray:
    """Validate trace slicing and return the retained saved-state indices."""
    if isinstance(start, bool) or not isinstance(start, (int, np.integer)):
        raise ValueError("start must be an integer saved-state index.")
    if isinstance(thin, bool) or not isinstance(thin, (int, np.integer)):
        raise ValueError("thin must be a positive integer.")
    if thin < 1:
        raise ValueError("thin must be a positive integer.")
    if start < 0 or start >= n_states:
        raise ValueError("start must select an existing saved trace row.")
    rows = np.arange(start, n_states, thin, dtype=np.int64)
    rows.setflags(write=False)
    return rows


def _validated_trace_arrays(
    problem: TransDimensionalProblem,
    trace: SamplingTrace,
) -> tuple[IntArray, IntArray, FloatArray]:
    """Validate the fixed-capacity fields needed for fine-grid reconstruction."""
    k_values = _readonly_int_array(trace.k, name="trace.k")
    nuclei = _readonly_int_array(trace.nuclei, name="trace.nuclei")
    coefficients = _readonly_float_array(trace.coefficients, name="trace.coefficients")

    if k_values.ndim != 1 or k_values.size == 0:
        raise ValueError("trace.k must be a non-empty one-dimensional array.")
    expected_shape = (k_values.size, problem.k_max)
    if nuclei.shape != expected_shape:
        raise ValueError(f"trace.nuclei must have shape {expected_shape}.")
    if coefficients.shape != expected_shape:
        raise ValueError(f"trace.coefficients must have shape {expected_shape}.")

    for row, k_value in enumerate(k_values):
        k = int(k_value)
        if not problem.k_min <= k <= problem.k_max:
            raise ValueError(f"trace.k[{row}] is outside the problem's supported range.")
        active_nuclei = nuclei[row, :k]
        if np.any(active_nuclei < 0) or np.any(active_nuclei >= problem.n_grid_cells):
            raise ValueError(f"trace.nuclei[{row}] contains an invalid active grid index.")
        if np.unique(active_nuclei).size != k:
            raise ValueError(f"trace.nuclei[{row}] must contain unique active indices.")
        if np.any(nuclei[row, k:] != -1):
            raise ValueError(f"trace.nuclei[{row}] must use -1 for inactive padding.")
        if np.any(coefficients[row, k:] != 0.0):
            raise ValueError(f"trace.coefficients[{row}] must use zero for inactive padding.")

    return k_values, nuclei, coefficients


@dataclass(frozen=True, slots=True)
class FineGridPosteriorSummary:
    """Immutable posterior projection and observation-space diagnostics.

    Attributes:
        trace_rows: Indices of the saved trace rows retained after ``start``
            and ``thin`` are applied.
        samples: Reconstructed coefficient fields with shape
            ``(n_retained, n_grid_cells)``.
        mean: Cell-wise posterior mean with shape ``(n_grid_cells,)``.
        quantile_levels: Strictly increasing probabilities associated with
            ``quantiles``.
        quantiles: Cell-wise posterior quantiles with shape
            ``(n_quantiles, n_grid_cells)``.
        predicted_observations: Observation-space prediction from the
            posterior-mean fine-grid field.
        rmse: Root mean squared difference between ``predicted_observations``
            and the supplied comparison vector.

    All arrays are copied and made read-only during construction.
    """

    trace_rows: IntArray
    samples: FloatArray
    mean: FloatArray
    quantile_levels: FloatArray
    quantiles: FloatArray
    predicted_observations: FloatArray
    rmse: float

    def __post_init__(self) -> None:
        """Validate summary shapes and enforce array immutability."""
        trace_rows = _readonly_int_array(self.trace_rows, name="trace_rows")
        samples = _readonly_float_array(self.samples, name="samples")
        mean = _readonly_float_array(self.mean, name="mean")
        quantile_levels = _validated_quantile_levels(self.quantile_levels)
        quantiles = _readonly_float_array(self.quantiles, name="quantiles")
        predicted = _readonly_float_array(
            self.predicted_observations,
            name="predicted_observations",
        )
        rmse = float(self.rmse)

        if trace_rows.ndim != 1 or trace_rows.size == 0:
            raise ValueError("trace_rows must be a non-empty one-dimensional array.")
        if samples.ndim != 2 or samples.shape[0] != trace_rows.size:
            raise ValueError("samples must have shape (n_retained, n_grid_cells).")
        if mean.shape != (samples.shape[1],):
            raise ValueError("mean must have shape (n_grid_cells,).")
        if quantiles.shape != (quantile_levels.size, samples.shape[1]):
            raise ValueError("quantiles must have shape (n_quantiles, n_grid_cells).")
        if predicted.ndim != 1:
            raise ValueError("predicted_observations must be one-dimensional.")
        if not np.isfinite(rmse) or rmse < 0.0:
            raise ValueError("rmse must be finite and non-negative.")

        object.__setattr__(self, "trace_rows", trace_rows)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "quantile_levels", quantile_levels)
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "predicted_observations", predicted)
        object.__setattr__(self, "rmse", rmse)


def reconstruct_fine_grid_samples(
    problem: TransDimensionalProblem,
    trace: SamplingTrace,
    *,
    start: int = 0,
    thin: int = 1,
    backend: Backend = "numpy",
) -> FloatArray:
    """Map retained padded Voronoi states onto the native fine grid.

    ``start`` indexes saved state rows, including the initial state at row
    zero; it does not index proposal diagnostics. Each retained row is
    reconstructed independently because both its partition and coefficients
    may differ from neighbouring rows.

    Args:
        problem: Numerical problem containing native-grid coordinates.
        trace: Fixed-capacity sampler trace to reconstruct.
        start: First saved state row to retain, typically the burn-in cutoff.
        thin: Positive stride between retained saved state rows.
        backend: Voronoi assignment implementation.

    Returns:
        Read-only coefficient fields with shape
        ``(n_retained, n_grid_cells)``.

    Raises:
        ValueError: If trace fields, active nucleus indices, padding, slicing,
            or ``backend`` are malformed.
    """
    if backend not in ("numpy", "numba"):
        raise ValueError("backend must be 'numpy' or 'numba'.")
    k_values, nuclei, coefficients = _validated_trace_arrays(problem, trace)
    rows = _selected_rows(k_values.size, start=start, thin=thin)
    samples = np.empty((rows.size, problem.n_grid_cells), dtype=np.float64)

    for output_row, trace_row in enumerate(rows):
        k = int(k_values[trace_row])
        if backend == "numpy":
            labels = assign_cells_numpy(problem.grid_coordinates, nuclei[trace_row], k)
        else:
            labels = assign_cells_numba(problem.grid_coordinates, nuclei[trace_row], k)
        samples[output_row] = coefficients[trace_row, labels]

    samples.setflags(write=False)
    return samples


def posterior_mean_prediction(
    problem: TransDimensionalProblem,
    posterior_mean: ArrayLike,
    comparison: ArrayLike,
) -> tuple[FloatArray, float]:
    """Predict observations from a posterior-mean grid and calculate RMSE.

    Args:
        problem: Numerical problem containing the fine-grid sensitivity matrix.
        posterior_mean: Mean coefficient field with shape ``(n_grid_cells,)``.
        comparison: Finite comparison vector with shape ``(n_observations,)``.

    Returns:
        Read-only posterior-mean predicted observations and their root mean
        squared error against ``comparison``.

    Raises:
        ValueError: If either supplied vector has an incompatible shape or
            contains non-finite values.
    """
    mean = _readonly_float_array(posterior_mean, name="posterior_mean")
    observed = _readonly_float_array(comparison, name="comparison")
    if mean.shape != (problem.n_grid_cells,):
        raise ValueError("posterior_mean must have shape (n_grid_cells,).")
    if observed.shape != (problem.n_observations,):
        raise ValueError("comparison must have shape (n_observations,).")

    predicted = np.asarray(problem.sensitivities @ mean, dtype=np.float64)
    rmse = sqrt(float(np.mean(np.square(predicted - observed))))
    predicted.setflags(write=False)
    return predicted, rmse


def summarize_fine_grid_posterior(
    problem: TransDimensionalProblem,
    trace: SamplingTrace,
    comparison: ArrayLike,
    *,
    start: int = 0,
    thin: int = 1,
    quantiles: ArrayLike = DEFAULT_QUANTILES,
    backend: Backend = "numpy",
) -> FineGridPosteriorSummary:
    """Reconstruct retained states and calculate native-grid summaries.

    Args:
        problem: Numerical problem containing grid geometry and sensitivities.
        trace: Fixed-capacity sampler trace.
        comparison: Observation-space vector used for the posterior-mean RMSE.
        start: First saved state row to retain, including row zero.
        thin: Positive stride between retained saved state rows.
        quantiles: Strictly increasing quantile probabilities. The default is
            the 5th, 50th, and 95th percentiles.
        backend: Voronoi assignment implementation.

    Returns:
        Immutable reconstructed samples, posterior summaries, and
        observation-space diagnostics.

    Raises:
        ValueError: If trace, slicing, quantiles, comparison, or backend inputs
            are malformed.
    """
    quantile_levels = _validated_quantile_levels(quantiles)
    samples = reconstruct_fine_grid_samples(
        problem,
        trace,
        start=start,
        thin=thin,
        backend=backend,
    )
    mean = np.asarray(np.mean(samples, axis=0), dtype=np.float64)
    quantile_values = np.asarray(np.quantile(samples, quantile_levels, axis=0), dtype=np.float64)
    predicted, rmse = posterior_mean_prediction(problem, mean, comparison)
    rows = _selected_rows(trace.k.size, start=start, thin=thin)
    return FineGridPosteriorSummary(
        trace_rows=rows,
        samples=samples,
        mean=mean,
        quantile_levels=quantile_levels,
        quantiles=quantile_values,
        predicted_observations=predicted,
        rmse=rmse,
    )


__all__ = [
    "DEFAULT_QUANTILES",
    "FineGridPosteriorSummary",
    "posterior_mean_prediction",
    "reconstruct_fine_grid_samples",
    "summarize_fine_grid_posterior",
]
