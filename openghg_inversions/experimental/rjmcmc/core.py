"""Numerical state and target functions for spatial trans-dimensional MCMC.

This module contains a deliberately small, framework-independent reference
implementation.  The public dataclasses own validated NumPy arrays; compiled
kernels operate only on arrays and scalars so they remain easy to compare with
their NumPy counterparts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, pi
from typing import Literal

from numba import njit
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Backend = Literal["numpy", "numba"]


def _readonly_float_array(values: ArrayLike, *, name: str) -> FloatArray:
    """Return a finite, read-only float64 copy of an input array."""
    array = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _readonly_int_array(values: ArrayLike) -> IntArray:
    """Return a read-only int64 copy of an input array."""
    array = np.array(values, dtype=np.int64, copy=True)
    array.setflags(write=False)
    return array


def uniform_log_k_prior(k_min: int, k_max: int) -> FloatArray:
    """Return a normalized discrete-uniform log prior over ``[k_min, k_max]``.

    Args:
        k_min: Smallest supported active-region count.
        k_max: Largest supported active-region count.

    Returns:
        Read-only log probabilities ordered from ``k_min`` through ``k_max``.

    Raises:
        ValueError: If the bounds do not satisfy ``1 <= k_min <= k_max``.
    """
    if k_min < 1 or k_max < k_min:
        raise ValueError("Require 1 <= k_min <= k_max.")
    result = np.full(k_max - k_min + 1, -log(k_max - k_min + 1), dtype=np.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TransDimensionalProblem:
    """Immutable inputs and declared priors for a spatial Voronoi inversion.

    Args:
        observations: Observation vector with shape ``(n_observations,)``.
        observation_sd: Fixed independent standard deviations for observations.
        sensitivities: Fine-grid design with shape
            ``(n_observations, n_grid_cells)``.
        grid_coordinates: Coordinates with shape ``(n_grid_cells, n_dimensions)``.
        k_min: Minimum number of active Voronoi regions.
        k_max: Maximum number of active Voronoi regions.
        log_k_prior: Normalized log probabilities for every integer from
            ``k_min`` through ``k_max``.
        coefficient_prior_mean: Arithmetic mean of the lognormal coefficient prior.
        coefficient_prior_sd: Arithmetic standard deviation of that prior.

    Raises:
        ValueError: If array shapes, numerical supports, active-count bounds,
            or normalized prior probabilities are malformed.
    """

    observations: FloatArray
    observation_sd: FloatArray
    sensitivities: FloatArray
    grid_coordinates: FloatArray
    k_min: int
    k_max: int
    log_k_prior: FloatArray
    coefficient_prior_mean: float
    coefficient_prior_sd: float

    def __post_init__(self) -> None:
        """Validate shapes, supports, and normalized prior probabilities."""
        observations = _readonly_float_array(self.observations, name="observations")
        observation_sd = _readonly_float_array(self.observation_sd, name="observation_sd")
        sensitivities = _readonly_float_array(self.sensitivities, name="sensitivities")
        coordinates = _readonly_float_array(self.grid_coordinates, name="grid_coordinates")
        log_k_prior = np.array(self.log_k_prior, dtype=np.float64, copy=True)

        if observations.ndim != 1:
            raise ValueError("observations must be one-dimensional.")
        if observation_sd.shape != observations.shape:
            raise ValueError("observation_sd must have the same shape as observations.")
        if np.any(observation_sd <= 0.0):
            raise ValueError("observation_sd must be strictly positive.")
        if sensitivities.ndim != 2 or sensitivities.shape[0] != observations.size:
            raise ValueError("sensitivities must have shape (n_observations, n_grid_cells).")
        if coordinates.ndim != 2 or coordinates.shape[0] != sensitivities.shape[1]:
            raise ValueError("grid_coordinates must have shape (n_grid_cells, n_dimensions).")
        if coordinates.shape[1] < 1:
            raise ValueError("grid_coordinates must contain at least one coordinate dimension.")
        if not (1 <= self.k_min <= self.k_max <= sensitivities.shape[1]):
            raise ValueError("Require 1 <= k_min <= k_max <= n_grid_cells.")
        if log_k_prior.shape != (self.k_max - self.k_min + 1,):
            raise ValueError("log_k_prior must contain one value for each supported k.")
        if np.any(np.isnan(log_k_prior)) or np.any(np.isposinf(log_k_prior)):
            raise ValueError("log_k_prior may contain finite values or -inf only.")
        finite = np.isfinite(log_k_prior)
        if not np.any(finite):
            raise ValueError("log_k_prior must assign positive mass to at least one k.")
        maximum = float(np.max(log_k_prior[finite]))
        log_total = maximum + log(float(np.exp(log_k_prior[finite] - maximum).sum()))
        if not np.isclose(log_total, 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("log_k_prior must be normalized (logsumexp equal to zero).")
        if not np.isfinite(self.coefficient_prior_mean) or self.coefficient_prior_mean <= 0.0:
            raise ValueError("coefficient_prior_mean must be finite and positive.")
        if not np.isfinite(self.coefficient_prior_sd) or self.coefficient_prior_sd <= 0.0:
            raise ValueError("coefficient_prior_sd must be finite and positive.")

        log_k_prior.setflags(write=False)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "observation_sd", observation_sd)
        object.__setattr__(self, "sensitivities", sensitivities)
        object.__setattr__(self, "grid_coordinates", coordinates)
        object.__setattr__(self, "log_k_prior", log_k_prior)

    @property
    def n_observations(self) -> int:
        """Number of observations in the numerical problem."""
        return int(self.observations.size)

    @property
    def nobs(self) -> int:
        """Short numerical-kernel alias for :attr:`n_observations`."""
        return self.n_observations

    @property
    def n_grid_cells(self) -> int:
        """Number of candidate nucleus cells in the numerical problem."""
        return int(self.sensitivities.shape[1])

    @property
    def ncell(self) -> int:
        """Short numerical-kernel alias for :attr:`n_grid_cells`."""
        return self.n_grid_cells


@dataclass(frozen=True)
class TransDimensionalState:
    """One self-consistent fixed-capacity state of the numerical sampler.

    States returned by :func:`build_state` satisfy the documented invariants;
    direct dataclass construction bypasses that validation.

    Attributes:
        k: Number of active Voronoi regions.
        nuclei: Canonically sorted nucleus indices with shape ``(k_max,)``.
            Active entries occupy ``[:k]`` and inactive entries equal ``-1``.
        coefficients: Region coefficients with shape ``(k_max,)``. Active
            values align with ``nuclei[:k]`` and inactive entries equal zero.
        labels: Active-region position assigned to each fine-grid cell, with
            shape ``(n_grid_cells,)``.
        design: Region-aggregated sensitivity matrix with shape
            ``(n_observations, k_max)``. Inactive columns equal zero.
        prediction: Model prediction with shape ``(n_observations,)``.
        residual: Prediction minus observations, with shape
            ``(n_observations,)``.
        log_likelihood: Normalized independent-Gaussian log likelihood.
        log_coefficient_prior: Normalized active-coefficient log prior.
        log_k_prior: Declared log probability of the active-region count.
        log_nucleus_prior: Conditional log probability of the active nucleus
            set given ``k``.
    """

    k: int
    nuclei: IntArray
    coefficients: FloatArray
    labels: IntArray
    design: FloatArray
    prediction: FloatArray
    residual: FloatArray
    log_likelihood: float
    log_coefficient_prior: float
    log_k_prior: float
    log_nucleus_prior: float

    @property
    def capacity(self) -> int:
        """Maximum active-region count represented by the padded arrays."""
        return int(self.nuclei.size)

    @property
    def log_target(self) -> float:
        """Return the complete normalized log target for this state."""
        return float(
            self.log_likelihood + self.log_coefficient_prior + self.log_k_prior + self.log_nucleus_prior
        )

    @property
    def active_nuclei(self) -> IntArray:
        """Return the canonical active nucleus indices."""
        return self.nuclei[: self.k]

    @property
    def active_coefficients(self) -> FloatArray:
        """Return coefficients aligned with the active nuclei."""
        return self.coefficients[: self.k]


def assign_cells_numpy(
    grid_coordinates: FloatArray,
    nuclei: IntArray,
    k: int,
) -> IntArray:
    """Assign every grid cell to its nearest active nucleus using NumPy.

    Equal distances are resolved in favour of the first active nucleus. The
    caller is responsible for keeping active nuclei in canonical order.

    Args:
        grid_coordinates: Fine-grid coordinates with shape
            ``(n_grid_cells, n_dimensions)``.
        nuclei: Padded nucleus indices whose first ``k`` entries are active.
        k: Number of active nuclei.

    Returns:
        Active-region positions for all grid cells, with shape
        ``(n_grid_cells,)``.
    """
    active_coordinates = grid_coordinates[nuclei[:k]]
    squared_distance = np.sum(
        np.square(grid_coordinates[:, np.newaxis, :] - active_coordinates[np.newaxis, :, :]),
        axis=2,
    )
    return np.asarray(np.argmin(squared_distance, axis=1), dtype=np.int64)


@njit(cache=True)
def assign_cells_numba(grid_coordinates: FloatArray, nuclei: IntArray, k: int) -> IntArray:
    """Assign every grid cell to its nearest active nucleus using Numba.

    Equal distances are resolved in favour of the first active nucleus.

    Args:
        grid_coordinates: Fine-grid coordinates with shape
            ``(n_grid_cells, n_dimensions)``.
        nuclei: Padded nucleus indices whose first ``k`` entries are active.
        k: Number of active nuclei.

    Returns:
        Active-region positions for all grid cells, with shape
        ``(n_grid_cells,)``.
    """
    n_grid, n_dimensions = grid_coordinates.shape
    labels = np.empty(n_grid, dtype=np.int64)
    for cell in range(n_grid):
        best_region = 0
        best_distance = np.inf
        for region in range(k):
            distance = 0.0
            nucleus = nuclei[region]
            for dimension in range(n_dimensions):
                difference = grid_coordinates[cell, dimension] - grid_coordinates[nucleus, dimension]
                distance += difference * difference
            if distance < best_distance:
                best_distance = distance
                best_region = region
        labels[cell] = best_region
    return labels


def aggregate_design_numpy(
    sensitivities: FloatArray,
    labels: IntArray,
    k: int,
    k_max: int,
) -> FloatArray:
    """Aggregate fine-grid sensitivity columns into padded region columns.

    Args:
        sensitivities: Fine-grid design with shape
            ``(n_observations, n_grid_cells)``.
        labels: Active-region position assigned to each fine-grid cell.
        k: Number of active regions represented by ``labels``.
        k_max: Fixed output capacity for region columns.

    Returns:
        Aggregated design with shape ``(n_observations, k_max)`` and zero
        inactive columns.
    """
    design = np.zeros((sensitivities.shape[0], k_max), dtype=np.float64)
    for cell, region in enumerate(labels):
        design[:, region] += sensitivities[:, cell]
    return design


@njit(cache=True)
def aggregate_design_numba(
    sensitivities: FloatArray,
    labels: IntArray,
    k: int,
    k_max: int,
) -> FloatArray:
    """Aggregate fine-grid sensitivity columns into padded region columns with Numba.

    Args:
        sensitivities: Fine-grid design with shape
            ``(n_observations, n_grid_cells)``.
        labels: Active-region position assigned to each fine-grid cell.
        k: Number of active regions represented by ``labels``.
        k_max: Fixed output capacity for region columns.

    Returns:
        Aggregated design with shape ``(n_observations, k_max)`` and zero
        inactive columns.
    """
    n_observations, n_grid = sensitivities.shape
    design = np.zeros((n_observations, k_max), dtype=np.float64)
    for cell in range(n_grid):
        region = labels[cell]
        for observation in range(n_observations):
            design[observation, region] += sensitivities[observation, cell]
    return design


def gaussian_log_likelihood_numpy(
    residual: FloatArray,
    observation_sd: FloatArray,
) -> float:
    """Return the normalized independent Gaussian log likelihood.

    Args:
        residual: Prediction-minus-observation residual vector.
        observation_sd: Positive standard deviations aligned with ``residual``.

    Returns:
        Sum of normalized Gaussian log densities for the residual vector.
    """
    standardized = residual / observation_sd
    return float(
        -0.5 * np.dot(standardized, standardized)
        - np.log(observation_sd).sum()
        - 0.5 * residual.size * log(2.0 * pi)
    )


@njit(cache=True)
def gaussian_log_likelihood_numba(residual: FloatArray, observation_sd: FloatArray) -> float:
    """Return the normalized independent Gaussian log likelihood with Numba.

    Args:
        residual: Prediction-minus-observation residual vector.
        observation_sd: Positive standard deviations aligned with ``residual``.

    Returns:
        Sum of normalized Gaussian log densities for the residual vector.
    """
    result = -0.5 * residual.size * np.log(2.0 * np.pi)
    for index in range(residual.size):
        standardized = residual[index] / observation_sd[index]
        result -= 0.5 * standardized * standardized + np.log(observation_sd[index])
    return result


def lognormal_mu_sigma(mean: float, standard_deviation: float) -> tuple[float, float]:
    """Convert arithmetic lognormal moments to normal-space parameters.

    Args:
        mean: Positive arithmetic mean of the lognormal distribution.
        standard_deviation: Positive arithmetic standard deviation.

    Returns:
        Normal-space ``(mu, sigma)`` parameters.
    """
    variance_ratio = (standard_deviation / mean) ** 2
    sigma = float(np.sqrt(np.log1p(variance_ratio)))
    mu = float(np.log(mean) - 0.5 * sigma**2)
    return mu, sigma


def lognormal_coefficient_log_prior_numpy(
    coefficients: FloatArray,
    k: int,
    mean: float,
    standard_deviation: float,
) -> float:
    """Return the normalized lognormal density of active coefficients.

    Args:
        coefficients: Padded coefficient array whose first ``k`` entries are
            active.
        k: Number of active coefficients.
        mean: Positive arithmetic mean of the lognormal prior.
        standard_deviation: Positive arithmetic standard deviation of the
            lognormal prior.

    Returns:
        Sum of normalized active-coefficient log densities, or negative
        infinity when an active coefficient is outside positive support.
    """
    active = coefficients[:k]
    if np.any(active <= 0.0) or not np.all(np.isfinite(active)):
        return -np.inf
    mu, sigma = lognormal_mu_sigma(mean, standard_deviation)
    z = (np.log(active) - mu) / sigma
    return float(-0.5 * np.dot(z, z) - np.log(active).sum() - k * log(sigma) - 0.5 * k * log(2.0 * pi))


@njit(cache=True)
def lognormal_coefficient_log_prior_numba(
    coefficients: FloatArray,
    k: int,
    mean: float,
    standard_deviation: float,
) -> float:
    """Return the normalized lognormal coefficient density with Numba.

    Args:
        coefficients: Padded coefficient array whose first ``k`` entries are
            active.
        k: Number of active coefficients.
        mean: Positive arithmetic mean of the lognormal prior.
        standard_deviation: Positive arithmetic standard deviation of the
            lognormal prior.

    Returns:
        Sum of normalized active-coefficient log densities, or negative
        infinity when an active coefficient is outside positive support.
    """
    variance_ratio = (standard_deviation / mean) ** 2
    sigma = np.sqrt(np.log1p(variance_ratio))
    mu = np.log(mean) - 0.5 * sigma * sigma
    result = -k * np.log(sigma) - 0.5 * k * np.log(2.0 * np.pi)
    for index in range(k):
        value = coefficients[index]
        if value <= 0.0 or not np.isfinite(value):
            return -np.inf
        z = (np.log(value) - mu) / sigma
        result -= 0.5 * z * z + np.log(value)
    return result


def uniform_nucleus_set_log_prior(n_grid_cells: int, k: int) -> float:
    """Return a normalized conditional log prior over unordered nucleus sets.

    Args:
        n_grid_cells: Number of candidate fine-grid nucleus cells.
        k: Number of active nuclei.

    Returns:
        ``log(1 / comb(n_grid_cells, k))``, or negative infinity when ``k`` is
        outside ``[0, n_grid_cells]``.
    """
    if not 0 <= k <= n_grid_cells:
        return -np.inf
    return float(-(lgamma(n_grid_cells + 1) - lgamma(k + 1) - lgamma(n_grid_cells - k + 1)))


def build_state(
    problem: TransDimensionalProblem,
    active_nuclei: ArrayLike,
    active_coefficients: ArrayLike,
    *,
    backend: Backend = "numpy",
) -> TransDimensionalState:
    """Build and validate a complete sampler state from its active values.

    Args:
        problem: Immutable numerical problem and target specification.
        active_nuclei: Unique flattened grid indices for active nuclei.
        active_coefficients: Finite coefficients paired with the supplied
            nuclei. Nonpositive active values produce a state with negative
            infinite coefficient-prior density and log target.
        backend: Numerical kernels used to derive labels and target caches.

    Returns:
        A canonical, fixed-capacity state with nuclei sorted in ascending order.

    Raises:
        ValueError: If active arrays, counts, nucleus indices, or ``backend``
            are malformed.
    """
    supplied_nuclei = np.asarray(active_nuclei, dtype=np.int64)
    supplied_coefficients = np.asarray(active_coefficients, dtype=np.float64)
    if supplied_nuclei.ndim != 1 or supplied_coefficients.ndim != 1:
        raise ValueError("active_nuclei and active_coefficients must be one-dimensional.")
    if supplied_nuclei.size != supplied_coefficients.size:
        raise ValueError("active_nuclei and active_coefficients must have equal length.")
    k = int(supplied_nuclei.size)
    if not problem.k_min <= k <= problem.k_max:
        raise ValueError("Active-region count is outside the problem's supported range.")
    if np.any(supplied_nuclei < 0) or np.any(supplied_nuclei >= problem.n_grid_cells):
        raise ValueError("Active nuclei must be valid flattened grid indices.")
    if np.unique(supplied_nuclei).size != k:
        raise ValueError("Active nuclei must be unique.")
    if not np.all(np.isfinite(supplied_coefficients)):
        raise ValueError("Active coefficients must be finite.")

    order = np.argsort(supplied_nuclei, kind="stable")
    sorted_nuclei = supplied_nuclei[order]
    sorted_coefficients = supplied_coefficients[order]
    nuclei = np.full(problem.k_max, -1, dtype=np.int64)
    coefficients = np.zeros(problem.k_max, dtype=np.float64)
    nuclei[:k] = sorted_nuclei
    coefficients[:k] = sorted_coefficients

    if backend == "numpy":
        labels = assign_cells_numpy(problem.grid_coordinates, nuclei, k)
        design = aggregate_design_numpy(problem.sensitivities, labels, k, problem.k_max)
    elif backend == "numba":
        labels = assign_cells_numba(problem.grid_coordinates, nuclei, k)
        design = aggregate_design_numba(problem.sensitivities, labels, k, problem.k_max)
    else:
        raise ValueError("backend must be 'numpy' or 'numba'.")

    prediction = design[:, :k] @ coefficients[:k]
    residual = prediction - problem.observations
    if backend == "numpy":
        log_likelihood = gaussian_log_likelihood_numpy(residual, problem.observation_sd)
        log_coefficient_prior = lognormal_coefficient_log_prior_numpy(
            coefficients,
            k,
            problem.coefficient_prior_mean,
            problem.coefficient_prior_sd,
        )
    else:
        log_likelihood = gaussian_log_likelihood_numba(residual, problem.observation_sd)
        log_coefficient_prior = lognormal_coefficient_log_prior_numba(
            coefficients,
            k,
            problem.coefficient_prior_mean,
            problem.coefficient_prior_sd,
        )

    for array in (nuclei, coefficients, labels, design, prediction, residual):
        array.setflags(write=False)
    return TransDimensionalState(
        k=k,
        nuclei=nuclei,
        coefficients=coefficients,
        labels=labels,
        design=design,
        prediction=prediction,
        residual=residual,
        log_likelihood=float(log_likelihood),
        log_coefficient_prior=float(log_coefficient_prior),
        log_k_prior=float(problem.log_k_prior[k - problem.k_min]),
        log_nucleus_prior=uniform_nucleus_set_log_prior(problem.n_grid_cells, k),
    )
