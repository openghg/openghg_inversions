"""Numerical state and target functions for spatial trans-dimensional MCMC.

This module contains a framework-independent NumPy/Numba implementation.
``build_state`` is the complete initialization and validation oracle;
``update_structural_state`` is the internal proposal fast path for one-nucleus
edits. Nuclei are stored in canonical index order, equal-distance cells belong
to the first canonical nucleus, and state arrays are read-only caches. The two
backends are required to construct identical caches. Incremental structural
updates reuse fixed-predictor caches and unaffected dynamic design columns,
while malformed source caches or unsupported multi-edits fall back to the full
builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import lgamma, log, pi
from typing import TYPE_CHECKING, Literal

from numba import njit
import numpy as np
from numpy.typing import ArrayLike, NDArray

from openghg_inversions.experimental.rjmcmc.likelihood import (
    IndependentSiteOUData,
    ou_log_likelihood_numba,
    ou_log_likelihood_numpy,
)

if TYPE_CHECKING:
    from openghg_inversions.experimental.rjmcmc.hierarchy import SharedLognormalHierarchy

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Backend = Literal["numpy", "numba"]


def _empty_readonly_float_array() -> FloatArray:
    """Return an empty immutable float64 vector for neutral state fields."""
    result = np.empty(0, dtype=np.float64)
    result.setflags(write=False)
    return result


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
class FixedDesignBlock:
    """Immutable always-active design columns and their coefficient priors.

    Args:
        design: Fixed design matrix with shape
            ``(n_observations, n_fixed_coefficients)``.
        coefficient_prior_mean: Positive arithmetic means of the independent
            lognormal coefficient priors.
        coefficient_prior_sd: Positive arithmetic standard deviations of the
            independent lognormal coefficient priors.

    Raises:
        ValueError: If the design or prior-moment arrays have malformed shapes,
            contain nonfinite values, or declare invalid lognormal moments.
    """

    design: FloatArray
    coefficient_prior_mean: FloatArray
    coefficient_prior_sd: FloatArray

    def __post_init__(self) -> None:
        """Own and validate the fixed design and its explicit prior moments."""
        design = _readonly_float_array(self.design, name="fixed block design")
        prior_mean = _readonly_float_array(
            self.coefficient_prior_mean,
            name="fixed coefficient_prior_mean",
        )
        prior_sd = _readonly_float_array(
            self.coefficient_prior_sd,
            name="fixed coefficient_prior_sd",
        )
        if design.ndim != 2 or design.shape[1] < 1:
            raise ValueError("fixed block design must be two-dimensional with at least one column.")
        expected_shape = (design.shape[1],)
        if prior_mean.shape != expected_shape or prior_sd.shape != expected_shape:
            raise ValueError("fixed coefficient prior moments must have one value per design column.")
        if np.any(prior_mean <= 0.0) or np.any(prior_sd <= 0.0):
            raise ValueError("fixed coefficient prior moments must be strictly positive.")

        object.__setattr__(self, "design", design)
        object.__setattr__(self, "coefficient_prior_mean", prior_mean)
        object.__setattr__(self, "coefficient_prior_sd", prior_sd)

    @property
    def n_coefficients(self) -> int:
        """Number of always-active coefficients represented by the block."""
        return int(self.design.shape[1])


@dataclass(frozen=True)
class InferredOUErrorModel:
    """Immutable inferred independent-site OU error-model specification.

    The model-data mismatch standard deviations and OU correlation timescales
    have independent normalized bounded-uniform priors. Bounds are expressed
    in the same units as the corresponding runtime parameters. The observation
    standard deviations stored in ``data`` are the fixed independent nugget.

    Args:
        data: Static observation alignment and OU traversal data.
        mismatch_sd_prior_lower: Lower prior bound for each mismatch group.
        mismatch_sd_prior_upper: Upper prior bound for each mismatch group.
        correlation_timescale_prior_lower: Lower prior bound for each shared
            correlation-timescale parameter.
        correlation_timescale_prior_upper: Upper prior bound for each shared
            correlation-timescale parameter.

    Raises:
        TypeError: If ``data`` is not :class:`IndependentSiteOUData`.
        ValueError: If prior bounds have incorrect shapes, are nonfinite, are
            non-positive, or do not define intervals with positive width.
    """

    data: IndependentSiteOUData
    mismatch_sd_prior_lower: FloatArray
    mismatch_sd_prior_upper: FloatArray
    correlation_timescale_prior_lower: FloatArray
    correlation_timescale_prior_upper: FloatArray

    def __post_init__(self) -> None:
        """Own and validate all normalized bounded-uniform prior bounds."""
        if not isinstance(self.data, IndependentSiteOUData):
            raise TypeError("data must be an IndependentSiteOUData instance.")
        mismatch_lower = _readonly_float_array(
            self.mismatch_sd_prior_lower,
            name="mismatch_sd_prior_lower",
        )
        mismatch_upper = _readonly_float_array(
            self.mismatch_sd_prior_upper,
            name="mismatch_sd_prior_upper",
        )
        timescale_lower = _readonly_float_array(
            self.correlation_timescale_prior_lower,
            name="correlation_timescale_prior_lower",
        )
        timescale_upper = _readonly_float_array(
            self.correlation_timescale_prior_upper,
            name="correlation_timescale_prior_upper",
        )
        expected_mismatch_shape = (self.data.n_mismatch_groups,)
        expected_timescale_shape = (self.data.n_tau_parameters,)
        if mismatch_lower.shape != expected_mismatch_shape or mismatch_upper.shape != expected_mismatch_shape:
            raise ValueError("mismatch SD prior bounds must have one value per mismatch group.")
        if (
            timescale_lower.shape != expected_timescale_shape
            or timescale_upper.shape != expected_timescale_shape
        ):
            raise ValueError("timescale prior bounds must have one value per timescale parameter.")
        if np.any(mismatch_lower <= 0.0) or np.any(timescale_lower <= 0.0):
            raise ValueError("error-model prior lower bounds must be strictly positive.")
        if np.any(mismatch_upper <= mismatch_lower) or np.any(timescale_upper <= timescale_lower):
            raise ValueError("error-model prior upper bounds must exceed lower bounds.")
        object.__setattr__(self, "mismatch_sd_prior_lower", mismatch_lower)
        object.__setattr__(self, "mismatch_sd_prior_upper", mismatch_upper)
        object.__setattr__(self, "correlation_timescale_prior_lower", timescale_lower)
        object.__setattr__(self, "correlation_timescale_prior_upper", timescale_upper)


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
        fixed_offset: Optional coefficient-independent prediction offset. ``None``
            is normalized to a read-only zero vector.
        fixed_block: Optional always-active design block and coefficient priors.
        error_model: Optional inferred OU model-data mismatch configuration.
        coefficient_hierarchy: Optional shared partially pooled prior for all
            active dynamic coefficients.

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
    fixed_offset: FloatArray | None = None
    fixed_block: FixedDesignBlock | None = None
    error_model: InferredOUErrorModel | None = None
    coefficient_hierarchy: SharedLognormalHierarchy | None = None

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

        if self.fixed_offset is None:
            fixed_offset = np.zeros(observations.shape, dtype=np.float64)
            fixed_offset.setflags(write=False)
        else:
            fixed_offset = _readonly_float_array(self.fixed_offset, name="fixed_offset")
            if fixed_offset.shape != observations.shape:
                raise ValueError("fixed_offset must have the same shape as observations.")
        if self.fixed_block is not None:
            if not isinstance(self.fixed_block, FixedDesignBlock):
                raise TypeError("fixed_block must be a FixedDesignBlock or None.")
            if self.fixed_block.design.shape[0] != observations.size:
                raise ValueError("fixed block design must have one row per observation.")
        if self.error_model is not None:
            if not isinstance(self.error_model, InferredOUErrorModel):
                raise TypeError("error_model must be an InferredOUErrorModel or None.")
            if self.error_model.data.n_observations != observations.size:
                raise ValueError("error-model data must contain one row per observation.")
            if not np.array_equal(self.error_model.data.observation_sd, observation_sd):
                raise ValueError("error-model observation_sd must exactly match problem.observation_sd.")
        if self.coefficient_hierarchy is not None:
            from openghg_inversions.experimental.rjmcmc.hierarchy import (
                SharedLognormalHierarchy,
            )

            if not isinstance(self.coefficient_hierarchy, SharedLognormalHierarchy):
                raise TypeError("coefficient_hierarchy must be a SharedLognormalHierarchy or None.")

        log_k_prior.setflags(write=False)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "observation_sd", observation_sd)
        object.__setattr__(self, "sensitivities", sensitivities)
        object.__setattr__(self, "grid_coordinates", coordinates)
        object.__setattr__(self, "log_k_prior", log_k_prior)
        object.__setattr__(self, "fixed_offset", fixed_offset)

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

    @property
    def n_fixed_coefficients(self) -> int:
        """Number of always-active coefficients in the optional fixed block."""
        return 0 if self.fixed_block is None else self.fixed_block.n_coefficients


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
        fixed_coefficients: Always-active coefficients aligned with the fixed
            block columns.
        dynamic_prediction: Prediction from the Voronoi design and active
            coefficients.
        fixed_prediction: Prediction from the fixed offset and always-active
            fixed block.
        prediction: Total model prediction with shape ``(n_observations,)``.
        residual: Prediction minus observations, with shape
            ``(n_observations,)``.
        log_likelihood: Normalized independent-Gaussian log likelihood.
        log_coefficient_prior: Normalized active-coefficient log prior.
        log_fixed_coefficient_prior: Normalized always-active coefficient log
            prior, or zero when there is no fixed block.
        log_k_prior: Declared log probability of the active-region count.
        log_nucleus_prior: Conditional log probability of the active nucleus
            set given ``k``.
        mismatch_sd: Inferred OU mismatch amplitudes, or an empty vector when
            the independent fixed-error likelihood is configured.
        correlation_timescale: Inferred OU correlation timescales, or an empty
            vector when the independent fixed-error likelihood is configured.
        eta: Logarithm of the shared arithmetic coefficient-prior mean.
        zeta: Logarithm of the shared arithmetic coefficient-prior SD.
        log_error_model_prior: Normalized prior density of inferred error
            parameters, or zero when no inferred error model is configured.
        log_coefficient_hyperprior: Normalized density of the shared hierarchy
            state, or zero when no coefficient hierarchy is configured.
    """

    k: int
    nuclei: IntArray
    coefficients: FloatArray
    labels: IntArray
    design: FloatArray
    fixed_coefficients: FloatArray
    dynamic_prediction: FloatArray
    fixed_prediction: FloatArray
    prediction: FloatArray
    residual: FloatArray
    log_likelihood: float
    log_coefficient_prior: float
    log_fixed_coefficient_prior: float
    log_k_prior: float
    log_nucleus_prior: float
    mismatch_sd: FloatArray = field(default_factory=_empty_readonly_float_array)
    correlation_timescale: FloatArray = field(default_factory=_empty_readonly_float_array)
    eta: float = 0.0
    zeta: float = 0.0
    log_error_model_prior: float = 0.0
    log_coefficient_hyperprior: float = 0.0

    @property
    def capacity(self) -> int:
        """Maximum active-region count represented by the padded arrays."""
        return int(self.nuclei.size)

    @property
    def log_target(self) -> float:
        """Return the complete normalized log target for this state."""
        return float(
            self.log_likelihood
            + self.log_coefficient_prior
            + self.log_fixed_coefficient_prior
            + self.log_k_prior
            + self.log_nucleus_prior
            + self.log_error_model_prior
            + self.log_coefficient_hyperprior
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
    for observation in range(n_observations):
        for cell in range(n_grid):
            region = labels[cell]
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


def _bounded_uniform_log_prior(
    values: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> float:
    """Return a normalized independent bounded-uniform log density."""
    if values.shape != lower.shape or values.shape != upper.shape:
        return -np.inf
    if np.any(values < lower) or np.any(values > upper):
        return -np.inf
    return float(-np.log(upper - lower).sum())


def _prepare_error_state(
    problem: TransDimensionalProblem,
    mismatch_sd: ArrayLike | None,
    correlation_timescale: ArrayLike | None,
) -> tuple[FloatArray, FloatArray]:
    """Validate and own configured error parameters or return neutral vectors."""
    error_model = problem.error_model
    if error_model is None:
        if mismatch_sd is not None or correlation_timescale is not None:
            raise ValueError("error parameters require an inferred error model.")
        return _empty_readonly_float_array(), _empty_readonly_float_array()
    if mismatch_sd is None or correlation_timescale is None:
        raise ValueError("mismatch_sd and correlation_timescale are required for an inferred error model.")
    mismatch = _readonly_float_array(mismatch_sd, name="mismatch_sd")
    timescale = _readonly_float_array(
        correlation_timescale,
        name="correlation_timescale",
    )
    if mismatch.shape != (error_model.data.n_mismatch_groups,):
        raise ValueError("mismatch_sd must have one value per mismatch group.")
    if timescale.shape != (error_model.data.n_tau_parameters,):
        raise ValueError("correlation_timescale must have one value per timescale parameter.")
    if np.any(mismatch <= 0.0) or np.any(timescale <= 0.0):
        raise ValueError("error-model parameters must be strictly positive.")
    return mismatch, timescale


def _prepare_hierarchy_state(
    problem: TransDimensionalProblem,
    coefficient_prior_mean: float | None,
    coefficient_prior_sd: float | None,
) -> tuple[float, float]:
    """Return validated log arithmetic moments for the dynamic prior."""
    from openghg_inversions.experimental.rjmcmc.hierarchy import (
        arithmetic_moments_to_log_state,
    )

    if (coefficient_prior_mean is None) != (coefficient_prior_sd is None):
        raise ValueError("coefficient_prior_mean and coefficient_prior_sd must be supplied together.")
    if problem.coefficient_hierarchy is None:
        if coefficient_prior_mean is not None:
            raise ValueError("shared coefficient moments require coefficient_hierarchy.")
        return 0.0, 0.0
    mean = problem.coefficient_prior_mean if coefficient_prior_mean is None else coefficient_prior_mean
    standard_deviation = (
        problem.coefficient_prior_sd if coefficient_prior_sd is None else coefficient_prior_sd
    )
    return arithmetic_moments_to_log_state(mean, standard_deviation)


def _evaluate_target_terms(
    problem: TransDimensionalProblem,
    residual: FloatArray,
    coefficients: FloatArray,
    k: int,
    fixed_coefficients: FloatArray,
    mismatch_sd: FloatArray,
    correlation_timescale: FloatArray,
    eta: float,
    zeta: float,
    backend: Backend,
) -> tuple[float, float, float, float, float]:
    """Evaluate all likelihood and coefficient/error prior target factors."""
    if problem.error_model is None:
        likelihood_function = (
            gaussian_log_likelihood_numpy if backend == "numpy" else gaussian_log_likelihood_numba
        )
        log_likelihood = likelihood_function(residual, problem.observation_sd)
        log_error_model_prior = 0.0
    else:
        error_model = problem.error_model
        ou_function = ou_log_likelihood_numpy if backend == "numpy" else ou_log_likelihood_numba
        log_likelihood = ou_function(
            residual,
            error_model.data,
            mismatch_sd,
            correlation_timescale,
        )
        log_error_model_prior = _bounded_uniform_log_prior(
            mismatch_sd,
            error_model.mismatch_sd_prior_lower,
            error_model.mismatch_sd_prior_upper,
        ) + _bounded_uniform_log_prior(
            correlation_timescale,
            error_model.correlation_timescale_prior_lower,
            error_model.correlation_timescale_prior_upper,
        )

    if problem.coefficient_hierarchy is None:
        coefficient_prior_function = (
            lognormal_coefficient_log_prior_numpy
            if backend == "numpy"
            else lognormal_coefficient_log_prior_numba
        )
        log_coefficient_prior = coefficient_prior_function(
            coefficients,
            k,
            problem.coefficient_prior_mean,
            problem.coefficient_prior_sd,
        )
        log_coefficient_hyperprior = 0.0
    else:
        from openghg_inversions.experimental.rjmcmc.hierarchy import (
            shared_coefficient_log_prior_numba,
            shared_coefficient_log_prior_numpy,
            shared_hyperprior_log_density_numba,
            shared_hyperprior_log_density_numpy,
        )

        hierarchy = problem.coefficient_hierarchy
        if backend == "numpy":
            log_coefficient_prior = shared_coefficient_log_prior_numpy(
                coefficients,
                k,
                eta,
                zeta,
            )
            log_coefficient_hyperprior = shared_hyperprior_log_density_numpy(
                eta,
                zeta,
                hierarchy,
            )
        else:
            log_coefficient_prior = shared_coefficient_log_prior_numba(
                coefficients,
                k,
                eta,
                zeta,
            )
            log_coefficient_hyperprior = shared_hyperprior_log_density_numba(
                eta,
                zeta,
                hierarchy.mean_hyperprior_median,
                hierarchy.mean_hyperprior_log_sd,
                hierarchy.sd_hyperprior_median,
                hierarchy.sd_hyperprior_log_sd,
            )

    if problem.fixed_block is None:
        log_fixed_coefficient_prior = 0.0
    else:
        fixed_prior_function = (
            lognormal_coefficient_log_prior_numpy
            if backend == "numpy"
            else lognormal_coefficient_log_prior_numba
        )
        log_fixed_coefficient_prior = sum(
            fixed_prior_function(
                fixed_coefficients[index : index + 1],
                1,
                float(problem.fixed_block.coefficient_prior_mean[index]),
                float(problem.fixed_block.coefficient_prior_sd[index]),
            )
            for index in range(problem.n_fixed_coefficients)
        )
    return (
        float(log_likelihood),
        float(log_coefficient_prior),
        float(log_fixed_coefficient_prior),
        float(log_error_model_prior),
        float(log_coefficient_hyperprior),
    )


def build_state(
    problem: TransDimensionalProblem,
    active_nuclei: ArrayLike,
    active_coefficients: ArrayLike,
    *,
    fixed_coefficients: ArrayLike | None = None,
    mismatch_sd: ArrayLike | None = None,
    correlation_timescale: ArrayLike | None = None,
    coefficient_prior_mean: float | None = None,
    coefficient_prior_sd: float | None = None,
    backend: Backend = "numpy",
) -> TransDimensionalState:
    """Build and validate a complete sampler state from its active values.

    Args:
        problem: Immutable numerical problem and target specification.
        active_nuclei: Unique flattened grid indices for active nuclei.
        active_coefficients: Finite coefficients paired with the supplied
            nuclei. Nonpositive active values produce a state with negative
            infinite coefficient-prior density and log target.
        fixed_coefficients: Explicit finite coefficients for the optional
            always-active fixed block. Required exactly when that block is
            nonempty. Nonpositive values produce negative infinite prior density.
        mismatch_sd: Positive inferred OU amplitudes, required exactly when an
            inferred error model is configured.
        correlation_timescale: Positive inferred OU timescales, required
            exactly when an inferred error model is configured.
        coefficient_prior_mean: Optional shared arithmetic mean for the
            partially pooled dynamic coefficient prior. Defaults to the
            problem's declared coefficient prior mean.
        coefficient_prior_sd: Optional shared arithmetic SD paired with
            ``coefficient_prior_mean``.
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

    if fixed_coefficients is None:
        supplied_fixed_coefficients = np.empty(0, dtype=np.float64)
        if problem.n_fixed_coefficients:
            raise ValueError("fixed_coefficients are required when the problem has a fixed block.")
    else:
        supplied_fixed_coefficients = np.asarray(fixed_coefficients, dtype=np.float64)
        if supplied_fixed_coefficients.ndim != 1:
            raise ValueError("fixed_coefficients must be one-dimensional.")
        if supplied_fixed_coefficients.shape != (problem.n_fixed_coefficients,):
            raise ValueError("fixed_coefficients must have one value per fixed block column.")
        if not np.all(np.isfinite(supplied_fixed_coefficients)):
            raise ValueError("fixed_coefficients must be finite.")
    owned_fixed_coefficients = np.array(supplied_fixed_coefficients, dtype=np.float64, copy=True)
    owned_mismatch_sd, owned_correlation_timescale = _prepare_error_state(
        problem,
        mismatch_sd,
        correlation_timescale,
    )
    eta, zeta = _prepare_hierarchy_state(
        problem,
        coefficient_prior_mean,
        coefficient_prior_sd,
    )

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

    dynamic_prediction = design[:, :k] @ coefficients[:k]
    fixed_prediction = np.array(problem.fixed_offset, dtype=np.float64, copy=True)
    if problem.fixed_block is not None:
        fixed_prediction += problem.fixed_block.design @ owned_fixed_coefficients
    prediction = dynamic_prediction + fixed_prediction
    residual = prediction - problem.observations
    (
        log_likelihood,
        log_coefficient_prior,
        log_fixed_coefficient_prior,
        log_error_model_prior,
        log_coefficient_hyperprior,
    ) = _evaluate_target_terms(
        problem,
        residual,
        coefficients,
        k,
        owned_fixed_coefficients,
        owned_mismatch_sd,
        owned_correlation_timescale,
        eta,
        zeta,
        backend,
    )

    for array in (
        nuclei,
        coefficients,
        labels,
        design,
        owned_fixed_coefficients,
        dynamic_prediction,
        fixed_prediction,
        prediction,
        residual,
        owned_mismatch_sd,
        owned_correlation_timescale,
    ):
        array.setflags(write=False)
    return TransDimensionalState(
        k=k,
        nuclei=nuclei,
        coefficients=coefficients,
        labels=labels,
        design=design,
        fixed_coefficients=owned_fixed_coefficients,
        dynamic_prediction=dynamic_prediction,
        fixed_prediction=fixed_prediction,
        prediction=prediction,
        residual=residual,
        log_likelihood=float(log_likelihood),
        log_coefficient_prior=float(log_coefficient_prior),
        log_fixed_coefficient_prior=float(log_fixed_coefficient_prior),
        log_k_prior=float(problem.log_k_prior[k - problem.k_min]),
        log_nucleus_prior=uniform_nucleus_set_log_prior(problem.n_grid_cells, k),
        mismatch_sd=owned_mismatch_sd,
        correlation_timescale=owned_correlation_timescale,
        eta=eta,
        zeta=zeta,
        log_error_model_prior=log_error_model_prior,
        log_coefficient_hyperprior=log_coefficient_hyperprior,
    )


def _canonical_structural_values(
    problem: TransDimensionalProblem,
    active_nuclei: ArrayLike,
    active_coefficients: ArrayLike,
) -> tuple[int, IntArray, FloatArray]:
    """Validate active values and return canonical padded state arrays.

    The returned tuple contains ``k``, a sorted ``(k_max,)`` nucleus array
    padded with ``-1``, and a correspondingly reordered coefficient array
    padded with zeros. Malformed dimensions, counts, indices, duplicates, and
    non-finite coefficients raise ``ValueError``.
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
    nuclei = np.full(problem.k_max, -1, dtype=np.int64)
    coefficients = np.zeros(problem.k_max, dtype=np.float64)
    nuclei[:k] = supplied_nuclei[order]
    coefficients[:k] = supplied_coefficients[order]
    return k, nuclei, coefficients


def _incremental_source_is_compatible(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
) -> bool:
    """Return whether a source state has the cache shapes needed for reuse."""
    if state.capacity != problem.k_max or not problem.k_min <= state.k <= problem.k_max:
        return False
    if state.nuclei.shape != (problem.k_max,) or state.coefficients.shape != (problem.k_max,):
        return False
    if state.labels.shape != (problem.n_grid_cells,):
        return False
    if state.design.shape != (problem.n_observations, problem.k_max):
        return False
    if state.fixed_coefficients.shape != (problem.n_fixed_coefficients,):
        return False
    if state.fixed_prediction.shape != (problem.n_observations,):
        return False
    expected_mismatch_count = 0 if problem.error_model is None else problem.error_model.data.n_mismatch_groups
    expected_timescale_count = 0 if problem.error_model is None else problem.error_model.data.n_tau_parameters
    if state.mismatch_sd.shape != (expected_mismatch_count,):
        return False
    if state.correlation_timescale.shape != (expected_timescale_count,):
        return False
    active_nuclei = state.nuclei[: state.k]
    if (
        np.any(active_nuclei < 0)
        or np.any(active_nuclei >= problem.n_grid_cells)
        or np.any(active_nuclei[1:] <= active_nuclei[:-1])
    ):
        return False
    return bool(np.all((state.labels >= 0) & (state.labels < state.k)))


def _nucleus_position_maps(
    old_nuclei: IntArray,
    new_nuclei: IntArray,
) -> tuple[IntArray, IntArray, int]:
    """Map old and new canonical positions by persistent nucleus identity.

    Missing identities use ``-1`` in both position maps. The final scalar is
    the unique added position, or ``-1`` when the edit has zero or multiple
    additions; callers separately count removals and additions.
    """
    old_to_new = np.full(old_nuclei.size, -1, dtype=np.int64)
    new_to_old = np.full(new_nuclei.size, -1, dtype=np.int64)
    old_position = 0
    new_position = 0
    while old_position < old_nuclei.size and new_position < new_nuclei.size:
        old_nucleus = old_nuclei[old_position]
        new_nucleus = new_nuclei[new_position]
        if old_nucleus == new_nucleus:
            old_to_new[old_position] = new_position
            new_to_old[new_position] = old_position
            old_position += 1
            new_position += 1
        elif old_nucleus < new_nucleus:
            old_position += 1
        else:
            new_position += 1
    added_positions = np.flatnonzero(new_to_old < 0)
    added_position = int(added_positions[0]) if added_positions.size == 1 else -1
    return old_to_new, new_to_old, added_position


def _assign_cells_incremental_numpy(
    grid_coordinates: FloatArray,
    old_labels: IntArray,
    new_nuclei: IntArray,
    old_to_new: IntArray,
    added_position: int,
) -> IntArray:
    """Update exact Voronoi labels after one edit using NumPy array kernels.

    Arrays use cell-major coordinates and canonical region-position labels.
    Cells whose old owner was removed search every final nucleus; all other
    cells retain their remapped owner unless the one added nucleus is nearer.
    Exact ties choose the lower canonical position, matching ``build_state``.
    """
    labels = np.asarray(old_to_new[old_labels], dtype=np.int64)
    needs_full_search = labels < 0
    if np.any(needs_full_search):
        cell_coordinates = grid_coordinates[needs_full_search]
        active_coordinates = grid_coordinates[new_nuclei]
        squared_distance = np.sum(
            np.square(cell_coordinates[:, np.newaxis, :] - active_coordinates[np.newaxis, :, :]),
            axis=2,
        )
        labels[needs_full_search] = np.argmin(squared_distance, axis=1)

    if added_position >= 0:
        stable_cells = np.flatnonzero(~needs_full_search)
        stable_coordinates = grid_coordinates[stable_cells]
        owner_coordinates = grid_coordinates[new_nuclei[labels[stable_cells]]]
        added_coordinates = grid_coordinates[new_nuclei[added_position]]
        owner_squared_distance = np.sum(
            np.square(stable_coordinates - owner_coordinates),
            axis=1,
        )
        added_squared_distance = np.sum(
            np.square(stable_coordinates - added_coordinates),
            axis=1,
        )
        added_wins = (added_squared_distance < owner_squared_distance) | (
            (added_squared_distance == owner_squared_distance) & (added_position < labels[stable_cells])
        )
        labels[stable_cells[added_wins]] = added_position
    return labels


@njit(cache=True)
def _assign_cells_incremental_numba(
    grid_coordinates: FloatArray,
    old_labels: IntArray,
    new_nuclei: IntArray,
    old_to_new: IntArray,
    added_position: int,
) -> IntArray:
    """Update exact Voronoi labels after one structural edit with Numba.

    Cells whose old owner was removed search every final nucleus; surviving
    cells compare only their remapped owner and the optional added nucleus.
    The strict distance comparison plus canonical-position tie check reproduces
    the complete assignment kernel.
    """
    n_grid, n_dimensions = grid_coordinates.shape
    labels = np.empty(n_grid, dtype=np.int64)
    for cell in range(n_grid):
        mapped_owner = old_to_new[old_labels[cell]]
        if mapped_owner < 0:
            best_region = 0
            best_distance = np.inf
            for region in range(new_nuclei.size):
                distance = 0.0
                nucleus = new_nuclei[region]
                for dimension in range(n_dimensions):
                    difference = grid_coordinates[cell, dimension] - grid_coordinates[nucleus, dimension]
                    distance += difference * difference
                if distance < best_distance:
                    best_distance = distance
                    best_region = region
            labels[cell] = best_region
            continue

        best_region = mapped_owner
        if added_position >= 0:
            best_distance = 0.0
            added_distance = 0.0
            best_nucleus = new_nuclei[best_region]
            added_nucleus = new_nuclei[added_position]
            for dimension in range(n_dimensions):
                best_difference = (
                    grid_coordinates[cell, dimension] - grid_coordinates[best_nucleus, dimension]
                )
                added_difference = (
                    grid_coordinates[cell, dimension] - grid_coordinates[added_nucleus, dimension]
                )
                best_distance += best_difference * best_difference
                added_distance += added_difference * added_difference
            if added_distance < best_distance or (
                added_distance == best_distance and added_position < best_region
            ):
                best_region = added_position
        labels[cell] = best_region
    return labels


def _membership_changed_regions_numpy(
    old_nuclei: IntArray,
    old_labels: IntArray,
    new_nuclei: IntArray,
    new_labels: IntArray,
    old_to_new: IntArray,
) -> NDArray[np.bool_]:
    """Return a final-region mask for changed owner identities with NumPy.

    A region is marked when it gains or loses any cell after comparing nucleus
    identities rather than canonical positions. The result has one entry per
    final active nucleus.
    """
    changed_regions = np.zeros(new_nuclei.size, dtype=np.bool_)
    membership_changed = old_nuclei[old_labels] != new_nuclei[new_labels]
    changed_regions[new_labels[membership_changed]] = True
    mapped_old_regions = old_to_new[old_labels[membership_changed]]
    changed_regions[mapped_old_regions[mapped_old_regions >= 0]] = True
    return changed_regions


@njit(cache=True)
def _membership_changed_regions_numba(
    old_nuclei: IntArray,
    old_labels: IntArray,
    new_nuclei: IntArray,
    new_labels: IntArray,
    old_to_new: IntArray,
) -> NDArray[np.bool_]:
    """Return a final-region mask for changed owner identities with Numba.

    The identity-based comparison is invariant to canonical column reordering;
    both surviving donors and final recipients are marked.
    """
    changed_regions = np.zeros(new_nuclei.size, dtype=np.bool_)
    for cell in range(old_labels.size):
        old_region = old_labels[cell]
        new_region = new_labels[cell]
        if old_nuclei[old_region] == new_nuclei[new_region]:
            continue
        mapped_old_region = old_to_new[old_region]
        if mapped_old_region >= 0:
            changed_regions[mapped_old_region] = True
        changed_regions[new_region] = True
    return changed_regions


def _aggregate_changed_design_numpy(
    sensitivities: FloatArray,
    labels: IntArray,
    changed_cells: IntArray,
    design: FloatArray,
) -> None:
    """Mutate a zeroed design cache with changed cells using NumPy.

    ``changed_cells`` must be in ascending global-cell order. This preserves
    the full builder's floating-point summation order for every rebuilt column.
    """
    for cell in changed_cells:
        design[:, labels[cell]] += sensitivities[:, cell]


@njit(cache=True)
def _aggregate_changed_design_numba(
    sensitivities: FloatArray,
    labels: IntArray,
    changed_cells: IntArray,
    design: FloatArray,
) -> None:
    """Mutate a zeroed design cache with changed cells using Numba.

    ``changed_cells`` must be in ascending global-cell order. Observation-major
    traversal matches the complete Numba aggregation kernel exactly.
    """
    n_observations = sensitivities.shape[0]
    for observation in range(n_observations):
        for changed_index in range(changed_cells.size):
            cell = changed_cells[changed_index]
            region = labels[cell]
            design[observation, region] += sensitivities[observation, cell]


def _rebuild_state_preserving_optional_parameters(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    active_nuclei: ArrayLike,
    active_coefficients: ArrayLike,
    backend: Backend,
) -> TransDimensionalState:
    """Call the full builder while preserving configured fixed dimensions."""
    if problem.error_model is None and problem.coefficient_hierarchy is None:
        return build_state(
            problem,
            active_nuclei,
            active_coefficients,
            fixed_coefficients=state.fixed_coefficients,
            backend=backend,
        )
    candidate = build_state(
        problem,
        active_nuclei,
        active_coefficients,
        fixed_coefficients=state.fixed_coefficients,
        mismatch_sd=state.mismatch_sd if problem.error_model is not None else None,
        correlation_timescale=(state.correlation_timescale if problem.error_model is not None else None),
        coefficient_prior_mean=(
            float(np.exp(state.eta)) if problem.coefficient_hierarchy is not None else None
        ),
        coefficient_prior_sd=(
            float(np.exp(state.zeta)) if problem.coefficient_hierarchy is not None else None
        ),
        backend=backend,
    )
    if problem.coefficient_hierarchy is not None and (
        candidate.eta != state.eta or candidate.zeta != state.zeta
    ):
        candidate = update_shared_hierarchy_state(
            problem,
            candidate,
            proposed_eta=state.eta,
            proposed_zeta=state.zeta,
            backend=backend,
        )
    return candidate


def update_structural_state(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    active_nuclei: ArrayLike,
    active_coefficients: ArrayLike,
    *,
    backend: Backend = "numpy",
) -> TransDimensionalState:
    """Build a structural candidate by incrementally updating exact caches.

    The final active nuclei and coefficients are the complete specification of
    the candidate. They may describe one insertion, one deletion, or one moved
    nucleus (one removal plus one insertion) relative to ``state``. Nuclei are
    canonically sorted and equal-distance ties choose the first canonical
    nucleus, exactly as in :func:`assign_cells_numba`.

    Design columns are matched by nucleus identity across canonical reordering.
    Columns whose final cell membership is unchanged are copied from ``state``;
    every changed final column is recomputed from zero in ascending fine-cell
    order. Consequently the design is independent of the edit path and matches
    a complete :func:`build_state` aggregation exactly. Unsupported multi-edit
    inputs or incompatible source-cache shapes safely use ``build_state``. The
    inputs and source state are not mutated; the result reuses the source's
    immutable fixed coefficients and fixed-prediction cache.

    Args:
        problem: Immutable numerical problem and target specification.
        state: Valid immutable source state associated with ``problem``.
        active_nuclei: Final unique flattened nucleus indices. The supplied
            order need not be canonical.
        active_coefficients: Finite final coefficients paired with the supplied
            nuclei, including the coefficient associated with a moved nucleus.
        backend: Numerical kernels used for incremental geometry, aggregation,
            likelihood, and priors.

    Returns:
        A canonical immutable candidate state with exact target caches.

    Raises:
        TypeError: If ``problem`` or ``state`` has the wrong type.
        ValueError: If final active values or ``backend`` are malformed.
    """
    if not isinstance(problem, TransDimensionalProblem):
        raise TypeError("problem must be a TransDimensionalProblem.")
    if not isinstance(state, TransDimensionalState):
        raise TypeError("state must be a TransDimensionalState.")
    if backend not in ("numpy", "numba"):
        raise ValueError("backend must be 'numpy' or 'numba'.")
    k, nuclei, coefficients = _canonical_structural_values(
        problem,
        active_nuclei,
        active_coefficients,
    )
    if not _incremental_source_is_compatible(problem, state):
        return _rebuild_state_preserving_optional_parameters(
            problem,
            state,
            nuclei[:k],
            coefficients[:k],
            backend,
        )

    old_nuclei = state.nuclei[: state.k]
    new_nuclei = nuclei[:k]
    old_to_new, new_to_old, added_position = _nucleus_position_maps(
        old_nuclei,
        new_nuclei,
    )
    removed_count = int(np.count_nonzero(old_to_new < 0))
    added_count = int(np.count_nonzero(new_to_old < 0))
    if removed_count > 1 or added_count > 1:
        return _rebuild_state_preserving_optional_parameters(
            problem,
            state,
            new_nuclei,
            coefficients[:k],
            backend,
        )
    if added_count == 0:
        added_position = -1

    if backend == "numpy":
        labels = _assign_cells_incremental_numpy(
            problem.grid_coordinates,
            state.labels,
            new_nuclei,
            old_to_new,
            added_position,
        )
    else:
        labels = _assign_cells_incremental_numba(
            problem.grid_coordinates,
            state.labels,
            new_nuclei,
            old_to_new,
            added_position,
        )

    if backend == "numpy":
        changed_regions = _membership_changed_regions_numpy(
            old_nuclei,
            state.labels,
            new_nuclei,
            labels,
            old_to_new,
        )
    else:
        changed_regions = _membership_changed_regions_numba(
            old_nuclei,
            state.labels,
            new_nuclei,
            labels,
            old_to_new,
        )
    changed_cells = np.flatnonzero(changed_regions[labels]).astype(np.int64, copy=False)
    if changed_cells.size == problem.n_grid_cells:
        if backend == "numpy":
            design = aggregate_design_numpy(problem.sensitivities, labels, k, problem.k_max)
        else:
            design = aggregate_design_numba(problem.sensitivities, labels, k, problem.k_max)
    else:
        design = np.zeros((problem.n_observations, problem.k_max), dtype=np.float64)
        for new_position, old_position in enumerate(new_to_old):
            if old_position >= 0 and not changed_regions[new_position]:
                design[:, new_position] = state.design[:, old_position]
        if backend == "numpy":
            _aggregate_changed_design_numpy(problem.sensitivities, labels, changed_cells, design)
        else:
            _aggregate_changed_design_numba(problem.sensitivities, labels, changed_cells, design)

    dynamic_prediction = design[:, :k] @ coefficients[:k]
    fixed_prediction = state.fixed_prediction
    prediction = dynamic_prediction + fixed_prediction
    residual = prediction - problem.observations
    (
        log_likelihood,
        log_coefficient_prior,
        log_fixed_coefficient_prior,
        log_error_model_prior,
        log_coefficient_hyperprior,
    ) = _evaluate_target_terms(
        problem,
        residual,
        coefficients,
        k,
        state.fixed_coefficients,
        state.mismatch_sd,
        state.correlation_timescale,
        state.eta,
        state.zeta,
        backend,
    )

    for array in (nuclei, coefficients, labels, design, dynamic_prediction, prediction, residual):
        array.setflags(write=False)
    return TransDimensionalState(
        k=k,
        nuclei=nuclei,
        coefficients=coefficients,
        labels=labels,
        design=design,
        fixed_coefficients=state.fixed_coefficients,
        dynamic_prediction=dynamic_prediction,
        fixed_prediction=fixed_prediction,
        prediction=prediction,
        residual=residual,
        log_likelihood=float(log_likelihood),
        log_coefficient_prior=float(log_coefficient_prior),
        log_fixed_coefficient_prior=float(log_fixed_coefficient_prior),
        log_k_prior=float(problem.log_k_prior[k - problem.k_min]),
        log_nucleus_prior=uniform_nucleus_set_log_prior(problem.n_grid_cells, k),
        mismatch_sd=state.mismatch_sd,
        correlation_timescale=state.correlation_timescale,
        eta=state.eta,
        zeta=state.zeta,
        log_error_model_prior=log_error_model_prior,
        log_coefficient_hyperprior=log_coefficient_hyperprior,
    )


def _validate_cached_coefficient_update(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    position: int,
    value: float,
    fixed: bool,
    backend: Backend,
) -> tuple[int, float]:
    """Validate the shape-level contract for a cache-preserving update."""
    if not isinstance(problem, TransDimensionalProblem):
        raise TypeError("problem must be a TransDimensionalProblem.")
    if not isinstance(state, TransDimensionalState):
        raise TypeError("state must be a TransDimensionalState.")
    if backend not in ("numpy", "numba"):
        raise ValueError("backend must be 'numpy' or 'numba'.")
    if state.capacity != problem.k_max:
        raise ValueError("state capacity must equal problem.k_max.")
    if state.coefficients.shape != (problem.k_max,) or state.labels.shape != (problem.n_grid_cells,):
        raise ValueError("state dynamic arrays are incompatible with the problem.")
    if state.design.shape != (problem.n_observations, problem.k_max):
        raise ValueError("state design is incompatible with the problem.")
    if state.fixed_coefficients.shape != (problem.n_fixed_coefficients,):
        raise ValueError("state fixed coefficients are incompatible with the problem.")
    expected_observation_shape = (problem.n_observations,)
    for name in ("dynamic_prediction", "fixed_prediction", "prediction", "residual"):
        if getattr(state, name).shape != expected_observation_shape:
            raise ValueError(f"state {name} is incompatible with the problem.")
    if not problem.k_min <= state.k <= problem.k_max:
        raise ValueError("state.k must lie within the problem's permitted range.")
    if isinstance(position, bool) or not isinstance(position, (int, np.integer)):
        raise ValueError("position must be an integer coefficient position.")
    validated_position = int(position)
    count = problem.n_fixed_coefficients if fixed else state.k
    if not 0 <= validated_position < count:
        coefficient_kind = "fixed" if fixed else "active"
        raise ValueError(f"position must select an {coefficient_kind} coefficient.")
    if isinstance(value, bool):
        raise ValueError("value must be finite.")
    validated_value = float(value)
    if not np.isfinite(validated_value):
        raise ValueError("value must be finite.")
    return validated_position, validated_value


def update_dynamic_coefficient_state(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    coefficient_position: int,
    proposed_coefficient: float,
    backend: Backend = "numpy",
) -> TransDimensionalState:
    """Update one dynamic coefficient while preserving geometry caches.

    The nuclei, Voronoi labels, aggregated design, fixed coefficients, and
    fixed prediction do not depend on a dynamic coefficient value and are
    shared with the source state. Prediction and target caches are recomputed
    in the same arithmetic order as :func:`build_state`, avoiding a new
    fine-grid assignment and design aggregation without changing the target.

    Args:
        problem: Immutable numerical problem and target specification.
        state: Valid immutable source state associated with ``problem``.
        coefficient_position: Zero-based active coefficient position.
        proposed_coefficient: Finite replacement value. Nonpositive values
            remain representable and receive negative-infinite prior density,
            matching :func:`build_state`.
        backend: Numerical likelihood and prior implementation.

    Returns:
        A complete immutable state with unchanged geometry caches shared by
        reference and all coefficient-dependent caches recomputed.
    """
    position, value = _validate_cached_coefficient_update(
        problem,
        state,
        position=coefficient_position,
        value=proposed_coefficient,
        fixed=False,
        backend=backend,
    )
    coefficients = np.array(state.coefficients, dtype=np.float64, copy=True)
    coefficients[position] = value
    dynamic_prediction = state.design[:, : state.k] @ coefficients[: state.k]
    prediction = dynamic_prediction + state.fixed_prediction
    residual = prediction - problem.observations
    (
        log_likelihood,
        log_coefficient_prior,
        log_fixed_coefficient_prior,
        log_error_model_prior,
        log_coefficient_hyperprior,
    ) = _evaluate_target_terms(
        problem,
        residual,
        coefficients,
        state.k,
        state.fixed_coefficients,
        state.mismatch_sd,
        state.correlation_timescale,
        state.eta,
        state.zeta,
        backend,
    )
    for array in (coefficients, dynamic_prediction, prediction, residual):
        array.setflags(write=False)
    return replace(
        state,
        coefficients=coefficients,
        dynamic_prediction=dynamic_prediction,
        prediction=prediction,
        residual=residual,
        log_likelihood=float(log_likelihood),
        log_coefficient_prior=float(log_coefficient_prior),
        log_fixed_coefficient_prior=float(log_fixed_coefficient_prior),
        log_error_model_prior=log_error_model_prior,
        log_coefficient_hyperprior=log_coefficient_hyperprior,
    )


def update_fixed_coefficient_state(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    coefficient_position: int,
    proposed_coefficient: float,
    backend: Backend = "numpy",
) -> TransDimensionalState:
    """Update one fixed-block coefficient while preserving dynamic caches.

    Dynamic coefficients, nuclei, Voronoi labels, the aggregated dynamic
    design, and the dynamic prediction are shared with the source state. The
    fixed prediction and target caches are recomputed in the same arithmetic
    order as :func:`build_state`.

    Args:
        problem: Immutable problem containing a nonempty fixed design block.
        state: Valid immutable source state associated with ``problem``.
        coefficient_position: Zero-based fixed-block coefficient position.
        proposed_coefficient: Finite replacement value. Nonpositive values
            receive negative-infinite prior density, matching ``build_state``.
        backend: Numerical likelihood and prior implementation.

    Returns:
        A complete immutable state with unchanged geometry and dynamic caches
        shared by reference.
    """
    position, value = _validate_cached_coefficient_update(
        problem,
        state,
        position=coefficient_position,
        value=proposed_coefficient,
        fixed=True,
        backend=backend,
    )
    fixed_block = problem.fixed_block
    if fixed_block is None:  # guarded by the validated nonzero fixed count
        raise ValueError("A fixed coefficient update requires a fixed block.")
    fixed_coefficients = np.array(state.fixed_coefficients, dtype=np.float64, copy=True)
    fixed_coefficients[position] = value
    fixed_prediction = np.array(problem.fixed_offset, dtype=np.float64, copy=True)
    fixed_prediction += fixed_block.design @ fixed_coefficients
    prediction = state.dynamic_prediction + fixed_prediction
    residual = prediction - problem.observations
    (
        log_likelihood,
        log_coefficient_prior,
        log_fixed_coefficient_prior,
        log_error_model_prior,
        log_coefficient_hyperprior,
    ) = _evaluate_target_terms(
        problem,
        residual,
        state.coefficients,
        state.k,
        fixed_coefficients,
        state.mismatch_sd,
        state.correlation_timescale,
        state.eta,
        state.zeta,
        backend,
    )
    for array in (fixed_coefficients, fixed_prediction, prediction, residual):
        array.setflags(write=False)
    return replace(
        state,
        fixed_coefficients=fixed_coefficients,
        fixed_prediction=fixed_prediction,
        prediction=prediction,
        residual=residual,
        log_likelihood=float(log_likelihood),
        log_coefficient_prior=float(log_coefficient_prior),
        log_fixed_coefficient_prior=float(log_fixed_coefficient_prior),
        log_error_model_prior=log_error_model_prior,
        log_coefficient_hyperprior=log_coefficient_hyperprior,
    )


def update_error_model_state(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    mismatch_sd_position: int | None = None,
    proposed_mismatch_sd: float | None = None,
    correlation_timescale_position: int | None = None,
    proposed_correlation_timescale: float | None = None,
    backend: Backend = "numpy",
) -> TransDimensionalState:
    """Update one inferred OU parameter while preserving model-state caches.

    Exactly one complete position/value pair must be supplied. Geometry,
    coefficients, predictions, and residuals are shared with the source state;
    only the selected error vector, OU likelihood, and normalized bounded-
    uniform error prior are recomputed.

    Args:
        problem: Problem containing an inferred OU error-model configuration.
        state: Valid source state associated with ``problem``.
        mismatch_sd_position: Optional zero-based mismatch-group position.
        proposed_mismatch_sd: Positive replacement mismatch amplitude.
        correlation_timescale_position: Optional zero-based timescale position.
        proposed_correlation_timescale: Positive replacement timescale.
        backend: Numerical OU likelihood implementation.

    Returns:
        Immutable candidate sharing every error-independent cache.

    Raises:
        TypeError: If ``problem`` or ``state`` has the wrong type.
        ValueError: If no inferred error model exists, the backend is unknown,
            or the requested update is malformed or outside positive support.
    """
    if not isinstance(problem, TransDimensionalProblem):
        raise TypeError("problem must be a TransDimensionalProblem.")
    if not isinstance(state, TransDimensionalState):
        raise TypeError("state must be a TransDimensionalState.")
    if backend not in ("numpy", "numba"):
        raise ValueError("backend must be 'numpy' or 'numba'.")
    error_model = problem.error_model
    if error_model is None:
        raise ValueError("An error-model update requires an inferred error model.")
    mismatch_pair = mismatch_sd_position is not None or proposed_mismatch_sd is not None
    timescale_pair = correlation_timescale_position is not None or proposed_correlation_timescale is not None
    if mismatch_pair == timescale_pair:
        raise ValueError("Supply exactly one complete error-parameter position/value pair.")

    mismatch_sd = state.mismatch_sd
    correlation_timescale = state.correlation_timescale
    if mismatch_pair:
        if mismatch_sd_position is None or proposed_mismatch_sd is None:
            raise ValueError("A mismatch update requires both position and proposed value.")
        if isinstance(mismatch_sd_position, bool) or not isinstance(
            mismatch_sd_position,
            (int, np.integer),
        ):
            raise ValueError("mismatch_sd_position must be an integer.")
        position = int(mismatch_sd_position)
        if not 0 <= position < error_model.data.n_mismatch_groups:
            raise ValueError("mismatch_sd_position is outside the configured groups.")
        value = float(proposed_mismatch_sd)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("proposed_mismatch_sd must be finite and strictly positive.")
        mismatch_sd = np.array(state.mismatch_sd, dtype=np.float64, copy=True)
        mismatch_sd[position] = value
        mismatch_sd.setflags(write=False)
    else:
        if correlation_timescale_position is None or proposed_correlation_timescale is None:
            raise ValueError("A timescale update requires both position and proposed value.")
        if isinstance(correlation_timescale_position, bool) or not isinstance(
            correlation_timescale_position,
            (int, np.integer),
        ):
            raise ValueError("correlation_timescale_position must be an integer.")
        position = int(correlation_timescale_position)
        if not 0 <= position < error_model.data.n_tau_parameters:
            raise ValueError("correlation_timescale_position is outside the configured parameters.")
        value = float(proposed_correlation_timescale)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("proposed_correlation_timescale must be finite and strictly positive.")
        correlation_timescale = np.array(
            state.correlation_timescale,
            dtype=np.float64,
            copy=True,
        )
        correlation_timescale[position] = value
        correlation_timescale.setflags(write=False)

    ou_function = ou_log_likelihood_numpy if backend == "numpy" else ou_log_likelihood_numba
    log_likelihood = ou_function(
        state.residual,
        error_model.data,
        mismatch_sd,
        correlation_timescale,
    )
    log_error_model_prior = _bounded_uniform_log_prior(
        mismatch_sd,
        error_model.mismatch_sd_prior_lower,
        error_model.mismatch_sd_prior_upper,
    ) + _bounded_uniform_log_prior(
        correlation_timescale,
        error_model.correlation_timescale_prior_lower,
        error_model.correlation_timescale_prior_upper,
    )
    return replace(
        state,
        mismatch_sd=mismatch_sd,
        correlation_timescale=correlation_timescale,
        log_likelihood=float(log_likelihood),
        log_error_model_prior=log_error_model_prior,
    )


def update_shared_hierarchy_state(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    proposed_eta: float,
    proposed_zeta: float,
    backend: Backend = "numpy",
) -> TransDimensionalState:
    """Update the shared lognormal hierarchy without rebuilding other caches.

    Args:
        problem: Problem containing a shared coefficient hierarchy.
        state: Valid source state associated with ``problem``.
        proposed_eta: Finite log arithmetic coefficient-prior mean.
        proposed_zeta: Finite log arithmetic coefficient-prior SD.
        backend: Numerical hierarchy-kernel implementation.

    Returns:
        Immutable candidate with only hierarchy coordinates and their dynamic
        conditional-prior and hyperprior caches changed.

    Raises:
        TypeError: If ``problem`` or ``state`` has the wrong type.
        ValueError: If no hierarchy is configured, the backend is unknown, or
            either proposed log coordinate is nonfinite.
    """
    if not isinstance(problem, TransDimensionalProblem):
        raise TypeError("problem must be a TransDimensionalProblem.")
    if not isinstance(state, TransDimensionalState):
        raise TypeError("state must be a TransDimensionalState.")
    if backend not in ("numpy", "numba"):
        raise ValueError("backend must be 'numpy' or 'numba'.")
    hierarchy = problem.coefficient_hierarchy
    if hierarchy is None:
        raise ValueError("A hierarchy update requires coefficient_hierarchy.")
    eta = float(proposed_eta)
    zeta = float(proposed_zeta)
    if not np.isfinite(eta) or not np.isfinite(zeta):
        raise ValueError("proposed_eta and proposed_zeta must be finite.")
    from openghg_inversions.experimental.rjmcmc.hierarchy import (
        shared_coefficient_log_prior_numba,
        shared_coefficient_log_prior_numpy,
        shared_hyperprior_log_density_numba,
        shared_hyperprior_log_density_numpy,
    )

    if backend == "numpy":
        log_coefficient_prior = shared_coefficient_log_prior_numpy(
            state.coefficients,
            state.k,
            eta,
            zeta,
        )
        log_coefficient_hyperprior = shared_hyperprior_log_density_numpy(
            eta,
            zeta,
            hierarchy,
        )
    else:
        log_coefficient_prior = shared_coefficient_log_prior_numba(
            state.coefficients,
            state.k,
            eta,
            zeta,
        )
        log_coefficient_hyperprior = shared_hyperprior_log_density_numba(
            eta,
            zeta,
            hierarchy.mean_hyperprior_median,
            hierarchy.mean_hyperprior_log_sd,
            hierarchy.sd_hyperprior_median,
            hierarchy.sd_hyperprior_log_sd,
        )
    return replace(
        state,
        eta=eta,
        zeta=zeta,
        log_coefficient_prior=float(log_coefficient_prior),
        log_coefficient_hyperprior=float(log_coefficient_hyperprior),
    )
