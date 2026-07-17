"""Analytic Gaussian target for fixed-coordinate dyadic product spaces.

This module provides a deliberately reduced Gaussian model for validating
partition inference. The inner domain uses a dynamic dyadic partition and a
permanent root-and-contrast vector. A separate outer coefficient block remains
active under every partition and has its own prior covariance. Inactive inner
contrasts have normalized Gaussian pseudo-priors.

The construction is not the exact Bocquet projection with aggregation error:
its partition marginal likelihood is intentionally allowed to depend on the
partition. Synthetic observations can be generated around an explicit
prior-forward mean, so no boundary-condition product is required.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .contrast import TreeContrastLayout
from .multiscale import MultiscaleDesign
from .product_space import ProductSpaceState
from .state import PartitionState
from .tree import DyadicTree

PartitionLogPrior = Callable[[PartitionState], float]


@dataclass(frozen=True, slots=True, eq=False)
class GaussianConditional:
    """Conditional Gaussian law for active inner and fixed outer coefficients.

    Attributes:
        mean: Posterior mean. Active inner coordinates come first, followed by
            outer coefficients.
        covariance: Posterior covariance in the same order as :attr:`mean`.
        active_inner_indices: Permanent inner-coordinate indices represented by
            the leading entries of :attr:`mean`.
    """

    mean: np.ndarray
    covariance: np.ndarray
    active_inner_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate and freeze a directly constructed conditional law."""
        mean = _vector(self.mean, name="mean")
        covariance = _positive_definite_matrix(
            self.covariance,
            size=mean.size,
            name="covariance",
        )
        indices = tuple(self.active_inner_indices)
        if any(isinstance(index, bool) or not isinstance(index, (int, np.integer)) for index in indices):
            raise TypeError("active_inner_indices must contain integers.")
        if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
            raise ValueError("active_inner_indices must be unique and non-negative.")
        if len(indices) > mean.size:
            raise ValueError("active_inner_indices cannot outnumber conditional coordinates.")
        object.__setattr__(self, "mean", _frozen(mean))
        object.__setattr__(self, "covariance", _frozen(covariance))
        object.__setattr__(self, "active_inner_indices", tuple(int(index) for index in indices))

    def __eq__(self, other: object) -> bool:
        """Return value equality for posterior moments and active indices."""
        if not isinstance(other, GaussianConditional):
            return NotImplemented
        return (
            self.active_inner_indices == other.active_inner_indices
            and np.array_equal(self.mean, other.mean)
            and np.array_equal(self.covariance, other.covariance)
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, eq=False)
class GaussianProductSpaceTarget:
    """Normalized continuous target and exact partition oracle.

    Construct targets with :meth:`from_grid`. All coefficients are anomalies
    around ``observation_mean``. The inner active prior is induced by IID
    finest-grid anomalies, while the always-active outer block uses a separate
    covariance.

    Attributes:
        observations: Observation vector.
        observation_mean: Prior-forward observation mean.
        observation_covariance: Fixed positive-definite error covariance.
        inner_design: Summed candidate columns for every dyadic node.
        contrast_layout: Permanent root-and-contrast coordinate layout.
        inner_prior_variances: Primitive Gaussian variances for all inner
            coordinates.
        inactive_pseudo_prior_variances: Normalized pseudo-prior variances for
            inactive inner contrasts.
        outer_design: Always-active outer-region design matrix.
        outer_prior_covariance: Separate positive-definite outer prior
            covariance.
        partition_log_prior: Log prior mass for a valid partition. The callable
            should include its normalizing constant when absolute augmented
            densities are required.
    """

    observations: np.ndarray
    observation_mean: np.ndarray
    observation_covariance: np.ndarray
    inner_design: MultiscaleDesign
    contrast_layout: TreeContrastLayout
    inner_prior_variances: np.ndarray
    inactive_pseudo_prior_variances: np.ndarray
    outer_design: np.ndarray
    outer_prior_covariance: np.ndarray
    partition_log_prior: PartitionLogPrior

    def __post_init__(self) -> None:
        """Enforce target invariants for classmethod and direct construction."""
        observations = _vector(self.observations, name="observations")
        if observations.size == 0:
            raise ValueError("observations must not be empty.")
        mean = _vector(self.observation_mean, name="observation_mean")
        if mean.shape != observations.shape:
            raise ValueError("observation_mean must have the same shape as observations.")
        error_covariance = _positive_definite_matrix(
            self.observation_covariance,
            size=observations.size,
            name="observation_covariance",
        )
        if not isinstance(self.inner_design, MultiscaleDesign):
            raise TypeError("inner_design must be a MultiscaleDesign.")
        design_values = _matrix(self.inner_design.values, name="inner_design.values")
        if design_values.shape != (observations.size, len(self.inner_design.tree.nodes)):
            raise ValueError("inner_design values must match observations and tree nodes.")
        if not isinstance(self.contrast_layout, TreeContrastLayout):
            raise TypeError("contrast_layout must be a TreeContrastLayout.")
        if self.contrast_layout.tree != self.inner_design.tree:
            raise ValueError("contrast_layout and inner_design must use the same tree.")

        inner_variances = _positive_variance_vector(
            self.inner_prior_variances,
            size=self.contrast_layout.coordinate_count,
            name="inner_prior_variances",
        )
        pseudo_variances = _positive_variance_vector(
            self.inactive_pseudo_prior_variances,
            size=self.contrast_layout.coordinate_count,
            name="inactive_pseudo_prior_variances",
        )
        outer_design = _matrix(self.outer_design, name="outer_design")
        if outer_design.shape[0] != observations.size:
            raise ValueError("outer_design observation count must match observations.")
        outer_covariance = _positive_definite_matrix(
            self.outer_prior_covariance,
            size=outer_design.shape[1],
            name="outer_prior_covariance",
        )
        if not callable(self.partition_log_prior):
            raise TypeError("partition_log_prior must be callable.")

        object.__setattr__(self, "observations", _frozen(observations))
        object.__setattr__(self, "observation_mean", _frozen(mean))
        object.__setattr__(self, "observation_covariance", _frozen(error_covariance))
        object.__setattr__(
            self,
            "inner_design",
            MultiscaleDesign(values=_frozen(design_values), tree=self.inner_design.tree),
        )
        object.__setattr__(self, "inner_prior_variances", _frozen(inner_variances))
        object.__setattr__(self, "inactive_pseudo_prior_variances", _frozen(pseudo_variances))
        object.__setattr__(self, "outer_design", _frozen(outer_design))
        object.__setattr__(self, "outer_prior_covariance", _frozen(outer_covariance))

    @classmethod
    def from_grid(
        cls,
        observations: npt.ArrayLike,
        inner_grid_design: npt.ArrayLike,
        tree: DyadicTree,
        observation_covariance: npt.ArrayLike,
        *,
        observation_mean: npt.ArrayLike | None = None,
        inner_prior_scale: float = 1.0,
        inactive_pseudo_prior_scale: float = 1.0,
        outer_design: npt.ArrayLike | None = None,
        outer_prior_covariance: npt.ArrayLike | None = None,
        partition_log_prior: PartitionLogPrior | None = None,
    ) -> GaussianProductSpaceTarget:
        """Construct a validated Gaussian product-space target.

        Args:
            observations: Finite one-dimensional observation vector.
            inner_grid_design: Fine inner design with shape ``(observation,
                row, column)``. Columns are summed within active regions.
            tree: Canonical tree matching the inner design's spatial shape.
            observation_covariance: Positive-definite covariance of observation
                and model-data mismatch errors.
            observation_mean: Optional prior-forward mean. Defaults to zero.
            inner_prior_scale: Positive finest-grid anomaly standard deviation.
            inactive_pseudo_prior_scale: Positive multiplier applied to each
                primitive inner standard deviation in the inactive
                pseudo-prior.
            outer_design: Optional always-active outer design with shape
                ``(observation, outer_region)``.
            outer_prior_covariance: Positive-definite prior covariance for the
                outer coefficients. Required when ``outer_design`` has columns.
            partition_log_prior: Optional log prior mass. Defaults to equal
                relative weight for every valid partition.

        Returns:
            Immutable validated target with precomputed multiscale columns.

        Raises:
            TypeError: If ``partition_log_prior`` is not callable.
            ValueError: If an array has incompatible shape, contains a
                non-finite value, a required covariance is missing, or a scale
                is not positive.
        """
        y = _vector(observations, name="observations")
        if y.size == 0:
            raise ValueError("observations must not be empty.")
        source_design = MultiscaleDesign.from_grid(inner_grid_design, tree)
        design = MultiscaleDesign(values=_frozen(source_design.values), tree=tree)
        if design.values.shape[0] != y.size:
            raise ValueError("inner_grid_design observation count must match observations.")

        mean = (
            np.zeros_like(y)
            if observation_mean is None
            else _vector(observation_mean, name="observation_mean")
        )
        if mean.shape != y.shape:
            raise ValueError("observation_mean must have the same shape as observations.")
        error_covariance = _positive_definite_matrix(
            observation_covariance,
            size=y.size,
            name="observation_covariance",
        )

        layout = TreeContrastLayout.from_tree(tree)
        inner_variances = layout.prior_variances(inner_prior_scale)
        pseudo_scale = _positive_finite_scale(
            inactive_pseudo_prior_scale,
            name="inactive_pseudo_prior_scale",
        )
        with np.errstate(over="ignore", invalid="ignore"):
            pseudo_variances = inner_variances * np.multiply(pseudo_scale, pseudo_scale)
        if not np.all(np.isfinite(pseudo_variances)) or np.any(pseudo_variances <= 0.0):
            raise ValueError("inactive_pseudo_prior_scale produces non-finite variances.")

        outer_matrix: npt.NDArray[np.float64]
        if outer_design is None:
            outer_matrix = np.empty((y.size, 0), dtype=float)
        else:
            outer_matrix = _matrix(outer_design, name="outer_design")
            if outer_matrix.shape[0] != y.size:
                raise ValueError("outer_design observation count must match observations.")

        outer_count = outer_matrix.shape[1]
        outer_covariance: npt.NDArray[np.float64]
        if outer_count == 0:
            if outer_prior_covariance is not None and np.asarray(outer_prior_covariance).size != 0:
                raise ValueError("outer_prior_covariance must be empty when outer_design has no columns.")
            outer_covariance = np.empty((0, 0), dtype=float)
        else:
            if outer_prior_covariance is None:
                raise ValueError("outer_prior_covariance is required when outer_design has columns.")
            outer_covariance = _positive_definite_matrix(
                outer_prior_covariance,
                size=outer_count,
                name="outer_prior_covariance",
            )

        if partition_log_prior is None:
            prior: PartitionLogPrior = _equal_partition_log_weight
        elif callable(partition_log_prior):
            prior = partition_log_prior
        else:
            raise TypeError("partition_log_prior must be callable.")

        return cls(
            observations=y,
            observation_mean=mean,
            observation_covariance=error_covariance,
            inner_design=design,
            contrast_layout=layout,
            inner_prior_variances=inner_variances,
            inactive_pseudo_prior_variances=pseudo_variances,
            outer_design=outer_matrix,
            outer_prior_covariance=outer_covariance,
            partition_log_prior=prior,
        )

    @property
    def tree(self) -> DyadicTree:
        """Return the dyadic tree shared by the design and contrast layout."""
        return self.inner_design.tree

    def log_density(self, state: ProductSpaceState) -> float:
        """Evaluate the augmented product-space log density.

        Args:
            state: Partition, permanent inner coordinates, and fixed outer
                coefficients.

        Returns:
            Log likelihood plus active priors, inactive normalized
            pseudo-priors, the outer prior, and partition log prior.

        Raises:
            ValueError: If state dimensions do not match the target or the
                partition prior returns a non-finite value.
        """
        self._validate_state(state)
        active_indices = self.contrast_layout.active_coordinate_indices(state.partition)
        inactive_indices = self.contrast_layout.inactive_coordinate_indices(state.partition)
        prediction = self.observation_mean.copy()
        prediction += self.inner_design.gather(state.partition) @ self.contrast_layout.decode(
            state.partition,
            state.inner_coordinates,
        )
        prediction += self.outer_design @ state.outer_coefficients

        log_prior = _checked_partition_log_prior(self.partition_log_prior(state.partition))

        return float(
            _multivariate_normal_logpdf(
                self.observations,
                prediction,
                self.observation_covariance,
            )
            + _independent_normal_logpdf(
                state.inner_coordinates[list(active_indices)],
                self.inner_prior_variances[list(active_indices)],
            )
            + _independent_normal_logpdf(
                state.inner_coordinates[list(inactive_indices)],
                self.inactive_pseudo_prior_variances[list(inactive_indices)],
            )
            + _multivariate_normal_logpdf(
                state.outer_coefficients,
                np.zeros_like(state.outer_coefficients),
                self.outer_prior_covariance,
            )
            + log_prior
        )

    def active_design(self, partition: PartitionState) -> tuple[np.ndarray, tuple[int, ...]]:
        """Return the likelihood design for active contrast and outer values.

        Args:
            partition: Valid inner partition.

        Returns:
            A matrix whose leading columns correspond to returned permanent
            inner-coordinate indices and whose trailing columns correspond to
            the always-active outer coefficients, together with those inner
            indices.
        """
        partition.validate(self.tree)
        active_indices = self.contrast_layout.active_coordinate_indices(partition)
        decoder = self.contrast_layout.decoder(partition)[:, active_indices]
        inner = self.inner_design.gather(partition) @ decoder
        return np.column_stack((inner, self.outer_design)), active_indices

    def log_marginal_partition_density(self, partition: PartitionState) -> float:
        """Integrate continuous coefficients for one partition exactly.

        Inactive pseudo-priors integrate to one and therefore do not appear in
        this expression.

        Args:
            partition: Valid inner partition.

        Returns:
            Normalized Gaussian log marginal likelihood plus partition log
            prior.
        """
        design, active_indices = self.active_design(partition)
        prior_covariance = _block_diagonal(
            np.diag(self.inner_prior_variances[list(active_indices)]),
            self.outer_prior_covariance,
        )
        marginal_covariance = self.observation_covariance + design @ prior_covariance @ design.T
        log_prior = _checked_partition_log_prior(self.partition_log_prior(partition))
        return (
            _multivariate_normal_logpdf(
                self.observations,
                self.observation_mean,
                marginal_covariance,
            )
            + log_prior
        )

    def partition_probabilities(
        self,
        partitions: Iterable[PartitionState],
    ) -> dict[PartitionState, float]:
        """Normalize exact marginal probabilities over explicit partitions.

        Args:
            partitions: Non-empty sequence of distinct valid partitions.

        Returns:
            Mapping from each supplied partition to its normalized probability.

        Raises:
            ValueError: If no partition is supplied or a partition is repeated.
        """
        states = tuple(partitions)
        if not states:
            raise ValueError("partitions must not be empty.")
        if len(set(states)) != len(states):
            raise ValueError("partitions must be distinct.")
        log_probabilities = np.array(
            [self.log_marginal_partition_density(partition) for partition in states],
            dtype=float,
        )
        if np.all(np.isneginf(log_probabilities)):
            raise ValueError("At least one partition must have positive prior predictive mass.")
        maximum = float(log_probabilities.max())
        weights = np.exp(log_probabilities - maximum)
        weights /= weights.sum()
        return dict(zip(states, weights, strict=True))

    def conditional_posterior(self, partition: PartitionState) -> GaussianConditional:
        """Return the exact Gaussian law conditional on one partition.

        Args:
            partition: Valid inner partition.

        Returns:
            Posterior for active inner coordinates followed by outer
            coefficients. Inactive inner coordinates are excluded because
            their conditional law is their pseudo-prior.
        """
        design, active_indices = self.active_design(partition)
        prior_covariance = _block_diagonal(
            np.diag(self.inner_prior_variances[list(active_indices)]),
            self.outer_prior_covariance,
        )
        prior_precision = _positive_definite_solve(prior_covariance, np.eye(prior_covariance.shape[0]))
        solved_design = _positive_definite_solve(self.observation_covariance, design)
        precision = prior_precision + design.T @ solved_design
        covariance = _positive_definite_solve(precision, np.eye(precision.shape[0]))
        residual = self.observations - self.observation_mean
        mean = covariance @ design.T @ _positive_definite_solve(self.observation_covariance, residual)
        return GaussianConditional(
            mean=_frozen(mean),
            covariance=_frozen(covariance),
            active_inner_indices=active_indices,
        )

    def draw_conditional_state(
        self,
        partition: PartitionState,
        rng: np.random.Generator,
    ) -> ProductSpaceState:
        """Draw active coefficients and refresh inactive pseudo-prior values.

        Args:
            partition: Partition held fixed during the Gaussian update.
            rng: Caller-owned NumPy random generator.

        Returns:
            A complete product-space state. Active inner and outer values come
            from their exact conditional posterior; inactive inner contrasts
            are independent pseudo-prior draws.

        Raises:
            TypeError: If ``rng`` is not a NumPy random generator.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        conditional = self.conditional_posterior(partition)
        joint = conditional.mean + np.linalg.cholesky(conditional.covariance) @ rng.standard_normal(
            conditional.mean.size
        )
        inner: npt.NDArray[np.float64] = np.empty(
            self.contrast_layout.coordinate_count,
            dtype=float,
        )
        active_count = len(conditional.active_inner_indices)
        inner[list(conditional.active_inner_indices)] = joint[:active_count]
        inactive_indices = self.contrast_layout.inactive_coordinate_indices(partition)
        if inactive_indices:
            inner[list(inactive_indices)] = rng.normal(
                scale=np.sqrt(self.inactive_pseudo_prior_variances[list(inactive_indices)])
            )
        outer = joint[active_count:]
        return ProductSpaceState(
            partition=partition,
            inner_coordinates=inner,
            outer_coefficients=outer,
        )

    def _validate_state(self, state: ProductSpaceState) -> None:
        """Validate state geometry and permanent coordinate dimensions."""
        state.partition.validate(self.tree)
        if state.inner_coordinates.shape != (self.contrast_layout.coordinate_count,):
            raise ValueError("inner_coordinates must match the target's permanent contrast dimension.")
        if state.outer_coefficients.shape != (self.outer_design.shape[1],):
            raise ValueError("outer_coefficients must match the target's outer-region dimension.")


def _equal_partition_log_weight(partition: PartitionState) -> float:
    """Return equal relative weight for every partition."""
    del partition
    return 0.0


def _checked_partition_log_prior(value: float) -> float:
    """Return a scalar log prior, allowing negative infinity for zero mass."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("partition_log_prior must return a real scalar.") from error
    if math.isnan(result) or result == math.inf:
        raise ValueError("partition_log_prior must return a finite scalar or negative infinity.")
    return result


def _independent_normal_logpdf(values: np.ndarray, variances: np.ndarray) -> float:
    """Return a normalized zero-mean independent Gaussian log density."""
    if values.size == 0:
        return 0.0
    return float(
        -0.5
        * (
            values.size * math.log(2.0 * math.pi)
            + np.log(variances).sum()
            + np.square(values).dot(1.0 / variances)
        )
    )


def _multivariate_normal_logpdf(
    value: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Return a normalized multivariate Gaussian log density by Cholesky solve."""
    if value.size == 0:
        return 0.0
    residual = value - mean
    cholesky = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(cholesky, residual)
    return float(
        -0.5
        * (value.size * math.log(2.0 * math.pi) + 2.0 * np.log(np.diag(cholesky)).sum() + whitened @ whitened)
    )


def _positive_definite_solve(matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    """Solve a positive-definite system with a Cholesky factorization."""
    cholesky = np.linalg.cholesky(matrix)
    intermediate = np.linalg.solve(cholesky, right_hand_side)
    return np.linalg.solve(cholesky.T, intermediate)


def _block_diagonal(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Construct a two-block dense covariance without a SciPy dependency."""
    result = np.zeros(
        (first.shape[0] + second.shape[0], first.shape[1] + second.shape[1]),
        dtype=float,
    )
    result[: first.shape[0], : first.shape[1]] = first
    result[first.shape[0] :, first.shape[1] :] = second
    return result


def _vector(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Return a finite floating-point vector."""
    source = np.asarray(values)
    if np.iscomplexobj(source):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(source, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _matrix(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Return a finite floating-point matrix."""
    source = np.asarray(values)
    if np.iscomplexobj(source):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(source, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _positive_definite_matrix(values: npt.ArrayLike, *, size: int, name: str) -> np.ndarray:
    """Return a finite symmetric positive-definite square matrix."""
    matrix = _matrix(values, name=name)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}).")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite.") from error
    return matrix


def _positive_variance_vector(values: npt.ArrayLike, *, size: int, name: str) -> np.ndarray:
    """Return a finite positive variance vector with an exact size."""
    variances = _vector(values, name=name)
    if variances.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if np.any(variances <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")
    return variances


def _positive_finite_scale(value: float, *, name: str) -> float:
    """Normalize one positive finite scalar scale."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be positive and finite.") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _frozen(values: np.ndarray) -> np.ndarray:
    """Copy an array and mark its storage read-only."""
    source = np.asarray(values)
    if np.iscomplexobj(source):
        raise ValueError("Internal arrays must be real-valued.")
    result = np.asarray(source, dtype=float).copy()
    result.setflags(write=False)
    return result


__all__ = [
    "GaussianConditional",
    "GaussianProductSpaceTarget",
    "PartitionLogPrior",
]
