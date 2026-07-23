"""Immutable NumPy reference states for fixed-tree Gamma--Beta inversion.

This module defines the active-only continuous coordinates used by the
experimental fixed-direction dyadic baseline.  A state consists of one
canonical frontier, a positive root total, and allocation fractions only for
the internal nodes split above that frontier.  Physical mass is propagated
conservatively through those splits.  Within an unresolved frontier node, its
mass is distributed over native cells in fixed nominal proportions.

All probability terms are fully normalized.  Root totals use a Gamma
shape--rate density, allocation fractions use Beta densities, and structural
frontiers use ``p(F) = p_K(K) / N_K``.  The target is defined with respect to
root-total and active-fraction coordinates, so it deliberately contains no
leaf-mass Jacobian.  Arrays owned by public objects are copied and made
read-only, making the full builder a deterministic correctness oracle for
later incremental or compiled implementations.

The observation target can also include a fixed concentration offset and an
always-active :class:`~openghg_inversions.experimental.rjmcmc.core.FixedDesignBlock`.
Its positive coefficients use the same independent arithmetic-moment
lognormal prior contract as the Voronoi reference implementation.

The principal entry points are :class:`GammaBetaTreePrior`,
:class:`TreePartitionPrior`, :class:`GammaBetaTreeProblem`,
:func:`build_gamma_beta_tree_state`, and :func:`render_cell_mass`.  Their
validated constructors and builder own their inputs; direct construction of a
state bypasses the builder's consistency guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite, lgamma, log, log1p, pi
from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import FixedDesignBlock, lognormal_coefficient_log_prior_numpy
from .dyadic_tree import CanonicalDyadicTree, DyadicFrontier, partition_counts_by_k

FloatArray: TypeAlias = NDArray[np.float64]

_LOG_TWO_PI = log(2.0 * pi)


def _readonly_float_array(values: ArrayLike, *, name: str) -> FloatArray:
    """Copy finite values into a read-only ``float64`` array.

    Args:
        values: Candidate numerical array.
        name: Field name used in validation errors.

    Returns:
        Owned, finite, read-only array with the input shape.

    Raises:
        ValueError: If any value is non-finite or cannot be converted to
            ``float64``.
    """
    result = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result.setflags(write=False)
    return result


def _positive_finite(value: float, *, name: str) -> float:
    """Normalize one scalar on finite positive support.

    Args:
        value: Candidate real scalar.
        name: Field name used in validation errors.

    Returns:
        Positive finite built-in float.

    Raises:
        TypeError: If ``value`` is Boolean or not float-convertible.
        ValueError: If ``value`` is non-finite or non-positive.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return result


def _cell_vector(
    values: ArrayLike,
    *,
    tree: CanonicalDyadicTree,
    name: str,
) -> FloatArray:
    """Normalize one positive cell field into C-order.

    Args:
        values: Candidate vector or array with ``tree.shape``.
        tree: Tree defining the native-cell count and spatial shape.
        name: Field name used in validation errors.

    Returns:
        Positive read-only vector with one entry per native cell.

    Raises:
        ValueError: If values are non-finite, non-positive, or have an
            incompatible shape.
    """
    result = _readonly_float_array(values, name=name)
    cell_count = len(tree.leaf_ids)
    if result.shape == tree.shape:
        result = np.array(result.reshape(-1), dtype=np.float64, copy=True)
    elif result.shape != (cell_count,):
        raise ValueError(f"{name} must have shape {tree.shape} or ({cell_count},).")
    if np.any(result <= 0.0):
        raise ValueError(f"{name} must be strictly positive in every cell.")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class GammaBetaTreePrior:
    """Normalized root-total and split-fraction prior on one fixed tree.

    Construct instances with :meth:`constant_concentration` or
    :meth:`additive_cell_alpha`.  Beta parameters are stored in node-ID order;
    terminal nodes contain zero placeholders and cannot be queried as split
    nodes.

    Attributes:
        tree: Canonical fixed tree to which this prior applies.
        nominal_cell_mass: Positive C-order cell masses used for expected
            allocation fractions and reduced-model within-node proportions.
        root_shape: Shape of the normalized Gamma root density.
        root_rate: Rate, not scale, of the normalized Gamma root density.
        beta_shape_by_node: Read-only ``(number_of_nodes, 2)`` array containing
            ordered child Beta shapes for internal nodes.

    Raises:
        TypeError: If ``tree`` or a scalar parameter has an invalid type.
        ValueError: If cell masses, root parameters, or Beta shapes are
            malformed.
    """

    tree: CanonicalDyadicTree
    nominal_cell_mass: FloatArray
    root_shape: float
    root_rate: float
    beta_shape_by_node: FloatArray

    def __post_init__(self) -> None:
        """Validate the fixed topology, root density, and split parameters."""
        if not isinstance(self.tree, CanonicalDyadicTree):
            raise TypeError("tree must be a CanonicalDyadicTree.")
        nominal = _cell_vector(
            self.nominal_cell_mass,
            tree=self.tree,
            name="nominal_cell_mass",
        )
        root_shape = _positive_finite(self.root_shape, name="root_shape")
        root_rate = _positive_finite(self.root_rate, name="root_rate")
        beta = _readonly_float_array(self.beta_shape_by_node, name="beta_shape_by_node")
        expected_shape = (len(self.tree.nodes), 2)
        if beta.shape != expected_shape:
            raise ValueError(f"beta_shape_by_node must have shape {expected_shape}.")
        internal = np.asarray(self.tree.internal_node_ids, dtype=np.int64)
        leaves = np.asarray(self.tree.leaf_ids, dtype=np.int64)
        if np.any(beta[internal] <= 0.0):
            raise ValueError("Every internal node must have two positive Beta shapes.")
        if np.any(beta[leaves] != 0.0):
            raise ValueError("Terminal nodes must have zero Beta-shape placeholders.")

        object.__setattr__(self, "nominal_cell_mass", nominal)
        object.__setattr__(self, "root_shape", root_shape)
        object.__setattr__(self, "root_rate", root_rate)
        object.__setattr__(self, "beta_shape_by_node", beta)

    @classmethod
    def constant_concentration(
        cls,
        tree: CanonicalDyadicTree,
        nominal_cell_mass: ArrayLike,
        *,
        concentration: float | None = None,
        kappa: float | None = None,
        root_mean: float | None = None,
        root_variance: float | None = None,
        root_shape: float | None = None,
        root_rate: float | None = None,
    ) -> GammaBetaTreePrior:
        """Build nominal-fraction Beta priors with one concentration.

        The root Gamma distribution may be specified either by its positive
        mean and variance or directly by positive shape and rate.  Exactly one
        complete parameter pair is required.

        Args:
            tree: Canonical fixed tree.
            nominal_cell_mass: Positive native-cell mass field.
            concentration: Positive common Beta concentration.
            kappa: Alias for ``concentration``.
            root_mean: Optional positive Gamma mean.
            root_variance: Optional positive Gamma variance.
            root_shape: Optional positive Gamma shape.
            root_rate: Optional positive Gamma rate.

        Returns:
            Immutable normalized Gamma--Beta prior.

        Raises:
            TypeError: If ``tree`` or a scalar has an invalid type.
            ValueError: If concentration, cell mass, or root parameters are
                malformed or ambiguously specified.
        """
        if not isinstance(tree, CanonicalDyadicTree):
            raise TypeError("tree must be a CanonicalDyadicTree.")
        if concentration is not None and kappa is not None:
            raise ValueError("Specify only one of concentration and kappa.")
        selected_concentration = concentration if concentration is not None else kappa
        if selected_concentration is None:
            raise ValueError("A positive concentration must be supplied.")
        normalized_concentration = _positive_finite(
            selected_concentration,
            name="concentration",
        )
        shape, rate = _resolve_root_parameters(
            root_mean=root_mean,
            root_variance=root_variance,
            root_shape=root_shape,
            root_rate=root_rate,
        )
        nominal = _cell_vector(
            nominal_cell_mass,
            tree=tree,
            name="nominal_cell_mass",
        )
        beta = np.zeros((len(tree.nodes), 2), dtype=np.float64)
        for node_id in tree.internal_node_ids:
            first_id, second_id = tree.children(node_id)
            first_mass = float(nominal[list(tree.node(first_id).cell_indices)].sum())
            second_mass = float(nominal[list(tree.node(second_id).cell_indices)].sum())
            first_fraction = first_mass / (first_mass + second_mass)
            beta[node_id] = (
                normalized_concentration * first_fraction,
                normalized_concentration * (1.0 - first_fraction),
            )
        return cls(
            tree=tree,
            nominal_cell_mass=nominal,
            root_shape=shape,
            root_rate=rate,
            beta_shape_by_node=beta,
        )

    @classmethod
    def additive_cell_alpha(
        cls,
        tree: CanonicalDyadicTree,
        nominal_cell_mass: ArrayLike,
        cell_alpha: ArrayLike,
        *,
        root_rate: float | None = None,
        rate: float | None = None,
    ) -> GammaBetaTreePrior:
        """Build the order-consistent Gamma--Dirichlet tree prior.

        Each internal-node Beta shape is the sum of positive cell alphas in
        the corresponding child.  The root shape is the sum over all cells,
        which recovers independent equal-rate cell Gamma variables when the
        tree is fully resolved. For refinement-invariant prior-mean rendered
        fields, ``cell_alpha`` should be proportional to
        ``nominal_cell_mass``; otherwise the prior allocation mean and the
        unresolved within-node rendering weights intentionally differ.

        Args:
            tree: Canonical fixed tree.
            nominal_cell_mass: Positive native-cell mass field used by the
                reduced observation model.
            cell_alpha: Positive additive Gamma/Dirichlet base measure.
            root_rate: Common positive Gamma rate.
            rate: Alias for ``root_rate``.

        Returns:
            Immutable additive Gamma--Beta prior.

        Raises:
            TypeError: If ``tree`` or the rate has an invalid type.
            ValueError: If arrays or the rate are malformed.
        """
        if not isinstance(tree, CanonicalDyadicTree):
            raise TypeError("tree must be a CanonicalDyadicTree.")
        if root_rate is not None and rate is not None:
            raise ValueError("Specify only one of root_rate and rate.")
        selected_rate = root_rate if root_rate is not None else rate
        if selected_rate is None:
            raise ValueError("A positive root_rate must be supplied.")
        normalized_rate = _positive_finite(selected_rate, name="root_rate")
        nominal = _cell_vector(
            nominal_cell_mass,
            tree=tree,
            name="nominal_cell_mass",
        )
        alpha = _cell_vector(cell_alpha, tree=tree, name="cell_alpha")
        beta = np.zeros((len(tree.nodes), 2), dtype=np.float64)
        for node_id in tree.internal_node_ids:
            first_id, second_id = tree.children(node_id)
            beta[node_id] = (
                float(alpha[list(tree.node(first_id).cell_indices)].sum()),
                float(alpha[list(tree.node(second_id).cell_indices)].sum()),
            )
        return cls(
            tree=tree,
            nominal_cell_mass=nominal,
            root_shape=float(alpha.sum()),
            root_rate=normalized_rate,
            beta_shape_by_node=beta,
        )

    def beta_parameters(self, node_id: int) -> tuple[float, float]:
        """Return normalized Beta shapes for one internal node.

        Args:
            node_id: Stable internal-node identifier.

        Returns:
            Ordered ``(first_child_shape, second_child_shape)``.

        Raises:
            KeyError: If the ID is unknown.
            ValueError: If the node is terminal.
        """
        node = self.tree.node(node_id)
        if node.is_cell:
            raise ValueError(f"Terminal node {node_id!r} has no Beta allocation.")
        alpha, beta = self.beta_shape_by_node[node.node_id]
        return float(alpha), float(beta)

    def draw_fraction_parameters(self, node_id: int) -> tuple[float, float]:
        """Return the Beta draw parameters for one internal node.

        Args:
            node_id: Stable internal-node identifier.

        Returns:
            Ordered first- and second-child Beta shapes.

        Raises:
            KeyError: If the ID is unknown.
            ValueError: If the node is terminal.
        """
        return self.beta_parameters(node_id)

    @property
    def beta_shapes(self) -> FloatArray:
        """Return ordered child Beta shapes in node-ID order.

        Returns:
            Read-only array with shape ``(number_of_nodes, 2)``.
        """
        return self.beta_shape_by_node

    def log_root_density(self, root_total: float) -> float:
        """Return the normalized Gamma shape--rate log density.

        Args:
            root_total: Candidate positive root mass.

        Returns:
            Normalized log density, or negative infinity outside support.

        Raises:
            TypeError: If conversion to float raises a type error.
            ValueError: If conversion to float raises a value error.
        """
        if isinstance(root_total, bool):
            return -np.inf
        value = float(root_total)
        if not isfinite(value) or value <= 0.0:
            return -np.inf
        return float(
            self.root_shape * log(self.root_rate)
            - lgamma(self.root_shape)
            + (self.root_shape - 1.0) * log(value)
            - self.root_rate * value
        )

    def log_fraction_density(self, node_id: int, fraction: float) -> float:
        """Return one normalized Beta allocation log density.

        Args:
            node_id: Stable internal-node identifier.
            fraction: Candidate first-child fraction.

        Returns:
            Normalized log density, or negative infinity outside ``(0, 1)``.

        Raises:
            KeyError: If the ID is unknown.
            TypeError: If ``fraction`` is not float-convertible.
            ValueError: If the node is terminal or float conversion fails.
        """
        alpha, beta = self.beta_parameters(node_id)
        if isinstance(fraction, bool):
            return -np.inf
        value = float(fraction)
        if not isfinite(value) or not 0.0 < value < 1.0:
            return -np.inf
        return float(
            lgamma(alpha + beta)
            - lgamma(alpha)
            - lgamma(beta)
            + (alpha - 1.0) * log(value)
            + (beta - 1.0) * log1p(-value)
        )


def _resolve_root_parameters(
    *,
    root_mean: float | None,
    root_variance: float | None,
    root_shape: float | None,
    root_rate: float | None,
) -> tuple[float, float]:
    """Resolve exactly one Gamma moment or shape--rate parameter pair.

    Args:
        root_mean: Optional positive Gamma mean.
        root_variance: Optional positive Gamma variance.
        root_shape: Optional positive Gamma shape.
        root_rate: Optional positive Gamma rate.

    Returns:
        Positive ``(shape, rate)`` pair.

    Raises:
        TypeError: If a supplied scalar has an invalid type.
        ValueError: If neither or both parameterizations are selected, a pair
            is incomplete, or a supplied value is outside positive support.
    """
    moments_supplied = root_mean is not None or root_variance is not None
    direct_supplied = root_shape is not None or root_rate is not None
    if moments_supplied and direct_supplied:
        raise ValueError("Specify the root Gamma by moments or shape and rate, not both.")
    if moments_supplied:
        if root_mean is None or root_variance is None:
            raise ValueError("root_mean and root_variance must be supplied together.")
        mean = _positive_finite(root_mean, name="root_mean")
        variance = _positive_finite(root_variance, name="root_variance")
        return mean * mean / variance, mean / variance
    if direct_supplied:
        if root_shape is None or root_rate is None:
            raise ValueError("root_shape and root_rate must be supplied together.")
        return (
            _positive_finite(root_shape, name="root_shape"),
            _positive_finite(root_rate, name="root_rate"),
        )
    raise ValueError("Root Gamma moments or shape and rate must be supplied.")


@dataclass(frozen=True, slots=True, eq=False)
class TreePartitionPrior:
    """Normalized ``p(F) = p_K(K) / N_K`` prior for one fixed tree.

    Args:
        tree: Canonical fixed tree.
        marginal_probability_by_k: Marginal probabilities indexed directly by
            active-region count.  The vector must be normalized and assign
            mass only to supported counts.

    Raises:
        TypeError: If ``tree`` has the wrong type.
        ValueError: If the marginal vector is malformed or not normalized.
    """

    tree: CanonicalDyadicTree
    marginal_probability_by_k: FloatArray
    _partition_counts: tuple[int, ...] = field(init=False, repr=False)
    _log_probability_by_k: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and cache exact counts and per-frontier probabilities."""
        if not isinstance(self.tree, CanonicalDyadicTree):
            raise TypeError("tree must be a CanonicalDyadicTree.")
        probabilities = _readonly_float_array(
            self.marginal_probability_by_k,
            name="marginal_probability_by_k",
        )
        if probabilities.ndim != 1 or probabilities.size < 2:
            raise ValueError("marginal_probability_by_k must be a one-dimensional K-indexed vector.")
        maximum_k = probabilities.size - 1
        if maximum_k > len(self.tree.leaf_ids):
            raise ValueError("marginal_probability_by_k cannot extend beyond the tree's terminal-cell count.")
        counts = partition_counts_by_k(self.tree, max_k=maximum_k)
        if np.any(probabilities < 0.0):
            raise ValueError("marginal_probability_by_k must be non-negative.")
        for k, probability in enumerate(probabilities):
            if probability > 0.0 and counts[k] == 0:
                raise ValueError(f"Positive prior mass was assigned to unsupported K={k}.")
        if not isclose(float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("marginal_probability_by_k must sum to one.")
        log_probabilities = np.full(len(counts), -np.inf, dtype=np.float64)
        for k, probability in enumerate(probabilities):
            if probability > 0.0:
                log_probabilities[k] = log(float(probability)) - log(counts[k])
        log_probabilities.setflags(write=False)

        object.__setattr__(self, "marginal_probability_by_k", probabilities)
        object.__setattr__(self, "_partition_counts", counts)
        object.__setattr__(self, "_log_probability_by_k", log_probabilities)

    @classmethod
    def from_marginal_probabilities(
        cls,
        tree: CanonicalDyadicTree,
        probabilities_by_k: ArrayLike,
    ) -> TreePartitionPrior:
        """Normalize non-negative marginal masses over supported ``K``.

        Args:
            tree: Canonical fixed tree.
            probabilities_by_k: Vector indexed directly by ``K``.

        Returns:
            Normalized fixed-tree partition prior.

        Raises:
            TypeError: If ``tree`` has the wrong type.
            ValueError: If masses are malformed, empty, or assign mass to an
                unsupported region count.
        """
        if not isinstance(tree, CanonicalDyadicTree):
            raise TypeError("tree must be a CanonicalDyadicTree.")
        masses = _readonly_float_array(probabilities_by_k, name="probabilities_by_k")
        if masses.ndim != 1 or masses.size < 2:
            raise ValueError("probabilities_by_k must be a one-dimensional K-indexed vector.")
        maximum_k = masses.size - 1
        if maximum_k > len(tree.leaf_ids):
            raise ValueError("probabilities_by_k cannot extend beyond the tree's terminal-cell count.")
        if np.any(masses < 0.0):
            raise ValueError("probabilities_by_k must be non-negative.")
        total = float(masses.sum())
        if total <= 0.0:
            raise ValueError("At least one supported K must have positive mass.")
        return cls(tree=tree, marginal_probability_by_k=masses / total)

    @classmethod
    def uniform_k(
        cls,
        tree: CanonicalDyadicTree,
        *,
        minimum_k: int | None = None,
        maximum_k: int | None = None,
    ) -> TreePartitionPrior:
        """Assign equal marginal mass to each supported ``K`` in a range.

        Args:
            tree: Canonical fixed tree.
            minimum_k: Smallest included region count; defaults to one.
            maximum_k: Largest included count; defaults to the cell count.

        Returns:
            Normalized uniform-``K`` partition prior.

        Raises:
            TypeError: If a bound is not integer-like.
            ValueError: If bounds are invalid or include an unsupported count.
        """
        lower, upper = _k_bounds(
            tree,
            minimum_k=minimum_k,
            maximum_k=maximum_k,
        )
        masses = np.zeros(upper + 1, dtype=np.float64)
        masses[lower : upper + 1] = 1.0
        return cls.from_marginal_probabilities(tree, masses)

    @classmethod
    def geometric(
        cls,
        tree: CanonicalDyadicTree,
        *,
        ratio: float = 0.5,
        continuation_probability: float | None = None,
        minimum_k: int | None = None,
        maximum_k: int | None = None,
    ) -> TreePartitionPrior:
        """Use a truncated geometric marginal over extra active regions.

        Args:
            tree: Canonical fixed tree.
            ratio: Positive finite ratio between successive ``K`` masses.
            continuation_probability: Alias for ``ratio``.
            minimum_k: Smallest included region count; defaults to one.
            maximum_k: Largest included count; defaults to the cell count.

        Returns:
            Normalized truncated-geometric partition prior.

        Raises:
            TypeError: If ``tree`` or a scalar has the wrong type.
            ValueError: If aliases conflict or the ratio or bounds are
                invalid.
        """
        if continuation_probability is not None:
            if ratio != 0.5:
                raise ValueError("Specify only one of ratio and continuation_probability.")
            ratio = continuation_probability
        normalized_ratio = _positive_finite(ratio, name="ratio")
        if normalized_ratio >= 1.0:
            raise ValueError("ratio must lie strictly between zero and one.")
        lower, upper = _k_bounds(
            tree,
            minimum_k=minimum_k,
            maximum_k=maximum_k,
        )
        masses = np.zeros(upper + 1, dtype=np.float64)
        logs = np.arange(upper - lower + 1, dtype=np.float64) * log(normalized_ratio)
        logs -= float(np.max(logs))
        masses[lower : upper + 1] = np.exp(logs)
        return cls.from_marginal_probabilities(tree, masses)

    @classmethod
    def geometric_extra_regions(
        cls,
        tree: CanonicalDyadicTree,
        *,
        continuation_probability: float = 0.5,
        minimum_regions: int | None = None,
        maximum_regions: int | None = None,
    ) -> TreePartitionPrior:
        """Build a truncated geometric prior using region-named arguments.

        Args:
            tree: Canonical fixed tree.
            continuation_probability: Ratio between successive ``K`` masses,
                strictly between zero and one.
            minimum_regions: Smallest included region count.
            maximum_regions: Largest included region count.

        Returns:
            Normalized truncated-geometric partition prior.

        Raises:
            TypeError: If ``tree`` or a scalar has the wrong type.
            ValueError: If the continuation probability or bounds are
                invalid.
        """
        return cls.geometric(
            tree,
            continuation_probability=continuation_probability,
            minimum_k=minimum_regions,
            maximum_k=maximum_regions,
        )

    @property
    def p_k(self) -> FloatArray:
        """Return the normalized marginal probability vector.

        Returns:
            Read-only probabilities indexed directly by active-region count.
        """
        return self.marginal_probability_by_k

    @property
    def pK(self) -> FloatArray:
        """Return the normalized marginal probability vector.

        Returns:
            Read-only probabilities indexed directly by active-region count.
        """
        return self.p_k

    @property
    def partition_counts(self) -> tuple[int, ...]:
        """Return exact frontier counts indexed directly by ``K``.

        Returns:
            Arbitrary-precision counts with index zero fixed at zero.
        """
        return self._partition_counts

    @property
    def partition_counts_by_k(self) -> tuple[int, ...]:
        """Return exact frontier counts indexed directly by ``K``.

        Returns:
            Arbitrary-precision counts with index zero fixed at zero.
        """
        return self.partition_counts

    @property
    def log_probability_by_k(self) -> FloatArray:
        """Return per-frontier log probabilities indexed directly by ``K``.

        Returns:
            Read-only ``log(p_K(K) / N_K)`` values, with negative infinity for
            excluded counts.
        """
        return self._log_probability_by_k

    def log_probability(self, frontier: DyadicFrontier) -> float:
        """Return the normalized log probability of one exact frontier.

        Args:
            frontier: Candidate exact frontier on this prior's tree.

        Returns:
            ``log(p_K(K) / N_K)``; negative infinity if ``K`` has zero mass.

        Raises:
            TypeError: If ``frontier`` has the wrong type.
            ValueError: If it is not an exact frontier on this tree.
        """
        if not isinstance(frontier, DyadicFrontier):
            raise TypeError("frontier must be a DyadicFrontier.")
        frontier.validate(self.tree)
        k = len(frontier)
        if k >= self._log_probability_by_k.size:
            return -np.inf
        return float(self._log_probability_by_k[k])

    def log_density(self, frontier: DyadicFrontier) -> float:
        """Return the normalized frontier log probability.

        Args:
            frontier: Candidate exact frontier on this prior's tree.

        Returns:
            ``log(p_K(K) / N_K)``; negative infinity if ``K`` has zero mass.

        Raises:
            TypeError: If ``frontier`` has the wrong type.
            ValueError: If it is not an exact frontier on this tree.
        """
        return self.log_probability(frontier)


def _k_bounds(
    tree: CanonicalDyadicTree,
    *,
    minimum_k: int | None,
    maximum_k: int | None,
) -> tuple[int, int]:
    """Validate and return inclusive supported ``K`` bounds.

    Args:
        tree: Tree defining the largest possible frontier.
        minimum_k: Optional inclusive lower bound.
        maximum_k: Optional inclusive upper bound.

    Returns:
        Validated inclusive ``(lower, upper)`` bounds.

    Raises:
        TypeError: If a supplied bound is not integer-like.
        ValueError: If bounds are out of order or outside the tree support.
    """
    lower = 1 if minimum_k is None else _integer_k(minimum_k, name="minimum_k")
    terminal_count = len(tree.leaf_ids)
    upper = terminal_count if maximum_k is None else _integer_k(maximum_k, name="maximum_k")
    if lower < 1 or upper > terminal_count or lower > upper:
        raise ValueError(f"Require 1 <= minimum_k <= maximum_k <= {terminal_count}.")
    return lower, upper


def _integer_k(value: int, *, name: str) -> int:
    """Return one non-Boolean integer-like region-count bound."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


@dataclass(frozen=True, slots=True, eq=False)
class GammaBetaTreeProblem:
    """Immutable observations and precomputed fixed-tree reduced designs.

    Args:
        observations: Finite vector with shape ``(n_observations,)``.
        observation_sd: Positive independent Gaussian standard deviations.
        sensitivity: Finite response per unit physical mass in each native
            cell, with shape ``(n_observations, n_cells)``. Standard RHIME
            ``fp_x_flux`` columns are responses to unit scaling and must first
            be divided by the corresponding nominal cell mass.
        prior: Fixed-tree Gamma--Beta prior.
        partition_prior: Normalized structural prior on the same tree.
        likelihood_power: Non-negative finite multiplier for the normalized
            Gaussian log likelihood.  Zero gives a prior-only target.
        fixed_offset: Optional coefficient-independent prediction offset.
            ``None`` is normalized to a read-only zero vector.
        fixed_block: Optional always-active design block with independent
            arithmetic-moment lognormal coefficient priors.

    Raises:
        TypeError: If priors have the wrong type or refer to different tree
            instances.
        ValueError: If arrays, supports, shapes, or the likelihood power are
            malformed.

    Attributes:
        node_nominal_mass: Read-only nominal totals in stable node-ID order.
        node_design: Read-only array with shape
            ``(n_observations, n_tree_nodes)``. Each column is the response
            per unit total in that node, using nominal within-node cell
            proportions.
        fixed_offset: Read-only coefficient-independent observation-space
            contribution.
    """

    observations: FloatArray
    observation_sd: FloatArray
    sensitivity: FloatArray
    prior: GammaBetaTreePrior
    partition_prior: TreePartitionPrior
    likelihood_power: float = 1.0
    fixed_offset: FloatArray | None = None
    fixed_block: FixedDesignBlock | None = None
    node_nominal_mass: FloatArray = field(init=False)
    node_design: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        """Validate inputs and precompute all native-node design columns."""
        if not isinstance(self.prior, GammaBetaTreePrior):
            raise TypeError("prior must be a GammaBetaTreePrior.")
        if not isinstance(self.partition_prior, TreePartitionPrior):
            raise TypeError("partition_prior must be a TreePartitionPrior.")
        if self.partition_prior.tree is not self.prior.tree:
            raise TypeError("prior and partition_prior must refer to the same tree instance.")
        observations = _readonly_float_array(self.observations, name="observations")
        observation_sd = _readonly_float_array(self.observation_sd, name="observation_sd")
        sensitivity = _readonly_float_array(self.sensitivity, name="sensitivity")
        if observations.ndim != 1:
            raise ValueError("observations must be one-dimensional.")
        if observation_sd.shape != observations.shape:
            raise ValueError("observation_sd must have the same shape as observations.")
        if np.any(observation_sd <= 0.0):
            raise ValueError("observation_sd must be strictly positive.")
        expected_design_shape = (observations.size, len(self.prior.tree.leaf_ids))
        if sensitivity.shape != expected_design_shape:
            raise ValueError(f"sensitivity must have shape {expected_design_shape}.")
        if isinstance(self.likelihood_power, bool):
            raise TypeError("likelihood_power must be a real number.")
        likelihood_power = float(self.likelihood_power)
        if not isfinite(likelihood_power) or likelihood_power < 0.0:
            raise ValueError("likelihood_power must be finite and non-negative.")
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

        nominal = self.prior.nominal_cell_mass
        node_mass = np.empty(len(self.tree.nodes), dtype=np.float64)
        node_design = np.empty((observations.size, len(self.tree.nodes)), dtype=np.float64)
        for node in reversed(self.tree.nodes):
            if node.is_cell:
                cell_index = node.cell_indices[0]
                node_mass[node.node_id] = nominal[cell_index]
                node_design[:, node.node_id] = sensitivity[:, cell_index]
                continue
            first_id, second_id = self.tree.children(node.node_id)
            first_mass = node_mass[first_id]
            second_mass = node_mass[second_id]
            mass = first_mass + second_mass
            node_mass[node.node_id] = mass
            node_design[:, node.node_id] = (
                first_mass * node_design[:, first_id] + second_mass * node_design[:, second_id]
            ) / mass
        node_mass.setflags(write=False)
        node_design.setflags(write=False)

        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "observation_sd", observation_sd)
        object.__setattr__(self, "sensitivity", sensitivity)
        object.__setattr__(self, "likelihood_power", likelihood_power)
        object.__setattr__(self, "fixed_offset", fixed_offset)
        object.__setattr__(self, "node_nominal_mass", node_mass)
        object.__setattr__(self, "node_design", node_design)

    @property
    def tree(self) -> CanonicalDyadicTree:
        """Return the fixed canonical tree shared by both priors.

        Returns:
            Exact tree instance used by the problem.
        """
        return self.prior.tree

    @property
    def sensitivities(self) -> FloatArray:
        """Return the response-per-unit-cell-mass matrix.

        Returns:
            Read-only matrix with shape ``(n_observations, n_cells)``. This is
            not raw RHIME ``fp_x_flux`` unless nominal cell masses are all one.
        """
        return self.sensitivity

    @property
    def node_design_columns(self) -> FloatArray:
        """Return per-unit-total design columns in node-ID order.

        Returns:
            Read-only matrix with shape
            ``(n_observations, number_of_tree_nodes)``.
        """
        return self.node_design

    @property
    def nominal_mass_by_node(self) -> FloatArray:
        """Return additive nominal masses in node-ID order.

        Returns:
            Read-only vector with one entry per tree node.
        """
        return self.node_nominal_mass

    @property
    def n_fixed_coefficients(self) -> int:
        """Return the number of always-active fixed coefficients.

        Returns:
            Zero when no fixed block is configured; otherwise the fixed design
            column count.
        """
        return 0 if self.fixed_block is None else self.fixed_block.n_coefficients


@dataclass(frozen=True, slots=True, eq=False)
class GammaBetaTreeState:
    """Immutable active-frontier state and complete cached target components.

    Valid, self-consistent instances with owned read-only arrays are produced
    by :func:`build_gamma_beta_tree_state`.  Direct dataclass construction
    bypasses those consistency checks.

    Attributes:
        problem: Exact problem instance used to build this state.
        frontier: Canonical active-node frontier.
        root_total: Positive total physical mass.
        active_fractions: Read-only fractions aligned with
            ``frontier.active_split_nodes(problem.tree)``.
        active_node_masses: Read-only propagated masses aligned with
            ``frontier.node_ids``.
        fixed_coefficients: Read-only positive always-active coefficients, or
            an empty vector when no fixed block is configured.
        dynamic_prediction: Cached tree contribution in observation space.
        fixed_prediction: Cached fixed offset plus always-active block
            contribution.
        prediction: Cached total observation-space prediction.
        residual: Cached ``prediction - observations`` residual.
        log_gaussian_likelihood: Raw normalized Gaussian log density.
        log_likelihood: Likelihood-power-scaled target component.
        log_root_prior: Normalized root Gamma log density.
        log_fraction_prior: Sum of normalized active Beta log densities.
        log_partition_prior: Normalized frontier log probability.
        log_fixed_coefficient_prior: Sum of normalized independent lognormal
            densities for the always-active coefficients.
    """

    problem: GammaBetaTreeProblem
    frontier: DyadicFrontier
    root_total: float
    active_fractions: FloatArray
    active_node_masses: FloatArray
    fixed_coefficients: FloatArray
    dynamic_prediction: FloatArray
    fixed_prediction: FloatArray
    prediction: FloatArray
    residual: FloatArray
    log_gaussian_likelihood: float
    log_likelihood: float
    log_root_prior: float
    log_fraction_prior: float
    log_partition_prior: float
    log_fixed_coefficient_prior: float

    @property
    def k(self) -> int:
        """Return the number of active frontier nodes.

        Returns:
            Positive active-region count.
        """
        return len(self.frontier)

    @property
    def split_fractions(self) -> FloatArray:
        """Return active fractions in canonical split-node order.

        Returns:
            Read-only vector aligned with
            ``frontier.active_split_nodes(problem.tree)``.
        """
        return self.active_fractions

    @property
    def log_allocation_prior(self) -> float:
        """Return the normalized active-fraction prior component.

        Returns:
            Sum of normalized Beta log densities for active splits.
        """
        return self.log_fraction_prior

    @property
    def log_target(self) -> float:
        """Return the complete likelihood-plus-prior log target.

        Returns:
            Sum of tempered likelihood, root, active-fraction, and partition
            log-density components plus the fixed-coefficient prior in
            root-plus-fraction coordinates.
        """
        return float(
            self.log_likelihood
            + self.log_root_prior
            + self.log_fraction_prior
            + self.log_partition_prior
            + self.log_fixed_coefficient_prior
        )


def build_gamma_beta_tree_state(
    problem: GammaBetaTreeProblem,
    *,
    frontier: DyadicFrontier,
    root_total: float,
    active_fractions: ArrayLike,
    fixed_coefficients: ArrayLike | None = None,
) -> GammaBetaTreeState:
    """Fully rebuild one immutable Gamma--Beta state.

    Args:
        problem: Fixed-tree observation model and priors.
        frontier: Exact active frontier on ``problem.tree``.
        root_total: Positive finite total mass.
        active_fractions: Fractions in canonical active-split-node order.
        fixed_coefficients: Positive coefficients aligned with
            ``problem.fixed_block``. This argument is required when a fixed
            block is configured and optional only when there is no fixed
            block.

    Returns:
        Immutable state with propagated masses, prediction, residual, and all
        normalized target components.

    Raises:
        TypeError: If ``problem`` or ``frontier`` has the wrong type.
        ValueError: If the frontier, root total, fraction shape, or fraction
            support is invalid, fixed coefficients do not match the declared
            block, or derived mass/prediction caches violate conservation or
            finiteness.
    """
    if not isinstance(problem, GammaBetaTreeProblem):
        raise TypeError("problem must be a GammaBetaTreeProblem.")
    if not isinstance(frontier, DyadicFrontier):
        raise TypeError("frontier must be a DyadicFrontier.")
    frontier.validate(problem.tree)
    normalized_root = _positive_finite(root_total, name="root_total")
    fractions = _readonly_float_array(active_fractions, name="active_fractions")
    split_node_ids = frontier.active_split_nodes(problem.tree)
    expected_shape = (len(split_node_ids),)
    if fractions.shape != expected_shape:
        raise ValueError(f"active_fractions must have shape {expected_shape}.")
    if np.any((fractions <= 0.0) | (fractions >= 1.0)):
        raise ValueError("active_fractions must lie strictly between zero and one.")
    normalized_fixed_coefficients = _prepare_fixed_coefficients(
        problem,
        fixed_coefficients,
    )

    active_masses = _propagate_frontier_masses(
        problem.tree,
        frontier,
        normalized_root,
        split_node_ids,
        fractions,
    )
    active_ids = np.asarray(frontier.node_ids, dtype=np.int64)
    with np.errstate(over="ignore", invalid="ignore"):
        dynamic_prediction = problem.node_design[:, active_ids] @ active_masses
        fixed_prediction = np.array(problem.fixed_offset, dtype=np.float64, copy=True)
        if problem.fixed_block is not None:
            fixed_prediction += problem.fixed_block.design @ normalized_fixed_coefficients
        prediction = dynamic_prediction + fixed_prediction
        residual = prediction - problem.observations
    if (
        not np.all(np.isfinite(dynamic_prediction))
        or not np.all(np.isfinite(fixed_prediction))
        or not np.all(np.isfinite(prediction))
        or not np.all(np.isfinite(residual))
    ):
        raise ValueError("state coordinates and designs must imply a finite prediction.")
    dynamic_prediction = _readonly_float_array(
        dynamic_prediction,
        name="dynamic_prediction",
    )
    fixed_prediction = _readonly_float_array(fixed_prediction, name="fixed_prediction")
    prediction = _readonly_float_array(prediction, name="prediction")
    residual = _readonly_float_array(residual, name="residual")
    standardized = residual / problem.observation_sd
    log_gaussian_likelihood = float(
        -0.5 * np.dot(standardized, standardized)
        - np.log(problem.observation_sd).sum()
        - 0.5 * problem.observations.size * _LOG_TWO_PI
    )
    log_likelihood = (
        0.0 if problem.likelihood_power == 0.0 else problem.likelihood_power * log_gaussian_likelihood
    )
    log_root_prior = problem.prior.log_root_density(normalized_root)
    log_fraction_prior = float(
        sum(
            problem.prior.log_fraction_density(node_id, float(fraction))
            for node_id, fraction in zip(split_node_ids, fractions, strict=True)
        )
    )
    log_partition_prior = problem.partition_prior.log_probability(frontier)
    log_fixed_coefficient_prior = _log_fixed_coefficient_prior(
        problem,
        normalized_fixed_coefficients,
    )
    return GammaBetaTreeState(
        problem=problem,
        frontier=frontier,
        root_total=normalized_root,
        active_fractions=fractions,
        active_node_masses=active_masses,
        fixed_coefficients=normalized_fixed_coefficients,
        dynamic_prediction=dynamic_prediction,
        fixed_prediction=fixed_prediction,
        prediction=prediction,
        residual=residual,
        log_gaussian_likelihood=log_gaussian_likelihood,
        log_likelihood=log_likelihood,
        log_root_prior=log_root_prior,
        log_fraction_prior=log_fraction_prior,
        log_partition_prior=log_partition_prior,
        log_fixed_coefficient_prior=log_fixed_coefficient_prior,
    )


def _prepare_fixed_coefficients(
    problem: GammaBetaTreeProblem,
    fixed_coefficients: ArrayLike | None,
) -> FloatArray:
    """Validate and own the optional always-active coefficient vector."""
    if problem.fixed_block is None:
        if fixed_coefficients is None:
            result = np.empty(0, dtype=np.float64)
            result.setflags(write=False)
            return result
        result = _readonly_float_array(fixed_coefficients, name="fixed_coefficients")
        if result.shape != (0,):
            raise ValueError("fixed_coefficients require a configured fixed_block.")
        return result
    if fixed_coefficients is None:
        raise ValueError("fixed_coefficients are required when fixed_block is configured.")
    result = _readonly_float_array(fixed_coefficients, name="fixed_coefficients")
    expected_shape = (problem.fixed_block.n_coefficients,)
    if result.shape != expected_shape:
        raise ValueError(f"fixed_coefficients must have shape {expected_shape}.")
    if np.any(result <= 0.0):
        raise ValueError("fixed_coefficients must be strictly positive.")
    return result


def _log_fixed_coefficient_prior(
    problem: GammaBetaTreeProblem,
    fixed_coefficients: FloatArray,
) -> float:
    """Return the normalized independent fixed-coefficient prior component."""
    if problem.fixed_block is None:
        return 0.0
    return float(
        sum(
            lognormal_coefficient_log_prior_numpy(
                fixed_coefficients[position : position + 1],
                1,
                float(mean),
                float(standard_deviation),
            )
            for position, (mean, standard_deviation) in enumerate(
                zip(
                    problem.fixed_block.coefficient_prior_mean,
                    problem.fixed_block.coefficient_prior_sd,
                    strict=True,
                )
            )
        )
    )


def _propagate_frontier_masses(
    tree: CanonicalDyadicTree,
    frontier: DyadicFrontier,
    root_total: float,
    split_node_ids: tuple[int, ...],
    active_fractions: FloatArray,
) -> FloatArray:
    """Propagate root mass to active frontier nodes.

    Args:
        tree: Canonical fixed tree.
        frontier: Valid exact active frontier.
        root_total: Positive root mass.
        split_node_ids: Active split ancestors in canonical node-ID order.
        active_fractions: Fractions aligned with ``split_node_ids``.

    Returns:
        Read-only masses aligned with ``frontier.node_ids``.  Their sum equals
        ``root_total`` within floating-point tolerance.

    Raises:
        ValueError: If fractions omit a required ancestor, propagation misses
            a frontier node, or propagated masses do not conserve total mass.
    """
    active = frozenset(frontier.node_ids)
    fractions_by_node = {
        node_id: float(fraction) for node_id, fraction in zip(split_node_ids, active_fractions, strict=True)
    }
    mass_by_node: dict[int, float] = {}
    pending = [(tree.root_id, root_total)]
    while pending:
        node_id, mass = pending.pop()
        if node_id in active:
            mass_by_node[node_id] = mass
            continue
        fraction = fractions_by_node.get(node_id)
        if fraction is None:
            raise ValueError("active_fractions do not cover every split ancestor.")
        first_id, second_id = tree.children(node_id)
        pending.append((second_id, (1.0 - fraction) * mass))
        pending.append((first_id, fraction * mass))
    if set(mass_by_node) != active:
        raise ValueError("frontier mass propagation did not reach every active node.")
    result = np.array([mass_by_node[node_id] for node_id in frontier.node_ids], dtype=np.float64)
    if not isclose(float(result.sum()), root_total, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError("Gamma--Beta propagation failed to conserve root mass.")
    result.setflags(write=False)
    return result


def render_cell_mass(
    problem: GammaBetaTreeProblem,
    state: GammaBetaTreeState,
) -> FloatArray:
    """Render active node totals onto native cells in nominal proportions.

    Args:
        problem: Fixed-tree problem used to build ``state``.
        state: Active-frontier Gamma--Beta state.

    Returns:
        Read-only C-order cell-mass vector whose sum equals
        ``state.root_total``.

    Raises:
        TypeError: If either argument has the wrong type.
        ValueError: If ``state`` was built for a different problem instance
            or rendered masses do not conserve the root total.
    """
    if not isinstance(problem, GammaBetaTreeProblem):
        raise TypeError("problem must be a GammaBetaTreeProblem.")
    if not isinstance(state, GammaBetaTreeState):
        raise TypeError("state must be a GammaBetaTreeState.")
    if state.problem is not problem:
        raise ValueError("state must have been built for the supplied problem instance.")
    cell_mass = np.zeros(len(problem.tree.leaf_ids), dtype=np.float64)
    nominal = problem.prior.nominal_cell_mass
    for node_id, node_total in zip(
        state.frontier.node_ids,
        state.active_node_masses,
        strict=True,
    ):
        indices = list(problem.tree.node(node_id).cell_indices)
        cell_mass[indices] = float(node_total) * nominal[indices] / problem.node_nominal_mass[node_id]
    if not isclose(float(cell_mass.sum()), state.root_total, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError("Rendered cell masses do not conserve the root total.")
    cell_mass.setflags(write=False)
    return cell_mass


__all__ = [
    "GammaBetaTreePrior",
    "GammaBetaTreeProblem",
    "GammaBetaTreeState",
    "TreePartitionPrior",
    "build_gamma_beta_tree_state",
    "render_cell_mass",
]
