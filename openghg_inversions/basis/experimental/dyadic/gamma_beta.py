"""Positive Gamma--Beta priors on grouped masked partition trees.

This experimental module implements the positive split construction described
in ``docs/plans/dyadic_partition_inference.md``. Each hard group has one
mean-one root scaling. Refinable groups are recursively intersected with a
canonical rectangular dyadic tree; every effective binary split receives a
Beta-distributed allocation fraction. Fixed groups, such as the InTEM outer
regions, remain one root and one leaf.

The partition topology and the concentration policy are separate. A
``KappaStrategy`` receives immutable split metadata, so depth-based,
group-specific, similarity-based, or learned policies can be compared without
changing the conservation and rendering code. The first implementation,
``DepthKappaStrategy``, increases concentration with effective split depth.
This makes fine sibling allocations more tightly coupled through their shared
parent. It does not guarantee equal covariance for every geographically
adjacent pair because adjacency and tree ancestry are different relations.

The implementation is pure NumPy and provisional. It samples a prior on a
fixed forest whose topology is either depth-limited or pruned to an exact
weighted terminal-region budget; it does not yet perform partition inference
or construct a PyMC model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import heapq
import math
from numbers import Integral
from typing import Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from .tree import DyadicTree, NodeId

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class GammaBetaSplitContext:
    """Static metadata supplied to a concentration strategy.

    Attributes:
        node_id: Stable forest node whose allocation is being split.
        group_name: Name of the hard group containing the split.
        depth: Number of stochastic Gamma--Beta splits above this node. The
            group root has depth zero.
        geometric_depth: Depth of the corresponding rectangular tile. This can
            exceed ``depth`` when a hard mask creates unary geometric branches.
        parent_expected_mass: Additive expected mass in the parent.
        child_expected_masses: Expected masses in the ordered children.
        child_grid_cell_counts: Numbers of supported grid cells in the ordered
            children.
    """

    node_id: int
    group_name: str
    depth: int
    geometric_depth: int
    parent_expected_mass: float
    child_expected_masses: tuple[float, float]
    child_grid_cell_counts: tuple[int, int]


class KappaStrategy(Protocol):
    """Policy for choosing one positive Beta concentration per tree split."""

    def __call__(self, context: GammaBetaSplitContext) -> float:
        """Return the positive concentration for ``context``.

        Args:
            context: Immutable group, depth, mass, and support metadata for one
                effective binary split.

        Returns:
            Positive finite Beta concentration ``kappa``.
        """
        ...


@dataclass(frozen=True, slots=True)
class DepthKappaStrategy:
    """Increase Beta concentration geometrically with effective split depth.

    The concentration at depth ``d`` is
    ``base_kappa * depth_multiplier**d``, optionally capped by ``max_kappa``.
    With ``depth_multiplier > 1``, fine sibling allocations remain closer to
    their parent allocation than coarse sibling allocations.

    Attributes:
        base_kappa: Concentration used for a root split.
        depth_multiplier: Multiplicative increase for every effective split
            level. Values above one tighten finer splits.
        max_kappa: Optional finite upper bound.
    """

    base_kappa: float = 2.0
    depth_multiplier: float = 2.0
    max_kappa: float | None = None

    def __post_init__(self) -> None:
        """Validate the depth policy parameters."""
        if not math.isfinite(self.base_kappa) or self.base_kappa <= 0.0:
            raise ValueError("base_kappa must be finite and positive.")
        if not math.isfinite(self.depth_multiplier) or self.depth_multiplier <= 0.0:
            raise ValueError("depth_multiplier must be finite and positive.")
        if self.max_kappa is not None:
            if not math.isfinite(self.max_kappa) or self.max_kappa <= 0.0:
                raise ValueError("max_kappa must be finite and positive when supplied.")
            if self.max_kappa < self.base_kappa:
                raise ValueError("max_kappa cannot be smaller than base_kappa.")

    def __call__(self, context: GammaBetaSplitContext) -> float:
        """Return concentration determined by effective split depth."""
        if self.max_kappa is not None and self.depth_multiplier > 1.0:
            log_growth = context.depth * math.log(self.depth_multiplier)
            if log_growth >= math.log(self.max_kappa / self.base_kappa):
                return float(self.max_kappa)
        try:
            kappa = self.base_kappa * self.depth_multiplier**context.depth
        except OverflowError as error:
            raise ValueError(
                "Depth-based kappa overflowed; supply max_kappa or smaller parameters."
            ) from error
        if not math.isfinite(kappa):
            raise ValueError("Depth-based kappa overflowed; supply max_kappa or smaller parameters.")
        return float(kappa)


@dataclass(frozen=True, slots=True)
class MomentSplitConstraint:
    """Reject refinements that imply unstable Beta shapes or child variances.

    The constraint is evaluated while a weighted terminal-region budget is
    selected. Rejected candidate splits remain terminal regions and their
    descendants are not considered. This makes ``target_regions`` an upper
    bound when ``allow_fewer_regions`` is true.

    Attributes:
        min_beta_shape: Optional lower bound on both ``kappa * p`` and
            ``kappa * (1 - p)``. This prevents extremely one-sided Beta
            allocations.
        max_child_variance: Optional upper bound on the exact scaling variance
            of either child after the proposed split.
        allow_fewer_regions: If true, return the admissible topology when the
            requested terminal-region budget cannot be reached. If false,
            raise ``ValueError`` instead.
    """

    min_beta_shape: float | None = 0.5
    max_child_variance: float | None = 9.0
    allow_fewer_regions: bool = True

    def __post_init__(self) -> None:
        """Validate optional moment thresholds and the fallback policy."""
        if self.min_beta_shape is not None:
            if not math.isfinite(self.min_beta_shape) or self.min_beta_shape <= 0.0:
                raise ValueError("min_beta_shape must be finite and positive when supplied.")
        if self.max_child_variance is not None:
            if not math.isfinite(self.max_child_variance) or self.max_child_variance <= 0.0:
                raise ValueError("max_child_variance must be finite and positive when supplied.")
        if not isinstance(self.allow_fewer_regions, bool):
            raise TypeError("allow_fewer_regions must be Boolean.")

    def accepts(
        self,
        context: GammaBetaSplitContext,
        *,
        parent_variance: float,
        kappa: float,
    ) -> bool:
        """Return whether one proposed split satisfies the configured limits.

        Args:
            context: Static expected-mass and topology metadata for the split.
            parent_variance: Exact variance of the mean-one parent scaling.
            kappa: Positive Beta concentration supplied by the active policy.

        Returns:
            True when both Beta shapes and both exact child variances satisfy
            every configured threshold.
        """
        first_mass, second_mass = context.child_expected_masses
        p = first_mass / (first_mass + second_mass)
        if self.min_beta_shape is not None:
            if min(kappa * p, kappa * (1.0 - p)) < self.min_beta_shape:
                return False
        if self.max_child_variance is not None:
            moments = gamma_beta_child_moments(
                parent_variance=parent_variance,
                first_expected_fraction=p,
                kappa=kappa,
            )
            if max(moments.first_variance, moments.second_variance) > self.max_child_variance:
                return False
        return True


@dataclass(frozen=True, slots=True, eq=False)
class GammaBetaGroupSpec:
    """Declare one hard-mask group and its root/refinement prior settings.

    Attributes:
        name: Stable unique group name.
        mask: Two-dimensional Boolean support on the full output grid.
        root_variance: Variance of the mean-one root scaling. Zero fixes the
            root scaling at one; a positive value uses a Gamma distribution.
        max_depth: Maximum number of effective binary splits on any group path.
            Zero leaves the group fixed as one region.
        target_regions: Optional terminal-region budget for the group.
            Candidate splits are considered through ``max_depth`` and selected
            by descending partition weight. The target is exact by default and
            becomes an upper bound when a supplied ``MomentSplitConstraint``
            allows fewer regions. It cannot be smaller than the number of
            disconnected components or larger than the available candidates.
    """

    name: str
    mask: npt.NDArray[np.bool_]
    root_variance: float = 1.0
    max_depth: int = 0
    target_regions: int | None = None

    def __post_init__(self) -> None:
        """Validate and freeze the group support and prior settings."""
        if not self.name:
            raise ValueError("Gamma-Beta group names must be non-empty.")
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("Gamma-Beta group masks must be two-dimensional.")
        if not mask.any():
            raise ValueError(f"Gamma-Beta group {self.name!r} has empty support.")
        if not math.isfinite(self.root_variance) or self.root_variance < 0.0:
            raise ValueError("root_variance must be finite and non-negative.")
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, Integral):
            raise TypeError("max_depth must be an integer.")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if self.target_regions is not None:
            if isinstance(self.target_regions, bool) or not isinstance(self.target_regions, Integral):
                raise TypeError("target_regions must be an integer when supplied.")
            if self.target_regions < 1:
                raise ValueError("target_regions must be positive when supplied.")

        mask = mask.copy()
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "max_depth", int(self.max_depth))
        if self.target_regions is not None:
            object.__setattr__(self, "target_regions", int(self.target_regions))


@dataclass(frozen=True, slots=True, eq=False)
class GammaBetaNode:
    """One effective masked node in a grouped Gamma--Beta forest.

    Attributes:
        node_id: Stable global node identifier in preorder.
        group_index: Index into ``GammaBetaForest.groups``.
        parent_id: Parent node identifier, or ``None`` for a group root.
        child_ids: Ordered child pair, or an empty tuple for a leaf.
        depth: Number of effective binary splits above this node.
        geometric_depth: Depth of the retained canonical rectangular tile.
        expected_mass: Additive expected mass inside this masked node.
        partition_weight: Additive topology-selection weight inside this node.
        flat_indices: Read-only full-grid flat indices covered by this node.
    """

    node_id: int
    group_index: int
    parent_id: int | None
    child_ids: tuple[int, ...]
    depth: int
    geometric_depth: int
    expected_mass: float
    partition_weight: float
    flat_indices: npt.NDArray[np.int64]


@dataclass(slots=True)
class _NodeRecord:
    """Mutable node record used only while constructing a forest."""

    group_index: int
    parent_id: int | None
    child_ids: list[int]
    depth: int
    geometric_depth: int
    expected_mass: float
    partition_weight: float
    flat_indices: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class GammaBetaChildMoments:
    """Analytic scaling-factor moments for one Gamma--Beta split.

    Attributes:
        first_variance: Variance of the first mean-one child scaling.
        second_variance: Variance of the second mean-one child scaling.
        covariance: Covariance between the two child scalings.
    """

    first_variance: float
    second_variance: float
    covariance: float


def gamma_beta_child_moments(
    *,
    parent_variance: float,
    first_expected_fraction: float,
    kappa: float,
) -> GammaBetaChildMoments:
    """Return analytic child moments for a mean-one Gamma--Beta split.

    Args:
        parent_variance: Variance of the mean-one parent scaling.
        first_expected_fraction: Expected mass fraction ``p`` assigned to the
            first child. Must lie strictly between zero and one.
        kappa: Positive finite Beta concentration.

    Returns:
        Variances of the two child scalings and their covariance.

    Raises:
        ValueError: If any input lies outside its mathematical domain.
    """
    if not math.isfinite(parent_variance) or parent_variance < 0.0:
        raise ValueError("parent_variance must be finite and non-negative.")
    if not math.isfinite(first_expected_fraction) or not 0.0 < first_expected_fraction < 1.0:
        raise ValueError("first_expected_fraction must be strictly between zero and one.")
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be finite and positive.")

    p = first_expected_fraction
    common = (1.0 + parent_variance) / (kappa + 1.0)
    return GammaBetaChildMoments(
        first_variance=parent_variance + common * (1.0 - p) / p,
        second_variance=parent_variance + common * p / (1.0 - p),
        covariance=(kappa * parent_variance - 1.0) / (kappa + 1.0),
    )


@dataclass(frozen=True, slots=True, eq=False)
class GammaBetaForest:
    """Fixed grouped topology and expected masses for Gamma--Beta sampling.

    Construct instances with :meth:`from_groups`. Nodes are stored in global
    preorder, so every parent precedes its children and sampling is one forward
    pass.

    Attributes:
        shape: Full two-dimensional grid shape.
        expected_mass: Read-only finite non-negative additive mass field.
        partition_weight: Read-only finite non-negative topology-selection
            weight field. This may differ from ``expected_mass``.
        groups: Ordered hard-group specifications.
        nodes: Effective masked nodes in global preorder.
        root_ids: Root node for every connected component. Components of one
            semantic group share a Gamma root draw during sampling.
        leaf_ids: Terminal node IDs in stable node order.
    """

    shape: tuple[int, int]
    expected_mass: FloatArray
    partition_weight: FloatArray
    groups: tuple[GammaBetaGroupSpec, ...]
    nodes: tuple[GammaBetaNode, ...]
    root_ids: tuple[int, ...]
    leaf_ids: tuple[int, ...]

    @classmethod
    def from_groups(
        cls,
        expected_mass: npt.ArrayLike,
        groups: Sequence[GammaBetaGroupSpec],
        *,
        partition_weight: npt.ArrayLike | None = None,
        kappa_strategy: KappaStrategy | None = None,
        split_constraint: MomentSplitConstraint | None = None,
        require_full_coverage: bool = False,
    ) -> GammaBetaForest:
        """Build a masked binary forest from hard group supports.

        Refinable groups use an independent canonical tree over their minimal
        bounding rectangle. Unary geometric branches caused by a mask are
        skipped and do not consume effective split depth. A proposed split is
        retained only when both supported children have positive expected mass;
        this avoids invalid zero-shape Beta distributions.

        Args:
            expected_mass: Finite non-negative additive mass on the full grid.
                This is normally absolute prior flux multiplied by grid area.
            groups: Ordered, non-overlapping hard group specifications.
            partition_weight: Optional finite non-negative field used to rank
                candidate splits for groups with ``target_regions``. Defaults
                to ``expected_mass``. It changes topology only; Gamma--Beta
                split means and conservation continue to use ``expected_mass``.
            kappa_strategy: Concentration policy used while checking
                ``split_constraint``. Required when a constraint is supplied.
            split_constraint: Optional moment admissibility rule applied while
                selecting a weighted terminal-region budget.
            require_full_coverage: If true, require every full-grid location to
                belong to exactly one group.

        Returns:
            Immutable grouped forest with cached expected masses and supports.

        Raises:
            ValueError: If masses, masks, names, coverage, or group totals are
                invalid.
        """
        mass = np.asarray(expected_mass, dtype=np.float64)
        if mass.ndim != 2:
            raise ValueError("expected_mass must be two-dimensional.")
        if not np.isfinite(mass).all() or (mass < 0.0).any():
            raise ValueError("expected_mass must be finite and non-negative.")
        if partition_weight is None:
            topology_weight = mass.copy()
        else:
            topology_weight = np.asarray(partition_weight, dtype=np.float64)
            if topology_weight.shape != mass.shape:
                raise ValueError("partition_weight must match expected_mass.")
            if not np.isfinite(topology_weight).all() or (topology_weight < 0.0).any():
                raise ValueError("partition_weight must be finite and non-negative.")
        if split_constraint is not None and kappa_strategy is None:
            raise ValueError("kappa_strategy is required when split_constraint is supplied.")

        group_tuple = tuple(groups)
        if not group_tuple:
            raise ValueError("At least one Gamma-Beta group is required.")
        if len({group.name for group in group_tuple}) != len(group_tuple):
            raise ValueError("Gamma-Beta group names must be unique.")

        coverage = np.zeros(mass.shape, dtype=np.int16)
        for group in group_tuple:
            if group.mask.shape != mass.shape:
                raise ValueError("Every Gamma-Beta group mask must match expected_mass.")
            coverage += group.mask
            if _finite_additive_sum(mass[group.mask], context=f"group {group.name!r}") <= 0.0:
                raise ValueError(f"Gamma-Beta group {group.name!r} must have positive expected mass.")
        if (coverage > 1).any():
            raise ValueError("Gamma-Beta group masks must not overlap.")
        if require_full_coverage and not np.all(coverage == 1):
            raise ValueError("Gamma-Beta groups must cover the full grid exactly once.")

        mass = mass.copy()
        mass.setflags(write=False)
        topology_weight = topology_weight.copy()
        topology_weight.setflags(write=False)
        records: list[_NodeRecord] = []
        root_ids: list[int] = []

        for group_index, group in enumerate(group_tuple):
            root_ids.extend(
                _append_group_trees(
                    records,
                    mass,
                    topology_weight,
                    group,
                    group_index,
                    kappa_strategy=kappa_strategy,
                    split_constraint=split_constraint,
                )
            )

        nodes: list[GammaBetaNode] = []
        for node_id, record in enumerate(records):
            flat_indices = record.flat_indices.copy()
            flat_indices.setflags(write=False)
            child_ids: tuple[int, ...]
            if record.child_ids:
                child_ids = (record.child_ids[0], record.child_ids[1])
            else:
                child_ids = ()
            nodes.append(
                GammaBetaNode(
                    node_id=node_id,
                    group_index=record.group_index,
                    parent_id=record.parent_id,
                    child_ids=child_ids,
                    depth=record.depth,
                    geometric_depth=record.geometric_depth,
                    expected_mass=record.expected_mass,
                    partition_weight=record.partition_weight,
                    flat_indices=flat_indices,
                )
            )

        leaf_ids = tuple(node.node_id for node in nodes if not node.child_ids)
        shape = (int(mass.shape[0]), int(mass.shape[1]))
        return cls(
            shape=shape,
            expected_mass=mass,
            partition_weight=topology_weight,
            groups=group_tuple,
            nodes=tuple(nodes),
            root_ids=tuple(root_ids),
            leaf_ids=leaf_ids,
        )

    def split_context(self, node_id: int) -> GammaBetaSplitContext:
        """Return strategy metadata for one internal node.

        Args:
            node_id: Internal forest node identifier.

        Returns:
            Immutable concentration-strategy context.

        Raises:
            KeyError: If ``node_id`` is not in the forest.
            ValueError: If ``node_id`` is a leaf.
        """
        try:
            node = self.nodes[node_id]
        except (IndexError, TypeError) as error:
            raise KeyError(f"Unknown Gamma-Beta node ID {node_id!r}.") from error
        if node.node_id != node_id:
            raise KeyError(f"Unknown Gamma-Beta node ID {node_id!r}.")
        if not node.child_ids:
            raise ValueError(f"Gamma-Beta leaf node {node_id!r} has no split context.")

        first, second = (self.nodes[child_id] for child_id in node.child_ids)
        return GammaBetaSplitContext(
            node_id=node.node_id,
            group_name=self.groups[node.group_index].name,
            depth=node.depth,
            geometric_depth=node.geometric_depth,
            parent_expected_mass=node.expected_mass,
            child_expected_masses=(first.expected_mass, second.expected_mass),
            child_grid_cell_counts=(first.flat_indices.size, second.flat_indices.size),
        )

    def sample(
        self,
        draws: int,
        *,
        kappa_strategy: KappaStrategy,
        rng: np.random.Generator | int | None = None,
    ) -> GammaBetaSamples:
        """Draw root Gamma scalings and top-down Beta split fractions.

        Args:
            draws: Positive number of independent prior draws.
            kappa_strategy: Concentration policy evaluated once per internal
                forest node.
            rng: NumPy generator or seed.

        Returns:
            Node scalings, split fractions, and cached concentrations.

        Raises:
            TypeError: If ``draws`` is not an integer.
            ValueError: If ``draws`` or a strategy result is invalid.
        """
        if isinstance(draws, bool) or not isinstance(draws, Integral):
            raise TypeError("draws must be an integer.")
        if draws < 1:
            raise ValueError("draws must be positive.")
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

        node_scalings = np.full((int(draws), len(self.nodes)), np.nan, dtype=np.float64)
        split_fractions = np.full_like(node_scalings, np.nan)
        kappa_by_node = np.full(len(self.nodes), np.nan, dtype=np.float64)

        roots_by_group = {
            group_index: tuple(
                node_id for node_id in self.root_ids if self.nodes[node_id].group_index == group_index
            )
            for group_index in range(len(self.groups))
        }
        for group_index, group in enumerate(self.groups):
            if group.root_variance == 0.0:
                root_scaling = np.ones(int(draws), dtype=np.float64)
            else:
                root_shape = 1.0 / group.root_variance
                root_scale = group.root_variance
                root_scaling = generator.gamma(
                    shape=root_shape,
                    scale=root_scale,
                    size=int(draws),
                )
            for root_id in roots_by_group[group_index]:
                node_scalings[:, root_id] = root_scaling

        for node in self.nodes:
            if not node.child_ids:
                continue
            context = self.split_context(node.node_id)
            kappa = float(kappa_strategy(context))
            if not math.isfinite(kappa) or kappa <= 0.0:
                raise ValueError(
                    f"Kappa strategy returned invalid value {kappa!r} "
                    f"for group {context.group_name!r}, node {node.node_id}."
                )

            first_id, second_id = node.child_ids
            first_mass, second_mass = context.child_expected_masses
            p = first_mass / (first_mass + second_mass)
            first_shape = kappa * p
            second_shape = kappa * (1.0 - p)
            if (
                not 0.0 < p < 1.0
                or not math.isfinite(first_shape)
                or not math.isfinite(second_shape)
                or first_shape <= 0.0
                or second_shape <= 0.0
            ):
                raise ValueError(
                    f"Expected masses for group {context.group_name!r}, node {node.node_id} "
                    "cannot be represented as positive finite Beta shapes."
                )
            rho = generator.beta(first_shape, second_shape, size=int(draws))
            parent_scaling = node_scalings[:, node.node_id]
            split_fractions[:, node.node_id] = rho
            kappa_by_node[node.node_id] = kappa
            node_scalings[:, first_id] = parent_scaling * rho / p
            node_scalings[:, second_id] = parent_scaling * (1.0 - rho) / (1.0 - p)

        return GammaBetaSamples(
            forest=self,
            node_scalings=node_scalings,
            split_fractions=split_fractions,
            kappa_by_node=kappa_by_node,
        )

    def leaf_labels(self, *, fill_value: int = 0) -> IntArray:
        """Render one stable positive label per terminal forest node.

        Args:
            fill_value: Integer label used outside all declared groups.

        Returns:
            Full-grid integer labels in stable leaf-node order.
        """
        labels: IntArray = np.full(self.shape, fill_value, dtype=np.int64)
        flattened = labels.reshape(-1)
        for label, node_id in enumerate(self.leaf_ids, start=1):
            flattened[self.nodes[node_id].flat_indices] = label
        return labels


@dataclass(frozen=True, slots=True, eq=False)
class GammaBetaSamples:
    """Prior draws attached to one fixed grouped Gamma--Beta forest.

    Attributes:
        forest: Fixed topology and expected masses used for every draw.
        node_scalings: ``(draw, node)`` mean-one scaling factors. Every entry
            is finite after successful sampling.
        split_fractions: ``(draw, node)`` Beta fractions. Leaves contain NaN
            because they have no split coordinate.
        kappa_by_node: One compiled concentration per internal node. Leaves
            contain NaN.
    """

    forest: GammaBetaForest
    node_scalings: FloatArray
    split_fractions: FloatArray
    kappa_by_node: FloatArray

    @property
    def draws(self) -> int:
        """Return the number of independent prior draws."""
        return int(self.node_scalings.shape[0])

    def to_grid(self, draw: int, *, fill_value: float = np.nan) -> FloatArray:
        """Render one draw as a native-grid scaling field.

        Args:
            draw: Zero-based draw index.
            fill_value: Value used outside all declared groups.

        Returns:
            Full-grid scaling field with each leaf scaling broadcast over its
            hard support.

        Raises:
            IndexError: If ``draw`` is outside the available range.
        """
        if draw < 0 or draw >= self.draws:
            raise IndexError(f"Draw index {draw} is outside [0, {self.draws}).")
        values: FloatArray = np.full(self.forest.shape, fill_value, dtype=np.float64)
        flattened = values.reshape(-1)
        for node_id in self.forest.leaf_ids:
            node = self.forest.nodes[node_id]
            flattened[node.flat_indices] = self.node_scalings[draw, node_id]
        return values

    def maximum_conservation_error(self) -> float:
        """Return the largest absolute parent/child expected-flux mismatch."""
        maximum = 0.0
        for node in self.forest.nodes:
            if not node.child_ids:
                continue
            first_id, second_id = node.child_ids
            first = self.forest.nodes[first_id]
            second = self.forest.nodes[second_id]
            parent_flux = node.expected_mass * self.node_scalings[:, node.node_id]
            child_flux = (
                first.expected_mass * self.node_scalings[:, first_id]
                + second.expected_mass * self.node_scalings[:, second_id]
            )
            maximum = max(maximum, float(np.max(np.abs(parent_flux - child_flux))))
        return maximum

    def analytic_leaf_covariance(
        self,
        *,
        root_variances: Mapping[str, float] | None = None,
    ) -> FloatArray:
        """Return the exact prior covariance of terminal-region scalings.

        The row and column order is ``forest.leaf_ids``. This is the induced
        covariance of the terminal-region state vector, not the covariance of
        the independent Gamma roots and Beta fractions used to parameterize it.
        The calculation uses the compiled concentrations in ``kappa_by_node``
        and is therefore independent of the finite Monte Carlo sample count.

        Args:
            root_variances: Optional variance overrides keyed by group name.
                This is useful for exact aggregate calibration on a fixed
                topology and fixed compiled split concentrations.

        Returns:
            Symmetric ``(leaf, leaf)`` covariance matrix.

        Raises:
            ValueError: If an override is unknown, negative, or non-finite.
        """
        variance_by_group = {group.name: group.root_variance for group in self.forest.groups}
        if root_variances is not None:
            unknown = set(root_variances) - set(variance_by_group)
            if unknown:
                raise ValueError(f"Unknown Gamma-Beta groups in root_variances: {sorted(unknown)!r}.")
            for group_name, variance in root_variances.items():
                value = float(variance)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError("Root variance overrides must be finite and non-negative.")
                variance_by_group[group_name] = value

        leaf_count = len(self.forest.leaf_ids)
        log_second_moment: FloatArray = np.zeros((leaf_count, leaf_count), dtype=np.float64)
        leaf_indices_by_node = _leaf_indices_by_node(self.forest)

        for group_index, group in enumerate(self.forest.groups):
            group_leaf_indices = np.asarray(
                [
                    leaf_index
                    for leaf_index, node_id in enumerate(self.forest.leaf_ids)
                    if self.forest.nodes[node_id].group_index == group_index
                ],
                dtype=np.int64,
            )
            log_second_moment[np.ix_(group_leaf_indices, group_leaf_indices)] += math.log1p(
                variance_by_group[group.name]
            )

        for node in self.forest.nodes:
            if not node.child_ids:
                continue
            context = self.forest.split_context(node.node_id)
            first_mass, second_mass = context.child_expected_masses
            p = first_mass / (first_mass + second_mass)
            kappa = float(self.kappa_by_node[node.node_id])
            if not math.isfinite(kappa) or kappa <= 0.0 or not 0.0 < p < 1.0:
                raise ValueError(
                    f"Cannot compute covariance for invalid split parameters at node {node.node_id}."
                )

            first_indices = leaf_indices_by_node[node.child_ids[0]]
            second_indices = leaf_indices_by_node[node.child_ids[1]]
            denominator = kappa + 1.0
            log_first_second_moment = math.log1p((1.0 - p) / (p * denominator))
            log_second_second_moment = math.log1p(p / ((1.0 - p) * denominator))
            log_cross_second_moment = math.log(kappa) - math.log(denominator)

            log_second_moment[np.ix_(first_indices, first_indices)] += log_first_second_moment
            log_second_moment[np.ix_(second_indices, second_indices)] += log_second_second_moment
            log_second_moment[np.ix_(first_indices, second_indices)] += log_cross_second_moment
            log_second_moment[np.ix_(second_indices, first_indices)] += log_cross_second_moment

        with np.errstate(over="raise", invalid="raise"):
            try:
                covariance = np.expm1(log_second_moment)
            except FloatingPointError as error:
                raise ValueError("Terminal scaling covariance exceeds float64 range.") from error
        return (covariance + covariance.T) / 2.0


def _leaf_indices_by_node(forest: GammaBetaForest) -> tuple[IntArray, ...]:
    """Return terminal-state indices descended from every forest node."""
    leaf_index = {node_id: index for index, node_id in enumerate(forest.leaf_ids)}
    descendants: list[IntArray | None] = [None] * len(forest.nodes)
    for node in reversed(forest.nodes):
        if node.child_ids:
            descendants[node.node_id] = np.concatenate(
                tuple(cast(IntArray, descendants[child_id]) for child_id in node.child_ids)
            )
        else:
            descendants[node.node_id] = np.asarray([leaf_index[node.node_id]], dtype=np.int64)
    return tuple(cast(IntArray, descendant) for descendant in descendants)


def _append_group_trees(
    records: list[_NodeRecord],
    expected_mass: FloatArray,
    partition_weight: FloatArray,
    group: GammaBetaGroupSpec,
    group_index: int,
    *,
    kappa_strategy: KappaStrategy | None,
    split_constraint: MomentSplitConstraint | None,
) -> tuple[int, ...]:
    """Append fixed support or weighted-budget trees for one hard group.

    Args:
        records: Global mutable forest records constructed so far.
        expected_mass: Full-grid additive mass used by the Gamma--Beta prior.
        partition_weight: Full-grid additive score used to prioritize splits.
        group: Hard support and refinement settings for the current group.
        group_index: Stable index of ``group`` in the forest group sequence.
        kappa_strategy: Optional concentration policy used by
            ``split_constraint``.
        split_constraint: Optional moment rule for candidate refinements.

    Returns:
        Root IDs for the retained group topology, one per refinable connected
        component or one for a fixed group.
    """
    support_rows, support_columns = np.where(group.mask)
    root_indices: IntArray = np.ravel_multi_index(
        (support_rows, support_columns), expected_mass.shape
    ).astype(np.int64)
    if group.max_depth == 0:
        root_id = len(records)
        records.append(
            _NodeRecord(
                group_index=group_index,
                parent_id=None,
                child_ids=[],
                depth=0,
                geometric_depth=0,
                expected_mass=_finite_additive_sum(
                    expected_mass.reshape(-1)[root_indices],
                    context=f"fixed group {group.name!r}",
                ),
                partition_weight=_finite_additive_sum(
                    partition_weight.reshape(-1)[root_indices],
                    context=f"fixed group {group.name!r} partition weight",
                ),
                flat_indices=root_indices,
            )
        )
        if group.target_regions not in (None, 1):
            raise ValueError(
                f"Gamma-Beta group {group.name!r} requests {group.target_regions} regions "
                "but max_depth=0 permits only one."
            )
        return (root_id,)

    group_start = len(records)
    root_ids: list[int] = []
    for component_mask in _connected_components(group.mask):
        root_ids.append(
            _append_component_tree(
                records,
                expected_mass,
                partition_weight,
                component_mask,
                group,
                group_index,
            )
        )
    if group.target_regions is None:
        return tuple(root_ids)
    return _retain_weighted_region_budget(
        records,
        group_start=group_start,
        root_ids=tuple(root_ids),
        target_regions=group.target_regions,
        group_name=group.name,
        root_variance=group.root_variance,
        kappa_strategy=kappa_strategy,
        split_constraint=split_constraint,
    )


def _append_component_tree(
    records: list[_NodeRecord],
    expected_mass: FloatArray,
    partition_weight: FloatArray,
    component_mask: npt.NDArray[np.bool_],
    group: GammaBetaGroupSpec,
    group_index: int,
) -> int:
    """Append one four-connected component's masked canonical tree.

    Args:
        records: Global mutable forest records constructed so far.
        expected_mass: Full-grid additive mass used by the Gamma--Beta prior.
        partition_weight: Full-grid additive score used to prioritize splits.
        component_mask: Full-grid support for one connected component.
        group: Hard group whose settings control candidate depth.
        group_index: Stable index of ``group`` in the forest group sequence.

    Returns:
        Root ID of the appended component tree.
    """
    support_rows, support_columns = np.where(component_mask)

    row_start = int(support_rows.min())
    row_stop = int(support_rows.max()) + 1
    column_start = int(support_columns.min())
    column_stop = int(support_columns.max()) + 1
    local_mask = component_mask[row_start:row_stop, column_start:column_stop]
    local_mass = expected_mass[row_start:row_stop, column_start:column_stop]
    local_partition_weight = partition_weight[row_start:row_stop, column_start:column_stop]
    local_shape = (int(local_mask.shape[0]), int(local_mask.shape[1]))
    full_shape = (int(expected_mass.shape[0]), int(expected_mass.shape[1]))
    tree = DyadicTree.from_shape(local_shape)
    return _append_masked_node(
        records,
        tree,
        tree.root_id,
        local_mask,
        local_mass,
        local_partition_weight,
        full_shape,
        row_start,
        column_start,
        group_index,
        parent_id=None,
        depth=0,
        max_depth=group.max_depth,
    )


def _connected_components(mask: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[np.bool_], ...]:
    """Return deterministic four-connected components of a Boolean mask."""
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[npt.NDArray[np.bool_]] = []
    rows, columns = mask.shape

    for start_row, start_column in zip(*np.where(mask & ~visited)):
        if visited[start_row, start_column]:
            continue
        component = np.zeros(mask.shape, dtype=bool)
        pending = [(int(start_row), int(start_column))]
        visited[start_row, start_column] = True
        while pending:
            row, column = pending.pop()
            component[row, column] = True
            for neighbor_row, neighbor_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if (
                    0 <= neighbor_row < rows
                    and 0 <= neighbor_column < columns
                    and mask[neighbor_row, neighbor_column]
                    and not visited[neighbor_row, neighbor_column]
                ):
                    visited[neighbor_row, neighbor_column] = True
                    pending.append((neighbor_row, neighbor_column))
        component.setflags(write=False)
        components.append(component)
    return tuple(components)


def _append_masked_node(
    records: list[_NodeRecord],
    tree: DyadicTree,
    canonical_id: NodeId,
    group_mask: npt.NDArray[np.bool_],
    local_mass: FloatArray,
    local_partition_weight: FloatArray,
    full_shape: tuple[int, int],
    row_offset: int,
    column_offset: int,
    group_index: int,
    *,
    parent_id: int | None,
    depth: int,
    max_depth: int,
) -> int:
    """Append one effective node, skipping unary masked geometric branches.

    Args:
        records: Global mutable forest records constructed so far.
        tree: Canonical dyadic tree over the component bounding rectangle.
        canonical_id: Canonical node whose masked support is being considered.
        group_mask: Component support relative to its bounding rectangle.
        local_mass: Expected-mass field on the bounding rectangle.
        local_partition_weight: Split-priority field on the bounding rectangle.
        full_shape: Shape of the full output grid.
        row_offset: Bounding-rectangle row offset in the full grid.
        column_offset: Bounding-rectangle column offset in the full grid.
        group_index: Stable index of the containing hard group.
        parent_id: Parent effective record ID, or ``None`` at a component root.
        depth: Effective binary split depth, excluding unary mask traversal.
        max_depth: Maximum effective split depth to retain as candidates.

    Returns:
        ID of the appended effective node.
    """
    canonical_id = _descend_unary_mask(tree, canonical_id, group_mask)
    tile = tree.tile(canonical_id)
    tile_mask = group_mask[tile.row_start : tile.row_stop, tile.col_start : tile.col_stop]
    local_rows, local_columns = np.where(tile_mask)
    full_rows = local_rows + row_offset + tile.row_start
    full_columns = local_columns + column_offset + tile.col_start
    flat_indices: IntArray = np.ravel_multi_index((full_rows, full_columns), full_shape).astype(np.int64)
    node_mass = _finite_additive_sum(
        local_mass[tile.row_start : tile.row_stop, tile.col_start : tile.col_stop][tile_mask],
        context=f"group index {group_index}, depth {depth}",
    )
    node_partition_weight = _finite_additive_sum(
        local_partition_weight[tile.row_start : tile.row_stop, tile.col_start : tile.col_stop][tile_mask],
        context=f"group index {group_index}, depth {depth} partition weight",
    )

    node_id = len(records)
    records.append(
        _NodeRecord(
            group_index=group_index,
            parent_id=parent_id,
            child_ids=[],
            depth=depth,
            geometric_depth=tile.depth,
            expected_mass=node_mass,
            partition_weight=node_partition_weight,
            flat_indices=flat_indices,
        )
    )

    canonical_children = tree.children(canonical_id)
    if depth >= max_depth or not canonical_children:
        return node_id

    supported_children = [
        child_id for child_id in canonical_children if _tile_support_count(tree, child_id, group_mask) > 0
    ]
    if len(supported_children) != 2:
        return node_id
    child_masses = [
        _tile_expected_mass(tree, child_id, group_mask, local_mass) for child_id in supported_children
    ]
    if any(child_mass <= 0.0 for child_mass in child_masses):
        return node_id

    child_ids = [
        _append_masked_node(
            records,
            tree,
            child_id,
            group_mask,
            local_mass,
            local_partition_weight,
            full_shape,
            row_offset,
            column_offset,
            group_index,
            parent_id=node_id,
            depth=depth + 1,
            max_depth=max_depth,
        )
        for child_id in supported_children
    ]
    records[node_id].child_ids.extend(child_ids)
    return node_id


def _retain_weighted_region_budget(
    records: list[_NodeRecord],
    *,
    group_start: int,
    root_ids: tuple[int, ...],
    target_regions: int,
    group_name: str,
    root_variance: float,
    kappa_strategy: KappaStrategy | None,
    split_constraint: MomentSplitConstraint | None,
) -> tuple[int, ...]:
    """Retain a weighted best-first region budget subject to moment limits.

    Args:
        records: Global mutable records containing a complete depth-limited
            candidate subtree for the current group at the end of the list.
        group_start: Index of the first current-group candidate record.
        root_ids: Candidate root IDs, one per disconnected component.
        target_regions: Exact number of terminal regions to retain.
        group_name: Human-readable group name used in validation errors.
        root_variance: Variance shared by every component root in the group.
        kappa_strategy: Concentration policy used by ``split_constraint``.
        split_constraint: Optional moment admissibility rule. If its
            ``allow_fewer_regions`` flag is true, the result can contain fewer
            terminal regions than requested.

    Returns:
        Remapped root IDs after unselected candidate descendants are removed.

    Raises:
        ValueError: If the target is below the component count or above the
            depth-limited candidate capacity.
    """
    minimum_regions = len(root_ids)
    candidate_leaf_count = sum(not record.child_ids for record in records[group_start:])
    if target_regions < minimum_regions:
        raise ValueError(
            f"Gamma-Beta group {group_name!r} needs at least {minimum_regions} regions "
            "to keep disconnected components separate."
        )
    if target_regions > candidate_leaf_count:
        raise ValueError(
            f"Gamma-Beta group {group_name!r} requests {target_regions} regions but "
            f"max_depth permits only {candidate_leaf_count}."
        )

    active: list[tuple[float, int, int]] = []
    for root_id in root_ids:
        root = records[root_id]
        if root.child_ids:
            heapq.heappush(
                active,
                (-root.partition_weight, -int(root.flat_indices.size), root_id),
            )

    selected_splits: set[int] = set()
    node_variances = {root_id: root_variance for root_id in root_ids}
    leaf_count = minimum_regions
    while leaf_count < target_regions:
        if not active:
            if split_constraint is not None and split_constraint.allow_fewer_regions:
                break
            raise ValueError(
                f"Gamma-Beta group {group_name!r} cannot reach {target_regions} regions; "
                f"only {leaf_count} satisfy the split constraints."
            )
        _, _, node_id = heapq.heappop(active)
        child_variances: tuple[float, float] | None = None
        if split_constraint is not None:
            if kappa_strategy is None:  # pragma: no cover - guarded by from_groups
                raise RuntimeError("A constrained region budget requires a kappa strategy.")
            context = _record_split_context(records, node_id, group_name)
            kappa = float(kappa_strategy(context))
            if not math.isfinite(kappa) or kappa <= 0.0:
                raise ValueError(
                    f"Kappa strategy returned invalid value {kappa!r} "
                    f"for group {group_name!r}, node {node_id}."
                )
            parent_variance = node_variances[node_id]
            if not split_constraint.accepts(
                context,
                parent_variance=parent_variance,
                kappa=kappa,
            ):
                continue
            first_mass, second_mass = context.child_expected_masses
            moments = gamma_beta_child_moments(
                parent_variance=parent_variance,
                first_expected_fraction=first_mass / (first_mass + second_mass),
                kappa=kappa,
            )
            child_variances = (moments.first_variance, moments.second_variance)

        selected_splits.add(node_id)
        leaf_count += 1
        for child_offset, child_id in enumerate(records[node_id].child_ids):
            child = records[child_id]
            if child_variances is not None:
                node_variances[child_id] = child_variances[child_offset]
            if child.child_ids:
                heapq.heappush(
                    active,
                    (-child.partition_weight, -int(child.flat_indices.size), child_id),
                )

    retained_old_ids: list[int] = []

    def visit(node_id: int) -> None:
        """Collect retained candidate nodes in preorder."""
        retained_old_ids.append(node_id)
        if node_id in selected_splits:
            for child_id in records[node_id].child_ids:
                visit(child_id)

    for root_id in root_ids:
        visit(root_id)

    id_map = {old_id: group_start + offset for offset, old_id in enumerate(retained_old_ids)}
    retained_records = []
    for old_id in retained_old_ids:
        old = records[old_id]
        retained_records.append(
            _NodeRecord(
                group_index=old.group_index,
                parent_id=None if old.parent_id is None else id_map[old.parent_id],
                child_ids=(
                    [id_map[child_id] for child_id in old.child_ids] if old_id in selected_splits else []
                ),
                depth=old.depth,
                geometric_depth=old.geometric_depth,
                expected_mass=old.expected_mass,
                partition_weight=old.partition_weight,
                flat_indices=old.flat_indices,
            )
        )

    del records[group_start:]
    records.extend(retained_records)
    return tuple(id_map[root_id] for root_id in root_ids)


def _record_split_context(
    records: list[_NodeRecord],
    node_id: int,
    group_name: str,
) -> GammaBetaSplitContext:
    """Build concentration metadata from mutable construction records.

    Args:
        records: Candidate records for the forest under construction.
        node_id: Candidate internal-node identifier.
        group_name: Human-readable hard-group name.

    Returns:
        Immutable context equivalent to ``GammaBetaForest.split_context``.
    """
    node = records[node_id]
    first, second = (records[child_id] for child_id in node.child_ids)
    return GammaBetaSplitContext(
        node_id=node_id,
        group_name=group_name,
        depth=node.depth,
        geometric_depth=node.geometric_depth,
        parent_expected_mass=node.expected_mass,
        child_expected_masses=(first.expected_mass, second.expected_mass),
        child_grid_cell_counts=(first.flat_indices.size, second.flat_indices.size),
    )


def _descend_unary_mask(
    tree: DyadicTree,
    node_id: NodeId,
    group_mask: npt.NDArray[np.bool_],
) -> NodeId:
    """Return the first descendant whose mask has zero or two supported children."""
    current_id = node_id
    while True:
        child_ids = tree.children(current_id)
        if not child_ids:
            return current_id
        supported = [
            child_id for child_id in child_ids if _tile_support_count(tree, child_id, group_mask) > 0
        ]
        if len(supported) != 1:
            return current_id
        current_id = supported[0]


def _tile_support_count(
    tree: DyadicTree,
    node_id: NodeId,
    group_mask: npt.NDArray[np.bool_],
) -> int:
    """Return supported grid-cell count inside one canonical tile."""
    tile = tree.tile(node_id)
    return int(group_mask[tile.row_start : tile.row_stop, tile.col_start : tile.col_stop].sum())


def _tile_expected_mass(
    tree: DyadicTree,
    node_id: NodeId,
    group_mask: npt.NDArray[np.bool_],
    local_mass: FloatArray,
) -> float:
    """Return masked expected mass inside one canonical tile."""
    tile = tree.tile(node_id)
    mask = group_mask[tile.row_start : tile.row_stop, tile.col_start : tile.col_stop]
    mass = local_mass[tile.row_start : tile.row_stop, tile.col_start : tile.col_stop]
    return _finite_additive_sum(mass[mask], context=f"canonical node {node_id}")


def _finite_additive_sum(values: npt.ArrayLike, *, context: str) -> float:
    """Return a finite float64 sum with a contextual overflow error."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            total = float(np.sum(values, dtype=np.float64))
    except FloatingPointError as error:
        raise ValueError(f"Additive sum overflowed for {context}.") from error
    if not math.isfinite(total):
        raise ValueError(f"Additive sum must be finite for {context}.")
    return total


__all__ = [
    "DepthKappaStrategy",
    "GammaBetaChildMoments",
    "GammaBetaForest",
    "GammaBetaGroupSpec",
    "GammaBetaNode",
    "GammaBetaSamples",
    "GammaBetaSplitContext",
    "KappaStrategy",
    "gamma_beta_child_moments",
]
