"""Canonical partition states and priors for grouped Gamma--Beta forests.

A grouped Gamma--Beta forest can contain several component roots and a fixed
maximum refinement below each root.  This module represents a partition by one
binary indicator for every internal node: an indicator is one when that node is
split.  Valid masks are ancestry closed, and their active nodes form a unique
frontier covering all declared hard groups.

The exact number of forest partitions at each region count is computed by
dynamic programming on the trees.  This supports a normalized prior that is
uniform over selected values of ``K`` and uniform over partitions conditional
on ``K`` without enumerating the partition catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from collections.abc import Mapping
from typing import Literal

import numpy as np
import numpy.typing as npt

from .gamma_beta import GammaBetaForest

MoveKind = Literal["split", "merge"]


@dataclass(frozen=True, slots=True, eq=False)
class GammaBetaPartitionMove:
    """One local split or merge proposal from a forest partition.

    Attributes:
        split_mask: Read-only canonical candidate mask.
        kind: ``"split"`` or ``"merge"``.
        node_id: Internal forest node changed by the proposal.
        log_q: Log proposal probability from the source state when neighbors
            are selected uniformly.
    """

    split_mask: npt.NDArray[np.bool_]
    kind: MoveKind
    node_id: int
    log_q: float

    def __post_init__(self) -> None:
        """Validate scalar metadata and freeze the candidate mask."""
        mask = np.asarray(self.split_mask)
        if mask.ndim != 1 or mask.dtype.kind not in "biu":
            raise ValueError("split_mask must be a one-dimensional binary mask.")
        if np.any((mask != 0) & (mask != 1)):
            raise ValueError("split_mask must contain only binary values.")
        if self.kind not in {"split", "merge"}:
            raise ValueError("kind must be 'split' or 'merge'.")
        if isinstance(self.node_id, bool) or not isinstance(self.node_id, Integral):
            raise TypeError("node_id must be an integer.")
        if not math.isfinite(self.log_q) or self.log_q > 0.0:
            raise ValueError("log_q must be finite and non-positive.")
        frozen = np.asarray(mask, dtype=np.bool_).copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "split_mask", frozen)
        object.__setattr__(self, "node_id", int(self.node_id))


@dataclass(frozen=True, slots=True, eq=False)
class GammaBetaPartitionLayout:
    """Canonical split-mask codec for one fixed Gamma--Beta forest.

    Construct layouts with :meth:`from_forest`.

    Attributes:
        forest: Fixed maximum grouped forest.
        split_node_ids: Internal node IDs in stable node order.
        split_index_by_node: Read-only node-indexed array containing the mask
            index for internal nodes and ``-1`` for leaves.
        partition_counts_by_k: Exact number of valid frontiers at each region
            count.  Indices below the component-root count contain zero.
    """

    forest: GammaBetaForest
    split_node_ids: tuple[int, ...]
    split_index_by_node: npt.NDArray[np.int64]
    partition_counts_by_k: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate static topology metadata and exact count identities."""
        if not isinstance(self.forest, GammaBetaForest):
            raise TypeError("forest must be a GammaBetaForest.")
        expected_split_nodes = tuple(
            node.node_id for node in self.forest.nodes if node.child_ids
        )
        if self.split_node_ids != expected_split_nodes:
            raise ValueError("split_node_ids must contain all internal nodes in node order.")
        index_by_node = np.asarray(self.split_index_by_node, dtype=np.int64)
        if index_by_node.shape != (len(self.forest.nodes),):
            raise ValueError("split_index_by_node has the wrong shape.")
        expected_indices = np.full(len(self.forest.nodes), -1, dtype=np.int64)
        for split_index, node_id in enumerate(self.split_node_ids):
            expected_indices[node_id] = split_index
        if not np.array_equal(index_by_node, expected_indices):
            raise ValueError("split_index_by_node does not match split_node_ids.")
        frozen = index_by_node.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "split_index_by_node", frozen)

        counts = tuple(self.partition_counts_by_k)
        if len(counts) != len(self.forest.leaf_ids) + 1:
            raise ValueError("partition_counts_by_k must end at the maximum leaf count.")
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in counts):
            raise ValueError("partition_counts_by_k must contain non-negative integers.")
        if counts[len(self.forest.root_ids)] < 1 or counts[-1] != 1:
            raise ValueError("partition_counts_by_k violates root or full-leaf identities.")
        object.__setattr__(self, "partition_counts_by_k", tuple(int(value) for value in counts))

    @classmethod
    def from_forest(cls, forest: GammaBetaForest) -> GammaBetaPartitionLayout:
        """Compile a split-mask codec and exact count table from a forest.

        Args:
            forest: Fixed maximum grouped Gamma--Beta forest.

        Returns:
            Immutable partition layout.

        Raises:
            TypeError: If ``forest`` has the wrong type.
        """
        if not isinstance(forest, GammaBetaForest):
            raise TypeError("forest must be a GammaBetaForest.")
        split_node_ids = tuple(node.node_id for node in forest.nodes if node.child_ids)
        index_by_node = np.full(len(forest.nodes), -1, dtype=np.int64)
        for split_index, node_id in enumerate(split_node_ids):
            index_by_node[node_id] = split_index
        counts = _forest_partition_counts(forest)
        return cls(
            forest=forest,
            split_node_ids=split_node_ids,
            split_index_by_node=index_by_node,
            partition_counts_by_k=counts,
        )

    @property
    def split_count(self) -> int:
        """Return the number of possible binary split indicators."""
        return len(self.split_node_ids)

    @property
    def minimum_regions(self) -> int:
        """Return the number of always-active component roots."""
        return len(self.forest.root_ids)

    @property
    def maximum_regions(self) -> int:
        """Return the terminal-node count in the maximum forest."""
        return len(self.forest.leaf_ids)

    def canonical_split_mask(self, values: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Validate, copy, and freeze one ancestry-closed binary mask.

        Args:
            values: Boolean or integer vector in :attr:`split_node_ids` order.

        Returns:
            Read-only Boolean mask.

        Raises:
            ValueError: If shape, values, or ancestry are invalid.
        """
        source = np.asarray(values)
        expected_shape = (self.split_count,)
        if source.shape != expected_shape:
            raise ValueError(f"split_mask must have shape {expected_shape}.")
        if source.dtype.kind not in "biu" or np.any((source != 0) & (source != 1)):
            raise ValueError("split_mask must contain only binary values.")
        mask = np.asarray(source, dtype=np.bool_).copy()
        for node_id in self.split_node_ids:
            if not mask[self.split_index_by_node[node_id]]:
                continue
            parent_id = self.forest.nodes[node_id].parent_id
            if parent_id is None:
                continue
            parent_index = self.split_index_by_node[parent_id]
            if parent_index >= 0 and not mask[parent_index]:
                raise ValueError("split_mask must be ancestry closed.")
        mask.setflags(write=False)
        return mask

    def active_node_ids(self, split_mask: npt.ArrayLike) -> tuple[int, ...]:
        """Decode a canonical split mask into its active forest frontier.

        Args:
            split_mask: Candidate mask in stable split-node order.

        Returns:
            Active node IDs in deterministic depth-first root order.
        """
        mask = self.canonical_split_mask(split_mask)
        active: list[int] = []

        def visit(node_id: int) -> None:
            """Append terminal active descendants of one reached node."""
            node = self.forest.nodes[node_id]
            split_index = self.split_index_by_node[node_id]
            if not node.child_ids or not mask[split_index]:
                active.append(node_id)
                return
            for child_id in node.child_ids:
                visit(child_id)

        for root_id in self.forest.root_ids:
            visit(root_id)
        return tuple(active)

    def split_mask_from_active(
        self,
        active_node_ids: tuple[int, ...],
    ) -> npt.NDArray[np.bool_]:
        """Encode an exact active frontier as a canonical split mask.

        Args:
            active_node_ids: Forest nodes that must cover every root exactly.

        Returns:
            Read-only canonical split mask.

        Raises:
            ValueError: If IDs do not define a complete non-overlapping
                frontier.
        """
        active = set()
        for node_id in active_node_ids:
            if isinstance(node_id, bool) or not isinstance(node_id, Integral):
                raise ValueError("active_node_ids must contain integer node IDs.")
            integer_id = int(node_id)
            if integer_id < 0 or integer_id >= len(self.forest.nodes):
                raise ValueError(f"Unknown active forest node {node_id!r}.")
            if integer_id in active:
                raise ValueError("active_node_ids must not contain duplicates.")
            active.add(integer_id)

        mask = np.zeros(self.split_count, dtype=np.bool_)
        consumed: set[int] = set()

        def visit(node_id: int) -> None:
            """Encode one root subtree and record consumed active nodes."""
            if node_id in active:
                consumed.add(node_id)
                return
            node = self.forest.nodes[node_id]
            if not node.child_ids:
                raise ValueError("active_node_ids do not cover every forest root.")
            mask[self.split_index_by_node[node_id]] = True
            for child_id in node.child_ids:
                visit(child_id)

        for root_id in self.forest.root_ids:
            visit(root_id)
        if consumed != active:
            raise ValueError("active_node_ids overlap or lie below another active node.")
        return self.canonical_split_mask(mask)

    def region_count(self, split_mask: npt.ArrayLike) -> int:
        """Return ``component roots + accepted splits`` for a valid mask."""
        mask = self.canonical_split_mask(split_mask)
        return self.minimum_regions + int(mask.sum())

    def initial_split_mask(self, region_count: int) -> npt.NDArray[np.bool_]:
        """Build a deterministic valid mask with one requested region count.

        Args:
            region_count: Target between :attr:`minimum_regions` and
                :attr:`maximum_regions` with positive partition count.

        Returns:
            Canonical mask obtained by splitting the first available active
            internal node repeatedly.

        Raises:
            TypeError: If ``region_count`` is not an integer.
            ValueError: If no partition exists at the requested count.
        """
        if isinstance(region_count, bool) or not isinstance(region_count, Integral):
            raise TypeError("region_count must be an integer.")
        target = int(region_count)
        if (
            target < self.minimum_regions
            or target > self.maximum_regions
            or self.partition_counts_by_k[target] == 0
        ):
            raise ValueError(f"No forest partition exists with K={target}.")
        mask = np.zeros(self.split_count, dtype=np.bool_)
        while self.region_count(mask) < target:
            active = self.active_node_ids(mask)
            node_id = next(
                node_id for node_id in active if self.forest.nodes[node_id].child_ids
            )
            mask[self.split_index_by_node[node_id]] = True
        return self.canonical_split_mask(mask)

    def neighbors(self, split_mask: npt.ArrayLike) -> tuple[GammaBetaPartitionMove, ...]:
        """Enumerate unique one-split and one-merge neighboring partitions.

        Args:
            split_mask: Source canonical partition mask.

        Returns:
            Moves in stable split-then-merge node order.  Every move has
            ``log_q = -log(number of source neighbors)``.
        """
        mask = self.canonical_split_mask(split_mask)
        active = set(self.active_node_ids(mask))
        candidates: list[tuple[npt.NDArray[np.bool_], MoveKind, int]] = []
        for node_id in self.split_node_ids:
            split_index = self.split_index_by_node[node_id]
            if node_id in active:
                candidate = mask.copy()
                candidate[split_index] = True
                candidates.append((candidate, "split", node_id))

        for node_id in self.split_node_ids:
            split_index = self.split_index_by_node[node_id]
            if not mask[split_index]:
                continue
            children = self.forest.nodes[node_id].child_ids
            if all(child_id in active for child_id in children):
                candidate = mask.copy()
                candidate[split_index] = False
                candidates.append((candidate, "merge", node_id))

        if not candidates:
            return ()
        log_q = -math.log(len(candidates))
        return tuple(
            GammaBetaPartitionMove(
                split_mask=self.canonical_split_mask(candidate),
                kind=kind,
                node_id=node_id,
                log_q=log_q,
            )
            for candidate, kind, node_id in candidates
        )


@dataclass(frozen=True, slots=True, eq=False)
class GammaBetaRegionCountPrior:
    """Normalized ``p(P) = p(K) / N_K`` prior for forest partitions.

    Attributes:
        layout: Canonical partition codec and exact count table.
        log_probability_by_k: Read-only lookup table of per-partition log
            probabilities indexed by region count.
    """

    layout: GammaBetaPartitionLayout
    log_probability_by_k: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate and freeze the symbolic prior lookup table."""
        if not isinstance(self.layout, GammaBetaPartitionLayout):
            raise TypeError("layout must be a GammaBetaPartitionLayout.")
        values = np.asarray(self.log_probability_by_k, dtype=np.float64)
        expected_shape = (self.layout.maximum_regions + 1,)
        if values.shape != expected_shape:
            raise ValueError(f"log_probability_by_k must have shape {expected_shape}.")
        if np.any(np.isnan(values)) or np.any(values == math.inf):
            raise ValueError("log_probability_by_k cannot contain NaN or positive infinity.")
        log_masses: list[float] = []
        for region_count, partition_count in enumerate(
            self.layout.partition_counts_by_k
        ):
            if partition_count and math.isfinite(float(values[region_count])):
                log_masses.append(
                    math.log(partition_count) + float(values[region_count])
                )
        if not log_masses:
            raise ValueError("At least one forest partition must have positive prior mass.")
        maximum_log_mass = max(log_masses)
        log_total = maximum_log_mass + math.log(
            sum(math.exp(value - maximum_log_mass) for value in log_masses)
        )
        if not math.isclose(log_total, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Partition prior probabilities must sum to one.")
        frozen = values.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "log_probability_by_k", frozen)

    @classmethod
    def from_marginal_probabilities(
        cls,
        layout: GammaBetaPartitionLayout,
        probabilities_by_k: Mapping[int, float] | npt.ArrayLike,
    ) -> GammaBetaRegionCountPrior:
        """Share declared marginal K probabilities uniformly within each K.

        Args:
            layout: Canonical forest partition layout.
            probabilities_by_k: Non-negative marginal masses indexed by K,
                either as a mapping or a vector with shape
                ``(maximum_regions + 1,)``. Values are normalized internally.

        Returns:
            Exactly normalized ``p(P) = p(K) / N_K`` prior.

        Raises:
            TypeError: If ``layout`` or mapping keys have invalid types.
            ValueError: If masses are invalid, unavailable K receives positive
                mass, or total supplied mass is zero.
        """
        if not isinstance(layout, GammaBetaPartitionLayout):
            raise TypeError("layout must be a GammaBetaPartitionLayout.")
        if isinstance(probabilities_by_k, Mapping):
            masses = np.zeros(layout.maximum_regions + 1, dtype=np.float64)
            for region_count, probability in probabilities_by_k.items():
                if isinstance(region_count, bool) or not isinstance(region_count, Integral):
                    raise TypeError("Region-count probability keys must be integers.")
                integer_count = int(region_count)
                if integer_count < 0 or integer_count > layout.maximum_regions:
                    raise ValueError(f"Unknown region count K={integer_count}.")
                masses[integer_count] = float(probability)
        else:
            masses = np.asarray(probabilities_by_k, dtype=np.float64)
            expected_shape = (layout.maximum_regions + 1,)
            if masses.shape != expected_shape:
                raise ValueError(f"probabilities_by_k must have shape {expected_shape}.")
            masses = masses.copy()
        if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
            raise ValueError("Region-count probabilities must be finite and non-negative.")
        for region_count, mass in enumerate(masses):
            if mass > 0.0 and layout.partition_counts_by_k[region_count] == 0:
                raise ValueError(f"Positive mass was assigned to unavailable K={region_count}.")
        total = float(masses.sum())
        if total <= 0.0:
            raise ValueError("At least one region count must have positive mass.")
        masses /= total

        table = np.full(layout.maximum_regions + 1, -math.inf, dtype=np.float64)
        for region_count, mass in enumerate(masses):
            if mass > 0.0:
                table[region_count] = math.log(float(mass)) - math.log(
                    layout.partition_counts_by_k[region_count]
                )
        return cls(layout=layout, log_probability_by_k=table)

    @classmethod
    def uniform_k(
        cls,
        layout: GammaBetaPartitionLayout,
        *,
        minimum_regions: int | None = None,
        maximum_regions: int | None = None,
    ) -> GammaBetaRegionCountPrior:
        """Assign equal mass to supported K and equal mass within each K.

        Args:
            layout: Canonical forest partition layout.
            minimum_regions: Smallest supported region count.  Defaults to the
                component-root count.
            maximum_regions: Largest supported region count.  Defaults to the
                maximum terminal count.

        Returns:
            Exactly normalized region-count prior.

        Raises:
            TypeError: If bounds are not integers.
            ValueError: If bounds are invalid or include an unavailable K.
        """
        if not isinstance(layout, GammaBetaPartitionLayout):
            raise TypeError("layout must be a GammaBetaPartitionLayout.")
        lower = layout.minimum_regions if minimum_regions is None else minimum_regions
        upper = layout.maximum_regions if maximum_regions is None else maximum_regions
        if (
            isinstance(lower, bool)
            or not isinstance(lower, Integral)
            or isinstance(upper, bool)
            or not isinstance(upper, Integral)
        ):
            raise TypeError("Region-count bounds must be integers.")
        lower = int(lower)
        upper = int(upper)
        if lower < layout.minimum_regions or upper > layout.maximum_regions or lower > upper:
            raise ValueError("Region-count bounds lie outside the forest partition range.")
        supported = tuple(
            region_count
            for region_count in range(lower, upper + 1)
            if layout.partition_counts_by_k[region_count] > 0
        )
        if len(supported) != upper - lower + 1:
            raise ValueError("Every K in the requested range must have a valid partition.")

        return cls.from_marginal_probabilities(
            layout,
            {region_count: 1.0 for region_count in supported},
        )

    @classmethod
    def geometric_extra_regions(
        cls,
        layout: GammaBetaPartitionLayout,
        *,
        continuation_probability: float = 0.5,
        minimum_regions: int | None = None,
        maximum_regions: int | None = None,
    ) -> GammaBetaRegionCountPrior:
        """Use a truncated geometric prior on splits beyond minimum K.

        If ``K_min`` is the lower bound and ``q`` is
        ``continuation_probability``, the unnormalized marginal mass is
        ``p(K) proportional to q**(K - K_min)``. Conditional on K, partitions
        remain uniform. Smaller ``q`` more strongly favors compact frontiers.

        Args:
            layout: Canonical forest partition layout.
            continuation_probability: Finite value strictly between zero and
                one.
            minimum_regions: Smallest supported K. Defaults to the component
                root count.
            maximum_regions: Largest supported K. Defaults to the maximum
                terminal count.

        Returns:
            Normalized truncated geometric partition prior.

        Raises:
            ValueError: If the continuation probability or bounds are invalid.
        """
        if not isinstance(layout, GammaBetaPartitionLayout):
            raise TypeError("layout must be a GammaBetaPartitionLayout.")
        if (
            not math.isfinite(continuation_probability)
            or continuation_probability <= 0.0
            or continuation_probability >= 1.0
        ):
            raise ValueError("continuation_probability must lie strictly between zero and one.")
        lower = layout.minimum_regions if minimum_regions is None else minimum_regions
        upper = layout.maximum_regions if maximum_regions is None else maximum_regions
        if (
            isinstance(lower, bool)
            or not isinstance(lower, Integral)
            or isinstance(upper, bool)
            or not isinstance(upper, Integral)
        ):
            raise TypeError("Region-count bounds must be integers.")
        lower = int(lower)
        upper = int(upper)
        if lower < layout.minimum_regions or upper > layout.maximum_regions or lower > upper:
            raise ValueError("Region-count bounds lie outside the forest partition range.")
        masses = {
            region_count: continuation_probability ** (region_count - lower)
            for region_count in range(lower, upper + 1)
        }
        return cls.from_marginal_probabilities(layout, masses)

    @property
    def marginal_probability_by_k(self) -> npt.NDArray[np.float64]:
        """Return read-only normalized marginal prior masses by region count."""
        result = np.zeros(self.layout.maximum_regions + 1, dtype=np.float64)
        for region_count, partition_count in enumerate(
            self.layout.partition_counts_by_k
        ):
            log_probability = self.log_probability_by_k[region_count]
            if partition_count and math.isfinite(float(log_probability)):
                result[region_count] = math.exp(
                    math.log(partition_count) + float(log_probability)
                )
        result.setflags(write=False)
        return result

    def __call__(self, split_mask: npt.ArrayLike) -> float:
        """Return the normalized log prior for one canonical partition mask."""
        region_count = self.layout.region_count(split_mask)
        return float(self.log_probability_by_k[region_count])


def _forest_partition_counts(forest: GammaBetaForest) -> tuple[int, ...]:
    """Return exact partition counts by K using tree-polynomial products."""
    memo: dict[int, tuple[int, ...]] = {}

    def node_counts(node_id: int) -> tuple[int, ...]:
        """Return count polynomial for one rooted binary subtree."""
        if node_id in memo:
            return memo[node_id]
        node = forest.nodes[node_id]
        if not node.child_ids:
            result = (0, 1)
        else:
            first = node_counts(node.child_ids[0])
            second = node_counts(node.child_ids[1])
            split_counts = _convolve_counts(first, second)
            result_values = list(split_counts)
            if len(result_values) < 2:
                result_values.extend([0] * (2 - len(result_values)))
            result_values[1] += 1
            result = tuple(result_values)
        memo[node_id] = result
        return result

    forest_counts: tuple[int, ...] = (1,)
    for root_id in forest.root_ids:
        forest_counts = _convolve_counts(forest_counts, node_counts(root_id))
    expected_length = len(forest.leaf_ids) + 1
    if len(forest_counts) < expected_length:
        forest_counts = forest_counts + (0,) * (expected_length - len(forest_counts))
    return forest_counts


def _convolve_counts(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    """Convolve integer partition-count polynomials without overflow."""
    result = [0] * (len(first) + len(second) - 1)
    for first_index, first_value in enumerate(first):
        if first_value == 0:
            continue
        for second_index, second_value in enumerate(second):
            if second_value:
                result[first_index + second_index] += first_value * second_value
    return tuple(result)


__all__ = [
    "GammaBetaPartitionLayout",
    "GammaBetaPartitionMove",
    "GammaBetaRegionCountPrior",
    "MoveKind",
]
