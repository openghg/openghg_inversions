"""Immutable canonical dyadic trees and exact partition frontiers.

This module provides the fixed spatial topology used by the experimental
Gamma--Beta reversible-jump baseline.  A rectangular root is bisected along
its longer axis, with rows winning ties.  The first child receives the lower
half of an extent and the second child receives any odd extra cell.  Nodes are
numbered deterministically in depth-first preorder.

A :class:`DyadicFrontier` is a canonical sorted tuple of node IDs that covers
the root exactly once.  Local split and merge operations return new frontiers;
tree geometry and source frontiers are never mutated.  Exhaustive enumeration
is supplied only as a tiny-tree oracle.  :class:`SubtreePartitionIndex` and
:func:`partition_counts_by_k` use arbitrary-precision Python integers without
materializing frontiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

#: Zero-based node identifier assigned in depth-first preorder.
NodeId: TypeAlias = int
IntArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DyadicNode:
    """One immutable rectangular node in a canonical dyadic tree.

    All spatial bounds are half-open.  ``cell_indices`` contains the covered
    grid cells as C-order flat indices, which lets state builders aggregate
    finest-grid arrays without reconstructing masks.

    Attributes:
        node_id: Stable zero-based depth-first preorder identifier.
        row_start: Inclusive first covered row.
        row_stop: Exclusive final covered row.
        col_start: Inclusive first covered column.
        col_stop: Exclusive final covered column.
        depth: Number of edges from the root.
        parent_id: Parent node identifier, or ``None`` for the root.
        child_ids: Ordered child pair, or an empty tuple for a grid cell.
        cell_indices: Covered C-order flat grid indices.
    """

    node_id: NodeId
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    depth: int
    parent_id: NodeId | None
    child_ids: tuple[NodeId, ...]
    cell_indices: tuple[int, ...]

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Return the node's half-open spatial bounds.

        Returns:
            ``(row_start, row_stop, col_start, col_stop)``.
        """
        return self.row_start, self.row_stop, self.col_start, self.col_stop

    @property
    def height(self) -> int:
        """Return the number of covered rows.

        Returns:
            Positive row count.
        """
        return self.row_stop - self.row_start

    @property
    def width(self) -> int:
        """Return the number of covered columns.

        Returns:
            Positive column count.
        """
        return self.col_stop - self.col_start

    @property
    def area(self) -> int:
        """Return the number of covered grid cells.

        Returns:
            Positive grid-cell count.
        """
        return self.height * self.width

    @property
    def is_cell(self) -> bool:
        """Return whether this node is one terminal grid cell.

        Returns:
            ``True`` exactly when the node has no children.
        """
        return not self.child_ids


@dataclass(frozen=True, slots=True, init=False)
class CanonicalDyadicTree:
    """Complete immutable canonical bisection tree for one rectangular grid.

    Construct a tree with :meth:`from_shape`.  All stored topology tuples use
    node-ID order, so direct indexing and iteration are deterministic.

    Attributes:
        shape: Positive ``(rows, columns)`` root shape.
        nodes: Complete nodes in contiguous node-ID order.
        leaf_ids: Terminal cell IDs in node-ID order.
        internal_node_ids: Splittable node IDs in node-ID order.
    """

    shape: tuple[int, int]
    nodes: tuple[DyadicNode, ...]
    leaf_ids: tuple[NodeId, ...]
    internal_node_ids: tuple[NodeId, ...]

    def __init__(self, shape: tuple[int, int]) -> None:
        """Build the canonical tree for ``shape``.

        Args:
            shape: Positive two-dimensional ``(rows, columns)`` grid shape.

        Raises:
            TypeError: If an extent is Boolean or not integer-like.
            ValueError: If the shape does not have two positive extents.
        """
        rows, columns = _validate_shape(shape)
        records: list[tuple[int, int, int, int, int, NodeId | None, tuple[NodeId, ...]]] = []
        pending: list[tuple[int, int, int, int, int, NodeId | None]] = [(0, rows, 0, columns, 0, None)]

        while pending:
            row_start, row_stop, col_start, col_stop, depth, parent_id = pending.pop()
            node_id = len(records)
            records.append((row_start, row_stop, col_start, col_stop, depth, parent_id, ()))
            if parent_id is not None:
                parent = records[parent_id]
                records[parent_id] = (*parent[:-1], (*parent[-1], node_id))

            child_bounds = _split_bounds(row_start, row_stop, col_start, col_stop)
            if child_bounds is None:
                continue
            first, second = child_bounds
            # The stack is last-in, first-out, so push the second child first.
            pending.append((*second, depth + 1, node_id))
            pending.append((*first, depth + 1, node_id))

        nodes = tuple(
            DyadicNode(
                node_id=node_id,
                row_start=row_start,
                row_stop=row_stop,
                col_start=col_start,
                col_stop=col_stop,
                depth=depth,
                parent_id=parent_id,
                child_ids=child_ids,
                cell_indices=tuple(
                    row * columns + column
                    for row in range(row_start, row_stop)
                    for column in range(col_start, col_stop)
                ),
            )
            for node_id, (
                row_start,
                row_stop,
                col_start,
                col_stop,
                depth,
                parent_id,
                child_ids,
            ) in enumerate(records)
        )
        object.__setattr__(self, "shape", (rows, columns))
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(
            self,
            "leaf_ids",
            tuple(node.node_id for node in nodes if node.is_cell),
        )
        object.__setattr__(
            self,
            "internal_node_ids",
            tuple(node.node_id for node in nodes if not node.is_cell),
        )

    @classmethod
    def from_shape(cls, shape: tuple[int, int]) -> CanonicalDyadicTree:
        """Return the complete canonical tree for ``shape``.

        Args:
            shape: Positive two-dimensional ``(rows, columns)`` grid shape.

        Returns:
            Newly constructed immutable tree.

        Raises:
            TypeError: If an extent is Boolean or not integer-like.
            ValueError: If the shape does not have two positive extents.
        """
        return cls(shape)

    @property
    def root_id(self) -> NodeId:
        """Return the canonical root identifier.

        Returns:
            Zero, the fixed preorder ID of the root.
        """
        return 0

    def node(self, node_id: NodeId) -> DyadicNode:
        """Return one node after validating its identifier.

        Args:
            node_id: Candidate integer-like node identifier.

        Returns:
            Immutable node metadata.

        Raises:
            KeyError: If ``node_id`` is not a node in this tree.
        """
        return self.nodes[self._node_index(node_id)]

    def children(self, node_id: NodeId) -> tuple[NodeId, ...]:
        """Return the ordered child IDs for ``node_id``.

        Args:
            node_id: Candidate integer-like node identifier.

        Returns:
            Two child IDs, or an empty tuple for a grid cell.

        Raises:
            KeyError: If ``node_id`` is not a node in this tree.
        """
        return self.node(node_id).child_ids

    def parent(self, node_id: NodeId) -> NodeId | None:
        """Return the parent ID for ``node_id``, or ``None`` for the root.

        Args:
            node_id: Candidate integer-like node identifier.

        Returns:
            Parent identifier or ``None``.

        Raises:
            KeyError: If ``node_id`` is not a node in this tree.
        """
        return self.node(node_id).parent_id

    def splittable_nodes(self, frontier: DyadicFrontier) -> tuple[NodeId, ...]:
        """Return active frontier nodes that have canonical children.

        Args:
            frontier: Exact frontier over this tree.

        Returns:
            Splittable IDs in stable node-ID order.

        Raises:
            TypeError: If ``frontier`` has the wrong type.
            ValueError: If ``frontier`` is invalid for this tree.
        """
        _require_frontier(frontier)
        frontier.validate(self)
        return tuple(node_id for node_id in frontier.node_ids if self.nodes[node_id].child_ids)

    def mergeable_parents(self, frontier: DyadicFrontier) -> tuple[NodeId, ...]:
        """Return parents whose two children are active frontier leaves.

        Args:
            frontier: Exact frontier over this tree.

        Returns:
            Mergeable parent IDs in stable node-ID order.

        Raises:
            TypeError: If ``frontier`` has the wrong type.
            ValueError: If ``frontier`` is invalid for this tree.
        """
        _require_frontier(frontier)
        frontier.validate(self)
        active = frozenset(frontier.node_ids)
        candidates = {
            parent_id
            for node_id in frontier.node_ids
            if (parent_id := self.nodes[node_id].parent_id) is not None
            and all(child_id in active for child_id in self.nodes[parent_id].child_ids)
        }
        return tuple(sorted(candidates))

    def _node_index(self, node_id: NodeId) -> int:
        """Return a checked tuple index for one candidate node ID.

        Args:
            node_id: Candidate integer-like node identifier.

        Returns:
            Built-in integer suitable for indexing :attr:`nodes`.

        Raises:
            KeyError: If ``node_id`` is Boolean, non-integer-like, negative,
                or beyond the final node ID.
        """
        if isinstance(node_id, bool):
            raise KeyError(f"Unknown dyadic node ID {node_id!r}.")
        try:
            node_index = index(node_id)
        except TypeError as error:
            raise KeyError(f"Unknown dyadic node ID {node_id!r}.") from error
        if node_index < 0 or node_index >= len(self.nodes):
            raise KeyError(f"Unknown dyadic node ID {node_id!r}.")
        return node_index


@dataclass(frozen=True, slots=True)
class DyadicFrontier:
    """Immutable exact frontier represented by canonical sorted node IDs.

    Args:
        node_ids: Unique integer-like node IDs.  Their association with a tree
            and exact coverage are checked by :meth:`validate`.

    Raises:
        TypeError: If an ID is Boolean or not integer-like.
        ValueError: If IDs contain duplicates.
    """

    node_ids: tuple[NodeId, ...]

    def __post_init__(self) -> None:
        """Normalize IDs to a unique ascending tuple."""
        normalized: list[int] = []
        try:
            source_ids = tuple(self.node_ids)
        except TypeError as error:
            raise TypeError("node_ids must be an iterable of integer node IDs.") from error
        for node_id in source_ids:
            if isinstance(node_id, bool):
                raise TypeError("node_ids must contain only integer node IDs.")
            try:
                normalized.append(index(node_id))
            except TypeError as error:
                raise TypeError("node_ids must contain only integer node IDs.") from error
        if len(set(normalized)) != len(normalized):
            raise ValueError("node_ids must not contain duplicates.")
        object.__setattr__(self, "node_ids", tuple(sorted(normalized)))

    @classmethod
    def root(cls, tree: CanonicalDyadicTree) -> DyadicFrontier:
        """Return the coarsest frontier containing only ``tree``'s root.

        Args:
            tree: Canonical tree to cover.

        Returns:
            One-node exact frontier.

        Raises:
            TypeError: If ``tree`` has the wrong type.
        """
        _require_tree(tree)
        return cls((tree.root_id,))

    def __len__(self) -> int:
        """Return the number of active regions.

        Returns:
            Length of the canonical node-ID tuple.
        """
        return len(self.node_ids)

    def validate(self, tree: CanonicalDyadicTree) -> None:
        """Require IDs to form a complete non-overlapping tree frontier.

        Args:
            tree: Canonical tree against which to validate this frontier.

        Raises:
            TypeError: If ``tree`` has the wrong type.
            ValueError: If the frontier is empty, contains an unknown ID,
                contains an ancestor and descendant, or leaves a coverage gap.
        """
        _require_tree(tree)
        if not self.node_ids:
            raise ValueError("A dyadic frontier must contain at least one node.")
        active = frozenset(self.node_ids)

        pending = [tree.root_id]
        reached_active: set[NodeId] = set()
        has_coverage_gap = False
        while pending:
            node_id = pending.pop()
            if node_id in active:
                reached_active.add(node_id)
                continue
            child_ids = tree.nodes[node_id].child_ids
            if not child_ids:
                has_coverage_gap = True
                continue
            pending.extend(child_ids)

        # Preserve the original sorted-ID error precedence while avoiding one
        # ancestor walk per active node. Valid IDs not reached by the root
        # traversal are necessarily hidden below another active node.
        for node_id in self.node_ids:
            if node_id < 0 or node_id >= len(tree.nodes):
                raise ValueError(f"Frontier node ID {node_id!r} is not in the tree.")
            if node_id not in reached_active:
                raise ValueError("A dyadic frontier cannot contain an ancestor and its descendant.")
        if has_coverage_gap:
            raise ValueError("Dyadic frontier nodes do not exactly cover the tree root.")

    def active_split_nodes(self, tree: CanonicalDyadicTree) -> tuple[NodeId, ...]:
        """Return internal nodes split to produce this exact frontier.

        Args:
            tree: Canonical tree defining frontier ancestry.

        Returns:
            Split-node IDs in stable preorder/node-ID order.

        Raises:
            TypeError: If ``tree`` has the wrong type.
            ValueError: If this frontier is invalid for ``tree``.
        """
        self.validate(tree)
        active = frozenset(self.node_ids)
        split_nodes: list[NodeId] = []
        pending = [tree.root_id]
        while pending:
            node_id = pending.pop()
            if node_id in active:
                continue
            split_nodes.append(node_id)
            pending.extend(reversed(tree.nodes[node_id].child_ids))
        return tuple(split_nodes)

    def split(self, tree: CanonicalDyadicTree, node_id: NodeId) -> DyadicFrontier:
        """Replace one active non-cell node with its two children.

        Args:
            tree: Canonical tree defining the split.
            node_id: Active node to split.

        Returns:
            New validated frontier; this frontier is unchanged.

        Raises:
            TypeError: If ``tree`` is not a :class:`CanonicalDyadicTree`.
            ValueError: If this frontier is invalid, the node is not active,
                or the node is a terminal cell.
        """
        self.validate(tree)
        if node_id not in self.node_ids:
            raise ValueError(f"Node {node_id!r} is not active and cannot be split.")
        child_ids = tree.children(node_id)
        if not child_ids:
            raise ValueError(f"Cell node {node_id!r} cannot be split.")
        result = DyadicFrontier(
            tuple(active_id for active_id in self.node_ids if active_id != node_id) + child_ids
        )
        result.validate(tree)
        return result

    def merge(self, tree: CanonicalDyadicTree, parent_id: NodeId) -> DyadicFrontier:
        """Replace one active sibling pair with its parent.

        Args:
            tree: Canonical tree defining the merge.
            parent_id: Parent whose two children must both be active.

        Returns:
            New validated frontier; this frontier is unchanged.

        Raises:
            TypeError: If ``tree`` is not a :class:`CanonicalDyadicTree`.
            KeyError: If ``parent_id`` is not in ``tree``.
            ValueError: If this frontier is invalid, the parent is a cell, or
                its complete child pair is not active.
        """
        self.validate(tree)
        child_ids = tree.children(parent_id)
        if not child_ids:
            raise ValueError(f"Cell node {parent_id!r} has no children to merge.")
        active = frozenset(self.node_ids)
        if not all(child_id in active for child_id in child_ids):
            raise ValueError("A merge requires both active children of the same parent.")
        result = DyadicFrontier(
            tuple(node_id for node_id in self.node_ids if node_id not in child_ids) + (parent_id,)
        )
        result.validate(tree)
        return result

    def render_labels(self, tree: CanonicalDyadicTree) -> IntArray:
        """Render compact positive labels in canonical frontier order.

        Args:
            tree: Canonical tree providing bounds and output shape.

        Returns:
            ``int64`` array with ``tree.shape``.  Label one corresponds to the
            first frontier node, label two to the second, and so on.

        Raises:
            TypeError: If ``tree`` has the wrong type.
            ValueError: If this frontier is invalid for ``tree``.
        """
        self.validate(tree)
        labels = np.zeros(tree.shape, dtype=np.int64)
        for label, node_id in enumerate(self.node_ids, start=1):
            node = tree.node(node_id)
            labels[
                node.row_start : node.row_stop,
                node.col_start : node.col_stop,
            ] = label
        return labels


@dataclass(frozen=True, slots=True, init=False)
class SubtreePartitionIndex:
    """Exact bounded count/rank index for canonical subtree frontiers.

    The index stores dynamic-programming counts for every tree node through
    ``max_k``.  Counts are arbitrary-precision Python integers.  Ranking and
    unranking follow exactly the recursive ordering used by
    :func:`enumerate_frontiers`: an unsplit node comes first, followed by the
    Cartesian product of the first child's frontiers and the second child's
    frontiers.  Exact-``K`` filtering preserves that underlying order.

    This class deliberately indexes frontiers below any node, not only full
    root frontiers.  A frontier returned for a non-root node therefore covers
    that subtree exactly but is not, by itself, a complete frontier of
    ``tree``.

    Args:
        tree: Canonical tree whose subtree partitions should be indexed.
        max_k: Largest subtree frontier size to index.

    Raises:
        TypeError: If ``tree`` or ``max_k`` has the wrong type.
        ValueError: If ``max_k`` lies outside the tree's supported range.

    Attributes:
        tree: Indexed canonical tree.
        max_k: Largest indexed frontier size.
    """

    tree: CanonicalDyadicTree
    max_k: int
    _counts: tuple[tuple[int, ...], ...]

    def __init__(self, tree: CanonicalDyadicTree, max_k: int) -> None:
        """Build exact bounded counts for every canonical subtree.

        Args:
            tree: Canonical tree whose subtree partitions should be indexed.
            max_k: Largest subtree frontier size to index.

        Raises:
            TypeError: If ``tree`` or ``max_k`` has the wrong type.
            ValueError: If ``max_k`` lies outside the tree's supported range.
        """
        _require_tree(tree)
        normalized_max_k = _validate_k(max_k, tree)
        if normalized_max_k is None:  # pragma: no cover - guarded by the signature
            raise TypeError("max_k must be an integer.")

        counts_by_node: list[tuple[int, ...] | None] = [None] * len(tree.nodes)
        for node in reversed(tree.nodes):
            capacity = min(normalized_max_k, node.area)
            counts = [0] * (capacity + 1)
            counts[1] = 1
            if node.child_ids:
                first_counts = counts_by_node[node.child_ids[0]]
                second_counts = counts_by_node[node.child_ids[1]]
                assert first_counts is not None
                assert second_counts is not None
                for first_k in range(1, len(first_counts)):
                    largest_second = min(len(second_counts) - 1, capacity - first_k)
                    for second_k in range(1, largest_second + 1):
                        counts[first_k + second_k] += first_counts[first_k] * second_counts[second_k]
            counts_by_node[node.node_id] = tuple(counts)

        object.__setattr__(self, "tree", tree)
        object.__setattr__(self, "max_k", normalized_max_k)
        object.__setattr__(
            self,
            "_counts",
            tuple(counts for counts in counts_by_node if counts is not None),
        )

    def counts_by_k(self, node_id: NodeId) -> tuple[int, ...]:
        """Return indexed frontier counts below one node.

        The returned tuple is indexed directly by ``K`` and has capacity
        ``min(max_k, node area)``.  Index zero is always zero.

        Args:
            node_id: Root node of the requested subtree.

        Returns:
            Immutable tuple of arbitrary-precision Python counts.

        Raises:
            KeyError: If ``node_id`` is not in :attr:`tree`.
        """
        node_index = self.tree._node_index(node_id)
        return self._counts[node_index]

    def count(self, node_id: NodeId, k: int) -> int:
        """Return the number of exact size-``k`` frontiers below one node.

        Args:
            node_id: Root node of the requested subtree.
            k: Exact number of active subtree regions.

        Returns:
            Arbitrary-precision Python count.

        Raises:
            KeyError: If ``node_id`` is not in :attr:`tree`.
            TypeError: If ``k`` is Boolean or not integer-like.
            ValueError: If ``k`` is not indexed or exceeds the subtree area.
        """
        normalized_k = self._validate_subtree_k(node_id, k)
        return self._counts[self.tree._node_index(node_id)][normalized_k]

    def rank(
        self,
        node_id: NodeId,
        k: int,
        frontier: DyadicFrontier,
    ) -> int:
        """Return the zero-based exact-``K`` rank of a subtree frontier.

        Args:
            node_id: Root node of the indexed subtree.
            k: Required number of active subtree regions.
            frontier: Exact frontier covering ``node_id``'s subtree.

        Returns:
            Zero-based rank in the order obtained by filtering the recursive
            exhaustive enumeration to size ``k``.

        Raises:
            KeyError: If ``node_id`` is not in :attr:`tree`.
            TypeError: If ``k`` or ``frontier`` has the wrong type.
            ValueError: If ``k`` is unsupported, ``frontier`` does not exactly
                cover the subtree, or its size is not ``k``.
        """
        normalized_k = self._validate_subtree_k(node_id, k)
        _validate_subtree_frontier(self.tree, node_id, frontier)
        if len(frontier) != normalized_k:
            raise ValueError(f"frontier has {len(frontier)} nodes but k is {normalized_k}.")

        weights = [0] * (self.max_k + 1)
        weights[normalized_k] = 1
        rank_value, frontier_k = self._weighted_rank(
            node_id,
            frozenset(frontier.node_ids),
            tuple(weights),
        )
        if frontier_k != normalized_k:  # pragma: no cover - checked above
            raise RuntimeError("Internal subtree rank size mismatch.")
        return rank_value

    def unrank(
        self,
        node_id: NodeId,
        k: int,
        rank: int,
    ) -> DyadicFrontier:
        """Return the exact size-``k`` subtree frontier at ``rank``.

        This method descends through dynamic-programming count blocks and
        never materializes all frontiers.

        Args:
            node_id: Root node of the indexed subtree.
            k: Required number of active subtree regions.
            rank: Zero-based rank in recursive enumeration order.

        Returns:
            Exact frontier covering ``node_id``'s subtree.

        Raises:
            KeyError: If ``node_id`` is not in :attr:`tree`.
            TypeError: If ``k`` or ``rank`` is Boolean or not integer-like.
            ValueError: If ``k`` is unsupported or ``rank`` is outside the
                exact-``K`` frontier range.
        """
        normalized_k = self._validate_subtree_k(node_id, k)
        normalized_rank = _normalize_rank(rank)
        count = self.count(node_id, normalized_k)
        if normalized_rank < 0 or normalized_rank >= count:
            raise ValueError(
                f"rank must lie between 0 and {count - 1} for node {index(node_id)} with k={normalized_k}."
            )

        weights = [0] * (self.max_k + 1)
        weights[normalized_k] = 1
        node_ids, frontier_k, residual = self._weighted_unrank(
            node_id,
            tuple(weights),
            normalized_rank,
        )
        if frontier_k != normalized_k or residual != 0:  # pragma: no cover
            raise RuntimeError("Internal subtree unrank mismatch.")
        return DyadicFrontier(node_ids)

    def _validate_subtree_k(self, node_id: NodeId, k: int) -> int:
        """Normalize an exact subtree frontier size.

        Args:
            node_id: Candidate subtree root.
            k: Candidate exact frontier size.

        Returns:
            Valid built-in integer size.

        Raises:
            KeyError: If ``node_id`` is not in :attr:`tree`.
            TypeError: If ``k`` is Boolean or not integer-like.
            ValueError: If ``k`` exceeds the indexed or subtree capacity.
        """
        node = self.tree.node(node_id)
        if isinstance(k, bool):
            raise TypeError("k must be an integer.")
        try:
            normalized_k = index(k)
        except TypeError as error:
            raise TypeError("k must be an integer.") from error
        maximum = min(self.max_k, node.area)
        if normalized_k < 1 or normalized_k > maximum:
            raise ValueError(f"k must lie between 1 and {maximum} for subtree node {node.node_id}.")
        return normalized_k

    def _weighted_rank(
        self,
        node_id: NodeId,
        active: frozenset[NodeId],
        weights: tuple[int, ...],
    ) -> tuple[int, int]:
        """Return weighted prefix rank and size for one known-valid frontier."""
        if node_id in active:
            return 0, 1

        node = self.tree.nodes[node_id]
        first_id, second_id = node.child_ids
        first_weights = self._first_child_weights(first_id, second_id, weights)
        first_rank, first_k = self._weighted_rank(first_id, active, first_weights)
        second_weights = self._shift_weights(weights, first_k)
        second_rank, second_k = self._weighted_rank(second_id, active, second_weights)
        return _weight_at(weights, 1) + first_rank + second_rank, first_k + second_k

    def _weighted_unrank(
        self,
        node_id: NodeId,
        weights: tuple[int, ...],
        rank: int,
    ) -> tuple[tuple[NodeId, ...], int, int]:
        """Select one weighted frontier and retain its within-block residual."""
        unsplit_weight = _weight_at(weights, 1)
        if rank < unsplit_weight:
            return (node_id,), 1, rank

        rank -= unsplit_weight
        first_id, second_id = self.tree.nodes[node_id].child_ids
        first_weights = self._first_child_weights(first_id, second_id, weights)
        first_ids, first_k, residual = self._weighted_unrank(
            first_id,
            first_weights,
            rank,
        )
        second_weights = self._shift_weights(weights, first_k)
        second_ids, second_k, residual = self._weighted_unrank(
            second_id,
            second_weights,
            residual,
        )
        return (*first_ids, *second_ids), first_k + second_k, residual

    def _first_child_weights(
        self,
        first_id: NodeId,
        second_id: NodeId,
        weights: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Return first-child block weights after summing second frontiers."""
        first_counts = self._counts[first_id]
        second_counts = self._counts[second_id]
        result = [0] * len(first_counts)
        for first_k in range(1, len(first_counts)):
            result[first_k] = sum(
                second_counts[second_k] * _weight_at(weights, first_k + second_k)
                for second_k in range(1, len(second_counts))
            )
        return tuple(result)

    def _shift_weights(
        self,
        weights: tuple[int, ...],
        first_k: int,
    ) -> tuple[int, ...]:
        """Return second-child weights conditional on a first-child size."""
        return tuple(_weight_at(weights, first_k + second_k) for second_k in range(self.max_k + 1))


def enumerate_frontiers(
    tree: CanonicalDyadicTree,
    *,
    k: int | None = None,
) -> tuple[DyadicFrontier, ...]:
    """Exhaustively enumerate exact frontiers of one tiny canonical tree.

    The unsplit root is first.  Split frontiers then follow the recursive
    Cartesian-product order of the first and second child subtrees.

    Args:
        tree: Canonical tree to enumerate.
        k: Optional exact number of active nodes to retain.

    Returns:
        Exact frontiers in deterministic recursive order.

    Raises:
        TypeError: If ``tree`` or ``k`` has the wrong type.
        ValueError: If ``k`` is outside ``[1, number of grid cells]``.

    Notes:
        The output grows exponentially.  This function is intended only for
        tiny tests and exact-oracle calculations.
    """
    _require_tree(tree)
    normalized_k = _validate_k(k, tree)
    node_sets = _enumerate_subtree(tree, tree.root_id)
    return tuple(
        DyadicFrontier(node_ids)
        for node_ids in node_sets
        if normalized_k is None or len(node_ids) == normalized_k
    )


def partition_counts_by_k(
    tree: CanonicalDyadicTree,
    *,
    max_k: int | None = None,
) -> tuple[int, ...]:
    """Count exact frontiers by active-region count using dynamic programming.

    Args:
        tree: Canonical tree whose partitions should be counted.
        max_k: Optional largest count to compute.  It must not exceed the
            number of terminal grid cells.

    Returns:
        Tuple indexed directly by ``K``.  Index zero is always zero and each
        remaining value is an arbitrary-precision Python :class:`int`.

    Raises:
        TypeError: If ``tree`` or ``max_k`` has the wrong type.
        ValueError: If ``max_k`` is outside the supported range.
    """
    _require_tree(tree)
    normalized_max = _validate_k(max_k, tree)
    if normalized_max is None:
        normalized_max = len(tree.leaf_ids)

    return SubtreePartitionIndex(tree, normalized_max).counts_by_k(tree.root_id)


def _require_tree(tree: CanonicalDyadicTree) -> None:
    """Require one canonical tree instance.

    Args:
        tree: Candidate tree value.

    Raises:
        TypeError: If ``tree`` is not a :class:`CanonicalDyadicTree`.
    """
    if not isinstance(tree, CanonicalDyadicTree):
        raise TypeError("tree must be a CanonicalDyadicTree.")


def _require_frontier(frontier: DyadicFrontier) -> None:
    """Require one dyadic frontier instance.

    Args:
        frontier: Candidate frontier value.

    Raises:
        TypeError: If ``frontier`` is not a :class:`DyadicFrontier`.
    """
    if not isinstance(frontier, DyadicFrontier):
        raise TypeError("frontier must be a DyadicFrontier.")


def _validate_subtree_frontier(
    tree: CanonicalDyadicTree,
    node_id: NodeId,
    frontier: DyadicFrontier,
) -> None:
    """Require a frontier to cover one canonical subtree exactly.

    Args:
        tree: Canonical tree supplying topology.
        node_id: Root of the subtree that must be covered.
        frontier: Candidate exact subtree frontier.

    Raises:
        KeyError: If ``node_id`` is not in ``tree``.
        TypeError: If ``frontier`` has the wrong type.
        ValueError: If the frontier is empty, contains an unknown node, or
            does not cover the requested subtree exactly without overlap.
    """
    tree.node(node_id)
    _require_frontier(frontier)
    if not frontier.node_ids:
        raise ValueError("A subtree frontier must contain at least one node.")
    for active_id in frontier.node_ids:
        try:
            tree.node(active_id)
        except KeyError as error:
            raise ValueError(f"Frontier node ID {active_id!r} is not in the tree.") from error

    active = frozenset(frontier.node_ids)
    reached: set[NodeId] = set()
    coverage_gap = False
    pending = [index(node_id)]
    while pending:
        current_id = pending.pop()
        if current_id in active:
            reached.add(current_id)
            continue
        child_ids = tree.nodes[current_id].child_ids
        if not child_ids:
            coverage_gap = True
            continue
        pending.extend(child_ids)
    if reached != active or coverage_gap:
        raise ValueError(f"frontier must exactly cover subtree node {index(node_id)} without overlap.")


def _normalize_rank(rank: int) -> int:
    """Normalize one candidate zero-based rank.

    Args:
        rank: Candidate integer-like rank.

    Returns:
        Built-in integer rank.

    Raises:
        TypeError: If ``rank`` is Boolean or not integer-like.
    """
    if isinstance(rank, bool):
        raise TypeError("rank must be an integer.")
    try:
        return index(rank)
    except TypeError as error:
        raise TypeError("rank must be an integer.") from error


def _weight_at(weights: tuple[int, ...], k: int) -> int:
    """Return one size weight, treating out-of-range sizes as zero."""
    if k < 0 or k >= len(weights):
        return 0
    return weights[k]


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Validate and normalize a positive two-dimensional grid shape.

    Args:
        shape: Candidate ``(rows, columns)`` shape.

    Returns:
        Positive extents as built-in integers.

    Raises:
        TypeError: If an extent is Boolean or not integer-like.
        ValueError: If the shape does not contain exactly two positive
            extents.
    """
    try:
        extents = tuple(shape)
    except TypeError as error:
        raise ValueError("Dyadic tree shape must contain exactly two extents.") from error
    if len(extents) != 2:
        raise ValueError("Dyadic tree shape must contain exactly two extents.")

    normalized: list[int] = []
    for extent in extents:
        if isinstance(extent, bool):
            raise TypeError("Dyadic tree extents must be integers.")
        try:
            value = index(extent)
        except TypeError as error:
            raise TypeError("Dyadic tree extents must be integers.") from error
        if value <= 0:
            raise ValueError("Dyadic tree extents must be positive.")
        normalized.append(value)
    return normalized[0], normalized[1]


def _split_bounds(
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """Return canonical child bounds, or ``None`` for one cell.

    The longer axis is split, rows win ties, and the second child receives an
    odd extra row or column.

    Args:
        row_start: Inclusive first row.
        row_stop: Exclusive final row.
        col_start: Inclusive first column.
        col_stop: Exclusive final column.

    Returns:
        Ordered half-open bounds for the first and second children, or
        ``None`` when the bounds describe one grid cell.
    """
    height = row_stop - row_start
    width = col_stop - col_start
    if height == 1 and width == 1:
        return None
    if height >= width:
        midpoint = row_start + height // 2
        return (
            (row_start, midpoint, col_start, col_stop),
            (midpoint, row_stop, col_start, col_stop),
        )
    midpoint = col_start + width // 2
    return (
        (row_start, row_stop, col_start, midpoint),
        (row_start, row_stop, midpoint, col_stop),
    )


def _enumerate_subtree(
    tree: CanonicalDyadicTree,
    node_id: NodeId,
) -> tuple[tuple[NodeId, ...], ...]:
    """Return all recursively ordered exact frontiers below one node.

    Args:
        tree: Canonical tree being enumerated.
        node_id: Root of the subtree to enumerate.

    Returns:
        Exact subtree frontiers with the unsplit node first, followed by child
        Cartesian products in deterministic recursive order.
    """
    child_ids = tree.children(node_id)
    if not child_ids:
        return ((node_id,),)
    first_frontiers = _enumerate_subtree(tree, child_ids[0])
    second_frontiers = _enumerate_subtree(tree, child_ids[1])
    return (
        (node_id,),
        *(tuple(sorted((*first, *second))) for first in first_frontiers for second in second_frontiers),
    )


def _validate_k(k: int | None, tree: CanonicalDyadicTree) -> int | None:
    """Normalize and bound an optional active-region count.

    Args:
        k: Candidate exact count or truncation limit.
        tree: Tree supplying the maximum terminal-cell count.

    Returns:
        Normalized Python integer, or ``None`` when no limit was requested.

    Raises:
        TypeError: If ``k`` is Boolean or not integer-like.
        ValueError: If ``k`` is outside the supported positive range.
    """
    if k is None:
        return None
    if isinstance(k, bool):
        raise TypeError("k must be an integer.")
    try:
        normalized = index(k)
    except TypeError as error:
        raise TypeError("k must be an integer.") from error
    if normalized < 1 or normalized > len(tree.leaf_ids):
        raise ValueError(f"k must lie between 1 and {len(tree.leaf_ids)}.")
    return normalized


__all__ = [
    "CanonicalDyadicTree",
    "DyadicFrontier",
    "DyadicNode",
    "NodeId",
    "SubtreePartitionIndex",
    "enumerate_frontiers",
    "partition_counts_by_k",
]
