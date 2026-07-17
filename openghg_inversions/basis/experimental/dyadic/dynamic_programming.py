"""Exact dynamic programming for additive scores on canonical dyadic trees.

For a score attached to every tree tile, the score of a partition is the sum
of the scores of its active tiles.  This module computes the globally maximal
valid frontier for every region count from one through a caller-supplied
limit.  At each node the recurrence compares leaving that node as one region
with splitting it and distributing the requested region count across its two
ordered children.

The implementation uses only Python and NumPy.  Equal floating-point scores
are resolved by the lexicographically smallest tuple of active preorder node
IDs, so results are reproducible and independent of dictionary or set order.

The current recurrence is intended as an experimental reference at region
counts in the hundreds.  It stores complete active-node tuples at each
subproblem and considers all feasible left/right count pairs, so its worst-case
work is quadratic in the requested region limit per internal node and tuple
storage adds further overhead.  A larger production implementation should use
compact scores plus backpointers rather than copying complete frontiers.
"""

from __future__ import annotations

from operator import index
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from .state import PartitionState
from .tree import DyadicTree, NodeId


class AdditivePartitionSolution(NamedTuple):
    """Globally optimal additive-score partition for one region count.

    Attributes:
        state: Valid immutable frontier with the requested number of regions.
        score: Sum of tile scores over ``state.active``.
    """

    state: PartitionState
    score: float


class _Candidate(NamedTuple):
    """Store one optimal subtree score and its ordered active node IDs."""

    score: float
    active: tuple[NodeId, ...]


def additive_partition_frontier(
    tree: DyadicTree,
    tile_scores: npt.ArrayLike,
    max_regions: int,
) -> dict[int, AdditivePartitionSolution]:
    """Compute exact optimal partitions for all counts through ``max_regions``.

    The returned dictionary is keyed by region count and contains every count
    from one through ``max_regions``.  It is computed in one bottom-up pass;
    callers can therefore inspect the complete score-versus-count frontier
    without running the recurrence separately for each count.

    Args:
        tree: Complete canonical dyadic tree defining valid partitions.
        tile_scores: One finite real additive score per tree node, indexed by
            the node IDs in ``tree.nodes``.
        max_regions: Largest region count to include.  It must be between one
            and the number of cell leaves, inclusive.

    Returns:
        A dictionary mapping each region count to its globally optimal state
        and score.  Equal-score alternatives use the lexicographically
        smallest ordered active-node tuple.

    Raises:
        TypeError: If ``tree`` is not a :class:`DyadicTree` or ``max_regions``
            is not an integer.
        ValueError: If ``tree`` is not a complete canonical tree, tile scores
            are not a finite real one-dimensional value per node, or
            ``max_regions`` is outside the valid range.

    Notes:
        The recurrence examines all feasible child-count pairs through
        ``max_regions`` and stores full active-node tuples for deterministic
        tie-breaking.  This is practical for the experimental grids and region
        counts used here, but it is not a memory-optimized large-scale solver.
    """
    _validate_tree(tree)
    scores = _validate_tile_scores(tile_scores, node_count=len(tree.nodes))
    region_limit = _validate_region_count(max_regions, tree, name="max_regions")
    return _build_frontier(tree, scores, region_limit)


def optimal_additive_partition(
    tree: DyadicTree,
    tile_scores: npt.ArrayLike,
    target_regions: int,
) -> AdditivePartitionSolution:
    """Return the exact best additive-score partition with a requested size.

    Args:
        tree: Complete canonical dyadic tree defining valid partitions.
        tile_scores: One finite real additive score per tree node, indexed by
            the node IDs in ``tree.nodes``.
        target_regions: Exact required number of active regions.

    Returns:
        The globally optimal valid partition state and its additive score.
        The result can also be unpacked as ``state, score``.

    Raises:
        TypeError: If ``tree`` is not a :class:`DyadicTree` or
            ``target_regions`` is not an integer.
        ValueError: If ``tree`` is not a complete canonical tree, tile scores
            are not a finite real one-dimensional value per node, or the
            requested count cannot be represented by the tree.
    """
    _validate_tree(tree)
    scores = _validate_tile_scores(tile_scores, node_count=len(tree.nodes))
    region_count = _validate_region_count(target_regions, tree, name="target_regions")
    return _build_frontier(tree, scores, region_count)[region_count]


def _build_frontier(
    tree: DyadicTree,
    tile_scores: np.ndarray,
    max_regions: int,
) -> dict[int, AdditivePartitionSolution]:
    """Run the validated bottom-up recurrence and construct root solutions.

    Args:
        tree: Validated complete canonical dyadic tree.
        tile_scores: Validated finite score vector indexed by node ID.
        max_regions: Validated largest root-frontier size to construct.

    Returns:
        Optimal root solutions keyed by every count through ``max_regions``.

    Raises:
        RuntimeError: If the validated tree unexpectedly lacks a partition for
            a requested count.
    """
    subtree_frontiers: dict[NodeId, tuple[_Candidate | None, ...]] = {}

    for tile in reversed(tree.nodes):
        node_id = tile.node_id
        capacity = min(max_regions, tile.area)
        candidates: list[_Candidate | None] = [None] * (capacity + 1)
        candidates[1] = _Candidate(float(tile_scores[node_id]), (node_id,))

        children = tree.children(node_id)
        if children:
            left_id, right_id = children
            left_frontier = subtree_frontiers[left_id]
            right_frontier = subtree_frontiers[right_id]
            for left_count in range(1, len(left_frontier)):
                left = left_frontier[left_count]
                if left is None:  # pragma: no cover - canonical subtrees have every count.
                    continue
                largest_right_count = min(len(right_frontier) - 1, capacity - left_count)
                for right_count in range(1, largest_right_count + 1):
                    right = right_frontier[right_count]
                    if right is None:  # pragma: no cover - canonical subtrees have every count.
                        continue
                    region_count = left_count + right_count
                    candidate = _Candidate(left.score + right.score, left.active + right.active)
                    if _is_better(candidate, candidates[region_count]):
                        candidates[region_count] = candidate

        subtree_frontiers[node_id] = tuple(candidates)

    root_frontier = subtree_frontiers[tree.root_id]
    frontier: dict[int, AdditivePartitionSolution] = {}
    for region_count in range(1, max_regions + 1):
        root_candidate = root_frontier[region_count]
        if root_candidate is None:  # pragma: no cover - validated full binary trees admit every count.
            raise RuntimeError(f"Canonical tree has no partition with {region_count} regions.")
        state = PartitionState(active=frozenset(root_candidate.active))
        state.validate(tree)
        frontier[region_count] = AdditivePartitionSolution(state=state, score=root_candidate.score)
    return frontier


def _is_better(candidate: _Candidate, incumbent: _Candidate | None) -> bool:
    """Return whether a candidate wins by score and then stable node ordering."""
    return (
        incumbent is None
        or candidate.score > incumbent.score
        or (candidate.score == incumbent.score and candidate.active < incumbent.active)
    )


def _validate_tree(tree: DyadicTree) -> None:
    """Require the exact complete tree produced by ``DyadicTree.from_shape``.

    Args:
        tree: Candidate tree to compare with the canonical tree for its shape.

    Raises:
        TypeError: If ``tree`` is not a :class:`DyadicTree`.
        ValueError: If its shape is invalid or its stored topology and tiles
            do not match the canonical complete tree for that shape.
    """
    if not isinstance(tree, DyadicTree):
        raise TypeError("tree must be a DyadicTree.")
    try:
        canonical = DyadicTree.from_shape(tree.shape)
    except (TypeError, ValueError) as error:
        raise ValueError("tree must have a valid canonical two-dimensional shape.") from error
    if tree != canonical:
        raise ValueError("tree must be a complete canonical DyadicTree for its shape.")


def _validate_tile_scores(tile_scores: npt.ArrayLike, *, node_count: int) -> np.ndarray:
    """Convert tile scores to a finite real vector with one value per node.

    Args:
        tile_scores: Candidate additive node scores.
        node_count: Required length of the score vector.

    Returns:
        A floating-point NumPy vector with shape ``(node_count,)``.

    Raises:
        ValueError: If the input is complex, non-numeric, non-finite, or has
            any shape other than ``(node_count,)``.
    """
    scores = np.asarray(tile_scores)
    if np.iscomplexobj(scores):
        raise ValueError("tile_scores must be real-valued.")
    try:
        scores = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("tile_scores must be numeric.") from error
    if scores.shape != (node_count,):
        raise ValueError(f"tile_scores must have shape ({node_count},).")
    if not np.all(np.isfinite(scores)):
        raise ValueError("tile_scores must contain only finite values.")
    return scores


def _validate_region_count(region_count: int, tree: DyadicTree, *, name: str) -> int:
    """Normalize an integer region count and ensure the tree can supply it.

    Args:
        region_count: Candidate count to normalize through ``operator.index``.
        tree: Validated tree providing the maximum leaf count.
        name: Parameter name used in validation messages.

    Returns:
        The region count as a built-in integer.

    Raises:
        TypeError: If ``region_count`` is a Boolean or not integer-like.
        ValueError: If the count is less than one or exceeds the leaf count.
    """
    if isinstance(region_count, bool):
        raise TypeError(f"{name} must be an integer.")
    try:
        normalized = index(region_count)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer.") from error
    if normalized < 1:
        raise ValueError(f"{name} must be at least 1.")
    if normalized > len(tree.leaf_ids):
        raise ValueError(f"{name} cannot exceed the tree's {len(tree.leaf_ids)} leaves.")
    return normalized


__all__ = [
    "AdditivePartitionSolution",
    "additive_partition_frontier",
    "optimal_additive_partition",
]
