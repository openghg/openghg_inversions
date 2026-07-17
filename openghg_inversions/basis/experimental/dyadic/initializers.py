"""Construct deterministic and seeded initial dyadic partitions.

The initializers in this module grow the canonical binary tree through
``PartitionState.split``.  They therefore return valid active frontiers without
depending on dense label maps or mutating a caller-owned state.  Greedy growth
uses :mod:`heapq` with node identifiers as its deterministic tie-breaker;
random growth is reproducible when supplied the same NumPy generator state.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .state import PartitionState
from .tree import DyadicTree, Tile


@dataclass(frozen=True, slots=True)
class InitializationResult:
    """A tree partition and the ordered splits used to construct it.

    Attributes:
        state: Final immutable partition state.
        split_history: Node identifiers in the order in which they were split.
    """

    state: PartitionState
    split_history: tuple[int, ...]


def greedy_partition(
    tree: DyadicTree,
    target_regions: int,
    split_gain: Callable[[int], float],
) -> InitializationResult:
    """Grow an exact-size partition by repeatedly taking the best split.

    Candidate nodes are ordered by decreasing ``split_gain``.  Equal gains are
    resolved by increasing node identifier, making the result independent of
    set iteration order.

    Args:
        tree: Canonical dyadic tree to partition.
        target_regions: Required number of active regions, at least one.
        split_gain: Callable returning the priority gain for a splittable node
            identifier.

    Returns:
        The final partition and its split history.

    Raises:
        ValueError: If ``target_regions`` is less than one or exceeds the
            number of leaves available from ``tree``.
    """
    _validate_target_regions(target_regions)
    state = PartitionState.root(tree)
    split_history: list[int] = []
    candidates: list[tuple[float, int]] = []
    _push_greedy_candidate(candidates, tree, tree.root_id, split_gain)

    while len(state.active) < target_regions:
        if not candidates:
            raise _impossible_target_error(target_regions)

        _, node_id = heapq.heappop(candidates)
        state = state.split(tree, node_id)
        split_history.append(node_id)
        children = tree.children(node_id)
        if not children:  # pragma: no cover - guarded when queued.
            raise RuntimeError(f"Queued node {node_id} cannot be split.")
        for child_id in children:
            _push_greedy_candidate(candidates, tree, child_id, split_gain)

    state.validate(tree)
    return InitializationResult(state=state, split_history=tuple(split_history))


def random_partition(
    tree: DyadicTree,
    target_regions: int,
    rng: np.random.Generator,
) -> InitializationResult:
    """Grow an exact-size random partition using a caller-supplied generator.

    At each step this function samples uniformly from the currently splittable
    active nodes, sorted by node identifier before indexing.  The input tree and
    all intermediate partition states remain unmodified.

    Args:
        tree: Canonical dyadic tree to partition.
        target_regions: Required number of active regions, at least one.
        rng: NumPy random generator controlling every random choice.

    Returns:
        The final partition and its split history.

    Raises:
        ValueError: If ``target_regions`` is less than one or exceeds the
            number of leaves available from ``tree``.
    """
    _validate_target_regions(target_regions)
    state = PartitionState.root(tree)
    split_history: list[int] = []

    while len(state.active) < target_regions:
        candidates = tuple(node_id for node_id in state.ordered_active() if tree.children(node_id))
        if not candidates:
            raise _impossible_target_error(target_regions)

        node_id = candidates[int(rng.integers(len(candidates)))]
        state = state.split(tree, node_id)
        split_history.append(node_id)

    state.validate(tree)
    return InitializationResult(state=state, split_history=tuple(split_history))


def threshold_partition(
    tree: DyadicTree,
    tile_score: Callable[[Tile], float],
    threshold: float,
) -> InitializationResult:
    """Split each eligible tile whose score is strictly above a threshold.

    Nodes are visited depth first, with each tree-provided first child visited
    before its sibling.  ``tile_score`` receives the tile object returned by
    ``tree.tile(node_id)``; leaves above the threshold remain active because
    they cannot be split.

    Args:
        tree: Canonical dyadic tree to partition.
        tile_score: Callable assigning a scalar score to a tree tile.
        threshold: Split threshold, applied using ``score > threshold``.

    Returns:
        The final partition and its deterministic split history.
    """
    state = PartitionState.root(tree)
    split_history: list[int] = []
    pending = [tree.root_id]

    while pending:
        node_id = pending.pop()
        children = tree.children(node_id)
        if not children or tile_score(tree.tile(node_id)) <= threshold:
            continue

        state = state.split(tree, node_id)
        split_history.append(node_id)
        first_child, second_child = children
        pending.append(second_child)
        pending.append(first_child)

    state.validate(tree)
    return InitializationResult(state=state, split_history=tuple(split_history))


def _push_greedy_candidate(
    candidates: list[tuple[float, int]],
    tree: DyadicTree,
    node_id: int,
    split_gain: Callable[[int], float],
) -> None:
    """Push a splittable node onto a max-gain heap with stable tie ordering."""
    if tree.children(node_id):
        heapq.heappush(candidates, (-float(split_gain(node_id)), node_id))


def _validate_target_regions(target_regions: int) -> None:
    """Reject region targets outside the non-empty partition domain."""
    if target_regions < 1:
        raise ValueError("target_regions must be at least 1.")


def _impossible_target_error(target_regions: int) -> ValueError:
    """Build the common error for targets larger than a tree can supply."""
    return ValueError(f"Cannot construct {target_regions} regions from the available tree leaves.")
