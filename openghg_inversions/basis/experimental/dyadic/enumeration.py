"""Exhaustive enumeration of canonical dyadic partition frontiers.

This module provides a small exact oracle for tests and experiments.  At each
tree node, a valid frontier either keeps that node active or combines one
recursively enumerated frontier from each ordered child.  The resulting order
is deterministic: the unsplit frontier appears first, followed by child
products in recursive tree order.

The number of frontiers grows exponentially with the number of grid cells, and
every returned state is materialized.  This implementation is therefore
intended only for tiny trees where exhaustive enumeration is useful as a
correctness oracle, not for production partition searches.
"""

from __future__ import annotations

from operator import index

from .state import PartitionState
from .tree import DyadicTree, NodeId


def enumerate_partitions(
    tree: DyadicTree,
    *,
    region_count: int | None = None,
) -> tuple[PartitionState, ...]:
    """Enumerate every valid partition frontier of a tiny dyadic tree.

    Frontiers are ordered recursively with the unsplit root first.  Split
    frontiers then follow the Cartesian-product order of the recursively
    enumerated first and second child frontiers.  Supplying ``region_count``
    retains only states with exactly that many active regions while preserving
    their relative order.

    Args:
        tree: Complete canonical dyadic tree to enumerate.
        region_count: Optional exact number of active regions.  It must be
            between one and the number of grid-cell leaves, inclusive.

    Returns:
        All valid partition states, or only those with the requested region
        count, in deterministic recursive order.

    Raises:
        TypeError: If ``tree`` is not a :class:`DyadicTree`, or if
            ``region_count`` is a Boolean or is not integer-like.
        ValueError: If ``tree`` is not complete and canonical, or if
            ``region_count`` is outside the valid range.

    Notes:
        Exhaustive enumeration has exponential time and output-space
        complexity in the number of grid cells.  Use this function only for
        tiny exact-oracle problems.
    """
    _validate_tree(tree)
    normalized_count = _validate_region_count(region_count, tree)
    frontiers = _enumerate_subtree_frontiers(tree, tree.root_id)
    return tuple(
        PartitionState(active=active)
        for active in frontiers
        if normalized_count is None or len(active) == normalized_count
    )


def count_partitions_by_region(
    tree: DyadicTree,
    *,
    max_regions: int | None = None,
) -> dict[int, int]:
    """Count valid frontiers by region count without enumerating them.

    The bottom-up recurrence counts the unsplit frontier at each node and all
    pairs of child frontiers when that node is split.  Counts use arbitrary
    precision Python integers, so the function remains exact when the total
    number of partitions is too large to materialize.

    Args:
        tree: Complete canonical dyadic tree.
        max_regions: Optional largest region count to return.  Defaults to the
            number of grid-cell leaves.

    Returns:
        Exact positive partition count for every region count from one through
        ``max_regions``.

    Raises:
        TypeError: If ``tree`` or ``max_regions`` has the wrong type.
        ValueError: If the tree is non-canonical or ``max_regions`` is outside
            the valid range.

    Notes:
        The recurrence requires quadratic work in ``max_regions`` per internal
        node but does not construct any :class:`PartitionState` values.
    """
    _validate_tree(tree)
    region_limit = _validate_region_count(max_regions, tree)
    if region_limit is None:
        region_limit = len(tree.leaf_ids)

    subtree_counts: dict[NodeId, tuple[int, ...]] = {}
    for tile in reversed(tree.nodes):
        capacity = min(region_limit, tile.area)
        counts = [0] * (capacity + 1)
        counts[1] = 1
        children = tree.children(tile.node_id)
        if children:
            left_counts = subtree_counts[children[0]]
            right_counts = subtree_counts[children[1]]
            for left_regions in range(1, len(left_counts)):
                largest_right = min(len(right_counts) - 1, capacity - left_regions)
                for right_regions in range(1, largest_right + 1):
                    counts[left_regions + right_regions] += (
                        left_counts[left_regions] * right_counts[right_regions]
                    )
        subtree_counts[tile.node_id] = tuple(counts)

    root_counts = subtree_counts[tree.root_id]
    return {region_count: root_counts[region_count] for region_count in range(1, region_limit + 1)}


def _enumerate_subtree_frontiers(
    tree: DyadicTree,
    node_id: NodeId,
) -> tuple[frozenset[NodeId], ...]:
    """Recursively enumerate valid frontiers rooted at one tree node.

    Args:
        tree: Validated complete canonical dyadic tree.
        node_id: Root of the subtree to enumerate.

    Returns:
        Subtree frontiers in deterministic recursive order, beginning with the
        unsplit node.
    """
    children = tree.children(node_id)
    if not children:
        return (frozenset({node_id}),)

    first_id, second_id = children
    split_frontiers = tuple(
        first | second
        for first in _enumerate_subtree_frontiers(tree, first_id)
        for second in _enumerate_subtree_frontiers(tree, second_id)
    )
    return (frozenset({node_id}), *split_frontiers)


def _validate_tree(tree: DyadicTree) -> None:
    """Require a complete canonical dyadic tree.

    Args:
        tree: Candidate tree to compare with the canonical tree for its shape.

    Raises:
        TypeError: If ``tree`` is not a :class:`DyadicTree`.
        ValueError: If its shape or stored topology is not canonical.
    """
    if not isinstance(tree, DyadicTree):
        raise TypeError("tree must be a DyadicTree.")
    try:
        canonical = DyadicTree.from_shape(tree.shape)
    except (TypeError, ValueError) as error:
        raise ValueError("tree must have a valid canonical two-dimensional shape.") from error
    if tree != canonical:
        raise ValueError("tree must be a complete canonical DyadicTree for its shape.")


def _validate_region_count(region_count: int | None, tree: DyadicTree) -> int | None:
    """Normalize and bound an optional exact region count.

    Args:
        region_count: Candidate count to normalize through ``operator.index``,
            or ``None`` to request every frontier.
        tree: Validated tree providing the maximum grid-cell leaf count.

    Returns:
        A built-in integer in the valid range, or ``None``.

    Raises:
        TypeError: If the count is a Boolean or not integer-like.
        ValueError: If the count is less than one or exceeds the number of
            grid-cell leaves.
    """
    if region_count is None:
        return None
    if isinstance(region_count, bool):
        raise TypeError("region_count must be an integer.")
    try:
        normalized = index(region_count)
    except TypeError as error:
        raise TypeError("region_count must be an integer.") from error
    if normalized < 1:
        raise ValueError("region_count must be at least 1.")
    if normalized > len(tree.leaf_ids):
        raise ValueError(f"region_count cannot exceed the tree's {len(tree.leaf_ids)} grid cells.")
    return normalized


__all__ = ["count_partitions_by_region", "enumerate_partitions"]
