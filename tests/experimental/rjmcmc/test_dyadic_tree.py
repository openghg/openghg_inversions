"""Tests for immutable canonical dyadic trees and exact frontiers."""

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from itertools import combinations

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.dyadic_tree import (
    CanonicalDyadicTree,
    DyadicFrontier,
    DyadicNode,
    SubtreePartitionIndex,
    enumerate_frontiers,
    partition_counts_by_k,
)


class _CountingNodes:
    """Count indexed topology reads while rejecting whole-tree iteration."""

    def __init__(self, nodes: tuple[DyadicNode, ...]) -> None:
        """Store the real node tuple behind an instrumented sequence."""
        self._nodes = nodes
        self.reads = 0

    def __len__(self) -> int:
        """Return the complete tree size without counting an indexed read."""
        return len(self._nodes)

    def __getitem__(self, node_id: int) -> DyadicNode:
        """Count and return one indexed node."""
        self.reads += 1
        return self._nodes[node_id]

    def __iter__(self) -> Iterator[DyadicNode]:
        """Reject iteration because candidate queries must remain frontier-local."""
        raise AssertionError("Candidate queries must not scan every tree node.")


def _enumerate_subtree_frontiers(
    tree: CanonicalDyadicTree,
    node_id: int,
) -> tuple[DyadicFrontier, ...]:
    """Build a tiny-tree reference in the public recursive ordering."""
    child_ids = tree.children(node_id)
    if not child_ids:
        return (DyadicFrontier((node_id,)),)
    first_frontiers = _enumerate_subtree_frontiers(tree, child_ids[0])
    second_frontiers = _enumerate_subtree_frontiers(tree, child_ids[1])
    return (
        DyadicFrontier((node_id,)),
        *(
            DyadicFrontier((*first.node_ids, *second.node_ids))
            for first in first_frontiers
            for second in second_frontiers
        ),
    )


def _reference_frontier_error(
    tree: CanonicalDyadicTree,
    frontier: DyadicFrontier,
) -> str | None:
    """Return the error category produced by the original validator."""
    if not frontier.node_ids:
        return "empty"
    active = frozenset(frontier.node_ids)
    for node_id in frontier.node_ids:
        if node_id < 0 or node_id >= len(tree.nodes):
            return "unknown"
        parent_id = tree.nodes[node_id].parent_id
        while parent_id is not None:
            if parent_id in active:
                return "ancestor"
            parent_id = tree.nodes[parent_id].parent_id

    pending = [tree.root_id]
    while pending:
        node_id = pending.pop()
        if node_id in active:
            continue
        child_ids = tree.nodes[node_id].child_ids
        if not child_ids:
            return "coverage"
        pending.extend(child_ids)
    return None


def _validation_error(
    tree: CanonicalDyadicTree,
    frontier: DyadicFrontier,
) -> str | None:
    """Return a stable category for the optimized validator's result."""
    try:
        frontier.validate(tree)
    except ValueError as error:
        message = str(error)
        if "at least one" in message:
            return "empty"
        if "not in the tree" in message:
            return "unknown"
        if "ancestor" in message:
            return "ancestor"
        if "exactly cover" in message:
            return "coverage"
        raise
    return None


def test_rectangular_geometry_uses_longer_axis_and_odd_second_child() -> None:
    """A 3x5 root splits columns and gives the second child the extra column."""
    tree = CanonicalDyadicTree.from_shape((3, 5))
    first_id, second_id = tree.children(tree.root_id)
    first = tree.node(first_id)
    second = tree.node(second_id)

    assert first.bounds == (0, 3, 0, 2)
    assert second.bounds == (0, 3, 2, 5)
    assert first.area + second.area == 15
    assert len(tree.nodes) == 29
    assert tuple(node.node_id for node in tree.nodes) == tuple(range(29))
    assert first.cell_indices == (0, 1, 5, 6, 10, 11)
    assert second.cell_indices == (2, 3, 4, 7, 8, 9, 12, 13, 14)


def test_square_geometry_splits_rows_on_ties_in_preorder() -> None:
    """A 2x2 square splits rows before recursively splitting columns."""
    tree = CanonicalDyadicTree.from_shape((2, 2))

    assert tuple(node.bounds for node in tree.nodes) == (
        (0, 2, 0, 2),
        (0, 1, 0, 2),
        (0, 1, 0, 1),
        (0, 1, 1, 2),
        (1, 2, 0, 2),
        (1, 2, 0, 1),
        (1, 2, 1, 2),
    )
    assert tree.children(0) == (1, 4)
    assert tree.children(1) == (2, 3)
    assert tree.children(4) == (5, 6)
    assert tree.leaf_ids == (2, 3, 5, 6)


def test_tree_nodes_and_frontiers_are_immutable_and_canonical() -> None:
    """Frozen values reject assignment and frontier IDs sort deterministically."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    frontier = DyadicFrontier((6, 3, 2, 5))

    assert frontier.node_ids == (2, 3, 5, 6)
    frontier.validate(tree)
    with pytest.raises(FrozenInstanceError):
        tree.shape = (1, 1)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        frontier.node_ids = (0,)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("frontier", "message"),
    [
        (DyadicFrontier(()), "at least one"),
        (DyadicFrontier((99,)), "not in the tree"),
        (DyadicFrontier((0, 1)), "ancestor"),
        (DyadicFrontier((1,)), "exactly cover"),
    ],
)
def test_frontier_validation_rejects_invalid_ids_overlap_and_coverage(
    frontier: DyadicFrontier,
    message: str,
) -> None:
    """Malformed node selections fail with the relevant frontier invariant."""
    tree = CanonicalDyadicTree.from_shape((2, 2))

    with pytest.raises(ValueError, match=message):
        frontier.validate(tree)


def test_frontier_constructor_rejects_duplicate_and_noninteger_ids() -> None:
    """Duplicate, Boolean, and non-integer IDs are not silently normalized."""
    with pytest.raises(ValueError, match="duplicates"):
        DyadicFrontier((1, 1))
    with pytest.raises(TypeError, match="integer"):
        DyadicFrontier((True,))
    with pytest.raises(TypeError, match="integer"):
        DyadicFrontier((1.5,))  # type: ignore[arg-type]


def test_split_merge_candidates_and_active_splits_are_reciprocal() -> None:
    """Every local split has its parent merge and preserves source frontiers."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    root = DyadicFrontier.root(tree)
    two_regions = root.split(tree, tree.root_id)
    three_regions = two_regions.split(tree, tree.children(tree.root_id)[0])

    assert root.node_ids == (0,)
    assert tree.splittable_nodes(root) == (0,)
    assert root.active_split_nodes(tree) == ()
    assert two_regions.node_ids == (1, 4)
    assert tree.splittable_nodes(two_regions) == (1, 4)
    assert tree.mergeable_parents(two_regions) == (0,)
    assert two_regions.active_split_nodes(tree) == (0,)
    assert three_regions.node_ids == (2, 3, 4)
    assert tree.mergeable_parents(three_regions) == (1,)
    assert three_regions.active_split_nodes(tree) == (0, 1)
    assert three_regions.merge(tree, 1) == two_regions
    assert two_regions.merge(tree, 0) == root
    assert root == DyadicFrontier((0,))


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (2, 2), (2, 3), (3, 2), (3, 3)])
def test_candidate_queries_match_full_tree_reference_for_every_frontier(
    shape: tuple[int, int],
) -> None:
    """Frontier-local candidate queries equal exhaustive full-tree scans."""
    tree = CanonicalDyadicTree.from_shape(shape)
    for frontier in enumerate_frontiers(tree):
        active = frozenset(frontier.node_ids)
        expected_splittable = tuple(node_id for node_id in frontier.node_ids if tree.nodes[node_id].child_ids)
        expected_mergeable = tuple(
            parent_id
            for parent_id in tree.internal_node_ids
            if all(child_id in active for child_id in tree.nodes[parent_id].child_ids)
        )

        split_nodes: list[int] = []
        pending = [tree.root_id]
        while pending:
            node_id = pending.pop()
            if node_id in active:
                continue
            split_nodes.append(node_id)
            pending.extend(reversed(tree.nodes[node_id].child_ids))

        assert tree.splittable_nodes(frontier) == expected_splittable
        assert tree.mergeable_parents(frontier) == expected_mergeable
        assert frontier.active_split_nodes(tree) == tuple(split_nodes)


def test_optimized_validation_matches_original_errors_for_all_tiny_subsets() -> None:
    """Every 2x3 node subset retains the original validation classification."""
    tree = CanonicalDyadicTree.from_shape((2, 3))
    node_ids = tuple(node.node_id for node in tree.nodes)
    for size in range(len(node_ids) + 1):
        for subset in combinations(node_ids, size):
            frontier = DyadicFrontier(subset)
            assert _validation_error(tree, frontier) == _reference_frontier_error(tree, frontier)
            for unknown_id in (-1, len(tree.nodes)):
                frontier_with_unknown = DyadicFrontier((*subset, unknown_id))
                assert _validation_error(
                    tree,
                    frontier_with_unknown,
                ) == _reference_frontier_error(tree, frontier_with_unknown)

    assert _validation_error(tree, DyadicFrontier((-1,))) == "unknown"
    assert _validation_error(tree, DyadicFrontier((len(tree.nodes),))) == "unknown"


def test_topology_queries_only_index_nodes_reachable_from_large_frontier() -> None:
    """Candidate work remains proportional to K and never scans the full tree."""
    tree = CanonicalDyadicTree.from_shape((64, 64))
    frontier = DyadicFrontier.root(tree)
    while len(frontier) < 64:
        frontier = frontier.split(tree, tree.splittable_nodes(frontier)[0])

    expected_splittable = tree.splittable_nodes(frontier)
    expected_mergeable = tree.mergeable_parents(frontier)
    expected_active_splits = frontier.active_split_nodes(tree)
    original_nodes = tree.nodes
    counting_nodes = _CountingNodes(original_nodes)
    object.__setattr__(tree, "nodes", counting_nodes)
    try:
        assert tree.splittable_nodes(frontier) == expected_splittable
        assert counting_nodes.reads <= 3 * len(frontier)

        counting_nodes.reads = 0
        assert tree.mergeable_parents(frontier) == expected_mergeable
        assert counting_nodes.reads <= 3 * len(frontier)

        counting_nodes.reads = 0
        assert frontier.active_split_nodes(tree) == expected_active_splits
        assert counting_nodes.reads <= 3 * len(frontier)
    finally:
        object.__setattr__(tree, "nodes", original_nodes)


def test_invalid_split_and_merge_requests_fail() -> None:
    """Cells, inactive nodes, and incomplete sibling pairs cannot be moved."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    two_regions = DyadicFrontier.root(tree).split(tree, 0)
    three_regions = two_regions.split(tree, 1)

    with pytest.raises(ValueError, match="not active"):
        two_regions.split(tree, 2)
    with pytest.raises(ValueError, match="Cell"):
        three_regions.split(tree, 2)
    with pytest.raises(ValueError, match="both active children"):
        three_regions.merge(tree, 0)


def test_render_labels_are_compact_and_follow_frontier_order() -> None:
    """Rendered labels cover the exact grid in canonical node-ID order."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    frontier = DyadicFrontier((2, 3, 4))

    np.testing.assert_array_equal(
        frontier.render_labels(tree),
        np.array([[1, 2], [3, 3]], dtype=np.int64),
    )


def test_two_by_two_has_five_exact_frontiers_and_reference_counts() -> None:
    """Enumeration and dynamic programming agree on all five 2x2 frontiers."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    frontiers = enumerate_frontiers(tree)

    assert tuple(frontier.node_ids for frontier in frontiers) == (
        (0,),
        (1, 4),
        (1, 5, 6),
        (2, 3, 4),
        (2, 3, 5, 6),
    )
    assert partition_counts_by_k(tree) == (0, 1, 1, 2, 1)
    assert all(isinstance(count, int) for count in partition_counts_by_k(tree))
    assert tuple(len(enumerate_frontiers(tree, k=k)) for k in range(1, len(tree.leaf_ids) + 1)) == (
        1,
        1,
        2,
        1,
    )


def test_partition_count_limit_and_large_counts_use_python_integers() -> None:
    """The DP can truncate at K while retaining arbitrary-precision counts."""
    small_tree = CanonicalDyadicTree.from_shape((4, 4))
    large_tree = CanonicalDyadicTree.from_shape((8, 16))

    assert partition_counts_by_k(small_tree, max_k=5) == (0, 1, 1, 2, 5, 14)
    counts = partition_counts_by_k(large_tree)
    assert max(counts) > np.iinfo(np.int64).max
    assert all(isinstance(count, int) for count in counts)


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (2, 2), (2, 3), (3, 2), (3, 3)])
def test_subtree_index_matches_every_tiny_enumerated_frontier(
    shape: tuple[int, int],
) -> None:
    """Counts, ranks, and unranking match exhaustive order below every node."""
    tree = CanonicalDyadicTree.from_shape(shape)
    index = SubtreePartitionIndex(tree, max_k=len(tree.leaf_ids))

    for node in tree.nodes:
        expected_all = _enumerate_subtree_frontiers(tree, node.node_id)
        assert index.counts_by_k(node.node_id) == tuple(
            0 if k == 0 else sum(len(frontier) == k for frontier in expected_all)
            for k in range(node.area + 1)
        )
        for k in range(1, node.area + 1):
            expected = tuple(frontier for frontier in expected_all if len(frontier) == k)
            assert index.count(node.node_id, k) == len(expected)
            assert tuple(index.unrank(node.node_id, k, rank) for rank in range(len(expected))) == expected
            assert tuple(index.rank(node.node_id, k, frontier) for frontier in expected) == tuple(
                range(len(expected))
            )


@pytest.mark.parametrize("shape", [(1, 4), (2, 2), (2, 3), (3, 3)])
def test_root_subtree_ranks_match_filtered_public_enumeration(
    shape: tuple[int, int],
) -> None:
    """Root ranking is identical to enumerate_frontiers exact-K filtering."""
    tree = CanonicalDyadicTree.from_shape(shape)
    index = SubtreePartitionIndex(tree, max_k=len(tree.leaf_ids))

    for k in range(1, len(tree.leaf_ids) + 1):
        expected = enumerate_frontiers(tree, k=k)
        assert (
            tuple(index.unrank(tree.root_id, k, rank) for rank in range(index.count(tree.root_id, k)))
            == expected
        )


def test_subtree_index_is_immutable_and_uses_arbitrary_precision_counts() -> None:
    """The frozen reusable index retains counts beyond NumPy integer range."""
    tree = CanonicalDyadicTree.from_shape((8, 16))
    index = SubtreePartitionIndex(tree, max_k=len(tree.leaf_ids))
    counts = index.counts_by_k(tree.root_id)

    assert max(counts) > np.iinfo(np.int64).max
    assert all(isinstance(count, int) for count in counts)
    with pytest.raises(FrozenInstanceError):
        index.max_k = 3  # type: ignore[misc]


def test_subtree_index_unranks_large_space_without_frontier_materialization() -> None:
    """First, middle, and last huge-space ranks round-trip exactly."""
    tree = CanonicalDyadicTree.from_shape((8, 16))
    index = SubtreePartitionIndex(tree, max_k=64)
    count = index.count(tree.root_id, 64)

    assert count > np.iinfo(np.int64).max
    for rank in (0, count // 2, count - 1):
        frontier = index.unrank(tree.root_id, 64, rank)
        assert len(frontier) == 64
        assert index.rank(tree.root_id, 64, frontier) == rank


def test_subtree_index_rejects_invalid_node_k_rank_and_frontier_inputs() -> None:
    """Malformed subtree indexing requests fail before ambiguous traversal."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    index = SubtreePartitionIndex(tree, max_k=3)
    first_child = tree.children(tree.root_id)[0]
    first_grandchild = tree.children(first_child)[0]

    with pytest.raises(KeyError, match="Unknown"):
        index.count(99, 1)
    with pytest.raises(TypeError, match="integer"):
        index.count(tree.root_id, True)
    with pytest.raises(TypeError, match="integer"):
        index.count(tree.root_id, 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="between 1 and 3"):
        index.count(tree.root_id, 0)
    with pytest.raises(ValueError, match="between 1 and 1"):
        index.count(first_grandchild, 2)
    with pytest.raises(TypeError, match="integer"):
        index.unrank(tree.root_id, 2, True)
    with pytest.raises(TypeError, match="integer"):
        index.unrank(tree.root_id, 2, 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank must lie"):
        index.unrank(tree.root_id, 2, -1)
    with pytest.raises(ValueError, match="rank must lie"):
        index.unrank(tree.root_id, 2, index.count(tree.root_id, 2))
    with pytest.raises(TypeError, match="DyadicFrontier"):
        index.rank(tree.root_id, 2, (1, 4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        index.rank(tree.root_id, 2, DyadicFrontier(()))
    with pytest.raises(ValueError, match="exactly cover subtree"):
        index.rank(first_child, 1, DyadicFrontier((tree.root_id,)))
    with pytest.raises(ValueError, match="not in the tree"):
        index.rank(tree.root_id, 2, DyadicFrontier((1, 99)))
    with pytest.raises(ValueError, match="has 1 nodes but k is 2"):
        index.rank(tree.root_id, 2, DyadicFrontier((tree.root_id,)))


@pytest.mark.parametrize(
    ("max_k", "error", "message"),
    [
        (True, TypeError, "integer"),
        (1.5, TypeError, "integer"),
        (0, ValueError, "between"),
        (5, ValueError, "between"),
    ],
)
def test_subtree_index_rejects_invalid_bounds(
    max_k: object,
    error: type[Exception],
    message: str,
) -> None:
    """The reusable index requires a positive tree-bounded integer limit."""
    tree = CanonicalDyadicTree.from_shape((2, 2))

    with pytest.raises(error, match=message):
        SubtreePartitionIndex(tree, max_k=max_k)  # type: ignore[arg-type]
