"""Tests for immutable canonical dyadic trees and exact frontiers."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.dyadic_tree import (
    CanonicalDyadicTree,
    DyadicFrontier,
    enumerate_frontiers,
    partition_counts_by_k,
)


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
