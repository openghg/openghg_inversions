"""Tests for exact canonical dyadic trees and immutable partition states."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic import DyadicTree, PartitionState, Tile


def _tile_mask(tree: DyadicTree, node_id: int) -> np.ndarray:
    """Return a Boolean grid mask for one tree tile."""
    tile = tree.tile(node_id)
    mask = np.zeros(tree.shape, dtype=bool)
    mask[tile.row_start : tile.row_stop, tile.col_start : tile.col_stop] = True
    return mask


def test_single_cell_tree_has_one_immutable_root() -> None:
    """A 1x1 shape is represented by one root that is also the only leaf."""
    tree = DyadicTree.from_shape((1, 1))

    assert tree.shape == (1, 1)
    assert tree.root_id == 0
    assert tree.nodes == (Tile(0, 0, 1, 0, 1, 0),)
    assert tree.children(tree.root_id) == ()
    assert tree.parent(tree.root_id) is None
    assert tree.leaf_ids == (tree.root_id,)
    assert tree.tile(tree.root_id).is_cell

    with pytest.raises(FrozenInstanceError):
        tree.tile(tree.root_id).depth = 1  # type: ignore[misc]


def test_odd_non_square_tree_uses_exact_midpoint_geometry() -> None:
    """Odd rectangular roots split their longer axis without padded cells."""
    tree = DyadicTree.from_shape((3, 5))
    first_id, second_id = tree.children(tree.root_id)
    first = tree.tile(first_id)
    second = tree.tile(second_id)

    assert first == Tile(first_id, 0, 3, 0, 2, 1)
    assert second == Tile(second_id, 0, 3, 2, 5, 1)
    assert first.area + second.area == 15
    assert len(tree.nodes) == 2 * 15 - 1

    first_child = tree.tile(tree.children(first_id)[0])
    assert (first_child.row_start, first_child.row_stop) == (0, 1)


def test_square_tiles_split_rows_on_ties() -> None:
    """The canonical square-tile tie rule bisects rows before columns."""
    tree = DyadicTree.from_shape((2, 2))
    first, second = (tree.tile(node_id) for node_id in tree.children(tree.root_id))

    assert (first.row_start, first.row_stop, first.col_start, first.col_stop) == (0, 1, 0, 2)
    assert (second.row_start, second.row_stop, second.col_start, second.col_stop) == (1, 2, 0, 2)


def test_node_ids_and_order_are_stable_for_equal_shapes() -> None:
    """Repeated construction assigns identical contiguous preorder node IDs."""
    first = DyadicTree.from_shape((3, 5))
    second = DyadicTree.from_shape((3, 5))

    assert first.nodes == second.nodes
    assert tuple(tile.node_id for tile in first.nodes) == tuple(range(len(first.nodes)))
    assert tuple(first.children(tile.node_id) for tile in first.nodes) == tuple(
        second.children(tile.node_id) for tile in second.nodes
    )
    assert tuple(first.parent(tile.node_id) for tile in first.nodes) == tuple(
        second.parent(tile.node_id) for tile in second.nodes
    )


def test_every_parent_is_covered_exactly_by_its_children() -> None:
    """Canonical child pairs are disjoint and exactly cover every parent."""
    tree = DyadicTree.from_shape((5, 3))

    for parent in tree.nodes:
        child_ids = tree.children(parent.node_id)
        if not child_ids:
            assert parent.is_cell
            continue
        child_coverage = sum((_tile_mask(tree, node_id) for node_id in child_ids), np.zeros(tree.shape, int))
        np.testing.assert_array_equal(child_coverage, _tile_mask(tree, parent.node_id))
        assert sum(tree.tile(node_id).area for node_id in child_ids) == parent.area


def test_split_and_merge_round_trip_without_mutation() -> None:
    """Splitting then merging returns the root state without changing inputs."""
    tree = DyadicTree.from_shape((3, 5))
    root_state = PartitionState.root(tree)
    original_active = root_state.active

    split_state = root_state.split(tree, tree.root_id)
    merged_state = split_state.merge(tree, tree.root_id)

    assert root_state.active is original_active
    assert root_state == PartitionState.root(tree)
    assert split_state.active == frozenset(tree.children(tree.root_id))
    assert merged_state == root_state
    assert split_state is not root_state
    assert merged_state is not split_state


@pytest.mark.parametrize(
    "state, message",
    [
        (PartitionState(active=frozenset()), "at least one"),
        (PartitionState(active=frozenset({999})), "not in the tree"),
    ],
)
def test_validation_rejects_empty_and_unknown_active_sets(
    state: PartitionState,
    message: str,
) -> None:
    """Validation rejects states that cannot identify a tree frontier."""
    tree = DyadicTree.from_shape((2, 3))

    with pytest.raises(ValueError, match=message):
        state.validate(tree)


def test_validation_rejects_gaps_and_ancestor_overlap() -> None:
    """Validation independently rejects incomplete and overlapping frontiers."""
    tree = DyadicTree.from_shape((2, 3))
    first_child = tree.children(tree.root_id)[0]

    with pytest.raises(ValueError, match="exactly cover"):
        PartitionState(active=frozenset({first_child})).validate(tree)
    with pytest.raises(ValueError, match="ancestor and descendant"):
        PartitionState(active=frozenset({tree.root_id, first_child})).validate(tree)


def test_labels_are_positive_compact_and_follow_stable_active_order() -> None:
    """Label rendering packs exact active tiles in deterministic node-ID order."""
    tree = DyadicTree.from_shape((3, 5))
    state = PartitionState.root(tree).split(tree, tree.root_id)
    labels = state.to_labels(tree)
    first_id, second_id = state.ordered_active()

    assert labels.shape == tree.shape
    assert labels.dtype == np.int64
    assert set(np.unique(labels)) == {1, 2}
    assert np.all(labels[_tile_mask(tree, first_id)] == 1)
    assert np.all(labels[_tile_mask(tree, second_id)] == 2)


def test_adjacent_non_siblings_cannot_be_merged() -> None:
    """Geometrically adjacent leaves with different parents are not mergeable."""
    tree = DyadicTree.from_shape((1, 4))
    state = PartitionState(active=frozenset(tree.leaf_ids))
    middle_left, middle_right = tree.leaf_ids[1:3]

    assert tree.tile(middle_left).col_stop == tree.tile(middle_right).col_start
    assert tree.parent(middle_left) != tree.parent(middle_right)
    with pytest.raises(ValueError, match="both active children of the same parent"):
        state.merge(tree, tree.root_id)
