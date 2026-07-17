"""Tests for immutable split, merge, and paired dyadic proposal moves."""

from __future__ import annotations

import math

import pytest

from openghg_inversions.basis.experimental.dyadic.proposals import (
    MergeMove,
    PairedMove,
    SplitMove,
    apply_move,
    enumerate_merge_moves,
    enumerate_paired_moves,
    enumerate_paired_neighbors,
    enumerate_split_moves,
    reverse_move,
)
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree


def _four_region_state(tree: DyadicTree) -> PartitionState:
    """Return the partition containing all four grandchildren of the root."""
    state = PartitionState.root(tree).split(tree, tree.root_id)
    for node_id in state.ordered_active():
        state = state.split(tree, node_id)
    return state


def test_split_and_merge_moves_round_trip_without_mutation() -> None:
    """A split and its reverse merge should restore the identical source state."""
    tree = DyadicTree.from_shape((4, 4))
    source = PartitionState.root(tree)
    source_active = source.active
    split = SplitMove(node_id=tree.root_id)

    split_state = apply_move(tree, source, split)
    restored = apply_move(tree, split_state, reverse_move(split))

    assert restored == source
    assert source.active is source_active
    assert reverse_move(MergeMove(parent_id=tree.root_id)) == split


def test_move_enumeration_returns_only_valid_stable_candidates() -> None:
    """Single-move enumeration should expose splittable leaves and true siblings."""
    tree = DyadicTree.from_shape((4, 4))
    state = _four_region_state(tree)
    root_children = tree.children(tree.root_id)

    assert enumerate_split_moves(tree, state) == tuple(
        SplitMove(node_id) for node_id in state.ordered_active()
    )
    assert enumerate_merge_moves(tree, state) == tuple(MergeMove(node_id) for node_id in root_children)


def test_paired_moves_preserve_region_count_and_exclude_no_op() -> None:
    """Every paired move should preserve K and avoid re-splitting its merge parent."""
    tree = DyadicTree.from_shape((4, 4))
    source = _four_region_state(tree)
    source_active = source.active

    moves = enumerate_paired_moves(tree, source)

    assert moves
    assert all(move.merge_parent_id != move.split_node_id for move in moves)
    assert all(len(apply_move(tree, source, move).active) == len(source.active) for move in moves)
    assert source.active is source_active


def test_paired_neighbors_are_unique_and_uniform() -> None:
    """Neighbor records should deduplicate states and normalize uniform ``log_q``."""
    tree = DyadicTree.from_shape((4, 4))
    source = _four_region_state(tree)

    neighbors = enumerate_paired_neighbors(tree, source)
    active_frontiers = [neighbor.state.active for neighbor in neighbors]

    assert len(active_frontiers) == len(set(active_frontiers))
    assert sum(math.exp(neighbor.log_q) for neighbor in neighbors) == pytest.approx(1.0)
    assert {neighbor.log_q for neighbor in neighbors} == {-math.log(len(neighbors))}


def test_every_paired_neighbor_has_a_reverse_edge() -> None:
    """Each fixed-count edge should have an enumerable move back to its source."""
    tree = DyadicTree.from_shape((4, 4))
    source = _four_region_state(tree)

    for neighbor in enumerate_paired_neighbors(tree, source):
        reverse = neighbor.move.reverse()
        assert isinstance(reverse, PairedMove)
        assert apply_move(tree, neighbor.state, reverse) == source
        assert reverse in enumerate_paired_moves(tree, neighbor.state)


def test_paired_move_rejects_no_op_resplit() -> None:
    """A merge followed by splitting the same parent should be rejected."""
    tree = DyadicTree.from_shape((4, 4))
    source = _four_region_state(tree)
    merge_parent = tree.children(tree.root_id)[0]

    with pytest.raises(ValueError, match="re-split"):
        apply_move(
            tree,
            source,
            PairedMove(merge_parent_id=merge_parent, split_node_id=merge_parent),
        )
