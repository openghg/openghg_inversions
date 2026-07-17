"""Enumerate and apply immutable local moves on dyadic partitions.

Single split and merge moves change the active-region count by one.  A paired
move first merges active siblings and then splits another active leaf, keeping
the count fixed.  Paired neighbors are deduplicated by their resulting active
frontier; their proposal probability is uniform over those unique states.

This module describes only proposal geometry.  It deliberately contains no
posterior target or acceptance calculation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

from .state import PartitionState
from .tree import DyadicTree


@dataclass(frozen=True, slots=True)
class SplitMove:
    """Split one active node into its two children."""

    node_id: int

    def reverse(self) -> MergeMove:
        """Return the merge that restores the pre-split partition."""
        return MergeMove(parent_id=self.node_id)


@dataclass(frozen=True, slots=True)
class MergeMove:
    """Merge two active sibling nodes into their parent."""

    parent_id: int

    def reverse(self) -> SplitMove:
        """Return the split that restores the pre-merge partition."""
        return SplitMove(node_id=self.parent_id)


@dataclass(frozen=True, slots=True)
class PairedMove:
    """Merge one sibling pair, then split a different active leaf.

    Attributes:
        merge_parent_id: Parent identifier whose active children are merged.
        split_node_id: Active node identifier split after the merge.
    """

    merge_parent_id: int
    split_node_id: int

    @property
    def merge(self) -> MergeMove:
        """Return the merge component of the paired move."""
        return MergeMove(parent_id=self.merge_parent_id)

    @property
    def split(self) -> SplitMove:
        """Return the split component of the paired move."""
        return SplitMove(node_id=self.split_node_id)

    def reverse(self) -> PairedMove:
        """Return the paired move restoring the source partition.

        The forward split must be merged first, after which the forward merge
        parent is active and can be split again.
        """
        return PairedMove(
            merge_parent_id=self.split_node_id,
            split_node_id=self.merge_parent_id,
        )


Move: TypeAlias = SplitMove | MergeMove | PairedMove


@dataclass(frozen=True, slots=True)
class PairedNeighbor:
    """A unique fixed-count neighbor and its uniform forward log probability.

    Attributes:
        state: Partition obtained by applying ``move``.
        move: Deterministic representative move for this resulting state.
        log_q: Natural logarithm of the uniform probability over all unique
            paired neighbors of the source state.
    """

    state: PartitionState
    move: PairedMove
    log_q: float


def enumerate_split_moves(tree: DyadicTree, state: PartitionState) -> tuple[SplitMove, ...]:
    """Return splittable active nodes in stable identifier order.

    Args:
        tree: Tree defining child relationships.
        state: Valid active partition frontier.

    Returns:
        One split move for every active node with children.
    """
    state.validate(tree)
    return tuple(SplitMove(node_id=node_id) for node_id in state.ordered_active() if tree.children(node_id))


def enumerate_merge_moves(tree: DyadicTree, state: PartitionState) -> tuple[MergeMove, ...]:
    """Return parents whose two children are active, without duplicates.

    Args:
        tree: Tree defining parent and child relationships.
        state: Valid active partition frontier.

    Returns:
        Merge moves ordered by increasing parent identifier.
    """
    state.validate(tree)
    parent_ids: set[int] = set()
    for node_id in state.active:
        parent_id = tree.parent(node_id)
        if parent_id is None:
            continue
        children = tree.children(parent_id)
        if children and all(child_id in state.active for child_id in children):
            parent_ids.add(parent_id)
    return tuple(MergeMove(parent_id=parent_id) for parent_id in sorted(parent_ids))


def enumerate_paired_moves(tree: DyadicTree, state: PartitionState) -> tuple[PairedMove, ...]:
    """Return one deterministic move for each unique fixed-count neighbor.

    Candidate moves merge an active sibling pair and then split a splittable
    leaf in the intermediate partition.  Re-splitting the just-merged parent is
    excluded because it would be a no-op.  If several candidates produce the
    same active frontier, only the first lexicographically generated move is
    retained.

    Args:
        tree: Tree defining partition relationships.
        state: Valid source partition.

    Returns:
        Paired moves ordered first by merge parent and then by split node.
    """
    return tuple(neighbor.move for neighbor in enumerate_paired_neighbors(tree, state))


def enumerate_paired_neighbors(
    tree: DyadicTree,
    state: PartitionState,
) -> tuple[PairedNeighbor, ...]:
    """Enumerate unique fixed-count neighbors with uniform forward ``log_q``.

    Args:
        tree: Tree defining partition relationships.
        state: Valid source partition.

    Returns:
        Unique neighbor records.  Every record has ``log_q = -log(N)``, where
        ``N`` is the number of unique returned states.  An isolated state
        returns an empty tuple.
    """
    state.validate(tree)
    unique: dict[frozenset[int], tuple[PartitionState, PairedMove]] = {}

    for merge_move in enumerate_merge_moves(tree, state):
        merged_state = apply_move(tree, state, merge_move)
        for split_move in enumerate_split_moves(tree, merged_state):
            if split_move.node_id == merge_move.parent_id:
                continue
            move = PairedMove(
                merge_parent_id=merge_move.parent_id,
                split_node_id=split_move.node_id,
            )
            neighbor_state = apply_move(tree, state, move)
            unique.setdefault(neighbor_state.active, (neighbor_state, move))

    if not unique:
        return ()

    log_q = -math.log(len(unique))
    return tuple(
        PairedNeighbor(state=neighbor_state, move=move, log_q=log_q)
        for neighbor_state, move in unique.values()
    )


def apply_move(tree: DyadicTree, state: PartitionState, move: Move) -> PartitionState:
    """Apply a valid local move and return a new partition state.

    Args:
        tree: Tree defining partition relationships.
        state: Source partition, which is never mutated.
        move: Split, merge, or merge-then-split move to apply.

    Returns:
        The validated destination partition.

    Raises:
        TypeError: If ``move`` is not a supported move type.
        ValueError: Propagated by ``PartitionState`` when the requested move is
            invalid for ``state``.
    """
    state.validate(tree)
    if isinstance(move, SplitMove):
        result = state.split(tree, move.node_id)
    elif isinstance(move, MergeMove):
        result = state.merge(tree, move.parent_id)
    elif isinstance(move, PairedMove):
        if move.merge_parent_id == move.split_node_id:
            raise ValueError("A paired move cannot re-split its merged parent.")
        result = state.merge(tree, move.merge_parent_id)
        result = result.split(tree, move.split_node_id)
    else:
        raise TypeError(f"Unsupported move type: {type(move).__name__}.")

    result.validate(tree)
    return result


def reverse_move(move: Move) -> Move:
    """Return the local move that reverses ``move`` on its destination state.

    Args:
        move: Split, merge, or paired move to reverse.

    Returns:
        The corresponding inverse move.

    Raises:
        TypeError: If ``move`` is not a supported move type.
    """
    if isinstance(move, (SplitMove, MergeMove, PairedMove)):
        return move.reverse()
    raise TypeError(f"Unsupported move type: {type(move).__name__}.")
