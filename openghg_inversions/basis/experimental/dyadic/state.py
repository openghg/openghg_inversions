"""Immutable active-frontier states for canonical dyadic trees.

A valid partition state is an exact frontier through one
:class:`~openghg_inversions.basis.experimental.dyadic.tree.DyadicTree`: active
tiles cover the root exactly once and no active tile is an ancestor of another.
Split and merge operations return new states and never mutate their input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tree import DyadicTree, NodeId


@dataclass(frozen=True, slots=True)
class PartitionState:
    """Immutable set of active tile IDs forming a dyadic partition frontier.

    The constructor normalizes the supplied collection to a ``frozenset``.
    Call :meth:`validate` when associating a state with a particular tree.

    Attributes:
        active: Active tile IDs. A valid set covers its tree's root exactly
            once without ancestor/descendant overlap.
    """

    active: frozenset[NodeId]

    def __post_init__(self) -> None:
        """Normalize active IDs to an immutable set."""
        object.__setattr__(self, "active", frozenset(self.active))

    @classmethod
    def root(cls, tree: DyadicTree) -> PartitionState:
        """Create the coarsest valid state containing only the root.

        Args:
            tree: Tree whose root should form the partition.

        Returns:
            A valid one-tile partition state.
        """
        return cls(active=frozenset({tree.root_id}))

    def ordered_active(self) -> tuple[NodeId, ...]:
        """Return active node IDs in stable ascending tree order."""
        return tuple(sorted(self.active))

    def validate(self, tree: DyadicTree) -> None:
        """Validate that this state is an exact frontier over ``tree``.

        Validation checks that every active ID belongs to the tree, that no
        active node has an active ancestor, and that every root-to-cell path
        encounters an active node. Together these establish exact, non-
        overlapping coverage of the root.

        Args:
            tree: Canonical tree against which to validate the state.

        Raises:
            ValueError: If the active set is empty, includes an unknown node,
                overlaps through ancestry, or leaves part of the root
                uncovered.
        """
        if not self.active:
            raise ValueError("A partition state must contain at least one active tile.")

        for node_id in self.active:
            try:
                tree.tile(node_id)
            except KeyError as error:
                raise ValueError(f"Active node ID {node_id!r} is not in the tree.") from error

        for node_id in self.ordered_active():
            ancestor_id = tree.parent(node_id)
            while ancestor_id is not None:
                if ancestor_id in self.active:
                    raise ValueError("A partition state cannot contain active ancestor and descendant tiles.")
                ancestor_id = tree.parent(ancestor_id)

        pending = [tree.root_id]
        while pending:
            node_id = pending.pop()
            if node_id in self.active:
                continue
            child_ids = tree.children(node_id)
            if not child_ids:
                raise ValueError("Active tiles do not exactly cover the tree root.")
            pending.extend(child_ids)

    def split(self, tree: DyadicTree, node_id: NodeId) -> PartitionState:
        """Replace one active tile with its two canonical children.

        Args:
            tree: Tree defining the partition and child relationships.
            node_id: Active non-cell node to split.

        Returns:
            A new valid partition state; this state is unchanged.

        Raises:
            ValueError: If this state is invalid for ``tree``, ``node_id`` is
                not active, or ``node_id`` is already a cell.
        """
        self.validate(tree)
        if node_id not in self.active:
            raise ValueError(f"Node {node_id!r} is not active and cannot be split.")
        child_ids = tree.children(node_id)
        if not child_ids:
            raise ValueError(f"Cell node {node_id!r} cannot be split.")
        return PartitionState(active=(self.active - {node_id}) | frozenset(child_ids))

    def merge(self, tree: DyadicTree, parent_id: NodeId) -> PartitionState:
        """Replace two active sibling tiles with their parent.

        Args:
            tree: Tree defining the partition and sibling relationships.
            parent_id: Parent whose complete child pair should be merged.

        Returns:
            A new valid partition state; this state is unchanged.

        Raises:
            KeyError: If ``parent_id`` is not in ``tree``.
            ValueError: If this state is invalid, ``parent_id`` is a cell, or
                both true children of ``parent_id`` are not active.
        """
        self.validate(tree)
        child_ids = tree.children(parent_id)
        if not child_ids:
            raise ValueError(f"Cell node {parent_id!r} has no children to merge.")
        if not frozenset(child_ids).issubset(self.active):
            raise ValueError("A merge requires both active children of the same parent.")
        return PartitionState(active=(self.active - frozenset(child_ids)) | {parent_id})

    def to_labels(self, tree: DyadicTree) -> np.ndarray:
        """Render this partition as compact positive integer labels.

        Labels are assigned from one in :meth:`ordered_active` order, giving a
        deterministic packed representation while canonical node IDs remain
        available separately as region identity.

        Args:
            tree: Tree providing tile bounds and output shape.

        Returns:
            A NumPy integer array with ``tree.shape`` and one positive label per
            active tile.

        Raises:
            ValueError: If this state is not an exact frontier over ``tree``.
        """
        self.validate(tree)
        labels = np.zeros(tree.shape, dtype=np.int64)
        for label, node_id in enumerate(self.ordered_active(), start=1):
            tile = tree.tile(node_id)
            labels[tile.row_start : tile.row_stop, tile.col_start : tile.col_stop] = label
        return labels


__all__ = ["PartitionState"]
