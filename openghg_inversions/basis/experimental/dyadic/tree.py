"""Canonical binary tree geometry for exact rectangular grids.

The tree recursively bisects the longer dimension of each tile at its integer
midpoint. Rows win ties for square tiles. Every non-cell tile therefore has two
ordered children that cover it exactly without overlap, and recursion ends at
individual grid cells.

Node IDs are zero-based integers assigned in depth-first preorder, visiting the
upper or left child first. This makes IDs and node ordering deterministic for a
given root shape while keeping all IDs contiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import TypeAlias

NodeId: TypeAlias = int


@dataclass(frozen=True, slots=True)
class Tile:
    """Immutable rectangular node in a canonical dyadic tree.

    Bounds use half-open row and column index intervals.

    Attributes:
        node_id: Stable integer identifier assigned in depth-first preorder.
        row_start: Inclusive first row index.
        row_stop: Exclusive final row index.
        col_start: Inclusive first column index.
        col_stop: Exclusive final column index.
        depth: Number of edges between this tile and the root.
    """

    node_id: NodeId
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    depth: int

    @property
    def height(self) -> int:
        """Return the number of rows covered by the tile."""
        return self.row_stop - self.row_start

    @property
    def width(self) -> int:
        """Return the number of columns covered by the tile."""
        return self.col_stop - self.col_start

    @property
    def area(self) -> int:
        """Return the number of grid cells covered by the tile."""
        return self.height * self.width

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Return half-open row and column bounds as one tuple."""
        return self.row_start, self.row_stop, self.col_start, self.col_stop

    @property
    def is_cell(self) -> bool:
        """Return whether the tile contains exactly one grid cell."""
        return self.height == 1 and self.width == 1


@dataclass(frozen=True, slots=True)
class DyadicTree:
    """Complete canonical binary partition tree for one rectangular shape.

    Construct trees with :meth:`from_shape`. The stored tuples are ordered by
    contiguous node ID, so iteration and downstream array construction are
    deterministic.

    Attributes:
        shape: Exact unpadded ``(rows, columns)`` root shape.
    """

    shape: tuple[int, int]
    _nodes: tuple[Tile, ...]
    _children: tuple[tuple[NodeId, ...], ...]
    _parents: tuple[NodeId | None, ...]
    _leaf_ids: tuple[NodeId, ...]

    @classmethod
    def from_shape(cls, shape: tuple[int, int]) -> DyadicTree:
        """Build the complete canonical tree for an exact 2D shape.

        Tiles are assigned contiguous IDs in depth-first preorder. Each tile
        is bisected along its longer dimension; rows win ties. Odd lengths are
        split at ``start + length // 2``, so the second child receives the
        extra row or column.

        Args:
            shape: Positive ``(rows, columns)`` grid shape.

        Returns:
            An immutable tree whose leaves are the individual grid cells.

        Raises:
            TypeError: If either extent is not an integer.
            ValueError: If ``shape`` is not length two or has a non-positive
                extent.
        """
        rows, cols = _validate_shape(shape)
        nodes: list[Tile] = []
        children: list[list[NodeId]] = []
        parents: list[NodeId | None] = []
        pending: list[tuple[int, int, int, int, int, NodeId | None]] = [(0, rows, 0, cols, 0, None)]

        while pending:
            row_start, row_stop, col_start, col_stop, depth, parent_id = pending.pop()
            node_id = len(nodes)
            tile = Tile(node_id, row_start, row_stop, col_start, col_stop, depth)
            nodes.append(tile)
            children.append([])
            parents.append(parent_id)
            if parent_id is not None:
                children[parent_id].append(node_id)

            child_bounds = _split_bounds(tile)
            if child_bounds is None:
                continue

            first, second = child_bounds
            # The stack is last-in, first-out; push the second child first to
            # assign preorder IDs to the upper or left subtree first.
            pending.append((*second, depth + 1, node_id))
            pending.append((*first, depth + 1, node_id))

        child_ids = tuple(tuple(node_children) for node_children in children)
        leaf_ids = tuple(node.node_id for node in nodes if not child_ids[node.node_id])
        return cls(
            shape=(rows, cols),
            _nodes=tuple(nodes),
            _children=child_ids,
            _parents=tuple(parents),
            _leaf_ids=leaf_ids,
        )

    @property
    def root_id(self) -> NodeId:
        """Return the stable root node ID."""
        return 0

    @property
    def nodes(self) -> tuple[Tile, ...]:
        """Return all tiles in stable node-ID order."""
        return self._nodes

    @property
    def leaf_ids(self) -> tuple[NodeId, ...]:
        """Return cell-node IDs in stable node-ID order."""
        return self._leaf_ids

    def tile(self, node_id: NodeId) -> Tile:
        """Return the tile associated with a node ID.

        Args:
            node_id: Integer node identifier in this tree.

        Returns:
            The immutable tile for ``node_id``.

        Raises:
            KeyError: If ``node_id`` does not identify a node in this tree.
        """
        return self._nodes[self._node_index(node_id)]

    def children(self, node_id: NodeId) -> tuple[NodeId, ...]:
        """Return a node's ordered child IDs.

        Args:
            node_id: Integer node identifier in this tree.

        Returns:
            Two child IDs for a non-cell tile, or an empty tuple for a cell.

        Raises:
            KeyError: If ``node_id`` does not identify a node in this tree.
        """
        return self._children[self._node_index(node_id)]

    def parent(self, node_id: NodeId) -> NodeId | None:
        """Return a node's parent ID, or ``None`` for the root.

        Args:
            node_id: Integer node identifier in this tree.

        Returns:
            The parent node ID, or ``None`` when ``node_id`` is the root.

        Raises:
            KeyError: If ``node_id`` does not identify a node in this tree.
        """
        return self._parents[self._node_index(node_id)]

    def _node_index(self, node_id: NodeId) -> int:
        """Return a checked tuple index for ``node_id``."""
        if isinstance(node_id, bool):
            raise KeyError(f"Unknown dyadic node ID {node_id!r}.")
        try:
            node_index = index(node_id)
        except TypeError as error:
            raise KeyError(f"Unknown dyadic node ID {node_id!r}.") from error
        if node_index < 0 or node_index >= len(self._nodes):
            raise KeyError(f"Unknown dyadic node ID {node_id!r}.")
        return node_index


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Validate and normalize a positive two-dimensional grid shape.

    Args:
        shape: Candidate ``(rows, columns)`` shape.

    Returns:
        The shape as a pair of built-in integers.

    Raises:
        TypeError: If either extent is not an integer.
        ValueError: If the shape is not length two or an extent is not
            positive.
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
    tile: Tile,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """Return canonical child bounds for one tile.

    Args:
        tile: Tile to bisect along its longer dimension. Rows win ties.

    Returns:
        Upper/lower or left/right child bounds, or ``None`` when ``tile`` is a
        cell.
    """
    if tile.is_cell:
        return None
    if tile.height >= tile.width:
        midpoint = tile.row_start + tile.height // 2
        return (
            (tile.row_start, midpoint, tile.col_start, tile.col_stop),
            (midpoint, tile.row_stop, tile.col_start, tile.col_stop),
        )

    midpoint = tile.col_start + tile.width // 2
    return (
        (tile.row_start, tile.row_stop, tile.col_start, midpoint),
        (tile.row_start, tile.row_stop, midpoint, tile.col_stop),
    )


__all__ = ["DyadicTree", "NodeId", "Tile"]
