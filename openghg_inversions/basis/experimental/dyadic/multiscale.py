"""Pure-NumPy construction of sum-preserving dyadic design matrices.

The fine-grid input convention is ``(observation, row, column)``.  A candidate
column for a dyadic tile is the sum of its fine-cell columns, matching a model
in which one regional multiplier applies to every cell in the tile.  Internal
node columns are built from child columns, while :func:`direct_gather` provides
an intentionally independent grid-slice implementation for parity tests.

This module handles design aggregation only.  Prior covariance is partition
dependent: a Bocquet-consistent objective should supply the covariance for a
partition as ``B_P = P B P.T`` rather than assume one covariance applies to
every gathered design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .state import PartitionState
from .tree import DyadicTree, Tile


@dataclass(frozen=True)
class CoarsenedGrid:
    """A sum-preserving coarsened observation grid and its physical support.

    Attributes:
        values: Coarsened values with shape ``(observation, coarse_row,
            coarse_column)``.  Boundary blocks contain sums over their partial
            physical support; no zero padding is included in the sums.
        support_counts: Number of fine cells contributing to each coarse block,
            with shape ``(coarse_row, coarse_column)``.
    """

    values: np.ndarray
    support_counts: np.ndarray


@dataclass(frozen=True)
class MultiscaleDesign:
    """Candidate observation columns for every node in a dyadic tree.

    Columns in ``values`` use the deterministic node order of the tree.  The
    current tree contract assigns contiguous integer IDs, so column ``node_id``
    is ``values[:, node_id]``.

    Attributes:
        values: Candidate matrix with shape ``(observation, tree_node)``.
        tree: Tree whose node IDs define the columns.

    Notes:
        This class deliberately does not store a prior covariance.  A
        Bocquet-consistent covariance must be constructed for each partition as
        ``B_P = P B P.T``.  An isotropic covariance is only a proof-of-concept
        benchmark, not a substitute for that aggregation.
    """

    values: np.ndarray
    tree: DyadicTree

    @classmethod
    def from_grid(cls, grid: npt.ArrayLike, tree: DyadicTree) -> MultiscaleDesign:
        """Build all candidate columns from a finite fine-grid design.

        Args:
            grid: Fine-grid contributions with shape ``(observation, row,
                column)``.
            tree: Deterministic tree whose shape equals the grid's spatial
                shape and whose node IDs are contiguous from zero.

        Returns:
            A design containing one summed observation column per tree node.

        Raises:
            ValueError: If the grid is not finite and three-dimensional, its
                shape does not match the tree, the tree IDs are not contiguous,
                or the tree topology is inconsistent with cell leaves.
        """
        values = _finite_float_array(grid, name="grid")
        if values.ndim != 3:
            raise ValueError("grid must have shape (observation, row, column).")
        if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
            raise ValueError("grid dimensions must all be non-empty.")
        if tuple(values.shape[1:]) != tuple(tree.shape):
            raise ValueError(f"grid spatial shape {values.shape[1:]} does not match tree shape {tree.shape}.")

        try:
            node_count = len(tree.nodes)
        except TypeError as exc:  # pragma: no cover - concrete tree invariant.
            raise ValueError("tree.nodes must be a sized deterministic collection.") from exc
        if node_count < 1:
            raise ValueError("tree must contain at least one node.")
        if tree.root_id < 0 or tree.root_id >= node_count:
            raise ValueError("tree.root_id is outside the contiguous node ID range.")

        columns = np.empty((values.shape[0], node_count), dtype=float)
        visited: set[int] = set()
        visiting: set[int] = set()

        def build(node_id: int) -> np.ndarray:
            """Build one node column recursively and memoize it."""
            if node_id < 0 or node_id >= node_count:
                raise ValueError(f"child node ID {node_id} is outside the tree node range.")
            if node_id in visited:
                return columns[:, node_id]
            if node_id in visiting:
                raise ValueError("tree contains a cycle.")

            visiting.add(node_id)
            tile = tree.tile(node_id)
            children = tuple(tree.children(node_id))
            if children:
                if tile.is_cell:
                    raise ValueError("a cell tile cannot have children.")
                column = np.zeros(values.shape[0], dtype=float)
                for child_id in children:
                    column += build(child_id)
            else:
                if not tile.is_cell or tile.area != 1:
                    raise ValueError("every childless tree node must be a one-cell tile.")
                row_start, row_stop, col_start, col_stop = _validated_bounds(tile, tree.shape)
                column = values[:, row_start:row_stop, col_start:col_stop].reshape(values.shape[0])

            columns[:, node_id] = column
            visiting.remove(node_id)
            visited.add(node_id)
            return columns[:, node_id]

        build(tree.root_id)
        if len(visited) != node_count:
            raise ValueError("tree contains nodes that are unreachable from its root.")
        return cls(values=columns, tree=tree)

    @property
    def H(self) -> np.ndarray:
        """Return the candidate matrix using its conventional mathematical name."""
        return self.values

    def gather(self, state: PartitionState) -> np.ndarray:
        """Gather active columns in the state's stable active-node order.

        Args:
            state: Partition state associated with this design's tree.

        Returns:
            Matrix with shape ``(observation, active_region)``.

        Raises:
            ValueError: If ``state`` is not an exact frontier for this design's
                tree or an active node ID is outside the candidate matrix.
        """
        state.validate(self.tree)
        active = state.ordered_active()
        if any(node_id < 0 or node_id >= self.values.shape[1] for node_id in active):
            raise ValueError("state contains a node ID outside the design.")
        return self.values[:, active]


def sum_coarsen_grid(grid: npt.ArrayLike, factor: int) -> CoarsenedGrid:
    """Sum spatial blocks while retaining boundary support counts.

    Args:
        grid: Finite values with shape ``(observation, row, column)``.
        factor: Positive integer block width along both spatial axes.

    Returns:
        Coarsened values and the number of physical fine cells in each block.
        Partial boundary blocks are retained rather than dropped or averaged.

    Raises:
        TypeError: If ``factor`` is not an integer.
        ValueError: If ``grid`` is not finite and three-dimensional, has an
            empty spatial axis, or ``factor`` is not positive.
    """
    values = _finite_float_array(grid, name="grid")
    if values.ndim != 3:
        raise ValueError("grid must have shape (observation, row, column).")
    if values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError("grid spatial dimensions must be non-empty.")
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)):
        raise TypeError("factor must be an integer.")
    if factor <= 0:
        raise ValueError("factor must be positive.")

    row_starts = np.arange(0, values.shape[1], factor)
    col_starts = np.arange(0, values.shape[2], factor)
    coarsened = np.add.reduceat(values, row_starts, axis=1)
    coarsened = np.add.reduceat(coarsened, col_starts, axis=2)

    row_counts = np.minimum(factor, values.shape[1] - row_starts)
    col_counts = np.minimum(factor, values.shape[2] - col_starts)
    support_counts = np.multiply.outer(row_counts, col_counts)
    return CoarsenedGrid(values=coarsened, support_counts=support_counts)


def direct_gather(
    grid: npt.ArrayLike,
    tree: DyadicTree,
    state: PartitionState,
) -> np.ndarray:
    """Gather active columns by direct fine-grid tile sums.

    This helper intentionally does not use :class:`MultiscaleDesign`; it is a
    small reference implementation for checking precomputed-column parity.

    Args:
        grid: Finite fine-grid contributions with shape ``(observation, row,
            column)``.
        tree: Tree defining the active tile bounds.
        state: Partition whose stable active order defines output columns.

    Returns:
        Directly aggregated matrix with shape ``(observation, active_region)``.

    Raises:
        ValueError: If the grid is invalid or does not match the tree shape, or
            a tile has invalid bounds.
    """
    values = _finite_float_array(grid, name="grid")
    if values.ndim != 3:
        raise ValueError("grid must have shape (observation, row, column).")
    if tuple(values.shape[1:]) != tuple(tree.shape):
        raise ValueError(f"grid spatial shape {values.shape[1:]} does not match tree shape {tree.shape}.")

    columns = []
    for node_id in state.ordered_active():
        bounds = _validated_bounds(tree.tile(node_id), tree.shape)
        row_start, row_stop, col_start, col_stop = bounds
        columns.append(values[:, row_start:row_stop, col_start:col_stop].sum(axis=(1, 2)))
    if not columns:
        return np.empty((values.shape[0], 0), dtype=float)
    return np.column_stack(columns)


def _finite_float_array(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Convert an array-like input to a finite floating-point array."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validated_bounds(tile: Tile, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return integer tile bounds after validating them against ``shape``."""
    try:
        row_start, row_stop, col_start, col_stop = tile.bounds
    except (TypeError, ValueError) as exc:
        raise ValueError("tile.bounds must contain four integer indices.") from exc
    bounds = (row_start, row_stop, col_start, col_stop)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in bounds):
        raise ValueError("tile bounds must be integers.")
    if not (0 <= row_start < row_stop <= shape[0] and 0 <= col_start < col_stop <= shape[1]):
        raise ValueError(f"tile bounds {bounds} lie outside tree shape {shape}.")
    if (row_stop - row_start) * (col_stop - col_start) != tile.area:
        raise ValueError("tile area is inconsistent with its bounds.")
    return bounds
