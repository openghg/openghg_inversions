"""Tests for sum-preserving dyadic multiscale design construction."""

from __future__ import annotations

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.multiscale import (
    MultiscaleDesign,
    direct_gather,
    sum_coarsen_grid,
)
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree


def test_sum_coarsen_grid_preserves_partial_blocks_and_support() -> None:
    """Coarsening should sum partial boundaries and count their fine cells."""
    grid = np.arange(1.0, 31.0).reshape(2, 3, 5)

    result = sum_coarsen_grid(grid, factor=2)

    expected = np.empty((2, 2, 3))
    for row in range(2):
        for column in range(3):
            expected[:, row, column] = grid[:, row * 2 : (row + 1) * 2, column * 2 : (column + 1) * 2].sum(
                axis=(1, 2)
            )
    np.testing.assert_array_equal(result.values, expected)
    np.testing.assert_array_equal(result.support_counts, [[4, 4, 2], [2, 2, 1]])
    np.testing.assert_array_equal(result.values.sum(axis=(1, 2)), grid.sum(axis=(1, 2)))


def test_candidate_columns_equal_direct_tile_sums() -> None:
    """Every recursively constructed candidate should equal a direct grid sum."""
    grid = np.arange(1.0, 13.0).reshape(3, 2, 2)
    tree = DyadicTree.from_shape((2, 2))

    design = MultiscaleDesign.from_grid(grid, tree)

    for node_id, tile in enumerate(tree.nodes):
        row_start, row_stop, col_start, col_stop = tile.bounds
        expected = grid[:, row_start:row_stop, col_start:col_stop].sum(axis=(1, 2))
        np.testing.assert_array_equal(design.H[:, node_id], expected)
    np.testing.assert_array_equal(design.H[:, 0], design.H[:, 1] + design.H[:, 4])


def test_gather_matches_independent_direct_aggregation() -> None:
    """Stable gathered columns should match direct active-tile aggregation."""
    grid = np.arange(1.0, 13.0).reshape(3, 2, 2)
    tree = DyadicTree.from_shape((2, 2))
    state = PartitionState(frozenset({1, 5, 6}))
    design = MultiscaleDesign.from_grid(grid, tree)

    gathered = design.gather(state)

    np.testing.assert_array_equal(gathered, direct_gather(grid, tree, state))
    np.testing.assert_array_equal(gathered[:, 1], grid[:, 1, 0])
    np.testing.assert_array_equal(gathered[:, 2], grid[:, 1, 1])


def test_multiscale_design_integrates_with_concrete_tree_and_state() -> None:
    """The concrete experimental types should satisfy the design contracts."""
    grid = np.arange(24.0).reshape(2, 3, 4)
    tree = DyadicTree.from_shape((3, 4))
    state = PartitionState.root(tree).split(tree, tree.root_id)

    design = MultiscaleDesign.from_grid(grid, tree)

    np.testing.assert_array_equal(design.gather(state), direct_gather(grid, tree, state))


def test_multiscale_rejects_invalid_inputs() -> None:
    """Multiscale helpers should reject invalid factors, values, and shapes."""
    tree = DyadicTree.from_shape((2, 2))
    with pytest.raises(ValueError, match="positive"):
        sum_coarsen_grid(np.ones((1, 2, 2)), factor=0)
    with pytest.raises(TypeError, match="integer"):
        sum_coarsen_grid(np.ones((1, 2, 2)), factor=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        sum_coarsen_grid(np.array([[[np.nan]]]), factor=1)
    with pytest.raises(ValueError, match="spatial shape"):
        MultiscaleDesign.from_grid(np.ones((1, 2, 3)), tree)
    with pytest.raises(ValueError, match="finite"):
        MultiscaleDesign.from_grid(np.full((1, 2, 2), np.inf), tree)


def test_gather_rejects_unknown_node_id() -> None:
    """Gathering should fail clearly when a state references no candidate column."""
    tree = DyadicTree.from_shape((2, 2))
    design = MultiscaleDesign.from_grid(np.ones((1, 2, 2)), tree)

    with pytest.raises(ValueError, match="outside"):
        design.gather(PartitionState(frozenset({7})))
