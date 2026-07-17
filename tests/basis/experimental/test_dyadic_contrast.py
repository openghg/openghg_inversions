"""Tests for fixed root-and-contrast coordinates on dyadic trees."""

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.contrast import TreeContrastLayout
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree


def test_layout_has_one_coordinate_per_finest_grid_cell() -> None:
    """A full binary tree should have one root/contrast degree per leaf."""
    tree = DyadicTree.from_shape((2, 3))

    layout = TreeContrastLayout.from_tree(tree)

    assert layout.coordinate_count == 6
    assert layout.coordinate_count == len(tree.leaf_ids)
    assert layout.split_node_ids == tuple(tile.node_id for tile in tree.nodes if tree.children(tile.node_id))


def test_active_indices_follow_partition_refinement() -> None:
    """Splitting a frontier should activate exactly its parent contrast."""
    tree = DyadicTree.from_shape((1, 3))
    layout = TreeContrastLayout.from_tree(tree)
    root = PartitionState.root(tree)
    first_split = root.split(tree, tree.root_id)
    splittable_child = next(node_id for node_id in first_split.active if tree.children(node_id))
    finest = first_split.split(tree, splittable_child)

    assert layout.active_coordinate_indices(root) == (0,)
    assert layout.active_coordinate_indices(first_split) == (0, layout.contrast_index(tree.root_id))
    assert layout.active_coordinate_indices(finest) == tuple(range(layout.coordinate_count))
    assert layout.inactive_coordinate_indices(finest) == ()


def test_decoder_preserves_parent_weighted_mean() -> None:
    """Decoded child means should retain the root mean under area weighting."""
    tree = DyadicTree.from_shape((1, 3))
    layout = TreeContrastLayout.from_tree(tree)
    partition = PartitionState.root(tree).split(tree, tree.root_id)
    coordinates = np.array([2.5, -1.2, 99.0])

    region_means = layout.decode(partition, coordinates)
    areas = np.array([tree.tile(node_id).area for node_id in partition.ordered_active()])

    np.testing.assert_allclose(region_means, [1.7, 2.9])
    assert region_means[0] - region_means[1] == pytest.approx(coordinates[1])
    assert np.average(region_means, weights=areas) == pytest.approx(coordinates[0])
    assert np.all(layout.decoder(partition)[:, layout.inactive_coordinate_indices(partition)] == 0.0)


@pytest.mark.parametrize("shape", [(1, 3), (2, 2), (2, 3)])
def test_contrast_prior_recovers_independent_finest_grid_prior(shape: tuple[int, int]) -> None:
    """Primitive contrast variances should decode to IID finest-grid anomalies."""
    tree = DyadicTree.from_shape(shape)
    layout = TreeContrastLayout.from_tree(tree)
    finest = PartitionState(active=frozenset(tree.leaf_ids))
    scale = 1.7
    decoder = layout.decoder(finest)

    decoded_covariance = decoder @ np.diag(layout.prior_variances(scale)) @ decoder.T

    np.testing.assert_allclose(decoded_covariance, scale**2 * np.eye(len(tree.leaf_ids)), atol=1e-12)


def test_contrast_prior_recovers_independent_active_region_means() -> None:
    """A coarse frontier should have variance scaled by each region's area."""
    tree = DyadicTree.from_shape((2, 3))
    layout = TreeContrastLayout.from_tree(tree)
    partition = PartitionState.root(tree).split(tree, tree.root_id)
    scale = 0.8
    active_indices = layout.active_coordinate_indices(partition)
    decoder = layout.decoder(partition)[:, active_indices]
    covariance = decoder @ np.diag(layout.prior_variances(scale)[list(active_indices)]) @ decoder.T
    expected = np.diag([scale**2 / tree.tile(node_id).area for node_id in partition.ordered_active()])

    np.testing.assert_allclose(covariance, expected, atol=1e-12)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_prior_variances_reject_invalid_scale(scale: float) -> None:
    """Primitive prior scales must be positive and finite."""
    layout = TreeContrastLayout.from_tree(DyadicTree.from_shape((1, 2)))

    with pytest.raises(ValueError, match="scale"):
        layout.prior_variances(scale)


def test_decode_rejects_wrong_coordinate_shape() -> None:
    """The permanent coordinate vector must match the finest-grid dimension."""
    tree = DyadicTree.from_shape((1, 2))
    layout = TreeContrastLayout.from_tree(tree)

    with pytest.raises(ValueError, match="shape"):
        layout.decode(PartitionState.root(tree), [0.0])


def test_direct_layout_construction_enforces_stable_split_ids() -> None:
    """The public constructor should reject incomplete coordinate layouts."""
    tree = DyadicTree.from_shape((1, 2))

    with pytest.raises(ValueError, match="split_node_ids"):
        TreeContrastLayout(tree=tree, split_node_ids=())


@pytest.mark.parametrize("scale", [1e154, 1e308])
def test_prior_variances_reject_overflowing_scale(scale: float) -> None:
    """Large finite scales must not create infinite primitive variances."""
    layout = TreeContrastLayout.from_tree(DyadicTree.from_shape((1, 2)))

    with pytest.raises(ValueError, match="non-finite primitive variances"):
        layout.prior_variances(scale)
