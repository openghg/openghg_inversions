"""Tests for exact fixed-coordinate dyadic partition Metropolis updates."""

import math

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.product_space import (
    ProductSpaceState,
    enumerate_partition_neighbors,
    partition_metropolis_step,
)
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree


def _one_by_three_partitions() -> tuple[DyadicTree, tuple[PartitionState, ...]]:
    """Return the three-state partition path for the smallest asymmetric tree."""
    tree = DyadicTree.from_shape((1, 3))
    root = PartitionState.root(tree)
    middle = root.split(tree, tree.root_id)
    splittable_child = next(node_id for node_id in middle.active if tree.children(node_id))
    finest = middle.split(tree, splittable_child)
    return tree, (root, middle, finest)


def test_product_space_state_copies_and_freezes_coordinates() -> None:
    """Stored coordinate vectors should not alias mutable caller arrays."""
    tree = DyadicTree.from_shape((1, 2))
    inner = np.array([1.0, 2.0])
    outer = np.array([3.0])
    state = ProductSpaceState(PartitionState.root(tree), inner, outer)
    inner[:] = -1.0
    outer[:] = -1.0

    np.testing.assert_array_equal(state.inner_coordinates, [1.0, 2.0])
    np.testing.assert_array_equal(state.outer_coefficients, [3.0])
    assert not state.inner_coordinates.flags.writeable
    assert not state.outer_coefficients.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        state.inner_coordinates[0] = 4.0


def test_product_space_state_has_array_aware_value_equality() -> None:
    """Equal coordinate values should compare cleanly while states remain unhashable."""
    tree = DyadicTree.from_shape((1, 2))
    first = ProductSpaceState(PartitionState.root(tree), [1.0, 2.0], [3.0])
    second = ProductSpaceState(PartitionState.root(tree), [1.0, 2.0], [3.0])
    different = ProductSpaceState(PartitionState.root(tree), [1.0, 4.0], [3.0])

    assert first == second
    assert first != different
    with pytest.raises(TypeError, match="unhashable"):
        hash(first)


def test_product_space_state_rejects_complex_coordinates() -> None:
    """Complex coordinates should fail instead of losing their imaginary parts."""
    tree = DyadicTree.from_shape((1, 2))

    with pytest.raises(ValueError, match="real-valued"):
        ProductSpaceState(PartitionState.root(tree), np.array([1.0 + 2.0j, 0.0]))


def test_neighbor_probabilities_reflect_asymmetric_degrees() -> None:
    """The 1x3 partition path should have proposal degrees one, two, and one."""
    tree, partitions = _one_by_three_partitions()

    neighbors = [enumerate_partition_neighbors(tree, partition) for partition in partitions]

    assert [len(items) for items in neighbors] == [1, 2, 1]
    assert [[math.exp(item.log_q) for item in items] for items in neighbors] == [
        [1.0],
        [0.5, 0.5],
        [1.0],
    ]
    for source, items in zip(partitions, neighbors, strict=True):
        for item in items:
            assert source in {
                reverse.partition for reverse in enumerate_partition_neighbors(tree, item.partition)
            }


def test_partition_metropolis_step_keeps_all_continuous_values_fixed() -> None:
    """An accepted structure update should not transport either coefficient block."""
    tree, (root, middle, _) = _one_by_three_partitions()
    current = ProductSpaceState(root, np.array([0.2, -0.4, 0.7]), np.array([1.3, -0.2]))

    transition = partition_metropolis_step(
        tree,
        current,
        log_density=lambda state: 100.0 if state.partition == middle else 0.0,
        rng=np.random.default_rng(3),
    )

    assert transition.accepted
    assert transition.state.partition == middle
    np.testing.assert_array_equal(transition.state.inner_coordinates, current.inner_coordinates)
    np.testing.assert_array_equal(transition.state.outer_coefficients, current.outer_coefficients)


def test_partition_metropolis_step_rejects_negative_infinite_candidate() -> None:
    """A zero-density candidate should be rejected without altering the state."""
    tree, (root, _, _) = _one_by_three_partitions()
    current = ProductSpaceState(root, np.zeros(3))

    transition = partition_metropolis_step(
        tree,
        current,
        log_density=lambda state: 0.0 if state.partition == root else -math.inf,
        rng=np.random.default_rng(4),
    )

    assert not transition.accepted
    assert transition.state is current
    assert transition.candidate.partition != root
    assert transition.log_acceptance_ratio == -math.inf


def test_partition_metropolis_step_obeys_detailed_balance_on_each_edge() -> None:
    """Target, proposal, and acceptance masses should match in both directions."""
    tree, partitions = _one_by_three_partitions()
    unnormalized = {
        partitions[0]: 1.0,
        partitions[1]: 2.0,
        partitions[2]: 5.0,
    }

    for source in partitions:
        for forward in enumerate_partition_neighbors(tree, source):
            destination = forward.partition
            reverse = next(
                item for item in enumerate_partition_neighbors(tree, destination) if item.partition == source
            )
            forward_ratio = (
                math.log(unnormalized[destination])
                - math.log(unnormalized[source])
                + reverse.log_q
                - forward.log_q
            )
            reverse_ratio = -forward_ratio
            forward_mass = unnormalized[source] * math.exp(forward.log_q) * min(1.0, math.exp(forward_ratio))
            reverse_mass = (
                unnormalized[destination] * math.exp(reverse.log_q) * min(1.0, math.exp(reverse_ratio))
            )
            assert forward_mass == pytest.approx(reverse_mass)


def test_one_grid_cell_tree_returns_an_isolated_transition() -> None:
    """A tree with one grid cell has no legal structure proposal."""
    tree = DyadicTree.from_shape((1, 1))
    current = ProductSpaceState(PartitionState.root(tree), np.zeros(1))

    transition = partition_metropolis_step(
        tree,
        current,
        log_density=lambda state: 0.0,
        rng=np.random.default_rng(1),
    )

    assert transition.state is current
    assert transition.candidate is current
    assert transition.move is None
    assert not transition.accepted


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_current_log_density_must_be_finite(value: float) -> None:
    """The source state must have a finite augmented target density."""
    tree = DyadicTree.from_shape((1, 2))
    current = ProductSpaceState(PartitionState.root(tree), np.zeros(2))

    with pytest.raises(ValueError, match="current log density"):
        partition_metropolis_step(
            tree,
            current,
            log_density=lambda state: value,
            rng=np.random.default_rng(1),
        )
