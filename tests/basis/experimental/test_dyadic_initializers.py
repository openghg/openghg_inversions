"""Tests for deterministic and random dyadic partition initializers."""

from __future__ import annotations

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.initializers import (
    greedy_partition,
    random_partition,
    threshold_partition,
)
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree


def test_greedy_partition_reaches_exact_target_with_stable_ties() -> None:
    """Equal greedy gains should be resolved by increasing node identifier."""
    tree = DyadicTree.from_shape((4, 4))

    result = greedy_partition(tree, target_regions=4, split_gain=lambda node_id: 1.0)

    assert result.split_history == (0, 1, 2)
    assert len(result.state.active) == 4
    result.state.validate(tree)


def test_greedy_partition_uses_highest_available_gain() -> None:
    """Greedy growth should reconsider child gains after every selected split."""
    tree = DyadicTree.from_shape((4, 4))
    first_child, second_child = tree.children(tree.root_id)
    favoured_grandchild = tree.children(first_child)[1]
    gains = {first_child: 2.0, second_child: 1.0, favoured_grandchild: 3.0}

    result = greedy_partition(
        tree,
        target_regions=4,
        split_gain=lambda node_id: gains.get(node_id, 0.0),
    )

    assert result.split_history == (tree.root_id, first_child, favoured_grandchild)
    assert len(result.state.active) == 4


def test_random_partition_is_reproducible_for_generator_seed() -> None:
    """Fresh generators with the same seed should produce identical growth."""
    tree = DyadicTree.from_shape((4, 4))

    first = random_partition(tree, 9, np.random.default_rng(174))
    second = random_partition(tree, 9, np.random.default_rng(174))

    assert first == second
    assert len(first.state.active) == 9
    first.state.validate(tree)


def test_threshold_partition_scores_tiles_and_uses_strict_comparison() -> None:
    """Threshold growth should split scored tiles only while score is greater."""
    tree = DyadicTree.from_shape((4, 4))
    first_child, second_child = tree.children(tree.root_id)
    second_grandchild = tree.children(second_child)[1]
    scores = {
        tree.root_id: 10.0,
        first_child: 5.0,
        second_child: 6.0,
        second_grandchild: 7.0,
    }

    result = threshold_partition(
        tree,
        tile_score=lambda tile: scores.get(tile.node_id, 0.0),
        threshold=5.0,
    )

    assert result.split_history == (tree.root_id, second_child, second_grandchild)
    assert len(result.state.active) == 4
    result.state.validate(tree)


@pytest.mark.parametrize("target_regions", [0, -1, 17])
def test_greedy_partition_rejects_impossible_targets(target_regions: int) -> None:
    """Greedy growth should reject empty or over-capacity targets."""
    with pytest.raises(ValueError, match="target_regions|Cannot construct"):
        greedy_partition(DyadicTree.from_shape((4, 4)), target_regions, lambda node_id: 0.0)


@pytest.mark.parametrize("target_regions", [0, -1, 17])
def test_random_partition_rejects_impossible_targets(target_regions: int) -> None:
    """Random growth should reject empty or over-capacity targets."""
    with pytest.raises(ValueError, match="target_regions|Cannot construct"):
        random_partition(DyadicTree.from_shape((4, 4)), target_regions, np.random.default_rng(1))
