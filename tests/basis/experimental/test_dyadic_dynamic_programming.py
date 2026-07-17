"""Tests for the exact additive-score dyadic dynamic-programming oracle."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic import (
    AdditivePartitionSolution,
    DyadicTree,
    PartitionState,
    additive_partition_frontier,
    optimal_additive_partition,
)


def _enumerate_subtree_frontiers(tree: DyadicTree, node_id: int) -> tuple[frozenset[int], ...]:
    """Enumerate all valid frontiers rooted at one node without using the oracle."""
    children = tree.children(node_id)
    if not children:
        return (frozenset({node_id}),)

    left_id, right_id = children
    split_frontiers = tuple(
        left | right
        for left in _enumerate_subtree_frontiers(tree, left_id)
        for right in _enumerate_subtree_frontiers(tree, right_id)
    )
    return (frozenset({node_id}), *split_frontiers)


def _brute_force_solution(
    tree: DyadicTree,
    tile_scores: np.ndarray,
    region_count: int,
) -> AdditivePartitionSolution:
    """Select the best enumerated frontier using the declared tie rule.

    Args:
        tree: Small tree whose frontiers should be exhaustively enumerated.
        tile_scores: Additive scores indexed by node ID.
        region_count: Exact number of active nodes required.

    Returns:
        Highest-scoring enumerated state, with lexicographic tie-breaking.
    """
    states = (
        PartitionState(active)
        for active in _enumerate_subtree_frontiers(tree, tree.root_id)
        if len(active) == region_count
    )
    scored_states = [
        AdditivePartitionSolution(
            state,
            float(sum(tile_scores[node_id] for node_id in state.ordered_active())),
        )
        for state in states
    ]
    return min(scored_states, key=lambda solution: (-solution.score, solution.state.ordered_active()))


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (2, 3)])
def test_frontier_matches_independent_brute_force(shape: tuple[int, int]) -> None:
    """Every score and state on a small-tree frontier should be globally optimal."""
    tree = DyadicTree.from_shape(shape)
    scores = np.random.default_rng(20260717).normal(size=len(tree.nodes))

    frontier = additive_partition_frontier(tree, scores, max_regions=len(tree.leaf_ids))

    assert tuple(frontier) == tuple(range(1, len(tree.leaf_ids) + 1))
    for region_count, solution in frontier.items():
        expected = _brute_force_solution(tree, scores, region_count)
        assert solution.state == expected.state
        assert solution.score == pytest.approx(expected.score)
        assert len(solution.state.active) == region_count
        solution.state.validate(tree)
        assert solution.score == pytest.approx(sum(scores[list(solution.state.ordered_active())]))


def test_requested_partition_matches_frontier_and_unpacks() -> None:
    """The requested-K helper should return the frontier state together with its score."""
    tree = DyadicTree.from_shape((2, 3))
    scores = np.linspace(-1.0, 2.0, num=len(tree.nodes))

    expected = additive_partition_frontier(tree, scores, max_regions=4)[4]
    result = optimal_additive_partition(tree, scores, target_regions=np.int64(4))
    state, score = result

    assert result == expected
    assert state == expected.state
    assert score == expected.score


def test_recurrence_distributes_regions_across_both_children() -> None:
    """A fixed count should use the globally best allocation between child subtrees."""
    tree = DyadicTree.from_shape((2, 2))
    scores = np.zeros(len(tree.nodes))
    left_id, right_id = tree.children(tree.root_id)
    left_children = tree.children(left_id)
    right_children = tree.children(right_id)
    scores[list(left_children)] = (5.0, 6.0)
    scores[list(right_children)] = (1.0, 2.0)

    solution = optimal_additive_partition(tree, scores, target_regions=3)

    assert solution.state.active == frozenset((*left_children, right_id))
    assert solution.score == 11.0


def test_equal_scores_have_deterministic_lexicographic_ties() -> None:
    """Repeated all-tie runs should choose the lexicographically first active IDs."""
    tree = DyadicTree.from_shape((2, 3))
    scores = np.zeros(len(tree.nodes))

    first = additive_partition_frontier(tree, scores, max_regions=len(tree.leaf_ids))
    second = additive_partition_frontier(tree, scores, max_regions=len(tree.leaf_ids))

    assert first == second
    for region_count, solution in first.items():
        expected = _brute_force_solution(tree, scores, region_count)
        assert solution == expected


@pytest.mark.parametrize("target_regions", [0, -1, 5])
def test_requested_partition_rejects_impossible_region_counts(target_regions: int) -> None:
    """The requested-K helper should reject empty, negative, and over-capacity counts."""
    tree = DyadicTree.from_shape((2, 2))

    with pytest.raises(ValueError, match="target_regions"):
        optimal_additive_partition(tree, np.zeros(len(tree.nodes)), target_regions)


@pytest.mark.parametrize("target_regions", [True, 1.5, "2"])
def test_requested_partition_rejects_noninteger_region_counts(target_regions: object) -> None:
    """Boolean and non-integral requested counts should not be coerced to integers."""
    tree = DyadicTree.from_shape((2, 2))

    with pytest.raises(TypeError, match="target_regions"):
        optimal_additive_partition(tree, np.zeros(len(tree.nodes)), target_regions)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_regions", [0, 5])
def test_frontier_rejects_impossible_maximum_counts(max_regions: int) -> None:
    """Frontier limits must lie within the region counts supplied by the tree."""
    tree = DyadicTree.from_shape((2, 2))

    with pytest.raises(ValueError, match="max_regions"):
        additive_partition_frontier(tree, np.zeros(len(tree.nodes)), max_regions)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_oracle_rejects_nonfinite_tile_scores(nonfinite: float) -> None:
    """NaN and infinite tile contributions must fail before the recurrence runs."""
    tree = DyadicTree.from_shape((2, 2))
    scores = np.zeros(len(tree.nodes))
    scores[tree.root_id] = nonfinite

    with pytest.raises(ValueError, match="finite"):
        additive_partition_frontier(tree, scores, max_regions=2)


@pytest.mark.parametrize(
    "tile_scores",
    [
        np.zeros(6),
        np.zeros((7, 1)),
        np.ones(7, dtype=complex) * 1j,
        ["not-a-score"] * 7,
    ],
)
def test_oracle_rejects_invalid_tile_score_arrays(tile_scores: object) -> None:
    """Scores must be a real numeric vector with exactly one entry per node."""
    tree = DyadicTree.from_shape((2, 2))

    with pytest.raises(ValueError, match="tile_scores"):
        additive_partition_frontier(tree, tile_scores, max_regions=2)


def test_oracle_rejects_noncanonical_tree_state() -> None:
    """A malformed tree container must not produce apparently valid partition states."""
    tree = DyadicTree.from_shape((2, 2))
    malformed = replace(tree, _leaf_ids=())

    with pytest.raises(ValueError, match="complete canonical"):
        additive_partition_frontier(malformed, np.zeros(len(tree.nodes)), max_regions=1)


def test_oracle_rejects_non_tree_input() -> None:
    """The tree contract should be checked explicitly before score processing."""
    with pytest.raises(TypeError, match="DyadicTree"):
        optimal_additive_partition(object(), [0.0], target_regions=1)  # type: ignore[arg-type]
