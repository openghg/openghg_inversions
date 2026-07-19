"""Tests for canonical grouped Gamma--Beta forest partitions."""

from __future__ import annotations

from itertools import product
import math

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.gamma_beta import (
    GammaBetaForest,
    GammaBetaGroupSpec,
)
from openghg_inversions.basis.experimental.dyadic.gamma_beta_partition import (
    GammaBetaPartitionLayout,
    GammaBetaRegionCountPrior,
)


def _forest() -> GammaBetaForest:
    """Return one four-leaf tree plus one fixed group root."""
    return GammaBetaForest.from_groups(
        np.ones((1, 5)),
        [
            GammaBetaGroupSpec(
                "inner",
                np.array([[True, True, True, True, False]]),
                max_depth=2,
            ),
            GammaBetaGroupSpec(
                "outer",
                np.array([[False, False, False, False, True]]),
                max_depth=0,
            ),
        ],
        require_full_coverage=True,
    )


def _canonical_masks(layout: GammaBetaPartitionLayout) -> tuple[np.ndarray, ...]:
    """Brute-force every canonical mask for a tiny test forest."""
    result: list[np.ndarray] = []
    for bits in product((0, 1), repeat=layout.split_count):
        try:
            result.append(layout.canonical_split_mask(bits))
        except ValueError:
            continue
    return tuple(result)


def test_codec_round_trip_and_region_count_identity() -> None:
    """Every canonical mask should round-trip through its active frontier."""
    layout = GammaBetaPartitionLayout.from_forest(_forest())

    for mask in _canonical_masks(layout):
        active = layout.active_node_ids(mask)
        encoded = layout.split_mask_from_active(active)

        np.testing.assert_array_equal(encoded, mask)
        assert layout.region_count(mask) == len(active)
        assert layout.region_count(mask) == layout.minimum_regions + int(mask.sum())


def test_dynamic_programming_counts_match_brute_force() -> None:
    """Exact tree-polynomial counts should equal all tiny canonical masks."""
    layout = GammaBetaPartitionLayout.from_forest(_forest())
    brute_force = np.zeros(layout.maximum_regions + 1, dtype=np.int64)
    for mask in _canonical_masks(layout):
        brute_force[layout.region_count(mask)] += 1

    assert layout.minimum_regions == 2
    assert layout.maximum_regions == 5
    assert layout.partition_counts_by_k == (0, 0, 1, 1, 2, 1)
    np.testing.assert_array_equal(layout.partition_counts_by_k, brute_force)


def test_uniform_k_prior_is_normalized_and_uniform_by_k() -> None:
    """Summed partition mass should be one with equal mass at every K."""
    layout = GammaBetaPartitionLayout.from_forest(_forest())
    prior = GammaBetaRegionCountPrior.uniform_k(layout)
    mass_by_k = np.zeros(layout.maximum_regions + 1, dtype=float)

    for mask in _canonical_masks(layout):
        mass_by_k[layout.region_count(mask)] += math.exp(prior(mask))

    np.testing.assert_allclose(mass_by_k[2:], np.full(4, 0.25), atol=1.0e-15)
    assert mass_by_k.sum() == pytest.approx(1.0, abs=1.0e-15)
    assert not prior.log_probability_by_k.flags.writeable


def test_neighbors_are_unique_reversible_local_moves() -> None:
    """Every split/merge edge should have one reverse edge and valid log q."""
    layout = GammaBetaPartitionLayout.from_forest(_forest())

    for mask in _canonical_masks(layout):
        neighbors = layout.neighbors(mask)
        encoded = {tuple(move.split_mask.tolist()) for move in neighbors}
        assert len(encoded) == len(neighbors)
        for move in neighbors:
            assert abs(layout.region_count(move.split_mask) - layout.region_count(mask)) == 1
            assert move.log_q == pytest.approx(-math.log(len(neighbors)))
            reverse_neighbors = layout.neighbors(move.split_mask)
            reverse = [
                candidate
                for candidate in reverse_neighbors
                if np.array_equal(candidate.split_mask, mask)
            ]
            assert len(reverse) == 1
            assert reverse[0].kind != move.kind


def test_deterministic_initial_masks_cover_each_available_k() -> None:
    """Stable initialization should construct every supported region count."""
    layout = GammaBetaPartitionLayout.from_forest(_forest())

    for region_count in range(layout.minimum_regions, layout.maximum_regions + 1):
        mask = layout.initial_split_mask(region_count)
        assert layout.region_count(mask) == region_count


def test_codec_rejects_noncanonical_and_invalid_frontiers() -> None:
    """Descendant-only masks and incomplete or overlapping frontiers are invalid."""
    layout = GammaBetaPartitionLayout.from_forest(_forest())

    with pytest.raises(ValueError, match="ancestry closed"):
        layout.canonical_split_mask(np.array([0, 1, 0]))
    with pytest.raises(ValueError, match="cover"):
        layout.split_mask_from_active((layout.forest.root_ids[0],))
    with pytest.raises(ValueError, match="overlap"):
        layout.split_mask_from_active(
            (layout.forest.root_ids[0], *layout.forest.leaf_ids)
        )


def test_counts_use_python_integers_for_large_forests() -> None:
    """Dynamic-programming counts must not overflow fixed-width integers."""
    forest = GammaBetaForest.from_groups(
        np.ones((8, 16)),
        [GammaBetaGroupSpec("full", np.ones((8, 16), dtype=bool), max_depth=7)],
        require_full_coverage=True,
    )
    layout = GammaBetaPartitionLayout.from_forest(forest)

    assert all(isinstance(value, int) for value in layout.partition_counts_by_k)
    assert max(layout.partition_counts_by_k) > np.iinfo(np.int64).max
