"""Tests for vectorized permanent Gamma-Beta forest coordinates."""

from __future__ import annotations

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.gamma_beta import (
    DepthKappaStrategy,
    GammaBetaForest,
    GammaBetaGroupSpec,
)
from openghg_inversions.basis.experimental.dyadic.gamma_beta_coordinates import (
    GammaBetaCoordinateLayout,
)


def _forest() -> GammaBetaForest:
    """Return one refinable and one fixed group on a tiny grid."""
    refinable = np.array([[True, True, True, True, False]])
    fixed = ~refinable
    return GammaBetaForest.from_groups(
        np.array([[1.0, 2.0, 3.0, 4.0, 2.0]]),
        [
            GammaBetaGroupSpec(
                "inner",
                refinable,
                root_variance=0.25,
                max_depth=2,
            ),
            GammaBetaGroupSpec(
                "outer",
                fixed,
                root_variance=0.0,
                max_depth=0,
            ),
        ],
        require_full_coverage=True,
    )


def test_vectorized_node_scalings_match_top_down_prior_samples() -> None:
    """Static path products should reproduce every sampled node scaling."""
    forest = _forest()
    strategy = DepthKappaStrategy(base_kappa=4.0, depth_multiplier=2.0)
    layout = GammaBetaCoordinateLayout.from_forest(
        forest,
        kappa_strategy=strategy,
    )
    samples = forest.sample(25, kappa_strategy=strategy, rng=3)
    roots_by_group = {
        group_index: next(
            node_id
            for node_id in forest.root_ids
            if forest.nodes[node_id].group_index == group_index
        )
        for group_index in range(len(forest.groups))
    }

    for draw in range(samples.draws):
        roots = np.array(
            [
                samples.node_scalings[draw, roots_by_group[group_index]]
                for group_index in range(len(forest.groups))
            ]
        )
        fractions = samples.split_fractions[draw, layout.internal_node_ids]
        np.testing.assert_allclose(
            layout.node_scalings(roots, fractions),
            samples.node_scalings[draw],
            rtol=1e-14,
            atol=1e-14,
        )


def test_path_matrices_encode_each_ancestor_branch_once() -> None:
    """Every non-root node should have one branch marker per ancestor split."""
    forest = _forest()
    layout = GammaBetaCoordinateLayout.from_forest(
        forest,
        kappa_strategy=DepthKappaStrategy(),
    )

    for node in forest.nodes:
        ancestor_count = 0
        parent_id = node.parent_id
        while parent_id is not None:
            ancestor_count += 1
            parent_id = forest.nodes[parent_id].parent_id
        encoded = layout.left_path[node.node_id] + layout.right_path[node.node_id]
        assert encoded.sum() == ancestor_count
        assert np.all(encoded <= 1)


def test_node_design_matches_direct_support_sums() -> None:
    """Every static node column should equal a direct grid-support sum."""
    forest = _forest()
    layout = GammaBetaCoordinateLayout.from_forest(
        forest,
        kappa_strategy=DepthKappaStrategy(),
    )
    grid = np.arange(15.0).reshape(3, 1, 5)
    design = layout.node_design(grid)
    flattened = grid.reshape(3, -1)

    for node in forest.nodes:
        np.testing.assert_array_equal(
            design[:, node.node_id],
            flattened[:, node.flat_indices].sum(axis=1),
        )


def test_rendered_full_leaf_frontier_matches_sample_grid() -> None:
    """Rendering maximum leaves should reproduce ``GammaBetaSamples.to_grid``."""
    forest = _forest()
    strategy = DepthKappaStrategy()
    layout = GammaBetaCoordinateLayout.from_forest(
        forest,
        kappa_strategy=strategy,
    )
    samples = forest.sample(2, kappa_strategy=strategy, rng=9)

    actual = layout.render_frontier_scalings(
        forest.leaf_ids,
        samples.node_scalings[0],
    )

    np.testing.assert_allclose(actual, samples.to_grid(0))


def test_coordinate_layout_rejects_invalid_values_and_frontiers() -> None:
    """Coordinate transforms should reject invalid fractions, roots, and coverage."""
    forest = _forest()
    layout = GammaBetaCoordinateLayout.from_forest(
        forest,
        kappa_strategy=DepthKappaStrategy(),
    )
    roots = np.ones(len(forest.groups))
    fractions = np.full(layout.split_count, 0.5)

    with pytest.raises(ValueError, match="strictly between"):
        layout.node_scalings(roots, np.zeros(layout.split_count))
    with pytest.raises(ValueError, match="must equal one"):
        layout.node_scalings(np.array([1.0, 2.0]), fractions)
    with pytest.raises(ValueError, match="cover"):
        layout.render_frontier_scalings((forest.root_ids[0],), np.ones(len(forest.nodes)))
    with pytest.raises(ValueError, match="overlap"):
        layout.render_frontier_scalings(
            (forest.root_ids[0], *forest.leaf_ids),
            np.ones(len(forest.nodes)),
        )
    with pytest.raises(ValueError, match="Unknown active forest node"):
        layout.render_frontier_scalings(
            (-1, *forest.root_ids),
            np.ones(len(forest.nodes)),
        )


def test_coordinate_layout_rejects_invalid_kappa() -> None:
    """A concentration strategy must return positive finite values."""
    forest = _forest()

    with pytest.raises(ValueError, match="invalid value"):
        GammaBetaCoordinateLayout.from_forest(
            forest,
            kappa_strategy=lambda context: np.nan,
        )
