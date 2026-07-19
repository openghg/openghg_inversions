"""Tests for the framework-independent Gamma--Beta product-space target."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from openghg_inversions.basis.experimental.dyadic.gamma_beta import (
    DepthKappaStrategy,
    GammaBetaForest,
    GammaBetaGroupSpec,
)
from openghg_inversions.basis.experimental.dyadic.gamma_beta_product_space import (
    GammaBetaProductSpaceTarget,
)


def _target() -> GammaBetaProductSpaceTarget:
    """Return a two-group target with one optional positive split."""
    forest = GammaBetaForest.from_groups(
        np.array([[1.0, 1.0, 2.0]]),
        [
            GammaBetaGroupSpec(
                "inner",
                np.array([[True, True, False]]),
                root_variance=0.25,
                max_depth=1,
            ),
            GammaBetaGroupSpec(
                "outer",
                np.array([[False, False, True]]),
                root_variance=0.0,
                max_depth=0,
            ),
        ],
        require_full_coverage=True,
    )
    return GammaBetaProductSpaceTarget.from_grid(
        observations=np.array([3.0, -0.5]),
        observation_mean=np.array([0.2, -0.1]),
        finest_grid_design=np.array(
            [
                [[1.0, 2.0, 0.5]],
                [[-0.5, 1.0, 2.0]],
            ]
        ),
        forest=forest,
        kappa_strategy=DepthKappaStrategy(base_kappa=4.0),
        observation_covariance=np.array([[0.25, 0.05], [0.05, 0.5]]),
    )


def test_prediction_matches_direct_fine_grid_calculation() -> None:
    """Active-node sums should equal the rendered finest-grid prediction."""
    target = _target()
    forest = target.coordinate_layout.forest
    roots = np.array([1.2, 1.0])
    fractions = np.array([0.35])
    node_scalings = target.coordinate_layout.node_scalings(roots, fractions)
    active = forest.leaf_ids
    grid_scalings = target.coordinate_layout.render_frontier_scalings(
        active,
        node_scalings,
    )
    finest_design = np.array(
        [
            [[1.0, 2.0, 0.5]],
            [[-0.5, 1.0, 2.0]],
        ]
    )

    actual = target.prediction(active, roots, fractions)
    expected = target.observation_mean + np.einsum(
        "oij,ij->o",
        finest_design,
        grid_scalings,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)


def test_unsplit_prediction_ignores_inactive_beta_fraction() -> None:
    """A parent frontier must not depend on an inactive descendant fraction."""
    target = _target()
    forest = target.coordinate_layout.forest

    first = target.prediction(forest.root_ids, np.array([0.8, 1.0]), np.array([0.1]))
    second = target.prediction(forest.root_ids, np.array([0.8, 1.0]), np.array([0.9]))

    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_normalized_log_likelihood_matches_scipy() -> None:
    """The NumPy oracle should include the full Gaussian normalization."""
    target = _target()
    active = target.coordinate_layout.forest.leaf_ids
    roots = np.array([1.2, 1.0])
    fractions = np.array([0.35])
    prediction = target.prediction(active, roots, fractions)

    actual = target.log_likelihood(active, roots, fractions)
    expected = multivariate_normal.logpdf(
        target.observations,
        mean=prediction,
        cov=cast(Any, target.observation_covariance),
    )

    assert actual == pytest.approx(expected, abs=1.0e-12)


def test_builder_accepts_vector_standard_deviations_and_freezes_arrays() -> None:
    """A diagonal target should validate vectors and expose immutable arrays."""
    base = _target()
    target = GammaBetaProductSpaceTarget.from_grid(
        observations=np.array([1.0, 2.0]),
        finest_grid_design=np.ones((2, 1, 3)),
        forest=base.coordinate_layout.forest,
        kappa_strategy=DepthKappaStrategy(),
        observation_sd=np.array([0.5, 2.0]),
    )

    np.testing.assert_array_equal(target.observation_covariance, np.diag([0.25, 4.0]))
    assert not target.observations.flags.writeable
    assert not target.node_design.flags.writeable


def test_target_rejects_invalid_covariance_and_observation_counts() -> None:
    """Covariance and finest-grid rows must agree with observations."""
    base = _target()
    arguments = {
        "observations": np.array([1.0, 2.0]),
        "finest_grid_design": np.ones((2, 1, 3)),
        "forest": base.coordinate_layout.forest,
        "kappa_strategy": DepthKappaStrategy(),
    }

    with pytest.raises(ValueError, match="positive definite"):
        GammaBetaProductSpaceTarget.from_grid(
            **arguments,
            observation_covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
        )
    with pytest.raises(ValueError, match="same observation count"):
        GammaBetaProductSpaceTarget.from_grid(
            **{**arguments, "finest_grid_design": np.ones((3, 1, 3))},
        )
