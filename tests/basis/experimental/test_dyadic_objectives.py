"""Tests for Gaussian DFS and the historical quadratic design proxy."""

from __future__ import annotations

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.multiscale import MultiscaleDesign
from openghg_inversions.basis.experimental.dyadic.objectives import (
    GaussianDFSObjective,
    IsotropicRegionCovariance,
    direct_observation_space_dfs,
    gaussian_dfs,
    isotropic_observation_space_dfs,
    prototype_quadratic_tile_scores,
)
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree


def test_sum_then_square_differs_from_summing_cell_scores() -> None:
    """Opposing cell signals should cancel before the prototype square."""
    fine_cell_columns = np.array([[1.0, -1.0]])
    tile_column = fine_cell_columns.sum(axis=1, keepdims=True)

    tile_score = prototype_quadratic_tile_scores(tile_column, [1.0], [2.0])
    fine_scores = prototype_quadratic_tile_scores(fine_cell_columns, [1.0], [1.0, 1.0])

    np.testing.assert_array_equal(tile_score, [0.0])
    assert fine_scores.sum() == 2.0


def test_gaussian_dfs_formulas_agree() -> None:
    """State-space and direct observation-space Gaussian formulas should agree."""
    design = np.array([[1.0, 0.3], [-0.2, 1.4], [0.7, -0.5]])
    covariance = np.array([[1.8, 0.4], [0.4, 0.9]])
    r_diag = np.array([0.5, 1.2, 0.8])

    state_space = gaussian_dfs(design, covariance, r_diag)
    observation_space = direct_observation_space_dfs(design, covariance, r_diag)

    assert state_space == pytest.approx(observation_space, rel=1e-12, abs=1e-12)
    assert 0.0 <= state_space <= design.shape[1]


def test_isotropic_observation_space_dfs_avoids_large_identity_matrix() -> None:
    """Specialized isotropic DFS should match the general observation-space formula."""
    design = np.array([[1.0, 0.3, -0.4], [-0.2, 1.4, 0.6]])
    r_diag = np.array([0.5, 1.2])
    tau = 1.7

    expected = direct_observation_space_dfs(design, tau**2 * np.eye(3), r_diag)
    actual = isotropic_observation_space_dfs(design, r_diag, tau)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_gaussian_dfs_is_invariant_to_state_permutation() -> None:
    """Permuting design columns and both covariance axes should preserve DFS."""
    design = np.array([[1.0, 0.3, -0.1], [-0.2, 1.4, 0.6], [0.7, -0.5, 1.1]])
    covariance = np.array([[1.8, 0.4, 0.2], [0.4, 0.9, -0.1], [0.2, -0.1, 1.2]])
    r_diag = np.array([0.5, 1.2, 0.8])
    permutation = np.array([2, 0, 1])

    expected = gaussian_dfs(design, covariance, r_diag)
    actual = gaussian_dfs(
        design[:, permutation],
        covariance[np.ix_(permutation, permutation)],
        r_diag,
    )

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_covariance_builder_is_called_for_each_state_and_design() -> None:
    """Objective evaluation should rebuild covariance on every score call."""
    tree = DyadicTree.from_shape((2, 2))
    design = MultiscaleDesign.from_grid(np.arange(8.0).reshape(2, 2, 2), tree)
    first_state = PartitionState.root(tree)
    second_state = first_state.split(tree, tree.root_id)
    calls: list[tuple[PartitionState, MultiscaleDesign]] = []

    def covariance_builder(state: PartitionState, current_design: MultiscaleDesign) -> np.ndarray:
        """Record each call and return state-specific isotropic covariance."""
        calls.append((state, current_design))
        return np.eye(len(state.ordered_active())) * (1.0 + len(calls))

    objective = GaussianDFSObjective([0.5, 0.8], covariance_builder)

    objective.score(first_state, design)
    objective(second_state, design)

    assert calls == [(first_state, design), (second_state, design)]


def test_isotropic_covariance_is_an_explicit_positive_benchmark() -> None:
    """The isotropic builder should return tau-squared on each active diagonal."""
    builder = IsotropicRegionCovariance(tau=1.5)
    tree = DyadicTree.from_shape((2, 2))
    design = MultiscaleDesign.from_grid(np.ones((1, 2, 2)), tree)
    state = PartitionState(frozenset({1, 5, 6}))

    covariance = builder(state, design)

    np.testing.assert_array_equal(covariance, 2.25 * np.eye(3))
    with pytest.raises(ValueError, match="positive"):
        IsotropicRegionCovariance(tau=0.0)


def test_gaussian_dfs_rejects_invalid_inputs() -> None:
    """Gaussian DFS should reject incompatible, non-finite, or non-SPD inputs."""
    valid_design = np.ones((2, 2))
    valid_covariance = np.eye(2)
    with pytest.raises(ValueError, match="two-dimensional"):
        gaussian_dfs(np.ones(2), valid_covariance, np.ones(2))
    with pytest.raises(ValueError, match="shape"):
        gaussian_dfs(valid_design, np.eye(3), np.ones(2))
    with pytest.raises(ValueError, match="positive"):
        gaussian_dfs(valid_design, valid_covariance, [1.0, 0.0])
    with pytest.raises(ValueError, match="symmetric"):
        gaussian_dfs(valid_design, [[1.0, 1.0], [0.0, 1.0]], np.ones(2))
    with pytest.raises(ValueError, match="positive definite"):
        gaussian_dfs(valid_design, [[1.0, 2.0], [2.0, 1.0]], np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        gaussian_dfs([[1.0, np.nan], [0.0, 1.0]], valid_covariance, np.ones(2))


def test_prototype_scores_reject_invalid_precision_and_support() -> None:
    """The quadratic proxy should require explicit positive matching vectors."""
    design = np.ones((2, 3))
    with pytest.raises(ValueError, match="length"):
        prototype_quadratic_tile_scores(design, [1.0], np.ones(3))
    with pytest.raises(ValueError, match="positive"):
        prototype_quadratic_tile_scores(design, np.ones(2), [1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="length"):
        prototype_quadratic_tile_scores(design, np.ones(2), np.ones(2))


def test_objective_rejects_invalid_observation_covariance() -> None:
    """Objective construction should reject non-vector or non-positive R diagonals."""
    builder = IsotropicRegionCovariance(1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        GaussianDFSObjective([[1.0]], builder)
    with pytest.raises(ValueError, match="positive"):
        GaussianDFSObjective([-1.0], builder)
