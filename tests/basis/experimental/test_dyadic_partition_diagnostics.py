"""Tests for Gaussian diagnostics on arbitrary labelled search partitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.partition_diagnostics import (
    GaussianPartitionDiagnostics,
    build_partition_diagnostics,
    emissions_compression_quality,
    gaussian_posterior_mean,
)
from openghg_inversions.basis.experimental.dyadic.rhime_gaussian import RHIMEGaussianMultiscale
from openghg_inversions.basis.experimental.dyadic.state import PartitionState


def _tiny_model() -> RHIMEGaussianMultiscale:
    """Build a small model containing one unsupported labelled search cell."""
    native_design = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]],
            [[0.0, 1.0, 2.0], [1.0, 3.0, 0.0]],
        ]
    )
    prior_flux = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    return RHIMEGaussianMultiscale.from_native_grid(
        native_design,
        prior_flux,
        [0.7, 1.1],
        coarsen_factor=1,
        relative_prior_sd=0.5,
    )


def _posterior_model() -> RHIMEGaussianMultiscale:
    """Build a four-observation model for held-out posterior checks."""
    native_design = np.array(
        [
            [[1.0, 0.2], [0.4, 0.7]],
            [[0.3, 1.1], [0.6, 0.1]],
            [[0.8, 0.5], [0.2, 1.0]],
            [[0.1, 0.4], [1.2, 0.3]],
        ]
    )
    return RHIMEGaussianMultiscale.from_native_grid(
        native_design,
        np.ones((2, 2)),
        [0.4, 0.6, 0.5, 0.8],
        coarsen_factor=1,
        relative_prior_sd=0.7,
    )


def test_arbitrary_labels_match_dense_grouped_covariance_formulas() -> None:
    """Grouped design, variance, scatter, DFS, and innovation should match dense formulas."""
    model = _tiny_model()
    labels = np.array([[10, 10, 30], [20, 30, 40]])
    diagnostics = build_partition_diagnostics(model, labels)

    expected_design = np.array([[3.0, 4.0, 8.0], [1.0, 1.0, 5.0]])
    expected_counts = np.array([2, 1, 2])
    expected_variances = 0.5**2 / expected_counts
    expected_signal = (expected_design * expected_variances) @ expected_design.T
    expected_aggregation = 0.5**2 * np.array([[2.5, 1.5], [1.5, 1.0]])
    expected_effective = np.diag([0.7, 1.1]) + expected_aggregation
    expected_dfs = np.trace(np.linalg.solve(model.innovation_covariance, expected_signal))

    np.testing.assert_array_equal(diagnostics.label_grid, labels)
    np.testing.assert_array_equal(diagnostics.supported_region_ids, [10, 20, 30])
    np.testing.assert_array_equal(diagnostics.supported_native_counts, expected_counts)
    np.testing.assert_allclose(diagnostics.regional_design, expected_design, atol=1e-14)
    np.testing.assert_allclose(diagnostics.prior_variances, expected_variances, atol=1e-14)
    np.testing.assert_allclose(diagnostics.reduced_signal_covariance, expected_signal, atol=1e-14)
    np.testing.assert_allclose(
        diagnostics.aggregation_error_covariance,
        expected_aggregation,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        diagnostics.effective_observation_covariance,
        expected_effective,
        atol=1e-14,
    )
    np.testing.assert_allclose(diagnostics.dfs, expected_dfs, atol=1e-14)
    np.testing.assert_allclose(
        diagnostics.effective_observation_covariance + diagnostics.reduced_signal_covariance,
        model.innovation_covariance,
        atol=1e-14,
    )
    assert not diagnostics.label_grid.flags.writeable
    assert 40 not in diagnostics.supported_region_ids


def test_dyadic_state_labels_match_existing_model_diagnostics() -> None:
    """Rendering a dyadic state as labels should preserve its score and covariances."""
    model = _posterior_model()
    tree = model.design.tree
    state = PartitionState.root(tree).split(tree, tree.root_id)
    first_child = tree.children(tree.root_id)[0]
    state = state.split(tree, first_child)

    diagnostics = GaussianPartitionDiagnostics.from_search_labels(model, state.to_labels(tree))
    expected_design, expected_variances = model.reduced_design_and_variance(state)

    np.testing.assert_allclose(diagnostics.regional_design, expected_design, atol=1e-14)
    np.testing.assert_allclose(diagnostics.prior_variances, expected_variances, atol=1e-14)
    np.testing.assert_allclose(
        diagnostics.reduced_signal_covariance,
        model.reduced_signal_covariance(state),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        diagnostics.aggregation_error_covariance,
        model.aggregation_error_covariance(state),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        diagnostics.effective_observation_covariance,
        model.effective_observation_covariance(state),
        atol=1e-14,
    )
    np.testing.assert_allclose(diagnostics.dfs, model.score(state), atol=1e-14)


def test_search_labels_expand_over_coarsened_partial_native_blocks() -> None:
    """Search labels should aggregate leaf columns and expanded native support at boundaries."""
    native_design = np.arange(1.0, 19.0).reshape(2, 3, 3)
    prior_flux = np.ones((3, 3))
    prior_flux[2, 2] = 0.0
    native_design[:, 2, 2] = 0.0
    model = RHIMEGaussianMultiscale.from_native_grid(
        native_design,
        prior_flux,
        [0.5, 0.9],
        coarsen_factor=2,
        relative_prior_sd=0.4,
    )

    diagnostics = build_partition_diagnostics(model, [[5, 5], [8, 9]])
    expected_design = np.column_stack(
        (native_design[:, :2, :].sum(axis=(1, 2)), native_design[:, 2, :2].sum(axis=1))
    )

    np.testing.assert_array_equal(diagnostics.supported_region_ids, [5, 8])
    np.testing.assert_array_equal(diagnostics.supported_native_counts, [6, 2])
    np.testing.assert_allclose(diagnostics.regional_design, expected_design, atol=1e-14)
    np.testing.assert_allclose(
        diagnostics.effective_observation_covariance + diagnostics.reduced_signal_covariance,
        model.innovation_covariance,
        atol=1e-13,
    )


def test_posterior_mean_matches_dense_solve_with_baselines_and_holdout() -> None:
    """Conjugate conditioning with site baselines and training rows should match a dense solve."""
    model = _posterior_model()
    diagnostics = build_partition_diagnostics(model, [[1, 1], [2, 2]])
    observations = np.array([2.4, 1.3, 2.1, 1.8])
    baseline_design = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    emission_mean = np.array([0.8, 1.2])
    baseline_mean = np.array([0.1, -0.2])
    baseline_variances = np.array([0.3, 0.4])
    training_mask = np.array([True, True, False, True])

    posterior_mean, predictions = gaussian_posterior_mean(
        diagnostics,
        observations,
        emission_prior_mean=emission_mean,
        baseline_design=baseline_design,
        baseline_prior_mean=baseline_mean,
        baseline_prior_variances=baseline_variances,
        training_subset=training_mask,
    )

    design = np.column_stack((diagnostics.regional_design, baseline_design))
    prior_mean = np.concatenate((emission_mean, baseline_mean))
    prior_variances = np.concatenate((diagnostics.prior_variances, baseline_variances))
    training_indices = np.flatnonzero(training_mask)
    training_design = design[training_indices]
    training_error = diagnostics.effective_observation_covariance[np.ix_(training_indices, training_indices)]
    inverse_error_design = np.linalg.solve(training_error, training_design)
    posterior_precision = np.diag(1.0 / prior_variances) + training_design.T @ inverse_error_design
    posterior_rhs = prior_mean / prior_variances + training_design.T @ np.linalg.solve(
        training_error,
        observations[training_indices],
    )
    expected_posterior = np.linalg.solve(posterior_precision, posterior_rhs)

    np.testing.assert_allclose(posterior_mean, expected_posterior, atol=1e-13)
    np.testing.assert_allclose(predictions, design @ expected_posterior, atol=1e-13)
    assert predictions.shape == observations.shape


def test_emissions_compression_quality_has_required_invariances() -> None:
    """Quality should be exact natively, bounded at root, label-invariant, and baseline-free."""
    model = _posterior_model()
    native = build_partition_diagnostics(model, [[1, 2], [3, 4]])
    root = build_partition_diagnostics(model, np.ones((2, 2), dtype=int))
    grouped = build_partition_diagnostics(model, [[1, 1], [2, 2]])
    permuted = build_partition_diagnostics(model, [[91, 91], [7, 7]])
    subset = np.array([0, 2, 3])

    assert emissions_compression_quality(model, native) == 1.0
    root_quality = emissions_compression_quality(model, root)
    assert 0.0 <= root_quality <= 1.0
    grouped_quality = emissions_compression_quality(model, grouped, observation_subset=subset)
    expected_quality = 1.0 - np.sum(
        np.diag(grouped.aggregation_error_covariance)[subset] / model.r_diag[subset]
    ) / np.sum(np.diag(model.full_signal_covariance)[subset] / model.r_diag[subset])
    np.testing.assert_allclose(grouped_quality, expected_quality, atol=1e-14)
    np.testing.assert_allclose(
        grouped_quality,
        emissions_compression_quality(model, permuted, observation_subset=subset),
        atol=1e-14,
    )

    baseline_design = np.column_stack(([1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]))
    gaussian_posterior_mean(
        grouped,
        [2.4, 1.3, 2.1, 1.8],
        baseline_design=baseline_design,
        baseline_prior_mean=[0.0, 0.0],
        baseline_prior_variances=[0.3, 0.4],
        training_subset=subset,
    )
    assert emissions_compression_quality(model, grouped, observation_subset=subset) == grouped_quality


def test_emissions_compression_quality_is_one_without_selected_signal() -> None:
    """A zero full-signal denominator should use the documented no-loss value one."""
    model = RHIMEGaussianMultiscale.from_native_grid(
        np.zeros((2, 1, 2)),
        np.ones((1, 2)),
        [0.5, 0.7],
        coarsen_factor=1,
    )
    diagnostics = build_partition_diagnostics(model, [[1, 1]])

    assert emissions_compression_quality(model, diagnostics, observation_subset=[1]) == 1.0


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (np.ones((2, 3, 1)), "two-dimensional"),
        (np.ones((1, 3)), "must match search-grid shape"),
        (np.array([[1.0, 1.5, 2.0], [2.0, 3.0, 4.0]]), "integral"),
        (np.array([[1, 0, 2], [2, 3, 4]]), "positive"),
        (np.array([[1.0, np.nan, 2.0], [2.0, 3.0, 4.0]]), "finite"),
        (np.array([[True, True, True], [True, True, True]]), "integral positive"),
    ],
)
def test_invalid_search_labels_are_rejected(labels: np.ndarray, message: str) -> None:
    """Malformed, non-integral, or uncovered search label grids should fail clearly."""
    with pytest.raises(ValueError, match=message):
        build_partition_diagnostics(_tiny_model(), labels)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_design": np.ones((4, 1))},
        {"baseline_prior_mean": [0.0], "baseline_prior_variances": [1.0]},
        {
            "baseline_design": np.ones((3, 1)),
            "baseline_prior_mean": [0.0],
            "baseline_prior_variances": [1.0],
        },
        {
            "baseline_design": np.ones((4, 2)),
            "baseline_prior_mean": [0.0],
            "baseline_prior_variances": [1.0, 1.0],
        },
        {
            "baseline_design": np.ones((4, 1)),
            "baseline_prior_mean": [0.0],
            "baseline_prior_variances": [0.0],
        },
        {"training_subset": [True, False]},
        {"training_subset": [0.0, 1.0]},
        {"training_subset": [0, 0]},
        {"training_subset": np.array([], dtype=int)},
    ],
)
def test_invalid_baseline_and_training_inputs_are_rejected(kwargs: dict[str, Any]) -> None:
    """Baseline priors and training subsets must be complete and dimensionally valid."""
    diagnostics = build_partition_diagnostics(_posterior_model(), [[1, 1], [2, 2]])

    with pytest.raises(ValueError):
        gaussian_posterior_mean(diagnostics, [2.4, 1.3, 2.1, 1.8], **kwargs)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: values[:-1],
        lambda values: np.array([1.0, np.nan, 2.0, 3.0]),
    ],
)
def test_invalid_observations_are_rejected(mutate: Callable[[np.ndarray], np.ndarray]) -> None:
    """Posterior conditioning should reject wrong-length or non-finite observations."""
    diagnostics = build_partition_diagnostics(_posterior_model(), [[1, 1], [2, 2]])
    observations = mutate(np.array([2.4, 1.3, 2.1, 1.8]))

    with pytest.raises(ValueError):
        gaussian_posterior_mean(diagnostics, observations)
