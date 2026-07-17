"""Tests for reusable dyadic sweep selection and resolution diagnostics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.rhime_gaussian import RHIMEGaussianMultiscale
from openghg_inversions.basis.experimental.dyadic.sweep_diagnostics import (
    CoarseningResolutionSummary,
    TemporalSelection,
    blocked_temporal_selection,
    native_cell_dfs,
    summarize_coarsening_resolution,
)


def _small_model() -> RHIMEGaussianMultiscale:
    """Build a small model with supported and unsupported native cells."""
    design = np.array(
        [
            [[1.0, 0.0, 2.0], [0.5, 0.0, 1.0]],
            [[0.2, 0.0, 0.3], [1.5, 0.0, 0.4]],
            [[0.7, 0.0, 1.2], [0.1, 0.0, 0.8]],
        ]
    )
    support = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    return RHIMEGaussianMultiscale.from_native_grid(
        design,
        support,
        [0.6, 0.8, 1.1],
        coarsen_factor=2,
        relative_prior_sd=0.7,
    )


def _factor_eight_model() -> RHIMEGaussianMultiscale:
    """Build a lightweight 293 by 391 model with one DFS-dominant cell."""
    design = np.zeros((1, 293, 391), dtype=float)
    design[0, 100, 200] = 1.0
    return RHIMEGaussianMultiscale.from_native_grid(
        design,
        np.ones((293, 391), dtype=float),
        [1.0],
        coarsen_factor=8,
        relative_prior_sd=1.0,
    )


def _temporal_rows() -> tuple[np.ndarray, np.ndarray]:
    """Return two sites sampled on one shared ten-hour wall-clock grid."""
    hourly = np.datetime64("2019-01-01T00:00:00", "ns") + np.arange(10) * np.timedelta64(1, "h")
    return np.repeat(np.array(["MHD", "TAC"]), 10), np.tile(hourly, 2)


def test_blocked_temporal_selection_applies_buffer_and_shared_stride() -> None:
    """A common block, symmetric buffer, and wall-clock stride should compose exactly."""
    sites, timestamps = _temporal_rows()

    selection = blocked_temporal_selection(
        sites,
        timestamps,
        holdout_start=np.datetime64("2019-01-01T04:00:00"),
        holdout_stop=np.datetime64("2019-01-01T06:00:00"),
        buffer_hours=1,
        thinning_hours=2,
    )

    hours = timestamps.astype("datetime64[h]").astype(int) % 24
    np.testing.assert_array_equal(selection.holdout_mask, np.isin(hours, [4, 5]))
    np.testing.assert_array_equal(selection.training_mask, np.isin(hours, [0, 2, 8]))
    assert selection.training_count == 6
    assert selection.holdout_count == 4
    assert selection.total_count == 20
    assert selection.buffer_excluded_count == 4
    assert selection.thinning_excluded_count == 6
    assert selection.stride_anchor == np.datetime64("2019-01-01T00:00:00", "ns")
    assert not selection.training_mask.flags.writeable
    assert not selection.holdout_mask.flags.writeable


def test_blocked_temporal_selection_rejects_shape_datetime_and_duplicate_errors() -> None:
    """Temporal inputs must be aligned datetime rows with unique site/time pairs."""
    sites, timestamps = _temporal_rows()
    start = np.datetime64("2019-01-01T04:00:00")
    stop = np.datetime64("2019-01-01T06:00:00")

    with pytest.raises(ValueError, match="same shape"):
        blocked_temporal_selection(sites[:-1], timestamps, holdout_start=start, holdout_stop=stop)
    with pytest.raises(ValueError, match="datetime dtype"):
        blocked_temporal_selection(sites, np.arange(sites.size), holdout_start=start, holdout_stop=stop)
    with pytest.raises(ValueError, match="NaT"):
        invalid_times = timestamps.copy()
        invalid_times[0] = np.datetime64("NaT")
        blocked_temporal_selection(sites, invalid_times, holdout_start=start, holdout_stop=stop)
    with pytest.raises(ValueError, match="unique"):
        duplicate_sites = sites.copy()
        duplicate_times = timestamps.copy()
        duplicate_sites[1] = duplicate_sites[0]
        duplicate_times[1] = duplicate_times[0]
        blocked_temporal_selection(
            duplicate_sites,
            duplicate_times,
            holdout_start=start,
            holdout_stop=stop,
        )


def test_blocked_temporal_selection_rejects_invalid_ranges_and_empty_masks() -> None:
    """Invalid interval controls and selections emptied by them should fail clearly."""
    sites, timestamps = _temporal_rows()

    with pytest.raises(ValueError, match="earlier"):
        blocked_temporal_selection(
            sites,
            timestamps,
            holdout_start=np.datetime64("2019-01-01T06:00:00"),
            holdout_stop=np.datetime64("2019-01-01T06:00:00"),
        )
    with pytest.raises(ValueError, match="non-negative"):
        blocked_temporal_selection(
            sites,
            timestamps,
            holdout_start=np.datetime64("2019-01-01T04:00:00"),
            holdout_stop=np.datetime64("2019-01-01T06:00:00"),
            buffer_hours=-1,
        )
    with pytest.raises(ValueError, match="positive integer"):
        blocked_temporal_selection(
            sites,
            timestamps,
            holdout_start=np.datetime64("2019-01-01T04:00:00"),
            holdout_stop=np.datetime64("2019-01-01T06:00:00"),
            thinning_hours=0,
        )
    with pytest.raises(TypeError, match="positive integer"):
        blocked_temporal_selection(
            sites,
            timestamps,
            holdout_start=np.datetime64("2019-01-01T04:00:00"),
            holdout_stop=np.datetime64("2019-01-01T06:00:00"),
            thinning_hours=1.5,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="holdout selection is empty"):
        blocked_temporal_selection(
            sites,
            timestamps,
            holdout_start=np.datetime64("2019-01-02T04:00:00"),
            holdout_stop=np.datetime64("2019-01-02T06:00:00"),
        )
    with pytest.raises(ValueError, match="training selection is empty"):
        blocked_temporal_selection(
            sites,
            timestamps,
            holdout_start=np.datetime64("2019-01-01T04:00:00"),
            holdout_stop=np.datetime64("2019-01-01T06:00:00"),
            buffer_hours=24,
        )


def test_temporal_selection_defensively_rejects_overlap() -> None:
    """The immutable mask container should reject overlapping memberships."""
    with pytest.raises(ValueError, match="must not overlap"):
        TemporalSelection(
            training_mask=np.array([True, False]),
            holdout_mask=np.array([True, False]),
            holdout_start=np.datetime64("2019-01-01T00:00:00"),
            holdout_stop=np.datetime64("2019-01-01T01:00:00"),
            buffer_hours=0.0,
            thinning_hours=None,
            stride_anchor=None,
            buffer_excluded_count=0,
            thinning_excluded_count=0,
        )


def test_native_cell_dfs_matches_dense_formula_and_full_grid_total() -> None:
    """Batched cell contributions should match dense solves and close to native DFS."""
    model = _small_model()
    contributions = native_cell_dfs(model, batch_size=2)
    flattened = model.native_design.reshape(model.native_design.shape[0], -1)
    expected = model.relative_prior_sd**2 * np.einsum(
        "ij,ij->j",
        flattened,
        np.linalg.solve(model.innovation_covariance, flattened),
    )

    np.testing.assert_allclose(contributions.ravel(), expected, atol=1e-14)
    np.testing.assert_array_equal(contributions[~model.native_support], 0.0)
    np.testing.assert_allclose(np.sum(contributions), model.full_grid_dfs, atol=1e-14)


def test_native_cell_dfs_rejects_invalid_batch_size_and_broken_closure() -> None:
    """Batch controls and model full-grid closure are validated explicitly."""
    model = _small_model()

    with pytest.raises(ValueError, match="positive integer"):
        native_cell_dfs(model, batch_size=0)
    with pytest.raises(TypeError, match="positive integer"):
        native_cell_dfs(model, batch_size=1.5)  # type: ignore[arg-type]
    with pytest.raises(ArithmeticError, match="do not close"):
        native_cell_dfs(replace(model, full_grid_dfs=model.full_grid_dfs + 0.1))


def test_factor_eight_summary_maps_native_edges_and_exposes_resolution_loss() -> None:
    """Factor-eight leaves should map a 293 by 391 grid including both partial edges."""
    model = _factor_eight_model()
    contributions = native_cell_dfs(model, batch_size=257)
    summary = summarize_coarsening_resolution(model, contributions)
    expected_leaf_dfs = float(np.sum(model.tile_scores[list(model.design.tree.leaf_ids)]))

    assert isinstance(summary, CoarseningResolutionSummary)
    assert summary.search_shape == (37, 49)
    assert summary.ordinary_block_width == 8
    assert summary.partial_final_row_height == 5
    assert summary.partial_final_column_width == 7
    np.testing.assert_allclose(summary.full_grid_dfs, 0.5, atol=1e-14)
    np.testing.assert_allclose(summary.all_search_leaves_dfs, expected_leaf_dfs, atol=1e-14)
    np.testing.assert_allclose(summary.all_leaf_retained_fraction, 1.0 / 64.0, atol=1e-14)
    np.testing.assert_allclose(
        summary.unresolved_dfs,
        summary.full_grid_dfs - summary.all_search_leaves_dfs,
        atol=1e-14,
    )
    assert (summary.top_native_cell_row, summary.top_native_cell_column) == (100, 200)
    np.testing.assert_allclose(summary.top_native_cell_dfs, summary.full_grid_dfs, atol=1e-14)
    assert summary.top_native_cell_fraction == 1.0
    assert summary.top_ten_native_cell_fraction == 1.0
    assert summary.block_dominant_cell_fraction == 1.0
    assert summary.maximum_within_nonzero_block_cell_fraction == 1.0
    assert summary.supported_native_cell_count == 293 * 391
    assert summary.all_search_leaves_dfs < summary.top_native_cell_dfs


def test_coarsening_summary_rejects_bad_shape_support_values_and_bounds() -> None:
    """Resolution summaries require native-shaped, supported, closing DFS values."""
    model = _small_model()
    contributions = native_cell_dfs(model)

    with pytest.raises(ValueError, match="shape"):
        summarize_coarsening_resolution(model, contributions[:, :-1])
    with pytest.raises(ValueError, match="non-negative"):
        negative = contributions.copy()
        negative[0, 0] = -1.0
        summarize_coarsening_resolution(model, negative)
    with pytest.raises(ValueError, match="outside"):
        unsupported = contributions.copy()
        unsupported[0, 1] = 1.0
        summarize_coarsening_resolution(model, unsupported)
    with pytest.raises(ArithmeticError, match="does not close"):
        summarize_coarsening_resolution(model, contributions * 0.5)
    with pytest.raises(ArithmeticError, match="exceeds"):
        excessive_scores = model.tile_scores.copy()
        excessive_scores[list(model.design.tree.leaf_ids)] = model.full_grid_dfs
        summarize_coarsening_resolution(replace(model, tile_scores=excessive_scores), contributions)
