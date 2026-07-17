"""Tests for the fixed-count Gaussian DFS demo orchestration."""

from __future__ import annotations

import numpy as np
import pytest

import openghg_inversions.basis.experimental.dyadic.demo_runner as demo_runner
from openghg_inversions.basis.experimental.dyadic.demo_runner import (
    DemoSearchConfig,
    run_fixed_count_dfs_search,
)


def test_demo_runner_is_reproducible_and_preserves_fixed_region_count() -> None:
    """A seeded synthetic run should replay and retain a valid fixed K."""
    rng = np.random.default_rng(12)
    grid = rng.normal(size=(8, 4, 5))
    variances = np.linspace(0.5, 1.2, grid.shape[0])
    config = DemoSearchConfig(
        target_regions=6,
        iterations=25,
        pilot_proposals=12,
        tau=0.8,
        seed=44,
        record_every=4,
    )

    first = run_fixed_count_dfs_search(grid, variances, config)
    second = run_fixed_count_dfs_search(grid, variances, config)

    assert first.result == second.result
    assert first.pilot_losses == second.pilot_losses
    assert len(first.initial_state.active) == 6
    assert len(first.result.final_state.active) == 6
    assert len(first.result.best_state.active) == 6
    assert first.result.best_score >= first.result.initial_score
    first.result.final_state.validate(first.tree)
    first.result.best_state.validate(first.tree)


def test_demo_runner_aggregates_partial_cell_support_for_proxy_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The initializer proxy should receive fine-cell support summed by tile."""
    captured_support: list[np.ndarray] = []

    def capture_scores(
        design: np.ndarray,
        precision: np.ndarray,
        support: np.ndarray,
    ) -> np.ndarray:
        """Capture candidate support and return deterministic zero priorities."""
        del precision
        captured_support.append(support.copy())
        return np.zeros(design.shape[1])

    monkeypatch.setattr(demo_runner, "prototype_quadratic_tile_scores", capture_scores)
    support_grid = np.array([[4.0, 2.0], [2.0, 1.0]])
    run = run_fixed_count_dfs_search(
        np.ones((3, 2, 2)),
        np.ones(3),
        DemoSearchConfig(target_regions=2, iterations=1, pilot_proposals=1),
        support_grid=support_grid,
    )

    assert len(captured_support) == 1
    assert captured_support[0][run.tree.root_id] == support_grid.sum()
    for node_id in run.tree.leaf_ids:
        tile = run.tree.tile(node_id)
        assert captured_support[0][node_id] == support_grid[tile.row_start, tile.col_start]


def test_demo_runner_rejects_invalid_inputs_and_configuration() -> None:
    """Malformed grids, variances, and impossible K values should fail clearly."""
    with pytest.raises(ValueError, match="target_regions"):
        DemoSearchConfig(target_regions=1)
    with pytest.raises(ValueError, match="contribution_grid"):
        run_fixed_count_dfs_search(np.ones((2, 3)), np.ones(2), DemoSearchConfig(target_regions=2))
    with pytest.raises(ValueError, match="r_diag"):
        run_fixed_count_dfs_search(np.ones((2, 2, 2)), [1.0], DemoSearchConfig(target_regions=2))
    with pytest.raises(ValueError, match="exceeds"):
        run_fixed_count_dfs_search(np.ones((2, 2, 2)), np.ones(2), DemoSearchConfig(target_regions=5))
    with pytest.raises(ValueError, match="support_grid"):
        run_fixed_count_dfs_search(
            np.ones((2, 2, 2)),
            np.ones(2),
            DemoSearchConfig(target_regions=2),
            support_grid=np.ones((3, 2)),
        )
