"""Tests for the fixed-count Gaussian DFS demo orchestration."""

from __future__ import annotations

from math import exp

import numpy as np
import pytest

import openghg_inversions.basis.experimental.dyadic.demo_runner as demo_runner
from openghg_inversions.basis.experimental.dyadic.demo_runner import (
    DemoSearchConfig,
    VariableKSearchConfig,
    excess_region_penalty,
    run_fixed_count_dfs_search,
    run_projected_variable_k_dfs_search,
    run_variable_k_dfs_search,
)
from openghg_inversions.basis.experimental.dyadic.proposals import MergeMove, SplitMove


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


def test_variable_k_runner_replays_and_uses_unpaired_moves() -> None:
    """Variable-K search should replay while allowing independent split and merge moves."""
    rng = np.random.default_rng(19)
    grid = rng.normal(size=(10, 4, 5))
    variances = np.linspace(0.7, 1.5, grid.shape[0])
    config = VariableKSearchConfig(
        initial_regions=5,
        free_regions=7,
        min_regions=2,
        max_regions=10,
        penalty_per_extra_region=0.05,
        paired_move_probability=0.0,
        iterations=40,
        pilot_proposals=15,
        seed=71,
        record_every=3,
    )

    first = run_variable_k_dfs_search(grid, variances, config)
    second = run_variable_k_dfs_search(grid, variances, config)

    assert first.result == second.result
    assert first.pilot_losses == second.pilot_losses
    assert config.min_regions <= len(first.result.best_state.active) <= config.max_regions
    assert config.min_regions <= len(first.result.final_state.active) <= config.max_regions
    assert all(
        config.min_regions <= len(step.current_state.active) <= config.max_regions
        for step in first.result.trace
    )
    assert first.result.best_score >= first.result.initial_score
    assert first.best_dfs >= 0.0
    assert first.cellwise_isotropic_dfs >= 0.0
    assert all(isinstance(step.move, (SplitMove, MergeMove)) for step in first.result.trace)
    assert any(len(step.current_state.active) != config.initial_regions for step in first.result.trace)
    first.result.best_state.validate(first.tree)


def test_projected_variable_k_runner_replays_and_respects_native_dfs_bound() -> None:
    """Projected search should replay and remain below no-reduction DFS."""
    rng = np.random.default_rng(23)
    flux = np.array(
        [
            [0.0, 1.0, 2.0, 0.5],
            [0.0, 1.5, 0.7, 1.2],
            [0.0, 0.3, 1.1, 2.5],
            [0.0, 0.8, 0.6, 1.7],
        ]
    )
    footprints = rng.uniform(0.0, 1.0, size=(6, *flux.shape))
    grid = footprints * flux
    variances = np.linspace(0.8, 1.3, grid.shape[0])
    config = VariableKSearchConfig(
        initial_regions=3,
        free_regions=3,
        min_regions=2,
        max_regions=4,
        penalty_per_extra_region=0.02,
        paired_move_probability=0.2,
        iterations=25,
        pilot_proposals=10,
        tau=0.7,
        seed=81,
        record_every=2,
    )

    first = run_projected_variable_k_dfs_search(
        grid,
        flux,
        variances,
        config,
        coarsen_factor=2,
    )
    second = run_projected_variable_k_dfs_search(
        grid,
        flux,
        variances,
        config,
        coarsen_factor=2,
    )

    assert first.result == second.result
    assert first.pilot_losses == second.pilot_losses
    assert first.model.design.tree.shape == (2, 2)
    assert first.initial_dfs <= first.full_grid_dfs + 1e-12
    assert first.final_dfs <= first.full_grid_dfs + 1e-12
    assert first.best_dfs <= first.full_grid_dfs + 1e-12
    assert first.result.best_score >= first.result.initial_score
    first.result.best_state.validate(first.model.design.tree)


def test_excess_region_penalty_has_a_free_region_threshold() -> None:
    """The variable-K penalty should remain zero through the configured free K."""
    config = VariableKSearchConfig(free_regions=6, penalty_per_extra_region=0.25)

    assert excess_region_penalty(4, config) == 0.0
    assert excess_region_penalty(6, config) == 0.0
    assert excess_region_penalty(9, config) == 0.75
    with pytest.raises(ValueError, match="region_count"):
        excess_region_penalty(-1, config)


def test_variable_k_temperature_targets_representative_loss_acceptance() -> None:
    """Schedule calibration should match configured representative-loss probabilities."""
    representative_loss = 4.0
    schedule = demo_runner._schedule_from_losses(
        (2.0, representative_loss, 6.0),
        reference_score=10.0,
        initial_loss_acceptance=0.5,
        final_loss_acceptance=0.02,
        hold_fraction=0.05,
        polish_fraction=0.2,
    )

    assert exp(-representative_loss / schedule.initial_temperature) == pytest.approx(0.5)
    assert exp(-representative_loss / schedule.final_temperature) == pytest.approx(0.02)
    assert schedule.hold_fraction == 0.05
    assert schedule.polish_fraction == 0.2


def test_demo_runner_rejects_invalid_inputs_and_configuration() -> None:
    """Malformed grids, variances, and impossible K values should fail clearly."""
    with pytest.raises(ValueError, match="target_regions"):
        DemoSearchConfig(target_regions=1)
    with pytest.raises(ValueError, match="initial_regions"):
        VariableKSearchConfig(initial_regions=1, min_regions=2)
    with pytest.raises(ValueError, match="penalty_per_extra_region"):
        VariableKSearchConfig(penalty_per_extra_region=-0.1)
    with pytest.raises(ValueError, match="loss acceptance probabilities"):
        VariableKSearchConfig(initial_loss_acceptance=0.2, final_loss_acceptance=0.3)
    with pytest.raises(ValueError, match="hold_fraction and polish_fraction"):
        VariableKSearchConfig(hold_fraction=0.6, polish_fraction=0.5)
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


@pytest.mark.parametrize(
    "field_name",
    ["target_regions", "iterations", "pilot_proposals", "seed", "record_every"],
)
@pytest.mark.parametrize("invalid_value", [2.5, True])
def test_fixed_count_integer_configuration_fields_are_strict(
    field_name: str,
    invalid_value: object,
) -> None:
    """Fixed-count integer fields should reject fractions and booleans."""
    with pytest.raises(TypeError, match=field_name):
        DemoSearchConfig(**{field_name: invalid_value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name",
    [
        "initial_regions",
        "free_regions",
        "min_regions",
        "max_regions",
        "iterations",
        "pilot_proposals",
        "seed",
        "record_every",
    ],
)
@pytest.mark.parametrize("invalid_value", [2.5, True])
def test_variable_count_integer_configuration_fields_are_strict(
    field_name: str,
    invalid_value: object,
) -> None:
    """Variable-count integer fields should reject fractions and booleans."""
    with pytest.raises(TypeError, match=field_name):
        VariableKSearchConfig(**{field_name: invalid_value})  # type: ignore[arg-type]
