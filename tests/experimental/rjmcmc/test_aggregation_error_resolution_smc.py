"""Focused contracts for the coarse-to-fine Gamma--Beta SMC experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.aggregation_error import (
    FourCellAggregationOracle,
    TwoCellAggregationOracle,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
    PartitionMassState,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_resolution_smc import (
    GaussianGuideSpecification,
    ResolutionSchedule,
    ResolutionSMCCheckpoint,
    ResolutionSMCConfig,
    ResolutionTree,
    breadth_first_schedule,
    direct_iid_likelihood_average,
    draw_prior_allocation_paths,
    draw_scrambled_sobol_allocation_paths,
    parent_first_priority_schedule,
    run_resolution_smc,
)


def _two_cell(
    *,
    shapes: tuple[float, float] = (0.35, 4.0),
) -> tuple[ResolutionTree, ResolutionSchedule, GaussianGuideSpecification]:
    tree = ResolutionTree.from_nested_chart(
        [101, 205],
        shapes,
        (101, 205),
    )
    schedule = breadth_first_schedule(tree)
    guide = GaussianGuideSpecification.build(
        [0.44, 0.91, -0.08],
        [[1.8, 0.1], [-0.4, 1.2], [0.8, -0.3]],
        [0.25, 0.32, 0.28],
    )
    return tree, schedule, guide


def _four_cell(
    *,
    chart: str = "row",
) -> tuple[ResolutionTree, ResolutionSchedule, GaussianGuideSpecification]:
    nested = {
        "row": ((11, 17), (23, 31)),
        "column": ((11, 23), (17, 31)),
        "chain": (11, (17, (23, 31))),
    }[chart]
    tree = ResolutionTree.from_nested_chart(
        [11, 17, 23, 31],
        [0.15, 0.18, 0.20, 0.12],
        nested,
    )
    schedule = breadth_first_schedule(tree)
    guide = GaussianGuideSpecification.build(
        [1.62, 0.08, 0.13, 0.06],
        [
            [2.00, 0.00, 0.10, 0.00],
            [0.00, 1.70, 0.00, 0.10],
            [0.05, 0.00, 1.90, 0.00],
            [0.00, 0.10, 0.00, 2.10],
        ],
        [0.12, 0.14, 0.13, 0.15],
    )
    return tree, schedule, guide


def _result(
    tree: ResolutionTree,
    schedule: ResolutionSchedule,
    guide: GaussianGuideSpecification,
    config: ResolutionSMCConfig,
    *,
    paths: np.ndarray | None = None,
):
    result, _ = run_resolution_smc(
        tree,
        schedule,
        guide,
        root_total=1.0,
        config=config,
        source_identity="focused-test-source",
        allocation_paths=paths,
    )
    assert result is not None
    return result


def test_tree_metadata_and_schedules_are_deterministic_parent_first() -> None:
    """Compatible charts should be deterministic but independently identified."""
    row_tree, row_schedule, guide = _four_cell(chart="row")
    repeated_tree, repeated_schedule, _ = _four_cell(chart="row")
    column_tree, column_schedule, _ = _four_cell(chart="column")

    assert row_tree.identity == repeated_tree.identity
    assert row_schedule.identity == repeated_schedule.identity
    assert row_tree.identity != column_tree.identity
    assert row_schedule.identity != column_schedule.identity
    assert row_tree.is_compatible(column_tree)
    assert row_schedule.coordinate_node_ids == (0, 1, 4)
    assert column_schedule.coordinate_node_ids == (0, 1, 4)
    assert row_schedule.frontier_after(row_tree, 0) == (row_tree.root_id,)
    assert row_schedule.frontier_after(row_tree, 2) == row_tree.leaf_node_ids

    favorable = parent_first_priority_schedule(
        row_tree,
        guide,
        root_total=1.0,
        favorable=True,
    )
    unfavorable = parent_first_priority_schedule(
        row_tree,
        guide,
        root_total=1.0,
        favorable=False,
    )
    assert favorable.coordinate_node_ids[0] == row_tree.root_id
    assert unfavorable.coordinate_node_ids[0] == row_tree.root_id
    assert set(favorable.coordinate_node_ids) == set(row_tree.internal_node_ids)
    assert set(unfavorable.coordinate_node_ids) == set(row_tree.internal_node_ids)
    assert favorable.identity != unfavorable.identity

    with pytest.raises(ValueError, match="current frontier"):
        ResolutionSchedule.build(row_tree, [(1,), (0,), (4,)], name="not-parent-first")


def test_initial_and_terminal_moments_match_reusable_dirichlet_closure() -> None:
    """The root closure and terminal zero covariance should be exact."""
    tree, schedule, guide = _four_cell()
    config = ResolutionSMCConfig(32, seed=701, resampling_policy="never")
    _, initial = run_resolution_smc(
        tree,
        schedule,
        guide,
        root_total=1.0,
        config=config,
        source_identity="focused-test-source",
        stop_after_level=0,
    )
    aggregation = AdditiveDirichletAggregation(
        tree.cell_alphas,
        guide.design,
        guide.noise_sd,
        np.eye(guide.observation.size),
    )
    state = PartitionMassState(np.zeros(4, dtype=np.int64), [1.0])
    expected_mean = aggregation.conditional_observation_mean(state)
    expected_covariance = aggregation.observation_residual_covariance(state)

    np.testing.assert_allclose(
        initial.frontier.observation_mean,
        np.repeat(expected_mean[np.newaxis, :], config.particle_count, axis=0),
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        initial.frontier.unresolved_covariance,
        np.repeat(
            expected_covariance[np.newaxis, :, :],
            config.particle_count,
            axis=0,
        ),
        rtol=0.0,
        atol=2.0e-16,
    )

    result = _result(tree, schedule, guide, config)
    np.testing.assert_allclose(
        np.sum(result.terminal_leaf_masses, axis=1),
        np.ones(config.particle_count),
        rtol=0.0,
        atol=2.0 * np.spacing(1.0),
    )
    assert result.diagnostics[-1].max_terminal_unresolved_covariance == 0.0
    assert all(diagnostic.max_mass_conservation_error <= np.spacing(1.0) for diagnostic in result.diagnostics)


@pytest.mark.parametrize(
    ("shapes", "fraction_order"),
    [
        ((45.0, 55.0), 48),
        ((0.35, 4.0), 96),
    ],
)
def test_two_cell_direct_iid_agrees_with_gauss_jacobi_oracle(
    shapes: tuple[float, float],
    fraction_order: int,
) -> None:
    """Large deterministic IID screens should cover the exact tiny oracle."""
    tree, schedule, guide = _two_cell(shapes=shapes)
    count = 65_536
    config = ResolutionSMCConfig(count, seed=8191, resampling_policy="never")
    direct = direct_iid_likelihood_average(
        tree,
        schedule,
        guide,
        root_total=1.0,
        config=config,
    )
    oracle = TwoCellAggregationOracle(
        gamma_shape=sum(shapes),
        gamma_rate=sum(shapes),
        beta_first_shape=shapes[0],
        beta_second_shape=shapes[1],
        fraction_order=fraction_order,
    )
    exact = oracle.coarse_conditional_likelihood(
        1.0,
        guide.observation,
        guide.design,
        guide.noise_sd,
    )
    samples = np.exp(direct.terminal_log_likelihoods)
    standard_error = float(np.std(samples, ddof=1) / math.sqrt(count))
    assert abs(direct.likelihood - exact) <= 5.0 * standard_error


def test_path_matched_direct_iid_and_no_resampling_smc_are_bitwise_identical() -> None:
    """The no-resampling normalizer is the same complete-path arithmetic mean."""
    tree, schedule, guide = _four_cell()
    config = ResolutionSMCConfig(1_024, seed=12345, resampling_policy="never")
    paths = draw_prior_allocation_paths(
        tree,
        schedule,
        particle_count=config.particle_count,
        seed=991,
    )
    direct = direct_iid_likelihood_average(
        tree,
        schedule,
        guide,
        root_total=1.0,
        config=config,
        allocation_paths=paths,
    )
    smc = _result(tree, schedule, guide, config, paths=paths)

    assert smc.log_likelihood == direct.log_likelihood
    assert smc.likelihood == direct.likelihood
    assert smc.allocation_paths_sha256 == direct.allocation_paths_sha256
    np.testing.assert_array_equal(
        smc.terminal_log_likelihoods,
        direct.terminal_log_likelihoods,
    )
    np.testing.assert_array_equal(smc.terminal_leaf_masses, direct.leaf_masses)
    assert smc.no_resampling_accumulator_error == pytest.approx(0.0, abs=2.0e-14)


def test_scrambled_sobol_paths_replay_nest_and_match_the_tiny_oracle() -> None:
    """The wider-R1 RQMC baseline should be replayable and correctly transformed."""
    tree, schedule, guide = _two_cell(shapes=(0.12, 0.18))
    small = draw_scrambled_sobol_allocation_paths(
        tree,
        schedule,
        particle_count=256,
        seed=1701,
    )
    replay = draw_scrambled_sobol_allocation_paths(
        tree,
        schedule,
        particle_count=256,
        seed=1701,
    )
    large = draw_scrambled_sobol_allocation_paths(
        tree,
        schedule,
        particle_count=4_096,
        seed=1701,
    )
    np.testing.assert_array_equal(small, replay)
    np.testing.assert_array_equal(small, large[: small.shape[0]])
    assert np.all((small > 0.0) & (small < 1.0))

    config = ResolutionSMCConfig(
        large.shape[0],
        seed=1701,
        resampling_policy="never",
    )
    direct = direct_iid_likelihood_average(
        tree,
        schedule,
        guide,
        root_total=1.0,
        config=config,
        allocation_paths=large,
    )
    oracle = TwoCellAggregationOracle(
        gamma_shape=0.30,
        gamma_rate=0.30,
        beta_first_shape=0.12,
        beta_second_shape=0.18,
        fraction_order=128,
    )
    exact = oracle.coarse_conditional_likelihood(
        1.0,
        guide.observation,
        guide.design,
        guide.noise_sd,
    )
    assert direct.likelihood == pytest.approx(exact, rel=2.0e-3)

    with pytest.raises(ValueError, match="power of two"):
        draw_scrambled_sobol_allocation_paths(
            tree,
            schedule,
            particle_count=100,
            seed=1701,
        )


def test_terminal_likelihood_agrees_with_four_cell_quadrature_oracle() -> None:
    """The four-cell continuous target should cover an independently charted oracle."""
    tree, schedule, guide = _four_cell(chart="row")
    config = ResolutionSMCConfig(131_072, seed=20260730, resampling_policy="never")
    direct = direct_iid_likelihood_average(
        tree,
        schedule,
        guide,
        root_total=1.0,
        config=config,
    )
    oracle = FourCellAggregationOracle(
        tree.cell_alphas,
        gamma_rate=float(np.sum(tree.cell_alphas)),
        fraction_order=64,
        chunk_size=8_192,
    )
    row = math.exp(
        oracle.conditional_log_likelihood(
            1.0,
            guide.observation,
            guide.design,
            guide.noise_sd,
            tiling="root",
            root_chart="row-first",
        )
    )
    column = math.exp(
        oracle.conditional_log_likelihood(
            1.0,
            guide.observation,
            guide.design,
            guide.noise_sd,
            tiling="root",
            root_chart="column-first",
        )
    )
    samples = np.exp(direct.terminal_log_likelihoods)
    standard_error = float(np.std(samples, ddof=1) / math.sqrt(config.particle_count))
    assert abs(row - column) <= max(1.0e-12, 0.25 * standard_error)
    assert abs(direct.likelihood - 0.5 * (row + column)) <= 5.0 * standard_error


def test_child_swap_is_equivariant_with_complemented_allocation_path() -> None:
    """Swapping children and Beta coordinates should preserve scientific masses."""
    tree, schedule, guide = _two_cell()
    swapped = ResolutionTree.from_nested_chart(
        tree.cell_ids,
        tree.cell_alphas,
        (205, 101),
    )
    swapped_schedule = breadth_first_schedule(swapped)
    config = ResolutionSMCConfig(4_096, seed=83, resampling_policy="never")
    paths = draw_prior_allocation_paths(
        tree,
        schedule,
        particle_count=config.particle_count,
        seed=109,
    )
    original = _result(tree, schedule, guide, config, paths=paths)
    permuted = _result(
        swapped,
        swapped_schedule,
        guide,
        config,
        paths=1.0 - paths,
    )

    np.testing.assert_allclose(
        permuted.terminal_leaf_masses,
        original.terminal_leaf_masses,
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        permuted.terminal_log_likelihoods,
        original.terminal_log_likelihoods,
        rtol=0.0,
        atol=3.0e-14,
    )
    assert permuted.likelihood == pytest.approx(original.likelihood, rel=2.0e-15)


def test_bootstrap_multinomial_resampling_and_deterministic_replay() -> None:
    """Independent RNG streams should replay weights, ancestry, and normalizer."""
    tree, schedule, guide = _four_cell()
    config = ResolutionSMCConfig(
        256,
        seed=731,
        resampling_policy="always",
        ess_fraction=0.5,
    )
    first = _result(tree, schedule, guide, config)
    second = _result(tree, schedule, guide, config)

    assert first.scientific_fingerprint == second.scientific_fingerprint
    assert first.log_likelihood == second.log_likelihood
    np.testing.assert_array_equal(first.ancestry, second.ancestry)
    np.testing.assert_array_equal(first.normalized_log_weights, second.normalized_log_weights)
    assert first.diagnostics[0].resampled
    assert not first.diagnostics[-1].resampled
    assert first.diagnostics[0].unique_ancestor_count <= config.particle_count


def test_piecewise_beta_guide_is_corrected_continuous_and_unbiased() -> None:
    """The bounded R2 proposal must retain the exact continuous terminal target."""
    tree, schedule, guide = _two_cell(shapes=(0.12, 0.18))
    estimates = []
    for seed in range(32):
        config = ResolutionSMCConfig(
            128,
            seed=30_000 + seed,
            resampling_policy="never",
            proposal_kind="piecewise_beta_guide",
            proposal_bin_count=16,
            proposal_audit_order=32,
        )
        result = _result(tree, schedule, guide, config)
        estimates.append(result.likelihood)
        diagnostic = result.diagnostics[0]
        assert diagnostic.proposal_kind == "piecewise_beta_guide"
        assert diagnostic.proposal_bin_count == 16
        assert diagnostic.proposal_guide_evaluation_count == 128 * (16 + 32)
        assert math.isfinite(diagnostic.proposal_normalizer_relative_error_max)
        assert diagnostic.proposal_normalizer_relative_error_max >= 0.0
        assert result.no_resampling_accumulator_error is None

    oracle = TwoCellAggregationOracle(
        gamma_shape=0.30,
        gamma_rate=0.30,
        beta_first_shape=0.12,
        beta_second_shape=0.18,
        fraction_order=128,
    ).coarse_conditional_likelihood(
        1.0,
        guide.observation,
        guide.design,
        guide.noise_sd,
    )
    values = np.asarray(estimates)
    standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size))
    assert abs(float(np.mean(values)) - oracle) <= 5.0 * standard_error


def test_guided_proposal_checkpoint_replays_and_requires_single_split_levels(
    tmp_path: Path,
) -> None:
    """Guided proposal state and RNG use should replay from an intermediate level."""
    tree, breadth, guide = _four_cell()
    schedule = parent_first_priority_schedule(
        tree,
        guide,
        root_total=1.0,
        favorable=True,
    )
    config = ResolutionSMCConfig(
        96,
        seed=8128,
        resampling_policy="always",
        proposal_kind="piecewise_beta_guide",
        proposal_bin_count=8,
        proposal_audit_order=16,
    )
    uninterrupted = _result(tree, schedule, guide, config)
    repeated = _result(tree, schedule, guide, config)
    assert repeated.scientific_fingerprint == uninterrupted.scientific_fingerprint

    _, partial = run_resolution_smc(
        tree,
        schedule,
        guide,
        root_total=1.0,
        config=config,
        source_identity="focused-test-source",
        stop_after_level=1,
    )
    path = tmp_path / "guided-boundary.npz"
    partial.save(path)
    loaded = ResolutionSMCCheckpoint.load(
        path,
        tree=tree,
        schedule=schedule,
        guide=guide,
        config=config,
        allocation_paths_sha256=None,
        source_identity="focused-test-source",
    )
    resumed, _ = run_resolution_smc(
        tree,
        schedule,
        guide,
        root_total=1.0,
        config=config,
        source_identity="focused-test-source",
        checkpoint=loaded,
    )
    assert resumed is not None
    assert resumed.scientific_fingerprint == uninterrupted.scientific_fingerprint

    with pytest.raises(ValueError, match="one eligible split"):
        _result(tree, breadth, guide, config)


def test_checkpoint_restart_replays_from_every_resolution_boundary(tmp_path: Path) -> None:
    """Every saved boundary should reproduce the uninterrupted scientific result."""
    tree, schedule, guide = _four_cell()
    config = ResolutionSMCConfig(384, seed=1877, resampling_policy="always")
    uninterrupted = _result(tree, schedule, guide, config)

    for boundary in range(len(schedule.batches) + 1):
        partial_result, partial = run_resolution_smc(
            tree,
            schedule,
            guide,
            root_total=1.0,
            config=config,
            source_identity="focused-test-source",
            stop_after_level=boundary,
        )
        assert (partial_result is not None) == (boundary == len(schedule.batches))
        path = tmp_path / f"boundary-{boundary}.npz"
        partial.save(path)
        loaded = ResolutionSMCCheckpoint.load(
            path,
            tree=tree,
            schedule=schedule,
            guide=guide,
            config=config,
            allocation_paths_sha256=None,
            source_identity="focused-test-source",
        )
        resumed, _ = run_resolution_smc(
            tree,
            schedule,
            guide,
            root_total=1.0,
            config=config,
            source_identity="focused-test-source",
            checkpoint=loaded,
        )
        assert resumed is not None
        assert resumed.scientific_fingerprint == uninterrupted.scientific_fingerprint
        assert resumed.log_likelihood == uninterrupted.log_likelihood
        np.testing.assert_array_equal(resumed.ancestry, uninterrupted.ancestry)
        np.testing.assert_array_equal(
            resumed.terminal_leaf_masses,
            uninterrupted.terminal_leaf_masses,
        )


def test_checkpoint_load_fails_closed_for_provenance_and_particle_tampering(
    tmp_path: Path,
) -> None:
    """Tree, schedule, input, seed, path, source, and particle corruption must fail."""
    tree, schedule, guide = _four_cell()
    config = ResolutionSMCConfig(64, seed=4099, resampling_policy="always")
    _, checkpoint = run_resolution_smc(
        tree,
        schedule,
        guide,
        root_total=1.0,
        config=config,
        source_identity="focused-test-source",
        stop_after_level=1,
    )
    path = tmp_path / "checkpoint.npz"
    checkpoint.save(path)

    column_tree, column_schedule, _ = _four_cell(chart="column")
    priority = parent_first_priority_schedule(
        tree,
        guide,
        root_total=1.0,
        favorable=True,
    )
    changed_guide = GaussianGuideSpecification.build(
        guide.observation + np.array([0.0, 0.0, 0.0, 1.0e-6]),
        guide.design,
        guide.noise_sd,
    )
    changed_seed = ResolutionSMCConfig(64, seed=4100, resampling_policy="always")
    changed_particles = ResolutionSMCConfig(65, seed=4099, resampling_policy="always")

    bad_calls: list[dict[str, Any]] = [
        dict(tree=column_tree, schedule=column_schedule, guide=guide, config=config),
        dict(tree=tree, schedule=priority, guide=guide, config=config),
        dict(tree=tree, schedule=schedule, guide=changed_guide, config=config),
        dict(tree=tree, schedule=schedule, guide=guide, config=changed_seed),
        dict(tree=tree, schedule=schedule, guide=guide, config=changed_particles),
    ]
    for arguments in bad_calls:
        with pytest.raises(ValueError, match="identity|configuration"):
            ResolutionSMCCheckpoint.load(
                path,
                **arguments,
                allocation_paths_sha256=None,
                source_identity="focused-test-source",
            )
    with pytest.raises(ValueError, match="allocation-path identity"):
        ResolutionSMCCheckpoint.load(
            path,
            tree=tree,
            schedule=schedule,
            guide=guide,
            config=config,
            allocation_paths_sha256="0" * 64,
            source_identity="focused-test-source",
        )
    with pytest.raises(ValueError, match="source identity"):
        ResolutionSMCCheckpoint.load(
            path,
            tree=tree,
            schedule=schedule,
            guide=guide,
            config=config,
            allocation_paths_sha256=None,
            source_identity="different-source",
        )

    with np.load(path, allow_pickle=False) as stored:
        payload = {name: np.array(stored[name], copy=True) for name in stored.files}
    metadata = json.loads(str(payload["metadata_json"].item()))
    metadata["completed_levels"] = 2
    tampered_metadata = np.asarray(json.dumps(metadata))
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(
        tampered,
        metadata_json=tampered_metadata,
        node_masses=payload["node_masses"],
        observation_mean=payload["observation_mean"],
        unresolved_covariance=payload["unresolved_covariance"],
        normalized_log_weights=payload["normalized_log_weights"],
        current_guide_log_likelihoods=payload["current_guide_log_likelihoods"],
        lineage=payload["lineage"],
        ancestry=payload["ancestry"],
    )
    with pytest.raises(ValueError, match="content digest"):
        ResolutionSMCCheckpoint.load(
            tampered,
            tree=tree,
            schedule=schedule,
            guide=guide,
            config=config,
            allocation_paths_sha256=None,
            source_identity="focused-test-source",
        )


def test_compatible_tree_charts_agree_in_expectation() -> None:
    """Row- and column-first charts should cover one common oracle."""
    row_tree, row_schedule, guide = _four_cell(chart="row")
    column_tree, column_schedule, _ = _four_cell(chart="column")
    count = 65_536
    config = ResolutionSMCConfig(count, seed=65_537, resampling_policy="never")
    row = direct_iid_likelihood_average(
        row_tree,
        row_schedule,
        guide,
        root_total=1.0,
        config=config,
    )
    column = direct_iid_likelihood_average(
        column_tree,
        column_schedule,
        guide,
        root_total=1.0,
        config=config,
    )
    row_samples = np.exp(row.terminal_log_likelihoods)
    column_samples = np.exp(column.terminal_log_likelihoods)
    pooled_standard_error = math.sqrt(
        float(np.var(row_samples, ddof=1) + np.var(column_samples, ddof=1)) / count
    )
    assert abs(row.likelihood - column.likelihood) <= 5.0 * pooled_standard_error


def test_invalid_target_inputs_and_nonfinite_arithmetic_fail_closed() -> None:
    """Malformed target or configuration inputs should never be repaired silently."""
    tree, schedule, guide = _two_cell()
    with pytest.raises(ValueError, match="strictly positive"):
        run_resolution_smc(
            tree,
            schedule,
            guide,
            root_total=0.0,
            config=ResolutionSMCConfig(8, seed=1, resampling_policy="never"),
            source_identity="focused-test-source",
        )
    with pytest.raises(ValueError, match="positive"):
        GaussianGuideSpecification.build(
            guide.observation,
            guide.design,
            [0.25, 0.0, 0.28],
        )
    with pytest.raises(ValueError, match="inside"):
        _result(
            tree,
            schedule,
            guide,
            ResolutionSMCConfig(2, seed=1, resampling_policy="never"),
            paths=np.array([[0.0], [0.5]]),
        )
