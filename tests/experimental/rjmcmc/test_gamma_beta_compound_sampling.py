"""Tests for the full-posterior Gamma--Beta compound reference sampler."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.core import FixedDesignBlock
from openghg_inversions.experimental.rjmcmc.dyadic_tree import (
    CanonicalDyadicTree,
    DyadicFrontier,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_compound_sampling import (
    GammaBetaCompoundConfig,
    GammaBetaCompoundSamplingResult,
    GammaBetaCompoundTrace,
    continue_gamma_beta_compound,
    sample_gamma_beta_compound,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_proposals import (
    propose_fixed_coefficient,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_tree import (
    GammaBetaTreePrior,
    GammaBetaTreeProblem,
    GammaBetaTreeState,
    TreePartitionPrior,
    build_gamma_beta_tree_state,
)
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings
from openghg_inversions.experimental.rjmcmc.sampling import PCG64State


def _problem(
    *,
    fixed_count: int = 0,
    likelihood_power: float = 0.0,
    partition_prior: TreePartitionPrior | None = None,
) -> GammaBetaTreeProblem:
    """Build a two-cell problem with an optional zero-design fixed block."""
    tree = partition_prior.tree if partition_prior is not None else CanonicalDyadicTree.from_shape((1, 2))
    n_cells = len(tree.leaf_ids)
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        np.ones(len(tree.leaf_ids)),
        concentration=2.0,
        root_mean=1.0,
        root_variance=0.25,
    )
    fixed_block = (
        None
        if fixed_count == 0
        else FixedDesignBlock(
            design=np.zeros((n_cells, fixed_count)),
            coefficient_prior_mean=np.ones(fixed_count),
            coefficient_prior_sd=np.full(fixed_count, 0.5),
        )
    )
    return GammaBetaTreeProblem(
        observations=np.zeros(n_cells),
        observation_sd=np.ones(n_cells),
        sensitivity=np.eye(n_cells),
        prior=prior,
        partition_prior=(TreePartitionPrior.uniform_k(tree) if partition_prior is None else partition_prior),
        likelihood_power=likelihood_power,
        fixed_block=fixed_block,
    )


def _root_state(problem: GammaBetaTreeProblem) -> GammaBetaTreeState:
    """Build the root frontier with unit continuous coordinates."""
    return build_gamma_beta_tree_state(
        problem,
        frontier=DyadicFrontier.root(problem.tree),
        root_total=1.0,
        active_fractions=[],
        fixed_coefficients=np.ones(problem.n_fixed_coefficients),
    )


def _assert_states_equal(
    actual: GammaBetaTreeState,
    expected: GammaBetaTreeState,
) -> None:
    """Assert exact equality of all state coordinates and cached targets."""
    assert actual.problem is expected.problem
    assert actual.frontier == expected.frontier
    assert actual.root_total == expected.root_total
    for name in (
        "active_fractions",
        "active_node_masses",
        "fixed_coefficients",
        "dynamic_prediction",
        "fixed_prediction",
        "prediction",
        "residual",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in (
        "log_gaussian_likelihood",
        "log_likelihood",
        "log_root_prior",
        "log_fraction_prior",
        "log_partition_prior",
        "log_fixed_coefficient_prior",
        "log_target",
    ):
        assert getattr(actual, name) == getattr(expected, name)


def _assert_traces_equal(
    actual: GammaBetaCompoundTrace,
    expected: GammaBetaCompoundTrace,
) -> None:
    """Assert exact equality of retained and every-attempt trace values."""
    assert actual.frontiers == expected.frontiers
    assert len(actual.split_fractions) == len(expected.split_fractions)
    for left, right in zip(actual.split_fractions, expected.split_fractions, strict=True):
        np.testing.assert_array_equal(left, right)
    for name in (
        "root_total",
        "fixed_coefficients",
        "k",
        "log_gaussian_likelihood",
        "log_likelihood",
        "log_root_prior",
        "log_fraction_prior",
        "log_partition_prior",
        "log_fixed_coefficient_prior",
        "log_target",
        "state_transition",
        "global_transition",
        "slot",
        "move",
        "valid",
        "accepted",
        "node_id",
        "coefficient_id",
        "k_before",
        "k_after",
        "log_acceptance_ratio",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def _concatenate_results(
    first: GammaBetaCompoundSamplingResult,
    second: GammaBetaCompoundSamplingResult,
) -> GammaBetaCompoundTrace:
    """Join two adjacent segment traces for a full-run comparison."""
    vector_names = (
        "root_total",
        "k",
        "log_gaussian_likelihood",
        "log_likelihood",
        "log_root_prior",
        "log_fraction_prior",
        "log_partition_prior",
        "log_fixed_coefficient_prior",
        "log_target",
        "state_transition",
        "global_transition",
        "slot",
        "move",
        "valid",
        "accepted",
        "node_id",
        "coefficient_id",
        "k_before",
        "k_after",
        "log_acceptance_ratio",
    )
    values = {
        name: np.concatenate((getattr(first.trace, name), getattr(second.trace, name)))
        for name in vector_names
    }
    return GammaBetaCompoundTrace(
        frontiers=first.trace.frontiers + second.trace.frontiers,
        split_fractions=first.trace.split_fractions + second.trace.split_fractions,
        fixed_coefficients=np.concatenate(
            (first.trace.fixed_coefficients, second.trace.fixed_coefficients),
            axis=0,
        ),
        **values,
    )


def test_default_cycle_schedules_every_kernel_and_fixed_position() -> None:
    """Six outer coefficients give the opportunity-matched 14-slot cycle."""
    tree = CanonicalDyadicTree.from_shape((1, 2))
    root_only = TreePartitionPrior.uniform_k(tree, maximum_k=1)
    problem = _problem(fixed_count=6, partition_prior=root_only)
    result = sample_gamma_beta_compound(
        problem,
        _root_state(problem),
        GammaBetaCompoundConfig(iterations=14, seed=91),
    )

    assert result.trace.slot.tolist() == [
        "structural",
        "structural",
        "root",
        "fraction",
        "fraction",
        "fraction",
        "fraction",
        "fraction",
        "fixed",
        "fixed",
        "fixed",
        "fixed",
        "fixed",
        "fixed",
    ]
    assert result.trace.move[2:].tolist() == [
        "root_refresh",
        "fraction_refresh",
        "fraction_refresh",
        "fraction_refresh",
        "fraction_refresh",
        "fraction_refresh",
        "fixed_coefficient",
        "fixed_coefficient",
        "fixed_coefficient",
        "fixed_coefficient",
        "fixed_coefficient",
        "fixed_coefficient",
    ]
    assert result.trace.valid[:2].tolist() == [False, False]
    assert result.trace.valid[3:8].tolist() == [False] * 5
    invalid = np.r_[0:2, 3:8]
    assert result.trace.accepted[invalid].tolist() == [False] * 7
    assert result.trace.node_id[invalid].tolist() == [-1] * 7
    assert result.trace.log_acceptance_ratio[invalid].tolist() == [-np.inf] * 7
    assert result.trace.k_before.tolist() == [1] * 14
    assert result.trace.k_after.tolist() == [1] * 14
    assert result.trace.coefficient_id.tolist()[-6:] == list(range(6))
    assert result.trace.global_transition.tolist() == list(range(1, 15))
    assert any(result.trace.accepted[-6:])
    assert not np.array_equal(
        result.trace.fixed_coefficients[0],
        result.trace.fixed_coefficients[-1],
    )
    assert result.checkpoint.state is result.final_state
    assert result.checkpoint.schedule_phase == 0


def test_seeded_compound_sampling_replays_every_draw_exactly() -> None:
    """The same seed and inputs replay retained states and all diagnostics."""
    problem = _problem(fixed_count=2)
    initial = _root_state(problem)
    config = GammaBetaCompoundConfig(
        iterations=73,
        seed=20260723,
        fixed_coefficient_proposal_sd=(0.3, 0.6),
    )
    retention = RetentionSettings(warmup_transitions=4, thin=5)

    first = sample_gamma_beta_compound(problem, initial, config, retention=retention)
    replay = sample_gamma_beta_compound(problem, initial, config, retention=retention)

    _assert_traces_equal(first.trace, replay.trace)
    _assert_states_equal(first.final_state, replay.final_state)
    assert first.checkpoint.rng_state == replay.checkpoint.rng_state


def test_invalid_structural_slots_still_consume_acceptance_uniforms() -> None:
    """Two root-only boundary attempts advance direction and acceptance draws."""
    tree = CanonicalDyadicTree.from_shape((1, 2))
    problem = _problem(partition_prior=TreePartitionPrior.uniform_k(tree, maximum_k=1))
    result = sample_gamma_beta_compound(
        problem,
        _root_state(problem),
        GammaBetaCompoundConfig(iterations=2, seed=17),
    )
    expected_rng = np.random.Generator(np.random.PCG64(17))
    expected_rng.random(4)

    assert result.trace.valid.tolist() == [False, False]
    assert result.trace.accepted.tolist() == [False, False]
    assert result.checkpoint.rng_state == PCG64State.from_generator(expected_rng)


def test_continuation_is_exact_from_an_awkward_mid_cycle_phase() -> None:
    """A split at a non-retained mid-cycle phase preserves RNG and schedule."""
    problem = _problem(fixed_count=2)
    initial = _root_state(problem)
    retention = RetentionSettings(warmup_transitions=7, thin=6)
    full = sample_gamma_beta_compound(
        problem,
        initial,
        GammaBetaCompoundConfig(iterations=47, seed=812, fixed_coefficient_proposal_sd=0.5),
        retention=retention,
    )
    first = sample_gamma_beta_compound(
        problem,
        initial,
        GammaBetaCompoundConfig(iterations=13, seed=812, fixed_coefficient_proposal_sd=0.5),
        retention=retention,
    )
    second = continue_gamma_beta_compound(
        problem,
        first.checkpoint,
        iterations=34,
    )

    _assert_traces_equal(_concatenate_results(first, second), full.trace)
    _assert_states_equal(second.final_state, full.final_state)
    assert second.checkpoint.rng_state == full.checkpoint.rng_state
    assert first.checkpoint.schedule_phase == 3
    assert second.checkpoint.schedule_phase == 7


def test_no_fixed_block_omits_fixed_slots_and_retains_zero_width_matrix() -> None:
    """A problem without outer coefficients has the eight-slot default cycle."""
    problem = _problem()
    result = sample_gamma_beta_compound(
        problem,
        _root_state(problem),
        GammaBetaCompoundConfig(iterations=16, seed=18),
    )

    assert result.checkpoint.kernel_settings.cycle_length == 8
    assert "fixed" not in result.trace.slot
    assert result.trace.fixed_coefficients.shape == (17, 0)
    assert np.all(result.trace.log_fixed_coefficient_prior == 0.0)


def test_checkpoint_rejects_inconsistent_schedule_phase() -> None:
    """Checkpoint phase must equal the completed-transition cycle remainder."""
    problem = _problem(fixed_count=1)
    result = sample_gamma_beta_compound(
        problem,
        _root_state(problem),
        GammaBetaCompoundConfig(iterations=4, seed=3),
    )

    with pytest.raises(ValueError, match="schedule_phase"):
        replace(result.checkpoint, schedule_phase=0)


def test_sampler_rejects_noncontiguous_positive_k_support() -> None:
    """Compound split/merge scheduling requires a support interval without gaps."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    partition_prior = TreePartitionPrior.from_marginal_probabilities(
        tree,
        [0.0, 0.5, 0.0, 0.5],
    )
    problem = _problem(partition_prior=partition_prior)

    with pytest.raises(ValueError, match="contiguous"):
        sample_gamma_beta_compound(
            problem,
            _root_state(problem),
            GammaBetaCompoundConfig(iterations=1, seed=0),
        )


def test_sampler_rejects_fixed_k_with_multiple_unreachable_frontiers() -> None:
    """A fixed K with multiple frontiers needs a topology-preserving kernel."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    partition_prior = TreePartitionPrior.uniform_k(
        tree,
        minimum_k=3,
        maximum_k=3,
    )
    problem = _problem(partition_prior=partition_prior)
    frontier = DyadicFrontier.root(tree).split(tree, tree.root_id)
    frontier = frontier.split(tree, frontier.node_ids[0])
    state = build_gamma_beta_tree_state(
        problem,
        frontier=frontier,
        root_total=1.0,
        active_fractions=[0.5, 0.5],
    )

    with pytest.raises(ValueError, match="no fixed-K topology move"):
        sample_gamma_beta_compound(
            problem,
            state,
            GammaBetaCompoundConfig(iterations=1, seed=0),
        )


def test_sampler_rejects_zero_fraction_slots_when_splits_are_supported() -> None:
    """A full-posterior schedule cannot freeze persistent active fractions."""
    problem = _problem()

    with pytest.raises(ValueError, match="fraction_refresh_slots must be positive"):
        sample_gamma_beta_compound(
            problem,
            _root_state(problem),
            GammaBetaCompoundConfig(
                iterations=1,
                seed=0,
                fraction_refresh_slots=0,
            ),
        )


def test_fixed_coefficient_proposal_has_symmetric_gaussian_accounting() -> None:
    """A deterministic fixed slot updates its design column and MH components."""
    base = _problem()
    problem = GammaBetaTreeProblem(
        observations=np.array([0.5, 2.0]),
        observation_sd=np.full(2, 0.1),
        sensitivity=base.sensitivity,
        prior=base.prior,
        partition_prior=base.partition_prior,
        fixed_block=FixedDesignBlock(
            design=np.array([[0.0], [1.0]]),
            coefficient_prior_mean=np.ones(1),
            coefficient_prior_sd=np.full(1, 0.5),
        ),
    )
    state = _root_state(problem)
    transition = propose_fixed_coefficient(
        problem,
        state,
        coefficient_position=0,
        proposed_coefficient=1.25,
        proposal_stdev=0.4,
    )

    assert transition.valid
    assert transition.move == "fixed_coefficient"
    assert transition.node_id is None
    assert transition.coefficient_id == 0
    assert transition.log_q_forward == pytest.approx(transition.log_q_reverse)
    assert transition.delta_log_likelihood > 5.0
    assert transition.delta_log_root_prior == 0.0
    assert transition.delta_log_fraction_prior == 0.0
    assert transition.delta_log_partition_prior == 0.0
    assert transition.log_acceptance_ratio == pytest.approx(
        transition.delta_log_likelihood + transition.delta_log_fixed_coefficient_prior
    )
    assert transition.candidate.fixed_coefficients.tolist() == [1.25]
    np.testing.assert_allclose(state.prediction, [0.5, 1.5])
    np.testing.assert_allclose(transition.candidate.prediction, [0.5, 1.75])

    sampled = sample_gamma_beta_compound(
        problem,
        state,
        GammaBetaCompoundConfig(
            iterations=90,
            seed=91,
            fixed_coefficient_proposal_sd=0.4,
        ),
    )
    fixed_attempts = sampled.trace.move == "fixed_coefficient"
    assert np.count_nonzero(fixed_attempts) == 10
    assert np.any(sampled.trace.accepted[fixed_attempts])
    assert np.any(sampled.trace.fixed_coefficients[:, 0] != 1.0)


def test_prior_only_chain_recovers_joint_topology_root_and_fraction_moments() -> None:
    """The compound schedule samples declared tiny-tree joint prior moments."""
    problem = _problem()
    result = sample_gamma_beta_compound(
        problem,
        _root_state(problem),
        GammaBetaCompoundConfig(iterations=40_000, seed=493),
        retention=RetentionSettings(warmup_transitions=4_000, thin=2),
    )
    split_fractions = np.array(
        [
            fraction[0]
            for frontier, fraction in zip(
                result.trace.frontiers,
                result.trace.split_fractions,
                strict=True,
            )
            if len(frontier) == 2
        ]
    )

    assert np.mean(result.trace.k == 1) == pytest.approx(0.5, abs=0.035)
    assert np.mean(result.trace.root_total) == pytest.approx(1.0, abs=0.035)
    assert np.var(result.trace.root_total) == pytest.approx(0.25, abs=0.035)
    assert split_fractions.size > 5_000
    assert np.mean(split_fractions) == pytest.approx(0.5, abs=0.025)
    assert np.var(split_fractions) == pytest.approx(1.0 / 12.0, abs=0.02)
