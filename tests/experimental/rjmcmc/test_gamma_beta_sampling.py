"""Tests for seeded structural-only Gamma--Beta tree sampling."""

from __future__ import annotations

from dataclasses import replace
from math import exp

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.dyadic_tree import (
    CanonicalDyadicTree,
    DyadicFrontier,
    enumerate_frontiers,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_proposals import (
    propose_merge,
    propose_split,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_sampling import (
    GammaBetaSamplerConfig,
    GammaBetaSamplingResult,
    GammaBetaTrace,
    continue_gamma_beta_tree,
    sample_gamma_beta_tree,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_tree import (
    GammaBetaTreePrior,
    GammaBetaTreeProblem,
    GammaBetaTreeState,
    TreePartitionPrior,
    build_gamma_beta_tree_state,
)
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings


def _problem(
    *,
    shape: tuple[int, int] = (2, 2),
    likelihood_power: float = 0.0,
) -> GammaBetaTreeProblem:
    """Return a compact problem with a uniform-K structural prior."""
    tree = CanonicalDyadicTree.from_shape(shape)
    cell_count = int(np.prod(shape))
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        np.arange(1.0, cell_count + 1.0),
        concentration=2.0,
        root_mean=1.0,
        root_variance=0.25,
    )
    return GammaBetaTreeProblem(
        observations=np.zeros(cell_count),
        observation_sd=np.ones(cell_count),
        sensitivity=np.eye(cell_count),
        prior=prior,
        partition_prior=TreePartitionPrior.uniform_k(tree),
        likelihood_power=likelihood_power,
    )


def _root_state(problem: GammaBetaTreeProblem, *, root_total: float = 1.0) -> GammaBetaTreeState:
    """Build the unresolved root state."""
    return build_gamma_beta_tree_state(
        problem,
        frontier=DyadicFrontier.root(problem.tree),
        root_total=root_total,
        active_fractions=np.empty(0),
    )


def _full_state(problem: GammaBetaTreeProblem, *, root_total: float = 1.0) -> GammaBetaTreeState:
    """Build the fully resolved state at prior-mean fractions."""
    frontier = DyadicFrontier.root(problem.tree)
    fractions: list[float] = []
    while splittable := problem.tree.splittable_nodes(frontier):
        node_id = splittable[0]
        alpha, beta = problem.prior.beta_parameters(node_id)
        frontier = frontier.split(problem.tree, node_id)
        split_nodes = frontier.active_split_nodes(problem.tree)
        fractions_by_node = dict(zip(split_nodes[:-1], fractions, strict=True))
        fractions_by_node[node_id] = alpha / (alpha + beta)
        fractions = [fractions_by_node[active_id] for active_id in split_nodes]
    return build_gamma_beta_tree_state(
        problem,
        frontier=frontier,
        root_total=root_total,
        active_fractions=np.asarray(fractions),
    )


def _assert_states_equal(actual: GammaBetaTreeState, expected: GammaBetaTreeState) -> None:
    """Compare exact scientific coordinates and deterministic target caches."""
    assert actual.problem is expected.problem
    assert actual.frontier == expected.frontier
    assert actual.root_total == expected.root_total
    for name in (
        "active_fractions",
        "active_node_masses",
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
        "log_target",
    ):
        assert getattr(actual, name) == getattr(expected, name)


def _assert_traces_equal(actual: GammaBetaTrace, expected: GammaBetaTrace) -> None:
    """Compare fixed and variable-dimensional trace fields exactly."""
    assert actual.frontiers == expected.frontiers
    assert len(actual.split_fractions) == len(expected.split_fractions)
    for actual_fractions, expected_fractions in zip(
        actual.split_fractions,
        expected.split_fractions,
        strict=True,
    ):
        np.testing.assert_array_equal(actual_fractions, expected_fractions)
    for name in (
        "root_total",
        "k",
        "log_target",
        "state_transition",
        "moves",
        "valid",
        "accepted",
        "node_id",
        "log_acceptance_ratio",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def _concatenate_segment_traces(
    first: GammaBetaSamplingResult,
    second: GammaBetaSamplingResult,
) -> GammaBetaTrace:
    """Join two adjacent segment traces for an exact full-run comparison."""
    return GammaBetaTrace(
        frontiers=first.trace.frontiers + second.trace.frontiers,
        split_fractions=first.trace.split_fractions + second.trace.split_fractions,
        root_total=np.concatenate((first.trace.root_total, second.trace.root_total)),
        k=np.concatenate((first.trace.k, second.trace.k)),
        log_target=np.concatenate((first.trace.log_target, second.trace.log_target)),
        state_transition=np.concatenate((first.trace.state_transition, second.trace.state_transition)),
        moves=np.concatenate((first.trace.moves, second.trace.moves)),
        valid=np.concatenate((first.trace.valid, second.trace.valid)),
        accepted=np.concatenate((first.trace.accepted, second.trace.accepted)),
        node_id=np.concatenate((first.trace.node_id, second.trace.node_id)),
        log_acceptance_ratio=np.concatenate(
            (
                first.trace.log_acceptance_ratio,
                second.trace.log_acceptance_ratio,
            )
        ),
    )


def test_seeded_structural_sampling_replays_exactly() -> None:
    """The same seed and inputs should replay every state and diagnostic."""
    problem = _problem()
    initial_state = _root_state(problem)
    config = GammaBetaSamplerConfig(iterations=89, seed=20260723)
    retention = RetentionSettings(warmup_transitions=3, thin=7)

    first = sample_gamma_beta_tree(problem, initial_state, config, retention=retention)
    replay = sample_gamma_beta_tree(problem, initial_state, config, retention=retention)

    _assert_traces_equal(first.trace, replay.trace)
    _assert_states_equal(first.final_state, replay.final_state)
    assert first.checkpoint.rng_state == replay.checkpoint.rng_state


def test_sampler_config_rejects_invalid_seed_support() -> None:
    """PCG64 seeds are normalized to non-negative non-Boolean integers."""
    assert GammaBetaSamplerConfig(iterations=1, seed=3).seed == 3
    with pytest.raises(TypeError, match="seed"):
        GammaBetaSamplerConfig(iterations=1, seed=True)
    with pytest.raises(ValueError, match="non-negative"):
        GammaBetaSamplerConfig(iterations=1, seed=-1)


def test_continuation_is_exact_across_an_awkward_retention_boundary() -> None:
    """Segmenting between retained coordinates must preserve RNG and thinning."""
    problem = _problem()
    initial_state = _root_state(problem)
    retention = RetentionSettings(warmup_transitions=5, thin=6)
    full = sample_gamma_beta_tree(
        problem,
        initial_state,
        GammaBetaSamplerConfig(iterations=43, seed=731),
        retention=retention,
    )
    first = sample_gamma_beta_tree(
        problem,
        initial_state,
        GammaBetaSamplerConfig(iterations=18, seed=731),
        retention=retention,
    )
    second = continue_gamma_beta_tree(
        problem,
        first.checkpoint,
        iterations=25,
    )

    _assert_traces_equal(_concatenate_segment_traces(first, second), full.trace)
    _assert_states_equal(second.final_state, full.final_state)
    assert second.checkpoint.rng_state == full.checkpoint.rng_state
    assert second.checkpoint.transitions_completed == 43


def test_checkpoint_rejects_malformed_continuation_coordinates() -> None:
    """Checkpoint construction validates global phase and state identity."""
    problem = _problem()
    result = sample_gamma_beta_tree(
        problem,
        _root_state(problem),
        GammaBetaSamplerConfig(iterations=2, seed=4),
    )
    other_problem = _problem()

    with pytest.raises(ValueError, match="non-negative"):
        replace(result.checkpoint, transitions_completed=-1)
    with pytest.raises(ValueError, match="belong"):
        replace(result.checkpoint, problem=other_problem)


def test_unavailable_directions_are_explicit_boundary_self_transitions() -> None:
    """Selected merge-at-root and split-at-full attempts remain recorded stays."""
    problem = _problem()
    cases = (
        (_root_state(problem), 0, "merge"),
        (_full_state(problem), 2, "split"),
    )

    for initial_state, seed, expected_move in cases:
        result = sample_gamma_beta_tree(
            problem,
            initial_state,
            GammaBetaSamplerConfig(iterations=1, seed=seed),
        )

        assert result.trace.moves.tolist() == [expected_move]
        assert result.trace.valid.tolist() == [False]
        assert result.trace.accepted.tolist() == [False]
        assert result.trace.node_id.tolist() == [-1]
        assert result.trace.log_acceptance_ratio.tolist() == [-np.inf]
        assert result.trace.state_transition.tolist() == [0, 1]
        assert result.trace.frontiers == (
            initial_state.frontier,
            initial_state.frontier,
        )
        _assert_states_equal(result.final_state, initial_state)


def test_sampler_rejects_an_initial_frontier_outside_prior_support() -> None:
    """The sampler fails clearly before iterating from a zero-mass K."""
    base = _problem(likelihood_power=0.0)
    problem = GammaBetaTreeProblem(
        observations=base.observations,
        observation_sd=base.observation_sd,
        sensitivity=base.sensitivity,
        prior=base.prior,
        partition_prior=TreePartitionPrior.uniform_k(
            base.tree,
            minimum_k=2,
            maximum_k=3,
        ),
        likelihood_power=0.0,
    )
    initial_state = _root_state(problem)

    assert initial_state.log_partition_prior == -np.inf
    with pytest.raises(ValueError, match="finite target support"):
        sample_gamma_beta_tree(
            problem,
            initial_state,
            GammaBetaSamplerConfig(iterations=1, seed=1),
        )


def test_two_by_two_structural_kernel_preserves_known_frontier_prior() -> None:
    """The five-state kernel should satisfy exact detailed balance and stationarity."""
    problem = _problem()
    frontiers = enumerate_frontiers(problem.tree)
    index_by_frontier = {frontier: index for index, frontier in enumerate(frontiers)}
    target = np.array([exp(problem.partition_prior.log_probability(frontier)) for frontier in frontiers])
    transition_matrix = np.zeros((len(frontiers), len(frontiers)))

    for source_index, frontier in enumerate(frontiers):
        state = build_gamma_beta_tree_state(
            problem,
            frontier=frontier,
            root_total=1.0,
            active_fractions=np.full(len(frontier) - 1, 0.5),
        )
        splittable = problem.tree.splittable_nodes(frontier)
        mergeable = problem.tree.mergeable_parents(frontier)
        if not splittable:
            transition_matrix[source_index, source_index] += 0.5
        for node_id in splittable:
            proposal = propose_split(
                problem,
                state,
                leaf_node_id=node_id,
                new_fraction=0.5,
            )
            probability = 0.5 / len(splittable)
            acceptance = min(1.0, exp(proposal.log_acceptance_ratio))
            candidate_index = index_by_frontier[proposal.candidate.frontier]
            transition_matrix[source_index, candidate_index] += probability * acceptance
            transition_matrix[source_index, source_index] += probability * (1.0 - acceptance)
        if not mergeable:
            transition_matrix[source_index, source_index] += 0.5
        for node_id in mergeable:
            proposal = propose_merge(
                problem,
                state,
                parent_node_id=node_id,
            )
            probability = 0.5 / len(mergeable)
            acceptance = min(1.0, exp(proposal.log_acceptance_ratio))
            candidate_index = index_by_frontier[proposal.candidate.frontier]
            transition_matrix[source_index, candidate_index] += probability * acceptance
            transition_matrix[source_index, source_index] += probability * (1.0 - acceptance)

    np.testing.assert_allclose(transition_matrix.sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(
        target[:, np.newaxis] * transition_matrix,
        target[np.newaxis, :] * transition_matrix.T,
        atol=1e-15,
    )
    np.testing.assert_allclose(target @ transition_matrix, target, atol=1e-15)


def test_prior_only_sampler_visits_all_edges_and_matches_frontier_mass() -> None:
    """A seeded chain traverses the tiny graph and recovers its exact prior."""
    problem = _problem()
    exact_frontiers = enumerate_frontiers(problem.tree)
    target = {
        frontier: exp(problem.partition_prior.log_probability(frontier)) for frontier in exact_frontiers
    }
    result = sample_gamma_beta_tree(
        problem,
        _root_state(problem),
        GammaBetaSamplerConfig(iterations=30_000, seed=7641),
        retention=RetentionSettings(warmup_transitions=2_000, thin=1),
    )
    counts = {frontier: 0 for frontier in exact_frontiers}
    for frontier in result.trace.frontiers:
        counts[frontier] += 1
    empirical = {frontier: count / len(result.trace.frontiers) for frontier, count in counts.items()}
    steps = np.diff(result.trace.k)
    immediate_reversals = np.sum((steps[:-1] != 0) & (steps[1:] == -steps[:-1]))

    assert all(count > 0 for count in counts.values())
    assert np.count_nonzero(result.trace.accepted) > 5_000
    assert immediate_reversals > 500
    for frontier in exact_frontiers:
        assert empirical[frontier] == pytest.approx(target[frontier], abs=0.025)


def test_two_cell_likelihood_prefers_the_resolving_split() -> None:
    """A split matching heterogeneous data should beat the unresolved average."""
    tree = CanonicalDyadicTree.from_shape((1, 2))
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        np.ones(2),
        concentration=2.0,
        root_mean=4.0,
        root_variance=1.0,
    )
    problem = GammaBetaTreeProblem(
        observations=np.array([1.0, 3.0]),
        observation_sd=np.array([0.2, 0.2]),
        sensitivity=np.eye(2),
        prior=prior,
        partition_prior=TreePartitionPrior.uniform_k(tree),
    )
    unresolved = _root_state(problem, root_total=4.0)
    split = propose_split(
        problem,
        unresolved,
        leaf_node_id=tree.root_id,
        new_fraction=0.25,
    )
    reverse = propose_merge(
        problem,
        split.candidate,
        parent_node_id=tree.root_id,
    )

    np.testing.assert_allclose(unresolved.prediction, [2.0, 2.0])
    np.testing.assert_allclose(split.candidate.prediction, [1.0, 3.0])
    assert split.delta_log_likelihood > 20.0
    assert split.log_acceptance_ratio > 20.0
    assert reverse.log_acceptance_ratio < -20.0
