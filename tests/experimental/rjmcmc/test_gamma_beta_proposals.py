"""Tests for deterministic active-only Gamma--Beta proposal accounting."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import log

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.dyadic_tree import (
    CanonicalDyadicTree,
    DyadicFrontier,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_proposals import (
    GammaBetaTransitionTerms,
    accept_or_reject,
    propose_fraction_refresh,
    propose_merge,
    propose_root_refresh,
    propose_split,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_tree import (
    GammaBetaTreePrior,
    GammaBetaTreeProblem,
    GammaBetaTreeState,
    TreePartitionPrior,
    build_gamma_beta_tree_state,
)


def _problem(*, likelihood_power: float = 1.0) -> GammaBetaTreeProblem:
    """Return a heterogeneous two-by-two fixed-tree problem."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        np.array([1.0, 2.0, 3.0, 4.0]),
        concentration=2.0,
        root_mean=1.0,
        root_variance=0.25,
    )
    return GammaBetaTreeProblem(
        observations=np.array([1.2, -0.4, 0.7]),
        observation_sd=np.array([0.5, 1.1, 0.8]),
        sensitivity=np.array(
            [
                [1.0, 0.5, -0.2, 0.3],
                [0.1, -0.4, 0.9, 0.2],
                [-0.3, 0.8, 0.4, 1.1],
            ]
        ),
        prior=prior,
        partition_prior=TreePartitionPrior.uniform_k(tree),
        likelihood_power=likelihood_power,
    )


def _state(
    problem: GammaBetaTreeProblem,
    frontier: DyadicFrontier,
    fractions: list[float],
) -> GammaBetaTreeState:
    """Build a source with a fixed root total and canonical fractions."""
    return build_gamma_beta_tree_state(
        problem,
        frontier=frontier,
        root_total=1.3,
        active_fractions=np.asarray(fractions),
    )


def _assert_same_state(
    actual: GammaBetaTreeState,
    expected: GammaBetaTreeState,
) -> None:
    """Compare the scientific coordinates and all target components."""
    assert actual.problem is expected.problem
    assert actual.frontier == expected.frontier
    assert actual.root_total == expected.root_total
    np.testing.assert_array_equal(actual.active_fractions, expected.active_fractions)
    for name in (
        "log_likelihood",
        "log_root_prior",
        "log_fraction_prior",
        "log_partition_prior",
        "log_target",
    ):
        assert getattr(actual, name) == pytest.approx(
            getattr(expected, name),
            rel=0.0,
            abs=1e-13,
        )


def test_split_merge_round_trip_has_exact_pointwise_reverse_terms() -> None:
    """A selected split and cherry merge should be reciprocal at one fraction."""
    problem = _problem()
    two_regions = DyadicFrontier.root(problem.tree).split(problem.tree, 0)
    source = _state(problem, two_regions, [0.4])
    split_probability = 0.3
    fraction = 0.35

    forward = propose_split(
        problem,
        source,
        leaf_node_id=1,
        new_fraction=fraction,
        split_direction_probability=split_probability,
    )
    reverse = propose_merge(
        problem,
        forward.candidate,
        parent_node_id=1,
        split_direction_probability=split_probability,
    )
    beta_logpdf = problem.prior.log_fraction_density(1, fraction)

    assert forward.valid and reverse.valid
    assert forward.node_id == reverse.node_id == 1
    assert forward.log_q_forward_direction == pytest.approx(log(split_probability))
    assert forward.log_q_forward_selection == pytest.approx(-log(2.0))
    assert forward.log_q_forward_auxiliary == pytest.approx(beta_logpdf)
    assert forward.log_q_reverse_direction == pytest.approx(log(1.0 - split_probability))
    assert forward.log_q_reverse_selection == 0.0
    assert forward.log_q_reverse_auxiliary == 0.0
    assert reverse.log_q_forward == pytest.approx(forward.log_q_reverse)
    assert reverse.log_q_reverse == pytest.approx(forward.log_q_forward)
    assert reverse.delta_log_likelihood == pytest.approx(-forward.delta_log_likelihood)
    assert reverse.delta_log_root_prior == pytest.approx(-forward.delta_log_root_prior)
    assert reverse.delta_log_fraction_prior == pytest.approx(-forward.delta_log_fraction_prior)
    assert reverse.delta_log_partition_prior == pytest.approx(-forward.delta_log_partition_prior)
    assert reverse.log_q_forward_direction == pytest.approx(forward.log_q_reverse_direction)
    assert reverse.log_q_forward_selection == pytest.approx(forward.log_q_reverse_selection)
    assert reverse.log_q_forward_auxiliary == pytest.approx(forward.log_q_reverse_auxiliary)
    assert reverse.log_q_reverse_direction == pytest.approx(forward.log_q_forward_direction)
    assert reverse.log_q_reverse_selection == pytest.approx(forward.log_q_forward_selection)
    assert reverse.log_q_reverse_auxiliary == pytest.approx(forward.log_q_forward_auxiliary)
    assert reverse.log_acceptance_ratio == pytest.approx(
        -forward.log_acceptance_ratio,
        rel=0.0,
        abs=1e-13,
    )
    _assert_same_state(reverse.candidate, source)


def test_split_records_complete_target_and_proposal_decomposition() -> None:
    """Every target component and normalized proposal factor remains explicit."""
    problem = _problem()
    source = _state(
        problem,
        DyadicFrontier.root(problem.tree).split(problem.tree, 0),
        [0.45],
    )
    transition = propose_split(
        problem,
        source,
        leaf_node_id=4,
        new_fraction=0.65,
    )
    candidate = transition.candidate
    beta_logpdf = problem.prior.log_fraction_density(4, 0.65)
    expected_partition_delta = problem.partition_prior.log_probability(
        candidate.frontier
    ) - problem.partition_prior.log_probability(source.frontier)

    assert transition.delta_log_likelihood == pytest.approx(candidate.log_likelihood - source.log_likelihood)
    assert transition.delta_log_root_prior == 0.0
    assert transition.delta_log_fraction_prior == pytest.approx(beta_logpdf)
    assert transition.delta_log_partition_prior == pytest.approx(expected_partition_delta)
    assert transition.log_q_forward_direction == pytest.approx(log(0.5))
    assert transition.log_q_forward_selection == pytest.approx(-log(2.0))
    assert transition.log_q_forward_auxiliary == pytest.approx(beta_logpdf)
    assert transition.log_q_reverse_direction == pytest.approx(log(0.5))
    assert transition.log_q_reverse_selection == 0.0
    assert transition.log_q_reverse_auxiliary == 0.0
    assert transition.log_target_delta == pytest.approx(
        transition.delta_log_likelihood
        + transition.delta_log_root_prior
        + transition.delta_log_fraction_prior
        + transition.delta_log_partition_prior
    )
    assert transition.log_q_forward == pytest.approx(
        transition.log_q_forward_direction
        + transition.log_q_forward_selection
        + transition.log_q_forward_auxiliary
    )
    assert transition.log_q_reverse == pytest.approx(
        transition.log_q_reverse_direction
        + transition.log_q_reverse_selection
        + transition.log_q_reverse_auxiliary
    )
    assert transition.log_acceptance_ratio == pytest.approx(
        transition.log_target_delta
        + transition.log_q_reverse
        - transition.log_q_forward
        + transition.log_jacobian
    )


def test_unequal_eligible_counts_use_source_and_candidate_frontiers() -> None:
    """A final 2x2 split should use one source split and two candidate cherries."""
    problem = _problem()
    three_regions = DyadicFrontier.root(problem.tree).split(problem.tree, 0).split(problem.tree, 1)
    source = _state(problem, three_regions, [0.4, 0.3])
    forward = propose_split(
        problem,
        source,
        leaf_node_id=4,
        new_fraction=0.6,
    )
    reverse = propose_merge(problem, forward.candidate, parent_node_id=4)

    assert problem.tree.splittable_nodes(source.frontier) == (4,)
    assert problem.tree.mergeable_parents(forward.candidate.frontier) == (1, 4)
    assert forward.log_q_forward_selection == 0.0
    assert forward.log_q_reverse_selection == pytest.approx(-log(2.0))
    assert reverse.log_q_forward_selection == pytest.approx(-log(2.0))
    assert reverse.log_q_reverse_selection == 0.0
    assert reverse.log_acceptance_ratio == pytest.approx(
        -forward.log_acceptance_ratio,
        rel=0.0,
        abs=1e-13,
    )


def test_structural_boundaries_and_ineligible_nodes_are_invalid_self_transitions() -> None:
    """Unavailable directions, terminal leaves, and non-cherries retain source."""
    problem = _problem()
    root = _state(problem, DyadicFrontier.root(problem.tree), [])
    three_frontier = root.frontier.split(problem.tree, 0).split(problem.tree, 1)
    three = _state(problem, three_frontier, [0.4, 0.3])
    full = _state(
        problem,
        three_frontier.split(problem.tree, 4),
        [0.4, 0.3, 0.6],
    )
    transitions = (
        propose_merge(problem, root, parent_node_id=-1),
        propose_split(problem, full, leaf_node_id=-1, new_fraction=0.5),
        propose_split(problem, three, leaf_node_id=2, new_fraction=0.5),
        propose_merge(problem, three, parent_node_id=0),
        propose_split(problem, root, leaf_node_id=0, new_fraction=0.0),
        propose_fraction_refresh(
            problem,
            root,
            split_node_id=-1,
            new_fraction=0.5,
        ),
    )

    for transition, source in zip(
        transitions,
        (root, full, three, three, root, root),
        strict=True,
    ):
        assert not transition.valid
        assert transition.reason
        assert transition.candidate is source
        assert transition.log_acceptance_ratio == -np.inf
        assert transition.log_target_delta == 0.0
        assert transition.log_q_forward == 0.0
        assert transition.log_q_reverse == 0.0
        assert transition.log_jacobian == 0.0
        assert accept_or_reject(source, transition, log_uniform=-np.inf) is source


def test_proposals_reject_sources_outside_partition_prior_support() -> None:
    """A zero-mass source is invalid even when its frontier is geometrically valid."""
    base = _problem(likelihood_power=0.0)
    probabilities = np.array([0.0, 0.5, 0.0, 0.5, 0.0])
    problem = GammaBetaTreeProblem(
        observations=base.observations,
        observation_sd=base.observation_sd,
        sensitivity=base.sensitivity,
        prior=base.prior,
        partition_prior=TreePartitionPrior.from_marginal_probabilities(
            base.tree,
            probabilities,
        ),
        likelihood_power=0.0,
    )
    frontier = DyadicFrontier.root(problem.tree).split(problem.tree, problem.tree.root_id)
    source = _state(problem, frontier, [0.4])

    assert source.log_partition_prior == -np.inf
    with pytest.raises(ValueError, match="finite target support"):
        propose_split(
            problem,
            source,
            leaf_node_id=frontier.node_ids[0],
            new_fraction=0.5,
        )


def test_proposal_into_excluded_k_is_a_valid_certain_rejection() -> None:
    """A supported source may propose a geometric state with zero prior mass."""
    base = _problem(likelihood_power=0.0)
    problem = GammaBetaTreeProblem(
        observations=base.observations,
        observation_sd=base.observation_sd,
        sensitivity=base.sensitivity,
        prior=base.prior,
        partition_prior=TreePartitionPrior.uniform_k(
            base.tree,
            minimum_k=2,
            maximum_k=2,
        ),
        likelihood_power=0.0,
    )
    frontier = DyadicFrontier.root(problem.tree).split(problem.tree, problem.tree.root_id)
    source = _state(problem, frontier, [0.4])
    transition = propose_split(
        problem,
        source,
        leaf_node_id=frontier.node_ids[0],
        new_fraction=0.5,
    )

    assert transition.valid
    assert transition.candidate.log_partition_prior == -np.inf
    assert transition.log_acceptance_ratio == -np.inf
    assert accept_or_reject(source, transition, log_uniform=-np.inf) is source


def test_structural_coordinate_insertion_has_explicit_unit_jacobian() -> None:
    """Split and merge terms must not include a physical leaf-mass Jacobian."""
    problem = _problem()
    source = _state(problem, DyadicFrontier.root(problem.tree), [])
    split = propose_split(problem, source, leaf_node_id=0, new_fraction=0.25)
    merge = propose_merge(problem, split.candidate, parent_node_id=0)

    assert split.log_jacobian == 0.0
    assert merge.log_jacobian == 0.0
    assert split.log_q_forward_auxiliary == pytest.approx(problem.prior.log_fraction_density(0, 0.25))
    assert split.delta_log_fraction_prior == pytest.approx(problem.prior.log_fraction_density(0, 0.25))


def test_prior_only_fraction_and_root_refreshes_cancel_exactly() -> None:
    """Independent prior proposals leave only the zero powered-likelihood change."""
    problem = _problem(likelihood_power=0.0)
    source = _state(
        problem,
        DyadicFrontier.root(problem.tree).split(problem.tree, 0).split(problem.tree, 1),
        [0.4, 0.3],
    )
    fraction = propose_fraction_refresh(
        problem,
        source,
        split_node_id=1,
        new_fraction=0.7,
    )
    root = propose_root_refresh(problem, source, new_root_total=0.8)

    assert fraction.delta_log_likelihood == 0.0
    assert fraction.delta_log_root_prior == 0.0
    assert fraction.delta_log_partition_prior == 0.0
    assert fraction.delta_log_fraction_prior == pytest.approx(
        fraction.log_q_forward_auxiliary - fraction.log_q_reverse_auxiliary
    )
    assert fraction.log_q_forward_direction == fraction.log_q_reverse_direction == 0.0
    assert fraction.log_q_forward_selection == pytest.approx(-log(2.0))
    assert fraction.log_q_reverse_selection == pytest.approx(-log(2.0))
    assert fraction.log_jacobian == 0.0
    assert fraction.log_acceptance_ratio == pytest.approx(0.0, abs=1e-13)
    assert root.delta_log_likelihood == 0.0
    assert root.delta_log_fraction_prior == 0.0
    assert root.delta_log_partition_prior == 0.0
    assert root.delta_log_root_prior == pytest.approx(
        root.log_q_forward_auxiliary - root.log_q_reverse_auxiliary
    )
    assert root.log_q_forward_direction == root.log_q_reverse_direction == 0.0
    assert root.log_q_forward_selection == root.log_q_reverse_selection == 0.0
    assert root.log_jacobian == 0.0
    assert root.log_acceptance_ratio == pytest.approx(0.0, abs=1e-13)
    assert accept_or_reject(source, fraction, log_uniform=-1.0) is fraction.candidate
    assert accept_or_reject(source, fraction, log_uniform=0.0) is source
    assert accept_or_reject(source, root, log_uniform=-1.0) is root.candidate


def test_transitions_are_immutable_and_reject_wrong_problem_states() -> None:
    """Frozen accounting cannot be altered and state/problem identity is enforced."""
    problem = _problem()
    source = _state(problem, DyadicFrontier.root(problem.tree), [])
    transition = propose_split(problem, source, leaf_node_id=0, new_fraction=0.4)

    assert isinstance(transition, GammaBetaTransitionTerms)
    with pytest.raises(FrozenInstanceError):
        transition.valid = False  # type: ignore[misc]

    other_problem = _problem()
    with pytest.raises(ValueError, match="built for problem"):
        propose_split(
            other_problem,
            source,
            leaf_node_id=0,
            new_fraction=0.4,
        )
