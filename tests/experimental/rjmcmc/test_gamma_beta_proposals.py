"""Tests for deterministic active-only Gamma--Beta proposal accounting."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import exp, log

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.dyadic_tree import (
    CanonicalDyadicTree,
    DyadicFrontier,
    SubtreePartitionIndex,
    enumerate_frontiers,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_proposals import (
    GammaBetaTransitionTerms,
    accept_or_reject,
    eligible_subtree_retile_blocks,
    propose_fraction_refresh,
    propose_merge,
    propose_relocate,
    propose_root_refresh,
    propose_split,
    propose_subtree_retile,
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


def _grid_problem(
    shape: tuple[int, int],
    *,
    likelihood_power: float = 0.0,
) -> GammaBetaTreeProblem:
    """Return a square-design problem on an arbitrary small dyadic grid."""
    tree = CanonicalDyadicTree.from_shape(shape)
    n_cells = shape[0] * shape[1]
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        np.arange(1.0, n_cells + 1.0),
        concentration=2.0,
        root_mean=1.0,
        root_variance=0.25,
    )
    return GammaBetaTreeProblem(
        observations=np.zeros(n_cells),
        observation_sd=np.ones(n_cells),
        sensitivity=np.eye(n_cells),
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


def _prior_only_state(
    problem: GammaBetaTreeProblem,
    frontier: DyadicFrontier,
) -> GammaBetaTreeState:
    """Build a deterministic prior-only state for one topology oracle."""
    fractions = [
        problem.prior.beta_parameters(node_id)[0] / sum(problem.prior.beta_parameters(node_id))
        for node_id in frontier.active_split_nodes(problem.tree)
    ]
    return _state(problem, frontier, fractions)


def _subtree_frontier_for_test(
    tree: CanonicalDyadicTree,
    frontier: DyadicFrontier,
    block_node_id: int,
) -> DyadicFrontier:
    """Extract active leaves geometrically contained in one subtree block."""
    block = tree.node(block_node_id)
    return DyadicFrontier(
        tuple(
            node_id
            for node_id in frontier.node_ids
            if (
                tree.node(node_id).row_start >= block.row_start
                and tree.node(node_id).row_stop <= block.row_stop
                and tree.node(node_id).col_start >= block.col_start
                and tree.node(node_id).col_stop <= block.col_stop
            )
        )
    )


def _finish_transition_matrix(matrix: np.ndarray) -> np.ndarray:
    """Put rejected proposal probability on the diagonal and validate mass."""
    off_diagonal_mass = matrix.sum(axis=1)
    assert np.all(off_diagonal_mass <= 1.0 + 1e-14)
    matrix[np.diag_indices_from(matrix)] += 1.0 - off_diagonal_mass
    return matrix


def _relocation_transition_matrix(
    problem: GammaBetaTreeProblem,
    frontiers: tuple[DyadicFrontier, ...],
) -> np.ndarray:
    """Integrate Beta auxiliaries out of the prior-only relocation kernel."""
    topology_index = {frontier: position for position, frontier in enumerate(frontiers)}
    matrix = np.zeros((len(frontiers), len(frontiers)))
    for source_position, frontier in enumerate(frontiers):
        state = _prior_only_state(problem, frontier)
        for merge_node_id in problem.tree.mergeable_parents(frontier):
            intermediate = frontier.merge(problem.tree, merge_node_id)
            destinations = tuple(
                node_id for node_id in problem.tree.splittable_nodes(intermediate) if node_id != merge_node_id
            )
            for destination_node_id in destinations:
                transition = propose_relocate(
                    problem,
                    state,
                    merge_parent_node_id=merge_node_id,
                    split_leaf_node_id=destination_node_id,
                    new_fraction=0.5,
                )
                assert transition.valid
                proposal_probability = exp(transition.log_q_forward_selection)
                acceptance_probability = min(1.0, exp(transition.log_acceptance_ratio))
                candidate_position = topology_index[transition.candidate.frontier]
                matrix[source_position, candidate_position] += proposal_probability * acceptance_probability
    return _finish_transition_matrix(matrix)


def _subtree_retile_transition_matrix(
    problem: GammaBetaTreeProblem,
    frontiers: tuple[DyadicFrontier, ...],
    index: SubtreePartitionIndex,
) -> np.ndarray:
    """Integrate Beta auxiliaries out of the prior-only retile kernel."""
    topology_index = {frontier: position for position, frontier in enumerate(frontiers)}
    matrix = np.zeros((len(frontiers), len(frontiers)))
    for source_position, frontier in enumerate(frontiers):
        state = _prior_only_state(problem, frontier)
        source_split_nodes = frozenset(frontier.active_split_nodes(problem.tree))
        for block_node_id, block_k in eligible_subtree_retile_blocks(problem, state, index):
            source_subtree = _subtree_frontier_for_test(
                problem.tree,
                frontier,
                block_node_id,
            )
            source_rank = index.rank(block_node_id, block_k, source_subtree)
            source_subtree_nodes = frozenset(source_subtree.node_ids)
            for replacement_rank in range(index.count(block_node_id, block_k)):
                if replacement_rank == source_rank:
                    continue
                replacement = index.unrank(
                    block_node_id,
                    block_k,
                    replacement_rank,
                )
                candidate_frontier = DyadicFrontier(
                    tuple(node_id for node_id in frontier.node_ids if node_id not in source_subtree_nodes)
                    + replacement.node_ids
                )
                candidate_split_nodes = frozenset(candidate_frontier.active_split_nodes(problem.tree))
                new_fractions = {node_id: 0.5 for node_id in candidate_split_nodes - source_split_nodes}
                transition = propose_subtree_retile(
                    problem,
                    state,
                    index,
                    block_node_id=block_node_id,
                    replacement_frontier=replacement,
                    new_fractions_by_node=new_fractions,
                )
                assert transition.valid
                proposal_probability = exp(transition.log_q_forward_selection)
                acceptance_probability = min(1.0, exp(transition.log_acceptance_ratio))
                candidate_position = topology_index[transition.candidate.frontier]
                matrix[source_position, candidate_position] += proposal_probability * acceptance_probability
    return _finish_transition_matrix(matrix)


def _reachable_positions(matrix: np.ndarray, start: int = 0) -> set[int]:
    """Return topology positions reachable through positive off-diagonal mass."""
    reached = {start}
    pending = [start]
    while pending:
        source = pending.pop()
        neighbours = {
            int(position) for position in np.flatnonzero(matrix[source] > 0.0) if position != source
        }
        unseen = neighbours - reached
        reached.update(unseen)
        pending.extend(unseen)
    return reached


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


def test_relocate_has_exact_pointwise_reverse_terms_at_fixed_k() -> None:
    """Merge-then-split relocation is reciprocal with both Beta densities."""
    problem = _problem(likelihood_power=0.0)
    source_frontier = DyadicFrontier.root(problem.tree).split(problem.tree, 0).split(problem.tree, 1)
    source = _state(problem, source_frontier, [0.4, 0.3])

    forward = propose_relocate(
        problem,
        source,
        merge_parent_node_id=1,
        split_leaf_node_id=4,
        new_fraction=0.6,
    )
    reverse = propose_relocate(
        problem,
        forward.candidate,
        merge_parent_node_id=4,
        split_leaf_node_id=1,
        new_fraction=0.3,
    )

    assert forward.valid and reverse.valid
    assert forward.move == "relocate"
    assert forward.node_id == 1
    assert forward.secondary_node_id == 4
    assert forward.block_leaf_count is None
    assert forward.candidate.k == source.k
    assert forward.log_q_forward_selection == 0.0
    assert forward.log_q_reverse_selection == 0.0
    assert forward.log_q_forward_auxiliary == pytest.approx(problem.prior.log_fraction_density(4, 0.6))
    assert forward.log_q_reverse_auxiliary == pytest.approx(problem.prior.log_fraction_density(1, 0.3))
    assert forward.log_q_forward_direction == forward.log_q_reverse_direction == 0.0
    assert forward.log_jacobian == 0.0
    assert forward.delta_log_partition_prior == 0.0
    assert forward.delta_log_fraction_prior == pytest.approx(
        forward.log_q_forward_auxiliary - forward.log_q_reverse_auxiliary
    )
    assert forward.log_acceptance_ratio == pytest.approx(0.0, abs=1e-13)
    assert reverse.log_q_forward == pytest.approx(forward.log_q_reverse)
    assert reverse.log_q_reverse == pytest.approx(forward.log_q_forward)
    assert reverse.log_acceptance_ratio == pytest.approx(
        -forward.log_acceptance_ratio,
        abs=1e-13,
    )
    _assert_same_state(reverse.candidate, source)


def test_relocate_uses_sequential_normalized_selection_counts() -> None:
    """Relocation accounts for both source cherries and intermediate leaves."""
    problem = _grid_problem((2, 4))
    root = problem.tree.root_id
    first, second = problem.tree.children(root)
    source_frontier = (
        DyadicFrontier.root(problem.tree)
        .split(problem.tree, root)
        .split(problem.tree, first)
        .split(problem.tree, second)
    )
    source = _state(problem, source_frontier, [0.4, 0.3, 0.6])
    intermediate = source.frontier.merge(problem.tree, first)
    destinations = tuple(
        node_id for node_id in problem.tree.splittable_nodes(intermediate) if node_id != first
    )
    destination = destinations[0]
    forward = propose_relocate(
        problem,
        source,
        merge_parent_node_id=first,
        split_leaf_node_id=destination,
        new_fraction=0.55,
    )
    reverse_intermediate = forward.candidate.frontier.merge(problem.tree, destination)
    reverse_destinations = tuple(
        node_id for node_id in problem.tree.splittable_nodes(reverse_intermediate) if node_id != destination
    )

    assert len(problem.tree.mergeable_parents(source.frontier)) == 2
    assert len(destinations) == 2
    assert forward.log_q_forward_selection == pytest.approx(-log(2.0) - log(2.0))
    assert forward.log_q_reverse_selection == pytest.approx(
        -log(len(problem.tree.mergeable_parents(forward.candidate.frontier))) - log(len(reverse_destinations))
    )
    reverse = propose_relocate(
        problem,
        forward.candidate,
        merge_parent_node_id=destination,
        split_leaf_node_id=first,
        new_fraction=0.3,
    )
    assert reverse.log_q_forward == pytest.approx(forward.log_q_reverse)
    assert reverse.log_q_reverse == pytest.approx(forward.log_q_forward)
    _assert_same_state(reverse.candidate, source)


def test_relocate_invalid_choices_are_explicit_self_transitions() -> None:
    """Relocation rejects missing cherries, same nodes, and invalid fractions."""
    problem = _problem()
    root = _state(problem, DyadicFrontier.root(problem.tree), [])
    source_frontier = root.frontier.split(problem.tree, 0).split(problem.tree, 1)
    source = _state(problem, source_frontier, [0.4, 0.3])
    full = _state(problem, source_frontier.split(problem.tree, 4), [0.4, 0.3, 0.6])
    transitions = (
        propose_relocate(
            problem,
            root,
            merge_parent_node_id=0,
            split_leaf_node_id=1,
            new_fraction=0.5,
        ),
        propose_relocate(
            problem,
            source,
            merge_parent_node_id=1,
            split_leaf_node_id=1,
            new_fraction=0.5,
        ),
        propose_relocate(
            problem,
            source,
            merge_parent_node_id=1,
            split_leaf_node_id=4,
            new_fraction=1.0,
        ),
        propose_relocate(
            problem,
            full,
            merge_parent_node_id=1,
            split_leaf_node_id=4,
            new_fraction=0.5,
        ),
    )

    for transition, expected_source in zip(
        transitions,
        (root, source, source, full),
        strict=True,
    ):
        assert not transition.valid
        assert transition.candidate is expected_source
        assert transition.reason
        assert transition.log_acceptance_ratio == -np.inf


def test_subtree_retile_has_exact_reverse_terms_and_common_fractions() -> None:
    """A same-size subtree tiling replacement is pointwise reversible."""
    problem = _problem(likelihood_power=0.0)
    root = problem.tree.root_id
    source_frontier = DyadicFrontier.root(problem.tree).split(problem.tree, root).split(problem.tree, 1)
    source = _state(problem, source_frontier, [0.4, 0.3])
    index = SubtreePartitionIndex(problem.tree, max_k=4)
    source_rank = index.rank(root, 3, source.frontier)
    replacement = index.unrank(root, 3, 1 - source_rank)

    assert eligible_subtree_retile_blocks(problem, source, index) == ((root, 3),)
    forward = propose_subtree_retile(
        problem,
        source,
        index,
        block_node_id=root,
        replacement_frontier=replacement,
        new_fractions_by_node={4: 0.6},
    )
    reverse = propose_subtree_retile(
        problem,
        forward.candidate,
        index,
        block_node_id=root,
        replacement_frontier=source.frontier,
        new_fractions_by_node={1: 0.3},
    )

    assert forward.valid and reverse.valid
    assert forward.move == "subtree_retile"
    assert forward.node_id == root
    assert forward.secondary_node_id is None
    assert forward.block_leaf_count == 3
    assert forward.candidate.k == source.k
    assert forward.log_q_forward_selection == 0.0
    assert forward.log_q_reverse_selection == 0.0
    assert forward.log_q_forward_auxiliary == pytest.approx(problem.prior.log_fraction_density(4, 0.6))
    assert forward.log_q_reverse_auxiliary == pytest.approx(problem.prior.log_fraction_density(1, 0.3))
    assert forward.delta_log_fraction_prior == pytest.approx(
        forward.log_q_forward_auxiliary - forward.log_q_reverse_auxiliary
    )
    assert forward.log_jacobian == 0.0
    assert forward.log_acceptance_ratio == pytest.approx(0.0, abs=1e-13)
    np.testing.assert_array_equal(forward.candidate.active_fractions, [0.4, 0.6])
    assert reverse.log_q_forward == pytest.approx(forward.log_q_reverse)
    assert reverse.log_q_reverse == pytest.approx(forward.log_q_forward)
    _assert_same_state(reverse.candidate, source)


def test_subtree_retile_normalizes_alternate_frontier_selection() -> None:
    """The exact subtree index supplies a nontrivial alternate-tiling count."""
    problem = _grid_problem((2, 4))
    root = problem.tree.root_id
    first, second = problem.tree.children(root)
    source_frontier = (
        DyadicFrontier.root(problem.tree)
        .split(problem.tree, root)
        .split(problem.tree, first)
        .split(problem.tree, second)
    )
    source = _state(problem, source_frontier, [0.4, 0.3, 0.6])
    index = SubtreePartitionIndex(problem.tree, max_k=4)
    source_rank = index.rank(root, source.k, source.frontier)
    replacement_rank = 0 if source_rank != 0 else 1
    replacement = index.unrank(root, source.k, replacement_rank)
    source_split = frozenset(source.frontier.active_split_nodes(problem.tree))
    candidate_split = frozenset(replacement.active_split_nodes(problem.tree))
    added = candidate_split - source_split
    proposed = {
        node_id: problem.prior.beta_parameters(node_id)[0] / sum(problem.prior.beta_parameters(node_id))
        for node_id in added
    }
    transition = propose_subtree_retile(
        problem,
        source,
        index,
        block_node_id=root,
        replacement_frontier=replacement,
        new_fractions_by_node=proposed,
    )

    alternatives = index.count(root, source.k) - 1
    assert alternatives > 1
    assert transition.valid
    assert transition.log_q_forward_selection == pytest.approx(-log(alternatives))
    assert transition.block_leaf_count == source.k


def test_subtree_retile_invalid_choices_are_self_transitions() -> None:
    """Retiling rejects wrong blocks, unchanged tilings, and fraction maps."""
    problem = _problem()
    root = problem.tree.root_id
    source_frontier = DyadicFrontier.root(problem.tree).split(problem.tree, root).split(problem.tree, 1)
    source = _state(problem, source_frontier, [0.4, 0.3])
    index = SubtreePartitionIndex(problem.tree, max_k=4)
    replacement = index.unrank(root, 3, 1 - index.rank(root, 3, source.frontier))
    transitions = (
        propose_subtree_retile(
            problem,
            source,
            index,
            block_node_id=2,
            replacement_frontier=replacement,
            new_fractions_by_node={4: 0.6},
        ),
        propose_subtree_retile(
            problem,
            source,
            index,
            block_node_id=root,
            replacement_frontier=source.frontier,
            new_fractions_by_node={},
        ),
        propose_subtree_retile(
            problem,
            source,
            index,
            block_node_id=root,
            replacement_frontier=replacement,
            new_fractions_by_node={},
        ),
        propose_subtree_retile(
            problem,
            source,
            index,
            block_node_id=root,
            replacement_frontier=replacement,
            new_fractions_by_node={4: 0.0},
        ),
    )
    for transition in transitions:
        assert not transition.valid
        assert transition.candidate is source
        assert transition.reason
        assert transition.log_acceptance_ratio == -np.inf

    other_index = SubtreePartitionIndex(_problem().tree, max_k=4)
    with pytest.raises(ValueError, match="problem.tree"):
        propose_subtree_retile(
            problem,
            source,
            other_index,
            block_node_id=root,
            replacement_frontier=replacement,
            new_fractions_by_node={4: 0.6},
        )


@pytest.mark.parametrize(
    "matrix_builder",
    [
        pytest.param(
            lambda problem, frontiers: _relocation_transition_matrix(
                problem,
                frontiers,
            ),
            id="relocation",
        ),
        pytest.param(
            lambda problem, frontiers: _subtree_retile_transition_matrix(
                problem,
                frontiers,
                SubtreePartitionIndex(problem.tree, max_k=4),
            ),
            id="subtree-retile",
        ),
    ],
)
def test_fixed_k_topology_kernels_preserve_exact_uniform_stationarity(
    matrix_builder,
) -> None:
    """Tiny prior-only topology matrices obey detailed balance exactly."""
    problem = _grid_problem((2, 4), likelihood_power=0.0)
    frontiers = enumerate_frontiers(problem.tree, k=4)
    transition_matrix = matrix_builder(problem, frontiers)
    uniform_probability = np.full(len(frontiers), 1.0 / len(frontiers))

    np.testing.assert_allclose(
        transition_matrix.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        transition_matrix,
        transition_matrix.T,
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        uniform_probability @ transition_matrix,
        uniform_probability,
        rtol=0.0,
        atol=2e-15,
    )
    assert np.array_equal(
        transition_matrix > 0.0,
        transition_matrix.T > 0.0,
    )


@pytest.mark.parametrize(
    "matrix_builder",
    [
        pytest.param(
            lambda problem, frontiers: _relocation_transition_matrix(
                problem,
                frontiers,
            ),
            id="relocation",
        ),
        pytest.param(
            lambda problem, frontiers: _subtree_retile_transition_matrix(
                problem,
                frontiers,
                SubtreePartitionIndex(problem.tree, max_k=4),
            ),
            id="subtree-retile",
        ),
    ],
)
def test_fixed_k_topology_kernels_connect_all_tiny_k4_frontiers(
    matrix_builder,
) -> None:
    """Each new fixed-K kernel connects the complete tiny K=4 state space."""
    problem = _grid_problem((2, 4), likelihood_power=0.0)
    frontiers = enumerate_frontiers(problem.tree, k=4)
    transition_matrix = matrix_builder(problem, frontiers)

    assert len(frontiers) == 5
    assert _reachable_positions(transition_matrix) == set(range(len(frontiers)))


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
