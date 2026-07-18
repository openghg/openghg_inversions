"""Finite-state oracle tests for the discrete local nucleus-location kernel.

These tests enumerate only a fixed-``k``, fixed-coefficient location subspace.
They do not claim to enumerate the sampler's continuous coefficient proposals
or its trans-dimensional birth and death moves.
"""

from __future__ import annotations

from itertools import permutations
from math import exp, log

import numpy as np
import pytest

from openghg_inversions.tdmcmc.core import (
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
)
from openghg_inversions.tdmcmc.proposals import propose_local_move


StateKey = tuple[tuple[int, float], ...]
_COEFFICIENTS = (0.7, 1.6)
_PROPOSAL_SCALE = 1.3


@pytest.fixture
def problem() -> TransDimensionalProblem:
    """Return a four-cell problem with irregular one-dimensional boundaries."""
    return TransDimensionalProblem(
        observations=np.array([1.0, -0.4]),
        observation_sd=np.array([0.8, 1.1]),
        sensitivities=np.array(
            [
                [1.0, 0.2, -0.3, 0.7],
                [0.1, -0.5, 1.2, 0.4],
            ]
        ),
        grid_coordinates=np.array([[0.0], [1.0], [3.0], [7.0]]),
        k_min=2,
        k_max=2,
        log_k_prior=np.array([0.0]),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.4,
    )


def _state_key(state: TransDimensionalState) -> StateKey:
    """Encode coefficient-bearing nuclei independently of canonical array order."""
    return tuple(
        (int(nucleus), float(coefficient))
        for nucleus, coefficient in zip(
            state.active_nuclei,
            state.active_coefficients,
            strict=True,
        )
    )


def _enumerate_location_states(problem: TransDimensionalProblem) -> list[TransDimensionalState]:
    """Enumerate every injective placement of two distinct fixed coefficients."""
    states_by_key: dict[StateKey, TransDimensionalState] = {}
    for locations in permutations(range(problem.ncell), len(_COEFFICIENTS)):
        state = build_state(problem, locations, _COEFFICIENTS)
        states_by_key[_state_key(state)] = state
    assert len(states_by_key) == 12
    return [states_by_key[key] for key in sorted(states_by_key)]


def _valid_choices(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
) -> list[tuple[int, int]]:
    """Return every selected-nucleus and unoccupied-destination pair."""
    occupied = set(int(nucleus) for nucleus in state.active_nuclei)
    return [
        (position, destination)
        for position in range(state.k)
        for destination in range(problem.ncell)
        if destination not in occupied
    ]


def _oracle_proposal_row(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    proposal_scale: float,
) -> dict[StateKey, float]:
    """Calculate one normalized location-proposal row without product helpers."""
    source_pairs = list(_state_key(state))
    occupied = {nucleus for nucleus, _ in source_pairs}
    row: dict[StateKey, float] = {}
    for position, (origin, coefficient) in enumerate(source_pairs):
        destinations = np.array(
            [cell for cell in range(problem.ncell) if cell not in occupied],
            dtype=np.int64,
        )
        offsets = problem.grid_coordinates[destinations] - problem.grid_coordinates[origin]
        squared_distances = np.sum(np.square(offsets), axis=1)
        weights = np.exp(-0.5 * squared_distances / proposal_scale**2)
        conditional_probabilities = weights / weights.sum()
        for destination, conditional_probability in zip(
            destinations,
            conditional_probabilities,
            strict=True,
        ):
            candidate_pairs = source_pairs.copy()
            candidate_pairs[position] = (int(destination), coefficient)
            candidate_key = tuple(sorted(candidate_pairs))
            row[candidate_key] = row.get(candidate_key, 0.0) + float(conditional_probability / state.k)
    return row


def test_local_proposal_matches_boundary_normalized_finite_oracle(
    problem: TransDimensionalProblem,
) -> None:
    """Product proposal terms should match exact finite-grid row normalization."""
    states = _enumerate_location_states(problem)
    states_by_key = {_state_key(state): state for state in states}
    source_key: StateKey = ((0, 0.7), (3, 1.6))
    source = states_by_key[source_key]
    oracle_row = _oracle_proposal_row(problem, source, _PROPOSAL_SCALE)

    assert source_key not in oracle_row
    assert sum(oracle_row.values()) == pytest.approx(1.0, rel=0.0, abs=1e-15)
    assert len(oracle_row) == source.k * (problem.ncell - source.k)

    near_weight = exp(-0.5 * (1.0 / _PROPOSAL_SCALE) ** 2)
    far_weight = exp(-0.5 * (3.0 / _PROPOSAL_SCALE) ** 2)
    near_candidate_key: StateKey = ((1, 0.7), (3, 1.6))
    expected_near_probability = 0.5 * near_weight / (near_weight + far_weight)
    assert oracle_row[near_candidate_key] == pytest.approx(expected_near_probability)

    product_row: dict[StateKey, float] = {}
    for position, destination in _valid_choices(problem, source):
        transition = propose_local_move(
            problem,
            source,
            move_position=position,
            new_nucleus=destination,
            proposal_scale=_PROPOSAL_SCALE,
        )
        candidate_key = _state_key(transition.candidate)
        product_row[candidate_key] = exp(transition.log_q_forward)
        reverse_oracle = _oracle_proposal_row(
            problem,
            transition.candidate,
            _PROPOSAL_SCALE,
        )[source_key]

        assert transition.valid
        assert exp(transition.log_q_forward) == pytest.approx(oracle_row[candidate_key])
        assert exp(transition.log_q_reverse) == pytest.approx(reverse_oracle)

    assert product_row == pytest.approx(oracle_row)
    assert sum(product_row.values()) == pytest.approx(1.0, rel=0.0, abs=1e-15)

    near_transition = propose_local_move(
        problem,
        source,
        move_position=0,
        new_nucleus=1,
        proposal_scale=_PROPOSAL_SCALE,
    )
    assert near_transition.log_q_forward != pytest.approx(near_transition.log_q_reverse)


def test_enumerated_local_mh_kernel_is_stochastic_reversible_and_stationary(
    problem: TransDimensionalProblem,
) -> None:
    """The complete finite location-only MH kernel should preserve its target."""
    states = _enumerate_location_states(problem)
    keys = [_state_key(state) for state in states]
    indices = {key: index for index, key in enumerate(keys)}
    oracle_rows = [_oracle_proposal_row(problem, state, _PROPOSAL_SCALE) for state in states]
    transition_matrix = np.zeros((len(states), len(states)), dtype=np.float64)
    rejected_mass = np.zeros(len(states), dtype=np.float64)

    for source_index, source in enumerate(states):
        source_key = keys[source_index]
        assert sum(oracle_rows[source_index].values()) == pytest.approx(
            1.0,
            rel=0.0,
            abs=1e-15,
        )
        for position, destination in _valid_choices(problem, source):
            transition = propose_local_move(
                problem,
                source,
                move_position=position,
                new_nucleus=destination,
                proposal_scale=_PROPOSAL_SCALE,
            )
            candidate_key = _state_key(transition.candidate)
            candidate_index = indices[candidate_key]
            q_forward = oracle_rows[source_index][candidate_key]
            q_reverse = oracle_rows[candidate_index][source_key]
            log_alpha = min(
                0.0,
                transition.candidate.log_target - source.log_target + log(q_reverse) - log(q_forward),
            )
            acceptance_probability = exp(log_alpha)

            assert exp(transition.log_q_forward) == pytest.approx(q_forward)
            assert exp(transition.log_q_reverse) == pytest.approx(q_reverse)
            assert min(0.0, transition.log_acceptance_ratio) == pytest.approx(log_alpha)

            transition_matrix[source_index, candidate_index] += q_forward * acceptance_probability
            rejected_mass[source_index] += q_forward * (1.0 - acceptance_probability)

        transition_matrix[source_index, source_index] += rejected_mass[source_index]

    np.testing.assert_allclose(transition_matrix.sum(axis=1), 1.0, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        np.diag(transition_matrix),
        rejected_mass,
        rtol=0.0,
        atol=1e-15,
    )
    assert np.any(rejected_mass > 0.0)
    assert np.all(transition_matrix >= 0.0)

    log_target = np.array([state.log_target for state in states])
    target_probability = np.exp(log_target - np.max(log_target))
    target_probability /= target_probability.sum()
    stationary_probability = target_probability @ transition_matrix
    probability_flux = target_probability[:, np.newaxis] * transition_matrix

    np.testing.assert_allclose(
        stationary_probability,
        target_probability,
        rtol=2e-13,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        probability_flux,
        probability_flux.T,
        rtol=2e-13,
        atol=2e-15,
    )
