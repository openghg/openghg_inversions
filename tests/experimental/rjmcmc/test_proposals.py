"""Focused tests for deterministic spatial TDMCMC proposal accounting."""

from __future__ import annotations

from collections.abc import Callable
from math import exp, log, pi

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.core import (
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
)
from openghg_inversions.experimental.rjmcmc.proposals import (
    TransitionTerms,
    accept_or_reject,
    propose_birth,
    propose_coefficient,
    propose_death,
    propose_global_move,
)


ProposalCall = Callable[[str], TransitionTerms]


@pytest.fixture
def problem() -> TransDimensionalProblem:
    """Return a small heterogeneous four-cell proposal problem."""
    return TransDimensionalProblem(
        observations=np.array([5.0, 1.0, -0.5]),
        observation_sd=np.array([0.8, 1.2, 0.5]),
        sensitivities=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.5, 0.0, 1.0, 0.0],
                [-1.0, 2.0, 0.0, 1.0],
            ]
        ),
        grid_coordinates=np.arange(4, dtype=float)[:, np.newaxis],
        k_min=1,
        k_max=3,
        log_k_prior=np.log(np.array([0.2, 0.5, 0.3])),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.4,
    )


@pytest.fixture
def state(problem: TransDimensionalProblem) -> TransDimensionalState:
    """Return a canonical two-nucleus source state."""
    return build_state(problem, [0, 3], [0.8, 1.2])


def _normal_log_density(value: float, mean: float, stdev: float) -> float:
    """Evaluate a normalized Gaussian log density independently."""
    standardized = (value - mean) / stdev
    return -0.5 * standardized**2 - log(stdev) - 0.5 * log(2.0 * pi)


def _assert_states_close(
    actual: TransDimensionalState,
    expected: TransDimensionalState,
) -> None:
    """Compare every discrete and cached floating-point state component."""
    assert actual.k == expected.k
    np.testing.assert_array_equal(actual.nuclei, expected.nuclei)
    np.testing.assert_array_equal(actual.coefficients, expected.coefficients)
    np.testing.assert_array_equal(actual.labels, expected.labels)
    for actual_array, expected_array in (
        (actual.design, expected.design),
        (actual.prediction, expected.prediction),
        (actual.residual, expected.residual),
    ):
        np.testing.assert_allclose(actual_array, expected_array, rtol=0.0, atol=1e-12)
    for actual_value, expected_value in (
        (actual.log_likelihood, expected.log_likelihood),
        (actual.log_coefficient_prior, expected.log_coefficient_prior),
        (actual.log_k_prior, expected.log_k_prior),
        (actual.log_nucleus_prior, expected.log_nucleus_prior),
        (actual.log_target, expected.log_target),
    ):
        assert actual_value == pytest.approx(expected_value, rel=0.0, abs=1e-12)


def _assert_transition_parity(actual: TransitionTerms, expected: TransitionTerms) -> None:
    """Compare candidate caches and every reported proposal term."""
    assert actual.move == expected.move
    assert actual.valid is expected.valid
    assert actual.reason == expected.reason
    _assert_states_close(actual.candidate, expected.candidate)
    for actual_value, expected_value in (
        (actual.log_target_delta, expected.log_target_delta),
        (actual.log_q_forward, expected.log_q_forward),
        (actual.log_q_reverse, expected.log_q_reverse),
        (actual.log_jacobian, expected.log_jacobian),
        (actual.log_acceptance_ratio, expected.log_acceptance_ratio),
    ):
        assert actual_value == pytest.approx(expected_value, rel=0.0, abs=1e-12)


def test_coefficient_proposal_exposes_complete_log_alpha(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
) -> None:
    """A coefficient update should report normalized forward and reverse terms."""
    transition = propose_coefficient(
        problem,
        state,
        coefficient_position=0,
        proposed_coefficient=0.95,
        proposal_stdev=0.25,
    )
    expected_forward = -log(state.k) + _normal_log_density(0.95, 0.8, 0.25)
    expected_reverse = -log(state.k) + _normal_log_density(0.8, 0.95, 0.25)
    expected_delta = transition.candidate.log_target - state.log_target

    assert transition.valid
    assert transition.log_target_delta == pytest.approx(expected_delta)
    assert transition.log_q_forward == pytest.approx(expected_forward)
    assert transition.log_q_reverse == pytest.approx(expected_reverse)
    assert transition.log_jacobian == 0.0
    assert transition.log_acceptance_ratio == pytest.approx(
        expected_delta + expected_reverse - expected_forward
    )


def test_paired_birth_death_ratios_and_pointwise_flux_match(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
) -> None:
    """A forced birth/death edge should be reciprocal and satisfy detailed balance."""
    birth = propose_birth(
        problem,
        state,
        new_nucleus=2,
        proposed_coefficient=1.1,
        proposal_stdev=0.3,
    )
    removed_position = int(np.flatnonzero(birth.candidate.active_nuclei == 2)[0])
    death = propose_death(
        problem,
        birth.candidate,
        remove_position=removed_position,
        proposal_stdev=0.3,
    )
    expected_birth_forward = -log(problem.n_grid_cells - state.k) + _normal_log_density(
        1.1,
        1.2,
        0.3,
    )

    assert birth.log_q_forward == pytest.approx(expected_birth_forward)
    assert birth.log_q_reverse == pytest.approx(-log(3.0))
    assert death.log_q_forward == pytest.approx(birth.log_q_reverse)
    assert death.log_q_reverse == pytest.approx(birth.log_q_forward)
    _assert_states_close(death.candidate, state)
    assert death.log_acceptance_ratio == pytest.approx(-birth.log_acceptance_ratio, abs=1e-12)

    log_forward_flux = state.log_target + birth.log_q_forward + min(0.0, birth.log_acceptance_ratio)
    log_reverse_flux = birth.candidate.log_target + death.log_q_forward + min(0.0, death.log_acceptance_ratio)
    assert exp(log_forward_flux) == pytest.approx(exp(log_reverse_flux), rel=1e-12)


def test_global_move_is_symmetric_and_reversible(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
) -> None:
    """A uniform global move should have equal normalized proposal probabilities."""
    forward = propose_global_move(
        problem,
        state,
        move_position=0,
        new_nucleus=1,
    )
    reverse_position = int(np.flatnonzero(forward.candidate.active_nuclei == 1)[0])
    reverse = propose_global_move(
        problem,
        forward.candidate,
        move_position=reverse_position,
        new_nucleus=0,
    )
    expected_log_q = -log(state.k) - log(problem.n_grid_cells - state.k)

    assert forward.log_q_forward == pytest.approx(expected_log_q)
    assert forward.log_q_reverse == pytest.approx(expected_log_q)
    assert reverse.log_q_forward == pytest.approx(expected_log_q)
    _assert_states_close(reverse.candidate, state)
    assert reverse.log_acceptance_ratio == pytest.approx(-forward.log_acceptance_ratio, abs=1e-12)


@pytest.mark.parametrize(
    "case",
    [
        "occupied-birth",
        "minimum-death",
        "occupied-move",
        "nonpositive-coefficient",
        "maximum-birth",
    ],
)
def test_invalid_proposals_are_nonmutating_self_transitions(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    case: str,
) -> None:
    """Invalid choices and dimension boundaries should retain the source object."""
    if case == "occupied-birth":
        source = state
        transition = propose_birth(
            problem,
            source,
            new_nucleus=0,
            proposed_coefficient=1.0,
            proposal_stdev=0.3,
        )
    elif case == "minimum-death":
        source = build_state(problem, [0], [1.0])
        transition = propose_death(problem, source, remove_position=0, proposal_stdev=0.3)
    elif case == "occupied-move":
        source = state
        transition = propose_global_move(problem, source, move_position=0, new_nucleus=3)
    elif case == "nonpositive-coefficient":
        source = state
        transition = propose_coefficient(
            problem,
            source,
            coefficient_position=0,
            proposed_coefficient=0.0,
            proposal_stdev=0.3,
        )
    else:
        source = build_state(problem, [0, 1, 3], [0.8, 1.0, 1.2])
        transition = propose_birth(
            problem,
            source,
            new_nucleus=2,
            proposed_coefficient=1.1,
            proposal_stdev=0.3,
        )

    nuclei_before = source.nuclei.copy()
    coefficients_before = source.coefficients.copy()
    assert not transition.valid
    assert transition.reason
    assert transition.candidate is source
    assert transition.log_acceptance_ratio == -np.inf
    assert accept_or_reject(source, transition, log_uniform=-np.inf) is source
    np.testing.assert_array_equal(source.nuclei, nuclei_before)
    np.testing.assert_array_equal(source.coefficients, coefficients_before)


def test_accept_or_reject_uses_strict_truncated_log_threshold(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
) -> None:
    """Injected log-uniform values should exercise both acceptance boundaries."""
    transition = propose_coefficient(
        problem,
        state,
        coefficient_position=1,
        proposed_coefficient=1.1,
        proposal_stdev=0.2,
    )
    threshold = min(0.0, transition.log_acceptance_ratio)

    assert accept_or_reject(state, transition, log_uniform=threshold - 1e-12) is transition.candidate
    assert accept_or_reject(state, transition, log_uniform=threshold) is state
    with pytest.raises(ValueError, match="non-positive"):
        accept_or_reject(state, transition, log_uniform=0.1)
    with pytest.raises(ValueError, match="NaN"):
        accept_or_reject(state, transition, log_uniform=np.nan)


def test_numpy_and_numba_proposal_candidates_have_parity(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
) -> None:
    """Every forced proposal should report backend-independent candidates and terms."""
    calls: tuple[ProposalCall, ...] = (
        lambda backend: propose_coefficient(
            problem,
            state,
            coefficient_position=0,
            proposed_coefficient=0.95,
            proposal_stdev=0.25,
            backend=backend,  # type: ignore[arg-type]
        ),
        lambda backend: propose_birth(
            problem,
            state,
            new_nucleus=2,
            proposed_coefficient=1.1,
            proposal_stdev=0.3,
            backend=backend,  # type: ignore[arg-type]
        ),
        lambda backend: propose_death(
            problem,
            state,
            remove_position=0,
            proposal_stdev=0.3,
            backend=backend,  # type: ignore[arg-type]
        ),
        lambda backend: propose_global_move(
            problem,
            state,
            move_position=0,
            new_nucleus=1,
            backend=backend,  # type: ignore[arg-type]
        ),
    )

    for call in calls:
        _assert_transition_parity(call("numba"), call("numpy"))
