"""Tests for the boundary-aware discrete Gaussian nucleus move."""

from __future__ import annotations

from math import exp, log

import numpy as np
import pytest

from openghg_inversions.tdmcmc.core import (
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
)
from openghg_inversions.tdmcmc.proposals import propose_local_move


@pytest.fixture
def problem() -> TransDimensionalProblem:
    """Return an irregular one-dimensional problem with asymmetric boundaries."""
    return TransDimensionalProblem(
        observations=np.array([2.0, -0.5]),
        observation_sd=np.array([0.7, 1.1]),
        sensitivities=np.array(
            [
                [1.0, 2.0, -0.5, 0.25],
                [0.0, 1.5, 2.0, -1.0],
            ]
        ),
        grid_coordinates=np.array([[0.0], [1.0], [3.0], [10.0]]),
        k_min=1,
        k_max=3,
        log_k_prior=np.log(np.array([0.2, 0.5, 0.3])),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.4,
    )


@pytest.fixture
def state(problem: TransDimensionalProblem) -> TransDimensionalState:
    """Return a two-nucleus state spanning both ends of the test grid."""
    return build_state(problem, [0, 3], [0.8, 1.2])


def _assert_states_close(
    actual: TransDimensionalState,
    expected: TransDimensionalState,
) -> None:
    """Compare complete fixed-capacity states, including derived caches."""
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


def test_local_move_uses_independently_normalized_boundary_weights(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
) -> None:
    """Forward and reverse terms should match independent boundary calculations."""
    scale = 1.0
    transition = propose_local_move(
        problem,
        state,
        move_position=0,
        new_nucleus=1,
        proposal_scale=scale,
    )
    forward_weights = np.exp(-np.square(np.array([1.0, 3.0])) / (2.0 * scale**2))
    reverse_weights = np.exp(-np.square(np.array([1.0, 2.0])) / (2.0 * scale**2))
    expected_forward = -log(state.k) + log(float(forward_weights[0] / forward_weights.sum()))
    expected_reverse = -log(state.k) + log(float(reverse_weights[0] / reverse_weights.sum()))

    assert transition.valid
    assert transition.log_q_forward == pytest.approx(expected_forward)
    assert transition.log_q_reverse == pytest.approx(expected_reverse)
    assert transition.log_q_forward != pytest.approx(transition.log_q_reverse)
    assert transition.log_jacobian == 0.0


def test_forward_reverse_moves_are_inverse_and_balance_pointwise_flux(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
) -> None:
    """A forced move and its reverse should have reciprocal log ratios and flux."""
    forward = propose_local_move(
        problem,
        state,
        move_position=0,
        new_nucleus=1,
        proposal_scale=1.0,
    )
    reverse_position = int(np.flatnonzero(forward.candidate.active_nuclei == 1)[0])
    reverse = propose_local_move(
        problem,
        forward.candidate,
        move_position=reverse_position,
        new_nucleus=0,
        proposal_scale=1.0,
    )

    _assert_states_close(reverse.candidate, state)
    assert reverse.log_q_forward == pytest.approx(forward.log_q_reverse)
    assert reverse.log_q_reverse == pytest.approx(forward.log_q_forward)
    assert reverse.log_acceptance_ratio == pytest.approx(-forward.log_acceptance_ratio)
    log_forward_flux = state.log_target + forward.log_q_forward + min(0.0, forward.log_acceptance_ratio)
    log_reverse_flux = (
        forward.candidate.log_target + reverse.log_q_forward + min(0.0, reverse.log_acceptance_ratio)
    )
    assert exp(log_forward_flux) == pytest.approx(exp(log_reverse_flux), rel=1e-12)


def test_local_move_preserves_coefficient_identity_after_canonical_sort(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
) -> None:
    """The selected nucleus should retain its coefficient when order changes."""
    transition = propose_local_move(
        problem,
        state,
        move_position=1,
        new_nucleus=1,
        proposal_scale=2.0,
    )

    np.testing.assert_array_equal(transition.candidate.active_nuclei, np.array([0, 1]))
    np.testing.assert_allclose(transition.candidate.active_coefficients, np.array([0.8, 1.2]))


def test_local_move_has_numpy_numba_backend_parity(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
) -> None:
    """NumPy and Numba state builders should yield the same transition."""
    numpy_transition = propose_local_move(
        problem,
        state,
        move_position=0,
        new_nucleus=1,
        proposal_scale=1.3,
        backend="numpy",
    )
    numba_transition = propose_local_move(
        problem,
        state,
        move_position=0,
        new_nucleus=1,
        proposal_scale=1.3,
        backend="numba",
    )

    _assert_states_close(numba_transition.candidate, numpy_transition.candidate)
    assert numba_transition.log_q_forward == pytest.approx(numpy_transition.log_q_forward)
    assert numba_transition.log_q_reverse == pytest.approx(numpy_transition.log_q_reverse)
    assert numba_transition.log_acceptance_ratio == pytest.approx(numpy_transition.log_acceptance_ratio)


@pytest.mark.parametrize(
    ("move_position", "new_nucleus"),
    [(-1, 1), (2, 1), (0, -1), (0, 4), (0, 3)],
)
def test_invalid_local_moves_are_nonmutating_self_transitions(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    move_position: int,
    new_nucleus: int,
) -> None:
    """Invalid positions, cells, and occupied destinations should not mutate state."""
    nuclei_before = state.nuclei.copy()
    coefficients_before = state.coefficients.copy()
    transition = propose_local_move(
        problem,
        state,
        move_position=move_position,
        new_nucleus=new_nucleus,
        proposal_scale=1.0,
    )

    assert not transition.valid
    assert transition.reason
    assert transition.candidate is state
    assert transition.log_acceptance_ratio == -np.inf
    np.testing.assert_array_equal(state.nuclei, nuclei_before)
    np.testing.assert_array_equal(state.coefficients, coefficients_before)


@pytest.mark.parametrize("proposal_scale", [0.0, -1.0, np.inf, np.nan])
def test_invalid_local_move_scale_raises(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    proposal_scale: float,
) -> None:
    """A nonpositive or nonfinite local proposal scale should raise."""
    with pytest.raises(ValueError, match="proposal_scale must be finite and positive"):
        propose_local_move(
            problem,
            state,
            move_position=0,
            new_nucleus=1,
            proposal_scale=proposal_scale,
        )
