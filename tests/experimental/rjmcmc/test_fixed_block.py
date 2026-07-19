"""Focused tests for always-active fixed predictor blocks."""

from __future__ import annotations

from math import exp, log, pi
from typing import Any

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import stats

from openghg_inversions.experimental.rjmcmc.core import (
    FixedDesignBlock,
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
    lognormal_mu_sigma,
)
from openghg_inversions.experimental.rjmcmc.proposals import (
    TransitionTerms,
    propose_birth,
    propose_coefficient,
    propose_death,
    propose_fixed_coefficient,
    propose_global_move,
    propose_local_move,
)


def _problem_kwargs() -> dict[str, Any]:
    """Return constructor inputs for a small fixed-plus-dynamic problem."""
    return {
        "observations": np.array([6.0, 1.5, -0.5]),
        "observation_sd": np.array([0.8, 1.2, 0.5]),
        "sensitivities": np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.5, 0.0, 1.0, 0.0],
                [-1.0, 2.0, 0.0, 1.0],
            ]
        ),
        "grid_coordinates": np.arange(4, dtype=np.float64)[:, np.newaxis],
        "k_min": 1,
        "k_max": 3,
        "log_k_prior": np.log(np.array([0.2, 0.5, 0.3])),
        "coefficient_prior_mean": 1.0,
        "coefficient_prior_sd": 0.4,
    }


def _fixed_block() -> FixedDesignBlock:
    """Return two heterogeneous always-active design columns."""
    return FixedDesignBlock(
        design=np.array([[2.0, -1.0], [0.5, 1.0], [1.0, 0.0]]),
        coefficient_prior_mean=np.array([1.2, 0.9]),
        coefficient_prior_sd=np.array([0.3, 0.2]),
    )


def _fixed_problem() -> TransDimensionalProblem:
    """Return a problem with a nonzero offset and fixed design block."""
    return TransDimensionalProblem(
        **_problem_kwargs(),
        fixed_offset=np.array([0.1, -0.2, 0.3]),
        fixed_block=_fixed_block(),
    )


def _normal_log_density(value: float, mean: float, stdev: float) -> float:
    """Evaluate one normalized Gaussian density independently."""
    standardized = (value - mean) / stdev
    return -0.5 * standardized**2 - log(stdev) - 0.5 * log(2.0 * pi)


def _assert_state_parity(actual: TransDimensionalState, expected: TransDimensionalState) -> None:
    """Compare every cache introduced or affected by the fixed block."""
    assert actual.k == expected.k
    for actual_array, expected_array in (
        (actual.nuclei, expected.nuclei),
        (actual.coefficients, expected.coefficients),
        (actual.labels, expected.labels),
        (actual.design, expected.design),
        (actual.fixed_coefficients, expected.fixed_coefficients),
        (actual.dynamic_prediction, expected.dynamic_prediction),
        (actual.fixed_prediction, expected.fixed_prediction),
        (actual.prediction, expected.prediction),
        (actual.residual, expected.residual),
    ):
        assert_allclose(actual_array, expected_array, rtol=0.0, atol=1e-12)
    for actual_value, expected_value in (
        (actual.log_likelihood, expected.log_likelihood),
        (actual.log_coefficient_prior, expected.log_coefficient_prior),
        (actual.log_fixed_coefficient_prior, expected.log_fixed_coefficient_prior),
        (actual.log_k_prior, expected.log_k_prior),
        (actual.log_nucleus_prior, expected.log_nucleus_prior),
        (actual.log_target, expected.log_target),
    ):
        assert actual_value == pytest.approx(expected_value, rel=0.0, abs=1e-12)


def test_fixed_block_is_owned_read_only_and_validated() -> None:
    """Fixed-block inputs should be owned and protected from cache mutation."""
    design = np.array([[1.0, 2.0], [3.0, 4.0]])
    means = np.array([1.0, 2.0])
    standard_deviations = np.array([0.2, 0.4])
    block = FixedDesignBlock(design, means, standard_deviations)

    design[:] = -99.0
    means[:] = -99.0
    standard_deviations[:] = -99.0
    assert_array_equal(block.design, [[1.0, 2.0], [3.0, 4.0]])
    assert_array_equal(block.coefficient_prior_mean, [1.0, 2.0])
    assert_array_equal(block.coefficient_prior_sd, [0.2, 0.4])
    assert block.n_coefficients == 2
    assert not block.design.flags.writeable
    assert not block.coefficient_prior_mean.flags.writeable
    assert not block.coefficient_prior_sd.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        block.design[0, 0] = 2.0


@pytest.mark.parametrize(
    ("design", "means", "standard_deviations", "message"),
    [
        (np.ones(2), np.ones(2), np.ones(2), "two-dimensional"),
        (np.empty((2, 0)), np.empty(0), np.empty(0), "at least one column"),
        (np.array([[np.nan]]), np.ones(1), np.ones(1), "finite"),
        (np.ones((2, 2)), np.ones(1), np.ones(2), "one value per"),
        (np.ones((2, 2)), np.ones(2), np.ones(1), "one value per"),
        (np.ones((2, 1)), np.zeros(1), np.ones(1), "strictly positive"),
        (np.ones((2, 1)), np.ones(1), -np.ones(1), "strictly positive"),
    ],
)
def test_fixed_block_rejects_invalid_inputs(
    design: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
    message: str,
) -> None:
    """Malformed fixed designs and lognormal moments should fail explicitly."""
    with pytest.raises(ValueError, match=message):
        FixedDesignBlock(design, means, standard_deviations)


def test_problem_normalizes_and_validates_fixed_inputs() -> None:
    """Problems should own offsets and enforce fixed design observation rows."""
    no_block = TransDimensionalProblem(**_problem_kwargs())
    assert no_block.fixed_offset is not None
    assert_array_equal(no_block.fixed_offset, np.zeros(3))
    assert not no_block.fixed_offset.flags.writeable
    assert no_block.fixed_block is None
    assert no_block.n_fixed_coefficients == 0

    kwargs = _problem_kwargs()
    with pytest.raises(ValueError, match="same shape"):
        TransDimensionalProblem(**kwargs, fixed_offset=np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        TransDimensionalProblem(**kwargs, fixed_offset=np.array([0.0, np.nan, 0.0]))
    with pytest.raises(ValueError, match="one row per observation"):
        TransDimensionalProblem(
            **kwargs,
            fixed_block=FixedDesignBlock(np.ones((2, 1)), np.ones(1), np.ones(1)),
        )
    with pytest.raises(TypeError, match="FixedDesignBlock"):
        TransDimensionalProblem(**kwargs, fixed_block=np.ones((3, 1)))  # type: ignore[arg-type]


def test_fixed_state_matches_hand_calculation_and_normalized_prior() -> None:
    """All prediction components and normalized fixed priors should close exactly."""
    problem = _fixed_problem()
    fixed_coefficients = np.array([1.5, 0.8])
    state = build_state(
        problem,
        [0, 3],
        [0.8, 1.2],
        fixed_coefficients=fixed_coefficients,
    )

    expected_dynamic = state.design[:, : state.k] @ state.active_coefficients
    expected_fixed = np.array([0.1, -0.2, 0.3]) + _fixed_block().design @ fixed_coefficients
    assert_allclose(state.dynamic_prediction, expected_dynamic, rtol=0.0, atol=0.0)
    assert_allclose(state.fixed_prediction, expected_fixed, rtol=0.0, atol=0.0)
    assert_allclose(state.prediction, expected_dynamic + expected_fixed, rtol=0.0, atol=0.0)
    assert_allclose(state.residual, state.prediction - problem.observations, rtol=0.0, atol=0.0)

    expected_prior = 0.0
    assert problem.fixed_block is not None
    for value, mean, standard_deviation in zip(
        fixed_coefficients,
        problem.fixed_block.coefficient_prior_mean,
        problem.fixed_block.coefficient_prior_sd,
        strict=True,
    ):
        mu, sigma = lognormal_mu_sigma(float(mean), float(standard_deviation))
        expected_prior += float(stats.lognorm.logpdf(value, s=sigma, scale=exp(mu)))
    assert state.log_fixed_coefficient_prior == pytest.approx(expected_prior, abs=1e-14)
    assert state.log_target == pytest.approx(
        state.log_likelihood
        + state.log_coefficient_prior
        + expected_prior
        + state.log_k_prior
        + state.log_nucleus_prior,
        abs=1e-14,
    )

    fixed_coefficients[:] = -100.0
    assert_array_equal(state.fixed_coefficients, [1.5, 0.8])
    for array in (
        state.fixed_coefficients,
        state.dynamic_prediction,
        state.fixed_prediction,
        state.prediction,
        state.residual,
    ):
        assert not array.flags.writeable


@pytest.mark.parametrize(
    ("fixed_coefficients", "message"),
    [
        (None, "required"),
        ([[1.0, 1.0]], "one-dimensional"),
        ([1.0], "one value per"),
        ([1.0, 1.0, 1.0], "one value per"),
        ([1.0, np.inf], "finite"),
    ],
)
def test_fixed_state_requires_explicit_valid_coefficients(
    fixed_coefficients: object,
    message: str,
) -> None:
    """A nonempty fixed block should require one explicit finite value per column."""
    with pytest.raises(ValueError, match=message):
        build_state(
            _fixed_problem(),
            [0],
            [1.0],
            fixed_coefficients=fixed_coefficients,  # type: ignore[arg-type]
        )


def test_nonpositive_fixed_coefficient_has_zero_prior_density() -> None:
    """Finite fixed coefficients outside lognormal support should remain explicit states."""
    state = build_state(
        _fixed_problem(),
        [0],
        [1.0],
        fixed_coefficients=[1.0, 0.0],
    )

    assert state.log_fixed_coefficient_prior == -np.inf
    assert state.log_target == -np.inf


def test_legacy_no_block_state_is_numerically_unchanged() -> None:
    """The no-block path should reduce exactly to the original dynamic model."""
    problem = TransDimensionalProblem(**_problem_kwargs())
    explicit_zero_problem = TransDimensionalProblem(
        **_problem_kwargs(),
        fixed_offset=np.zeros(3),
    )
    state = build_state(problem, [0, 3], [0.8, 1.2])
    explicit_zero_state = build_state(explicit_zero_problem, [0, 3], [0.8, 1.2])

    _assert_state_parity(state, explicit_zero_state)
    assert state.fixed_coefficients.shape == (0,)
    assert_array_equal(state.fixed_prediction, np.zeros(problem.nobs))
    assert_array_equal(state.dynamic_prediction, state.prediction)
    assert state.log_fixed_coefficient_prior == 0.0
    assert state.log_target == (
        state.log_likelihood + state.log_coefficient_prior + state.log_k_prior + state.log_nucleus_prior
    )


def test_fixed_coefficient_proposal_is_reciprocal_with_explicit_q_terms() -> None:
    """An always-active coefficient update should expose reciprocal proposal terms."""
    problem = _fixed_problem()
    source = build_state(problem, [0, 3], [0.8, 1.2], fixed_coefficients=[1.5, 0.8])
    forward = propose_fixed_coefficient(
        problem,
        source,
        coefficient_position=1,
        proposed_coefficient=0.95,
        proposal_stdev=0.25,
    )
    reverse = propose_fixed_coefficient(
        problem,
        forward.candidate,
        coefficient_position=1,
        proposed_coefficient=0.8,
        proposal_stdev=0.25,
    )

    expected_forward = -log(2.0) + _normal_log_density(0.95, 0.8, 0.25)
    expected_reverse = -log(2.0) + _normal_log_density(0.8, 0.95, 0.25)
    assert forward.move == "fixed_coefficient"
    assert forward.log_q_forward == pytest.approx(expected_forward)
    assert forward.log_q_reverse == pytest.approx(expected_reverse)
    assert forward.log_jacobian == 0.0
    assert_allclose(forward.candidate.dynamic_prediction, source.dynamic_prediction)
    assert_allclose(
        forward.candidate.fixed_prediction - source.fixed_prediction,
        problem.fixed_block.design[:, 1] * 0.15,  # type: ignore[union-attr]
        rtol=0.0,
        atol=1e-15,
    )
    _assert_state_parity(reverse.candidate, source)
    assert reverse.log_acceptance_ratio == pytest.approx(-forward.log_acceptance_ratio, abs=1e-12)

    log_forward_flux = source.log_target + forward.log_q_forward + min(0.0, forward.log_acceptance_ratio)
    log_reverse_flux = (
        forward.candidate.log_target + reverse.log_q_forward + min(0.0, reverse.log_acceptance_ratio)
    )
    assert exp(log_forward_flux) == pytest.approx(exp(log_reverse_flux), rel=1e-12)


def test_fixed_coefficient_proposal_rejects_absent_block_and_invalid_value() -> None:
    """Missing fixed positions and values outside prior support should self-transition."""
    no_block_problem = TransDimensionalProblem(**_problem_kwargs())
    no_block_state = build_state(no_block_problem, [0], [1.0])
    absent = propose_fixed_coefficient(
        no_block_problem,
        no_block_state,
        coefficient_position=0,
        proposed_coefficient=1.0,
        proposal_stdev=0.2,
    )
    fixed_problem = _fixed_problem()
    fixed_state = build_state(fixed_problem, [0], [1.0], fixed_coefficients=[1.0, 1.0])
    invalid = propose_fixed_coefficient(
        fixed_problem,
        fixed_state,
        coefficient_position=0,
        proposed_coefficient=0.0,
        proposal_stdev=0.2,
    )

    assert not absent.valid
    assert absent.candidate is no_block_state
    assert not invalid.valid
    assert invalid.candidate is fixed_state


def test_all_existing_proposals_preserve_fixed_coefficients() -> None:
    """Dynamic structural and coefficient candidates should retain the fixed block."""
    problem = _fixed_problem()
    source = build_state(problem, [0, 3], [0.8, 1.2], fixed_coefficients=[1.5, 0.8])
    transitions: tuple[TransitionTerms, ...] = (
        propose_coefficient(
            problem,
            source,
            coefficient_position=0,
            proposed_coefficient=0.9,
            proposal_stdev=0.2,
        ),
        propose_birth(
            problem,
            source,
            new_nucleus=2,
            proposed_coefficient=1.1,
            proposal_stdev=0.3,
        ),
        propose_death(problem, source, remove_position=0, proposal_stdev=0.3),
        propose_global_move(problem, source, move_position=0, new_nucleus=1),
        propose_local_move(
            problem,
            source,
            move_position=0,
            new_nucleus=1,
            proposal_scale=1.0,
        ),
    )

    for transition in transitions:
        assert transition.valid
        assert_array_equal(transition.candidate.fixed_coefficients, source.fixed_coefficients)
        assert_allclose(transition.candidate.fixed_prediction, source.fixed_prediction)


def test_dimension_changing_pair_keeps_fixed_prior_and_detailed_balance() -> None:
    """A nonzero fixed block should affect likelihood without altering RJ reciprocity."""
    problem = _fixed_problem()
    source = build_state(problem, [0, 3], [0.8, 1.2], fixed_coefficients=[1.5, 0.8])
    upward = propose_birth(
        problem,
        source,
        new_nucleus=2,
        proposed_coefficient=1.1,
        proposal_stdev=0.3,
    )
    remove_position = int(np.flatnonzero(upward.candidate.active_nuclei == 2)[0])
    downward = propose_death(
        problem,
        upward.candidate,
        remove_position=remove_position,
        proposal_stdev=0.3,
    )

    assert upward.candidate.log_fixed_coefficient_prior == source.log_fixed_coefficient_prior
    assert_allclose(upward.candidate.fixed_prediction, source.fixed_prediction)
    assert downward.log_q_forward == pytest.approx(upward.log_q_reverse)
    assert downward.log_q_reverse == pytest.approx(upward.log_q_forward)
    _assert_state_parity(downward.candidate, source)
    assert downward.log_acceptance_ratio == pytest.approx(-upward.log_acceptance_ratio, abs=1e-12)


def test_fixed_block_has_numpy_numba_parity() -> None:
    """Fixed caches and priors should not depend on the dynamic-kernel backend."""
    problem = _fixed_problem()
    numpy_state = build_state(
        problem,
        [3, 0],
        [1.2, 0.8],
        fixed_coefficients=[1.5, 0.8],
        backend="numpy",
    )
    numba_state = build_state(
        problem,
        [3, 0],
        [1.2, 0.8],
        fixed_coefficients=[1.5, 0.8],
        backend="numba",
    )

    _assert_state_parity(numba_state, numpy_state)
    numpy_transition = propose_fixed_coefficient(
        problem,
        numpy_state,
        coefficient_position=0,
        proposed_coefficient=1.4,
        proposal_stdev=0.2,
        backend="numpy",
    )
    numba_transition = propose_fixed_coefficient(
        problem,
        numba_state,
        coefficient_position=0,
        proposed_coefficient=1.4,
        proposal_stdev=0.2,
        backend="numba",
    )
    _assert_state_parity(numba_transition.candidate, numpy_transition.candidate)
    assert numba_transition.log_acceptance_ratio == pytest.approx(
        numpy_transition.log_acceptance_ratio,
        abs=1e-12,
    )
