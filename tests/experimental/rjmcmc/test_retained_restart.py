"""Regression tests for retained collection and exact in-memory continuation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.core import (
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
    uniform_log_k_prior,
)
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings
from openghg_inversions.experimental.rjmcmc.sampling import (
    SCHEDULE_ID,
    SamplerConfig,
    SamplingResult,
    continue_sample,
    sample,
)


def _problem() -> TransDimensionalProblem:
    """Return a small informative problem with several structural states."""
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
        log_k_prior=uniform_log_k_prior(1, 3),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.4,
    )


def _config(*, iterations: int, nucleus_move: str) -> SamplerConfig:
    """Return common seeded global or local kernel settings."""
    return SamplerConfig(
        iterations=iterations,
        coefficient_proposal_sd=0.15,
        birth_proposal_sd=0.25,
        seed=481,
        nucleus_move=nucleus_move,  # type: ignore[arg-type]
        local_move_scale=0.8 if nucleus_move == "local" else None,
    )


def _assert_state_equal(actual: TransDimensionalState, expected: TransDimensionalState) -> None:
    """Assert exact equality for every cached state component."""
    for state_field in fields(TransDimensionalState):
        actual_value = getattr(actual, state_field.name)
        expected_value = getattr(expected, state_field.name)
        if isinstance(actual_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def _assert_split_matches_full(
    full: SamplingResult,
    first: SamplingResult,
    continued: SamplingResult,
) -> None:
    """Compare concatenated retained states, diagnostics, and terminal data."""
    full_transition = full.trace.state_transition
    first_transition = first.trace.state_transition
    continued_transition = continued.trace.state_transition
    assert full_transition is not None
    assert first_transition is not None
    assert continued_transition is not None
    np.testing.assert_array_equal(
        np.concatenate((first_transition, continued_transition)),
        full_transition,
    )
    for name in ("k", "nuclei", "coefficients", "log_target"):
        np.testing.assert_array_equal(
            np.concatenate((getattr(first.trace, name), getattr(continued.trace, name))),
            getattr(full.trace, name),
        )
    for name in ("moves", "accepted", "log_acceptance_ratio"):
        np.testing.assert_array_equal(
            np.concatenate((getattr(first.trace, name), getattr(continued.trace, name))),
            getattr(full.trace, name),
        )
    _assert_state_equal(continued.final_state, full.final_state)
    _assert_state_equal(continued.checkpoint.state, full.checkpoint.state)
    assert continued.checkpoint.rng_state == full.checkpoint.rng_state
    assert continued.checkpoint.transitions_completed == full.checkpoint.transitions_completed
    assert continued.checkpoint.kernel_settings == full.checkpoint.kernel_settings
    assert continued.checkpoint.retention == full.checkpoint.retention
    assert continued.checkpoint.schedule_id == full.checkpoint.schedule_id == SCHEDULE_ID


@pytest.mark.parametrize("nucleus_move", ["global", "local"])
def test_split_at_awkward_schedule_phase_matches_uninterrupted_chain(
    nucleus_move: str,
) -> None:
    """Splitting after transition seven should preserve every global phase exactly."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])
    retention = RetentionSettings(warmup_transitions=5, thin=4)

    full = sample(problem, initial, _config(iterations=23, nucleus_move=nucleus_move), retention)
    first = sample(problem, initial, _config(iterations=7, nucleus_move=nucleus_move), retention)
    continued = continue_sample(problem, first.checkpoint, iterations=16)

    _assert_split_matches_full(full, first, continued)
    np.testing.assert_array_equal(full.trace.state_transition, [5, 9, 13, 17, 21])
    np.testing.assert_array_equal(first.trace.state_transition, [5])
    np.testing.assert_array_equal(continued.trace.state_transition, [9, 13, 17, 21])


def test_default_retention_preserves_complete_trace_and_seeded_stream() -> None:
    """Omitted and explicit default retention should save transitions zero through N."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])
    config = _config(iterations=12, nucleus_move="global")

    implicit = sample(problem, initial, config)
    explicit = sample(problem, initial, config, RetentionSettings())

    np.testing.assert_array_equal(implicit.trace.state_transition, np.arange(13))
    for name in (
        "k",
        "nuclei",
        "coefficients",
        "log_target",
        "moves",
        "accepted",
        "log_acceptance_ratio",
    ):
        np.testing.assert_array_equal(getattr(implicit.trace, name), getattr(explicit.trace, name))
    assert implicit.checkpoint.rng_state == explicit.checkpoint.rng_state
    _assert_state_equal(implicit.final_state, explicit.final_state)


def test_segments_may_have_no_retained_states_without_losing_continuation() -> None:
    """Warmup-only segments should return shaped empty arrays and a usable checkpoint."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])
    retention = RetentionSettings(warmup_transitions=10, thin=3)

    first = sample(problem, initial, _config(iterations=3, nucleus_move="global"), retention)
    second = continue_sample(problem, first.checkpoint, iterations=2)

    assert first.trace.k.shape == second.trace.k.shape == (0,)
    assert first.trace.nuclei.shape == second.trace.nuclei.shape == (0, problem.k_max)
    assert first.trace.coefficients.shape == second.trace.coefficients.shape == (0, problem.k_max)
    assert first.trace.log_target.shape == second.trace.log_target.shape == (0,)
    assert first.trace.state_transition is not None
    assert second.trace.state_transition is not None
    assert first.trace.state_transition.size == second.trace.state_transition.size == 0
    assert first.trace.moves.shape == (3,)
    assert second.trace.moves.shape == (2,)
    assert second.checkpoint.transitions_completed == 5
    _assert_state_equal(second.final_state, second.checkpoint.state)


def test_continuation_never_duplicates_a_retained_boundary_state() -> None:
    """A retained incoming checkpoint boundary should not start the resumed trace."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])
    retention = RetentionSettings(warmup_transitions=0, thin=7)

    first = sample(problem, initial, _config(iterations=7, nucleus_move="global"), retention)
    continued = continue_sample(problem, first.checkpoint, iterations=7)

    np.testing.assert_array_equal(first.trace.state_transition, [0, 7])
    np.testing.assert_array_equal(continued.trace.state_transition, [14])


def test_checkpoint_rejects_different_problem_or_schedule() -> None:
    """Continuation should fail closed for a different target object or schedule ID."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])
    result = sample(problem, initial, _config(iterations=7, nucleus_move="global"))

    with pytest.raises(ValueError, match="exact in-memory problem"):
        continue_sample(_problem(), result.checkpoint, iterations=1)
    with pytest.raises(ValueError, match="schedule"):
        continue_sample(
            problem,
            replace(result.checkpoint, schedule_id="future_schedule_v2"),
            iterations=1,
        )


def test_checkpoint_kernel_and_retention_settings_are_immutable() -> None:
    """A caller should not be able to mutate continuation settings in place."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])
    result = sample(problem, initial, _config(iterations=7, nucleus_move="global"))

    with pytest.raises(FrozenInstanceError):
        result.checkpoint.kernel_settings.birth_proposal_sd = 9.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.checkpoint.retention.thin = 9  # type: ignore[misc]
