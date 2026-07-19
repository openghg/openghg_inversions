"""Sampling tests for optional always-active fixed design coefficients."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc import (
    FIXED_BLOCK_SCHEDULE_ID,
    SCHEDULE_ID,
    FixedDesignBlock,
    RetentionSettings,
    SamplerCheckpoint,
    build_state,
    continue_sample,
)
from openghg_inversions.experimental.rjmcmc.core import (
    TransDimensionalProblem,
    TransDimensionalState,
    uniform_log_k_prior,
)
from openghg_inversions.experimental.rjmcmc.sampling import (
    SamplerConfig,
    SamplingResult,
    SamplingTrace,
    _draw_transition,
    sample,
)


def _problem(*, n_fixed: int = 3) -> TransDimensionalProblem:
    """Return a tiny problem with a configurable always-active design block."""
    fixed_design = np.array(
        [
            [1.0, 0.2, -0.1],
            [0.1, 1.0, 0.4],
            [-0.3, 0.4, 1.0],
        ]
    )[:, :n_fixed]
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
        fixed_block=FixedDesignBlock(
            design=fixed_design,
            coefficient_prior_mean=np.ones(n_fixed),
            coefficient_prior_sd=np.full(n_fixed, 0.5),
        ),
    )


def _initial(problem: TransDimensionalProblem) -> TransDimensionalState:
    """Build a finite initial state for a fixed-block problem."""
    return build_state(
        problem,
        [0, 3],
        [0.8, 1.2],
        fixed_coefficients=np.linspace(0.9, 1.1, problem.n_fixed_coefficients),
    )


def _config(*, iterations: int, fixed_scale: float | None = None) -> SamplerConfig:
    """Return common seeded kernel settings for fixed-block tests."""
    return SamplerConfig(
        iterations=iterations,
        coefficient_proposal_sd=0.15,
        birth_proposal_sd=0.25,
        fixed_coefficient_proposal_sd=fixed_scale,
        seed=481,
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


def _assert_results_equal(actual: SamplingResult, expected: SamplingResult) -> None:
    """Assert exact replay equality for traces, state, and continuation data."""
    for name in (
        "k",
        "nuclei",
        "coefficients",
        "fixed_coefficients",
        "log_target",
        "moves",
        "accepted",
        "log_acceptance_ratio",
        "state_transition",
    ):
        np.testing.assert_array_equal(
            getattr(actual.trace, name),
            getattr(expected.trace, name),
        )
    _assert_state_equal(actual.final_state, expected.final_state)
    assert actual.checkpoint.rng_state == expected.checkpoint.rng_state
    assert actual.checkpoint.kernel_settings == expected.checkpoint.kernel_settings
    assert actual.checkpoint.schedule_id == expected.checkpoint.schedule_id


def test_fixed_block_uses_replayable_five_slot_schedule() -> None:
    """A fixed block should add one seeded update without replacing RJ slots."""
    problem = _problem()
    initial = _initial(problem)
    config = _config(iterations=25, fixed_scale=0.08)

    first = sample(problem, initial, config)
    second = sample(problem, initial, config)

    _assert_results_equal(first, second)
    np.testing.assert_array_equal(first.trace.moves[0::5], "coefficient")
    np.testing.assert_array_equal(first.trace.moves[1::5], "fixed_coefficient")
    assert np.all(np.isin(first.trace.moves[2::5], ["birth", "death"]))
    assert np.all(np.isin(first.trace.moves[3::5], ["birth", "death"]))
    np.testing.assert_array_equal(first.trace.moves[4::5], "global_move")
    assert first.trace.fixed_coefficients.shape == (26, 3)
    assert first.checkpoint.schedule_id == FIXED_BLOCK_SCHEDULE_ID


def test_seeded_fixed_draws_can_select_every_fixed_column() -> None:
    """Uniform fixed-column draws should reach every always-active coefficient."""
    problem = _problem()
    initial = _initial(problem)
    config = _config(iterations=1, fixed_scale=0.01)
    rng = np.random.default_rng(178)
    selected: set[int] = set()

    for _ in range(40):
        transition = _draw_transition(problem, initial, config, rng, "fixed_coefficient")
        changed = np.flatnonzero(transition.candidate.fixed_coefficients != initial.fixed_coefficients)
        assert transition.move == "fixed_coefficient"
        assert transition.valid
        assert changed.size == 1
        selected.add(int(changed[0]))

    assert selected == set(range(problem.n_fixed_coefficients))


def test_missing_fixed_scale_reuses_dynamic_coefficient_scale() -> None:
    """An omitted fixed scale should exactly match the explicit dynamic scale."""
    problem = _problem()
    initial = _initial(problem)
    fallback = _config(iterations=1)
    explicit = _config(iterations=1, fixed_scale=fallback.coefficient_proposal_sd)

    fallback_transition = _draw_transition(
        problem,
        initial,
        fallback,
        np.random.default_rng(72),
        "fixed_coefficient",
    )
    explicit_transition = _draw_transition(
        problem,
        initial,
        explicit,
        np.random.default_rng(72),
        "fixed_coefficient",
    )

    assert fallback.fixed_coefficient_proposal_sd is None
    assert fallback_transition.log_q_forward == explicit_transition.log_q_forward
    assert fallback_transition.log_q_reverse == explicit_transition.log_q_reverse
    _assert_state_equal(fallback_transition.candidate, explicit_transition.candidate)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, np.nan])
def test_invalid_fixed_proposal_scale_is_rejected(scale: float) -> None:
    """An explicit fixed-coefficient scale must be finite and positive."""
    with pytest.raises(ValueError, match="fixed_coefficient_proposal_sd"):
        _config(iterations=1, fixed_scale=scale)


def test_fixed_checkpoint_continues_at_awkward_five_slot_phase() -> None:
    """Splitting after slot seven should preserve schedule, RNG, and retention phases."""
    problem = _problem()
    initial = _initial(problem)
    retention = RetentionSettings(warmup_transitions=5, thin=4)

    full = sample(problem, initial, _config(iterations=23), retention)
    first = sample(problem, initial, _config(iterations=7), retention)
    continued = continue_sample(problem, first.checkpoint, iterations=16)

    for name in (
        "k",
        "nuclei",
        "coefficients",
        "fixed_coefficients",
        "log_target",
        "moves",
        "accepted",
        "log_acceptance_ratio",
        "state_transition",
    ):
        np.testing.assert_array_equal(
            np.concatenate((getattr(first.trace, name), getattr(continued.trace, name))),
            getattr(full.trace, name),
        )
    _assert_state_equal(continued.final_state, full.final_state)
    assert continued.checkpoint.rng_state == full.checkpoint.rng_state
    assert continued.checkpoint.kernel_settings == full.checkpoint.kernel_settings
    assert continued.checkpoint.schedule_id == FIXED_BLOCK_SCHEDULE_ID


def test_no_block_preserves_original_seeded_trace_and_zero_width_storage() -> None:
    """Adding fixed-block support must not perturb the established four-slot chain."""
    fixed_problem = _problem()
    problem = TransDimensionalProblem(
        observations=fixed_problem.observations,
        observation_sd=fixed_problem.observation_sd,
        sensitivities=fixed_problem.sensitivities,
        grid_coordinates=fixed_problem.grid_coordinates,
        k_min=fixed_problem.k_min,
        k_max=fixed_problem.k_max,
        log_k_prior=fixed_problem.log_k_prior,
        coefficient_prior_mean=fixed_problem.coefficient_prior_mean,
        coefficient_prior_sd=fixed_problem.coefficient_prior_sd,
    )
    initial = build_state(problem, [0, 3], [0.8, 1.2])
    common = {
        "iterations": 8,
        "coefficient_proposal_sd": 0.15,
        "birth_proposal_sd": 0.25,
        "seed": 481,
    }

    legacy = sample(problem, initial, SamplerConfig(**common))
    irrelevant_fixed_scale = sample(
        problem,
        initial,
        SamplerConfig(**common, fixed_coefficient_proposal_sd=9.0),
    )

    for name in (
        "k",
        "nuclei",
        "coefficients",
        "fixed_coefficients",
        "log_target",
        "moves",
        "accepted",
        "log_acceptance_ratio",
        "state_transition",
    ):
        np.testing.assert_array_equal(
            getattr(legacy.trace, name),
            getattr(irrelevant_fixed_scale.trace, name),
        )
    _assert_state_equal(legacy.final_state, irrelevant_fixed_scale.final_state)
    assert legacy.checkpoint.rng_state == irrelevant_fixed_scale.checkpoint.rng_state
    np.testing.assert_array_equal(legacy.trace.k, [2, 2, 1, 1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(
        legacy.trace.moves,
        [
            "coefficient",
            "death",
            "birth",
            "global_move",
            "coefficient",
            "birth",
            "birth",
            "global_move",
        ],
    )
    np.testing.assert_array_equal(
        legacy.trace.accepted,
        [False, True, False, True, True, True, False, False],
    )
    assert legacy.trace.fixed_coefficients.shape == (9, 0)
    assert legacy.trace.moves.dtype == np.dtype("<U16")
    assert legacy.checkpoint.schedule_id == SCHEDULE_ID
    assert legacy.checkpoint.rng_state.state == 249982125806589537733108505078275425325


def test_fixed_trace_shapes_survive_empty_retention_and_legacy_construction() -> None:
    """Empty retained segments and old trace call sites should remain well-shaped."""
    problem = _problem(n_fixed=2)
    result = sample(
        problem,
        _initial(problem),
        _config(iterations=3),
        RetentionSettings(warmup_transitions=10),
    )

    assert result.trace.fixed_coefficients.shape == (0, 2)
    assert result.final_state.fixed_coefficients.shape == (2,)
    assert isinstance(result.checkpoint, SamplerCheckpoint)

    legacy = SamplingTrace(
        k=np.array([1, 1], dtype=np.int64),
        nuclei=np.array([[0], [0]], dtype=np.int64),
        coefficients=np.array([[1.0], [1.1]]),
        log_target=np.zeros(2),
        moves=np.array(["coefficient"]),
        accepted=np.array([True]),
        log_acceptance_ratio=np.zeros(1),
    )
    assert legacy.fixed_coefficients.shape == (2, 0)
