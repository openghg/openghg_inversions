"""Parity tests for cache-preserving coefficient-only state updates."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.core import (
    Backend,
    FixedDesignBlock,
    TransDimensionalProblem,
    TransDimensionalState,
    aggregate_design_numba,
    aggregate_design_numpy,
    build_state,
    uniform_log_k_prior,
    update_dynamic_coefficient_state,
    update_fixed_coefficient_state,
)
from openghg_inversions.experimental.rjmcmc.checkpoint_io import (
    load_checkpoint,
    save_checkpoint,
)
from openghg_inversions.experimental.rjmcmc.proposals import accept_or_reject
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings
from openghg_inversions.experimental.rjmcmc.sampling import (
    FIXED_BLOCK_SCHEDULE_ID,
    KernelSettings,
    PCG64State,
    SamplerCheckpoint,
    SamplerConfig,
    SamplingResult,
    continue_sample,
    sample,
)
import openghg_inversions.experimental.rjmcmc.proposals as proposal_module


@pytest.mark.parametrize(
    ("n_observations", "n_grid", "k", "k_max"),
    [(1, 1, 1, 1), (3, 17, 4, 6), (19, 101, 13, 20)],
)
def test_observation_major_numba_aggregation_is_bitwise_exact(
    n_observations: int,
    n_grid: int,
    k: int,
    k_max: int,
) -> None:
    """Loop interchange must preserve each output's cell accumulation order."""
    rng = np.random.default_rng(8193 + n_grid)
    sensitivities = np.ascontiguousarray(rng.normal(size=(n_observations, n_grid)))
    labels = rng.integers(0, k, size=n_grid, dtype=np.int64)

    expected = aggregate_design_numpy(sensitivities, labels, k, k_max)
    actual = aggregate_design_numba(sensitivities, labels, k, k_max)

    np.testing.assert_array_equal(actual, expected)


def _problem() -> TransDimensionalProblem:
    """Return a heterogeneous fixed-plus-dynamic parity problem."""
    rng = np.random.default_rng(5401)
    return TransDimensionalProblem(
        observations=rng.normal(size=7),
        observation_sd=np.linspace(0.4, 1.0, 7),
        sensitivities=rng.normal(size=(7, 8)),
        grid_coordinates=np.column_stack(
            (np.arange(8, dtype=np.float64), np.square(np.arange(8, dtype=np.float64)))
        ),
        k_min=1,
        k_max=6,
        log_k_prior=uniform_log_k_prior(1, 6),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=2.0,
        fixed_offset=rng.normal(size=7),
        fixed_block=FixedDesignBlock(
            design=rng.normal(size=(7, 3)),
            coefficient_prior_mean=np.array([0.8, 1.1, 1.4]),
            coefficient_prior_sd=np.array([0.3, 0.5, 0.7]),
        ),
    )


def _state(problem: TransDimensionalProblem, *, backend: Backend) -> TransDimensionalState:
    """Build a finite canonical source state."""
    return build_state(
        problem,
        [7, 0, 3, 5],
        [1.4, 0.7, 1.1, 0.9],
        fixed_coefficients=[0.9, 1.3, 1.0],
        backend=backend,
    )


def _full_dynamic_rebuild(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    coefficient_position: int,
    proposed_coefficient: float,
    backend: Backend,
) -> TransDimensionalState:
    """Return the pre-optimization dynamic-coefficient candidate."""
    coefficients = np.array(state.active_coefficients, copy=True)
    coefficients[coefficient_position] = proposed_coefficient
    return build_state(
        problem,
        state.active_nuclei,
        coefficients,
        fixed_coefficients=state.fixed_coefficients,
        backend=backend,
    )


def _full_fixed_rebuild(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    *,
    coefficient_position: int,
    proposed_coefficient: float,
    backend: Backend,
) -> TransDimensionalState:
    """Return the pre-optimization fixed-coefficient candidate."""
    fixed_coefficients = np.array(state.fixed_coefficients, copy=True)
    fixed_coefficients[coefficient_position] = proposed_coefficient
    return build_state(
        problem,
        state.active_nuclei,
        state.active_coefficients,
        fixed_coefficients=fixed_coefficients,
        backend=backend,
    )


def _assert_state_exact(actual: TransDimensionalState, expected: TransDimensionalState) -> None:
    """Require bitwise equality for every state field."""
    for state_field in fields(TransDimensionalState):
        actual_value = getattr(actual, state_field.name)
        expected_value = getattr(expected, state_field.name)
        if isinstance(actual_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value
    assert actual.log_target == expected.log_target


@pytest.mark.parametrize("source_backend", ["numpy", "numba"])
@pytest.mark.parametrize("candidate_backend", ["numpy", "numba"])
def test_dynamic_coefficient_fast_path_exactly_matches_full_rebuild(
    source_backend: Backend,
    candidate_backend: Backend,
) -> None:
    """A dynamic update should reuse geometry without changing any target cache."""
    problem = _problem()
    source = _state(problem, backend=source_backend)
    candidate = update_dynamic_coefficient_state(
        problem,
        source,
        coefficient_position=2,
        proposed_coefficient=1.37,
        backend=candidate_backend,
    )
    rebuilt = _full_dynamic_rebuild(
        problem,
        source,
        coefficient_position=2,
        proposed_coefficient=1.37,
        backend=candidate_backend,
    )

    _assert_state_exact(candidate, rebuilt)
    assert candidate.nuclei is source.nuclei
    assert candidate.labels is source.labels
    assert candidate.design is source.design
    assert candidate.fixed_coefficients is source.fixed_coefficients
    assert candidate.fixed_prediction is source.fixed_prediction
    assert candidate.coefficients is not source.coefficients
    assert candidate.dynamic_prediction is not source.dynamic_prediction
    for state_field in fields(TransDimensionalState):
        value = getattr(candidate, state_field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable


@pytest.mark.parametrize("source_backend", ["numpy", "numba"])
@pytest.mark.parametrize("candidate_backend", ["numpy", "numba"])
def test_fixed_coefficient_fast_path_exactly_matches_full_rebuild(
    source_backend: Backend,
    candidate_backend: Backend,
) -> None:
    """A fixed update should reuse geometry/dynamics without changing the target."""
    problem = _problem()
    source = _state(problem, backend=source_backend)
    candidate = update_fixed_coefficient_state(
        problem,
        source,
        coefficient_position=1,
        proposed_coefficient=0.83,
        backend=candidate_backend,
    )
    rebuilt = _full_fixed_rebuild(
        problem,
        source,
        coefficient_position=1,
        proposed_coefficient=0.83,
        backend=candidate_backend,
    )

    _assert_state_exact(candidate, rebuilt)
    assert candidate.nuclei is source.nuclei
    assert candidate.coefficients is source.coefficients
    assert candidate.labels is source.labels
    assert candidate.design is source.design
    assert candidate.dynamic_prediction is source.dynamic_prediction
    assert candidate.fixed_coefficients is not source.fixed_coefficients
    assert candidate.fixed_prediction is not source.fixed_prediction
    for state_field in fields(TransDimensionalState):
        value = getattr(candidate, state_field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable


def _assert_result_exact(actual: SamplingResult, expected: SamplingResult) -> None:
    """Require exact trace, final-state, RNG, and kernel replay parity."""
    for trace_field in fields(type(actual.trace)):
        np.testing.assert_array_equal(
            getattr(actual.trace, trace_field.name),
            getattr(expected.trace, trace_field.name),
        )
    _assert_state_exact(actual.final_state, expected.final_state)
    assert actual.checkpoint.rng_state == expected.checkpoint.rng_state
    assert actual.checkpoint.transitions_completed == expected.checkpoint.transitions_completed
    assert actual.checkpoint.kernel_settings == expected.checkpoint.kernel_settings
    assert actual.checkpoint.retention == expected.checkpoint.retention
    assert actual.checkpoint.schedule_id == expected.checkpoint.schedule_id


def test_seeded_sampler_is_exactly_unchanged_from_full_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fast path must not alter draws, acceptance, state, or checkpoint RNG."""
    problem = _problem()
    initial = _state(problem, backend="numba")
    config = SamplerConfig(
        iterations=250,
        coefficient_proposal_sd=0.22,
        birth_proposal_sd=0.31,
        fixed_coefficient_proposal_sd=0.18,
        seed=94812,
        backend="numba",
        nucleus_move="local",
        local_move_scale=4.0,
    )
    accelerated = sample(problem, initial, config)

    monkeypatch.setattr(
        proposal_module,
        "update_dynamic_coefficient_state",
        _full_dynamic_rebuild,
    )
    monkeypatch.setattr(
        proposal_module,
        "update_fixed_coefficient_state",
        _full_fixed_rebuild,
    )
    rebuilt = sample(problem, initial, config)

    _assert_result_exact(accelerated, rebuilt)


def test_accepted_fast_state_round_trips_and_continues_exactly(tmp_path: Path) -> None:
    """A durable checkpoint should validate and replay an accepted fast candidate."""
    problem = _problem()
    initial = _state(problem, backend="numba")
    transition = proposal_module.propose_coefficient(
        problem,
        initial,
        coefficient_position=2,
        proposed_coefficient=1.37,
        proposal_stdev=0.22,
        backend="numba",
    )
    accepted = accept_or_reject(initial, transition, log_uniform=-np.inf)
    assert transition.valid
    assert accepted is transition.candidate
    assert accepted.design is initial.design

    kernel = KernelSettings(
        coefficient_proposal_sd=0.22,
        birth_proposal_sd=0.31,
        fixed_coefficient_proposal_sd=0.18,
        backend="numba",
        nucleus_move="local",
        local_move_scale=4.0,
    )
    checkpoint = SamplerCheckpoint(
        problem=problem,
        state=accepted,
        rng_state=PCG64State.from_generator(np.random.default_rng(9182)),
        transitions_completed=1,
        kernel_settings=kernel,
        retention=RetentionSettings(warmup_transitions=3, thin=2),
        schedule_id=FIXED_BLOCK_SCHEDULE_ID,
    )
    expected = continue_sample(problem, checkpoint, iterations=19)
    path = tmp_path / "accepted-fast-state.npz"
    manifest = {"kind": "accepted-coefficient-fast-path", "version": 1}
    save_checkpoint(path, checkpoint, run_manifest=manifest)
    loaded = load_checkpoint(path, problem, expected_run_manifest=manifest)
    actual = continue_sample(problem, loaded, iterations=19)

    _assert_state_exact(loaded.state, accepted)
    _assert_result_exact(actual, expected)
