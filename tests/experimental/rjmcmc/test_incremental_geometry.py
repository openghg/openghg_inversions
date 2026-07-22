"""Regression tests for exact incremental Voronoi structural updates."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import ArrayLike

import openghg_inversions.experimental.rjmcmc.core as core_module
import openghg_inversions.experimental.rjmcmc.proposals as proposal_module
from openghg_inversions.experimental.rjmcmc.checkpoint_io import load_checkpoint, save_checkpoint
from openghg_inversions.experimental.rjmcmc.core import (
    Backend,
    FixedDesignBlock,
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
    uniform_log_k_prior,
    update_structural_state,
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


def _problem(*, duplicate_coordinates: bool = False) -> TransDimensionalProblem:
    """Return a deterministic problem with ties and a nonempty fixed block."""
    coordinates = np.arange(9, dtype=np.float64)
    if duplicate_coordinates:
        coordinates[4] = coordinates[3]
    sensitivities = np.array(
        [
            [1.0, -2.0, 4.0, 8.0, -16.0, 32.0, 64.0, -128.0, 256.0],
            [0.25, 0.5, -0.75, 1.25, 2.5, -3.5, 5.0, 7.5, -11.0],
            [3.0, -1.0, 2.0, -4.0, 6.0, -8.0, 10.0, -12.0, 14.0],
            [-0.2, 0.3, 0.7, -1.1, 1.3, 1.7, -1.9, 2.3, 2.9],
        ],
        dtype=np.float64,
    )
    return TransDimensionalProblem(
        observations=np.array([2.0, -1.0, 0.5, 3.0]),
        observation_sd=np.array([0.7, 1.1, 0.9, 1.3]),
        sensitivities=sensitivities,
        grid_coordinates=coordinates[:, np.newaxis],
        k_min=1,
        k_max=6,
        log_k_prior=uniform_log_k_prior(1, 6),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.8,
        fixed_offset=np.array([0.1, -0.2, 0.3, -0.4]),
        fixed_block=FixedDesignBlock(
            design=np.array(
                [
                    [1.0, 0.5],
                    [0.0, -1.0],
                    [2.0, 0.25],
                    [-0.5, 1.5],
                ]
            ),
            coefficient_prior_mean=np.array([1.0, 1.2]),
            coefficient_prior_sd=np.array([0.4, 0.6]),
        ),
    )


def _source(problem: TransDimensionalProblem, backend: Backend) -> TransDimensionalState:
    """Build the common three-region source with intentionally unsorted input."""
    return build_state(
        problem,
        [8, 0, 4],
        [1.4, 0.7, 1.1],
        fixed_coefficients=[0.9, 1.3],
        backend=backend,
    )


def _two_dimensional_problem() -> TransDimensionalProblem:
    """Return a compact square-grid problem for genuinely 2-D geometry edits."""
    coordinates = np.array(
        [(x, y) for y in range(3) for x in range(3)],
        dtype=np.float64,
    )
    sensitivities = np.arange(1, 37, dtype=np.float64).reshape(4, 9)
    sensitivities[1::2] *= -0.25
    return TransDimensionalProblem(
        observations=np.array([1.0, -2.0, 0.5, 3.0]),
        observation_sd=np.array([0.8, 1.1, 0.9, 1.2]),
        sensitivities=sensitivities,
        grid_coordinates=coordinates,
        k_min=1,
        k_max=5,
        log_k_prior=uniform_log_k_prior(1, 5),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.7,
    )


def _assert_state_exact(
    actual: TransDimensionalState,
    expected: TransDimensionalState,
) -> None:
    """Require bitwise equality for every cached state field."""
    for state_field in fields(TransDimensionalState):
        actual_value = getattr(actual, state_field.name)
        expected_value = getattr(expected, state_field.name)
        if isinstance(actual_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value
    assert actual.log_target == expected.log_target


def _assert_result_exact(actual: SamplingResult, expected: SamplingResult) -> None:
    """Require exact trace, final-state, continuation, and RNG parity."""
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


def _full_structural_rebuild(
    problem: TransDimensionalProblem,
    state: TransDimensionalState,
    active_nuclei: ArrayLike,
    active_coefficients: ArrayLike,
    *,
    backend: Backend = "numpy",
) -> TransDimensionalState:
    """Adapt complete structural candidate values to the full state builder."""
    return build_state(
        problem,
        active_nuclei,
        active_coefficients,
        fixed_coefficients=state.fixed_coefficients,
        backend=backend,
    )


@pytest.mark.parametrize("source_backend", ["numpy", "numba"])
@pytest.mark.parametrize("candidate_backend", ["numpy", "numba"])
@pytest.mark.parametrize(
    ("nuclei", "coefficients"),
    [
        ([8, 2, 0, 4], [1.4, 0.95, 0.7, 1.1]),
        ([8, 0], [1.4, 0.7]),
        ([8, 0, 6], [1.4, 0.7, 1.25]),
    ],
    ids=("insertion", "deletion", "move"),
)
def test_incremental_structural_edits_match_full_rebuild_bitwise(
    source_backend: Backend,
    candidate_backend: Backend,
    nuclei: list[int],
    coefficients: list[float],
) -> None:
    """Insertion, deletion, and movement must exactly match a full rebuild."""
    problem = _problem()
    source = _source(problem, source_backend)

    candidate = update_structural_state(
        problem,
        source,
        nuclei,
        coefficients,
        backend=candidate_backend,
    )
    rebuilt = build_state(
        problem,
        nuclei,
        coefficients,
        fixed_coefficients=source.fixed_coefficients,
        backend=candidate_backend,
    )

    _assert_state_exact(candidate, rebuilt)
    np.testing.assert_array_equal(candidate.active_nuclei, np.sort(nuclei))


def test_supported_structural_edits_never_call_full_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supported insertion, deletion, and move edits must stay on the fast path."""
    problem = _problem()
    source = _source(problem, "numpy")
    edits = (
        ([0, 2, 4, 8], [0.7, 0.95, 1.1, 1.4]),
        ([0, 8], [0.7, 1.4]),
        ([0, 6, 8], [0.7, 1.25, 1.4]),
    )
    expected = [
        build_state(
            problem,
            nuclei,
            coefficients,
            fixed_coefficients=source.fixed_coefficients,
        )
        for nuclei, coefficients in edits
    ]

    def fail_full_rebuild(*args: object, **kwargs: object) -> TransDimensionalState:
        """Fail if a supported one-nucleus edit reaches the fallback builder."""
        raise AssertionError("supported structural edit called build_state")

    monkeypatch.setattr(core_module, "build_state", fail_full_rebuild)
    for (nuclei, coefficients), rebuilt in zip(edits, expected, strict=True):
        candidate = update_structural_state(problem, source, nuclei, coefficients)
        _assert_state_exact(candidate, rebuilt)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize(
    ("source_nuclei", "source_coefficients", "final_nuclei", "final_coefficients"),
    [
        ([0, 4, 8], [0.7, 1.1, 1.4], [4, 8], [1.1, 1.4]),
        ([0, 4, 8], [0.7, 1.1, 1.4], [0, 4], [0.7, 1.1]),
        ([0], [0.7], [8], [0.7]),
    ],
    ids=("delete-first", "delete-last", "move-at-k-one"),
)
def test_two_dimensional_edge_edits_match_full_rebuild_exactly(
    backend: Backend,
    source_nuclei: list[int],
    source_coefficients: list[float],
    final_nuclei: list[int],
    final_coefficients: list[float],
) -> None:
    """Two-dimensional boundary deletions and a k=1 move must remain exact."""
    problem = _two_dimensional_problem()
    source = build_state(problem, source_nuclei, source_coefficients, backend=backend)

    candidate = update_structural_state(
        problem,
        source,
        final_nuclei,
        final_coefficients,
        backend=backend,
    )
    rebuilt = build_state(
        problem,
        final_nuclei,
        final_coefficients,
        backend=backend,
    )

    _assert_state_exact(candidate, rebuilt)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_incremental_insertion_resolves_ties_by_canonical_nucleus(
    backend: Backend,
) -> None:
    """Exact ties must choose the lower canonical position on either side."""
    problem = _problem()
    source = build_state(
        problem,
        [2, 6],
        [0.8, 1.2],
        fixed_coefficients=[0.9, 1.3],
        backend=backend,
    )

    candidate = update_structural_state(
        problem,
        source,
        [6, 4, 2],
        [1.2, 1.0, 0.8],
        backend=backend,
    )

    np.testing.assert_array_equal(candidate.active_nuclei, [2, 4, 6])
    assert candidate.labels[3] == 0  # Nucleus 2 wins its tie with added nucleus 4.
    assert candidate.labels[5] == 1  # Added nucleus 4 wins its tie with nucleus 6.
    rebuilt = build_state(
        problem,
        [2, 4, 6],
        [0.8, 1.0, 1.2],
        fixed_coefficients=source.fixed_coefficients,
        backend=backend,
    )
    _assert_state_exact(candidate, rebuilt)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_duplicate_nucleus_coordinates_allow_an_exact_empty_region(
    backend: Backend,
) -> None:
    """Distinct nuclei at duplicate coordinates should preserve an empty region."""
    problem = _problem(duplicate_coordinates=True)
    source = build_state(
        problem,
        [0, 3],
        [0.7, 1.1],
        fixed_coefficients=[0.9, 1.3],
        backend=backend,
    )

    candidate = update_structural_state(
        problem,
        source,
        [4, 0, 3],
        [1.6, 0.7, 1.1],
        backend=backend,
    )
    rebuilt = build_state(
        problem,
        [4, 0, 3],
        [1.6, 0.7, 1.1],
        fixed_coefficients=source.fixed_coefficients,
        backend=backend,
    )

    _assert_state_exact(candidate, rebuilt)
    assert not np.any(candidate.labels == 2)
    np.testing.assert_array_equal(candidate.design[:, 2], 0.0)


def test_incremental_update_preserves_source_fixed_cache_and_owns_inputs() -> None:
    """Structural updates must not mutate source/input arrays or expose writable caches."""
    problem = _problem()
    source = _source(problem, "numpy")
    source_snapshot = {
        state_field.name: np.array(getattr(source, state_field.name), copy=True)
        for state_field in fields(TransDimensionalState)
        if isinstance(getattr(source, state_field.name), np.ndarray)
    }
    nuclei = np.array([8, 6, 0], dtype=np.int64)
    coefficients = np.array([1.4, 1.25, 0.7])

    candidate = update_structural_state(
        problem,
        source,
        nuclei,
        coefficients,
        backend="numba",
    )
    nuclei[:] = 1
    coefficients[:] = -100.0

    np.testing.assert_array_equal(candidate.active_nuclei, [0, 6, 8])
    np.testing.assert_array_equal(candidate.active_coefficients, [0.7, 1.25, 1.4])
    assert candidate.fixed_coefficients is source.fixed_coefficients
    assert candidate.fixed_prediction is source.fixed_prediction
    for name, expected in source_snapshot.items():
        np.testing.assert_array_equal(getattr(source, name), expected)
    for state_field in fields(TransDimensionalState):
        value = getattr(candidate, state_field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable


def test_unsupported_multi_edit_uses_full_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """More than one removal/addition should take the safe full-rebuild fallback."""
    problem = _problem()
    source = _source(problem, "numpy")
    original_build_state = core_module.build_state
    calls: list[tuple[np.ndarray, np.ndarray, Backend]] = []

    def recording_build_state(
        problem_arg: TransDimensionalProblem,
        nuclei_arg: ArrayLike,
        coefficients_arg: ArrayLike,
        *,
        fixed_coefficients: ArrayLike | None = None,
        backend: Backend = "numpy",
    ) -> TransDimensionalState:
        """Record fallback inputs before delegating to the real builder."""
        calls.append(
            (
                np.array(nuclei_arg, dtype=np.int64, copy=True),
                np.array(coefficients_arg, dtype=np.float64, copy=True),
                backend,
            )
        )
        return original_build_state(
            problem_arg,
            nuclei_arg,
            coefficients_arg,
            fixed_coefficients=fixed_coefficients,
            backend=backend,
        )

    monkeypatch.setattr(core_module, "build_state", recording_build_state)
    candidate = update_structural_state(
        problem,
        source,
        [8, 1, 6],
        [1.4, 0.75, 1.25],
        backend="numba",
    )
    rebuilt = original_build_state(
        problem,
        [8, 1, 6],
        [1.4, 0.75, 1.25],
        fixed_coefficients=source.fixed_coefficients,
        backend="numba",
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], [1, 6, 8])
    np.testing.assert_array_equal(calls[0][1], [0.75, 1.25, 1.4])
    assert calls[0][2] == "numba"
    _assert_state_exact(candidate, rebuilt)


def test_incremental_design_reuses_unchanged_memberships_by_value() -> None:
    """Unchanged columns retain source values while changed columns rebuild exactly."""
    problem = _problem()
    source = _source(problem, "numpy")
    candidate = update_structural_state(
        problem,
        source,
        [8, 6, 0, 4],
        [1.4, 1.25, 0.7, 1.1],
        backend="numpy",
    )
    rebuilt = build_state(
        problem,
        [8, 6, 0, 4],
        [1.4, 1.25, 0.7, 1.1],
        fixed_coefficients=source.fixed_coefficients,
        backend="numpy",
    )

    old_members = {
        int(nucleus): np.flatnonzero(source.labels == position)
        for position, nucleus in enumerate(source.active_nuclei)
    }
    new_members = {
        int(nucleus): np.flatnonzero(candidate.labels == position)
        for position, nucleus in enumerate(candidate.active_nuclei)
    }
    unchanged = [
        nucleus
        for nucleus in old_members.keys() & new_members.keys()
        if np.array_equal(old_members[nucleus], new_members[nucleus])
    ]
    affected = [
        nucleus
        for nucleus in new_members
        if nucleus not in old_members or not np.array_equal(old_members[nucleus], new_members[nucleus])
    ]

    assert unchanged
    assert affected
    assert not np.shares_memory(candidate.design, source.design)
    for nucleus in unchanged:
        old_position = int(np.flatnonzero(source.active_nuclei == nucleus)[0])
        new_position = int(np.flatnonzero(candidate.active_nuclei == nucleus)[0])
        np.testing.assert_array_equal(
            candidate.design[:, new_position],
            source.design[:, old_position],
        )
    for nucleus in affected:
        new_position = int(np.flatnonzero(candidate.active_nuclei == nucleus)[0])
        np.testing.assert_array_equal(
            candidate.design[:, new_position],
            rebuilt.design[:, new_position],
        )


def test_seeded_sampler_matches_full_structural_rebuild_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incremental geometry must preserve mixed dimension and nucleus replay."""
    problem = _problem()
    initial = _source(problem, "numba")
    config = SamplerConfig(
        iterations=200,
        coefficient_proposal_sd=0.18,
        birth_proposal_sd=0.25,
        fixed_coefficient_proposal_sd=0.16,
        seed=74192,
        backend="numba",
        nucleus_move="local",
        local_move_scale=2.0,
    )
    accelerated = sample(problem, initial, config)

    monkeypatch.setattr(
        proposal_module,
        "update_structural_state",
        _full_structural_rebuild,
    )
    rebuilt = sample(problem, initial, config)

    assert {"birth", "death", "local_move"} <= set(accelerated.trace.moves)
    _assert_result_exact(accelerated, rebuilt)


def test_global_move_proposal_matches_full_structural_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A global nucleus move must preserve its candidate and proposal accounting."""
    problem = _problem()
    source = _source(problem, "numba")
    accelerated = proposal_module.propose_global_move(
        problem,
        source,
        move_position=1,
        new_nucleus=2,
        backend="numba",
    )

    monkeypatch.setattr(
        proposal_module,
        "update_structural_state",
        _full_structural_rebuild,
    )
    rebuilt = proposal_module.propose_global_move(
        problem,
        source,
        move_position=1,
        new_nucleus=2,
        backend="numba",
    )

    _assert_state_exact(accelerated.candidate, rebuilt.candidate)
    for name in (
        "log_target_delta",
        "log_q_forward",
        "log_q_reverse",
        "log_jacobian",
        "move",
        "valid",
        "reason",
        "log_acceptance_ratio",
    ):
        assert getattr(accelerated, name) == getattr(rebuilt, name)


def test_accepted_incremental_structural_state_round_trips_checkpoint(
    tmp_path: Path,
) -> None:
    """An accepted structural fast-path state must persist and continue exactly."""
    problem = _problem()
    initial = _source(problem, "numba")
    transition = proposal_module.propose_birth(
        problem,
        initial,
        new_nucleus=2,
        proposed_coefficient=0.95,
        proposal_stdev=0.25,
        backend="numba",
    )
    accepted = accept_or_reject(initial, transition, log_uniform=-np.inf)
    assert transition.valid
    assert accepted is transition.candidate

    kernel = KernelSettings(
        coefficient_proposal_sd=0.18,
        birth_proposal_sd=0.25,
        fixed_coefficient_proposal_sd=0.16,
        backend="numba",
        nucleus_move="local",
        local_move_scale=2.0,
    )
    checkpoint = SamplerCheckpoint(
        problem=problem,
        state=accepted,
        rng_state=PCG64State.from_generator(np.random.default_rng(1837)),
        transitions_completed=1,
        kernel_settings=kernel,
        retention=RetentionSettings(warmup_transitions=2, thin=2),
        schedule_id=FIXED_BLOCK_SCHEDULE_ID,
    )
    expected = continue_sample(problem, checkpoint, iterations=17)
    path = tmp_path / "accepted-incremental-structural-state.npz"
    manifest = {"kind": "incremental-structural-state", "version": 1}
    save_checkpoint(path, checkpoint, run_manifest=manifest)
    loaded = load_checkpoint(path, problem, expected_run_manifest=manifest)
    actual = continue_sample(problem, loaded, iterations=17)

    _assert_state_exact(loaded.state, accepted)
    _assert_result_exact(actual, expected)
