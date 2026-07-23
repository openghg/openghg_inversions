"""Focused regression tests for opt-in structural mixing diagnostics."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.experimental.rjmcmc.core import (
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
    uniform_log_k_prior,
)
from openghg_inversions.experimental.rjmcmc.mixing_diagnostics import (
    StructuralDiagnostics,
    StructuralDiagnosticsProvenance,
    _StructuralDiagnosticsBuffer,
    concatenate_structural_diagnostics,
    derive_nucleus_residence_intervals,
    derive_region_lineage_intervals,
    derive_structural_diagnostics,
)
from openghg_inversions.experimental.rjmcmc.proposals import (
    propose_death,
    propose_global_move,
)
from openghg_inversions.experimental.rjmcmc.sampling import (
    FIXED_BLOCK_SCHEDULE_ID,
    LUNT_OPPORTUNITY_MATCHED_FIXED_BLOCK_SCHEDULE_ID,
    LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_ID,
    LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_ID,
    SCHEDULE_ID,
    SamplerConfig,
    SamplingResult,
    _structural_event_count,
    continue_sample,
    sample,
)
from openghg_inversions.experimental.rjmcmc.xarray_output import (
    structural_diagnostics_from_dataset,
    structural_diagnostics_to_dataset,
)

_PROVENANCE = StructuralDiagnosticsProvenance(
    chain_id="tiny-chain-0",
    problem_fingerprint="0" * 64,
)


def _problem(*, k_min: int = 1, k_max: int = 3) -> TransDimensionalProblem:
    """Return a tiny problem whose structural metrics are easy to verify."""
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
        k_min=k_min,
        k_max=k_max,
        log_k_prior=uniform_log_k_prior(k_min, k_max),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.4,
    )


def _config(
    *,
    iterations: int,
    collect: bool,
    backend: str = "numpy",
) -> SamplerConfig:
    """Return common seeded sampler settings with optional diagnostics."""
    return SamplerConfig(
        iterations=iterations,
        coefficient_proposal_sd=0.15,
        birth_proposal_sd=0.25,
        seed=481,
        backend=backend,  # type: ignore[arg-type]
        collect_structural_diagnostics=collect,
        structural_diagnostics_provenance=_PROVENANCE if collect else None,
    )


def _assert_state_equal(
    actual: TransDimensionalState,
    expected: TransDimensionalState,
) -> None:
    """Assert exact equality of every cached state component."""
    for state_field in fields(TransDimensionalState):
        actual_value = getattr(actual, state_field.name)
        expected_value = getattr(expected, state_field.name)
        if isinstance(actual_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def _assert_kernel_results_equal(
    actual: SamplingResult,
    expected: SamplingResult,
) -> None:
    """Assert that output-only diagnostics did not alter a sampled chain."""
    for trace_field in fields(actual.trace):
        np.testing.assert_array_equal(
            getattr(actual.trace, trace_field.name),
            getattr(expected.trace, trace_field.name),
        )
    _assert_state_equal(actual.final_state, expected.final_state)
    _assert_state_equal(actual.checkpoint.state, expected.checkpoint.state)
    assert actual.checkpoint.rng_state == expected.checkpoint.rng_state
    assert actual.checkpoint.transitions_completed == expected.checkpoint.transitions_completed
    assert actual.checkpoint.kernel_settings == expected.checkpoint.kernel_settings
    assert actual.checkpoint.retention == expected.checkpoint.retention
    assert actual.checkpoint.schedule_id == expected.checkpoint.schedule_id


def _assert_diagnostics_equal(
    actual: StructuralDiagnostics,
    expected: StructuralDiagnostics,
) -> None:
    """Assert exact equality of diagnostic arrays and problem metadata."""
    for diagnostic_field in fields(StructuralDiagnostics):
        actual_value = getattr(actual, diagnostic_field.name)
        expected_value = getattr(expected, diagnostic_field.name)
        if isinstance(actual_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


@pytest.mark.parametrize(
    ("schedule_id", "cycle_length", "structural_phases"),
    [
        (SCHEDULE_ID, 4, (1, 2, 3)),
        (FIXED_BLOCK_SCHEDULE_ID, 5, (2, 3, 4)),
        (
            LUNT_OPPORTUNITY_MATCHED_FIXED_BLOCK_SCHEDULE_ID,
            14,
            (0, 1, 2),
        ),
        (LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_ID, 16, (0, 1, 2)),
        (
            LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_ID,
            17,
            (0, 1, 2),
        ),
    ],
)
@pytest.mark.parametrize(
    ("transitions_completed", "iterations"),
    [(0, 1), (1, 7), (13, 23), (31, 41)],
)
def test_structural_event_preallocation_matches_each_schedule(
    schedule_id: str,
    cycle_length: int,
    structural_phases: tuple[int, ...],
    transitions_completed: int,
    iterations: int,
) -> None:
    """Exact preallocation should handle every schedule and mid-cycle start."""
    expected = sum(
        (transition % cycle_length) in structural_phases
        for transition in range(
            transitions_completed,
            transitions_completed + iterations,
        )
    )

    assert (
        _structural_event_count(
            schedule_id=schedule_id,
            transitions_completed=transitions_completed,
            iterations=iterations,
        )
        == expected
    )


def _assert_diagnostics_backend_close(
    actual: StructuralDiagnostics,
    expected: StructuralDiagnostics,
) -> None:
    """Assert numerical backend parity up to floating-point roundoff."""
    for diagnostic_field in fields(StructuralDiagnostics):
        actual_value = getattr(actual, diagnostic_field.name)
        expected_value = getattr(expected, diagnostic_field.name)
        if isinstance(actual_value, np.ndarray) and np.issubdtype(actual_value.dtype, np.floating):
            np.testing.assert_allclose(
                actual_value,
                expected_value,
                rtol=1e-14,
                atol=1e-14,
            )
        elif isinstance(actual_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def test_collection_is_output_only_and_records_all_structural_outcomes() -> None:
    """Diagnostics should preserve the RNG stream and distinguish all outcomes."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])

    disabled = sample(problem, initial, _config(iterations=24, collect=False))
    enabled = sample(problem, initial, _config(iterations=24, collect=True))

    assert disabled.structural_diagnostics is None
    diagnostics = enabled.structural_diagnostics
    assert diagnostics is not None
    _assert_kernel_results_equal(enabled, disabled)
    np.testing.assert_array_equal(
        diagnostics.transition,
        np.flatnonzero(np.isin(enabled.trace.moves, diagnostics.move)) + 1,
    )
    np.testing.assert_array_equal(
        diagnostics.move,
        enabled.trace.moves[diagnostics.transition - 1],
    )
    assert np.any(diagnostics.valid & diagnostics.accepted)
    assert np.any(diagnostics.valid & ~diagnostics.accepted)
    assert np.any(~diagnostics.valid)
    assert np.all(diagnostics.invalid_reason[diagnostics.valid] == "")
    assert np.all(diagnostics.invalid_reason[~diagnostics.valid] != "")
    np.testing.assert_array_equal(
        diagnostics.result_k,
        np.where(
            diagnostics.accepted,
            diagnostics.candidate_k,
            diagnostics.source_k,
        ),
    )
    row_bytes = sum(
        getattr(diagnostics, field.name).nbytes
        for field in fields(StructuralDiagnostics)
        if isinstance(getattr(diagnostics, field.name), np.ndarray)
        and getattr(diagnostics, field.name).shape == (diagnostics.size,)
    )
    assert row_bytes / diagnostics.size < 300


def test_large_finite_predictions_do_not_break_output_only_collection() -> None:
    """Overflow-resistant norms should preserve an extreme finite trajectory."""
    template = _problem()
    problem = TransDimensionalProblem(
        observations=np.zeros(template.n_observations),
        observation_sd=np.full(template.n_observations, 1.0e200),
        sensitivities=template.sensitivities * 1.0e200,
        grid_coordinates=template.grid_coordinates,
        k_min=template.k_min,
        k_max=template.k_max,
        log_k_prior=template.log_k_prior,
        coefficient_prior_mean=template.coefficient_prior_mean,
        coefficient_prior_sd=template.coefficient_prior_sd,
    )
    initial = build_state(problem, [0, 3], [0.8, 1.2])

    disabled = sample(problem, initial, _config(iterations=12, collect=False))
    enabled = sample(problem, initial, _config(iterations=12, collect=True))

    _assert_kernel_results_equal(enabled, disabled)
    diagnostics = enabled.structural_diagnostics
    assert diagnostics is not None
    assert np.all(~np.isnan(diagnostics.prediction_change_l2))
    assert np.any(diagnostics.prediction_change_l2 > 1.0e199)


def test_boundary_attempts_remain_typed_invalid_self_transitions() -> None:
    """Unavailable dimension directions should have explicit invalid diagnostics."""
    problem = _problem(k_min=2, k_max=2)
    initial = build_state(problem, [0, 3], [0.8, 1.2])

    result = sample(problem, initial, _config(iterations=4, collect=True))

    diagnostics = result.structural_diagnostics
    assert diagnostics is not None
    dimension = np.isin(diagnostics.move, ["birth", "death"])
    assert dimension.sum() == 2
    assert np.all(~diagnostics.valid[dimension])
    assert np.all(~diagnostics.accepted[dimension])
    assert np.all(diagnostics.source_k[dimension] == 2)
    assert np.all(diagnostics.candidate_k[dimension] == 2)
    assert np.all(diagnostics.result_k[dimension] == 2)
    assert np.all(diagnostics.log_acceptance_ratio[dimension] == -np.inf)
    assert np.all(diagnostics.owner_changed_cell_count[dimension] == 0)
    assert np.all(diagnostics.owner_changed_cell_fraction[dimension] == 0.0)
    assert np.all(diagnostics.source_nucleus[dimension] == -1)
    assert np.all(diagnostics.candidate_nucleus[dimension] == -1)


def test_cached_target_components_reconstruct_target_and_mh_deltas() -> None:
    """Cached component deltas should sum to the target and MH log ratio."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])

    result = sample(problem, initial, _config(iterations=24, collect=True))

    diagnostics = result.structural_diagnostics
    assert diagnostics is not None
    component_sum = (
        diagnostics.delta_log_likelihood
        + diagnostics.delta_log_coefficient_prior
        + diagnostics.delta_log_fixed_coefficient_prior
        + diagnostics.delta_log_k_prior
        + diagnostics.delta_log_nucleus_prior
        + diagnostics.delta_log_error_model_prior
        + diagnostics.delta_log_coefficient_hyperprior
    )
    np.testing.assert_allclose(
        component_sum,
        diagnostics.delta_log_target,
        rtol=0.0,
        atol=2e-14,
    )
    valid = diagnostics.valid
    np.testing.assert_allclose(
        diagnostics.delta_log_target[valid]
        + diagnostics.log_q_reverse[valid]
        - diagnostics.log_q_forward[valid]
        + diagnostics.log_jacobian[valid],
        diagnostics.log_acceptance_ratio[valid],
        rtol=0.0,
        atol=2e-14,
    )


def test_manual_move_uses_owner_identity_and_cached_prediction_metrics() -> None:
    """Canonical label reordering must not corrupt ownership or norm metrics."""
    problem = _problem()
    source = build_state(problem, [2, 3], [0.8, 1.2])
    transition = propose_global_move(
        problem,
        source,
        move_position=1,
        new_nucleus=0,
    )
    assert transition.valid
    candidate = transition.candidate
    source_owner = source.active_nuclei[source.labels]
    candidate_owner = candidate.active_nuclei[candidate.labels]
    changed = source_owner != candidate_owner
    assert np.count_nonzero(source.labels != candidate.labels) != np.count_nonzero(changed)

    buffer = _StructuralDiagnosticsBuffer(
        problem,
        source,
        capacity=1,
        provenance=_PROVENANCE,
    )
    buffer.append(
        transition_number=4,
        source=source,
        transition=transition,
        result=candidate,
        accepted=True,
    )
    diagnostics = buffer.finalize(candidate)

    assert diagnostics.source_nucleus[0] == 3
    assert diagnostics.candidate_nucleus[0] == 0
    assert diagnostics.owner_changed_cell_count[0] == np.count_nonzero(changed) == 3
    assert diagnostics.owner_changed_cell_fraction[0] == pytest.approx(0.75)
    affected = np.union1d(source_owner[changed], candidate_owner[changed])
    expected_affected = np.count_nonzero(np.isin(candidate.active_nuclei, affected))
    assert diagnostics.affected_candidate_design_column_count[0] == expected_affected == 2
    prediction_change = candidate.prediction - source.prediction
    assert diagnostics.prediction_change_l2[0] == pytest.approx(np.linalg.norm(prediction_change))
    assert diagnostics.observation_error_standardized_prediction_change_l2[0] == pytest.approx(
        np.linalg.norm(prediction_change / problem.observation_sd)
    )
    candidate_position = int(np.searchsorted(candidate.active_nuclei, 0))
    assert diagnostics.event_region_observation_error_standardized_design_l2[0] == pytest.approx(
        np.linalg.norm(candidate.design[:, candidate_position] / problem.observation_sd)
    )
    assert diagnostics.coefficient_contrast[0] == pytest.approx(np.log(1.2 / 0.8))


def test_segments_concatenate_to_uninterrupted_global_diagnostics(
    tmp_path: Path,
) -> None:
    """Restarted structural rows should reproduce an uninterrupted global stream."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])

    full = sample(problem, initial, _config(iterations=23, collect=True))
    first = sample(problem, initial, _config(iterations=7, collect=True))
    second = continue_sample(
        problem,
        first.checkpoint,
        iterations=16,
        collect_structural_diagnostics=True,
        structural_diagnostics_provenance=_PROVENANCE,
    )

    assert full.structural_diagnostics is not None
    assert first.structural_diagnostics is not None
    assert second.structural_diagnostics is not None
    restored_segments = []
    for segment, diagnostics in enumerate([first.structural_diagnostics, second.structural_diagnostics]):
        path = tmp_path / f"diagnostics_{segment}.nc"
        structural_diagnostics_to_dataset(diagnostics).to_netcdf(path)
        with xr.open_dataset(path) as stored:
            restored_segments.append(structural_diagnostics_from_dataset(stored.load()))
    concatenated = concatenate_structural_diagnostics(restored_segments)
    _assert_diagnostics_equal(concatenated, full.structural_diagnostics)
    _assert_state_equal(second.final_state, full.final_state)
    _assert_state_equal(second.checkpoint.state, full.checkpoint.state)
    assert second.checkpoint.rng_state == full.checkpoint.rng_state
    assert second.checkpoint.transitions_completed == full.checkpoint.transitions_completed
    assert second.checkpoint.kernel_settings == full.checkpoint.kernel_settings
    assert second.checkpoint.retention == full.checkpoint.retention
    assert second.checkpoint.schedule_id == full.checkpoint.schedule_id
    derived = derive_structural_diagnostics(concatenated)
    assert int(derived.k_step.sum()) == second.final_state.k - initial.k
    discontinuous = replace(
        restored_segments[1],
        segment_transition_start=restored_segments[1].segment_transition_start - 1,
    )
    with pytest.raises(ValueError, match="transition bounds"):
        concatenate_structural_diagnostics([restored_segments[0], discontinuous])
    different_chain = replace(
        restored_segments[1],
        provenance=StructuralDiagnosticsProvenance(
            chain_id="tiny-chain-1",
            problem_fingerprint=_PROVENANCE.problem_fingerprint,
        ),
    )
    with pytest.raises(ValueError, match="metadata"):
        concatenate_structural_diagnostics([restored_segments[0], different_chain])


def test_opt_in_collection_requires_explicit_chain_and_problem_provenance() -> None:
    """Output-only collection must not create artifacts with ambiguous identity."""
    with pytest.raises(ValueError, match="SHA-256"):
        StructuralDiagnosticsProvenance(
            chain_id="tiny-chain-0",
            problem_fingerprint="not-a-durable-fingerprint",
        )
    with pytest.raises(TypeError, match="structural_diagnostics_provenance"):
        SamplerConfig(
            iterations=4,
            coefficient_proposal_sd=0.15,
            birth_proposal_sd=0.25,
            collect_structural_diagnostics=True,
        )
    with pytest.raises(ValueError, match="only valid"):
        replace(
            _config(iterations=4, collect=False),
            structural_diagnostics_provenance=_PROVENANCE,
        )


def _synthetic_derived_diagnostics() -> StructuralDiagnostics:
    """Return a hand-specified stream for reversal, lineage, and residence tests."""
    size = 7
    zeros_float = np.zeros(size, dtype=np.float64)
    zeros_int = np.zeros(size, dtype=np.int64)
    return StructuralDiagnostics(
        transition=np.array([2, 3, 6, 7, 9, 10, 12]),
        move=np.array(["birth", "death", "birth", "death", "global_move", "birth", "death"]),
        invalid_reason_code=np.zeros(size, dtype=np.uint8),
        valid=np.ones(size, dtype=np.bool_),
        accepted=np.ones(size, dtype=np.bool_),
        source_k=np.array([1, 2, 1, 2, 1, 1, 2]),
        candidate_k=np.array([2, 1, 2, 1, 1, 2, 1]),
        result_k=np.array([2, 1, 2, 1, 1, 2, 1]),
        delta_log_likelihood=zeros_float,
        delta_log_coefficient_prior=zeros_float,
        delta_log_fixed_coefficient_prior=zeros_float,
        delta_log_k_prior=zeros_float,
        delta_log_nucleus_prior=zeros_float,
        delta_log_error_model_prior=zeros_float,
        delta_log_coefficient_hyperprior=zeros_float,
        delta_log_target=zeros_float,
        log_q_forward=zeros_float,
        log_q_reverse=zeros_float,
        log_jacobian=zeros_float,
        log_acceptance_ratio=zeros_float,
        source_nucleus=np.array([-1, 5, -1, 0, 7, -1, 2]),
        candidate_nucleus=np.array([5, -1, 7, -1, 2, 3, -1]),
        owner_changed_cell_count=zeros_int,
        owner_changed_cell_fraction=zeros_float,
        affected_candidate_design_column_count=zeros_int,
        prediction_change_l2=zeros_float,
        observation_error_standardized_prediction_change_l2=zeros_float,
        event_region_observation_error_standardized_design_l2=zeros_float,
        coefficient_contrast=zeros_float,
        initial_nuclei=np.array([0]),
        final_nuclei=np.array([3]),
        segment_transition_start=0,
        segment_transition_end=12,
        n_grid_cells=8,
        n_observations=3,
        provenance=_PROVENANCE,
    )


def test_derived_reversals_and_region_age_respect_left_censoring() -> None:
    """Derived events should separate lineage age from nucleus-cell residence."""
    derived = derive_structural_diagnostics(_synthetic_derived_diagnostics())

    np.testing.assert_array_equal(derived.k_step, [1, -1, 1, -1, 0, 1, -1])
    np.testing.assert_array_equal(
        derived.adjacent_accepted_opposite_k_reversal,
        [False, True, False, True, False, False, False],
    )
    np.testing.assert_array_equal(
        derived.exact_endpoint_reversal,
        [False, True, False, False, False, False, False],
    )
    np.testing.assert_array_equal(
        derived.removed_region_lineage_age,
        [-1, 1, -1, -1, -1, -1, 6],
    )
    np.testing.assert_array_equal(
        derived.removed_region_lineage_age_left_censored,
        [False, False, False, True, False, False, False],
    )
    residence = derive_nucleus_residence_intervals(_synthetic_derived_diagnostics())
    np.testing.assert_array_equal(residence.nucleus, [5, 0, 7, 2, 3])
    np.testing.assert_array_equal(residence.start_transition, [2, -1, 6, 9, 10])
    np.testing.assert_array_equal(residence.end_transition, [3, 7, 9, 12, -1])
    np.testing.assert_array_equal(
        residence.left_censored,
        [False, True, False, False, False],
    )
    np.testing.assert_array_equal(
        residence.right_censored,
        [False, False, False, False, True],
    )
    lineages = derive_region_lineage_intervals(_synthetic_derived_diagnostics())
    np.testing.assert_array_equal(lineages.lineage_id, [1, 0, 2, 3])
    np.testing.assert_array_equal(lineages.origin_nucleus, [5, 0, 7, 3])
    np.testing.assert_array_equal(lineages.start_transition, [2, -1, 6, 10])
    np.testing.assert_array_equal(lineages.end_transition, [3, 7, 12, -1])
    np.testing.assert_array_equal(
        lineages.left_censored,
        [False, True, False, False],
    )
    np.testing.assert_array_equal(
        lineages.right_censored,
        [False, False, False, True],
    )


def test_location_move_transfers_left_censored_region_lineage_age() -> None:
    """Relocating an initial region must not reset its unknown creation time."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])
    move = propose_global_move(
        problem,
        initial,
        move_position=0,
        new_nucleus=1,
    )
    assert move.valid
    moved = move.candidate
    moved_position = int(np.searchsorted(moved.active_nuclei, 1))
    deletion = propose_death(
        problem,
        moved,
        remove_position=moved_position,
        proposal_stdev=0.25,
    )
    assert deletion.valid

    buffer = _StructuralDiagnosticsBuffer(
        problem,
        initial,
        capacity=2,
        provenance=_PROVENANCE,
    )
    buffer.append(
        transition_number=1,
        source=initial,
        transition=move,
        result=moved,
        accepted=True,
    )
    buffer.append(
        transition_number=2,
        source=moved,
        transition=deletion,
        result=deletion.candidate,
        accepted=True,
    )
    diagnostics = buffer.finalize(deletion.candidate)
    derived = derive_structural_diagnostics(diagnostics)

    np.testing.assert_array_equal(derived.removed_region_lineage_age, [-1, -1])
    np.testing.assert_array_equal(
        derived.removed_region_lineage_age_left_censored,
        [False, True],
    )


def test_numpy_and_numba_structural_diagnostics_are_identical() -> None:
    """Both state backends should emit identical structural diagnostics."""
    problem = _problem()
    numpy_initial = build_state(problem, [0, 3], [0.8, 1.2], backend="numpy")
    numba_initial = build_state(problem, [0, 3], [0.8, 1.2], backend="numba")

    numpy_result = sample(
        problem,
        numpy_initial,
        _config(iterations=12, collect=True, backend="numpy"),
    )
    numba_result = sample(
        problem,
        numba_initial,
        _config(iterations=12, collect=True, backend="numba"),
    )

    assert numpy_result.structural_diagnostics is not None
    assert numba_result.structural_diagnostics is not None
    _assert_diagnostics_backend_close(
        numpy_result.structural_diagnostics,
        numba_result.structural_diagnostics,
    )


def test_structural_dataset_uses_global_transition_not_retained_draw() -> None:
    """The persistence boundary should preserve the proposal-level coordinate."""
    problem = _problem()
    initial = build_state(problem, [0, 3], [0.8, 1.2])
    result = sample(problem, initial, _config(iterations=12, collect=True))
    diagnostics = result.structural_diagnostics
    assert diagnostics is not None

    dataset = structural_diagnostics_to_dataset(
        diagnostics,
        metadata={"run_id": "tiny-profile"},
    )
    restored = structural_diagnostics_from_dataset(
        dataset,
        required_metadata={"run_id": "tiny-profile"},
    )

    assert dataset.sizes["structural_transition"] == diagnostics.size
    assert dataset.sizes["initial_region"] == diagnostics.initial_nuclei.size
    assert dataset.sizes["final_region"] == diagnostics.final_nuclei.size
    np.testing.assert_array_equal(
        dataset.structural_transition,
        diagnostics.transition,
    )
    np.testing.assert_array_equal(dataset.move, diagnostics.move)
    np.testing.assert_array_equal(dataset.initial_nuclei, diagnostics.initial_nuclei)
    np.testing.assert_array_equal(dataset.final_nuclei, diagnostics.final_nuclei)
    assert "draw" not in dataset.dims
    assert dataset.attrs["run_id"] == "tiny-profile"
    assert dataset.attrs["n_grid_cells"] == problem.n_grid_cells
    assert dataset.attrs["chain_id"] == _PROVENANCE.chain_id
    assert dataset.attrs["problem_fingerprint"] == _PROVENANCE.problem_fingerprint
    _assert_diagnostics_equal(restored, diagnostics)
    assert (
        "not full OU covariance whitening"
        in (dataset["observation_error_standardized_prediction_change_l2"].attrs["description"])
    )

    with pytest.raises(ValueError, match="reserved attributes"):
        structural_diagnostics_to_dataset(
            diagnostics,
            metadata={"n_grid_cells": problem.n_grid_cells + 1},
        )
    with pytest.raises(ValueError, match="required metadata"):
        structural_diagnostics_from_dataset(
            dataset,
            required_metadata={"run_id": "different-chain"},
        )
