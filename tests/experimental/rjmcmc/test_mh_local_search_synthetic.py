"""Focused gates for the MH-guided synthetic local-search harness."""

from __future__ import annotations

import argparse
from dataclasses import fields
import importlib.util
import json
import math
from pathlib import Path
import shutil
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.full_tiling_compound_sampling import (
    FullTilingCompoundConfig,
    FullTilingCompoundTrace,
    FullTilingMovementDiagnostics,
    continue_full_tiling_compound,
    sample_full_tiling_compound,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_io import (
    full_tiling_state_fingerprint,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_synthetic import (
    EVALUATION_SCHEMA,
    TRAINING_SCHEMA,
    build_stage_definition,
    common_native_totals,
    json_sha256,
    load_evaluation_artifact,
    load_training_artifact,
    materialize_replicate,
    prepare_fixed_basis_reference,
    problem_from_training,
    reconstruct_native_fields,
    scenario_topology,
    stage_operators,
    state_on_tiling,
    tiling_from_bounds,
    topology_bounds,
    topology_sha256,
    validate_artifact_pair,
    validate_stage_definition,
    write_envelope,
)

_DRIVER_PATH = Path(__file__).parents[3] / "examples" / "rjmcmc" / "mh_local_search_synthetic.py"


@pytest.fixture(scope="module")
def driver() -> ModuleType:
    """Load the command driver without invoking its entry point."""
    specification = importlib.util.spec_from_file_location(
        "mh_local_search_synthetic_driver",
        _DRIVER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load the synthetic experiment driver")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def s0_pair() -> tuple[Any, Any]:
    """Return one frozen S0 training/evaluation pair."""
    return materialize_replicate(
        build_stage_definition("s0"),
        scenario="edge-one",
        replicate=0,
    )


def test_stage_definitions_are_deterministic_and_separate_rows() -> None:
    """Definitions resolve operators, topologies, witnesses, and hashes."""
    first = build_stage_definition("s0")
    second = build_stage_definition("s0")
    assert json_sha256(first) == json_sha256(second)
    assert validate_stage_definition(first) == first
    assert first["p0_sha256"] == topology_sha256(tiling_from_bounds((2, 4), first["p0_bounds"]))
    training, training_ids, heldout, heldout_ids = stage_operators("s0")
    assert training.shape == (16, 8)
    assert heldout.shape == (10, 8)
    assert not set(training_ids) & set(heldout_ids)
    assert np.all(np.sum(training, axis=1) == 1.0)
    assert np.all(np.sum(heldout, axis=1) == 1.0)

    p0 = tiling_from_bounds((2, 4), first["p0_bounds"])
    scenarios = cast(dict[str, dict[str, Any]], first["scenarios"])
    for scenario in ("aligned", "edge-one", "relocation-one"):
        destination, witness = scenario_topology(p0, cast(Any, scenario))
        assert topology_bounds(destination) == tuple(
            tuple(row) for row in scenarios[scenario]["pstar_bounds"]
        )
        assert witness == scenarios[scenario]["witness"]
        truth = np.asarray(scenarios[scenario]["truth"], dtype=np.float64)
        weighted_mean = math.fsum(float(item) / truth.size for item in truth.ravel())
        assert weighted_mean == 1.0

    replay_drift = json.loads(json.dumps(first))
    replay_drift["training_operator"][0][0] = 0.5
    with pytest.raises(ValueError, match="exactly replay"):
        validate_stage_definition(replay_drift)
    with pytest.raises(ValueError, match="exactly replay"):
        materialize_replicate(
            replay_drift,
            scenario="aligned",
            replicate=0,
        )


def test_s1_operator_definition_is_reproducible_and_normalized() -> None:
    """The atmospheric-like operators use their frozen independent streams."""
    training_a, ids_a, heldout_a, heldout_ids_a = stage_operators("s1")
    training_b, ids_b, heldout_b, heldout_ids_b = stage_operators("s1")
    np.testing.assert_array_equal(training_a, training_b)
    np.testing.assert_array_equal(heldout_a, heldout_b)
    assert ids_a == ids_b
    assert heldout_ids_a == heldout_ids_b
    np.testing.assert_allclose(np.sum(training_a, axis=1), 1.0, rtol=0.0, atol=8 * np.spacing(1.0))
    np.testing.assert_allclose(np.sum(heldout_a, axis=1), 1.0, rtol=0.0, atol=8 * np.spacing(1.0))


def test_artifacts_are_physically_separate_and_fail_closed(
    tmp_path: Path,
    s0_pair: tuple[Any, Any],
) -> None:
    """Training has no sealed values and checksum/schema drift is rejected."""
    training, evaluation = s0_pair
    training_path = tmp_path / "training.json"
    evaluation_path = tmp_path / "evaluation.json"
    write_envelope(training_path, TRAINING_SCHEMA, training.payload())
    write_envelope(evaluation_path, EVALUATION_SCHEMA, evaluation.payload())
    loaded_training = load_training_artifact(training_path)
    loaded_evaluation = load_evaluation_artifact(evaluation_path)
    assert loaded_training.definition_sha256 == loaded_evaluation.definition_sha256
    assert loaded_training.cell_id == loaded_evaluation.cell_id
    assert loaded_training.generation_commitment == (loaded_evaluation.generation_commitment)
    validate_artifact_pair(loaded_training, loaded_evaluation)
    np.testing.assert_array_equal(loaded_training.operator, training.operator)
    np.testing.assert_array_equal(loaded_evaluation.truth, evaluation.truth)
    training_payload = json.loads(training_path.read_text())["payload"]
    assert not {
        "truth",
        "heldout_operator",
        "heldout_noiseless",
        "heldout_observations",
        "pstar_bounds",
        "witness",
        "scenario",
        "noise_seed",
    } & set(training_payload)

    corrupt = json.loads(training_path.read_text())
    corrupt["payload"]["unexpected"] = 1
    corrupt["payload_sha256"] = json_sha256(corrupt["payload"])
    extra_path = tmp_path / "training-extra.json"
    extra_path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="payload keys"):
        load_training_artifact(extra_path)

    checksum = json.loads(training_path.read_text())
    checksum["payload"]["observations"][0] += 1.0
    checksum_path = tmp_path / "training-corrupt.json"
    checksum_path.write_text(json.dumps(checksum))
    with pytest.raises(ValueError, match="checksum"):
        load_training_artifact(checksum_path)

    rehashed_training = json.loads(training_path.read_text())
    rehashed_training["payload"]["observations"][0] += 1.0
    rehashed_training["payload_sha256"] = json_sha256(rehashed_training["payload"])
    rehashed_training_path = tmp_path / "training-rehashed-drift.json"
    rehashed_training_path.write_text(json.dumps(rehashed_training))
    drifted_training = load_training_artifact(rehashed_training_path)
    with pytest.raises(ValueError, match="sealed noise replay"):
        validate_artifact_pair(drifted_training, loaded_evaluation)

    lossy_integer = json.loads(training_path.read_text())
    lossy_integer["payload"]["replicate"] = 0.0
    lossy_integer["payload_sha256"] = json_sha256(lossy_integer["payload"])
    lossy_path = tmp_path / "training-lossy-integer.json"
    lossy_path.write_text(json.dumps(lossy_integer))
    with pytest.raises(ValueError, match="exact integer"):
        load_training_artifact(lossy_path)

    wrong_definition = json.loads(training_path.read_text())
    wrong_definition["payload"]["definition_sha256"] = "0" * 64
    wrong_definition["payload_sha256"] = json_sha256(wrong_definition["payload"])
    wrong_definition_path = tmp_path / "training-definition-drift.json"
    wrong_definition_path.write_text(json.dumps(wrong_definition))
    with pytest.raises(ValueError, match="frozen definition"):
        load_training_artifact(wrong_definition_path)

    inconsistent_evaluation = json.loads(evaluation_path.read_text())
    inconsistent_evaluation["payload"]["heldout_noiseless"][0] += 0.01
    inconsistent_evaluation["payload_sha256"] = json_sha256(inconsistent_evaluation["payload"])
    inconsistent_path = tmp_path / "evaluation-inconsistent.json"
    inconsistent_path.write_text(json.dumps(inconsistent_evaluation))
    with pytest.raises(ValueError, match="heldout_noiseless"):
        load_evaluation_artifact(inconsistent_path)

    rehashed_heldout = json.loads(evaluation_path.read_text())
    rehashed_heldout["payload"]["heldout_observations"][0] += 1.0
    rehashed_heldout["payload_sha256"] = json_sha256(rehashed_heldout["payload"])
    rehashed_heldout_path = tmp_path / "evaluation-rehashed-heldout-drift.json"
    rehashed_heldout_path.write_text(json.dumps(rehashed_heldout))
    with pytest.raises(ValueError, match="observation realization"):
        load_evaluation_artifact(rehashed_heldout_path)


def test_native_reconstruction_matches_independent_rectangle_oracle(
    s0_pair: tuple[Any, Any],
) -> None:
    """Aligned masses reconstruct by nominal mass, including duplicates."""
    training, _ = s0_pair
    bounds = np.asarray([training.p0_bounds, training.p0_bounds], dtype=np.int64)
    masses = np.asarray(((0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4)))
    fields = reconstruct_native_fields(bounds, masses, training.nominal_weight)
    independent = np.empty_like(fields)
    normalized = training.nominal_weight / training.nominal_weight.sum()
    for draw in range(2):
        for leaf_index, (r0, r1, c0, c1) in enumerate(bounds[draw]):
            independent[draw, r0:r1, c0:c1] = masses[draw, leaf_index] / normalized[r0:r1, c0:c1].sum()
    np.testing.assert_array_equal(fields, independent)
    np.testing.assert_array_equal(np.mean(fields, axis=0), fields[0])
    contributions = common_native_totals(
        np.ones((2, *training.shape), dtype=np.float64),
        training.nominal_weight,
    )
    np.testing.assert_array_equal(
        contributions,
        np.asarray([[1.0, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]] * 2),
    )


def test_common_conditioned_state_can_fork_fixed_and_mobile(
    s0_pair: tuple[Any, Any],
) -> None:
    """Both schedules accept one exact immutable branch state."""
    training, _ = s0_pair
    problem = problem_from_training(training)
    branch = state_on_tiling(
        problem,
        tiling_from_bounds(training.shape, training.p0_bounds),
    )
    fingerprint = full_tiling_state_fingerprint(problem, branch)
    fixed = sample_full_tiling_compound(
        problem,
        branch,
        FullTilingCompoundConfig(
            iterations=6,
            seed=1,
            pair_allocation_refresh_slots=5,
            structure_mode="fixed_basis",
        ),
        collect_movement_diagnostics=True,
    )
    assert full_tiling_state_fingerprint(problem, branch) == fingerprint
    mobile = sample_full_tiling_compound(
        problem,
        branch,
        FullTilingCompoundConfig(
            iterations=8,
            seed=2,
            pair_allocation_refresh_slots=5,
            structure_mode="mobile",
        ),
        collect_movement_diagnostics=True,
    )
    assert full_tiling_state_fingerprint(problem, branch) == fingerprint
    assert fixed.trace.rectangle_bounds.shape[0] == 2
    assert mobile.trace.rectangle_bounds.shape[0] == 2
    assert np.count_nonzero(np.isin(mobile.trace.move, ("edge_flip", "resolution_relocation"))) == 2
    np.testing.assert_array_equal(
        fixed.trace.rectangle_bounds[0],
        mobile.trace.rectangle_bounds[0],
    )
    np.testing.assert_array_equal(fixed.trace.leaf_masses[0], mobile.trace.leaf_masses[0])


@pytest.mark.parametrize(("mode", "cycle_length"), (("fixed_basis", 6), ("mobile", 8)))
def test_two_hundred_cycle_chunking_matches_direct_science_rng_and_checkpoint(
    s0_pair: tuple[Any, Any],
    mode: str,
    cycle_length: int,
) -> None:
    """Two 100-cycle segments are replay-identical to one direct 200-cycle segment."""
    training, _ = s0_pair
    direct_problem = problem_from_training(training)
    direct_initial = state_on_tiling(
        direct_problem,
        tiling_from_bounds(training.shape, training.p0_bounds),
    )
    config = FullTilingCompoundConfig(
        iterations=200 * cycle_length,
        seed=9876,
        pair_allocation_refresh_slots=5,
        structure_mode=cast(Any, mode),
    )
    direct = sample_full_tiling_compound(
        direct_problem,
        direct_initial,
        config,
        collect_movement_diagnostics=True,
    )

    chunk_problem = problem_from_training(training)
    chunk_initial = state_on_tiling(
        chunk_problem,
        tiling_from_bounds(training.shape, training.p0_bounds),
    )
    first = sample_full_tiling_compound(
        chunk_problem,
        chunk_initial,
        FullTilingCompoundConfig(
            iterations=100 * cycle_length,
            seed=9876,
            pair_allocation_refresh_slots=5,
            structure_mode=cast(Any, mode),
        ),
        collect_movement_diagnostics=True,
    )
    second = continue_full_tiling_compound(
        chunk_problem,
        first.checkpoint,
        iterations=100 * cycle_length,
        collect_movement_diagnostics=True,
    )
    retained_names = {
        "rectangle_bounds",
        "leaf_masses",
        "root_total",
        "fixed_coefficients",
        "log_gaussian_likelihood",
        "log_likelihood",
        "log_root_prior",
        "log_allocation_prior",
        "log_structural_prior",
        "log_fixed_coefficient_prior",
        "log_target",
        "state_transition",
    }
    for item in fields(FullTilingCompoundTrace):
        first_values = np.asarray(getattr(first.trace, item.name))
        second_values = np.asarray(getattr(second.trace, item.name))
        combined = np.concatenate((first_values, second_values), axis=0)
        if item.name in retained_names:
            assert first_values.shape[0] == 101
        np.testing.assert_array_equal(combined, getattr(direct.trace, item.name))
    assert direct.movement_diagnostics is not None
    assert first.movement_diagnostics is not None
    assert second.movement_diagnostics is not None
    for item in fields(FullTilingMovementDiagnostics):
        if item.name in ("proposal_elapsed_ns", "diagnostic_elapsed_ns"):
            continue
        combined = np.concatenate(
            (
                np.asarray(getattr(first.movement_diagnostics, item.name)),
                np.asarray(getattr(second.movement_diagnostics, item.name)),
            )
        )
        np.testing.assert_array_equal(
            combined,
            getattr(direct.movement_diagnostics, item.name),
        )
    assert direct.checkpoint.rng_state.generator().bit_generator.state == (
        second.checkpoint.rng_state.generator().bit_generator.state
    )
    assert direct.checkpoint.transitions_completed == second.checkpoint.transitions_completed
    assert direct.checkpoint.schedule_phase == second.checkpoint.schedule_phase
    assert direct.checkpoint.schedule_id == second.checkpoint.schedule_id
    assert direct.checkpoint.kernel_settings == second.checkpoint.kernel_settings
    assert full_tiling_state_fingerprint(
        direct_problem,
        direct.final_state,
    ) == full_tiling_state_fingerprint(chunk_problem, second.final_state)


def test_reference_bridge_accepts_arbitrary_planted_topology(
    s0_pair: tuple[Any, Any],
) -> None:
    """Reference preparation is topology-general and does not sample NUTS."""
    training, evaluation = s0_pair
    _, state, data = prepare_fixed_basis_reference(training, evaluation.pstar_bounds)
    assert topology_bounds(state.allocation.tiling) == evaluation.pstar_bounds
    assert data.k == training.k
    assert data.observations.shape == training.observations.shape


def test_practical_cli_has_no_evaluation_or_truth_argument(driver: ModuleType) -> None:
    """The practical sampler parser cannot receive the sealed payload."""
    run_parser = next(
        action.choices["run-pair"] for action in driver.parser()._actions if getattr(action, "choices", None)
    )
    destinations = {action.dest for action in run_parser._actions}
    assert "training" in destinations
    assert (
        not {
            "evaluation",
            "truth",
            "heldout",
            "pstar",
            "witness",
            "scenario",
            "conditioning_cycles",
            "production_cycles",
            "pair_slots",
            "chunk_cycles",
        }
        & destinations
    )
    assert not driver._returned_after_departure(np.asarray([True, True, True]))
    assert not driver._returned_after_departure(np.asarray([True, False, False]))
    assert driver._returned_after_departure(np.asarray([True, False, True]))


def test_practical_cli_rejects_noncurrent_source_revision(
    tmp_path: Path,
    driver: ModuleType,
    s0_pair: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strict wrapper refuses caller provenance that is not current HEAD."""
    training, _ = s0_pair
    training_path = tmp_path / "training.json"
    write_envelope(training_path, TRAINING_SCHEMA, training.payload())
    monkeypatch.setattr(driver, "_current_clean_revision", lambda: "a" * 40)
    with pytest.raises(ValueError, match="current exact full Git SHA"):
        driver.command_run_pair(
            argparse.Namespace(
                training=training_path,
                output_directory=tmp_path / "run",
                source_revision="b" * 40,
            )
        )


def test_bounded_s0_driver_writes_complete_sealed_pair_and_analysis(
    tmp_path: Path,
    driver: ModuleType,
    s0_pair: tuple[Any, Any],
) -> None:
    """One frozen chunk exercises conditioning, branching, continuation IO, and scoring."""
    training, evaluation = s0_pair
    training_path = tmp_path / "input" / "training.json"
    evaluation_path = tmp_path / "sealed" / "evaluation.json"
    write_envelope(training_path, TRAINING_SCHEMA, training.payload())
    write_envelope(evaluation_path, EVALUATION_SCHEMA, evaluation.payload())
    run_directory = tmp_path / "practical-run"
    driver._run_short_pair_for_test(
        training_path=training_path,
        output_directory=run_directory,
        conditioning_cycles=1,
        production_cycles=100,
    )
    assert (run_directory / "complete.json").is_file()
    assert not any("evaluation" in path.name for path in run_directory.rglob("*"))
    practical_manifest = json.loads((run_directory / "manifest.json").read_text())
    assert "scenario" not in practical_manifest
    assert "noise_seed" not in practical_manifest
    assert practical_manifest["cell_id"] == training.cell_id
    assert practical_manifest["definition_sha256"] == training.definition_sha256
    assert practical_manifest["generation_commitment"] == (training.generation_commitment)
    with np.load(run_directory / "fixed" / "trace.npz", allow_pickle=False) as fixed:
        assert fixed["cycle"].shape == (100,)
        assert np.count_nonzero(np.isin(fixed["attempt_move"], ("edge_flip", "resolution_relocation"))) == 0
    with np.load(run_directory / "mobile" / "trace.npz", allow_pickle=False) as mobile:
        assert mobile["cycle"].shape == (100,)
        assert (
            np.count_nonzero(np.isin(mobile["attempt_move"], ("edge_flip", "resolution_relocation"))) == 200
        )

    analysis_directory = tmp_path / "analysis"
    with pytest.raises(ValueError, match="frozen stage budgets"):
        driver.command_analyze(
            argparse.Namespace(
                training=training_path,
                evaluation=evaluation_path,
                run_directory=run_directory,
                output_directory=analysis_directory,
            )
        )
    driver._analyze_short_run_for_test(
        training_path=training_path,
        evaluation_path=evaluation_path,
        run_directory=run_directory,
        output_directory=analysis_directory,
    )
    analysis = json.loads((analysis_directory / "analysis.json").read_text())
    assert analysis["fixed"]["all_cycle_heldout_rmse"] >= 0.0
    assert analysis["mobile"]["all_cycle_heldout_rmse"] >= 0.0
    assert analysis["cell_id"] == training.cell_id
    assert (analysis_directory / "complete.json").is_file()
    _, mismatched_evaluation = materialize_replicate(
        build_stage_definition("s0"),
        scenario="relocation-one",
        replicate=0,
    )
    mismatch_path = tmp_path / "sealed" / "evaluation-mismatch.json"
    write_envelope(
        mismatch_path,
        EVALUATION_SCHEMA,
        mismatched_evaluation.payload(),
    )
    with pytest.raises(ValueError, match="same cell"):
        driver._analyze_short_run_for_test(
            training_path=training_path,
            evaluation_path=mismatch_path,
            run_directory=run_directory,
            output_directory=tmp_path / "mismatch-analysis",
        )
    with pytest.raises(FileExistsError):
        driver._run_short_pair_for_test(
            training_path=training_path,
            output_directory=run_directory,
            conditioning_cycles=1,
            production_cycles=100,
        )


def _refresh_fixed_completion_hashes(run_directory: Path, driver: ModuleType) -> None:
    arm_complete_path = run_directory / "fixed" / "complete.json"
    arm_complete = json.loads(arm_complete_path.read_text())
    arm_complete["files"]["trace.npz"] = driver.file_sha256(run_directory / "fixed" / "trace.npz")
    arm_complete_path.write_text(driver.canonical_json(arm_complete) + "\n")
    top_complete_path = run_directory / "complete.json"
    top_complete = json.loads(top_complete_path.read_text())
    top_complete["files"]["fixed/complete.json"] = driver.file_sha256(arm_complete_path)
    top_complete_path.write_text(driver.canonical_json(top_complete) + "\n")


def test_completion_corruption_and_trace_invariants_fail_closed(
    tmp_path: Path,
    driver: ModuleType,
    s0_pair: tuple[Any, Any],
) -> None:
    """Completion schemas, byte corruption, and self-consistent trace drift all fail."""
    training, evaluation = s0_pair
    training_path = tmp_path / "training.json"
    evaluation_path = tmp_path / "evaluation.json"
    write_envelope(training_path, TRAINING_SCHEMA, training.payload())
    write_envelope(evaluation_path, EVALUATION_SCHEMA, evaluation.payload())
    pristine = tmp_path / "pristine"
    driver._run_short_pair_for_test(
        training_path=training_path,
        output_directory=pristine,
        conditioning_cycles=1,
        production_cycles=100,
    )

    schema_run = tmp_path / "schema-run"
    shutil.copytree(pristine, schema_run)
    arm_complete_path = schema_run / "fixed" / "complete.json"
    arm_complete = json.loads(arm_complete_path.read_text())
    arm_complete["schema"] = "wrong"
    arm_complete_path.write_text(driver.canonical_json(arm_complete) + "\n")
    top_complete_path = schema_run / "complete.json"
    top_complete = json.loads(top_complete_path.read_text())
    top_complete["files"]["fixed/complete.json"] = driver.file_sha256(arm_complete_path)
    top_complete_path.write_text(driver.canonical_json(top_complete) + "\n")
    with pytest.raises(ValueError, match="fixed completion schema"):
        driver._analyze_short_run_for_test(
            training_path=training_path,
            evaluation_path=evaluation_path,
            run_directory=schema_run,
            output_directory=tmp_path / "schema-analysis",
        )

    corrupt_run = tmp_path / "corrupt-run"
    shutil.copytree(pristine, corrupt_run)
    with (corrupt_run / "fixed" / "trace.npz").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="fixed completion checksum"):
        driver._analyze_short_run_for_test(
            training_path=training_path,
            evaluation_path=evaluation_path,
            run_directory=corrupt_run,
            output_directory=tmp_path / "corrupt-analysis",
        )

    invariant_run = tmp_path / "invariant-run"
    shutil.copytree(pristine, invariant_run)
    trace_path = invariant_run / "fixed" / "trace.npz"
    arrays = driver._load_trace(trace_path)
    arrays["rectangle_bounds"] = arrays["rectangle_bounds"].astype(np.float64)
    arrays["rectangle_bounds"][0, 0, 1] += 0.5
    trace_path.unlink()
    driver._create_npz(trace_path, arrays)
    _refresh_fixed_completion_hashes(invariant_run, driver)
    with pytest.raises(ValueError, match="exact ndarray dtype"):
        driver._analyze_short_run_for_test(
            training_path=training_path,
            evaluation_path=evaluation_path,
            run_directory=invariant_run,
            output_directory=tmp_path / "invariant-analysis",
        )

    topology_run = tmp_path / "topology-run"
    shutil.copytree(pristine, topology_run)
    topology_trace_path = topology_run / "fixed" / "trace.npz"
    topology_arrays = driver._load_trace(topology_trace_path)
    topology_arrays["rectangle_bounds"][0, 0] = np.asarray((0, 2, 0, 4))
    topology_trace_path.unlink()
    driver._create_npz(topology_trace_path, topology_arrays)
    _refresh_fixed_completion_hashes(topology_run, driver)
    with pytest.raises(ValueError, match="retained topology at cycle 1 is invalid"):
        driver._analyze_short_run_for_test(
            training_path=training_path,
            evaluation_path=evaluation_path,
            run_directory=topology_run,
            output_directory=tmp_path / "topology-analysis",
        )
