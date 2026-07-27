#!/usr/bin/env python3
"""Run the bounded MH-guided synthetic local-search experiment.

The ``run-pair`` command intentionally has no evaluation-artifact argument.
It conditions and samples using a training-only artifact.  Truth and
held-out data are opened only by ``analyze`` and ``prepare-reference``.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
from pathlib import Path
import re
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from openghg_inversions.experimental.rjmcmc.full_tiling_compound_sampling import (
    FIXED_BASIS_COMPOUND_SCHEDULE_ID,
    FullTilingCompoundConfig,
    FullTilingMovementDiagnostics,
    FullTilingCompoundSamplingResult,
    continue_full_tiling_compound,
    sample_full_tiling_compound,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_io import (
    full_tiling_state_fingerprint,
    save_full_tiling_checkpoint,
)
from openghg_inversions.experimental.rjmcmc.full_tiling import TilingState
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    build_full_tiling_posterior_state,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_conditional_reference import (
    validate_conditional_reference_record,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_retry_authorization import (
    validate_retry_authorization_bundle,
    validate_retry_authorization_token,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_synthetic import (
    DEFINITION_SCHEMA,
    EVALUATION_SCHEMA,
    TRAINING_SCHEMA,
    build_stage_definition,
    canonical_json,
    common_native_totals,
    file_sha256,
    frozen_local_reference_seeds,
    frozen_oracle_settings,
    frozen_stage_budgets,
    json_sha256,
    load_evaluation_artifact,
    load_training_artifact,
    materialize_replicate,
    prepare_fixed_basis_reference,
    problem_from_training,
    read_envelope,
    reconstruct_native_fields,
    state_on_tiling,
    tiling_from_bounds,
    topology_bounds,
    topology_sha256,
    validate_artifact_pair,
    validate_local_reference_trace,
    validate_stage_definition,
    write_envelope,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_PAIR_MANIFEST_KEYS = frozenset(
    (
        "schema",
        "stage",
        "replicate",
        "definition_sha256",
        "cell_id",
        "generation_commitment",
        "source_revision",
        "training_sha256",
        "branch_state_fingerprint",
        "budget_profile",
        "conditioning_arm",
        "conditioning_cycles",
        "conditioning_schedule_id",
        "conditioning_seed",
        "conditioning_sampler_seconds",
        "production_cycles",
        "chunk_cycles",
        "pair_slots",
        "fixed_seed",
        "mobile_seed",
    )
)
_PAIR_RETRY_MANIFEST_KEYS = _PAIR_MANIFEST_KEYS | {"retry_authorization_token_sha256"}
_ORACLE_MANIFEST_KEYS = frozenset(
    (
        "schema",
        "stage",
        "scenario",
        "replicate",
        "definition_sha256",
        "cell_id",
        "generation_commitment",
        "source_revision",
        "training_sha256",
        "evaluation_sha256",
        "pstar_sha256",
        "witness_sha256",
        "branch_state_fingerprint",
        "budget_profile",
        "conditioning_arm",
        "conditioning_cycles",
        "conditioning_schedule_id",
        "conditioning_seed",
        "conditioning_sampler_seconds",
        "production_cycles",
        "chunk_cycles",
        "pair_slots",
        "oracle_seed",
    )
)
_ORACLE_RETRY_MANIFEST_KEYS = _ORACLE_MANIFEST_KEYS | {"retry_authorization_token_sha256"}
_S0_INDEX_KEYS = frozenset(
    (
        "schema",
        "candidate_revision",
        "definition_path",
        "definition_file_sha256",
        "cells",
        "reference_artifacts",
        "conditional_references",
    )
)
_S0_RETRY_INDEX_KEYS = _S0_INDEX_KEYS | {"retry_authorization"}
_S0_RETRY_AUTHORIZATION_KEYS = frozenset(
    (
        "authorization_completion_path",
        "authorization_completion_sha256",
        "primary_certificate_completion_path",
        "primary_certificate_completion_sha256",
        "primary_nuts_completion_path",
        "primary_nuts_completion_sha256",
        "primary_local_completion_path",
        "primary_local_completion_sha256",
    )
)
_S0_CELL_KEYS = frozenset(
    (
        "scenario",
        "replicate",
        "training_path",
        "training_sha256",
        "evaluation_path",
        "evaluation_sha256",
        "practical_run_directory",
        "practical_complete_sha256",
        "practical_analysis_directory",
        "practical_analysis_complete_sha256",
        "oracle_run_directory",
        "oracle_complete_sha256",
    )
)
_REFERENCE_ARTIFACT_KEYS = frozenset(("path", "sha256"))
_CONDITIONAL_REFERENCE_KEYS = frozenset(
    (
        "cell_id",
        "definition_sha256",
        "topology_sha256",
        "nuts_artifact_sha256",
        "local_artifact_sha256",
        "profile",
        "pass",
        "divergences",
        "worst_rhat_variable",
        "worst_rhat_value",
        "min_bulk_ess_variable",
        "min_bulk_ess_value",
        "min_tail_ess_variable",
        "min_tail_ess_value",
        "worst_local_mcse_sd_projection",
        "worst_local_mcse_sd_value",
        "worst_half_difference_sd_projection",
        "worst_half_difference_sd_value",
        "worst_local_vs_nuts_tolerance_projection",
        "worst_local_vs_nuts_tolerance_value",
        "first_failed_gate",
    )
)
TRACE_RETAINED_FIELDS = (
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
)
TRACE_ATTEMPT_FIELDS = (
    "global_transition",
    "slot",
    "move",
    "valid",
    "accepted",
    "log_acceptance_ratio",
    "invalid_reason",
)
MOVEMENT_FIELDS = tuple(item.name for item in fields(FullTilingMovementDiagnostics))


def _create_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(dict(payload)) + "\n")
        handle.flush()


def _create_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()


def _create_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(cast(Any, handle), **cast(Any, arrays))
        handle.flush()


def _read_strict_json(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {token}")),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return cast(dict[str, object], value)


def _verified_hashes(directory: Path, names: Sequence[str]) -> dict[str, str]:
    hashes = {name: file_sha256(directory / name) for name in names}
    if any(file_sha256(directory / name) != digest for name, digest in hashes.items()):
        raise RuntimeError("independent output checksum validation failed")
    return hashes


def _write_complete(directory: Path, names: Sequence[str]) -> None:
    _create_json(
        directory / "complete.json",
        {
            "schema": "openghg_inversions.mh_local_search_complete.v1",
            "files": _verified_hashes(directory, names),
        },
    )


def _conditioned_state(
    training: Any,
    *,
    conditioning_cycles: int,
    pair_slots: int,
) -> tuple[Any, Any, str, float]:
    return _conditioned_state_on_tiling(
        training,
        bounds=training.p0_bounds,
        seed=training.conditioning_seed,
        conditioning_cycles=conditioning_cycles,
        pair_slots=pair_slots,
    )


def _conditioned_state_on_tiling(
    training: Any,
    *,
    bounds: object,
    seed: int,
    conditioning_cycles: int,
    pair_slots: int,
) -> tuple[Any, Any, str, float]:
    problem = problem_from_training(training)
    expected_tiling = tiling_from_bounds(training.shape, bounds)
    initial = state_on_tiling(problem, expected_tiling)
    started = perf_counter()
    result = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(
            iterations=conditioning_cycles * (1 + pair_slots),
            seed=seed,
            pair_allocation_refresh_slots=pair_slots,
            structure_mode="fixed_basis",
        ),
    )
    elapsed = perf_counter() - started
    if result.checkpoint.schedule_id != FIXED_BASIS_COMPOUND_SCHEDULE_ID:
        raise RuntimeError("conditioning did not use the frozen fixed-basis schedule")
    branch = result.final_state
    if topology_bounds(branch.allocation.tiling) != topology_bounds(expected_tiling):
        raise RuntimeError("conditioning changed the fixed basis")
    return problem, branch, full_tiling_state_fingerprint(problem, branch), elapsed


def _run_arm(
    *,
    problem: Any,
    branch_state: Any,
    mode: str,
    seed: int,
    cycles: int,
    chunk_cycles: int,
    pair_slots: int,
    output_directory: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    cycle_length = 1 + pair_slots + (2 if mode == "mobile" else 0)
    retained: dict[str, list[NDArray[Any]]] = {name: [] for name in TRACE_RETAINED_FIELDS}
    attempted: dict[str, list[NDArray[Any]]] = {name: [] for name in TRACE_ATTEMPT_FIELDS}
    movement: dict[str, list[NDArray[Any]]] = {name: [] for name in MOVEMENT_FIELDS}
    chunk_end: list[int] = []
    chunk_seconds: list[float] = []
    checkpoint = None
    result: FullTilingCompoundSamplingResult | None = None

    for chunk_start in range(0, cycles, chunk_cycles):
        this_cycles = min(chunk_cycles, cycles - chunk_start)
        started = perf_counter()
        if checkpoint is None:
            result = sample_full_tiling_compound(
                problem,
                branch_state,
                FullTilingCompoundConfig(
                    iterations=this_cycles * cycle_length,
                    seed=seed,
                    pair_allocation_refresh_slots=pair_slots,
                    structure_mode=cast(Any, mode),
                ),
                collect_movement_diagnostics=True,
            )
            retained_slice = slice(1, None)
        else:
            result = continue_full_tiling_compound(
                problem,
                checkpoint,
                iterations=this_cycles * cycle_length,
                collect_movement_diagnostics=True,
            )
            retained_slice = slice(None)
        elapsed = perf_counter() - started
        checkpoint = result.checkpoint
        diagnostics = result.movement_diagnostics
        if diagnostics is None:
            raise RuntimeError("movement diagnostics were not collected")
        for name in TRACE_RETAINED_FIELDS:
            retained[name].append(np.asarray(getattr(result.trace, name))[retained_slice])
        for name in TRACE_ATTEMPT_FIELDS:
            attempted[name].append(np.asarray(getattr(result.trace, name)))
        for name in MOVEMENT_FIELDS:
            movement[name].append(np.asarray(getattr(diagnostics, name)))
        chunk_end.append(chunk_start + this_cycles)
        chunk_seconds.append(elapsed)

    if result is None or checkpoint is None:
        raise RuntimeError("production run contained no chunks")
    arrays: dict[str, NDArray[Any]] = {
        name: np.concatenate(parts, axis=0) for name, parts in retained.items()
    }
    arrays.update({f"attempt_{name}": np.concatenate(parts, axis=0) for name, parts in attempted.items()})
    arrays.update({f"movement_{name}": np.concatenate(parts, axis=0) for name, parts in movement.items()})
    arrays["cycle"] = np.arange(1, cycles + 1, dtype=np.int64)
    arrays["chunk_end_cycle"] = np.asarray(chunk_end, dtype=np.int64)
    arrays["chunk_sampler_seconds"] = np.asarray(chunk_seconds, dtype=np.float64)
    if arrays["rectangle_bounds"].shape[0] != cycles:
        raise RuntimeError("production trace does not retain every post-cycle state")
    structural = np.isin(arrays["attempt_move"], ("edge_flip", "resolution_relocation"))
    expected_structural = 2 * cycles if mode == "mobile" else 0
    if int(np.count_nonzero(structural)) != expected_structural:
        raise RuntimeError("structure schedule did not produce the declared opportunities")
    if mode == "fixed_basis" and np.any(arrays["rectangle_bounds"] != arrays["rectangle_bounds"][0]):
        raise RuntimeError("fixed-basis production changed topology")

    output_directory.mkdir(parents=False, exist_ok=False)
    _create_npz(output_directory / "trace.npz", arrays)
    checkpoint_manifest = {**manifest, "arm": mode, "cycle_length": cycle_length}
    save_full_tiling_checkpoint(
        output_directory / "checkpoint.npz",
        checkpoint,
        run_manifest=checkpoint_manifest,
    )
    summary: dict[str, object] = {
        "schema": "openghg_inversions.mh_local_search_arm_summary.v1",
        "arm": mode,
        "cycles": cycles,
        "cycle_length": cycle_length,
        "retained_states": int(arrays["cycle"].size),
        "atomic_transitions": int(arrays["attempt_global_transition"].size),
        "structural_attempts": int(np.count_nonzero(structural)),
        "valid_structural": int(np.count_nonzero(structural & arrays["attempt_valid"])),
        "accepted_structural": int(np.count_nonzero(structural & arrays["attempt_accepted"])),
        "sampler_seconds": math.fsum(chunk_seconds),
        "final_state_fingerprint": full_tiling_state_fingerprint(problem, result.final_state),
        "training_sha256": manifest["training_sha256"],
        "branch_state_fingerprint": manifest["branch_state_fingerprint"],
    }
    _create_json(output_directory / "summary.json", summary)
    _write_complete(
        output_directory,
        ("trace.npz", "checkpoint.npz", "summary.json"),
    )
    return summary


def command_define(args: argparse.Namespace) -> None:
    definition = build_stage_definition(cast(Any, args.stage))
    write_envelope(args.output, DEFINITION_SCHEMA, definition)


def command_materialize(args: argparse.Namespace) -> None:
    definition = read_envelope(args.definition, schema=DEFINITION_SCHEMA)
    training, evaluation = materialize_replicate(
        definition,
        scenario=cast(Any, args.scenario),
        replicate=args.replicate,
    )
    write_envelope(args.training_output, TRAINING_SCHEMA, training.payload())
    write_envelope(args.evaluation_output, EVALUATION_SCHEMA, evaluation.payload())


def command_materialize_training(args: argparse.Namespace) -> None:
    definition = read_envelope(args.definition, schema=DEFINITION_SCHEMA)
    training, _ = materialize_replicate(
        definition,
        scenario=cast(Any, args.scenario),
        replicate=args.replicate,
    )
    write_envelope(args.training_output, TRAINING_SCHEMA, training.payload())


def command_materialize_evaluation(args: argparse.Namespace) -> None:
    definition = read_envelope(args.definition, schema=DEFINITION_SCHEMA)
    expected_training, evaluation = materialize_replicate(
        definition,
        scenario=cast(Any, args.scenario),
        replicate=args.replicate,
    )
    published_training = load_training_artifact(args.training)
    if canonical_json(published_training.payload()) != canonical_json(expected_training.payload()):
        raise ValueError("published training artifact differs from frozen materialization")
    write_envelope(args.evaluation_output, EVALUATION_SCHEMA, evaluation.payload())


def _current_clean_revision() -> str:
    revision = subprocess.run(
        ("git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if _FULL_SHA.fullmatch(revision) is None:
        raise RuntimeError("Git did not return an exact full source SHA")
    status = subprocess.run(
        ("git", "-C", str(_REPOSITORY_ROOT), "status", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("practical sampling requires a clean source worktree")
    return revision


def _frozen_reference_profile_budgets(
    stage: str,
    profile: str,
) -> tuple[int, int, int]:
    """Return the sole primary or factor-four branch/reference budget."""
    conditioning_cycles, production_cycles, pair_slots = frozen_stage_budgets(cast(Any, stage))
    if profile == "primary":
        factor = 1
    elif profile == "factor4":
        factor = 4
    else:
        raise ValueError("reference profile must be primary or factor4")
    return conditioning_cycles * factor, production_cycles * factor, pair_slots


def _retry_authorization_for_profile(
    *,
    profile: str,
    token_path: Path | None,
    source_revision: str,
    definition_sha256: str,
    stage: str,
) -> str | None:
    if profile == "factor4":
        if token_path is None:
            raise ValueError("factor4 requires --retry-authorization-token")
        return validate_retry_authorization_token(
            token_path,
            source_revision=source_revision,
            definition_sha256=definition_sha256,
            stage=stage,
        )
    if token_path is not None:
        raise ValueError("retry authorization cannot be supplied to a primary run")
    return None


def _run_pair(
    *,
    training_path: Path,
    output_directory: Path,
    conditioning_cycles: int,
    production_cycles: int,
    pair_slots: int,
    source_revision: str,
    budget_profile: str,
    retry_authorization_token_sha256: str | None,
) -> None:
    if production_cycles < 1 or conditioning_cycles < 1:
        raise ValueError("cycle counts must be positive")
    if pair_slots != 5:
        raise ValueError("allocation-pair refresh slots are frozen at five")
    chunk_cycles = 100
    if production_cycles % chunk_cycles:
        raise ValueError("production cycles must be divisible by 100")
    if _FULL_SHA.fullmatch(source_revision) is None:
        raise ValueError("source_revision must be an exact lower-case full Git SHA")
    training = load_training_artifact(training_path)
    if budget_profile != "test-short":
        expected = _frozen_reference_profile_budgets(training.stage, budget_profile)
        if (conditioning_cycles, production_cycles, pair_slots) != expected:
            raise ValueError("pair run does not match its frozen budget profile")
    if (budget_profile == "factor4") != (retry_authorization_token_sha256 is not None):
        raise ValueError("pair retry lineage does not match its budget profile")
    output_directory.mkdir(parents=True, exist_ok=False)
    training_digest = file_sha256(training_path)
    problem, branch, branch_digest, conditioning_seconds = _conditioned_state(
        training,
        conditioning_cycles=conditioning_cycles,
        pair_slots=pair_slots,
    )
    branch_arrays: dict[str, NDArray[Any]] = {
        "rectangle_bounds": np.asarray(topology_bounds(branch.allocation.tiling), dtype=np.int64),
        "leaf_masses": np.asarray(branch.leaf_masses, dtype=np.float64),
        "fixed_coefficients": np.asarray(branch.fixed_coefficients, dtype=np.float64),
        "log_target": np.asarray(branch.log_target, dtype=np.float64),
    }
    _create_npz(output_directory / "branch_state.npz", branch_arrays)
    manifest: dict[str, object] = {
        "schema": "openghg_inversions.mh_local_search_pair_manifest.v1",
        "stage": training.stage,
        "replicate": training.replicate,
        "definition_sha256": training.definition_sha256,
        "cell_id": training.cell_id,
        "generation_commitment": training.generation_commitment,
        "source_revision": source_revision,
        "training_sha256": training_digest,
        "branch_state_fingerprint": branch_digest,
        "budget_profile": budget_profile,
        "conditioning_arm": "fixed_basis",
        "conditioning_cycles": conditioning_cycles,
        "conditioning_schedule_id": FIXED_BASIS_COMPOUND_SCHEDULE_ID,
        "conditioning_seed": training.conditioning_seed,
        "conditioning_sampler_seconds": conditioning_seconds,
        "production_cycles": production_cycles,
        "chunk_cycles": chunk_cycles,
        "pair_slots": pair_slots,
        "fixed_seed": training.fixed_seed,
        "mobile_seed": training.mobile_seed,
    }
    if retry_authorization_token_sha256 is not None:
        _require_digest(
            retry_authorization_token_sha256,
            name="retry_authorization_token_sha256",
        )
        manifest["schema"] = "openghg_inversions.mh_local_search_pair_manifest.v2"
        manifest["retry_authorization_token_sha256"] = retry_authorization_token_sha256
    _create_json(output_directory / "manifest.json", manifest)
    fixed_start = full_tiling_state_fingerprint(problem, branch)
    fixed = _run_arm(
        problem=problem,
        branch_state=branch,
        mode="fixed_basis",
        seed=training.fixed_seed,
        cycles=production_cycles,
        chunk_cycles=chunk_cycles,
        pair_slots=pair_slots,
        output_directory=output_directory / "fixed",
        manifest=manifest,
    )
    if full_tiling_state_fingerprint(problem, branch) != fixed_start:
        raise RuntimeError("fixed arm mutated the immutable branch state")
    mobile = _run_arm(
        problem=problem,
        branch_state=branch,
        mode="mobile",
        seed=training.mobile_seed,
        cycles=production_cycles,
        chunk_cycles=chunk_cycles,
        pair_slots=pair_slots,
        output_directory=output_directory / "mobile",
        manifest=manifest,
    )
    if fixed["branch_state_fingerprint"] != mobile["branch_state_fingerprint"]:
        raise RuntimeError("arm branch-state identities differ")
    _write_complete(
        output_directory,
        ("branch_state.npz", "manifest.json", "fixed/complete.json", "mobile/complete.json"),
    )


def _run_short_pair_for_test(
    *,
    training_path: Path,
    output_directory: Path,
    conditioning_cycles: int,
    production_cycles: int,
) -> None:
    revision = subprocess.run(
        ("git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _run_pair(
        training_path=training_path,
        output_directory=output_directory,
        conditioning_cycles=conditioning_cycles,
        production_cycles=production_cycles,
        pair_slots=5,
        source_revision=revision,
        budget_profile="test-short",
        retry_authorization_token_sha256=None,
    )


def command_run_pair(args: argparse.Namespace) -> None:
    training = load_training_artifact(args.training)
    profile = getattr(args, "profile", "primary")
    conditioning_cycles, production_cycles, pair_slots = _frozen_reference_profile_budgets(
        training.stage,
        profile,
    )
    revision = _current_clean_revision()
    if args.source_revision != revision:
        raise ValueError("--source-revision must equal the current exact full Git SHA")
    retry_digest = _retry_authorization_for_profile(
        profile=profile,
        token_path=getattr(args, "retry_authorization_token", None),
        source_revision=revision,
        definition_sha256=training.definition_sha256,
        stage=training.stage,
    )
    _run_pair(
        training_path=args.training,
        output_directory=args.output_directory,
        conditioning_cycles=conditioning_cycles,
        production_cycles=production_cycles,
        pair_slots=pair_slots,
        source_revision=revision,
        budget_profile=profile,
        retry_authorization_token_sha256=retry_digest,
    )


def _run_oracle(
    *,
    training_path: Path,
    evaluation_path: Path,
    output_directory: Path,
    conditioning_cycles: int,
    production_cycles: int,
    conditioning_seed: int,
    sampler_seed: int,
    pair_slots: int,
    source_revision: str,
    budget_profile: str,
    retry_authorization_token_sha256: str | None,
) -> None:
    if production_cycles % 100 or pair_slots != 5:
        raise ValueError("oracle schedule violates the frozen 100-cycle/five-pair contract")
    if _FULL_SHA.fullmatch(source_revision) is None:
        raise ValueError("source_revision must be an exact lower-case full Git SHA")
    training = load_training_artifact(training_path)
    evaluation = load_evaluation_artifact(evaluation_path)
    validate_artifact_pair(training, evaluation)
    if budget_profile != "test-short":
        expected = _frozen_reference_profile_budgets(training.stage, budget_profile)
        if (conditioning_cycles, production_cycles, pair_slots) != expected:
            raise ValueError("oracle run does not match its frozen budget profile")
    if (budget_profile == "factor4") != (retry_authorization_token_sha256 is not None):
        raise ValueError("oracle retry lineage does not match its budget profile")
    output_directory.mkdir(parents=True, exist_ok=False)
    problem, branch, branch_digest, conditioning_seconds = _conditioned_state_on_tiling(
        training,
        bounds=evaluation.pstar_bounds,
        seed=conditioning_seed,
        conditioning_cycles=conditioning_cycles,
        pair_slots=pair_slots,
    )
    pstar = tiling_from_bounds(training.shape, evaluation.pstar_bounds)
    branch_arrays: dict[str, NDArray[Any]] = {
        "rectangle_bounds": np.asarray(topology_bounds(branch.allocation.tiling), dtype=np.int64),
        "leaf_masses": np.asarray(branch.leaf_masses, dtype=np.float64),
        "fixed_coefficients": np.asarray(branch.fixed_coefficients, dtype=np.float64),
        "log_target": np.asarray(branch.log_target, dtype=np.float64),
    }
    _create_npz(output_directory / "branch_state.npz", branch_arrays)
    manifest: dict[str, object] = {
        "schema": "openghg_inversions.mh_local_search_oracle_manifest.v1",
        "stage": training.stage,
        "scenario": evaluation.scenario,
        "replicate": training.replicate,
        "definition_sha256": training.definition_sha256,
        "cell_id": training.cell_id,
        "generation_commitment": training.generation_commitment,
        "source_revision": source_revision,
        "training_sha256": file_sha256(training_path),
        "evaluation_sha256": file_sha256(evaluation_path),
        "pstar_sha256": topology_sha256(pstar),
        "witness_sha256": json_sha256(dict(evaluation.witness)),
        "branch_state_fingerprint": branch_digest,
        "budget_profile": budget_profile,
        "conditioning_arm": "fixed_basis",
        "conditioning_cycles": conditioning_cycles,
        "conditioning_schedule_id": FIXED_BASIS_COMPOUND_SCHEDULE_ID,
        "conditioning_seed": conditioning_seed,
        "conditioning_sampler_seconds": conditioning_seconds,
        "production_cycles": production_cycles,
        "chunk_cycles": 100,
        "pair_slots": pair_slots,
        "oracle_seed": sampler_seed,
    }
    if retry_authorization_token_sha256 is not None:
        _require_digest(
            retry_authorization_token_sha256,
            name="retry_authorization_token_sha256",
        )
        manifest["schema"] = "openghg_inversions.mh_local_search_oracle_manifest.v2"
        manifest["retry_authorization_token_sha256"] = retry_authorization_token_sha256
    _create_json(output_directory / "manifest.json", manifest)
    _run_arm(
        problem=problem,
        branch_state=branch,
        mode="fixed_basis",
        seed=sampler_seed,
        cycles=production_cycles,
        chunk_cycles=100,
        pair_slots=pair_slots,
        output_directory=output_directory / "oracle",
        manifest=manifest,
    )
    trace = _validated_trace(
        output_directory / "oracle" / "trace.npz",
        arm="fixed",
        manifest=manifest,
        shape=training.shape,
        k=training.k,
    )
    score, arrays = _score_trace(
        trace,
        nominal_weight=training.nominal_weight,
        truth=evaluation.truth,
        heldout_operator=evaluation.heldout_operator,
        heldout_noiseless=evaluation.heldout_noiseless,
        heldout_observations=evaluation.heldout_observations,
        heldout_sd=evaluation.heldout_sd,
        p0_bounds=training.p0_bounds,
        pstar_bounds=evaluation.pstar_bounds,
    )
    _create_json(
        output_directory / "analysis.json",
        {
            "schema": "openghg_inversions.mh_local_search_oracle_analysis.v1",
            "stage": training.stage,
            "scenario": evaluation.scenario,
            "replicate": training.replicate,
            "definition_sha256": training.definition_sha256,
            "cell_id": training.cell_id,
            "generation_commitment": training.generation_commitment,
            "topology_sha256": topology_sha256(pstar),
            "oracle": score,
        },
    )
    _create_npz(
        output_directory / "analysis.npz",
        {f"oracle_{name}": value for name, value in arrays.items()},
    )
    _write_complete(
        output_directory,
        (
            "branch_state.npz",
            "manifest.json",
            "oracle/complete.json",
            "analysis.json",
            "analysis.npz",
        ),
    )


def _run_oracle_short_for_test(
    *,
    training_path: Path,
    evaluation_path: Path,
    output_directory: Path,
    conditioning_cycles: int = 1,
    production_cycles: int = 100,
) -> None:
    training = load_training_artifact(training_path)
    _, _, conditioning_seed, sampler_seed, pair_slots = frozen_oracle_settings(
        training.stage,
        training.replicate,
    )
    revision = subprocess.run(
        ("git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _run_oracle(
        training_path=training_path,
        evaluation_path=evaluation_path,
        output_directory=output_directory,
        conditioning_cycles=conditioning_cycles,
        production_cycles=production_cycles,
        conditioning_seed=conditioning_seed,
        sampler_seed=sampler_seed,
        pair_slots=pair_slots,
        source_revision=revision,
        budget_profile="test-short",
        retry_authorization_token_sha256=None,
    )


def command_run_oracle(args: argparse.Namespace) -> None:
    training = load_training_artifact(args.training)
    profile = getattr(args, "profile", "primary")
    (
        conditioning_cycles,
        production_cycles,
        conditioning_seed,
        sampler_seed,
        pair_slots,
    ) = frozen_oracle_settings(training.stage, training.replicate)
    expected_conditioning, expected_production, expected_pair_slots = _frozen_reference_profile_budgets(
        training.stage, profile
    )
    if pair_slots != expected_pair_slots:
        raise RuntimeError("oracle pair-slot settings differ from the frozen profile")
    conditioning_cycles = expected_conditioning
    production_cycles = expected_production
    revision = _current_clean_revision()
    if args.source_revision != revision:
        raise ValueError("--source-revision must equal the current exact full Git SHA")
    retry_digest = _retry_authorization_for_profile(
        profile=profile,
        token_path=getattr(args, "retry_authorization_token", None),
        source_revision=revision,
        definition_sha256=training.definition_sha256,
        stage=training.stage,
    )
    _run_oracle(
        training_path=args.training,
        evaluation_path=args.evaluation,
        output_directory=args.output_directory,
        conditioning_cycles=conditioning_cycles,
        production_cycles=production_cycles,
        conditioning_seed=conditioning_seed,
        sampler_seed=sampler_seed,
        pair_slots=pair_slots,
        source_revision=revision,
        budget_profile=profile,
        retry_authorization_token_sha256=retry_digest,
    )


def _load_trace(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _exact_manifest_int(manifest: Mapping[str, object], name: str) -> int:
    value = manifest[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"manifest {name} must be an exact integer")
    return value


def _validated_trace(
    path: Path,
    *,
    arm: str,
    manifest: Mapping[str, object],
    shape: tuple[int, int],
    k: int,
) -> dict[str, NDArray[Any]]:
    trace = _load_trace(path)
    expected_keys = {
        *TRACE_RETAINED_FIELDS,
        *(f"attempt_{name}" for name in TRACE_ATTEMPT_FIELDS),
        *(f"movement_{name}" for name in MOVEMENT_FIELDS),
        "cycle",
        "chunk_end_cycle",
        "chunk_sampler_seconds",
    }
    if set(trace) != expected_keys:
        raise ValueError(f"{arm} trace fields are incompatible")
    expected_dtypes: dict[str, np.dtype[Any]] = {
        "rectangle_bounds": np.dtype(np.int64),
        "leaf_masses": np.dtype(np.float64),
        "root_total": np.dtype(np.float64),
        "fixed_coefficients": np.dtype(np.float64),
        "log_gaussian_likelihood": np.dtype(np.float64),
        "log_likelihood": np.dtype(np.float64),
        "log_root_prior": np.dtype(np.float64),
        "log_allocation_prior": np.dtype(np.float64),
        "log_structural_prior": np.dtype(np.float64),
        "log_fixed_coefficient_prior": np.dtype(np.float64),
        "log_target": np.dtype(np.float64),
        "state_transition": np.dtype(np.int64),
        "attempt_global_transition": np.dtype(np.int64),
        "attempt_slot": np.dtype("U15"),
        "attempt_move": np.dtype("U24"),
        "attempt_valid": np.dtype(np.bool_),
        "attempt_accepted": np.dtype(np.bool_),
        "attempt_log_acceptance_ratio": np.dtype(np.float64),
        "attempt_invalid_reason": np.dtype("U96"),
        "cycle": np.dtype(np.int64),
        "chunk_end_cycle": np.dtype(np.int64),
        "chunk_sampler_seconds": np.dtype(np.float64),
    }
    movement_integer = (
        "global_transition",
        "proposal_elapsed_ns",
        "diagnostic_elapsed_ns",
        "source_merge_count",
        "destination_catalogue_size",
        "pair_catalogue_size",
        "design_cache_misses",
        "changed_native_cell_count",
        "fixed_position",
        "slice_left_steps",
        "slice_right_steps",
        "slice_shrink_draws",
        "slice_log_density_evaluations",
    )
    movement_float = (
        "changed_nominal_mass",
        "standardized_prediction_l2",
        "root_abs_displacement",
        "root_abs_log_displacement",
        "allocation_share_l1_displacement",
        "fixed_abs_displacement",
        "fixed_abs_log_displacement",
    )
    for name in movement_integer:
        expected_dtypes[f"movement_{name}"] = np.dtype(np.int64)
    for name in movement_float:
        expected_dtypes[f"movement_{name}"] = np.dtype(np.float64)
    expected_dtypes["movement_move"] = np.dtype("U24")
    expected_dtypes["movement_valid"] = np.dtype(np.bool_)
    expected_dtypes["movement_accepted"] = np.dtype(np.bool_)
    if any(trace[name].dtype != dtype for name, dtype in expected_dtypes.items()):
        raise ValueError(f"{arm} trace contains an incompatible exact ndarray dtype")
    cycles = _exact_manifest_int(manifest, "production_cycles")
    pair_slots = _exact_manifest_int(manifest, "pair_slots")
    chunk_cycles = _exact_manifest_int(manifest, "chunk_cycles")
    cycle_length = 1 + pair_slots + (2 if arm == "mobile" else 0)
    if not np.array_equal(trace["cycle"], np.arange(1, cycles + 1, dtype=np.int64)):
        raise ValueError(f"{arm} retained cycle coordinates are inconsistent")
    if not np.array_equal(
        trace["state_transition"],
        cycle_length * np.arange(1, cycles + 1, dtype=np.int64),
    ):
        raise ValueError(f"{arm} retained transition coordinates are inconsistent")
    attempts = cycles * cycle_length
    if not np.array_equal(
        trace["attempt_global_transition"],
        np.arange(1, attempts + 1, dtype=np.int64),
    ):
        raise ValueError(f"{arm} attempt coordinates are inconsistent")
    expected_chunks = np.arange(chunk_cycles, cycles + 1, chunk_cycles, dtype=np.int64)
    if (
        not np.array_equal(trace["chunk_end_cycle"], expected_chunks)
        or trace["chunk_sampler_seconds"].shape != expected_chunks.shape
        or np.any(~np.isfinite(trace["chunk_sampler_seconds"]))
        or np.any(trace["chunk_sampler_seconds"] < 0.0)
    ):
        raise ValueError(f"{arm} chunk timing coordinates are inconsistent")
    bounds = trace["rectangle_bounds"]
    masses = trace["leaf_masses"]
    if (
        bounds.ndim != 3
        or bounds.shape[0] != cycles
        or bounds.shape[1] != k
        or bounds.shape[2] != 4
        or masses.shape != bounds.shape[:2]
        or trace["root_total"].shape != (cycles,)
        or np.any(~np.isfinite(masses))
        or np.any(masses <= 0.0)
        or not np.allclose(
            trace["root_total"],
            np.sum(masses, axis=1),
            rtol=1e-12,
            atol=0.0,
        )
    ):
        raise ValueError(f"{arm} retained scientific coordinates are inconsistent")
    for draw, draw_bounds in enumerate(bounds):
        try:
            retained_tiling = tiling_from_bounds(shape, draw_bounds)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{arm} retained topology at cycle {draw + 1} is invalid") from error
        if retained_tiling.k != k:
            raise ValueError(f"{arm} retained topology has inconsistent K")
    finite_retained = (
        "root_total",
        "fixed_coefficients",
        "log_gaussian_likelihood",
        "log_likelihood",
        "log_root_prior",
        "log_allocation_prior",
        "log_structural_prior",
        "log_fixed_coefficient_prior",
        "log_target",
    )
    if any(
        getattr(trace[name], "shape")[0] != cycles or np.any(~np.isfinite(trace[name]))
        for name in finite_retained
    ):
        raise ValueError(f"{arm} retained target arrays are inconsistent")
    if not np.allclose(
        trace["log_target"],
        trace["log_likelihood"]
        + trace["log_root_prior"]
        + trace["log_allocation_prior"]
        + trace["log_structural_prior"]
        + trace["log_fixed_coefficient_prior"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(f"{arm} retained target decomposition is inconsistent")
    for name in TRACE_ATTEMPT_FIELDS:
        if trace[f"attempt_{name}"].shape != (attempts,):
            raise ValueError(f"{arm} attempt {name} shape is inconsistent")
    if (
        np.any(trace["attempt_accepted"] & ~trace["attempt_valid"])
        or np.any(np.isnan(trace["attempt_log_acceptance_ratio"]))
        or np.any(trace["attempt_log_acceptance_ratio"] == math.inf)
    ):
        raise ValueError(f"{arm} attempt diagnostics are inconsistent")
    for name in ("global_transition", "move", "valid", "accepted"):
        if not np.array_equal(
            trace[f"movement_{name}"],
            trace[f"attempt_{name}"],
        ):
            raise ValueError(f"{arm} movement diagnostics disagree with attempts")
    structural = np.isin(
        trace["attempt_move"],
        ("edge_flip", "resolution_relocation"),
    )
    declared_structural = 2 * cycles if arm == "mobile" else 0
    if int(np.count_nonzero(structural)) != declared_structural:
        raise ValueError(f"{arm} structural opportunity count is inconsistent")
    if arm == "fixed" and np.any(bounds != bounds[0]):
        raise ValueError("fixed-basis trace changed topology")
    return trace


def _log_mean_exp(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(np.mean(np.exp(values - maximum), axis=axis))


def _returned_after_departure(at_origin: NDArray[np.bool_]) -> bool:
    departed = False
    for at_origin_now in at_origin:
        if not bool(at_origin_now):
            departed = True
        elif departed:
            return True
    return False


def _score_trace(
    trace: Mapping[str, NDArray[Any]],
    *,
    nominal_weight: NDArray[np.float64],
    truth: NDArray[np.float64],
    heldout_operator: NDArray[np.float64],
    heldout_noiseless: NDArray[np.float64],
    heldout_observations: NDArray[np.float64],
    heldout_sd: NDArray[np.float64],
    p0_bounds: tuple[tuple[int, int, int, int], ...],
    pstar_bounds: tuple[tuple[int, int, int, int], ...],
) -> tuple[dict[str, object], dict[str, NDArray[Any]]]:
    fields_native = reconstruct_native_fields(
        cast(Any, trace["rectangle_bounds"]),
        cast(Any, trace["leaf_masses"]),
        nominal_weight,
    )
    draws = fields_native.shape[0]
    means = np.cumsum(fields_native, axis=0) / np.arange(1, draws + 1)[:, None, None]
    all_mean = means[-1]
    late_mean = np.mean(fields_native[draws // 2 :], axis=0)
    best_index = int(np.argmax(trace["log_target"]))
    heldout_draw = fields_native.reshape(draws, -1) @ heldout_operator.T

    def heldout_rmse(field: NDArray[np.float64]) -> float:
        prediction = heldout_operator @ field.ravel(order="C")
        return float(np.sqrt(np.mean((prediction - heldout_noiseless) ** 2)))

    def native_rmse(field: NDArray[np.float64]) -> float:
        return float(np.sqrt(np.sum(nominal_weight * (field - truth) ** 2) / float(nominal_weight.sum())))

    normalized = (
        -0.5 * ((heldout_observations[None, :] - heldout_draw) / heldout_sd[None, :]) ** 2
        - np.log(heldout_sd[None, :])
        - 0.5 * math.log(2.0 * math.pi)
    )
    bounds = np.asarray(trace["rectangle_bounds"], dtype=np.int64)
    p0 = np.asarray(p0_bounds, dtype=np.int64)
    pstar = np.asarray(pstar_bounds, dtype=np.int64)
    topology_ids = np.asarray(
        [
            topology_sha256(
                tiling_from_bounds(
                    cast(tuple[int, int], nominal_weight.shape),
                    bounds[index],
                )
            )
            for index in range(draws)
        ]
    )
    unique, counts = np.unique(topology_ids, return_counts=True)
    star_rows = np.all(bounds == pstar[None, :, :], axis=(1, 2))
    p0_rows = np.all(bounds == p0[None, :, :], axis=(1, 2))
    structural = np.isin(
        trace["attempt_move"],
        ("edge_flip", "resolution_relocation"),
    )
    structural_stats: dict[str, object] = {}
    for move in ("edge_flip", "resolution_relocation"):
        rows = trace["attempt_move"] == move
        structural_stats[move] = {
            "attempted": int(np.count_nonzero(rows)),
            "valid": int(np.count_nonzero(rows & trace["attempt_valid"])),
            "accepted": int(np.count_nonzero(rows & trace["attempt_accepted"])),
        }
    result: dict[str, object] = {
        "all_cycle_heldout_rmse": heldout_rmse(all_mean),
        "last_half_heldout_rmse": heldout_rmse(late_mean),
        "final_heldout_rmse": heldout_rmse(fields_native[-1]),
        "best_training_target_heldout_rmse": heldout_rmse(fields_native[best_index]),
        "all_cycle_native_rmse": native_rmse(all_mean),
        "last_half_native_rmse": native_rmse(late_mean),
        "final_native_rmse": native_rmse(fields_native[-1]),
        "best_training_target_native_rmse": native_rmse(fields_native[best_index]),
        "heldout_gaussian_mixture_log_score": float(np.sum(_log_mean_exp(normalized, axis=0))),
        "best_training_target_cycle": best_index + 1,
        "unique_topologies": int(unique.size),
        "topology_residence": {
            str(key): float(count / draws) for key, count in zip(unique, counts, strict=True)
        },
        "p0_residence_fraction": float(np.mean(p0_rows)),
        "pstar_residence_fraction": float(np.mean(star_rows)),
        "pstar_first_hit_cycle": int(np.flatnonzero(star_rows)[0] + 1) if np.any(star_rows) else None,
        "returned_to_p0": _returned_after_departure(p0_rows),
        "structural": structural_stats,
        "structural_attempts": int(np.count_nonzero(structural)),
    }
    arrays: dict[str, NDArray[Any]] = {
        "native_field": fields_native,
        "cumulative_mean": means,
        "cumulative_heldout_rmse": np.sqrt(
            np.mean(
                (means.reshape(draws, -1) @ heldout_operator.T - heldout_noiseless[None, :]) ** 2,
                axis=1,
            )
        ),
        "common_totals": common_native_totals(fields_native, nominal_weight),
        "cumulative_common_totals": common_native_totals(means, nominal_weight),
        "topology_sha256": topology_ids,
    }
    return result, arrays


def _validate_completion_tree(
    run_directory: Path,
    *,
    children: Sequence[str],
) -> None:
    complete = _read_strict_json(run_directory / "complete.json")
    if (
        set(complete) != {"schema", "files"}
        or complete["schema"] != "openghg_inversions.mh_local_search_complete.v1"
        or not isinstance(complete["files"], dict)
    ):
        raise ValueError("run completion schema is incompatible")
    for name, digest in complete["files"].items():
        if file_sha256(run_directory / name) != digest:
            raise ValueError(f"run completion checksum mismatch for {name}")
    for arm in children:
        arm_complete = _read_strict_json(run_directory / arm / "complete.json")
        if (
            set(arm_complete) != {"schema", "files"}
            or arm_complete["schema"] != "openghg_inversions.mh_local_search_complete.v1"
            or not isinstance(arm_complete["files"], dict)
        ):
            raise ValueError(f"{arm} completion schema is incompatible")
        for name, digest in arm_complete["files"].items():
            if file_sha256(run_directory / arm / name) != digest:
                raise ValueError(f"{arm} completion checksum mismatch for {name}")


def _validate_run_completion(run_directory: Path) -> None:
    _validate_completion_tree(run_directory, children=("fixed", "mobile"))


def _validate_oracle_completion(run_directory: Path) -> None:
    _validate_completion_tree(run_directory, children=("oracle",))


def _analyze(
    *,
    training_path: Path,
    evaluation_path: Path,
    run_directory: Path,
    output_directory: Path,
    enforce_frozen_budgets: bool,
) -> None:
    training = load_training_artifact(training_path)
    evaluation = load_evaluation_artifact(evaluation_path)
    validate_artifact_pair(training, evaluation)
    _validate_run_completion(run_directory)
    manifest = _read_strict_json(run_directory / "manifest.json")
    is_retry = manifest.get("budget_profile") == "factor4"
    expected_manifest_keys = _PAIR_RETRY_MANIFEST_KEYS if is_retry else _PAIR_MANIFEST_KEYS
    expected_manifest_schema = (
        "openghg_inversions.mh_local_search_pair_manifest.v2"
        if is_retry
        else "openghg_inversions.mh_local_search_pair_manifest.v1"
    )
    if frozenset(manifest) != expected_manifest_keys or manifest["schema"] != expected_manifest_schema:
        raise ValueError("pair manifest schema is incompatible")
    for digest_name in (
        "definition_sha256",
        "cell_id",
        "generation_commitment",
        "training_sha256",
        "branch_state_fingerprint",
    ):
        digest = manifest[digest_name]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"pair manifest {digest_name} is not a SHA-256 digest")
    conditioning_seconds = manifest["conditioning_sampler_seconds"]
    if (
        isinstance(conditioning_seconds, bool)
        or not isinstance(conditioning_seconds, (int, float))
        or not math.isfinite(conditioning_seconds)
        or conditioning_seconds < 0.0
    ):
        raise ValueError("pair manifest conditioning time is invalid")
    if (
        manifest["training_sha256"] != file_sha256(training_path)
        or manifest["definition_sha256"] != training.definition_sha256
        or manifest["cell_id"] != training.cell_id
        or manifest["generation_commitment"] != training.generation_commitment
        or manifest["stage"] != training.stage
        or manifest["replicate"] != training.replicate
        or manifest["conditioning_arm"] != "fixed_basis"
        or manifest["conditioning_schedule_id"] != FIXED_BASIS_COMPOUND_SCHEDULE_ID
        or manifest["conditioning_seed"] != training.conditioning_seed
        or manifest["fixed_seed"] != training.fixed_seed
        or manifest["mobile_seed"] != training.mobile_seed
    ):
        raise ValueError("run did not use the supplied training artifact")
    source_revision = manifest["source_revision"]
    if not isinstance(source_revision, str) or _FULL_SHA.fullmatch(source_revision) is None:
        raise ValueError("pair manifest source revision is not an exact full Git SHA")
    conditioning_cycles = _exact_manifest_int(manifest, "conditioning_cycles")
    production_cycles = _exact_manifest_int(manifest, "production_cycles")
    pair_slots = _exact_manifest_int(manifest, "pair_slots")
    chunk_cycles = _exact_manifest_int(manifest, "chunk_cycles")
    if chunk_cycles != 100 or pair_slots != 5:
        raise ValueError("pair manifest violates the frozen schedule")
    if enforce_frozen_budgets:
        budget_profile = manifest["budget_profile"]
        if budget_profile not in ("primary", "factor4"):
            raise ValueError(
                "pair manifest violates the frozen stage budgets: budget profile is incompatible"
            )
        expected_conditioning, expected_production, expected_pair_slots = _frozen_reference_profile_budgets(
            training.stage, cast(str, budget_profile)
        )
        if (
            conditioning_cycles,
            production_cycles,
            pair_slots,
        ) != (
            expected_conditioning,
            expected_production,
            expected_pair_slots,
        ):
            raise ValueError("pair manifest violates the frozen stage budgets")
        if is_retry:
            _require_digest(
                manifest["retry_authorization_token_sha256"],
                name="retry_authorization_token_sha256",
            )
    fixed_trace = _validated_trace(
        run_directory / "fixed" / "trace.npz",
        arm="fixed",
        manifest=manifest,
        shape=training.shape,
        k=training.k,
    )
    mobile_trace = _validated_trace(
        run_directory / "mobile" / "trace.npz",
        arm="mobile",
        manifest=manifest,
        shape=training.shape,
        k=training.k,
    )
    p0 = np.asarray(training.p0_bounds, dtype=np.int64)
    if np.any(fixed_trace["rectangle_bounds"] != p0[None, :, :]):
        raise ValueError("fixed trace does not preserve the declared P0")
    score_args = {
        "nominal_weight": training.nominal_weight,
        "truth": evaluation.truth,
        "heldout_operator": evaluation.heldout_operator,
        "heldout_noiseless": evaluation.heldout_noiseless,
        "heldout_observations": evaluation.heldout_observations,
        "heldout_sd": evaluation.heldout_sd,
        "p0_bounds": training.p0_bounds,
        "pstar_bounds": evaluation.pstar_bounds,
    }
    fixed, fixed_arrays = _score_trace(fixed_trace, **score_args)
    mobile, mobile_arrays = _score_trace(mobile_trace, **score_args)
    fixed_curve = fixed_arrays["cumulative_heldout_rmse"]
    mobile_curve = mobile_arrays["cumulative_heldout_rmse"]
    below = np.flatnonzero(mobile_curve < fixed_curve)
    ratio = float(cast(Any, mobile["all_cycle_heldout_rmse"])) / float(
        cast(Any, fixed["all_cycle_heldout_rmse"])
    )
    fixed_chunk = np.cumsum(fixed_trace["chunk_sampler_seconds"])
    mobile_chunk = np.cumsum(mobile_trace["chunk_sampler_seconds"])
    equal_wall_budget = float(min(fixed_chunk[-1], mobile_chunk[-1]))
    fixed_prefix = int(np.searchsorted(fixed_chunk, equal_wall_budget, side="right"))
    mobile_prefix = int(np.searchsorted(mobile_chunk, equal_wall_budget, side="right"))
    fixed_equal_cycles = int(fixed_trace["chunk_end_cycle"][fixed_prefix - 1]) if fixed_prefix else 0
    mobile_equal_cycles = int(mobile_trace["chunk_end_cycle"][mobile_prefix - 1]) if mobile_prefix else 0
    heldout = evaluation.heldout_operator
    heldout_truth = evaluation.heldout_noiseless
    equal_wall_fixed_rmse: float | None = None
    equal_wall_mobile_rmse: float | None = None
    equal_wall_ratio: float | None = None
    if fixed_equal_cycles and mobile_equal_cycles:
        fixed_equal_field = np.mean(
            fixed_arrays["native_field"][:fixed_equal_cycles],
            axis=0,
        )
        mobile_equal_field = np.mean(
            mobile_arrays["native_field"][:mobile_equal_cycles],
            axis=0,
        )
        equal_wall_fixed_rmse = float(
            np.sqrt(np.mean((heldout @ fixed_equal_field.ravel() - heldout_truth) ** 2))
        )
        equal_wall_mobile_rmse = float(
            np.sqrt(np.mean((heldout @ mobile_equal_field.ravel() - heldout_truth) ** 2))
        )
        equal_wall_ratio = equal_wall_mobile_rmse / equal_wall_fixed_rmse
    summary: dict[str, object] = {
        "schema": "openghg_inversions.mh_local_search_analysis.v1",
        "stage": training.stage,
        "scenario": evaluation.scenario,
        "replicate": training.replicate,
        "definition_sha256": training.definition_sha256,
        "cell_id": training.cell_id,
        "generation_commitment": training.generation_commitment,
        "fixed": fixed,
        "mobile": mobile,
        "mobile_over_fixed_primary_rmse": ratio,
        "mobile_over_fixed_native_rmse": float(cast(Any, mobile["all_cycle_native_rmse"]))
        / float(cast(Any, fixed["all_cycle_native_rmse"])),
        "first_cycle_mobile_cumulative_below_fixed": int(below[0] + 1) if below.size else None,
        "equal_wall": {
            "sampler_seconds": equal_wall_budget,
            "fixed_cycles": fixed_equal_cycles,
            "mobile_cycles": mobile_equal_cycles,
            "fixed_heldout_rmse": equal_wall_fixed_rmse,
            "mobile_heldout_rmse": equal_wall_mobile_rmse,
            "mobile_over_fixed_rmse": equal_wall_ratio,
        },
    }
    output_directory.mkdir(parents=True, exist_ok=False)
    _create_json(output_directory / "analysis.json", summary)
    output_arrays: dict[str, NDArray[Any]] = {}
    for arm, arrays in (("fixed", fixed_arrays), ("mobile", mobile_arrays)):
        output_arrays.update({f"{arm}_{name}": value for name, value in arrays.items()})
    _create_npz(output_directory / "analysis.npz", output_arrays)
    _write_complete(output_directory, ("analysis.json", "analysis.npz"))


def _analyze_short_run_for_test(
    *,
    training_path: Path,
    evaluation_path: Path,
    run_directory: Path,
    output_directory: Path,
) -> None:
    _analyze(
        training_path=training_path,
        evaluation_path=evaluation_path,
        run_directory=run_directory,
        output_directory=output_directory,
        enforce_frozen_budgets=False,
    )


def command_analyze(args: argparse.Namespace) -> None:
    _analyze(
        training_path=args.training,
        evaluation_path=args.evaluation,
        run_directory=args.run_directory,
        output_directory=args.output_directory,
        enforce_frozen_budgets=True,
    )


def _load_persisted_branch(
    *,
    training: Any,
    evaluation: Any,
    topology: str,
    branch_run_directory: Path,
    training_sha256: str,
    evaluation_sha256: str,
    source_revision: str,
    expected_budget_profile: str,
    expected_conditioning_cycles: int,
    expected_production_cycles: int,
    expected_retry_authorization_token_sha256: str | None,
) -> tuple[Any, Any, str, str]:
    is_retry = expected_budget_profile == "factor4"
    if topology == "p0":
        _validate_run_completion(branch_run_directory)
        expected_schema = (
            "openghg_inversions.mh_local_search_pair_manifest.v2"
            if is_retry
            else "openghg_inversions.mh_local_search_pair_manifest.v1"
        )
        expected_keys = _PAIR_RETRY_MANIFEST_KEYS if is_retry else _PAIR_MANIFEST_KEYS
        expected_bounds = training.p0_bounds
        expected_conditioning_seed = training.conditioning_seed
    elif topology == "pstar":
        _validate_oracle_completion(branch_run_directory)
        expected_schema = (
            "openghg_inversions.mh_local_search_oracle_manifest.v2"
            if is_retry
            else "openghg_inversions.mh_local_search_oracle_manifest.v1"
        )
        expected_keys = _ORACLE_RETRY_MANIFEST_KEYS if is_retry else _ORACLE_MANIFEST_KEYS
        expected_bounds = evaluation.pstar_bounds
        _, _, expected_conditioning_seed, _, _ = frozen_oracle_settings(
            training.stage,
            training.replicate,
        )
    else:
        raise ValueError("topology must be p0 or pstar")
    manifest = _read_strict_json(branch_run_directory / "manifest.json")
    if frozenset(manifest) != expected_keys or manifest["schema"] != expected_schema:
        raise ValueError("branch-run manifest schema is incompatible")
    if (
        manifest["definition_sha256"] != training.definition_sha256
        or manifest["cell_id"] != training.cell_id
        or manifest["generation_commitment"] != training.generation_commitment
        or manifest["stage"] != training.stage
        or manifest["replicate"] != training.replicate
        or manifest["training_sha256"] != training_sha256
        or manifest["branch_state_fingerprint"] is None
        or manifest["source_revision"] != source_revision
    ):
        raise ValueError("branch-run manifest does not identify the requested cell")
    conditioning_cycles = _exact_manifest_int(manifest, "conditioning_cycles")
    production_cycles = _exact_manifest_int(manifest, "production_cycles")
    pair_slots = _exact_manifest_int(manifest, "pair_slots")
    chunk_cycles = _exact_manifest_int(manifest, "chunk_cycles")
    conditioning_seed = _exact_manifest_int(manifest, "conditioning_seed")
    if (
        manifest["budget_profile"] != expected_budget_profile
        or manifest["conditioning_arm"] != "fixed_basis"
        or manifest["conditioning_schedule_id"] != FIXED_BASIS_COMPOUND_SCHEDULE_ID
        or conditioning_cycles != expected_conditioning_cycles
        or production_cycles != expected_production_cycles
        or pair_slots != 5
        or chunk_cycles != 100
        or conditioning_seed != expected_conditioning_seed
        or manifest.get("retry_authorization_token_sha256") != expected_retry_authorization_token_sha256
    ):
        raise ValueError("branch-run conditioning provenance is incompatible")
    if topology == "pstar" and (
        manifest["evaluation_sha256"] != evaluation_sha256
        or manifest["scenario"] != evaluation.scenario
        or manifest["pstar_sha256"]
        != topology_sha256(tiling_from_bounds(training.shape, evaluation.pstar_bounds))
        or manifest["witness_sha256"] != json_sha256(dict(evaluation.witness))
    ):
        raise ValueError("oracle branch-run manifest does not identify the evaluation artifact")
    branch_path = branch_run_directory / "branch_state.npz"
    with np.load(branch_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "rectangle_bounds",
            "leaf_masses",
            "fixed_coefficients",
            "log_target",
        }:
            raise ValueError("persisted branch fields are incompatible")
        bounds = np.array(archive["rectangle_bounds"], copy=True)
        masses = np.array(archive["leaf_masses"], copy=True)
        fixed = np.array(archive["fixed_coefficients"], copy=True)
        log_target = np.array(archive["log_target"], copy=True)
    if (
        bounds.dtype != np.dtype(np.int64)
        or masses.dtype != np.dtype(np.float64)
        or fixed.dtype != np.dtype(np.float64)
        or log_target.dtype != np.dtype(np.float64)
        or bounds.shape != (training.k, 4)
        or masses.shape != (training.k,)
        or fixed.shape != (0,)
        or log_target.shape != ()
        or np.any(~np.isfinite(masses))
        or np.any(masses <= 0.0)
        or not np.isfinite(log_target.item())
    ):
        raise ValueError("persisted branch arrays are incompatible")
    tiling = tiling_from_bounds(training.shape, bounds)
    if topology_bounds(tiling) != expected_bounds:
        raise ValueError("persisted branch topology differs from the requested topology")
    problem = problem_from_training(training)
    state = build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(tiling, masses),
        fixed_coefficients=fixed,
    )
    fingerprint = full_tiling_state_fingerprint(problem, state)
    if fingerprint != manifest["branch_state_fingerprint"] or float(log_target) != state.log_target:
        raise ValueError("persisted branch does not rebuild its audited state")
    return problem, state, fingerprint, file_sha256(branch_path)


def _run_local_reference(
    *,
    training_path: Path,
    evaluation_path: Path,
    topology: str,
    branch_run_directory: Path,
    output_directory: Path,
    production_cycles: int,
    seeds: tuple[int, int, int, int],
    source_revision: str,
    profile: str,
    expected_conditioning_cycles: int,
    expected_branch_production_cycles: int,
    retry_authorization_token_sha256: str | None,
) -> None:
    if production_cycles % 100 or production_cycles % 20:
        raise ValueError("local-reference production must use complete 100-cycle chunks and 20 equal batches")
    if production_cycles != expected_branch_production_cycles:
        raise ValueError("local-reference and branch production budgets differ")
    if _FULL_SHA.fullmatch(source_revision) is None:
        raise ValueError("source_revision must be an exact lower-case full Git SHA")
    training = load_training_artifact(training_path)
    evaluation = load_evaluation_artifact(evaluation_path)
    validate_artifact_pair(training, evaluation)
    if profile != "test-short":
        expected = _frozen_reference_profile_budgets(training.stage, profile)
        if (
            expected_conditioning_cycles,
            expected_branch_production_cycles,
            5,
        ) != expected:
            raise ValueError("local-reference run does not match its frozen budget profile")
    if (profile == "factor4") != (retry_authorization_token_sha256 is not None):
        raise ValueError("local-reference retry lineage does not match its profile")
    if training.replicate != 0:
        raise ValueError("local-reference runs are frozen to replicate zero")
    if evaluation.scenario == "aligned" and topology != "p0":
        raise ValueError("the aligned cell has only one unique reference topology")
    if seeds != frozen_local_reference_seeds(training.stage):
        raise ValueError("local-reference seeds differ from the frozen catalogue")
    problem, branch, branch_fingerprint, branch_sha256 = _load_persisted_branch(
        training=training,
        evaluation=evaluation,
        topology=topology,
        branch_run_directory=branch_run_directory,
        training_sha256=file_sha256(training_path),
        evaluation_sha256=file_sha256(evaluation_path),
        source_revision=source_revision,
        expected_budget_profile=profile,
        expected_conditioning_cycles=expected_conditioning_cycles,
        expected_production_cycles=expected_branch_production_cycles,
        expected_retry_authorization_token_sha256=retry_authorization_token_sha256,
    )
    expected_tiling = (
        tiling_from_bounds(training.shape, training.p0_bounds)
        if topology == "p0"
        else tiling_from_bounds(training.shape, evaluation.pstar_bounds)
    )
    if topology == "p0":
        branch_conditioning_seed = training.conditioning_seed
    else:
        _, _, branch_conditioning_seed, _, _ = frozen_oracle_settings(
            training.stage,
            training.replicate,
        )
    output_directory.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, object] = {
        "schema": "openghg_inversions.mh_local_search_local_reference_manifest.v1",
        "stage": training.stage,
        "scenario": evaluation.scenario,
        "replicate": training.replicate,
        "definition_sha256": training.definition_sha256,
        "cell_id": training.cell_id,
        "generation_commitment": training.generation_commitment,
        "source_revision": source_revision,
        "training_sha256": file_sha256(training_path),
        "evaluation_sha256": file_sha256(evaluation_path),
        "topology": topology,
        "topology_sha256": topology_sha256(expected_tiling),
        "branch_run_complete_sha256": file_sha256(branch_run_directory / "complete.json"),
        "branch_state_sha256": branch_sha256,
        "branch_state_fingerprint": branch_fingerprint,
        "profile": profile,
        "branch_conditioning_cycles": expected_conditioning_cycles,
        "branch_conditioning_arm": "fixed_basis",
        "branch_conditioning_schedule_id": FIXED_BASIS_COMPOUND_SCHEDULE_ID,
        "branch_conditioning_seed": branch_conditioning_seed,
        "production_cycles": production_cycles,
        "chunk_cycles": 100,
        "pair_slots": 5,
        "seeds": list(seeds),
        "chains": 4,
        "batches_per_chain": 20,
    }
    if retry_authorization_token_sha256 is not None:
        _require_digest(
            retry_authorization_token_sha256,
            name="retry_authorization_token_sha256",
        )
        manifest["schema"] = "openghg_inversions.mh_local_search_local_reference_manifest.v2"
        manifest["retry_authorization_token_sha256"] = retry_authorization_token_sha256
    _create_json(output_directory / "manifest.json", manifest)
    root_total: list[NDArray[Any]] = []
    leaf_masses: list[NDArray[Any]] = []
    common_totals: list[NDArray[Any]] = []
    rectangle_bounds: list[NDArray[Any]] = []
    child_completions: list[str] = []
    for chain_index, seed in enumerate(seeds):
        chain_name = f"chain-{chain_index}"
        _run_arm(
            problem=problem,
            branch_state=branch,
            mode="fixed_basis",
            seed=seed,
            cycles=production_cycles,
            chunk_cycles=100,
            pair_slots=5,
            output_directory=output_directory / chain_name,
            manifest=manifest,
        )
        trace = validate_local_reference_trace(
            output_directory / chain_name / "trace.npz",
            manifest=manifest,
            shape=training.shape,
            k=training.k,
            expected_bounds=np.asarray(
                topology_bounds(expected_tiling),
                dtype=np.int64,
            ),
            problem=problem,
        )
        root_total.append(trace["root_total"])
        leaf_masses.append(trace["leaf_masses"])
        rectangle_bounds.append(trace["rectangle_bounds"])
        fields_native = reconstruct_native_fields(
            cast(Any, trace["rectangle_bounds"]),
            cast(Any, trace["leaf_masses"]),
            training.nominal_weight,
        )
        common_totals.append(common_native_totals(fields_native, training.nominal_weight))
        child_completions.append(f"{chain_name}/complete.json")
    roots = np.stack(root_total)
    masses = np.stack(leaf_masses)
    totals = np.stack(common_totals)
    all_bounds = np.stack(rectangle_bounds)
    if np.any(all_bounds != np.asarray(topology_bounds(expected_tiling))[None, None, :, :]):
        raise RuntimeError("local-reference chain changed its fixed topology")
    batch_size = production_cycles // 20
    batch_totals = totals.reshape(4, 20, batch_size, 9).mean(axis=2)
    diagnostics_arrays: dict[str, NDArray[Any]] = {
        "root_total": roots,
        "leaf_masses": masses,
        "common_totals": totals,
        "batch_common_totals": batch_totals,
        "first_half_common_mean": totals[:, : production_cycles // 2].mean(axis=1),
        "second_half_common_mean": totals[:, production_cycles // 2 :].mean(axis=1),
        "rectangle_bounds": all_bounds,
        "seeds": np.asarray(seeds, dtype=np.int64),
    }
    _create_npz(output_directory / "diagnostics_input.npz", diagnostics_arrays)
    variable_names = [
        "root_total",
        *[f"leaf_mass[{r0}:{r1},{c0}:{c1}]" for r0, r1, c0, c1 in topology_bounds(expected_tiling)],
    ]
    _create_json(
        output_directory / "diagnostics_manifest.json",
        {
            "schema": "openghg_inversions.mh_local_search_local_diagnostics_input.v1",
            "cell_id": training.cell_id,
            "definition_sha256": training.definition_sha256,
            "topology_sha256": topology_sha256(expected_tiling),
            "draws_per_chain": production_cycles,
            "chains": 4,
            "batches_per_chain": 20,
            "parameter_variable_names": variable_names,
            "projection_names": [
                "whole_domain",
                "top_half",
                "bottom_half",
                "left_half",
                "right_half",
                "top_left",
                "top_right",
                "bottom_left",
                "bottom_right",
            ],
        },
    )
    _write_complete(
        output_directory,
        (
            "manifest.json",
            "diagnostics_input.npz",
            "diagnostics_manifest.json",
            *child_completions,
        ),
    )


def _run_local_reference_short_for_test(
    *,
    training_path: Path,
    evaluation_path: Path,
    topology: str,
    branch_run_directory: Path,
    output_directory: Path,
    production_cycles: int = 100,
) -> None:
    training = load_training_artifact(training_path)
    seeds = frozen_local_reference_seeds(training.stage)
    revision = subprocess.run(
        ("git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _run_local_reference(
        training_path=training_path,
        evaluation_path=evaluation_path,
        topology=topology,
        branch_run_directory=branch_run_directory,
        output_directory=output_directory,
        production_cycles=production_cycles,
        seeds=seeds,
        source_revision=revision,
        profile="test-short",
        expected_conditioning_cycles=1,
        expected_branch_production_cycles=production_cycles,
        retry_authorization_token_sha256=None,
    )


def command_run_local_reference(args: argparse.Namespace) -> None:
    training = load_training_artifact(args.training)
    profile = getattr(args, "profile", "primary")
    conditioning_cycles, production_cycles, _ = _frozen_reference_profile_budgets(
        training.stage,
        profile,
    )
    seeds = frozen_local_reference_seeds(training.stage)
    revision = _current_clean_revision()
    if args.source_revision != revision:
        raise ValueError("--source-revision must equal the current exact full Git SHA")
    retry_digest = _retry_authorization_for_profile(
        profile=profile,
        token_path=getattr(args, "retry_authorization_token", None),
        source_revision=revision,
        definition_sha256=training.definition_sha256,
        stage=training.stage,
    )
    _run_local_reference(
        training_path=args.training,
        evaluation_path=args.evaluation,
        topology=args.topology,
        branch_run_directory=args.branch_run_directory,
        output_directory=args.output_directory,
        production_cycles=production_cycles,
        seeds=seeds,
        source_revision=revision,
        profile=profile,
        expected_conditioning_cycles=conditioning_cycles,
        expected_branch_production_cycles=production_cycles,
        retry_authorization_token_sha256=retry_digest,
    )


def command_prepare_reference(args: argparse.Namespace) -> None:
    training = load_training_artifact(args.training)
    evaluation = load_evaluation_artifact(args.evaluation)
    validate_artifact_pair(training, evaluation)
    _, state, data = prepare_fixed_basis_reference(training, evaluation.pstar_bounds)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, NDArray[Any]] = {
        "rectangle_bounds": np.asarray(topology_bounds(state.allocation.tiling), dtype=np.int64),
    }
    for item in fields(data):
        value = getattr(data, item.name)
        if isinstance(value, np.ndarray):
            arrays[item.name] = value
        elif isinstance(value, (int, float)):
            arrays[item.name] = np.asarray(value)
    _create_npz(args.output_directory / "reference_input.npz", arrays)
    _create_json(
        args.output_directory / "manifest.json",
        {
            "schema": "openghg_inversions.mh_local_search_reference_input.v1",
            "training_sha256": file_sha256(args.training),
            "evaluation_sha256": file_sha256(args.evaluation),
            "definition_sha256": training.definition_sha256,
            "cell_id": training.cell_id,
            "generation_commitment": training.generation_commitment,
            "topology_sha256": topology_sha256(state.allocation.tiling),
            "interface": "prepare_fixed_basis_nuts",
            "execution_status": "not_attempted",
        },
    )
    _write_complete(args.output_directory, ("reference_input.npz", "manifest.json"))


def _require_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be an exact lower-case SHA-256")
    return value


def _index_path(index_path: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else index_path.parent / path


def _validate_standalone_completion(path: Path) -> None:
    completion = _read_strict_json(path)
    if (
        frozenset(completion) != {"schema", "files"}
        or completion["schema"] != "openghg_inversions.mh_local_search_complete.v1"
        or not isinstance(completion["files"], dict)
    ):
        raise ValueError(f"{path} is not a compatible completion artifact")
    for name, digest in cast(dict[str, object], completion["files"]).items():
        expected = _require_digest(digest, name=f"{path} file digest")
        child = path.parent / name
        if file_sha256(child) != expected:
            raise ValueError(f"{path} has a child checksum mismatch for {name}")
        if name.endswith("/complete.json"):
            _validate_standalone_completion(child)


def _validate_reference_completion(path: Path) -> None:
    """Validate one raw local or NUTS reference completion catalogue."""
    completion = _read_strict_json(path)
    if completion.get("schema") == "openghg_inversions.mh_local_search_complete.v1":
        _validate_standalone_completion(path)
        return
    if (
        frozenset(completion)
        != {
            "schema",
            "status",
            "checksums_sha256",
            "files",
            "first_failed_gate",
        }
        or completion.get("schema") != "openghg_inversions.mh_local_search_nuts_completion.v1"
        or completion.get("status") != "complete"
        or not isinstance(completion.get("files"), dict)
    ):
        raise ValueError(f"{path} is not a compatible raw reference completion")
    files = cast(dict[str, object], completion["files"])
    if set(files) != {
        "trace.nc",
        "manifest.json",
        "summary.json",
        "checksums.json",
    }:
        raise ValueError(f"{path} NUTS completion file catalogue is incompatible")
    for name, digest in files.items():
        expected = _require_digest(digest, name=f"{path} artifact digest")
        if file_sha256(path.parent / name) != expected:
            raise ValueError(f"{path} has an artifact checksum mismatch for {name}")
    if completion["checksums_sha256"] != files["checksums.json"]:
        raise ValueError(f"{path} checksum catalogue digest is inconsistent")


def _conditional_reference_failure(
    record: Mapping[str, object],
) -> str | None:
    gates = (
        ("divergences", int(cast(Any, record["divergences"])) == 0),
        ("worst_rhat", float(cast(Any, record["worst_rhat_value"])) <= 1.01),
        ("min_bulk_ess", float(cast(Any, record["min_bulk_ess_value"])) >= 200.0),
        ("min_tail_ess", float(cast(Any, record["min_tail_ess_value"])) >= 200.0),
        (
            "local_mcse_over_nuts_sd",
            float(cast(Any, record["worst_local_mcse_sd_value"])) <= 0.05,
        ),
        (
            "half_difference_over_nuts_sd",
            float(cast(Any, record["worst_half_difference_sd_value"])) <= 0.10,
        ),
        (
            "local_vs_nuts_tolerance",
            float(cast(Any, record["worst_local_vs_nuts_tolerance_value"])) <= 1.0,
        ),
    )
    return next((name for name, passed in gates if not passed), None)


def _validate_conditional_reference(
    record: Mapping[str, object],
    *,
    artifact_digests: set[str],
) -> dict[str, object]:
    if frozenset(record) != _CONDITIONAL_REFERENCE_KEYS:
        raise ValueError("conditional-reference certificate keys are incompatible")
    for name in (
        "cell_id",
        "definition_sha256",
        "topology_sha256",
        "nuts_artifact_sha256",
        "local_artifact_sha256",
    ):
        _require_digest(record[name], name=f"conditional reference {name}")
    if (
        record["nuts_artifact_sha256"] not in artifact_digests
        or record["local_artifact_sha256"] not in artifact_digests
    ):
        raise ValueError("conditional reference cites an unindexed raw artifact")
    if record["profile"] not in ("primary", "retry1"):
        raise ValueError("conditional reference profile is incompatible")
    if not isinstance(record["pass"], bool):
        raise ValueError("conditional reference pass must be Boolean")
    divergences = record["divergences"]
    if isinstance(divergences, bool) or not isinstance(divergences, int) or divergences < 0:
        raise ValueError("conditional reference divergences must be a non-negative integer")
    for name in (
        "worst_rhat_variable",
        "min_bulk_ess_variable",
        "min_tail_ess_variable",
        "worst_local_mcse_sd_projection",
        "worst_half_difference_sd_projection",
        "worst_local_vs_nuts_tolerance_projection",
    ):
        if not isinstance(record[name], str) or not record[name]:
            raise ValueError(f"conditional reference {name} must be non-empty")
    for name in (
        "worst_rhat_value",
        "min_bulk_ess_value",
        "min_tail_ess_value",
        "worst_local_mcse_sd_value",
        "worst_half_difference_sd_value",
        "worst_local_vs_nuts_tolerance_value",
    ):
        value = record[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"conditional reference {name} must be finite and non-negative")
    failure = _conditional_reference_failure(record)
    if bool(record["pass"]) != (failure is None) or record["first_failed_gate"] != failure:
        raise ValueError("conditional reference pass/first-failure certification is inconsistent")
    return dict(record)


def _extreme_reference_record(
    records: Sequence[Mapping[str, object]],
    *,
    value_name: str,
    variable_name: str,
    minimum: bool,
) -> dict[str, object]:
    ordered = sorted(
        records,
        key=lambda record: (
            float(cast(Any, record[value_name])),
            str(record[variable_name]),
            str(record["cell_id"]),
            str(record["topology_sha256"]),
        ),
        reverse=not minimum,
    )
    selected = ordered[0]
    return {
        "cell_id": selected["cell_id"],
        "topology_sha256": selected["topology_sha256"],
        "variable": selected[variable_name],
        "value": selected[value_name],
    }


def _score_raw_trace(
    *,
    trace: Mapping[str, NDArray[Any]],
    training: Any,
    evaluation: Any,
) -> dict[str, object]:
    score, _ = _score_trace(
        trace,
        nominal_weight=training.nominal_weight,
        truth=evaluation.truth,
        heldout_operator=evaluation.heldout_operator,
        heldout_noiseless=evaluation.heldout_noiseless,
        heldout_observations=evaluation.heldout_observations,
        heldout_sd=evaluation.heldout_sd,
        p0_bounds=training.p0_bounds,
        pstar_bounds=evaluation.pstar_bounds,
    )
    return score


def _rebuild_equal_wall(
    *,
    fixed_trace: Mapping[str, NDArray[Any]],
    mobile_trace: Mapping[str, NDArray[Any]],
    training: Any,
    evaluation: Any,
) -> dict[str, object]:
    score_args = {
        "nominal_weight": training.nominal_weight,
        "truth": evaluation.truth,
        "heldout_operator": evaluation.heldout_operator,
        "heldout_noiseless": evaluation.heldout_noiseless,
        "heldout_observations": evaluation.heldout_observations,
        "heldout_sd": evaluation.heldout_sd,
        "p0_bounds": training.p0_bounds,
        "pstar_bounds": evaluation.pstar_bounds,
    }
    _, fixed_arrays = _score_trace(fixed_trace, **score_args)
    _, mobile_arrays = _score_trace(mobile_trace, **score_args)
    fixed_chunk = np.cumsum(fixed_trace["chunk_sampler_seconds"])
    mobile_chunk = np.cumsum(mobile_trace["chunk_sampler_seconds"])
    budget = float(min(fixed_chunk[-1], mobile_chunk[-1]))
    fixed_prefix = int(np.searchsorted(fixed_chunk, budget, side="right"))
    mobile_prefix = int(np.searchsorted(mobile_chunk, budget, side="right"))
    fixed_cycles = int(fixed_trace["chunk_end_cycle"][fixed_prefix - 1]) if fixed_prefix else 0
    mobile_cycles = int(mobile_trace["chunk_end_cycle"][mobile_prefix - 1]) if mobile_prefix else 0
    fixed_rmse: float | None = None
    mobile_rmse: float | None = None
    ratio: float | None = None
    if fixed_cycles and mobile_cycles:
        fixed_field = np.mean(fixed_arrays["native_field"][:fixed_cycles], axis=0)
        mobile_field = np.mean(mobile_arrays["native_field"][:mobile_cycles], axis=0)
        heldout = evaluation.heldout_operator
        truth = evaluation.heldout_noiseless
        fixed_rmse = float(np.sqrt(np.mean((heldout @ fixed_field.ravel() - truth) ** 2)))
        mobile_rmse = float(np.sqrt(np.mean((heldout @ mobile_field.ravel() - truth) ** 2)))
        ratio = mobile_rmse / fixed_rmse
    return {
        "sampler_seconds": budget,
        "fixed_cycles": fixed_cycles,
        "mobile_cycles": mobile_cycles,
        "fixed_heldout_rmse": fixed_rmse,
        "mobile_heldout_rmse": mobile_rmse,
        "mobile_over_fixed_rmse": ratio,
    }


def _gate(
    gates: list[dict[str, object]],
    *,
    name: str,
    passed: bool,
    details: Mapping[str, object],
) -> None:
    gates.append({"name": name, "pass": passed, "details": dict(details)})


def _stage_oracle_learnability_passes(
    stage: str,
    values: Mapping[str, object],
) -> bool:
    return int(cast(Any, values["oracle_below_one_count"])) >= 3 and float(
        cast(Any, values["median_oracle_over_fixed_heldout"])
    ) <= (0.80 if stage == "s0" else 0.90)


def _stage_utility_passes(
    stage: str,
    scenario: str,
    values: Mapping[str, object],
) -> bool:
    heldout = cast(Sequence[float], values["mobile_over_fixed_heldout"])
    if scenario == "aligned":
        return float(cast(Any, values["median_mobile_over_fixed_heldout"])) <= 1.10 and (
            stage != "s0" or max(heldout) <= 1.25
        )
    return (
        float(cast(Any, values["median_mobile_over_fixed_heldout"])) <= (0.90 if stage == "s0" else 0.95)
        and int(cast(Any, values["mobile_heldout_below_one_count"])) >= 3
        and float(cast(Any, values["median_mobile_over_fixed_native"])) <= (0.95 if stage == "s0" else 0.98)
        and (stage != "s1" or max(heldout) <= 1.20)
    )


def _validate_s0_retry_promotion(
    *,
    selected_budget_profile: str,
    retry_paths: Mapping[str, Path] | None,
    indexed_retry_token_digest: str | None,
    selected_retry_token_digest: str | None,
    reference_inputs: Mapping[str, tuple[Path, Path]],
    candidate_revision: str,
    stage: str = "s0",
) -> None:
    """Reissue the sealed retry before an aggregate may promote factor4."""
    if stage not in ("s0", "s1"):
        raise ValueError("retry-promotion stage is incompatible")
    if selected_budget_profile != "factor4":
        if (
            retry_paths is not None
            or indexed_retry_token_digest is not None
            or selected_retry_token_digest is not None
        ):
            raise ValueError("primary S0 matrix cannot cite a retry authorization")
        return
    if (
        retry_paths is None
        or indexed_retry_token_digest is None
        or selected_retry_token_digest != indexed_retry_token_digest
    ):
        raise ValueError("factor-four S0 matrix lacks one complete retry authorization")
    primary_record = _read_strict_json(
        retry_paths["primary_certificate"].parent / "conditional_reference.json"
    )
    primary_cell_id = primary_record.get("cell_id")
    if primary_cell_id not in reference_inputs:
        raise ValueError("retry authorization cites an unknown representative cell")
    retry_training_path, retry_evaluation_path = reference_inputs[cast(str, primary_cell_id)]
    replayed_retry_digest = validate_retry_authorization_bundle(
        directory=retry_paths["authorization"].parent,
        training_path=retry_training_path,
        evaluation_path=retry_evaluation_path,
        primary_certificate_directory=retry_paths["primary_certificate"].parent,
        primary_nuts_directory=retry_paths["primary_nuts"].parent,
        primary_local_directory=retry_paths["primary_local"].parent,
        source_revision=candidate_revision,
    )
    if replayed_retry_digest != indexed_retry_token_digest:
        raise ValueError("S0 retry authorization differs from evidence-based replay")


def _aggregate_s0(
    *,
    index_path: Path,
    output_directory: Path,
    enforce_frozen_budgets: bool,
    enforce_current_revision: bool,
    expected_stage: str = "s0",
) -> None:
    index = _read_strict_json(index_path)
    if (
        frozenset(index) not in (_S0_INDEX_KEYS, _S0_RETRY_INDEX_KEYS)
        or index["schema"] != f"openghg_inversions.mh_local_search_{expected_stage}_index.v1"
    ):
        raise ValueError("S0 aggregate index schema is incompatible")
    candidate_revision = index["candidate_revision"]
    if not isinstance(candidate_revision, str) or _FULL_SHA.fullmatch(candidate_revision) is None:
        raise ValueError("S0 candidate revision must be an exact full Git SHA")
    if enforce_current_revision and candidate_revision != _current_clean_revision():
        raise ValueError("S0 candidate revision must equal the current clean exact full Git SHA")
    definition_path = _index_path(
        index_path,
        index["definition_path"],
        name="definition_path",
    )
    if file_sha256(definition_path) != _require_digest(
        index["definition_file_sha256"],
        name="definition_file_sha256",
    ):
        raise ValueError("definition file checksum mismatch")
    definition = validate_stage_definition(read_envelope(definition_path, schema=DEFINITION_SCHEMA))
    if definition["stage"] != expected_stage:
        raise ValueError(f"aggregate requires the frozen {expected_stage.upper()} definition")
    definition_digest = json_sha256(definition)
    retry_paths: dict[str, Path] | None = None
    indexed_retry_token_digest: str | None = None
    retry_raw = index.get("retry_authorization")
    if retry_raw is not None:
        if not isinstance(retry_raw, dict) or frozenset(retry_raw) != _S0_RETRY_AUTHORIZATION_KEYS:
            raise ValueError("S0 retry-authorization index is incompatible")
        retry_paths = {}
        for role in (
            "authorization",
            "primary_certificate",
            "primary_nuts",
            "primary_local",
        ):
            path = _index_path(
                index_path,
                retry_raw[f"{role}_completion_path"],
                name=f"{role}_completion_path",
            )
            digest = _require_digest(
                retry_raw[f"{role}_completion_sha256"],
                name=f"{role}_completion_sha256",
            )
            if path.name != "complete.json" or file_sha256(path) != digest:
                raise ValueError(f"S0 retry-authorization completion mismatch for {role}")
            retry_paths[role] = path
        indexed_retry_token_digest = validate_retry_authorization_token(
            retry_paths["authorization"].parent / "token.json",
            source_revision=candidate_revision,
            definition_sha256=definition_digest,
            stage=expected_stage,
        )
    artifacts_raw = index["reference_artifacts"]
    if not isinstance(artifacts_raw, list) or len(artifacts_raw) != 10:
        raise ValueError("reference_artifacts must contain ten raw completions")
    artifact_paths: dict[str, Path] = {}
    for raw in artifacts_raw:
        if not isinstance(raw, dict) or frozenset(raw) != _REFERENCE_ARTIFACT_KEYS:
            raise ValueError("reference artifact record keys are incompatible")
        path = _index_path(index_path, raw["path"], name="reference artifact path")
        digest = _require_digest(raw["sha256"], name="reference artifact sha256")
        if digest in artifact_paths or file_sha256(path) != digest:
            raise ValueError("reference artifact checksum is missing, duplicate, or mismatched")
        _validate_reference_completion(path)
        artifact_paths[digest] = path
    cells_raw = index["cells"]
    if not isinstance(cells_raw, list) or len(cells_raw) != 12:
        raise ValueError("S0 aggregate index must contain exactly twelve cells")
    cell_results: dict[tuple[str, int], dict[str, object]] = {}
    reference_inputs: dict[str, tuple[Path, Path]] = {}
    reference_branches: dict[str, dict[str, dict[str, str]]] = {}
    selected_budget_profile: str | None = None
    selected_retry_token_digest: str | None = None
    expected_cells = {
        (scenario, replicate)
        for scenario in ("aligned", "edge-one", "relocation-one")
        for replicate in range(4)
    }
    for raw in cells_raw:
        if not isinstance(raw, dict) or frozenset(raw) != _S0_CELL_KEYS:
            raise ValueError("S0 cell index keys are incompatible")
        scenario = raw["scenario"]
        replicate = raw["replicate"]
        if (
            scenario not in ("aligned", "edge-one", "relocation-one")
            or isinstance(replicate, bool)
            or not isinstance(replicate, int)
            or (scenario, replicate) not in expected_cells
            or (scenario, replicate) in cell_results
        ):
            raise ValueError("S0 cell identity is missing, duplicated, or incompatible")
        training_path = _index_path(
            index_path,
            raw["training_path"],
            name="training_path",
        )
        evaluation_path = _index_path(
            index_path,
            raw["evaluation_path"],
            name="evaluation_path",
        )
        if file_sha256(training_path) != _require_digest(
            raw["training_sha256"], name="training_sha256"
        ) or file_sha256(evaluation_path) != _require_digest(
            raw["evaluation_sha256"], name="evaluation_sha256"
        ):
            raise ValueError("S0 cell input checksum mismatch")
        training = load_training_artifact(training_path)
        evaluation = load_evaluation_artifact(evaluation_path)
        validate_artifact_pair(training, evaluation)
        if (
            training.stage != expected_stage
            or evaluation.stage != expected_stage
            or evaluation.scenario != scenario
            or training.replicate != replicate
            or training.definition_sha256 != definition_digest
        ):
            raise ValueError("S0 cell artifact identity differs from its index")
        practical_run = _index_path(
            index_path,
            raw["practical_run_directory"],
            name="practical_run_directory",
        )
        practical_analysis = _index_path(
            index_path,
            raw["practical_analysis_directory"],
            name="practical_analysis_directory",
        )
        oracle_run = _index_path(
            index_path,
            raw["oracle_run_directory"],
            name="oracle_run_directory",
        )
        if (
            file_sha256(practical_run / "complete.json")
            != _require_digest(
                raw["practical_complete_sha256"],
                name="practical_complete_sha256",
            )
            or file_sha256(practical_analysis / "complete.json")
            != _require_digest(
                raw["practical_analysis_complete_sha256"],
                name="practical_analysis_complete_sha256",
            )
            or file_sha256(oracle_run / "complete.json")
            != _require_digest(
                raw["oracle_complete_sha256"],
                name="oracle_complete_sha256",
            )
        ):
            raise ValueError("S0 run/analysis completion checksum mismatch")
        _validate_run_completion(practical_run)
        _validate_completion_tree(practical_analysis, children=())
        _validate_oracle_completion(oracle_run)
        practical_manifest = _read_strict_json(practical_run / "manifest.json")
        oracle_manifest = _read_strict_json(oracle_run / "manifest.json")
        practical_budget_profile = practical_manifest.get("budget_profile")
        oracle_budget_profile = oracle_manifest.get("budget_profile")
        permitted_budget_profiles = ("primary", "factor4") if enforce_frozen_budgets else ("test-short",)
        factor4 = practical_budget_profile == "factor4"
        expected_pair_keys = _PAIR_RETRY_MANIFEST_KEYS if factor4 else _PAIR_MANIFEST_KEYS
        expected_pair_schema = (
            "openghg_inversions.mh_local_search_pair_manifest.v2"
            if factor4
            else "openghg_inversions.mh_local_search_pair_manifest.v1"
        )
        expected_oracle_keys = _ORACLE_RETRY_MANIFEST_KEYS if factor4 else _ORACLE_MANIFEST_KEYS
        expected_oracle_schema = (
            "openghg_inversions.mh_local_search_oracle_manifest.v2"
            if factor4
            else "openghg_inversions.mh_local_search_oracle_manifest.v1"
        )
        if (
            frozenset(practical_manifest) != expected_pair_keys
            or practical_manifest["schema"] != expected_pair_schema
            or practical_manifest["source_revision"] != candidate_revision
            or practical_manifest["cell_id"] != training.cell_id
            or practical_manifest["conditioning_seed"] != training.conditioning_seed
            or practical_manifest["fixed_seed"] != training.fixed_seed
            or practical_manifest["mobile_seed"] != training.mobile_seed
            or frozenset(oracle_manifest) != expected_oracle_keys
            or oracle_manifest["schema"] != expected_oracle_schema
            or oracle_manifest["source_revision"] != candidate_revision
            or oracle_manifest["cell_id"] != training.cell_id
            or practical_budget_profile not in permitted_budget_profiles
            or oracle_budget_profile != practical_budget_profile
            or practical_manifest["conditioning_arm"] != "fixed_basis"
            or oracle_manifest["conditioning_arm"] != "fixed_basis"
            or practical_manifest["conditioning_schedule_id"] != FIXED_BASIS_COMPOUND_SCHEDULE_ID
            or oracle_manifest["conditioning_schedule_id"] != FIXED_BASIS_COMPOUND_SCHEDULE_ID
        ):
            raise ValueError("S0 run manifest provenance is incompatible")
        if selected_budget_profile is None:
            selected_budget_profile = cast(str, practical_budget_profile)
        elif practical_budget_profile != selected_budget_profile:
            raise ValueError("S0 cell matrix mixes primary and factor-four budgets")
        practical_retry_digest = practical_manifest.get("retry_authorization_token_sha256")
        oracle_retry_digest = oracle_manifest.get("retry_authorization_token_sha256")
        if (
            practical_retry_digest != oracle_retry_digest
            or practical_retry_digest != indexed_retry_token_digest
            or factor4 != (practical_retry_digest is not None)
        ):
            raise ValueError("S0 run matrix is not bound to one retry-authorization token")
        if selected_retry_token_digest is None:
            selected_retry_token_digest = cast(str | None, practical_retry_digest)
        elif practical_retry_digest != selected_retry_token_digest:
            raise ValueError("S0 run matrix mixes retry-authorization tokens")
        (
            _,
            _,
            expected_oracle_conditioning_seed,
            expected_oracle_seed,
            _,
        ) = frozen_oracle_settings(training.stage, training.replicate)
        if (
            oracle_manifest["conditioning_seed"] != expected_oracle_conditioning_seed
            or oracle_manifest["oracle_seed"] != expected_oracle_seed
        ):
            raise ValueError("S0 oracle manifest seed provenance is incompatible")
        if enforce_frozen_budgets:
            expected_conditioning, expected_production, expected_pair_slots = (
                _frozen_reference_profile_budgets(
                    training.stage,
                    cast(str, practical_budget_profile),
                )
            )
            if (
                practical_manifest["conditioning_cycles"] != expected_conditioning
                or practical_manifest["production_cycles"] != expected_production
                or practical_manifest["pair_slots"] != expected_pair_slots
                or practical_manifest["chunk_cycles"] != 100
                or oracle_manifest["conditioning_cycles"] != expected_conditioning
                or oracle_manifest["production_cycles"] != expected_production
                or oracle_manifest["pair_slots"] != expected_pair_slots
                or oracle_manifest["chunk_cycles"] != 100
            ):
                raise ValueError("S0 raw run violates the frozen scientific budgets")
        practical_analysis_json = _read_strict_json(practical_analysis / "analysis.json")
        oracle_analysis_json = _read_strict_json(oracle_run / "analysis.json")
        if (
            practical_analysis_json.get("schema") != "openghg_inversions.mh_local_search_analysis.v1"
            or practical_analysis_json.get("cell_id") != training.cell_id
            or oracle_analysis_json.get("schema") != "openghg_inversions.mh_local_search_oracle_analysis.v1"
            or oracle_analysis_json.get("cell_id") != training.cell_id
        ):
            raise ValueError("S0 analysis identity is incompatible")
        fixed_trace = _validated_trace(
            practical_run / "fixed" / "trace.npz",
            arm="fixed",
            manifest=practical_manifest,
            shape=training.shape,
            k=training.k,
        )
        mobile_trace = _validated_trace(
            practical_run / "mobile" / "trace.npz",
            arm="mobile",
            manifest=practical_manifest,
            shape=training.shape,
            k=training.k,
        )
        oracle_trace = _validated_trace(
            oracle_run / "oracle" / "trace.npz",
            arm="fixed",
            manifest=oracle_manifest,
            shape=training.shape,
            k=training.k,
        )
        fixed_score = _score_raw_trace(
            trace=fixed_trace,
            training=training,
            evaluation=evaluation,
        )
        mobile_score = _score_raw_trace(
            trace=mobile_trace,
            training=training,
            evaluation=evaluation,
        )
        oracle_score = _score_raw_trace(
            trace=oracle_trace,
            training=training,
            evaluation=evaluation,
        )
        equal_wall = _rebuild_equal_wall(
            fixed_trace=fixed_trace,
            mobile_trace=mobile_trace,
            training=training,
            evaluation=evaluation,
        )
        if canonical_json(practical_analysis_json["equal_wall"]) != canonical_json(equal_wall):
            raise ValueError("equal-wall analysis does not rebuild from raw traces")
        cell_results[(cast(str, scenario), cast(int, replicate))] = {
            "cell_id": training.cell_id,
            "p0_sha256": topology_sha256(tiling_from_bounds(training.shape, training.p0_bounds)),
            "pstar_sha256": topology_sha256(tiling_from_bounds(training.shape, evaluation.pstar_bounds)),
            "fixed": fixed_score,
            "mobile": mobile_score,
            "oracle": oracle_score,
            "equal_wall": equal_wall,
        }
        if replicate == 0:
            reference_inputs[training.cell_id] = (training_path, evaluation_path)
            reference_branches[training.cell_id] = {
                "p0": {
                    "complete_sha256": file_sha256(practical_run / "complete.json"),
                    "state_sha256": file_sha256(practical_run / "branch_state.npz"),
                    "state_fingerprint": cast(
                        str,
                        practical_manifest["branch_state_fingerprint"],
                    ),
                },
                "pstar": {
                    "complete_sha256": file_sha256(oracle_run / "complete.json"),
                    "state_sha256": file_sha256(oracle_run / "branch_state.npz"),
                    "state_fingerprint": cast(
                        str,
                        oracle_manifest["branch_state_fingerprint"],
                    ),
                },
            }
    if set(cell_results) != expected_cells:
        raise ValueError("S0 aggregate cell matrix is incomplete")
    _validate_s0_retry_promotion(
        selected_budget_profile=cast(str, selected_budget_profile),
        retry_paths=retry_paths,
        indexed_retry_token_digest=indexed_retry_token_digest,
        selected_retry_token_digest=selected_retry_token_digest,
        reference_inputs=reference_inputs,
        candidate_revision=candidate_revision,
        stage=expected_stage,
    )
    references_raw = index["conditional_references"]
    if not isinstance(references_raw, list) or len(references_raw) != 5:
        raise ValueError("S0 aggregate requires exactly five conditional references")
    references: list[dict[str, object]] = []
    reference_audits: list[dict[str, object]] = []
    for raw in references_raw:
        if not isinstance(raw, dict):
            raise ValueError("conditional-reference entries must be objects")
        record = _validate_conditional_reference(
            raw,
            artifact_digests=set(artifact_paths),
        )
        cell_id = cast(str, record["cell_id"])
        if cell_id not in reference_inputs:
            raise ValueError("conditional reference cites an unknown representative cell")
        training_path, evaluation_path = reference_inputs[cell_id]
        nuts_completion = artifact_paths[cast(str, record["nuts_artifact_sha256"])]
        local_completion = artifact_paths[cast(str, record["local_artifact_sha256"])]
        if nuts_completion.name != "complete.json" or local_completion.name != "complete.json":
            raise ValueError("raw reference artifacts must identify complete.json files")
        replayed = validate_conditional_reference_record(
            record,
            training_path=training_path,
            evaluation_path=evaluation_path,
            nuts_directory=nuts_completion.parent,
            local_directory=local_completion.parent,
        )
        if replayed.audit["source_revision"] != candidate_revision:
            raise ValueError("conditional reference source revision differs from the candidate")
        topology_audit = replayed.audit.get("topology")
        local_audit = replayed.audit.get("local")
        if not isinstance(topology_audit, Mapping) or not isinstance(local_audit, Mapping):
            raise ValueError("conditional reference replay audit is incomplete")
        topology_role = topology_audit.get("role")
        if topology_role not in ("p0", "pstar"):
            raise ValueError("conditional reference replay topology role is incompatible")
        expected_branch = reference_branches[cell_id][cast(str, topology_role)]
        if (
            local_audit.get("profile") != selected_budget_profile
            or local_audit.get("retry_authorization_token_sha256") != selected_retry_token_digest
            or local_audit.get("branch_run_complete_sha256") != expected_branch["complete_sha256"]
            or local_audit.get("branch_state_sha256") != expected_branch["state_sha256"]
            or local_audit.get("branch_state_fingerprint") != expected_branch["state_fingerprint"]
        ):
            raise ValueError("conditional reference local chain is not tied to the indexed branch")
        references.append(dict(replayed.record))
        reference_audits.append(dict(replayed.audit))
    cited_artifacts = [
        cast(str, record[name])
        for record in references
        for name in ("nuts_artifact_sha256", "local_artifact_sha256")
    ]
    if len(set(cited_artifacts)) != 10 or set(cited_artifacts) != set(artifact_paths):
        raise ValueError("conditional references must cite each indexed raw artifact exactly once")
    required_reference_keys = {
        (
            cast(str, cell_results[(scenario, 0)]["cell_id"]),
            cast(
                str,
                cell_results[(scenario, 0)]["p0_sha256" if topology == "p0" else "pstar_sha256"],
            ),
        )
        for scenario, topology in (
            ("aligned", "p0"),
            ("edge-one", "p0"),
            ("edge-one", "pstar"),
            ("relocation-one", "p0"),
            ("relocation-one", "pstar"),
        )
    }
    actual_reference_keys = {
        (cast(str, record["cell_id"]), cast(str, record["topology_sha256"])) for record in references
    }
    if actual_reference_keys != required_reference_keys or any(
        record["definition_sha256"] != definition_digest for record in references
    ):
        raise ValueError("conditional references do not cover the five frozen cells/topologies")
    failed_reference_records = [record for record in references if not bool(record["pass"])]
    if failed_reference_records:
        first = failed_reference_records[0]
        raise ValueError(
            "conditional reference failed for "
            f"{first['cell_id']}/{first['topology_sha256']}: "
            f"{first['first_failed_gate']}"
        )
    gates: list[dict[str, object]] = []
    _gate(gates, name="artifacts_complete_finite", passed=True, details={"cells": 12})
    invalid_validity: list[str] = []
    invalid_acceptance: list[str] = []
    for (scenario, replicate), result in sorted(cell_results.items()):
        structural = cast(Mapping[str, object], cast(Mapping[str, object], result["mobile"])["structural"])
        if any(
            int(cast(Any, cast(Mapping[str, object], structural[move])["valid"])) < 1
            for move in ("edge_flip", "resolution_relocation")
        ):
            invalid_validity.append(f"{scenario}/replicate-{replicate}")
        accepted = sum(
            int(cast(Any, cast(Mapping[str, object], structural[move])["accepted"]))
            for move in ("edge_flip", "resolution_relocation")
        )
        if scenario != "aligned" and accepted < 1:
            invalid_acceptance.append(f"{scenario}/replicate-{replicate}")
    _gate(
        gates,
        name="mobile_valid_each_structural_move",
        passed=not invalid_validity,
        details={"failed_cells": invalid_validity},
    )
    _gate(
        gates,
        name="misaligned_accepts_structural_move",
        passed=not invalid_acceptance,
        details={"failed_cells": invalid_acceptance},
    )
    hit_counts = {
        scenario: sum(
            cast(Mapping[str, object], cell_results[(scenario, replicate)]["mobile"])["pstar_first_hit_cycle"]
            is not None
            for replicate in range(4)
        )
        for scenario in ("edge-one", "relocation-one")
    }
    if expected_stage == "s0":
        _gate(
            gates,
            name="one_move_pstar_visits",
            passed=all(count >= 3 for count in hit_counts.values()),
            details=hit_counts,
        )
    _gate(
        gates,
        name="conditional_references",
        passed=True,
        details={"certificates": 5},
    )
    scenario_results: dict[str, object] = {}
    for scenario in ("aligned", "edge-one", "relocation-one"):
        heldout_ratios: list[float] = []
        native_ratios: list[float] = []
        oracle_ratios: list[float] = []
        equal_wall_ratios: list[float] = []
        for replicate in range(4):
            result = cell_results[(scenario, replicate)]
            fixed = cast(Mapping[str, object], result["fixed"])
            mobile = cast(Mapping[str, object], result["mobile"])
            oracle = cast(Mapping[str, object], result["oracle"])
            heldout_ratios.append(
                float(cast(Any, mobile["all_cycle_heldout_rmse"]))
                / float(cast(Any, fixed["all_cycle_heldout_rmse"]))
            )
            native_ratios.append(
                float(cast(Any, mobile["all_cycle_native_rmse"]))
                / float(cast(Any, fixed["all_cycle_native_rmse"]))
            )
            oracle_ratios.append(
                float(cast(Any, oracle["all_cycle_heldout_rmse"]))
                / float(cast(Any, fixed["all_cycle_heldout_rmse"]))
            )
            equal_wall = cast(Mapping[str, object], result["equal_wall"])
            equal_wall_ratio = equal_wall["mobile_over_fixed_rmse"]
            if (
                expected_stage == "s1"
                and enforce_frozen_budgets
                and (not isinstance(equal_wall_ratio, (int, float)) or isinstance(equal_wall_ratio, bool))
            ):
                raise ValueError("equal-wall comparison is incomplete")
            if isinstance(equal_wall_ratio, (int, float)) and not isinstance(equal_wall_ratio, bool):
                equal_wall_ratios.append(float(equal_wall_ratio))
        scenario_results[scenario] = {
            "mobile_over_fixed_heldout": heldout_ratios,
            "mobile_over_fixed_native": native_ratios,
            "oracle_over_fixed_heldout": oracle_ratios,
            "mobile_over_fixed_equal_wall_heldout": equal_wall_ratios,
            "median_mobile_over_fixed_heldout": float(np.median(heldout_ratios)),
            "median_mobile_over_fixed_native": float(np.median(native_ratios)),
            "median_oracle_over_fixed_heldout": float(np.median(oracle_ratios)),
            "median_mobile_over_fixed_equal_wall_heldout": (
                float(np.median(equal_wall_ratios)) if equal_wall_ratios else None
            ),
            "mobile_heldout_below_one_count": sum(value < 1.0 for value in heldout_ratios),
            "oracle_below_one_count": sum(value < 1.0 for value in oracle_ratios),
        }
    for scenario in ("edge-one", "relocation-one"):
        values = cast(Mapping[str, object], scenario_results[scenario])
        _gate(
            gates,
            name=f"oracle_learnability_{scenario}",
            passed=_stage_oracle_learnability_passes(expected_stage, values),
            details=values,
        )
    aligned = cast(Mapping[str, object], scenario_results["aligned"])
    _gate(
        gates,
        name="utility_aligned",
        passed=_stage_utility_passes(expected_stage, "aligned", aligned),
        details=aligned,
    )
    for scenario in ("edge-one", "relocation-one"):
        values = cast(Mapping[str, object], scenario_results[scenario])
        _gate(
            gates,
            name=f"utility_{scenario}",
            passed=_stage_utility_passes(expected_stage, scenario, values),
            details=values,
        )
    gates_by_name = {cast(str, gate["name"]): bool(gate["pass"]) for gate in gates}
    for scenario in ("aligned", "edge-one", "relocation-one"):
        scenario_cell_prefix = f"{scenario}/"
        reference_cell_id = cell_results[(scenario, 0)]["cell_id"]
        scenario_gate_results: list[tuple[str, bool]] = [
            ("artifacts_complete_finite", True),
            (
                "mobile_valid_each_structural_move",
                not any(cell.startswith(scenario_cell_prefix) for cell in invalid_validity),
            ),
        ]
        if scenario != "aligned":
            scenario_gate_results.extend(
                (
                    (
                        "misaligned_accepts_structural_move",
                        not any(cell.startswith(scenario_cell_prefix) for cell in invalid_acceptance),
                    ),
                    *(
                        (("one_move_pstar_visits", hit_counts[scenario] >= 3),)
                        if expected_stage == "s0"
                        else ()
                    ),
                )
            )
        scenario_gate_results.append(
            (
                "conditional_references",
                all(bool(record["pass"]) for record in references if record["cell_id"] == reference_cell_id),
            )
        )
        if scenario != "aligned":
            scenario_gate_results.append(
                (
                    f"oracle_learnability_{scenario}",
                    gates_by_name[f"oracle_learnability_{scenario}"],
                )
            )
        scenario_gate_results.append((f"utility_{scenario}", gates_by_name[f"utility_{scenario}"]))
        scenario_first_failed = next(
            (name for name, passed in scenario_gate_results if not passed),
            None,
        )
        cast(dict[str, object], scenario_results[scenario]).update(
            {
                "pass": scenario_first_failed is None,
                "first_failed_gate": scenario_first_failed,
            }
        )
    first_failed = next(
        (cast(str, gate["name"]) for gate in gates if not bool(gate["pass"])),
        None,
    )
    extrema = {
        "worst_rhat": _extreme_reference_record(
            references,
            value_name="worst_rhat_value",
            variable_name="worst_rhat_variable",
            minimum=False,
        ),
        "minimum_bulk_ess": _extreme_reference_record(
            references,
            value_name="min_bulk_ess_value",
            variable_name="min_bulk_ess_variable",
            minimum=True,
        ),
        "minimum_tail_ess": _extreme_reference_record(
            references,
            value_name="min_tail_ess_value",
            variable_name="min_tail_ess_variable",
            minimum=True,
        ),
        "worst_local_mcse_over_sd": _extreme_reference_record(
            references,
            value_name="worst_local_mcse_sd_value",
            variable_name="worst_local_mcse_sd_projection",
            minimum=False,
        ),
        "worst_half_difference_over_sd": _extreme_reference_record(
            references,
            value_name="worst_half_difference_sd_value",
            variable_name="worst_half_difference_sd_projection",
            minimum=False,
        ),
        "worst_local_vs_nuts_tolerance": _extreme_reference_record(
            references,
            value_name="worst_local_vs_nuts_tolerance_value",
            variable_name="worst_local_vs_nuts_tolerance_projection",
            minimum=False,
        ),
    }
    report: dict[str, object] = {
        "schema": f"openghg_inversions.mh_local_search_{expected_stage}_decision.v1",
        "candidate_revision": candidate_revision,
        "definition_sha256": definition_digest,
        "pass": first_failed is None,
        "first_failed_gate": first_failed,
        "gates": gates,
        "scenario_results": scenario_results,
        "conditional_reference_extrema": extrema,
        "conditional_reference_audits": reference_audits,
        "budget_profile": selected_budget_profile,
        "retry_authorization_token_sha256": selected_retry_token_digest,
        "claim_boundary": (
            "finite-budget paired improvement for the predeclared seeds and "
            "witnessed local planted truths only; no partition-posterior or "
            "convergence claim"
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=False)
    _create_json(output_directory / "decision.json", report)
    lines = [
        f"# {expected_stage.upper()} MH-guided local-search decision",
        "",
        f"- Candidate: `{candidate_revision}`",
        f"- Decision: `{'PASS' if first_failed is None else 'FAIL'}`",
        f"- First failed gate: `{first_failed or 'none'}`",
        "",
        "This is a finite-budget algorithm result. It is not evidence of a "
        "converged partition posterior or valid partition marginalization.",
        "",
    ]
    _create_text(output_directory / "report.md", "\n".join(lines))
    _write_complete(output_directory, ("decision.json", "report.md"))


def command_aggregate_s0(args: argparse.Namespace) -> None:
    _aggregate_s0(
        index_path=args.index,
        output_directory=args.output_directory,
        enforce_frozen_budgets=True,
        enforce_current_revision=True,
    )


def command_aggregate_s1(args: argparse.Namespace) -> None:
    _aggregate_s0(
        index_path=args.index,
        output_directory=args.output_directory,
        enforce_frozen_budgets=True,
        enforce_current_revision=True,
        expected_stage="s1",
    )


def _aggregate_s0_short_for_test(
    *,
    index_path: Path,
    output_directory: Path,
) -> None:
    _aggregate_s0(
        index_path=index_path,
        output_directory=output_directory,
        enforce_frozen_budgets=False,
        enforce_current_revision=False,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)

    define = commands.add_parser("define")
    define.add_argument("--stage", choices=("s0", "s1"), required=True)
    define.add_argument("--output", type=Path, required=True)
    define.set_defaults(function=command_define)

    materialize = commands.add_parser("materialize")
    materialize.add_argument("--definition", type=Path, required=True)
    materialize.add_argument(
        "--scenario",
        choices=("aligned", "edge-one", "relocation-one"),
        required=True,
    )
    materialize.add_argument("--replicate", type=int, choices=range(4), required=True)
    materialize.add_argument("--training-output", type=Path, required=True)
    materialize.add_argument("--evaluation-output", type=Path, required=True)
    materialize.set_defaults(function=command_materialize)

    materialize_training = commands.add_parser("materialize-training")
    materialize_training.add_argument("--definition", type=Path, required=True)
    materialize_training.add_argument(
        "--scenario",
        choices=("aligned", "edge-one", "relocation-one"),
        required=True,
    )
    materialize_training.add_argument("--replicate", type=int, choices=range(4), required=True)
    materialize_training.add_argument("--training-output", type=Path, required=True)
    materialize_training.set_defaults(function=command_materialize_training)

    materialize_evaluation = commands.add_parser("materialize-evaluation")
    materialize_evaluation.add_argument("--definition", type=Path, required=True)
    materialize_evaluation.add_argument("--training", type=Path, required=True)
    materialize_evaluation.add_argument(
        "--scenario",
        choices=("aligned", "edge-one", "relocation-one"),
        required=True,
    )
    materialize_evaluation.add_argument("--replicate", type=int, choices=range(4), required=True)
    materialize_evaluation.add_argument("--evaluation-output", type=Path, required=True)
    materialize_evaluation.set_defaults(function=command_materialize_evaluation)

    run = commands.add_parser("run-pair")
    run.add_argument("--training", type=Path, required=True)
    run.add_argument("--output-directory", type=Path, required=True)
    run.add_argument("--source-revision", required=True)
    run.add_argument(
        "--profile",
        choices=("primary", "factor4"),
        default="primary",
    )
    run.add_argument("--retry-authorization-token", type=Path)
    run.set_defaults(function=command_run_pair)

    oracle = commands.add_parser("run-oracle")
    oracle.add_argument("--training", type=Path, required=True)
    oracle.add_argument("--evaluation", type=Path, required=True)
    oracle.add_argument("--output-directory", type=Path, required=True)
    oracle.add_argument("--source-revision", required=True)
    oracle.add_argument(
        "--profile",
        choices=("primary", "factor4"),
        default="primary",
    )
    oracle.add_argument("--retry-authorization-token", type=Path)
    oracle.set_defaults(function=command_run_oracle)

    local_reference = commands.add_parser("run-local-reference")
    local_reference.add_argument("--training", type=Path, required=True)
    local_reference.add_argument("--evaluation", type=Path, required=True)
    local_reference.add_argument(
        "--topology",
        choices=("p0", "pstar"),
        required=True,
    )
    local_reference.add_argument(
        "--branch-run-directory",
        type=Path,
        required=True,
    )
    local_reference.add_argument("--output-directory", type=Path, required=True)
    local_reference.add_argument("--source-revision", required=True)
    local_reference.add_argument(
        "--profile",
        choices=("primary", "factor4"),
        default="primary",
    )
    local_reference.add_argument("--retry-authorization-token", type=Path)
    local_reference.set_defaults(function=command_run_local_reference)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("--training", type=Path, required=True)
    analyze.add_argument("--evaluation", type=Path, required=True)
    analyze.add_argument("--run-directory", type=Path, required=True)
    analyze.add_argument("--output-directory", type=Path, required=True)
    analyze.set_defaults(function=command_analyze)

    reference = commands.add_parser("prepare-reference")
    reference.add_argument("--training", type=Path, required=True)
    reference.add_argument("--evaluation", type=Path, required=True)
    reference.add_argument("--output-directory", type=Path, required=True)
    reference.set_defaults(function=command_prepare_reference)

    aggregate = commands.add_parser("aggregate-s0")
    aggregate.add_argument("--index", type=Path, required=True)
    aggregate.add_argument("--output-directory", type=Path, required=True)
    aggregate.set_defaults(function=command_aggregate_s0)

    aggregate_s1 = commands.add_parser("aggregate-s1")
    aggregate_s1.add_argument("--index", type=Path, required=True)
    aggregate_s1.add_argument("--output-directory", type=Path, required=True)
    aggregate_s1.set_defaults(function=command_aggregate_s1)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
