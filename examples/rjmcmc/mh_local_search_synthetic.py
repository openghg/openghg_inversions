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
from openghg_inversions.experimental.rjmcmc.mh_local_search_synthetic import (
    DEFINITION_SCHEMA,
    EVALUATION_SCHEMA,
    TRAINING_SCHEMA,
    build_stage_definition,
    canonical_json,
    common_native_totals,
    file_sha256,
    frozen_stage_budgets,
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
        "conditioning_cycles",
        "conditioning_seed",
        "conditioning_sampler_seconds",
        "production_cycles",
        "chunk_cycles",
        "pair_slots",
        "fixed_seed",
        "mobile_seed",
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
    problem = problem_from_training(training)
    initial = state_on_tiling(
        problem,
        tiling_from_bounds(training.shape, training.p0_bounds),
    )
    started = perf_counter()
    result = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(
            iterations=conditioning_cycles * (1 + pair_slots),
            seed=training.conditioning_seed,
            pair_allocation_refresh_slots=pair_slots,
            structure_mode="fixed_basis",
        ),
    )
    elapsed = perf_counter() - started
    branch = result.final_state
    if topology_bounds(branch.allocation.tiling) != training.p0_bounds:
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


def _run_pair(
    *,
    training_path: Path,
    output_directory: Path,
    conditioning_cycles: int,
    production_cycles: int,
    pair_slots: int,
    source_revision: str,
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
    output_directory.mkdir(parents=True, exist_ok=False)
    training = load_training_artifact(training_path)
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
        "conditioning_cycles": conditioning_cycles,
        "conditioning_seed": training.conditioning_seed,
        "conditioning_sampler_seconds": conditioning_seconds,
        "production_cycles": production_cycles,
        "chunk_cycles": chunk_cycles,
        "pair_slots": pair_slots,
        "fixed_seed": training.fixed_seed,
        "mobile_seed": training.mobile_seed,
    }
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
    )


def command_run_pair(args: argparse.Namespace) -> None:
    training = load_training_artifact(args.training)
    conditioning_cycles, production_cycles, pair_slots = frozen_stage_budgets(training.stage)
    revision = _current_clean_revision()
    if args.source_revision != revision:
        raise ValueError("--source-revision must equal the current exact full Git SHA")
    _run_pair(
        training_path=args.training,
        output_directory=args.output_directory,
        conditioning_cycles=conditioning_cycles,
        production_cycles=production_cycles,
        pair_slots=pair_slots,
        source_revision=revision,
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


def _validate_run_completion(run_directory: Path) -> None:
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
    for arm in ("fixed", "mobile"):
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
    if (
        frozenset(manifest) != _PAIR_MANIFEST_KEYS
        or manifest["schema"] != "openghg_inversions.mh_local_search_pair_manifest.v1"
    ):
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
        expected_conditioning, expected_production, expected_pair_slots = frozen_stage_budgets(training.stage)
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

    run = commands.add_parser("run-pair")
    run.add_argument("--training", type=Path, required=True)
    run.add_argument("--output-directory", type=Path, required=True)
    run.add_argument("--source-revision", required=True)
    run.set_defaults(function=command_run_pair)

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
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
