"""Strict P0 local-versus-NUTS conditional-reference certification.

The certifier consumes completed sampling bundles and recomputes every
diagnostic used by the frozen S0 aggregate gate.  The compact ``record`` it
returns has exactly the schema already accepted by the aggregate analyzer;
the separate ``audit`` distinguishes bundle-completion hashes from scientific
trace hashes and records the complete provenance and per-projection arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .full_tiling_compound_sampling import FIXED_BASIS_COMPOUND_SCHEDULE_ID
from .full_tiling_io import (
    full_tiling_state_fingerprint,
    load_full_tiling_checkpoint,
)
from .mh_local_search_nuts_reference import (
    PROJECTION_NAMES,
    SAMPLER_SEED,
    START_SEEDS,
    prepare_s0_nuts_reference,
    reference_profile,
    summarize_reference_trace,
    validate_reference_trace,
)
from .mh_local_search_synthetic import (
    SyntheticEvaluationArtifact,
    SyntheticTrainingArtifact,
    canonical_json,
    common_native_totals,
    file_sha256,
    frozen_local_reference_seeds,
    frozen_oracle_settings,
    frozen_stage_budgets,
    load_evaluation_artifact,
    load_training_artifact,
    problem_from_training,
    reconstruct_native_fields,
    topology_bounds,
    topology_sha256,
    validate_local_reference_trace,
    validate_artifact_pair,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

CONDITIONAL_REFERENCE_KEYS = frozenset(
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
_LOCAL_MANIFEST_KEYS = frozenset(
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
        "topology",
        "topology_sha256",
        "branch_run_complete_sha256",
        "branch_state_sha256",
        "branch_state_fingerprint",
        "profile",
        "branch_conditioning_cycles",
        "branch_conditioning_seed",
        "branch_conditioning_arm",
        "branch_conditioning_schedule_id",
        "production_cycles",
        "chunk_cycles",
        "pair_slots",
        "seeds",
        "chains",
        "batches_per_chain",
    )
)
_LOCAL_RETRY_MANIFEST_KEYS = _LOCAL_MANIFEST_KEYS | {"retry_authorization_token_sha256"}
_NUTS_MANIFEST_KEYS = frozenset(
    (
        "schema",
        "cell_id",
        "definition_sha256",
        "cell_name",
        "scenario",
        "topology_role",
        "topology_sha256",
        "training",
        "evaluation",
        "source_revision",
        "sampler",
        "backend",
        "preflight",
    )
)
_NUTS_RETRY_MANIFEST_KEYS = _NUTS_MANIFEST_KEYS | {
    "retry_source_nuts_completion_sha256",
    "retry_source_first_failed_gate",
}
_NUTS_RETRY_GATES = frozenset(
    (
        "zero_divergences",
        "rank_normalized_rhat",
        "bulk_ess",
        "tail_ess",
    )
)
_NUTS_SAMPLER_KEYS = frozenset(
    (
        "name",
        "profile",
        "chains",
        "chain_method",
        "draws",
        "tune",
        "target_accept",
        "max_tree_depth",
        "dense_mass",
        "sampler_seed",
        "jitter",
        "starts",
    )
)
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_LOCAL_TOP_FILES = frozenset(
    (
        "manifest.json",
        "diagnostics_input.npz",
        "diagnostics_manifest.json",
        "chain-0/complete.json",
        "chain-1/complete.json",
        "chain-2/complete.json",
        "chain-3/complete.json",
    )
)
_LOCAL_CHILD_FILES = frozenset(("trace.npz", "checkpoint.npz", "summary.json"))
_LOCAL_SUMMARY_KEYS = frozenset(
    (
        "schema",
        "arm",
        "cycles",
        "cycle_length",
        "retained_states",
        "atomic_transitions",
        "structural_attempts",
        "valid_structural",
        "accepted_structural",
        "sampler_seconds",
        "final_state_fingerprint",
        "training_sha256",
        "branch_state_fingerprint",
    )
)


@dataclass(frozen=True, slots=True)
class ConditionalReferenceCertificate:
    """Aggregate-compatible gate record plus its detailed recomputation audit."""

    record: Mapping[str, object]
    audit: Mapping[str, object]


def _strict_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {token}")),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{path} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return cast(dict[str, object], value)


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lower-case SHA-256")
    return value


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _require_exact_json(actual: object, expected: object, *, name: str) -> None:
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(f"{name} differs from the recomputed value")


def _validate_generic_completion(
    path: Path,
    *,
    expected_files: frozenset[str],
) -> dict[str, str]:
    completion = _strict_json(path)
    if (
        frozenset(completion) != {"schema", "files"}
        or completion["schema"] != "openghg_inversions.mh_local_search_complete.v1"
        or not isinstance(completion["files"], dict)
        or frozenset(cast(dict[str, object], completion["files"])) != expected_files
    ):
        raise ValueError(f"{path} is not a compatible local completion")
    files: dict[str, str] = {}
    for name, raw_digest in cast(dict[str, object], completion["files"]).items():
        child_name = Path(name) if isinstance(name, str) else Path()
        if not isinstance(name, str) or not name or child_name.is_absolute() or ".." in child_name.parts:
            raise ValueError(f"{path} contains an invalid child path")
        expected = _digest(raw_digest, name=f"{path} child digest")
        child = path.parent / child_name
        if not child.is_file() or child.is_symlink():
            raise ValueError(f"{path} child is not a regular in-bundle file: {name}")
        if file_sha256(child) != expected:
            raise ValueError(f"{path} child checksum mismatch for {name}")
        files[name] = expected
    return files


def _validate_nuts_completion(directory: Path) -> dict[str, str]:
    completion_path = directory / "complete.json"
    completion = _strict_json(completion_path)
    if (
        frozenset(completion)
        != {
            "schema",
            "status",
            "checksums_sha256",
            "files",
            "first_failed_gate",
        }
        or completion["schema"] != "openghg_inversions.mh_local_search_nuts_completion.v1"
        or completion["status"] != "complete"
        or not isinstance(completion["files"], dict)
    ):
        raise ValueError("NUTS completion schema is incompatible")
    files = cast(dict[str, object], completion["files"])
    expected_names = {
        "trace.nc",
        "manifest.json",
        "summary.json",
        "checksums.json",
    }
    if set(files) != expected_names:
        raise ValueError("NUTS completion file catalogue is incompatible")
    validated: dict[str, str] = {}
    for name, raw_digest in files.items():
        expected = _digest(raw_digest, name=f"NUTS {name} digest")
        if file_sha256(directory / name) != expected:
            raise ValueError(f"NUTS completion checksum mismatch for {name}")
        validated[name] = expected
    checksums_digest = _digest(
        completion["checksums_sha256"],
        name="NUTS checksums_sha256",
    )
    if checksums_digest != validated["checksums.json"]:
        raise ValueError("NUTS completion and checksum-file digests disagree")
    checksums = _strict_json(directory / "checksums.json")
    if (
        frozenset(checksums) != {"schema", "files"}
        or checksums["schema"] != "openghg_inversions.mh_local_search_nuts_checksums.v1"
        or not isinstance(checksums["files"], dict)
        or set(cast(dict[str, object], checksums["files"])) != {"trace.nc", "manifest.json", "summary.json"}
    ):
        raise ValueError("NUTS checksum manifest is incompatible")
    for name, raw_digest in cast(dict[str, object], checksums["files"]).items():
        expected = _digest(raw_digest, name=f"NUTS checksums {name}")
        if expected != validated[name]:
            raise ValueError(f"NUTS checksum catalogues disagree for {name}")
    return validated


def _validated_nuts(
    *,
    directory: Path,
    training_path: Path,
    evaluation_path: Path,
    training: SyntheticTrainingArtifact,
    evaluation: SyntheticEvaluationArtifact,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    import arviz as az

    files = _validate_nuts_completion(directory)
    manifest = _strict_json(directory / "manifest.json")
    manifest_schema = manifest.get("schema")
    if manifest_schema == "openghg_inversions.mh_local_search_nuts_manifest.v1":
        expected_manifest_keys = _NUTS_MANIFEST_KEYS
        is_retry_manifest = False
    elif manifest_schema == "openghg_inversions.mh_local_search_nuts_manifest.v2":
        expected_manifest_keys = _NUTS_RETRY_MANIFEST_KEYS
        is_retry_manifest = True
    else:
        raise ValueError("NUTS manifest schema is incompatible")
    if frozenset(manifest) != expected_manifest_keys:
        raise ValueError("NUTS manifest schema is incompatible")
    topology_role = manifest["topology_role"]
    if topology_role not in ("p0", "pstar"):
        raise ValueError("NUTS topology role is incompatible")
    if evaluation.scenario == "aligned" and topology_role != "p0":
        raise ValueError("aligned has only one unique conditional-reference topology")
    if manifest["cell_name"] != f"{manifest['scenario']}-{topology_role}":
        raise ValueError("NUTS cell name and topology role disagree")
    setup = prepare_s0_nuts_reference(
        training,
        evaluation,
        topology_role=cast(Any, topology_role),
    )
    topology_digest = topology_sha256(setup.data.tiling)
    identity = {
        "cell_id": training.cell_id,
        "definition_sha256": training.definition_sha256,
        "scenario": setup.cell_name.rsplit("-", 1)[0],
        "topology_sha256": topology_digest,
    }
    for name, expected in identity.items():
        if manifest[name] != expected:
            raise ValueError(f"NUTS manifest {name} differs from the frozen target")
    for name, path in (("training", training_path), ("evaluation", evaluation_path)):
        value = manifest[name]
        if (
            not isinstance(value, dict)
            or frozenset(value) != {"path", "sha256"}
            or value["sha256"] != file_sha256(path)
        ):
            raise ValueError(f"NUTS manifest {name} identity is incompatible")
    source_revision = manifest["source_revision"]
    if not isinstance(source_revision, str) or _FULL_SHA.fullmatch(source_revision) is None:
        raise ValueError("NUTS source revision is not an exact full SHA")

    sampler = manifest["sampler"]
    if not isinstance(sampler, dict) or frozenset(sampler) != _NUTS_SAMPLER_KEYS:
        raise ValueError("NUTS sampler manifest is incompatible")
    profile = reference_profile(cast(Any, sampler["profile"]))
    if not is_retry_manifest and profile.name != "primary":
        raise ValueError("only the primary NUTS profile may use manifest v1")
    if is_retry_manifest:
        if profile.name != "retry1":
            raise ValueError("only the retry1 NUTS profile may use manifest v2")
        _digest(
            manifest["retry_source_nuts_completion_sha256"],
            name="NUTS retry source completion",
        )
        if manifest["retry_source_first_failed_gate"] not in _NUTS_RETRY_GATES:
            raise ValueError("NUTS retry source failure gate is incompatible")
    expected_sampler = {
        "name": "pymc_numpyro_nuts",
        "chains": 4,
        "chain_method": "vectorized",
        "draws": profile.draws,
        "tune": profile.tune,
        "target_accept": profile.target_accept,
        "max_tree_depth": profile.max_tree_depth,
        "dense_mass": profile.dense_mass,
        "sampler_seed": SAMPLER_SEED,
        "jitter": False,
    }
    for name, expected in expected_sampler.items():
        if sampler[name] != expected:
            raise ValueError(f"NUTS sampler {name} violates the frozen profile")
    starts = sampler["starts"]
    if not isinstance(starts, list) or len(starts) != 4:
        raise ValueError("NUTS sampler must record exactly four starts")
    for recorded, expected_start, seed in zip(
        starts,
        setup.starts,
        START_SEEDS,
        strict=True,
    ):
        expected = {
            "profile": expected_start.profile,
            "seed": seed,
            "root_total": float(expected_start.state.root_total),
            "leaf_share": np.asarray(
                expected_start.state.leaf_masses / expected_start.state.root_total,
                dtype=np.float64,
            ).tolist(),
            "expected_constrained_log_target": float(expected_start.state.log_target),
        }
        _require_exact_json(recorded, expected, name="NUTS constrained start")
    backend = manifest["backend"]
    backend_keys = frozenset(
        (
            "pymc_version",
            "pytensor_version",
            "jax_version",
            "numpyro_version",
            "arviz_version",
            "pytensor_floatX",
            "jax_enable_x64",
            "jax_backend",
        )
    )
    if (
        not isinstance(backend, dict)
        or frozenset(backend) != backend_keys
        or any(
            not isinstance(backend[name], str) or not cast(str, backend[name])
            for name in (
                "pymc_version",
                "pytensor_version",
                "jax_version",
                "numpyro_version",
                "arviz_version",
            )
        )
        or backend.get("pytensor_floatX") != "float64"
        or backend.get("jax_enable_x64") is not True
        or backend.get("jax_backend") != "cpu"
    ):
        raise ValueError("NUTS backend is not certified float64 CPU")
    preflight = manifest["preflight"]
    if not isinstance(preflight, list) or len(preflight) != 4:
        raise ValueError("NUTS manifest must contain four preflight audits")
    preflight_keys = frozenset(
        (
            *backend_keys,
            "model_value_variables_float64",
            "model_value_variable_count",
            "constrained_log_target",
            "expected_log_target",
            "log_target_difference",
            "log_target_absolute_tolerance",
        )
    )
    for audit, start in zip(preflight, setup.starts, strict=True):
        if not isinstance(audit, dict) or frozenset(audit) != preflight_keys:
            raise ValueError("NUTS preflight audit has an incompatible exact schema")
        for name in backend_keys:
            if audit[name] != backend[name]:
                raise ValueError("NUTS preflight runtime provenance differs from the backend")
        variable_count = audit["model_value_variable_count"]
        if (
            audit["model_value_variables_float64"] is not True
            or isinstance(variable_count, bool)
            or not isinstance(variable_count, int)
            or variable_count != 2
        ):
            raise ValueError("NUTS preflight model coordinates are not exactly the certified float64 pair")
        expected_target = float(start.state.log_target)
        try:
            constrained_target = float(cast(Any, audit["constrained_log_target"]))
            recorded_expected = float(cast(Any, audit["expected_log_target"]))
            difference = float(cast(Any, audit["log_target_difference"]))
            tolerance = float(cast(Any, audit["log_target_absolute_tolerance"]))
        except (TypeError, ValueError) as error:
            raise ValueError("NUTS preflight target fields must be finite real scalars") from error
        expected_tolerance = 5.0e-10 * max(1.0, abs(expected_target))
        if (
            not all(
                math.isfinite(value)
                for value in (constrained_target, recorded_expected, difference, tolerance)
            )
            or recorded_expected != expected_target
            or difference != constrained_target - expected_target
            or tolerance != expected_tolerance
            or abs(difference) > tolerance
        ):
            raise ValueError("NUTS preflight target parity failed")

    reopened = az.from_netcdf(directory / "trace.nc")
    try:
        trace_audit = validate_reference_trace(
            reopened,
            data=setup.data,
            expected_draws=profile.draws,
        )
        diagnostics = summarize_reference_trace(
            reopened,
            data=setup.data,
            nominal_weight=training.nominal_weight,
        )
    finally:
        getattr(reopened, "close")()
    summary = _strict_json(directory / "summary.json")
    expected_summary_identity = {
        "schema": "openghg_inversions.mh_local_search_nuts_summary.v1",
        "status": "complete",
        "cell_id": training.cell_id,
        "definition_sha256": training.definition_sha256,
        "topology_sha256": topology_digest,
        "nuts_artifact_sha256": files["trace.nc"],
        "profile": profile.name,
    }
    for name, expected in expected_summary_identity.items():
        if summary.get(name) != expected:
            raise ValueError(f"NUTS summary {name} is incompatible")
    completion = _strict_json(directory / "complete.json")
    if completion["first_failed_gate"] != summary.get("first_failed_gate"):
        raise ValueError("NUTS completion and summary failure gates disagree")
    trace_validation = summary.get("trace_validation")
    if (
        not isinstance(trace_validation, dict)
        or trace_validation.get("in_memory") != trace_audit
        or trace_validation.get("reopened_netcdf") != trace_audit
    ):
        raise ValueError("NUTS summary trace validation does not recompute")
    for name, expected in diagnostics.items():
        _require_exact_json(
            summary.get(name),
            expected,
            name=f"NUTS summary {name}",
        )
    nuts_audit: dict[str, object] = {
        "profile": profile.name,
        "source_revision": source_revision,
        "completion_sha256": file_sha256(directory / "complete.json"),
        "trace_sha256": files["trace.nc"],
        "manifest_sha256": files["manifest.json"],
        "summary_sha256": files["summary.json"],
        "checksums_sha256": files["checksums.json"],
        "sampler_seed": SAMPLER_SEED,
        "start_seeds": list(START_SEEDS),
        "draws_per_chain": profile.draws,
        "warmup_per_chain": profile.tune,
        "target_accept": profile.target_accept,
        "max_tree_depth": profile.max_tree_depth,
        "dense_mass": profile.dense_mass,
        "trace_validation": trace_audit,
    }
    if is_retry_manifest:
        nuts_audit["retry_source_nuts_completion_sha256"] = manifest["retry_source_nuts_completion_sha256"]
        nuts_audit["retry_source_first_failed_gate"] = manifest["retry_source_first_failed_gate"]
    return (
        diagnostics,
        nuts_audit,
        topology_digest,
        cast(str, topology_role),
    )


def _validated_local(
    *,
    directory: Path,
    training_path: Path,
    evaluation_path: Path,
    training: SyntheticTrainingArtifact,
    evaluation: SyntheticEvaluationArtifact,
    topology_digest: str,
    topology_role: str,
    source_revision: str,
    test_short_budget: tuple[int, int] | None,
) -> tuple[FloatArray, FloatArray, dict[str, object]]:
    completion_path = directory / "complete.json"
    files = _validate_generic_completion(
        completion_path,
        expected_files=_LOCAL_TOP_FILES,
    )
    manifest = _strict_json(directory / "manifest.json")
    manifest_schema = manifest.get("schema")
    if manifest_schema == "openghg_inversions.mh_local_search_local_reference_manifest.v1":
        expected_manifest_keys = _LOCAL_MANIFEST_KEYS
    elif manifest_schema == "openghg_inversions.mh_local_search_local_reference_manifest.v2":
        expected_manifest_keys = _LOCAL_RETRY_MANIFEST_KEYS
    else:
        raise ValueError("local-reference manifest schema is incompatible")
    if frozenset(manifest) != expected_manifest_keys:
        raise ValueError("local-reference manifest schema is incompatible")
    conditioning_cycles, base_production_cycles, pair_slots = frozen_stage_budgets("s0")
    local_profile = manifest["profile"]
    if local_profile == "primary":
        factor = 1
        expected_conditioning_cycles = conditioning_cycles
        production_cycles = base_production_cycles
    elif local_profile == "factor4":
        factor = 4
        expected_conditioning_cycles = conditioning_cycles * factor
        production_cycles = base_production_cycles * factor
    elif local_profile == "test-short" and test_short_budget is not None:
        expected_conditioning_cycles = _exact_int(
            test_short_budget[0],
            name="test conditioning cycles",
            minimum=1,
        )
        production_cycles = _exact_int(
            test_short_budget[1],
            name="test production cycles",
            minimum=100,
        )
        if production_cycles % 100:
            raise ValueError("test production cycles must contain complete 100-cycle chunks")
    else:
        raise ValueError("local-reference profile must be primary or factor4")
    if (local_profile == "factor4" and not cast(str, manifest_schema).endswith(".v2")) or (
        local_profile != "factor4" and not cast(str, manifest_schema).endswith(".v1")
    ):
        raise ValueError("local-reference profile and manifest schema disagree")
    retry_authorization_token_sha256: str | None = None
    if local_profile == "factor4":
        retry_authorization_token_sha256 = _digest(
            manifest["retry_authorization_token_sha256"],
            name="local-reference retry authorization token",
        )
    expected_conditioning_seed = (
        training.conditioning_seed if topology_role == "p0" else frozen_oracle_settings("s0", 0)[2]
    )
    seeds = frozen_local_reference_seeds("s0")
    expected_identity = {
        "stage": "s0",
        "scenario": evaluation.scenario,
        "replicate": 0,
        "definition_sha256": training.definition_sha256,
        "cell_id": training.cell_id,
        "generation_commitment": training.generation_commitment,
        "source_revision": source_revision,
        "training_sha256": file_sha256(training_path),
        "evaluation_sha256": file_sha256(evaluation_path),
        "topology": topology_role,
        "topology_sha256": topology_digest,
        "profile": local_profile,
        "branch_conditioning_cycles": expected_conditioning_cycles,
        "branch_conditioning_seed": expected_conditioning_seed,
        "branch_conditioning_arm": "fixed_basis",
        "branch_conditioning_schedule_id": FIXED_BASIS_COMPOUND_SCHEDULE_ID,
        "production_cycles": production_cycles,
        "chunk_cycles": 100,
        "pair_slots": pair_slots,
        "seeds": list(seeds),
        "chains": 4,
        "batches_per_chain": 20,
    }
    for name, expected in expected_identity.items():
        if manifest[name] != expected:
            raise ValueError(f"local-reference manifest {name} violates the frozen plan")
    for name in (
        "branch_run_complete_sha256",
        "branch_state_sha256",
        "branch_state_fingerprint",
    ):
        _digest(manifest[name], name=f"local-reference {name}")
    expected_topology_bounds = training.p0_bounds if topology_role == "p0" else evaluation.pstar_bounds
    expected_bounds = np.asarray(expected_topology_bounds, dtype=np.int64)
    problem = problem_from_training(training)
    all_masses: list[FloatArray] = []
    all_roots: list[FloatArray] = []
    all_totals: list[FloatArray] = []
    chain_audits: list[dict[str, object]] = []
    cycle_length = 1 + pair_slots
    checkpoint_manifest = {
        **manifest,
        "arm": "fixed_basis",
        "cycle_length": cycle_length,
    }
    for chain in range(4):
        completion_name = f"chain-{chain}/complete.json"
        child_directory = directory / f"chain-{chain}"
        child_files = _validate_generic_completion(
            child_directory / "complete.json",
            expected_files=_LOCAL_CHILD_FILES,
        )
        if files[completion_name] != file_sha256(child_directory / "complete.json"):
            raise ValueError("local-reference parent and child completion catalogues disagree")
        trace = validate_local_reference_trace(
            child_directory / "trace.npz",
            manifest=manifest,
            shape=training.shape,
            k=training.k,
            expected_bounds=expected_bounds,
            problem=problem,
        )
        summary = _strict_json(child_directory / "summary.json")
        if frozenset(summary) != _LOCAL_SUMMARY_KEYS:
            raise ValueError("local-reference arm summary has an incompatible exact schema")
        expected_summary = {
            "schema": "openghg_inversions.mh_local_search_arm_summary.v1",
            "arm": "fixed_basis",
            "cycles": production_cycles,
            "cycle_length": cycle_length,
            "retained_states": production_cycles,
            "atomic_transitions": production_cycles * cycle_length,
            "structural_attempts": 0,
            "valid_structural": 0,
            "accepted_structural": 0,
            "sampler_seconds": math.fsum(trace["chunk_sampler_seconds"].tolist()),
            "training_sha256": manifest["training_sha256"],
            "branch_state_fingerprint": manifest["branch_state_fingerprint"],
        }
        for name, expected in expected_summary.items():
            if summary[name] != expected:
                raise ValueError(f"local-reference arm summary {name} does not rebuild exactly")
        summary_fingerprint = _digest(
            summary["final_state_fingerprint"],
            name="local-reference final-state fingerprint",
        )
        checkpoint = load_full_tiling_checkpoint(
            child_directory / "checkpoint.npz",
            problem,
            expected_run_manifest=checkpoint_manifest,
        )
        if (
            checkpoint.schedule_id != FIXED_BASIS_COMPOUND_SCHEDULE_ID
            or checkpoint.transitions_completed != production_cycles * cycle_length
            or checkpoint.schedule_phase != 0
            or checkpoint.kernel_settings.pair_allocation_refresh_slots != pair_slots
        ):
            raise ValueError("local-reference checkpoint transition identity is incompatible")
        state = checkpoint.state
        state_bounds = np.asarray(topology_bounds(state.allocation.tiling), dtype=np.int64)
        exact_final_arrays = {
            "rectangle_bounds": state_bounds,
            "leaf_masses": state.leaf_masses,
            "fixed_coefficients": state.fixed_coefficients,
        }
        if any(
            not np.array_equal(trace[name][-1], expected) for name, expected in exact_final_arrays.items()
        ):
            raise ValueError("local-reference final trace and checkpoint coordinates disagree")
        exact_final_scalars = {
            "root_total": state.root_total,
            "log_gaussian_likelihood": state.log_gaussian_likelihood,
            "log_likelihood": state.log_likelihood,
            "log_root_prior": state.log_root_prior,
            "log_allocation_prior": state.log_allocation_prior,
            "log_structural_prior": 0.0,
            "log_fixed_coefficient_prior": state.log_fixed_coefficient_prior,
            "log_target": state.log_target,
        }
        if any(trace[name][-1].item() != expected for name, expected in exact_final_scalars.items()):
            raise ValueError("local-reference final trace and checkpoint targets disagree")
        checkpoint_fingerprint = full_tiling_state_fingerprint(problem, state)
        if checkpoint_fingerprint != summary_fingerprint:
            raise ValueError("local-reference checkpoint and arm-summary final states disagree")
        masses = cast(FloatArray, trace["leaf_masses"])
        roots = cast(FloatArray, trace["root_total"])
        repeated_bounds = np.broadcast_to(
            expected_bounds,
            (production_cycles, *expected_bounds.shape),
        )
        fields = reconstruct_native_fields(
            repeated_bounds,
            masses,
            training.nominal_weight,
        )
        all_masses.append(masses)
        all_roots.append(roots)
        all_totals.append(common_native_totals(fields, training.nominal_weight))
        chain_audits.append(
            {
                "chain": chain,
                "completion_sha256": files[completion_name],
                "trace_sha256": child_files["trace.npz"],
                "checkpoint_sha256": child_files["checkpoint.npz"],
                "summary_sha256": child_files["summary.json"],
                "final_state_fingerprint": checkpoint_fingerprint,
                "transitions_completed": checkpoint.transitions_completed,
                "schedule_phase": checkpoint.schedule_phase,
                "schedule_id": checkpoint.schedule_id,
            }
        )
    masses = np.stack(all_masses)
    roots = np.stack(all_roots)
    totals = np.stack(all_totals)
    batch_size = production_cycles // 20
    batches = totals.reshape(4, 20, batch_size, len(PROJECTION_NAMES)).mean(axis=2)
    first_half = totals[:, : production_cycles // 2].mean(axis=1)
    second_half = totals[:, production_cycles // 2 :].mean(axis=1)
    expected_arrays: dict[str, NDArray[Any]] = {
        "root_total": roots,
        "leaf_masses": masses,
        "common_totals": totals,
        "batch_common_totals": batches,
        "first_half_common_mean": first_half,
        "second_half_common_mean": second_half,
        "rectangle_bounds": np.broadcast_to(
            expected_bounds,
            (4, production_cycles, *expected_bounds.shape),
        ),
        "seeds": np.asarray(seeds, dtype=np.int64),
    }
    with np.load(directory / "diagnostics_input.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(expected_arrays):
            raise ValueError("local diagnostics archive fields are incompatible")
        for name, expected in expected_arrays.items():
            if not np.array_equal(archive[name], expected):
                raise ValueError(f"local diagnostics {name} does not rebuild from raw traces")
    diagnostics_manifest = _strict_json(directory / "diagnostics_manifest.json")
    expected_diagnostics_manifest = {
        "schema": "openghg_inversions.mh_local_search_local_diagnostics_input.v1",
        "cell_id": training.cell_id,
        "definition_sha256": training.definition_sha256,
        "topology_sha256": topology_digest,
        "draws_per_chain": production_cycles,
        "chains": 4,
        "batches_per_chain": 20,
        "parameter_variable_names": [
            "root_total",
            *[
                f"leaf_mass[{r0}:{r1},{c0}:{c1}]"
                for r0, r1, c0, c1 in topology_bounds(
                    prepare_s0_nuts_reference(
                        training,
                        evaluation,
                        topology_role=cast(Any, topology_role),
                    ).data.tiling
                )
            ],
        ],
        "projection_names": list(PROJECTION_NAMES),
    }
    _require_exact_json(
        diagnostics_manifest,
        expected_diagnostics_manifest,
        name="local diagnostics manifest",
    )
    batch_means = batches.reshape(80, len(PROJECTION_NAMES))
    local_mcse = np.std(batch_means, axis=0, ddof=1) / math.sqrt(80.0)
    late_batch_means = batches[:, 10:, :].reshape(40, len(PROJECTION_NAMES))
    local_late_mcse = np.std(late_batch_means, axis=0, ddof=1) / math.sqrt(40.0)
    local_audit: dict[str, object] = {
        "completion_sha256": file_sha256(completion_path),
        "diagnostics_input_sha256": files["diagnostics_input.npz"],
        "manifest_sha256": files["manifest.json"],
        "diagnostics_manifest_sha256": files["diagnostics_manifest.json"],
        "source_revision": source_revision,
        "profile": local_profile,
        "branch_conditioning_cycles": expected_conditioning_cycles,
        "branch_conditioning_seed": expected_conditioning_seed,
        "branch_conditioning_arm": "fixed_basis",
        "branch_conditioning_schedule_id": FIXED_BASIS_COMPOUND_SCHEDULE_ID,
        "production_cycles_per_chain": production_cycles,
        "chains": 4,
        "seeds": list(seeds),
        "pair_slots_per_cycle": pair_slots,
        "batches_per_chain": 20,
        "batch_size": batch_size,
        "branch_run_complete_sha256": manifest["branch_run_complete_sha256"],
        "branch_state_sha256": manifest["branch_state_sha256"],
        "branch_state_fingerprint": manifest["branch_state_fingerprint"],
        "chains_audit": chain_audits,
        "projection_mean": dict(zip(PROJECTION_NAMES, np.mean(totals, axis=(0, 1)).tolist(), strict=True)),
        "projection_first_half_mean": dict(
            zip(PROJECTION_NAMES, np.mean(first_half, axis=0).tolist(), strict=True)
        ),
        "projection_second_half_mean": dict(
            zip(PROJECTION_NAMES, np.mean(second_half, axis=0).tolist(), strict=True)
        ),
        "projection_mcse": dict(zip(PROJECTION_NAMES, local_mcse.tolist(), strict=True)),
        "projection_late_window_mcse": dict(zip(PROJECTION_NAMES, local_late_mcse.tolist(), strict=True)),
        "mcse_formula": ("sample_sd_ddof1_of_80_pooled_batch_means_divided_by_sqrt_80"),
        "late_window_mcse_formula": (
            "sample_sd_ddof1_of_last_10_batch_means_per_chain_40_total_divided_by_sqrt_40"
        ),
    }
    if retry_authorization_token_sha256 is not None:
        local_audit["retry_authorization_token_sha256"] = retry_authorization_token_sha256
    return (
        np.asarray(local_mcse, dtype=np.float64),
        np.asarray(local_late_mcse, dtype=np.float64),
        local_audit,
    )


def _extreme(
    values: Mapping[str, float],
    *,
    minimum: bool = False,
) -> tuple[str, float]:
    selector = min if minimum else max
    name = selector(values, key=lambda item: values[item])
    return name, values[name]


def _failure(record: Mapping[str, object]) -> str | None:
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


def certify_conditional_reference(
    *,
    training_path: Path,
    evaluation_path: Path,
    nuts_directory: Path,
    local_directory: Path,
    _test_short_budget: tuple[int, int] | None = None,
) -> ConditionalReferenceCertificate:
    """Reopen two completed bundles and certify one of the five frozen cells."""
    training = load_training_artifact(training_path)
    evaluation = load_evaluation_artifact(evaluation_path)
    validate_artifact_pair(training, evaluation)
    if training.stage != "s0" or training.replicate != 0:
        raise ValueError("minimum conditional-reference integration supports S0 replicate zero")
    nuts, nuts_audit, topology_digest, topology_role = _validated_nuts(
        directory=nuts_directory,
        training_path=training_path,
        evaluation_path=evaluation_path,
        training=training,
        evaluation=evaluation,
    )
    source_revision = cast(str, nuts_audit["source_revision"])
    local_mcse, local_late_mcse, local_audit = _validated_local(
        directory=local_directory,
        training_path=training_path,
        evaluation_path=evaluation_path,
        training=training,
        evaluation=evaluation,
        topology_digest=topology_digest,
        topology_role=topology_role,
        source_revision=source_revision,
        test_short_budget=_test_short_budget,
    )
    projections = cast(Mapping[str, Mapping[str, float]], nuts["projections"])
    nuts_sd = np.asarray(
        [projections[name]["sd"] for name in PROJECTION_NAMES],
        dtype=np.float64,
    )
    nuts_mean = np.asarray(
        [projections[name]["mean"] for name in PROJECTION_NAMES],
        dtype=np.float64,
    )
    nuts_mcse = np.asarray(
        [projections[name]["mcse_mean"] for name in PROJECTION_NAMES],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(nuts_sd)) or np.any(nuts_sd <= 0.0):
        raise ValueError("NUTS projection SD must be finite and strictly positive")
    first = cast(Mapping[str, float], local_audit["projection_first_half_mean"])
    second = cast(Mapping[str, float], local_audit["projection_second_half_mean"])
    local_first = np.asarray([first[name] for name in PROJECTION_NAMES], dtype=np.float64)
    local_second = np.asarray([second[name] for name in PROJECTION_NAMES], dtype=np.float64)
    local_mcse_sd = local_mcse / nuts_sd
    half_difference_sd = np.abs(local_first - local_second) / nuts_sd
    combined_mcse = np.sqrt(local_late_mcse * local_late_mcse + nuts_mcse * nuts_mcse)
    tolerance = np.maximum(0.05 * nuts_sd, 3.0 * combined_mcse)
    local_vs_nuts = np.abs(local_second - nuts_mean) / tolerance
    per_projection = {
        name: {
            "nuts_mean": float(nuts_mean[index]),
            "nuts_sd": float(nuts_sd[index]),
            "nuts_mcse": float(nuts_mcse[index]),
            "local_first_half_mean": float(local_first[index]),
            "local_second_half_mean": float(local_second[index]),
            "local_mcse": float(local_mcse[index]),
            "local_late_window_mcse": float(local_late_mcse[index]),
            "local_mcse_over_nuts_sd": float(local_mcse_sd[index]),
            "half_difference_over_nuts_sd": float(half_difference_sd[index]),
            "combined_mcse": float(combined_mcse[index]),
            "comparison_tolerance": float(tolerance[index]),
            "local_vs_nuts_tolerance": float(local_vs_nuts[index]),
        }
        for index, name in enumerate(PROJECTION_NAMES)
    }
    if any(not math.isfinite(value) for item in per_projection.values() for value in item.values()):
        raise ValueError("conditional-reference projection arithmetic is non-finite")
    local_mcse_values = {name: per_projection[name]["local_mcse_over_nuts_sd"] for name in PROJECTION_NAMES}
    half_values = {name: per_projection[name]["half_difference_over_nuts_sd"] for name in PROJECTION_NAMES}
    comparison_values = {name: per_projection[name]["local_vs_nuts_tolerance"] for name in PROJECTION_NAMES}
    worst_local_mcse = _extreme(local_mcse_values)
    worst_half = _extreme(half_values)
    worst_comparison = _extreme(comparison_values)
    record: dict[str, object] = {
        "cell_id": training.cell_id,
        "definition_sha256": training.definition_sha256,
        "topology_sha256": topology_digest,
        "nuts_artifact_sha256": nuts_audit["completion_sha256"],
        "local_artifact_sha256": local_audit["completion_sha256"],
        "profile": nuts_audit["profile"],
        "pass": False,
        "divergences": nuts["divergences"],
        "worst_rhat_variable": nuts["worst_rhat_variable"],
        "worst_rhat_value": nuts["worst_rhat_value"],
        "min_bulk_ess_variable": nuts["minimum_bulk_ess_variable"],
        "min_bulk_ess_value": nuts["minimum_bulk_ess_value"],
        "min_tail_ess_variable": nuts["minimum_tail_ess_variable"],
        "min_tail_ess_value": nuts["minimum_tail_ess_value"],
        "worst_local_mcse_sd_projection": worst_local_mcse[0],
        "worst_local_mcse_sd_value": worst_local_mcse[1],
        "worst_half_difference_sd_projection": worst_half[0],
        "worst_half_difference_sd_value": worst_half[1],
        "worst_local_vs_nuts_tolerance_projection": worst_comparison[0],
        "worst_local_vs_nuts_tolerance_value": worst_comparison[1],
        "first_failed_gate": None,
    }
    failure = _failure(record)
    record["first_failed_gate"] = failure
    record["pass"] = failure is None
    if frozenset(record) != CONDITIONAL_REFERENCE_KEYS:
        raise RuntimeError("conditional-reference record schema drifted")
    audit: dict[str, object] = {
        "schema": "openghg_inversions.mh_local_search_conditional_reference_audit.v1",
        "source_revision": source_revision,
        "target": {
            "stage": training.stage,
            "scenario": evaluation.scenario,
            "replicate": training.replicate,
            "cell_id": training.cell_id,
            "definition_sha256": training.definition_sha256,
            "generation_commitment": training.generation_commitment,
            "training_sha256": file_sha256(training_path),
            "evaluation_sha256": file_sha256(evaluation_path),
        },
        "topology": {
            "role": topology_role,
            "sha256": topology_digest,
            "rectangle_bounds": [
                list(row)
                for row in (training.p0_bounds if topology_role == "p0" else evaluation.pstar_bounds)
            ],
        },
        "nuts": nuts_audit,
        "local": local_audit,
        "per_projection": per_projection,
        "comparison_formulas": {
            "local_mcse_over_nuts_sd": "local_mcse / nuts_posterior_sd",
            "half_difference_over_nuts_sd": (
                "abs(local_first_half_mean - local_second_half_mean) / nuts_posterior_sd"
            ),
            "combined_mcse": "sqrt(local_late_window_mcse**2 + nuts_mcse_mean**2)",
            "comparison_tolerance": ("max(0.05 * nuts_posterior_sd, 3 * combined_mcse)"),
            "local_vs_nuts_tolerance": ("abs(local_second_half_mean - nuts_mean) / comparison_tolerance"),
        },
        "gate_order": [
            "divergences",
            "worst_rhat",
            "min_bulk_ess",
            "min_tail_ess",
            "local_mcse_over_nuts_sd",
            "half_difference_over_nuts_sd",
            "local_vs_nuts_tolerance",
        ],
        "certificate": record,
    }
    return ConditionalReferenceCertificate(record=record, audit=audit)


def validate_conditional_reference_record(
    record: Mapping[str, object],
    *,
    training_path: Path,
    evaluation_path: Path,
    nuts_directory: Path,
    local_directory: Path,
    _test_short_budget: tuple[int, int] | None = None,
) -> ConditionalReferenceCertificate:
    """Recompute raw bundles and require exact equality with an indexed record."""
    if frozenset(record) != CONDITIONAL_REFERENCE_KEYS:
        raise ValueError("conditional-reference record keys are incompatible")
    replayed = certify_conditional_reference(
        training_path=training_path,
        evaluation_path=evaluation_path,
        nuts_directory=nuts_directory,
        local_directory=local_directory,
        _test_short_budget=_test_short_budget,
    )
    _require_exact_json(
        dict(record),
        dict(replayed.record),
        name="conditional-reference record",
    )
    return replayed


__all__ = [
    "CONDITIONAL_REFERENCE_KEYS",
    "ConditionalReferenceCertificate",
    "certify_conditional_reference",
    "validate_conditional_reference_record",
]
