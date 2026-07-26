#!/usr/bin/env python3
"""Run one of the five frozen S0 fixed-topology NumPyro NUTS references."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence, cast

import numpy as np

from openghg_inversions.experimental.rjmcmc.fixed_basis_nuts import (
    require_fixed_basis_nuts_float64,
    sample_fixed_basis_nuts,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_nuts_reference import (
    SAMPLER_SEED,
    preflight_s0_nuts_reference,
    prepare_s0_nuts_reference,
    reference_profile,
    summarize_reference_trace,
    validate_reference_trace,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_retry_authorization import (
    PrimaryNUTSFailure,
    validate_primary_nuts_retry_source,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_synthetic import (
    canonical_json,
    file_sha256,
    load_evaluation_artifact,
    load_training_artifact,
    topology_sha256,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
TRACE_FILENAME = "trace.nc"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "summary.json"
CHECKSUM_FILENAME = "checksums.json"
COMPLETION_FILENAME = "complete.json"
_PAYLOAD_FILENAMES = (TRACE_FILENAME, MANIFEST_FILENAME, SUMMARY_FILENAME)


def _create_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(dict(payload)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _strict_json_object(path: Path) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"{path} contains duplicate JSON key {name!r}")
            result[name] = value
        return result

    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {token}")),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(f"{path} is not strict JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one JSON object")
    return value


def _current_clean_revision() -> str:
    revision = subprocess.run(
        ("git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "-C", str(_REPOSITORY_ROOT), "status", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if _FULL_SHA.fullmatch(revision) is None or status:
        raise RuntimeError("NUTS reference execution requires a clean exact source revision")
    return revision


def _write_and_reopen_trace(
    inference_data: Any,
    output_path: Path,
    *,
    data: Any,
    expected_draws: int,
) -> tuple[dict[str, object], dict[str, object]]:
    in_memory = validate_reference_trace(
        inference_data,
        data=data,
        expected_draws=expected_draws,
    )
    temporary = output_path.with_name(f".{output_path.name}.tmp.nc")
    try:
        inference_data.to_netcdf(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        import arviz as az

        reopened = az.from_netcdf(temporary)
        try:
            reopened_audit = validate_reference_trace(
                reopened,
                data=data,
                expected_draws=expected_draws,
            )
        finally:
            getattr(reopened, "close")()
        if in_memory != reopened_audit:
            raise RuntimeError("reopened NetCDF validation differs from in-memory validation")
        os.replace(temporary, output_path)
        return in_memory, reopened_audit
    finally:
        temporary.unlink(missing_ok=True)


def _audit_staged_outputs(
    directory: Path,
    *,
    expected_manifest: Mapping[str, object],
    expected_summary: Mapping[str, object],
    expected_checksums: Mapping[str, object],
    first_pass_hashes: Mapping[str, str],
    data: Any,
    nominal_weight: Any,
    expected_draws: int,
    expected_trace_audit: Mapping[str, object],
) -> dict[str, str]:
    """Reopen every staged artifact and perform an independent digest pass."""
    if (directory / COMPLETION_FILENAME).exists():
        raise RuntimeError("NUTS completion was written before the final publication audit")
    if set(first_pass_hashes) != set(_PAYLOAD_FILENAMES):
        raise RuntimeError("NUTS first-pass payload catalogue is incompatible")
    expected_json = {
        MANIFEST_FILENAME: dict(expected_manifest),
        SUMMARY_FILENAME: dict(expected_summary),
        CHECKSUM_FILENAME: dict(expected_checksums),
    }
    for name, expected in expected_json.items():
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"NUTS staged JSON is not a regular file: {name}")
        reopened = _strict_json_object(path)
        if canonical_json(reopened) != canonical_json(expected):
            raise RuntimeError(f"NUTS staged JSON changed semantically: {name}")
        if path.read_text(encoding="utf-8") != canonical_json(expected) + "\n":
            raise RuntimeError(f"NUTS staged JSON is not exact canonical JSON: {name}")
    checksum_files = expected_checksums.get("files")
    if (
        expected_checksums.get("schema") != "openghg_inversions.mh_local_search_nuts_checksums.v1"
        or not isinstance(checksum_files, dict)
        or set(checksum_files) != set(_PAYLOAD_FILENAMES)
        or checksum_files != dict(first_pass_hashes)
    ):
        raise RuntimeError("NUTS checksum catalogue is incompatible")

    trace_path = directory / TRACE_FILENAME
    if not trace_path.is_file() or trace_path.is_symlink():
        raise RuntimeError("NUTS staged trace is not a regular file")
    import arviz as az

    reopened_trace = az.from_netcdf(trace_path)
    try:
        trace_audit = validate_reference_trace(
            reopened_trace,
            data=data,
            expected_draws=expected_draws,
        )
        scientific_summary = summarize_reference_trace(
            reopened_trace,
            data=data,
            nominal_weight=nominal_weight,
        )
    finally:
        getattr(reopened_trace, "close")()
    if trace_audit != dict(expected_trace_audit):
        raise RuntimeError("final NUTS trace validation differs from the staged trace audit")
    for name, expected in scientific_summary.items():
        if canonical_json(expected_summary.get(name)) != canonical_json(expected):
            raise RuntimeError(f"final NUTS trace does not reproduce summary field {name}")

    second_pass_hashes = {name: file_sha256(directory / name) for name in _PAYLOAD_FILENAMES}
    if second_pass_hashes != dict(first_pass_hashes):
        raise RuntimeError("NUTS payload changed between checksum passes")
    checksum_digest = file_sha256(directory / CHECKSUM_FILENAME)
    if expected_summary.get("nuts_artifact_sha256") != second_pass_hashes[TRACE_FILENAME]:
        raise RuntimeError("NUTS summary trace digest differs from the final trace")
    return {
        **second_pass_hashes,
        CHECKSUM_FILENAME: checksum_digest,
    }


def _manifest(
    *,
    arguments: argparse.Namespace,
    setup: Any,
    profile: Any,
    backend: Mapping[str, object],
    preflight: Sequence[Mapping[str, object]],
    retry_failure: PrimaryNUTSFailure | None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": "openghg_inversions.mh_local_search_nuts_manifest.v1",
        "cell_id": arguments.cell_id,
        "definition_sha256": arguments.definition_sha256,
        "cell_name": setup.cell_name,
        "scenario": setup.cell_name.rsplit("-", 1)[0],
        "topology_role": setup.topology_role,
        "topology_sha256": topology_sha256(setup.data.tiling),
        "training": {
            "path": str(arguments.training.resolve()),
            "sha256": file_sha256(arguments.training),
        },
        "evaluation": {
            "path": str(arguments.evaluation.resolve()),
            "sha256": file_sha256(arguments.evaluation),
        },
        "source_revision": arguments.source_revision,
        "sampler": {
            "name": "pymc_numpyro_nuts",
            "profile": profile.name,
            "chains": 4,
            "chain_method": "vectorized",
            "draws": profile.draws,
            "tune": profile.tune,
            "target_accept": profile.target_accept,
            "max_tree_depth": profile.max_tree_depth,
            "dense_mass": profile.dense_mass,
            "sampler_seed": SAMPLER_SEED,
            "jitter": False,
            "starts": [
                {
                    "profile": start.profile,
                    "seed": start.seed,
                    "root_total": float(start.state.root_total),
                    "leaf_share": np.asarray(
                        start.state.leaf_masses / start.state.root_total,
                        dtype=np.float64,
                    ).tolist(),
                    "expected_constrained_log_target": float(start.state.log_target),
                }
                for start in setup.starts
            ],
        },
        "backend": dict(backend),
        "preflight": [dict(item) for item in preflight],
    }
    if retry_failure is not None:
        result["schema"] = "openghg_inversions.mh_local_search_nuts_manifest.v2"
        result["retry_source_nuts_completion_sha256"] = retry_failure.completion_sha256
        result["retry_source_first_failed_gate"] = retry_failure.first_failed_gate
    return result


def _validate_identity_arguments(
    arguments: argparse.Namespace,
    *,
    training: Any,
) -> None:
    if arguments.cell_id != training.cell_id:
        raise ValueError("--cell-id must equal the training artifact cell_id")
    if arguments.definition_sha256 != training.definition_sha256:
        raise ValueError("--definition-sha256 must equal the training artifact definition_sha256")
    if _FULL_SHA.fullmatch(arguments.source_revision) is None:
        raise ValueError("--source-revision must be an exact lower-case full Git SHA")


def run(
    arguments: argparse.Namespace,
    *,
    enforce_clean_revision: bool = True,
) -> dict[str, object]:
    """Execute one fixed S0 reference or its complete preflight."""
    if arguments.output_directory.exists():
        raise FileExistsError(f"output path already exists: {arguments.output_directory}")
    if not arguments.output_directory.parent.is_dir():
        raise FileNotFoundError("output-directory parent does not exist")
    if enforce_clean_revision:
        current = _current_clean_revision()
        if arguments.source_revision != current:
            raise ValueError("--source-revision must equal the current exact source revision")

    training = load_training_artifact(arguments.training)
    evaluation = load_evaluation_artifact(arguments.evaluation)
    _validate_identity_arguments(arguments, training=training)
    profile = reference_profile(cast(Any, arguments.profile))
    setup = prepare_s0_nuts_reference(
        training,
        evaluation,
        topology_role=cast(Any, arguments.topology),
    )
    primary_nuts_directory = getattr(arguments, "primary_nuts_directory", None)
    retry_failure: PrimaryNUTSFailure | None = None
    if profile.name == "primary":
        if primary_nuts_directory is not None:
            raise ValueError("primary NUTS cannot cite retry lineage")
    else:
        if primary_nuts_directory is None:
            raise ValueError("retry1 requires --primary-nuts-directory")
        retry_failure = validate_primary_nuts_retry_source(
            training_path=arguments.training,
            evaluation_path=arguments.evaluation,
            primary_nuts_directory=primary_nuts_directory,
            topology_role=arguments.topology,
            source_revision=arguments.source_revision,
        )
    backend = require_fixed_basis_nuts_float64()
    model, preflight = preflight_s0_nuts_reference(setup)
    manifest = _manifest(
        arguments=arguments,
        setup=setup,
        profile=profile,
        backend=backend,
        preflight=preflight,
        retry_failure=retry_failure,
    )
    if arguments.dry_run:
        return {
            "schema": "openghg_inversions.mh_local_search_nuts_summary.v1",
            "status": "dry_run",
            "cell_id": training.cell_id,
            "definition_sha256": training.definition_sha256,
            "topology_sha256": topology_sha256(setup.data.tiling),
            "profile": profile.name,
            "manifest": manifest,
            "first_failed_gate": None,
        }

    started = perf_counter()
    inference_data = sample_fixed_basis_nuts(
        model,
        setup.data,
        draws=profile.draws,
        tune=profile.tune,
        seed=SAMPLER_SEED,
        target_accept=profile.target_accept,
        chains=4,
        cores=1,
        chain_method="vectorized",
        progressbar=bool(arguments.progressbar),
        max_tree_depth=profile.max_tree_depth,
        dense_mass=profile.dense_mass,
        initvals=tuple(start.initvals for start in setup.starts),
    )
    sampling_seconds = perf_counter() - started
    scientific_summary = summarize_reference_trace(
        inference_data,
        data=setup.data,
        nominal_weight=training.nominal_weight,
    )

    arguments.output_directory.mkdir()
    audits = _write_and_reopen_trace(
        inference_data,
        arguments.output_directory / TRACE_FILENAME,
        data=setup.data,
        expected_draws=profile.draws,
    )
    trace_digest = file_sha256(arguments.output_directory / TRACE_FILENAME)
    summary: dict[str, object] = {
        "schema": "openghg_inversions.mh_local_search_nuts_summary.v1",
        "status": "complete",
        "cell_id": training.cell_id,
        "definition_sha256": training.definition_sha256,
        "topology_sha256": topology_sha256(setup.data.tiling),
        "nuts_artifact_sha256": trace_digest,
        "profile": profile.name,
        "sampling_seconds": sampling_seconds,
        "trace_validation": {
            "in_memory": audits[0],
            "reopened_netcdf": audits[1],
        },
        **scientific_summary,
    }
    _create_json(arguments.output_directory / MANIFEST_FILENAME, manifest)
    _create_json(arguments.output_directory / SUMMARY_FILENAME, summary)
    first_pass_hashes = {name: file_sha256(arguments.output_directory / name) for name in _PAYLOAD_FILENAMES}
    checksum_payload = {
        "schema": "openghg_inversions.mh_local_search_nuts_checksums.v1",
        "files": first_pass_hashes,
    }
    _create_json(arguments.output_directory / CHECKSUM_FILENAME, checksum_payload)
    if not math.isfinite(sampling_seconds):
        raise RuntimeError("sampling time must be finite")
    final_hashes = _audit_staged_outputs(
        arguments.output_directory,
        expected_manifest=manifest,
        expected_summary=summary,
        expected_checksums=checksum_payload,
        first_pass_hashes=first_pass_hashes,
        data=setup.data,
        nominal_weight=training.nominal_weight,
        expected_draws=profile.draws,
        expected_trace_audit=audits[1],
    )
    _create_json(
        arguments.output_directory / COMPLETION_FILENAME,
        {
            "schema": "openghg_inversions.mh_local_search_nuts_completion.v1",
            "status": "complete",
            "checksums_sha256": final_hashes[CHECKSUM_FILENAME],
            "files": final_hashes,
            "first_failed_gate": summary["first_failed_gate"],
        },
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--training", type=Path, required=True)
    result.add_argument("--evaluation", type=Path, required=True)
    result.add_argument("--topology", choices=("p0", "pstar"), required=True)
    result.add_argument("--profile", choices=("primary", "retry1"), required=True)
    result.add_argument("--primary-nuts-directory", type=Path)
    result.add_argument("--output-directory", type=Path, required=True)
    result.add_argument("--source-revision", required=True)
    result.add_argument("--cell-id", required=True)
    result.add_argument("--definition-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--progressbar", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    result = run(arguments)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
