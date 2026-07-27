#!/usr/bin/env python3
"""Authenticate and merge the complete BP1 direct-NLE development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import conditional_residual_image_flow_tiny_screen as screen
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_flow import (
    ConditionalResidualImageFlow,
)

SCHEMA = "rjmcmc-conditional-residual-image-flow-development-certificate-v1"
LOCK_SCHEMA = "rjmcmc-conditional-residual-image-flow-common-lock-v1"


def _canonical_json(payload: object) -> str:
    """Return strict canonical JSON."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(payload: object) -> str:
    """Return the digest of canonical JSON."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    """Return the SHA-256 identity of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    """Read strict JSON without accepting non-finite constants."""
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read strict JSON: {path}") from error


def _expected_stems() -> tuple[str, ...]:
    """Return the source-pinned complete G1 matrix stems."""
    return tuple(
        f"{regime}__{family}__root__S{sample_count}"
        f"__base{screen.DEVELOPMENT_SELECTION_SEED}"
        for regime, family, tiling in screen.DEVELOPMENT_MATRIX
        for sample_count in screen.DEVELOPMENT_SAMPLE_COUNTS
        if tiling == "root"
    )


def _common_lock(
    passes: dict[str, dict[int, bool]],
    *,
    sample_counts: Sequence[int] = screen.DEVELOPMENT_SAMPLE_COUNTS,
) -> int | None:
    """Return the smallest common all-larger two-size passing suffix."""
    counts = tuple(sample_counts)
    if len(counts) < 2:
        raise ValueError("at least two source-pinned training sizes are required")
    for start in range(len(counts) - 1):
        suffix = counts[start:]
        if all(
            all(case_results.get(sample_count) is True for sample_count in suffix)
            for case_results in passes.values()
        ):
            return counts[start]
    return None


def _validate_marker(
    marker_path: Path,
    *,
    result: dict[str, Any],
    report_sha256: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    """Authenticate one task completion marker."""
    marker = _read_json(marker_path)
    expected = {
        "schema": "rjmcmc-conditional-residual-image-flow-task-complete-v1",
        "case_id": result["case_id"],
        "training_sample_count": result["training_sample_count"],
        "base_seed": result["base_seed"],
        "task_pass": result["task_pass"],
        "artifact_sha256": artifact_sha256,
        "report_sha256": report_sha256,
    }
    if marker != expected:
        raise ValueError(f"completion marker does not authenticate task: {marker_path}")
    return marker


def _validate_task(
    input_directory: Path,
    stem: str,
    *,
    expected_source_revision: str,
    expected_driver_sha256: str,
    expected_protocol_sha256: str,
    expected_base_seed: int,
) -> dict[str, Any]:
    """Authenticate one report, flow artifact, and completion marker."""
    report_path = input_directory / f"{stem}.json"
    marker_path = input_directory / f"{stem}.complete.json"
    if (
        not report_path.is_file()
        or report_path.is_symlink()
        or not marker_path.is_file()
        or marker_path.is_symlink()
    ):
        raise ValueError(f"task report or marker is absent or unsafe: {stem}")
    envelope = _read_json(report_path)
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "sha256"}:
        raise ValueError(f"task report envelope is malformed: {stem}")
    if envelope["sha256"] != _sha256_json(envelope["payload"]):
        raise ValueError(f"task report envelope digest does not match: {stem}")
    payload = envelope["payload"]
    if not isinstance(payload, dict) or set(payload) != {"result", "artifact"}:
        raise ValueError(f"task report payload is malformed: {stem}")
    result = payload["result"]
    artifact_record = payload["artifact"]
    if (
        not isinstance(result, dict)
        or result.get("schema") != screen.SCHEMA
        or result.get("profile") != "development"
        or result.get("source")
        != {
            "git_revision": expected_source_revision,
            "driver_sha256": expected_driver_sha256,
        }
        or result.get("protocol", {}).get("sha256")
        != expected_protocol_sha256
        or result.get("protocol", {}).get("name") != screen.PROTOCOL
        or result.get("base_seed") != expected_base_seed
    ):
        raise ValueError(f"task source or protocol identity does not match: {stem}")
    expected_stem = (
        f"{result.get('case_id')}__S{result.get('training_sample_count')}"
        f"__base{result.get('base_seed')}"
    )
    if expected_stem != stem:
        raise ValueError(f"task stem does not match its result: {stem}")
    if not isinstance(artifact_record, dict) or set(artifact_record) != {
        "path",
        "sha256",
    }:
        raise ValueError(f"task artifact record is malformed: {stem}")
    if artifact_record["path"] != f"{stem}.flow":
        raise ValueError(f"task artifact path is not canonical: {stem}")
    artifact_path = input_directory / artifact_record["path"]
    if not artifact_path.is_file() or artifact_path.is_symlink():
        raise ValueError(f"task flow artifact is absent or unsafe: {stem}")
    artifact_sha256 = _sha256_path(artifact_path)
    if (
        artifact_sha256 != artifact_record["sha256"]
        or artifact_sha256 != result.get("selected_artifact_sha256")
    ):
        raise ValueError(f"task flow artifact digest does not match: {stem}")
    artifact_bytes = artifact_path.read_bytes()
    replay = ConditionalResidualImageFlow.from_bytes(
        artifact_bytes,
        expected_sha256=artifact_sha256,
    )
    if replay.to_bytes() != artifact_bytes:
        raise ValueError(f"task flow artifact does not replay canonically: {stem}")
    report_sha256 = _sha256_path(report_path)
    marker = _validate_marker(
        marker_path,
        result=result,
        report_sha256=report_sha256,
        artifact_sha256=artifact_sha256,
    )
    return {
        "stem": stem,
        "case_id": result["case_id"],
        "training_sample_count": result["training_sample_count"],
        "task_pass": result["task_pass"],
        "fit_development_pass": result["fit_development_pass"],
        "selected_generalization_pass": result[
            "selected_generalization_pass"
        ],
        "artifact_replay_pass": result["artifact_replay_pass"],
        "scientific_pass": result["evaluation"]["scientific_pass"],
        "metrics": result["evaluation"]["metrics"],
        "checks": result["evaluation"]["checks"],
        "log_evidence": result["evaluation"]["posterior_summary"][
            "log_evidence"
        ],
        "artifact_sha256": artifact_sha256,
        "report_sha256": report_sha256,
        "marker_sha256": _sha256_path(marker_path),
        "marker": marker,
    }


def merge_development(
    input_directory: Path,
    *,
    expected_source_revision: str,
    expected_driver_sha256: str,
    expected_protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Authenticate all 24 tasks and determine the common lock."""
    expected = _expected_stems()
    expected_files = {
        f"{stem}{suffix}"
        for stem in expected
        for suffix in (".flow", ".json", ".complete.json")
    }
    observed_files = {
        path.name
        for path in input_directory.iterdir()
        if path.is_file() or path.is_symlink()
    }
    missing = sorted(expected_files - observed_files)
    unexpected = sorted(observed_files - expected_files)
    task_records: list[dict[str, Any]] = []
    validation_errors: list[dict[str, str]] = []
    if not missing and not unexpected:
        for stem in expected:
            try:
                task_records.append(
                    _validate_task(
                        input_directory,
                        stem,
                        expected_source_revision=expected_source_revision,
                        expected_driver_sha256=expected_driver_sha256,
                        expected_protocol_sha256=expected_protocol_sha256,
                        expected_base_seed=screen.DEVELOPMENT_SELECTION_SEED,
                    )
                )
            except ValueError as error:
                validation_errors.append(
                    {
                        "stem": stem,
                        "error": str(error),
                    }
                )
    complete_matrix = bool(
        not missing
        and not unexpected
        and not validation_errors
        and len(task_records) == len(expected)
    )
    passes: dict[str, dict[int, bool]] = {
        f"{regime}__{family}__root": {}
        for regime, family, tiling in screen.DEVELOPMENT_MATRIX
        if tiling == "root"
    }
    if complete_matrix:
        for record in task_records:
            passes[record["case_id"]][record["training_sample_count"]] = bool(
                record["task_pass"]
            )
    locked_sample_count = _common_lock(passes) if complete_matrix else None
    terminal_reason: str | None
    if not complete_matrix:
        terminal_reason = "complete authenticated 24-task matrix is absent"
    elif locked_sample_count is None:
        terminal_reason = (
            "no common all-six-case all-larger passing suffix of length at least two"
        )
    else:
        terminal_reason = None
    certificate = {
        "schema": SCHEMA,
        "source": {
            "git_revision": expected_source_revision,
            "driver_sha256": expected_driver_sha256,
        },
        "protocol_sha256": expected_protocol_sha256,
        "expected_task_count": len(expected),
        "authenticated_task_count": len(task_records),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "validation_errors": validation_errors,
        "complete_matrix": complete_matrix,
        "passes": passes,
        "locked_sample_count": locked_sample_count,
        "lock_published": locked_sample_count is not None,
        "terminal_reason": terminal_reason,
        "tasks": task_records,
    }
    lock: dict[str, Any] | None = None
    if locked_sample_count is not None:
        selected = {
            record["case_id"]: {
                "artifact_sha256": record["artifact_sha256"],
                "report_sha256": record["report_sha256"],
                "log_evidence": record["log_evidence"],
            }
            for record in task_records
            if record["training_sample_count"] == locked_sample_count
        }
        lock_payload = {
            "schema": LOCK_SCHEMA,
            "source": certificate["source"],
            "protocol_sha256": expected_protocol_sha256,
            "locked_sample_count": locked_sample_count,
            "selection_seed": screen.DEVELOPMENT_SELECTION_SEED,
            "confirmation_seeds": list(screen.CONFIRMATION_SEEDS),
            "selected_artifacts": selected,
            "certificate_sha256": _sha256_json(certificate),
        }
        lock = {
            "payload": lock_payload,
            "sha256": _sha256_json(lock_payload),
        }
    return certificate, lock


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write bytes atomically without replacing existing evidence."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Merge one complete G1 development matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    certificate, lock = merge_development(
        args.input_directory,
        expected_source_revision=args.expected_source_revision,
        expected_driver_sha256=args.expected_driver_sha256,
        expected_protocol_sha256=args.expected_protocol_sha256,
    )
    certificate_envelope = {
        "payload": certificate,
        "sha256": _sha256_json(certificate),
    }
    certificate_path = args.output_directory / "development-certificate.json"
    _atomic_write(
        certificate_path,
        (_canonical_json(certificate_envelope) + "\n").encode("utf-8"),
    )
    lock_path: Path | None = None
    if lock is not None:
        published_lock_path = args.output_directory / "common-lock.json"
        _atomic_write(
            published_lock_path,
            (_canonical_json(lock) + "\n").encode("utf-8"),
        )
        lock_path = published_lock_path
    marker_payload = {
        "schema": "rjmcmc-conditional-residual-image-flow-development-merge-complete-v1",
        "certificate_sha256": _sha256_path(certificate_path),
        "lock_published": lock is not None,
        "lock_sha256": (
            _sha256_path(lock_path)
            if lock_path is not None
            else None
        ),
        "terminal_reason": certificate["terminal_reason"],
    }
    marker_path = args.output_directory / "MERGE_COMPLETE.json"
    _atomic_write(
        marker_path,
        (_canonical_json(marker_payload) + "\n").encode("utf-8"),
    )
    print(_canonical_json(marker_payload))


if __name__ == "__main__":
    main()
