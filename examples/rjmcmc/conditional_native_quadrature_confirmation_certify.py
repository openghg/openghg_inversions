#!/usr/bin/env python
"""Authenticate and certify all 18 native-quadrature G2 shards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from examples.rjmcmc import conditional_native_quadrature_confirmation as confirmation
from examples.rjmcmc import conditional_native_quadrature_tiny_screen as screen
from examples.rjmcmc import conditional_residual_image_flow_certify as common

SCHEMA = "rjmcmc-conditional-native-quadrature-g2-certificate-v1"
MARKER_SCHEMA = "rjmcmc-conditional-native-quadrature-g2-complete-v1"


def _expected_stems(locked_order: int) -> tuple[str, ...]:
    """Return the exact 18-task G2 matrix."""
    return tuple(
        f"{regime}__{family}__root__O{locked_order}__base{seed}"
        for regime, family, tiling in screen.DEVELOPMENT_MATRIX
        for seed in screen.CONFIRMATION_SEEDS
        if tiling == "root"
    )


def _validate_task(
    input_directory: Path,
    stem: str,
    *,
    lock_sha256: str,
    locked_order: int,
    expected_source_revision: str,
    expected_driver_sha256: str,
    expected_protocol_sha256: str,
    selected_artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate one G2 report and completion marker."""
    report_path = input_directory / f"{stem}.json"
    marker_path = input_directory / f"{stem}.complete.json"
    if (
        not report_path.is_file()
        or report_path.is_symlink()
        or not marker_path.is_file()
        or marker_path.is_symlink()
    ):
        raise ValueError(f"G2 task report or marker is absent or unsafe: {stem}")
    envelope = common._read_json(report_path)  # pyright: ignore[reportPrivateUsage]
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"payload", "sha256"}
        or envelope["sha256"]
        != common._sha256_json(  # pyright: ignore[reportPrivateUsage]
            envelope["payload"]
        )
    ):
        raise ValueError(f"G2 task envelope is malformed: {stem}")
    result = envelope["payload"]
    if (
        not isinstance(result, dict)
        or result.get("schema") != confirmation.SCHEMA
        or result.get("source")
        != {
            "git_revision": expected_source_revision,
            "driver_sha256": expected_driver_sha256,
        }
        or result.get("protocol_sha256") != expected_protocol_sha256
        or result.get("quadrature_order") != locked_order
        or result.get("base_seed") not in screen.CONFIRMATION_SEEDS
        or result.get("lock_sha256") != lock_sha256
    ):
        raise ValueError(f"G2 task source or lock identity does not match: {stem}")
    expected_stem = (
        f"{result.get('case_id')}__O{result.get('quadrature_order')}__base{result.get('base_seed')}"
    )
    case_id = result.get("case_id")
    selected = selected_artifacts.get(case_id) if isinstance(case_id, str) else None
    if (
        expected_stem != stem
        or not isinstance(selected, dict)
        or result.get("artifact_sha256") != selected.get("artifact_sha256")
    ):
        raise ValueError(f"G2 task identity does not match its lock: {stem}")
    report_sha256 = common._sha256_path(  # pyright: ignore[reportPrivateUsage]
        report_path
    )
    marker = common._read_json(marker_path)  # pyright: ignore[reportPrivateUsage]
    expected_marker = {
        "schema": confirmation.MARKER_SCHEMA,
        "case_id": result["case_id"],
        "quadrature_order": locked_order,
        "base_seed": result["base_seed"],
        "task_pass": result["task_pass"],
        "artifact_sha256": result["artifact_sha256"],
        "lock_sha256": lock_sha256,
        "report_sha256": report_sha256,
    }
    if marker != expected_marker:
        raise ValueError(f"G2 completion marker does not authenticate: {stem}")
    return {
        "stem": stem,
        "case_id": result["case_id"],
        "base_seed": result["base_seed"],
        "quadrature_order": locked_order,
        "task_pass": result["task_pass"],
        "scientific_pass": result["evaluation"]["scientific_pass"],
        "simulator_pass": result["simulator_audit"]["pass"],
        "metrics": result["evaluation"]["metrics"],
        "checks": result["evaluation"]["checks"],
        "log_evidence": result["evaluation"]["posterior_summary"]["log_evidence"],
        "simulator_audit": result["simulator_audit"],
        "artifact_sha256": result["artifact_sha256"],
        "report_sha256": report_sha256,
        "marker_sha256": common._sha256_path(  # pyright: ignore[reportPrivateUsage]
            marker_path
        ),
    }


def certify_confirmation(
    input_directory: Path,
    *,
    lock_path: Path,
    expected_source_revision: str,
    expected_driver_sha256: str,
    expected_protocol_sha256: str,
) -> dict[str, Any]:
    """Authenticate all G2 shards and determine holdout eligibility."""
    lock, locked_order = confirmation._authenticate_lock(  # pyright: ignore[reportPrivateUsage]
        lock_path,
        expected_source_revision=expected_source_revision,
        expected_driver_sha256=expected_driver_sha256,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    lock_sha256 = common._sha256_path(  # pyright: ignore[reportPrivateUsage]
        lock_path
    )
    expected = _expected_stems(locked_order)
    expected_files = {f"{stem}{suffix}" for stem in expected for suffix in (".json", ".complete.json")}
    observed_files = {path.name for path in input_directory.iterdir() if path.is_file() or path.is_symlink()}
    missing = sorted(expected_files - observed_files)
    unexpected = sorted(observed_files - expected_files)
    tasks: list[dict[str, Any]] = []
    validation_errors: list[dict[str, str]] = []
    if not missing and not unexpected:
        for stem in expected:
            try:
                tasks.append(
                    _validate_task(
                        input_directory,
                        stem,
                        lock_sha256=lock_sha256,
                        locked_order=locked_order,
                        expected_source_revision=expected_source_revision,
                        expected_driver_sha256=expected_driver_sha256,
                        expected_protocol_sha256=expected_protocol_sha256,
                        selected_artifacts=lock["selected_artifacts"],
                    )
                )
            except ValueError as error:
                validation_errors.append({"stem": stem, "error": str(error)})
    complete_matrix = bool(
        not missing and not unexpected and not validation_errors and len(tasks) == len(expected)
    )
    evidence_ranges: dict[str, float] = {}
    seed_invariant_density: dict[str, bool] = {}
    for case_id in sorted(lock["selected_artifacts"]):
        case_tasks = [task for task in tasks if task["case_id"] == case_id]
        if len(case_tasks) != len(screen.CONFIRMATION_SEEDS):
            continue
        evidences = [float(task["log_evidence"]) for task in case_tasks]
        evidence_ranges[case_id] = max(evidences) - min(evidences)
        reference = case_tasks[0]
        seed_invariant_density[case_id] = all(
            task["artifact_sha256"] == reference["artifact_sha256"]
            and task["metrics"] == reference["metrics"]
            and task["checks"] == reference["checks"]
            and task["log_evidence"] == reference["log_evidence"]
            for task in case_tasks[1:]
        )
    evidence_range_pass = bool(
        len(evidence_ranges) == len(lock["selected_artifacts"])
        and all(
            value <= c1.THRESHOLDS["between_bank_log_evidence_range_nat"]
            for value in evidence_ranges.values()
        )
    )
    seed_invariance_pass = bool(
        len(seed_invariant_density) == len(lock["selected_artifacts"])
        and all(seed_invariant_density.values())
    )
    all_tasks_pass = bool(
        complete_matrix
        and all(task["task_pass"] for task in tasks)
        and evidence_range_pass
        and seed_invariance_pass
    )
    return {
        "schema": SCHEMA,
        "source": {
            "git_revision": expected_source_revision,
            "driver_sha256": expected_driver_sha256,
        },
        "protocol_sha256": expected_protocol_sha256,
        "lock_sha256": lock_sha256,
        "locked_order": locked_order,
        "expected_task_count": len(expected),
        "authenticated_task_count": len(tasks),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "validation_errors": validation_errors,
        "complete_matrix": complete_matrix,
        "between_seed_log_evidence_ranges_nat": evidence_ranges,
        "between_seed_log_evidence_threshold_nat": c1.THRESHOLDS["between_bank_log_evidence_range_nat"],
        "evidence_range_pass": evidence_range_pass,
        "seed_invariant_density": seed_invariant_density,
        "seed_invariance_pass": seed_invariance_pass,
        "all_tasks_pass": all_tasks_pass,
        "holdout_eligible": all_tasks_pass,
        "selected_artifacts": lock["selected_artifacts"],
        "tasks": tasks,
    }


def main() -> None:
    """Certify one complete G2 matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    certificate = certify_confirmation(
        args.input_directory,
        lock_path=args.lock,
        expected_source_revision=args.expected_source_revision,
        expected_driver_sha256=args.expected_driver_sha256,
        expected_protocol_sha256=args.expected_protocol_sha256,
    )
    envelope = {
        "payload": certificate,
        "sha256": common._sha256_json(  # pyright: ignore[reportPrivateUsage]
            certificate
        ),
    }
    certificate_path = args.output_directory / "confirmation-certificate.json"
    common._atomic_write(  # pyright: ignore[reportPrivateUsage]
        certificate_path,
        (
            common._canonical_json(envelope) + "\n"  # pyright: ignore[reportPrivateUsage]
        ).encode("utf-8"),
    )
    marker = {
        "schema": MARKER_SCHEMA,
        "certificate_sha256": common._sha256_path(  # pyright: ignore[reportPrivateUsage]
            certificate_path
        ),
        "all_tasks_pass": certificate["all_tasks_pass"],
        "holdout_eligible": certificate["holdout_eligible"],
    }
    marker_path = args.output_directory / "G2_COMPLETE.json"
    common._atomic_write(  # pyright: ignore[reportPrivateUsage]
        marker_path,
        (
            common._canonical_json(marker) + "\n"  # pyright: ignore[reportPrivateUsage]
        ).encode("utf-8"),
    )
    print(
        common._canonical_json(marker)  # pyright: ignore[reportPrivateUsage]
    )


if __name__ == "__main__":
    main()
