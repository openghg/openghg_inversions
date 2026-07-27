#!/usr/bin/env python
"""Authenticate all 18 BP1 sbi-NSF confirmation shards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from examples.rjmcmc import conditional_residual_image_flow_certify as common
from examples.rjmcmc import conditional_residual_image_sbi_nsf_certify as development
from examples.rjmcmc import conditional_residual_image_sbi_nsf_tiny_screen as screen

SCHEMA = "rjmcmc-conditional-residual-image-sbi-nsf-confirmation-certificate-v1"
ELIGIBLE_SCHEMA = "rjmcmc-conditional-residual-image-sbi-nsf-holdout-eligible-v1"
MARKER_SCHEMA = "rjmcmc-conditional-residual-image-sbi-nsf-confirmation-complete-v1"


def _authenticate_lock(
    lock_path: Path,
    *,
    expected_source_revision: str,
    expected_driver_sha256: str,
    expected_protocol_sha256: str,
) -> tuple[dict[str, Any], int]:
    """Authenticate the G1 common lock and return its locked size."""
    if not lock_path.is_file() or lock_path.is_symlink():
        raise ValueError("G1 common lock is absent or unsafe")
    lock = common._read_json(lock_path)  # pyright: ignore[reportPrivateUsage]
    if not isinstance(lock, dict) or set(lock) != {"payload", "sha256"}:
        raise ValueError("G1 common lock envelope is malformed")
    if lock["sha256"] != common._sha256_json(  # pyright: ignore[reportPrivateUsage]
        lock["payload"]
    ):
        raise ValueError("G1 common lock digest does not match")
    payload = lock["payload"]
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != development.LOCK_SCHEMA
        or payload.get("source")
        != {
            "git_revision": expected_source_revision,
            "driver_sha256": expected_driver_sha256,
        }
        or payload.get("protocol_sha256") != expected_protocol_sha256
        or payload.get("selection_seed") != screen.DEVELOPMENT_SELECTION_SEED
        or payload.get("confirmation_seeds") != list(screen.CONFIRMATION_SEEDS)
    ):
        raise ValueError("G1 common lock source or protocol identity does not match")
    locked_sample_count = payload.get("locked_sample_count")
    if locked_sample_count not in screen.DEVELOPMENT_SAMPLE_COUNTS:
        raise ValueError("G1 common lock training size is not source-pinned")
    selected = payload.get("selected_artifacts")
    expected_cases = {
        f"{regime}__{family}__root"
        for regime, family, tiling in screen.DEVELOPMENT_MATRIX
        if tiling == "root"
    }
    if not isinstance(selected, dict) or set(selected) != expected_cases:
        raise ValueError("G1 common lock does not select every case")
    return lock, int(locked_sample_count)


def certify_confirmation(
    input_directory: Path,
    *,
    lock_path: Path,
    expected_source_revision: str,
    expected_driver_sha256: str,
    expected_protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Authenticate the full confirmation matrix and apply repeat gates."""
    lock, locked_sample_count = _authenticate_lock(
        lock_path,
        expected_source_revision=expected_source_revision,
        expected_driver_sha256=expected_driver_sha256,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    case_ids = tuple(
        f"{regime}__{family}__root"
        for regime, family, tiling in screen.DEVELOPMENT_MATRIX
        if tiling == "root"
    )
    expected_stems = tuple(
        f"{case_id}__S{locked_sample_count}__base{seed}"
        for case_id in case_ids
        for seed in screen.CONFIRMATION_SEEDS
    )
    expected_files = {
        f"{stem}{suffix}"
        for stem in expected_stems
        for suffix in (".nsf", ".json", ".complete.json")
    }
    observed_files = {
        path.name
        for path in input_directory.iterdir()
        if path.is_file() or path.is_symlink()
    }
    missing = sorted(expected_files - observed_files)
    unexpected = sorted(observed_files - expected_files)
    tasks: list[dict[str, Any]] = []
    validation_errors: list[dict[str, str]] = []
    if not missing and not unexpected:
        for case_id in case_ids:
            for seed in screen.CONFIRMATION_SEEDS:
                stem = f"{case_id}__S{locked_sample_count}__base{seed}"
                try:
                    task = development._validate_task(
                        input_directory,
                        stem,
                        expected_source_revision=expected_source_revision,
                        expected_driver_sha256=expected_driver_sha256,
                        expected_protocol_sha256=expected_protocol_sha256,
                        expected_base_seed=seed,
                    )
                    task["base_seed"] = seed
                    tasks.append(task)
                except ValueError as error:
                    validation_errors.append({"stem": stem, "error": str(error)})
    complete_matrix = bool(
        not missing
        and not unexpected
        and not validation_errors
        and len(tasks) == len(expected_stems)
    )
    evidence_gates: dict[str, dict[str, float | bool]] = {}
    if complete_matrix:
        for case_id in case_ids:
            evidences = [
                float(task["log_evidence"])
                for task in tasks
                if task["case_id"] == case_id
            ]
            evidence_range = max(evidences) - min(evidences)
            threshold = float(
                c1.THRESHOLDS["between_bank_log_evidence_range_nat"]
            )
            evidence_gates[case_id] = {
                "range_nat": evidence_range,
                "threshold_nat": threshold,
                "pass": bool(evidence_range <= threshold),
            }
    all_task_pass = bool(
        complete_matrix and all(task["task_pass"] for task in tasks)
    )
    all_evidence_pass = bool(
        complete_matrix
        and all(gate["pass"] for gate in evidence_gates.values())
        and len(evidence_gates) == len(case_ids)
    )
    holdout_eligible = bool(all_task_pass and all_evidence_pass)
    if not complete_matrix:
        terminal_reason: str | None = (
            "complete authenticated 18-task confirmation matrix is absent"
        )
    elif not all_task_pass:
        terminal_reason = "one or more confirmation shards failed"
    elif not all_evidence_pass:
        terminal_reason = "one or more between-seed evidence gates failed"
    else:
        terminal_reason = None
    certificate = {
        "schema": SCHEMA,
        "source": {
            "git_revision": expected_source_revision,
            "driver_sha256": expected_driver_sha256,
        },
        "protocol_sha256": expected_protocol_sha256,
        "g1_lock_raw_sha256": common._sha256_path(  # pyright: ignore[reportPrivateUsage]
            lock_path
        ),
        "g1_lock_envelope_sha256": lock["sha256"],
        "locked_sample_count": locked_sample_count,
        "expected_task_count": len(expected_stems),
        "authenticated_task_count": len(tasks),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "validation_errors": validation_errors,
        "complete_matrix": complete_matrix,
        "all_task_pass": all_task_pass,
        "evidence_gates": evidence_gates,
        "all_evidence_pass": all_evidence_pass,
        "holdout_eligible": holdout_eligible,
        "terminal_reason": terminal_reason,
        "tasks": tasks,
    }
    eligible: dict[str, Any] | None = None
    if holdout_eligible:
        eligible_payload = {
            "schema": ELIGIBLE_SCHEMA,
            "source": certificate["source"],
            "protocol_sha256": expected_protocol_sha256,
            "locked_sample_count": locked_sample_count,
            "g1_lock_raw_sha256": certificate["g1_lock_raw_sha256"],
            "confirmation_certificate_sha256": common._sha256_json(  # pyright: ignore[reportPrivateUsage]
                certificate
            ),
            "artifacts": {
                f"{task['case_id']}__base{task['base_seed']}": task[
                    "artifact_sha256"
                ]
                for task in tasks
            },
            "protected_action_authorized": False,
        }
        eligible = {
            "payload": eligible_payload,
            "sha256": common._sha256_json(  # pyright: ignore[reportPrivateUsage]
                eligible_payload
            ),
        }
    return certificate, eligible


def main() -> None:
    """Certify one complete G2 confirmation matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    certificate, eligible = certify_confirmation(
        args.input_directory,
        lock_path=args.lock_path,
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
    eligible_path: Path | None = None
    if eligible is not None:
        published_path = args.output_directory / "HOLDOUT_ELIGIBLE.json"
        common._atomic_write(  # pyright: ignore[reportPrivateUsage]
            published_path,
            (
                common._canonical_json(eligible) + "\n"  # pyright: ignore[reportPrivateUsage]
            ).encode("utf-8"),
        )
        eligible_path = published_path
    marker = {
        "schema": MARKER_SCHEMA,
        "certificate_sha256": common._sha256_path(  # pyright: ignore[reportPrivateUsage]
            certificate_path
        ),
        "holdout_eligible": eligible is not None,
        "eligible_sha256": (
            common._sha256_path(eligible_path)  # pyright: ignore[reportPrivateUsage]
            if eligible_path is not None
            else None
        ),
        "terminal_reason": certificate["terminal_reason"],
        "protected_action_authorized": False,
    }
    marker_path = args.output_directory / "CONFIRMATION_COMPLETE.json"
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
