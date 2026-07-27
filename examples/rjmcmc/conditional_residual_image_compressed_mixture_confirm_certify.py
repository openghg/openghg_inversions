#!/usr/bin/env python3
"""Certify the complete exact-mixture independent-scramble matrix."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import (
    conditional_residual_image_compressed_mixture_certify as development_certify,
)
from examples.rjmcmc import (
    conditional_residual_image_compressed_mixture_confirm as confirmation,
)
from examples.rjmcmc import (
    conditional_residual_image_compressed_mixture_tiny_screen as development,
)

SCHEMA = "rjmcmc-compressed-mixture-confirmation-decision-v1"


def _finite_float(value: object, label: str) -> float:
    """Return one finite non-boolean numerical field or fail closed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numerical")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _evaluation_metric(
    report: dict[str, Any],
    *,
    branch: str,
    name: str,
    label: str,
) -> float:
    """Read one required exact-evaluation metric from an artifact."""
    try:
        value = report[branch]["exact_evaluation"]["metrics"][name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{label} is missing required {branch} metric {name!r}") from error
    return _finite_float(value, f"{label} {branch} metric {name!r}")


def _log_evidence(
    report: dict[str, Any],
    *,
    branch: str,
    label: str,
) -> float:
    """Read one required exact-evaluation log evidence from an artifact."""
    try:
        value = report[branch]["exact_evaluation"]["posterior_summary"]["log_evidence"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{label} is missing required {branch} log evidence") from error
    return _finite_float(value, f"{label} {branch} log evidence")


def _driver_sha256() -> str:
    """Return the exact digest of the confirmation driver."""
    return hashlib.sha256(Path(confirmation.__file__).read_bytes()).hexdigest()


def certify_confirmation(
    *,
    report_directory: Path,
    expected_revision: str,
) -> dict[str, Any]:
    """Authenticate and merge all 18 frozen confirmation artifacts."""
    if report_directory.is_symlink() or not report_directory.is_dir():
        raise ValueError("report directory must be a real directory")
    paths = sorted(report_directory.glob("*.json"))
    case_ids = development._expected_development_case_ids()
    expected_pairs = {(case_id, seed) for case_id in case_ids for seed in confirmation.SOURCE_SEEDS}
    if len(paths) != len(expected_pairs):
        raise ValueError(f"confirmation directory must contain exactly {len(expected_pairs)} JSON artifacts")
    reports: dict[tuple[str, int], dict[str, Any]] = {}
    raw_sha256: dict[str, str] = {}
    expected_driver_sha256 = _driver_sha256()
    expected_protocol_sha256 = development._sha256_json(confirmation._protocol_payload())
    for path in paths:
        report, raw_digest = development_certify._read_canonical_json(path)
        expected_scalars = {
            "schema": confirmation.SCHEMA,
            "protocol": confirmation.PROTOCOL,
            "protocol_sha256": expected_protocol_sha256,
            "source_git_revision": expected_revision,
            "driver_sha256": expected_driver_sha256,
            "observed_residual_used_for_basis_selection": False,
            "retuning_performed": False,
            "protected_catalogue_accessed": False,
            "production_output_written": False,
            "structural_inference_licensed": False,
        }
        for name, expected in expected_scalars.items():
            if report.get(name) != expected:
                raise ValueError(f"{path}: confirmation field {name!r} is not frozen")
        development_record = report.get("development")
        if not isinstance(development_record, dict):
            raise ValueError(f"{path}: development identity is absent")
        expected_development = {
            "revision": confirmation.DEVELOPMENT_REVISION,
            "source_decision_raw_sha256": (confirmation.DEVELOPMENT_SOURCE_DECISION_RAW_SHA256),
            "source_lock_sha256": (confirmation.DEVELOPMENT_SOURCE_LOCK_SHA256),
            "compression_decision_raw_sha256": (confirmation.DEVELOPMENT_COMPRESSION_DECISION_RAW_SHA256),
            "locked_source_sample_count": (confirmation.LOCKED_SOURCE_SAMPLE_COUNT),
            "locked_component_count": confirmation.LOCKED_COMPONENT_COUNT,
        }
        if development_record != expected_development:
            raise ValueError(f"{path}: development decision identity changed")
        case_id = report.get("case_id")
        source_seed = report.get("source_seed")
        if not isinstance(case_id, str) or isinstance(source_seed, bool) or not isinstance(source_seed, int):
            raise ValueError(f"{path}: case/seed types are invalid")
        pair = (case_id, source_seed)
        if pair not in expected_pairs:
            raise ValueError(f"{path}: case/seed pair is not frozen")
        if pair in reports:
            raise ValueError(f"duplicate confirmation pair: {pair}")
        if report.get("cluster_seed") != confirmation.CLUSTER_SEED:
            raise ValueError(f"{path}: cluster seed changed")
        reports[pair] = report
        raw_sha256[f"{case_id}__seed{source_seed}"] = raw_digest
    if set(reports) != expected_pairs:
        raise ValueError("confirmation artifacts do not cover the exact matrix")

    failures = [
        {
            "case_id": case_id,
            "source_seed": source_seed,
            "failed_checks": sorted(
                name
                for name, passed in reports[(case_id, source_seed)]["confirmation_checks"].items()
                if passed is not True
            ),
        }
        for case_id, source_seed in sorted(expected_pairs)
        if reports[(case_id, source_seed)].get("scientific_pass") is not True
    ]
    between_seed_name = "between_bank_log_evidence_range_nat"
    metric_names = tuple(name for name in development.c1.THRESHOLDS if name != between_seed_name)
    source_metric_maxima = {
        name: max(
            _evaluation_metric(
                report,
                branch="source",
                name=name,
                label=f"{case_id}/seed{source_seed}",
            )
            for (case_id, source_seed), report in reports.items()
        )
        for name in metric_names
    }
    compression_metric_maxima = {
        name: max(
            _evaluation_metric(
                report,
                branch="compression",
                name=name,
                label=f"{case_id}/seed{source_seed}",
            )
            for (case_id, source_seed), report in reports.items()
        )
        for name in metric_names
    }
    source_evidence_ranges = {
        case_id: (
            max(
                _log_evidence(
                    reports[(case_id, source_seed)],
                    branch="source",
                    label=f"{case_id}/seed{source_seed}",
                )
                for source_seed in confirmation.SOURCE_SEEDS
            )
            - min(
                _log_evidence(
                    reports[(case_id, source_seed)],
                    branch="source",
                    label=f"{case_id}/seed{source_seed}",
                )
                for source_seed in confirmation.SOURCE_SEEDS
            )
        )
        for case_id in case_ids
    }
    compression_evidence_ranges = {
        case_id: (
            max(
                _log_evidence(
                    reports[(case_id, source_seed)],
                    branch="compression",
                    label=f"{case_id}/seed{source_seed}",
                )
                for source_seed in confirmation.SOURCE_SEEDS
            )
            - min(
                _log_evidence(
                    reports[(case_id, source_seed)],
                    branch="compression",
                    label=f"{case_id}/seed{source_seed}",
                )
                for source_seed in confirmation.SOURCE_SEEDS
            )
        )
        for case_id in case_ids
    }
    between_seed_threshold = float(development.c1.THRESHOLDS[between_seed_name])
    between_seed_failures = [
        {
            "case_id": case_id,
            "source_evidence_range_nat": source_evidence_ranges[case_id],
            "compression_evidence_range_nat": (compression_evidence_ranges[case_id]),
            "threshold_nat": between_seed_threshold,
        }
        for case_id in case_ids
        if source_evidence_ranges[case_id] > between_seed_threshold
        or compression_evidence_ranges[case_id] > between_seed_threshold
    ]
    maximum_mean_closure = max(
        float(report["compression"]["moment_diagnostics"]["mean_maximum_absolute_difference_from_source"])
        for report in reports.values()
    )
    maximum_covariance_closure = max(
        float(
            report["compression"]["moment_diagnostics"]["covariance_maximum_absolute_difference_from_source"]
        )
        for report in reports.values()
    )
    return {
        "schema": SCHEMA,
        "protocol": confirmation.PROTOCOL,
        "protocol_sha256": expected_protocol_sha256,
        "source_git_revision": expected_revision,
        "driver_sha256": expected_driver_sha256,
        "development_revision": confirmation.DEVELOPMENT_REVISION,
        "development_source_decision_raw_sha256": (confirmation.DEVELOPMENT_SOURCE_DECISION_RAW_SHA256),
        "development_source_lock_sha256": (confirmation.DEVELOPMENT_SOURCE_LOCK_SHA256),
        "development_compression_decision_raw_sha256": (
            confirmation.DEVELOPMENT_COMPRESSION_DECISION_RAW_SHA256
        ),
        "locked_source_sample_count": (confirmation.LOCKED_SOURCE_SAMPLE_COUNT),
        "locked_component_count": confirmation.LOCKED_COMPONENT_COUNT,
        "matrix_case_ids": list(case_ids),
        "source_seeds": list(confirmation.SOURCE_SEEDS),
        "cluster_seed": confirmation.CLUSTER_SEED,
        "artifact_count": len(reports),
        "artifact_raw_sha256": raw_sha256,
        "source_metric_maxima": source_metric_maxima,
        "compression_metric_maxima": compression_metric_maxima,
        "source_between_seed_log_evidence_range_nat_by_case": (source_evidence_ranges),
        "compression_between_seed_log_evidence_range_nat_by_case": (compression_evidence_ranges),
        "between_seed_log_evidence_range_threshold_nat": (between_seed_threshold),
        "maximum_compression_mean_closure_error": maximum_mean_closure,
        "maximum_compression_covariance_closure_error": (maximum_covariance_closure),
        "failures": failures,
        "between_seed_failures": between_seed_failures,
        "eligible": not failures and not between_seed_failures,
        "protected_catalogue_accessed": False,
        "production_output_written": False,
        "structural_inference_licensed": False,
    }


def _revision(value: str) -> str:
    """Validate one complete lower-case Git revision."""
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("revision must be a 40-character lower-case Git SHA")
    return value


def _parser() -> argparse.ArgumentParser:
    """Build the confirmation-certification CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument(
        "--expected-source-revision",
        type=_revision,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Certify and atomically publish the complete confirmation decision."""
    args = _parser().parse_args(argv)
    decision = certify_confirmation(
        report_directory=args.report_directory,
        expected_revision=args.expected_source_revision,
    )
    development._write_atomic_json(args.output, decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
