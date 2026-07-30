#!/usr/bin/env python3
"""Merge one corrected exploratory array without inventing pass thresholds."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from examples.rjmcmc import score_regularized_flow_corrected_array as arrays
from examples.rjmcmc import score_regularized_flow_corrected_exploration as experiment

SCHEMA = "rjmcmc-score-nle-corrected-exploration-summary-v1"
_LEGACY_COMPLETION_KEYS = frozenset(
    {
        "schema",
        "attempt_id",
        "source_git_revision",
        "report",
        "report_payload_sha256",
        "report_file_sha256",
        "artifact_metadata_sha256",
        "serialized_artifact_file_sha256",
        "completion_marker_published_last",
    }
)
_PROMOTION_COMPLETION_KEYS = _LEGACY_COMPLETION_KEYS | frozenset(
    {
        "runtime_identity_sha256",
        "execution_identity_sha256",
    }
)
_COMPLETION_KEYS = _LEGACY_COMPLETION_KEYS


@dataclass(frozen=True)
class _AuthenticatedAttempt:
    report: dict[str, Any]
    report_file_sha256: str
    serialized_artifact_file_sha256: str


def _slug(
    matrix_row: tuple[str, str, int, str, int],
    attempt_tag: str,
) -> str:
    mode, case_id, sample_count, config_id, init_index = matrix_row
    return f"{mode}__{case_id}__S{sample_count}__{config_id}__init{init_index}__{attempt_tag}"


def _sha256_json(payload: object) -> str:
    compact = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(compact.encode("ascii")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_attempt(
    attempt_root: Path,
    *,
    source_git_revision: str,
    matrix_id: str,
    array_task_id: int,
    matrix_row: arrays.MatrixRow,
    attempt_tag: str,
) -> _AuthenticatedAttempt:
    report_path = attempt_root / "report.json"
    complete_path = attempt_root / "COMPLETE.json"
    artifact_path = attempt_root / "artifact.bin"
    if not report_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError("attempt report or completion marker is absent.")
    if report_path.is_symlink() or complete_path.is_symlink():
        raise ValueError("attempt report and completion must be regular files.")
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise ValueError("completed attempt artifact must be a regular file.")
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes.decode("ascii"))
    completion = json.loads(complete_path.read_text(encoding="ascii"))
    promotion = matrix_id in arrays.PROMOTION_MATRIX_SPECS
    expected_completion_keys = _PROMOTION_COMPLETION_KEYS if promotion else _LEGACY_COMPLETION_KEYS
    if set(completion) != expected_completion_keys:
        raise ValueError("attempt completion marker has the wrong exact schema.")
    if completion["schema"] != experiment.SCHEMA:
        raise ValueError("attempt completion marker schema differs.")
    if completion["completion_marker_published_last"] is not True:
        raise ValueError("attempt completion marker was not published last.")
    if completion["source_git_revision"] != source_git_revision:
        raise ValueError("attempt completion marker source revision differs.")
    if completion["report"] != str(report_path):
        raise ValueError("attempt completion marker names the wrong report.")
    report_file_sha256 = hashlib.sha256(report_bytes).hexdigest()
    if completion["report_file_sha256"] != report_file_sha256:
        raise ValueError("attempt report file SHA-256 differs.")
    without_sha = dict(report)
    observed_sha = without_sha.pop("sha256", None)
    if _sha256_json(without_sha) != observed_sha:
        raise ValueError("attempt report SHA-256 does not replay.")
    if report.get("schema") != experiment.SCHEMA:
        raise ValueError("attempt report schema differs.")
    if report.get("source_git_revision") != source_git_revision:
        raise ValueError("attempt source revision differs from merger.")
    if completion["report_payload_sha256"] != observed_sha:
        raise ValueError("attempt completion marker does not bind its report.")
    if completion["attempt_id"] != report.get("attempt_id"):
        raise ValueError("attempt completion marker has the wrong identity.")
    if promotion:
        runtime_identity = report.get("runtime_identity")
        execution_identity = report.get("execution_identity")
        if not isinstance(runtime_identity, dict) or not isinstance(
            execution_identity,
            dict,
        ):
            raise ValueError("promotion runtime/execution identity is absent.")
        runtime_without_sha = dict(runtime_identity)
        runtime_sha = runtime_without_sha.pop("sha256", None)
        execution_without_sha = dict(execution_identity)
        execution_sha = execution_without_sha.pop("sha256", None)
        if (
            _sha256_json(runtime_without_sha) != runtime_sha
            or _sha256_json(execution_without_sha) != execution_sha
            or report.get("runtime_identity_sha256") != runtime_sha
            or report.get("execution_identity_sha256") != execution_sha
            or completion["runtime_identity_sha256"] != runtime_sha
            or completion["execution_identity_sha256"] != execution_sha
            or execution_identity.get("environment") != experiment.PROMOTION_EXECUTION_ENVIRONMENT
            or execution_identity.get("jax_x64_enabled") is not True
        ):
            raise ValueError("promotion runtime/execution identity differs.")
        scientific_target = report.get("scientific_target")
        catalogue_identity = report.get("catalogue_identity")
        target = report.get("target")
        candidate = report.get("candidate")
        domain_evidence_sha256 = report.get("domain_evidence_sha256")
        if (
            not isinstance(scientific_target, dict)
            or not isinstance(catalogue_identity, dict)
            or not isinstance(target, dict)
            or not isinstance(candidate, dict)
            or report.get("scientific_target_sha256") != _sha256_json(scientific_target)
            or report.get("catalogue_identity_sha256") != _sha256_json(catalogue_identity)
            or report.get("candidate_sha256") != _sha256_json(candidate)
            or scientific_target
            != {
                key: target[key]
                for key in (
                    "case_id",
                    "tiny_root_definitions_sha256",
                    "spectrum_sha256",
                    "scientific_input_sha256",
                    "oracle_reference_sha256",
                )
            }
            or catalogue_identity
            != {
                "mode": target["mode"],
                "sample_count": target["sample_count"],
                "domain_evidence_sha256": domain_evidence_sha256,
            }
        ):
            raise ValueError("promotion target/catalogue/candidate identity differs.")
    expected_matrix_identity = arrays.frozen_matrix_identity(
        matrix_id,
        array_task_id,
    )
    if report.get("matrix_identity") != expected_matrix_identity:
        raise ValueError("attempt matrix identity differs from the frozen row.")
    target = report.get("target", {})
    candidate = report.get("candidate", {})
    observed_row = (
        target.get("mode"),
        target.get("case_id"),
        target.get("sample_count"),
        candidate.get("config_id"),
        report.get("initialization_index"),
    )
    if observed_row != matrix_row:
        raise ValueError("attempt report does not map to its frozen matrix row.")
    if report.get("attempt_tag") != attempt_tag:
        raise ValueError("attempt report has the wrong attempt tag.")
    if report.get("execution", {}).get("output_root") != str(attempt_root):
        raise ValueError("attempt report records the wrong output root.")
    if attempt_root.parent.name != "attempts":
        raise ValueError("attempt is not stored below an attempts directory.")
    artifact_file_sha256 = _sha256_file(artifact_path)
    artifact = report.get("artifact", {})
    if artifact_file_sha256 != artifact.get("serialized_sha256"):
        raise ValueError("attempt serialized artifact SHA-256 differs.")
    if artifact_file_sha256 != artifact.get("serialized_artifact_file_sha256"):
        raise ValueError("attempt report artifact file SHA-256 differs.")
    if completion["serialized_artifact_file_sha256"] != artifact_file_sha256:
        raise ValueError("attempt completion artifact file SHA-256 differs.")
    if completion["artifact_metadata_sha256"] != artifact.get("artifact_sha256"):
        raise ValueError("attempt completion marker has the wrong artifact.")
    return _AuthenticatedAttempt(
        report=report,
        report_file_sha256=report_file_sha256,
        serialized_artifact_file_sha256=artifact_file_sha256,
    )


def _finite_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("candidate metrics must be finite and non-empty.")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "sample_sd": (float(np.std(array, ddof=1)) if array.size > 1 else 0.0),
    }


def merge(
    matrix_id: str,
    attempt_tag: str,
    source_git_revision: str,
    output_root: Path,
) -> dict[str, Any]:
    matrix = arrays.MATRICES[matrix_id]
    expected_attempt_count = arrays.EXPECTED_MATRIX_ATTEMPT_COUNTS[matrix_id]
    if len(matrix) != expected_attempt_count:
        raise RuntimeError("frozen matrix length differs from its declared count.")
    reports: list[_AuthenticatedAttempt] = []
    missing: list[str] = []
    failures: list[dict[str, str]] = []
    for array_task_id, row in enumerate(matrix):
        slug = _slug(row, attempt_tag)
        attempt_root = output_root / "attempts" / slug
        if (attempt_root / "FAILURE.json").is_file():
            failures.append(json.loads((attempt_root / "FAILURE.json").read_text(encoding="ascii")))
            continue
        try:
            reports.append(
                _load_attempt(
                    attempt_root,
                    source_git_revision=source_git_revision,
                    matrix_id=matrix_id,
                    array_task_id=array_task_id,
                    matrix_row=row,
                    attempt_tag=attempt_tag,
                )
            )
        except FileNotFoundError:
            missing.append(slug)
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, object]] = []
    for authenticated in reports:
        report = authenticated.report
        target = report["target"]
        candidate = report["candidate"]
        scientific = report["scientific_evaluation"]
        scientific_metrics_interpretable = scientific.get("scientific_metrics_interpretable")
        if type(scientific_metrics_interpretable) is not bool:
            raise ValueError("scientific metric interpretability must be an exact boolean.")
        row = {
            "attempt_id": report["attempt_id"],
            "matrix_identity": report["matrix_identity"],
            "mode": target["mode"],
            "case_id": target["case_id"],
            "sample_count": target["sample_count"],
            "config_id": candidate["config_id"],
            "architecture_family": candidate["architecture_family"],
            "initialization_index": report["initialization_index"],
            "model_selection_nll_nat_per_dimension": report["model_selection"]["nll_nat_per_dimension"],
            "reporting_nll_nat_per_dimension": report["reporting_test"]["nll_nat_per_dimension"],
            "reporting_partial_score_risk": report["reporting_test"]["fisher_scaled_partial_score_risk"],
            "reporting_observation_score_risk": report["reporting_test"][
                "fisher_scaled_observation_score_risk"
            ],
            "scientific_metrics_interpretable": (scientific_metrics_interpretable),
            "absolute_log_evidence_error_nat": (
                scientific["evidence"]["absolute_learned_error_from_adaptive_reference_nat"]
                if scientific_metrics_interpretable
                else None
            ),
            "posterior_weighted_p99_log_likelihood_error_nat": (
                scientific["pointwise"]["exact_posterior_weighted_p99_absolute_log_likelihood_error_nat"]
                if scientific_metrics_interpretable
                else None
            ),
            "posterior_mean_error_reference_sd": (
                scientific["posterior"]["mean_error_reference_sd"]
                if scientific_metrics_interpretable
                else None
            ),
            "posterior_sd_relative_error": (
                scientific["posterior"]["sd_relative_error"] if scientific_metrics_interpretable else None
            ),
            "scaled_gradient_error": (
                scientific["gradient"]["scaled_error"] if scientific_metrics_interpretable else None
            ),
            "runtime_seconds": report["execution"]["runtime_seconds"],
            "maximum_rss_kib": report["execution"]["maximum_rss_kib"],
            "artifact_sha256": report["artifact"]["artifact_sha256"],
            "report_sha256": report["sha256"],
            "report_file_sha256": authenticated.report_file_sha256,
            "serialized_artifact_file_sha256": (authenticated.serialized_artifact_file_sha256),
        }
        rows.append(row)
        group_key = (
            str(target["mode"]),
            str(target["case_id"]),
            int(target["sample_count"]),
            str(candidate["config_id"]),
        )
        grouped[group_key].append(row)
    aggregate: list[dict[str, object]] = []
    operational_metric_names = (
        "model_selection_nll_nat_per_dimension",
        "reporting_nll_nat_per_dimension",
        "reporting_partial_score_risk",
        "reporting_observation_score_risk",
        "runtime_seconds",
        "maximum_rss_kib",
    )
    scientific_metric_names = (
        "absolute_log_evidence_error_nat",
        "posterior_weighted_p99_log_likelihood_error_nat",
        "posterior_mean_error_reference_sd",
        "posterior_sd_relative_error",
        "scaled_gradient_error",
    )
    for key, candidate_rows in sorted(grouped.items()):
        mode, case_id, sample_count, config_id = key
        interpretable_rows = [
            row for row in candidate_rows if row["scientific_metrics_interpretable"] is True
        ]
        metrics: dict[str, dict[str, float] | None] = {
            metric: _finite_summary([float(row[metric]) for row in candidate_rows])
            for metric in operational_metric_names
        }
        metrics.update(
            {
                metric: (
                    _finite_summary([float(row[metric]) for row in interpretable_rows])
                    if interpretable_rows
                    else None
                )
                for metric in scientific_metric_names
            }
        )
        aggregate.append(
            {
                "mode": mode,
                "case_id": case_id,
                "sample_count": sample_count,
                "config_id": config_id,
                "initialization_count": len(candidate_rows),
                "scientific_metrics_interpretable": (len(interpretable_rows) == len(candidate_rows)),
                "scientific_metrics_interpretable_count": len(interpretable_rows),
                "scientific_metrics_non_interpretable_count": (len(candidate_rows) - len(interpretable_rows)),
                "metrics": metrics,
            }
        )
    expected_group_counts = Counter(
        (mode, case_id, sample_count, config_id) for mode, case_id, sample_count, config_id, _ in matrix
    )
    observed_group_counts = {
        (
            str(group["mode"]),
            str(group["case_id"]),
            int(cast(int, group["sample_count"])),
            str(group["config_id"]),
        ): int(cast(int, group["initialization_count"]))
        for group in aggregate
    }
    scientific_metrics_interpretable_count = sum(
        row["scientific_metrics_interpretable"] is True for row in rows
    )
    complete = (
        len(reports) == expected_attempt_count
        and not missing
        and not failures
        and observed_group_counts == dict(expected_group_counts)
    )
    without_sha: dict[str, Any] = {
        "schema": SCHEMA,
        "matrix_id": matrix_id,
        "attempt_tag": attempt_tag,
        "source_git_revision": source_git_revision,
        "expected_attempt_count": expected_attempt_count,
        "complete_attempt_count": len(reports),
        "missing_attempts": missing,
        "failures": failures,
        "rows": rows,
        "candidate_aggregates": aggregate,
        "scientific_metrics_interpretable": (
            bool(rows) and scientific_metrics_interpretable_count == len(rows)
        ),
        "scientific_metrics_interpretable_count": (scientific_metrics_interpretable_count),
        "scientific_metrics_non_interpretable_count": (len(rows) - scientific_metrics_interpretable_count),
        "complete": complete,
        "scientific_decision": (
            "none: exploratory metrics are reported without post-hoc promotion thresholds"
        ),
        "approximate_evidence_used_as_structural_weight": False,
    }
    return {
        **without_sha,
        "sha256": _sha256_json(without_sha),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-id", choices=tuple(arrays.MATRICES), required=True)
    parser.add_argument("--attempt-tag", required=True)
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = merge(
        arguments.matrix_id,
        arguments.attempt_tag,
        arguments.source_git_revision,
        arguments.output_root,
    )
    summary_root = arguments.output_root / "summaries" / f"{arguments.matrix_id}__{arguments.attempt_tag}"
    if summary_root.exists():
        raise FileExistsError("refusing to replace an existing array summary.")
    summary_root.mkdir(parents=True)
    experiment._atomic_json(summary_root / "summary.json", result)
    if not result["complete"]:
        experiment._atomic_json(
            summary_root / "INCOMPLETE.json",
            {
                "schema": SCHEMA,
                "summary_sha256": result["sha256"],
                "completion_marker_published": False,
            },
        )
        raise RuntimeError("array summary is incomplete.")
    experiment._atomic_json(
        summary_root / "COMPLETE.json",
        {
            "schema": SCHEMA,
            "summary_sha256": result["sha256"],
            "completion_marker_published_last": True,
        },
    )


if __name__ == "__main__":
    main()
