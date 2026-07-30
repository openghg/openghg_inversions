#!/usr/bin/env python3
"""Publish one cross-size/all-seed corrected NLE promotion certificate."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import special, stats

from examples.rjmcmc import score_regularized_flow_corrected_exploration as experiment
from examples.rjmcmc import score_regularized_flow_corrected_promotion_merge as promotion
from openghg_inversions.experimental.rjmcmc import aggregation_error_tiny_oracle

SCHEMA = "rjmcmc-score-nle-corrected-promotion-certificate-v2"
DEVELOPMENT_MATRIX_IDS = frozenset(
    {
        "promotion_development_s4096",
        "promotion_development_s16384",
    }
)
FINAL_MATRIX_IDS = DEVELOPMENT_MATRIX_IDS | promotion.CONFIRMATION_MATRIX_IDS
CROSS_SIZE_THRESHOLDS = {
    "prior_weighted_median_absolute_log_likelihood_difference_nat": 0.05,
    "posterior_weighted_p99_absolute_log_likelihood_difference_nat": 0.20,
}
_SUMMARY_COMPLETION_KEYS = frozenset(
    {
        "schema",
        "artifact_source_git_revision",
        "evaluation_source_git_revision",
        "matrix_id",
        "attempt_tag",
        "summary_path",
        "summary_payload_sha256",
        "summary_file_sha256",
        "promotion_pass",
        "completion_marker_published_last",
    }
)


def _logsumexp_scalar(values: np.ndarray) -> float:
    return float(cast(Any, special.logsumexp)(values))


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


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(
        int(np.searchsorted(cumulative, probability, side="left")),
        values.size - 1,
    )
    return float(values[order[index]])


def _load_summary(
    summary_root: Path,
    *,
    artifact_source_git_revision: str,
    evaluation_source_git_revision: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    summary_path = summary_root / "summary.json"
    complete_path = summary_root / "COMPLETE.json"
    if (
        not summary_path.is_file()
        or summary_path.is_symlink()
        or not complete_path.is_file()
        or complete_path.is_symlink()
    ):
        raise ValueError("promotion summary and completion must be regular files.")
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes.decode("ascii"))
    completion = json.loads(complete_path.read_text(encoding="ascii"))
    if set(completion) != _SUMMARY_COMPLETION_KEYS:
        raise ValueError("promotion summary completion has the wrong exact schema.")
    without_sha = dict(summary)
    payload_sha = without_sha.pop("sha256", None)
    file_sha = hashlib.sha256(summary_bytes).hexdigest()
    if (
        completion["schema"] != promotion.SCHEMA
        or completion["artifact_source_git_revision"] != artifact_source_git_revision
        or completion["evaluation_source_git_revision"] != evaluation_source_git_revision
        or completion["matrix_id"] != summary.get("matrix_id")
        or completion["attempt_tag"] != summary.get("attempt_tag")
        or completion["summary_path"] != str(summary_path)
        or completion["summary_payload_sha256"] != payload_sha
        or completion["summary_file_sha256"] != file_sha
        or completion["promotion_pass"] != summary.get("promotion_pass")
        or completion["completion_marker_published_last"] is not True
        or summary.get("schema") != promotion.SCHEMA
        or summary.get("artifact_source_git_revision") != artifact_source_git_revision
        or summary.get("evaluation_source_git_revision") != evaluation_source_git_revision
        or _sha256_json(without_sha) != payload_sha
    ):
        raise ValueError("promotion summary identity does not replay.")
    return summary, {
        "summary_root": str(summary_root),
        "summary_path": str(summary_path),
        "completion_path": str(complete_path),
        "summary_payload_sha256": str(payload_sha),
        "summary_file_sha256": file_sha,
        "completion_file_sha256": _sha256_file(complete_path),
    }


def _primary_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {
        str(row["case_id"]): row
        for row in summary["selected_rows"]
        if row["config_id"] == promotion.PRIMARY_CONFIG_ID
    }
    if set(rows) != set(experiment.PROMOTION_CASES):
        raise ValueError("promotion summary lacks one all-six primary row.")
    return rows


def _artifact(row: dict[str, Any]) -> experiment.ScoreRegularizedRootFlow:
    path = Path(str(row["selected_artifact_path"]))
    expected_sha = str(row["selected_artifact_file_sha256"])
    if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected_sha:
        raise ValueError("selected artifact no longer matches its promotion summary.")
    return experiment.ScoreRegularizedRootFlow.from_bytes(
        path.read_bytes(),
        expected_sha256=expected_sha,
    )


def _cross_size_row(
    case_id: str,
    small_row: dict[str, Any],
    large_row: dict[str, Any],
    oracle_case: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one cross-size case inside one disposable process."""
    small_artifact = _artifact(small_row)
    large_artifact = _artifact(large_row)
    case = aggregation_error_tiny_oracle.tiny_root_case(case_id)
    shapes, rate, _, observation, _ = case.arrays()
    count = experiment.GRID_COUNTS[-1]
    totals = np.asarray(
        stats.gamma.ppf(
            (np.arange(count, dtype=np.float64) + 0.5) / count,
            a=float(shapes.sum()),
            scale=1.0 / rate,
        ),
        dtype=np.float64,
    )
    small = experiment._vectorized_artifact_log_likelihood(
        small_artifact,
        observation,
        totals,
    )
    large = experiment._vectorized_artifact_log_likelihood(
        large_artifact,
        observation,
        totals,
    )
    difference = np.abs(small - large)
    exact = np.asarray(
        aggregation_error_tiny_oracle.root_conditional_log_likelihood(
            case_id,
            totals,
            fraction_order=int(oracle_case["reference"]["fraction_order"]),
        ),
        dtype=np.float64,
    )
    preflight_rows = {int(row["count"]): row for row in oracle_case["metric_grid_preflight"]["rows"]}
    expected_grid = preflight_rows[count]
    if (
        expected_grid["total_grid_sha256"]
        != hashlib.sha256(np.ascontiguousarray(totals, dtype="<f8").tobytes()).hexdigest()
        or expected_grid["exact_log_likelihood_sha256"]
        != hashlib.sha256(np.ascontiguousarray(exact, dtype="<f8").tobytes()).hexdigest()
    ):
        raise ValueError("cross-size exact grid differs from the oracle preflight.")
    exact_weights = np.exp(exact - _logsumexp_scalar(exact))
    prior_weights = np.full(count, 1.0 / count)
    metrics = {
        "prior_weighted_median_absolute_log_likelihood_difference_nat": (
            _weighted_quantile(difference, prior_weights, 0.5)
        ),
        "posterior_weighted_p99_absolute_log_likelihood_difference_nat": (
            _weighted_quantile(difference, exact_weights, 0.99)
        ),
        "absolute_log_evidence_difference_nat": abs(_logsumexp_scalar(small) - _logsumexp_scalar(large)),
    }
    checks = {key: metrics[key] <= threshold for key, threshold in CROSS_SIZE_THRESHOLDS.items()}
    return {
        "case_id": case_id,
        "grid_count": count,
        "small_selected_artifact_file_sha256": (small_row["selected_artifact_file_sha256"]),
        "large_selected_artifact_file_sha256": (large_row["selected_artifact_file_sha256"]),
        "metrics": metrics,
        "checks": checks,
        "pass": all(checks.values()),
        "evidence_difference_role": ("leakage diagnostic only; never a structural weight"),
    }


def _cross_size_row_isolated(
    case_id: str,
    small_row: dict[str, Any],
    large_row: dict[str, Any],
    oracle_case: dict[str, Any],
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        return executor.submit(
            _cross_size_row,
            case_id,
            small_row,
            large_row,
            oracle_case,
        ).result()


def _cross_size_rows(
    small_summary: dict[str, Any],
    large_summary: dict[str, Any],
    oracle_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    small_rows = _primary_rows(small_summary)
    large_rows = _primary_rows(large_summary)
    rows: list[dict[str, Any]] = []
    for case_id in experiment.PROMOTION_CASES:
        row = _cross_size_row_isolated(
            case_id,
            small_rows[case_id],
            large_rows[case_id],
            oracle_bundle["selected_cases"][case_id],
        )
        if (
            row.get("case_id") != case_id
            or row.get("small_selected_artifact_file_sha256")
            != small_rows[case_id]["selected_artifact_file_sha256"]
            or row.get("large_selected_artifact_file_sha256")
            != large_rows[case_id]["selected_artifact_file_sha256"]
        ):
            raise ValueError("isolated cross-size row does not bind its selected inputs.")
        rows.append(row)
    return rows


def certify(
    summary_roots: list[Path],
    *,
    artifact_source_git_revision: str,
    evaluation_source_git_revision: str,
    output_root: Path,
) -> dict[str, Any]:
    """Authenticate the exact development pair or the full five matrices."""
    loaded = [
        _load_summary(
            root,
            artifact_source_git_revision=artifact_source_git_revision,
            evaluation_source_git_revision=evaluation_source_git_revision,
        )
        for root in summary_roots
    ]
    summaries = {str(summary["matrix_id"]): summary for summary, _ in loaded}
    if len(summaries) != len(loaded):
        raise ValueError("certificate inputs contain duplicate matrix IDs.")
    observed = frozenset(summaries)
    if observed == DEVELOPMENT_MATRIX_IDS:
        phase = "development"
    elif observed == FINAL_MATRIX_IDS:
        phase = "final_confirmation"
    else:
        raise ValueError("certificate inputs are not the frozen development or final set.")
    oracle_bundle = experiment._load_oracle_bundle(
        output_root / "oracle" / "oracle_bundle.json",
        artifact_source_git_revision,
        promotion=True,
    )
    small = summaries["promotion_development_s4096"]
    large = summaries["promotion_development_s16384"]
    cross_size = _cross_size_rows(small, large, oracle_bundle)
    matrix_checks = {
        matrix_id: (summary.get("complete") is True and summary.get("promotion_pass") is True)
        for matrix_id, summary in sorted(summaries.items())
    }
    confirmation_seeds = {
        int(summary["expected_base_seed"])
        for matrix_id, summary in summaries.items()
        if matrix_id in promotion.CONFIRMATION_MATRIX_IDS
    }
    checks = {
        "all_input_matrix_summaries_pass": all(matrix_checks.values()),
        "common_all_six_cross_size_stability": all(row["pass"] for row in cross_size),
        "development_sizes_are_exactly_4096_and_16384": {
            int(row["sample_count"]) for summary in (small, large) for row in _primary_rows(summary).values()
        }
        == {4_096, 16_384},
        "confirmation_seed_set_is_frozen": (
            phase == "development" or confirmation_seeds == {2_731, 3_731, 4_731}
        ),
    }
    certificate_pass = all(checks.values())
    without_sha: dict[str, Any] = {
        "schema": SCHEMA,
        "artifact_source_git_revision": artifact_source_git_revision,
        "evaluation_source_git_revision": evaluation_source_git_revision,
        "phase": phase,
        "input_matrix_ids": sorted(summaries),
        "input_summary_identities": [
            {
                "matrix_id": summary["matrix_id"],
                "attempt_tag": summary["attempt_tag"],
                **identity,
            }
            for summary, identity in sorted(
                loaded,
                key=lambda item: str(item[0]["matrix_id"]),
            )
        ],
        "matrix_checks": matrix_checks,
        "cross_size_thresholds": CROSS_SIZE_THRESHOLDS,
        "cross_size_rows": cross_size,
        "checks": checks,
        "certificate_pass": certificate_pass,
        "eligible_for_confirmation": (phase == "development" and certificate_pass),
        "promotion_pass": (phase == "final_confirmation" and certificate_pass),
        "scientific_invariant": (
            "one common native model; exact marginal invariant to partition and K; "
            "approximate evidence differences are leakage diagnostics only"
        ),
    }
    return {**without_sha, "sha256": _sha256_json(without_sha)}


def _publish(
    certificate: dict[str, Any],
    *,
    certificate_root: Path,
) -> None:
    if certificate_root.exists() or certificate_root.is_symlink():
        raise FileExistsError("refusing to replace an existing promotion certificate.")
    certificate_root.mkdir(parents=True)
    report_path = certificate_root / "certificate.json"
    report_bytes = experiment._json_bytes(certificate)
    report_file_sha256 = hashlib.sha256(report_bytes).hexdigest()
    experiment._atomic_json(report_path, certificate)
    experiment._atomic_json(
        certificate_root / "COMPLETE.json",
        {
            "schema": SCHEMA,
            "artifact_source_git_revision": certificate["artifact_source_git_revision"],
            "evaluation_source_git_revision": certificate["evaluation_source_git_revision"],
            "phase": certificate["phase"],
            "certificate_path": str(report_path),
            "certificate_payload_sha256": certificate["sha256"],
            "certificate_file_sha256": report_file_sha256,
            "certificate_pass": certificate["certificate_pass"],
            "completion_marker_published_last": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-root", type=Path, action="append", required=True)
    parser.add_argument("--artifact-source-git-revision", required=True)
    parser.add_argument("--evaluation-source-git-revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--certificate-tag", required=True)
    arguments = parser.parse_args()
    certificate = certify(
        arguments.summary_root,
        artifact_source_git_revision=arguments.artifact_source_git_revision,
        evaluation_source_git_revision=arguments.evaluation_source_git_revision,
        output_root=arguments.output_root,
    )
    certificate_root = (
        arguments.output_root
        / "promotion_certificates"
        / f"{certificate['phase']}__{arguments.certificate_tag}"
    )
    _publish(certificate, certificate_root=certificate_root)
    if not certificate["certificate_pass"]:
        raise RuntimeError("corrected promotion certificate is a scientific hard stop.")


if __name__ == "__main__":
    main()
