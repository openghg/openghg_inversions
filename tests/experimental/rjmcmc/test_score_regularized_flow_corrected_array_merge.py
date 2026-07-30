from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, cast

import pytest

from examples.rjmcmc import (
    score_regularized_flow_corrected_array as array_driver,
)
from examples.rjmcmc import (
    score_regularized_flow_corrected_exploration as experiment,
)
from examples.rjmcmc import (
    score_regularized_flow_corrected_merge as merger,
)


SOURCE_GIT_REVISION = "a" * 40
ATTEMPT_TAG = "test-attempt"


def _write_attempt(
    output_root: Path,
    matrix_id: str,
    array_task_id: int,
    *,
    scientific_metrics_interpretable: bool = True,
    scientific_metric_value: float = 1.0,
) -> Path:
    matrix_row = array_driver.MATRICES[matrix_id][array_task_id]
    mode, case_id, sample_count, config_id, init_index = matrix_row
    attempt_root = output_root / "attempts" / merger._slug(matrix_row, ATTEMPT_TAG)
    attempt_root.mkdir(parents=True)
    artifact_bytes = f"artifact-{matrix_id}-{array_task_id}".encode("ascii")
    artifact_file_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    (attempt_root / "artifact.bin").write_bytes(artifact_bytes)
    report_without_sha: dict[str, Any] = {
        "schema": experiment.SCHEMA,
        "source_git_revision": SOURCE_GIT_REVISION,
        "attempt_id": f"attempt-{matrix_id}-{array_task_id}",
        "attempt_tag": ATTEMPT_TAG,
        "matrix_identity": array_driver.frozen_matrix_identity(
            matrix_id,
            array_task_id,
        ),
        "target": {
            "mode": mode,
            "case_id": case_id,
            "sample_count": sample_count,
        },
        "candidate": {
            "config_id": config_id,
            "architecture_family": "test-flow",
        },
        "initialization_index": init_index,
        "model_selection": {"nll_nat_per_dimension": 1.0},
        "reporting_test": {
            "nll_nat_per_dimension": 2.0,
            "fisher_scaled_partial_score_risk": 3.0,
            "fisher_scaled_observation_score_risk": 4.0,
        },
        "scientific_evaluation": {
            "scientific_metrics_interpretable": (scientific_metrics_interpretable),
            "evidence": {"absolute_learned_error_from_adaptive_reference_nat": (scientific_metric_value)},
            "pointwise": {
                "exact_posterior_weighted_p99_absolute_log_likelihood_error_nat": (scientific_metric_value)
            },
            "posterior": {
                "mean_error_reference_sd": scientific_metric_value,
                "sd_relative_error": scientific_metric_value,
            },
            "gradient": {"scaled_error": scientific_metric_value},
        },
        "execution": {
            "runtime_seconds": 5.0,
            "maximum_rss_kib": 6.0,
            "output_root": str(attempt_root),
        },
        "artifact": {
            "artifact_sha256": artifact_file_sha256,
            "serialized_sha256": artifact_file_sha256,
            "serialized_artifact_file_sha256": artifact_file_sha256,
        },
    }
    report = {
        **report_without_sha,
        "sha256": merger._sha256_json(report_without_sha),
    }
    report_bytes = json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("ascii")
    report_path = attempt_root / "report.json"
    report_path.write_bytes(report_bytes)
    completion = {
        "schema": experiment.SCHEMA,
        "attempt_id": report["attempt_id"],
        "source_git_revision": SOURCE_GIT_REVISION,
        "report": str(report_path),
        "report_payload_sha256": report["sha256"],
        "report_file_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "artifact_metadata_sha256": artifact_file_sha256,
        "serialized_artifact_file_sha256": artifact_file_sha256,
        "completion_marker_published_last": True,
    }
    (attempt_root / "COMPLETE.json").write_text(
        json.dumps(completion, sort_keys=True),
        encoding="ascii",
    )
    return attempt_root


def test_frozen_array_matrices_cover_declared_attempt_counts() -> None:
    assert (
        {name: len(matrix) for name, matrix in array_driver.MATRICES.items()}
        == array_driver.EXPECTED_MATRIX_ATTEMPT_COUNTS
        == {
            "compile_canary": 4,
            "overfit": 16,
            "standard_s4096": 36,
            "observation_canary": 8,
            "standard_s16384_nll": 12,
            "standard_s16384_partial": 12,
            "standard_s16384_pretrain": 12,
        }
    )
    assert set(array_driver.MATRICES["compile_canary"]) == {
        ("overfit", case_id, 256, config_id, 0)
        for case_id in (
            "near_gaussian__two_cell__root",
            "skewed__four_cell__root",
        )
        for config_id in (
            "fisher_partial_joint",
            "fisher_observation_joint",
        )
    }
    expected_16k_configs = {
        "standard_s16384_nll": "nll_only",
        "standard_s16384_partial": "fisher_partial_joint",
        "standard_s16384_pretrain": "nll_pretrain_then_partial",
    }
    for matrix_id, config_id in expected_16k_configs.items():
        matrix = array_driver.MATRICES[matrix_id]
        assert {row[2] for row in matrix} == {16_384}
        assert {row[3] for row in matrix} == {config_id}
        assert {row[4] for row in matrix} == set(range(4))
        assert {row[1] for row in matrix} == set(experiment.SELECTED_CASES)
    for matrix in array_driver.MATRICES.values():
        assert len(set(matrix)) == len(matrix)


def test_array_passes_exact_frozen_matrix_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(arguments: object, attempt_root: Path) -> None:
        captured["arguments"] = arguments
        captured["attempt_root"] = attempt_root

    matrix_id = "compile_canary"
    task_id = 3
    monkeypatch.setattr(experiment, "run_attempt", capture)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(task_id))
    monkeypatch.setenv(
        "SLURM_ARRAY_TASK_COUNT",
        str(len(array_driver.MATRICES[matrix_id])),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_regularized_flow_corrected_array.py",
            "--matrix-id",
            matrix_id,
            "--attempt-tag",
            ATTEMPT_TAG,
            "--source-git-revision",
            SOURCE_GIT_REVISION,
            "--oracle-bundle",
            str(tmp_path / "oracle.json"),
            "--output-root",
            str(tmp_path),
        ],
    )

    array_driver.main()

    arguments = cast(argparse.Namespace, captured["arguments"])
    identity = array_driver.frozen_matrix_identity(matrix_id, task_id)
    assert arguments.matrix_id == matrix_id
    assert arguments.array_task_id == task_id
    assert arguments.array_task_count == 4
    assert arguments.matrix_task_id == task_id
    assert arguments.matrix_task_count == 4
    assert arguments.matrix_row == array_driver.MATRICES[matrix_id][task_id]
    assert arguments.matrix_identity == identity


def test_merger_authenticates_exact_files_schema_and_matrix_mapping(
    tmp_path: Path,
) -> None:
    attempt_root = _write_attempt(tmp_path, "compile_canary", 0)
    row = array_driver.MATRICES["compile_canary"][0]

    loaded = merger._load_attempt(
        attempt_root,
        source_git_revision=SOURCE_GIT_REVISION,
        matrix_id="compile_canary",
        array_task_id=0,
        matrix_row=row,
        attempt_tag=ATTEMPT_TAG,
    )

    assert loaded.report["matrix_identity"] == (array_driver.frozen_matrix_identity("compile_canary", 0))
    assert loaded.report_file_sha256 == merger._sha256_file(attempt_root / "report.json")
    assert loaded.serialized_artifact_file_sha256 == merger._sha256_file(attempt_root / "artifact.bin")

    completion_path = attempt_root / "COMPLETE.json"
    completion = json.loads(completion_path.read_text(encoding="ascii"))
    completion["unexpected"] = True
    completion_path.write_text(json.dumps(completion), encoding="ascii")
    with pytest.raises(ValueError, match="wrong exact schema"):
        merger._load_attempt(
            attempt_root,
            source_git_revision=SOURCE_GIT_REVISION,
            matrix_id="compile_canary",
            array_task_id=0,
            matrix_row=row,
            attempt_tag=ATTEMPT_TAG,
        )


def test_merger_rejects_file_or_frozen_row_tampering(tmp_path: Path) -> None:
    attempt_root = _write_attempt(tmp_path, "compile_canary", 0)
    row = array_driver.MATRICES["compile_canary"][0]
    artifact_path = attempt_root / "artifact.bin"
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="artifact"):
        merger._load_attempt(
            attempt_root,
            source_git_revision=SOURCE_GIT_REVISION,
            matrix_id="compile_canary",
            array_task_id=0,
            matrix_row=row,
            attempt_tag=ATTEMPT_TAG,
        )

    report_root = tmp_path / "report-tamper"
    report_attempt = _write_attempt(report_root, "compile_canary", 0)
    report_path = report_attempt / "report.json"
    report_path.write_bytes(report_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="report file SHA-256"):
        merger._load_attempt(
            report_attempt,
            source_git_revision=SOURCE_GIT_REVISION,
            matrix_id="compile_canary",
            array_task_id=0,
            matrix_row=row,
            attempt_tag=ATTEMPT_TAG,
        )

    other_root = tmp_path / "other"
    other_attempt = _write_attempt(other_root, "compile_canary", 0)
    with pytest.raises(ValueError, match="matrix identity"):
        merger._load_attempt(
            other_attempt,
            source_git_revision=SOURCE_GIT_REVISION,
            matrix_id="compile_canary",
            array_task_id=1,
            matrix_row=array_driver.MATRICES["compile_canary"][1],
            attempt_tag=ATTEMPT_TAG,
        )


def test_merger_never_aggregates_non_interpretable_scientific_metrics(
    tmp_path: Path,
) -> None:
    matrix_id = "observation_canary"
    for task_id, row in enumerate(array_driver.MATRICES[matrix_id]):
        interpretable = row[1] != "near_gaussian__two_cell__root" or (row[4] in (0, 2))
        metric_value = float(row[4] + 1) if interpretable else 10_000.0
        _write_attempt(
            tmp_path,
            matrix_id,
            task_id,
            scientific_metrics_interpretable=interpretable,
            scientific_metric_value=metric_value,
        )

    summary = merger.merge(
        matrix_id,
        ATTEMPT_TAG,
        SOURCE_GIT_REVISION,
        tmp_path,
    )

    assert summary["complete"] is True
    assert summary["scientific_metrics_interpretable"] is False
    assert summary["scientific_metrics_interpretable_count"] == 6
    assert summary["scientific_metrics_non_interpretable_count"] == 2
    non_interpretable_rows = [row for row in summary["rows"] if not row["scientific_metrics_interpretable"]]
    assert len(non_interpretable_rows) == 2
    assert all(row["absolute_log_evidence_error_nat"] is None for row in non_interpretable_rows)
    near_gaussian = next(
        aggregate
        for aggregate in summary["candidate_aggregates"]
        if aggregate["case_id"] == "near_gaussian__two_cell__root"
    )
    assert near_gaussian["scientific_metrics_interpretable"] is False
    assert near_gaussian["scientific_metrics_interpretable_count"] == 2
    assert near_gaussian["scientific_metrics_non_interpretable_count"] == 2
    assert near_gaussian["metrics"]["absolute_log_evidence_error_nat"]["mean"] == pytest.approx(2.0)
