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
    promotion = matrix_id in array_driver.PROMOTION_MATRIX_SPECS
    target = {
        "mode": mode,
        "case_id": case_id,
        "sample_count": sample_count,
    }
    candidate = {
        "config_id": config_id,
        "architecture_family": "test-flow",
    }
    if promotion:
        target.update(
            {
                "tiny_root_definitions_sha256": "definitions",
                "spectrum_sha256": "spectrum",
                "scientific_input_sha256": "scientific-input",
                "oracle_reference_sha256": "oracle-reference",
            }
        )
        spec = array_driver.PROMOTION_MATRIX_SPECS[matrix_id]
        candidate.update(
            {
                "learning_rate": spec["learning_rate"],
                "batch_size": spec["batch_size"],
                "microbatch_size": spec["microbatch_size"],
                "maximum_total_epochs": spec["maximum_total_epochs"],
                "patience": spec["patience"],
            }
        )
    report_without_sha: dict[str, Any] = {
        "schema": experiment.SCHEMA,
        "source_git_revision": SOURCE_GIT_REVISION,
        "attempt_id": f"attempt-{matrix_id}-{array_task_id}",
        "attempt_tag": ATTEMPT_TAG,
        "matrix_identity": array_driver.frozen_matrix_identity(
            matrix_id,
            array_task_id,
        ),
        "target": target,
        "candidate": candidate,
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
    if promotion:
        runtime_without_sha = {
            "schema": "test-runtime-identity",
            "python_version": "3.10.0",
        }
        execution_without_sha = {
            "schema": "test-execution-identity",
            "environment": experiment.PROMOTION_EXECUTION_ENVIRONMENT,
            "jax_x64_enabled": True,
        }
        runtime_identity = {
            **runtime_without_sha,
            "sha256": merger._sha256_json(runtime_without_sha),
        }
        execution_identity = {
            **execution_without_sha,
            "sha256": merger._sha256_json(execution_without_sha),
        }
        domain_evidence_sha256 = {domain: f"domain-{domain}" for domain in experiment.domains.PUBLIC_DOMAINS}
        scientific_target = {
            key: target[key]
            for key in (
                "case_id",
                "tiny_root_definitions_sha256",
                "spectrum_sha256",
                "scientific_input_sha256",
                "oracle_reference_sha256",
            )
        }
        catalogue_identity = {
            "mode": mode,
            "sample_count": sample_count,
            "domain_evidence_sha256": domain_evidence_sha256,
        }
        report_without_sha.update(
            {
                "runtime_identity": runtime_identity,
                "execution_identity": execution_identity,
                "runtime_identity_sha256": runtime_identity["sha256"],
                "execution_identity_sha256": execution_identity["sha256"],
                "scientific_target": scientific_target,
                "scientific_target_sha256": merger._sha256_json(scientific_target),
                "catalogue_identity": catalogue_identity,
                "catalogue_identity_sha256": merger._sha256_json(catalogue_identity),
                "candidate_sha256": merger._sha256_json(candidate),
                "domain_evidence_sha256": domain_evidence_sha256,
            }
        )
        report_without_sha["execution"].update(
            {
                "runtime_identity_sha256": runtime_identity["sha256"],
                "execution_identity_sha256": execution_identity["sha256"],
            }
        )
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
    if promotion:
        completion.update(
            {
                "runtime_identity_sha256": report["runtime_identity_sha256"],
                "execution_identity_sha256": report["execution_identity_sha256"],
            }
        )
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
            "overfit_q3_extended": 4,
            "standard_s4096": 36,
            "observation_canary": 8,
            "standard_s16384_nll": 12,
            "standard_s16384_partial": 12,
            "standard_s16384_pretrain": 12,
            "promotion_development_s4096": 48,
            "promotion_development_s16384": 48,
            "promotion_confirmation_s16384_seed2731": 24,
            "promotion_confirmation_s16384_seed3731": 24,
            "promotion_confirmation_s16384_seed4731": 24,
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
    assert set(array_driver.MATRICES["overfit_q3_extended"]) == {
        (
            "overfit",
            "skewed__four_cell__root",
            256,
            "nll_only",
            init_index,
        )
        for init_index in range(4)
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


def test_legacy_e2_matrix_rows_remain_exact() -> None:
    expected = {
        "compile_canary": tuple(
            ("overfit", case_id, 256, config_id, 0)
            for case_id in (
                "near_gaussian__two_cell__root",
                "skewed__four_cell__root",
            )
            for config_id in (
                "fisher_partial_joint",
                "fisher_observation_joint",
            )
        ),
        "overfit": tuple(
            ("overfit", case_id, 256, config_id, init_index)
            for case_id in (
                "near_gaussian__two_cell__root",
                "skewed__four_cell__root",
            )
            for config_id in ("nll_only", "fisher_partial_joint")
            for init_index in range(4)
        ),
        "overfit_q3_extended": tuple(
            (
                "overfit",
                "skewed__four_cell__root",
                256,
                "nll_only",
                init_index,
            )
            for init_index in range(4)
        ),
        "standard_s4096": tuple(
            ("standard", case_id, 4_096, config_id, init_index)
            for case_id in experiment.SELECTED_CASES
            for config_id in (
                "nll_only",
                "fisher_partial_joint",
                "nll_pretrain_then_partial",
            )
            for init_index in range(4)
        ),
        "observation_canary": tuple(
            (
                "standard",
                case_id,
                4_096,
                "fisher_observation_joint",
                init_index,
            )
            for case_id in (
                "near_gaussian__two_cell__root",
                "skewed__four_cell__root",
            )
            for init_index in range(4)
        ),
        "standard_s16384_nll": tuple(
            ("standard", case_id, 16_384, "nll_only", init_index)
            for case_id in experiment.SELECTED_CASES
            for init_index in range(4)
        ),
        "standard_s16384_partial": tuple(
            (
                "standard",
                case_id,
                16_384,
                "fisher_partial_joint",
                init_index,
            )
            for case_id in experiment.SELECTED_CASES
            for init_index in range(4)
        ),
        "standard_s16384_pretrain": tuple(
            (
                "standard",
                case_id,
                16_384,
                "nll_pretrain_then_partial",
                init_index,
            )
            for case_id in experiment.SELECTED_CASES
            for init_index in range(4)
        ),
    }
    assert {matrix_id: array_driver.MATRICES[matrix_id] for matrix_id in expected} == expected
    assert all(
        "promotion_spec" not in array_driver.frozen_matrix_identity(matrix_id, 0) for matrix_id in expected
    )


def test_promotion_matrices_freeze_cases_seeds_and_hyperparameters() -> None:
    development = {
        "promotion_development_s4096": 4_096,
        "promotion_development_s16384": 16_384,
    }
    confirmation = {
        "promotion_confirmation_s16384_seed2731": 2731,
        "promotion_confirmation_s16384_seed3731": 3731,
        "promotion_confirmation_s16384_seed4731": 4731,
    }
    common = {
        "learning_rate": 5.0e-4,
        "batch_size": 1024,
        "microbatch_size": 64,
        "maximum_total_epochs": 40,
        "patience": 6,
    }
    for matrix_id, sample_count in development.items():
        rows = array_driver.MATRICES[matrix_id]
        assert {row[0] for row in rows} == {"promotion"}
        assert {row[1] for row in rows} == set(experiment.PROMOTION_CASES)
        assert {row[2] for row in rows} == {sample_count}
        assert {row[3] for row in rows} == {
            "nll_only",
            "fisher_observation_joint",
        }
        assert {row[4] for row in rows} == set(range(4))
        assert array_driver.PROMOTION_MATRIX_SPECS[matrix_id] == {
            **common,
            "base_seed": 1731,
            "role": "development-size-stability",
        }
    for matrix_id, base_seed in confirmation.items():
        rows = array_driver.MATRICES[matrix_id]
        assert {row[0] for row in rows} == {"promotion"}
        assert {row[1] for row in rows} == set(experiment.PROMOTION_CASES)
        assert {row[2] for row in rows} == {16_384}
        assert {row[3] for row in rows} == {"fisher_observation_joint"}
        assert {row[4] for row in rows} == set(range(4))
        assert array_driver.PROMOTION_MATRIX_SPECS[matrix_id] == {
            **common,
            "base_seed": base_seed,
            "role": "independent-confirmation",
        }
    for matrix_id in array_driver.PROMOTION_MATRIX_SPECS:
        identity = array_driver.frozen_matrix_identity(matrix_id, 0)
        assert identity["promotion_spec"] == (array_driver.PROMOTION_MATRIX_SPECS[matrix_id])


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


def test_promotion_array_derives_frozen_spec_and_ignores_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    matrix_id = "promotion_confirmation_s16384_seed2731"
    task_id = 0
    spec = array_driver.PROMOTION_MATRIX_SPECS[matrix_id]
    monkeypatch.setattr(
        experiment,
        "run_attempt",
        lambda arguments, attempt_root: captured.update(
            arguments=arguments,
            attempt_root=attempt_root,
        ),
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(task_id))
    monkeypatch.setenv(
        "SLURM_ARRAY_TASK_COUNT",
        str(len(array_driver.MATRICES[matrix_id])),
    )

    def arguments(*, base_seed: int, output_root: Path = tmp_path) -> list[str]:
        return [
            "score_regularized_flow_corrected_array.py",
            "--matrix-id",
            matrix_id,
            "--attempt-tag",
            ATTEMPT_TAG,
            "--base-seed",
            str(base_seed),
            "--source-git-revision",
            SOURCE_GIT_REVISION,
            "--oracle-bundle",
            str(tmp_path / "oracle.json"),
            "--output-root",
            str(output_root),
            "--learning-rate",
            str(spec["learning_rate"]),
            "--batch-size",
            str(spec["batch_size"]),
            "--microbatch-size",
            str(spec["microbatch_size"]),
            "--max-epochs",
            str(spec["maximum_total_epochs"]),
            "--patience",
            str(spec["patience"]),
        ]

    monkeypatch.setattr(sys, "argv", arguments(base_seed=2731))
    array_driver.main()
    attempt_arguments = cast(argparse.Namespace, captured["arguments"])
    assert attempt_arguments.base_seed == 2731
    assert attempt_arguments.matrix_identity == (array_driver.frozen_matrix_identity(matrix_id, task_id))

    override_root = tmp_path / "override"
    monkeypatch.setattr(
        sys,
        "argv",
        arguments(base_seed=2732, output_root=override_root),
    )
    array_driver.main()
    overridden_arguments = cast(argparse.Namespace, captured["arguments"])
    assert overridden_arguments.base_seed == 2731
    assert overridden_arguments.learning_rate == spec["learning_rate"]
    assert overridden_arguments.batch_size == spec["batch_size"]
    assert overridden_arguments.microbatch_size == spec["microbatch_size"]
    assert overridden_arguments.max_epochs == spec["maximum_total_epochs"]
    assert overridden_arguments.patience == spec["patience"]


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


def test_promotion_runtime_execution_and_completion_identities_are_bound(
    tmp_path: Path,
) -> None:
    matrix_id = "promotion_development_s4096"
    task_id = 0
    attempt_root = _write_attempt(tmp_path, matrix_id, task_id)
    row = array_driver.MATRICES[matrix_id][task_id]

    loaded = merger._load_attempt(
        attempt_root,
        source_git_revision=SOURCE_GIT_REVISION,
        matrix_id=matrix_id,
        array_task_id=task_id,
        matrix_row=row,
        attempt_tag=ATTEMPT_TAG,
    )

    runtime = loaded.report["runtime_identity"]
    execution = loaded.report["execution_identity"]
    completion = json.loads((attempt_root / "COMPLETE.json").read_text(encoding="ascii"))
    assert set(completion) == merger._PROMOTION_COMPLETION_KEYS
    assert runtime["sha256"] == loaded.report["runtime_identity_sha256"]
    assert execution["sha256"] == loaded.report["execution_identity_sha256"]
    assert completion["runtime_identity_sha256"] == runtime["sha256"]
    assert completion["execution_identity_sha256"] == execution["sha256"]
    assert execution["environment"] == (experiment.PROMOTION_EXECUTION_ENVIRONMENT)

    completion["runtime_identity_sha256"] = "0" * 64
    (attempt_root / "COMPLETE.json").write_text(
        json.dumps(completion, sort_keys=True),
        encoding="ascii",
    )
    with pytest.raises(
        ValueError,
        match="promotion runtime/execution identity differs",
    ):
        merger._load_attempt(
            attempt_root,
            source_git_revision=SOURCE_GIT_REVISION,
            matrix_id=matrix_id,
            array_task_id=task_id,
            matrix_row=row,
            attempt_tag=ATTEMPT_TAG,
        )


def test_promotion_execution_identity_rejects_rehashed_environment_mutation(
    tmp_path: Path,
) -> None:
    matrix_id = "promotion_development_s4096"
    task_id = 0
    attempt_root = _write_attempt(tmp_path, matrix_id, task_id)
    report_path = attempt_root / "report.json"
    completion_path = attempt_root / "COMPLETE.json"
    report = json.loads(report_path.read_text(encoding="ascii"))
    completion = json.loads(completion_path.read_text(encoding="ascii"))
    execution = dict(report["execution_identity"])
    execution_without_sha = dict(execution)
    execution_without_sha.pop("sha256")
    execution_without_sha["environment"] = {
        **execution_without_sha["environment"],
        "OMP_NUM_THREADS": "2",
    }
    execution = {
        **execution_without_sha,
        "sha256": merger._sha256_json(execution_without_sha),
    }
    report["execution_identity"] = execution
    report["execution_identity_sha256"] = execution["sha256"]
    report["execution"]["execution_identity_sha256"] = execution["sha256"]
    report_without_sha = dict(report)
    report_without_sha.pop("sha256")
    report["sha256"] = merger._sha256_json(report_without_sha)
    report_bytes = json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("ascii")
    report_path.write_bytes(report_bytes)
    completion["execution_identity_sha256"] = execution["sha256"]
    completion["report_payload_sha256"] = report["sha256"]
    completion["report_file_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    completion_path.write_text(
        json.dumps(completion, sort_keys=True),
        encoding="ascii",
    )

    with pytest.raises(
        ValueError,
        match="promotion runtime/execution identity differs",
    ):
        merger._load_attempt(
            attempt_root,
            source_git_revision=SOURCE_GIT_REVISION,
            matrix_id=matrix_id,
            array_task_id=task_id,
            matrix_row=array_driver.MATRICES[matrix_id][task_id],
            attempt_tag=ATTEMPT_TAG,
        )


def test_legacy_completion_schema_remains_exact(
    tmp_path: Path,
) -> None:
    matrix_id = "compile_canary"
    task_id = 0
    attempt_root = _write_attempt(tmp_path, matrix_id, task_id)
    completion_path = attempt_root / "COMPLETE.json"
    completion = json.loads(completion_path.read_text(encoding="ascii"))
    assert set(completion) == merger._LEGACY_COMPLETION_KEYS
    assert "runtime_identity_sha256" not in completion
    assert "execution_identity_sha256" not in completion
    merger._load_attempt(
        attempt_root,
        source_git_revision=SOURCE_GIT_REVISION,
        matrix_id=matrix_id,
        array_task_id=task_id,
        matrix_row=array_driver.MATRICES[matrix_id][task_id],
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
