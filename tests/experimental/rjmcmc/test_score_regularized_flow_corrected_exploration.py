"""Tests for the corrected array-friendly exploration driver."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from examples.rjmcmc import (
    score_regularized_flow_corrected_array as array_driver,
    score_regularized_flow_corrected_exploration as driver,
    score_regularized_flow_corrected_merge as merger,
    score_regularized_flow_corrected_oracle as oracle_driver,
)
from openghg_inversions.experimental.rjmcmc import aggregation_error_tiny_oracle


def _reference(case_id: str) -> dict[str, object]:
    without_sha = {
        "schema": aggregation_error_tiny_oracle.SCHEMA,
        "case_id": case_id,
        "definitions_sha256": aggregation_error_tiny_oracle.definitions_sha256(),
        "method": "test_reference",
        "fraction_order": 4,
        "lower_log_total": -5.0,
        "upper_log_total": 3.0,
        "epsabs": 1.0e-8,
        "epsrel": 1.0e-8,
        "posterior_mode_total": 1.0,
        "log_evidence": -2.0,
        "posterior_mean_total": 1.0,
        "posterior_sd_total": 0.2,
        "posterior_lower_0_025": 0.6,
        "posterior_median": 1.0,
        "posterior_upper_0_975": 1.4,
        "scaled_quadrature_error": 1.0e-10,
        "represented_prior_mass": 1.0,
        "represented_posterior_mass": 1.0,
        "posterior_mass_accounting": "test",
        "mode_included": True,
    }
    return {
        **without_sha,
        "sha256": driver._sha256_json(without_sha),
    }


def _write_bundle(path: Path, source_git_revision: str) -> None:
    selected_cases = {
        case_id: {
            "order_ladder": [
                _reference(case_id),
                _reference(case_id),
            ],
            "reference": _reference(case_id),
            "pass": True,
        }
        for case_id in driver.SELECTED_CASES
    }
    boundary_without_sha = {
        "schema": aggregation_error_tiny_oracle.SCHEMA,
        "case_id": aggregation_error_tiny_oracle.BOUNDARY_CASE_ID,
        "pass": True,
    }
    boundary = {
        **boundary_without_sha,
        "sha256": driver._sha256_json(boundary_without_sha),
    }
    without_sha = {
        "schema": driver.ORACLE_BUNDLE_SCHEMA,
        "source_git_revision": source_git_revision,
        "tiny_root_definitions_sha256": (aggregation_error_tiny_oracle.definitions_sha256()),
        "selected_cases": selected_cases,
        "boundary_independent_certificate": boundary,
        "pass": True,
    }
    payload = {
        **without_sha,
        "sha256": driver._sha256_json(without_sha),
    }
    report_bytes = json.dumps(payload).encode("ascii")
    path.write_bytes(report_bytes)
    completion = {
        "schema": driver.ORACLE_BUNDLE_SCHEMA,
        "source_git_revision": source_git_revision,
        "report_path": str(path),
        "oracle_bundle_payload_sha256": payload["sha256"],
        "oracle_bundle_file_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "completion_marker_published_last": True,
    }
    (path.parent / "COMPLETE.json").write_text(
        json.dumps(completion),
        encoding="ascii",
    )


def _passing_oracle_bundle(source_git_revision: str) -> dict[str, object]:
    without_sha: dict[str, object] = {
        "schema": oracle_driver.SCHEMA,
        "source_git_revision": source_git_revision,
        "tiny_root_definitions_sha256": (aggregation_error_tiny_oracle.definitions_sha256()),
        "selected_cases": {},
        "boundary_independent_certificate": {},
        "checks": {},
        "pass": True,
        "runtime_seconds": 1.25,
    }
    return {
        **without_sha,
        "sha256": oracle_driver._sha256_json(without_sha),
    }


def test_private_pcg64_seed_material_is_replayable_and_role_separated() -> None:
    for case_id in driver.domains.CASE_IDS:
        simulator_seeds = {
            driver.domains.domain_stream_seed(
                731,
                case_id=case_id,
                domain=domain,
                stream_name=stream_name,
            )
            for domain in driver.domains.PUBLIC_DOMAINS
            for stream_name in driver.domains.SIMULATOR_STREAMS
        }
        for config_id in driver.CONFIG_IDS:
            stage_count = len(driver._stage_plan(config_id, 40))
            for init_index in range(4):
                initialization, optimizers = driver._private_stream_plan(
                    731,
                    case_id=case_id,
                    init_index=init_index,
                    stage_count=stage_count,
                )
                replay = driver._private_stream_plan(
                    731,
                    case_id=case_id,
                    init_index=init_index,
                    stage_count=stage_count,
                )
                assert (initialization, optimizers) == replay
                records = (initialization, *optimizers)
                source_seeds = {int(record["pcg64_source_seed"]) for record in records}
                jax_seeds = {int(record["derived_jax_seed"]) for record in records}
                assert len(source_seeds) == len(records)
                assert len(jax_seeds) == len(records)
                assert source_seeds.isdisjoint(simulator_seeds)
                assert all(0 <= seed < 2**32 for seed in jax_seeds)


def test_frozen_array_matrices_have_complete_unique_attempts() -> None:
    assert {name: len(matrix) for name, matrix in array_driver.MATRICES.items()} == {
        "compile_canary": 4,
        "overfit": 16,
        "standard_s4096": 36,
        "observation_canary": 8,
        "standard_s16384_nll": 12,
        "standard_s16384_partial": 12,
        "standard_s16384_pretrain": 12,
    }
    for matrix in array_driver.MATRICES.values():
        assert len(set(matrix)) == len(matrix)
        assert all(init_index in range(4) for *_, init_index in matrix)


def test_corrected_slurm_assets_use_shared_nodes_and_array_contract() -> None:
    assets = Path("docs/plans/rjmcmc_score_regularized_nle_corrected_assets")
    oracle_text = (assets / "run_corrected_oracle.sbatch").read_text(encoding="utf-8")
    array_text = (assets / "run_corrected_array.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --exclusive" not in oracle_text
    assert "#SBATCH --exclusive" not in array_text
    assert "--time=01:00:00" in oracle_text
    assert "--time=01:00:00" in array_text
    assert "--mem=8G" in oracle_text
    assert "--mem=8G" in array_text
    assert "SLURM_ARRAY_TASK_ID" in array_text
    assert "SLURM_ARRAY_JOB_ID" in array_text
    for text in (oracle_text, array_text):
        assert "#SBATCH --output=" not in text
        assert "#SBATCH --error=" not in text
        git_module = "module load git/2.45.1-pqk5"
        assert git_module in text
        assert text.index(git_module) < text.index('git -C "${source_root}"')
        assert 'exec "${python_bin}" \\' in text
    assert oracle_text.rstrip().endswith('--output-root "${run_root}"')
    assert array_text.rstrip().endswith('--patience "${NLE_PATIENCE}"')


def test_oracle_completion_binds_payload_and_exact_report_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_git_revision = "a" * 40
    bundle = _passing_oracle_bundle(source_git_revision)
    publication_order: list[Path] = []
    original_atomic_json = oracle_driver._atomic_json

    def record_atomic_json(path: Path, payload: object) -> str:
        digest = original_atomic_json(path, payload)
        publication_order.append(path)
        return digest

    monkeypatch.setattr(oracle_driver, "_atomic_json", record_atomic_json)
    report, completion_path = oracle_driver._publish_bundle(
        tmp_path,
        bundle,
        source_git_revision,
    )

    assert publication_order == [report, completion_path]
    expected_report_bytes = oracle_driver._pretty_json_bytes(bundle)
    assert report.read_bytes() == expected_report_bytes
    completion = json.loads(completion_path.read_text(encoding="ascii"))
    assert set(completion) == {
        "schema",
        "source_git_revision",
        "report_path",
        "oracle_bundle_payload_sha256",
        "oracle_bundle_file_sha256",
        "completion_marker_published_last",
    }
    assert completion == {
        "schema": oracle_driver.SCHEMA,
        "source_git_revision": source_git_revision,
        "report_path": str(report),
        "oracle_bundle_payload_sha256": bundle["sha256"],
        "oracle_bundle_file_sha256": hashlib.sha256(expected_report_bytes).hexdigest(),
        "completion_marker_published_last": True,
    }


def test_oracle_publication_is_create_only_and_preserves_first_evidence(
    tmp_path: Path,
) -> None:
    source_git_revision = "a" * 40
    bundle = _passing_oracle_bundle(source_git_revision)
    report, completion = oracle_driver._publish_bundle(
        tmp_path,
        bundle,
        source_git_revision,
    )
    original_report = report.read_bytes()
    original_completion = completion.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to replace"):
        oracle_driver._publish_bundle(
            tmp_path,
            bundle,
            source_git_revision,
        )

    assert report.read_bytes() == original_report
    assert completion.read_bytes() == original_completion


def test_merger_reports_missing_array_tasks_without_a_scientific_decision(
    tmp_path: Path,
) -> None:
    summary = merger.merge(
        "compile_canary",
        "missing-test",
        "a" * 40,
        tmp_path,
    )
    assert not summary["complete"]
    assert summary["complete_attempt_count"] == 0
    assert len(summary["missing_attempts"]) == 4
    assert summary["approximate_evidence_used_as_structural_weight"] is False
    assert summary["scientific_decision"].startswith("none:")


def test_one_small_overfit_attempt_publishes_completion_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "GRID_COUNTS", (16, 32, 64))
    source_git_revision = "a" * 40
    bundle = tmp_path / "oracle_bundle.json"
    _write_bundle(bundle, source_git_revision)
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    arguments = argparse.Namespace(
        mode="overfit",
        case_id="near_gaussian__two_cell__root",
        sample_count=256,
        config_id="nll_only",
        init_index=0,
        attempt_tag="test",
        base_seed=731,
        source_git_revision=source_git_revision,
        oracle_bundle=bundle,
        output_root=tmp_path,
        learning_rate=5.0e-4,
        batch_size=256,
        microbatch_size=64,
        max_epochs=1,
        patience=0,
    )
    report = driver.run_attempt(arguments, attempt_root)
    assert report["interpretation"]["status"] == ("exploratory_result_not_promotion")
    assert report["interpretation"]["overfit_validation_role"] == ("same catalogue optimizer diagnostic")
    assert all(report["streams"]["separation"]["checks"].values())
    assert report["initialization_loss_diagnostics"]["measured_before_loss_weights_applied"]
    assert report["artifact"]["byte_replay_pass"]
    assert len(report["scientific_evaluation"]["grid"]["ladder"]) == 3
    assert report["scientific_evaluation"]["vectorized_public_likelihood_parity"]["pass"]
    assert (attempt_root / "artifact.bin").is_file()
    assert (attempt_root / "report.json").is_file()
    report_bytes = (attempt_root / "report.json").read_bytes()
    completion = json.loads((attempt_root / "COMPLETE.json").read_text(encoding="ascii"))
    assert completion["completion_marker_published_last"]
    assert completion["report_payload_sha256"] == report["sha256"]
    assert completion["report_file_sha256"] == hashlib.sha256(report_bytes).hexdigest()
    assert (
        completion["serialized_artifact_file_sha256"]
        == hashlib.sha256((attempt_root / "artifact.bin").read_bytes()).hexdigest()
    )
