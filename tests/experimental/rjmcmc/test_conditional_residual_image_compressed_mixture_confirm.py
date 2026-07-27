"""Tests for exact-mixture independent-scramble confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from examples.rjmcmc import (
    conditional_residual_image_compressed_mixture_certify as development_certify,
)
from examples.rjmcmc import (
    conditional_residual_image_compressed_mixture_confirm as confirmation,
)
from examples.rjmcmc import (
    conditional_residual_image_compressed_mixture_confirm_certify as confirm_certify,
)
from examples.rjmcmc import (
    conditional_residual_image_compressed_mixture_tiny_screen as development,
)


def _write_canonical(path: Path, payload: object) -> str:
    """Write canonical JSON and return its raw SHA-256."""
    raw = (development._canonical_json(payload) + "\n").encode("ascii")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _source_decision(revision: str) -> dict[str, Any]:
    """Build one minimal all-six eligible development source lock."""
    case_ids = development._expected_development_case_ids()
    payload: dict[str, Any] = {
        "schema": development.SOURCE_LOCK_SCHEMA,
        "protocol_sha256": development._sha256_json(development._protocol_payload()),
        "source_git_revision": revision,
        "source_driver_sha256": development._driver_sha256(),
        "a1_definitions_sha256": development.c1.A1_DEFINITIONS_SHA256,
        "development_seed": development.DEVELOPMENT_SEED,
        "minimum_passing_suffix_length": (development.DEVELOPMENT_MINIMUM_SOURCE_SUFFIX),
        "source_sample_counts": list(development.DEVELOPMENT_SOURCE_SAMPLE_COUNTS),
        "matrix_case_ids": list(case_ids),
        "locked_sample_count": 65_536,
        "eligible": True,
        "structural_inference_licensed": False,
        "case_certificates": {
            case_id: {
                "case_id": case_id,
                "sample_count": 65_536,
                "scientific_pass": True,
            }
            for case_id in case_ids
        },
    }
    payload["source_lock_sha256"] = development._source_lock_sha256(payload)
    return payload


def _compression_decision(
    revision: str,
    source_lock: dict[str, Any],
) -> dict[str, Any]:
    """Build one minimal eligible development compression decision."""
    return {
        "schema": development_certify.COMPRESSION_DECISION_SCHEMA,
        "protocol": development.PROTOCOL,
        "protocol_sha256": development._sha256_json(development._protocol_payload()),
        "source_git_revision": revision,
        "source_driver_sha256": development._driver_sha256(),
        "source_lock_sha256": source_lock["source_lock_sha256"],
        "locked_source_sample_count": 65_536,
        "locked_component_count": 256,
        "component_counts": list(development.DEVELOPMENT_COMPONENT_COUNTS),
        "eligible": True,
        "confirmation_status": "deferred_to_later_protocol_stage",
        "protected_catalogue_accessed": False,
        "production_output_written": False,
        "structural_inference_licensed": False,
    }


def _patch_development_identities(
    monkeypatch: pytest.MonkeyPatch,
    *,
    revision: str,
    source_raw_sha256: str,
    source_lock_sha256: str,
    compression_raw_sha256: str,
) -> None:
    """Point confirmation validation at synthetic immutable decisions."""
    monkeypatch.setattr(confirmation, "DEVELOPMENT_REVISION", revision)
    monkeypatch.setattr(
        confirmation,
        "DEVELOPMENT_SOURCE_DECISION_RAW_SHA256",
        source_raw_sha256,
    )
    monkeypatch.setattr(
        confirmation,
        "DEVELOPMENT_SOURCE_LOCK_SHA256",
        source_lock_sha256,
    )
    monkeypatch.setattr(
        confirmation,
        "DEVELOPMENT_COMPRESSION_DECISION_RAW_SHA256",
        compression_raw_sha256,
    )


def test_development_decisions_are_bound_by_raw_and_internal_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirmation must authenticate both frozen development decisions."""
    revision = "e" * 40
    source_lock = _source_decision(revision)
    source_path = tmp_path / "source.json"
    source_raw_sha256 = _write_canonical(source_path, source_lock)
    compression = _compression_decision(revision, source_lock)
    compression_path = tmp_path / "compression.json"
    compression_raw_sha256 = _write_canonical(
        compression_path,
        compression,
    )
    _patch_development_identities(
        monkeypatch,
        revision=revision,
        source_raw_sha256=source_raw_sha256,
        source_lock_sha256=source_lock["source_lock_sha256"],
        compression_raw_sha256=compression_raw_sha256,
    )

    certificate = confirmation.validate_development_inputs(
        source_decision_path=source_path,
        compression_decision_path=compression_path,
    )

    assert certificate["eligible"] is True
    assert certificate["development_locked_source_sample_count"] == 65_536
    assert certificate["development_locked_component_count"] == 256
    compression["locked_component_count"] = 512
    _write_canonical(compression_path, compression)
    with pytest.raises(ValueError, match="raw SHA-256"):
        confirmation.validate_development_inputs(
            source_decision_path=source_path,
            compression_decision_path=compression_path,
        )


def _confirmation_report(
    *,
    case_id: str,
    source_seed: int,
    revision: str,
    scientific_pass: bool = True,
    log_evidence: float = 0.0,
) -> dict[str, Any]:
    """Build one minimal confirmation artifact for certifier tests."""
    metrics = {
        name: 0.0 for name in development.c1.THRESHOLDS if name != "between_bank_log_evidence_range_nat"
    }
    checks = {
        "source_scientific_pass": scientific_pass,
        "compressed_scientific_pass": scientific_pass,
        "compression_mean_closure": True,
        "compression_covariance_closure": True,
        "compression_kl_bound_finite": True,
    }
    return {
        "schema": confirmation.SCHEMA,
        "protocol": confirmation.PROTOCOL,
        "protocol_sha256": development._sha256_json(confirmation._protocol_payload()),
        "source_git_revision": revision,
        "driver_sha256": confirm_certify._driver_sha256(),
        "development": {
            "revision": confirmation.DEVELOPMENT_REVISION,
            "source_decision_raw_sha256": (confirmation.DEVELOPMENT_SOURCE_DECISION_RAW_SHA256),
            "source_lock_sha256": (confirmation.DEVELOPMENT_SOURCE_LOCK_SHA256),
            "compression_decision_raw_sha256": (confirmation.DEVELOPMENT_COMPRESSION_DECISION_RAW_SHA256),
            "locked_source_sample_count": (confirmation.LOCKED_SOURCE_SAMPLE_COUNT),
            "locked_component_count": confirmation.LOCKED_COMPONENT_COUNT,
        },
        "case_id": case_id,
        "source_seed": source_seed,
        "cluster_seed": confirmation.CLUSTER_SEED,
        "source": {
            "exact_evaluation": {
                "metrics": metrics,
                "posterior_summary": {"log_evidence": log_evidence},
            }
        },
        "compression": {
            "exact_evaluation": {
                "metrics": metrics,
                "posterior_summary": {"log_evidence": log_evidence},
            },
            "moment_diagnostics": {
                "mean_maximum_absolute_difference_from_source": 0.0,
                "covariance_maximum_absolute_difference_from_source": 0.0,
            },
        },
        "confirmation_checks": checks,
        "observed_residual_used_for_basis_selection": False,
        "retuning_performed": False,
        "protected_catalogue_accessed": False,
        "production_output_written": False,
        "structural_inference_licensed": False,
        "scientific_pass": scientific_pass,
    }


def _write_confirmation_matrix(
    directory: Path,
    *,
    revision: str,
    failing_pair: tuple[str, int] | None = None,
    evidence_override: dict[tuple[str, int], float] | None = None,
) -> None:
    """Write the exact 18-artifact confirmation matrix."""
    directory.mkdir()
    for case_id in development._expected_development_case_ids():
        for source_seed in confirmation.SOURCE_SEEDS:
            pair = (case_id, source_seed)
            report = _confirmation_report(
                case_id=case_id,
                source_seed=source_seed,
                revision=revision,
                scientific_pass=pair != failing_pair,
                log_evidence=(0.0 if evidence_override is None else evidence_override.get(pair, 0.0)),
            )
            _write_canonical(
                directory / f"{case_id}__seed{source_seed}.json",
                report,
            )


def test_confirmation_certifier_requires_all_eighteen_passes(
    tmp_path: Path,
) -> None:
    """The common decision must fail if even one frozen pair fails."""
    revision = "f" * 40
    case_id = development._expected_development_case_ids()[-1]
    failing_pair = (case_id, confirmation.SOURCE_SEEDS[-1])
    report_directory = tmp_path / "confirmation"
    _write_confirmation_matrix(
        report_directory,
        revision=revision,
        failing_pair=failing_pair,
    )

    decision = confirm_certify.certify_confirmation(
        report_directory=report_directory,
        expected_revision=revision,
    )

    assert decision["artifact_count"] == 18
    assert decision["eligible"] is False
    assert decision["failures"] == [
        {
            "case_id": case_id,
            "source_seed": confirmation.SOURCE_SEEDS[-1],
            "failed_checks": [
                "compressed_scientific_pass",
                "source_scientific_pass",
            ],
        }
    ]


def test_confirmation_certifier_passes_complete_matrix(
    tmp_path: Path,
) -> None:
    """All 18 passing artifacts should produce one eligible decision."""
    revision = "1" * 40
    report_directory = tmp_path / "confirmation"
    _write_confirmation_matrix(report_directory, revision=revision)

    decision = confirm_certify.certify_confirmation(
        report_directory=report_directory,
        expected_revision=revision,
    )

    assert decision["eligible"] is True
    assert decision["failures"] == []
    assert decision["between_seed_failures"] == []
    assert set(decision["artifact_raw_sha256"]) == {
        f"{case_id}__seed{source_seed}"
        for case_id in development._expected_development_case_ids()
        for source_seed in confirmation.SOURCE_SEEDS
    }


def test_confirmation_certifier_enforces_between_seed_evidence_range(
    tmp_path: Path,
) -> None:
    """Independent-scramble evidence spread is a separate common gate."""
    revision = "2" * 40
    case_id = development._expected_development_case_ids()[0]
    pair = (case_id, confirmation.SOURCE_SEEDS[-1])
    report_directory = tmp_path / "confirmation"
    _write_confirmation_matrix(
        report_directory,
        revision=revision,
        evidence_override={pair: 1.0},
    )

    decision = confirm_certify.certify_confirmation(
        report_directory=report_directory,
        expected_revision=revision,
    )

    assert decision["failures"] == []
    assert decision["eligible"] is False
    assert decision["between_seed_failures"] == [
        {
            "case_id": case_id,
            "source_evidence_range_nat": 1.0,
            "compression_evidence_range_nat": 1.0,
            "threshold_nat": development.c1.THRESHOLDS["between_bank_log_evidence_range_nat"],
        }
    ]


def test_confirmation_certifier_requires_every_per_shard_metric(
    tmp_path: Path,
) -> None:
    """Missing per-shard science is rejected instead of silently skipped."""
    revision = "3" * 40
    report_directory = tmp_path / "confirmation"
    _write_confirmation_matrix(report_directory, revision=revision)
    path = sorted(report_directory.glob("*.json"))[0]
    report = json.loads(path.read_text(encoding="ascii"))
    report["source"]["exact_evaluation"]["metrics"].pop("absolute_log_evidence_error_nat")
    _write_canonical(path, report)

    with pytest.raises(ValueError, match="missing required source metric"):
        confirm_certify.certify_confirmation(
            report_directory=report_directory,
            expected_revision=revision,
        )


def test_confirmation_certifier_requires_finite_log_evidence(
    tmp_path: Path,
) -> None:
    """Cross-seed evidence certification fails closed on non-numeric input."""
    revision = "4" * 40
    report_directory = tmp_path / "confirmation"
    _write_confirmation_matrix(report_directory, revision=revision)
    path = sorted(report_directory.glob("*.json"))[0]
    report = json.loads(path.read_text(encoding="ascii"))
    report["source"]["exact_evaluation"]["posterior_summary"]["log_evidence"] = "not-a-number"
    _write_canonical(path, report)

    with pytest.raises(ValueError, match="source log evidence must be numerical"):
        confirm_certify.certify_confirmation(
            report_directory=report_directory,
            expected_revision=revision,
        )


def test_confirmation_rejects_unfrozen_seed_before_reading_inputs() -> None:
    """An unplanned source scramble must fail before external input access."""
    with pytest.raises(ValueError, match="source_seed"):
        confirmation.run_confirmation(
            case_id="near_gaussian__two_cell__root",
            source_seed=123,
            source_decision_path=Path("absent-source.json"),
            compression_decision_path=Path("absent-compression.json"),
            include_timings=False,
        )
