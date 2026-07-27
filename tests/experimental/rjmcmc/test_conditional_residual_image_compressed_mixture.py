"""Tests for the two-stage exact-mixture development screen and certifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from examples.rjmcmc import (
    conditional_residual_image_compressed_mixture_certify as certify,
)
from examples.rjmcmc import (
    conditional_residual_image_compressed_mixture_tiny_screen as screen,
)


def _write_canonical(path: Path, payload: object) -> None:
    """Write canonical JSON used as immutable test input."""
    path.write_text(certify._canonical_json(payload) + "\n", encoding="ascii")


def _source_report(
    *,
    case_id: str,
    revision: str,
    passes: tuple[bool, ...],
) -> dict[str, Any]:
    """Build one minimal authenticated source report for merger tests."""
    evaluations = []
    for sample_count, scientific_pass in zip(
        screen.DEVELOPMENT_SOURCE_SAMPLE_COUNTS,
        passes,
        strict=True,
    ):
        certificate = {
            "case_id": case_id,
            "sample_count": sample_count,
            "input_sha256": "1" * 64,
            "source_artifact_sha256": f"{sample_count:064x}",
            "scientific_pass": scientific_pass,
        }
        evaluations.append(
            {
                "sample_count": sample_count,
                "exact_vs_source": {"scientific_pass": scientific_pass},
                "merger_certificate": certificate,
            }
        )
    return {
        "schema": screen.SCHEMA,
        "protocol": screen.PROTOCOL,
        "profile": "development",
        "stage": "source",
        "source_git_revision": revision,
        "driver_sha256": screen._driver_sha256(),
        "protocol_sha256": screen._sha256_json(screen._protocol_payload()),
        "selected_case_id": case_id,
        "protected_catalogue_accessed": False,
        "production_output_written": False,
        "structural_inference_licensed": False,
        "case": {
            "case_id": case_id,
            "stage": "source",
            "source_bank": {
                "sample_counts": list(screen.DEVELOPMENT_SOURCE_SAMPLE_COUNTS),
                "minimum_common_passing_suffix_length": (screen.DEVELOPMENT_MINIMUM_SOURCE_SUFFIX),
                "evaluations": evaluations,
            },
        },
    }


def _compression_report(
    *,
    case_id: str,
    revision: str,
    source_lock: dict[str, Any],
    passes: tuple[bool, ...],
) -> dict[str, Any]:
    """Build one minimal authenticated compression report for merger tests."""
    evaluations = [
        {
            "component_count": component_count,
            "exact_vs_compressed": {"scientific_pass": scientific_pass},
        }
        for component_count, scientific_pass in zip(
            screen.DEVELOPMENT_COMPONENT_COUNTS,
            passes,
            strict=True,
        )
    ]
    return {
        "schema": screen.SCHEMA,
        "protocol": screen.PROTOCOL,
        "profile": "development",
        "stage": "compression",
        "source_git_revision": revision,
        "driver_sha256": screen._driver_sha256(),
        "protocol_sha256": screen._sha256_json(screen._protocol_payload()),
        "selected_case_id": case_id,
        "protected_catalogue_accessed": False,
        "production_output_written": False,
        "structural_inference_licensed": False,
        "case": {
            "case_id": case_id,
            "stage": "compression",
            "authenticated_common_source_lock": {
                "source_lock_sha256": source_lock["source_lock_sha256"],
                "rebuild_certificate_matched": True,
            },
            "compression": {
                "component_counts": list(screen.DEVELOPMENT_COMPONENT_COUNTS),
                "minimum_passing_suffix_length": (screen.DEVELOPMENT_MINIMUM_COMPRESSION_SUFFIX),
                "evaluations": evaluations,
            },
        },
    }


def _write_matrix(
    directory: Path,
    *,
    reports: dict[str, dict[str, Any]],
) -> None:
    """Write exactly one canonical report per frozen development case."""
    directory.mkdir()
    for case_id, report in reports.items():
        _write_canonical(directory / f"{case_id}.json", report)


def test_source_smoke_replays_without_timings() -> None:
    """The bounded source stage must replay exactly without timing fields."""
    kwargs = {
        "profile": "smoke",
        "stage": "source",
        "case_id": "near_gaussian__two_cell__root",
        "source_sample_counts": (1_024, 4_096),
        "include_timings": False,
    }
    first = screen.run_screen(**kwargs)  # type: ignore[arg-type]
    second = screen.run_screen(**kwargs)  # type: ignore[arg-type]

    assert first == second
    assert first["stage"] == "source"
    assert first["scientific_pass"] is True
    assert first["case"]["compression_evaluated"] is False
    assert first["case"]["source_bank"]["case_passing_suffix_start"] == 1_024
    assert all(
        evaluation["build_seconds"] is None and evaluation["exact_vs_source"]["evaluation_seconds"] is None
        for evaluation in first["case"]["source_bank"]["evaluations"]
    )


def test_stage_controls_fail_closed() -> None:
    """Compression must require a development lock and source must reject it."""
    with pytest.raises(ValueError, match="requires source_lock_path"):
        screen.run_screen(
            profile="development",
            stage="compression",
            case_id="near_gaussian__two_cell__root",
            include_timings=False,
        )
    with pytest.raises(ValueError, match="cannot consume source_lock_path"):
        screen.run_screen(
            profile="smoke",
            stage="source",
            case_id="near_gaussian__two_cell__root",
            source_lock_path=Path("unexpected.json"),
            include_timings=False,
        )


def test_source_merger_selects_one_common_suffix(
    tmp_path: Path,
) -> None:
    """The source merger must select the smallest all-case passing suffix."""
    revision = "a" * 40
    case_ids = certify._case_ids()
    reports = {
        case_id: _source_report(
            case_id=case_id,
            revision=revision,
            passes=(case_id != case_ids[-1], True, True),
        )
        for case_id in case_ids
    }
    report_directory = tmp_path / "source"
    _write_matrix(report_directory, reports=reports)

    decision = certify.merge_source(
        report_directory=report_directory,
        expected_revision=revision,
    )

    assert decision["schema"] == screen.SOURCE_LOCK_SCHEMA
    assert decision["eligible"] is True
    assert decision["locked_sample_count"] == 262_144
    assert set(decision["case_certificates"]) == set(case_ids)
    assert decision["source_lock_sha256"] == screen._source_lock_sha256(decision)


def test_source_merger_preserves_an_ineligible_decision(
    tmp_path: Path,
) -> None:
    """No common two-size suffix must yield a decision rather than a lock."""
    revision = "b" * 40
    reports = {
        case_id: _source_report(
            case_id=case_id,
            revision=revision,
            passes=(True, False, True),
        )
        for case_id in certify._case_ids()
    }
    report_directory = tmp_path / "source"
    _write_matrix(report_directory, reports=reports)

    decision = certify.merge_source(
        report_directory=report_directory,
        expected_revision=revision,
    )

    assert decision["schema"] == certify.SOURCE_DECISION_SCHEMA
    assert decision["eligible"] is False
    assert decision["locked_sample_count"] is None
    assert "source_lock_sha256" not in decision


def test_compression_merger_authenticates_lock_and_selects_suffix(
    tmp_path: Path,
) -> None:
    """Compression must use one source lock and one all-case component suffix."""
    revision = "c" * 40
    case_ids = certify._case_ids()
    source_reports = {
        case_id: _source_report(
            case_id=case_id,
            revision=revision,
            passes=(True, True, True),
        )
        for case_id in case_ids
    }
    source_directory = tmp_path / "source"
    _write_matrix(source_directory, reports=source_reports)
    source_lock = certify.merge_source(
        report_directory=source_directory,
        expected_revision=revision,
    )
    source_lock_path = tmp_path / "source-lock.json"
    _write_canonical(source_lock_path, source_lock)

    common_passes = (False, False, True, False, True, True, True)
    compression_reports = {
        case_id: _compression_report(
            case_id=case_id,
            revision=revision,
            source_lock=source_lock,
            passes=common_passes,
        )
        for case_id in case_ids
    }
    compression_directory = tmp_path / "compression"
    _write_matrix(compression_directory, reports=compression_reports)

    decision = certify.merge_compression(
        report_directory=compression_directory,
        source_lock_path=source_lock_path,
        expected_revision=revision,
    )

    assert decision["eligible"] is True
    assert decision["locked_source_sample_count"] == 65_536
    assert decision["locked_component_count"] == 256
    assert decision["confirmation_status"] == "deferred_to_later_protocol_stage"


def test_canonical_and_self_digest_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    """Noncanonical reports and modified source locks must be rejected."""
    revision = "d" * 40
    reports = {
        case_id: _source_report(
            case_id=case_id,
            revision=revision,
            passes=(True, True, True),
        )
        for case_id in certify._case_ids()
    }
    report_directory = tmp_path / "source"
    _write_matrix(report_directory, reports=reports)
    first_path = next(report_directory.glob("*.json"))
    payload = json.loads(first_path.read_text(encoding="ascii"))
    first_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="canonical JSON"):
        certify.merge_source(
            report_directory=report_directory,
            expected_revision=revision,
        )

    _write_canonical(first_path, payload)
    lock = certify.merge_source(
        report_directory=report_directory,
        expected_revision=revision,
    )
    lock["locked_sample_count"] = 262_144
    lock_path = tmp_path / "tampered-lock.json"
    _write_canonical(lock_path, lock)
    with pytest.raises(ValueError, match="self-digest"):
        screen._load_source_lock(
            lock_path,
            source_revision=revision,
            driver_sha256=screen._driver_sha256(),
        )
