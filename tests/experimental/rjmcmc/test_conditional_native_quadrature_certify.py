"""Authentication and lock tests for the native-quadrature G1 merger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from examples.rjmcmc import conditional_native_quadrature_certify as certify
from examples.rjmcmc import conditional_native_quadrature_confirmation as confirmation
from examples.rjmcmc import (
    conditional_native_quadrature_confirmation_certify as confirmation_certify,
)
from examples.rjmcmc import conditional_native_quadrature_tiny_screen as screen
from examples.rjmcmc import conditional_residual_image_flow_certify as common
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_native_quadrature import (
    ConditionalNativeQuadrature,
)

_SOURCE_REVISION = "a" * 40
_DRIVER_SHA256 = "b" * 64


def _artifact() -> ConditionalNativeQuadrature:
    """Return one small authenticated quadrature artifact."""
    aggregation = AdditiveDirichletAggregation(
        np.asarray([1.2, 2.1], dtype=np.float64),
        np.asarray([[1.0, 0.2], [0.4, 1.3]], dtype=np.float64),
        np.asarray([0.3, 0.5], dtype=np.float64),
        np.eye(2, dtype=np.float64),
    )
    return ConditionalNativeQuadrature.from_aggregation(
        aggregation,
        np.zeros(2, dtype=np.int64),
        np.arange(2, dtype=np.int64),
        quadrature_order=24,
        chart="single",
        source_git_revision=_SOURCE_REVISION,
        driver_sha256=_DRIVER_SHA256,
        protocol_sha256=screen.DEVELOPMENT_PROTOCOL_SHA256,
        source_provenance="native-quadrature certifier fixture",
    )


def _write_task(directory: Path) -> tuple[str, str, str, str]:
    """Write one syntactically valid development task."""
    artifact = _artifact()
    result = {
        "schema": screen.SCHEMA,
        "profile": "development",
        "source": {
            "git_revision": _SOURCE_REVISION,
            "driver_sha256": _DRIVER_SHA256,
        },
        "protocol": {
            "name": screen.PROTOCOL,
            "sha256": screen.DEVELOPMENT_PROTOCOL_SHA256,
            "payload": screen._protocol_payload(),
        },
        "case_id": "near_gaussian__two_cell__root",
        "quadrature_order": 24,
        "component_count": artifact.component_count,
        "base_seed": screen.DEVELOPMENT_SELECTION_SEED,
        "chart": artifact.chart,
        "selected_artifact_sha256": artifact.artifact_sha256,
        "task_pass": True,
        "artifact_replay_pass": True,
        "moment_audit": {"pass": True},
        "chart_audit": {"finite": True},
        "evaluation": {
            "scientific_pass": True,
            "metrics": {"scaled_coordinate_gradient_error": 0.01},
            "checks": {"scaled_coordinate_gradient_error": True},
            "posterior_summary": {"log_evidence": -1.5},
        },
    }
    screen._write_result(directory, result, artifact.to_bytes())
    stem = "near_gaussian__two_cell__root__O24__base731"
    return (
        stem,
        _SOURCE_REVISION,
        _DRIVER_SHA256,
        screen.DEVELOPMENT_PROTOCOL_SHA256,
    )


def test_expected_matrix_and_common_suffix_rule() -> None:
    """The merger expects 24 tasks and a two-order common suffix."""
    assert len(certify._expected_stems()) == 24
    cases = {
        f"{regime}__{family}__root": {
            24: False,
            32: True,
            40: True,
            48: True,
        }
        for regime, family, _tiling in screen.DEVELOPMENT_MATRIX
    }
    assert (
        common._common_lock(
            cases,
            sample_counts=screen.DEVELOPMENT_QUADRATURE_ORDERS,
        )
        == 32
    )
    cases["boundary_heavy__four_cell__root"][40] = False
    assert (
        common._common_lock(
            cases,
            sample_counts=screen.DEVELOPMENT_QUADRATURE_ORDERS,
        )
        is None
    )


def test_task_authentication_replays_canonical_artifact(
    tmp_path: Path,
) -> None:
    """A valid report, marker, and binary artifact authenticate."""
    stem, revision, driver, protocol = _write_task(tmp_path)
    record = certify._validate_task(
        tmp_path,
        stem,
        expected_source_revision=revision,
        expected_driver_sha256=driver,
        expected_protocol_sha256=protocol,
        expected_base_seed=screen.DEVELOPMENT_SELECTION_SEED,
    )

    assert record["task_pass"] is True
    assert record["artifact_replay_pass"] is True
    assert record["quadrature_order"] == 24


def test_marker_tampering_fails_closed(tmp_path: Path) -> None:
    """Completion-marker mutation invalidates the task."""
    stem, revision, driver, protocol = _write_task(tmp_path)
    marker_path = tmp_path / f"{stem}.complete.json"
    marker = json.loads(marker_path.read_text())
    marker["task_pass"] = False
    marker_path.write_text(json.dumps(marker, separators=(",", ":"), sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="completion marker"):
        certify._validate_task(
            tmp_path,
            stem,
            expected_source_revision=revision,
            expected_driver_sha256=driver,
            expected_protocol_sha256=protocol,
            expected_base_seed=screen.DEVELOPMENT_SELECTION_SEED,
        )


def test_incomplete_matrix_cannot_publish_lock(tmp_path: Path) -> None:
    """One valid task cannot be mistaken for the complete matrix."""
    _write_task(tmp_path)
    certificate, lock = certify.merge_development(
        tmp_path,
        expected_source_revision=_SOURCE_REVISION,
        expected_driver_sha256=_DRIVER_SHA256,
        expected_protocol_sha256=screen.DEVELOPMENT_PROTOCOL_SHA256,
    )

    assert certificate["complete_matrix"] is False
    assert certificate["lock_published"] is False
    assert certificate["terminal_reason"] == ("complete authenticated 24-task matrix is absent")
    assert lock is None


def _write_lock(path: Path) -> None:
    """Write one complete synthetic G1 lock."""
    selected = {
        f"{regime}__{family}__root": {
            "artifact_sha256": "d" * 64,
            "report_sha256": "e" * 64,
            "log_evidence": -1.0,
            "component_count": 24 if family == "two_cell" else 24**3,
            "chart": "single" if family == "two_cell" else "column-first",
        }
        for regime, family, _tiling in screen.DEVELOPMENT_MATRIX
    }
    payload = {
        "schema": certify.LOCK_SCHEMA,
        "source": {
            "git_revision": _SOURCE_REVISION,
            "driver_sha256": _DRIVER_SHA256,
        },
        "protocol_sha256": screen.DEVELOPMENT_PROTOCOL_SHA256,
        "locked_order": 24,
        "selection_seed": screen.DEVELOPMENT_SELECTION_SEED,
        "confirmation_seeds": list(screen.CONFIRMATION_SEEDS),
        "confirmation_sample_count": screen.CONFIRMATION_SAMPLE_COUNT,
        "selected_artifacts": selected,
        "certificate_sha256": "f" * 64,
    }
    envelope = {
        "payload": payload,
        "sha256": common._sha256_json(payload),
    }
    path.write_text(json.dumps(envelope, separators=(",", ":"), sort_keys=True) + "\n")


def test_confirmation_requires_complete_authenticated_lock(
    tmp_path: Path,
) -> None:
    """G2 cannot begin without a complete source-matched G1 lock."""
    lock_path = tmp_path / "common-lock.json"
    _write_lock(lock_path)
    lock, order = confirmation._authenticate_lock(
        lock_path,
        expected_source_revision=_SOURCE_REVISION,
        expected_driver_sha256=_DRIVER_SHA256,
        expected_protocol_sha256=screen.DEVELOPMENT_PROTOCOL_SHA256,
    )

    assert order == 24
    assert len(lock["selected_artifacts"]) == 6
    assert len(confirmation_certify._expected_stems(order)) == 18

    envelope = json.loads(lock_path.read_text())
    envelope["payload"]["selected_artifacts"].pop("boundary_heavy__four_cell__root")
    envelope["sha256"] = common._sha256_json(envelope["payload"])
    lock_path.write_text(json.dumps(envelope, separators=(",", ":"), sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="every case"):
        confirmation._authenticate_lock(
            lock_path,
            expected_source_revision=_SOURCE_REVISION,
            expected_driver_sha256=_DRIVER_SHA256,
            expected_protocol_sha256=screen.DEVELOPMENT_PROTOCOL_SHA256,
        )


def test_component_frequency_groups_have_nonsparse_expected_counts() -> None:
    """Canonical frequency bins preserve mass and meet the expected-count rule."""
    probabilities = np.full(48**3, 1.0 / 48**3, dtype=np.float64)
    group_ids, grouped = confirmation._frequency_groups(
        probabilities,
        sample_count=screen.CONFIRMATION_SAMPLE_COUNT,
    )

    assert group_ids.shape == probabilities.shape
    assert np.sum(grouped) == pytest.approx(1.0, abs=1e-15)
    assert np.min(grouped * screen.CONFIRMATION_SAMPLE_COUNT) >= 20.0
    assert np.array_equal(np.unique(group_ids), np.arange(grouped.size))
