"""Tests for the BP1 direct-NLE confirmation certifier."""

from __future__ import annotations

import json
from pathlib import Path

from examples.rjmcmc import conditional_residual_image_flow_certify as development
from examples.rjmcmc import (
    conditional_residual_image_flow_confirmation_certify as confirmation,
)
from examples.rjmcmc import conditional_residual_image_flow_tiny_screen as screen


def _lock(path: Path) -> Path:
    """Write one structurally valid authenticated G1 lock."""
    source = {
        "git_revision": "0" * 40,
        "driver_sha256": "1" * 64,
    }
    selected = {
        f"{regime}__{family}__root": {
            "artifact_sha256": "3" * 64,
            "report_sha256": "4" * 64,
            "log_evidence": -1.0,
        }
        for regime, family, tiling in screen.DEVELOPMENT_MATRIX
        if tiling == "root"
    }
    payload = {
        "schema": development.LOCK_SCHEMA,
        "source": source,
        "protocol_sha256": "2" * 64,
        "locked_sample_count": 65_536,
        "selection_seed": screen.DEVELOPMENT_SELECTION_SEED,
        "confirmation_seeds": list(screen.CONFIRMATION_SEEDS),
        "selected_artifacts": selected,
        "certificate_sha256": "5" * 64,
    }
    envelope = {
        "payload": payload,
        "sha256": development._sha256_json(payload),
    }
    path.write_text(
        json.dumps(envelope, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_confirmation_requires_complete_authenticated_matrix(
    tmp_path: Path,
) -> None:
    """An empty G2 directory must stop without holdout eligibility."""
    lock_path = _lock(tmp_path / "common-lock.json")
    input_directory = tmp_path / "confirmation"
    input_directory.mkdir()
    certificate, eligible = confirmation.certify_confirmation(
        input_directory,
        lock_path=lock_path,
        expected_source_revision="0" * 40,
        expected_driver_sha256="1" * 64,
        expected_protocol_sha256="2" * 64,
    )

    assert eligible is None
    assert not certificate["complete_matrix"]
    assert not certificate["holdout_eligible"]
    assert certificate["authenticated_task_count"] == 0
    assert len(certificate["missing_files"]) == 54
    assert certificate["terminal_reason"] == (
        "complete authenticated 18-task confirmation matrix is absent"
    )


def test_confirmation_lock_source_is_authenticated(tmp_path: Path) -> None:
    """G2 must not accept a lock from another source revision."""
    lock_path = _lock(tmp_path / "common-lock.json")
    input_directory = tmp_path / "confirmation"
    input_directory.mkdir()

    try:
        confirmation.certify_confirmation(
            input_directory,
            lock_path=lock_path,
            expected_source_revision="f" * 40,
            expected_driver_sha256="1" * 64,
            expected_protocol_sha256="2" * 64,
        )
    except ValueError as error:
        assert "source or protocol" in str(error)
    else:  # pragma: no cover
        raise AssertionError("mismatched G1 lock source was accepted")
