"""Authentication and lock tests for the sbi-NSF G1 merger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from examples.rjmcmc import conditional_residual_image_flow_certify as common
from examples.rjmcmc import conditional_residual_image_sbi_nsf_certify as certify
from examples.rjmcmc import (
    conditional_residual_image_sbi_nsf_confirmation_certify as confirmation,
)
from examples.rjmcmc import conditional_residual_image_sbi_nsf_tiny_screen as screen
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mdn import (
    ResidualImageContext,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_sbi_nsf import (
    ConditionalResidualImageSbiNsf,
    conditional_residual_unit_covariances,
    make_conditional_residual_nsf,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)


def _artifact() -> ConditionalResidualImageSbiNsf:
    """Return one initialized authenticated NSF."""
    aggregation = AdditiveDirichletAggregation(
        np.asarray([1.2, 2.1], dtype=np.float64),
        np.asarray([[1.0, 0.2], [0.4, 1.3]], dtype=np.float64),
        np.asarray([0.3, 0.5], dtype=np.float64),
        np.eye(2, dtype=np.float64),
    )
    context = ResidualImageContext.from_aggregation(
        aggregation,
        np.zeros(2, dtype=np.int64),
        np.arange(2, dtype=np.int64),
        source_provenance="sbi-NSF certifier fixture",
    )
    return ConditionalResidualImageSbiNsf(
        context,
        conditional_residual_unit_covariances(aggregation, context),
        np.zeros(1),
        np.ones(1),
        make_conditional_residual_nsf(1, 1, source_seed=17),
        initialization_seed=17,
        source_provenance="sbi-NSF certifier artifact",
    )


def _write_task(directory: Path) -> tuple[str, str, str, str]:
    """Write one syntactically valid development task."""
    artifact = _artifact()
    source_revision = "a" * 40
    driver_sha256 = "b" * 64
    protocol_sha256 = "c" * 64
    result = {
        "schema": screen.SCHEMA,
        "profile": "development",
        "source": {
            "git_revision": source_revision,
            "driver_sha256": driver_sha256,
        },
        "protocol": {
            "name": screen.PROTOCOL,
            "sha256": protocol_sha256,
        },
        "case_id": "near_gaussian__two_cell__root",
        "training_sample_count": 4_096,
        "base_seed": screen.DEVELOPMENT_SELECTION_SEED,
        "selected_artifact_sha256": artifact.artifact_sha256,
        "task_pass": True,
        "fit_development_pass": True,
        "selected_generalization_pass": True,
        "artifact_replay_pass": True,
        "evaluation": {
            "scientific_pass": True,
            "metrics": {"scaled_coordinate_gradient_error": 0.01},
            "checks": {"scaled_coordinate_gradient_error": True},
            "posterior_summary": {"log_evidence": -1.5},
        },
    }
    screen._write_result(directory, result, artifact.to_bytes())
    stem = "near_gaussian__two_cell__root__S4096__base731"
    return stem, source_revision, driver_sha256, protocol_sha256


def test_expected_matrix_and_common_suffix_rule() -> None:
    """The merger must expect exactly 24 tasks and a two-size suffix."""
    assert len(certify._expected_stems()) == 24
    cases = {
        f"{regime}__{family}__root": {
            4_096: False,
            16_384: True,
            65_536: True,
            262_144: True,
        }
        for regime, family, _tiling in screen.DEVELOPMENT_MATRIX
    }
    assert common._common_lock(
        cases,
        sample_counts=screen.DEVELOPMENT_SAMPLE_COUNTS,
    ) == 16_384
    cases["boundary_heavy__four_cell__root"][65_536] = False
    assert (
        common._common_lock(
            cases,
            sample_counts=screen.DEVELOPMENT_SAMPLE_COUNTS,
        )
        is None
    )


def test_task_authentication_replays_non_pickle_artifact(tmp_path: Path) -> None:
    """A valid report, marker, and raw-tensor artifact must authenticate."""
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


def test_marker_tampering_fails_closed(tmp_path: Path) -> None:
    """Completion-marker mutation must invalidate the task."""
    stem, revision, driver, protocol = _write_task(tmp_path)
    marker_path = tmp_path / f"{stem}.complete.json"
    marker = json.loads(marker_path.read_text())
    marker["task_pass"] = False
    marker_path.write_text(
        json.dumps(marker, separators=(",", ":"), sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="completion marker"):
        certify._validate_task(
            tmp_path,
            stem,
            expected_source_revision=revision,
            expected_driver_sha256=driver,
            expected_protocol_sha256=protocol,
            expected_base_seed=screen.DEVELOPMENT_SELECTION_SEED,
        )


def test_confirmation_requires_authenticated_complete_g1_lock(
    tmp_path: Path,
) -> None:
    """G2 cannot begin without a source-matched six-case lock."""
    revision = "a" * 40
    driver = "b" * 64
    protocol = "c" * 64
    selected = {
        f"{regime}__{family}__root": {
            "artifact_sha256": "d" * 64,
            "report_sha256": "e" * 64,
            "log_evidence": -1.0,
        }
        for regime, family, _tiling in screen.DEVELOPMENT_MATRIX
    }
    payload = {
        "schema": certify.LOCK_SCHEMA,
        "source": {
            "git_revision": revision,
            "driver_sha256": driver,
        },
        "protocol_sha256": protocol,
        "locked_sample_count": 65_536,
        "selection_seed": screen.DEVELOPMENT_SELECTION_SEED,
        "confirmation_seeds": list(screen.CONFIRMATION_SEEDS),
        "selected_artifacts": selected,
        "certificate_sha256": "f" * 64,
    }
    lock = {
        "payload": payload,
        "sha256": common._sha256_json(payload),
    }
    lock_path = tmp_path / "common-lock.json"
    lock_path.write_text(
        json.dumps(lock, separators=(",", ":"), sort_keys=True) + "\n"
    )
    _lock, sample_count = confirmation._authenticate_lock(
        lock_path,
        expected_source_revision=revision,
        expected_driver_sha256=driver,
        expected_protocol_sha256=protocol,
    )
    assert sample_count == 65_536
    payload["selected_artifacts"].pop("boundary_heavy__four_cell__root")
    lock["sha256"] = common._sha256_json(payload)
    lock_path.write_text(
        json.dumps(lock, separators=(",", ":"), sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="every case"):
        confirmation._authenticate_lock(
            lock_path,
            expected_source_revision=revision,
            expected_driver_sha256=driver,
            expected_protocol_sha256=protocol,
        )
