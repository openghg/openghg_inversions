"""Focused tests for the native-quadrature BP1 driver."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from examples.rjmcmc import conditional_native_quadrature_tiny_screen as driver
from openghg_inversions.experimental.rjmcmc.aggregation_error_native_quadrature import (
    ConditionalNativeQuadrature,
)

_GIT_SHA = "0" * 40
_DRIVER_SHA = "1" * 64


def test_frozen_protocol_digest_and_matrix() -> None:
    assert driver._protocol_sha256() == driver.DEVELOPMENT_PROTOCOL_SHA256
    assert driver.DEVELOPMENT_QUADRATURE_ORDERS == (24, 32, 40, 48)
    assert len(driver.DEVELOPMENT_MATRIX) == 6
    assert all(case[2] == "root" for case in driver.DEVELOPMENT_MATRIX)
    assert driver._protocol_payload()["protected"] == {
        "catalogue_id": driver.PROTECTED_HOLDOUT_CATALOGUE_ID,
        "protected_action_authorized": False,
        "g3_requires_passing_g2_holdout_eligible_certificate": True,
    }


def test_smoke_builds_replayable_density_and_simulator() -> None:
    result, artifact_bytes = driver.run_case(
        regime_name="near_gaussian",
        family="two_cell",
        quadrature_order=8,
        base_seed=731,
        profile="smoke",
        source_git_revision=_GIT_SHA,
        driver_sha256=_DRIVER_SHA,
    )
    replayed = ConditionalNativeQuadrature.from_bytes(
        artifact_bytes,
        expected_sha256=result["selected_artifact_sha256"],
    )

    assert result["task_pass"] is True
    assert result["artifact_replay_pass"] is True
    assert result["moment_audit"]["pass"] is True
    assert result["sample_audit"]["pass"] is True
    assert result["protocol"]["sha256"] == driver.DEVELOPMENT_PROTOCOL_SHA256
    assert replayed.to_bytes() == artifact_bytes


def test_result_publication_writes_completion_marker_last(
    tmp_path: Path,
) -> None:
    result, artifact_bytes = driver.run_case(
        regime_name="near_gaussian",
        family="two_cell",
        quadrature_order=8,
        base_seed=731,
        profile="smoke",
        source_git_revision=_GIT_SHA,
        driver_sha256=_DRIVER_SHA,
    )
    paths = driver._write_result(tmp_path, result, artifact_bytes)
    artifact_path = Path(paths["artifact"])
    report_path = Path(paths["report"])
    marker_path = Path(paths["completion_marker"])
    report = json.loads(report_path.read_text())
    marker = json.loads(marker_path.read_text())

    assert artifact_path.suffix == ".nq"
    assert report["payload"]["artifact"]["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert marker["artifact_sha256"] == result["selected_artifact_sha256"]
    assert marker["report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert marker_path.stat().st_mtime_ns >= report_path.stat().st_mtime_ns
    assert report_path.stat().st_mtime_ns >= artifact_path.stat().st_mtime_ns


@pytest.mark.parametrize(
    ("profile", "regime", "family", "order", "message"),
    [
        ("development", "near_gaussian", "two_cell", 8, "source-pinned"),
        ("smoke", "boundary_heavy", "two_cell", 8, "not available"),
        ("smoke", "near_gaussian", "four_cell", 8, "not available"),
    ],
)
def test_driver_rejects_undeclared_cases_and_orders(
    profile: str,
    regime: str,
    family: str,
    order: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        driver.run_case(
            regime_name=regime,
            family=family,  # type: ignore[arg-type]
            quadrature_order=order,
            base_seed=731,
            profile=profile,  # type: ignore[arg-type]
            source_git_revision=_GIT_SHA,
            driver_sha256=_DRIVER_SHA,
        )


def test_driver_rejects_short_source_identity() -> None:
    with pytest.raises(ValueError, match="complete"):
        driver.run_case(
            regime_name="near_gaussian",
            family="two_cell",
            quadrature_order=8,
            base_seed=731,
            profile="smoke",
            source_git_revision="abc",
            driver_sha256=_DRIVER_SHA,
        )
