"""Tests for the BP1 direct-NLE development merger."""

from __future__ import annotations

from pathlib import Path

from examples.rjmcmc import conditional_residual_image_flow_certify as certify
from examples.rjmcmc import conditional_residual_image_flow_tiny_screen as screen


def _passes(start_index: int) -> dict[str, dict[int, bool]]:
    """Return six cases whose passing suffix begins at ``start_index``."""
    return {
        f"{regime}__{family}__root": {
            sample_count: index >= start_index
            for index, sample_count in enumerate(
                screen.DEVELOPMENT_SAMPLE_COUNTS
            )
        }
        for regime, family, tiling in screen.DEVELOPMENT_MATRIX
        if tiling == "root"
    }


def test_common_lock_is_smallest_all_case_two_size_suffix() -> None:
    """The lock must be common, suffix-based, and contain at least two sizes."""
    passes = _passes(2)
    assert certify._common_lock(passes) == 65_536

    passes["boundary_heavy__four_cell__root"][65_536] = False
    assert certify._common_lock(passes) is None

    assert certify._common_lock(_passes(1)) == 16_384
    assert certify._common_lock(_passes(0)) == 4_096


def test_incomplete_matrix_publishes_no_lock(tmp_path: Path) -> None:
    """Missing tasks are recorded as a hard stop rather than excluded."""
    certificate, lock = certify.merge_development(
        tmp_path,
        expected_source_revision="0" * 40,
        expected_driver_sha256="1" * 64,
        expected_protocol_sha256="2" * 64,
    )

    assert lock is None
    assert not certificate["complete_matrix"]
    assert certificate["authenticated_task_count"] == 0
    assert len(certificate["missing_files"]) == 72
    assert certificate["terminal_reason"] == (
        "complete authenticated 24-task matrix is absent"
    )
