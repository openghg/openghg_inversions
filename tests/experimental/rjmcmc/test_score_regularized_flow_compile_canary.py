"""Focused tests for the score-regularized mixed-derivative compile canary."""

from __future__ import annotations

import json

import jax.numpy as jnp
import pytest

from examples.rjmcmc import score_regularized_flow_compile_canary as canary


FULL_SHA = "0" * 40


def test_canary_rejects_unknown_dimension_and_non_full_revision() -> None:
    with pytest.raises(ValueError, match="dimension must be"):
        canary.run_canary(
            dimension=2,
            source_git_revision=FULL_SHA,
        )
    with pytest.raises(ValueError, match="full Git SHA"):
        canary.run_canary(
            dimension=1,
            source_git_revision="short",
        )


def test_canary_gradient_summary_requires_finite_float64_leaves() -> None:
    assert canary._finite_gradient_summary(
        {"first": jnp.asarray([1.0, -2.0], dtype=jnp.float64)}
    ) == (1, 2, 2.0)
    with pytest.raises(RuntimeError, match="not all float64"):
        canary._finite_gradient_summary(
            {"first": jnp.asarray([1.0], dtype=jnp.float32)}
        )


def test_canary_publishes_marker_last_and_refuses_replacement(tmp_path) -> None:
    payload = {
        "schema": canary.SCHEMA,
        "dimension": 1,
        "compile_pass": True,
    }
    report, marker = canary._publish(tmp_path, payload)
    envelope = json.loads(report.read_text(encoding="utf-8"))
    completion = json.loads(marker.read_text(encoding="utf-8"))
    assert envelope["payload"] == payload
    assert envelope["sha256"] == canary._sha256_json(payload)
    assert completion["report"] == report.name
    assert marker.stat().st_mtime_ns >= report.stat().st_mtime_ns
    with pytest.raises(FileExistsError, match="refusing to replace"):
        canary._publish(tmp_path, payload)
