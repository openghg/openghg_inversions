"""Focused tests for the score-regularized N1 exact-oracle task driver."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.rjmcmc import score_regularized_flow_tiny_certify as certify
from examples.rjmcmc import score_regularized_flow_tiny_domains as tiny_domains
from examples.rjmcmc import score_regularized_flow_tiny_screen as screen
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_artifact import (
    ScoreRegularizedRootFlow,
)


FULL_SHA = "0" * 40


@pytest.fixture(scope="module")
def smoke_result() -> tuple[dict[str, Any], bytes]:
    """Run the real frozen-architecture smoke fit once for this module."""
    return screen.run_case(
        regime_name="near_gaussian",
        family="two_cell",
        training_sample_count=screen.SMOKE_SAMPLE_COUNTS[0],
        base_seed=screen.SMOKE_BASE_SEED,
        profile="smoke",
        source_git_revision=FULL_SHA,
        driver_sha256=screen._driver_sha256(),
    )


def test_development_protocol_is_frozen_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The complete development identity must match and reject placeholders."""
    assert screen._protocol_sha256() == screen.DEVELOPMENT_PROTOCOL_SHA256
    assert len(screen.DEVELOPMENT_PROTOCOL_SHA256) == 64
    payload = screen._protocol_payload()
    assert payload["training_sample_counts"] == (
        4_096,
        16_384,
        65_536,
        262_144,
    )
    assert payload["fit"] == {
        "initialization_count": 2,
        "optimizer": "adam",
        "learning_rate": 5.0e-4,
        "batch_size": 1_024,
        "score_microbatch_size": 64,
        "maximum_epochs": 100,
        "early_stopping_patience": 10,
        "internal_validation_proportion": 0.1,
        "objective": "nll_per_q_plus_raw_log_mass_score_mse_per_q",
        "mass_score_autodiff": (
            "forward-jvp-in-raw-log-mass-then-reverse-parameter-gradient"
        ),
        "selection": (
            "minimum independent model-selection composite loss then "
            "initialization index"
        ),
    }
    assert payload["cpu_xla_flags"] == (
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 "
        "--xla_cpu_parallel_codegen_split_count=1"
    )
    screen._validate_development_protocol()
    monkeypatch.setattr(screen, "DEVELOPMENT_PROTOCOL_SHA256", "")
    with pytest.raises(RuntimeError, match="has not been frozen"):
        screen._validate_development_protocol()


def test_private_seed_contract_replays_and_separates_streams() -> None:
    """Initialization and optimizer keys must use the exact declared bytes."""
    case_id = "near_gaussian__two_cell__root"
    expected: dict[str, int] = {}
    for stream in screen.PRIVATE_TASK_STREAMS:
        digest = hashlib.sha256(screen.PROTOCOL.encode("ascii"))
        digest.update((731).to_bytes(8, "little", signed=False))
        digest.update(case_id.encode("ascii"))
        digest.update(tiny_domains.TRAINING_DOMAIN.encode("ascii"))
        digest.update(stream.encode("ascii"))
        expected[stream] = int.from_bytes(digest.digest()[:4], "little")
        assert screen._task_stream_seed(
            731,
            case_id=case_id,
            stream_name=stream,
        ) == expected[stream]
    assert len(set(expected.values())) == len(expected)
    with pytest.raises(ValueError, match="unknown private"):
        screen._task_stream_seed(
            731,
            case_id=case_id,
            stream_name="protected-holdout",
        )
    with pytest.raises(ValueError, match="six frozen"):
        screen._task_stream_seed(
            731,
            case_id="unknown",
            stream_name="optimizer-0",
        )


def test_selection_uses_external_composite_then_initialization_index() -> None:
    """The external tie rule must not depend on histories or reporting data."""
    attempts = [
        {
            "initialization": 0,
            "model_selection": {"composite_loss": 1.5},
            "reporting_test": {"composite_loss": -100.0},
        },
        {
            "initialization": 1,
            "model_selection": {"composite_loss": 1.0},
            "reporting_test": {"composite_loss": 100.0},
        },
    ]
    assert screen._select_initialization(attempts) == 1
    attempts[0]["model_selection"]["composite_loss"] = 1.0
    assert screen._select_initialization(attempts) == 0
    with pytest.raises(ValueError, match="at least one"):
        screen._select_initialization([])
    attempts[1]["model_selection"]["composite_loss"] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        screen._select_initialization(attempts)


def test_smoke_fit_is_finite_domain_separated_and_exactly_replayable(
    smoke_result: tuple[dict[str, Any], bytes],
) -> None:
    """The small N0 profile must produce a strict finite normalized artifact."""
    result, artifact_bytes = smoke_result
    assert result["schema"] == screen.SCHEMA
    assert result["profile"] == "smoke"
    assert result["case_id"] == "near_gaussian__two_cell__root"
    assert result["task_pass"]
    assert result["fit_pass"]
    assert result["finite_score_pass"]
    assert result["artifact_replay_pass"]
    assert result["selected_generalization_pass"]
    assert len(result["attempts"]) == 1
    attempt = result["attempts"][0]
    for domain_name in ("model_selection", "reporting_test"):
        summary = attempt[domain_name]
        for key in (
            "nll_nat_per_draw",
            "nll_mcse_nat_per_draw",
            "mass_score_risk_per_dimension",
            "mass_score_risk_mcse_per_dimension",
            "observation_score_risk_per_dimension",
            "observation_score_risk_mcse_per_dimension",
            "composite_loss",
        ):
            assert np.isfinite(summary[key])
    evidence = result["domain_evidence"]
    domain_hashes = [
        evidence[name]["array_sha256"]["standardized_draw"]
        for name in tiny_domains.PUBLIC_DOMAINS
    ]
    assert len(set(domain_hashes)) == len(domain_hashes)
    assert {
        evidence[name]["spectrum_sha256"]
        for name in tiny_domains.PUBLIC_DOMAINS
    } == {result["spectrum_sha256"]}
    assert result["access_audit"] == {
        "realized_mf_accessed": False,
        "protected_catalogue_accessed": False,
        "paris_inversions_written": False,
    }
    observed_sha = hashlib.sha256(artifact_bytes).hexdigest()
    assert observed_sha == result["selected_artifact_sha256"]
    replay = ScoreRegularizedRootFlow.from_bytes(
        artifact_bytes,
        expected_sha256=observed_sha,
    )
    assert replay.to_bytes() == artifact_bytes
    assert replay.artifact_sha256 == observed_sha
    regime = screen.c1._regime("near_gaussian")
    _, _, _, observation, _ = screen.c1._case_arrays(
        regime,
        "two_cell",
    )
    assert np.isfinite(replay.log_likelihood(observation, 1.0))
    repeated_result, repeated_bytes = screen.run_case(
        regime_name="near_gaussian",
        family="two_cell",
        training_sample_count=screen.SMOKE_SAMPLE_COUNTS[0],
        base_seed=screen.SMOKE_BASE_SEED,
        profile="smoke",
        source_git_revision=FULL_SHA,
        driver_sha256=screen._driver_sha256(),
    )
    assert repeated_bytes == artifact_bytes
    assert repeated_result == result


def test_real_driver_schema_matches_the_strict_n1_certifier(
    smoke_result: tuple[dict[str, Any], bytes],
) -> None:
    """The certifier schema must not drift away from real driver output."""
    result, _ = smoke_result
    assert set(result) == certify._RESULT_KEYS
    assert set(result["attempts"][0]) == certify._ATTEMPT_KEYS
    assert (
        set(result["attempts"][0]["model_selection"])
        == certify._SCORE_SUMMARY_KEYS
    )
    assert set(result["evaluation"]) == certify._EVALUATION_KEYS
    for evidence in result["domain_evidence"].values():
        assert set(evidence) == certify._DOMAIN_EVIDENCE_KEYS


def test_create_only_publication_orders_and_authenticates_all_files(
    smoke_result: tuple[dict[str, Any], bytes],
    tmp_path: Path,
) -> None:
    """Artifact and report identities must be sealed before the last marker."""
    result, artifact_bytes = smoke_result
    paths = screen._write_result(tmp_path / "fresh", result, artifact_bytes)
    artifact_path = Path(paths["artifact"])
    report_path = Path(paths["report"])
    marker_path = Path(paths["completion_marker"])
    assert artifact_path.suffix == ".score-flow"
    envelope = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(envelope) == {"payload", "sha256"}
    assert envelope["sha256"] == screen._sha256_json(envelope["payload"])
    assert envelope["payload"]["artifact"] == {
        "path": artifact_path.name,
        "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
    }
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker == {
        "schema": "rjmcmc-score-regularized-flow-task-complete-v1",
        "case_id": result["case_id"],
        "training_sample_count": result["training_sample_count"],
        "base_seed": result["base_seed"],
        "task_pass": result["task_pass"],
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    assert artifact_path.stat().st_mtime_ns <= report_path.stat().st_mtime_ns
    assert report_path.stat().st_mtime_ns <= marker_path.stat().st_mtime_ns
    with pytest.raises(FileExistsError, match="refusing to replace"):
        screen._write_result(tmp_path / "fresh", result, artifact_bytes)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"profile": "other"}, "profile must"),
        ({"regime_name": "unknown"}, "six frozen"),
        ({"training_sample_count": 128}, "not source-pinned"),
        ({"base_seed": 1_877}, "base_seed is not"),
        ({"source_git_revision": "short"}, "full lower-case Git SHA"),
        ({"driver_sha256": "short"}, "lower-case SHA-256"),
        ({"driver_sha256": "0" * 64}, "does not match"),
    ],
)
def test_malformed_profile_case_size_seed_and_source_identities_refuse(
    overrides: dict[str, Any],
    message: str,
) -> None:
    """Malformed task identities must fail before simulation or fitting."""
    arguments: dict[str, Any] = {
        "regime_name": "near_gaussian",
        "family": "two_cell",
        "training_sample_count": screen.SMOKE_SAMPLE_COUNTS[0],
        "base_seed": screen.SMOKE_BASE_SEED,
        "profile": "smoke",
        "source_git_revision": FULL_SHA,
        "driver_sha256": screen._driver_sha256(),
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        screen.run_case(**arguments)


def test_output_paths_refuse_protected_and_production_domains(
    smoke_result: tuple[dict[str, Any], bytes],
    tmp_path: Path,
) -> None:
    """The driver must have no writable protected or production path."""
    result, artifact_bytes = smoke_result
    for name in ("protected-holdout", "PARIS_inversions"):
        output = tmp_path / name / "task"
        with pytest.raises(ValueError, match="must not be protected"):
            screen._write_result(output, result, artifact_bytes)
        assert not output.exists()


def test_output_directory_symlink_is_rejected(
    smoke_result: tuple[dict[str, Any], bytes],
    tmp_path: Path,
) -> None:
    result, artifact_bytes = smoke_result
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        screen._write_result(linked, result, artifact_bytes)
    assert not tuple(real.iterdir())
