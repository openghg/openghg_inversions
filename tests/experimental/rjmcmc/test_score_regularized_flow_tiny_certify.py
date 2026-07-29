"""Strict tests for the score-regularized N1 development merger."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from examples.rjmcmc import score_regularized_flow_tiny_certify as certify
from examples.rjmcmc import score_regularized_flow_tiny_domains as tiny_domains
from examples.rjmcmc import score_regularized_flow_tiny_screen as screen
from openghg_inversions.experimental.rjmcmc.aggregation_error_exact_mixture import (
    RootResidualSpectrum,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_artifact import (
    GAMMA_LOG_MASS_CONDITIONING_RULE,
    ScoreRegularizedRootFlow,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_training import (
    make_score_regularized_conditional_flow,
)

REVISION = "a" * 40
DRIVER_SHA256 = screen._driver_sha256()
PROTOCOL_SHA256 = screen._protocol_sha256()
_HASH = "c" * 64


def _canonical_bytes(value: object) -> bytes:
    return (certify._canonical_json(value) + "\n").encode("ascii")


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value))


def _spectrum() -> RootResidualSpectrum:
    return RootResidualSpectrum(
        [0.4],
        [1.2],
        [[1.0]],
        [0.3],
        total_variance=0.3,
        discarded_variance=0.0,
        requested_retained_variance_fraction=1.0,
        eigenvalue_tolerance=1.0e-14,
        cell_alphas_sha256="d" * 64,
        design_sha256="e" * 64,
        noise_sd_sha256="f" * 64,
    )


@pytest.fixture(scope="module")
def initialized_flow() -> Any:
    """One cheap q=1 flow tree reused by every synthetic artifact."""
    return make_score_regularized_conditional_flow(1, source_seed=101)


def _score_summary(*, offset: float, sample_count: int) -> dict[str, Any]:
    nll = 1.0 + offset
    mass_risk = 0.2 + offset
    observation_risk = 0.3 + offset
    return {
        "sample_count": sample_count,
        "nll_nat_per_draw": nll,
        "nll_mcse_nat_per_draw": 0.01,
        "nll_nat_per_dimension": nll,
        "mass_score_risk_per_dimension": mass_risk,
        "mass_score_risk_mcse_per_dimension": 0.01,
        "observation_score_risk_per_dimension": observation_risk,
        "observation_score_risk_mcse_per_dimension": 0.01,
        "composite_loss": nll + mass_risk,
    }


def _attempt(
    index: int,
    *,
    artifact_sha256: str,
    selected_offset: float,
) -> dict[str, Any]:
    model_selection = _score_summary(
        offset=selected_offset,
        sample_count=screen.MODEL_SELECTION_SAMPLE_COUNT,
    )
    reporting_test = _score_summary(
        offset=selected_offset + 0.01,
        sample_count=screen.REPORTING_TEST_SAMPLE_COUNT,
    )
    nll_gap = 0.01
    pooled_mcse = 2.0**0.5 * 0.01
    generalization_threshold = max(
        screen.GENERALIZATION_NAT_PER_DIMENSION,
        screen.GENERALIZATION_MCSE_MULTIPLIER * pooled_mcse,
    )
    return {
        "initialization": index,
        "initialization_seed": 101 + index,
        "optimizer_seed": 201 + index,
        "epochs": 2,
        "best_epoch": 1,
        "stopped_early": False,
        "training_composite_loss_history": [1.5, 1.2],
        "internal_validation_composite_loss_history": [1.6, 1.3],
        "model_selection": model_selection,
        "reporting_test": reporting_test,
        "absolute_model_selection_test_nll_gap_nat_per_draw": nll_gap,
        "pooled_nll_mcse_nat_per_draw": pooled_mcse,
        "generalization_threshold_nat_per_draw": generalization_threshold,
        "generalization_pass": True,
        "absolute_model_selection_test_mass_score_risk_gap": nll_gap,
        "pooled_mass_score_risk_mcse": pooled_mcse,
        "mass_score_five_mcse_agreement": True,
        "absolute_model_selection_test_observation_score_risk_gap": nll_gap,
        "pooled_observation_score_risk_mcse": pooled_mcse,
        "observation_score_five_mcse_agreement": True,
        "artifact_sha256": artifact_sha256,
    }


def _domain_evidence(
    domain: str,
    *,
    case_id: str,
    sample_count: int,
    spectrum_sha256: str,
    scientific_input_sha256: str,
    artifact: ScoreRegularizedRootFlow,
) -> dict[str, Any]:
    payload = {
        "schema": tiny_domains.EVIDENCE_SCHEMA,
        "protocol": screen.PROTOCOL,
        "case_id": case_id,
        "domain": domain,
        "base_seed": screen.DEVELOPMENT_BASE_SEED,
        "sample_count": sample_count,
        "gamma_shape": artifact.gamma_shape,
        "gamma_rate": artifact.gamma_rate,
        "conditioning_center": artifact.condition_center,
        "conditioning_scale": artifact.condition_scale,
        "stream_seeds": {stream: index + 1 for index, stream in enumerate(tiny_domains.SIMULATOR_STREAMS)},
        "scientific_input_sha256": scientific_input_sha256,
        "spectrum_sha256": spectrum_sha256,
        "allocation_artifact_sha256": _HASH,
        "array_sha256": {
            name: _HASH
            for name in (
                "total_mass",
                "raw_log_mass",
                "allocation_residual",
                "gaussian_noise",
                "standardized_draw",
                "mass_score_target",
                "observation_score_target",
            )
        },
    }
    return {**payload, "sha256": certify._sha256_json(payload)}


def _evaluation(*, passing: bool) -> dict[str, Any]:
    metrics = {
        name: 0.0 if passing else threshold + 1.0
        for name, threshold in c1.THRESHOLDS.items()
        if name != "between_bank_log_evidence_range_nat"
    }
    checks = {name: value <= c1.THRESHOLDS[name] for name, value in metrics.items()}
    checks["finite_normalized_likelihood"] = True
    return {
        "metrics": metrics,
        "checks": checks,
        "scientific_pass": passing,
        "posterior_summary": {"log_evidence": -1.25},
        "posterior_errors_by_coordinate": {},
        "gradient_audits": [],
        "diagnostics": {"normalized_density_by_construction": True},
    }


def _write_task(
    directory: Path,
    *,
    case_id: str,
    sample_count: int,
    passing: bool,
    initialized_flow: Any,
) -> str:
    stem = f"{case_id}__S{sample_count}__base{screen.DEVELOPMENT_BASE_SEED}"
    artifact = ScoreRegularizedRootFlow(
        _spectrum(),
        1,
        43.0,
        43.0,
        initialized_flow,
        conditioning_rule_id=GAMMA_LOG_MASS_CONDITIONING_RULE,
        initialization_seed=101,
        source_provenance=(
            f"{screen.PROTOCOL}:{case_id}:base={screen.DEVELOPMENT_BASE_SEED}:initialization=0:git={REVISION}"
        ),
    )
    artifact_bytes = artifact.to_bytes()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    artifact_path = directory / f"{stem}.score-flow"
    artifact_path.write_bytes(artifact_bytes)

    spectrum_sha256 = tiny_domains._spectrum_sha256(artifact.spectrum)
    scientific_input_sha256 = "9" * 64
    attempts = [
        _attempt(0, artifact_sha256=artifact_sha256, selected_offset=0.0),
        _attempt(1, artifact_sha256="8" * 64, selected_offset=0.2),
    ]
    result = {
        "schema": screen.SCHEMA,
        "protocol": {
            "name": screen.PROTOCOL,
            "sha256": PROTOCOL_SHA256,
            "payload": screen._protocol_payload(),
        },
        "profile": "development",
        "source": {
            "git_revision": REVISION,
            "driver_sha256": DRIVER_SHA256,
        },
        "runtime": screen._runtime_versions(),
        "case_id": case_id,
        "training_sample_count": sample_count,
        "base_seed": screen.DEVELOPMENT_BASE_SEED,
        "leading_rank": 1,
        "spectrum_sha256": spectrum_sha256,
        "scientific_input_sha256": scientific_input_sha256,
        "domain_evidence": {
            domain: _domain_evidence(
                domain,
                case_id=case_id,
                sample_count={
                    tiny_domains.TRAINING_DOMAIN: sample_count,
                    tiny_domains.MODEL_SELECTION_VALIDATION_DOMAIN: (screen.MODEL_SELECTION_SAMPLE_COUNT),
                    tiny_domains.DEVELOPMENT_REPORTING_TEST_DOMAIN: (screen.REPORTING_TEST_SAMPLE_COUNT),
                }[domain],
                spectrum_sha256=spectrum_sha256,
                scientific_input_sha256=scientific_input_sha256,
                artifact=artifact,
            )
            for domain in tiny_domains.PUBLIC_DOMAINS
        },
        "fit_controls": screen._fit_controls("development"),
        "attempts": attempts,
        "selected_initialization": 0,
        "selection_rule": ("minimum independent model-selection composite loss then initialization index"),
        "selected_artifact_sha256": artifact_sha256,
        "artifact_replay_pass": True,
        "finite_score_pass": True,
        "fit_pass": True,
        "selected_generalization_pass": True,
        "evaluation": _evaluation(passing=passing),
        "access_audit": {
            "realized_mf_accessed": False,
            "protected_catalogue_accessed": False,
            "paris_inversions_written": False,
        },
        "task_pass": passing,
    }
    payload = {
        "result": result,
        "artifact": {
            "path": artifact_path.name,
            "sha256": artifact_sha256,
        },
    }
    report = {"payload": payload, "sha256": certify._sha256_json(payload)}
    report_path = directory / f"{stem}.json"
    _write_canonical(report_path, report)
    marker = {
        "schema": certify.TASK_MARKER_SCHEMA,
        "case_id": case_id,
        "training_sample_count": sample_count,
        "base_seed": screen.DEVELOPMENT_BASE_SEED,
        "task_pass": passing,
        "artifact_sha256": artifact_sha256,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    _write_canonical(directory / f"{stem}.complete.json", marker)
    return stem


def _write_matrix(
    directory: Path,
    initialized_flow: Any,
    *,
    pass_rule: Callable[[str, int], bool],
) -> None:
    directory.mkdir()
    for case_id in certify._case_ids():
        for sample_count in screen.DEVELOPMENT_SAMPLE_COUNTS:
            _write_task(
                directory,
                case_id=case_id,
                sample_count=sample_count,
                passing=pass_rule(case_id, sample_count),
                initialized_flow=initialized_flow,
            )


def _read(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="ascii"))
    assert path.read_bytes() == _canonical_bytes(result)
    return result


def _rewrite_report_and_marker(
    directory: Path,
    stem: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    report_path = directory / f"{stem}.json"
    report = _read(report_path)
    mutate(report)
    report["sha256"] = certify._sha256_json(report["payload"])
    _write_canonical(report_path, report)
    marker_path = directory / f"{stem}.complete.json"
    marker = _read(marker_path)
    marker["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    _write_canonical(marker_path, marker)


def test_valid_matrix_locks_the_smallest_common_two_size_suffix(
    tmp_path: Path,
    initialized_flow: Any,
) -> None:
    inputs = tmp_path / "tasks"
    _write_matrix(
        inputs,
        initialized_flow,
        pass_rule=lambda _case, size: size >= 65_536,
    )
    outputs = tmp_path / "merge"
    marker = certify.publish_merge(
        inputs,
        outputs,
        expected_source_revision=REVISION,
        expected_driver_sha256=DRIVER_SHA256,
        expected_protocol_sha256=PROTOCOL_SHA256,
    )

    certificate = _read(outputs / "development-certificate.json")
    lock = _read(outputs / "common-lock.json")
    assert certificate["sha256"] == certify._sha256_json(certificate["payload"])
    assert certificate["payload"]["complete_matrix"] is True
    assert certificate["payload"]["authenticated_task_count"] == 24
    assert certificate["payload"]["locked_sample_count"] == 65_536
    assert lock["payload"]["locked_sample_count"] == 65_536
    assert len(lock["payload"]["selected_tasks"]) == 6
    assert marker["lock_published"] is True
    assert _read(outputs / "MERGE_COMPLETE.json") == marker


def test_no_two_size_suffix_is_a_published_terminal_hard_stop(
    tmp_path: Path,
    initialized_flow: Any,
) -> None:
    inputs = tmp_path / "tasks"
    _write_matrix(
        inputs,
        initialized_flow,
        pass_rule=lambda _case, size: size == 262_144,
    )
    outputs = tmp_path / "merge"
    marker = certify.publish_merge(
        inputs,
        outputs,
        expected_source_revision=REVISION,
        expected_driver_sha256=DRIVER_SHA256,
        expected_protocol_sha256=PROTOCOL_SHA256,
    )
    certificate = _read(outputs / "development-certificate.json")["payload"]
    assert certificate["complete_matrix"] is True
    assert certificate["locked_sample_count"] is None
    assert certificate["lock_published"] is False
    assert "terminal N1 architecture stop" in certificate["terminal_reason"]
    assert not (outputs / "common-lock.json").exists()
    assert marker["lock_published"] is False
    assert marker["lock_sha256"] is None


@pytest.mark.parametrize("fault", ["missing", "unexpected"])
def test_missing_or_unexpected_filename_rejects_the_entire_matrix(
    tmp_path: Path,
    initialized_flow: Any,
    fault: str,
) -> None:
    inputs = tmp_path / "tasks"
    _write_matrix(inputs, initialized_flow, pass_rule=lambda _case, _size: True)
    first_stem = certify._expected_stems()[0]
    if fault == "missing":
        (inputs / f"{first_stem}.complete.json").unlink()
    else:
        (inputs / "unexpected.txt").write_text("unexpected", encoding="ascii")
    certificate, lock = certify.merge_development(
        inputs,
        expected_source_revision=REVISION,
        expected_driver_sha256=DRIVER_SHA256,
        expected_protocol_sha256=PROTOCOL_SHA256,
    )
    assert certificate["complete_matrix"] is False
    assert certificate[f"{fault}_files"]
    assert certificate["authenticated_task_count"] == 0
    assert certificate["locked_sample_count"] is None
    assert lock is None


def _tamper_report(report: dict[str, Any]) -> None:
    report["payload"]["result"]["evaluation"]["metrics"]["absolute_log_evidence_error_nat"] = 1.0


def _tamper_source(report: dict[str, Any]) -> None:
    report["payload"]["result"]["source"]["git_revision"] = "7" * 40


def _tamper_protocol(report: dict[str, Any]) -> None:
    report["payload"]["result"]["protocol"]["sha256"] = "6" * 64


@pytest.mark.parametrize(
    "fault",
    ["report", "artifact", "marker", "source", "protocol"],
)
def test_tampered_task_evidence_is_rejected_without_a_partial_lock(
    tmp_path: Path,
    initialized_flow: Any,
    fault: str,
) -> None:
    inputs = tmp_path / "tasks"
    _write_matrix(inputs, initialized_flow, pass_rule=lambda _case, _size: True)
    stem = certify._expected_stems()[0]
    if fault == "artifact":
        with (inputs / f"{stem}.score-flow").open("ab") as handle:
            handle.write(b"tamper")
    elif fault == "marker":
        marker_path = inputs / f"{stem}.complete.json"
        marker = _read(marker_path)
        marker["report_sha256"] = "0" * 64
        _write_canonical(marker_path, marker)
    else:
        mutation = {
            "report": _tamper_report,
            "source": _tamper_source,
            "protocol": _tamper_protocol,
        }[fault]
        _rewrite_report_and_marker(inputs, stem, mutation)

    certificate, lock = certify.merge_development(
        inputs,
        expected_source_revision=REVISION,
        expected_driver_sha256=DRIVER_SHA256,
        expected_protocol_sha256=PROTOCOL_SHA256,
    )
    assert certificate["complete_matrix"] is False
    assert certificate["validation_errors"]
    assert certificate["locked_sample_count"] is None
    assert certificate["lock_published"] is False
    assert lock is None


def test_publication_is_create_only_and_preserves_first_evidence(
    tmp_path: Path,
    initialized_flow: Any,
) -> None:
    inputs = tmp_path / "tasks"
    _write_matrix(
        inputs,
        initialized_flow,
        pass_rule=lambda _case, size: size >= 65_536,
    )
    outputs = tmp_path / "merge"
    arguments = {
        "expected_source_revision": REVISION,
        "expected_driver_sha256": DRIVER_SHA256,
        "expected_protocol_sha256": PROTOCOL_SHA256,
    }
    certify.publish_merge(inputs, outputs, **arguments)
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs.iterdir()}
    with pytest.raises(FileExistsError, match="refusing to replace"):
        certify.publish_merge(inputs, outputs, **arguments)
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs.iterdir()}
    assert after == before
    assert set(after) == {
        "development-certificate.json",
        "common-lock.json",
        "MERGE_COMPLETE.json",
    }


def test_common_lock_never_accepts_an_isolated_or_partial_pass() -> None:
    cases = ("a", "b")
    counts = (1, 2, 3, 4)
    passes = {
        "a": {1: False, 2: True, 3: True, 4: True},
        "b": {1: False, 2: True, 3: False, 4: True},
    }
    assert (
        certify._common_lock(
            passes,
            case_ids=cases,
            sample_counts=counts,
        )
        is None
    )
    complete = copy.deepcopy(passes)
    complete["b"][3] = True
    assert (
        certify._common_lock(
            complete,
            case_ids=cases,
            sample_counts=counts,
        )
        == 2
    )
