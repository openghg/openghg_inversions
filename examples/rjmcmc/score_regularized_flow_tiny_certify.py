#!/usr/bin/env python3
"""Strictly authenticate and merge the complete N1 development matrix.

This merger is intentionally pure: it never fits, repairs, or re-scores a
task.  It authenticates the 24 immutable task bundles, independently derives
each task decision from the report's scientific, external-generalization,
finite-score, and artifact-replay evidence, and then applies the predeclared
common-suffix rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from examples.rjmcmc import score_regularized_flow_tiny_domains as tiny_domains
from examples.rjmcmc import score_regularized_flow_tiny_screen as screen
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_artifact import (
    ScoreRegularizedRootFlow,
)

SCHEMA = "rjmcmc-score-regularized-flow-development-certificate-v1"
LOCK_SCHEMA = "rjmcmc-score-regularized-flow-common-lock-v1"
MERGE_MARKER_SCHEMA = "rjmcmc-score-regularized-flow-development-merge-complete-v1"
TASK_MARKER_SCHEMA = "rjmcmc-score-regularized-flow-task-complete-v1"

_SHA256_LENGTH = 64
_GIT_SHA_LENGTH = 40
_RESULT_KEYS = {
    "schema",
    "protocol",
    "profile",
    "source",
    "runtime",
    "case_id",
    "training_sample_count",
    "base_seed",
    "leading_rank",
    "spectrum_sha256",
    "scientific_input_sha256",
    "domain_evidence",
    "fit_controls",
    "attempts",
    "selected_initialization",
    "selection_rule",
    "selected_artifact_sha256",
    "artifact_replay_pass",
    "finite_score_pass",
    "fit_pass",
    "selected_generalization_pass",
    "evaluation",
    "access_audit",
    "task_pass",
}
_ATTEMPT_KEYS = {
    "initialization",
    "initialization_seed",
    "optimizer_seed",
    "epochs",
    "best_epoch",
    "stopped_early",
    "training_composite_loss_history",
    "internal_validation_composite_loss_history",
    "model_selection",
    "reporting_test",
    "absolute_model_selection_test_nll_gap_nat_per_draw",
    "pooled_nll_mcse_nat_per_draw",
    "generalization_threshold_nat_per_draw",
    "generalization_pass",
    "absolute_model_selection_test_mass_score_risk_gap",
    "pooled_mass_score_risk_mcse",
    "mass_score_five_mcse_agreement",
    "absolute_model_selection_test_observation_score_risk_gap",
    "pooled_observation_score_risk_mcse",
    "observation_score_five_mcse_agreement",
    "artifact_sha256",
}
_SCORE_SUMMARY_KEYS = {
    "sample_count",
    "nll_nat_per_draw",
    "nll_mcse_nat_per_draw",
    "nll_nat_per_dimension",
    "mass_score_risk_per_dimension",
    "mass_score_risk_mcse_per_dimension",
    "observation_score_risk_per_dimension",
    "observation_score_risk_mcse_per_dimension",
    "composite_loss",
}
_DOMAIN_EVIDENCE_KEYS = {
    "schema",
    "protocol",
    "case_id",
    "domain",
    "base_seed",
    "sample_count",
    "gamma_shape",
    "gamma_rate",
    "conditioning_center",
    "conditioning_scale",
    "stream_seeds",
    "scientific_input_sha256",
    "spectrum_sha256",
    "allocation_artifact_sha256",
    "array_sha256",
    "sha256",
}
_EVALUATION_KEYS = {
    "metrics",
    "checks",
    "scientific_pass",
    "posterior_summary",
    "posterior_errors_by_coordinate",
    "gradient_audits",
    "diagnostics",
}
_SELECTION_RULE = "minimum independent model-selection composite loss then initialization index"


def _canonical_json(payload: object) -> str:
    """Return strict canonical ASCII JSON."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(payload: object) -> str:
    """Return the SHA-256 identity of one canonical JSON value."""
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _sha256_path(path: Path) -> str:
    """Return the SHA-256 identity of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_digest(value: object, *, name: str, length: int = _SHA256_LENGTH) -> str:
    """Return one strict lower-case hexadecimal identity."""
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{name} must be a {length}-character lower-case hexadecimal digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a {length}-character lower-case hexadecimal digest") from error
    if decoded.hex() != value:
        raise ValueError(f"{name} must be lower-case hexadecimal")
    return value


def _require_mapping(value: object, *, name: str) -> dict[str, Any]:
    """Return one string-keyed JSON mapping."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> dict[str, Any]:
    """Return one mapping only when its key schema is exact."""
    result = _require_mapping(value, name=name)
    if set(result) != expected:
        raise ValueError(f"{name} has an unexpected schema")
    return result


def _finite(value: object, *, name: str) -> float:
    """Return one finite real, rejecting Booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Return one non-negative integer, rejecting Booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _read_canonical_json(path: Path) -> Any:
    """Read one regular, non-symlink canonical JSON file."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required regular non-symlink file is absent: {path.name}")
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read strict JSON: {path.name}") from error
    expected = (_canonical_json(value) + "\n").encode("ascii")
    if raw != expected:
        raise ValueError(f"JSON is not in canonical publication form: {path.name}")
    return value


def _case_ids() -> tuple[str, ...]:
    """Return the six source-pinned root case identifiers."""
    return tuple(
        f"{regime}__{family}__root"
        for regime, family, tiling in screen.DEVELOPMENT_MATRIX
        if tiling == "root"
    )


def _expected_stems() -> tuple[str, ...]:
    """Return all 24 canonical N1 task stems."""
    return tuple(
        f"{case_id}__S{sample_count}__base{screen.DEVELOPMENT_BASE_SEED}"
        for case_id in _case_ids()
        for sample_count in screen.DEVELOPMENT_SAMPLE_COUNTS
    )


def _common_lock(
    passes: Mapping[str, Mapping[int, bool]],
    *,
    case_ids: Sequence[str] | None = None,
    sample_counts: Sequence[int] | None = None,
) -> int | None:
    """Return the smallest common all-case, all-larger passing suffix.

    A qualifying suffix contains at least two predeclared sizes.  Missing
    results are failures; an earlier isolated pass cannot form a partial lock.
    """
    cases = tuple(_case_ids() if case_ids is None else case_ids)
    counts = tuple(screen.DEVELOPMENT_SAMPLE_COUNTS if sample_counts is None else sample_counts)
    if not cases:
        raise ValueError("at least one frozen case is required")
    if len(counts) < 2 or any(first >= second for first, second in zip(counts, counts[1:])):
        raise ValueError("sample counts must be strictly increasing with at least two sizes")
    if set(passes) != set(cases):
        return None
    for start in range(len(counts) - 1):
        suffix = counts[start:]
        if all(all(passes[case].get(sample_count) is True for sample_count in suffix) for case in cases):
            return counts[start]
    return None


def _validate_score_summary(
    value: object,
    *,
    name: str,
    expected_sample_count: int,
    leading_rank: int,
) -> dict[str, Any]:
    """Validate one finite model-selection or reporting-test score summary."""
    summary = _require_exact_keys(value, _SCORE_SUMMARY_KEYS, name=name)
    if summary["sample_count"] != expected_sample_count:
        raise ValueError(f"{name} has an unexpected sample count")
    finite = {
        key: _finite(summary[key], name=f"{name}.{key}") for key in _SCORE_SUMMARY_KEYS - {"sample_count"}
    }
    if finite["nll_mcse_nat_per_draw"] < 0.0:
        raise ValueError(f"{name} NLL MCSE must be non-negative")
    if finite["mass_score_risk_mcse_per_dimension"] < 0.0:
        raise ValueError(f"{name} mass-score MCSE must be non-negative")
    if finite["observation_score_risk_mcse_per_dimension"] < 0.0:
        raise ValueError(f"{name} observation-score MCSE must be non-negative")
    if finite["mass_score_risk_per_dimension"] < 0.0:
        raise ValueError(f"{name} mass-score risk must be non-negative")
    if finite["observation_score_risk_per_dimension"] < 0.0:
        raise ValueError(f"{name} observation-score risk must be non-negative")
    if not math.isclose(
        finite["nll_nat_per_dimension"],
        finite["nll_nat_per_draw"] / leading_rank,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{name} NLL dimension normalization is inconsistent")
    if not math.isclose(
        finite["composite_loss"],
        finite["nll_nat_per_dimension"] + finite["mass_score_risk_per_dimension"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{name} composite loss is inconsistent")
    return summary


def _validate_attempt(
    value: object,
    *,
    index: int,
    leading_rank: int,
) -> dict[str, Any]:
    """Validate one fitted initialization and recompute its external gates."""
    attempt = _require_exact_keys(
        value,
        _ATTEMPT_KEYS,
        name=f"attempt[{index}]",
    )
    if attempt["initialization"] != index:
        raise ValueError("attempt initialization order is not canonical")
    _nonnegative_integer(
        attempt["initialization_seed"],
        name=f"attempt[{index}].initialization_seed",
    )
    _nonnegative_integer(
        attempt["optimizer_seed"],
        name=f"attempt[{index}].optimizer_seed",
    )
    epochs = _nonnegative_integer(
        attempt["epochs"],
        name=f"attempt[{index}].epochs",
    )
    best_epoch = _nonnegative_integer(
        attempt["best_epoch"],
        name=f"attempt[{index}].best_epoch",
    )
    if epochs < 1 or best_epoch >= epochs:
        raise ValueError("attempt epoch evidence is inconsistent")
    if not isinstance(attempt["stopped_early"], bool):
        raise ValueError("attempt stopped_early must be Boolean")
    for history_name in (
        "training_composite_loss_history",
        "internal_validation_composite_loss_history",
    ):
        history = attempt[history_name]
        if (
            not isinstance(history, list)
            or len(history) != epochs
            or not all(
                math.isfinite(_finite(item, name=f"attempt[{index}].{history_name}")) for item in history
            )
        ):
            raise ValueError(f"attempt {history_name} is malformed")
    model_selection = _validate_score_summary(
        attempt["model_selection"],
        name=f"attempt[{index}].model_selection",
        expected_sample_count=screen.MODEL_SELECTION_SAMPLE_COUNT,
        leading_rank=leading_rank,
    )
    reporting_test = _validate_score_summary(
        attempt["reporting_test"],
        name=f"attempt[{index}].reporting_test",
        expected_sample_count=screen.REPORTING_TEST_SAMPLE_COUNT,
        leading_rank=leading_rank,
    )

    nll_gap = abs(float(model_selection["nll_nat_per_draw"]) - float(reporting_test["nll_nat_per_draw"]))
    pooled_nll_mcse = math.hypot(
        float(model_selection["nll_mcse_nat_per_draw"]),
        float(reporting_test["nll_mcse_nat_per_draw"]),
    )
    threshold = max(
        screen.GENERALIZATION_NAT_PER_DIMENSION * leading_rank,
        screen.GENERALIZATION_MCSE_MULTIPLIER * pooled_nll_mcse,
    )
    mass_gap = abs(
        float(model_selection["mass_score_risk_per_dimension"])
        - float(reporting_test["mass_score_risk_per_dimension"])
    )
    pooled_mass_mcse = math.hypot(
        float(model_selection["mass_score_risk_mcse_per_dimension"]),
        float(reporting_test["mass_score_risk_mcse_per_dimension"]),
    )
    observation_gap = abs(
        float(model_selection["observation_score_risk_per_dimension"])
        - float(reporting_test["observation_score_risk_per_dimension"])
    )
    pooled_observation_mcse = math.hypot(
        float(model_selection["observation_score_risk_mcse_per_dimension"]),
        float(reporting_test["observation_score_risk_mcse_per_dimension"]),
    )
    expected_scalars = {
        "absolute_model_selection_test_nll_gap_nat_per_draw": nll_gap,
        "pooled_nll_mcse_nat_per_draw": pooled_nll_mcse,
        "generalization_threshold_nat_per_draw": threshold,
        "absolute_model_selection_test_mass_score_risk_gap": mass_gap,
        "pooled_mass_score_risk_mcse": pooled_mass_mcse,
        "absolute_model_selection_test_observation_score_risk_gap": observation_gap,
        "pooled_observation_score_risk_mcse": pooled_observation_mcse,
    }
    for key, expected in expected_scalars.items():
        observed = _finite(attempt[key], name=f"attempt[{index}].{key}")
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"attempt {key} arithmetic is inconsistent")
    expected_booleans = {
        "generalization_pass": nll_gap <= threshold,
        "mass_score_five_mcse_agreement": (
            mass_gap <= screen.GENERALIZATION_MCSE_MULTIPLIER * pooled_mass_mcse
        ),
        "observation_score_five_mcse_agreement": (
            observation_gap <= screen.GENERALIZATION_MCSE_MULTIPLIER * pooled_observation_mcse
        ),
    }
    for key, expected in expected_booleans.items():
        if attempt[key] is not expected:
            raise ValueError(f"attempt {key} is inconsistent")
    _require_digest(attempt["artifact_sha256"], name=f"attempt[{index}].artifact_sha256")
    return attempt


def _validate_domain_evidence(
    value: object,
    *,
    case_id: str,
    base_seed: int,
    training_sample_count: int,
    spectrum_sha256: str,
    scientific_input_sha256: str,
) -> None:
    """Authenticate the three independent simulator-domain envelopes."""
    domains = _require_exact_keys(
        value,
        set(tiny_domains.PUBLIC_DOMAINS),
        name="domain_evidence",
    )
    expected_counts = {
        tiny_domains.TRAINING_DOMAIN: training_sample_count,
        tiny_domains.MODEL_SELECTION_VALIDATION_DOMAIN: (screen.MODEL_SELECTION_SAMPLE_COUNT),
        tiny_domains.DEVELOPMENT_REPORTING_TEST_DOMAIN: (screen.REPORTING_TEST_SAMPLE_COUNT),
    }
    for domain_name, expected_count in expected_counts.items():
        evidence = _require_exact_keys(
            domains[domain_name],
            _DOMAIN_EVIDENCE_KEYS,
            name=f"domain_evidence.{domain_name}",
        )
        expected_identity = {
            "schema": tiny_domains.EVIDENCE_SCHEMA,
            "protocol": screen.PROTOCOL,
            "case_id": case_id,
            "domain": domain_name,
            "base_seed": base_seed,
            "sample_count": expected_count,
            "spectrum_sha256": spectrum_sha256,
            "scientific_input_sha256": scientific_input_sha256,
        }
        for key, expected in expected_identity.items():
            if evidence[key] != expected:
                raise ValueError(f"domain evidence {domain_name}.{key} is inconsistent")
        _require_digest(
            evidence["allocation_artifact_sha256"],
            name=f"domain_evidence.{domain_name}.allocation_artifact_sha256",
        )
        _require_digest(
            evidence["sha256"],
            name=f"domain_evidence.{domain_name}.sha256",
        )
        for scalar in (
            "gamma_shape",
            "gamma_rate",
            "conditioning_center",
            "conditioning_scale",
        ):
            _finite(evidence[scalar], name=f"domain_evidence.{domain_name}.{scalar}")
        seeds = _require_mapping(
            evidence["stream_seeds"],
            name=f"domain_evidence.{domain_name}.stream_seeds",
        )
        if set(seeds) != set(tiny_domains.SIMULATOR_STREAMS):
            raise ValueError(f"domain evidence {domain_name} has unexpected streams")
        for stream, seed in seeds.items():
            _nonnegative_integer(seed, name=f"domain_evidence.{domain_name}.{stream}")
        arrays = _require_mapping(
            evidence["array_sha256"],
            name=f"domain_evidence.{domain_name}.array_sha256",
        )
        if set(arrays) != {
            "total_mass",
            "raw_log_mass",
            "allocation_residual",
            "gaussian_noise",
            "standardized_draw",
            "mass_score_target",
            "observation_score_target",
        }:
            raise ValueError(f"domain evidence {domain_name} has unexpected arrays")
        for array_name, digest in arrays.items():
            _require_digest(
                digest,
                name=f"domain_evidence.{domain_name}.array_sha256.{array_name}",
            )
        payload_without_hash = dict(evidence)
        recorded_hash = payload_without_hash.pop("sha256")
        if _sha256_json(payload_without_hash) != recorded_hash:
            raise ValueError(f"domain evidence {domain_name} hash does not replay")


def _validate_evaluation(value: object) -> tuple[dict[str, Any], bool]:
    """Validate C1 metrics and independently derive the scientific decision."""
    evaluation = _require_exact_keys(
        value,
        _EVALUATION_KEYS,
        name="evaluation",
    )
    metrics = _require_exact_keys(
        evaluation["metrics"],
        set(c1.THRESHOLDS) - {"between_bank_log_evidence_range_nat"},
        name="evaluation.metrics",
    )
    checks = _require_exact_keys(
        evaluation["checks"],
        set(metrics) | {"finite_normalized_likelihood"},
        name="evaluation.checks",
    )
    for metric_name, threshold in c1.THRESHOLDS.items():
        if metric_name not in metrics:
            continue
        metric = _finite(metrics[metric_name], name=f"evaluation.metrics.{metric_name}")
        if checks[metric_name] is not bool(metric <= threshold):
            raise ValueError(f"evaluation check {metric_name} is inconsistent")
    if checks["finite_normalized_likelihood"] is not True:
        scientific_pass = False
    else:
        scientific_pass = bool(all(value is True for value in checks.values()))
    if not all(isinstance(value, bool) for value in checks.values()):
        raise ValueError("evaluation checks must all be Boolean")
    if evaluation["scientific_pass"] is not scientific_pass:
        raise ValueError("evaluation scientific_pass is inconsistent")
    return evaluation, scientific_pass


def _validate_task(
    input_directory: Path,
    stem: str,
    *,
    expected_source_revision: str,
    expected_driver_sha256: str,
    expected_protocol_sha256: str,
) -> dict[str, Any]:
    """Authenticate one task report, fitted artifact, and final marker."""
    report_path = input_directory / f"{stem}.json"
    artifact_path = input_directory / f"{stem}.score-flow"
    marker_path = input_directory / f"{stem}.complete.json"

    envelope = _require_exact_keys(
        _read_canonical_json(report_path),
        {"payload", "sha256"},
        name=f"{stem} report envelope",
    )
    payload = _require_exact_keys(
        envelope["payload"],
        {"result", "artifact"},
        name=f"{stem} report payload",
    )
    if envelope["sha256"] != _sha256_json(payload):
        raise ValueError("report envelope digest does not replay")
    result = _require_exact_keys(
        payload["result"],
        _RESULT_KEYS,
        name=f"{stem} result",
    )
    if result["schema"] != screen.SCHEMA or result["profile"] != "development":
        raise ValueError("result schema or profile is not the frozen N1 development schema")
    protocol = _require_exact_keys(
        result["protocol"],
        {"name", "sha256", "payload"},
        name="result.protocol",
    )
    if (
        protocol["name"] != screen.PROTOCOL
        or protocol["sha256"] != expected_protocol_sha256
        or _canonical_json(protocol["payload"]) != _canonical_json(screen._protocol_payload())
        or _sha256_json(protocol["payload"]) != expected_protocol_sha256
    ):
        raise ValueError("result protocol identity does not match")
    if result["source"] != {
        "git_revision": expected_source_revision,
        "driver_sha256": expected_driver_sha256,
    }:
        raise ValueError("result source identity does not match")
    if result["runtime"] != screen._runtime_versions():
        raise ValueError("result runtime identity does not match")

    case_id, size_part, seed_part = stem.rsplit("__", 2)
    training_sample_count = int(size_part.removeprefix("S"))
    base_seed = int(seed_part.removeprefix("base"))
    if (
        case_id not in _case_ids()
        or training_sample_count not in screen.DEVELOPMENT_SAMPLE_COUNTS
        or base_seed != screen.DEVELOPMENT_BASE_SEED
        or result["case_id"] != case_id
        or result["training_sample_count"] != training_sample_count
        or result["base_seed"] != base_seed
    ):
        raise ValueError("result filename identity does not match its frozen task")
    leading_rank = _nonnegative_integer(
        result["leading_rank"],
        name="result.leading_rank",
    )
    if leading_rank < 1:
        raise ValueError("N1 leading rank must be positive")
    spectrum_sha256 = _require_digest(
        result["spectrum_sha256"],
        name="result.spectrum_sha256",
    )
    scientific_input_sha256 = _require_digest(
        result["scientific_input_sha256"],
        name="result.scientific_input_sha256",
    )
    _validate_domain_evidence(
        result["domain_evidence"],
        case_id=case_id,
        base_seed=base_seed,
        training_sample_count=training_sample_count,
        spectrum_sha256=spectrum_sha256,
        scientific_input_sha256=scientific_input_sha256,
    )
    if result["fit_controls"] != screen._fit_controls("development"):
        raise ValueError("result fit controls do not match the frozen protocol")
    if result["selection_rule"] != _SELECTION_RULE:
        raise ValueError("result selection rule does not match")
    attempts_raw = result["attempts"]
    if not isinstance(attempts_raw, list) or len(attempts_raw) != screen.INITIALIZATION_COUNT:
        raise ValueError("result does not contain both frozen initializations")
    attempts = [
        _validate_attempt(attempt, index=index, leading_rank=leading_rank)
        for index, attempt in enumerate(attempts_raw)
    ]
    selected_index = _nonnegative_integer(
        result["selected_initialization"],
        name="result.selected_initialization",
    )
    expected_selected_index = min(
        range(len(attempts)),
        key=lambda index: (
            attempts[index]["model_selection"]["composite_loss"],
            index,
        ),
    )
    if selected_index != expected_selected_index:
        raise ValueError("selected initialization does not replay")
    selected_attempt = attempts[selected_index]

    artifact_record = _require_exact_keys(
        payload["artifact"],
        {"path", "sha256"},
        name=f"{stem} artifact record",
    )
    if artifact_record["path"] != artifact_path.name:
        raise ValueError("artifact path is not canonical")
    if not artifact_path.is_file() or artifact_path.is_symlink():
        raise ValueError("artifact is absent or unsafe")
    artifact_sha256 = _sha256_path(artifact_path)
    _require_digest(artifact_record["sha256"], name="artifact record digest")
    if (
        artifact_record["sha256"] != artifact_sha256
        or result["selected_artifact_sha256"] != artifact_sha256
        or selected_attempt["artifact_sha256"] != artifact_sha256
    ):
        raise ValueError("artifact hashes do not agree")
    artifact_bytes = artifact_path.read_bytes()
    replay = ScoreRegularizedRootFlow.from_bytes(
        artifact_bytes,
        expected_sha256=artifact_sha256,
    )
    artifact_replay_pass = bool(
        replay.to_bytes() == artifact_bytes and replay.artifact_sha256 == artifact_sha256
    )
    expected_provenance = (
        f"{screen.PROTOCOL}:{case_id}:base={base_seed}:"
        f"initialization={selected_index}:git={expected_source_revision}"
    )
    training_evidence = result["domain_evidence"][tiny_domains.TRAINING_DOMAIN]
    if (
        replay.leading_rank != leading_rank
        or replay.spectrum.retained_rank != leading_rank
        or tiny_domains._spectrum_sha256(replay.spectrum) != spectrum_sha256
        or replay.gamma_shape != training_evidence["gamma_shape"]
        or replay.gamma_rate != training_evidence["gamma_rate"]
        or replay.condition_center != training_evidence["conditioning_center"]
        or replay.condition_scale != training_evidence["conditioning_scale"]
        or replay.initialization_seed != selected_attempt["initialization_seed"]
        or replay.source_provenance != expected_provenance
    ):
        raise ValueError("artifact scientific context does not match its task report")
    if result["artifact_replay_pass"] is not artifact_replay_pass:
        raise ValueError("recorded artifact replay decision is inconsistent")

    finite_score_pass = bool(
        all(
            math.isfinite(float(attempt[domain][metric]))
            for attempt in attempts
            for domain in ("model_selection", "reporting_test")
            for metric in (
                "mass_score_risk_per_dimension",
                "observation_score_risk_per_dimension",
            )
        )
    )
    fit_pass = bool(len(attempts) == screen.INITIALIZATION_COUNT and finite_score_pass)
    generalization_pass = bool(selected_attempt["generalization_pass"])
    _, scientific_pass = _validate_evaluation(result["evaluation"])
    task_pass = bool(
        fit_pass and generalization_pass and finite_score_pass and artifact_replay_pass and scientific_pass
    )
    expected_decisions = {
        "artifact_replay_pass": artifact_replay_pass,
        "finite_score_pass": finite_score_pass,
        "fit_pass": fit_pass,
        "selected_generalization_pass": generalization_pass,
        "task_pass": task_pass,
    }
    for key, expected in expected_decisions.items():
        if result[key] is not expected:
            raise ValueError(f"recorded decision {key} is inconsistent")
    if result["access_audit"] != {
        "realized_mf_accessed": False,
        "protected_catalogue_accessed": False,
        "paris_inversions_written": False,
    }:
        raise ValueError("task access audit is not sealed")

    report_sha256 = _sha256_path(report_path)
    marker = _read_canonical_json(marker_path)
    expected_marker = {
        "schema": TASK_MARKER_SCHEMA,
        "case_id": case_id,
        "training_sample_count": training_sample_count,
        "base_seed": base_seed,
        "task_pass": task_pass,
        "artifact_sha256": artifact_sha256,
        "report_sha256": report_sha256,
    }
    if marker != expected_marker:
        raise ValueError("completion marker does not authenticate the task bundle")
    return {
        "stem": stem,
        "case_id": case_id,
        "training_sample_count": training_sample_count,
        "base_seed": base_seed,
        "leading_rank": leading_rank,
        "task_pass": task_pass,
        "scientific_pass": scientific_pass,
        "generalization_pass": generalization_pass,
        "finite_score_pass": finite_score_pass,
        "artifact_replay_pass": artifact_replay_pass,
        "metrics": result["evaluation"]["metrics"],
        "score_risks": {
            "model_selection": selected_attempt["model_selection"],
            "reporting_test": selected_attempt["reporting_test"],
        },
        "log_evidence": result["evaluation"]["posterior_summary"]["log_evidence"],
        "spectrum_sha256": spectrum_sha256,
        "scientific_input_sha256": scientific_input_sha256,
        "artifact_sha256": artifact_sha256,
        "report_payload_sha256": envelope["sha256"],
        "report_sha256": report_sha256,
        "marker_sha256": _sha256_path(marker_path),
    }


def merge_development(
    input_directory: Path,
    *,
    expected_source_revision: str,
    expected_driver_sha256: str,
    expected_protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Authenticate the complete N1 matrix and derive the common lock."""
    _require_digest(
        expected_source_revision,
        name="expected_source_revision",
        length=_GIT_SHA_LENGTH,
    )
    _require_digest(expected_driver_sha256, name="expected_driver_sha256")
    _require_digest(expected_protocol_sha256, name="expected_protocol_sha256")
    if expected_driver_sha256 != screen._driver_sha256():
        raise ValueError("expected_driver_sha256 does not match the imported driver")
    if (
        expected_protocol_sha256 != screen.DEVELOPMENT_PROTOCOL_SHA256
        or expected_protocol_sha256 != screen._protocol_sha256()
    ):
        raise ValueError("expected_protocol_sha256 does not match the frozen imported protocol")
    if not input_directory.is_dir() or input_directory.is_symlink():
        raise ValueError("input_directory must be a real directory")

    expected_stems = _expected_stems()
    expected_files = {
        f"{stem}{suffix}" for stem in expected_stems for suffix in (".score-flow", ".json", ".complete.json")
    }
    observed_files = {path.name for path in input_directory.iterdir()}
    missing = sorted(expected_files - observed_files)
    unexpected = sorted(observed_files - expected_files)
    tasks: list[dict[str, Any]] = []
    validation_errors: list[dict[str, str]] = []
    if not missing and not unexpected:
        for stem in expected_stems:
            try:
                tasks.append(
                    _validate_task(
                        input_directory,
                        stem,
                        expected_source_revision=expected_source_revision,
                        expected_driver_sha256=expected_driver_sha256,
                        expected_protocol_sha256=expected_protocol_sha256,
                    )
                )
            except (OSError, TypeError, ValueError) as error:
                validation_errors.append({"stem": stem, "error": str(error)})
    complete_matrix = bool(
        not missing and not unexpected and not validation_errors and len(tasks) == len(expected_stems)
    )
    passes: dict[str, dict[int, bool]] = {case_id: {} for case_id in _case_ids()}
    if complete_matrix:
        for task in tasks:
            passes[task["case_id"]][task["training_sample_count"]] = bool(task["task_pass"])
    locked_sample_count = _common_lock(passes) if complete_matrix else None
    if not complete_matrix:
        terminal_reason: str | None = "complete authenticated 24-task N1 matrix is absent"
    elif locked_sample_count is None:
        terminal_reason = (
            "no common all-six-case all-larger passing suffix of length at least two; "
            "terminal N1 architecture stop"
        )
    else:
        terminal_reason = None

    certificate = {
        "schema": SCHEMA,
        "source": {
            "git_revision": expected_source_revision,
            "driver_sha256": expected_driver_sha256,
        },
        "protocol": {
            "name": screen.PROTOCOL,
            "sha256": expected_protocol_sha256,
        },
        "development_base_seed": screen.DEVELOPMENT_BASE_SEED,
        "sample_counts": list(screen.DEVELOPMENT_SAMPLE_COUNTS),
        "case_ids": list(_case_ids()),
        "expected_task_count": len(expected_stems),
        "authenticated_task_count": len(tasks),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "validation_errors": validation_errors,
        "complete_matrix": complete_matrix,
        "passes": {
            case_id: {
                str(sample_count): passes[case_id].get(sample_count, False)
                for sample_count in screen.DEVELOPMENT_SAMPLE_COUNTS
            }
            for case_id in _case_ids()
        },
        "locked_sample_count": locked_sample_count,
        "lock_published": locked_sample_count is not None,
        "terminal_reason": terminal_reason,
        "tasks": tasks,
    }
    lock: dict[str, Any] | None = None
    if locked_sample_count is not None:
        selected = {
            task["case_id"]: {
                "artifact_sha256": task["artifact_sha256"],
                "report_sha256": task["report_sha256"],
                "marker_sha256": task["marker_sha256"],
                "log_evidence": task["log_evidence"],
            }
            for task in tasks
            if task["training_sample_count"] == locked_sample_count
        }
        lock_payload = {
            "schema": LOCK_SCHEMA,
            "source": certificate["source"],
            "protocol": certificate["protocol"],
            "development_base_seed": screen.DEVELOPMENT_BASE_SEED,
            "confirmation_base_seeds": list(screen.CONFIRMATION_BASE_SEEDS),
            "locked_sample_count": locked_sample_count,
            "selected_tasks": selected,
            "certificate_payload_sha256": _sha256_json(certificate),
        }
        lock = {
            "payload": lock_payload,
            "sha256": _sha256_json(lock_payload),
        }
    return certificate, lock


def _publish_create_only(path: Path, payload: bytes) -> None:
    """Atomically hard-link one new file, never replacing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing evidence: {path}")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_merge(
    input_directory: Path,
    output_directory: Path,
    *,
    expected_source_revision: str,
    expected_driver_sha256: str,
    expected_protocol_sha256: str,
) -> dict[str, Any]:
    """Publish the certificate, optional common lock, and final marker."""
    resolved_output = output_directory.resolve(strict=False)
    lowered_parts = tuple(part.lower() for part in resolved_output.parts)
    if "paris_inversions" in lowered_parts or any("protected" in part for part in lowered_parts):
        raise ValueError("output_directory must not be protected or below PARIS_inversions")
    certificate, lock = merge_development(
        input_directory,
        expected_source_revision=expected_source_revision,
        expected_driver_sha256=expected_driver_sha256,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    if output_directory.is_symlink():
        raise ValueError("output_directory must not be a symlink")
    certificate_payload = {
        "payload": certificate,
        "sha256": _sha256_json(certificate),
    }
    certificate_path = output_directory / "development-certificate.json"
    _publish_create_only(
        certificate_path,
        (_canonical_json(certificate_payload) + "\n").encode("ascii"),
    )
    lock_path: Path | None = None
    if lock is not None:
        lock_path = output_directory / "common-lock.json"
        _publish_create_only(
            lock_path,
            (_canonical_json(lock) + "\n").encode("ascii"),
        )
    marker = {
        "schema": MERGE_MARKER_SCHEMA,
        "certificate_sha256": _sha256_path(certificate_path),
        "lock_published": lock_path is not None,
        "lock_sha256": _sha256_path(lock_path) if lock_path is not None else None,
        "terminal_reason": certificate["terminal_reason"],
    }
    _publish_create_only(
        output_directory / "MERGE_COMPLETE.json",
        (_canonical_json(marker) + "\n").encode("ascii"),
    )
    return marker


def main() -> None:
    """Merge one complete source-pinned N1 development matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    arguments = parser.parse_args()
    marker = publish_merge(
        arguments.input_directory,
        arguments.output_directory,
        expected_source_revision=arguments.expected_source_revision,
        expected_driver_sha256=arguments.expected_driver_sha256,
        expected_protocol_sha256=arguments.expected_protocol_sha256,
    )
    print(_canonical_json(marker))


if __name__ == "__main__":
    main()
