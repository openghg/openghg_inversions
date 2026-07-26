#!/usr/bin/env python3
"""Merge and certify restartable residual-image GMM development shards.

The first phase authenticates the frozen 24-shard development matrix, derives
the common two-size suffix lock, and publishes a self-digested lock envelope.
The second phase re-authenticates those shards and exactly 18 lock-bound
confirmation shards, recomputes every gate, and publishes a self-digested
development certificate.  Neither phase reads the protected holdout.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from examples.rjmcmc import conditional_residual_image_gmm_tiny_screen as gmm
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mdn import (
    ConditionalResidualImageMDN,
    ResidualImageContext,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)

LOCK_SCHEMA = "conditional-residual-image-gmm-common-lock-v1"
CERTIFICATE_SCHEMA = "conditional-residual-image-gmm-development-certificate-v1"
CERTIFICATION_PROTOCOL = "conditional-residual-image-gmm-sharded-certifier-v1"
_FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MODEL_GATE_NAMES = tuple(name for name in c1.THRESHOLDS if name != "between_bank_log_evidence_range_nat")
# The independent simulator domains contain up to 131,072 rows.  Different
# BP1 CPU families produced at most 1.11e-9 nat/draw variation in their
# vectorized log-density reductions, while the scientific gates are no
# tighter than 2e-2.  These remain replay-roundoff allowances, not model
# tolerances; domain, prefix, envelope, and artifact identities remain exact.
_GENERALIZATION_NLL_REPLAY_ABS_TOL = 1.0e-8
_GENERALIZATION_MCSE_REPLAY_ABS_TOL = 1.0e-10
# Exact-grid science replay is much smaller.  The largest observed cross-node
# difference was 1.46e-11 in one learned-gradient coordinate.
_SCIENTIFIC_REPLAY_ABS_TOL = 1.0e-10
_REPLAY_REL_TOL = 1.0e-11
_GENERALIZATION_NLL_FIELDS = frozenset(
    {
        "validation_nll_nat_per_draw",
        "simulator_test_nll_nat_per_draw",
        "absolute_nll_gap_nat_per_draw",
    }
)

FloatArray = NDArray[np.float64]
DomainBank = tuple[dict[str, FloatArray], dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class _ExactCase:
    """Exact source-pinned numerical objects needed for one shard replay."""

    case_id: str
    input_sha256: str
    aggregation: AdditiveDirichletAggregation
    labels: c1.IntArray
    context: ResidualImageContext
    observation: FloatArray
    masses: FloatArray
    log_prior: FloatArray
    exact_log_likelihood: FloatArray
    exact_summary: dict[str, Any]
    gradient_states: list[dict[str, Any]]
    validation_state_mask: NDArray[np.bool_]
    quadrature: dict[str, int]


def _canonical_json(payload: object) -> str:
    """Return the driver's strict canonical JSON representation."""
    return c1._canonical_json(payload)


def _sha256_bytes(value: bytes) -> str:
    """Return the lower-case SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def _driver_source_sha256() -> str:
    """Return the exact current scientific-driver identity."""
    return _sha256_bytes(Path(gmm.__file__).read_bytes())


def _certifier_source_sha256() -> str:
    """Return the exact current certifier identity."""
    return _sha256_bytes(Path(__file__).read_bytes())


def _certification_protocol_sha256() -> str:
    """Return the identity of the frozen sharded certification protocol."""
    return c1._sha256_json(
        {
            "certification_protocol": CERTIFICATION_PROTOCOL,
            "lock_schema": LOCK_SCHEMA,
            "certificate_schema": CERTIFICATE_SCHEMA,
            "scientific_schema": gmm.SCHEMA,
            "scientific_protocol": gmm.PROTOCOL,
            "frozen_development_protocol_sha256": gmm.DEVELOPMENT_PROTOCOL_SHA256,
            "matrix": gmm.DEVELOPMENT_MATRIX,
            "sample_counts": gmm.DEVELOPMENT_SAMPLE_COUNTS,
            "development_selection_seed": gmm.DEVELOPMENT_SELECTION_SEED,
            "confirmation_seeds": gmm.CONFIRMATION_SEEDS,
            "minimum_passing_suffix_length": 2,
            "thresholds": c1.THRESHOLDS,
            "generalization_nll_replay_abs_tolerance": (_GENERALIZATION_NLL_REPLAY_ABS_TOL),
            "generalization_mcse_replay_abs_tolerance": (_GENERALIZATION_MCSE_REPLAY_ABS_TOL),
            "scientific_replay_abs_tolerance": _SCIENTIFIC_REPLAY_ABS_TOL,
            "replay_relative_tolerance": _REPLAY_REL_TOL,
            "development_shard_count": 24,
            "confirmation_shard_count": 18,
        }
    )


def _git_output(source_directory: Path, *arguments: str) -> str:
    """Run one bounded read-only Git query and return stdout."""
    result = subprocess.run(
        ["git", "-C", str(source_directory), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _validate_live_source(source_directory: Path, expected_source_revision: str) -> None:
    """Require the imported driver and certifier to reside in the pinned worktree."""
    imported_root = Path(__file__).resolve().parents[2]
    if source_directory.resolve() != imported_root:
        raise ValueError("source directory does not contain the imported certifier")
    if _git_output(source_directory, "rev-parse", "HEAD").strip() != expected_source_revision:
        raise ValueError("live source HEAD does not match the expected revision")
    if _git_output(source_directory, "status", "--porcelain") not in ("", "?? .pixi\n"):
        raise ValueError("live source contains changes other than the authenticated .pixi link")


def _require_equal(observed: object, expected: object, label: str) -> None:
    """Reject values that differ under canonical JSON comparison."""
    if _canonical_json(observed) != _canonical_json(expected):
        raise ValueError(f"{label} does not match the frozen protocol")


def _replay_tolerance(actual: float, expected: float, *, absolute_floor: float) -> float:
    """Return a frozen scale-aware cross-node replay tolerance."""
    return max(
        absolute_floor,
        _REPLAY_REL_TOL * max(abs(actual), abs(expected)),
    )


def _require_replayed_science(observed: object, replayed: object, label: str) -> None:
    """Compare scientific replay recursively with scoped float tolerance."""
    if isinstance(replayed, dict):
        if not isinstance(observed, dict) or set(observed) != set(replayed):
            raise ValueError(f"{label} does not match the authenticated replay")
        for key, expected in replayed.items():
            _require_replayed_science(
                observed[key],
                expected,
                f"{label}.{key}",
            )
        return
    if isinstance(replayed, list):
        if not isinstance(observed, list) or len(observed) != len(replayed):
            raise ValueError(f"{label} does not match the authenticated replay")
        for index, (value, expected) in enumerate(zip(observed, replayed, strict=True)):
            _require_replayed_science(
                value,
                expected,
                f"{label}[{index}]",
            )
        return
    if isinstance(replayed, float):
        actual = _finite_number(observed, label)
        tolerance = _replay_tolerance(
            actual,
            replayed,
            absolute_floor=_SCIENTIFIC_REPLAY_ABS_TOL,
        )
        if not math.isclose(
            actual,
            replayed,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError(f"{label} does not match the authenticated replay")
        return
    if type(observed) is not type(replayed) or observed != replayed:
        raise ValueError(f"{label} does not match the authenticated replay")


def _four_bank_evidence_range_gate(evidence: list[float], *, label: str) -> tuple[float, bool]:
    """Evaluate the evidence-range gate outside its replay-roundoff margin."""
    if len(evidence) != 4 or any(not math.isfinite(value) for value in evidence):
        raise ValueError(f"{label} must contain four finite log evidences")
    evidence_range = float(max(evidence) - min(evidence))
    threshold = c1.THRESHOLDS["between_bank_log_evidence_range_nat"]
    endpoint_tolerance = max(
        _replay_tolerance(
            value,
            value,
            absolute_floor=_SCIENTIFIC_REPLAY_ABS_TOL,
        )
        for value in evidence
    )
    if abs(evidence_range - threshold) <= 2.0 * endpoint_tolerance:
        raise ValueError(f"{label} is too close to its pass threshold for portable replay")
    return evidence_range, evidence_range <= threshold


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    """Return a string-keyed JSON object or reject it."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _require_list(value: object, label: str) -> list[Any]:
    """Return a JSON array or reject it."""
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return cast(list[Any], value)


def _finite_number(value: object, label: str) -> float:
    """Return one finite non-Boolean number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], str]:
    """Read one canonical JSON object and return it with its raw digest."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path} must be one regular non-symlink file")
    raw = path.read_bytes()

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        text = raw.decode("ascii")
        payload = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{path} contains non-finite JSON value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not strict ASCII JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    if text != f"{_canonical_json(payload)}\n":
        raise ValueError(f"{path} is not newline-terminated canonical JSON")
    return cast(dict[str, Any], payload), _sha256_bytes(raw)


def _regular_file_names(directory: Path, label: str) -> set[str]:
    """Return the exact flat regular-file membership of a real directory."""
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"{label} must be a real directory")
    names: set[str] = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{label} contains a symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file() or path.parent != directory:
            raise ValueError(f"{label} must contain only flat regular files")
        names.add(path.name)
    return names


def _case_ids() -> tuple[str, ...]:
    """Return the six frozen development case IDs."""
    return tuple("__".join(case) for case in gmm.DEVELOPMENT_MATRIX)


def _exact_case(expected_case: tuple[str, str, str]) -> _ExactCase:
    """Reconstruct one source-pinned exact case without fitting a model."""
    regime_name, family_name, tiling = expected_case
    if tiling != "root":
        raise ValueError("GMM development certification is root-only")
    family = cast(c1.Family, family_name)
    regime = c1._regime(regime_name)
    shapes, rate, design, observation, noise = c1._case_arrays(regime, family)
    labels = c1.labels_for_tiling(family, "root")
    masses, log_prior = c1._mass_grid(
        shapes=shapes,
        rate=rate,
        family=family,
        tiling="root",
        total_order=regime.total_order,
        fraction_order=regime.fraction_order,
    )
    exact_log_likelihood = c1._exact_log_likelihood(
        masses=masses,
        shapes=shapes,
        rate=rate,
        design=design,
        observation=observation,
        noise=noise,
        family=family,
        tiling="root",
        total_order=regime.total_order,
        fraction_order=regime.fraction_order,
    )
    exact_summary = c1._posterior_summary(masses, log_prior, exact_log_likelihood)
    prior_mean_coordinate = c1._anchor_coordinate(shapes, rate, labels)

    def exact_function(value: FloatArray) -> float:
        return float(
            c1._exact_log_likelihood(
                masses=c1.coordinate_to_masses(value)[None, :],
                shapes=shapes,
                rate=rate,
                design=design,
                observation=observation,
                noise=noise,
                family=family,
                tiling="root",
                total_order=regime.total_order,
                fraction_order=regime.fraction_order,
            )[0]
        )

    gradient_states = [
        {
            "state_id": state_id,
            "coordinate": coordinate.tolist(),
            "exact_coordinate_gradient": c1._centered_gradient(
                exact_function,
                coordinate,
            ).tolist(),
        }
        for state_id, coordinate in c1._gradient_state_coordinates(
            masses=masses,
            log_prior=log_prior,
            exact_log_likelihood=exact_log_likelihood,
            prior_mean_coordinate=prior_mean_coordinate,
        )
    ]
    aggregation = AdditiveDirichletAggregation(
        shapes,
        design,
        noise,
        np.eye(observation.size, dtype=np.float64),
    )
    case_id = "__".join(expected_case)
    context = ResidualImageContext.from_aggregation(
        aggregation,
        labels,
        np.arange(shapes.size, dtype=np.int64),
        source_provenance=f"{gmm.PROTOCOL}:{case_id}:residual-image-context",
    )
    return _ExactCase(
        case_id=case_id,
        input_sha256=c1._case_input_sha256(
            regime,
            family,
            "root",
            regime.total_order,
            regime.fraction_order,
        ),
        aggregation=aggregation,
        labels=labels,
        context=context,
        observation=np.asarray(observation, dtype=np.float64),
        masses=masses,
        log_prior=log_prior,
        exact_log_likelihood=exact_log_likelihood,
        exact_summary=exact_summary,
        gradient_states=gradient_states,
        validation_state_mask=c1._development_validation_state_mask(
            masses,
            total_order=regime.total_order,
            fraction_order=regime.fraction_order,
        ),
        quadrature={
            "total_order": regime.total_order,
            "fraction_order": regime.fraction_order,
            "mass_state_count": int(masses.shape[0]),
        },
    )


def _exact_cases() -> dict[str, _ExactCase]:
    """Build each frozen exact case once for one certification phase."""
    return {case.case_id: case for case in map(_exact_case, gmm.DEVELOPMENT_MATRIX)}


def _validate_exact_case_record(
    case: Mapping[str, Any],
    exact: _ExactCase,
    *,
    label: str,
) -> None:
    """Bind one reported case to its source-reconstructed numerical inputs."""
    expected = {
        "case_id": exact.case_id,
        "input_sha256": exact.input_sha256,
        "context_sha256": exact.context.artifact_sha256,
        "residual_image_rank": exact.context.residual_rank,
        "quadrature": exact.quadrature,
        "exact_posterior_summary": exact.exact_summary,
    }
    for field, value in expected.items():
        _require_equal(case.get(field), value, f"{label} exact {field}")


def _domain_bank(
    exact: _ExactCase,
    *,
    sample_count: int,
    base_seed: int,
    cache: dict[tuple[str, int, int], DomainBank],
) -> DomainBank:
    """Regenerate and cache one frozen simulator-domain bank."""
    training_count = (
        max(gmm.DEVELOPMENT_SAMPLE_COUNTS) if base_seed == gmm.DEVELOPMENT_SELECTION_SEED else sample_count
    )
    key = (exact.case_id, base_seed, training_count)
    if key not in cache:
        cache[key] = gmm._domain_draw_bundle(
            exact.aggregation,
            exact.labels,
            exact.context,
            case_id=exact.case_id,
            training_sample_count=training_count,
            validation_sample_count=gmm.VALIDATION_SAMPLE_COUNT,
            test_sample_count=gmm.TEST_SAMPLE_COUNT,
            base_seed=base_seed,
        )
    return cache[key]


def _development_filename(case_id: str, sample_count: int) -> str:
    """Return the authoritative development-shard filename."""
    return f"{case_id}__S{sample_count}.json"


def _confirmation_filename(case_id: str, seed: int) -> str:
    """Return the authoritative confirmation-shard filename."""
    return f"{case_id}__seed{seed}.json"


def _validate_report_identity(
    report: Mapping[str, Any],
    *,
    case_id: str,
    expected_source_revision: str,
    expected_driver_sha256: str,
) -> None:
    """Authenticate identities shared by development and confirmation reports."""
    required = {
        "schema": gmm.SCHEMA,
        "protocol": gmm.PROTOCOL,
        "profile": "development",
        "selected_case_id": case_id,
        "per_case_atomic_output": True,
        "source_git_revision": expected_source_revision,
        "driver_sha256": expected_driver_sha256,
        "a1_definitions_sha256": c1.A1_DEFINITIONS_SHA256,
        "protocol_sha256": gmm._protocol_sha256(),
        "frozen_development_protocol_sha256": gmm.DEVELOPMENT_PROTOCOL_SHA256,
        "sample_counts": list(gmm.DEVELOPMENT_SAMPLE_COUNTS),
        "repeat_seeds": [gmm.DEVELOPMENT_SELECTION_SEED, *gmm.CONFIRMATION_SEEDS],
        "matrix_catalogue": gmm.matrix_catalogue(),
        "development_pass": False,
        "eligible_for_protected_holdout": False,
        "protected_holdout_pass": None,
        "scientific_pass": False,
        "scientific_pass_available": False,
        "structural_inference_licensed": False,
        "held_out_information_read": False,
    }
    for field, expected in required.items():
        _require_equal(report.get(field), expected, f"{case_id} report {field}")


def _validate_generalization(value: object, label: str) -> bool:
    """Recompute the frozen validation-versus-test generalization gate."""
    result = _require_mapping(value, f"{label} generalization")
    expected_keys = {
        "residual_dimension",
        "validation_nll_nat_per_draw",
        "simulator_test_nll_nat_per_draw",
        "absolute_nll_gap_nat_per_draw",
        "validation_nll_mcse_nat_per_draw",
        "simulator_test_nll_mcse_nat_per_draw",
        "pooled_nll_mcse_nat_per_draw",
        "fixed_floor_nat_per_draw",
        "threshold_nat_per_draw",
        "pass",
    }
    if set(result) != expected_keys:
        raise ValueError(f"{label} generalization has an unexpected schema")
    dimension = result["residual_dimension"]
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError(f"{label} residual dimension is invalid")
    validation_nll = _finite_number(result["validation_nll_nat_per_draw"], label)
    test_nll = _finite_number(result["simulator_test_nll_nat_per_draw"], label)
    gap = _finite_number(result["absolute_nll_gap_nat_per_draw"], label)
    validation_mcse = _finite_number(result["validation_nll_mcse_nat_per_draw"], label)
    test_mcse = _finite_number(result["simulator_test_nll_mcse_nat_per_draw"], label)
    pooled = _finite_number(result["pooled_nll_mcse_nat_per_draw"], label)
    floor = _finite_number(result["fixed_floor_nat_per_draw"], label)
    threshold = _finite_number(result["threshold_nat_per_draw"], label)
    expected_values = {
        "gap": abs(test_nll - validation_nll),
        "pooled": math.hypot(validation_mcse, test_mcse),
        "floor": gmm.GENERALIZATION_NAT_PER_DIMENSION * dimension,
    }
    expected_values["threshold"] = max(
        expected_values["floor"],
        gmm.GENERALIZATION_MCSE_MULTIPLIER * expected_values["pooled"],
    )
    if (
        gap != expected_values["gap"]
        or pooled != expected_values["pooled"]
        or floor != expected_values["floor"]
        or threshold != expected_values["threshold"]
    ):
        raise ValueError(f"{label} generalization arithmetic is inconsistent")
    passed = gap <= threshold
    if result["pass"] is not passed:
        raise ValueError(f"{label} generalization pass is inconsistent")
    return passed


def _require_replayed_generalization(
    observed: Mapping[str, Any],
    replayed: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Compare regenerated generalization evidence with narrow float tolerance."""
    if set(observed) != set(replayed):
        raise ValueError(f"{label} replayed generalization has an unexpected schema")
    for field, expected in replayed.items():
        value = observed[field]
        if isinstance(expected, float):
            actual = _finite_number(value, f"{label} {field}")
            absolute_floor = (
                _GENERALIZATION_NLL_REPLAY_ABS_TOL
                if field in _GENERALIZATION_NLL_FIELDS
                else _GENERALIZATION_MCSE_REPLAY_ABS_TOL
            )
            tolerance = _replay_tolerance(
                actual,
                expected,
                absolute_floor=absolute_floor,
            )
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError(f"{label} {field} does not match the simulator replay")
        elif value != expected or type(value) is not type(expected):
            raise ValueError(f"{label} {field} does not match the simulator replay")
    observed_gap = _finite_number(
        observed["absolute_nll_gap_nat_per_draw"],
        f"{label} observed absolute NLL gap",
    )
    observed_threshold = _finite_number(
        observed["threshold_nat_per_draw"],
        f"{label} observed threshold",
    )
    replayed_gap = _finite_number(
        replayed["absolute_nll_gap_nat_per_draw"],
        f"{label} replayed absolute NLL gap",
    )
    replayed_threshold = _finite_number(
        replayed["threshold_nat_per_draw"],
        f"{label} replayed threshold",
    )
    gate_margin = _GENERALIZATION_NLL_REPLAY_ABS_TOL
    if (
        abs(observed_gap - observed_threshold) <= gate_margin
        or abs(replayed_gap - replayed_threshold) <= gate_margin
    ):
        raise ValueError(f"{label} is too close to its pass threshold for portable replay")


def _replayed_generalization(
    artifact: ConditionalResidualImageMDN,
    domain_draws: Mapping[str, FloatArray],
) -> dict[str, float | int | bool]:
    """Recompute validation/test NLL evidence from a zero-input artifact."""
    if artifact.region_count != 1:
        raise ValueError("development GMM artifact must have exactly one region")
    log_weights, means, factors = artifact._components(np.ones(artifact.region_count, dtype=np.float64))
    covariances = factors @ np.transpose(factors, (0, 2, 1))
    fit = gmm.GaussianMixtureFit(
        weights=np.exp(log_weights),
        means=means,
        covariances=covariances,
        initialization=0,
        iterations=0,
        training_mean_log_likelihood=0.0,
        validation_mean_log_likelihood=0.0,
        validation_nll=0.0,
        convergence_streak=0,
        objective_history=(),
    )
    return gmm._simulator_test_generalization(
        domain_draws[gmm.VALIDATION_DOMAIN],
        domain_draws[gmm.TEST_DOMAIN],
        fit,
    )


def _reverify_scientific_gates(
    *,
    artifact: ConditionalResidualImageMDN,
    exact: _ExactCase,
    nominated: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and authenticate one fitted artifact's scientific gates."""
    replay = gmm._evaluate_artifact(
        artifact=artifact,
        observation=exact.observation,
        masses=exact.masses,
        log_prior=exact.log_prior,
        exact_log_likelihood=exact.exact_log_likelihood,
        exact_summary=exact.exact_summary,
        gradient_states=exact.gradient_states,
        validation_state_mask=exact.validation_state_mask,
    )
    for field in (
        "metrics",
        "checks",
        "posterior_summary",
        "posterior_errors_by_coordinate",
        "gradient_audits",
        "diagnostics",
    ):
        _require_replayed_science(
            nominated.get(field),
            replay[field],
            f"{exact.case_id} replayed {field}",
        )
    checks = _require_mapping(replay["checks"], f"{exact.case_id} replayed checks")
    if set(checks) != set(_MODEL_GATE_NAMES):
        raise ValueError(f"{exact.case_id} replayed model-gate schema changed")
    recorded_metrics = _require_mapping(
        nominated.get("metrics"),
        f"{exact.case_id} recorded metrics",
    )
    replayed_metrics = _require_mapping(
        replay["metrics"],
        f"{exact.case_id} replayed metrics",
    )
    for name in _MODEL_GATE_NAMES:
        recorded_metric = _finite_number(
            recorded_metrics[name],
            f"{exact.case_id} recorded metric {name}",
        )
        replayed_metric = _finite_number(
            replayed_metrics[name],
            f"{exact.case_id} replayed metric {name}",
        )
        gate_margin = _replay_tolerance(
            recorded_metric,
            replayed_metric,
            absolute_floor=_SCIENTIFIC_REPLAY_ABS_TOL,
        )
        if (
            abs(recorded_metric - c1.THRESHOLDS[name]) <= gate_margin
            or abs(replayed_metric - c1.THRESHOLDS[name]) <= gate_margin
        ):
            raise ValueError(
                f"{exact.case_id} metric {name} is too close to its threshold for portable replay"
            )
    return replay


def _validate_evaluation(
    evaluation: Mapping[str, Any],
    *,
    exact: _ExactCase,
    case_record: Mapping[str, Any],
    domain_bank_cache: dict[tuple[str, int, int], DomainBank],
    sample_count: int,
    base_seed: int,
    expected_source_revision: str,
    expected_driver_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Authenticate one fitted evaluation and recompute all individual gates."""
    case_id = exact.case_id
    context_sha256 = exact.context.artifact_sha256
    _validate_exact_case_record(case_record, exact, label=label)
    if evaluation.get("sample_count") != sample_count or evaluation.get("base_seed") != base_seed:
        raise ValueError(f"{label} sample count or base seed changed")
    metrics = _require_mapping(evaluation.get("metrics"), f"{label} metrics")
    checks = _require_mapping(evaluation.get("checks"), f"{label} checks")
    if set(metrics) != set(_MODEL_GATE_NAMES) or set(checks) != set(_MODEL_GATE_NAMES):
        raise ValueError(f"{label} does not contain exactly the frozen model gates")
    recomputed_checks: dict[str, bool] = {}
    for name in _MODEL_GATE_NAMES:
        metric = _finite_number(metrics[name], f"{label} metric {name}")
        recomputed_checks[name] = metric <= c1.THRESHOLDS[name]
        if checks[name] is not recomputed_checks[name]:
            raise ValueError(f"{label} check {name} disagrees with its threshold")
    model_pass = all(recomputed_checks.values())
    if evaluation.get("scientific_model_gates_pass") is not model_pass:
        raise ValueError(f"{label} scientific model-gate pass is inconsistent")

    training = _require_mapping(evaluation.get("training"), f"{label} training")
    if (
        training.get("training_sample_count") != sample_count
        or training.get("base_seed") != base_seed
        or training.get("minimum_valid_initializations") != gmm.MINIMUM_VALID_INITIALIZATIONS
    ):
        raise ValueError(f"{label} training identity changed")
    attempts = _require_list(training.get("initialization_attempts"), f"{label} attempts")
    if len(attempts) != gmm.INITIALIZATION_COUNT:
        raise ValueError(f"{label} has the wrong initialization count")
    valid_count = sum(
        isinstance(attempt, dict) and attempt.get("status") == "converged" for attempt in attempts
    )
    valid_pass = valid_count >= gmm.MINIMUM_VALID_INITIALIZATIONS
    if (
        training.get("valid_initialization_count") != valid_count
        or training.get("valid_initialization_pass") is not valid_pass
    ):
        raise ValueError(f"{label} valid-initialization gate is inconsistent")
    generalization_pass = _validate_generalization(
        training.get("simulator_test_generalization"),
        label,
    )
    generalization = _require_mapping(
        training["simulator_test_generalization"],
        f"{label} generalization",
    )
    if (
        training.get("validation_nll") != generalization["validation_nll_nat_per_draw"]
        or training.get("test_nll") != generalization["simulator_test_nll_nat_per_draw"]
    ):
        raise ValueError(f"{label} recorded NLLs disagree with generalization evidence")
    fit_pass = valid_pass and generalization_pass
    if (
        training.get("fit_development_pass") is not fit_pass
        or evaluation.get("fit_development_pass") is not fit_pass
    ):
        raise ValueError(f"{label} fit-development gate is inconsistent")

    envelope = _require_mapping(
        training.get("fitted_bundle_envelope"),
        f"{label} fitted envelope",
    )
    envelope_sha256 = envelope.get("sha256")
    if not isinstance(envelope_sha256, str) or _SHA256_RE.fullmatch(envelope_sha256) is None:
        raise ValueError(f"{label} fitted-envelope digest is malformed")
    artifact = gmm.validate_fitted_bundle_envelope(
        envelope,
        expected_sha256=envelope_sha256,
        expected_source_git_revision=expected_source_revision,
        expected_driver_sha256=expected_driver_sha256,
    )
    envelope_payload = _require_mapping(envelope.get("payload"), f"{label} envelope payload")
    envelope_case = _require_mapping(envelope_payload.get("case"), f"{label} envelope case")
    envelope_training = _require_mapping(
        envelope_payload.get("training"),
        f"{label} envelope training",
    )
    envelope_artifact = _require_mapping(
        envelope_payload.get("artifact"),
        f"{label} envelope artifact",
    )
    if envelope_case != {"case_id": case_id, "context_sha256": context_sha256}:
        raise ValueError(f"{label} fitted envelope is bound to the wrong case")
    for field in (
        "training_sample_count",
        "training_prefix_sha256",
        "initialization_attempts",
        "selected_initialization",
        "valid_initialization_count",
        "minimum_valid_initializations",
    ):
        _require_equal(
            envelope_training.get(field),
            training.get(field),
            f"{label} envelope training {field}",
        )
    _require_equal(
        envelope_payload.get("domains"),
        training.get("domain_artifacts"),
        f"{label} envelope domains",
    )
    _require_equal(
        envelope_payload.get("generalization"),
        generalization,
        f"{label} envelope generalization",
    )
    if (
        envelope_artifact.get("sha256") != training.get("artifact_sha256")
        or envelope_artifact.get("payload") != training.get("artifact_payload")
        or artifact.artifact_sha256 != training.get("artifact_sha256")
    ):
        raise ValueError(f"{label} fitted artifact identity is inconsistent")
    if artifact.context.artifact_sha256 != exact.context.artifact_sha256:
        raise ValueError(f"{label} fitted artifact context is not the exact case context")

    domains = _require_mapping(training.get("domain_artifacts"), f"{label} domains")
    domain_draws, replayed_domains = _domain_bank(
        exact,
        sample_count=sample_count,
        base_seed=base_seed,
        cache=domain_bank_cache,
    )
    _require_equal(domains, replayed_domains, f"{label} simulator domains")
    replayed_prefix_sha256 = gmm._array_sha256(domain_draws[gmm.TRAINING_DOMAIN][:sample_count])
    if training.get("training_prefix_sha256") != replayed_prefix_sha256:
        raise ValueError(f"{label} training prefix does not match the simulator replay")
    replayed_generalization = _replayed_generalization(artifact, domain_draws)
    _require_replayed_generalization(
        generalization,
        replayed_generalization,
        label=f"{label} generalization",
    )
    for field, expected in (
        ("validation_nll", replayed_generalization["validation_nll_nat_per_draw"]),
        ("test_nll", replayed_generalization["simulator_test_nll_nat_per_draw"]),
    ):
        observed = _finite_number(training.get(field), f"{label} {field}")
        if not math.isclose(
            observed,
            cast(float, expected),
            rel_tol=0.0,
            abs_tol=_replay_tolerance(
                observed,
                cast(float, expected),
                absolute_floor=_GENERALIZATION_NLL_REPLAY_ABS_TOL,
            ),
        ):
            raise ValueError(f"{label} {field} does not match the simulator replay")
    replay = _reverify_scientific_gates(
        artifact=artifact,
        exact=exact,
        nominated=evaluation,
    )
    replay_model_pass = bool(replay["scientific_pass"])
    if (
        evaluation.get("scientific_model_gates_pass") is not replay_model_pass
        or model_pass is not replay_model_pass
    ):
        raise ValueError(f"{label} scientific model-gate pass does not match the exact replay")
    replay_fit_pass = bool(valid_pass and replayed_generalization["pass"])
    if fit_pass is not replay_fit_pass:
        raise ValueError(f"{label} fit-development pass does not match the simulator replay")
    passed = replay_model_pass and replay_fit_pass
    if evaluation.get("scientific_pass") is not passed:
        raise ValueError(f"{label} scientific pass does not match the exact replay")
    replay_posterior = _require_mapping(replay["posterior_summary"], f"{label} replay posterior")
    log_evidence = _finite_number(
        replay_posterior.get("log_evidence"),
        f"{label} replay log evidence",
    )
    return {
        "pass": passed,
        "log_evidence": log_evidence,
        "fitted_bundle_sha256": envelope_sha256,
        "artifact_sha256": training["artifact_sha256"],
    }


def _validate_development_shard(
    report: Mapping[str, Any],
    *,
    raw_sha256: str,
    case_id: str,
    exact: _ExactCase,
    domain_bank_cache: dict[tuple[str, int, int], DomainBank],
    sample_count: int,
    expected_source_revision: str,
    expected_driver_sha256: str,
) -> dict[str, Any]:
    """Validate one of the 24 immutable development-size shards."""
    _validate_report_identity(
        report,
        case_id=case_id,
        expected_source_revision=expected_source_revision,
        expected_driver_sha256=expected_driver_sha256,
    )
    if (
        report.get("execution_mode") != "development_size_shard"
        or report.get("executed_development_sample_count") != sample_count
    ):
        raise ValueError(f"{case_id} S={sample_count} is not the expected development shard")
    cases = _require_list(report.get("cases"), f"{case_id} cases")
    if len(cases) != 1:
        raise ValueError(f"{case_id} S={sample_count} must contain exactly one case")
    case = _require_mapping(cases[0], f"{case_id} case")
    if (
        case.get("case_id") != case_id
        or case.get("profile") != "development"
        or case.get("executed_development_sample_count") != sample_count
        or case.get("executed_confirmation_seed") is not None
        or case.get("confirmation_evaluations") != []
    ):
        raise ValueError(f"{case_id} S={sample_count} case identity changed")
    input_sha256 = case.get("input_sha256")
    context_sha256 = case.get("context_sha256")
    if (
        not isinstance(input_sha256, str)
        or _SHA256_RE.fullmatch(input_sha256) is None
        or not isinstance(context_sha256, str)
        or _SHA256_RE.fullmatch(context_sha256) is None
    ):
        raise ValueError(f"{case_id} has malformed input or context identity")
    evaluations = _require_list(
        case.get("development_evaluations"),
        f"{case_id} development evaluations",
    )
    if len(evaluations) != 1:
        raise ValueError(f"{case_id} S={sample_count} must contain exactly one evaluation")
    evaluation = _require_mapping(evaluations[0], f"{case_id} S={sample_count} evaluation")
    audit = _validate_evaluation(
        evaluation,
        exact=exact,
        case_record=case,
        domain_bank_cache=domain_bank_cache,
        sample_count=sample_count,
        base_seed=gmm.DEVELOPMENT_SELECTION_SEED,
        expected_source_revision=expected_source_revision,
        expected_driver_sha256=expected_driver_sha256,
        label=f"{case_id} development S={sample_count}",
    )
    nested = _require_mapping(
        case.get("development_nested_training_bank"),
        f"{case_id} nested bank",
    )
    training = _require_mapping(evaluation["training"], f"{case_id} training")
    expected_nested = {
        "largest_sample_count": max(gmm.DEVELOPMENT_SAMPLE_COUNTS),
        "artifact_sha256": training["domain_artifacts"][gmm.TRAINING_DOMAIN]["artifact_sha256"],
        "full_draws_sha256": training["domain_artifacts"][gmm.TRAINING_DOMAIN]["draws_sha256"],
        "prefixes": {str(sample_count): training["training_prefix_sha256"]},
    }
    _require_equal(nested, expected_nested, f"{case_id} nested training bank")
    return {
        "raw_sha256": raw_sha256,
        "input_sha256": input_sha256,
        "context_sha256": context_sha256,
        "exact_posterior_summary": case.get("exact_posterior_summary"),
        "quadrature": case.get("quadrature"),
        "nested_artifact_sha256": nested["artifact_sha256"],
        "nested_draws_sha256": nested["full_draws_sha256"],
        "domain_artifacts": training["domain_artifacts"],
        "evaluation": evaluation,
        **audit,
    }


def _load_development_shards(
    directory: Path,
    *,
    expected_source_revision: str,
    expected_driver_sha256: str,
    exact_cases: dict[str, _ExactCase] | None = None,
    domain_bank_cache: dict[tuple[str, int, int], DomainBank] | None = None,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Load and authenticate exactly the frozen 24 development shards."""
    phase_exact_cases = _exact_cases() if exact_cases is None else exact_cases
    phase_domain_banks = {} if domain_bank_cache is None else domain_bank_cache
    expected_names = {
        _development_filename(case_id, count)
        for case_id in _case_ids()
        for count in gmm.DEVELOPMENT_SAMPLE_COUNTS
    }
    observed_names = _regular_file_names(directory, "development shard directory")
    if observed_names != expected_names:
        raise ValueError(
            "development shard directory must contain exactly the frozen 24 files; "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for case_id in _case_ids():
        by_size: dict[int, dict[str, Any]] = {}
        for count in gmm.DEVELOPMENT_SAMPLE_COUNTS:
            payload, raw_sha256 = _read_canonical_json(directory / _development_filename(case_id, count))
            by_size[count] = _validate_development_shard(
                payload,
                raw_sha256=raw_sha256,
                case_id=case_id,
                exact=phase_exact_cases[case_id],
                domain_bank_cache=phase_domain_banks,
                sample_count=count,
                expected_source_revision=expected_source_revision,
                expected_driver_sha256=expected_driver_sha256,
            )
        reference = by_size[gmm.DEVELOPMENT_SAMPLE_COUNTS[0]]
        for shard in by_size.values():
            for field in (
                "input_sha256",
                "context_sha256",
                "exact_posterior_summary",
                "quadrature",
                "nested_artifact_sha256",
                "nested_draws_sha256",
                "domain_artifacts",
            ):
                _require_equal(shard[field], reference[field], f"{case_id} shared {field}")
        result[case_id] = by_size
    return result


def _write_atomic_envelope(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish one canonical self-digested envelope without overwrite."""
    envelope = {"payload": dict(payload), "sha256": c1._sha256_json(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(f"{_canonical_json(envelope)}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return envelope


def merge_development(
    *,
    source_directory: Path,
    development_directory: Path,
    output_lock: Path,
    expected_source_revision: str,
) -> dict[str, Any]:
    """Authenticate 24 development shards and publish the common lock."""
    if _FULL_SHA_RE.fullmatch(expected_source_revision) is None:
        raise ValueError("expected source revision must be a full lower-case 40-hex Git SHA")
    if output_lock.exists() or output_lock.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {output_lock}")
    gmm._validate_development_protocol()
    _validate_live_source(source_directory, expected_source_revision)
    driver_sha256 = _driver_source_sha256()
    shards = _load_development_shards(
        development_directory,
        expected_source_revision=expected_source_revision,
        expected_driver_sha256=driver_sha256,
    )
    development_pattern = [
        {
            "sample_count": count,
            "pass": all(shards[case_id][count]["pass"] for case_id in _case_ids()),
        }
        for count in gmm.DEVELOPMENT_SAMPLE_COUNTS
    ]
    locked = c1._stable_lock_sample_count(
        gmm.DEVELOPMENT_SAMPLE_COUNTS,
        [entry["pass"] for entry in development_pattern],
        minimum_suffix_length=2,
    )
    if locked is None:
        raise ValueError("development shards do not establish a common two-size passing suffix")
    cases: dict[str, Any] = {}
    for case_id in _case_ids():
        nominated = shards[case_id][locked]
        cases[case_id] = {
            "development_input_raw_sha256": nominated["raw_sha256"],
            "input_sha256": nominated["input_sha256"],
            "context_sha256": nominated["context_sha256"],
            "nominated_fitted_bundle_sha256": nominated["fitted_bundle_sha256"],
            "nominated_artifact_sha256": nominated["artifact_sha256"],
        }
    lock_payload: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "certification_protocol": CERTIFICATION_PROTOCOL,
        "certification_protocol_sha256": _certification_protocol_sha256(),
        "source_git_revision": expected_source_revision,
        "scientific_driver_sha256": driver_sha256,
        "frozen_development_protocol_sha256": gmm.DEVELOPMENT_PROTOCOL_SHA256,
        "a1_definitions_sha256": c1.A1_DEFINITIONS_SHA256,
        "matrix_catalogue": gmm.matrix_catalogue(),
        "sample_counts": list(gmm.DEVELOPMENT_SAMPLE_COUNTS),
        "development_selection_seed": gmm.DEVELOPMENT_SELECTION_SEED,
        "confirmation_seeds": list(gmm.CONFIRMATION_SEEDS),
        "minimum_passing_suffix_length": 2,
        "development_pass_pattern": development_pattern,
        "locked_sample_count": locked,
        "cases": cases,
    }
    return _write_atomic_envelope(output_lock, lock_payload)


def _read_lock(
    path: Path,
    *,
    expected_source_revision: str,
    expected_driver_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    """Read and fully authenticate one common-lock envelope."""
    envelope, raw_sha256 = _read_canonical_json(path)
    if set(envelope) != {"payload", "sha256"}:
        raise ValueError("common lock has an unexpected envelope schema")
    payload = _require_mapping(envelope.get("payload"), "common lock payload")
    internal_sha256 = c1._sha256_json(payload)
    if envelope.get("sha256") != internal_sha256:
        raise ValueError("common lock internal digest does not match its payload")
    expected_fields = {
        "schema",
        "certification_protocol",
        "certification_protocol_sha256",
        "source_git_revision",
        "scientific_driver_sha256",
        "frozen_development_protocol_sha256",
        "a1_definitions_sha256",
        "matrix_catalogue",
        "sample_counts",
        "development_selection_seed",
        "confirmation_seeds",
        "minimum_passing_suffix_length",
        "development_pass_pattern",
        "locked_sample_count",
        "cases",
    }
    if set(payload) != expected_fields:
        raise ValueError("common lock payload has an unexpected schema")
    expected = {
        "schema": LOCK_SCHEMA,
        "certification_protocol": CERTIFICATION_PROTOCOL,
        "certification_protocol_sha256": _certification_protocol_sha256(),
        "source_git_revision": expected_source_revision,
        "scientific_driver_sha256": expected_driver_sha256,
        "frozen_development_protocol_sha256": gmm.DEVELOPMENT_PROTOCOL_SHA256,
        "a1_definitions_sha256": c1.A1_DEFINITIONS_SHA256,
        "matrix_catalogue": gmm.matrix_catalogue(),
        "sample_counts": list(gmm.DEVELOPMENT_SAMPLE_COUNTS),
        "development_selection_seed": gmm.DEVELOPMENT_SELECTION_SEED,
        "confirmation_seeds": list(gmm.CONFIRMATION_SEEDS),
        "minimum_passing_suffix_length": 2,
    }
    for field, value in expected.items():
        _require_equal(payload.get(field), value, f"common lock {field}")
    locked = payload.get("locked_sample_count")
    if isinstance(locked, bool) or locked not in gmm.DEVELOPMENT_SAMPLE_COUNTS:
        raise ValueError("common lock has an invalid sample count")
    cases = _require_mapping(payload.get("cases"), "common lock cases")
    if set(cases) != set(_case_ids()):
        raise ValueError("common lock does not contain exactly the six frozen cases")
    return payload, internal_sha256, raw_sha256


def _validate_confirmation_shard(
    report: Mapping[str, Any],
    *,
    case_id: str,
    exact: _ExactCase,
    domain_bank_cache: dict[tuple[str, int, int], DomainBank],
    seed: int,
    lock_payload: Mapping[str, Any],
    lock_internal_sha256: str,
    lock_raw_sha256: str,
    expected_source_revision: str,
    expected_driver_sha256: str,
) -> dict[str, Any]:
    """Validate one of the 18 immutable confirmation-seed shards."""
    _validate_report_identity(
        report,
        case_id=case_id,
        expected_source_revision=expected_source_revision,
        expected_driver_sha256=expected_driver_sha256,
    )
    locked = cast(int, lock_payload["locked_sample_count"])
    if (
        report.get("execution_mode") != "confirmation_seed_shard"
        or report.get("executed_confirmation_seed") != seed
        or report.get("confirmation_lock_internal_sha256") != lock_internal_sha256
        or report.get("confirmation_lock_raw_sha256") != lock_raw_sha256
        or report.get("confirmation_locked_sample_count") != locked
    ):
        raise ValueError(f"{case_id} seed={seed} is not bound to the common lock")
    cases = _require_list(report.get("cases"), f"{case_id} confirmation cases")
    if len(cases) != 1:
        raise ValueError(f"{case_id} seed={seed} must contain exactly one case")
    case = _require_mapping(cases[0], f"{case_id} confirmation case")
    lock_case = _require_mapping(lock_payload["cases"][case_id], f"{case_id} lock case")
    if (
        case.get("case_id") != case_id
        or case.get("executed_confirmation_seed") != seed
        or case.get("executed_development_sample_count") is not None
        or case.get("development_evaluations") != []
        or case.get("confirmation_sample_count") != locked
        or case.get("input_sha256") != lock_case["input_sha256"]
        or case.get("context_sha256") != lock_case["context_sha256"]
    ):
        raise ValueError(f"{case_id} seed={seed} case identity changed")
    evaluations = _require_list(
        case.get("confirmation_evaluations"),
        f"{case_id} confirmation evaluations",
    )
    if len(evaluations) != 1:
        raise ValueError(f"{case_id} seed={seed} must contain exactly one evaluation")
    evaluation = _require_mapping(evaluations[0], f"{case_id} seed={seed} evaluation")
    audit = _validate_evaluation(
        evaluation,
        exact=exact,
        case_record=case,
        domain_bank_cache=domain_bank_cache,
        sample_count=locked,
        base_seed=seed,
        expected_source_revision=expected_source_revision,
        expected_driver_sha256=expected_driver_sha256,
        label=f"{case_id} confirmation seed={seed}",
    )
    return {"evaluation": evaluation, **audit}


def certify_confirmation(
    *,
    source_directory: Path,
    development_directory: Path,
    confirmation_directory: Path,
    common_lock: Path,
    output_certificate: Path,
    expected_source_revision: str,
    expected_lock_raw_sha256: str,
) -> dict[str, Any]:
    """Authenticate all shards and publish the development certificate."""
    if _FULL_SHA_RE.fullmatch(expected_source_revision) is None:
        raise ValueError("expected source revision must be a full lower-case 40-hex Git SHA")
    if _SHA256_RE.fullmatch(expected_lock_raw_sha256) is None:
        raise ValueError("expected lock raw SHA-256 must be a lower-case 64-hex digest")
    if output_certificate.exists() or output_certificate.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {output_certificate}")
    gmm._validate_development_protocol()
    _validate_live_source(source_directory, expected_source_revision)
    driver_sha256 = _driver_source_sha256()
    exact_cases = _exact_cases()
    domain_bank_cache: dict[tuple[str, int, int], DomainBank] = {}
    lock_payload, lock_internal_sha256, lock_raw_sha256 = _read_lock(
        common_lock,
        expected_source_revision=expected_source_revision,
        expected_driver_sha256=driver_sha256,
    )
    if lock_raw_sha256 != expected_lock_raw_sha256:
        raise ValueError("common lock raw SHA-256 does not match the expected binding")
    development = _load_development_shards(
        development_directory,
        expected_source_revision=expected_source_revision,
        expected_driver_sha256=driver_sha256,
        exact_cases=exact_cases,
        domain_bank_cache=domain_bank_cache,
    )
    locked = cast(int, lock_payload["locked_sample_count"])
    for case_id in _case_ids():
        lock_case = _require_mapping(lock_payload["cases"][case_id], f"{case_id} lock case")
        nominated = development[case_id][locked]
        expected_lock_case = {
            "development_input_raw_sha256": nominated["raw_sha256"],
            "input_sha256": nominated["input_sha256"],
            "context_sha256": nominated["context_sha256"],
            "nominated_fitted_bundle_sha256": nominated["fitted_bundle_sha256"],
            "nominated_artifact_sha256": nominated["artifact_sha256"],
        }
        _require_equal(lock_case, expected_lock_case, f"{case_id} lock nomination")

    expected_names = {
        _confirmation_filename(case_id, seed) for case_id in _case_ids() for seed in gmm.CONFIRMATION_SEEDS
    }
    observed_names = _regular_file_names(confirmation_directory, "confirmation shard directory")
    if observed_names != expected_names:
        raise ValueError(
            "confirmation shard directory must contain exactly the frozen 18 files; "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )
    case_summaries: list[dict[str, Any]] = []
    all_cases_pass = True
    for case_id in _case_ids():
        confirmation_audits: list[dict[str, Any]] = []
        for seed in gmm.CONFIRMATION_SEEDS:
            report, _ = _read_canonical_json(confirmation_directory / _confirmation_filename(case_id, seed))
            confirmation_audits.append(
                _validate_confirmation_shard(
                    report,
                    case_id=case_id,
                    exact=exact_cases[case_id],
                    domain_bank_cache=domain_bank_cache,
                    seed=seed,
                    lock_payload=lock_payload,
                    lock_internal_sha256=lock_internal_sha256,
                    lock_raw_sha256=lock_raw_sha256,
                    expected_source_revision=expected_source_revision,
                    expected_driver_sha256=driver_sha256,
                )
            )
        nominated = development[case_id][locked]
        evidence = [
            nominated["log_evidence"],
            *[audit["log_evidence"] for audit in confirmation_audits],
        ]
        evidence_range, evidence_pass = _four_bank_evidence_range_gate(
            evidence,
            label=f"{case_id} four-bank evidence range",
        )
        confirmation_pass = all(audit["pass"] for audit in confirmation_audits)
        case_pass = bool(nominated["pass"] and confirmation_pass and evidence_pass)
        all_cases_pass = all_cases_pass and case_pass
        case_summaries.append(
            {
                "case_id": case_id,
                "input_sha256": nominated["input_sha256"],
                "context_sha256": nominated["context_sha256"],
                "locked_sample_count": locked,
                "nominated_development_raw_sha256": nominated["raw_sha256"],
                "nominated_development_evaluation": nominated["evaluation"],
                "nominated_fitted_bundle_sha256": nominated["fitted_bundle_sha256"],
                "nominated_artifact_sha256": nominated["artifact_sha256"],
                "confirmation_evaluations": [audit["evaluation"] for audit in confirmation_audits],
                "confirmation_individual_passes": [
                    {"base_seed": seed, "pass": audit["pass"]}
                    for seed, audit in zip(
                        gmm.CONFIRMATION_SEEDS,
                        confirmation_audits,
                        strict=True,
                    )
                ],
                "four_bank_log_evidence_range_nat": evidence_range,
                "four_bank_log_evidence_range_pass": evidence_pass,
                "confirmation_pass": confirmation_pass,
                "development_pass": case_pass,
            }
        )
    payload: dict[str, Any] = {
        "schema": CERTIFICATE_SCHEMA,
        "certification_protocol": CERTIFICATION_PROTOCOL,
        "certification_protocol_sha256": _certification_protocol_sha256(),
        "certifier_source_sha256": _certifier_source_sha256(),
        "source_git_revision": expected_source_revision,
        "scientific_driver_sha256": driver_sha256,
        "frozen_development_protocol_sha256": gmm.DEVELOPMENT_PROTOCOL_SHA256,
        "a1_definitions_sha256": c1.A1_DEFINITIONS_SHA256,
        "matrix_catalogue": gmm.matrix_catalogue(),
        "common_lock_raw_sha256": lock_raw_sha256,
        "common_lock_sha256": lock_internal_sha256,
        "locked_sample_count": locked,
        "confirmation_seeds": list(gmm.CONFIRMATION_SEEDS),
        "execution_certified": True,
        "decision": "pass" if all_cases_pass else "hard_stop",
        "development_pass": all_cases_pass,
        "eligible_for_protected_holdout": all_cases_pass,
        "protected_holdout_pass": None,
        "scientific_pass": False,
        "scientific_pass_available": False,
        "scientific_pass_reason": ("the separately sealed protected holdout has not been evaluated"),
        "structural_inference_licensed": False,
        "held_out_information_read": False,
        "cases": case_summaries,
    }
    return _write_atomic_envelope(output_certificate, payload)


def _parser() -> argparse.ArgumentParser:
    """Build the strict two-phase certification CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source-dir", type=Path, required=True)
    common.add_argument("--development-dir", type=Path, required=True)
    common.add_argument("--expected-source-revision", required=True)

    merge = subparsers.add_parser("merge-development", parents=[common])
    merge.add_argument("--output-lock", type=Path, required=True)

    confirmation = subparsers.add_parser("certify-confirmation", parents=[common])
    confirmation.add_argument("--confirmation-dir", type=Path, required=True)
    confirmation.add_argument("--common-lock", type=Path, required=True)
    confirmation.add_argument("--expected-lock-raw-sha256", required=True)
    confirmation.add_argument("--output-certificate", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one certification phase and publish its immutable output."""
    args = _parser().parse_args(argv)
    if args.command == "merge-development":
        merge_development(
            source_directory=args.source_dir,
            development_directory=args.development_dir,
            output_lock=args.output_lock,
            expected_source_revision=args.expected_source_revision,
        )
    else:
        certify_confirmation(
            source_directory=args.source_dir,
            development_directory=args.development_dir,
            confirmation_directory=args.confirmation_dir,
            common_lock=args.common_lock,
            output_certificate=args.output_certificate,
            expected_source_revision=args.expected_source_revision,
            expected_lock_raw_sha256=args.expected_lock_raw_sha256,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
