#!/usr/bin/env python3
"""Certify the sealed residual-image GMM protected density holdout.

This executable is intentionally separate from model development.  It accepts
one completed six-case development certificate, the immutable directory of
development shards referenced by that certificate, and one sealed protected
catalogue.  The frozen runtime, source, eligible certificate, and all six
nominated artifacts are authenticated and scientifically replayed before the
catalogue is touched.  Its raw bytes are then authenticated against the
SHA-256 pin in the scientific driver before JSON parsing or protected
numerical work.

For each source-pinned root case, the certifier promotes only the development
seed-731 fitted bundle at the common locked training size.  It authenticates
that normalized portable artifact, reconstructs the exact context, replays the
unchanged likelihood, gradient, evidence, and posterior gates, and evaluates
one new protected scrambled-Sobol residual bank.  The protected Sobol seed is
the little-endian first 64 bits of SHA-256 over canonical JSON containing the
catalogue schema, split domain, case ID, and the secret master seed hex.  This
rule is identified by :data:`PROTECTED_SEED_DERIVATION`.

No fitting, architecture selection, threshold selection, or retry path exists
here.  The output is canonical JSON published without overwrite.  Even a
scientific pass licenses only the fixed root representations tested by the
development protocol; structural inference is always prohibited.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from examples.rjmcmc import conditional_residual_image_gmm_certify as development_certifier
from examples.rjmcmc import conditional_residual_image_gmm_tiny_screen as gmm
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mdn import (
    ConditionalResidualImageMDN,
    ResidualImageContext,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)

FloatArray = NDArray[np.float64]

SCHEMA = "rjmcmc-conditional-residual-image-gmm-protected-certification-v1"
CERTIFICATION_PROTOCOL = "conditional-residual-image-gmm-protected-density-holdout-certifier-v1"
CATALOGUE_SCHEMA = "conditional-residual-image-protected-density-holdout-catalogue-v1"
PROTECTED_SPLIT_DOMAIN = "protected-density-holdout"
PROTECTED_SEED_DERIVATION = (
    "sha256(canonical-json({schema,split_domain,case_id,master_seed_hex}))/little-endian-first-64"
)

_SHA256_HEX_LENGTH = 64
_GIT_SHA_HEX_LENGTH = 40
_EVALUATION_GATE_KEYS = tuple(name for name in c1.THRESHOLDS if name != "between_bank_log_evidence_range_nat")


@dataclass(frozen=True)
class _ExactCase:
    """Exact source-pinned numerical objects needed for one protected replay."""

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


@dataclass(frozen=True)
class _PreparedCase:
    """Authenticated development promotion prepared before opening the seal."""

    case_id: str
    locked_sample_count: int
    shard_raw_sha256: str
    fitted_bundle_sha256: str
    artifact_sha256: str
    artifact: ConditionalResidualImageMDN
    exact: _ExactCase
    generalization: dict[str, Any]
    scientific_gate_pass: bool


def _canonical_json(payload: object) -> str:
    """Return the scientific driver's strict canonical JSON representation."""
    return c1._canonical_json(payload)


def _sha256_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest for raw bytes."""
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    """Return whether a value is a canonical lowercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    """Return a string-keyed JSON mapping or reject the input."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _require_list(value: object, label: str) -> list[Any]:
    """Return a JSON list or reject the input."""
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return cast(list[Any], value)


def _json_object_pairs(label: str) -> Any:
    """Return a duplicate-rejecting object-pairs hook for one labelled input."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    return object_pairs


def _parse_canonical_json(raw: bytes, *, label: str) -> dict[str, Any]:
    """Parse one newline-terminated canonical ASCII JSON object."""
    try:
        text = raw.decode("ascii")
        payload = json.loads(
            text,
            object_pairs_hook=_json_object_pairs(label),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite JSON value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict ASCII JSON") from error
    result = _require_mapping(payload, label)
    if text != f"{_canonical_json(result)}\n":
        raise ValueError(f"{label} is not newline-terminated canonical JSON")
    return result


def _read_protected_catalogue(path: Path) -> tuple[dict[str, Any], str]:
    """Authenticate raw protected bytes before parsing and validate the schema.

    Args:
        path: Sealed protected catalogue path.

    Returns:
        The validated catalogue and its raw SHA-256 digest.

    Raises:
        ValueError: If the file, raw digest, canonical encoding, or frozen
            catalogue contract is invalid.
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError("protected catalogue must be one regular non-symlink file")
    raw = path.read_bytes()
    raw_sha256 = _sha256_bytes(raw)
    # This comparison deliberately precedes decoding, JSON parsing, or science.
    if raw_sha256 != gmm.PROTECTED_HOLDOUT_CATALOGUE_SHA256:
        raise ValueError("protected catalogue raw SHA-256 does not match the source pin")
    catalogue = _parse_canonical_json(raw, label="protected catalogue")
    expected_fields = {
        "schema",
        "case_ids",
        "protocol",
        "construction_method",
        "sample_count_per_case",
        "split_domain",
        "master_seed_hex",
    }
    if set(catalogue) != expected_fields:
        raise ValueError("protected catalogue has an unexpected schema")
    expected_case_ids = ["__".join(case) for case in gmm.DEVELOPMENT_MATRIX]
    master_seed_hex = catalogue["master_seed_hex"]
    if (
        catalogue["schema"] != CATALOGUE_SCHEMA
        or catalogue["case_ids"] != expected_case_ids
        or catalogue["protocol"] != gmm.PROTOCOL
        or catalogue["construction_method"] != gmm.CONSTRUCTION_METHOD
        or catalogue["sample_count_per_case"] != gmm.PROTECTED_HOLDOUT_SAMPLE_COUNT
        or catalogue["split_domain"] != PROTECTED_SPLIT_DOMAIN
        or not _is_sha256(master_seed_hex)
    ):
        raise ValueError("protected catalogue does not match the frozen six-case contract")
    return catalogue, raw_sha256


def _protected_seed(master_seed_hex: str, *, case_id: str) -> int:
    """Derive the source-pinned protected Sobol seed for one case.

    Args:
        master_seed_hex: Canonical 256-bit lowercase hexadecimal master seed.
        case_id: One of the six frozen development case IDs.

    Returns:
        An unsigned 64-bit integer interpreted little-endian from the first
        eight bytes of the SHA-256 digest.

    Raises:
        ValueError: If either input is outside the frozen catalogue.
    """
    expected_case_ids = {"__".join(case) for case in gmm.DEVELOPMENT_MATRIX}
    if not _is_sha256(master_seed_hex):
        raise ValueError("protected master seed must be 64 lowercase hexadecimal characters")
    if case_id not in expected_case_ids:
        raise ValueError("protected seed requested for an unknown case")
    material = _canonical_json(
        {
            "schema": CATALOGUE_SCHEMA,
            "split_domain": PROTECTED_SPLIT_DOMAIN,
            "case_id": case_id,
            "master_seed_hex": master_seed_hex,
        }
    ).encode("ascii")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _read_certificate(
    path: Path,
    *,
    expected_raw_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    """Read and authenticate a canonical completed-certificate envelope."""
    if not _is_sha256(expected_raw_sha256):
        raise ValueError("expected development certificate SHA-256 is not canonical")
    if path.is_symlink() or not path.is_file():
        raise ValueError("development certificate must be one regular non-symlink file")
    raw = path.read_bytes()
    raw_sha256 = _sha256_bytes(raw)
    if raw_sha256 != expected_raw_sha256:
        raise ValueError("development certificate raw SHA-256 does not match")
    envelope = _parse_canonical_json(raw, label="development certificate")
    if set(envelope) != {"payload", "sha256"}:
        raise ValueError("development certificate has an unexpected envelope schema")
    payload = _require_mapping(envelope["payload"], "development certificate payload")
    internal_sha256 = c1._sha256_json(payload)
    if envelope["sha256"] != internal_sha256:
        raise ValueError("development certificate internal digest does not match")
    return payload, internal_sha256, raw_sha256


def _certificate_cases(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize certificate case summaries to a six-entry mapping."""
    raw_cases = _require_list(payload.get("cases"), "development certificate cases")
    expected_ids = tuple("__".join(case) for case in gmm.DEVELOPMENT_MATRIX)
    cases: dict[str, dict[str, Any]] = {}
    for raw_case in raw_cases:
        case = _require_mapping(raw_case, "development certificate case")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in cases:
            raise ValueError("development certificate has an invalid or duplicate case ID")
        cases[case_id] = case
    if tuple(cases) != expected_ids:
        raise ValueError("development certificate does not contain exactly six frozen cases")
    return cases


def _finite_number(value: object, label: str) -> float:
    """Return one finite non-Boolean numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validate_certificate_case(
    case: Mapping[str, Any],
    *,
    expected_case_id: str,
    locked_sample_count: int,
) -> None:
    """Recompute one certificate case's nomination and confirmation decision."""
    expected_fields = {
        "case_id",
        "input_sha256",
        "context_sha256",
        "locked_sample_count",
        "nominated_development_raw_sha256",
        "nominated_development_evaluation",
        "nominated_fitted_bundle_sha256",
        "nominated_artifact_sha256",
        "confirmation_evaluations",
        "confirmation_individual_passes",
        "four_bank_log_evidence_range_nat",
        "four_bank_log_evidence_range_pass",
        "confirmation_pass",
        "development_pass",
    }
    if set(case) != expected_fields:
        raise ValueError(f"{expected_case_id} certificate summary has an unexpected schema")
    nominated = _require_mapping(
        case["nominated_development_evaluation"],
        f"{expected_case_id} nominated development evaluation",
    )
    training = _require_mapping(
        nominated.get("training"),
        f"{expected_case_id} nominated development training",
    )
    envelope = _require_mapping(
        training.get("fitted_bundle_envelope"),
        f"{expected_case_id} nominated fitted bundle",
    )
    if (
        case["case_id"] != expected_case_id
        or not _is_sha256(case["input_sha256"])
        or not _is_sha256(case["context_sha256"])
        or not _is_sha256(case["nominated_development_raw_sha256"])
        or case["locked_sample_count"] != locked_sample_count
        or nominated.get("sample_count") != locked_sample_count
        or nominated.get("base_seed") != gmm.DEVELOPMENT_SELECTION_SEED
        or nominated.get("scientific_pass") is not True
        or nominated.get("scientific_model_gates_pass") is not True
        or nominated.get("fit_development_pass") is not True
        or training.get("training_sample_count") != locked_sample_count
        or training.get("artifact_sha256") != case["nominated_artifact_sha256"]
        or envelope.get("sha256") != case["nominated_fitted_bundle_sha256"]
        or not _is_sha256(case["nominated_fitted_bundle_sha256"])
        or not _is_sha256(case["nominated_artifact_sha256"])
        or case["confirmation_pass"] is not True
        or case["development_pass"] is not True
    ):
        raise ValueError(f"{expected_case_id} certificate nomination is not the locked seed-731 pass")
    confirmations = [
        _require_mapping(value, f"{expected_case_id} confirmation evaluation")
        for value in _require_list(
            case["confirmation_evaluations"],
            f"{expected_case_id} confirmation evaluations",
        )
    ]
    expected_confirmations = [
        {
            "base_seed": seed,
            "pass": True,
        }
        for seed in gmm.CONFIRMATION_SEEDS
    ]
    observed_confirmations: list[dict[str, Any]] = []
    if len(confirmations) != len(gmm.CONFIRMATION_SEEDS):
        raise ValueError(f"{expected_case_id} certificate has incomplete confirmation evidence")
    evidence = [
        _finite_number(
            _require_mapping(
                nominated.get("posterior_summary"),
                f"{expected_case_id} nominated posterior summary",
            ).get("log_evidence"),
            f"{expected_case_id} nominated log evidence",
        )
    ]
    for seed, confirmation in zip(gmm.CONFIRMATION_SEEDS, confirmations, strict=True):
        if (
            confirmation.get("base_seed") != seed
            or confirmation.get("sample_count") != locked_sample_count
            or confirmation.get("scientific_pass") is not True
        ):
            raise ValueError(f"{expected_case_id} confirmation identity or gate changed")
        observed_confirmations.append({"base_seed": seed, "pass": True})
        evidence.append(
            _finite_number(
                _require_mapping(
                    confirmation.get("posterior_summary"),
                    f"{expected_case_id} confirmation posterior summary",
                ).get("log_evidence"),
                f"{expected_case_id} confirmation log evidence",
            )
        )
    evidence_range = float(max(evidence) - min(evidence))
    if (
        observed_confirmations != expected_confirmations
        or case["confirmation_individual_passes"] != expected_confirmations
        or _finite_number(
            case["four_bank_log_evidence_range_nat"],
            f"{expected_case_id} four-bank evidence range",
        )
        != evidence_range
        or case["four_bank_log_evidence_range_pass"]
        is not (evidence_range <= c1.THRESHOLDS["between_bank_log_evidence_range_nat"])
        or case["four_bank_log_evidence_range_pass"] is not True
    ):
        raise ValueError(f"{expected_case_id} four-bank evidence gate is inconsistent")


def _validate_certificate(
    payload: Mapping[str, Any],
    *,
    expected_source_revision: str,
) -> tuple[int, dict[str, dict[str, Any]]]:
    """Validate the completed six-case development decision."""
    expected_fields = {
        "schema",
        "certification_protocol",
        "certification_protocol_sha256",
        "certifier_source_sha256",
        "source_git_revision",
        "scientific_driver_sha256",
        "frozen_development_protocol_sha256",
        "a1_definitions_sha256",
        "matrix_catalogue",
        "common_lock_raw_sha256",
        "common_lock_sha256",
        "locked_sample_count",
        "confirmation_seeds",
        "execution_certified",
        "decision",
        "development_pass",
        "eligible_for_protected_holdout",
        "protected_holdout_pass",
        "scientific_pass",
        "scientific_pass_available",
        "scientific_pass_reason",
        "structural_inference_licensed",
        "held_out_information_read",
        "cases",
    }
    if set(payload) != expected_fields:
        raise ValueError("development certificate payload has an unexpected schema")
    source_revision = payload.get("source_git_revision")
    driver_sha256 = payload.get("scientific_driver_sha256")
    matrix_catalogue = _require_mapping(
        payload.get("matrix_catalogue"),
        "development certificate matrix catalogue",
    )
    protected = _require_mapping(
        matrix_catalogue.get("protected_holdout"),
        "development certificate protected catalogue declaration",
    )
    locked = payload.get("locked_sample_count")
    if (
        payload.get("schema") != development_certifier.CERTIFICATE_SCHEMA
        or payload.get("certification_protocol") != development_certifier.CERTIFICATION_PROTOCOL
        or payload.get("certification_protocol_sha256")
        != development_certifier._certification_protocol_sha256()
        or payload.get("certifier_source_sha256") != development_certifier._certifier_source_sha256()
        or source_revision != expected_source_revision
        or not isinstance(source_revision, str)
        or len(source_revision) != _GIT_SHA_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in source_revision)
        or driver_sha256 != development_certifier._driver_source_sha256()
        or payload.get("frozen_development_protocol_sha256") != gmm.DEVELOPMENT_PROTOCOL_SHA256
        or payload.get("a1_definitions_sha256") != c1.A1_DEFINITIONS_SHA256
        or matrix_catalogue != gmm.matrix_catalogue()
        or protected.get("sha256") != gmm.PROTECTED_HOLDOUT_CATALOGUE_SHA256
        or not _is_sha256(payload.get("common_lock_raw_sha256"))
        or not _is_sha256(payload.get("common_lock_sha256"))
        or isinstance(locked, bool)
        or not isinstance(locked, int)
        or locked not in gmm.DEVELOPMENT_SAMPLE_COUNTS
        or payload.get("confirmation_seeds") != list(gmm.CONFIRMATION_SEEDS)
        or payload.get("execution_certified") is not True
        or payload.get("decision") != "pass"
        or payload.get("development_pass") is not True
        or payload.get("eligible_for_protected_holdout") is not True
        or payload.get("protected_holdout_pass") is not None
        or payload.get("scientific_pass") is not False
        or payload.get("scientific_pass_available") is not False
        or payload.get("structural_inference_licensed") is not False
        or payload.get("held_out_information_read") is not False
        or not isinstance(payload.get("scientific_pass_reason"), str)
    ):
        raise ValueError("development certificate is not a completed eligible frozen-protocol result")
    cases = _certificate_cases(payload)
    for case_id, case in cases.items():
        _validate_certificate_case(
            case,
            expected_case_id=case_id,
            locked_sample_count=locked,
        )
    return locked, cases


def _regular_file_digest_map(directory: Path) -> dict[str, Path]:
    """Index the exact immutable 24-shard development bundle by raw digest."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("development shard directory must be one real directory")
    expected_names = {
        development_certifier._development_filename(case_id, sample_count)
        for case_id in development_certifier._case_ids()
        for sample_count in gmm.DEVELOPMENT_SAMPLE_COUNTS
    }
    observed_names = development_certifier._regular_file_names(
        directory,
        "development shard directory",
    )
    if observed_names != expected_names:
        raise ValueError(
            "development shard directory must contain exactly the frozen 24 files; "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )
    result: dict[str, Path] = {}
    for name in sorted(expected_names):
        path = directory / name
        digest = _sha256_bytes(path.read_bytes())
        if digest in result:
            raise ValueError("development shard directory contains duplicate raw content")
        result[digest] = path
    if not result:
        raise ValueError("development shard directory contains no files")
    return result


def _nominated_raw_sha256(case: Mapping[str, Any], *, case_id: str) -> str:
    """Return the certificate-pinned raw shard digest for one nomination."""
    candidates = (
        case.get("nominated_development_raw_sha256"),
        case.get("development_input_raw_sha256"),
        case.get("nominated_shard_raw_sha256"),
    )
    values = {value for value in candidates if value is not None}
    if len(values) != 1:
        raise ValueError(f"{case_id} certificate has no unambiguous nominated raw shard SHA-256")
    value = values.pop()
    if not _is_sha256(value):
        raise ValueError(f"{case_id} nominated raw shard SHA-256 is malformed")
    return cast(str, value)


def _read_nominated_evaluation(
    path: Path,
    *,
    raw_sha256: str,
    case_id: str,
    locked_sample_count: int,
    source_revision: str,
    driver_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one raw-SHA-selected seed-731 locked development evaluation."""
    raw = path.read_bytes()
    if _sha256_bytes(raw) != raw_sha256:
        raise ValueError(f"{case_id} nominated development shard raw SHA-256 changed")
    shard = _parse_canonical_json(raw, label=f"{case_id} nominated development shard")
    if (
        shard.get("schema") != gmm.SCHEMA
        or shard.get("protocol") != gmm.PROTOCOL
        or shard.get("profile") != "development"
        or shard.get("selected_case_id") != case_id
        or shard.get("execution_mode") not in {"development_size_shard", "development_ladder"}
        or shard.get("source_git_revision") != source_revision
        or shard.get("driver_sha256") != driver_sha256
        or shard.get("protocol_sha256") != gmm._protocol_sha256()
        or shard.get("frozen_development_protocol_sha256") != gmm.DEVELOPMENT_PROTOCOL_SHA256
        or shard.get("a1_definitions_sha256") != c1.A1_DEFINITIONS_SHA256
        or shard.get("structural_inference_licensed") is not False
    ):
        raise ValueError(f"{case_id} nominated development shard identity changed")
    shard_cases = _require_list(shard.get("cases"), f"{case_id} nominated shard cases")
    if len(shard_cases) != 1:
        raise ValueError(f"{case_id} nominated development shard must contain one case")
    shard_case = _require_mapping(shard_cases[0], f"{case_id} nominated shard case")
    evaluations = _require_list(
        shard_case.get("development_evaluations"),
        f"{case_id} nominated development evaluations",
    )
    matches = [
        _require_mapping(evaluation, f"{case_id} nominated evaluation")
        for evaluation in evaluations
        if isinstance(evaluation, dict)
        and evaluation.get("sample_count") == locked_sample_count
        and evaluation.get("base_seed") == gmm.DEVELOPMENT_SELECTION_SEED
    ]
    if len(matches) != 1:
        raise ValueError(f"{case_id} must promote exactly one seed-731 evaluation at the common locked size")
    evaluation = matches[0]
    if shard_case.get("case_id") != case_id:
        raise ValueError(f"{case_id} nominated shard contains the wrong case")
    return shard_case, evaluation


def _exact_case(expected_case: tuple[str, str, str]) -> _ExactCase:
    """Reconstruct one exact case without fitting or selecting a model."""
    regime_name, family_name, tiling = expected_case
    if tiling != "root":
        raise ValueError("protected GMM certification is root-only")
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
    )


def _require_same_json(observed: object, expected: object, label: str) -> None:
    """Reject a recorded value unless it equals a recomputed JSON value."""
    if _canonical_json(observed) != _canonical_json(expected):
        raise ValueError(f"{label} does not match the authenticated replay")


def _reverify_scientific_gates(
    *,
    artifact: ConditionalResidualImageMDN,
    exact: _ExactCase,
    nominated: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and authenticate the nominated model's unchanged gates."""
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
        _require_same_json(
            nominated.get(field),
            replay[field],
            f"{exact.case_id} nominated {field}",
        )
    if (
        set(cast(dict[str, Any], replay["checks"])) != set(_EVALUATION_GATE_KEYS)
        or nominated.get("scientific_model_gates_pass") is not True
        or nominated.get("scientific_pass") is not True
        or replay["scientific_pass"] is not True
    ):
        raise ValueError(f"{exact.case_id} nominated unchanged scientific gates do not pass")
    return replay


def _protected_density_gate(
    protected_draws: FloatArray,
    *,
    artifact: ConditionalResidualImageMDN,
    development_generalization: Mapping[str, Any],
) -> dict[str, float | int | bool]:
    """Compare protected and development-validation NLL using the frozen gate."""
    if (
        protected_draws.ndim != 2
        or protected_draws.shape[0] < 2
        or protected_draws.shape[1] != artifact.residual_rank
        or not np.all(np.isfinite(protected_draws))
    ):
        raise ValueError("protected residual draws must be a finite aligned matrix")
    log_weights, means, factors = artifact._components(np.ones(artifact.region_count, dtype=np.float64))
    weights = np.exp(log_weights)
    covariances = factors @ np.transpose(factors, (0, 2, 1))
    protected_values = -gmm._mixture_log_likelihood_values(
        protected_draws,
        weights,
        means,
        covariances,
    )
    validation_nll = development_generalization.get("validation_nll_nat_per_draw")
    validation_mcse = development_generalization.get("validation_nll_mcse_nat_per_draw")
    if (
        isinstance(validation_nll, bool)
        or not isinstance(validation_nll, (int, float))
        or not math.isfinite(validation_nll)
        or isinstance(validation_mcse, bool)
        or not isinstance(validation_mcse, (int, float))
        or not math.isfinite(validation_mcse)
        or validation_mcse < 0.0
    ):
        raise ValueError("development validation NLL evidence is malformed")
    protected_nll = float(np.mean(protected_values))
    protected_mcse = math.sqrt(float(np.var(protected_values, ddof=1)) / protected_values.size)
    combined_mcse = math.hypot(float(validation_mcse), protected_mcse)
    absolute_gap = abs(protected_nll - float(validation_nll))
    fixed_floor = gmm.GENERALIZATION_NAT_PER_DIMENSION * artifact.residual_rank
    threshold = max(
        fixed_floor,
        gmm.GENERALIZATION_MCSE_MULTIPLIER * combined_mcse,
    )
    return {
        "residual_dimension": artifact.residual_rank,
        "sample_count": int(protected_values.size),
        "development_validation_nll_nat_per_draw": float(validation_nll),
        "development_validation_nll_mcse_nat_per_draw": float(validation_mcse),
        "protected_nll_nat_per_draw": protected_nll,
        "protected_nll_mcse_nat_per_draw": protected_mcse,
        "combined_nll_mcse_nat_per_draw": combined_mcse,
        "absolute_nll_gap_nat_per_draw": absolute_gap,
        "fixed_floor_nat_per_draw": fixed_floor,
        "threshold_nat_per_draw": threshold,
        "pass": bool(absolute_gap <= threshold),
    }


def _case_digest(case: Mapping[str, Any], *names: str) -> str:
    """Return one unambiguous canonical digest recorded for a certificate case."""
    values = {case.get(name) for name in names if case.get(name) is not None}
    if len(values) != 1:
        raise ValueError(f"certificate case has no unambiguous {names[0]}")
    value = values.pop()
    if not _is_sha256(value):
        raise ValueError(f"certificate case has malformed {names[0]}")
    return cast(str, value)


def _prepare_nominated_case(
    *,
    expected_case: tuple[str, str, str],
    certificate_case: Mapping[str, Any],
    shard_path: Path,
    shard_raw_sha256: str,
    locked_sample_count: int,
    source_revision: str,
    driver_sha256: str,
) -> _PreparedCase:
    """Authenticate and replay one promotion before opening the seal."""
    case_id = "__".join(expected_case)
    shard_case, nominated = _read_nominated_evaluation(
        shard_path,
        raw_sha256=shard_raw_sha256,
        case_id=case_id,
        locked_sample_count=locked_sample_count,
        source_revision=source_revision,
        driver_sha256=driver_sha256,
    )
    _require_same_json(
        nominated,
        certificate_case.get("nominated_development_evaluation"),
        f"{case_id} certificate/shard nominated evaluation",
    )
    if (
        nominated.get("base_seed") != gmm.DEVELOPMENT_SELECTION_SEED
        or nominated.get("sample_count") != locked_sample_count
    ):
        raise ValueError(f"{case_id} attempted to promote a non-seed-731 or unlocked artifact")
    training = _require_mapping(nominated.get("training"), f"{case_id} nominated training")
    if training.get("training_sample_count") != locked_sample_count:
        raise ValueError(f"{case_id} nominated training size is not the common lock")
    envelope = _require_mapping(
        training.get("fitted_bundle_envelope"),
        f"{case_id} nominated fitted bundle",
    )
    expected_envelope_sha256 = _case_digest(
        certificate_case,
        "nominated_fitted_bundle_sha256",
        "fitted_bundle_sha256",
    )
    expected_artifact_sha256 = _case_digest(
        certificate_case,
        "nominated_artifact_sha256",
        "artifact_sha256",
        "normalized_artifact_sha256",
    )
    artifact = gmm.validate_fitted_bundle_envelope(
        envelope,
        expected_sha256=expected_envelope_sha256,
        expected_source_git_revision=source_revision,
        expected_driver_sha256=driver_sha256,
    )
    if (
        artifact.artifact_sha256 != expected_artifact_sha256
        or training.get("artifact_sha256") != expected_artifact_sha256
    ):
        raise ValueError(f"{case_id} normalized promoted artifact identity changed")
    exact = _exact_case(expected_case)
    if (
        shard_case.get("input_sha256") != exact.input_sha256
        or certificate_case.get("input_sha256") != exact.input_sha256
        or shard_case.get("context_sha256") != exact.context.artifact_sha256
        or certificate_case.get("context_sha256") != exact.context.artifact_sha256
        or artifact.context.artifact_sha256 != exact.context.artifact_sha256
    ):
        raise ValueError(f"{case_id} exact input or residual-image context changed")
    replay = _reverify_scientific_gates(
        artifact=artifact,
        exact=exact,
        nominated=nominated,
    )
    generalization = _require_mapping(
        training.get("simulator_test_generalization"),
        f"{case_id} development generalization evidence",
    )
    if (
        training.get("validation_nll") != generalization.get("validation_nll_nat_per_draw")
        or nominated.get("fit_development_pass") is not True
        or training.get("fit_development_pass") is not True
        or generalization.get("pass") is not True
    ):
        raise ValueError(f"{case_id} development density gates do not pass")
    return _PreparedCase(
        case_id=case_id,
        locked_sample_count=locked_sample_count,
        shard_raw_sha256=shard_raw_sha256,
        fitted_bundle_sha256=expected_envelope_sha256,
        artifact_sha256=expected_artifact_sha256,
        artifact=artifact,
        exact=exact,
        generalization=generalization,
        scientific_gate_pass=bool(replay["scientific_pass"]),
    )


def _certify_prepared_case(
    prepared: _PreparedCase,
    *,
    master_seed_hex: str,
) -> dict[str, Any]:
    """Apply the protected density gate to one authenticated promotion."""
    case_id = prepared.case_id
    protected_seed = _protected_seed(master_seed_hex, case_id=case_id)
    protected_draws, protected_bank_sha256 = gmm._residual_image_draws(
        prepared.exact.aggregation,
        prepared.exact.labels,
        prepared.exact.context,
        sample_count=gmm.PROTECTED_HOLDOUT_SAMPLE_COUNT,
        source_seed=protected_seed,
        source_provenance=(
            f"{gmm.PROTOCOL}:{case_id}:S={gmm.PROTECTED_HOLDOUT_SAMPLE_COUNT}:domain={PROTECTED_SPLIT_DOMAIN}"
        ),
    )
    density_gate = _protected_density_gate(
        protected_draws,
        artifact=prepared.artifact,
        development_generalization=prepared.generalization,
    )
    scientific_gate_pass = prepared.scientific_gate_pass
    case_pass = bool(scientific_gate_pass and density_gate["pass"])
    return {
        "case_id": case_id,
        "locked_training_sample_count": prepared.locked_sample_count,
        "promoted_development_seed": gmm.DEVELOPMENT_SELECTION_SEED,
        "nominated_development_shard_raw_sha256": prepared.shard_raw_sha256,
        "promoted_fitted_bundle_sha256": prepared.fitted_bundle_sha256,
        "promoted_normalized_artifact_sha256": prepared.artifact_sha256,
        "protected_sobol_seed": protected_seed,
        "protected_bank_sha256": protected_bank_sha256,
        "protected_draws_sha256": gmm._array_sha256(protected_draws),
        "unchanged_scientific_gates_pass": scientific_gate_pass,
        "protected_density_gate": density_gate,
        "pass": case_pass,
        "structural_inference_licensed": False,
    }


def certification_protocol_sha256() -> str:
    """Return the frozen protected certification protocol identity."""
    return c1._sha256_json(
        {
            "schema": SCHEMA,
            "certification_protocol": CERTIFICATION_PROTOCOL,
            "scientific_schema": gmm.SCHEMA,
            "scientific_protocol": gmm.PROTOCOL,
            "frozen_development_protocol_sha256": gmm.DEVELOPMENT_PROTOCOL_SHA256,
            "catalogue_schema": CATALOGUE_SCHEMA,
            "catalogue_raw_sha256": gmm.PROTECTED_HOLDOUT_CATALOGUE_SHA256,
            "matrix": gmm.DEVELOPMENT_MATRIX,
            "construction_method": gmm.CONSTRUCTION_METHOD,
            "sample_count_per_case": gmm.PROTECTED_HOLDOUT_SAMPLE_COUNT,
            "split_domain": PROTECTED_SPLIT_DOMAIN,
            "seed_derivation": PROTECTED_SEED_DERIVATION,
            "promoted_seed": gmm.DEVELOPMENT_SELECTION_SEED,
            "generalization_nat_per_dimension": gmm.GENERALIZATION_NAT_PER_DIMENSION,
            "generalization_mcse_multiplier": gmm.GENERALIZATION_MCSE_MULTIPLIER,
            "thresholds": c1.THRESHOLDS,
        }
    )


def certify(
    *,
    source_directory: Path,
    expected_source_revision: str,
    development_certificate: Path,
    expected_development_certificate_sha256: str,
    development_shards_directory: Path,
    protected_catalogue: Path,
    output: Path,
) -> dict[str, Any]:
    """Run the one-shot protected holdout and publish an immutable certificate.

    Args:
        source_directory: Clean source tree containing the imported certifier.
        expected_source_revision: Full lowercase Git revision binding all
            development and protected evidence.
        development_certificate: Canonical completed development certificate.
        expected_development_certificate_sha256: Externally recorded raw
            SHA-256 digest of ``development_certificate``.
        development_shards_directory: Immutable shards containing each
            certificate-nominated full seed-731 evaluation.
        protected_catalogue: Sealed canonical protected catalogue.
        output: Fresh output JSON path.

    Returns:
        The protected certification payload.

    Raises:
        FileExistsError: If ``output`` already exists.
        ValueError: If any input identity or scientific invariant fails.
    """
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    # Authenticate the frozen runtime, live source, passing development
    # certificate, and its nominated development artifacts before the sealed
    # protected input is touched.  Once eligibility is established,
    # ``_read_protected_catalogue`` still authenticates the raw bytes before
    # parsing them.
    gmm._validate_development_protocol()
    development_certifier._validate_live_source(
        source_directory,
        expected_source_revision,
    )
    certificate, certificate_internal_sha256, certificate_raw_sha256 = _read_certificate(
        development_certificate,
        expected_raw_sha256=expected_development_certificate_sha256,
    )
    locked_sample_count, certificate_cases = _validate_certificate(
        certificate,
        expected_source_revision=expected_source_revision,
    )
    shard_by_digest = _regular_file_digest_map(development_shards_directory)
    source_revision = cast(str, certificate["source_git_revision"])
    driver_sha256 = cast(
        str,
        certificate.get("scientific_driver_sha256", certificate.get("driver_sha256")),
    )
    prepared_cases: list[_PreparedCase] = []
    for expected_case in gmm.DEVELOPMENT_MATRIX:
        case_id = "__".join(expected_case)
        certificate_case = certificate_cases[case_id]
        shard_raw_sha256 = _nominated_raw_sha256(certificate_case, case_id=case_id)
        shard_path = shard_by_digest.get(shard_raw_sha256)
        if shard_path is None:
            raise ValueError(f"{case_id} nominated development shard is absent")
        prepared_cases.append(
            _prepare_nominated_case(
                expected_case=expected_case,
                certificate_case=certificate_case,
                shard_path=shard_path,
                shard_raw_sha256=shard_raw_sha256,
                locked_sample_count=locked_sample_count,
                source_revision=source_revision,
                driver_sha256=driver_sha256,
            )
        )
    catalogue, catalogue_raw_sha256 = _read_protected_catalogue(protected_catalogue)
    master_seed_hex = cast(str, catalogue["master_seed_hex"])
    case_results = [
        _certify_prepared_case(prepared, master_seed_hex=master_seed_hex) for prepared in prepared_cases
    ]
    all_six_pass = bool(
        len(case_results) == len(gmm.DEVELOPMENT_MATRIX) and all(case["pass"] for case in case_results)
    )
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "certification_protocol": CERTIFICATION_PROTOCOL,
        "certification_protocol_sha256": certification_protocol_sha256(),
        "certifier_source_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "scientific_driver_sha256": driver_sha256,
        "source_git_revision": source_revision,
        "frozen_development_protocol_sha256": gmm.DEVELOPMENT_PROTOCOL_SHA256,
        "development_certificate_internal_sha256": certificate_internal_sha256,
        "development_certificate_raw_sha256": certificate_raw_sha256,
        "protected_catalogue_raw_sha256": catalogue_raw_sha256,
        "protected_catalogue_schema": CATALOGUE_SCHEMA,
        "protected_split_domain": PROTECTED_SPLIT_DOMAIN,
        "protected_seed_derivation": PROTECTED_SEED_DERIVATION,
        "common_locked_training_sample_count": locked_sample_count,
        "promoted_development_seed": gmm.DEVELOPMENT_SELECTION_SEED,
        "case_count": len(case_results),
        "passing_case_count": sum(bool(case["pass"]) for case in case_results),
        "cases": case_results,
        "decision": "pass" if all_six_pass else "hard_stop",
        "protected_holdout_pass": all_six_pass,
        "scientific_pass_available": True,
        "scientific_pass": all_six_pass,
        "structural_inference_licensed": False,
    }
    _write_atomic_json(output, summary)
    return summary


def _write_atomic_json(path: Path, payload: object) -> None:
    """Publish canonical JSON atomically while refusing replacement."""
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
        ) as handle:
            temporary = Path(handle.name)
            handle.write(f"{_canonical_json(payload)}\n")
            handle.flush()
            os.fsync(handle.fileno())
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


def _parser() -> argparse.ArgumentParser:
    """Build the strict protected-certification command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--development-certificate", type=Path, required=True)
    parser.add_argument(
        "--expected-development-certificate-sha256",
        required=True,
    )
    parser.add_argument("--development-shards-directory", type=Path, required=True)
    parser.add_argument("--protected-catalogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run protected certification from command-line arguments."""
    arguments = _parser().parse_args(argv)
    certify(
        source_directory=arguments.source_directory,
        expected_source_revision=arguments.expected_source_revision,
        development_certificate=arguments.development_certificate,
        expected_development_certificate_sha256=(arguments.expected_development_certificate_sha256),
        development_shards_directory=arguments.development_shards_directory,
        protected_catalogue=arguments.protected_catalogue,
        output=arguments.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
