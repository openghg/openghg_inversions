"""Fail-closed retry lineage for the bounded synthetic local-search experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, cast

from .mh_local_search_conditional_reference import (
    validate_conditional_reference_record,
)
from .mh_local_search_synthetic import (
    SyntheticEvaluationArtifact,
    SyntheticTrainingArtifact,
    canonical_json,
    file_sha256,
    load_evaluation_artifact,
    load_training_artifact,
    validate_artifact_pair,
)

_DIGEST = re.compile(r"[0-9a-f]{64}")
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_LOCAL_FAILURE_GATES = frozenset(
    (
        "local_mcse_over_nuts_sd",
        "half_difference_over_nuts_sd",
        "local_vs_nuts_tolerance",
    )
)
_NUTS_FAILURE_GATES = frozenset(("zero_divergences", "rank_normalized_rhat", "bulk_ess", "tail_ess"))
RETRY_AUTHORIZATION_TOKEN_KEYS = frozenset(
    (
        "schema",
        "source_revision",
        "definition_sha256",
        "scope",
        "authorized_branch_profile",
        "primary_conditional_reference_completion_sha256",
        "primary_nuts_completion_sha256",
        "primary_local_completion_sha256",
    )
)


@dataclass(frozen=True, slots=True)
class RetryAuthorization:
    """Truth-free token plus the evidence-bearing issuance audit."""

    token: Mapping[str, object]
    audit: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PrimaryNUTSFailure:
    """Validated lineage required before the sole NUTS retry."""

    completion_sha256: str
    first_failed_gate: str


def _strict_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {token}")),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{path} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return cast(dict[str, object], value)


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lower-case SHA-256")
    return value


def _validate_conditional_completion(directory: Path) -> tuple[dict[str, object], str]:
    completion_path = directory / "complete.json"
    completion = _strict_json(completion_path)
    if (
        frozenset(completion)
        != {
            "schema",
            "status",
            "pass",
            "first_failed_gate",
            "files",
        }
        or completion["schema"] != "openghg_inversions.mh_local_search_conditional_reference_completion.v1"
        or completion["status"] != "complete"
        or not isinstance(completion["pass"], bool)
        or not isinstance(completion["files"], dict)
        or frozenset(cast(dict[str, object], completion["files"]))
        != {"conditional_reference.json", "audit.json"}
    ):
        raise ValueError("primary conditional-reference completion is incompatible")
    for name, raw_digest in cast(dict[str, object], completion["files"]).items():
        expected = _digest(raw_digest, name=f"conditional-reference {name} digest")
        path = directory / name
        if not path.is_file() or path.is_symlink() or file_sha256(path) != expected:
            raise ValueError(f"primary conditional-reference checksum mismatch for {name}")
    return completion, file_sha256(completion_path)


def issue_factor4_retry_authorization(
    *,
    training_path: Path,
    evaluation_path: Path,
    primary_certificate_directory: Path,
    primary_nuts_directory: Path,
    primary_local_directory: Path,
    source_revision: str,
) -> RetryAuthorization:
    """Issue the sole factor-four retry only from a replayed local-gate failure."""
    if _FULL_SHA.fullmatch(source_revision) is None:
        raise ValueError("source_revision must be an exact lower-case full Git SHA")
    training = load_training_artifact(training_path)
    evaluation = load_evaluation_artifact(evaluation_path)
    validate_artifact_pair(training, evaluation)
    completion, completion_digest = _validate_conditional_completion(primary_certificate_directory)
    record = _strict_json(primary_certificate_directory / "conditional_reference.json")
    replayed = validate_conditional_reference_record(
        record,
        training_path=training_path,
        evaluation_path=evaluation_path,
        nuts_directory=primary_nuts_directory,
        local_directory=primary_local_directory,
    )
    replayed_record = replayed.record
    replayed_audit = replayed.audit
    nuts_audit = replayed_audit.get("nuts")
    local_audit = replayed_audit.get("local")
    if (
        replayed_audit.get("source_revision") != source_revision
        or not isinstance(nuts_audit, Mapping)
        or not isinstance(local_audit, Mapping)
    ):
        raise ValueError("primary conditional-reference provenance is incompatible")
    failure = replayed_record["first_failed_gate"]
    nuts_passed = (
        int(cast(Any, replayed_record["divergences"])) == 0
        and float(cast(Any, replayed_record["worst_rhat_value"])) <= 1.01
        and float(cast(Any, replayed_record["min_bulk_ess_value"])) >= 200.0
        and float(cast(Any, replayed_record["min_tail_ess_value"])) >= 200.0
    )
    if (
        replayed_record["profile"] not in ("primary", "retry1")
        or nuts_audit.get("profile") != replayed_record["profile"]
        or local_audit.get("profile") != "primary"
        or replayed_record["pass"] is not False
        or completion["pass"] is not False
        or completion["first_failed_gate"] != failure
        or not nuts_passed
        or failure not in _LOCAL_FAILURE_GATES
    ):
        raise ValueError(
            "factor-four authorization requires a failed primary certificate "
            "whose NUTS gates passed and whose first failure is local"
        )
    nuts_completion_digest = file_sha256(primary_nuts_directory / "complete.json")
    local_completion_digest = file_sha256(primary_local_directory / "complete.json")
    if (
        replayed_record["nuts_artifact_sha256"] != nuts_completion_digest
        or replayed_record["local_artifact_sha256"] != local_completion_digest
    ):
        raise ValueError("primary conditional-reference bundle identity is inconsistent")
    token: dict[str, object] = {
        "schema": "openghg_inversions.mh_local_search_retry_authorization_token.v1",
        "source_revision": source_revision,
        "definition_sha256": training.definition_sha256,
        "scope": f"{training.stage}-homogeneous-factor4-branch-matrix-v1",
        "authorized_branch_profile": "factor4",
        "primary_conditional_reference_completion_sha256": completion_digest,
        "primary_nuts_completion_sha256": nuts_completion_digest,
        "primary_local_completion_sha256": local_completion_digest,
    }
    if frozenset(token) != RETRY_AUTHORIZATION_TOKEN_KEYS:
        raise RuntimeError("retry-authorization token schema drifted")
    audit: dict[str, object] = {
        "schema": "openghg_inversions.mh_local_search_retry_authorization_audit.v1",
        "source_revision": source_revision,
        "training_sha256": file_sha256(training_path),
        "evaluation_sha256": file_sha256(evaluation_path),
        "conditional_reference_completion_sha256": completion_digest,
        "conditional_reference_record": dict(replayed_record),
        "conditional_reference_audit": dict(replayed_audit),
        "authorized_token": token,
    }
    return RetryAuthorization(token=token, audit=audit)


def validate_retry_authorization_token(
    path: Path,
    *,
    source_revision: str,
    definition_sha256: str,
    stage: str | None = None,
) -> str:
    """Validate the sealed truth-free token and return its content address."""
    token = _strict_json(path)
    if (
        frozenset(token) != RETRY_AUTHORIZATION_TOKEN_KEYS
        or token["schema"] != "openghg_inversions.mh_local_search_retry_authorization_token.v1"
        or token["source_revision"] != source_revision
        or token["definition_sha256"] != definition_sha256
        or (stage is not None and token["scope"] != f"{stage}-homogeneous-factor4-branch-matrix-v1")
        or (
            stage is None
            and token["scope"]
            not in (
                "s0-homogeneous-factor4-branch-matrix-v1",
                "s1-homogeneous-factor4-branch-matrix-v1",
            )
        )
        or token["authorized_branch_profile"] != "factor4"
    ):
        raise ValueError("retry-authorization token identity is incompatible")
    for name in (
        "definition_sha256",
        "primary_conditional_reference_completion_sha256",
        "primary_nuts_completion_sha256",
        "primary_local_completion_sha256",
    ):
        _digest(token[name], name=f"retry-authorization {name}")
    if _FULL_SHA.fullmatch(cast(str, token["source_revision"])) is None:
        raise ValueError("retry-authorization source revision is incompatible")
    return file_sha256(path)


def validate_retry_authorization_bundle(
    *,
    directory: Path,
    training_path: Path,
    evaluation_path: Path,
    primary_certificate_directory: Path,
    primary_nuts_directory: Path,
    primary_local_directory: Path,
    source_revision: str,
) -> str:
    """Reissue from archived evidence and require an exact sealed bundle."""
    training = load_training_artifact(training_path)
    completion = _strict_json(directory / "complete.json")
    if (
        frozenset(completion) != {"schema", "status", "token_sha256", "files"}
        or completion["schema"] != "openghg_inversions.mh_local_search_retry_authorization_completion.v1"
        or completion["status"] != "complete"
        or not isinstance(completion["files"], dict)
        or frozenset(cast(dict[str, object], completion["files"])) != {"token.json", "audit.json"}
    ):
        raise ValueError("retry-authorization completion is incompatible")
    files = cast(dict[str, object], completion["files"])
    for name, raw_digest in files.items():
        expected = _digest(raw_digest, name=f"retry-authorization {name} digest")
        path = directory / name
        if not path.is_file() or path.is_symlink() or file_sha256(path) != expected:
            raise ValueError(f"retry-authorization checksum mismatch for {name}")
    token_digest = _digest(
        completion["token_sha256"],
        name="retry-authorization token_sha256",
    )
    if token_digest != files["token.json"]:
        raise ValueError("retry-authorization completion token digest is inconsistent")
    expected = issue_factor4_retry_authorization(
        training_path=training_path,
        evaluation_path=evaluation_path,
        primary_certificate_directory=primary_certificate_directory,
        primary_nuts_directory=primary_nuts_directory,
        primary_local_directory=primary_local_directory,
        source_revision=source_revision,
    )
    token = _strict_json(directory / "token.json")
    audit = _strict_json(directory / "audit.json")
    if canonical_json(token) != canonical_json(expected.token) or canonical_json(audit) != canonical_json(
        expected.audit
    ):
        raise ValueError("retry-authorization bundle differs from evidence-based reissuance")
    validated_digest = validate_retry_authorization_token(
        directory / "token.json",
        source_revision=source_revision,
        definition_sha256=cast(str, token["definition_sha256"]),
        stage=training.stage,
    )
    if validated_digest != token_digest:
        raise ValueError("retry-authorization token content address is inconsistent")
    return token_digest


def validate_primary_nuts_retry_source(
    *,
    training_path: Path,
    evaluation_path: Path,
    primary_nuts_directory: Path,
    topology_role: str,
    source_revision: str,
) -> PrimaryNUTSFailure:
    """Recompute and validate the failed primary NUTS bundle cited by retry1."""
    from .mh_local_search_conditional_reference import _validated_nuts

    training: SyntheticTrainingArtifact = load_training_artifact(training_path)
    evaluation: SyntheticEvaluationArtifact = load_evaluation_artifact(evaluation_path)
    validate_artifact_pair(training, evaluation)
    summary, audit, _, validated_role = _validated_nuts(
        directory=primary_nuts_directory,
        training_path=training_path,
        evaluation_path=evaluation_path,
        training=training,
        evaluation=evaluation,
    )
    failure = summary.get("first_failed_gate")
    if (
        validated_role != topology_role
        or audit.get("source_revision") != source_revision
        or audit.get("profile") != "primary"
        or failure not in _NUTS_FAILURE_GATES
    ):
        raise ValueError("retry1 requires a validated failed primary NUTS completion for the same target")
    return PrimaryNUTSFailure(
        completion_sha256=file_sha256(primary_nuts_directory / "complete.json"),
        first_failed_gate=cast(str, failure),
    )


__all__ = [
    "PrimaryNUTSFailure",
    "RETRY_AUTHORIZATION_TOKEN_KEYS",
    "RetryAuthorization",
    "issue_factor4_retry_authorization",
    "validate_primary_nuts_retry_source",
    "validate_retry_authorization_bundle",
    "validate_retry_authorization_token",
]
