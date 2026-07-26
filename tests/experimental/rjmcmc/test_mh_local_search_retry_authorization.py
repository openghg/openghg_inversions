"""Retry lineage gates for the bounded synthetic local-search experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from openghg_inversions.experimental.rjmcmc import (
    mh_local_search_conditional_reference as conditional,
)
from openghg_inversions.experimental.rjmcmc import (
    mh_local_search_retry_authorization as retry,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_retry_authorization import (
    PrimaryNUTSFailure,
    RETRY_AUTHORIZATION_TOKEN_KEYS,
    issue_factor4_retry_authorization,
    validate_primary_nuts_retry_source,
    validate_retry_authorization_bundle,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_synthetic import (
    EVALUATION_SCHEMA,
    TRAINING_SCHEMA,
    build_stage_definition,
    canonical_json,
    file_sha256,
    materialize_replicate,
    write_envelope,
)

_EXAMPLES = Path(__file__).parents[3] / "examples" / "rjmcmc"


def _load_driver(name: str) -> ModuleType:
    path = _EXAMPLES / f"{name}.py"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _create_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _issuance_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    training, evaluation = materialize_replicate(
        build_stage_definition("s0"),
        scenario="edge-one",
        replicate=0,
    )
    training_path = tmp_path / "training.json"
    evaluation_path = tmp_path / "evaluation.json"
    write_envelope(training_path, TRAINING_SCHEMA, training.payload())
    write_envelope(evaluation_path, EVALUATION_SCHEMA, evaluation.payload())
    nuts = tmp_path / "primary-nuts"
    local = tmp_path / "primary-local"
    certificate = tmp_path / "primary-certificate"
    for directory in (nuts, local, certificate):
        directory.mkdir()
    _create_json(nuts / "complete.json", {"schema": "test.primary.nuts.v1"})
    _create_json(local / "complete.json", {"schema": "test.primary.local.v1"})
    record: dict[str, object] = {
        "cell_id": training.cell_id,
        "definition_sha256": training.definition_sha256,
        "topology_sha256": "1" * 64,
        "nuts_artifact_sha256": file_sha256(nuts / "complete.json"),
        "local_artifact_sha256": file_sha256(local / "complete.json"),
        "profile": "primary",
        "pass": False,
        "divergences": 0,
        "worst_rhat_variable": "root_total",
        "worst_rhat_value": 1.0,
        "min_bulk_ess_variable": "root_total",
        "min_bulk_ess_value": 300.0,
        "min_tail_ess_variable": "root_total",
        "min_tail_ess_value": 300.0,
        "worst_local_mcse_sd_projection": "whole_domain",
        "worst_local_mcse_sd_value": 0.06,
        "worst_half_difference_sd_projection": "whole_domain",
        "worst_half_difference_sd_value": 0.02,
        "worst_local_vs_nuts_tolerance_projection": "whole_domain",
        "worst_local_vs_nuts_tolerance_value": 0.5,
        "first_failed_gate": "local_mcse_over_nuts_sd",
    }
    audit = {
        "source_revision": "a" * 40,
        "nuts": {"profile": "primary"},
        "local": {"profile": "primary"},
    }
    _create_json(certificate / "conditional_reference.json", record)
    _create_json(certificate / "audit.json", audit)
    _create_json(
        certificate / "complete.json",
        {
            "schema": "openghg_inversions.mh_local_search_conditional_reference_completion.v1",
            "status": "complete",
            "pass": False,
            "first_failed_gate": "local_mcse_over_nuts_sd",
            "files": {
                "conditional_reference.json": file_sha256(certificate / "conditional_reference.json"),
                "audit.json": file_sha256(certificate / "audit.json"),
            },
        },
    )
    replay = SimpleNamespace(record=record, audit=audit)
    monkeypatch.setattr(
        retry,
        "validate_conditional_reference_record",
        lambda *args, **kwargs: replay,
    )
    return {
        "training": training,
        "training_path": training_path,
        "evaluation_path": evaluation_path,
        "nuts": nuts,
        "local": local,
        "certificate": certificate,
        "record": record,
        "audit": audit,
        "replay": replay,
    }


@pytest.mark.parametrize("nuts_profile", ["primary", "retry1"])
def test_authorization_is_truth_free_and_only_local_failure_can_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nuts_profile: str,
) -> None:
    item = _issuance_fixture(tmp_path, monkeypatch)
    item["record"]["profile"] = nuts_profile
    item["audit"]["nuts"]["profile"] = nuts_profile
    authorization = issue_factor4_retry_authorization(
        training_path=item["training_path"],
        evaluation_path=item["evaluation_path"],
        primary_certificate_directory=item["certificate"],
        primary_nuts_directory=item["nuts"],
        primary_local_directory=item["local"],
        source_revision="a" * 40,
    )
    assert frozenset(authorization.token) == RETRY_AUTHORIZATION_TOKEN_KEYS
    token_text = canonical_json(dict(authorization.token))
    for forbidden in (
        "evaluation",
        "scenario",
        "truth",
        "witness",
        "heldout",
        "cell_id",
        "first_failed_gate",
    ):
        assert forbidden not in token_text

    item["record"]["divergences"] = 1
    item["record"]["first_failed_gate"] = "divergences"
    item["replay"].record = item["record"]
    with pytest.raises(ValueError, match="NUTS gates passed"):
        issue_factor4_retry_authorization(
            training_path=item["training_path"],
            evaluation_path=item["evaluation_path"],
            primary_certificate_directory=item["certificate"],
            primary_nuts_directory=item["nuts"],
            primary_local_directory=item["local"],
            source_revision="a" * 40,
        )


def test_bundle_reissuance_rejects_forged_rehashed_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _issuance_fixture(tmp_path, monkeypatch)
    driver = _load_driver("mh_local_search_retry_authorization")
    output = tmp_path / "authorization"
    driver.run(
        argparse.Namespace(
            training=item["training_path"],
            evaluation=item["evaluation_path"],
            primary_certificate_directory=item["certificate"],
            primary_nuts_directory=item["nuts"],
            primary_local_directory=item["local"],
            output_directory=output,
            source_revision="a" * 40,
        ),
        enforce_clean_revision=False,
    )
    expected = validate_retry_authorization_bundle(
        directory=output,
        training_path=item["training_path"],
        evaluation_path=item["evaluation_path"],
        primary_certificate_directory=item["certificate"],
        primary_nuts_directory=item["nuts"],
        primary_local_directory=item["local"],
        source_revision="a" * 40,
    )
    assert expected == file_sha256(output / "token.json")
    aggregate_driver = _load_driver("mh_local_search_synthetic")
    retry_paths = {
        "authorization": output / "complete.json",
        "primary_certificate": item["certificate"] / "complete.json",
        "primary_nuts": item["nuts"] / "complete.json",
        "primary_local": item["local"] / "complete.json",
    }
    aggregate_driver._validate_s0_retry_promotion(
        selected_budget_profile="factor4",
        retry_paths=retry_paths,
        indexed_retry_token_digest=expected,
        selected_retry_token_digest=expected,
        reference_inputs={
            item["training"].cell_id: (
                item["training_path"],
                item["evaluation_path"],
            )
        },
        candidate_revision="a" * 40,
    )

    forged = json.loads((output / "token.json").read_text(encoding="utf-8"))
    forged["primary_local_completion_sha256"] = "f" * 64
    _create_json(output / "token.json", forged)
    completion = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    completion["token_sha256"] = file_sha256(output / "token.json")
    completion["files"]["token.json"] = completion["token_sha256"]
    _create_json(output / "complete.json", completion)
    with pytest.raises(ValueError, match="evidence-based reissuance"):
        aggregate_driver._validate_s0_retry_promotion(
            selected_budget_profile="factor4",
            retry_paths=retry_paths,
            indexed_retry_token_digest=file_sha256(output / "token.json"),
            selected_retry_token_digest=file_sha256(output / "token.json"),
            reference_inputs={
                item["training"].cell_id: (
                    item["training_path"],
                    item["evaluation_path"],
                )
            },
            candidate_revision="a" * 40,
        )


def test_authorization_completion_fails_closed_on_second_pass_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _issuance_fixture(tmp_path, monkeypatch)
    driver = _load_driver("mh_local_search_retry_authorization")
    output = tmp_path / "unstable-authorization"
    real_digest = driver.file_sha256
    calls: dict[str, int] = {}

    def drifting_digest(path: Path) -> str:
        digest = real_digest(path)
        if path.parent == output and path.name in ("token.json", "audit.json"):
            calls[path.name] = calls.get(path.name, 0) + 1
            if calls[path.name] > 1:
                return "f" * 64
        return digest

    monkeypatch.setattr(driver, "file_sha256", drifting_digest)
    with pytest.raises(
        RuntimeError,
        match="independent retry-authorization validation failed",
    ):
        driver.run(
            argparse.Namespace(
                training=item["training_path"],
                evaluation=item["evaluation_path"],
                primary_certificate_directory=item["certificate"],
                primary_nuts_directory=item["nuts"],
                primary_local_directory=item["local"],
                output_directory=output,
                source_revision="a" * 40,
            ),
            enforce_clean_revision=False,
        )
    assert not (output / "complete.json").exists()


def test_retry1_nuts_requires_recomputed_failed_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _issuance_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        conditional,
        "_validated_nuts",
        lambda **kwargs: (
            {"first_failed_gate": "bulk_ess"},
            {"source_revision": "a" * 40, "profile": "primary"},
            "1" * 64,
            "p0",
        ),
    )
    failure = validate_primary_nuts_retry_source(
        training_path=item["training_path"],
        evaluation_path=item["evaluation_path"],
        primary_nuts_directory=item["nuts"],
        topology_role="p0",
        source_revision="a" * 40,
    )
    assert failure == PrimaryNUTSFailure(
        completion_sha256=file_sha256(item["nuts"] / "complete.json"),
        first_failed_gate="bulk_ess",
    )

    monkeypatch.setattr(
        conditional,
        "_validated_nuts",
        lambda **kwargs: (
            {"first_failed_gate": None},
            {"source_revision": "a" * 40, "profile": "primary"},
            "1" * 64,
            "p0",
        ),
    )
    with pytest.raises(ValueError, match="failed primary"):
        validate_primary_nuts_retry_source(
            training_path=item["training_path"],
            evaluation_path=item["evaluation_path"],
            primary_nuts_directory=item["nuts"],
            topology_role="p0",
            source_revision="a" * 40,
        )
