"""Focused tests for deterministic C1-RQMC bundle certification."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from examples.rjmcmc import conditional_allocation_likelihood_rqmc_certify as certify
from examples.rjmcmc import conditional_allocation_likelihood_rqmc_tiny_screen as rqmc

REVISION = "a" * 40
SHARED = copy.deepcopy(certify._EXPECTED_SHARED_SOBOL_METADATA)
VALIDATE_LIVE_SOURCE = certify._validate_live_source


@pytest.fixture(autouse=True)
def _accept_synthetic_source_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep bundle tests focused while live-Git validation is tested separately."""
    monkeypatch.setattr(certify, "_validate_live_source", lambda *_: None)


def _write_json(path: Path, payload: object) -> None:
    """Write one newline-terminated canonical driver artifact."""
    path.write_text(f"{rqmc.c1._canonical_json(payload)}\n", encoding="ascii")


def _bank(case: tuple[str, str, str]) -> dict[str, Any]:
    """Return the authentic frozen RQMC construction metadata for one case."""
    artifact = certify._expected_artifact(
        case,
        sample_count=rqmc.DEVELOPMENT_SAMPLE_COUNTS[0],
        seed=rqmc.DEVELOPMENT_SELECTION_SEED,
    )
    provenance = rqmc._artifact_construction_provenance(artifact)
    return {
        "construction_method": rqmc.BANK_CONSTRUCTION_METHOD,
        "randomization": "independent_scrambled_sobol_per_source_seed",
        "sample_count_requirement": "power_of_two",
        **provenance,
    }


def _evaluation(
    case: tuple[str, str, str],
    case_bank: dict[str, Any],
    *,
    sample_count: int,
    seed: int,
    passed: bool,
) -> dict[str, Any]:
    """Return one compact evaluation carrying all fields audited by merger."""
    artifact = certify._expected_artifact(
        case,
        sample_count=sample_count,
        seed=seed,
    )
    metrics = {
        name: 0.0 if passed else float(rqmc.THRESHOLDS[name]) + 1.0
        for name in certify._EVALUATION_THRESHOLD_KEYS
    }
    checks = {
        name: bool(metrics[name] <= rqmc.THRESHOLDS[name]) for name in certify._EVALUATION_THRESHOLD_KEYS
    }
    return {
        "artifact_sha256": artifact.sha256,
        "sample_count": sample_count,
        "source_seed": seed,
        "scramble_seed": seed,
        "construction_method": rqmc.BANK_CONSTRUCTION_METHOD,
        "bank_construction_provenance": {
            field: case_bank[field] for field in rqmc._ARTIFACT_PROVENANCE_FIELDS
        },
        "metrics": metrics,
        "checks": checks,
        "scientific_pass_without_repeat_evidence_gate": passed,
        "posterior_summary": {"log_evidence": -1.25},
    }


def _atomic_payload(
    case: tuple[str, str, str],
    *,
    passed: bool,
) -> dict[str, Any]:
    """Return one internally consistent atomic development artifact."""
    case_id = "__".join(case)
    case_bank = _bank(case)
    development_passes = [passed] * len(rqmc.DEVELOPMENT_SAMPLE_COUNTS)
    development = [
        _evaluation(
            case,
            case_bank,
            sample_count=count,
            seed=rqmc.DEVELOPMENT_SELECTION_SEED,
            passed=case_pass,
        )
        for count, case_pass in zip(
            rqmc.DEVELOPMENT_SAMPLE_COUNTS,
            development_passes,
            strict=True,
        )
    ]
    locked = rqmc.DEVELOPMENT_SAMPLE_COUNTS[0] if passed else None
    confirmation = (
        [
            _evaluation(case, case_bank, sample_count=locked, seed=seed, passed=True)
            for seed in rqmc.CONFIRMATION_SEEDS
        ]
        if locked is not None
        else []
    )
    case_payload = {
        "case_id": case_id,
        "profile": "development",
        "regime": case[0],
        "family": case[1],
        "tiling": case[2],
        "bank": case_bank,
        "evidence_merger_group_id": f"{case[0]}__{case[1]}",
        "evidence_merger_thresholds": rqmc.MERGER_THRESHOLDS,
        "development_seed": rqmc.DEVELOPMENT_SELECTION_SEED,
        "confirmation_seeds": list(rqmc.CONFIRMATION_SEEDS),
        "development_evaluations": development,
        "development_pass_pattern": [
            {"sample_count": count, "pass": case_pass}
            for count, case_pass in zip(
                rqmc.DEVELOPMENT_SAMPLE_COUNTS,
                development_passes,
                strict=True,
            )
        ],
        "minimum_passing_suffix_length": 2,
        "confirmation_evaluations": confirmation,
        "locked_sample_count": locked,
        "development_lock_eligible": passed,
        "between_bank_log_evidence_range_nat": 0.0 if passed else None,
        "between_bank_log_evidence_range_pass": passed,
        "confirmation_pass": True if passed else None,
        "scientific_pass": passed,
    }
    report_bank = {
        "construction_method": rqmc.BANK_CONSTRUCTION_METHOD,
        "randomization": "independent_scrambled_sobol_per_source_seed",
        "sample_count_requirement": "power_of_two",
        **SHARED,
        "sobol_catalogue_sha256": {
            case_id: case_bank["sobol_catalogue_sha256"],
        },
        "sobol_block_dimensions": {
            case_id: case_bank["sobol_block_dimensions"],
        },
    }
    return {
        "schema": rqmc.SCHEMA,
        "protocol": rqmc.PROTOCOL,
        "profile": "development",
        "selected_case_id": case_id,
        "per_case_atomic_output": True,
        "source_git_revision": REVISION,
        "driver_sha256": certify._driver_source_sha256(),
        "a1_source_revision": rqmc.A1_SOURCE_REVISION,
        "a1_numerical_source_sha256": rqmc.A1_NUMERICAL_SOURCE_SHA256,
        "a1_definitions_sha256": rqmc.A1_DEFINITIONS_SHA256,
        "required_development_construction_scipy_version": (rqmc.DEVELOPMENT_SCIPY_VERSION),
        "bank": report_bank,
        "protocol_sha256": rqmc._protocol_sha256(
            sample_counts=rqmc.DEVELOPMENT_SAMPLE_COUNTS,
            repeat_seeds=rqmc.DEVELOPMENT_REPEAT_SEEDS,
            matrix=(case,),
        ),
        "frozen_full_development_protocol_sha256": (rqmc.DEVELOPMENT_PROTOCOL_SHA256),
        "sample_counts": list(rqmc.DEVELOPMENT_SAMPLE_COUNTS),
        "repeat_seeds": list(rqmc.DEVELOPMENT_REPEAT_SEEDS),
        "thresholds": rqmc.THRESHOLDS,
        "merger_thresholds": rqmc.MERGER_THRESHOLDS,
        "matrix_catalogue": rqmc.matrix_catalogue(),
        "structural_inference_licensed": False,
        "full_c1_pass": False,
        "independent_evidence_merger_status": "pending_not_implemented",
        "cases": [case_payload],
        "scientific_pass": passed,
    }


def _smoke_payload() -> dict[str, Any]:
    """Return the bounded atomic smoke identity used by preflight."""
    case = rqmc.SMOKE_MATRIX[0]
    case_id = "__".join(case)
    case_bank = _bank(case)
    return {
        "schema": rqmc.SCHEMA,
        "protocol": rqmc.PROTOCOL,
        "profile": "smoke",
        "selected_case_id": case_id,
        "per_case_atomic_output": True,
        "source_git_revision": REVISION,
        "driver_sha256": certify._driver_source_sha256(),
        "a1_source_revision": rqmc.A1_SOURCE_REVISION,
        "a1_numerical_source_sha256": rqmc.A1_NUMERICAL_SOURCE_SHA256,
        "a1_definitions_sha256": rqmc.A1_DEFINITIONS_SHA256,
        "required_development_construction_scipy_version": (rqmc.DEVELOPMENT_SCIPY_VERSION),
        "bank": {
            "construction_method": rqmc.BANK_CONSTRUCTION_METHOD,
            **SHARED,
            "sobol_catalogue_sha256": {
                case_id: case_bank["sobol_catalogue_sha256"],
            },
            "sobol_block_dimensions": {
                case_id: case_bank["sobol_block_dimensions"],
            },
        },
        "sample_counts": [64],
        "repeat_seeds": [731],
        "protocol_sha256": rqmc._protocol_sha256(
            sample_counts=(64,),
            repeat_seeds=(731,),
            matrix=rqmc.SMOKE_MATRIX,
        ),
        "structural_inference_licensed": False,
        "cases": [{"case_id": case_id, "bank": case_bank}],
    }


def _bundle(tmp_path: Path, *, failing_case: int | None = None) -> tuple[Path, Path]:
    """Create one complete synthetic cases/preflight input bundle."""
    cases = tmp_path / "cases"
    preflight = tmp_path / "preflight"
    cases.mkdir()
    preflight.mkdir()
    for index, case in enumerate(rqmc.DEVELOPMENT_MATRIX):
        _write_json(
            cases / f"{'__'.join(case)}.json",
            _atomic_payload(case, passed=index != failing_case),
        )
    _write_json(preflight / "smoke.json", _smoke_payload())
    (preflight / "PREFLIGHT_COMPLETE.txt").write_text(
        f"RQMC C1 preflight complete for {REVISION}\n",
        encoding="ascii",
    )
    (preflight / "preflight.log").write_text(
        (
            f"revision={REVISION}\n"
            f"head={REVISION}\n"
            "status_porcelain_begin\n"
            "status_porcelain_end\n"
            f"scipy={rqmc.DEVELOPMENT_SCIPY_VERSION}\n"
            "focused_pytest_begin\n"
            "focused_pytest_pass\n"
            "focused_ruff_begin\n"
            "focused_ruff_pass\n"
            "focused_pyright_begin\n"
            "focused_pyright_pass\n"
            "smoke_begin\n"
            "smoke_pass\n"
        ),
        encoding="ascii",
    )
    return cases, preflight


@pytest.mark.parametrize(
    ("failing_case", "decision", "pass_count"),
    [(None, "pass", 9), (4, "hard_stop", 8)],
)
def test_certifier_publishes_complete_pass_or_hard_stop_bundle(
    tmp_path: Path,
    failing_case: int | None,
    decision: str,
    pass_count: int,
) -> None:
    """Scientific failure should hard-stop without invalidating execution."""
    cases, preflight = _bundle(tmp_path, failing_case=failing_case)
    output = tmp_path / "report"

    summary = certify.certify(
        source_directory=Path(certify.__file__).resolve().parents[2],
        cases_directory=cases,
        preflight_directory=preflight,
        output_directory=output,
        expected_source_revision=REVISION,
    )

    assert summary["decision"] == decision
    assert summary["execution_certified"] is True
    assert summary["scientific_case_pass_count"] == pass_count
    assert {path.name for path in output.iterdir()} == {
        certify.SUMMARY_FILENAME,
        certify.RESULTS_FILENAME,
        certify.MANIFEST_FILENAME,
        certify.COMPLETE_FILENAME,
    }
    complete = json.loads((output / certify.COMPLETE_FILENAME).read_text())
    manifest = (output / certify.MANIFEST_FILENAME).read_bytes()
    assert complete["decision"] == decision
    assert complete["manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert complete["manifest_entry_count"] == 14


@pytest.mark.parametrize(
    "corruption",
    [
        "extra_case",
        "source_revision",
        "noncanonical",
        "extra_preflight",
        "missing_check",
        "flipped_check",
        "source_seed",
        "artifact_sha256",
    ],
)
def test_certifier_rejects_corrupt_or_extra_inputs(tmp_path: Path, corruption: str) -> None:
    """Immutable input identities and exact directory contracts fail closed."""
    cases, preflight = _bundle(tmp_path)
    first_case = cases / f"{'__'.join(rqmc.DEVELOPMENT_MATRIX[0])}.json"
    if corruption == "extra_case":
        _write_json(cases / "extra.json", {})
    elif corruption == "source_revision":
        payload = json.loads(first_case.read_text())
        payload["source_git_revision"] = "b" * 40
        _write_json(first_case, payload)
    elif corruption == "noncanonical":
        payload = json.loads(first_case.read_text())
        first_case.write_text(json.dumps(payload, indent=2), encoding="ascii")
    elif corruption == "extra_preflight":
        (preflight / "unexpected.txt").write_text("unexpected\n")
    else:
        payload = json.loads(first_case.read_text())
        evaluation = payload["cases"][0]["development_evaluations"][0]
        if corruption == "missing_check":
            evaluation["checks"].pop(next(iter(evaluation["checks"])))
        elif corruption == "flipped_check":
            name = next(iter(evaluation["checks"]))
            evaluation["checks"][name] = not evaluation["checks"][name]
        elif corruption == "source_seed":
            evaluation["source_seed"] += 1
        else:
            evaluation["artifact_sha256"] = "0" * 64
        _write_json(first_case, payload)

    with pytest.raises(ValueError):
        certify.certify(
            source_directory=Path(certify.__file__).resolve().parents[2],
            cases_directory=cases,
            preflight_directory=preflight,
            output_directory=tmp_path / "report",
            expected_source_revision=REVISION,
        )


def test_certifier_refuses_to_overwrite_report(tmp_path: Path) -> None:
    """An existing report path must fail before any input inspection."""
    output = tmp_path / "report"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        certify.certify(
            source_directory=Path(certify.__file__).resolve().parents[2],
            cases_directory=tmp_path / "missing-cases",
            preflight_directory=tmp_path / "missing-preflight",
            output_directory=output,
            expected_source_revision=REVISION,
        )


def test_live_source_validation_requires_imported_clean_pinned_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Certification must bind the imported code to a clean full-SHA checkout."""
    root = Path(certify.__file__).resolve().parents[2]
    observed = {
        ("rev-parse", "HEAD"): f"{REVISION}\n",
        ("status", "--porcelain"): "",
    }
    monkeypatch.setattr(
        certify,
        "_git_output",
        lambda _directory, *arguments: observed[arguments],
    )

    VALIDATE_LIVE_SOURCE(root, REVISION)
    observed[("status", "--porcelain")] = " M changed.py\n"
    with pytest.raises(ValueError, match="not clean"):
        VALIDATE_LIVE_SOURCE(root, REVISION)
    with pytest.raises(ValueError, match="does not contain"):
        VALIDATE_LIVE_SOURCE(root / "elsewhere", REVISION)
