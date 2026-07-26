#!/usr/bin/env python3
"""Certify and merge the nine atomic C1-RQMC development artifacts.

This program performs no scientific calculation.  It validates that an
immutable directory contains exactly the nine source-pinned development
cases produced by ``conditional_allocation_likelihood_rqmc_tiny_screen.py``,
recomputes their lock decisions from the recorded checks, and publishes a
fresh, checksummed report directory.
"""

from __future__ import annotations

import argparse
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

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import conditional_allocation_likelihood_rqmc_tiny_screen as rqmc
from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1

SCHEMA = "rjmcmc-conditional-allocation-c1-rqmc-certification-v1"
CERTIFICATION_PROTOCOL = "conditional-allocation-c1-rqmc-nine-case-certifier-v1"
SUMMARY_FILENAME = "summary.json"
RESULTS_FILENAME = "RESULTS.md"
MANIFEST_FILENAME = "sha256sums.txt"
COMPLETE_FILENAME = "COMPLETE.json"

_FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PER_CASE_SOBOL_FIELDS = frozenset(
    {
        "sobol_catalogue_sha256",
        "sobol_block_dimensions",
    }
)
_SHARED_SOBOL_FIELDS = tuple(
    field for field in rqmc._ARTIFACT_PROVENANCE_FIELDS if field not in _PER_CASE_SOBOL_FIELDS
)
_EXPECTED_SHARED_SOBOL_METADATA: dict[str, Any] = {
    "construction_scipy_version": rqmc.DEVELOPMENT_SCIPY_VERSION,
    "quasi_random_engine": "scipy.stats.qmc.Sobol",
    "sobol_bits": 52,
    "sobol_scramble": True,
    "sobol_optimization": None,
    "inverse_transform": "scipy.special.betaincinv",
    "dimension_order": "stable-id-region-signature/count-balanced-breadth-first",
    "sobol_block_rule": "contiguous-canonical-node-catalogue/max-dimension-21201",
    "sobol_seed_derivation": (
        "sha256(schema-v2,source-seed,node-count,block-index,catalogue-sha256)/little-endian-first-64"
    ),
}
_EVALUATION_THRESHOLD_KEYS = tuple(
    key for key in rqmc.THRESHOLDS if key != "between_bank_log_evidence_range_nat"
)


def _canonical_json(payload: object) -> str:
    """Return the strict canonical JSON representation used by the driver."""
    return c1._canonical_json(payload)


def _sha256_bytes(value: bytes) -> str:
    """Return a lower-case SHA-256 digest."""
    return hashlib.sha256(value).hexdigest()


def _certifier_source_sha256() -> str:
    """Return the exact identity of this certifier source."""
    return _sha256_bytes(Path(__file__).read_bytes())


def _driver_source_sha256() -> str:
    """Return the exact identity of the RQMC scientific driver source."""
    return _sha256_bytes(Path(rqmc.__file__).read_bytes())


def _git_output(source_directory: Path, *arguments: str) -> str:
    """Run one bounded read-only Git query and return its exact stdout."""
    result = subprocess.run(
        ["git", "-C", str(source_directory), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _validate_pixi_environment(source_directory: Path) -> None:
    """Require the worktree environment link to resolve to canonical BP1 pixi."""
    pixi = source_directory / ".pixi"
    expected_pixi = Path("/group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi")
    if not pixi.is_symlink() or pixi.resolve() != expected_pixi:
        raise ValueError("live source .pixi link does not resolve to the canonical BP1 environment")


def _validate_live_source(source_directory: Path, expected_source_revision: str) -> None:
    """Require this imported certifier to reside in the clean pinned worktree."""
    imported_root = Path(__file__).resolve().parents[2]
    if source_directory.resolve() != imported_root:
        raise ValueError("source directory does not contain the imported certifier")
    observed_revision = _git_output(source_directory, "rev-parse", "HEAD").strip()
    if observed_revision != expected_source_revision:
        raise ValueError("live source HEAD does not match the expected revision")
    status = _git_output(source_directory, "status", "--porcelain")
    if status not in ("", "?? .pixi\n"):
        raise ValueError("live source contains changes other than the authenticated .pixi link")
    _validate_pixi_environment(source_directory)


def _certification_protocol_sha256() -> str:
    """Return the identity of the frozen merger/certification protocol."""
    return c1._sha256_json(
        {
            "schema": SCHEMA,
            "certification_protocol": CERTIFICATION_PROTOCOL,
            "scientific_schema": rqmc.SCHEMA,
            "scientific_protocol": rqmc.PROTOCOL,
            "frozen_full_development_protocol_sha256": (rqmc.DEVELOPMENT_PROTOCOL_SHA256),
            "required_scipy_version": rqmc.DEVELOPMENT_SCIPY_VERSION,
            "development_matrix": rqmc.DEVELOPMENT_MATRIX,
            "sample_counts": rqmc.DEVELOPMENT_SAMPLE_COUNTS,
            "repeat_seeds": rqmc.DEVELOPMENT_REPEAT_SEEDS,
            "thresholds": rqmc.THRESHOLDS,
            "merger_thresholds": rqmc.MERGER_THRESHOLDS,
            "construction_method": rqmc.BANK_CONSTRUCTION_METHOD,
            "shared_sobol_fields": _SHARED_SOBOL_FIELDS,
            "per_case_sobol_fields": sorted(_PER_CASE_SOBOL_FIELDS),
        }
    )


def _expected_case_ids() -> tuple[str, ...]:
    """Return the source-pinned case IDs in their declared matrix order."""
    return tuple("__".join(case) for case in rqmc.DEVELOPMENT_MATRIX)


def _require_equal(observed: object, expected: object, label: str) -> None:
    """Fail with a useful label unless two canonical JSON values agree."""
    if _canonical_json(observed) != _canonical_json(expected):
        raise ValueError(f"{label} does not match the frozen protocol")


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    """Return a string-keyed mapping or reject the artifact."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _require_list(value: object, label: str) -> list[Any]:
    """Return a list or reject the artifact."""
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return cast(list[Any], value)


def _read_canonical_json(path: Path) -> dict[str, Any]:
    """Read one canonical JSON object, rejecting duplicate keys and drift."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    raw = path.read_bytes()
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
    return cast(dict[str, Any], payload)


def _regular_files(directory: Path, *, label: str) -> tuple[Path, ...]:
    """Return sorted, relative regular files under one immutable directory."""
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"{label} must be a real directory")
    paths: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"{label} contains a symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"{label} contains a non-regular entry: {path}")
        paths.append(path.relative_to(directory))
    if not paths:
        raise ValueError(f"{label} contains no regular files")
    return tuple(paths)


def _validate_preflight(
    directory: Path,
    *,
    expected_source_revision: str,
    expected_driver_sha256: str,
) -> tuple[Path, ...]:
    """Validate the committed preflight contract and return its three files."""
    paths = _regular_files(directory, label="preflight directory")
    expected = {
        Path("preflight.log"),
        Path("smoke.json"),
        Path("PREFLIGHT_COMPLETE.txt"),
    }
    if set(paths) != expected:
        raise ValueError(
            "preflight directory must contain exactly preflight.log, smoke.json, and PREFLIGHT_COMPLETE.txt"
        )
    marker = (directory / "PREFLIGHT_COMPLETE.txt").read_text(encoding="ascii")
    expected_marker = f"RQMC C1 preflight complete for {expected_source_revision}\n"
    if marker != expected_marker:
        raise ValueError("preflight completion marker does not match the expected revision")
    log = (directory / "preflight.log").read_text(encoding="utf-8")
    allowed_status_blocks = (
        "status_porcelain_begin\nstatus_porcelain_end\n",
        "status_porcelain_begin\n?? .pixi\nstatus_porcelain_end\n",
    )
    if not any(block in log for block in allowed_status_blocks):
        raise ValueError("preflight log omits the authenticated clean-source status")
    required_log_fragments = (
        f"revision={expected_source_revision}\n",
        f"head={expected_source_revision}\n",
        f"scipy={rqmc.DEVELOPMENT_SCIPY_VERSION}\n",
        "focused_pytest_begin\n",
        "focused_pytest_pass\n",
        "focused_ruff_begin\n",
        "focused_ruff_pass\n",
        "focused_pyright_begin\n",
        "focused_pyright_pass\n",
        "smoke_begin\n",
        "smoke_pass\n",
    )
    if any(fragment not in log for fragment in required_log_fragments):
        raise ValueError("preflight log omits a required source, environment, or gate identity")

    smoke = _read_canonical_json(directory / "smoke.json")
    smoke_case_id = "__".join(rqmc.SMOKE_MATRIX[0])
    required_smoke = {
        "schema": rqmc.SCHEMA,
        "protocol": rqmc.PROTOCOL,
        "profile": "smoke",
        "selected_case_id": smoke_case_id,
        "per_case_atomic_output": True,
        "source_git_revision": expected_source_revision,
        "driver_sha256": expected_driver_sha256,
        "a1_source_revision": rqmc.A1_SOURCE_REVISION,
        "a1_numerical_source_sha256": rqmc.A1_NUMERICAL_SOURCE_SHA256,
        "a1_definitions_sha256": rqmc.A1_DEFINITIONS_SHA256,
        "required_development_construction_scipy_version": (rqmc.DEVELOPMENT_SCIPY_VERSION),
        "sample_counts": [64],
        "repeat_seeds": [731],
        "protocol_sha256": rqmc._protocol_sha256(
            sample_counts=(64,),
            repeat_seeds=(731,),
            matrix=rqmc.SMOKE_MATRIX,
        ),
        "structural_inference_licensed": False,
    }
    for field, expected_value in required_smoke.items():
        _require_equal(smoke.get(field), expected_value, f"preflight smoke {field}")
    cases = _require_list(smoke.get("cases"), "preflight smoke cases")
    if len(cases) != 1:
        raise ValueError("preflight smoke must contain exactly one case")
    smoke_case = _require_mapping(cases[0], "preflight smoke case")
    if smoke_case.get("case_id") != smoke_case_id:
        raise ValueError("preflight smoke contains the wrong case")
    _validate_sobol_metadata(
        smoke,
        smoke_case,
        case_id=smoke_case_id,
        shared_metadata=None,
    )
    return paths


def _validate_sobol_metadata(
    report: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    case_id: str,
    shared_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate report-, case-, and evaluation-level construction metadata."""
    report_bank = _require_mapping(report.get("bank"), f"{case_id} report bank")
    case_bank = _require_mapping(case.get("bank"), f"{case_id} case bank")
    if report_bank.get("construction_method") != rqmc.BANK_CONSTRUCTION_METHOD:
        raise ValueError(f"{case_id} has the wrong bank construction method")
    if case_bank.get("construction_method") != rqmc.BANK_CONSTRUCTION_METHOD:
        raise ValueError(f"{case_id} case has the wrong bank construction method")

    observed_shared: dict[str, Any] = {}
    for field in _SHARED_SOBOL_FIELDS:
        if field not in report_bank or field not in case_bank:
            raise ValueError(f"{case_id} omits shared Sobol metadata {field}")
        _require_equal(
            report_bank[field],
            case_bank[field],
            f"{case_id} report/case Sobol metadata {field}",
        )
        observed_shared[field] = report_bank[field]
    if shared_metadata is not None:
        _require_equal(observed_shared, shared_metadata, f"{case_id} shared Sobol metadata")
    _require_equal(
        observed_shared,
        _EXPECTED_SHARED_SOBOL_METADATA,
        f"{case_id} frozen shared Sobol metadata",
    )

    observed_scipy = observed_shared.get("construction_scipy_version")
    if observed_scipy != rqmc.DEVELOPMENT_SCIPY_VERSION:
        raise ValueError(
            f"{case_id} was constructed with SciPy {observed_scipy!r}, not {rqmc.DEVELOPMENT_SCIPY_VERSION}"
        )
    catalogue_map = _require_mapping(
        report_bank.get("sobol_catalogue_sha256"),
        f"{case_id} Sobol catalogue map",
    )
    block_map = _require_mapping(
        report_bank.get("sobol_block_dimensions"),
        f"{case_id} Sobol block map",
    )
    if set(catalogue_map) != {case_id} or set(block_map) != {case_id}:
        raise ValueError(f"{case_id} report Sobol maps must contain exactly that case")
    catalogue = case_bank.get("sobol_catalogue_sha256")
    blocks = case_bank.get("sobol_block_dimensions")
    if not isinstance(catalogue, str) or _SHA256_RE.fullmatch(catalogue) is None:
        raise ValueError(f"{case_id} has an invalid Sobol catalogue identity")
    if not isinstance(blocks, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 21_201 for value in blocks
    ):
        raise ValueError(f"{case_id} has invalid Sobol block dimensions")
    _require_equal(catalogue_map[case_id], catalogue, f"{case_id} Sobol catalogue")
    _require_equal(block_map[case_id], blocks, f"{case_id} Sobol blocks")
    return observed_shared


def _evaluation_pass(evaluation: Mapping[str, Any], label: str) -> bool:
    """Recompute every recorded scientific check from its finite metric."""
    metrics = _require_mapping(evaluation.get("metrics"), f"{label} metrics")
    checks = _require_mapping(evaluation.get("checks"), f"{label} checks")
    expected_keys = set(_EVALUATION_THRESHOLD_KEYS)
    if set(metrics) != expected_keys or set(checks) != expected_keys:
        raise ValueError(f"{label} metrics and checks must contain every frozen evaluation gate")
    recomputed: dict[str, bool] = {}
    for name in _EVALUATION_THRESHOLD_KEYS:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{label} metric {name} must be a finite number")
        if not isinstance(checks[name], bool):
            raise ValueError(f"{label} check {name} must be Boolean")
        recomputed[name] = float(value) <= rqmc.THRESHOLDS[name]
        if checks[name] is not recomputed[name]:
            raise ValueError(f"{label} check {name} disagrees with its metric and threshold")
    expected = all(recomputed.values())
    if evaluation.get("scientific_pass_without_repeat_evidence_gate") is not expected:
        raise ValueError(f"{label} scientific pass disagrees with its checks")
    return expected


def _expected_artifact(
    expected_case: tuple[str, str, str],
    *,
    sample_count: int,
    seed: int,
) -> Any:
    """Rebuild one deterministic RQMC bank solely to authenticate its identity."""
    regime_name, family, tiling = expected_case
    typed_family = cast(c1.Family, family)
    typed_tiling = cast(c1.Tiling, tiling)
    regime = c1._regime(regime_name)
    shapes, _, design, _, noise = c1._case_arrays(regime, typed_family)
    labels = c1.labels_for_tiling(typed_family, typed_tiling)
    aggregation = rqmc.AdditiveDirichletAggregation(
        shapes,
        design,
        noise,
        np.eye(noise.size, dtype=np.float64),
    )
    return rqmc._build_artifact(
        aggregation,
        labels,
        sample_count=sample_count,
        seed=seed,
        case_id="__".join(expected_case),
    )


def _validate_evaluation_provenance(
    evaluation: Mapping[str, Any],
    *,
    expected_case: tuple[str, str, str],
    case_id: str,
    case_bank: Mapping[str, Any],
    sample_count: int,
    seed: int,
    label: str,
) -> None:
    """Validate one bank evaluation against its rebuilt deterministic artifact."""
    if evaluation.get("construction_method") != rqmc.BANK_CONSTRUCTION_METHOD:
        raise ValueError(f"{label} has the wrong construction method")
    if evaluation.get("sample_count") != sample_count:
        raise ValueError(f"{label} has the wrong sample count")
    if evaluation.get("source_seed") != seed or evaluation.get("scramble_seed") != seed:
        raise ValueError(f"{label} has the wrong authoritative RQMC seed")
    provenance = _require_mapping(
        evaluation.get("bank_construction_provenance"),
        f"{label} construction provenance",
    )
    expected = {field: case_bank[field] for field in rqmc._ARTIFACT_PROVENANCE_FIELDS}
    _require_equal(provenance, expected, f"{case_id} evaluation construction provenance")
    artifact = _expected_artifact(
        expected_case,
        sample_count=sample_count,
        seed=seed,
    )
    observed_sha = evaluation.get("artifact_sha256")
    if (
        not isinstance(observed_sha, str)
        or _SHA256_RE.fullmatch(observed_sha) is None
        or observed_sha != artifact.sha256
    ):
        raise ValueError(f"{label} has the wrong deterministic artifact identity")
    _require_equal(
        provenance,
        rqmc._artifact_construction_provenance(artifact),
        f"{label} rebuilt artifact provenance",
    )


def _validate_case_decision(
    report: Mapping[str, Any],
    case: Mapping[str, Any],
    expected_case: tuple[str, str, str],
) -> dict[str, Any]:
    """Recompute one case lock, confirmation, and evidence-range decision."""
    case_id = "__".join(expected_case)
    case_bank = _require_mapping(case.get("bank"), f"{case_id} case bank")
    development = _require_list(
        case.get("development_evaluations"),
        f"{case_id} development evaluations",
    )
    if len(development) != len(rqmc.DEVELOPMENT_SAMPLE_COUNTS):
        raise ValueError(f"{case_id} has the wrong number of development evaluations")
    development_passes: list[bool] = []
    for sample_count, raw in zip(rqmc.DEVELOPMENT_SAMPLE_COUNTS, development, strict=True):
        evaluation = _require_mapping(raw, f"{case_id} development S={sample_count}")
        if evaluation.get("sample_count") != sample_count:
            raise ValueError(f"{case_id} development sample ladder changed")
        if evaluation.get("scramble_seed") != rqmc.DEVELOPMENT_SELECTION_SEED:
            raise ValueError(f"{case_id} development scramble seed changed")
        _validate_evaluation_provenance(
            evaluation,
            expected_case=expected_case,
            case_id=case_id,
            case_bank=case_bank,
            sample_count=sample_count,
            seed=rqmc.DEVELOPMENT_SELECTION_SEED,
            label=f"{case_id} development S={sample_count}",
        )
        development_passes.append(_evaluation_pass(evaluation, f"{case_id} development S={sample_count}"))
    expected_pattern = [
        {"sample_count": count, "pass": passed}
        for count, passed in zip(rqmc.DEVELOPMENT_SAMPLE_COUNTS, development_passes, strict=True)
    ]
    _require_equal(
        case.get("development_pass_pattern"),
        expected_pattern,
        f"{case_id} development pass pattern",
    )
    if case.get("minimum_passing_suffix_length") != 2:
        raise ValueError(f"{case_id} minimum passing suffix length changed")
    locked = c1._stable_lock_sample_count(
        rqmc.DEVELOPMENT_SAMPLE_COUNTS,
        development_passes,
        minimum_suffix_length=2,
    )
    if case.get("locked_sample_count") != locked:
        raise ValueError(f"{case_id} recorded the wrong locked sample count")
    if case.get("development_lock_eligible") is not (locked is not None):
        raise ValueError(f"{case_id} lock eligibility disagrees with its development suffix")
    if case.get("development_seed") != rqmc.DEVELOPMENT_SELECTION_SEED:
        raise ValueError(f"{case_id} recorded the wrong development seed")
    _require_equal(
        case.get("confirmation_seeds"),
        list(rqmc.CONFIRMATION_SEEDS),
        f"{case_id} confirmation seed catalogue",
    )

    confirmation = _require_list(
        case.get("confirmation_evaluations"),
        f"{case_id} confirmation evaluations",
    )
    expected_confirmation_count = len(rqmc.CONFIRMATION_SEEDS) if locked is not None else 0
    if len(confirmation) != expected_confirmation_count:
        raise ValueError(f"{case_id} has incomplete or unexpected confirmation evaluations")
    confirmation_checks: list[dict[str, Any]] = []
    for seed, raw in zip(rqmc.CONFIRMATION_SEEDS, confirmation):
        evaluation = _require_mapping(raw, f"{case_id} confirmation seed={seed}")
        if evaluation.get("sample_count") != locked or evaluation.get("scramble_seed") != seed:
            raise ValueError(f"{case_id} confirmation bank identity changed")
        _validate_evaluation_provenance(
            evaluation,
            expected_case=expected_case,
            case_id=case_id,
            case_bank=case_bank,
            sample_count=cast(int, locked),
            seed=seed,
            label=f"{case_id} confirmation seed={seed}",
        )
        confirmation_checks.append(
            {
                "seed": seed,
                "pass": _evaluation_pass(evaluation, f"{case_id} confirmation seed={seed}"),
            }
        )

    locked_evaluations: list[Mapping[str, Any]] = []
    if locked is not None:
        locked_index = rqmc.DEVELOPMENT_SAMPLE_COUNTS.index(locked)
        locked_evaluations = [
            _require_mapping(development[locked_index], f"{case_id} locked development"),
            *[_require_mapping(raw, f"{case_id} confirmation") for raw in confirmation],
        ]
    evidence_values: list[float] = []
    for index, evaluation in enumerate(locked_evaluations):
        posterior = _require_mapping(
            evaluation.get("posterior_summary"),
            f"{case_id} locked posterior summary {index}",
        )
        value = posterior.get("log_evidence")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{case_id} has an invalid locked log evidence")
        evidence_values.append(float(value))
    evidence_range = float(max(evidence_values) - min(evidence_values)) if evidence_values else None
    recorded_range = case.get("between_bank_log_evidence_range_nat")
    if evidence_range is None:
        if recorded_range is not None:
            raise ValueError(f"{case_id} recorded evidence range without a lock")
    elif not isinstance(recorded_range, (int, float)) or float(recorded_range) != evidence_range:
        raise ValueError(f"{case_id} recorded the wrong evidence range")
    evidence_pass = (
        evidence_range is not None
        and evidence_range <= rqmc.THRESHOLDS["between_bank_log_evidence_range_nat"]
    )
    if case.get("between_bank_log_evidence_range_pass") is not evidence_pass:
        raise ValueError(f"{case_id} evidence-range decision is inconsistent")
    confirmation_pass: bool | None
    if confirmation:
        confirmation_pass = all(item["pass"] for item in confirmation_checks) and evidence_pass
    else:
        confirmation_pass = None
    if case.get("confirmation_pass") is not confirmation_pass:
        raise ValueError(f"{case_id} confirmation decision is inconsistent")
    scientific_pass = locked is not None and confirmation_pass is True and evidence_pass
    if case.get("scientific_pass") is not scientific_pass:
        raise ValueError(f"{case_id} scientific decision is inconsistent")
    if report.get("scientific_pass") is not scientific_pass:
        raise ValueError(f"{case_id} report and case scientific decisions disagree")

    suffix = (
        list(rqmc.DEVELOPMENT_SAMPLE_COUNTS[rqmc.DEVELOPMENT_SAMPLE_COUNTS.index(locked) :]) if locked else []
    )
    return {
        "case_id": case_id,
        "locked_sample_count": locked,
        "development_pass_pattern": expected_pattern,
        "development_passing_suffix": suffix,
        "confirmation_checks": confirmation_checks,
        "between_bank_log_evidence_range_nat": evidence_range,
        "scientific_pass": scientific_pass,
    }


def _validate_atomic_case(
    payload: Mapping[str, Any],
    *,
    expected_case: tuple[str, str, str],
    expected_source_revision: str,
    expected_driver_sha256: str,
    shared_sobol_metadata: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one source-pinned, one-case development artifact."""
    case_id = "__".join(expected_case)
    required_top_level = {
        "schema": rqmc.SCHEMA,
        "protocol": rqmc.PROTOCOL,
        "profile": "development",
        "selected_case_id": case_id,
        "per_case_atomic_output": True,
        "source_git_revision": expected_source_revision,
        "driver_sha256": expected_driver_sha256,
        "a1_source_revision": rqmc.A1_SOURCE_REVISION,
        "a1_numerical_source_sha256": rqmc.A1_NUMERICAL_SOURCE_SHA256,
        "a1_definitions_sha256": rqmc.A1_DEFINITIONS_SHA256,
        "required_development_construction_scipy_version": (rqmc.DEVELOPMENT_SCIPY_VERSION),
        "protocol_sha256": rqmc._protocol_sha256(
            sample_counts=rqmc.DEVELOPMENT_SAMPLE_COUNTS,
            repeat_seeds=rqmc.DEVELOPMENT_REPEAT_SEEDS,
            matrix=(expected_case,),
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
    }
    for field, expected in required_top_level.items():
        _require_equal(payload.get(field), expected, f"{case_id} {field}")
    cases = _require_list(payload.get("cases"), f"{case_id} cases")
    if len(cases) != 1:
        raise ValueError(f"{case_id} atomic artifact must contain exactly one case")
    case = _require_mapping(cases[0], f"{case_id} case")
    required_case_identity = {
        "case_id": case_id,
        "profile": "development",
        "regime": expected_case[0],
        "family": expected_case[1],
        "tiling": expected_case[2],
        "evidence_merger_group_id": f"{expected_case[0]}__{expected_case[1]}",
        "evidence_merger_thresholds": rqmc.MERGER_THRESHOLDS,
    }
    for field, expected in required_case_identity.items():
        _require_equal(case.get(field), expected, f"{case_id} case {field}")
    observed_shared = _validate_sobol_metadata(
        payload,
        case,
        case_id=case_id,
        shared_metadata=shared_sobol_metadata,
    )
    return _validate_case_decision(payload, case, expected_case), observed_shared


def _manifest_entries(
    *,
    cases_directory: Path,
    case_paths: Sequence[Path],
    preflight_directory: Path,
    preflight_paths: Sequence[Path],
    report_directory: Path,
) -> list[tuple[str, str]]:
    """Return deterministic SHA-256 entries for inputs and report products."""
    entries: list[tuple[str, str]] = []
    for relative in case_paths:
        entries.append(
            (
                _sha256_bytes((cases_directory / relative).read_bytes()),
                f"cases/{relative.as_posix()}",
            )
        )
    for relative in preflight_paths:
        entries.append(
            (
                _sha256_bytes((preflight_directory / relative).read_bytes()),
                f"preflight/{relative.as_posix()}",
            )
        )
    for name in (SUMMARY_FILENAME, RESULTS_FILENAME):
        entries.append(
            (
                _sha256_bytes((report_directory / name).read_bytes()),
                f"report/{name}",
            )
        )
    names = [name for _, name in entries]
    if len(names) != len(set(names)):
        raise RuntimeError("manifest logical paths are not unique")
    return entries


def _results_markdown(summary: Mapping[str, Any]) -> str:
    """Render the compact human-readable certification decision."""
    lines = [
        "# C1-RQMC development certification",
        "",
        f"- Decision: **{summary['decision']}**",
        f"- Certified execution: **{str(summary['execution_certified']).lower()}**",
        f"- Scientifically passing cases: **{summary['scientific_case_pass_count']}/9**",
        f"- Source revision: `{summary['source_git_revision']}`",
        "",
        "| Case | Locked S | Development suffix | Confirmation | Evidence range (nat) | Pass |",
        "|---|---:|---|---|---:|---:|",
    ]
    for case in cast(list[dict[str, Any]], summary["cases"]):
        suffix = ", ".join(str(value) for value in case["development_passing_suffix"]) or "none"
        confirmation = (
            ", ".join(
                f"{item['seed']}:{'pass' if item['pass'] else 'fail'}" for item in case["confirmation_checks"]
            )
            or "not run"
        )
        evidence = case["between_bank_log_evidence_range_nat"]
        evidence_text = "n/a" if evidence is None else f"{evidence:.9g}"
        lines.append(
            f"| `{case['case_id']}` | {case['locked_sample_count'] or 'none'} | "
            f"{suffix} | {confirmation} | {evidence_text} | "
            f"{'yes' if case['scientific_pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "A `hard_stop` decision certifies a complete, internally consistent execution "
            "whose scientific gates did not all pass. It is not an infrastructure failure.",
            "",
        ]
    )
    return "\n".join(lines)


def certify(
    *,
    source_directory: Path,
    cases_directory: Path,
    preflight_directory: Path,
    output_directory: Path,
    expected_source_revision: str,
) -> dict[str, Any]:
    """Validate all inputs and atomically publish one fresh report directory."""
    if _FULL_SHA_RE.fullmatch(expected_source_revision) is None:
        raise ValueError("expected source revision must be a full lower-case 40-hex Git SHA")
    _validate_live_source(source_directory, expected_source_revision)
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"refusing to replace existing report directory: {output_directory}")

    expected_ids = _expected_case_ids()
    expected_names = {f"{case_id}.json" for case_id in expected_ids}
    case_paths = _regular_files(cases_directory, label="cases directory")
    observed_names = {path.as_posix() for path in case_paths}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        extra = sorted(observed_names - expected_names)
        raise ValueError(
            f"cases directory does not contain exactly the frozen nine cases; missing={missing}, extra={extra}"
        )
    expected_driver_sha256 = _driver_source_sha256()
    preflight_paths = _validate_preflight(
        preflight_directory,
        expected_source_revision=expected_source_revision,
        expected_driver_sha256=expected_driver_sha256,
    )
    summaries: list[dict[str, Any]] = []
    shared_sobol_metadata: dict[str, Any] | None = None
    for expected_case in rqmc.DEVELOPMENT_MATRIX:
        case_id = "__".join(expected_case)
        payload = _read_canonical_json(cases_directory / f"{case_id}.json")
        case_summary, observed_shared = _validate_atomic_case(
            payload,
            expected_case=expected_case,
            expected_source_revision=expected_source_revision,
            expected_driver_sha256=expected_driver_sha256,
            shared_sobol_metadata=shared_sobol_metadata,
        )
        if shared_sobol_metadata is None:
            shared_sobol_metadata = observed_shared
        summaries.append(case_summary)

    scientific_pass_count = sum(case["scientific_pass"] for case in summaries)
    decision = "pass" if scientific_pass_count == len(summaries) else "hard_stop"
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "certification_protocol": CERTIFICATION_PROTOCOL,
        "certification_protocol_sha256": _certification_protocol_sha256(),
        "certifier_source_sha256": _certifier_source_sha256(),
        "scientific_driver_sha256": expected_driver_sha256,
        "source_git_revision": expected_source_revision,
        "frozen_full_development_protocol_sha256": (rqmc.DEVELOPMENT_PROTOCOL_SHA256),
        "required_and_observed_scipy_version": rqmc.DEVELOPMENT_SCIPY_VERSION,
        "execution_certified": True,
        "decision": decision,
        "decision_semantics": (
            "pass only when all nine cases pass; hard_stop still certifies complete execution"
        ),
        "scientific_case_pass_count": scientific_pass_count,
        "scientific_case_count": len(summaries),
        "structural_inference_licensed": False,
        "shared_sobol_metadata": shared_sobol_metadata,
        "cases": summaries,
    }

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.",
            dir=output_directory.parent,
        )
    )
    published = False
    try:
        (temporary / SUMMARY_FILENAME).write_text(
            f"{_canonical_json(summary)}\n",
            encoding="ascii",
        )
        (temporary / RESULTS_FILENAME).write_text(
            _results_markdown(summary),
            encoding="utf-8",
        )
        manifest_entries = _manifest_entries(
            cases_directory=cases_directory,
            case_paths=case_paths,
            preflight_directory=preflight_directory,
            preflight_paths=preflight_paths,
            report_directory=temporary,
        )
        manifest_text = "".join(f"{digest}  {name}\n" for digest, name in manifest_entries)
        (temporary / MANIFEST_FILENAME).write_text(manifest_text, encoding="ascii")
        complete = {
            "schema": "rjmcmc-conditional-allocation-c1-rqmc-complete-v1",
            "source_git_revision": expected_source_revision,
            "decision": decision,
            "execution_certified": True,
            "manifest_filename": MANIFEST_FILENAME,
            "manifest_sha256": _sha256_bytes(manifest_text.encode("ascii")),
            "manifest_entry_count": len(manifest_entries),
            "certification_protocol_sha256": summary["certification_protocol_sha256"],
            "certifier_source_sha256": summary["certifier_source_sha256"],
        }
        # COMPLETE is deliberately the last file written before atomic publish.
        (temporary / COMPLETE_FILENAME).write_text(
            f"{_canonical_json(complete)}\n",
            encoding="ascii",
        )
        if output_directory.exists():
            raise FileExistsError(f"refusing to replace existing report directory: {output_directory}")
        os.rename(temporary, output_directory)
        published = True
    finally:
        if not published and temporary.exists():
            for path in sorted(temporary.iterdir(), reverse=True):
                path.unlink()
            temporary.rmdir()
    return summary


def _parser() -> argparse.ArgumentParser:
    """Build the strict certification CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", "--cases-directory", dest="cases_directory", type=Path, required=True)
    parser.add_argument(
        "--source-dir", "--source-directory", dest="source_directory", type=Path, required=True
    )
    parser.add_argument(
        "--preflight-dir",
        "--preflight-directory",
        dest="preflight_directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir", "--output-directory", dest="output_directory", type=Path, required=True
    )
    parser.add_argument("--expected-source-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the frozen bundle and publish the certification report."""
    args = _parser().parse_args(argv)
    certify(
        source_directory=args.source_directory,
        cases_directory=args.cases_directory,
        preflight_directory=args.preflight_directory,
        output_directory=args.output_directory,
        expected_source_revision=args.expected_source_revision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
