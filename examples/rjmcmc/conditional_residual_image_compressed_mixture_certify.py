#!/usr/bin/env python3
"""Merge exact-mixture source and compression development screens.

This certifier is intentionally separate from the per-case scientific driver.
The source stage selects one sample count common to all six frozen tiny cases.
The compression stage then selects one component count common to all six
cases, after verifying that every shard consumed that authenticated source
lock.  Neither decision licenses structural inference or production output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Sequence, cast

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rjmcmc import (
    conditional_residual_image_compressed_mixture_tiny_screen as screen,
)

SOURCE_DECISION_SCHEMA = "rjmcmc-compressed-mixture-common-source-decision-v1"
COMPRESSION_DECISION_SCHEMA = "rjmcmc-compressed-mixture-common-compression-decision-v1"

Stage = Literal["source", "compression"]


def _canonical_json(payload: object) -> str:
    """Return strict canonical JSON text."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(values: bytes) -> str:
    """Return the lower-case SHA-256 digest of bytes."""
    return hashlib.sha256(values).hexdigest()


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], str]:
    """Read one non-symlink canonical JSON artifact and its raw digest."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular non-symlink file: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"artifact is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must contain one JSON object: {path}")
    if raw != (_canonical_json(payload) + "\n").encode("ascii"):
        raise ValueError(f"artifact is not exact canonical JSON plus one newline: {path}")
    return cast(dict[str, Any], payload), _sha256_bytes(raw)


def _write_atomic_json(path: Path, payload: object) -> None:
    """Publish canonical JSON once without partial or overwritten output."""
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
            stream.write(_canonical_json(payload))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _case_ids() -> tuple[str, ...]:
    """Return the frozen ordered six-case development catalogue."""
    return tuple("__".join(case) for case in screen.DEVELOPMENT_MATRIX)


def _reports(
    directory: Path,
    *,
    expected_revision: str,
    stage: Stage,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Load and authenticate exactly one report for every frozen case."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("report directory must be a real directory")
    paths = sorted(directory.glob("*.json"))
    expected_case_ids = _case_ids()
    if len(paths) != len(expected_case_ids):
        raise ValueError(f"{stage} directory must contain exactly {len(expected_case_ids)} JSON artifacts")
    reports: dict[str, dict[str, Any]] = {}
    raw_sha256: dict[str, str] = {}
    reference_driver_sha256: str | None = None
    reference_protocol_sha256: str | None = None
    for path in paths:
        report, digest = _read_canonical_json(path)
        expected_scalars = {
            "schema": screen.SCHEMA,
            "protocol": screen.PROTOCOL,
            "profile": "development",
            "stage": stage,
            "source_git_revision": expected_revision,
            "protected_catalogue_accessed": False,
            "production_output_written": False,
            "structural_inference_licensed": False,
        }
        for name, expected in expected_scalars.items():
            if report.get(name) != expected:
                raise ValueError(f"{path}: field {name!r} does not match the frozen protocol")
        driver_sha256 = report.get("driver_sha256")
        protocol_sha256 = report.get("protocol_sha256")
        if not isinstance(driver_sha256, str) or len(driver_sha256) != 64:
            raise ValueError(f"{path}: driver_sha256 is malformed")
        if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64:
            raise ValueError(f"{path}: protocol_sha256 is malformed")
        if driver_sha256 != screen._driver_sha256():
            raise ValueError(f"{path}: driver_sha256 does not match this checkout")
        if protocol_sha256 != screen._sha256_json(screen._protocol_payload()):
            raise ValueError(f"{path}: protocol_sha256 does not match this protocol")
        if reference_driver_sha256 is None:
            reference_driver_sha256 = driver_sha256
            reference_protocol_sha256 = protocol_sha256
        elif driver_sha256 != reference_driver_sha256 or protocol_sha256 != reference_protocol_sha256:
            raise ValueError("reports disagree about driver or scientific protocol identity")
        case_id = report.get("selected_case_id")
        case = report.get("case")
        if (
            not isinstance(case_id, str)
            or case_id not in expected_case_ids
            or not isinstance(case, dict)
            or case.get("case_id") != case_id
            or case.get("stage") != stage
        ):
            raise ValueError(f"{path}: selected case identity is invalid")
        if case_id in reports:
            raise ValueError(f"duplicate report for {case_id}")
        reports[case_id] = report
        raw_sha256[case_id] = digest
    if tuple(case_id for case_id in expected_case_ids if case_id in reports) != expected_case_ids:
        raise ValueError("reports do not cover the exact frozen case catalogue")
    return reports, raw_sha256


def _common_suffix_start(
    values: Sequence[int],
    joint_passes: Sequence[bool],
    *,
    minimum_suffix_length: int,
) -> int | None:
    """Return the smallest value beginning a sufficiently long joint suffix."""
    return screen._stable_lock(
        values,
        joint_passes,
        minimum_suffix_length=minimum_suffix_length,
    )


def merge_source(
    *,
    report_directory: Path,
    expected_revision: str,
) -> dict[str, Any]:
    """Merge six source-bank ladders into one authenticated common decision."""
    reports, raw_sha256 = _reports(
        report_directory,
        expected_revision=expected_revision,
        stage="source",
    )
    sample_counts = screen.DEVELOPMENT_SOURCE_SAMPLE_COUNTS
    case_ids = _case_ids()
    per_case: dict[str, dict[int, dict[str, Any]]] = {}
    for case_id in case_ids:
        source_bank = reports[case_id]["case"].get("source_bank")
        if (
            not isinstance(source_bank, dict)
            or source_bank.get("sample_counts") != list(sample_counts)
            or source_bank.get("minimum_common_passing_suffix_length")
            != screen.DEVELOPMENT_MINIMUM_SOURCE_SUFFIX
        ):
            raise ValueError(f"{case_id}: source ladder does not match the protocol")
        evaluations = source_bank.get("evaluations")
        if not isinstance(evaluations, list) or len(evaluations) != len(sample_counts):
            raise ValueError(f"{case_id}: source evaluation ladder is incomplete")
        by_count: dict[int, dict[str, Any]] = {}
        for expected_count, evaluation in zip(sample_counts, evaluations, strict=True):
            if not isinstance(evaluation, dict) or evaluation.get("sample_count") != expected_count:
                raise ValueError(f"{case_id}: source evaluations are not in canonical order")
            certificate = evaluation.get("merger_certificate")
            exact_evaluation = evaluation.get("exact_vs_source")
            if (
                not isinstance(certificate, dict)
                or certificate.get("case_id") != case_id
                or certificate.get("sample_count") != expected_count
                or not isinstance(exact_evaluation, dict)
                or certificate.get("scientific_pass") is not exact_evaluation.get("scientific_pass")
            ):
                raise ValueError(f"{case_id}: source merger certificate is inconsistent")
            by_count[expected_count] = evaluation
        per_case[case_id] = by_count

    joint_passes = [
        all(
            per_case[case_id][sample_count]["exact_vs_source"].get("scientific_pass") is True
            for case_id in case_ids
        )
        for sample_count in sample_counts
    ]
    locked_sample_count = _common_suffix_start(
        sample_counts,
        joint_passes,
        minimum_suffix_length=screen.DEVELOPMENT_MINIMUM_SOURCE_SUFFIX,
    )
    first_report = reports[case_ids[0]]
    common: dict[str, Any] = {
        "schema": SOURCE_DECISION_SCHEMA,
        "protocol": screen.PROTOCOL,
        "protocol_sha256": first_report["protocol_sha256"],
        "source_git_revision": expected_revision,
        "source_driver_sha256": first_report["driver_sha256"],
        "a1_definitions_sha256": screen.c1.A1_DEFINITIONS_SHA256,
        "development_seed": screen.DEVELOPMENT_SEED,
        "source_sample_counts": list(sample_counts),
        "minimum_passing_suffix_length": screen.DEVELOPMENT_MINIMUM_SOURCE_SUFFIX,
        "matrix_case_ids": list(case_ids),
        "joint_scientific_pass_by_sample_count": {
            str(sample_count): passed
            for sample_count, passed in zip(sample_counts, joint_passes, strict=True)
        },
        "source_report_raw_sha256": raw_sha256,
        "locked_sample_count": locked_sample_count,
        "eligible": locked_sample_count is not None,
        "protected_catalogue_accessed": False,
        "production_output_written": False,
        "structural_inference_licensed": False,
    }
    if locked_sample_count is None:
        return common
    common["schema"] = screen.SOURCE_LOCK_SCHEMA
    common["case_certificates"] = {
        case_id: per_case[case_id][locked_sample_count]["merger_certificate"] for case_id in case_ids
    }
    common["source_lock_sha256"] = screen._source_lock_sha256(common)
    return common


def merge_compression(
    *,
    report_directory: Path,
    source_lock_path: Path,
    expected_revision: str,
) -> dict[str, Any]:
    """Merge six compression ladders into one common development decision."""
    reports, raw_sha256 = _reports(
        report_directory,
        expected_revision=expected_revision,
        stage="compression",
    )
    case_ids = _case_ids()
    first_report = reports[case_ids[0]]
    source_lock = screen._load_source_lock(
        source_lock_path,
        source_revision=expected_revision,
        driver_sha256=first_report["driver_sha256"],
    )
    component_counts = screen.DEVELOPMENT_COMPONENT_COUNTS
    per_case: dict[str, dict[int, dict[str, Any]]] = {}
    for case_id in case_ids:
        case = reports[case_id]["case"]
        authenticated = case.get("authenticated_common_source_lock")
        compression = case.get("compression")
        if (
            not isinstance(authenticated, dict)
            or authenticated.get("source_lock_sha256") != source_lock["source_lock_sha256"]
            or authenticated.get("rebuild_certificate_matched") is not True
            or not isinstance(compression, dict)
            or compression.get("component_counts") != list(component_counts)
            or compression.get("minimum_passing_suffix_length")
            != screen.DEVELOPMENT_MINIMUM_COMPRESSION_SUFFIX
        ):
            raise ValueError(f"{case_id}: compression did not consume the frozen source lock")
        evaluations = compression.get("evaluations")
        if not isinstance(evaluations, list) or len(evaluations) != len(component_counts):
            raise ValueError(f"{case_id}: compression ladder is incomplete")
        by_count: dict[int, dict[str, Any]] = {}
        for expected_count, evaluation in zip(component_counts, evaluations, strict=True):
            exact_evaluation = evaluation.get("exact_vs_compressed")
            if (
                not isinstance(evaluation, dict)
                or evaluation.get("component_count") != expected_count
                or not isinstance(exact_evaluation, dict)
            ):
                raise ValueError(f"{case_id}: compression evaluations are not canonical")
            by_count[expected_count] = evaluation
        per_case[case_id] = by_count

    joint_passes = [
        all(
            per_case[case_id][component_count]["exact_vs_compressed"].get("scientific_pass") is True
            for case_id in case_ids
        )
        for component_count in component_counts
    ]
    locked_component_count = _common_suffix_start(
        component_counts,
        joint_passes,
        minimum_suffix_length=screen.DEVELOPMENT_MINIMUM_COMPRESSION_SUFFIX,
    )
    return {
        "schema": COMPRESSION_DECISION_SCHEMA,
        "protocol": screen.PROTOCOL,
        "protocol_sha256": first_report["protocol_sha256"],
        "source_git_revision": expected_revision,
        "source_driver_sha256": first_report["driver_sha256"],
        "source_lock_sha256": source_lock["source_lock_sha256"],
        "locked_source_sample_count": source_lock["locked_sample_count"],
        "development_seed": screen.DEVELOPMENT_SEED,
        "component_counts": list(component_counts),
        "minimum_passing_suffix_length": screen.DEVELOPMENT_MINIMUM_COMPRESSION_SUFFIX,
        "matrix_case_ids": list(case_ids),
        "joint_scientific_pass_by_component_count": {
            str(component_count): passed
            for component_count, passed in zip(component_counts, joint_passes, strict=True)
        },
        "compression_report_raw_sha256": raw_sha256,
        "locked_component_count": locked_component_count,
        "eligible": locked_component_count is not None,
        "confirmation_status": "deferred_to_later_protocol_stage",
        "protected_catalogue_accessed": False,
        "production_output_written": False,
        "structural_inference_licensed": False,
    }


def _revision(value: str) -> str:
    """Validate one complete lower-case Git revision."""
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("revision must be a 40-character lower-case Git SHA")
    return value


def _parser() -> argparse.ArgumentParser:
    """Build the two-stage certification CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("merge-source", "merge-compression"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--report-directory", type=Path, required=True)
        subparser.add_argument("--expected-source-revision", type=_revision, required=True)
        subparser.add_argument("--output", type=Path, required=True)
        if command == "merge-compression":
            subparser.add_argument("--source-lock", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Merge one development stage and publish its decision atomically."""
    args = _parser().parse_args(argv)
    if args.command == "merge-source":
        decision = merge_source(
            report_directory=args.report_directory,
            expected_revision=args.expected_source_revision,
        )
    else:
        decision = merge_compression(
            report_directory=args.report_directory,
            source_lock_path=args.source_lock,
            expected_revision=args.expected_source_revision,
        )
    _write_atomic_json(args.output, decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
