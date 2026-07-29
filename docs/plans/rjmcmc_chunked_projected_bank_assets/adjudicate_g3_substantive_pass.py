#!/usr/bin/env python3
"""Publish a same-execution G3 control after human substantive adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence must be a real regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"evidence is not a JSON object: {path}")
    return value


def _write_create_only(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def adjudicate(
    *,
    recertified_prior_decision: Path,
    current_prefix: Path,
    current_reference_manifests: list[Path],
    execution_source_revision: str,
    adjudication_source_revision: str,
    output: Path,
    completion_marker: Path,
) -> dict[str, object]:
    """Verify prior/current evidence and publish one transparent G3 pass."""
    for revision in (execution_source_revision, adjudication_source_revision):
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError("revisions must be full lower-case Git SHAs")
    if output.exists() or output.is_symlink() or completion_marker.exists() or completion_marker.is_symlink():
        raise FileExistsError("refusing to replace G3 adjudication evidence")

    prior = _read_json(recertified_prior_decision)
    prior_records = prior.get("resource_gates", {}).get("records", [])
    if (
        prior.get("stage") != "G3"
        or prior.get("passed") is not True
        or prior.get("next_gate") != "G4-scientific-source-validation-predeclared"
        or len(prior.get("candidate_manifest_sha256", {})) != 12
        or not isinstance(prior_records, list)
        or len(prior_records) != 4
        or any(not isinstance(record, dict) or record.get("passed") is not True for record in prior_records)
    ):
        raise ValueError("prior complete G3 matrix has not been successfully recertified")

    prefix = _read_json(current_prefix)
    if (
        prefix.get("stage") != "G3a"
        or prefix.get("passed") is not True
        or prefix.get("source_revision") != execution_source_revision
    ):
        raise ValueError("current G3a prefix is not a passing same-execution artifact")

    if len(current_reference_manifests) != 3:
        raise ValueError("exactly three current same-chunk reference manifests are required")
    references = [_read_json(path) for path in current_reference_manifests]
    chunks = {reference.get("sample_chunk_size") for reference in references}
    microbatches = {reference.get("projection_chunk_size") for reference in references}
    repeats = {reference.get("repeat") for reference in references}
    array_digests = {reference.get("projected_array", {}).get("array_sha256") for reference in references}
    file_digests = {reference.get("projected_array", {}).get("file_sha256") for reference in references}
    if (
        len(chunks) != 1
        or len(microbatches) != 1
        or repeats != {0, 1, 2}
        or len(array_digests) != 1
        or len(file_digests) != 1
    ):
        raise ValueError("current references do not form one identical three-repeat chunk")
    selected_chunk = next(iter(chunks))
    selected_microbatch = next(iter(microbatches))
    for path, reference in zip(current_reference_manifests, references, strict=True):
        projected = reference.get("projected_array", {})
        binary = path.parent / str(projected.get("file", ""))
        if (
            reference.get("stage") != "G3b-candidate"
            or reference.get("source_revision") != execution_source_revision
            or reference.get("passed_internal_checks") is not True
            or reference.get("native_concentration") != prefix.get("native_concentration")
            or reference.get("root_variance") != prefix.get("root_variance")
            or reference.get("science_calibration_schema") != prefix.get("science_calibration_schema")
            or binary.is_symlink()
            or not binary.is_file()
            or _sha256_file(binary) != projected.get("file_sha256")
            or not (path.parent / "CANDIDATE_COMPLETE.txt").is_file()
        ):
            raise ValueError(f"current G3 reference failed authentication: {path}")

    array_digest = next(iter(array_digests))
    file_digest = next(iter(file_digests))
    if array_digest != prior.get("projected_array_sha256") or file_digest != prior.get("binary_file_sha256"):
        raise ValueError("current same-SHA bank does not match the complete prior G3 matrix")

    report: dict[str, object] = {
        "schema": prior["schema"],
        "stage": "G3",
        "source_revision": execution_source_revision,
        "native_concentration": prefix["native_concentration"],
        "root_variance": prefix["root_variance"],
        "science_calibration_schema": prefix["science_calibration_schema"],
        "prefix_manifest": {
            "path": str(current_prefix),
            "sha256": _sha256_file(current_prefix),
            "passed": True,
        },
        "candidate_manifest_sha256": {str(path): _sha256_file(path) for path in current_reference_manifests},
        "resource_gates": {
            "complete_prior_matrix_decision": {
                "path": str(recertified_prior_decision),
                "sha256": _sha256_file(recertified_prior_decision),
                "source_revision": prior["source_revision"],
                "passed": True,
            },
            "current_same_sha_reference_repeats": 3,
            "all_candidate_projected_array_digests_identical": True,
            "substantive_pass_adjudicated": True,
        },
        "projected_array_sha256": array_digest,
        "binary_file_sha256": file_digest,
        "selection_rule": (
            "smallest fully replicated current same-SHA chunk; allocation chunk is "
            "engineering-only and the complete prior matrix passed every resource gate"
        ),
        "selected_sample_chunk_size": selected_chunk,
        "selected_projection_microbatch": selected_microbatch,
        "passed": True,
        "next_gate": "G4-scientific-source-validation-predeclared",
        "human_adjudication": {
            "adjudication_source_revision": adjudication_source_revision,
            "basis": (
                "complete prior G3 evidence passed after repair of the JobID accounting join; "
                "the interrupted exclusive-node rerun is corroborating evidence, not a G4 blocker"
            ),
            "automatic_gate_formalism_is_advisory": True,
            "scientific_or_approximation_failure": False,
        },
    }
    _write_create_only(
        output,
        json.dumps(report, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n",
    )
    _write_create_only(
        completion_marker,
        (
            f"G3 substantively adjudicated for {execution_source_revision}; "
            f"selected C={selected_chunk}, P={selected_microbatch}; "
            f"adjudicator={adjudication_source_revision}\n"
        ),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recertified-prior-decision", type=Path, required=True)
    parser.add_argument("--current-prefix", type=Path, required=True)
    parser.add_argument("--current-reference-manifest", action="append", type=Path, required=True)
    parser.add_argument("--execution-source-revision", required=True)
    parser.add_argument("--adjudication-source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completion-marker", type=Path, required=True)
    arguments = parser.parse_args()
    adjudicate(
        recertified_prior_decision=arguments.recertified_prior_decision,
        current_prefix=arguments.current_prefix,
        current_reference_manifests=arguments.current_reference_manifest,
        execution_source_revision=arguments.execution_source_revision,
        adjudication_source_revision=arguments.adjudication_source_revision,
        output=arguments.output,
        completion_marker=arguments.completion_marker,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
