#!/usr/bin/env python3
"""Certify one completed synthetic local-versus-NUTS reference comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping, Sequence

from openghg_inversions.experimental.rjmcmc.mh_local_search_conditional_reference import (
    certify_conditional_reference,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_synthetic import (
    canonical_json,
    file_sha256,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_PAYLOAD_FILENAMES = ("conditional_reference.json", "audit.json")


def _current_clean_revision() -> str:
    revision = subprocess.run(
        ("git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "-C", str(_REPOSITORY_ROOT), "status", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if _FULL_SHA.fullmatch(revision) is None or status:
        raise RuntimeError("certificate publication requires a clean exact source revision")
    return revision


def _create_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(dict(payload)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _strict_json_object(path: Path) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"{path} contains duplicate JSON key {name!r}")
            result[name] = value
        return result

    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {token}")),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(f"{path} is not strict JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one JSON object")
    return value


def _audit_staged_outputs(
    directory: Path,
    *,
    expected_record: Mapping[str, object],
    expected_audit: Mapping[str, object],
    first_pass_hashes: Mapping[str, str],
) -> dict[str, str]:
    """Independently reopen and rehash staged payloads before completion."""
    if (directory / "complete.json").exists():
        raise RuntimeError("conditional-reference completion was written before final audit")
    expected_payloads = {
        "conditional_reference.json": dict(expected_record),
        "audit.json": dict(expected_audit),
    }
    if set(first_pass_hashes) != set(_PAYLOAD_FILENAMES):
        raise RuntimeError("conditional-reference first-pass catalogue is incompatible")
    for name, expected in expected_payloads.items():
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"conditional-reference staged payload is not a regular file: {name}")
        reopened = _strict_json_object(path)
        if canonical_json(reopened) != canonical_json(expected):
            raise RuntimeError(f"conditional-reference staged payload changed semantically: {name}")
        if path.read_text(encoding="utf-8") != canonical_json(expected) + "\n":
            raise RuntimeError(f"conditional-reference staged payload is not exact canonical JSON: {name}")
    second_pass_hashes = {name: file_sha256(directory / name) for name in _PAYLOAD_FILENAMES}
    if second_pass_hashes != dict(first_pass_hashes):
        raise RuntimeError("conditional-reference payload changed between checksum passes")
    return second_pass_hashes


def run(
    arguments: argparse.Namespace,
    *,
    enforce_clean_revision: bool = True,
) -> dict[str, object]:
    if arguments.output_directory.exists():
        raise FileExistsError(f"output path already exists: {arguments.output_directory}")
    if not arguments.output_directory.parent.is_dir():
        raise FileNotFoundError("output-directory parent does not exist")
    certificate = certify_conditional_reference(
        training_path=arguments.training,
        evaluation_path=arguments.evaluation,
        nuts_directory=arguments.nuts_directory,
        local_directory=arguments.local_directory,
        _test_short_budget=getattr(arguments, "_test_short_budget", None),
    )
    source_revision = certificate.audit["source_revision"]
    if enforce_clean_revision:
        current = _current_clean_revision()
        if arguments.source_revision != current or source_revision != current:
            raise ValueError("--source-revision, raw sampling revision, and current revision must match")
    elif arguments.source_revision != source_revision:
        raise ValueError("--source-revision must match the raw sampling revision")

    arguments.output_directory.mkdir()
    _create_json(
        arguments.output_directory / "conditional_reference.json",
        certificate.record,
    )
    _create_json(arguments.output_directory / "audit.json", certificate.audit)
    first_pass_hashes = {name: file_sha256(arguments.output_directory / name) for name in _PAYLOAD_FILENAMES}
    files = _audit_staged_outputs(
        arguments.output_directory,
        expected_record=certificate.record,
        expected_audit=certificate.audit,
        first_pass_hashes=first_pass_hashes,
    )
    _create_json(
        arguments.output_directory / "complete.json",
        {
            "schema": "openghg_inversions.mh_local_search_conditional_reference_completion.v1",
            "status": "complete",
            "pass": certificate.record["pass"],
            "first_failed_gate": certificate.record["first_failed_gate"],
            "files": files,
        },
    )
    return dict(certificate.record)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--training", type=Path, required=True)
    result.add_argument("--evaluation", type=Path, required=True)
    result.add_argument("--nuts-directory", type=Path, required=True)
    result.add_argument("--local-directory", type=Path, required=True)
    result.add_argument("--output-directory", type=Path, required=True)
    result.add_argument("--source-revision", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
