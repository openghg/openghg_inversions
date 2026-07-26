#!/usr/bin/env python3
"""Strict checksum and create-only status primitives for the S0 harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence, cast

_DIGEST = re.compile(r"[0-9a-f]{64}")


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def strict_json(path: Path) -> dict[str, object]:
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(dict(payload)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def audit_completion(path: Path, *, _active: set[Path] | None = None) -> str:
    if path.is_symlink():
        raise ValueError(f"completion must not be a symlink: {path}")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"completion is not a regular file: {path}")
    active = set() if _active is None else _active
    if path in active:
        raise ValueError(f"completion graph contains a cycle: {path}")
    active.add(path)
    completion = strict_json(path)
    files = completion.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"completion has no exact files catalogue: {path}")
    first_pass: dict[str, str] = {}
    for raw_name, raw_digest in cast(dict[str, object], files).items():
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or Path(raw_name).is_absolute()
            or ".." in Path(raw_name).parts
            or not isinstance(raw_digest, str)
            or _DIGEST.fullmatch(raw_digest) is None
        ):
            raise ValueError(f"completion has an invalid file entry: {path}")
        child = path.parent / raw_name
        if not child.is_file() or child.is_symlink():
            raise ValueError(f"completion child is missing or not regular: {child}")
        digest = file_sha256(child)
        if digest != raw_digest:
            raise ValueError(f"completion checksum mismatch: {child}")
        first_pass[raw_name] = digest
        if raw_name.endswith("/complete.json"):
            audit_completion(child, _active=active)
    reopened = strict_json(path)
    if canonical_json(reopened) != canonical_json(completion):
        raise RuntimeError(f"completion changed during independent replay: {path}")
    for name, expected in first_pass.items():
        if file_sha256(path.parent / name) != expected:
            raise RuntimeError(f"artifact changed during independent rehash: {name}")
    active.remove(path)
    return file_sha256(path)


def command_audit(arguments: argparse.Namespace) -> None:
    print(audit_completion(arguments.completion))


def command_write_status(arguments: argparse.Namespace) -> None:
    artifact_digest: str | None = None
    if arguments.artifact_completion is not None:
        artifact_digest = audit_completion(arguments.artifact_completion)
    payload: dict[str, object] = {
        "schema": "openghg_inversions.mh_local_search_s0_job_status.v1",
        "stage": arguments.stage,
        "state": arguments.state,
        "task_id": arguments.task_id,
        "job_id": arguments.job_id,
        "source_revision": os.environ.get("FULL_SHA"),
        "pixi_lock_sha256": os.environ.get("PIXI_LOCK_SHA256"),
        "harness_sha256": os.environ.get("HARNESS_SHA256"),
        "artifact_completion_sha256": artifact_digest,
    }
    create_json(arguments.path, payload)
    if strict_json(arguments.path) != payload:
        raise RuntimeError("create-only status did not reopen exactly")


def command_seal(arguments: argparse.Namespace) -> None:
    completion = arguments.directory / "complete.json"
    if completion.exists():
        raise FileExistsError(f"completion already exists: {completion}")
    files: dict[str, str] = {}
    for name in arguments.file:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in files:
            raise ValueError(f"invalid or duplicate seal path: {name}")
        path = arguments.directory / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"seal input is not a regular file: {path}")
        files[name] = file_sha256(path)
    if not files:
        raise ValueError("a sealed artifact must contain at least one file")
    for name, expected in files.items():
        if file_sha256(arguments.directory / name) != expected:
            raise RuntimeError(f"seal input changed during second rehash: {name}")
    create_json(
        completion,
        {
            "schema": "openghg_inversions.mh_local_search_s0_harness_artifact.v1",
            "files": files,
        },
    )
    audit_completion(completion)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--completion", type=Path, required=True)
    audit.set_defaults(function=command_audit)
    status = commands.add_parser("write-status")
    status.add_argument("--path", type=Path, required=True)
    status.add_argument("--stage", required=True)
    status.add_argument("--state", choices=("started", "complete", "failed"), required=True)
    status.add_argument("--task-id", required=True)
    status.add_argument("--job-id", required=True)
    status.add_argument("--artifact-completion", type=Path)
    status.set_defaults(function=command_write_status)
    seal = commands.add_parser("seal")
    seal.add_argument("--directory", type=Path, required=True)
    seal.add_argument("--file", action="append", required=True)
    seal.set_defaults(function=command_seal)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    arguments.function(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
