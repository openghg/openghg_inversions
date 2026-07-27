#!/usr/bin/env python3
"""Freeze, verify, and independently finalize an immutable S1 harness run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Mapping, Sequence, cast

_DIGEST = re.compile(r"[0-9a-f]{64}")
_FULL_SHA = re.compile(r"[0-9a-f]{40}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _strict_json(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {token}")),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return cast(dict[str, object], value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _create_json(path: Path, payload: Mapping[str, object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_canonical_json(dict(payload)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _catalogue(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or _DIGEST.fullmatch(parts[0]) is None or not parts[1] or parts[1] in result:
            raise ValueError("files.sha256 is not a canonical checksum catalogue")
        result[parts[1]] = parts[0]
    return result


def _source_inventory(source: Path) -> tuple[list[str], dict[str, str], str]:
    inventory = _strict_json(source / "inventory.json")
    if (
        frozenset(inventory) != {"schema", "files", "generated_last", "checksum_catalogue"}
        or inventory["schema"] != "openghg_inversions.mh_local_search_s1_harness_inventory.v1"
        or inventory["generated_last"] != ["complete.json"]
        or inventory["checksum_catalogue"] != "files.sha256"
        or not isinstance(inventory["files"], list)
    ):
        raise ValueError("harness inventory schema is incompatible")
    names = cast(list[object], inventory["files"])
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(cast(list[str], names))) != len(names)
        or "files.sha256" in names
        or "complete.json" in names
    ):
        raise ValueError("harness inventory names are incompatible")
    expected_names = cast(list[str], names)
    actual = {path.name for path in source.iterdir() if path.is_file() and not path.is_symlink()}
    if actual != set(expected_names) | {"files.sha256"}:
        raise ValueError("harness source directory differs from its exact inventory")
    catalogue = _catalogue(source / "files.sha256")
    if set(catalogue) != set(expected_names):
        raise ValueError("harness checksum catalogue differs from inventory")
    for name, expected in catalogue.items():
        path = source / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"harness source checksum mismatch for {name}")
    return expected_names, catalogue, _sha256(source / "files.sha256")


def _verify_harness(
    directory: Path,
    *,
    expected_harness_sha256: str,
    expected_source_revision: str,
    expected_pixi_lock_sha256: str,
) -> dict[str, object]:
    completion = _strict_json(directory / "complete.json")
    if (
        frozenset(completion)
        != {
            "schema",
            "source_revision",
            "pixi_lock_sha256",
            "harness_sha256",
            "files_sha256",
            "files",
        }
        or completion["schema"] != "openghg_inversions.mh_local_search_s1_frozen_harness.v1"
        or completion["source_revision"] != expected_source_revision
        or completion["pixi_lock_sha256"] != expected_pixi_lock_sha256
        or completion["harness_sha256"] != expected_harness_sha256
        or not isinstance(completion["files"], dict)
    ):
        raise ValueError("frozen harness completion identity is incompatible")
    files = cast(dict[str, object], completion["files"])
    catalogue = _catalogue(directory / "files.sha256")
    if set(files) != set(catalogue) | {"files.sha256"}:
        raise ValueError("frozen harness completion differs from its catalogue")
    for name, expected in catalogue.items():
        if files.get(name) != expected:
            raise ValueError(f"frozen harness catalogue identity mismatch for {name}")
    for name, raw_digest in files.items():
        if (
            not isinstance(name, str)
            or not isinstance(raw_digest, str)
            or _DIGEST.fullmatch(raw_digest) is None
            or _sha256(directory / name) != raw_digest
        ):
            raise ValueError(f"frozen harness checksum mismatch for {name}")
    if completion["files_sha256"] != files.get("files.sha256"):
        raise ValueError("frozen harness catalogue digest is inconsistent")
    if _sha256(directory / "files.sha256") != expected_harness_sha256:
        raise ValueError("frozen harness content address is inconsistent")
    reopened = _strict_json(directory / "complete.json")
    if _canonical_json(reopened) != _canonical_json(completion):
        raise RuntimeError("frozen harness completion changed during replay")
    for name, expected in files.items():
        if _sha256(directory / name) != expected:
            raise RuntimeError(f"frozen harness changed during second rehash: {name}")
    return completion


def command_freeze(arguments: argparse.Namespace) -> None:
    if _FULL_SHA.fullmatch(arguments.source_revision) is None:
        raise ValueError("source revision must be an exact full SHA")
    if _DIGEST.fullmatch(arguments.pixi_lock_sha256) is None:
        raise ValueError("pixi lock digest must be an exact SHA-256")
    names, _, harness_digest = _source_inventory(arguments.source)
    if arguments.expected_harness_sha256 != harness_digest:
        raise ValueError("requested harness address differs from files.sha256")
    arguments.destination.mkdir(parents=False, exist_ok=False)
    for name in (*names, "files.sha256"):
        source = arguments.source / name
        destination = arguments.destination / name
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
    first = {name: _sha256(arguments.destination / name) for name in (*names, "files.sha256")}
    _source_inventory(arguments.destination)
    for name, expected in first.items():
        if _sha256(arguments.destination / name) != expected:
            raise RuntimeError(f"frozen harness changed during second rehash: {name}")
    _create_json(
        arguments.destination / "complete.json",
        {
            "schema": "openghg_inversions.mh_local_search_s1_frozen_harness.v1",
            "source_revision": arguments.source_revision,
            "pixi_lock_sha256": arguments.pixi_lock_sha256,
            "harness_sha256": harness_digest,
            "files_sha256": first["files.sha256"],
            "files": first,
        },
    )
    _verify_harness(
        arguments.destination,
        expected_harness_sha256=harness_digest,
        expected_source_revision=arguments.source_revision,
        expected_pixi_lock_sha256=arguments.pixi_lock_sha256,
    )
    executable_suffixes = {".py", ".sh", ".sbatch"}
    for path in arguments.destination.iterdir():
        mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        if path.suffix in executable_suffixes:
            mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        path.chmod(mode)
    arguments.destination.chmod(
        stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    )
    print(harness_digest)


def command_verify_harness(arguments: argparse.Namespace) -> None:
    _verify_harness(
        arguments.harness_directory,
        expected_harness_sha256=arguments.expected_harness_sha256,
        expected_source_revision=arguments.expected_source_revision,
        expected_pixi_lock_sha256=arguments.expected_pixi_lock_sha256,
    )


def _audit_completion(completion_path: Path) -> str:
    helper = completion_path.parents[2] / "harness" / "20-audit-artifact.py"
    completed = subprocess.run(
        (
            sys.executable,
            str(helper),
            "audit",
            "--completion",
            str(completion_path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    digest = completed.stdout.strip()
    if _DIGEST.fullmatch(digest) is None:
        raise RuntimeError("artifact auditor did not return a SHA-256")
    return digest


def command_finalize(arguments: argparse.Namespace) -> None:
    profile = arguments.profile
    prerequisite = arguments.run_root / "status/s0-prerequisite.complete.json"
    subprocess.run(
        (
            sys.executable,
            str(arguments.run_root / "harness/00-check-s0-prerequisite.py"),
            "--verify-output",
            str(prerequisite),
        ),
        check=True,
    )
    index_path = arguments.run_root / "17-index" / profile / "index.json"
    aggregate = arguments.run_root / "18-aggregate" / profile
    aggregate_digest = _audit_completion(aggregate / "complete.json")
    replay = arguments.run_root / "19-final-replay" / profile
    pixi_executable = Path(os.environ.get("PIXI_EXE", "/user/work/bm13805/.pixi/bin/pixi"))
    if not pixi_executable.is_absolute() or not os.access(pixi_executable, os.X_OK):
        raise RuntimeError("PIXI_EXE must identify an executable absolute path")
    command = (
        str(pixi_executable),
        "run",
        "--as-is",
        "-e",
        "dev",
        "python",
        "examples/rjmcmc/mh_local_search_synthetic.py",
        "aggregate-s1",
        "--index",
        str(index_path),
        "--output-directory",
        str(replay),
    )
    subprocess.run(command, cwd=arguments.repo_root, check=True)
    replay_digest = _audit_completion(replay / "complete.json")
    if _canonical_json(_strict_json(aggregate / "decision.json")) != _canonical_json(
        _strict_json(replay / "decision.json")
    ):
        raise RuntimeError("independent aggregate replay differs from promoted decision")
    payload = {
        "schema": "openghg_inversions.mh_local_search_s1_final_completion.v1",
        "profile": profile,
        "source_revision": os.environ["FULL_SHA"],
        "pixi_lock_sha256": os.environ["PIXI_LOCK_SHA256"],
        "harness_sha256": os.environ["HARNESS_SHA256"],
        "s0_prerequisite_sha256": _sha256(prerequisite),
        "index_sha256": _sha256(index_path),
        "aggregate_completion_sha256": aggregate_digest,
        "replay_completion_sha256": replay_digest,
        "decision_sha256": _sha256(aggregate / "decision.json"),
    }
    status_path = arguments.run_root / "status" / f"final-{profile}.complete.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    _create_json(status_path, payload)
    if _strict_json(status_path) != payload or _sha256(aggregate / "complete.json") != aggregate_digest:
        raise RuntimeError("final completion failed independent replay/rehash")
    _create_json(arguments.run_root / "complete.json", payload)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--destination", type=Path, required=True)
    freeze.add_argument("--source-revision", required=True)
    freeze.add_argument("--pixi-lock-sha256", required=True)
    freeze.add_argument("--expected-harness-sha256", required=True)
    freeze.set_defaults(function=command_freeze)
    verify = commands.add_parser("verify-harness")
    verify.add_argument("--harness-directory", type=Path, required=True)
    verify.add_argument("--expected-harness-sha256", required=True)
    verify.add_argument("--expected-source-revision", required=True)
    verify.add_argument("--expected-pixi-lock-sha256", required=True)
    verify.set_defaults(function=command_verify_harness)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--repo-root", type=Path, required=True)
    finalize.add_argument("--profile", choices=("primary", "factor4"), required=True)
    finalize.set_defaults(function=command_finalize)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    arguments.function(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
