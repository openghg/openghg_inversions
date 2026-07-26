#!/usr/bin/env python3
"""Issue the sole sealed factor-four retry authorization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping, Sequence, cast

from openghg_inversions.experimental.rjmcmc.mh_local_search_retry_authorization import (
    issue_factor4_retry_authorization,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_synthetic import (
    canonical_json,
    file_sha256,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_FULL_SHA = re.compile(r"[0-9a-f]{40}")


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
        raise RuntimeError("retry authorization requires a clean exact source revision")
    return revision


def _create_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(dict(payload)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _reopen_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {token}")),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(f"published retry artifact is not strict JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"published retry artifact is not one JSON object: {path}")
    return cast(dict[str, object], value)


def run(
    arguments: argparse.Namespace,
    *,
    enforce_clean_revision: bool = True,
) -> dict[str, object]:
    if arguments.output_directory.exists():
        raise FileExistsError(f"output path already exists: {arguments.output_directory}")
    if not arguments.output_directory.parent.is_dir():
        raise FileNotFoundError("output-directory parent does not exist")
    if enforce_clean_revision and arguments.source_revision != _current_clean_revision():
        raise ValueError("--source-revision must equal the current exact source revision")
    authorization = issue_factor4_retry_authorization(
        training_path=arguments.training,
        evaluation_path=arguments.evaluation,
        primary_certificate_directory=arguments.primary_certificate_directory,
        primary_nuts_directory=arguments.primary_nuts_directory,
        primary_local_directory=arguments.primary_local_directory,
        source_revision=arguments.source_revision,
    )
    arguments.output_directory.mkdir()
    token_path = arguments.output_directory / "token.json"
    _create_json(token_path, authorization.token)
    _create_json(arguments.output_directory / "audit.json", authorization.audit)
    files = {name: file_sha256(arguments.output_directory / name) for name in ("token.json", "audit.json")}
    payloads = {
        "token.json": dict(authorization.token),
        "audit.json": dict(authorization.audit),
    }
    for name, expected_payload in payloads.items():
        path = arguments.output_directory / name
        reopened = _reopen_json(path)
        second_digest = file_sha256(path)
        if canonical_json(reopened) != canonical_json(expected_payload) or second_digest != files[name]:
            raise RuntimeError(f"independent retry-authorization validation failed for {name}")
    _create_json(
        arguments.output_directory / "complete.json",
        {
            "schema": "openghg_inversions.mh_local_search_retry_authorization_completion.v1",
            "status": "complete",
            "token_sha256": files["token.json"],
            "files": files,
        },
    )
    return {
        "schema": "openghg_inversions.mh_local_search_retry_authorization_summary.v1",
        "token_path": str(token_path),
        "token_sha256": files["token.json"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--training", type=Path, required=True)
    result.add_argument("--evaluation", type=Path, required=True)
    result.add_argument("--primary-certificate-directory", type=Path, required=True)
    result.add_argument("--primary-nuts-directory", type=Path, required=True)
    result.add_argument("--primary-local-directory", type=Path, required=True)
    result.add_argument("--output-directory", type=Path, required=True)
    result.add_argument("--source-revision", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
