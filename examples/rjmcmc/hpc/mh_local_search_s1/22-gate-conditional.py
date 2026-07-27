#!/usr/bin/env python3
"""Gate five conditional references and issue the sole eligible factor4 token."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence, cast

_LOCAL_GATES = frozenset(
    (
        "local_mcse_over_nuts_sd",
        "half_difference_over_nuts_sd",
        "local_vs_nuts_tolerance",
    )
)


def _strict(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return cast(dict[str, object], value)


def _canonical(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _create(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_canonical(dict(payload)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_float(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"conditional record field {key} must be finite")
    return float(value)


def _maps(harness: Path) -> tuple[list[tuple[str, str, int, str]], dict[int, str]]:
    references = [
        (parts[1], parts[2], int(parts[3]), parts[4])
        for line in (harness / "reference-map.tsv").read_text().splitlines()[1:]
        if (parts := line.split("\t"))
    ]
    cells = {
        int(parts[0]): parts[3]
        for line in (harness / "cell-map.tsv").read_text().splitlines()[1:]
        if (parts := line.split("\t"))
    }
    return references, cells


def run(arguments: argparse.Namespace) -> dict[str, object]:
    references, cells = _maps(arguments.harness_directory)
    selection = cast(
        dict[str, str],
        _strict(arguments.run_root / "status" / "nuts-selection.json")["selected"],
    )
    failures: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for _, topology, cell_task, key in references:
        nuts_profile = selection[key]
        directory = (
            arguments.run_root
            / "16-conditional"
            / f"local-{arguments.profile}"
            / f"nuts-{nuts_profile}"
            / key
        )
        completion = _strict(directory / "complete.json")
        record = _strict(directory / "conditional_reference.json")
        audit = _strict(directory / "audit.json")
        if (
            completion.get("pass") != record.get("pass")
            or completion.get("first_failed_gate") != record.get("first_failed_gate")
            or record.get("profile") != nuts_profile
        ):
            raise ValueError(f"conditional completion identity mismatch for {key}")
        item = {
            "reference_key": key,
            "nuts_profile": nuts_profile,
            "completion_sha256": _sha(directory / "complete.json"),
            "pass": record["pass"],
            "first_failed_gate": record["first_failed_gate"],
        }
        audits.append(item)
        if not record["pass"]:
            nuts_passed = (
                record["divergences"] == 0
                and _finite_float(record, "worst_rhat_value") <= 1.01
                and _finite_float(record, "min_bulk_ess_value") >= 200.0
                and _finite_float(record, "min_tail_ess_value") >= 200.0
            )
            local = cast(dict[str, object], audit["local"])
            if (
                arguments.profile != "primary"
                or local.get("profile") != "primary"
                or not nuts_passed
                or record["first_failed_gate"] not in _LOCAL_GATES
            ):
                raise RuntimeError(
                    f"conditional-reference hard stop for {key}: {record['first_failed_gate']}"
                )
            failures.append(
                {
                    **item,
                    "cell_key": cells[cell_task],
                    "topology": topology,
                    "certificate_directory": str(directory),
                }
            )
    action = "aggregate"
    authorization: dict[str, object] | None = None
    if failures:
        action = "factor4"
        source = failures[0]
        key = cast(str, source["reference_key"])
        topology = cast(str, source["topology"])
        cell_key = cast(str, source["cell_key"])
        nuts_profile = selection[key]
        local_stage = "14-local-p0" if topology == "p0" else "15-local-pstar"
        local_directory = arguments.run_root / local_stage / "primary" / key
        output = arguments.run_root / "16-conditional" / "retry-authorization" / key
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            (
                sys.executable,
                str(arguments.repo_root / "examples/rjmcmc/mh_local_search_retry_authorization.py"),
                "--training",
                str(arguments.run_root / "03-materialize/training" / f"{cell_key}.json"),
                "--evaluation",
                str(arguments.run_root / "04-materialize-evaluation/sealed-evaluation" / f"{cell_key}.json"),
                "--primary-certificate-directory",
                cast(str, source["certificate_directory"]),
                "--primary-nuts-directory",
                str(arguments.run_root / "12-nuts" / key / nuts_profile),
                "--primary-local-directory",
                str(local_directory),
                "--output-directory",
                str(output),
                "--source-revision",
                arguments.source_revision,
            ),
            cwd=arguments.repo_root,
            check=True,
        )
        authorization = {
            "reference_key": key,
            "directory": str(output),
            "completion_sha256": _sha(output / "complete.json"),
            "token_sha256": _sha(output / "token.json"),
            "primary_certificate_directory": source["certificate_directory"],
            "primary_nuts_directory": str(arguments.run_root / "12-nuts" / key / nuts_profile),
            "primary_local_directory": str(local_directory),
        }
    payload = {
        "schema": "openghg_inversions.mh_local_search_s1_conditional_gate.v1",
        "profile": arguments.profile,
        "action": action,
        "audits": audits,
        "eligible_failures": failures,
        "authorization": authorization,
    }
    _create(
        arguments.run_root / "status" / f"conditional-{arguments.profile}-gate.json",
        payload,
    )
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--harness-directory", type=Path, required=True)
    result.add_argument("--repo-root", type=Path, required=True)
    result.add_argument("--source-revision", required=True)
    result.add_argument("--profile", choices=("primary", "factor4"), required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    print(_canonical(run(parser().parse_args(argv))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
