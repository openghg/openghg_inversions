#!/usr/bin/env python3
"""Build the strict 12-cell/5-reference S1 aggregate index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence, cast


def _strict(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, object], value)


def _canonical(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(arguments: argparse.Namespace) -> Path:
    harness = Path(os.environ["HARNESS_DIR"])
    cell_rows = [
        (parts[1], int(parts[2]), parts[3])
        for line in (harness / "cell-map.tsv").read_text().splitlines()[1:]
        if (parts := line.split("\t"))
    ]
    reference_rows = [
        (parts[1], parts[2], int(parts[3]), parts[4])
        for line in (harness / "reference-map.tsv").read_text().splitlines()[1:]
        if (parts := line.split("\t"))
    ]
    selection = cast(
        dict[str, str],
        _strict(arguments.run_root / "status/nuts-selection.json")["selected"],
    )
    definition = arguments.run_root / "04-materialize-evaluation/definition.json"
    cells: list[dict[str, object]] = []
    for scenario, replicate, cell_key in cell_rows:
        training = arguments.run_root / "03-materialize/training" / f"{cell_key}.json"
        evaluation = arguments.run_root / "04-materialize-evaluation/sealed-evaluation" / f"{cell_key}.json"
        pair = arguments.run_root / "10-pair" / arguments.profile / cell_key
        analysis = arguments.run_root / "13-analysis" / arguments.profile / cell_key
        oracle = arguments.run_root / "11-oracle" / arguments.profile / cell_key
        cells.append(
            {
                "scenario": scenario,
                "replicate": replicate,
                "training_path": str(training),
                "training_sha256": _sha(training),
                "evaluation_path": str(evaluation),
                "evaluation_sha256": _sha(evaluation),
                "practical_run_directory": str(pair),
                "practical_complete_sha256": _sha(pair / "complete.json"),
                "practical_analysis_directory": str(analysis),
                "practical_analysis_complete_sha256": _sha(analysis / "complete.json"),
                "oracle_run_directory": str(oracle),
                "oracle_complete_sha256": _sha(oracle / "complete.json"),
            }
        )
    artifacts: list[dict[str, str]] = []
    references: list[dict[str, object]] = []
    for _, topology, cell_task, key in reference_rows:
        nuts_profile = selection[key]
        nuts = arguments.run_root / "12-nuts" / key / nuts_profile
        local_stage = "14-local-p0" if topology == "p0" else "15-local-pstar"
        local = arguments.run_root / local_stage / arguments.profile / key
        conditional = (
            arguments.run_root
            / "16-conditional"
            / f"local-{arguments.profile}"
            / f"nuts-{nuts_profile}"
            / key
        )
        for directory in (nuts, local):
            completion = directory / "complete.json"
            artifacts.append({"path": str(completion), "sha256": _sha(completion)})
        references.append(_strict(conditional / "conditional_reference.json"))
    index: dict[str, object] = {
        "schema": "openghg_inversions.mh_local_search_s1_index.v1",
        "candidate_revision": arguments.source_revision,
        "definition_path": str(definition),
        "definition_file_sha256": _sha(definition),
        "cells": cells,
        "reference_artifacts": artifacts,
        "conditional_references": references,
    }
    if arguments.profile == "factor4":
        gate = _strict(arguments.run_root / "status/conditional-primary-gate.json")
        authorization = cast(dict[str, object], gate["authorization"])
        auth_directory = Path(cast(str, authorization["directory"]))
        primary_certificate = Path(cast(str, authorization["primary_certificate_directory"]))
        primary_nuts = Path(cast(str, authorization["primary_nuts_directory"]))
        primary_local = Path(cast(str, authorization["primary_local_directory"]))
        index["retry_authorization"] = {
            "authorization_completion_path": str(auth_directory / "complete.json"),
            "authorization_completion_sha256": _sha(auth_directory / "complete.json"),
            "primary_certificate_completion_path": str(primary_certificate / "complete.json"),
            "primary_certificate_completion_sha256": _sha(primary_certificate / "complete.json"),
            "primary_nuts_completion_path": str(primary_nuts / "complete.json"),
            "primary_nuts_completion_sha256": _sha(primary_nuts / "complete.json"),
            "primary_local_completion_path": str(primary_local / "complete.json"),
            "primary_local_completion_sha256": _sha(primary_local / "complete.json"),
        }
    output = arguments.run_root / "17-index" / arguments.profile
    output.mkdir(parents=True, exist_ok=False)
    index_path = output / "index.json"
    descriptor = os.open(index_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_canonical(index) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    digest = _sha(index_path)
    if _strict(index_path) != index or _sha(index_path) != digest:
        raise RuntimeError("index changed during independent replay/rehash")
    completion = {
        "schema": "openghg_inversions.mh_local_search_s1_harness_artifact.v1",
        "files": {"index.json": digest},
    }
    descriptor = os.open(output / "complete.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_canonical(completion) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return index_path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--profile", choices=("primary", "factor4"), required=True)
    result.add_argument("--source-revision", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    print(run(parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
