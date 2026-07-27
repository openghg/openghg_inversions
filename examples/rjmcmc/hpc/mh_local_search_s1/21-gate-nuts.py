#!/usr/bin/env python3
"""Replay five NUTS bundles and publish a create-only sparse retry decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence, cast

from openghg_inversions.experimental.rjmcmc.mh_local_search_conditional_reference import (
    _validated_nuts,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_synthetic import (
    load_evaluation_artifact,
    load_training_artifact,
    validate_artifact_pair,
)

_RETRY_GATES = frozenset(("zero_divergences", "rank_normalized_rhat", "bulk_ess", "tail_ess"))


def _canonical(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _create(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _references(path: Path) -> list[tuple[int, str, str, int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines[0] != "task_id\tscenario\ttopology\tcell_task_id\treference_key":
        raise ValueError("reference map header is incompatible")
    return [
        (int(parts[0]), parts[1], parts[2], int(parts[3]), parts[4])
        for line in lines[1:]
        if (parts := line.split("\t"))
    ]


def _cells(path: Path) -> dict[int, str]:
    return {
        int(parts[0]): parts[3]
        for line in path.read_text(encoding="utf-8").splitlines()[1:]
        if (parts := line.split("\t"))
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(arguments: argparse.Namespace) -> dict[str, object]:
    status = arguments.run_root / "status"
    references = _references(arguments.harness_directory / "reference-map.tsv")
    cells = _cells(arguments.harness_directory / "cell-map.tsv")
    retry_tasks: set[int] = set()
    if arguments.phase == "retry1":
        primary_gate = json.loads((status / "nuts-primary-gate.json").read_text(encoding="utf-8"))
        if (
            not isinstance(primary_gate, dict)
            or primary_gate.get("action") != "retry1"
            or not isinstance(primary_gate.get("retry_task_ids"), list)
        ):
            raise ValueError("retry1 requires one compatible primary gate")
        retry_tasks = {int(cast(Any, task)) for task in primary_gate["retry_task_ids"]}
        if not retry_tasks:
            raise ValueError("retry1 task set must be non-empty")
    selected: dict[str, str] = {}
    retries: list[tuple[int, str]] = []
    audits: list[dict[str, object]] = []
    for task, _, topology, cell_task, key in references:
        cell_key = cells[cell_task]
        training_path = arguments.run_root / "03-materialize" / "training" / f"{cell_key}.json"
        evaluation_path = (
            arguments.run_root / "04-materialize-evaluation" / "sealed-evaluation" / f"{cell_key}.json"
        )
        training = load_training_artifact(training_path)
        evaluation = load_evaluation_artifact(evaluation_path)
        validate_artifact_pair(training, evaluation)
        retry_directory = arguments.run_root / "12-nuts" / key / "retry1"
        if arguments.phase == "retry1":
            profile = "retry1" if task in retry_tasks else "primary"
            if (task in retry_tasks) != retry_directory.is_dir():
                raise ValueError(f"retry1 directory set differs from authorized sparse tasks: {key}")
        else:
            profile = "primary"
        summary, audit, _, role = _validated_nuts(
            directory=arguments.run_root / "12-nuts" / key / profile,
            training_path=training_path,
            evaluation_path=evaluation_path,
            training=training,
            evaluation=evaluation,
        )
        if role != topology or audit["profile"] != profile:
            raise ValueError(f"NUTS target/profile identity mismatch for {key}")
        failure = summary["first_failed_gate"]
        audits.append(
            {
                "reference_key": key,
                "profile": profile,
                "completion_sha256": audit["completion_sha256"],
                "first_failed_gate": failure,
            }
        )
        if failure is None:
            selected[key] = profile
        elif arguments.phase == "primary" and failure in _RETRY_GATES:
            retries.append((task, key))
        else:
            raise RuntimeError(f"NUTS hard stop for {key}: {failure}")
    action = "retry1" if retries else "conditional"
    payload: dict[str, object] = {
        "schema": "openghg_inversions.mh_local_search_s1_nuts_gate.v1",
        "phase": arguments.phase,
        "action": action,
        "selected": selected,
        "retry_task_ids": [task for task, _ in retries],
        "audits": audits,
    }
    gate_path = status / f"nuts-{arguments.phase}-gate.json"
    _create(gate_path, _canonical(payload) + "\n")
    if retries:
        retry_path = status / "nuts-retry-map.tsv"
        text = "task_id\treference_key\n" + "".join(f"{task}\t{key}\n" for task, key in retries)
        _create(retry_path, text)
    else:
        selection = {
            "schema": "openghg_inversions.mh_local_search_s1_nuts_selection.v1",
            "selected": selected,
            "gate_sha256": _sha(gate_path),
        }
        _create(status / "nuts-selection.json", _canonical(selection) + "\n")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--harness-directory", type=Path, required=True)
    result.add_argument("--phase", choices=("primary", "retry1"), required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    print(_canonical(run(parser().parse_args(argv))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
