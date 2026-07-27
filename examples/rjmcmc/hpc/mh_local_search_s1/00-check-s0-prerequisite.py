#!/usr/bin/env python3
"""Replay the exact passing S0 eligibility evidence without copying artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence, cast

S0_ROOT = Path(
    "/group/chem/acrg/brendan_for_codex/rjmcmc_mh_guided_local_search_synthetic/"
    "e9e422fe3ab973898cffbd38df00b689efe212b8/"
    "harness-2d9dc06812ab0802a3723c4cb7ef6e66612106d791a924b5558b3f49570f7106"
)
EXPECTED = {
    "source_revision": "e9e422fe3ab973898cffbd38df00b689efe212b8",
    "harness_sha256": "2d9dc06812ab0802a3723c4cb7ef6e66612106d791a924b5558b3f49570f7106",
    "root_completion_sha256": "cdeda8440bfd71119f0509529620ebc5be48a06d37b3d18665357103185491f8",
    "aggregate_completion_sha256": "c8af93d7cc3159810822db0ecd849a3ef3f2efaec9d40e018081aff0eae30a35",
    "decision_sha256": "2cef819c704f0d062cdb38dc09111fa08e230cf2d21ff4b9ba1dd059df1803ef",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid JSON {token}")),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return cast(dict[str, object], value)


def _canonical(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _create(path: Path, payload: Mapping[str, object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_canonical(dict(payload)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def replay() -> dict[str, object]:
    completion_path = S0_ROOT / "complete.json"
    aggregate_path = S0_ROOT / "18-aggregate/factor4/complete.json"
    decision_path = S0_ROOT / "18-aggregate/factor4/decision.json"
    actual = {
        "source_revision": _strict(completion_path).get("source_revision"),
        "harness_sha256": _strict(completion_path).get("harness_sha256"),
        "root_completion_sha256": _sha256(completion_path),
        "aggregate_completion_sha256": _sha256(aggregate_path),
        "decision_sha256": _sha256(decision_path),
    }
    if actual != EXPECTED:
        raise ValueError("S0 prerequisite identity differs from the frozen passing evidence")
    completion = _strict(completion_path)
    decision = _strict(decision_path)
    if (
        completion.get("schema") != "openghg_inversions.mh_local_search_s0_final_completion.v1"
        or completion.get("profile") != "factor4"
        or completion.get("aggregate_completion_sha256") != EXPECTED["aggregate_completion_sha256"]
        or completion.get("decision_sha256") != EXPECTED["decision_sha256"]
        or decision.get("schema") != "openghg_inversions.mh_local_search_s0_decision.v1"
        or decision.get("candidate_revision") != EXPECTED["source_revision"]
        or decision.get("pass") is not True
        or decision.get("first_failed_gate") is not None
    ):
        raise ValueError("S0 prerequisite is not an exact passing final decision")
    return {
        "schema": "openghg_inversions.mh_local_search_s1_s0_prerequisite.v1",
        "status": "pass",
        "s0_root": str(S0_ROOT),
        **EXPECTED,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-output", type=Path)
    arguments = parser.parse_args(argv)
    payload = replay()
    if arguments.output is not None:
        _create(arguments.output, payload)
    if arguments.verify_output is not None and _canonical(_strict(arguments.verify_output)) != _canonical(
        payload
    ):
        raise ValueError("recorded S0 prerequisite differs from exact replay")
    print(_canonical(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
