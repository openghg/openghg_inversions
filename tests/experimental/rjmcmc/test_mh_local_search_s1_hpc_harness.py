"""Regression tests for the immutable BP1 S1 Slurm harness."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, cast

import pytest

from openghg_inversions.experimental.rjmcmc.mh_local_search_synthetic import (
    EVALUATION_SCHEMA,
    TRAINING_SCHEMA,
    build_stage_definition,
    materialize_replicate,
    write_envelope,
)

_ROOT = Path(__file__).parents[3]
_HARNESS = _ROOT / "examples/rjmcmc/hpc/mh_local_search_s1"
_STATIC_FILES = {
    "00-common.sh",
    "00-check-s0-prerequisite.py",
    "01-preflight.sbatch",
    "02-flow-oracle.sbatch",
    "03-materialize.sbatch",
    "04-materialize-evaluation.sbatch",
    "10-pair-array.sbatch",
    "11-oracle-array.sbatch",
    "12-nuts-array.sbatch",
    "13-analyze-array.sbatch",
    "14-local-p0-array.sbatch",
    "15-local-pstar-array.sbatch",
    "16-conditional-array.sbatch",
    "17-build-index.sbatch",
    "18-aggregate.sbatch",
    "20-audit-artifact.py",
    "21-gate-nuts.py",
    "22-gate-conditional.py",
    "23-build-index.py",
    "24-final-audit.py",
    "30-submit-primary.sh",
    "31-submit-nuts-retry1.sh",
    "32-submit-factor4.sh",
    "cell-map.tsv",
    "reference-map.tsv",
    "resources.json",
    "command-spec.json",
    "inventory.json",
}


def _load(name: str) -> ModuleType:
    path = _HARNESS / name
    specification = importlib.util.spec_from_file_location(path.stem, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(specification)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_inventory_maps_resources_and_is_s1_only() -> None:
    inventory = json.loads((_HARNESS / "inventory.json").read_text())
    assert set(inventory["files"]) == _STATIC_FILES
    assert inventory["generated_last"] == ["complete.json"]
    assert {path.name for path in _HARNESS.iterdir() if path.is_file()} == (_STATIC_FILES | {"files.sha256"})
    assert not any(path.is_symlink() for path in _HARNESS.iterdir())

    cells = (_HARNESS / "cell-map.tsv").read_text().splitlines()
    assert cells[0] == "task_id\tscenario\treplicate\tcell_key"
    assert len(cells) == 13
    assert [line.split("\t")[0] for line in cells[1:]] == [str(index) for index in range(12)]
    references = (_HARNESS / "reference-map.tsv").read_text().splitlines()
    assert len(references) == 6
    assert [line.split("\t")[2] for line in references[1:]] == [
        "p0",
        "p0",
        "pstar",
        "p0",
        "pstar",
    ]

    resources = json.loads((_HARNESS / "resources.json").read_text())
    assert resources["account"] == "chem007981"
    assert resources["partition"] is None
    assert resources["stages"]["pair"]["array"] == "0-11%6"
    assert resources["stages"]["nuts_primary"]["array"] == "0-4%2"
    assert resources["stages"]["factor4_local"]["time"] == "72:00:00"
    combined = "\n".join((_HARNESS / name).read_text(errors="strict") for name in _STATIC_FILES).lower()
    assert "paris_inversions" not in combined
    assert "--partition" not in combined
    assert '"stage": "s1"' in combined
    assert '"s1_submission": true' in combined
    assert "--stage s1" in combined
    assert "aggregate-s1" in combined
    command_spec = json.loads((_HARNESS / "command-spec.json").read_text())
    assert command_spec["s0_prerequisite"]["decision_sha256"] == (
        "2cef819c704f0d062cdb38dc09111fa08e230cf2d21ff4b9ba1dd059df1803ef"
    )
    assert command_spec["frozen_definition"] == {
        "payload_sha256": "24099340c5e192bbd258e32270e61247bbad33769da277c39d143f15702f819d",
        "envelope_sha256": "d90c89431d2b19d83084bb51ddd556f3457cf4f53498dd4772b5de9dc6755d60",
        "training_operator_sha256": "382df6a86723ae6988c336ef6825a6b9476d3ad3817e30c425a20932fa58cada",
        "heldout_operator_sha256": "3a789ca21a2d6aea0b541dd344ec97c2e1cd163ec21367bf23902ce25d5a4c56",
        "p0_sha256": "e38df6bdef46ea93b77debfa3b2c4c4efa44cbc816128c8f4376fd5facdd23a1",
        "edge_pstar_sha256": "2a12c303c176e120f8937910800128f1465cf4414b9cf29c2b4102745c8b51c2",
        "relocation_pstar_sha256": "203177d3a2a8ec6e0b8f6bb749a0fef98e96323bf2c16ddf1d246906c96fdce2",
    }
    assert command_spec["mobile_schedule"] == {
        "structural_slots_per_cycle": 2,
        "edge_flip_probability_per_slot": 0.5,
        "resolution_relocation_probability_per_slot": 0.5,
        "root_slice_slots_per_cycle": 1,
        "allocation_pair_slots_per_cycle": 5,
    }


def test_sbatch_provenance_isolation_and_dependency_barriers() -> None:
    for path in sorted(_HARNESS.glob("*.sbatch")):
        text = path.read_text()
        assert "#SBATCH --account=chem007981" in text
        assert "#SBATCH --partition" not in text
        assert "module load git/2.45.1-pqk5" in text
        assert 'source "${HARNESS_DIR:?}/00-common.sh"' in text
    common = (_HARNESS / "00-common.sh").read_text()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert f"export {variable}=1" in common
    assert 'run --manifest-path "${REPO_ROOT}/pyproject.toml"' in common
    pair = (_HARNESS / "10-pair-array.sbatch").read_text()
    assert "pixi_python_at" in pair
    assert "sealed-evaluation" not in pair
    assert " scenario " not in pair
    assert "unset RUN_ROOT HARNESS_DIR BRANCH_PROFILE RETRY_AUTHORIZATION_TOKEN" in pair
    materialize = (_HARNESS / "03-materialize.sbatch").read_text()
    assert "materialize-training" in materialize
    assert "evaluation-output" not in materialize
    assert '"${output}/definition.json"' not in materialize
    assert "mktemp -d" in materialize
    deferred = (_HARNESS / "04-materialize-evaluation.sbatch").read_text()
    assert "materialize-evaluation" in deferred
    assert '"${output}/definition.json"' in deferred
    assert "04-materialize-evaluation/definition.json" in (_HARNESS / "23-build-index.py").read_text()

    submit = (_HARNESS / "30-submit-primary.sh").read_text()
    duplicate_guard = submit.index("refusing duplicate submission before sbatch")
    sbatch_call = submit.index("sbatch --parsable", duplicate_guard)
    assert duplicate_guard < sbatch_call
    assert "local-p0-${profile}" in submit
    assert "local-pstar-${profile}" in submit
    assert "analysis-${profile}" in submit
    assert 'cd "${REPO_ROOT}"' in submit
    pair_submission = submit.index("pair-primary")
    evaluation_submission = submit.index("materialize-evaluation", pair_submission)
    oracle_submission = submit.index("oracle-primary", evaluation_submission)
    assert pair_submission < evaluation_submission < oracle_submission
    factor4 = (_HARNESS / "32-submit-factor4.sh").read_text()
    assert '"${local_p0}:${local_pstar}:${analysis}"' in factor4

    aggregate = (_HARNESS / "18-aggregate.sbatch").read_text()
    assert aggregate.index("harness_job_complete") < aggregate.index("24-final-audit.py")
    preflight = (_HARNESS / "01-preflight.sbatch").read_text()
    assert "test_mh_local_search_s1_hpc_harness.py" in preflight
    assert "provenance.json" in preflight
    for identity in ("numpyro_version", "pytensor_version", "pixi_sha256"):
        assert identity in preflight
    assert "s0-prerequisite.complete.json" in preflight


def test_s0_prerequisite_replays_exact_passing_evidence(tmp_path: Path) -> None:
    prerequisite = _load("00-check-s0-prerequisite.py")
    payload = prerequisite.replay()
    assert payload["status"] == "pass"
    assert payload["source_revision"] == "e9e422fe3ab973898cffbd38df00b689efe212b8"
    output = tmp_path / "s0-prerequisite.complete.json"
    prerequisite.main(["--output", str(output)])
    prerequisite.main(["--verify-output", str(output)])


def test_all_shell_payloads_parse() -> None:
    scripts = sorted(_HARNESS.glob("*.sh")) + sorted(_HARNESS.glob("*.sbatch"))
    completed = subprocess.run(
        ["bash", "-n", *(str(path) for path in scripts)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_recursive_audit_is_create_only_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    audit = _load("20-audit-artifact.py")
    leaf = tmp_path / "leaf"
    leaf.mkdir()
    (leaf / "value.txt").write_text("stable\n")
    audit.command_seal(argparse.Namespace(directory=leaf, file=["value.txt"]))
    expected = audit.audit_completion(leaf / "complete.json")
    assert len(expected) == 64
    with pytest.raises(FileExistsError):
        audit.command_seal(argparse.Namespace(directory=leaf, file=["value.txt"]))
    alias = tmp_path / "alias.json"
    alias.symlink_to(leaf / "complete.json")
    with pytest.raises(ValueError, match="symlink"):
        audit.audit_completion(alias)
    (leaf / "value.txt").write_text("changed\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit.audit_completion(leaf / "complete.json")


def test_frozen_harness_is_content_addressed_and_tamper_evident(
    tmp_path: Path,
) -> None:
    final = _load("24-final-audit.py")
    harness_sha = _sha(_HARNESS / "files.sha256")
    destination = tmp_path / "frozen"
    arguments = argparse.Namespace(
        source=_HARNESS,
        destination=destination,
        source_revision="a" * 40,
        pixi_lock_sha256="b" * 64,
        expected_harness_sha256=harness_sha,
    )
    final.command_freeze(arguments)
    final._verify_harness(
        destination,
        expected_harness_sha256=harness_sha,
        expected_source_revision="a" * 40,
        expected_pixi_lock_sha256="b" * 64,
    )
    helper = destination / "20-audit-artifact.py"
    helper.chmod(stat.S_IRUSR | stat.S_IWUSR)
    helper.write_text(helper.read_text() + "\n# tampered\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        final._verify_harness(
            destination,
            expected_harness_sha256=harness_sha,
            expected_source_revision="a" * 40,
            expected_pixi_lock_sha256="b" * 64,
        )


def test_nuts_retry_is_sparse_and_merges_primary_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _load("21-gate-nuts.py")
    run_root = tmp_path / "run"
    training_root = run_root / "03-materialize/training"
    evaluation_root = run_root / "04-materialize-evaluation/sealed-evaluation"
    training_root.mkdir(parents=True)
    evaluation_root.mkdir(parents=True)
    definition = build_stage_definition("s1")
    for scenario, key in (
        ("aligned", "cell-00"),
        ("edge-one", "cell-04"),
        ("relocation-one", "cell-08"),
    ):
        training, evaluation = materialize_replicate(
            definition,
            scenario=cast(Any, scenario),
            replicate=0,
        )
        write_envelope(training_root / f"{key}.json", TRAINING_SCHEMA, training.payload())
        write_envelope(
            evaluation_root / f"{key}.json",
            EVALUATION_SCHEMA,
            evaluation.payload(),
        )

    def fake_validated_nuts(**kwargs: object) -> tuple[dict[str, object], dict[str, object], None, str]:
        directory = Path(str(kwargs["directory"]))
        key = directory.parent.name
        profile = directory.name
        failed = "rank_normalized_rhat" if key == "edge-one-p0" and profile == "primary" else None
        role = "pstar" if key.endswith("pstar") else "p0"
        return (
            {"first_failed_gate": failed},
            {"profile": profile, "completion_sha256": "c" * 64},
            None,
            role,
        )

    monkeypatch.setattr(gate, "_validated_nuts", fake_validated_nuts)
    primary = gate.run(
        argparse.Namespace(
            run_root=run_root,
            harness_directory=_HARNESS,
            phase="primary",
        )
    )
    assert primary["action"] == "retry1"
    assert primary["retry_task_ids"] == [1]
    assert not (run_root / "status/nuts-selection.json").exists()
    (run_root / "12-nuts/edge-one-p0/retry1").mkdir(parents=True)
    retried = gate.run(
        argparse.Namespace(
            run_root=run_root,
            harness_directory=_HARNESS,
            phase="retry1",
        )
    )
    assert retried["action"] == "conditional"
    selection = json.loads((run_root / "status/nuts-selection.json").read_text())["selected"]
    assert set(selection) == {
        "aligned-p0",
        "edge-one-p0",
        "edge-one-pstar",
        "relocation-one-p0",
        "relocation-one-pstar",
    }
    assert selection["edge-one-p0"] == "retry1"
    assert set(selection.values()) == {"primary", "retry1"}


def test_conditional_gate_creates_only_retry_authorization_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _load("22-gate-conditional.py")
    run_root = tmp_path / "run"
    status = run_root / "status"
    status.mkdir(parents=True)
    references = [line.split("\t") for line in (_HARNESS / "reference-map.tsv").read_text().splitlines()[1:]]
    selection = {parts[4]: "primary" for parts in references}
    (status / "nuts-selection.json").write_text(
        json.dumps({"selected": selection}),
        encoding="utf-8",
    )
    failed_key = references[0][4]
    for _, _, _, _, key in references:
        directory = run_root / "16-conditional" / "local-primary" / "nuts-primary" / key
        directory.mkdir(parents=True)
        passed = key != failed_key
        failed_gate = None if passed else "local_mcse_over_nuts_sd"
        record: dict[str, object] = {
            "profile": "primary",
            "pass": passed,
            "first_failed_gate": failed_gate,
        }
        if not passed:
            record.update(
                {
                    "divergences": 0,
                    "worst_rhat_value": 1.0,
                    "min_bulk_ess_value": 500.0,
                    "min_tail_ess_value": 500.0,
                }
            )
        (directory / "conditional_reference.json").write_text(json.dumps(record))
        (directory / "complete.json").write_text(
            json.dumps(
                {
                    "pass": passed,
                    "first_failed_gate": failed_gate,
                }
            )
        )
        (directory / "audit.json").write_text(json.dumps({"local": {"profile": "primary"}}))

    authorization_parent = run_root / "16-conditional" / "retry-authorization"
    assert not authorization_parent.exists()
    calls = 0

    def fake_run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        output = Path(command[command.index("--output-directory") + 1])
        assert output.parent == authorization_parent
        assert output.parent.is_dir()
        if output.exists():
            raise FileExistsError(f"output path already exists: {output}")
        output.mkdir()
        (output / "token.json").write_text('{"authorized":true}\n')
        (output / "complete.json").write_text('{"status":"complete"}\n')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    arguments = argparse.Namespace(
        run_root=run_root,
        harness_directory=_HARNESS,
        repo_root=_ROOT,
        source_revision="a" * 40,
        profile="primary",
    )
    result = gate.run(arguments)
    assert result["action"] == "factor4"
    assert result["authorization"]["reference_key"] == failed_key
    assert calls == 1
    token = authorization_parent / failed_key / "token.json"
    assert token.read_text() == '{"authorized":true}\n'

    with pytest.raises(FileExistsError, match="output path already exists"):
        gate.run(arguments)
    assert calls == 2
    assert token.read_text() == '{"authorized":true}\n'
