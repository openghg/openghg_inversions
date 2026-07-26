"""Focused tests for the exact tiny MH-flow R0 oracle."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, cast

import pytest

from openghg_inversions.experimental.rjmcmc import mh_local_search_flow_oracle as flow_oracle
from openghg_inversions.experimental.rjmcmc.mh_local_search_flow_oracle import (
    AUDIT_FILENAME,
    AUDIT_SCHEMA,
    COMPLETION_FILENAME,
    publish_flow_oracle,
    run_flow_oracle,
    validate_flow_oracle_bundle,
)

_DRIVER_PATH = Path(__file__).parents[3] / "examples" / "rjmcmc" / "mh_local_search_flow_oracle.py"
_TEST_HEAD = "0123456789abcdef0123456789abcdef01234567"


def _canonical_json(value: object) -> str:
    """Return the certificate's deterministic strict JSON representation."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 of one test artifact."""
    return sha256(path.read_bytes()).hexdigest()


def _rehash_bundle(
    output: Path,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    """Apply semantic drift while making both checksum layers self-consistent."""
    audit_path = output / AUDIT_FILENAME
    envelope = json.loads(audit_path.read_text(encoding="utf-8"))
    payload = cast(dict[str, Any], envelope["payload"])
    mutation(payload)
    envelope["payload_sha256"] = sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    audit_path.write_text(_canonical_json(envelope) + "\n", encoding="utf-8")
    completion_path = output / COMPLETION_FILENAME
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["files"][AUDIT_FILENAME] = _file_sha256(audit_path)
    completion_path.write_text(_canonical_json(completion) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def driver() -> ModuleType:
    """Load the standalone driver without invoking its entry point."""
    specification = importlib.util.spec_from_file_location(
        "mh_local_search_flow_oracle_driver",
        _DRIVER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load flow-oracle driver")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit() -> dict[str, object]:
    """Run the deterministic in-memory oracle once for focused assertions."""
    return run_flow_oracle(source_revision="unit-test-revision")


def test_exact_tiny_posterior_flow_oracle_passes_every_catalogue_gate(
    audit: dict[str, object],
) -> None:
    """All eight topologies and three fractions satisfy the R0 identities."""
    assert audit["status"] == "pass"
    checks = cast(dict[str, object], audit["checks"])
    assert checks and all(value is True for value in checks.values())

    catalogue = cast(dict[str, Any], audit["catalogue"])
    counts = cast(dict[str, int], catalogue["counts"])
    assert catalogue["topologies"] == 8
    assert catalogue["fractions"] == [0.25, 0.5, 0.75]
    assert catalogue["relocation_destination_slots_per_merge"] == 6
    assert len(catalogue["topology_sha256"]) == 8
    assert len(set(catalogue["topology_sha256"])) == 8
    assert counts["edge_attempts"] == counts["edge_valid"] + counts["edge_invalid"]
    assert counts["relocation_attempts"] == (counts["relocation_valid"] + counts["relocation_invalid"])
    assert counts["valid_reverses"] == counts["edge_valid"] + counts["relocation_valid"]
    assert counts["exact_invalid_self_transitions"] == (counts["edge_invalid"] + counts["relocation_invalid"])
    assert counts["relocation_invalid"] > 0
    assert counts["edge_valid"] > 0
    assert counts["relocation_valid"] > 0
    sampler_law = cast(dict[str, object], audit["sampler_law"])
    assert (
        sampler_law["schedule_id"]
        == "full_tiling_2_mixed_structure_1_root_slice_n_pair_allocation_fixed_sweep_v2"
    )
    assert sampler_law["move_mixture_weights"] == {
        "edge_flip": 0.5,
        "resolution_relocation": 0.5,
    }

    likelihood = cast(dict[str, float], audit["likelihood"])
    assert likelihood["range"] > 0.0
    assert likelihood["nonzero_transition_deltas"] > 0
    maxima = cast(dict[str, dict[str, float]], audit["maximum_discrepancies"])
    assert maxima
    for maximum in maxima.values():
        assert 0.0 <= maximum["tolerance_fraction"] <= 1.0


def test_in_memory_audit_is_deterministic(audit: dict[str, object]) -> None:
    """The same revision and runtime produce byte-stable scientific payloads."""
    assert run_flow_oracle(source_revision="unit-test-revision") == audit


def test_create_only_bundle_is_checksummed_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completion is written last and detects later audit corruption."""
    monkeypatch.setattr(flow_oracle, "_clean_git_head", lambda: _TEST_HEAD)
    output = tmp_path / "flow-oracle"
    published = publish_flow_oracle(
        output,
        source_revision=_TEST_HEAD,
    )
    assert published == run_flow_oracle(source_revision=_TEST_HEAD)
    assert validate_flow_oracle_bundle(output) == published

    envelope = json.loads((output / AUDIT_FILENAME).read_text(encoding="utf-8"))
    assert envelope["schema"] == AUDIT_SCHEMA
    canonical = json.dumps(
        envelope["payload"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert envelope["payload_sha256"] == sha256(canonical.encode("utf-8")).hexdigest()
    completion = json.loads((output / COMPLETION_FILENAME).read_text(encoding="utf-8"))
    assert set(completion["files"]) == {AUDIT_FILENAME}

    with pytest.raises(FileExistsError):
        publish_flow_oracle(
            output,
            source_revision=_TEST_HEAD,
        )

    envelope["payload"]["status"] = "corrupt"
    (output / AUDIT_FILENAME).write_text(
        json.dumps(envelope, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum"):
        validate_flow_oracle_bundle(output)


@pytest.mark.parametrize(
    ("name", "mutation", "message"),
    (
        (
            "static-model-drift",
            lambda payload: payload["model"]["observations"].__setitem__(0, 99.0),
            "model identity",
        ),
        (
            "static-model-type-drift",
            lambda payload: payload["model"].__setitem__("k", 4.0),
            "model identity",
        ),
        (
            "top-level-schema-drift",
            lambda payload: payload.__setitem__("unexpected", True),
            "payload keys",
        ),
        (
            "sampler-law-drift",
            lambda payload: payload["sampler_law"].__setitem__(
                "availability_renormalization",
                True,
            ),
            "sampler-law",
        ),
        (
            "count-relation-drift",
            lambda payload: payload["catalogue"]["counts"].__setitem__(
                "edge_attempts",
                44,
            ),
            "counts",
        ),
        (
            "false-check-drift",
            lambda payload: payload["checks"].__setitem__(
                "accepted_pointwise_mh_flow_is_equal",
                False,
            ),
            "literal true",
        ),
        (
            "maximum-drift",
            lambda payload: payload["maximum_discrepancies"]["accepted_flow_equality"].__setitem__(
                "absolute_difference", 1.0
            ),
            "exceeds",
        ),
    ),
)
def test_self_consistently_rehashed_semantic_and_schema_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Recomputing both hashes cannot turn semantically invalid data into a pass."""
    monkeypatch.setattr(flow_oracle, "_clean_git_head", lambda: _TEST_HEAD)
    output = tmp_path / name
    publish_flow_oracle(output, source_revision=_TEST_HEAD)
    _rehash_bundle(output, mutation)
    with pytest.raises(ValueError, match=message):
        validate_flow_oracle_bundle(output)


def test_publication_requires_exact_current_clean_lowercase_git_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication rejects malformed, mismatched, and dirty provenance before creating output."""
    monkeypatch.setattr(flow_oracle, "_clean_git_head", lambda: _TEST_HEAD)
    with pytest.raises(ValueError, match="lowercase full 40-hex"):
        publish_flow_oracle(
            tmp_path / "uppercase",
            source_revision=_TEST_HEAD.upper(),
        )
    with pytest.raises(ValueError, match="current clean Git HEAD"):
        publish_flow_oracle(
            tmp_path / "mismatch",
            source_revision="f" * 40,
        )

    def dirty() -> str:
        raise RuntimeError("flow-oracle publication requires a clean Git worktree")

    monkeypatch.setattr(flow_oracle, "_clean_git_head", dirty)
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        publish_flow_oracle(
            tmp_path / "dirty",
            source_revision=_TEST_HEAD,
        )
    assert not any((tmp_path / name).exists() for name in ("uppercase", "mismatch", "dirty"))


def test_schedule_identity_drift_fails_before_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed compound schedule requires an explicit oracle version review."""
    monkeypatch.setattr(
        flow_oracle,
        "FULL_TILING_COMPOUND_SCHEDULE_ID",
        "unreviewed-schedule",
    )
    with pytest.raises(RuntimeError, match="schedule identity drifted"):
        run_flow_oracle(source_revision="in-memory-test")


def test_driver_publishes_compact_machine_readable_summary(
    tmp_path: Path,
    driver: ModuleType,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI publishes one certified bundle and reports its exact location."""
    monkeypatch.setattr(flow_oracle, "_clean_git_head", lambda: _TEST_HEAD)
    output = tmp_path / "driver-flow-oracle"
    assert (
        driver.main(
            [
                "--output-directory",
                str(output),
                "--source-revision",
                _TEST_HEAD,
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "pass"
    assert summary["source_revision"] == _TEST_HEAD
    assert summary["topologies"] == 8
    assert Path(summary["certificate"]) == output / AUDIT_FILENAME
    assert validate_flow_oracle_bundle(output)["source_revision"] == _TEST_HEAD
