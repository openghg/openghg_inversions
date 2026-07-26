"""Tests for the separate C1-RQMC conditional-allocation tiny screen."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from examples.rjmcmc import (
    conditional_allocation_likelihood_rqmc_tiny_screen as rqmc,
)
from examples.rjmcmc import (
    conditional_allocation_likelihood_tiny_screen as pcg,
)


def _smoke_report() -> dict[str, Any]:
    """Return one bounded timing-free RQMC smoke report."""
    return rqmc.run_screen(
        profile="smoke",
        sample_counts=(64,),
        repeat_seeds=(731,),
        include_timings=False,
    )


def test_rqmc_protocol_reuses_the_exact_frozen_c1_definitions() -> None:
    """RQMC should change only bank construction, not the C1 experiment."""
    assert rqmc.SCHEMA == ("rjmcmc-conditional-allocation-c1-rqmc-tiny-screen-v1")
    assert rqmc.PROTOCOL == ("conditional-allocation-c1-full-rank-scrambled-sobol-balanced-dirichlet-bank-v1")
    assert rqmc.BANK_CONSTRUCTION_METHOD == "scrambled_sobol_balanced_dirichlet"
    assert rqmc.DEVELOPMENT_SCIPY_VERSION == "1.15.2"
    assert rqmc.DEVELOPMENT_PROTOCOL_SHA256 == (
        "dcb2ef2bebb0c7eefafbd49a225c864e1b8a7478c568c168ed1640dd91ea9f4b"
    )
    assert rqmc.DEVELOPMENT_MATRIX is pcg.DEVELOPMENT_MATRIX
    assert rqmc.DEVELOPMENT_SAMPLE_COUNTS is pcg.DEVELOPMENT_SAMPLE_COUNTS
    assert rqmc.DEVELOPMENT_REPEAT_SEEDS is pcg.DEVELOPMENT_REPEAT_SEEDS
    assert rqmc.THRESHOLDS is pcg.THRESHOLDS
    assert rqmc.MERGER_THRESHOLDS is pcg.MERGER_THRESHOLDS
    assert pcg.a1_definitions_sha256() == rqmc.A1_DEFINITIONS_SHA256

    catalogue = rqmc.matrix_catalogue()
    assert catalogue["development"] == [list(case) for case in pcg.DEVELOPMENT_MATRIX]
    assert catalogue["held_out_information_read"] is False
    assert catalogue["held_out_catalogue"] == {
        "id": pcg.HELD_OUT_CATALOGUE_ID,
        "sha256": pcg.HELD_OUT_CATALOGUE_SHA256,
        "numerical_values_present": False,
        "executable_here": False,
    }


def test_bounded_rqmc_smoke_replays_with_method_provenance() -> None:
    """A fixed scramble should replay and retain method identity everywhere."""
    first = _smoke_report()
    second = _smoke_report()

    assert first == second
    assert first["schema"] == rqmc.SCHEMA
    assert first["protocol"] == rqmc.PROTOCOL
    assert first["required_development_construction_scipy_version"] == "1.15.2"
    report_bank = first["bank"]
    assert report_bank["construction_method"] == rqmc.BANK_CONSTRUCTION_METHOD
    assert report_bank["construction_scipy_version"] == rqmc.scipy_version
    assert report_bank["quasi_random_engine"] == "scipy.stats.qmc.Sobol"
    assert report_bank["sobol_bits"] == 52
    assert report_bank["sobol_scramble"] is True
    assert report_bank["sobol_optimization"] is None
    assert report_bank["inverse_transform"] == "scipy.special.betaincinv"
    assert report_bank["dimension_order"] == ("stable-id-region-signature/count-balanced-breadth-first")
    assert report_bank["sobol_block_rule"] == ("contiguous-canonical-node-catalogue/max-dimension-21201")
    assert report_bank["sobol_seed_derivation"] == (
        "sha256(schema-v2,source-seed,node-count,block-index,catalogue-sha256)/little-endian-first-64"
    )
    assert report_bank["protocol_hash_inclusion"] == {
        "algorithm_token_included": True,
        "required_development_scipy_version_included": True,
        "observed_runtime_and_derived_provenance_included": False,
        "reason": (
            "development fails unless the observed runtime matches the hashed "
            "SciPy requirement; catalogue/block identities are deterministic "
            "consequences of already pinned case inputs"
        ),
    }
    assert first["held_out_information_read"] is False
    assert first["observed_residual_used_for_basis_selection"] is False
    assert first["structural_inference_licensed"] is False
    assert first["full_c1_pass"] is False
    assert len(first["cases"]) == 1
    case = first["cases"][0]
    assert case["bank"]["construction_method"] == (rqmc.BANK_CONSTRUCTION_METHOD)
    case_id = case["case_id"]
    assert report_bank["sobol_catalogue_sha256"] == {case_id: case["bank"]["sobol_catalogue_sha256"]}
    assert report_bank["sobol_block_dimensions"] == {case_id: case["bank"]["sobol_block_dimensions"]}
    assert len(case["bank"]["sobol_catalogue_sha256"]) == 64
    assert case["bank"]["sobol_block_dimensions"] == [1]
    evaluation = case["development_evaluations"][0]
    assert evaluation["construction_method"] == (rqmc.BANK_CONSTRUCTION_METHOD)
    assert evaluation["bank_construction_provenance"] == {
        field: case["bank"][field] for field in rqmc._ARTIFACT_PROVENANCE_FIELDS
    }
    assert evaluation["source_seed"] == 731
    assert evaluation["scramble_seed"] == 731
    assert evaluation["sample_count"] == 64
    assert all(math.isfinite(float(value)) for value in evaluation["metrics"].values())
    assert evaluation["build_seconds"] is None
    assert evaluation["evaluation_seconds"] is None
    serialized = json.dumps(first, allow_nan=False)
    assert '"masses"' not in serialized
    assert "column" not in serialized


def test_smoke_protocol_digest_is_method_specific_and_replays() -> None:
    """The exact matrix and RQMC method must participate in protocol identity."""
    report = _smoke_report()
    expected = pcg._sha256_json(
        {
            "schema": rqmc.SCHEMA,
            "protocol": rqmc.PROTOCOL,
            "bank_construction_method": rqmc.BANK_CONSTRUCTION_METHOD,
            "required_development_construction_scipy_version": (rqmc.DEVELOPMENT_SCIPY_VERSION),
            "thresholds": rqmc.THRESHOLDS,
            "merger_thresholds": rqmc.MERGER_THRESHOLDS,
            "gradient_step": pcg.GRADIENT_STEP,
            "sample_counts": [64],
            "repeat_seeds": [731],
            "matrix": rqmc.SMOKE_MATRIX,
        }
    )

    assert report["protocol_sha256"] == expected
    assert (
        report["protocol_sha256"]
        != pcg.run_screen(
            profile="smoke",
            sample_counts=(64,),
            repeat_seeds=(731,),
            include_timings=False,
        )["protocol_sha256"]
    )


@pytest.mark.parametrize(
    ("sample_counts", "repeat_seeds", "message"),
    [
        ((64,), rqmc.DEVELOPMENT_REPEAT_SEEDS, "source-pinned"),
        (
            rqmc.DEVELOPMENT_SAMPLE_COUNTS,
            (731,),
            "source-pinned",
        ),
        (
            (96,),
            (731,),
            "powers of two",
        ),
    ],
)
def test_direct_development_and_rqmc_overrides_fail_closed(
    sample_counts: tuple[int, ...],
    repeat_seeds: tuple[int, ...],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Development settings and the Sobol power-of-two rule are immutable."""
    profile: rqmc.Profile = "development" if message == "source-pinned" else "smoke"
    if profile == "development":
        monkeypatch.setattr(
            rqmc,
            "scipy_version",
            rqmc.DEVELOPMENT_SCIPY_VERSION,
        )
    with pytest.raises(ValueError, match=message):
        rqmc.run_case(
            regime_name="near_gaussian",
            family="two_cell",
            tiling="root",
            sample_counts=sample_counts,
            repeat_seeds=repeat_seeds,
            profile=profile,
            include_timings=False,
        )


def test_run_screen_rejects_development_overrides_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The matrix helper must not allow post-protocol tuning."""
    monkeypatch.setattr(
        rqmc,
        "run_case",
        lambda **_: pytest.fail("invalid development must not execute"),
    )

    with pytest.raises(ValueError, match="source-pinned"):
        rqmc.run_screen(
            profile="development",
            sample_counts=(64,),
            repeat_seeds=None,
            include_timings=False,
        )


def test_development_scipy_pin_fails_before_scientific_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Development must not span distinct SciPy Sobol implementations."""
    original_run_case = rqmc.run_case
    monkeypatch.setattr(rqmc, "scipy_version", "0.0.invalid")
    monkeypatch.setattr(
        rqmc,
        "run_case",
        lambda **_: pytest.fail("wrong SciPy version must not execute"),
    )

    with pytest.raises(RuntimeError, match="development requires SciPy 1.15.2"):
        rqmc.run_screen(
            profile="development",
            include_timings=False,
        )
    with pytest.raises(RuntimeError, match="development requires SciPy 1.15.2"):
        original_run_case(
            regime_name="near_gaussian",
            family="two_cell",
            tiling="root",
            sample_counts=rqmc.DEVELOPMENT_SAMPLE_COUNTS,
            repeat_seeds=rqmc.DEVELOPMENT_REPEAT_SEEDS,
            profile="development",
            include_timings=False,
        )


def test_source_revision_is_validated_before_case_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad source provenance should stop before building an RQMC bank."""
    monkeypatch.setattr(
        pcg,
        "_source_revision",
        lambda _: (_ for _ in ()).throw(RuntimeError("bad source revision")),
    )
    monkeypatch.setattr(
        rqmc,
        "run_case",
        lambda **_: pytest.fail("bad source revision must not execute"),
    )

    with pytest.raises(RuntimeError, match="bad source revision"):
        rqmc.run_screen(profile="smoke", include_timings=False)


def test_stale_full_development_protocol_fails_before_one_case_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Slurm case must validate the full matrix protocol before running."""
    monkeypatch.setattr(
        rqmc,
        "scipy_version",
        rqmc.DEVELOPMENT_SCIPY_VERSION,
    )
    monkeypatch.setattr(rqmc, "DEVELOPMENT_PROTOCOL_SHA256", "0" * 64)
    monkeypatch.setattr(
        rqmc,
        "run_case",
        lambda **_: pytest.fail("stale protocol must not execute a case"),
    )

    with pytest.raises(RuntimeError, match="protocol identity changed"):
        rqmc.run_screen(
            profile="development",
            case_id="near_gaussian__two_cell__root",
            include_timings=False,
        )


def test_one_case_cli_publishes_atomically_and_refuses_replacement(
    tmp_path: Path,
) -> None:
    """One Slurm-friendly case should publish once without a partial file."""
    output = tmp_path / "near_gaussian__two_cell__root.json"
    arguments = [
        "--profile",
        "smoke",
        "--case-id",
        "near_gaussian__two_cell__root",
        "--sample-counts",
        "64",
        "--repeat-seeds",
        "731",
        "--no-timings",
        "--output",
        str(output),
    ]

    assert rqmc.main(arguments) == 0
    payload = json.loads(output.read_text())
    assert payload["selected_case_id"] == ("near_gaussian__two_cell__root")
    assert payload["per_case_atomic_output"] is True
    assert [case["case_id"] for case in payload["cases"]] == ["near_gaussian__two_cell__root"]
    assert not list(tmp_path.glob(".*.tmp"))

    with pytest.raises(FileExistsError, match="refusing to replace"):
        rqmc.main(arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        ("--profile", "development", "--output", "unused.json", "--sample-counts", "64"),
        ("--profile", "development", "--output", "unused.json", "--repeat-seeds", "731"),
        ("--profile", "smoke", "--output", "unused.json", "--sample-counts", "96"),
        ("--profile", "smoke"),
    ],
)
def test_cli_invalid_contracts_leave_science_unrun(
    arguments: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI validation should stop before constructing an allocation bank."""
    monkeypatch.setattr(
        rqmc,
        "run_screen",
        lambda **_: pytest.fail("invalid CLI must not execute"),
    )

    with pytest.raises((SystemExit, ValueError)):
        rqmc.main(arguments)


def test_source_contains_no_protected_numerical_definitions() -> None:
    """The RQMC executable should import, not duplicate, A1 numeric cases."""
    source = Path(rqmc.__file__).read_text()

    assert "(0.35, 4.0)" not in source
    assert "(3.00, -0.20, 0.05, 0.80)" not in source
    assert "heldout" not in rqmc._parser().get_default("profile")


def test_direct_script_invocation_lists_the_frozen_matrix() -> None:
    """The Slurm-style script path should resolve its sibling C1 module."""
    completed = subprocess.run(
        [sys.executable, str(Path(rqmc.__file__)), "--list-matrix"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema"] == rqmc.SCHEMA
    assert payload["bank_construction_method"] == (rqmc.BANK_CONSTRUCTION_METHOD)
    assert payload["development"] == [list(case) for case in rqmc.DEVELOPMENT_MATRIX]
