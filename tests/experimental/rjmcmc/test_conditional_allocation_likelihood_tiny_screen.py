"""Tests for the reusable C1 conditional-allocation tiny screen."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as screen
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mixture import (
    ConditionalAllocationMixture,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)


def _smoke_report() -> dict[str, Any]:
    """Return one fast, timing-free exact-versus-bank smoke result."""
    return screen.run_screen(
        profile="smoke",
        sample_counts=(64,),
        repeat_seeds=(731,),
        include_timings=False,
    )


def test_smoke_replays_and_passes_with_aggregate_metrics_only() -> None:
    """The fixed bank and exact oracle should replay a passing smoke report."""
    first = _smoke_report()
    second = _smoke_report()

    assert first == second
    assert first["schema"] == screen.SCHEMA
    assert first["scientific_pass"] is True
    assert first["held_out_information_read"] is False
    assert first["held_out_execution_available"] is False
    assert first["observed_residual_used_for_basis_selection"] is False
    assert first["structural_inference_licensed"] is False
    assert first["full_c1_pass"] is False
    assert first["a1_definitions_sha256"] == screen.A1_DEFINITIONS_SHA256
    assert len(first["cases"]) == 1
    case = first["cases"][0]
    assert case["summary_basis"] == {
        "kind": "identity",
        "rank": 3,
        "observation_count": 3,
        "selection": "fixed_full_rank_independent_of_observed_residual",
    }
    assert case["scientific_pass"] is True
    assert case["mass_grid"]["pointwise_gate_split"]["alters_evidence_or_posterior_quadrature"] is False
    assert case["locked_sample_count"] == 64
    assert case["development_lock_eligible"] is True
    evaluation = case["development_evaluations"][0]
    assert evaluation["build_seconds"] is None
    assert evaluation["evaluation_seconds"] is None
    assert evaluation["evaluation_states_per_second"] is None
    assert evaluation["scientific_pass_without_repeat_evidence_gate"] is True
    expected_metrics = {
        "median_absolute_conditional_log_likelihood_error_nat",
        "p99_absolute_conditional_log_likelihood_error_nat",
        "scaled_coordinate_gradient_error",
        "absolute_log_evidence_error_nat",
        "posterior_mean_error_reference_sd",
        "posterior_sd_relative_error",
        "interval_endpoint_error_reference_sd",
    }
    assert set(evaluation["metrics"]) == expected_metrics
    assert len(evaluation["gradient_audits"]) >= 3
    assert set(evaluation["posterior_summary"]["coordinates"]) == {
        "total_mass",
        "region_mass_0",
    }
    assert evaluation["diagnostics"]["pointwise_gate_weighting"]["median"] == (
        "normalized quadrature prior weights on the C1 development validation view"
    )
    serialized = json.dumps(first, allow_nan=False)
    assert "projected_unit_mass_residual_factors" not in serialized
    assert '"masses"' not in serialized


def test_bank_mass_gradient_uses_exact_coordinate_chain_rule() -> None:
    """Analytic bank gradients should match coordinate finite differences."""
    regime = screen.REGIMES[1]
    shapes, _, design, observation, noise = screen._case_arrays(
        regime,
        "four_cell",
    )
    labels = screen.labels_for_tiling("four_cell", "row")
    aggregation = AdditiveDirichletAggregation(
        shapes,
        design,
        noise,
        np.eye(observation.size),
    )
    artifact = ConditionalAllocationMixture.from_aggregation(
        aggregation,
        labels,
        sample_count=257,
        source_seed=5_023,
        source_provenance="C1 coordinate-chain unit test",
    )
    coordinate = np.asarray([math.log(1.1), math.log(1.7)])
    masses = screen.coordinate_to_masses(coordinate)
    _, mass_gradient = artifact.log_likelihood_and_mass_gradient(
        observation,
        masses,
    )
    analytic = screen.mass_gradient_to_coordinate_gradient(
        masses,
        mass_gradient,
    )
    observed = screen._centered_gradient(
        lambda value: artifact.log_likelihood(
            observation,
            screen.coordinate_to_masses(value),
        ),
        coordinate,
    )

    # The deliberately preserved A1 step is 2**-14, so truncation rather than
    # the analytic chain rule limits this comparison.
    np.testing.assert_allclose(analytic, observed, rtol=1.0e-6, atol=3.0e-9)


def test_cli_catalogue_is_blind_and_does_not_run_matrix(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matrix inspection should expose only an opaque held-out identity."""
    monkeypatch.setattr(
        screen,
        "run_screen",
        lambda **_: pytest.fail("matrix listing must not execute a screen"),
    )

    assert screen.main(["--list-matrix"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["held_out_information_read"] is False
    assert payload["development"] == [list(case) for case in screen.DEVELOPMENT_MATRIX]
    assert payload["held_out_catalogue"] == {
        "id": screen.HELD_OUT_CATALOGUE_ID,
        "sha256": screen.HELD_OUT_CATALOGUE_SHA256,
        "numerical_values_present": False,
        "executable_here": False,
    }
    with pytest.raises(SystemExit, match="cannot be combined"):
        screen.main(["--list-matrix", "--output", "forbidden.json"])


def test_source_revision_is_explicit_when_git_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compute nodes without Git should require and retain the expected SHA."""
    revision = "1" * 40
    monkeypatch.setattr(screen, "_git_revision", lambda: None)

    assert screen._source_revision(revision) == revision
    with pytest.raises(RuntimeError, match="required when Git is unavailable"):
        screen._source_revision(None)
    with pytest.raises(ValueError, match="40-character"):
        screen._source_revision("ABC")


def test_source_revision_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit SHA must agree with Git whenever Git is available."""
    monkeypatch.setattr(screen, "_git_revision", lambda: "1" * 40)

    with pytest.raises(RuntimeError, match="does not match"):
        screen._source_revision("2" * 40)


def test_cli_failure_leaves_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scientific or implementation exception must leave no partial JSON."""
    output = tmp_path / "result.json"

    def fail(**_: object) -> dict[str, Any]:
        raise RuntimeError("deliberate screen failure")

    monkeypatch.setattr(screen, "run_screen", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        screen.main(["--profile", "smoke", "--output", str(output)])

    assert not output.exists()
    assert not list(tmp_path.iterdir())


def test_atomic_output_refuses_to_replace_existing_file(
    tmp_path: Path,
) -> None:
    """A repeated task must preserve its first durable result."""
    output = tmp_path / "result.json"
    output.write_text('{"preserved":true}\n')

    with pytest.raises(FileExistsError, match="refusing to replace"):
        screen._write_atomic_json(output, {"preserved": False})

    assert output.read_text() == '{"preserved":true}\n'


@pytest.mark.parametrize(
    "arguments",
    [
        ("--profile", "heldout", "--output", "unused.json"),
        ("--profile", "smoke"),
        ("--profile", "smoke", "--output", "unused.json", "--sample-counts", "0"),
        (
            "--profile",
            "smoke",
            "--output",
            "unused.json",
            "--repeat-seeds",
            "1,1",
        ),
        (
            "--profile",
            "development",
            "--output",
            "unused.json",
            "--sample-counts",
            "64",
        ),
    ],
)
def test_cli_rejects_invalid_contracts_without_running(
    arguments: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Argument validation should fail before the scientific matrix runs."""
    monkeypatch.setattr(
        screen,
        "run_screen",
        lambda **_: pytest.fail("invalid CLI must not execute a screen"),
    )

    with pytest.raises(SystemExit):
        screen.main(arguments)


def test_development_case_selection_runs_only_one_atomic_case(
    tmp_path: Path,
) -> None:
    """A Slurm array cell should publish one independently mergeable JSON."""
    case_id = "near_gaussian__two_cell__root"
    output = tmp_path / f"{case_id}.json"

    assert (
        screen.main(
            [
                "--profile",
                "development",
                "--case-id",
                case_id,
                "--no-timings",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text())
    assert payload["selected_case_id"] == case_id
    assert payload["per_case_atomic_output"] is True
    assert [case["case_id"] for case in payload["cases"]] == [case_id]
    assert payload["sample_counts"] == list(screen.DEVELOPMENT_SAMPLE_COUNTS)
    assert payload["repeat_seeds"] == list(screen.DEVELOPMENT_REPEAT_SEEDS)


def test_development_source_contains_no_protected_numerical_definitions() -> None:
    """The development executable should expose only the held-out digest."""
    source = Path(screen.__file__).read_text()

    assert '"column": [0, 1, 0, 1]' not in source
    assert "(3.00, -0.20, 0.05, 0.80)" not in source
    assert {regime.name for regime in screen.REGIMES} == {
        "near_gaussian",
        "skewed",
        "boundary_heavy",
        "equal_footprint",
    }
    with pytest.raises(ValueError, match="does not support tiling"):
        screen.labels_for_tiling("four_cell", "column")  # type: ignore[arg-type]


def test_direct_development_case_rejects_protocol_overrides() -> None:
    """The public case helper must not bypass source-pinned development values."""
    with pytest.raises(
        ValueError,
        match="source-pinned sample counts and seeds",
    ):
        screen.run_case(
            regime_name="near_gaussian",
            family="two_cell",
            tiling="root",
            sample_counts=(64,),
            repeat_seeds=(731,),
            profile="development",
            include_timings=False,
        )


def test_bank_size_lock_precedes_independent_confirmation() -> None:
    """Confirmation seeds must run only at the smallest passing S."""
    report = screen.run_screen(
        profile="smoke",
        sample_counts=(64, 256),
        repeat_seeds=(731, 1_877, 4_099),
        include_timings=False,
    )
    case = report["cases"][0]

    assert [result["sample_count"] for result in case["development_evaluations"]] == [
        64,
        256,
    ]
    assert case["locked_sample_count"] == 64
    assert [result["sample_count"] for result in case["confirmation_evaluations"]] == [
        64,
        64,
    ]
    assert [result["source_seed"] for result in case["confirmation_evaluations"]] == [
        1_877,
        4_099,
    ]
    assert case["confirmation_can_retune_lock"] is False


def test_stable_lock_requires_a_long_enough_passing_suffix() -> None:
    """An isolated largest-bank pass must not certify convergence."""
    counts = (64, 256, 1_024, 4_096, 16_384)

    assert (
        screen._stable_lock_sample_count(
            counts,
            (False, False, True, True, True),
            minimum_suffix_length=2,
        )
        == 1_024
    )
    assert (
        screen._stable_lock_sample_count(
            counts,
            (False, True, False, False, True),
            minimum_suffix_length=2,
        )
        is None
    )


def test_weighted_quantile_and_evidence_use_declared_weights() -> None:
    """Independent arithmetic should reproduce weighted scoring primitives."""
    values = np.asarray([10.0, 20.0, 30.0])
    weights = np.asarray([0.1, 0.7, 0.2])

    assert screen._weighted_quantile(values, weights, 0.5) == 20.0
    masses = np.asarray([[1.0], [2.0], [3.0]])
    log_prior = np.log(weights)
    log_likelihood = np.log(np.asarray([0.5, 0.25, 0.125]))
    summary = screen._posterior_summary(
        masses,
        log_prior,
        log_likelihood,
    )
    expected_evidence = math.log(float(weights @ np.exp(log_likelihood)))

    assert summary["log_evidence"] == pytest.approx(expected_evidence)


def test_two_region_control_reports_every_retained_coordinate() -> None:
    """Two-region summaries should include both masses, share, and log ratio."""
    case = screen.run_case(
        regime_name="near_gaussian",
        family="two_cell",
        tiling="fine",
        sample_counts=(64,),
        repeat_seeds=(731,),
        profile="control",
        include_timings=False,
    )
    summary = case["development_evaluations"][0]["posterior_summary"]

    assert set(summary["coordinates"]) == {
        "total_mass",
        "region_mass_0",
        "region_mass_1",
        "first_region_share",
        "log_first_to_second_region_mass_ratio",
    }
    assert case["development_lock_eligible"] is False
    assert case["common_native_projection"]["bank_posterior_summary_available"] is False
