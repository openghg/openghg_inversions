"""Focused tests for restartable residual-image GMM certification."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from examples.rjmcmc import conditional_residual_image_gmm_certify as certify
from examples.rjmcmc import conditional_residual_image_gmm_tiny_screen as gmm

REVISION = "a" * 40
_REAL_VALIDATE_EVALUATION = certify._validate_evaluation


@pytest.fixture(autouse=True)
def _accept_synthetic_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep bundle tests focused on certification rather than Git or model replay."""
    monkeypatch.setattr(certify, "_validate_live_source", lambda *_args: None)
    monkeypatch.setattr(gmm, "_validate_development_protocol", lambda: None)

    def validate_evaluation(
        evaluation: Mapping[str, Any],
        *,
        sample_count: int,
        base_seed: int,
        **_kwargs: object,
    ) -> dict[str, Any]:
        assert evaluation["sample_count"] == sample_count
        assert evaluation["base_seed"] == base_seed
        training = evaluation["training"]
        return {
            "pass": evaluation["scientific_pass"],
            "log_evidence": evaluation["posterior_summary"]["log_evidence"],
            "fitted_bundle_sha256": training["fitted_bundle_envelope"]["sha256"],
            "artifact_sha256": training["artifact_sha256"],
        }

    monkeypatch.setattr(certify, "_validate_evaluation", validate_evaluation)


def _digest(label: str) -> str:
    """Return a deterministic synthetic SHA-256 identity."""
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    """Write one canonical, newline-terminated shard."""
    path.write_text(f"{certify._canonical_json(payload)}\n", encoding="ascii")


def _report_identity(case_id: str) -> dict[str, Any]:
    """Return top-level identities common to both shard modes."""
    return {
        "schema": gmm.SCHEMA,
        "protocol": gmm.PROTOCOL,
        "profile": "development",
        "selected_case_id": case_id,
        "per_case_atomic_output": True,
        "source_git_revision": REVISION,
        "driver_sha256": certify._driver_source_sha256(),
        "a1_definitions_sha256": certify.c1.A1_DEFINITIONS_SHA256,
        "protocol_sha256": gmm._protocol_sha256(),
        "frozen_development_protocol_sha256": gmm.DEVELOPMENT_PROTOCOL_SHA256,
        "sample_counts": list(gmm.DEVELOPMENT_SAMPLE_COUNTS),
        "repeat_seeds": [gmm.DEVELOPMENT_SELECTION_SEED, *gmm.CONFIRMATION_SEEDS],
        "matrix_catalogue": gmm.matrix_catalogue(),
        "development_pass": False,
        "eligible_for_protected_holdout": False,
        "protected_holdout_pass": None,
        "scientific_pass": False,
        "scientific_pass_available": False,
        "structural_inference_licensed": False,
        "held_out_information_read": False,
    }


def _domain_artifacts(case_id: str) -> dict[str, dict[str, Any]]:
    """Return stable synthetic seed-731 domain identities for one case."""
    return {
        domain: {
            "sample_count": count,
            "source_seed": gmm._domain_seed(
                gmm.DEVELOPMENT_SELECTION_SEED,
                case_id=case_id,
                domain=domain,
            ),
            "artifact_sha256": _digest(f"{case_id}:{domain}:artifact"),
            "draws_sha256": _digest(f"{case_id}:{domain}:draws"),
        }
        for domain, count in {
            gmm.TRAINING_DOMAIN: max(gmm.DEVELOPMENT_SAMPLE_COUNTS),
            gmm.VALIDATION_DOMAIN: gmm.VALIDATION_SAMPLE_COUNT,
            gmm.TEST_DOMAIN: gmm.TEST_SAMPLE_COUNT,
        }.items()
    }


def _evaluation(
    case_id: str,
    *,
    sample_count: int,
    seed: int,
    evidence: float,
) -> dict[str, Any]:
    """Return a compact evaluation consumed by the patched model audit."""
    artifact_sha256 = _digest(f"{case_id}:{sample_count}:{seed}:artifact")
    envelope_sha256 = _digest(f"{case_id}:{sample_count}:{seed}:envelope")
    domains = _domain_artifacts(case_id)
    if seed != gmm.DEVELOPMENT_SELECTION_SEED:
        for domain, count in {
            gmm.TRAINING_DOMAIN: sample_count,
            gmm.VALIDATION_DOMAIN: gmm.VALIDATION_SAMPLE_COUNT,
            gmm.TEST_DOMAIN: gmm.TEST_SAMPLE_COUNT,
        }.items():
            domains[domain] = {
                "sample_count": count,
                "source_seed": gmm._domain_seed(seed, case_id=case_id, domain=domain),
                "artifact_sha256": _digest(f"{case_id}:{seed}:{domain}:artifact"),
                "draws_sha256": _digest(f"{case_id}:{seed}:{domain}:draws"),
            }
    return {
        "sample_count": sample_count,
        "base_seed": seed,
        "scientific_pass": True,
        "posterior_summary": {"log_evidence": evidence},
        "training": {
            "domain_artifacts": domains,
            "training_prefix_sha256": _digest(f"{case_id}:{sample_count}:{seed}:prefix"),
            "artifact_sha256": artifact_sha256,
            "fitted_bundle_envelope": {"sha256": envelope_sha256},
        },
    }


@pytest.fixture
def real_replayed_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, Any],
    certify._ExactCase,
    dict[str, Any],
    dict[tuple[str, int, int], certify.DomainBank],
]:
    """Build a bounded real artifact evaluation accepted by scientific replay."""
    monkeypatch.setattr(gmm, "scipy_version", gmm.DEVELOPMENT_SCIPY_VERSION)
    exact = certify._exact_case(gmm.DEVELOPMENT_MATRIX[0])
    dimension = exact.context.residual_rank
    sample_count = gmm.DEVELOPMENT_SAMPLE_COUNTS[0]
    base_seed = gmm.DEVELOPMENT_SELECTION_SEED
    phase = np.linspace(-1.0, 1.0, 24, dtype=np.float64)
    small_draws = np.column_stack([phase + 0.03 * coordinate for coordinate in range(dimension)])

    def small_domain_bundle(
        _aggregation: object,
        _labels: object,
        _context: object,
        *,
        case_id: str,
        training_sample_count: int,
        validation_sample_count: int,
        test_sample_count: int,
        base_seed: int,
    ) -> certify.DomainBank:
        """Return small draws with the frozen protocol's reported domain counts."""
        draws = {
            gmm.TRAINING_DOMAIN: small_draws,
            gmm.VALIDATION_DOMAIN: small_draws,
            gmm.TEST_DOMAIN: small_draws,
        }
        counts = {
            gmm.TRAINING_DOMAIN: training_sample_count,
            gmm.VALIDATION_DOMAIN: validation_sample_count,
            gmm.TEST_DOMAIN: test_sample_count,
        }
        artifacts = {
            domain: {
                "sample_count": counts[domain],
                "source_seed": gmm._domain_seed(base_seed, case_id=case_id, domain=domain),
                "artifact_sha256": _digest(f"{case_id}:{base_seed}:{domain}:small-bank"),
                "draws_sha256": gmm._array_sha256(domain_draws),
            }
            for domain, domain_draws in draws.items()
        }
        return draws, artifacts

    monkeypatch.setattr(gmm, "_domain_draw_bundle", small_domain_bundle)
    domain_draws, domain_artifacts = small_domain_bundle(
        exact.aggregation,
        exact.labels,
        exact.context,
        case_id=exact.case_id,
        training_sample_count=max(gmm.DEVELOPMENT_SAMPLE_COUNTS),
        validation_sample_count=gmm.VALIDATION_SAMPLE_COUNT,
        test_sample_count=gmm.TEST_SAMPLE_COUNT,
        base_seed=base_seed,
    )
    component_means = np.linspace(
        -0.8,
        0.8,
        gmm.COMPONENT_COUNT * dimension,
        dtype=np.float64,
    ).reshape(gmm.COMPONENT_COUNT, dimension)
    fit = gmm.GaussianMixtureFit(
        weights=np.full(gmm.COMPONENT_COUNT, 1.0 / gmm.COMPONENT_COUNT),
        means=component_means,
        covariances=np.repeat(
            np.eye(dimension, dtype=np.float64)[None, :, :],
            gmm.COMPONENT_COUNT,
            axis=0,
        ),
        initialization=0,
        iterations=1,
        training_mean_log_likelihood=-1.0,
        validation_mean_log_likelihood=-1.0,
        validation_nll=1.0,
        convergence_streak=gmm.CONVERGENCE_STREAK,
        objective_history=(-1.0,),
    )
    artifact = gmm._fit_as_zero_input_mdn(
        exact.context,
        fit,
        source_provenance="real certifier replay regression",
    )
    generalization = gmm._simulator_test_generalization(
        domain_draws[gmm.VALIDATION_DOMAIN],
        domain_draws[gmm.TEST_DOMAIN],
        fit,
    )
    attempts = [
        {
            "initialization": initialization,
            "status": "converged",
        }
        for initialization in range(gmm.INITIALIZATION_COUNT)
    ]
    prefix_sha256 = gmm._array_sha256(domain_draws[gmm.TRAINING_DOMAIN][:sample_count])
    envelope = gmm._fitted_bundle_envelope(
        artifact,
        case_id=exact.case_id,
        context_sha256=exact.context.artifact_sha256,
        source_git_revision=REVISION,
        driver_sha256=certify._driver_source_sha256(),
        domain_artifacts=domain_artifacts,
        training_prefix_sha256=prefix_sha256,
        training_sample_count=sample_count,
        attempts=attempts,
        selected_initialization=0,
        generalization=generalization,
    )
    replay = gmm._evaluate_artifact(
        artifact=artifact,
        observation=exact.observation,
        masses=exact.masses,
        log_prior=exact.log_prior,
        exact_log_likelihood=exact.exact_log_likelihood,
        exact_summary=exact.exact_summary,
        gradient_states=exact.gradient_states,
        validation_state_mask=exact.validation_state_mask,
    )
    replay["scientific_model_gates_pass"] = replay["scientific_pass"]
    replay["fit_development_pass"] = bool(generalization["pass"])
    replay["scientific_pass"] = bool(replay["scientific_model_gates_pass"] and replay["fit_development_pass"])
    replay["sample_count"] = sample_count
    replay["base_seed"] = base_seed
    replay["training"] = {
        "training_sample_count": sample_count,
        "training_prefix_sha256": prefix_sha256,
        "validation_sample_count": gmm.VALIDATION_SAMPLE_COUNT,
        "test_sample_count": gmm.TEST_SAMPLE_COUNT,
        "base_seed": base_seed,
        "domain_artifacts": domain_artifacts,
        "initialization_attempts": attempts,
        "valid_initialization_count": gmm.INITIALIZATION_COUNT,
        "minimum_valid_initializations": gmm.MINIMUM_VALID_INITIALIZATIONS,
        "valid_initialization_pass": True,
        "selected_initialization": 0,
        "iterations": fit.iterations,
        "convergence_streak": fit.convergence_streak,
        "training_mean_log_likelihood": fit.training_mean_log_likelihood,
        "validation_nll": generalization["validation_nll_nat_per_draw"],
        "test_nll": generalization["simulator_test_nll_nat_per_draw"],
        "simulator_test_generalization": generalization,
        "fit_development_pass": bool(generalization["pass"]),
        "artifact_sha256": artifact.artifact_sha256,
        "artifact_payload": artifact.payload,
        "fitted_bundle_envelope": envelope,
    }
    case_record = {
        "case_id": exact.case_id,
        "input_sha256": exact.input_sha256,
        "context_sha256": exact.context.artifact_sha256,
        "residual_image_rank": exact.context.residual_rank,
        "quadrature": exact.quadrature,
        "exact_posterior_summary": exact.exact_summary,
    }
    return replay, exact, case_record, {}


@pytest.mark.parametrize("corruption", ["metric", "posterior_evidence", "validation_test_nll"])
def test_real_scientific_replay_rejects_coherently_redigested_tampering(
    real_replayed_evaluation: tuple[
        dict[str, Any],
        certify._ExactCase,
        dict[str, Any],
        dict[tuple[str, int, int], certify.DomainBank],
    ],
    corruption: str,
) -> None:
    """Real exact and simulator replay must reject internally coherent tampering."""
    evaluation, exact, case_record, cache = real_replayed_evaluation
    kwargs = {
        "exact": exact,
        "case_record": case_record,
        "domain_bank_cache": cache,
        "sample_count": gmm.DEVELOPMENT_SAMPLE_COUNTS[0],
        "base_seed": gmm.DEVELOPMENT_SELECTION_SEED,
        "expected_source_revision": REVISION,
        "expected_driver_sha256": certify._driver_source_sha256(),
        "label": "real replay regression",
    }
    assert _REAL_VALIDATE_EVALUATION(evaluation, **kwargs)["artifact_sha256"]

    corrupted = copy.deepcopy(evaluation)
    if corruption == "metric":
        name = "median_absolute_conditional_log_likelihood_error_nat"
        corrupted["metrics"][name] += 0.001
        corrupted["checks"][name] = corrupted["metrics"][name] <= c1.THRESHOLDS[name]
    elif corruption == "posterior_evidence":
        corrupted["posterior_summary"]["log_evidence"] += 0.001
        evidence_error = abs(
            corrupted["posterior_summary"]["log_evidence"] - exact.exact_summary["log_evidence"]
        )
        corrupted["metrics"]["absolute_log_evidence_error_nat"] = evidence_error
        corrupted["checks"]["absolute_log_evidence_error_nat"] = (
            evidence_error <= c1.THRESHOLDS["absolute_log_evidence_error_nat"]
        )
    else:
        training = corrupted["training"]
        generalization = training["simulator_test_generalization"]
        generalization["validation_nll_nat_per_draw"] += 0.25
        generalization["simulator_test_nll_nat_per_draw"] += 0.25
        training["validation_nll"] = generalization["validation_nll_nat_per_draw"]
        training["test_nll"] = generalization["simulator_test_nll_nat_per_draw"]
        envelope = training["fitted_bundle_envelope"]
        envelope["payload"]["generalization"] = copy.deepcopy(generalization)
        envelope["sha256"] = c1._sha256_json(envelope["payload"])

    corrupted["scientific_model_gates_pass"] = all(corrupted["checks"].values())
    corrupted["scientific_pass"] = bool(
        corrupted["scientific_model_gates_pass"] and corrupted["fit_development_pass"]
    )
    with pytest.raises(ValueError, match="replay|simulator"):
        _REAL_VALIDATE_EVALUATION(corrupted, **kwargs)


def test_real_scientific_replay_accepts_scoped_cross_node_roundoff(
    real_replayed_evaluation: tuple[
        dict[str, Any],
        certify._ExactCase,
        dict[str, Any],
        dict[tuple[str, int, int], certify.DomainBank],
    ],
) -> None:
    """Observed BP1-sized NLL and gradient roundoff must remain replayable."""
    evaluation, exact, case_record, cache = real_replayed_evaluation
    rounded = copy.deepcopy(evaluation)
    rounded["gradient_audits"][0]["learned_coordinate_gradient"][0] += (
        0.5 * certify._SCIENTIFIC_REPLAY_ABS_TOL
    )
    training = rounded["training"]
    generalization = training["simulator_test_generalization"]
    nll_offset = 0.5 * certify._GENERALIZATION_NLL_REPLAY_ABS_TOL
    generalization["validation_nll_nat_per_draw"] += nll_offset
    generalization["simulator_test_nll_nat_per_draw"] += nll_offset
    generalization["absolute_nll_gap_nat_per_draw"] = abs(
        generalization["simulator_test_nll_nat_per_draw"] - generalization["validation_nll_nat_per_draw"]
    )
    training["validation_nll"] = generalization["validation_nll_nat_per_draw"]
    training["test_nll"] = generalization["simulator_test_nll_nat_per_draw"]
    envelope = training["fitted_bundle_envelope"]
    envelope["payload"]["generalization"] = copy.deepcopy(generalization)
    envelope["sha256"] = c1._sha256_json(envelope["payload"])

    audit = _REAL_VALIDATE_EVALUATION(
        rounded,
        exact=exact,
        case_record=case_record,
        domain_bank_cache=cache,
        sample_count=gmm.DEVELOPMENT_SAMPLE_COUNTS[0],
        base_seed=gmm.DEVELOPMENT_SELECTION_SEED,
        expected_source_revision=REVISION,
        expected_driver_sha256=certify._driver_source_sha256(),
        label="cross-node roundoff regression",
    )

    assert audit["artifact_sha256"] == training["artifact_sha256"]


def test_replay_tolerance_keeps_identities_exact_and_fails_near_gates() -> None:
    """Replay tolerances apply only to floats and never decide a gate."""
    replayed = {
        "value": 1.0,
        "count": 3,
        "pass": True,
        "nested": [0.25],
    }
    observed = copy.deepcopy(replayed)
    observed["value"] += 0.5 * certify._SCIENTIFIC_REPLAY_ABS_TOL
    observed["nested"][0] -= 0.5 * certify._SCIENTIFIC_REPLAY_ABS_TOL
    certify._require_replayed_science(observed, replayed, "roundoff")

    wrong_count = copy.deepcopy(observed)
    wrong_count["count"] = 4
    with pytest.raises(ValueError, match="authenticated replay"):
        certify._require_replayed_science(wrong_count, replayed, "identity")

    generalization = {
        "residual_dimension": 1,
        "validation_nll_nat_per_draw": 1.0,
        "simulator_test_nll_nat_per_draw": 1.02,
        "absolute_nll_gap_nat_per_draw": 0.02,
        "validation_nll_mcse_nat_per_draw": 0.0,
        "simulator_test_nll_mcse_nat_per_draw": 0.0,
        "pooled_nll_mcse_nat_per_draw": 0.0,
        "fixed_floor_nat_per_draw": 0.02,
        "threshold_nat_per_draw": 0.02,
        "pass": True,
    }
    with pytest.raises(ValueError, match="too close"):
        certify._require_replayed_generalization(
            generalization,
            generalization,
            label="ambiguous gate",
        )

    threshold = c1.THRESHOLDS["between_bank_log_evidence_range_nat"]
    with pytest.raises(ValueError, match="too close"):
        certify._four_bank_evidence_range_gate(
            [-2.0, -2.0 + threshold, -1.99, -1.98],
            label="ambiguous evidence range",
        )
    evidence_range, passed = certify._four_bank_evidence_range_gate(
        [-2.0, -2.0 + threshold / 2.0, -1.99, -1.98],
        label="separated evidence range",
    )
    assert evidence_range == pytest.approx(threshold / 2.0, abs=1.0e-15)
    assert passed is True


def _development_report(case_id: str, sample_count: int) -> dict[str, Any]:
    """Return one internally consistent development-size shard."""
    report = _report_identity(case_id)
    evaluation = _evaluation(
        case_id,
        sample_count=sample_count,
        seed=gmm.DEVELOPMENT_SELECTION_SEED,
        evidence=-1.0,
    )
    domains = evaluation["training"]["domain_artifacts"]
    report.update(
        {
            "execution_mode": "development_size_shard",
            "executed_development_sample_count": sample_count,
            "cases": [
                {
                    "case_id": case_id,
                    "profile": "development",
                    "input_sha256": _digest(f"{case_id}:input"),
                    "context_sha256": _digest(f"{case_id}:context"),
                    "exact_posterior_summary": {"log_evidence": -1.0},
                    "quadrature": {"mass_state_count": 3},
                    "executed_development_sample_count": sample_count,
                    "executed_confirmation_seed": None,
                    "development_evaluations": [evaluation],
                    "confirmation_evaluations": [],
                    "development_nested_training_bank": {
                        "largest_sample_count": max(gmm.DEVELOPMENT_SAMPLE_COUNTS),
                        "artifact_sha256": domains[gmm.TRAINING_DOMAIN]["artifact_sha256"],
                        "full_draws_sha256": domains[gmm.TRAINING_DOMAIN]["draws_sha256"],
                        "prefixes": {str(sample_count): evaluation["training"]["training_prefix_sha256"]},
                    },
                }
            ],
        }
    )
    return report


def _write_development_bundle(directory: Path) -> None:
    """Write the exact 24-file synthetic development bundle."""
    directory.mkdir()
    for case_id in certify._case_ids():
        for count in gmm.DEVELOPMENT_SAMPLE_COUNTS:
            _write_json(
                directory / certify._development_filename(case_id, count),
                _development_report(case_id, count),
            )


def _confirmation_report(
    case_id: str,
    seed: int,
    *,
    lock_internal_sha256: str,
    lock_raw_sha256: str,
    locked_sample_count: int,
) -> dict[str, Any]:
    """Return one lock-bound confirmation-seed shard."""
    report = _report_identity(case_id)
    report.update(
        {
            "execution_mode": "confirmation_seed_shard",
            "executed_confirmation_seed": seed,
            "confirmation_lock_internal_sha256": lock_internal_sha256,
            "confirmation_lock_raw_sha256": lock_raw_sha256,
            "confirmation_locked_sample_count": locked_sample_count,
            "cases": [
                {
                    "case_id": case_id,
                    "input_sha256": _digest(f"{case_id}:input"),
                    "context_sha256": _digest(f"{case_id}:context"),
                    "executed_development_sample_count": None,
                    "executed_confirmation_seed": seed,
                    "development_evaluations": [],
                    "confirmation_sample_count": locked_sample_count,
                    "confirmation_evaluations": [
                        _evaluation(
                            case_id,
                            sample_count=locked_sample_count,
                            seed=seed,
                            evidence=-1.0 + (seed % 3) * 0.01,
                        )
                    ],
                }
            ],
        }
    )
    return report


def _write_confirmation_bundle(directory: Path, lock_path: Path) -> str:
    """Write the exact 18-file confirmation bundle and return the raw lock digest."""
    lock_envelope, lock_raw_sha256 = certify._read_canonical_json(lock_path)
    lock_payload = lock_envelope["payload"]
    lock_internal_sha256 = lock_envelope["sha256"]
    locked_sample_count = lock_payload["locked_sample_count"]
    directory.mkdir()
    for case_id in certify._case_ids():
        for seed in gmm.CONFIRMATION_SEEDS:
            _write_json(
                directory / certify._confirmation_filename(case_id, seed),
                _confirmation_report(
                    case_id,
                    seed,
                    lock_internal_sha256=lock_internal_sha256,
                    lock_raw_sha256=lock_raw_sha256,
                    locked_sample_count=locked_sample_count,
                ),
            )
    return lock_raw_sha256


def test_two_phase_certification_publishes_bound_development_pass(tmp_path: Path) -> None:
    """The complete 24+18 matrix should publish a holdout-eligible certificate."""
    development = tmp_path / "development"
    confirmations = tmp_path / "confirmations"
    lock = tmp_path / "common_lock.json"
    certificate = tmp_path / "development_certificate.json"
    _write_development_bundle(development)

    lock_envelope = certify.merge_development(
        source_directory=Path(certify.__file__).resolve().parents[2],
        development_directory=development,
        output_lock=lock,
        expected_source_revision=REVISION,
    )
    raw_lock_sha256 = _write_confirmation_bundle(confirmations, lock)
    certificate_envelope = certify.certify_confirmation(
        source_directory=Path(certify.__file__).resolve().parents[2],
        development_directory=development,
        confirmation_directory=confirmations,
        common_lock=lock,
        output_certificate=certificate,
        expected_source_revision=REVISION,
        expected_lock_raw_sha256=raw_lock_sha256,
    )

    assert lock_envelope["payload"]["locked_sample_count"] == gmm.DEVELOPMENT_SAMPLE_COUNTS[0]
    payload = certificate_envelope["payload"]
    assert payload["execution_certified"] is True
    assert payload["development_pass"] is True
    assert payload["eligible_for_protected_holdout"] is True
    assert payload["scientific_pass"] is False
    assert payload["common_lock_raw_sha256"] == raw_lock_sha256
    assert len(payload["cases"]) == 6
    assert all(case["nominated_development_raw_sha256"] for case in payload["cases"])
    assert json.loads(certificate.read_text()) == certificate_envelope


@pytest.mark.parametrize("corruption", ["extra", "noncanonical", "wrong_lock_sha"])
def test_certifier_rejects_membership_encoding_and_lock_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    """Shard membership, canonical JSON, and the raw lock binding fail closed."""
    development = tmp_path / "development"
    lock = tmp_path / "common_lock.json"
    _write_development_bundle(development)
    if corruption == "extra":
        _write_json(development / "extra.json", {})
    elif corruption == "noncanonical":
        first = development / next(iter(sorted(path.name for path in development.iterdir())))
        first.write_text(json.dumps(json.loads(first.read_text()), indent=2), encoding="ascii")

    if corruption != "wrong_lock_sha":
        with pytest.raises(ValueError):
            certify.merge_development(
                source_directory=Path(certify.__file__).resolve().parents[2],
                development_directory=development,
                output_lock=lock,
                expected_source_revision=REVISION,
            )
        return

    certify.merge_development(
        source_directory=Path(certify.__file__).resolve().parents[2],
        development_directory=development,
        output_lock=lock,
        expected_source_revision=REVISION,
    )
    confirmations = tmp_path / "confirmations"
    _write_confirmation_bundle(confirmations, lock)
    with pytest.raises(ValueError, match="raw SHA-256"):
        certify.certify_confirmation(
            source_directory=Path(certify.__file__).resolve().parents[2],
            development_directory=development,
            confirmation_directory=confirmations,
            common_lock=lock,
            output_certificate=tmp_path / "certificate.json",
            expected_source_revision=REVISION,
            expected_lock_raw_sha256="0" * 64,
        )


def test_generalization_gate_is_recomputed() -> None:
    """Recorded generalization booleans cannot contradict their finite metrics."""
    evidence = {
        "residual_dimension": 2,
        "validation_nll_nat_per_draw": 1.0,
        "simulator_test_nll_nat_per_draw": 1.0,
        "absolute_nll_gap_nat_per_draw": 0.0,
        "validation_nll_mcse_nat_per_draw": 0.001,
        "simulator_test_nll_mcse_nat_per_draw": 0.001,
        "pooled_nll_mcse_nat_per_draw": math.hypot(0.001, 0.001),
        "fixed_floor_nat_per_draw": 0.04,
        "threshold_nat_per_draw": 0.04,
        "pass": True,
    }
    assert certify._validate_generalization(evidence, "synthetic") is True
    evidence["pass"] = False
    with pytest.raises(ValueError, match="pass is inconsistent"):
        certify._validate_generalization(evidence, "synthetic")


def test_certifier_refuses_to_overwrite_outputs(tmp_path: Path) -> None:
    """An existing output must fail before shard inspection."""
    output = tmp_path / "lock.json"
    output.write_text("existing\n", encoding="ascii")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        certify.merge_development(
            source_directory=Path(certify.__file__).resolve().parents[2],
            development_directory=tmp_path / "missing",
            output_lock=output,
            expected_source_revision=REVISION,
        )
