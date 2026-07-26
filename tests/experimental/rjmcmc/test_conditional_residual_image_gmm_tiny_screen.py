"""Tests for the root-only residual-image Gaussian-mixture screen."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.rjmcmc import conditional_allocation_likelihood_tiny_screen as c1
from examples.rjmcmc import conditional_residual_image_gmm_tiny_screen as gmm
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mdn import (
    ResidualImageContext,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (
    AdditiveDirichletAggregation,
)


def _root_problem() -> tuple[
    AdditiveDirichletAggregation,
    np.ndarray,
    ResidualImageContext,
]:
    """Return the smallest source-pinned root residual-image problem."""
    regime = c1._regime("near_gaussian")
    shapes, _, design, observation, noise = c1._case_arrays(
        regime,
        "two_cell",
    )
    labels = c1.labels_for_tiling("two_cell", "root")
    aggregation = AdditiveDirichletAggregation(
        shapes,
        design,
        noise,
        np.eye(observation.size, dtype=np.float64),
    )
    context = ResidualImageContext.from_aggregation(
        aggregation,
        labels,
        np.arange(shapes.size, dtype=np.int64),
        source_provenance="root-GMM unit-test context",
    )
    return aggregation, labels, context


def _separated_training_samples() -> np.ndarray:
    """Return eight well-separated deterministic clusters for quick EM."""
    clusters: list[np.ndarray] = []
    phase = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
    for index in range(gmm.COMPONENT_COUNT):
        clusters.append(
            np.column_stack(
                (
                    4.0 * index + 0.03 * np.cos(phase),
                    (-1.0) ** index + 0.03 * np.sin(phase),
                )
            )
        )
    return np.asarray(np.concatenate(clusters, axis=0), dtype=np.float64)


def _manual_fit(dimension: int) -> gmm.GaussianMixtureFit:
    """Return a finite eight-component fit suitable for export tests."""
    weights = np.arange(1, gmm.COMPONENT_COUNT + 1, dtype=np.float64)
    weights /= weights.sum()
    means = np.linspace(
        -0.8,
        0.9,
        gmm.COMPONENT_COUNT * dimension,
        dtype=np.float64,
    ).reshape(gmm.COMPONENT_COUNT, dimension)
    covariances = np.empty(
        (gmm.COMPONENT_COUNT, dimension, dimension),
        dtype=np.float64,
    )
    for component in range(gmm.COMPONENT_COUNT):
        lower = np.eye(dimension, dtype=np.float64) * (0.18 + 0.03 * component)
        if dimension > 1:
            lower[np.tril_indices(dimension, k=-1)] = 0.004 * (component + 1)
        covariances[component] = lower @ lower.T
    return gmm.GaussianMixtureFit(
        weights=weights,
        means=means,
        covariances=covariances,
        initialization=1,
        iterations=12,
        training_mean_log_likelihood=-1.2,
        validation_mean_log_likelihood=-1.3,
        validation_nll=1.3,
        convergence_streak=gmm.CONVERGENCE_STREAK,
        objective_history=(-1.5, -1.3, -1.2),
    )


def _manual_fitted_envelope() -> dict[str, Any]:
    """Return one valid fitted envelope without running EM."""
    _, _, context = _root_problem()
    artifact = gmm._fit_as_zero_input_mdn(
        context,
        _manual_fit(context.residual_rank),
        source_provenance="manual fitted-envelope test",
    )
    domains = {
        domain: {
            "sample_count": 16 * (index + 1),
            "source_seed": index + 1,
            "artifact_sha256": f"{index + 3:064x}",
            "draws_sha256": f"{index + 6:064x}",
        }
        for index, domain in enumerate(
            (
                gmm.TRAINING_DOMAIN,
                gmm.VALIDATION_DOMAIN,
                gmm.TEST_DOMAIN,
            )
        )
    }
    attempts = [
        {
            "initialization": initialization,
            "status": "converged",
            "iterations": 12 + initialization,
            "training_mean_log_likelihood": -1.2,
            "validation_nll": 1.3,
            "objective_history": [-1.5, -1.3, -1.2],
        }
        for initialization in range(gmm.INITIALIZATION_COUNT)
    ]
    return gmm._fitted_bundle_envelope(
        artifact,
        case_id="near_gaussian__two_cell__root",
        context_sha256=context.artifact_sha256,
        source_git_revision="1" * 40,
        driver_sha256="2" * 64,
        domain_artifacts=domains,
        training_prefix_sha256="a" * 64,
        training_sample_count=16,
        attempts=attempts,
        selected_initialization=1,
        generalization={
            "residual_dimension": context.residual_rank,
            "pass": True,
        },
    )


def _fake_matrix_case_runner(
    pass_patterns: dict[str, tuple[bool, bool, bool, bool]],
    *,
    confirmation_pass_by_case: dict[str, bool] | None = None,
    confirmation_log_evidence_by_case: (dict[str, tuple[float, float, float]] | None) = None,
    calls: list[dict[str, Any]] | None = None,
) -> Any:
    """Return a lightweight ``run_case`` replacement for common-lock tests."""
    confirmation_results = {} if confirmation_pass_by_case is None else confirmation_pass_by_case
    confirmation_evidence = (
        {} if confirmation_log_evidence_by_case is None else confirmation_log_evidence_by_case
    )

    def fake_run_case(**kwargs: Any) -> dict[str, Any]:
        regime_name = str(kwargs["regime_name"])
        family = str(kwargs["family"])
        case_id = f"{regime_name}__{family}__root"
        sample_counts = tuple(kwargs["sample_counts"])
        repeat_seeds = tuple(kwargs["repeat_seeds"])
        run_ladder = bool(kwargs.get("run_development_ladder", True))
        confirmation_sample_count = kwargs.get("confirmation_sample_count")
        if calls is not None:
            calls.append(
                {
                    "case_id": case_id,
                    "run_development_ladder": run_ladder,
                    "confirmation_sample_count": confirmation_sample_count,
                }
            )
        context_sha256 = f"context:{case_id}"
        input_sha256 = f"input:{case_id}"
        if run_ladder:
            evaluations = [
                {
                    "sample_count": sample_count,
                    "scientific_pass": passed,
                    "posterior_summary": {"log_evidence": 0.0},
                }
                for sample_count, passed in zip(
                    sample_counts,
                    pass_patterns[case_id],
                    strict=True,
                )
            ]
            return {
                "case_id": case_id,
                "context_sha256": context_sha256,
                "input_sha256": input_sha256,
                "development_evaluations": evaluations,
                "confirmation_sample_count": None,
                "confirmation_evaluations": [],
                "confirmation_pass": None,
                "scientific_pass": False,
            }
        assert isinstance(confirmation_sample_count, int)
        passes = confirmation_results.get(case_id, True)
        log_evidence = confirmation_evidence.get(case_id, (0.0, 0.0, 0.0))
        confirmation_evaluations = [
            {
                "sample_count": confirmation_sample_count,
                "base_seed": base_seed,
                "scientific_pass": passes,
                "posterior_summary": {
                    "log_evidence": evidence,
                },
                "training": {"artifact_sha256": (f"{case_id}:{confirmation_sample_count}:{base_seed}")},
            }
            for base_seed, evidence in zip(
                repeat_seeds[1:],
                log_evidence,
                strict=True,
            )
        ]
        return {
            "case_id": case_id,
            "context_sha256": context_sha256,
            "input_sha256": input_sha256,
            "development_evaluations": [],
            "confirmation_sample_count": confirmation_sample_count,
            "confirmation_evaluations": confirmation_evaluations,
            "confirmation_pass": passes,
            "scientific_pass": passes,
        }

    return fake_run_case


def _all_case_pass_patterns(
    pattern: tuple[bool, bool, bool, bool],
) -> dict[str, tuple[bool, bool, bool, bool]]:
    """Assign one development pass pattern to all six frozen cases."""
    return {"__".join(case): pattern for case in gmm.DEVELOPMENT_MATRIX}


def test_protocol_constants_and_six_case_matrix_are_source_pinned() -> None:
    """The executable matrix and data budgets must match the frozen plan."""
    assert gmm.SCHEMA == "rjmcmc-conditional-residual-image-gmm-tiny-screen-v1"
    assert gmm.PROTOCOL == "conditional-residual-image-root-full-covariance-gmm-v1"
    assert gmm.CONSTRUCTION_METHOD == "scrambled_sobol_balanced_dirichlet"
    assert gmm.COMPONENT_COUNT == 8
    assert gmm.INITIALIZATION_COUNT == 3
    assert gmm.MINIMUM_VALID_INITIALIZATIONS == 2
    assert gmm.DEVELOPMENT_SAMPLE_COUNTS == (
        4_096,
        16_384,
        65_536,
        262_144,
    )
    assert gmm.VALIDATION_SAMPLE_COUNT == 65_536
    assert gmm.TEST_SAMPLE_COUNT == 131_072
    assert gmm.PROTECTED_HOLDOUT_SAMPLE_COUNT == 131_072
    assert gmm.DEVELOPMENT_SELECTION_SEED == 731
    assert gmm.CONFIRMATION_SEEDS == (1_877, 4_099, 8_317)
    assert len(gmm.DEVELOPMENT_MATRIX) == 6
    assert set(gmm.DEVELOPMENT_MATRIX) == {
        (regime, family, "root")
        for regime in ("near_gaussian", "skewed", "boundary_heavy")
        for family in ("two_cell", "four_cell")
    }

    catalogue = gmm.matrix_catalogue()
    assert catalogue["development"] == [list(case) for case in gmm.DEVELOPMENT_MATRIX]
    assert catalogue["held_out_information_read"] is False
    assert catalogue["protected_holdout"] == {
        "id": gmm.PROTECTED_HOLDOUT_CATALOGUE_ID,
        "sha256": gmm.PROTECTED_HOLDOUT_CATALOGUE_SHA256,
        "sample_count": gmm.PROTECTED_HOLDOUT_SAMPLE_COUNT,
        "numerical_values_present": False,
        "executable_here": False,
    }
    assert len(gmm.PROTECTED_HOLDOUT_CATALOGUE_SHA256) == 64
    assert gmm._protocol_sha256() == gmm.DEVELOPMENT_PROTOCOL_SHA256


def test_fitted_bundle_envelope_replays_and_binds_all_training_evidence() -> None:
    """The authenticated envelope must replay its portable fitted density."""
    envelope = _manual_fitted_envelope()

    artifact = gmm.validate_fitted_bundle_envelope(
        envelope,
        expected_sha256=envelope["sha256"],
        expected_source_git_revision="1" * 40,
        expected_driver_sha256="2" * 64,
    )

    assert artifact.artifact_sha256 == envelope["payload"]["artifact"]["sha256"]
    assert envelope["payload"]["protocol"]["protected_holdout_catalogue_sha256"] == (
        gmm.PROTECTED_HOLDOUT_CATALOGUE_SHA256
    )
    assert set(envelope["payload"]["domains"]) == {
        gmm.TRAINING_DOMAIN,
        gmm.VALIDATION_DOMAIN,
        gmm.TEST_DOMAIN,
    }
    assert envelope["payload"]["training"]["valid_initialization_count"] == 3
    assert envelope["payload"]["training"]["training_prefix_sha256"] == "a" * 64


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("protocol", "name", "post-hoc-protocol", "protocol identity"),
        ("runtime", "numpy_version", "", "runtime identity"),
        ("training", "initialization_attempts", [], "initialization evidence"),
    ],
)
def test_fitted_bundle_envelope_rejects_semantic_tampering(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    """Re-digesting malformed metadata must not bypass semantic checks."""
    envelope = copy.deepcopy(_manual_fitted_envelope())
    envelope["payload"][section][field] = value
    envelope["sha256"] = c1._sha256_json(envelope["payload"])

    with pytest.raises(ValueError, match=message):
        gmm.validate_fitted_bundle_envelope(envelope)


def test_trusted_fitted_bundle_digest_rejects_any_rehashed_tamper() -> None:
    """A trusted envelope digest must bind source and simulator split identities."""
    envelope = _manual_fitted_envelope()
    trusted_sha256 = envelope["sha256"]
    tampered = copy.deepcopy(envelope)
    tampered["payload"]["domains"][gmm.TEST_DOMAIN]["draws_sha256"] = "f" * 64
    tampered["sha256"] = c1._sha256_json(tampered["payload"])

    with pytest.raises(ValueError, match="expected_sha256"):
        gmm.validate_fitted_bundle_envelope(
            tampered,
            expected_sha256=trusted_sha256,
        )


def test_em_replays_exactly_with_eight_finite_spd_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic start must replay and retain valid covariances."""
    monkeypatch.setattr(gmm, "CONVERGENCE_NAT_PER_DRAW", 1.0e-6)
    samples = _separated_training_samples()
    validation = np.array(samples[::-1], copy=True)

    first = gmm._fit_one_initialization(
        samples,
        validation,
        initialization=0,
    )
    second = gmm._fit_one_initialization(
        samples,
        validation,
        initialization=0,
    )

    assert first.weights.shape == (gmm.COMPONENT_COUNT,)
    assert first.means.shape == (gmm.COMPONENT_COUNT, 2)
    assert first.covariances.shape == (gmm.COMPONENT_COUNT, 2, 2)
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.means, second.means)
    np.testing.assert_array_equal(first.covariances, second.covariances)
    assert first.objective_history == second.objective_history
    assert np.all(first.weights > 0.0)
    assert np.isclose(first.weights.sum(), 1.0, rtol=0.0, atol=1.0e-14)
    assert np.all(np.isfinite(first.means))
    assert np.all(np.isfinite(first.covariances))
    for covariance in first.covariances:
        np.linalg.cholesky(covariance)


def test_em_empty_component_branch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exactly empty EM component must stop instead of being repaired."""
    samples = _separated_training_samples()
    validation = np.array(samples[::-1], copy=True)

    def empty_component_log_densities(
        values: np.ndarray,
        _weights: np.ndarray,
        _means: np.ndarray,
        _covariances: np.ndarray,
    ) -> np.ndarray:
        result = np.zeros(
            (values.shape[0], gmm.COMPONENT_COUNT),
            dtype=np.float64,
        )
        result[:, -1] = -np.inf
        return result

    monkeypatch.setattr(
        gmm,
        "_component_log_densities",
        empty_component_log_densities,
    )

    with pytest.raises(FloatingPointError, match="empty or non-finite"):
        gmm._fit_one_initialization(
            samples,
            validation,
            initialization=0,
        )


def test_validation_test_generalization_gate_passes_and_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The independent simulator test must be able to veto a fitted model."""
    fit = _manual_fit(1)
    validation = np.zeros((16, 1), dtype=np.float64)
    simulator_test = np.ones((32, 1), dtype=np.float64)

    monkeypatch.setattr(
        gmm,
        "_mixture_log_likelihood_values",
        lambda samples, *_: np.full(samples.shape[0], -1.0, dtype=np.float64),
    )
    passing = gmm._simulator_test_generalization(
        validation,
        simulator_test,
        fit,
    )
    assert passing["pass"] is True
    assert passing["absolute_nll_gap_nat_per_draw"] == 0.0
    assert passing["threshold_nat_per_draw"] == pytest.approx(gmm.GENERALIZATION_NAT_PER_DIMENSION)

    monkeypatch.setattr(
        gmm,
        "_mixture_log_likelihood_values",
        lambda samples, *_: np.full(
            samples.shape[0],
            -1.0 if np.all(samples == 0.0) else -4.0,
            dtype=np.float64,
        ),
    )
    failing = gmm._simulator_test_generalization(
        validation,
        simulator_test,
        fit,
    )
    assert failing["pass"] is False
    assert failing["absolute_nll_gap_nat_per_draw"] == pytest.approx(3.0)


def test_less_than_two_valid_initializations_cannot_pass_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected fit is insufficient unless at least two starts converge."""
    _, _, context = _root_problem()
    fit = _manual_fit(context.residual_rank)
    attempts = [
        {
            "initialization": 0,
            "status": "failed",
            "reason": "synthetic failure",
        },
        {
            "initialization": 1,
            "status": "converged",
            "iterations": 12,
            "training_mean_log_likelihood": -1.2,
            "validation_nll": 1.3,
            "objective_history": [-1.5, -1.3, -1.2],
        },
        {
            "initialization": 2,
            "status": "failed",
            "reason": "synthetic failure",
        },
    ]
    monkeypatch.setattr(
        gmm,
        "fit_gaussian_mixture",
        lambda *_: (fit, attempts),
    )
    monkeypatch.setattr(
        gmm,
        "_simulator_test_generalization",
        lambda *_: {
            "residual_dimension": context.residual_rank,
            "simulator_test_nll_nat_per_draw": 1.3,
            "pass": True,
        },
    )
    draws = np.zeros((16, context.residual_rank), dtype=np.float64)
    domain_artifacts = {
        domain: {
            "sample_count": 16,
            "source_seed": index,
            "artifact_sha256": f"{index + 1:064x}",
            "draws_sha256": f"{index + 4:064x}",
        }
        for index, domain in enumerate(
            (
                gmm.TRAINING_DOMAIN,
                gmm.VALIDATION_DOMAIN,
                gmm.TEST_DOMAIN,
            )
        )
    }

    _, report = gmm._fit_training_bundle(
        context,
        case_id="near_gaussian__two_cell__root",
        training_draws=draws,
        validation_draws=draws,
        test_draws=draws,
        domain_artifacts=domain_artifacts,
        sample_count=16,
        base_seed=731,
        source_git_revision="1" * 40,
        driver_sha256="2" * 64,
    )

    assert report["valid_initialization_count"] == 1
    assert report["valid_initialization_pass"] is False
    assert report["fit_development_pass"] is False


def test_zero_input_mdn_export_preserves_the_raw_gmm_density() -> None:
    """Portable export must not change any fitted mixture density."""
    _, _, context = _root_problem()
    fit = _manual_fit(context.residual_rank)
    artifact = gmm._fit_as_zero_input_mdn(
        context,
        fit,
        source_provenance="root-GMM export parity",
    )
    masses = np.asarray([1.7], dtype=np.float64)
    log_weights, means, factors = artifact._components(masses)
    exported_covariances = factors @ np.swapaxes(factors, 1, 2)

    np.testing.assert_allclose(
        np.exp(log_weights),
        fit.weights,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(means, fit.means)
    np.testing.assert_allclose(
        exported_covariances,
        fit.covariances,
        rtol=3.0e-15,
        atol=3.0e-17,
    )
    coordinates = np.linspace(
        -2.0,
        2.0,
        13 * context.residual_rank,
        dtype=np.float64,
    ).reshape(13, context.residual_rank)
    expected = gmm._stable_logsumexp_rows(
        gmm._component_log_densities(
            coordinates,
            fit.weights,
            fit.means,
            fit.covariances,
        )
    )
    observed = gmm._stable_logsumexp_rows(
        gmm._component_log_densities(
            coordinates,
            np.exp(log_weights),
            means,
            exported_covariances,
        )
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2.0e-14)


def test_training_sobol_prefixes_replay_and_data_domains_are_disjoint() -> None:
    """Training ladders must nest while split/domain keys remain distinct."""
    aggregation, labels, context = _root_problem()
    case_id = "near_gaussian__two_cell__root"
    training_seed = gmm._domain_seed(
        731,
        case_id=case_id,
        domain=gmm.TRAINING_DOMAIN,
    )
    replay_seed = gmm._domain_seed(
        731,
        case_id=case_id,
        domain=gmm.TRAINING_DOMAIN,
    )
    domain_seeds = {
        domain: gmm._domain_seed(731, case_id=case_id, domain=domain)
        for domain in (
            gmm.TRAINING_DOMAIN,
            gmm.VALIDATION_DOMAIN,
            gmm.TEST_DOMAIN,
        )
    }
    another_case_seed = gmm._domain_seed(
        731,
        case_id="skewed__two_cell__root",
        domain=gmm.TRAINING_DOMAIN,
    )
    another_split_seed = gmm._domain_seed(
        1_877,
        case_id=case_id,
        domain=gmm.TRAINING_DOMAIN,
    )

    assert training_seed == replay_seed
    assert len(set(domain_seeds.values())) == 3
    assert another_case_seed not in set(domain_seeds.values())
    assert another_split_seed not in set(domain_seeds.values())
    small, _ = gmm._residual_image_draws(
        aggregation,
        labels,
        context,
        sample_count=16,
        source_seed=training_seed,
        source_provenance="nested training prefix S=16",
    )
    replay, _ = gmm._residual_image_draws(
        aggregation,
        labels,
        context,
        sample_count=16,
        source_seed=training_seed,
        source_provenance="nested training prefix replay",
    )
    large, _ = gmm._residual_image_draws(
        aggregation,
        labels,
        context,
        sample_count=64,
        source_seed=training_seed,
        source_provenance="nested training prefix S=64",
    )
    validation, _ = gmm._residual_image_draws(
        aggregation,
        labels,
        context,
        sample_count=16,
        source_seed=domain_seeds[gmm.VALIDATION_DOMAIN],
        source_provenance="disjoint validation split",
    )

    np.testing.assert_array_equal(small, replay)
    np.testing.assert_array_equal(small, large[: small.shape[0]])
    assert not np.array_equal(small, validation)
    with pytest.raises(ValueError, match="protected or unknown"):
        gmm._domain_seed(
            731,
            case_id=case_id,
            domain=gmm.PROTECTED_HOLDOUT_CATALOGUE_ID,
        )


def test_training_bundle_uses_fixed_validation_and_test_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the nested training bank may vary over the development ladder."""
    aggregation, labels, context = _root_problem()
    calls: list[tuple[str, int, int]] = []

    def fake_draws(
        _aggregation: AdditiveDirichletAggregation,
        _labels: np.ndarray,
        _context: ResidualImageContext,
        *,
        sample_count: int,
        source_seed: int,
        source_provenance: str,
    ) -> tuple[np.ndarray, str]:
        domain = source_provenance.rsplit("=", maxsplit=1)[-1]
        calls.append((domain, sample_count, source_seed))
        return (
            np.linspace(
                -1.0,
                1.0,
                16 * context.residual_rank,
                dtype=np.float64,
            ).reshape(16, context.residual_rank),
            f"{len(calls):064x}",
        )

    monkeypatch.setattr(gmm, "_residual_image_draws", fake_draws)

    draws, artifacts = gmm._domain_draw_bundle(
        aggregation,
        labels,
        context,
        case_id="near_gaussian__two_cell__root",
        training_sample_count=4_096,
        validation_sample_count=gmm.VALIDATION_SAMPLE_COUNT,
        test_sample_count=gmm.TEST_SAMPLE_COUNT,
        base_seed=731,
    )

    assert [(domain, count) for domain, count, _ in calls] == [
        (gmm.TRAINING_DOMAIN, 4_096),
        (gmm.VALIDATION_DOMAIN, gmm.VALIDATION_SAMPLE_COUNT),
        (gmm.TEST_DOMAIN, gmm.TEST_SAMPLE_COUNT),
    ]
    assert len({seed for _, _, seed in calls}) == 3
    assert set(draws) == {
        gmm.TRAINING_DOMAIN,
        gmm.VALIDATION_DOMAIN,
        gmm.TEST_DOMAIN,
    }
    assert {domain: artifact["sample_count"] for domain, artifact in artifacts.items()} == {
        gmm.TRAINING_DOMAIN: 4_096,
        gmm.VALIDATION_DOMAIN: gmm.VALIDATION_SAMPLE_COUNT,
        gmm.TEST_DOMAIN: gmm.TEST_SAMPLE_COUNT,
    }


def test_protected_holdout_and_development_overrides_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither held-out values nor post-declaration development tuning may run."""
    monkeypatch.setattr(
        gmm,
        "run_case",
        lambda **_: pytest.fail("invalid protocol must stop before science"),
    )
    with pytest.raises(ValueError, match="deliberately unavailable"):
        gmm.run_screen(profile="protected")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source-pinned"):
        gmm.run_screen(
            profile="development",
            sample_counts=(4_096,),
        )
    with pytest.raises(ValueError, match="source-pinned"):
        gmm.run_screen(
            profile="development",
            repeat_seeds=(731,),
        )


def test_suffix_lock_rejects_an_isolated_largest_bank_pass() -> None:
    """One pass at the largest training size must remain a hard stop."""
    counts = gmm.DEVELOPMENT_SAMPLE_COUNTS
    assert (
        c1._stable_lock_sample_count(
            counts,
            (False, False, True, True),
            minimum_suffix_length=2,
        )
        == 65_536
    )
    assert (
        c1._stable_lock_sample_count(
            counts,
            (False, False, False, True),
            minimum_suffix_length=2,
        )
        is None
    )


def test_common_lock_uses_the_all_case_passing_suffix_and_one_sample_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All six cases must confirm at the first common two-size suffix."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        gmm,
        "run_case",
        _fake_matrix_case_runner(
            _all_case_pass_patterns((False, True, True, True)),
            calls=calls,
        ),
    )

    report = gmm.run_screen(profile="development")

    assert report["common_training_lock"]["status"] == "locked_and_confirmed"
    assert report["common_training_lock"]["locked_sample_count"] == 16_384
    assert report["common_training_lock"]["development_pass_pattern"] == [
        {"sample_count": 4_096, "pass": False},
        {"sample_count": 16_384, "pass": True},
        {"sample_count": 65_536, "pass": True},
        {"sample_count": 262_144, "pass": True},
    ]
    confirmation_calls = [call for call in calls if not call["run_development_ladder"]]
    assert len(confirmation_calls) == len(gmm.DEVELOPMENT_MATRIX)
    assert {call["confirmation_sample_count"] for call in confirmation_calls} == {16_384}
    assert report["common_confirmation_evidence"]["complete"] is True
    assert report["common_confirmation_evidence"]["all_cases_pass"] is True
    assert report["development_pass"] is True
    assert report["eligible_for_protected_holdout"] is True
    assert report["protected_holdout_pass"] is None
    assert report["scientific_pass"] is False
    assert report["scientific_pass_available"] is False
    for case in report["cases"]:
        assert case["confirmation_sample_count"] == 16_384
        assert [entry["sample_count"] for entry in case["confirmation_evaluations"]] == [
            16_384,
            16_384,
            16_384,
        ]
        assert [entry["base_seed"] for entry in case["confirmation_evaluations"]] == list(
            gmm.CONFIRMATION_SEEDS
        )


def test_one_late_case_prevents_an_earlier_common_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-case early suffix cannot override a later common suffix."""
    patterns = _all_case_pass_patterns((False, True, True, True))
    patterns["boundary_heavy__four_cell__root"] = (
        False,
        False,
        True,
        True,
    )
    monkeypatch.setattr(
        gmm,
        "run_case",
        _fake_matrix_case_runner(patterns),
    )

    report = gmm.run_screen(profile="development")

    assert report["common_training_lock"]["development_pass_pattern"] == [
        {"sample_count": 4_096, "pass": False},
        {"sample_count": 16_384, "pass": False},
        {"sample_count": 65_536, "pass": True},
        {"sample_count": 262_144, "pass": True},
    ]
    assert report["common_training_lock"]["locked_sample_count"] == 65_536
    assert report["common_training_lock"]["status"] == "locked_and_confirmed"
    assert {case["confirmation_sample_count"] for case in report["cases"]} == {65_536}
    assert report["development_pass"] is True
    assert report["eligible_for_protected_holdout"] is True
    assert report["protected_holdout_pass"] is None
    assert report["scientific_pass"] is False


def test_isolated_largest_all_case_pass_is_a_global_hard_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single all-case pass at the largest size cannot trigger confirmation."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        gmm,
        "run_case",
        _fake_matrix_case_runner(
            _all_case_pass_patterns((False, False, False, True)),
            calls=calls,
        ),
    )

    report = gmm.run_screen(profile="development")

    assert report["common_training_lock"]["status"] == ("hard_stop_isolated_largest_size_pass")
    assert report["common_training_lock"]["locked_sample_count"] is None
    assert report["common_confirmation_evidence"] == {
        "requested": False,
        "complete": False,
        "all_cases_pass": None,
        "cases": {},
    }
    assert all(call["run_development_ladder"] for call in calls)
    assert report["scientific_pass"] is False


def test_partial_development_case_cannot_lock_confirm_or_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One Slurm case is evidence for aggregation, not a certifiable screen."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        gmm,
        "run_case",
        _fake_matrix_case_runner(
            _all_case_pass_patterns((True, True, True, True)),
            calls=calls,
        ),
    )

    report = gmm.run_screen(
        profile="development",
        case_id="near_gaussian__two_cell__root",
    )

    assert len(report["cases"]) == 1
    assert report["common_training_lock"]["status"] == ("unavailable_partial_matrix")
    assert report["common_training_lock"]["locked_sample_count"] is None
    assert report["common_confirmation_evidence"]["requested"] is False
    assert len(calls) == 1
    assert calls[0]["run_development_ladder"] is True
    assert report["scientific_pass"] is False


def test_every_common_size_confirmation_is_required_for_global_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One failed case confirmation must invalidate an otherwise common lock."""
    confirmation_passes = {"__".join(case): True for case in gmm.DEVELOPMENT_MATRIX}
    confirmation_passes["skewed__four_cell__root"] = False
    monkeypatch.setattr(
        gmm,
        "run_case",
        _fake_matrix_case_runner(
            _all_case_pass_patterns((False, False, True, True)),
            confirmation_pass_by_case=confirmation_passes,
        ),
    )

    report = gmm.run_screen(profile="development")

    assert report["common_training_lock"]["locked_sample_count"] == 65_536
    assert report["common_training_lock"]["status"] == ("hard_stop_common_confirmation_failure")
    assert report["common_confirmation_evidence"]["requested"] is True
    assert report["common_confirmation_evidence"]["complete"] is True
    assert report["common_confirmation_evidence"]["all_cases_pass"] is False
    failed = report["common_confirmation_evidence"]["cases"]["skewed__four_cell__root"]
    assert failed["pass"] is False
    assert report["scientific_pass"] is False


def test_between_bank_evidence_range_can_veto_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Individually accurate repeats must also agree within 0.05 nat."""
    failing_case = "boundary_heavy__four_cell__root"
    monkeypatch.setattr(
        gmm,
        "run_case",
        _fake_matrix_case_runner(
            _all_case_pass_patterns((False, True, True, True)),
            confirmation_log_evidence_by_case={
                failing_case: (0.0, 0.03, 0.06),
            },
        ),
    )

    report = gmm.run_screen(profile="development")

    evidence = report["common_confirmation_evidence"]["cases"][failing_case]
    assert evidence["pass_without_repeat_evidence_gate"] is True
    assert evidence["between_bank_log_evidence_range_nat"] == pytest.approx(0.06)
    assert evidence["between_bank_log_evidence_range_pass"] is False
    assert evidence["pass"] is False
    assert report["common_confirmation_evidence"]["all_cases_pass"] is False
    assert report["common_training_lock"]["status"] == ("hard_stop_common_confirmation_failure")
    assert report["development_pass"] is False
    assert report["eligible_for_protected_holdout"] is False


def test_bounded_smoke_screen_replays_without_claiming_certification() -> None:
    """The one-case smoke profile should replay without opening the holdout."""
    first = gmm.run_screen(
        profile="smoke",
        sample_counts=gmm.SMOKE_SAMPLE_COUNTS,
        repeat_seeds=gmm.SMOKE_REPEAT_SEEDS,
    )
    second = gmm.run_screen(
        profile="smoke",
        sample_counts=gmm.SMOKE_SAMPLE_COUNTS,
        repeat_seeds=gmm.SMOKE_REPEAT_SEEDS,
    )

    for report in (first, second):
        report.pop("elapsed_seconds")
        report["cases"][0]["development_evaluations"][0].pop("fit_and_evaluate_seconds")
    assert first == second
    assert first["profile"] == "smoke"
    assert first["sample_counts"] == [4_096]
    assert first["repeat_seeds"] == [731]
    assert first["smoke_pass"] is True
    assert first["development_pass"] is None
    assert first["eligible_for_protected_holdout"] is False
    assert first["protected_holdout_pass"] is None
    assert first["scientific_pass"] is False
    assert first["scientific_pass_available"] is False
    assert first["structural_inference_licensed"] is False
    assert first["held_out_information_read"] is False
    assert len(first["cases"]) == 1
    assert first["cases"][0]["locked_sample_count"] == 4_096
    assert first["cases"][0]["scientific_pass"] is True


@pytest.mark.parametrize(
    ("training", "validation"),
    [
        (np.empty((0, 1)), np.ones((8, 1))),
        (np.ones((8, 1)), np.empty((0, 1))),
        (np.ones((8, 1)), np.ones((8, 2))),
        (np.full((8, 1), np.nan), np.ones((8, 1))),
    ],
)
def test_malformed_training_matrices_fail_closed(
    training: np.ndarray,
    validation: np.ndarray,
) -> None:
    """Empty, non-finite, or dimension-mismatched fitting data are invalid."""
    with pytest.raises(RuntimeError, match="all three"):
        gmm.fit_gaussian_mixture(training, validation)


def test_empty_components_and_non_spd_covariance_fail_closed() -> None:
    """The normalized raw evaluator must reject invalid mixtures."""
    samples = np.zeros((3, 1), dtype=np.float64)
    means = np.zeros((gmm.COMPONENT_COUNT, 1), dtype=np.float64)
    covariances = np.repeat(
        np.eye(1, dtype=np.float64)[None, :, :],
        gmm.COMPONENT_COUNT,
        axis=0,
    )
    bad_weights = np.full(
        gmm.COMPONENT_COUNT,
        1.0 / gmm.COMPONENT_COUNT,
        dtype=np.float64,
    )
    bad_weights[0] = 0.0
    with pytest.raises(ValueError, match="shapes or weights"):
        gmm._component_log_densities(
            samples,
            bad_weights,
            means,
            covariances,
        )

    weights = np.full(
        gmm.COMPONENT_COUNT,
        1.0 / gmm.COMPONENT_COUNT,
        dtype=np.float64,
    )
    covariances[0, 0, 0] = -1.0
    with pytest.raises(FloatingPointError, match="positive definite"):
        gmm._component_log_densities(
            samples,
            weights,
            means,
            covariances,
        )


def test_source_contains_no_protected_numerical_catalogue() -> None:
    """Only an opaque declaration, never held-out operators, may be shipped."""
    source = Path(gmm.__file__).read_text()

    assert gmm.PROTECTED_HOLDOUT_CATALOGUE_ID in source
    assert '"protected"' not in gmm._parser().get_default("profile")
    assert "numerical_values_present" in source
    assert '"numerical_values_present": False' in source
