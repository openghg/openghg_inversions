#!/usr/bin/env python3
"""Select starts blindly, then score one corrected all-six promotion array."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

from examples.rjmcmc import score_regularized_flow_corrected_array as arrays
from examples.rjmcmc import score_regularized_flow_corrected_exploration as experiment
from examples.rjmcmc import score_regularized_flow_corrected_merge as exploratory_merge

SCHEMA = "rjmcmc-score-nle-corrected-promotion-summary-v1"
PRIMARY_CONFIG_ID = "fisher_observation_joint"
COMPARATOR_CONFIG_ID = "nll_only"
FIXED_INITIALIZATION_INDICES = (0, 1, 2, 3)
SCIENTIFIC_THRESHOLDS = {
    "prior_weighted_median_absolute_log_likelihood_error_nat": 0.05,
    "posterior_weighted_p99_absolute_log_likelihood_error_nat": 0.20,
    "scaled_retained_mass_gradient_error": 0.05,
    "absolute_log_evidence_error_nat": 0.05,
    "posterior_mean_error_reference_sd": 0.05,
    "posterior_sd_relative_error": 0.02,
    "posterior_interval_endpoint_error_reference_sd": 0.05,
}
DEVELOPMENT_MATRIX_IDS = frozenset(
    {
        "promotion_development_s4096",
        "promotion_development_s16384",
    }
)
CONFIRMATION_MATRIX_IDS = frozenset(
    {
        "promotion_confirmation_s16384_seed2731",
        "promotion_confirmation_s16384_seed3731",
        "promotion_confirmation_s16384_seed4731",
    }
)
PROMOTION_MATRIX_IDS = DEVELOPMENT_MATRIX_IDS | CONFIRMATION_MATRIX_IDS


@dataclass(frozen=True)
class SelectedAttempt:
    """One model-selection-only choice with authenticated file identities."""

    report: dict[str, Any]
    report_file_sha256: str
    artifact_file_sha256: str
    all_start_manifest: tuple[dict[str, object], ...]


def _sha256_json(payload: object) -> str:
    compact = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(compact.encode("ascii")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object, *, name: str) -> float:
    result = float(cast(float, value))
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _base_seed(report: dict[str, Any]) -> int:
    evidence = report.get("domain_evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(experiment.domains.PUBLIC_DOMAINS):
        raise ValueError("attempt does not contain every public-domain identity.")
    seeds = {raw.get("base_seed") for raw in evidence.values() if isinstance(raw, dict)}
    if len(seeds) != 1:
        raise ValueError("attempt public domains do not share one base seed.")
    seed = seeds.pop()
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("attempt base seed is not unsigned 64-bit.")
    return seed


def _selection_manifest(
    reports: list[exploratory_merge._AuthenticatedAttempt],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "initialization_index": int(item.report["initialization_index"]),
            "attempt_id": item.report["attempt_id"],
            "model_selection_nll_nat_per_dimension": _finite(
                item.report["model_selection"]["nll_nat_per_dimension"],
                name="model-selection NLL per dimension",
            ),
            "report_payload_sha256": item.report["sha256"],
            "report_file_sha256": item.report_file_sha256,
            "serialized_artifact_file_sha256": item.serialized_artifact_file_sha256,
            "runtime_identity_sha256": item.report["runtime_identity_sha256"],
            "execution_identity_sha256": item.report["execution_identity_sha256"],
        }
        for item in sorted(
            reports,
            key=lambda value: int(value.report["initialization_index"]),
        )
    )


def _select_attempt(
    reports: list[exploratory_merge._AuthenticatedAttempt],
) -> SelectedAttempt:
    """Select exclusively by independent model-selection NLL and fixed tie rule."""
    indices = sorted(int(item.report["initialization_index"]) for item in reports)
    if indices != list(FIXED_INITIALIZATION_INDICES):
        raise ValueError("promotion group does not contain the four fixed starts.")
    manifest = _selection_manifest(reports)
    selected = min(
        reports,
        key=lambda item: (
            _finite(
                item.report["model_selection"]["nll_nat_per_dimension"],
                name="model-selection NLL per dimension",
            ),
            int(item.report["initialization_index"]),
        ),
    )
    return SelectedAttempt(
        report=selected.report,
        report_file_sha256=selected.report_file_sha256,
        artifact_file_sha256=selected.serialized_artifact_file_sha256,
        all_start_manifest=manifest,
    )


def _selected_row(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Small selection-only helper retained for direct unit tests."""
    indices = sorted(int(report["initialization_index"]) for report in reports)
    if indices != list(FIXED_INITIALIZATION_INDICES):
        raise ValueError("promotion group does not contain the four fixed starts.")
    selected = min(
        reports,
        key=lambda report: (
            _finite(
                report["model_selection"]["nll_nat_per_dimension"],
                name="model-selection NLL per dimension",
            ),
            int(report["initialization_index"]),
        ),
    )
    return {
        "selected_initialization_index": selected["initialization_index"],
        "selected_attempt_id": selected["attempt_id"],
        "model_selection_nll_by_initialization": {
            str(report["initialization_index"]): report["model_selection"]["nll_nat_per_dimension"]
            for report in sorted(
                reports,
                key=lambda report: int(report["initialization_index"]),
            )
        },
    }


def _evaluate_selected(
    selected: SelectedAttempt,
    oracle_bundle: dict[str, Any],
    *,
    exact_grid_cache: dict[int, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate reporting and oracle diagnostics only after blind selection."""
    report = selected.report
    if report.get("reporting_test") != {
        "status": "deferred_until_model_selection_only_start_selection",
        "selection_uses_reporting_data": False,
    }:
        raise ValueError("promotion attempt evaluated reporting data before selection.")
    if report.get("scientific_evaluation") != {
        "status": "deferred_until_model_selection_only_start_selection",
        "selection_uses_oracle_evidence": False,
    }:
        raise ValueError("promotion attempt evaluated oracle evidence before selection.")
    attempt_root = Path(str(report["execution"]["output_root"]))
    artifact_path = attempt_root / "artifact.bin"
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise ValueError("selected artifact must be a regular file.")
    if _sha256_file(artifact_path) != selected.artifact_file_sha256:
        raise ValueError("selected artifact file SHA-256 changed after authentication.")
    artifact_bytes = artifact_path.read_bytes()
    artifact = experiment.ScoreRegularizedRootFlow.from_bytes(
        artifact_bytes,
        expected_sha256=selected.artifact_file_sha256,
    )
    case_id = str(report["target"]["case_id"])
    sample_count = int(report["target"]["sample_count"])
    base_seed = _base_seed(report)
    reporting = experiment.domains.simulate_tiny_score_domain(
        case_id,
        domain=experiment.domains.DEVELOPMENT_REPORTING_TEST_DOMAIN,
        sample_count=sample_count,
        base_seed=base_seed,
    )
    expected_domain = report["domain_evidence"][experiment.domains.DEVELOPMENT_REPORTING_TEST_DOMAIN]
    if reporting.evidence.payload() != expected_domain:
        raise ValueError("selected reporting domain does not replay its evidence.")
    scale = report["training_loss_scale_diagnostics"]
    reporting_test = experiment._score_domain(
        artifact.flow,
        reporting,
        partial_scale=_finite(
            scale["partial_score_rms"],
            name="partial-score scale",
        ),
        observation_scales=tuple(
            _finite(value, name="observation-score scale")
            for value in scale["observation_score_rms_by_coordinate"]
        ),
    )
    oracle_case = oracle_bundle["selected_cases"][case_id]
    scientific = experiment._scientific_grid_evaluation(
        artifact,
        case_id,
        oracle_case["reference"],
        interpolate_posterior_quantiles=True,
        exact_grid_cache=exact_grid_cache,
        promotion_oracle_case=oracle_case,
    )
    return {
        **report,
        "reporting_test": reporting_test,
        "scientific_evaluation": scientific,
        "post_selection_evaluation": {
            "reporting_and_oracle_evaluated_after_selection": True,
            "selection_used_reporting_data": False,
            "selection_used_oracle_evidence": False,
            "artifact_path": str(artifact_path),
            "artifact_file_sha256": selected.artifact_file_sha256,
            "report_file_sha256": selected.report_file_sha256,
        },
    }


def _nll_agreement(report: dict[str, Any]) -> dict[str, float | bool]:
    model_selection = report["model_selection"]
    reporting = report["reporting_test"]
    rank = int(report["candidate"]["architecture_rank"])
    observed = abs(
        _finite(model_selection["nll_nat_per_draw"], name="model-selection NLL")
        - _finite(reporting["nll_nat_per_draw"], name="reporting NLL")
    )
    pooled_mcse = math.hypot(
        _finite(
            model_selection["nll_iid_mcse_nat_per_draw"],
            name="model-selection NLL MCSE",
        ),
        _finite(
            reporting["nll_iid_mcse_nat_per_draw"],
            name="reporting NLL MCSE",
        ),
    )
    tolerance = max(0.02 * rank, 5.0 * pooled_mcse)
    return {
        "absolute_difference_nat_per_draw": observed,
        "pooled_iid_mcse_nat_per_draw": pooled_mcse,
        "tolerance_nat_per_draw": tolerance,
        "pass": observed <= tolerance,
    }


def _selected_scientific_checks(
    report: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, float]]:
    scientific = report["scientific_evaluation"]
    pointwise = scientific["pointwise"]
    posterior = scientific["posterior"]
    normalization = scientific["normalization"]
    interpretable = scientific["metric_interpretability"]
    model_selection = report["model_selection"]
    reporting = report["reporting_test"]
    metrics = {
        "prior_weighted_median_absolute_log_likelihood_error_nat": _finite(
            pointwise["prior_weighted_median_absolute_log_likelihood_error_nat"],
            name="prior-weighted median likelihood error",
        ),
        "posterior_weighted_p99_absolute_log_likelihood_error_nat": _finite(
            pointwise["exact_posterior_weighted_p99_absolute_log_likelihood_error_nat"],
            name="posterior-weighted p99 likelihood error",
        ),
        "scaled_retained_mass_gradient_error": _finite(
            scientific["gradient"]["scaled_error"],
            name="scaled retained-mass gradient error",
        ),
        "absolute_log_evidence_error_nat": _finite(
            scientific["evidence"]["absolute_learned_error_from_adaptive_reference_nat"],
            name="absolute log-evidence error",
        ),
        "posterior_mean_error_reference_sd": _finite(
            posterior["mean_error_reference_sd"],
            name="posterior-mean error",
        ),
        "posterior_sd_relative_error": _finite(
            posterior["sd_relative_error"],
            name="posterior-SD error",
        ),
        "posterior_interval_endpoint_error_reference_sd": _finite(
            posterior["interval_endpoint_error_reference_sd"],
            name="posterior interval endpoint error",
        ),
    }
    metric_flags = {
        "prior_weighted_median_absolute_log_likelihood_error_nat": (
            "prior_weighted_median_log_likelihood_error"
        ),
        "posterior_weighted_p99_absolute_log_likelihood_error_nat": (
            "posterior_weighted_p99_log_likelihood_error"
        ),
        "scaled_retained_mass_gradient_error": "retained_mass_gradient",
        "absolute_log_evidence_error_nat": "log_evidence",
        "posterior_mean_error_reference_sd": "posterior_moments",
        "posterior_sd_relative_error": "posterior_moments",
        "posterior_interval_endpoint_error_reference_sd": "posterior_quantiles",
    }
    checks = {
        metric: (interpretable.get(metric_flags[metric]) is True and value <= SCIENTIFIC_THRESHOLDS[metric])
        for metric, value in metrics.items()
    }
    finite_score_values = all(
        math.isfinite(_finite(container[key], name=key))
        for container in (model_selection, reporting)
        for key in (
            "nll_nat_per_draw",
            "fisher_scaled_partial_score_risk",
            "fisher_scaled_observation_score_risk",
        )
    )
    nll_agreement = _nll_agreement(report)
    checks.update(
        {
            "oracle_certificate_valid": (scientific.get("oracle_certificate_valid") is True),
            "normalized_density_by_flow_and_jacobian_construction": (
                normalization.get("normalized_density_by_flow_and_jacobian_construction") is True
            ),
            "finite_on_complete_metric_grid": (normalization.get("finite_on_complete_metric_grid") is True),
            "finite_model_selection_and_reporting_risks": finite_score_values,
            "model_selection_reporting_nll_agreement": bool(nll_agreement["pass"]),
            "vectorized_public_likelihood_parity": (
                scientific.get("vectorized_public_likelihood_parity", {}).get("pass") is True
            ),
        }
    )
    metrics.update(
        {
            "model_selection_reporting_nll_absolute_difference_nat_per_draw": float(
                nll_agreement["absolute_difference_nat_per_draw"]
            ),
            "model_selection_reporting_nll_tolerance_nat_per_draw": float(
                nll_agreement["tolerance_nat_per_draw"]
            ),
        }
    )
    return checks, metrics


def _post_selection_row(
    selected: SelectedAttempt,
    evaluated: dict[str, Any],
) -> dict[str, Any]:
    checks, metrics = _selected_scientific_checks(evaluated)
    return {
        "case_id": evaluated["target"]["case_id"],
        "sample_count": evaluated["target"]["sample_count"],
        "config_id": evaluated["candidate"]["config_id"],
        "selected_initialization_index": evaluated["initialization_index"],
        "selected_attempt_id": evaluated["attempt_id"],
        "selection_rule": (
            "minimum independent model-selection NLL per retained dimension; "
            "exact tie resolved by lowest initialization index"
        ),
        "selection_used_reporting_data": False,
        "selection_used_oracle_evidence": False,
        "all_start_manifest": list(selected.all_start_manifest),
        "selected_report_payload_sha256": evaluated["sha256"],
        "selected_report_file_sha256": selected.report_file_sha256,
        "selected_artifact_file_sha256": selected.artifact_file_sha256,
        "selected_runtime_identity_sha256": evaluated["runtime_identity_sha256"],
        "selected_execution_identity_sha256": evaluated["execution_identity_sha256"],
        "selected_artifact_path": evaluated["post_selection_evaluation"]["artifact_path"],
        "model_selection_nll_by_initialization": {
            str(row["initialization_index"]): row["model_selection_nll_nat_per_dimension"]
            for row in selected.all_start_manifest
        },
        "model_selection_nll_nat_per_draw": evaluated["model_selection"]["nll_nat_per_draw"],
        "reporting_nll_nat_per_draw": evaluated["reporting_test"]["nll_nat_per_draw"],
        "model_selection_observation_score_risk": evaluated["model_selection"][
            "fisher_scaled_observation_score_risk"
        ],
        "reporting_observation_score_risk": evaluated["reporting_test"][
            "fisher_scaled_observation_score_risk"
        ],
        "scientific_metrics": metrics,
        "metric_interpretability": evaluated["scientific_evaluation"]["metric_interpretability"],
        "learned_log_likelihood_sha256": evaluated["scientific_evaluation"]["learned_log_likelihood_sha256"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def _primary_comparator_checks(
    primary: dict[str, Any],
    comparator: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, float]]:
    primary_reporting = primary["reporting_test"]
    comparator_reporting = comparator["reporting_test"]
    rank = int(primary["candidate"]["architecture_rank"])
    nll_difference = _finite(
        primary_reporting["nll_nat_per_draw"],
        name="primary reporting NLL",
    ) - _finite(
        comparator_reporting["nll_nat_per_draw"],
        name="comparator reporting NLL",
    )
    nll_tolerance = max(
        0.02 * rank,
        5.0
        * math.hypot(
            _finite(
                primary_reporting["nll_iid_mcse_nat_per_draw"],
                name="primary reporting NLL MCSE",
            ),
            _finite(
                comparator_reporting["nll_iid_mcse_nat_per_draw"],
                name="comparator reporting NLL MCSE",
            ),
        ),
    )

    def improvement(domain: str) -> tuple[float, float]:
        primary_domain = primary[domain]
        comparator_domain = comparator[domain]
        difference = _finite(
            comparator_domain["fisher_scaled_observation_score_risk"],
            name=f"comparator {domain} observation risk",
        ) - _finite(
            primary_domain["fisher_scaled_observation_score_risk"],
            name=f"primary {domain} observation risk",
        )
        threshold = 5.0 * math.hypot(
            _finite(
                primary_domain["fisher_scaled_observation_score_risk_iid_mcse"],
                name=f"primary {domain} observation-risk MCSE",
            ),
            _finite(
                comparator_domain["fisher_scaled_observation_score_risk_iid_mcse"],
                name=f"comparator {domain} observation-risk MCSE",
            ),
        )
        return difference, threshold

    model_selection_improvement, model_selection_threshold = improvement("model_selection")
    reporting_improvement, reporting_threshold = improvement("reporting_test")
    checks = {
        "primary_reporting_nll_noninferior": nll_difference <= nll_tolerance,
        "primary_model_selection_observation_score_improves_by_five_mcse": (
            model_selection_improvement > model_selection_threshold
        ),
        "primary_reporting_observation_score_improves_by_five_mcse": (
            reporting_improvement > reporting_threshold
        ),
    }
    diagnostics = {
        "primary_minus_comparator_reporting_nll_nat_per_draw": nll_difference,
        "reporting_nll_noninferiority_tolerance_nat_per_draw": nll_tolerance,
        "comparator_minus_primary_model_selection_observation_score_risk": (model_selection_improvement),
        "model_selection_five_pooled_mcse_threshold": model_selection_threshold,
        "comparator_minus_primary_reporting_observation_score_risk": (reporting_improvement),
        "reporting_five_pooled_mcse_threshold": reporting_threshold,
    }
    return checks, diagnostics


def merge(
    matrix_id: str,
    attempt_tag: str,
    source_git_revision: str,
    output_root: Path,
) -> dict[str, Any]:
    """Authenticate attempts, select blindly, then evaluate selected artifacts."""
    if matrix_id not in PROMOTION_MATRIX_IDS:
        raise ValueError("matrix is not a frozen promotion matrix.")
    promotion_spec = dict(arrays.PROMOTION_MATRIX_SPECS[matrix_id])
    expected_base_seed = int(promotion_spec["base_seed"])
    matrix = arrays.MATRICES[matrix_id]
    if len(matrix) != arrays.EXPECTED_MATRIX_ATTEMPT_COUNTS[matrix_id]:
        raise RuntimeError("promotion matrix length differs from its declaration.")
    authenticated: list[exploratory_merge._AuthenticatedAttempt] = []
    missing: list[str] = []
    failures: list[dict[str, str]] = []
    for task_id, row in enumerate(matrix):
        slug = exploratory_merge._slug(row, attempt_tag)
        attempt_root = output_root / "attempts" / slug
        failure_path = attempt_root / "FAILURE.json"
        if failure_path.is_file():
            failures.append(json.loads(failure_path.read_text(encoding="ascii")))
            continue
        try:
            loaded = exploratory_merge._load_attempt(
                attempt_root,
                source_git_revision=source_git_revision,
                matrix_id=matrix_id,
                array_task_id=task_id,
                matrix_row=row,
                attempt_tag=attempt_tag,
            )
        except FileNotFoundError:
            missing.append(slug)
            continue
        if _base_seed(loaded.report) != expected_base_seed:
            raise ValueError("promotion attempt has the wrong base seed.")
        candidate = loaded.report["candidate"]
        observed_candidate_spec = {
            "learning_rate": candidate["learning_rate"],
            "batch_size": candidate["batch_size"],
            "microbatch_size": candidate["microbatch_size"],
            "maximum_total_epochs": candidate["maximum_total_epochs"],
            "patience": candidate["patience"],
        }
        expected_candidate_spec = {key: promotion_spec[key] for key in observed_candidate_spec}
        if observed_candidate_spec != expected_candidate_spec:
            raise ValueError("promotion candidate differs from the frozen spec.")
        authenticated.append(loaded)
    expected_groups = Counter(
        (case_id, sample_count, config_id) for _, case_id, sample_count, config_id, _ in matrix
    )
    grouped: dict[
        tuple[str, int, str],
        list[exploratory_merge._AuthenticatedAttempt],
    ] = defaultdict(list)
    for item in authenticated:
        report = item.report
        key = (
            str(report["target"]["case_id"]),
            int(report["target"]["sample_count"]),
            str(report["candidate"]["config_id"]),
        )
        grouped[key].append(item)
    complete = (
        len(authenticated) == len(matrix)
        and not missing
        and not failures
        and Counter({key: len(value) for key, value in grouped.items()}) == expected_groups
    )
    selected_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    if complete:
        # This mapping is fully determined using model-selection fields before
        # the oracle bundle is loaded or any reporting data are scored.
        selected_by_key = {key: _select_attempt(items) for key, items in sorted(grouped.items())}
        oracle_bundle = experiment._load_oracle_bundle(
            output_root / "oracle" / "oracle_bundle.json",
            source_git_revision,
            promotion=True,
        )
        exact_grid_caches: dict[str, dict[int, Any]] = defaultdict(dict)
        evaluated_by_key = {
            key: _evaluate_selected(
                selected,
                oracle_bundle,
                exact_grid_cache=exact_grid_caches[key[0]],
            )
            for key, selected in selected_by_key.items()
        }
        selected_rows = [
            _post_selection_row(selected_by_key[key], evaluated_by_key[key])
            for key in sorted(selected_by_key)
        ]
        if matrix_id in DEVELOPMENT_MATRIX_IDS:
            for case_id, sample_count in sorted(
                {(case_id, sample_count) for case_id, sample_count, _ in selected_by_key}
            ):
                primary = evaluated_by_key[(case_id, sample_count, PRIMARY_CONFIG_ID)]
                comparator = evaluated_by_key[(case_id, sample_count, COMPARATOR_CONFIG_ID)]
                checks, diagnostics = _primary_comparator_checks(
                    primary,
                    comparator,
                )
                comparisons.append(
                    {
                        "case_id": case_id,
                        "sample_count": sample_count,
                        "checks": checks,
                        "diagnostics": diagnostics,
                        "pass": all(checks.values()),
                    }
                )
    primary_rows = [row for row in selected_rows if row["config_id"] == PRIMARY_CONFIG_ID]
    promotion_pass = (
        complete
        and len(primary_rows) == len(experiment.PROMOTION_CASES)
        and all(row["pass"] for row in primary_rows)
        and all(comparison["pass"] for comparison in comparisons)
    )
    without_sha: dict[str, Any] = {
        "schema": SCHEMA,
        "matrix_id": matrix_id,
        "attempt_tag": attempt_tag,
        "source_git_revision": source_git_revision,
        "expected_base_seed": expected_base_seed,
        "promotion_spec": promotion_spec,
        "expected_attempt_count": len(matrix),
        "complete_attempt_count": len(authenticated),
        "missing_attempts": missing,
        "failures": failures,
        "complete": complete,
        "candidate_definition": {
            "primary_config_id": PRIMARY_CONFIG_ID,
            "comparator_config_id": (COMPARATOR_CONFIG_ID if matrix_id in DEVELOPMENT_MATRIX_IDS else None),
            "fixed_initialization_indices": list(FIXED_INITIALIZATION_INDICES),
            "selection_rule": (
                "minimum independent model-selection NLL per retained "
                "dimension; exact tie resolved by lowest initialization index"
            ),
            "selection_uses_reporting_data": False,
            "selection_uses_oracle_evidence": False,
            "approximate_evidence_used_as_structural_weight": False,
        },
        "scientific_thresholds": SCIENTIFIC_THRESHOLDS,
        "selected_rows": selected_rows,
        "primary_comparator_checks": comparisons,
        "promotion_pass": promotion_pass,
    }
    return {**without_sha, "sha256": _sha256_json(without_sha)}


def _publish(
    result: dict[str, Any],
    *,
    summary_root: Path,
) -> None:
    if summary_root.exists() or summary_root.is_symlink():
        raise FileExistsError("refusing to replace an existing promotion summary.")
    summary_root.mkdir(parents=True)
    summary_path = summary_root / "summary.json"
    summary_file_sha256 = hashlib.sha256(experiment._json_bytes(result)).hexdigest()
    experiment._atomic_json(summary_path, result)
    if not result["complete"]:
        experiment._atomic_json(
            summary_root / "INCOMPLETE.json",
            {
                "schema": SCHEMA,
                "source_git_revision": result["source_git_revision"],
                "matrix_id": result["matrix_id"],
                "attempt_tag": result["attempt_tag"],
                "summary_path": str(summary_path),
                "summary_payload_sha256": result["sha256"],
                "summary_file_sha256": summary_file_sha256,
                "completion_marker_published": False,
            },
        )
        raise RuntimeError("promotion array summary is incomplete.")
    experiment._atomic_json(
        summary_root / "COMPLETE.json",
        {
            "schema": SCHEMA,
            "source_git_revision": result["source_git_revision"],
            "matrix_id": result["matrix_id"],
            "attempt_tag": result["attempt_tag"],
            "summary_path": str(summary_path),
            "summary_payload_sha256": result["sha256"],
            "summary_file_sha256": summary_file_sha256,
            "promotion_pass": result["promotion_pass"],
            "completion_marker_published_last": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-id",
        choices=tuple(sorted(PROMOTION_MATRIX_IDS)),
        required=True,
    )
    parser.add_argument("--attempt-tag", required=True)
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = merge(
        arguments.matrix_id,
        arguments.attempt_tag,
        arguments.source_git_revision,
        arguments.output_root,
    )
    summary_root = (
        arguments.output_root / "promotion_summaries" / f"{arguments.matrix_id}__{arguments.attempt_tag}"
    )
    _publish(result, summary_root=summary_root)


if __name__ == "__main__":
    main()
