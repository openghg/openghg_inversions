"""Tests for the corrected array-friendly exploration driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from examples.rjmcmc import (
    score_regularized_flow_corrected_array as array_driver,
    score_regularized_flow_corrected_exploration as driver,
    score_regularized_flow_corrected_merge as merger,
    score_regularized_flow_corrected_oracle as oracle_driver,
)
from openghg_inversions.experimental.rjmcmc import aggregation_error_tiny_oracle


def _reference(
    case_id: str,
    *,
    fraction_order: int = 4,
) -> dict[str, object]:
    without_sha = {
        "schema": aggregation_error_tiny_oracle.SCHEMA,
        "case_id": case_id,
        "definitions_sha256": aggregation_error_tiny_oracle.definitions_sha256(),
        "method": "adaptive_log_total_with_gauss_jacobi_allocation",
        "fraction_order": fraction_order,
        "lower_log_total": -5.0,
        "upper_log_total": 3.0,
        "epsabs": 1.0e-10,
        "epsrel": 1.0e-10,
        "posterior_mode_total": 1.0,
        "log_evidence": -2.0,
        "posterior_mean_total": 1.0,
        "posterior_sd_total": 0.2,
        "posterior_lower_0_025": 0.6,
        "posterior_median": 1.0,
        "posterior_upper_0_975": 1.4,
        "scaled_quadrature_error": 1.0e-10,
        "maximum_relative_moment_quadrature_error": 1.0e-10,
        "maximum_scaled_cdf_quadrature_error": 1.0e-10,
        "represented_prior_mass": 1.0,
        "represented_posterior_mass": 1.0,
        "posterior_mass_accounting": (
            "conservative lower bound from omitted Gamma prior mass times "
            "the global normalized-Gaussian density upper bound"
        ),
        "mode_included": True,
    }
    return {
        **without_sha,
        "sha256": driver._sha256_json(without_sha),
    }


def _native_reference(
    case_id: str,
    *,
    lower_log_mass: float,
) -> dict[str, object]:
    return _hashed(
        {
            "schema": aggregation_error_tiny_oracle.SCHEMA,
            "definitions_sha256": (aggregation_error_tiny_oracle.definitions_sha256()),
            "case_id": case_id,
            "method": "adaptive_native_two_dimensional_log_masses",
            "lower_log_mass": lower_log_mass,
            "upper_log_mass": 3.0,
            "epsabs": 2.0e-8,
            "epsrel": 2.0e-8,
            "log_evidence": -2.0,
            "posterior_mean_total": 1.0,
            "posterior_sd_total": 0.2,
            "scaled_quadrature_error": 1.0e-10,
            "maximum_inner_scaled_quadrature_error": 1.0e-10,
        }
    )


def _hashed(payload: dict[str, object]) -> dict[str, object]:
    return {
        **payload,
        "sha256": driver._sha256_json(payload),
    }


def _rehash(mapping: dict[str, object]) -> None:
    without_sha = dict(mapping)
    without_sha.pop("sha256", None)
    mapping["sha256"] = driver._sha256_json(without_sha)


def _rewrite_bundle(path: Path, payload: dict[str, object]) -> None:
    _rehash(payload)
    report_bytes = json.dumps(payload).encode("ascii")
    path.write_bytes(report_bytes)
    completion_path = path.parent / "COMPLETE.json"
    completion = json.loads(completion_path.read_text(encoding="ascii"))
    completion["oracle_bundle_payload_sha256"] = payload["sha256"]
    completion["oracle_bundle_file_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    completion_path.write_text(
        json.dumps(completion),
        encoding="ascii",
    )


def _promotion_metric_grid_preflight(
    case_id: str,
    reference: dict[str, object],
) -> dict[str, object]:
    posterior = {
        key: reference[key]
        for key in (
            "log_evidence",
            "posterior_mean_total",
            "posterior_sd_total",
            "posterior_lower_0_025",
            "posterior_median",
            "posterior_upper_0_975",
        )
    }
    without_sha = {
        "case_id": case_id,
        "construction": "equal-prior-probability midpoint bins",
        "posterior_quantile_rule": ("within-bin interpolation under piecewise-constant likelihood"),
        "counts": list(oracle_driver.METRIC_GRID_COUNTS),
        "rows": [
            {
                "count": count,
                "posterior": dict(posterior),
                "total_grid_sha256": hashlib.sha256(f"total-{case_id}-{count}".encode("ascii")).hexdigest(),
                "exact_log_likelihood_sha256": hashlib.sha256(
                    f"exact-{case_id}-{count}".encode("ascii")
                ).hexdigest(),
                "finite": True,
            }
            for count in oracle_driver.METRIC_GRID_COUNTS
        ],
        "last_two": {
            "log_evidence_delta_nat": 0.0,
            "posterior_mean_delta_reference_sd": 0.0,
            "posterior_sd_relative_delta": 0.0,
            "posterior_quantile_delta_reference_sd": 0.0,
        },
        "final_errors_from_adaptive_reference": {
            "log_evidence_error_nat": 0.0,
            "posterior_mean_error_reference_sd": 0.0,
            "posterior_sd_relative_error": 0.0,
            "posterior_quantile_error_reference_sd": 0.0,
        },
        "checks": {
            "finite": True,
            "last_two_evidence": True,
            "last_two_mean": True,
            "last_two_sd": True,
            "last_two_quantiles": True,
            "adaptive_evidence": True,
            "adaptive_mean": True,
            "adaptive_sd": True,
            "adaptive_quantiles": True,
        },
        "pass": True,
    }
    return _hashed(without_sha)


def _promotion_gradient_preflight(
    case_id: str,
    reference: dict[str, object],
    previous_fraction_order: int,
) -> dict[str, object]:
    without_sha = {
        "case_id": case_id,
        "state": "adaptive exact posterior mode total",
        "total_mass": reference["posterior_mode_total"],
        "previous_fraction_order": previous_fraction_order,
        "final_fraction_order": reference["fraction_order"],
        "final_order_step_ladder": [
            {
                "log_total_step": step,
                "gradient": 1.0,
            }
            for step in oracle_driver.GRADIENT_LOG_TOTAL_STEPS
        ],
        "previous_order_final_step_gradient": 1.0,
        "reference_gradient": 1.0,
        "diagnostics": {
            "finite_difference_refinement_scaled_error": 0.0,
            "allocation_order_refinement_scaled_error": 0.0,
        },
        "tolerance": oracle_driver.GRADIENT_NUMERICAL_TOLERANCE,
        "checks": {
            "finite": True,
            "finite_difference_refined": True,
            "allocation_order_refined": True,
        },
        "pass": True,
    }
    return _hashed(without_sha)


def _promotion_boundary_certificate(
    primary_ladder: list[dict[str, object]],
) -> dict[str, object]:
    case_id = aggregation_error_tiny_oracle.BOUNDARY_CASE_ID
    checks = {
        "primary_log_evidence_converged": True,
        "independent_log_evidence_agrees": True,
        "independent_posterior_mean_agrees": True,
        "independent_posterior_sd_agrees": True,
        "primary_posterior_mean_converged": True,
        "primary_posterior_sd_converged": True,
        "primary_posterior_endpoints_converged": True,
        "independent_tail_log_evidence_converged": True,
        "independent_tail_posterior_mean_converged": True,
        "independent_tail_posterior_sd_converged": True,
        "primary_scaled_quadrature_error_small": True,
        "independent_outer_scaled_quadrature_error_small": True,
        "independent_inner_scaled_quadrature_error_small": True,
        "support_retains_prior_mass": True,
        "support_retains_posterior_mass": True,
        "posterior_mode_included": True,
    }
    return _hashed(
        {
            "schema": aggregation_error_tiny_oracle.SCHEMA,
            "case_id": case_id,
            "definitions_sha256": (aggregation_error_tiny_oracle.definitions_sha256()),
            "primary_order_ladder": primary_ladder,
            "independent_tail_ladder": [
                _native_reference(case_id, lower_log_mass=lower) for lower in (-40.0, -80.0, -120.0)
            ],
            "diagnostics": {
                "primary_log_evidence_delta_nat": 0.0,
                "independent_log_evidence_delta_nat": 0.0,
                "independent_posterior_mean_delta_reference_sd": 0.0,
                "independent_posterior_sd_relative_delta": 0.0,
                "primary_posterior_mean_delta_reference_sd": 0.0,
                "primary_posterior_sd_relative_delta": 0.0,
                "primary_posterior_endpoint_delta_reference_sd": 0.0,
                "independent_tail_log_evidence_delta_nat": 0.0,
                "independent_tail_posterior_mean_delta_reference_sd": 0.0,
                "independent_tail_posterior_sd_relative_delta": 0.0,
            },
            "checks": checks,
            "pass": True,
        }
    )


def _native_summary_namespace(
    case_id: str,
    *,
    lower_log_mass: float,
    inner_error: float = 1.0e-10,
) -> SimpleNamespace:
    payload = _native_reference(
        case_id,
        lower_log_mass=lower_log_mass,
    )
    payload["maximum_inner_scaled_quadrature_error"] = inner_error
    _rehash(payload)
    return SimpleNamespace(
        log_evidence=-2.0,
        posterior_mean_total=1.0,
        posterior_sd_total=0.2,
        scaled_quadrature_error=1.0e-10,
        maximum_inner_scaled_quadrature_error=inner_error,
        payload=lambda: payload,
    )


def _write_bundle(
    path: Path,
    source_git_revision: str,
    *,
    promotion: bool = False,
) -> None:
    schema = driver.PROMOTION_ORACLE_BUNDLE_SCHEMA if promotion else driver.LEGACY_ORACLE_BUNDLE_SCHEMA
    case_ids = driver.PROMOTION_CASES if promotion else driver.SELECTED_CASES
    selected_cases: dict[str, object] = {}
    for case_id in case_ids:
        orders = driver.PROMOTION_ORACLE_ORDER_LADDERS[case_id] if promotion else (4, 4)
        ladder = [_reference(case_id, fraction_order=order) for order in orders]
        raw_case: dict[str, object] = {
            "order_ladder": ladder,
            "reference": ladder[-1],
            "pass": True,
        }
        if promotion:
            posterior = {
                "log_evidence": -2.0,
                "posterior_mean_total": 1.0,
                "posterior_sd_total": 0.2,
                "posterior_lower_0_025": 0.6,
                "posterior_median": 1.0,
                "posterior_upper_0_975": 1.4,
            }
            zero_grid_delta = {
                "log_evidence_delta_nat": 0.0,
                "posterior_mean_delta_reference_sd": 0.0,
                "posterior_sd_relative_delta": 0.0,
                "posterior_quantile_delta_reference_sd": 0.0,
            }
            zero_reference_error = {
                "log_evidence_error_nat": 0.0,
                "posterior_mean_error_reference_sd": 0.0,
                "posterior_sd_relative_error": 0.0,
                "posterior_quantile_error_reference_sd": 0.0,
            }
            grid_checks = {
                "finite": True,
                "last_two_evidence": True,
                "last_two_mean": True,
                "last_two_sd": True,
                "last_two_quantiles": True,
                "adaptive_evidence": True,
                "adaptive_mean": True,
                "adaptive_sd": True,
                "adaptive_quantiles": True,
            }
            grid_without_sha = {
                "case_id": case_id,
                "construction": "equal-prior-probability midpoint bins",
                "posterior_quantile_rule": ("within-bin interpolation under piecewise-constant likelihood"),
                "counts": list(oracle_driver.METRIC_GRID_COUNTS),
                "rows": [
                    {
                        "count": count,
                        "posterior": dict(posterior),
                        "total_grid_sha256": hashlib.sha256(
                            f"total-{case_id}-{count}".encode("ascii")
                        ).hexdigest(),
                        "exact_log_likelihood_sha256": hashlib.sha256(
                            f"exact-{case_id}-{count}".encode("ascii")
                        ).hexdigest(),
                        "finite": True,
                    }
                    for count in oracle_driver.METRIC_GRID_COUNTS
                ],
                "last_two": zero_grid_delta,
                "final_errors_from_adaptive_reference": zero_reference_error,
                "checks": grid_checks,
                "pass": True,
            }
            raw_case["metric_grid_preflight"] = _hashed(grid_without_sha)
            orders = driver.PROMOTION_ORACLE_ORDER_LADDERS[case_id]
            gradient_without_sha = {
                "case_id": case_id,
                "state": "adaptive exact posterior mode total",
                "total_mass": 1.0,
                "previous_fraction_order": orders[-2],
                "final_fraction_order": orders[-1],
                "final_order_step_ladder": [
                    {
                        "log_total_step": step,
                        "gradient": 1.0,
                    }
                    for step in oracle_driver.GRADIENT_LOG_TOTAL_STEPS
                ],
                "previous_order_final_step_gradient": 1.0,
                "reference_gradient": 1.0,
                "diagnostics": {
                    "finite_difference_refinement_scaled_error": 0.0,
                    "allocation_order_refinement_scaled_error": 0.0,
                },
                "tolerance": oracle_driver.GRADIENT_NUMERICAL_TOLERANCE,
                "checks": {
                    "finite": True,
                    "finite_difference_refined": True,
                    "allocation_order_refined": True,
                },
                "pass": True,
            }
            raw_case["gradient_preflight"] = _hashed(
                gradient_without_sha,
            )
            if case_id == aggregation_error_tiny_oracle.BOUNDARY_CASE_ID:
                raw_case["last_two_log_evidence_delta_nat"] = 0.0
                raw_case["checks"] = {
                    "boundary_independent_certificate": True,
                    "metric_grid_preflight": True,
                    "gradient_preflight": True,
                }
            else:
                raw_case.update(
                    {
                        "last_two_log_evidence_delta_nat": 0.0,
                        "last_two_posterior_mean_delta_reference_sd": 0.0,
                        "last_two_posterior_sd_relative_delta": 0.0,
                        "last_two_posterior_endpoint_delta_reference_sd": 0.0,
                        "checks": {
                            "log_evidence_converged": True,
                            "posterior_mean_converged": True,
                            "posterior_sd_converged": True,
                            "posterior_endpoints_converged": True,
                            "represented_prior_mass": True,
                            "represented_posterior_mass": True,
                            "posterior_mode_included": True,
                            "scaled_quadrature_error_small": True,
                            "moment_quadrature_error_small": True,
                            "cdf_quadrature_error_small": True,
                            "metric_grid_preflight": True,
                            "gradient_preflight": True,
                        },
                    }
                )
        selected_cases[case_id] = raw_case
    boundary_case_id = aggregation_error_tiny_oracle.BOUNDARY_CASE_ID
    if promotion:
        boundary_selected = cast(
            dict[str, object],
            selected_cases[boundary_case_id],
        )
        boundary_primary = cast(
            list[dict[str, object]],
            boundary_selected["order_ladder"],
        )
        boundary_native = [
            _native_reference(
                boundary_case_id,
                lower_log_mass=lower,
            )
            for lower in (-40.0, -80.0, -120.0)
        ]
        boundary_diagnostics = {
            "primary_log_evidence_delta_nat": 0.0,
            "independent_log_evidence_delta_nat": 0.0,
            "independent_posterior_mean_delta_reference_sd": 0.0,
            "independent_posterior_sd_relative_delta": 0.0,
            "primary_posterior_mean_delta_reference_sd": 0.0,
            "primary_posterior_sd_relative_delta": 0.0,
            "primary_posterior_endpoint_delta_reference_sd": 0.0,
            "independent_tail_log_evidence_delta_nat": 0.0,
            "independent_tail_posterior_mean_delta_reference_sd": 0.0,
            "independent_tail_posterior_sd_relative_delta": 0.0,
        }
        boundary_checks = {
            "primary_log_evidence_converged": True,
            "independent_log_evidence_agrees": True,
            "independent_posterior_mean_agrees": True,
            "independent_posterior_sd_agrees": True,
            "primary_posterior_mean_converged": True,
            "primary_posterior_sd_converged": True,
            "primary_posterior_endpoints_converged": True,
            "independent_tail_log_evidence_converged": True,
            "independent_tail_posterior_mean_converged": True,
            "independent_tail_posterior_sd_converged": True,
            "primary_scaled_quadrature_error_small": True,
            "independent_outer_scaled_quadrature_error_small": True,
            "independent_inner_scaled_quadrature_error_small": True,
            "support_retains_prior_mass": True,
            "support_retains_posterior_mass": True,
            "posterior_mode_included": True,
        }
        boundary_without_sha = {
            "schema": aggregation_error_tiny_oracle.SCHEMA,
            "case_id": boundary_case_id,
            "definitions_sha256": (aggregation_error_tiny_oracle.definitions_sha256()),
            "primary_order_ladder": boundary_primary,
            "independent_tail_ladder": boundary_native,
            "diagnostics": boundary_diagnostics,
            "checks": boundary_checks,
            "pass": True,
        }
    else:
        boundary_without_sha = {
            "schema": aggregation_error_tiny_oracle.SCHEMA,
            "case_id": boundary_case_id,
            "pass": True,
        }
    boundary = _hashed(boundary_without_sha)
    independent_certificates: dict[str, object] = {}
    if promotion:
        for case_id in (
            "near_gaussian__four_cell__root",
            "skewed__four_cell__root",
            "boundary_heavy__four_cell__root",
        ):
            column = {
                **_reference(
                    case_id,
                    fraction_order=(driver.PROMOTION_ORACLE_ORDER_LADDERS[case_id][-1]),
                ),
                "method": ("adaptive_log_total_with_column_first_gauss_jacobi_allocation"),
            }
            column_without_sha = dict(column)
            column_without_sha.pop("sha256")
            column["sha256"] = driver._sha256_json(column_without_sha)
            selected_case = cast(
                dict[str, object],
                selected_cases[case_id],
            )
            reference = cast(
                dict[str, object],
                selected_case["reference"],
            )
            chart_diagnostics = {
                "absolute_log_evidence_delta_nat": 0.0,
                "posterior_mean_delta_reference_sd": 0.0,
                "posterior_sd_relative_delta": 0.0,
                "posterior_endpoint_delta_reference_sd": 0.0,
            }
            chart_checks = {
                "log_evidence": True,
                "posterior_mean": True,
                "posterior_sd": True,
                "posterior_endpoints": True,
                "column_normalizer_quadrature_error": True,
                "column_moment_quadrature_error": True,
                "column_cdf_quadrature_error": True,
                "column_represented_prior_mass": True,
                "column_represented_posterior_mass": True,
                "column_posterior_mode_included": True,
            }
            certificate_without_sha: dict[str, object] = {
                "case_id": case_id,
                "method": ("adaptive row-first versus column-first Dirichlet charts"),
                "fraction_order": (driver.PROMOTION_ORACLE_ORDER_LADDERS[case_id][-1]),
                "row_reference_sha256": reference["sha256"],
                "column_summary": column,
                "diagnostics": chart_diagnostics,
                "checks": chart_checks,
                "pass": True,
            }
            if case_id == "boundary_heavy__four_cell__root":
                fixed_checks = {
                    "fixed_log_total_converged": True,
                    "fixed_log_total_agrees_with_adaptive": True,
                }
                certificate_without_sha["fixed_log_total_column_chart"] = {
                    "method": "fixed Gauss-Legendre in log(total)",
                    "chart": "column-first",
                    "fraction_order": 24,
                    "prior_tail_probability": 1.0e-15,
                    "total_order_ladder": [512, 1024, 2048],
                    "log_evidence_ladder": [-2.0, -2.0, -2.0],
                    "diagnostics": {
                        "last_two_log_evidence_delta_nat": 0.0,
                        "adaptive_primary_delta_nat": 0.0,
                    },
                    "checks": fixed_checks,
                    "pass": True,
                }
                certificate_without_sha["checks"] = {
                    **chart_checks,
                    **fixed_checks,
                }
            independent_certificates[case_id] = _hashed(certificate_without_sha)
        native_summaries = [
            _native_reference(
                "skewed__two_cell__root",
                lower_log_mass=lower,
            )
            for lower in (-80.0, -120.0)
        ]
        native_diagnostics = {
            "tail_log_evidence_delta_nat": 0.0,
            "primary_log_evidence_delta_nat": 0.0,
            "primary_posterior_mean_delta_reference_sd": 0.0,
            "primary_posterior_sd_relative_delta": 0.0,
        }
        native_checks = {
            "tail_evidence": True,
            "primary_evidence": True,
            "primary_mean": True,
            "primary_sd": True,
            "native_outer_quadrature_errors": True,
            "native_inner_quadrature_errors": True,
        }
        native_without_sha = {
            "case_id": "skewed__two_cell__root",
            "method": "adaptive native two-dimensional log masses",
            "lower_log_mass_ladder": [-80.0, -120.0],
            "summaries": native_summaries,
            "diagnostics": native_diagnostics,
            "checks": native_checks,
            "pass": True,
        }
        independent_certificates["skewed__two_cell__root"] = _hashed(native_without_sha)
    without_sha = {
        "schema": schema,
        "source_git_revision": source_git_revision,
        "tiny_root_definitions_sha256": (aggregation_error_tiny_oracle.definitions_sha256()),
        "selected_cases": selected_cases,
        "boundary_independent_certificate": boundary,
        "pass": True,
    }
    if promotion:
        without_sha["independent_certificates"] = independent_certificates
        without_sha["checks"] = {
            **{
                f"{case_id}__converged": True
                for case_id in driver.PROMOTION_CASES
                if case_id != boundary_case_id
            },
            "boundary_independent_certificate": True,
            "boundary_heavy__two_cell__root__metric_grid": True,
            "boundary_heavy__two_cell__root__gradient": True,
            "near_gaussian__four_cell__root__independent_chart": True,
            "skewed__four_cell__root__independent_chart": True,
            "boundary_heavy__four_cell__root__independent_chart": True,
            "boundary_heavy__four_cell__root__fixed_log_total": True,
            "skewed__two_cell__root__independent_native": True,
        }
        without_sha["runtime_seconds"] = 1.0
    payload = {
        **without_sha,
        "sha256": driver._sha256_json(without_sha),
    }
    report_bytes = json.dumps(payload).encode("ascii")
    path.write_bytes(report_bytes)
    completion = {
        "schema": schema,
        "source_git_revision": source_git_revision,
        "report_path": str(path),
        "oracle_bundle_payload_sha256": payload["sha256"],
        "oracle_bundle_file_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "completion_marker_published_last": True,
    }
    (path.parent / "COMPLETE.json").write_text(
        json.dumps(completion),
        encoding="ascii",
    )


def _passing_oracle_bundle(source_git_revision: str) -> dict[str, object]:
    without_sha: dict[str, object] = {
        "schema": oracle_driver.SCHEMA,
        "source_git_revision": source_git_revision,
        "tiny_root_definitions_sha256": (aggregation_error_tiny_oracle.definitions_sha256()),
        "selected_cases": {},
        "boundary_independent_certificate": {},
        "checks": {},
        "pass": True,
        "runtime_seconds": 1.25,
    }
    return {
        **without_sha,
        "sha256": oracle_driver._sha256_json(without_sha),
    }


def test_private_pcg64_seed_material_is_replayable_and_role_separated() -> None:
    for case_id in driver.domains.CASE_IDS:
        simulator_seeds = {
            driver.domains.domain_stream_seed(
                731,
                case_id=case_id,
                domain=domain,
                stream_name=stream_name,
            )
            for domain in driver.domains.PUBLIC_DOMAINS
            for stream_name in driver.domains.SIMULATOR_STREAMS
        }
        for config_id in driver.CONFIG_IDS:
            stage_count = len(driver._stage_plan(config_id, 40))
            for init_index in range(4):
                initialization, optimizers = driver._private_stream_plan(
                    731,
                    case_id=case_id,
                    init_index=init_index,
                    stage_count=stage_count,
                )
                replay = driver._private_stream_plan(
                    731,
                    case_id=case_id,
                    init_index=init_index,
                    stage_count=stage_count,
                )
                assert (initialization, optimizers) == replay
                records = (initialization, *optimizers)
                source_seeds = {int(record["pcg64_source_seed"]) for record in records}
                jax_seeds = {int(record["derived_jax_seed"]) for record in records}
                assert len(source_seeds) == len(records)
                assert len(jax_seeds) == len(records)
                assert source_seeds.isdisjoint(simulator_seeds)
                assert all(0 <= seed < 2**32 for seed in jax_seeds)


def test_frozen_array_matrices_have_complete_unique_attempts() -> None:
    assert {name: len(matrix) for name, matrix in array_driver.MATRICES.items()} == {
        "compile_canary": 4,
        "overfit": 16,
        "overfit_q3_extended": 4,
        "standard_s4096": 36,
        "observation_canary": 8,
        "standard_s16384_nll": 12,
        "standard_s16384_partial": 12,
        "standard_s16384_pretrain": 12,
        "promotion_development_s4096": 48,
        "promotion_development_s16384": 48,
        "promotion_confirmation_s16384_seed2731": 24,
        "promotion_confirmation_s16384_seed3731": 24,
        "promotion_confirmation_s16384_seed4731": 24,
    }
    for matrix in array_driver.MATRICES.values():
        assert len(set(matrix)) == len(matrix)
        assert all(init_index in range(4) for *_, init_index in matrix)


def test_oracle_v2_and_promotion_catalogue_cover_all_six_cases() -> None:
    expected = set(aggregation_error_tiny_oracle.CASE_IDS)
    assert driver.LEGACY_ORACLE_BUNDLE_SCHEMA == "rjmcmc-score-nle-corrected-oracle-bundle-v1"
    assert oracle_driver.SCHEMA == "rjmcmc-score-nle-corrected-oracle-bundle-v2"
    assert driver.PROMOTION_ORACLE_BUNDLE_SCHEMA == oracle_driver.SCHEMA
    assert set(driver.PROMOTION_CASES) == expected
    assert (
        set(oracle_driver.CASE_ORDER_LADDERS) | {aggregation_error_tiny_oracle.BOUNDARY_CASE_ID} == expected
    )
    assert aggregation_error_tiny_oracle.BOUNDARY_CASE_ID not in (oracle_driver.CASE_ORDER_LADDERS)
    assert oracle_driver.METRIC_GRID_COUNTS == driver.GRID_COUNTS[-2:]
    assert driver.PROMOTION_ORACLE_ORDER_LADDERS == {
        **oracle_driver.CASE_ORDER_LADDERS,
        aggregation_error_tiny_oracle.BOUNDARY_CASE_ID: (16, 32, 64),
    }


def test_oracle_v2_builder_executes_every_predeclared_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def summary(
        case_id: str,
        order: int,
        *,
        root_chart: str = "row-first",
    ) -> SimpleNamespace:
        payload = _reference(case_id, fraction_order=order)
        if root_chart == "column-first":
            payload["method"] = "adaptive_log_total_with_column_first_gauss_jacobi_allocation"
            _rehash(payload)
        return SimpleNamespace(
            sha256=payload["sha256"],
            log_evidence=-2.0,
            posterior_mean_total=1.0,
            posterior_sd_total=0.2,
            posterior_lower_0_025=0.6,
            posterior_median=1.0,
            posterior_upper_0_975=1.4,
            represented_prior_mass=1.0,
            represented_posterior_mass=1.0,
            mode_included=True,
            scaled_quadrature_error=1.0e-10,
            maximum_relative_moment_quadrature_error=1.0e-10,
            maximum_scaled_cdf_quadrature_error=1.0e-10,
            maximum_inner_scaled_quadrature_error=1.0e-10,
            payload=lambda: payload,
        )

    def adaptive(
        case_id: str,
        *,
        fraction_order: int,
        root_chart: str = "row-first",
    ) -> SimpleNamespace:
        if root_chart == "row-first":
            calls.append((case_id, fraction_order))
        return summary(
            case_id,
            fraction_order,
            root_chart=root_chart,
        )

    boundary_case_id = aggregation_error_tiny_oracle.BOUNDARY_CASE_ID
    boundary_ladder = [
        _reference(boundary_case_id, fraction_order=order)
        for order in driver.PROMOTION_ORACLE_ORDER_LADDERS[boundary_case_id]
    ]
    boundary = _promotion_boundary_certificate(boundary_ladder)
    monkeypatch.setattr(
        aggregation_error_tiny_oracle,
        "adaptive_log_total_summary",
        adaptive,
    )
    monkeypatch.setattr(
        aggregation_error_tiny_oracle,
        "boundary_oracle_certificate",
        lambda: boundary,
    )
    monkeypatch.setattr(
        aggregation_error_tiny_oracle,
        "native_log_mass_summary",
        lambda case_id, *, lower_log_mass: _native_summary_namespace(
            case_id,
            lower_log_mass=lower_log_mass,
        ),
    )
    monkeypatch.setattr(
        oracle_driver,
        "_metric_grid_preflight",
        _promotion_metric_grid_preflight,
    )
    monkeypatch.setattr(
        oracle_driver,
        "_gradient_preflight",
        _promotion_gradient_preflight,
    )
    monkeypatch.setattr(
        oracle_driver,
        "_fixed_log_total_evidence",
        lambda *args, **kwargs: -2.0,
    )

    bundle = oracle_driver.build_bundle("a" * 40)

    assert calls == [
        (case_id, order) for case_id, orders in oracle_driver.CASE_ORDER_LADDERS.items() for order in orders
    ]
    assert set(bundle["selected_cases"]) == set(aggregation_error_tiny_oracle.CASE_IDS)
    assert bundle["pass"] is True
    without_sha = dict(bundle)
    observed_sha = without_sha.pop("sha256")
    assert observed_sha == oracle_driver._sha256_json(without_sha)


def test_gradient_preflight_freezes_allocation_orders_and_step_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, float]] = []
    mode_total = 2.0

    def exact(
        _case_id: str,
        total: float,
        *,
        fraction_order: int,
    ) -> float:
        calls.append(
            (
                fraction_order,
                math.log(float(total) / mode_total),
            )
        )
        return (1.0 + fraction_order * 1.0e-6) * math.log(float(total) / mode_total)

    monkeypatch.setattr(
        aggregation_error_tiny_oracle,
        "root_conditional_log_likelihood",
        exact,
    )
    reference = {
        "posterior_mode_total": mode_total,
        "fraction_order": 32,
    }

    preflight = oracle_driver._gradient_preflight(
        "near_gaussian__two_cell__root",
        reference,
        previous_fraction_order=16,
    )

    steps = [2.0**-12, 2.0**-13, 2.0**-14]
    assert preflight["previous_fraction_order"] == 16
    assert preflight["final_fraction_order"] == 32
    assert [row["log_total_step"] for row in preflight["final_order_step_ladder"]] == steps
    assert [order for order, _ in calls] == [32] * 6 + [16] * 2
    assert [offset for _, offset in calls] == pytest.approx(
        [signed_step for step in steps for signed_step in (step, -step)] + [steps[-1], -steps[-1]],
        abs=1.0e-15,
    )
    assert all(preflight["checks"].values())
    assert preflight["pass"] is True
    without_sha = dict(preflight)
    observed_sha = without_sha.pop("sha256")
    assert observed_sha == oracle_driver._sha256_json(without_sha)


def test_independent_oracles_gate_their_own_numerical_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def summary(
        case_id: str,
        order: int,
        *,
        column: bool = False,
        column_error: bool = False,
    ) -> SimpleNamespace:
        payload = _reference(case_id, fraction_order=order)
        if column:
            payload["method"] = "adaptive_log_total_with_column_first_gauss_jacobi_allocation"
        if column_error:
            payload["scaled_quadrature_error"] = 2.0e-6
        _rehash(payload)
        return SimpleNamespace(
            sha256=payload["sha256"],
            log_evidence=-2.0,
            posterior_mean_total=1.0,
            posterior_sd_total=0.2,
            posterior_lower_0_025=0.6,
            posterior_median=1.0,
            posterior_upper_0_975=1.4,
            represented_prior_mass=1.0,
            represented_posterior_mass=1.0,
            mode_included=True,
            scaled_quadrature_error=(2.0e-6 if column_error else 1.0e-10),
            maximum_relative_moment_quadrature_error=1.0e-10,
            maximum_scaled_cdf_quadrature_error=1.0e-10,
            payload=lambda: payload,
        )

    monkeypatch.setattr(
        aggregation_error_tiny_oracle,
        "adaptive_log_total_summary",
        lambda case_id, *, fraction_order, root_chart="row-first": summary(
            case_id,
            fraction_order,
            column=(root_chart == "column-first"),
            column_error=(root_chart == "column-first" and case_id == "near_gaussian__four_cell__root"),
        ),
    )
    monkeypatch.setattr(
        aggregation_error_tiny_oracle,
        "native_log_mass_summary",
        lambda case_id, *, lower_log_mass: _native_summary_namespace(
            case_id,
            lower_log_mass=lower_log_mass,
            inner_error=2.0e-6,
        ),
    )
    boundary_case_id = aggregation_error_tiny_oracle.BOUNDARY_CASE_ID
    boundary_ladder = [
        _reference(boundary_case_id, fraction_order=order)
        for order in driver.PROMOTION_ORACLE_ORDER_LADDERS[boundary_case_id]
    ]
    monkeypatch.setattr(
        aggregation_error_tiny_oracle,
        "boundary_oracle_certificate",
        lambda: _promotion_boundary_certificate(boundary_ladder),
    )
    monkeypatch.setattr(
        oracle_driver,
        "_metric_grid_preflight",
        _promotion_metric_grid_preflight,
    )
    monkeypatch.setattr(
        oracle_driver,
        "_gradient_preflight",
        _promotion_gradient_preflight,
    )
    monkeypatch.setattr(
        oracle_driver,
        "_fixed_log_total_evidence",
        lambda *args, **kwargs: -2.0,
    )

    bundle = oracle_driver.build_bundle("a" * 40)

    near_column_checks = bundle["independent_certificates"]["near_gaussian__four_cell__root"]["checks"]
    native_checks = bundle["independent_certificates"]["skewed__two_cell__root"]["checks"]
    assert near_column_checks["column_normalizer_quadrature_error"] is False
    assert near_column_checks["column_moment_quadrature_error"] is True
    assert near_column_checks["column_cdf_quadrature_error"] is True
    assert native_checks["native_inner_quadrature_errors"] is False
    assert native_checks["native_outer_quadrature_errors"] is True
    assert bundle["pass"] is False


def test_oracle_loader_keeps_legacy_and_promotion_schemas_distinct(
    tmp_path: Path,
) -> None:
    source_git_revision = "a" * 40
    legacy_path = tmp_path / "legacy" / "oracle_bundle.json"
    legacy_path.parent.mkdir()
    _write_bundle(legacy_path, source_git_revision)
    legacy = driver._load_oracle_bundle(
        legacy_path,
        source_git_revision,
        promotion=False,
    )
    assert set(legacy["selected_cases"]) == set(driver.SELECTED_CASES)
    with pytest.raises(ValueError, match="wrong schema"):
        driver._load_oracle_bundle(
            legacy_path,
            source_git_revision,
            promotion=True,
        )

    promotion_path = tmp_path / "promotion" / "oracle_bundle.json"
    promotion_path.parent.mkdir()
    _write_bundle(
        promotion_path,
        source_git_revision,
        promotion=True,
    )
    promotion = driver._load_oracle_bundle(
        promotion_path,
        source_git_revision,
        promotion=True,
    )
    assert set(promotion["selected_cases"]) == set(driver.PROMOTION_CASES)
    with pytest.raises(ValueError, match="wrong schema"):
        driver._load_oracle_bundle(
            promotion_path,
            source_git_revision,
            promotion=False,
        )


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    (
        (
            "swapped_grid_case",
            "promotion metric-grid preflight identity differs.",
        ),
        (
            "wrong_column_order",
            "four-cell column-chart summary identity differs.",
        ),
        (
            "wrong_column_definitions",
            "four-cell column-chart summary identity differs.",
        ),
        (
            "wrong_column_reference",
            "four-cell column-chart summary identity differs.",
        ),
        (
            "wrong_native_case",
            "native-log-mass summary SHA-256 differs.",
        ),
        (
            "wrong_native_bound",
            "native-log-mass summary SHA-256 differs.",
        ),
        (
            "altered_fixed_route",
            "boundary four-cell fixed-log-total certificate differs.",
        ),
        (
            "wrong_gradient_order",
            "promotion gradient preflight identity differs.",
        ),
        (
            "wrong_gradient_step",
            "promotion gradient preflight identity differs.",
        ),
        (
            "column_scaled_error",
            ("four-cell independent certificate numerical checks do not replay."),
        ),
        (
            "column_support",
            ("four-cell independent certificate numerical checks do not replay."),
        ),
        (
            "column_chart_comparison",
            "four-cell chart diagnostics do not replay.",
        ),
        (
            "column_check_map",
            ("four-cell independent certificate numerical checks do not replay."),
        ),
        (
            "column_integer_check",
            ("four-cell independent certificate numerical checks do not replay."),
        ),
        (
            "skew_native_outer_error",
            "skew native-log-mass certificate numerical checks do not replay.",
        ),
        (
            "skew_native_inner_error",
            "skew native-log-mass certificate numerical checks do not replay.",
        ),
        (
            "skew_native_comparison",
            "skew native-log-mass diagnostics do not replay.",
        ),
        (
            "skew_native_check_map",
            "skew native-log-mass certificate numerical checks do not replay.",
        ),
        (
            "boundary_primary_order",
            "adaptive oracle summary semantics differ.",
        ),
        (
            "boundary_tail_bound",
            "native-log-mass oracle summary semantics differ.",
        ),
        (
            "boundary_nested_definitions",
            "native-log-mass oracle summary semantics differ.",
        ),
        (
            "boundary_check_map",
            "boundary independent certificate numerical checks do not replay.",
        ),
        (
            "fixed_ladder_delta",
            "fixed-log-total diagnostics do not replay.",
        ),
        (
            "fixed_check_map",
            "fixed-log-total certificate numerical checks do not replay.",
        ),
        (
            "top_integer_check",
            "promotion oracle top-level numerical checks do not replay.",
        ),
    ),
)
def test_promotion_oracle_loader_rejects_rehashed_semantic_tamper(
    tmp_path: Path,
    tamper: str,
    expected_error: str,
) -> None:
    source_git_revision = "a" * 40
    path = tmp_path / tamper / "oracle_bundle.json"
    path.parent.mkdir()
    _write_bundle(path, source_git_revision, promotion=True)
    untouched = driver._load_oracle_bundle(
        path,
        source_git_revision,
        promotion=True,
    )
    assert untouched["pass"] is True
    payload = cast(
        dict[str, object],
        json.loads(path.read_text(encoding="ascii")),
    )
    selected = cast(dict[str, object], payload["selected_cases"])
    certificates = cast(
        dict[str, object],
        payload["independent_certificates"],
    )
    if tamper == "swapped_grid_case":
        raw_case = cast(
            dict[str, object],
            selected["near_gaussian__two_cell__root"],
        )
        grid = cast(dict[str, object], raw_case["metric_grid_preflight"])
        grid["case_id"] = "skewed__two_cell__root"
        _rehash(grid)
    elif tamper.startswith("wrong_gradient"):
        raw_case = cast(
            dict[str, object],
            selected["near_gaussian__two_cell__root"],
        )
        gradient = cast(
            dict[str, object],
            raw_case["gradient_preflight"],
        )
        if tamper == "wrong_gradient_order":
            gradient["previous_fraction_order"] = 8
        else:
            ladder = cast(
                list[dict[str, object]],
                gradient["final_order_step_ladder"],
            )
            ladder[-1]["log_total_step"] = 2.0**-15
        _rehash(gradient)
    elif tamper.startswith("column_"):
        certificate = cast(
            dict[str, object],
            certificates["near_gaussian__four_cell__root"],
        )
        column = cast(dict[str, object], certificate["column_summary"])
        if tamper == "column_scaled_error":
            column["scaled_quadrature_error"] = 2.0e-6
            _rehash(column)
        elif tamper == "column_support":
            column["represented_prior_mass"] = 0.9
            _rehash(column)
        elif tamper == "column_chart_comparison":
            column["log_evidence"] = -1.9
            _rehash(column)
        elif tamper == "column_integer_check":
            checks = cast(dict[str, object], certificate["checks"])
            checks["log_evidence"] = 1
        else:
            checks = cast(dict[str, object], certificate["checks"])
            checks["unrecognized_check"] = True
        _rehash(certificate)
    elif tamper.startswith("skew_native_"):
        certificate = cast(
            dict[str, object],
            certificates["skewed__two_cell__root"],
        )
        summaries = cast(list[dict[str, object]], certificate["summaries"])
        if tamper == "skew_native_outer_error":
            summaries[-1]["scaled_quadrature_error"] = 2.0e-6
            _rehash(summaries[-1])
        elif tamper == "skew_native_inner_error":
            summaries[-1]["maximum_inner_scaled_quadrature_error"] = 2.0e-6
            _rehash(summaries[-1])
        elif tamper == "skew_native_comparison":
            summaries[-1]["posterior_mean_total"] = 1.1
            _rehash(summaries[-1])
        else:
            checks = cast(dict[str, object], certificate["checks"])
            checks["unrecognized_check"] = True
        _rehash(certificate)
    elif tamper.startswith("boundary_"):
        boundary_certificate = cast(
            dict[str, object],
            payload["boundary_independent_certificate"],
        )
        if tamper == "boundary_primary_order":
            primary = cast(
                list[dict[str, object]],
                boundary_certificate["primary_order_ladder"],
            )
            primary[0]["fraction_order"] = 8
            _rehash(primary[0])
        elif tamper == "boundary_tail_bound":
            tail = cast(
                list[dict[str, object]],
                boundary_certificate["independent_tail_ladder"],
            )
            tail[0]["lower_log_mass"] = -41.0
            _rehash(tail[0])
        elif tamper == "boundary_nested_definitions":
            tail = cast(
                list[dict[str, object]],
                boundary_certificate["independent_tail_ladder"],
            )
            tail[0]["definitions_sha256"] = "0" * 64
            _rehash(tail[0])
        else:
            checks = cast(
                dict[str, object],
                boundary_certificate["checks"],
            )
            checks["unrecognized_check"] = True
        _rehash(boundary_certificate)
    elif tamper.startswith("fixed_"):
        certificate = cast(
            dict[str, object],
            certificates["boundary_heavy__four_cell__root"],
        )
        fixed = cast(
            dict[str, object],
            certificate["fixed_log_total_column_chart"],
        )
        if tamper == "fixed_ladder_delta":
            ladder = cast(list[float], fixed["log_evidence_ladder"])
            ladder[-1] = -1.9
        else:
            fixed_checks = cast(dict[str, object], fixed["checks"])
            fixed_checks["unrecognized_check"] = True
            certificate_checks = cast(
                dict[str, object],
                certificate["checks"],
            )
            certificate_checks["unrecognized_check"] = True
        _rehash(certificate)
    elif tamper.startswith("wrong_column"):
        certificate = cast(
            dict[str, object],
            certificates["near_gaussian__four_cell__root"],
        )
        column = cast(dict[str, object], certificate["column_summary"])
        if tamper == "wrong_column_order":
            column["fraction_order"] = 12
            _rehash(column)
        elif tamper == "wrong_column_definitions":
            column["definitions_sha256"] = "0" * 64
            _rehash(column)
        else:
            certificate["row_reference_sha256"] = "0" * 64
        _rehash(certificate)
    elif tamper.startswith("wrong_native"):
        certificate = cast(
            dict[str, object],
            certificates["skewed__two_cell__root"],
        )
        summaries = cast(list[dict[str, object]], certificate["summaries"])
        if tamper == "wrong_native_case":
            summaries[0]["case_id"] = "near_gaussian__two_cell__root"
        else:
            summaries[0]["lower_log_mass"] = -81.0
        _rehash(summaries[0])
        _rehash(certificate)
    elif tamper == "top_integer_check":
        top_checks = cast(dict[str, object], payload["checks"])
        top_checks["boundary_independent_certificate"] = 1
    else:
        certificate = cast(
            dict[str, object],
            certificates["boundary_heavy__four_cell__root"],
        )
        fixed = cast(
            dict[str, object],
            certificate["fixed_log_total_column_chart"],
        )
        fixed["prior_tail_probability"] = 1.0e-12
        _rehash(certificate)
    _rewrite_bundle(path, payload)

    with pytest.raises(ValueError, match=re.escape(expected_error)):
        driver._load_oracle_bundle(
            path,
            source_git_revision,
            promotion=True,
        )


def test_corrected_slurm_assets_use_shared_nodes_and_array_contract() -> None:
    assets = Path("docs/plans/rjmcmc_score_regularized_nle_corrected_assets")
    oracle_text = (assets / "run_corrected_oracle.sbatch").read_text(encoding="utf-8")
    array_text = (assets / "run_corrected_array.sbatch").read_text(encoding="utf-8")
    merger_text = (assets / "run_corrected_promotion_merge.sbatch").read_text(encoding="utf-8")
    certifier_text = (assets / "run_corrected_promotion_certify.sbatch").read_text(encoding="utf-8")
    expected_resources = {
        oracle_text: ("#SBATCH --mem=2G", "#SBATCH --time=00:45:00"),
        array_text: ("#SBATCH --mem=5G", "#SBATCH --time=00:20:00"),
        merger_text: ("#SBATCH --mem=3G", "#SBATCH --time=00:45:00"),
        certifier_text: ("#SBATCH --mem=3G", "#SBATCH --time=00:30:00"),
    }
    for text, resources in expected_resources.items():
        assert "#SBATCH --exclusive" not in text
        assert all(resource in text for resource in resources)
    assert "SLURM_ARRAY_TASK_ID" in array_text
    assert "SLURM_ARRAY_JOB_ID" in array_text
    for text in expected_resources:
        assert "#SBATCH --output=" not in text
        assert "#SBATCH --error=" not in text
        git_module = "module load git/2.45.1-pqk5"
        assert git_module in text
        assert text.index(git_module) < text.index('git -C "${source_root}"')
        assert 'exec "${python_bin}" \\' in text
    assert oracle_text.rstrip().endswith('--output-root "${run_root}"')
    assert array_text.rstrip().endswith('--patience "${NLE_PATIENCE}"')
    assert merger_text.rstrip().endswith('--output-root "${run_root}"')
    assert certifier_text.rstrip().endswith('--certificate-tag "${NLE_CERTIFICATE_TAG}"')
    for text in (merger_text, certifier_text):
        assert "NLE_ARTIFACT_REVISION" in text
        assert '--artifact-source-git-revision "${NLE_ARTIFACT_REVISION}"' in text
        assert '--evaluation-source-git-revision "${NLE_REVISION}"' in text


def test_promotion_execution_identity_freezes_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in driver.PROMOTION_EXECUTION_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    identity = driver._execution_identity(promotion=True)
    without_sha = dict(identity)
    observed_sha = without_sha.pop("sha256")
    assert identity["environment"] == driver.PROMOTION_EXECUTION_ENVIRONMENT
    assert identity["jax_x64_enabled"] is True
    assert identity["autodiff_schedule"] == {
        "observation_score_parameter_gradient": ("outer_reverse_parameter_gradient_over_forward_jvp"),
        "partial_score_parameter_gradient": ("outer_reverse_parameter_gradient_over_forward_jvp"),
    }
    assert observed_sha == driver._sha256_json(without_sha)

    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(
        ValueError,
        match="compiler/thread environment differs",
    ):
        driver._execution_identity(promotion=True)
    exploratory = driver._execution_identity(promotion=False)
    assert exploratory["environment"]["OMP_NUM_THREADS"] == "2"


def test_runtime_identity_binds_versions_and_lock_bytes() -> None:
    identity = driver._runtime_identity()
    without_sha = dict(identity)
    observed_sha = without_sha.pop("sha256")
    assert set(identity["packages"]) == {
        "equinox",
        "flowjax",
        "jax",
        "jaxlib",
        "numpy",
        "optax",
        "paramax",
        "scipy",
    }
    assert len(identity["pixi_lock_sha256"]) == 64
    assert observed_sha == driver._sha256_json(without_sha)


def test_oracle_completion_binds_payload_and_exact_report_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_git_revision = "a" * 40
    bundle = _passing_oracle_bundle(source_git_revision)
    publication_order: list[Path] = []
    original_atomic_json = oracle_driver._atomic_json

    def record_atomic_json(path: Path, payload: object) -> str:
        digest = original_atomic_json(path, payload)
        publication_order.append(path)
        return digest

    monkeypatch.setattr(oracle_driver, "_atomic_json", record_atomic_json)
    report, completion_path = oracle_driver._publish_bundle(
        tmp_path,
        bundle,
        source_git_revision,
    )

    assert publication_order == [report, completion_path]
    expected_report_bytes = oracle_driver._pretty_json_bytes(bundle)
    assert report.read_bytes() == expected_report_bytes
    completion = json.loads(completion_path.read_text(encoding="ascii"))
    assert set(completion) == {
        "schema",
        "source_git_revision",
        "report_path",
        "oracle_bundle_payload_sha256",
        "oracle_bundle_file_sha256",
        "completion_marker_published_last",
    }
    assert completion == {
        "schema": oracle_driver.SCHEMA,
        "source_git_revision": source_git_revision,
        "report_path": str(report),
        "oracle_bundle_payload_sha256": bundle["sha256"],
        "oracle_bundle_file_sha256": hashlib.sha256(expected_report_bytes).hexdigest(),
        "completion_marker_published_last": True,
    }


def test_oracle_publication_is_create_only_and_preserves_first_evidence(
    tmp_path: Path,
) -> None:
    source_git_revision = "a" * 40
    bundle = _passing_oracle_bundle(source_git_revision)
    report, completion = oracle_driver._publish_bundle(
        tmp_path,
        bundle,
        source_git_revision,
    )
    original_report = report.read_bytes()
    original_completion = completion.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to replace"):
        oracle_driver._publish_bundle(
            tmp_path,
            bundle,
            source_git_revision,
        )

    assert report.read_bytes() == original_report
    assert completion.read_bytes() == original_completion


def test_merger_reports_missing_array_tasks_without_a_scientific_decision(
    tmp_path: Path,
) -> None:
    summary = merger.merge(
        "compile_canary",
        "missing-test",
        "a" * 40,
        tmp_path,
    )
    assert not summary["complete"]
    assert summary["complete_attempt_count"] == 0
    assert len(summary["missing_attempts"]) == 4
    assert summary["approximate_evidence_used_as_structural_weight"] is False
    assert summary["scientific_decision"].startswith("none:")


def test_one_small_overfit_attempt_publishes_completion_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "GRID_COUNTS", (16, 32, 64))
    source_git_revision = "a" * 40
    bundle = tmp_path / "oracle_bundle.json"
    _write_bundle(bundle, source_git_revision)
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    arguments = argparse.Namespace(
        mode="overfit",
        case_id="near_gaussian__two_cell__root",
        sample_count=256,
        config_id="nll_only",
        init_index=0,
        attempt_tag="test",
        base_seed=731,
        source_git_revision=source_git_revision,
        oracle_bundle=bundle,
        output_root=tmp_path,
        learning_rate=5.0e-4,
        batch_size=256,
        microbatch_size=64,
        max_epochs=1,
        patience=0,
    )
    report = driver.run_attempt(arguments, attempt_root)
    assert report["interpretation"]["status"] == ("exploratory_result_not_promotion")
    assert report["interpretation"]["overfit_validation_role"] == ("same catalogue optimizer diagnostic")
    assert all(report["streams"]["separation"]["checks"].values())
    assert report["initialization_loss_diagnostics"]["measured_before_loss_weights_applied"]
    assert report["artifact"]["byte_replay_pass"]
    replay = report["artifact"]["canonical_replay"]
    assert replay["canonical_replay_used_for_scientific_evaluation"]
    assert replay["diagnostic"] == "non_authoritative_layout_roundoff_diagnostic"
    assert replay["gating"] is False
    assert replay["flow_tree_structure_identical"]
    assert replay["flow_float_leaves_bitwise_identical"]
    assert replay["spectrum_arrays_bitwise_identical"]
    assert replay["canonical_values_finite"]
    assert replay["trained_values_finite"]
    assert replay["within_advisory_roundoff_range"]
    assert len(report["scientific_evaluation"]["grid"]["ladder"]) == 3
    assert report["scientific_evaluation"]["vectorized_public_likelihood_parity"]["pass"]
    assert (attempt_root / "artifact.bin").is_file()
    assert (attempt_root / "report.json").is_file()
    report_bytes = (attempt_root / "report.json").read_bytes()
    completion = json.loads((attempt_root / "COMPLETE.json").read_text(encoding="ascii"))
    assert completion["completion_marker_published_last"]
    assert completion["report_payload_sha256"] == report["sha256"]
    assert completion["report_file_sha256"] == hashlib.sha256(report_bytes).hexdigest()
    assert (
        completion["serialized_artifact_file_sha256"]
        == hashlib.sha256((attempt_root / "artifact.bin").read_bytes()).hexdigest()
    )


def test_learned_gradient_step_failure_only_invalidates_gradient_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = "near_gaussian__two_cell__root"
    case = aggregation_error_tiny_oracle.tiny_root_case(case_id)
    shapes, rate, _, _, _ = case.arrays()
    gamma_shape = float(shapes.sum())
    mode_total = (gamma_shape - 1.0) / rate
    reference = _reference(
        case_id,
        fraction_order=(driver.PROMOTION_ORACLE_ORDER_LADDERS[case_id][-1]),
    )
    reference.update(
        {
            "posterior_mode_total": mode_total,
            "log_evidence": 0.0,
            "posterior_mean_total": gamma_shape / rate,
            "posterior_sd_total": math.sqrt(gamma_shape) / rate,
            "posterior_lower_0_025": float(
                driver.stats.gamma.ppf(
                    0.025,
                    a=gamma_shape,
                    scale=1.0 / rate,
                )
            ),
            "posterior_median": float(
                driver.stats.gamma.ppf(
                    0.5,
                    a=gamma_shape,
                    scale=1.0 / rate,
                )
            ),
            "posterior_upper_0_975": float(
                driver.stats.gamma.ppf(
                    0.975,
                    a=gamma_shape,
                    scale=1.0 / rate,
                )
            ),
        }
    )
    exact_cache: dict[int, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for count in driver.GRID_COUNTS:
        totals = np.asarray(
            driver.stats.gamma.ppf(
                (np.arange(count, dtype=np.float64) + 0.5) / count,
                a=gamma_shape,
                scale=1.0 / rate,
            ),
            dtype=np.float64,
        )
        exact = np.zeros(count, dtype=np.float64)
        exact_cache[count] = exact
        rows.append(
            {
                "count": count,
                "total_grid_sha256": hashlib.sha256(
                    np.ascontiguousarray(totals, dtype="<f8").tobytes()
                ).hexdigest(),
                "exact_log_likelihood_sha256": hashlib.sha256(
                    np.ascontiguousarray(exact, dtype="<f8").tobytes()
                ).hexdigest(),
            }
        )
    promotion_case = {
        "reference": reference,
        "pass": True,
        "metric_grid_preflight": {
            "rows": rows,
            "pass": True,
        },
        "gradient_preflight": {
            "reference_gradient": 0.0,
            "pass": True,
        },
    }
    artifact = cast(
        driver.ScoreRegularizedRootFlow,
        SimpleNamespace(
            log_likelihood_batch=lambda _observation, totals: np.zeros_like(
                totals,
                dtype=np.float64,
            ),
            log_likelihood=lambda _observation, total: (
                math.log(float(total) / mode_total) + 1.0e7 * math.log(float(total) / mode_total) ** 3
            ),
        ),
    )
    monkeypatch.setattr(
        driver,
        "_vectorized_artifact_log_likelihood",
        lambda _artifact, _observation, totals: np.zeros_like(
            totals,
            dtype=np.float64,
        ),
    )

    evaluation = driver._scientific_grid_evaluation(
        artifact,
        case_id,
        reference,
        interpolate_posterior_quantiles=True,
        exact_grid_cache=exact_cache,
        promotion_oracle_case=promotion_case,
    )

    flags = evaluation["metric_interpretability"]
    assert flags == {
        "prior_weighted_median_log_likelihood_error": True,
        "posterior_weighted_p99_log_likelihood_error": True,
        "log_evidence": True,
        "posterior_moments": True,
        "posterior_quantiles": True,
        "retained_mass_gradient": False,
    }
    assert evaluation["scientific_metrics_interpretable"] is False
    gradient = cast(dict[str, float], evaluation["gradient"])
    assert gradient["last_two_scaled_error_delta"] > gradient["numerical_tolerance"]


def test_scientific_scoring_rejects_exact_grid_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = "near_gaussian__two_cell__root"
    count = 16
    monkeypatch.setattr(driver, "GRID_COUNTS", (count,))
    reference = _reference(case_id, fraction_order=32)
    promotion_case = {
        "reference": reference,
        "pass": True,
        "metric_grid_preflight": {
            "rows": [
                {
                    "count": count,
                    "total_grid_sha256": "0" * 64,
                    "exact_log_likelihood_sha256": "1" * 64,
                }
            ],
            "pass": True,
        },
        "gradient_preflight": {
            "reference_gradient": 0.0,
            "pass": True,
        },
    }
    artifact = cast(
        driver.ScoreRegularizedRootFlow,
        SimpleNamespace(),
    )

    with pytest.raises(
        ValueError,
        match="exact grid differs from the oracle preflight",
    ):
        driver._scientific_grid_evaluation(
            artifact,
            case_id,
            reference,
            exact_grid_cache={count: np.zeros(count)},
            promotion_oracle_case=promotion_case,
        )


def _diagnostic_fake(
    values: np.ndarray,
    *,
    flow: object | None = None,
) -> driver.ScoreRegularizedRootFlow:
    spectrum = SimpleNamespace(
        observation_mean_design=np.asarray((0.2, -0.1)),
        noise_sd=np.asarray((0.7, 1.1)),
        basis=np.eye(2),
        eigenvalues=np.asarray((0.8, 0.25)),
    )
    return cast(
        driver.ScoreRegularizedRootFlow,
        SimpleNamespace(
            flow=np.asarray((1.0, 2.0)) if flow is None else flow,
            spectrum=spectrum,
            log_likelihood_batch=lambda _observation, _totals: values,
        ),
    )


def test_transient_layout_roundoff_is_recorded_but_does_not_gate() -> None:
    trained_values = np.asarray((-4.449186, 0.519406, -1.342784))
    canonical_values = trained_values.copy()
    canonical_values[-1] += 2.398081733190338e-14
    diagnostic = driver._canonical_replay_diagnostics(
        _diagnostic_fake(trained_values),
        _diagnostic_fake(canonical_values),
        np.asarray((0.2, -0.4)),
        np.asarray((0.5, 1.0, 1.5)),
    )
    assert diagnostic["gating"] is False
    assert diagnostic["within_advisory_roundoff_range"]
    assert float(
        diagnostic["trained_to_canonical_likelihood_max_absolute_error_nat"]  # type: ignore[arg-type]
    ) == pytest.approx(2.398081733190338e-14)
    assert float(
        diagnostic["trained_to_canonical_likelihood_max_output_ulp_error"]  # type: ignore[arg-type]
    ) == pytest.approx(108.0)


def test_canonical_replay_diagnostic_still_gates_tree_and_leaf_identity() -> None:
    values = np.asarray((-1.0, -2.0, -3.0))
    with pytest.raises(RuntimeError, match="tree structure"):
        driver._canonical_replay_diagnostics(
            _diagnostic_fake(values),
            _diagnostic_fake(
                values,
                flow={"parameters": np.asarray((1.0, 2.0))},
            ),
            np.asarray((0.2, -0.4)),
            np.asarray((0.5, 1.0, 1.5)),
        )
    with pytest.raises(AssertionError):
        driver._canonical_replay_diagnostics(
            _diagnostic_fake(values),
            _diagnostic_fake(values, flow=np.asarray((1.0, 3.0))),
            np.asarray((0.2, -0.4)),
            np.asarray((0.5, 1.0, 1.5)),
        )
