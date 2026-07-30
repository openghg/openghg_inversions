#!/usr/bin/env python3
"""Build the observation-blind corrected tiny-root oracle bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, cast

import numpy as np
from scipy import special, stats

from openghg_inversions.experimental.rjmcmc import (
    aggregation_error_tiny_oracle as oracle,
)

SCHEMA = "rjmcmc-score-nle-corrected-oracle-bundle-v2"
PROMOTION_CASES = (
    "near_gaussian__two_cell__root",
    "near_gaussian__four_cell__root",
    "skewed__two_cell__root",
    "skewed__four_cell__root",
    "boundary_heavy__two_cell__root",
    "boundary_heavy__four_cell__root",
)
CASE_ORDER_LADDERS = {
    "near_gaussian__two_cell__root": (16, 32),
    "near_gaussian__four_cell__root": (8, 12, 16),
    "skewed__two_cell__root": (8, 16, 32),
    "skewed__four_cell__root": (8, 12, 16),
    "boundary_heavy__four_cell__root": (12, 16, 24),
}
LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT = 0.0025
METRIC_GRID_COUNTS = (4_096, 8_192)
METRIC_GRID_EVIDENCE_TOLERANCE_NAT = 0.005
METRIC_GRID_LOCATION_TOLERANCE_REFERENCE_SD = 0.005
METRIC_GRID_SD_RELATIVE_TOLERANCE = 0.002
GRADIENT_NUMERICAL_TOLERANCE = 0.005
GRADIENT_LOG_TOTAL_STEPS = (2.0**-12, 2.0**-13, 2.0**-14)
FOUR_CELL_CHART_CASES = (
    "near_gaussian__four_cell__root",
    "skewed__four_cell__root",
    "boundary_heavy__four_cell__root",
)


def _logsumexp_scalar(values: np.ndarray) -> float:
    return float(cast(Any, special.logsumexp)(values))


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )


def _sha256_json(payload: object) -> str:
    compact = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(compact.encode("ascii")).hexdigest()


def _pretty_json_bytes(payload: object) -> bytes:
    """Return the exact on-disk JSON representation for an oracle file."""
    return f"{_canonical_json(payload)}\n".encode("ascii")


def _atomic_json(path: Path, payload: object) -> str:
    """Create one JSON file atomically and return its exact file-byte digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing evidence: {path}")
    content = _pretty_json_bytes(payload)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


def _posterior_grid_summary(
    totals: np.ndarray,
    log_likelihood: np.ndarray,
    *,
    gamma_shape: float,
    gamma_rate: float,
) -> dict[str, float]:
    """Summarize equal-prior-probability bins with within-bin quantiles."""
    log_weights = log_likelihood - _logsumexp_scalar(log_likelihood)
    weights = np.exp(log_weights)
    mean = float(weights @ totals)
    sd = math.sqrt(float(weights @ np.square(totals - mean)))

    def quantile(probability: float) -> float:
        cumulative = np.cumsum(weights)
        index = min(
            int(np.searchsorted(cumulative, probability, side="left")),
            weights.size - 1,
        )
        previous = 0.0 if index == 0 else float(cumulative[index - 1])
        fraction = (probability - previous) / float(weights[index])
        prior_probability = (index + min(max(fraction, 0.0), 1.0)) / weights.size
        return float(
            stats.gamma.ppf(
                prior_probability,
                a=gamma_shape,
                scale=1.0 / gamma_rate,
            )
        )

    return {
        "log_evidence": (_logsumexp_scalar(log_likelihood) - math.log(log_likelihood.size)),
        "posterior_mean_total": mean,
        "posterior_sd_total": sd,
        "posterior_lower_0_025": quantile(0.025),
        "posterior_median": quantile(0.5),
        "posterior_upper_0_975": quantile(0.975),
    }


def _metric_grid_preflight(
    case_id: str,
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Certify the exact evaluation grid before any learned model is fit."""
    case = oracle.tiny_root_case(case_id)
    shapes, rate, _, _, _ = case.arrays()
    gamma_shape = float(shapes.sum())
    rows: list[dict[str, Any]] = []
    for count in METRIC_GRID_COUNTS:
        probabilities = (np.arange(count, dtype=np.float64) + 0.5) / count
        totals = np.asarray(
            stats.gamma.ppf(
                probabilities,
                a=gamma_shape,
                scale=1.0 / rate,
            ),
            dtype=np.float64,
        )
        exact = np.asarray(
            oracle.root_conditional_log_likelihood(
                case_id,
                totals,
                fraction_order=int(reference["fraction_order"]),
            ),
            dtype=np.float64,
        )
        rows.append(
            {
                "count": count,
                "posterior": _posterior_grid_summary(
                    totals,
                    exact,
                    gamma_shape=gamma_shape,
                    gamma_rate=rate,
                ),
                "total_grid_sha256": hashlib.sha256(
                    np.ascontiguousarray(totals, dtype="<f8").tobytes()
                ).hexdigest(),
                "exact_log_likelihood_sha256": hashlib.sha256(
                    np.ascontiguousarray(exact, dtype="<f8").tobytes()
                ).hexdigest(),
                "finite": bool(np.all(np.isfinite(exact))),
            }
        )
    previous = rows[-2]["posterior"]
    final = rows[-1]["posterior"]
    reference_sd = float(reference["posterior_sd_total"])
    last_two = {
        "log_evidence_delta_nat": abs(float(final["log_evidence"]) - float(previous["log_evidence"])),
        "posterior_mean_delta_reference_sd": abs(
            float(final["posterior_mean_total"]) - float(previous["posterior_mean_total"])
        )
        / reference_sd,
        "posterior_sd_relative_delta": abs(
            float(final["posterior_sd_total"]) - float(previous["posterior_sd_total"])
        )
        / reference_sd,
        "posterior_quantile_delta_reference_sd": max(
            abs(float(final[key]) - float(previous[key]))
            for key in (
                "posterior_lower_0_025",
                "posterior_median",
                "posterior_upper_0_975",
            )
        )
        / reference_sd,
    }
    adaptive = {
        "log_evidence_error_nat": abs(float(final["log_evidence"]) - float(reference["log_evidence"])),
        "posterior_mean_error_reference_sd": abs(
            float(final["posterior_mean_total"]) - float(reference["posterior_mean_total"])
        )
        / reference_sd,
        "posterior_sd_relative_error": abs(float(final["posterior_sd_total"]) - reference_sd) / reference_sd,
        "posterior_quantile_error_reference_sd": max(
            abs(float(final[key]) - float(reference[key]))
            for key in (
                "posterior_lower_0_025",
                "posterior_median",
                "posterior_upper_0_975",
            )
        )
        / reference_sd,
    }
    checks = {
        "finite": all(bool(row["finite"]) for row in rows),
        "last_two_evidence": (last_two["log_evidence_delta_nat"] <= METRIC_GRID_EVIDENCE_TOLERANCE_NAT),
        "last_two_mean": (
            last_two["posterior_mean_delta_reference_sd"] <= METRIC_GRID_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "last_two_sd": (last_two["posterior_sd_relative_delta"] <= METRIC_GRID_SD_RELATIVE_TOLERANCE),
        "last_two_quantiles": (
            last_two["posterior_quantile_delta_reference_sd"] <= METRIC_GRID_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "adaptive_evidence": (adaptive["log_evidence_error_nat"] <= METRIC_GRID_EVIDENCE_TOLERANCE_NAT),
        "adaptive_mean": (
            adaptive["posterior_mean_error_reference_sd"] <= METRIC_GRID_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "adaptive_sd": (adaptive["posterior_sd_relative_error"] <= METRIC_GRID_SD_RELATIVE_TOLERANCE),
        "adaptive_quantiles": (
            adaptive["posterior_quantile_error_reference_sd"] <= METRIC_GRID_LOCATION_TOLERANCE_REFERENCE_SD
        ),
    }
    without_sha: dict[str, Any] = {
        "case_id": case_id,
        "construction": "equal-prior-probability midpoint bins",
        "posterior_quantile_rule": ("within-bin interpolation under piecewise-constant likelihood"),
        "counts": list(METRIC_GRID_COUNTS),
        "rows": rows,
        "last_two": last_two,
        "final_errors_from_adaptive_reference": adaptive,
        "checks": checks,
        "pass": all(checks.values()),
    }
    return {**without_sha, "sha256": _sha256_json(without_sha)}


def _gradient_preflight(
    case_id: str,
    reference: dict[str, Any],
    *,
    previous_fraction_order: int,
) -> dict[str, Any]:
    """Certify allocation-order and finite-difference refinement."""
    total = float(reference["posterior_mode_total"])
    final_order = int(reference["fraction_order"])

    def gradient(order: int, step: float) -> float:
        upper = float(
            oracle.root_conditional_log_likelihood(
                case_id,
                total * math.exp(step),
                fraction_order=order,
            )
        )
        lower = float(
            oracle.root_conditional_log_likelihood(
                case_id,
                total * math.exp(-step),
                fraction_order=order,
            )
        )
        return (upper - lower) / (2.0 * step)

    final_step_ladder = [
        {
            "log_total_step": step,
            "gradient": gradient(final_order, step),
        }
        for step in GRADIENT_LOG_TOTAL_STEPS
    ]
    previous_order_gradient = gradient(
        previous_fraction_order,
        GRADIENT_LOG_TOTAL_STEPS[-1],
    )
    final_gradient = float(final_step_ladder[-1]["gradient"])
    previous_step_gradient = float(final_step_ladder[-2]["gradient"])
    diagnostics = {
        "finite_difference_refinement_scaled_error": (
            abs(final_gradient - previous_step_gradient) / (1.0 + abs(final_gradient))
        ),
        "allocation_order_refinement_scaled_error": (
            abs(final_gradient - previous_order_gradient) / (1.0 + abs(final_gradient))
        ),
    }
    checks = {
        "finite": all(math.isfinite(float(row["gradient"])) for row in final_step_ladder)
        and math.isfinite(previous_order_gradient),
        "finite_difference_refined": (
            diagnostics["finite_difference_refinement_scaled_error"] <= GRADIENT_NUMERICAL_TOLERANCE
        ),
        "allocation_order_refined": (
            diagnostics["allocation_order_refinement_scaled_error"] <= GRADIENT_NUMERICAL_TOLERANCE
        ),
    }
    without_sha: dict[str, Any] = {
        "case_id": case_id,
        "state": "adaptive exact posterior mode total",
        "total_mass": total,
        "previous_fraction_order": previous_fraction_order,
        "final_fraction_order": final_order,
        "final_order_step_ladder": final_step_ladder,
        "previous_order_final_step_gradient": previous_order_gradient,
        "reference_gradient": final_gradient,
        "diagnostics": diagnostics,
        "tolerance": GRADIENT_NUMERICAL_TOLERANCE,
        "checks": checks,
        "pass": all(checks.values()),
    }
    return {**without_sha, "sha256": _sha256_json(without_sha)}


def _fixed_log_total_evidence(
    case_id: str,
    *,
    fraction_order: int,
    total_order: int,
) -> float:
    """Independent fixed Gauss-Legendre integral in log total."""
    case = oracle.tiny_root_case(case_id)
    shapes, rate, _, _, _ = case.arrays()
    gamma_shape = float(shapes.sum())
    lower = float(
        stats.gamma.ppf(
            1.0e-15,
            a=gamma_shape,
            scale=1.0 / rate,
        )
    )
    upper = float(
        stats.gamma.ppf(
            1.0 - 1.0e-15,
            a=gamma_shape,
            scale=1.0 / rate,
        )
    )
    lower_z = math.log(max(lower, np.finfo(np.float64).tiny))
    upper_z = math.log(upper)
    nodes, weights = np.polynomial.legendre.leggauss(total_order)
    half_width = 0.5 * (upper_z - lower_z)
    center = 0.5 * (upper_z + lower_z)
    z = center + half_width * nodes
    totals = np.exp(z)
    conditional = np.asarray(
        oracle.root_conditional_log_likelihood(
            case_id,
            totals,
            fraction_order=fraction_order,
            root_chart="column-first",
        ),
        dtype=np.float64,
    )
    log_prior_with_jacobian = (
        gamma_shape * math.log(rate) - float(special.gammaln(gamma_shape)) + gamma_shape * z - rate * totals
    )
    return _logsumexp_scalar(np.log(weights) + math.log(half_width) + log_prior_with_jacobian + conditional)


def _publish_bundle(
    output_root: Path,
    bundle: dict[str, Any],
    source_git_revision: str,
) -> tuple[Path, Path]:
    """Publish one create-only bundle and bind its payload and file bytes."""
    report = output_root / "oracle" / "oracle_bundle.json"
    completion = output_root / "oracle" / "COMPLETE.json"
    for path in (report, completion):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to replace existing oracle evidence: {path}")
    without_sha = dict(bundle)
    payload_sha256 = without_sha.pop("sha256", None)
    if payload_sha256 != _sha256_json(without_sha):
        raise ValueError("oracle bundle canonical payload SHA-256 does not replay.")
    if bundle.get("source_git_revision") != source_git_revision:
        raise ValueError("oracle bundle source revision does not match publication.")
    report_file_sha256 = _atomic_json(report, bundle)
    if not bundle["pass"]:
        raise RuntimeError("corrected oracle bundle did not converge.")
    _atomic_json(
        completion,
        {
            "schema": SCHEMA,
            "source_git_revision": source_git_revision,
            "report_path": str(report),
            "oracle_bundle_payload_sha256": payload_sha256,
            "oracle_bundle_file_sha256": report_file_sha256,
            "completion_marker_published_last": True,
        },
    )
    return report, completion


def _finite_number(mapping: dict[str, Any], key: str, *, context: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{context} field {key!r} is not finite.")
    return float(value)


def _require_exact_keys(
    mapping: dict[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if set(mapping) != expected:
        raise ValueError(f"{context} has the wrong exact schema.")


def _require_hashed_mapping(mapping: dict[str, Any], *, context: str) -> None:
    without_sha = dict(mapping)
    observed_sha = without_sha.pop("sha256", None)
    if not isinstance(observed_sha, str) or _sha256_json(without_sha) != observed_sha:
        raise ValueError(f"{context} SHA-256 does not replay.")


def _require_checks(
    payload: dict[str, Any],
    expected: dict[str, bool],
    *,
    context: str,
) -> None:
    observed = payload.get("checks")
    if (
        not isinstance(observed, dict)
        or set(observed) != set(expected)
        or any(
            type(observed[key]) is not bool or observed[key] is not value for key, value in expected.items()
        )
        or type(payload.get("pass")) is not bool
        or payload.get("pass") is not all(expected.values())
    ):
        raise ValueError(f"{context} numerical checks do not replay.")


def _validate_adaptive_summary(
    summary: dict[str, Any],
    *,
    case_id: str,
    fraction_order: int,
    chart: str,
) -> None:
    _require_exact_keys(
        summary,
        set(oracle.AdaptiveRootSummary.__dataclass_fields__),
        context="adaptive oracle summary",
    )
    _require_hashed_mapping(summary, context="adaptive oracle summary")
    expected_method = {
        "row-first": "adaptive_log_total_with_gauss_jacobi_allocation",
        "column-first": ("adaptive_log_total_with_column_first_gauss_jacobi_allocation"),
    }[chart]
    if (
        summary.get("schema") != oracle.SCHEMA
        or summary.get("case_id") != case_id
        or summary.get("definitions_sha256") != oracle.definitions_sha256()
        or summary.get("method") != expected_method
        or summary.get("fraction_order") != fraction_order
        or summary.get("epsabs") != 1.0e-10
        or summary.get("epsrel") != 1.0e-10
        or summary.get("posterior_mass_accounting")
        != (
            "conservative lower bound from omitted Gamma prior mass times "
            "the global normalized-Gaussian density upper bound"
        )
        or not isinstance(summary.get("mode_included"), bool)
    ):
        raise ValueError("adaptive oracle summary semantics differ.")
    finite_keys = {
        "lower_log_total",
        "upper_log_total",
        "posterior_mode_total",
        "log_evidence",
        "posterior_mean_total",
        "posterior_sd_total",
        "posterior_lower_0_025",
        "posterior_median",
        "posterior_upper_0_975",
        "scaled_quadrature_error",
        "maximum_relative_moment_quadrature_error",
        "maximum_scaled_cdf_quadrature_error",
        "represented_prior_mass",
        "represented_posterior_mass",
    }
    values = {key: _finite_number(summary, key, context="adaptive oracle summary") for key in finite_keys}
    if (
        values["lower_log_total"] >= values["upper_log_total"]
        or values["posterior_mode_total"] <= 0.0
        or not (
            values["lower_log_total"] < math.log(values["posterior_mode_total"]) < values["upper_log_total"]
        )
        or values["posterior_sd_total"] <= 0.0
        or not (
            values["posterior_lower_0_025"] <= values["posterior_median"] <= values["posterior_upper_0_975"]
        )
        or any(
            values[key] < 0.0
            for key in (
                "scaled_quadrature_error",
                "maximum_relative_moment_quadrature_error",
                "maximum_scaled_cdf_quadrature_error",
            )
        )
        or not 0.0 <= values["represented_prior_mass"] <= 1.0
        or not 0.0 <= values["represented_posterior_mass"] <= 1.0
    ):
        raise ValueError("adaptive oracle summary numerical values are invalid.")


def _validate_native_summary(
    summary: dict[str, Any],
    *,
    case_id: str,
    lower_log_mass: float,
) -> None:
    _require_exact_keys(
        summary,
        set(oracle.NativeLogMassSummary.__dataclass_fields__),
        context="native-log-mass oracle summary",
    )
    _require_hashed_mapping(summary, context="native-log-mass oracle summary")
    if (
        summary.get("schema") != oracle.SCHEMA
        or summary.get("case_id") != case_id
        or summary.get("definitions_sha256") != oracle.definitions_sha256()
        or summary.get("method") != "adaptive_native_two_dimensional_log_masses"
        or summary.get("lower_log_mass") != lower_log_mass
        or summary.get("epsabs") != 2.0e-8
        or summary.get("epsrel") != 2.0e-8
    ):
        raise ValueError("native-log-mass oracle summary semantics differ.")
    values = {
        key: _finite_number(
            summary,
            key,
            context="native-log-mass oracle summary",
        )
        for key in (
            "lower_log_mass",
            "upper_log_mass",
            "log_evidence",
            "posterior_mean_total",
            "posterior_sd_total",
            "scaled_quadrature_error",
            "maximum_inner_scaled_quadrature_error",
        )
    }
    if (
        values["lower_log_mass"] >= values["upper_log_mass"]
        or values["posterior_mean_total"] <= 0.0
        or values["posterior_sd_total"] <= 0.0
        or values["scaled_quadrature_error"] < 0.0
        or values["maximum_inner_scaled_quadrature_error"] < 0.0
    ):
        raise ValueError("native-log-mass oracle summary values are invalid.")


def _primary_diagnostics(
    reference: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, float]:
    reference_sd = _finite_number(
        reference,
        "posterior_sd_total",
        context="primary oracle reference",
    )
    if reference_sd <= 0.0:
        raise ValueError("primary oracle reference SD is not positive.")
    return {
        "last_two_log_evidence_delta_nat": abs(
            _finite_number(reference, "log_evidence", context="primary oracle")
            - _finite_number(previous, "log_evidence", context="primary oracle")
        ),
        "last_two_posterior_mean_delta_reference_sd": abs(
            _finite_number(
                reference,
                "posterior_mean_total",
                context="primary oracle",
            )
            - _finite_number(
                previous,
                "posterior_mean_total",
                context="primary oracle",
            )
        )
        / reference_sd,
        "last_two_posterior_sd_relative_delta": abs(
            reference_sd
            - _finite_number(
                previous,
                "posterior_sd_total",
                context="primary oracle",
            )
        )
        / reference_sd,
        "last_two_posterior_endpoint_delta_reference_sd": max(
            abs(
                _finite_number(reference, key, context="primary oracle")
                - _finite_number(previous, key, context="primary oracle")
            )
            for key in (
                "posterior_lower_0_025",
                "posterior_median",
                "posterior_upper_0_975",
            )
        )
        / reference_sd,
    }


def _validate_metric_grid_preflight(
    preflight: dict[str, Any],
    *,
    case_id: str,
    reference: dict[str, Any],
) -> None:
    _require_exact_keys(
        preflight,
        {
            "case_id",
            "construction",
            "posterior_quantile_rule",
            "counts",
            "rows",
            "last_two",
            "final_errors_from_adaptive_reference",
            "checks",
            "pass",
            "sha256",
        },
        context="metric-grid preflight",
    )
    _require_hashed_mapping(preflight, context="metric-grid preflight")
    if (
        preflight.get("case_id") != case_id
        or preflight.get("construction") != "equal-prior-probability midpoint bins"
        or preflight.get("posterior_quantile_rule")
        != "within-bin interpolation under piecewise-constant likelihood"
        or preflight.get("counts") != list(METRIC_GRID_COUNTS)
    ):
        raise ValueError("metric-grid preflight semantics differ.")
    rows = preflight.get("rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("metric-grid preflight rows are malformed.")
    posterior_keys = {
        "log_evidence",
        "posterior_mean_total",
        "posterior_sd_total",
        "posterior_lower_0_025",
        "posterior_median",
        "posterior_upper_0_975",
    }
    for row, count in zip(rows, METRIC_GRID_COUNTS, strict=True):
        if not isinstance(row, dict):
            raise ValueError("metric-grid row is not a mapping.")
        _require_exact_keys(
            row,
            {
                "count",
                "posterior",
                "total_grid_sha256",
                "exact_log_likelihood_sha256",
                "finite",
            },
            context="metric-grid row",
        )
        posterior = row.get("posterior")
        if (
            row.get("count") != count
            or row.get("finite") is not True
            or not isinstance(posterior, dict)
            or set(posterior) != posterior_keys
            or any(
                not isinstance(row.get(key), str)
                or len(str(row.get(key))) != 64
                or any(character not in "0123456789abcdef" for character in str(row.get(key)))
                for key in (
                    "total_grid_sha256",
                    "exact_log_likelihood_sha256",
                )
            )
        ):
            raise ValueError("metric-grid row semantics differ.")
        posterior_values = {
            key: _finite_number(
                posterior,
                key,
                context="metric-grid posterior",
            )
            for key in posterior_keys
        }
        if posterior_values["posterior_sd_total"] <= 0.0 or not (
            posterior_values["posterior_lower_0_025"]
            <= posterior_values["posterior_median"]
            <= posterior_values["posterior_upper_0_975"]
        ):
            raise ValueError("metric-grid posterior values are invalid.")
    previous = rows[-2]["posterior"]
    final = rows[-1]["posterior"]
    reference_sd = _finite_number(
        reference,
        "posterior_sd_total",
        context="metric-grid reference",
    )
    last_two = {
        "log_evidence_delta_nat": abs(float(final["log_evidence"]) - float(previous["log_evidence"])),
        "posterior_mean_delta_reference_sd": abs(
            float(final["posterior_mean_total"]) - float(previous["posterior_mean_total"])
        )
        / reference_sd,
        "posterior_sd_relative_delta": abs(
            float(final["posterior_sd_total"]) - float(previous["posterior_sd_total"])
        )
        / reference_sd,
        "posterior_quantile_delta_reference_sd": max(
            abs(float(final[key]) - float(previous[key]))
            for key in (
                "posterior_lower_0_025",
                "posterior_median",
                "posterior_upper_0_975",
            )
        )
        / reference_sd,
    }
    adaptive = {
        "log_evidence_error_nat": abs(float(final["log_evidence"]) - float(reference["log_evidence"])),
        "posterior_mean_error_reference_sd": abs(
            float(final["posterior_mean_total"]) - float(reference["posterior_mean_total"])
        )
        / reference_sd,
        "posterior_sd_relative_error": abs(float(final["posterior_sd_total"]) - reference_sd) / reference_sd,
        "posterior_quantile_error_reference_sd": max(
            abs(float(final[key]) - float(reference[key]))
            for key in (
                "posterior_lower_0_025",
                "posterior_median",
                "posterior_upper_0_975",
            )
        )
        / reference_sd,
    }
    expected_checks = {
        "finite": True,
        "last_two_evidence": (last_two["log_evidence_delta_nat"] <= METRIC_GRID_EVIDENCE_TOLERANCE_NAT),
        "last_two_mean": (
            last_two["posterior_mean_delta_reference_sd"] <= METRIC_GRID_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "last_two_sd": (last_two["posterior_sd_relative_delta"] <= METRIC_GRID_SD_RELATIVE_TOLERANCE),
        "last_two_quantiles": (
            last_two["posterior_quantile_delta_reference_sd"] <= METRIC_GRID_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "adaptive_evidence": (adaptive["log_evidence_error_nat"] <= METRIC_GRID_EVIDENCE_TOLERANCE_NAT),
        "adaptive_mean": (
            adaptive["posterior_mean_error_reference_sd"] <= METRIC_GRID_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "adaptive_sd": (adaptive["posterior_sd_relative_error"] <= METRIC_GRID_SD_RELATIVE_TOLERANCE),
        "adaptive_quantiles": (
            adaptive["posterior_quantile_error_reference_sd"] <= METRIC_GRID_LOCATION_TOLERANCE_REFERENCE_SD
        ),
    }
    if (
        preflight.get("last_two") != last_two
        or preflight.get("final_errors_from_adaptive_reference") != adaptive
    ):
        raise ValueError("metric-grid preflight diagnostics do not replay.")
    _require_checks(
        preflight,
        expected_checks,
        context="metric-grid preflight",
    )


def _validate_gradient_preflight(
    preflight: dict[str, Any],
    *,
    case_id: str,
    reference: dict[str, Any],
    previous_fraction_order: int,
) -> None:
    _require_exact_keys(
        preflight,
        {
            "case_id",
            "state",
            "total_mass",
            "previous_fraction_order",
            "final_fraction_order",
            "final_order_step_ladder",
            "previous_order_final_step_gradient",
            "reference_gradient",
            "diagnostics",
            "tolerance",
            "checks",
            "pass",
            "sha256",
        },
        context="gradient preflight",
    )
    _require_hashed_mapping(preflight, context="gradient preflight")
    ladder = preflight.get("final_order_step_ladder")
    if not isinstance(ladder, list) or len(ladder) != len(GRADIENT_LOG_TOTAL_STEPS):
        raise ValueError("gradient preflight step ladder is malformed.")
    gradients: list[float] = []
    for row, step in zip(ladder, GRADIENT_LOG_TOTAL_STEPS, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != {"log_total_step", "gradient"}
            or row.get("log_total_step") != step
        ):
            raise ValueError("gradient preflight step semantics differ.")
        gradients.append(_finite_number(row, "gradient", context="gradient preflight"))
    previous_gradient = _finite_number(
        preflight,
        "previous_order_final_step_gradient",
        context="gradient preflight",
    )
    final_gradient = gradients[-1]
    diagnostics = {
        "finite_difference_refinement_scaled_error": (
            abs(final_gradient - gradients[-2]) / (1.0 + abs(final_gradient))
        ),
        "allocation_order_refinement_scaled_error": (
            abs(final_gradient - previous_gradient) / (1.0 + abs(final_gradient))
        ),
    }
    expected_checks = {
        "finite": True,
        "finite_difference_refined": (
            diagnostics["finite_difference_refinement_scaled_error"] <= GRADIENT_NUMERICAL_TOLERANCE
        ),
        "allocation_order_refined": (
            diagnostics["allocation_order_refinement_scaled_error"] <= GRADIENT_NUMERICAL_TOLERANCE
        ),
    }
    if (
        preflight.get("case_id") != case_id
        or preflight.get("state") != "adaptive exact posterior mode total"
        or preflight.get("total_mass") != reference.get("posterior_mode_total")
        or preflight.get("previous_fraction_order") != previous_fraction_order
        or preflight.get("final_fraction_order") != reference.get("fraction_order")
        or preflight.get("reference_gradient") != final_gradient
        or preflight.get("diagnostics") != diagnostics
        or preflight.get("tolerance") != GRADIENT_NUMERICAL_TOLERANCE
    ):
        raise ValueError("gradient preflight diagnostics do not replay.")
    _require_checks(
        preflight,
        expected_checks,
        context="gradient preflight",
    )


def _validate_boundary_certificate(
    certificate: dict[str, Any],
    *,
    selected_case: dict[str, Any],
) -> dict[str, bool]:
    _require_exact_keys(
        certificate,
        {
            "schema",
            "case_id",
            "definitions_sha256",
            "primary_order_ladder",
            "independent_tail_ladder",
            "diagnostics",
            "checks",
            "pass",
            "sha256",
        },
        context="boundary independent certificate",
    )
    _require_hashed_mapping(
        certificate,
        context="boundary independent certificate",
    )
    primary = certificate.get("primary_order_ladder")
    independent = certificate.get("independent_tail_ladder")
    if (
        certificate.get("schema") != oracle.SCHEMA
        or certificate.get("case_id") != oracle.BOUNDARY_CASE_ID
        or certificate.get("definitions_sha256") != oracle.definitions_sha256()
        or not isinstance(primary, list)
        or not isinstance(independent, list)
        or len(primary) != 3
        or len(independent) != 3
    ):
        raise ValueError("boundary independent certificate semantics differ.")
    for summary, order in zip(primary, (16, 32, 64), strict=True):
        if not isinstance(summary, dict):
            raise ValueError("boundary primary summary is malformed.")
        _validate_adaptive_summary(
            summary,
            case_id=oracle.BOUNDARY_CASE_ID,
            fraction_order=order,
            chart="row-first",
        )
    for summary, lower in zip(
        independent,
        (-40.0, -80.0, -120.0),
        strict=True,
    ):
        if not isinstance(summary, dict):
            raise ValueError("boundary native summary is malformed.")
        _validate_native_summary(
            summary,
            case_id=oracle.BOUNDARY_CASE_ID,
            lower_log_mass=lower,
        )
    if primary != selected_case.get("order_ladder"):
        raise ValueError("boundary primary ladder differs from selected case.")
    reference = primary[-1]
    previous = primary[-2]
    independent_reference = independent[-1]
    independent_previous = independent[-2]
    reference_sd = float(reference["posterior_sd_total"])
    diagnostics = {
        "primary_log_evidence_delta_nat": abs(
            float(reference["log_evidence"]) - float(previous["log_evidence"])
        ),
        "independent_log_evidence_delta_nat": abs(
            float(reference["log_evidence"]) - float(independent_reference["log_evidence"])
        ),
        "independent_posterior_mean_delta_reference_sd": abs(
            float(reference["posterior_mean_total"]) - float(independent_reference["posterior_mean_total"])
        )
        / reference_sd,
        "independent_posterior_sd_relative_delta": abs(
            float(reference["posterior_sd_total"]) - float(independent_reference["posterior_sd_total"])
        )
        / reference_sd,
        "primary_posterior_mean_delta_reference_sd": abs(
            float(reference["posterior_mean_total"]) - float(previous["posterior_mean_total"])
        )
        / reference_sd,
        "primary_posterior_sd_relative_delta": abs(
            float(reference["posterior_sd_total"]) - float(previous["posterior_sd_total"])
        )
        / reference_sd,
        "primary_posterior_endpoint_delta_reference_sd": max(
            abs(float(reference[key]) - float(previous[key]))
            for key in (
                "posterior_lower_0_025",
                "posterior_median",
                "posterior_upper_0_975",
            )
        )
        / reference_sd,
        "independent_tail_log_evidence_delta_nat": abs(
            float(independent_reference["log_evidence"]) - float(independent_previous["log_evidence"])
        ),
        "independent_tail_posterior_mean_delta_reference_sd": abs(
            float(independent_reference["posterior_mean_total"])
            - float(independent_previous["posterior_mean_total"])
        )
        / reference_sd,
        "independent_tail_posterior_sd_relative_delta": abs(
            float(independent_reference["posterior_sd_total"])
            - float(independent_previous["posterior_sd_total"])
        )
        / reference_sd,
    }
    expected_checks = {
        "primary_log_evidence_converged": (
            diagnostics["primary_log_evidence_delta_nat"] <= oracle.PRIMARY_LOG_EVIDENCE_TOLERANCE_NAT
        ),
        "independent_log_evidence_agrees": (
            diagnostics["independent_log_evidence_delta_nat"] <= oracle.INDEPENDENT_LOG_EVIDENCE_TOLERANCE_NAT
        ),
        "independent_posterior_mean_agrees": (
            diagnostics["independent_posterior_mean_delta_reference_sd"]
            <= oracle.POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "independent_posterior_sd_agrees": (
            diagnostics["independent_posterior_sd_relative_delta"] <= oracle.POSTERIOR_SD_RELATIVE_TOLERANCE
        ),
        "primary_posterior_mean_converged": (
            diagnostics["primary_posterior_mean_delta_reference_sd"]
            <= oracle.POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "primary_posterior_sd_converged": (
            diagnostics["primary_posterior_sd_relative_delta"] <= oracle.POSTERIOR_SD_RELATIVE_TOLERANCE
        ),
        "primary_posterior_endpoints_converged": (
            diagnostics["primary_posterior_endpoint_delta_reference_sd"]
            <= oracle.POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "independent_tail_log_evidence_converged": (
            diagnostics["independent_tail_log_evidence_delta_nat"]
            <= oracle.INDEPENDENT_LOG_EVIDENCE_TOLERANCE_NAT
        ),
        "independent_tail_posterior_mean_converged": (
            diagnostics["independent_tail_posterior_mean_delta_reference_sd"]
            <= oracle.POSTERIOR_LOCATION_TOLERANCE_REFERENCE_SD
        ),
        "independent_tail_posterior_sd_converged": (
            diagnostics["independent_tail_posterior_sd_relative_delta"]
            <= oracle.POSTERIOR_SD_RELATIVE_TOLERANCE
        ),
        "primary_scaled_quadrature_error_small": (float(reference["scaled_quadrature_error"]) <= 1.0e-6),
        "independent_outer_scaled_quadrature_error_small": (
            float(independent_reference["scaled_quadrature_error"]) <= 1.0e-6
        ),
        "independent_inner_scaled_quadrature_error_small": (
            float(independent_reference["maximum_inner_scaled_quadrature_error"]) <= 1.0e-6
        ),
        "support_retains_prior_mass": (float(reference["represented_prior_mass"]) >= 1.0 - 1.0e-12),
        "support_retains_posterior_mass": (float(reference["represented_posterior_mass"]) >= 1.0 - 1.0e-6),
        "posterior_mode_included": reference["mode_included"] is True,
    }
    if certificate.get("diagnostics") != diagnostics:
        raise ValueError("boundary independent diagnostics do not replay.")
    _require_checks(
        certificate,
        expected_checks,
        context="boundary independent certificate",
    )
    return expected_checks


def _validate_four_cell_certificate(
    certificate: dict[str, Any],
    *,
    case_id: str,
    reference: dict[str, Any],
) -> dict[str, bool]:
    includes_fixed = case_id == "boundary_heavy__four_cell__root"
    expected_keys = {
        "case_id",
        "method",
        "fraction_order",
        "row_reference_sha256",
        "column_summary",
        "diagnostics",
        "checks",
        "pass",
        "sha256",
    }
    if includes_fixed:
        expected_keys.add("fixed_log_total_column_chart")
    _require_exact_keys(
        certificate,
        expected_keys,
        context="four-cell independent certificate",
    )
    _require_hashed_mapping(
        certificate,
        context="four-cell independent certificate",
    )
    column = certificate.get("column_summary")
    if not isinstance(column, dict):
        raise ValueError("four-cell column summary is malformed.")
    order = int(reference["fraction_order"])
    _validate_adaptive_summary(
        column,
        case_id=case_id,
        fraction_order=order,
        chart="column-first",
    )
    if (
        certificate.get("case_id") != case_id
        or certificate.get("method") != "adaptive row-first versus column-first Dirichlet charts"
        or certificate.get("fraction_order") != order
        or certificate.get("row_reference_sha256") != reference.get("sha256")
    ):
        raise ValueError("four-cell independent certificate semantics differ.")
    reference_sd = float(reference["posterior_sd_total"])
    diagnostics = {
        "absolute_log_evidence_delta_nat": abs(
            float(reference["log_evidence"]) - float(column["log_evidence"])
        ),
        "posterior_mean_delta_reference_sd": abs(
            float(reference["posterior_mean_total"]) - float(column["posterior_mean_total"])
        )
        / reference_sd,
        "posterior_sd_relative_delta": abs(
            float(reference["posterior_sd_total"]) - float(column["posterior_sd_total"])
        )
        / reference_sd,
        "posterior_endpoint_delta_reference_sd": max(
            abs(float(reference[key]) - float(column[key]))
            for key in (
                "posterior_lower_0_025",
                "posterior_median",
                "posterior_upper_0_975",
            )
        )
        / reference_sd,
    }
    expected_checks = {
        "log_evidence": (
            diagnostics["absolute_log_evidence_delta_nat"] <= LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT
        ),
        "posterior_mean": (diagnostics["posterior_mean_delta_reference_sd"] <= 0.005),
        "posterior_sd": (diagnostics["posterior_sd_relative_delta"] <= 0.002),
        "posterior_endpoints": (diagnostics["posterior_endpoint_delta_reference_sd"] <= 0.005),
        "column_normalizer_quadrature_error": (float(column["scaled_quadrature_error"]) <= 1.0e-6),
        "column_moment_quadrature_error": (
            float(column["maximum_relative_moment_quadrature_error"]) <= 1.0e-6
        ),
        "column_cdf_quadrature_error": (float(column["maximum_scaled_cdf_quadrature_error"]) <= 1.0e-6),
        "column_represented_prior_mass": (float(column["represented_prior_mass"]) >= 1.0 - 1.0e-12),
        "column_represented_posterior_mass": (float(column["represented_posterior_mass"]) >= 1.0 - 1.0e-6),
        "column_posterior_mode_included": column["mode_included"] is True,
    }
    if certificate.get("diagnostics") != diagnostics:
        raise ValueError("four-cell chart diagnostics do not replay.")
    if includes_fixed:
        fixed = certificate.get("fixed_log_total_column_chart")
        if not isinstance(fixed, dict):
            raise ValueError("fixed-log-total certificate is malformed.")
        _require_exact_keys(
            fixed,
            {
                "method",
                "chart",
                "fraction_order",
                "prior_tail_probability",
                "total_order_ladder",
                "log_evidence_ladder",
                "diagnostics",
                "checks",
                "pass",
            },
            context="fixed-log-total certificate",
        )
        values = fixed.get("log_evidence_ladder")
        if (
            fixed.get("method") != "fixed Gauss-Legendre in log(total)"
            or fixed.get("chart") != "column-first"
            or fixed.get("fraction_order") != order
            or fixed.get("prior_tail_probability") != 1.0e-15
            or fixed.get("total_order_ladder") != [512, 1024, 2048]
            or not isinstance(values, list)
            or len(values) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            )
        ):
            raise ValueError("fixed-log-total certificate semantics differ.")
        fixed_diagnostics = {
            "last_two_log_evidence_delta_nat": abs(float(values[-1]) - float(values[-2])),
            "adaptive_primary_delta_nat": abs(float(values[-1]) - float(reference["log_evidence"])),
        }
        fixed_checks = {
            "fixed_log_total_converged": (
                fixed_diagnostics["last_two_log_evidence_delta_nat"] <= LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT
            ),
            "fixed_log_total_agrees_with_adaptive": (
                fixed_diagnostics["adaptive_primary_delta_nat"] <= LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT
            ),
        }
        if fixed.get("diagnostics") != fixed_diagnostics:
            raise ValueError("fixed-log-total diagnostics do not replay.")
        _require_checks(
            fixed,
            fixed_checks,
            context="fixed-log-total certificate",
        )
        expected_checks = {**expected_checks, **fixed_checks}
    _require_checks(
        certificate,
        expected_checks,
        context="four-cell independent certificate",
    )
    return expected_checks


def _validate_skew_native_certificate(
    certificate: dict[str, Any],
    *,
    reference: dict[str, Any],
) -> dict[str, bool]:
    _require_exact_keys(
        certificate,
        {
            "case_id",
            "method",
            "lower_log_mass_ladder",
            "summaries",
            "diagnostics",
            "checks",
            "pass",
            "sha256",
        },
        context="skew native-log-mass certificate",
    )
    _require_hashed_mapping(
        certificate,
        context="skew native-log-mass certificate",
    )
    summaries = certificate.get("summaries")
    if (
        certificate.get("case_id") != "skewed__two_cell__root"
        or certificate.get("method") != "adaptive native two-dimensional log masses"
        or certificate.get("lower_log_mass_ladder") != [-80.0, -120.0]
        or not isinstance(summaries, list)
        or len(summaries) != 2
    ):
        raise ValueError("skew native-log-mass certificate semantics differ.")
    for summary, lower in zip(summaries, (-80.0, -120.0), strict=True):
        if not isinstance(summary, dict):
            raise ValueError("skew native-log-mass summary is malformed.")
        _validate_native_summary(
            summary,
            case_id="skewed__two_cell__root",
            lower_log_mass=lower,
        )
    final = summaries[-1]
    previous = summaries[-2]
    reference_sd = float(reference["posterior_sd_total"])
    diagnostics = {
        "tail_log_evidence_delta_nat": abs(float(final["log_evidence"]) - float(previous["log_evidence"])),
        "primary_log_evidence_delta_nat": abs(
            float(reference["log_evidence"]) - float(final["log_evidence"])
        ),
        "primary_posterior_mean_delta_reference_sd": abs(
            float(reference["posterior_mean_total"]) - float(final["posterior_mean_total"])
        )
        / reference_sd,
        "primary_posterior_sd_relative_delta": abs(
            float(reference["posterior_sd_total"]) - float(final["posterior_sd_total"])
        )
        / reference_sd,
    }
    expected_checks = {
        "tail_evidence": (
            diagnostics["tail_log_evidence_delta_nat"] <= LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT
        ),
        "primary_evidence": (
            diagnostics["primary_log_evidence_delta_nat"] <= LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT
        ),
        "primary_mean": (diagnostics["primary_posterior_mean_delta_reference_sd"] <= 0.005),
        "primary_sd": (diagnostics["primary_posterior_sd_relative_delta"] <= 0.002),
        "native_outer_quadrature_errors": all(
            float(summary["scaled_quadrature_error"]) <= 1.0e-6 for summary in summaries
        ),
        "native_inner_quadrature_errors": all(
            float(summary["maximum_inner_scaled_quadrature_error"]) <= 1.0e-6 for summary in summaries
        ),
    }
    if certificate.get("diagnostics") != diagnostics:
        raise ValueError("skew native-log-mass diagnostics do not replay.")
    _require_checks(
        certificate,
        expected_checks,
        context="skew native-log-mass certificate",
    )
    return expected_checks


def validate_bundle_semantics(bundle: dict[str, Any]) -> None:
    """Recompute every promotion-oracle numerical gate from nested values."""
    _require_exact_keys(
        bundle,
        {
            "schema",
            "source_git_revision",
            "tiny_root_definitions_sha256",
            "selected_cases",
            "boundary_independent_certificate",
            "independent_certificates",
            "checks",
            "pass",
            "runtime_seconds",
            "sha256",
        },
        context="promotion oracle bundle",
    )
    _require_hashed_mapping(bundle, context="promotion oracle bundle")
    runtime_seconds = _finite_number(
        bundle,
        "runtime_seconds",
        context="promotion oracle bundle",
    )
    selected = bundle.get("selected_cases")
    certificates = bundle.get("independent_certificates")
    if (
        bundle.get("schema") != SCHEMA
        or bundle.get("tiny_root_definitions_sha256") != oracle.definitions_sha256()
        or runtime_seconds < 0.0
        or not isinstance(selected, dict)
        or set(selected) != set(PROMOTION_CASES)
        or not isinstance(certificates, dict)
        or set(certificates)
        != {
            "near_gaussian__four_cell__root",
            "skewed__two_cell__root",
            "skewed__four_cell__root",
            "boundary_heavy__four_cell__root",
        }
    ):
        raise ValueError("promotion oracle bundle catalogue semantics differ.")
    expected_top_checks: dict[str, bool] = {}
    for case_id in PROMOTION_CASES:
        case_payload = selected[case_id]
        if not isinstance(case_payload, dict):
            raise ValueError("promotion selected case is malformed.")
        is_boundary_two = case_id == oracle.BOUNDARY_CASE_ID
        expected_case_keys = {
            "order_ladder",
            "reference",
            "last_two_log_evidence_delta_nat",
            "metric_grid_preflight",
            "gradient_preflight",
            "checks",
            "pass",
        }
        if not is_boundary_two:
            expected_case_keys.update(
                {
                    "last_two_posterior_mean_delta_reference_sd",
                    "last_two_posterior_sd_relative_delta",
                    "last_two_posterior_endpoint_delta_reference_sd",
                }
            )
        _require_exact_keys(
            case_payload,
            expected_case_keys,
            context=f"selected oracle case {case_id}",
        )
        ladder = case_payload.get("order_ladder")
        orders = (16, 32, 64) if is_boundary_two else CASE_ORDER_LADDERS[case_id]
        if (
            not isinstance(ladder, list)
            or len(ladder) != len(orders)
            or case_payload.get("reference") != ladder[-1]
        ):
            raise ValueError("promotion primary ladder semantics differ.")
        for summary, order in zip(ladder, orders, strict=True):
            if not isinstance(summary, dict):
                raise ValueError("promotion primary summary is malformed.")
            _validate_adaptive_summary(
                summary,
                case_id=case_id,
                fraction_order=order,
                chart="row-first",
            )
        reference = ladder[-1]
        _validate_metric_grid_preflight(
            case_payload["metric_grid_preflight"],
            case_id=case_id,
            reference=reference,
        )
        _validate_gradient_preflight(
            case_payload["gradient_preflight"],
            case_id=case_id,
            reference=reference,
            previous_fraction_order=orders[-2],
        )
        if not is_boundary_two:
            diagnostics = _primary_diagnostics(reference, ladder[-2])
            expected_case_checks = {
                "log_evidence_converged": (
                    diagnostics["last_two_log_evidence_delta_nat"] <= LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT
                ),
                "posterior_mean_converged": (
                    diagnostics["last_two_posterior_mean_delta_reference_sd"] <= 0.005
                ),
                "posterior_sd_converged": (diagnostics["last_two_posterior_sd_relative_delta"] <= 0.002),
                "posterior_endpoints_converged": (
                    diagnostics["last_two_posterior_endpoint_delta_reference_sd"] <= 0.005
                ),
                "represented_prior_mass": (float(reference["represented_prior_mass"]) >= 1.0 - 1.0e-12),
                "represented_posterior_mass": (
                    float(reference["represented_posterior_mass"]) >= 1.0 - 1.0e-6
                ),
                "posterior_mode_included": reference["mode_included"] is True,
                "scaled_quadrature_error_small": (float(reference["scaled_quadrature_error"]) <= 1.0e-6),
                "moment_quadrature_error_small": (
                    float(reference["maximum_relative_moment_quadrature_error"]) <= 1.0e-6
                ),
                "cdf_quadrature_error_small": (
                    float(reference["maximum_scaled_cdf_quadrature_error"]) <= 1.0e-6
                ),
                "metric_grid_preflight": (case_payload["metric_grid_preflight"]["pass"] is True),
                "gradient_preflight": (case_payload["gradient_preflight"]["pass"] is True),
            }
            for key, value in diagnostics.items():
                if case_payload.get(key) != value:
                    raise ValueError("promotion primary diagnostics do not replay.")
            _require_checks(
                case_payload,
                expected_case_checks,
                context=f"selected oracle case {case_id}",
            )
            expected_top_checks[f"{case_id}__converged"] = all(expected_case_checks.values())
    boundary = bundle.get("boundary_independent_certificate")
    if not isinstance(boundary, dict):
        raise ValueError("boundary independent certificate is malformed.")
    boundary_checks = _validate_boundary_certificate(
        boundary,
        selected_case=selected[oracle.BOUNDARY_CASE_ID],
    )
    boundary_case = selected[oracle.BOUNDARY_CASE_ID]
    if (
        boundary_case.get("last_two_log_evidence_delta_nat")
        != boundary["diagnostics"]["primary_log_evidence_delta_nat"]
    ):
        raise ValueError("boundary selected-case diagnostics do not replay.")
    boundary_case_checks = {
        "boundary_independent_certificate": all(boundary_checks.values()),
        "metric_grid_preflight": (boundary_case["metric_grid_preflight"]["pass"] is True),
        "gradient_preflight": (boundary_case["gradient_preflight"]["pass"] is True),
    }
    _require_checks(
        boundary_case,
        boundary_case_checks,
        context="boundary selected oracle case",
    )
    expected_top_checks["boundary_independent_certificate"] = all(boundary_checks.values())
    expected_top_checks["boundary_heavy__two_cell__root__metric_grid"] = (
        boundary_case["metric_grid_preflight"]["pass"] is True
    )
    expected_top_checks["boundary_heavy__two_cell__root__gradient"] = (
        boundary_case["gradient_preflight"]["pass"] is True
    )
    for case_id in FOUR_CELL_CHART_CASES:
        certificate = certificates[case_id]
        if not isinstance(certificate, dict):
            raise ValueError("four-cell independent certificate is malformed.")
        chart_checks = _validate_four_cell_certificate(
            certificate,
            case_id=case_id,
            reference=selected[case_id]["reference"],
        )
        expected_top_checks[f"{case_id}__independent_chart"] = all(
            chart_checks[key]
            for key in (
                "log_evidence",
                "posterior_mean",
                "posterior_sd",
                "posterior_endpoints",
                "column_normalizer_quadrature_error",
                "column_moment_quadrature_error",
                "column_cdf_quadrature_error",
                "column_represented_prior_mass",
                "column_represented_posterior_mass",
                "column_posterior_mode_included",
            )
        )
        if case_id == "boundary_heavy__four_cell__root":
            expected_top_checks["boundary_heavy__four_cell__root__fixed_log_total"] = (
                chart_checks["fixed_log_total_converged"]
                and chart_checks["fixed_log_total_agrees_with_adaptive"]
            )
    skew_certificate = certificates["skewed__two_cell__root"]
    if not isinstance(skew_certificate, dict):
        raise ValueError("skew native certificate is malformed.")
    skew_checks = _validate_skew_native_certificate(
        skew_certificate,
        reference=selected["skewed__two_cell__root"]["reference"],
    )
    expected_top_checks["skewed__two_cell__root__independent_native"] = all(skew_checks.values())
    _require_checks(
        bundle,
        expected_top_checks,
        context="promotion oracle top-level",
    )


def build_bundle(source_git_revision: str) -> dict[str, Any]:
    """Build all six references and the independent two-cell boundary certificate."""
    if PROMOTION_CASES != oracle.CASE_IDS:
        raise RuntimeError("frozen promotion cases differ from the tiny catalogue.")
    started = time.perf_counter()
    selected: dict[str, Any] = {}
    independent_certificates: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for case_id, orders in CASE_ORDER_LADDERS.items():
        ladder = [
            oracle.adaptive_log_total_summary(
                case_id,
                fraction_order=order,
            )
            for order in orders
        ]
        reference = ladder[-1]
        previous = ladder[-2]
        delta = abs(reference.log_evidence - previous.log_evidence)
        location_delta = (
            abs(reference.posterior_mean_total - previous.posterior_mean_total) / reference.posterior_sd_total
        )
        sd_delta = (
            abs(reference.posterior_sd_total - previous.posterior_sd_total) / reference.posterior_sd_total
        )
        endpoint_delta = (
            max(
                abs(reference.posterior_lower_0_025 - previous.posterior_lower_0_025),
                abs(reference.posterior_median - previous.posterior_median),
                abs(reference.posterior_upper_0_975 - previous.posterior_upper_0_975),
            )
            / reference.posterior_sd_total
        )
        case_checks = {
            "log_evidence_converged": (delta <= LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT),
            "posterior_mean_converged": location_delta <= 0.005,
            "posterior_sd_converged": sd_delta <= 0.002,
            "posterior_endpoints_converged": endpoint_delta <= 0.005,
            "represented_prior_mass": (reference.represented_prior_mass >= 1.0 - 1.0e-12),
            "represented_posterior_mass": (reference.represented_posterior_mass >= 1.0 - 1.0e-6),
            "posterior_mode_included": reference.mode_included,
            "scaled_quadrature_error_small": (reference.scaled_quadrature_error <= 1.0e-6),
            "moment_quadrature_error_small": (reference.maximum_relative_moment_quadrature_error <= 1.0e-6),
            "cdf_quadrature_error_small": (reference.maximum_scaled_cdf_quadrature_error <= 1.0e-6),
        }
        case_pass = all(case_checks.values())
        grid_preflight = _metric_grid_preflight(
            case_id,
            reference.payload(),
        )
        gradient_preflight = _gradient_preflight(
            case_id,
            reference.payload(),
            previous_fraction_order=orders[-2],
        )
        case_checks["metric_grid_preflight"] = bool(grid_preflight["pass"])
        case_checks["gradient_preflight"] = bool(gradient_preflight["pass"])
        case_pass = all(case_checks.values())
        selected[case_id] = {
            "order_ladder": [summary.payload() for summary in ladder],
            "reference": reference.payload(),
            "last_two_log_evidence_delta_nat": delta,
            "last_two_posterior_mean_delta_reference_sd": (location_delta),
            "last_two_posterior_sd_relative_delta": sd_delta,
            "last_two_posterior_endpoint_delta_reference_sd": (endpoint_delta),
            "metric_grid_preflight": grid_preflight,
            "gradient_preflight": gradient_preflight,
            "checks": case_checks,
            "pass": case_pass,
        }
        checks[f"{case_id}__converged"] = case_pass
        if case_id in FOUR_CELL_CHART_CASES:
            column = oracle.adaptive_log_total_summary(
                case_id,
                fraction_order=orders[-1],
                root_chart="column-first",
            )
            chart_diagnostics = {
                "absolute_log_evidence_delta_nat": abs(reference.log_evidence - column.log_evidence),
                "posterior_mean_delta_reference_sd": abs(
                    reference.posterior_mean_total - column.posterior_mean_total
                )
                / reference.posterior_sd_total,
                "posterior_sd_relative_delta": abs(reference.posterior_sd_total - column.posterior_sd_total)
                / reference.posterior_sd_total,
                "posterior_endpoint_delta_reference_sd": max(
                    abs(reference.posterior_lower_0_025 - column.posterior_lower_0_025),
                    abs(reference.posterior_median - column.posterior_median),
                    abs(reference.posterior_upper_0_975 - column.posterior_upper_0_975),
                )
                / reference.posterior_sd_total,
            }
            chart_checks = {
                "log_evidence": (
                    chart_diagnostics["absolute_log_evidence_delta_nat"]
                    <= LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT
                ),
                "posterior_mean": (chart_diagnostics["posterior_mean_delta_reference_sd"] <= 0.005),
                "posterior_sd": (chart_diagnostics["posterior_sd_relative_delta"] <= 0.002),
                "posterior_endpoints": (chart_diagnostics["posterior_endpoint_delta_reference_sd"] <= 0.005),
                "column_normalizer_quadrature_error": (column.scaled_quadrature_error <= 1.0e-6),
                "column_moment_quadrature_error": (column.maximum_relative_moment_quadrature_error <= 1.0e-6),
                "column_cdf_quadrature_error": (column.maximum_scaled_cdf_quadrature_error <= 1.0e-6),
                "column_represented_prior_mass": (column.represented_prior_mass >= 1.0 - 1.0e-12),
                "column_represented_posterior_mass": (column.represented_posterior_mass >= 1.0 - 1.0e-6),
                "column_posterior_mode_included": column.mode_included,
            }
            without_certificate_sha: dict[str, Any] = {
                "case_id": case_id,
                "method": "adaptive row-first versus column-first Dirichlet charts",
                "fraction_order": orders[-1],
                "row_reference_sha256": reference.sha256,
                "column_summary": column.payload(),
                "diagnostics": chart_diagnostics,
                "checks": chart_checks,
                "pass": all(chart_checks.values()),
            }
            independent_certificates[case_id] = {
                **without_certificate_sha,
                "sha256": _sha256_json(without_certificate_sha),
            }
            checks[f"{case_id}__independent_chart"] = all(chart_checks.values())
    skew_reference = selected["skewed__two_cell__root"]["reference"]
    skew_native = [
        oracle.native_log_mass_summary(
            "skewed__two_cell__root",
            lower_log_mass=lower,
        )
        for lower in (-80.0, -120.0)
    ]
    skew_native_final = skew_native[-1]
    skew_native_diagnostics = {
        "tail_log_evidence_delta_nat": abs(skew_native[-1].log_evidence - skew_native[-2].log_evidence),
        "primary_log_evidence_delta_nat": abs(
            float(skew_reference["log_evidence"]) - skew_native_final.log_evidence
        ),
        "primary_posterior_mean_delta_reference_sd": abs(
            float(skew_reference["posterior_mean_total"]) - skew_native_final.posterior_mean_total
        )
        / float(skew_reference["posterior_sd_total"]),
        "primary_posterior_sd_relative_delta": abs(
            float(skew_reference["posterior_sd_total"]) - skew_native_final.posterior_sd_total
        )
        / float(skew_reference["posterior_sd_total"]),
    }
    skew_native_checks = {
        "tail_evidence": (
            skew_native_diagnostics["tail_log_evidence_delta_nat"] <= LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT
        ),
        "primary_evidence": (
            skew_native_diagnostics["primary_log_evidence_delta_nat"]
            <= LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT
        ),
        "primary_mean": (skew_native_diagnostics["primary_posterior_mean_delta_reference_sd"] <= 0.005),
        "primary_sd": (skew_native_diagnostics["primary_posterior_sd_relative_delta"] <= 0.002),
        "native_outer_quadrature_errors": all(
            summary.scaled_quadrature_error <= 1.0e-6 for summary in skew_native
        ),
        "native_inner_quadrature_errors": all(
            summary.maximum_inner_scaled_quadrature_error <= 1.0e-6 for summary in skew_native
        ),
    }
    skew_without_sha: dict[str, Any] = {
        "case_id": "skewed__two_cell__root",
        "method": "adaptive native two-dimensional log masses",
        "lower_log_mass_ladder": [-80.0, -120.0],
        "summaries": [summary.payload() for summary in skew_native],
        "diagnostics": skew_native_diagnostics,
        "checks": skew_native_checks,
        "pass": all(skew_native_checks.values()),
    }
    independent_certificates["skewed__two_cell__root"] = {
        **skew_without_sha,
        "sha256": _sha256_json(skew_without_sha),
    }
    checks["skewed__two_cell__root__independent_native"] = all(skew_native_checks.values())
    boundary = oracle.boundary_oracle_certificate()
    boundary_reference = boundary["primary_order_ladder"][-1]
    boundary_gradient_preflight = _gradient_preflight(
        oracle.BOUNDARY_CASE_ID,
        boundary_reference,
        previous_fraction_order=32,
    )
    selected[oracle.BOUNDARY_CASE_ID] = {
        "order_ladder": boundary["primary_order_ladder"],
        "reference": boundary_reference,
        "last_two_log_evidence_delta_nat": boundary["diagnostics"]["primary_log_evidence_delta_nat"],
        "metric_grid_preflight": _metric_grid_preflight(
            oracle.BOUNDARY_CASE_ID,
            boundary_reference,
        ),
        "gradient_preflight": boundary_gradient_preflight,
        "pass": False,
    }
    selected[oracle.BOUNDARY_CASE_ID]["checks"] = {
        "boundary_independent_certificate": bool(boundary["pass"]),
        "metric_grid_preflight": bool(selected[oracle.BOUNDARY_CASE_ID]["metric_grid_preflight"]["pass"]),
        "gradient_preflight": bool(boundary_gradient_preflight["pass"]),
    }
    selected[oracle.BOUNDARY_CASE_ID]["pass"] = all(selected[oracle.BOUNDARY_CASE_ID]["checks"].values())
    checks["boundary_independent_certificate"] = bool(boundary["pass"])
    checks["boundary_heavy__two_cell__root__metric_grid"] = bool(
        selected[oracle.BOUNDARY_CASE_ID]["metric_grid_preflight"]["pass"]
    )
    checks["boundary_heavy__two_cell__root__gradient"] = bool(boundary_gradient_preflight["pass"])
    boundary_four_reference = selected["boundary_heavy__four_cell__root"]["reference"]
    fixed_orders = (512, 1_024, 2_048)
    fixed_values = [
        _fixed_log_total_evidence(
            "boundary_heavy__four_cell__root",
            fraction_order=int(boundary_four_reference["fraction_order"]),
            total_order=order,
        )
        for order in fixed_orders
    ]
    fixed_diagnostics = {
        "last_two_log_evidence_delta_nat": abs(fixed_values[-1] - fixed_values[-2]),
        "adaptive_primary_delta_nat": abs(fixed_values[-1] - float(boundary_four_reference["log_evidence"])),
    }
    fixed_checks = {
        "fixed_log_total_converged": (
            fixed_diagnostics["last_two_log_evidence_delta_nat"] <= LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT
        ),
        "fixed_log_total_agrees_with_adaptive": (
            fixed_diagnostics["adaptive_primary_delta_nat"] <= LOG_EVIDENCE_CONVERGENCE_TOLERANCE_NAT
        ),
    }
    boundary_four_certificate = independent_certificates["boundary_heavy__four_cell__root"]
    boundary_four_without_sha = {
        key: value for key, value in boundary_four_certificate.items() if key != "sha256"
    }
    boundary_four_without_sha["fixed_log_total_column_chart"] = {
        "method": "fixed Gauss-Legendre in log(total)",
        "chart": "column-first",
        "fraction_order": int(boundary_four_reference["fraction_order"]),
        "prior_tail_probability": 1.0e-15,
        "total_order_ladder": list(fixed_orders),
        "log_evidence_ladder": fixed_values,
        "diagnostics": fixed_diagnostics,
        "checks": fixed_checks,
        "pass": all(fixed_checks.values()),
    }
    boundary_four_without_sha["checks"] = {
        **boundary_four_without_sha["checks"],
        **fixed_checks,
    }
    boundary_four_without_sha["pass"] = all(boundary_four_without_sha["checks"].values())
    independent_certificates["boundary_heavy__four_cell__root"] = {
        **boundary_four_without_sha,
        "sha256": _sha256_json(boundary_four_without_sha),
    }
    checks["boundary_heavy__four_cell__root__fixed_log_total"] = all(fixed_checks.values())
    if set(selected) != set(PROMOTION_CASES):
        raise RuntimeError("corrected oracle bundle does not contain all six cases.")
    selected = {case_id: selected[case_id] for case_id in PROMOTION_CASES}
    without_sha: dict[str, Any] = {
        "schema": SCHEMA,
        "source_git_revision": source_git_revision,
        "tiny_root_definitions_sha256": oracle.definitions_sha256(),
        "selected_cases": selected,
        "boundary_independent_certificate": boundary,
        "independent_certificates": independent_certificates,
        "checks": checks,
        "pass": all(checks.values()),
        "runtime_seconds": time.perf_counter() - started,
    }
    bundle = {
        **without_sha,
        "sha256": _sha256_json(without_sha),
    }
    validate_bundle_semantics(bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    if len(arguments.source_git_revision) != 40 or any(
        character not in "0123456789abcdef" for character in arguments.source_git_revision
    ):
        raise ValueError("source_git_revision must be a full lower-case Git SHA.")
    bundle = build_bundle(arguments.source_git_revision)
    _publish_bundle(
        arguments.output_root,
        bundle,
        arguments.source_git_revision,
    )


if __name__ == "__main__":
    main()
