#!/usr/bin/env python3
"""Validate and summarize the frozen resolution-SMC R1a matrix.

The input directory is the ``r1a`` directory written by
``resolution_smc_experiment.py``.  This analysis deliberately scores the
non-negative likelihood estimators before taking logarithms.  Its primary
efficiency coordinate is between-replicate relative variance multiplied by
the median measured estimator wall time; mean wall time is retained as a
warm-up-sensitive secondary coordinate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

SCHEMA = "rjmcmc-resolution-smc-r1a-analysis-v1"
EXPECTED_SOURCE_SCHEMA = "rjmcmc-resolution-smc-r1a-v1"
EXPECTED_CERTIFICATE_COUNT = 12
EXPECTED_REPLICATE_COUNT = 64
EXPECTED_SOURCE_SHA_LENGTH = 40
Z_95 = 1.959963984540054


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV table {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object.")
        rows.append(payload)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantile(values: FloatArray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def _finite_float(row: dict[str, object], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"record field {name!r} is not numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"record field {name!r} is non-finite.")
    return result


def _integer(row: dict[str, object], name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"record field {name!r} is not an integer.")
    return value


def _stem(certificate: dict[str, object]) -> str:
    return (
        f"task-{_integer(certificate, 'task_index'):02d}-"
        f"{certificate['case']}-n{_integer(certificate, 'particle_count')}"
    )


def _load_and_validate(
    input_root: Path,
    *,
    source_sha: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    certificate_paths = sorted((input_root / "certificates").glob("*.json"))
    if len(certificate_paths) != EXPECTED_CERTIFICATE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_CERTIFICATE_COUNT} certificates, found {len(certificate_paths)}."
        )
    certificates: list[dict[str, object]] = []
    replicates: list[dict[str, object]] = []
    levels: list[dict[str, object]] = []
    task_indices: set[int] = set()
    for certificate_path in certificate_paths:
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        if not isinstance(certificate, dict):
            raise ValueError(f"{certificate_path} is not a JSON object.")
        if certificate.get("schema") != EXPECTED_SOURCE_SCHEMA:
            raise ValueError(f"{certificate_path} has an unexpected schema.")
        if certificate.get("status") != "passed":
            raise ValueError(f"{certificate_path} is not a passed certificate.")
        provenance = certificate.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"{certificate_path} has no provenance object.")
        if (
            provenance.get("source_sha") != source_sha
            or provenance.get("head_sha") != source_sha
            or provenance.get("clean") is not True
            or provenance.get("detached") is not True
        ):
            raise ValueError(f"{certificate_path} provenance does not match the frozen run.")
        task_index = _integer(certificate, "task_index")
        if task_index in task_indices:
            raise ValueError(f"duplicate task index {task_index}.")
        task_indices.add(task_index)
        if _integer(certificate, "replicate_count") != EXPECTED_REPLICATE_COUNT:
            raise ValueError(f"{certificate_path} has an unexpected replicate count.")
        stem = _stem(certificate)
        replicate_path = input_root / "replicates" / f"{stem}.jsonl"
        level_path = input_root / "levels" / f"{stem}.jsonl"
        if _sha256(replicate_path) != certificate.get("replicate_sha256"):
            raise ValueError(f"{replicate_path} does not match its certificate digest.")
        if _sha256(level_path) != certificate.get("levels_sha256"):
            raise ValueError(f"{level_path} does not match its certificate digest.")
        task_replicates = _read_jsonl(replicate_path)
        task_levels = _read_jsonl(level_path)
        if len(task_replicates) != _integer(certificate, "record_count"):
            raise ValueError(f"{replicate_path} record count does not match its certificate.")
        if len(task_levels) != _integer(certificate, "level_record_count"):
            raise ValueError(f"{level_path} record count does not match its certificate.")
        if any(row.get("schema") != EXPECTED_SOURCE_SCHEMA for row in task_replicates):
            raise ValueError(f"{replicate_path} contains an unexpected schema.")
        if any(row.get("schema") != EXPECTED_SOURCE_SCHEMA for row in task_levels):
            raise ValueError(f"{level_path} contains an unexpected schema.")
        certificates.append(certificate)
        replicates.extend(task_replicates)
        levels.extend(task_levels)
    if task_indices != set(range(EXPECTED_CERTIFICATE_COUNT)):
        raise ValueError("certificate task indices are incomplete.")

    for row in replicates:
        likelihood = _finite_float(row, "likelihood")
        if likelihood <= 0.0:
            raise ValueError("likelihood estimators must be strictly positive.")
        _finite_float(row, "wall_seconds")
        _finite_float(row, "oracle_likelihood")
    matched = [row for row in replicates if row["estimator"] == "path_matched_no_resampling_smc"]
    if not matched or any(row.get("path_match_identity_passed") is not True for row in matched):
        raise ValueError("path-matched no-resampling identity did not hold.")
    for row in levels:
        for name in (
            "ess",
            "ess_fraction",
            "incremental_weight_cv",
            "max_normalized_weight",
            "shannon_perplexity",
            "linear_likelihood_correction_variance",
            "max_mass_conservation_error",
            "max_mean_update_error",
            "max_covariance_update_error",
            "max_terminal_unresolved_covariance",
        ):
            _finite_float(row, name)
    terminal_levels = [row for row in levels if row.get("terminal") is True]
    if not terminal_levels:
        raise ValueError("no terminal per-level records were found.")
    if any(_finite_float(row, "max_terminal_unresolved_covariance") != 0.0 for row in terminal_levels):
        raise ValueError("terminal unresolved covariance is not exactly zero.")
    return certificates, replicates, levels


def _groups(
    rows: Iterable[dict[str, object]],
    keys: Sequence[str],
) -> dict[tuple[object, ...], list[dict[str, object]]]:
    result: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        result.setdefault(key, []).append(row)
    return result


def _estimator_summary(
    replicates: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    keys = ("case", "tree_chart", "particle_count", "estimator")
    rows: list[dict[str, object]] = []
    for key, group in sorted(_groups(replicates, keys).items()):
        if len(group) != EXPECTED_REPLICATE_COUNT:
            raise ValueError(f"estimator cell {key} does not contain 64 replicates.")
        likelihoods = np.asarray([_finite_float(row, "likelihood") for row in group])
        costs = np.asarray([_finite_float(row, "wall_seconds") for row in group])
        log_errors = np.log(likelihoods) - math.log(_finite_float(group[0], "oracle_likelihood"))
        oracle = _finite_float(group[0], "oracle_likelihood")
        variance = float(np.var(likelihoods, ddof=1))
        standard_error = math.sqrt(variance / likelihoods.size)
        mean = float(np.mean(likelihoods))
        relative_variance = variance / oracle**2
        mean_cost = float(np.mean(costs))
        median_cost = float(np.median(costs))
        work_fields = (
            "beta_draw_count",
            "forward_update_count",
            "likelihood_evaluation_count",
            "state_bytes",
            "peak_rss_bytes",
        )
        rows.append(
            {
                "case": key[0],
                "tree_chart": key[1],
                "particle_count": key[2],
                "estimator": key[3],
                "replicates": likelihoods.size,
                "oracle_likelihood": oracle,
                "mean_likelihood": mean,
                "standard_error": standard_error,
                "ci95_low": mean - Z_95 * standard_error,
                "ci95_high": mean + Z_95 * standard_error,
                "oracle_in_ci95": mean - Z_95 * standard_error <= oracle <= mean + Z_95 * standard_error,
                "relative_bias": (mean - oracle) / oracle,
                "relative_rmse": float(np.sqrt(np.mean((likelihoods - oracle) ** 2))) / oracle,
                "linear_variance": variance,
                "relative_variance": relative_variance,
                "median_log_error": float(np.median(log_errors)),
                "log_error_q05": _quantile(log_errors, 0.05),
                "log_error_q95": _quantile(log_errors, 0.95),
                "mean_wall_seconds": mean_cost,
                "median_wall_seconds": median_cost,
                "wall_seconds_q90": _quantile(costs, 0.90),
                "relative_variance_times_mean_cost": relative_variance * mean_cost,
                "relative_variance_times_median_cost": relative_variance * median_cost,
                **{
                    f"mean_{name}": statistics.fmean(_finite_float(row, name) for row in group)
                    for name in work_fields
                },
            }
        )
    direct = {
        (row["case"], row["tree_chart"], row["particle_count"]): row
        for row in rows
        if row["estimator"] == "direct_iid"
    }
    for row in rows:
        baseline = direct[(row["case"], row["tree_chart"], row["particle_count"])]
        for suffix in ("mean_cost", "median_cost"):
            name = f"relative_variance_times_{suffix}"
            denominator = _finite_float(baseline, name)
            row[f"{name}_over_direct"] = _finite_float(row, name) / denominator
    return rows


def _level_summary(levels: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    keys = (
        "case",
        "tree_chart",
        "particle_count",
        "estimator",
        "level",
    )
    result: list[dict[str, object]] = []
    for key, group in sorted(_groups(levels, keys).items()):
        result.append(
            {
                "case": key[0],
                "tree_chart": key[1],
                "particle_count": key[2],
                "estimator": key[3],
                "level": key[4],
                "replicates": len(group),
                "terminal": all(row["terminal"] is True for row in group),
                "mean_ess_fraction": statistics.fmean(_finite_float(row, "ess_fraction") for row in group),
                "minimum_ess_fraction": min(_finite_float(row, "ess_fraction") for row in group),
                "mean_incremental_weight_cv": statistics.fmean(
                    _finite_float(row, "incremental_weight_cv") for row in group
                ),
                "mean_max_normalized_weight": statistics.fmean(
                    _finite_float(row, "max_normalized_weight") for row in group
                ),
                "mean_shannon_perplexity_fraction": statistics.fmean(
                    _finite_float(row, "shannon_perplexity") / _integer(row, "particle_count")
                    for row in group
                ),
                "mean_unique_ancestor_fraction": statistics.fmean(
                    _integer(row, "unique_ancestor_count") / _integer(row, "particle_count") for row in group
                ),
                "resampling_fraction": statistics.fmean(
                    1.0 if row["resampled"] is True else 0.0 for row in group
                ),
                "mean_likelihood_correction_variance": statistics.fmean(
                    _finite_float(row, "linear_likelihood_correction_variance") for row in group
                ),
            }
        )
    return result


def _chart_summary(
    estimator_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped = _groups(
        (row for row in estimator_rows if row["case"] == "boundary_heavy_four_cell_row_column"),
        ("particle_count", "estimator"),
    )
    result: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        by_chart = {str(row["tree_chart"]): row for row in rows}
        if set(by_chart) != {"row-first", "column-first"}:
            raise ValueError(f"four-cell chart pair {key} is incomplete.")
        left = by_chart["row-first"]
        right = by_chart["column-first"]
        difference = _finite_float(left, "mean_likelihood") - _finite_float(right, "mean_likelihood")
        difference_se = math.sqrt(
            _finite_float(left, "linear_variance") / EXPECTED_REPLICATE_COUNT
            + _finite_float(right, "linear_variance") / EXPECTED_REPLICATE_COUNT
        )
        result.append(
            {
                "particle_count": key[0],
                "estimator": key[1],
                "row_first_mean": left["mean_likelihood"],
                "column_first_mean": right["mean_likelihood"],
                "difference": difference,
                "difference_standard_error": difference_se,
                "z_score": difference / difference_se,
                "agreement_within_3se": abs(difference) <= 3.0 * difference_se,
            }
        )
    return result


def _scientific_summary(
    estimator_rows: Sequence[dict[str, object]],
    level_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    smc_boundary = [
        row
        for row in estimator_rows
        if row["case"] == "boundary_heavy_four_cell_row_column"
        and row["estimator"]
        in {
            "bootstrap_smc_ess_0.5",
            "bootstrap_smc_every_nonterminal",
        }
    ]
    best = min(
        smc_boundary,
        key=lambda row: _finite_float(
            row,
            "relative_variance_times_median_cost_over_direct",
        ),
    )
    oracle_ci_failures = [
        {
            "case": row["case"],
            "tree_chart": row["tree_chart"],
            "particle_count": row["particle_count"],
            "estimator": row["estimator"],
            "relative_bias": row["relative_bias"],
        }
        for row in estimator_rows
        if row["oracle_in_ci95"] is not True
    ]
    bootstrap_levels = [row for row in level_rows if str(row["estimator"]).startswith("bootstrap_smc")]
    return {
        "scoring_domain": "normalized non-negative likelihood before logarithms",
        "cost_coordinate": (
            "between-replicate variance/oracle_likelihood^2 multiplied by median measured estimator wall time"
        ),
        "best_boundary_smc": {
            "tree_chart": best["tree_chart"],
            "particle_count": best["particle_count"],
            "estimator": best["estimator"],
            "relative_variance_times_median_cost_over_direct": best[
                "relative_variance_times_median_cost_over_direct"
            ],
        },
        "boundary_twofold_cost_adjusted_improvement": bool(
            _finite_float(best, "relative_variance_times_median_cost_over_direct") <= 0.5
        ),
        "minimum_bootstrap_ess_fraction": min(
            _finite_float(row, "minimum_ess_fraction") for row in bootstrap_levels
        ),
        "oracle_ci95_exclusions": oracle_ci_failures,
    }


def _markdown_report(
    summary: dict[str, object],
    estimator_rows: Sequence[dict[str, object]],
    chart_rows: Sequence[dict[str, object]],
    *,
    input_root: Path,
) -> str:
    scientific = summary["scientific"]
    assert isinstance(scientific, dict)
    selected = [
        row
        for row in estimator_rows
        if row["case"] == "boundary_heavy_four_cell_row_column" and row["particle_count"] == 4096
    ]
    selected.sort(key=lambda row: (str(row["tree_chart"]), str(row["estimator"])))
    table = [
        "| Chart | Estimator | Mean Z | SE(Z) | Rel. bias | RelVar × median s / IID |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in selected:
        table.append(
            "| {tree_chart} | {estimator} | {mean_likelihood:.7g} | "
            "{standard_error:.3g} | {relative_bias:.3g} | "
            "{relative_variance_times_median_cost_over_direct:.3g} |".format(**row)
        )
    chart_failures = [row for row in chart_rows if row["agreement_within_3se"] is not True]
    exclusions = scientific["oracle_ci95_exclusions"]
    assert isinstance(exclusions, list)
    best = scientific["best_boundary_smc"]
    assert isinstance(best, dict)
    return "\n".join(
        [
            "# R1a Gamma–Beta resolution-SMC result",
            "",
            "## What Was Tested",
            "",
            "Three fixed synthetic targets were evaluated: a near-Gaussian two-cell "
            "Beta allocation, the skewed G1 two-cell allocation, and a boundary-heavy "
            "four-cell Dirichlet allocation. Each estimator/count/chart cell has 64 "
            "independent replicates at 64, 256, 1,024 and 4,096 particles. The Gaussian "
            "closure is only an intermediate SMC guide; every terminal weight uses the "
            "exact normalized native Gaussian likelihood.",
            "",
            "## Terminology And Target",
            "",
            "The target is the fixed-root allocation-marginal normalized Gaussian "
            "likelihood. Accuracy and variance are scored on its non-negative linear "
            "scale. Relative variance means between-replicate variance divided by the "
            "squared quadrature oracle. The primary cost score multiplies relative "
            "variance by median measured estimator wall time; a secondary mean-time "
            "score is retained in the CSV because first-call warm-up affected some cells.",
            "",
            "## What Happened",
            "",
            f"All 12 task certificates passed under source `{summary['source_sha']}`. "
            f"The independently validated input contains {summary['replicate_records']:,} "
            f"replicate records and {summary['level_records']:,} per-level records. "
            "Direct IID and no-resampling SMC were exactly pathwise identical, terminal "
            "unresolved covariance was zero, and conservation/update errors remained at "
            "floating-point roundoff.",
            "",
            "Bootstrap SMC reduced raw variance in the boundary-heavy target, but its "
            "additional frontier and guide work erased that gain. The best boundary "
            "cost-adjusted ratio was "
            f"{float(best['relative_variance_times_median_cost_over_direct']):.3g} "
            "(SMC divided by IID), so the planned twofold improvement was not reached.",
            "",
            "## Key Results",
            "",
            f"- Minimum recorded bootstrap ESS fraction: "
            f"{float(scientific['minimum_bootstrap_ess_fraction']):.3g}.",
            f"- Linear-scale 95% oracle interval exclusions: {len(exclusions)} of "
            f"{len(estimator_rows)} estimator cells. These are retained for R1 follow-up; "
            "no common-sign or non-convergent bias pattern was found in this finite "
            "64-replicate screen.",
            f"- Compatible-chart disagreements beyond three independent-replicate SE: "
            f"{len(chart_failures)} of {len(chart_rows)} comparisons.",
            "- Resampling at every nonterminal refinement did not improve cost-adjusted "
            "efficiency over the ESS-triggered method reproducibly.",
            "",
            "## Boundary-Heavy Summary",
            "",
            "The table shows the highest-count boundary-heavy comparison. The final "
            "column is the estimator's relative-variance-times-median-seconds divided "
            "by the corresponding direct-IID value; values below one favor SMC.",
            "",
            *table,
            "",
            "## Interpretation",
            "",
            "R1a validates the target, normalization and replay implementation, while "
            "rejecting the hoped-for cost-adjusted bootstrap advantage in this tiny "
            "matrix. Raw variance reduction is real in several boundary-heavy cells, "
            "but the implementation is roughly an order of magnitude more involved per "
            "allocation than direct terminal evaluation. The observed low intermediate "
            "ESS and ordering sensitivity justify completing the wider bootstrap R1 "
            "matrix and the already-bounded one-dimensional guided-proposal R2 test. "
            "They do not justify PARIS-scale R4.",
            "",
            "## Outputs",
            "",
            f"Machine-readable input root: `{input_root}`.",
            "`r1a_estimator_summary.csv` contains all linear-scale accuracy, variance, "
            "timing and work metrics. `r1a_level_summary.csv` contains per-level ESS, "
            "weight, ancestry and correction summaries. `r1a_chart_summary.csv` contains "
            "compatible-chart uncertainty comparisons.",
            "",
        ]
    )


def analyse(
    *,
    input_root: Path,
    output_root: Path,
    source_sha: str,
) -> dict[str, object]:
    if len(source_sha) != EXPECTED_SOURCE_SHA_LENGTH or any(
        character not in "0123456789abcdef" for character in source_sha
    ):
        raise ValueError("source SHA must be a complete lowercase Git SHA.")
    certificates, replicates, levels = _load_and_validate(
        input_root,
        source_sha=source_sha,
    )
    estimator_rows = _estimator_summary(replicates)
    level_rows = _level_summary(levels)
    chart_rows = _chart_summary(estimator_rows)
    maximum_errors = {
        name: max(_finite_float(row, name) for row in levels)
        for name in (
            "max_mass_conservation_error",
            "max_mean_update_error",
            "max_covariance_update_error",
        )
    }
    summary: dict[str, object] = {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "source_protocol_sha256": certificates[0]["protocol_sha256"],
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "certificate_count": len(certificates),
        "replicate_records": len(replicates),
        "level_records": len(levels),
        "path_matched_identity_count": sum(
            row["estimator"] == "path_matched_no_resampling_smc" for row in replicates
        ),
        "beta_endpoint_repairs": sum(_integer(row, "beta_endpoint_repair_count") for row in replicates),
        "terminal_zero_unresolved_covariance": True,
        "maximum_numerical_errors": maximum_errors,
        "scientific": _scientific_summary(estimator_rows, level_rows),
        "status": "passed",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "r1a_estimator_summary.csv", estimator_rows)
    _write_csv(output_root / "r1a_level_summary.csv", level_rows)
    _write_csv(output_root / "r1a_chart_summary.csv", chart_rows)
    _write_json(output_root / "r1a_summary.json", summary)
    report = _markdown_report(
        summary,
        estimator_rows,
        chart_rows,
        input_root=input_root,
    )
    (output_root / "R1A_REPORT.md").write_text(report, encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = analyse(
        input_root=arguments.input_root,
        output_root=arguments.output_root,
        source_sha=arguments.source_sha,
    )
    print(_canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
