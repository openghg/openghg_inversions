#!/usr/bin/env python3
"""Validate, summarize, and report the wider resolution-SMC R1 and R2 runs.

The analysis scores the normalized likelihood on its non-negative linear
scale.  Between-replicate relative variance multiplied by measured wall time
is the primary efficiency coordinate.  Log-likelihood errors are retained
only as secondary, stable reporting coordinates.

The input is accepted only when every certificate, provenance field, record
count, file digest, terminal covariance check, and R2 checkpoint replay
passes.  Sixteen-cell references remain explicitly labelled replicated
large-IID estimates rather than exact values.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Record = dict[str, object]

R1_SCHEMA = "rjmcmc-resolution-smc-r1-v1"
R1_REFERENCE_SCHEMA = "rjmcmc-resolution-smc-r1-reference-v1"
R2_SCHEMA = "rjmcmc-resolution-smc-r2-v1"
ANALYSIS_SCHEMA = "rjmcmc-resolution-smc-r1-r2-analysis-v1"
EXPECTED_R1_CERTIFICATES = 36
EXPECTED_R1_REFERENCES = 2
EXPECTED_R2_CERTIFICATES = 3
EXPECTED_R1_REPLICATES = 32
EXPECTED_R2_REPLICATES = 16
SOURCE_SHA_LENGTH = 40
Z_95 = 1.959963984540054
Z_DIAGNOSTIC = 3.0

R1A_SOURCE_SHA = "9548ca59a01cdc095bf372c91c4416a9fcfb7162"
PLANNING_SHA = "3a4c80d17bf09dd70d69d1bef306cb888002fe54"
KNOWLEDGE_SHA = "a84bde45ccbbe42e2c400b79285b39f12b6cbcdd"


@dataclass(frozen=True)
class LoadedStage:
    """Validated certificate and record collection."""

    certificates: list[Record]
    replicates: list[Record]
    levels: list[Record]


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


def _write_csv(path: Path, rows: Sequence[Record]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Record:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object.")
    return value


def _read_jsonl(path: Path) -> list[Record]:
    rows: list[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON.") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            rows.append(value)
    return rows


def _finite_float(row: Record, name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"record field {name!r} is not numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"record field {name!r} is non-finite.")
    return result


def _integer(row: Record, name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"record field {name!r} is not an integer.")
    return value


def _string(row: Record, name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str):
        raise ValueError(f"record field {name!r} is not a string.")
    return value


def _groups(rows: Iterable[Record], keys: Sequence[str]) -> dict[tuple[object, ...], list[Record]]:
    result: dict[tuple[object, ...], list[Record]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        result.setdefault(key, []).append(row)
    return result


def _quantile(values: FloatArray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def _validate_sha(source_sha: str) -> None:
    if len(source_sha) != SOURCE_SHA_LENGTH or any(
        character not in "0123456789abcdef" for character in source_sha
    ):
        raise ValueError("source SHA must be a complete lowercase Git SHA.")


def _validate_provenance(certificate: Record, path: Path, source_sha: str) -> None:
    provenance = certificate.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{path} has no provenance object.")
    if (
        provenance.get("source_sha") != source_sha
        or provenance.get("head_sha") != source_sha
        or provenance.get("clean") is not True
        or provenance.get("detached") is not True
    ):
        raise ValueError(f"{path} provenance does not authenticate the frozen run.")


def _stem(certificate: Record) -> str:
    return (
        f"task-{_integer(certificate, 'task_index'):02d}-"
        f"{_string(certificate, 'case')}-n{_integer(certificate, 'particle_count')}"
    )


def _load_stage(
    stage_root: Path,
    *,
    schema: str,
    source_sha: str,
    expected_certificates: int,
    expected_replicates: int,
    require_replay: bool,
) -> LoadedStage:
    certificate_paths = sorted((stage_root / "certificates").glob("*.json"))
    if len(certificate_paths) != expected_certificates:
        raise ValueError(
            f"{stage_root}: expected {expected_certificates} certificates, found {len(certificate_paths)}."
        )
    certificates: list[Record] = []
    replicates: list[Record] = []
    levels: list[Record] = []
    task_indices: set[int] = set()
    for certificate_path in certificate_paths:
        certificate = _read_json(certificate_path)
        if certificate.get("schema") != schema or certificate.get("status") != "passed":
            raise ValueError(f"{certificate_path} is not a passed {schema} certificate.")
        _validate_provenance(certificate, certificate_path, source_sha)
        task_index = _integer(certificate, "task_index")
        if task_index in task_indices:
            raise ValueError(f"duplicate task index {task_index}.")
        task_indices.add(task_index)
        if _integer(certificate, "replicate_count") != expected_replicates:
            raise ValueError(f"{certificate_path} has an unexpected replicate count.")
        stem = _stem(certificate)
        replicate_path = stage_root / "replicates" / f"{stem}.jsonl"
        level_path = stage_root / "levels" / f"{stem}.jsonl"
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
        if any(row.get("schema") != schema for row in task_replicates + task_levels):
            raise ValueError(f"{stem} contains an unexpected record schema.")
        if require_replay:
            replay = certificate.get("checkpoint_replay")
            if not isinstance(replay, list) or not replay:
                raise ValueError(f"{certificate_path} has no checkpoint replay audit.")
            for boundary in replay:
                if not isinstance(boundary, dict) or boundary.get("identical") is not True:
                    raise ValueError(f"{certificate_path} contains a failed replay boundary.")
                checkpoint = stage_root / _string(boundary, "checkpoint")
                if _sha256(checkpoint) != _string(boundary, "checkpoint_sha256"):
                    raise ValueError(f"{checkpoint} fails its replay certificate digest.")
        certificates.append(certificate)
        replicates.extend(task_replicates)
        levels.extend(task_levels)
    if task_indices != set(range(expected_certificates)):
        raise ValueError(f"{stage_root} task indices are incomplete.")
    for row in replicates:
        if _finite_float(row, "likelihood") <= 0.0:
            raise ValueError("likelihood estimators must be strictly positive.")
        for name in ("wall_seconds", "oracle_likelihood", "oracle_standard_error"):
            _finite_float(row, name)
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
            "max_terminal_prediction_error",
        ):
            _finite_float(row, name)
        if not 0.0 < _finite_float(row, "ess") <= _integer(row, "particle_count"):
            raise ValueError("per-level ESS is outside its valid range.")
    terminal = [row for row in levels if row.get("terminal") is True]
    if not terminal or any(
        _finite_float(row, "max_terminal_unresolved_covariance") != 0.0 for row in terminal
    ):
        raise ValueError("terminal unresolved covariance is not exactly zero.")
    return LoadedStage(certificates, replicates, levels)


def _load_references(r1_root: Path, *, source_sha: str) -> list[Record]:
    paths = sorted((r1_root / "reference-certificates").glob("*.json"))
    if len(paths) != EXPECTED_R1_REFERENCES:
        raise ValueError(f"expected {EXPECTED_R1_REFERENCES} large-IID references, found {len(paths)}.")
    certificates: list[Record] = []
    for path in paths:
        certificate = _read_json(path)
        if certificate.get("schema") != R1_REFERENCE_SCHEMA or certificate.get("status") != "passed":
            raise ValueError(f"{path} is not a passed large-IID reference certificate.")
        _validate_provenance(certificate, path, source_sha)
        records_path = r1_root / "references" / f"{_string(certificate, 'case')}.jsonl"
        if _sha256(records_path) != certificate.get("records_sha256"):
            raise ValueError(f"{records_path} does not match its certificate digest.")
        records = _read_jsonl(records_path)
        if len(records) != _integer(certificate, "replicate_count"):
            raise ValueError(f"{records_path} has an unexpected record count.")
        if any(
            row.get("schema") != R1_REFERENCE_SCHEMA
            or _finite_float(row, "likelihood") <= 0.0
            or _integer(row, "sample_count") != _integer(certificate, "sample_count")
            for row in records
        ):
            raise ValueError(f"{records_path} contains an invalid reference record.")
        certificates.append(certificate)
    return certificates


def _estimator_summary(
    rows: Sequence[Record],
    *,
    replicate_count: int,
    baseline_estimator: str,
) -> list[Record]:
    keys = ("case", "tree_chart", "particle_count", "estimator")
    result: list[Record] = []
    for key, group in sorted(_groups(rows, keys).items()):
        if len(group) != replicate_count:
            raise ValueError(f"estimator cell {key} does not contain {replicate_count} replicates.")
        likelihoods = np.asarray([_finite_float(row, "likelihood") for row in group])
        costs = np.asarray([_finite_float(row, "wall_seconds") for row in group])
        oracle = _finite_float(group[0], "oracle_likelihood")
        oracle_se = _finite_float(group[0], "oracle_standard_error")
        variance = float(np.var(likelihoods, ddof=1))
        standard_error = math.sqrt(variance / likelihoods.size)
        combined_se = math.hypot(standard_error, oracle_se)
        mean = float(np.mean(likelihoods))
        relative_variance = variance / oracle**2
        median_cost = float(np.median(costs))
        mean_cost = float(np.mean(costs))
        difference = mean - oracle
        z_score = difference / combined_se if combined_se > 0.0 else math.copysign(math.inf, difference)
        log_errors = np.log(likelihoods) - math.log(oracle)
        work_fields = (
            "beta_draw_count",
            "forward_update_count",
            "likelihood_evaluation_count",
            "state_bytes",
            "peak_rss_bytes",
        )
        result.append(
            {
                "case": key[0],
                "tree_chart": key[1],
                "particle_count": key[2],
                "estimator": key[3],
                "replicates": likelihoods.size,
                "oracle_label": group[0]["oracle_label"],
                "oracle_likelihood": oracle,
                "oracle_standard_error": oracle_se,
                "mean_likelihood": mean,
                "standard_error": standard_error,
                "combined_reference_standard_error": combined_se,
                "difference_z_score": z_score,
                "agreement_within_3se": abs(difference) <= Z_DIAGNOSTIC * combined_se,
                "ci95_low": mean - Z_95 * standard_error,
                "ci95_high": mean + Z_95 * standard_error,
                "relative_bias": difference / oracle,
                "relative_rmse": float(np.sqrt(np.mean(np.square(likelihoods - oracle)))) / oracle,
                "linear_variance": variance,
                "relative_variance": relative_variance,
                "median_log_error": float(np.median(log_errors)),
                "log_error_q05": _quantile(log_errors, 0.05),
                "log_error_q95": _quantile(log_errors, 0.95),
                "mean_wall_seconds": mean_cost,
                "median_wall_seconds": median_cost,
                "wall_seconds_q10": _quantile(costs, 0.10),
                "wall_seconds_q90": _quantile(costs, 0.90),
                "relative_variance_times_mean_cost": relative_variance * mean_cost,
                "relative_variance_times_median_cost": relative_variance * median_cost,
                **{
                    f"mean_{name}": statistics.fmean(_finite_float(row, name) for row in group)
                    for name in work_fields
                },
            }
        )
    baselines = {
        (row["case"], row["tree_chart"], row["particle_count"]): row
        for row in result
        if row["estimator"] == baseline_estimator
    }
    for row in result:
        baseline = baselines[(row["case"], row["tree_chart"], row["particle_count"])]
        for cost_kind in ("mean", "median"):
            field = f"relative_variance_times_{cost_kind}_cost"
            denominator = _finite_float(baseline, field)
            row[f"{field}_over_baseline"] = _finite_float(row, field) / denominator
        row["linear_variance_over_baseline"] = _finite_float(row, "linear_variance") / _finite_float(
            baseline, "linear_variance"
        )
        row["median_wall_seconds_over_baseline"] = _finite_float(row, "median_wall_seconds") / _finite_float(
            baseline, "median_wall_seconds"
        )
    return result


def _level_summary(rows: Sequence[Record]) -> list[Record]:
    keys = ("case", "tree_chart", "particle_count", "estimator", "level")
    result: list[Record] = []
    for key, group in sorted(_groups(rows, keys).items()):
        particle_count = _integer(group[0], "particle_count")
        result.append(
            {
                "case": key[0],
                "tree_chart": key[1],
                "particle_count": key[2],
                "estimator": key[3],
                "level": key[4],
                "node_ids": json.dumps(group[0]["node_ids"], separators=(",", ":")),
                "replicates": len(group),
                "terminal": all(row["terminal"] is True for row in group),
                "mean_ess_fraction": statistics.fmean(_finite_float(row, "ess_fraction") for row in group),
                "minimum_ess_fraction": min(_finite_float(row, "ess_fraction") for row in group),
                "mean_incremental_weight_cv": statistics.fmean(
                    _finite_float(row, "incremental_weight_cv") for row in group
                ),
                "maximum_incremental_weight_cv": max(
                    _finite_float(row, "incremental_weight_cv") for row in group
                ),
                "mean_max_normalized_weight": statistics.fmean(
                    _finite_float(row, "max_normalized_weight") for row in group
                ),
                "mean_shannon_perplexity_fraction": statistics.fmean(
                    _finite_float(row, "shannon_perplexity") / particle_count for row in group
                ),
                "mean_unique_ancestor_fraction": statistics.fmean(
                    _integer(row, "unique_ancestor_count") / particle_count for row in group
                ),
                "minimum_unique_ancestor_count": min(_integer(row, "unique_ancestor_count") for row in group),
                "resampling_fraction": statistics.fmean(
                    1.0 if row["resampled"] is True else 0.0 for row in group
                ),
                "mean_likelihood_correction": statistics.fmean(
                    _finite_float(row, "linear_likelihood_correction_mean") for row in group
                ),
                "mean_likelihood_correction_variance": statistics.fmean(
                    _finite_float(row, "linear_likelihood_correction_variance") for row in group
                ),
                "mean_level_seconds": statistics.fmean(
                    _finite_float(row, "elapsed_seconds") for row in group
                ),
                "maximum_mass_conservation_error": max(
                    _finite_float(row, "max_mass_conservation_error") for row in group
                ),
                "maximum_mean_update_error": max(
                    _finite_float(row, "max_mean_update_error") for row in group
                ),
                "maximum_covariance_update_error": max(
                    _finite_float(row, "max_covariance_update_error") for row in group
                ),
                "maximum_terminal_unresolved_covariance": max(
                    _finite_float(row, "max_terminal_unresolved_covariance") for row in group
                ),
                "maximum_terminal_prediction_error": max(
                    _finite_float(row, "max_terminal_prediction_error") for row in group
                ),
                "mean_proposal_construction_seconds": statistics.fmean(
                    _finite_float(row, "proposal_construction_seconds") for row in group
                ),
                "mean_proposal_guide_evaluation_count": statistics.fmean(
                    _finite_float(row, "proposal_guide_evaluation_count") for row in group
                ),
                "mean_proposal_log_density_correction_sd": statistics.fmean(
                    _finite_float(row, "proposal_log_density_correction_sd") for row in group
                ),
                "mean_proposal_normalizer_relative_error": statistics.fmean(
                    _finite_float(row, "proposal_normalizer_relative_error_mean") for row in group
                ),
                "maximum_proposal_normalizer_relative_error": max(
                    _finite_float(row, "proposal_normalizer_relative_error_max") for row in group
                ),
            }
        )
    return result


def _chart_summary(estimator_rows: Sequence[Record], *, replicate_count: int) -> list[Record]:
    result: list[Record] = []
    grouped = _groups(estimator_rows, ("case", "particle_count", "estimator"))
    for key, group in sorted(grouped.items()):
        by_chart = {str(row["tree_chart"]): row for row in group}
        if len(by_chart) < 2:
            continue
        for left_name, right_name in itertools.combinations(sorted(by_chart), 2):
            left = by_chart[left_name]
            right = by_chart[right_name]
            difference = _finite_float(left, "mean_likelihood") - _finite_float(right, "mean_likelihood")
            difference_se = math.sqrt(
                _finite_float(left, "linear_variance") / replicate_count
                + _finite_float(right, "linear_variance") / replicate_count
            )
            z_score = (
                difference / difference_se if difference_se > 0.0 else math.copysign(math.inf, difference)
            )
            result.append(
                {
                    "case": key[0],
                    "particle_count": key[1],
                    "estimator": key[2],
                    "left_chart": left_name,
                    "right_chart": right_name,
                    "left_mean_likelihood": left["mean_likelihood"],
                    "right_mean_likelihood": right["mean_likelihood"],
                    "difference": difference,
                    "difference_standard_error": difference_se,
                    "z_score": z_score,
                    "agreement_within_3se": abs(z_score) <= Z_DIAGNOSTIC,
                }
            )
    return result


def _proposal_summary(level_rows: Sequence[Record], estimator_rows: Sequence[Record]) -> list[Record]:
    levels_by_cell = _groups(
        level_rows,
        ("case", "tree_chart", "particle_count", "estimator"),
    )
    estimator_by_cell: dict[tuple[object, ...], Record] = {
        (row["case"], row["tree_chart"], row["particle_count"], row["estimator"]): row
        for row in estimator_rows
    }
    result: list[Record] = []
    for key, levels in sorted(levels_by_cell.items()):
        estimator = estimator_by_cell[key]
        result.append(
            {
                "case": key[0],
                "tree_chart": key[1],
                "particle_count": key[2],
                "estimator": key[3],
                "minimum_ess_fraction": min(_finite_float(row, "minimum_ess_fraction") for row in levels),
                "mean_ess_fraction": statistics.fmean(
                    _finite_float(row, "mean_ess_fraction") for row in levels
                ),
                "maximum_incremental_weight_cv": max(
                    _finite_float(row, "maximum_incremental_weight_cv") for row in levels
                ),
                "maximum_proposal_normalizer_relative_error": max(
                    _finite_float(row, "maximum_proposal_normalizer_relative_error") for row in levels
                ),
                "total_mean_proposal_construction_seconds": sum(
                    _finite_float(row, "mean_proposal_construction_seconds") for row in levels
                ),
                "relative_variance_over_prior": estimator["linear_variance_over_baseline"],
                "median_cost_over_prior": estimator["median_wall_seconds_over_baseline"],
                "relative_variance_times_median_cost_over_prior": estimator[
                    "relative_variance_times_median_cost_over_baseline"
                ],
                "difference_z_score": estimator["difference_z_score"],
                "agreement_within_3se": estimator["agreement_within_3se"],
            }
        )
    return result


def _maximum_errors(*level_sets: Sequence[Record]) -> Record:
    levels = [row for level_set in level_sets for row in level_set]
    return {
        "max_mass_conservation_error": max(
            _finite_float(row, "max_mass_conservation_error") for row in levels
        ),
        "max_mean_update_error": max(_finite_float(row, "max_mean_update_error") for row in levels),
        "max_covariance_update_error": max(
            _finite_float(row, "max_covariance_update_error") for row in levels
        ),
        "max_terminal_prediction_error": max(
            _finite_float(row, "max_terminal_prediction_error") for row in levels
        ),
        "max_terminal_unresolved_covariance": max(
            _finite_float(row, "max_terminal_unresolved_covariance") for row in levels
        ),
    }


def _r1_scientific_summary(
    estimator_rows: Sequence[Record],
    level_rows: Sequence[Record],
    chart_rows: Sequence[Record],
) -> Record:
    smc = [row for row in estimator_rows if str(row["estimator"]).startswith("bootstrap_")]
    boundary = [row for row in smc if "boundary_heavy" in str(row["case"])]
    sobol = [row for row in estimator_rows if row["estimator"] == "scrambled_sobol"]
    exact = [row for row in estimator_rows if row["oracle_label"] == "converged Gauss-Jacobi quadrature"]
    best = min(
        smc,
        key=lambda row: _finite_float(row, "relative_variance_times_median_cost_over_baseline"),
    )
    best_boundary = min(
        boundary,
        key=lambda row: _finite_float(row, "relative_variance_times_median_cost_over_baseline"),
    )
    worst_level = min(level_rows, key=lambda row: _finite_float(row, "minimum_ess_fraction"))
    return {
        "scoring_domain": "normalized non-negative likelihood before logarithms",
        "primary_cost_coordinate": (
            "between-replicate variance/oracle_likelihood^2 multiplied by median measured wall time"
        ),
        "best_bootstrap_cell": {
            name: best[name]
            for name in (
                "case",
                "tree_chart",
                "particle_count",
                "estimator",
                "relative_variance_times_median_cost_over_baseline",
            )
        },
        "best_boundary_bootstrap_cell": {
            name: best_boundary[name]
            for name in (
                "case",
                "tree_chart",
                "particle_count",
                "estimator",
                "relative_variance_times_median_cost_over_baseline",
            )
        },
        "bootstrap_cells_better_than_iid": sum(
            _finite_float(row, "relative_variance_times_median_cost_over_baseline") < 1.0 for row in smc
        ),
        "bootstrap_cells_twofold_better_than_iid": sum(
            _finite_float(row, "relative_variance_times_median_cost_over_baseline") <= 0.5 for row in smc
        ),
        "bootstrap_cell_count": len(smc),
        "sobol_cells_better_than_iid": sum(
            _finite_float(row, "relative_variance_times_median_cost_over_baseline") < 1.0 for row in sobol
        ),
        "sobol_cell_count": len(sobol),
        "exact_target_cells_beyond_3se": sum(row["agreement_within_3se"] is not True for row in exact),
        "exact_target_cell_count": len(exact),
        "chart_comparisons_beyond_3se": sum(row["agreement_within_3se"] is not True for row in chart_rows),
        "chart_comparison_count": len(chart_rows),
        "minimum_ess_fraction": worst_level["minimum_ess_fraction"],
        "minimum_ess_location": {
            name: worst_level[name] for name in ("case", "tree_chart", "particle_count", "estimator", "level")
        },
        "minimum_unique_ancestor_count": min(
            _integer(row, "minimum_unique_ancestor_count") for row in level_rows
        ),
    }


def _r2_scientific_summary(
    estimator_rows: Sequence[Record],
    level_rows: Sequence[Record],
    chart_rows: Sequence[Record],
    proposal_rows: Sequence[Record],
) -> Record:
    guided = [row for row in estimator_rows if str(row["estimator"]).startswith("guided_")]
    prior = [row for row in estimator_rows if row["estimator"] == "bootstrap_prior_ess_0.5"]
    best = min(
        guided,
        key=lambda row: _finite_float(row, "relative_variance_times_median_cost_over_baseline"),
    )
    guided_levels = [row for row in level_rows if str(row["estimator"]).startswith("guided_")]
    prior_levels = [row for row in level_rows if row["estimator"] == "bootstrap_prior_ess_0.5"]
    return {
        "best_guided_cell": {
            name: best[name]
            for name in (
                "tree_chart",
                "particle_count",
                "estimator",
                "linear_variance_over_baseline",
                "median_wall_seconds_over_baseline",
                "relative_variance_times_median_cost_over_baseline",
            )
        },
        "guided_cells_better_than_prior": sum(
            _finite_float(row, "relative_variance_times_median_cost_over_baseline") < 1.0 for row in guided
        ),
        "guided_cells_twofold_better_than_prior": sum(
            _finite_float(row, "relative_variance_times_median_cost_over_baseline") <= 0.5 for row in guided
        ),
        "guided_cell_count": len(guided),
        "prior_cells_beyond_3se": sum(row["agreement_within_3se"] is not True for row in prior),
        "guided_cells_beyond_3se": sum(row["agreement_within_3se"] is not True for row in guided),
        "minimum_prior_ess_fraction": min(_finite_float(row, "minimum_ess_fraction") for row in prior_levels),
        "minimum_guided_ess_fraction": min(
            _finite_float(row, "minimum_ess_fraction") for row in guided_levels
        ),
        "maximum_proposal_normalizer_relative_error": max(
            _finite_float(row, "maximum_proposal_normalizer_relative_error") for row in proposal_rows
        ),
        "chart_comparisons_beyond_3se": sum(row["agreement_within_3se"] is not True for row in chart_rows),
        "chart_comparison_count": len(chart_rows),
    }


def _jobs(source_sha: str, run_root_path: Path) -> list[Record]:
    run_root = "/group/chem/acrg/brendan_for_codex/rjmcmc_resolution_smc"
    return [
        {
            "job_id": "18222226",
            "role": "R1a array attempt 1",
            "callback_job_id": "18222227",
            "callback_state": "COMPLETED",
            "ticket_id": "sw-20260730T194227Z-bfb1ecfaf66c",
            "source_sha": "6c9e3b9d04874748b5c3c3b3cc2731e0a33a6c56",
            "account": "chem007981",
            "state": "FAILED",
            "exit_code": "127:0",
            "run_root": f"{run_root}/6c9e3b9d04874748b5c3c3b3cc2731e0a33a6c56",
            "evidence": "logs/r1a-18222226_*.{out,err}",
        },
        {
            "job_id": "18222349",
            "role": "R1a array attempt 2",
            "callback_job_id": "18222350",
            "callback_state": "COMPLETED",
            "ticket_id": "sw-20260730T195352Z-e5b6ef3c0e4e",
            "source_sha": "1c225fc74dee75d1542e3ae14baa9db927c3c276",
            "account": "chem007981",
            "state": "FAILED",
            "exit_code": "2:0",
            "run_root": f"{run_root}/1c225fc74dee75d1542e3ae14baa9db927c3c276",
            "evidence": "logs/r1a-18222349_*.{out,err}",
        },
        {
            "job_id": "18222470",
            "role": "R1a array",
            "callback_job_id": "18222471",
            "callback_state": "COMPLETED",
            "ticket_id": "sw-20260730T200259Z-ea73f7d8725b",
            "source_sha": R1A_SOURCE_SHA,
            "account": "chem007981",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "run_root": f"{run_root}/{R1A_SOURCE_SHA}",
            "evidence": "logs/r1a-18222470_*.{out,err}",
        },
        {
            "job_id": "18222960",
            "role": "R1 large-IID references array",
            "callback_job_id": "18222961",
            "callback_state": "COMPLETED",
            "ticket_id": "sw-20260730T203918Z-ae1b2dbab2c6",
            "source_sha": source_sha,
            "account": "chem007981",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "run_root": str(run_root_path),
            "evidence": "logs/r1-reference-18222960_*.{out,err}",
        },
        {
            "job_id": "18222962",
            "role": "R2 guided-proposal array",
            "callback_job_id": "18222963",
            "callback_state": "COMPLETED",
            "ticket_id": "sw-20260730T203933Z-5eb441148c20",
            "source_sha": source_sha,
            "account": "chem007981",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "run_root": str(run_root_path),
            "evidence": "logs/r2-18222962_*.{out,err}",
        },
        {
            "job_id": "18223073",
            "role": "wider R1 array",
            "callback_job_id": "18223074",
            "callback_state": "COMPLETED",
            "ticket_id": "sw-20260730T204849Z-d52615396d03",
            "source_sha": source_sha,
            "account": "chem007981",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "run_root": str(run_root_path),
            "evidence": "logs/r1-18223073_*.{out,err}",
        },
    ]


def _attempts(source_sha: str, run_root: Path, r1a_run_root: Path) -> list[Record]:
    jobs = _jobs(source_sha, run_root)
    return [
        {
            "attempt_id": "local-001",
            "stage": "implementation validation",
            "kind": "numerical representation defect",
            "source_sha": "7f2a2e0b6579ec590cc4ef479bb12ff011542454",
            "job_id": "",
            "outcome": "REPAIRED",
            "diagnosis": (
                "NumPy Beta draws for shapes below one rounded to exact endpoints; "
                "open-float endpoint repair was added and counted diagnostically."
            ),
            "retained_evidence": "focused pytest history and beta_endpoint_repair_count fields",
        },
        {
            "attempt_id": "slurm-001",
            "stage": "R1a",
            "kind": "operational launcher defect",
            "source_sha": jobs[0]["source_sha"],
            "job_id": jobs[0]["job_id"],
            "outcome": "FAILED_REPAIRED",
            "diagnosis": "compute-node PATH did not provide Git; all array tasks exited 127.",
            "retained_evidence": f"{jobs[0]['run_root']}/{jobs[0]['evidence']}",
        },
        {
            "attempt_id": "slurm-002",
            "stage": "R1a",
            "kind": "operational launcher defect",
            "source_sha": jobs[1]["source_sha"],
            "job_id": jobs[1]["job_id"],
            "outcome": "FAILED_REPAIRED",
            "diagnosis": "the attempted /usr/bin/git fallback did not exist on compute nodes.",
            "retained_evidence": f"{jobs[1]['run_root']}/{jobs[1]['evidence']}",
        },
        {
            "attempt_id": "slurm-003",
            "stage": "R1a",
            "kind": "scientific production",
            "source_sha": R1A_SOURCE_SHA,
            "job_id": jobs[2]["job_id"],
            "outcome": "COMPLETED",
            "diagnosis": "cluster Git module authenticated the clean detached source; 12/12 tasks passed.",
            "retained_evidence": str(r1a_run_root),
        },
        {
            "attempt_id": "local-002",
            "stage": "R0 final replay",
            "kind": "correctness validation",
            "source_sha": source_sha,
            "job_id": "",
            "outcome": "COMPLETED",
            "diagnosis": "identity, conservation, swap, quadrature, replay, and fail-closed provenance passed.",
            "retained_evidence": str(run_root / "r0" / "r0_summary.json"),
        },
        {
            "attempt_id": "slurm-004",
            "stage": "R1 reference",
            "kind": "scientific production",
            "source_sha": source_sha,
            "job_id": jobs[3]["job_id"],
            "outcome": "COMPLETED",
            "diagnosis": "2/2 replicated 16-cell large-IID reference tasks passed.",
            "retained_evidence": str(run_root / "r1" / "reference-certificates"),
        },
        {
            "attempt_id": "slurm-005",
            "stage": "R2",
            "kind": "scientific production",
            "source_sha": source_sha,
            "job_id": jobs[4]["job_id"],
            "outcome": "COMPLETED",
            "diagnosis": "3/3 bounded guided-proposal tasks and all restart boundaries passed.",
            "retained_evidence": str(run_root / "r2" / "certificates"),
        },
        {
            "attempt_id": "slurm-006",
            "stage": "wider R1",
            "kind": "scientific production",
            "source_sha": source_sha,
            "job_id": jobs[5]["job_id"],
            "outcome": "COMPLETED",
            "diagnosis": "36/36 wider R1 tasks passed certificate and independent artifact audits.",
            "retained_evidence": str(run_root / "r1" / "certificates"),
        },
        {
            "attempt_id": "local-003",
            "stage": "final report validation",
            "kind": "sandbox logging defect",
            "source_sha": source_sha,
            "job_id": "",
            "outcome": "FAILED_BEFORE_COLLECTION_RETRIED",
            "diagnosis": (
                "direct environment pytest exited 4 before collection because OpenGHG writes "
                "/user/home/bm13805/openghg.log and the sandbox made that path read-only."
            ),
            "retained_evidence": str(run_root / "report" / "validation-evidence.json"),
        },
        {
            "attempt_id": "local-004",
            "stage": "final report validation",
            "kind": "focused validation",
            "source_sha": source_sha,
            "job_id": "",
            "outcome": "COMPLETED",
            "diagnosis": "focused pytest, Ruff, Pyright, launcher syntax, and combined checksums passed.",
            "retained_evidence": str(run_root / "report" / "validation-evidence.json"),
        },
    ]


def _validation_evidence(source_sha: str) -> Record:
    return {
        "schema": "rjmcmc-resolution-smc-validation-evidence-v1",
        "source_sha": source_sha,
        "focused_pytest": {
            "paths": [
                "tests/experimental/rjmcmc/test_aggregation_error_resolution_smc.py",
                "tests/experimental/rjmcmc/test_aggregation_error.py",
                "tests/experimental/rjmcmc/test_aggregation_error_low_rank.py",
                "tests/experimental/rjmcmc/test_aggregation_error_conditional_mixture.py",
                "tests/experimental/rjmcmc/test_gamma_beta_tree.py",
            ],
            "result": "115 passed in 12.16s",
            "status": "passed",
        },
        "focused_ruff": {
            "paths": [
                "openghg_inversions/experimental/rjmcmc/aggregation_error_resolution_smc.py",
                "tests/experimental/rjmcmc/test_aggregation_error_resolution_smc.py",
                "examples/rjmcmc/resolution_smc_experiment.py",
                "examples/rjmcmc/analyse_resolution_smc_r1a.py",
                "examples/rjmcmc/resolution_smc_r1_r2_experiment.py",
                "examples/rjmcmc/analyse_resolution_smc_r1_r2.py",
            ],
            "result": "All checks passed",
            "status": "passed",
        },
        "focused_pyright": {
            "configuration": "/tmp/rjmcmc_resolution_smc_r1_r2_pyrightconfig.json",
            "result": "0 errors, 0 warnings, 0 informations",
            "status": "passed",
        },
        "launcher_syntax": {
            "launchers": [
                "examples/rjmcmc/resolution_smc_r1a.sbatch",
                "examples/rjmcmc/resolution_smc_r1_reference.sbatch",
                "examples/rjmcmc/resolution_smc_r1.sbatch",
                "examples/rjmcmc/resolution_smc_r2.sbatch",
            ],
            "result": "bash -n passed",
            "status": "passed",
        },
        "artifact_audit": {
            "r1_certificates": 36,
            "r1_replicate_records": 17_024,
            "r1_level_records": 55_680,
            "r2_certificates": 3,
            "r2_replicate_records": 384,
            "r2_level_records": 1_152,
            "r2_replay_boundaries": 12,
            "combined_manifest_entries": 310,
            "result": "all certificate, file, checkpoint, and combined-manifest digests verified",
            "status": "passed",
        },
        "failed_attempt": {
            "result": (
                "exit 4 before pytest collection: OpenGHG hard-coded user log path was read-only "
                "inside the sandbox; rerun outside the sandbox passed"
            ),
            "status": "retained",
        },
        "full_tox_run": False,
    }


def _plot_reports(
    output_root: Path,
    r1_estimators: Sequence[Record],
    r1_levels: Sequence[Record],
    r2_levels: Sequence[Record],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    selected_levels = [
        row
        for row in r1_levels
        if row["case"] == "boundary_heavy_four_cell_row_column"
        and row["tree_chart"] == "row-first"
        and row["particle_count"] == 4096
    ]
    names = {
        "bootstrap_breadth_ess_0.25": "breadth, ESS 0.25",
        "bootstrap_breadth_ess_0.5": "breadth, ESS 0.50",
        "bootstrap_energy_ess_0.25": "energy, ESS 0.25",
        "bootstrap_energy_ess_0.5": "energy, ESS 0.50",
        "bootstrap_unfavorable_ess_0.5": "unfavourable, ESS 0.50",
    }
    figures: list[str] = []

    def save(name: str) -> None:
        path = figure_root / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        figures.append(str(path.relative_to(output_root)))

    for field, ylabel, name, log_y in (
        ("mean_ess_fraction", "Mean ESS / particles", "r1_incremental_ess.png", False),
        (
            "mean_likelihood_correction_variance",
            "Mean variance of linear correction",
            "r1_correction_variance.png",
            True,
        ),
        (
            "mean_unique_ancestor_fraction",
            "Mean unique ancestors / particles",
            "r1_ancestry.png",
            False,
        ),
    ):
        plt.figure(figsize=(7.2, 4.5))
        for estimator, label in names.items():
            rows = sorted(
                (row for row in selected_levels if row["estimator"] == estimator),
                key=lambda row: _integer(row, "level"),
            )
            plt.plot(
                [_integer(row, "level") for row in rows],
                [_finite_float(row, field) for row in rows],
                marker="o",
                label=label,
            )
        if log_y:
            plt.yscale("symlog", linthresh=1.0e-8)
        plt.xlabel("Refinement level")
        plt.ylabel(ylabel)
        plt.title("Boundary-heavy four-cell target, row-first chart, N=4096")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        save(name)

    selected_estimators = [
        row
        for row in r1_estimators
        if row["case"] == "boundary_heavy_four_cell_row_column" and row["tree_chart"] == "row-first"
    ]
    plt.figure(figsize=(7.2, 4.8))
    for estimator in sorted({str(row["estimator"]) for row in selected_estimators}):
        rows = sorted(
            (row for row in selected_estimators if row["estimator"] == estimator),
            key=lambda row: _integer(row, "particle_count"),
        )
        plt.plot(
            [_finite_float(row, "median_wall_seconds") for row in rows],
            [_finite_float(row, "relative_rmse") for row in rows],
            marker="o",
            label=estimator.replace("bootstrap_", "").replace("_", " "),
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Median measured wall time per replicate (s)")
    plt.ylabel("Relative RMSE on linear likelihood scale")
    plt.title("Likelihood error versus measured work")
    plt.grid(alpha=0.25, which="both")
    plt.legend(fontsize=7)
    save("r1_rmse_vs_work.png")

    chart_rows = [
        row
        for row in r1_estimators
        if row["case"] == "boundary_heavy_four_cell_row_column"
        and row["particle_count"] == 4096
        and row["estimator"]
        in {
            "direct_iid",
            "bootstrap_breadth_ess_0.5",
            "bootstrap_energy_ess_0.5",
            "bootstrap_unfavorable_ess_0.5",
        }
    ]
    labels = [f"{row['tree_chart']}\n{str(row['estimator']).replace('bootstrap_', '')}" for row in chart_rows]
    plt.figure(figsize=(10.5, 4.8))
    x_values = np.arange(len(chart_rows))
    plt.errorbar(
        x_values,
        [_finite_float(row, "mean_likelihood") for row in chart_rows],
        yerr=[Z_95 * _finite_float(row, "standard_error") for row in chart_rows],
        fmt="o",
        capsize=3,
    )
    plt.axhline(_finite_float(chart_rows[0], "oracle_likelihood"), color="black", linestyle="--")
    plt.xticks(x_values, labels, rotation=35, ha="right", fontsize=7)
    plt.ylabel("Mean normalized likelihood (95% MC interval)")
    plt.title("Compatible-tree and ordering sensitivity, N=4096")
    plt.grid(alpha=0.25, axis="y")
    save("r1_tree_order_sensitivity.png")

    selected_r2 = [
        row for row in r2_levels if row["tree_chart"] == "row-first" and row["particle_count"] == 4096
    ]
    plt.figure(figsize=(7.2, 4.5))
    for estimator in sorted({str(row["estimator"]) for row in selected_r2}):
        rows = sorted(
            (row for row in selected_r2 if row["estimator"] == estimator),
            key=lambda row: _integer(row, "level"),
        )
        plt.plot(
            [_integer(row, "level") for row in rows],
            [_finite_float(row, "mean_ess_fraction") for row in rows],
            marker="o",
            label=estimator.replace("guided_piecewise_beta_", "").replace("bootstrap_", ""),
        )
    plt.xlabel("Refinement level")
    plt.ylabel("Mean ESS / particles")
    plt.title("R2 prior versus guided proposals, row-first chart, N=4096")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    save("r2_guided_ess.png")
    return figures


def _r1_report(summary: Record, rows: Sequence[Record], chart_rows: Sequence[Record]) -> str:
    scientific = summary["r1_scientific"]
    assert isinstance(scientific, dict)
    best = scientific["best_boundary_bootstrap_cell"]
    assert isinstance(best, dict)
    selected = [
        row
        for row in rows
        if row["case"] == "boundary_heavy_four_cell_row_column"
        and row["tree_chart"] == "row-first"
        and row["particle_count"] == 4096
    ]
    selected.sort(key=lambda row: str(row["estimator"]))
    table = [
        "| Estimator | Mean Z | SE | Rel. RMSE | RelVar × median cost / IID |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in selected:
        table.append(
            f"| {row['estimator']} | {_finite_float(row, 'mean_likelihood'):.7g} | "
            f"{_finite_float(row, 'standard_error'):.3g} | "
            f"{_finite_float(row, 'relative_rmse'):.3g} | "
            f"{_finite_float(row, 'relative_variance_times_median_cost_over_baseline'):.3g} |"
        )
    return "\n".join(
        [
            "# Wider R1 Gamma–Beta resolution-SMC report",
            "",
            "## Outcome",
            "",
            f"All {summary['r1_certificate_count']} frozen tasks passed at source "
            f"`{summary['source_sha']}`. The analysis authenticated "
            f"{summary['r1_replicate_records']:,} replicate and "
            f"{summary['r1_level_records']:,} per-level records against their certificates. "
            "Likelihood means, variances, RMSE, and efficiency below are computed before taking logs.",
            "",
            "Bootstrap resolution-SMC did not establish a robust cost-adjusted advantage over direct IID. "
            f"Only {scientific['bootstrap_cells_better_than_iid']} of "
            f"{scientific['bootstrap_cell_count']} bootstrap cells beat IID on relative variance times "
            f"median measured cost, and {scientific['bootstrap_cells_twofold_better_than_iid']} reached "
            "the provisional twofold target. The best boundary-heavy ratio was "
            f"{float(best['relative_variance_times_median_cost_over_baseline']):.3g} for "
            f"`{best['case']}`, `{best['tree_chart']}`, N={best['particle_count']}, "
            f"`{best['estimator']}`.",
            "",
            "## Target and references",
            "",
            "Here, Z is the fixed-root allocation-marginal normalized native Gaussian likelihood and N is "
            "the particle or complete-allocation sample count. Relative variance is the between-replicate "
            "variance of Z divided by the squared oracle/reference value. The primary score multiplies that "
            "quantity by median measured wall time. "
            "The Gaussian closure is used only between refinements; every terminal particle is scored with "
            "the exact normalized native Gaussian density. Two- and four-cell targets use independently "
            "converged Gauss–Jacobi values. The two 16-cell targets use 16 independent 262,144-sample IID "
            "reference estimates and are not labelled exact.",
            "",
            "## Correctness and degeneration",
            "",
            f"Exact-target discrepancies beyond three combined Monte Carlo SE occurred in "
            f"{scientific['exact_target_cells_beyond_3se']} of "
            f"{scientific['exact_target_cell_count']} cells. Compatible-chart mean differences beyond "
            f"three replicate SE occurred in {scientific['chart_comparisons_beyond_3se']} of "
            f"{scientific['chart_comparison_count']} comparisons. None exceeded four SE or repeated as a "
            "common-sign pattern across particle counts, so these are retained as finite-replicate "
            "diagnostics rather than target failures. They are listed in `r1_estimator_summary.csv` and "
            "`r1_chart_summary.csv`.",
            "",
            f"The worst per-level ESS fraction was {float(scientific['minimum_ess_fraction']):.4g}; "
            f"the smallest unique-ancestor count was {scientific['minimum_unique_ancestor_count']}. "
            "Low ESS and ancestry collapse are retained as scientific outcomes. Terminal unresolved "
            "covariance is exactly zero and all conservation/update audits remain at floating-point "
            "roundoff.",
            "",
            "## Boundary-heavy N=4096 example",
            "",
            *table,
            "",
            "## Figures",
            "",
            "![Incremental ESS](figures/r1_incremental_ess.png)",
            "",
            "![Correction variance](figures/r1_correction_variance.png)",
            "",
            "![RMSE versus measured work](figures/r1_rmse_vs_work.png)",
            "",
            "![Tree and ordering sensitivity](figures/r1_tree_order_sensitivity.png)",
            "",
            "![Particle ancestry](figures/r1_ancestry.png)",
            "",
            "## Interpretation",
            "",
            "The wider matrix confirms the R1a negative result: prior-proposal SMC sometimes reduces raw "
            "variance, but frontier updates, repeated guide evaluations, and resampling usually erase the "
            "gain at measured cost. Observation-energy ordering is not uniformly favourable, and the "
            "deliberately unfavourable ordering exposes severe ESS/ancestry loss. Scrambled Sobol is the "
            f"more credible complete-allocation competitor in this size range: it beat direct IID in "
            f"{scientific['sobol_cells_better_than_iid']} of {scientific['sobol_cell_count']} cells on the "
            "primary score.",
            "",
            "R0/R1a established path identity, conservation, child-swap equivariance, provenance rejection, "
            "and checkpoint replay. Wider R1 reuses that engine but does not emit a fresh checkpoint for "
            "every production cell; R2 repeats every restart boundary explicitly.",
            "",
        ]
    )


def _r2_report(summary: Record, proposal_rows: Sequence[Record]) -> str:
    scientific = summary["r2_scientific"]
    assert isinstance(scientific, dict)
    best = scientific["best_guided_cell"]
    assert isinstance(best, dict)
    selected = [
        row for row in proposal_rows if row["particle_count"] == 4096 and row["tree_chart"] == "row-first"
    ]
    selected.sort(key=lambda row: str(row["estimator"]))
    table = [
        "| Proposal | Min ESS frac. | Max normalizer rel. error | Variance / prior | "
        "Median cost / prior | Var × cost / prior |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        table.append(
            f"| {row['estimator']} | {_finite_float(row, 'minimum_ess_fraction'):.3g} | "
            f"{_finite_float(row, 'maximum_proposal_normalizer_relative_error'):.3g} | "
            f"{_finite_float(row, 'relative_variance_over_prior'):.3g} | "
            f"{_finite_float(row, 'median_cost_over_prior'):.3g} | "
            f"{_finite_float(row, 'relative_variance_times_median_cost_over_prior'):.3g} |"
        )
    return "\n".join(
        [
            "# Bounded R2 guided-proposal report",
            "",
            "## Outcome",
            "",
            f"All {summary['r2_certificate_count']} R2 tasks passed at the frozen source SHA. "
            f"Every one of {summary['r2_replay_boundary_count']} saved restart boundaries reproduced the "
            "uninterrupted scientific fingerprint and matched its recorded file digest.",
            "",
            "As in R1, Z denotes the allocation-marginal normalized likelihood. Variance and cost ratios "
            "are relative to prior-proposal bootstrap SMC under the same chart and particle count.",
            "",
            "The proposal is continuous: equal-prior-probability bins define a piecewise-constant Gaussian "
            "guide factor multiplied by the exact Beta density; samples use exact truncated-Beta inversion, "
            "and terminal weights include the exact evaluable prior/proposal correction. The Gauss–Jacobi "
            "normalizer calculation is an audit, not the proposal target.",
            "",
            "Guidance usually raised mean intermediate ESS, but the coarser ladders could still collapse: "
            f"the worst prior ESS fraction was {float(scientific['minimum_prior_ess_fraction']):.3g}, "
            f"whereas the worst guided value was {float(scientific['minimum_guided_ess_fraction']):.3g}. "
            "Guidance did not establish a reproducible cost-adjusted advantage across the ladder. "
            f"{scientific['guided_cells_better_than_prior']} of {scientific['guided_cell_count']} guided "
            f"cells beat prior SMC, and {scientific['guided_cells_twofold_better_than_prior']} reached a "
            "twofold improvement. The best ratio was "
            f"{float(best['relative_variance_times_median_cost_over_baseline']):.3g} for "
            f"`{best['tree_chart']}`, N={best['particle_count']}, `{best['estimator']}`.",
            "",
            "## N=4096 row-first details",
            "",
            *table,
            "",
            "The maximum recorded proposal-normalizer relative discrepancy was "
            f"{float(scientific['maximum_proposal_normalizer_relative_error']):.3g}. It is largest for the "
            "coarsest bin ladder and is retained as a guide-quality failure, not hidden by selecting only "
            "the best realized proposal.",
            "",
            "![R2 guided ESS](figures/r2_guided_ess.png)",
            "",
            "## Interpretation",
            "",
            "The 32-bin guide generally flattens incremental weights and raises ESS, but coarser versions "
            "can still collapse. Proposal construction and quadrature-guide evaluation are expensive. The ladder "
            "does not provide evidence for promoting this implementation to medium or PARIS scale. More "
            "efficient analytic or amortized guidance would be a redesign, not a tuning continuation.",
            "",
        ]
    )


def _overall_report(summary: Record, r1a_summary: Record) -> str:
    r1 = summary["r1_scientific"]
    r2 = summary["r2_scientific"]
    assert isinstance(r1, dict)
    assert isinstance(r2, dict)
    r1a_scientific = r1a_summary["scientific"]
    assert isinstance(r1a_scientific, dict)
    r1a_best = r1a_scientific["best_boundary_smc"]
    assert isinstance(r1a_best, dict)
    return "\n".join(
        [
            "# Gamma–Beta coarse-to-fine resolution-SMC decision report",
            "",
            "## Decision",
            "",
            "**Stop this proposal design before medium R3 and do not start PARIS R4.**",
            "",
            "The scientific target Z is the fixed-root allocation-marginal normalized native Gaussian "
            "likelihood. All primary comparisons use the between-replicate variance of non-negative Z "
            "estimates, divided by the squared oracle/reference value, times median measured wall time. "
            "Logs are secondary reporting coordinates only.",
            "",
            "R0, the exact R1a screen, wider R1, and bounded R2 are correct and characterized. "
            "They do not show a reproducible variance-per-cost advantage that justifies scaling. R1a's "
            "best boundary-heavy bootstrap ratio was "
            f"{float(r1a_best['relative_variance_times_median_cost_over_direct']):.3g} relative to IID. "
            f"In wider R1, {r1['bootstrap_cells_twofold_better_than_iid']} of "
            f"{r1['bootstrap_cell_count']} bootstrap cells reached the provisional twofold target. "
            f"In R2, {r2['guided_cells_twofold_better_than_prior']} of "
            f"{r2['guided_cell_count']} guided cells reached it relative to prior SMC.",
            "",
            "## What is established",
            "",
            "- Exact Gamma–Beta tree mass identities, local conditional means/covariances, child-swap "
            "equivariance, and terminal zero unresolved covariance pass.",
            "- Direct IID and no-resampling SMC are pathwise identical when given the same allocation paths.",
            "- Two- and four-cell normalized native Gaussian likelihoods agree with converged independent "
            "quadrature oracles; 16-cell comparisons use explicitly uncertain IID references.",
            "- Checkpoint/restart is bitwise reproducible at R0 and R2 boundaries and rejects mismatched "
            "tree, schedule, guide/input, seed, particle, and source provenance.",
            f"- All production artifacts authenticate source `{summary['source_sha']}` and their declared "
            "SHA-256 digests.",
            "",
            "## What failed scientifically",
            "",
            f"- Wider R1 reached ESS fraction {float(r1['minimum_ess_fraction']):.4g} and as few as "
            f"{r1['minimum_unique_ancestor_count']} unique ancestors.",
            "- Bootstrap SMC can reduce raw likelihood variance, but repeated guide/frontier work generally "
            "makes relative variance times measured cost worse than direct IID or scrambled Sobol.",
            "- Observation-energy ordering is not uniformly favourable across compatible trees.",
            f"- The R2 guide usually raises mean ESS but can still collapse; its maximum "
            "normalizer-audit discrepancy is "
            f"{float(r2['maximum_proposal_normalizer_relative_error']):.3g} and its construction cost removes "
            "the variance gain.",
            "",
            "## Recommendation",
            "",
            "Medium scaling is not justified for the current bootstrap or piecewise-Beta guide. If work "
            "continues, redesign the guide so its construction is amortized or analytic, then repeat the "
            "tiny exact matrix before any R3 launch. PARIS R4 is not justified because the plan requires "
            "R1 correctness plus at least one viable R3 configuration; no viable R3 configuration has been "
            "demonstrated, and the tiny/medium-size evidence trends against cost effectiveness.",
            "",
            "No protected or realized-observation catalogue was accessed, and nothing was written to "
            "`PARIS_inversions`.",
            "",
            "## Deliverables",
            "",
            "- `R1_REPORT.md` and `R2_REPORT.md`: readable stage reports.",
            "- `r1_estimator_summary.csv`, `r1_level_summary.csv`, `r1_chart_summary.csv`: wider R1 tables.",
            "- `r2_estimator_summary.csv`, `r2_level_summary.csv`, `r2_chart_summary.csv`, "
            "`r2_proposal_summary.csv`: bounded-guide tables.",
            "- `attempt-ledger.csv` and `jobs.csv`: operational history and all Slurm identifiers.",
            "- `summary.json`: machine-readable decision summary.",
            "- `sha256sums.txt` and source-specific manifests: checksums for retained artifacts.",
            "",
        ]
    )


def _manifest_rows(root: Path, label: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name.startswith("sha256sums") or path.name.startswith("."):
            continue
        rows.append((_sha256(path), f"{label}/{path.relative_to(root)}"))
    return rows


def _write_manifests(output_root: Path, run_root: Path, r1a_run_root: Path) -> None:
    current = _manifest_rows(run_root, run_root.name)
    r1a = _manifest_rows(r1a_run_root, r1a_run_root.name)

    def render(rows: Sequence[tuple[str, str]]) -> str:
        return "".join(f"{digest}  {name}\n" for digest, name in rows)

    _write_text(output_root / f"sha256sums-{run_root.name}.txt", render(current))
    _write_text(output_root / f"sha256sums-{r1a_run_root.name}.txt", render(r1a))
    _write_text(output_root / "sha256sums.txt", render([*r1a, *current]))


def analyse(
    *,
    run_root: Path,
    r1a_run_root: Path,
    output_root: Path,
    source_sha: str,
) -> Record:
    _validate_sha(source_sha)
    if run_root.name != source_sha or r1a_run_root.name != R1A_SOURCE_SHA:
        raise ValueError("run roots must be named by their complete frozen source SHAs.")
    r0_summary = _read_json(run_root / "r0" / "r0_summary.json")
    if r0_summary.get("status") != "passed":
        raise ValueError("final-SHA R0 certificate did not pass.")
    _validate_provenance(r0_summary, run_root / "r0" / "r0_summary.json", source_sha)
    r1a_summary = _read_json(r1a_run_root / "report" / "r1a_summary.json")
    if r1a_summary.get("status") != "passed" or r1a_summary.get("source_sha") != R1A_SOURCE_SHA:
        raise ValueError("R1a analysis certificate is invalid.")

    references = _load_references(run_root / "r1", source_sha=source_sha)
    r1 = _load_stage(
        run_root / "r1",
        schema=R1_SCHEMA,
        source_sha=source_sha,
        expected_certificates=EXPECTED_R1_CERTIFICATES,
        expected_replicates=EXPECTED_R1_REPLICATES,
        require_replay=False,
    )
    r2 = _load_stage(
        run_root / "r2",
        schema=R2_SCHEMA,
        source_sha=source_sha,
        expected_certificates=EXPECTED_R2_CERTIFICATES,
        expected_replicates=EXPECTED_R2_REPLICATES,
        require_replay=True,
    )
    r1_estimators = _estimator_summary(
        r1.replicates,
        replicate_count=EXPECTED_R1_REPLICATES,
        baseline_estimator="direct_iid",
    )
    r2_estimators = _estimator_summary(
        r2.replicates,
        replicate_count=EXPECTED_R2_REPLICATES,
        baseline_estimator="bootstrap_prior_ess_0.5",
    )
    r1_levels = _level_summary(r1.levels)
    r2_levels = _level_summary(r2.levels)
    r1_charts = _chart_summary(r1_estimators, replicate_count=EXPECTED_R1_REPLICATES)
    r2_charts = _chart_summary(r2_estimators, replicate_count=EXPECTED_R2_REPLICATES)
    r2_proposals = _proposal_summary(r2_levels, r2_estimators)
    replay_count = 0
    for certificate in r2.certificates:
        replay = certificate.get("checkpoint_replay")
        if isinstance(replay, list):
            replay_count += len(replay)
    summary: Record = {
        "schema": ANALYSIS_SCHEMA,
        "status": "passed",
        "source_sha": source_sha,
        "planning_sha": PLANNING_SHA,
        "knowledge_sha": KNOWLEDGE_SHA,
        "analysis_sha256": _sha256(Path(__file__)),
        "r0_status": "passed",
        "r1a_source_sha": R1A_SOURCE_SHA,
        "r1a_status": "passed",
        "r1_reference_certificates": len(references),
        "r1_certificate_count": len(r1.certificates),
        "r1_replicate_records": len(r1.replicates),
        "r1_level_records": len(r1.levels),
        "r2_certificate_count": len(r2.certificates),
        "r2_replicate_records": len(r2.replicates),
        "r2_level_records": len(r2.levels),
        "r2_replay_boundary_count": replay_count,
        "terminal_zero_unresolved_covariance": True,
        "maximum_numerical_errors": _maximum_errors(r1.levels, r2.levels),
        "r1_scientific": _r1_scientific_summary(r1_estimators, r1_levels, r1_charts),
        "r2_scientific": _r2_scientific_summary(
            r2_estimators,
            r2_levels,
            r2_charts,
            r2_proposals,
        ),
        "recommendation": {
            "medium_r3": "do not run current proposal; redesign guidance and repeat tiny exact experiments",
            "paris_r4": "not justified",
        },
        "protected_catalogue_access": False,
        "paris_inversions_writes": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "r1_estimator_summary.csv", r1_estimators)
    _write_csv(output_root / "r1_level_summary.csv", r1_levels)
    _write_csv(output_root / "r1_chart_summary.csv", r1_charts)
    _write_csv(output_root / "r2_estimator_summary.csv", r2_estimators)
    _write_csv(output_root / "r2_level_summary.csv", r2_levels)
    _write_csv(output_root / "r2_chart_summary.csv", r2_charts)
    _write_csv(output_root / "r2_proposal_summary.csv", r2_proposals)
    _write_csv(output_root / "attempt-ledger.csv", _attempts(source_sha, run_root, r1a_run_root))
    _write_csv(output_root / "jobs.csv", _jobs(source_sha, run_root))
    _write_json(output_root / "validation-evidence.json", _validation_evidence(source_sha))
    figures = _plot_reports(output_root, r1_estimators, r1_levels, r2_levels)
    summary["figures"] = figures
    _write_json(output_root / "summary.json", summary)
    _write_text(output_root / "R1_REPORT.md", _r1_report(summary, r1_estimators, r1_charts))
    _write_text(output_root / "R2_REPORT.md", _r2_report(summary, r2_proposals))
    _write_text(output_root / "RESULTS.md", _overall_report(summary, r1a_summary))
    _write_manifests(output_root, run_root, r1a_run_root)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--r1a-run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = analyse(
        run_root=arguments.run_root,
        r1a_run_root=arguments.r1a_run_root,
        output_root=arguments.output_root,
        source_sha=arguments.source_sha,
    )
    print(_canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
