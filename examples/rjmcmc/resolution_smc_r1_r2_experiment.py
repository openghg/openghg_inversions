#!/usr/bin/env python3
"""Run the wider synthetic R1 and bounded guided-proposal R2 experiments.

This driver accepts only source-embedded synthetic inputs.  R1 extends the
exact R1a screen to the planned two-, four-, and sixteen-cell cases, valid
tree charts, observation-energy schedules, ESS thresholds, and independently
scrambled Sobol baselines.  Sixteen-cell targets use separately replicated
large-IID references and are never labelled exact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Literal, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from openghg_inversions.experimental.rjmcmc.aggregation_error import (
    FourCellAggregationOracle,
    TwoCellAggregationOracle,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_resolution_smc import (
    GaussianGuideSpecification,
    ResolutionSMCCheckpoint,
    ResolutionSMCConfig,
    ResolutionSMCResult,
    ResolutionSchedule,
    ResolutionTree,
    breadth_first_schedule,
    direct_iid_likelihood_average,
    draw_prior_allocation_paths,
    draw_scrambled_sobol_allocation_paths,
    parent_first_priority_schedule,
    run_resolution_smc,
)

FloatArray: TypeAlias = NDArray[np.float64]
Family = Literal["two_cell", "four_cell", "sixteen_cell"]
OracleKind = Literal["gauss_jacobi", "large_iid_reference"]
TreeChart = Literal["balanced", "row-first", "column-first", "chain"]
NestedChart: TypeAlias = int | tuple["NestedChart", "NestedChart"]

SCHEMA = "rjmcmc-resolution-smc-r1-v1"
REFERENCE_SCHEMA = "rjmcmc-resolution-smc-r1-reference-v1"
R2_SCHEMA = "rjmcmc-resolution-smc-r2-v1"
SOURCE_PLANNING_SHA = "3a4c80d17bf09dd70d69d1bef306cb888002fe54"
KNOWLEDGE_SHA = "a84bde45ccbbe42e2c400b79285b39f12b6cbcdd"
ROOT_TOTAL = 1.0
PARTICLE_COUNTS = (64, 256, 1_024, 4_096)
REPLICATE_COUNT = 32
REFERENCE_SAMPLE_COUNT = 262_144
REFERENCE_REPLICATE_COUNT = 16
R2_PARTICLE_COUNTS = (256, 1_024, 4_096)
R2_REPLICATE_COUNT = 16
R2_PROPOSAL_BIN_COUNTS = (8, 16, 32)
R2_PROPOSAL_AUDIT_ORDER = 64
BASE_SEED = 20260731


@dataclass(frozen=True)
class WideCase:
    """One fixed synthetic wider-R1 target."""

    name: str
    family: Family
    shapes: tuple[float, ...]
    design: tuple[tuple[float, ...], ...]
    observation: tuple[float, ...]
    noise_sd: tuple[float, ...]
    oracle_kind: OracleKind
    oracle_orders: tuple[int, ...]


def _design16() -> tuple[tuple[float, ...], ...]:
    columns = np.arange(16, dtype=np.float64)
    rows = np.stack(
        (
            0.75 + 0.30 * np.cos((columns + 0.5) * math.pi / 8.0),
            0.15 + 0.55 * np.sin((columns + 1.0) * math.pi / 9.0),
            ((columns % 4.0) - 1.5) / 2.5,
            ((columns // 4.0) - 1.5) / 2.2,
            np.cos((columns + 1.0) * math.pi / 5.0),
            np.sin((columns + 0.25) * math.pi / 6.0),
            ((columns * 5.0) % 13.0 - 6.0) / 7.0,
            ((columns * 7.0) % 17.0 - 8.0) / 8.0,
        )
    )
    return tuple(tuple(float(value) for value in row) for row in rows)


NEAR_FOUR_DESIGN = (
    (1.00, 0.82, 0.45, 0.30),
    (0.15, 0.42, 0.90, 1.10),
    (-0.50, -0.10, 0.35, 0.55),
    (0.70, 0.62, 0.78, 0.85),
)
ROW_COLUMN_DESIGN = (
    (2.00, 0.00, 0.10, 0.00),
    (0.00, 1.70, 0.00, 0.10),
    (0.05, 0.00, 1.90, 0.00),
    (0.00, 0.10, 0.00, 2.10),
)
GENERIC_BOUNDARY_DESIGN = (
    (1.80, 0.10, 0.50, -0.20),
    (-0.40, 1.20, 0.20, 0.85),
    (0.80, -0.30, 1.45, 0.10),
    (0.20, 0.35, -0.15, 1.60),
)

CASES = (
    WideCase(
        "near_gaussian_two_cell",
        "two_cell",
        (45.0, 55.0),
        ((1.0, 0.7), (0.2, 1.1), (-0.5, 0.3)),
        (0.93, 0.71, -0.08),
        (0.42, 0.55, 0.48),
        "gauss_jacobi",
        (48, 64, 96),
    ),
    WideCase(
        "skewed_g1_two_cell",
        "two_cell",
        (0.35, 4.0),
        ((1.8, 0.1), (-0.4, 1.2), (0.8, -0.3)),
        (0.44, 0.91, -0.08),
        (0.25, 0.32, 0.28),
        "gauss_jacobi",
        (64, 96, 128),
    ),
    WideCase(
        "boundary_heavy_two_cell",
        "two_cell",
        (0.12, 0.18),
        ((2.0, 0.0), (0.0, 1.7), (1.0, -1.0)),
        (1.75, 0.08, 0.94),
        (0.12, 0.14, 0.13),
        "gauss_jacobi",
        (96, 128, 192),
    ),
    WideCase(
        "balanced_four_cell_generic_contrasts",
        "four_cell",
        (40.0, 35.0, 45.0, 30.0),
        NEAR_FOUR_DESIGN,
        (0.72, 0.64, 0.04, 0.79),
        (0.40, 0.48, 0.45, 0.52),
        "gauss_jacobi",
        (32, 48, 64),
    ),
    WideCase(
        "balanced_four_cell_row_column",
        "four_cell",
        (40.0, 35.0, 45.0, 30.0),
        ROW_COLUMN_DESIGN,
        (1.03, 0.31, 0.47, 0.26),
        (0.30, 0.32, 0.31, 0.34),
        "gauss_jacobi",
        (32, 48, 64),
    ),
    WideCase(
        "boundary_heavy_four_cell_row_column",
        "four_cell",
        (0.15, 0.18, 0.20, 0.12),
        ROW_COLUMN_DESIGN,
        (1.62, 0.08, 0.13, 0.06),
        (0.12, 0.14, 0.13, 0.15),
        "gauss_jacobi",
        (48, 64, 96),
    ),
    WideCase(
        "boundary_heavy_four_cell_generic_contrasts",
        "four_cell",
        (0.15, 0.18, 0.20, 0.12),
        GENERIC_BOUNDARY_DESIGN,
        (0.23, 0.83, 0.36, 1.12),
        (0.22, 0.30, 0.26, 0.34),
        "gauss_jacobi",
        (48, 64, 96),
    ),
    WideCase(
        "balanced_sixteen_cell",
        "sixteen_cell",
        tuple(6.0 + float(index % 4) for index in range(16)),
        _design16(),
        (0.71, 0.49, -0.07, 0.03, 0.11, 0.28, -0.04, 0.06),
        (0.34, 0.38, 0.31, 0.36, 0.40, 0.35, 0.39, 0.37),
        "large_iid_reference",
        (),
    ),
    WideCase(
        "skewed_sixteen_cell",
        "sixteen_cell",
        (
            0.22,
            0.35,
            0.55,
            0.90,
            1.40,
            2.20,
            3.50,
            5.50,
            0.28,
            0.45,
            0.72,
            1.15,
            1.80,
            2.80,
            4.40,
            7.00,
        ),
        _design16(),
        (0.82, 0.41, -0.16, 0.11, 0.24, 0.19, -0.10, 0.14),
        (0.22, 0.27, 0.24, 0.26, 0.30, 0.25, 0.29, 0.28),
        "large_iid_reference",
        (),
    ),
)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def protocol_payload() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "source_planning_sha": SOURCE_PLANNING_SHA,
        "knowledge_sha": KNOWLEDGE_SHA,
        "root_total": ROOT_TOTAL,
        "particle_counts": PARTICLE_COUNTS,
        "replicate_count": REPLICATE_COUNT,
        "reference_sample_count": REFERENCE_SAMPLE_COUNT,
        "reference_replicate_count": REFERENCE_REPLICATE_COUNT,
        "base_seed": BASE_SEED,
        "cases": [asdict(case) for case in CASES],
        "estimators": (
            "direct_iid",
            "scrambled_sobol",
            "bootstrap_breadth_ess_0.25",
            "bootstrap_breadth_ess_0.5",
            "bootstrap_energy_ess_0.25",
            "bootstrap_energy_ess_0.5",
            "bootstrap_unfavorable_ess_0.5",
        ),
        "tree_charts": "balanced; row-first, column-first and chain where applicable",
        "comparison_scale": "linear normalized likelihood",
    }


PROTOCOL_SHA256 = _sha256_json(protocol_payload())
R2_PROTOCOL_SHA256 = _sha256_json(
    {
        "schema": R2_SCHEMA,
        "r1_protocol_sha256": PROTOCOL_SHA256,
        "case": "boundary_heavy_four_cell_row_column",
        "particle_counts": R2_PARTICLE_COUNTS,
        "replicate_count": R2_REPLICATE_COUNT,
        "tree_charts": ("row-first", "column-first"),
        "schedule": "one-node favorable observation-energy",
        "baseline": "prior bootstrap ESS 0.5",
        "proposal": (
            "equal-prior-mass piecewise-constant guide times exact Beta density; "
            "exact truncated-Beta sampling and density correction"
        ),
        "proposal_bin_counts": R2_PROPOSAL_BIN_COUNTS,
        "proposal_audit_order": R2_PROPOSAL_AUDIT_ORDER,
    }
)


def _seed64(*parts: object) -> int:
    digest = hashlib.sha256(str(BASE_SEED).encode("ascii"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(str(part).encode("ascii"))
    return int.from_bytes(digest.digest()[:8], "little")


def _seed128(*parts: object) -> int:
    digest = hashlib.sha256(str(BASE_SEED).encode("ascii"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(str(part).encode("ascii"))
    return int.from_bytes(digest.digest()[:16], "little")


def _git_output(arguments: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _authenticate_source(source_sha: str) -> dict[str, object]:
    if len(source_sha) != 40 or any(character not in "0123456789abcdef" for character in source_sha):
        raise ValueError("source SHA must be a complete lowercase Git SHA.")
    head = _git_output(("rev-parse", "HEAD"))
    if head != source_sha:
        raise RuntimeError("worktree HEAD does not match the declared source SHA.")
    if _git_output(("status", "--porcelain")):
        raise RuntimeError("scientific worktree is not clean.")
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if symbolic.returncode == 0:
        raise RuntimeError("scientific execution requires a detached worktree.")
    return {
        "source_sha": source_sha,
        "head_sha": head,
        "clean": True,
        "detached": True,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": PROTOCOL_SHA256,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
    temporary.replace(path)


def _record_float(row: dict[str, object], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"record field {name!r} is not numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"record field {name!r} is non-finite.")
    return result


def _nested_balanced(ids: tuple[int, ...]) -> NestedChart:
    if len(ids) == 1:
        return ids[0]
    middle = len(ids) // 2
    return (_nested_balanced(ids[:middle]), _nested_balanced(ids[middle:]))


def _nested_chain(ids: tuple[int, ...]) -> NestedChart:
    if len(ids) == 1:
        return ids[0]
    return (ids[0], _nested_chain(ids[1:]))


def _charts(case: WideCase) -> tuple[TreeChart, ...]:
    if case.family == "two_cell":
        return ("balanced",)
    if case.family == "four_cell":
        return ("row-first", "column-first", "chain")
    return ("balanced", "chain")


def _tree(case: WideCase, chart: TreeChart) -> ResolutionTree:
    ids = tuple(range(len(case.shapes)))
    if chart == "balanced":
        nested = _nested_balanced(ids)
    elif chart == "row-first" and len(ids) == 4:
        nested = ((0, 1), (2, 3))
    elif chart == "column-first" and len(ids) == 4:
        nested = ((0, 2), (1, 3))
    elif chart == "chain":
        nested = _nested_chain(ids)
    else:
        raise ValueError(f"chart {chart!r} is invalid for {case.name}.")
    return ResolutionTree.from_nested_chart(ids, case.shapes, nested)


def _guide(case: WideCase) -> GaussianGuideSpecification:
    return GaussianGuideSpecification.build(
        case.observation,
        case.design,
        case.noise_sd,
    )


def _schedules(
    tree: ResolutionTree,
    guide: GaussianGuideSpecification,
    chart: TreeChart,
) -> dict[str, ResolutionSchedule]:
    return {
        "breadth": breadth_first_schedule(tree, name=f"{chart}-breadth-first"),
        "energy": parent_first_priority_schedule(
            tree,
            guide,
            root_total=ROOT_TOTAL,
            favorable=True,
            name=f"{chart}-observation-energy",
        ),
        "unfavorable": parent_first_priority_schedule(
            tree,
            guide,
            root_total=ROOT_TOTAL,
            favorable=False,
            name=f"{chart}-unfavorable-observation-energy",
        ),
    }


def _exact_oracle(case: WideCase) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    if case.family == "two_cell":
        for order in case.oracle_orders:
            oracle = TwoCellAggregationOracle(
                gamma_shape=float(sum(case.shapes)),
                gamma_rate=float(sum(case.shapes)),
                beta_first_shape=case.shapes[0],
                beta_second_shape=case.shapes[1],
                fraction_order=order,
            )
            log_value = float(
                oracle.coarse_conditional_log_likelihood(
                    ROOT_TOTAL,
                    case.observation,
                    case.design,
                    case.noise_sd,
                )
            )
            rows.append({"order": order, "chart": "beta", "log_likelihood": log_value})
    elif case.family == "four_cell":
        for order in case.oracle_orders:
            oracle4 = FourCellAggregationOracle(
                np.asarray(case.shapes),
                gamma_rate=float(sum(case.shapes)),
                fraction_order=order,
                chunk_size=16_384,
            )
            for chart in ("row-first", "column-first"):
                log_value = float(
                    oracle4.conditional_log_likelihood(
                        ROOT_TOTAL,
                        case.observation,
                        case.design,
                        case.noise_sd,
                        tiling="root",
                        root_chart=chart,
                    )
                )
                rows.append({"order": order, "chart": chart, "log_likelihood": log_value})
    else:
        raise ValueError("sixteen-cell cases require a replicated reference.")
    final_order = case.oracle_orders[-1]
    final = np.asarray([_record_float(row, "log_likelihood") for row in rows if row["order"] == final_order])
    previous_order = case.oracle_orders[-2]
    previous = np.asarray(
        [_record_float(row, "log_likelihood") for row in rows if row["order"] == previous_order]
    )
    if np.ptp(final) > 2.0e-10 or abs(float(np.mean(final) - np.mean(previous))) > 2.0e-10:
        raise RuntimeError(f"{case.name} quadrature has not converged.")
    for row in rows:
        row["likelihood"] = math.exp(_record_float(row, "log_likelihood"))
        row["case"] = case.name
    return math.exp(float(np.mean(final))), rows


def reference_tasks() -> tuple[WideCase, ...]:
    return tuple(case for case in CASES if case.oracle_kind == "large_iid_reference")


def run_reference_task(
    *,
    task_index: int,
    output_root: Path,
    source_sha: str,
) -> dict[str, object]:
    provenance = _authenticate_source(source_sha)
    tasks = reference_tasks()
    if not 0 <= task_index < len(tasks):
        raise ValueError("reference task index is out of range.")
    case = tasks[task_index]
    tree = _tree(case, "balanced")
    schedule = breadth_first_schedule(tree)
    guide = _guide(case)
    estimates: list[float] = []
    within_variances: list[float] = []
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for replicate in range(REFERENCE_REPLICATE_COUNT):
        seed = _seed128(case.name, "reference", replicate)
        config = ResolutionSMCConfig(
            REFERENCE_SAMPLE_COUNT,
            seed=seed,
            resampling_policy="never",
        )
        result = direct_iid_likelihood_average(
            tree,
            schedule,
            guide,
            root_total=ROOT_TOTAL,
            config=config,
        )
        linear = np.exp(result.terminal_log_likelihoods)
        estimate = result.likelihood
        estimates.append(estimate)
        within_variances.append(float(np.var(linear, ddof=1)))
        rows.append(
            {
                "schema": REFERENCE_SCHEMA,
                "protocol_sha256": PROTOCOL_SHA256,
                "case": case.name,
                "replicate": replicate,
                "sample_count": REFERENCE_SAMPLE_COUNT,
                "seed": seed,
                "likelihood": estimate,
                "sample_variance": within_variances[-1],
                "elapsed_seconds": result.elapsed_seconds,
                "peak_rss_bytes": result.peak_rss_bytes,
            }
        )
    estimate_array = np.asarray(estimates)
    mean = float(np.mean(estimate_array))
    replicate_variance = float(np.var(estimate_array, ddof=1))
    path = output_root / "references" / f"{case.name}.jsonl"
    _write_jsonl(path, rows)
    certificate = {
        "schema": REFERENCE_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "case": case.name,
        "task_index": task_index,
        "sample_count": REFERENCE_SAMPLE_COUNT,
        "replicate_count": REFERENCE_REPLICATE_COUNT,
        "mean_likelihood": mean,
        "replicate_standard_error": math.sqrt(replicate_variance / REFERENCE_REPLICATE_COUNT),
        "pooled_iid_standard_error": math.sqrt(
            float(np.mean(within_variances)) / (REFERENCE_SAMPLE_COUNT * REFERENCE_REPLICATE_COUNT)
        ),
        "label": "replicated large-IID reference, not exact",
        "records_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "provenance": provenance,
        "status": "passed",
    }
    _write_json(output_root / "reference-certificates" / f"{case.name}.json", certificate)
    return certificate


def _oracle(
    case: WideCase,
    *,
    output_root: Path,
    source_sha: str,
) -> tuple[float, float, str, list[dict[str, object]]]:
    if case.oracle_kind == "gauss_jacobi":
        value, convergence = _exact_oracle(case)
        return value, 0.0, "converged Gauss-Jacobi quadrature", convergence
    certificate_path = output_root / "reference-certificates" / f"{case.name}.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    provenance = certificate.get("provenance", {})
    if (
        certificate.get("status") != "passed"
        or certificate.get("protocol_sha256") != PROTOCOL_SHA256
        or not isinstance(provenance, dict)
        or provenance.get("source_sha") != source_sha
    ):
        raise RuntimeError(f"{case.name} reference certificate is invalid.")
    records_path = output_root / "references" / f"{case.name}.jsonl"
    if hashlib.sha256(records_path.read_bytes()).hexdigest() != certificate["records_sha256"]:
        raise RuntimeError(f"{case.name} reference records fail their digest.")
    return (
        float(certificate["mean_likelihood"]),
        float(certificate["replicate_standard_error"]),
        str(certificate["label"]),
        [],
    )


def _array_sha256(values: NDArray[np.generic]) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(
        _canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)}).encode("ascii")
    )
    digest.update(array.tobytes())
    return digest.hexdigest()


def _direct_record(
    *,
    case: WideCase,
    chart: TreeChart,
    schedule: ResolutionSchedule,
    particle_count: int,
    replicate: int,
    estimator: str,
    seed: int,
    oracle: float,
    oracle_se: float,
    oracle_label: str,
    path_seconds: float,
    paths: FloatArray,
    result_likelihood: float,
    result_log_likelihood: float,
    estimator_seconds: float,
    peak_rss_bytes: int,
    terminal_logs: FloatArray,
) -> dict[str, object]:
    coordinate_work = particle_count * paths.shape[1]
    return {
        "schema": SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "case": case.name,
        "family": case.family,
        "particle_count": particle_count,
        "replicate": replicate,
        "estimator": estimator,
        "tree_chart": chart,
        "schedule": schedule.name,
        "seed": seed,
        "oracle_likelihood": oracle,
        "oracle_standard_error": oracle_se,
        "oracle_label": oracle_label,
        "likelihood": result_likelihood,
        "log_likelihood": result_log_likelihood,
        "linear_error": result_likelihood - oracle,
        "relative_error": result_likelihood / oracle - 1.0,
        "path_generation_seconds": path_seconds,
        "estimator_seconds": estimator_seconds,
        "wall_seconds": path_seconds + estimator_seconds,
        "beta_draw_count": coordinate_work,
        "beta_endpoint_repair_count": int(
            np.count_nonzero((paths == np.nextafter(0.0, 1.0)) | (paths == np.nextafter(1.0, 0.0)))
        ),
        "forward_update_count": coordinate_work,
        "likelihood_evaluation_count": particle_count,
        "state_bytes": int(paths.nbytes + terminal_logs.nbytes),
        "peak_rss_bytes": peak_rss_bytes,
        "allocation_paths_sha256": _array_sha256(paths),
        "terminal_log_likelihoods_sha256": _array_sha256(terminal_logs),
    }


def _smc_record(
    *,
    case: WideCase,
    chart: TreeChart,
    schedule: ResolutionSchedule,
    particle_count: int,
    replicate: int,
    estimator: str,
    seed: int,
    oracle: float,
    oracle_se: float,
    oracle_label: str,
    result: ResolutionSMCResult,
    schema: str = SCHEMA,
    protocol_sha256: str = PROTOCOL_SHA256,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    diagnostics = result.diagnostics
    record = {
        "schema": schema,
        "protocol_sha256": protocol_sha256,
        "case": case.name,
        "family": case.family,
        "particle_count": particle_count,
        "replicate": replicate,
        "estimator": estimator,
        "tree_chart": chart,
        "schedule": schedule.name,
        "seed": seed,
        "oracle_likelihood": oracle,
        "oracle_standard_error": oracle_se,
        "oracle_label": oracle_label,
        "likelihood": result.likelihood,
        "log_likelihood": result.log_likelihood,
        "linear_error": result.likelihood - oracle,
        "relative_error": result.likelihood / oracle - 1.0,
        "path_generation_seconds": 0.0,
        "estimator_seconds": result.elapsed_seconds,
        "wall_seconds": result.elapsed_seconds,
        "beta_draw_count": sum(item.beta_draw_count for item in diagnostics),
        "beta_endpoint_repair_count": sum(item.beta_endpoint_repair_count for item in diagnostics),
        "forward_update_count": sum(item.forward_update_count for item in diagnostics),
        "likelihood_evaluation_count": sum(item.likelihood_evaluation_count for item in diagnostics),
        "state_bytes": max(item.state_bytes for item in diagnostics),
        "peak_rss_bytes": result.peak_rss_bytes,
        "scientific_fingerprint": result.scientific_fingerprint,
        "tree_identity": result.tree_identity,
        "schedule_identity": result.schedule_identity,
    }
    levels = [
        {
            "schema": schema,
            "protocol_sha256": protocol_sha256,
            "case": case.name,
            "particle_count": particle_count,
            "replicate": replicate,
            "estimator": estimator,
            "tree_chart": chart,
            "seed": seed,
            "level_ancestry_sha256": _array_sha256(result.ancestry[level]),
            **asdict(diagnostic),
        }
        for level, diagnostic in enumerate(diagnostics)
    ]
    return record, levels


def _run_chart_replicate(
    *,
    case: WideCase,
    chart: TreeChart,
    particle_count: int,
    replicate: int,
    oracle: float,
    oracle_se: float,
    oracle_label: str,
    source_sha: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tree = _tree(case, chart)
    guide = _guide(case)
    schedules = _schedules(tree, guide, chart)
    records: list[dict[str, object]] = []
    levels: list[dict[str, object]] = []

    for estimator, sobol in (("direct_iid", False), ("scrambled_sobol", True)):
        seed = (
            _seed64(case.name, chart, particle_count, replicate, estimator)
            if sobol
            else _seed128(case.name, chart, particle_count, replicate, estimator)
        )
        path_start = time.perf_counter()
        if sobol:
            paths = draw_scrambled_sobol_allocation_paths(
                tree,
                schedules["breadth"],
                particle_count=particle_count,
                seed=seed,
            )
        else:
            paths = draw_prior_allocation_paths(
                tree,
                schedules["breadth"],
                particle_count=particle_count,
                seed=seed,
            )
        path_seconds = time.perf_counter() - path_start
        config = ResolutionSMCConfig(
            particle_count,
            seed=seed,
            resampling_policy="never",
        )
        direct = direct_iid_likelihood_average(
            tree,
            schedules["breadth"],
            guide,
            root_total=ROOT_TOTAL,
            config=config,
            allocation_paths=paths,
        )
        records.append(
            _direct_record(
                case=case,
                chart=chart,
                schedule=schedules["breadth"],
                particle_count=particle_count,
                replicate=replicate,
                estimator=estimator,
                seed=seed,
                oracle=oracle,
                oracle_se=oracle_se,
                oracle_label=oracle_label,
                path_seconds=path_seconds,
                paths=paths,
                result_likelihood=direct.likelihood,
                result_log_likelihood=direct.log_likelihood,
                estimator_seconds=direct.elapsed_seconds,
                peak_rss_bytes=direct.peak_rss_bytes,
                terminal_logs=direct.terminal_log_likelihoods,
            )
        )

    configurations = (
        ("bootstrap_breadth_ess_0.25", "breadth", 0.25),
        ("bootstrap_breadth_ess_0.5", "breadth", 0.50),
        ("bootstrap_energy_ess_0.25", "energy", 0.25),
        ("bootstrap_energy_ess_0.5", "energy", 0.50),
        ("bootstrap_unfavorable_ess_0.5", "unfavorable", 0.50),
    )
    for estimator, schedule_name, ess_fraction in configurations:
        seed = _seed128(
            case.name,
            chart,
            particle_count,
            replicate,
            estimator,
        )
        config = ResolutionSMCConfig(
            particle_count,
            seed=seed,
            resampling_policy="ess",
            ess_fraction=ess_fraction,
        )
        result, _ = run_resolution_smc(
            tree,
            schedules[schedule_name],
            guide,
            root_total=ROOT_TOTAL,
            config=config,
            source_identity=source_sha,
        )
        assert result is not None
        record, diagnostic_rows = _smc_record(
            case=case,
            chart=chart,
            schedule=schedules[schedule_name],
            particle_count=particle_count,
            replicate=replicate,
            estimator=estimator,
            seed=seed,
            oracle=oracle,
            oracle_se=oracle_se,
            oracle_label=oracle_label,
            result=result,
        )
        records.append(record)
        levels.extend(diagnostic_rows)
    return records, levels


def r1_tasks() -> tuple[tuple[WideCase, int], ...]:
    return tuple((case, count) for case in CASES for count in PARTICLE_COUNTS)


def run_r1_task(
    *,
    task_index: int,
    output_root: Path,
    source_sha: str,
) -> dict[str, object]:
    provenance = _authenticate_source(source_sha)
    tasks = r1_tasks()
    if not 0 <= task_index < len(tasks):
        raise ValueError("R1 task index is out of range.")
    case, particle_count = tasks[task_index]
    oracle, oracle_se, oracle_label, convergence = _oracle(
        case,
        output_root=output_root,
        source_sha=source_sha,
    )
    records: list[dict[str, object]] = []
    levels: list[dict[str, object]] = []
    started = time.perf_counter()
    for replicate in range(REPLICATE_COUNT):
        for chart in _charts(case):
            chart_records, chart_levels = _run_chart_replicate(
                case=case,
                chart=chart,
                particle_count=particle_count,
                replicate=replicate,
                oracle=oracle,
                oracle_se=oracle_se,
                oracle_label=oracle_label,
                source_sha=source_sha,
            )
            records.extend(chart_records)
            levels.extend(chart_levels)
    if any(_record_float(row, "likelihood") <= 0.0 for row in records):
        raise RuntimeError("R1 produced a non-positive or non-finite estimator.")
    stem = f"task-{task_index:02d}-{case.name}-n{particle_count}"
    replicate_path = output_root / "replicates" / f"{stem}.jsonl"
    level_path = output_root / "levels" / f"{stem}.jsonl"
    _write_jsonl(replicate_path, records)
    _write_jsonl(level_path, levels)
    certificate = {
        "schema": SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "task_index": task_index,
        "case": case.name,
        "particle_count": particle_count,
        "replicate_count": REPLICATE_COUNT,
        "charts": _charts(case),
        "oracle_likelihood": oracle,
        "oracle_standard_error": oracle_se,
        "oracle_label": oracle_label,
        "oracle_convergence": convergence,
        "record_count": len(records),
        "level_record_count": len(levels),
        "replicate_sha256": hashlib.sha256(replicate_path.read_bytes()).hexdigest(),
        "levels_sha256": hashlib.sha256(level_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "provenance": provenance,
        "status": "passed",
    }
    _write_json(output_root / "certificates" / f"{stem}.json", certificate)
    return certificate


def r2_tasks() -> tuple[tuple[WideCase, int], ...]:
    case = next(candidate for candidate in CASES if candidate.name == "boundary_heavy_four_cell_row_column")
    return tuple((case, count) for count in R2_PARTICLE_COUNTS)


def _run_r2_chart_replicate(
    *,
    case: WideCase,
    chart: TreeChart,
    particle_count: int,
    replicate: int,
    oracle: float,
    source_sha: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tree = _tree(case, chart)
    guide = _guide(case)
    schedule = parent_first_priority_schedule(
        tree,
        guide,
        root_total=ROOT_TOTAL,
        favorable=True,
        name=f"{chart}-r2-observation-energy",
    )
    configurations = [
        (
            "bootstrap_prior_ess_0.5",
            ResolutionSMCConfig(
                particle_count,
                seed=_seed128(
                    "r2",
                    case.name,
                    chart,
                    particle_count,
                    replicate,
                    "prior",
                ),
                resampling_policy="ess",
                ess_fraction=0.5,
                proposal_audit_order=R2_PROPOSAL_AUDIT_ORDER,
            ),
        )
    ]
    configurations.extend(
        (
            f"guided_piecewise_beta_bins_{bin_count}",
            ResolutionSMCConfig(
                particle_count,
                seed=_seed128(
                    "r2",
                    case.name,
                    chart,
                    particle_count,
                    replicate,
                    "guided",
                    bin_count,
                ),
                resampling_policy="ess",
                ess_fraction=0.5,
                proposal_kind="piecewise_beta_guide",
                proposal_bin_count=bin_count,
                proposal_audit_order=R2_PROPOSAL_AUDIT_ORDER,
            ),
        )
        for bin_count in R2_PROPOSAL_BIN_COUNTS
    )
    records: list[dict[str, object]] = []
    levels: list[dict[str, object]] = []
    for estimator, config in configurations:
        result, _ = run_resolution_smc(
            tree,
            schedule,
            guide,
            root_total=ROOT_TOTAL,
            config=config,
            source_identity=source_sha,
        )
        assert result is not None
        record, diagnostic_rows = _smc_record(
            case=case,
            chart=chart,
            schedule=schedule,
            particle_count=particle_count,
            replicate=replicate,
            estimator=estimator,
            seed=config.seed,
            oracle=oracle,
            oracle_se=0.0,
            oracle_label="converged Gauss-Jacobi quadrature",
            result=result,
            schema=R2_SCHEMA,
            protocol_sha256=R2_PROTOCOL_SHA256,
        )
        record["proposal_kind"] = config.proposal_kind
        record["proposal_bin_count"] = (
            config.proposal_bin_count if config.proposal_kind == "piecewise_beta_guide" else 0
        )
        record["proposal_audit_order"] = config.proposal_audit_order
        records.append(record)
        levels.extend(diagnostic_rows)
    return records, levels


def _r2_replay_audit(
    *,
    case: WideCase,
    particle_count: int,
    task_index: int,
    output_root: Path,
    source_sha: str,
) -> list[dict[str, object]]:
    tree = _tree(case, "row-first")
    guide = _guide(case)
    schedule = parent_first_priority_schedule(
        tree,
        guide,
        root_total=ROOT_TOTAL,
        favorable=True,
        name="row-first-r2-replay",
    )
    config = ResolutionSMCConfig(
        particle_count,
        seed=_seed128("r2", case.name, particle_count, "replay"),
        resampling_policy="ess",
        ess_fraction=0.5,
        proposal_kind="piecewise_beta_guide",
        proposal_bin_count=16,
        proposal_audit_order=R2_PROPOSAL_AUDIT_ORDER,
    )
    uninterrupted, _ = run_resolution_smc(
        tree,
        schedule,
        guide,
        root_total=ROOT_TOTAL,
        config=config,
        source_identity=source_sha,
    )
    assert uninterrupted is not None
    rows: list[dict[str, object]] = []
    for boundary in range(len(schedule.batches) + 1):
        partial_result, checkpoint = run_resolution_smc(
            tree,
            schedule,
            guide,
            root_total=ROOT_TOTAL,
            config=config,
            source_identity=source_sha,
            stop_after_level=boundary,
        )
        checkpoint_path = output_root / "checkpoints" / f"task-{task_index:02d}-boundary-{boundary}.npz"
        checkpoint.save(checkpoint_path)
        loaded = ResolutionSMCCheckpoint.load(
            checkpoint_path,
            tree=tree,
            schedule=schedule,
            guide=guide,
            config=config,
            allocation_paths_sha256=None,
            source_identity=source_sha,
        )
        resumed, _ = run_resolution_smc(
            tree,
            schedule,
            guide,
            root_total=ROOT_TOTAL,
            config=config,
            source_identity=source_sha,
            checkpoint=loaded,
        )
        assert resumed is not None
        identical = resumed.scientific_fingerprint == uninterrupted.scientific_fingerprint
        if not identical:
            raise RuntimeError("guided R2 checkpoint replay changed the estimator.")
        rows.append(
            {
                "boundary": boundary,
                "partial_was_terminal": partial_result is not None,
                "checkpoint": str(checkpoint_path.relative_to(output_root)),
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "scientific_fingerprint": resumed.scientific_fingerprint,
                "identical": identical,
            }
        )
    return rows


def run_r2_task(
    *,
    task_index: int,
    output_root: Path,
    source_sha: str,
) -> dict[str, object]:
    provenance = _authenticate_source(source_sha)
    provenance["protocol_sha256"] = R2_PROTOCOL_SHA256
    tasks = r2_tasks()
    if not 0 <= task_index < len(tasks):
        raise ValueError("R2 task index is out of range.")
    case, particle_count = tasks[task_index]
    oracle, convergence = _exact_oracle(case)
    replay = _r2_replay_audit(
        case=case,
        particle_count=particle_count,
        task_index=task_index,
        output_root=output_root,
        source_sha=source_sha,
    )
    records: list[dict[str, object]] = []
    levels: list[dict[str, object]] = []
    started = time.perf_counter()
    for replicate in range(R2_REPLICATE_COUNT):
        for chart in ("row-first", "column-first"):
            chart_records, chart_levels = _run_r2_chart_replicate(
                case=case,
                chart=chart,
                particle_count=particle_count,
                replicate=replicate,
                oracle=oracle,
                source_sha=source_sha,
            )
            records.extend(chart_records)
            levels.extend(chart_levels)
    if any(_record_float(row, "likelihood") <= 0.0 for row in records):
        raise RuntimeError("R2 produced a non-positive or non-finite estimator.")
    stem = f"task-{task_index:02d}-{case.name}-n{particle_count}"
    replicate_path = output_root / "replicates" / f"{stem}.jsonl"
    level_path = output_root / "levels" / f"{stem}.jsonl"
    _write_jsonl(replicate_path, records)
    _write_jsonl(level_path, levels)
    certificate = {
        "schema": R2_SCHEMA,
        "protocol_sha256": R2_PROTOCOL_SHA256,
        "task_index": task_index,
        "case": case.name,
        "particle_count": particle_count,
        "replicate_count": R2_REPLICATE_COUNT,
        "charts": ("row-first", "column-first"),
        "proposal_bin_counts": R2_PROPOSAL_BIN_COUNTS,
        "proposal_audit_order": R2_PROPOSAL_AUDIT_ORDER,
        "oracle_likelihood": oracle,
        "oracle_convergence": convergence,
        "record_count": len(records),
        "level_record_count": len(levels),
        "replicate_sha256": hashlib.sha256(replicate_path.read_bytes()).hexdigest(),
        "levels_sha256": hashlib.sha256(level_path.read_bytes()).hexdigest(),
        "checkpoint_replay": replay,
        "elapsed_seconds": time.perf_counter() - started,
        "provenance": provenance,
        "status": "passed",
    }
    _write_json(output_root / "certificates" / f"{stem}.json", certificate)
    return certificate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "protocol",
            "r2-protocol",
            "list-reference-tasks",
            "reference-task",
            "list-r1-tasks",
            "r1-task",
            "list-r2-tasks",
            "r2-task",
        ),
    )
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-sha")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.mode == "protocol":
        print(_canonical_json(protocol_payload()))
        return 0
    if arguments.mode == "r2-protocol":
        print(
            _canonical_json(
                {
                    "schema": R2_SCHEMA,
                    "protocol_sha256": R2_PROTOCOL_SHA256,
                    "particle_counts": R2_PARTICLE_COUNTS,
                    "replicate_count": R2_REPLICATE_COUNT,
                    "proposal_bin_counts": R2_PROPOSAL_BIN_COUNTS,
                    "proposal_audit_order": R2_PROPOSAL_AUDIT_ORDER,
                }
            )
        )
        return 0
    if arguments.mode == "list-reference-tasks":
        print(
            _canonical_json(
                [{"task_index": index, "case": case.name} for index, case in enumerate(reference_tasks())]
            )
        )
        return 0
    if arguments.mode == "list-r1-tasks":
        print(
            _canonical_json(
                [
                    {
                        "task_index": index,
                        "case": case.name,
                        "particle_count": count,
                    }
                    for index, (case, count) in enumerate(r1_tasks())
                ]
            )
        )
        return 0
    if arguments.mode == "list-r2-tasks":
        print(
            _canonical_json(
                [
                    {
                        "task_index": index,
                        "case": case.name,
                        "particle_count": count,
                    }
                    for index, (case, count) in enumerate(r2_tasks())
                ]
            )
        )
        return 0
    if arguments.task_index is None or arguments.output_root is None or arguments.source_sha is None:
        raise SystemExit("--task-index, --output-root and --source-sha are required.")
    if arguments.mode == "reference-task":
        result = run_reference_task(
            task_index=arguments.task_index,
            output_root=arguments.output_root,
            source_sha=arguments.source_sha,
        )
    elif arguments.mode == "r1-task":
        result = run_r1_task(
            task_index=arguments.task_index,
            output_root=arguments.output_root,
            source_sha=arguments.source_sha,
        )
    else:
        result = run_r2_task(
            task_index=arguments.task_index,
            output_root=arguments.output_root,
            source_sha=arguments.source_sha,
        )
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
