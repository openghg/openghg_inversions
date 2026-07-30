#!/usr/bin/env python3
"""Run the R0 and R1a resolution-SMC experiments.

The frozen R1a matrix is deliberately tiny in scientific dimension and broad
in independent replication.  Each Slurm array task owns one
``(case, particle_count)`` cell and writes:

* one JSONL row per estimator replicate;
* one JSONL row per SMC resolution level; and
* one strict JSON task certificate.

No realized atmospheric observation or protected catalogue is accepted by
this driver.  Every scientific input is embedded below and authenticated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Literal, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from openghg_inversions.experimental.rjmcmc.aggregation_error import (
    FourCellAggregationOracle,
    TwoCellAggregationOracle,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_resolution_smc import (
    DirectIIDResult,
    GaussianGuideSpecification,
    ResolutionSMCCheckpoint,
    ResolutionSMCConfig,
    ResolutionSMCResult,
    ResolutionSchedule,
    ResolutionTree,
    breadth_first_schedule,
    direct_iid_likelihood_average,
    draw_prior_allocation_paths,
    parent_first_priority_schedule,
    run_resolution_smc,
)

FloatArray: TypeAlias = NDArray[np.float64]
Family = Literal["two_cell", "four_cell"]
TreeChart = Literal["balanced", "row-first", "column-first"]

SCHEMA = "rjmcmc-resolution-smc-r1a-v1"
R0_SCHEMA = "rjmcmc-resolution-smc-r0-v1"
SOURCE_PLANNING_SHA = "3a4c80d17bf09dd70d69d1bef306cb888002fe54"
KNOWLEDGE_SHA = "a84bde45ccbbe42e2c400b79285b39f12b6cbcdd"
ROOT_TOTAL = 1.0
PARTICLE_COUNTS = (64, 256, 1_024, 4_096)
REPLICATE_COUNT = 64
BASE_SEED = 20260730


@dataclass(frozen=True)
class FrozenCase:
    """One exact fixed-root R1a target."""

    name: str
    family: Family
    shapes: tuple[float, ...]
    design: tuple[tuple[float, ...], ...]
    observation: tuple[float, ...]
    noise_sd: tuple[float, ...]
    oracle_orders: tuple[int, ...]


CASES = (
    FrozenCase(
        name="near_gaussian_two_cell",
        family="two_cell",
        shapes=(45.0, 55.0),
        design=((1.0, 0.7), (0.2, 1.1), (-0.5, 0.3)),
        observation=(0.93, 0.71, -0.08),
        noise_sd=(0.42, 0.55, 0.48),
        oracle_orders=(16, 32, 48, 64, 96),
    ),
    FrozenCase(
        name="skewed_g1_two_cell",
        family="two_cell",
        shapes=(0.35, 4.0),
        design=((1.8, 0.1), (-0.4, 1.2), (0.8, -0.3)),
        observation=(0.44, 0.91, -0.08),
        noise_sd=(0.25, 0.32, 0.28),
        oracle_orders=(16, 32, 48, 64, 96, 128),
    ),
    FrozenCase(
        name="boundary_heavy_four_cell_row_column",
        family="four_cell",
        shapes=(0.15, 0.18, 0.20, 0.12),
        design=(
            (2.00, 0.00, 0.10, 0.00),
            (0.00, 1.70, 0.00, 0.10),
            (0.05, 0.00, 1.90, 0.00),
            (0.00, 0.10, 0.00, 2.10),
        ),
        observation=(1.62, 0.08, 0.13, 0.06),
        noise_sd=(0.12, 0.14, 0.13, 0.15),
        oracle_orders=(16, 24, 32, 40, 48, 64, 96),
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


def _array_sha256(values: NDArray[np.generic]) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(
        _canonical_json(
            {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
        ).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def protocol_payload() -> dict[str, object]:
    """Return the complete frozen R1a protocol."""
    return {
        "schema": SCHEMA,
        "source_planning_sha": SOURCE_PLANNING_SHA,
        "knowledge_sha": KNOWLEDGE_SHA,
        "root_total": ROOT_TOTAL,
        "particle_counts": PARTICLE_COUNTS,
        "replicate_count": REPLICATE_COUNT,
        "base_seed": BASE_SEED,
        "cases": [asdict(case) for case in CASES],
        "estimators": [
            "direct_iid",
            "path_matched_no_resampling_smc",
            "bootstrap_smc_ess_0.5",
            "bootstrap_smc_every_nonterminal",
        ],
        "alternative_chart": "column-first for every four-cell estimator",
        "rng": "NumPy PCG64; SHA-256-derived independent seeds",
        "resampling": "multinomial",
        "comparison_scale": "linear normalized likelihood",
    }


PROTOCOL_SHA256 = _sha256_json(protocol_payload())


def _case_identity(case: FrozenCase) -> str:
    return _sha256_json(asdict(case))


def _seed(*parts: object) -> int:
    digest = hashlib.sha256()
    digest.update(str(BASE_SEED).encode("ascii"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(str(part).encode("ascii"))
    return int.from_bytes(digest.digest()[:16], byteorder="little", signed=False)


def _case(name: str) -> FrozenCase:
    for case in CASES:
        if case.name == name:
            return case
    raise ValueError(f"unknown case {name!r}")


def _tree(case: FrozenCase, chart: TreeChart) -> ResolutionTree:
    cell_ids = np.arange(len(case.shapes), dtype=np.int64)
    if case.family == "two_cell":
        if chart != "balanced":
            raise ValueError("two-cell cases support only the balanced chart label.")
        nested = (0, 1)
    elif chart == "row-first":
        nested = ((0, 1), (2, 3))
    elif chart == "column-first":
        nested = ((0, 2), (1, 3))
    else:
        raise ValueError("four-cell cases require row-first or column-first.")
    return ResolutionTree.from_nested_chart(cell_ids, case.shapes, nested)


def _guide(case: FrozenCase) -> GaussianGuideSpecification:
    return GaussianGuideSpecification.build(
        case.observation,
        case.design,
        case.noise_sd,
    )


def _schedule(tree: ResolutionTree, chart: TreeChart) -> ResolutionSchedule:
    return breadth_first_schedule(tree, name=f"{chart}-breadth-first")


def _charts(case: FrozenCase) -> tuple[TreeChart, ...]:
    return ("balanced",) if case.family == "two_cell" else ("row-first", "column-first")


def _oracle_convergence(case: FrozenCase) -> list[dict[str, object]]:
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
            value = float(
                oracle.coarse_conditional_log_likelihood(
                    ROOT_TOTAL,
                    case.observation,
                    case.design,
                    case.noise_sd,
                )
            )
            rows.append(
                {
                    "case": case.name,
                    "order": order,
                    "chart": "beta",
                    "log_likelihood": value,
                    "likelihood": math.exp(value),
                }
            )
    else:
        for order in case.oracle_orders:
            oracle4 = FourCellAggregationOracle(
                np.asarray(case.shapes, dtype=np.float64),
                gamma_rate=float(sum(case.shapes)),
                fraction_order=order,
                chunk_size=16_384,
            )
            for chart in ("row-first", "column-first"):
                value = float(
                    oracle4.conditional_log_likelihood(
                        ROOT_TOTAL,
                        case.observation,
                        case.design,
                        case.noise_sd,
                        tiling="root",
                        root_chart=chart,
                    )
                )
                rows.append(
                    {
                        "case": case.name,
                        "order": order,
                        "chart": chart,
                        "log_likelihood": value,
                        "likelihood": math.exp(value),
                    }
                )
    return rows


def _oracle(case: FrozenCase) -> tuple[float, list[dict[str, object]]]:
    rows = _oracle_convergence(case)
    last_order = case.oracle_orders[-1]
    final = [row for row in rows if row["order"] == last_order]
    log_values = np.asarray([row["log_likelihood"] for row in final], dtype=np.float64)
    if np.ptp(log_values) > 2.0e-10:
        raise RuntimeError(f"{case.name} final oracle charts have not converged.")
    if len(case.oracle_orders) >= 2:
        previous_order = case.oracle_orders[-2]
        previous = np.asarray(
            [row["log_likelihood"] for row in rows if row["order"] == previous_order],
            dtype=np.float64,
        )
        if abs(float(np.mean(log_values) - np.mean(previous))) > 2.0e-10:
            raise RuntimeError(f"{case.name} oracle order ladder has not converged.")
    return float(math.exp(float(np.mean(log_values)))), rows


def _git_output(arguments: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _authenticate_source(source_sha: str, *, require_detached: bool) -> dict[str, object]:
    if len(source_sha) != 40 or any(character not in "0123456789abcdef" for character in source_sha):
        raise ValueError("source SHA must be a complete lowercase 40-character Git SHA.")
    head = _git_output(["rev-parse", "HEAD"])
    if head != source_sha:
        raise RuntimeError(f"worktree HEAD {head} does not match declared source {source_sha}.")
    status = _git_output(["status", "--porcelain"])
    if status:
        raise RuntimeError("scientific worktree is not clean.")
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    detached = symbolic.returncode != 0
    if require_detached and not detached:
        raise RuntimeError("scientific execution requires a detached worktree.")
    driver_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "source_sha": source_sha,
        "head_sha": head,
        "clean": True,
        "detached": detached,
        "driver_sha256": driver_sha256,
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
            handle.write(_canonical_json(row))
            handle.write("\n")
    temporary.replace(path)


def _endpoint_count(paths: FloatArray) -> int:
    lower = np.nextafter(0.0, 1.0)
    upper = np.nextafter(1.0, 0.0)
    return int(np.count_nonzero((paths == lower) | (paths == upper)))


def _direct_record(
    *,
    case: FrozenCase,
    chart: TreeChart,
    particle_count: int,
    replicate: int,
    oracle: float,
    path_seed: int,
    path_elapsed: float,
    paths: FloatArray,
    result: DirectIIDResult,
) -> dict[str, object]:
    coordinate_count = paths.shape[1]
    return {
        "schema": SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "case": case.name,
        "case_identity": _case_identity(case),
        "family": case.family,
        "particle_count": particle_count,
        "replicate": replicate,
        "estimator": "direct_iid",
        "tree_chart": chart,
        "schedule": f"{chart}-breadth-first",
        "seed": path_seed,
        "likelihood": result.likelihood,
        "log_likelihood": result.log_likelihood,
        "oracle_likelihood": oracle,
        "linear_error": result.likelihood - oracle,
        "relative_error": result.likelihood / oracle - 1.0,
        "wall_seconds": result.elapsed_seconds + path_elapsed,
        "estimator_seconds": result.elapsed_seconds,
        "path_generation_seconds": path_elapsed,
        "peak_rss_bytes": result.peak_rss_bytes,
        "state_bytes": int(paths.nbytes + result.leaf_masses.nbytes + result.terminal_log_likelihoods.nbytes),
        "beta_draw_count": particle_count * coordinate_count,
        "beta_endpoint_repair_count": _endpoint_count(paths),
        "forward_update_count": particle_count * coordinate_count,
        "likelihood_evaluation_count": particle_count,
        "allocation_paths_sha256": result.allocation_paths_sha256,
        "terminal_log_likelihoods_sha256": _array_sha256(result.terminal_log_likelihoods),
        "terminal_leaf_masses_sha256": _array_sha256(result.leaf_masses),
        "path_match_identity_passed": None,
    }


def _smc_records(
    *,
    case: FrozenCase,
    chart: TreeChart,
    particle_count: int,
    replicate: int,
    oracle: float,
    estimator: str,
    seed: int,
    path_elapsed: float,
    conceptual_beta_draw_count: int,
    conceptual_endpoint_repairs: int,
    result: ResolutionSMCResult,
    direct: DirectIIDResult | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    path_match = None
    if direct is not None:
        path_match = (
            result.log_likelihood == direct.log_likelihood
            and result.likelihood == direct.likelihood
            and np.array_equal(
                result.terminal_log_likelihoods,
                direct.terminal_log_likelihoods,
            )
            and np.array_equal(result.terminal_leaf_masses, direct.leaf_masses)
        )
        if not path_match:
            raise RuntimeError("path-matched IID and no-resampling SMC disagree.")
    diagnostics = result.diagnostics
    wall_seconds = result.elapsed_seconds + path_elapsed
    beta_draw_count = (
        conceptual_beta_draw_count
        if conceptual_beta_draw_count
        else sum(item.beta_draw_count for item in diagnostics)
    )
    endpoint_repairs = (
        conceptual_endpoint_repairs
        if conceptual_beta_draw_count
        else sum(item.beta_endpoint_repair_count for item in diagnostics)
    )
    record = {
        "schema": SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "case": case.name,
        "case_identity": _case_identity(case),
        "family": case.family,
        "particle_count": particle_count,
        "replicate": replicate,
        "estimator": estimator,
        "tree_chart": chart,
        "schedule": f"{chart}-breadth-first",
        "seed": seed,
        "likelihood": result.likelihood,
        "log_likelihood": result.log_likelihood,
        "accumulator_log_likelihood": result.accumulator_log_likelihood,
        "no_resampling_accumulator_error": result.no_resampling_accumulator_error,
        "oracle_likelihood": oracle,
        "linear_error": result.likelihood - oracle,
        "relative_error": result.likelihood / oracle - 1.0,
        "wall_seconds": wall_seconds,
        "estimator_seconds": result.elapsed_seconds,
        "path_generation_seconds": path_elapsed,
        "peak_rss_bytes": result.peak_rss_bytes,
        "state_bytes": max(item.state_bytes for item in diagnostics),
        "beta_draw_count": beta_draw_count,
        "beta_endpoint_repair_count": endpoint_repairs,
        "forward_update_count": sum(item.forward_update_count for item in diagnostics),
        "likelihood_evaluation_count": sum(item.likelihood_evaluation_count for item in diagnostics),
        "resampling_event_count": sum(item.resampled for item in diagnostics),
        "minimum_ess_fraction": min(item.ess_fraction for item in diagnostics),
        "terminal_unique_ancestor_count": diagnostics[-1].unique_ancestor_count,
        "allocation_paths_sha256": result.allocation_paths_sha256,
        "terminal_log_likelihoods_sha256": _array_sha256(result.terminal_log_likelihoods),
        "terminal_leaf_masses_sha256": _array_sha256(result.terminal_leaf_masses),
        "final_weights_sha256": _array_sha256(result.normalized_log_weights),
        "ancestry_sha256": _array_sha256(result.ancestry),
        "scientific_fingerprint": result.scientific_fingerprint,
        "path_match_identity_passed": path_match,
    }
    level_rows: list[dict[str, object]] = []
    for level, diagnostic in enumerate(diagnostics):
        level_row = {
            "schema": SCHEMA,
            "protocol_sha256": PROTOCOL_SHA256,
            "case": case.name,
            "particle_count": particle_count,
            "replicate": replicate,
            "estimator": estimator,
            "tree_chart": chart,
            "seed": seed,
            "level_ancestry_sha256": _array_sha256(result.ancestry[level]),
            **asdict(diagnostic),
        }
        level_rows.append(level_row)
    return record, level_rows


def _run_one_chart_replicate(
    *,
    case: FrozenCase,
    chart: TreeChart,
    particle_count: int,
    replicate: int,
    oracle: float,
    source_sha: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tree = _tree(case, chart)
    schedule = _schedule(tree, chart)
    guide = _guide(case)
    path_seed = _seed(case.name, particle_count, replicate, chart, "matched-path")
    path_start = time.perf_counter()
    paths = draw_prior_allocation_paths(
        tree,
        schedule,
        particle_count=particle_count,
        seed=path_seed,
    )
    path_elapsed = time.perf_counter() - path_start
    never = ResolutionSMCConfig(
        particle_count,
        seed=path_seed,
        resampling_policy="never",
    )
    direct = direct_iid_likelihood_average(
        tree,
        schedule,
        guide,
        root_total=ROOT_TOTAL,
        config=never,
        allocation_paths=paths,
    )
    no_resampling, _ = run_resolution_smc(
        tree,
        schedule,
        guide,
        root_total=ROOT_TOTAL,
        config=never,
        source_identity=source_sha,
        allocation_paths=paths,
    )
    assert no_resampling is not None
    coordinate_work = particle_count * len(schedule.coordinate_node_ids)
    endpoint_repairs = _endpoint_count(paths)
    records = [
        _direct_record(
            case=case,
            chart=chart,
            particle_count=particle_count,
            replicate=replicate,
            oracle=oracle,
            path_seed=path_seed,
            path_elapsed=path_elapsed,
            paths=paths,
            result=direct,
        )
    ]
    no_record, level_rows = _smc_records(
        case=case,
        chart=chart,
        particle_count=particle_count,
        replicate=replicate,
        oracle=oracle,
        estimator="path_matched_no_resampling_smc",
        seed=path_seed,
        path_elapsed=path_elapsed,
        conceptual_beta_draw_count=coordinate_work,
        conceptual_endpoint_repairs=endpoint_repairs,
        result=no_resampling,
        direct=direct,
    )
    records.append(no_record)
    levels = list(level_rows)

    configurations = (
        (
            "bootstrap_smc_ess_0.5",
            ResolutionSMCConfig(
                particle_count,
                seed=_seed(case.name, particle_count, replicate, chart, "bootstrap-0.5"),
                resampling_policy="ess",
                ess_fraction=0.5,
            ),
        ),
        (
            "bootstrap_smc_every_nonterminal",
            ResolutionSMCConfig(
                particle_count,
                seed=_seed(case.name, particle_count, replicate, chart, "bootstrap-always"),
                resampling_policy="always",
                ess_fraction=0.5,
            ),
        ),
    )
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
        record, diagnostic_rows = _smc_records(
            case=case,
            chart=chart,
            particle_count=particle_count,
            replicate=replicate,
            oracle=oracle,
            estimator=estimator,
            seed=config.seed,
            path_elapsed=0.0,
            conceptual_beta_draw_count=0,
            conceptual_endpoint_repairs=0,
            result=result,
            direct=None,
        )
        records.append(record)
        levels.extend(diagnostic_rows)
    return records, levels


def r1a_tasks() -> tuple[tuple[FrozenCase, int], ...]:
    return tuple((case, particle_count) for case in CASES for particle_count in PARTICLE_COUNTS)


def run_r1a_task(
    *,
    task_index: int,
    output_root: Path,
    source_sha: str,
) -> dict[str, object]:
    """Run one exact R1a case/count cell with all 64 replicates."""
    tasks = r1a_tasks()
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"task_index must lie in [0, {len(tasks) - 1}].")
    provenance = _authenticate_source(source_sha, require_detached=True)
    case, particle_count = tasks[task_index]
    oracle, convergence = _oracle(case)
    records: list[dict[str, object]] = []
    levels: list[dict[str, object]] = []
    task_start = time.perf_counter()
    for replicate in range(REPLICATE_COUNT):
        for chart in _charts(case):
            replicate_records, replicate_levels = _run_one_chart_replicate(
                case=case,
                chart=chart,
                particle_count=particle_count,
                replicate=replicate,
                oracle=oracle,
                source_sha=source_sha,
            )
            records.extend(replicate_records)
            levels.extend(replicate_levels)
    if not all(row["path_match_identity_passed"] in (None, True) for row in records):
        raise RuntimeError("at least one pathwise identity check failed.")
    for row in records:
        likelihood = row["likelihood"]
        if not isinstance(likelihood, (int, float)) or not math.isfinite(likelihood) or likelihood <= 0.0:
            raise RuntimeError("R1a produced a non-positive or non-finite likelihood estimate.")
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
        "case_identity": _case_identity(case),
        "particle_count": particle_count,
        "replicate_count": REPLICATE_COUNT,
        "charts": _charts(case),
        "oracle_likelihood": oracle,
        "oracle_convergence": convergence,
        "record_count": len(records),
        "level_record_count": len(levels),
        "replicate_sha256": hashlib.sha256(replicate_path.read_bytes()).hexdigest(),
        "levels_sha256": hashlib.sha256(level_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - task_start,
        "provenance": provenance,
        "status": "passed",
    }
    _write_json(output_root / "certificates" / f"{stem}.json", certificate)
    return certificate


def _r0_corruption_checks(
    *,
    tree: ResolutionTree,
    schedule: ResolutionSchedule,
    guide: GaussianGuideSpecification,
    config: ResolutionSMCConfig,
    source_sha: str,
    output_root: Path,
) -> dict[str, object]:
    _, checkpoint = run_resolution_smc(
        tree,
        schedule,
        guide,
        root_total=ROOT_TOTAL,
        config=config,
        source_identity=source_sha,
        stop_after_level=1,
    )
    checkpoint_path = output_root / "checkpoints" / "r0-boundary-1.npz"
    checkpoint.save(checkpoint_path)
    column_tree = ResolutionTree.from_nested_chart(
        tree.cell_ids,
        tree.cell_alphas,
        ((0, 2), (1, 3)),
    )
    column_schedule = breadth_first_schedule(
        column_tree,
        name="column-first-breadth-first",
    )
    priority_schedule = parent_first_priority_schedule(
        tree,
        guide,
        root_total=ROOT_TOTAL,
        favorable=True,
    )
    changed_guide = GaussianGuideSpecification.build(
        guide.observation + np.array([0.0, 0.0, 0.0, 1.0e-6]),
        guide.design,
        guide.noise_sd,
    )
    changed_seed = replace(config, seed=config.seed + 1)
    changed_particles = replace(config, particle_count=config.particle_count + 1)
    variants: dict[str, dict[str, Any]] = {
        "tree": {
            "tree": column_tree,
            "schedule": column_schedule,
            "guide": guide,
            "config": config,
        },
        "schedule": {
            "tree": tree,
            "schedule": priority_schedule,
            "guide": guide,
            "config": config,
        },
        "input": {
            "tree": tree,
            "schedule": schedule,
            "guide": changed_guide,
            "config": config,
        },
        "seed": {
            "tree": tree,
            "schedule": schedule,
            "guide": guide,
            "config": changed_seed,
        },
        "particle_metadata": {
            "tree": tree,
            "schedule": schedule,
            "guide": guide,
            "config": changed_particles,
        },
    }
    failures: dict[str, str] = {}
    for name, arguments in variants.items():
        try:
            ResolutionSMCCheckpoint.load(
                checkpoint_path,
                **arguments,
                allocation_paths_sha256=None,
                source_identity=source_sha,
            )
        except ValueError as error:
            failures[name] = str(error)
        else:
            raise RuntimeError(f"checkpoint corruption variant {name} did not fail closed.")
    try:
        ResolutionSMCCheckpoint.load(
            checkpoint_path,
            tree=tree,
            schedule=schedule,
            guide=guide,
            config=config,
            allocation_paths_sha256=None,
            source_identity="0" * 40,
        )
    except ValueError as error:
        failures["source"] = str(error)
    else:
        raise RuntimeError("checkpoint source corruption did not fail closed.")
    return {
        "checkpoint": str(checkpoint_path.relative_to(output_root)),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "fail_closed_variants": failures,
        "passed": set(failures) == {"tree", "schedule", "input", "seed", "particle_metadata", "source"},
    }


def run_r0(*, output_root: Path, source_sha: str) -> dict[str, object]:
    """Run the complete focused R0 identity and replay preflight."""
    provenance = _authenticate_source(source_sha, require_detached=True)
    output_root.mkdir(parents=True, exist_ok=True)
    checks: dict[str, object] = {}
    oracle_rows: list[dict[str, object]] = []
    oracle_values: dict[str, float] = {}
    for case in CASES:
        oracle, convergence = _oracle(case)
        oracle_values[case.name] = oracle
        oracle_rows.extend(convergence)
    checks["oracle_convergence"] = {
        "values": oracle_values,
        "rows": oracle_rows,
        "passed": True,
    }

    path_rows = []
    for case in CASES:
        chart = _charts(case)[0]
        tree = _tree(case, chart)
        schedule = _schedule(tree, chart)
        guide = _guide(case)
        config = ResolutionSMCConfig(4_096, _seed("r0", case.name), resampling_policy="never")
        paths = draw_prior_allocation_paths(
            tree,
            schedule,
            particle_count=config.particle_count,
            seed=_seed("r0-path", case.name),
        )
        direct = direct_iid_likelihood_average(
            tree,
            schedule,
            guide,
            root_total=ROOT_TOTAL,
            config=config,
            allocation_paths=paths,
        )
        smc, _ = run_resolution_smc(
            tree,
            schedule,
            guide,
            root_total=ROOT_TOTAL,
            config=config,
            source_identity=source_sha,
            allocation_paths=paths,
        )
        assert smc is not None
        identical = (
            smc.log_likelihood == direct.log_likelihood
            and smc.likelihood == direct.likelihood
            and np.array_equal(
                smc.terminal_log_likelihoods,
                direct.terminal_log_likelihoods,
            )
            and np.array_equal(smc.terminal_leaf_masses, direct.leaf_masses)
        )
        path_rows.append(
            {
                "case": case.name,
                "identical": identical,
                "allocation_paths_sha256": direct.allocation_paths_sha256,
                "normalizer_accumulator_error": smc.no_resampling_accumulator_error,
                "max_terminal_mass_error": float(
                    np.max(np.abs(np.sum(smc.terminal_leaf_masses, axis=1) - ROOT_TOTAL))
                ),
                "terminal_unresolved_covariance": smc.diagnostics[-1].max_terminal_unresolved_covariance,
            }
        )
    checks["pathwise_identity_and_conservation"] = {
        "rows": path_rows,
        "passed": all(
            row["identical"]
            and row["terminal_unresolved_covariance"] == 0.0
            and row["max_terminal_mass_error"] <= 8.0 * np.spacing(ROOT_TOTAL) * 4
            for row in path_rows
        ),
    }

    two_case = _case("skewed_g1_two_cell")
    tree = _tree(two_case, "balanced")
    swapped = ResolutionTree.from_nested_chart(
        tree.cell_ids,
        tree.cell_alphas,
        (1, 0),
    )
    guide = _guide(two_case)
    schedule = _schedule(tree, "balanced")
    swapped_schedule = breadth_first_schedule(swapped, name="child-swapped")
    config = ResolutionSMCConfig(4_096, _seed("r0-child-swap"), resampling_policy="never")
    paths = draw_prior_allocation_paths(
        tree,
        schedule,
        particle_count=config.particle_count,
        seed=_seed("r0-child-swap-path"),
    )
    original, _ = run_resolution_smc(
        tree,
        schedule,
        guide,
        root_total=ROOT_TOTAL,
        config=config,
        source_identity=source_sha,
        allocation_paths=paths,
    )
    permuted, _ = run_resolution_smc(
        swapped,
        swapped_schedule,
        guide,
        root_total=ROOT_TOTAL,
        config=config,
        source_identity=source_sha,
        allocation_paths=1.0 - paths,
    )
    assert original is not None and permuted is not None
    child_mass_error = float(np.max(np.abs(original.terminal_leaf_masses - permuted.terminal_leaf_masses)))
    child_log_error = float(
        np.max(np.abs(original.terminal_log_likelihoods - permuted.terminal_log_likelihoods))
    )
    checks["child_swap_equivariance"] = {
        "max_leaf_mass_error": child_mass_error,
        "max_terminal_log_likelihood_error": child_log_error,
        "passed": child_mass_error <= 2.0e-16 and child_log_error <= 4.0e-14,
    }

    four_case = _case("boundary_heavy_four_cell_row_column")
    row_tree = _tree(four_case, "row-first")
    row_schedule = _schedule(row_tree, "row-first")
    four_guide = _guide(four_case)
    replay_config = ResolutionSMCConfig(
        1_024,
        _seed("r0-replay"),
        resampling_policy="always",
    )
    uninterrupted, _ = run_resolution_smc(
        row_tree,
        row_schedule,
        four_guide,
        root_total=ROOT_TOTAL,
        config=replay_config,
        source_identity=source_sha,
    )
    assert uninterrupted is not None
    replay_rows = []
    for boundary in range(len(row_schedule.batches) + 1):
        partial_result, partial = run_resolution_smc(
            row_tree,
            row_schedule,
            four_guide,
            root_total=ROOT_TOTAL,
            config=replay_config,
            source_identity=source_sha,
            stop_after_level=boundary,
        )
        checkpoint_path = output_root / "checkpoints" / f"r0-boundary-{boundary}.npz"
        partial.save(checkpoint_path)
        loaded = ResolutionSMCCheckpoint.load(
            checkpoint_path,
            tree=row_tree,
            schedule=row_schedule,
            guide=four_guide,
            config=replay_config,
            allocation_paths_sha256=None,
            source_identity=source_sha,
        )
        resumed, _ = run_resolution_smc(
            row_tree,
            row_schedule,
            four_guide,
            root_total=ROOT_TOTAL,
            config=replay_config,
            source_identity=source_sha,
            checkpoint=loaded,
        )
        assert resumed is not None
        replay_rows.append(
            {
                "boundary": boundary,
                "partial_was_terminal": partial_result is not None,
                "checkpoint": str(checkpoint_path.relative_to(output_root)),
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "scientific_fingerprint": resumed.scientific_fingerprint,
                "identical": resumed.scientific_fingerprint == uninterrupted.scientific_fingerprint,
            }
        )
    repeated, _ = run_resolution_smc(
        row_tree,
        row_schedule,
        four_guide,
        root_total=ROOT_TOTAL,
        config=replay_config,
        source_identity=source_sha,
    )
    assert repeated is not None
    checks["deterministic_checkpoint_replay"] = {
        "uninterrupted_fingerprint": uninterrupted.scientific_fingerprint,
        "repeat_fingerprint": repeated.scientific_fingerprint,
        "boundaries": replay_rows,
        "passed": repeated.scientific_fingerprint == uninterrupted.scientific_fingerprint
        and all(row["identical"] for row in replay_rows),
    }
    checks["provenance_fail_closed"] = _r0_corruption_checks(
        tree=row_tree,
        schedule=row_schedule,
        guide=four_guide,
        config=replay_config,
        source_sha=source_sha,
        output_root=output_root,
    )

    compatible_rows = []
    chart_estimates = []
    chart_variances = []
    count = 65_536
    for chart in ("row-first", "column-first"):
        compatible_tree = _tree(four_case, cast(TreeChart, chart))
        compatible_schedule = _schedule(compatible_tree, cast(TreeChart, chart))
        compatible_config = ResolutionSMCConfig(
            count,
            _seed("r0-compatible", chart),
            resampling_policy="never",
        )
        direct = direct_iid_likelihood_average(
            compatible_tree,
            compatible_schedule,
            four_guide,
            root_total=ROOT_TOTAL,
            config=compatible_config,
        )
        samples = np.exp(direct.terminal_log_likelihoods)
        sample_variance = float(np.var(samples, ddof=1))
        chart_estimates.append(direct.likelihood)
        chart_variances.append(sample_variance)
        compatible_rows.append(
            {
                "chart": chart,
                "likelihood": direct.likelihood,
                "standard_error": math.sqrt(sample_variance / count),
                "relative_oracle_error": direct.likelihood / oracle_values[four_case.name] - 1.0,
            }
        )
    difference = chart_estimates[0] - chart_estimates[1]
    difference_se = math.sqrt(sum(chart_variances) / count)
    checks["compatible_tree_expectation"] = {
        "rows": compatible_rows,
        "difference": difference,
        "difference_standard_error": difference_se,
        "z_score": difference / difference_se,
        "passed": abs(difference) <= 5.0 * difference_se,
    }

    passed = all(isinstance(value, dict) and bool(value.get("passed")) for value in checks.values())
    summary = {
        "schema": R0_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "provenance": provenance,
        "checks": checks,
        "status": "passed" if passed else "failed",
    }
    _write_json(output_root / "r0_summary.json", summary)
    _write_jsonl(output_root / "r0_oracle_convergence.jsonl", oracle_rows)
    if not passed:
        raise RuntimeError("at least one R0 hard contract failed.")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("protocol", "r0", "r1a-task", "list-r1a-tasks"),
    )
    parser.add_argument("--source-sha")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--task-index", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.mode == "protocol":
        print(_canonical_json(protocol_payload()))
        return 0
    if arguments.mode == "list-r1a-tasks":
        print(
            _canonical_json(
                [
                    {
                        "task_index": index,
                        "case": case.name,
                        "particle_count": particle_count,
                    }
                    for index, (case, particle_count) in enumerate(r1a_tasks())
                ]
            )
        )
        return 0
    if arguments.source_sha is None or arguments.output_root is None:
        raise SystemExit("--source-sha and --output-root are required for scientific modes.")
    if arguments.mode == "r0":
        summary = run_r0(
            output_root=arguments.output_root,
            source_sha=arguments.source_sha,
        )
        print(_canonical_json(summary))
        return 0
    if arguments.task_index is None:
        raise SystemExit("--task-index is required for r1a-task mode.")
    certificate = run_r1a_task(
        task_index=arguments.task_index,
        output_root=arguments.output_root,
        source_sha=arguments.source_sha,
    )
    print(_canonical_json(certificate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
