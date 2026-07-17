"""Compare dyadic resolution, region count, and temporal thinning.

This emissions-only experiment uses the committed full-week TAC/MHD design.
Five blocked cross-validation folds each hold out a complete UTC day at both
sites and exclude a symmetric temporal buffer from training. Basis construction
and scoring use timestamps, emissions sensitivities, prior flux, and fixed
observation-error variances. Mole-fraction targets and residuals are not used,
but the fixture's observation-error variances include variability estimated
from observed mole fractions. Boundary-condition sensitivities are not used.

The main comparison evaluates nested dyadic greedy partitions and exact
additive dynamic-programming frontiers. At one configured search resolution it
also compares no-mask axis-parallel and quadtree partitions constructed from
the training native-cell DFS field. Those baselines use a different
construction objective, but all candidates receive the same
projection-consistent training and holdout diagnostics.

A separate robustness comparison thins only the training rows of the central
fold on every wall-clock phase of a configurable stride. The complete hourly
holdout is unchanged. This is a sensitivity analysis to temporal thinning, not
a simulation of correlated observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from openghg_inversions.basis.algorithms import (
    GreedyAxisParallelSplitStrategy,
    quadtree_algorithm,
)
from openghg_inversions.basis.experimental.dyadic.demo_data import (
    DemoDesignData,
    load_tac_mhd_week_demo_data,
)
from openghg_inversions.basis.experimental.dyadic.dynamic_programming import (
    additive_partition_frontier,
)
from openghg_inversions.basis.experimental.dyadic.initializers import greedy_partition
from openghg_inversions.basis.experimental.dyadic.multiscale import sum_coarsen_grid
from openghg_inversions.basis.experimental.dyadic.partition_diagnostics import (
    build_partition_diagnostics,
    emissions_compression_quality,
)
from openghg_inversions.basis.experimental.dyadic.rhime_gaussian import (
    RHIMEGaussianMultiscale,
)
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.sweep_diagnostics import (
    TemporalSelection,
    blocked_temporal_selection,
    native_cell_dfs,
    summarize_coarsening_resolution,
)

_DEFAULT_HOLDOUT_DAYS = (
    "2019-01-02",
    "2019-01-03",
    "2019-01-04",
    "2019-01-05",
    "2019-01-06",
)
_DEFAULT_REGION_COUNTS = (16, 31, 64, 250)
_COMPARISON_BLOCK_WIDTH = 4
_COMPARISON_REGION_COUNTS = frozenset((64, 250))
_COARSE_DP_BLOCK_WIDTH = 8
_WEEK_FIXTURE_FILENAMES = (
    "obs_mhd_ch4_10m_2019-01-01_2019-01-07_data.nc",
    "obs_tac_ch4_185m_2019-01-01_2019-02-01_data.nc",
    "footprints_mhd_europe_name_10m_2019-01-01_2019-01-07_data.nc",
    "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc",
    "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc",
)
_ALGORITHM_ORDER = {
    "dyadic_greedy": 0,
    "dyadic_exact_dp": 1,
    "axis_parallel_no_mask": 2,
    "quadtree_no_mask": 3,
}

CandidateRow = dict[str, Any]
ResolutionRow = dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    """Build the emissions-only resolution-sweep command-line parser.

    Returns:
        Parser with bounded scientific defaults for the five-fold sweep and
        central-fold thinning comparison.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("tests/data"))
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("docs/plans/figures/dyadic_resolution_sweep"),
    )
    parser.add_argument("--block-widths", type=int, nargs="+", default=[8, 4, 2])
    parser.add_argument(
        "--region-counts",
        type=int,
        nargs="+",
        default=list(_DEFAULT_REGION_COUNTS),
    )
    parser.add_argument(
        "--holdout-days",
        nargs="+",
        default=list(_DEFAULT_HOLDOUT_DAYS),
        help="UTC dates whose complete days define the main blocked folds.",
    )
    parser.add_argument(
        "--thinning-holdout-day",
        default="2019-01-04",
        help="One holdout day, normally the central main fold, used for thinning robustness.",
    )
    parser.add_argument("--buffer-hours", type=float, default=24.0)
    parser.add_argument("--relative-prior-sd", type=float, default=0.5)
    parser.add_argument("--model-error-ppb", type=float, default=5.0)
    parser.add_argument(
        "--fine-dp-max-regions",
        type=int,
        default=64,
        help="DP frontier limit for search blocks finer than eight native cells.",
    )
    parser.add_argument(
        "--thinning-stride-hours",
        type=int,
        default=6,
        help="Training-only wall-clock stride; every phase offset is evaluated.",
    )
    parser.add_argument("--seed", type=int, default=20260717)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the blocked-fold and temporal-thinning resolution comparisons.

    Args:
        argv: Optional command-line arguments. ``None`` reads process arguments.

    Returns:
        Zero after all CSV, JSON, Markdown, and PNG outputs are written.

    Raises:
        ValueError: If configuration values are invalid or cannot be represented
            by a requested search grid.
        RuntimeError: If the central native-cell DFS map is not constructed.
    """
    args = build_parser().parse_args(argv)
    block_widths = _positive_unique(args.block_widths, name="block_widths", reverse=True)
    region_counts = _positive_unique(args.region_counts, name="region_counts")
    holdout_days = _normalized_days(args.holdout_days, name="holdout_days")
    thinning_day = _normalized_days([args.thinning_holdout_day], name="thinning_holdout_day")[0]
    if thinning_day not in holdout_days:
        raise ValueError("thinning_holdout_day must also appear in holdout_days.")
    if args.buffer_hours < 0.0:
        raise ValueError("buffer_hours must be non-negative.")
    if args.relative_prior_sd <= 0.0:
        raise ValueError("relative_prior_sd must be positive.")
    if args.model_error_ppb < 0.0:
        raise ValueError("model_error_ppb must be non-negative.")
    if args.fine_dp_max_regions < 1:
        raise ValueError("fine_dp_max_regions must be positive.")
    if args.thinning_stride_hours < 1:
        raise ValueError("thinning_stride_hours must be positive.")

    data = load_tac_mhd_week_demo_data(args.data_directory)
    r_diag = np.square(data.error) + args.model_error_ppb**2
    rows: list[CandidateRow] = []
    resolution_rows: list[ResolutionRow] = []
    holdout_models: dict[tuple[str, int], tuple[RHIMEGaussianMultiscale, float]] = {}
    central_cell_dfs: np.ndarray | None = None

    for holdout_day in holdout_days:
        selection = _blocked_day_selection(
            data,
            holdout_day,
            buffer_hours=args.buffer_hours,
        )
        fold_id = _fold_id(holdout_day)
        shared_cell_dfs: np.ndarray | None = None
        shared_cell_dfs_seconds = 0.0
        for block_width in block_widths:
            holdout_model, holdout_model_seconds = _build_model(
                data,
                r_diag,
                selection.holdout_mask,
                block_width=block_width,
                relative_prior_sd=args.relative_prior_sd,
            )
            holdout_models[(str(holdout_day), block_width)] = (
                holdout_model,
                holdout_model_seconds,
            )
            shared_cell_dfs, shared_cell_dfs_seconds = _run_configuration(
                rows,
                resolution_rows,
                data,
                r_diag,
                selection.training_mask,
                selection.holdout_mask,
                holdout_model,
                scenario="main_blocked_cv",
                fold_id=fold_id,
                holdout_day=holdout_day,
                phase_offset_hours=None,
                phase_anchor=None,
                training_stride_hours=None,
                block_width=block_width,
                region_counts=region_counts,
                fine_dp_max_regions=args.fine_dp_max_regions,
                relative_prior_sd=args.relative_prior_sd,
                holdout_model_seconds=holdout_model_seconds,
                holdout_model_reused=False,
                cell_dfs=shared_cell_dfs,
                cell_dfs_seconds=shared_cell_dfs_seconds,
                include_no_mask_comparisons=block_width == _COMPARISON_BLOCK_WIDTH,
                seed=args.seed,
            )
        if holdout_day == thinning_day:
            central_cell_dfs = None if shared_cell_dfs is None else shared_cell_dfs.copy()

    robustness_counts = [count for count in region_counts if count in _COMPARISON_REGION_COUNTS]
    if robustness_counts:
        _run_thinning_robustness(
            rows,
            resolution_rows,
            holdout_models,
            data,
            r_diag,
            thinning_day=thinning_day,
            region_counts=robustness_counts,
            stride_hours=args.thinning_stride_hours,
            buffer_hours=args.buffer_hours,
            fine_dp_max_regions=args.fine_dp_max_regions,
            relative_prior_sd=args.relative_prior_sd,
            seed=args.seed,
        )

    _add_dp_gaps(rows)
    rows.sort(key=_candidate_sort_key)
    resolution_rows.sort(key=_resolution_sort_key)
    if central_cell_dfs is None:
        raise RuntimeError("The sweep did not construct the central-fold native-cell DFS field.")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_directory / "emissions_holdout_sweep.csv"
    resolution_path = args.output_directory / "coarsening_resolution.csv"
    figure_path = args.output_directory / "emissions_holdout_sweep.png"
    report_path = args.output_directory / "emissions_holdout_sweep.md"
    manifest_path = args.output_directory / "emissions_holdout_sweep_manifest.json"
    _write_csv(candidate_path, rows)
    _write_csv(resolution_path, resolution_rows)
    _write_figure(
        figure_path,
        rows,
        resolution_rows,
        central_cell_dfs,
        data.lat,
        data.lon,
        region_counts=region_counts,
        thinning_counts=robustness_counts,
        thinning_stride_hours=args.thinning_stride_hours,
        thinning_day=thinning_day,
    )

    manifest: dict[str, Any] = {
        "method": "emissions-only blocked temporal cross-validation",
        "observation_use": {
            "uses_mole_fraction_targets_or_residuals": False,
            "error_weights_include_observed_within_hour_variability": True,
            "interpretation": (
                "no response-residual scoring; observed concentrations enter indirectly through fixed "
                "variability-based uncertainty weights"
            ),
        },
        "uses_boundary_condition_sensitivity": False,
        "input_provenance": {
            "data_directory_argument": str(args.data_directory),
            "fixture_sha256": _fixture_hashes(args.data_directory),
        },
        "source_provenance": _source_hashes(),
        "main_folds": [
            {
                "fold_id": _fold_id(day),
                "holdout_start": str(day.astype("datetime64[ns]")),
                "holdout_stop": str((day + np.timedelta64(1, "D")).astype("datetime64[ns]")),
            }
            for day in holdout_days
        ],
        "buffer_hours_before_and_after": args.buffer_hours,
        "block_widths_native_cells": block_widths,
        "region_counts": region_counts,
        "comparison": {
            "algorithms": ["axis_parallel_no_mask", "quadtree_no_mask"],
            "block_width": _COMPARISON_BLOCK_WIDTH,
            "region_counts": [count for count in region_counts if count in _COMPARISON_REGION_COUNTS],
            "construction_objective": (
                "search-grid sums of training native-cell DFS; differs from the projected dyadic objective"
            ),
        },
        "dynamic_programming": {
            "coarse_block_width_threshold": _COARSE_DP_BLOCK_WIDTH,
            "fine_dp_max_regions": args.fine_dp_max_regions,
            "one_frontier_call_per_fold_width_or_thinning_phase": True,
        },
        "thinning_robustness": {
            "holdout_day": str(thinning_day),
            "block_width": _COMPARISON_BLOCK_WIDTH,
            "region_counts": robustness_counts,
            "training_stride_hours": args.thinning_stride_hours,
            "phase_offsets_hours": list(range(args.thinning_stride_hours)),
            "holdout_remains_complete_hourly_day": True,
            "interpretation": "training-thinning robustness; not a simulation of correlation",
        },
        "relative_prior_sd": args.relative_prior_sd,
        "model_error_ppb": args.model_error_ppb,
        "seed": args.seed,
        "candidate_statuses": sorted({str(row["candidate_status"]) for row in rows}),
        "outputs": {
            "candidate_csv": candidate_path.name,
            "resolution_csv": resolution_path.name,
            "figure_png": figure_path.name,
            "report_markdown": report_path.name,
        },
        "timing_note": "wall-clock timings are diagnostic and machine dependent",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, rows, resolution_rows, manifest_path.name)
    print(f"Wrote {len(rows)} candidate rows and {len(resolution_rows)} resolution rows")
    return 0


def _positive_unique(values: Sequence[int], *, name: str, reverse: bool = False) -> list[int]:
    """Validate, deduplicate, and sort positive integer command-line values."""
    if not values or any(value < 1 for value in values):
        raise ValueError(f"{name} must contain positive integers.")
    return sorted(set(values), reverse=reverse)


def _fixture_hashes(data_directory: Path) -> dict[str, str]:
    """Return SHA-256 hashes for every full-week input fixture.

    Args:
        data_directory: Directory containing the full-week demo inputs.

    Returns:
        Mapping from stable fixture filename to hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If a required fixture is absent.
    """
    hashes: dict[str, str] = {}
    for filename in _WEEK_FIXTURE_FILENAMES:
        path = data_directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required provenance fixture does not exist: {path}")
        hashes[filename] = _sha256_file(path)
    return hashes


def _source_hashes() -> dict[str, str]:
    """Return hashes for the driver and local partition implementation files."""
    repository_root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__).resolve()]
    source_paths.extend(
        sorted((repository_root / "openghg_inversions/basis/experimental/dyadic").glob("*.py"))
    )
    source_paths.extend(
        repository_root / relative_path
        for relative_path in (
            "openghg_inversions/basis/algorithms/__init__.py",
            "openghg_inversions/basis/algorithms/_constrained.py",
            "openghg_inversions/basis/algorithms/_quadtree.py",
        )
    )
    return {str(path.relative_to(repository_root)): _sha256_file(path) for path in source_paths}


def _sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_days(values: Sequence[str], *, name: str) -> list[np.datetime64]:
    """Normalize unique UTC day strings in chronological order.

    Args:
        values: Date-like strings that must resolve exactly to UTC midnights.
        name: Configuration name used in validation errors.

    Returns:
        Unique day-resolution NumPy datetimes in chronological order.

    Raises:
        ValueError: If no values are supplied or one includes a non-midnight
            time component.
    """
    if not values:
        raise ValueError(f"{name} must not be empty.")
    days: list[np.datetime64] = []
    for value in values:
        instant = np.datetime64(value, "ns")
        day = instant.astype("datetime64[D]")
        if np.isnat(instant) or instant != day.astype("datetime64[ns]"):
            raise ValueError(f"{name} values must identify complete UTC days at midnight.")
        days.append(day)
    return sorted(set(days))


def _blocked_day_selection(
    data: DemoDesignData,
    holdout_day: np.datetime64,
    *,
    buffer_hours: float,
    training_stride_hours: int | None = None,
) -> TemporalSelection:
    """Build one complete-day temporal selection from site/timestamp metadata.

    Args:
        data: Full-week design whose site and timestamp arrays define rows.
        holdout_day: UTC day-resolution holdout start.
        buffer_hours: Symmetric exclusion buffer before and after the day.
        training_stride_hours: Optional training-only wall-clock stride.

    Returns:
        Immutable training and holdout masks plus selection metadata.
    """
    start = holdout_day.astype("datetime64[ns]")
    stop = (holdout_day + np.timedelta64(1, "D")).astype("datetime64[ns]")
    return blocked_temporal_selection(
        data.sites,
        data.times,
        holdout_start=start,
        holdout_stop=stop,
        buffer_hours=buffer_hours,
        thinning_hours=training_stride_hours,
    )


def _fold_id(holdout_day: np.datetime64) -> str:
    """Return the stable identifier for one complete-day holdout fold."""
    return f"fold_{holdout_day}"


def _build_model(
    data: DemoDesignData,
    r_diag: np.ndarray,
    mask: np.ndarray,
    *,
    block_width: int,
    relative_prior_sd: float,
) -> tuple[RHIMEGaussianMultiscale, float]:
    """Build and time one masked Gaussian multiscale model.

    Args:
        data: Native emissions design and prior flux.
        r_diag: Base diagonal observation covariance for all rows.
        mask: Boolean rows used in this model.
        block_width: Native-cell width of one search-grid cell.
        relative_prior_sd: Common native scaling prior standard deviation.

    Returns:
        Constructed model and elapsed wall-clock seconds.
    """
    started = perf_counter()
    model = RHIMEGaussianMultiscale.from_native_grid(
        data.G[mask],
        data.prior_flux,
        r_diag[mask],
        coarsen_factor=block_width,
        relative_prior_sd=relative_prior_sd,
    )
    return model, perf_counter() - started


def _run_configuration(
    rows: list[CandidateRow],
    resolution_rows: list[ResolutionRow],
    data: DemoDesignData,
    r_diag: np.ndarray,
    training_mask: np.ndarray,
    holdout_mask: np.ndarray,
    holdout_model: RHIMEGaussianMultiscale,
    *,
    scenario: str,
    fold_id: str,
    holdout_day: np.datetime64,
    phase_offset_hours: int | None,
    phase_anchor: np.datetime64 | None,
    training_stride_hours: int | None,
    block_width: int,
    region_counts: Sequence[int],
    fine_dp_max_regions: int,
    relative_prior_sd: float,
    holdout_model_seconds: float,
    holdout_model_reused: bool,
    cell_dfs: np.ndarray | None,
    cell_dfs_seconds: float,
    include_no_mask_comparisons: bool,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Construct and evaluate every candidate for one model configuration.

    The greedy path and exact DP frontier are each computed once and reused for
    every requested region count. Optional no-mask comparisons are constructed
    independently at K=64 and K=250 from the coarsened training native-cell
    DFS field.

    Args:
        rows: Mutable candidate output rows.
        resolution_rows: Mutable coarsening-resolution output rows.
        data: Full native emissions design and metadata.
        r_diag: Base diagonal observation covariance for all rows.
        training_mask: Rows used for basis construction.
        holdout_mask: Complete hourly rows used only for evaluation.
        holdout_model: Prebuilt holdout-only model at ``block_width``.
        scenario: Stable scenario identifier.
        fold_id: Stable blocked-fold identifier.
        holdout_day: Complete UTC holdout day.
        phase_offset_hours: Thinning phase, or ``None`` for main folds.
        phase_anchor: First wall-clock phase timestamp, or ``None``.
        training_stride_hours: Training stride, or ``None`` without thinning.
        block_width: Native-cell width of one search-grid cell.
        region_counts: Requested structural region counts.
        fine_dp_max_regions: DP cap below the coarse-width threshold.
        relative_prior_sd: Common native scaling prior standard deviation.
        holdout_model_seconds: Time used to build ``holdout_model``.
        holdout_model_reused: Whether the holdout model came from an earlier run.
        cell_dfs: Optional native-cell DFS field shared across block widths.
        cell_dfs_seconds: Original time used to compute a shared field.
        include_no_mask_comparisons: Whether to add axis/quadtree baselines.
        seed: Deterministic quadtree seed.

    Returns:
        Native-cell DFS field and the original elapsed computation time.

    Raises:
        ValueError: If a requested region count exceeds the search-grid leaves.
    """
    training_model, training_model_seconds = _build_model(
        data,
        r_diag,
        training_mask,
        block_width=block_width,
        relative_prior_sd=relative_prior_sd,
    )
    reused_cell_dfs = cell_dfs is not None
    if cell_dfs is None:
        started = perf_counter()
        cell_dfs = native_cell_dfs(training_model)
        cell_dfs_seconds = perf_counter() - started

    tree = training_model.design.tree
    maximum_regions = max(region_counts)
    if maximum_regions > len(tree.leaf_ids):
        raise ValueError(
            f"K={maximum_regions} exceeds {len(tree.leaf_ids)} leaves at block width {block_width}."
        )

    summary = summarize_coarsening_resolution(training_model, cell_dfs)
    all_leaf_labels = PartitionState(frozenset(tree.leaf_ids)).to_labels(tree)
    ceiling_started = perf_counter()
    holdout_all_leaf = build_partition_diagnostics(holdout_model, all_leaf_labels)
    holdout_all_leaf_compression = emissions_compression_quality(holdout_model, holdout_all_leaf)
    holdout_full_trace, holdout_all_leaf_aggregation_trace = _compression_traces(
        holdout_model,
        holdout_all_leaf.aggregation_error_covariance,
    )
    ceiling_seconds = perf_counter() - ceiling_started
    resolution_rows.append(
        {
            "scenario": scenario,
            "fold_id": fold_id,
            "holdout_day": str(holdout_day),
            "phase_offset_hours": phase_offset_hours,
            "phase_anchor": None if phase_anchor is None else str(phase_anchor),
            "training_stride_hours": training_stride_hours,
            "block_width": block_width,
            "training_rows": int(np.count_nonzero(training_mask)),
            "holdout_rows": int(np.count_nonzero(holdout_mask)),
            "training_model_seconds": training_model_seconds,
            "holdout_model_seconds": holdout_model_seconds,
            "holdout_model_reused": holdout_model_reused,
            "native_cell_dfs_seconds": cell_dfs_seconds,
            "native_cell_dfs_reused_across_widths": reused_cell_dfs,
            "holdout_ceiling_evaluation_seconds": ceiling_seconds,
            "model_timing_scope": "one model construction for the selected rows and block width",
            "native_cell_dfs_timing_scope": (
                "one native-cell Cholesky solve field; reused across block widths for the same training mask"
            ),
            "holdout_all_leaf_compression_ceiling": holdout_all_leaf_compression,
            "holdout_full_weighted_trace": holdout_full_trace,
            "holdout_all_leaf_aggregation_weighted_trace": holdout_all_leaf_aggregation_trace,
            "holdout_all_leaf_dfs": holdout_all_leaf.dfs,
            "holdout_all_leaf_dfs_fraction": _safe_fraction(
                holdout_all_leaf.dfs,
                holdout_model.full_grid_dfs,
            ),
            **asdict(summary),
            "top_native_cell_latitude": float(data.lat[summary.top_native_cell_row]),
            "top_native_cell_longitude": float(data.lon[summary.top_native_cell_column]),
        }
    )

    greedy_started = perf_counter()
    greedy_result = greedy_partition(tree, maximum_regions, training_model.split_gain)
    greedy_seconds = perf_counter() - greedy_started
    greedy_states = _states_from_split_history(training_model, greedy_result.split_history, region_counts)

    dp_limit = maximum_regions
    if block_width < _COARSE_DP_BLOCK_WIDTH:
        dp_limit = min(dp_limit, fine_dp_max_regions)
    dp_started = perf_counter()
    dp_frontier = additive_partition_frontier(tree, training_model.tile_scores, dp_limit)
    dp_seconds = perf_counter() - dp_started

    common = {
        "scenario": scenario,
        "fold_id": fold_id,
        "holdout_day": str(holdout_day),
        "phase_offset_hours": phase_offset_hours,
        "phase_anchor": None if phase_anchor is None else str(phase_anchor),
        "training_stride_hours": training_stride_hours,
        "block_width": block_width,
        "training_rows": int(np.count_nonzero(training_mask)),
        "holdout_rows": int(np.count_nonzero(holdout_mask)),
        "training_model_seconds": training_model_seconds,
        "holdout_model_seconds": holdout_model_seconds,
        "holdout_model_reused": holdout_model_reused,
        "native_cell_dfs_seconds": cell_dfs_seconds,
        "model_timing_scope": "training and holdout model construction; shared by candidates",
        "holdout_all_leaf_compression_ceiling": holdout_all_leaf_compression,
    }
    for target_regions in region_counts:
        _append_candidate(
            rows,
            common,
            algorithm="dyadic_greedy",
            labels=greedy_states[target_regions].to_labels(tree),
            training_model=training_model,
            holdout_model=holdout_model,
            target_regions=target_regions,
            solver_seconds=greedy_seconds,
            solver_timing_scope=f"one nested greedy path through K={maximum_regions}; shared by requested K",
            construction_objective="greedy gains in additive projected training DFS on the dyadic tree",
        )
        if target_regions <= dp_limit:
            _append_candidate(
                rows,
                common,
                algorithm="dyadic_exact_dp",
                labels=dp_frontier[target_regions].state.to_labels(tree),
                training_model=training_model,
                holdout_model=holdout_model,
                target_regions=target_regions,
                solver_seconds=dp_seconds,
                solver_timing_scope=f"one complete exact DP frontier through K={dp_limit}; shared by requested K",
                construction_objective="global additive projected training DFS optimum on the dyadic tree",
            )
        else:
            _append_omitted_candidate(
                rows,
                common,
                algorithm="dyadic_exact_dp",
                target_regions=target_regions,
                reason=f"fine-grid DP capped at K={dp_limit}",
                solver_seconds=dp_seconds,
                solver_timing_scope=f"one complete exact DP frontier through K={dp_limit}; requested K omitted",
                construction_objective="global additive projected training DFS optimum on the dyadic tree",
            )

    comparison_counts = [count for count in region_counts if count in _COMPARISON_REGION_COUNTS]
    if include_no_mask_comparisons and comparison_counts:
        weights = sum_coarsen_grid(cell_dfs[np.newaxis, ...], block_width).values[0]
        class_mask = np.ones(weights.shape, dtype=bool)
        for target_regions in comparison_counts:
            axis_started = perf_counter()
            axis_labels = GreedyAxisParallelSplitStrategy()(weights, class_mask, target_regions)
            axis_seconds = perf_counter() - axis_started
            _append_candidate(
                rows,
                common,
                algorithm="axis_parallel_no_mask",
                labels=axis_labels,
                training_model=training_model,
                holdout_model=holdout_model,
                target_regions=target_regions,
                solver_seconds=axis_seconds,
                solver_timing_scope="one independent no-mask axis-parallel construction at requested K",
                construction_objective=(
                    "axis-parallel partition of search-grid sums of training native-cell DFS; "
                    "different from the projected dyadic objective"
                ),
            )

            quadtree_started = perf_counter()
            quadtree_labels = np.asarray(
                quadtree_algorithm(weights, nbasis=target_regions, seed=seed),
                dtype=np.int64,
            )
            quadtree_seconds = perf_counter() - quadtree_started
            _append_candidate(
                rows,
                common,
                algorithm="quadtree_no_mask",
                labels=quadtree_labels,
                training_model=training_model,
                holdout_model=holdout_model,
                target_regions=target_regions,
                solver_seconds=quadtree_seconds,
                solver_timing_scope="one independent no-mask quadtree construction at requested K",
                construction_objective=(
                    "quadtree partition of search-grid sums of training native-cell DFS; "
                    "different from the projected dyadic objective"
                ),
            )
    return cell_dfs, cell_dfs_seconds


def _run_thinning_robustness(
    rows: list[CandidateRow],
    resolution_rows: list[ResolutionRow],
    holdout_models: dict[tuple[str, int], tuple[RHIMEGaussianMultiscale, float]],
    data: DemoDesignData,
    r_diag: np.ndarray,
    *,
    thinning_day: np.datetime64,
    region_counts: Sequence[int],
    stride_hours: int,
    buffer_hours: float,
    fine_dp_max_regions: int,
    relative_prior_sd: float,
    seed: int,
) -> None:
    """Evaluate all wall-clock phases of central-fold training thinning.

    Args:
        rows: Mutable candidate output rows.
        resolution_rows: Mutable coarsening-resolution output rows.
        holdout_models: Main-fold holdout model cache, updated if necessary.
        data: Full native emissions design and metadata.
        r_diag: Base diagonal observation covariance for all rows.
        thinning_day: Complete hourly holdout retained for every phase.
        region_counts: Requested K values, normally 64 and 250.
        stride_hours: Positive training-only wall-clock stride.
        buffer_hours: Symmetric exclusion buffer before and after the day.
        fine_dp_max_regions: Exact DP cap at width four.
        relative_prior_sd: Common native scaling prior standard deviation.
        seed: Deterministic seed passed to candidate builders.
    """
    unthinned = _blocked_day_selection(data, thinning_day, buffer_hours=buffer_hours)
    phase_zero = _blocked_day_selection(
        data,
        thinning_day,
        buffer_hours=buffer_hours,
        training_stride_hours=stride_hours,
    )
    if phase_zero.stride_anchor is None:
        raise RuntimeError("A positive thinning stride did not produce a wall-clock anchor.")

    cache_key = (str(thinning_day), _COMPARISON_BLOCK_WIDTH)
    holdout_model_reused = cache_key in holdout_models
    if holdout_model_reused:
        holdout_model, holdout_model_seconds = holdout_models[cache_key]
    else:
        holdout_model, holdout_model_seconds = _build_model(
            data,
            r_diag,
            unthinned.holdout_mask,
            block_width=_COMPARISON_BLOCK_WIDTH,
            relative_prior_sd=relative_prior_sd,
        )
        holdout_models[cache_key] = holdout_model, holdout_model_seconds

    for phase_offset in range(stride_hours):
        training_mask, phase_anchor = _phase_training_mask(
            data.times,
            unthinned.training_mask,
            phase_zero,
            phase_offset_hours=phase_offset,
        )
        _run_configuration(
            rows,
            resolution_rows,
            data,
            r_diag,
            training_mask,
            unthinned.holdout_mask,
            holdout_model,
            scenario="thinning_robustness",
            fold_id=_fold_id(thinning_day),
            holdout_day=thinning_day,
            phase_offset_hours=phase_offset,
            phase_anchor=phase_anchor,
            training_stride_hours=stride_hours,
            block_width=_COMPARISON_BLOCK_WIDTH,
            region_counts=region_counts,
            fine_dp_max_regions=fine_dp_max_regions,
            relative_prior_sd=relative_prior_sd,
            holdout_model_seconds=holdout_model_seconds,
            holdout_model_reused=holdout_model_reused,
            cell_dfs=None,
            cell_dfs_seconds=0.0,
            include_no_mask_comparisons=False,
            seed=seed,
        )


def _phase_training_mask(
    timestamps: np.ndarray,
    eligible_training_mask: np.ndarray,
    phase_zero: TemporalSelection,
    *,
    phase_offset_hours: int,
) -> tuple[np.ndarray, np.datetime64]:
    """Select one wall-clock thinning phase from eligible training rows.

    ``blocked_temporal_selection`` defines the shared stride anchor and phase
    zero. Additional phases retain timestamps congruent to ``anchor + phase``
    while preserving the exact same holdout and buffer exclusions.

    Args:
        timestamps: Original observation timestamps for all sites.
        eligible_training_mask: Unthinned training mask after holdout/buffer exclusion.
        phase_zero: Selection returned by ``blocked_temporal_selection`` with
            ``training_stride_hours`` enabled.
        phase_offset_hours: Integer phase in ``[0, stride)``.

    Returns:
        Boolean training mask and its wall-clock phase anchor.

    Raises:
        ValueError: If stride metadata or phase bounds are invalid, or the phase
            selects no training rows.
    """
    stride_hours = phase_zero.thinning_hours
    anchor = phase_zero.stride_anchor
    if stride_hours is None or anchor is None:
        raise ValueError("phase_zero must contain thinning stride metadata.")
    if phase_offset_hours < 0 or phase_offset_hours >= stride_hours:
        raise ValueError("phase_offset_hours must lie in [0, training_stride_hours).")
    normalized_times = np.asarray(timestamps).astype("datetime64[ns]")
    phase_anchor = anchor + np.timedelta64(phase_offset_hours, "h")
    stride_delta = np.timedelta64(stride_hours, "h").astype("timedelta64[ns]")
    on_phase = (normalized_times - phase_anchor) % stride_delta == np.timedelta64(0, "ns")
    training_mask = np.asarray(eligible_training_mask, dtype=bool) & on_phase
    if phase_offset_hours == 0 and not np.array_equal(training_mask, phase_zero.training_mask):
        raise RuntimeError("Derived phase-zero mask disagrees with blocked_temporal_selection.")
    if not np.any(training_mask):
        raise ValueError(f"training thinning phase {phase_offset_hours} is empty.")
    return training_mask, phase_anchor


def _states_from_split_history(
    model: RHIMEGaussianMultiscale,
    split_history: tuple[int, ...],
    region_counts: Sequence[int],
) -> dict[int, PartitionState]:
    """Replay one nested greedy path and retain requested fixed-K states.

    Args:
        model: Gaussian model whose tree supplied ``split_history``.
        split_history: Greedy split node IDs in execution order.
        region_counts: Requested structural region counts.

    Returns:
        Mapping from every requested count to the corresponding nested state.

    Raises:
        RuntimeError: If the history does not reach every requested count.
    """
    tree = model.design.tree
    state = PartitionState.root(tree)
    requested = set(region_counts)
    states = {1: state} if 1 in requested else {}
    for node_id in split_history:
        state = state.split(tree, node_id)
        if len(state.active) in requested:
            states[len(state.active)] = state
    missing = requested - set(states)
    if missing:
        raise RuntimeError(f"Greedy split history is missing region counts: {sorted(missing)}")
    return states


def _append_candidate(
    rows: list[CandidateRow],
    common: CandidateRow,
    *,
    algorithm: str,
    labels: np.ndarray,
    training_model: RHIMEGaussianMultiscale,
    holdout_model: RHIMEGaussianMultiscale,
    target_regions: int,
    solver_seconds: float,
    solver_timing_scope: str,
    construction_objective: str,
) -> None:
    """Evaluate one labelled basis under common projected diagnostics.

    Args:
        rows: Mutable output row collection.
        common: Shared fold/model/timing metadata.
        algorithm: Stable candidate algorithm label.
        labels: Positive labels on the common search grid.
        training_model: Model built only from eligible training rows.
        holdout_model: Model built only from the complete hourly holdout.
        target_regions: Requested structural region count.
        solver_seconds: Candidate construction wall time.
        solver_timing_scope: Human-readable scope of the solver timing.
        construction_objective: Objective used to construct the labels.
    """
    evaluation_started = perf_counter()
    training = build_partition_diagnostics(training_model, labels)
    holdout = build_partition_diagnostics(holdout_model, labels)
    holdout_compression = emissions_compression_quality(holdout_model, holdout)
    holdout_full_trace, holdout_aggregation_trace = _compression_traces(
        holdout_model,
        holdout.aggregation_error_covariance,
    )
    evaluation_seconds = perf_counter() - evaluation_started
    row = _candidate_template(
        common,
        algorithm=algorithm,
        target_regions=target_regions,
        solver_seconds=solver_seconds,
        solver_timing_scope=solver_timing_scope,
        construction_objective=construction_objective,
    )
    row.update(
        {
            "candidate_status": "completed",
            "actual_k": int(np.unique(labels).size),
            "effective_training_k": int(training.supported_region_ids.size),
            "effective_holdout_k": int(holdout.supported_region_ids.size),
            "training_dfs": training.dfs,
            "training_full_grid_dfs": training_model.full_grid_dfs,
            "training_dfs_fraction": _safe_fraction(training.dfs, training_model.full_grid_dfs),
            "holdout_compression": holdout_compression,
            "holdout_full_weighted_trace": holdout_full_trace,
            "holdout_aggregation_weighted_trace": holdout_aggregation_trace,
            "holdout_only_dfs": holdout.dfs,
            "holdout_full_grid_dfs": holdout_model.full_grid_dfs,
            "holdout_only_dfs_fraction": _safe_fraction(holdout.dfs, holdout_model.full_grid_dfs),
            "evaluation_seconds": evaluation_seconds,
        }
    )
    rows.append(row)


def _append_omitted_candidate(
    rows: list[CandidateRow],
    common: CandidateRow,
    *,
    algorithm: str,
    target_regions: int,
    reason: str,
    solver_seconds: float,
    solver_timing_scope: str,
    construction_objective: str,
) -> None:
    """Append an explicit candidate row for a configured solver omission."""
    row = _candidate_template(
        common,
        algorithm=algorithm,
        target_regions=target_regions,
        solver_seconds=solver_seconds,
        solver_timing_scope=solver_timing_scope,
        construction_objective=construction_objective,
    )
    row["candidate_status"] = "omitted_fine_dp_limit"
    row["omission_reason"] = reason
    rows.append(row)


def _candidate_template(
    common: CandidateRow,
    *,
    algorithm: str,
    target_regions: int,
    solver_seconds: float,
    solver_timing_scope: str,
    construction_objective: str,
) -> CandidateRow:
    """Return one stable-schema candidate row with empty diagnostic values."""
    return {
        **common,
        "algorithm": algorithm,
        "candidate_status": None,
        "omission_reason": None,
        "target_k": target_regions,
        "actual_k": None,
        "effective_training_k": None,
        "effective_holdout_k": None,
        "construction_objective": construction_objective,
        "training_dfs": None,
        "training_full_grid_dfs": None,
        "training_dfs_fraction": None,
        "holdout_compression": None,
        "holdout_full_weighted_trace": None,
        "holdout_aggregation_weighted_trace": None,
        "holdout_only_dfs": None,
        "holdout_full_grid_dfs": None,
        "holdout_only_dfs_fraction": None,
        "dp_training_dfs_gap": None,
        "solver_seconds": solver_seconds,
        "solver_timing_scope": solver_timing_scope,
        "evaluation_seconds": None,
        "evaluation_timing_scope": "training and holdout projected diagnostics plus compression",
    }


def _safe_fraction(numerator: float, denominator: float) -> float:
    """Return a finite ratio, requiring a positive denominator."""
    if denominator <= 0.0:
        raise ValueError("DFS fraction denominator must be positive.")
    return numerator / denominator


def _compression_traces(
    model: RHIMEGaussianMultiscale,
    aggregation_covariance: np.ndarray,
) -> tuple[float, float]:
    """Return weighted full-signal and aggregation traces for pooling.

    Args:
        model: Holdout model defining the full emissions covariance and
            diagonal base observation covariance.
        aggregation_covariance: Partition aggregation-error covariance on the
            same observation rows.

    Returns:
        The denominator and numerator of the emissions-compression loss,
        respectively.

    Raises:
        ValueError: If covariance shapes are incompatible or weighted traces
            are not finite and non-negative.
    """
    full_signal = np.asarray(model.full_signal_covariance, dtype=float)
    aggregation = np.asarray(aggregation_covariance, dtype=float)
    r_diag = np.asarray(model.r_diag, dtype=float)
    expected_shape = (r_diag.size, r_diag.size)
    if full_signal.shape != expected_shape or aggregation.shape != expected_shape:
        raise ValueError("Compression covariance shapes must match the observation count.")

    full_trace = float(np.sum(np.diag(full_signal) / r_diag))
    aggregation_trace = float(np.sum(np.diag(aggregation) / r_diag))
    if not np.isfinite(full_trace) or full_trace <= 0.0:
        raise ValueError("Full-signal weighted trace must be positive and finite.")
    tolerance = 1e-10 * max(1.0, full_trace)
    if not np.isfinite(aggregation_trace) or aggregation_trace < -tolerance:
        raise ValueError("Aggregation weighted trace must be finite and non-negative.")
    return full_trace, max(aggregation_trace, 0.0)


def _add_dp_gaps(rows: list[CandidateRow]) -> None:
    """Attach like-for-like exact-DP training DFS gaps where available."""
    exact = {
        _comparison_key(row): float(row["training_dfs"])
        for row in rows
        if row["algorithm"] == "dyadic_exact_dp" and row["candidate_status"] == "completed"
    }
    for row in rows:
        key = _comparison_key(row)
        if (
            row["candidate_status"] == "completed"
            and row["algorithm"] in {"dyadic_greedy", "dyadic_exact_dp"}
            and key in exact
        ):
            row["dp_training_dfs_gap"] = exact[key] - float(row["training_dfs"])


def _comparison_key(row: CandidateRow) -> tuple[Any, ...]:
    """Return fields identifying one like-for-like fixed-K comparison."""
    return (
        row["scenario"],
        row["fold_id"],
        row["phase_offset_hours"],
        row["block_width"],
        row["target_k"],
    )


def _candidate_sort_key(row: CandidateRow) -> tuple[Any, ...]:
    """Return the deterministic output order for candidate rows."""
    return (
        0 if row["scenario"] == "main_blocked_cv" else 1,
        row["holdout_day"],
        -1 if row["phase_offset_hours"] is None else row["phase_offset_hours"],
        -int(row["block_width"]),
        int(row["target_k"]),
        _ALGORITHM_ORDER[str(row["algorithm"])],
    )


def _resolution_sort_key(row: ResolutionRow) -> tuple[Any, ...]:
    """Return the deterministic output order for resolution rows."""
    return (
        0 if row["scenario"] == "main_blocked_cv" else 1,
        row["holdout_day"],
        -1 if row["phase_offset_hours"] is None else row["phase_offset_hours"],
        -int(row["block_width"]),
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write a nonempty sequence of stable diagnostic dictionaries."""
    if not rows:
        raise ValueError("Cannot write an empty diagnostic CSV.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(
    path: Path,
    rows: Sequence[CandidateRow],
    resolution_rows: Sequence[ResolutionRow],
    cell_dfs: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    region_counts: Sequence[int],
    thinning_counts: Sequence[int],
    thinning_stride_hours: int,
    thinning_day: np.datetime64,
) -> None:
    """Plot fold ranges, coarsening ceilings, phase sensitivity, and cell DFS.

    Args:
        path: Destination PNG path.
        rows: Candidate rows including completed and omitted records.
        resolution_rows: Coarsening-resolution rows for all configurations.
        cell_dfs: Central unthinned fold native-cell DFS field.
        latitudes: Native-grid latitude coordinates.
        longitudes: Native-grid longitude coordinates.
        region_counts: Configured main candidate K values.
        thinning_counts: Configured thinning candidate K values.
        thinning_stride_hours: Number of wall-clock phases in robustness runs.
        thinning_day: Holdout day represented by the cell map and thinning panel.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    completed_main = [
        row for row in rows if row["scenario"] == "main_blocked_cv" and row["candidate_status"] == "completed"
    ]
    main_groups: dict[tuple[str, int], list[CandidateRow]] = defaultdict(list)
    for row in completed_main:
        main_groups[str(row["algorithm"]), int(row["block_width"])].append(row)
    for (algorithm, block_width), group in sorted(
        main_groups.items(),
        key=lambda item: (-item[0][1], _ALGORITHM_ORDER[item[0][0]]),
    ):
        x_values: list[int] = []
        medians: list[float] = []
        minima: list[float] = []
        maxima: list[float] = []
        for target_k in region_counts:
            values = [float(row["holdout_compression"]) for row in group if row["target_k"] == target_k]
            if values:
                x_values.append(target_k)
                medians.append(float(np.median(values)))
                minima.append(min(values))
                maxima.append(max(values))
        if x_values:
            (line,) = axes[0, 0].plot(
                x_values,
                medians,
                marker="o",
                label=f"{_algorithm_label(algorithm)}, width {block_width}",
            )
            axes[0, 0].fill_between(x_values, minima, maxima, color=line.get_color(), alpha=0.14)
    axes[0, 0].set(
        xlabel="Requested basis regions K",
        ylabel="Held-out compression",
        title="Main folds: median and full range",
    )
    axes[0, 0].legend(fontsize="small", ncols=2)

    main_resolution = [row for row in resolution_rows if row["scenario"] == "main_blocked_cv"]
    widths = sorted({int(row["block_width"]) for row in main_resolution})
    for field, label, marker in (
        ("all_leaf_retained_fraction", "training all-leaf DFS fraction", "o"),
        ("holdout_all_leaf_compression_ceiling", "holdout all-leaf compression", "s"),
    ):
        medians = []
        lower = []
        upper = []
        for width in widths:
            values = [float(row[field]) for row in main_resolution if row["block_width"] == width]
            median = float(np.median(values))
            medians.append(median)
            lower.append(median - min(values))
            upper.append(max(values) - median)
        axes[0, 1].errorbar(
            widths,
            medians,
            yerr=np.asarray([lower, upper]),
            marker=marker,
            capsize=4,
            label=label,
        )
    axes[0, 1].set(
        xlabel="Native cells per search-leaf side",
        ylabel="Retained fraction / compression",
        title="Coarsening ceiling: median and fold range",
        xticks=widths,
    )
    axes[0, 1].legend(fontsize="small")

    thinning = [
        row
        for row in rows
        if row["scenario"] == "thinning_robustness" and row["candidate_status"] == "completed"
    ]
    thinning_groups: dict[tuple[str, int], list[CandidateRow]] = defaultdict(list)
    for row in thinning:
        thinning_groups[str(row["algorithm"]), int(row["target_k"])].append(row)
    for (algorithm, target_k), group in sorted(
        thinning_groups.items(),
        key=lambda item: (item[0][1], _ALGORITHM_ORDER[item[0][0]]),
    ):
        ordered = sorted(group, key=lambda row: int(row["phase_offset_hours"]))
        axes[1, 0].plot(
            [int(row["phase_offset_hours"]) for row in ordered],
            [float(row["holdout_compression"]) for row in ordered],
            marker="o",
            label=f"{_algorithm_label(algorithm)}, K={target_k}",
        )
    configured_k = ", ".join(str(value) for value in thinning_counts) or "none"
    axes[1, 0].set(
        xlabel=f"Wall-clock phase offset (hours; stride {thinning_stride_hours} h)",
        ylabel="Held-out compression",
        title=f"Training-thinning phase sensitivity (configured K: {configured_k})",
        xticks=list(range(thinning_stride_hours)),
    )
    if thinning_groups:
        axes[1, 0].legend(fontsize="small")
    else:
        axes[1, 0].text(0.5, 0.5, "No configured thinning candidates", ha="center", va="center")

    positive = np.where(cell_dfs > 0.0, cell_dfs, np.nan)
    extent = (
        float(np.min(longitudes)),
        float(np.max(longitudes)),
        float(np.min(latitudes)),
        float(np.max(latitudes)),
    )
    image = axes[1, 1].imshow(
        np.log10(positive),
        origin="lower",
        cmap="viridis",
        aspect="auto",
        extent=extent,
    )
    top_row, top_column = np.unravel_index(int(np.nanargmax(cell_dfs)), cell_dfs.shape)
    axes[1, 1].scatter(
        [longitudes[top_column]],
        [latitudes[top_row]],
        marker="x",
        color="red",
        s=55,
        label="largest cell DFS",
    )
    axes[1, 1].set(
        xlabel="Longitude",
        ylabel="Latitude",
        title=f"Native-cell DFS, unthinned {_fold_id(thinning_day)} training",
    )
    axes[1, 1].legend(loc="upper right", fontsize="small")
    fig.colorbar(image, ax=axes[1, 1], label="log10 native-cell DFS")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _algorithm_label(algorithm: str) -> str:
    """Return a compact human-readable algorithm label."""
    return {
        "dyadic_greedy": "dyadic greedy",
        "dyadic_exact_dp": "dyadic exact DP",
        "axis_parallel_no_mask": "axis-parallel (no mask)",
        "quadtree_no_mask": "quadtree (no mask)",
    }[algorithm]


def _write_report(
    path: Path,
    rows: Sequence[CandidateRow],
    resolution_rows: Sequence[ResolutionRow],
    manifest_name: str,
) -> None:
    """Write compact median/range summaries across folds and thinning phases.

    Args:
        path: Destination Markdown path.
        rows: Candidate rows including explicit solver omissions.
        resolution_rows: Coarsening-resolution rows for all configurations.
        manifest_name: Adjacent JSON manifest filename.
    """
    completed_main = [
        row for row in rows if row["scenario"] == "main_blocked_cv" and row["candidate_status"] == "completed"
    ]
    main_groups: dict[tuple[int, str, int], list[CandidateRow]] = defaultdict(list)
    for row in completed_main:
        main_groups[int(row["block_width"]), str(row["algorithm"]), int(row["target_k"])].append(row)

    lines = [
        "# Dyadic Resolution and Emissions Holdout Sweep",
        "",
        "Five folds hold out complete UTC days across both sites, with the configured buffer on both sides. "
        "Mole-fraction targets and residuals are not scored, although observed within-hour variability "
        "contributes to the fixed uncertainty weights. Boundary-condition sensitivities are not used.",
        "",
        "## Main blocked-fold summary",
        "",
        "Values are median [minimum, maximum] across folds.",
        "",
        "| block width | algorithm | target K | actual K | training DFS fraction | holdout compression | "
        "pooled holdout compression | holdout DFS fraction | DP training gap |",
        "| ---: | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for (block_width, algorithm, target_k), group in sorted(
        main_groups.items(),
        key=lambda item: (-item[0][0], item[0][2], _ALGORITHM_ORDER[item[0][1]]),
    ):
        gaps = [float(row["dp_training_dfs_gap"]) for row in group if row["dp_training_dfs_gap"] is not None]
        lines.append(
            f"| {block_width} | {_algorithm_label(algorithm)} | {target_k} | "
            f"{_median_range([float(row['actual_k']) for row in group], digits=1)} | "
            f"{_median_range([float(row['training_dfs_fraction']) for row in group])} | "
            f"{_median_range([float(row['holdout_compression']) for row in group])} | "
            f"{_pooled_compression(group):.4f} | "
            f"{_median_range([float(row['holdout_only_dfs_fraction']) for row in group])} | "
            f"{_median_range(gaps, scientific=True) if gaps else '-'} |"
        )

    main_resolution = [row for row in resolution_rows if row["scenario"] == "main_blocked_cv"]
    resolution_groups: dict[int, list[ResolutionRow]] = defaultdict(list)
    for row in main_resolution:
        resolution_groups[int(row["block_width"])].append(row)
    lines.extend(
        [
            "",
            "## Coarsening ceiling",
            "",
            "| block width | search shape | training all-leaf DFS fraction | unresolved training DFS | "
            "holdout all-leaf compression | pooled holdout compression | top cell share | top 10 share |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for block_width, group in sorted(resolution_groups.items(), reverse=True):
        shapes = sorted({str(row["search_shape"]) for row in group})
        lines.append(
            f"| {block_width} | {', '.join(shapes)} | "
            f"{_median_range([float(row['all_leaf_retained_fraction']) for row in group])} | "
            f"{_median_range([float(row['unresolved_dfs']) for row in group])} | "
            f"{_median_range([float(row['holdout_all_leaf_compression_ceiling']) for row in group])} | "
            f"{_pooled_resolution_compression(group):.4f} | "
            f"{_median_range([float(row['top_native_cell_fraction']) for row in group])} | "
            f"{_median_range([float(row['top_ten_native_cell_fraction']) for row in group])} |"
        )

    thinning = [
        row
        for row in rows
        if row["scenario"] == "thinning_robustness" and row["candidate_status"] == "completed"
    ]
    thinning_groups: dict[tuple[str, int], list[CandidateRow]] = defaultdict(list)
    for row in thinning:
        thinning_groups[str(row["algorithm"]), int(row["target_k"])].append(row)
    lines.extend(
        [
            "",
            "## Training-thinning phase sensitivity",
            "",
            "Values are median [minimum, maximum] across wall-clock phase offsets; the hourly holdout is unchanged.",
            "",
            "| algorithm | target K | completed phases | training DFS fraction | holdout compression | "
            "holdout DFS fraction |",
            "| --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for (algorithm, target_k), group in sorted(
        thinning_groups.items(),
        key=lambda item: (item[0][1], _ALGORITHM_ORDER[item[0][0]]),
    ):
        lines.append(
            f"| {_algorithm_label(algorithm)} | {target_k} | {len(group)} | "
            f"{_median_range([float(row['training_dfs_fraction']) for row in group])} | "
            f"{_median_range([float(row['holdout_compression']) for row in group])} | "
            f"{_median_range([float(row['holdout_only_dfs_fraction']) for row in group])} |"
        )

    omitted = [row for row in rows if row["candidate_status"] != "completed"]
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Exact DP is the fixed-K oracle only for the additive projected dyadic objective. Each frontier is "
            "computed once per fold/width or thinning phase.",
            "- Axis-parallel and quadtree candidates have no spatial mask and use coarsened training native-cell "
            "DFS as their construction field. Their construction objective differs from the dyadic objective.",
            "- Training thinning tests phase sensitivity to fewer closely spaced rows. It does not simulate "
            "temporal correlation or replace a non-diagonal likelihood.",
            f"- {len(omitted)} configured candidate rows were explicitly omitted by the fine-grid DP limit; "
            "their metric fields are blank in the candidate CSV.",
            f"- Configuration, fixture/source hashes, and timing scopes are recorded in `{manifest_name}` and the CSVs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _median_range(
    values: Sequence[float],
    *,
    digits: int = 4,
    scientific: bool = False,
) -> str:
    """Format a median and full range for one nonempty numeric sequence."""
    if not values:
        raise ValueError("Cannot summarize an empty value sequence.")
    formatter = f".{'3e' if scientific else f'{digits}f'}"
    median = float(np.median(values))
    return f"{median:{formatter}} [{min(values):{formatter}}, {max(values):{formatter}}]"


def _pooled_compression(rows: Sequence[CandidateRow]) -> float:
    """Pool candidate compression traces before taking their ratio."""
    full_trace = sum(float(row["holdout_full_weighted_trace"]) for row in rows)
    aggregation_trace = sum(float(row["holdout_aggregation_weighted_trace"]) for row in rows)
    return 1.0 - aggregation_trace / full_trace


def _pooled_resolution_compression(rows: Sequence[ResolutionRow]) -> float:
    """Pool all-leaf compression traces before taking their ratio."""
    full_trace = sum(float(row["holdout_full_weighted_trace"]) for row in rows)
    aggregation_trace = sum(float(row["holdout_all_leaf_aggregation_weighted_trace"]) for row in rows)
    return 1.0 - aggregation_trace / full_trace


if __name__ == "__main__":
    raise SystemExit(main())
