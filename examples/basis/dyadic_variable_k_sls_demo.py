"""Run variable-K dyadic SLS on repository-local TAC/MHD data.

Run from the repository root with the current checkout pinned::

    HOME=/tmp MPLCONFIGDIR=/tmp PYTHONPATH=. .venv/bin/python \
        examples/basis/dyadic_variable_k_sls_demo.py

The optimizer uses independent split and merge moves plus optional paired
shape moves. Its utility is Gaussian benchmark DFS minus a declared linear
penalty above a free region count. It is not posterior inference over ``K``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from openghg_inversions.basis.experimental.dyadic.demo_data import (
    DemoDesignData,
    load_tac_mhd_demo_data,
    load_tac_mhd_week_demo_data,
)
from openghg_inversions.basis.experimental.dyadic.demo_runner import (
    VariableKSearchConfig,
    VariableKSearchRun,
    excess_region_penalty,
    run_variable_k_dfs_search,
)
from openghg_inversions.basis.experimental.dyadic.objectives import (
    GaussianDFSObjective,
    IsotropicRegionCovariance,
)
from openghg_inversions.basis.experimental.dyadic.proposals import MergeMove, PairedMove, SplitMove
from openghg_inversions.basis.experimental.dyadic.visualization import (
    SLSVisualizationFrame,
    render_partition_comparison,
    render_search_gif,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the variable-count demo command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("tests/data"))
    parser.add_argument("--data-period", choices=("day", "week"), default="week")
    parser.add_argument("--output-directory", type=Path, default=Path("docs/plans/figures/dyadic_variable_k"))
    parser.add_argument("--coarsen", type=int, default=8)
    parser.add_argument("--initial-regions", type=int, default=24)
    parser.add_argument("--free-regions", type=int, default=32)
    parser.add_argument("--min-regions", type=int, default=2)
    parser.add_argument("--max-regions", type=int, default=80)
    parser.add_argument("--penalty", type=float, default=0.03)
    parser.add_argument("--paired-move-probability", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--pilot-proposals", type=int, default=150)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--no-gif", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run variable-count SLS and write diagnostics and visual artifacts.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Process exit status, zero on success.
    """
    command_arguments = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(command_arguments)
    data = _load_demo_data(args.data_directory, args.data_period)
    coarsened = data.coarsen(args.coarsen)
    r_diag = np.square(np.maximum(data.error, data.min_error))
    config = VariableKSearchConfig(
        initial_regions=args.initial_regions,
        free_regions=args.free_regions,
        min_regions=args.min_regions,
        max_regions=args.max_regions,
        penalty_per_extra_region=args.penalty,
        paired_move_probability=args.paired_move_probability,
        iterations=args.iterations,
        pilot_proposals=args.pilot_proposals,
        tau=args.tau,
        seed=args.seed,
        record_every=args.record_every,
    )

    search_started = perf_counter()
    run = run_variable_k_dfs_search(
        coarsened.values,
        r_diag,
        config,
        support_grid=coarsened.support_counts,
    )
    search_seconds = perf_counter() - search_started
    objective = GaussianDFSObjective(r_diag, IsotropicRegionCovariance(config.tau))

    args.output_directory.mkdir(parents=True, exist_ok=True)
    prefix = args.output_directory / f"tac_mhd_{args.data_period}_variable_k"
    background = _design_background(coarsened.values, r_diag)
    frames = _visualization_frames(run, objective, max_frames=args.max_frames)
    render_partition_comparison(
        background,
        run.tree,
        run.initial_state,
        run.result.best_state,
        run.initial_dfs,
        run.best_dfs,
        prefix.with_name(f"{prefix.name}_summary.png"),
        background_label="Weighted sensitivity magnitude",
        title="Variable-K dyadic SLS: Gaussian DFS and explicit complexity penalty",
    )
    if not args.no_gif:
        render_search_gif(
            background,
            run.tree,
            frames,
            prefix.with_suffix(".gif"),
            background_label="Weighted sensitivity magnitude",
            title=(
                "Variable-K dyadic SLS: Gaussian DFS "
                f"(penalty={config.penalty_per_extra_region:g} above K={config.free_regions})"
            ),
            fps=args.fps,
        )

    _write_trace(prefix.with_name(f"{prefix.name}_trace.csv"), run, objective)
    _write_manifest(
        prefix.with_name(f"{prefix.name}_manifest.json"),
        run,
        observations=coarsened.values.shape[0],
        native_shape=data.G.shape,
        coarsened_shape=coarsened.values.shape,
        coarsen=args.coarsen,
        data_period=args.data_period,
        benchmark_error_description=data.benchmark_error_description,
        error_diagnostics=_error_dominance_summary(data.error, data.min_error, data.sites),
        generation_argv=[str(Path(__file__)), *command_arguments],
        search_seconds=search_seconds,
        gif_written=not args.no_gif,
    )
    print(
        f"Gaussian benchmark DFS: {run.initial_dfs:.6g} -> {run.best_dfs:.6g}; "
        f"K: {len(run.initial_state.active)} -> {len(run.result.best_state.active)}; "
        f"best utility={run.result.best_score:.6g}; output={args.output_directory}"
    )
    return 0


def _load_demo_data(data_directory: Path, data_period: str) -> DemoDesignData:
    """Load the selected one-day or full-week TAC/MHD benchmark.

    Args:
        data_directory: Directory containing the committed test fixtures.
        data_period: ``"day"`` for the frozen 47-row regression fixture or
            ``"week"`` for the aligned 333-row hourly benchmark.

    Returns:
        Fine-grid contribution data and benchmark observation errors.

    Raises:
        ValueError: If ``data_period`` is not supported.
    """
    if data_period == "day":
        return load_tac_mhd_demo_data(data_directory)
    if data_period == "week":
        return load_tac_mhd_week_demo_data(data_directory)
    raise ValueError("data_period must be 'day' or 'week'.")


def _design_background(contributions: np.ndarray, r_diag: np.ndarray) -> np.ndarray:
    """Return the fixed precision-weighted sensitivity background."""
    magnitude = np.sqrt(np.sum(np.square(contributions) / r_diag[:, np.newaxis, np.newaxis], axis=0))
    return np.log1p(magnitude)


def _visualization_frames(
    run: VariableKSearchRun,
    objective: GaussianDFSObjective,
    *,
    max_frames: int,
) -> tuple[SLSVisualizationFrame, ...]:
    """Convert variable-count utility steps to DFS-labelled animation frames.

    Args:
        run: Completed variable-count search.
        objective: Unpenalized Gaussian DFS evaluator.
        max_frames: Maximum number of frames including the initializer.

    Returns:
        Selected frames whose plotted scores are DFS rather than utility.
    """
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2.")
    best_dfs = run.initial_dfs
    frames = [
        SLSVisualizationFrame(
            state=run.initial_state,
            iteration=0,
            current_score=run.initial_dfs,
            best_score=run.initial_dfs,
            temperature=run.schedule.initial_temperature,
            accepted=False,
        )
    ]
    for step in run.result.trace:
        current_dfs = objective(step.current_state, run.design)
        best_dfs = max(best_dfs, current_dfs)
        frames.append(
            SLSVisualizationFrame(
                state=step.current_state,
                iteration=step.iteration + 1,
                current_score=current_dfs,
                best_score=best_dfs,
                temperature=step.temperature,
                accepted=step.accepted,
            )
        )
    if len(frames) <= max_frames:
        return tuple(frames)
    selected = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return tuple(frames[index] for index in np.unique(selected))


def _write_trace(path: Path, run: VariableKSearchRun, objective: GaussianDFSObjective) -> None:
    """Write DFS, K, penalty, utility, and move diagnostics to CSV.

    Args:
        path: Destination CSV path.
        run: Completed variable-count search.
        objective: Unpenalized Gaussian DFS evaluator.
    """
    fieldnames = [
        "iteration",
        "temperature",
        "accepted",
        "new_best_utility",
        "regions",
        "dfs",
        "best_dfs_seen",
        "penalty",
        "current_utility",
        "best_utility",
        "candidate_utility",
        "move_type",
        "merge_parent_id",
        "split_node_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        best_dfs = run.initial_dfs
        for step in run.result.trace:
            region_count = len(step.current_state.active)
            move_type, merge_id, split_id = _move_fields(step.move)
            current_dfs = objective(step.current_state, run.design)
            best_dfs = max(best_dfs, current_dfs)
            writer.writerow(
                {
                    "iteration": step.iteration,
                    "temperature": step.temperature,
                    "accepted": step.accepted,
                    "new_best_utility": step.new_best,
                    "regions": region_count,
                    "dfs": current_dfs,
                    "best_dfs_seen": best_dfs,
                    "penalty": excess_region_penalty(region_count, run.config),
                    "current_utility": step.current_score,
                    "best_utility": step.best_score,
                    "candidate_utility": step.candidate_score,
                    "move_type": move_type,
                    "merge_parent_id": merge_id,
                    "split_node_id": split_id,
                }
            )


def _move_fields(move: SplitMove | MergeMove | PairedMove) -> tuple[str, int | str, int | str]:
    """Return CSV-safe move type and node identifiers."""
    if isinstance(move, SplitMove):
        return "split", "", move.node_id
    if isinstance(move, MergeMove):
        return "merge", move.parent_id, ""
    return "paired", move.merge_parent_id, move.split_node_id


def _error_dominance_summary(
    error: np.ndarray,
    min_error: np.ndarray,
    sites: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Summarize how the minimum-error floor affects each site.

    Args:
        error: Positive observation-error estimates.
        min_error: Positive minimum-mismatch floors aligned to ``error``.
        sites: Site labels aligned to ``error``.

    Returns:
        Per-site row counts, floor-selection counts, and error ranges suitable
        for JSON serialization.

    Raises:
        ValueError: If inputs are not aligned one-dimensional arrays.
    """
    error_values = np.asarray(error, dtype=np.float64)
    floor_values = np.asarray(min_error, dtype=np.float64)
    site_values = np.asarray(sites)
    if (
        error_values.ndim != 1
        or floor_values.shape != error_values.shape
        or site_values.shape != error_values.shape
    ):
        raise ValueError("error, min_error, and sites must be aligned one-dimensional arrays.")

    result: dict[str, dict[str, float | int]] = {}
    for site in np.unique(site_values):
        site_mask = site_values == site
        site_error = error_values[site_mask]
        site_floor = floor_values[site_mask]
        effective = np.maximum(site_error, site_floor)
        floor_selected = site_floor >= site_error
        result[str(site)] = {
            "rows": int(site_mask.sum()),
            "floor_selected_rows": int(floor_selected.sum()),
            "floor_selected_fraction": float(floor_selected.mean()),
            "observation_error_min": float(site_error.min()),
            "observation_error_max": float(site_error.max()),
            "minimum_error_min": float(site_floor.min()),
            "minimum_error_max": float(site_floor.max()),
            "effective_error_min": float(effective.min()),
            "effective_error_max": float(effective.max()),
        }
    return result


def _write_manifest(
    path: Path,
    run: VariableKSearchRun,
    *,
    observations: int,
    native_shape: tuple[int, ...],
    coarsened_shape: tuple[int, ...],
    coarsen: int,
    data_period: str,
    benchmark_error_description: str,
    error_diagnostics: dict[str, dict[str, float | int]],
    generation_argv: list[str],
    search_seconds: float,
    gif_written: bool,
) -> None:
    """Write variable-count assumptions and outcomes as JSON.

    Args:
        path: Destination JSON path.
        run: Completed variable-count search.
        observations: Number of observation rows used by the objective.
        native_shape: Native contribution-grid shape.
        coarsened_shape: Search contribution-grid shape.
        coarsen: Sum-preserving spatial block factor.
        data_period: Selected benchmark window, ``"day"`` or ``"week"``.
        benchmark_error_description: Definition and limitations of the fixed
            observation errors used in the search objective.
        error_diagnostics: Per-site summary of whether the minimum-error floor
            determines the effective fixed covariance.
        generation_argv: Script path and command-line arguments used to create
            the artifact set.
        search_seconds: Wall-clock search time.
        gif_written: Whether GIF rendering was requested.
    """
    initial_k = len(run.initial_state.active)
    final_k = len(run.result.final_state.active)
    best_k = len(run.result.best_state.active)
    payload = {
        "method": "variable-count stochastic local search",
        "posterior_sampler": False,
        "objective": "Gaussian benchmark DFS minus explicit excess-region penalty",
        "penalty_is_log_prior": False,
        "config": asdict(run.config),
        "generation_argv": generation_argv,
        "data_period": data_period,
        "observations": observations,
        "native_shape": native_shape,
        "coarsened_shape": coarsened_shape,
        "coarsen": coarsen,
        "covariance": {
            "kind": "isotropic region benchmark",
            "bocquet_consistent": False,
            "required_future_transform": "B_P = P B P.T",
        },
        "observation_covariance": benchmark_error_description,
        "observation_error_diagnostics": error_diagnostics,
        "initial": {"regions": initial_k, "dfs": run.initial_dfs, "utility": run.result.initial_score},
        "final": {"regions": final_k, "dfs": run.final_dfs, "utility": run.result.final_score},
        "best_utility_state": {
            "regions": best_k,
            "dfs": run.best_dfs,
            "penalty": excess_region_penalty(best_k, run.config),
            "utility": run.result.best_score,
        },
        "accepted_moves": run.result.accepted_moves,
        "evaluated_moves": run.result.evaluated_moves,
        "pilot_positive_losses": len(run.pilot_losses),
        "initial_temperature": run.schedule.initial_temperature,
        "final_temperature": run.schedule.final_temperature,
        "search_seconds": search_seconds,
        "gif_written": gif_written,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
