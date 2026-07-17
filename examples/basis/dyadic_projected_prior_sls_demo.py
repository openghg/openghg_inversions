"""Run projection-consistent variable-K dyadic SLS on TAC/MHD fixtures.

The regional columns are sums of RHIME's native footprint-times-prior-flux
columns. A prior-weighted restriction makes projected coefficients independent
of the unresolved Gaussian residual, whose covariance is added to observation
error. The search is still an optimizer, not posterior inference over the
partition or region count.
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
    ProjectedVariableKSearchRun,
    VariableKSearchConfig,
    excess_region_penalty,
    run_projected_variable_k_dfs_search,
)
from openghg_inversions.basis.experimental.dyadic.proposals import MergeMove, PairedMove, SplitMove
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree
from openghg_inversions.basis.experimental.dyadic.visualization import (
    SLSVisualizationFrame,
    render_partition_comparison,
    render_search_gif,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the projection-consistent demo command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("tests/data"))
    parser.add_argument("--data-period", choices=("day", "week"), default="week")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("docs/plans/figures/dyadic_projected_prior"),
    )
    parser.add_argument("--coarsen", type=int, default=8)
    parser.add_argument("--initial-regions", type=int, default=24)
    parser.add_argument("--free-regions", type=int, default=32)
    parser.add_argument("--min-regions", type=int, default=2)
    parser.add_argument("--max-regions", type=int, default=80)
    parser.add_argument("--penalty", type=float, default=0.03)
    parser.add_argument("--paired-move-probability", type=float, default=0.2)
    parser.add_argument("--initial-loss-acceptance", type=float, default=0.5)
    parser.add_argument("--final-loss-acceptance", type=float, default=0.01)
    parser.add_argument("--hold-fraction", type=float, default=0.05)
    parser.add_argument("--polish-fraction", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--pilot-proposals", type=int, default=300)
    parser.add_argument("--relative-prior-sd", type=float, default=1.0)
    parser.add_argument("--flux-tolerance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--no-gif", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the search and write its diagnostics, report, and visual artifacts."""
    command_arguments = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(command_arguments)
    data = _load_demo_data(args.data_directory, args.data_period)
    r_diag = np.square(np.maximum(data.error, data.min_error))
    config = VariableKSearchConfig(
        initial_regions=args.initial_regions,
        free_regions=args.free_regions,
        min_regions=args.min_regions,
        max_regions=args.max_regions,
        penalty_per_extra_region=args.penalty,
        paired_move_probability=args.paired_move_probability,
        initial_loss_acceptance=args.initial_loss_acceptance,
        final_loss_acceptance=args.final_loss_acceptance,
        hold_fraction=args.hold_fraction,
        polish_fraction=args.polish_fraction,
        iterations=args.iterations,
        pilot_proposals=args.pilot_proposals,
        tau=args.relative_prior_sd,
        seed=args.seed,
        record_every=args.record_every,
    )

    search_started = perf_counter()
    run = run_projected_variable_k_dfs_search(
        data.G,
        data.prior_flux,
        r_diag,
        config,
        coarsen_factor=args.coarsen,
        flux_tolerance=args.flux_tolerance,
    )
    search_seconds = perf_counter() - search_started

    args.output_directory.mkdir(parents=True, exist_ok=True)
    prefix = args.output_directory / f"tac_mhd_{args.data_period}_projected_variable_k"
    background = _design_background(
        run.model.design.values[:, run.model.design.tree.leaf_ids],
        r_diag,
        run.model.design.tree,
    )
    frames = _visualization_frames(run, max_frames=args.max_frames)
    render_partition_comparison(
        background,
        run.model.design.tree,
        run.initial_state,
        run.result.best_state,
        run.initial_dfs,
        run.best_dfs,
        prefix.with_name(f"{prefix.name}_summary.png"),
        background_label="Weighted sensitivity magnitude",
        title="Projected-prior variable-K SLS: initial and best-utility states",
        score_label="Projected Gaussian DFS",
    )
    if not args.no_gif:
        render_search_gif(
            background,
            run.model.design.tree,
            frames,
            prefix.with_suffix(".gif"),
            background_label="Weighted sensitivity magnitude",
            title=(
                "Projected-prior variable-K SLS "
                f"(penalty={config.penalty_per_extra_region:g} above K={config.free_regions})"
            ),
            score_label="Penalized utility",
            score_axis_label="Penalized utility / native-grid DFS bound",
            show_region_count=True,
            fps=args.fps,
        )

    trace_path = prefix.with_name(f"{prefix.name}_trace.csv")
    manifest_path = prefix.with_name(f"{prefix.name}_manifest.json")
    report_path = prefix.with_name(f"{prefix.name}_report.md")
    _write_trace(trace_path, run)
    manifest = _manifest(
        run,
        data=data,
        r_diag=r_diag,
        data_period=args.data_period,
        coarsen=args.coarsen,
        flux_tolerance=args.flux_tolerance,
        generation_argv=[str(Path(__file__)), *command_arguments],
        search_seconds=search_seconds,
        gif_written=not args.no_gif,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, run, manifest, prefix)
    print(
        f"Projected Gaussian DFS: {run.initial_dfs:.6g} -> {run.best_dfs:.6g}; "
        f"K: {len(run.initial_state.active)} -> {len(run.result.best_state.active)}; "
        f"native-grid bound={run.full_grid_dfs:.6g}; output={args.output_directory}"
    )
    return 0


def _load_demo_data(data_directory: Path, data_period: str) -> DemoDesignData:
    """Load the selected one-day or full-week TAC/MHD fixture."""
    if data_period == "day":
        return load_tac_mhd_demo_data(data_directory)
    if data_period == "week":
        return load_tac_mhd_week_demo_data(data_directory)
    raise ValueError("data_period must be 'day' or 'week'.")


def _design_background(
    leaf_columns: np.ndarray,
    r_diag: np.ndarray,
    tree: DyadicTree,
) -> np.ndarray:
    """Return a precision-weighted sensitivity background on the search grid."""
    magnitude = np.sqrt(np.sum(np.square(leaf_columns) / r_diag[:, np.newaxis], axis=0))
    background = np.empty(tree.shape, dtype=float)
    for value, node_id in zip(magnitude, tree.leaf_ids):
        tile = tree.tile(node_id)
        background[tile.row_start, tile.col_start] = np.log1p(value)
    return background


def _visualization_frames(
    run: ProjectedVariableKSearchRun,
    *,
    max_frames: int,
) -> tuple[SLSVisualizationFrame, ...]:
    """Convert search records to animation frames with the true DFS bound."""
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2.")
    frames = [
        SLSVisualizationFrame(
            state=run.initial_state,
            iteration=0,
            current_score=run.result.initial_score,
            best_score=run.result.initial_score,
            temperature=run.schedule.initial_temperature,
            accepted=False,
            full_grid_dfs=run.full_grid_dfs,
        )
    ]
    frames.extend(
        SLSVisualizationFrame(
            state=step.current_state,
            iteration=step.iteration + 1,
            current_score=step.current_score,
            best_score=step.best_score,
            temperature=step.temperature,
            accepted=step.accepted,
            full_grid_dfs=run.full_grid_dfs,
        )
        for step in run.result.trace
    )
    if len(frames) <= max_frames:
        return tuple(frames)
    selected = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return tuple(frames[index] for index in np.unique(selected))


def _write_trace(path: Path, run: ProjectedVariableKSearchRun) -> None:
    """Write DFS, K, penalty, utility, and move diagnostics to CSV."""
    fieldnames = [
        "iteration",
        "temperature",
        "accepted",
        "new_best_utility",
        "regions",
        "effective_regions",
        "dfs",
        "best_accepted_dfs",
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
            current_dfs = run.model.score(step.current_state)
            best_dfs = max(best_dfs, current_dfs)
            writer.writerow(
                {
                    "iteration": step.iteration,
                    "temperature": step.temperature,
                    "accepted": step.accepted,
                    "new_best_utility": step.new_best,
                    "regions": region_count,
                    "effective_regions": run.model.effective_region_count(step.current_state),
                    "dfs": current_dfs,
                    "best_accepted_dfs": best_dfs,
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


def _manifest(
    run: ProjectedVariableKSearchRun,
    *,
    data: DemoDesignData,
    r_diag: np.ndarray,
    data_period: str,
    coarsen: int,
    flux_tolerance: float,
    generation_argv: list[str],
    search_seconds: float,
    gif_written: bool,
) -> dict[str, object]:
    """Build a machine-readable statement of assumptions and outcomes."""
    best_state = run.result.best_state
    reduced_signal = run.model.reduced_signal_covariance(best_state)
    aggregation = run.model.aggregation_error_covariance(best_state)
    invariant_error = np.linalg.norm(
        np.diag(r_diag) + aggregation + reduced_signal - run.model.innovation_covariance
    )
    initial_k = len(run.initial_state.active)
    best_k = len(best_state.active)
    return {
        "method": "projection-consistent variable-count stochastic local search",
        "posterior_sampler": False,
        "objective": "projected Gaussian DFS minus explicit excess-region penalty",
        "config": asdict(run.config),
        "generation_argv": generation_argv,
        "data_period": data_period,
        "observations": int(data.G.shape[0]),
        "native_shape": data.G.shape,
        "search_shape": run.model.design.tree.shape,
        "coarsen": coarsen,
        "flux_tolerance": flux_tolerance,
        "supported_native_cells": int(np.count_nonzero(np.abs(data.prior_flux) > flux_tolerance)),
        "covariance": {
            "fine_prior": "B = relative_prior_sd**2 * diag(prior_flux**2)",
            "regional_variance": "relative_prior_sd**2 / supported_native_cells_in_region",
            "restriction": "(U.T B^-1 U)^-1 U.T B^-1",
            "prolongation": "U = diag(prior_flux) A.T",
            "projected_and_residual_independent": True,
            "aggregation_error_included": True,
        },
        "base_observation_covariance": {
            "kind": "fixed diagonal demo benchmark",
            "description": data.benchmark_error_description,
            "minimum_variance": float(r_diag.min()),
            "maximum_variance": float(r_diag.max()),
        },
        "initial": {
            "regions": initial_k,
            "effective_regions": run.model.effective_region_count(run.initial_state),
            "dfs": run.initial_dfs,
            "utility": run.result.initial_score,
        },
        "best_utility_state": {
            "regions": best_k,
            "effective_regions": run.model.effective_region_count(best_state),
            "dfs": run.best_dfs,
            "penalty": excess_region_penalty(best_k, run.config),
            "utility": run.result.best_score,
        },
        "native_grid_reference": {
            "dfs": run.full_grid_dfs,
            "comparable_upper_bound": True,
            "best_fraction": run.best_dfs / run.full_grid_dfs if run.full_grid_dfs else 0.0,
        },
        "best_covariance_diagnostics": {
            "reduced_signal_trace": float(np.trace(reduced_signal)),
            "aggregation_error_trace": float(np.trace(aggregation)),
            "aggregation_error_minimum_eigenvalue": float(np.linalg.eigvalsh(aggregation).min()),
            "innovation_closure_frobenius_error": float(invariant_error),
        },
        "accepted_moves": run.result.accepted_moves,
        "evaluated_moves": run.result.evaluated_moves,
        "pilot_positive_losses": len(run.pilot_losses),
        "initial_temperature": run.schedule.initial_temperature,
        "final_temperature": run.schedule.final_temperature,
        "search_seconds": search_seconds,
        "gif_written": gif_written,
    }


def _write_report(
    path: Path,
    run: ProjectedVariableKSearchRun,
    manifest: dict[str, object],
    prefix: Path,
) -> None:
    """Write a concise shareable Markdown report beside generated artifacts."""
    initial_k = len(run.initial_state.active)
    best_k = len(run.result.best_state.active)
    lines = [
        "# Projection-Consistent Dyadic SLS Demonstration",
        "",
        "## Result",
        "",
        f"- Starting point: greedy exact-DFS dyadic initializer with K={initial_k}.",
        f"- Best-utility state: K={best_k} after {run.result.evaluated_moves} evaluated local moves.",
        f"- DFS at initializer/best-utility state: {run.initial_dfs:.6g} to {run.best_dfs:.6g}.",
        f"- Native-grid no-reduction DFS: {run.full_grid_dfs:.6g}.",
        (
            f"- Best-utility state reaches {run.best_dfs / run.full_grid_dfs:.2%} of the bound."
            if run.full_grid_dfs
            else "- Native-grid DFS is zero, so a recovered fraction is undefined."
        ),
        "",
        "## Inputs and score",
        "",
        "The displayed background is the precision-weighted magnitude of the",
        "coarsened footprint-times-prior-flux columns. It is context only; the",
        "search score uses the full Gaussian observation-space covariance.",
        "Regional columns are summed RHIME columns. Regional prior variance is",
        "`relative_prior_sd**2 / native_support`, and unresolved fine-grid",
        "variation is included in the effective observation covariance.",
        "",
        "The prior-weighted restriction makes the regional coefficient and",
        "unresolved residual independent. Consequently the reduced signal plus",
        "aggregation error equals the same native innovation covariance for",
        "every partition, and the native-grid DFS is a valid upper bound.",
        "",
        "## Artifacts",
        "",
        f"- Static comparison: `{prefix.name}_summary.png`",
        f"- Animation: `{prefix.name}.gif`",
        f"- Search trace: `{prefix.name}_trace.csv`",
        f"- Machine-readable assumptions: `{prefix.name}_manifest.json`",
        "",
        "## Limitations",
        "",
        "This is stochastic local-search optimization, not partition posterior",
        "inference. It assumes independent Gaussian native relative-scaling",
        "errors and uses a fixed diagonal observation covariance benchmark.",
        "The production RHIME mismatch model is not changed or reproduced.",
        "",
        f"Manifest method: `{manifest['method']}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
