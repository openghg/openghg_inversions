"""Run the experimental fixed-count dyadic SLS demo on TAC/MHD test data.

Run from the repository root with the current checkout pinned, for example::

    HOME=/tmp MPLCONFIGDIR=/tmp PYTHONPATH=. .venv/bin/python \
        examples/basis/dyadic_sls_demo.py --coarsen 8 --target-regions 32

The search maximizes Gaussian benchmark DFS under an isotropic regional prior.
It is stochastic local optimization, not posterior MCMC, and the isotropic
covariance is not the Bocquet-consistent ``B_P = P B P.T`` transformation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from openghg_inversions.basis.experimental.dyadic.demo_data import load_tac_mhd_demo_data
from openghg_inversions.basis.experimental.dyadic.demo_runner import (
    DemoSearchConfig,
    DemoSearchRun,
    run_fixed_count_dfs_search,
)
from openghg_inversions.basis.experimental.dyadic.multiscale import MultiscaleDesign
from openghg_inversions.basis.experimental.dyadic.objectives import (
    GaussianDFSObjective,
    IsotropicRegionCovariance,
)
from openghg_inversions.basis.experimental.dyadic.proposals import PairedMove
from openghg_inversions.basis.experimental.dyadic.search import SearchStep
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.visualization import (
    SLSVisualizationFrame,
    render_partition_comparison,
    render_search_gif,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the repository-local demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("tests/data"))
    parser.add_argument("--output-directory", type=Path, default=Path("docs/plans/figures/dyadic_sls"))
    parser.add_argument("--coarsen", type=int, default=8, help="Sum-preserving spatial block factor.")
    parser.add_argument("--target-regions", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--pilot-proposals", type=int, default=100)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF rendering after the search.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load test data, run fixed-count SLS, and write demo artifacts.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Process exit status, zero on success.
    """
    args = build_parser().parse_args(argv)
    data_started = perf_counter()
    data = load_tac_mhd_demo_data(args.data_directory)
    coarsened = data.coarsen(args.coarsen)
    sigma = np.maximum(data.error, data.min_error)
    r_diag = np.square(sigma)
    data_seconds = perf_counter() - data_started
    config = DemoSearchConfig(
        target_regions=args.target_regions,
        iterations=args.iterations,
        pilot_proposals=args.pilot_proposals,
        tau=args.tau,
        seed=args.seed,
        record_every=args.record_every,
    )
    search_started = perf_counter()
    run = run_fixed_count_dfs_search(
        coarsened.values,
        r_diag,
        config,
        support_grid=coarsened.support_counts,
    )
    search_seconds = perf_counter() - search_started

    args.output_directory.mkdir(parents=True, exist_ok=True)
    prefix = args.output_directory / "tac_mhd_sls"
    background = _design_background(coarsened.values, r_diag)
    site_scores = _site_diagnostics(run, coarsened.values, r_diag, data.sites)
    frames = _visualization_frames(run, max_frames=args.max_frames)

    render_partition_comparison(
        background,
        run.tree,
        run.initial_state,
        run.result.best_state,
        run.result.initial_score,
        run.result.best_score,
        prefix.with_name(f"{prefix.name}_summary.png"),
        background_label="Weighted sensitivity magnitude",
        title="TAC/MHD dyadic SLS: isotropic Gaussian DFS benchmark",
    )
    if not args.no_gif:
        render_search_gif(
            background,
            run.tree,
            frames,
            prefix.with_suffix(".gif"),
            background_label="Weighted sensitivity magnitude",
            title="TAC/MHD dyadic SLS: isotropic Gaussian DFS benchmark",
            fps=args.fps,
        )

    _write_trace(prefix.with_name(f"{prefix.name}_trace.csv"), run)
    _write_manifest(
        prefix.with_name(f"{prefix.name}_manifest.json"),
        run,
        coarsen=args.coarsen,
        native_shape=data.G.shape,
        coarsened_shape=coarsened.values.shape,
        site_scores=site_scores,
        data_seconds=data_seconds,
        search_seconds=search_seconds,
        gif_written=not args.no_gif,
    )
    print(
        f"Gaussian benchmark DFS: {run.result.initial_score:.6g} -> "
        f"{run.result.best_score:.6g}; K={config.target_regions}; "
        f"evaluations={run.result.evaluated_moves}; output={args.output_directory}"
    )
    return 0


def _design_background(contributions: np.ndarray, r_diag: np.ndarray) -> np.ndarray:
    """Return fixed observation-precision-weighted sensitivity magnitude."""
    magnitude = np.sqrt(np.sum(np.square(contributions) / r_diag[:, np.newaxis, np.newaxis], axis=0))
    return np.log1p(magnitude)


def _site_diagnostics(
    run: DemoSearchRun,
    contributions: np.ndarray,
    r_diag: np.ndarray,
    sites: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Recompute initial and best DFS independently for each site.

    Args:
        run: Combined-site search result.
        contributions: Coarsened observation contribution grid.
        r_diag: Combined observation covariance diagonal.
        sites: Site label aligned to the observation axis.

    Returns:
        Initial and best Gaussian benchmark DFS keyed by site name.
    """
    diagnostics: dict[str, dict[str, float]] = {}
    covariance = IsotropicRegionCovariance(run.config.tau)
    for site in ("MHD", "TAC"):
        mask = sites == site
        design = MultiscaleDesign.from_grid(contributions[mask], run.tree)
        objective = GaussianDFSObjective(r_diag[mask], covariance)
        diagnostics[site] = {
            "initial_dfs": objective(run.initial_state, design),
            "best_dfs": objective(run.result.best_state, design),
        }
    return diagnostics


def _visualization_frames(run: DemoSearchRun, *, max_frames: int) -> tuple[SLSVisualizationFrame, ...]:
    """Convert a bounded search trace to animation records.

    Args:
        run: Completed fixed-count search.
        max_frames: Maximum number of animation frames including the initial
            state.

    Returns:
        Strictly iteration-ordered visualization records.

    Raises:
        ValueError: If ``max_frames`` is smaller than two.
    """
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
        )
    ]
    frames.extend(_frame_from_step(step) for step in run.result.trace)
    if len(frames) <= max_frames:
        return tuple(frames)
    selected = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return tuple(frames[index] for index in np.unique(selected))


def _frame_from_step(step: SearchStep[PartitionState, PairedMove]) -> SLSVisualizationFrame:
    """Convert one generic search step to a visualization frame."""
    return SLSVisualizationFrame(
        state=step.current_state,
        iteration=step.iteration + 1,
        current_score=step.current_score,
        best_score=step.best_score,
        temperature=step.temperature,
        accepted=step.accepted,
    )


def _write_trace(path: Path, run: DemoSearchRun) -> None:
    """Write bounded proposal diagnostics to a CSV file.

    Args:
        path: Destination CSV path.
        run: Completed search whose trace should be serialized.
    """
    fieldnames = [
        "iteration",
        "temperature",
        "candidate_score",
        "current_score",
        "best_score",
        "accepted",
        "new_best",
        "regions",
        "merge_parent_id",
        "split_node_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for step in run.result.trace:
            writer.writerow(
                {
                    "iteration": step.iteration,
                    "temperature": step.temperature,
                    "candidate_score": step.candidate_score,
                    "current_score": step.current_score,
                    "best_score": step.best_score,
                    "accepted": step.accepted,
                    "new_best": step.new_best,
                    "regions": len(step.current_state.active),
                    "merge_parent_id": step.move.merge_parent_id,
                    "split_node_id": step.move.split_node_id,
                }
            )


def _write_manifest(
    path: Path,
    run: DemoSearchRun,
    *,
    coarsen: int,
    native_shape: tuple[int, ...],
    coarsened_shape: tuple[int, ...],
    site_scores: dict[str, dict[str, float]],
    data_seconds: float,
    search_seconds: float,
    gif_written: bool,
) -> None:
    """Write assumptions, configuration, and final diagnostics as JSON.

    Args:
        path: Destination JSON path.
        run: Completed search.
        coarsen: Sum-preserving spatial coarsening factor.
        native_shape: Native observation/grid shape.
        coarsened_shape: Search observation/grid shape.
        site_scores: Separately recomputed site diagnostics.
        data_seconds: Wall-clock fixture loading and coarsening time.
        search_seconds: Wall-clock initializer, pilot, and SLS time.
        gif_written: Whether GIF rendering was requested.
    """
    payload = {
        "method": "fixed-count stochastic local search",
        "objective": "Gaussian benchmark DFS",
        "posterior_sampler": False,
        "covariance": {
            "kind": "isotropic region benchmark",
            "tau": run.config.tau,
            "bocquet_consistent": False,
            "required_future_transform": "B_P = P B P.T",
        },
        "observation_covariance": (
            "fixed benchmark diag(max(error, min_error)**2); min_error is a "
            "minimum-mismatch floor, not the inferred production total-error process"
        ),
        "config": asdict(run.config),
        "coarsen": coarsen,
        "native_shape": native_shape,
        "coarsened_shape": coarsened_shape,
        "initial_dfs": run.result.initial_score,
        "final_dfs": run.result.final_score,
        "best_dfs": run.result.best_score,
        "site_diagnostics": site_scores,
        "accepted_moves": run.result.accepted_moves,
        "evaluated_moves": run.result.evaluated_moves,
        "pilot_positive_losses": len(run.pilot_losses),
        "initial_temperature": run.schedule.initial_temperature,
        "final_temperature": run.schedule.final_temperature,
        "data_seconds": data_seconds,
        "search_seconds": search_seconds,
        "gif_written": gif_written,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
