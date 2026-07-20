"""Compare latent dyadic partitions with fixed checkerboard inversions.

This synthetic benchmark adapts the local Lunt et al. (2016) checkerboard case
from ``codex/tdmcmc-numba-rewrite`` to the Gaussian product-space sampler.  It
uses an ``8 x 8`` grid, alternating 0.5/1.5 scaling factors on sixteen regular
``2 x 2`` regions, smooth footprint-like sensitivities, and independent 5 ppb
noise.  Sixty-four observations condition each inversion and thirty-two
independent sensitivity rows assess posterior predictions.

Three otherwise identical inversions are compared:

* the true fixed 16-region partition, an oracle geometry benchmark;
* the coarser fixed 8-region partition used to initialize the search;
* latent ``K`` and ``P`` sampled with local split/merge product-space updates.

The benchmark is an implementation proof of concept, not a reproduction of
the paper or evidence for production inversion performance.  Pass
``--output-directory`` to save recovery and sampler-diagnostic plots alongside
the machine-readable summary.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from openghg_inversions.basis.experimental.dyadic.gaussian_product_space import (
    GaussianProductSpaceTarget,
)
from openghg_inversions.basis.experimental.dyadic.gaussian_product_space_sampler import (
    sample_collapsed_gaussian_product_space,
    sample_gaussian_product_space,
)
from openghg_inversions.basis.experimental.dyadic.partition_prior import (
    RegionCountPartitionPrior,
)
from openghg_inversions.basis.experimental.dyadic.product_space import ProductSpaceState
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree

_GRID_SHAPE = (8, 8)
_TRAIN_OBSERVATIONS = 64
_HOLDOUT_OBSERVATIONS = 32
_OBSERVATION_SD = 5.0


@dataclass(frozen=True, slots=True, eq=False)
class CheckerboardCase:
    """Complete deterministic train/holdout checkerboard experiment."""

    tree: DyadicTree
    target: GaussianProductSpaceTarget
    truth: np.ndarray
    train_design: np.ndarray
    holdout_design: np.ndarray
    holdout_observations: np.ndarray
    holdout_noiseless: np.ndarray
    coarse_partition: PartitionState
    wrong_partition: PartitionState
    truth_partition: PartitionState


@dataclass(frozen=True, slots=True)
class InversionMetrics:
    """Common posterior metrics for one fixed or latent inversion."""

    prediction_rmse_ppb: float
    predictive_log_density: float
    grid_rmse: float
    recovered_checkerboard_contrast: float
    mean_regions: float
    minimum_regions: int
    maximum_regions: int


@dataclass(frozen=True, slots=True)
class CheckerboardBenchmark:
    """Fixed and latent inversion results plus sampler diagnostics."""

    fixed_truth: InversionMetrics
    fixed_wrong: InversionMetrics
    fixed_coarse: InversionMetrics
    latent: InversionMetrics
    latent_sampler: str
    latent_partition_acceptance_rate: float
    latent_split_acceptance_rate: float | None
    latent_merge_acceptance_rate: float | None
    latent_warmup_acceptance_rate: float | None
    latent_unique_partitions: int
    latent_runtime_seconds: float
    latent_beats_wrong_prediction_rmse: bool
    latent_beats_coarse_prediction_rmse: bool
    latent_oracle_log_score_difference_per_observation: float
    latent_oracle_log_score_noninferior: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable benchmark summary."""
        return asdict(self)


def checkerboard_truth() -> np.ndarray:
    """Return alternating 0.5/1.5 values on sixteen regular 2 by 2 blocks."""
    rows, columns = np.indices(_GRID_SHAPE)
    block_parity = (rows // 2 + columns // 2) % 2
    return np.where(block_parity == 0, 0.5, 1.5).astype(float)


def regular_depth_partition(tree: DyadicTree, depth: int) -> PartitionState:
    """Return the complete frontier at one tree depth.

    Args:
        tree: Balanced dyadic tree.
        depth: Non-negative depth represented by a complete frontier.

    Returns:
        Valid partition containing every node at ``depth``.

    Raises:
        ValueError: If no nodes exist at ``depth`` or the nodes do not form a
            complete frontier.
    """
    active = frozenset(tile.node_id for tile in tree.nodes if tile.depth == depth)
    if not active:
        raise ValueError(f"Tree has no nodes at depth {depth}.")
    partition = PartitionState(active=active)
    partition.validate(tree)
    return partition


def irregular_same_count_partition(
    tree: DyadicTree,
    coarse_partition: PartitionState,
) -> PartitionState:
    """Construct a deterministic K=16 partition that misses 2 by 2 truth blocks.

    Args:
        tree: Canonical ``8 x 8`` benchmark tree.
        coarse_partition: Complete depth-three K=8 frontier.

    Returns:
        Valid K=16 frontier made by twice refining four coarse regions while
        leaving the other four coarse.  It has the same K as the truth but
        deliberately different boundaries.
    """
    state = coarse_partition
    for node_id in coarse_partition.ordered_active()[:4]:
        first_child = tree.children(node_id)[0]
        state = state.split(tree, node_id)
        state = state.split(tree, first_child)
    if len(state.active) != 16:
        raise RuntimeError("Irregular benchmark construction did not produce K=16.")
    return state


def build_checkerboard_case(
    *,
    minimum_regions: int = 8,
    maximum_regions: int = 28,
    seed: int = 2016,
) -> CheckerboardCase:
    """Build smooth sensitivities and synthetic noisy observations.

    Args:
        minimum_regions: Smallest region count with positive prior mass.
        maximum_regions: Largest region count with positive prior mass.  The
            marginal prior is uniform over the inclusive range; conditional on
            K, every valid partition receives equal mass.
        seed: Seed controlling sensitivities and independent train/holdout
            observation noise.

    Returns:
        Deterministic Gaussian product-space benchmark case.

    Raises:
        ValueError: If the requested region-count interval is invalid.
    """
    if not 1 <= minimum_regions <= maximum_regions <= math.prod(_GRID_SHAPE):
        raise ValueError("region-count bounds must define a valid inclusive interval.")
    rows, columns = _GRID_SHAPE
    observation_count = _TRAIN_OBSERVATIONS + _HOLDOUT_OBSERVATIONS
    grid_coordinates = np.array(
        [(row, column) for row in range(rows) for column in range(columns)],
        dtype=float,
    )
    row_offsets = grid_coordinates[:, 0] - 3.5
    column_offsets = grid_coordinates[:, 1] - 3.5
    radial_distance = row_offsets * row_offsets + column_offsets * column_offsets
    prior_flux = 0.75 + 0.35 * np.exp(-radial_distance / 12.0) + 0.15 * grid_coordinates[:, 1] / 7.0

    rng = np.random.default_rng(seed)
    centres = np.column_stack(
        (
            rng.uniform(-0.25, 7.25, observation_count),
            rng.uniform(-0.25, 7.25, observation_count),
        )
    )
    widths = rng.uniform(0.55, 1.35, observation_count)
    amplitudes = rng.uniform(50.0, 130.0, observation_count)
    design = np.empty((observation_count, rows * columns), dtype=float)
    for observation in range(observation_count):
        squared_distance = np.sum(
            np.square(grid_coordinates - centres[observation]),
            axis=1,
        )
        weights = prior_flux * np.exp(-squared_distance / (2.0 * widths[observation] ** 2))
        design[observation] = amplitudes[observation] * weights / weights.sum()

    truth = checkerboard_truth()
    noiseless = design @ truth.reshape(-1)
    observations = noiseless + rng.normal(scale=_OBSERVATION_SD, size=observation_count)
    train_design = design[:_TRAIN_OBSERVATIONS].reshape(_TRAIN_OBSERVATIONS, rows, columns)
    holdout_design = design[_TRAIN_OBSERVATIONS:].reshape(_HOLDOUT_OBSERVATIONS, rows, columns)
    tree = DyadicTree.from_shape(_GRID_SHAPE)
    coarse_partition = regular_depth_partition(tree, 3)
    wrong_partition = irregular_same_count_partition(tree, coarse_partition)
    truth_partition = regular_depth_partition(tree, 4)
    partition_prior = RegionCountPartitionPrior.uniform_k(
        tree,
        minimum_regions=minimum_regions,
        maximum_regions=maximum_regions,
    )

    target = GaussianProductSpaceTarget.from_grid(
        observations=observations[:_TRAIN_OBSERVATIONS],
        observation_mean=train_design.reshape(_TRAIN_OBSERVATIONS, -1).sum(axis=1),
        inner_grid_design=train_design,
        tree=tree,
        observation_covariance=np.eye(_TRAIN_OBSERVATIONS) * _OBSERVATION_SD**2,
        inner_prior_scale=1.0,
        inactive_pseudo_prior_scale=1.0,
        partition_log_prior=partition_prior,
    )
    return CheckerboardCase(
        tree=tree,
        target=target,
        truth=truth,
        train_design=train_design,
        holdout_design=holdout_design,
        holdout_observations=observations[_TRAIN_OBSERVATIONS:],
        holdout_noiseless=noiseless[_TRAIN_OBSERVATIONS:],
        coarse_partition=coarse_partition,
        wrong_partition=wrong_partition,
        truth_partition=truth_partition,
    )


def sample_fixed_partition(
    target: GaussianProductSpaceTarget,
    partition: PartitionState,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[ProductSpaceState, ...]:
    """Draw independent exact Gaussian states for one fixed partition.

    Args:
        target: Gaussian target shared with the latent inversion.
        partition: Fixed valid frontier.
        draws: Positive number of posterior draws.
        rng: Caller-owned random generator.

    Returns:
        Independent conditional posterior states.
    """
    if draws < 1:
        raise ValueError("draws must be at least 1.")
    return tuple(target.draw_conditional_state(partition, rng) for _ in range(draws))


def summarize_states(
    case: CheckerboardCase,
    states: tuple[ProductSpaceState, ...],
) -> InversionMetrics:
    """Calculate holdout and native-grid metrics for posterior states.

    Args:
        case: Benchmark inputs and declared truth.
        states: Non-empty posterior state sequence.

    Returns:
        Prediction, reconstruction, contrast, and region-count metrics.
    """
    if not states:
        raise ValueError("states must not be empty.")
    grids = np.stack([_scaling_grid(case.target, state) for state in states])
    holdout_matrix = case.holdout_design.reshape(_HOLDOUT_OBSERVATIONS, -1)
    predictions = grids.reshape(len(states), -1) @ holdout_matrix.T
    posterior_prediction = predictions.mean(axis=0)
    posterior_grid = grids.mean(axis=0)
    low_mean = float(np.mean(posterior_grid[case.truth == 0.5]))
    high_mean = float(np.mean(posterior_grid[case.truth == 1.5]))
    region_counts = np.fromiter(
        (len(state.partition.active) for state in states),
        dtype=np.int64,
        count=len(states),
    )
    return InversionMetrics(
        prediction_rmse_ppb=float(np.sqrt(np.mean(np.square(posterior_prediction - case.holdout_noiseless)))),
        predictive_log_density=_posterior_predictive_log_density(
            case.holdout_observations,
            predictions,
            observation_sd=_OBSERVATION_SD,
        ),
        grid_rmse=float(np.sqrt(np.mean(np.square(posterior_grid - case.truth)))),
        recovered_checkerboard_contrast=high_mean - low_mean,
        mean_regions=float(region_counts.mean()),
        minimum_regions=int(region_counts.min()),
        maximum_regions=int(region_counts.max()),
    )


def run_benchmark(
    *,
    draws: int = 4_000,
    warmup: int = 2_000,
    minimum_regions: int = 8,
    maximum_regions: int = 28,
    sampler: str = "augmented",
    seed: int = 481,
    output_directory: Path | None = None,
) -> CheckerboardBenchmark:
    """Run matched fixed and latent checkerboard inversions.

    Args:
        draws: Retained draws for all three inversions.
        warmup: Discarded latent-chain transition cycles.
        minimum_regions: Smallest latent region count with positive prior mass.
        maximum_regions: Largest latent region count with positive prior mass.
        sampler: ``"augmented"`` for product-space MH-within-Gibbs or
            ``"collapsed"`` for exact marginal Gaussian partition MH.
        seed: Seed for case construction and independent posterior streams.
        output_directory: Optional directory for PNG plots and the JSON
            benchmark summary.

    Returns:
        Fixed/latent metrics and explicit comparison gates.
    """
    if sampler not in {"augmented", "collapsed"}:
        raise ValueError("sampler must be 'augmented' or 'collapsed'.")
    case = build_checkerboard_case(
        minimum_regions=minimum_regions,
        maximum_regions=maximum_regions,
        seed=2016,
    )
    seed_sequence = np.random.SeedSequence(seed)
    fixed_truth_rng, fixed_wrong_rng, fixed_coarse_rng, latent_rng = (
        np.random.default_rng(child) for child in seed_sequence.spawn(4)
    )
    fixed_truth_states = sample_fixed_partition(
        case.target,
        case.truth_partition,
        draws=draws,
        rng=fixed_truth_rng,
    )
    fixed_coarse_states = sample_fixed_partition(
        case.target,
        case.coarse_partition,
        draws=draws,
        rng=fixed_coarse_rng,
    )
    fixed_wrong_states = sample_fixed_partition(
        case.target,
        case.wrong_partition,
        draws=draws,
        rng=fixed_wrong_rng,
    )
    started = perf_counter()
    sampler_function = (
        sample_gaussian_product_space if sampler == "augmented" else sample_collapsed_gaussian_product_space
    )
    latent_trace = sampler_function(
        case.target,
        case.coarse_partition,
        draws=draws,
        warmup=warmup,
        rng=latent_rng,
    )
    latent_runtime = perf_counter() - started
    latent_states = tuple(latent_trace.state(draw) for draw in range(latent_trace.draw_count))

    fixed_truth = summarize_states(case, fixed_truth_states)
    fixed_wrong = summarize_states(case, fixed_wrong_states)
    fixed_coarse = summarize_states(case, fixed_coarse_states)
    latent = summarize_states(case, latent_states)
    log_score_difference_per_observation = (
        latent.predictive_log_density - fixed_truth.predictive_log_density
    ) / _HOLDOUT_OBSERVATIONS
    benchmark = CheckerboardBenchmark(
        fixed_truth=fixed_truth,
        fixed_wrong=fixed_wrong,
        fixed_coarse=fixed_coarse,
        latent=latent,
        latent_sampler=sampler,
        latent_partition_acceptance_rate=latent_trace.partition_acceptance_rate,
        latent_split_acceptance_rate=latent_trace.move_acceptance_rate("split"),
        latent_merge_acceptance_rate=latent_trace.move_acceptance_rate("merge"),
        latent_warmup_acceptance_rate=latent_trace.warmup_acceptance_rate,
        latent_unique_partitions=len(set(latent_trace.partitions)),
        latent_runtime_seconds=latent_runtime,
        latent_beats_wrong_prediction_rmse=(latent.prediction_rmse_ppb < fixed_wrong.prediction_rmse_ppb),
        latent_beats_coarse_prediction_rmse=(latent.prediction_rmse_ppb < fixed_coarse.prediction_rmse_ppb),
        latent_oracle_log_score_difference_per_observation=log_score_difference_per_observation,
        latent_oracle_log_score_noninferior=log_score_difference_per_observation >= -0.05,
    )
    if output_directory is not None:
        _write_benchmark_outputs(
            case,
            fixed_truth_states=fixed_truth_states,
            fixed_wrong_states=fixed_wrong_states,
            fixed_coarse_states=fixed_coarse_states,
            latent_states=latent_states,
            latent_trace=latent_trace,
            benchmark=benchmark,
            output_directory=output_directory,
        )
    return benchmark


def _write_benchmark_outputs(
    case: CheckerboardCase,
    *,
    fixed_truth_states: tuple[ProductSpaceState, ...],
    fixed_wrong_states: tuple[ProductSpaceState, ...],
    fixed_coarse_states: tuple[ProductSpaceState, ...],
    latent_states: tuple[ProductSpaceState, ...],
    latent_trace: Any,
    benchmark: CheckerboardBenchmark,
    output_directory: Path,
) -> None:
    """Write reproducible checkerboard recovery and diagnostic artifacts.

    Args:
        case: Complete synthetic checkerboard case.
        fixed_truth_states: Posterior states for the oracle partition.
        fixed_wrong_states: Posterior states for the misspecified K=16 partition.
        fixed_coarse_states: Posterior states for the coarse K=8 partition.
        latent_states: Retained states for latent K and P.
        latent_trace: Sampler trace supplying retained region counts.
        benchmark: Scalar benchmark summary written beside the plots.
        output_directory: Destination directory, created when needed.
    """
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    posterior_grids = {
        "Oracle fixed K=16": _posterior_mean_grid(case, fixed_truth_states),
        "Wrong fixed K=16": _posterior_mean_grid(case, fixed_wrong_states),
        "Coarse fixed K=8": _posterior_mean_grid(case, fixed_coarse_states),
        "Latent K and P": _posterior_mean_grid(case, latent_states),
    }

    figure, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), constrained_layout=True)
    maps = (
        ("Planted truth", case.truth),
        *posterior_grids.items(),
        ("Latent minus truth", posterior_grids["Latent K and P"] - case.truth),
    )
    shared_image = None
    error_image = None
    for axis, (title, values) in zip(axes.flat, maps, strict=True):
        if title == "Latent minus truth":
            limit = max(0.05, float(np.max(np.abs(values))))
            error_image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit)
        else:
            shared_image = axis.imshow(values, cmap="viridis", vmin=0.35, vmax=1.65)
        axis.set_title(title)
        axis.set_xticks(range(_GRID_SHAPE[1]))
        axis.set_yticks(range(_GRID_SHAPE[0]))
        axis.set_xlabel("Grid column")
        axis.set_ylabel("Grid row")
    if shared_image is None or error_image is None:
        raise RuntimeError("Checkerboard plot construction did not create both color scales.")
    figure.colorbar(shared_image, ax=axes[:, :2], shrink=0.85, label="Scaling factor")
    figure.colorbar(error_image, ax=axes[:, 2], shrink=0.85, label="Scaling-factor error")
    figure.suptitle("Lunt-inspired Gaussian checkerboard recovery")
    figure.savefig(output_directory / "checkerboard_recovery.png", dpi=180)
    plt.close(figure)

    region_counts = np.fromiter(
        (len(partition.active) for partition in latent_trace.partitions),
        dtype=np.int64,
        count=len(latent_trace.partitions),
    )
    holdout_matrix = case.holdout_design.reshape(_HOLDOUT_OBSERVATIONS, -1)
    latent_predictions = posterior_grids["Latent K and P"].reshape(-1) @ holdout_matrix.T
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    count_values, count_frequencies = np.unique(region_counts, return_counts=True)
    axes[0].bar(count_values, count_frequencies / count_frequencies.sum(), color="#287271")
    axes[0].axvline(16, color="#c44536", linestyle="--", label="Truth K=16")
    axes[0].set(xlabel="Retained K", ylabel="Posterior frequency", title="Latent region count")
    axes[0].legend(frameon=False)

    order = np.argsort(case.holdout_noiseless)
    axes[1].plot(case.holdout_noiseless[order], color="#222222", label="Noiseless truth")
    axes[1].plot(latent_predictions[order], color="#287271", label="Latent posterior mean")
    axes[1].set(xlabel="Holdout row (sorted)", ylabel="Signal", title="Holdout prediction")
    axes[1].legend(frameon=False)

    names = ("Oracle", "Wrong K=16", "Coarse K=8", "Latent")
    values = (
        benchmark.fixed_truth.prediction_rmse_ppb,
        benchmark.fixed_wrong.prediction_rmse_ppb,
        benchmark.fixed_coarse.prediction_rmse_ppb,
        benchmark.latent.prediction_rmse_ppb,
    )
    axes[2].bar(names, values, color=("#4c956c", "#d68c45", "#9c6644", "#287271"))
    axes[2].set(ylabel="Holdout RMSE", title="Predictive comparison")
    axes[2].tick_params(axis="x", rotation=25)
    figure.suptitle("Checkerboard partition diagnostics")
    figure.savefig(output_directory / "checkerboard_diagnostics.png", dpi=180)
    plt.close(figure)

    (output_directory / "checkerboard_summary.json").write_text(
        json.dumps(benchmark.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _posterior_mean_grid(
    case: CheckerboardCase,
    states: tuple[ProductSpaceState, ...],
) -> np.ndarray:
    """Return the posterior mean native-grid scaling field."""
    if not states:
        raise ValueError("states must not be empty.")
    return np.mean(
        np.stack([_scaling_grid(case.target, state) for state in states]),
        axis=0,
    )


def _scaling_grid(
    target: GaussianProductSpaceTarget,
    state: ProductSpaceState,
) -> np.ndarray:
    """Decode one product-space state to native-grid scaling factors."""
    regional_anomalies = target.contrast_layout.decode(
        state.partition,
        state.inner_coordinates,
    )
    grid = np.empty(target.tree.shape, dtype=float)
    for anomaly, node_id in zip(
        regional_anomalies,
        state.partition.ordered_active(),
        strict=True,
    ):
        tile = target.tree.tile(node_id)
        grid[tile.row_start : tile.row_stop, tile.col_start : tile.col_stop] = 1.0 + anomaly
    return grid


def _posterior_predictive_log_density(
    observations: np.ndarray,
    predictions: np.ndarray,
    *,
    observation_sd: float,
) -> float:
    """Return summed pointwise log density of a sampled Gaussian mixture."""
    standardized = (observations[np.newaxis, :] - predictions) / observation_sd
    component_log_density = (
        -0.5 * np.square(standardized) - math.log(observation_sd) - 0.5 * math.log(2.0 * math.pi)
    )
    maximum = component_log_density.max(axis=0)
    pointwise = maximum + np.log(np.mean(np.exp(component_log_density - maximum[np.newaxis, :]), axis=0))
    return float(pointwise.sum())


def _parse_args() -> argparse.Namespace:
    """Parse command-line controls for the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=4_000)
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--minimum-regions", type=int, default=8)
    parser.add_argument("--maximum-regions", type=int, default=28)
    parser.add_argument("--sampler", choices=("augmented", "collapsed"), default="augmented")
    parser.add_argument("--seed", type=int, default=481)
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and print a JSON summary."""
    args = _parse_args()
    result = run_benchmark(
        draws=args.draws,
        warmup=args.warmup,
        minimum_regions=args.minimum_regions,
        maximum_regions=args.maximum_regions,
        sampler=args.sampler,
        seed=args.seed,
        output_directory=args.output_directory,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
