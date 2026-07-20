"""Run positive latent partition inference with TAC/MHD and InTEM groups.

This experiment is the first data-backed scale-up of the Gamma--Beta product
space.  It uses committed TAC/MHD footprint-times-flux sensitivities but
generates emissions-only synthetic observations, so known-corrupt boundary
condition fixtures are neither required nor opened.  Six InTEM outer groups
remain fixed regions; inner land and ocean have separate masked candidate trees
and root priors.

The maximum topology is calibrated with the existing controlled Gamma--Beta
policy and a 50% UK country-total prior uncertainty target.  A conservative
high-level split of the main inner-land component defines the true field.  The
command compares latent K/P with the true fixed partition and an underfit fixed
root partition.  Pass ``--output-directory`` to save recovery maps, sampler
diagnostics, and a machine-readable summary.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pymc as pm

from dyadic_gamma_beta_calibration import build_calibrated_case
from openghg_inversions.basis.experimental.dyadic.demo_data import (
    DemoDesignData,
    load_tac_mhd_demo_data,
)
from openghg_inversions.basis.experimental.dyadic.gamma_beta_coordinates import (
    GammaBetaCoordinateLayout,
)
from openghg_inversions.basis.experimental.dyadic.gamma_beta_partition import (
    GammaBetaPartitionLayout,
    GammaBetaRegionCountPrior,
)
from openghg_inversions.basis.experimental.dyadic.gamma_beta_product_space import (
    GammaBetaProductSpaceTarget,
)
from openghg_inversions.basis.experimental.dyadic.pymc_gamma_beta_product_space import (
    build_pymc_gamma_beta_product_space_model,
)

_LAND_GROUP_NAME = "inner_land"
_TRUTH_FIRST_CHILD_SCALING = 0.85


@dataclass(frozen=True, slots=True, eq=False)
class IntemGammaBetaRecoveryCase:
    """Complete synthetic TAC/MHD positive partition-recovery case."""

    data: DemoDesignData
    coordinate_layout: GammaBetaCoordinateLayout
    partition_layout: GammaBetaPartitionLayout
    train_target: GammaBetaProductSpaceTarget
    holdout_target: GammaBetaProductSpaceTarget
    latent_prior: GammaBetaRegionCountPrior
    fixed_underfit_prior: GammaBetaRegionCountPrior
    fixed_truth_prior: GammaBetaRegionCountPrior
    truth_split_mask: npt.NDArray[np.bool_]
    truth_node_id: int
    truth_field: npt.NDArray[np.float64]
    train_indices: npt.NDArray[np.int64]
    holdout_indices: npt.NDArray[np.int64]
    train_noiseless: npt.NDArray[np.float64]
    holdout_noiseless: npt.NDArray[np.float64]
    k_continuation_probability: float


@dataclass(frozen=True, slots=True)
class IntemGammaBetaFitSummary:
    """Posterior and sampler diagnostics for one InTEM-grouped fit."""

    name: str
    draws: int
    tune: int
    mean_k: float
    minimum_k: int
    maximum_k: int
    truth_partition_draw_frequency: float
    unique_partitions: int
    full_field_rmse: float
    inner_land_field_rmse: float
    holdout_prediction_rmse: float
    holdout_log_predictive_density: float
    partition_acceptance_rate: float
    divergence_count: int


@dataclass(frozen=True, slots=True)
class IntemGammaBetaRecoveryBenchmark:
    """Latent, true-fixed, and underfit-fixed data-backed comparison."""

    observation_count: int
    train_count: int
    holdout_count: int
    grid_shape: tuple[int, int]
    forest_node_count: int
    component_root_count: int
    maximum_leaf_count: int
    possible_split_count: int
    truth_k: int
    k_continuation_probability: float
    latent: IntemGammaBetaFitSummary
    fixed_true: IntemGammaBetaFitSummary
    fixed_underfit: IntemGammaBetaFitSummary
    latent_matches_fixed_true: bool
    latent_beats_fixed_underfit: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable benchmark report."""
        return asdict(self)


@dataclass(frozen=True, slots=True, eq=False)
class _IntemGammaBetaFit:
    """Scalar summary plus posterior arrays needed for visual diagnostics."""

    summary: IntemGammaBetaFitSummary
    posterior_mean_field: npt.NDArray[np.float64]
    posterior_mean_holdout: npt.NDArray[np.float64]
    region_counts: npt.NDArray[np.int64]


def build_case(
    *,
    data_directory: Path = Path("tests/data"),
    design_data: DemoDesignData | None = None,
    inner_regions: int = 32,
    data_seed: int = 1701,
    k_continuation_probability: float = 0.5,
) -> IntemGammaBetaRecoveryCase:
    """Build the calibrated InTEM forest and synthetic observation targets.

    Args:
        data_directory: Directory containing committed inversion fixtures.
        design_data: Optional preloaded 47-row design. Supplying this is useful
            for leakage-regression tests; topology still uses training rows only.
        inner_regions: Candidate terminal-region budget across inner land/ocean.
        data_seed: Seed for independent synthetic observation errors.
        k_continuation_probability: Truncated geometric prior continuation
            probability for each additional split.

    Returns:
        Matched targets, partition priors, and conservative split truth.
    """
    data = load_tac_mhd_demo_data(data_directory) if design_data is None else design_data
    holdout = np.arange(data.y.size) % 3 == 2
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    training_data = replace(
        data,
        G=data.G[train_indices],
        y=data.y[train_indices],
        error=data.error[train_indices],
        min_error=data.min_error[train_indices],
        sites=data.sites[train_indices],
        times=data.times[train_indices],
    )
    calibrated = build_calibrated_case(
        data_directory=data_directory,
        data=training_data,
        topology_weight_mode="sensitivity",
        target_relative_standard_deviation=0.5,
        draws=2,
        inner_regions=inner_regions,
        max_depth=8,
        seed=20260718,
    )
    prior_case = calibrated.case
    forest = prior_case.forest
    coordinate_layout = GammaBetaCoordinateLayout.from_forest(
        forest,
        kappa_strategy=prior_case.strategy,
    )
    partition_layout = GammaBetaPartitionLayout.from_forest(forest)
    truth_node_id = _truth_land_root(forest)
    truth_mask = np.zeros(partition_layout.split_count, dtype=np.bool_)
    truth_mask[partition_layout.split_index_by_node[truth_node_id]] = True
    truth_mask = partition_layout.canonical_split_mask(truth_mask)

    group_roots = np.ones(len(forest.groups), dtype=np.float64)
    fractions = coordinate_layout.expected_fraction_by_split.copy()
    truth_split_index = coordinate_layout.internal_node_ids.index(truth_node_id)
    fractions[truth_split_index] *= _TRUTH_FIRST_CHILD_SCALING
    node_scalings = coordinate_layout.node_scalings(group_roots, fractions)
    truth_field = coordinate_layout.render_frontier_scalings(
        partition_layout.active_node_ids(truth_mask),
        node_scalings,
    )
    noiseless = np.einsum("oij,ij->o", data.G, truth_field, optimize=True)
    observations = noiseless + np.random.default_rng(data_seed).normal(
        scale=data.error,
        size=noiseless.size,
    )
    train_target = GammaBetaProductSpaceTarget.from_grid(
        observations=observations[train_indices],
        finest_grid_design=data.G[train_indices],
        forest=forest,
        kappa_strategy=prior_case.strategy,
        observation_sd=data.error[train_indices],
    )
    holdout_target = GammaBetaProductSpaceTarget.from_grid(
        observations=observations[holdout_indices],
        finest_grid_design=data.G[holdout_indices],
        forest=forest,
        kappa_strategy=prior_case.strategy,
        observation_sd=data.error[holdout_indices],
    )
    truth_k = partition_layout.region_count(truth_mask)
    minimum_k = partition_layout.minimum_regions
    return IntemGammaBetaRecoveryCase(
        data=data,
        coordinate_layout=coordinate_layout,
        partition_layout=partition_layout,
        train_target=train_target,
        holdout_target=holdout_target,
        latent_prior=GammaBetaRegionCountPrior.geometric_extra_regions(
            partition_layout,
            continuation_probability=k_continuation_probability,
        ),
        fixed_underfit_prior=GammaBetaRegionCountPrior.uniform_k(
            partition_layout,
            minimum_regions=minimum_k,
            maximum_regions=minimum_k,
        ),
        fixed_truth_prior=GammaBetaRegionCountPrior.uniform_k(
            partition_layout,
            minimum_regions=truth_k,
            maximum_regions=truth_k,
        ),
        truth_split_mask=truth_mask,
        truth_node_id=truth_node_id,
        truth_field=_frozen(truth_field),
        train_indices=_frozen_integer(train_indices),
        holdout_indices=_frozen_integer(holdout_indices),
        train_noiseless=_frozen(noiseless[train_indices]),
        holdout_noiseless=_frozen(noiseless[holdout_indices]),
        k_continuation_probability=k_continuation_probability,
    )


def run_benchmark(
    case: IntemGammaBetaRecoveryCase,
    *,
    draws: int = 1_000,
    tune: int = 1_000,
    sampling_seed: int = 20260719,
    target_accept: float = 0.95,
    output_directory: Path | None = None,
) -> IntemGammaBetaRecoveryBenchmark:
    """Run latent, true-fixed, and underfit-fixed compound chains.

    Args:
        case: Data-backed case from :func:`build_case`.
        draws: Retained draws per fit.
        tune: NUTS tuning draws per fit.
        sampling_seed: Base seed with deterministic comparator offsets.
        target_accept: NUTS target acceptance probability.
        output_directory: Optional directory for PNG plots and the JSON
            benchmark summary.

    Returns:
        Complete matched recovery comparison.
    """
    layout = case.partition_layout
    latent_result = _sample_fit(
        case,
        name="latent_k_p",
        prior=case.latent_prior,
        initial_mask=layout.initial_split_mask(layout.minimum_regions),
        draws=draws,
        tune=tune,
        seed=sampling_seed,
        target_accept=target_accept,
        include_swap_moves=True,
        fixed_split_mask=None,
    )
    fixed_true_result = _sample_fit(
        case,
        name="fixed_true_partition",
        prior=case.fixed_truth_prior,
        initial_mask=case.truth_split_mask,
        draws=draws,
        tune=tune,
        seed=sampling_seed + 1,
        target_accept=target_accept,
        include_swap_moves=False,
        fixed_split_mask=case.truth_split_mask,
    )
    fixed_underfit_result = _sample_fit(
        case,
        name="fixed_underfit_roots",
        prior=case.fixed_underfit_prior,
        initial_mask=layout.initial_split_mask(layout.minimum_regions),
        draws=draws,
        tune=tune,
        seed=sampling_seed + 2,
        target_accept=target_accept,
        include_swap_moves=False,
        fixed_split_mask=layout.initial_split_mask(layout.minimum_regions),
    )
    latent = latent_result.summary
    fixed_true = fixed_true_result.summary
    fixed_underfit = fixed_underfit_result.summary
    benchmark = IntemGammaBetaRecoveryBenchmark(
        observation_count=case.data.y.size,
        train_count=case.train_indices.size,
        holdout_count=case.holdout_indices.size,
        grid_shape=case.coordinate_layout.forest.shape,
        forest_node_count=len(case.coordinate_layout.forest.nodes),
        component_root_count=layout.minimum_regions,
        maximum_leaf_count=layout.maximum_regions,
        possible_split_count=layout.split_count,
        truth_k=layout.region_count(case.truth_split_mask),
        k_continuation_probability=case.k_continuation_probability,
        latent=latent,
        fixed_true=fixed_true,
        fixed_underfit=fixed_underfit,
        latent_matches_fixed_true=(
            latent.holdout_prediction_rmse <= fixed_true.holdout_prediction_rmse + 0.25
            and latent.inner_land_field_rmse <= fixed_true.inner_land_field_rmse + 0.1
        ),
        latent_beats_fixed_underfit=(
            latent.holdout_prediction_rmse < fixed_underfit.holdout_prediction_rmse
            and latent.inner_land_field_rmse < fixed_underfit.inner_land_field_rmse
        ),
    )
    if output_directory is not None:
        _write_benchmark_outputs(
            case,
            latent=latent_result,
            fixed_true=fixed_true_result,
            fixed_underfit=fixed_underfit_result,
            benchmark=benchmark,
            output_directory=output_directory,
        )
    return benchmark


def _sample_fit(
    case: IntemGammaBetaRecoveryCase,
    *,
    name: str,
    prior: GammaBetaRegionCountPrior,
    initial_mask: npt.ArrayLike,
    draws: int,
    tune: int,
    seed: int,
    target_accept: float,
    include_swap_moves: bool,
    fixed_split_mask: npt.ArrayLike | None,
) -> _IntemGammaBetaFit:
    """Run and summarize one data-backed compound chain.

    Args:
        case: Shared synthetic observation and candidate-forest case.
        name: Stable label for the fit summary.
        prior: Partition prior used by the structural step.
        initial_mask: Canonical initial partition mask.
        draws: Number of retained posterior draws.
        tune: Number of NUTS tuning draws.
        seed: Random seed shared by PyMC and the partition step.
        target_accept: NUTS target acceptance probability.
        include_swap_moves: Allow fixed-K partition relocation proposals.
        fixed_split_mask: Optional exact point-mass partition for a comparator.

    Returns:
        Posterior recovery and sampler diagnostics for the fit.
    """
    adapter = build_pymc_gamma_beta_product_space_model(
        case.train_target,
        prior,
        initial_split_mask=initial_mask,
        fixed_split_mask=fixed_split_mask,
    )
    steps = adapter.step_methods(
        partition_rng=seed,
        include_swap_moves=include_swap_moves,
        nuts_kwargs={"target_accept": target_accept},
    )
    with adapter.model:
        trace: Any = pm.sample(
            draws=draws,
            tune=tune,
            chains=1,
            cores=1,
            step=list(steps),
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
            return_inferencedata=True,
        )

    layout = case.partition_layout
    masks = np.asarray(trace.posterior["split_mask"]).reshape(-1, layout.split_count)
    node_scalings = np.asarray(trace.posterior["node_scalings"]).reshape(
        -1,
        len(layout.forest.nodes),
    )
    holdout_predictions = np.empty(
        (masks.shape[0], case.holdout_indices.size),
        dtype=np.float64,
    )
    field_sum = np.zeros(layout.forest.shape, dtype=np.float64)
    mask_keys: set[bytes] = set()
    truth_count = 0
    for draw_index, (mask, scalings) in enumerate(zip(masks, node_scalings, strict=True)):
        canonical = layout.canonical_split_mask(mask)
        active = layout.active_node_ids(canonical)
        active_indices = np.asarray(active, dtype=np.int64)
        holdout_predictions[draw_index] = (
            case.holdout_target.observation_mean
            + case.holdout_target.node_design[:, active_indices] @ scalings[active_indices]
        )
        field_sum += case.coordinate_layout.render_frontier_scalings(active, scalings)
        key = canonical.tobytes()
        mask_keys.add(key)
        truth_count += int(np.array_equal(canonical, case.truth_split_mask))

    posterior_mean_field = field_sum / masks.shape[0]
    posterior_mean_prediction = holdout_predictions.mean(axis=0)
    land_mask = next(group.mask for group in layout.forest.groups if group.name == _LAND_GROUP_NAME)
    region_counts = layout.minimum_regions + masks.sum(axis=1)
    if fixed_split_mask is not None:
        expected_mask = layout.canonical_split_mask(fixed_split_mask)
        if not np.all(masks == expected_mask):
            raise RuntimeError("A fixed-partition comparator changed its split mask.")
    return _IntemGammaBetaFit(
        summary=IntemGammaBetaFitSummary(
            name=name,
            draws=masks.shape[0],
            tune=tune,
            mean_k=float(region_counts.mean()),
            minimum_k=int(region_counts.min()),
            maximum_k=int(region_counts.max()),
            truth_partition_draw_frequency=truth_count / masks.shape[0],
            unique_partitions=len(mask_keys),
            full_field_rmse=_rmse(posterior_mean_field, case.truth_field),
            inner_land_field_rmse=_rmse(
                posterior_mean_field[land_mask],
                case.truth_field[land_mask],
            ),
            holdout_prediction_rmse=_rmse(
                posterior_mean_prediction,
                case.holdout_noiseless,
            ),
            holdout_log_predictive_density=_independent_normal_log_predictive_density(
                observations=case.holdout_target.observations,
                predictions=holdout_predictions,
                standard_deviations=np.asarray(
                    case.data.error[case.holdout_indices],
                    dtype=np.float64,
                ),
            ),
            partition_acceptance_rate=_partition_acceptance_rate(trace),
            divergence_count=int(np.asarray(trace.sample_stats["diverging"]).sum()),
        ),
        posterior_mean_field=_frozen(posterior_mean_field),
        posterior_mean_holdout=_frozen(posterior_mean_prediction),
        region_counts=_frozen_integer(region_counts),
    )


def _write_benchmark_outputs(
    case: IntemGammaBetaRecoveryCase,
    *,
    latent: _IntemGammaBetaFit,
    fixed_true: _IntemGammaBetaFit,
    fixed_underfit: _IntemGammaBetaFit,
    benchmark: IntemGammaBetaRecoveryBenchmark,
    output_directory: Path,
) -> None:
    """Write recovery maps, posterior diagnostics, and scalar JSON results.

    Args:
        case: Shared synthetic TAC/MHD recovery case.
        latent: Latent K/P posterior result.
        fixed_true: Oracle fixed-partition posterior result.
        fixed_underfit: Root-only fixed-partition posterior result.
        benchmark: Scalar comparison summary.
        output_directory: Destination directory, created when needed.
    """
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    latitude = case.data.lat
    longitude = case.data.lon
    sensitivity_weight = np.mean(
        np.abs(case.data.G[case.train_indices]),
        axis=0,
    )
    positive_weight = sensitivity_weight[sensitivity_weight > 0.0]
    if positive_weight.size == 0:
        raise ValueError("Expected at least one positive training sensitivity.")
    weight_floor = float(np.quantile(positive_weight, 0.01))
    field_values = (
        case.truth_field,
        latent.posterior_mean_field,
        fixed_true.posterior_mean_field,
        fixed_underfit.posterior_mean_field,
    )
    field_min = min(float(np.min(values)) for values in field_values)
    field_max = max(float(np.max(values)) for values in field_values)

    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.6), constrained_layout=True)
    weight_image = axes[0, 0].pcolormesh(
        longitude,
        latitude,
        np.log10(np.maximum(sensitivity_weight, weight_floor)),
        shading="auto",
        cmap="magma",
    )
    axes[0, 0].set_title("Mean absolute training sensitivity (1% floor)")
    figure.colorbar(weight_image, ax=axes[0, 0], label="log10 sensitivity")

    field_panels = (
        (axes[0, 1], "Planted truth", case.truth_field),
        (axes[0, 2], "Latent K and P", latent.posterior_mean_field),
        (axes[1, 0], "Fixed planted partition", fixed_true.posterior_mean_field),
        (axes[1, 1], "Fixed root-only partition", fixed_underfit.posterior_mean_field),
    )
    field_image = None
    for axis, title, values in field_panels:
        field_image = axis.pcolormesh(
            longitude,
            latitude,
            values,
            shading="auto",
            cmap="viridis",
            vmin=field_min,
            vmax=field_max,
        )
        axis.set_title(title)
    if field_image is None:
        raise RuntimeError("Recovery plot construction did not create a field scale.")
    figure.colorbar(
        field_image,
        ax=(axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]),
        location="bottom",
        shrink=0.8,
        pad=0.08,
        label="Scaling factor",
    )

    error = latent.posterior_mean_field - case.truth_field
    error_limit = max(0.05, float(np.max(np.abs(error))))
    error_image = axes[1, 2].pcolormesh(
        longitude,
        latitude,
        error,
        shading="auto",
        cmap="RdBu_r",
        vmin=-error_limit,
        vmax=error_limit,
    )
    axes[1, 2].set_title("Latent posterior mean minus truth")
    figure.colorbar(error_image, ax=axes[1, 2], label="Scaling-factor error")
    for axis in axes.flat:
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
    figure.suptitle("TAC/MHD InTEM Gamma-Beta product-space recovery")
    figure.savefig(output_directory / "gamma_beta_intem_recovery_maps.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    k_values, k_frequencies = np.unique(latent.region_counts, return_counts=True)
    axes[0].bar(k_values, k_frequencies / k_frequencies.sum(), color="#287271")
    axes[0].axvline(benchmark.truth_k, color="#c44536", linestyle="--", label=f"Truth K={benchmark.truth_k}")
    axes[0].set(xlabel="Retained K", ylabel="Posterior frequency", title="Latent region count")
    axes[0].legend(frameon=False)

    order = np.argsort(case.holdout_noiseless)
    axes[1].plot(case.holdout_noiseless[order], color="#222222", label="Noiseless truth")
    axes[1].plot(latent.posterior_mean_holdout[order], color="#287271", label="Latent")
    axes[1].plot(fixed_true.posterior_mean_holdout[order], color="#4c956c", label="Fixed planted")
    axes[1].plot(fixed_underfit.posterior_mean_holdout[order], color="#9c6644", label="Fixed roots")
    axes[1].set(xlabel="Holdout row (sorted)", ylabel="Signal", title="Holdout prediction")
    axes[1].legend(frameon=False)

    names = ("Latent", "Fixed planted", "Fixed roots")
    values = (
        latent.summary.inner_land_field_rmse,
        fixed_true.summary.inner_land_field_rmse,
        fixed_underfit.summary.inner_land_field_rmse,
    )
    axes[2].bar(names, values, color=("#287271", "#4c956c", "#9c6644"))
    axes[2].set(ylabel="Inner-land RMSE", title="Field recovery")
    axes[2].tick_params(axis="x", rotation=25)
    figure.suptitle("Gamma-Beta product-space diagnostics")
    figure.savefig(output_directory / "gamma_beta_intem_diagnostics.png", dpi=180)
    plt.close(figure)

    (output_directory / "gamma_beta_intem_summary.json").write_text(
        json.dumps(benchmark.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _truth_land_root(forest: Any) -> int:
    """Return the highest-weight refinable inner-land component root."""
    candidates = tuple(
        root_id
        for root_id in forest.root_ids
        if forest.groups[forest.nodes[root_id].group_index].name == _LAND_GROUP_NAME
        and forest.nodes[root_id].child_ids
    )
    if not candidates:
        raise ValueError("Expected at least one refinable inner-land component root.")
    return max(
        candidates,
        key=lambda node_id: (
            forest.nodes[node_id].partition_weight,
            forest.nodes[node_id].expected_mass,
            -node_id,
        ),
    )


def _independent_normal_log_predictive_density(
    *,
    observations: npt.NDArray[np.float64],
    predictions: npt.NDArray[np.float64],
    standard_deviations: npt.NDArray[np.float64],
) -> float:
    """Return summed posterior-draw mixture density for holdout rows."""
    residual = observations[np.newaxis, :] - predictions
    draw_log_density = (
        -0.5 * np.square(residual / standard_deviations)
        - np.log(standard_deviations)
        - 0.5 * math.log(2.0 * math.pi)
    )
    maxima = draw_log_density.max(axis=0)
    return float(np.sum(maxima + np.log(np.mean(np.exp(draw_log_density - maxima), axis=0))))


def _partition_acceptance_rate(trace: Any) -> float:
    """Extract the single custom structural acceptance statistic."""
    names = tuple(str(name) for name in trace.sample_stats.data_vars)
    accepted_names = tuple(name for name in names if name == "accepted" or name.endswith("_accepted"))
    if len(accepted_names) != 1:
        raise ValueError(f"Expected one accepted statistic, found {accepted_names!r}.")
    return float(np.asarray(trace.sample_stats[accepted_names[0]]).mean())


def _rmse(first: npt.ArrayLike, second: npt.ArrayLike) -> float:
    """Return root mean square difference between equal-shaped arrays."""
    difference = np.asarray(first, dtype=np.float64) - np.asarray(
        second,
        dtype=np.float64,
    )
    return float(np.sqrt(np.mean(np.square(difference))))


def _frozen(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a read-only float64 copy."""
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _frozen_integer(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Return a read-only int64 copy."""
    result = np.asarray(values, dtype=np.int64).copy()
    result.setflags(write=False)
    return result


def main(arguments: list[str] | None = None) -> int:
    """Run the data-backed benchmark and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("tests/data"))
    parser.add_argument("--inner-regions", type=int, default=32)
    parser.add_argument("--draws", type=int, default=1_000)
    parser.add_argument("--tune", type=int, default=1_000)
    parser.add_argument("--data-seed", type=int, default=1701)
    parser.add_argument("--k-continuation-probability", type=float, default=0.5)
    parser.add_argument("--sampling-seed", type=int, default=20260719)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--indent", type=int, default=2)
    parsed = parser.parse_args(arguments)
    benchmark = run_benchmark(
        build_case(
            data_directory=parsed.data_directory,
            inner_regions=parsed.inner_regions,
            data_seed=parsed.data_seed,
            k_continuation_probability=parsed.k_continuation_probability,
        ),
        draws=parsed.draws,
        tune=parsed.tune,
        sampling_seed=parsed.sampling_seed,
        target_accept=parsed.target_accept,
        output_directory=parsed.output_directory,
    )
    print(json.dumps(benchmark.as_dict(), indent=parsed.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
