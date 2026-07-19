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
root partition.  It prints JSON and writes no data products.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
    truth_partition_probability: float
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


def build_case(
    *,
    data_directory: Path = Path("tests/data"),
    inner_regions: int = 32,
    data_seed: int = 1701,
    k_continuation_probability: float = 0.5,
) -> IntemGammaBetaRecoveryCase:
    """Build the calibrated InTEM forest and synthetic observation targets.

    Args:
        data_directory: Directory containing committed inversion fixtures.
        inner_regions: Candidate terminal-region budget across inner land/ocean.
        data_seed: Seed for independent synthetic observation errors.
        k_continuation_probability: Truncated geometric prior continuation
            probability for each additional split.

    Returns:
        Matched targets, partition priors, and conservative split truth.
    """
    calibrated = build_calibrated_case(
        data_directory=data_directory,
        topology_weight_mode="sensitivity",
        target_relative_standard_deviation=0.5,
        draws=2,
        inner_regions=inner_regions,
        max_depth=8,
        seed=20260718,
    )
    prior_case = calibrated.case
    forest = prior_case.forest
    data = load_tac_mhd_demo_data(data_directory)
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
    holdout = np.arange(observations.size) % 3 == 2
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
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
) -> IntemGammaBetaRecoveryBenchmark:
    """Run latent, true-fixed, and underfit-fixed compound chains.

    Args:
        case: Data-backed case from :func:`build_case`.
        draws: Retained draws per fit.
        tune: NUTS tuning draws per fit.
        sampling_seed: Base seed with deterministic comparator offsets.
        target_accept: NUTS target acceptance probability.

    Returns:
        Complete matched recovery comparison.
    """
    layout = case.partition_layout
    latent = _sample_fit(
        case,
        name="latent_k_p",
        prior=case.latent_prior,
        initial_mask=layout.initial_split_mask(layout.minimum_regions),
        draws=draws,
        tune=tune,
        seed=sampling_seed,
        target_accept=target_accept,
        include_swap_moves=True,
    )
    fixed_true = _sample_fit(
        case,
        name="fixed_true_partition",
        prior=case.fixed_truth_prior,
        initial_mask=case.truth_split_mask,
        draws=draws,
        tune=tune,
        seed=sampling_seed + 1,
        target_accept=target_accept,
        include_swap_moves=False,
    )
    fixed_underfit = _sample_fit(
        case,
        name="fixed_underfit_roots",
        prior=case.fixed_underfit_prior,
        initial_mask=layout.initial_split_mask(layout.minimum_regions),
        draws=draws,
        tune=tune,
        seed=sampling_seed + 2,
        target_accept=target_accept,
        include_swap_moves=False,
    )
    return IntemGammaBetaRecoveryBenchmark(
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
            latent.holdout_prediction_rmse
            <= fixed_true.holdout_prediction_rmse + 0.25
            and latent.inner_land_field_rmse <= fixed_true.inner_land_field_rmse + 0.1
        ),
        latent_beats_fixed_underfit=(
            latent.holdout_prediction_rmse < fixed_underfit.holdout_prediction_rmse
            and latent.inner_land_field_rmse < fixed_underfit.inner_land_field_rmse
        ),
    )


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
) -> IntemGammaBetaFitSummary:
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

    Returns:
        Posterior recovery and sampler diagnostics for the fit.
    """
    adapter = build_pymc_gamma_beta_product_space_model(
        case.train_target,
        prior,
        initial_split_mask=initial_mask,
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
    for draw_index, (mask, scalings) in enumerate(
        zip(masks, node_scalings, strict=True)
    ):
        canonical = layout.canonical_split_mask(mask)
        active = layout.active_node_ids(canonical)
        active_indices = np.asarray(active, dtype=np.int64)
        holdout_predictions[draw_index] = (
            case.holdout_target.observation_mean
            + case.holdout_target.node_design[:, active_indices]
            @ scalings[active_indices]
        )
        field_sum += case.coordinate_layout.render_frontier_scalings(active, scalings)
        key = canonical.tobytes()
        mask_keys.add(key)
        truth_count += int(np.array_equal(canonical, case.truth_split_mask))

    posterior_mean_field = field_sum / masks.shape[0]
    posterior_mean_prediction = holdout_predictions.mean(axis=0)
    land_mask = next(
        group.mask
        for group in layout.forest.groups
        if group.name == _LAND_GROUP_NAME
    )
    region_counts = layout.minimum_regions + masks.sum(axis=1)
    return IntemGammaBetaFitSummary(
        name=name,
        draws=masks.shape[0],
        tune=tune,
        mean_k=float(region_counts.mean()),
        minimum_k=int(region_counts.min()),
        maximum_k=int(region_counts.max()),
        truth_partition_probability=truth_count / masks.shape[0],
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
            standard_deviations=case.data.error[case.holdout_indices],
        ),
        partition_acceptance_rate=_partition_acceptance_rate(trace),
        divergence_count=int(np.asarray(trace.sample_stats["diverging"]).sum()),
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
    return float(
        np.sum(
            maxima
            + np.log(np.mean(np.exp(draw_log_density - maxima), axis=0))
        )
    )


def _partition_acceptance_rate(trace: Any) -> float:
    """Extract the single custom structural acceptance statistic."""
    names = tuple(str(name) for name in trace.sample_stats.data_vars)
    accepted_names = tuple(
        name
        for name in names
        if name == "accepted" or name.endswith("_accepted")
    )
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
    )
    print(json.dumps(benchmark.as_dict(), indent=parsed.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
