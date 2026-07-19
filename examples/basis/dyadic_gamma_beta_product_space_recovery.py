"""Benchmark latent positive partitions against matched fixed inversions.

This deliberately identifiable synthetic case adapts the smallest recovery test
from ``codex/tdmcmc-numba-rewrite`` to the projectively consistent Gamma--Beta
product space.  Separate observation blocks see the left and right grid cells.
The true scaling field is ``[0.5, 1.5]``, which is represented exactly by a
mean-one root scaling and one Beta split fraction.

Three PyMC fits use identical observations, residual covariance, Gamma--Beta
coordinate priors, NUTS settings, and permanent dimensions:

* latent ``K`` and ``P`` with a uniform prior over the split and unsplit models;
* oracle fixed ``K=2`` and the true split partition;
* underfit fixed ``K=1`` and the unsplit partition.

The command prints a JSON report.  It is an experimental implementation gate,
not a production inversion or a reproduction of Lunt et al.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pymc as pm

from openghg_inversions.basis.experimental.dyadic.gamma_beta import (
    DepthKappaStrategy,
    GammaBetaForest,
    GammaBetaGroupSpec,
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

_TRAIN_PER_GRID_CELL = 20
_HOLDOUT_PER_GRID_CELL = 10
_OBSERVATION_SD = 0.05
_ROOT_VARIANCE = 0.25
_KAPPA = 2.0


@dataclass(frozen=True, slots=True, eq=False)
class GammaBetaRecoveryCase:
    """Complete matched positive split-recovery experiment.

    Attributes:
        forest: Shared two-grid-cell Gamma--Beta forest.
        layout: Canonical split-mask layout.
        latent_prior: Uniform prior over ``K=1`` and ``K=2``.
        fixed_unsplit_prior: Degenerate prior on ``K=1``.
        fixed_split_prior: Degenerate prior on ``K=2``.
        train_target: Noisy training target.
        holdout_target: Independent noisy held-out target.
        truth: Declared two-grid-cell scaling field.
        holdout_noiseless: Noise-free held-out prediction.
    """

    forest: GammaBetaForest
    layout: GammaBetaPartitionLayout
    latent_prior: GammaBetaRegionCountPrior
    fixed_unsplit_prior: GammaBetaRegionCountPrior
    fixed_split_prior: GammaBetaRegionCountPrior
    train_target: GammaBetaProductSpaceTarget
    holdout_target: GammaBetaProductSpaceTarget
    truth: npt.NDArray[np.float64]
    holdout_noiseless: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GammaBetaFitSummary:
    """Posterior and sampler diagnostics for one matched fit."""

    name: str
    draws: int
    tune: int
    mean_k: float
    split_probability: float
    posterior_mean_field: tuple[float, float]
    field_rmse: float
    holdout_prediction_rmse: float
    holdout_log_predictive_density: float
    partition_acceptance_rate: float
    divergence_count: int


@dataclass(frozen=True, slots=True)
class GammaBetaRecoveryBenchmark:
    """Latent, oracle fixed, and underfit fixed comparison."""

    latent: GammaBetaFitSummary
    fixed_true_split: GammaBetaFitSummary
    fixed_underfit_unsplit: GammaBetaFitSummary
    latent_matches_fixed_true: bool
    latent_beats_fixed_underfit: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable benchmark report."""
        return asdict(self)


def build_case(*, seed: int = 1701) -> GammaBetaRecoveryCase:
    """Build deterministic train and holdout positive recovery targets.

    Args:
        seed: Seed for independent Gaussian training and holdout errors.

    Returns:
        Shared forest, priors, targets, and declared truth.
    """
    forest = GammaBetaForest.from_groups(
        np.ones((1, 2)),
        [
            GammaBetaGroupSpec(
                "inner",
                np.ones((1, 2), dtype=bool),
                root_variance=_ROOT_VARIANCE,
                max_depth=1,
            )
        ],
        require_full_coverage=True,
    )
    layout = GammaBetaPartitionLayout.from_forest(forest)
    strategy = DepthKappaStrategy(base_kappa=_KAPPA)
    truth = np.array([0.5, 1.5], dtype=np.float64)
    train_design = _separated_design(_TRAIN_PER_GRID_CELL)
    holdout_design = _separated_design(_HOLDOUT_PER_GRID_CELL)
    train_noiseless = np.einsum(
        "oij,ij->o",
        train_design,
        truth.reshape(1, 2),
    )
    holdout_noiseless = np.einsum(
        "oij,ij->o",
        holdout_design,
        truth.reshape(1, 2),
    )
    generator = np.random.default_rng(seed)
    train_observations = train_noiseless + generator.normal(
        scale=_OBSERVATION_SD,
        size=train_noiseless.size,
    )
    holdout_observations = holdout_noiseless + generator.normal(
        scale=_OBSERVATION_SD,
        size=holdout_noiseless.size,
    )
    train_target = GammaBetaProductSpaceTarget.from_grid(
        observations=train_observations,
        finest_grid_design=train_design,
        forest=forest,
        kappa_strategy=strategy,
        observation_sd=_OBSERVATION_SD,
    )
    holdout_target = GammaBetaProductSpaceTarget.from_grid(
        observations=holdout_observations,
        finest_grid_design=holdout_design,
        forest=forest,
        kappa_strategy=strategy,
        observation_sd=_OBSERVATION_SD,
    )
    truth.setflags(write=False)
    holdout_noiseless.setflags(write=False)
    return GammaBetaRecoveryCase(
        forest=forest,
        layout=layout,
        latent_prior=GammaBetaRegionCountPrior.uniform_k(layout),
        fixed_unsplit_prior=GammaBetaRegionCountPrior.uniform_k(
            layout,
            minimum_regions=1,
            maximum_regions=1,
        ),
        fixed_split_prior=GammaBetaRegionCountPrior.uniform_k(
            layout,
            minimum_regions=2,
            maximum_regions=2,
        ),
        train_target=train_target,
        holdout_target=holdout_target,
        truth=truth,
        holdout_noiseless=holdout_noiseless,
    )


def run_benchmark(
    case: GammaBetaRecoveryCase,
    *,
    draws: int = 1_000,
    tune: int = 1_000,
    seed: int = 20260719,
    target_accept: float = 0.9,
) -> GammaBetaRecoveryBenchmark:
    """Run matched latent, oracle, and underfit PyMC inversions.

    Args:
        case: Recovery case from :func:`build_case`.
        draws: Positive retained draws per fit.
        tune: Non-negative NUTS tuning draws per fit.
        seed: Base seed; fixed comparator seeds use deterministic offsets.
        target_accept: NUTS target acceptance probability.

    Returns:
        Matched posterior and sampler comparison.
    """
    latent = _sample_fit(
        case,
        name="latent_k_p",
        prior=case.latent_prior,
        initial_mask=case.layout.initial_split_mask(1),
        draws=draws,
        tune=tune,
        seed=seed,
        target_accept=target_accept,
    )
    fixed_true = _sample_fit(
        case,
        name="fixed_true_split",
        prior=case.fixed_split_prior,
        initial_mask=case.layout.initial_split_mask(2),
        draws=draws,
        tune=tune,
        seed=seed + 1,
        target_accept=target_accept,
    )
    fixed_underfit = _sample_fit(
        case,
        name="fixed_underfit_unsplit",
        prior=case.fixed_unsplit_prior,
        initial_mask=case.layout.initial_split_mask(1),
        draws=draws,
        tune=tune,
        seed=seed + 2,
        target_accept=target_accept,
    )
    return GammaBetaRecoveryBenchmark(
        latent=latent,
        fixed_true_split=fixed_true,
        fixed_underfit_unsplit=fixed_underfit,
        latent_matches_fixed_true=(
            latent.holdout_prediction_rmse
            <= fixed_true.holdout_prediction_rmse + 0.01
            and latent.field_rmse <= fixed_true.field_rmse + 0.05
        ),
        latent_beats_fixed_underfit=(
            latent.holdout_prediction_rmse
            < fixed_underfit.holdout_prediction_rmse
            and latent.field_rmse < fixed_underfit.field_rmse
        ),
    )


def _sample_fit(
    case: GammaBetaRecoveryCase,
    *,
    name: str,
    prior: GammaBetaRegionCountPrior,
    initial_mask: npt.ArrayLike,
    draws: int,
    tune: int,
    seed: int,
    target_accept: float,
) -> GammaBetaFitSummary:
    """Run one compound chain and summarize held-out positive recovery."""
    adapter = build_pymc_gamma_beta_product_space_model(
        case.train_target,
        prior,
        initial_split_mask=initial_mask,
    )
    steps = adapter.step_methods(
        partition_rng=seed,
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

    masks = np.asarray(trace.posterior["split_mask"]).reshape(-1, case.layout.split_count)
    node_scalings = np.asarray(trace.posterior["node_scalings"]).reshape(
        -1,
        len(case.forest.nodes),
    )
    holdout_predictions = np.empty(
        (masks.shape[0], case.holdout_target.observations.size),
        dtype=np.float64,
    )
    fields = np.empty((masks.shape[0], *case.forest.shape), dtype=np.float64)
    for draw_index, (mask, scalings) in enumerate(
        zip(masks, node_scalings, strict=True)
    ):
        active = case.layout.active_node_ids(mask)
        active_indices = np.asarray(active, dtype=np.int64)
        holdout_predictions[draw_index] = (
            case.holdout_target.observation_mean
            + case.holdout_target.node_design[:, active_indices]
            @ scalings[active_indices]
        )
        fields[draw_index] = case.holdout_target.coordinate_layout.render_frontier_scalings(
            active,
            scalings,
        )

    posterior_mean_prediction = holdout_predictions.mean(axis=0)
    posterior_mean_field = fields.mean(axis=0).reshape(-1)
    log_predictive = _independent_normal_log_predictive_density(
        observations=case.holdout_target.observations,
        predictions=holdout_predictions,
        standard_deviation=_OBSERVATION_SD,
    )
    return GammaBetaFitSummary(
        name=name,
        draws=masks.shape[0],
        tune=tune,
        mean_k=float(case.layout.minimum_regions + masks.sum(axis=1).mean()),
        split_probability=float(masks.mean()),
        posterior_mean_field=(
            float(posterior_mean_field[0]),
            float(posterior_mean_field[1]),
        ),
        field_rmse=float(
            np.sqrt(np.mean(np.square(posterior_mean_field - case.truth)))
        ),
        holdout_prediction_rmse=float(
            np.sqrt(
                np.mean(
                    np.square(posterior_mean_prediction - case.holdout_noiseless)
                )
            )
        ),
        holdout_log_predictive_density=log_predictive,
        partition_acceptance_rate=_partition_acceptance_rate(trace),
        divergence_count=int(np.asarray(trace.sample_stats["diverging"]).sum()),
    )


def _separated_design(rows_per_grid_cell: int) -> npt.NDArray[np.float64]:
    """Return observations that separately identify each of two grid cells."""
    design = np.zeros((2 * rows_per_grid_cell, 1, 2), dtype=np.float64)
    design[:rows_per_grid_cell, 0, 0] = 1.0
    design[rows_per_grid_cell:, 0, 1] = 1.0
    return design


def _independent_normal_log_predictive_density(
    *,
    observations: npt.NDArray[np.float64],
    predictions: npt.NDArray[np.float64],
    standard_deviation: float,
) -> float:
    """Return summed draw-mixture log density for independent holdout rows."""
    residual = observations[np.newaxis, :] - predictions
    draw_log_density = (
        -0.5 * np.square(residual / standard_deviation)
        - math.log(standard_deviation)
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
    """Return the custom structural acceptance rate from compound sample stats."""
    names = tuple(str(name) for name in trace.sample_stats.data_vars)
    accepted_names = tuple(
        name
        for name in names
        if name == "accepted" or name.endswith("_accepted")
    )
    if len(accepted_names) != 1:
        raise ValueError(
            "Expected exactly one structural accepted statistic, found "
            f"{accepted_names!r}."
        )
    return float(np.asarray(trace.sample_stats[accepted_names[0]]).mean())


def main(arguments: list[str] | None = None) -> int:
    """Run the benchmark and print its JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=1_000)
    parser.add_argument("--tune", type=int, default=1_000)
    parser.add_argument("--data-seed", type=int, default=1701)
    parser.add_argument("--sampling-seed", type=int, default=20260719)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--indent", type=int, default=2)
    parsed = parser.parse_args(arguments)
    benchmark = run_benchmark(
        build_case(seed=parsed.data_seed),
        draws=parsed.draws,
        tune=parsed.tune,
        seed=parsed.sampling_seed,
        target_accept=parsed.target_accept,
    )
    print(json.dumps(benchmark.as_dict(), indent=parsed.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
