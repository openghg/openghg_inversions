"""Run an exact 4 by 4 InTEM partition-recovery benchmark.

The benchmark faithfully adapts the packaged-region loading, inner-rectangle
selection, and smooth synthetic sensitivity construction used by
``dyadic_intem_product_space_demo.py``.  It creates 48 synthetic rows: the
first 32 condition every inversion and the final 16 are used only for
holdout evaluation.  Seven residual InTEM-region coefficients remain active
under every inner partition.

The declared inner truth is a regular four-quadrant frontier generated from
four moderate root-and-contrast coordinates.  Exact conditional Gaussian
results are reported for that partition, a predeclared different partition
with the same ``K=4``, and an underfit ``K=2`` partition.  The latent result
integrates the conditional posterior over all 677 valid 4 by 4 frontiers.
Every comparison uses one shared Gaussian target, coefficient prior, and
observation covariance ``R``.

The partition prior is explicit: ``p(K)`` is uniform on the documented range
``K=1, ..., 16``, while ``p(P | K)=1/N_K`` uses counts obtained from exact
enumeration.  Thus ``p(P)=p(K)/N_K`` and the prior sums to one over the 677
frontiers.  Posterior means and covariance contributions are evaluated
analytically (Rao-Blackwellized); no Monte Carlo sampling is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.resources import as_file, files
import json
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import xarray as xr

from openghg_inversions.basis.experimental.dyadic.contrast import TreeContrastLayout
from openghg_inversions.basis.experimental.dyadic.enumeration import enumerate_partitions
from openghg_inversions.basis.experimental.dyadic.gaussian_product_space import (
    GaussianProductSpaceTarget,
)
from openghg_inversions.basis.experimental.dyadic.gaussian_product_space_sampler import (
    sample_collapsed_gaussian_product_space,
    sample_gaussian_product_space,
)
from openghg_inversions.basis.experimental.dyadic.multiscale import MultiscaleDesign
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree

_INTEM_RESOURCE = "outer_region_definition_EUROPE.nc"
_INNER_SHAPE = (4, 4)
_INNER_CENTRE = (52.0, 0.0)
_OUTER_LABELS = tuple(range(7))
_ROW_COUNT = 48
_TRAIN_ROW_COUNT = 32
_HOLDOUT_ROW_COUNT = 16
_MINIMUM_REGIONS = 1
_MAXIMUM_REGIONS = 16
_OBSERVATION_ERROR_SD = 0.05
_INNER_PRIOR_SD = 0.8
_OUTER_PRIOR_SD = 0.4


@dataclass(frozen=True, slots=True)
class InnerRectangle:
    """Index and coordinate bounds of the variable 4 by 4 inner rectangle.

    Attributes:
        row_start: Inclusive first latitude index.
        row_stop: Exclusive final latitude index.
        column_start: Inclusive first longitude index.
        column_stop: Exclusive final longitude index.
        latitude_min: Latitude at ``row_start``.
        latitude_max: Latitude at ``row_stop - 1``.
        longitude_min: Longitude at ``column_start``.
        longitude_max: Longitude at ``column_stop - 1``.
    """

    row_start: int
    row_stop: int
    column_start: int
    column_stop: int
    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float

    @property
    def row_slice(self) -> slice:
        """Return the half-open inner latitude-index slice."""
        return slice(self.row_start, self.row_stop)

    @property
    def column_slice(self) -> slice:
        """Return the half-open inner longitude-index slice."""
        return slice(self.column_start, self.column_stop)


@dataclass(frozen=True, slots=True, eq=False)
class IntemRecoveryCase:
    """Complete matched-target exact InTEM recovery experiment.

    The case keeps training and holdout arrays separate.  :attr:`target`
    contains only the training rows; all fixed and latent calculations reuse
    that same immutable target.  ``truth_coordinates`` has the permanent
    16-coordinate layout, with inactive truth coordinates set to zero.

    Attributes:
        rectangle: Selected inner-domain bounds on the packaged InTEM grid.
        outer_labels: Seven raw InTEM labels retained outside the rectangle.
        tree: Canonical 4 by 4 dyadic tree.
        partitions: All 677 exact frontiers in deterministic order.
        partition_counts: Exact ``N_K`` counts from :attr:`partitions`.
        minimum_regions: Smallest K with positive prior mass.
        maximum_regions: Largest K with positive prior mass.
        truth_partition: Regular four-quadrant K=4 truth frontier.
        wrong_partition: Predeclared non-quadrant K=4 comparison frontier.
        underfit_partition: Predeclared K=2 comparison frontier.
        truth_coordinates: Permanent root-and-contrast truth coordinates.
        inner_truth: Native 4 by 4 piecewise-constant truth.
        outer_truth: Truth for the seven always-active outer coefficients.
        train_row_indices: Original indices of the 32 training rows.
        holdout_row_indices: Original indices of the 16 holdout rows.
        train_inner_design: Inner fine-grid training sensitivities.
        holdout_inner_design: Inner fine-grid holdout sensitivities.
        train_outer_design: Seven-column outer training design.
        holdout_outer_design: Seven-column outer holdout design.
        train_observation_mean: Known training prior-forward mean.
        holdout_observation_mean: Known holdout prior-forward mean.
        train_observations: Noisy observations used for conditioning.
        holdout_observations: Noisy observations used only for scoring.
        holdout_noiseless: Noiseless holdout signal used for RMSE.
        train_observation_covariance: Shared training covariance R.
        holdout_observation_covariance: Holdout block of the same R model.
        target: Shared Gaussian target conditioned on training rows only.
    """

    rectangle: InnerRectangle
    outer_labels: tuple[int, ...]
    tree: DyadicTree
    partitions: tuple[PartitionState, ...]
    partition_counts: dict[int, int]
    minimum_regions: int
    maximum_regions: int
    truth_partition: PartitionState
    wrong_partition: PartitionState
    underfit_partition: PartitionState
    truth_coordinates: np.ndarray
    inner_truth: np.ndarray
    outer_truth: np.ndarray
    train_row_indices: np.ndarray
    holdout_row_indices: np.ndarray
    train_inner_design: np.ndarray
    holdout_inner_design: np.ndarray
    train_outer_design: np.ndarray
    holdout_outer_design: np.ndarray
    train_observation_mean: np.ndarray
    holdout_observation_mean: np.ndarray
    train_observations: np.ndarray
    holdout_observations: np.ndarray
    holdout_noiseless: np.ndarray
    train_observation_covariance: np.ndarray
    holdout_observation_covariance: np.ndarray
    target: GaussianProductSpaceTarget


@dataclass(frozen=True, slots=True)
class ExactRecoveryMetrics:
    """Rao-Blackwellized recovery metrics for one fixed or latent model.

    Attributes:
        holdout_log_predictive_density: Joint 16-row posterior predictive log
            density, including observation and coefficient uncertainty.
        noiseless_holdout_rmse: RMSE of the posterior predictive mean against
            the noiseless holdout signal.
        field_rmse: RMSE of the posterior mean inner field against truth.
        outer_coefficient_rmse: RMSE of the seven posterior mean outer
            coefficients against truth.
        expected_inner_regions: Fixed K or posterior expected K for a mixture.
    """

    holdout_log_predictive_density: float
    noiseless_holdout_rmse: float
    field_rmse: float
    outer_coefficient_rmse: float
    expected_inner_regions: float


@dataclass(frozen=True, slots=True)
class ExactPartitionDiagnostics:
    """Normalized exact prior and posterior diagnostics for inner K and P.

    Attributes:
        partition_count: Number of enumerated frontiers.
        minimum_regions: Smallest K with positive prior mass.
        maximum_regions: Largest K with positive prior mass.
        partition_counts_by_k: Exact ``N_K`` enumeration counts.
        prior_mass_total: Sum of ``p(K)/N_K`` over every frontier.
        prior_mass_by_k: Exact marginal prior mass for each K.
        posterior_mass_by_k: Exact marginal posterior mass for each K.
        posterior_expected_k: Exact posterior expected K.
        posterior_map_k: K of the highest-probability frontier.
        posterior_map_partition_index: Enumeration index of the MAP frontier.
        posterior_map_partition_probability: Exact MAP frontier probability.
        truth_partition_index: Enumeration index of the declared truth.
        truth_partition_probability: Exact posterior mass of the truth P.
        wrong_partition_index: Enumeration index of the wrong K=4 P.
        wrong_partition_probability: Exact posterior mass of the wrong K=4 P.
        underfit_partition_index: Enumeration index of the K=2 P.
        underfit_partition_probability: Exact posterior mass of the K=2 P.
    """

    partition_count: int
    minimum_regions: int
    maximum_regions: int
    partition_counts_by_k: dict[int, int]
    prior_mass_total: float
    prior_mass_by_k: dict[int, float]
    posterior_mass_by_k: dict[int, float]
    posterior_expected_k: float
    posterior_map_k: int
    posterior_map_partition_index: int
    posterior_map_partition_probability: float
    truth_partition_index: int
    truth_partition_probability: float
    wrong_partition_index: int
    wrong_partition_probability: float
    underfit_partition_index: int
    underfit_partition_probability: float


@dataclass(frozen=True, slots=True)
class IntemRecoveryBenchmark:
    """Exact fixed-partition and latent-mixture benchmark results.

    Attributes:
        fixed_truth: Conditional Gaussian metrics for the true K=4 frontier.
        fixed_wrong_k4: Metrics for the predeclared wrong K=4 frontier.
        fixed_underfit: Metrics for the underfit K=2 frontier.
        latent_677_partition_mixture: Exact posterior-mixture metrics over all
            frontiers.
        diagnostics: Exact normalized K and P prior/posterior summaries.
    """

    fixed_truth: ExactRecoveryMetrics
    fixed_wrong_k4: ExactRecoveryMetrics
    fixed_underfit: ExactRecoveryMetrics
    latent_677_partition_mixture: ExactRecoveryMetrics
    diagnostics: ExactPartitionDiagnostics

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable benchmark summary.

        Returns:
            Nested dictionary containing all metrics and diagnostics.
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SampledPartitionDiagnostics:
    """Local-chain fidelity and Rao-Blackwellized recovery diagnostics.

    Attributes:
        sampler: ``"augmented"`` product-space or ``"collapsed"`` marginal
            Gaussian transition.
        draws: Number of retained partitions.
        warmup: Number of discarded transition cycles.
        partition_acceptance_rate: Overall retained structural acceptance.
        split_acceptance_rate: Acceptance among retained split proposals.
        merge_acceptance_rate: Acceptance among retained merge proposals.
        unique_partitions: Number of distinct retained frontiers.
        sampled_expected_k: Empirical posterior mean region count.
        exact_expected_k: Exact 677-component posterior mean region count.
        k_total_variation_distance: Total variation between sampled and exact
            marginal K distributions.
        partition_total_variation_distance: Total variation between sampled
            and exact probabilities over all 677 partitions.
        sampled_truth_probability: Retained frequency of the declared truth P.
        exact_truth_probability: Exact posterior probability of the truth P.
        sampled_mixture: Recovery metrics after weighting exact conditional
            Gaussian components by sampled partition frequencies.
    """

    sampler: str
    draws: int
    warmup: int
    partition_acceptance_rate: float
    split_acceptance_rate: float | None
    merge_acceptance_rate: float | None
    unique_partitions: int
    sampled_expected_k: float
    exact_expected_k: float
    k_total_variation_distance: float
    partition_total_variation_distance: float
    sampled_truth_probability: float
    exact_truth_probability: float
    sampled_mixture: ExactRecoveryMetrics


@dataclass(frozen=True, slots=True)
class SampledRecoveryBenchmark:
    """Exact oracle and one non-enumerating local-chain comparison."""

    exact: IntemRecoveryBenchmark
    sampled: SampledPartitionDiagnostics

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable exact and sampled summary."""
        return asdict(self)


@dataclass(frozen=True, slots=True, eq=False)
class _GaussianComponent:
    """Analytic posterior summaries conditional on one inner partition."""

    log_partition_weight: float
    predictive_mean: np.ndarray
    predictive_covariance: np.ndarray
    field_mean: np.ndarray
    outer_mean: np.ndarray


def load_intem_europe_regions() -> xr.DataArray:
    """Load and validate the packaged raw EUROPE InTEM region classes.

    Returns:
        Two-dimensional integer ``region(lat, lon)`` array with labels zero
        through six.

    Raises:
        ValueError: If the packaged region definition has an unexpected
            dimension order, data type, or label set.
    """
    resource = files("openghg_inversions.basis").joinpath(_INTEM_RESOURCE)
    with as_file(resource) as path, xr.open_dataset(path) as dataset:
        if "region" not in dataset:
            raise ValueError("The packaged InTEM definition must contain 'region'.")
        regions = dataset["region"].load()

    values = np.asarray(regions.values)
    if regions.dims != ("lat", "lon") or regions.ndim != 2:
        raise ValueError("The packaged InTEM region array must use dimensions ('lat', 'lon').")
    if values.dtype.kind not in "iu":
        raise ValueError("The packaged InTEM labels must be integers.")
    if tuple(int(value) for value in np.unique(values)) != _OUTER_LABELS:
        raise ValueError("The EUROPE InTEM definition must contain raw labels 0 through 6.")
    return regions


def select_inner_rectangle(regions: xr.DataArray) -> InnerRectangle:
    """Select the nearest 4 by 4 rectangle wholly inside InTEM class six.

    Args:
        regions: Raw InTEM ``region(lat, lon)`` classes.

    Returns:
        Deterministically selected rectangle nearest latitude 52 degrees
        north and longitude zero.

    Raises:
        ValueError: If ``regions`` has incompatible dimensions or no valid
            rectangle exists.
    """
    if regions.dims != ("lat", "lon") or regions.ndim != 2:
        raise ValueError("regions must use dimensions ('lat', 'lon').")
    values = np.asarray(regions.values)
    windows = np.lib.stride_tricks.sliding_window_view(values == values.max(), _INNER_SHAPE)
    valid_starts = np.argwhere(windows.all(axis=(-2, -1)))
    if valid_starts.size == 0:
        raise ValueError("No 4 by 4 rectangle lies wholly inside the InTEM inner class.")

    latitude = np.asarray(regions["lat"].values, dtype=float)
    longitude = np.asarray(regions["lon"].values, dtype=float)
    height, width = _INNER_SHAPE
    row_centres = (latitude[valid_starts[:, 0]] + latitude[valid_starts[:, 0] + height - 1]) / 2.0
    column_centres = (longitude[valid_starts[:, 1]] + longitude[valid_starts[:, 1] + width - 1]) / 2.0
    longitude_scale = math.cos(math.radians(_INNER_CENTRE[0]))
    distances = np.square(row_centres - _INNER_CENTRE[0]) + np.square(
        longitude_scale * (column_centres - _INNER_CENTRE[1])
    )
    row_start, column_start = (int(value) for value in valid_starts[int(np.argmin(distances))])
    row_stop = row_start + height
    column_stop = column_start + width
    return InnerRectangle(
        row_start=row_start,
        row_stop=row_stop,
        column_start=column_start,
        column_stop=column_stop,
        latitude_min=float(latitude[row_start]),
        latitude_max=float(latitude[row_stop - 1]),
        longitude_min=float(longitude[column_start]),
        longitude_max=float(longitude[column_stop - 1]),
    )


def decode_inner_field(
    layout: TreeContrastLayout,
    partition: PartitionState,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Decode root-and-contrast coordinates to a native-grid field.

    Args:
        layout: Permanent contrast layout for the inner tree.
        partition: Valid frontier selecting the active contrasts and regions.
        coordinates: Full permanent coordinate vector.  Inactive values do
            not affect the result.

    Returns:
        Piecewise-constant field with ``layout.tree.shape``.
    """
    regional_values = layout.decode(partition, coordinates)
    field = np.empty(layout.tree.shape, dtype=float)
    for value, node_id in zip(regional_values, partition.ordered_active(), strict=True):
        tile = layout.tree.tile(node_id)
        field[tile.row_start : tile.row_stop, tile.col_start : tile.col_stop] = value
    return field


def build_recovery_case(*, seed: int = 20260719) -> IntemRecoveryCase:
    """Build the deterministic 32-train/16-holdout InTEM recovery case.

    The prior on ``K`` is uniform from one through sixteen, inclusive, and
    partitions are uniform conditional on K using exact enumeration counts.
    Only the first 32 rows are passed to the shared Gaussian target.

    Args:
        seed: Integer seed controlling only the 48 independent observation
            errors.  Designs and declared truths do not depend on this seed.

    Returns:
        Complete synthetic data, matched Gaussian target, declared fixed
        partitions, and all 677 latent partitions.

    Raises:
        TypeError: If ``seed`` is not an integer.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    regions = load_intem_europe_regions()
    rectangle = select_inner_rectangle(regions)
    inner_design, outer_design = _synthetic_designs(regions, rectangle, row_count=_ROW_COUNT)
    tree = DyadicTree.from_shape(_INNER_SHAPE)
    partitions = enumerate_partitions(tree)
    partition_counts = dict(sorted(Counter(len(partition.active) for partition in partitions).items()))
    truth_partition, wrong_partition, underfit_partition = _declared_partitions(tree)

    layout = TreeContrastLayout.from_tree(tree)
    truth_coordinates = np.zeros(layout.coordinate_count, dtype=float)
    truth_coordinates[list(layout.active_coordinate_indices(truth_partition))] = np.array(
        [0.12, 0.42, -0.32, 0.36]
    )
    inner_truth = decode_inner_field(layout, truth_partition, truth_coordinates)
    outer_truth = np.array([-0.22, 0.12, 0.18, -0.14, 0.07, 0.24, -0.09], dtype=float)
    observation_mean = 1800.0 + np.linspace(-0.15, 0.15, _ROW_COUNT)
    noiseless = (
        observation_mean
        + np.einsum("ijk,jk->i", inner_design, inner_truth, optimize=True)
        + outer_design @ outer_truth
    )
    observations = noiseless + np.random.default_rng(seed).normal(
        scale=_OBSERVATION_ERROR_SD,
        size=_ROW_COUNT,
    )
    train = slice(0, _TRAIN_ROW_COUNT)
    holdout = slice(_TRAIN_ROW_COUNT, _ROW_COUNT)
    train_covariance = np.eye(_TRAIN_ROW_COUNT) * _OBSERVATION_ERROR_SD**2
    holdout_covariance = np.eye(_HOLDOUT_ROW_COUNT) * _OBSERVATION_ERROR_SD**2
    log_k_probability = -math.log(_MAXIMUM_REGIONS - _MINIMUM_REGIONS + 1)

    def partition_log_prior(partition: PartitionState) -> float:
        """Return log p(P)=log p(K)-log N_K from exact counts."""
        region_count = len(partition.active)
        if not _MINIMUM_REGIONS <= region_count <= _MAXIMUM_REGIONS:
            return -math.inf
        return log_k_probability - math.log(partition_counts[region_count])

    target = GaussianProductSpaceTarget.from_grid(
        observations=observations[train],
        observation_mean=observation_mean[train],
        inner_grid_design=inner_design[train],
        tree=tree,
        observation_covariance=train_covariance,
        inner_prior_scale=_INNER_PRIOR_SD,
        inactive_pseudo_prior_scale=1.0,
        outer_design=outer_design[train],
        outer_prior_covariance=np.eye(len(_OUTER_LABELS)) * _OUTER_PRIOR_SD**2,
        partition_log_prior=partition_log_prior,
    )
    return IntemRecoveryCase(
        rectangle=rectangle,
        outer_labels=_OUTER_LABELS,
        tree=tree,
        partitions=partitions,
        partition_counts=partition_counts,
        minimum_regions=_MINIMUM_REGIONS,
        maximum_regions=_MAXIMUM_REGIONS,
        truth_partition=truth_partition,
        wrong_partition=wrong_partition,
        underfit_partition=underfit_partition,
        truth_coordinates=_frozen(truth_coordinates),
        inner_truth=_frozen(inner_truth),
        outer_truth=_frozen(outer_truth),
        train_row_indices=_frozen_integer(np.arange(_TRAIN_ROW_COUNT)),
        holdout_row_indices=_frozen_integer(np.arange(_TRAIN_ROW_COUNT, _ROW_COUNT)),
        train_inner_design=_frozen(inner_design[train]),
        holdout_inner_design=_frozen(inner_design[holdout]),
        train_outer_design=_frozen(outer_design[train]),
        holdout_outer_design=_frozen(outer_design[holdout]),
        train_observation_mean=_frozen(observation_mean[train]),
        holdout_observation_mean=_frozen(observation_mean[holdout]),
        train_observations=_frozen(observations[train]),
        holdout_observations=_frozen(observations[holdout]),
        holdout_noiseless=_frozen(noiseless[holdout]),
        train_observation_covariance=_frozen(train_covariance),
        holdout_observation_covariance=_frozen(holdout_covariance),
        target=target,
    )


def evaluate_recovery_case(case: IntemRecoveryCase) -> IntemRecoveryBenchmark:
    """Evaluate exact fixed and 677-partition latent posterior predictions.

    Args:
        case: Synthetic case returned by :func:`build_recovery_case`.

    Returns:
        Required holdout, field, outer-coefficient, and exact K/P metrics.
        Continuous coefficients are integrated analytically within every
        partition, and latent summaries are weighted by exact posterior
        partition probabilities.
    """
    components, posterior_weights = _components_and_posterior_weights(case)
    partition_indices = {partition: index for index, partition in enumerate(case.partitions)}
    truth_index = partition_indices[case.truth_partition]
    wrong_index = partition_indices[case.wrong_partition]
    underfit_index = partition_indices[case.underfit_partition]

    fixed_truth = _fixed_metrics(case, components[truth_index], len(case.truth_partition.active))
    fixed_wrong = _fixed_metrics(case, components[wrong_index], len(case.wrong_partition.active))
    fixed_underfit = _fixed_metrics(
        case,
        components[underfit_index],
        len(case.underfit_partition.active),
    )
    latent = _mixture_metrics(case, components, posterior_weights)

    region_counts = np.array([len(partition.active) for partition in case.partitions], dtype=int)
    prior_weights = np.exp(
        np.array([case.target.partition_log_prior(partition) for partition in case.partitions])
    )
    prior_mass_by_k = {
        region_count: float(prior_weights[region_counts == region_count].sum())
        for region_count in range(case.minimum_regions, case.maximum_regions + 1)
    }
    posterior_mass_by_k = {
        region_count: float(posterior_weights[region_counts == region_count].sum())
        for region_count in range(case.minimum_regions, case.maximum_regions + 1)
    }
    map_index = int(np.argmax(posterior_weights))
    diagnostics = ExactPartitionDiagnostics(
        partition_count=len(case.partitions),
        minimum_regions=case.minimum_regions,
        maximum_regions=case.maximum_regions,
        partition_counts_by_k=case.partition_counts,
        prior_mass_total=float(prior_weights.sum()),
        prior_mass_by_k=prior_mass_by_k,
        posterior_mass_by_k=posterior_mass_by_k,
        posterior_expected_k=float(posterior_weights @ region_counts),
        posterior_map_k=int(region_counts[map_index]),
        posterior_map_partition_index=map_index,
        posterior_map_partition_probability=float(posterior_weights[map_index]),
        truth_partition_index=truth_index,
        truth_partition_probability=float(posterior_weights[truth_index]),
        wrong_partition_index=wrong_index,
        wrong_partition_probability=float(posterior_weights[wrong_index]),
        underfit_partition_index=underfit_index,
        underfit_partition_probability=float(posterior_weights[underfit_index]),
    )
    return IntemRecoveryBenchmark(
        fixed_truth=fixed_truth,
        fixed_wrong_k4=fixed_wrong,
        fixed_underfit=fixed_underfit,
        latent_677_partition_mixture=latent,
        diagnostics=diagnostics,
    )


def run_benchmark(*, seed: int = 20260719) -> IntemRecoveryBenchmark:
    """Build and evaluate the exact recovery benchmark.

    Args:
        seed: Integer synthetic observation-error seed.

    Returns:
        Deterministic exact benchmark results for the requested seed.
    """
    return evaluate_recovery_case(build_recovery_case(seed=seed))


def sample_recovery_case(
    case: IntemRecoveryCase,
    *,
    draws: int = 20_000,
    warmup: int = 2_000,
    sampler: str = "augmented",
    seed: int = 481,
) -> SampledRecoveryBenchmark:
    """Compare a non-enumerating local chain with the exact 677-state oracle.

    Exact conditional Gaussian components are reused to Rao-Blackwellize the
    sampled predictive and field metrics.  The local chain therefore needs to
    approximate only the partition probabilities in this comparison.

    Args:
        case: Matched synthetic InTEM recovery case.
        draws: Positive retained partition count.
        warmup: Non-negative discarded transition cycles.
        sampler: ``"augmented"`` product-space MH-within-Gibbs or
            ``"collapsed"`` exact marginal Gaussian partition MH.
        seed: Random seed for the local chain.

    Returns:
        Exact benchmark and sampled structural-fidelity diagnostics.

    Raises:
        ValueError: If ``sampler`` is unsupported.  Draw-control validation is
            delegated to the selected reusable sampler.
    """
    if sampler not in {"augmented", "collapsed"}:
        raise ValueError("sampler must be 'augmented' or 'collapsed'.")
    sampler_function = (
        sample_gaussian_product_space
        if sampler == "augmented"
        else sample_collapsed_gaussian_product_space
    )
    trace = sampler_function(
        case.target,
        case.wrong_partition,
        draws=draws,
        warmup=warmup,
        rng=np.random.default_rng(seed),
    )
    components, exact_weights = _components_and_posterior_weights(case)
    partition_indices = {partition: index for index, partition in enumerate(case.partitions)}
    sampled_counts = np.zeros(len(case.partitions), dtype=np.int64)
    for partition in trace.partitions:
        sampled_counts[partition_indices[partition]] += 1
    sampled_weights = sampled_counts / sampled_counts.sum()
    region_counts = np.array([len(partition.active) for partition in case.partitions], dtype=int)
    exact_by_k = np.bincount(region_counts, weights=exact_weights, minlength=case.maximum_regions + 1)
    sampled_by_k = np.bincount(
        region_counts,
        weights=sampled_weights,
        minlength=case.maximum_regions + 1,
    )
    truth_index = partition_indices[case.truth_partition]
    exact = evaluate_recovery_case(case)
    return SampledRecoveryBenchmark(
        exact=exact,
        sampled=SampledPartitionDiagnostics(
            sampler=sampler,
            draws=trace.draw_count,
            warmup=warmup,
            partition_acceptance_rate=trace.partition_acceptance_rate,
            split_acceptance_rate=trace.move_acceptance_rate("split"),
            merge_acceptance_rate=trace.move_acceptance_rate("merge"),
            unique_partitions=len(set(trace.partitions)),
            sampled_expected_k=float(sampled_weights @ region_counts),
            exact_expected_k=exact.diagnostics.posterior_expected_k,
            k_total_variation_distance=float(0.5 * np.abs(sampled_by_k - exact_by_k).sum()),
            partition_total_variation_distance=float(0.5 * np.abs(sampled_weights - exact_weights).sum()),
            sampled_truth_probability=float(sampled_weights[truth_index]),
            exact_truth_probability=exact.diagnostics.truth_partition_probability,
            sampled_mixture=_mixture_metrics(case, components, sampled_weights),
        ),
    )


def run_sampled_benchmark(
    *,
    data_seed: int = 20260719,
    draws: int = 20_000,
    warmup: int = 2_000,
    sampler: str = "augmented",
    sampler_seed: int = 481,
) -> SampledRecoveryBenchmark:
    """Build the recovery case and run one exact-versus-local comparison.

    Args:
        data_seed: Synthetic observation-error seed.
        draws: Retained local-chain partitions.
        warmup: Discarded local-chain cycles.
        sampler: ``"augmented"`` or ``"collapsed"`` transition.
        sampler_seed: Random seed for partition sampling.

    Returns:
        Exact oracle and local-chain diagnostics.
    """
    return sample_recovery_case(
        build_recovery_case(seed=data_seed),
        draws=draws,
        warmup=warmup,
        sampler=sampler,
        seed=sampler_seed,
    )


def _declared_partitions(
    tree: DyadicTree,
) -> tuple[PartitionState, PartitionState, PartitionState]:
    """Construct the quadrant truth, same-K wrong P, and K=2 underfit P.

    Args:
        tree: Canonical 4 by 4 tree used by the benchmark.

    Returns:
        Truth K=4, wrong K=4, and underfit K=2 frontiers in that order.
    """
    underfit = PartitionState.root(tree).split(tree, tree.root_id)
    upper_id, lower_id = tree.children(tree.root_id)
    truth = underfit.split(tree, upper_id).split(tree, lower_id)
    upper_left_id, _ = tree.children(upper_id)
    wrong = underfit.split(tree, upper_id).split(tree, upper_left_id)
    truth.validate(tree)
    wrong.validate(tree)
    underfit.validate(tree)
    return truth, wrong, underfit


def _synthetic_designs(
    regions: xr.DataArray,
    rectangle: InnerRectangle,
    *,
    row_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the demo's smooth local and fixed outer-region sensitivities.

    Args:
        regions: Raw packaged InTEM labels.
        rectangle: Selected variable inner rectangle.
        row_count: Number of deterministic sensitivity rows.

    Returns:
        Inner design with shape ``(row_count, 4, 4)`` and seven-column outer
        design.  Each full sensitivity row sums to 1.35.
    """
    region_values = np.asarray(regions.values)
    rows, columns = np.indices(region_values.shape, dtype=float)
    inner_mask = np.zeros(region_values.shape, dtype=bool)
    inner_mask[rectangle.row_slice, rectangle.column_slice] = True
    outer_masks = tuple((region_values == label) & ~inner_mask for label in _OUTER_LABELS)
    inner_design = np.empty((row_count, *_INNER_SHAPE), dtype=float)
    outer_design = np.empty((row_count, len(_OUTER_LABELS)), dtype=float)

    for row_index in range(row_count):
        inner_offset = row_index % math.prod(_INNER_SHAPE)
        local_row = rectangle.row_start + inner_offset // _INNER_SHAPE[1]
        local_column = rectangle.column_start + inner_offset % _INNER_SHAPE[1]
        local_scale = 0.65 + 0.15 * (row_index % 3)
        local = np.exp(
            -0.5
            * (
                np.square((rows - local_row) / local_scale)
                + np.square((columns - local_column) / local_scale)
            )
        )
        local /= local.sum()

        broad_row = 20 + (37 * row_index) % (region_values.shape[0] - 40)
        broad_column = 20 + (53 * row_index) % (region_values.shape[1] - 40)
        broad = np.exp(
            -0.5 * (np.square((rows - broad_row) / 24.0) + np.square((columns - broad_column) / 32.0))
        )
        broad /= broad.sum()
        sensitivity = local + 0.35 * broad
        inner_design[row_index] = sensitivity[rectangle.row_slice, rectangle.column_slice]
        outer_design[row_index] = [float(sensitivity[outer_mask].sum()) for outer_mask in outer_masks]
    return inner_design, outer_design


def _components_and_posterior_weights(
    case: IntemRecoveryCase,
) -> tuple[tuple[_GaussianComponent, ...], np.ndarray]:
    """Return every conditional component and normalized exact P weights."""
    holdout_design = MultiscaleDesign.from_grid(case.holdout_inner_design, case.tree)
    components = tuple(
        _conditional_component(case, partition, holdout_design) for partition in case.partitions
    )
    log_weights = np.array(
        [component.log_partition_weight for component in components],
        dtype=float,
    )
    maximum = float(log_weights.max())
    weights = np.exp(log_weights - maximum)
    weights /= weights.sum()
    return components, weights


def _conditional_component(
    case: IntemRecoveryCase,
    partition: PartitionState,
    holdout_design: MultiscaleDesign,
) -> _GaussianComponent:
    """Return exact posterior weight and coefficient moments conditional on P.

    The posterior precision is formed in coefficient space.  The marginal
    likelihood uses the matrix determinant lemma and Woodbury quadratic, so
    no 32 by 32 partition-specific covariance factorization is needed.

    Args:
        case: Matched training target and holdout data.
        partition: Frontier held fixed for this Gaussian component.
        holdout_design: Precomputed multiscale holdout inner design.

    Returns:
        Unnormalized log posterior partition weight, exact holdout predictive
        moments, and posterior mean inner/outer coefficients.
    """
    training_design, active_indices = case.target.active_design(partition)
    outer_variances = np.diag(case.target.outer_prior_covariance)
    prior_variances = np.concatenate(
        (case.target.inner_prior_variances[list(active_indices)], outer_variances)
    )
    error_variances = np.diag(case.target.observation_covariance)
    residual = case.target.observations - case.target.observation_mean
    weighted_design = training_design / error_variances[:, np.newaxis]
    precision = np.diag(1.0 / prior_variances) + training_design.T @ weighted_design
    factor = cho_factor(precision, lower=True, check_finite=False)
    information = training_design.T @ (residual / error_variances)
    conditional_mean = cho_solve(factor, information, check_finite=False)
    conditional_covariance = cho_solve(
        factor,
        np.eye(precision.shape[0]),
        check_finite=False,
    )
    log_determinant = (
        np.log(error_variances).sum() + np.log(prior_variances).sum() + 2.0 * np.log(np.diag(factor[0])).sum()
    )
    residual_quadratic = residual @ (residual / error_variances) - information @ conditional_mean
    log_marginal_likelihood = -0.5 * (
        residual.size * math.log(2.0 * math.pi) + log_determinant + residual_quadratic
    )

    regional_decoder = case.target.contrast_layout.decoder(partition)[:, active_indices]
    inner_active_design = holdout_design.gather(partition) @ regional_decoder
    prediction_design = np.column_stack((inner_active_design, case.holdout_outer_design))
    predictive_mean = case.holdout_observation_mean + prediction_design @ conditional_mean
    predictive_covariance = (
        case.holdout_observation_covariance + prediction_design @ conditional_covariance @ prediction_design.T
    )

    field_operator = np.empty((math.prod(_INNER_SHAPE), len(active_indices)), dtype=float)
    field_operator.fill(np.nan)
    for region_index, node_id in enumerate(partition.ordered_active()):
        tile = case.tree.tile(node_id)
        cell_rows = np.arange(math.prod(_INNER_SHAPE)).reshape(_INNER_SHAPE)[
            tile.row_start : tile.row_stop,
            tile.col_start : tile.col_stop,
        ]
        field_operator[cell_rows.ravel()] = regional_decoder[region_index]
    active_count = len(active_indices)
    return _GaussianComponent(
        log_partition_weight=(float(log_marginal_likelihood) + case.target.partition_log_prior(partition)),
        predictive_mean=predictive_mean,
        predictive_covariance=predictive_covariance,
        field_mean=(field_operator @ conditional_mean[:active_count]).reshape(_INNER_SHAPE),
        outer_mean=conditional_mean[active_count:],
    )


def _fixed_metrics(
    case: IntemRecoveryCase,
    component: _GaussianComponent,
    region_count: int,
) -> ExactRecoveryMetrics:
    """Calculate exact posterior-mean metrics for one fixed partition.

    Args:
        case: Benchmark truth and holdout arrays.
        component: Exact conditional Gaussian summaries.
        region_count: Fixed number of active inner regions.

    Returns:
        Joint predictive-density and posterior-mean recovery metrics.
    """
    return ExactRecoveryMetrics(
        holdout_log_predictive_density=_multivariate_normal_logpdf(
            case.holdout_observations,
            component.predictive_mean,
            component.predictive_covariance,
        ),
        noiseless_holdout_rmse=_rmse(component.predictive_mean, case.holdout_noiseless),
        field_rmse=_rmse(component.field_mean, case.inner_truth),
        outer_coefficient_rmse=_rmse(component.outer_mean, case.outer_truth),
        expected_inner_regions=float(region_count),
    )


def _mixture_metrics(
    case: IntemRecoveryCase,
    components: tuple[_GaussianComponent, ...],
    weights: np.ndarray,
) -> ExactRecoveryMetrics:
    """Calculate exact Rao-Blackwellized metrics for the latent P mixture.

    Args:
        case: Benchmark truth, holdout arrays, and partition sequence.
        components: Conditional Gaussian summaries in partition order.
        weights: Normalized exact posterior partition probabilities.

    Returns:
        Gaussian-mixture predictive density and posterior-weighted mean
        recovery metrics.
    """
    predictive_mean = sum(
        (weight * component.predictive_mean for weight, component in zip(weights, components, strict=True)),
        start=np.zeros(_HOLDOUT_ROW_COUNT),
    )
    field_mean = sum(
        (weight * component.field_mean for weight, component in zip(weights, components, strict=True)),
        start=np.zeros(_INNER_SHAPE),
    )
    outer_mean = sum(
        (weight * component.outer_mean for weight, component in zip(weights, components, strict=True)),
        start=np.zeros(len(_OUTER_LABELS)),
    )
    component_log_densities = np.array(
        [
            _multivariate_normal_logpdf(
                case.holdout_observations,
                component.predictive_mean,
                component.predictive_covariance,
            )
            for component in components
        ]
    )
    positive = weights > 0.0
    log_terms = np.log(weights[positive]) + component_log_densities[positive]
    maximum = float(log_terms.max())
    log_predictive_density = maximum + math.log(float(np.exp(log_terms - maximum).sum()))
    region_counts = np.array([len(partition.active) for partition in case.partitions], dtype=float)
    return ExactRecoveryMetrics(
        holdout_log_predictive_density=log_predictive_density,
        noiseless_holdout_rmse=_rmse(predictive_mean, case.holdout_noiseless),
        field_rmse=_rmse(field_mean, case.inner_truth),
        outer_coefficient_rmse=_rmse(outer_mean, case.outer_truth),
        expected_inner_regions=float(weights @ region_counts),
    )


def _multivariate_normal_logpdf(
    value: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Return a normalized multivariate Gaussian log density.

    Args:
        value: Evaluation vector.
        mean: Gaussian mean with the same shape as ``value``.
        covariance: Positive-definite Gaussian covariance.

    Returns:
        Scalar normalized log density calculated by Cholesky solve.
    """
    residual = value - mean
    cholesky = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(cholesky, residual)
    return float(
        -0.5
        * (value.size * math.log(2.0 * math.pi) + 2.0 * np.log(np.diag(cholesky)).sum() + whitened @ whitened)
    )


def _rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Return root mean square error between equally shaped arrays."""
    return float(np.sqrt(np.mean(np.square(estimate - truth))))


def _frozen(values: np.ndarray) -> np.ndarray:
    """Return a floating-point copy with read-only storage."""
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


def _frozen_integer(values: np.ndarray) -> np.ndarray:
    """Return an integer copy with read-only storage."""
    result = np.asarray(values, dtype=np.int64).copy()
    result.setflags(write=False)
    return result


def _parse_args() -> argparse.Namespace:
    """Parse command-line controls for the exact benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--sampler", choices=("exact", "augmented", "collapsed"), default="exact")
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--sampler-seed", type=int, default=481)
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and print the complete JSON result."""
    args = _parse_args()
    if args.sampler == "exact":
        result = run_benchmark(seed=args.seed)
    else:
        result = run_sampled_benchmark(
            data_seed=args.seed,
            draws=args.draws,
            warmup=args.warmup,
            sampler=args.sampler,
            sampler_seed=args.sampler_seed,
        )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
