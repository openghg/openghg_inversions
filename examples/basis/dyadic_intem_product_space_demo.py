"""Demonstrate dyadic product-space inference inside fixed InTEM regions.

The example loads the packaged EUROPE InTEM map, chooses a deterministic
``4 x 4`` inner rectangle near southern England, and leaves the seven raw
InTEM classes outside that rectangle as fixed outer regions.  The inner
partition is variable; the outer coefficients remain active in every model
and use a separate, tighter Gaussian prior.

All observations are synthetic.  Smooth footprint-like sensitivity rows are
constructed directly from the packaged grid, then combined with declared
inner and outer truths and Gaussian noise.  No stored mole fractions,
boundary-condition product, catalog access, or production RHIME preparation
is involved.  The command compares native PyMC samples with the exact
677-partition Gaussian oracle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.resources import as_file, files
import json
import math
from typing import Any

import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions.basis.experimental.dyadic.enumeration import enumerate_partitions
from openghg_inversions.basis.experimental.dyadic.gaussian_product_space import (
    GaussianProductSpaceTarget,
)
from openghg_inversions.basis.experimental.dyadic.pymc_product_space import (
    build_pymc_product_space_model,
)
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree

_INTEM_RESOURCE = "outer_region_definition_EUROPE.nc"
_INNER_SHAPE = (4, 4)
_INNER_CENTRE = (52.0, 0.0)
_OUTER_LABELS = tuple(range(7))


@dataclass(frozen=True, slots=True)
class InnerRectangle:
    """Index and coordinate bounds of the variable inner rectangle."""

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
        """Return the inner latitude-index slice."""
        return slice(self.row_start, self.row_stop)

    @property
    def column_slice(self) -> slice:
        """Return the inner longitude-index slice."""
        return slice(self.column_start, self.column_stop)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the rectangle's grid shape."""
        return self.row_stop - self.row_start, self.column_stop - self.column_start


@dataclass(frozen=True, slots=True, eq=False)
class IntemSyntheticCase:
    """Complete synthetic fixed-outer, variable-inner Gaussian experiment."""

    regions: xr.DataArray
    rectangle: InnerRectangle
    outer_labels: tuple[int, ...]
    inner_grid_design: np.ndarray
    outer_design: np.ndarray
    inner_truth: np.ndarray
    outer_truth: np.ndarray
    observation_mean: np.ndarray
    observations: np.ndarray
    observation_covariance: np.ndarray
    inner_prior_sd: float
    outer_prior_sd: float
    target: GaussianProductSpaceTarget
    partitions: tuple[PartitionState, ...]


@dataclass(frozen=True, slots=True)
class IntemDemoSummary:
    """Exact and sampled summaries emitted by the command-line demonstration."""

    rectangle: InnerRectangle
    observation_count: int
    partition_count: int
    outer_region_count: int
    inner_prior_sd: float
    outer_prior_sd: float
    exact_map_index: int
    exact_map_inner_regions: int
    exact_map_probability: float
    exact_expected_inner_regions: float
    sampled_expected_inner_regions: float
    sampled_unique_partitions: int
    inner_region_count_total_variation_distance: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "rectangle": {
                "row_start": self.rectangle.row_start,
                "row_stop": self.rectangle.row_stop,
                "column_start": self.rectangle.column_start,
                "column_stop": self.rectangle.column_stop,
                "latitude_min": self.rectangle.latitude_min,
                "latitude_max": self.rectangle.latitude_max,
                "longitude_min": self.rectangle.longitude_min,
                "longitude_max": self.rectangle.longitude_max,
            },
            "observation_count": self.observation_count,
            "partition_count": self.partition_count,
            "outer_region_count": self.outer_region_count,
            "inner_prior_sd": self.inner_prior_sd,
            "outer_prior_sd": self.outer_prior_sd,
            "exact_map_index": self.exact_map_index,
            "exact_map_inner_regions": self.exact_map_inner_regions,
            "exact_map_probability": self.exact_map_probability,
            "exact_expected_inner_regions": self.exact_expected_inner_regions,
            "sampled_expected_inner_regions": self.sampled_expected_inner_regions,
            "sampled_unique_partitions": self.sampled_unique_partitions,
            "inner_region_count_total_variation_distance": (self.inner_region_count_total_variation_distance),
        }


def load_intem_europe_regions() -> xr.DataArray:
    """Load and validate the packaged raw EUROPE InTEM region classes.

    Returns:
        Two-dimensional integer ``region(lat, lon)`` array with raw labels
        ``0`` through ``6``.

    Raises:
        ValueError: If the packaged data do not match the documented InTEM
            layout used by this example.
    """
    resource = files("openghg_inversions.basis").joinpath(_INTEM_RESOURCE)
    with as_file(resource) as path, xr.open_dataset(path) as dataset:
        if "region" not in dataset:
            raise ValueError("The packaged InTEM definition must contain 'region'.")
        regions = dataset["region"].load()

    if regions.dims != ("lat", "lon") or regions.ndim != 2:
        raise ValueError("The packaged InTEM region array must use dimensions ('lat', 'lon').")
    values = np.asarray(regions.values)
    if values.dtype.kind not in "iu":
        raise ValueError("The packaged InTEM labels must be integers.")
    if tuple(int(value) for value in np.unique(values)) != _OUTER_LABELS:
        raise ValueError("The EUROPE InTEM definition must contain raw labels 0 through 6.")
    return regions


def select_inner_rectangle(
    regions: xr.DataArray,
    *,
    shape: tuple[int, int] = _INNER_SHAPE,
    centre: tuple[float, float] = _INNER_CENTRE,
) -> InnerRectangle:
    """Choose the nearest rectangle lying wholly in the legacy inner class.

    Args:
        regions: Raw InTEM ``region(lat, lon)`` classes.
        shape: Positive rectangle height and width in grid locations.
        centre: Preferred latitude and longitude centre in degrees.

    Returns:
        Deterministically selected rectangle.  The legacy inner class is the
        maximum raw label, which is ``6`` for EUROPE.

    Raises:
        ValueError: If inputs are invalid or no requested rectangle lies
            wholly inside the legacy inner class.
    """
    if regions.dims != ("lat", "lon") or regions.ndim != 2:
        raise ValueError("regions must use dimensions ('lat', 'lon').")
    if len(shape) != 2 or any(isinstance(size, bool) or not isinstance(size, int) for size in shape):
        raise ValueError("shape must contain two integer sizes.")
    height, width = shape
    if height < 1 or width < 1 or height > regions.shape[0] or width > regions.shape[1]:
        raise ValueError("shape must define a non-empty rectangle within regions.")
    preferred_latitude, preferred_longitude = (float(value) for value in centre)
    if not math.isfinite(preferred_latitude) or not math.isfinite(preferred_longitude):
        raise ValueError("centre coordinates must be finite.")

    values = np.asarray(regions.values)
    inner_mask = values == values.max()
    windows = np.lib.stride_tricks.sliding_window_view(inner_mask, (height, width))
    valid_starts = np.argwhere(windows.all(axis=(-2, -1)))
    if valid_starts.size == 0:
        raise ValueError("No requested rectangle lies wholly inside the InTEM inner class.")

    latitude = np.asarray(regions["lat"].values, dtype=float)
    longitude = np.asarray(regions["lon"].values, dtype=float)
    row_centres = (latitude[valid_starts[:, 0]] + latitude[valid_starts[:, 0] + height - 1]) / 2.0
    column_centres = (longitude[valid_starts[:, 1]] + longitude[valid_starts[:, 1] + width - 1]) / 2.0
    longitude_scale = math.cos(math.radians(preferred_latitude))
    distances = np.square(row_centres - preferred_latitude) + np.square(
        longitude_scale * (column_centres - preferred_longitude)
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


def build_synthetic_case(
    *,
    observation_count: int = 24,
    inner_prior_sd: float = 1.0,
    outer_prior_sd: float = 0.5,
    observation_error_sd: float = 0.08,
    region_penalty: float = 0.12,
    seed: int = 20260717,
) -> IntemSyntheticCase:
    """Build the deterministic InTEM inner/outer product-space experiment.

    Args:
        observation_count: Number of synthetic sensitivity rows and observations.
        inner_prior_sd: Finest-grid standard deviation inside the rectangle.
        outer_prior_sd: Standard deviation for each fixed outer coefficient.
        observation_error_sd: Independent synthetic observation-error standard deviation.
        region_penalty: Exponential log-prior penalty per additional inner region.
        seed: Observation-noise seed.

    Returns:
        Geometry, synthetic data, exact target, and all 677 inner partitions.

    Raises:
        ValueError: If a numeric configuration value is invalid.
    """
    if isinstance(observation_count, bool) or not isinstance(observation_count, int):
        raise ValueError("observation_count must be an integer.")
    if observation_count < 1:
        raise ValueError("observation_count must be positive.")
    for value, name in (
        (inner_prior_sd, "inner_prior_sd"),
        (outer_prior_sd, "outer_prior_sd"),
        (observation_error_sd, "observation_error_sd"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if not math.isfinite(region_penalty) or region_penalty < 0.0:
        raise ValueError("region_penalty must be finite and non-negative.")

    regions = load_intem_europe_regions()
    rectangle = select_inner_rectangle(regions)
    inner_design, outer_design = _synthetic_designs(
        regions,
        rectangle,
        observation_count=observation_count,
    )
    inner_truth = np.array(
        [
            [-0.8, -0.5, 0.2, 0.5],
            [-0.6, 0.1, 0.9, 0.7],
            [-0.2, 0.8, 1.2, 0.4],
            [-0.4, 0.3, 0.6, -0.1],
        ],
        dtype=float,
    )
    outer_truth = np.array([-0.25, 0.10, 0.20, -0.15, 0.05, 0.30, -0.10], dtype=float)
    observation_mean = 1800.0 + np.linspace(-0.2, 0.2, observation_count)
    noiseless = (
        observation_mean
        + np.einsum("ijk,jk->i", inner_design, inner_truth, optimize=True)
        + outer_design @ outer_truth
    )
    rng = np.random.default_rng(seed)
    observations = noiseless + rng.normal(scale=observation_error_sd, size=observation_count)
    observation_covariance = np.eye(observation_count) * observation_error_sd**2

    tree = DyadicTree.from_shape(_INNER_SHAPE)
    partitions = enumerate_partitions(tree)

    def partition_log_prior(partition: PartitionState) -> float:
        """Penalize each additional inner basis region exponentially."""
        partition.validate(tree)
        return -region_penalty * (len(partition.active) - 1)

    target = GaussianProductSpaceTarget.from_grid(
        observations=observations,
        observation_mean=observation_mean,
        inner_grid_design=inner_design,
        tree=tree,
        observation_covariance=observation_covariance,
        inner_prior_scale=inner_prior_sd,
        inactive_pseudo_prior_scale=1.0,
        outer_design=outer_design,
        outer_prior_covariance=np.eye(len(_OUTER_LABELS)) * outer_prior_sd**2,
        partition_log_prior=partition_log_prior,
    )
    return IntemSyntheticCase(
        regions=regions,
        rectangle=rectangle,
        outer_labels=_OUTER_LABELS,
        inner_grid_design=inner_design,
        outer_design=outer_design,
        inner_truth=inner_truth,
        outer_truth=outer_truth,
        observation_mean=observation_mean,
        observations=observations,
        observation_covariance=observation_covariance,
        inner_prior_sd=inner_prior_sd,
        outer_prior_sd=outer_prior_sd,
        target=target,
        partitions=partitions,
    )


def run_demo(
    case: IntemSyntheticCase,
    *,
    draws: int = 2_000,
    tune: int = 1_000,
    seed: int = 20260717,
) -> IntemDemoSummary:
    """Sample the product space and compare it with exact partition weights.

    Args:
        case: Synthetic experiment returned by :func:`build_synthetic_case`.
        draws: Number of retained PyMC draws.
        tune: Number of PyMC tuning draws.
        seed: Compound-step and chain seed.

    Returns:
        Compact exact-versus-sampled partition summary.

    Raises:
        ValueError: If ``draws`` is not positive or ``tune`` is negative.
    """
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer.")
    if isinstance(tune, bool) or not isinstance(tune, int) or tune < 0:
        raise ValueError("tune must be a non-negative integer.")

    exact_by_partition = case.target.partition_probabilities(case.partitions)
    exact = np.array([exact_by_partition[partition] for partition in case.partitions])
    map_index = int(np.argmax(exact))
    adapter = build_pymc_product_space_model(
        case.target,
        case.partitions,
        initial_partition=case.partitions[map_index],
    )
    steps = adapter.step_methods(nuts_kwargs={"target_accept": 0.85})
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

    sampled_indices = np.asarray(trace.posterior["partition_index"]).reshape(-1)
    sampled = np.bincount(sampled_indices, minlength=len(case.partitions)) / sampled_indices.size
    region_counts = np.array([len(partition.active) for partition in case.partitions], dtype=int)
    exact_by_region_count = np.bincount(region_counts, weights=exact)
    sampled_by_region_count = np.bincount(region_counts, weights=sampled)
    return IntemDemoSummary(
        rectangle=case.rectangle,
        observation_count=case.observations.size,
        partition_count=len(case.partitions),
        outer_region_count=len(case.outer_labels),
        inner_prior_sd=case.inner_prior_sd,
        outer_prior_sd=case.outer_prior_sd,
        exact_map_index=map_index,
        exact_map_inner_regions=int(region_counts[map_index]),
        exact_map_probability=float(exact[map_index]),
        exact_expected_inner_regions=float(exact @ region_counts),
        sampled_expected_inner_regions=float(sampled @ region_counts),
        sampled_unique_partitions=int(np.count_nonzero(sampled)),
        inner_region_count_total_variation_distance=float(
            0.5 * np.abs(sampled_by_region_count - exact_by_region_count).sum()
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for the synthetic demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--tune", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--region-penalty", type=float, default=0.12)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the demonstration and print its JSON report."""
    args = build_parser().parse_args(argv)
    case = build_synthetic_case(region_penalty=args.region_penalty, seed=args.seed)
    summary = run_demo(case, draws=args.draws, tune=args.tune, seed=args.seed)
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 0


def _synthetic_designs(
    regions: xr.DataArray,
    rectangle: InnerRectangle,
    *,
    observation_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build smooth local sensitivities and fixed outer-region aggregates.

    Args:
        regions: Raw InTEM region classes.
        rectangle: Dynamic inner rectangle.
        observation_count: Number of sensitivity rows to construct.

    Returns:
        Inner design with shape ``(observation, 4, 4)`` and outer design with
        one column for each raw InTEM label outside the rectangle.
    """
    region_values = np.asarray(regions.values)
    rows, columns = np.indices(region_values.shape, dtype=float)
    inner_mask = np.zeros(region_values.shape, dtype=bool)
    inner_mask[rectangle.row_slice, rectangle.column_slice] = True
    outer_masks = tuple((region_values == label) & ~inner_mask for label in _OUTER_LABELS)
    inner_design: np.ndarray = np.empty((observation_count, *_INNER_SHAPE), dtype=float)
    outer_design: np.ndarray = np.empty(
        (observation_count, len(_OUTER_LABELS)),
        dtype=float,
    )

    for observation_index in range(observation_count):
        inner_offset = observation_index % (_INNER_SHAPE[0] * _INNER_SHAPE[1])
        local_row = rectangle.row_start + inner_offset // _INNER_SHAPE[1]
        local_column = rectangle.column_start + inner_offset % _INNER_SHAPE[1]
        local_scale = 0.65 + 0.15 * (observation_index % 3)
        local = np.exp(
            -0.5
            * (
                np.square((rows - local_row) / local_scale)
                + np.square((columns - local_column) / local_scale)
            )
        )
        local /= local.sum()

        broad_row = 20 + (37 * observation_index) % (region_values.shape[0] - 40)
        broad_column = 20 + (53 * observation_index) % (region_values.shape[1] - 40)
        broad = np.exp(
            -0.5 * (np.square((rows - broad_row) / 24.0) + np.square((columns - broad_column) / 32.0))
        )
        broad /= broad.sum()
        sensitivity = local + 0.35 * broad
        inner_design[observation_index] = sensitivity[
            rectangle.row_slice,
            rectangle.column_slice,
        ]
        outer_design[observation_index] = [float(sensitivity[outer_mask].sum()) for outer_mask in outer_masks]

    return inner_design, outer_design


if __name__ == "__main__":
    raise SystemExit(main())
