"""Demonstrate a grouped Gamma--Beta prior on the EUROPE test grid.

The example uses the committed TAC/MHD OGI fixtures only to obtain a realistic
prior-flux field and grid. Expected additive mass is absolute prior flux times
grid area. The packaged InTEM classes ``0`` through ``5`` remain six fixed
outer regions. InTEM class ``6`` is intersected with the packaged land/ocean
mask and refined as two semantic groups with one local tree per connected
component. A fixed inner-region budget is allocated between land and ocean in
proportion to the standard mean footprint-times-flux basis weight. Candidate
dyadic splits within each group are then selected by that same weight.

This is a prior-simulation prototype, not an inversion. It demonstrates the
depth-pluggable concentration boundary, positive scaling fields, exact
parent/child flux conservation, fixed outer geometry, and land/ocean-separated
inner refinements.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.resources import as_file, files
import json
import math
from pathlib import Path
from typing import Any, Literal

from matplotlib import colors
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr

from openghg_inversions._country_file import load_country_dataset
from openghg_inversions.basis.algorithms import allocate_nbasis_by_class
from openghg_inversions.basis.experimental.dyadic.covariance_fit import (
    ExponentialLengthScaleFit,
    fit_projected_exponential_length_scale,
    projected_exponential_covariance,
)
from openghg_inversions.basis.experimental.dyadic.demo_data import load_tac_mhd_week_demo_data
from openghg_inversions.basis.experimental.dyadic.gamma_beta import (
    DepthKappaStrategy,
    GammaBetaForest,
    GammaBetaGroupSpec,
    GammaBetaSamples,
    MomentSplitConstraint,
    gamma_beta_child_moments,
)
from openghg_inversions.utils import areagrid

_DEFAULT_OUTPUT = Path("docs/plans/figures/dyadic_gamma_beta_intem")
_INTEM_RESOURCE = "outer_region_definition_EUROPE.nc"
_LAND_OCEAN_RESOURCE = "country-EUROPE-UKMO-landsea-2023.nc"
_COUNTRY_FILE = "country_EUROPE.nc"
_OUTER_REGION_COUNT = 6
_INNER_CLASS = 6
_OCEAN_CLASS = 0
_LAND_CLASS = 1


@dataclass(frozen=True, slots=True, eq=False)
class IntemGammaBetaCase:
    """Inputs and draws for the grouped EUROPE Gamma--Beta demonstration."""

    latitude: npt.NDArray[np.float64]
    longitude: npt.NDArray[np.float64]
    prior_flux: npt.NDArray[np.float64]
    expected_mass: npt.NDArray[np.float64]
    partition_weight: npt.NDArray[np.float64]
    intem_classes: npt.NDArray[np.int64]
    land_ocean: npt.NDArray[np.int64]
    group_labels: npt.NDArray[np.int64]
    forest: GammaBetaForest
    samples: GammaBetaSamples
    strategy: DepthKappaStrategy
    inner_region_targets: tuple[int, int]
    topology_weight_mode: Literal["sensitivity", "flat"]
    split_constraint: MomentSplitConstraint | None


@dataclass(frozen=True, slots=True)
class IntemGammaBetaSummary:
    """Compact diagnostics written by the demonstration."""

    draws: int
    group_count: int
    component_root_count: int
    fixed_outer_region_count: int
    inner_land_grid_cells: int
    inner_ocean_grid_cells: int
    inner_land_components: int
    inner_ocean_components: int
    inner_region_budget: int
    inner_land_regions: int
    inner_ocean_regions: int
    inner_land_partition_weight: float
    inner_ocean_partition_weight: float
    stochastic_coordinate_count: int
    forest_node_count: int
    leaf_count: int
    internal_split_count: int
    minimum_kappa: float
    maximum_kappa: float
    minimum_beta_shape: float
    minimum_leaf_variance: float
    median_leaf_variance: float
    maximum_leaf_variance: float
    median_inner_land_correlation: float
    terminal_covariance_rank: int
    maximum_conservation_error: float
    maximum_leaf_mean_error: float
    expected_mass_weighted_leaf_mean_error: float
    depth_diagnostics: tuple[dict[str, float | int], ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "draws": self.draws,
            "group_count": self.group_count,
            "component_root_count": self.component_root_count,
            "fixed_outer_region_count": self.fixed_outer_region_count,
            "inner_land_grid_cells": self.inner_land_grid_cells,
            "inner_ocean_grid_cells": self.inner_ocean_grid_cells,
            "inner_land_components": self.inner_land_components,
            "inner_ocean_components": self.inner_ocean_components,
            "inner_region_budget": self.inner_region_budget,
            "inner_land_regions": self.inner_land_regions,
            "inner_ocean_regions": self.inner_ocean_regions,
            "inner_land_partition_weight": self.inner_land_partition_weight,
            "inner_ocean_partition_weight": self.inner_ocean_partition_weight,
            "stochastic_coordinate_count": self.stochastic_coordinate_count,
            "forest_node_count": self.forest_node_count,
            "leaf_count": self.leaf_count,
            "internal_split_count": self.internal_split_count,
            "minimum_kappa": self.minimum_kappa,
            "maximum_kappa": self.maximum_kappa,
            "minimum_beta_shape": self.minimum_beta_shape,
            "minimum_leaf_variance": self.minimum_leaf_variance,
            "median_leaf_variance": self.median_leaf_variance,
            "maximum_leaf_variance": self.maximum_leaf_variance,
            "median_inner_land_correlation": self.median_inner_land_correlation,
            "terminal_covariance_rank": self.terminal_covariance_rank,
            "maximum_conservation_error": self.maximum_conservation_error,
            "maximum_leaf_mean_error": self.maximum_leaf_mean_error,
            "expected_mass_weighted_leaf_mean_error": self.expected_mass_weighted_leaf_mean_error,
            "depth_diagnostics": list(self.depth_diagnostics),
        }


@dataclass(frozen=True, slots=True, eq=False)
class IntemDistanceCovarianceComparison:
    """Distance-kernel fits and matrices for the terminal regional state."""

    latitude: npt.NDArray[np.float64]
    longitude: npt.NDArray[np.float64]
    group_labels: npt.NDArray[np.int64]
    restriction_transpose: npt.NDArray[np.float64]
    current_covariance: npt.NDArray[np.float64]
    current_correlation: npt.NDArray[np.float64]
    projected_unit_grid_covariance: npt.NDArray[np.float64]
    same_sigma_covariance: npt.NDArray[np.float64]
    covariance_fit_correlation: npt.NDArray[np.float64]
    correlation_fit_correlation: npt.NDArray[np.float64]
    group_scale_covariance: npt.NDArray[np.float64]
    covariance_fit: ExponentialLengthScaleFit
    correlation_fit: ExponentialLengthScaleFit

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-serializable fit and covariance diagnostics."""

        def fit_dict(fit: ExponentialLengthScaleFit) -> dict[str, float | int | bool | None]:
            """Convert one immutable fit result to JSON-compatible scalars."""
            return {
                "length_scale_degrees": fit.length_scale,
                "rmse": fit.rmse,
                "relative_rmse": fit.relative_rmse,
                "target_model_correlation": fit.target_model_correlation,
                "pair_count": fit.pair_count,
                "converged": fit.converged,
            }

        return {
            "fit_group": "inner_land",
            "restriction": "expected-mass-weighted regional scaling average",
            "kernel": "exp(-abs(dlat)/ell) * exp(-abs(dlon)/ell)",
            "covariance_fit": fit_dict(self.covariance_fit),
            "correlation_fit": fit_dict(self.correlation_fit),
            "current_maximum_variance": float(np.max(np.diag(self.current_covariance))),
            "same_sigma_maximum_variance": float(np.max(np.diag(self.same_sigma_covariance))),
            "projected_unit_grid_maximum_variance": float(
                np.max(np.diag(self.projected_unit_grid_covariance))
            ),
            "group_scale_maximum_variance": float(np.max(np.diag(self.group_scale_covariance))),
        }


def build_case(
    *,
    data_directory: Path = Path("tests/data"),
    draws: int = 2_000,
    inner_regions: int = 250,
    max_depth: int = 8,
    base_kappa: float = 2.0,
    depth_multiplier: float = 2.0,
    max_kappa: float | None = 128.0,
    inner_root_variance: float = 1.0,
    inner_land_root_variance: float | None = None,
    inner_ocean_root_variance: float | None = None,
    outer_root_variance: float = 0.25,
    topology_weight_mode: Literal["sensitivity", "flat"] = "sensitivity",
    split_constraint: MomentSplitConstraint | None = None,
    seed: int = 20260718,
) -> IntemGammaBetaCase:
    """Build and sample the realistic grouped Gamma--Beta prior.

    Args:
        data_directory: Directory containing the committed TAC/MHD fixtures.
        draws: Number of independent prior draws.
        inner_regions: Total terminal-region budget allocated between inner
            land and ocean by standard basis weight.
        max_depth: Candidate Beta-tree depth available to satisfy each allocated
            inner-region target.
        base_kappa: Concentration at the first split in each component.
        depth_multiplier: Concentration multiplier per effective split depth.
        max_kappa: Optional concentration cap.
        inner_root_variance: Mean-one Gamma variance shared by components of
            each inner semantic group when a group-specific override is absent.
        inner_land_root_variance: Optional override for the inner-land Gamma
            root variance.
        inner_ocean_root_variance: Optional override for the inner-ocean Gamma
            root variance.
        outer_root_variance: Mean-one Gamma variance for each fixed InTEM outer
            region.
        topology_weight_mode: ``"sensitivity"`` allocates and prioritizes
            refinements using mean absolute footprint-times-flux sensitivity.
            ``"flat"`` gives every mapped grid cell equal topology weight.
            Gamma--Beta conservation mass is unchanged in either mode.
        split_constraint: Optional Beta-shape and child-variance limits applied
            during weighted best-first refinement.
        seed: NumPy random seed.

    Returns:
        Aligned masks, expected masses, grouped forest, and prior draws.

    Raises:
        ValueError: If fewer than two draws are requested, no candidate inner
            refinement is available, or the inner-region budget is incompatible
            with component and depth constraints.
    """
    if draws < 2:
        raise ValueError("draws must be at least 2 for empirical depth diagnostics.")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1 for the inner refinement demo.")
    land_root_variance = (
        inner_root_variance if inner_land_root_variance is None else inner_land_root_variance
    )
    ocean_root_variance = (
        inner_root_variance if inner_ocean_root_variance is None else inner_ocean_root_variance
    )
    data = load_tac_mhd_week_demo_data(data_directory)
    intem, land_ocean = _load_aligned_masks(data.lat, data.lon)
    area = areagrid(np.asarray(data.lat, dtype=np.float64), np.asarray(data.lon, dtype=np.float64))
    prior_flux = np.asarray(data.prior_flux, dtype=np.float64)
    expected_mass = np.abs(prior_flux) * np.asarray(area, dtype=np.float64)
    sensitivity_weight = np.asarray(
        np.mean(np.abs(data.G), axis=0, dtype=np.float64),
        dtype=np.float64,
    )
    if topology_weight_mode == "sensitivity":
        partition_weight = sensitivity_weight
    elif topology_weight_mode == "flat":
        partition_weight = np.ones(expected_mass.shape, dtype=np.float64)
    else:
        raise ValueError("topology_weight_mode must be 'sensitivity' or 'flat'.")

    groups: list[GammaBetaGroupSpec] = []
    group_labels = np.full(intem.shape, -1, dtype=np.int64)
    for outer_class in range(_OUTER_REGION_COUNT):
        mask = intem == outer_class
        groups.append(
            GammaBetaGroupSpec(
                name=f"intem_outer_{outer_class}",
                mask=mask,
                root_variance=outer_root_variance,
                max_depth=0,
            )
        )
        group_labels[mask] = outer_class

    inner_ocean = (intem == _INNER_CLASS) & (land_ocean == _OCEAN_CLASS)
    inner_land = (intem == _INNER_CLASS) & (land_ocean == _LAND_CLASS)
    minimum_per_class = max(
        _four_connected_component_count(inner_ocean),
        _four_connected_component_count(inner_land),
    )
    inner_classes = xr.DataArray(
        np.where(intem == _INNER_CLASS, land_ocean, np.nan),
        dims=("lat", "lon"),
        coords={"lat": data.lat, "lon": data.lon},
    )
    basis_weights = xr.DataArray(
        partition_weight,
        dims=("lat", "lon"),
        coords=inner_classes.coords,
    )
    inner_allocation = allocate_nbasis_by_class(
        basis_weights,
        inner_classes,
        inner_regions,
        allocation="weight" if topology_weight_mode == "sensitivity" else "area",
        min_regions_per_class=minimum_per_class,
    )
    ocean_regions = inner_allocation[_OCEAN_CLASS]
    land_regions = inner_allocation[_LAND_CLASS]
    groups.extend(
        (
            GammaBetaGroupSpec(
                name="inner_ocean",
                mask=inner_ocean,
                root_variance=ocean_root_variance,
                max_depth=max_depth,
                target_regions=ocean_regions,
            ),
            GammaBetaGroupSpec(
                name="inner_land",
                mask=inner_land,
                root_variance=land_root_variance,
                max_depth=max_depth,
                target_regions=land_regions,
            ),
        )
    )
    group_labels[inner_ocean] = _OUTER_REGION_COUNT
    group_labels[inner_land] = _OUTER_REGION_COUNT + 1
    if (group_labels < 0).any():
        raise ValueError("InTEM outer and inner land/ocean groups must cover the EUROPE grid.")

    strategy = DepthKappaStrategy(
        base_kappa=base_kappa,
        depth_multiplier=depth_multiplier,
        max_kappa=max_kappa,
    )
    forest = GammaBetaForest.from_groups(
        expected_mass,
        groups,
        partition_weight=partition_weight,
        kappa_strategy=strategy if split_constraint is not None else None,
        split_constraint=split_constraint,
        require_full_coverage=True,
    )
    samples = forest.sample(draws, kappa_strategy=strategy, rng=seed)
    return IntemGammaBetaCase(
        latitude=np.asarray(data.lat, dtype=np.float64),
        longitude=np.asarray(data.lon, dtype=np.float64),
        prior_flux=prior_flux,
        expected_mass=expected_mass,
        partition_weight=partition_weight,
        intem_classes=intem,
        land_ocean=land_ocean,
        group_labels=group_labels,
        forest=forest,
        samples=samples,
        strategy=strategy,
        inner_region_targets=(ocean_regions, land_regions),
        topology_weight_mode=topology_weight_mode,
        split_constraint=split_constraint,
    )


def summarize_case(case: IntemGammaBetaCase) -> IntemGammaBetaSummary:
    """Compute topology, conservation, moment, and depth diagnostics.

    Args:
        case: Demonstration inputs and prior draws.

    Returns:
        Compact deterministic and Monte Carlo diagnostics.
    """
    forest = case.forest
    samples = case.samples
    group_names = tuple(group.name for group in forest.groups)
    root_groups = [forest.nodes[node_id].group_index for node_id in forest.root_ids]
    inner_ocean_index = group_names.index("inner_ocean")
    inner_land_index = group_names.index("inner_land")
    inner_nodes = [node for node in forest.nodes if node.group_index >= _OUTER_REGION_COUNT]
    internal_nodes = [node for node in inner_nodes if node.child_ids]
    inner_ocean_leaves = [
        node_id for node_id in forest.leaf_ids if forest.nodes[node_id].group_index == inner_ocean_index
    ]
    inner_land_leaves = [
        node_id for node_id in forest.leaf_ids if forest.nodes[node_id].group_index == inner_land_index
    ]
    kappas = np.asarray([samples.kappa_by_node[node.node_id] for node in internal_nodes])
    leaf_means = samples.node_scalings[:, forest.leaf_ids].mean(axis=0)
    leaf_masses = np.asarray([forest.nodes[node_id].expected_mass for node_id in forest.leaf_ids])
    leaf_mean_errors = np.abs(leaf_means - 1.0)
    covariance = samples.analytic_leaf_covariance()
    correlation = _covariance_to_correlation(covariance)
    leaf_variances = np.diag(covariance)
    inner_land_leaf_indices = np.asarray(
        [
            leaf_index
            for leaf_index, node_id in enumerate(forest.leaf_ids)
            if forest.nodes[node_id].group_index == inner_land_index
        ],
        dtype=np.int64,
    )
    inner_land_correlation = correlation[np.ix_(inner_land_leaf_indices, inner_land_leaf_indices)]
    inner_land_off_diagonal = ~np.eye(inner_land_leaf_indices.size, dtype=bool)
    beta_shapes: list[float] = []
    for node in internal_nodes:
        context = forest.split_context(node.node_id)
        first_mass, second_mass = context.child_expected_masses
        p = first_mass / (first_mass + second_mass)
        kappa = samples.kappa_by_node[node.node_id]
        beta_shapes.extend((kappa * p, kappa * (1.0 - p)))
    return IntemGammaBetaSummary(
        draws=samples.draws,
        group_count=len(forest.groups),
        component_root_count=len(forest.root_ids),
        fixed_outer_region_count=_OUTER_REGION_COUNT,
        inner_land_grid_cells=int(np.count_nonzero(case.group_labels == _OUTER_REGION_COUNT + 1)),
        inner_ocean_grid_cells=int(np.count_nonzero(case.group_labels == _OUTER_REGION_COUNT)),
        inner_land_components=root_groups.count(inner_land_index),
        inner_ocean_components=root_groups.count(inner_ocean_index),
        inner_region_budget=sum(case.inner_region_targets),
        inner_land_regions=len(inner_land_leaves),
        inner_ocean_regions=len(inner_ocean_leaves),
        inner_land_partition_weight=float(
            case.partition_weight[case.group_labels == _OUTER_REGION_COUNT + 1].sum()
        ),
        inner_ocean_partition_weight=float(
            case.partition_weight[case.group_labels == _OUTER_REGION_COUNT].sum()
        ),
        stochastic_coordinate_count=len(forest.groups) + len(internal_nodes),
        forest_node_count=len(forest.nodes),
        leaf_count=len(forest.leaf_ids),
        internal_split_count=len(internal_nodes),
        minimum_kappa=float(kappas.min()),
        maximum_kappa=float(kappas.max()),
        minimum_beta_shape=float(np.min(beta_shapes)),
        minimum_leaf_variance=float(np.min(leaf_variances)),
        median_leaf_variance=float(np.median(leaf_variances)),
        maximum_leaf_variance=float(np.max(leaf_variances)),
        median_inner_land_correlation=float(np.median(inner_land_correlation[inner_land_off_diagonal])),
        terminal_covariance_rank=int(np.linalg.matrix_rank(covariance)),
        maximum_conservation_error=samples.maximum_conservation_error(),
        maximum_leaf_mean_error=float(np.max(leaf_mean_errors)),
        expected_mass_weighted_leaf_mean_error=float(np.average(leaf_mean_errors, weights=leaf_masses)),
        depth_diagnostics=_depth_diagnostics(case),
    )


def build_distance_covariance_comparison(
    case: IntemGammaBetaCase,
) -> IntemDistanceCovarianceComparison:
    """Fit conventional distance covariance to the inner-land terminal state.

    Args:
        case: Demonstration inputs and sampled Gamma--Beta prior.

    Returns:
        Two inner-land projected length-scale fits and full grouped reference
        matrices.

    Notes:
        The covariance fit fixes each regional marginal standard deviation to
        the Gamma--Beta value, while the correlation fit weights every
        regional pair equally. Both construct a native-grid covariance and
        apply the expected-mass restriction before evaluating residuals.
    """
    covariance = case.samples.analytic_leaf_covariance()
    correlation = _covariance_to_correlation(covariance)
    restriction_transpose = _leaf_restriction_transpose(case)
    latitude, longitude = _leaf_weighted_centroids(case, restriction_transpose)
    leaf_groups = np.asarray(
        [case.forest.nodes[node_id].group_index for node_id in case.forest.leaf_ids],
        dtype=np.int64,
    )
    group_names = tuple(group.name for group in case.forest.groups)
    inner_land_group = group_names.index("inner_land")
    inner_land_indices = np.flatnonzero(leaf_groups == inner_land_group)
    if inner_land_indices.size < 2:
        raise ValueError("At least two inner-land regions are required for a distance covariance fit.")

    standard_deviation = np.sqrt(np.diag(covariance))
    land_index = np.ix_(inner_land_indices, inner_land_indices)
    land_latitude, land_longitude, land_projection = _crop_projection_to_support(
        case.latitude,
        case.longitude,
        restriction_transpose[:, inner_land_indices],
    )
    covariance_fit = fit_projected_exponential_length_scale(
        covariance[land_index],
        land_latitude,
        land_longitude,
        land_projection,
        standard_deviation=standard_deviation[inner_land_indices],
    )
    correlation_fit = fit_projected_exponential_length_scale(
        correlation[land_index],
        land_latitude,
        land_longitude,
        land_projection,
        standard_deviation=np.ones(inner_land_indices.size, dtype=np.float64),
    )

    covariance_fit_projection = projected_exponential_covariance(
        case.latitude,
        case.longitude,
        restriction_transpose,
        covariance_fit.length_scale,
        class_labels=case.group_labels,
    )
    covariance_fit_correlation = _covariance_to_correlation(covariance_fit_projection)
    same_sigma_covariance = (
        standard_deviation[:, None] * standard_deviation[None, :] * covariance_fit_correlation
    )
    projected_unit_grid_covariance = projected_exponential_covariance(
        case.latitude,
        case.longitude,
        restriction_transpose,
        correlation_fit.length_scale,
        class_labels=case.group_labels,
    )
    correlation_fit_correlation = _covariance_to_correlation(projected_unit_grid_covariance)
    group_scale_standard_deviation = np.asarray(
        [math.sqrt(case.forest.groups[group_index].root_variance) for group_index in leaf_groups],
        dtype=np.float64,
    )
    group_scale_covariance = (
        group_scale_standard_deviation[:, None]
        * group_scale_standard_deviation[None, :]
        * correlation_fit_correlation
    )
    return IntemDistanceCovarianceComparison(
        latitude=latitude,
        longitude=longitude,
        group_labels=leaf_groups,
        restriction_transpose=restriction_transpose,
        current_covariance=covariance,
        current_correlation=correlation,
        projected_unit_grid_covariance=projected_unit_grid_covariance,
        same_sigma_covariance=same_sigma_covariance,
        covariance_fit_correlation=covariance_fit_correlation,
        correlation_fit_correlation=correlation_fit_correlation,
        group_scale_covariance=group_scale_covariance,
        covariance_fit=covariance_fit,
        correlation_fit=correlation_fit,
    )


def write_report(
    case: IntemGammaBetaCase,
    summary: IntemGammaBetaSummary,
    output_directory: Path,
) -> None:
    """Write the figure, JSON summary, and short Markdown report.

    Args:
        case: Demonstration inputs and prior draws.
        summary: Diagnostics returned by :func:`summarize_case`.
        output_directory: Destination directory.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "intem_gamma_beta_summary.json"
    figure_path = output_directory / "intem_gamma_beta_summary.png"
    covariance_matrix_path = output_directory / "intem_gamma_beta_covariance_matrix.png"
    covariance_maps_path = output_directory / "intem_gamma_beta_covariance_maps.png"
    correlation_maps_path = output_directory / "intem_gamma_beta_correlation_maps.png"
    distance_fit_path = output_directory / "intem_gamma_beta_distance_fit.json"
    distance_matrices_path = output_directory / "intem_gamma_beta_distance_fit_matrices.png"
    distance_maps_path = output_directory / "intem_gamma_beta_distance_fit_maps.png"
    report_path = output_directory / "intem_gamma_beta_report.md"
    summary_path.write_text(json.dumps(summary.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n")
    _plot_case(case, summary, figure_path)
    covariance = case.samples.analytic_leaf_covariance()
    correlation = _covariance_to_correlation(covariance)
    selections = _representative_leaf_indices(case, covariance)
    _plot_covariance_matrices(case, covariance, correlation, covariance_matrix_path)
    _plot_leaf_metric_maps(
        case,
        covariance,
        selections,
        covariance_maps_path,
        metric_name="Covariance",
    )
    _plot_leaf_metric_maps(
        case,
        correlation,
        selections,
        correlation_maps_path,
        metric_name="Correlation",
    )
    comparison = build_distance_covariance_comparison(case)
    distance_fit_path.write_text(
        json.dumps(comparison.diagnostics(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _plot_distance_fit_matrices(case, comparison, distance_matrices_path)
    _plot_distance_fit_maps(case, comparison, distance_maps_path)
    report_path.write_text(_report_markdown(case, summary, comparison))


def _load_aligned_masks(
    latitude: npt.ArrayLike,
    longitude: npt.ArrayLike,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Load and validate packaged InTEM and land/ocean masks.

    Args:
        latitude: Target one-dimensional latitude coordinates.
        longitude: Target one-dimensional longitude coordinates.

    Returns:
        Integer InTEM and binary land/ocean arrays on the target grid.
    """
    target_latitude = np.asarray(latitude, dtype=np.float64)
    target_longitude = np.asarray(longitude, dtype=np.float64)
    intem_resource = files("openghg_inversions.basis").joinpath(_INTEM_RESOURCE)
    land_ocean_resource = files("openghg_inversions.basis.algorithms").joinpath(_LAND_OCEAN_RESOURCE)

    with as_file(intem_resource) as path, xr.open_dataset(path) as dataset:
        intem_data = dataset["region"].load()
    with as_file(land_ocean_resource) as path, xr.open_dataset(path) as dataset:
        land_ocean_data = dataset["country"].load()

    for name, data, tolerance in (
        ("InTEM", intem_data, 1.0e-4),
        ("land/ocean", land_ocean_data, 1.0e-8),
    ):
        if data.dims != ("lat", "lon") or data.shape != (target_latitude.size, target_longitude.size):
            raise ValueError(f"{name} mask must use the target ('lat', 'lon') grid.")
        if not np.allclose(data["lat"], target_latitude, rtol=0.0, atol=tolerance):
            raise ValueError(f"{name} latitude coordinates do not align to the target grid.")
        if not np.allclose(data["lon"], target_longitude, rtol=0.0, atol=tolerance):
            raise ValueError(f"{name} longitude coordinates do not align to the target grid.")

    intem = _validated_integer_mask(
        intem_data,
        name="EUROPE InTEM",
        allowed=np.arange(_INNER_CLASS + 1),
    )
    land_ocean = _validated_integer_mask(
        land_ocean_data,
        name="EUROPE land/ocean",
        allowed=np.array([_OCEAN_CLASS, _LAND_CLASS]),
    )
    return intem, land_ocean


def load_country_mask(
    case: IntemGammaBetaCase,
    data_directory: Path,
    country_name: str,
) -> npt.NDArray[np.bool_]:
    """Load one named country mask aligned to a demonstration case.

    Args:
        case: Target EUROPE-grid demonstration case.
        data_directory: Directory containing ``country_EUROPE.nc``.
        country_name: Exact entry in the country file's ``name`` variable.

    Returns:
        Read-only Boolean mask selecting the requested country.

    Raises:
        FileNotFoundError: If the country file is unavailable.
        ValueError: If the name is missing or coordinates do not align.
    """
    country_path = data_directory / _COUNTRY_FILE
    dataset = load_country_dataset(country_path)
    country = dataset["country"]
    names = np.asarray(dataset["name"].values, dtype=str)
    matches = np.flatnonzero(names == country_name)
    if matches.size != 1:
        raise ValueError(f"Country name {country_name!r} must appear exactly once.")
    if country.dims != ("lat", "lon") or country.shape != case.forest.shape:
        raise ValueError("Country mask must use the demonstration ('lat', 'lon') grid.")
    if not np.allclose(country["lat"], case.latitude, rtol=0.0, atol=1.0e-4):
        raise ValueError("Country-mask latitude coordinates do not align to the target grid.")
    if not np.allclose(country["lon"], case.longitude, rtol=0.0, atol=1.0e-4):
        raise ValueError("Country-mask longitude coordinates do not align to the target grid.")
    mask = np.asarray(country.values == int(matches[0]), dtype=bool)
    if not mask.any():
        raise ValueError(f"Country name {country_name!r} has no grid support.")
    mask.setflags(write=False)
    return mask


def _validated_integer_mask(
    values: npt.ArrayLike,
    *,
    name: str,
    allowed: npt.ArrayLike,
) -> npt.NDArray[np.int64]:
    """Validate integer-valued mask labels before converting their dtype.

    Args:
        values: Candidate numeric mask labels.
        name: Human-readable mask name used in errors.
        allowed: Complete allowed integer label set.

    Returns:
        Integer mask with the same shape as ``values``.

    Raises:
        ValueError: If values are non-finite, fractional, or outside the exact
            allowed label set.
    """
    numeric = np.asarray(values)
    try:
        finite = np.isfinite(numeric)
    except TypeError as error:
        raise ValueError(f"{name} mask labels must be finite integers.") from error
    if not finite.all():
        raise ValueError(f"{name} mask labels must be finite integers.")
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError(f"{name} mask labels must be integers before conversion.")
    labels = rounded.astype(np.int64)
    expected = np.sort(np.asarray(allowed, dtype=np.int64))
    if not np.array_equal(np.unique(labels), expected):
        raise ValueError(f"{name} mask must contain exactly labels {expected.tolist()}.")
    return labels


def _four_connected_component_count(mask: npt.NDArray[np.bool_]) -> int:
    """Count four-connected components in one non-empty Boolean mask."""
    remaining = np.asarray(mask, dtype=bool).copy()
    count = 0
    while remaining.any():
        start_row, start_column = np.argwhere(remaining)[0]
        pending = [(int(start_row), int(start_column))]
        remaining[start_row, start_column] = False
        count += 1
        while pending:
            row, column = pending.pop()
            for next_row, next_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if (
                    0 <= next_row < remaining.shape[0]
                    and 0 <= next_column < remaining.shape[1]
                    and remaining[next_row, next_column]
                ):
                    remaining[next_row, next_column] = False
                    pending.append((next_row, next_column))
    return count


def _depth_diagnostics(case: IntemGammaBetaCase) -> tuple[dict[str, float | int], ...]:
    """Summarize concentration and analytic sibling dependence by depth."""
    node_variance = np.full(len(case.forest.nodes), np.nan, dtype=np.float64)
    for root_id in case.forest.root_ids:
        root = case.forest.nodes[root_id]
        node_variance[root_id] = case.forest.groups[root.group_index].root_variance

    correlation_by_node: dict[int, float] = {}
    fraction_sd_by_node: dict[int, float] = {}
    for node in case.forest.nodes:
        if not node.child_ids:
            continue
        context = case.forest.split_context(node.node_id)
        first_mass, second_mass = context.child_expected_masses
        p = first_mass / (first_mass + second_mass)
        kappa = float(case.samples.kappa_by_node[node.node_id])
        moments = gamma_beta_child_moments(
            parent_variance=float(node_variance[node.node_id]),
            first_expected_fraction=p,
            kappa=kappa,
        )
        first_id, second_id = node.child_ids
        node_variance[first_id] = moments.first_variance
        node_variance[second_id] = moments.second_variance
        denominator = math.sqrt(moments.first_variance * moments.second_variance)
        if denominator > 0.0:
            correlation_by_node[node.node_id] = moments.covariance / denominator
        fraction_sd_by_node[node.node_id] = math.sqrt(p * (1.0 - p) / (kappa + 1.0))

    rows: list[dict[str, float | int]] = []
    for depth in sorted({node.depth for node in case.forest.nodes if node.child_ids}):
        nodes = [node for node in case.forest.nodes if node.child_ids and node.depth == depth]
        kappas = np.asarray([case.samples.kappa_by_node[node.node_id] for node in nodes])
        correlations = [correlation_by_node[node.node_id] for node in nodes]
        fraction_sd = [fraction_sd_by_node[node.node_id] for node in nodes]
        rows.append(
            {
                "depth": depth,
                "split_count": len(nodes),
                "kappa": float(np.median(kappas)),
                "median_sibling_correlation": float(np.median(correlations)),
                "median_split_fraction_sd": float(np.median(fraction_sd)),
            }
        )
    return tuple(rows)


def _covariance_to_correlation(covariance: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Normalize a covariance matrix while preserving undefined zero-variance rows.

    Args:
        covariance: Square finite covariance matrix.

    Returns:
        Correlation matrix. Entries involving a zero-variance state are NaN.

    Raises:
        ValueError: If the matrix is not square and finite or has a materially
            negative diagonal entry.
    """
    values = np.asarray(covariance, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if not np.isfinite(values).all():
        raise ValueError("covariance must be finite.")
    variance = np.diag(values)
    tolerance = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(variance)))) * 32.0
    if (variance < -tolerance).any():
        raise ValueError("covariance has a negative diagonal entry.")
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    denominator = np.outer(standard_deviation, standard_deviation)
    correlation = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(values, denominator, out=correlation, where=denominator > 0.0)
    finite = np.isfinite(correlation)
    correlation[finite] = np.clip(correlation[finite], -1.0, 1.0)
    return correlation


def _leaf_restriction_transpose(
    case: IntemGammaBetaCase,
) -> npt.NDArray[np.float64]:
    """Return the expected-mass regional restriction as an ``M x K`` matrix.

    Args:
        case: Demonstration inputs defining expected mass and terminal supports.

    Returns:
        Transposed restriction. Each column sums to one and maps native scaling
        factors to the corresponding expected-mass-weighted regional scaling.

    Raises:
        ValueError: If any terminal region has zero expected mass.
    """
    labels = case.forest.leaf_labels().reshape(-1) - 1
    expected_mass = case.expected_mass.reshape(-1)
    leaf_count = len(case.forest.leaf_ids)
    totals = np.bincount(labels, weights=expected_mass, minlength=leaf_count)
    if np.any(totals <= 0.0):
        raise ValueError("Every terminal region must have positive expected mass.")
    restriction = np.zeros((labels.size, leaf_count), dtype=np.float64)
    restriction[np.arange(labels.size), labels] = expected_mass / totals[labels]
    return restriction


def _leaf_weighted_centroids(
    case: IntemGammaBetaCase,
    restriction_transpose: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return expected-mass centroids used only to select display regions.

    Args:
        case: Demonstration inputs defining native coordinates.
        restriction_transpose: Expected-mass restriction with shape ``(M, K)``.

    Returns:
        Paired latitude and longitude arrays in terminal leaf order.
    """
    latitude_grid, longitude_grid = np.meshgrid(
        case.latitude,
        case.longitude,
        indexing="ij",
    )
    return (
        restriction_transpose.T @ latitude_grid.reshape(-1),
        restriction_transpose.T @ longitude_grid.reshape(-1),
    )


def _crop_projection_to_support(
    latitude: npt.NDArray[np.float64],
    longitude: npt.NDArray[np.float64],
    projection: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Crop a grid-by-region projection to its active rectangular bounds.

    Args:
        latitude: Native latitude coordinates.
        longitude: Native longitude coordinates.
        projection: Full-grid transposed restriction with shape ``(M, K)``.

    Returns:
        Cropped coordinates and row-major transposed restriction.

    Raises:
        ValueError: If the projection is inconsistent with the grid or has no
            active support.
    """
    grid_shape = (latitude.size, longitude.size)
    if projection.ndim != 2 or projection.shape[0] != math.prod(grid_shape):
        raise ValueError("projection must have one row per native grid location.")
    projection_grid = projection.reshape(*grid_shape, projection.shape[1])
    active = np.any(projection_grid != 0.0, axis=2)
    rows, columns = np.where(active)
    if rows.size == 0:
        raise ValueError("projection must have active native-grid support.")
    row_slice = slice(int(rows.min()), int(rows.max()) + 1)
    column_slice = slice(int(columns.min()), int(columns.max()) + 1)
    cropped = projection_grid[row_slice, column_slice, :]
    return (
        latitude[row_slice],
        longitude[column_slice],
        cropped.reshape(-1, projection.shape[1]),
    )


def _representative_leaf_indices(
    case: IntemGammaBetaCase,
    covariance: npt.NDArray[np.float64],
) -> tuple[tuple[str, int], ...]:
    """Select geographically recognizable and high-variance terminal regions.

    Args:
        case: Demonstration inputs and retained terminal labels.
        covariance: Terminal scaling covariance in leaf order.

    Returns:
        Six unique ``(display name, zero-based leaf index)`` selections.
    """
    labels = case.forest.leaf_labels()
    requested = (
        ("Mace Head", -9.9, 53.3),
        ("Tacolneston", 1.14, 52.5),
        ("Central Europe", 10.0, 50.0),
        ("Scandinavia", 15.0, 65.0),
        ("Mediterranean", 10.0, 40.0),
    )
    selected: list[tuple[str, int]] = []
    used: set[int] = set()
    for name, longitude, latitude in requested:
        row = int(np.argmin(np.abs(case.latitude - latitude)))
        column = int(np.argmin(np.abs(case.longitude - longitude)))
        leaf_index = int(labels[row, column]) - 1
        if leaf_index not in used:
            selected.append((name, leaf_index))
            used.add(leaf_index)

    variance_rank = 0
    for leaf_index in np.argsort(np.diag(covariance))[::-1]:
        index = int(leaf_index)
        if index not in used:
            variance_rank += 1
            name = "Highest variance" if variance_rank == 1 else f"High variance {variance_rank}"
            selected.append((name, index))
            used.add(index)
        if len(selected) == 6:
            break
    if len(selected) != 6:
        raise ValueError("Representative covariance maps require six unique terminal regions.")
    return tuple(selected)


def _plot_covariance_matrices(
    case: IntemGammaBetaCase,
    covariance: npt.NDArray[np.float64],
    correlation: npt.NDArray[np.float64],
    path: Path,
) -> None:
    """Plot exact covariance and correlation matrices in terminal leaf order.

    Args:
        case: Demonstration inputs defining leaf groups and order.
        covariance: Exact terminal scaling covariance matrix.
        correlation: Correlation matrix derived from ``covariance``.
        path: Destination image path.
    """
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    maximum_absolute_covariance = float(np.max(np.abs(covariance)))
    covariance_norm = colors.SymLogNorm(
        linthresh=max(maximum_absolute_covariance * 1.0e-3, float(np.finfo(np.float64).eps)),
        vmin=-maximum_absolute_covariance,
        vmax=maximum_absolute_covariance,
    )
    covariance_image = axes[0].imshow(
        covariance,
        origin="lower",
        aspect="equal",
        cmap="coolwarm",
        norm=covariance_norm,
        interpolation="nearest",
    )
    axes[0].set_title("Exact covariance of terminal scaling factors")
    figure.colorbar(covariance_image, ax=axes[0], shrink=0.82, label="Scaling covariance")

    correlation_image = axes[1].imshow(
        correlation,
        origin="lower",
        aspect="equal",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[1].set_title("Corresponding correlation matrix")
    figure.colorbar(correlation_image, ax=axes[1], shrink=0.82, label="Correlation")

    group_counts = np.asarray(
        [
            sum(case.forest.nodes[node_id].group_index == group_index for node_id in case.forest.leaf_ids)
            for group_index in range(len(case.forest.groups))
        ],
        dtype=np.int64,
    )
    outer_count = int(group_counts[:_OUTER_REGION_COUNT].sum())
    ocean_count = int(group_counts[_OUTER_REGION_COUNT])
    boundaries = np.asarray([outer_count, outer_count + ocean_count], dtype=np.int64)
    for axis in axes:
        for boundary in boundaries:
            axis.axhline(boundary - 0.5, color="black", linewidth=0.45, alpha=0.65)
            axis.axvline(boundary - 0.5, color="black", linewidth=0.45, alpha=0.65)
        axis.text(
            0.025,
            0.975,
            f"Outer: 0-{outer_count - 1}\n"
            f"Ocean: {outer_count}-{outer_count + ocean_count - 1}\n"
            f"Land: {outer_count + ocean_count}-{len(case.forest.leaf_ids) - 1}",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        axis.set_xlabel("Terminal-region state index")
        axis.set_ylabel("Terminal-region state index")
    figure.suptitle("Analytic Gamma--Beta prior dependence; groups ordered outer, ocean, land")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_distance_fit_matrices(
    case: IntemGammaBetaCase,
    comparison: IntemDistanceCovarianceComparison,
    path: Path,
) -> None:
    """Compare Gamma--Beta matrices with fitted distance-kernel references.

    Args:
        case: Demonstration inputs defining semantic group boundaries.
        comparison: Fitted distance covariance comparison.
        path: Destination image path.
    """
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    maximum_covariance = float(np.max(np.abs(comparison.current_covariance)))
    covariance_norm = colors.SymLogNorm(
        linthresh=max(maximum_covariance * 1.0e-3, float(np.finfo(np.float64).eps)),
        vmin=-maximum_covariance,
        vmax=maximum_covariance,
    )
    covariance_panels = (
        (
            comparison.current_covariance,
            "Gamma--Beta covariance",
            f"max variance={np.max(np.diag(comparison.current_covariance)):.3g}",
        ),
        (
            comparison.same_sigma_covariance,
            "Distance covariance; Gamma--Beta sigmas",
            f"ell={comparison.covariance_fit.length_scale:.3g} degrees",
        ),
        (
            comparison.projected_unit_grid_covariance,
            "Raw projected distance covariance",
            "unit native variance; exact P B P.T",
        ),
    )
    for axis, (matrix, title, subtitle) in zip(axes[0], covariance_panels):
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="equal",
            cmap="coolwarm",
            norm=covariance_norm,
            interpolation="nearest",
        )
        axis.set_title(f"{title}\n{subtitle}")
        figure.colorbar(image, ax=axis, shrink=0.78, label="Scaling covariance")

    correlation_panels = (
        (comparison.current_correlation, "Gamma--Beta correlation"),
        (
            comparison.covariance_fit_correlation,
            "Distance correlation at covariance-fit ell",
        ),
        (
            comparison.correlation_fit_correlation,
            "Distance correlation at correlation-fit ell",
        ),
    )
    for axis, (matrix, title) in zip(axes[1], correlation_panels):
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="equal",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.78, label="Correlation")

    outer_count = sum(
        case.forest.nodes[node_id].group_index < _OUTER_REGION_COUNT for node_id in case.forest.leaf_ids
    )
    ocean_count = sum(
        case.forest.nodes[node_id].group_index == _OUTER_REGION_COUNT for node_id in case.forest.leaf_ids
    )
    for axis in axes.flat:
        for boundary in (outer_count, outer_count + ocean_count):
            axis.axhline(boundary - 0.5, color="black", linewidth=0.45, alpha=0.65)
            axis.axvline(boundary - 0.5, color="black", linewidth=0.45, alpha=0.65)
        axis.set_xlabel("Terminal-region state index")
        axis.set_ylabel("Terminal-region state index")
    figure.suptitle("Inner-land exponential length-scale fit; ocean dependence is extrapolated")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _distance_fit_selections(
    case: IntemGammaBetaCase,
    comparison: IntemDistanceCovarianceComparison,
) -> tuple[tuple[str, int], ...]:
    """Select two inner-land and one inner-ocean region for fit maps.

    Args:
        case: Demonstration inputs defining group names.
        comparison: Leaf centroids and class labels in terminal order.

    Returns:
        Three unique named leaf indices.
    """
    group_names = tuple(group.name for group in case.forest.groups)
    land_group = group_names.index("inner_land")
    ocean_group = group_names.index("inner_ocean")
    requested = (
        ("Western inner land", 52.0, -2.0, land_group),
        ("Central European land", 50.0, 10.0, land_group),
        ("Mediterranean ocean", 40.0, 10.0, ocean_group),
    )
    selections: list[tuple[str, int]] = []
    used: set[int] = set()
    for name, latitude, longitude, group_index in requested:
        candidates = np.flatnonzero(comparison.group_labels == group_index)
        candidates = np.asarray([index for index in candidates if int(index) not in used], dtype=np.int64)
        if candidates.size == 0:
            raise ValueError(f"No unused terminal region is available for {name}.")
        distances = np.square(comparison.latitude[candidates] - latitude) + np.square(
            comparison.longitude[candidates] - longitude
        )
        leaf_index = int(candidates[int(np.argmin(distances))])
        selections.append((name, leaf_index))
        used.add(leaf_index)
    return tuple(selections)


def _plot_distance_fit_maps(
    case: IntemGammaBetaCase,
    comparison: IntemDistanceCovarianceComparison,
    path: Path,
) -> None:
    """Map current, fitted, and residual correlations for selected regions.

    Args:
        case: Demonstration inputs defining terminal supports and coordinates.
        comparison: Fitted distance covariance comparison.
        path: Destination image path.
    """
    selections = _distance_fit_selections(case, comparison)
    labels = case.forest.leaf_labels()
    extent = (
        float(case.longitude.min()),
        float(case.longitude.max()),
        float(case.latitude.min()),
        float(case.latitude.max()),
    )
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    correlation_image = None
    residual_image = None
    for row, (name, leaf_index) in enumerate(selections):
        current = comparison.current_correlation[leaf_index]
        fitted = comparison.correlation_fit_correlation[leaf_index]
        matrices = (current, fitted, current - fitted)
        for column, (axis, values) in enumerate(zip(axes[row], matrices)):
            regional_grid = _regional_values_to_grid(labels, values)
            if column < 2:
                image = axis.imshow(
                    regional_grid,
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
                correlation_image = image
            else:
                image = axis.imshow(
                    regional_grid,
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                    cmap="coolwarm",
                    vmin=-1.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
                residual_image = image
            _add_terminal_boundaries(axis, case, color="white", linewidth=0.55, alpha=0.75)
            _add_terminal_boundaries(axis, case, color="black", linewidth=0.18, alpha=0.8)
            axis.contour(
                labels == leaf_index + 1,
                levels=(0.5,),
                origin="lower",
                extent=extent,
                colors="black",
                linewidths=1.0,
            )
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
        axes[row, 0].set_ylabel(f"{name}\nLatitude")
    for axis, title in zip(
        axes[0],
        (
            "Gamma--Beta correlation",
            "Projected distance correlation",
            "Gamma--Beta minus projected distance",
        ),
    ):
        axis.set_title(title)
    if correlation_image is None or residual_image is None:
        raise ValueError("Distance-fit maps require at least one selected terminal region.")
    figure.colorbar(
        correlation_image,
        ax=axes[:, :2].ravel().tolist(),
        shrink=0.75,
        label="Scaling correlation",
    )
    figure.colorbar(
        residual_image,
        ax=axes[:, 2].ravel().tolist(),
        shrink=0.75,
        label="Correlation residual",
    )
    figure.suptitle(
        "Distance reference fitted to inner-land pairs; Mediterranean ocean row is an extrapolation"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _regional_values_to_grid(
    labels: npt.ArrayLike,
    regional_values: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Broadcast one regional value over each positive terminal-label support.

    Args:
        labels: Positive integer terminal labels with arbitrary grid shape.
        regional_values: One value per label in consecutive label order.

    Returns:
        Grid with a constant value inside every labelled terminal region.

    Raises:
        ValueError: If labels or regional values are invalid or inconsistent.
    """
    label_values = np.asarray(labels)
    values = np.asarray(regional_values, dtype=np.float64)
    if label_values.ndim != 2 or values.ndim != 1:
        raise ValueError("labels must be two-dimensional and regional_values one-dimensional.")
    if not np.issubdtype(label_values.dtype, np.integer) or np.any(label_values < 1):
        raise ValueError("labels must contain positive integers.")
    if int(np.max(label_values)) > values.size or not np.isfinite(values).all():
        raise ValueError("regional_values must be finite and cover every terminal label.")
    return values[label_values - 1]


def _add_terminal_boundaries(
    axis: Any,
    case: IntemGammaBetaCase,
    *,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    """Overlay exact terminal-region boundaries on a geographic axis.

    Args:
        axis: Matplotlib axis receiving one line collection.
        case: Demonstration inputs defining coordinates and terminal labels.
        color: Boundary line color.
        linewidth: Boundary line width in points.
        alpha: Boundary opacity.
    """
    labels = case.forest.leaf_labels()
    latitude_edges = _coordinate_edges(case.latitude)
    longitude_edges = _coordinate_edges(case.longitude)
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    rows, columns = np.where(labels[:, :-1] != labels[:, 1:])
    for row, column in zip(rows, columns):
        longitude = float(longitude_edges[column + 1])
        segments.append(
            (
                (longitude, float(latitude_edges[row])),
                (longitude, float(latitude_edges[row + 1])),
            )
        )

    rows, columns = np.where(labels[:-1, :] != labels[1:, :])
    for row, column in zip(rows, columns):
        latitude = float(latitude_edges[row + 1])
        segments.append(
            (
                (float(longitude_edges[column]), latitude),
                (float(longitude_edges[column + 1]), latitude),
            )
        )

    axis.add_collection(
        LineCollection(
            segments,
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            zorder=3,
        )
    )


def _coordinate_edges(coordinates: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return plotting edges surrounding ordered one-dimensional centers."""
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("coordinates must contain at least two finite one-dimensional values.")
    midpoints = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        (
            np.asarray([values[0] - (midpoints[0] - values[0])]),
            midpoints,
            np.asarray([values[-1] + (values[-1] - midpoints[-1])]),
        )
    )


def _plot_leaf_metric_maps(
    case: IntemGammaBetaCase,
    matrix: npt.NDArray[np.float64],
    selections: tuple[tuple[str, int], ...],
    path: Path,
    *,
    metric_name: str,
) -> None:
    """Map selected rows of a terminal covariance or correlation matrix.

    Args:
        case: Demonstration inputs defining terminal supports and coordinates.
        matrix: Terminal covariance or correlation matrix in leaf order.
        selections: Named zero-based terminal indices to display.
        path: Destination image path.
        metric_name: Either ``"Covariance"`` or ``"Correlation"``.

    Raises:
        ValueError: If ``metric_name`` is unsupported.
    """
    if metric_name not in {"Covariance", "Correlation"}:
        raise ValueError("metric_name must be 'Covariance' or 'Correlation'.")
    labels = case.forest.leaf_labels()
    extent = (
        float(case.longitude.min()),
        float(case.longitude.max()),
        float(case.latitude.min()),
        float(case.latitude.max()),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    selected_rows = np.asarray([matrix[leaf_index] for _, leaf_index in selections])
    if metric_name == "Covariance":
        maximum = float(np.nanmax(np.abs(selected_rows)))
        norm: colors.Normalize = colors.SymLogNorm(
            linthresh=max(maximum * 1.0e-3, float(np.finfo(np.float64).eps)),
            vmin=-maximum,
            vmax=maximum,
        )
        colorbar_label = "Scaling covariance"
    else:
        norm = colors.Normalize(vmin=-1.0, vmax=1.0)
        colorbar_label = "Scaling correlation"

    image = None
    for axis, (name, leaf_index) in zip(axes.flat, selections):
        grid = _regional_values_to_grid(labels, matrix[leaf_index])
        image = axis.imshow(
            grid,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="coolwarm",
            norm=norm,
            interpolation="nearest",
        )
        selected_mask = labels == leaf_index + 1
        axis.contour(
            selected_mask,
            levels=(0.5,),
            origin="lower",
            extent=extent,
            colors="black",
            linewidths=1.0,
        )
        node = case.forest.nodes[case.forest.leaf_ids[leaf_index]]
        group_name = case.forest.groups[node.group_index].name.replace("inner_", "")
        variance = matrix[leaf_index, leaf_index] if metric_name == "Covariance" else np.nan
        variance_text = f", variance={variance:.3g}" if metric_name == "Covariance" else ""
        axis.set_title(f"{name}: region {leaf_index + 1} ({group_name}){variance_text}")
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
    if image is None:
        raise ValueError("At least one terminal region is required for spatial maps.")
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, label=colorbar_label)
    figure.suptitle(f"{metric_name} between each selected terminal scaling and the full regional state")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_case(
    case: IntemGammaBetaCase,
    summary: IntemGammaBetaSummary,
    path: Path,
) -> None:
    """Plot groups, terminal regions, prior draws, and depth diagnostics."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    extent = (
        float(case.longitude.min()),
        float(case.longitude.max()),
        float(case.latitude.min()),
        float(case.latitude.max()),
    )
    axes[0, 0].imshow(case.group_labels, origin="lower", extent=extent, aspect="auto", cmap="tab10")
    axes[0, 0].set_title("Six fixed outer groups + inner ocean/land")
    positive_weight = case.partition_weight[case.partition_weight > 0.0]
    weight_floor = float(np.min(positive_weight))
    weight_image = axes[0, 1].imshow(
        np.log10(np.maximum(case.partition_weight, weight_floor)),
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="magma",
    )
    axes[0, 1].set_title("log10 mean footprint-times-flux weight")
    figure.colorbar(weight_image, ax=axes[0, 1], shrink=0.75)
    axes[0, 2].imshow(
        case.forest.leaf_labels(),
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="nipy_spectral",
        interpolation="nearest",
    )
    axes[0, 2].set_title(
        "Terminal regions: "
        f"{summary.inner_land_regions} land + {summary.inner_ocean_regions} ocean + "
        f"{summary.fixed_outer_region_count} outer"
    )

    draw_indices = np.linspace(0, case.samples.draws - 1, 2, dtype=int)
    rendered = [case.samples.to_grid(int(draw)) for draw in draw_indices]
    finite = np.concatenate([values[np.isfinite(values)] for values in rendered])
    lower, upper = np.quantile(finite, (0.02, 0.98))
    for axis, draw, values in zip((axes[1, 0], axes[1, 1]), draw_indices, rendered):
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis",
            vmin=float(lower),
            vmax=float(upper),
        )
        axis.set_title(f"Prior scaling draw {draw}")
        figure.colorbar(image, ax=axis, shrink=0.75)

    depths = np.asarray([row["depth"] for row in summary.depth_diagnostics], dtype=int)
    correlations = np.asarray(
        [row["median_sibling_correlation"] for row in summary.depth_diagnostics],
        dtype=float,
    )
    kappas = np.asarray([row["kappa"] for row in summary.depth_diagnostics], dtype=float)
    diagnostic_axis = axes[1, 2]
    diagnostic_axis.plot(depths, correlations, marker="o", color="tab:blue", label="sibling correlation")
    diagnostic_axis.set_xlabel("Effective split depth")
    diagnostic_axis.set_ylabel("Median sibling correlation", color="tab:blue")
    diagnostic_axis.tick_params(axis="y", labelcolor="tab:blue")
    kappa_axis = diagnostic_axis.twinx()
    kappa_axis.plot(depths, kappas, marker="s", color="tab:red", label="kappa")
    kappa_axis.set_ylabel("Kappa", color="tab:red")
    kappa_axis.tick_params(axis="y", labelcolor="tab:red")
    diagnostic_axis.set_title("Depth policy and induced tree-local dependence")

    for axis in axes.flat[:-1]:
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
    figure.suptitle("Experimental Gamma--Beta prior with InTEM and land/ocean groups")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _report_markdown(
    case: IntemGammaBetaCase,
    summary: IntemGammaBetaSummary,
    comparison: IntemDistanceCovarianceComparison,
) -> str:
    """Return the shareable Markdown report body.

    Args:
        case: Demonstration inputs and prior draws.
        summary: Gamma--Beta topology and numerical diagnostics.
        comparison: Inner-land distance-covariance fits.

    Returns:
        Complete Markdown report text.
    """
    depth_rows = "\n".join(
        "| {depth} | {split_count} | {kappa:.3g} | {median_sibling_correlation:.3f} | "
        "{median_split_fraction_sd:.3f} |".format(**row)
        for row in summary.depth_diagnostics
    )
    covariance_fit_correlation = comparison.covariance_fit.target_model_correlation
    covariance_fit_correlation_text = (
        "undefined" if covariance_fit_correlation is None else f"{covariance_fit_correlation:.4f}"
    )
    correlation_fit_correlation = comparison.correlation_fit.target_model_correlation
    correlation_fit_correlation_text = (
        "undefined" if correlation_fit_correlation is None else f"{correlation_fit_correlation:.4f}"
    )
    return f"""# InTEM land/ocean Gamma--Beta prior prototype

This prior-only demonstration uses absolute OGI test prior flux multiplied by
grid area as additive split mass. InTEM outer classes 0--5 remain fixed
geometries with uncertain Gamma root scalings. InTEM class 6 is split into
inner-ocean and inner-land groups; disconnected components receive separate
local Beta trees but share their semantic group's Gamma root draw.

![Grouped Gamma--Beta prior](intem_gamma_beta_summary.png)

## Configuration and topology

- draws: {summary.draws}
- depth policy: `kappa(d) = {case.strategy.base_kappa:g} * {case.strategy.depth_multiplier:g}**d`
- kappa cap: {case.strategy.max_kappa}
- semantic groups: {summary.group_count}
- component roots: {summary.component_root_count}
- fixed InTEM outer regions: {summary.fixed_outer_region_count}
- inner land/ocean grid cells: {summary.inner_land_grid_cells} / {summary.inner_ocean_grid_cells}
- inner land/ocean components: {summary.inner_land_components} / {summary.inner_ocean_components}
- inner land/ocean total basis weight: {summary.inner_land_partition_weight:.6g} / {summary.inner_ocean_partition_weight:.6g}
- allocated inner land/ocean regions: {summary.inner_land_regions} / {summary.inner_ocean_regions}
- total inner-region budget: {summary.inner_region_budget}
- terminal regions: {summary.leaf_count}
- stochastic prior coordinates: {summary.stochastic_coordinate_count}
- stochastic Beta splits: {summary.internal_split_count}

## Numerical checks

- maximum parent/child conservation error: {summary.maximum_conservation_error:.6e}
- maximum empirical leaf mean error from one: {summary.maximum_leaf_mean_error:.4f}
- expected-mass-weighted empirical leaf mean error: {summary.expected_mass_weighted_leaf_mean_error:.4f}
- kappa range: {summary.minimum_kappa:g}--{summary.maximum_kappa:g}
- minimum Beta shape: {summary.minimum_beta_shape:.4g}
- terminal scaling variance range: {summary.minimum_leaf_variance:.4g}--{summary.maximum_leaf_variance:.4g}
- median terminal scaling variance: {summary.median_leaf_variance:.4g}
- median off-diagonal inner-land correlation: {summary.median_inner_land_correlation:.4f}
- terminal scaling covariance rank: {summary.terminal_covariance_rank} of {summary.leaf_count}

| Effective depth | Splits | Kappa | Median sibling correlation | Median split-fraction SD |
| ---: | ---: | ---: | ---: | ---: |
{depth_rows}

Sibling correlations and split-fraction standard deviations in this table are
analytic. Correlation is not a function of kappa alone: it also depends on the
inherited parent variance and each split's expected-mass fraction.

## Exact terminal-state covariance

The following matrices are analytic, not empirical estimates from the 2,000
draws. Rows and columns are terminal-region scaling factors in forest leaf
order: six fixed outer regions, three ocean supports, then 247 land regions.
Black divider lines mark those semantic groups. Covariance colors use a
symmetric logarithmic normalization so both small and large values remain
visible.

![Terminal covariance and correlation matrices](intem_gamma_beta_covariance_matrix.png)

Each map below takes one matrix row and broadcasts its values over the
corresponding terminal-region supports. The outlined region is the selected
state-vector element. Covariance is shown first; normalized correlation is
shown separately because scaling variances differ by nearly two orders of
magnitude.

![Selected terminal covariance maps](intem_gamma_beta_covariance_maps.png)

![Selected terminal correlation maps](intem_gamma_beta_correlation_maps.png)

## Exponential distance-covariance comparison

As a conventional reference, define a unit-variance covariance between native
grid locations `a` and `b` as

```text
B[a, b] = exp(-abs(lat[a] - lat[b]) / ell)
          exp(-abs(lon[a] - lon[b]) / ell).
```

The terminal scaling in region `r` is the expected-flux-mass-weighted native
scaling, so the restriction is

```text
P[r, a] = expected_mass[a] / sum(expected_mass in r)
```

inside region `r` and zero elsewhere. The regional distance covariance is then
computed exactly as `B_P = P B P.T`. The separable implementation applies this
without materializing the full native-grid covariance. It therefore integrates
over the actual terminal-region supports rather than approximating each region
by a centroid.

Only unique off-diagonal inner-land pairs enter the least-squares fits. Six
InTEM outer regions remain separate hard groups, while inner land and inner
ocean are the other two groups; native covariance is zero across group
boundaries. The fitted land scale is applied to ocean only as an extrapolation.

Two fits answer different questions:

- covariance fit with Gamma--Beta marginal standard deviations fixed:
  `ell = {comparison.covariance_fit.length_scale:.4g}` degrees, RMSE
  `{comparison.covariance_fit.rmse:.4g}`, relative RMSE
  `{comparison.covariance_fit.relative_rmse:.4g}`, target/model pair
  correlation `{covariance_fit_correlation_text}` over
  {comparison.covariance_fit.pair_count} pairs;
- correlation fit with every marginal standard deviation fixed to one:
  `ell = {comparison.correlation_fit.length_scale:.4g}` degrees, RMSE
  `{comparison.correlation_fit.rmse:.4g}`, relative RMSE
  `{comparison.correlation_fit.relative_rmse:.4g}`, target/model pair
  correlation `{correlation_fit_correlation_text}` over
  {comparison.correlation_fit.pair_count} pairs.

![Distance covariance matrix comparison](intem_gamma_beta_distance_fit_matrices.png)

The first covariance fit normalizes the projected matrix to correlation and
then restores every Gamma--Beta regional standard deviation, including the
maximum variance of {np.max(np.diag(comparison.same_sigma_covariance)):.4g}.
It diagnoses only the dependence shape. The third covariance panel is the raw
`P B P.T` result with unit native-grid variance; its largest regional-average
variance is {np.max(np.diag(comparison.projected_unit_grid_covariance)):.4g}.
A further group-scale reference sets regional standard deviation to one for
inner land/ocean and 0.5 for each outer region, giving maximum variance
{np.max(np.diag(comparison.group_scale_covariance)):.4g}. Both are different
priors, not alternative fits with the Gamma--Beta marginals.

The maps compare Gamma--Beta correlation with the normalized regional
correlation obtained from `P B P.T`. Both matrix rows are therefore displayed
on exactly the same terminal regions. Thin black-and-white lines mark every
terminal boundary because smoothly varying regional values can otherwise hide
the piecewise-constant boxes. They include two land regions and one ocean
region so the shared-ocean-root behavior is visible. The ocean row is not
evidence for an ocean length scale because no ocean pairs were fit.

![Distance correlation map comparison](intem_gamma_beta_distance_fit_maps.png)

This is a useful baseline but not a definitive spatial model. Projection now
preserves irregular and disconnected support geometry, but one common scale in
latitude and longitude degrees is not isotropic in physical distance. A
stronger version would use a physical-distance kernel on a suitable projected
coordinate system before applying the same restriction. More importantly,
geographic distance alone is a weak scientific reason for flux correlation. A
similarity-space construction could add land cover, sector, climatology, or
other prior features; observation-derived features would need filtering or
held-out data to avoid leakage.

## Interpretation

Increasing kappa with depth narrows fine split fractions around their expected
prior-flux allocation. Leaves that share a recent ancestor therefore tend to
move together more strongly. This is tree-local dependence: two geographically
adjacent terminal regions separated by an old tree boundary need not have the
same covariance as siblings.

The cap of 128 makes deep split fractions very tight: at a diverging split the
normalized left/right cross-moment multiplier is `128 / 129`, about 0.992.
That does not make terminal variances uniformly small. Unequal expected-mass
fractions and repeated same-branch multipliers produce scaling variances from
{summary.minimum_leaf_variance:.3g} to {summary.maximum_leaf_variance:.3g} in
this layout. The covariance and correlation plots therefore give a more useful
picture than kappa alone.

Region counts use a different quantity from the Gamma--Beta conservation mass.
The allocation and weighted best-first refinement use mean absolute TAC/MHD
footprint-times-flux sensitivity, matching the standard constrained-basis
weight construction. Gamma--Beta split means continue to use absolute prior
flux times grid area. This prevents a geometrically complex but low-sensitivity
ocean mask from consuming most of the resolution budget.

The terminal-region count is a geometric count, not the current prior's number
of independent coordinates. Disconnected components within one semantic group
share its Gamma root. In particular, the three terminal ocean components share
one ocean root scaling because none receives an internal split at this weight
allocation. Giving those components independent totals would require separate
component roots or a group-level allocation split.

These are covariances of dimensionless scaling factors. Prior-flux covariance
would multiply matrix entry `(a, b)` by the expected flux masses of regions
`a` and `b`; tiny-mass regions with large scaling variance would then receive
less visual weight. The shared ocean root also makes its three unsplit supports
identical random variables, so the terminal scaling covariance is singular.
This is why the ocean map looks especially poor. Credible alternatives are to
give disconnected ocean components independent roots, place a separate
Dirichlet/Beta allocation layer above component roots, or replace the shared
root with an explicit spatial/similarity covariance. If the intended model is
one ocean coefficient, the three supports should instead be represented as one
state-vector element.

The variance of 226 is also not fixed by increasing kappa alone. It comes from
repeated multiplicative allocation along highly unequal branches. Possible
controls include requiring minimum Beta shape parameters rather than merely
capping kappa, stopping or rejecting extreme-mass splits, solving kappa from a
target terminal variance/correlation, or using a covariance model with fixed
marginal variances. Each changes the prior and should be assessed on region
flux totals as well as dimensionless scaling factors.

The maximum unweighted leaf-mean error is sensitive to tiny expected-mass
regions with heavy scaling-factor tails. The expected-mass-weighted diagnostic
is the relevant check for their effect on total prior flux.

The six outer geometries are fixed, but their scaling factors remain uncertain.
This prototype does not yet infer the active partition, use observations, or
construct a PyMC likelihood.
"""


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for the demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("tests/data"))
    parser.add_argument("--output-directory", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--inner-regions", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--base-kappa", type=float, default=2.0)
    parser.add_argument("--depth-multiplier", type=float, default=2.0)
    parser.add_argument("--max-kappa", type=float, default=128.0)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the demonstration and write report artifacts."""
    args = build_parser().parse_args(argv)
    case = build_case(
        data_directory=args.data_directory,
        draws=args.draws,
        inner_regions=args.inner_regions,
        max_depth=args.max_depth,
        base_kappa=args.base_kappa,
        depth_multiplier=args.depth_multiplier,
        max_kappa=args.max_kappa,
        seed=args.seed,
    )
    summary = summarize_case(case)
    write_report(case, summary, args.output_directory)
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
