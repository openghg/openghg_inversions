"""Generate documentation plots for OGI-048 basis algorithm options.

The generated page explains the modular constrained-basis options and scores
250-region variants on the repository TAC/EUROPE test fixture. The headline
score is a forward-model perturbation diagnostic: compare the observation-space
response of deterministic fine-grid flux-scale perturbations with the response
after each perturbation is projected to one weighted mean value per basis
region.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure

from openghg_inversions.basis.algorithms import (
    AllocationMode,
    AxisParallelSplitStep,
    GreedyAxisParallelSplitStrategy,
    InertialSplitStep,
    LatLonGridGeometry,
    region_constrained_basis,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "docs" / "plans" / "figures" / "ogi_048_basis_options"
REPORT_PATH = ROOT / "docs" / "plans" / "ogi_048_basis_algorithm_options.md"
SCORES_PATH = ROOT / "docs" / "plans" / "ogi_048_basis_option_scores.csv"

FLUX_PATH = ROOT / "tests" / "data" / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc"
FOOTPRINT_PATH = ROOT / "tests" / "data" / "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc"
COUNTRY_PATH = ROOT / "tests" / "data" / "country_EUROPE.nc"

TARGET_REGIONS = 250
SMOOTH_PERTURBATIONS = (
    "lat_gradient",
    "lon_gradient",
    "western_europe_blob",
    "nordic_blob",
)
BOUNDARY_ALIGNED_PERTURBATIONS = (
    "land_ocean_contrast",
    "selected_country_patch",
)
SELECTED_COUNTRIES = (
    "UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND",
    "FRANCE",
    "GERMANY",
    "SPAIN",
    "ITALY",
    "POLAND",
    "UKRAINE",
    "SWEDEN",
    "NORWAY",
    "FINLAND",
    "TURKEY",
    "ROMANIA",
    "RUSSIAN FEDERATION",
)


@dataclass(frozen=True)
class Scenario:
    """One basis option combination."""

    class_mode: str
    allocation: str
    split_step: str
    split_mode: str
    geometry: str
    labels: xr.DataArray
    projected_flux: xr.DataArray
    smooth_perturbation_mean_nrmse: float
    smooth_perturbation_max_nrmse: float
    boundary_perturbation_mean_nrmse: float
    all_perturbation_mean_nrmse: float
    perturbation_scores: dict[str, float]
    prior_flux_obs_nrmse: float
    prior_flux_obs_rmse: float
    prior_flux_obs_bias: float
    prior_flux_obs_corr: float
    flux_field_nrmse: float
    actual_regions: int


def main() -> None:
    """Generate the documentation page, plots, and score CSV."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    flux, footprint, weights, country = load_inputs()
    geometry = LatLonGridGeometry.from_dataarray(weights)
    class_modes = build_region_class_modes(country, weights)
    perturbations = build_perturbations(weights, country)
    y_full = modelled_observations(footprint, flux)

    scenarios = build_scenarios(
        flux=flux,
        footprint=footprint,
        weights=weights,
        class_modes=class_modes,
        geometry=geometry,
        perturbations=perturbations,
        y_full=y_full,
    )
    scores = scenario_scores(scenarios)
    scores.to_csv(SCORES_PATH, index=False)

    class_figure = plot_region_classes(class_modes)
    basis_figure = plot_basis_contrasts(scenarios)
    score_heatmap = plot_score_heatmap(scores)
    score_ranked = plot_ranked_scores(scores)

    write_report(
        class_figure=class_figure,
        basis_figure=basis_figure,
        score_heatmap=score_heatmap,
        score_ranked=score_ranked,
        scores=scores,
    )


def load_inputs() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Load flux, footprint, weight, and country test fields."""
    flux = (
        xr.open_dataset(FLUX_PATH)
        .flux.squeeze("time", drop=True)
        .transpose("lat", "lon")
        .astype(np.float64)
        .load()
        .rename("flux")
    )
    footprint = (
        xr.open_dataset(FOOTPRINT_PATH)
        .fp.transpose("time", "lat", "lon")
        .astype(np.float64)
        .load()
        .rename("footprint")
    )
    footprint_mean = footprint.mean("time").transpose("lat", "lon")
    flux = align_to_test_grid(reference=footprint_mean, target=flux, target_name="flux")
    footprint = align_time_grid_to_test_grid(
        reference=footprint_mean,
        target=footprint,
        target_name="footprint",
    )

    weights = (footprint.mean("time") * flux).fillna(0.0).rename("weight")
    max_weight = float(weights.max())
    if max_weight > 0.0:
        weights = weights / max_weight

    country = xr.open_dataset(COUNTRY_PATH).country.transpose("lat", "lon").load()
    country = align_to_test_grid(reference=weights, target=country, target_name="country")
    return flux, footprint, weights, country


def align_to_test_grid(
    *,
    reference: xr.DataArray,
    target: xr.DataArray,
    target_name: str,
    atol: float = 1e-4,
) -> xr.DataArray:
    """Align a test-data field after checking it uses the same near-identical grid."""
    if target.dims != reference.dims:
        raise ValueError(
            f"{target_name} dims {target.dims!r} do not match reference dims {reference.dims!r}."
        )

    for dim in reference.dims:
        if target.sizes[dim] != reference.sizes[dim]:
            raise ValueError(
                f"{target_name} dimension {dim!r} has size {target.sizes[dim]}, "
                f"expected {reference.sizes[dim]}."
            )
        if dim in target.coords and dim in reference.coords:
            target_coord = np.asarray(target.coords[dim].values, dtype=float)
            reference_coord = np.asarray(reference.coords[dim].values, dtype=float)
            if not np.allclose(target_coord, reference_coord, rtol=0.0, atol=atol):
                max_delta = float(np.max(np.abs(target_coord - reference_coord)))
                raise ValueError(
                    f"{target_name} coordinate {dim!r} differs from the reference grid "
                    f"by up to {max_delta:g}, exceeding tolerance {atol:g}."
                )

    return target.assign_coords({dim: reference.coords[dim] for dim in reference.dims})


def align_time_grid_to_test_grid(
    *,
    reference: xr.DataArray,
    target: xr.DataArray,
    target_name: str,
) -> xr.DataArray:
    """Align a time-varying field on the same latitude/longitude grid."""
    if "time" not in target.dims:
        raise ValueError(f"{target_name} must include a 'time' dimension.")
    aligned_slice = align_to_test_grid(
        reference=reference,
        target=target.isel(time=0, drop=True),
        target_name=target_name,
    )
    return target.assign_coords({dim: aligned_slice.coords[dim] for dim in reference.dims})


def build_region_class_modes(country: xr.DataArray, weights: xr.DataArray) -> dict[str, xr.DataArray]:
    """Build the three region-class fields used in the documentation."""
    country_ds = xr.open_dataset(COUNTRY_PATH).load()
    country_names = [str(value) for value in country_ds.name.values]
    selected_codes = {index for index, name in enumerate(country_names) if name in set(SELECTED_COUNTRIES)}
    selected_short_names = {index: short_country_label(country_names[index]) for index in selected_codes}
    country_values = country.astype(int)

    all_classes = xr.DataArray(
        np.full(weights.shape, "all", dtype=object),
        dims=weights.dims,
        coords=weights.coords,
        name="region_class",
    )
    landsea = xr.where(country_values > 0, "land", "ocean").astype(object)
    landsea = landsea.rename("region_class").assign_coords(weights.coords)

    selected = xr.full_like(country_values, "other_land", dtype=object)
    selected = selected.where(country_values > 0, "ocean")
    for code, label in selected_short_names.items():
        selected = selected.where(country_values != code, label)
    selected = selected.rename("region_class").assign_coords(weights.coords)

    return {
        "no_mask": all_classes,
        "land_sea": landsea,
        "selected_countries": selected,
    }


def short_country_label(name: str) -> str:
    """Return compact labels for selected countries."""
    replacements = {
        "UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND": "UK",
        "RUSSIAN FEDERATION": "Russia",
    }
    if name in replacements:
        return replacements[name]
    return name.title()


def build_perturbations(weights: xr.DataArray, country: xr.DataArray) -> dict[str, xr.DataArray]:
    """Build deterministic fine-grid flux-scale perturbation fields."""
    lat = weights.lat
    lon = weights.lon
    lat_grid, lon_grid = xr.broadcast(lat, lon)
    country_values = country.astype(int)

    selected_country_ids = country_ids_for_names(SELECTED_COUNTRIES)
    selected_country_mask = xr.zeros_like(country_values, dtype=bool)
    for country_id in selected_country_ids:
        selected_country_mask = selected_country_mask | (country_values == country_id)

    perturbations = {
        "lat_gradient": normalize_perturbation(lat_grid - lat_grid.mean()),
        "lon_gradient": normalize_perturbation(lon_grid - lon_grid.mean()),
        "western_europe_blob": normalize_perturbation(
            gaussian_blob(lat_grid, lon_grid, center_lat=50.0, center_lon=2.0, sigma_lat=8.0, sigma_lon=10.0)
        ),
        "nordic_blob": normalize_perturbation(
            gaussian_blob(lat_grid, lon_grid, center_lat=62.0, center_lon=15.0, sigma_lat=7.0, sigma_lon=11.0)
        ),
        "land_ocean_contrast": normalize_perturbation(xr.where(country_values > 0, 1.0, -1.0)),
        "selected_country_patch": normalize_perturbation(xr.where(selected_country_mask, 1.0, 0.0)),
    }
    return {name: field.rename(name) for name, field in perturbations.items()}


def country_ids_for_names(country_names: tuple[str, ...]) -> set[int]:
    """Return country fixture numeric IDs for requested names."""
    country_ds = xr.open_dataset(COUNTRY_PATH).load()
    all_names = [str(value) for value in country_ds.name.values]
    requested = set(country_names)
    return {index for index, name in enumerate(all_names) if name in requested}


def gaussian_blob(
    lat_grid: xr.DataArray,
    lon_grid: xr.DataArray,
    *,
    center_lat: float,
    center_lon: float,
    sigma_lat: float,
    sigma_lon: float,
) -> xr.DataArray:
    """Return a smooth Gaussian perturbation on the lat/lon grid."""
    lat_term = ((lat_grid - center_lat) / sigma_lat) ** 2
    lon_term = ((lon_grid - center_lon) / sigma_lon) ** 2
    return xr.apply_ufunc(np.exp, -0.5 * (lat_term + lon_term))


def normalize_perturbation(field: xr.DataArray) -> xr.DataArray:
    """Center and scale a perturbation to unit RMS."""
    centered = field - field.mean()
    rms = float(np.sqrt(np.mean(np.asarray(centered.values, dtype=np.float64) ** 2)))
    if rms == 0.0:
        raise ValueError("Cannot normalize a constant perturbation.")
    return centered / rms


def build_scenarios(
    *,
    flux: xr.DataArray,
    footprint: xr.DataArray,
    weights: xr.DataArray,
    class_modes: dict[str, xr.DataArray],
    geometry: LatLonGridGeometry,
    perturbations: dict[str, xr.DataArray],
    y_full: xr.DataArray,
) -> list[Scenario]:
    """Build and score each feasible option combination."""
    scenarios: list[Scenario] = []
    for class_mode, region_classes in class_modes.items():
        allocations = ("single_class",) if class_mode == "no_mask" else ("area", "weight")
        for allocation_name in allocations:
            allocation = "weight" if allocation_name == "single_class" else allocation_name
            for split_step_name in ("axis_parallel", "inertial"):
                for split_mode in ("count", "balanced"):
                    for geometry_name, split_geometry in (("row_column", None), ("lat_lon_metres", geometry)):
                        labels = build_basis_labels(
                            weights=weights,
                            region_classes=region_classes,
                            allocation=allocation,
                            split_step_name=split_step_name,
                            split_mode=split_mode,
                            geometry=split_geometry,
                        )
                        projected_flux = project_flux_to_regions(flux, labels)
                        y_projected = modelled_observations(footprint, projected_flux)
                        perturbation_scores = score_perturbation_reconstruction(
                            footprint=footprint,
                            flux=flux,
                            labels=labels,
                            weights=weights,
                            perturbations=perturbations,
                        )
                        scenarios.append(
                            Scenario(
                                class_mode=class_mode,
                                allocation=allocation_name,
                                split_step=split_step_name,
                                split_mode=split_mode,
                                geometry=geometry_name,
                                labels=labels,
                                projected_flux=projected_flux,
                                smooth_perturbation_mean_nrmse=mean_scores(
                                    perturbation_scores, SMOOTH_PERTURBATIONS
                                ),
                                smooth_perturbation_max_nrmse=max_scores(
                                    perturbation_scores, SMOOTH_PERTURBATIONS
                                ),
                                boundary_perturbation_mean_nrmse=mean_scores(
                                    perturbation_scores, BOUNDARY_ALIGNED_PERTURBATIONS
                                ),
                                all_perturbation_mean_nrmse=float(
                                    np.mean(list(perturbation_scores.values()))
                                ),
                                perturbation_scores=perturbation_scores,
                                prior_flux_obs_nrmse=normalized_rmse(y_projected, y_full),
                                prior_flux_obs_rmse=rmse(y_projected, y_full),
                                prior_flux_obs_bias=float((y_projected - y_full).mean()),
                                prior_flux_obs_corr=correlation(y_projected, y_full),
                                flux_field_nrmse=normalized_rmse(projected_flux, flux),
                                actual_regions=count_regions(labels),
                            )
                        )
    return scenarios


def build_basis_labels(
    *,
    weights: xr.DataArray,
    region_classes: xr.DataArray,
    allocation: str,
    split_step_name: str,
    split_mode: str,
    geometry: LatLonGridGeometry | None,
) -> xr.DataArray:
    """Build one 250-target basis label map."""
    balanced = split_mode == "balanced"
    if split_step_name == "axis_parallel":
        split_step = AxisParallelSplitStep(balanced=balanced, clean_splits=True, geometry=geometry)
    elif split_step_name == "inertial":
        split_step = InertialSplitStep(balanced=balanced, geometry=geometry)
    else:
        raise ValueError(f"Unknown split step {split_step_name!r}.")

    labels = region_constrained_basis(
        weights,
        region_classes,
        TARGET_REGIONS,
        allocation=cast(AllocationMode, allocation),
        split_strategy=GreedyAxisParallelSplitStrategy(split_step=split_step),
    )
    return labels.astype(np.int32).rename("basis")


def mean_scores(scores: dict[str, float], names: tuple[str, ...]) -> float:
    """Return the mean score for a named subset of perturbations."""
    return float(np.mean([scores[name] for name in names]))


def max_scores(scores: dict[str, float], names: tuple[str, ...]) -> float:
    """Return the maximum score for a named subset of perturbations."""
    return float(np.max([scores[name] for name in names]))


def score_perturbation_reconstruction(
    *,
    footprint: xr.DataArray,
    flux: xr.DataArray,
    labels: xr.DataArray,
    weights: xr.DataArray,
    perturbations: dict[str, xr.DataArray],
) -> dict[str, float]:
    """Score observation-space reconstruction for fine-grid perturbations."""
    scores: dict[str, float] = {}
    fp_x_flux = footprint * flux
    for name, perturbation in perturbations.items():
        projected = project_field_to_regions(perturbation, labels, mean_weights=weights)
        y_full = (fp_x_flux * perturbation).sum(("lat", "lon")).rename("perturbation_observation")
        y_projected = (fp_x_flux * projected).sum(("lat", "lon")).rename("perturbation_observation")
        scores[name] = normalized_rmse(y_projected, y_full)
    return scores


def project_flux_to_regions(flux: xr.DataArray, labels: xr.DataArray) -> xr.DataArray:
    """Approximate a flux field by one cell-mean value per basis region."""
    return project_field_to_regions(flux, labels, mean_weights=None).rename("projected_flux")


def project_field_to_regions(
    field: xr.DataArray,
    labels: xr.DataArray,
    *,
    mean_weights: xr.DataArray | None,
) -> xr.DataArray:
    """Approximate a field by one mean value per positive basis region."""
    field_values = np.asarray(field.values, dtype=np.float64)
    label_values = np.asarray(labels.values, dtype=np.int64)
    max_label = int(label_values.max())
    if mean_weights is None:
        flat_weights = np.ones(field_values.size, dtype=np.float64)
    else:
        flat_weights = np.asarray(mean_weights.values, dtype=np.float64).ravel()
        if float(np.sum(flat_weights[label_values.ravel() > 0])) == 0.0:
            flat_weights = np.ones(field_values.size, dtype=np.float64)
    sums = np.bincount(
        label_values.ravel(),
        weights=field_values.ravel() * flat_weights,
        minlength=max_label + 1,
    )
    weight_sums = np.bincount(label_values.ravel(), weights=flat_weights, minlength=max_label + 1)
    means = np.divide(sums, weight_sums, out=np.zeros_like(sums), where=weight_sums > 0)
    projected = means[label_values]
    projected = np.where(label_values > 0, projected, 0.0)
    return xr.DataArray(projected, dims=field.dims, coords=field.coords, name=f"projected_{field.name}")


def modelled_observations(footprint: xr.DataArray, flux: xr.DataArray) -> xr.DataArray:
    """Return modelled observations from a flux field and footprints."""
    return (footprint * flux).sum(("lat", "lon")).rename("modelled_observation")


def rmse(candidate: xr.DataArray, reference: xr.DataArray) -> float:
    """Return root mean squared error."""
    difference = np.asarray((candidate - reference).values, dtype=np.float64)
    return float(np.sqrt(np.mean(difference**2)))


def normalized_rmse(candidate: xr.DataArray, reference: xr.DataArray) -> float:
    """Return RMSE normalized by reference RMS."""
    reference_values = np.asarray(reference.values, dtype=np.float64)
    denominator = float(np.sqrt(np.mean(reference_values**2)))
    if denominator == 0.0:
        return np.nan
    return rmse(candidate, reference) / denominator


def correlation(candidate: xr.DataArray, reference: xr.DataArray) -> float:
    """Return Pearson correlation for flattened values."""
    candidate_values = np.asarray(candidate.values, dtype=np.float64).ravel()
    reference_values = np.asarray(reference.values, dtype=np.float64).ravel()
    if np.std(candidate_values) == 0.0 or np.std(reference_values) == 0.0:
        return np.nan
    return float(np.corrcoef(candidate_values, reference_values)[0, 1])


def count_regions(labels: xr.DataArray) -> int:
    """Return the number of positive basis regions."""
    values = np.asarray(labels.values)
    return int(np.count_nonzero(np.unique(values[values > 0])))


def scenario_scores(scenarios: list[Scenario]) -> pd.DataFrame:
    """Return a compact score table for all scenarios."""
    records = []
    for scenario in scenarios:
        record = {
            "class_mode": scenario.class_mode,
            "allocation": scenario.allocation,
            "split_step": scenario.split_step,
            "split_mode": scenario.split_mode,
            "geometry": scenario.geometry,
            "target_regions": TARGET_REGIONS,
            "actual_regions": scenario.actual_regions,
            "smooth_perturbation_mean_nrmse": scenario.smooth_perturbation_mean_nrmse,
            "smooth_perturbation_max_nrmse": scenario.smooth_perturbation_max_nrmse,
            "boundary_perturbation_mean_nrmse": scenario.boundary_perturbation_mean_nrmse,
            "all_perturbation_mean_nrmse": scenario.all_perturbation_mean_nrmse,
            "prior_flux_obs_nrmse": scenario.prior_flux_obs_nrmse,
            "prior_flux_obs_rmse": scenario.prior_flux_obs_rmse,
            "prior_flux_obs_bias": scenario.prior_flux_obs_bias,
            "prior_flux_obs_corr": scenario.prior_flux_obs_corr,
            "flux_field_nrmse": scenario.flux_field_nrmse,
        }
        for perturbation_name, score in scenario.perturbation_scores.items():
            record[f"perturbation_{perturbation_name}_nrmse"] = score
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values("smooth_perturbation_mean_nrmse")


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return row dictionaries with a Pyright-friendly type."""
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def plot_region_classes(class_modes: dict[str, xr.DataArray]) -> str:
    """Plot the region-class masks used for constrained runs."""
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4.3), constrained_layout=True)
    for ax, (name, classes) in zip(axes, class_modes.items(), strict=True):
        values, labels = categorical_codes(classes)
        base_colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, 20))
        cmap = ListedColormap(base_colors[: len(labels)])
        norm = BoundaryNorm(np.arange(len(labels) + 1) - 0.5, cmap.N)
        mesh = ax.pcolormesh(classes.lon, classes.lat, values, shading="auto", cmap=cmap, norm=norm)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.75, ticks=np.arange(len(labels)))
        cbar.ax.set_yticklabels([label.replace("_", " ") for label in labels])
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    return save_figure(fig, "region_class_modes.png")


def plot_basis_contrasts(scenarios: list[Scenario]) -> str:
    """Plot a small set of basis maps contrasting the main independent options."""
    contrast_specs = [
        ("land/sea axis count row", "land_sea", "area", "axis_parallel", "count", "row_column"),
        ("lat/lon geometry", "land_sea", "area", "axis_parallel", "count", "lat_lon_metres"),
        ("balanced split", "land_sea", "area", "axis_parallel", "balanced", "row_column"),
        ("inertial split", "land_sea", "area", "inertial", "count", "row_column"),
        ("no mask", "no_mask", "single_class", "axis_parallel", "count", "row_column"),
        ("selected countries", "selected_countries", "area", "axis_parallel", "count", "row_column"),
        ("weight allocation", "selected_countries", "weight", "axis_parallel", "count", "row_column"),
        ("inertial + metre geometry", "selected_countries", "area", "inertial", "count", "lat_lon_metres"),
    ]
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 13), sharex=True, sharey=True)
    for ax, spec in zip(axes.ravel(), contrast_specs, strict=True):
        title, class_mode, allocation, split_step, split_mode, geometry = spec
        scenario = find_scenario(scenarios, class_mode, allocation, split_step, split_mode, geometry)
        ax.pcolormesh(
            scenario.labels.lon,
            scenario.labels.lat,
            scenario.labels.values.astype(float),
            shading="auto",
            cmap="nipy_spectral",
            rasterized=True,
        )
        ax.set_title(
            f"{title}\nsmooth NRMSE={scenario.smooth_perturbation_mean_nrmse:.3f}, "
            f"regions={scenario.actual_regions}"
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    fig.tight_layout()
    return save_figure(fig, "basis_option_contrasts_250.png")


def plot_score_heatmap(scores: pd.DataFrame) -> str:
    """Plot all option scores as small heatmaps."""
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12), constrained_layout=True)
    row_order = [
        ("axis_parallel", "count"),
        ("axis_parallel", "balanced"),
        ("inertial", "count"),
        ("inertial", "balanced"),
    ]
    geometry_order = ["row_column", "lat_lon_metres"]
    for row, class_mode in enumerate(("no_mask", "land_sea", "selected_countries")):
        allocations = ["single_class"] if class_mode == "no_mask" else ["area", "weight"]
        for col, allocation in enumerate(allocations):
            ax = axes[row, col]
            subset: pd.DataFrame = scores.loc[
                (scores["class_mode"] == class_mode) & (scores["allocation"] == allocation),
                :,
            ]
            matrix = np.full((len(row_order), len(geometry_order)), np.nan)
            for i, (split_step, split_mode) in enumerate(row_order):
                for j, geometry in enumerate(geometry_order):
                    match: pd.DataFrame = subset.loc[
                        (subset["split_step"] == split_step)
                        & (subset["split_mode"] == split_mode)
                        & (subset["geometry"] == geometry),
                        :,
                    ]
                    if len(match) == 1:
                        matrix[i, j] = float(dataframe_records(match)[0]["smooth_perturbation_mean_nrmse"])
            image = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.isfinite(matrix[i, j]):
                        ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white")
            ax.set_xticks(
                np.arange(len(geometry_order)), labels=[label.replace("_", "\n") for label in geometry_order]
            )
            ax.set_yticks(np.arange(len(row_order)), labels=[f"{a}\n{b}" for a, b in row_order])
            ax.set_title(f"{class_mode.replace('_', ' ')}; allocation={allocation}")
            fig.colorbar(image, ax=ax, shrink=0.75, label="mean smooth perturbation NRMSE")
        if len(allocations) == 1:
            axes[row, 1].axis("off")
    return save_figure(fig, "basis_option_score_heatmaps_250.png")


def plot_ranked_scores(scores: pd.DataFrame) -> str:
    """Plot all option combinations sorted by observation NRMSE."""
    ordered = scores.reset_index(drop=True).copy()
    labels = [
        f"{row['class_mode']}/{row['allocation']}/{row['split_step']}/{row['split_mode']}/{row['geometry']}"
        for row in dataframe_records(ordered)
    ]
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    colors = {"no_mask": "tab:blue", "land_sea": "tab:orange", "selected_countries": "tab:green"}
    y = np.arange(len(ordered))
    ax.scatter(
        ordered["smooth_perturbation_mean_nrmse"],
        y,
        c=[colors[value] for value in ordered["class_mode"]],
        s=34,
    )
    ax.set_yticks(y, labels=labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean observation NRMSE from weighted smooth-perturbation projection")
    ax.set_title("250-region option combinations sorted by smooth-perturbation reconstruction score")
    for class_mode, color in colors.items():
        ax.scatter([], [], color=color, label=class_mode.replace("_", " "))
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.4)
    return save_figure(fig, "basis_option_ranked_scores_250.png")


def categorical_codes(classes: xr.DataArray) -> tuple[np.ndarray, list[str]]:
    """Convert string classes to integer codes for plotting."""
    flat = np.asarray(classes.values, dtype=object).ravel()
    labels = sorted({str(value) for value in flat})
    mapping = {label: index for index, label in enumerate(labels)}
    values = np.array([mapping[str(value)] for value in flat], dtype=float).reshape(classes.shape)
    return values, labels


def find_scenario(
    scenarios: list[Scenario],
    class_mode: str,
    allocation: str,
    split_step: str,
    split_mode: str,
    geometry: str,
) -> Scenario:
    """Return one scenario by its option keys."""
    for scenario in scenarios:
        if (
            scenario.class_mode == class_mode
            and scenario.allocation == allocation
            and scenario.split_step == split_step
            and scenario.split_mode == split_mode
            and scenario.geometry == geometry
        ):
            return scenario
    raise ValueError(f"No scenario found for {(class_mode, allocation, split_step, split_mode, geometry)!r}.")


def save_figure(fig: Figure, filename: str) -> str:
    """Save a figure and return the markdown-relative path."""
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return f"figures/ogi_048_basis_options/{filename}"


def write_report(
    *,
    class_figure: str,
    basis_figure: str,
    score_heatmap: str,
    score_ranked: str,
    scores: pd.DataFrame,
) -> None:
    """Write the documentation page."""
    best_rows = top_score_rows(scores, n=8)
    selected_country_text = ", ".join(short_country_label(name) for name in SELECTED_COUNTRIES)
    lines = [
        "# Constrained Basis Algorithm Options",
        "",
        "This page explains the lower-level constrained basis algorithms and compares 250-target-region variants on the repository TAC/EUROPE test data.",
        "",
        "## How The Algorithm Is Built",
        "",
        "The constrained basis path is a small orchestration framework rather than one fixed algorithm. The caller supplies a two-dimensional importance field, `weights`, and a two-dimensional `region_classes` mask. The algorithm partitions each mapped class independently, then offsets labels so basis labels are globally unique and never cross class boundaries.",
        "",
        "The independently variable pieces are:",
        "",
        "- **Region classes**: no mask, land/ocean, countries, grouped countries, or any caller-supplied class field.",
        "- **Class allocation**: explicit per-class counts, automatic allocation by class total weight, or automatic allocation by cell count.",
        "- **Greedy orchestration**: repeatedly split the currently highest-weight partition until the target is reached or no acceptable split remains.",
        "- **Partition step**: axis-parallel row/column splits or inertial principal-axis splits.",
        "- **Split mode**: count-based splits or balanced splits near half parent weight.",
        "- **Geometry**: row/column index geometry or local lat/lon metre geometry for split-shape decisions.",
        "- **Split stopping**: optional policies that reject proposed child regions. When stopping is enabled, the requested region count is an upper target.",
        "",
        "The important separation is that `weights` define contribution/importance, while `geometry` defines physical coordinates for split shape. Lat/lon geometry does not change contribution weights, class allocation, or posterior weighting.",
        "",
        "For the no-mask score rows below, allocation is reported as `single_class` because there is only one class. The generator uses the normal weight-allocation API internally, but no inter-class allocation decision is being tested in that case.",
        "",
        "## Region Class Modes",
        "",
        "The plots below use three class modes. The selected-country mode treats ocean as one class, keeps selected large European-domain countries as separate classes, and groups all remaining land as `other_land`. The selected countries are: "
        f"{selected_country_text}.",
        "",
        f"![Region class modes]({class_figure})",
        "",
        "## Forward-Model Compression Score",
        "",
        "A normal multiplicative prior basis exactly reproduces the prior modelled observations when all basis coefficients are one: summing projected `H` over all regions gives the same `sum(fp * flux)` as the full grid. That RMSE is therefore a trivial zero and is not useful for comparing basis shapes.",
        "",
        "Instead, this page uses deterministic perturbation-reconstruction diagnostics. Fine-grid flux-scale perturbation fields are applied to `fp * flux`, then each perturbation is projected to one contribution-weighted mean value per basis region. This is an optimistic representability diagnostic: it uses the known perturbation field and the same TAC fixture, not coefficients estimated from noisy held-out observations.",
        "",
        "The headline score and ranking use only smooth perturbations: latitude and longitude gradients plus western-Europe and Nordic Gaussian blobs. Boundary-aligned perturbations, a land/ocean contrast and a selected-country patch, are reported separately because they can tautologically reward basis masks that hard-code the same boundaries.",
        "",
        "A secondary score also projects the prior flux field itself to one cell-mean value per region and compares modelled observations from full and projected flux. That is a prior observation-space compression score; low observation NRMSE can still coexist with poor spatial flux-field reconstruction.",
        "",
        "Neither score is a posterior-quality metric, and neither replaces posterior or synthetic-recovery tests.",
        "",
        "## Basis Map Contrasts",
        "",
        f"![Basis option contrasts]({basis_figure})",
        "",
        "## Scores For 250-Region Options",
        "",
        f"All scored combinations are written to `{SCORES_PATH.relative_to(ROOT)}`.",
        "",
        "The heatmap color scales are local to each panel so within-panel differences remain visible. Use the printed values and ranked score plot for cross-panel comparisons.",
        "",
        f"![Score heatmaps]({score_heatmap})",
        "",
        f"![Ranked scores]({score_ranked})",
        "",
        "### Best Smooth-Perturbation Scores",
        "",
        "| rank | class mode | allocation | split step | split mode | geometry | actual regions | smooth perturb NRMSE | max smooth NRMSE | boundary perturb NRMSE | prior obs NRMSE | flux-field NRMSE |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        *best_rows,
        "",
        "## Interpretation",
        "",
        "- Lower smooth-perturbation NRMSE means the basis preserves the TAC observation response of the deterministic smooth fine-grid perturbations more efficiently.",
        "- Lower boundary-perturbation NRMSE means the basis preserves perturbations that match land/ocean or selected-country boundaries. It is useful context, but it is not used for the headline ranking.",
        "- Lower prior observation NRMSE means the basis preserves the prior forward model better under a region-mean flux-field approximation, not that it preserves the full spatial flux field.",
        "- Balanced splits often help when the score is dominated by high-contribution areas, but they are not guaranteed to produce visually regular regions.",
        "- Region classes impose hard boundaries, which can help interpretability but can also spend regions on low-contribution classes.",
        "- Lat/lon metre geometry is a physical-coordinate correction. It can change region shapes, especially for inertial or high-latitude splits, but it is not itself a weight-balancing rule.",
        "- Split-stopping policies are not included in the score matrix because they can return fewer than 250 actual regions, making direct comparison less clean.",
        "",
        "## What This Does Not Prove",
        "",
        "These scores do not show whether an inversion posterior improves. For that, use a posterior or posterior-equivalent test: prior/error-weighted `H`, held-out observations, observation-error weighting, linear-Gaussian posterior covariance and resolution, synthetic truth recovery, or paired HPC-CI posterior runs.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def top_score_rows(scores: pd.DataFrame, *, n: int) -> list[str]:
    """Format the best score rows for markdown."""
    rows: list[str] = []
    for rank, row in enumerate(dataframe_records(scores.head(n)), start=1):
        rows.append(
            f"| {rank} | {row['class_mode']} | {row['allocation']} | {row['split_step']} | "
            f"{row['split_mode']} | {row['geometry']} | {row['actual_regions']} | "
            f"{row['smooth_perturbation_mean_nrmse']:.4f} | "
            f"{row['smooth_perturbation_max_nrmse']:.4f} | "
            f"{row['boundary_perturbation_mean_nrmse']:.4f} | "
            f"{row['prior_flux_obs_nrmse']:.4f} | {row['flux_field_nrmse']:.4f} |"
        )
    return rows


if __name__ == "__main__":
    main()
