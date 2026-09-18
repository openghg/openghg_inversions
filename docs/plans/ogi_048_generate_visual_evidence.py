"""Generate visual evidence for OGI-048 lat/lon split geometry.

This script uses repository test data to compare count-based split steps with
and without ``LatLonGridGeometry``. It writes PNG figures plus a markdown report
under ``docs/plans`` so the output can be attached to the PR and later refined
into documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes

from openghg_inversions.basis.algorithms import (
    AxisParallelSplitStep,
    GreedySplitStrategy,
    InertialSplitStep,
    LatLonGridGeometry,
    region_constrained_basis,
)
from openghg_inversions.basis.basis_functions import BasisFunctions


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "docs" / "plans" / "figures" / "ogi_048_visual_evidence"
REPORT_PATH = ROOT / "docs" / "plans" / "ogi_048_visual_evidence.md"
REGION_METRICS_PATH = ROOT / "docs" / "plans" / "ogi_048_region_metrics.csv"
SUMMARY_METRICS_PATH = ROOT / "docs" / "plans" / "ogi_048_summary_metrics.csv"
SINGULAR_VALUES_PATH = ROOT / "docs" / "plans" / "ogi_048_sensitivity_singular_values.csv"
SENSITIVITY_SUMMARY_PATH = ROOT / "docs" / "plans" / "ogi_048_sensitivity_summary.csv"

FLUX_PATH = ROOT / "tests" / "data" / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc"
FOOTPRINT_PATH = ROOT / "tests" / "data" / "footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc"
COUNTRY_PATH = ROOT / "tests" / "data" / "country_EUROPE.nc"
BUCKET_BASIS_PATH = ROOT / "tests" / "data" / "basis" / "EUROPE" / "bucket_ch4-test_basis_EUROPE_2019.nc"
QUADTREE_BASIS_PATH = ROOT / "tests" / "data" / "basis" / "EUROPE" / "quadtree_ch4-test_basis_EUROPE_2019.nc"

BASIS_COUNTS = (50, 100, 250, 500)


@dataclass(frozen=True)
class Scenario:
    """One generated basis comparison scenario."""

    split_step: str
    geometry: str
    classes: str
    nbasis: int
    actual_regions: int
    basis_functions: BasisFunctions


def main() -> None:
    """Generate all figures and the markdown report."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    weights, flux, country, fp_x_flux = load_inputs()
    cell_area = estimate_cell_area_km2(weights)
    landsea_classes = make_landsea_classes(country, weights)
    all_classes = xr.DataArray(
        np.full(weights.shape, "all", dtype=object),
        dims=weights.dims,
        coords=weights.coords,
        name="region_class",
    )
    geometry = LatLonGridGeometry.from_dataarray(weights)

    scenarios: list[Scenario] = []
    labels_by_key: dict[tuple[str, str, str, int], xr.DataArray] = {}

    for split_step_name in ("axis-parallel", "inertial"):
        for class_name, region_classes in (("all-cells", all_classes), ("ocean-land", landsea_classes)):
            for geometry_name, split_geometry in (("row-column", None), ("lat-lon-metres", geometry)):
                for nbasis in BASIS_COUNTS:
                    labels = build_basis_labels(
                        weights=weights,
                        region_classes=region_classes,
                        nbasis=nbasis,
                        split_step_name=split_step_name,
                        geometry=split_geometry,
                    )
                    if labels_cross_region_classes(labels, region_classes):
                        raise AssertionError(
                            f"{split_step_name}/{class_name}/{geometry_name}/{nbasis} crosses class boundaries."
                        )
                    basis_functions = BasisFunctions.from_flat_basis(
                        basis_flat=labels.rename("basis"),
                        flux=flux.rename("flux"),
                        region_labels="basis_values",
                        metadata={
                            "openghg_inversions:basis_artifact_source": "ogi-048-visual-evidence",
                            "openghg_inversions:split_step": split_step_name,
                            "openghg_inversions:geometry": geometry_name,
                            "openghg_inversions:region_classes": class_name,
                        },
                    )
                    labels_by_key[(split_step_name, class_name, geometry_name, nbasis)] = (
                        validated_flat_basis(basis_functions, labels)
                    )
                    scenarios.append(
                        Scenario(
                            split_step=split_step_name,
                            geometry=geometry_name,
                            classes=class_name,
                            nbasis=nbasis,
                            actual_regions=count_regions(labels),
                            basis_functions=basis_functions,
                        )
                    )

    input_figure = plot_inputs(weights, landsea_classes)
    matrix_figures = []
    for split_step_name in ("axis-parallel", "inertial"):
        for class_name in ("all-cells", "ocean-land"):
            matrix_figures.append(
                plot_matrix(
                    labels_by_key=labels_by_key,
                    split_step_name=split_step_name,
                    class_name=class_name,
                    landsea_classes=landsea_classes,
                )
            )

    region_metrics = build_region_metrics(
        labels_by_key=labels_by_key,
        weights=weights,
        cell_area=cell_area,
    )
    summary_metrics = summarize_region_metrics(region_metrics)
    region_metrics.to_csv(REGION_METRICS_PATH, index=False)
    summary_metrics.to_csv(SUMMARY_METRICS_PATH, index=False)

    quantitative_figures = []
    for split_step_name in ("axis-parallel", "inertial"):
        for class_name in ("all-cells", "ocean-land"):
            quantitative_figures.append(
                plot_quantitative_metrics(
                    region_metrics=region_metrics,
                    split_step_name=split_step_name,
                    class_name=class_name,
                )
            )

    singular_values, sensitivity_summary = build_sensitivity_singular_metrics(
        scenarios=scenarios,
        fp_x_flux=fp_x_flux,
    )
    singular_values.to_csv(SINGULAR_VALUES_PATH, index=False)
    sensitivity_summary.to_csv(SENSITIVITY_SUMMARY_PATH, index=False)

    sensitivity_figures = []
    for split_step_name in ("axis-parallel", "inertial"):
        for class_name in ("all-cells", "ocean-land"):
            sensitivity_figures.append(
                plot_sensitivity_singular_metrics(
                    singular_values=singular_values,
                    split_step_name=split_step_name,
                    class_name=class_name,
                )
            )

    write_report(
        scenarios=scenarios,
        input_figure=input_figure,
        matrix_figures=matrix_figures,
        quantitative_figures=quantitative_figures,
        summary_metrics=summary_metrics,
        sensitivity_figures=sensitivity_figures,
        sensitivity_summary=sensitivity_summary,
    )


def load_inputs() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Load and align the test-data weight, flux, and country fields."""
    flux = (
        xr.open_dataset(FLUX_PATH)
        .flux.squeeze("time", drop=True)
        .transpose("lat", "lon")
        .astype(np.float64)
        .load()
    )
    footprint = (
        xr.open_dataset(FOOTPRINT_PATH).fp.mean("time").transpose("lat", "lon").astype(np.float64).load()
    )
    flux = align_to_test_grid(reference=footprint, target=flux, target_name="flux")
    footprint_full = (
        xr.open_dataset(FOOTPRINT_PATH).fp.transpose("time", "lat", "lon").astype(np.float64).load()
    )
    footprint_full = align_time_grid_to_test_grid(
        reference=footprint,
        target=footprint_full,
        target_name="footprint_full",
    )
    fp_x_flux = (footprint_full * flux).fillna(0.0).rename("fp_x_flux")
    weights = (footprint * flux).fillna(0.0)
    max_weight = float(weights.max())
    if max_weight > 0.0:
        weights = weights / max_weight
    weights = weights.rename("weight")

    country = xr.open_dataset(COUNTRY_PATH).country.transpose("lat", "lon").load()
    country = align_to_test_grid(reference=weights, target=country, target_name="country")
    flux = flux.assign_coords({dim: weights.coords[dim] for dim in weights.dims})
    fp_x_flux = fp_x_flux.assign_coords({dim: weights.coords[dim] for dim in weights.dims})
    return weights, flux, country, fp_x_flux


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
    """Align a time-varying test-data field on the same latitude/longitude grid."""
    if "time" not in target.dims:
        raise ValueError(f"{target_name} must include a 'time' dimension.")
    grid_slice = target.isel(time=0, drop=True)
    aligned_slice = align_to_test_grid(reference=reference, target=grid_slice, target_name=target_name)
    return target.assign_coords({dim: aligned_slice.coords[dim] for dim in reference.dims})


def estimate_cell_area_km2(weights: xr.DataArray) -> xr.DataArray:
    """Estimate latitude/longitude cell areas in square kilometres."""
    if weights.dims != ("lat", "lon"):
        raise ValueError("Cell-area estimates require weights ordered as ('lat', 'lon').")

    earth_radius_m = 6_371_008.8
    lat_edges = np.clip(coordinate_edges(weights.lat.values), -90.0, 90.0)
    lon_edges = coordinate_edges(weights.lon.values)
    lat_term = np.abs(np.diff(np.sin(np.deg2rad(lat_edges))))
    lon_term = np.abs(np.diff(np.deg2rad(lon_edges)))
    area_km2 = (earth_radius_m**2 * lat_term[:, np.newaxis] * lon_term[np.newaxis, :]) / 1.0e6
    return xr.DataArray(
        area_km2,
        dims=weights.dims,
        coords=weights.coords,
        name="cell_area_km2",
    )


def coordinate_edges(values: np.ndarray) -> np.ndarray:
    """Return midpoint-derived coordinate edges for monotonic cell centres."""
    centres = np.asarray(values, dtype=np.float64)
    if centres.ndim != 1 or centres.size < 2:
        raise ValueError("Coordinate edge estimates require at least two one-dimensional centres.")

    deltas = np.diff(centres)
    if not ((deltas > 0.0).all() or (deltas < 0.0).all()):
        raise ValueError("Coordinate centres must be monotonic.")

    midpoints = (centres[:-1] + centres[1:]) / 2.0
    first = centres[0] - (midpoints[0] - centres[0])
    last = centres[-1] + (centres[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def make_landsea_classes(country: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    """Convert the test country grid into ocean/land region classes."""
    classes = xr.where(country > 0, "land", "ocean").astype(object)
    return classes.rename("region_class").assign_coords({dim: weights.coords[dim] for dim in weights.dims})


def validated_flat_basis(basis_functions: BasisFunctions, labels: xr.DataArray) -> xr.DataArray:
    """Return the flat basis after checking the BasisFunctions object is coherent."""
    flat_basis = basis_functions.flat_basis()
    if isinstance(flat_basis, dict):
        raise TypeError("Expected a single-source BasisFunctions object, got a multi-source flat basis.")

    state_dim = basis_functions.operator.meta.state_dim
    actual_regions = count_regions(labels)
    matrix_regions = basis_functions.operator.basis_matrix.sizes[state_dim]
    if matrix_regions != actual_regions:
        raise AssertionError(
            f"BasisFunctions state dimension has {matrix_regions} regions, expected {actual_regions}."
        )

    xr.testing.assert_equal(flat_basis.load(), labels)
    return cast(xr.DataArray, flat_basis)


def labels_cross_region_classes(labels: xr.DataArray, region_classes: xr.DataArray) -> bool:
    """Return true if any positive basis label spans multiple region classes."""
    label_values = np.asarray(labels.values).ravel()
    class_values = np.asarray(region_classes.values, dtype=object).ravel()
    _, class_codes = np.unique(class_values, return_inverse=True)
    order = np.lexsort((class_codes, label_values))
    sorted_labels = label_values[order]
    sorted_class_codes = class_codes[order]
    positive = sorted_labels > 0
    sorted_labels = sorted_labels[positive]
    sorted_class_codes = sorted_class_codes[positive]
    if sorted_labels.size == 0:
        return False

    label_change = np.r_[True, sorted_labels[1:] != sorted_labels[:-1]]
    class_change = np.r_[False, sorted_class_codes[1:] != sorted_class_codes[:-1]]
    return bool(np.any(class_change & ~label_change))


def build_basis_labels(
    *,
    weights: xr.DataArray,
    region_classes: xr.DataArray,
    nbasis: int,
    split_step_name: str,
    geometry: LatLonGridGeometry | None,
) -> xr.DataArray:
    """Build one count-based region-constrained basis label map."""
    if split_step_name == "axis-parallel":
        split_step = AxisParallelSplitStep(balanced=False, clean_splits=True, geometry=geometry)
    elif split_step_name == "inertial":
        split_step = InertialSplitStep(balanced=False, geometry=geometry)
    else:
        raise ValueError(f"Unknown split step {split_step_name!r}.")

    labels = region_constrained_basis(
        weights,
        region_classes,
        nbasis,
        allocation="area",
        split_strategy=GreedySplitStrategy(split_step=split_step),
    )
    return labels.astype(np.int32).rename("basis")


def count_regions(labels: xr.DataArray) -> int:
    """Return the number of positive basis regions."""
    values = np.asarray(labels.values)
    return int(np.count_nonzero(np.unique(values[values > 0])))


def build_region_metrics(
    *,
    labels_by_key: dict[tuple[str, str, str, int], xr.DataArray],
    weights: xr.DataArray,
    cell_area: xr.DataArray,
) -> pd.DataFrame:
    """Compute one record per generated basis region."""
    records: list[dict[str, float | int | str]] = []
    weight_values = np.asarray(weights.values, dtype=np.float64)
    area_values = np.asarray(cell_area.values, dtype=np.float64)

    for split_step_name, class_name, geometry_name, nbasis in sorted(labels_by_key):
        labels = labels_by_key[(split_step_name, class_name, geometry_name, nbasis)]
        label_values = np.asarray(labels.values, dtype=np.int64)
        positive = label_values > 0
        region_ids = np.unique(label_values[positive])
        total_weight = float(weight_values[positive].sum())
        total_area_km2 = float(area_values[positive].sum())
        equal_weight = total_weight / len(region_ids) if len(region_ids) else np.nan

        for region_id in region_ids:
            mask = label_values == region_id
            region_weight = float(weight_values[mask].sum())
            region_area_km2 = float(area_values[mask].sum())
            records.append(
                {
                    "split_step": split_step_name,
                    "classes": class_name,
                    "geometry": geometry_name,
                    "target_regions": nbasis,
                    "actual_regions": len(region_ids),
                    "region_id": int(region_id),
                    "weight": region_weight,
                    "weight_share": safe_divide(region_weight, total_weight),
                    "equal_weight_ratio": safe_divide(region_weight, equal_weight),
                    "area_km2": region_area_km2,
                    "area_share": safe_divide(region_area_km2, total_area_km2),
                    "weight_density": safe_divide(region_weight, region_area_km2),
                    "weight_per_million_km2": safe_divide(region_weight, region_area_km2 / 1.0e6),
                }
            )

    metrics = pd.DataFrame.from_records(records)
    medians = metrics.groupby(
        ["split_step", "classes", "geometry", "target_regions"],
        sort=False,
    )["weight_density"].transform("median")
    metrics["density_ratio_to_median"] = metrics["weight_density"] / medians
    return metrics


def summarize_region_metrics(region_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-region metric spread for each generated scenario."""
    records: list[dict[str, float | int | str]] = []
    group_columns = ["split_step", "classes", "geometry", "target_regions"]
    for keys, group_obj in region_metrics.groupby(group_columns, sort=True):
        split_step_name, class_name, geometry_name, nbasis = cast(tuple[str, str, str, int], keys)
        group = cast(pd.DataFrame, group_obj)
        equal_ratios = group["equal_weight_ratio"].to_numpy(dtype=np.float64)
        weights = group["weight"].to_numpy(dtype=np.float64)
        densities = group["weight_density"].to_numpy(dtype=np.float64)
        areas = group["area_km2"].to_numpy(dtype=np.float64)
        records.append(
            {
                "split_step": str(split_step_name),
                "classes": str(class_name),
                "geometry": str(geometry_name),
                "target_regions": int(nbasis),
                "actual_regions": int(group["actual_regions"].iloc[0]),
                "min_equal_weight_ratio": float(np.min(equal_ratios)),
                "p10_equal_weight_ratio": float(np.quantile(equal_ratios, 0.1)),
                "median_equal_weight_ratio": float(np.median(equal_ratios)),
                "p90_equal_weight_ratio": float(np.quantile(equal_ratios, 0.9)),
                "max_equal_weight_ratio": float(np.max(equal_ratios)),
                "weight_cv": coefficient_of_variation(weights),
                "weight_gini": gini_coefficient(weights),
                "area_cv": coefficient_of_variation(areas),
                "density_cv": coefficient_of_variation(densities),
                "zero_weight_regions": int(np.count_nonzero(weights <= 0.0)),
            }
        )

    return pd.DataFrame.from_records(records)


def safe_divide(numerator: float, denominator: float) -> float:
    """Return a finite quotient or NaN for invalid denominators."""
    if denominator <= 0.0 or not np.isfinite(denominator):
        return np.nan
    return numerator / denominator


def coefficient_of_variation(values: np.ndarray) -> float:
    """Return standard deviation divided by mean."""
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.nan
    mean = float(np.mean(finite_values))
    if mean == 0.0:
        return np.nan
    return float(np.std(finite_values) / mean)


def gini_coefficient(values: np.ndarray) -> float:
    """Return the Gini coefficient for non-negative values."""
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.nan
    if np.min(finite_values) < 0.0:
        raise ValueError("Gini coefficient requires non-negative values.")
    total = float(np.sum(finite_values))
    if total == 0.0:
        return np.nan

    sorted_values = np.sort(finite_values)
    nvalue = sorted_values.size
    weighted_sum = float(np.sum((np.arange(1, nvalue + 1) * sorted_values)))
    return (2.0 * weighted_sum / (nvalue * total)) - ((nvalue + 1.0) / nvalue)


def build_sensitivity_singular_metrics(
    *,
    scenarios: list[Scenario],
    fp_x_flux: xr.DataArray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute singular-value spectra for unprojected and basis-projected H."""
    spectrum_records: list[dict[str, float | int | str]] = []
    summary_records: list[dict[str, float | int | str]] = []

    unprojected = unprojected_sensitivity_matrix(fp_x_flux)
    unprojected_singular_values = singular_values_from_observation_matrix(unprojected)
    add_singular_spectrum_records(
        records=spectrum_records,
        matrix_kind="unprojected-grid",
        split_step="unprojected-grid",
        classes="all-cells",
        geometry="grid-cells",
        target_regions=unprojected.shape[1],
        state_count=unprojected.shape[1],
        observation_count=unprojected.shape[0],
        singular_values=unprojected_singular_values,
    )
    summary_records.append(
        sensitivity_summary_record(
            matrix_kind="unprojected-grid",
            split_step="unprojected-grid",
            classes="all-cells",
            geometry="grid-cells",
            target_regions=unprojected.shape[1],
            state_count=unprojected.shape[1],
            observation_count=unprojected.shape[0],
            singular_values=unprojected_singular_values,
        )
    )

    for scenario in scenarios:
        h_matrix = basis_projected_sensitivity_matrix(scenario.basis_functions, fp_x_flux)
        singular_values = singular_values_from_observation_matrix(h_matrix)
        add_singular_spectrum_records(
            records=spectrum_records,
            matrix_kind="basis-projected",
            split_step=scenario.split_step,
            classes=scenario.classes,
            geometry=scenario.geometry,
            target_regions=scenario.nbasis,
            state_count=h_matrix.shape[1],
            observation_count=h_matrix.shape[0],
            singular_values=singular_values,
        )
        summary_records.append(
            sensitivity_summary_record(
                matrix_kind="basis-projected",
                split_step=scenario.split_step,
                classes=scenario.classes,
                geometry=scenario.geometry,
                target_regions=scenario.nbasis,
                state_count=h_matrix.shape[1],
                observation_count=h_matrix.shape[0],
                singular_values=singular_values,
            )
        )

    return pd.DataFrame.from_records(spectrum_records), pd.DataFrame.from_records(summary_records)


def unprojected_sensitivity_matrix(fp_x_flux: xr.DataArray) -> np.ndarray:
    """Return observation-by-grid-cell unprojected sensitivity values."""
    fp_values = np.asarray(
        fp_x_flux.transpose("time", "lat", "lon").fillna(0.0).values,
        dtype=np.float64,
    )
    return fp_values.reshape(fp_values.shape[0], -1)


def basis_projected_sensitivity_matrix(
    basis_functions: BasisFunctions,
    fp_x_flux: xr.DataArray,
) -> np.ndarray:
    """Return observation-by-basis-state projected sensitivity values."""
    state_dim = basis_functions.operator.meta.state_dim
    h = basis_functions.sensitivity(fp_x_flux).transpose("time", state_dim)
    return np.asarray(h.values, dtype=np.float64)


def singular_values_from_observation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Compute singular values from a rows-by-columns sensitivity matrix."""
    finite_matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float64), copy=False)
    gram = finite_matrix @ finite_matrix.T
    eigenvalues = np.linalg.eigvalsh(gram)
    singular_values = np.sqrt(np.clip(eigenvalues, 0.0, None))
    return singular_values[::-1]


def add_singular_spectrum_records(
    *,
    records: list[dict[str, float | int | str]],
    matrix_kind: str,
    split_step: str,
    classes: str,
    geometry: str,
    target_regions: int,
    state_count: int,
    observation_count: int,
    singular_values: np.ndarray,
) -> None:
    """Append one row per singular value."""
    energy = singular_values**2
    total_energy = float(np.sum(energy))
    leading = float(singular_values[0]) if singular_values.size else np.nan
    cumulative_energy = np.cumsum(energy)

    for index, singular_value in enumerate(singular_values, start=1):
        energy_fraction = safe_divide(float(energy[index - 1]), total_energy)
        records.append(
            {
                "matrix": matrix_kind,
                "split_step": split_step,
                "classes": classes,
                "geometry": geometry,
                "target_regions": target_regions,
                "state_count": state_count,
                "observation_count": observation_count,
                "singular_index": index,
                "singular_value": float(singular_value),
                "singular_value_normalized": safe_divide(float(singular_value), leading),
                "energy_fraction": energy_fraction,
                "cumulative_energy_fraction": safe_divide(
                    float(cumulative_energy[index - 1]),
                    total_energy,
                ),
            }
        )


def sensitivity_summary_record(
    *,
    matrix_kind: str,
    split_step: str,
    classes: str,
    geometry: str,
    target_regions: int,
    state_count: int,
    observation_count: int,
    singular_values: np.ndarray,
) -> dict[str, float | int | str]:
    """Return one summary record for a singular-value spectrum."""
    energy = singular_values**2
    total_energy = float(np.sum(energy))
    leading = float(singular_values[0]) if singular_values.size else np.nan
    positive = singular_values[singular_values > leading * 1.0e-12] if np.isfinite(leading) else np.array([])
    condition_number = safe_divide(float(positive[0]), float(positive[-1])) if positive.size else np.nan
    energy_probability = energy / total_energy if total_energy > 0.0 else np.array([])

    return {
        "matrix": matrix_kind,
        "split_step": split_step,
        "classes": classes,
        "geometry": geometry,
        "target_regions": target_regions,
        "state_count": state_count,
        "observation_count": observation_count,
        "leading_singular_value": leading,
        "frobenius_norm": float(np.sqrt(total_energy)),
        "stable_rank": safe_divide(total_energy, leading**2),
        "effective_rank": entropy_effective_rank(energy_probability),
        "rank_90_energy": energy_rank(energy_probability, 0.90),
        "rank_95_energy": energy_rank(energy_probability, 0.95),
        "rank_99_energy": energy_rank(energy_probability, 0.99),
        "condition_number_thresholded": condition_number,
    }


def entropy_effective_rank(probabilities: np.ndarray) -> float:
    """Return entropy effective rank for singular-value energy probabilities."""
    positive = probabilities[probabilities > 0.0]
    if positive.size == 0:
        return np.nan
    return float(np.exp(-np.sum(positive * np.log(positive))))


def energy_rank(probabilities: np.ndarray, threshold: float) -> int:
    """Return rank needed to reach a cumulative energy threshold."""
    if probabilities.size == 0:
        return 0
    cumulative = np.cumsum(probabilities)
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def plot_inputs(weights: xr.DataArray, landsea_classes: xr.DataArray) -> str:
    """Plot source weights, ocean/land classes, and existing test basis files."""
    bucket = (
        xr.open_dataset(BUCKET_BASIS_PATH).basis.squeeze("time", drop=True).transpose("lat", "lon").load()
    )
    quadtree = (
        xr.open_dataset(QUADTREE_BASIS_PATH).basis.squeeze("time", drop=True).transpose("lat", "lon").load()
    )

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()

    log_weight = np.log10(weights.where(weights > 0.0))
    mesh = axes[0].pcolormesh(weights.lon, weights.lat, log_weight, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=axes[0], shrink=0.8, label="log10 normalized footprint x flux")
    axes[0].set_title("TAC test weight field")

    land_values = xr.where(landsea_classes == "land", 1, 0)
    mesh = axes[1].pcolormesh(
        weights.lon, weights.lat, land_values, shading="auto", cmap="BrBG", vmin=0, vmax=1
    )
    fig.colorbar(mesh, ax=axes[1], shrink=0.8, ticks=[0, 1], label="0=ocean, 1=land")
    axes[1].set_title("Country-derived ocean/land classes")

    plot_basis_panel(axes[2], bucket, title=f"Existing bucket test basis ({count_regions(bucket)} regions)")
    plot_basis_panel(
        axes[3], quadtree, title=f"Existing quadtree test basis ({count_regions(quadtree)} regions)"
    )

    for ax in axes:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    filename = "inputs_and_existing_test_basis.png"
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return f"figures/ogi_048_visual_evidence/{filename}"


def plot_matrix(
    *,
    labels_by_key: dict[tuple[str, str, str, int], xr.DataArray],
    split_step_name: str,
    class_name: str,
    landsea_classes: xr.DataArray,
) -> str:
    """Plot one split-step/class matrix across region counts and geometry modes."""
    fig, axes = plt.subplots(
        nrows=len(BASIS_COUNTS),
        ncols=2,
        figsize=(11, 13),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    geometry_names = ("row-column", "lat-lon-metres")

    for row, nbasis in enumerate(BASIS_COUNTS):
        for col, geometry_name in enumerate(geometry_names):
            labels = labels_by_key[(split_step_name, class_name, geometry_name, nbasis)]
            ax = axes[row, col]
            title = f"{geometry_name}, target {nbasis}, actual {count_regions(labels)}"
            plot_basis_panel(ax, labels, title=title)
            overlay_land_outline(ax, landsea_classes)
            if row == len(BASIS_COUNTS) - 1:
                ax.set_xlabel("lon")
            else:
                ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel("lat")
            else:
                ax.set_ylabel("")

    fig.suptitle(f"{split_step_name}; classes={class_name}; count-based splits", fontsize=14)
    filename = f"{split_step_name.replace('-', '_')}_{class_name.replace('-', '_')}.png"
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return f"figures/ogi_048_visual_evidence/{filename}"


def plot_quantitative_metrics(
    *,
    region_metrics: pd.DataFrame,
    split_step_name: str,
    class_name: str,
) -> str:
    """Plot sorted per-region weight and density metrics for one scenario family."""
    fig, axes = plt.subplots(
        nrows=len(BASIS_COUNTS),
        ncols=2,
        figsize=(12, 13),
        sharex="col",
        constrained_layout=True,
    )

    for row, nbasis in enumerate(BASIS_COUNTS):
        subset = cast(
            pd.DataFrame,
            region_metrics.loc[
                (
                    (region_metrics["split_step"] == split_step_name)
                    & (region_metrics["classes"] == class_name)
                    & (region_metrics["target_regions"] == nbasis)
                ),
                :,
            ],
        )
        plot_sorted_metric(
            axes[row, 0],
            subset,
            metric="equal_weight_ratio",
            title=f"target {nbasis}: total weight per region",
            ylabel="region weight / equal-weight target",
        )
        axes[row, 0].axhline(1.0, color="0.45", linewidth=0.8, linestyle=":")

        plot_sorted_metric(
            axes[row, 1],
            subset,
            metric="density_ratio_to_median",
            title=f"target {nbasis}: weight per km2",
            ylabel="region density / scenario median",
        )
        axes[row, 1].axhline(1.0, color="0.45", linewidth=0.8, linestyle=":")

        if row == len(BASIS_COUNTS) - 1:
            axes[row, 0].set_xlabel("Region percentile, sorted low to high")
            axes[row, 1].set_xlabel("Region percentile, sorted low to high")
        if row == 0:
            axes[row, 0].legend(loc="best", fontsize=8)

    fig.suptitle(f"{split_step_name}; classes={class_name}; quantitative distributions", fontsize=14)
    filename = f"{split_step_name.replace('-', '_')}_{class_name.replace('-', '_')}_quantitative.png"
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return f"figures/ogi_048_visual_evidence/{filename}"


def plot_sorted_metric(
    ax: Axes,
    metrics: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot one sorted per-region metric, overlaid by geometry mode."""
    for geometry_name in ("row-column", "lat-lon-metres"):
        values = (
            metrics.loc[metrics["geometry"] == geometry_name, metric]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=np.float64)
        )
        if values.size == 0:
            continue

        sorted_values = np.sort(values)
        positive = sorted_values[sorted_values > 0.0]
        floor = float(np.min(positive) / 10.0) if positive.size else 1.0e-12
        sorted_values = np.clip(sorted_values, floor, None)
        percentile = np.linspace(0.0, 100.0, sorted_values.size)
        ax.plot(percentile, sorted_values, label=geometry_name, linewidth=1.4)

    ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.4, alpha=0.35)


def plot_sensitivity_singular_metrics(
    *,
    singular_values: pd.DataFrame,
    split_step_name: str,
    class_name: str,
) -> str:
    """Plot normalized singular spectra for one scenario family."""
    fig, axes = plt.subplots(
        nrows=len(BASIS_COUNTS),
        ncols=2,
        figsize=(12, 13),
        sharex="col",
        constrained_layout=True,
    )
    baseline = cast(
        pd.DataFrame,
        singular_values.loc[singular_values["matrix"] == "unprojected-grid", :],
    )

    for row, nbasis in enumerate(BASIS_COUNTS):
        subset = cast(
            pd.DataFrame,
            singular_values.loc[
                (
                    (singular_values["matrix"] == "basis-projected")
                    & (singular_values["split_step"] == split_step_name)
                    & (singular_values["classes"] == class_name)
                    & (singular_values["target_regions"] == nbasis)
                ),
                :,
            ],
        )
        plot_singular_spectrum_panel(
            axes[row, 0],
            baseline=baseline,
            metrics=subset,
            metric="singular_value_normalized",
            title=f"target {nbasis}: normalized singular values",
            ylabel="singular value / leading singular value",
            log_scale=True,
        )
        plot_singular_spectrum_panel(
            axes[row, 1],
            baseline=baseline,
            metrics=subset,
            metric="cumulative_energy_fraction",
            title=f"target {nbasis}: cumulative singular energy",
            ylabel="cumulative energy fraction",
            log_scale=False,
        )
        axes[row, 1].set_ylim(0.0, 1.02)

        if row == len(BASIS_COUNTS) - 1:
            axes[row, 0].set_xlabel("Singular value index")
            axes[row, 1].set_xlabel("Singular value index")
        if row == 0:
            axes[row, 0].legend(loc="best", fontsize=8)

    fig.suptitle(f"{split_step_name}; classes={class_name}; sensitivity singular values", fontsize=14)
    filename = f"{split_step_name.replace('-', '_')}_{class_name.replace('-', '_')}_sensitivity_svd.png"
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return f"figures/ogi_048_visual_evidence/{filename}"


def plot_singular_spectrum_panel(
    ax: Axes,
    *,
    baseline: pd.DataFrame,
    metrics: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    log_scale: bool,
) -> None:
    """Plot one singular-spectrum metric, with unprojected baseline."""
    plot_singular_line(
        ax,
        baseline,
        metric=metric,
        label="unprojected grid",
        color="black",
        linestyle="--",
    )
    for geometry_name, color in (("row-column", "tab:blue"), ("lat-lon-metres", "tab:orange")):
        plot_singular_line(
            ax,
            cast(pd.DataFrame, metrics.loc[metrics["geometry"] == geometry_name, :]),
            metric=metric,
            label=geometry_name,
            color=color,
            linestyle="-",
        )

    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.4, alpha=0.35)


def plot_singular_line(
    ax: Axes,
    metrics: pd.DataFrame,
    *,
    metric: str,
    label: str,
    color: str,
    linestyle: str,
) -> None:
    """Plot one singular-value metric line."""
    if metrics.empty:
        return
    values = metrics.sort_values("singular_index")
    x = values["singular_index"].to_numpy(dtype=np.float64)
    y = values[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
    if y.size == 0:
        return
    if y.size != x.size:
        x = np.arange(1, y.size + 1, dtype=np.float64)
    positive = y[y > 0.0]
    if positive.size:
        y = np.clip(y, float(np.min(positive) / 10.0), None)
    ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=1.2)


def plot_basis_panel(ax: Axes, labels: xr.DataArray, *, title: str) -> None:
    """Plot one basis label map without a colorbar."""
    values = labels.values.astype(float)
    ax.pcolormesh(labels.lon, labels.lat, values, shading="auto", cmap="nipy_spectral", rasterized=True)
    ax.set_title(title, fontsize=10)


def overlay_land_outline(ax: Axes, landsea_classes: xr.DataArray) -> None:
    """Draw a thin outline around land cells for spatial orientation."""
    land_values = xr.where(landsea_classes == "land", 1.0, 0.0)
    ax.contour(
        landsea_classes.lon,
        landsea_classes.lat,
        land_values,
        levels=[0.5],
        colors="black",
        linewidths=0.25,
    )


def write_report(
    *,
    scenarios: list[Scenario],
    input_figure: str,
    matrix_figures: list[str],
    quantitative_figures: list[str],
    summary_metrics: pd.DataFrame,
    sensitivity_figures: list[str],
    sensitivity_summary: pd.DataFrame,
) -> None:
    """Write the markdown evidence report."""
    lines = [
        "# OGI-048 Visual Evidence: Lat/Lon Split Geometry",
        "",
        "Generated from repository test data on the EUROPE grid.",
        "",
        "Inputs:",
        "",
        "- Flux: `tests/data/flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc`.",
        "- Footprint: `tests/data/footprints_tac_europe_name_185m_2019-01-01_2019-01-07_data.nc`.",
        "- Region classes: `tests/data/country_EUROPE.nc`, using country code `0` as an `ocean`/non-country proxy and positive country codes as `land`.",
        "- Existing baseline plots: `bucket_ch4-test_basis_EUROPE_2019.nc` and `quadtree_ch4-test_basis_EUROPE_2019.nc`.",
        "",
        "Method:",
        "",
        "- Each generated map is wrapped in a `BasisFunctions.from_flat_basis(...)` object before plotting; the script checks the object state dimension and flat-basis round trip.",
        "- `row-column` uses the old count-based coordinate system with no `SplitGeometry`.",
        "- `lat-lon-metres` uses `LatLonGridGeometry.from_dataarray(weights)`.",
        "- `axis-parallel` uses `AxisParallelSplitStep(balanced=False, clean_splits=True, ...)`.",
        "- `inertial` uses `InertialSplitStep(balanced=False, ...)`.",
        '- Class allocation uses `allocation="area"` so the ocean/land split changes class boundaries but not contribution-weight allocation.',
        "- Flux, footprint, and country fixtures have near-identical grids; the script checks dimensions and coordinates before assigning a common test grid.",
        "- Generated labels are checked so no basis label crosses the configured `all`, `ocean`, or `land` class boundary.",
        "- Quantitative metrics use the same normalized footprint-times-flux weights as the splitter, plus spherical latitude/longitude cell-area estimates for density plots.",
        f"- Per-region metrics are written to `{REGION_METRICS_PATH.relative_to(ROOT)}` and scenario summaries to `{SUMMARY_METRICS_PATH.relative_to(ROOT)}`.",
        f"- Sensitivity singular values use time-resolved `fp_x_flux` from the TAC test footprint and are written to `{SINGULAR_VALUES_PATH.relative_to(ROOT)}` with summaries in `{SENSITIVITY_SUMMARY_PATH.relative_to(ROOT)}`.",
        "- These are algorithm-level visual checks only; PR #482 does not add config or wrapper routing for geometry.",
        "",
        f"![Inputs and existing test basis]({input_figure})",
        "",
        "## Generated Basis Matrices",
        "",
    ]

    for figure in matrix_figures:
        title = Path(figure).stem.replace("_", " ")
        lines.extend([f"### {title.title()}", "", f"![{title}]({figure})", ""])

    lines.extend(
        [
            "## Quantitative Region Metrics",
            "",
            "The quantitative plots sort individual generated basis regions from low to high. "
            "The left column shows total region weight divided by the equal-weight target for that scenario; "
            "higher lower-tail values and lower spread mean fewer very-low-weight regions. "
            "The right column shows region weight per estimated square kilometre, normalized by that scenario's median density.",
            "",
            "Zero-valued metrics, if present, are clipped to the plotting floor on the log-scale plots.",
            "",
            "Readout: in this count-based setup, lat/lon geometry is mainly a geometric correction. "
            "It does not systematically improve per-region weight distributions. "
            "Axis-parallel all-cell summaries are nearly unchanged, while ocean/land and inertial cases are mixed. "
            "That suggests weight-balance improvements should come from balanced splitting, allocation, or split-stopping policy choices rather than geometry alone.",
            "",
        ]
    )

    for figure in quantitative_figures:
        title = Path(figure).stem.replace("_", " ")
        lines.extend([f"### {title.title()}", "", f"![{title}]({figure})", ""])

    lines.extend(
        [
            "## Paired Quantitative Summary",
            "",
            "For `p10 weight/equal`, higher is better for avoiding low-weight basis regions. "
            "For `weight CV`, `weight Gini`, and `density CV`, lower means less spread. "
            "These count-based runs are not expected to optimize region weights directly.",
            "",
            "| split step | classes | target | p10 weight/equal row | p10 weight/equal metres | weight CV row | weight CV metres | weight Gini row | weight Gini metres | density CV row | density CV metres |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(paired_summary_rows(summary_metrics))
    lines.append("")

    lines.extend(
        [
            "## Sensitivity Matrix Singular Values",
            "",
            "The unprojected grid sensitivity matrix is built as observations by grid cells from time-resolved `fp_x_flux`. "
            "Projected matrices are computed with `BasisFunctions.sensitivity(fp_x_flux)` for each generated basis. "
            "Plots compare singular values normalized by the leading singular value and cumulative singular-energy fraction, so they focus on spectrum shape rather than raw scale.",
            "",
            baseline_sensitivity_summary(sensitivity_summary),
            "",
            "Readout: increasing the target region count has a clearer effect on the projected H spectrum than switching from row/column to lat/lon-metre geometry. "
            "Geometry changes some low-target ocean/land and inertial spectra, but the paired stable-rank, effective-rank, and rank99 metrics do not show a consistent improvement from geometry alone.",
            "",
        ]
    )

    for figure in sensitivity_figures:
        title = Path(figure).stem.replace("_", " ")
        lines.extend([f"### {title.title()}", "", f"![{title}]({figure})", ""])

    lines.extend(
        [
            "### Paired Sensitivity Summary",
            "",
            "Stable rank and effective rank summarize spectrum spread; `rank99` is the number of singular modes needed for 99% of singular energy. "
            "Higher rank metrics mean the projected H keeps more independent observation-space modes, though this does not by itself imply better posterior behavior.",
            "",
            "| split step | classes | target | stable rank row | stable rank metres | effective rank row | effective rank metres | rank99 row | rank99 metres |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(paired_sensitivity_rows(sensitivity_summary))
    lines.append("")

    lines.extend(
        [
            "## Scenario Summary",
            "",
            "| split step | classes | geometry | target regions | actual regions |",
            "|---|---|---|---:|---:|",
        ]
    )
    for scenario in scenarios:
        lines.append(
            f"| {scenario.split_step} | {scenario.classes} | {scenario.geometry} | "
            f"{scenario.nbasis} | {scenario.actual_regions} |"
        )

    lines.extend(
        [
            "",
            "## Notes For PR Review",
            "",
            "- The clearest visual difference is expected at high latitudes, where one degree of longitude is physically shorter than one degree at lower latitudes.",
            "- The ocean/land split should prevent a generated label from crossing the ocean/land class boundary.",
            "- Inertial splits use physical coordinates for projection only when `LatLonGridGeometry` is supplied.",
            "- In this count-based setup, geometry changes the shape and physical interpretation of splits but does not by itself make the greedy priority weight-balanced.",
            "- Sensitivity singular values compare observation-space rank retention, not spatial smoothness or posterior uncertainty directly.",
            "- Color values are label IDs, so colors are useful for shape inspection but should not be interpreted as stable region identity across panels.",
            "",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def paired_summary_rows(summary_metrics: pd.DataFrame) -> list[str]:
    """Return markdown rows comparing row/column and physical-geometry summaries."""
    rows: list[str] = []
    for split_step_name in ("axis-parallel", "inertial"):
        for class_name in ("all-cells", "ocean-land"):
            for nbasis in BASIS_COUNTS:
                row = summary_record(summary_metrics, split_step_name, class_name, "row-column", nbasis)
                metres = summary_record(
                    summary_metrics, split_step_name, class_name, "lat-lon-metres", nbasis
                )
                rows.append(
                    "| "
                    f"{split_step_name} | {class_name} | {nbasis} | "
                    f"{format_metric(row['p10_equal_weight_ratio'])} | "
                    f"{format_metric(metres['p10_equal_weight_ratio'])} | "
                    f"{format_metric(row['weight_cv'])} | "
                    f"{format_metric(metres['weight_cv'])} | "
                    f"{format_metric(row['weight_gini'])} | "
                    f"{format_metric(metres['weight_gini'])} | "
                    f"{format_metric(row['density_cv'])} | "
                    f"{format_metric(metres['density_cv'])} |"
                )
    return rows


def baseline_sensitivity_summary(sensitivity_summary: pd.DataFrame) -> str:
    """Return a short markdown sentence for the unprojected grid H spectrum."""
    baseline = sensitivity_summary.loc[sensitivity_summary["matrix"] == "unprojected-grid", :]
    if len(baseline) != 1:
        raise ValueError("Expected exactly one unprojected-grid sensitivity summary row.")
    row = baseline.iloc[0]
    return (
        "Unprojected grid baseline: "
        f"{int(row['observation_count'])} observations by {int(row['state_count'])} grid-cell states; "
        f"stable rank {format_metric(row['stable_rank'])}, "
        f"effective rank {format_metric(row['effective_rank'])}, "
        f"rank99 {int(row['rank_99_energy'])}."
    )


def paired_sensitivity_rows(sensitivity_summary: pd.DataFrame) -> list[str]:
    """Return markdown rows comparing row/column and physical-geometry H spectra."""
    rows: list[str] = []
    for split_step_name in ("axis-parallel", "inertial"):
        for class_name in ("all-cells", "ocean-land"):
            for nbasis in BASIS_COUNTS:
                row = sensitivity_summary_record_for(
                    sensitivity_summary,
                    split_step_name,
                    class_name,
                    "row-column",
                    nbasis,
                )
                metres = sensitivity_summary_record_for(
                    sensitivity_summary,
                    split_step_name,
                    class_name,
                    "lat-lon-metres",
                    nbasis,
                )
                rows.append(
                    "| "
                    f"{split_step_name} | {class_name} | {nbasis} | "
                    f"{format_metric(row['stable_rank'])} | "
                    f"{format_metric(metres['stable_rank'])} | "
                    f"{format_metric(row['effective_rank'])} | "
                    f"{format_metric(metres['effective_rank'])} | "
                    f"{int(row['rank_99_energy'])} | "
                    f"{int(metres['rank_99_energy'])} |"
                )
    return rows


def sensitivity_summary_record_for(
    sensitivity_summary: pd.DataFrame,
    split_step_name: str,
    class_name: str,
    geometry_name: str,
    nbasis: int,
) -> pd.Series:
    """Select one projected sensitivity summary row."""
    row = sensitivity_summary[
        (sensitivity_summary["matrix"] == "basis-projected")
        & (sensitivity_summary["split_step"] == split_step_name)
        & (sensitivity_summary["classes"] == class_name)
        & (sensitivity_summary["geometry"] == geometry_name)
        & (sensitivity_summary["target_regions"] == nbasis)
    ]
    if len(row) != 1:
        raise ValueError(
            f"Expected one sensitivity summary row for "
            f"{split_step_name}/{class_name}/{geometry_name}/{nbasis}."
        )
    return row.iloc[0]


def summary_record(
    summary_metrics: pd.DataFrame,
    split_step_name: str,
    class_name: str,
    geometry_name: str,
    nbasis: int,
) -> pd.Series:
    """Select one scenario summary row."""
    row = summary_metrics[
        (summary_metrics["split_step"] == split_step_name)
        & (summary_metrics["classes"] == class_name)
        & (summary_metrics["geometry"] == geometry_name)
        & (summary_metrics["target_regions"] == nbasis)
    ]
    if len(row) != 1:
        raise ValueError(
            f"Expected one summary row for {split_step_name}/{class_name}/{geometry_name}/{nbasis}."
        )
    return row.iloc[0]


def format_metric(value: object) -> str:
    """Format a numeric metric for compact markdown tables."""
    if isinstance(value, (int, float, np.floating)):
        metric = float(value)
    else:
        metric = float(str(value))
    if not np.isfinite(metric):
        return "nan"
    if abs(metric) >= 100.0 or (abs(metric) < 0.001 and metric != 0.0):
        return f"{metric:.2e}"
    return f"{metric:.3f}"


if __name__ == "__main__":
    main()
