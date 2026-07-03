"""Generate documentation plots for OGI-048 basis algorithm options.

The generated page explains the modular constrained-basis options and scores
250-target-region variants with Blue Pebble OpenGHG data. Scoring uses
month-specific temporal cross-validation: for TAC and MHD, January and July
2019 are split into held-out one-week windows with a two-day buffer on each
side. Basis weights are built from the remaining observations in the same
month, and scores are computed on the held-out week.
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
from openghg.retrieve import get_flux, get_footprint

from openghg_inversions.basis.algorithms import (
    AllocationMode,
    AxisParallelSplitStep,
    GreedyAxisParallelSplitStrategy,
    InertialSplitStep,
    LatLonGridGeometry,
    quadtree_algorithm,
    region_constrained_basis,
    weighted_algorithm,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "docs" / "plans" / "figures" / "ogi_048_basis_options"
REPORT_PATH = ROOT / "docs" / "plans" / "ogi_048_basis_algorithm_options.md"
SCORES_PATH = ROOT / "docs" / "plans" / "ogi_048_basis_option_scores.csv"
SPLIT_SCORES_PATH = ROOT / "docs" / "plans" / "ogi_048_basis_option_split_scores.csv"

COUNTRY_PATH = ROOT / "tests" / "data" / "country_EUROPE.nc"
LPDM_COUNTRY_PATH = Path("/group/chem/acrg/LPDM/countries/country_EUROPE.nc")

BLUE_PEBBLE_STORE = "shared_store_zarr"
BLUE_PEBBLE_DOMAIN = "EUROPE"
BLUE_PEBBLE_FLUX_SOURCE = "edgarv80_wetchartsv131"

TARGET_REGIONS = 250
QUADTREE_SEED = 42
HOLDOUT_DAYS = 7
BUFFER_DAYS = 2
HOLDOUT_START_DAYS = (6, 20)
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
class SiteSpec:
    """One footprint site used in the Blue Pebble CV score."""

    site: str
    inlet: str


@dataclass(frozen=True)
class MonthSpec:
    """One month scored separately in the Blue Pebble CV score."""

    label: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class SplitSpec:
    """One temporal cross-validation split."""

    split_id: str
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    buffer_start: pd.Timestamp
    buffer_end: pd.Timestamp


@dataclass(frozen=True)
class MonthInputs:
    """Loaded flux, footprint, and country fields for one site/month."""

    flux: xr.DataArray
    footprint: xr.DataArray
    country: xr.DataArray
    country_names: tuple[str, ...]


@dataclass(frozen=True)
class CandidateLabels:
    """One generated basis-label field before scoring."""

    basis_family: str
    class_mode: str
    allocation: str
    split_step: str
    split_mode: str
    geometry: str
    labels: xr.DataArray
    actual_regions: int


@dataclass(frozen=True)
class Scenario:
    """One scored basis option for one CV split."""

    basis_family: str
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


SITES = (
    SiteSpec(site="TAC", inlet="185m"),
    SiteSpec(site="MHD", inlet="10m"),
)
MONTHS = (
    MonthSpec(label="January", start=pd.Timestamp("2019-01-01"), end=pd.Timestamp("2019-02-01")),
    MonthSpec(label="July", start=pd.Timestamp("2019-07-01"), end=pd.Timestamp("2019-08-01")),
)


def main() -> None:
    """Generate the documentation page, plots, and score CSVs."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    split_scores, representative_scenarios, representative_class_modes = build_cross_validation_scores()
    split_scores.to_csv(SPLIT_SCORES_PATH, index=False)
    scores = aggregate_split_scores(split_scores)
    scores.to_csv(SCORES_PATH, index=False)

    class_figure = plot_region_classes(representative_class_modes)
    basis_figure = plot_basis_contrasts(representative_scenarios)
    score_heatmap = plot_score_heatmap(scores)
    score_ranked = plot_ranked_scores(scores)

    write_report(
        class_figure=class_figure,
        basis_figure=basis_figure,
        score_heatmap=score_heatmap,
        score_ranked=score_ranked,
        scores=scores,
        split_scores=split_scores,
    )


def build_cross_validation_scores() -> tuple[pd.DataFrame, list[Scenario], dict[str, xr.DataArray]]:
    """Build split-level scores for all sites, months, and basis candidates."""
    records: list[dict[str, Any]] = []
    representative_scenarios: list[Scenario] = []
    representative_class_modes: dict[str, xr.DataArray] | None = None

    for site in SITES:
        for month in MONTHS:
            print(f"Loading {site.site} {site.inlet} {month.label}", flush=True)
            inputs = load_blue_pebble_month(site=site, month=month)
            perturbations = build_perturbations(inputs.flux, inputs.country, inputs.country_names)
            splits = build_month_splits(month)

            for split in splits:
                print(f"Scoring {site.site} {month.label} {split.split_id}", flush=True)
                train_footprint = select_training_footprint(inputs.footprint, month=month, split=split)
                holdout_footprint = inputs.footprint.sel(time=slice(split.holdout_start, split.holdout_end))
                holdout_footprint = holdout_footprint.where(holdout_footprint.time < split.holdout_end, drop=True)

                if train_footprint.sizes["time"] == 0:
                    raise ValueError(f"No training observations for {site.site} {month.label} {split.split_id}.")
                if holdout_footprint.sizes["time"] == 0:
                    raise ValueError(f"No holdout observations for {site.site} {month.label} {split.split_id}.")

                weights = build_weights(train_footprint, inputs.flux)
                geometry = LatLonGridGeometry.from_dataarray(weights)
                class_modes = build_region_class_modes(inputs.country, weights, inputs.country_names)
                candidate_labels = build_candidate_labels(
                    weights=weights,
                    class_modes=class_modes,
                    geometry=geometry,
                )

                scenarios = score_candidates(
                    candidates=candidate_labels,
                    flux=inputs.flux,
                    holdout_footprint=holdout_footprint,
                    train_weights=weights,
                    perturbations=perturbations,
                )
                if not representative_scenarios:
                    representative_scenarios = scenarios
                    representative_class_modes = class_modes

                records.extend(
                    scenario_record(
                        scenario,
                        site=site,
                        month=month,
                        split=split,
                        train_observations=train_footprint.sizes["time"],
                        holdout_observations=holdout_footprint.sizes["time"],
                    )
                    for scenario in scenarios
                )

    if representative_class_modes is None:
        raise RuntimeError("No representative class modes were generated.")

    return pd.DataFrame.from_records(records), representative_scenarios, representative_class_modes


def load_blue_pebble_month(*, site: SiteSpec, month: MonthSpec) -> MonthInputs:
    """Load one site/month from the Blue Pebble shared OpenGHG zarr store."""
    footprint_dataset = get_footprint(
        domain=BLUE_PEBBLE_DOMAIN,
        site=site.site,
        inlet=site.inlet,
        model="NAME",
        species="inert",
        start_date=iso_date(month.start),
        end_date=iso_date(month.end),
        store=BLUE_PEBBLE_STORE,
    ).data
    footprint = (
        footprint_dataset.fp.transpose("time", "lat", "lon")
        .astype(np.float64)
        .load()
        .rename("footprint")
    )

    flux_dataset = get_flux(
        species="ch4",
        domain=BLUE_PEBBLE_DOMAIN,
        source=BLUE_PEBBLE_FLUX_SOURCE,
        start_date=iso_date(month.start),
        end_date=iso_date(month.end),
        store=BLUE_PEBBLE_STORE,
    ).data
    flux = (
        flux_dataset.flux.squeeze("time", drop=True)
        .transpose("lat", "lon")
        .astype(np.float64)
        .load()
        .rename("flux")
    )

    footprint_mean = footprint.mean("time").transpose("lat", "lon")
    flux = align_to_test_grid(reference=footprint_mean, target=flux, target_name="flux")
    footprint = align_time_grid_to_test_grid(
        reference=footprint_mean,
        target=footprint,
        target_name="footprint",
    )
    country, country_names = load_country(reference=footprint_mean)
    return MonthInputs(flux=flux, footprint=footprint, country=country, country_names=country_names)


def iso_date(timestamp: pd.Timestamp) -> str:
    """Return a YYYY-MM-DD date string for OpenGHG retrieval calls."""
    return timestamp.strftime("%Y-%m-%d")


def country_path() -> Path:
    """Return the country mask path, falling back to the Blue Pebble LPDM copy."""
    if COUNTRY_PATH.exists():
        return COUNTRY_PATH
    if LPDM_COUNTRY_PATH.exists():
        return LPDM_COUNTRY_PATH
    raise FileNotFoundError(f"Could not find {COUNTRY_PATH} or {LPDM_COUNTRY_PATH}.")


def load_country(reference: xr.DataArray) -> tuple[xr.DataArray, tuple[str, ...]]:
    """Load the EUROPE country mask and align it to the scoring grid."""
    path = country_path()
    country_ds = xr.open_dataset(path).load()
    country = country_ds.country.transpose("lat", "lon")
    country = align_to_test_grid(reference=reference, target=country, target_name="country")
    country_names = tuple(str(value) for value in country_ds.name.values)
    return country, country_names


def align_to_test_grid(
    *,
    reference: xr.DataArray,
    target: xr.DataArray,
    target_name: str,
    atol: float = 1e-4,
) -> xr.DataArray:
    """Align a field after checking it uses the same near-identical grid."""
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


def build_month_splits(month: MonthSpec) -> tuple[SplitSpec, ...]:
    """Return one-week holdout splits for a month."""
    splits = []
    for day in HOLDOUT_START_DAYS:
        holdout_start = pd.Timestamp(year=month.start.year, month=month.start.month, day=day)
        holdout_end = holdout_start + pd.Timedelta(days=HOLDOUT_DAYS)
        if holdout_end > month.end:
            raise ValueError(f"Holdout starting on day {day} extends beyond {month.label}.")
        splits.append(
            SplitSpec(
                split_id=f"{month.label.lower()}_{day:02d}_{day + HOLDOUT_DAYS - 1:02d}",
                holdout_start=holdout_start,
                holdout_end=holdout_end,
                buffer_start=holdout_start - pd.Timedelta(days=BUFFER_DAYS),
                buffer_end=holdout_end + pd.Timedelta(days=BUFFER_DAYS),
            )
        )
    return tuple(splits)


def select_training_footprint(
    footprint: xr.DataArray,
    *,
    month: MonthSpec,
    split: SplitSpec,
) -> xr.DataArray:
    """Select in-month training observations outside the holdout plus buffer window."""
    in_month = (footprint.time >= month.start) & (footprint.time < month.end)
    outside_buffer = (footprint.time < split.buffer_start) | (footprint.time >= split.buffer_end)
    return footprint.where(in_month & outside_buffer, drop=True)


def build_weights(footprint: xr.DataArray, flux: xr.DataArray) -> xr.DataArray:
    """Build normalized basis-construction weights from training footprints."""
    weights = (footprint.mean("time") * flux).fillna(0.0).rename("weight")
    max_weight = float(weights.max())
    if max_weight > 0.0:
        weights = weights / max_weight
    return weights


def build_region_class_modes(
    country: xr.DataArray,
    weights: xr.DataArray,
    country_names: tuple[str, ...],
) -> dict[str, xr.DataArray]:
    """Build the three region-class fields used in the documentation."""
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


def build_perturbations(
    reference: xr.DataArray,
    country: xr.DataArray,
    country_names: tuple[str, ...],
) -> dict[str, xr.DataArray]:
    """Build deterministic fine-grid flux-scale perturbation fields."""
    lat = reference.lat
    lon = reference.lon
    lat_grid, lon_grid = xr.broadcast(lat, lon)
    country_values = country.astype(int)

    selected_country_ids = country_ids_for_names(SELECTED_COUNTRIES, country_names)
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


def country_ids_for_names(country_names: tuple[str, ...], all_country_names: tuple[str, ...]) -> set[int]:
    """Return country fixture numeric IDs for requested names."""
    requested = set(country_names)
    return {index for index, name in enumerate(all_country_names) if name in requested}


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


def build_candidate_labels(
    *,
    weights: xr.DataArray,
    class_modes: dict[str, xr.DataArray],
    geometry: LatLonGridGeometry,
) -> list[CandidateLabels]:
    """Build all feasible constrained and legacy comparison basis candidates."""
    candidates: list[CandidateLabels] = []
    for class_mode, region_classes in class_modes.items():
        allocations = ("single_class",) if class_mode == "no_mask" else ("area", "weight")
        for allocation_name in allocations:
            allocation = "weight" if allocation_name == "single_class" else allocation_name
            for split_step_name in ("axis_parallel", "inertial"):
                for split_mode in ("count", "balanced"):
                    for geometry_name, split_geometry in (("row_column", None), ("lat_lon_metres", geometry)):
                        labels = build_constrained_basis_labels(
                            weights=weights,
                            region_classes=region_classes,
                            allocation=allocation,
                            split_step_name=split_step_name,
                            split_mode=split_mode,
                            geometry=split_geometry,
                        )
                        candidates.append(
                            CandidateLabels(
                                basis_family="region_constrained",
                                class_mode=class_mode,
                                allocation=allocation_name,
                                split_step=split_step_name,
                                split_mode=split_mode,
                                geometry=geometry_name,
                                labels=labels,
                                actual_regions=count_regions(labels),
                            )
                        )

    candidates.extend(build_legacy_candidate_labels(weights))
    return candidates


def build_constrained_basis_labels(
    *,
    weights: xr.DataArray,
    region_classes: xr.DataArray,
    allocation: str,
    split_step_name: str,
    split_mode: str,
    geometry: LatLonGridGeometry | None,
) -> xr.DataArray:
    """Build one 250-target constrained basis label map."""
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


def build_legacy_candidate_labels(weights: xr.DataArray) -> list[CandidateLabels]:
    """Build legacy bucketbasisfunction and quadtreebasisfunction comparison labels."""
    candidates: list[CandidateLabels] = []
    normalized_grid = np.asarray(weights.fillna(0.0).values, dtype=np.float64)
    grid_max = float(np.max(normalized_grid))
    if grid_max <= 0.0:
        return candidates
    normalized_grid = normalized_grid / grid_max

    legacy_specs = (
        ("bucketbasisfunction", lambda: weighted_algorithm(normalized_grid, nregion=TARGET_REGIONS)),
        ("quadtreebasisfunction", lambda: quadtree_algorithm(normalized_grid, nbasis=TARGET_REGIONS, seed=QUADTREE_SEED)),
    )
    for basis_family, build_labels in legacy_specs:
        try:
            raw_labels = build_labels()
        except Exception as exc:  # pragma: no cover - used by evidence-generation script only
            print(f"Skipping {basis_family}: {exc}")
            continue
        labels = xr.DataArray(
            np.asarray(raw_labels, dtype=np.int32),
            dims=weights.dims,
            coords=weights.coords,
            name="basis",
        )
        candidates.append(
            CandidateLabels(
                basis_family=basis_family,
                class_mode="legacy",
                allocation="legacy",
                split_step=basis_family,
                split_mode="legacy",
                geometry="row_column",
                labels=labels,
                actual_regions=count_regions(labels),
            )
        )
    return candidates


def score_candidates(
    *,
    candidates: list[CandidateLabels],
    flux: xr.DataArray,
    holdout_footprint: xr.DataArray,
    train_weights: xr.DataArray,
    perturbations: dict[str, xr.DataArray],
) -> list[Scenario]:
    """Score each candidate on one held-out footprint window."""
    scenarios: list[Scenario] = []
    y_full = modelled_observations(holdout_footprint, flux)
    for candidate in candidates:
        projected_flux = project_flux_to_regions(flux, candidate.labels)
        y_projected = modelled_observations(holdout_footprint, projected_flux)
        perturbation_scores = score_perturbation_reconstruction(
            footprint=holdout_footprint,
            flux=flux,
            labels=candidate.labels,
            weights=train_weights,
            perturbations=perturbations,
        )
        scenarios.append(
            Scenario(
                basis_family=candidate.basis_family,
                class_mode=candidate.class_mode,
                allocation=candidate.allocation,
                split_step=candidate.split_step,
                split_mode=candidate.split_mode,
                geometry=candidate.geometry,
                labels=candidate.labels,
                projected_flux=projected_flux,
                smooth_perturbation_mean_nrmse=mean_scores(perturbation_scores, SMOOTH_PERTURBATIONS),
                smooth_perturbation_max_nrmse=max_scores(perturbation_scores, SMOOTH_PERTURBATIONS),
                boundary_perturbation_mean_nrmse=mean_scores(
                    perturbation_scores, BOUNDARY_ALIGNED_PERTURBATIONS
                ),
                all_perturbation_mean_nrmse=float(np.mean(list(perturbation_scores.values()))),
                perturbation_scores=perturbation_scores,
                prior_flux_obs_nrmse=normalized_rmse(y_projected, y_full),
                prior_flux_obs_rmse=rmse(y_projected, y_full),
                prior_flux_obs_bias=float((y_projected - y_full).mean()),
                prior_flux_obs_corr=correlation(y_projected, y_full),
                flux_field_nrmse=normalized_rmse(projected_flux, flux),
                actual_regions=candidate.actual_regions,
            )
        )
    return scenarios


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
    """Score held-out observation-space reconstruction for fine-grid perturbations."""
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


def scenario_record(
    scenario: Scenario,
    *,
    site: SiteSpec,
    month: MonthSpec,
    split: SplitSpec,
    train_observations: int,
    holdout_observations: int,
) -> dict[str, Any]:
    """Return one split-level score record."""
    record: dict[str, Any] = {
        "site": site.site,
        "inlet": site.inlet,
        "month": month.label,
        "split_id": split.split_id,
        "holdout_start": split.holdout_start.date().isoformat(),
        "holdout_end": (split.holdout_end - pd.Timedelta(days=1)).date().isoformat(),
        "buffer_start": split.buffer_start.date().isoformat(),
        "buffer_end": (split.buffer_end - pd.Timedelta(days=1)).date().isoformat(),
        "train_observations": train_observations,
        "holdout_observations": holdout_observations,
        "basis_family": scenario.basis_family,
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
    return record


def aggregate_split_scores(split_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate split-level scores to one row per site, month, and candidate."""
    group_columns = [
        "site",
        "inlet",
        "month",
        "basis_family",
        "class_mode",
        "allocation",
        "split_step",
        "split_mode",
        "geometry",
        "target_regions",
    ]
    numeric_columns = [
        "smooth_perturbation_mean_nrmse",
        "smooth_perturbation_max_nrmse",
        "boundary_perturbation_mean_nrmse",
        "all_perturbation_mean_nrmse",
        "prior_flux_obs_nrmse",
        "prior_flux_obs_rmse",
        "prior_flux_obs_bias",
        "prior_flux_obs_corr",
        "flux_field_nrmse",
        *[f"perturbation_{name}_nrmse" for name in (*SMOOTH_PERTURBATIONS, *BOUNDARY_ALIGNED_PERTURBATIONS)],
    ]
    grouped = split_scores.groupby(group_columns, sort=False, dropna=False)
    scores = grouped[numeric_columns].mean().reset_index()
    scores["n_splits"] = grouped.size().to_numpy()
    scores["actual_regions"] = grouped["actual_regions"].mean().to_numpy()
    scores["actual_regions_min"] = grouped["actual_regions"].min().to_numpy()
    scores["actual_regions_max"] = grouped["actual_regions"].max().to_numpy()
    scores["candidate"] = [candidate_display_label(row) for row in dataframe_records(scores)]
    return scores.sort_values(["site", "month", "smooth_perturbation_mean_nrmse"]).reset_index(drop=True)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return row dictionaries with a Pyright-friendly type."""
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def candidate_display_label(row: dict[str, Any]) -> str:
    """Return a compact label for a candidate row."""
    basis_family = str(row["basis_family"])
    if basis_family != "region_constrained":
        return basis_family
    return (
        f"{row['class_mode']}/{row['allocation']}/{row['split_step']}/"
        f"{row['split_mode']}/{row['geometry']}"
    )


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
        ("land/sea axis count row", "region_constrained", "land_sea", "area", "axis_parallel", "count", "row_column"),
        (
            "lat/lon geometry",
            "region_constrained",
            "land_sea",
            "area",
            "axis_parallel",
            "count",
            "lat_lon_metres",
        ),
        (
            "balanced split",
            "region_constrained",
            "land_sea",
            "area",
            "axis_parallel",
            "balanced",
            "row_column",
        ),
        ("inertial split", "region_constrained", "land_sea", "area", "inertial", "count", "row_column"),
        ("no mask", "region_constrained", "no_mask", "single_class", "axis_parallel", "count", "row_column"),
        (
            "selected countries",
            "region_constrained",
            "selected_countries",
            "area",
            "axis_parallel",
            "count",
            "row_column",
        ),
        (
            "weight allocation",
            "region_constrained",
            "selected_countries",
            "weight",
            "axis_parallel",
            "count",
            "row_column",
        ),
        (
            "inertial + metre geometry",
            "region_constrained",
            "selected_countries",
            "area",
            "inertial",
            "count",
            "lat_lon_metres",
        ),
        ("bucketbasisfunction", "bucketbasisfunction", "legacy", "legacy", "bucketbasisfunction", "legacy", "row_column"),
        (
            "quadtreebasisfunction",
            "quadtreebasisfunction",
            "legacy",
            "legacy",
            "quadtreebasisfunction",
            "legacy",
            "row_column",
        ),
    ]
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 16), sharex=True, sharey=True)
    for ax, spec in zip(axes.ravel(), contrast_specs, strict=True):
        title, basis_family, class_mode, allocation, split_step, split_mode, geometry = spec
        scenario = find_scenario(scenarios, basis_family, class_mode, allocation, split_step, split_mode, geometry)
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
    """Plot site/month CV scores for all candidates as a heatmap."""
    ordered_candidates = (
        scores.groupby("candidate", sort=False)["smooth_perturbation_mean_nrmse"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    context_order = [
        f"{site.site}\n{month.label}"
        for site in SITES
        for month in MONTHS
    ]
    matrix = np.full((len(ordered_candidates), len(context_order)), np.nan)
    for row_index, candidate in enumerate(ordered_candidates):
        for col_index, context in enumerate(context_order):
            site, month = context.split("\n")
            match = scores.loc[
                (scores["candidate"] == candidate) & (scores["site"] == site) & (scores["month"] == month),
                "smooth_perturbation_mean_nrmse",
            ]
            if len(match) == 1:
                matrix[row_index, col_index] = float(match.iloc[0])

    height = max(10.0, 0.28 * len(ordered_candidates))
    fig, ax = plt.subplots(figsize=(12, height), constrained_layout=True)
    image = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(context_order)), labels=context_order)
    ax.set_yticks(np.arange(len(ordered_candidates)), labels=ordered_candidates, fontsize=6)
    ax.set_xlabel("Held-out footprint context")
    ax.set_title("Mean CV smooth-perturbation NRMSE by site and month")
    fig.colorbar(image, ax=ax, shrink=0.75, label="mean split NRMSE")
    return save_figure(fig, "basis_option_score_heatmaps_250.png")


def plot_ranked_scores(scores: pd.DataFrame) -> str:
    """Plot option combinations sorted by overall CV observation NRMSE."""
    grouped = (
        scores.groupby(
            [
                "candidate",
                "basis_family",
                "class_mode",
                "allocation",
                "split_step",
                "split_mode",
                "geometry",
            ],
            sort=False,
        )["smooth_perturbation_mean_nrmse"]
        .mean()
        .reset_index()
        .sort_values("smooth_perturbation_mean_nrmse")
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(12, 11), constrained_layout=True)
    colors = {
        "region_constrained": "tab:blue",
        "bucketbasisfunction": "tab:orange",
        "quadtreebasisfunction": "tab:green",
    }
    y = np.arange(len(grouped))
    ax.scatter(
        grouped["smooth_perturbation_mean_nrmse"],
        y,
        c=[colors.get(value, "tab:gray") for value in grouped["basis_family"]],
        s=34,
    )
    ax.set_yticks(y, labels=grouped["candidate"], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Mean CV smooth-perturbation NRMSE across TAC/MHD January/July")
    ax.set_title("250-region option combinations sorted by cross-validated reconstruction score")
    for basis_family, color in colors.items():
        ax.scatter([], [], color=color, label=basis_family)
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
    basis_family: str,
    class_mode: str,
    allocation: str,
    split_step: str,
    split_mode: str,
    geometry: str,
) -> Scenario:
    """Return one scenario by its option keys."""
    for scenario in scenarios:
        if (
            scenario.basis_family == basis_family
            and scenario.class_mode == class_mode
            and scenario.allocation == allocation
            and scenario.split_step == split_step
            and scenario.split_mode == split_mode
            and scenario.geometry == geometry
        ):
            return scenario
    raise ValueError(
        f"No scenario found for {(basis_family, class_mode, allocation, split_step, split_mode, geometry)!r}."
    )


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
    split_scores: pd.DataFrame,
) -> None:
    """Write the documentation page."""
    best_rows = top_score_rows(scores, n=5)
    selected_country_text = ", ".join(short_country_label(name) for name in SELECTED_COUNTRIES)
    lines = [
        "# Constrained Basis Algorithm Options",
        "",
        "This page explains the lower-level constrained basis algorithms and compares 250-target-region variants on Blue Pebble OpenGHG data.",
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
        "The comparison also includes legacy `bucketbasisfunction` and `quadtreebasisfunction` rows generated from the same training weights. Their actual region counts can differ from the 250 target.",
        "",
        "The important separation is that `weights` define contribution/importance, while `geometry` defines physical coordinates for split shape. Lat/lon geometry does not change contribution weights, class allocation, or posterior weighting.",
        "",
        "For the no-mask score rows below, allocation is reported as `single_class` because there is only one class. The generator uses the normal weight-allocation API internally, but no inter-class allocation decision is being tested in that case.",
        "",
        "## Blue Pebble Cross-Validation Data",
        "",
        f"The script reads footprints and CH4 flux from OpenGHG store `{BLUE_PEBBLE_STORE}` on Blue Pebble. It uses TAC 185m and MHD 10m EUROPE NAME inert footprints, and the monthly `{BLUE_PEBBLE_FLUX_SOURCE}` CH4 flux product.",
        "",
        "January and July 2019 are scored separately. Each site/month uses two temporal CV splits: a one-week holdout starting on days "
        f"{HOLDOUT_START_DAYS[0]} and {HOLDOUT_START_DAYS[1]}, with a two-day buffer excluded before and after the held-out week. Basis weights are built only from the remaining in-month footprints.",
        "",
        f"Aggregate scores are written to `{SCORES_PATH.relative_to(ROOT)}` and split-level scores are written to `{SPLIT_SCORES_PATH.relative_to(ROOT)}`. The split-level table contains {len(split_scores)} scored rows.",
        "",
        "## Region Class Modes",
        "",
        "The plots below use three class modes. The selected-country mode treats ocean as one class, keeps selected large European-domain countries as separate classes, and groups all remaining land as `other_land`. The selected countries are: "
        f"{selected_country_text}.",
        "",
        f"![Region class modes]({class_figure})",
        "",
        "## Held-Out Forward-Model Compression Score",
        "",
        "A normal multiplicative prior basis exactly reproduces the prior modelled observations when all basis coefficients are one: summing projected `H` over all regions gives the same `sum(fp * flux)` as the full grid. That RMSE is therefore a trivial zero and is not useful for comparing basis shapes.",
        "",
        "Instead, this page uses deterministic perturbation-reconstruction diagnostics on held-out footprints. Fine-grid flux-scale perturbation fields are applied to held-out `fp * flux`, then each perturbation is projected to one training-weighted mean value per basis region. This is still an optimistic representability diagnostic, but the footprint data used for scoring are not used to construct the basis weights.",
        "",
        "The headline score and ranking use only smooth perturbations: latitude and longitude gradients plus western-Europe and Nordic Gaussian blobs. Boundary-aligned perturbations, a land/ocean contrast and a selected-country patch, are reported separately because they can tautologically reward basis masks that hard-code the same boundaries.",
        "",
        "A secondary score also projects the prior flux field itself to one cell-mean value per region and compares held-out modelled observations from full and projected flux. That is a prior observation-space compression score; low observation NRMSE can still coexist with poor spatial flux-field reconstruction.",
        "",
        "Neither score is a posterior-quality metric, and neither replaces posterior or synthetic-recovery tests.",
        "",
        "## Basis Map Contrasts",
        "",
        f"![Basis option contrasts]({basis_figure})",
        "",
        "## Scores For 250-Region Options",
        "",
        "The heatmap shows the mean split score for each site/month context. The ranked plot averages the four site/month aggregate rows for each candidate.",
        "",
        f"![Score heatmaps]({score_heatmap})",
        "",
        f"![Ranked scores]({score_ranked})",
        "",
        "### Best Smooth-Perturbation Scores By Site And Month",
        "",
        "| site | month | rank | candidate | regions | splits | smooth perturb NRMSE | max smooth NRMSE | boundary perturb NRMSE | prior obs NRMSE | flux-field NRMSE |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        *best_rows,
        "",
        "## Interpretation",
        "",
        "- Lower smooth-perturbation NRMSE means the basis preserves the held-out observation response of the deterministic smooth fine-grid perturbations more efficiently.",
        "- Lower boundary-perturbation NRMSE means the basis preserves perturbations that match land/ocean or selected-country boundaries. It is useful context, but it is not used for the headline ranking.",
        "- Lower prior observation NRMSE means the basis preserves the prior forward model better under a region-mean flux-field approximation, not that it preserves the full spatial flux field.",
        "- Balanced splits often help when the score is dominated by high-contribution areas, but they are not guaranteed to produce visually regular regions.",
        "- Region classes impose hard boundaries, which can help interpretability but can also spend regions on low-contribution classes.",
        "- Lat/lon metre geometry is a physical-coordinate correction. It can change region shapes, especially for inertial or high-latitude splits, but it is not itself a weight-balancing rule.",
        "- Split-stopping policies are not included in the score matrix because they can return fewer than 250 actual regions, making direct comparison less clean.",
        "",
        "## What This Does Not Prove",
        "",
        "These scores do not show whether an inversion posterior improves. For that, use a posterior or posterior-equivalent test: prior/error-weighted `H`, observation-error weighting, linear-Gaussian posterior covariance and resolution, synthetic truth recovery, or paired HPC-CI posterior runs.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def top_score_rows(scores: pd.DataFrame, *, n: int) -> list[str]:
    """Format the best score rows for markdown."""
    rows: list[str] = []
    for (site, month), group_obj in scores.groupby(["site", "month"], sort=False):
        group = cast(pd.DataFrame, group_obj).sort_values("smooth_perturbation_mean_nrmse").head(n)
        for rank, row in enumerate(dataframe_records(group), start=1):
            rows.append(
                f"| {site} | {month} | {rank} | {row['candidate']} | "
                f"{row['actual_regions']:.1f} | {row['n_splits']} | "
                f"{row['smooth_perturbation_mean_nrmse']:.4f} | "
                f"{row['smooth_perturbation_max_nrmse']:.4f} | "
                f"{row['boundary_perturbation_mean_nrmse']:.4f} | "
                f"{row['prior_flux_obs_nrmse']:.4f} | {row['flux_field_nrmse']:.4f} |"
            )
    return rows


if __name__ == "__main__":
    main()
