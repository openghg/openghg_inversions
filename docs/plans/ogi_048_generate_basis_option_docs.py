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
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.figure import Figure
from openghg.retrieve import get_flux, get_footprint

from openghg_inversions.basis._functions import _mean_fp_times_mean_flux
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
OVERALL_SCORES_PATH = ROOT / "docs" / "plans" / "ogi_048_basis_option_overall_scores.csv"

COUNTRY_PATH = ROOT / "tests" / "data" / "country_EUROPE.nc"
LPDM_COUNTRY_PATH = Path("/group/chem/acrg/LPDM/countries/country_EUROPE.nc")

BLUE_PEBBLE_STORE = "shared_store_zarr"
BLUE_PEBBLE_DOMAIN = "EUROPE"
BLUE_PEBBLE_FLUX_SOURCE = "edgarv80_wetchartsv131"

TARGET_REGIONS = 250
QUADTREE_SEED = 42
HOLDOUT_DAYS = 7
BUFFER_DAYS = 2
HOLDOUT_START_DAYS = (6, 13, 20)
OBJECTIVE_GROUPS = ("no_mask", "land_sea", "selected_countries")
OBJECTIVE_LABELS = {
    "no_mask": "No Mask",
    "land_sea": "Land/Sea Mask",
    "selected_countries": "Selected Countries",
}
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
    flux_for_weights: xr.DataArray
    footprint: xr.DataArray
    country: xr.DataArray
    country_names: tuple[str, ...]


@dataclass(frozen=True)
class RepresentativeFields:
    """Representative fields used to explain the scoring inputs."""

    month_label: str
    split_id: str
    flux: xr.DataArray
    fp_x_flux: xr.DataArray


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
    heldout_cv_nrmse: float
    heldout_cv_rmse: float
    heldout_cv_bias: float
    heldout_cv_corr: float
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

    split_scores, representative_scenarios, representative_class_modes, representative_fields = (
        build_cross_validation_scores()
    )
    split_scores.to_csv(SPLIT_SCORES_PATH, index=False)
    scores = aggregate_split_scores(split_scores)
    scores.to_csv(SCORES_PATH, index=False)
    overall_scores = aggregate_overall_scores(split_scores)
    overall_scores.to_csv(OVERALL_SCORES_PATH, index=False)

    class_figure = plot_region_classes(representative_class_modes)
    field_figure = plot_input_fields(representative_fields)
    basis_figure = plot_basis_contrasts(representative_scenarios, overall_scores)
    score_heatmap = plot_score_heatmap(scores)
    score_ranked = plot_ranked_scores(overall_scores)

    write_report(
        class_figure=class_figure,
        field_figure=field_figure,
        basis_figure=basis_figure,
        score_heatmap=score_heatmap,
        score_ranked=score_ranked,
        scores=scores,
        split_scores=split_scores,
        overall_scores=overall_scores,
    )


def build_cross_validation_scores() -> tuple[
    pd.DataFrame,
    list[Scenario],
    dict[str, xr.DataArray],
    RepresentativeFields,
]:
    """Build split-level scores for all sites, months, and basis candidates."""
    records: list[dict[str, Any]] = []
    representative_scenarios: list[Scenario] = []
    representative_class_modes: dict[str, xr.DataArray] | None = None
    representative_fields: RepresentativeFields | None = None
    basis_training_sites = ",".join(site.site for site in SITES)

    for month in MONTHS:
        inputs_by_site: dict[str, MonthInputs] = {}
        for site in SITES:
            print(f"Loading {site.site} {site.inlet} {month.label}", flush=True)
            inputs_by_site[site.site] = load_blue_pebble_month(site=site, month=month)

        splits = build_month_splits(month)
        reference_inputs = inputs_by_site[SITES[0].site]

        for split in splits:
            print(f"Building shared {month.label} basis for {split.split_id}", flush=True)
            train_footprints: dict[str, xr.DataArray] = {}
            holdout_footprints: dict[str, xr.DataArray] = {}
            for site in SITES:
                inputs = inputs_by_site[site.site]
                train_footprint = select_training_footprint(inputs.footprint, month=month, split=split)
                holdout_footprint = inputs.footprint.sel(time=slice(split.holdout_start, split.holdout_end))
                holdout_footprint = holdout_footprint.where(holdout_footprint.time < split.holdout_end, drop=True)

                if train_footprint.sizes["time"] == 0:
                    raise ValueError(f"No training observations for {site.site} {month.label} {split.split_id}.")
                if holdout_footprint.sizes["time"] == 0:
                    raise ValueError(f"No holdout observations for {site.site} {month.label} {split.split_id}.")

                train_footprints[site.site] = train_footprint
                holdout_footprints[site.site] = holdout_footprint

            fp_x_flux = build_combined_fp_x_flux(
                footprints=[train_footprints[site.site] for site in SITES],
                flux=reference_inputs.flux_for_weights,
            )
            weights = normalize_weights(fp_x_flux)
            geometry = LatLonGridGeometry.from_dataarray(weights)
            country = align_to_test_grid(reference=weights, target=reference_inputs.country, target_name="country")
            class_modes = build_region_class_modes(country, weights, reference_inputs.country_names)
            candidate_labels = build_candidate_labels(
                weights=weights,
                class_modes=class_modes,
                geometry=geometry,
            )
            if representative_fields is None:
                representative_fields = RepresentativeFields(
                    month_label=month.label,
                    split_id=split.split_id,
                    flux=reference_inputs.flux,
                    fp_x_flux=fp_x_flux,
                )

            for site in SITES:
                print(f"Scoring {site.site} {month.label} {split.split_id}", flush=True)
                inputs = inputs_by_site[site.site]
                scenarios = score_candidates(
                    candidates=candidate_labels,
                    flux=inputs.flux,
                    holdout_footprint=holdout_footprints[site.site],
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
                        basis_training_sites=basis_training_sites,
                        basis_train_observations=sum(
                            train_footprints[training_site.site].sizes["time"] for training_site in SITES
                        ),
                        score_site_holdout_observations=holdout_footprints[site.site].sizes["time"],
                    )
                    for scenario in scenarios
                )

    if representative_class_modes is None:
        raise RuntimeError("No representative class modes were generated.")
    if representative_fields is None:
        raise RuntimeError("No representative input fields were generated.")

    return pd.DataFrame.from_records(records), representative_scenarios, representative_class_modes, representative_fields


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
    flux_for_weights = flux_dataset.flux.transpose("time", "lat", "lon").astype(np.float64).load().rename("flux")
    flux = flux_for_weights.mean("time").transpose("lat", "lon").rename("flux")

    footprint_mean = footprint.mean("time").transpose("lat", "lon")
    flux = align_to_test_grid(reference=footprint_mean, target=flux, target_name="flux")
    flux_for_weights = align_time_grid_to_test_grid(
        reference=footprint_mean,
        target=flux_for_weights,
        target_name="flux",
    )
    footprint = align_time_grid_to_test_grid(
        reference=footprint_mean,
        target=footprint,
        target_name="footprint",
    )
    country, country_names = load_country(reference=footprint_mean)
    return MonthInputs(
        flux=flux,
        flux_for_weights=flux_for_weights,
        footprint=footprint,
        country=country,
        country_names=country_names,
    )


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


def build_combined_fp_x_flux(*, footprints: list[xr.DataArray], flux: xr.DataArray) -> xr.DataArray:
    """Build wrapper-equivalent mean footprint times mean flux from all training footprints."""
    return _mean_fp_times_mean_flux(flux, footprints).fillna(0.0).rename("fp_x_flux")


def normalize_weights(fp_x_flux: xr.DataArray) -> xr.DataArray:
    """Normalize a footprint-times-flux field for basis construction."""
    weights = fp_x_flux.rename("weight")
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


def build_candidate_labels(
    *,
    weights: xr.DataArray,
    class_modes: dict[str, xr.DataArray],
    geometry: LatLonGridGeometry,
) -> list[CandidateLabels]:
    """Build all feasible constrained and legacy comparison basis candidates."""
    candidates: list[CandidateLabels] = []
    for class_mode, region_classes in class_modes.items():
        allocations = ("single_class",) if class_mode == "no_mask" else ("weight",)
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
        (
            "bucketbasisfunction",
            "land_sea",
            lambda: weighted_algorithm(normalized_grid, nregion=TARGET_REGIONS),
        ),
        (
            "quadtreebasisfunction",
            "no_mask",
            lambda: quadtree_algorithm(normalized_grid, nbasis=TARGET_REGIONS, seed=QUADTREE_SEED),
        ),
    )
    for basis_family, class_mode, build_labels in legacy_specs:
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
                class_mode=class_mode,
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
) -> list[Scenario]:
    """Score each candidate on one held-out footprint window."""
    scenarios: list[Scenario] = []
    y_full = modelled_observations(holdout_footprint, flux)
    for candidate in candidates:
        labels = align_to_test_grid(reference=flux, target=candidate.labels, target_name="basis")
        projected_flux = project_flux_to_regions(flux, labels)
        y_projected = modelled_observations(holdout_footprint, projected_flux)
        scenarios.append(
            Scenario(
                basis_family=candidate.basis_family,
                class_mode=candidate.class_mode,
                allocation=candidate.allocation,
                split_step=candidate.split_step,
                split_mode=candidate.split_mode,
                geometry=candidate.geometry,
                labels=labels,
                heldout_cv_nrmse=normalized_rmse(y_projected, y_full),
                heldout_cv_rmse=rmse(y_projected, y_full),
                heldout_cv_bias=float((y_projected - y_full).mean()),
                heldout_cv_corr=correlation(y_projected, y_full),
                actual_regions=candidate.actual_regions,
            )
        )
    return scenarios


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
    basis_training_sites: str,
    basis_train_observations: int,
    score_site_holdout_observations: int,
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
        "basis_training_sites": basis_training_sites,
        "basis_train_observations": basis_train_observations,
        "score_site_holdout_observations": score_site_holdout_observations,
        "basis_family": scenario.basis_family,
        "class_mode": scenario.class_mode,
        "allocation": scenario.allocation,
        "split_step": scenario.split_step,
        "split_mode": scenario.split_mode,
        "geometry": scenario.geometry,
        "target_regions": TARGET_REGIONS,
        "actual_regions": scenario.actual_regions,
        "heldout_cv_nrmse": scenario.heldout_cv_nrmse,
        "heldout_cv_rmse": scenario.heldout_cv_rmse,
        "heldout_cv_bias": scenario.heldout_cv_bias,
        "heldout_cv_corr": scenario.heldout_cv_corr,
    }
    return record


def aggregate_split_scores(split_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate split-level scores to one row per site, month, and candidate."""
    group_columns = [
        "site",
        "inlet",
        "month",
        "basis_training_sites",
        "basis_family",
        "class_mode",
        "allocation",
        "split_step",
        "split_mode",
        "geometry",
        "target_regions",
    ]
    numeric_columns = [
        "heldout_cv_nrmse",
        "heldout_cv_rmse",
        "heldout_cv_bias",
        "heldout_cv_corr",
    ]
    grouped = split_scores.groupby(group_columns, sort=False, dropna=False)
    scores = grouped[numeric_columns].mean().reset_index()
    scores["n_splits"] = grouped.size().to_numpy()
    scores["actual_regions"] = grouped["actual_regions"].mean().to_numpy()
    scores["actual_regions_min"] = grouped["actual_regions"].min().to_numpy()
    scores["actual_regions_max"] = grouped["actual_regions"].max().to_numpy()
    scores["candidate"] = [candidate_display_label(row) for row in dataframe_records(scores)]
    return scores.sort_values(["site", "month", "class_mode", "heldout_cv_nrmse"]).reset_index(drop=True)


def aggregate_overall_scores(split_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate split-level scores to one overall row per candidate."""
    group_columns = [
        "basis_family",
        "class_mode",
        "allocation",
        "split_step",
        "split_mode",
        "geometry",
        "target_regions",
        "basis_training_sites",
    ]
    grouped = split_scores.groupby(group_columns, sort=False, dropna=False)
    scores = grouped[["heldout_cv_nrmse", "heldout_cv_rmse", "heldout_cv_bias", "heldout_cv_corr"]].mean()
    scores = scores.reset_index()
    scores["n_score_rows"] = grouped.size().to_numpy()
    scores["n_basis_splits"] = grouped[["month", "split_id"]].apply(
        lambda frame: frame.drop_duplicates().shape[0]
    ).to_numpy()
    scores["actual_regions"] = grouped["actual_regions"].mean().to_numpy()
    scores["actual_regions_min"] = grouped["actual_regions"].min().to_numpy()
    scores["actual_regions_max"] = grouped["actual_regions"].max().to_numpy()
    scores["candidate"] = [candidate_display_label(row) for row in dataframe_records(scores)]
    return scores.sort_values(["class_mode", "heldout_cv_nrmse"]).reset_index(drop=True)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return row dictionaries with a Pyright-friendly type."""
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def candidate_display_label(row: dict[str, Any]) -> str:
    """Return a compact label for a candidate row."""
    basis_family = str(row["basis_family"])
    if basis_family == "bucketbasisfunction":
        return "weighted/bucket"
    if basis_family == "quadtreebasisfunction":
        return "quadtree"
    return (
        f"{row['allocation']}/{row['split_step']}/"
        f"{row['split_mode']}/{row['geometry']}"
    )


def objective_label(class_mode: str) -> str:
    """Return a report label for an objective group."""
    return OBJECTIVE_LABELS.get(class_mode, class_mode.replace("_", " ").title())


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


def plot_input_fields(fields: RepresentativeFields) -> str:
    """Plot representative flux and footprint-times-flux fields."""
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5), constrained_layout=True)
    plot_log_field(
        axes[0],
        fields.flux,
        title=f"{fields.month_label} prior flux",
        colorbar_label="flux",
        fig=fig,
    )
    plot_log_field(
        axes[1],
        fields.fp_x_flux,
        title=f"{fields.split_id} training fp x flux",
        colorbar_label="fp x flux",
        fig=fig,
    )
    return save_figure(fig, "basis_option_input_fields_250.png")


def plot_basis_contrasts(scenarios: list[Scenario], overall_scores: pd.DataFrame) -> str:
    """Plot best representative basis maps for each objective group."""
    row_specs = best_map_rows(overall_scores)
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 10.5), sharex=True, sharey=True)
    for row_index, class_mode in enumerate(OBJECTIVE_GROUPS):
        for col_index, row in enumerate(row_specs[class_mode]):
            ax = axes[row_index, col_index]
            scenario = find_scenario(
                scenarios,
                str(row["basis_family"]),
                str(row["class_mode"]),
                str(row["allocation"]),
                str(row["split_step"]),
                str(row["split_mode"]),
                str(row["geometry"]),
            )
            plot_basis_label_map(ax, scenario.labels)
            rank_label = "legacy" if row["basis_family"] != "region_constrained" else f"rank {col_index + 1}"
            ax.set_title(
                f"{objective_label(class_mode)} {rank_label}\n"
                f"{row['candidate']}\n"
                f"NRMSE={row['heldout_cv_nrmse']:.3f}, regions={scenario.actual_regions}",
                fontsize=9,
            )
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")

        for col_index in range(len(row_specs[class_mode]), axes.shape[1]):
            axes[row_index, col_index].axis("off")

    fig.tight_layout()
    return save_figure(fig, "basis_option_contrasts_250.png")


def best_map_rows(overall_scores: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Return candidate rows to display as maps for each objective group."""
    rows: dict[str, list[dict[str, Any]]] = {}
    legacy_by_group = {
        "no_mask": "quadtreebasisfunction",
        "land_sea": "bucketbasisfunction",
    }
    for class_mode in OBJECTIVE_GROUPS:
        group = overall_scores.loc[overall_scores["class_mode"] == class_mode].sort_values("heldout_cv_nrmse")
        legacy_family = legacy_by_group.get(class_mode)
        if legacy_family is None:
            selected = group.head(4)
        else:
            constrained = group.loc[group["basis_family"] == "region_constrained"].head(3)
            legacy = group.loc[group["basis_family"] == legacy_family].head(1)
            selected = pd.concat([constrained, legacy], ignore_index=True)
        rows[class_mode] = dataframe_records(selected)
    return rows


def plot_basis_label_map(ax: plt.Axes, labels: xr.DataArray) -> None:
    """Plot basis labels with shuffled colors so adjacent labels are visually distinct."""
    shuffled = shuffled_label_values(labels)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")
    ax.pcolormesh(
        labels.lon,
        labels.lat,
        shuffled,
        shading="auto",
        cmap=cmap,
        rasterized=True,
    )


def shuffled_label_values(labels: xr.DataArray) -> np.ndarray:
    """Return deterministic shuffled plotting values for positive basis labels."""
    label_values = np.asarray(labels.values, dtype=np.int64)
    shuffled = np.full(label_values.shape, np.nan, dtype=np.float64)
    positive_labels = np.unique(label_values[label_values > 0])
    if positive_labels.size == 0:
        return shuffled
    rng = np.random.default_rng(QUADTREE_SEED)
    color_values = np.linspace(0.0, 1.0, positive_labels.size, endpoint=True)
    rng.shuffle(color_values)
    for label, color_value in zip(positive_labels, color_values, strict=True):
        shuffled[label_values == label] = color_value
    return shuffled


def plot_log_field(ax: plt.Axes, field: xr.DataArray, *, title: str, colorbar_label: str, fig: Figure) -> None:
    """Plot a positive field on a log colour scale."""
    values = np.asarray(field.values, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise ValueError(f"{title!r} has no positive values for log-scale plotting.")
    vmin = float(np.nanpercentile(positive, 2.0))
    vmax = float(np.nanpercentile(positive, 98.0))
    if vmin <= 0.0 or vmin >= vmax:
        vmin = float(np.min(positive))
        vmax = float(np.max(positive))
    masked = field.where(field > 0.0)
    mesh = ax.pcolormesh(
        field.lon,
        field.lat,
        masked,
        shading="auto",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="magma",
        rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    fig.colorbar(mesh, ax=ax, shrink=0.75, label=colorbar_label)


def plot_score_heatmap(scores: pd.DataFrame) -> str:
    """Plot site/month CV scores for all candidates as grouped heatmaps."""
    metric = "heldout_cv_nrmse"
    context_order = [
        f"{site.site}\n{month.label}"
        for site in SITES
        for month in MONTHS
    ]
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 13), constrained_layout=True)
    image = None
    vmin = float(scores[metric].min())
    vmax = float(scores[metric].max())
    for ax, class_mode in zip(axes, OBJECTIVE_GROUPS, strict=True):
        group = scores.loc[scores["class_mode"] == class_mode]
        ordered_candidates = group.groupby("candidate", sort=False)[metric].mean().sort_values().index.tolist()
        matrix = np.full((len(ordered_candidates), len(context_order)), np.nan)
        for row_index, candidate in enumerate(ordered_candidates):
            for col_index, context in enumerate(context_order):
                site, month = context.split("\n")
                match = group.loc[
                    (group["candidate"] == candidate) & (group["site"] == site) & (group["month"] == month),
                    metric,
                ]
                if len(match) == 1:
                    matrix[row_index, col_index] = float(match.iloc[0])

        image = ax.imshow(matrix, cmap="viridis_r", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(context_order)), labels=context_order)
        ax.set_yticks(np.arange(len(ordered_candidates)), labels=ordered_candidates, fontsize=6)
        ax.set_title(f"{objective_label(class_mode)} objective")
        ax.set_xlabel("Held-out footprint context")

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="mean held-out CV NRMSE")
    return save_figure(fig, "basis_option_score_heatmaps_250.png")


def plot_ranked_scores(scores: pd.DataFrame) -> str:
    """Plot option combinations sorted by overall held-out CV observation NRMSE."""
    metric = "heldout_cv_nrmse"
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 8), sharex=True, constrained_layout=True)
    colors = {
        "region_constrained": "tab:blue",
        "bucketbasisfunction": "tab:orange",
        "quadtreebasisfunction": "tab:green",
    }
    for ax, class_mode in zip(axes, OBJECTIVE_GROUPS, strict=True):
        group = scores.loc[scores["class_mode"] == class_mode].sort_values(metric).reset_index(drop=True)
        y = np.arange(len(group))
        ax.scatter(
            group[metric],
            y,
            c=[colors.get(value, "tab:gray") for value in group["basis_family"]],
            s=34,
        )
        ax.set_yticks(y, labels=group["candidate"], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Overall held-out CV NRMSE")
        ax.set_title(objective_label(class_mode))
        ax.grid(True, axis="x", linewidth=0.4, alpha=0.4)
    for basis_family, color in colors.items():
        axes[-1].scatter([], [], color=color, label=basis_family)
    axes[-1].legend(loc="lower right", fontsize=8)
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
    field_figure: str,
    basis_figure: str,
    score_heatmap: str,
    score_ranked: str,
    scores: pd.DataFrame,
    split_scores: pd.DataFrame,
    overall_scores: pd.DataFrame,
) -> None:
    """Write the documentation page."""
    overall_rows = overall_score_rows(overall_scores, n=5)
    context_rows = context_score_rows(scores, n=2)
    selected_country_text = ", ".join(short_country_label(name) for name in SELECTED_COUNTRIES)
    holdout_day_text = ", ".join(str(day) for day in HOLDOUT_START_DAYS)
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
        "- **Class allocation**: explicit per-class counts, automatic allocation by class total weight, or automatic allocation by cell count. The Blue Pebble score matrix below uses only weight allocation for masked objectives.",
        "- **Greedy orchestration**: repeatedly split the currently highest-weight partition until the target is reached or no acceptable split remains.",
        "- **Partition step**: axis-parallel row/column splits or inertial principal-axis splits.",
        "- **Split mode**: count-based splits or balanced splits near half parent weight.",
        "- **Geometry**: row/column index geometry or local lat/lon metre geometry for split-shape decisions.",
        "- **Split stopping**: optional policies that reject proposed child regions. When stopping is enabled, the requested region count is an upper target.",
        "",
        "The comparison also includes legacy `bucketbasisfunction` and `quadtreebasisfunction` rows generated from the same training weights. Their actual region counts can differ from the 250 target. `bucketbasisfunction` is shown as `weighted/bucket`; in this codebase the `weighted_algorithm` alias uses the land/sea weighted bucket splitter, so it is grouped with the land/sea objective. `quadtreebasisfunction` is shown as `quadtree` and grouped with the no-mask objective.",
        "",
        "The important separation is that `weights` define contribution/importance, while `geometry` defines physical coordinates for split shape. Lat/lon geometry does not change contribution weights, class allocation, or posterior weighting. The Blue Pebble generator builds these weights with the same multi-site footprint-times-flux reduction used by the production `basis_functions_wrapper` path.",
        "",
        "For the no-mask score rows below, allocation is reported as `single_class` because there is only one class. The generator uses the normal weight-allocation API internally, but no inter-class allocation decision is being tested in that case.",
        "",
        "## Option Shorthand",
        "",
        "Candidate labels use the format `allocation/split_step/split_mode/geometry`. The objective group is shown separately because no mask, land/sea mask, and selected-country mask answer different scientific basis-design questions.",
        "",
        "| shorthand | meaning |",
        "|---|---|",
        "| `no_mask` | one class over the full domain; no hard class boundary is imposed |",
        "| `land_sea` | two hard classes, land and ocean |",
        "| `selected_countries` | ocean, selected countries, and `other_land` are separate classes |",
        "| `single_class` | the no-mask allocation case; there is no inter-class allocation decision |",
        "| `weight` | allocate target regions to classes by total training weight |",
        "| `axis_parallel` | split a region with a row- or column-aligned cut |",
        "| `inertial` | split a region using its principal weighted axis |",
        "| `count` | choose splits by child cell counts |",
        "| `balanced` | choose splits near half of the parent-region weight |",
        "| `row_column` | use grid row/column coordinates when evaluating split shape |",
        "| `lat_lon_metres` | use local metre-scaled longitude/latitude coordinates for split shape |",
        "| `weighted/bucket` | legacy `bucketbasisfunction`; uses the land/sea weighted bucket algorithm |",
        "| `quadtree` | legacy `quadtreebasisfunction`; recursively subdivides the grid without a class mask |",
        "| `CV` | cross-validation; here, temporal holdout scoring with one shared basis per month/split |",
        "| `NRMSE` | RMSE divided by the RMS of the full-grid held-out modelled observation |",
        "| `fp` | OpenGHG/NAME footprint field |",
        "| `H` | basis sensitivity matrix produced by projecting `fp * flux` onto basis regions |",
        "| `CH4` | methane |",
        "| `NAME` | the Numerical Atmospheric-dispersion Modelling Environment transport model |",
        "| `TAC`, `MHD` | Tacolneston and Mace Head measurement sites |",
        "",
        "## Blue Pebble Cross-Validation Data",
        "",
        f"The script reads footprints and CH4 flux from OpenGHG store `{BLUE_PEBBLE_STORE}` on Blue Pebble. It uses TAC 185m and MHD 10m EUROPE NAME inert footprints, and the monthly `{BLUE_PEBBLE_FLUX_SOURCE}` CH4 flux product.",
        "",
        "January and July 2019 are scored separately. Each month uses three temporal CV splits: a one-week holdout starting on days "
        f"{holdout_day_text}, with a two-day buffer excluded before and after the held-out week. For each month/split, one shared basis is built from the combined remaining TAC and MHD in-month training footprints, matching the production multi-site basis objective. Held-out scores are then reported separately for TAC and MHD.",
        "",
        "Masked constrained candidates use `weight` allocation only, so the generated evidence focuses on the allocation mode used for the current recommendation.",
        "",
        f"Per-score-site/month aggregate scores are written to `{SCORES_PATH.relative_to(ROOT)}`, split-level scores are written to `{SPLIT_SCORES_PATH.relative_to(ROOT)}`, and overall all-score-site/month/split scores are written to `{OVERALL_SCORES_PATH.relative_to(ROOT)}`. The split-level table contains {len(split_scores)} scored rows and includes `basis_training_sites`, `basis_train_observations`, and `score_site_holdout_observations` to make the shared-basis training set explicit.",
        "",
        "## Representative Input Fields",
        "",
        "The log-scale maps below show the representative monthly prior flux and the combined training `fp_x_flux` field used to construct the first displayed basis split. The `fp_x_flux` field is normalized before candidate generation, but the plotted field is the unnormalized footprint-times-flux product.",
        "",
        f"![Representative input fields]({field_figure})",
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
        "A normal multiplicative prior basis exactly reproduces the prior modelled observations when all basis coefficients are one: summing projected `H` over all regions gives the same `sum(fp * flux)` as the full grid. That direct RMSE is therefore a trivial zero and is not useful for comparing basis shapes.",
        "",
        "The score used here is a held-out prior-flux observation-space compression score. For each candidate basis, the prior flux field is approximated by one cell-mean value per basis region. Held-out modelled observations from this projected flux field are compared with held-out modelled observations from the full grid. Lower held-out CV NRMSE means the shared basis preserves the full-grid prior-flux observation response more efficiently on footprints that were not used to construct the basis weights.",
        "",
        "Only this held-out CV score is included in these tables and plots. It is still not a posterior-quality metric and does not replace posterior or synthetic-recovery tests.",
        "",
        "## Best Representative Basis Maps",
        "",
        "The map figure shows the best overall held-out CV candidates by objective, using one representative January basis split for display. No-mask and land/sea rows show the best three constrained candidates plus the matching legacy option. Selected-country has no legacy counterpart, so it shows the best four constrained candidates.",
        "",
        f"![Basis option contrasts]({basis_figure})",
        "",
        "## Grouped Scores For 250-Region Options",
        "",
        "The heatmap shows the mean split score for each held-out site/month context, grouped by objective. The ranked plot uses the overall score averaged over all TAC/MHD January/July split rows for each candidate.",
        "",
        f"![Score heatmaps]({score_heatmap})",
        "",
        f"![Ranked scores]({score_ranked})",
        "",
        "### Overall Held-Out CV Scores",
        "",
        "| objective | rank | candidate | regions | score rows | basis splits | CV NRMSE | CV RMSE | CV bias | CV corr |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        *overall_rows,
        "",
        "### Best Held-Out CV Scores By Site, Month, And Objective",
        "",
        "| objective | score site | month | rank | candidate | regions | CV splits | CV NRMSE | CV RMSE | CV bias | CV corr |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
        *context_rows,
        "",
        "## Interpretation",
        "",
        "- Lower held-out CV NRMSE means the basis preserves the prior forward model better under a region-mean flux-field approximation on held-out footprints.",
        "- Balanced splits often help when the score is dominated by high-contribution areas, but they are not guaranteed to produce visually regular regions.",
        "- Region classes impose hard boundaries, which can help interpretability but can also spend regions on low-contribution classes.",
        "- Lat/lon metre geometry is a physical-coordinate correction. It can change region shapes, especially for inertial or high-latitude splits, but it is not itself a weight-balancing rule.",
        "- The three objective groups should not be read as one single efficiency race. No mask, land/sea, and selected-country masks are often chosen for scientific or reporting reasons as well as basis efficiency.",
        "- Split-stopping policies are not included in the score matrix because they can return fewer than 250 actual regions, making direct comparison less clean.",
        "",
        "## What This Does Not Prove",
        "",
        "These scores do not show whether an inversion posterior improves. For that, use a posterior or posterior-equivalent test: prior/error-weighted `H`, observation-error weighting, linear-Gaussian posterior covariance and resolution, synthetic truth recovery, or paired HPC-CI posterior runs.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def overall_score_rows(overall_scores: pd.DataFrame, *, n: int) -> list[str]:
    """Format the best overall score rows for markdown."""
    rows: list[str] = []
    for class_mode in OBJECTIVE_GROUPS:
        group = overall_scores.loc[overall_scores["class_mode"] == class_mode].sort_values("heldout_cv_nrmse").head(n)
        for rank, row in enumerate(dataframe_records(group), start=1):
            rows.append(
                f"| {objective_label(class_mode)} | {rank} | {row['candidate']} | "
                f"{row['actual_regions']:.1f} | {row['n_score_rows']} | {row['n_basis_splits']} | "
                f"{format_float(row['heldout_cv_nrmse'])} | {format_float(row['heldout_cv_rmse'])} | "
                f"{format_float(row['heldout_cv_bias'])} | {format_float(row['heldout_cv_corr'])} |"
            )
    return rows


def context_score_rows(scores: pd.DataFrame, *, n: int) -> list[str]:
    """Format the best per-context score rows for markdown."""
    rows: list[str] = []
    for class_mode in OBJECTIVE_GROUPS:
        objective_scores = scores.loc[scores["class_mode"] == class_mode]
        for (site, month), group_obj in objective_scores.groupby(["site", "month"], sort=False):
            group = cast(pd.DataFrame, group_obj).sort_values("heldout_cv_nrmse").head(n)
            for rank, row in enumerate(dataframe_records(group), start=1):
                rows.append(
                    f"| {objective_label(class_mode)} | {site} | {month} | {rank} | {row['candidate']} | "
                    f"{row['actual_regions']:.1f} | {row['n_splits']} | "
                    f"{format_float(row['heldout_cv_nrmse'])} | {format_float(row['heldout_cv_rmse'])} | "
                    f"{format_float(row['heldout_cv_bias'])} | {format_float(row['heldout_cv_corr'])} |"
                )
    return rows


def format_float(value: Any, *, digits: int = 4) -> str:
    """Format a numeric score for markdown."""
    numeric = float(value)
    if np.isnan(numeric):
        return "nan"
    if numeric != 0.0 and abs(numeric) < 10**-digits:
        return f"{numeric:.3e}"
    return f"{numeric:.{digits}f}"


if __name__ == "__main__":
    main()
