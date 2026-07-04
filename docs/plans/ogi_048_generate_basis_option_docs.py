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
from heapq import heappop, heappush
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
    InertialSplitStep,
    LatLonGridGeometry,
    MaxChildPCAEccentricity,
    allocate_nbasis_by_class,
    quadtree_algorithm,
    weighted_algorithm,
)
from openghg_inversions.basis.algorithms._weighted import bucket_value_split


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "docs" / "plans" / "figures" / "ogi_048_basis_options"
REPORT_PATH = ROOT / "docs" / "plans" / "ogi_048_basis_algorithm_options.md"
SCORES_PATH = ROOT / "docs" / "plans" / "ogi_048_basis_option_scores.csv"
SPLIT_SCORES_PATH = ROOT / "docs" / "plans" / "ogi_048_basis_option_split_scores.csv"
OVERALL_SCORES_PATH = ROOT / "docs" / "plans" / "ogi_048_basis_option_overall_scores.csv"
SPLIT_HISTORY_PATH = ROOT / "docs" / "plans" / "ogi_048_basis_option_split_history.csv.gz"
REGION_DIAGNOSTICS_PATH = ROOT / "docs" / "plans" / "ogi_048_basis_option_region_diagnostics.csv.gz"
ECCENTRICITY_FIX_CASES_PATH = ROOT / "docs" / "plans" / "ogi_048_basis_option_eccentricity_fix_cases.csv"

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
NEIGHBOUR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))
ECCENTRICITY_FIX_THRESHOLD = 10.0
ECCENTRICITY_FIX_PER_OBJECTIVE_INFINITE_CASE_COUNT = 1
ECCENTRICITY_FIX_PER_OBJECTIVE_FINITE_CASE_COUNT = 1
ECCENTRICITY_FIX_TOTAL_CASE_COUNT = 8
CONTRAST_MIN_LAMBDA = 1.0e-18
CONTRAST_ACCEPTANCE_LABEL = "contrast_lambda_1e-18"
OBJECTIVE_GROUPS = ("no_mask", "land_sea", "selected_countries", "fixed_outer")
ECCENTRICITY_OBJECTIVE_GROUPS = ("no_mask", "land_sea", "selected_countries")
OBJECTIVE_LABELS = {
    "no_mask": "No Mask",
    "land_sea": "Land/Sea Mask",
    "selected_countries": "Selected Countries",
    "fixed_outer": "Fixed Outer Regions",
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
GridNode = tuple[int, int]
GridPartition = list[GridNode]


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
class ContrastDesign:
    """Precomputed design arrays for contrast-score diagnostics."""

    weighted_contribution: np.ndarray
    cell_weight: np.ndarray
    grid_shape: tuple[int, int]
    min_contrast_lambda: float


@dataclass(frozen=True)
class BasisBuildContext:
    """Fields needed to rebuild one month/split basis case."""

    month: MonthSpec
    split: SplitSpec
    weights: xr.DataArray
    class_modes: dict[str, xr.DataArray]
    geometry: LatLonGridGeometry


@dataclass(frozen=True)
class CandidateLabels:
    """One generated basis-label field before scoring."""

    basis_family: str
    class_mode: str
    allocation: str
    split_step: str
    split_mode: str
    geometry: str
    outer_treatment: str
    split_acceptance: str
    contrast_min_lambda: float | None
    labels: xr.DataArray
    actual_regions: int
    split_history: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class Scenario:
    """One scored basis option for one CV split."""

    basis_family: str
    class_mode: str
    allocation: str
    split_step: str
    split_mode: str
    geometry: str
    outer_treatment: str
    split_acceptance: str
    contrast_min_lambda: float | None
    labels: xr.DataArray
    heldout_cv_nrmse: float
    heldout_cv_rmse: float
    heldout_cv_bias: float
    heldout_cv_corr: float
    actual_regions: int


@dataclass(frozen=True)
class EccentricityFixCase:
    """One current-vs-fixed inertial diagnostic case."""

    case_id: int
    month: str
    split_id: str
    class_mode: str
    allocation: str
    split_mode: str
    geometry: str
    current_worst_region: int
    current_labels: xr.DataArray
    fixed_labels: xr.DataArray
    current_summary: dict[str, Any]
    fixed_summary: dict[str, Any]


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

    (
        split_scores,
        split_history,
        region_diagnostics,
        representative_scenarios,
        representative_class_modes,
        representative_fields,
        basis_contexts,
    ) = (
        build_cross_validation_scores()
    )
    split_scores.to_csv(SPLIT_SCORES_PATH, index=False)
    split_history.to_csv(SPLIT_HISTORY_PATH, index=False)
    region_diagnostics.to_csv(REGION_DIAGNOSTICS_PATH, index=False)
    scores = aggregate_split_scores(split_scores)
    scores.to_csv(SCORES_PATH, index=False)
    overall_scores = aggregate_overall_scores(split_scores)
    overall_scores.to_csv(OVERALL_SCORES_PATH, index=False)

    class_figure = plot_region_classes(representative_class_modes)
    field_figure = plot_input_fields(representative_fields)
    basis_figure = plot_basis_contrasts(representative_scenarios, overall_scores)
    score_heatmap = plot_score_heatmap(scores)
    score_ranked = plot_ranked_scores(overall_scores)
    eccentricity_fix_cases = build_eccentricity_fix_cases(region_diagnostics, basis_contexts)
    eccentricity_fix_cases_table(eccentricity_fix_cases).to_csv(ECCENTRICITY_FIX_CASES_PATH, index=False)
    eccentricity_fix_figure = plot_eccentricity_fix_cases(eccentricity_fix_cases)

    write_report(
        class_figure=class_figure,
        field_figure=field_figure,
        basis_figure=basis_figure,
        eccentricity_fix_figure=eccentricity_fix_figure,
        score_heatmap=score_heatmap,
        score_ranked=score_ranked,
        scores=scores,
        split_scores=split_scores,
        split_history=split_history,
        overall_scores=overall_scores,
        region_diagnostics=region_diagnostics,
        eccentricity_fix_cases=eccentricity_fix_cases,
    )


def build_cross_validation_scores() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[Scenario],
    dict[str, xr.DataArray],
    RepresentativeFields,
    list[BasisBuildContext],
]:
    """Build split-level scores for all sites, months, and basis candidates."""
    records: list[dict[str, Any]] = []
    split_history_records: list[dict[str, Any]] = []
    region_diagnostic_records: list[dict[str, Any]] = []
    representative_scenarios: list[Scenario] = []
    representative_class_modes: dict[str, xr.DataArray] | None = None
    representative_fields: RepresentativeFields | None = None
    basis_contexts: list[BasisBuildContext] = []
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
            contrast_design = build_contrast_design(
                footprints=[train_footprints[site.site] for site in SITES],
                flux=reference_inputs.flux,
                weights=weights,
            )
            geometry = LatLonGridGeometry.from_dataarray(weights)
            country = align_to_test_grid(reference=weights, target=reference_inputs.country, target_name="country")
            class_modes = build_region_class_modes(country, weights, reference_inputs.country_names)
            basis_contexts.append(
                BasisBuildContext(
                    month=month,
                    split=split,
                    weights=weights,
                    class_modes=class_modes,
                    geometry=geometry,
                )
            )
            candidate_labels = build_candidate_labels(
                weights=weights,
                class_modes=class_modes,
                geometry=geometry,
                contrast_design=contrast_design,
            )
            for candidate in candidate_labels:
                split_history_records.extend(
                    candidate_split_history_records(
                        candidate,
                        month=month,
                        split=split,
                        basis_training_sites=basis_training_sites,
                    )
                )
                region_diagnostic_records.extend(
                    candidate_region_diagnostic_records(
                        candidate,
                        weights=weights,
                        month=month,
                        split=split,
                        basis_training_sites=basis_training_sites,
                    )
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

    return (
        pd.DataFrame.from_records(records),
        pd.DataFrame.from_records(split_history_records),
        pd.DataFrame.from_records(region_diagnostic_records),
        representative_scenarios,
        representative_class_modes,
        representative_fields,
        basis_contexts,
    )


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


def build_contrast_design(
    *,
    footprints: list[xr.DataArray],
    flux: xr.DataArray,
    weights: xr.DataArray,
) -> ContrastDesign:
    """Build training-only contrast-score arrays for one shared basis split.

    Rows are design footprint observations. Native-cell masses are the monthly
    prior flux field with a tiny positive floor; observed mole-fraction values
    are not used.
    """
    aligned_flux = align_to_test_grid(reference=weights, target=flux, target_name="contrast_cell_weight")
    cell_weight = aligned_flux.fillna(0.0)
    positive = np.asarray(cell_weight.values, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    floor = float(np.nanpercentile(positive, 0.1)) * 1.0e-6 if positive.size else 1.0
    cell_weight_values = np.asarray(cell_weight.clip(min=floor).values, dtype=np.float64)

    contribution_parts = [
        np.asarray(footprint.fillna(0.0).transpose("time", *weights.dims).values, dtype=np.float64)
        for footprint in footprints
    ]
    contribution = np.concatenate(contribution_parts, axis=0).reshape((-1, cell_weight_values.size))
    weighted_contribution = contribution * cell_weight_values.ravel().reshape(1, -1)
    return ContrastDesign(
        weighted_contribution=weighted_contribution,
        cell_weight=cell_weight_values.ravel(),
        grid_shape=cell_weight_values.shape,
        min_contrast_lambda=CONTRAST_MIN_LAMBDA,
    )


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
    contrast_design: ContrastDesign,
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
                        labels, split_history = build_constrained_basis_labels(
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
                                outer_treatment="full_domain",
                                split_acceptance="none",
                                contrast_min_lambda=None,
                                labels=labels,
                                actual_regions=count_regions(labels),
                                split_history=split_history,
                            )
                        )
                        if split_step_name == "axis_parallel":
                            contrast_policy = RecordingContrastSplitAcceptance(contrast_design)
                            labels, split_history = build_constrained_basis_labels(
                                weights=weights,
                                region_classes=region_classes,
                                allocation=allocation,
                                split_step_name=split_step_name,
                                split_mode=split_mode,
                                geometry=split_geometry,
                                split_acceptance=contrast_policy,
                            )
                            candidates.append(
                                CandidateLabels(
                                    basis_family="region_constrained",
                                    class_mode=class_mode,
                                    allocation=allocation_name,
                                    split_step=split_step_name,
                                    split_mode=split_mode,
                                    geometry=geometry_name,
                                    outer_treatment="full_domain",
                                    split_acceptance=CONTRAST_ACCEPTANCE_LABEL,
                                    contrast_min_lambda=CONTRAST_MIN_LAMBDA,
                                    labels=labels,
                                    actual_regions=count_regions(labels),
                                    split_history=split_history,
                                )
                            )

    candidates.extend(build_legacy_candidate_labels(weights))
    candidates.extend(build_fixed_outer_candidate_labels(weights=weights, landsea_classes=class_modes["land_sea"]))
    return candidates


def build_constrained_basis_labels(
    *,
    weights: xr.DataArray,
    region_classes: xr.DataArray,
    allocation: str,
    split_step_name: str,
    split_mode: str,
    geometry: LatLonGridGeometry | None,
    split_acceptance: Any | None = None,
) -> tuple[xr.DataArray, tuple[dict[str, Any], ...]]:
    """Build one 250-target constrained basis label map."""
    balanced = split_mode == "balanced"
    if split_step_name == "axis_parallel":
        split_step = AxisParallelSplitStep(balanced=balanced, clean_splits=True, geometry=geometry)
    elif split_step_name == "inertial":
        split_step = InertialSplitStep(balanced=balanced, geometry=geometry)
    else:
        raise ValueError(f"Unknown split step {split_step_name!r}.")

    labels, split_history = recording_region_constrained_basis(
        weights=weights,
        region_classes=region_classes,
        nbasis=TARGET_REGIONS,
        allocation=cast(AllocationMode, allocation),
        split_step=split_step,
        split_acceptance=split_acceptance,
    )
    return labels.astype(np.int32).rename("basis"), tuple(split_history)


def recording_region_constrained_basis(
    *,
    weights: xr.DataArray,
    region_classes: xr.DataArray,
    nbasis: int,
    allocation: AllocationMode,
    split_step: Any,
    split_acceptance: Any | None = None,
) -> tuple[xr.DataArray, list[dict[str, Any]]]:
    """Generate constrained labels while recording accepted greedy splits."""
    weights, region_classes = align_recording_inputs(weights=weights, region_classes=region_classes)
    weight_values = validated_weight_values(weights)
    class_values = region_classes.to_numpy()
    mapped_classes = mapped_class_values(class_values)
    labels = np.zeros(weight_values.shape, dtype=np.int64)
    history: list[dict[str, Any]] = []

    if not mapped_classes:
        return labels_dataarray(labels, weights), history

    targets = allocate_nbasis_by_class(
        weights,
        region_classes,
        nbasis,
        allocation=allocation,
    )

    next_label = 1
    for class_value in mapped_classes:
        target_regions = targets[class_value]
        class_mask = class_values == class_value
        class_weights = np.where(class_mask, weight_values, 0.0)
        class_weight_sum = float(class_weights.sum())
        if class_weight_sum == 0.0:
            class_weights = class_mask.astype(np.float64)
            class_weight_sum = float(class_weights.sum())

        nodes = node_list_from_mask(class_mask)
        partitions, class_history = recording_greedy_partition(
            nodes=nodes,
            target_regions=target_regions,
            weights=class_weights,
            split_step=split_step,
            split_acceptance=split_acceptance,
        )
        for record in class_history:
            record.update(
                {
                    "class_value": str(class_value),
                    "class_target_regions": int(target_regions),
                    "class_cells": int(np.count_nonzero(class_mask)),
                    "class_weight": class_weight_sum,
                }
            )
        history.extend(class_history)

        for nodes in partitions:
            rows, cols = node_indices(nodes)
            labels[rows, cols] = next_label
            next_label += 1

    return labels_dataarray(labels, weights), history


class RecordingContrastSplitAcceptance:
    """Fast contrast-score gate for this generator's diagnostic sweep."""

    def __init__(self, design: ContrastDesign) -> None:
        self.design = design
        self.last_diagnostics: dict[str, Any] = {}

    def __call__(
        self,
        parent: GridPartition,
        children: list[GridPartition],
        weights: np.ndarray,
    ) -> bool:
        """Return true when the proposed binary split exceeds the threshold."""
        del parent, weights
        if len(children) != 2:
            self.last_diagnostics = {}
            raise ValueError("RecordingContrastSplitAcceptance requires a binary split.")

        score = self.score_split(children[0], children[1])
        accepted = score["contrast_lambda"] >= self.design.min_contrast_lambda
        self.last_diagnostics = {
            **score,
            "contrast_min_lambda": self.design.min_contrast_lambda,
            "contrast_accepts": accepted,
            "contrast_uncalibrated": True,
        }
        return accepted

    def score_split(self, child_a: GridPartition, child_b: GridPartition) -> dict[str, float]:
        """Return mass-preserving split contrast diagnostics for two children."""
        index_a = flat_node_indices(child_a, self.design.grid_shape)
        index_b = flat_node_indices(child_b, self.design.grid_shape)
        weight_a = self.design.cell_weight[index_a]
        weight_b = self.design.cell_weight[index_b]
        mu_a = float(weight_a.sum())
        mu_b = float(weight_b.sum())
        mu_g = mu_a + mu_b
        if mu_a <= 0.0 or mu_b <= 0.0 or not np.isfinite(mu_g):
            raise ValueError("Contrast child masses must be positive and finite.")

        h_a = self.design.weighted_contribution[:, index_a].sum(axis=1)
        h_b = self.design.weighted_contribution[:, index_b].sum(axis=1)
        contrast = (mu_b / mu_g) * h_a - (mu_a / mu_g) * h_b
        lambda_value = float(np.sum(contrast**2))
        return {
            "contrast_lambda": lambda_value,
            "contrast_delta_dfs": float(lambda_value / (1.0 + lambda_value)),
            "contrast_delta_eig": float(0.5 * np.log1p(lambda_value)),
            "contrast_mu_a": mu_a,
            "contrast_mu_b": mu_b,
        }


def flat_node_indices(nodes: GridPartition, shape: tuple[int, int]) -> np.ndarray:
    """Return flat indices for grid-index nodes."""
    rows, cols = node_indices(nodes)
    return np.ravel_multi_index((np.asarray(rows), np.asarray(cols)), shape)


def align_recording_inputs(
    *,
    weights: xr.DataArray,
    region_classes: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Align generator inputs in the same shape expected by constrained basis code."""
    if weights.ndim != 2 or region_classes.ndim != 2:
        raise ValueError("weights and region_classes must be two-dimensional.")
    if set(weights.dims) != set(region_classes.dims):
        raise ValueError("weights and region_classes must use the same dimensions.")
    region_classes = region_classes.transpose(*weights.dims)
    return xr.align(weights, region_classes, join="exact")


def validated_weight_values(weights: xr.DataArray) -> np.ndarray:
    """Return finite non-negative weights for recording diagnostics."""
    values = np.asarray(weights.to_numpy(), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite.")
    if (values < 0.0).any():
        raise ValueError("weights must be non-negative.")
    return values


def mapped_class_values(class_values: np.ndarray) -> list[Any]:
    """Return first-seen non-null class values."""
    classes: list[Any] = []
    for value in pd.unique(class_values.ravel()):
        if pd.isna(value):
            continue
        classes.append(value)
    return classes


def labels_dataarray(labels: np.ndarray, template: xr.DataArray) -> xr.DataArray:
    """Return a label DataArray aligned to the template weight field."""
    return xr.DataArray(labels, dims=template.dims, coords=template.coords, name="basis")


def recording_greedy_partition(
    *,
    nodes: GridPartition,
    target_regions: int,
    weights: np.ndarray,
    split_step: Any,
    split_acceptance: Any | None = None,
) -> tuple[list[GridPartition], list[dict[str, Any]]]:
    """Apply the production greedy queue pattern while recording split attempts."""
    active: list[tuple[tuple[float, int, int], int, int, GridPartition]] = []
    done: list[GridPartition] = []
    history: list[dict[str, Any]] = []
    next_partition_id = 1
    queue_counter = 0
    split_event = 0
    current_regions = 0

    def push_active(partition_nodes: GridPartition, *, partition_id: int, depth: int) -> None:
        nonlocal queue_counter
        if not partition_nodes:
            return
        priority = (-node_weight(partition_nodes, weights), -len(partition_nodes), queue_counter)
        queue_counter += 1
        heappush(active, (priority, partition_id, depth, partition_nodes))

    if nodes:
        current_regions = 1
        if len(nodes) > 1:
            push_active(nodes, partition_id=next_partition_id, depth=0)
        else:
            done.append(nodes)
        next_partition_id += 1

    while current_regions < target_regions and active:
        _priority, parent_id, parent_depth, parent_nodes = heappop(active)
        child_partitions = [child for child in split_step(parent_nodes, weights) if child]
        split_event += 1

        accepted = True
        rejection_reason = ""
        split_diagnostics: dict[str, Any] = {}
        if len(child_partitions) < 2:
            accepted = False
            rejection_reason = "unsplittable"
        elif current_regions - 1 + len(child_partitions) > target_regions:
            accepted = False
            rejection_reason = "target_exceeded"
        elif split_acceptance is not None:
            accepted = split_acceptance_allows(
                split_acceptance,
                parent_nodes,
                child_partitions,
                weights,
                target_regions,
            )
            split_diagnostics = split_acceptance_diagnostics(split_acceptance)
            if not accepted:
                rejection_reason = "split_acceptance"

        if not accepted:
            history.extend(
                split_history_rows(
                    split_event=split_event,
                    accepted=False,
                    rejection_reason=rejection_reason,
                    target_regions=target_regions,
                    current_regions_before=current_regions,
                    parent_id=parent_id,
                    parent_depth=parent_depth,
                    parent_nodes=parent_nodes,
                    child_entries=[(None, child) for child in child_partitions],
                    weights=weights,
                    split_diagnostics=split_diagnostics,
                )
            )
            done.append(parent_nodes)
            continue

        child_entries: list[tuple[int | None, GridPartition]] = []
        for child in child_partitions:
            child_entries.append((next_partition_id, child))
            next_partition_id += 1

        history.extend(
            split_history_rows(
                split_event=split_event,
                accepted=True,
                rejection_reason="",
                target_regions=target_regions,
                current_regions_before=current_regions,
                parent_id=parent_id,
                parent_depth=parent_depth,
                parent_nodes=parent_nodes,
                child_entries=child_entries,
                weights=weights,
                split_diagnostics=split_diagnostics,
            )
        )

        current_regions -= 1
        for child_id, child_nodes in child_entries:
            current_regions += 1
            if len(child_nodes) > 1:
                push_active(child_nodes, partition_id=cast(int, child_id), depth=parent_depth + 1)
            else:
                done.append(child_nodes)

    while active:
        _priority, _partition_id, _depth, active_nodes = heappop(active)
        done.append(active_nodes)

    return done, history


def split_history_rows(
    *,
    split_event: int,
    accepted: bool,
    rejection_reason: str,
    target_regions: int,
    current_regions_before: int,
    parent_id: int,
    parent_depth: int,
    parent_nodes: GridPartition,
    child_entries: list[tuple[int | None, GridPartition]],
    weights: np.ndarray,
    split_diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return one split-history row per proposed child partition."""
    if not child_entries:
        child_entries = [(None, [])]
    split_diagnostics = {} if split_diagnostics is None else split_diagnostics
    parent_metrics = partition_shape_metrics(parent_nodes, weights=weights, prefix="parent")
    parent_weight = float(parent_metrics["parent_weight"])
    rows: list[dict[str, Any]] = []
    for child_index, (child_id, child_nodes) in enumerate(child_entries):
        child_metrics = partition_shape_metrics(child_nodes, weights=weights, prefix="child")
        child_weight = float(child_metrics["child_weight"])
        rows.append(
            {
                "split_event": split_event,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "target_regions": target_regions,
                "current_regions_before": current_regions_before,
                "parent_partition_id": parent_id,
                "child_partition_id": child_id,
                "child_index": child_index,
                "parent_depth": parent_depth,
                "child_depth": parent_depth + 1 if accepted else parent_depth,
                "child_weight_share_of_parent": child_weight / parent_weight if parent_weight > 0.0 else np.nan,
                **split_diagnostics,
                **parent_metrics,
                **child_metrics,
            }
        )
    return rows


def candidate_split_history_records(
    candidate: CandidateLabels,
    *,
    month: MonthSpec,
    split: SplitSpec,
    basis_training_sites: str,
) -> list[dict[str, Any]]:
    """Return split-history rows with candidate and CV context fields."""
    rows: list[dict[str, Any]] = []
    for record in candidate.split_history:
        rows.append(
            {
                **basis_context_record(month=month, split=split, basis_training_sites=basis_training_sites),
                **candidate_key_record(candidate),
                **record,
            }
        )
    return rows


def candidate_region_diagnostic_records(
    candidate: CandidateLabels,
    *,
    weights: xr.DataArray,
    month: MonthSpec,
    split: SplitSpec,
    basis_training_sites: str,
) -> list[dict[str, Any]]:
    """Return final-region shape diagnostics for one candidate basis."""
    labels = align_to_test_grid(reference=weights, target=candidate.labels, target_name="basis")
    label_values = np.asarray(labels.values, dtype=np.int64)
    weight_values = np.asarray(weights.values, dtype=np.float64)
    lat_values = np.asarray(labels.coords[labels.dims[0]].values, dtype=np.float64)
    lon_values = np.asarray(labels.coords[labels.dims[1]].values, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for region_label in np.unique(label_values[label_values > 0]):
        nodes = node_list_from_mask(label_values == region_label)
        rows.append(
            {
                **basis_context_record(month=month, split=split, basis_training_sites=basis_training_sites),
                **candidate_key_record(candidate),
                "region_label": int(region_label),
                **partition_shape_metrics(
                    nodes,
                    weights=weight_values,
                    prefix="region",
                    lat_values=lat_values,
                    lon_values=lon_values,
                ),
            }
        )
    return rows


def build_eccentricity_fix_cases(
    region_diagnostics: pd.DataFrame,
    basis_contexts: list[BasisBuildContext],
) -> list[EccentricityFixCase]:
    """Rebuild the worst inertial cases with the eccentricity split guard enabled."""
    context_by_key = {(context.month.label, context.split.split_id): context for context in basis_contexts}
    cases: list[EccentricityFixCase] = []
    for case_id, row in enumerate(worst_eccentricity_case_rows(region_diagnostics), start=1):
        context = context_by_key[(str(row["month"]), str(row["split_id"]))]
        class_mode = str(row["class_mode"])
        allocation_name = str(row["allocation"])
        allocation = "weight" if allocation_name == "single_class" else allocation_name
        split_mode = str(row["split_mode"])
        geometry_name = str(row["geometry"])
        split_geometry = context.geometry if geometry_name == "lat_lon_metres" else None
        region_classes = context.class_modes[class_mode]

        current_labels, _current_history = build_constrained_basis_labels(
            weights=context.weights,
            region_classes=region_classes,
            allocation=allocation,
            split_step_name="inertial",
            split_mode=split_mode,
            geometry=split_geometry,
        )
        fixed_labels, _fixed_history = build_constrained_basis_labels(
            weights=context.weights,
            region_classes=region_classes,
            allocation=allocation,
            split_step_name="inertial",
            split_mode=split_mode,
            geometry=split_geometry,
            split_acceptance=MaxChildPCAEccentricity(
                max_child_pca_eccentricity=ECCENTRICITY_FIX_THRESHOLD,
                geometry=split_geometry,
            ),
        )
        current_summary = label_shape_summary(current_labels, context.weights)
        fixed_summary = label_shape_summary(fixed_labels, context.weights)
        cases.append(
            EccentricityFixCase(
                case_id=case_id,
                month=context.month.label,
                split_id=context.split.split_id,
                class_mode=class_mode,
                allocation=allocation_name,
                split_mode=split_mode,
                geometry=geometry_name,
                current_worst_region=int(row["region_label"]),
                current_labels=current_labels,
                fixed_labels=fixed_labels,
                current_summary=current_summary,
                fixed_summary=fixed_summary,
            )
        )
    return cases


def worst_eccentricity_case_rows(region_diagnostics: pd.DataFrame) -> list[dict[str, Any]]:
    """Return objective-balanced current inertial settings with the worst region shapes."""
    constrained_inertial = region_diagnostics.loc[
        (region_diagnostics["basis_family"] == "region_constrained")
        & (region_diagnostics["split_step"] == "inertial")
    ].copy()
    case_columns = ["month", "split_id", "class_mode", "allocation", "split_mode", "geometry"]
    selected_rows: list[pd.Series] = []
    selected_keys: set[tuple[Any, ...]] = set()

    infinite = constrained_inertial.loc[np.isinf(constrained_inertial["region_pca_eccentricity"])].sort_values(
        ["region_bbox_aspect_ratio", "region_connected_components", "region_cells"],
        ascending=[False, False, False],
    )
    finite = constrained_inertial.loc[np.isfinite(constrained_inertial["region_pca_eccentricity"])].sort_values(
        ["region_pca_eccentricity", "region_connected_components", "region_bbox_aspect_ratio"],
        ascending=[False, False, False],
    )

    def add_case_rows(rows: pd.DataFrame, max_new_rows: int) -> None:
        for _, candidate in rows.iterrows():
            if max_new_rows <= 0:
                break
            key = tuple(candidate[column] for column in case_columns)
            if key in selected_keys:
                continue
            selected_rows.append(candidate)
            selected_keys.add(key)
            max_new_rows -= 1

    for objective in ECCENTRICITY_OBJECTIVE_GROUPS:
        add_case_rows(
            infinite.loc[infinite["class_mode"] == objective].drop_duplicates(case_columns),
            ECCENTRICITY_FIX_PER_OBJECTIVE_INFINITE_CASE_COUNT,
        )
    for objective in ECCENTRICITY_OBJECTIVE_GROUPS:
        add_case_rows(
            finite.loc[finite["class_mode"] == objective].drop_duplicates(case_columns),
            ECCENTRICITY_FIX_PER_OBJECTIVE_FINITE_CASE_COUNT,
        )

    remaining_cases = ECCENTRICITY_FIX_TOTAL_CASE_COUNT - len(selected_rows)
    if remaining_cases > 0:
        add_case_rows(infinite.drop_duplicates(case_columns), remaining_cases)
    remaining_cases = ECCENTRICITY_FIX_TOTAL_CASE_COUNT - len(selected_rows)
    if remaining_cases > 0:
        add_case_rows(finite.drop_duplicates(case_columns), remaining_cases)

    selected = pd.DataFrame(selected_rows)
    return dataframe_records(selected)


def label_shape_summary(labels: xr.DataArray, weights: xr.DataArray) -> dict[str, Any]:
    """Summarize final-region shape diagnostics for one label field."""
    label_values = np.asarray(labels.values, dtype=np.int64)
    weight_values = np.asarray(weights.values, dtype=np.float64)
    rows = [
        partition_shape_metrics(
            node_list_from_mask(label_values == region_label),
            weights=weight_values,
            prefix="region",
        )
        for region_label in np.unique(label_values[label_values > 0])
    ]
    metrics = pd.DataFrame.from_records(rows)
    finite_eccentricity = metrics["region_pca_eccentricity"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "actual_regions": count_regions(labels),
        "max_pca_eccentricity": float(metrics["region_pca_eccentricity"].max()),
        "max_finite_pca_eccentricity": float(finite_eccentricity.max()) if not finite_eccentricity.empty else np.nan,
        "infinite_pca_eccentricity_regions": int(np.isinf(metrics["region_pca_eccentricity"]).sum()),
        "max_bbox_aspect_ratio": float(metrics["region_bbox_aspect_ratio"].max()),
        "multi_component_regions": int((metrics["region_connected_components"] > 1).sum()),
    }


def eccentricity_fix_cases_table(cases: list[EccentricityFixCase]) -> pd.DataFrame:
    """Return one compact row per current-vs-fixed eccentricity case."""
    records: list[dict[str, Any]] = []
    for case in cases:
        records.append(
            {
                "case_id": case.case_id,
                "month": case.month,
                "split_id": case.split_id,
                "class_mode": case.class_mode,
                "allocation": case.allocation,
                "split_step": "inertial",
                "split_mode": case.split_mode,
                "geometry": case.geometry,
                "eccentricity_fix_threshold": ECCENTRICITY_FIX_THRESHOLD,
                "current_worst_region": case.current_worst_region,
                **prefixed_summary(case.current_summary, "current"),
                **prefixed_summary(case.fixed_summary, "fixed"),
            }
        )
    return pd.DataFrame.from_records(records)


def prefixed_summary(summary: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Prefix summary keys for tabular current-vs-fixed output."""
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def split_acceptance_allows(
    policy: Any,
    parent: GridPartition,
    children: list[GridPartition],
    weights: np.ndarray,
    target_regions: int | None = None,
) -> bool:
    """Return true when a generator split acceptance policy accepts children."""
    if target_regions is not None:
        accept_split = getattr(policy, "accept_split", None)
        if accept_split is not None:
            return bool(accept_split(parent, children, weights, target_regions))
    return bool(policy(parent, children, weights))


def split_acceptance_diagnostics(policy: Any) -> dict[str, Any]:
    """Return diagnostics recorded by a split-acceptance policy, if present."""
    diagnostics = getattr(policy, "last_diagnostics", None)
    if diagnostics is None:
        return {}
    return dict(diagnostics)


def basis_context_record(
    *,
    month: MonthSpec,
    split: SplitSpec,
    basis_training_sites: str,
) -> dict[str, Any]:
    """Return context fields common to split and region diagnostics."""
    return {
        "month": month.label,
        "split_id": split.split_id,
        "holdout_start": split.holdout_start.date().isoformat(),
        "holdout_end": (split.holdout_end - pd.Timedelta(days=1)).date().isoformat(),
        "basis_training_sites": basis_training_sites,
    }


def candidate_key_record(candidate: CandidateLabels) -> dict[str, Any]:
    """Return identifying fields for one candidate."""
    return {
        "basis_family": candidate.basis_family,
        "class_mode": candidate.class_mode,
        "allocation": candidate.allocation,
        "split_step": candidate.split_step,
        "split_mode": candidate.split_mode,
        "geometry": candidate.geometry,
        "outer_treatment": candidate.outer_treatment,
        "split_acceptance": candidate.split_acceptance,
        "contrast_min_lambda": candidate.contrast_min_lambda,
        "candidate": candidate_display_label(
            {
                "basis_family": candidate.basis_family,
                "allocation": candidate.allocation,
                "split_step": candidate.split_step,
                "split_mode": candidate.split_mode,
                "geometry": candidate.geometry,
                "outer_treatment": candidate.outer_treatment,
                "split_acceptance": candidate.split_acceptance,
            }
        ),
        "target_regions": TARGET_REGIONS,
        "actual_regions": candidate.actual_regions,
    }


def partition_shape_metrics(
    nodes: GridPartition,
    *,
    weights: np.ndarray,
    prefix: str,
    lat_values: np.ndarray | None = None,
    lon_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return compactness/eccentricity diagnostics for a grid-node partition."""
    if not nodes:
        return {
            f"{prefix}_cells": 0,
            f"{prefix}_weight": 0.0,
            f"{prefix}_row_min": np.nan,
            f"{prefix}_row_max": np.nan,
            f"{prefix}_col_min": np.nan,
            f"{prefix}_col_max": np.nan,
            f"{prefix}_bbox_height_cells": 0,
            f"{prefix}_bbox_width_cells": 0,
            f"{prefix}_bbox_area_cells": 0,
            f"{prefix}_bbox_aspect_ratio": np.nan,
            f"{prefix}_bbox_fill_fraction": np.nan,
            f"{prefix}_connected_components": 0,
            f"{prefix}_largest_component_fraction": np.nan,
            f"{prefix}_perimeter_edges": 0,
            f"{prefix}_grid_compactness": np.nan,
            f"{prefix}_pca_major_variance": np.nan,
            f"{prefix}_pca_minor_variance": np.nan,
            f"{prefix}_pca_eccentricity": np.nan,
        }

    rows, cols = node_indices(nodes)
    row_min = int(min(rows))
    row_max = int(max(rows))
    col_min = int(min(cols))
    col_max = int(max(cols))
    height = row_max - row_min + 1
    width = col_max - col_min + 1
    bbox_area = height * width
    perimeter = grid_perimeter(nodes)
    component_count, largest_component = connected_component_summary(nodes)
    pca_major, pca_minor, pca_eccentricity = pca_shape_summary(nodes)

    metrics: dict[str, Any] = {
        f"{prefix}_cells": len(nodes),
        f"{prefix}_weight": float(weights[rows, cols].sum()),
        f"{prefix}_row_min": row_min,
        f"{prefix}_row_max": row_max,
        f"{prefix}_col_min": col_min,
        f"{prefix}_col_max": col_max,
        f"{prefix}_bbox_height_cells": height,
        f"{prefix}_bbox_width_cells": width,
        f"{prefix}_bbox_area_cells": bbox_area,
        f"{prefix}_bbox_aspect_ratio": max(height, width) / max(1, min(height, width)),
        f"{prefix}_bbox_fill_fraction": len(nodes) / bbox_area if bbox_area > 0 else np.nan,
        f"{prefix}_connected_components": component_count,
        f"{prefix}_largest_component_fraction": largest_component / len(nodes),
        f"{prefix}_perimeter_edges": perimeter,
        f"{prefix}_grid_compactness": (4.0 * np.pi * len(nodes) / perimeter**2) if perimeter > 0 else np.nan,
        f"{prefix}_pca_major_variance": pca_major,
        f"{prefix}_pca_minor_variance": pca_minor,
        f"{prefix}_pca_eccentricity": pca_eccentricity,
    }
    if lat_values is not None and lon_values is not None:
        metrics.update(
            {
                f"{prefix}_lat_min": float(np.min(lat_values[rows])),
                f"{prefix}_lat_max": float(np.max(lat_values[rows])),
                f"{prefix}_lon_min": float(np.min(lon_values[cols])),
                f"{prefix}_lon_max": float(np.max(lon_values[cols])),
            }
        )
    return metrics


def pca_shape_summary(nodes: GridPartition) -> tuple[float, float, float]:
    """Return unweighted PCA variance and eccentricity diagnostics for a partition."""
    if len(nodes) < 2:
        return np.nan, np.nan, np.nan
    coords = np.asarray(nodes, dtype=np.float64)
    centered = coords - coords.mean(axis=0)
    covariance = centered.T @ centered / len(nodes)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(eigenvalues).all():
        return np.nan, np.nan, np.nan
    minor = float(max(eigenvalues[0], 0.0))
    major = float(max(eigenvalues[-1], 0.0))
    if major == 0.0:
        eccentricity = np.nan
    elif minor <= 1.0e-12:
        eccentricity = np.inf
    else:
        eccentricity = float(np.sqrt(major / minor))
    return major, minor, eccentricity


def connected_component_summary(nodes: GridPartition) -> tuple[int, int]:
    """Return component count and largest 4-neighbour component size."""
    remaining = set(nodes)
    component_count = 0
    largest_component = 0
    while remaining:
        component_count += 1
        stack = [remaining.pop()]
        component_size = 0
        while stack:
            row, col = stack.pop()
            component_size += 1
            for drow, dcol in NEIGHBOUR_OFFSETS:
                neighbour = (row + drow, col + dcol)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
        largest_component = max(largest_component, component_size)
    return component_count, largest_component


def grid_perimeter(nodes: GridPartition) -> int:
    """Return the number of exposed 4-neighbour cell edges."""
    node_set = set(nodes)
    perimeter = 0
    for row, col in node_set:
        for drow, dcol in NEIGHBOUR_OFFSETS:
            if (row + drow, col + dcol) not in node_set:
                perimeter += 1
    return perimeter


def node_list_from_mask(mask: np.ndarray) -> GridPartition:
    """Return grid-index nodes selected by a Boolean mask."""
    return list(zip(*np.where(mask)))


def node_indices(nodes: GridPartition) -> tuple[list[int], list[int]]:
    """Split grid-index nodes into row and column index lists."""
    if not nodes:
        return [], []
    rows, cols = zip(*nodes)
    return list(rows), list(cols)


def node_weight(nodes: GridPartition, weights: np.ndarray) -> float:
    """Return total weight for one partition."""
    rows, cols = node_indices(nodes)
    return float(weights[rows, cols].sum())


def build_fixed_outer_candidate_labels(
    *,
    weights: xr.DataArray,
    landsea_classes: xr.DataArray,
) -> list[CandidateLabels]:
    """Build fixed-outer quadtree and weighted/bucket comparison labels."""
    outer_regions = load_outer_regions(reference=weights)
    inner_mask = outer_regions == int(outer_regions.max())
    row_slice, col_slice = inner_bbox_slices(inner_mask)
    inner_weights = weights.isel({weights.dims[0]: row_slice, weights.dims[1]: col_slice})
    inner_landsea = landsea_classes.isel({weights.dims[0]: row_slice, weights.dims[1]: col_slice})
    normalized_grid = np.asarray(inner_weights.fillna(0.0).values, dtype=np.float64)
    grid_max = float(np.max(normalized_grid))
    if grid_max <= 0.0:
        return []
    normalized_grid = normalized_grid / grid_max

    landsea_mask = np.asarray(inner_landsea.values == "land", dtype=np.int64)
    fixed_specs = (
        (
            "bucketbasisfunction",
            lambda: weighted_bucket_with_landsea_mask(
                normalized_grid,
                landsea_mask=landsea_mask,
                nregion=TARGET_REGIONS,
            ),
        ),
        (
            "quadtreebasisfunction",
            lambda: quadtree_algorithm(normalized_grid, nbasis=TARGET_REGIONS, seed=QUADTREE_SEED),
        ),
    )

    candidates: list[CandidateLabels] = []
    for basis_family, build_inner_labels in fixed_specs:
        try:
            raw_inner_labels = build_inner_labels()
        except Exception as exc:  # pragma: no cover - used by evidence-generation script only
            print(f"Skipping fixed outer {basis_family}: {exc}")
            continue
        inner_labels = xr.DataArray(
            np.asarray(raw_inner_labels, dtype=np.int32),
            dims=inner_weights.dims,
            coords=inner_weights.coords,
            name="basis",
        )
        labels = reinsert_fixed_outer_labels(
            outer_regions=outer_regions,
            inner_labels=inner_labels,
            row_slice=row_slice,
            col_slice=col_slice,
        )
        candidates.append(
            CandidateLabels(
                basis_family=basis_family,
                class_mode="fixed_outer",
                allocation="legacy",
                split_step=basis_family,
                split_mode="legacy",
                geometry="row_column",
                outer_treatment="fixed_outer",
                split_acceptance="none",
                contrast_min_lambda=None,
                labels=labels,
                actual_regions=count_regions(labels),
            )
        )
    return candidates


def load_outer_regions(*, reference: xr.DataArray) -> xr.DataArray:
    """Load fixed EUROPE outer regions aligned to the scoring grid."""
    path = ROOT / "openghg_inversions" / "basis" / f"outer_region_definition_{BLUE_PEBBLE_DOMAIN}.nc"
    outer_regions = xr.open_dataset(path).region.transpose(*reference.dims).load()
    return align_to_test_grid(reference=reference, target=outer_regions, target_name="outer_regions")


def inner_bbox_slices(inner_mask: xr.DataArray) -> tuple[slice, slice]:
    """Return bounding-box index slices for the fixed-outer inner region."""
    mask_values = np.asarray(inner_mask.values, dtype=bool)
    rows, cols = np.where(mask_values)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Fixed outer region file does not contain an inner region.")
    row_slice = slice(int(rows.min()), int(rows.max()) + 1)
    col_slice = slice(int(cols.min()), int(cols.max()) + 1)
    if not mask_values[row_slice, col_slice].all():
        raise ValueError("Fixed outer inner region is not rectangular; update the generator reinsert logic.")
    return row_slice, col_slice


def reinsert_fixed_outer_labels(
    *,
    outer_regions: xr.DataArray,
    inner_labels: xr.DataArray,
    row_slice: slice,
    col_slice: slice,
) -> xr.DataArray:
    """Insert inner labels into fixed outer regions following the production offset rule."""
    inner_index = int(outer_regions.max())
    values = np.asarray(outer_regions.values, dtype=np.int32).copy()
    values[row_slice, col_slice] = np.asarray(inner_labels.values, dtype=np.int32) + inner_index - 1
    values = values + 1
    return xr.DataArray(values, dims=outer_regions.dims, coords=outer_regions.coords, name="basis")


def weighted_bucket_with_landsea_mask(
    grid: np.ndarray,
    *,
    landsea_mask: np.ndarray,
    nregion: int,
    bucket: float = 1.0,
    tol: int = 1,
) -> np.ndarray:
    """Return weighted/bucket labels using a caller-aligned land/sea mask."""
    bucket_value = optimize_bucket_with_landsea_mask(
        grid,
        landsea_mask=landsea_mask,
        bucket=bucket,
        nregion=nregion,
        tol=tol,
    )
    return bucket_split_landsea_with_mask(grid, bucket_value, landsea_mask=landsea_mask)


def optimize_bucket_with_landsea_mask(
    grid: np.ndarray,
    *,
    landsea_mask: np.ndarray,
    bucket: float,
    nregion: int,
    tol: int,
) -> float:
    """Optimize bucket value for a weighted/bucket split with aligned land/sea mask."""
    current_bucket = bucket
    current_tol = tol
    for _ in range(10):
        for iteration in range(1000):
            current_nregion = int(
                np.max(bucket_split_landsea_with_mask(grid, current_bucket, landsea_mask=landsea_mask))
            )
            if nregion - current_tol <= current_nregion <= nregion + current_tol:
                print(
                    "optimize_bucket_with_landsea_mask found bucket value "
                    f"{current_bucket} after {iteration} iterations with tolerance {current_tol}."
                )
                return current_bucket
            if current_nregion < nregion + current_tol:
                current_bucket *= 0.995
            else:
                current_bucket *= 1.005
        current_tol += 1
    raise RuntimeError("Could not optimize fixed-outer weighted/bucket region count.")


def bucket_split_landsea_with_mask(
    grid: np.ndarray,
    bucket: float,
    *,
    landsea_mask: np.ndarray,
) -> np.ndarray:
    """Split weighted rectangles and then separate each rectangle by land/sea."""
    regions = bucket_value_split(grid, bucket)
    labels = np.zeros(shape=grid.shape, dtype=np.int32)
    for ymin, ymax, xmin, xmax in regions:
        sea_rows, sea_cols = np.where(landsea_mask[ymin:ymax, xmin:xmax] == 0)
        land_rows, land_cols = np.where(landsea_mask[ymin:ymax, xmin:xmax] == 1)
        label = int(np.max(labels))
        if len(sea_rows) > 0:
            label += 1
            labels[sea_rows + ymin, sea_cols + xmin] = label
        if len(land_rows) > 0:
            label += 1
            labels[land_rows + ymin, land_cols + xmin] = label
    return labels


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
                outer_treatment="full_domain",
                split_acceptance="none",
                contrast_min_lambda=None,
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
                outer_treatment=candidate.outer_treatment,
                split_acceptance=candidate.split_acceptance,
                contrast_min_lambda=candidate.contrast_min_lambda,
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
        "outer_treatment": scenario.outer_treatment,
        "split_acceptance": scenario.split_acceptance,
        "contrast_min_lambda": scenario.contrast_min_lambda,
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
        "outer_treatment",
        "split_acceptance",
        "contrast_min_lambda",
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
        "outer_treatment",
        "split_acceptance",
        "contrast_min_lambda",
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
        label = "weighted/bucket"
    elif basis_family == "quadtreebasisfunction":
        label = "quadtree"
    else:
        label = (
            f"{row['allocation']}/{row['split_step']}/"
            f"{row['split_mode']}/{row['geometry']}"
        )

    if str(row.get("split_acceptance", "none")) != "none":
        label = f"{label}/contrast"
    if str(row.get("outer_treatment", "full_domain")) == "fixed_outer":
        label = f"fixed_outer/{label}"
    return label


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
    fig, axes = plt.subplots(
        nrows=len(OBJECTIVE_GROUPS),
        ncols=4,
        figsize=(16, 3.5 * len(OBJECTIVE_GROUPS)),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes)
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
                str(row["outer_treatment"]),
                str(row["split_acceptance"]),
            )
            plot_basis_label_map(ax, scenario.labels)
            rank_label = "reference" if row["basis_family"] != "region_constrained" else f"rank {col_index + 1}"
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


def plot_eccentricity_fix_cases(cases: list[EccentricityFixCase]) -> str:
    """Plot current and eccentricity-guarded labels for the worst inertial cases."""
    if not cases:
        raise ValueError("At least one eccentricity fix case is required.")

    fig, axes = plt.subplots(
        nrows=len(cases),
        ncols=2,
        figsize=(11.5, 3.2 * len(cases)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(cases) == 1:
        axes = np.asarray([axes])

    for row_index, case in enumerate(cases):
        current_ax = axes[row_index, 0]
        fixed_ax = axes[row_index, 1]
        plot_basis_label_map(current_ax, case.current_labels)
        overlay_region_outline(current_ax, case.current_labels, case.current_worst_region)
        plot_basis_label_map(fixed_ax, case.fixed_labels)
        current_ax.set_title(
            f"{case_label(case)}\n"
            f"current: inf {case.current_summary['infinite_pca_eccentricity_regions']}, "
            f"max finite {format_float(case.current_summary['max_finite_pca_eccentricity'], digits=1)}, "
            f"regions {case.current_summary['actual_regions']}",
            fontsize=9,
        )
        fixed_ax.set_title(
            f"ecc <= {ECCENTRICITY_FIX_THRESHOLD:g}\n"
            f"fixed: inf {case.fixed_summary['infinite_pca_eccentricity_regions']}, "
            f"max finite {format_float(case.fixed_summary['max_finite_pca_eccentricity'], digits=1)}, "
            f"regions {case.fixed_summary['actual_regions']}",
            fontsize=9,
        )
        for ax in (current_ax, fixed_ax):
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")

    return save_figure(fig, "basis_option_inertial_eccentricity_fix_cases_250.png")


def overlay_region_outline(ax: plt.Axes, labels: xr.DataArray, region_label: int) -> None:
    """Outline one basis region on a label map."""
    mask = xr.where(labels == region_label, 1.0, 0.0)
    ax.contour(labels.lon, labels.lat, mask, levels=[0.5], colors="black", linewidths=0.8)


def case_label(case: EccentricityFixCase) -> str:
    """Return a compact label for an eccentricity diagnostic row."""
    return (
        f"{objective_label(case.class_mode)} {case.month} {case.split_id}\n"
        f"{case.allocation}/inertial/{case.split_mode}/{case.geometry}"
    )


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
            constrained = group.loc[
                (group["basis_family"] == "region_constrained")
                & (group["outer_treatment"] == "full_domain")
            ].head(3)
            legacy = group.loc[
                (group["basis_family"] == legacy_family)
                & (group["outer_treatment"] == "full_domain")
            ].head(1)
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
    fig, axes = plt.subplots(
        nrows=len(OBJECTIVE_GROUPS),
        ncols=1,
        figsize=(12, 4.3 * len(OBJECTIVE_GROUPS)),
        constrained_layout=True,
    )
    axes = np.asarray(axes).ravel()
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
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(OBJECTIVE_GROUPS),
        figsize=(5.0 * len(OBJECTIVE_GROUPS), 8),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).ravel()
    for ax, class_mode in zip(axes, OBJECTIVE_GROUPS, strict=True):
        group = scores.loc[scores["class_mode"] == class_mode].sort_values(metric).reset_index(drop=True)
        y = np.arange(len(group))
        ax.scatter(
            group[metric],
            y,
            c=[ranked_score_color(row) for row in dataframe_records(group)],
            s=34,
        )
        ax.set_yticks(y, labels=group["candidate"], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Overall held-out CV NRMSE")
        ax.set_title(objective_label(class_mode))
        ax.grid(True, axis="x", linewidth=0.4, alpha=0.4)
    legend_specs = {
        "region_constrained": "tab:blue",
        "contrast_score": "tab:red",
        "weighted/bucket": "tab:orange",
        "quadtree": "tab:green",
        "fixed_outer": "tab:purple",
    }
    for label, color in legend_specs.items():
        axes[-1].scatter([], [], color=color, label=label)
    axes[-1].legend(loc="lower right", fontsize=8)
    return save_figure(fig, "basis_option_ranked_scores_250.png")


def ranked_score_color(row: dict[str, Any]) -> str:
    """Return plot colour for one ranked-score candidate."""
    if str(row["outer_treatment"]) == "fixed_outer":
        return "tab:purple"
    if str(row["split_acceptance"]) != "none":
        return "tab:red"
    if str(row["basis_family"]) == "bucketbasisfunction":
        return "tab:orange"
    if str(row["basis_family"]) == "quadtreebasisfunction":
        return "tab:green"
    return "tab:blue"


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
    outer_treatment: str,
    split_acceptance: str,
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
            and scenario.outer_treatment == outer_treatment
            and scenario.split_acceptance == split_acceptance
        ):
            return scenario
    raise ValueError(
        "No scenario found for "
        f"{(basis_family, class_mode, allocation, split_step, split_mode, geometry, outer_treatment, split_acceptance)!r}."
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
    eccentricity_fix_figure: str,
    score_heatmap: str,
    score_ranked: str,
    scores: pd.DataFrame,
    split_scores: pd.DataFrame,
    split_history: pd.DataFrame,
    overall_scores: pd.DataFrame,
    region_diagnostics: pd.DataFrame,
    eccentricity_fix_cases: list[EccentricityFixCase],
) -> None:
    """Write the documentation page."""
    overall_rows = overall_score_rows(overall_scores, n=5)
    context_rows = context_score_rows(scores, n=2)
    narrow_rows = narrow_region_rows(region_diagnostics, n=8)
    contrast_rows = axis_contrast_comparison_rows(overall_scores, split_history)
    fixed_outer_rows = fixed_outer_comparison_rows(overall_scores)
    eccentricity_case_rows = eccentricity_fix_case_rows(eccentricity_fix_cases)
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
        "The comparison also includes legacy `bucketbasisfunction` and `quadtreebasisfunction` rows generated from the same training weights. Their actual region counts can differ from the 250 target. `bucketbasisfunction` is shown as `weighted/bucket`; in this codebase the `weighted_algorithm` alias uses the land/sea weighted bucket splitter, so it is grouped with the land/sea objective. `quadtreebasisfunction` is shown as `quadtree` and grouped with the no-mask objective. Fixed-outer rows keep the package EUROPE InTEM outer regions fixed and build the inner region with quadtree or weighted/bucket splitting.",
        "",
        "The important separation is that `weights` define contribution/importance, while `geometry` defines physical coordinates for split shape. Lat/lon geometry does not change contribution weights, class allocation, or posterior weighting. The Blue Pebble generator builds these weights with the same multi-site footprint-times-flux reduction used by the production `basis_functions_wrapper` path.",
        "",
        f"Axis-parallel contrast rows append `/contrast` to the candidate label. They use the mass-preserving contrast score with `tau=1`, identity design covariance, and `min_contrast_lambda={CONTRAST_MIN_LAMBDA:.1e}`. This is an uncalibrated ranking/debugging threshold, not a calibrated expected-information-gain value.",
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
        "| `fixed_outer` | package EUROPE InTEM outer regions are fixed; only the inner region is generated |",
        "| `full_domain` | candidate generated across the full EUROPE domain |",
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
        "| `contrast` | optional axis-parallel split gate using the mass-preserving contrast score |",
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
        "Masked constrained candidates use `weight` allocation only, so the generated evidence focuses on the allocation mode used for the current recommendation. Contrast-score diagnostics use only training footprints and prior flux/mass weights; they do not use observed mole fractions, residuals, or held-out footprints.",
        "",
        f"Per-score-site/month aggregate scores are written to `{SCORES_PATH.relative_to(ROOT)}`, split-level scores are written to `{SPLIT_SCORES_PATH.relative_to(ROOT)}`, and overall all-score-site/month/split scores are written to `{OVERALL_SCORES_PATH.relative_to(ROOT)}`. The split-level table contains {len(split_scores)} scored rows and includes `basis_training_sites`, `basis_train_observations`, and `score_site_holdout_observations` to make the shared-basis training set explicit.",
        "",
        f"Constrained split-history diagnostics are written to `{SPLIT_HISTORY_PATH.relative_to(ROOT)}`. Final-region shape diagnostics for all candidates are written to `{REGION_DIAGNOSTICS_PATH.relative_to(ROOT)}`; those include bounding-box aspect ratio, fill fraction, 4-neighbour connected-component counts, grid compactness, and PCA eccentricity.",
        "",
        f"Current-vs-eccentricity-guard case diagnostics are written to `{ECCENTRICITY_FIX_CASES_PATH.relative_to(ROOT)}`. The guarded cases use `MaxChildPCAEccentricity(max_child_pca_eccentricity={ECCENTRICITY_FIX_THRESHOLD:g})` as a split-stopping policy, so the requested 250 regions becomes an upper target.",
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
        "The map figure shows the best overall held-out CV candidates by objective, using one representative January basis split for display. No-mask and land/sea rows show the best three full-domain constrained candidates plus the matching full-domain legacy option. Selected-country has no legacy counterpart, so it shows the best four constrained candidates. Fixed-outer rows show the available fixed-outer reference candidates.",
        "",
        f"![Basis option contrasts]({basis_figure})",
        "",
        "### Narrow-Region Diagnostics",
        "",
        "The table below lists the highest-eccentricity final regions among constrained inertial candidates. These diagnostics are not ranking scores; they are included to trace the narrow regions visible in the masked inertial maps.",
        "",
        "| objective | month | split | candidate | region | cells | bbox aspect | fill | components | PCA ecc. | compactness |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *narrow_rows,
        "",
        "### Eccentricity-Guarded Diagnostic Cases",
        "",
        "The figure below rebuilds an objective-balanced set of worst current inertial settings with the same month, split, objective, split mode, and geometry. Each objective first contributes the setting containing its worst infinite-eccentricity region and the setting containing its worst finite-eccentricity region; those setting-level summaries may still include other infinite-eccentricity regions. The remaining rows are filled by the global worst distinct settings. The left column is the current algorithm, with the worst current region outlined. The right column adds `MaxChildPCAEccentricity`; it rejects proposed child partitions whose PCA eccentricity is infinite or above the threshold.",
        "",
        f"![Inertial eccentricity fix cases]({eccentricity_fix_figure})",
        "",
        "| case | objective | month | split | option | current regions | fixed regions | current inf ecc | fixed inf ecc | current max finite ecc | fixed max finite ecc | current multi-comp | fixed multi-comp |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *eccentricity_case_rows,
        "",
        "## Grouped Scores For 250-Region Options",
        "",
        "The heatmap shows the mean split score for each held-out site/month context, grouped by objective. The ranked plot uses the overall score averaged over all TAC/MHD January/July split rows for each candidate.",
        "",
        f"![Score heatmaps]({score_heatmap})",
        "",
        f"![Ranked scores]({score_ranked})",
        "",
        "### Axis-Parallel Contrast Gate Diagnostics",
        "",
        "The table below pairs each full-domain axis-parallel baseline with its contrast-gated counterpart. The contrast score uses `tau=1` and identity design covariance, so `lambda` and `delta_eig` are useful here only as uncalibrated split-ranking quantities.",
        "",
        "| objective | option | baseline regions | contrast regions | rejected splits | baseline CV NRMSE | contrast CV NRMSE | delta NRMSE | median lambda |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *contrast_rows,
        "",
        "### Fixed-Outer Diagnostics",
        "",
        "The fixed-outer rows hold the package EUROPE outer regions fixed and build the inner region with the listed legacy splitter. The weighted/bucket fixed-inner diagnostic uses a cropped land/sea mask so land/sea separation remains aligned after cropping.",
        "",
        "| fixed candidate | full-domain comparator | fixed regions | full regions | fixed CV NRMSE | full CV NRMSE | delta NRMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
        *fixed_outer_rows,
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
        "- The objective groups should not be read as one single efficiency race. No mask, land/sea, selected-country masks, and fixed outer regions are often chosen for scientific or reporting reasons as well as basis efficiency.",
        "- Split-stopping policies can return fewer than 250 actual regions. Contrast rows should therefore be read with their actual region counts and rejection counts, not just their nominal target.",
        "",
        "## What This Does Not Prove",
        "",
        "These scores do not show whether an inversion posterior improves. For that, use a posterior or posterior-equivalent test: prior/error-weighted `H`, observation-error weighting, linear-Gaussian posterior covariance and resolution, synthetic truth recovery, or paired HPC-CI posterior runs.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def axis_contrast_comparison_rows(overall_scores: pd.DataFrame, split_history: pd.DataFrame) -> list[str]:
    """Format paired axis-parallel baseline vs contrast-gated score rows."""
    rows: list[str] = []
    axis_rows = overall_scores.loc[
        (overall_scores["basis_family"] == "region_constrained")
        & (overall_scores["split_step"] == "axis_parallel")
        & (overall_scores["outer_treatment"] == "full_domain")
    ]
    baseline_rows = axis_rows.loc[axis_rows["split_acceptance"] == "none"]
    contrast_rows = axis_rows.loc[axis_rows["split_acceptance"] == CONTRAST_ACCEPTANCE_LABEL]
    event_counts = contrast_split_event_counts(split_history)
    key_columns = ["class_mode", "allocation", "split_mode", "geometry"]
    for contrast_row in dataframe_records(contrast_rows.sort_values(["class_mode", "split_mode", "geometry"])):
        baseline = baseline_rows
        for column in key_columns:
            baseline = baseline.loc[baseline[column] == contrast_row[column]]
        if baseline.empty:
            continue
        baseline_row = dataframe_records(baseline.head(1))[0]
        event_key = tuple(contrast_row[column] for column in key_columns)
        event_summary = event_counts.get(event_key, {})
        rejected = int(event_summary.get("rejected", 0))
        median_lambda = event_summary.get("median_lambda", np.nan)
        delta_nrmse = float(contrast_row["heldout_cv_nrmse"]) - float(baseline_row["heldout_cv_nrmse"])
        rows.append(
            f"| {objective_label(str(contrast_row['class_mode']))} | "
            f"{contrast_row['allocation']}/axis_parallel/{contrast_row['split_mode']}/{contrast_row['geometry']} | "
            f"{baseline_row['actual_regions']:.1f} | {contrast_row['actual_regions']:.1f} | {rejected} | "
            f"{format_float(baseline_row['heldout_cv_nrmse'])} | "
            f"{format_float(contrast_row['heldout_cv_nrmse'])} | {format_float(delta_nrmse)} | "
            f"{format_float(median_lambda, digits=3)} |"
        )
    return rows or ["| _none_ |  |  |  |  |  |  |  |  |"]


def contrast_split_event_counts(split_history: pd.DataFrame) -> dict[tuple[Any, ...], dict[str, float]]:
    """Return accepted/rejected event counts and median score for contrast candidates."""
    if split_history.empty or "split_acceptance" not in split_history:
        return {}
    contrast = split_history.loc[split_history["split_acceptance"] == CONTRAST_ACCEPTANCE_LABEL].copy()
    if contrast.empty:
        return {}
    event_columns = [
        "month",
        "split_id",
        "basis_training_sites",
        "class_mode",
        "allocation",
        "split_mode",
        "geometry",
        "split_event",
        "accepted",
        "contrast_lambda",
    ]
    events = contrast[event_columns].drop_duplicates()
    grouped = events.groupby(["class_mode", "allocation", "split_mode", "geometry"], sort=False, dropna=False)
    summary: dict[tuple[Any, ...], dict[str, float]] = {}
    for key, group_obj in grouped:
        group = cast(pd.DataFrame, group_obj)
        summary[cast(tuple[Any, ...], key)] = {
            "accepted": float(group["accepted"].sum()),
            "rejected": float((~group["accepted"].astype(bool)).sum()),
            "median_lambda": float(group["contrast_lambda"].median()),
        }
    return summary


def fixed_outer_comparison_rows(overall_scores: pd.DataFrame) -> list[str]:
    """Format fixed-outer score rows with full-domain comparator deltas."""
    rows: list[str] = []
    fixed_rows = overall_scores.loc[overall_scores["outer_treatment"] == "fixed_outer"].sort_values(
        "heldout_cv_nrmse"
    )
    comparator_class = {
        "bucketbasisfunction": "land_sea",
        "quadtreebasisfunction": "no_mask",
    }
    for fixed_row in dataframe_records(fixed_rows):
        class_mode = comparator_class.get(str(fixed_row["basis_family"]))
        comparator = overall_scores.loc[
            (overall_scores["basis_family"] == fixed_row["basis_family"])
            & (overall_scores["class_mode"] == class_mode)
            & (overall_scores["outer_treatment"] == "full_domain")
        ]
        if comparator.empty:
            continue
        comparator_row = dataframe_records(comparator.head(1))[0]
        delta_nrmse = float(fixed_row["heldout_cv_nrmse"]) - float(comparator_row["heldout_cv_nrmse"])
        rows.append(
            f"| {fixed_row['candidate']} | {objective_label(str(comparator_row['class_mode']))} "
            f"{comparator_row['candidate']} | {fixed_row['actual_regions']:.1f} | "
            f"{comparator_row['actual_regions']:.1f} | {format_float(fixed_row['heldout_cv_nrmse'])} | "
            f"{format_float(comparator_row['heldout_cv_nrmse'])} | {format_float(delta_nrmse)} |"
        )
    return rows or ["| _none_ |  |  |  |  |  |  |"]


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


def narrow_region_rows(region_diagnostics: pd.DataFrame, *, n: int) -> list[str]:
    """Format the highest-eccentricity inertial region diagnostics."""
    if region_diagnostics.empty:
        return ["| _none_ |  |  |  |  |  |  |  |  |  |  |"]

    constrained_inertial = region_diagnostics.loc[
        (region_diagnostics["basis_family"] == "region_constrained")
        & (region_diagnostics["split_step"] == "inertial")
    ].copy()
    if constrained_inertial.empty:
        return ["| _none_ |  |  |  |  |  |  |  |  |  |  |"]

    constrained_inertial["sort_eccentricity"] = constrained_inertial["region_pca_eccentricity"].replace(
        np.inf,
        1.0e12,
    )
    constrained_inertial = constrained_inertial.sort_values(
        ["sort_eccentricity", "region_bbox_aspect_ratio", "region_connected_components"],
        ascending=[False, False, False],
    ).head(n)

    rows: list[str] = []
    for row in dataframe_records(constrained_inertial):
        rows.append(
            f"| {objective_label(str(row['class_mode']))} | {row['month']} | {row['split_id']} | "
            f"{row['candidate']} | {row['region_label']} | {row['region_cells']} | "
            f"{format_float(row['region_bbox_aspect_ratio'], digits=2)} | "
            f"{format_float(row['region_bbox_fill_fraction'], digits=2)} | "
            f"{row['region_connected_components']} | "
            f"{format_float(row['region_pca_eccentricity'], digits=2)} | "
            f"{format_float(row['region_grid_compactness'], digits=3)} |"
        )
    return rows


def eccentricity_fix_case_rows(cases: list[EccentricityFixCase]) -> list[str]:
    """Format current-vs-fixed diagnostic case rows for markdown."""
    rows: list[str] = []
    for case in cases:
        rows.append(
            f"| {case.case_id} | {objective_label(case.class_mode)} | {case.month} | {case.split_id} | "
            f"{case.allocation}/inertial/{case.split_mode}/{case.geometry} | "
            f"{case.current_summary['actual_regions']} | {case.fixed_summary['actual_regions']} | "
            f"{case.current_summary['infinite_pca_eccentricity_regions']} | "
            f"{case.fixed_summary['infinite_pca_eccentricity_regions']} | "
            f"{format_float(case.current_summary['max_finite_pca_eccentricity'], digits=1)} | "
            f"{format_float(case.fixed_summary['max_finite_pca_eccentricity'], digits=1)} | "
            f"{case.current_summary['multi_component_regions']} | "
            f"{case.fixed_summary['multi_component_regions']} |"
        )
    return rows


def format_float(value: Any, *, digits: int = 4) -> str:
    """Format a numeric score for markdown."""
    numeric = float(value)
    if np.isnan(numeric):
        return "nan"
    if np.isinf(numeric):
        return "inf" if numeric > 0.0 else "-inf"
    if numeric != 0.0 and abs(numeric) < 10**-digits:
        return f"{numeric:.3e}"
    return f"{numeric:.{digits}f}"


if __name__ == "__main__":
    main()
