"""Nested-domain preparation and execution for modern RHIME.

Nested domains are independent spatial resolutions of the same emissions
source.  They are deliberately not represented as RHIME emissions sectors:
the outer and inner basis operators may have different grids, state labels,
and region counts.  This module preserves those objects separately, masks the
outer prior response over the inner footprint extent before either basis is
built, and combines only the two observation-aligned sensitivity matrices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Any

from dask import compute as dask_compute
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions.array_ops import to_dense
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.models import (
    RhimeLikelihoodBuilder,
    StateActivity,
    build_nested_rhime_model_from_spec,
    get_rhime_likelihood_result,
)

from .builders import RhimeModelBuilderContext, RhimeModelBuildResult, validate_model_build_result
from .params import RhimeRunnerSetup, params_from_config
from .runner import (
    RhimeResult,
    assemble_rhime_inputs,
    build_rhime_basis,
    build_rhime_sensitivities,
    filter_rhime_observations,
    make_standard_rhime_result,
    materialize_pymc_inputs,
    resolve_rhime_options,
    retrieve_or_reload_rhime_data,
    sample_rhime_model,
    with_prepared_rhime_sites,
)
from .sampling import RhimeSampler
from .specs import RhimeRunSpec

__all__ = [
    "NestedRhimePreparedInputs",
    "NestedRhimeResult",
    "build_nested_rhime_model_result",
    "align_inner_merged_to_outer_observations",
    "combine_nested_rhime_inputs",
    "mask_outer_merged_for_inner_domain",
    "prepare_nested_rhime_inputs",
    "run_rhime_nested",
    "run_rhime_nested_from_prepared_inputs",
]


@dataclass(frozen=True)
class NestedRhimePreparedInputs:
    """Prepared outer, inner, and combined nested-domain inputs.

    ``outer`` and ``inner`` retain their native basis operators and flux grids.
    ``combined`` retains the outer basis for the ordinary RHIME validation
    contract and adds the aligned inner sensitivity as ``H_inner``.  The inner
    basis remains available explicitly on ``inner`` and is never coerced onto
    the outer grid.
    """

    outer: RhimePreparedInputs
    inner: RhimePreparedInputs
    combined: RhimePreparedInputs
    time_tolerance: str | pd.Timedelta | None = None
    inner_state_dim: str = "inner_region"

    def validated(self) -> NestedRhimePreparedInputs:
        """Revalidate both native preparations and rebuild their combination."""
        return combine_nested_rhime_inputs(
            self.outer,
            self.inner,
            time_tolerance=self.time_tolerance,
            inner_state_dim=self.inner_state_dim,
        )


@dataclass
class NestedRhimeResult:
    """A modern RHIME result with both native-domain preparation artifacts."""

    rhime_result: RhimeResult
    prepared_inputs: NestedRhimePreparedInputs

    @property
    def idata(self):
        """Return the sampled ArviZ inference data."""
        return self.rhime_result.idata

    @property
    def model(self):
        """Return the nested PyMC model."""
        return self.rhime_result.model

    @property
    def run_spec(self):
        """Return the retained-site-aligned run specification."""
        return self.rhime_result.run_spec

    @property
    def model_spec(self):
        """Return the model specification used by the nested graph."""
        return self.rhime_result.model_spec

    @property
    def output_spec(self):
        """Return the validated output specification."""
        return self.rhime_result.output_spec

    @property
    def inv_inputs(self) -> xr.Dataset:
        """Return combined inputs containing ``H`` and ``H_inner``."""
        return self.rhime_result.inv_inputs

    @property
    def outer_basis_functions(self):
        """Return the retained native outer-domain basis."""
        return self.prepared_inputs.outer.basis_functions

    @property
    def inner_basis_functions(self):
        """Return the retained native inner-domain basis."""
        return self.prepared_inputs.inner.basis_functions

    @property
    def sampler(self) -> RhimeSampler:
        """Return the sampler settings used by the run."""
        return self.rhime_result.sampler

    @property
    def model_build_result(self) -> RhimeModelBuildResult | None:
        """Return the model and explicit variable-role manifest."""
        return self.rhime_result.model_build_result

    @property
    def output_metadata(self) -> dict[str, Any]:
        """Return RHIME timing and nested-domain output metadata."""
        return self.rhime_result.output_metadata


def _state_dimension(sensitivity: xr.DataArray, *, label: str) -> str:
    """Return the single non-measurement state dimension."""
    if "nmeasure" not in sensitivity.dims:
        raise ValueError(f"{label} sensitivity must contain the `nmeasure` dimension.")
    state_dims = [str(dim) for dim in sensitivity.dims if dim != "nmeasure"]
    if len(state_dims) != 1:
        raise ValueError(f"{label} sensitivity must have exactly one state dimension; found {state_dims!r}.")
    return state_dims[0]


def _measurement_index(prepared: RhimePreparedInputs, *, label: str) -> pd.MultiIndex:
    """Return the canonical site/time measurement index."""
    index = prepared.inv_inputs.indexes.get("nmeasure")
    if not isinstance(index, pd.MultiIndex) or tuple(index.names) != ("site", "time"):
        raise ValueError(f"{label} inputs require an nmeasure MultiIndex with levels ('site', 'time').")
    return index


def _inner_measurement_positions(
    outer_index: pd.MultiIndex,
    inner_index: pd.MultiIndex,
    *,
    time_tolerance: str | pd.Timedelta | None,
) -> list[int]:
    """Map each outer site/time observation to an inner observation position."""
    tolerance = None if time_tolerance is None else pd.Timedelta(time_tolerance)
    positions = [-1] * len(outer_index)
    missing: list[tuple[str, object]] = []

    outer_sites = outer_index.get_level_values("site")
    inner_sites = inner_index.get_level_values("site")
    for site in dict.fromkeys(str(value) for value in outer_sites):
        outer_site_positions = [index for index, value in enumerate(outer_sites) if str(value) == site]
        inner_site_positions = [index for index, value in enumerate(inner_sites) if str(value) == site]
        if not inner_site_positions:
            missing.extend((site, outer_index[index][1]) for index in outer_site_positions)
            continue

        inner_times = pd.DatetimeIndex([inner_index[index][1] for index in inner_site_positions])
        if inner_times.has_duplicates:
            raise ValueError(f"Inner nested inputs contain duplicate observation times for site {site!r}.")
        order = inner_times.argsort()
        sorted_times = inner_times.take(order)
        sorted_positions = [inner_site_positions[int(index)] for index in order]
        outer_times = pd.DatetimeIndex([outer_index[index][1] for index in outer_site_positions])
        if tolerance is None:
            site_indexer = sorted_times.get_indexer(outer_times)
        else:
            site_indexer = sorted_times.get_indexer(
                outer_times,
                method="nearest",
                tolerance=tolerance,
            )

        for outer_position, mapped_position in zip(outer_site_positions, site_indexer, strict=True):
            inner_position = int(mapped_position)
            if inner_position < 0:
                missing.append((site, outer_index[outer_position][1]))
            else:
                positions[outer_position] = sorted_positions[inner_position]

    if missing:
        preview = ", ".join(f"{site}@{time}" for site, time in missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        alignment = "exact timestamps" if tolerance is None else f"nearest timestamps within {tolerance}"
        raise ValueError(
            "Inner nested inputs cannot be aligned to every outer observation using "
            f"{alignment}: {preview}{suffix}."
        )

    return positions


def combine_nested_rhime_inputs(
    outer: RhimePreparedInputs,
    inner: RhimePreparedInputs,
    *,
    time_tolerance: str | pd.Timedelta | None = None,
    inner_state_dim: str = "inner_region",
) -> NestedRhimePreparedInputs:
    """Combine independently prepared native grids at the sensitivity boundary.

    Exact site/time alignment is required by default.  Set ``time_tolerance``
    to opt into nearest-time alignment for footprints whose timestamps differ
    slightly from the observations.  Unmatched observations are rejected;
    they are never silently duplicated or converted to zero sensitivity.
    """
    outer = outer.validated()
    inner = inner.validated()
    if outer.sites != inner.sites:
        raise ValueError(
            "Nested RHIME outer and inner prepared inputs must retain the same sites in the same order; "
            f"outer={outer.sites!r}, inner={inner.sites!r}."
        )
    if outer.averaging_period != inner.averaging_period:
        raise ValueError(
            "Nested RHIME outer and inner averaging-period metadata must match; "
            f"outer={outer.averaging_period!r}, inner={inner.averaging_period!r}."
        )

    outer_h = outer.inv_inputs["H"]
    inner_h = inner.inv_inputs["H"]
    outer_state_dim = _state_dimension(outer_h, label="Outer")
    source_coord = outer_h.coords.get("source")
    inner_source_coord = inner_h.coords.get("source")
    if "source" in outer_h.dims or "source" in inner_h.dims:
        raise ValueError(
            "Nested-domain preparation currently supports one emissions source; use ordinary "
            "run_rhime_multisector for same-grid sector separation."
        )
    if source_coord is not None and inner_source_coord is not None:
        if str(source_coord.item()) != str(inner_source_coord.item()):
            raise ValueError(
                "Nested RHIME outer and inner sensitivities must describe the same emissions source."
            )
    if inner_state_dim == outer_state_dim:
        raise ValueError(f"`inner_state_dim` must differ from the outer state dimension {outer_state_dim!r}.")
    if not inner_state_dim or inner_state_dim in outer.inv_inputs.dims:
        raise ValueError(
            "`inner_state_dim` must be a non-empty dimension name absent from the outer inputs; "
            f"got {inner_state_dim!r}."
        )

    outer_index = _measurement_index(outer, label="Outer")
    inner_index = _measurement_index(inner, label="Inner")
    if outer_index.has_duplicates:
        raise ValueError("Outer nested inputs contain duplicate site/time observations.")
    positions = _inner_measurement_positions(
        outer_index,
        inner_index,
        time_tolerance=time_tolerance,
    )
    native_inner_state_dim = _state_dimension(inner_h, label="Inner")
    selected_inner = (
        inner_h.isel(nmeasure=positions)
        .transpose(native_inner_state_dim, "nmeasure")
        .rename({native_inner_state_dim: inner_state_dim})
    )
    inner_state_coordinate = selected_inner.coords.get(inner_state_dim)
    combined_dataset = outer.inv_inputs.copy(deep=False)
    combined_dataset["H_inner"] = xr.Variable(
        (inner_state_dim, "nmeasure"),
        selected_inner.data,
        attrs=dict(selected_inner.attrs),
    )
    if inner_state_coordinate is not None:
        combined_dataset = combined_dataset.assign_coords({inner_state_dim: inner_state_coordinate.variable})
    combined = RhimePreparedInputs(
        inv_inputs=combined_dataset,
        basis_functions=outer.basis_functions,
        site_metadata=outer.site_metadata,
    )
    return NestedRhimePreparedInputs(
        outer=outer,
        inner=inner,
        combined=combined,
        time_tolerance=time_tolerance,
        inner_state_dim=inner_state_dim,
    )


def _extent_mask(
    inner_dataset: xr.Dataset,
    *,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
) -> xr.DataArray:
    """Return a lazy target-grid mask for the rectangular inner extent."""
    if "lat" not in inner_dataset.coords or "lon" not in inner_dataset.coords:
        raise ValueError("Inner-domain datasets must contain indexed `lat` and `lon` coordinates.")
    inner_lat = inner_dataset.get_index("lat")
    inner_lon = inner_dataset.get_index("lon")
    if inner_lat.empty or inner_lon.empty:
        raise ValueError("Inner-domain latitude and longitude coordinates must not be empty.")
    lat_mask = (target_lat >= inner_lat.min()) & (target_lat <= inner_lat.max())
    lon_mask = (target_lon >= inner_lon.min()) & (target_lon <= inner_lon.max())
    return lat_mask & lon_mask


def _mask_spatial_variables(dataset: xr.Dataset, mask: xr.DataArray) -> xr.Dataset:
    """Return a shallow dataset copy with prior-response arrays lazily masked."""
    result = dataset.copy(deep=False)
    for name in ("fp", "fp_x_flux", "fp_x_flux_sectoral"):
        if name in result and set(mask.dims).issubset(result[name].dims):
            result[name] = result[name].where(~mask, other=0.0)
    return result


def _replace_data_object_data(value: Any, dataset: xr.Dataset) -> Any:
    """Return an OpenGHG data object with replacement data and retained metadata."""
    if is_dataclass(value):
        return replace(value, data=dataset)
    metadata = getattr(value, "metadata", None)
    try:
        return type(value)(data=dataset, metadata=metadata)
    except TypeError as exc:
        raise TypeError(
            "Nested-domain masking requires flux entries with replaceable `data` and `metadata`."
        ) from exc


def _select_merged_sites(merged: RhimeMergedData, sites: Sequence[str]) -> RhimeMergedData:
    """Return a borrowed merged-data view restricted to a common site set."""
    selected_sites = tuple(str(site) for site in sites)
    selected = {
        key: value for key, value in merged.fp_all.items() if key.startswith(".") or key in selected_sites
    }
    scales = selected.get(".scales")
    if isinstance(scales, Mapping):
        selected[".scales"] = {site: scales[site] for site in selected_sites if site in scales}
    return RhimeMergedData(
        fp_all=selected,
        site_options=merged.site_options.retain_sites(selected_sites, context="Nested-domain alignment"),
    )


def _retain_common_sites(
    outer: RhimeMergedData,
    inner: RhimeMergedData,
) -> tuple[RhimeMergedData, RhimeMergedData]:
    """Select sites present in both domains, preserving outer order."""
    common_sites = tuple(site for site in outer.sites if site in set(inner.sites))
    if not common_sites:
        raise ValueError("No common sites remain across outer and inner nested-domain data preparation.")
    return _select_merged_sites(outer, common_sites), _select_merged_sites(inner, common_sites)


def align_inner_merged_to_outer_observations(
    outer: RhimeMergedData,
    inner: RhimeMergedData,
    *,
    time_tolerance: str | pd.Timedelta | None = None,
) -> RhimeMergedData:
    """Select inner scenarios at the filtered outer observation timestamps.

    Observation filters are evaluated once on the canonical outer scenario.
    This function then applies that retained observation selection to every
    inner-domain array. Exact matching is the default; nearest selection is
    site-local and requires an explicit tolerance.
    """
    if outer.sites != inner.sites:
        raise ValueError("Nested time alignment requires identical outer and inner site order.")
    tolerance = None if time_tolerance is None else pd.Timedelta(time_tolerance)
    fp_all = dict(inner.fp_all)
    for site in outer.sites:
        outer_dataset = outer.fp_all[site]
        inner_dataset = inner.fp_all[site]
        if not isinstance(outer_dataset, xr.Dataset) or not isinstance(inner_dataset, xr.Dataset):
            raise TypeError("Modern nested time alignment requires per-site xarray Datasets.")
        if "time" not in outer_dataset.indexes or "time" not in inner_dataset.indexes:
            raise ValueError(f"Nested datasets for site {site!r} require indexed time coordinates.")
        outer_times = pd.DatetimeIndex(outer_dataset.get_index("time"))
        inner_times = pd.DatetimeIndex(inner_dataset.get_index("time"))
        if outer_times.has_duplicates or inner_times.has_duplicates:
            raise ValueError(f"Nested datasets for site {site!r} contain duplicate timestamps.")
        order = inner_times.argsort()
        sorted_times = inner_times.take(order)
        if tolerance is None:
            indexer = sorted_times.get_indexer(outer_times)
        else:
            indexer = sorted_times.get_indexer(
                outer_times,
                method="nearest",
                tolerance=tolerance,
            )
        missing = indexer < 0
        if bool(missing.any()):
            missing_times = outer_times[missing]
            preview = ", ".join(str(time) for time in missing_times[:5])
            suffix = "" if len(missing_times) <= 5 else f", ... ({len(missing_times)} total)"
            alignment = "exact timestamps" if tolerance is None else f"nearest within {tolerance}"
            raise ValueError(
                f"Inner nested scenario for site {site!r} cannot align using {alignment}: "
                f"{preview}{suffix}."
            )
        native_positions = order[indexer]
        aligned = inner_dataset.isel(time=native_positions).assign_coords(
            time=outer_dataset["time"].variable
        )
        fp_all[site] = aligned
    return RhimeMergedData(fp_all=fp_all, site_options=inner.site_options)


def mask_outer_merged_for_inner_domain(
    outer: RhimeMergedData,
    inner: RhimeMergedData,
) -> RhimeMergedData:
    """Mask outer footprint response and prior flux over the inner extent.

    The returned datasets are shallow copies and ``where`` preserves Dask
    laziness.  The borrowed outer and inner merged-data objects are not
    modified.  The union of all retained inner site extents is applied to the
    retained outer flux used by basis generation and post-run reconstruction.
    """
    if outer.sites != inner.sites:
        raise ValueError("Masking requires outer and inner merged data aligned to identical sites.")

    fp_all = dict(outer.fp_all)
    for site in outer.sites:
        outer_dataset = outer.fp_all[site]
        inner_dataset = inner.fp_all[site]
        if not isinstance(outer_dataset, xr.Dataset) or not isinstance(inner_dataset, xr.Dataset):
            raise TypeError("Modern nested-domain preparation requires per-site xarray Datasets.")
        if "lat" not in outer_dataset.coords or "lon" not in outer_dataset.coords:
            raise ValueError(f"Outer-domain dataset for site {site!r} is missing lat/lon coordinates.")
        site_mask = _extent_mask(
            inner_dataset,
            target_lat=outer_dataset["lat"],
            target_lon=outer_dataset["lon"],
        )
        fp_all[site] = _mask_spatial_variables(outer_dataset, site_mask)

    flux_entries = outer.fp_all.get(".flux")
    if not isinstance(flux_entries, Mapping) or not flux_entries:
        raise ValueError("Outer nested-domain merged data requires flux entries under `.flux`.")
    masked_flux_entries: dict[str, Any] = {}
    for source, flux_data in flux_entries.items():
        flux_dataset = getattr(flux_data, "data", None)
        if not isinstance(flux_dataset, xr.Dataset) or "flux" not in flux_dataset:
            raise TypeError(f"Outer flux entry {source!r} does not contain an xarray `flux` field.")
        flux = flux_dataset["flux"]
        if "lat" not in flux.coords or "lon" not in flux.coords:
            raise ValueError(f"Outer flux entry {source!r} is missing lat/lon coordinates.")
        union_mask: xr.DataArray | None = None
        for site in inner.sites:
            site_mask = _extent_mask(
                inner.fp_all[site],
                target_lat=flux["lat"],
                target_lon=flux["lon"],
            )
            union_mask = site_mask if union_mask is None else union_mask | site_mask
        if union_mask is None:  # guarded by the non-empty site invariant
            raise ValueError("Nested-domain masking requires at least one retained site.")
        masked_dataset = flux_dataset.copy(deep=False)
        masked_dataset["flux"] = flux.where(~union_mask, other=0.0)
        masked_flux_entries[str(source)] = _replace_data_object_data(flux_data, masked_dataset)
    fp_all[".flux"] = masked_flux_entries

    return RhimeMergedData(fp_all=fp_all, site_options=outer.site_options)


def _inner_domain_name(outer_domain: str, inner_domain: str) -> str:
    """Return the OpenGHG nested domain name without duplicating its prefix."""
    outer = str(outer_domain).strip()
    inner = str(inner_domain).strip()
    if not inner:
        raise ValueError("`inner_domain` must be a non-empty OpenGHG domain name or suffix.")
    return inner if inner == outer or inner.startswith(f"{outer}-") else f"{outer}-{inner}"


def _prepare_one_domain(
    merged: RhimeMergedData,
    data_args: Mapping[str, Any],
) -> RhimePreparedInputs:
    """Run the modern basis, sensitivity, and assembly stages for one grid."""
    basis_functions = build_rhime_basis(merged, data_args)
    site_data = build_rhime_sensitivities(
        merged,
        basis_functions,
        data_args,
        multisector=False,
    )
    return assemble_rhime_inputs(merged, basis_functions, site_data, data_args)


def _nested_sensitivity_reductions(
    merged: RhimeMergedData,
    *,
    label: str,
) -> list[xr.DataArray]:
    """Return lazy per-site absolute ``fp_x_flux`` reductions."""
    reductions: list[xr.DataArray] = []
    for site in merged.sites:
        dataset = merged.fp_all[site]
        if not isinstance(dataset, xr.Dataset):
            raise TypeError(f"{label} nested site {site!r} must be an xarray Dataset.")
        if "fp_x_flux" in dataset:
            sensitivity = dataset["fp_x_flux"]
        elif "fp_x_flux_sectoral" in dataset:
            sensitivity = dataset["fp_x_flux_sectoral"]
        else:
            raise ValueError(
                f"{label} nested site {site!r} requires `fp_x_flux` or `fp_x_flux_sectoral` "
                "for automatic basis-budget allocation. Set `inner_nbasis` explicitly to "
                "avoid automatic allocation."
            )
        reductions.append(abs(sensitivity).fillna(0.0).sum())
    if not reductions:
        raise ValueError(f"{label} nested data contains no retained sites for basis-budget allocation.")
    return reductions


def _allocate_nested_nbasis(
    outer: RhimeMergedData,
    inner: RhimeMergedData,
    *,
    total_nbasis: int,
) -> tuple[int, int]:
    """Materialize and split a total nested basis budget by sensitivity.

    The square-root sensitivity ratio damps differences between grids. The
    inner share is bounded to 35--60 percent, matching the final behavior on
    the legacy inner-domain branch while keeping execution explicit in the
    modern preparation stage.
    """
    if isinstance(total_nbasis, bool) or not isinstance(total_nbasis, Integral) or total_nbasis < 2:
        raise ValueError("Nested automatic basis allocation requires integer `nbasis >= 2`.")
    total_nbasis = int(total_nbasis)

    outer_reductions = _nested_sensitivity_reductions(outer, label="Outer")
    inner_reductions = _nested_sensitivity_reductions(inner, label="Inner")
    reductions = [*outer_reductions, *inner_reductions]
    materialization_start = timer_start()
    computed = dask_compute(*(to_dense(reduction).data for reduction in reductions))
    outer_total = float(sum(np.asarray(value).item() for value in computed[: len(outer_reductions)]))
    inner_total = float(sum(np.asarray(value).item() for value in computed[len(outer_reductions) :]))
    if not np.isfinite(outer_total) or not np.isfinite(inner_total):
        raise ValueError("Nested fp_x_flux sensitivity totals must be finite.")

    damped_outer = outer_total**0.5
    damped_inner = inner_total**0.5
    damped_total = damped_outer + damped_inner
    damped_inner_share = 0.5 if damped_total == 0.0 else damped_inner / damped_total
    inner_share = min(0.60, max(0.35, damped_inner_share))
    inner_nbasis = max(1, min(total_nbasis - 1, int(round(total_nbasis * inner_share))))
    outer_nbasis = total_nbasis - inner_nbasis
    log_timing(
        "rhime.nested_basis_budget_materialize",
        timer_seconds(materialization_start),
        total_nbasis=total_nbasis,
        outer_nbasis=outer_nbasis,
        inner_nbasis=inner_nbasis,
        outer_sensitivity=outer_total,
        inner_sensitivity=inner_total,
    )
    return outer_nbasis, inner_nbasis


def prepare_nested_rhime_inputs(
    setup: RhimeRunnerSetup,
    *,
    inner_domain: str,
    inner_footprint_store: str | None = None,
    inner_emissions_store: str | None = None,
    inner_basis_algorithm: str | None = None,
    inner_nbasis: int | None = None,
    inner_fp_basis_case: str | None = None,
    inner_basis_directory: str | None = None,
    inner_country_directory: str | None = None,
    inner_basis_output_path: str | None = None,
    inner_reload_merged_data: bool = False,
    inner_save_merged_data: bool = False,
    inner_merged_data_dir: str | None = None,
    inner_merged_data_name: str | None = None,
    time_tolerance: str | pd.Timedelta | None = None,
) -> NestedRhimePreparedInputs:
    """Retrieve and prepare native outer and inner grids with no double count."""
    if setup.run_spec.split_by_sectors or len(setup.run_spec.model.sectors) != 1:
        raise ValueError("Nested RHIME preparation currently requires a standard one-source setup.")

    outer_args = dict(setup.data_args)
    inner_args = dict(outer_args)
    inner_args.update(
        {
            "domain": _inner_domain_name(outer_args["domain"], inner_domain),
            "footprint_store": inner_footprint_store or outer_args.get("footprint_store", "user"),
            "emissions_store": inner_emissions_store or outer_args.get("emissions_store", "user"),
            "use_bc": False,
            "bc_input": None,
            "output_name": f"{outer_args['output_name']}_inner",
            "basis_algorithm": inner_basis_algorithm
            if inner_basis_algorithm is not None
            else (None if inner_fp_basis_case is not None else "quadtree"),
            "fp_basis_case": inner_fp_basis_case,
            "basis_output_path": inner_basis_output_path,
            "fix_basis_outer_regions": False,
            "reload_merged_data": inner_reload_merged_data,
            "save_merged_data": inner_save_merged_data,
            "merged_data_dir": inner_merged_data_dir,
            "merged_data_name": inner_merged_data_name,
        }
    )
    if inner_nbasis is not None:
        inner_args["nbasis"] = inner_nbasis
    if inner_basis_directory is not None:
        inner_args["basis_directory"] = inner_basis_directory
    if inner_country_directory is not None:
        inner_args["country_directory"] = inner_country_directory

    preparation_start = timer_start()
    outer_merged = retrieve_or_reload_rhime_data(outer_args, multisector=False)
    inner_merged = retrieve_or_reload_rhime_data(inner_args, multisector=False)
    outer_merged, inner_merged = _retain_common_sites(outer_merged, inner_merged)
    outer_filtered = filter_rhime_observations(outer_merged, outer_args)
    outer_filtered, inner_merged = _retain_common_sites(outer_filtered, inner_merged)
    inner_filtered = align_inner_merged_to_outer_observations(
        outer_filtered,
        inner_merged,
        time_tolerance=time_tolerance,
    )
    masked_outer = mask_outer_merged_for_inner_domain(outer_filtered, inner_filtered)

    if (
        inner_nbasis is None
        and outer_args.get("fp_basis_case") is None
        and inner_args.get("fp_basis_case") is None
    ):
        outer_nbasis, allocated_inner_nbasis = _allocate_nested_nbasis(
            masked_outer,
            inner_filtered,
            total_nbasis=outer_args.get("nbasis", 100),
        )
        outer_args["nbasis"] = outer_nbasis
        inner_args["nbasis"] = allocated_inner_nbasis

    outer_prepared = _prepare_one_domain(masked_outer, outer_args)
    inner_prepared = _prepare_one_domain(inner_filtered, inner_args)
    prepared = combine_nested_rhime_inputs(
        outer_prepared,
        inner_prepared,
        time_tolerance=time_tolerance,
    )
    log_timing(
        "rhime.prepare_nested_inputs",
        timer_seconds(preparation_start),
        sites=len(prepared.outer.sites),
        outer_regions=prepared.combined.inv_inputs["H"].sizes.get("region"),
        inner_regions=prepared.combined.inv_inputs["H_inner"].sizes.get("inner_region"),
    )
    return prepared


def build_nested_rhime_model_result(
    *,
    prepared: NestedRhimePreparedInputs,
    model_inputs: xr.Dataset,
    run_spec: RhimeRunSpec,
    inner_x_prior: Mapping[str, Any] | None = None,
    inner_state_activity: StateActivity | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> RhimeModelBuildResult:
    """Build and describe the first-class nested RHIME model."""
    model = build_nested_rhime_model_from_spec(
        model_inputs,
        run_spec.model,
        inner_x_prior=None if inner_x_prior is None else dict(inner_x_prior),
        inner_state_activity=inner_state_activity,
        likelihood_builder=likelihood_builder,
    )
    likelihood = get_rhime_likelihood_result(model)
    roles = {
        "observation": "mf",
        "observation_error": "mf_error",
        "minimum_error": "min_error",
        "flux_scale:outer": "x_outer",
        "flux_contribution:outer": "mu_outer",
        "emissions_sensitivity:outer": "hx_outer",
        "flux_scale:inner": "x_inner",
        "flux_contribution:inner": "mu_inner",
        "emissions_sensitivity:inner": "hx_inner",
        **likelihood.variable_roles,
    }
    if run_spec.model.use_bc:
        roles.update(
            {
                "baseline": "mu_bc",
                "baseline_scale": "bc",
                "baseline_sensitivity": "hbc",
            }
        )
    if run_spec.model.add_offset:
        roles["offset"] = "offset"
    result = RhimeModelBuildResult(
        model=model,
        variable_roles=roles,
        supported_output_formats=("none",),
        metadata={
            "kind": "builtin_nested",
            "outer_state_dimension": _state_dimension(
                prepared.combined.inv_inputs["H"],
                label="Outer",
            ),
            "inner_state_dimension": _state_dimension(
                prepared.combined.inv_inputs["H_inner"],
                label="Inner",
            ),
            **({} if not likelihood.metadata else {"likelihood": dict(likelihood.metadata)}),
        },
    )
    validate_model_build_result(
        result,
        context=RhimeModelBuilderContext(
            prepared_inputs=prepared.combined,
            run_spec=run_spec,
            multisector=False,
        ),
        builder_kind="likelihood" if likelihood_builder is not None else "model",
    )
    return result


def run_rhime_nested_from_prepared_inputs(
    *,
    prepared_inputs: NestedRhimePreparedInputs,
    run_spec: RhimeRunSpec,
    sampler: RhimeSampler | None = None,
    inner_x_prior: Mapping[str, Any] | None = None,
    inner_state_activity: StateActivity | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> NestedRhimeResult:
    """Build and sample nested RHIME from independently prepared native grids."""
    if likelihood_builder is not None and not callable(likelihood_builder):
        raise TypeError(
            f"`likelihood_builder` must be callable or None; got {type(likelihood_builder).__name__}."
        )
    if run_spec.output.output_format != "none":
        raise ValueError(
            "Nested RHIME currently supports output_format='none' only. The result retains both "
            "native bases and labelled posterior blocks; single-grid InversionOutput/PARIS writers "
            "must not be used because they would discard or mis-grid the inner posterior."
        )
    if run_spec.split_by_sectors or len(run_spec.model.sectors) != 1:
        raise ValueError("Nested RHIME currently requires a standard one-sector run specification.")

    prepared_inputs = prepared_inputs.validated()
    run_spec = with_prepared_rhime_sites(run_spec, prepared_inputs.combined)
    model_inputs = materialize_pymc_inputs(
        prepared_inputs.combined,
        aggregation_error_mode=run_spec.model.aggregation_error_mode,
        additional_variables=("H_inner",),
    )
    active_sampler = RhimeSampler() if sampler is None else sampler
    build_and_sample_start = timer_start()
    build_result = build_nested_rhime_model_result(
        prepared=prepared_inputs,
        model_inputs=model_inputs,
        run_spec=run_spec,
        inner_x_prior=inner_x_prior,
        inner_state_activity=inner_state_activity,
        likelihood_builder=likelihood_builder,
    )
    idata = sample_rhime_model(build_result, active_sampler, use_variable_roles=True)
    result = make_standard_rhime_result(
        prepared=prepared_inputs.combined,
        run_spec=run_spec,
        sampler=active_sampler,
        model_build_result=build_result,
        idata=idata,
        build_and_sample_seconds=timer_seconds(build_and_sample_start),
        likelihood_builder=likelihood_builder,
    )
    result.output_metadata["nested_domains"] = {
        "outer_state_dimension": build_result.metadata["outer_state_dimension"],
        "inner_state_dimension": build_result.metadata["inner_state_dimension"],
    }
    return NestedRhimeResult(rhime_result=result, prepared_inputs=prepared_inputs)


_NESTED_PARAMETER_NAMES = frozenset(
    {
        "inner_domain",
        "inner_footprint_store",
        "inner_emissions_store",
        "inner_basis_algorithm",
        "inner_nbasis",
        "inner_fp_basis_case",
        "inner_basis_directory",
        "inner_country_directory",
        "inner_basis_output_path",
        "inner_reload_merged_data",
        "inner_save_merged_data",
        "inner_merged_data_dir",
        "inner_merged_data_name",
        "inner_time_tolerance",
        "inner_x_prior",
    }
)


def run_rhime_nested(
    *,
    config_file: str | Path | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    **kwargs: Any,
) -> NestedRhimeResult:
    """Run a standard one-source RHIME inversion on nested native grids.

    Nested options may be supplied directly or in the INI file. Ordinary
    options use the same modern vocabulary as :func:`run_rhime`.
    ``output_format='none'`` is required until a dual-grid output schema is
    available.
    """
    if likelihood_builder is not None and not callable(likelihood_builder):
        raise TypeError(
            f"`likelihood_builder` must be callable or None; got {type(likelihood_builder).__name__}."
        )
    params = (
        params_from_config(config_file, extra_kwargs=kwargs, normalise=False)
        if config_file is not None
        else dict(kwargs)
    )
    nested_options = {name: params.pop(name) for name in tuple(params) if name in _NESTED_PARAMETER_NAMES}
    inner_domain = nested_options.pop("inner_domain", None)
    if inner_domain is None:
        raise ValueError("`run_rhime_nested` requires `inner_domain`.")
    inner_x_prior = nested_options.pop("inner_x_prior", None)
    if inner_x_prior is not None and not isinstance(inner_x_prior, Mapping):
        raise ValueError("`inner_x_prior` must be a prior mapping/dict or None.")
    time_tolerance = nested_options.pop("inner_time_tolerance", None)

    setup = resolve_rhime_options(params=params, multisector=False)
    if setup.run_spec.output.output_format != "none":
        raise ValueError("`run_rhime_nested` currently requires output_format='none'.")
    prepared = prepare_nested_rhime_inputs(
        setup,
        inner_domain=str(inner_domain),
        time_tolerance=time_tolerance,
        **nested_options,
    )
    return run_rhime_nested_from_prepared_inputs(
        prepared_inputs=prepared,
        run_spec=setup.run_spec,
        sampler=setup.sampler,
        inner_x_prior=inner_x_prior,
        likelihood_builder=likelihood_builder,
    )
