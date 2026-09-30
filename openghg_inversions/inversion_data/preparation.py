"""Scientific preparation and compatibility entry points for RHIME inversions.

``prepare_rhime_inputs`` returns backend-neutral observations, sensitivities,
basis metadata, and site metadata; component-specific model arrays are
intentionally absent.

The durable ``RhimePreparedInputs`` contract lives in :mod:`.prepared` and
remains importable here for compatibility. It validates labelled relationships
when it is constructed. When the retained basis-functions object
provides ``validated()``, preparation uses the returned copy after that method
has rechecked its mutable flux, operator data, and source labels. Compatible
objects without that hook are retained unchanged.

The compatibility ``prepare_rhime_inputs`` runner delegates store/cache access
to :mod:`.acquisition`. Scientific preparation may write basis artifacts, emit
warnings and progress messages,
and record timing information. These backend-neutral preparation functions do
not construct a PyMC model.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import xarray as xr

from openghg_inversions._timing import log_timing, timed, timer_seconds, timer_start
from openghg_inversions.basis import make_basis_functions
from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.boundary_sensitivity import (
    scale_satellite_boundary_sensitivity_to_column_signal as _scale_satellite_bc_sensitivity_to_column_signal,
)
from openghg_inversions.filters import filtering
from openghg_inversions.flux_sanitization import FluxNonFiniteCheck
from openghg_inversions.inversion_data.acquisition import (
    RhimeMergedData as RhimeMergedData,
    SiteBooleanOption as SiteBooleanOption,
    SiteInletOption as SiteInletOption,
    SiteIntegerOption as SiteIntegerOption,
    SiteStringOption as SiteStringOption,
    _SiteOptions,
    _drop_sites_missing_from_loaded_data as _drop_sites_missing_from_loaded_data,
    _normalise_site_booleans as _normalise_site_booleans,
    _normalise_site_inlets as _normalise_site_inlets,
    _normalise_site_integers as _normalise_site_integers,
    _normalise_site_strings as _normalise_site_strings,
    _prepare_merged_data,
    _select_fp_all_sites as _select_fp_all_sites,
    _validate_loaded_sector_layout as _validate_loaded_sector_layout,
    _validate_loaded_time_resolved_selector as _validate_loaded_time_resolved_selector,
)
# Preserve the established preparation imports while the durable contract has
# an owner independent of retrieval and preparation mechanics.
from openghg_inversions.inversion_data.prepared_inputs import (
    RHIME_PREPARED_INPUTS_SCHEMA as RHIME_PREPARED_INPUTS_SCHEMA,
    RHIME_PREPARED_INPUTS_SCHEMA_VERSION as RHIME_PREPARED_INPUTS_SCHEMA_VERSION,
    RhimePreparedInputs as RhimePreparedInputs,
    _make_site_metadata,
)
from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.model_error import normalise_min_error_options

MinErrorConfig = Literal["percentile", "residual"] | dict[str, float] | None | int | float


def _make_inv_inputs(
    *,
    fp_data: dict,
    sites: Sequence[str],
    start_date: str,
    bc_freq: str | None,
    min_error: MinErrorConfig,
    calculate_min_error: Literal["percentile", "residual"] | None,
    min_error_per_site: bool,
) -> xr.Dataset:
    """Create backend-neutral inversion inputs with min-error compatibility.

    Args:
        fp_data: Filtered per-site observations and sensitivity data.
        sites: Retained sites in observation order.
        start_date: Anchor for fixed-duration boundary-condition periods.
        bc_freq: Optional boundary-condition period frequency.
        min_error: Minimum-error value or calculation method.
        calculate_min_error: Deprecated minimum-error calculation argument.
        min_error_per_site: Whether calculated minimum error varies by site.

    Returns:
        Canonical observation-aligned inputs without component-specific model
        data.

    Warns:
        FutureWarning: If ``calculate_min_error`` is supplied.
    """
    if calculate_min_error is not None:
        warnings.warn(
            "`calculate_min_error` is deprecated. Please use `min_error` to pass the calculation method instead.",
            FutureWarning,
            stacklevel=3,
        )
        min_error = calculate_min_error

    if min_error is None:
        min_error = 0.0
    elif isinstance(min_error, int) and not isinstance(min_error, bool):
        min_error = float(min_error)
    elif isinstance(min_error, dict):
        missing_sites = [site for site in sites if site not in min_error]
        if missing_sites:
            raise ValueError(
                "`min_error` dictionaries must include a value for every retained site. "
                f"Missing site(s): {missing_sites!r}."
            )

    return make_inv_inputs(
        fp_data,
        sites=list(sites),
        bc_freq=bc_freq,
        min_error=min_error,
        min_error_per_site=min_error_per_site,
        start_date=start_date,
    )


def _warn_for_nan_inputs(inv_inputs: xr.Dataset, *, use_bc: bool) -> None:
    """Warn when prepared sensitivity matrices contain NaN values."""
    if np.isnan(inv_inputs.H.values).any():
        warnings.warn(f"H matrix contains {np.isnan(inv_inputs.H.values).flatten().sum()} NaN values")
    if use_bc and "H_bc" in inv_inputs and np.isnan(inv_inputs.H_bc.values).any():
        warnings.warn(f"H_bc matrix contains {np.isnan(inv_inputs.H_bc.values).flatten().sum()} NaN values")


def _apply_filters_and_drop_empty_sites(
    *,
    fp_data: dict,
    site_options: _SiteOptions,
    filters: Any,
) -> tuple[dict, _SiteOptions]:
    """Apply filters and keep site-aligned metadata in sync."""
    if filters is not None:
        try:
            fp_data = filtering(fp_data, filters)
        except ValueError:
            for site in site_options.sites:
                fp_data[site] = fp_data[site].compute()
            fp_data = filtering(fp_data, filters)

    dropped_sites = []
    for site in site_options.sites:
        if fp_data[site].sizes.get("time", 0) == 0:
            dropped_sites.append(site)
            del fp_data[site]
    if dropped_sites:
        keep_indices = [index for index, site in enumerate(site_options.sites) if site not in dropped_sites]
        if not keep_indices:
            raise ValueError(f"No sites remain after filtering. Dropped sites: {dropped_sites}.")

        site_options = site_options.select_indices(keep_indices)
        print(f"\nDropping {dropped_sites} sites as no data passed the filtering.\n")

    return fp_data, site_options


def _set_domain_attrs(fp_data: dict, sites: Sequence[str], domain: str) -> None:
    """Attach the legacy domain attribute expected by downstream code."""
    for site in sites:
        fp_data[site].attrs["Domain"] = domain


def _bc_basis_directory_arg(bc_basis_directory: str | Path | None) -> str | None:
    """Normalize BC basis directory arguments for legacy helpers."""
    return str(bc_basis_directory) if isinstance(bc_basis_directory, Path) else bc_basis_directory


def _validate_multisector_sensitivity_sources(
    sensitivity: xr.DataArray,
    *,
    site: str,
    flux_sources: list[str],
) -> xr.DataArray:
    """Validate and order one site's source-resolved sensitivity."""
    if "source" not in sensitivity.coords:
        raise ValueError(
            f"Site {site!r} sensitivity is missing the 'source' coordinate required for "
            f"flux source(s) {flux_sources!r}."
        )

    source_labels = [str(source) for source in sensitivity.coords["source"].values]
    available_sources = list(dict.fromkeys(source_labels))
    duplicate_sources = (
        [source for source in available_sources if source_labels.count(source) > 1]
        if "source" in sensitivity.dims
        else []
    )
    missing_sources = [source for source in flux_sources if source not in available_sources]
    extra_sources = [source for source in available_sources if source not in flux_sources]
    if duplicate_sources or missing_sources or extra_sources:
        raise ValueError(
            f"Site {site!r} sensitivity source layout does not match requested flux sources; "
            f"missing source(s): {missing_sources!r}; extra source(s): {extra_sources!r}; "
            f"duplicate source(s): {duplicate_sources!r}."
        )
    if "source" in sensitivity.dims:
        return sensitivity.sel(source=flux_sources)
    return sensitivity


def _rhime_site_data_from_basis_functions(
    *,
    merged: RhimeMergedData,
    basis_functions: BasisFunctions,
    domain: str,
    split_by_sectors: bool,
    flux_sources: list[str],
    use_bc: bool,
    bc_basis_case: str,
    bc_basis_directory: str | None,
) -> dict:
    """Apply retained basis functions to one prepared merged-data stage."""
    fp_data = {site: merged.fp_all[site].copy() for site in merged.sites}
    fp_x_flux_name = "fp_x_flux_sectoral" if split_by_sectors else "fp_x_flux"

    for site in merged.sites:
        if fp_data[site].sizes.get("time", 0) == 0:
            continue
        fp_x_flux = fp_data[site][fp_x_flux_name]
        timing_start = timer_start()
        sensitivity = basis_functions.sensitivity(fp_x_flux)
        state_dims = [dim for dim in sensitivity.dims if dim not in fp_x_flux.dims]
        if "region" in sensitivity.dims:
            state_dim = "region"
        elif len(state_dims) == 1:
            state_dim = cast(str, state_dims[0])
        else:
            raise ValueError(
                "Could not identify the RHIME sensitivity state dimension from "
                f"sensitivity dims {sensitivity.dims!r} and fp_x_flux dims {fp_x_flux.dims!r}."
            )
        if split_by_sectors:
            sensitivity = _validate_multisector_sensitivity_sources(
                sensitivity,
                site=site,
                flux_sources=flux_sources,
            )
        if "source" in sensitivity.coords and "source" not in sensitivity.dims:
            fp_data[site] = fp_data[site].drop_vars(fp_x_flux_name)
            orphan_dims = [
                dim
                for dim in fp_x_flux.dims
                if dim in fp_data[site].dims
                and all(dim not in variable.dims for variable in fp_data[site].data_vars.values())
            ]
            if orphan_dims:
                fp_data[site] = fp_data[site].drop_dims(orphan_dims)
        fp_data[site]["H"] = sensitivity
        log_timing(
            "rhime.prepare_inputs.footprint_sensitivity",
            timer_seconds(timing_start),
            site=site,
            nmeasure=fp_data[site].sizes.get("time"),
            state_size=sensitivity.sizes.get(state_dim),
            sources=sensitivity.sizes.get("source"),
        )

    if use_bc:
        with timed("rhime.prepare_inputs.bc_sensitivity", sites=len(merged.sites)):
            fp_data = bc_sensitivity(
                fp_data,
                domain=domain,
                basis_case=bc_basis_case,
                bc_basis_directory=bc_basis_directory,
            )

    return fp_data


def _filter_merged_inversion_data(
    *,
    merged: RhimeMergedData,
    filters: Any,
) -> RhimeMergedData:
    """Filter merged RHIME data as a separate pre-basis preparation stage.

    Args:
        merged: Merged site data and site-aligned metadata from data gathering
            or reload.
        filters: Filter configuration accepted by
            :func:`openghg_inversions.filters.filtering`.

    Returns:
        Merged data containing filtered site datasets, with empty sites and
        all of their aligned options removed. If no filters are configured and
        all sites contain data, the original merged data are returned.

    Raises:
        ValueError: If every requested site is removed by filtering.
    """
    if filters is None and all(merged.fp_all[site].sizes.get("time", 0) > 0 for site in merged.sites):
        return merged

    fp_data = {site: merged.fp_all[site].copy() for site in merged.sites}
    fp_data, site_options = _apply_filters_and_drop_empty_sites(
        fp_data=fp_data,
        site_options=merged.site_options,
        filters=filters,
    )
    fp_all = _select_fp_all_sites({**merged.fp_all, **fp_data}, site_options.sites)
    return RhimeMergedData(fp_all=fp_all, site_options=site_options)


def prepare_rhime_inputs(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: SiteStringOption,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: list[str],
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    emissions_domain: str | None = None,
    met_model: SiteStringOption = None,
    fp_model: str | None = None,
    fp_height: SiteStringOption = None,
    fp_species: str | None = None,
    time_resolved: SiteBooleanOption = None,
    inlet: SiteInletOption = None,
    instrument: SiteStringOption = None,
    max_level: SiteIntegerOption = None,
    calibration_scale: str | None = None,
    obs_data_level: SiteStringOption = None,
    platform: SiteStringOption = None,
    use_tracer: bool = False,
    use_bc: bool = True,
    fp_basis_case: str | None = None,
    basis_directory: str | None = None,
    bc_basis_case: str = "NESW",
    bc_basis_directory: str | Path | None = None,
    country_directory: str | None = None,
    outer_regions_path: str | Path | None = None,
    bc_input: str | None = None,
    basis_algorithm: str = "weighted",
    nbasis: int = 100,
    filters: Any = None,
    fix_basis_outer_regions: bool = False,
    averaging_error: bool = True,
    bc_freq: str | None = None,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
    basis_output_path: str | None = None,
    min_error: MinErrorConfig = 0.0,
    min_error_options: Mapping[str, Any] | None = None,
    flux_non_finite_check: FluxNonFiniteCheck = "lazy",
) -> RhimePreparedInputs:
    """Prepare modern RHIME inputs without exposing legacy fixedbasis containers.

    Observation filters are applied once to merged data before basis loading or
    generation. The same filtered site datasets and aligned metadata are then
    used for sensitivity construction.

    Args:
        species: Primary gas or tracer name used for object-store lookup and
            output naming.
        sites: Requested observation site names.
        domain: Model domain name.
        averaging_period: Observation averaging period, either scalar or
            site-aligned.
        start_date: Inclusive inversion start date.
        end_date: Exclusive inversion end date.
        output_name: Base output name used for data and basis artifacts.
        flux_sources: OpenGHG flux ``source`` values requested for the run.
        split_by_sectors: Whether to keep sector-resolved sensitivity inputs
            with a ``source`` provenance coordinate. Semantic sector names are
            applied later by the model specification.
        time_resolved: Footprint time-resolution selector, either scalar or
            aligned to ``sites``. ``True`` requests high-frequency footprints,
            ``False`` requests integrated footprints, and ``None`` leaves the
            store selection unspecified.
        inlet: Inlet selector, either scalar or aligned to ``sites``. Entries
            may be strings, legacy ``slice`` selectors, or ``None``.
        fp_height: Footprint inlet height, either scalar or aligned to
            ``sites``.
        instrument: Observation instrument, either scalar or aligned to
            ``sites``.
        platform: Observation platform, either scalar or aligned to ``sites``.
        obs_data_level: Observation data level, either scalar or aligned to
            ``sites``.
        met_model: Footprint meteorological model, either scalar or aligned to
            ``sites``.
        max_level: Maximum column level, either scalar or aligned to ``sites``.
            Entries must be integers or ``None``.
        outer_regions_path: Optional direct path to the fixed outer-region map
            used when ``fix_basis_outer_regions`` is true.
        emissions_domain: Optional flux-domain metadata selector. If it differs
            from ``domain``, flux is interpolated onto the footprint grid.
        min_error: Numeric minimum error or ``"residual"``/``"percentile"``
            calculation method.
        min_error_options: Calculated minimum-error options. The only supported
            key is boolean ``by_site``.
        use_tracer: Unsupported placeholder for tracer inversions, where an
            additional species constrains the primary species through linked
            forward models.
        flux_non_finite_check: Non-finite flux handling mode. ``"lazy"``
            applies zero-fill lazily and records attrs; ``"count"`` computes
            count metadata once and warns if non-finite values are present.

    Returns:
        Modern RHIME prepared inputs containing canonical ``inv_inputs`` and a
        retained ``BasisFunctions`` object.

    Raises:
        ValueError: If site options are empty, duplicated, misaligned, or have
            invalid types, or if minimum-error options are invalid.
    """
    min_error_options = normalise_min_error_options(min_error_options)
    with timed("rhime.prepare_inputs.merged_data", sites=len(sites), split_by_sectors=split_by_sectors):
        merged = _prepare_merged_data(
            species=species,
            sites=sites,
            domain=domain,
            averaging_period=averaging_period,
            start_date=start_date,
            end_date=end_date,
            output_name=output_name,
            flux_sources=flux_sources,
            split_by_sectors=split_by_sectors,
            bc_store=bc_store,
            obs_store=obs_store,
            footprint_store=footprint_store,
            emissions_store=emissions_store,
            emissions_domain=emissions_domain,
            met_model=met_model,
            fp_model=fp_model,
            fp_height=fp_height,
            fp_species=fp_species,
            time_resolved=time_resolved,
            inlet=inlet,
            instrument=instrument,
            max_level=max_level,
            calibration_scale=calibration_scale,
            obs_data_level=obs_data_level,
            platform=platform,
            use_tracer=use_tracer,
            use_bc=use_bc,
            bc_input=bc_input,
            averaging_error=averaging_error,
            reload_merged_data=reload_merged_data,
            save_merged_data=save_merged_data,
            merged_data_dir=merged_data_dir,
            merged_data_name=merged_data_name,
            flux_non_finite_check=flux_non_finite_check,
        )

    with timed("rhime.prepare_inputs.obs_filtering", sites=len(merged.sites), filters=filters is not None):
        filtered_merged = _filter_merged_inversion_data(merged=merged, filters=filters)

    with timed(
        "rhime.prepare_inputs.basis_build",
        basis_algorithm=basis_algorithm,
        nbasis=nbasis,
        fp_basis_case=fp_basis_case,
    ):
        basis_functions = make_basis_functions(
            basis_algorithm=basis_algorithm,
            nbasis=nbasis,
            fp_basis_case=fp_basis_case,
            basis_directory=basis_directory,
            country_directory=country_directory,
            outer_regions_path=outer_regions_path,
            fp_all=filtered_merged.fp_all,
            species=species,
            domain=domain,
            start_date=start_date,
            fix_outer_regions=fix_basis_outer_regions,
            emissions_name=flux_sources,
            outputname=output_name,
            output_path=basis_output_path,
        )

    with timed("rhime.prepare_inputs.footprint_sensitivity_total", sites=len(filtered_merged.sites)):
        fp_data = _rhime_site_data_from_basis_functions(
            merged=filtered_merged,
            basis_functions=basis_functions,
            domain=domain,
            split_by_sectors=split_by_sectors,
            flux_sources=flux_sources,
            use_bc=use_bc,
            bc_basis_case=bc_basis_case,
            bc_basis_directory=_bc_basis_directory_arg(bc_basis_directory),
        )
    basis_source = basis_functions.basis_artifact_source or "generated"
    _set_domain_attrs(fp_data, filtered_merged.sites, domain)

    with timed("rhime.prepare_inputs.make_inv_inputs", sites=len(filtered_merged.sites)):
        inv_inputs = _make_inv_inputs(
            fp_data=fp_data,
            sites=filtered_merged.sites,
            start_date=start_date,
            bc_freq=bc_freq,
            min_error=min_error,
            calculate_min_error=None,
            min_error_per_site=min_error_options["by_site"],
        )
    inv_inputs = _scale_satellite_bc_sensitivity_to_column_signal(
        inv_inputs,
        sites=filtered_merged.sites,
        platform=filtered_merged.site_options.platform,
        observation_max_level=filtered_merged.site_options.max_level,
        footprint_max_level=tuple(
            fp_data[site].attrs.get("footprint_max_level") for site in filtered_merged.sites
        ),
    )
    _warn_for_nan_inputs(inv_inputs, use_bc=use_bc)
    log_timing(
        "rhime.prepare_inputs.prepared_dims",
        0.0,
        nmeasure=inv_inputs.sizes.get("nmeasure"),
        sites=len(filtered_merged.sites),
        regions=inv_inputs.sizes.get("region"),
        sources=inv_inputs.sizes.get("source"),
        basis_source=basis_source,
    )

    return RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=basis_functions,
        site_metadata=_make_site_metadata(
            sites=filtered_merged.sites,
            averaging_period=filtered_merged.averaging_period,
        ),
    )
