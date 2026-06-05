"""Shared runner-level inversion data preparation.

This module separates legacy fixedbasis preparation from the modern RHIME
prepared-input contract. New RHIME callers use ``flux_sources`` to name OpenGHG
flux ``source`` metadata values, while lower-level compatibility helpers still
pass those values through older ``emissions_name`` parameters internally.

``species`` is the primary gas or tracer name used for object-store lookup and
output naming. ``use_tracer`` is retained as an explicit unsupported option for
the current RHIME preparation path because tracer inversions require linked
forward models that are not represented here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
import warnings

import numpy as np
import xarray as xr

from openghg_inversions._timing import log_timing, timed, timer_seconds, timer_start
from openghg_inversions.basis import basis_functions_wrapper, make_basis_functions
from openghg_inversions.basis._helpers import _legacy_multisource_h_if_needed, bc_sensitivity
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.filters import filtering
from openghg_inversions.inversion_data.get_data import convert_to_list, data_processing_surface_notracer
from openghg_inversions.inversion_data.serialise import load_merged_data
from openghg_inversions.inversion_inputs import make_inv_inputs

MinErrorConfig = Literal["percentile", "residual"] | dict[str, float] | None | int | float


@dataclass
class FixedBasisPreparedData:
    """Data prepared for the legacy fixedbasis runner.

    Args:
        fp_all: Raw merged-data container returned by data gathering or reload.
        fp_data: Forward-model data after basis functions and filters.
        inv_inputs: Canonical inversion inputs consumed by model builders.
        sites: Retained site names after data gathering and filtering.
        averaging_period: Averaging periods aligned to retained sites.
        basis_objects: Basis objects returned by ``basis_functions_wrapper``.
        basis_artifact_source: Source of the flux basis artifact.
    """

    fp_all: dict
    fp_data: dict | None = None
    inv_inputs: xr.Dataset | None = None
    sites: list[str] = field(default_factory=list)
    averaging_period: list[str | None] = field(default_factory=list)
    basis_objects: dict[str, BasisFunctions] = field(default_factory=dict)
    basis_artifact_source: str = "generated"


@dataclass(frozen=True)
class RhimePreparedInputs:
    """Modern RHIME preparation contract.

    Args:
        inv_inputs: Canonical inversion inputs consumed by RHIME model
            builders.
        basis_functions: Retained flux basis object used to derive
            output-boundary basis and flux arrays.
        sites: Retained site names after data gathering and filtering.
        averaging_period: Averaging periods aligned to retained sites.
        basis_artifact_source: Description of whether the basis was generated
            or loaded from an artifact.
        site_lats: Release latitudes aligned to ``sites``, when available.
        site_lons: Release longitudes aligned to ``sites``, when available.
    """

    inv_inputs: xr.Dataset
    basis_functions: BasisFunctions
    sites: tuple[str, ...]
    averaging_period: tuple[str | None, ...]
    basis_artifact_source: str
    site_lats: tuple[float, ...] | None = None
    site_lons: tuple[float, ...] | None = None


@dataclass
class _MergedInversionData:
    """Merged data and site-aligned metadata shared by preparation paths."""

    fp_all: dict
    sites: list[str]
    averaging_period: list[str | None]


def _filter_site_aligned_value(value: object, keep_indices: list[int]) -> object:
    """Filter values that are aligned to the sites list."""
    if value is None or isinstance(value, str | bytes):
        return value
    if not isinstance(value, Sequence):
        return value
    return [item for index, item in enumerate(value) if index in keep_indices]


def _first_scalar_data_value(ds: xr.Dataset, names: tuple[str, ...]) -> float:
    """Return the first scalar value found in a dataset variable or coordinate."""
    for name in names:
        if name not in ds:
            continue
        values = np.asarray(ds[name].values).reshape(-1)
        if values.size:
            return float(values[0])
    return np.nan


def _site_release_coordinates(
    fp_data: dict[str, xr.Dataset],
    sites: Sequence[str],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return release lat/lon values aligned to retained sites."""
    lats: list[float] = []
    lons: list[float] = []
    for site in sites:
        site_data = fp_data.get(site)
        if site_data is None:
            lats.append(np.nan)
            lons.append(np.nan)
            continue
        lats.append(_first_scalar_data_value(site_data, ("release_lat", "sitelats")))
        lons.append(_first_scalar_data_value(site_data, ("release_lon", "sitelons")))
    return tuple(lats), tuple(lons)


def _drop_sites_missing_from_loaded_data(
    *,
    fp_all: dict,
    sites: list[str],
    inlet: Any,
    fp_height: Any,
    instrument: Any,
    max_level: int | None,
    averaging_period: list[str | None],
) -> tuple[list[str], Any, Any, Any, int | None, list[str | None]]:
    """Align site-level options when loaded merged data lacks requested sites."""
    sites_merged = [site for site in fp_all if not site.startswith(".")]
    if all(site in sites_merged for site in sites):
        return sites, inlet, fp_height, instrument, max_level, list(averaging_period)

    keep_indices = [index for index, site in enumerate(sites) if site in sites_merged]
    dropped_sites = [site for site in sites if site not in sites_merged]
    if not keep_indices:
        raise ValueError(
            "Loaded merged data does not include any requested sites. "
            f"Requested sites: {sites}. Available merged-data sites: {sites_merged}."
        )

    sites = [site for index, site in enumerate(sites) if index in keep_indices]
    averaging_period = [period for index, period in enumerate(averaging_period) if index in keep_indices]

    print(f"\nDropping {dropped_sites} sites as they are not included in the merged data object.\n")
    return (
        sites,
        _filter_site_aligned_value(inlet, keep_indices),
        _filter_site_aligned_value(fp_height, keep_indices),
        _filter_site_aligned_value(instrument, keep_indices),
        max_level,
        averaging_period,
    )


def _select_fp_all_sites(fp_all: dict, sites: list[str]) -> dict:
    """Keep only requested site entries and metadata from a merged-data object."""
    site_names = set(sites)
    return {key: value for key, value in fp_all.items() if key.startswith(".") or key in site_names}


def _normalise_averaging_period(
    averaging_period: list[str | None] | str | None, *, nsites: int
) -> list[str | None]:
    """Normalize and validate site-aligned averaging periods."""
    normalized = convert_to_list(averaging_period, length=nsites, name="averaging_period")
    invalid_periods = [period for period in normalized if period is not None and not isinstance(period, str)]
    if invalid_periods:
        raise ValueError(
            f"`averaging_period` entries must be strings or None. Invalid value(s): {invalid_periods!r}."
        )
    return normalized


def _make_inv_inputs(
    *,
    fp_data: dict,
    sites: list[str],
    start_date: str,
    bc_freq: str | None,
    sigma_freq: str | None,
    min_error: MinErrorConfig,
    calculate_min_error: Literal["percentile", "residual"] | None,
    min_error_options: dict | None,
) -> xr.Dataset:
    """Create canonical inversion inputs with legacy min-error compatibility."""
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

    min_error_options = min_error_options or {}
    return make_inv_inputs(
        fp_data,
        sites=sites,
        bc_freq=bc_freq,
        sigma_freq=sigma_freq,
        min_error=min_error,
        min_error_per_site=min_error_options.get("by_site", False),
        start_date=start_date,
    )


def _warn_for_nan_inputs(inv_inputs: xr.Dataset, *, use_bc: bool) -> None:
    """Warn when prepared sensitivity matrices contain NaN values."""
    if np.isnan(inv_inputs.H.values).any():
        warnings.warn(f"H matrix contains {np.isnan(inv_inputs.H.values).flatten().sum()} NaN values")
    if use_bc and "H_bc" in inv_inputs and np.isnan(inv_inputs.H_bc.values).any():
        warnings.warn(f"H_bc matrix contains {np.isnan(inv_inputs.H_bc.values).flatten().sum()} NaN values")


def _prepare_merged_data(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: list[str | None] | str | None,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: list[str] | None,
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    met_model: Any = None,
    fp_model: str | None = None,
    fp_height: Any = None,
    fp_species: str | None = None,
    inlet: Any = None,
    instrument: Any = None,
    max_level: int | None = None,
    calibration_scale: str | None = None,
    obs_data_level: Any = None,
    platform: Any = None,
    use_tracer: bool = False,
    use_bc: bool = True,
    bc_input: str | None = None,
    averaging_error: bool = True,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
) -> _MergedInversionData:
    """Gather or reload merged data and align site metadata.

    ``flux_sources`` contains modern OpenGHG flux ``source`` values. This
    helper passes them to lower-level data loading through the legacy
    ``emissions_name`` argument.
    """
    if use_tracer:
        raise ValueError("Tracer inversions are not supported by this preparation path.")
    if not sites:
        raise ValueError("At least one site must be specified for inversion data preparation.")

    averaging_period = _normalise_averaging_period(averaging_period, nsites=len(sites))
    rerun_merge = True
    fp_all: dict | None = None
    if reload_merged_data and merged_data_dir is not None:
        try:
            fp_all = load_merged_data(merged_data_dir, species, start_date, output_name, merged_data_name)
        except ValueError as exc:
            print(f"{exc}, re-running data merge.")
        else:
            print("Successfully read in merged data.\n")
            fp_all[".split_by_sectors"] = split_by_sectors
            rerun_merge = False
            sites, inlet, fp_height, instrument, max_level, averaging_period = (
                _drop_sites_missing_from_loaded_data(
                    fp_all=fp_all,
                    sites=sites,
                    inlet=inlet,
                    fp_height=fp_height,
                    instrument=instrument,
                    max_level=max_level,
                    averaging_period=averaging_period,
                )
            )
            fp_all = _select_fp_all_sites(fp_all, sites)
    elif reload_merged_data:
        print("Cannot reload merged data without a value for `merged_data_dir`; re-running data merge.")

    if rerun_merge:
        (
            fp_all,
            sites,
            inlet,
            fp_height,
            instrument,
            averaging_period,
        ) = data_processing_surface_notracer(
            species=species,
            sites=sites,
            domain=domain,
            averaging_period=averaging_period,
            start_date=start_date,
            end_date=end_date,
            obs_data_level=obs_data_level,
            platform=platform,
            met_model=met_model,
            fp_model=fp_model,
            fp_height=fp_height,
            fp_species=fp_species,
            emissions_name=flux_sources,
            inlet=inlet,
            instrument=instrument,
            max_level=max_level,
            calibration_scale=calibration_scale,
            use_bc=use_bc,
            bc_input=bc_input,
            bc_store=bc_store,
            obs_store=obs_store,
            footprint_store=footprint_store,
            emissions_store=emissions_store,
            split_by_sectors=split_by_sectors,
            averagingerror=averaging_error,
            save_merged_data=save_merged_data,
            merged_data_name=merged_data_name,
            merged_data_dir=merged_data_dir,
            output_name=output_name,
        )

    if fp_all is None:
        raise RuntimeError("Data preparation did not create or load merged data.")
    if not sites:
        raise ValueError("No sites remain after data gathering.")

    return _MergedInversionData(
        fp_all=fp_all,
        sites=sites,
        averaging_period=cast(list[str | None], averaging_period),
    )


def _apply_filters_and_drop_empty_sites(
    *,
    fp_data: dict,
    sites: list[str],
    averaging_period: list[str | None],
    filters: Any,
) -> tuple[dict, list[str], list[str | None]]:
    """Apply filters and keep site-aligned metadata in sync."""
    if filters is not None:
        try:
            fp_data = filtering(fp_data, filters)
        except ValueError:
            for site in sites:
                fp_data[site] = fp_data[site].compute()
            fp_data = filtering(fp_data, filters)

    dropped_sites = []
    for site in sites:
        if fp_data[site].time.values.shape[0] == 0:
            dropped_sites.append(site)
            del fp_data[site]
    if dropped_sites:
        keep_indices = [index for index, site in enumerate(sites) if site not in dropped_sites]
        if not keep_indices:
            raise ValueError(f"No sites remain after filtering. Dropped sites: {dropped_sites}.")

        sites = [site for index, site in enumerate(sites) if index in keep_indices]
        averaging_period = [period for index, period in enumerate(averaging_period) if index in keep_indices]
        print(f"\nDropping {dropped_sites} sites as no data passed the filtering.\n")

    return fp_data, sites, averaging_period


def _set_domain_attrs(fp_data: dict, sites: list[str], domain: str) -> None:
    """Attach the legacy domain attribute expected by downstream code."""
    for site in sites:
        fp_data[site].attrs["Domain"] = domain


def _bc_basis_directory_arg(bc_basis_directory: str | Path | None) -> str | None:
    """Normalize BC basis directory arguments for legacy helpers."""
    return str(bc_basis_directory) if isinstance(bc_basis_directory, Path) else bc_basis_directory


def _rhime_site_data_from_basis_functions(
    *,
    fp_all: dict,
    basis_functions: BasisFunctions,
    sites: list[str],
    domain: str,
    split_by_sectors: bool,
    flux_sources: list[str],
    use_bc: bool,
    bc_basis_case: str,
    bc_basis_directory: str | None,
) -> dict:
    """Build RHIME site datasets using retained basis functions only."""
    fp_data = {site: fp_all[site].copy() for site in sites}
    fp_x_flux_name = "fp_x_flux_sectoral" if split_by_sectors else "fp_x_flux"

    for site in sites:
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
        if state_dim != "region" and state_dim in sensitivity.dims:
            sensitivity = sensitivity.rename({state_dim: "region"})
            state_dim = "region"
        if split_by_sectors:
            sensitivity = _legacy_multisource_h_if_needed(
                sensitivity,
                state_dim=state_dim,
                flux_sources=flux_sources,
            )
        fp_data[site]["H"] = sensitivity
        log_timing(
            "rhime.prepare_inputs.footprint_sensitivity",
            timer_seconds(timing_start),
            site=site,
            nmeasure=fp_data[site].sizes.get("time"),
            regions=sensitivity.sizes.get("region"),
            sources=sensitivity.sizes.get("source"),
        )

    if use_bc:
        with timed("rhime.prepare_inputs.bc_sensitivity", sites=len(sites)):
            fp_data = bc_sensitivity(
                fp_data,
                domain=domain,
                basis_case=bc_basis_case,
                bc_basis_directory=bc_basis_directory,
            )

    return fp_data


def prepare_fixedbasis_inversion_data(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: list[str | None] | str | None,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: list[str] | None,
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    met_model: Any = None,
    fp_model: str | None = None,
    fp_height: Any = None,
    fp_species: str | None = None,
    inlet: Any = None,
    instrument: Any = None,
    max_level: int | None = None,
    calibration_scale: str | None = None,
    obs_data_level: Any = None,
    platform: Any = None,
    use_tracer: bool = False,
    use_bc: bool = True,
    fp_basis_case: str | None = None,
    basis_directory: str | None = None,
    bc_basis_case: str = "NESW",
    bc_basis_directory: str | Path | None = None,
    country_directory: str | None = None,
    bc_input: str | None = None,
    basis_algorithm: str = "weighted",
    nbasis: int = 100,
    filters: Any = None,
    fix_basis_outer_regions: bool = False,
    averaging_error: bool = True,
    bc_freq: str | None = None,
    sigma_freq: str | None = None,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
    basis_output_path: str | None = None,
    min_error: MinErrorConfig = 0.0,
    calculate_min_error: Literal["percentile", "residual"] | None = None,
    min_error_options: dict | None = None,
    return_basis_objects: bool = False,
    merged_data_only: bool = False,
) -> FixedBasisPreparedData:
    """Prepare data for legacy fixedbasisMCMC and its output adapters."""
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
        met_model=met_model,
        fp_model=fp_model,
        fp_height=fp_height,
        fp_species=fp_species,
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
    )

    if merged_data_only:
        return FixedBasisPreparedData(
            fp_all=merged.fp_all,
            sites=merged.sites,
            averaging_period=merged.averaging_period,
        )

    bc_basis_directory_arg = _bc_basis_directory_arg(bc_basis_directory)

    basis_result = basis_functions_wrapper(
        basis_algorithm=basis_algorithm,
        nbasis=nbasis,
        fp_basis_case=fp_basis_case,
        bc_basis_case=bc_basis_case,
        basis_directory=basis_directory,
        bc_basis_directory=bc_basis_directory_arg,
        country_directory=country_directory,
        fp_all=merged.fp_all,
        use_bc=use_bc,
        species=species,
        domain=domain,
        start_date=start_date,
        fix_outer_regions=fix_basis_outer_regions,
        emissions_name=flux_sources,
        outputname=output_name,
        output_path=basis_output_path,
        return_basis_objects=True,
    )
    fp_data, fixedbasis_basis_objects = cast(tuple[dict, dict[str, BasisFunctions]], basis_result)
    basis_source = fixedbasis_basis_objects["emissions"].basis_artifact_source or "generated"
    basis_objects = fixedbasis_basis_objects if return_basis_objects else {}

    fp_data, prepared_sites, prepared_averaging_period = _apply_filters_and_drop_empty_sites(
        fp_data=fp_data,
        sites=merged.sites,
        averaging_period=merged.averaging_period,
        filters=filters,
    )
    _set_domain_attrs(fp_data, prepared_sites, domain)

    inv_inputs = _make_inv_inputs(
        fp_data=fp_data,
        sites=prepared_sites,
        start_date=start_date,
        bc_freq=bc_freq,
        sigma_freq=sigma_freq,
        min_error=min_error,
        calculate_min_error=calculate_min_error,
        min_error_options=min_error_options,
    )
    _warn_for_nan_inputs(inv_inputs, use_bc=use_bc)

    return FixedBasisPreparedData(
        fp_all=merged.fp_all,
        fp_data=fp_data,
        inv_inputs=inv_inputs,
        basis_objects=basis_objects,
        basis_artifact_source=basis_source,
        sites=prepared_sites,
        averaging_period=prepared_averaging_period,
    )


def prepare_rhime_inputs(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: list[str | None] | str | None,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: list[str],
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    met_model: Any = None,
    fp_model: str | None = None,
    fp_height: Any = None,
    fp_species: str | None = None,
    inlet: Any = None,
    instrument: Any = None,
    max_level: int | None = None,
    calibration_scale: str | None = None,
    obs_data_level: Any = None,
    platform: Any = None,
    use_tracer: bool = False,
    use_bc: bool = True,
    fp_basis_case: str | None = None,
    basis_directory: str | None = None,
    bc_basis_case: str = "NESW",
    bc_basis_directory: str | Path | None = None,
    country_directory: str | None = None,
    bc_input: str | None = None,
    basis_algorithm: str = "weighted",
    nbasis: int = 100,
    filters: Any = None,
    fix_basis_outer_regions: bool = False,
    averaging_error: bool = True,
    bc_freq: str | None = None,
    sigma_freq: str | None = None,
    reload_merged_data: bool = False,
    save_merged_data: bool = False,
    merged_data_dir: str | None = None,
    merged_data_name: str | None = None,
    basis_output_path: str | None = None,
    min_error: MinErrorConfig = 0.0,
    min_error_options: dict | None = None,
) -> RhimePreparedInputs:
    """Prepare modern RHIME inputs without exposing legacy fixedbasis containers.

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
            with a ``source`` coordinate.
        use_tracer: Unsupported placeholder for tracer inversions, where an
            additional species constrains the primary species through linked
            forward models.

    Returns:
        Modern RHIME prepared inputs containing canonical ``inv_inputs`` and a
        retained ``BasisFunctions`` object.
    """
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
            met_model=met_model,
            fp_model=fp_model,
            fp_height=fp_height,
            fp_species=fp_species,
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
        )

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
            fp_all=merged.fp_all,
            species=species,
            domain=domain,
            start_date=start_date,
            fix_outer_regions=fix_basis_outer_regions,
            emissions_name=flux_sources,
            outputname=output_name,
            output_path=basis_output_path,
        )
    basis_source = basis_functions.basis_artifact_source or "generated"

    with timed("rhime.prepare_inputs.footprint_sensitivity_total", sites=len(merged.sites)):
        fp_data = _rhime_site_data_from_basis_functions(
            fp_all=merged.fp_all,
            basis_functions=basis_functions,
            sites=merged.sites,
            domain=domain,
            split_by_sectors=split_by_sectors,
            flux_sources=flux_sources,
            use_bc=use_bc,
            bc_basis_case=bc_basis_case,
            bc_basis_directory=_bc_basis_directory_arg(bc_basis_directory),
        )
    with timed("rhime.prepare_inputs.obs_filtering", sites=len(merged.sites), filters=filters is not None):
        fp_data, prepared_sites, prepared_averaging_period = _apply_filters_and_drop_empty_sites(
            fp_data=fp_data,
            sites=merged.sites,
            averaging_period=merged.averaging_period,
            filters=filters,
        )
    _set_domain_attrs(fp_data, prepared_sites, domain)

    with timed("rhime.prepare_inputs.make_inv_inputs", sites=len(prepared_sites)):
        inv_inputs = _make_inv_inputs(
            fp_data=fp_data,
            sites=prepared_sites,
            start_date=start_date,
            bc_freq=bc_freq,
            sigma_freq=sigma_freq,
            min_error=min_error,
            calculate_min_error=None,
            min_error_options=min_error_options,
        )
    _warn_for_nan_inputs(inv_inputs, use_bc=use_bc)
    site_lats, site_lons = _site_release_coordinates(fp_data, prepared_sites)
    log_timing(
        "rhime.prepare_inputs.prepared_dims",
        0.0,
        nmeasure=inv_inputs.sizes.get("nmeasure"),
        sites=len(prepared_sites),
        regions=inv_inputs.sizes.get("region"),
        sources=inv_inputs.sizes.get("source"),
        basis_source=basis_source,
    )

    return RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=basis_functions,
        basis_artifact_source=basis_source,
        sites=tuple(prepared_sites),
        averaging_period=tuple(prepared_averaging_period),
        site_lats=site_lats,
        site_lons=site_lons,
    )
