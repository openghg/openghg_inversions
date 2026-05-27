"""Shared runner-level inversion data preparation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
import warnings

import numpy as np
import xarray as xr

from openghg_inversions.array_ops import get_xr_dummies
from openghg_inversions.basis import basis_functions_wrapper
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.filters import filtering
from openghg_inversions.inversion_data.get_data import convert_to_list, data_processing_surface_notracer
from openghg_inversions.inversion_data.serialise import load_merged_data
from openghg_inversions.inversion_inputs import make_inv_inputs

MinErrorConfig = Literal["percentile", "residual"] | dict[str, float] | None | int | float


@dataclass
class PreparedInversionData:
    """Data prepared for RHIME-style inversion runners.

    Args:
        fp_all: Raw merged-data container returned by data gathering or reload.
        sites: Retained site names after data gathering and filtering.
        averaging_period: Averaging periods aligned to retained sites.
        fp_data: Forward-model data after basis functions and filters.
        inv_inputs: Canonical inversion inputs consumed by model builders.
        basis: Basis matrix aligned to ``inv_inputs.region``, when requested.
        flux: Prior flux field from the retained emissions basis object, when
            basis objects were requested.
        basis_objects: Basis objects returned by ``basis_functions_wrapper``.
    """

    fp_all: dict
    sites: list[str]
    averaging_period: list[str | None]
    fp_data: dict | None = None
    inv_inputs: xr.Dataset | None = None
    basis: xr.DataArray | None = None
    flux: xr.DataArray | None = None
    basis_objects: dict[str, BasisFunctions] = field(default_factory=dict)


def _filter_site_aligned_value(value: object, keep_indices: list[int]) -> object:
    """Filter values that are aligned to the sites list."""
    if value is None or isinstance(value, str | bytes):
        return value
    if not isinstance(value, Sequence):
        return value
    return [item for index, item in enumerate(value) if index in keep_indices]


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


def prepare_inversion_data(
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
) -> PreparedInversionData:
    """Gather data, apply basis functions and filters, and create inversion inputs.

    This helper owns the runner-level preparation shared by legacy
    ``fixedbasisMCMC`` and modern RHIME runners. It intentionally does not build
    PyMC models, sample, or create output products.
    """
    if use_tracer:
        raise ValueError("Tracer inversions are not supported by this RHIME preparation path.")
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

    if merged_data_only:
        return PreparedInversionData(
            fp_all=fp_all,
            sites=sites,
            averaging_period=cast(list[str | None], averaging_period),
        )

    bc_basis_directory_arg = (
        str(bc_basis_directory) if isinstance(bc_basis_directory, Path) else bc_basis_directory
    )

    basis_result = basis_functions_wrapper(
        basis_algorithm=basis_algorithm,
        nbasis=nbasis,
        fp_basis_case=fp_basis_case,
        bc_basis_case=bc_basis_case,
        basis_directory=basis_directory,
        bc_basis_directory=bc_basis_directory_arg,
        country_directory=country_directory,
        fp_all=fp_all,
        use_bc=use_bc,
        species=species,
        domain=domain,
        start_date=start_date,
        fix_outer_regions=fix_basis_outer_regions,
        emissions_name=flux_sources,
        outputname=output_name,
        output_path=basis_output_path,
        return_basis_objects=return_basis_objects,
    )
    if return_basis_objects:
        fp_data, basis_objects = cast(tuple[dict, dict[str, BasisFunctions]], basis_result)
    else:
        fp_data = cast(dict, basis_result)
        basis_objects = {}

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

    for site in sites:
        fp_data[site].attrs["Domain"] = domain

    inv_inputs = _make_inv_inputs(
        fp_data=fp_data,
        sites=sites,
        start_date=start_date,
        bc_freq=bc_freq,
        sigma_freq=sigma_freq,
        min_error=min_error,
        calculate_min_error=calculate_min_error,
        min_error_options=min_error_options,
    )
    _warn_for_nan_inputs(inv_inputs, use_bc=use_bc)

    basis = None
    flux = None
    if return_basis_objects:
        basis = get_xr_dummies(fp_data[".basis"], cat_dim="region", categories=inv_inputs.region)
        flux = basis_objects["emissions"].flux

    return PreparedInversionData(
        fp_all=fp_all,
        fp_data=fp_data,
        inv_inputs=inv_inputs,
        basis=basis,
        flux=flux,
        basis_objects=basis_objects,
        sites=sites,
        averaging_period=cast(list[str | None], averaging_period),
    )
