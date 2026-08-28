"""Prepare data for the temporary legacy ``fixedbasisMCMC`` workflow.

This module owns the fixed-basis compatibility contract.  It composes shared
retrieval, filtering, basis, and labelled-array mechanics from
``openghg_inversions.inversion_data.preparation`` without making that legacy
orchestration part of the modern RHIME preparation surface.

New workflows should use the named stages in
``openghg_inversions.rhime.preparation``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import xarray as xr

from openghg_inversions.basis import basis_functions_wrapper
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.flux_sanitization import FluxNonFiniteCheck
from openghg_inversions.inversion_data.preparation import (
    MinErrorConfig,
    SiteInletOption,
    SiteIntegerOption,
    SiteStringOption,
    _apply_filters_and_drop_empty_sites,
    _bc_basis_directory_arg,
    _make_inv_inputs,
    _prepare_merged_data,
    _scale_satellite_bc_sensitivity_to_column_signal,
    _select_fp_all_sites,
    _set_domain_attrs,
    _warn_for_nan_inputs,
)
from openghg_inversions.model_error import normalise_min_error_options
from openghg_inversions.sigma import SigmaAlignment

__all__ = ["FixedBasisPreparedData", "prepare_fixedbasis_inversion_data"]


@dataclass
class FixedBasisPreparedData:
    """Data prepared for the legacy fixed-basis runner.

    Args:
        fp_all: Raw merged-data container returned by data gathering or reload.
        fp_data: Forward-model data after basis functions and filters.
        inv_inputs: Canonical inversion inputs consumed by the legacy model.
        sites: Retained site names after data gathering and filtering.
        averaging_period: Averaging periods aligned to retained sites.
        basis_objects: Basis objects returned by ``basis_functions_wrapper``.
        basis_artifact_source: Source of the flux basis artifact.
        basis_artifact_path: Path to the flux basis artifact, when loaded or saved.
        is_column: Whether any retained observation is a column observation.
    """

    fp_all: dict
    fp_data: dict | None = None
    inv_inputs: xr.Dataset | None = None
    sites: list[str] = field(default_factory=list)
    averaging_period: list[str | None] = field(default_factory=list)
    basis_objects: dict[str, BasisFunctions] = field(default_factory=dict)
    basis_artifact_source: str = "generated"
    basis_artifact_path: str | None = None
    is_column: bool = False


def prepare_fixedbasis_inversion_data(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: SiteStringOption,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: list[str] | None,
    split_by_sectors: bool = False,
    bc_store: str = "user",
    obs_store: str = "user",
    footprint_store: str = "user",
    emissions_store: str = "user",
    met_model: SiteStringOption = None,
    fp_model: str | None = None,
    fp_height: SiteStringOption = None,
    fp_species: str | None = None,
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
    min_error_options: Mapping[str, Any] | None = None,
    return_basis_objects: bool = False,
    merged_data_only: bool = False,
    flux_non_finite_check: FluxNonFiniteCheck = "lazy",
) -> FixedBasisPreparedData:
    """Prepare data for legacy ``fixedbasisMCMC`` and its output adapters.

    This adapter preserves the fixed-basis inversion-input contract. Unless
    ``merged_data_only`` is true, the returned ``inv_inputs`` includes
    ``sigma_freq_index(nmeasure)`` derived from ``sigma_freq`` and anchored to
    ``start_date``. Modern RHIME preparation intentionally omits this
    component-specific variable.

    Args:
        sites: Requested observation site names. At least one unique site is
            required.
        averaging_period: Observation averaging period, either scalar or
            aligned to ``sites``.
        inlet: Inlet selector, either scalar or aligned to ``sites``. Entries
            may be strings, legacy ``slice`` selectors, or ``None``.
        fp_height: Footprint inlet height, either scalar or aligned to ``sites``.
        instrument: Observation instrument, either scalar or aligned to ``sites``.
        platform: Observation platform, either scalar or aligned to ``sites``.
        obs_data_level: Observation data level, either scalar or aligned to ``sites``.
        met_model: Footprint meteorological model, either scalar or aligned to ``sites``.
        max_level: Maximum column level, either scalar or aligned to ``sites``.
            Entries must be integers or ``None``.
        min_error: Numeric minimum error or ``"residual"``/``"percentile"``
            calculation method.
        calculate_min_error: Deprecated calculation-method spelling.
        min_error_options: Calculated minimum-error options. The only supported
            key is boolean ``by_site``.

    Returns:
        Prepared legacy data, including forward-model inputs and optional basis
        objects. When ``merged_data_only`` is true, only merged data and
        retained site metadata are populated.

    Raises:
        ValueError: If site options are empty, duplicated, misaligned, or have
            invalid types, or if minimum-error options are invalid.

    Warns:
        FutureWarning: If deprecated ``calculate_min_error`` is supplied.
    """
    min_error_options = normalise_min_error_options(min_error_options)
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
        flux_non_finite_check=flux_non_finite_check,
    )

    if merged_data_only:
        return FixedBasisPreparedData(
            fp_all=merged.fp_all,
            sites=list(merged.sites),
            averaging_period=list(merged.averaging_period),
            is_column=merged.site_options.is_column,
        )

    basis_result = basis_functions_wrapper(
        basis_algorithm=basis_algorithm,
        nbasis=nbasis,
        fp_basis_case=fp_basis_case,
        bc_basis_case=bc_basis_case,
        basis_directory=basis_directory,
        bc_basis_directory=_bc_basis_directory_arg(bc_basis_directory),
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
    emissions_basis = fixedbasis_basis_objects["emissions"]
    basis_source = emissions_basis.basis_artifact_source or "generated"
    basis_path = getattr(emissions_basis, "basis_artifact_path", None)
    basis_objects = fixedbasis_basis_objects if return_basis_objects else {}

    fp_data, prepared_site_options = _apply_filters_and_drop_empty_sites(
        fp_data=fp_data,
        site_options=merged.site_options,
        filters=filters,
    )
    prepared_sites = list(prepared_site_options.sites)
    prepared_averaging_period = list(prepared_site_options.averaging_period)
    _set_domain_attrs(fp_data, prepared_sites, domain)

    inv_inputs = _make_inv_inputs(
        fp_data=fp_data,
        sites=prepared_sites,
        start_date=start_date,
        bc_freq=bc_freq,
        min_error=min_error,
        calculate_min_error=calculate_min_error,
        min_error_per_site=min_error_options["by_site"],
    )
    inv_inputs = _scale_satellite_bc_sensitivity_to_column_signal(
        inv_inputs,
        sites=prepared_sites,
        platform=prepared_site_options.platform,
        observation_max_level=prepared_site_options.max_level,
        footprint_max_level=tuple(fp_data[site].attrs.get("max_level") for site in prepared_sites),
    )
    sigma_alignment = SigmaAlignment.from_frequency(
        inv_inputs["site_indicator"],
        frequency=sigma_freq,
        anchor_time=start_date,
    )
    inv_inputs["sigma_freq_index"] = sigma_alignment.period_index.rename("sigma_freq_index")
    _warn_for_nan_inputs(inv_inputs, use_bc=use_bc)

    return FixedBasisPreparedData(
        fp_all=_select_fp_all_sites(merged.fp_all, prepared_sites),
        fp_data=fp_data,
        inv_inputs=inv_inputs,
        basis_objects=basis_objects,
        basis_artifact_source=basis_source,
        basis_artifact_path=basis_path,
        sites=prepared_sites,
        averaging_period=prepared_averaging_period,
        is_column=prepared_site_options.is_column,
    )
