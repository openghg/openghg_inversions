"""Deprecated acquisition-and-preparation entry point for RHIME inversions.

``prepare_rhime_inputs`` returns backend-neutral observations, sensitivities,
basis metadata, and site metadata; component-specific model arrays are
intentionally absent.

The durable ``RhimePreparedInputs`` contract lives in :mod:`openghg_inversions.inversion_data.prepared_inputs` and
remains importable here for compatibility. It validates labelled relationships
when it is constructed. When the retained basis-functions object
provides ``validated()``, preparation uses the returned copy after that method
has rechecked its mutable flux, operator data, and source labels. Compatible
objects without that hook are retained unchanged.

Scientific implementation lives in :mod:`openghg_inversions.rhime.preparation`;
this compatibility module delegates only to those canonical stages.

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
from typing import Any

from openghg_inversions.flux_sanitization import FluxNonFiniteCheck
from openghg_inversions.inversion_data._site_options import (
    SiteBooleanOption as SiteBooleanOption,
    SiteInletOption as SiteInletOption,
    SiteIntegerOption as SiteIntegerOption,
    SiteStringOption as SiteStringOption,
)
from openghg_inversions.inversion_data.acquisition import (
    RhimeMergedData as RhimeMergedData,
    _retrieve_or_reload_merged_data,
    _select_fp_all_sites as _select_fp_all_sites,
)
# Preserve the established preparation imports while the durable contract has
# an owner independent of retrieval and preparation mechanics.
from openghg_inversions.inversion_data.prepared_inputs import (
    RHIME_PREPARED_INPUTS_SCHEMA as RHIME_PREPARED_INPUTS_SCHEMA,
    RHIME_PREPARED_INPUTS_SCHEMA_VERSION as RHIME_PREPARED_INPUTS_SCHEMA_VERSION,
    RhimePreparedInputs as RhimePreparedInputs,
)
from openghg_inversions.model_error import (
    MinErrorConfig as MinErrorConfig,
    normalise_min_error,
    normalise_min_error_options,
)


def prepare_rhime_inputs(
    *,
    species: str,
    sites: list[str],
    domain: str,
    averaging_period: SiteStringOption,
    start_date: str,
    end_date: str,
    output_name: str,
    flux_sources: Sequence[str],
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
    """Deprecated acquisition-and-preparation compatibility entry point.

    Use ``RhimeMergedData.from_options`` or ``RhimeMergedData.load`` followed by the named scientific stages in
    :mod:`openghg_inversions.rhime.preparation`. This adapter preserves the
    established arguments and returns the same durable prepared-input contract.

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
        use_tracer: Unsupported linked-species flag. Omit or keep ``False``;
            ``True`` raises ``ValueError`` before acquisition.
        flux_non_finite_check: Non-finite flux handling mode. ``"lazy"``
            applies zero-fill lazily and records attrs; ``"count"`` computes
            count metadata once and warns if non-finite values are present.

    Returns:
        Modern RHIME prepared inputs containing canonical ``inv_inputs`` and a
        retained ``BasisFunctions`` object.

    Raises:
        ValueError: If site options are empty, duplicated, misaligned, or have
            invalid types, if minimum-error options are invalid, or if
            ``use_tracer=True``.

    Warns:
        DeprecationWarning: On every call; use acquisition and the named
            scientific stages directly.
    """
    from openghg_inversions.rhime.preparation import (
        assemble_rhime_inputs,
        build_rhime_basis,
        build_rhime_sensitivities,
        filter_rhime_observations,
    )

    warnings.warn(
        "prepare_rhime_inputs is deprecated; use RhimeMergedData acquisition followed by "
        "the scientific stages in openghg_inversions.rhime.preparation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if use_tracer:
        raise ValueError("`use_tracer=True` is not supported; tracer inversions are not implemented.")
    min_error = normalise_min_error(min_error)
    min_error_options = normalise_min_error_options(min_error_options)
    merged = _retrieve_or_reload_merged_data(
        sites=sites,
        averaging_period=averaging_period,
        inlet=inlet,
        fp_height=fp_height,
        instrument=instrument,
        platform=platform,
        obs_data_level=obs_data_level,
        met_model=met_model,
        max_level=max_level,
        time_resolved=time_resolved,
        species=species,
        domain=domain,
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
        fp_model=fp_model,
        fp_species=fp_species,
        calibration_scale=calibration_scale,
        use_bc=use_bc,
        bc_input=bc_input,
        averaging_error=averaging_error,
        reload_merged_data=reload_merged_data,
        save_merged_data=save_merged_data,
        merged_data_dir=merged_data_dir,
        merged_data_name=merged_data_name,
        flux_non_finite_check=flux_non_finite_check,
    )

    filtered = filter_rhime_observations(merged, filters=filters)
    basis_functions = build_rhime_basis(
        filtered,
        species=species,
        domain=domain,
        start_date=start_date,
        flux_sources=flux_sources,
        output_name=output_name,
        basis_algorithm=basis_algorithm,
        nbasis=nbasis,
        fp_basis_case=fp_basis_case,
        basis_directory=basis_directory,
        country_directory=country_directory,
        outer_regions_path=outer_regions_path,
        fix_basis_outer_regions=fix_basis_outer_regions,
        basis_output_path=basis_output_path,
    )
    site_data = build_rhime_sensitivities(
        filtered,
        basis_functions,
        domain=domain,
        flux_sources=flux_sources,
        use_bc=use_bc,
        bc_basis_case=bc_basis_case,
        bc_basis_directory=bc_basis_directory,
        multisector=split_by_sectors,
    )
    return assemble_rhime_inputs(
        filtered,
        basis_functions,
        site_data,
        domain=domain,
        start_date=start_date,
        bc_freq=bc_freq,
        min_error=min_error,
        min_error_options=min_error_options,
        use_bc=use_bc,
    )
