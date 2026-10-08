"""Named scientific preparation stages for RHIME recipes.

The functions in this module form the backend-neutral preparation spine used
by :func:`openghg_inversions.rhime.run_rhime` and copied project runners:

``filter -> basis -> sensitivities -> labelled assembly``.

Acquisition is owned by :mod:`openghg_inversions.inversion_data.acquisition`;
stages accept explicit resolved choices.

Merged data and xarray objects supplied to these stages are borrowed.  Stages
return new handoffs when they need to attach variables or metadata and never
mutate caller-owned datasets.  Retrieval may read OpenGHG stores or a merged
data cache and may write a requested merged-data artifact.  Filtering may
compute when a selected filter cannot operate lazily.  Basis construction may
read, fit, or write a basis artifact, and sensitivity construction may execute
the basis and boundary-condition algorithms.  Labelled assembly validates the
durable :class:`~openghg_inversions.inversion_data.RhimePreparedInputs`
artifact but does not make its arrays eager. Backend-specific materialization
is kept in :mod:`openghg_inversions.rhime.materialization` so this module reads
as the scientific transformation from merged observations to labelled inputs.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr

from openghg_inversions._timing import log_timing, timed, timer_seconds, timer_start
from openghg_inversions.basis import make_basis_functions
from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.boundary_sensitivity import scale_satellite_boundary_sensitivity_to_column_signal
from openghg_inversions.filters import filtering
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.inversion_data._site_options import SiteOptions
from openghg_inversions.inversion_data.prepared_inputs import _make_site_metadata
from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.model_error import MinErrorConfig
from openghg_inversions.rhime.specs import RhimeRunSpec

__all__ = [
    "assemble_rhime_inputs",
    "build_rhime_basis",
    "build_rhime_sensitivities",
    "filter_rhime_observations",
    "with_prepared_rhime_sites",
]


def with_prepared_rhime_sites(
    run_spec: RhimeRunSpec,
    prepared: RhimePreparedInputs,
) -> RhimeRunSpec:
    """Align run provenance to the observations retained by preparation."""
    return replace(
        run_spec,
        sites=tuple(prepared.sites),
        averaging_period=tuple(prepared.averaging_period),
    )


def filter_rhime_observations(
    merged: RhimeMergedData,
    *,
    filters: Any = None,
) -> RhimeMergedData:
    """Filter borrowed observations and remove empty sites with aligned metadata.

    The stage may compute site data if a filter cannot operate lazily.  It
    returns a new merged-data handoff when filtering changes data and never
    constructs basis functions or model inputs.

    Args:
        merged: Borrowed observations, shared scientific data and aligned
            site selectors from acquisition or reload.
        filters: Filter name, list of names, or mapping from site names to
            filters accepted by :func:`openghg_inversions.filters.filtering`.
            ``None`` applies no filters. The caller's mapping is not modified.

    Returns:
        Merged data with empty sites removed and all selectors aligned to the
        retained sites. Returns ``merged`` itself when no filtering or site
        removal is needed.

    Raises:
        ValueError: If no sites remain after filtering.
        KeyError: If a requested filter is unknown.
    """
    if isinstance(filters, dict):
        # The legacy filter implementation normalizes this working mapping.
        filters = dict(filters)

    with timed(
        "rhime.prepare_inputs.obs_filtering",
        sites=len(merged.sites),
        filters=filters is not None,
    ):
        return _filter_merged_inversion_data(merged=merged, filters=filters)


def build_rhime_basis(
    merged: RhimeMergedData,
    *,
    species: str,
    domain: str,
    start_date: str,
    flux_sources: Sequence[str],
    output_name: str,
    basis_algorithm: str = "weighted",
    nbasis: int = 100,
    fp_basis_case: str | None = None,
    basis_directory: str | Path | None = None,
    country_directory: str | Path | None = None,
    outer_regions_path: str | Path | None = None,
    fix_basis_outer_regions: bool = False,
    basis_output_path: str | Path | None = None,
    allow_empty_inner_region: bool = False,
) -> BasisFunctions:
    """Load or fit the retained RHIME basis for filtered observations.

    This stage may read or write basis artifacts and may execute the selected
    basis algorithm.  It treats ``merged`` as borrowed and does not build
    sensitivities.

    Args:
        merged: Filtered observations and prior fluxes used to load or fit the
            basis. Its scientific arrays remain borrowed.
        species: Gas used in basis artifact naming.
        domain: Domain used for basis and country-grid lookup.
        start_date: Run start date used to select and name basis artifacts.
        flux_sources: OpenGHG flux source names used for basis weighting.
        output_name: Run label used when saving a generated basis.
        basis_algorithm: Algorithm to generate a basis when no saved
            ``fp_basis_case`` is selected.
        nbasis: Requested number of generated basis regions.
        fp_basis_case: Saved basis case to load; takes precedence over
            ``basis_algorithm``.
        basis_directory: Directory containing saved basis cases.
        country_directory: Directory containing country grids needed by the
            selected basis algorithm.
        outer_regions_path: Optional fixed outer-region map.
        fix_basis_outer_regions: Keep the outer-region partition fixed while
            generating the inner partition.
        basis_output_path: Destination for a generated basis artifact;
            ``None`` disables saving.
        allow_empty_inner_region: Allow a fixed inner label with no remaining
            sensitivity, as occurs after nested-domain overlap masking.

    Returns:
        Retained basis functions and their prior fluxes, suitable for
        sensitivity construction and later scientific reconstruction.

    Raises:
        ValueError: If neither a saved basis nor an algorithm is selected,
            the algorithm is unknown, or the selected basis inputs are
            incompatible.
    """
    with timed(
        "rhime.prepare_inputs.basis_build",
        basis_algorithm=basis_algorithm,
        nbasis=nbasis,
        fp_basis_case=fp_basis_case,
    ):
        return make_basis_functions(
            basis_algorithm=basis_algorithm,
            nbasis=nbasis,
            fp_basis_case=fp_basis_case,
            basis_directory=basis_directory,
            country_directory=country_directory,
            outer_regions_path=outer_regions_path,
            fp_all=merged.to_legacy_fp_all(),
            species=species,
            domain=domain,
            start_date=start_date,
            fix_outer_regions=fix_basis_outer_regions,
            emissions_name=flux_sources,
            outputname=output_name,
            output_path=basis_output_path,
            allow_empty_inner_region=allow_empty_inner_region,
        )


def build_rhime_sensitivities(
    merged: RhimeMergedData,
    basis_functions: BasisFunctions,
    *,
    domain: str,
    flux_sources: Sequence[str],
    use_bc: bool = True,
    bc_basis_case: str = "NESW",
    bc_basis_directory: str | Path | None = None,
    multisector: bool,
) -> dict[str, xr.Dataset]:
    """Construct labelled flux and optional boundary-condition sensitivities.

    The stage creates per-site dataset copies, computes the basis projection,
    and may load boundary-condition basis data.  ``merged`` and
    ``basis_functions`` remain borrowed.

    Args:
        merged: Filtered observations and footprints with aligned site options.
        basis_functions: Retained basis and fluxes to project onto observations.
        domain: Domain used for boundary-condition basis lookup.
        flux_sources: Source names used to align the flux sensitivities.
        use_bc: Whether to construct boundary-condition sensitivities.
        bc_basis_case: Boundary-condition basis case used when ``use_bc`` is
            true.
        bc_basis_directory: Optional directory containing that basis case.
        multisector: Whether to retain separate source sensitivities instead
            of the standard single-source layout.

    Returns:
        Per-site datasets containing flux and, when requested,
        boundary-condition sensitivities for labelled assembly.

    Raises:
        ValueError: If the basis or source layout is incompatible with the
            selected sensitivity layout.
    """
    with timed("rhime.prepare_inputs.footprint_sensitivity_total", sites=len(merged.sites)):
        return _rhime_site_data_from_basis_functions(
            merged=merged,
            basis_functions=basis_functions,
            domain=domain,
            split_by_sectors=multisector,
            flux_sources=flux_sources,
            use_bc=use_bc,
            bc_basis_case=bc_basis_case,
            bc_basis_directory=str(bc_basis_directory) if isinstance(bc_basis_directory, Path) else bc_basis_directory,
        )


def assemble_rhime_inputs(
    merged: RhimeMergedData,
    basis_functions: BasisFunctions,
    site_data: Mapping[str, xr.Dataset],
    *,
    domain: str,
    start_date: str,
    bc_freq: str | None = None,
    min_error: MinErrorConfig = 0.0,
    min_error_options: Mapping[str, Any] | None = None,
    use_bc: bool = True,
) -> RhimePreparedInputs:
    """Construct and validate durable, backend-neutral RHIME model inputs.

    The stage attaches domain metadata to shallow per-site copies, assembles
    observation-aligned arrays, applies the satellite boundary-condition
    scaling, and retains basis and site metadata. It also preserves the legacy
    construction of the minimum-error floor and boundary-condition temporal
    parameterization. Those are inverse-model settings, not properties of the
    acquired data; moving them to their model components is a later semantic
    change. This stage does not cross the PyMC materialization boundary.

    Args:
        merged: Borrowed filtered data and authoritative retained site options.
        basis_functions: Basis and prior fluxes to retain with the inputs.
        site_data: Per-site datasets from sensitivity construction. Domain
            metadata is attached to shallow copies, leaving these datasets
            unchanged.
        domain: Domain label attached to the assembled site data.
        start_date: Requested start date used to anchor temporal parameters.
        bc_freq: Frequency of boundary-condition scaling parameters; ``None``
            keeps one period across the run.
        min_error: Minimum observation-error floor as a numeric value or a
            mapping covering every retained site, or the named ``"residual"``
            or ``"percentile"`` calculation.
        min_error_options: Resolved mapping with boolean ``by_site``, as
            supplied by ``RhimeConfig``. ``None`` uses a shared calculation.
            Explicit keyword calls must supply already-normalized options.
        use_bc: Whether missing boundary-condition inputs should be checked.

    Returns:
        Validated, backend-neutral inputs with observation-aligned arrays,
        retained basis functions and site metadata. Arrays may remain lazy;
        model input materialization is a separate operation.

    Raises:
        ValueError: If assembled inputs fail their alignment or scientific
            input contracts.
        KeyError: If the resolved ``by_site`` option is absent.
    """
    owned_site_data = {site: dataset.copy(deep=False) for site, dataset in site_data.items()}
    for site in merged.sites:
        owned_site_data[site].attrs["Domain"] = domain
    # These inverse-model settings are materialized into labelled arrays here
    # to preserve the existing numerical contract while preparation is split.
    with timed("rhime.prepare_inputs.make_inv_inputs", sites=len(merged.sites)):
        inv_inputs = _make_inv_inputs(
            fp_data=owned_site_data,
            sites=merged.sites,
            start_date=start_date,
            bc_freq=bc_freq,
            min_error=min_error,
            min_error_per_site=False if min_error_options is None else min_error_options["by_site"],
        )
    inv_inputs = scale_satellite_boundary_sensitivity_to_column_signal(
        inv_inputs,
        sites=merged.sites,
        platform=merged.platform,
        observation_max_level=merged.site_options.max_level,
        footprint_max_level=tuple(
            owned_site_data[site].attrs.get("footprint_max_level") for site in merged.sites
        ),
    )
    _warn_for_nan_inputs(inv_inputs, use_bc=use_bc)
    basis_source = basis_functions.basis_artifact_source or "generated"
    log_timing(
        "rhime.prepare_inputs.prepared_dims",
        0.0,
        nmeasure=inv_inputs.sizes.get("nmeasure"),
        sites=len(merged.sites),
        regions=inv_inputs.sizes.get("region"),
        sources=inv_inputs.sizes.get("source"),
        basis_source=basis_source,
    )
    site_metadata = _make_site_metadata(
        sites=merged.sites,
        averaging_period=merged.averaging_period,
    )
    def footprint_attr(site: str, name: str) -> str:
        value = owned_site_data[site].attrs.get(f"footprint_{name}")
        return value if isinstance(value, str) and value.strip().lower() not in {"", "none", "not_set"} else ""

    for name in ("transport_model", "transport_model_version", "met_model"):
        site_metadata[name] = (
            "site",
            [footprint_attr(site, name) for site in merged.sites],
        )
    return RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=basis_functions,
        site_metadata=site_metadata,
    )


def _make_inv_inputs(
    *,
    fp_data: dict,
    sites: Sequence[str],
    start_date: str,
    bc_freq: str | None,
    min_error: MinErrorConfig,
    min_error_per_site: bool,
) -> xr.Dataset:
    """Create backend-neutral inversion inputs with the resolved error floor.

    Args:
        fp_data: Filtered per-site observations and sensitivity data.
        sites: Retained sites in observation order.
        start_date: Anchor for fixed-duration boundary-condition periods.
        bc_freq: Optional boundary-condition period frequency.
        min_error: Minimum-error value or calculation method.
        min_error_per_site: Whether calculated minimum error varies by site.

    Returns:
        Canonical observation-aligned inputs without component-specific model
        data.
    """
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
        warnings.warn(
            f"H matrix contains {np.isnan(inv_inputs.H.values).flatten().sum()} NaN values", stacklevel=3
        )
    if use_bc and "H_bc" in inv_inputs and np.isnan(inv_inputs.H_bc.values).any():
        warnings.warn(
            f"H_bc matrix contains {np.isnan(inv_inputs.H_bc.values).flatten().sum()} NaN values",
            stacklevel=3,
        )


def _apply_filters_and_drop_empty_sites(
    *,
    fp_data: dict,
    site_options: SiteOptions,
    filters: Any,
) -> tuple[dict, SiteOptions]:
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


def _validate_multisector_sensitivity_sources(
    sensitivity: xr.DataArray,
    *,
    site: str,
    flux_sources: Sequence[str],
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
        return sensitivity.sel(source=list(flux_sources))
    return sensitivity


def _rhime_site_data_from_basis_functions(
    *,
    merged: RhimeMergedData,
    basis_functions: BasisFunctions,
    domain: str,
    split_by_sectors: bool,
    flux_sources: Sequence[str],
    use_bc: bool,
    bc_basis_case: str,
    bc_basis_directory: str | None,
) -> dict:
    """Apply retained basis functions to one prepared merged-data stage."""
    fp_data = {site: merged.site_data[site].copy() for site in merged.sites}
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
    if filters is None and all(merged.site_data[site].sizes.get("time", 0) > 0 for site in merged.sites):
        return merged

    fp_data = {site: merged.site_data[site].copy() for site in merged.sites}
    fp_data, site_options = _apply_filters_and_drop_empty_sites(
        fp_data=fp_data,
        site_options=merged.site_options,
        filters=filters,
    )
    return RhimeMergedData(
        site_data={site: fp_data[site] for site in site_options.sites},
        flux_data=merged.flux_data,
        boundary_data=merged.boundary_data,
        site_options=site_options,
        split_by_sectors=merged.split_by_sectors,
        provenance={
            **merged.provenance,
            "inputs": {
                key: value for key, value in merged.provenance["inputs"].items()
                if not key.startswith(("observations:", "footprints:"))
                or key.split(":", 1)[1] in site_options.sites
            },
        },
        acquisition={**merged.acquisition, "stage": "filtered"},
    )
