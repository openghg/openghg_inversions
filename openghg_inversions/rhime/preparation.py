"""Named scientific preparation stages for RHIME recipes.

The functions in this module form the backend-neutral preparation spine used
by :func:`openghg_inversions.rhime.run_rhime` and copied project runners:

``filter -> basis -> sensitivities -> labelled assembly``.

Acquisition is owned by :mod:`openghg_inversions.inversion_data.acquisition`;
stages accept explicit resolved choices and established mapping calls.

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

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import xarray as xr

from openghg_inversions._timing import log_timing, timed
from openghg_inversions.basis import make_basis_functions
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.inversion_data import preparation as inversion_preparation
from openghg_inversions.model_error import normalise_min_error_options
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
    data_args: Mapping[str, Any] | None = None,
    *,
    filters: Any = None,
) -> RhimeMergedData:
    """Filter borrowed observations and remove empty sites with aligned metadata.

    The stage may compute site data if a filter cannot operate lazily.  It
    returns a new merged-data handoff when filtering changes data and never
    constructs basis functions or model inputs.
    """
    if data_args is not None:
        filters = data_args["filters"]
    if isinstance(filters, dict):
        # The legacy filter implementation normalizes this working mapping.
        filters = dict(filters)

    with timed(
        "rhime.prepare_inputs.obs_filtering",
        sites=len(merged.sites),
        filters=filters is not None,
    ):
        return inversion_preparation._filter_merged_inversion_data(merged=merged, filters=filters)


def build_rhime_basis(
    merged: RhimeMergedData,
    data_args: Mapping[str, Any] | None = None,
    *,
    species: str | None = None,
    domain: str | None = None,
    start_date: str | None = None,
    flux_sources: Sequence[str] | None = None,
    output_name: str | None = None,
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
    """
    if data_args is not None:
        basis_algorithm = data_args["basis_algorithm"]
        nbasis = data_args["nbasis"]
        fp_basis_case = data_args["fp_basis_case"]
        basis_directory = data_args["basis_directory"]
        country_directory = data_args["country_directory"]
        outer_regions_path = data_args["outer_regions_path"]
        species = data_args["species"]
        domain = data_args["domain"]
        start_date = data_args["start_date"]
        fix_basis_outer_regions = data_args["fix_basis_outer_regions"]
        flux_sources = data_args["flux_sources"]
        output_name = data_args["output_name"]
        basis_output_path = data_args["basis_output_path"]
        allow_empty_inner_region = data_args.get("allow_empty_inner_region", False)

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
            fp_all=merged.fp_all,
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
    data_args: Mapping[str, Any] | None = None,
    *,
    domain: str | None = None,
    flux_sources: Sequence[str] | None = None,
    use_bc: bool = True,
    bc_basis_case: str = "NESW",
    bc_basis_directory: str | Path | None = None,
    multisector: bool,
) -> dict[str, xr.Dataset]:
    """Construct labelled flux and optional boundary-condition sensitivities.

    The stage creates per-site dataset copies, computes the basis projection,
    and may load boundary-condition basis data.  ``merged`` and
    ``basis_functions`` remain borrowed.
    """
    if data_args is not None:
        domain = data_args["domain"]
        flux_sources = data_args["flux_sources"]
        use_bc = data_args["use_bc"]
        bc_basis_case = data_args["bc_basis_case"]
        bc_basis_directory = data_args["bc_basis_directory"]

    with timed("rhime.prepare_inputs.footprint_sensitivity_total", sites=len(merged.sites)):
        return inversion_preparation._rhime_site_data_from_basis_functions(
            merged=merged,
            basis_functions=basis_functions,
            domain=domain,
            split_by_sectors=multisector,
            flux_sources=flux_sources,
            use_bc=use_bc,
            bc_basis_case=bc_basis_case,
            bc_basis_directory=inversion_preparation._bc_basis_directory_arg(
                bc_basis_directory
            ),
        )


def assemble_rhime_inputs(
    merged: RhimeMergedData,
    basis_functions: BasisFunctions,
    site_data: Mapping[str, xr.Dataset],
    data_args: Mapping[str, Any] | None = None,
    *,
    domain: str | None = None,
    start_date: str | None = None,
    bc_freq: str | None = None,
    min_error: inversion_preparation.MinErrorConfig = 0.0,
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
    """
    if data_args is not None:
        domain = data_args["domain"]
        start_date = data_args["start_date"]
        bc_freq = data_args["bc_freq"]
        min_error = data_args["min_error"]
        use_bc = data_args["use_bc"]

        min_error_options = normalise_min_error_options(data_args["min_error_options"])

    owned_site_data = {site: dataset.copy(deep=False) for site, dataset in site_data.items()}
    inversion_preparation._set_domain_attrs(owned_site_data, merged.sites, domain)
    # These inverse-model settings are materialized into labelled arrays here
    # to preserve the existing numerical contract while preparation is split.
    with timed("rhime.prepare_inputs.make_inv_inputs", sites=len(merged.sites)):
        inv_inputs = inversion_preparation._make_inv_inputs(
            fp_data=owned_site_data,
            sites=merged.sites,
            start_date=start_date,
            bc_freq=bc_freq,
            min_error=min_error,
            calculate_min_error=None,
            min_error_per_site=False if min_error_options is None else min_error_options["by_site"],
        )
    inv_inputs = inversion_preparation._scale_satellite_bc_sensitivity_to_column_signal(
        inv_inputs,
        sites=merged.sites,
        platform=merged.platform,
        observation_max_level=merged.site_options.max_level,
        footprint_max_level=tuple(
            owned_site_data[site].attrs.get("footprint_max_level") for site in merged.sites
        ),
    )
    inversion_preparation._warn_for_nan_inputs(inv_inputs, use_bc=use_bc)
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
    site_metadata = inversion_preparation._make_site_metadata(
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
