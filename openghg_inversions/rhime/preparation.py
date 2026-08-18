"""Named scientific preparation stages for RHIME recipes.

The functions in this module form the backend-neutral preparation spine used
by :func:`openghg_inversions.rhime.run_rhime` and copied project runners:

``retrieve/reload -> filter -> basis -> sensitivities -> labelled assembly``.

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

from collections.abc import Mapping
from typing import Any

import xarray as xr

from openghg_inversions._timing import log_timing, timed
from openghg_inversions.basis import make_basis_functions
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.inversion_data import preparation as inversion_preparation
from openghg_inversions.model_error import normalise_min_error_options

__all__ = [
    "assemble_rhime_inputs",
    "build_rhime_basis",
    "build_rhime_sensitivities",
    "filter_rhime_observations",
    "retrieve_or_reload_rhime_data",
]


def retrieve_or_reload_rhime_data(
    data_args: Mapping[str, Any],
    *,
    multisector: bool,
    merged_data: RhimeMergedData | None = None,
) -> RhimeMergedData:
    """Retrieve, reload, or accept externally supplied merged RHIME data.

    Passing ``merged_data`` is the explicit no-I/O path.  The object remains
    borrowed and is returned unchanged after a sector-layout compatibility check.
    Otherwise this stage may read OpenGHG stores or a local merged artifact,
    optionally write merged data, sanitize flux arrays, print progress, and
    emit warnings.  ``data_args`` is never mutated.
    """
    if merged_data is not None:
        stored_multisector = bool(merged_data.fp_all.get(".split_by_sectors", False))
        if stored_multisector != multisector:
            raise ValueError(
                "External RHIME merged data has an incompatible sector layout: "
                f"artifact split_by_sectors={stored_multisector!r}, "
                f"runner multisector={multisector!r}."
            )
        return merged_data

    with timed(
        "rhime.prepare_inputs.merged_data",
        sites=len(data_args["sites"]),
        split_by_sectors=multisector,
    ):
        return inversion_preparation._prepare_merged_data(
            species=data_args["species"],
            sites=data_args["sites"],
            domain=data_args["domain"],
            averaging_period=data_args["averaging_period"],
            start_date=data_args["start_date"],
            end_date=data_args["end_date"],
            output_name=data_args["output_name"],
            flux_sources=data_args["flux_sources"],
            split_by_sectors=multisector,
            bc_store=data_args["bc_store"],
            obs_store=data_args["obs_store"],
            footprint_store=data_args["footprint_store"],
            emissions_store=data_args["emissions_store"],
            met_model=data_args["met_model"],
            fp_model=data_args["fp_model"],
            fp_height=data_args["fp_height"],
            fp_species=data_args["fp_species"],
            inlet=data_args["inlet"],
            instrument=data_args["instrument"],
            max_level=data_args["max_level"],
            calibration_scale=data_args["calibration_scale"],
            obs_data_level=data_args["obs_data_level"],
            platform=data_args["platform"],
            use_tracer=data_args["use_tracer"],
            use_bc=data_args["use_bc"],
            bc_input=data_args["bc_input"],
            averaging_error=data_args["averaging_error"],
            reload_merged_data=data_args["reload_merged_data"],
            save_merged_data=data_args["save_merged_data"],
            merged_data_dir=data_args["merged_data_dir"],
            merged_data_name=data_args["merged_data_name"],
            flux_non_finite_check=data_args["flux_non_finite_check"],
        )


def filter_rhime_observations(
    merged: RhimeMergedData,
    data_args: Mapping[str, Any],
) -> RhimeMergedData:
    """Filter borrowed observations and remove empty sites with aligned metadata.

    The stage may compute site data if a filter cannot operate lazily.  It
    returns a new merged-data handoff when filtering changes data and never
    constructs basis functions or model inputs.
    """
    filters = data_args["filters"]
    with timed(
        "rhime.prepare_inputs.obs_filtering",
        sites=len(merged.sites),
        filters=filters is not None,
    ):
        return inversion_preparation._filter_merged_inversion_data(merged=merged, filters=filters)


def build_rhime_basis(
    merged: RhimeMergedData,
    data_args: Mapping[str, Any],
) -> BasisFunctions:
    """Load or fit the retained RHIME basis for filtered observations.

    This stage may read or write basis artifacts and may execute the selected
    basis algorithm.  It treats ``merged`` as borrowed and does not build
    sensitivities.
    """
    basis_algorithm = data_args["basis_algorithm"]
    nbasis = data_args["nbasis"]
    fp_basis_case = data_args["fp_basis_case"]
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
            basis_directory=data_args["basis_directory"],
            country_directory=data_args["country_directory"],
            fp_all=merged.fp_all,
            species=data_args["species"],
            domain=data_args["domain"],
            start_date=data_args["start_date"],
            fix_outer_regions=data_args["fix_basis_outer_regions"],
            emissions_name=data_args["flux_sources"],
            outputname=data_args["output_name"],
            output_path=data_args["basis_output_path"],
        )


def build_rhime_sensitivities(
    merged: RhimeMergedData,
    basis_functions: BasisFunctions,
    data_args: Mapping[str, Any],
    *,
    multisector: bool,
) -> dict[str, xr.Dataset]:
    """Construct labelled flux and optional boundary-condition sensitivities.

    The stage creates per-site dataset copies, computes the basis projection,
    and may load boundary-condition basis data.  ``merged`` and
    ``basis_functions`` remain borrowed.
    """
    with timed("rhime.prepare_inputs.footprint_sensitivity_total", sites=len(merged.sites)):
        return inversion_preparation._rhime_site_data_from_basis_functions(
            merged=merged,
            basis_functions=basis_functions,
            domain=data_args["domain"],
            split_by_sectors=multisector,
            flux_sources=data_args["flux_sources"],
            use_bc=data_args["use_bc"],
            bc_basis_case=data_args["bc_basis_case"],
            bc_basis_directory=inversion_preparation._bc_basis_directory_arg(
                data_args["bc_basis_directory"]
            ),
        )


def assemble_rhime_inputs(
    merged: RhimeMergedData,
    basis_functions: BasisFunctions,
    site_data: Mapping[str, xr.Dataset],
    data_args: Mapping[str, Any],
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
    owned_site_data = {site: dataset.copy(deep=False) for site, dataset in site_data.items()}
    inversion_preparation._set_domain_attrs(owned_site_data, merged.sites, data_args["domain"])
    # These inverse-model settings are materialized into labelled arrays here
    # to preserve the existing numerical contract while preparation is split.
    min_error_options = normalise_min_error_options(data_args["min_error_options"])
    with timed("rhime.prepare_inputs.make_inv_inputs", sites=len(merged.sites)):
        inv_inputs = inversion_preparation._make_inv_inputs(
            fp_data=owned_site_data,
            sites=merged.sites,
            start_date=data_args["start_date"],
            bc_freq=data_args["bc_freq"],
            min_error=data_args["min_error"],
            calculate_min_error=None,
            min_error_per_site=min_error_options["by_site"],
        )
    inv_inputs = inversion_preparation._scale_satellite_bc_sensitivity_to_column_signal(
        inv_inputs,
        sites=merged.sites,
        platform=merged.platform,
    )
    inversion_preparation._warn_for_nan_inputs(inv_inputs, use_bc=data_args["use_bc"])
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
    return RhimePreparedInputs(
        inv_inputs=inv_inputs,
        basis_functions=basis_functions,
        site_metadata=inversion_preparation._make_site_metadata(
            sites=merged.sites,
            averaging_period=merged.averaging_period,
        ),
    )
