"""Prepare coherent CO2 inputs from acquired observations and native transport."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.basis import prepare_basis
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact
from openghg_inversions.basis.boundary import BoundaryBasis
from openghg_inversions.basis.covariance_products import PreserveBucketProlongation
from openghg_inversions.boundary_sensitivity import BoundaryRepresentation
from openghg_inversions.coherent_reduction import reduce_native_gaussian
from openghg_inversions.inversion_data import MergedData, PreparedInputs, select_boundary_representation
from openghg_inversions.inversion_data.observation_errors import prepare_observation_errors
from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.rhime.preparation import (
    assemble_inputs,
    build_sensitivities,
    filter_observations,
)

from .co2_affine_output import produce_bucket_affine_flux_map
from .co2_preparation import Co2PreparedInputs, prepare_co2_inputs
from .configuration import Co2NativePrior


def prepare_co2_from_merged_data(
    merged_data: MergedData,
    *,
    domain: str,
    start_date: str,
    flux_sources: tuple[str, ...],
    use_bc: bool,
    boundary_representation: BoundaryRepresentation,
    filters: Any,
    averaging_error: bool,
    basis_algorithm: str | None,
    nbasis: int,
    fix_outer_regions: bool,
    basis_inputs: Mapping[str, Any],
    bc_basis: BoundaryBasis,
    prior: Co2NativePrior,
    aggregation_error_rank: int | None,
    bc_centering_weights: xr.DataArray | None,
) -> tuple[Co2PreparedInputs, AffineFluxMapArtifact]:
    """Build a coherent CO2 preparation and affine companion from acquired data.

    All sources share a dimensionless native scaling field. Their summed
    response is the native sensitivity, and the first source weights basis design.

    Args:
        merged_data: Borrowed observations, transport and native flux.
        domain: Transport domain.
        start_date: Requested window start.
        flux_sources: Ordered source identities used for preparation.
        use_bc: Whether to construct boundary sensitivities.
        boundary_representation: Ordinary or centered prepared boundary inputs.
        filters: Observation filters applied before deriving errors.
        averaging_error: Include variability when deriving errors.
        basis_algorithm: Algorithm used to generate a flux partition.
        nbasis: Target number of regions for a generated partition.
        fix_outer_regions: Keep predefined outer regions fixed during generation.
        basis_inputs: Resolved options and resources for the flux basis.
        bc_basis: Resolved spatial boundary basis.
        prior: Mean and covariance choices for native flux scaling.
        aggregation_error_rank: Maximum stored rank, or None for exact covariance.
        bc_centering_weights: Loaded weights for weighted additive boundaries.

    Returns:
        Prepared CO2 inputs and their unsaved affine companion.
    """
    merged_data.validate_scientific_inputs(
        split_by_sectors=False, use_bc=use_bc, flux_sources=flux_sources,
        require_footprints=basis_inputs.get("basis_path") is None and basis_inputs.get("basis_partition") is None,
    )
    site_data = merged_data.site_data
    if use_bc:
        site_data = select_boundary_representation(site_data, boundary_representation)
    filtered, site_options = filter_observations(site_data, merged_data.site_options, filters=filters)
    filtered = prepare_observation_errors(filtered, averaging_error=averaging_error)

    runtime_flux = resolve_runtime_flux(merged_data.flux_data, explicit_surface=True)
    basis_functions = prepare_basis(
        site_data=filtered, runtime_flux=runtime_flux,
        weighting_flux=merged_data.flux_data[flux_sources[0]]["flux"],
        domain=domain, start_date=start_date, basis_algorithm=basis_algorithm,
        nbasis=nbasis, fix_outer_regions=fix_outer_regions,
        **basis_inputs,
    )
    sensitivities = build_sensitivities(
        filtered, basis_functions, flux_sources=flux_sources, use_bc=use_bc, bc_basis=bc_basis, multisector=False,
    )
    canonical = assemble_inputs(
        sensitivities, basis_functions, site_options=site_options,
        domain=domain, start_date=start_date, use_bc=use_bc,
    )

    native_sensitivity = _native_sensitivity(filtered, canonical)
    native_mean = xr.DataArray(
        np.full((native_sensitivity.sizes["lat"], native_sensitivity.sizes["lon"]), prior.mean),
        dims=("lat", "lon"), coords={"lat": native_sensitivity.lat, "lon": native_sensitivity.lon},
        attrs={"units": "1"},
    )
    covariance = SeparableExponentialCovariance(
        native_sensitivity.lat, native_sensitivity.lon, sigma=prior.sigma,
        correlation_length=prior.correlation_length,
        latitude_correlation_length=prior.latitude_correlation_length,
        longitude_correlation_length=prior.longitude_correlation_length,
    )
    reduction = reduce_native_gaussian(
        covariance=covariance, strategy=PreserveBucketProlongation(basis_functions.operator.basis_matrix),
        state_dim=basis_functions.operator.meta.state_dim, native_mean=native_mean,
        native_sensitivity=native_sensitivity, observation_dim="observation",
    )
    prepared = prepare_co2_inputs(
        canonical, reduction, aggregation_error_rank=aggregation_error_rank,
        bc_centering_weights=bc_centering_weights,
        provenance={"native_preparation": {
            "boundary_representation": boundary_representation,
            "native_prior": asdict(prior),
            "flux_sources": list(flux_sources),
            "basis_algorithm": basis_algorithm,
        }},
    )
    affine = produce_bucket_affine_flux_map(prepared, native_mean, prepared_inputs_id="unsaved")
    return prepared, affine


def _native_sensitivity(site_data: dict[str, xr.Dataset], canonical: PreparedInputs) -> xr.DataArray:
    """Gather each retained row's native ``fp_x_flux`` in canonical observation order.

    Canonical rows are grouped by site in retained-site order, so gathering
    each site's retained times in turn reproduces the ``nmeasure`` labels.
    """
    index = cast(pd.MultiIndex, canonical.inv_inputs.indexes["nmeasure"])
    sites, times = index.get_level_values("site"), index.get_level_values("time")
    parts = [
        site_data[site]["fp_x_flux"].sel(time=times[sites == site]).reset_coords(drop=True)
        for site in pd.unique(sites)
    ]
    gathered = xr.concat(parts, dim="time", join="exact").transpose("time", "lat", "lon")
    if not pd.MultiIndex.from_arrays([np.repeat(pd.unique(sites), [len(p.time) for p in parts]),
                                      gathered.indexes["time"]]).equals(index):
        raise ValueError("Canonical CO2 rows are not grouped by retained site.")
    return gathered.drop_vars("time").rename(time="observation").assign_coords(
        xr.Coordinates.from_pandas_multiindex(index, "observation")
    )


__all__ = ["prepare_co2_from_merged_data"]
