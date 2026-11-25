"""Prepare coherent CO2 inputs from acquired observations and native transport."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import xarray as xr

from openghg_inversions.basis import prepare_basis
from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact
from openghg_inversions.basis.boundary import BoundaryBasis
from openghg_inversions.basis.covariance_products import PreserveBucketProlongation
from openghg_inversions.boundary_sensitivity import BoundaryRepresentation
from openghg_inversions.coherent_reduction import reduce_native_gaussian
from openghg_inversions.inversion_data import MergedData, select_boundary_representation
from openghg_inversions.inversion_data.observation_errors import prepare_observation_errors
from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.rhime.preparation import (
    assemble_inputs,
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
    end_date: str,
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
        end_date: Requested window end.
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

    Raises:
        ValueError: If acquisition facts or scientific inputs conflict with
            the request, or no usable observations remain.
    """
    merged_data.validate_for_preparation(
        species="co2", domain=domain, start_date=start_date, end_date=end_date, split_by_sectors=False,
    )
    if set(merged_data.flux_data) != set(flux_sources):
        raise ValueError(
            "Acquired CO2 flux sources must match the configured source set: "
            f"acquired {sorted(merged_data.flux_data)!r}, configured {sorted(set(flux_sources))!r}."
        )
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
    sensitivities = {
        site: dataset.drop_vars(("mf_mod", "mf_mod_sectoral"), errors="ignore")
        for site, dataset in filtered.items()
    }
    if use_bc:
        sensitivities = bc_sensitivity(sensitivities, bc_basis=bc_basis)
    else:
        # Assembly reads the reference and marker as requests for centered sensitivities.
        for site, dataset in sensitivities.items():
            view = dataset.drop_vars("bc_reference", errors="ignore")
            view.attrs = {key: value for key, value in dataset.attrs.items() if key != "boundary_representation"}
            sensitivities[site] = view
    canonical = assemble_inputs(
        sensitivities, basis_functions, site_options=site_options,
        domain=domain, start_date=start_date, use_bc=use_bc, flux_response="fp_x_flux",
    )
    inputs = canonical.inv_inputs
    if not inputs.sizes["nmeasure"]:
        raise ValueError("No usable observations remain for coherent CO2 reduction.")

    native_sensitivity = inputs.fp_x_flux.rename(nmeasure="observation").assign_attrs(units=inputs.mf.attrs["units"])
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
    canonical = replace(canonical, inv_inputs=inputs.drop_vars("fp_x_flux"))
    prepared = prepare_co2_inputs(
        canonical, reduction, aggregation_error_rank=aggregation_error_rank,
        bc_centering_weights=bc_centering_weights,
        provenance={
            "domain": domain, "start_date": start_date, "end_date": end_date,
            "native_preparation": {
                "boundary_representation": boundary_representation,
                "native_prior": asdict(prior),
                "flux_sources": list(flux_sources),
                "basis_algorithm": basis_algorithm,
            },
        },
    )
    affine = produce_bucket_affine_flux_map(prepared, native_mean, prepared_inputs_id="unsaved")
    return prepared, affine


__all__ = ["prepare_co2_from_merged_data"]
