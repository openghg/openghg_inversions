"""Prepare coherent CO2 inputs from acquired observations and native transport."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from openghg_inversions.basis import prepare_basis
from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact
from openghg_inversions.basis.boundary import BoundaryBasis
from openghg_inversions.basis.basis_functions import basis_functions_from_flat_basis
from openghg_inversions.basis.covariance_products import PreserveBucketProlongation
from openghg_inversions.boundary_sensitivity import BoundaryRepresentation
from openghg_inversions.coherent_reduction import reduce_native_gaussian
from openghg_inversions.inversion_data import MergedData, select_boundary_representation
from openghg_inversions.inversion_data._site_options import is_satellite_platform
from openghg_inversions.inversion_data.observation_errors import prepare_observation_errors
from openghg_inversions.models.boundary import Choice, boundary_preparation
from openghg_inversions.models.boundary.contracts import validate_centering_weights
from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.source_covariance import IndependentSourceCovariance
from openghg_inversions.rhime.preparation import (
    assemble_inputs,
    filter_observations,
    resolve_basis_inputs,
    resolve_bc_basis,
)

from .co2_affine_output import produce_bucket_affine_flux_map
from .co2_preparation import Co2PreparedInputs, prepare_co2_inputs
from .configuration import Co2NativePreparationConfig


@dataclass(frozen=True)
class Co2PreparationResources:
    """Resolved resources for scientific preparation of acquired CO2 data.

    Attributes:
        basis_inputs: Resolved flux basis arguments for ``prepare_basis``.
        bc_basis: Loaded boundary basis or a builder that uses site coordinates.
        bc_centering_weights: Loaded weights for weighted boundary centering.
    """

    basis_inputs: dict[str, Any]
    bc_basis: BoundaryBasis
    bc_centering_weights: xr.DataArray | None


def _load_centering_weights(path: Path | None) -> xr.DataArray | None:
    """Load and validate the optional weights for the prepared boundary states."""
    if path is None:
        return None
    with xr.open_dataset(path) as dataset:
        if "bc_centering_weights" not in dataset:
            raise ValueError(f"{path} must contain a bc_centering_weights variable.")
        weights = dataset["bc_centering_weights"].load()
    validate_centering_weights(weights.transpose("bc_curtain", "bc_period"))
    return weights


def resolve_co2_preparation_resources(config: Co2NativePreparationConfig) -> Co2PreparationResources:
    """Resolve selected scientific resources before retrieving observations and transport.

    Built-in boundary basis builders are evaluated later against site coordinates.

    Args:
        config: Resolved CO2 choices for native scientific preparation.

    Returns:
        Resources to pass to scientific preparation.

    Raises:
        OSError: If a selected file is missing or unreadable.
        KeyError: If a selected basis map or partition lacks its required variable.
        ValueError: If selected basis inputs or centering weights are invalid.
    """
    return Co2PreparationResources(
        basis_inputs=resolve_basis_inputs(config.basis),
        bc_basis=resolve_bc_basis(config.bc_basis, use_bc=config.use_bc),
        bc_centering_weights=_load_centering_weights(config.bc_centering_weights_path),
    )


def prepare_co2_from_merged_data(
    merged_data: MergedData,
    config: Co2NativePreparationConfig,
    *,
    flux_sources: tuple[str, ...],
    resources: Co2PreparationResources,
    domain: str,
    start_date: str,
    end_date: str,
    boundary: Choice | None,
    boundary_representation: BoundaryRepresentation | None,
) -> tuple[Co2PreparedInputs, AffineFluxMapArtifact]:
    """Build coherent CO2 inputs and their affine companion from acquired data.

    A scalar prior SD uses the summed response and a shared native scaling
    field. Source-specific SDs retain separate responses and independent prior
    fields. The first source weights basis design in either case.

    Args:
        merged_data: Borrowed observations, transport and native flux.
        config: Resolved choices for native scientific preparation.
        flux_sources: Requested inventories in basis-weighting order.
        resources: Basis and centering resources resolved for these choices.
        domain: Requested transport domain.
        start_date: Requested window start.
        end_date: Requested window end.
        boundary: Configured boundary correction, if any.
        boundary_representation: Explicit representation, or None for the
            default required by the boundary correction.

    Returns:
        Prepared CO2 inputs and their unsaved affine companion.

    Raises:
        ValueError: If acquisition facts or scientific inputs conflict with
            the request, or no usable observations remain.
    """
    basis = config.basis
    use_bc = config.use_bc
    representation = boundary_preparation(boundary, boundary_representation).representation
    basis_inputs = resources.basis_inputs
    prior = config.native_prior
    stdev = prior.stdev
    split_by_sectors = isinstance(stdev, dict)
    if split_by_sectors and set(stdev) != set(flux_sources):
        raise ValueError("Native prior standard deviations must specify exactly one value per flux source.")

    merged_data.validate_for_preparation(
        species="co2", domain=domain, start_date=start_date, end_date=end_date, split_by_sectors=split_by_sectors,
    )
    if set(merged_data.flux_data) != set(flux_sources):
        raise ValueError(
            "Acquired CO2 flux sources must match the configured source set: "
            f"acquired {sorted(merged_data.flux_data)!r}, configured {sorted(set(flux_sources))!r}."
        )
    merged_data.validate_scientific_inputs(
        split_by_sectors=split_by_sectors, use_bc=use_bc, flux_sources=flux_sources,
        require_footprints=basis_inputs.get("basis_path") is None and basis_inputs.get("basis_partition") is None,
    )
    site_data = merged_data.site_data
    if use_bc:
        site_data = select_boundary_representation(site_data, representation)
    filtered, site_options = filter_observations(site_data, merged_data.site_options, filters=config.filters)
    filtered = prepare_observation_errors(filtered, averaging_error=config.averaging_error)

    explicit_surface = not any(map(is_satellite_platform, site_options.platform))
    runtime_flux = resolve_runtime_flux(
        merged_data.flux_data, split_by_sectors=split_by_sectors, explicit_surface=explicit_surface,
    )
    basis_functions = prepare_basis(
        site_data=filtered, runtime_flux=runtime_flux,
        weighting_flux=merged_data.flux_data[flux_sources[0]]["flux"],
        domain=domain, start_date=start_date, basis_algorithm=basis.basis_algorithm,
        nbasis=basis.nbasis, fix_outer_regions=basis.fix_basis_outer_regions, seed=basis.basis_seed,
        **basis_inputs,
    )
    if split_by_sectors:
        partitions = basis_functions.flat_basis()
        if isinstance(partitions, xr.DataArray):
            basis_functions = basis_functions_from_flat_basis(
                basis_flat=dict.fromkeys(flux_sources, partitions),
                flux=runtime_flux, metadata=basis_functions.metadata,
            )
    sensitivities = {}
    for site, dataset in filtered.items():
        # Avoid gathering unused grid diagnostics with the native response.
        grid_coordinates = [
            name for name, coord in dataset.coords.items()
            if name not in dataset.xindexes and {"lat", "lon"}.intersection(coord.dims)
        ]
        sensitivities[site] = dataset.drop_vars(("mf_mod", "mf_mod_sectoral", *grid_coordinates), errors="ignore")
    if use_bc:
        sensitivities = bc_sensitivity(sensitivities, bc_basis=resources.bc_basis)
    else:
        # Assembly reads the reference and marker as requests for centered sensitivities.
        for site, dataset in sensitivities.items():
            view = dataset.drop_vars("bc_reference", errors="ignore")
            view.attrs = {key: value for key, value in dataset.attrs.items() if key != "boundary_representation"}
            sensitivities[site] = view
    response = "fp_x_flux_sectoral" if split_by_sectors else "fp_x_flux"
    canonical = assemble_inputs(
        sensitivities, basis_functions, site_options=site_options,
        domain=domain, start_date=start_date, use_bc=use_bc, flux_response=response,
    )
    inputs = canonical.inv_inputs
    if not inputs.sizes["nmeasure"]:
        raise ValueError("No usable observations remain for coherent CO2 reduction.")

    native_sensitivity = inputs[response].reset_coords(drop=True).rename(nmeasure="observation")
    native_sensitivity = native_sensitivity.assign_attrs(units=inputs.mf.attrs["units"])
    operator = basis_functions.operator
    covariance_options = {
        "correlation_length": prior.correlation_length,
        "latitude_correlation_length": prior.latitude_correlation_length,
        "longitude_correlation_length": prior.longitude_correlation_length,
    }
    if isinstance(stdev, dict):
        source_labels = list(basis_functions.source_labels or ())
        native_sensitivity = native_sensitivity.sel(source=source_labels).rename(source="native_source")
        covariance = IndependentSourceCovariance({
            source: SeparableExponentialCovariance(
                native_sensitivity.lat, native_sensitivity.lon, sigma=stdev[source], **covariance_options,
            )
            for source in source_labels
        })
    else:
        covariance = SeparableExponentialCovariance(
            native_sensitivity.lat, native_sensitivity.lon, sigma=stdev, **covariance_options,
        )
    native_dims = covariance.native_dims
    native_mean = xr.full_like(
        native_sensitivity.isel(observation=0, drop=True).transpose(*native_dims), prior.mean, dtype=np.float64,
    ).assign_attrs(units="1")
    prolongation = operator._native_prolongation(native_mean, native_dims=native_dims)
    reduction = reduce_native_gaussian(
        covariance=covariance, strategy=PreserveBucketProlongation(prolongation),
        state_dim=operator.meta.state_dim, native_mean=native_mean,
        native_sensitivity=native_sensitivity, observation_dim="observation",
    )
    if not split_by_sectors:
        canonical = replace(canonical, inv_inputs=inputs.drop_dims(("lat", "lon")))
    prepared = prepare_co2_inputs(
        canonical, reduction, aggregation_error_rank=config.aggregation_error_rank,
        bc_centering_weights=resources.bc_centering_weights,
        provenance={
            "domain": domain, "start_date": start_date, "end_date": end_date,
            "native_preparation": {
                "boundary_representation": representation,
                "native_prior": asdict(prior),
                "flux_sources": list(flux_sources),
                "basis_algorithm": basis.basis_algorithm,
            },
        },
    )
    affine = produce_bucket_affine_flux_map(prepared, native_mean, prepared_inputs_id="unsaved")
    return prepared, affine


__all__ = [
    "Co2PreparationResources",
    "prepare_co2_from_merged_data",
    "resolve_co2_preparation_resources",
]
