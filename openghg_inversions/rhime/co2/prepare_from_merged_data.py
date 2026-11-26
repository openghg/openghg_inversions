"""Prepare coherent CO2 inputs from acquired observations and native transport."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr

from openghg_inversions.basis import prepare_basis
from openghg_inversions.basis._helpers import bc_sensitivity
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact
from openghg_inversions.basis.boundary import BoundaryBasis
from openghg_inversions.basis.covariance_products import PreserveBucketProlongation
from openghg_inversions.basis.operators import MultiSourceBucketBasisOperator
from openghg_inversions.boundary_sensitivity import BoundaryRepresentation
from openghg_inversions.coherent_reduction import reduce_native_gaussian
from openghg_inversions.inversion_data import MergedData, select_boundary_representation
from openghg_inversions.inversion_data.observation_errors import prepare_observation_errors
from openghg_inversions.models.boundary import Choice, boundary_preparation
from openghg_inversions.models.boundary.contracts import validate_centering_weights
from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.rhime.preparation import (
    assemble_inputs,
    build_sensitivities,
    filter_observations,
    resolve_basis_inputs,
    resolve_bc_basis,
)

from .co2_affine_output import produce_bucket_affine_flux_map
from .co2_preparation import Co2PreparedInputs, prepare_co2_inputs
from .co2_native_preparation import (
    MultiSourceNativeResources,
    _multisource_basis,
    _native_source_sensitivity,
    _sectoral_response_only,
    prepare_multisource_native_co2_inputs,
)
from .configuration import Co2MultiSourceNativePrior, Co2NativePreparationConfig


@dataclass(frozen=True)
class Co2PreparationResources:
    """Resolved resources for scientific preparation of acquired CO2 data.

    Attributes:
        basis_inputs: Resolved flux basis arguments for ``prepare_basis``.
        bc_basis: Loaded boundary basis or a builder that uses site coordinates.
        bc_centering_weights: Loaded weights for weighted boundary centering.
        multisource: Class labels, retained regions and calibration targets for per-source priors.
    """

    basis_inputs: dict[str, Any]
    bc_basis: BoundaryBasis
    bc_centering_weights: xr.DataArray | None
    multisource: MultiSourceNativeResources | None = None


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
    prior = config.native_prior
    return Co2PreparationResources(
        basis_inputs=resolve_basis_inputs(config.basis),
        bc_basis=resolve_bc_basis(config.bc_basis, use_bc=config.use_bc),
        bc_centering_weights=_load_centering_weights(config.bc_centering_weights_path),
        multisource=MultiSourceNativeResources.load(prior) if isinstance(prior, Co2MultiSourceNativePrior) else None,
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

    A shared prior uses the summed native response. Per-source priors jointly
    reduce the source-resolved native responses. The first source weights basis design.
    The same operation handles supplied and freshly acquired merged data.

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
        site_data = select_boundary_representation(site_data, representation)
    filtered, site_options = filter_observations(site_data, merged_data.site_options, filters=config.filters)
    filtered = prepare_observation_errors(filtered, averaging_error=config.averaging_error)

    runtime_flux = resolve_runtime_flux(merged_data.flux_data, explicit_surface=True)
    prior = config.native_prior
    if isinstance(prior, Co2MultiSourceNativePrior):
        basis_functions = _multisource_basis(
            config, basis_inputs, filtered, merged_data, runtime_flux, flux_sources=flux_sources,
            domain=domain, start_date=start_date,
        )
        sensitivities = build_sensitivities(
            _sectoral_response_only(filtered), basis_functions, flux_sources=flux_sources,
            use_bc=use_bc, bc_basis=resources.bc_basis, multisector=True,
        )
        if not use_bc:
            sensitivities = _without_centered_boundary(sensitivities)
        canonical = assemble_inputs(
            sensitivities, basis_functions, site_options=site_options,
            domain=domain, start_date=start_date, use_bc=use_bc,
        )
        if not canonical.inv_inputs.sizes["nmeasure"]:
            raise ValueError("No usable observations remain for coherent CO2 reduction.")
        operator = cast(MultiSourceBucketBasisOperator, basis_functions.operator)
        sensitivity = _native_source_sensitivity(filtered, canonical, operator.source_labels)
        return prepare_multisource_native_co2_inputs(
            canonical, prior, native_sensitivity=sensitivity, resources=resources.multisource,
            aggregation_error_rank=config.aggregation_error_rank,
            bc_centering_weights=resources.bc_centering_weights,
            provenance={
                "domain": domain, "start_date": start_date, "end_date": end_date,
                "native_preparation": {
                    "boundary_representation": representation, "flux_sources": list(flux_sources),
                    "basis_algorithm": basis.basis_algorithm,
                },
            },
        )
    basis_functions = prepare_basis(
        site_data=filtered, runtime_flux=runtime_flux,
        weighting_flux=merged_data.flux_data[flux_sources[0]]["flux"],
        domain=domain, start_date=start_date, basis_algorithm=basis.basis_algorithm,
        nbasis=basis.nbasis, fix_outer_regions=basis.fix_basis_outer_regions, seed=basis.basis_seed,
        **basis_inputs,
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
        sensitivities = _without_centered_boundary(sensitivities)
    canonical = assemble_inputs(
        sensitivities, basis_functions, site_options=site_options,
        domain=domain, start_date=start_date, use_bc=use_bc, flux_response="fp_x_flux",
    )
    inputs = canonical.inv_inputs
    if not inputs.sizes["nmeasure"]:
        raise ValueError("No usable observations remain for coherent CO2 reduction.")

    native_sensitivity = inputs.fp_x_flux.reset_coords(drop=True).rename(nmeasure="observation")
    native_sensitivity = native_sensitivity.assign_attrs(units=inputs.mf.attrs["units"])
    native_mean = xr.DataArray(
        np.full((native_sensitivity.sizes["lat"], native_sensitivity.sizes["lon"]), prior.mean),
        dims=("lat", "lon"), coords={"lat": native_sensitivity.lat, "lon": native_sensitivity.lon},
        attrs={"units": "1"},
    )
    covariance = SeparableExponentialCovariance(
        native_sensitivity.lat, native_sensitivity.lon, sigma=prior.stdev,
        correlation_length=prior.correlation_length,
        latitude_correlation_length=prior.latitude_correlation_length,
        longitude_correlation_length=prior.longitude_correlation_length,
    )
    reduction = reduce_native_gaussian(
        covariance=covariance, strategy=PreserveBucketProlongation(basis_functions.operator.basis_matrix),
        state_dim=basis_functions.operator.meta.state_dim, native_mean=native_mean,
        native_sensitivity=native_sensitivity, observation_dim="observation",
    )
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


def _without_centered_boundary(site_data: dict[str, xr.Dataset]) -> dict[str, xr.Dataset]:
    """Omit centered boundary markers when assembly has no boundary inputs."""
    result = {}
    for site, dataset in site_data.items():
        view = dataset.drop_vars("bc_reference", errors="ignore")
        view.attrs = {key: value for key, value in dataset.attrs.items() if key != "boundary_representation"}
        result[site] = view
    return result


__all__ = [
    "Co2PreparationResources",
    "prepare_co2_from_merged_data",
    "resolve_co2_preparation_resources",
]
