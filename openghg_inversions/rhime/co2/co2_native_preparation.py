"""Build a CO2 preparation from configured acquisition and native choices.

The steps run in scientific order: acquire (or accept) merged surface data,
select the boundary representation the configured choice needs, select
observations, derive observation errors, build the basis and canonical inputs,
reduce the native Gaussian prior coherently, and form the CO2 artifact with its
affine companion. Saving belongs to :mod:`openghg_inversions.rhime.co2.handoffs`.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.basis import prepare_basis
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact
from openghg_inversions.basis.covariance_products import PreserveBucketProlongation
from openghg_inversions.coherent_reduction import reduce_native_gaussian
from openghg_inversions.inversion_data import MergedData, PreparedInputs, select_boundary_representation
from openghg_inversions.inversion_data.observation_errors import prepare_observation_errors
from openghg_inversions.models.boundary import Preparation
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

if TYPE_CHECKING:
    from .configuration import Co2NativePreparationConfig


def prepare_native_co2(
    native: Co2NativePreparationConfig,
    *,
    domain: str,
    start_date: str,
    end_date: str,
    boundary: Preparation,
    merged_data: MergedData | None = None,
) -> tuple[Co2PreparedInputs, AffineFluxMapArtifact]:
    """Acquire and prepare one coherent CO2 artifact and its affine companion.

    Selected basis and weight files are read before acquisition. Without
    ``merged_data``, CO2 surface data are acquired for the requested window;
    supplied merged data are checked against that window instead. One merged
    dataset serves either boundary representation.

    All flux sources share one dimensionless native scaling field, so their
    summed response ``fp_x_flux`` is the native sensitivity. The first
    configured source weights basis design.

    Args:
        native: Configured acquisition and native preparation choices.
        domain: Transport domain.
        start_date: Requested window start.
        end_date: Requested window end.
        boundary: Boundary preparation the configured model needs.
        merged_data: Optional borrowed merged data, skipping acquisition.

    Returns:
        The CO2 prepared inputs and their unsaved affine companion.

    Raises:
        ValueError: If supplied data conflict with the request, cannot provide
            the boundary representation, or preparation contracts fail.
    """
    # Resolve file-backed resources before any acquisition.
    basis_inputs = resolve_basis_inputs(native.basis)
    bc_basis = resolve_bc_basis(native.basis)
    weights = None
    if native.bc_centering_weights_path is not None:
        weights = _load_centering_weights(native.bc_centering_weights_path)

    acquisition = dict(native.acquisition)
    flux_sources = tuple(acquisition["flux_sources"])
    use_bc = bool(acquisition.get("use_bc", True))
    if merged_data is None:
        merged_data = MergedData.from_options(
            species="co2", domain=domain, start_date=start_date, end_date=end_date, **acquisition,
        )
    else:
        merged_data.validate_for_preparation(
            species="co2", domain=domain, start_date=start_date, end_date=end_date, split_by_sectors=False,
        )

    merged_data.validate_scientific_inputs(
        split_by_sectors=False, use_bc=use_bc, flux_sources=flux_sources,
        require_footprints=basis_inputs.get("basis_path") is None and basis_inputs.get("basis_partition") is None,
    )
    site_data = select_boundary_representation(merged_data.site_data, boundary.representation)
    filtered, site_options = filter_observations(site_data, merged_data.site_options, filters=native.filters)
    filtered = prepare_observation_errors(filtered, averaging_error=native.averaging_error)

    runtime_flux = resolve_runtime_flux(merged_data.flux_data, explicit_surface=True)
    basis_functions = prepare_basis(
        site_data=filtered, runtime_flux=runtime_flux,
        weighting_flux=merged_data.flux_data[flux_sources[0]]["flux"],
        domain=domain, start_date=start_date, basis_algorithm=native.basis.basis_algorithm,
        nbasis=native.basis.nbasis, fix_outer_regions=native.basis.fix_basis_outer_regions,
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
    prior = native.native_prior
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
        canonical, reduction, aggregation_error_rank=native.aggregation_error_rank,
        bc_centering_weights=weights,
        provenance={"native_preparation": {
            "boundary_representation": boundary.representation,
            "native_prior": asdict(prior),
            "flux_sources": list(flux_sources),
            "basis_algorithm": native.basis.basis_algorithm,
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


def _load_centering_weights(path: Path) -> xr.DataArray:
    """Load and check the one ``bc_centering_weights`` array before acquisition starts."""
    with xr.open_dataset(path) as dataset:
        if "bc_centering_weights" not in dataset:
            raise ValueError(f"{path} must contain a bc_centering_weights variable.")
        weights = dataset["bc_centering_weights"].load()
    validate_centering_weights(weights.transpose("bc_curtain", "bc_period"))
    return weights


__all__ = ["prepare_native_co2"]
