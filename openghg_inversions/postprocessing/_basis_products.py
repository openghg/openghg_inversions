"""Basis-backed postprocessing product helpers."""

from __future__ import annotations

from typing import cast

import xarray as xr

from openghg_inversions.array_ops import align_sparse_lat_lon, sparse_xr_dot
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.flux_sanitization import copy_flux_nonfinite_attrs, sanitize_flux_nonfinite

BASIS_RECONSTRUCTION_PATH_ATTR = "basis_reconstruction_path"
BASIS_ARTIFACT_SOURCE_OUTPUT_ATTR = "basis_artifact_source"
BASIS_ARTIFACT_PATH_OUTPUT_ATTR = "basis_artifact_path"

BASIS_RECONSTRUCTION_OPERATOR_BACKED = "operator-backed"

BASIS_ARTIFACT_SOURCE_GENERATED = "generated"
BASIS_ARTIFACT_SOURCE_LOADED_DATATREE = "loaded-datatree"
BASIS_ARTIFACT_SOURCE_LOADED_LEGACY_FLAT = "loaded-legacy-flat"

_OUTPUT_BASIS_ARTIFACT_SOURCE_LABELS = {
    "generated": BASIS_ARTIFACT_SOURCE_GENERATED,
    "datatree": BASIS_ARTIFACT_SOURCE_LOADED_DATATREE,
    "legacy_flat": BASIS_ARTIFACT_SOURCE_LOADED_LEGACY_FLAT,
}

_TRACE_STATE_DIMS = ("region", "nx")


def basis_artifact_output_label(basis_functions: BasisFunctions) -> str:
    """Return the stable output label for a retained basis artifact source."""
    source = getattr(basis_functions, "basis_artifact_source", None) or BASIS_ARTIFACT_SOURCE_GENERATED
    return _OUTPUT_BASIS_ARTIFACT_SOURCE_LABELS.get(source, source)


def add_basis_reconstruction_metadata(
    ds: xr.Dataset,
    basis_functions: BasisFunctions,
    *,
    reconstruction_path: str = BASIS_RECONSTRUCTION_OPERATOR_BACKED,
) -> xr.Dataset:
    """Attach stable basis provenance metadata to a derived output dataset."""
    result = ds.copy(deep=False)
    result.attrs = dict(result.attrs)
    result.attrs[BASIS_RECONSTRUCTION_PATH_ATTR] = reconstruction_path
    result.attrs[BASIS_ARTIFACT_SOURCE_OUTPUT_ATTR] = basis_artifact_output_label(basis_functions)
    basis_artifact_path = getattr(basis_functions, "basis_artifact_path", None)
    if basis_artifact_path is not None:
        result.attrs[BASIS_ARTIFACT_PATH_OUTPUT_ATTR] = basis_artifact_path
    return result


def _trace_state_dim(ds: xr.Dataset, basis_functions: BasisFunctions | None = None) -> str:
    """Return the trace state dimension used by a postprocessing dataset."""
    if basis_functions is not None and basis_functions.operator.meta.state_dim in ds.dims:
        return basis_functions.operator.meta.state_dim

    for dim in _TRACE_STATE_DIMS:
        if dim in ds.dims:
            return dim

    raise ValueError(f"Could not find a basis state dimension in dataset dims {tuple(ds.dims)}.")


def _to_operator_state_dim(state: xr.Dataset, basis_functions: BasisFunctions) -> xr.Dataset:
    """Rename a standard trace state dimension to the operator state dimension."""
    operator_state_dim = basis_functions.operator.meta.state_dim
    if operator_state_dim in state.dims:
        return state

    trace_state_dim = _trace_state_dim(state)
    return state.rename({trace_state_dim: operator_state_dim})


def _from_operator_state_dim(
    state: xr.DataArray, basis_functions: BasisFunctions, trace_state_dim: str
) -> xr.DataArray:
    """Rename the operator state dimension back to the trace state dimension."""
    operator_state_dim = basis_functions.operator.meta.state_dim
    if operator_state_dim != trace_state_dim and operator_state_dim in state.dims:
        return state.rename({operator_state_dim: trace_state_dim})
    return state


def _interpolate_dataset(
    state: xr.Dataset,
    basis_functions: BasisFunctions,
    *,
    weights: xr.DataArray | None,
) -> xr.Dataset:
    """Interpolate each data variable from basis state space to the grid."""
    operator_state = _to_operator_state_dim(state, basis_functions)
    data_vars = {}
    for name, data in operator_state.data_vars.items():
        interpolated = (
            basis_functions.interpolate(data)
            if weights is None
            else basis_functions.operator.interpolate(data, weights=weights)
        )
        interpolated.attrs = state[name].attrs
        data_vars[name] = interpolated
    result = xr.Dataset(data_vars, attrs=state.attrs)
    if weights is not None:
        return _transpose_flux_weighted_dataset(result)
    if "flux_time" in result.dims:
        return _transpose_inversion_grid_dataset(result)
    return result


def _transpose_flux_weighted_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Match the legacy dimension order for prior-flux-weighted grid products."""
    leading_dims = [dim for dim in ("flux_time", "lat", "lon", "latitude", "longitude") if dim in ds.dims]
    trailing_dims = [dim for dim in ds.dims if dim not in leading_dims]
    return ds.transpose(*leading_dims, *trailing_dims)


def _transpose_inversion_grid_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Match the legacy dimension order for inversion-grid flux products."""
    leading_dims = [dim for dim in ("lat", "lon", "latitude", "longitude", "flux_time") if dim in ds.dims]
    trailing_dims = [dim for dim in ds.dims if dim not in leading_dims]
    return ds.transpose(*leading_dims, *trailing_dims)


def _operator_region_flux_mean(basis_functions: BasisFunctions, flux: xr.DataArray) -> xr.DataArray:
    """Return mean prior flux in each basis region using the retained operator."""
    basis = align_sparse_lat_lon(basis_functions.operator.basis_matrix, flux)
    grid_dims = list(basis_functions.operator.meta.grid_dims)
    return ((basis * flux).sum(grid_dims) / basis.sum(grid_dims)).fillna(0.0)


def _sanitize_postprocessing_flux(flux: xr.DataArray, *, context: str) -> xr.DataArray:
    """Apply the late fallback non-finite policy for old retained artifacts."""
    return sanitize_flux_nonfinite(
        flux,
        context=context,
        warn=True,
    )


def _with_flux_nonfinite_metadata(ds: xr.Dataset, flux: xr.DataArray) -> xr.Dataset:
    """Propagate non-finite flux policy metadata onto output datasets."""
    return cast(xr.Dataset, copy_flux_nonfinite_attrs(ds, flux))


def _with_flux_nonfinite_dataarray_metadata(da: xr.DataArray, flux: xr.DataArray) -> xr.DataArray:
    """Propagate non-finite flux policy metadata onto output arrays."""
    return cast(xr.DataArray, copy_flux_nonfinite_attrs(da, flux))


def reconstruct_flux_stats(
    basis_functions: BasisFunctions,
    flux: xr.DataArray,
    stats_ds: xr.Dataset,
    *,
    report_flux_on_inversion_grid: bool,
) -> xr.Dataset:
    """Reconstruct gridded flux statistics with the retained basis operator."""
    flux = _sanitize_postprocessing_flux(flux, context="operator-backed flux reconstruction")
    if report_flux_on_inversion_grid:
        region_flux = _operator_region_flux_mean(basis_functions, flux)
        state = _to_operator_state_dim(stats_ds, basis_functions) * region_flux
        return _with_flux_nonfinite_metadata(_interpolate_dataset(state, basis_functions, weights=None), flux)

    return _with_flux_nonfinite_metadata(_interpolate_dataset(stats_ds, basis_functions, weights=flux), flux)


def reconstruct_scale_factor_stats(
    basis_functions: BasisFunctions,
    stats_ds: xr.Dataset,
) -> xr.Dataset:
    """Reconstruct gridded scale-factor statistics with the retained operator."""
    return _interpolate_dataset(stats_ds, basis_functions, weights=None)


def make_x_to_country_matrix(
    basis_functions: BasisFunctions,
    flux: xr.DataArray,
    x_trace: xr.Dataset,
    *,
    country_matrix: xr.DataArray,
    area_grid: xr.DataArray,
    sparse: bool = False,
) -> xr.DataArray:
    """Construct a basis-state to country-total matrix."""
    flux = _sanitize_postprocessing_flux(flux, context="operator-backed country flux reconstruction")
    trace_state_dim = _trace_state_dim(x_trace, basis_functions)
    basis = align_sparse_lat_lon(basis_functions.operator.basis_matrix, flux)
    basis = _from_operator_state_dim(basis, basis_functions, trace_state_dim)
    flux_x_basis = align_sparse_lat_lon(flux * basis, area_grid)
    result = sparse_xr_dot(country_matrix, area_grid * flux_x_basis)
    result = result if sparse else result.as_numpy()
    return _with_flux_nonfinite_dataarray_metadata(result, flux)
