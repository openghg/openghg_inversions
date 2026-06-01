"""Basis-backed postprocessing product helpers."""

from __future__ import annotations

import xarray as xr

from openghg_inversions.array_ops import align_sparse_lat_lon, sparse_xr_dot
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.postprocessing.inversion_output import (
    InversionOutput,
    standard_flux,
    standard_trace_dataset,
)

_TRACE_STATE_DIMS = ("region", "nx")


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
        interpolated = basis_functions.operator.interpolate(data, weights=weights)
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


def reconstruct_flux_stats(
    inv_out: InversionOutput,
    stats_ds: xr.Dataset,
    *,
    report_flux_on_inversion_grid: bool,
) -> xr.Dataset:
    """Reconstruct gridded flux statistics with the retained basis operator."""
    basis_functions = inv_out.basis_functions
    flux = standard_flux(inv_out)
    if report_flux_on_inversion_grid:
        region_flux = _operator_region_flux_mean(basis_functions, flux)
        state = _to_operator_state_dim(stats_ds, basis_functions) * region_flux
        return _interpolate_dataset(state, basis_functions, weights=None)

    return _interpolate_dataset(stats_ds, basis_functions, weights=flux)


def reconstruct_scale_factor_stats(
    inv_out: InversionOutput,
    stats_ds: xr.Dataset,
) -> xr.Dataset:
    """Reconstruct gridded scale-factor statistics with the retained operator."""
    return _interpolate_dataset(stats_ds, inv_out.basis_functions, weights=None)


def make_x_to_country_matrix(
    inv_out: InversionOutput,
    *,
    country_matrix: xr.DataArray,
    area_grid: xr.DataArray,
    sparse: bool = False,
) -> xr.DataArray:
    """Construct a basis-state to country-total matrix."""
    basis_functions = inv_out.basis_functions
    flux = standard_flux(inv_out)
    x_trace = standard_trace_dataset(inv_out, var_names="x")
    trace_state_dim = _trace_state_dim(x_trace, basis_functions)
    basis = align_sparse_lat_lon(basis_functions.operator.basis_matrix, flux)
    basis = _from_operator_state_dim(basis, basis_functions, trace_state_dim)
    flux_x_basis = align_sparse_lat_lon(flux * basis, area_grid)
    result = sparse_xr_dot(country_matrix, area_grid * flux_x_basis)
    return result if sparse else result.as_numpy()
