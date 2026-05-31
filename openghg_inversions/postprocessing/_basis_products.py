"""Basis-backed postprocessing product helpers.

These helpers keep modern RHIME product reconstruction close to retained
``BasisFunctions`` while preserving the flat-basis fallback used by
``LegacyInversionOutput``.
"""

from __future__ import annotations

from typing import cast

import xarray as xr

from openghg_inversions.array_ops import align_sparse_lat_lon, sparse_xr_dot
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.postprocessing.inversion_output import (
    InversionOutput,
    PostprocessingInput,
    StandardPostprocessingOutput,
)

_TRACE_STATE_DIMS = ("region", "nx")


def _modern_output(
    original: PostprocessingInput, view: StandardPostprocessingOutput
) -> InversionOutput | None:
    """Return the modern output represented by ``original`` or ``view``."""
    if isinstance(original, InversionOutput):
        return original

    modern = getattr(view, "modern_output", None)
    return modern if isinstance(modern, InversionOutput) else None


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
    original: PostprocessingInput,
    view: StandardPostprocessingOutput,
    stats_ds: xr.Dataset,
    *,
    report_flux_on_inversion_grid: bool,
) -> xr.Dataset:
    """Reconstruct gridded flux statistics with an operator-backed modern path.

    Args:
        original: Original postprocessing input supplied by the caller.
        view: Normalised standard postprocessing view.
        stats_ds: Basis-state trace statistics.
        report_flux_on_inversion_grid: If True, report regional mean flux on
            the inversion grid. Otherwise report prior-flux-weighted grid flux.

    Returns:
        Dataset of gridded flux statistics.
    """
    modern = _modern_output(original, view)
    if modern is not None:
        basis_functions = modern.basis_functions
        if report_flux_on_inversion_grid:
            region_flux = _operator_region_flux_mean(basis_functions, view.flux)
            state = _to_operator_state_dim(stats_ds, basis_functions) * region_flux
            return _interpolate_dataset(state, basis_functions, weights=None)

        return _interpolate_dataset(stats_ds, basis_functions, weights=view.flux)

    if report_flux_on_inversion_grid:
        agg_flux = ((view.basis * view.flux).sum(["lat", "lon"]) / view.basis.sum(["lat", "lon"])).fillna(0.0)
        return cast(xr.Dataset, sparse_xr_dot(view.basis, agg_flux * stats_ds))

    return cast(xr.Dataset, sparse_xr_dot((view.flux * view.basis), stats_ds))


def reconstruct_scale_factor_stats(
    original: PostprocessingInput,
    view: StandardPostprocessingOutput,
    stats_ds: xr.Dataset,
) -> xr.Dataset:
    """Reconstruct gridded scale-factor statistics with retained operators when available."""
    modern = _modern_output(original, view)
    if modern is not None:
        return _interpolate_dataset(stats_ds, modern.basis_functions, weights=None)

    return cast(xr.Dataset, sparse_xr_dot(view.basis, stats_ds))


def make_x_to_country_matrix(
    original: PostprocessingInput,
    view: StandardPostprocessingOutput,
    *,
    country_matrix: xr.DataArray,
    area_grid: xr.DataArray,
    sparse: bool = False,
) -> xr.DataArray:
    """Construct a basis-state to country-total matrix.

    Modern RHIME outputs use retained ``BasisFunctions.operator.basis_matrix``.
    Legacy outputs keep the existing flat-basis calculation.
    """
    modern = _modern_output(original, view)
    if modern is not None:
        basis_functions = modern.basis_functions
        x_trace = view.get_trace_dataset(var_names="x")
        trace_state_dim = _trace_state_dim(x_trace, basis_functions)
        basis = align_sparse_lat_lon(basis_functions.operator.basis_matrix, view.flux)
        basis = _from_operator_state_dim(basis, basis_functions, trace_state_dim)
        flux_x_basis = align_sparse_lat_lon(view.flux * basis, area_grid)
        result = sparse_xr_dot(country_matrix, area_grid * flux_x_basis)
        return result if sparse else result.as_numpy()

    basis = align_sparse_lat_lon(view.basis, view.flux)
    flux_x_basis = align_sparse_lat_lon(view.flux * basis, area_grid)
    result = sparse_xr_dot(country_matrix, area_grid * flux_x_basis)
    return result if sparse else result.as_numpy()
