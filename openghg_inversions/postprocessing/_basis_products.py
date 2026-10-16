"""Basis-backed postprocessing product helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import dask.array as da
import numpy as np
import sparse
import xarray as xr

from openghg_inversions.array_ops import (
    align_sparse_lat_lon,
    force_align,
    sparse_xr_dot,
    to_dense,
)
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.basis.operators import BucketBasisOperator
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
    _chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Interpolate state variables, densifying bounded output tiles before assembly."""
    operator_state = _to_operator_state_dim(state, basis_functions)
    weighted_basis = None
    if (
        weights is not None
        and _chunks is not None
        and isinstance(basis_functions.operator, BucketBasisOperator)
    ):
        weighted_basis = _bounded_weighted_basis(basis_functions, weights, _chunks)
    data_vars = {}
    for name, data in operator_state.data_vars.items():
        if weighted_basis is not None:
            interpolated = xr.dot(weighted_basis, data, dim=basis_functions.operator.meta.state_dim)
        else:
            interpolated = (
                basis_functions.interpolate(data)
                if weights is None
                else basis_functions.operator.interpolate(data, weights=weights)
            )
        if _chunks is not None:
            # Sparse matrix-vector contraction may still return sparse tiles.
            # Densify those tiles before the full output is assembled.
            interpolated = to_dense(interpolated)
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
    # Sparse concatenation distinguishes -0.0 and +0.0 fill values.
    weighted_basis = basis * flux + 0.0
    return (weighted_basis.sum(grid_dims) / basis.sum(grid_dims)).fillna(0.0)


def _chunk_basis_functions(
    basis_functions: BasisFunctions, chunks: Mapping[str, int | tuple[int, ...]]
) -> BasisFunctions:
    """Give reconstruction its own chunked geometry without modifying the retained basis."""
    operator = basis_functions.operator
    if isinstance(operator, BucketBasisOperator):
        operator = operator.with_chunks(
            {
                **{dim: size for dim, size in chunks.items() if dim in operator.meta.grid_dims},
                operator.meta.state_dim: operator.basis_matrix.variable.chunksizes[operator.meta.state_dim],
            }
        )
        return replace(basis_functions, operator=operator)
    return basis_functions


def _weighted_basis_block(basis, weights, fill, *, grid_ndim):
    """Multiply one sparse tile using its original producer's signed-zero fill."""
    basis = basis.reshape((*basis.shape, *((1,) * (weights.ndim - grid_ndim))))
    weights = weights.reshape((*weights.shape[:grid_ndim], 1, *weights.shape[grid_ndim:]))
    fill = fill.reshape((*fill.shape[:grid_ndim], 1, *fill.shape[grid_ndim:]))
    return sparse.elemwise(lambda b, w, f: np.where(b == 0, f, b * w), basis, weights, fill)


def _first_weighted_fill(block, *, basis_fill):
    """Keep one fill scalar per original producer, including its zero sign."""
    return np.multiply(basis_fill, block[(0,) * block.ndim]).reshape((1,) * block.ndim)


def _bounded_weighted_basis(
    basis_functions: BasisFunctions, flux: xr.DataArray, limits: Mapping[str, int]
) -> xr.DataArray:
    """Bound weighted geometry while preserving legacy sparse zero/NaN products.

    Sparse multiplication chooses its fill from the first weight in each block,
    and distinguishes signed zeros when pruning stored coefficients. Keep that
    choice when subdividing blocks so zero-weight cells retain their NaN masks.
    Only one fill scalar per original block is added; related summaries share it.
    """
    operator = basis_functions.operator
    matrix = operator.basis_matrix
    grid_dims = operator.meta.grid_dims
    flux = force_align(flux, matrix, dims=list(grid_dims)).transpose(*grid_dims, ...)
    # Inspect the original graph's chunk metadata without executing its unbounded product.
    reference = matrix * flux
    original_chunks = {dim: reference.variable.chunksizes[dim] for dim in flux.dims}
    original_flux = flux.chunk(original_chunks)
    fills = da.map_blocks(
        _first_weighted_fill,
        original_flux.data,
        basis_fill=np.zeros((), dtype=matrix.dtype),
        chunks=tuple((1,) * len(chunk) for chunk in original_flux.chunks),
        meta=np.empty((0,) * flux.ndim, dtype=reference.dtype),
    )
    chunks = {}
    for axis, dim in enumerate(flux.dims):
        old = original_chunks[dim]
        limit = limits.get(dim, max(old))
        subdivisions = [tuple(min(limit, n - start) for start in range(0, n, limit)) for n in old]
        chunks[dim] = tuple(size for pieces in subdivisions for size in pieces)
        fill_indices = np.repeat(np.arange(len(old)), [len(pieces) for pieces in subdivisions])
        fills = da.take(fills, fill_indices, axis=axis)
    matrix = _chunk_basis_functions(basis_functions, chunks).operator.basis_matrix.transpose(
        *grid_dims, operator.meta.state_dim
    )
    flux = flux.chunk(chunks)
    dims = (*matrix.dims, *(dim for dim in flux.dims if dim not in grid_dims))
    weighted = da.blockwise(
        _weighted_basis_block,
        dims,
        matrix.data,
        matrix.dims,
        flux.data,
        flux.dims,
        fills,
        flux.dims,
        # Scalar fill blocks correspond to refined flux blocks, not individual cells.
        align_arrays=False,
        grid_ndim=len(grid_dims),
        meta=sparse.COO.from_numpy(np.empty((0,) * len(dims), dtype=reference.dtype)),
    )
    return xr.DataArray(weighted, dims=dims, coords=reference.coords).rename(reference.name)


def reconstruct_flux_stats(
    basis_functions: BasisFunctions,
    flux: xr.DataArray,
    stats_ds: xr.Dataset,
    *,
    report_flux_on_inversion_grid: bool,
    _chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Reconstruct gridded flux statistics with the retained basis operator.

    Args:
        basis_functions: Retained basis operator and flux metadata.
        flux: Prior flux used to weight or scale reconstructed statistics.
        stats_ds: Statistics in basis-state space.
        report_flux_on_inversion_grid: Report region-mean flux when ``True``;
            otherwise interpolate with prior-flux weights.
        _chunks: Internal spatial/time chunk limits for reconstruction of either
            compact summaries or complete draws. Compact state payloads are
            made dense, preserving outer Dask laziness when present, to avoid
            densely occupied sparse grid outputs. Any remaining sparse output
            tiles are made dense before assembly.
            Region means retain the original spatial reduction order, with
            time chunks bounded before the interpolation operator is chunked.
            When inversion-grid reporting is enabled, supplying ``_chunks`` eagerly
            computes the compact region/time prior-flux means before lazy interpolation.
            Native-grid weighting retains original sparse fill choices so signed
            zeros and missing interval endpoints do not change with block size.

    Returns:
        Reconstructed gridded statistics carrying non-finite policy metadata.

    Warns:
        NonFiniteFluxWarning: If old retained flux has no sanitation metadata
            and the lazy zero-fill backstop is applied.
    """
    flux = sanitize_flux_nonfinite(
        flux,
        context="operator-backed flux reconstruction",
        warn=True,
    )
    region_flux = None
    if report_flux_on_inversion_grid:
        if _chunks is not None:
            # Time chunks do not change the original spatial reduction order.
            flux = flux.chunk(
                {
                    dim: size
                    for dim, size in _chunks.items()
                    if dim in flux.dims and dim not in basis_functions.operator.meta.grid_dims
                }
            )
        region_flux = _operator_region_flux_mean(basis_functions, flux)
        if _chunks is not None:
            # This compact state/time boundary preserves the original spatial
            # reduction order and avoids repeating global region means per tile.
            region_flux = region_flux.compute(scheduler="single-threaded").as_numpy()

    if _chunks is not None:
        stats_ds = stats_ds.map(to_dense)
        if region_flux is not None:
            basis_functions = _chunk_basis_functions(basis_functions, _chunks)
            region_flux = region_flux.chunk(
                {dim: size for dim, size in _chunks.items() if dim in region_flux.dims}
            )

    if region_flux is not None:
        state = _to_operator_state_dim(stats_ds, basis_functions) * region_flux
        if _chunks is not None:
            state = state.map(to_dense)
        return copy_flux_nonfinite_attrs(
            _interpolate_dataset(state, basis_functions, weights=None, _chunks=_chunks), flux
        )

    return copy_flux_nonfinite_attrs(
        _interpolate_dataset(stats_ds, basis_functions, weights=flux, _chunks=_chunks), flux
    )


def reconstruct_scale_factor_stats(
    basis_functions: BasisFunctions,
    stats_ds: xr.Dataset,
    *,
    _chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Reconstruct scale summaries with optional borrowed-basis spatial chunk limits.

    Supplying ``_chunks`` constructs a separately chunked interpolation operator;
    interpolation remains lazy and the borrowed basis and statistics are unchanged.
    """
    if _chunks is not None:
        basis_functions = _chunk_basis_functions(basis_functions, _chunks)
        stats_ds = stats_ds.map(to_dense)
    return _interpolate_dataset(stats_ds, basis_functions, weights=None, _chunks=_chunks)


def make_x_to_country_matrix(
    basis_functions: BasisFunctions,
    flux: xr.DataArray,
    x_trace: xr.Dataset,
    *,
    country_matrix: xr.DataArray,
    area_grid: xr.DataArray,
    sparse: bool = False,
) -> xr.DataArray:
    """Construct a basis-state to country-total matrix.

    Args:
        basis_functions: Retained basis operator used for reconstruction.
        flux: Prior flux on the operator grid.
        x_trace: Trace dataset providing the state-dimension convention.
        country_matrix: Country masks aligned to the flux grid.
        area_grid: Grid-cell areas used to form country totals.
        sparse: Preserve sparse output when ``True``.

    Returns:
        State-to-country matrix carrying non-finite policy metadata.

    Warns:
        NonFiniteFluxWarning: If old retained flux has no sanitation metadata
            and the lazy zero-fill backstop is applied.
    """
    flux = sanitize_flux_nonfinite(
        flux,
        context="operator-backed country flux reconstruction",
        warn=True,
    )
    trace_state_dim = _trace_state_dim(x_trace, basis_functions)
    basis = align_sparse_lat_lon(basis_functions.operator.basis_matrix, flux)
    basis = _from_operator_state_dim(basis, basis_functions, trace_state_dim)
    flux_x_basis = align_sparse_lat_lon(flux * basis, area_grid)
    result = sparse_xr_dot(country_matrix, area_grid * flux_x_basis)
    result = result if sparse else result.as_numpy()
    return copy_flux_nonfinite_attrs(result, flux)
