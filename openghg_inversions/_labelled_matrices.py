"""Shared labelled square-matrix axis construction and diagnostics.

Numerical kernels use distinct row and column dimensions so xarray never
contracts a square matrix along both axes accidentally.  Column axes retain
the row axis's typed :class:`pandas.Index` or :class:`pandas.MultiIndex`;
MultiIndex level names and one-dimensional auxiliary-coordinate names are
renamed deterministically to avoid collisions.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
import xarray as xr


def matrix_column_dim(row_dim: str, occupied: tuple[Hashable, ...]) -> str:
    """Return a deterministic unoccupied column dimension for ``row_dim``."""
    candidate = f"{row_dim}_cov"
    suffix = 2
    while candidate in occupied:
        candidate = f"{row_dim}_cov_{suffix}"
        suffix += 1
    return candidate


def renamed_column_coordinates(
    array: xr.DataArray,
    *,
    row_dim: str,
    column_dim: str,
) -> dict[str, xr.DataArray]:
    """Return typed column-axis coordinates copied from a labelled row axis.

    Args:
        array: Array containing the source row axis.
        row_dim: Dimension whose index and auxiliary coordinates are copied.
        column_dim: Distinct destination dimension.

    Returns:
        Coordinates for ``column_dim``. MultiIndex levels and auxiliary
        coordinates are renamed with a ``_cov`` suffix when necessary.
    """
    index = array.get_index(row_dim)
    result: dict[str, xr.DataArray] = {}
    reserved = {str(name) for name in (*array.dims, *array.coords)} | {column_dim}
    index_level_names: set[str] = set()
    if isinstance(index, pd.MultiIndex):
        renamed_levels: list[str] = []
        for position, raw_name in enumerate(index.names):
            base = f"{raw_name}_cov" if raw_name is not None else f"{column_dim}_level_{position}"
            candidate = base
            suffix = 2
            while candidate in reserved or candidate in renamed_levels:
                candidate = f"{base}_{suffix}"
                suffix += 1
            renamed_levels.append(candidate)
            reserved.add(candidate)
            if raw_name is not None:
                index_level_names.add(str(raw_name))
        renamed = index.rename(renamed_levels)
        coordinates = xr.Coordinates.from_pandas_multiindex(renamed, column_dim)
        result.update({str(name): coordinate for name, coordinate in coordinates.items()})
    else:
        result[column_dim] = xr.DataArray(
            index.to_numpy(copy=False),
            dims=column_dim,
            attrs=array.coords[row_dim].attrs,
        )

    for raw_name, coordinate in array.coords.items():
        name = str(raw_name)
        if name == row_dim or name in index_level_names or tuple(coordinate.dims) != (row_dim,):
            continue
        base = f"{name}_cov"
        candidate = base
        suffix = 2
        while candidate in reserved or candidate in result:
            candidate = f"{base}_{suffix}"
            suffix += 1
        reserved.add(candidate)
        result[candidate] = xr.DataArray(
            coordinate.data,
            dims=column_dim,
            attrs=coordinate.attrs,
        )
    return result


def to_column_axis(
    array: xr.DataArray,
    *,
    row_dim: str,
    column_dim: str,
    leading_dims: tuple[str, ...],
) -> xr.DataArray:
    """Rename one labelled row axis to a typed, collision-safe column axis."""
    ordered = array.transpose(*leading_dims, row_dim)
    coords = {
        str(name): coordinate
        for name, coordinate in ordered.coords.items()
        if set(coordinate.dims).issubset(leading_dims)
    }
    coords.update(renamed_column_coordinates(ordered, row_dim=row_dim, column_dim=column_dim))
    return xr.DataArray(
        ordered.data,
        dims=(*leading_dims, column_dim),
        coords=coords,
        name=ordered.name,
        attrs=ordered.attrs,
    )


def with_square_matrix_diagnostics(
    array: xr.DataArray,
    *,
    mathematical_name: str,
    require_positive_definite: bool = False,
    maximum_eigen_diagnostic_size: int = 512,
) -> xr.DataArray:
    """Validate a labelled square matrix once and attach numerical diagnostics.

    Args:
        array: Eager two-dimensional square matrix with distinct dimensions.
        mathematical_name: Mathematical label stored in attributes and errors.
        require_positive_definite: Whether to reject non-positive or
            numerically rank-deficient matrices.
        maximum_eigen_diagnostic_size: Largest matrix for which a full
            eigendecomposition is performed when positivity is not required.

    Returns:
        ``array`` with symmetry and optional eigenvalue diagnostics attached.

    Raises:
        ValueError: If the layout is not square, values are complex or
            non-finite, symmetry fails, or requested positive definiteness
            fails.
    """
    if array.ndim != 2 or array.dims[0] == array.dims[1] or array.shape[0] != array.shape[1]:
        raise ValueError(f"{mathematical_name} must be a square matrix on distinct labelled axes")
    raw_values = np.asarray(array.values)
    if np.iscomplexobj(raw_values) or not np.all(np.isfinite(raw_values)):
        raise ValueError(f"{mathematical_name} must contain only finite real values")
    values = np.asarray(raw_values, dtype=np.float64)
    asymmetry = float(np.max(np.abs(values - values.T))) if values.size else 0.0
    scale = max(1.0, float(np.max(np.abs(values))) if values.size else 1.0)
    tolerance = 1e-10
    if asymmetry > tolerance * scale:
        raise ValueError(f"{mathematical_name} is not symmetric within tolerance {tolerance:g}")

    attrs: dict[str, object] = {
        **array.attrs,
        "mathematical_name": mathematical_name,
        "symmetry_absolute_error": asymmetry,
        "diagnostic_tolerance": tolerance,
    }
    should_diagnose_eigenvalues = (
        require_positive_definite or values.shape[0] <= maximum_eigen_diagnostic_size
    )
    if should_diagnose_eigenvalues:
        eigenvalues = np.linalg.eigvalsh((values + values.T) * 0.5)
        minimum = float(eigenvalues[0]) if eigenvalues.size else 0.0
        maximum = float(eigenvalues[-1]) if eigenvalues.size else 0.0
        if require_positive_definite and (
            not eigenvalues.size or maximum <= 0.0 or minimum <= 1e-12 * maximum
        ):
            raise ValueError(f"{mathematical_name} must be positive definite and full rank")
        attrs.update(minimum_eigenvalue=minimum, psd_diagnostic="full_eigendecomposition")
    else:
        attrs.update(
            minimum_eigenvalue=np.nan,
            psd_diagnostic=f"skipped_full_eigendecomposition_above_{maximum_eigen_diagnostic_size}",
        )
    return array.assign_attrs(attrs)
