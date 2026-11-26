"""Shared labelled square-matrix axis construction.

Numerical kernels use distinct row and column dimensions so xarray never
contracts a square matrix along both axes accidentally.  Column axes retain
the row axis's typed :class:`pandas.Index` or :class:`pandas.MultiIndex`;
MultiIndex level names and one-dimensional auxiliary-coordinate names are
renamed deterministically to avoid collisions.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable

import numpy as np
import pandas as pd
import xarray as xr


def coordinates_on_dims(array: xr.DataArray, dims: Iterable[Hashable]) -> dict[str, xr.DataArray]:
    """Return the coordinates of ``array`` that lie only on ``dims``.

    Scalar coordinates are included, since they lie on no dimension. This is
    how a result built from new values keeps the labels of the axes it shares
    with ``array``.

    Args:
        array: Source of the coordinates.
        dims: Dimensions the returned coordinates may use.

    Returns:
        Coordinates keyed by name, including indexes, MultiIndex levels and
        auxiliary coordinates on those dimensions.
    """
    allowed = set(dims)
    return {
        str(name): coordinate for name, coordinate in array.coords.items() if set(coordinate.dims).issubset(allowed)
    }


def square_matrix(
    values: np.ndarray,
    labels: xr.DataArray,
    *,
    row_dim: str,
    column_dim: str,
    name: str | None = None,
    attrs: dict[str, object] | None = None,
) -> xr.DataArray:
    """Label a square matrix with one axis of ``labels`` on both its rows and columns.

    Rows keep the ``row_dim`` coordinates of ``labels``. Columns receive the
    typed, collision-safe copies from :func:`renamed_column_coordinates`, so
    xarray never contracts both axes at once.

    Args:
        values: Square matrix ordered like the ``row_dim`` index of ``labels``.
        labels: Array whose ``row_dim`` axis labels both matrix axes.
        row_dim: Row dimension shared with ``labels``.
        column_dim: Distinct, unoccupied column dimension.
        name: Optional result name.
        attrs: Optional result attributes.

    Returns:
        The labelled ``(row_dim, column_dim)`` matrix.
    """
    return xr.DataArray(
        values,
        dims=(row_dim, column_dim),
        coords={
            **coordinates_on_dims(labels, {row_dim}),
            **renamed_column_coordinates(labels, row_dim=row_dim, column_dim=column_dim),
        },
        name=name,
        attrs=attrs or {},
    )


def matrix_column_dim(row_dim: str, occupied: Iterable[Hashable]) -> str:
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
        result[column_dim].attrs = array.coords[row_dim].attrs
        for raw_name, renamed_name in zip(index.names, renamed_levels, strict=True):
            if raw_name is not None and raw_name in array.coords:
                result[renamed_name].attrs = array.coords[raw_name].attrs
    else:
        result[column_dim] = xr.DataArray(
            xr.IndexVariable(
                column_dim,
                index.rename(column_dim),
                attrs=array.coords[row_dim].attrs,
            )
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
    coords = coordinates_on_dims(ordered, leading_dims)
    coords.update(renamed_column_coordinates(ordered, row_dim=row_dim, column_dim=column_dim))
    return xr.DataArray(
        ordered.data,
        dims=(*leading_dims, column_dim),
        coords=coords,
        name=ordered.name,
        attrs=ordered.attrs,
    )
