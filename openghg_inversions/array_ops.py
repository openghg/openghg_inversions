"""General methods for xarray Datasets and DataArrays.

The functions here are not specific to OpenGHG inversions: they
add functionality missing from xarray. These functions should accept
xarray Datasets and DataArrays, and return either a Dataset or a DataArray.

Functions
---------
get_xr_dummies
    Applies pandas ``get_dummies`` to xarray DataArrays.
sparse_xr_dot
    Multiplies a Dataset or DataArray by a DataArray with sparse
    underlying array. The built-in xarray functionality doesn't work correctly.


Coordinate Alignment Policy
---------------------------

Spatial coordinates (e.g. ``lat`` and ``lon``) are treated as structural
invariants of the inversion workflow. Although grids are expected to be
identical, small floating-point differences can arise from I/O, reprojection,
or sparse/dask operations.

Because xarray uses strict, label-based alignment, even negligible coordinate
differences can trigger unintended reindexing, interpolation, or alignment
errors.

To avoid this, we:

1. Validate numerical equivalence within tolerance.
2. Force coordinate identity via ``assign_coords`` when validation passes.

This ensures deterministic arithmetic, prevents silent interpolation, and
keeps grid equivalence an explicit, testable invariant.

The main function that does this is ``force_align``; there is a legacy
function ``align_sparse_lat_lon`` that was introduced as a work-around to
an xarray issue, which now is written in terms of ``force_align``.

"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, overload, TypeVar

from dask.array.core import Array as DaskArray
import numpy as np
import pandas as pd
import xarray as xr
from sparse import COO, SparseArray
from xarray.core.common import DataWithCoords, is_chunked_array  # type: ignore

# type for xr.Dataset *or* xr.DataArray
DataSetOrArray = TypeVar("DataSetOrArray", bound=DataWithCoords)


def get_xr_dummies(
    da: xr.DataArray,
    categories: Sequence[Any] | pd.Index | xr.DataArray | np.ndarray | None = None,
    cat_dim: str = "categories",
    return_sparse: bool = True,
) -> xr.DataArray:
    """Create 0-1 dummy matrix from DataArray with values that correspond to categories.

    If the values of `da` are integers 0-N, then the result has N + 1 columns, and the (i, j) coordiante
    of the result is 1 if `da[i] == j`, and is 0 otherwise.

    This function works like the pandas function `get_dummies`, but preserves the coordinates of
    the input data, and allowing the user to specify coordinates for the categories used to make the
    "dummies" (or "one-hot encoding").

    Args:
        da: DataArray encoding categories.
        categories: optional coordinates for categories.
        cat_dim: dimension for categories coordinate
        sparse: if True, store values in sparse.COO matrix

    Returns:
        Dummy matrix corresponding to the input vector. Its dimensions are the same as the
            input DataArray, plus an additional "categories" dimension, which  has one value for each
            distinct value in the input DataArray.
    """
    # stack if `da` is not one dimensional
    stack_dim = ""
    if len(da.dims) > 1:
        stack_dim = "".join([str(dim) for dim in da.dims])
        da = da.stack({stack_dim: da.dims})

    dummies = pd.get_dummies(da.values, dtype="float32", sparse=return_sparse)

    # put dummies into DataArray with the right coords and dims
    values = COO.from_scipy_sparse(dummies.sparse.to_coo()) if return_sparse else dummies.values
    if categories is None:
        categories = np.arange(values.shape[1])
    coords = da.coords.merge({cat_dim: categories}).coords  # coords.merge returns Dataset, we want the coords
    result = xr.DataArray(values, coords=coords)

    # if we stacked `da`, unstack result before returning
    return result.unstack(stack_dim) if stack_dim else result


@overload
def sparse_xr_dot(da1: xr.DataArray, da2: xr.DataArray, dim: list[str] | None = None) -> xr.DataArray: ...


@overload
def sparse_xr_dot(da1: xr.DataArray, da2: xr.Dataset, dim: list[str] | None = None) -> xr.Dataset: ...


def sparse_xr_dot(
    da1: xr.DataArray, da2: xr.DataArray | xr.Dataset, dim: list[str] | None = None
) -> xr.DataArray | xr.Dataset:
    """Compute the matrix "dot" of a tuple of DataArrays with sparse.COO values.

    This multiplies and sums over all common dimensions of the input DataArrays, and
    preserves the coordinates and dimensions that are not summed over.

    Common dimensions are automatically selected by name. The input arrays must  have at
    least one dimension in common. All matching dimensions will be used for multiplication.

    Compared to just using da1 @ da2, this function has two advantages:
    1. if da1 is sparse but not a dask array, then da1 @ da2 will fail if da2 is a dask array
    2. da2 can be a Dataset, and current DataArray @ Dataset is not allowed by xarray

    Args:
        da1, da2: xr.DataArrays to multiply and sum along common dimensions.
        dim: optional list of dimensions to sum over; if `None`, then all common
          dimensions are summed over.

    Returns:
        xr.Dataset or xr.DataArray: containing the result of matrix/tensor multiplication.
            The type that is returned will be the same as the type of `da2`.
    """
    if isinstance(da1.data, SparseArray) and not is_chunked_array(da1):  # type: ignore
        da1 = da1.chunk()

    if isinstance(da2, xr.DataArray):
        if dim is None:
            return da1 @ da2
        return xr.dot(da1, da2, dim=dim)

    if dim is None:
        return da2.map(lambda x: da1 @ x)
    return da2.map(lambda x: xr.dot(da1, x, dim=dim))


def align_sparse_lat_lon(
    sparse_da: xr.DataArray,
    other_array: xr.DataArray | xr.Dataset,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> xr.DataArray:
    """Align lat/lon coordinates of sparse_da with those from other_array.

    NOTE: Workaround for xarray issue #3445:
    https://github.com/pydata/xarray/issues/3445

    Validates numerical closeness of coordinates before forcing coordinate
    identity. No interpolation or reindexing is performed.

    Args:
        sparse_da: DataArray with sparse backend.
        other_array: Dataset or DataArray providing canonical lat/lon coords.
        rtol: Relative tolerance for coordinate comparison.
        atol: Absolute tolerance for coordinate comparison.

    Returns:
        Copy of sparse_da with lat/lon coords replaced by those from other_array.

    Raises:
        ValueError: If sizes differ or coordinates differ beyond tolerance.
    """
    return force_align(
        sparse_da,
        other_array,
        dims=("lat", "lon"),
        rtol=rtol,
        atol=atol,
    )


def _sparse_dask_to_dense(da: DaskArray) -> DaskArray:
    """Convert chunks of dask array from sparse to dense."""
    return da.map_blocks(lambda arr: arr.todense())  # type: ignore


def to_dense(da: xr.DataArray) -> xr.DataArray:
    """Convert sparse to numpy.

    If the data array has chunks, these are preserved, but the underlying arrays are converted.
    Does nothing if chunks are already numpy.
    """
    if not isinstance(da.data, DaskArray):  # type: ignore
        return da.as_numpy()

    # check chunk types
    if isinstance(da.data._meta, SparseArray):
        # hack to apply the Sparse `todense()` method to chunks
        return xr.apply_ufunc(_sparse_dask_to_dense, da, dask="allowed")

    return da


T = TypeVar("T", xr.DataArray, xr.Dataset)


def force_align(
    obj: T,
    reference: xr.DataArray | xr.Dataset,
    *,
    dims: Iterable[Hashable],
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> T:
    """Force coordinate identity with a reference object along given dimensions.

    This function verifies that `obj` and `reference` share numerically
    equivalent coordinates (within tolerance) along the specified dimensions.
    If validation succeeds, the coordinates of `obj` are reassigned to be
    identical (object identity) to those of `reference`.

    No interpolation or reindexing is performed. If coordinates differ beyond
    tolerance, a ValueError is raised.

    Args:
        obj: The xarray DataArray or Dataset whose coordinates will be
            validated and reassigned.
        reference: The object providing canonical coordinates. Must contain
            the specified dimensions.
        dims: Dimensions along which to validate and enforce coordinate
            identity (e.g., ["lat", "lon"]).
        rtol: Relative tolerance for coordinate comparison. Defaults to
            NumPy's default (1e-5).
        atol: Absolute tolerance for coordinate comparison. Defaults to
            NumPy's default (1e-8).

    Returns:
        The same type as `obj`, with validated coordinates reassigned.

    Raises:
        ValueError: If a dimension is missing, sizes differ, or coordinates
            differ beyond tolerance.
    """

    dims = tuple(dims)

    for dim in dims:
        if dim not in obj.dims:
            raise ValueError(f"Dimension {dim!r} not present in object.")
        if dim not in reference.dims:
            raise ValueError(f"Dimension {dim!r} not present in reference.")

        if obj.sizes[dim] != reference.sizes[dim]:
            raise ValueError(f"Size mismatch along {dim!r}: " f"{obj.sizes[dim]} != {reference.sizes[dim]}")

        if not np.allclose(
            obj.coords[dim].values,
            reference.coords[dim].values,
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError(
                f"Coordinate mismatch along {dim!r} beyond tolerance " f"(rtol={rtol}, atol={atol})."
            )

    coord_updates = {dim: reference.coords[dim] for dim in dims}

    # assign_coords is non-mutating and preserves type
    return obj.assign_coords(coord_updates)


# -----------------------------------------------
# Gather concat for concatenating ragged arrays
# -----------------------------------------------


def concat_gather_data_arrays(
    da_dict: Mapping[Hashable, xr.DataArray],
    key_dim: str,
    ragged_dim: str,
    stack_dim: str | None = None,
    **concat_kwargs,
) -> xr.DataArray:
    """Concatenate DataArrays by gathering along ragged coordinate.

    For example, if the keys are site codes and the ragged dimension is time,
    then the "stacked dimension" will be the usual `nmeasure` coordinate.

    Args:
        da_dict: dictionary of DataArrays
        key_dim: dimension name for the keys of the dictionary
        ragged_dim: name of the ragged dimension
        stack_dim: name for the "stacked" multi-index dimension
        **concat_kwargs: arguments to pass to xr.concat

    Returns:
        Combined DataArray with new stacked dimension.

    """
    stack_dim = stack_dim or (key_dim + "_" + ragged_dim)

    pieces: list[xr.DataArray] = []
    key_vals: list[np.ndarray] = []
    ragged_vals: list[np.ndarray] = []

    for k, v in da_dict.items():
        piece = v.rename({ragged_dim: stack_dim})
        pieces.append(piece)

        n = piece.sizes[stack_dim]

        # make site indicator
        key_val = np.full(n, k)
        key_vals.append(key_val)

        # record times
        ragged_vals.append(v[ragged_dim].values)

    # concat pieces
    da = xr.concat(pieces, dim=stack_dim, **concat_kwargs)

    # now create and assign multi-index
    key_indicator = np.concatenate(key_vals)
    concat_ragged = np.concatenate(ragged_vals)
    multiindex = pd.MultiIndex.from_arrays([key_indicator, concat_ragged], names=[key_dim, ragged_dim])
    xr_multiindex = xr.Coordinates.from_pandas_multiindex(multiindex, stack_dim)

    da = da.assign_coords(xr_multiindex)

    return da


def concat_gather_datasets(
    ds_dict: Mapping[Hashable, xr.Dataset],
    key_dim: str,
    ragged_dim: str,
    stack_dim: str | None = None,
    **concat_kwargs,
) -> xr.Dataset:
    """Concatenate dictionary of xr.Datasets by gathering ragged coordinates.

    This assumes that all datasets have the same data variables.

    TODO: need to handle missing data variables.
    """
    dvs = next(iter(ds_dict.values())).data_vars

    # check that all data vars are present
    for k, v in ds_dict.items():
        if any(dv not in v.data_vars for dv in dvs):
            missing_dvs = [dv for dv in dvs if dv not in v.data_vars]
            raise ValueError(
                f"Datasets do not all have the same data variables: Dataset for key {k} missing {missing_dvs}"
            )

    gathered_dvs = {}

    for dv in dvs:
        da_dict = {k: v[dv] for k, v in ds_dict.items()}
        gathered_dvs[dv] = concat_gather_data_arrays(da_dict, key_dim, ragged_dim, stack_dim, **concat_kwargs)

    return xr.Dataset(gathered_dvs)


def concat_gather_datatree(
    dt: xr.DataTree, key_dim: str, ragged_dim: str, stack_dim: str | None = None, **concat_kwargs
) -> xr.Dataset:
    """Concatenate xr.DataTree children by gathering ragged coordinates.

    This assumes that all children have the same data variables.
    """
    ds_dict = {str(k): v.to_dataset() for k, v in dt.items()}
    dvs = next(iter(ds_dict.values())).data_vars

    gathered_dvs = {}

    for dv in dvs:
        da_dict = {k: v[dv] for k, v in ds_dict.items()}
        gathered_dvs[dv] = concat_gather_data_arrays(da_dict, key_dim, ragged_dim, stack_dim, **concat_kwargs)

    return xr.Dataset(gathered_dvs)
