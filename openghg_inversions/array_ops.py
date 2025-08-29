"""General methods for xarray Datasets and DataArrays.

The functions here are not specific to OpenGHG inversions: they
add functionality missing from xarray. These functions should accept
xarray Datasets and DataArrays, and return either a Dataset or a DataArray.


`get_xr_dummies` applies pandas `get_dummies` to xarray DataArrays.

`sparse_xr_dot` multiplies a Dataset or DataArray by a DataArray
 with sparse underlying array. The built-in xarray functionality doesn't
work correctly.
"""

from typing import Any, overload, TypeVar
from collections.abc import Sequence

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


def sparse_xr_dot(da1: xr.DataArray, da2: xr.DataArray | xr.Dataset, dim: list[str] | None = None) -> xr.DataArray | xr.Dataset:
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
        xr.Dataset or xr.DataArray containing the result of matrix/tensor multiplication.
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


def align_sparse_lat_lon(sparse_da: xr.DataArray, other_array: DataWithCoords) -> xr.DataArray:
    """Align lat/lon coordinates of sparse_da with lat/lon coordinates from other_array.

    NOTE: This is a work-around for an xarray Issue: https://github.com/pydata/xarray/issues/3445

    Args:
        sparse_da: xarray DataArray with sparse underlying array
        other_array: xarray Dataset or DataArray whose lat/lon coordinates should be used
            to replace the lat/lon coordinates in sparse_da

    Returns:
        copy of sparse_da with lat/lon coords from other_array
    """
    if len(sparse_da.lon) != len(other_array.lon):
        raise ValueError("Both arrays must have the same number lon "
                         f"coordinates: {len(sparse_da.lon)} != {len(other_array.lon)}")
    if len(sparse_da.lat) != len(other_array.lat):
        raise ValueError("Both arrays must have the same number lat "
                         f"coordinates: {len(sparse_da.lat)} != {len(other_array.lat)}")

    return sparse_da.assign_coords(lat=other_array.lat, lon=other_array.lon)


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
