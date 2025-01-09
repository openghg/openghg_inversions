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

    dummies = pd.get_dummies(da.values, dtype=int, sparse=return_sparse)

    # put dummies into DataArray with the right coords and dims
    values = COO.from_scipy_sparse(dummies.sparse.to_coo()) if return_sparse else dummies.values
    if categories is None:
        categories = np.arange(values.shape[1])
    coords = da.coords.merge({cat_dim: categories}).coords  # coords.merge returns Dataset, we want the coords
    result = xr.DataArray(values, coords=coords)

    # if we stacked `da`, unstack result before returning
    return result.unstack(stack_dim) if stack_dim else result


@overload
def sparse_xr_dot(da1: xr.DataArray, da2: xr.DataArray) -> xr.DataArray: ...


@overload
def sparse_xr_dot(da1: xr.DataArray, da2: xr.Dataset) -> xr.Dataset: ...


def sparse_xr_dot(da1: xr.DataArray, da2: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
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

    Returns:
        xr.Dataset or xr.DataArray containing the result of matrix/tensor multiplication.
        The type that is returned will be the same as the type of `da2`.
    """
    if isinstance(da1.data, SparseArray) and not is_chunked_array(da1):  # type: ignore
        da1 = da1.chunk()

    if isinstance(da2, xr.DataArray):
        return da1 @ da2

    return da2.map(lambda x: da1 @ x)
