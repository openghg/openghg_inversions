"""General methods for xarray Datasets and DataArrays.

The functions here are not specific to OpenGHG inversions: they
add functionality missing from xarray. These functions should accept
xarray Datasets and DataArrays, and return either a Dataset or a DataArray.


`get_xr_dummies` applies pandas `get_dummies` to xarray DataArrays.

`sparse_xr_dot` multiplies a Dataset or DataArray by a DataArray
 with sparse underlying array. The built-in xarray functionality doesn't
work correctly.
"""

from typing import Any, TypeVar
from collections.abc import Sequence

import numpy as np
import pandas as pd
import sparse
import xarray as xr
from sparse import COO
from xarray.core.common import DataWithCoords


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


def sparse_xr_dot(
    da1: xr.DataArray,
    da2: DataSetOrArray,
    debug: bool = False,
) -> DataSetOrArray:
    """Compute the matrix "dot" of a tuple of DataArrays with sparse.COO values.

    This multiplies and sums over all common dimensions of the input DataArrays, and
    preserves the coordinates and dimensions that are not summed over.

    Common dimensions are automatically selected by name. The input arrays must  have at
    least one dimension in common. All matching dimensions will be used for multiplication.

    NOTE: this function shouldn't be necessary, but `da1 @ da2` doesn't work properly if the
    values of `da1` and `da2` are `sparse.COO` arrays.

    Args:
        da1, da2: xr.DataArrays to multiply and sum along common dimensions.
        debug: if true, will print the dimensions of the inputs to `sparse.tensordot`
            as well as the dimension of the result.
        along_dim: name

    Returns:
        xr.Dataset or xr.DataArray containing the result of matrix/tensor multiplication.
        The type that is returned will be the same as the type of `da2`.

    Raises:
        ValueError if the input DataArrays have no common dimensions to multiply.
    """
    common_dims = set(da1.dims).intersection(set(da2.dims))
    nc = len(common_dims)

    dims1 = set(da1.dims) - common_dims
    dims2 = set(da2.dims) - common_dims

    broadcast_dims = list(dims1) + list(dims2)


    if nc == 0:
        raise ValueError(f"DataArrays \n{da1}\n{da2}\n have no common dimensions. Cannot compute `dot`.")

    tensor_dot_axes = tuple([tuple(range(-nc, 0))] * 2)
    input_core_dims = [list(common_dims)] * 2

    if debug:
        print("common dims:", common_dims)
        print("dims1:", dims1)
        print("dims2:", dims2)
        print("broadcast dims:", broadcast_dims)


    # xarray will insert new axes into broadcast dims so that the number of axes
    # in da1 and da2 are equal, unless the broadcast dim to be added would come first (from left to right)
    # we need to remove these axes, because sparse.tensordot does not
    to_select1 = []
    to_select2 = []

    for dims, to_select in zip([dims1, dims2], [to_select1, to_select2]):
        for bdim in broadcast_dims:
            if bdim in dims:
                to_select.append(slice(None))
            elif to_select:
                to_select.append(0)

    to_select = tuple(to_select1 + to_select2)

    if debug:
        print("select:", to_select)

    # compute tensor dot on last nc coordinates (because core dims are moved to end)
    # and then drop 1D coordinates resulting from summing
    def _func(x, y):
        result = sparse.tensordot(x, y, axes=tensor_dot_axes)  # type: ignore

        if debug:
            print("raw _func result shape:", result.shape)
        return result[to_select]

    def wrapper(da1, da2):
        for arr in [da1, da2]:
            print(f"_func received array of type {type(arr)}, shape {arr.shape}")
        result = _func(da1, da2)
        print(f"_func result shape: {result.shape}\n")
        return result

    func = wrapper if debug else _func

    # return xr.apply_ufunc(func, da1, da2.as_numpy(), input_core_dims=input_core_dims, join="outer")
    return xr.apply_ufunc(
        func,
        da1.transpose(..., *common_dims),  # fix for issue with xarray 2025.1.0 release
        da2.transpose(..., *common_dims),  # this makes removing broadcast dims easier..
        input_core_dims=input_core_dims,
        join="outer",
        dask="parallelized",
        output_dtypes=[da1.dtype],
    )
