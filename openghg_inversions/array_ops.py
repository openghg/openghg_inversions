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

import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Literal, TypeVar, overload

import numpy as np
import pandas as pd
import xarray as xr
from dask.array.core import Array as DaskArray
from sparse import COO, SparseArray
from xarray.core.common import DataWithCoords, is_chunked_array  # type: ignore

# type for xr.Dataset *or* xr.DataArray
DataSetOrArray = TypeVar("DataSetOrArray", bound=DataWithCoords)


def validate_covariance_coordinates(
    covariance: xr.DataArray,
    *,
    dim: str,
    covariance_dim: str | None = None,
) -> None:
    """Require a square covariance matrix to repeat one coordinate exactly.

    The conventional column dimension is ``f"{dim}_cov"``. PyMC treats the
    row and column dimensions as distinct and does not verify that their labels
    denote the same ordered values, so that scientific invariant is checked
    before labels reach the model backend.

    Args:
        covariance: Labelled covariance matrix to validate.
        dim: Row dimension and canonical coordinate name.
        covariance_dim: Column dimension. Defaults to ``f"{dim}_cov"``.

    Raises:
        ValueError: If dimensions, shape, coordinates, or coordinate ordering
            do not follow the covariance convention.
    """
    covariance_dim = f"{dim}_cov" if covariance_dim is None else covariance_dim
    if covariance.dims != (dim, covariance_dim):
        raise ValueError(
            "Covariance matrix must have dims "
            f"({dim!r}, {covariance_dim!r}); got {covariance.dims!r}."
        )
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"Covariance matrix must be square; got shape {covariance.shape!r}.")
    missing = [name for name in (dim, covariance_dim) if name not in covariance.coords]
    if missing:
        raise ValueError(f"Covariance matrix must carry coordinate(s) {missing!r}.")
    for name in (dim, covariance_dim):
        index = covariance.get_index(name)
        if not index.is_unique:
            duplicate = index[index.duplicated()][0]
            raise ValueError(
                f"Covariance coordinate {name!r} must contain unique labels; "
                f"duplicate {duplicate!r}."
            )
    if not np.array_equal(
        np.asarray(covariance.coords[dim].values),
        np.asarray(covariance.coords[covariance_dim].values),
    ):
        raise ValueError(
            f"Covariance coordinates {dim!r} and {covariance_dim!r} must contain "
            "the same values in the same order."
        )


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
    """Convert sparse array payloads to dense NumPy arrays.

    For an eager sparse input, this returns an eager NumPy-backed DataArray. For
    a Dask array with sparse chunks, the outer Dask collection and its chunking
    are preserved while a lazy block operation converts each chunk to NumPy when
    executed. An array whose Dask chunks are already dense is returned unchanged.

    Args:
        da: DataArray whose eager data or Dask chunk payloads may be sparse.

    Returns:
        DataArray with dense NumPy data or lazily densified Dask chunks.
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
            raise ValueError(f"Size mismatch along {dim!r}: {obj.sizes[dim]} != {reference.sizes[dim]}")

        if not np.allclose(
            obj.coords[dim].values,
            reference.coords[dim].values,
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError(
                f"Coordinate mismatch along {dim!r} beyond tolerance (rtol={rtol}, atol={atol})."
            )

    coord_updates = {dim: reference.coords[dim] for dim in dims}

    # assign_coords is non-mutating and preserves type
    return obj.assign_coords(coord_updates)


# -----------------------------------------------
#  Concat for dictionary of DataArrays
# -----------------------------------------------


def concat_data_arrays(
    da_dict: Mapping[str, xr.DataArray],
    key_dim: str,
    **concat_kwargs,
) -> xr.DataArray:
    to_concat = [v.expand_dims({key_dim: [k]}) for k, v in da_dict.items()]
    return xr.concat(to_concat, dim=key_dim, **concat_kwargs)


# -----------------------------------------------
# Gather concat for concatenating ragged arrays
# -----------------------------------------------


def concat_gather_data_arrays(
    da_dict: Mapping[str, xr.DataArray],
    key_dim: str,
    ragged_dim: str,
    stack_dim: str | None = None,
    **concat_kwargs,
) -> xr.DataArray:
    """Concatenate DataArrays by gathering along a ragged coordinate.

    For example, if the keys are site codes and the ragged dimension is time,
    then the stacked dimension is the usual ``nmeasure`` coordinate with
    ``(site, time)`` levels. The same operation can gather source-specific
    state blocks into ``(source, region_in_source)`` without rectangular
    padding. Any alignment policy for dimensions other than ``ragged_dim``
    should be passed explicitly through ``concat_kwargs``.

    Args:
        da_dict: DataArrays keyed by the values to record in ``key_dim``.
        key_dim: Dimension name for the mapping keys.
        ragged_dim: Name of the ragged dimension.
        stack_dim: Name for the gathered MultiIndex dimension.
        **concat_kwargs: Arguments passed to :func:`xarray.concat`.

    Returns:
        Combined DataArray with a new gathered dimension.

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


def select_gathered_data_array(
    da: xr.DataArray,
    *,
    key: Any,
    key_dim: str,
    ragged_dim: str,
    stack_dim: str,
) -> xr.DataArray:
    """Select one key from a gathered DataArray and restore its ragged index.

    This is the labelled inverse of one branch of
    :func:`concat_gather_data_arrays`. The gathered dimension remains named
    ``stack_dim``, but its ``(key_dim, ragged_dim)`` MultiIndex is reduced to
    the selected ``ragged_dim`` labels.

    Args:
        da: DataArray containing a gathered MultiIndex dimension.
        key: Key value to select from the gathered index.
        key_dim: Name of the key level in the gathered MultiIndex.
        ragged_dim: Name of the ragged level in the gathered MultiIndex.
        stack_dim: Name of the gathered dimension.

    Returns:
        Selected DataArray indexed by the original ragged labels.

    Raises:
        TypeError: If the gathered dimension does not use a pandas MultiIndex.
        ValueError: If the gathered levels or requested key are invalid.
    """
    index = da.indexes.get(stack_dim)
    if not isinstance(index, pd.MultiIndex):
        raise TypeError(f"Dimension {stack_dim!r} must have a pandas MultiIndex.")
    if list(index.names) != [key_dim, ragged_dim]:
        raise ValueError(
            f"Dimension {stack_dim!r} must gather ({key_dim!r}, {ragged_dim!r}); "
            f"found levels {list(index.names)!r}."
        )
    if key not in index.get_level_values(key_dim):
        raise ValueError(f"Gathered dimension {stack_dim!r} does not contain {key_dim} value {key!r}.")

    positions = np.flatnonzero(index.get_level_values(key_dim) == key)
    selected = da.isel({stack_dim: positions}).reset_index(stack_dim)
    selected = selected.drop_vars(key_dim)
    return selected.set_index({stack_dim: ragged_dim})


def concat_gather_datasets(
    ds_dict: Mapping[str, xr.Dataset],
    key_dim: str,
    ragged_dim: str,
    stack_dim: str | None = None,
    missing_data_vars: Literal["error", "drop"] = "error",
    **concat_kwargs,
) -> xr.Dataset:
    """Concatenate dictionary of xr.Datasets by gathering ragged coordinates.

    Args:
        ds_dict: Mapping from key values to datasets to concatenate.
        key_dim: Name for the key dimension level in the output MultiIndex.
        ragged_dim: Ragged coordinate dimension to gather across datasets.
        stack_dim: Optional name for the gathered dimension in the output.
        missing_data_vars: Policy for handling data variables that are not shared
            by every dataset. Use ``"error"`` to raise a ``ValueError`` listing
            the per-dataset differences, or ``"drop"`` to keep only the
            intersection of shared data variables and emit a warning when any are
            dropped.
        **concat_kwargs: Additional keyword arguments passed to ``xr.concat``.

    Returns:
        Dataset containing gathered versions of the selected data variables.

    Raises:
        ValueError: If ``missing_data_vars="error"`` and the datasets do not
            all have identical data-variable sets, or if ``missing_data_vars`` is
            not recognised.
    """
    dvs = _resolve_shared_data_vars(ds_dict, missing_data_vars=missing_data_vars)

    gathered_dvs = {}

    for dv in dvs:
        da_dict = {k: v[dv] for k, v in ds_dict.items()}
        gathered_dvs[dv] = concat_gather_data_arrays(da_dict, key_dim, ragged_dim, stack_dim, **concat_kwargs)

    return xr.Dataset(gathered_dvs)


def concat_gather_datatree(
    dt: xr.DataTree,
    key_dim: str,
    ragged_dim: str,
    stack_dim: str | None = None,
    missing_data_vars: Literal["error", "drop"] = "error",
    **concat_kwargs,
) -> xr.Dataset:
    """Concatenate xr.DataTree children by gathering ragged coordinates.

    Args:
        dt: DataTree whose children will be converted to datasets and gathered.
        key_dim: Name for the key dimension level in the output MultiIndex.
        ragged_dim: Ragged coordinate dimension to gather across children.
        stack_dim: Optional name for the gathered dimension in the output.
        missing_data_vars: Policy for handling child data variables that are not
            shared by every dataset. See ``concat_gather_datasets``.
        **concat_kwargs: Additional keyword arguments passed to ``xr.concat``.

    Returns:
        Dataset containing gathered versions of the selected data variables.
    """
    ds_dict = {str(k): v.to_dataset() for k, v in dt.items()}
    dvs = _resolve_shared_data_vars(ds_dict, missing_data_vars=missing_data_vars)

    gathered_dvs = {}

    for dv in dvs:
        da_dict = {k: v[dv] for k, v in ds_dict.items()}
        gathered_dvs[dv] = concat_gather_data_arrays(da_dict, key_dim, ragged_dim, stack_dim, **concat_kwargs)

    return xr.Dataset(gathered_dvs)


def _resolve_shared_data_vars(
    ds_dict: Mapping[str, xr.Dataset], *, missing_data_vars: Literal["error", "drop"]
) -> list[Hashable]:
    """Resolve data variables to gather under the requested missing-data policy.

    Args:
        ds_dict: Mapping from dataset key to dataset.
        missing_data_vars: Policy controlling how non-shared data variables are
            handled.

    Returns:
        Ordered list of data variable names to gather.

    Raises:
        ValueError: If datasets have non-identical data-variable sets and
            ``missing_data_vars="error"``, or if the policy is unrecognised.
    """
    if missing_data_vars not in {"error", "drop"}:
        raise ValueError(f"Unknown missing_data_vars policy: {missing_data_vars!r}")

    data_var_sets = {k: set(v.data_vars) for k, v in ds_dict.items()}
    union = set().union(*data_var_sets.values())
    intersection = set.intersection(*data_var_sets.values())

    if missing_data_vars == "error" and any(
        data_vars != intersection for data_vars in data_var_sets.values()
    ):
        differences = []
        for key, dataset in ds_dict.items():
            dataset_vars = set(dataset.data_vars)
            missing = sorted(union - dataset_vars)
            extra = sorted(dataset_vars - intersection)
            details = []
            if missing:
                details.append(f"missing {missing}")
            if extra:
                details.append(f"extra {extra}")
            if details:
                differences.append(f"{key}: " + ", ".join(details))

        raise ValueError("Datasets do not all have the same data variables: " + "; ".join(differences))

    dropped = sorted(union - intersection)
    if missing_data_vars == "drop" and dropped:
        warnings.warn(
            f"Dropping data variables not shared by all datasets: {dropped}",
            UserWarning,
            stacklevel=2,
        )

    # Keep only shared vars, but preserve the first dataset's variable order.
    first_dataset = next(iter(ds_dict.values()))
    return [dv for dv in first_dataset.data_vars if dv in intersection]


# ----------------------------------------
# Align to multi-index
# ----------------------------------------


def align_to_multi_index_level_values(
    da: xr.DataArray,
    *,
    multi_index: xr.DataArray,
    multi_dim: str,
    level: str,
    other_dim: str,
) -> xr.DataArray:
    """Broadcast `da(other_dim, ...)` onto `multi_dim` using a MultiIndex level.

    Raises a ValueError if any other MultiIndex level names are already present
    as coordinate variables on `da`, because this creates ambiguous/conflicting
    index ownership when the MultiIndex is assigned.

    Args:
        da: DataArray to align, e.g. `fp_x_flux` with dimension `other_dim="source"`.
        multi_index: Coordinate for the MultiIndex dimension (e.g. `basis_matrix["state"]`).
            Must be a MultiIndex coordinate that includes the level named by `level`.
        multi_dim: Name of the MultiIndex dimension to align onto (e.g. `"state"`).
        level: Name of the MultiIndex level within `multi_index` to use for alignment
            (e.g. `"source"`).
        other_dim: Dimension on `da` that corresponds to `level` (e.g. `"source"`).

    Returns:
        A DataArray aligned to `multi_dim`, with a canonical MultiIndex coordinate
        installed on `multi_dim`.

    Raises:
        ValueError: If `level` is not a level of `multi_index`, or if other level
            names are already present as coordinates on `da`.
    """
    if other_dim not in da.dims:
        return da

    # Check if da has dims or coords with the same name as the levels of multi_index, besides
    # the level/other_dim that we want to broadcast over.

    # Determine MultiIndex level names
    try:
        mi = multi_index.to_index()  # pandas.MultiIndex
    except Exception as e:  # pragma: no cover
        raise ValueError("multi_index must be a MultiIndex coordinate DataArray.") from e

    if getattr(mi, "names", None) is None:
        raise ValueError("multi_index must be backed by a pandas.MultiIndex.")

    level_names = list(mi.names)
    if level not in level_names:
        raise ValueError(f"Requested level {level!r} not found in multi_index levels {level_names!r}.")

    # If other MultiIndex levels already exist as coordinate variables on `da`,
    # assigning the MultiIndex will conflict (xarray will try to merge coords with same names).
    other_levels = [n for n in level_names if n != level]
    present_as_coords = [n for n in other_levels if n in da.coords]
    if present_as_coords:
        raise ValueError(
            "Cannot align to MultiIndex level because the following MultiIndex level "
            f"name(s) already exist as coordinate variables on the input DataArray: "
            f"{present_as_coords!r}. "
            "This causes coordinate/index conflicts when installing the MultiIndex. "
            "Drop or rename these coordinate(s) first, or avoid this alignment path."
        )

    # Optionally warn if other levels appear as *dimensions* (but not coords).
    # This can still be surprising semantically, even if it does not immediately conflict.
    present_as_dims_only = [n for n in other_levels if (n in da.dims and n not in da.coords)]
    if present_as_dims_only:
        warnings.warn(
            "Aligning to MultiIndex level while other level name(s) are present "
            f"as dimensions without coordinates: {present_as_dims_only!r}. "
            "This may be semantically ambiguous; consider stacking instead.",
            UserWarning,
            stacklevel=2,
        )

    # Vectorized selection in the order of the level values (dim becomes multi_dim)
    level_values = multi_index[level]
    da_sel = da.sel({other_dim: level_values})

    # Drop the old coordinate variable for the selected dimension to avoid the
    # subtle xarray MultiIndex registry/ownership bug.
    if other_dim in da_sel.coords:
        da_sel = da_sel.reset_coords(other_dim, drop=True)

    # Install canonical MultiIndex on multi_dim (brings back level coords cleanly)
    da_sel = da_sel.assign_coords({multi_dim: multi_index})

    return da_sel
