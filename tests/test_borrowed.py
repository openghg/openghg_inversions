"""Runtime tests for experimental borrowed-array static markers."""

import dask.array as da
import numpy as np
import xarray as xr

from openghg_inversions.borrowed import BorrowedDataArray, BorrowedNDArray, borrow


def test_borrow_numpy_preserves_runtime_identity() -> None:
    """Borrowing a NumPy array adds only a static marker."""
    array = np.arange(3)

    borrowed: BorrowedNDArray = borrow(array)

    assert borrowed is array
    assert type(borrowed) is np.ndarray
    borrowed[0] = 10  # type: ignore[assignment]
    assert array[0] == 10


def test_borrowed_numpy_copy_is_independent() -> None:
    """Copying a borrowed NumPy array returns independent mutable data."""
    array = np.arange(3)
    borrowed: BorrowedNDArray = borrow(array)

    mutable_copy = borrowed.copy()
    mutable_copy[0] = 10

    assert type(mutable_copy) is np.ndarray
    assert not np.shares_memory(mutable_copy, array)
    assert array[0] == 0


def test_borrow_data_array_preserves_runtime_identity() -> None:
    """Borrowing a DataArray preserves the object and its NumPy buffer."""
    data = np.arange(3)
    data_array = xr.DataArray(data, dims="x")

    borrowed: BorrowedDataArray = borrow(data_array)

    assert borrowed is data_array
    assert type(borrowed) is xr.DataArray
    assert borrowed.data is data


def test_borrowed_data_array_copy_respects_xarray_depth() -> None:
    """Shallow DataArray copies share data while deep copies are independent."""
    data_array = xr.DataArray(np.arange(3), dims="x")
    borrowed: BorrowedDataArray = borrow(data_array)

    shallow_copy = borrowed.copy(deep=False)
    deep_copy = borrowed.copy(deep=True)

    assert shallow_copy.data is data_array.data
    assert deep_copy.data is not data_array.data

    deep_copy.data[0] = 10
    assert data_array.data[0] == 0


def test_borrow_dask_data_array_preserves_lazy_graph() -> None:
    """Borrowing a Dask-backed DataArray neither copies nor rebuilds its graph."""
    lazy_data = da.arange(6, chunks="auto")
    data_array = xr.DataArray(lazy_data, dims="x")

    borrowed: BorrowedDataArray = borrow(data_array)

    assert borrowed is data_array
    assert borrowed.data is lazy_data
    assert borrowed.data.__dask_graph__() is lazy_data.__dask_graph__()


def test_borrowed_dask_data_array_shallow_copy_preserves_graph() -> None:
    """A shallow DataArray copy preserves the exact lazy backend and graph."""
    lazy_data = da.arange(6, chunks="auto")
    borrowed: BorrowedDataArray = borrow(xr.DataArray(lazy_data, dims="x"))

    shallow_copy = borrowed.copy(deep=False)

    assert shallow_copy.data is lazy_data
    assert shallow_copy.data.__dask_graph__() is lazy_data.__dask_graph__()
