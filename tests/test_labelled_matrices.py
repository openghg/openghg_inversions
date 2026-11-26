"""Test shared labelled square-matrix construction."""

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions._labelled_matrices import coordinates_on_dims, square_matrix


def _labels() -> xr.DataArray:
    """Return an array whose state axis has a MultiIndex, an auxiliary and a scalar coordinate."""
    index = pd.MultiIndex.from_tuples([("a", 0), ("a", 1), ("b", 0)], names=("source", "region"))
    return xr.DataArray(
        np.zeros((2, 3)),
        dims=("lat", "state"),
        coords={
            "lat": [50.0, 51.0],
            "weight": ("state", [1.0, 2.0, 3.0]),
            "campaign": "test",
            **xr.Coordinates.from_pandas_multiindex(index, "state"),
        },
    )


def test_coordinates_on_dims_keeps_only_coordinates_on_those_dims() -> None:
    """Index, MultiIndex levels, auxiliary and scalar coordinates on the axis are kept."""
    coords = coordinates_on_dims(_labels(), {"state"})

    assert set(coords) == {"state", "source", "region", "weight", "campaign"}


def test_square_matrix_labels_rows_and_columns_from_one_axis() -> None:
    """Both axes carry the state labels, with typed, renamed column coordinates."""
    labels = _labels()
    values = np.arange(9.0).reshape(3, 3)

    matrix = square_matrix(values, labels, row_dim="state", column_dim="state_cov", name="c", attrs={"units": "1"})

    assert matrix.dims == ("state", "state_cov")
    np.testing.assert_array_equal(matrix.values, values)
    assert matrix.indexes["state"].equals(labels.indexes["state"])
    assert list(matrix.indexes["state_cov"]) == list(labels.indexes["state"])
    assert matrix.indexes["state_cov"].names == ["source_cov", "region_cov"]
    np.testing.assert_array_equal(matrix["weight_cov"].values, [1.0, 2.0, 3.0])
    assert matrix.name == "c"
    assert matrix.attrs == {"units": "1"}
