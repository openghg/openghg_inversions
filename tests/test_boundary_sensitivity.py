"""Tests for the sampled boundary-sensitivity preparation owner."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.boundary_sensitivity import BoundaryAlignment
from openghg_inversions.inversion_inputs import make_inv_inputs


def _sensitivity(data: object | None = None) -> xr.DataArray:
    times = pd.date_range("2020-01-01", periods=3, freq="12h")
    return xr.DataArray(
        np.arange(6, dtype=float).reshape(2, 3) if data is None else data,
        dims=("bc_region", "time"),
        coords={"bc_region": ["north", "south"], "time": times},
        attrs={"units": "ppm"},
    )


def test_prepare_boundary_sensitivity_labels_one_period_and_provenance() -> None:
    result = BoundaryAlignment.prepare(_sensitivity()).data

    assert result.dims == ("bc_region", "time")
    assert result.indexes["bc_region"].names == ["bc_curtain", "bc_period"]
    assert result.attrs["units"] == "ppm"
    assert result.attrs["bc_frequency"] == "none"
    np.testing.assert_allclose(result.values, _sensitivity().values)


def test_prepare_boundary_sensitivity_reorders_and_rejects_missing_observations() -> None:
    sensitivity = _sensitivity()
    reordered = sensitivity.time[[2, 0]]

    result = BoundaryAlignment.prepare(sensitivity, observation_labels=reordered).data

    np.testing.assert_array_equal(result.time, reordered)
    with pytest.raises(ValueError, match="missing observation label"):
        BoundaryAlignment.prepare(
            sensitivity,
            observation_labels=xr.DataArray(
                [np.datetime64("2021-01-01")], dims="time", name="time"
            ),
        )


def test_prepare_boundary_sensitivity_validates_states() -> None:
    with pytest.raises(ValueError, match="bc_region.*unique"):
        BoundaryAlignment.prepare(_sensitivity().assign_coords(bc_region=["north", "north"]))


def test_prepare_boundary_sensitivity_preserves_lazy_data_and_borrowed_input() -> None:
    source = _sensitivity(da.arange(6, chunks=3).reshape((2, 3)))
    original = source.copy(deep=False)

    result = BoundaryAlignment.prepare(source, frequency="12h", anchor_time="2020-01-01").data

    assert hasattr(result.data, "__dask_graph__")
    xr.testing.assert_identical(source, original)


@pytest.mark.parametrize("lazy", [False, True])
def test_make_inv_inputs_drops_nan_boundary_rows_for_eager_and_lazy_data(lazy: bool) -> None:
    time = _sensitivity().time
    h_bc = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    if lazy:
        h_bc = da.from_array(h_bc, chunks=(2, 3))
    site_data = xr.Dataset(
        {
            "H": (("region", "time"), np.ones((1, 3))),
            "H_bc": (("bc_region", "time"), h_bc),
            "mf": ("time", np.ones(3)),
            "mf_error": ("time", np.ones(3)),
            "mf_repeatability": ("time", np.ones(3)),
            "mf_variability": ("time", np.ones(3)),
        },
        coords={"region": [0], "bc_region": ["north", "south"], "time": time},
    )

    result = make_inv_inputs({"AAA": site_data}, sites=["AAA"])

    assert result.sizes["nmeasure"] == 2
    assert np.isfinite(result["H_bc"]).all()
