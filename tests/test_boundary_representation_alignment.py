"""Ordinary and centered boundaries share model-owned observation periods."""

import dask.array as da
from dask.callbacks import Callback
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.boundary_sensitivity import (
    align_boundary_operands, bind_boundary_sensitivity,
)


@pytest.mark.parametrize("frequency,anchor", [(None, None), ("monthly", None), ("1D", "2020-01-30")])
def test_reusable_ordinary_and_centered_share_selected_periods(frequency, anchor):
    times = pd.to_datetime(["2020-01-30T12:00", "2020-01-31T12:00", "2020-02-01T12:00"])
    anomaly = xr.DataArray(da.from_array([[.1, .3, -.2], [.4, -.1, .2]], chunks=(1, 2)),
                          dims=("bc_region", "nmeasure"),
                          coords={"bc_region": ["land", "ocean"], "nmeasure": ["a", "b", "c"],
                                  "time": ("nmeasure", times)}, attrs={"units": "ppm"})
    raw = xr.Dataset({"H_bc_anomaly": anomaly, "H_bc_mass": xr.full_like(anomaly, .5).assign_attrs(units="1"),
                      "bc_reference": xr.DataArray(400., attrs={"units": "ppm"})},
                     attrs={"boundary_representation": "centered-v1"})
    absolute = (raw.H_bc_anomaly + raw.bc_reference * raw.H_bc_mass).assign_attrs(
        units="ppm", boundary_sensitivity_preparation="raw-spatial")
    observations = xr.DataArray([1., 2.], dims="nmeasure", coords={"nmeasure": ["c", "a"]})
    calls = []
    with Callback(posttask=lambda *args: calls.append(args)):
        ordinary = bind_boundary_sensitivity(absolute, observations, frequency=frequency, anchor_time=anchor)
        centered = align_boundary_operands(raw, frequency=frequency, anchor_time=anchor,
                                          observation_labels=observations.nmeasure)
    assert not calls
    assert hasattr(ordinary.data, "__dask_graph__")
    expected = (centered.H_bc_anomaly + centered.bc_reference * centered.H_bc_mass).transpose(*ordinary.dims)
    xr.testing.assert_allclose(ordinary.compute(), expected.compute())
    assert list(ordinary.nmeasure.values) == ["c", "a"]
    assert absolute.attrs["boundary_sensitivity_preparation"] == "raw-spatial"
    # Previously expanded data have a different contract from reusable raw data.
    with pytest.raises(ValueError, match="raw spatial"):
        bind_boundary_sensitivity(ordinary, observations, frequency="monthly")


@pytest.mark.parametrize("frequency,anchor", [(None, "NaT"), ("monthly", "2020-01-01"), ("1D", "NaT")])
def test_ordinary_binding_rejects_invalid_or_ignored_anchor(frequency, anchor):
    sensitivity = xr.DataArray([[1.]], dims=("nmeasure", "bc_region"),
                               coords={"nmeasure": [0], "bc_region": ["north"]})
    observations = xr.DataArray([1.], dims="nmeasure", coords={"nmeasure": [0]})
    with pytest.raises(ValueError, match="anchor_time"):
        bind_boundary_sensitivity(sensitivity, observations, frequency=frequency, anchor_time=anchor)
