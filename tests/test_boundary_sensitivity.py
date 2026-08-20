"""Tests for the sampled boundary-sensitivity preparation owner."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.boundary_sensitivity import BoundarySensitivity


def _sensitivity(data: object | None = None) -> xr.DataArray:
    times = pd.date_range("2020-01-01", periods=3, freq="12h")
    return xr.DataArray(
        np.arange(6, dtype=float).reshape(2, 3) if data is None else data,
        dims=("bc_region", "time"),
        coords={"bc_region": ["north", "south"], "time": times},
        attrs={"units": "ppm"},
    )


def test_prepare_boundary_sensitivity_labels_one_period_and_provenance() -> None:
    result = BoundarySensitivity.prepare(_sensitivity()).data

    assert result.dims == ("bc_region", "time")
    assert result.indexes["bc_region"].names == ["bc_curtain", "bc_period"]
    assert result.attrs["units"] == "ppm"
    assert result.attrs["bc_frequency"] == "none"
    np.testing.assert_allclose(result.values, _sensitivity().values)


def test_prepare_boundary_sensitivity_reorders_and_rejects_missing_observations() -> None:
    sensitivity = _sensitivity()
    reordered = sensitivity.time[[2, 0]]

    result = BoundarySensitivity.prepare(sensitivity, observation_labels=reordered).data

    np.testing.assert_array_equal(result.time, reordered)
    with pytest.raises(ValueError, match="missing observation label"):
        BoundarySensitivity.prepare(
            sensitivity,
            observation_labels=xr.DataArray(
                [np.datetime64("2021-01-01")], dims="time", name="time"
            ),
        )


def test_prepare_boundary_sensitivity_validates_states_and_values() -> None:
    with pytest.raises(ValueError, match="bc_region.*unique"):
        BoundarySensitivity.prepare(_sensitivity().assign_coords(bc_region=["north", "north"]))
    with pytest.raises(ValueError, match="finite"):
        BoundarySensitivity.prepare(_sensitivity([[1.0, np.nan, 2.0], [3.0, 4.0, 5.0]]))


def test_prepare_boundary_sensitivity_preserves_lazy_data_and_borrowed_input() -> None:
    source = _sensitivity(da.arange(6, chunks=3).reshape((2, 3)))
    original = source.copy(deep=False)

    result = BoundarySensitivity.prepare(source, frequency="12h", anchor_time="2020-01-01").data

    assert hasattr(result.data, "__dask_graph__")
    xr.testing.assert_identical(source, original)


def test_boundary_sensitivity_installs_without_mutating_gathered_inputs() -> None:
    raw = _sensitivity().rename(time="nmeasure").assign_coords(
        time=("nmeasure", _sensitivity().time.values)
    )
    inputs = xr.Dataset({"H_bc": raw, "mf": ("nmeasure", np.ones(3))})
    original = inputs.copy(deep=False)

    result = BoundarySensitivity.prepare(inputs["H_bc"]).install(inputs)

    xr.testing.assert_identical(inputs, original)
    assert result["H_bc"].indexes["bc_region"].names == ["bc_curtain", "bc_period"]
