from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.inversion_data.scenario import _snap_footprint_times_to_obs


def _data_object(times: pd.DatetimeIndex) -> SimpleNamespace:
    return SimpleNamespace(data=xr.Dataset(coords={"time": times}))


def test_snap_footprint_times_to_obs_handles_submicrosecond_precision_gap() -> None:
    obs_times = pd.DatetimeIndex(
        [
            "2023-01-01T04:32:05.088015872",
            "2023-01-01T04:32:13.916010752",
            "2023-01-01T04:32:21.642522624",
        ]
    )
    footprint_times = obs_times.floor("us")
    obs_data = _data_object(obs_times)
    footprint_data = _data_object(footprint_times)

    assert len(obs_times.intersection(footprint_times)) == 0

    _snap_footprint_times_to_obs(obs_data, footprint_data)

    np.testing.assert_array_equal(footprint_data.data.time.values, obs_data.data.time.values)


def test_snap_footprint_times_to_obs_handles_second_precision_gap() -> None:
    obs_times = pd.DatetimeIndex(
        [
            "2023-01-01T04:32:05.800000000",
            "2023-01-01T04:32:13.100000000",
            "2023-01-01T04:32:21.500000000",
        ]
    )
    footprint_times = obs_times.floor("s")
    obs_data = _data_object(obs_times)
    footprint_data = _data_object(footprint_times)

    _snap_footprint_times_to_obs(obs_data, footprint_data)

    np.testing.assert_array_equal(footprint_data.data.time.values, obs_data.data.time.values)


def test_snap_footprint_times_to_obs_promotes_coarse_datetime_dtype() -> None:
    """Snapping must retain nanosecond observations from a microsecond coordinate."""
    obs_times = np.array(
        ["2023-01-01T04:32:05.088015872", "2023-01-01T04:32:13.916010752"],
        dtype="datetime64[ns]",
    )
    footprint_times = obs_times.astype("datetime64[us]")
    obs_data = _data_object(obs_times)
    footprint_data = _data_object(footprint_times)

    assert footprint_data.data.time.dtype == np.dtype("datetime64[us]")

    _snap_footprint_times_to_obs(obs_data, footprint_data)

    assert footprint_data.data.time.dtype == np.dtype("datetime64[ns]")
    np.testing.assert_array_equal(footprint_data.data.time.values, obs_data.data.time.values)


def test_snap_footprint_times_to_obs_skips_ambiguous_matches() -> None:
    obs_times = pd.DatetimeIndex(["2023-01-01T00:00:00.000000000"])
    footprint_times = pd.DatetimeIndex(
        [
            "2023-01-01T00:00:00.000000100",
            "2023-01-01T00:00:00.000000200",
        ]
    )
    obs_data = _data_object(obs_times)
    footprint_data = _data_object(footprint_times)

    _snap_footprint_times_to_obs(obs_data, footprint_data)

    np.testing.assert_array_equal(footprint_data.data.time.values, footprint_times.values)
