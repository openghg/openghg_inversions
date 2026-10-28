"""Native release history and declared flux interval support."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from openghg_inversions.inversion_data._surface_support import (
    required_flux_support,
    validate_flux_coverage,
)


@pytest.mark.parametrize("history", [24, 120, 720])
@pytest.mark.parametrize("release", ["2020-03-01", "2021-01-01", "2020-03-16"])
@pytest.mark.parametrize("legacy", [False, True])
def test_required_support_uses_actual_history_and_only_release_months(history, release, legacy):
    first = pd.Timestamp(release)
    hours = np.arange(history + int(legacy))
    fp = xr.Dataset(coords={"time": pd.date_range(first, periods=2, freq="h"), "H_back": hours})
    if legacy:
        fp["fp_HiTRes"] = xr.DataArray(da.ones((2, len(hours))), dims=("time", "H_back"))
    with Callback(pretask=lambda *args: pytest.fail("support computed payload")):
        start, end = required_flux_support(fp, mode="time_resolved")
    assert start == min(first - pd.Timedelta(hours=history), first.to_period("M").start_time)
    assert end == (first.to_period("M") + 1).start_time


def test_support_accepts_exact_exclusive_end_and_does_not_require_history_month():
    fp = xr.Dataset(coords={"time": pd.date_range("2020-03-01", periods=2, freq="h"), "H_back": range(24)})
    times = pd.date_range("2020-02-29", "2020-04-01", freq="h", inclusive="left")
    flux = xr.Dataset({"flux": ("time", da.ones(len(times), chunks=24))}, coords={"time": times})
    with Callback(pretask=lambda *args: pytest.fail("coverage computed payload")):
        validate_flux_coverage(flux, fp, source="a", mode="time_resolved")
    for sliced in (flux.isel(time=slice(1, None)), flux.isel(time=slice(None, -1))):
        with pytest.raises(ValueError, match="Inadequate flux coverage"):
            validate_flux_coverage(sliced, fp, source="a", mode="time_resolved")


@pytest.mark.parametrize("cadence", [3, 24 * 31])
def test_explicit_resolved_rejects_provider_monthly_fallback(cadence):
    fp = xr.Dataset(coords={"time": pd.date_range("2020-01-01", periods=2, freq="h"), "H_back": [0, 1]})
    times = pd.date_range("2019-11-01", periods=1000, freq=f"{cadence}h")
    flux = xr.Dataset({"flux": ("time", np.ones(len(times)))}, coords={"time": times})
    with pytest.raises(ValueError, match="provider would substitute monthly flux"):
        validate_flux_coverage(flux, fp, source="a", mode="time_resolved")
