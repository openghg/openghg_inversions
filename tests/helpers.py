"""Helper functions for tests.

Mainly for creating fake data.
"""
import numpy  as np
import pandas as pd
import xarray as xr


def lat_lon_data(nlat: int, nlon: int, values: np.ndarray | list | None = None) -> xr.DataArray:
    lat = np.arange(nlat)
    lon = np.arange(nlon)

    if values is None:
        values = np.ones((nlat, nlon))

    return xr.DataArray(values, coords=[lat, lon], dims=["lat", "lon"])


rng = np.random.default_rng(seed=123)


def basis_function(nlat: int, nlon: int, nbasis: int) -> xr.DataArray:
    values = rng.integers(1, nbasis, size=nlat*nlon, endpoint=True).reshape((nlat, nlon))
    return lat_lon_data(nlat, nlon, values)


def lat_lon_time_data(nlat: int, nlon: int, start: str, end: str, ntime: int, values: np.ndarray | list | None = None) -> xr.DataArray:
    lat = np.arange(nlat)
    lon = np.arange(nlon)
    time = pd.date_range(start, end, ntime)

    if values is None:
        values = np.ones((nlat, nlon, ntime))

    return xr.DataArray(values, coords=[lat, lon, time], dims=["lat", "lon", "time"])


def footprint(nlat: int, nlon: int, start: str, end: str, ntime: int) -> xr.DataArray:
    values = np.broadcast_to(np.arange(1, ntime + 1), (nlat, nlon, ntime))

    return lat_lon_time_data(nlat, nlon, start, end, ntime, values)
