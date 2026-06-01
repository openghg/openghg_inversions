import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions import utils
from openghg_inversions.postprocessing.legacy_outputs import _compute_apriori_flux


def test_compute_apriori_flux_handles_missing_month():
    """Apriori flux weighting handles skipped monthly flux periods."""
    flux = xr.DataArray(
        np.array([[[1.0, 3.0]]]),
        dims=["lat", "lon", "flux_time"],
        coords={
            "lat": [0.0],
            "lon": [0.0],
            "flux_time": pd.to_datetime(["2019-01-01", "2019-03-01"]),
        },
    )
    times = xr.DataArray(
        pd.to_datetime(["2019-01-15", "2019-01-20", "2019-03-10", "2019-03-20"]),
        dims=["nmeasure"],
    )

    apriori_flux = _compute_apriori_flux(flux, "2019-01-01", "2019-04-01", times)

    xr.testing.assert_allclose(
        apriori_flux, xr.DataArray([[2.0]], dims=["lat", "lon"], coords={"lat": [0.0], "lon": [0.0]})
    )


def test_map_times_to_available_period_positions_handles_gappy_flux_months():
    """Period mapping uses the nearest available monthly flux positions."""
    times = pd.to_datetime(["2019-01-15", "2019-01-20", "2019-03-10", "2019-04-20"])
    flux_times = pd.to_datetime(["2019-01-01", "2019-03-01", "2019-04-01"])

    positions = utils._map_times_to_available_period_positions(times, flux_times, "monthly")

    np.testing.assert_array_equal(positions, np.array([0, 0, 1, 2]))


def test_compute_apriori_flux_handles_multi_year_flux_time():
    """Apriori flux weighting handles yearly flux periods across multiple years."""
    flux = xr.DataArray(
        np.array([[[1.0, 2.0, 3.0]]]),
        dims=["lat", "lon", "flux_time"],
        coords={
            "lat": [0.0],
            "lon": [0.0],
            "flux_time": pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01"]),
        },
    )
    times = xr.DataArray(
        pd.to_datetime(["2023-03-15", "2023-11-20", "2024-07-10", "2025-04-20"]),
        dims=["nmeasure"],
    )

    apriori_flux = _compute_apriori_flux(flux, "2023-01-01", "2025-05-01", times)

    xr.testing.assert_allclose(
        apriori_flux,
        xr.DataArray([[1.75]], dims=["lat", "lon"], coords={"lat": [0.0], "lon": [0.0]}),
    )
