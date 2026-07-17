"""Tests for the repository-local TAC/MHD dyadic demo adapter."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.experimental.dyadic.demo_data import (
    DemoDesignData,
    load_tac_mhd_demo_data,
    load_tac_mhd_week_demo_data,
)

_DATA_DIRECTORY = Path(__file__).parents[2] / "data"


@pytest.fixture(scope="module")
def demo_data() -> DemoDesignData:
    """Load the real committed TAC/MHD demo fixtures once for this module."""
    return load_tac_mhd_demo_data(_DATA_DIRECTORY)


@pytest.fixture(scope="module")
def week_demo_data() -> DemoDesignData:
    """Load the committed full-week TAC/MHD fixtures once for this module."""
    return load_tac_mhd_week_demo_data(_DATA_DIRECTORY)


def test_load_tac_mhd_demo_data_reconstructs_frozen_row_design(demo_data: DemoDesignData) -> None:
    """The adapter should preserve frozen rows and match frozen sensitivity totals."""
    assert demo_data.G.shape == (47, 293, 391)
    assert demo_data.G.dtype == np.float32
    assert demo_data.prior_flux.shape == demo_data.G.shape[1:]
    assert demo_data.prior_flux.dtype == np.float32
    assert demo_data.y.shape == demo_data.error.shape == demo_data.min_error.shape == (47,)
    assert np.count_nonzero(demo_data.sites == "MHD") == 23
    assert np.count_nonzero(demo_data.sites == "TAC") == 24
    assert np.all(demo_data.sites[:23] == "MHD")
    assert np.all(demo_data.sites[23:] == "TAC")
    assert demo_data.times[0] == np.datetime64("2019-01-01T01:00:00", "ns")
    assert demo_data.times[22] == np.datetime64("2019-01-01T23:00:00", "ns")
    assert demo_data.times[23] == np.datetime64("2019-01-01T00:00:00", "ns")
    assert demo_data.times[-1] == np.datetime64("2019-01-01T23:00:00", "ns")
    assert np.isfinite(demo_data.G).all()
    assert np.all(demo_data.error > 0.0)
    assert np.all(demo_data.min_error > 0.0)

    with np.load(_DATA_DIRECTORY / "frozen_mhd_tac_make_inv_inputs_hbmcmc.npz") as frozen:
        reference_totals = frozen["mcmc__Hx"].sum(axis=0, dtype=np.float64)
    reconstructed_totals = demo_data.G.sum(axis=(1, 2), dtype=np.float64)
    np.testing.assert_allclose(reconstructed_totals, reference_totals, rtol=4.0e-6, atol=2.0e-5)


def test_load_tac_mhd_demo_data_adopts_footprint_coordinates(demo_data: DemoDesignData) -> None:
    """Small flux rounding differences should be accepted without replacing the footprint grid."""
    with (
        xr.open_dataset(
            _DATA_DIRECTORY / "footprints_mhd_europe_name_10m_2019-01-01_2019-01-07_data.nc"
        ) as footprint,
        xr.open_dataset(
            _DATA_DIRECTORY / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc"
        ) as flux,
    ):
        footprint_lat = footprint["latitude"].values
        footprint_lon = footprint["longitude"].values
        flux_lat = flux["lat"].values
        flux_lon = flux["lon"].values

    np.testing.assert_array_equal(demo_data.lat, footprint_lat)
    np.testing.assert_array_equal(demo_data.lon, footprint_lon)
    assert not np.array_equal(footprint_lat, flux_lat)
    assert not np.array_equal(footprint_lon, flux_lon)
    np.testing.assert_allclose(footprint_lat, flux_lat, rtol=0.0, atol=1.0e-4)
    np.testing.assert_allclose(footprint_lon, flux_lon, rtol=0.0, atol=1.0e-4)


def test_demo_data_coarsening_preserves_every_observation_sum(demo_data: DemoDesignData) -> None:
    """Boundary-aware factor-eight coarsening should preserve each design-row total."""
    coarsened = demo_data.coarsen(8)

    assert coarsened.values.shape == (47, 37, 49)
    assert coarsened.support_counts.shape == (37, 49)
    np.testing.assert_allclose(
        coarsened.values.sum(axis=(1, 2)),
        demo_data.G.sum(axis=(1, 2)),
        rtol=2.0e-6,
        atol=2.0e-5,
    )


def test_load_tac_mhd_week_demo_data_aligns_all_available_hours(
    week_demo_data: DemoDesignData,
) -> None:
    """The week adapter should retain exact site-major hours without imputation."""
    assert week_demo_data.G.shape == (333, 293, 391)
    assert week_demo_data.G.dtype == np.float32
    assert week_demo_data.prior_flux.shape == week_demo_data.G.shape[1:]
    assert week_demo_data.prior_flux.dtype == np.float32
    assert week_demo_data.y.shape == week_demo_data.error.shape == week_demo_data.min_error.shape == (333,)
    assert np.count_nonzero(week_demo_data.sites == "MHD") == 165
    assert np.count_nonzero(week_demo_data.sites == "TAC") == 168
    assert np.all(week_demo_data.sites[:165] == "MHD")
    assert np.all(week_demo_data.sites[165:] == "TAC")

    mhd_times = week_demo_data.times[week_demo_data.sites == "MHD"]
    tac_times = week_demo_data.times[week_demo_data.sites == "TAC"]
    full_week = np.arange(
        np.datetime64("2019-01-01T00:00:00", "ns"),
        np.datetime64("2019-01-08T00:00:00", "ns"),
        np.timedelta64(1, "h"),
    )
    expected_missing_mhd = np.array(
        [
            "2019-01-01T00:00:00",
            "2019-01-07T13:00:00",
            "2019-01-07T22:00:00",
        ],
        dtype="datetime64[ns]",
    )
    np.testing.assert_array_equal(full_week[~np.isin(full_week, mhd_times)], expected_missing_mhd)
    np.testing.assert_array_equal(tac_times, full_week)
    assert np.unique(mhd_times).size == mhd_times.size
    assert np.unique(tac_times).size == tac_times.size
    assert np.isfinite(week_demo_data.G).all()
    assert np.isfinite(week_demo_data.y).all()
    assert np.all(week_demo_data.error > 0.0)
    assert np.all(week_demo_data.min_error > 0.0)
    assert "not the production model-error likelihood" in week_demo_data.benchmark_error_description


def test_load_tac_mhd_week_demo_data_reproduces_frozen_first_day(
    week_demo_data: DemoDesignData,
) -> None:
    """First-day observations, errors, and design totals should match the frozen fixture."""
    with np.load(_DATA_DIRECTORY / "frozen_mhd_tac_make_inv_inputs_hbmcmc.npz") as frozen:
        frozen_sites = np.where(frozen["mcmc__siteindicator"] == 0, "MHD", "TAC")
        frozen_times = np.asarray(frozen["post__Ytime"], dtype="datetime64[ns]")
        frozen_y = np.asarray(frozen["mcmc__Y"], dtype=np.float64)
        frozen_error = np.asarray(frozen["mcmc__error"], dtype=np.float64)
        reference_totals = frozen["mcmc__Hx"].sum(axis=0, dtype=np.float64)

    matched_indices = np.array(
        [
            np.flatnonzero((week_demo_data.sites == site) & (week_demo_data.times == time)).item()
            for site, time in zip(frozen_sites, frozen_times)
        ]
    )
    np.testing.assert_allclose(week_demo_data.y[matched_indices], frozen_y, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        week_demo_data.error[matched_indices],
        frozen_error,
        rtol=0.0,
        atol=1.0e-7,
    )
    reconstructed_totals = week_demo_data.G[matched_indices].sum(axis=(1, 2), dtype=np.float64)
    np.testing.assert_allclose(reconstructed_totals, reference_totals, rtol=4.0e-6, atol=2.0e-5)


def test_load_tac_mhd_week_demo_data_adopts_aligned_footprint_grid(
    week_demo_data: DemoDesignData,
) -> None:
    """The full-week design should retain footprint coordinates after flux tolerance checks."""
    with (
        xr.open_dataset(
            _DATA_DIRECTORY / "footprints_mhd_europe_name_10m_2019-01-01_2019-01-07_data.nc"
        ) as footprint,
        xr.open_dataset(
            _DATA_DIRECTORY / "flux_total_ch4_europe_edgar7_2019-01-01_2019-12-31_data.nc"
        ) as flux,
    ):
        footprint_lat = footprint["latitude"].values
        footprint_lon = footprint["longitude"].values
        flux_lat = flux["lat"].values
        flux_lon = flux["lon"].values

    np.testing.assert_array_equal(week_demo_data.lat, footprint_lat)
    np.testing.assert_array_equal(week_demo_data.lon, footprint_lon)
    np.testing.assert_allclose(week_demo_data.lat, flux_lat, rtol=0.0, atol=1.0e-4)
    np.testing.assert_allclose(week_demo_data.lon, flux_lon, rtol=0.0, atol=1.0e-4)


def test_week_demo_data_coarsening_preserves_every_observation_sum(
    week_demo_data: DemoDesignData,
) -> None:
    """Boundary-aware coarsening should preserve all 333 full-week design-row totals."""
    coarsened = week_demo_data.coarsen(8)

    assert coarsened.values.shape == (333, 37, 49)
    assert coarsened.support_counts.shape == (37, 49)
    np.testing.assert_allclose(
        coarsened.values.sum(axis=(1, 2)),
        week_demo_data.G.sum(axis=(1, 2)),
        rtol=2.0e-6,
        atol=2.0e-5,
    )
