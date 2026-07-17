"""Tests for the repository-local TAC/MHD dyadic demo adapter."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.experimental.dyadic.demo_data import DemoDesignData, load_tac_mhd_demo_data

_DATA_DIRECTORY = Path(__file__).parents[2] / "data"


@pytest.fixture(scope="module")
def demo_data() -> DemoDesignData:
    """Load the real committed TAC/MHD demo fixtures once for this module."""
    return load_tac_mhd_demo_data(_DATA_DIRECTORY)


def test_load_tac_mhd_demo_data_reconstructs_frozen_row_design(demo_data: DemoDesignData) -> None:
    """The adapter should preserve frozen rows and match frozen sensitivity totals."""
    assert demo_data.G.shape == (47, 293, 391)
    assert demo_data.G.dtype == np.float32
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
