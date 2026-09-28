"""Focused regression tests for retained postprocessing helpers."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.postprocessing.make_paris_outputs import (
    DEFAULT_PARIS_TEMPLATE_VERSION,
    _assign_flux_time_bounds,
    _flux_interval_midpoints,
    infer_flux_frequency,
    paris_template_files,
)


def test_flux_interval_midpoints_filter_non_overlapping_times() -> None:
    """Only flux periods overlapping the inversion contribute output times."""
    flux_times = [pd.Timestamp(f"{year}-01-01") for year in range(2012, 2025)]

    midpoints, valid_indices = _flux_interval_midpoints(
        flux_times,
        pd.DateOffset(years=1),
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2024-01-01"),
    )

    assert valid_indices == [11]
    assert midpoints == [pd.Timestamp("2023-07-02 12:00:00")]


def test_flux_interval_midpoints_clip_annual_prior_to_june_inversion() -> None:
    """A retained January annual prior is clipped to the June run period."""
    midpoints, valid_indices = _flux_interval_midpoints(
        [pd.Timestamp("2019-01-01")],
        pd.DateOffset(years=1),
        pd.Timestamp("2019-06-01"),
        pd.Timestamp("2019-07-01"),
    )

    assert valid_indices == [0]
    assert midpoints == [pd.Timestamp("2019-06-16")]


@pytest.mark.parametrize(
    ("time_period", "expected"),
    [
        ("annual", "yearly"),
        ("1 YEAR", "yearly"),
        ("monthly", "monthly"),
        ("1 Month", "monthly"),
    ],
)
def test_infer_flux_frequency_normalizes_period_spellings(
    time_period: str,
    expected: str,
) -> None:
    flux = xr.DataArray([1.0], dims="flux_time", attrs={"time_period": time_period})

    assert infer_flux_frequency(flux) == expected


def test_infer_flux_frequency_recognizes_calendar_periods_without_attrs() -> None:
    annual = xr.DataArray(
        np.ones(3),
        dims="flux_time",
        coords={"flux_time": pd.to_datetime(["2019-01-01", "2020-01-01", "2021-01-01"])},
    )
    monthly = xr.DataArray(
        np.ones(3),
        dims="flux_time",
        coords={"flux_time": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])},
    )

    assert infer_flux_frequency(annual) == "yearly"
    assert infer_flux_frequency(monthly) == "monthly"


@pytest.mark.parametrize("time_period", ["2 years", "3 months", "0 days", "-1 day"])
def test_infer_flux_frequency_rejects_unsupported_period(time_period: str) -> None:
    flux = xr.DataArray([1.0], dims="flux_time", attrs={"time_period": time_period})

    with pytest.raises(ValueError, match="Flux period"):
        infer_flux_frequency(flux)


def test_assign_flux_time_bounds_reports_empty_flux_times() -> None:
    flux = xr.Dataset(coords={"time": pd.DatetimeIndex([])})

    with pytest.raises(ValueError, match="no flux timestamps"):
        _assign_flux_time_bounds(
            flux,
            flux_frequency="yearly",
            inv_start=pd.Timestamp("2021-01-01"),
            inv_end=pd.Timestamp("2021-02-01"),
        )


def test_paris_template_registry_requires_explicit_latest() -> None:
    """PARIS output keeps legacy templates as the compatibility default."""
    legacy = paris_template_files(DEFAULT_PARIS_TEMPLATE_VERSION)
    latest = paris_template_files("latest")

    assert DEFAULT_PARIS_TEMPLATE_VERSION == "legacy"
    assert legacy.concentration_version == "v03"
    assert legacy.flux_version == "legacy"
    assert latest.concentration_version == "v04"
    assert latest.flux_version == "v03"
    assert legacy.concentration.exists()
    assert legacy.flux.exists()
    assert latest.concentration.exists()
    assert latest.flux.exists()
