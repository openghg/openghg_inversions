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



@pytest.mark.parametrize("flux_frequency", ["NaT", "0 days", "-1 day"])
def test_assign_flux_time_bounds_rejects_nonpositive_explicit_period(flux_frequency):
    """Explicit PARIS periods must be finite and positive."""
    flux = xr.Dataset(coords={"time": pd.to_datetime(["2021-01-01"])})

    with pytest.raises(ValueError, match="positive fixed duration"):
        _assign_flux_time_bounds(
            flux,
            flux_frequency=flux_frequency,
            inv_start=pd.Timestamp("2021-01-01"),
            inv_end=pd.Timestamp("2021-02-01"),
        )


def test_paris_flux_output_timestamp(inv_out, europe_country_file):
    """Check that the flux output time coordinate is the midpoint of the inversion period.

    The flux file has a yearly period but the inversion is shorter; the output
    timestamp should be the midpoint of the overlap between the flux interval
    and the inversion period (i.e. the midpoint of the inversion period itself),
    not 6 months into the flux's own year.
    """
    flux_outs = paris_flux_output(inv_out, country_file=europe_country_file, flux_frequency="yearly")

    # time is stored as days since Unix epoch; convert back for comparison
    actual = pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(flux_outs.time.values[0]))
    expected = inv_out.period_midpoint

    assert actual == expected


def test_paris_flux_output_uses_january_annual_period_for_june_run(inv_out, europe_country_file):
    """Public PARIS output clips a retained January annual period to a June run."""
    inv_out.run_metadata["start_date"] = "2019-06-01"
    inv_out.run_metadata["end_date"] = "2019-07-01"

    assert infer_flux_frequency(inv_out.flux) == "yearly"
    flux_outs = paris_flux_output(
        inv_out,
        country_file=europe_country_file,
        flux_frequency=infer_flux_frequency(inv_out.flux),
    )

    assert flux_outs.sizes["time"] == 1
    actual = pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(flux_outs.time.values[0]))
    assert actual == pd.Timestamp("2019-06-16")


def test_latest_paris_flux_output_reports_clipped_annual_midpoint_and_bounds(inv_out, europe_country_file):
    """Latest PARIS flux reports the exact midpoint and bounds of a clipped annual prior."""
    inv_out.run_metadata["start_date"] = "2019-06-01"
    inv_out.run_metadata["end_date"] = "2019-07-01"

    flux_outs = paris_flux_output(
        inv_out,
        country_file=europe_country_file,
        inversion_grid=False,
        flux_frequency=infer_flux_frequency(inv_out.flux),
        template_version="latest",
    )

    epoch = pd.Timestamp("1970-01-01")
    actual = epoch + pd.Timedelta(days=float(flux_outs.time.values[0]))
    bounds = epoch + pd.to_timedelta(flux_outs.time_bnds.values[0], unit="D")

    assert flux_outs.sizes["time"] == 1
    assert actual == pd.Timestamp("2019-06-16")
    assert list(bounds) == [pd.Timestamp("2019-06-01"), pd.Timestamp("2019-07-01")]


def test_legacy_paris_concentration_shifts_hourly_observations_to_midpoints(inv_out):
    """Legacy PARIS concentration shifts hourly observation starts by 30 minutes."""
    observation_starts = pd.DatetimeIndex(observation_inputs_for_outputs(inv_out).time.values).unique()
    expected = (observation_starts + pd.Timedelta(minutes=30) - pd.Timestamp("1970-01-01")) / pd.Timedelta(
        days=1
    )

    result = paris_concentration_outputs(inv_out, obs_avg_period="1h")

    assert result.time.dims == ("time",)
    np.testing.assert_array_equal(result.time.values, expected.to_numpy())


def test_latest_paris_concentration_reports_hourly_midpoints_and_bounds(inv_out):
    """Latest PARIS concentration reports hourly midpoints and exact start/end bounds."""
    epoch = pd.Timestamp("1970-01-01")
    starts = pd.DatetimeIndex(observation_inputs_for_outputs(inv_out).time.values)
    expected_starts = (starts - epoch) / pd.Timedelta(days=1)
    expected_ends = (starts + pd.Timedelta(hours=1) - epoch) / pd.Timedelta(days=1)
    expected_midpoints = (starts + pd.Timedelta(minutes=30) - epoch) / pd.Timedelta(days=1)

    result = paris_concentration_outputs(inv_out, obs_avg_period="1h", template_version="latest")

    assert result.time.dims == ("index",)
    assert result.time_bnds.dims == ("index", "nbnds")
    np.testing.assert_array_equal(result.time.values, expected_midpoints.to_numpy())
    np.testing.assert_array_equal(
        result.time_bnds.values,
        np.column_stack([expected_starts.to_numpy(), expected_ends.to_numpy()]),
    )


def test_basic_outputs(inv_out, europe_country_file):
    """Test creation of basic output for EUROPE domain.

    The default stats calculated are "mean" and "quantile".
    Check that these are all present.
    """
    outs = basic_output(inv_out, country_file=europe_country_file)

    conc_vars = ["y_posterior_predictive", "y_prior_predictive"]
    for x in ["flux", "scaling", "country", "mu_bc"]:
        for y in ["prior", "posterior"]:
            conc_vars.append(x + "_" + y)

    stats = ["mean", "quantile"]

    for cv in conc_vars:
        for stat in stats:
            assert cv + "_" + stat in outs


def test_fixedbasis_flux_and_country_outputs_use_modern_basis_functions(inv_out, europe_country_file):
    """Fixedbasis postprocessing reconstructs products from retained basis functions."""
    flux_outs = make_flux_outputs(
        inv_out,
        include_scale_factors=False,
        report_flux_on_inversion_grid=False,
    )
    country_outs = make_country_outputs(inv_out, country_file=europe_country_file, country_regions="paris")

    assert "flux_posterior_mean" in flux_outs
    assert "country_posterior_mean" in country_outs


@pytest.mark.parametrize("offset", [False, True])
def test_make_paris_outputs(inv_out, europe_country_file, tmpdir, offset):
    """Check that we can create and save PARIS outputs for EUROPE domain"""

    if offset:
        # fake an offset trace
        inv_out.trace.posterior["offset"] = xr.ones_like(inv_out.trace.posterior["mu_bc"])
        inv_out.trace.prior["offset"] = xr.ones_like(inv_out.trace.prior["mu_bc"])

    print(inv_out.trace.posterior)

    flux_outs, conc_outs = make_paris_outputs(
        inv_out, country_file=europe_country_file, obs_avg_period="1h", domain="europe"
    )

    if offset:
        assert "Yapriori_bias" in conc_outs

    # check we can write to netCDF
    flux_outs.to_netcdf(tmpdir / "flux.nc")
    conc_outs.to_netcdf(tmpdir / "conc.nc")


def test_paris_template_registry_requires_explicit_latest():
    """PARIS output keeps the legacy templates by default for the next release."""
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
