"""Synthetic regression scaffolding for monthly sigma/bc with missing month."""

import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.hbmcmc.hbmcmc import make_inv_inputs_legacy
from openghg_inversions.hbmcmc.inversion_pymc import (
    _map_times_to_available_month_positions,
    _weighted_apriori_flux_for_months,
    inferpymc,
)


def _synthetic_fp_data_one_site_with_missing_month() -> dict[str, xr.Dataset]:
    """Create one-site daily data over 3 months with the middle month missing."""
    all_days = pd.date_range("2019-01-01", "2019-04-01", freq="1D", inclusive="left")
    times = all_days[all_days.month != 2]  # keep Jan + Mar, remove Feb entirely

    ntime = len(times)
    nregion = 3
    bc_regions = ["n", "e", "s", "w"]

    h = np.vstack(
        [
            np.linspace(0.2, 0.8, ntime),
            np.linspace(0.6, 0.1, ntime),
            np.linspace(0.1, 0.4, ntime),
        ]
    ).astype(np.float32)
    h_bc = (np.arange(1, len(bc_regions) + 1)[:, None] * np.linspace(0.01, 0.1, ntime)[None, :]).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "H": (("region", "time"), h),
            "H_bc": (("bc_region", "time"), h_bc),
            "mf": (("time",), np.linspace(1800.0, 1810.0, ntime).astype(np.float32)),
            "mf_error": (("time",), np.full(ntime, 0.2, dtype=np.float32)),
            "mf_repeatability": (("time",), np.full(ntime, 0.05, dtype=np.float32)),
            "mf_variability": (("time",), np.full(ntime, 0.05, dtype=np.float32)),
        },
        coords={
            "time": times,
            "region": np.arange(nregion),
            "bc_region": bc_regions,
        },
    )

    return {"AAA": ds}


def test_make_inv_inputs_month_gap_monthly_indices_are_non_contiguous():
    fp_data = _synthetic_fp_data_one_site_with_missing_month()

    mcmc_args, _ = make_inv_inputs_legacy(
        fp_data=fp_data,
        sites=["AAA"],
        start_date="2019-01-01",
        use_bc=True,
        bc_freq="monthly",
        sigma_freq="monthly",
        min_error=0.0,
        calculate_min_error=None,
        min_error_options={},
    )

    uniq = np.unique(mcmc_args["sigma_freq_index"])
    np.testing.assert_array_equal(uniq, np.array([0, 2]))


def test_inferpymc_smoke_runs_for_month_gap():
    fp_data = _synthetic_fp_data_one_site_with_missing_month()

    mcmc_args, _ = make_inv_inputs_legacy(
        fp_data=fp_data,
        sites=["AAA"],
        start_date="2019-01-01",
        use_bc=True,
        bc_freq="monthly",
        sigma_freq="monthly",
        min_error=0.0,
        calculate_min_error=None,
        min_error_options={},
    )

    result = inferpymc(
        **mcmc_args,
        xprior={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
        bcprior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
        sigprior={"pdf": "uniform", "lower": 0.1, "upper": 0.4},
        nuts_sampler="pymc",
        nit=1,
        burn=0,
        tune=0,
        nchain=1,
        sigma_per_site=True,
        verbose=False,
        use_bc=True,
        sampler_kwargs={"compute_convergence_checks": False},
    )

    assert "sigouts" in result
    assert "trace" in result


def test_weighted_apriori_flux_handles_missing_month():
    flux_array_all = np.array([[[1.0, 3.0]]], dtype=np.float32)
    month_index = np.array([0, 0, 2, 2], dtype=int)

    apriori_flux = _weighted_apriori_flux_for_months(flux_array_all, month_index)

    np.testing.assert_allclose(apriori_flux, np.array([[2.0]], dtype=np.float32))


def test_map_times_to_available_month_positions_handles_gappy_flux_months():
    times = pd.to_datetime(["2019-01-15", "2019-01-20", "2019-03-10", "2019-04-20"])
    flux_times = pd.to_datetime(["2019-01-01", "2019-03-01", "2019-04-01"])

    positions = _map_times_to_available_month_positions(times, flux_times)

    np.testing.assert_array_equal(positions, np.array([0, 0, 1, 2]))


def test_map_times_to_available_month_positions_handles_multi_year_flux_time():
    times = pd.to_datetime(["2023-03-15", "2023-11-20", "2024-07-10", "2025-04-20"])
    flux_times = pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01"])

    positions = _map_times_to_available_month_positions(times, flux_times)

    np.testing.assert_array_equal(positions, np.array([0, 0, 1, 2]))
