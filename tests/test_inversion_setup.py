import pytest
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.hbmcmc.inversionsetup import sigma_freq_indicies

# -------------------------
# Fixtures: mocked fp_data
# -------------------------


def _make_time_index(start_date: str, end_date: str, *, twoperday=True, start_offset_hours=0):
    """
    Create time coordinate: twice per day between start_date and end_date (end exclusive),
    with optional offset so first timestamp is not exactly on start_date.
    """
    start = pd.Timestamp(start_date) + pd.Timedelta(hours=start_offset_hours)
    end = pd.Timestamp(end_date)
    freq = "12h" if twoperday else "1D"
    # end exclusive is handy: old code uses < end_date in places
    return pd.date_range(start, end, freq=freq, inclusive="left")


def _make_fp_data(site: str, *, bc_regions, time_index: pd.DatetimeIndex):
    """
    fp_data mock: {"MHD": xr.Dataset(...)} with only H_bc needed by old functions.
    Old inversionsetup functions read H_bc as fp_data[site].H_bc.values[cord, dateloc],
    i.e. H_bc dims must be (bc_region, time).
    """
    nreg = len(bc_regions)
    nt = len(time_index)

    # deterministic values: region-coded, time-coded
    # shape (bc_region, time)
    vals = (np.arange(nreg)[:, None] + 1.0) * (np.arange(nt)[None, :] + 1.0)

    ds = xr.Dataset(
        data_vars={
            "H_bc": (("bc_region", "time"), vals),
        },
        coords={
            "bc_region": list(bc_regions),
            "time": time_index,
        },
    )
    return {site: ds}


# ------------------------------------------
# Reference: "expand H_bc by a period index"
# ------------------------------------------


def _expand_Hbc_by_period(H_bc: xr.DataArray, period_index: np.ndarray, nperiod: int) -> np.ndarray:
    """
    Build reference sensitivity matrix like the *intended* meaning:
    for each region r and period p:
      row = r*nperiod + p
      columns = time
      equals H_bc[r, t] if period_index[t] == p else 0

    Returns np.ndarray shape (nregions*nperiod, ntime)
    """
    # Ensure dims (bc_region, time)
    H = H_bc.transpose("bc_region", "time").values
    nreg, nt = H.shape
    out = np.zeros((nreg * nperiod, nt), dtype=float)

    for r in range(nreg):
        for p in range(nperiod):
            mask = period_index == p
            out[r * nperiod + p, mask] = H[r, mask]
    return out


def _monthly_period_index(times: pd.DatetimeIndex, start_date: str, end_date: str):
    """
    Match inversionsetup.monthly_bcs binning:
      allmonth = date_range(start_date, end_date, freq="MS")[:-1]
      each time is assigned period m if allmonth[m] <= t < allmonth[m+1]
    """
    allmonth = pd.date_range(start_date, end_date, freq="MS")[:-1]

    # Need the right-edge for the last month bin
    allmonth_edges = pd.date_range(start_date, end_date, freq="MS")  # includes last edge
    # Example: start Jan 1, end Mar 10 => edges [Jan1, Feb1, Mar1]
    # bins: Jan, Feb  (since allmonth is edges[:-1])

    tvals = times.values.astype("datetime64[ns]")
    period_index = np.full(len(times), fill_value=-1, dtype=int)

    for m in range(len(allmonth_edges) - 1):
        left = allmonth_edges[m].to_datetime64()
        right = allmonth_edges[m + 1].to_datetime64()
        mask = np.logical_and(tvals >= left, tvals < right)
        period_index[mask] = m

    # months present in data (equivalent to resample("MS").mean().time)
    # Do this without PeriodIndex "MS" conversion
    month_starts = pd.DatetimeIndex(pd.to_datetime(times.strftime("%Y-%m-01")))
    pmonth = pd.DatetimeIndex(np.unique(month_starts.values))

    return period_index, allmonth, pmonth


def test_sigma_freq_indicies():
    ytime = pd.date_range("2020-06-01", "2021-06-01", freq="4h")
    sigma_freq = "monthly"

    sigma_freq_index = sigma_freq_indicies(ytime, sigma_freq)
    nsigma_time = np.unique(sigma_freq_index)
    nsigma_site = [0]
    sigma = np.arange(len(nsigma_time)).reshape(1, -1)

    try:
        sigma[nsigma_site, sigma_freq_index]
    except IndexError:
        pytest.fail("Indexing sigma with nsigma_site and sigma_freq_index failed.")
