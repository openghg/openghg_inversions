import pytest
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.hbmcmc.inversionsetup import sigma_freq_indicies
from openghg_inversions.hbmcmc.inversionsetup import monthly_bcs, create_bc_sensitivity

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
    nmonth = len(allmonth)

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


# def _monthly_period_index(times: pd.DatetimeIndex, start_date: str, end_date: str):
#     """
#     Match inversionsetup.monthly_bcs logic:
#       allmonth = date_range(start_date, end_date, freq="MS")[:-1]
#       months included only if present in fp_data[site].resample(time="MS").mean().time.values
#     We emulate that by:
#       - defining allmonth
#       - defining pmonth = months present in data (month starts)
#       - assigning each time to the index of its month within allmonth, skipping months not in pmonth.
#     Returns:
#       period_index over time (int labels into 0..nmonth-1 but some may be unused)
#       allmonth (DatetimeIndex)
#       pmonth (DatetimeIndex)
#     """
#     allmonth = pd.date_range(start_date, end_date, freq="MS")[:-1]

#     # months present in data
#     # month starts for each timestamp:
#     # month_starts = times.to_period("M").to_timestamp("MS")
#     month_starts_int = times.month
#     start_year = pd.Timestamp(year=pd.to_datetime(start_date).year, month=1, day=1)
#     month_starts = pd.to_datetime([start_year + pd.DateOffset(month=i - 1) for i in month_starts_int])
#     pmonth = pd.DatetimeIndex(np.unique(month_starts.values))

#     # Map month start -> position in allmonth
#     month_to_pos = {m: i for i, m in enumerate(allmonth)}

#     # Period index is position in allmonth; months not in allmonth become -1 (shouldn't happen often)
#     period_index = np.array([month_to_pos.get(ms, -1) for ms in month_starts], dtype=int)
#     return period_index, allmonth, pmonth


@pytest.mark.parametrize(
    "bc_regions",
    [
        ["n", "e", "s", "w"],
        ["a", "b", "c", "d", "e", "f", "g"],
    ],
)
@pytest.mark.parametrize(
    "case",
    [
        # starts exactly on start_date, spans Jan->Mar (2-3 months depending on end)
        dict(start_date="2019-01-01", end_date="2019-03-10", start_offset_hours=0),
        # starts not on start_date (offset), still spans multiple months
        dict(start_date="2019-01-01", end_date="2019-03-10", start_offset_hours=6),
        # extreme: time starts in February while start_date is January
        dict(start_date="2019-01-01", end_date="2019-03-10", start_offset_hours=24 * 31 + 6),
    ],
)
def test_monthly_bcs_matches_reference(bc_regions, case):
    site = "MHD"
    times = _make_time_index(
        case["start_date"], case["end_date"], start_offset_hours=case["start_offset_hours"]
    )
    fp_data = _make_fp_data(site, bc_regions=bc_regions, time_index=times)
    H_bc = fp_data[site]["H_bc"]

    # Old implementation
    hmbc = monthly_bcs(case["start_date"], case["end_date"], site, fp_data)

    # Reference expansion: only months present in resample("MS") should contribute.
    period_index, allmonth, pmonth = _monthly_period_index(times, case["start_date"], case["end_date"])
    nmonth = len(allmonth)

    # monthly_bcs *skips* months not in pmonth by leaving corresponding rows all-zero.
    ref = _expand_Hbc_by_period(H_bc, period_index=period_index, nperiod=nmonth)

    assert hmbc.shape == ref.shape
    np.testing.assert_allclose(hmbc, ref, rtol=0, atol=0)

    # Sanity: if data starts in Feb, then all Jan rows should be all-zero
    if times.min() >= pd.Timestamp("2019-02-01"):
        # first month in allmonth is Jan 2019 => period 0; check all regions' Jan rows are zero
        for r in range(len(bc_regions)):
            assert np.allclose(hmbc[r * nmonth + 0, :], 0.0)


@pytest.mark.parametrize(
    "bc_regions",
    [
        ["n", "e", "s", "w"],
        ["a", "b", "c", "d", "e", "f", "g"],
    ],
)
@pytest.mark.parametrize(
    "freq",
    [
        "8D",
        "12H",
    ],
)
@pytest.mark.parametrize(
    "start_offset_hours",
    [
        0,
        6,
    ],
)
def test_create_bc_sensitivity_matches_reference(bc_regions, freq, start_offset_hours):
    site = "MHD"
    start_date = "2019-01-01"
    end_date = "2019-03-10"

    # For 12H freq, we want times more frequent than daily; use 6-hourly for richer coverage.
    time_freq = "6h" if freq.upper().endswith("H") else "12h"
    times = pd.date_range(
        pd.Timestamp(start_date) + pd.Timedelta(hours=start_offset_hours),
        pd.Timestamp(end_date),
        freq=time_freq,
        inclusive="left",
    )

    fp_data = _make_fp_data(site, bc_regions=bc_regions, time_index=times)
    H_bc = fp_data[site]["H_bc"]

    # Old implementation
    hmbc = create_bc_sensitivity(start_date, end_date, site, fp_data, freq=freq)

    # Reference: build period bins using same date_range logic as create_bc_sensitivity
    # It creates `alldates= with end_date + dys, then ndates = sum(alldates < end_date)
    dys = int("".join([s for s in freq if s.isdigit()]))
    alldates = pd.date_range(
        pd.to_datetime(start_date),
        pd.to_datetime(end_date) + pd.DateOffset(days=dys),
        freq=freq,
    )
    ndates = np.sum(alldates < pd.to_datetime(end_date))

    # period_index: each time falls into bin m where alldates[m] <= t < alldates[m+1]
    tvals = times.values.astype("datetime64[ns]")
    period_index = np.full(len(times), fill_value=-1, dtype=int)
    for m in range(ndates):
        mask = np.logical_and(tvals >= alldates[m].to_datetime64(), tvals < alldates[m + 1].to_datetime64())
        period_index[mask] = m

    ref = _expand_Hbc_by_period(H_bc, period_index=period_index, nperiod=ndates)

    assert hmbc.shape == ref.shape
    np.testing.assert_allclose(hmbc, ref, rtol=0, atol=0)


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
