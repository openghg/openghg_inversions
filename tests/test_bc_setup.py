import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.hbmcmc.inversionsetup import monthly_bcs, create_bc_sensitivity
from openghg_inversions.inversion_inputs import _transform_bc_freq

# -------------------------
# Shared fixtures/helpers
# -------------------------


def _make_time_index(start_date: str, end_date: str, *, freq="12h", start_offset_hours=0):
    start = pd.Timestamp(start_date) + pd.Timedelta(hours=start_offset_hours)
    end = pd.Timestamp(end_date)
    return pd.date_range(start, end, freq=freq, inclusive="left")


def _make_H_bc(*, bc_regions, times: pd.DatetimeIndex) -> xr.DataArray:
    """
    H_bc(time, bc_region) with deterministic values.
    """
    nreg = len(bc_regions)
    nt = len(times)
    vals = (np.arange(nreg)[:, None] + 1.0) * (np.arange(nt)[None, :] + 1.0)
    return xr.DataArray(
        vals.T,
        dims=("time", "bc_region"),
        coords={"time": times, "bc_region": list(bc_regions)},
        name="H_bc",
    )


def _make_fp_data(site: str, H_bc: xr.DataArray) -> dict:
    """
    Old inversionsetup functions expect fp_data[site]["H_bc"] to have dims (bc_region, time).
    """
    ds = xr.Dataset({"H_bc": H_bc.transpose("bc_region", "time")})
    return {site: ds}


def _period_index_from_edges(times: pd.DatetimeIndex, edges: pd.DatetimeIndex) -> np.ndarray:
    """
    Assign each time to a bin m where edges[m] <= t < edges[m+1].
    Returns period_index with values in [0, nbin-1], or -1 if not in any bin.
    """
    tvals = times.values.astype("datetime64[ns]")
    period_index = np.full(len(times), fill_value=-1, dtype=int)
    for m in range(len(edges) - 1):
        left = edges[m].to_datetime64()
        right = edges[m + 1].to_datetime64()
        mask = np.logical_and(tvals >= left, tvals < right)
        period_index[mask] = m
    return period_index


def _expand_Hbc_by_period(H_bc: xr.DataArray, *, period_index: np.ndarray, nperiod: int) -> np.ndarray:
    """
    Expand H_bc into (bc_region*nperiod, time) as a sensitivity matrix.
    """
    H = H_bc.transpose("bc_region", "time").values
    nreg, nt = H.shape
    out = np.zeros((nreg * nperiod, nt), dtype=float)
    for r in range(nreg):
        for p in range(nperiod):
            mask = period_index == p
            out[r * nperiod + p, mask] = H[r, mask]
    return out


def _monthly_edges(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Month-start edges. Matches the binning logic in monthly_bcs, but includes the right edge.
    """
    return pd.date_range(start_date, end_date, freq="MS")


def _freq_edges(start_date: str, end_date: str, freq: str) -> pd.DatetimeIndex:
    """
    Edges for create_bc_sensitivity: it uses date_range(start, end+offset, freq=freq)
    and then ndates = sum(alldates < end). We return the same edges array (including extra edge).
    """
    # mimic the old dys logic (days offset only)
    digits = "".join([s for s in freq if s.isdigit()])
    dys = int(digits) if digits else 0

    return pd.date_range(
        pd.to_datetime(start_date),
        pd.to_datetime(end_date) + pd.DateOffset(days=dys),
        freq=freq,
    )


# -------------------------
# Check for "dead" parameters
# -------------------------


def find_all_zero_rows(Hmbc: np.ndarray, *, atol=0.0) -> np.ndarray:
    # rows where max abs == 0
    return np.where(np.nanmax(np.abs(Hmbc), axis=1) <= atol)[0]


def find_all_zero_cols(Hmbc: np.ndarray, *, atol=0.0) -> np.ndarray:
    return np.where(np.nanmax(np.abs(Hmbc), axis=0) <= atol)[0]


def report_dead_bc_periods(H_bc_expanded: xr.DataArray, *, atol=0.0) -> xr.Dataset:
    """
    H_bc_expanded dims: bc_region, time
    where bc_region is a stack of (bc_curtain, bc_period).
    Returns a Dataset listing dead rows and dead periods.
    """
    # Ensure we have a MultiIndex on bc_region
    if not isinstance(H_bc_expanded.indexes.get("bc_region", None), pd.MultiIndex):
        try:
            H_bc_expanded = H_bc_expanded.unstack("bc_region").stack(bc_region=("bc_curtain", "bc_period"))
        except Exception:
            pass

    H2 = H_bc_expanded.unstack("bc_region")  # dims: bc_curtain, bc_period, time

    row_max = np.abs(H2).max("time", skipna=True)
    dead = row_max <= atol  # dims: bc_curtain, bc_period

    # dead periods across all curtains
    dead_period = dead.all("bc_curtain")

    return xr.Dataset(
        {
            "dead_row": dead,
            "dead_period": dead_period,
        }
    )


# -------------------------
# Parametrisation
# -------------------------

BC_REGIONS_CASES = [
    ["n", "e", "s", "w"],
    ["a", "b", "c", "d", "e", "f", "g"],
]

MONTH_CASES = [
    dict(start_date="2019-01-01", end_date="2019-03-10", start_offset_hours=0),
    dict(start_date="2019-01-01", end_date="2019-03-10", start_offset_hours=6),
    dict(start_date="2019-01-01", end_date="2019-03-10", start_offset_hours=24 * 31 + 6),
]


# -------------------------
# Tests: OLD functions vs reference
# -------------------------


@pytest.mark.parametrize("bc_regions", BC_REGIONS_CASES)
@pytest.mark.parametrize("case", MONTH_CASES)
def test_monthly_bcs_matches_reference(bc_regions, case):
    site = "MHD"
    times = _make_time_index(
        case["start_date"], case["end_date"], freq="12h", start_offset_hours=case["start_offset_hours"]
    )
    H_bc = _make_H_bc(bc_regions=bc_regions, times=times)
    fp_data = _make_fp_data(site, H_bc)

    hmbc = monthly_bcs(case["start_date"], case["end_date"], site, fp_data)

    edges = _monthly_edges(case["start_date"], case["end_date"])
    # monthly_bcs uses allmonth = edges[:-1], i.e. nperiod = len(edges)-1
    nperiod = len(edges) - 1
    period_index = _period_index_from_edges(times, edges)
    ref = _expand_Hbc_by_period(H_bc, period_index=period_index, nperiod=nperiod)

    assert hmbc.shape == ref.shape
    np.testing.assert_allclose(hmbc, ref, rtol=0, atol=0)

    zero_rows = find_all_zero_rows(hmbc)
    assert len(zero_rows) == 0

    zero_cols = find_all_zero_cols(hmbc)
    assert len(zero_cols) == 0


@pytest.mark.parametrize("bc_regions", BC_REGIONS_CASES)
@pytest.mark.parametrize("freq", ["8D", "12H"])
@pytest.mark.parametrize("start_offset_hours", [0, 6])
def test_create_bc_sensitivity_matches_reference(bc_regions, freq, start_offset_hours):
    site = "MHD"
    start_date = "2019-01-01"
    end_date = "2019-03-10"

    time_freq = "6h" if freq.upper().endswith("H") else "12h"
    times = _make_time_index(start_date, end_date, freq=time_freq, start_offset_hours=start_offset_hours)
    H_bc = _make_H_bc(bc_regions=bc_regions, times=times)
    fp_data = _make_fp_data(site, H_bc)

    hmbc = create_bc_sensitivity(start_date, end_date, site, fp_data, freq=freq)

    edges = _freq_edges(start_date, end_date, freq=freq)
    # old code uses ndates = sum(edges < end_date)  (number of bins)
    ndates = int(np.sum(edges < pd.to_datetime(end_date)))
    period_index = _period_index_from_edges(times, edges[: ndates + 1])
    ref = _expand_Hbc_by_period(H_bc, period_index=period_index, nperiod=ndates)

    assert hmbc.shape == ref.shape
    np.testing.assert_allclose(hmbc, ref, rtol=0, atol=0)

    zero_rows = find_all_zero_rows(hmbc)
    assert len(zero_rows) == 0

    zero_cols = find_all_zero_cols(hmbc)
    assert len(zero_cols) == 0


# -------------------------
# Tests: NEW _transform_bc_freq vs same reference
# -------------------------


@pytest.mark.parametrize("bc_regions", BC_REGIONS_CASES)
@pytest.mark.parametrize("case", MONTH_CASES)
def test_transform_bc_freq_monthly_matches_reference(bc_regions, case):
    times = _make_time_index(
        case["start_date"], case["end_date"], freq="12h", start_offset_hours=case["start_offset_hours"]
    )
    H_bc = _make_H_bc(bc_regions=bc_regions, times=times)

    out = _transform_bc_freq(H_bc, freq="monthly")
    out_np = out.transpose("bc_region", "time").values

    edges = _monthly_edges(case["start_date"], case["end_date"])
    nperiod = len(edges) - 1
    period_index = _period_index_from_edges(times, edges)
    ref = _expand_Hbc_by_period(H_bc, period_index=period_index, nperiod=nperiod)

    assert out_np.shape == ref.shape
    np.testing.assert_allclose(out_np, ref, rtol=0, atol=0)

    rep = report_dead_bc_periods(out, atol=0.0)

    # list dead (curtain, period) pairs
    dead_pairs = np.argwhere(rep["dead_row"].values)
    assert not bool(
        rep["dead_row"].any()
    ), f"Dead BC rows (bc_curtain, bc_period) count={dead_pairs.shape[0]}"

    dead_periods = rep["dead_period"].bc_period.values[rep["dead_period"].values]
    assert not bool(
        rep["dead_period"].any()
    ), f"Dead BC periods (no times assigned in ANY curtain): {dead_periods}"


@pytest.mark.parametrize("bc_regions", BC_REGIONS_CASES)
@pytest.mark.parametrize("freq", ["8D", "12H"])
@pytest.mark.parametrize("start_offset_hours", [0, 6])
def test_transform_bc_freq_pandas_freq_matches_reference(
    bc_regions,
    freq,
    start_offset_hours,
):
    start_date = "2019-01-01"
    end_date = "2019-03-10"

    time_freq = "6h" if freq.upper().endswith("H") else "12h"
    times = _make_time_index(start_date, end_date, freq=time_freq, start_offset_hours=start_offset_hours)
    H_bc = _make_H_bc(bc_regions=bc_regions, times=times)

    out = _transform_bc_freq(H_bc, freq=freq, anchor_time=start_date)
    out_np = out.transpose("bc_region", "time").values

    edges = _freq_edges(start_date, end_date, freq=freq)
    ndates = int(np.sum(edges < pd.to_datetime(end_date)))
    period_index = _period_index_from_edges(times, edges[: ndates + 1])
    ref = _expand_Hbc_by_period(H_bc, period_index=period_index, nperiod=ndates)

    # debugging check
    # Compute "which period" each time is assigned to for both methods.

    # 1) period_index_old from edges
    p_old = period_index

    # 2) period_index_new from _transform_bc_freq:
    # out has stacked bc_region=(bc_curtain, bc_period). Unstack to get (bc_curtain, bc_period, time)
    out_u = out.unstack("bc_region").transpose(..., "time")  # dims: bc_curtain, bc_period, time

    # pick one curtain (all same assignment), and find bc_period with non-zero at each time
    # (since exactly one period should be active per time)
    act = (np.abs(out_u.isel(bc_curtain=0)) > 0).values  # shape (bc_period, time)
    p_new = act.argmax(axis=0)  # integer period per time

    # Compare
    diff = np.where(p_old != p_new)[0]
    print("n_diff_times:", diff.size)
    print("first_diffs:", diff[:20])
    if diff.size:
        j = diff[0]
        print("time_at_first_diff:", times[j], "p_old:", p_old[j], "p_new:", p_new[j])
        # show local neighborhood
        sl = slice(max(0, j - 5), min(len(times), j + 6))
        print("times window:", times[sl])
        print("p_old window:", p_old[sl])
        print("p_new window:", p_new[sl])

    assert out_np.shape == ref.shape
    np.testing.assert_allclose(out_np, ref, rtol=0, atol=0)

    rep = report_dead_bc_periods(out, atol=0.0)

    # list dead (curtain, period) pairs
    dead_pairs = np.argwhere(rep["dead_row"].values)
    assert not bool(
        rep["dead_row"].any()
    ), f"Dead BC rows (bc_curtain, bc_period) count={dead_pairs.shape[0]}"

    dead_periods = rep["dead_period"].bc_period.values[rep["dead_period"].values]
    assert not bool(
        rep["dead_period"].any()
    ), f"Dead BC periods (no times assigned in ANY curtain): {dead_periods}"
