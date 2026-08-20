import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.hbmcmc.inversionsetup import monthly_bcs, create_bc_sensitivity
from openghg_inversions.boundary_sensitivity import BoundaryAlignment

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
    Old inversionsetup functions expect fp_data[site]["H_bc"] dims (bc_region, time).
    """
    ds = xr.Dataset({"H_bc": H_bc.transpose("bc_region", "time")})
    return {site: ds}


def _period_index_from_edges(times: pd.DatetimeIndex, edges: pd.DatetimeIndex) -> np.ndarray:
    """
    Assign each time to a bin m where edges[m] <= t < edges[m+1].
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
    Month-start edges for correct monthly binning.
    """
    return pd.date_range(start_date, end_date, freq="MS")


def _monthly_period_index_like_transform(times: pd.DatetimeIndex) -> tuple[np.ndarray, int]:
    """
    Period index matching inversion_inputs.make_freq_indicator(..., freq="monthly"):
      period = month - month(time.min()) + 12*(year - year(time.min()))
    Returns:
      period_index: array[int] for each time
      nperiod: number of periods (= max(period_index)+1)
    """
    if len(times) == 0:
        return np.array([], dtype=int), 0
    t0 = times.min()
    period_index = (times.month - t0.month) + 12 * (times.year - t0.year)
    period_index = np.asarray(period_index, dtype=int)
    nperiod = int(period_index.max() + 1)
    return period_index, nperiod


def _freq_edges_old_create_bc(start_date: str, end_date: str, freq: str) -> pd.DatetimeIndex:
    """
    Edges matching create_bc_sensitivity old logic.
    """
    freq = freq.replace("H", "h")
    digits = "".join([s for s in freq if s.isdigit()])
    dys = int(digits) if digits else 0
    return pd.date_range(
        pd.to_datetime(start_date),
        pd.to_datetime(end_date) + pd.DateOffset(days=dys),
        freq=freq,
    )


# -------------------------
# Dead parameter/time checks
# -------------------------


def find_all_zero_rows(Hmbc: np.ndarray, *, atol=0.0) -> np.ndarray:
    return np.where(np.nanmax(np.abs(Hmbc), axis=1) <= atol)[0]


def find_all_zero_cols(Hmbc: np.ndarray, *, atol=0.0) -> np.ndarray:
    return np.where(np.nanmax(np.abs(Hmbc), axis=0) <= atol)[0]


def report_dead_bc_periods(H_bc_expanded: xr.DataArray, *, atol=0.0) -> xr.Dataset:
    """
    H_bc_expanded dims: bc_region, time
    where bc_region is a stack of (bc_curtain, bc_period).
    """
    if not isinstance(H_bc_expanded.indexes.get("bc_region", None), pd.MultiIndex):
        # make sure it's a MultiIndex if possible
        try:
            H_bc_expanded = H_bc_expanded.unstack("bc_region").stack(bc_region=("bc_curtain", "bc_period"))
        except Exception:
            pass

    H2 = H_bc_expanded.unstack("bc_region")  # dims: bc_curtain, bc_period, time
    row_max = np.abs(H2).max("time", skipna=True)
    dead = row_max <= atol
    dead_period = dead.all("bc_curtain")

    return xr.Dataset({"dead_row": dead, "dead_period": dead_period})


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


# ============================================================================
# OLD FUNCTIONS: regression tests (they do what they do)
# ============================================================================


@pytest.mark.parametrize("bc_regions", BC_REGIONS_CASES)
@pytest.mark.parametrize("case", MONTH_CASES)
def test_monthly_bcs_matches_reference(bc_regions, case):
    """
    Regression for old monthly_bcs: matches edge-binning reference.
    """
    site = "MHD"
    times = _make_time_index(
        case["start_date"], case["end_date"], freq="12h", start_offset_hours=case["start_offset_hours"]
    )
    H_bc = _make_H_bc(bc_regions=bc_regions, times=times)
    fp_data = _make_fp_data(site, H_bc)

    hmbc = monthly_bcs(case["start_date"], case["end_date"], site, fp_data)

    edges = _monthly_edges(case["start_date"], case["end_date"])
    nperiod = len(edges) - 1
    period_index = _period_index_from_edges(times, edges)
    ref = _expand_Hbc_by_period(H_bc, period_index=period_index, nperiod=nperiod)

    assert hmbc.shape == ref.shape
    np.testing.assert_allclose(hmbc, ref, rtol=0, atol=0)


@pytest.mark.parametrize("bc_regions", BC_REGIONS_CASES)
@pytest.mark.parametrize("case", MONTH_CASES)
@pytest.mark.xfail(
    reason="Old monthly_bcs can produce dead BC params/periods in corner cases; keep as evidence.",
    strict=False,
)
def test_monthly_bcs_has_no_dead_rows_or_cols(bc_regions, case):
    """
    Evidence test: old monthly_bcs SHOULD ideally have no dead rows/cols,
    but currently can fail. Marked xfail.
    """
    site = "MHD"
    times = _make_time_index(
        case["start_date"], case["end_date"], freq="12h", start_offset_hours=case["start_offset_hours"]
    )
    H_bc = _make_H_bc(bc_regions=bc_regions, times=times)
    fp_data = _make_fp_data(site, H_bc)

    hmbc = monthly_bcs(case["start_date"], case["end_date"], site, fp_data)

    assert len(find_all_zero_rows(hmbc)) == 0
    assert len(find_all_zero_cols(hmbc)) == 0


@pytest.mark.parametrize("bc_regions", BC_REGIONS_CASES)
@pytest.mark.parametrize("freq", ["8D", "12H"])
@pytest.mark.parametrize("start_offset_hours", [0, 6])
def test_create_bc_sensitivity_matches_reference(bc_regions, freq, start_offset_hours):
    """
    Regression for old create_bc_sensitivity: matches old edge logic reference.
    """
    site = "MHD"
    start_date = "2019-01-01"
    end_date = "2019-03-10"

    time_freq = "6h" if freq.upper().endswith("H") else "12h"
    times = _make_time_index(start_date, end_date, freq=time_freq, start_offset_hours=start_offset_hours)
    H_bc = _make_H_bc(bc_regions=bc_regions, times=times)
    fp_data = _make_fp_data(site, H_bc)

    hmbc = create_bc_sensitivity(start_date, end_date, site, fp_data, freq=freq)

    edges = _freq_edges_old_create_bc(start_date, end_date, freq=freq)
    ndates = int(np.sum(edges < pd.to_datetime(end_date)))
    period_index = _period_index_from_edges(times, edges[: ndates + 1])
    ref = _expand_Hbc_by_period(H_bc, period_index=period_index, nperiod=ndates)

    assert hmbc.shape == ref.shape
    np.testing.assert_allclose(hmbc, ref, rtol=0, atol=0)


# ============================================================================
# NEW FUNCTION: treated as the "correct" behaviour (no dead params)
# ============================================================================


@pytest.mark.parametrize("bc_regions", BC_REGIONS_CASES)
@pytest.mark.parametrize("case", MONTH_CASES)
def test_prepare_boundary_sensitivity_monthly_matches_reference_and_no_dead(bc_regions, case):
    """
    New monthly expansion should be the default 'correct' answer:
    - matches edge-binning reference on [start_date, end_date)
    - has no dead params/periods and no uncovered times
    """
    start_date = case["start_date"]
    end_date = case["end_date"]

    times = _make_time_index(start_date, end_date, freq="12h", start_offset_hours=case["start_offset_hours"])
    H_bc = _make_H_bc(bc_regions=bc_regions, times=times)

    out = BoundaryAlignment.prepare(H_bc, frequency="monthly").data
    out_np = out.transpose("bc_region", "time").values

    # Monthly boundary periods are based on time.min(),
    # not on requested start_date/end_date edges.
    period_index, nperiod = _monthly_period_index_like_transform(times)
    ref = _expand_Hbc_by_period(H_bc, period_index=period_index, nperiod=nperiod)

    assert out_np.shape == ref.shape
    np.testing.assert_allclose(out_np, ref, rtol=0, atol=0)

    rep = report_dead_bc_periods(out, atol=0.0)
    assert not bool(rep["dead_row"].any()), "New monthly transform produced dead BC rows"
    assert not bool(rep["dead_period"].any()), "New monthly transform produced a dead BC period"

    # Also check uncovered time steps (all-zero columns)
    col_max = np.abs(out).max("bc_region", skipna=True)
    assert not bool((col_max <= 0.0).any()), "New monthly transform produced uncovered time columns"


@pytest.mark.parametrize("bc_regions", BC_REGIONS_CASES)
@pytest.mark.parametrize("freq", ["8D", "12H"])
@pytest.mark.parametrize("start_offset_hours", [0, 6])
def test_prepare_boundary_sensitivity_freq_matches_reference_and_no_dead(bc_regions, freq, start_offset_hours):
    """
    New freq-string expansion treated as correct.
    For correctness relative to old create_bc_sensitivity, boundary preparation must be anchored
    to start_date (not time.min(), not pandas floor origin). This test assumes that.
    """
    start_date = "2019-01-01"
    end_date = "2019-03-10"

    time_freq = "6h" if freq.upper().endswith("H") else "12h"
    times = _make_time_index(start_date, end_date, freq=time_freq, start_offset_hours=start_offset_hours)
    H_bc = _make_H_bc(bc_regions=bc_regions, times=times)

    out = BoundaryAlignment.prepare(H_bc, frequency=freq, anchor_time=start_date).data
    out_np = out.transpose("bc_region", "time").values

    edges = _freq_edges_old_create_bc(start_date, end_date, freq=freq)
    ndates = int(np.sum(edges < pd.to_datetime(end_date)))
    period_index = _period_index_from_edges(times, edges[: ndates + 1])
    ref = _expand_Hbc_by_period(H_bc, period_index=period_index, nperiod=ndates)

    assert out_np.shape == ref.shape
    np.testing.assert_allclose(out_np, ref, rtol=0, atol=0)

    rep = report_dead_bc_periods(out, atol=0.0)
    assert not bool(rep["dead_row"].any()), "New freq transform produced dead BC rows"
    assert not bool(rep["dead_period"].any()), "New freq transform produced a dead BC period"

    col_max = np.abs(out).max("bc_region", skipna=True)
    assert not bool((col_max <= 0.0).any()), "New freq transform produced uncovered time columns"
