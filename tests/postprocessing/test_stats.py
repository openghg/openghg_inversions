"""Tests for postprocessing statistics helpers."""

from __future__ import annotations

import numpy as np
import xarray as xr

from openghg_inversions.postprocessing.stats import mode_kde


def test_mode_kde_handles_nan_rows_with_dask_chunks() -> None:
    """KDE mode should filter NaNs per row instead of dropping all draws globally."""
    data = np.array(
        [
            [np.nan, 1.0, 1.0],
            [np.nan, 2.0, np.nan],
            [np.nan, 3.0, 2.0],
        ]
    )
    ds = xr.Dataset({"y": (("draw", "nmeasure"), data)})

    result = mode_kde(ds, chunk_dim="nmeasure", chunk_size=1).compute()

    assert result["y_mode"].dims == ("nmeasure",)
    assert np.isnan(result["y_mode"].values[0])
    assert np.isfinite(result["y_mode"].values[1])
    assert np.isfinite(result["y_mode"].values[2])


def test_mode_kde_handles_single_finite_value() -> None:
    """Rows with one finite value should return that value without calling scipy KDE."""
    ds = xr.Dataset({"y": (("draw", "nmeasure"), np.array([[np.nan], [4.2], [np.nan]]))})

    result = mode_kde(ds, chunk_dim="nmeasure", chunk_size=1).compute()

    np.testing.assert_allclose(result["y_mode"].values, [4.2])
