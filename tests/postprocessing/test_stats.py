"""Tests for postprocessing statistics helpers."""

from __future__ import annotations

import numpy as np
import pytest
import sparse
import xarray as xr

from openghg_inversions.postprocessing.stats import hdi, mean, median, mode, mode_kde, quantiles, stdev


def _sparse_dataset(*, draw_chunk_size: int | None = None) -> xr.Dataset:
    """Create a small dataset backed by a sparse array or sparse dask chunks."""
    data = sparse.COO.from_numpy(np.tile([0.0, 3.0], (5, 1)))
    result = xr.Dataset({"value": (("draw", "place"), data)})
    if draw_chunk_size is not None:
        return result.chunk({"draw": draw_chunk_size, "place": 1})
    return result


@pytest.mark.parametrize(
    ("statistic", "kwargs", "result_name", "expected"),
    [
        pytest.param(quantiles, {"quantiles": [0.5]}, "value_quantile", [[0.0, 3.0]], id="quantiles"),
        pytest.param(median, {}, "value_median", [0.0, 3.0], id="median"),
        pytest.param(mode, {}, "value_mode", [0.0, 3.0], id="mode"),
        pytest.param(mode_kde, {}, "value_mode", [0.0, 3.0], id="mode-kde"),
        pytest.param(hdi, {"hdi_prob": 0.6}, "value_hdi_60", [[0.0, 0.0], [3.0, 3.0]], id="hdi"),
    ],
)
@pytest.mark.parametrize(
    "draw_chunk_size",
    [
        pytest.param(None, id="eager"),
        pytest.param(5, id="one-draw-chunk"),
        pytest.param(2, id="multiple-draw-chunks"),
    ],
)
def test_dense_statistics_accept_sparse_data(
    statistic, kwargs, result_name, expected, draw_chunk_size
) -> None:
    """Statistics requiring dense operations should own conversion of sparse inputs."""
    result = statistic(_sparse_dataset(draw_chunk_size=draw_chunk_size), **kwargs)

    if hasattr(result[result_name].data, "_meta"):
        assert isinstance(result[result_name].data._meta, np.ndarray)
    computed = result.compute()
    assert isinstance(computed[result_name].data, np.ndarray)
    np.testing.assert_allclose(computed[result_name], expected)


@pytest.mark.parametrize(
    ("statistic", "result_name", "expected"),
    [
        pytest.param(mean, "value_mean", [0.0, 3.0], id="mean"),
        pytest.param(stdev, "value_stdev", [0.0, 0.0], id="stdev"),
    ],
)
@pytest.mark.parametrize(
    "draw_chunk_size",
    [pytest.param(5, id="one-draw-chunk"), pytest.param(2, id="multiple-draw-chunks")],
)
def test_sparse_compatible_statistics_preserve_sparse_chunks(
    statistic, result_name, expected, draw_chunk_size
) -> None:
    """Sparse-compatible statistics should not densify their dask chunks."""
    result = statistic(_sparse_dataset(draw_chunk_size=draw_chunk_size))

    assert isinstance(result[result_name].data._meta, sparse.COO)
    assert result[result_name].chunks == ((1, 1),)
    computed = result.compute()
    assert isinstance(computed[result_name].data, sparse.COO)
    np.testing.assert_allclose(computed[result_name].data.todense(), expected)


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
