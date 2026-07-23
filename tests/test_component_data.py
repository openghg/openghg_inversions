"""Regression tests for pure sigma component data and reconstruction."""

from dataclasses import FrozenInstanceError

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import openghg_inversions.component_data as component_data
from openghg_inversions.component_data import (
    SigmaComponentData,
    prepare_sigma_component_data,
    reconstruct_sigma_aligned,
)


def _site_indicator(
    *,
    values: tuple[int | float, ...] = (0, 0, 1, 1),
    times: tuple[str, ...] = ("2019-01-01", "2019-01-02", "2019-03-01", "2019-03-02"),
) -> xr.DataArray:
    """Build an observation-aligned site indicator with labelled coordinates."""
    site_labels = np.asarray(["MHD" if value == 0 else "TAC" for value in values])
    return xr.DataArray(
        np.asarray(values),
        dims=("nmeasure",),
        coords={
            "nmeasure": np.arange(len(values)),
            "site": ("nmeasure", site_labels),
            "time": ("nmeasure", pd.to_datetime(times)),
        },
        name="site_indicator",
    )


def _posterior_sigma(nsite: int = 2, ntime: int = 3) -> xr.Dataset:
    """Build deterministic posterior sigma draws for alignment tests."""
    values = np.arange(2 * 3 * nsite * ntime, dtype=float).reshape(2, 3, nsite, ntime)
    return xr.Dataset(
        {
            "sigma": (
                ("chain", "draw", "nsigma_site", "nsigma_time"),
                values,
            )
        },
        coords={
            "chain": [4, 7],
            "draw": [10, 20, 30],
            "nsigma_site": np.arange(nsite),
            "nsigma_time": np.arange(ntime),
        },
    )


def _model_data(
    site_index: tuple[int, ...] = (0, 1, 1, 0, 1),
    freq_index: tuple[int, ...] = (2, 0, 1, 1, 2),
) -> xr.Dataset:
    """Build labelled sigma indexes as they appear in model constant data."""
    site = np.asarray(["MHD" if value == 0 else "TAC" for value in site_index])
    time = pd.date_range("2019-01-01", periods=len(site_index), freq="12h")
    return xr.Dataset(
        {
            "site_indicator": ("nmeasure", np.asarray(site_index)),
            "sigma_freq_index": ("nmeasure", np.asarray(freq_index)),
        },
        coords={
            "nmeasure": np.arange(len(site_index)),
            "site": ("nmeasure", site),
            "time": ("nmeasure", time),
        },
    )


def test_prepare_sigma_component_data_is_independent_of_pymc() -> None:
    """Pure sigma preparation does not depend on PyMC or PyTensor objects."""
    prepared = prepare_sigma_component_data(_site_indicator(), sigma_freq=None)

    assert isinstance(prepared, SigmaComponentData)
    assert {"pymc", "pytensor", "pm", "pt"}.isdisjoint(vars(component_data))


def test_prepare_sigma_component_data_has_effective_indexes_and_names() -> None:
    """Preparation exposes the effective per-site and shared model-data indexes."""
    site_indicator = _site_indicator()
    freq_index = xr.DataArray(
        [0, 0, 1, 1],
        dims=("nmeasure",),
        coords=site_indicator.coords,
        name="sigma_freq_index",
    )

    per_site = prepare_sigma_component_data(site_indicator, sigma_freq_index=freq_index)
    shared = prepare_sigma_component_data(
        site_indicator,
        sigma_freq_index=freq_index,
        per_site=False,
    )

    assert isinstance(per_site, SigmaComponentData)
    np.testing.assert_array_equal(per_site.site_index, [0, 0, 1, 1])
    np.testing.assert_array_equal(per_site.freq_index, [0, 0, 1, 1])
    assert per_site.site_index_name == "site_indicator"
    assert per_site.freq_index_name == "sigma_freq_index"
    assert (per_site.nsigma_site, per_site.nsigma_time) == (2, 2)

    np.testing.assert_array_equal(shared.site_index, np.zeros(4, dtype=int))
    assert shared.site_index_name == "sigma_site_indicator"
    assert shared.freq_index_name == "sigma_freq_index"
    assert (shared.nsigma_site, shared.nsigma_time) == (1, 2)

    with pytest.raises(FrozenInstanceError):
        setattr(shared, "site_index", site_indicator)


def test_prepare_sigma_component_data_none_frequency_is_one_shared_period() -> None:
    """A missing sigma frequency produces a zero index for one effective period."""
    prepared = prepare_sigma_component_data(_site_indicator(), sigma_freq=None)

    np.testing.assert_array_equal(prepared.freq_index, np.zeros(4, dtype=int))
    assert prepared.nsigma_time == 1


def test_prepare_sigma_component_data_compacts_monthly_gaps() -> None:
    """Monthly preparation compacts absent calendar months into consecutive periods."""
    site_indicator = _site_indicator(
        values=(0, 0, 1, 1, 0),
        times=("2019-01-02", "2019-01-20", "2019-03-01", "2019-03-18", "2019-06-01"),
    )

    prepared = prepare_sigma_component_data(site_indicator, sigma_freq="monthly")

    np.testing.assert_array_equal(prepared.freq_index, [0, 0, 1, 1, 2])
    assert prepared.nsigma_time == 3


def test_prepare_sigma_component_data_anchors_fixed_duration_periods() -> None:
    """Fixed-duration periods remain anchored when early observations are absent."""
    site_indicator = _site_indicator(
        values=(0, 0, 0),
        times=("2019-01-08", "2019-01-09", "2019-01-15"),
    )

    prepared = prepare_sigma_component_data(
        site_indicator,
        sigma_freq="8D",
        anchor_time="2019-01-01",
    )

    np.testing.assert_array_equal(prepared.freq_index, [0, 1, 1])


@pytest.mark.parametrize(
    ("site_values", "freq_values"),
    [
        ((0.0, 0.5, 1.0, 1.0), (0, 0, 1, 1)),
        ((0, 0, 1, 1), (0, -1, 1, 1)),
        ((0, 0, 1, 1), (0 + 0j, 0 + 0j, 1 + 0j, 1 + 0j)),
    ],
)
def test_prepare_sigma_component_data_rejects_malformed_indexes(
    site_values: tuple[complex | float | int, ...],
    freq_values: tuple[complex | float | int, ...],
) -> None:
    """Preparation rejects non-integral, negative, or complex alignment indexes."""
    site_indicator = _site_indicator()
    site_indicator.data = np.asarray(site_values)
    freq_index = xr.DataArray(
        np.asarray(freq_values),
        dims=("nmeasure",),
        coords=site_indicator.coords,
    )

    with pytest.raises(ValueError, match="index|indicator"):
        prepare_sigma_component_data(site_indicator, sigma_freq_index=freq_index)


def test_reconstruct_sigma_aligned_from_explicit_model_data() -> None:
    """Reconstruction vectorizes posterior sigma over labelled observation indexes."""
    posterior = _posterior_sigma()
    model_data = _model_data()
    idata = az.InferenceData(posterior=posterior)

    assert "sigma_aligned" not in posterior
    actual = reconstruct_sigma_aligned(idata, model_data)
    expected = posterior["sigma"].isel(
        nsigma_site=model_data["site_indicator"],
        nsigma_time=model_data["sigma_freq_index"],
    )

    assert actual.dims == ("chain", "draw", "nmeasure")
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.coords["chain"], posterior.coords["chain"])
    np.testing.assert_array_equal(actual.coords["draw"], posterior.coords["draw"])
    np.testing.assert_array_equal(actual.coords["nmeasure"], model_data.coords["nmeasure"])
    np.testing.assert_array_equal(actual.coords["site"], model_data.coords["site"])
    np.testing.assert_array_equal(actual.coords["time"], model_data.coords["time"])
    assert actual.coords["site"].dims == ("nmeasure",)
    assert actual.coords["time"].dims == ("nmeasure",)


def test_reconstruct_sigma_aligned_defaults_to_constant_data_and_shared_index() -> None:
    """Default reconstruction prefers the effective shared-site indicator."""
    posterior = _posterior_sigma(nsite=1, ntime=2)
    model_data = _model_data(
        site_index=(1, 1, 1, 1, 1),
        freq_index=(0, 1, 0, 1, 1),
    )
    model_data["sigma_site_indicator"] = xr.zeros_like(model_data["site_indicator"])
    idata = az.InferenceData(posterior=posterior, constant_data=model_data)

    actual = reconstruct_sigma_aligned(idata)
    expected = posterior["sigma"].isel(
        nsigma_site=model_data["sigma_site_indicator"],
        nsigma_time=model_data["sigma_freq_index"],
    )

    assert "sigma_aligned" not in posterior
    assert actual.dims == ("chain", "draw", "nmeasure")
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.coords["site"], model_data.coords["site"])
    np.testing.assert_array_equal(actual.coords["time"], model_data.coords["time"])
    assert actual.coords["site"].dims == ("nmeasure",)
    assert actual.coords["time"].dims == ("nmeasure",)


def test_reconstruct_sigma_aligned_accepts_named_frequency_index() -> None:
    """Reconstruction accepts a component's explicitly named frequency index."""
    posterior = _posterior_sigma()
    model_data = _model_data().rename({"sigma_freq_index": "custom_sigma_period"})
    idata = az.InferenceData(posterior=posterior, constant_data=model_data)

    actual = reconstruct_sigma_aligned(idata, freq_index_name="custom_sigma_period")
    expected = posterior["sigma"].isel(
        nsigma_site=model_data["site_indicator"],
        nsigma_time=model_data["custom_sigma_period"],
    )

    assert actual.dims == ("chain", "draw", "nmeasure")
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.coords["site"], model_data.coords["site"])
    np.testing.assert_array_equal(actual.coords["time"], model_data.coords["time"])
