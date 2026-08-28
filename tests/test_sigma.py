"""Tests for backend-neutral sigma alignment."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import openghg_inversions.sigma as sigma_module
from openghg_inversions.sigma import SigmaAlignment


def _index(
    values: tuple[float | int, ...] = (0, 0, 1, 1),
    *,
    times: tuple[str | None, ...] = ("2019-01-02", "2019-01-20", "2019-03-01", "2019-06-01"),
) -> xr.DataArray:
    """Build a labelled observation index."""
    return xr.DataArray(
        np.asarray(values),
        dims=("nmeasure",),
        coords={
            "nmeasure": ["a", "b", "c", "d"],
            "time": ("nmeasure", pd.to_datetime(times)),
        },
    )


def test_sigma_alignment_is_backend_independent_and_canonical() -> None:
    """Alignment normalises indexes without exposing backend dependencies."""
    alignment = SigmaAlignment.from_indices(
        _index((0.0, 0.0, 1.0, 1.0)),
        _index((2.0, 0.0, 1.0, 2.0)),
    )

    assert {"pymc", "pytensor", "arviz", "pm", "pt"}.isdisjoint(vars(sigma_module))
    assert alignment.site_index.name == "sigma_site_index"
    assert alignment.period_index.name == "sigma_period_index"
    assert np.issubdtype(alignment.site_index.dtype, np.integer)
    assert (alignment.nsite, alignment.nperiod) == (2, 3)


def test_shared_sigma_replaces_site_positions_with_zeros() -> None:
    """Shared sigma retains labels while using one latent site position."""
    alignment = SigmaAlignment.from_indices(_index(), _index((0, 1, 0, 1)), per_site=False)

    np.testing.assert_array_equal(alignment.site_index, np.zeros(4, dtype=int))
    assert alignment.nsite == 1
    assert "time" in alignment.site_index.coords


@pytest.mark.parametrize(
    ("frequency", "times", "anchor_time", "expected"),
    [
        (None, None, None, [0, 0, 0, 0]),
        (
            "monthly",
            ("2019-01-02", "2019-01-20", "2019-03-01", "2019-06-01"),
            None,
            [0, 0, 1, 2],
        ),
        (
            "8D",
            ("2019-01-08", "2019-01-09", "2019-01-15", "2019-01-15"),
            "2019-01-01",
            [0, 1, 1, 1],
        ),
    ],
)
def test_from_frequency_derives_compact_periods(
    frequency: str | None,
    times: tuple[str, ...] | None,
    anchor_time: str | None,
    expected: list[int],
) -> None:
    """Frequency construction handles defaults, calendar gaps, and anchors."""
    site = _index() if times is None else _index(times=times)
    if frequency is None:
        site = site.drop_vars("time")

    alignment = SigmaAlignment.from_frequency(
        site,
        frequency=frequency,
        anchor_time=anchor_time,
    )

    np.testing.assert_array_equal(alignment.period_index, expected)


def test_from_observations_derives_site_positions_and_periods() -> None:
    """Observation labels are the source of sigma site alignment."""
    observations = xr.DataArray(
        np.ones(4),
        dims="nmeasure",
        coords={
            "site": ("nmeasure", ["TAC", "TAC", "MHD", "MHD"]),
            "time": ("nmeasure", pd.to_datetime(["2019-01-01", "2019-01-09"] * 2)),
        },
    )

    alignment = SigmaAlignment.from_observations(
        observations,
        frequency="8D",
        anchor_time="2019-01-01",
    )

    np.testing.assert_array_equal(alignment.site_index, [0, 0, 1, 1])
    np.testing.assert_array_equal(alignment.period_index, [0, 1, 0, 1])


@pytest.mark.parametrize(
    "values",
    [
        (False, False, True, True),
        (0.0, 0.5, 1.0, 1.0),
        (0.0, np.nan, 1.0, 1.0),
        (0, -1, 1, 1),
        (0, 0, 1, np.iinfo(np.uint64).max),
        (0.0, 0.0, 1.0, 1e100),
        ("0", "0", "1", "1"),
    ],
)
def test_from_indices_rejects_invalid_values(values: tuple[object, ...]) -> None:
    """Indexes reject non-integral, non-finite, negative, and overflowing values."""
    invalid = _index()
    invalid.data = np.asarray(values)

    with pytest.raises(ValueError, match="integer|range"):
        SigmaAlignment.from_indices(invalid, _index())


def test_alignment_rejects_invalid_shapes_and_conflicting_coordinates() -> None:
    """Indexes must share a valid observation dimension and coordinate metadata."""
    with pytest.raises(ValueError, match="non-empty vector"):
        SigmaAlignment.from_indices(xr.DataArray([0], dims=("other",)), _index())
    with pytest.raises(ValueError, match="same observation length"):
        SigmaAlignment.from_indices(_index(), xr.DataArray([0, 1], dims=("nmeasure",)))

    period = _index()
    period.coords["time"].attrs["standard_name"] = "forecast_reference_time"
    site = _index()
    site.coords["time"].attrs["standard_name"] = "time"
    with pytest.raises(ValueError, match="incompatible coordinate"):
        SigmaAlignment.from_indices(site, period)


@pytest.mark.parametrize("frequency", ["monthly", "8D"])
def test_from_frequency_requires_complete_timestamps(frequency: str) -> None:
    """Derived periods require present, complete observation timestamps."""
    with pytest.raises(ValueError, match="timestamps"):
        SigmaAlignment.from_frequency(_index().drop_vars("time"), frequency=frequency)
    with pytest.raises(ValueError, match="timestamps"):
        SigmaAlignment.from_frequency(
            _index(times=("2019-01-01", None, "2019-01-03", "2019-01-04")),
            frequency=frequency,
        )


def test_align_vectorises_latent_sigma_over_observations() -> None:
    """Latent sigma values are aligned without losing trace dimensions."""
    alignment = SigmaAlignment.from_indices(_index(), _index((2, 0, 1, 2)))
    sigma = xr.DataArray(
        np.arange(24).reshape(2, 2, 3, 2),
        dims=("chain", "draw", "nsigma_time", "nsigma_site"),
        name="sigma",
    )

    actual = alignment.align(sigma)

    assert actual.name == "sigma_aligned"
    assert actual.dims == ("chain", "draw", "nmeasure")
    np.testing.assert_array_equal(
        actual,
        sigma.isel(
            nsigma_site=alignment.site_index,
            nsigma_time=alignment.period_index,
        ),
    )
