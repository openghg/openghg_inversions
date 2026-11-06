"""Tests for the sampled boundary-sensitivity preparation owner."""

import dask.array as da
from dask import delayed
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.boundary_sensitivity import (
    BoundaryAlignment,
    scale_satellite_boundary_sensitivity_to_column_signal,
)
from openghg_inversions.inversion_inputs import make_inv_inputs


def _sensitivity(data: object | None = None) -> xr.DataArray:
    times = pd.date_range("2020-01-01", periods=3, freq="12h")
    return xr.DataArray(
        np.arange(6, dtype=float).reshape(2, 3) if data is None else data,
        dims=("bc_region", "time"),
        coords={"bc_region": ["north", "south"], "time": times},
        attrs={"units": "ppm"},
    )


def test_prepare_boundary_sensitivity_labels_one_period_and_provenance() -> None:
    result = BoundaryAlignment.prepare(_sensitivity()).data

    assert result.dims == ("bc_region", "time")
    assert result.indexes["bc_region"].names == ["bc_curtain", "bc_period"]
    assert result.attrs["units"] == "ppm"
    assert result.attrs["bc_frequency"] == "none"
    np.testing.assert_array_equal(result.bc_period, np.repeat(np.datetime64("2020-01-01", "ns"), 2))
    np.testing.assert_allclose(result.values, _sensitivity().values)


def test_timestamp_free_whole_window_has_explicit_label() -> None:
    source = _sensitivity().rename(time="nmeasure").assign_coords(nmeasure=[3, 2, 1])
    result = BoundaryAlignment.prepare(source).data
    np.testing.assert_array_equal(result.bc_period, ["whole_window", "whole_window"])
    np.testing.assert_array_equal(result, source)


@pytest.mark.parametrize(
    ("frequency", "anchor", "starts"),
    [("monthly", None, ["2020-02-01", "2020-03-01"]),
     ("10D", "2020-01-01", ["2020-01-31", "2020-03-01"])],
)
def test_period_mask_uses_actual_calendar_start_labels_with_gaps(frequency, anchor, starts) -> None:
    source = _sensitivity().assign_coords(time=pd.to_datetime(["2020-03-02", "2020-02-03", "2020-02-04"]))
    result = BoundaryAlignment.prepare(source, frequency=frequency, anchor_time=anchor).data
    np.testing.assert_array_equal(result.bc_period, np.tile(np.array(starts, dtype="datetime64[ns]"), 2))
    np.testing.assert_array_equal(result.sum("bc_region"), source.sum("bc_region"))
    np.testing.assert_array_equal(result.isel(bc_region=[0, 2], time=0), [0, 0])
    np.testing.assert_array_equal(result.isel(bc_region=[1, 3], time=[1, 2]), [[0, 0], [0, 0]])


@pytest.mark.parametrize("frequency", ["0h", "-1h", "NaT", "nan", "inf"])
@pytest.mark.filterwarnings("error")
def test_boundary_frequency_rejects_invalid_duration(frequency: str) -> None:
    with pytest.raises(ValueError, match="Invalid frequency"):
        BoundaryAlignment.prepare(_sensitivity(), frequency=frequency)


def test_prepare_boundary_sensitivity_reorders_observations() -> None:
    sensitivity = _sensitivity()
    reordered = sensitivity.time[[2, 0]]

    result = BoundaryAlignment.prepare(sensitivity, observation_labels=reordered).data

    np.testing.assert_array_equal(result.time, reordered)
    with pytest.raises(KeyError):
        BoundaryAlignment.prepare(
            sensitivity,
            observation_labels=xr.DataArray(
                [np.datetime64("2021-01-01")], dims="time", name="time"
            ),
        )

def test_prepare_boundary_sensitivity_preserves_lazy_data_and_borrowed_input() -> None:
    source = _sensitivity(da.arange(6, chunks=3).reshape((2, 3)))
    original = source.copy(deep=False)

    result = BoundaryAlignment.prepare(source, frequency="12h", anchor_time="2020-01-01").data

    assert hasattr(result.data, "__dask_graph__")
    xr.testing.assert_identical(source, original)


@pytest.mark.parametrize(("frequency", "expected_executions"), [("12h", 1), (None, 1)])
def test_prepare_boundary_sensitivity_computes_lazy_time_only_when_needed(
    frequency: str | None, expected_executions: int
) -> None:
    executions: list[None] = []

    @delayed
    def lazy_time() -> np.ndarray:
        executions.append(None)
        return pd.date_range("2020-01-01", periods=3, freq="12h").values

    source = xr.DataArray(
        da.arange(6, chunks=3).reshape((2, 3)),
        dims=("bc_region", "nmeasure"),
        coords={
            "bc_region": ["north", "south"],
            "nmeasure": [0, 1, 2],
            "time": ("nmeasure", da.from_delayed(lazy_time(), shape=(3,), dtype="datetime64[ns]")),
        },
    )

    result = BoundaryAlignment.prepare(source, frequency=frequency).data

    assert len(executions) == expected_executions
    assert hasattr(result.data, "__dask_graph__")


@pytest.mark.parametrize(
    ("frequency", "anchor_time", "expected"),
    [
        (None, "2019-01-01", "not-applicable"),
        ("monthly", "2019-01-01", "not-applicable"),
        ("12h", None, "2020-01-01T00:00:00.000000000"),
        ("12h", "2019-01-01", "2019-01-01"),
    ],
)
def test_prepare_boundary_sensitivity_records_effective_anchor(
    frequency: str | None, anchor_time: str | None, expected: str
) -> None:
    result = BoundaryAlignment.prepare(
        _sensitivity().isel(time=[2, 0, 1]), frequency=frequency, anchor_time=anchor_time
    ).data

    assert result.attrs["bc_anchor_time"] == expected


@pytest.mark.parametrize("lazy", [False, True])
def test_make_inv_inputs_drops_nan_boundary_rows_for_eager_and_lazy_data(lazy: bool) -> None:
    time = _sensitivity().time
    h_bc = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    if lazy:
        h_bc = da.from_array(h_bc, chunks=(2, 3))
    site_data = xr.Dataset(
        {
            "H": (("region", "time"), np.ones((1, 3))),
            "H_bc": (("bc_region", "time"), h_bc),
            "mf": ("time", np.ones(3)),
            "mf_error": ("time", np.ones(3)),
            "mf_repeatability": ("time", np.ones(3)),
            "mf_variability": ("time", np.ones(3)),
        },
        coords={"region": [0], "bc_region": ["north", "south"], "time": time},
    )

    result = make_inv_inputs({"AAA": site_data}, sites=["AAA"])

    assert result.sizes["nmeasure"] == 2
    assert np.isfinite(result["H_bc"]).all()


def test_satellite_boundary_scaling_uses_footprint_level_provenance() -> None:
    """Different verified footprint and observation levels retain the workaround."""
    inputs = xr.Dataset(
        {
            "H_bc": (("bc_region", "nmeasure"), [[100.0, 200.0], [300.0, 400.0]]),
            "mf": ("nmeasure", [50.0, 100.0]),
            "mf_prior_factor": ("nmeasure", [0.0, 0.0]),
            "mf_prior_upper_level_factor": ("nmeasure", [350.0, 300.0]),
            "site": ("nmeasure", ["OCO2-EASTASIA", "OCO2-EASTASIA"]),
        }
    )

    inputs["bc_centering_weights"] = xr.DataArray([2.0, 1.0], dims="bc_region", attrs={"units": "1"})
    original = inputs.copy(deep=True)
    result = scale_satellite_boundary_sensitivity_to_column_signal(
        inputs,
        sites=["OCO2-EASTASIA"],
        platform=["satellite"],
        observation_max_level=[3],
        footprint_max_level=[17],
    )

    np.testing.assert_allclose(result["H_bc"], [[12.5, 50.0], [37.5, 100.0]])

    xr.testing.assert_identical(result.bc_centering_weights, inputs.bc_centering_weights)
    xr.testing.assert_identical(inputs, original)


def test_satellite_boundary_scaling_skips_matching_provenance_levels() -> None:
    """The exported scaler retains its call signature and skips matching levels."""
    inputs = xr.Dataset(
        {
            "H_bc": (("bc_region", "nmeasure"), [[100.0, 200.0], [300.0, 400.0]]),
            "mf": ("nmeasure", [50.0, 100.0]),
            "mf_prior_factor": ("nmeasure", [0.0, 0.0]),
            "mf_prior_upper_level_factor": ("nmeasure", [350.0, 300.0]),
            "site": ("nmeasure", ["OCO2-EASTASIA", "OCO2-EASTASIA"]),
        }
    )

    result = scale_satellite_boundary_sensitivity_to_column_signal(
        inputs,
        sites=["OCO2-EASTASIA"],
        platform=["satellite"],
        observation_max_level=[17],
        footprint_max_level=[17],
    )

    xr.testing.assert_identical(result, inputs)


def test_satellite_boundary_scaling_keeps_legacy_call_signature() -> None:
    """Callers without vertical-level provenance retain the legacy scaling."""
    inputs = xr.Dataset(
        {
            "H_bc": (("bc_region", "nmeasure"), [[100.0]]),
            "mf": ("nmeasure", [50.0]),
            "mf_prior_factor": ("nmeasure", [0.0]),
            "mf_prior_upper_level_factor": ("nmeasure", [350.0]),
            "site": ("nmeasure", ["OCO2-EASTASIA"]),
        }
    )

    result = scale_satellite_boundary_sensitivity_to_column_signal(
        inputs,
        sites=["OCO2-EASTASIA"],
        platform=["satellite"],
    )

    np.testing.assert_allclose(result["H_bc"], [[12.5]])
