"""Focused preparation contracts for the CO2/O2 recipe."""

from __future__ import annotations

from dask import array as da
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.rhime.co2 import prepare_co2_o2_inputs


def _inputs(
    *,
    co2_o2_ocean_loading: float = 0.0,
    o2_co2_ocean_loading: float = 0.0,
) -> dict[str, object]:
    state = ["gpp:1", "ter:1", "ff:1", "co2-ocean:1", "o2-ocean:1"]
    mean = xr.DataArray(
        np.ones(5),
        dims="state",
        coords={
            "state": state,
            "source": ("state", ["GPP", "TER", "FF", "ocean", "ocean"]),
            "tracer_scope": ("state", ["shared", "shared", "shared", "co2", "o2"]),
        },
    )
    co2_labels = ["c1", "c2"]
    o2_labels = ["o1", "o2", "o3"]
    co2 = xr.DataArray(
        da.from_array([2.0, 3.0], chunks=1),
        dims="co2_measure",
        coords={
            "co2_measure": co2_labels,
            "time": ("co2_measure", np.array(["2021-01-01", "2021-01-03"], dtype="datetime64[D]")),
        },
    )
    o2 = xr.DataArray(
        da.from_array([-4.0, -5.0, -6.0], chunks=1),
        dims="o2_measure",
        coords={
            "o2_measure": o2_labels,
            "time": (
                "o2_measure",
                np.array(["2021-01-02", "2021-01-04", "2021-01-05"], dtype="datetime64[D]"),
            ),
        },
    )
    return {
        "co2_observations": co2,
        "o2_observations": o2,
        "co2_prior_forward_mean": co2 - 0.25,
        "o2_prior_forward_mean": o2 + 0.5,
        "co2_operator": xr.DataArray(
            da.from_array(
                [[1, 2, 3, 4, co2_o2_ocean_loading], [0.5, 1, 1.5, 2, 0]],
                chunks=(1, 5),
            ),
            dims=("co2_measure", "state"),
            coords={"co2_measure": co2_labels, "state": state},
        ),
        "o2_operator": xr.DataArray(
            da.from_array(
                [
                    [-1, -2, -3, o2_co2_ocean_loading, 5],
                    [-0.5, -1, -1.5, 0, 2.5],
                    [-0.2, -0.4, -0.6, 0, 1],
                ],
                chunks=(1, 5),
            ),
            dims=("o2_measure", "state"),
            coords={"o2_measure": o2_labels, "state": state},
        ),
        "co2_aggregation_covariance": xr.DataArray(
            da.from_array(np.eye(2), chunks=(1, 2)),
            dims=("co2_measure", "co2_measure_cov"),
            coords={"co2_measure": co2_labels, "co2_measure_cov": co2_labels},
        ),
        "co2_o2_aggregation_covariance": xr.DataArray(
            da.from_array(np.zeros((2, 3)), chunks=(1, 3)),
            dims=("co2_measure", "o2_measure"),
            coords={"co2_measure": co2_labels, "o2_measure": o2_labels},
        ),
        "o2_aggregation_covariance": xr.DataArray(
            da.from_array(np.eye(3), chunks=(1, 3)),
            dims=("o2_measure", "o2_measure_cov"),
            coords={"o2_measure": o2_labels, "o2_measure_cov": o2_labels},
        ),
        "retained_prior": CorrelatedLognormalPrior(mean, np.eye(5) * 0.01),
        "co2_units": "ppm",
        "o2_units": "per meg",
    }


def test_preparation_preserves_lazy_channels_with_staggered_unequal_times() -> None:
    prepared = prepare_co2_o2_inputs(**_inputs())

    assert isinstance(prepared.observations.data, da.Array)
    assert isinstance(prepared.prior_forward_mean.data, da.Array)
    assert isinstance(prepared.co2_operator.data, da.Array)
    assert isinstance(prepared.o2_operator.data, da.Array)
    assert not isinstance(prepared.aggregation_error.covariance.data, da.Array)
    assert prepared.observations["tracer"].values.tolist() == ["co2", "co2", "o2", "o2", "o2"]
    assert prepared.observations["within_tracer_observation"].values.tolist() == [
        "c1",
        "c2",
        "o1",
        "o2",
        "o3",
    ]
    np.testing.assert_array_equal(
        prepared.observations["time"],
        np.array(
            ["2021-01-01", "2021-01-03", "2021-01-02", "2021-01-04", "2021-01-05"],
            dtype="datetime64[D]",
        ),
    )


def test_rejects_co2_loading_on_o2_ocean_state() -> None:
    with pytest.raises(ValueError, match="CO2 operator.*O2-specific ocean"):
        prepare_co2_o2_inputs(**_inputs(co2_o2_ocean_loading=0.1))


def test_rejects_o2_loading_on_co2_ocean_state() -> None:
    with pytest.raises(ValueError, match="O2 operator.*CO2-specific ocean"):
        prepare_co2_o2_inputs(**_inputs(o2_co2_ocean_loading=0.1))
