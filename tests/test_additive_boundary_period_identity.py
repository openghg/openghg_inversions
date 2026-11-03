"""Boundary state labels carry calendar meaning before numerical execution."""

from dataclasses import replace

import dask.array as da
from dask import delayed
import numpy as np
import pandas as pd
import pytensor
import pytest
import xarray as xr

from openghg_inversions.models.boundary import prepare_boundary
from openghg_inversions.serialization import (
    decode_cf_multiindexes,
    encode_cf_multiindexes,
    open_datatree_loaded,
    save_datatree,
)


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def native(times=("2020-01-01", "2020-01-20", "2020-02-01", "2020-02-20")):
    rows = np.arange(len(times))
    observations = xr.DataArray(
        np.zeros(len(rows)), dims="nmeasure",
        coords={"nmeasure": rows, "time": ("nmeasure", pd.to_datetime(times))},
        attrs={"units": "ppm"},
    )
    mass = xr.DataArray(
        np.tile([0.7, 0.3], (len(rows), 1)), dims=("nmeasure", "bc_region"),
        coords={"nmeasure": rows, "bc_region": ["north", "south"]}, attrs={"units": "1"},
    )
    operands = xr.Dataset(
        {"H_bc_anomaly": (mass * 0.2).assign_attrs(units="ppm"),
         "H_bc_mass": mass, "bc_reference": xr.DataArray(400.0, attrs={"units": "ppm"})},
        attrs={"boundary_representation": "centered-v1"},
    )
    return operands, observations


def weights(bound):
    # Full-state weights are deliberately nonseparable across curtain and period.
    return xr.DataArray(
        [1.0, 4.0, 9.0, 2.0], dims="bc_region", coords=bound.bc_region.coords,
        attrs={"units": "1"}, name="bc_centering_weights",
    )


def prepare(raw, weight, observations, frequency=None, anchor=None):
    return prepare_boundary(
        observations, boundary_operands=raw, bc_frequency=frequency, bc_anchor_time=anchor,
        bc_mean_shift_prior={"pdf": "uniform", "lower": -0.5, "upper": 0.5},
        bc_anomaly_scale=0.04, bc_centering_weights=weight,
    )


def calendar(raw, observations, frequency):
    return prepare_boundary(
        observations, boundary_operands=raw, bc_frequency=frequency,
        bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
    ).data


def never_compute(array):
    @delayed
    def fail():
        raise AssertionError("Mismatch evaluated a borrowed numerical payload")
    return array.copy(data=da.from_delayed(fail(), shape=array.shape, dtype=array.dtype))


@pytest.mark.parametrize("mismatch", ["frequency", "anchor", "window"])
def test_calendar_mismatch_rejected_before_mass_or_observation_compute(mismatch):
    raw, obs = native()
    if mismatch == "anchor":
        raw, obs = native(("2020-01-10", "2020-01-20", "2020-02-10", "2020-02-20"))
    frequency = "45D" if mismatch == "anchor" else "monthly"
    anchor = "2020-01-01" if mismatch == "anchor" else None
    original = prepare_boundary(
        obs, boundary_operands=raw, bc_frequency=frequency, bc_anchor_time=anchor,
        bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.1},
    ).data
    weight = never_compute(weights(original))
    if mismatch == "frequency":
        frequency, anchor = "45D", "2020-01-01"
    elif mismatch == "anchor":
        anchor = "2020-01-02"
    else:
        obs = obs.assign_coords(time=("nmeasure", obs.time.values + np.timedelta64(31, "D")))
    raw = raw.assign(
        H_bc_mass=never_compute(raw.H_bc_mass), H_bc_anomaly=never_compute(raw.H_bc_anomaly)
    )
    with pytest.raises(ValueError, match="align|index"):
        prepare(raw, weight, never_compute(obs), frequency, anchor)


def test_equivalent_duration_and_resolved_origin_preserve_nonseparable_prior():
    raw, obs = native()
    # One state has no transport, but its independently supplied weight remains.
    mass = raw.H_bc_mass.where(~((raw.nmeasure < 3) & (raw.bc_region == "north")), 0.0)
    raw = raw.assign(H_bc_mass=mass.assign_attrs(units="1"))
    first = calendar(raw, obs, "45D")
    selected = weights(first)
    result = prepare(raw, selected, obs, "1080h", "2020-01-01")
    xr.testing.assert_identical(result.data.bc_centering_weights, selected)
    np.testing.assert_array_equal(result.data.bc_period, first.bc_period)
    assert result.data.sizes["bc_region"] == 4
    assert float(result.data.H_bc_mass.isel(bc_region=0).sum()) == 0.0
    assert np.linalg.det(selected.values.reshape(2, 2)) != 0.0
    assert result.mean_prior == {"pdf": "uniform", "lower": -0.5, "upper": 0.5}
    assert result.anomaly_scale == 0.04


def test_single_temporal_period_requires_period_labels_before_payload_compute():
    raw, obs = native(("2020-01-01", "2020-01-20"))
    curtain_weights = xr.DataArray(
        [1.0, 3.0], dims="bc_curtain", coords={"bc_curtain": ["north", "south"]},
        attrs={"units": "1"},
    )
    bound = prepare(raw, curtain_weights, obs)
    np.testing.assert_array_equal(bound.data.bc_centering_weights, [1.0, 3.0])
    raw = raw.assign(H_bc_mass=never_compute(raw.H_bc_mass))
    with pytest.raises(ValueError, match="period|bc_region|dimension"):
        prepare(raw, never_compute(curtain_weights), never_compute(obs), "monthly")


@pytest.mark.parametrize("extension", ["nc", "zarr"])
def test_first_class_calendar_labels_roundtrip_without_identity_metadata(tmp_path, extension):
    raw, obs = native()
    bound = calendar(raw, obs, "monthly")
    expected = prepare(raw, weights(bound), obs, "monthly")
    path = tmp_path / f"weights.{extension}"
    save_datatree(xr.DataTree(encode_cf_multiindexes(weights(bound).to_dataset(), "bc_region")), path)
    loaded = open_datatree_loaded(path).to_dataset()
    restored = decode_cf_multiindexes(loaded, "bc_region").bc_centering_weights
    assert "boundary_period_identity" not in restored.bc_region.attrs
    np.testing.assert_array_equal(restored.bc_period, pd.to_datetime(["2020-01-01", "2020-02-01"] * 2))
    actual = prepare(raw, restored, obs, "monthly")
    xr.testing.assert_identical(actual.data, expected.data)
    subset = prepare(raw, restored, obs.isel(nmeasure=[0, 3]), "monthly")
    xr.testing.assert_identical(subset.data.bc_centering_weights, expected.data.bc_centering_weights)
    xr.testing.assert_identical(subset.data.H_bc_mass, expected.data.H_bc_mass.isel(nmeasure=[0, 3]))


@pytest.mark.parametrize("extension", ["nc", "zarr"])
def test_linked_prepared_weights_preserve_period_identity(tmp_path, extension):
    from test_co2_o2_fixed_ou import _prepared
    from openghg_inversions.rhime.co2.co2_o2_preparation import Co2O2PreparedInputs

    initial = _prepared()
    operands, selected_weights, observations = {}, {}, {}
    for channel, scale in (("co2", 1.0), ("o2", 10.0)):
        times = initial.observations.time.values[initial.observations.species.values == channel]
        raw, obs = native(times)
        sensitivity = getattr(initial, f"{channel}_sensitivity")
        labels = sensitivity[sensitivity.dims[0]].values
        raw = raw.assign_coords(nmeasure=labels).reindex(bc_region=["north", "south", "unused"], fill_value=0)
        observations[channel] = obs.assign_coords(nmeasure=labels)
        operands[channel] = raw
        hours = [0, 4] if channel == "co2" else [0, 2, 4]
        periods = np.datetime64("2021-01-01T00", "ns") + np.array(hours).astype("timedelta64[h]")
        values = scale * np.arange(1, 3 * len(periods) + 1, dtype=float).reshape(3, len(periods))
        selected_weights[channel] = xr.DataArray(
            values, dims=("bc_curtain", "bc_period"),
            coords={"bc_curtain": ["north", "south", "unused"], "bc_period": periods},
            attrs={"units": "1"}, name="bc_centering_weights",
        )
        assert np.linalg.matrix_rank(values) == 2

    prepared = replace(initial, boundary_operands=operands, bc_centering_weights=selected_weights)
    path = tmp_path / f"linked.{extension}"
    prepared.save(path)
    loaded = Co2O2PreparedInputs.load(path)
    assert set(loaded.bc_centering_weights) == {"co2", "o2"}
    for channel in ("co2", "o2"):
        actual = loaded.bc_centering_weights[channel]
        expected = selected_weights[channel]
        xr.testing.assert_identical(actual, expected)
        assert isinstance(actual.indexes["bc_period"], pd.DatetimeIndex)
        np.testing.assert_array_equal(actual.sel(bc_curtain="unused"), expected.sel(bc_curtain="unused"))
        bound = prepare(loaded.boundary_operands[channel], actual, observations[channel], "2h")
        xr.testing.assert_identical(bound.data.bc_centering_weights.unstack("bc_region"), expected)
        np.testing.assert_array_equal(
            bound.data.H_bc_mass.unstack("bc_region").sel(bc_curtain="unused"), 0.0
        )
