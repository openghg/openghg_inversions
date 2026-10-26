"""Ordinary and centered boundaries share model-owned observation periods."""

import dask.array as da
from dask.callbacks import Callback
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.boundary_sensitivity import (
    align_boundary_operands, bind_boundary_sensitivity,
)


@pytest.mark.parametrize("frequency,anchor", [(None, None), ("monthly", None), ("1D", "2020-01-30")])
def test_reusable_ordinary_and_centered_share_selected_periods(frequency, anchor):
    times = pd.to_datetime(["2020-01-30T12:00", "2020-01-31T12:00", "2020-02-01T12:00"])
    anomaly = xr.DataArray(da.from_array([[.1, .3, -.2], [.4, -.1, .2]], chunks=(1, 2)),
                          dims=("bc_region", "nmeasure"),
                          coords={"bc_region": ["land", "ocean"], "nmeasure": ["a", "b", "c"],
                                  "time": ("nmeasure", times)}, attrs={"units": "ppm"})
    raw = xr.Dataset({"H_bc_anomaly": anomaly, "H_bc_mass": xr.full_like(anomaly, .5).assign_attrs(units="1"),
                      "bc_reference": xr.DataArray(400., attrs={"units": "ppm"})},
                     attrs={"boundary_representation": "centered-v1"})
    absolute = (raw.H_bc_anomaly + raw.bc_reference * raw.H_bc_mass).assign_attrs(
        units="ppm", boundary_sensitivity_preparation="raw-spatial")
    observations = xr.DataArray([1., 2.], dims="nmeasure", coords={"nmeasure": ["c", "a"]})
    calls = []
    with Callback(posttask=lambda *args: calls.append(args)):
        ordinary = bind_boundary_sensitivity(absolute, observations, frequency=frequency, anchor_time=anchor)
        centered = align_boundary_operands(raw, frequency=frequency, anchor_time=anchor,
                                          observation_labels=observations.nmeasure)
    assert not calls
    assert hasattr(ordinary.data, "__dask_graph__")
    expected = (centered.H_bc_anomaly + centered.bc_reference * centered.H_bc_mass).transpose(*ordinary.dims)
    xr.testing.assert_allclose(ordinary.compute(), expected.compute())
    assert list(ordinary.nmeasure.values) == ["c", "a"]
    assert absolute.attrs["boundary_sensitivity_preparation"] == "raw-spatial"
    # Previously expanded data have a different contract from reusable raw data.
    with pytest.raises(ValueError, match="raw spatial"):
        bind_boundary_sensitivity(ordinary, observations, frequency="monthly")


@pytest.mark.parametrize("frequency,anchor", [(None, "NaT"), ("monthly", "2020-01-01"), ("1D", "NaT")])
def test_ordinary_binding_rejects_invalid_or_ignored_anchor(frequency, anchor):
    sensitivity = xr.DataArray([[1.]], dims=("nmeasure", "bc_region"),
                               coords={"nmeasure": [0], "bc_region": ["north"]})
    observations = xr.DataArray([1.], dims="nmeasure", coords={"nmeasure": [0]})
    with pytest.raises(ValueError, match="anchor_time"):
        bind_boundary_sensitivity(sensitivity, observations, frequency=frequency, anchor_time=anchor)


@pytest.mark.parametrize("centered", [False, True])
def test_public_linked_preparation_uses_observation_times_for_boundary_periods(centered):
    """Stale boundary auxiliaries cannot change the authoritative monthly mask."""
    import numpy as np
    import pytensor

    from openghg_inversions.rhime.co2 import prepare_co2_o2_inputs
    from openghg_inversions.rhime.co2.co2_o2_runner import build_rhime_co2_o2
    from test_centered_co2_boundaries import _operands
    from test_rhime_co2_o2 import _inputs, _independent_error

    inputs = _inputs()
    inputs["o2_units"] = "ppm"
    boundaries = {}
    originals = {}
    for channel in ("co2", "o2"):
        observations = inputs[f"{channel}_observations"]
        dim = str(observations.dims[0])
        times = np.array(["2020-01-01", "2020-02-01", "2020-02-02"], dtype="datetime64[D]")[:observations.size]
        observations = observations.assign_coords(time=(dim, times))
        inputs[f"{channel}_observations"] = observations
        raw = _operands(observations).assign_coords(
            time=("nmeasure", np.arange(observations.size).astype("timedelta64[D]") + np.datetime64("2020-01-01"))
        )
        if centered:
            boundaries[channel] = raw
        else:
            boundaries[channel] = (raw.H_bc_anomaly + raw.bc_reference * raw.H_bc_mass).rename(
                nmeasure=dim,
            ).transpose(dim, "bc_region").assign_attrs(units="ppm", boundary_sensitivity_preparation="raw-spatial")
        originals[channel] = boundaries[channel].copy(deep=True)
    inputs["boundary_operands" if centered else "boundary_sensitivity"] = boundaries
    prepared = prepare_co2_o2_inputs(**inputs)
    with pytensor.config.change_flags(floatX="float64"):
        built = build_rhime_co2_o2(
            prepared_inputs=prepared, independent_error_sd=_independent_error(prepared),
            bc_freq="monthly", bc_prior={channel: {"pdf": "normal", "mu": 1., "sigma": .1} for channel in boundaries},
        )
        point = built.model.initial_point()
    for channel in boundaries:
        assert point[f"{channel}_bc"].size == 4
        xr.testing.assert_identical(boundaries[channel], originals[channel])


@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("lazy_time", [False, True])
def test_indexed_boundary_time_conflict_is_rejected_without_computing_payload(centered, lazy_time):
    import numpy as np
    import pytensor
    from dask import delayed

    from openghg_inversions.rhime.co2.centered_boundary import bind_boundary_operands

    @delayed
    def forbidden_payload():
        pytest.fail("Boundary payload was computed during indexed-time validation")

    labels = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    authoritative = np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[D]")
    payload = xr.DataArray(
        da.from_delayed(forbidden_payload(), shape=(1, 2), dtype=float),
        dims=("bc_region", "time"), coords={"bc_region": ["north"], "time": labels}, attrs={"units": "ppm"},
    )
    observations = xr.DataArray([1., 2.], dims="nmeasure", coords={
        "nmeasure": labels,
        "time": ("nmeasure", da.from_array(authoritative, chunks=1) if lazy_time else authoritative),
    }, attrs={"units": "ppm"})
    with pytensor.config.change_flags(floatX="float64"), pytest.raises(ValueError, match="Indexed boundary time labels conflict"):
        if centered:
            bind_boundary_operands(
                xr.Dataset({"H_bc_anomaly": payload, "H_bc_mass": xr.ones_like(payload).assign_attrs(units="1"),
                            "bc_reference": xr.DataArray(400., attrs={"units": "ppm"})},
                           attrs={"boundary_representation": "centered-v1"}),
                observations, frequency="monthly",
            )
        else:
            bind_boundary_sensitivity(payload, observations, frequency="monthly")


@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("frequency", [None, "monthly"])
def test_public_linked_native_time_axis_builds_with_raw_boundary_operands(centered, frequency):
    """A native time axis can consume centered operands indexed by nmeasure."""
    import numpy as np
    import pytensor

    from openghg_inversions.rhime.co2 import prepare_co2_o2_inputs
    from openghg_inversions.rhime.co2.co2_o2_runner import build_rhime_co2_o2
    from test_centered_co2_boundaries import _operands
    from test_rhime_co2_o2 import _inputs, _independent_error

    inputs = _inputs()
    inputs["o2_units"] = "ppm"
    times = np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[D]")
    for name, value in inputs.items():
        if not isinstance(value, xr.DataArray):
            continue
        for old_dim, new_dim in (("co2_measure", "time"), ("co2_measure_cov", "time_cov")):
            if old_dim in value.dims:
                value = value.assign_coords({old_dim: times}).rename({old_dim: new_dim})
        inputs[name] = value
    operands = _operands(inputs["co2_observations"])
    assert "time" not in operands.coords
    if centered:
        inputs["boundary_operands"] = {"co2": operands}
    else:
        inputs["boundary_sensitivity"] = {
            "co2": (operands.H_bc_anomaly + operands.bc_reference * operands.H_bc_mass)
            .rename(nmeasure="time").transpose("time", "bc_region")
            .assign_attrs(units="ppm", boundary_sensitivity_preparation="raw-spatial"),
        }
    prepared = prepare_co2_o2_inputs(**inputs)
    original = operands.copy(deep=True)
    with pytensor.config.change_flags(floatX="float64"):
        built = build_rhime_co2_o2(
            prepared_inputs=prepared, independent_error_sd=_independent_error(prepared), bc_freq=frequency,
            bc_prior={"co2": {"pdf": "normal", "mu": 1., "sigma": .1}},
        )
        assert built.model.initial_point()["co2_bc"].size == (2 if frequency is None else 4)
    xr.testing.assert_identical(operands, original)


@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("frequency", [None, "monthly"])
@pytest.mark.parametrize("raw_dim", ["nmeasure", "time"])
@pytest.mark.parametrize("native_dim", ["nmeasure", "time"])
def test_native_datetime_dimension_restoration_preserves_rows_and_periods(centered, frequency, raw_dim, native_dim):
    import numpy as np
    import pytensor

    from openghg_inversions.rhime.co2.centered_boundary import bind_boundary_operands

    labels = np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[D]")
    anomaly = xr.DataArray(da.from_array([[1., 2.]], chunks=(1, 1)), dims=("bc_region", raw_dim),
                           coords={"bc_region": ["north"], raw_dim: labels}, attrs={"units": "ppm"})
    if raw_dim != "time":
        anomaly = anomaly.assign_coords(time=(raw_dim, np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")))
    operands = xr.Dataset({
        "H_bc_anomaly": anomaly, "H_bc_mass": xr.ones_like(anomaly).assign_attrs(units="1"),
        "bc_reference": xr.DataArray(400., attrs={"units": "ppm"}),
    }, attrs={"boundary_representation": "centered-v1"}).isel({raw_dim: [1, 0]})
    observations = xr.DataArray([3., 4.], dims=native_dim, coords={native_dim: labels}, attrs={"units": "ppm"})
    if native_dim != "time":
        observations = observations.assign_coords(time=(native_dim, labels))
    original = operands.copy(deep=True)
    with pytensor.config.change_flags(floatX="float64"), Callback(posttask=lambda *args: pytest.fail("Boundary payload computed during binding")):
        if centered:
            aligned = bind_boundary_operands(operands, observations, frequency=frequency)
            result = aligned.H_bc_anomaly + aligned.bc_reference * aligned.H_bc_mass
        else:
            raw = (operands.H_bc_anomaly + operands.bc_reference * operands.H_bc_mass).assign_attrs(
                units="ppm", boundary_sensitivity_preparation="raw-spatial",
            )
            result = bind_boundary_sensitivity(raw, observations, frequency=frequency)
    assert hasattr(result.data, "__dask_graph__")
    np.testing.assert_array_equal(result[native_dim], observations[native_dim])
    assert result.sizes["bc_region"] == (1 if frequency is None else 2)
    np.testing.assert_allclose(result.sum("bc_region"), [401., 402.])
    xr.testing.assert_identical(operands, original)
