"""Resolved temporal weights retain calendar meaning without changing their prior."""

from dataclasses import replace

import dask.array as da
from dask import delayed
import numpy as np
import pandas as pd
import pytensor
import pytest
import xarray as xr

from openghg_inversions.models.additive_boundary import prepare_additive_boundary
from openghg_inversions.models.boundary import bind_boundary_operands
from openghg_inversions.rhime.co2.co2_preparation import _independent_boundary_weights
from openghg_inversions.serialization import (
    decode_cf_multiindexes,
    encode_cf_multiindexes,
    save_datatree,
    open_datatree_loaded,
)


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def native(times=("2020-01-01", "2020-01-20", "2020-02-01", "2020-02-20")):
    rows = np.arange(len(times))
    observations = xr.DataArray(
        np.zeros(len(rows)),
        dims="nmeasure",
        coords={
            "nmeasure": rows,
            "time": ("nmeasure", pd.to_datetime(times)),
        },
        attrs={"units": "ppm"},
    )
    mass = xr.DataArray(
        np.tile([0.7, 0.3], (len(rows), 1)),
        dims=("nmeasure", "bc_region"),
        coords={"nmeasure": rows, "bc_region": ["north", "south"]},
        attrs={"units": "1"},
    )
    operands = xr.Dataset(
        {
            "H_bc_anomaly": (mass * 0.2).assign_attrs(units="ppm"),
            "H_bc_mass": mass,
            "bc_reference": xr.DataArray(400.0, attrs={"units": "ppm"}),
        },
        attrs={"boundary_representation": "centered-v1"},
    )
    return operands, observations


def weights(bound):
    # Full-state weights are deliberately nonseparable across curtain and period.
    return xr.DataArray(
        [1.0, 4.0, 9.0, 2.0],
        dims="bc_region",
        coords=bound.bc_region.coords,
        attrs={"units": "1"},
        name="bc_centering_weights",
    )


def prepare(bound, weight, observations):
    return prepare_additive_boundary(
        bound, weight, observations, {"pdf": "uniform", "lower": -0.5, "upper": 0.5}, 0.04, None, None
    )


def never_compute(array):
    @delayed
    def fail():
        raise AssertionError("Mismatch evaluated a borrowed numerical payload")

    return array.copy(data=da.from_delayed(fail(), shape=array.shape, dtype=array.dtype))


@pytest.mark.parametrize("mismatch", ["frequency", "anchor", "window", "missing_identity"])
def test_calendar_mismatch_rejected_before_mass_or_observation_compute(mismatch):
    raw, obs = native()
    if mismatch == "anchor":
        raw, obs = native(("2020-01-10", "2020-01-20", "2020-02-10", "2020-02-20"))
    original_frequency = "monthly" if mismatch != "anchor" else "45D"
    original = bind_boundary_operands(
        raw, obs, frequency=original_frequency, anchor_time="2020-01-01" if mismatch == "anchor" else None
    )
    weight = weights(original)
    frequency, anchor = original_frequency, None
    if mismatch == "frequency":
        frequency, anchor = "45D", "2020-01-01"
    elif mismatch == "anchor":
        anchor = "2020-01-02"
    elif mismatch == "window":
        obs = obs.assign_coords(time=("nmeasure", obs.time.values + np.timedelta64(31, "D")))
    else:
        weight.bc_region.attrs = {}
    bound = bind_boundary_operands(raw, obs, frequency=frequency, anchor_time=anchor)
    assert bound.indexes["bc_region"].equals(original.indexes["bc_region"])
    if mismatch == "frequency":
        assert not bound.H_bc_mass.equals(original.H_bc_mass)
    bound = bound.assign(
        H_bc_mass=never_compute(bound.H_bc_mass), H_bc_anomaly=never_compute(bound.H_bc_anomaly)
    )
    with pytest.raises(ValueError, match="period identity"):
        prepare(bound, weight, never_compute(obs))


def test_equivalent_duration_and_resolved_origin_preserve_nonseparable_prior():
    raw, obs = native()
    first = bind_boundary_operands(raw, obs, frequency="45D")
    second = bind_boundary_operands(raw, obs, frequency="1080h", anchor_time="2020-01-01")
    result = prepare(second, weights(first), obs)
    np.testing.assert_allclose(result.weights, np.array([1.0, 4.0, 9.0, 2.0]) / 16)
    np.testing.assert_allclose(result.weights @ result.basis, 0.0, atol=2e-16)
    expected_projection = np.eye(4) - np.ones((4, 1)) @ result.weights.values[None, :]
    np.testing.assert_allclose(
        result.basis.values @ result.basis.values.T, expected_projection @ expected_projection.T, atol=3e-16
    )


@pytest.mark.parametrize("extension", ["nc", "zarr"])
def test_independent_storage_axis_restores_calendar_identity_and_frozen_prior(tmp_path, extension):
    raw, obs = native()
    bound = bind_boundary_operands(raw, obs, frequency="monthly")
    expected = prepare(bound, weights(bound), obs)
    stored = _independent_boundary_weights(weights(bound)).to_dataset(name="bc_centering_weights")
    path = tmp_path / f"weights.{extension}"
    save_datatree(xr.DataTree(encode_cf_multiindexes(stored, "bc_centering_state")), path)
    loaded = open_datatree_loaded(path).to_dataset()
    restored = decode_cf_multiindexes(loaded, "bc_centering_state").bc_centering_weights.rename(
        {
            "bc_centering_state": "bc_region",
            "bc_centering_curtain": "bc_curtain",
            "bc_centering_period": "bc_period",
        }
    )
    assert restored.bc_region.attrs == weights(bound).bc_region.attrs
    actual = prepare(bound, restored, obs)
    xr.testing.assert_identical(actual.weights, expected.weights)
    np.testing.assert_array_equal(actual.basis, expected.basis)
    # Frozen reuse on a subset does not reselect temporal weights or their mean.
    subset = prepare(bound.isel(nmeasure=[0, 3]), restored, obs.isel(nmeasure=[0, 3]))
    xr.testing.assert_identical(subset.weights, expected.weights)
    np.testing.assert_array_equal(subset.basis, expected.basis)


def linked_inputs():
    from test_co2_o2_fixed_ou import _prepared

    initial = _prepared()
    observations = initial.observations
    times = observations.time.values
    operands, selected_weights = {}, {}
    for channel in ("co2", "o2"):
        positions = np.flatnonzero(observations.species.values == channel)
        raw, obs = native(times[positions])
        native_sensitivity = getattr(initial, f"{channel}_sensitivity")
        labels = native_sensitivity[native_sensitivity.dims[0]].values
        raw = raw.assign_coords(nmeasure=labels)
        obs = obs.assign_coords(nmeasure=labels)
        operands[channel] = raw
        bound = bind_boundary_operands(raw, obs, frequency="2h")
        selected_weights[channel] = xr.DataArray(
            np.arange(1, bound.sizes["bc_region"] + 1, dtype=float),
            dims="bc_region",
            coords=bound.bc_region.coords,
            attrs={"units": "1"},
            name="bc_centering_weights",
        )
    prepared = replace(
        initial, observations=observations, boundary_operands=operands, bc_centering_weights=selected_weights
    )
    return prepared


@pytest.mark.parametrize("extension", ["nc", "zarr"])
def test_linked_prepared_weights_preserve_period_identity(tmp_path, extension):
    from openghg_inversions.rhime.co2.co2_o2_preparation import Co2O2PreparedInputs

    prepared = linked_inputs()
    selected_weights = prepared.bc_centering_weights
    path = tmp_path / f"linked.{extension}"
    prepared.save(path)
    loaded = Co2O2PreparedInputs.load(path)
    for channel in ("co2", "o2"):
        xr.testing.assert_identical(loaded.bc_centering_weights[channel], selected_weights[channel])


def test_metadata_free_whole_window_spatial_weights_remain_valid():
    raw, obs = native()
    bound = bind_boundary_operands(raw, obs)
    spatial = xr.DataArray(
        [1.0, 3.0], dims="bc_region", coords={"bc_region": ["north", "south"]}, attrs={"units": "1"}
    )
    result = prepare(bound, spatial, obs)
    np.testing.assert_allclose(result.weights, [0.25, 0.75])


@pytest.mark.parametrize("cached", [False, True])
def test_single_direct_builders_reject_period_mismatch_before_payloads(cached):
    from test_co2_cached_sigma import _inputs, _build
    from openghg_inversions.correlated_state import CorrelatedLognormalPrior
    from openghg_inversions.rhime.co2.co2_model import build_co2_model
    from openghg_inversions.observation_error import resolve_aggregation_error

    raw, obs = native()
    weight = weights(bind_boundary_operands(raw, obs, frequency="monthly"))
    raw = raw.assign(H_bc_mass=never_compute(raw.H_bc_mass))
    inputs = _inputs()
    inputs["mf"] = never_compute(obs)
    options = dict(
        boundary_operands=raw,
        bc_centering_weights=weight,
        bc_freq="45D",
        bc_anchor_time="2020-01-01",
        bc_mean_shift_prior={"pdf": "normal", "mu": 0.0, "sigma": 0.1},
        bc_anomaly_scale=0.04,
    )
    with pytest.raises(ValueError, match="period identity"):
        if cached:
            _build(inputs=inputs, **options)
        else:
            build_co2_model(
                inputs.H,
                retained_prior=CorrelatedLognormalPrior(
                    inputs.alpha_prior_mean, inputs.alpha_prior_covariance, covariance_dim="region_cov"
                ),
                fixed_prior_contribution=inputs.fixed_prior_contribution,
                observations=inputs.mf,
                observation_error=inputs.mf_error,
                aggregation_error=resolve_aggregation_error(inputs, "dense"),
                **options,
            )


@pytest.mark.parametrize("cached", [False, True])
def test_linked_direct_builders_reject_period_mismatch_before_payloads(cached):
    from test_co2_o2_fixed_ou import _kwargs
    from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
    from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model

    prepared = linked_inputs()
    kwargs = _kwargs(prepared)
    kwargs["observations"] = never_compute(kwargs["observations"])
    operands = {
        channel: raw.assign(H_bc_mass=never_compute(raw.H_bc_mass))
        for channel, raw in prepared.boundary_operands.items()
    }
    kwargs.update(
        boundary_operands=operands,
        bc_centering_weights=prepared.bc_centering_weights,
        bc_mean_shift_prior={channel: {"pdf": "normal", "mu": 0.0, "sigma": 0.1} for channel in operands},
        bc_anomaly_scale={channel: 0.04 for channel in operands},
        bc_freq="3h",
    )
    with pytest.raises(ValueError, match="period identity"):
        if cached:
            build_co2_o2_cached_sigma_model(
                **kwargs, tau_hours=4.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.1}
            )
        else:
            build_co2_o2_model(**kwargs)


@pytest.mark.parametrize("cached", [False, True])
def test_single_public_materialization_rejects_weights_before_payloads(cached):
    from types import SimpleNamespace
    from test_co2_cached_sigma import _inputs
    from openghg_inversions.rhime.co2.co2_runner import materialize_co2_inputs

    raw, obs = native()
    weight = weights(bind_boundary_operands(raw, obs, frequency="monthly"))
    inputs = _inputs().assign(raw.data_vars).assign_coords(time=obs.time)
    inputs.attrs.update(raw.attrs)
    inputs["mf"] = never_compute(obs)
    inputs["H_bc_mass"] = never_compute(inputs.H_bc_mass)
    inputs["bc_centering_weights"] = _independent_boundary_weights(weight)
    prepared = SimpleNamespace(inv_inputs=inputs, rhime_inputs=None)
    from openghg_inversions.rhime.co2.co2_cached_sigma_runner import build_rhime_co2_cached_sigma

    options = dict(
        prepared_inputs=prepared,
        use_bc=True,
        bc_mean_shift_prior={"pdf": "normal", "mu": 0.0, "sigma": 0.1},
        bc_anomaly_scale=0.04,
        bc_freq="45D",
        bc_anchor_time="2020-01-01",
    )
    with pytest.raises(ValueError, match="period identity"):
        if cached:
            build_rhime_co2_cached_sigma(
                **options, tau_hours=4.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.1}
            )
        else:
            materialize_co2_inputs(**options)
