"""Tracer-local boundary and offset equations on unequal observation axes."""

import json

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.models import StateActivity, restore_inferencedata_coords, get_coord_registry
from openghg_inversions.rhime.co2 import (
    build_co2_o2_model,
    prepare_co2_o2_inputs,
    resolve_co2_family_config,
    run_rhime_co2_o2_from_prepared_inputs,
)
from openghg_inversions.rhime.co2 import co2_o2_runner
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.serialization import save_trace, load_trace
from test_rhime_co2_o2 import _inputs, _independent_error


def _prepared(channels=()):
    inputs = _inputs()
    inputs["o2_units"] = "ppm"
    for channel in ("co2", "o2"):
        obs = inputs[f"{channel}_observations"]
        inputs[f"{channel}_observations"] = obs.assign_coords(
            site=(obs.dims[0], [f"{channel}-{i % 2}" for i in range(obs.size)]),
            time=(obs.dims[0], np.arange(obs.size).astype("datetime64[D]")),
        )
    inputs["boundary_sensitivity"] = {
        channel: xr.DataArray(
            np.arange(1, inputs[f"{channel}_observations"].size * 2 + 1).reshape(-1, 2),
            dims=(f"{channel}_measure", "boundary"),
            coords={
                f"{channel}_measure": inputs[f"{channel}_observations"][f"{channel}_measure"],
                "boundary": ["north", "south"],
                "basis_group": ("boundary", [f"{channel}-active", f"{channel}-fixed"]),
            },
        )
        for channel in channels
    }
    return prepare_co2_o2_inputs(**inputs)


def _model(prepared, **kwargs):
    return build_co2_o2_model(
        observations=prepared.observations,
        fixed_prior_contribution=prepared.fixed_prior_contribution,
        co2_sensitivity=prepared.co2_sensitivity,
        o2_sensitivity=prepared.o2_sensitivity,
        aggregation_error=prepared.aggregation_error,
        retained_prior=prepared.retained_prior,
        independent_error_sd=_independent_error(prepared),
        boundary_sensitivity=prepared.boundary_sensitivity,
        state_activity=StateActivity(active=False),
        **kwargs,
    )


@pytest.mark.parametrize("channels", [("co2",), ("o2",), ("co2", "o2")])
def test_boundary_fixed_and_active_state_reconstructs_exact_joint_sum(channels):
    prepared = _prepared(channels)
    activities = {
        channel: StateActivity(
            fixed_groups=(f"{channel}-fixed",),
            active=xr.DataArray([True, True], dims="boundary", coords={"boundary": ["north", "south"]}),
            fixed_value=xr.DataArray([1.0, 2.0], dims="boundary", coords={"boundary": ["north", "south"]}),
        )
        for channel in channels
    }
    model = _model(prepared, bc_state_activity=activities)
    with model:
        trace = pm.sample_prior_predictive(draws=3, random_seed=42)
    trace = restore_inferencedata_coords(trace, get_coord_registry(model))
    expected = np.zeros((1, 3, 5))
    for channel in channels:
        scale = trace.prior[f"{channel}_bc"].values
        np.testing.assert_array_equal(scale[..., 1], 2.0)
        contribution = scale @ prepared.boundary_sensitivity[channel].values.T
        selected = slice(0, 2) if channel == "co2" else slice(2, 5)
        expected[..., selected] += contribution
        padded = np.zeros_like(expected)
        padded[..., selected] = contribution
        np.testing.assert_allclose(trace.prior[f"{channel}_mu_bc"], padded)
        assert trace.prior[f"{channel}_bc"][f"{channel}_boundary"].values.tolist() == ["north", "south"]
    np.testing.assert_allclose(trace.prior.baseline_concentration, expected)
    flux = trace.prior.co2_o2_flux_contribution + trace.constant_data.fixed_prior_contribution
    np.testing.assert_allclose(trace.prior.modelled_concentration, flux + expected)
    np.testing.assert_allclose(
        trace.constant_data.aggregation_error_covariance, prepared.aggregation_error.covariance
    )


@pytest.mark.parametrize("per_site,frequency", [(False, None), (True, None), (True, "1D")])
def test_independent_offsets_and_boundary_reconstruct_from_draws(per_site, frequency, tmp_path):
    prepared = _prepared(("co2", "o2"))
    model = _model(
        prepared,
        bc_state_activity={
            channel: StateActivity(active=False, fixed_value=1.5) for channel in ("co2", "o2")
        },
        offset_prior={channel: {"pdf": "normal", "mu": 0.0, "sigma": 2.0} for channel in ("co2", "o2")},
        offset_args={channel: {"per_site": per_site, "offset_freq": frequency} for channel in ("co2", "o2")},
    )
    with model:
        trace = pm.sample_prior_predictive(draws=3, random_seed=7)
    trace = restore_inferencedata_coords(trace, get_coord_registry(model))
    path = tmp_path / "baselines.nc"
    save_trace(trace, path)
    trace = load_trace(path)
    for channel in ("co2", "o2"):
        opposite = slice(2, 5) if channel == "co2" else slice(0, 2)
        np.testing.assert_array_equal(trace.prior[f"{channel}_offset"][..., opposite], 0.0)
        coefficients = trace.prior[f"{channel}_offset_latent"].values
        if not per_site:
            coefficients = coefficients[..., None]
        predicted = coefficients @ trace.constant_data[f"{channel}_offset_design"].values.T
        np.testing.assert_allclose(trace.prior[f"{channel}_offset"], predicted)
    np.testing.assert_allclose(
        trace.prior.baseline_concentration,
        trace.prior.boundary_concentration + trace.prior.offset_concentration,
    )
    np.testing.assert_allclose(
        trace.prior.modelled_concentration,
        trace.prior.co2_o2_flux_contribution
        + trace.constant_data.fixed_prior_contribution
        + trace.prior.baseline_concentration,
    )


def test_configuration_and_runner_forward_same_channel_options(monkeypatch):
    prepared = _prepared(("co2",))
    setup = resolve_co2_family_config(
        {
            "format_version": 1,
            "recipe": "co2_o2",
            "variant": "linked",
            "channels": {
                "co2": {
                    "units": "ppm",
                    "independent_error_sd": 0.1,
                    "boundary": {
                        "enabled": True,
                        "activity": {"active": False, "fixed_value": 1.25},
                        "prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.2},
                    },
                },
                "o2": {
                    "units": "ppm",
                    "independent_error_sd": 0.1,
                    "offset": {"per_site": False, "prior": {"pdf": "normal", "mu": 0.0, "sigma": 2.0}},
                },
            },
        }
    )
    configured = dict(setup.runner_arguments(prepared))
    configured["sampler"] = RhimeSampler(draws=2, tune=0, chains=1, burn=0, nuts_sampler="pymc")

    def prior_sample(built, sampler):
        with built.model:
            trace = pm.sample_prior_predictive(draws=2, random_seed=8)
        return restore_inferencedata_coords(trace, get_coord_registry(built.model))

    monkeypatch.setattr(co2_o2_runner, "sample_rhime_model", prior_sample)
    trace = run_rhime_co2_o2_from_prepared_inputs(**configured)
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    assert roles["co2_boundary_concentration"] == "co2_mu_bc"
    assert roles["o2_offset_concentration"] == "o2_offset"
    np.testing.assert_array_equal(trace.prior.co2_bc, 1.25)
    assert trace.prior.co2_bc.attrs["units"] == "dimensionless boundary scale"
    assert trace.prior.co2_mu_bc.attrs["tracer"] == "co2"
    assert trace.prior.o2_offset.attrs["units"] == "ppm"


@pytest.mark.parametrize(
    "options,match",
    [
        ({"bc_prior": {"o2": {}}}, "require boundary_sensitivity"),
        ({"offset_args": {"co2": {"per_site": False}}}, "require offset_prior"),
        ({"offset_prior": {"co": {}}}, "keyed only"),
    ],
)
def test_unused_or_unknown_channel_options_fail(options, match):
    with pytest.raises(ValueError, match=match):
        _model(_prepared(), **options)


def _native_multiindex_inputs(*, extra_level=False, co2_dim="co2_measure"):
    inputs = _inputs()
    inputs["o2_units"] = "ppm"
    for channel in ("co2", "o2"):
        dim = f"{channel}_measure"
        count = inputs[f"{channel}_observations"].size
        index = pd.MultiIndex.from_arrays(
            [[f"{channel}-{i % 2}" for i in range(count)], pd.date_range("2024-01-01", periods=count)],
            names=("site", "time"),
        )
        if extra_level:
            index = pd.MultiIndex.from_arrays(
                [index.get_level_values("site"), index.get_level_values("time"), ["10m"] * count],
                names=("site", "time", "inlet"),
            )
        for name, array in inputs.items():
            if isinstance(array, xr.DataArray) and dim in array.dims:
                inputs[name] = array.drop_vars(dim).assign_coords(
                    xr.Coordinates.from_pandas_multiindex(
                        index.set_names([f"{level}_o2" for level in index.names])
                        if name == "co2_o2_aggregation_covariance" and channel == "o2" else index, dim
                    )
                )
        covariance = inputs[f"{channel}_aggregation_covariance"]
        covariance_dim = f"{dim}_cov"
        inputs[f"{channel}_aggregation_covariance"] = covariance.drop_vars(covariance_dim).assign_coords(
            xr.Coordinates.from_pandas_multiindex(index.set_names([f"{level}_cov" for level in index.names]), covariance_dim)
        )
    if co2_dim != "co2_measure":
        for name, array in inputs.items():
            if isinstance(array, xr.DataArray):
                inputs[name] = array.rename({
                    dim: replacement
                    for dim, replacement in (("co2_measure", co2_dim), ("co2_measure_cov", f"{co2_dim}_cov"))
                    if dim in array.dims
                })
    return inputs


@pytest.mark.parametrize("per_site,frequency", [(False, None), (True, None), (True, "1D")])
@pytest.mark.parametrize("extra_level,co2_dim", [(False, "co2_measure"), (True, "co2_measure"), (False, "observation")])
def test_offsets_restore_native_site_time_multiindexes(per_site, frequency, extra_level, co2_dim):
    prepared = prepare_co2_o2_inputs(**_native_multiindex_inputs(extra_level=extra_level, co2_dim=co2_dim))
    model = _model(
        prepared,
        offset_prior={channel: {"pdf": "normal", "mu": 0.0, "sigma": 2.0} for channel in ("co2", "o2")},
        offset_args={channel: {"per_site": per_site, "offset_freq": frequency} for channel in ("co2", "o2")},
    )
    with model:
        trace = pm.sample_prior_predictive(draws=3, random_seed=12)
    trace = restore_inferencedata_coords(trace, get_coord_registry(model))
    assert trace.prior.indexes["observation"].equals(prepared.observations.indexes["observation"])
    if extra_level:
        assert trace.prior.inlet.values.tolist() == ["10m"] * 5
    for channel in ("co2", "o2"):
        design = trace.constant_data[f"{channel}_offset_design"]
        coefficients = trace.prior[f"{channel}_offset_latent"].values
        if not per_site:
            coefficients = coefficients[..., None]
        np.testing.assert_allclose(trace.prior[f"{channel}_offset"], coefficients @ design.values.T)
        other_rows = prepared.observations.species.values != channel
        np.testing.assert_array_equal(trace.prior[f"{channel}_offset"][..., other_rows], 0)
    xr.testing.assert_allclose(
        trace.prior.modelled_concentration,
        trace.prior.co2_o2_flux_contribution + trace.constant_data.fixed_prior_contribution + trace.prior.baseline_concentration,
    )
