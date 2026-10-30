"""Public additive BC reconstruction preserves the uncertainty budget on replay."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.rhime.co2.co2_outputs import reconstruct_co2_concentrations
from openghg_inversions.rhime.co2.co2_preparation import Co2PreparedInputs
from openghg_inversions.rhime.co2.configuration import Co2ModelConfig
from openghg_inversions.rhime.co2.inference import annotate_co2_inference, build_co2_inference
from openghg_inversions.serialization import load_trace, save_trace
from test_co2_affine_output import _prepared


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("anomaly_scale", [0.0, 0.08])
def test_additive_public_build_save_reload_and_reconstruction(tmp_path, cached, centered, anomaly_scale):
    prepared, _ = _prepared()
    inputs = prepared.inv_inputs.copy(deep=False)
    response = xr.DataArray(
        [[0.6, 0.3, 0.0]], dims=("nmeasure", "bc_region"),
        coords={**inputs.mf.coords, "bc_region": ["north", "south", "east"]},
        attrs={"units": "1"},
    )
    weight_coords = {"bc_centering_state": response.bc_region.values}
    if centered:
        weight_coords = xr.Coordinates.from_pandas_multiindex(
            pd.MultiIndex.from_product(
                [response.bc_region.values, [0]], names=["bc_curtain", "bc_period"],
            ), "bc_centering_state",
        )
    inputs["bc_centering_weights"] = xr.DataArray(
        [2.0, 1.0, 3.0], dims="bc_centering_state",
        coords=weight_coords, attrs={"units": "1"},
    )
    if centered:
        inputs["H_bc_mass"] = response
        inputs["H_bc_anomaly"] = (response * np.array([0.2, -0.1, 0.0])).assign_attrs(units="ppm")
        inputs["bc_reference"] = xr.DataArray(400.0, attrs={"units": "ppm"})
        inputs.attrs["boundary_representation"] = "centered-v1"
    else:
        inputs["G_bc"] = response
        inputs["H_bc"] = (response * np.array([400.2, 399.9, 400.0])).assign_attrs(units="ppm")
    prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))
    payload = tmp_path / "prepared.nc"
    prepared.save(payload)
    restored_prepared = Co2PreparedInputs.load(payload)
    np.testing.assert_allclose(restored_prepared.inv_inputs.bc_centering_weights, [2, 1, 3])
    config = {
        "model": {"boundary": {"enabled": True, "mean_shift_prior": {
            "pdf": "normal", "mu": 0.0, "sigma": 0.15,
        }, "anomaly_scale": anomaly_scale}},
        "likelihood": {"kind": "additive_sigma", "no_model_error": True},
    }
    if cached:
        config["likelihood"] = {"kind": "fixed_ou", "tau_hours": 4.0,
                                "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.2}}
    options = Co2ModelConfig.from_params(config).model_arguments()
    built, cache = build_co2_inference(restored_prepared, options)
    assert "bc_mean_shift" in built.model
    assert ("bc_anomaly_latent" in built.model) == (anomaly_scale > 0)
    if cache is not None:
        assert ("bc_anomaly_latent" in {state.name for state in cache.states}) == (anomaly_scale > 0)
    with built.model:
        trace = pm.sample_prior_predictive(draws=8, random_seed=745)
    registry = get_coord_registry(built.model)
    trace = restore_inferencedata_coords(trace, registry)
    trace = annotate_co2_inference(trace, built, restored_prepared, options, cache)
    # Treat fixed prior draws as a small saved posterior for reconstruction;
    # this checks the I/O contract without a convergence-dependent test.
    trace["posterior"] = trace.prior.to_dataset().isel(draw=slice(0, 3))
    path = tmp_path / "trace.nc"
    save_trace(trace, path)
    replay = load_trace(path)
    components = reconstruct_co2_concentrations(replay, restored_prepared)
    roles = json.loads(replay.attrs["rhime_variable_roles"])
    assert "boundary_scale" not in roles
    assert replay.constant_data.bc_anomaly_scale.item() == pytest.approx(anomaly_scale)
    assert replay.constant_data.bc_anomaly_scale.attrs["units"] == "ppm"
    if anomaly_scale == 0:
        np.testing.assert_array_equal(replay.prior.bc_anomaly, 0.0)
        np.testing.assert_array_equal(replay.posterior.bc_anomaly, 0.0)
    assert np.abs(replay.prior_predictive.y - replay.prior.modelled_concentration).max() < 5.0
    for when in ("prior", "posterior"):
        np.testing.assert_allclose(
            components[f"boundary_{when}"],
            components[f"boundary_reference_{when}"]
            + components[f"boundary_mean_contribution_{when}"]
            + components[f"boundary_anomaly_contribution_{when}"],
        )
        np.testing.assert_allclose(
            components[f"modelled_{when}"],
            components[f"pollution_{when}"] + components[f"boundary_{when}"],
        )
        assert components[f"boundary_mean_shift_{when}"].attrs["units"] == "ppm"
    xr.testing.assert_identical(
        replay.prior_predictive.y, trace.prior_predictive.y,
    )
    assert replay.prior_predictive.y.attrs["units"] == "ppm"
    assert components.sizes["bc_region"] == 3
    assert components.sizes["prior_draw"] == 8
    assert components.sizes["draw"] == 3
