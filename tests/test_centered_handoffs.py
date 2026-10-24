"""Configured saved handoffs bind raw centered boundaries to model periods."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytensor
import pytest
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.rhime.co2 import Co2ModelConfig, Co2O2ModelConfig
from openghg_inversions.rhime.co2 import handoffs, co2_o2_handoffs
from openghg_inversions.serialization import load_trace
from test_centered_co2_boundaries import _centered_prepared, _operands
from test_rhime_co2_o2_baselines import _prepared as _linked
from test_rhime_co2_o2 import _independent_error


@pytest.fixture(autouse=True)
def double_precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _ordinary():
    prepared = _centered_prepared()
    inputs = prepared.inv_inputs.reset_index("nmeasure").isel(nmeasure=[0, 0, 0])
    inputs = inputs.assign_coords(time=("nmeasure", pd.date_range("2020-01-01", periods=3)))
    inputs = inputs.set_index(nmeasure=["site", "time"])
    return replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))


def _joint():
    prepared = _linked()
    operands = {}
    for channel in ("co2", "o2"):
        native = select_gathered_data_array(
            prepared.observations, key=channel, key_dim="species",
            ragged_dim="channel_observation", stack_dim="observation",
        )
        operands[channel] = _operands(native)
    return replace(prepared, boundary_operands=operands, independent_error_sd=_independent_error(prepared))


def _config(linked, **periods):
    boundary = {"enabled": True, "prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.01}, **periods}
    if linked:
        return {"channels": {channel: {"units": "ppm", "boundary": dict(boundary)} for channel in ("co2", "o2")}}
    return {"model": {"boundary": boundary}, "likelihood": {"kind": "additive_sigma", "no_model_error": True}}


@pytest.mark.parametrize("linked", [False, True])
def test_saved_centered_periods_prior_predictive_save_reopen(tmp_path, linked):
    module = co2_o2_handoffs if linked else handoffs
    prepared = _joint() if linked else _ordinary()
    anchor = "1970-01-01T12:00:00" if linked else "2020-01-01T12:00:00"
    config = _config(linked, frequency="2D", anchor_time=anchor)
    manifest = module.save_preparation(
        prepared, output_dir=tmp_path / "prepared", domain="EUROPE",
        start_date="1969-12-01" if linked else "2019-12-01", end_date="2020-02-01",
    )
    result = module.prior_predictive(manifest, model=config, output_dir=tmp_path / "prior", draws=3, random_seed=13)
    assert result["status"] == "pass"
    trace = load_trace(tmp_path / "prior" / "prior-predictive.nc")
    metadata = json.loads(trace.attrs["rhime_model_metadata"])
    assert metadata["bc_frequency"] == "2D"
    assert metadata["bc_anchor_time"] == anchor
    observed = prepared.observations if linked else prepared.inv_inputs.mf
    xr.testing.assert_allclose(trace.observed_data.y, observed)
    assert trace.prior_predictive.y.attrs["concentration_representation"] == "absolute-v1"
    # Prediction noise is small on the absolute concentration scale, even when q is hundreds of ppm.
    np.testing.assert_allclose(trace.prior_predictive.y, trace.prior.modelled_concentration, atol=5, rtol=0)
    channels = ("co2", "o2") if linked else (None,)
    for channel in channels:
        operands = prepared.boundary_operands[channel] if linked else prepared.inv_inputs
        prefix = f"{channel}_" if linked else ""
        scale = trace.prior[f"{prefix}bc"]
        dim = str(scale.dims[-1])
        assert scale.sizes[dim] == 4
        assert scale[f"{prefix}bc_period"].values.tolist() == [0, 1, 0, 1]
        assert scale[f"{prefix}bc_curtain"].values.tolist() == ["north", "north", "south", "south"]
        # Midnight samples straddle the noon anchor: first period, then second.
        absolute = (operands.H_bc_anomaly + operands.bc_reference * operands.H_bc_mass).values
        periods = np.array([0, 1, 1])[:absolute.shape[-1]]
        selected_scales = scale.values.reshape(*scale.shape[:-1], 2, 2)[..., periods]
        expected = (selected_scales * absolute).sum(axis=-2)
        actual = trace.prior[f"{prefix}mu_bc"].values
        if linked:
            selected = prepared.observations.species.values == channel
            np.testing.assert_allclose(actual[..., selected], expected)
            np.testing.assert_allclose(actual[..., ~selected], 0)
        else:
            np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("linked", [False, True])
def test_absolute_boundary_defaults_compatible_but_period_override_rejected(linked):
    if linked:
        prepared = _linked(("co2", "o2"))
        prepared = replace(prepared, independent_error_sd=_independent_error(prepared))
        owner = Co2O2ModelConfig
    else:
        prepared = _ordinary()
        inputs = prepared.inv_inputs.drop_vars(["H_bc_anomaly", "H_bc_mass", "bc_reference"])
        inputs["H_bc"] = prepared.inv_inputs.H_bc_anomaly.variable
        inputs.attrs.pop("boundary_representation", None)
        prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))
        owner = Co2ModelConfig
    assert "bc_freq" not in owner.from_params(_config(linked)).bind(prepared)
    with pytest.raises(ValueError, match="legacy absolute H_bc"):
        owner.from_params(_config(linked, frequency="2D")).bind(prepared)


@pytest.mark.parametrize("linked", [False, True])
@pytest.mark.parametrize("periods", [
    {"frequency": "-1D"}, {"frequency": "not-a-period"}, {"frequency": 2},
    {"anchor_time": "2020-01-01"}, {"frequency": "monthly", "anchor_time": "2020-01-01"},
    {"frequency": "2D", "anchor_time": "NaT"}, {"frequency": "2D", "anchor_time": "invalid"},
])
def test_invalid_period_choices_rejected_at_configuration_owner(linked, periods):
    owner = Co2O2ModelConfig if linked else Co2ModelConfig
    with pytest.raises((ValueError, TypeError)):
        owner.from_params(_config(linked, **periods))


def test_linked_periods_must_agree_and_forward_concrete_names():
    config = _config(True, frequency="2D", anchor_time="1970-01-01T12:00:00")
    config["channels"]["o2"]["boundary"]["frequency"] = "48h"
    bound = Co2O2ModelConfig.from_params(config).bind(_joint())
    assert bound["bc_freq"] == "48h"
    assert bound["bc_anchor_time"] == "1970-01-01T12:00:00"
    config["channels"]["o2"]["boundary"]["frequency"] = "3D"
    with pytest.raises(ValueError, match="must agree"):
        Co2O2ModelConfig.from_params(config)


@pytest.mark.parametrize("route", ["full", "saved", "linked_saved"])
def test_sampling_routes_forward_bound_period_choices(tmp_path, monkeypatch, route):
    from openghg_inversions.rhime.co2 import execution

    linked = route == "linked_saved"
    module = co2_o2_handoffs if linked else handoffs
    prepared = _joint() if linked else _ordinary()
    anchor = "1970-01-01T12:00:00" if linked else "2020-01-01T12:00:00"
    config = _config(linked, frequency="2D", anchor_time=anchor)
    facts = {"start_date": "1969-12-01", "end_date": "2020-02-01", "domain": "EUROPE"}

    class ReachedSampling(Exception):
        pass

    def sample(incoming=None, choices=None, sampler=None, **kwargs):
        options = kwargs if linked else choices
        assert options["bc_freq"] == "2D"
        assert options["bc_anchor_time"] == anchor
        raise ReachedSampling

    if route == "full":
        monkeypatch.setattr(execution, "sample_co2_inference", sample)
        config.update(prepared_inputs=facts, outputs={"output_format": "none"})
        with pytest.raises(ReachedSampling):
            execution.run_co2(params=config, prepared=prepared)
    else:
        manifest = module.save_preparation(prepared, output_dir=tmp_path / "prepared", **facts)
        monkeypatch.setattr(module, "build_rhime_co2_o2" if linked else "sample_co2_inference", sample)
        with pytest.raises(ReachedSampling):
            module.sample(manifest, model=config, output_dir=tmp_path / "samples")
    assert not (tmp_path / "samples").exists()


@pytest.mark.parametrize("linked", [False, True])
def test_periods_require_enabled_boundary(linked):
    config = _config(linked, frequency="2D")
    boundary = config["channels"]["co2"]["boundary"] if linked else config["model"]["boundary"]
    boundary["enabled"] = False
    boundary.pop("prior")
    owner = Co2O2ModelConfig if linked else Co2ModelConfig
    with pytest.raises(ValueError, match="enabled=true"):
        owner.from_params(config)


def test_linked_period_override_rejects_active_absolute_channel():
    prepared = _joint()
    absolute = _linked(("o2",)).boundary_sensitivity
    prepared = replace(prepared, boundary_operands={"co2": prepared.boundary_operands["co2"]}, boundary_sensitivity=absolute)
    config = _config(True, frequency="2D")
    with pytest.raises(ValueError, match="legacy absolute H_bc"):
        Co2O2ModelConfig.from_params(config).bind(prepared)
    config["channels"]["o2"]["boundary"] = {"enabled": False}
    assert Co2O2ModelConfig.from_params(config).bind(prepared)["bc_freq"] == "2D"


@pytest.mark.parametrize("linked,cached", [(False, False), (False, True), (True, False)])
def test_saved_centered_periods_reject_fixed_ou(tmp_path, linked, cached):
    module = co2_o2_handoffs if linked else handoffs
    prepared = _joint() if linked else _ordinary()
    config = _config(linked, frequency="2D")
    amplitude = {"site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.1}} if cached else {"fixed_site_amplitudes": 0.1}
    config["likelihood"] = {"kind": "fixed_ou", "tau_hours": 2.0, **amplitude}
    manifest = module.save_preparation(
        prepared, output_dir=tmp_path / "prepared", domain="EUROPE",
        start_date="1969-12-01", end_date="2020-02-01",
    )
    with pytest.raises(ValueError, match="fixed-OU"):
        module.prior_predictive(manifest, model=config, output_dir=tmp_path / "prior", draws=2)
    assert not (tmp_path / "prior").exists()
