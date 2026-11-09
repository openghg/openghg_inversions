"""Saved handoffs bind raw ordinary and centered boundaries to model periods."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytensor
import pytest
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.inversion_inputs import normalise_frequency
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



def _raw_ordinary(linked):
    prepared = _joint() if linked else _ordinary()
    if linked:
        boundaries = {
            channel: (operands.H_bc_anomaly + operands.bc_reference * operands.H_bc_mass).rename(
                nmeasure=str((prepared.co2_sensitivity if channel == "co2" else prepared.o2_sensitivity).dims[0])
            ).transpose(..., "bc_region").assign_attrs(
                units="ppm", boundary_sensitivity_preparation="raw-spatial"
            )
            for channel, operands in prepared.boundary_operands.items()
        }
        return replace(prepared, boundary_operands={}, boundary_sensitivity=boundaries)
    inputs = prepared.inv_inputs
    absolute = (inputs.H_bc_anomaly + inputs.bc_reference * inputs.H_bc_mass).assign_attrs(
        units="ppm", boundary_sensitivity_preparation="raw-spatial"
    )
    inputs = inputs.drop_vars(["H_bc_anomaly", "H_bc_mass", "bc_reference"]).assign(H_bc=absolute)
    inputs.attrs.pop("boundary_representation", None)
    return replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))


def _frequencies(options, linked):
    """Return the period frequency of each bound boundary choice."""
    choices = options["boundary"] if linked else {None: options.get("boundary")}
    return {channel: choice.frequency for channel, choice in choices.items()}


def _config(linked, **periods):
    boundary = {"enabled": True, "prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.01}, **periods}
    if linked:
        return {"channels": {channel: {"units": "ppm", "boundary": dict(boundary)} for channel in ("co2", "o2")}}
    return {"model": {"boundary": boundary}, "likelihood": {"kind": "additive_sigma", "no_model_error": True}}


@pytest.mark.parametrize("linked", [False, True])
@pytest.mark.parametrize("centered", [False, True])
def test_saved_raw_periods_prior_predictive_save_reopen(tmp_path, linked, centered):
    module = co2_o2_handoffs if linked else handoffs
    prepared = (_joint() if linked else _ordinary()) if centered else _raw_ordinary(linked)
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
    facts = metadata["boundary_metadata"].values() if linked else [metadata]
    for channel_facts in facts:
        assert channel_facts["bc_frequency"] == "2D"
        assert channel_facts["bc_anchor_time"] == anchor
        assert ("boundary_representation" in channel_facts) == centered
    observed = prepared.observations if linked else prepared.inv_inputs.mf
    xr.testing.assert_allclose(trace.observed_data.y, observed)
    if centered:
        assert trace.prior_predictive.y.attrs["concentration_representation"] == "absolute-v1"
    # atol=5 checks restoration of the large reference, allowing predictive noise.
    # Tight density/full-gradient precision is tested in test_centered_fixed_ou.
    np.testing.assert_allclose(trace.prior_predictive.y, trace.prior.modelled_concentration, atol=5, rtol=0)
    channels = ("co2", "o2") if linked else (None,)
    for channel in channels:
        prefix = f"{channel}_" if linked else ""
        scale = trace.prior[f"{prefix}bc"]
        dim = str(scale.dims[-1])
        assert scale.sizes[dim] == 4
        times = observed.time.sel(species=channel).values if linked else observed.time.values
        duration = np.timedelta64(2, "D")
        first = np.datetime64(anchor) + ((times.min() - np.datetime64(anchor)) // duration) * duration
        np.testing.assert_array_equal(scale[f"{prefix}bc_period"], np.tile([first, first + duration], 2))
        assert scale[f"{prefix}bc_curtain"].values.tolist() == ["north", "north", "south", "south"]
        # Midnight samples straddle the noon anchor: first period, then second.
        if centered:
            operands = prepared.boundary_operands[channel] if linked else prepared.inv_inputs
            absolute = (operands.H_bc_anomaly + operands.bc_reference * operands.H_bc_mass).values
        else:
            raw = prepared.boundary_sensitivity[channel] if linked else prepared.inv_inputs.H_bc
            absolute = raw.transpose("bc_region", ...).values
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

    # Reuse the same saved preparation with a single model-owned period.
    # This constructs another graph without multiplying posterior sampling.
    reopened = module.load_preparation(manifest).prepared
    owner = Co2O2ModelConfig if linked else Co2ModelConfig
    choices = owner.from_params(_config(linked))
    options = choices.bind(reopened) if linked else choices.model_arguments()
    if linked:
        from openghg_inversions.rhime.co2.co2_o2_runner import build_rhime_co2_o2
        rebuilt = build_rhime_co2_o2(prepared_inputs=reopened, **options)
    else:
        from openghg_inversions.rhime.co2.co2_runner import materialize_co2_inputs, build_co2_from_materialized
        inputs, priors, boundary = materialize_co2_inputs(prepared_inputs=reopened, **options)
        options = {name: value for name, value in options.items() if name != "boundary"}
        rebuilt = build_co2_from_materialized(inputs, boundary=boundary, prepared_inputs=reopened, priors=priors, **options)
        if centered:
            # The model builder owns the centered-likelihood rule; the runner defers to it.
            options = {name: value for name, value in options.items() if name != "no_model_error"}
            options["likelihood_builder"] = lambda *args, **kwargs: None
            with pytest.raises(ValueError, match="built-in fixed-OU.*custom"):
                build_co2_from_materialized(inputs, boundary=boundary, prepared_inputs=reopened, priors=priors, **options)
    point = rebuilt.model.initial_point()
    for name in (("co2_bc", "o2_bc") if linked else ("bc",)):
        assert point[name].size == 2


@pytest.mark.parametrize("linked", [False, True])
def test_raw_ordinary_boundary_period_options_bind(linked):
    prepared = _raw_ordinary(linked)
    owner = Co2O2ModelConfig if linked else Co2ModelConfig
    choices = owner.from_params(_config(linked))
    options = choices.bind(prepared) if linked else choices.model_arguments()
    assert set(_frequencies(options, linked).values()) == {None}
    choices = owner.from_params(_config(linked, frequency="2D"))
    options = choices.bind(prepared) if linked else choices.model_arguments()
    assert set(_frequencies(options, linked).values()) == {"2D"}


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
    # Both channels carry one shared concrete calendar.
    frequencies = set(_frequencies(bound, True).values())
    assert len(frequencies) == 1
    assert normalise_frequency(frequencies.pop()) == normalise_frequency("48h")
    assert {choice.anchor_time for choice in bound["boundary"].values()} == {"1970-01-01T12:00:00"}
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
        choices = options["boundary"].values() if linked else [options["boundary"]]
        assert {choice.frequency for choice in choices} == {"2D"}
        assert {choice.anchor_time for choice in choices} == {anchor}
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


def test_linked_period_override_supports_mixed_raw_channel_representations(tmp_path):
    prepared = _joint()
    ordinary = _raw_ordinary(True)
    prepared = replace(prepared, boundary_operands={"co2": prepared.boundary_operands["co2"]},
                       boundary_sensitivity={"o2": ordinary.boundary_sensitivity["o2"]})
    config = _config(True, frequency="2D", anchor_time="1970-01-01T12:00:00")
    assert set(_frequencies(Co2O2ModelConfig.from_params(config).bind(prepared), True).values()) == {"2D"}
    manifest = co2_o2_handoffs.save_preparation(
        prepared, output_dir=tmp_path / "prepared", domain="EUROPE",
        start_date="1969-12-01", end_date="1970-02-01",
    )
    result = co2_o2_handoffs.prior_predictive(
        manifest, model=config, output_dir=tmp_path / "prior", draws=2, random_seed=13,
    )
    assert result["status"] == "pass"


@pytest.mark.parametrize("linked", [False, True])
def test_saved_centered_periods_support_fixed_ou(tmp_path, linked):
    module = co2_o2_handoffs if linked else handoffs
    prepared = _joint() if linked else _ordinary()
    config = _config(linked, frequency="2D")
    config["likelihood"] = {"kind": "fixed_ou", "tau_hours": 2.0, "fixed_site_amplitudes": 0.1}
    manifest = module.save_preparation(
        prepared, output_dir=tmp_path / "prepared", domain="EUROPE",
        start_date="1969-12-01", end_date="2020-02-01",
    )
    result = module.prior_predictive(
        manifest, model=config, output_dir=tmp_path / "prior", draws=2, random_seed=13,
    )
    assert result["status"] == "pass"
    trace = load_trace(tmp_path / "prior" / "prior-predictive.nc")
    observed = prepared.observations if linked else prepared.inv_inputs.mf
    xr.testing.assert_allclose(trace.observed_data.y, observed)
    assert trace.prior_predictive.y.attrs["concentration_representation"] == "absolute-v1"
    # atol=5 tests reference restoration only; independent density/gradient
    # oracles in test_centered_fixed_ou retain their tight tolerances.
    np.testing.assert_allclose(trace.prior_predictive.y, trace.prior.modelled_concentration, atol=5, rtol=0)
    # Native linked channel calendars cover different numbers of 2-day periods.
    expected_states = {"co2_": 2, "o2_": 4} if linked else {"": 4}
    for prefix, count in expected_states.items():
        assert trace.prior[f"{prefix}bc"].shape[-1] == count


def test_saved_centered_cached_ou_prior_predictions_remain_absolute(tmp_path):
    from openghg_inversions.postprocessing.boundary_outputs import restore_absolute_predictives
    from openghg_inversions.serialization import save_trace

    prepared = _ordinary()
    config = _config(False, frequency="2D", anchor_time="2020-01-01T12:00:00")
    config["likelihood"] = {
        "kind": "fixed_ou", "tau_hours": 2.0,
        "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.1},
    }
    manifest = handoffs.save_preparation(
        prepared, output_dir=tmp_path / "prepared", domain="EUROPE",
        start_date="2019-12-01", end_date="2020-02-01",
    )
    result = handoffs.prior_predictive(
        manifest, model=config, output_dir=tmp_path / "prior", draws=3, random_seed=13,
    )
    assert result["status"] == "pass"
    trace = load_trace(tmp_path / "prior" / "prior-predictive.nc")
    metadata = json.loads(trace.attrs["rhime_model_metadata"])
    assert metadata["boundary_representation"] == "centered-v1"
    assert metadata["bc_frequency"] == "2D"
    assert metadata["bc_anchor_time"] == "2020-01-01T12:00:00"
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    assert roles["observation"] == "y"
    assert roles["boundary_concentration"] == "mu_bc"
    assert roles["boundary_anomaly"] == "H_bc_anomaly"
    assert roles["boundary_mass"] == "H_bc_mass"
    assert "boundary_sensitivity" not in roles
    xr.testing.assert_allclose(trace.observed_data.y, prepared.inv_inputs.mf)
    assert trace.prior_predictive.y.attrs["concentration_representation"] == "absolute-v1"
    assert trace.prior.bc.shape[-1] == 4
    output_dim = str(trace.prior.mu_bc.dims[-1])
    absolute_design = (
        trace.constant_data.H_bc_anomaly + trace.constant_data.bc_reference * trace.constant_data.H_bc_mass
    ).transpose(output_dim, ...).values
    expected_boundary = np.einsum("...j,ij->...i", trace.prior.bc.values, absolute_design)
    np.testing.assert_allclose(trace.prior.mu_bc, expected_boundary)
    # This noisy predictive check detects missing large-reference restoration.
    # Separate full-gradient oracles establish numerical precision.
    np.testing.assert_allclose(
        trace.prior_predictive.y, trace.prior.modelled_concentration, atol=5, rtol=0,
    )
    observed = trace.observed_data.y.copy(deep=True)
    predictions = trace.prior_predictive.y.copy(deep=True)
    restore_absolute_predictives(trace)
    xr.testing.assert_identical(trace.observed_data.y, observed)
    xr.testing.assert_identical(trace.prior_predictive.y, predictions)
    path = tmp_path / "reopened.nc"
    save_trace(trace, path)
    reopened = load_trace(path)
    restore_absolute_predictives(reopened)
    xr.testing.assert_identical(reopened.observed_data.y, observed)
    xr.testing.assert_identical(reopened.prior_predictive.y, predictions)


@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_inference_graph_consumes_boundary_periods(centered, cached):
    from openghg_inversions.rhime.co2.inference import build_co2_inference

    prepared = _ordinary() if centered else _raw_ordinary(False)
    config = _config(False, frequency="2D", anchor_time="2020-01-01T12:00:00")
    if cached:
        config["likelihood"] = {
            "kind": "fixed_ou", "tau_hours": 2.0,
            "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 0.1},
        }
    built, cache = build_co2_inference(prepared, Co2ModelConfig.from_params(config).model_arguments())
    assert (cache is not None) == cached
    assert built.model.initial_point()["bc"].size == 4
    # The borrowed raw preparation remains reusable with a different period choice.
    single, _ = build_co2_inference(
        prepared, Co2ModelConfig.from_params(_config(False)).model_arguments(),
    )
    assert single.model.initial_point()["bc"].size == 2


def test_in_memory_linked_period_override_supports_mixed_raw_channel_representations():
    prepared = _joint()
    ordinary = _raw_ordinary(True)
    prepared = replace(prepared, boundary_operands={"co2": prepared.boundary_operands["co2"]},
                       boundary_sensitivity={"o2": ordinary.boundary_sensitivity["o2"]})
    config = _config(True, frequency="2D", anchor_time="1970-01-01T12:00:00")
    from openghg_inversions.rhime.co2.co2_o2_runner import build_rhime_co2_o2

    bound = Co2O2ModelConfig.from_params(config).bind(prepared)
    assert set(_frequencies(bound, True).values()) == {"2D"}
    built = build_rhime_co2_o2(prepared_inputs=prepared, **bound)
    point = built.model.initial_point()
    for channel in ("co2", "o2"):
        assert point[f"{channel}_bc"].size == 4
