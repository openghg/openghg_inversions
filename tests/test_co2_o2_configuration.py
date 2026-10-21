"""Linked phase choices consume only the inputs owned by that phase."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from openghg_inversions.rhime.co2.co2_o2_configuration import (
    load_co2_o2_params,
    resolve_co2_o2_model,
    resolve_co2_o2_preparation,
    resolve_co2_o2_prior_seed,
    resolve_co2_o2_sampler,
    validate_co2_o2_units,
)
from openghg_inversions.rhime.co2.co2_o2_runner import build_rhime_co2_o2

from test_co2_o2_fixed_ou import _prepared
from test_rhime_co2_o2 import _independent_error


def _config():
    return {
        "format_version": 1,
        "recipe": "co2_o2",
        "channels": {
            "co2": {"units": "ppm", "independent_error_sd": 0.2},
            "o2": {"units": "ppm", "independent_error_sd": 0.3},
        },
        "likelihood": {"kind": "fixed_independent"},
    }


def test_loading_sources_and_channel_overrides_preserve_unselected_choices(tmp_path):
    config = _config()
    config["channels"]["co2"]["offset"] = {"prior": {"pdf": "normal", "mu": 0, "sigma": 1}}
    original = deepcopy(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    overrides = {"channels": {"co2": {"independent_error_sd": 0.5}}}
    from_file = load_co2_o2_params(params_file=path, overrides=overrides)
    assert from_file == load_co2_o2_params(params=config, overrides=overrides)
    assert from_file["channels"]["co2"] == {**config["channels"]["co2"], "independent_error_sd": 0.5}
    assert from_file["channels"]["o2"] == config["channels"]["o2"]
    assert config == original
    toml = tmp_path / "config.toml"
    toml.write_text('format_version = 1\nrecipe = "co2_o2"\n')
    assert load_co2_o2_params(config_file=toml) == {"format_version": 1, "recipe": "co2_o2"}


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"format_version": True}, "integer 1"),
        ({"recipe": "co2"}, "recipe"),
        ({"variant": "cached_fixed_ou"}, "no variant selector"),
        ({"unknown": {}}, "config.unknown"),
    ],
)
def test_loader_rejects_unsupported_headers(changes, message):
    with pytest.raises(ValueError, match=message):
        load_co2_o2_params(params={**_config(), **changes})


def test_phase_resolution_ignores_inactive_choices_and_keeps_requested_facts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_co2_o2_preparation({
        "prepared_inputs": {"path": "inputs.nc", "start_date": "2019-12-01", "end_date": "2020-02-01", "domain": "EUROPE"},
        "channels": "invalid", "sampling": "invalid", "outputs": "invalid",
    }, require_path=True) == {
        "prepared_path": tmp_path / "inputs.nc", "start_date": "2019-12-01", "end_date": "2020-02-01", "domain": "EUROPE",
    }
    config = {**_config(), "prepared_inputs": "invalid", "sampling": "invalid", "outputs": "invalid"}
    arguments = resolve_co2_o2_model(config, _prepared())
    np.testing.assert_allclose(arguments["independent_error_sd"], [0.2, 0.2, 0.3, 0.3, 0.3])
    assert "sampler" not in arguments and "prepared_inputs" not in arguments
    assert resolve_co2_o2_prior_seed({"sampling": {"random_seed": 7, "draws": "invalid"}}) == 7
    sampler = resolve_co2_o2_sampler({"sampling": {"draws": 8}, "channels": "invalid", "outputs": "invalid"})
    assert sampler.draws == 8
    assert sampler.nuts_sampler == "numpyro"
    with pytest.raises(ValueError, match="prepared_inputs.path"):
        resolve_co2_o2_preparation({}, require_path=True)


def test_model_uses_saved_errors_and_replay_boundary_rejects_conflicting_values():
    prepared = _prepared()
    saved = _independent_error(prepared)
    prepared = replace(prepared, independent_error_sd=saved)
    config = _config()
    for channel in config["channels"].values():
        channel.pop("independent_error_sd")
    arguments = resolve_co2_o2_model(config, prepared)
    assert arguments["independent_error_sd"] is saved
    with pytest.raises(ValueError, match="match the prepared artifact values"):
        build_rhime_co2_o2(prepared_inputs=prepared, **resolve_co2_o2_model(_config(), prepared))
    with pytest.raises(ValueError, match="prepared artifact"):
        resolve_co2_o2_model(config, replace(prepared, independent_error_sd=None))


def test_fixed_ou_configuration_retains_species_site_mappings_and_backend():
    config = _config()
    config["likelihood"] = {
        "kind": "fixed_ou", "tau_hours": {"co2:A": 6, "o2:A": 8, "o2:B": 2},
        "fixed_site_amplitudes": {"co2:A": 0.4, "o2:A": 0.6, "o2:B": 0.0},
    }
    arguments = resolve_co2_o2_model(config, _prepared())
    assert arguments["tau_hours"] == config["likelihood"]["tau_hours"]
    assert arguments["fixed_site_amplitudes"] == config["likelihood"]["fixed_site_amplitudes"]
    assert resolve_co2_o2_sampler(config).nuts_sampler == "pymc"
    with pytest.raises(ValueError, match="pymc"):
        resolve_co2_o2_sampler({**config, "sampling": {"nuts_sampler": "numpyro"}})


@pytest.mark.parametrize(
    "likelihood",
    [
        {"kind": "fixed_ou", "tau_hours": 2},
        {"kind": "fixed_ou", "tau_hours": 2, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1}},
        {"kind": "fixed_ou", "tau_hours": 2, "site_amplitude_prior_scale": 1},
    ],
)
def test_new_configured_path_rejects_inferred_ou_explicitly(likelihood):
    with pytest.raises(ValueError, match="inferred OU"):
        resolve_co2_o2_model({**_config(), "likelihood": likelihood}, _prepared())


@pytest.mark.parametrize("units", [["ppm", "ppm", "per meg", "per meg", "per meg"], ["bananas"] * 5, ["permeg"] * 5])
def test_actual_artifact_units_are_validated_without_model_settings(units):
    prepared = _prepared()
    prepared = replace(prepared, observations=prepared.observations.assign_coords(observation_units=("observation", units)))
    with pytest.raises(ValueError, match="identical|mol/mol|OPE-86"):
        validate_co2_o2_units(prepared)


def test_model_rejects_configured_units_that_differ_from_artifact():
    config = _config()
    for channel in config["channels"].values():
        channel["units"] = "ppb"
    with pytest.raises(ValueError, match="do not match prepared"):
        resolve_co2_o2_model(config, _prepared())
