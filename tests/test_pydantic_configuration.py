"""Validated configuration construction owns containers and borrows numerical leaves."""

from dataclasses import replace
from types import MappingProxyType

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter, ValidationError

from openghg_inversions.inference.sampling import RhimeSampler, resolve_sampler
from openghg_inversions.inversion_data import SiteOptions
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime.params import DataConfig, RhimeConfig
from openghg_inversions.rhime.specs import AdditiveSigmaSettings, FixedErrorSettings, PollutionEventSettings
from tests.test_rhime_configuration import _request


@pytest.mark.parametrize("owner,options,field", [
    (PollutionEventSettings, {"sigma_per_site": "false"}, "sigma_per_site"),
    (PollutionEventSettings, {"pollution_events_from_obs": 1}, "pollution_events_from_obs"),
    (AdditiveSigmaSettings, {"use_minimum_error_floor": "yes"}, "use_minimum_error_floor"),
    (FixedErrorSettings, {"sigma_freq": "1D"}, "sigma_freq"),
    (RhimeSampler, {"draws": True}, "draws"),
    (RhimeSampler, {"draws": 3.2}, "draws"),
    (RhimeSampler, {"progressbar": "false"}, "progressbar"),
])
def test_constructors_reject_invalid_values_at_their_field(owner, options, field):
    with pytest.raises(ValidationError) as caught:
        owner(**options)
    assert field in caught.value.errors()[0]["loc"]


@pytest.mark.parametrize("change,field", [
    ({"use_bc": "false"}, "use_bc"),
    ({"nbasis": 0}, "nbasis"),
    ({"flux_sources": ()}, "flux_sources"),
    ({"unknown": 1}, "unknown"),
])
def test_resolved_data_construction_is_validated(change, field):
    config = RhimeConfig.from_params(_request(), multisector=False)
    with pytest.raises(ValidationError) as caught:
        replace(config.data, **change)
    assert field in caught.value.errors()[0]["loc"]


def test_site_selector_construction_validates_entries_and_owns_sequences():
    options = SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h")
    with pytest.raises(ValidationError, match="max_level"):
        replace(options, max_level=(True,))
    original = ["1h"]
    updated = replace(options, averaging_period=original)
    original.append("2h")
    assert updated.averaging_period == ("1h",)


def test_nested_configuration_allows_serialization_only_sampler():
    config = RhimeConfig.from_params(_request(), multisector=False)
    no_sampler = replace(config, sampler=None)
    assert RhimeConfig.from_dict(no_sampler.to_dict()).sampler is None
    with pytest.raises(TypeError, match="sampler"):
        resolve_sampler(no_sampler.sampler)
    with pytest.raises(ValidationError, match="sampler"):
        replace(config, sampler="unknown")


def test_constructor_validation_borrows_lazy_prior_values_and_rejects_lazy_scalars():
    calls = []

    @dask.delayed
    def payload():
        calls.append("executed")
        return np.array(0.5)

    lazy = da.from_delayed(payload(), shape=(), dtype=float)
    prior = {"pdf": "normal", "mu": xr.DataArray(lazy), "sigma": 1., "nested": ["owned"]}
    settings = PollutionEventSettings(sigma_prior=MappingProxyType(prior))
    prior["nested"].append("caller")
    assert settings.sigma_prior["mu"] is prior["mu"]
    assert settings.sigma_prior["nested"] == ["owned"]
    for owner, options in (
        (PollutionEventSettings, {"power": lazy}),
        (PollutionEventSettings, {"sigma_per_site": lazy}),
        (RhimeSampler, {"draws": lazy}),
    ):
        with pytest.raises(ValidationError):
            owner(**options)
    config = RhimeConfig.from_params(_request(), multisector=False)
    with pytest.raises(ValueError, match="draws"):
        RhimeConfig.from_params(_request(draws=lazy), multisector=False)
    with pytest.raises(ValueError, match="min_error"):
        replace(config.data, min_error=lazy)
    assert calls == []


def test_resolved_defaults_and_data_roundtrip_use_existing_owners():
    config = RhimeConfig.from_params(_request(start_date="2020-02-01"), multisector=False)
    assert config.model.likelihood.sigma_freq_anchor == "2020-02-01"
    assert DataConfig.from_params(config.data.to_params()) == config.data
    assert PollutionEventSettings().sigma_freq_anchor is None


@pytest.mark.parametrize("settings", [PollutionEventSettings(), AdditiveSigmaSettings(), FixedErrorSettings()])
def test_component_json_matches_its_generated_schema(settings):
    adapter = TypeAdapter(type(settings))
    schema = adapter.json_schema(mode="serialization")
    wire = adapter.dump_python(settings, mode="json")
    Draft202012Validator(schema).validate(wire)
    assert adapter.validate_json(adapter.dump_json(settings)) == settings
    Draft202012Validator.check_schema(schema)


def test_output_contract_mutation_is_rejected_before_json_key_conversion():
    contract = OutputContract.create(variable_roles=MappingProxyType({"flux_scale": "x"}))
    contract.metadata["collision"] = {1: "integer evidence", "1": "string evidence"}
    for boundary in (contract.to_dict, contract.with_updates):
        with pytest.raises(ValidationError) as caught:
            boundary()
        assert caught.value.errors()[0]["loc"][:2] == ("metadata", "collision")


def test_output_contract_schema_requires_saved_state_and_safe_json():
    contract = OutputContract.create(variable_roles={"flux_scale": "x"})
    schema = OutputContract.model_json_schema()
    wire = contract.to_dict()
    Draft202012Validator(schema).validate(wire)
    for field in wire:
        missing = dict(wire)
        del missing[field]
        assert list(Draft202012Validator(schema).iter_errors(missing))
        with pytest.raises(ValidationError):
            OutputContract.from_dict(missing)
    with pytest.raises(ValidationError, match="metadata"):
        OutputContract.create(variable_roles={"flux_scale": "x"}, metadata={"bad": float("inf")})


def test_runnable_schema_export_example_roundtrips_paths_and_inlet_slices(tmp_path):
    import json
    from examples.pydantic_records import run_example

    paths = run_example(tmp_path)
    for path in paths:
        value = json.loads(path.read_text())
        if path.name.endswith(".schema.json"):
            Draft202012Validator.check_schema(value)
    data = json.loads((tmp_path / "resolved-data.json").read_text())
    assert data["inlet"] == [{"type": "slice", "start": "10m", "stop": "100m", "step": None}]
    assert DataConfig.from_params(data).site_options.inlet == (slice("10m", "100m"),)
