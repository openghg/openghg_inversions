"""Saved scientific descriptions do not reconstruct executable configuration."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd

import pytest
import xarray as xr
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError
from pytensor.tensor import get_underlying_scalar_constant_value

from openghg_inversions.models._prior_records import PriorDescription
from openghg_inversions.models.priors import prior_records, trace_prior_records
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime import _standard_stages as stages
from openghg_inversions.rhime._model_records import ModelDescription
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.rhime.specs import RhimeModelSpec
from openghg_inversions.rhime.outputs import annotate_sample_trace, make_model_output
from openghg_inversions.rhime.standard import build_standard_model, materialize_standard_inputs
from tests.helpers import make_trace
from tests.test_staged_workflow import _params, _prepared


@pytest.mark.parametrize("lower,upper", [(0., np.inf), (-np.inf, 2.), (-np.inf, np.inf)])
def test_unbounded_prior_builds_and_returns_in_memory_output(lower, upper):
    prior = {"pdf": "truncatednormal", "mu": 1., "sigma": .2, "lower": lower, "upper": upper}
    config = RhimeConfig.from_params(_params(
        x_prior=prior, output_format="none", save_inversion_output=False,
    ), multisector=False)
    prepared = _prepared()
    inputs, priors = materialize_standard_inputs(prepared, config.model)
    built = build_standard_model(inputs, config.model, priors=priors)
    posterior = make_trace(posterior=xr.Dataset(
        {"x": (("chain", "draw", "region"), [[[1.], [1.]]])},
        coords={"chain": [0], "draw": [0, 1], "region": [0]},
    ))
    annotate_sample_trace(built, posterior)
    output = make_model_output(
        prepared=prepared, model_spec=config.model, posterior=posterior,
        output_contract=built.output_contract, start_date=config.data.start_date,
        end_date=config.data.end_date, split_by_sectors=False,
    )
    expected = dict(prior, lower=None if lower == -np.inf else lower,
                    upper=None if upper == np.inf else upper)
    assert output.trace is posterior
    assert output.model_metadata["constructed_priors"]["x"] == expected
    assert output.model_metadata["sectors"][0]["x_prior"] == expected
    assert prior_records(built.model)["x"] == expected
    assert json.loads(json.dumps(prior_records(built.model), allow_nan=False))["x"] == expected
    assert config.model.sectors[0].x_prior == prior
    # Check the actual graph endpoints, without sampling or changing the distribution arguments.
    assert get_underlying_scalar_constant_value(built.model["x"].owner.inputs[-2]) == lower
    assert get_underlying_scalar_constant_value(built.model["x"].owner.inputs[-1]) == upper


def test_older_trace_unbounded_endpoints_normalize_without_mutating_the_trace():
    old = {"x": {"pdf": "truncatednormal", "lower": -np.inf, "upper": np.inf, "mu": 1., "sigma": .2}}
    trace = make_trace()
    encoded = json.dumps(old)
    trace.attrs["rhime_priors"] = encoded
    assert trace_prior_records(trace) == {"x": dict(old["x"], lower=None, upper=None)}
    assert trace.attrs["rhime_priors"] == encoded


@pytest.mark.parametrize("pdf,parameter,value", [
    ("truncatednormal", "lower", np.inf),
    ("truncatednormal", "upper", -np.inf),
    ("truncatednormal", "lower", np.nan),
    ("truncatednormal", "upper", np.nan),
    ("truncatednormal", "mu", np.inf),
    ("truncatednormal", "sigma", np.inf),
    ("normal", "upper", np.inf),
])
def test_nonfinite_prior_parameters_are_not_normalized_into_valid_descriptions(pdf, parameter, value):
    prior = {"pdf": pdf, "mu": 1., "sigma": .2, parameter: value}
    with pytest.raises(ValidationError):
        PriorDescription.model_validate(prior)
    trace = make_trace()
    trace.attrs["rhime_priors"] = json.dumps({"x": prior})
    with pytest.raises(ValueError, match="valid prior descriptions"):
        trace_prior_records(trace)


@pytest.mark.parametrize("priors", [
    {},
    {"x": {"pdf": "normal", "mu": {"variable": "registered_mean"}, "sigma": 1.}},
    {"x": {"pdf": "normal", "mu": {"symbolic": True}, "sigma": 1.}},
    {"x": {"pdf": "truncatednormal", "mu": 1., "sigma": .2, "lower": 0., "upper": None}},
])
def test_public_checkpoint_replay_keeps_unused_and_symbolic_prior_descriptions(tmp_path, monkeypatch, priors):
    config = RhimeConfig.from_params(_params(mismatch_model="pollution_event", power={"pdf": "normal", "mu": 1., "sigma": .1}), multisector=False)
    prepared = stages.save_preparation(_prepared(), data_config=config.data, output_dir=tmp_path / "prepare")
    trace = make_trace(posterior=xr.Dataset(
        {"x": (("chain", "draw", "region"), [[[1.], [1.]]])},
        coords={"chain": [0], "draw": [0, 1], "region": [0]},
    ))
    trace.attrs["rhime_priors"] = json.dumps(priors)
    contract = OutputContract.create(variable_roles={"flux_scale": "x"}, supported_output_formats=("none", "inv_out"))
    saved = stages.save_samples(
        prepared, trace, model=config.model, output_contract=contract, actual_inference={},
        preparation_generation=stages.load_preparation(prepared)[2], output_dir=tmp_path / "sample",
    )
    record = json.loads(saved.read_text())
    description = ModelDescription.model_validate(record["model"])
    Draft202012Validator(ModelDescription.model_json_schema()).validate(record["model"])
    assert description.likelihood.options.power is None
    monkeypatch.setattr(RhimeModelSpec, "__init__", lambda *a, **k: pytest.fail("executable restoration"))
    restored = stages.load_samples(saved)
    assert restored.model_metadata["constructed_priors"] == priors
    assert restored.model_metadata["sectors"][0]["x_prior"] == priors.get("x")
    assert restored.model_metadata["likelihood"]["power"] is None


def test_saved_model_nested_errors_and_missing_scientific_choices():
    config = RhimeConfig.from_params(_params(mismatch_model="pollution_event"), multisector=False)
    trace = make_trace()
    trace.attrs["rhime_priors"] = "{}"
    record = config.model.to_record(trace=trace, data_config=config.data)
    record["likelihood"]["options"]["unexpected"] = True
    with pytest.raises(ValidationError) as caught:
        ModelDescription.model_validate(record)
    assert caught.value.errors()[0]["loc"] == ("likelihood", "pollution_event", "options", "unexpected")
    del record["likelihood"]["options"]["unexpected"]
    del record["likelihood"]["options"]["sigma_per_site"]
    with pytest.raises(ValidationError, match="sigma_per_site"):
        ModelDescription.model_validate(record)


def test_saved_prior_references_are_names_checked_without_array_execution():
    description = PriorDescription.model_validate({"pdf": "normal", "mu": {"data": "missing"}, "sigma": 1.})
    with pytest.raises(ValueError, match="Missing registered prior data"):
        description.validate_references(make_trace())
    for value in ({"pdf": "normal", "mu": {"data": 1}}, {"pdf": "normal", "mu": float("nan")},
                  {"pdf": "normal", "mu": {"symbolic": 1}}, {"mu": 1.}):
        with pytest.raises(ValidationError):
            PriorDescription.model_validate(value)


@pytest.mark.parametrize("anchor", [pd.Timestamp("2019-01-01"), np.datetime64("2019-01-01")])
def test_model_writer_projects_supported_temporal_anchors_explicitly(anchor):
    config = RhimeConfig.from_params(_params(mismatch_model="pollution_event"), multisector=False)
    model = replace(config.model, likelihood=replace(config.model.likelihood, sigma_freq_anchor=anchor))
    trace = make_trace()
    trace.attrs["rhime_priors"] = "{}"
    record = model.to_record(trace=trace, data_config=config.data)
    assert isinstance(record["likelihood"]["options"]["sigma_freq_anchor"], str)
    ModelDescription.model_validate(record)


def test_constructed_trace_owns_every_reported_prior_slot():
    config = RhimeConfig.from_params(_params(
        mismatch_model="pollution_event", add_offset=True,
        power={"pdf": "normal", "mu": 1., "sigma": .1},
    ), multisector=False)
    actual = {name: {"pdf": "normal", "mu": index, "sigma": 1.}
              for index, name in enumerate(("x", "bc", "offset_latent", "sigma", "power"))}
    trace = make_trace()
    trace.attrs["rhime_priors"] = json.dumps(actual)
    record = config.model.to_record(trace=trace, data_config=config.data)
    stale = {"pdf": "normal", "mu": 999., "sigma": 1.}
    record.update(bc_prior=stale, offset_prior=stale)
    record["sectors"][0]["x_prior"] = stale
    record["likelihood"]["options"].update(sigma_prior=stale, power=stale)
    metadata = ModelDescription.model_validate(record).to_output_metadata(
        data_config=config.data, trace=trace, output_contract=OutputContract.create(variable_roles={"flux_scale": "x"}),
    )
    assert metadata["sectors"][0]["x_prior"] == actual["x"]
    assert metadata["bc_prior"] == actual["bc"]
    assert metadata["offset_prior"] == actual["offset_latent"]
    assert metadata["likelihood"]["sigma_prior"] == actual["sigma"]
    assert metadata["likelihood"]["power"] == actual["power"]


def test_config_json_rejects_runtime_state_policy_without_computing_it():
    import dask
    import dask.array as da
    from openghg_inversions.models.state_activity import StateActivity

    calls = []

    @dask.delayed
    def mask():
        calls.append("computed")
        return np.array([True])

    config = RhimeConfig.from_params(_params(), multisector=False)
    policy = StateActivity(active=xr.DataArray(da.from_delayed(mask(), shape=(1,), dtype=bool), dims="region"))
    model = replace(config.model, state_activity=policy)
    adapter = TypeAdapter(RhimeModelSpec)
    wire = json.loads(adapter.dump_json(config.model))
    Draft202012Validator(adapter.json_schema(mode="serialization")).validate(wire)
    wire["state_activity"] = {"active": False}
    with pytest.raises(ValidationError, match="StateActivity is runtime-only"):
        adapter.validate_json(json.dumps(wire))
    with pytest.raises(PydanticSerializationError, match="StateActivity is runtime-only"):
        adapter.dump_json(model)
    assert calls == []
