"""Descriptive priors survive ordinary products without claiming graph restoration."""

from dataclasses import replace
import json

import dask
import dask.array as da
import numpy as np
import pymc as pm
import pytest
import xarray as xr
from pydantic import ValidationError
from pytensor.graph.basic import Variable

from openghg_inversions.models.priors import PriorRecord, parse_prior, prior_records, trace_prior_records
from openghg_inversions.rhime import _standard_stages as stages
from openghg_inversions.rhime._model_records import ModelRecord
from openghg_inversions.rhime.outputs import make_standard_rhime_outputs
from openghg_inversions.rhime.specs import RhimeModelSpec, RhimeOutputSpec, SectorSpec
from tests.helpers import make_trace
from tests.test_standard_handoffs import _saved


def test_nullable_scalar_prior_roundtrips_as_executable_input_and_metadata():
    jsonschema = pytest.importorskip("jsonschema")
    with pm.Model() as model:
        parse_prior("x", {"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.2, "lower": 0.0, "upper": None})
    captured = prior_records(model)
    prior = PriorRecord.model_validate(captured["x"])
    assert prior.to_runtime(trace=None) == captured["x"]
    assert prior.for_postprocessing(trace=None)["upper"] is None
    trace = xr.DataTree(xr.Dataset(attrs={"rhime_priors": json.dumps(captured)}))
    assert trace_prior_records(trace) == captured
    spec = RhimeModelSpec("ch4", "EUROPE", (SectorSpec("total", "inventory", {}, "total"),), use_bc=False)
    record = spec.to_record(priors=captured)
    jsonschema.Draft202012Validator(ModelRecord.model_json_schema()).validate(record)
    assert RhimeModelSpec.from_record(record).sectors[0].x_prior == captured["x"]


@pytest.mark.parametrize("description", [{"variable": "scale"}, {"symbolic": True}])
def test_prior_descriptions_and_nullable_values_never_execute(description):
    calls = []

    @dask.delayed
    def values():
        calls.append(1)
        return np.array([1.0])

    data = xr.DataArray(da.from_delayed(values(), shape=(1,), dtype=float), dims="region")
    captured = {"pdf": "truncatednormal", "mu": {"data": "x_prior_mu"},
                "sigma": description, "lower": None, "upper": None}
    trace = make_trace(constant_data=xr.Dataset({"x_prior_mu": data}))
    trace.attrs["rhime_priors"] = json.dumps({"x": captured})
    record = PriorRecord.model_validate(captured)
    assert record.for_postprocessing(trace=trace) == captured
    assert trace_prior_records(trace) == {"x": captured}
    assert calls == []
    with pytest.raises(ValueError, match="original model recipe is required"):
        record.to_runtime(trace=trace)
    with pytest.raises(ValueError, match="Missing registered prior data 'x_prior_mu'"):
        record.for_postprocessing(trace=None)
    assert calls == []


@pytest.mark.parametrize("invalid", [
    {}, {"variable": ""}, {"variable": " "}, {"variable": None},
    {"symbolic": False}, {"symbolic": 1}, {"symbolic": None},
    {"variable": "scale", "data": "parameter"}, {"symbolic": True, "extra": 1},
])
def test_prior_descriptor_schema_and_runtime_reject_the_same_malformed_shapes(invalid):
    jsonschema = pytest.importorskip("jsonschema")
    with pytest.raises(ValidationError):
        PriorRecord.model_validate({"sigma": invalid})
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(PriorRecord.model_json_schema()).validate({"sigma": invalid})


@pytest.mark.parametrize("operation", ["validate_references", "for_postprocessing", "to_runtime"])
def test_mutated_prior_roots_are_validated_before_projection_or_reference_lookup(operation):
    prior = PriorRecord.model_validate({"pdf": "normal"})
    prior.root[1] = "integer evidence"
    prior.root["1"] = "string evidence"
    with pytest.raises(ValidationError):
        getattr(prior, operation)(trace=None)
    assert prior.root[1] == "integer evidence" and prior.root["1"] == "string evidence"
    del prior.root[1]
    prior.root["mu"] = {"data": "missing"}
    with pytest.raises(ValueError, match="Missing registered prior data 'missing'"):
        getattr(prior, operation)(trace=None)
    prior.root["mu"] = {"variable": "scale"}
    if operation == "to_runtime":
        with pytest.raises(ValueError, match="original model recipe is required"):
            prior.to_runtime(trace=None)
    else:
        getattr(prior, operation)(trace=None)


@pytest.mark.parametrize("unnamed", [False, True])
def test_hierarchical_priors_survive_live_outputs_and_saved_posterior_replay(tmp_path, monkeypatch, unnamed):
    preparation, original_sample = _saved(tmp_path)
    output = RhimeOutputSpec(output_format="inv_out", save_inversion_output=False)
    original = stages.postprocess(original_sample, output=output)
    prepared, _ = stages.load_preparation(preparation)
    posterior = stages.load_trace(original_sample.parent / "posterior.nc")

    with pm.Model(coords={"region": [0]}) as model:
        scale = parse_prior("scale", {"pdf": "halfnormal", "sigma": 0.2})
        dependency = scale + 0.1 if unnamed else scale
        parse_prior("x", {"pdf": "truncatednormal", "mu": 1.0, "sigma": dependency,
                          "lower": None, "upper": None}, dims="region")
    captured = prior_records(model)
    description = {"symbolic": True} if unnamed else {"variable": "scale"}
    assert captured["x"]["sigma"] == description
    assert captured["x"]["lower"] is None and captured["x"]["upper"] is None
    posterior.attrs["rhime_priors"] = json.dumps(captured)

    def forbidden(*args, **kwargs):
        pytest.fail("metadata or posterior output attempted executable graph restoration")

    monkeypatch.setattr(Variable, "eval", forbidden)
    monkeypatch.setattr(ModelRecord, "to_runtime", forbidden)
    live = make_standard_rhime_outputs(
        result=replace(original, model=model, idata=posterior, inv_out=None, outputs={}), prepared=prepared,
    )
    assert live.inv_out.model_metadata["constructed_priors"] == captured
    assert live.inv_out.model_metadata["sectors"][0]["x_prior"] == captured["x"]

    sampled = stages.save_samples(
        preparation, posterior, model=original.model_spec, output_contract=original.output_contract,
        actual_inference={}, build_and_sample_seconds=0.0, output_dir=tmp_path / "hierarchical-samples",
    )
    replay = stages.postprocess(sampled, output=output)
    assert replay.model is None and replay.sampler is None
    assert replay.inv_out.model_metadata["constructed_priors"] == captured
    assert replay.model_spec.sectors[0].x_prior == captured["x"]
    xr.testing.assert_identical(replay.idata["posterior"], posterior["posterior"])
    # Descriptive support cannot replace the sampled state a product needs.
    incomplete = make_trace(posterior=posterior["posterior"].to_dataset().rename({"x": "x_latent"}))
    incomplete.attrs["rhime_priors"] = json.dumps(captured)
    with pytest.raises(ValueError, match="reconstruction state"):
        stages.save_samples(
            preparation, incomplete, model=original.model_spec, output_contract=original.output_contract,
            actual_inference={}, build_and_sample_seconds=0.0, output_dir=tmp_path / "incomplete",
        )


def test_only_executable_model_restoration_rejects_symbolic_descriptions(tmp_path):
    _, sampled = _saved(tmp_path)
    raw = json.loads(sampled.read_text())["model"]
    raw["sectors"][0]["x_prior"] = {"pdf": "normal", "mu": 1.0, "sigma": {"variable": "scale"}}
    record = ModelRecord.model_validate(raw)
    assert record.for_postprocessing().sectors[0].x_prior == raw["sectors"][0]["x_prior"]
    with pytest.raises(ValueError, match="Cannot restore executable prior parameter 'sigma'"):
        RhimeModelSpec.from_record(raw)
