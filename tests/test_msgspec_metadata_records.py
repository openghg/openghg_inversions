"""The msgspec alternative accepts the same actual handoff JSON as Pydantic."""

from dataclasses import replace
from datetime import datetime, timedelta
import json
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

msgspec = pytest.importorskip("msgspec")
jsonschema = pytest.importorskip("jsonschema")

from openghg_inversions.experimental.metadata_records import (  # noqa: E402
    AdditiveSigmaRecord,
    FixedErrorRecord,
    ModelRecord,
    OutputContractRecord,
    PollutionEventRecord,
    PreparationRecord,
    SampleRecord,
    convert_record,
    decode_record,
    encode_record,
)
from openghg_inversions.postprocessing.contracts import OutputContract  # noqa: E402
from openghg_inversions.rhime import _standard_stages as stages  # noqa: E402
from openghg_inversions.rhime.params import RhimeConfig  # noqa: E402
from openghg_inversions.rhime.specs import (  # noqa: E402
    AdditiveSigmaSettings,
    FixedErrorSettings,
    PollutionEventSettings,
)
from tests.helpers import make_trace  # noqa: E402
from tests.test_standard_handoffs import _saved  # noqa: E402
from tests.test_staged_workflow import _params  # noqa: E402


def test_public_writer_manifests_roundtrip_and_match_independent_schema(tmp_path):
    preparation, sampled = _saved(tmp_path)
    for path, expected_type in ((preparation, PreparationRecord), (sampled, SampleRecord)):
        raw = json.loads(path.read_text())
        decoded = decode_record(path.read_bytes())
        assert isinstance(decoded, expected_type)
        assert json.loads(encode_record(decoded)) == raw
        assert msgspec.to_builtins(convert_record(raw, expected_type)) == raw
        schema = msgspec.json.schema(expected_type)
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(raw)
    assert isinstance(decoded.model.likelihood, FixedErrorRecord)
    assert decoded.posterior == "posterior.nc"
    for schema_flag in ([], ["--schema"]):
        result = subprocess.run(
            [sys.executable, "-m", "openghg_inversions.experimental.metadata_records", str(sampled), *schema_flag],
            check=True, capture_output=True, text=True,
        )
        assert json.loads(result.stdout) == (msgspec.json.schema(SampleRecord) if schema_flag else raw)
    del raw["actual_inference"], raw["provenance"]
    defaulted = convert_record(raw, SampleRecord)
    assert defaulted.actual_inference is None and defaulted.provenance == {}


@pytest.mark.parametrize("settings,record_type", [
    (FixedErrorSettings(), FixedErrorRecord),
    (AdditiveSigmaSettings(sigma_freq_anchor=np.datetime64("2019-01-01")), AdditiveSigmaRecord),
    (PollutionEventSettings(power=1.99, sigma_freq_anchor=datetime(2019, 1, 1)), PollutionEventRecord),
])
def test_likelihood_variants_and_registered_prior_references_survive_real_writer(
    tmp_path, settings, record_type,
):
    preparation, _ = _saved(tmp_path)
    config = RhimeConfig.from_params(_params(), multisector=False)
    prior = {"pdf": "normal", "mu": {"data": "x_prior_mu"}, "sigma": 0.2}
    posterior = make_trace(
        posterior=xr.Dataset({"x": (("chain", "draw", "region"), [[[1.0]]])}),
        constant_data=xr.Dataset({"x_prior_mu": ("region", [1.2])}),
    )
    posterior.attrs["rhime_priors"] = json.dumps({"x": prior})
    path = stages.save_samples(
        preparation, posterior, model=replace(config.model, likelihood=settings), actual_inference={},
        output_contract=OutputContract.create(variable_roles={"flux_scale": "x"}),
        build_and_sample_seconds=0.0, output_dir=tmp_path / "variant",
    )
    decoded = decode_record(path.read_bytes(), SampleRecord)
    assert isinstance(decoded.model.likelihood, record_type)
    assert decoded.model.sectors[0].x_prior["mu"].data == "x_prior_mu"
    assert json.loads(encode_record(decoded)) == json.loads(path.read_text())
    jsonschema.Draft202012Validator(msgspec.json.schema(SampleRecord)).validate(json.loads(path.read_text()))
    missing_kind = json.loads(path.read_text())["model"]["likelihood"]
    del missing_kind["kind"]
    for validate in (convert_record, lambda raw, cls: decode_record(json.dumps(raw), cls)):
        with pytest.raises(msgspec.ValidationError, match="missing required field `kind`"):
            validate(missing_kind, record_type)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(msgspec.json.schema(record_type)).validate(missing_kind)


@pytest.mark.parametrize("target", ["prepare", "sample", "output_contract", "nested_output_contract"])
def test_schema_fields_are_required_under_known_and_dispatched_decoding(tmp_path, target):
    preparation, sampled = _saved(tmp_path)
    raw = json.loads((preparation if target == "prepare" else sampled).read_text())
    record_type = PreparationRecord if target == "prepare" else SampleRecord
    if target == "output_contract":
        raw = raw["output_contract"]
        record_type = OutputContractRecord
    del (raw["output_contract"] if target == "nested_output_contract" else raw)["schema"]
    with pytest.raises(msgspec.ValidationError, match="missing required field `schema`"):
        convert_record(raw, record_type)
    with pytest.raises(msgspec.ValidationError, match="missing required field `schema`"):
        decode_record(json.dumps(raw), record_type)
    if target != "output_contract":
        with pytest.raises(msgspec.ValidationError, match="schema"):
            decode_record(json.dumps(raw))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(msgspec.json.schema(record_type)).validate(raw)


@pytest.mark.parametrize("field,value,match", [
    ("schema_version", True, "Expected `int`, got `bool`"),
    ("schema_version", 2, "Invalid enum value 2"),
    ("unexpected", None, "unknown field `unexpected`"),
    ("posterior", " ", "matching regex"),
])
def test_strict_manifest_errors_name_the_broken_field(tmp_path, field, value, match):
    _, sampled = _saved(tmp_path)
    raw = json.loads(sampled.read_text())
    raw[field] = value
    with pytest.raises(msgspec.ValidationError, match=match):
        convert_record(raw, SampleRecord)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(msgspec.json.schema(SampleRecord)).validate(raw)


def test_nested_likelihood_errors_are_local_and_arrays_are_references(tmp_path):
    import dask
    import dask.array as da

    _, sampled = _saved(tmp_path)
    raw = json.loads(sampled.read_text())["model"]
    raw["likelihood"]["options"]["sigma_per_site"] = True
    with pytest.raises(msgspec.ValidationError, match="unknown field `sigma_per_site`.*likelihood.options"):
        convert_record(raw, ModelRecord)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(msgspec.json.schema(ModelRecord)).validate(raw)
    raw["likelihood"]["options"] = {}
    raw["sectors"][0]["x_prior"] = {"mu": np.array([1.0, 2.0])}
    with pytest.raises(msgspec.ValidationError, match="got `array`.*sectors\\[0\\].x_prior"):
        convert_record(raw, ModelRecord)
    calls = []

    @dask.delayed
    def payload():
        calls.append(1)
        return np.ones(2)

    lazy = xr.DataArray(da.from_delayed(payload(), shape=(2,), dtype=float), dims="region")
    raw["sectors"][0]["x_prior"] = {"mu": lazy}
    with pytest.raises(TypeError, match="DataArray.*not JSON serializable"):
        convert_record(raw, ModelRecord)
    assert calls == []


def test_finite_json_and_explicit_numpy_temporal_policy(tmp_path):
    _, sampled = _saved(tmp_path)
    raw = json.loads(sampled.read_text())["output_contract"]
    raw["metadata"] = {"count": np.int64(3), "anchor": np.datetime64("2019-01-01"),
                       "duration": timedelta(seconds=2), "ratio": np.float64(1.2)}
    record = convert_record(raw, OutputContractRecord)
    assert record.metadata == {"count": 3, "anchor": "2019-01-01", "duration": "0:00:02", "ratio": 1.2}
    for invalid in (float("nan"), float("inf")):
        record.metadata["nested"] = {"invalid": invalid}
        assert b'"invalid":null' in msgspec.json.encode(record)  # Native encoding loses nonfinite values.
        with pytest.raises(ValueError, match="finite JSON"):
            encode_record(record)
        with pytest.raises(ValueError, match="finite JSON"):
            convert_record(msgspec.to_builtins(record), OutputContractRecord)
    with pytest.raises(msgspec.DecodeError):
        decode_record(b'{"metadata": {"invalid": NaN}}', OutputContractRecord)


@pytest.mark.parametrize("field,value", [
    ("variable_roles", {}),
    ("supported_output_formats", ["inv_out"]),
    ("state_dimension_mapping", {"trace": "region"}),
    ("state_dimension_mapping", {"trace": "region", "basis": "region", "unexpected": "region"}),
])
def test_output_contract_relations_are_in_runtime_and_schema(tmp_path, field, value):
    _, sampled = _saved(tmp_path)
    raw = json.loads(sampled.read_text())["output_contract"]
    raw[field] = value
    with pytest.raises(msgspec.ValidationError):
        convert_record(raw, OutputContractRecord)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(msgspec.json.schema(OutputContractRecord)).validate(raw)


@pytest.mark.parametrize("description", [None, {"data": "parameter"}, {"variable": "scale"}, {"symbolic": True}])
def test_prior_provenance_roundtrips_without_executable_restoration(tmp_path, description):
    _, sampled = _saved(tmp_path)
    raw = json.loads(sampled.read_text())["model"]
    raw["sectors"][0]["x_prior"] = {"pdf": "normal", "sigma": description, "lower": None}
    record = convert_record(raw, ModelRecord)
    assert json.loads(encode_record(record)) == raw
    jsonschema.Draft202012Validator(msgspec.json.schema(ModelRecord)).validate(raw)


@pytest.mark.parametrize("description", [
    {}, {"data": None}, {"variable": ""}, {"variable": " "}, {"symbolic": False}, {"symbolic": 1},
    {"variable": "scale", "data": "parameter"}, {"symbolic": True, "extra": 1},
])
def test_prior_descriptor_runtime_and_schema_reject_the_same_shapes(tmp_path, description):
    _, sampled = _saved(tmp_path)
    raw = json.loads(sampled.read_text())["model"]
    raw["sectors"][0]["x_prior"] = {"sigma": description}
    with pytest.raises(msgspec.ValidationError):
        convert_record(raw, ModelRecord)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(msgspec.json.schema(ModelRecord)).validate(raw)
