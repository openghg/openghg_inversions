"""Typed completion markers fail before numerical loading or publication."""

import json

import dask.array as da
from dask.callbacks import Callback
import pytest
from pydantic import ValidationError

from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime import _standard_stages, multisector_stages
from openghg_inversions.rhime._manifest_records import (
    Co2PreparationRecord, Co2PreparationWriteRecord, Co2SampleReadRecord, Co2SampleWriteRecord,
    MultisectorSampleRecord, StandardSampleRecord, checkpoint_dict, read_checkpoint,
)


def _sample(family="standard"):
    from openghg_inversions.rhime.params import RhimeConfig
    from tests.helpers import make_trace
    from tests.test_staged_workflow import _params

    config = RhimeConfig.from_params(_params(), multisector=False)
    trace = make_trace()
    trace.attrs["rhime_priors"] = json.dumps({"x": config.model.sectors[0].x_prior})
    return {
        "schema": f"openghg_inversions.{family}.sample", "schema_version": 1,
        "producer": "openghg_inversions", "stage": "sample",
        "preparation": "../prepared/prepare-manifest.json",
        "preparation_generation": "original-generation", "posterior": "posterior.nc",
        "output_contract": OutputContract.create(variable_roles={"flux_scale": "x"}).to_dict(),
        "model": config.model.to_record(trace=trace, data_config=config.data),
        "actual_inference": {"draws": 4}, "provenance": {},
    }


@pytest.mark.parametrize("family,stages", [
    ("standard", _standard_stages), ("multisector", multisector_stages),
])
@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("stage", "prepare"),
    ("posterior", ""), ("preparation_generation", ""),
    ("provenance", None), ("provenance", []), ("unrecognised", 1),
])
def test_bad_sample_fields_fail_before_numerical_io(tmp_path, monkeypatch, family, stages, field, value):
    record = {**_sample(family), field: value}
    manifest = tmp_path / "sample.json"
    manifest.write_text(json.dumps(record))
    monkeypatch.setattr(stages, "_load_preparation", lambda *a, **k: pytest.fail("loaded numerical inputs"))
    monkeypatch.setattr(stages, "load_trace", lambda *a, **k: pytest.fail("loaded trace"))
    with pytest.raises(ValueError, match=field):
        stages.load_samples(manifest)


@pytest.mark.parametrize("record_type", [StandardSampleRecord, MultisectorSampleRecord])
def test_record_reports_multiple_field_paths_and_preserves_open_metadata(record_type):
    family = "standard" if record_type is StandardSampleRecord else "multisector"
    raw = _sample(family)
    raw["provenance"] = {"extension": {"nullable": None, "labels": ["FF", "bio"]}}
    checked = record_type.model_validate(raw)
    assert checkpoint_dict(checked) == raw
    raw["posterior"] = 3
    raw["preparation_generation"] = None
    with pytest.raises(ValidationError) as error:
        record_type.model_validate(raw)
    assert {item["loc"] for item in error.value.errors()} >= {("posterior",), ("preparation_generation",)}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "integer key"}, da.ones(2, chunks=1)])
def test_record_rejects_non_json_metadata_without_executing_arrays(value):
    executed = []
    with Callback(pretask=lambda *args: executed.append(args)):
        with pytest.raises(ValidationError):
            StandardSampleRecord.model_validate({**_sample(), "actual_inference": {"value": value}})
    assert executed == []


def test_mutated_nested_metadata_is_checked_before_json_projection():
    record = StandardSampleRecord.model_validate(_sample())
    record.actual_inference[1] = "would collide with a string key"
    with pytest.raises(ValidationError):
        checkpoint_dict(record)


@pytest.mark.parametrize("description", [None, "malformed", [], {"invalid": float("nan")}])
def test_co2_reader_keeps_forgiving_descriptions_but_writer_is_strict(description):
    raw = _sample("co2")
    raw.pop("output_contract")
    raw.pop("model")
    raw["inference_kind"] = "mcmc"
    raw["actual_inference"] = description
    raw["provenance"] = description
    restored = Co2SampleReadRecord.model_validate(raw)
    assert restored.actual_inference == restored.provenance == {}
    with pytest.raises(ValidationError):
        Co2SampleWriteRecord.model_validate(raw)


def test_generated_manifest_schema_describes_actual_written_checkpoint(tmp_path):
    from tests.test_standard_handoffs import _saved

    _, manifest = _saved(tmp_path)
    raw = json.loads(manifest.read_text())
    schema = StandardSampleRecord.model_json_schema(by_alias=True)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(raw)
    assert schema["properties"]["schema"]["const"] == raw["schema"]
    assert read_checkpoint(manifest, StandardSampleRecord)[1].preparation_generation == raw["preparation_generation"]


def test_invalid_new_description_leaves_existing_checkpoint_untouched(tmp_path, monkeypatch):
    from tests.test_standard_handoffs import _contract, _model, _saved

    preparation, manifest = _saved(tmp_path)
    original = manifest.read_bytes()
    posterior = _standard_stages.load_trace(manifest.parent / "posterior.nc")
    generation = _standard_stages.load_preparation(preparation)[2]
    model = _model(manifest)
    contract = _contract(manifest)
    monkeypatch.setattr(_standard_stages, "save_trace", lambda *a, **k: pytest.fail("wrote trace"))
    with pytest.raises(ValidationError, match="actual_inference"):
        _standard_stages.save_samples(
            preparation, posterior, model=model, output_contract=contract,
            actual_inference={"draws": float("nan")}, preparation_generation=generation,
            output_dir=manifest.parent,
        )
    assert manifest.read_bytes() == original


def test_co2_optional_affine_and_unavailable_provenance_survive_disk_loading(tmp_path):
    from tests.test_co2_handoffs import FACTS, _components_fixture
    from openghg_inversions.rhime.co2 import handoffs

    prepared, _ = _components_fixture()
    manifest = handoffs.save_preparation(prepared, output_dir=tmp_path, **FACTS)
    raw = json.loads(manifest.read_text())
    raw["affine_reconstruction"] = None
    raw["provenance"] = "unavailable historical description"
    manifest.write_text(json.dumps(raw))
    loaded = handoffs.load_preparation(manifest)
    assert loaded.affine_payload is None
    assert loaded.start_date == FACTS["start_date"]
    assert read_checkpoint(manifest, Co2PreparationRecord)[1].provenance == {}
    with pytest.raises(ValidationError, match="provenance"):
        Co2PreparationWriteRecord.model_validate(raw)
