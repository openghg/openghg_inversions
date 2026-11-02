"""The Pydantic prototype validates the JSON written by public handoff APIs."""

import json

import dask
import dask.array as da
from jsonschema import Draft202012Validator
import numpy as np
from pydantic import ValidationError
import pytest
import xarray as xr

from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime import _standard_stages as stages
from openghg_inversions.rhime._manifest_records import PrepareManifest, SampleManifest
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.rhime.specs import RhimeOutputSpec
from tests.helpers import make_trace
from tests.test_staged_workflow import _params
from tests.test_standard_handoffs import _saved


def test_public_writers_produce_json_matching_generated_schemas(tmp_path, monkeypatch):
    preparation, samples = _saved(tmp_path)
    for manifest, record_type in ((preparation, PrepareManifest), (samples, SampleManifest)):
        wire = json.loads(manifest.read_text())
        schema = record_type.model_json_schema()
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(wire)
        record = record_type.load(manifest)
        assert json.loads(record.model_dump_json(by_alias=True)) == wire
        assert not list(manifest.parent.glob(".*-manifest.tmp"))
        copied = tmp_path / f"copy-{record.stage}"
        copied.mkdir()
        assert record_type.load(record.save(copied)) == record
        assert not list(copied.glob(".*-manifest.tmp"))
    assert (preparation.parent / "prepared-inputs.nc").is_file()
    assert (samples.parent / "posterior.nc").is_file()
    assert isinstance(record.output_contract, OutputContract)
    monkeypatch.setattr(OutputContract, "from_dict", lambda *args: pytest.fail("revalidated loaded contract"))
    result = stages.postprocess(samples, output=RhimeOutputSpec(output_format="none"))
    assert isinstance(result.output_contract, OutputContract)
    assert result.run_spec.model.species == "ch4"
    assert result.output_metadata["build_and_sample_seconds"] == 2.5
    assert result.output_metadata["sampler"]["draws"] == 10


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("producer", "another-program"),
    ("family", "multisector"), ("stage", "prepare"), ("posterior", ""),
    ("posterior", 5), ("build_and_sample_seconds", -1), ("unknown", "discard me"),
])
def test_manifest_errors_have_field_paths_before_artifact_loading(tmp_path, monkeypatch, field, value):
    _, samples = _saved(tmp_path)
    wire = json.loads(samples.read_text())
    wire[field] = value
    samples.write_text(json.dumps(wire))
    monkeypatch.setattr(stages, "load_preparation", lambda *args: pytest.fail("loaded invalid manifest"))
    with pytest.raises(ValidationError) as caught:
        stages.postprocess(samples, output=RhimeOutputSpec(output_format="none"))
    assert caught.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("nested", ["model", "output_contract"])
def test_nested_record_errors_keep_their_location(tmp_path, monkeypatch, nested):
    _, samples = _saved(tmp_path)
    wire = json.loads(samples.read_text())
    if nested == "model":
        wire["model"]["sectors"][0]["flux_source"] = 4
        expected = ("model", "sectors", 0, "flux_source")
    else:
        wire["output_contract"]["variable_roles"]["flux_scale"] = 4
        expected = ("output_contract", "variable_roles", "flux_scale")
    samples.write_text(json.dumps(wire))
    monkeypatch.setattr(stages, "load_preparation", lambda *args: pytest.fail("loaded invalid manifest"))
    with pytest.raises(ValidationError) as caught:
        stages.postprocess(samples, output=RhimeOutputSpec(output_format="none"))
    assert caught.value.errors()[0]["loc"] == expected


@pytest.mark.parametrize("metadata,duration", [
    ({"options": {"target_accept": float("nan")}}, 0.0),
    ({}, float("inf")), ({}, -1.0), ({}, True),
])
def test_sample_writer_validates_before_creating_artifacts(tmp_path, monkeypatch, metadata, duration):
    config = RhimeConfig.from_params(_params(), multisector=False)
    monkeypatch.setattr(stages, "save_trace", lambda *args: pytest.fail("wrote invalid sample"))
    with pytest.raises(ValidationError):
        stages.save_samples(
            tmp_path / "prepared.json", make_trace(posterior=xr.Dataset()), model=config.model,
            output_contract=OutputContract.create(variable_roles={"flux_scale": "x"}),
            actual_inference=metadata, build_and_sample_seconds=duration, output_dir=tmp_path / "samples",
        )
    assert not (tmp_path / "samples").exists()


def test_output_record_copies_json_and_refuses_lazy_arrays_without_computing():
    metadata = {"builder": {"sources": ["bio"]}}
    contract = OutputContract.create(variable_roles={"flux_scale": "x"}, metadata=metadata)
    metadata["builder"]["sources"].append("fossil")
    wire = contract.to_dict()
    wire["metadata"]["builder"]["sources"].append("other")
    assert contract.metadata == {"builder": {"sources": ["bio"]}}
    assert contract.with_updates(metadata={}).metadata == {}
    calls = []

    @dask.delayed
    def lazy():
        calls.append(1)
        return np.ones(1)

    array = xr.DataArray(da.from_delayed(lazy(), shape=(1,), dtype=float), dims="region")
    with pytest.raises(ValidationError):
        OutputContract.create(variable_roles={"flux_scale": "x"}, metadata={"array": array})
    contract.metadata["array"] = array
    with pytest.raises(ValidationError):
        contract.to_dict()
    with pytest.raises(ValidationError):
        contract.with_updates()
    assert calls == []


@pytest.mark.parametrize("change", [
    {"supported_output_formats": ["paris"]},
    {"state_dimension_mapping": {"trace": "region"}},
    {"variable_roles": {" ": "x"}},
])
def test_generated_output_schema_expresses_custom_constraints(change):
    wire = OutputContract.create(variable_roles={"flux_scale": "x"}).to_dict()
    wire.update(change)
    assert list(Draft202012Validator(OutputContract.model_json_schema()).iter_errors(wire))
    with pytest.raises(ValidationError):
        OutputContract.model_validate_json(json.dumps(wire))


@pytest.mark.parametrize("field", [
    "schema", "schema_version", "variable_roles", "supported_output_formats", "metadata", "state_dimension_mapping",
])
def test_required_record_fields_are_not_filled_with_runtime_defaults(field):
    wire = OutputContract.create(variable_roles={"flux_scale": "x"}).to_dict()
    assert field in OutputContract.model_json_schema()["required"]
    del wire[field]
    with pytest.raises(ValidationError) as caught:
        OutputContract.from_dict(wire)
    assert caught.value.errors()[0]["loc"] == (field,)


def test_python_alias_does_not_replace_required_wire_schema_field():
    wire = OutputContract.create(variable_roles={"flux_scale": "x"}).to_dict()
    wire["schema_name"] = wire.pop("schema")
    with pytest.raises(ValidationError) as caught:
        OutputContract.from_dict(wire)
    assert {error["loc"] for error in caught.value.errors()} == {("schema",), ("schema_name",)}


@pytest.mark.parametrize("formats", [None, "none", 1])
def test_runtime_formats_report_validation_errors_for_invalid_containers(formats):
    with pytest.raises(ValidationError) as caught:
        OutputContract.create(variable_roles={"flux_scale": "x"}, supported_output_formats=formats)
    assert caught.value.errors()[0]["loc"] == ("supported_output_formats",)


def test_output_view_updates_validate_and_own_containers():
    original = OutputContract.create(variable_roles={"flux_scale": "x"}, metadata={"sources": ["bio"]})
    changed = original.with_updates(variable_roles={"flux_scale": "inner_scale"})
    changed.metadata["sources"].append("other")
    assert original.variable_roles == {"flux_scale": "x"}
    assert original.metadata == {"sources": ["bio"]}
    with pytest.raises(ValidationError, match="both 'trace' and 'basis'"):
        original.with_updates(state_dimension_mapping={"trace": "region"})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        original.with_updates(misspelled_roles={})


@pytest.mark.parametrize("invalid", [float("nan"), {1: "integer evidence", "1": "string evidence"}])
def test_existing_contract_is_revalidated_at_sample_write_boundary(tmp_path, monkeypatch, invalid):
    contract = OutputContract.create(variable_roles={"flux_scale": "x"})
    contract.metadata["unexpected"] = invalid
    config = RhimeConfig.from_params(_params(), multisector=False)
    monkeypatch.setattr(stages, "save_trace", lambda *args: pytest.fail("wrote invalid contract"))
    with pytest.raises(ValidationError) as caught:
        stages.save_samples(
            tmp_path / "prepared.json", make_trace(posterior=xr.Dataset()), model=config.model,
            output_contract=contract, actual_inference={}, build_and_sample_seconds=0.0,
            output_dir=tmp_path / "samples",
        )
    assert caught.value.errors()[0]["loc"][:2] == ("metadata", "unexpected")
    assert not (tmp_path / "samples").exists()


def test_lazy_duration_is_rejected_without_computing_or_writing(tmp_path, monkeypatch):
    calls = []

    @dask.delayed
    def duration():
        calls.append(1)
        return np.array(1.5)

    lazy = da.from_delayed(duration(), shape=(), dtype=float)
    preparation, samples = _saved(tmp_path)
    wire = json.loads(samples.read_text()) | {"build_and_sample_seconds": lazy}
    with pytest.raises(ValidationError) as caught:
        SampleManifest(**wire)
    assert caught.value.errors()[0]["loc"] == ("build_and_sample_seconds",)
    config = RhimeConfig.from_params(_params(), multisector=False)
    monkeypatch.setattr(stages, "save_trace", lambda *args: pytest.fail("wrote sample with lazy duration"))
    with pytest.raises(ValidationError):
        stages.save_samples(
            preparation, make_trace(posterior=xr.Dataset()), model=config.model,
            output_contract=OutputContract.create(variable_roles={"flux_scale": "x"}),
            actual_inference={}, build_and_sample_seconds=lazy, output_dir=tmp_path / "lazy-duration",
        )
    assert calls == []
    assert not (tmp_path / "lazy-duration").exists()


def test_manifest_save_revalidates_mutable_metadata_before_writing(tmp_path):
    _, samples = _saved(tmp_path)
    record = SampleManifest.load(samples)
    record.provenance["changed_after_construction"] = float("nan")
    destination = tmp_path / "invalid-copy"
    destination.mkdir()
    with pytest.raises(ValidationError):
        record.save(destination)
    assert not list(destination.iterdir())


def test_preparation_save_rejects_mutated_key_collision_before_writing(tmp_path):
    record = PrepareManifest(
        schema="openghg_inversions.standard.prepare", schema_version=1,
        producer="openghg_inversions", family="standard", stage="prepare",
        prepared_inputs="prepared-inputs.nc", data={}, basis_artifact_source=None,
    )
    record.data["collision"] = {1: "integer evidence", "1": "string evidence"}
    with pytest.raises(ValidationError) as caught:
        record.save(tmp_path)
    assert caught.value.errors()[0]["loc"][:2] == ("data", "collision")
    assert not list(tmp_path.iterdir())


def test_sample_save_rejects_mutated_nested_contract_before_writing(tmp_path):
    _, samples = _saved(tmp_path)
    record = SampleManifest.load(samples)
    record.output_contract.metadata["collision"] = {1: "integer evidence", "1": "string evidence"}
    destination = tmp_path / "invalid-copy"
    destination.mkdir()
    with pytest.raises(ValidationError) as caught:
        record.save(destination)
    assert caught.value.errors()[0]["loc"][:3] == ("output_contract", "metadata", "collision")
    assert not list(destination.iterdir())
    assert SampleManifest.load(samples).output_contract.metadata == {}
