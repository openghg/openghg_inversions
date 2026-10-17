"""Ordinary CO2 checkpoint integrity and graph-free scientific replay."""

from dataclasses import replace
import json

import pytest
import xarray as xr

from openghg_inversions.basis.affine_flux_map_io import load as load_affine
from openghg_inversions.rhime.co2 import handoffs
from openghg_inversions.rhime.co2.co2_affine_output import produce_bucket_affine_flux_map
from openghg_inversions.rhime.co2.co2_outputs import Co2OutputOptions
from openghg_inversions.inference.sampling import RhimeSampler
from test_co2_affine_output import _prepared
from test_co2_staged_outputs import _components_fixture as _raw_components_fixture


def _components_fixture():
    prepared, trace = _raw_components_fixture()
    for group in trace.children.values():
        for name, value in group.data_vars.items():
            value.attrs["units"] = "1" if name == "state_custom" else "ppm"
    return prepared, trace


FACTS = {"start_date": "2019-12-01", "end_date": "2020-02-01", "domain": "EUROPE"}


def _model(**likelihood):
    return {"likelihood": {"kind": "additive_sigma", "no_model_error": True, **likelihood}}



@pytest.mark.parametrize("missing", ["start_date", "end_date", "domain"])
def test_imported_preparation_requires_missing_requested_facts(tmp_path, missing):
    prepared, _ = _prepared()
    source = tmp_path / "source.nc"
    prepared.save(source)
    facts = {name: value for name, value in FACTS.items() if name != missing}
    with pytest.raises(ValueError, match=missing):
        handoffs.prepare(prepared_path=source, output_dir=tmp_path / "missing", **facts)
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("extension", [".nc", ".zarr"])
def test_original_dates_are_distinct_from_observation_extrema_and_survive_import(tmp_path, extension):
    prepared, _ = _prepared()
    prepared = replace(prepared, provenance={**prepared.provenance, **FACTS})
    source = tmp_path / f"source{extension}"
    prepared.save(source)
    manifest = handoffs.prepare(prepared_path=source, output_dir=tmp_path / "import")
    loaded = handoffs.load_preparation(manifest)
    assert loaded.start_date == FACTS["start_date"]
    assert loaded.end_date == FACTS["end_date"]
    assert loaded.domain == FACTS["domain"]
    assert loaded.payload == source.resolve()
    assert list(manifest.parent.iterdir()) == [manifest]
    xr.testing.assert_identical(loaded.prepared.inv_inputs, prepared.inv_inputs)
    assert str(prepared.inv_inputs.time.values[0])[:10] == "2020-01-01"


def test_new_affine_pair_uses_saved_content_binding_and_manifest_last(tmp_path, monkeypatch):
    prepared, native = _prepared()
    artifact = produce_bucket_affine_flux_map(prepared, native, prepared_inputs_id="in-memory")
    manifest = handoffs.save_preparation(prepared, affine_artifact=artifact, output_dir=tmp_path / "new", **FACTS)
    loaded = handoffs.load_preparation(manifest)
    bound = handoffs.load_and_bind_affine_flux_map(loaded.affine_payload, loaded.payload)
    assert bound.artifact.prepared_inputs_id != "in-memory"
    xr.testing.assert_equal(bound.affine_map.flux, prepared.basis_functions.flux.rename("flux"))
    broken = replace(artifact, projection_provenance={"projection_strategy": "wrong"})
    with pytest.raises(ValueError, match="projection provenance"):
        handoffs.save_preparation(prepared, affine_artifact=broken, output_dir=tmp_path / "broken", **FACTS)
    assert not (tmp_path / "broken" / "prepare-manifest.json").exists()


@pytest.mark.parametrize("extension", [".nc", ".zarr"])
def test_imported_affine_pair_is_referenced_without_copying(tmp_path, extension):
    from openghg_inversions.basis.affine_flux_map_io import save as save_affine
    from openghg_inversions.rhime.co2.co2_affine_output import prepared_inputs_content_id

    prepared, native = _prepared()
    source = tmp_path / f"external-prepared{extension}"
    companion = tmp_path / f"external-affine{extension}"
    prepared.save(source)
    artifact = produce_bucket_affine_flux_map(prepared, native, prepared_inputs_id=prepared_inputs_content_id(source))
    save_affine(artifact, companion)
    manifest = handoffs.prepare(
        prepared_path=source, affine_path=companion, output_dir=tmp_path / "referenced", **FACTS,
    )
    restored = handoffs.load_preparation(manifest)
    assert restored.payload == source.resolve()
    assert restored.affine_payload == companion.resolve()
    assert list(manifest.parent.iterdir()) == [manifest]
    assert handoffs.load_and_bind_affine_flux_map(
        restored.affine_payload, restored.payload,
    ).artifact.prepared_inputs_id == artifact.prepared_inputs_id


def test_replaced_preparation_rejects_saved_generation_before_writing(tmp_path):
    prepared, trace = _components_fixture()
    manifest = handoffs.save_preparation(prepared, output_dir=tmp_path / "prep", **FACTS)
    loaded = handoffs.load_preparation(manifest)
    handoffs.save_preparation(prepared, output_dir=tmp_path / "prep", **FACTS)
    with pytest.raises(ValueError, match="replaced"):
        handoffs.save_samples(
            manifest, trace, actual_inference=RhimeSampler(draws=4, chains=1).describe(), preparation_generation=loaded.generation, output_dir=tmp_path / "sample",
        )
    assert not (tmp_path / "sample").exists()


def test_sample_writer_rejects_missing_role_even_without_immediate_products(tmp_path):
    prepared, trace = _components_fixture()
    manifest = handoffs.save_preparation(prepared, output_dir=tmp_path / "prep", **FACTS)
    generation = handoffs.load_preparation(manifest).generation
    trace["posterior"] = trace.posterior.to_dataset().drop_vars("state_custom")
    with pytest.raises(ValueError, match="flux_scale"):
        handoffs.save_samples(
            manifest, trace, actual_inference=RhimeSampler(draws=4, chains=1).describe(), preparation_generation=generation, output_dir=tmp_path / "sample",
        )
    assert not (tmp_path / "sample").exists()


def test_affine_source_or_projection_mismatch_never_completes(tmp_path):
    prepared, native = _prepared()
    artifact = produce_bucket_affine_flux_map(prepared, native, prepared_inputs_id="in-memory")
    manifest = handoffs.save_preparation(prepared, affine_artifact=artifact, output_dir=tmp_path / "prep", **FACTS)
    companion = manifest.parent / "affine-flux-map.nc"
    incompatible = load_affine(companion)
    assert incompatible.prepared_inputs_id
    with pytest.raises(ValueError, match="start_date"):
        handoffs.prepare(
            prepared_path=manifest.parent / "prepared-inputs.nc", affine_path=companion,
            output_dir=tmp_path / "incomplete", end_date=FACTS["end_date"], domain=FACTS["domain"],
        )
    assert not (tmp_path / "incomplete").exists()


def test_graph_free_replay_keeps_recorded_sampler_and_provenance(tmp_path, monkeypatch):
    prepared, trace = _components_fixture()
    trace.attrs["rhime_priors"] = "{}"
    prep = handoffs.save_preparation(prepared, output_dir=tmp_path / "prep", **FACTS)
    generation = handoffs.load_preparation(prep).generation
    sample = handoffs.save_samples(
        prep, trace, actual_inference=RhimeSampler(draws=4, chains=1).describe(), preparation_generation=generation, output_dir=tmp_path / "sample",
    )
    record = json.loads(sample.read_text())
    record["actual_inference"] = {"draws": 25}
    record["provenance"] = {"version": "saved-version", "revision": None, "dirty": None}
    sample.write_text(json.dumps(record))

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must not construct or sample")

    monkeypatch.setattr(handoffs, "prepare_co2_inputs", forbidden)
    monkeypatch.setattr(handoffs.pm, "sample_prior_predictive", forbidden)
    monkeypatch.setattr(handoffs, "sample_rhime_model", forbidden)
    monkeypatch.setattr(handoffs, "build_co2_from_materialized", forbidden)
    monkeypatch.setattr(handoffs, "materialize_co2_inputs", forbidden)
    monkeypatch.setattr(handoffs, "installed_ogi_provenance", forbidden)
    output = Co2OutputOptions(output_format="basic")
    result = handoffs.postprocess(sample, output=output)
    assert result.sampler is None
    assert result.output_metadata["sampler"] == {"draws": 25}
    assert result.output_metadata["provenance"] == record["provenance"]
    assert result.start_date == FACTS["start_date"]
    assert result.end_date == FACTS["end_date"]
    assert "basic" in result.outputs


def test_sample_checkpoint_cannot_overwrite_its_preparation(tmp_path):
    prepared, trace = _components_fixture()
    prep = handoffs.save_preparation(prepared, output_dir=tmp_path / "prep", **FACTS)
    generation = handoffs.load_preparation(prep).generation
    # Sample payload redirects onto the numerical preparation.
    (prep.parent / "posterior.nc").symlink_to(prep.parent / "prepared-inputs.nc")
    with pytest.raises(ValueError, match="overwrite"):
        handoffs.save_samples(prep, trace, actual_inference=RhimeSampler(draws=4, chains=1).describe(), preparation_generation=generation, output_dir=prep.parent)


def test_readiness_scientific_and_loading_failures_propagate_before_destination(tmp_path, monkeypatch):
    prepared, _ = _prepared()
    prep = handoffs.save_preparation(prepared, output_dir=tmp_path / "prep", **FACTS)

    def invalid(**kwargs):
        raise ValueError("invalid coherent likelihood")

    monkeypatch.setattr(handoffs, "materialize_co2_inputs", invalid)
    with pytest.raises(ValueError, match="invalid coherent likelihood"):
        handoffs.prior_predictive(prep, model=_model(), output_dir=tmp_path / "check", draws=2)
    assert not (tmp_path / "check").exists()
    (prep.parent / "prepared-inputs.nc").unlink()
    with pytest.raises(FileNotFoundError):
        handoffs.prior_predictive(prep, model=_model(), output_dir=tmp_path / "missing", draws=2)
    assert not (tmp_path / "missing").exists()


def test_failed_sample_write_invalidates_old_completion_marker(tmp_path, monkeypatch):
    prepared, trace = _components_fixture()
    prep = handoffs.save_preparation(prepared, output_dir=tmp_path / "prep", **FACTS)
    generation = handoffs.load_preparation(prep).generation
    sample = handoffs.save_samples(
        prep, trace, actual_inference=RhimeSampler(draws=4, chains=1).describe(), preparation_generation=generation, output_dir=tmp_path / "sample",
    )

    def failed(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(handoffs, "save_trace", failed)
    with pytest.raises(OSError, match="disk unavailable"):
        handoffs.save_samples(
            prep, trace, actual_inference=RhimeSampler(draws=4, chains=1).describe(), preparation_generation=generation, output_dir=tmp_path / "sample",
        )
    assert not sample.exists()


@pytest.mark.parametrize("recorded", [None, "missing", "malformed"])
def test_missing_or_malformed_descriptive_metadata_stays_unavailable(tmp_path, recorded):
    prepared, trace = _components_fixture()
    prep = handoffs.save_preparation(prepared, output_dir=tmp_path / "prep", **FACTS)
    sample = handoffs.save_samples(
        prep, trace, actual_inference={"draws": 4},
        preparation_generation=handoffs.load_preparation(prep).generation, output_dir=tmp_path / "sample",
    )
    record = json.loads(sample.read_text())
    for name in ("actual_inference", "provenance"):
        if recorded == "missing":
            record.pop(name)
        else:
            record[name] = recorded
    sample.write_text(json.dumps(record))
    loaded = handoffs.load_samples(sample)
    assert loaded.sampler is None
    assert loaded.output_metadata["sampler"] == {}
    assert loaded.output_metadata["provenance"] == {}


@pytest.mark.parametrize("sampler", [None, "unknown"])
def test_invalid_live_sampler_fails_before_loading_preparation(tmp_path, sampler):
    with pytest.raises((TypeError, ValueError), match="sampler"):
        handoffs.sample(tmp_path / "missing.json", model=_model(), sampler=sampler, output_dir=tmp_path / "sample")
    assert not (tmp_path / "sample").exists()


def test_sample_reader_rejects_unsupported_inference_kind(tmp_path):
    prepared, trace = _components_fixture()
    prep = handoffs.save_preparation(prepared, output_dir=tmp_path / "prep", **FACTS)
    sample = handoffs.save_samples(
        prep, trace, actual_inference={"draws": 4},
        preparation_generation=handoffs.load_preparation(prep).generation, output_dir=tmp_path / "sample",
    )
    record = json.loads(sample.read_text())
    record["inference_kind"] = "variational"
    sample.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="inference_kind"):
        handoffs.load_samples(sample)


def test_imported_preparation_manifest_cannot_alias_external_numerical_file(tmp_path):
    prepared, _ = _prepared()
    source = tmp_path / "external.nc"
    prepared.save(source)
    original = source.read_bytes()
    destination = tmp_path / "reference"
    destination.mkdir()
    (destination / "prepare-manifest.json").symlink_to(source)
    with pytest.raises(ValueError, match="beneath|overwrite"):
        handoffs.prepare(prepared_path=source, output_dir=destination, **FACTS)
    assert source.read_bytes() == original


@pytest.mark.parametrize("phase", ["sample", "prior_predictive"])
def test_inference_loads_decoded_model_without_reading_unused_affine(tmp_path, monkeypatch, phase):
    prepared, native = _prepared()
    artifact = produce_bucket_affine_flux_map(prepared, native, prepared_inputs_id="in-memory")
    prep = handoffs.save_preparation(prepared, affine_artifact=artifact, output_dir=tmp_path / "prep", **FACTS)
    (prep.parent / "affine-flux-map.nc").unlink()
    captured = []

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference must not load an output reconstruction")

    def model_boundary(**kwargs):
        captured.append(kwargs)
        raise RuntimeError("model boundary reached")

    monkeypatch.setattr(handoffs, "load_and_bind_affine_flux_map", forbidden)
    monkeypatch.setattr(handoffs, "materialize_co2_inputs", model_boundary)
    with pytest.raises(RuntimeError, match="model boundary reached"):
        getattr(handoffs, phase)(
            prep, model={"model": {"boundary": {"enabled": False}},
                         "likelihood": {"kind": "additive_sigma", "no_model_error": True}},
            output_dir=tmp_path / "inference",
        )
    assert captured[0]["use_bc"] is False
    assert captured[0]["no_model_error"] is True
    assert not (tmp_path / "inference").exists()


@pytest.mark.parametrize("companion", ["missing", "incompatible"])
def test_replay_checks_affine_companion_even_without_products(tmp_path, companion):
    from openghg_inversions.basis.affine_flux_map_io import save as save_affine

    prepared, trace = _components_fixture()
    _, native = _prepared()
    artifact = produce_bucket_affine_flux_map(prepared, native, prepared_inputs_id="in-memory")
    prep = handoffs.save_preparation(prepared, affine_artifact=artifact, output_dir=tmp_path / "prep", **FACTS)
    samples = handoffs.save_samples(
        prep, trace, actual_inference={"draws": 3},
        preparation_generation=handoffs.load_preparation(prep).generation,
        output_dir=tmp_path / "sample",
    )
    if companion == "missing":
        (prep.parent / "affine-flux-map.nc").unlink()
        expected = FileNotFoundError
    else:
        recorded = json.loads(prep.read_text())
        original = load_affine(prep.parent / "affine-flux-map.nc")
        save_affine(replace(original, projection_provenance={"projection_strategy": "wrong"}),
                    prep.parent / "incompatible.nc")
        recorded["affine_reconstruction"] = "incompatible.nc"
        prep.write_text(json.dumps(recorded))
        expected = ValueError
    with pytest.raises(expected):
        handoffs.postprocess(samples, output=Co2OutputOptions(output_format="none", output_path=tmp_path / "products"))
    assert not (tmp_path / "products").exists()


def test_decoded_prior_readiness_uses_shared_scientific_trace_annotation(tmp_path):
    from openghg_inversions.serialization import load_trace

    prepared, _ = _prepared()
    prep = handoffs.save_preparation(prepared, output_dir=tmp_path / "prep", **FACTS)
    result = handoffs.prior_predictive(
        prep, model=_model(), output_dir=tmp_path / "check", draws=2, random_seed=19,
    )
    assert result["status"] == "pass"
    trace = load_trace(tmp_path / "check" / "prior-predictive.nc")
    assert trace.prior_predictive.indexes["nmeasure"].equals(prepared.inv_inputs.indexes["nmeasure"])
    metadata = json.loads(trace.attrs["rhime_model_metadata"])
    assert metadata["no_model_error"] is True
    assert metadata["likelihood"] == "additive_sigma"
    assert metadata["use_bc"] is False
    assert "rhime_priors" in trace.attrs
