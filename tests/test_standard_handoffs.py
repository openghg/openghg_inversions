"""Current standard numerical handoffs and graph-free output replay."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.rhime import _standard_stages as stages
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.rhime.specs import RhimeOutputSpec
from tests.helpers import make_trace
from tests.test_staged_workflow import _params, _prepared


def _saved(tmp_path):
    config = RhimeConfig.from_params(_params(), multisector=False)
    preparation = stages.save_preparation(_prepared(), data=config.data, output_dir=tmp_path / "prepared")
    posterior = make_trace(posterior=xr.Dataset(
        {"x": (("chain", "draw", "region"), [[[1.0], [2.0]], [[10.0], [20.0]]])},
        coords={"chain": [0, 1], "draw": [0, 1], "region": [0]},
    ))
    sampled = stages.save_samples(
        preparation, posterior, model=config.model, actual_inference=config.sampler.describe(),
        output_contract=OutputContract(variable_roles={"flux_scale": "x"},
                                       supported_output_formats=("none", "inv_out")),
        build_and_sample_seconds=2.5, output_dir=tmp_path / "samples",
    )
    return preparation, sampled


def test_manifest_relative_and_absolute_references_preserve_inputs(tmp_path, monkeypatch):
    preparation, _ = _saved(tmp_path)
    record = json.loads(preparation.read_text())
    assert record["prepared_inputs"] == "prepared-inputs.nc"
    assert "configuration_identity" not in record
    monkeypatch.setenv("RUN_ROOT", str(tmp_path / "wrong-root"))
    monkeypatch.chdir(tmp_path.parent)
    prepared, context = stages.load_preparation(preparation)
    xr.testing.assert_identical(prepared.inv_inputs, _prepared().inv_inputs)
    assert context["end_date"] == "2019-02-01"  # observations end on January 2
    record["prepared_inputs"] = str(preparation.parent / record["prepared_inputs"])
    preparation.write_text(json.dumps(record))
    xr.testing.assert_identical(stages.load_preparation(preparation)[0].inv_inputs, prepared.inv_inputs)


def test_fresh_destinations_and_manifest_last(tmp_path, monkeypatch):
    preparation, _ = _saved(tmp_path)
    config = RhimeConfig.from_params(_params(), multisector=False)
    with pytest.raises(FileExistsError):
        stages.save_preparation(_prepared(), data=config.data, output_dir=preparation.parent)
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(stages.RhimePreparedInputs, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        stages.save_preparation(_prepared(), data=config.data, output_dir=tmp_path / "failed")
    assert not (tmp_path / "failed" / "prepare-manifest.json").exists()


def test_replay_in_fresh_process_uses_saved_model_and_inference(tmp_path):
    _, sampled = _saved(tmp_path)
    script = '''
import sys
from pathlib import Path
from openghg_inversions.rhime import _standard_stages as stages
from openghg_inversions.rhime.specs import RhimeOutputSpec
from openghg_inversions.inference.sampling import RhimeSampler
root = Path(sys.argv[1])
def forbidden(*args, **kwargs):
    raise AssertionError("replay performed acquisition, graph construction or inference")
stages._build_prepared_model = forbidden
stages.materialize_pymc_inputs = forbidden
stages.RhimeMergedData.from_options = forbidden
RhimeSampler.sample = forbidden
stages.resolve_sampler = forbidden
result = stages.postprocess(root / "samples/sample-manifest.json", output=RhimeOutputSpec(
    output_format="inv_out", output_path=str(root / "outputs"), save_inversion_output=True))
assert result.sampler is None and result.model is None and result.model_build_result is None
assert result.run_spec.end_date == "2019-02-01"
assert result.output_metadata["build_and_sample_seconds"] == 2.5
assert result.output_metadata["sampler"]["draws"] == 10
'''
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, capture_output=True, text=True)
    saved = next((tmp_path / "outputs").glob("*inversion_output.nc"))
    loaded = InversionOutput.load(saved)
    np.testing.assert_array_equal(loaded.trace["posterior"]["x"], [[[1.0], [2.0]], [[10.0], [20.0]]])
    assert sampled.exists()


@pytest.mark.parametrize("corruption", ["schema", "missing", "contract", "posterior", "model"])
def test_bad_records_fail_without_graph_or_products(tmp_path, monkeypatch, corruption):
    _, sampled = _saved(tmp_path)
    record = json.loads(sampled.read_text())
    if corruption == "schema":
        record["schema_version"] = True
    elif corruption == "missing":
        del record["output_contract"]
    elif corruption == "contract":
        record["output_contract"]["state_dimension_mapping"] = {"trace": "region"}
    elif corruption == "posterior":
        record["posterior"] = "missing.nc"
    else:
        del record["model"]["use_bc"]
    sampled.write_text(json.dumps(record))
    monkeypatch.setattr(stages, "_build_prepared_model", lambda *a, **k: pytest.fail("graph construction"))
    with pytest.raises((ValueError, FileNotFoundError)):
        stages.postprocess(sampled, output=RhimeOutputSpec(output_path=str(tmp_path / "products")))
    assert not (tmp_path / "products").exists()


def test_changed_preparation_policy_fails_before_sampling(tmp_path, monkeypatch):
    preparation, _ = _saved(tmp_path)
    monkeypatch.setattr(stages, "sample_rhime_model", lambda *a, **k: pytest.fail("inference"))
    with pytest.raises(ValueError, match="conflicts with preparation"):
        stages.sample(preparation, model={"bc_freq": "monthly"}, output_dir=tmp_path / "new-samples")
    assert not (tmp_path / "new-samples").exists()


def test_prior_readiness_catches_graph_failure_but_not_save_failure(tmp_path, monkeypatch):
    preparation, _ = _saved(tmp_path)
    def fail(*args, **kwargs):
        raise ValueError("missing scientific input")
    monkeypatch.setattr(stages, "_build_prepared_model", fail)
    result = stages.prior_predictive(preparation, model={"mismatch_model": "fixed_error"},
                                    output_dir=tmp_path / "prior")
    assert result["status"] == "fail"
    assert "missing scientific input" in result["message"]
    with pytest.raises(FileNotFoundError):
        stages.prior_predictive(tmp_path / "absent.json", model={}, output_dir=tmp_path / "missing")


def test_missing_descriptive_sampler_metadata_does_not_select_defaults(tmp_path, monkeypatch):
    _, sampled = _saved(tmp_path)
    record = json.loads(sampled.read_text())
    del record["actual_inference"]
    sampled.write_text(json.dumps(record))
    monkeypatch.setattr(stages, "resolve_sampler", lambda *a, **k: pytest.fail("selected live sampler"))
    result = stages.postprocess(sampled, output=RhimeOutputSpec(output_format="inv_out", save_inversion_output=False))
    assert result.output_metadata["sampler"] is None
    assert result.sampler is None


def test_replay_refuses_existing_write_destination_but_allows_in_memory(tmp_path):
    _, sampled = _saved(tmp_path)
    output_dir = tmp_path / "products"
    output_dir.mkdir()
    with pytest.raises(FileExistsError):
        stages.postprocess(sampled, output=RhimeOutputSpec(output_path=str(output_dir)))
    result = stages.postprocess(sampled, output=RhimeOutputSpec(output_path=str(output_dir), save_inversion_output=False))
    assert result.inv_out is not None
    assert not list(output_dir.iterdir())


@pytest.mark.parametrize(("name", "value"), [
    ("species", None), ("domain", ""), ("start_date", []), ("end_date", 7),
    ("use_bc", "False"), ("flux_sources", "inventory"), ("flux_sources", []),
])
def test_malformed_preparation_facts_fail_at_reader(tmp_path, name, value):
    preparation, _ = _saved(tmp_path)
    record = json.loads(preparation.read_text())
    record["context"][name] = value
    preparation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=name):
        stages.load_preparation(preparation)


@pytest.mark.parametrize("name", ["species", "domain", "flux_sources", "use_bc"])
def test_saved_model_cannot_redefine_preparation_facts(tmp_path, name):
    preparation, sampled = _saved(tmp_path)
    record = json.loads(sampled.read_text())
    if name == "flux_sources":
        record["model"]["sectors"][0]["flux_source"] = "other-source"
    elif name == "use_bc":
        prep_record = json.loads(preparation.read_text())
        prep_record["context"]["use_bc"] = False
        preparation.write_text(json.dumps(prep_record))
        record["model"]["use_bc"] = True
    else:
        record["model"][name] = "other"
    sampled.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="preparation"):
        stages.postprocess(sampled, output=RhimeOutputSpec(output_format="none"))


def test_grouped_resource_paths_resolve_beside_source(tmp_path):
    config = tmp_path / "request.json"
    config.write_text(json.dumps({"data": {"basis_path": "basis.nc"},
                                  "output": {"country_file": "country.nc"}}))
    loaded = stages.load_stage_params(params_file=config)
    assert loaded["data"]["basis_path"] == tmp_path / "basis.nc"
    assert loaded["output"]["country_file"] == tmp_path / "country.nc"


@pytest.mark.parametrize("export", ["outside.nc", "outside.zarr", "prepared/basis.nc", "prepared/basis.zarr"])
def test_prepare_preserves_exact_basis_export(tmp_path, monkeypatch, export):
    from dataclasses import replace
    from openghg_inversions.rhime import standard
    config = RhimeConfig.from_params(_params(), multisector=False).data
    destination = tmp_path / "prepared"
    target = tmp_path / export
    config = replace(config, basis_output_path=target)
    monkeypatch.setattr(stages.RhimeMergedData, "from_options", lambda **kwargs: object())
    monkeypatch.setattr(stages, "resolve_basis_inputs", lambda data: {})
    monkeypatch.setattr(standard, "prepare_standard", lambda *args, **kwargs: _prepared())
    manifest = stages.prepare(config, output_dir=destination)
    assert manifest.is_file()
    assert target.exists()
    loaded, _ = stages.load_preparation(manifest)
    assert loaded.basis_artifact_path == str(target)


@pytest.mark.parametrize("evidence,status", [("finite", "pass"), ("nonfinite", "fail"), ("missing", "fail"), ("empty", "fail")])
def test_prior_readiness_requires_finite_declared_prediction(tmp_path, monkeypatch, evidence, status):
    from contextlib import nullcontext
    from types import SimpleNamespace
    preparation, _ = _saved(tmp_path)
    contract = OutputContract(variable_roles={"concentration": "observations"})
    monkeypatch.setattr(stages, "_build_prepared_model", lambda *a, **k: SimpleNamespace(
        model=nullcontext(), output_contract=contract))
    group = "prior" if evidence == "missing" else "prior_predictive"
    prior = make_trace(**{group: xr.Dataset({"observations": (("chain", "draw"),
                                                              [[np.nan if evidence == "nonfinite" else 1.]])})})
    if evidence == "empty":
        prior = prior.isel(draw=slice(0, 0))
    monkeypatch.setattr(stages.pm, "sample_prior_predictive", lambda *a, **k: prior)
    result = stages.prior_predictive(preparation, model={"mismatch_model": "fixed_error"},
                                    output_dir=tmp_path / "prior")
    assert result["status"] == status
    evidence_path = tmp_path / "prior" / "prior-predictive.nc"
    assert evidence_path.is_file()
    assert (tmp_path / "prior" / "prior-predictive-readiness.json").is_file()
    reopened = stages.load_trace(evidence_path)
    xr.testing.assert_identical(reopened, prior)
    if evidence == "empty":
        assert "no concentration draws" in result["message"]


@pytest.mark.parametrize("empty", [False, True])
def test_prior_serialization_errors_propagate(tmp_path, monkeypatch, empty):
    from contextlib import nullcontext
    from types import SimpleNamespace
    preparation, _ = _saved(tmp_path)
    monkeypatch.setattr(stages, "_build_prepared_model", lambda *a, **k: SimpleNamespace(
        model=nullcontext(), output_contract=OutputContract(variable_roles={"concentration": "y"})))
    prior = make_trace(prior_predictive=xr.Dataset({"y": (("chain", "draw"), [[1.]])}))
    if empty:
        prior = prior.isel(draw=slice(0, 0))
    monkeypatch.setattr(stages.pm, "sample_prior_predictive", lambda *a, **k: prior)
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(stages, "save_trace", fail)
    with pytest.raises(OSError, match="disk full"):
        stages.prior_predictive(preparation, model={"mismatch_model": "fixed_error"},
                                output_dir=tmp_path / "prior")
    assert not (tmp_path / "prior" / "prior-predictive-readiness.json").exists()


@pytest.mark.parametrize("sampler", [None, "unknown", ""])
def test_sample_rejects_invalid_live_sampler_before_loading(tmp_path, monkeypatch, sampler):
    monkeypatch.setattr(stages, "load_preparation", lambda *a, **k: pytest.fail("loaded preparation"))
    monkeypatch.setattr(stages, "sample_rhime_model", lambda *a, **k: pytest.fail("inference"))
    with pytest.raises(TypeError, match="sampler"):
        stages.sample(tmp_path / "missing.json", model={}, sampler=sampler, output_dir=tmp_path / "samples")
    assert not (tmp_path / "samples").exists()


def test_failed_sample_save_leaves_no_completed_manifest(tmp_path, monkeypatch):
    preparation, sampled = _saved(tmp_path)
    record = json.loads(sampled.read_text())
    assert record["inference_kind"] == "mcmc"
    assert record["model"]["likelihood"] == {"kind": "fixed_error", "options": {}}
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(stages, "save_trace", fail)
    with pytest.raises(OSError, match="disk full"):
        stages.save_samples(
            preparation, make_trace(posterior=xr.Dataset({"x": (("chain", "draw", "region"), [[[1.]]])})), model=stages._saved_model(record["model"]),
            output_contract=OutputContract.from_dict(record["output_contract"]), actual_inference={},
            build_and_sample_seconds=0., output_dir=tmp_path / "failed",
        )
    assert not (tmp_path / "failed" / "sample-manifest.json").exists()


@pytest.mark.parametrize("inference_kind", [None, "map", "nested"])
def test_replay_rejects_unknown_inference_kind_before_loading(tmp_path, monkeypatch, inference_kind):
    _, sampled = _saved(tmp_path)
    record = json.loads(sampled.read_text())
    record["inference_kind"] = inference_kind
    sampled.write_text(json.dumps(record))
    monkeypatch.setattr(stages, "load_preparation", lambda *a, **k: pytest.fail("loaded preparation"))
    with pytest.raises(ValueError, match="inference_kind"):
        stages.postprocess(sampled, output=RhimeOutputSpec(output_format="none"))


def test_independent_diagnosis_accepts_current_sample_manifest(tmp_path, monkeypatch):
    from openghg_inversions.rhime._stage_checks import diagnose_rhime_stage
    _, sampled = _saved(tmp_path)
    monkeypatch.setattr(stages, "_build_prepared_model", lambda *a, **k: pytest.fail("graph construction"))
    check = diagnose_rhime_stage(posterior=sampled.parent / "posterior.nc", sample_manifest=sampled,
                                 output_dir=tmp_path / "diagnosis")
    assert check["status"] in {"unknown", "fail"}
    assert check["measured_values"]["draws_per_chain"] == 2


def test_explicit_latent_only_trace_cannot_complete_handoff_or_replay(tmp_path):
    from openghg_inversions.serialization import save_trace
    preparation, sampled = _saved(tmp_path)
    record = json.loads(sampled.read_text())
    trace = make_trace(posterior=xr.Dataset({"x_latent": (("chain", "draw", "region"), [[[1.]]])}))
    with pytest.raises(ValueError, match="reconstruction state.*x"):
        stages.save_samples(
            preparation, trace, model=stages._saved_model(record["model"]),
            output_contract=OutputContract.from_dict(record["output_contract"]), actual_inference={},
            build_and_sample_seconds=0., output_dir=tmp_path / "incomplete",
        )
    assert not (tmp_path / "incomplete").exists()
    save_trace(trace, sampled.parent / record["posterior"])
    with pytest.raises(ValueError, match="reconstruction state.*x"):
        stages.postprocess(sampled, output=RhimeOutputSpec(output_format="none"))


@pytest.mark.parametrize("grouped_override", [False, True])
@pytest.mark.parametrize("canonical_file", [False, True])
def test_grouped_loader_merges_raw_aliases_before_translation(tmp_path, monkeypatch, grouped_override, canonical_file):
    config_dir = tmp_path / "config"
    invocation = tmp_path / "invocation"
    config_dir.mkdir()
    invocation.mkdir()
    source = config_dir / "choices.json"
    source.write_text(json.dumps({"output": {
        "output_path" if canonical_file else "outputpath": "file-products",
    }, "data": {"basis_path": "basis.nc"}}))
    overrides = {"outputpath": "override-products"}
    if grouped_override:
        overrides = {"output": overrides}
    monkeypatch.chdir(invocation)
    with pytest.warns(DeprecationWarning, match="Ignoring" if canonical_file else "deprecated"):
        loaded = stages.load_stage_params(params_file=source, overrides=overrides)
    expected = config_dir / "file-products" if canonical_file else invocation / "override-products"
    assert loaded["output"]["output_path"] == expected
    assert loaded["data"]["basis_path"] == config_dir / "basis.nc"


def test_grouped_loader_flat_canonical_override_wins_file_alias(tmp_path, monkeypatch):
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    source = tmp_path / "choices.json"
    source.write_text(json.dumps({"output": {"outputpath": "file-products"}}))
    monkeypatch.chdir(invocation)
    with pytest.warns(DeprecationWarning, match="Ignoring"):
        loaded = stages.load_stage_params(params_file=source, overrides={"output_path": "products"})
    assert loaded["output"]["output_path"] == invocation / "products"
