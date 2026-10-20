"""Multisector checkpoint boundaries, readiness and manifest-only CLI phases."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pymc
import pytest
import xarray as xr

from examples.multisector_handoffs import model_options, synthetic_inputs
from openghg_inversions import serialization
from openghg_inversions.cli import main
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime import multisector_stages as stages
from openghg_inversions.rhime.params import DataConfig, resolve_model_options
from openghg_inversions.rhime.specs import RhimeOutputSpec
from tests.helpers import make_trace


@pytest.fixture
def preparation(tmp_path):
    merged, options, _, _, _ = synthetic_inputs(tmp_path / "inputs")
    data = DataConfig.from_params({**options, "sites": ["TAC", "MHD"],
                                  "averaging_period": ["1h", "3h"]}, multisector=True)
    manifest = stages.prepare(data, merged_data=merged, output_dir=tmp_path / "prepared")
    return manifest, stages.load_preparation(manifest)


def _trace_and_contract(prepared):
    posterior = make_trace(posterior=xr.Dataset(
        {"x_biosphere": (("chain", "draw", "state"), np.ones((2, 3, 1))),
         "x_fossil": (("chain", "draw", "state"), np.full((2, 3, 1), 1.2))},
        coords={"chain": [0, 1], "draw": [0, 1, 2], "state": prepared.inv_inputs.state},
    ))
    posterior.attrs["rhime_priors"] = json.dumps({
        "x_biosphere": {"pdf": "normal", "mu": .9, "sigma": .2},
        "x_fossil": {"pdf": "normal", "mu": 1.1, "sigma": .2},
    })
    contract = OutputContract(
        variable_roles={"flux_scale:biosphere": "x_biosphere", "flux_scale:fossil": "x_fossil"},
        supported_output_formats=("none", "inv_out"),
    )
    return posterior, contract


def _save_sample(tmp_path, preparation):
    manifest, (prepared, data, generation) = preparation
    posterior, contract = _trace_and_contract(prepared)
    return stages.save_samples(
        manifest, posterior, output_contract=contract,
        model=resolve_model_options(model_options(.2), data_config=data),
        actual_inference={"draws": 3, "chains": 2}, output_dir=tmp_path / "samples",
        preparation_generation=generation,
    )


def test_preparation_preserves_retained_subset_and_original_window(preparation):
    manifest, (prepared, data, generation) = preparation
    assert manifest.is_absolute() and generation
    assert prepared.sites == ("TAC",)
    assert data.site_options.sites == ("TAC", "MHD")
    assert data.end_date == "2019-02-01"
    assert tuple(prepared.inv_inputs.H.source.values) == prepared.basis_functions.source_labels
    assert prepared.basis_functions.source_labels == ("inventory", "supplemental")
    assert data.flux_sources == ("supplemental", "inventory")
    record = json.loads(manifest.read_text())
    assert record["prepared_inputs"] == "prepared-inputs.nc"
    assert "configuration_identity" not in record and "model" not in record


def test_external_preparation_reference_resolves_beside_manifest(preparation, tmp_path, monkeypatch):
    manifest, (prepared, _, _) = preparation
    record = json.loads(manifest.read_text())
    record["prepared_inputs"] = str(manifest.parent / record["prepared_inputs"])
    manifest.write_text(json.dumps(record))
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setenv("RUN_ROOT", str(tmp_path / "unrelated"))
    restored, _, _ = stages.load_preparation(manifest)
    xr.testing.assert_identical(restored.inv_inputs, prepared.inv_inputs)


def test_preparation_generation_replacement_invalidates_samples(preparation, tmp_path):
    sampled = _save_sample(tmp_path, preparation)
    manifest, (prepared, data, original) = preparation
    stages.save_preparation(prepared, data_config=data, output_dir=manifest.parent)
    with pytest.raises(ValueError, match="was replaced"):
        stages.load_samples(sampled)
    posterior, contract = _trace_and_contract(prepared)
    with pytest.raises(ValueError, match="was replaced"):
        stages.save_samples(
            manifest, posterior, output_contract=contract,
            model=resolve_model_options(model_options(.2), data_config=data),
            actual_inference={}, output_dir=tmp_path / "stale", preparation_generation=original,
        )
    assert not (tmp_path / "stale").exists()


def test_failed_preparation_save_invalidates_old_completion_marker(preparation, monkeypatch):
    manifest, (prepared, data, _) = preparation
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(stages.RhimePreparedInputs, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        stages.save_preparation(prepared, data_config=data, output_dir=manifest.parent)
    assert not manifest.exists()


def test_wrong_family_fails_before_numerical_load(preparation, monkeypatch):
    manifest, _ = preparation
    record = json.loads(manifest.read_text())
    record["data"]["split_by_sectors"] = False
    manifest.write_text(json.dumps(record))
    monkeypatch.setattr(stages.RhimePreparedInputs, "load", lambda *a, **k: pytest.fail("loaded numerical data"))
    with pytest.raises(ValueError, match="split_by_sectors"):
        stages.load_preparation(manifest)


def test_direct_wrong_family_fails_before_acquisition(preparation, tmp_path, monkeypatch):
    _, (_, data, _) = preparation
    monkeypatch.setattr(stages, "acquire_merged_data", lambda *a, **k: pytest.fail("acquired data"))
    with pytest.raises(ValueError, match="split_by_sectors=True"):
        stages.prepare(replace(data, split_by_sectors=False), output_dir=tmp_path / "wrong")


@pytest.mark.parametrize("contents", [None, "not a prepared NetCDF"])
def test_preparation_uses_ordinary_numerical_loader(preparation, contents):
    manifest, _ = preparation
    payload = manifest.parent / "prepared-inputs.nc"
    if contents is None:
        payload.unlink()
    else:
        payload.write_text(contents)
    with pytest.raises((OSError, ValueError)):
        stages.load_preparation(manifest)


def test_external_sample_save_requires_declared_roles(preparation, tmp_path):
    manifest, (prepared, data, generation) = preparation
    posterior, contract = _trace_and_contract(prepared)
    posterior["posterior"] = posterior["posterior"].to_dataset().drop_vars("x_fossil")
    with pytest.raises(ValueError, match="x_fossil"):
        stages.save_samples(
            manifest, posterior, output_contract=contract,
            model=resolve_model_options(model_options(.2), data_config=data), actual_inference={},
            output_dir=tmp_path / "invalid", preparation_generation=generation,
        )
    assert not (tmp_path / "invalid").exists()


def test_replay_uses_recorded_inference_with_no_live_sampler(preparation, tmp_path, monkeypatch):
    sampled = _save_sample(tmp_path, preparation)
    monkeypatch.setattr(stages, "resolve_sampler", lambda *a, **k: pytest.fail("fresh sampler"))
    monkeypatch.setattr(stages, "build_multisector_model", lambda *a, **k: pytest.fail("built graph"))
    result = stages.postprocess(sampled, output=RhimeOutputSpec(output_format="none"))
    assert result.output_metadata["sampler"] == {"draws": 3, "chains": 2}
    assert result.run_metadata["end_date"] == "2019-02-01" and result.run_metadata["sites"] == ["TAC"]


def test_readiness_construction_failure_becomes_failed_check(preparation, tmp_path, monkeypatch):
    manifest, _ = preparation
    def fail(*args, **kwargs):
        raise ValueError("missing configured prior flux")
    monkeypatch.setattr(stages, "materialize_multisector_inputs", fail)
    report = stages.prior_predictive(manifest, model=model_options(.2), output_dir=tmp_path / "ready", draws=5)
    assert report["status"] == "fail" and "missing configured prior flux" in report["message"]
    assert (tmp_path / "ready" / "prior-predictive-readiness.json").exists()


@pytest.mark.parametrize("values,status", [([], "fail"), ([np.nan], "fail"), ([1.0], "pass")])
def test_readiness_classifies_predictive_evidence(preparation, tmp_path, monkeypatch, values, status):
    manifest, _ = preparation
    prior = make_trace(prior_predictive=xr.Dataset({"y": (("chain", "draw", "nmeasure"), [[values]])}))
    monkeypatch.setattr(stages, "materialize_multisector_inputs", lambda *a: (None, {}))
    monkeypatch.setattr(stages, "build_multisector_model", lambda *a, **k: SimpleNamespace(
        model=nullcontext(), output_contract=OutputContract(variable_roles={"concentration": "y"})))
    monkeypatch.setattr(pymc, "sample_prior_predictive", lambda *a, **k: prior)
    result = stages.prior_predictive(manifest, model=model_options(.2), output_dir=tmp_path / "ready", draws=1)
    assert result["status"] == status
    assert (tmp_path / "ready" / "prior-predictive.nc").exists()


def test_readiness_serialization_errors_propagate(preparation, tmp_path, monkeypatch):
    manifest, _ = preparation
    prior = make_trace(prior_predictive=xr.Dataset({"y": (("chain", "draw"), [[1.]])}))
    monkeypatch.setattr(stages, "materialize_multisector_inputs", lambda *a: (None, {}))
    monkeypatch.setattr(stages, "build_multisector_model", lambda *a, **k: SimpleNamespace(
        model=nullcontext(), output_contract=OutputContract(variable_roles={"concentration": "y"})))
    monkeypatch.setattr(pymc, "sample_prior_predictive", lambda *a, **k: prior)
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(serialization, "save_trace", fail)
    with pytest.raises(OSError, match="disk full"):
        stages.prior_predictive(manifest, model=model_options(.2), output_dir=tmp_path / "ready")
    assert not (tmp_path / "ready" / "prior-predictive-readiness.json").exists()


@pytest.mark.parametrize("draws", [0, -1, True])
def test_readiness_invalid_draws_fail_before_loading(tmp_path, draws):
    with pytest.raises(ValueError, match="draws must be a positive integer"):
        stages.prior_predictive(tmp_path / "missing.json", model={}, output_dir=tmp_path / "ready", draws=draws)


def test_phase_cli_uses_one_manifest_and_only_active_options(tmp_path, monkeypatch, capsys):
    choices = tmp_path / "choices.json"
    choices.write_text(json.dumps({"data": {"sites": 42}, "model": {"use_bc": False},
                                   "sampler": {"draws": 7}, "output": {"output_format": "invalid"}}))
    calls = []
    monkeypatch.setattr(stages, "sample", lambda manifest, **kw: calls.append((manifest, kw)) or "samples.json")
    main(["sample", "--model", "multisector", "--params-file", str(choices),
          "--preparation-manifest", "saved.json", "--kwargs", '{"draws":9}',
          "--output-dir", str(tmp_path / "sample")])
    assert calls[0][0] == "saved.json" and calls[0][1]["model"] == {"use_bc": False}
    assert calls[0][1]["sampler"].draws == 9
    assert "samples.json" in capsys.readouterr().out


def test_postprocess_cli_requires_only_current_output_choices(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(stages, "postprocess", lambda manifest, **kw: calls.append((manifest, kw)))
    main(["postprocess", "--model", "multisector", "--sample-manifest", "sampled.json",
          "--output-dir", str(tmp_path / "products"), "--kwargs", '{"output_format":"inv_out","output_name":"again"}'])
    assert calls[0][0] == "sampled.json" and calls[0][1]["output"].output_name == "again"


def test_multisector_cli_rejects_retired_artifact_arguments(tmp_path):
    with pytest.raises(ValueError, match="one manifest"):
        main(["sample", "--model", "multisector", "--preparation-manifest", "saved.json",
              "--prepared-inputs", "inputs.nc", "--output-dir", str(tmp_path)])


def test_multisector_cli_strict_readiness_keeps_failure_status(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "prior_predictive", lambda *a, **k: {"status": "fail"})
    with pytest.raises(SystemExit) as error:
        main(["prior-predictive", "--model", "multisector", "--preparation-manifest", "saved.json",
              "--output-dir", str(tmp_path), "--strict"])
    assert error.value.code == 1


@pytest.mark.parametrize("selected,saved", [("multisector", "standard"), ("standard", "multisector"), ("co2", "multisector")])
def test_cli_rejects_mismatched_family_before_resolving_config(tmp_path, selected, saved):
    manifest = tmp_path / "prepared.json"
    manifest.write_text(json.dumps({"schema": f"openghg_inversions.{saved}.prepare"}))
    with pytest.raises(ValueError, match="does not match"):
        main(["sample", "--model", selected, "--preparation-manifest", str(manifest),
              "--output-dir", str(tmp_path / "sample")])


def test_prepare_cli_resolves_data_without_other_phase_choices(tmp_path, monkeypatch, capsys):
    merged, options, _, _, _ = synthetic_inputs(tmp_path / "inputs")
    choices = tmp_path / "prepare.json"
    choices.write_text(json.dumps({"data": options, "model": {"mismatch_model": "invalid"},
                                   "sampler": {"draws": "invalid"}, "output": {"output_format": "invalid"}}))
    calls = []
    def prepare(data, *, output_dir):
        calls.append((data, output_dir))
        return output_dir / "prepare-manifest.json"
    monkeypatch.setattr(stages, "prepare", prepare)
    main(["prepare", "--model", "multisector", "--params-file", str(choices),
          "--output-dir", str(tmp_path / "checkpoint")])
    assert calls[0][0].split_by_sectors
    assert calls[0][0].flux_sources == ("supplemental", "inventory")
    assert "prepare-manifest.json" in capsys.readouterr().out
    assert merged.split_by_sectors


@pytest.mark.parametrize("corruption", ["schema", "version", "generation", "missing_posterior", "contract"])
def test_invalid_sample_cannot_fall_back_to_graph(preparation, tmp_path, monkeypatch, corruption):
    sampled = _save_sample(tmp_path, preparation)
    record = json.loads(sampled.read_text())
    if corruption == "schema":
        record["schema"] = "openghg_inversions.standard.sample"
    elif corruption == "version":
        record["schema_version"] = 2
    elif corruption == "generation":
        record["preparation_generation"] = ""
    elif corruption == "missing_posterior":
        (sampled.parent / record["posterior"]).unlink()
    else:
        record["output_contract"]["variable_roles"]["flux_scale:fossil"] = "missing_flux_state"
    sampled.write_text(json.dumps(record))
    monkeypatch.setattr(stages, "build_multisector_model", lambda *a, **k: pytest.fail("graph fallback"))
    with pytest.raises((ValueError, FileNotFoundError)):
        stages.postprocess(sampled, output=RhimeOutputSpec(output_path=tmp_path / "invalid-products"))
    assert not (tmp_path / "invalid-products").exists()


@pytest.mark.parametrize("export", ["exports/basis.nc", "exports/basis.zarr", "prepared/basis.nc", "prepared/basis.zarr"])
def test_basis_export_keeps_exact_destination_and_prepared_source_flux(tmp_path, export):
    from openghg_inversions.basis import BasisFunctions
    merged, options, flux, _, _ = synthetic_inputs(tmp_path / "inputs")
    export_path = tmp_path / export
    data = DataConfig.from_params({**options, "basis_output_path": export_path}, multisector=True)
    manifest = stages.prepare(data, merged_data=merged, output_dir=tmp_path / "prepared")
    prepared, recorded, _ = stages.load_preparation(manifest)
    assert recorded.basis_output_path == export_path
    saved_basis = BasisFunctions.load(export_path)
    for source, field in flux.items():
        np.testing.assert_allclose(saved_basis.flux.sel(source=source), field)
    xr.testing.assert_equal(saved_basis.flux, prepared.basis_functions.flux)
    assert not (tmp_path / "prepared" / "prepared").exists()


@pytest.mark.parametrize("sampler", [None, "invalid"])
def test_saved_sampling_rejects_invalid_sampler_before_loading(tmp_path, sampler):
    with pytest.raises((TypeError, ValueError), match="sampler"):
        stages.sample(tmp_path / "missing.json", model={}, sampler=sampler, output_dir=tmp_path / "samples")


@pytest.mark.parametrize("schema", [None, 42])
def test_cli_leaves_malformed_schema_to_manifest_loader(tmp_path, schema):
    manifest = tmp_path / "prepared.json"
    manifest.write_text(json.dumps({"schema": schema}))
    with pytest.raises(ValueError, match="version-1 manifest"):
        main(["sample", "--model", "multisector", "--preparation-manifest", str(manifest),
              "--output-dir", str(tmp_path / "sample")])
