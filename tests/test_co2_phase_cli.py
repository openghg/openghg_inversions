"""Ordinary CO2 phases consume their own choices and one scientific handoff."""
from openghg_inversions.rhime.co2 import inference as co2_inference

import json

import pytest

from openghg_inversions.cli import main
from openghg_inversions.rhime.co2.execution import load_co2_params, resolve_co2_output
from openghg_inversions.rhime.co2.configuration import resolve_co2_model, resolve_co2_preparation


def test_inference_phase_ignores_inactive_resources_and_uses_current_sampler(tmp_path, monkeypatch, capsys):
    from openghg_inversions.rhime.co2 import handoffs

    manifest = tmp_path / "prepare-manifest.json"
    manifest.write_text(json.dumps({"schema": "openghg_inversions.co2.prepare"}))
    choices = tmp_path / "choices.json"
    choices.write_text(json.dumps({
        "likelihood": {"kind": "additive_sigma", "no_model_error": True},
        "sampling": {"draws": 7, "chains": 1},
        "prepared_inputs": {"path": 42}, "outputs": {"output_path": 42},
    }))
    calls = []

    def sample(preparation, *, model, sampler, output_dir):
        calls.append((preparation, model, sampler.draws, output_dir))
        return output_dir / "sample-manifest.json"

    monkeypatch.setattr(handoffs, "sample", sample)
    main(["sample", "--model", "co2", "--params-file", str(choices),
          "--preparation-manifest", str(manifest), "--output-dir", str(tmp_path / "sample")])
    assert calls[0][0] == str(manifest)
    assert calls[0][1]["likelihood"]["no_model_error"] is True
    assert calls[0][2:] == (7, tmp_path / "sample")
    assert "sample-manifest.json" in capsys.readouterr().out


def test_replay_cli_resolves_outputs_without_model_or_sampler(tmp_path, monkeypatch):
    from openghg_inversions.rhime.co2 import handoffs

    manifest = tmp_path / "sample-manifest.json"
    manifest.write_text(json.dumps({"schema": "openghg_inversions.co2.sample"}))
    calls = []
    monkeypatch.setattr(handoffs, "postprocess", lambda manifest, **kwargs: calls.append((manifest, kwargs)))
    main(["postprocess", "--model", "co2", "--sample-manifest", str(manifest),
          "--kwargs", '{"outputs":{"output_format":"basic","output_name":"later"}}',
          "--output-dir", str(tmp_path / "products")])
    output = calls[0][1]["output"]
    assert output.output_name == "later"
    assert output.output_path == tmp_path / "products"


def test_missing_co2_handoff_reports_the_file_without_legacy_config(tmp_path):
    with pytest.raises(SystemExit) as caught:
        main(["sample", "--model", "co2", "--preparation-manifest", str(tmp_path / "missing.json"),
              "--output-dir", str(tmp_path / "sample")])
    assert caught.value.code == 2
    assert not (tmp_path / "sample").exists()


def test_configured_full_rejects_mismatched_country_grid_before_materialization(tmp_path, monkeypatch):
    from examples.co2_handoffs import configuration, synthetic_inputs
    from openghg_inversions.rhime.co2 import execution
    import xarray as xr

    _, _, prepared, affine, countries = synthetic_inputs(tmp_path)
    with xr.open_dataset(countries) as source:
        mask = source.load()
    mask = mask.assign_coords(lon=mask.lon + 1)
    mask.to_netcdf(countries, mode="w")
    params = configuration(tmp_path / "products", countries)

    def forbidden(*args, **kwargs):
        raise AssertionError("Reporting incompatibility must fail before model materialization")

    monkeypatch.setattr(co2_inference, "materialize_co2_inputs", forbidden)
    with pytest.raises(ValueError, match="align"):
        execution.run_co2(params=params, prepared=prepared, affine_artifact=affine)
    assert not (tmp_path / "products").exists()


def test_co2_prior_choices_need_no_sampling_or_output_sections():
    choices = resolve_co2_model({
        "likelihood": {"kind": "additive_sigma", "no_model_error": True},
        "sampling": {"draws": "invalid inactive value"},
        "outputs": {"output_format": "invalid inactive value"},
    })
    assert choices["no_model_error"] is True


def test_co2_current_manifest_rejects_wrong_family_before_config(tmp_path):
    manifest = tmp_path / "prepare.json"
    manifest.write_text(json.dumps({"schema": "openghg_inversions.co2.prepare"}))
    with pytest.raises(ValueError, match="does not match"):
        main(["sample", "--preparation-manifest", str(manifest), "--output-dir", str(tmp_path)])


def test_co2_file_and_override_paths_use_one_invocation_base(tmp_path, monkeypatch):
    source = tmp_path / "config" / "choices.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"prepared_inputs": {"path": "prepared.nc"},
                                  "outputs": {"country_file": "countries.nc"}}))
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    monkeypatch.chdir(invocation)
    options = load_co2_params(params_file=source, overrides={"outputs": {"country_file": "other.nc"}})
    assert resolve_co2_preparation(options)["prepared_path"] == invocation / "prepared.nc"
    assert resolve_co2_output(options)["output"].country_file == invocation / "other.nc"


def test_diagnosis_accepts_one_co2_sample_manifest_without_preparation(tmp_path, monkeypatch):
    from openghg_inversions.rhime import _stage_checks
    from openghg_inversions.serialization import save_trace
    import numpy as np
    import xarray as xr

    posterior = tmp_path / "posterior.nc"
    save_trace(xr.DataTree.from_dict({"posterior": xr.Dataset({
        "state": (("chain", "draw"), np.arange(20.).reshape(2, 10)),
    })}), posterior)
    manifest = tmp_path / "sample-manifest.json"
    manifest.write_text(json.dumps({"schema": "openghg_inversions.co2.sample", "schema_version": 1,
                                   "producer": "openghg_inversions", "stage": "sample", "posterior": "posterior.nc"}))
    check = _stage_checks.diagnose_rhime_stage(sample_manifest=manifest, output_dir=tmp_path / "diagnose")
    assert check["status"] in {"fail", "unknown", "pass"}
    assert (tmp_path / "diagnose" / "sampler-convergence.json").exists()


@pytest.mark.parametrize("version", [True, "1", 2])
def test_phase_rejects_unsupported_format_versions(version):
    with pytest.raises(ValueError, match="format_version"):
        load_co2_params(params={"format_version": version})


def test_partial_component_override_preserves_the_other_choices():
    original = {"sampling": {"draws": 30, "tune": 11, "chains": 2, "random_seed": 73,
                             "sample_prior_predictive": 8},
                "outputs": {"output_format": "basic", "output_name": "retained", "save_trace": "trace.nc"}}
    override = {"sampling": {"draws": 10}, "outputs": {"output_name": "later"}}
    loaded = load_co2_params(params=original, overrides=override)
    assert loaded["sampling"] == {**original["sampling"], "draws": 10}
    assert loaded["outputs"] == {**original["outputs"], "output_name": "later"}
    assert original["sampling"]["draws"] == 30
    assert original["outputs"]["output_name"] == "retained"
    assert override["sampling"] == {"draws": 10}


def test_prior_cli_preserves_seed_without_resolving_inactive_mcmc_choices(tmp_path, monkeypatch):
    from openghg_inversions.rhime.co2 import handoffs

    manifest = tmp_path / "prepare.json"
    manifest.write_text(json.dumps({"schema": "openghg_inversions.co2.prepare"}))
    calls = []

    def prior(preparation, **kwargs):
        calls.append((preparation, kwargs))
        return {"status": "pass"}

    monkeypatch.setattr(handoffs, "prior_predictive", prior)
    main(["prior-predictive", "--model", "co2", "--preparation-manifest", str(manifest),
          "--output-dir", str(tmp_path / "check"), "--kwargs", json.dumps({
              "likelihood": {"kind": "additive_sigma", "no_model_error": True},
              "sampling": {"random_seed": 73, "draws": "inactive-invalid"},
          })])
    assert calls[0][1]["random_seed"] == 73
    assert calls[0][1]["model"] == {"likelihood": {"kind": "additive_sigma", "no_model_error": True}}


def test_unknown_co2_manifest_schema_fails_before_configuration_or_writes(tmp_path):
    manifest = tmp_path / "unknown.json"
    manifest.write_text(json.dumps({"schema": "unrelated.prepare"}))
    with pytest.raises(SystemExit) as caught:
        main(["sample", "--model", "co2", "--preparation-manifest", str(manifest),
              "--output-dir", str(tmp_path / "sample")])
    assert caught.value.code == 2
    assert not (tmp_path / "sample").exists()


@pytest.mark.parametrize("claims", [
    {"start_date": "2020-03-01", "end_date": "2020-04-01"},
    {"start_date": ""},
])
def test_full_rejects_positional_date_conflicts_with_imported_request(tmp_path, monkeypatch, claims):
    from examples.co2_handoffs import configuration, synthetic_inputs
    from openghg_inversions.rhime.co2 import execution

    _, _, prepared, _, countries = synthetic_inputs(tmp_path)
    params = configuration(tmp_path / "rejected", countries)
    params["outputs"] = {"output_format": "none"}

    def forbidden(*args, **kwargs):
        raise AssertionError("Conflicting request facts must fail before inference")

    monkeypatch.setattr(co2_inference, "materialize_co2_inputs", forbidden)
    with pytest.raises(ValueError, match="requested|conflict|start_date"):
        execution.run_co2(params=params, prepared=prepared, **claims)
    assert not (tmp_path / "rejected").exists()
