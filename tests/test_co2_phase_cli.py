"""Ordinary CO2 phases consume their own choices and one scientific handoff."""

import json

import pytest

from openghg_inversions.cli import main
from openghg_inversions.rhime.co2.execution import load_co2_params, resolve_co2_phase_options


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
        calls.append((preparation, model.runner_kwargs, sampler.draws, output_dir))
        return output_dir / "sample-manifest.json"

    monkeypatch.setattr(handoffs, "sample", sample)
    main(["sample", "--model", "co2", "--params-file", str(choices),
          "--preparation-manifest", str(manifest), "--output-dir", str(tmp_path / "sample")])
    assert calls[0][0] == str(manifest)
    assert calls[0][1]["no_model_error"] is True
    assert calls[0][2:] == (7, tmp_path / "sample")
    assert "sample-manifest.json" in capsys.readouterr().out


def test_replay_cli_resolves_outputs_without_model_or_sampler(tmp_path, monkeypatch):
    from openghg_inversions.rhime.co2 import handoffs

    manifest = tmp_path / "sample-manifest.json"
    manifest.write_text(json.dumps({"schema": "openghg_inversions.co2.sample"}))
    calls = []
    monkeypatch.setattr(handoffs, "postprocess", lambda manifest, **kwargs: calls.append((manifest, kwargs)))
    main(["postprocess", "--model", "co2", "--sample-manifest", str(manifest),
          "--kwargs", '{"outputs":{"output_format":"basic","output_name":"later","reconstruction_path":42}}',
          "--output-dir", str(tmp_path / "products")])
    output = calls[0][1]["output"]
    assert output.output_name == "later"
    assert output.output_path == tmp_path / "products"


def test_missing_co2_handoff_reports_the_file_without_legacy_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.json"):
        main(["sample", "--model", "co2", "--preparation-manifest", str(tmp_path / "missing.json"),
              "--output-dir", str(tmp_path / "sample")])
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

    monkeypatch.setattr(execution, "materialize_co2_inputs", forbidden)
    with pytest.raises(ValueError, match="align"):
        execution.run_co2_from_config(params=params, prepared=prepared, affine_artifact=affine)
    assert not (tmp_path / "products").exists()


def test_co2_prior_choices_need_no_sampling_or_output_sections():
    choices = resolve_co2_phase_options({
        "likelihood": {"kind": "additive_sigma", "no_model_error": True},
        "sampling": {"draws": "invalid inactive value"},
        "outputs": {"output_format": "invalid inactive value"},
    }, phase="prior")
    assert choices["model"].runner_kwargs["no_model_error"] is True


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
    assert options["prepared_inputs"]["path"] == invocation / "prepared.nc"
    assert options["outputs"]["country_file"] == invocation / "other.nc"


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
        resolve_co2_phase_options({"format_version": version}, phase="prepare")
