"""Phase selection resolves only the active configuration owners."""

import json
from pathlib import Path

import pytest

from openghg_inversions.cli import main
from openghg_inversions.rhime import stages


def test_sample_cli_accepts_one_manifest_and_ignores_inactive_values(tmp_path, monkeypatch, capsys):
    choices = tmp_path / "choices.json"
    choices.write_text(json.dumps({"data": {"sites": 42}, "model": {"use_bc": False},
                                   "sampler": {"draws": 7}, "output": {"output_format": "invalid"}}))
    calls = []

    def sample(manifest, *, model, sampler, output_dir):
        calls.append((manifest, model, sampler.draws, output_dir))
        return output_dir / "sample-manifest.json"

    monkeypatch.setattr(stages, "sample", sample)
    main(["sample", "--params-file", str(choices), "--preparation-manifest", "saved.json",
          "--kwargs", '{"draws":9,"use_bc":true}', "--output-dir", str(tmp_path / "sample")])
    assert calls == [("saved.json", {"use_bc": True}, 9, tmp_path / "sample")]
    assert "sample-manifest.json" in capsys.readouterr().out


def test_postprocess_cli_needs_only_sample_and_output_choices(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(stages, "postprocess", lambda manifest, *, output: calls.append((manifest, output)))
    main(["postprocess", "--sample-manifest", "samples.json", "--output-dir", str(tmp_path / "products"),
          "--kwargs", '{"output_format":"inv_out","output_name":"again"}'])
    assert calls[0][0] == "samples.json"
    assert calls[0][1].output_name == "again"
    assert Path(calls[0][1].output_path) == tmp_path / "products"


def test_phase_cli_rejects_explicit_wrong_phase_override(tmp_path):
    with pytest.raises(ValueError, match="for this phase"):
        main(["sample", "--preparation-manifest", "missing.json", "--output-dir", str(tmp_path / "sample"),
              "--kwargs", '{"filters":"noon"}'])


@pytest.mark.parametrize("model", ["standard", "multisector", "co2"])
@pytest.mark.parametrize("phase,retired", [
    ("prior-predictive", "--prepared-inputs"), ("sample", "--prepared-inputs"),
    ("postprocess", "--prepared-inputs"), ("postprocess", "--posterior"),
    ("postprocess", "--preparation-manifest"),
])
def test_cli_rejects_retired_artifact_arguments_before_execution(tmp_path, capsys, model, phase, retired):
    manifest_arg = "--sample-manifest" if phase == "postprocess" else "--preparation-manifest"
    with pytest.raises(SystemExit) as error:
        main([phase, "--model", model, manifest_arg, "saved.json", retired, "inputs.nc",
              "--output-dir", str(tmp_path / "unopened")])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not (tmp_path / "unopened").exists()


def test_prior_cli_strict_preserves_scientific_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "prior_predictive", lambda *args, **kwargs: {"status": "fail"})
    with pytest.raises(SystemExit) as exc:
        main(["prior-predictive", "--preparation-manifest", "prepared.json",
              "--output-dir", str(tmp_path / "readiness"), "--strict"])
    assert exc.value.code == 1


def test_sample_cli_translates_alias_before_phase_claim(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(stages, "sample", lambda manifest, **kwargs: calls.append(kwargs))
    with pytest.warns(DeprecationWarning):
        main(["sample", "--preparation-manifest", "prepared.json", "--output-dir", str(tmp_path / "sample"),
              "--kwargs", '{"emissions_name":["inventory"]}'])
    assert calls[0]["model"]["flux_sources"] == ["inventory"]


def test_saved_family_mismatch_fails_before_config_resolution(tmp_path):
    manifest = tmp_path / "prepared.json"
    manifest.write_text(json.dumps({"schema": "openghg_inversions.standard.prepare"}))
    with pytest.raises(ValueError, match="does not match"):
        main(["sample", "--model", "co2", "--preparation-manifest", str(manifest),
              "--output-dir", str(tmp_path / "sample")])


def test_phase_alias_overrides_preserve_canonical_file_precedence(tmp_path, monkeypatch):
    choices = tmp_path / "choices.json"
    choices.write_text(json.dumps({"x_prior": {"pdf": "normal", "sigma": 0.1}}))
    calls = []
    monkeypatch.setattr(stages, "sample", lambda manifest, **kwargs: calls.append(kwargs))
    with pytest.warns(DeprecationWarning, match="Ignoring"):
        main(["sample", "--params-file", str(choices), "--preparation-manifest", "prepared.json",
              "--output-dir", str(tmp_path / "samples"),
              "--kwargs", '{"xprior":{"pdf":"normal","sigma":0.5}}'])
    assert calls[0]["model"]["x_prior"]["sigma"] == 0.1


def test_phase_reports_all_unsuitable_override_keys(tmp_path):
    with pytest.raises(ValueError) as caught:
        main(["sample", "--preparation-manifest", "missing.json", "--output-dir", str(tmp_path),
              "--kwargs", '{"filters":"noon","unknown":1,"outputname":"wrong"}'])
    for name in ("filters", "unknown", "outputname"):
        assert name in str(caught.value)


def test_phase_paths_use_invocation_directory(tmp_path, monkeypatch):
    from argparse import Namespace
    from openghg_inversions.cli import _rhime_phase_options

    config_dir = tmp_path / "config"
    invocation = tmp_path / "invocation"
    config_dir.mkdir()
    invocation.mkdir()
    choices = config_dir / "choices.json"
    choices.write_text(json.dumps({"data": {"basis_path": "operator.nc", "basis_output_path": "old.nc"},
                                   "sampler": {"draws": "invalid"}}))
    monkeypatch.chdir(invocation)
    args = Namespace(config=None, params_file=str(choices), kwargs={"basis_output_path": "../exports/basis.nc"})
    result = _rhime_phase_options(args, phase="prepare")
    assert result["data"]["basis_path"] == invocation / "operator.nc"
    assert result["data"]["basis_output_path"] == tmp_path / "exports/basis.nc"
    assert result["sampler"] == {}


def test_phase_aliases_do_not_import_hbmcmc(tmp_path, monkeypatch):
    import builtins
    from argparse import Namespace
    from openghg_inversions.cli import _rhime_phase_options

    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        assert "hbmcmc" not in name
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    args = Namespace(config=None, params_file=None, kwargs={"xprior": {"pdf": "normal", "sigma": 0.2}})
    with pytest.warns(DeprecationWarning):
        result = _rhime_phase_options(args, phase="sample")
    assert result["model"]["x_prior"]["sigma"] == 0.2


def test_flat_saved_model_claims_are_not_discarded(tmp_path):
    from argparse import Namespace
    from openghg_inversions.cli import _rhime_phase_options

    path = tmp_path / "choices.json"
    claims = {"use_bc": False, "bc_freq": "monthly", "mismatch_model": "fixed_error"}
    path.write_text(json.dumps(claims))
    args = Namespace(config=None, params_file=str(path), kwargs=None)
    flat = _rhime_phase_options(args, phase="sample")
    path.write_text(json.dumps({"model": claims}))
    assert flat == _rhime_phase_options(args, phase="sample")
    assert flat["model"] == claims


def test_prior_readiness_rejects_explicit_sampler_overrides(tmp_path):
    with pytest.raises(ValueError) as caught:
        main(["prior-predictive", "--preparation-manifest", "missing.json", "--output-dir", str(tmp_path),
              "--kwargs", '{"tune":99,"nuts_sampler":"numpyro"}'])
    assert "tune" in str(caught.value) and "nuts_sampler" in str(caught.value)


@pytest.mark.parametrize("grouped", [False, True])
def test_phase_preserves_resource_migration_guidance_in_aggregate_error(grouped):
    from argparse import Namespace
    from openghg_inversions.cli import _rhime_phase_options
    from openghg_inversions.rhime._runner_compat import normalise_param_aliases

    removed = {"country_directory": "old", "bc_basis_case": "old", "bc_basis_directory": "old"}
    kwargs = {"data": removed} if grouped else dict(removed)
    kwargs.update(unknown=1, draws=10)
    args = Namespace(config=None, params_file=None, kwargs=kwargs)
    with pytest.raises(ValueError) as caught:
        _rhime_phase_options(args, phase="prepare")
    message = str(caught.value)
    for name in (*removed, "unknown", "draws"):
        assert name in message
    with pytest.raises(ValueError) as full:
        normalise_param_aliases(removed)
    assert str(full.value) in message
    for replacement in ("country_file", "landsea_path", "outer_regions_path", "bc_basis_paths", "HBMCMC"):
        assert replacement in message


def test_phase_errors_explain_every_removed_option():
    from argparse import Namespace
    from openghg_inversions.cli import _rhime_phase_options

    args = Namespace(config=None, params_file=None,
                     kwargs={"mcmc_type": "nuts", "calculate_min_error": True, "reparameterise_log_normal": True})
    with pytest.raises(ValueError) as caught:
        _rhime_phase_options(args, phase="sample")
    for replacement in ("nuts_sampler", "`min_error`", "reparameterise"):
        assert replacement in str(caught.value)
