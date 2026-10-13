"""Phase selection resolves only the active configuration owners."""

import json
from pathlib import Path

import pytest

from openghg_inversions.cli import main
from openghg_inversions.rhime import stages


def test_sample_cli_accepts_one_manifest_and_ignores_inactive_values(tmp_path, monkeypatch, capsys):
    choices = tmp_path / "choices.json"
    choices.write_text(json.dumps({"preparation": {"sites": 42}, "model": {"use_bc": False},
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
    with pytest.raises(ValueError, match="do not belong"):
        main(["sample", "--preparation-manifest", "missing.json", "--output-dir", str(tmp_path / "sample"),
              "--kwargs", '{"filters":"daily_median"}'])


def test_standard_cli_rejects_retired_artifact_arguments(tmp_path):
    with pytest.raises(ValueError, match="one manifest"):
        main(["sample", "--preparation-manifest", "saved.json", "--prepared-inputs", "inputs.nc",
              "--output-dir", str(tmp_path / "sample")])


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
    manifest.write_text(json.dumps({"family": "standard"}))
    with pytest.raises(ValueError, match="does not match"):
        main(["sample", "--model", "co2", "--preparation-manifest", str(manifest),
              "--output-dir", str(tmp_path / "sample")])
