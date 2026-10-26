"""Output owners validate applicable values; preparation defers unused output values."""

from importlib import import_module
import json

import pytest

from openghg_inversions.cli import main
from tests.test_staged_workflow import _params


PHASE_OWNERS = {
    "standard": "openghg_inversions.rhime.stages",
    "multisector": "openghg_inversions.rhime.multisector_stages",
    "co2": "openghg_inversions.rhime.co2.handoffs",
    "co2_o2": "openghg_inversions.rhime.co2.co2_o2_handoffs",
}


@pytest.mark.parametrize("model", PHASE_OWNERS)
def test_postprocess_rejects_unknown_output_format_before_replay(tmp_path, monkeypatch, model):
    output_group = "output" if model in {"standard", "multisector"} else "outputs"
    choices = tmp_path / "choices.json"
    choices.write_text(json.dumps({output_group: {"output_format": "not-a-supported-format"}}))
    manifest = tmp_path / "sample.json"
    manifest.write_text(json.dumps({"schema": f"openghg_inversions.{model}.sample"}))
    destination = tmp_path / "unopened"
    monkeypatch.setattr(
        import_module(PHASE_OWNERS[model]), "postprocess",
        lambda *a, **k: pytest.fail("replayed saved science before validating output choices"),
    )

    with pytest.raises(ValueError, match="output"):
        main(["postprocess", "--model", model, "--params-file", str(choices),
              "--sample-manifest", str(manifest), "--output-dir", str(destination)])
    assert not destination.exists()


@pytest.mark.parametrize("model", PHASE_OWNERS)
def test_preparation_defers_known_inactive_output_values(tmp_path, monkeypatch, model):
    monkeypatch.chdir(tmp_path)
    rhime = model in {"standard", "multisector"}
    options = _params(flux_sources=["first", "second"] if model == "multisector" else ["first"]) if rhime else {
        "prepared_inputs": {"path": "borrowed-inputs.nc"},
    }
    options["output" if rhime else "outputs"] = {
        "output_format": "not-a-supported-format", "save_trace": 42,
    }
    choices = tmp_path / "choices.json"
    choices.write_text(json.dumps(options))
    destination = tmp_path / "preparation"
    calls = []

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return destination / "prepare-manifest.json"

    monkeypatch.setattr(import_module(PHASE_OWNERS[model]), "prepare", prepare)
    main(["prepare", "--model", model, "--params-file", str(choices), "--output-dir", str(destination)])

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs["output_dir"] == destination
    if rhime:
        assert args[0].site_options.sites == ("TAC",)
        assert args[0].flux_sources == tuple(options["flux_sources"])
    else:
        assert kwargs["prepared_path"] == tmp_path / "borrowed-inputs.nc"
    assert not destination.exists()
