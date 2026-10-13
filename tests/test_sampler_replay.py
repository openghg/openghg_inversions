"""Saved outputs describe completed inference without running it again."""

from dataclasses import replace
import json

import pytest
import xarray as xr

from openghg_inversions.rhime import _standard_stages as stages
from openghg_inversions.rhime.outputs import RhimeResult, _make_inversion_output
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import RhimeModelSpec, RhimeOutputSpec, RhimeRunSpec
from test_staged_workflow import _prepared, _saved_output_binding


def _forbidden(*args, **kwargs):
    raise AssertionError("Replay must not construct a model or run inference.")


def test_result_default_does_not_construct_a_sampler(monkeypatch):
    model = RhimeModelSpec(species="ch4", domain="EUROPE", sectors=())
    output = RhimeOutputSpec(output_format="none")
    run = RhimeRunSpec("2019-01-01", "2019-02-01", ("TAC",), ("1h",), model, output)
    monkeypatch.setattr(RhimeSampler, "__init__", _forbidden)
    result = RhimeResult(run, model, output, xr.Dataset(), xr.DataTree())
    assert result.sampler is None
    assert "sampler" not in result.output_metadata


def test_saved_sampler_metadata_matches_live_output_despite_current_defaults(tmp_path, monkeypatch):
    arguments = _saved_output_binding(tmp_path)
    sampler = RhimeSampler(draws=17, burn=2, tune=13, chains=3, nuts_sampler="numpyro")
    path = arguments["sample_manifest"]
    manifest = json.loads(path.read_text())
    recorded = json.loads(json.dumps({name: getattr(sampler, name) for name in sampler.__slots__}))
    # Runtime-only or future sampler options are descriptive provenance at replay.
    recorded["unavailable_runtime_plugin"] = {"module": "not_installed"}
    manifest["effective_configuration"]["sampler"] = recorded
    path.write_text(json.dumps(manifest))
    with monkeypatch.context() as patch:
        patch.setattr(RhimeSampler, "sample", _forbidden)
        patch.setattr(stages, "_build_prepared_model", _forbidden)
        replay = stages.postprocess_rhime_stage(**arguments)
    live = replace(replay, sampler=sampler, output_metadata={})
    assert live.sampler is sampler
    expected = {"draws": 17, "burn": 2, "tune": 13, "chains": 3, "nuts_sampler": "numpyro"}
    assert live.output_metadata["sampler"] == replay.output_metadata["sampler"] == expected
    assert replay.sampler is None
    sampler.draws = 900
    live_output = _make_inversion_output(result=live, prepared=_prepared())
    assert live_output.output_metadata["sampler"] == replay.inv_out.output_metadata["sampler"] == expected
    replay_manifest = json.loads((arguments["output_dir"] / "postprocess-manifest.json").read_text())
    assert replay_manifest["effective_configuration"]["sampler"] == recorded


@pytest.mark.parametrize(
    "recorded",
    [None, {}, "unavailable", {"draws": "invalid"}, {"tune": []},
     {"chains": None}, {"burn": "inf"}, {"nuts_sampler": "unavailable_backend"}],
)
def test_unavailable_sampler_description_does_not_block_products(tmp_path, monkeypatch, recorded):
    arguments = _saved_output_binding(tmp_path)
    path = arguments["sample_manifest"]
    manifest = json.loads(path.read_text())
    if recorded is None:
        del manifest["effective_configuration"]["sampler"]
    else:
        manifest["effective_configuration"]["sampler"] = recorded
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(RhimeSampler, "sample", _forbidden)
    monkeypatch.setattr(stages, "_build_prepared_model", _forbidden)
    replay = stages.postprocess_rhime_stage(**arguments)
    assert replay.sampler is None
    assert replay.inv_out.output_metadata["sampler"] == {}
    assert list(arguments["output_dir"].glob("*.nc"))
    output = json.loads((arguments["output_dir"] / "postprocess-manifest.json").read_text())
    assert output["effective_configuration"]["sampler"] == recorded


def test_partial_sampler_description_does_not_invent_defaults(tmp_path, monkeypatch):
    arguments = _saved_output_binding(tmp_path)
    path = arguments["sample_manifest"]
    manifest = json.loads(path.read_text())
    manifest["effective_configuration"]["sampler"] = {"draws": 17, "burn": 2}
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(RhimeSampler, "sample", _forbidden)
    monkeypatch.setattr(stages, "_build_prepared_model", _forbidden)
    replay = stages.postprocess_rhime_stage(**arguments)
    assert replay.inv_out.output_metadata["sampler"] == {"draws": 17, "burn": 2}
