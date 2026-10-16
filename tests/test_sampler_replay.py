"""Saved outputs describe completed inference without running it again."""

import json

import pytest
import xarray as xr

from openghg_inversions.rhime import _standard_stages as stages, standard
from openghg_inversions.rhime.outputs import RhimeResult
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.rhime.specs import RhimeModelSpec, RhimeOutputSpec, RhimeRunSpec
from tests.test_standard_handoffs import _saved


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
    _, path = _saved(tmp_path)
    output_dir = tmp_path / "products"
    sampler = RhimeSampler(draws=17, burn=2, tune=13, chains=3, nuts_sampler="numpyro")
    manifest = json.loads(path.read_text())
    recorded = sampler.describe()
    # Runtime-only or future sampler options are descriptive provenance at replay.
    manifest["provenance"] = {"unavailable_runtime_plugin": {"module": "not_installed"}}
    manifest["actual_inference"] = recorded
    path.write_text(json.dumps(manifest))
    with monkeypatch.context() as patch:
        patch.setattr(RhimeSampler, "sample", _forbidden)
        patch.setattr(standard, "build_standard_model", _forbidden)
        replay = stages.postprocess(path, output=RhimeOutputSpec(output_path=str(output_dir)))
    expected = {"draws": 17, "burn": 2, "tune": 13, "chains": 3, "nuts_sampler": "numpyro"}
    assert replay.output_metadata["sampler"] == expected
    assert replay.provenance == manifest["provenance"]
    sampler.draws = 900
    assert replay.output_metadata["sampler"] == expected
    assert json.loads(path.read_text())["actual_inference"] == recorded


@pytest.mark.parametrize(
    "recorded",
    [None, {}, "unavailable", {"draws": "invalid"}, {"tune": []},
     {"chains": None}, {"burn": "inf"}, {"nuts_sampler": "unavailable_backend"}],
)
def test_unavailable_sampler_description_does_not_block_products(tmp_path, monkeypatch, recorded):
    _, path = _saved(tmp_path)
    output_dir = tmp_path / "products"
    manifest = json.loads(path.read_text())
    manifest["actual_inference"] = recorded
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(RhimeSampler, "sample", _forbidden)
    monkeypatch.setattr(standard, "build_standard_model", _forbidden)
    replay = stages.postprocess(path, output=RhimeOutputSpec(output_path=str(output_dir)))
    assert replay.output_metadata["sampler"] == recorded
    assert list(output_dir.glob("*.nc"))


def test_partial_sampler_description_does_not_invent_defaults(tmp_path, monkeypatch):
    _, path = _saved(tmp_path)
    output_dir = tmp_path / "products"
    manifest = json.loads(path.read_text())
    manifest["actual_inference"] = {"draws": 17, "burn": 2}
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(RhimeSampler, "sample", _forbidden)
    monkeypatch.setattr(standard, "build_standard_model", _forbidden)
    replay = stages.postprocess(path, output=RhimeOutputSpec(output_path=str(output_dir)))
    assert replay.output_metadata["sampler"] == {"draws": 17, "burn": 2}
