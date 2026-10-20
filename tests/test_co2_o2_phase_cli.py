"""Installed linked stages use joint scientific inputs and phase-owned choices."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from examples.co2_o2_handoffs import FACTS, synthetic_inputs
from openghg_inversions.cli import main
from openghg_inversions.rhime.co2 import co2_o2_handoffs as handoffs
from openghg_inversions.serialization import load_trace


CHANNELS = {"co2": {"units": "ppm"}, "o2": {"units": "ppm"}}


def _cli(*arguments):
    executable = Path(sys.executable).with_name("openghg-inversions")
    result = subprocess.run([str(executable), *map(str, arguments)], text=True,
                            capture_output=True, env=os.environ.copy(), timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_installed_unequal_axis_sequence_retains_joint_science(tmp_path):
    prepared = synthetic_inputs()
    source = tmp_path / "linked.nc"
    prepared.save(source)
    params = tmp_path / "choices.json"
    params.write_text(json.dumps({
        "format_version": 1, "recipe": "co2_o2", "prepared_inputs": {"path": str(source)},
        "channels": CHANNELS,
        "sampling": {"draws": 6, "tune": 6, "chains": 2, "nuts_sampler": "pymc",
                     "random_seed": 19, "sample_prior_predictive": 3, "sample_posterior_predictive": True},
        "outputs": {"output_format": "basic", "output_name": "linked", "save_concentrations": True},
    }))
    prepare_dir, sample_dir = tmp_path / "prepare", tmp_path / "sample"
    prepare = _cli("prepare", "--model", "co2_o2", "--params-file", params, "--output-dir", prepare_dir)
    prep_manifest = Path(prepare.stdout.strip().splitlines()[-1])
    assert prep_manifest == prepare_dir / "prepare-manifest.json"
    _cli("prior-predictive", "--model", "co2_o2", "--params-file", params,
         "--preparation-manifest", prep_manifest, "--draws", "3", "--strict", "--output-dir", tmp_path / "prior")
    sample = _cli("sample", "--model", "co2_o2", "--params-file", params,
                  "--preparation-manifest", prep_manifest, "--output-dir", sample_dir)
    sample_manifest = Path(sample.stdout.strip().splitlines()[-1])
    _cli("diagnose", "--sample-manifest", sample_manifest, "--output-dir", tmp_path / "diagnose")
    _cli("postprocess", "--model", "co2_o2", "--sample-manifest", sample_manifest,
         "--params-file", params, "--output-dir", tmp_path / "products")

    prior = load_trace(tmp_path / "prior" / "prior-predictive.nc")
    trace = load_trace(sample_dir / "posterior.nc")
    assert prior.prior_predictive.sizes["observation"] == 5
    assert trace.posterior.sizes["observation"] == 5
    assert trace.observed_data.indexes["observation"].equals(prepared.observations.indexes["observation"])
    assert trace.posterior.species.values.tolist() == ["co2", "co2", "o2", "o2", "o2"]
    assert trace.posterior.observation_units.values.tolist() == ["ppm"] * 5
    roles = json.loads(trace.attrs["rhime_variable_roles"])
    assert trace.posterior[roles["flux_scale"]].source.values.tolist() == ["GPP", "TER", "FF", "ocean", "ocean"]
    np.testing.assert_allclose(trace.constant_data.aggregation_error_covariance,
                               prepared.aggregation_error.covariance)
    assert roles["modelled_concentration"] in trace.posterior
    assert roles["flux_scale"] in trace.posterior
    assert (tmp_path / "diagnose" / "sampler-convergence.json").exists()
    products = list((tmp_path / "products").glob("*.nc"))
    assert products
    for species, count in (("co2", 2), ("o2", 3)):
        matching = [path for path in products if species in path.name and "components" in path.name]
        assert matching
        with xr.open_dataset(matching[0]) as components:
            assert count in components.sizes.values()
    loaded = handoffs.load_samples(sample_manifest)
    assert (loaded.start_date, loaded.end_date, loaded.domain) == tuple(FACTS.values())


def test_sample_ignores_inactive_preparation_and_outputs(tmp_path, monkeypatch):
    manifest = tmp_path / "prepare.json"
    manifest.write_text(json.dumps({"schema": "openghg_inversions.co2_o2.prepare"}))
    calls = []

    def sample(preparation, *, model, sampler, output_dir):
        calls.append((preparation, model, sampler.draws))
        return output_dir / "sample-manifest.json"

    monkeypatch.setattr(handoffs, "sample", sample)
    main(["sample", "--model", "co2_o2", "--preparation-manifest", str(manifest),
          "--output-dir", str(tmp_path / "sample"), "--kwargs", json.dumps({
              "recipe": "co2_o2", "channels": CHANNELS, "sampling": {"draws": 7, "chains": 1},
              "prepared_inputs": {"path": 42}, "outputs": {"output_path": 42},
          })])
    assert calls[0][0] == str(manifest)
    assert calls[0][1]["channels"] == CHANNELS
    assert calls[0][2] == 7


def test_prior_only_resolves_seed_without_inactive_sampler(tmp_path, monkeypatch):
    manifest = tmp_path / "prepare.json"
    manifest.write_text(json.dumps({"schema": "openghg_inversions.co2_o2.prepare"}))
    calls = []

    def prior(preparation, **kwargs):
        calls.append(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(handoffs, "prior_predictive", prior)
    main(["prior-predictive", "--model", "co2_o2", "--preparation-manifest", str(manifest),
          "--output-dir", str(tmp_path / "prior"), "--kwargs", json.dumps({
              "recipe": "co2_o2", "channels": CHANNELS,
              "sampling": {"random_seed": 73, "draws": "invalid inactive value"},
              "outputs": {"output_format": "invalid inactive value"},
          })])
    assert calls[0]["random_seed"] == 73


def test_postprocess_requires_only_current_output_choices(tmp_path, monkeypatch):
    manifest = tmp_path / "sample.json"
    manifest.write_text(json.dumps({"schema": "openghg_inversions.co2_o2.sample"}))
    calls = []
    monkeypatch.setattr(handoffs, "postprocess", lambda manifest, **kwargs: calls.append(kwargs))
    main(["postprocess", "--model", "co2_o2", "--sample-manifest", str(manifest),
          "--output-dir", str(tmp_path / "products"), "--kwargs", json.dumps({
              "outputs": {"output_format": "basic", "output_name": "later"},
              "sampling": {"draws": "invalid inactive value"}, "channels": {"co2": "invalid inactive value"},
          })])
    assert calls[0]["output"].output_name == "later"
    assert calls[0]["output"].output_path == tmp_path / "products"


@pytest.mark.parametrize("selected,saved", [("co2", "co2_o2"), ("co2_o2", "co2"), ("standard", "co2_o2")])
def test_wrong_family_fails_before_writes(tmp_path, selected, saved):
    manifest = tmp_path / "prepare.json"
    manifest.write_text(json.dumps({"schema": f"openghg_inversions.{saved}.prepare"}))
    with pytest.raises(ValueError, match="does not match"):
        main(["sample", "--model", selected, "--preparation-manifest", str(manifest),
              "--output-dir", str(tmp_path / "sample")])
    assert not (tmp_path / "sample").exists()
