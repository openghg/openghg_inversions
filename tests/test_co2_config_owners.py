"""Configuration owners parse active choices without numerical execution."""

from dataclasses import replace
import json

import dask.array as da
from dask.callbacks import Callback
import pytest
import xarray as xr

from openghg_inversions.rhime.co2 import (
    Co2ModelConfig, Co2O2ModelConfig, Co2PreparationConfig, Co2O2PreparationConfig,
)
from openghg_inversions.rhime.co2.co2_o2_configuration import load_co2_o2_params
from openghg_inversions.rhime.co2.execution import load_co2_params
from openghg_inversions.rhime.sampling import RhimeSampler
from test_co2_o2_fixed_ou import _prepared


@pytest.mark.parametrize("linked", [False, True])
def test_file_mapping_overrides_reach_same_owners(tmp_path, linked):
    if linked:
        config = {"channels": {"co2": {"units": "ppm"}, "o2": {"units": "ppm"}}}
        loader, model_cls, prep_cls = load_co2_o2_params, Co2O2ModelConfig, Co2O2PreparationConfig
        overrides = {"channels": {"co2": {"independent_error_sd": .2}, "o2": {"independent_error_sd": .3}}}
        toml = '[channels.co2]\nunits="ppm"\n[channels.o2]\nunits="ppm"\n'
    else:
        config = {"likelihood": {"kind": "additive_sigma"}}
        loader, model_cls, prep_cls = load_co2_params, Co2ModelConfig, Co2PreparationConfig
        overrides = {"likelihood": {"sigma_prior": {"pdf": "halfnormal", "sigma": .4}}}
        toml = '[likelihood]\nkind="additive_sigma"\n'
    overrides["prepared_inputs"] = {"path": "selected.nc", "domain": "EUROPE"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    toml_path = tmp_path / "config.toml"
    toml_path.write_text(toml)
    requests = [loader(params=config, overrides=overrides), loader(params_file=path, overrides=overrides),
                loader(config_file=toml_path, overrides=overrides)]
    assert all(model_cls.from_params(item) == model_cls.from_params(requests[0]) for item in requests)
    assert all(prep_cls.from_params(item) == prep_cls.from_params(requests[0]) for item in requests)
    assert prep_cls.from_params(requests[0]).prepared_path.name == "selected.nc"
    assert "prepared_inputs" not in config


def test_construction_owns_containers_and_borrows_opaque_numerics():
    lazy = xr.Dataset({"x": ("i", da.ones(4, chunks=2))})
    prior = {"pdf": "normal", "metadata": {"labels": ["a"], "numeric": lazy}}
    ordinary = Co2ModelConfig(bc_prior=prior)
    linked = Co2O2ModelConfig("ppm", "ppm", {}, bc_prior={"co2": prior})
    prior["metadata"]["labels"].append("b")
    assert ordinary.bc_prior["metadata"]["labels"] == ["a"]
    assert linked.bc_prior["co2"]["metadata"]["labels"] == ["a"]
    assert ordinary.bc_prior["metadata"]["numeric"] is lazy
    assert linked.bc_prior["co2"]["metadata"]["numeric"] is lazy


def test_active_science_parse_never_constructs_sampler_or_computes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("configuration parsing must not execute scientific work")
    monkeypatch.setattr(RhimeSampler, "__init__", forbidden)
    ordinary = {"likelihood": {"kind": "scalar_sigma", "eigenbasis_path": "missing.nc",
                               "sigma_prior": {"pdf": "halfnormal", "sigma": 1}},
                "sampling": "invalid", "prepared_inputs": "invalid", "outputs": "invalid"}
    linked = {"channels": {"co2": {"units": "ppm"}, "o2": {"units": "ppm"}},
              "sampling": "invalid", "prepared_inputs": "invalid", "outputs": "invalid"}
    with Callback(pretask=forbidden):
        assert Co2ModelConfig.from_params(ordinary).likelihood_kwargs["eigenbasis_path"].name == "missing.nc"
        model = Co2O2ModelConfig.from_params(linked)
        prepared = _prepared()
        error = prepared.observations.copy(data=da.ones(prepared.observations.size, chunks=2))
        assert model.bind(replace(prepared, independent_error_sd=error))["independent_error_sd"] is error


@pytest.mark.parametrize("linked", [False, True])
def test_saved_sampling_accepts_concrete_owner_without_reparsing(tmp_path, monkeypatch, linked):
    from types import SimpleNamespace
    from openghg_inversions.rhime.co2 import handoffs, co2_o2_handoffs

    module = co2_o2_handoffs if linked else handoffs
    model_cls = Co2O2ModelConfig if linked else Co2ModelConfig
    model = (Co2O2ModelConfig.from_params({"channels": {"co2": {"units": "ppm", "independent_error_sd": .2},
                                                       "o2": {"units": "ppm", "independent_error_sd": .3}}})
             if linked else Co2ModelConfig.from_params({"likelihood": {"kind": "additive_sigma"}}))
    prepared = _prepared()
    record = SimpleNamespace(prepared=prepared, payload=tmp_path / "payload.nc", affine_payload=None, generation="g")
    monkeypatch.setattr(module, "load_preparation", lambda *a: record)
    monkeypatch.setattr(module, "check_checkpoint_targets", lambda *a, **kw: None)
    monkeypatch.setattr(module, "save_samples", lambda *a, **kw: tmp_path / "sample.json")

    def forbidden(*args, **kwargs):
        raise AssertionError("resolved owners must not be parsed again")
    monkeypatch.setattr(model_cls, "from_params", forbidden)
    if linked:
        monkeypatch.setattr(module, "build_rhime_co2_o2", lambda **kw: object())
        def sample(built, sampler):
            assert sampler.nuts_sampler == "numpyro"
            return object()
        monkeypatch.setattr(module, "sample_rhime_model", sample)
        monkeypatch.setattr(module, "annotate_co2_o2_trace", lambda trace, **kw: trace)
    else:
        def sample(prepared, choices, sampler, **kwargs):
            assert choices == model.model_arguments()
            assert sampler.nuts_sampler == "numpyro"
            return object(), {}
        monkeypatch.setattr(module, "sample_co2_inference", sample)
    assert module.sample(tmp_path / "prepare.json", model=model, output_dir=tmp_path / "samples") == tmp_path / "sample.json"
