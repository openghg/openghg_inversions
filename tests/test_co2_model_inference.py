"""Scientific CO2 likelihood choices own their graph and inference method."""

from dataclasses import replace
import json
import pytensor

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.inference.sampling import RhimeSampler
from openghg_inversions.rhime.co2 import execution, handoffs, inference
from openghg_inversions.rhime.co2.co2_outputs import Co2Output
from openghg_inversions.rhime.co2.co2_affine_output import produce_bucket_affine_flux_map
from openghg_inversions.rhime.co2.configuration import resolve_co2_model, _resolve_co2_sampling
from test_co2_affine_output import _prepared


FACTS = {"start_date": "2020-01-01", "end_date": "2020-02-01", "domain": "EUROPE"}


def _request():
    return {
        "prepared_inputs": FACTS,
        "likelihood": {"kind": "fixed_ou", "tau_hours": {"MHD": 4.0, "DROPPED": 7.0},
                       "site_amplitude_prior": {"pdf": "gamma", "alpha": 2.0, "beta": 3.0},
                       "initial_site_amplitudes": {"MHD": 0.4, "DROPPED": 0.8}},
        "sampling": {"sigma_target_accept": 0.85, "state_target_accept": 0.92, "draws": 3, "tune": 3, "chains": 1, "random_seed": 14,
                     "progressbar": False, "sample_prior_predictive": 2},
        "outputs": {"output_format": "basic"},
    }


@pytest.mark.parametrize("kind, choices, backend, cached", [
    ("additive_sigma", {"no_model_error": True}, "numpyro", False),
    ("site_sigma", {"site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0}}, "pymc", False),
    ("site_sigma", {"fixed_site_amplitudes": {"MHD": 0.5}}, "pymc", False),
    ("fixed_ou", {"tau_hours": 4.0, "fixed_site_amplitudes": 0.5}, "pymc", False),
    ("fixed_ou", {"tau_hours": 4.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0}}, "pymc", True),
    ("scalar_sigma", {"eigenbasis_path": "missing.nc", "sigma_prior": {"pdf": "halfnormal", "sigma": 1.0}}, "pymc", False),
])
def test_scientific_likelihood_selects_backend_and_matched_graph(kind, choices, backend, cached):
    request = {"likelihood": {"kind": kind, **choices}}
    assert _resolve_co2_sampling(request, model=resolve_co2_model(request)).sampler.nuts_sampler == backend
    assert inference.uses_cached_ou(resolve_co2_model(request)) is cached


@pytest.mark.parametrize("sampling", [
    {"nuts_sampler": "pymc"}, {"target_accept": 0.9},
    {"sample_posterior_predictive": ["epsilon"]},
])
def test_configured_sampler_rejection_precedes_input_loading(monkeypatch, sampling):
    request = _request()
    request["sampling"] = sampling
    request["prepared_inputs"] = {"path": "missing.nc"}

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid sampler policy must fail before loading or building")

    monkeypatch.setattr(execution.Co2PreparedInputs, "load", forbidden)
    monkeypatch.setattr(inference, "build_co2_inference", forbidden)
    with pytest.raises(ValueError, match="sampler|target_accept|supports only"):
        execution.run_co2(params=request)


@pytest.mark.parametrize("sampler", [
    RhimeSampler(nuts_sampler="numpyro"), RhimeSampler(sample_kwargs={"step": object()}),
    RhimeSampler(posterior_predictive_kwargs={"predictions": True}),
])
def test_direct_sampler_policy_is_validated_before_model_building(monkeypatch, sampler):
    monkeypatch.setattr(inference, "build_co2_inference", lambda *args: pytest.fail("built graph"))
    with pytest.raises(ValueError, match="pymc|CompoundStep|only random_seed"):
        inference.sample_co2_inference(object(), resolve_co2_model(_request()), sampler)


@pytest.mark.parametrize("fixed_states,baseline", [(False, None), (True, None), (True, "boundary"), (True, "offset")])
def test_configured_inferred_ou_returns_products_and_saved_prior(tmp_path, monkeypatch, fixed_states, baseline):
    prepared, native = _prepared()
    inputs = prepared.inv_inputs
    if fixed_states:
        inputs = inputs.assign(state_is_active=xr.zeros_like(inputs.alpha_prior_mean, dtype=bool))
        prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))
    request = _request()
    if baseline == "boundary":
        inputs = prepared.inv_inputs.assign(H_bc=xr.DataArray([[0.1]], dims=("nmeasure", "bc_region"),
            coords={"nmeasure": prepared.inv_inputs.nmeasure, "bc_region": ["background"]}, attrs={"units": "ppm"}))
        prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))
        request["model"] = {"boundary": {"enabled": True, "prior": {"pdf": "normal", "mu": 1.0, "sigma": 0.1}}}
    elif baseline == "offset":
        request["model"] = {"offset": {"prior": {"pdf": "normal", "mu": 0.0, "sigma": 0.1}, "per_site": False}}
    affine = produce_bucket_affine_flux_map(prepared, native, prepared_inputs_id="in-memory")
    result = execution.run_co2(params=request, prepared=prepared, affine_artifact=affine)
    assert isinstance(result, Co2Output)
    assert result.idata.posterior.sizes["draw"] == 3
    assert result.idata.prior.sizes["draw"] == 2
    assert result.idata.posterior.ou_site_amplitude.min() > 0
    assert np.isfinite(result.idata.log_likelihood.y).all()
    assert "concentration_components" in result.outputs
    assert "native_flux" in result.outputs
    assert result.idata.posterior.attrs["rhime_recipe"] == "co2"
    actual = result.output_metadata["sampler"]
    assert json.loads(result.idata.attrs["rhime_actual_inference"]) == actual
    assert actual["sigma_target_accept"] == 0.85
    has_states = not fixed_states or baseline is not None
    assert actual["algorithm"] == ("cached_ou_sigma_then_state_nuts" if has_states else "cached_ou_sigma_nuts")
    assert (actual.get("state_target_accept") == 0.92) is has_states
    if fixed_states:
        np.testing.assert_allclose(result.idata.posterior.flux_scaling, 1.0)
    manifest = handoffs.save_preparation(
        prepared, affine_artifact=affine, output_dir=tmp_path / "prepared", **FACTS,
    )
    report = handoffs.prior_predictive(manifest, model=request, output_dir=tmp_path / "prior", draws=2, random_seed=13)
    assert report["status"] == "pass"
    saved = handoffs.sample(
        manifest, model=request, sampling=request["sampling"], output_dir=tmp_path / "samples",
    )
    record = json.loads(saved.read_text())
    assert record["actual_inference"] == actual

    def forbidden(*args, **kwargs):
        pytest.fail("Replay built a graph or sampler")

    monkeypatch.setattr(inference, "build_co2_inference", forbidden)
    monkeypatch.setattr(RhimeSampler, "__init__", forbidden)
    restored = handoffs.load_samples(saved)
    assert restored.output_metadata["sampler"] == actual
    assert json.loads(restored.idata.attrs["rhime_actual_inference"]) == actual
    assert restored.idata.posterior.ou_site_amplitude.min() > 0
    assert restored.idata.posterior.attrs["rhime_recipe"] == "co2"


@pytest.mark.parametrize("name,value", [("sigma_target_accept", "invalid"), ("state_target_accept", 2.0)])
@pytest.mark.parametrize("phase", ["full", "saved"])
def test_posterior_tuning_rejection_precedes_input_loading(monkeypatch, tmp_path, name, value, phase):
    request = _request()
    request["sampling"][name] = value

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid posterior tuning must fail before loading inputs")

    monkeypatch.setattr(execution.Co2PreparedInputs, "load", forbidden)
    monkeypatch.setattr(handoffs, "load_preparation", forbidden)
    with pytest.raises((TypeError, ValueError), match=name):
        if phase == "full":
            execution.run_co2(params=request)
        else:
            handoffs.sample("missing.json", model=request, sampling=request["sampling"], output_dir=tmp_path)


def test_prior_prediction_ignores_invalid_posterior_tuning(tmp_path):
    from openghg_inversions.serialization import load_trace

    prepared, _ = _prepared()
    manifest = handoffs.save_preparation(prepared, output_dir=tmp_path / "prepared", **FACTS)
    request = _request()
    request["sampling"].update(sigma_target_accept="invalid", state_target_accept=2.0)
    report = handoffs.prior_predictive(
        manifest, model=request, output_dir=tmp_path / "prior", draws=2, random_seed=13,
    )
    assert report["status"] == "pass"
    trace = load_trace(tmp_path / "prior" / "prior-predictive.nc")
    assert trace.prior_predictive.y.sizes["draw"] == 2
    assert np.isfinite(trace.prior_predictive.y).all()
    assert request["sampling"]["sigma_target_accept"] == "invalid"
    assert request["sampling"]["state_target_accept"] == 2.0


@pytest.mark.parametrize("likelihood,error,match", [
    ([], TypeError, "likelihood"),
    ({"kind": "invalid"}, ValueError, "likelihood.kind"),
    ({"kind": "fixed_ou", "tau_hours": -1.0}, ValueError, "tau_hours"),
    ({"kind": "additive_sigma", "sigma_target_accept": "invalid"}, ValueError, "sigma_target_accept"),
    ({**_request()["likelihood"], "unknown": True}, ValueError, "unknown"),
])
def test_prior_resolution_still_validates_scientific_choices(likelihood, error, match):
    with pytest.raises(error, match=match):
        resolve_co2_model({"likelihood": likelihood})


def test_all_fixed_ou_empty_state_graph_uses_float32():
    prepared, _ = _prepared()
    inputs = prepared.inv_inputs.assign(state_is_active=xr.zeros_like(prepared.inv_inputs.alpha_prior_mean, dtype=bool))
    prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))
    with pytensor.config.change_flags(floatX="float32"):
        built, cached = inference.build_co2_inference(prepared, resolve_co2_model(_request()))
    assert cached is not None and not cached.states
    assert cached.cached_state_logp.dtype == "float32"
    assert built.model["ou_site_amplitude"].dtype == "float32"


@pytest.mark.parametrize("value", [10**400, float("nan"), float("inf"), True])
def test_invalid_tuning_metadata_does_not_block_replay(value):
    assert inference.describe_co2_inference({"algorithm": "nuts", "target_accept": value}) == {
        "algorithm": "nuts",
    }


@pytest.mark.parametrize("target", [None, 0.93])
def test_ordinary_inference_preserves_sampler_default_and_records_source(monkeypatch, target):
    sampler = RhimeSampler(sample_kwargs={} if target is None else {"target_accept": target})
    monkeypatch.setattr(inference, "build_co2_inference", lambda *args: (object(), None))
    monkeypatch.setattr(inference, "annotate_co2_inference", lambda trace, *args: trace)

    def sample(built, active):
        assert active is sampler
        assert (active.sample_kwargs or {}) == ({} if target is None else {"target_accept": target})
        return xr.DataTree()

    monkeypatch.setattr(inference, "sample_rhime_model", sample)
    trace, actual = inference.sample_co2_inference(object(), {}, sampler)
    assert inference.describe_co2_inference(json.loads(trace.attrs["rhime_actual_inference"])) == actual
    if target is None:
        assert actual["target_accept_source"] == "pm.sample_default"
        assert "target_accept" not in actual
    else:
        assert actual["target_accept"] == target
        assert "target_accept_source" not in actual
