"""Scientific CO2 likelihood choices own their graph and inference method."""

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.inference.sampling import RhimeSampler
from openghg_inversions.rhime.co2 import execution, handoffs, inference
from openghg_inversions.rhime.co2.co2_outputs import Co2Output
from openghg_inversions.rhime.co2.co2_affine_output import produce_bucket_affine_flux_map
from openghg_inversions.rhime.co2.configuration import resolve_co2_model, resolve_co2_sampler
from test_co2_affine_output import _prepared


FACTS = {"start_date": "2020-01-01", "end_date": "2020-02-01", "domain": "EUROPE"}


def _request():
    return {
        "prepared_inputs": FACTS,
        "likelihood": {"kind": "fixed_ou", "tau_hours": {"MHD": 4.0, "DROPPED": 7.0},
                       "site_amplitude_prior": {"pdf": "gamma", "alpha": 2.0, "beta": 3.0},
                       "initial_site_amplitudes": {"MHD": 0.4, "DROPPED": 0.8},
                       "sigma_target_accept": 0.85, "state_target_accept": 0.92},
        "sampling": {"draws": 3, "tune": 3, "chains": 1, "random_seed": 14,
                     "progressbar": False, "sample_prior_predictive": 2},
        "outputs": {"output_format": "basic"},
    }


@pytest.mark.parametrize("kind, choices, backend, cached", [
    ("additive_sigma", {"no_model_error": True}, "pymc", False),
    ("site_sigma", {"site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0}}, "pymc", False),
    ("fixed_ou", {"tau_hours": 4.0, "fixed_site_amplitudes": 0.5}, "pymc", False),
    ("fixed_ou", {"tau_hours": 4.0, "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0}}, "pymc", True),
    ("scalar_sigma", {"eigenbasis_path": "missing.nc", "sigma_prior": {"pdf": "halfnormal", "sigma": 1.0}}, "pymc", False),
])
def test_scientific_likelihood_selects_backend_and_matched_graph(kind, choices, backend, cached):
    request = {"likelihood": {"kind": kind, **choices}}
    assert resolve_co2_sampler(request).nuts_sampler == backend
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
    RhimeSampler(nuts_sampler="numpyro"),
    RhimeSampler(sample_kwargs={"step": object()}),
    RhimeSampler(posterior_predictive_kwargs={"predictions": True}),
])
def test_saved_sampler_rejection_precedes_preparation_loading(monkeypatch, tmp_path, sampler):
    def forbidden(*args, **kwargs):
        raise AssertionError("Sampler policy must fail before checkpoint loading")

    monkeypatch.setattr(handoffs, "load_preparation", forbidden)
    with pytest.raises(ValueError, match="pymc|CompoundStep|only random_seed"):
        handoffs.sample("missing.json", model=_request(), output_dir=tmp_path, sampler=sampler)


@pytest.mark.parametrize("fixed_states", [False, True])
def test_configured_inferred_ou_returns_products_and_saved_prior(tmp_path, fixed_states):
    prepared, native = _prepared()
    inputs = prepared.inv_inputs
    if fixed_states:
        inputs = inputs.assign(state_is_active=xr.zeros_like(inputs.alpha_prior_mean, dtype=bool))
        prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))
    request = _request()
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
    if fixed_states:
        np.testing.assert_allclose(result.idata.posterior.flux_scaling, 1.0)
    manifest = handoffs.save_preparation(
        prepared, affine_artifact=affine, output_dir=tmp_path / "prepared", **FACTS,
    )
    report = handoffs.prior_predictive(manifest, model=request, output_dir=tmp_path / "prior", draws=2, random_seed=13)
    assert report["status"] == "pass"
    if not fixed_states:
        saved = handoffs.sample(
            manifest, model=request, output_dir=tmp_path / "samples", sampler=resolve_co2_sampler(request),
        )
        restored = handoffs.load_samples(saved)
        assert restored.idata.posterior.ou_site_amplitude.min() > 0
        assert restored.idata.posterior.attrs["rhime_recipe"] == "co2"


@pytest.mark.parametrize("name,value", [("sigma_target_accept", "invalid"), ("state_target_accept", 2.0)])
@pytest.mark.parametrize("phase", ["full", "saved"])
def test_posterior_tuning_rejection_precedes_input_loading(monkeypatch, tmp_path, name, value, phase):
    request = _request()
    request["likelihood"][name] = value

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid posterior tuning must fail before loading inputs")

    monkeypatch.setattr(execution.Co2PreparedInputs, "load", forbidden)
    monkeypatch.setattr(handoffs, "load_preparation", forbidden)
    with pytest.raises((TypeError, ValueError), match=name):
        if phase == "full":
            execution.run_co2(params=request)
        else:
            handoffs.sample("missing.json", model=request, output_dir=tmp_path)


def test_prior_prediction_ignores_invalid_posterior_tuning(tmp_path):
    from openghg_inversions.serialization import load_trace

    prepared, _ = _prepared()
    manifest = handoffs.save_preparation(prepared, output_dir=tmp_path / "prepared", **FACTS)
    request = _request()
    request["likelihood"].update(sigma_target_accept="invalid", state_target_accept=2.0)
    report = handoffs.prior_predictive(
        manifest, model=request, output_dir=tmp_path / "prior", draws=2, random_seed=13,
    )
    assert report["status"] == "pass"
    trace = load_trace(tmp_path / "prior" / "prior-predictive.nc")
    assert trace.prior_predictive.y.sizes["draw"] == 2
    assert np.isfinite(trace.prior_predictive.y).all()
    assert request["likelihood"]["sigma_target_accept"] == "invalid"
    assert request["likelihood"]["state_target_accept"] == 2.0


@pytest.mark.parametrize("likelihood,error,match", [
    ([], TypeError, "likelihood"),
    ({"kind": "invalid"}, ValueError, "likelihood.kind"),
    ({"kind": "fixed_ou", "tau_hours": -1.0}, ValueError, "tau_hours"),
    ({"kind": "additive_sigma", "sigma_target_accept": "invalid"}, ValueError, "sigma_target_accept"),
    ({**_request()["likelihood"], "unknown": True}, ValueError, "unknown"),
])
def test_prior_resolution_still_validates_scientific_choices(likelihood, error, match):
    with pytest.raises(error, match=match):
        resolve_co2_model({"likelihood": likelihood}, prior_only=True)
