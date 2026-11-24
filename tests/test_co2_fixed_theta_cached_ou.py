"""Fixed-θ state sampling on the cached fixed-OU route (openghg_inversions#1109, stage 1).

The supplied site amplitudes are constant data. The cached state quadratic is
built once at those values and stock state NUTS samples the states alone. The
reference is stock PyMC NUTS on the same graph, whose observed likelihood is
the ordinary fixed-amplitude fixed-OU distribution.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pymc as pm
import pytensor
import pytest
import xarray as xr
from pymc.blocking import DictToArrayBijection
from pymc.initial_point import make_initial_point_fn
from scipy.stats import multivariate_normal

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.inference.cached_sigma import make_cached_state_nuts_step
from openghg_inversions.inference.fixed_amplitudes import (
    FixedSiteAmplitudeFile,
    FixedSiteAmplitudes,
    in_concentration_units,
    load_fixed_site_amplitudes,
    resolve_fixed_site_amplitudes,
)
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import build_co2_cached_sigma_model, execution, handoffs, inference
from openghg_inversions.rhime.co2.configuration import Co2ModelConfig, _resolve_co2_sampling
from test_co2_affine_output import _prepared
from test_co2_cached_sigma import _dense_fixed_ou_covariance, _inputs


TAU = {"AAA": 3.0, "BBB": 7.0}
AMPLITUDES = {"AAA": 0.3, "BBB": 0.7}


@pytest.fixture(autouse=True)
def dense_oracle_precision():
    """Keep strict double-precision oracles independent of package import order."""
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _build(*, fixed=AMPLITUDES, inputs: xr.Dataset | None = None, **kwargs):
    inputs = _inputs() if inputs is None else inputs
    prior = CorrelatedLognormalPrior(
        inputs["alpha_prior_mean"], inputs["alpha_prior_covariance"], covariance_dim="region_cov",
    )
    return build_co2_cached_sigma_model(
        inputs["H"],
        retained_prior=prior,
        fixed_prior_contribution=inputs["fixed_prior_contribution"],
        observations=inputs["mf"],
        observation_error=inputs["mf_error"],
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        tau_hours=TAU,
        fixed_site_amplitudes=fixed,
        **kwargs,
    )


def _start(cached, seed: int = 769):
    return make_initial_point_fn(
        model=cached.model, overrides=cached.initial_values, return_transformed=True,
    )(seed)


def _state_step(cached, **kwargs):
    return make_cached_state_nuts_step(
        model=cached.model, states=cached.states, cached_state_logp=cached.cached_state_logp,
        initial_point=_start(cached), **kwargs,
    )


@pytest.mark.parametrize("offset", [False, True])
def test_fixed_theta_state_density_matches_stock_fixed_ou_density(offset):
    """The cached state density equals the stock observed density plus priors, with gradients."""
    kwargs = {"offset_prior": {"pdf": "normal", "mu": 0.0, "sigma": 0.2}, "offset_per_site": False} if offset else {}
    cached = _build(**kwargs)
    model = cached.model
    assert cached.fixed_amplitudes is not None
    assert "ou_site_amplitude" not in {rv.name for rv in model.free_RVs}
    np.testing.assert_array_equal(model["ou_site_amplitude"].get_value(), [0.3, 0.7])

    step = _state_step(cached)
    state_values = [model.rvs_to_values[rv] for rv in cached.states]
    stock_logp = model.compile_logp()
    stock_dlogp = model.compile_dlogp(vars=cached.states)
    mean_fn = model.compile_fn(model.replace_rvs_by_values([cached.modelled_mean])[0], on_unused_input="ignore")
    covariance = _dense_fixed_ou_covariance(_inputs(), np.asarray([0.3, 0.7]), TAU)
    observations = _inputs()["mf"].values
    prior_logp = model.compile_logp(vars=model.free_RVs)
    rng = np.random.default_rng(31)
    start = _start(cached)
    for _ in range(5):
        point = {name: value + rng.normal(scale=0.3, size=np.shape(value)) for name, value in start.items()}
        raveled = DictToArrayBijection.map({value.name: point[value.name] for value in state_values})
        cached_logp, cached_grad = step._logp_dlogp_func(raveled.data)
        np.testing.assert_allclose(cached_logp, stock_logp(point), rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(cached_grad, stock_dlogp(point), rtol=1e-8, atol=1e-9)
        dense = multivariate_normal(mean_fn(point), covariance).logpdf(observations)
        np.testing.assert_allclose(cached_logp - prior_logp(point), dense, rtol=1e-10, atol=1e-9)


def _batch_mcse(draws: np.ndarray, batches: int = 20) -> np.ndarray:
    """Batch-means Monte Carlo standard error over (chain, draw, ...) samples."""
    chains, n = draws.shape[:2]
    size = n // batches
    means = draws[:, : size * batches].reshape(chains, batches, size, *draws.shape[2:]).mean(axis=2)
    means = means.reshape(chains * batches, *draws.shape[2:])
    return means.std(axis=0, ddof=1) / np.sqrt(chains * batches)


def test_fixed_theta_sampling_matches_stock_pymc_nuts():
    """Cached fixed-θ state NUTS and stock PyMC NUTS target the same posterior."""
    cached = _build()
    common = {"draws": 2000, "tune": 1000, "chains": 2, "cores": 1, "progressbar": False,
              "compute_convergence_checks": False}
    with cached.model:
        stock = pm.sample(random_seed=11, **common)
        fixed = pm.sample(step=_state_step(cached, rng=np.random.default_rng(12)), random_seed=12,
                          initvals=cached.initial_values, **common)
    stock_draws = np.asarray(stock.posterior["flux_scaling"].values)
    fixed_draws = np.asarray(fixed.posterior["flux_scaling"].values)
    tolerance = 4.0 * np.hypot(_batch_mcse(stock_draws), _batch_mcse(fixed_draws))
    np.testing.assert_array_less(np.abs(stock_draws.mean((0, 1)) - fixed_draws.mean((0, 1))), tolerance)
    np.testing.assert_allclose(fixed_draws.std((0, 1)), stock_draws.std((0, 1)), rtol=0.1)
    assert "ou_site_amplitude" not in fixed.posterior


def test_fixed_amplitudes_reject_inferred_options_and_singular_covariance():
    with pytest.raises(ValueError, match="exactly one"):
        _build(site_amplitude_prior={"pdf": "halfnormal", "sigma": 1.0})
    with pytest.raises(ValueError, match="initial_site_amplitudes"):
        _build(initial_site_amplitudes=0.4)
    with pytest.raises(ValueError, match="non-negative"):
        _build(fixed={"AAA": -0.1, "BBB": 0.2})
    inputs = _inputs()
    with pytest.raises(ValueError, match="active state"):
        _build(state_activity=StateActivity(active=np.zeros(2, dtype=bool), fixed_value=1.0), inputs=inputs)


def _posterior_file(path: Path, values: np.ndarray, units: str | None = "ppm") -> Path:
    posterior = xr.Dataset(
        {"ou_site_amplitude": (("chain", "draw", "ou_site"), values, {} if units is None else {"units": units})},
        coords={"ou_site": ["MHD"]},
    )
    xr.DataTree.from_dict({"posterior": posterior}).to_netcdf(path)
    return path


def test_amplitude_file_summarises_posterior_draws_and_records_source(tmp_path):
    draws = np.asarray([[[0.2], [0.5], [0.3]], [[0.4], [0.9], [0.6]]])
    path = _posterior_file(tmp_path / "posterior.nc", draws)
    median = load_fixed_site_amplitudes(FixedSiteAmplitudeFile(path))
    assert median.values == {"MHD": pytest.approx(0.45)}
    assert median.source["kind"] == "file"
    assert median.source["path"] == str(path.resolve())
    assert median.source["group"] == "posterior"
    assert median.source["statistic"] == "median"
    assert median.source["n_samples"] == 6
    assert median.source["units"] == "ppm" and median.units == "ppm"
    assert len(median.source["sha256"]) == 64
    mean = load_fixed_site_amplitudes(FixedSiteAmplitudeFile(path, statistic="mean"))
    assert mean.values == {"MHD": pytest.approx(draws.mean())}

    flat = tmp_path / "amplitudes.nc"
    xr.Dataset(
        {"ou_site_amplitude": ("ou_site", [0.25], {"units": "ppm"})}, coords={"ou_site": ["MHD"]},
    ).to_netcdf(flat)
    plain = load_fixed_site_amplitudes(FixedSiteAmplitudeFile(flat))
    assert plain.values == {"MHD": 0.25}
    assert plain.source["statistic"] is None and plain.source["group"] is None
    assert resolve_fixed_site_amplitudes(0.4) == FixedSiteAmplitudes(0.4, {"kind": "supplied"})

    with pytest.raises(ValueError, match="no variable"):
        load_fixed_site_amplitudes(FixedSiteAmplitudeFile(flat, variable="missing"))
    negative = _posterior_file(tmp_path / "negative.nc", -draws)
    with pytest.raises(ValueError, match="non-negative"):
        load_fixed_site_amplitudes(FixedSiteAmplitudeFile(negative))
    with pytest.raises(ValueError, match="statistic"):
        FixedSiteAmplitudeFile(path, statistic="mode")  # type: ignore[arg-type]


def test_amplitude_file_units_are_declared_and_converted(tmp_path):
    ppb = load_fixed_site_amplitudes(FixedSiteAmplitudeFile(_posterior_file(tmp_path / "ppb.nc", np.full((1, 2, 1), 400.0), "ppb")))
    assert ppb.values == {"MHD": 400.0} and ppb.units == "ppb"
    converted = in_concentration_units(ppb, "ppm")
    assert converted.values == {"MHD": pytest.approx(0.4)}
    assert converted.units == "ppm"
    assert converted.source["units"] == "ppb" and converted.source["fit_units"] == "ppm"
    assert converted.source["unit_factor"] == pytest.approx(1e-3)
    assert in_concentration_units(FixedSiteAmplitudes(0.4), "ppm") == FixedSiteAmplitudes(0.4)
    with pytest.raises(ValueError, match="Could not convert"):
        in_concentration_units(FixedSiteAmplitudes({"MHD": 1.0}, units="K"), "ppm")

    unitless = _posterior_file(tmp_path / "unitless.nc", np.full((1, 2, 1), 0.4), None)
    with pytest.raises(ValueError, match="no units attribute"):
        load_fixed_site_amplitudes(FixedSiteAmplitudeFile(unitless))
    declared = load_fixed_site_amplitudes(FixedSiteAmplitudeFile(unitless, units="ppm"))
    assert declared.units == "ppm" and declared.source["units"] == "ppm"
    ppb_path = Path(ppb.source["path"])
    with pytest.raises(ValueError, match="declares units 'ppb'"):
        load_fixed_site_amplitudes(FixedSiteAmplitudeFile(ppb_path, units="ppm"))
    assert load_fixed_site_amplitudes(FixedSiteAmplitudeFile(ppb_path, units="1e-9")).units == "ppb"


def test_amplitude_file_units_are_converted_to_the_fit_units(tmp_path):
    prepared, _ = _prepared()
    assert prepared.inv_inputs.mf.attrs["units"] == "ppm"
    path = _posterior_file(tmp_path / "ppb.nc", np.full((1, 3, 1), 400.0), "ppb")
    request = _request({"path": str(path)})
    model = Co2ModelConfig.from_params(request).model_arguments()
    sampling = _resolve_co2_sampling(request, model=model)
    trace, actual = inference.sample_co2_inference(
        prepared, model, sampling.sampler, state_target_accept=sampling.state_target_accept,
    )
    np.testing.assert_allclose(trace.constant_data["ou_site_amplitude"], [0.4])
    assert trace.constant_data["ou_site_amplitude"].attrs["units"] == "ppm"
    source = actual["ou_site_amplitude_source"]
    assert source["units"] == "ppb" and source["fit_units"] == "ppm"
    assert source["unit_factor"] == pytest.approx(1e-3)

    unitless = _posterior_file(tmp_path / "unitless.nc", np.full((1, 3, 1), 400.0), None)
    request = _request({"path": str(unitless)})
    model = Co2ModelConfig.from_params(request).model_arguments()
    with pytest.raises(ValueError, match="no units attribute"):
        inference.sample_co2_inference(prepared, model, _resolve_co2_sampling(request, model=model).sampler)
    request = _request({"path": str(unitless), "units": "ppb"})
    model = Co2ModelConfig.from_params(request).model_arguments()
    assert model["likelihood_kwargs"]["fixed_site_amplitudes"].units == "ppb"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forbid_sampling(*args, **kwargs):
    pytest.fail("Output overlap must be rejected before sampling.")


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_full_run_outputs_cannot_overwrite_the_amplitude_file(tmp_path, monkeypatch, alias):
    prepared, _ = _prepared()
    source = _posterior_file(tmp_path / "amplitudes.nc", np.full((1, 3, 1), 0.4))
    before = _sha256(source)
    trace_path = tmp_path / "trace.nc"
    if alias == "same":
        trace_path = source
    elif alias == "symlink":
        trace_path.symlink_to(source)
    else:
        os.link(source, trace_path)
    params = _request({"path": str(source)})
    params["prepared_inputs"] = {"domain": "EUROPE", "start_date": "2019-01-01", "end_date": "2019-02-01"}
    params["outputs"] = {"output_format": "basic", "output_path": str(tmp_path / "out"), "save_trace": str(trace_path)}
    monkeypatch.setattr(execution, "sample_co2_inference", _forbid_sampling)
    with pytest.raises(ValueError, match="overlaps an input or output artifact"):
        execution.run_co2(params=params, prepared=prepared)
    assert _sha256(source) == before


def test_staged_sampling_cannot_overwrite_the_amplitude_file(tmp_path, monkeypatch):
    prepared, _ = _prepared()
    preparation = handoffs.save_preparation(
        prepared, output_dir=tmp_path / "prepared", domain="EUROPE", start_date="2019-01-01", end_date="2019-02-01",
    )
    samples = tmp_path / "samples"
    samples.mkdir()
    source = _posterior_file(samples / "posterior.nc", np.full((1, 3, 1), 0.4))
    before = _sha256(source)
    params = _request({"path": str(source)})
    monkeypatch.setattr(handoffs, "sample_co2_inference", _forbid_sampling)
    with pytest.raises(ValueError, match="overlaps an input or output artifact"):
        handoffs.sample(preparation, model=params, sampling=params["sampling"], output_dir=samples)
    generation = json.loads(preparation.read_text())["generation"]
    recorded = {"ou_site_amplitude_source": {"kind": "file", "path": str(source)}}
    monkeypatch.setattr(handoffs, "validate_co2_trace", lambda *args: None)
    with pytest.raises(ValueError, match="overlaps an input or output artifact"):
        handoffs.save_samples(
            preparation, xr.DataTree(), actual_inference=recorded, preparation_generation=generation,
            output_dir=samples,
        )
    assert _sha256(source) == before


def _request(fixed: object, *, cached_state: bool = True) -> dict:
    likelihood = {"kind": "fixed_ou", "tau_hours": 4.0, "fixed_site_amplitudes": fixed}
    if cached_state:
        likelihood["cached_state_quadratic"] = True
    return {
        "likelihood": likelihood,
        "sampling": {"state_target_accept": 0.92, "draws": 3, "tune": 3, "chains": 1, "random_seed": 14,
                     "progressbar": False, "sample_prior_predictive": 2},
    }


@pytest.mark.parametrize("source", ["values", "file"])
def test_configured_fixed_theta_runs_cached_state_nuts_and_records_source(tmp_path, source):
    prepared, _ = _prepared()
    if source == "file":
        path = _posterior_file(tmp_path / "posterior.nc", np.asarray([[[0.3], [0.5], [0.4]]]))
        fixed: object = {"path": str(path), "statistic": "median"}
    else:
        fixed = {"MHD": 0.4}
    request = _request(fixed)
    model = Co2ModelConfig.from_params(request).model_arguments()
    assert inference.uses_cached_ou(model) and inference.uses_fixed_theta_cached_ou(model)
    sampling = _resolve_co2_sampling(request, model=model)
    trace, actual = inference.sample_co2_inference(
        prepared, model, sampling.sampler, state_target_accept=sampling.state_target_accept,
    )
    assert actual["algorithm"] == "cached_ou_fixed_sigma_state_nuts"
    assert actual["state_target_accept"] == 0.92
    assert "sigma_target_accept" not in actual
    assert actual["ou_site_amplitude_treatment"] == "fixed"
    assert actual["ou_site_amplitude_source"]["kind"] == ("file" if source == "file" else "supplied")
    assert json.loads(trace.attrs["rhime_actual_inference"]) == actual
    assert inference.describe_co2_inference(actual)["ou_site_amplitude_source"] == actual["ou_site_amplitude_source"]
    assert "ou_site_amplitude" not in trace.posterior
    np.testing.assert_allclose(trace.constant_data["ou_site_amplitude"], [0.4])
    assert trace.constant_data["ou_site_amplitude"].attrs["units"] == "ppm"
    metadata = json.loads(trace.attrs["rhime_model_metadata"])
    assert metadata["ou_site_amplitude_treatment"] == "fixed"
    assert metadata["ou_site_amplitude_source"] == actual["ou_site_amplitude_source"]
    assert "conditional" in metadata["posterior_scope"]
    assert trace.posterior.sizes["draw"] == 3
    assert np.isfinite(trace.log_likelihood["y"]).all()
    assert np.isfinite(trace.posterior_predictive["y"]).all()


def test_stock_fixed_amplitudes_are_unchanged_without_opt_in():
    request = _request({"MHD": 0.4}, cached_state=False)
    model = Co2ModelConfig.from_params(request).model_arguments()
    assert not inference.uses_cached_ou(model)
    off = {"likelihood": {**request["likelihood"], "cached_state_quadratic": False}}
    assert Co2ModelConfig.from_params(off).model_arguments() == model


@pytest.mark.parametrize("likelihood,sampling,match", [
    ({"fixed_site_amplitudes": {"path": "posterior.nc"}}, {}, "cached_state_quadratic = true"),
    ({"site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0}, "cached_state_quadratic": True}, {},
     "applies to fixed_site_amplitudes"),
    ({"fixed_site_amplitudes": {"path": "posterior.nc", "unknown": 1}, "cached_state_quadratic": True}, {},
     "accepts only"),
    ({"fixed_site_amplitudes": 0.4, "cached_state_quadratic": True}, {"sigma_target_accept": 0.8},
     "sigma_target_accept"),
])
def test_fixed_theta_configuration_rejections(likelihood, sampling, match):
    request = {"likelihood": {"kind": "fixed_ou", "tau_hours": 4.0, **likelihood}, "sampling": sampling}
    with pytest.raises(ValueError, match=match):
        model = Co2ModelConfig.from_params(request).model_arguments()
        _resolve_co2_sampling(request, model=model)


def test_direct_file_amplitudes_require_cached_route():
    prepared, _ = _prepared()
    model = Co2ModelConfig.from_params(_request(0.4, cached_state=False)).model_arguments()
    likelihood = dict(model["likelihood_kwargs"])
    likelihood["fixed_site_amplitudes"] = FixedSiteAmplitudeFile(Path("missing.nc"))
    model = {**model, "likelihood_kwargs": likelihood}
    with pytest.raises(ValueError, match="cached_state_quadratic"):
        inference.sample_co2_inference(prepared, model, _resolve_co2_sampling({}, model=model).sampler)


def test_all_fixed_state_activity_cannot_use_fixed_theta():
    prepared, _ = _prepared()
    inputs = prepared.inv_inputs.assign(state_is_active=xr.zeros_like(prepared.inv_inputs.alpha_prior_mean, dtype=bool))
    prepared = replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs))
    model = Co2ModelConfig.from_params(_request({"MHD": 0.4})).model_arguments()
    with pytest.raises(ValueError, match="active state"):
        inference.build_co2_inference(prepared, model)
