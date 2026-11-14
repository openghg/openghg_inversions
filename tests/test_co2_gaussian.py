"""Configured exact fixed-noise Gaussian estimation for the single-tracer CO2 recipe."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as sl
import xarray as xr

from examples.co2_handoffs import configuration, synthetic_inputs
from openghg_inversions.rhime.co2 import execution, handoffs
from openghg_inversions.rhime.co2 import inference as co2_inference
from openghg_inversions.rhime.co2.co2_gaussian import GAUSSIAN_ALGORITHM, check_co2_gaussian_model
from openghg_inversions.rhime.co2.co2_outputs import validate_co2_trace
from openghg_inversions.rhime.co2.configuration import Co2ModelConfig, _resolve_co2_sampling
from openghg_inversions.serialization import load_trace, save_trace

GAUSSIAN_SAMPLING = {"method": "gaussian", "draws": 40, "random_seed": 11, "sample_prior_predictive": 20,
                     "sample_posterior_predictive": False}


@pytest.fixture
def case(tmp_path):
    _, _, prepared, affine, countries = synthetic_inputs(tmp_path)
    params = configuration(tmp_path / "products", countries)
    params["sampling"] = dict(GAUSSIAN_SAMPLING)
    return prepared, affine, params


def _dense_reference(prepared, boundary_sd=0.1, offset_sd=0.1):
    """Kalman form of the same linear model, assembled from the prepared inputs."""
    inputs = prepared.inv_inputs
    design = np.hstack([inputs.H.values, inputs.H_bc.values, np.ones((inputs.sizes["nmeasure"], 1))])
    innovation = inputs.mf.values - inputs.fixed_prior_contribution.values
    prior_mean = np.concatenate([inputs.alpha_prior_mean.values, [1.0, 0.0]])
    prior_covariance = sl.block_diag(inputs.alpha_prior_covariance.values, [[boundary_sd**2]], [[offset_sd**2]])
    factor = inputs.low_rank_factor.values
    noise = factor @ factor.T + np.diag(inputs.mf_error.values**2 + inputs.diagonal_residual_variance.values)
    gain = np.linalg.solve(design @ prior_covariance @ design.T + noise, design @ prior_covariance).T
    mean = prior_mean + gain @ (innovation - design @ prior_mean)
    return prior_mean, prior_covariance, mean, prior_covariance - gain @ design @ prior_covariance


def test_configured_gaussian_matches_dense_posterior_and_feeds_products(case, double_precision):
    prepared, affine, params = case
    result = execution.run_co2(params=params, prepared=prepared, affine_artifact=affine)
    trace = result.idata
    validate_co2_trace(trace, prepared)
    actual = json.loads(trace.attrs["rhime_actual_inference"])
    assert actual["algorithm"] == GAUSSIAN_ALGORITHM and actual["seed"] == 11
    assert result.output_metadata["sampler"] == actual
    assert dict(trace.posterior.sizes)["chain"] == 1 and trace.posterior.sizes["draw"] == 40
    assert trace.prior.sizes["draw"] == 20
    assert {"native_flux", "country_flux", "basic"} <= result.outputs.keys()

    exact = trace["gaussian_posterior"].dataset
    assert list(exact.state_variable.values) == ["flux_scaling", "flux_scaling", "bc", "offset_latent"]
    prior_mean, prior_covariance, mean, covariance = _dense_reference(prepared)
    np.testing.assert_allclose(exact.prior_mean, prior_mean, rtol=1e-12)
    np.testing.assert_allclose(exact.prior_covariance, prior_covariance, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(exact.posterior_mean, mean, rtol=1e-10)
    np.testing.assert_allclose(exact.posterior_covariance, covariance, rtol=1e-9, atol=1e-13)
    diagnostics = json.loads(exact.attrs["gls_diagnostics"])
    assert diagnostics["noise_rank"] == 3 and diagnostics["prior_rank"] == 4

    # Draws pass through the unchanged graph: every deterministic matches its affine equation.
    posterior = trace.posterior.dataset
    inputs = prepared.inv_inputs
    expected = (inputs.fixed_prior_contribution + xr.dot(inputs.H, posterior.flux_scaling, dim="region")
                + (inputs.H_bc * posterior.bc).sum("bc_region") + posterior.offset_latent.sum("offset_term"))
    xr.testing.assert_allclose(posterior.modelled_concentration, expected.transpose(*posterior.modelled_concentration.dims))
    assert posterior.flux_scaling.attrs["units"] == "1"
    assert posterior.modelled_concentration.attrs["units"] == "ppm"


def test_gaussian_samples_round_trip_through_saved_handoffs(case, tmp_path, double_precision):
    prepared, affine, params = case
    manifest = handoffs.save_preparation(prepared, output_dir=tmp_path / "prepare", affine_artifact=affine,
                                         **params["prepared_inputs"])
    model = Co2ModelConfig.from_params(params)
    samples = handoffs.sample(manifest, model=model, sampling=params["sampling"], output_dir=tmp_path / "sample")
    record = json.loads(samples.read_text())
    assert record["inference_kind"] == "exact_gaussian"
    assert record["actual_inference"]["seed"] == 11
    loaded = handoffs.load_samples(samples)
    assert loaded.output_metadata["sampler"]["algorithm"] == GAUSSIAN_ALGORITHM
    assert loaded.output_metadata["sampler"]["seed"] == 11
    again = handoffs.sample(manifest, model=model, sampling=params["sampling"], output_dir=tmp_path / "again")
    xr.testing.assert_equal(handoffs.load_samples(again).idata.posterior.dataset, loaded.idata.posterior.dataset)
    saved = load_trace(samples.parent / "posterior.nc")
    xr.testing.assert_identical(saved["gaussian_posterior"].dataset, loaded.idata["gaussian_posterior"].dataset)
    path = tmp_path / "copy.nc"
    save_trace(loaded.idata, path)
    xr.testing.assert_identical(load_trace(path)["gaussian_posterior"].dataset, saved["gaussian_posterior"].dataset)
    products = handoffs.postprocess(samples, output=execution.resolve_co2_output({"outputs": {
        "output_format": "basic", "output_path": str(tmp_path / "replay")}})["output"])
    assert "native_flux" in products.outputs


@pytest.mark.parametrize(
    "change,message",
    [
        ({"likelihood": {"kind": "fixed_ou", "tau_hours": 24.0, "fixed_site_amplitudes": 0.5}}, "fixed-OU"),
        ({"likelihood": {"kind": "site_sigma", "site_amplitude_prior": {"pdf": "halfnormal", "sigma": 1.0}}},
         "additive_sigma"),
        ({"likelihood": {"kind": "additive_sigma"}}, r"fixed_model_mismatch = \.\.\., e\.g\. the median R"),
        ({"model": {"boundary": {"enabled": True}}}, "Normal boundary prior"),
        ({"model": {"offset": {"prior": {"pdf": "lognormal", "mu": 0.0, "sigma": 0.1}}}}, "Normal offset prior"),
        ({"sampling": {**GAUSSIAN_SAMPLING, "tune": 10}}, "NUTS-only option"),
        ({"sampling": {"method": "laplace"}}, "sampling.method"),
    ],
    ids=["fixed-ou", "site-sigma", "sampled-sigma", "truncated-boundary", "lognormal-offset", "nuts-option",
         "unknown-method"],
)
def test_non_gaussian_choices_are_refused_before_materialization(case, monkeypatch, double_precision, change, message):
    prepared, affine, params = case
    params = deepcopy(params)
    params.update(change)

    def forbidden(**kwargs):
        raise AssertionError("Refused Gaussian choices must fail before reading arrays")

    monkeypatch.setattr(co2_inference, "materialize_co2_inputs", forbidden)
    with pytest.raises(ValueError, match=message):
        execution.run_co2(params=params, prepared=prepared, affine_artifact=affine)


def test_gaussian_requires_double_precision(case):
    prepared, affine, params = case
    with pytest.raises(ValueError, match="floatX=float64"):
        execution.run_co2(params=params, prepared=prepared, affine_artifact=affine)


def test_lognormal_or_mixed_biosphere_priors_are_refused():
    for family, selector in (("lognormal", [True, True]), ("gaussian", [True, False])):
        biosphere = SimpleNamespace(data=xr.Dataset({"biosphere": ("region", selector)}, attrs={"family": family}))
        with pytest.raises(ValueError, match="biosphere prior"):
            check_co2_gaussian_model({"no_model_error": True, "biosphere_prior": biosphere})


def test_default_sampling_stays_nuts(case, monkeypatch):
    _, _, params = case
    params["sampling"] = {"draws": 5, "tune": 5, "chains": 1, "random_seed": 3}
    model = Co2ModelConfig.from_params(params).sampler_arguments()
    default = _resolve_co2_sampling(params, model=model)
    explicit = _resolve_co2_sampling({"sampling": {**params["sampling"], "method": "nuts"}}, model=model)
    assert default == explicit and default.method == "nuts"
    assert default.sampler.nuts_sampler == "numpyro" and default.sampler.chains == 1

    def forbidden(*args, **kwargs):
        raise AssertionError("The default method must not reach the Gaussian estimator")

    from openghg_inversions.rhime.co2 import co2_gaussian

    monkeypatch.setattr(co2_gaussian, "sample_co2_gaussian", forbidden)
    monkeypatch.setattr(co2_inference, "build_co2_inference", lambda *args: ("built", None))
    monkeypatch.setattr(co2_inference, "sample_rhime_model", lambda built, sampler: xr.DataTree())
    monkeypatch.setattr(co2_inference, "annotate_co2_inference", lambda trace, *args: trace)
    _, actual = co2_inference.sample_co2_inference(None, {}, default.sampler)
    assert actual["algorithm"] == "nuts"
