"""Configured exact fixed-noise Gaussian estimation for the single-tracer CO2 recipe."""

from copy import deepcopy
from dataclasses import replace
import json
import logging
import warnings
from types import SimpleNamespace

import mpmath
import numpy as np
import pandas as pd
import pytest
import scipy.linalg as sl
import xarray as xr

from examples.co2_handoffs import configuration, synthetic_inputs
from openghg_inversions.rhime.co2 import execution, handoffs
from openghg_inversions.rhime.co2 import inference as co2_inference
from openghg_inversions.inference.gls import paige_gls, psd_factor
from openghg_inversions.rhime.co2 import co2_gaussian
from openghg_inversions.rhime.co2.co2_gaussian import GAUSSIAN_ALGORITHM, check_co2_gaussian_model
from openghg_inversions.rhime.co2.co2_outputs import validate_co2_trace
from openghg_inversions.rhime.co2.configuration import Co2ModelConfig, _resolve_co2_sampling
from openghg_inversions.rhime.co2.model_mismatch import (
    EmpiricalBayesMismatch, GaussianMarginal, empirical_bayes_mismatch, parse_fixed_model_mismatch,
    per_site_mismatch, type2_ml_mismatch,
)
from scipy.stats import multivariate_normal
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
    monkeypatch.setattr(co2_gaussian, "materialize_co2_inputs", forbidden)
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

    monkeypatch.setattr(co2_gaussian, "sample_co2_gaussian", forbidden)
    monkeypatch.setattr(co2_inference, "build_co2_inference", lambda *args: ("built", None))
    monkeypatch.setattr(co2_inference, "sample_rhime_model", lambda built, sampler: xr.DataTree())
    monkeypatch.setattr(co2_inference, "annotate_co2_inference", lambda trace, *args: trace)
    _, actual = co2_inference.sample_co2_inference(None, {}, default.sampler)
    assert actual["algorithm"] == "nuts"


def test_calibrated_gaussian_biosphere_prior_is_used_and_recorded(case, double_precision):
    prepared, _, params = case
    inputs = prepared.inv_inputs
    covariance = inputs.alpha_prior_covariance * 1.5
    calibration = xr.Dataset(
        {"baseline_mean": inputs.alpha_prior_mean, "baseline_covariance": inputs.alpha_prior_covariance,
         "covariance": covariance, "biosphere": xr.ones_like(inputs.alpha_prior_mean, dtype=bool)},
        attrs={"family": "gaussian", "calibration_report": "{}"},
    )
    biosphere = SimpleNamespace(data=calibration, to_dataset=lambda: calibration)
    model = {**Co2ModelConfig.from_params(params).model_arguments(), "biosphere_prior": biosphere}
    sampling = _resolve_co2_sampling(params, model=model)
    trace, actual = co2_inference.sample_co2_inference(prepared, model, sampling.sampler, method="gaussian")
    exact = trace["gaussian_posterior"].dataset
    assert "state_prior_substitution" not in actual and "state_prior_substitution" not in exact.attrs
    np.testing.assert_allclose(exact.prior_covariance.values[:2, :2], covariance.values, rtol=1e-12)
    assert json.loads(trace.attrs["rhime_model_metadata"])["prior"] == "gaussian"
    assert "biosphere_prior" in trace.children
    altered = calibration.assign(baseline_covariance=covariance)
    biosphere = SimpleNamespace(data=altered, to_dataset=lambda: altered)
    with pytest.raises(ValueError, match="baseline arithmetic moments"):
        co2_inference.sample_co2_inference(prepared, {**model, "biosphere_prior": biosphere}, sampling.sampler,
                                           method="gaussian")


def _dense_inputs(aggregation, error):
    index = np.arange(aggregation.shape[0])
    return xr.Dataset(
        {"mf_error": ("nmeasure", error), "aggregation_error_covariance": (("nmeasure", "nmeasure_cov"), aggregation)},
        coords={"nmeasure": index, "nmeasure_cov": index},
    )


def test_dense_noise_keeps_small_aggregation_modes_above_the_mismatch():
    """Aggregation modes from 1e2 down to 1e-14 with R = 1e-12: none may be replaced by R alone."""
    rng = np.random.default_rng(7)
    nobs, n, mismatch = 60, 4, 1e-12
    vectors = np.linalg.qr(rng.normal(size=(nobs, nobs)))[0]
    aggregation = (vectors * np.logspace(2, -14, nobs)) @ vectors.T
    aggregation = 0.5 * (aggregation + aggregation.T)
    noise = aggregation + mismatch * np.eye(nobs)
    design = rng.normal(size=(nobs, n))
    prior_mean, prior_factor = rng.normal(size=n), np.diag([1.0, 2.0, 0.5, 1.5])
    y = design @ (prior_mean + prior_factor @ rng.normal(size=n)) + np.linalg.cholesky(noise) @ rng.normal(size=nobs)

    factor, record = co2_gaussian._co2_noise_factor(
        _dense_inputs(aggregation, np.full(nobs, np.sqrt(mismatch))), SimpleNamespace(aggregation_error_mode="dense"),
        0.0,
    )
    assert record["noise_factor_method"] == "cholesky(A + R)"
    np.testing.assert_allclose(factor @ factor.T, noise, rtol=0, atol=1e-14)

    mpmath.mp.dps = 40
    to_mp = lambda array: mpmath.matrix(array.tolist())  # noqa: E731
    noise_inverse = (to_mp(aggregation) + mpmath.diag([mpmath.mpf(mismatch)] * nobs)) ** -1
    prior_precision = to_mp(prior_factor @ prior_factor.T) ** -1
    covariance = (to_mp(design).T * noise_inverse * to_mp(design) + prior_precision) ** -1
    mean = covariance * (to_mp(design).T * noise_inverse * to_mp(y[:, None]) + prior_precision * to_mp(prior_mean[:, None]))
    reference_mean = np.array([float(value) for value in mean])
    reference_sd = np.sqrt(np.diag(np.array(covariance.tolist(), dtype=float)))

    def errors(noise_factor):
        posterior = paige_gls(y, design, noise_factor, prior_mean, prior_factor)
        return (np.max(np.abs(posterior.mean - reference_mean) / reference_sd),
                np.max(np.abs(np.sqrt(np.diag(posterior.covariance)) / reference_sd - 1)))

    mean_error, sd_error = errors(factor)
    assert mean_error < 0.01 and sd_error < 1e-3
    # A relative eigenvalue tolerance drops modes comparable to R and visibly biases the posterior.
    truncated = psd_factor(aggregation, rtol=nobs * np.finfo(float).eps)
    assert truncated.diagnostics["dropped_positive_max"] > 0.5 * mismatch
    mean_error, sd_error = errors(np.hstack([truncated.factor, np.sqrt(mismatch) * np.eye(nobs)]))
    assert mean_error > 0.3 and sd_error > 0.05


def test_singular_dense_noise_falls_back_to_clipped_eigenvalues():
    rng = np.random.default_rng(8)
    low_rank = rng.normal(size=(6, 2))
    factor, record = co2_gaussian._co2_noise_factor(
        _dense_inputs(low_rank @ low_rank.T, np.zeros(6)), SimpleNamespace(aggregation_error_mode="dense"), 0.0,
    )
    assert record["noise_factor_method"] == "[eigh_clip(A) | diag(sqrt(R))]"
    assert record["aggregation_psd_factor"]["dropped_positive_max"] == 0.0
    np.testing.assert_allclose(factor @ factor.T, low_rank @ low_rank.T, atol=1e-12)


LOGGER = "openghg_inversions.rhime.co2.co2_gaussian"


def _run_gaussian(prepared, params, **overrides):
    model = {**Co2ModelConfig.from_params(params).model_arguments(), **overrides}
    sampling = _resolve_co2_sampling(params, model=model)
    return co2_inference.sample_co2_inference(prepared, model, sampling.sampler, method="gaussian")


def test_scalar_site_and_measurement_mismatch_forms_agree(case, double_precision):
    prepared, _, params = case
    results = {}
    for name, value in (("scalar", 0.3), ("site", {"MHD": 0.3})):
        changed = deepcopy(params)
        changed["likelihood"]["fixed_model_mismatch"] = value
        results[name] = _run_gaussian(prepared, changed)
    array = xr.full_like(prepared.inv_inputs.mf, 0.3)
    results["array"] = _run_gaussian(prepared, params, fixed_model_mismatch=array)
    sources = {"scalar": "specified scalar", "site": "specified per site", "array": "specified per measurement"}
    reference = results["scalar"][0]["gaussian_posterior"].dataset
    for name, (trace, actual) in results.items():
        exact = trace["gaussian_posterior"].dataset
        xr.testing.assert_allclose(exact[["posterior_mean", "posterior_covariance"]],
                                   reference[["posterior_mean", "posterior_covariance"]], rtol=1e-12)
        assert actual["model_mismatch"]["source"] == sources[name]
        assert actual["model_mismatch"]["sigma_by_site"] == {"MHD": 0.3}
        assert json.loads(exact.attrs["model_mismatch"])["source"] == sources[name]
        np.testing.assert_allclose(trace.constant_data.fixed_model_mismatch, 0.3)


def test_per_site_mismatch_names_exactly_the_observed_sites(case, double_precision):
    observations = xr.DataArray([1.0, 2.0, 3.0], dims="nmeasure", coords={"site": ("nmeasure", ["A", "B", "A"])})
    np.testing.assert_array_equal(per_site_mismatch({"A": 0.1, "B": 0.2}, observations), [0.1, 0.2, 0.1])
    with pytest.raises(ValueError, match=r"missing: \['B'\]"):
        per_site_mismatch({"A": 0.1}, observations)
    with pytest.raises(ValueError, match=r"unknown: \['C'\]"):
        per_site_mismatch({"A": 0.1, "B": 0.2, "C": 0.3}, observations)
    prepared, affine, params = case
    params["likelihood"]["fixed_model_mismatch"] = {"XXX": 0.1}
    with pytest.raises(ValueError, match="unknown"):
        execution.run_co2(params=params, prepared=prepared, affine_artifact=affine)
    params["sampling"] = {"draws": 5, "tune": 5, "chains": 1}
    with pytest.raises(ValueError, match='require sampling.method = "gaussian"'):
        execution.run_co2(params=params, prepared=prepared, affine_artifact=affine)


def test_mismatch_configuration_forms():
    path = "likelihood.fixed_model_mismatch"
    assert parse_fixed_model_mismatch(0.4, path) == 0.4
    assert dict(parse_fixed_model_mismatch({"MHD": 0.4}, path)) == {"MHD": 0.4}
    assert parse_fixed_model_mismatch({"empirical_bayes": "prior_residual_rms", "floor": 0.2}, path) == \
        EmpiricalBayesMismatch(mode="prior_residual_rms", floor=0.2)
    assert parse_fixed_model_mismatch(
        {"empirical_bayes": "prior_residual_moment", "floor": 0.0, "allow_zero": True}, path,
    ).allow_zero
    for bad, message in (({"empirical_bayes": "prior_residual_moment", "floor": 0.1, "ceiling": 1.0}, "Unknown"),
                         ({"empirical_bayes": "prior_residual_moment"}, "floor is required.*NUTS"),
                         ({"empirical_bayes": "type_ii", "floor": 0.1}, "empirical_bayes must be one of"),
                         ({"MHD": -1.0}, "non-negative"), ({}, "empty")):
        with pytest.raises(ValueError, match=message):
            parse_fixed_model_mismatch(bad, path)


@pytest.mark.parametrize("mode,floor", [("prior_residual_moment", 0.0), ("prior_residual_rms", 0.0),
                                        ("prior_residual_moment", 5.0)])
def test_empirical_bayes_matches_hand_computed_prior_residual(case, caplog, double_precision, mode, floor):
    prepared, _, params = case
    params["likelihood"]["fixed_model_mismatch"] = {"empirical_bayes": mode, "floor": floor}
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        trace, actual = _run_gaussian(prepared, params)
    inputs = prepared.inv_inputs
    residual = (inputs.mf - inputs.fixed_prior_contribution - inputs.H.values @ inputs.alpha_prior_mean.values
                - inputs.H_bc.values[:, 0] * 1.0).values
    prior_variance = (np.einsum("ij,jk,ik->i", inputs.H.values, inputs.alpha_prior_covariance.values, inputs.H.values)
                      + inputs.H_bc.values[:, 0] ** 2 * 0.1**2 + 0.1**2)
    aggregation = (inputs.low_rank_factor**2).sum("agg_rank").values + inputs.diagonal_residual_variance.values
    error = inputs.mf_error.values ** 2
    mean_r2 = np.mean(residual**2)
    raw_variance = mean_r2 if mode == "prior_residual_rms" else mean_r2 - prior_variance.mean() - aggregation.mean() - error.mean()
    expected = max(np.sqrt(max(raw_variance, 0.0)), floor)

    record = json.loads(trace["gaussian_posterior"].dataset.attrs["model_mismatch"])
    site = record["empirical_bayes"]["by_site"]["MHD"]
    assert record["source"] == "empirical Bayes" and record["empirical_bayes"]["label"] == "empirical Bayes"
    np.testing.assert_allclose(
        [site["mean_residual_squared"], site["mean_prior_state_variance"], site["mean_aggregation_variance"],
         site["mean_mf_error_variance"], site["raw_variance"], site["sigma"]],
        [mean_r2, prior_variance.mean(), aggregation.mean(), error.mean(), raw_variance, expected], rtol=1e-10,
    )
    assert site["floor_applied"] == (floor > np.sqrt(max(raw_variance, 0.0)))
    assert actual["model_mismatch"]["sigma_by_site"]["MHD"] == pytest.approx(expected, rel=1e-10)
    assert actual["model_mismatch"]["empirical_bayes_mode"] == mode
    np.testing.assert_allclose(trace.constant_data.fixed_model_mismatch, expected, rtol=1e-10)
    eb = [message for message in caplog.messages if "Empirical Bayes" in message]
    assert len(eb) == 1
    assert mode in eb[0] and "does not include sigma uncertainty" in eb[0] and "used twice" in eb[0]
    assert sum("moment-matched" in message for message in caplog.messages) == 1
    assert actual["state_prior_substitution"] == "lognormal→normal (moment-matched)"
    assert trace["gaussian_posterior"].dataset.attrs["state_prior_substitution"] == actual["state_prior_substitution"]


def test_specified_mismatch_logs_only_the_prior_substitution(case, caplog, double_precision):
    prepared, _, params = case
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _run_gaussian(prepared, params)
    assert len(caplog.messages) == 1 and "lognormal→normal (moment-matched)" in caplog.messages[0]


def test_prior_residual_moment_recovers_a_known_mismatch():
    """Two sites with 20000 synthetic prior residuals each; the moment estimate is within sampling error."""
    rng = np.random.default_rng(12)
    n, truth = 20000, {"A": 0.7, "B": 0.25}
    sites = np.repeat(["A", "B"], n)
    prior_variance, aggregation, error = rng.uniform(0.5, 1.5, 2 * n), rng.uniform(0.1, 0.3, 2 * n), np.full(2 * n, 0.01)
    sigma2 = np.array([truth[site] ** 2 for site in sites])
    total = prior_variance + aggregation + error + sigma2
    residual = np.sqrt(total) * rng.standard_normal(2 * n)
    sigma, record = empirical_bayes_mismatch(
        EmpiricalBayesMismatch(floor=0.0), sites=sites, residual=residual, prior_state_variance=prior_variance,
        aggregation_variance=aggregation, error_variance=error,
    )
    for site, value in truth.items():
        rows = sites == site
        standard_error = np.sqrt(2 / n) * np.sqrt(np.mean(total[rows] ** 2))
        assert abs(record["by_site"][site]["raw_variance"] - value**2) < 4 * standard_error
        assert np.all(sigma[rows] == record["by_site"][site]["sigma"])
    rms, _ = empirical_bayes_mismatch(
        EmpiricalBayesMismatch(mode="prior_residual_rms", floor=0.0), sites=sites, residual=residual,
        prior_state_variance=prior_variance, aggregation_variance=aggregation, error_variance=error,
    )
    assert np.all(rms > sigma)


def test_negligible_floor_bound_mismatch_is_refused_unless_allowed():
    sites = np.array(["A", "A", "B", "B"])
    terms = dict(sites=sites, residual=np.zeros(4), prior_state_variance=np.ones(4),
                 aggregation_variance=np.full(4, 0.2), error_variance=np.full(4, 1e-12))
    with pytest.raises(ValueError, match="at the floor 0 at every site.*allow_zero"):
        empirical_bayes_mismatch(EmpiricalBayesMismatch(floor=0.0), **terms)
    sigma, record = empirical_bayes_mismatch(EmpiricalBayesMismatch(floor=0.0, allow_zero=True), **terms)
    assert np.all(sigma == 0) and all(site["at_floor"] for site in record["by_site"].values())
    sigma, _ = empirical_bayes_mismatch(EmpiricalBayesMismatch(floor=0.5), **terms)
    np.testing.assert_array_equal(sigma, 0.5)


def test_moment_mode_fails_when_a_shared_mode_dominates_the_prior_spread():
    """One shared boundary-like scalar of variance 100 swamps a true sigma of 1 at every site.

    Per-site means of r**2 then share one draw of that scalar, so the subtraction of its expected
    variance is dominated by one chi-squared(1) sample: the estimate lands on the floor or far from
    the truth in essentially every draw, although each site has 400 observations.
    """
    sites = np.repeat(["A", "B", "C", "D", "E"], 400)
    shared_variance, truth = 100.0, 1.0
    prior_variance = np.full(sites.size, shared_variance)
    recovered = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        residual = np.sqrt(shared_variance) * rng.standard_normal() + truth * rng.standard_normal(sites.size)
        _, record = empirical_bayes_mismatch(
            EmpiricalBayesMismatch(floor=0.05), sites=sites, residual=residual, prior_state_variance=prior_variance,
            aggregation_variance=np.zeros(sites.size), error_variance=np.zeros(sites.size),
        )
        recovered += all(abs(site["sigma"] - truth) < 0.5 for site in record["by_site"].values())
    assert recovered <= 1


def _two_site(prepared):
    """Relabel the synthetic case's three observations onto two sites, keeping the MultiIndex."""
    index = prepared.inv_inputs.indexes["nmeasure"]
    relabelled = pd.MultiIndex.from_arrays([["MHD", "TAC", "TAC"], index.get_level_values("time")], names=index.names)
    inputs = prepared.inv_inputs.drop_vars(["nmeasure", *index.names]).assign_coords(
        xr.Coordinates.from_pandas_multiindex(relabelled, "nmeasure"))
    sites = xr.Dataset({"averaging_period": ("site", ["1h", "1h"])}, coords={"site": ["MHD", "TAC"]})
    return replace(prepared, rhime_inputs=replace(prepared.rhime_inputs, inv_inputs=inputs, site_metadata=sites))


def test_two_site_per_site_and_empirical_bayes_mismatch(case, double_precision):
    prepared, _, params = case
    prepared = _two_site(prepared)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*MultiIndex.*", category=DeprecationWarning)
        site_params = deepcopy(params)
        site_params["likelihood"]["fixed_model_mismatch"] = {"MHD": 0.2, "TAC": 0.5}
        site_trace, site_actual = _run_gaussian(prepared, site_params)
        array = xr.full_like(prepared.inv_inputs.mf, 0.0).copy(data=[0.2, 0.5, 0.5])
        array_trace, _ = _run_gaussian(prepared, params, fixed_model_mismatch=array)
        eb_params = deepcopy(params)
        eb_params["likelihood"]["fixed_model_mismatch"] = {"empirical_bayes": "prior_residual_moment", "floor": 0.05}
        eb_trace, eb_actual = _run_gaussian(prepared, eb_params)
    assert site_actual["model_mismatch"]["sigma_by_site"] == {"MHD": 0.2, "TAC": 0.5}
    np.testing.assert_allclose(site_trace.constant_data.fixed_model_mismatch, [0.2, 0.5, 0.5])
    for name in ("posterior_mean", "posterior_covariance"):
        np.testing.assert_allclose(site_trace["gaussian_posterior"].dataset[name],
                                   array_trace["gaussian_posterior"].dataset[name], rtol=1e-12)

    inputs = prepared.inv_inputs
    residual = (inputs.mf - inputs.fixed_prior_contribution - inputs.H.values @ inputs.alpha_prior_mean.values
                - inputs.H_bc.values[:, 0]).values
    prior_variance = (np.einsum("ij,jk,ik->i", inputs.H.values, inputs.alpha_prior_covariance.values, inputs.H.values)
                      + inputs.H_bc.values[:, 0] ** 2 * 0.1**2 + 0.1**2)
    aggregation = (inputs.low_rank_factor**2).sum("agg_rank").values + inputs.diagonal_residual_variance.values
    total = residual**2 - prior_variance - aggregation - inputs.mf_error.values**2
    expected = {site: max(np.sqrt(max(total[rows].mean(), 0.0)), 0.05)
                for site, rows in (("MHD", [0]), ("TAC", [1, 2]))}
    by_site = json.loads(eb_trace["gaussian_posterior"].dataset.attrs["model_mismatch"])["empirical_bayes"]["by_site"]
    assert set(by_site) == {"MHD", "TAC"} and by_site["TAC"]["n_observations"] == 2
    for site, value in expected.items():
        assert by_site[site]["sigma"] == pytest.approx(value, rel=1e-10)
        assert eb_actual["model_mismatch"]["sigma_by_site"][site] == pytest.approx(value, rel=1e-10)
    np.testing.assert_allclose(eb_trace.constant_data.fixed_model_mismatch,
                               [expected["MHD"], expected["TAC"], expected["TAC"]], rtol=1e-10)


def test_numpy_scalar_mismatch_is_recorded_as_a_scalar(case, double_precision):
    prepared, _, params = case
    _, actual = _run_gaussian(prepared, params, fixed_model_mismatch=np.float32(0.25))
    assert actual["model_mismatch"]["source"] == "specified scalar"
    assert actual["model_mismatch"]["sigma_by_site"]["MHD"] == pytest.approx(0.25)


def test_prior_predictive_maps_site_tables_and_refuses_empirical_bayes(case, tmp_path):
    prepared, affine, params = case
    manifest = handoffs.save_preparation(prepared, output_dir=tmp_path / "prepare", affine_artifact=affine,
                                         **params["prepared_inputs"])
    params["likelihood"]["fixed_model_mismatch"] = {"MHD": 0.3}
    result = handoffs.prior_predictive(manifest, model=Co2ModelConfig.from_params(params),
                                       output_dir=tmp_path / "prior", draws=5, random_seed=1)
    assert result["status"] == "pass"
    params["likelihood"]["fixed_model_mismatch"] = {"empirical_bayes": "prior_residual_moment", "floor": 0.1}
    with pytest.raises(ValueError, match="empirical-Bayes"):
        handoffs.prior_predictive(manifest, model=Co2ModelConfig.from_params(params),
                                  output_dir=tmp_path / "eb-prior", draws=5)


def _marginal_case(seed=0, n_per_site=20, exact=False):
    rng = np.random.default_rng(seed)
    codes = np.repeat([0, 1, 2], n_per_site)
    n = codes.size
    state, aggregation = rng.normal(size=(n, 5)), 0.5 * rng.normal(size=(n, 4))
    diagonal, error = rng.uniform(0.1, 0.3, n), np.full(n, 0.01)
    residual = 2 * rng.normal(size=n)
    if exact:
        marginal = GaussianMarginal(residual, codes, state, error, dense=aggregation @ aggregation.T + np.diag(diagonal))
    else:
        marginal = GaussianMarginal(residual, codes, np.hstack([state, aggregation]), diagonal + error)
    covariance = state @ state.T + aggregation @ aggregation.T + np.diag(diagonal + error)
    return marginal, residual, codes, covariance


@pytest.mark.parametrize("exact", [False, True], ids=["low-rank", "exact"])
def test_type2_marginal_likelihood_and_gradient(exact):
    marginal, residual, codes, covariance = _marginal_case(exact=exact)
    log_sigma = np.log([0.5, 1.0, 2.0])
    value, gradient = marginal(log_sigma)
    expected = multivariate_normal(np.zeros(residual.size), covariance + np.diag(np.exp(2 * log_sigma)[codes]))
    assert value == pytest.approx(expected.logpdf(residual), rel=1e-12)
    step = 1e-6
    finite = [(marginal(log_sigma + step * unit)[0] - marginal(log_sigma - step * unit)[0]) / (2 * step)
              for unit in np.eye(3)]
    np.testing.assert_allclose(gradient, finite, rtol=1e-6)


@pytest.mark.parametrize("truth", [(1.0, 1.0, 1.0, 1.0), (0.3, 0.8, 1.6, 3.0)], ids=["homoscedastic", "heteroscedastic"])
def test_type2_ml_recovers_sigma_despite_a_shared_mode(truth):
    """One shared mode of variance 100 defeats the moment mode but not the marginal likelihood."""
    rng = np.random.default_rng(3)
    n_per_site = 400
    sites = np.repeat(["A", "B", "C", "D"], n_per_site)
    n = sites.size
    state = np.hstack([np.full((n, 1), 10.0), 0.3 * rng.normal(size=(n, 6))])
    aggregation = 0.2 * rng.normal(size=(n, 3))
    diagonal, error = np.full(n, 0.05), np.full(n, 0.01)
    sigma_true = np.repeat(truth, n_per_site)
    residual = (state @ rng.standard_normal(state.shape[1]) + aggregation @ rng.standard_normal(3)
                + np.sqrt(diagonal + error + sigma_true**2) * rng.standard_normal(n))
    aggregation_variance = np.sum(aggregation**2, axis=1) + diagonal
    choice = EmpiricalBayesMismatch(mode="type2_ml", floor=0.05)
    _, record = type2_ml_mismatch(
        choice, sites=sites, residual=residual, state_factor=state, aggregation_factor=aggregation,
        aggregation_diagonal=diagonal, aggregation_dense=None, aggregation_variance=aggregation_variance,
        error_variance=error, aggregation_representation="test",
    )
    assert record["converged"] and record["log_marginal_likelihood"] >= record["log_marginal_likelihood_start"]
    for site, value in zip("ABCD", truth):
        assert record["by_site"][site]["sigma"] == pytest.approx(value, rel=0.15)
    moment, _ = empirical_bayes_mismatch(
        EmpiricalBayesMismatch(floor=0.05), sites=sites, residual=residual,
        prior_state_variance=np.sum(state**2, axis=1), aggregation_variance=aggregation_variance, error_variance=error,
    )
    assert np.max(np.abs(moment / sigma_true - 1)) > 0.5


def test_type2_ml_floor_and_allow_zero():
    sites = np.repeat(["A", "B"], 10)
    terms = dict(sites=sites, residual=np.zeros(20), state_factor=np.ones((20, 1)),
                 aggregation_factor=np.zeros((20, 0)), aggregation_diagonal=np.full(20, 0.2), aggregation_dense=None,
                 aggregation_variance=np.full(20, 0.2), error_variance=np.full(20, 1e-12),
                 aggregation_representation="test")
    sigma, record = type2_ml_mismatch(EmpiricalBayesMismatch(mode="type2_ml", floor=0.3), **terms)
    np.testing.assert_allclose(sigma, 0.3)
    assert all(site["at_floor"] and site["floor_applied"] for site in record["by_site"].values())
    with pytest.raises(ValueError, match="at the floor 0 at every site"):
        type2_ml_mismatch(EmpiricalBayesMismatch(mode="type2_ml", floor=0.0), **terms)
    sigma, _ = type2_ml_mismatch(EmpiricalBayesMismatch(mode="type2_ml", floor=0.0, allow_zero=True), **terms)
    assert np.all(sigma < 1e-5)


def test_type2_ml_configuration():
    path = "likelihood.fixed_model_mismatch"
    choice = parse_fixed_model_mismatch(
        {"empirical_bayes": "type2_ml", "floor": 0.1, "aggregation": "exact", "max_iterations": 50}, path)
    assert (choice.mode, choice.aggregation, choice.max_iterations, choice.rank) == ("type2_ml", "exact", 50, 512)
    with pytest.raises(ValueError, match="require empirical_bayes"):
        parse_fixed_model_mismatch({"empirical_bayes": "prior_residual_moment", "floor": 0.1, "rank": 10}, path)
    with pytest.raises(ValueError, match="low_rank' or 'exact"):
        parse_fixed_model_mismatch({"empirical_bayes": "type2_ml", "floor": 0.1, "aggregation": "dense"}, path)


@pytest.mark.parametrize("aggregation", ["low_rank", "exact"])
def test_type2_ml_through_the_configured_estimator(case, caplog, double_precision, aggregation):
    prepared, _, params = case
    prepared = _two_site(prepared)
    params["likelihood"]["fixed_model_mismatch"] = {"empirical_bayes": "type2_ml", "floor": 0.05,
                                                    "aggregation": aggregation}
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        trace, actual = _run_gaussian(prepared, params)
    record = json.loads(trace["gaussian_posterior"].dataset.attrs["model_mismatch"])["empirical_bayes"]
    assert record["mode"] == "type2_ml" and record["aggregation"] == aggregation and record["label"] == "empirical Bayes"
    assert "exact dense" in record["aggregation_representation"] if aggregation == "exact" else "Woodbury" in \
        record["aggregation_representation"]
    for name in ("optimiser", "iterations", "converged", "log_marginal_likelihood", "log_marginal_likelihood_start",
                 "log_marginal_likelihood_floor"):
        assert name in record
    assert record["log_marginal_likelihood"] >= record["log_marginal_likelihood_floor"] - 1e-9
    summary = actual["model_mismatch"]
    assert summary["empirical_bayes_mode"] == "type2_ml" and set(summary["sigma_by_site"]) == {"MHD", "TAC"}
    assert summary["type2_ml_converged"] == record["converged"]
    np.testing.assert_allclose(
        trace.constant_data.fixed_model_mismatch,
        [summary["sigma_by_site"]["MHD"], summary["sigma_by_site"]["TAC"], summary["sigma_by_site"]["TAC"]],
    )
    eb = [message for message in caplog.messages if "Empirical Bayes model mismatch" in message]
    assert len(eb) == 1 and "type2_ml" in eb[0] and "used twice" in eb[0]
