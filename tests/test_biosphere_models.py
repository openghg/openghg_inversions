"""The same calibrated state in ordinary, cached, and linked recipes."""

import numpy as np
import pytest
import pytensor
import xarray as xr

from openghg_inversions.biosphere_prior import BIOSPHERE_PRIOR_SCOPE, calibrate_biosphere_prior
from openghg_inversions.correlated_state import CorrelatedPrior
from openghg_inversions.models.biosphere import resolve_retained_flux_prior
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import (
    build_co2_model,
    build_co2_cached_sigma_model,
    build_co2_o2_model,
    build_co2_o2_cached_sigma_model,
)
from boundary_test_utils import bound_model_arguments
from test_biosphere_prior import case
from test_co2_o2_fixed_ou import _prepared, _kwargs


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64", cast_policy="numpy+floatX"):
        yield


def state_and_mean(model, point):
    return model.compile_fn(
        model.replace_rvs_by_values([model.flux_scaling, model.modelled_concentration]),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )(point)


@pytest.mark.parametrize("full", [True, False])
def test_full_and_enhancement_ordinary_cached_prediction_density_and_no_default_change(full):
    baseline, inputs = case()
    calibration = resolve_retained_flux_prior(
        baseline, calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=0.5),
    )
    observation = xr.DataArray(
        [1.2, 0.4, 0.8, -0.5],
        dims="nmeasure",
        coords={
            "nmeasure": np.arange(4),
            "site": ("nmeasure", ["A", "A", "B", "B"]),
            "time": ("nmeasure", np.datetime64("2021-01-01") + np.arange(4).astype("timedelta64[h]")),
        },
        attrs={"units": "ppm"},
    )
    h = xr.DataArray(
        [[-2, 1, -1, 2, 0.3], [-1, 0.8, -2, 1, 0.7], [0.1, 0.2, -0.3, 1, 0.4], [-0.5, 1, -0.7, 2, 0.8]],
        dims=("nmeasure", "state"),
        coords={"nmeasure": observation.nmeasure, **baseline.mean.coords},
    )
    kwargs = dict(
        flux_sensitivity=h,
        retained_prior=baseline,
        observations=observation,
        observation_error=xr.full_like(observation, 0.2),
        fixed_prior_contribution=xr.zeros_like(observation).rename("fixed_prior_contribution"),
        aggregation_error=resolve_aggregation_error(observation.to_dataset(name="mf"), "none"),
    )
    if full:
        kwargs["boundary_sensitivity"] = xr.DataArray(
            [[0.1], [0.2], [0.3], [0.4]],
            dims=("nmeasure", "bc_region"),
            coords={"nmeasure": observation.nmeasure, "bc_region": ["level"]},
        )
        kwargs["bc_prior"] = {"pdf": "normal", "mu": 0, "sigma": 0.2}
    kwargs = bound_model_arguments(kwargs)
    likelihood = dict(tau_hours=5, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    original = build_co2_model(
        **kwargs, likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=likelihood
    )
    identity = resolve_retained_flux_prior(
        baseline, calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=1),
    )
    same = build_co2_model(
        **{**kwargs, "retained_prior": identity},
        likelihood_builder=add_fixed_ou_gaussian_likelihood,
        likelihood_kwargs=likelihood,
    )
    point = original.initial_point()
    np.testing.assert_allclose(original.compile_logp()(point), same.compile_logp()(point), rtol=1e-14)
    cached = build_co2_cached_sigma_model(**{**kwargs, "retained_prior": calibration}, **likelihood)
    ordinary = build_co2_model(
        **{**kwargs, "retained_prior": calibration},
        likelihood_builder=add_fixed_ou_gaussian_likelihood,
        likelihood_kwargs=likelihood,
    )
    point = ordinary.initial_point()
    point["flux_scaling_latent"] = np.array([0.1, -0.3, 0.5, -0.2, 0.4])
    point["ou_site_amplitude_log__"] = np.log([0.4, 0.6])
    cached_point = cached.model.initial_point()
    cached_point.update(point)
    x, mean = state_and_mean(ordinary, point)
    xc, mc = state_and_mean(cached.model, cached_point)
    np.testing.assert_allclose(x, xc, rtol=1e-14)
    np.testing.assert_allclose(mean, mc, rtol=1e-14)
    np.testing.assert_allclose(
        ordinary.compile_logp()(point), cached.model.compile_logp()(cached_point), rtol=1e-12
    )
    np.testing.assert_allclose(
        cached.target.log_likelihood_from_mean(mean, np.array([0.4, 0.6])),
        ordinary.compile_logp(vars=[ordinary.y])(point),
        rtol=1e-12,
    )
    assert len([rv for rv in ordinary.free_RVs if rv.name == "flux_scaling_latent"]) == 1
    for model, values in ((ordinary, point), (cached.model, cached_point)):
        assert_normalized_density_and_gradient(model, values, calibration, kwargs, full=full)
    np.testing.assert_allclose(x, physical_state(calibration, point["flux_scaling_latent"]), rtol=1e-14)
    # GPP and TER are still separate columns and states.
    assert x.shape == (5,) and h.shape == (4, 5)


def test_linked_species_share_one_calibration_and_density():
    prepared = _prepared()
    kwargs = _kwargs(prepared)
    baseline = CorrelatedPrior.from_lognormal(prepared.retained_prior)
    mean = baseline.mean
    weights = xr.DataArray(
        [[-10, 0, 0, 0, 0], [0, 8, 0, 0, 0]],
        dims=("process", "state"),
        coords={"process": ["GPP", "TER"], **mean.coords},
        attrs={"units": "kg/year"},
    )
    reference = xr.DataArray(
        [-10.0, 8.0], dims="process", coords={"process": ["GPP", "TER"]}, attrs={"units": "kg/year"}
    )
    calibration = resolve_retained_flux_prior(baseline, calibrate_biosphere_prior(
        baseline,
        biosphere=mean.copy(data=[True, True, False, False, False]),
        weights=weights,
        reference=reference,
        residual_sd_multiplier=0.5,
    ))
    likelihood = dict(tau_hours=5, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    ordinary = build_co2_o2_model(**{**kwargs, "retained_prior": calibration}, **likelihood)
    cached = build_co2_o2_cached_sigma_model(**{**kwargs, "retained_prior": calibration}, **likelihood)
    point = ordinary.initial_point()
    point["flux_scaling_latent"] = np.array([0.1, -0.3, 0.2, -0.1, 0.4])
    x, mean = state_and_mean(ordinary, point)
    xc, mc = state_and_mean(cached.model, point)
    np.testing.assert_allclose(x, xc)
    np.testing.assert_allclose(mean, mc)
    np.testing.assert_allclose(ordinary.compile_logp()(point), cached.model.compile_logp()(point), rtol=1e-12)
    assert len([rv for rv in cached.model.free_RVs if rv.name.startswith("flux_scaling")]) == 1
    assert not hasattr(cached.model, "biosphere_prior")
    for model in (ordinary, cached.model):
        assert_normalized_density_and_gradient(model, point, calibration, kwargs, linked=True)
    np.testing.assert_allclose(x, physical_state(calibration, point["flux_scaling_latent"]), rtol=1e-14)


def test_linked_runner_prior_prediction_serializes_calibration_and_reconstructs(tmp_path):
    import json
    import pymc as pm
    from openghg_inversions.models import get_coord_registry, restore_inferencedata_coords
    from openghg_inversions.rhime.co2 import build_rhime_co2_o2, annotate_co2_o2_trace
    from openghg_inversions.serialization import save_trace, load_trace
    from test_rhime_co2_o2 import _independent_error

    prepared = _prepared()
    mean = prepared.retained_prior.mean
    weights = xr.DataArray(
        [[-10, 0, 0, 0, 0], [0, 8, 0, 0, 0]],
        dims=("process", "state"),
        coords={"process": ["GPP", "TER"], **mean.coords},
        attrs={"units": "kg/year"},
    )
    reference = xr.DataArray(
        [-10.0, 8.0], dims="process", coords={"process": ["GPP", "TER"]}, attrs={"units": "kg/year"}
    )
    calibration = calibrate_biosphere_prior(
        CorrelatedPrior.from_lognormal(prepared.retained_prior),
        biosphere=mean.copy(data=[True, True, False, False, False]),
        weights=weights,
        reference=reference,
        residual_sd_multiplier=0.5,
    )
    built = build_rhime_co2_o2(
        prepared_inputs=prepared,
        independent_error_sd=_independent_error(prepared),
        biosphere_prior=calibration,
    )
    assert built.output_contract.metadata["prior"] == "correlated arithmetic-moment lognormal"
    with built.model:
        trace = pm.sample_prior_predictive(draws=4, random_seed=817)
    trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
    annotate_co2_o2_trace(trace, built=built, biosphere_prior=calibration)
    assert json.loads(trace.attrs["rhime_model_metadata"])["biosphere_prior_scope"] == BIOSPHERE_PRIOR_SCOPE
    save_trace(trace, tmp_path / "prior.nc")
    restored = load_trace(tmp_path / "prior.nc")
    np.testing.assert_array_equal(restored.biosphere_prior.covariance, calibration.data.covariance)
    x = restored.prior.flux_scaling
    expected = xr.dot(prepared.co2_sensitivity, x, dim="state")
    np.testing.assert_allclose(
        restored.prior.co2_o2_flux_contribution.sel(species="co2").to_numpy(),
        expected.transpose("chain", "draw", "co2_measure"),
    )


@pytest.mark.parametrize("state_dim", ["state", "flux_state"])
def test_calibrated_prior_rejects_reordered_or_renamed_state_axis(state_dim):
    # Latent factors are applied by position, so a reordered axis would
    # silently move coefficients between columns.
    baseline, inputs = case()
    calibration = resolve_retained_flux_prior(
        baseline, calibrate_biosphere_prior(baseline, **inputs, residual_sd_multiplier=0.5),
    )
    observation = xr.DataArray(
        [1.2, 0.4, 0.8],
        dims="nmeasure",
        coords={"nmeasure": np.arange(3), "site": ("nmeasure", ["A", "A", "B"])},
        attrs={"units": "ppm"},
    )
    labels = baseline.mean.state.values[::-1]
    h = xr.DataArray(
        np.arange(15.0).reshape(3, 5) + 1,
        dims=("nmeasure", state_dim),
        coords={"nmeasure": observation.nmeasure, state_dim: labels},
    )
    match = "same state dimension" if state_dim != "state" else "same order"
    with pytest.raises(ValueError, match=match):
        build_co2_model(
            h,
            retained_prior=calibration,
            observations=observation,
            observation_error=xr.full_like(observation, 0.2),
            fixed_prior_contribution=xr.zeros_like(observation).rename("fixed_prior_contribution"),
            aggregation_error=resolve_aggregation_error(observation.to_dataset(name="mf"), "none"),
        )


def physical_state(prior, z):
    """Independent NumPy arithmetic-moment map, with no model/cache expressions."""
    if getattr(prior, "lognormal", True) is None:
        # Copula marginals: SciPy quantiles of the correlated latent normal.
        return prior.transform_latent(prior.latent_correlation_cholesky.to_numpy() @ z)
    m = prior.mean.to_numpy()
    b = prior.arithmetic_covariance.to_numpy()
    v = np.log1p(b / np.outer(m, m))
    return np.exp(np.log(m) - np.diag(v) / 2 + np.linalg.cholesky(v) @ z)


def assert_normalized_density_and_gradient(model, point, prior, kwargs, *, linked=False, full=False):
    """Check the whole transformed density and gradient against dense equations."""
    from scipy.stats import halfnorm, multivariate_normal, norm

    observation = kwargs["observations"]
    hours = observation.time.to_numpy().astype("datetime64[m]").astype(float) / 60
    if linked:
        groups = np.array([f"{sp}:{site}" for sp, site in zip(observation.species.values, observation.site.values)])
        error = kwargs["independent_error_sd"].to_numpy()
    else:
        groups = observation.site.to_numpy()
        error = kwargs["observation_error"].to_numpy()
    labels = np.unique(groups)
    aggregation = kwargs["aggregation_error"].covariance
    base = np.diag(error**2)
    if aggregation is not None:
        base += aggregation.to_numpy()

    def density(values):
        z = values["flux_scaling_latent"]
        x = physical_state(prior, z)
        mean = kwargs["fixed_prior_contribution"].to_numpy().copy()
        if linked:
            for channel in ("co2", "o2"):
                mean[observation.species.values == channel] += kwargs[f"{channel}_sensitivity"].to_numpy() @ x
        else:
            mean += kwargs["flux_sensitivity"].to_numpy() @ x
        result = norm.logpdf(z).sum()
        if full:
            # Physical boundary coefficient is a direct Normal RV in this fixture.
            mean += np.arange(1, 5) * 0.1 * values["bc"][0]
            result += norm.logpdf(values["bc"], scale=0.2).sum()
        log_amplitude = values["ou_site_amplitude_log__"]
        amplitude = np.exp(log_amplitude)
        covariance = base.copy()
        for index, label in enumerate(labels):
            rows = np.flatnonzero(groups == label)
            covariance[np.ix_(rows, rows)] += amplitude[index]**2 * np.exp(
                -np.abs(hours[rows, None] - hours[None, rows]) / 5
            )
        result += halfnorm.logpdf(amplitude, scale=0.75).sum() + log_amplitude.sum()
        return result + multivariate_normal.logpdf(observation.to_numpy(), mean=mean, cov=covariance)

    np.testing.assert_allclose(model.compile_logp()(point), density(point), rtol=1e-11, atol=1e-11)
    gradient = []
    gradient_by_name = {}
    for variable in model.value_vars:
        value = point[variable.name]
        component_gradient = []
        for index in np.ndindex(value.shape):
            plus = {key: item.copy() for key, item in point.items()}
            minus = {key: item.copy() for key, item in point.items()}
            plus[variable.name][index] += 1e-5
            minus[variable.name][index] -= 1e-5
            component_gradient.append((density(plus) - density(minus)) / 2e-5)
        gradient.extend(component_gradient)
        gradient_by_name[variable.name] = component_gradient
    actual_gradient = model.compile_dlogp()(point)
    np.testing.assert_allclose(actual_gradient, gradient, rtol=2e-7, atol=2e-7)
    print(
        f"linked={linked} full={full}: "
        f"density error={abs(model.compile_logp()(point) - density(point)):.3g}, "
        f"gradient max error={np.max(np.abs(actual_gradient - gradient)):.3g}"
    )


@pytest.mark.parametrize("cached", [False, True])
def test_single_inference_forwards_calibration_and_serializes_it(cached):
    import json
    import pymc as pm
    from openghg_inversions.models import get_coord_registry, restore_inferencedata_coords
    from openghg_inversions.rhime.co2.inference import build_co2_inference, annotate_co2_inference
    from test_co2_affine_output import _prepared as single_prepared

    prepared, _ = single_prepared()
    inputs = prepared.inv_inputs
    baseline = CorrelatedPrior(
        inputs.alpha_prior_mean, inputs.alpha_prior_covariance,
        covariance_dim=str(inputs.alpha_prior_covariance.dims[1]),
    )
    dim = baseline.state_dim
    weights = xr.DataArray(
        [[-2., 0.], [0., 3.]], dims=("process", dim),
        coords={"process": ["GPP", "TER"], dim: baseline.mean[dim]}, attrs={"units": "kg/year"},
    )
    calibration = calibrate_biosphere_prior(
        baseline, biosphere=xr.ones_like(baseline.mean, dtype=bool), weights=weights,
        reference=(weights * baseline.mean).sum(dim).assign_attrs(units="kg/year"),
        residual_sd_multiplier=.8,
    )
    choices = {"biosphere_prior": calibration, "no_model_error": True}
    if cached:
        choices.update(
            likelihood_builder=add_fixed_ou_gaussian_likelihood,
            likelihood_kwargs={"tau_hours": 5., "site_amplitude_prior": {"pdf": "halfnormal", "sigma": .75}},
        )
        choices.pop("no_model_error")
    built, cached_graph = build_co2_inference(prepared, choices)
    with built.model:
        trace = pm.sample_prior_predictive(draws=2, random_seed=37)
    trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
    trace = annotate_co2_inference(trace, built, prepared, choices, cached_graph)
    xr.testing.assert_equal(trace["biosphere_prior"].to_dataset(), calibration.to_dataset())
    assert json.loads(trace.attrs["rhime_model_metadata"])["biosphere_prior_scope"] == BIOSPHERE_PRIOR_SCOPE
    assert built.output_contract.metadata["prior"] == "correlated arithmetic-moment lognormal"
    assert not hasattr(built.model, "biosphere_prior")
    assert not np.array_equal(calibration.data.covariance, baseline.arithmetic_covariance)
