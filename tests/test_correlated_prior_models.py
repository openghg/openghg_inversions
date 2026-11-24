"""The correlated flux prior through the ordinary, cached and linked builders."""

import numpy as np
import pymc as pm
import pytest
import pytensor
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedPrior
from openghg_inversions.models import StateActivity
from openghg_inversions.models.components import add_correlated_state_with_activity
from openghg_inversions.models.coords import registered_model
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.models.state_activity import resolve_state_activity
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import (
    build_co2_model,
    build_co2_cached_sigma_model,
    build_co2_o2_model,
    build_co2_o2_cached_sigma_model,
)
from boundary_test_utils import bound_model_arguments
from test_co2_o2_fixed_ou import _prepared, _kwargs


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64", cast_policy="numpy+floatX"):
        yield


def correlated_prior():
    """Return a five-state prior with nonzero cross-covariances."""
    labels = ["g-a", "t-a", "g-b", "fossil", "outer"]
    mean = xr.DataArray([1.1, 1, 0.9, 1.2, 1], dims="state", coords={"state": labels})
    latent = np.array(
        [
            [0.15, 0.02, 0.01, 0.015, 0.005],
            [0.02, 0.12, 0.01, 0.005, 0.01],
            [0.01, 0.01, 0.1, 0.008, 0.004],
            [0.015, 0.005, 0.008, 0.08, 0.01],
            [0.005, 0.01, 0.004, 0.01, 0.07],
        ]
    )
    return CorrelatedPrior(mean, np.outer(mean, mean) * np.expm1(latent))


def state_and_mean(model, point):
    return model.compile_fn(
        model.replace_rvs_by_values([model.flux_scaling, model.modelled_concentration]),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )(point)


@pytest.mark.parametrize("full", [True, False])
def test_full_and_enhancement_ordinary_cached_prediction_and_density(full):
    prior = correlated_prior()
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
        coords={"nmeasure": observation.nmeasure, **prior.mean.coords},
    )
    kwargs = dict(
        flux_sensitivity=h,
        retained_prior=prior,
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
    cached = build_co2_cached_sigma_model(**kwargs, **likelihood)
    ordinary = build_co2_model(
        **kwargs, likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=likelihood,
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
    for model, values in ((ordinary, point), (cached.model, cached_point)):
        assert_normalized_density_and_gradient(model, values, prior, kwargs, full=full)
    np.testing.assert_allclose(x, physical_state(prior, point["flux_scaling_latent"]), rtol=1e-14)


def test_linked_species_share_one_state_and_density():
    prepared = _prepared()
    kwargs = _kwargs(prepared)
    prior = CorrelatedPrior.from_lognormal(prepared.retained_prior)
    likelihood = dict(tau_hours=5, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    ordinary = build_co2_o2_model(**{**kwargs, "retained_prior": prior}, **likelihood)
    cached = build_co2_o2_cached_sigma_model(**{**kwargs, "retained_prior": prior}, **likelihood)
    point = ordinary.initial_point()
    point["flux_scaling_latent"] = np.array([0.1, -0.3, 0.2, -0.1, 0.4])
    x, mean = state_and_mean(ordinary, point)
    xc, mc = state_and_mean(cached.model, point)
    np.testing.assert_allclose(x, xc)
    np.testing.assert_allclose(mean, mc)
    np.testing.assert_allclose(ordinary.compile_logp()(point), cached.model.compile_logp()(point), rtol=1e-12)
    assert len([rv for rv in cached.model.free_RVs if rv.name.startswith("flux_scaling")]) == 1
    for model in (ordinary, cached.model):
        assert_normalized_density_and_gradient(model, point, prior, kwargs, linked=True)
    np.testing.assert_allclose(x, physical_state(prior, point["flux_scaling_latent"]), rtol=1e-14)


@pytest.mark.parametrize("state_dim", ["state", "flux_state"])
def test_prior_rejects_reordered_or_renamed_state_axis(state_dim):
    # Latent factors are applied by position, so a reordered axis would
    # silently move coefficients between columns.
    prior = correlated_prior()
    observation = xr.DataArray(
        [1.2, 0.4, 0.8],
        dims="nmeasure",
        coords={"nmeasure": np.arange(3), "site": ("nmeasure", ["A", "A", "B"])},
        attrs={"units": "ppm"},
    )
    labels = prior.mean.state.values[::-1]
    h = xr.DataArray(
        np.arange(15.0).reshape(3, 5) + 1,
        dims=("nmeasure", state_dim),
        coords={"nmeasure": observation.nmeasure, state_dim: labels},
    )
    match = "same state dimension" if state_dim != "state" else "same order"
    with pytest.raises(ValueError, match=match):
        build_co2_model(
            h,
            retained_prior=prior,
            observations=observation,
            observation_error=xr.full_like(observation, 0.2),
            fixed_prior_contribution=xr.zeros_like(observation).rename("fixed_prior_contribution"),
            aggregation_error=resolve_aggregation_error(observation.to_dataset(name="mf"), "none"),
        )


def test_prior_draws_match_joint_moments():
    prior = correlated_prior()
    activity = resolve_state_activity(xr.zeros_like(prior.mean, dtype=bool), None)
    with registered_model() as model:
        add_correlated_state_with_activity(activity, prior, var_name="flux_scaling")
    # Vectorized compiled draws are secondary to the analytic moments.
    with model:
        draws = pm.draw(model.flux_scaling, draws=30000, random_seed=719)
    np.testing.assert_allclose(draws.mean(0), prior.mean, atol=0.012)
    np.testing.assert_allclose(np.cov(draws, rowvar=False), prior.arithmetic_covariance, rtol=0.08, atol=0.005)
    assert np.all(draws > 0)


def test_correlated_prior_rejects_unknown_or_misaligned_marginals():
    prior = correlated_prior()
    marginal = prior.marginal.copy(data=["lognormal", "lognormal", "beta", "lognormal", "lognormal"])
    with pytest.raises(ValueError, match="Unsupported state marginal"):
        CorrelatedPrior(prior.mean, prior.arithmetic_covariance, marginal=marginal)
    with pytest.raises(ValueError, match="same order"):
        CorrelatedPrior(prior.mean, prior.arithmetic_covariance, marginal=prior.marginal[::-1])


def test_all_fixed_state_has_no_sampled_latent():
    prior = correlated_prior()
    activity = resolve_state_activity(
        xr.zeros_like(prior.mean, dtype=bool), StateActivity(active=False, fixed_value=1.7)
    )
    with registered_model() as model:
        state = add_correlated_state_with_activity(activity, prior, var_name="flux_scaling")
    assert state.latent is None
    assert not model.free_RVs
    np.testing.assert_array_equal(model.compile_fn(state.state)({}), np.full(5, 1.7))


def test_partial_activity_samples_the_exact_active_lognormal_marginal():
    active = np.array([True, False, True, False, True])
    prior = correlated_prior()
    activity = resolve_state_activity(
        xr.zeros_like(prior.mean, dtype=bool), StateActivity(active=active, fixed_value=1.7)
    )
    z = np.array([0.3, -0.2, 0.4])
    with registered_model() as model:
        state = add_correlated_state_with_activity(activity, prior, var_name="flux_scaling")
    m = prior.mean.to_numpy()[active]
    b = prior.arithmetic_covariance.to_numpy()[np.ix_(active, active)]
    latent_covariance = np.log1p(b / np.outer(m, m))
    expected = np.full(5, 1.7)
    expected[active] = np.exp(np.log(m) - np.diag(latent_covariance) / 2 + np.linalg.cholesky(latent_covariance) @ z)
    actual = model.compile_fn(
        model.replace_rvs_by_values([state.state]), inputs=model.value_vars
    )({"flux_scaling_active_latent": z})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    assert [rv.name for rv in model.free_RVs] == ["flux_scaling_active_latent"]


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
