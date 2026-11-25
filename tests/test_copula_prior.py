"""Per-state marginals on the correlated flux prior: a Gaussian copula matched to arithmetic moments."""

import numpy as np
import pytest
import pytensor
from scipy import stats
import xarray as xr

from openghg_inversions.correlated_state import STATE_MARGINALS, CorrelatedPrior
from openghg_inversions.models import StateActivity
from openghg_inversions.models.components import add_correlated_state_with_activity, prepare_active_correlated_prior
from openghg_inversions.models.coords import registered_model
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.models.state_activity import resolve_state_activity
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import (
    build_co2_cached_sigma_model,
    build_co2_model,
    build_co2_o2_cached_sigma_model,
    build_co2_o2_model,
)
from boundary_test_utils import bound_model_arguments
from test_correlated_prior_models import assert_normalized_density_and_gradient, correlated_prior, state_and_mean
from test_co2_o2_fixed_ou import _kwargs, _prepared

SHAPED = ("normal", "weibull", "gamma")
LABELS = ["g-a", "t-a", "g-b", "fossil", "outer"]
_STARTUP_FLOATX = pytensor.config.floatX


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64", cast_policy="numpy+floatX"):
        yield


@pytest.fixture
def jax_backend():
    """Preserve the session's JAX precision across the copula checks."""
    jax = pytest.importorskip("jax")
    # A first import sets the JAX process default from PyTensor floatX.
    with pytensor.config.change_flags(floatX=_STARTUP_FLOATX):
        pytest.importorskip("pytensor.link.jax.dispatch")
    x64 = jax.config.x64_enabled
    try:
        yield jax
    finally:
        jax.config.update("jax_enable_x64", x64)


def biosphere_marginal(prior, family):
    """``family`` on the three GPP/TER states of ``correlated_prior()``, lognormal on fossil and outer."""
    return prior.mean.copy(data=np.array([family] * 3 + ["lognormal"] * 2, dtype=object))


def wide_prior(families, *, cv=(0.9, 0.7, 1.3, 0.5, 0.6), correlation=None):
    """Five states with large coefficients of variation and mixed-sign correlations."""
    mean = xr.DataArray([1.1, 1.0, 0.9, 1.2, 1.0], dims="state", coords={"state": LABELS})
    sd = np.asarray(cv) * mean.to_numpy()
    if correlation is None:
        correlation = np.array(
            [
                [1.0, 0.5, 0.2, -0.1, 0.0],
                [0.5, 1.0, 0.1, 0.0, -0.2],
                [0.2, 0.1, 1.0, 0.15, 0.05],
                [-0.1, 0.0, 0.15, 1.0, 0.3],
                [0.0, -0.2, 0.05, 0.3, 1.0],
            ]
        )
    marginal = mean.copy(data=np.array(families, dtype=object))
    return CorrelatedPrior(mean, correlation * np.outer(sd, sd), marginal=marginal)


def quadrature_moments(prior, nodes=160):
    """Means and covariances of every state and pair by 1-D and 2-D Gauss-Hermite quadrature."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / np.sqrt(2 * np.pi)
    r = prior.latent_correlation.to_numpy()
    size = r.shape[0]
    mean = np.array([(w * prior.transform_latent(np.outer(x, np.eye(size)[i]))[:, i]).sum() for i in range(size)])
    covariance = np.empty((size, size))
    first, second = np.meshgrid(x, x, indexing="ij")
    weights = np.outer(w, w)
    for i in range(size):
        for j in range(size):
            u = np.zeros(first.shape + (size,))
            u[..., i] = first
            u[..., j] = r[i, j] * first + np.sqrt(max(1 - r[i, j] ** 2, 0.0)) * second if i != j else first
            values = prior.transform_latent(u) - mean
            covariance[i, j] = (weights * values[..., i] * values[..., j]).sum()
    return mean, covariance


@pytest.mark.parametrize("family", SHAPED)
def test_copula_reproduces_the_arithmetic_mean_and_covariance_exactly(family):
    prior = wide_prior([family, family, family, "lognormal", "lognormal"])
    assert prior.lognormal is None
    mean, covariance = quadrature_moments(prior)
    np.testing.assert_allclose(mean, prior.mean.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(covariance, prior.arithmetic_covariance.to_numpy(), rtol=1e-9, atol=1e-12)


def test_mixed_copula_draws_match_the_requested_moments():
    prior = wide_prior(["weibull", "gamma", "normal", "lognormal", "weibull"], cv=(0.6, 0.5, 0.4, 0.5, 0.7))
    draws = prior.sample(400_000, rng=20261120)
    sd = np.sqrt(np.diag(prior.arithmetic_covariance.to_numpy()))
    standard_error = draws.std(axis=0) / np.sqrt(draws.shape[0])
    assert np.all(np.abs(draws.mean(axis=0) - prior.mean.to_numpy()) < 4 * standard_error)
    target = prior.arithmetic_covariance.to_numpy() / np.outer(sd, sd)
    np.testing.assert_allclose(np.corrcoef(draws.T), target, atol=0.01)
    np.testing.assert_allclose(draws.std(axis=0) / sd, 1, atol=0.01)
    assert (draws[:, [0, 1, 3, 4]] > 0).all()


def test_copula_shapes_differ_only_in_skew():
    """Same mean and SD, decreasing right skew: lognormal, gamma, Weibull, normal."""
    medians = {}
    for family in STATE_MARGINALS:
        draws = wide_prior([family] * 5, cv=(0.8,) * 5, correlation=np.eye(5)).sample(200_000, rng=1)[:, 0]
        medians[family] = np.median(draws)
    assert medians["lognormal"] < medians["gamma"] < medians["weibull"] < medians["normal"]


def test_all_lognormal_marginals_build_the_lognormal_prior():
    baseline = correlated_prior()
    explicit = CorrelatedPrior(
        baseline.mean, baseline.arithmetic_covariance, marginal=biosphere_marginal(baseline, "lognormal")
    )
    assert explicit.lognormal is not None
    np.testing.assert_array_equal(explicit.lognormal.latent_cholesky, baseline.lognormal.latent_cholesky)
    latent_sd = np.sqrt(np.diag(baseline.lognormal.latent_covariance.to_numpy()))
    z = np.array([0.3, -0.2, 0.4, 0.1, -0.5])
    np.testing.assert_allclose(
        baseline.transform_latent(baseline.latent_correlation_cholesky.to_numpy() @ z),
        np.exp(baseline.lognormal.latent_mean.to_numpy() + baseline.lognormal.latent_cholesky.to_numpy() @ z),
        rtol=1e-13,
    )
    np.testing.assert_allclose(baseline.marginal_shape, latent_sd)


@pytest.mark.parametrize("family", SHAPED)
def test_model_state_is_the_copula_map_with_partial_activity(family):
    baseline = correlated_prior()
    prior = CorrelatedPrior(
        baseline.mean, baseline.arithmetic_covariance, marginal=biosphere_marginal(baseline, family)
    )
    active = np.array([True, False, True, True, False])
    activity = resolve_state_activity(
        xr.zeros_like(prior.mean, dtype=bool), StateActivity(active=active, fixed_value=1.7)
    )
    active_prior = prepare_active_correlated_prior(activity, prior, var_name="flux_scaling")
    assert list(active_prior.marginal.values) == list(prior.marginal.values[active])
    # Pairwise matching: the active latent correlation is the full one's active block.
    np.testing.assert_allclose(
        active_prior.latent_correlation.to_numpy(),
        prior.latent_correlation.to_numpy()[np.ix_(active, active)],
        rtol=1e-12,
        atol=1e-14,
    )
    z = np.array([0.3, -1.2, 2.1])
    with registered_model() as model:
        state = add_correlated_state_with_activity(activity, prior, var_name="flux_scaling")
    actual = model.compile_fn(model.replace_rvs_by_values([state.state]), inputs=model.value_vars)(
        {"flux_scaling_active_latent": z}
    )[0]
    expected = np.full(5, 1.7)
    expected[active] = active_prior.transform_latent(active_prior.latent_correlation_cholesky.to_numpy() @ z)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    assert [rv.name for rv in model.free_RVs] == ["flux_scaling_active_latent"]


@pytest.mark.parametrize("family", SHAPED)
def test_backend_map_is_finite_with_finite_gradient_in_the_tails(family):
    prior = wide_prior([family] * 5, cv=(1.5, 0.3, 2.0, 0.8, 1.0), correlation=np.eye(5))
    activity = resolve_state_activity(xr.zeros_like(prior.mean, dtype=bool), StateActivity(active=True))
    with registered_model() as model:
        add_correlated_state_with_activity(activity, prior, var_name="flux_scaling")
    for value in (-30.0, -8.0, -3.0, 0.0, 1e-9, 3.0, 8.0, 14.2, 30.0):
        point = {"flux_scaling_latent": np.full(5, value)}
        state = model.compile_fn(model.replace_rvs_by_values([model["flux_scaling"]]), inputs=model.value_vars)(point)[0]
        np.testing.assert_allclose(state, prior.transform_latent(np.full(5, value)), rtol=1e-10)
        gradient = model.compile_fn(
            pytensor.grad(model.replace_rvs_by_values([model["flux_scaling"]])[0].sum(), model.value_vars[0]),
            inputs=model.value_vars,
        )(point)
        assert np.isfinite(gradient).all()


@pytest.mark.parametrize("family", STATE_MARGINALS)
def test_copula_map_compiles_for_jax_samplers(family, jax_backend):
    jax_backend.config.update("jax_enable_x64", True)
    prior = wide_prior([family, "weibull", "normal", "lognormal", family], correlation=np.eye(5))
    activity = resolve_state_activity(xr.zeros_like(prior.mean, dtype=bool), StateActivity(active=True))
    with registered_model() as model:
        add_correlated_state_with_activity(activity, prior, var_name="flux_scaling")
    point = {"flux_scaling_latent": np.array([-9.0, 2.5, 0.0, 1.0, 20.0])}
    outputs = model.replace_rvs_by_values([model["flux_scaling"]])
    jax_state = model.compile_fn(outputs, inputs=model.value_vars, mode="JAX")(point)[0]
    c_state = model.compile_fn(outputs, inputs=model.value_vars)(point)[0]
    # Gamma uses TensorFlow Probability's inverse incomplete gamma under JAX.
    np.testing.assert_allclose(jax_state, c_state, rtol=1e-12 if family != "gamma" else 1e-9)
    jax_gradient = model.compile_dlogp(jacobian=False, mode="JAX")(point) if hasattr(model, "compile_dlogp") else None
    c_gradient = model.compile_dlogp(jacobian=False)(point)
    np.testing.assert_allclose(jax_gradient, c_gradient, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("family", ["gamma", "weibull"])
@pytest.mark.parametrize("float_x", ["float32", "float64"])
def test_positive_marginals_preserve_values_and_gradients_in_jax(family, float_x, jax_backend):
    """Check the sampler's native JAX gradient and PyTensor's own gradient."""
    from pymc.sampling.jax import get_jaxified_graph

    jax_backend.config.update("jax_enable_x64", float_x == "float64")
    cv = (0.05, 0.1, 0.14, 0.3, 2.0) if family == "gamma" else (0.0013, 0.05, 0.14, 0.3, 2.0)
    prior = wide_prior([family] * 5, cv=cv, correlation=np.eye(5))
    activity = resolve_state_activity(xr.zeros_like(prior.mean, dtype=bool), StateActivity(active=True))
    distribution = stats.gamma if family == "gamma" else stats.weibull_min
    shape, scale = prior.marginal_shape.to_numpy(), prior.marginal_scale.to_numpy()
    with pytensor.config.change_flags(floatX=float_x):
        with registered_model() as model:
            add_correlated_state_with_activity(activity, prior, var_name="flux_scaling")
        state = model.replace_rvs_by_values([model["flux_scaling"]])[0]
        assert state.dtype == float_x
        symbolic_grad = model.compile_fn(
            pytensor.grad(state.sum(), model.value_vars[0]), inputs=model.value_vars,
        )
        jax_state = get_jaxified_graph(inputs=model.value_vars, outputs=[state])
        jax_grad = jax_backend.jit(jax_backend.grad(lambda latent: jax_state(latent)[0].sum()))
        tail = (5.5, 7.9) if family == "gamma" else (-30.0, 13.4, 20.0)
        for value in (-12.0, -6.0, -1.0, 0.0, 2.0, *tail):
            latent = np.full(5, value, dtype=float_x)
            expected = prior.transform_latent(latent.astype(np.float64))
            slope = np.exp(stats.norm.logpdf(float(value)) - distribution.logpdf(expected, shape, scale=scale))
            actual_state = jax_state(latent)[0]
            actual_gradient = jax_grad(latent)
            assert actual_state.dtype == actual_gradient.dtype == np.dtype(float_x)
            tolerance = 1e-3 if float_x == "float32" else 3e-8
            np.testing.assert_allclose(actual_state, expected, rtol=tolerance, atol=1e-35)
            np.testing.assert_allclose(actual_gradient, slope, rtol=tolerance, atol=1e-35)
            np.testing.assert_allclose(symbolic_grad({"flux_scaling_latent": latent}), slope, rtol=tolerance, atol=1e-35)


def single_species_kwargs(prior):
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
    return bound_model_arguments(
        dict(
            flux_sensitivity=h,
            retained_prior=prior,
            observations=observation,
            observation_error=xr.full_like(observation, 0.2),
            fixed_prior_contribution=xr.zeros_like(observation).rename("fixed_prior_contribution"),
            aggregation_error=resolve_aggregation_error(observation.to_dataset(name="mf"), "none"),
        )
    )


@pytest.mark.parametrize("family", SHAPED)
def test_ordinary_and_cached_co2_builders_share_density_and_gradient(family):
    baseline = correlated_prior()
    prior = CorrelatedPrior(
        baseline.mean, baseline.arithmetic_covariance, marginal=biosphere_marginal(baseline, family)
    )
    kwargs = single_species_kwargs(prior)
    likelihood = dict(tau_hours=5, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    ordinary = build_co2_model(
        **kwargs, likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=likelihood
    )
    cached = build_co2_cached_sigma_model(**kwargs, **likelihood)
    point = ordinary.initial_point()
    point["flux_scaling_latent"] = np.array([0.1, -0.3, 1.5, -0.2, 0.4])
    point["ou_site_amplitude_log__"] = np.log([0.4, 0.6])
    cached_point = cached.model.initial_point()
    cached_point.update(point)
    x, mean = state_and_mean(ordinary, point)
    xc, mc = state_and_mean(cached.model, cached_point)
    np.testing.assert_allclose(x, xc, rtol=1e-14)
    np.testing.assert_allclose(mean, mc, rtol=1e-14)
    np.testing.assert_allclose(ordinary.compile_logp()(point), cached.model.compile_logp()(cached_point), rtol=1e-12)
    for model, values in ((ordinary, point), (cached.model, cached_point)):
        assert_normalized_density_and_gradient(model, values, prior, kwargs)


@pytest.mark.parametrize("family", SHAPED)
def test_ordinary_and_cached_linked_builders_share_density_and_gradient(family):
    prepared = _prepared()
    kwargs = _kwargs(prepared)
    lognormal = prepared.retained_prior
    marginal = lognormal.mean.copy(data=np.array([family, family, "lognormal", "lognormal", "lognormal"], dtype=object))
    prior = CorrelatedPrior(
        lognormal.mean, lognormal.arithmetic_covariance, marginal=marginal, covariance_dim=lognormal.covariance_dim
    )
    likelihood = dict(tau_hours=5, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    ordinary = build_co2_o2_model(**{**kwargs, "retained_prior": prior}, **likelihood)
    cached = build_co2_o2_cached_sigma_model(**{**kwargs, "retained_prior": prior}, **likelihood)
    point = ordinary.initial_point()
    point["flux_scaling_latent"] = np.array([0.1, -0.3, 0.2, -0.1, 1.4])
    x, mean = state_and_mean(ordinary, point)
    xc, mc = state_and_mean(cached.model, point)
    np.testing.assert_allclose(x, xc, rtol=1e-14)
    np.testing.assert_allclose(mean, mc, rtol=1e-14)
    np.testing.assert_allclose(ordinary.compile_logp()(point), cached.model.compile_logp()(point), rtol=1e-12)
    for model in (ordinary, cached.model):
        assert_normalized_density_and_gradient(model, point, prior, kwargs, linked=True)


def test_unreachable_pair_covariance_fails_clearly():
    with pytest.raises(ValueError, match=r"No latent correlation .* \('g-a', 't-a'\)"):
        wide_prior(
            ["gamma", "gamma", "lognormal", "lognormal", "lognormal"],
            cv=(2.0, 2.0, 0.5, 0.5, 0.5),
            correlation=np.diag([1.0] * 5) + np.diag([-0.8, 0, 0, 0], 1) + np.diag([-0.8, 0, 0, 0], -1),
        )


def test_non_positive_definite_latent_correlation_fails_clearly():
    correlation = np.full((5, 5), -0.24) + 1.24 * np.eye(5)
    assert np.linalg.eigvalsh(correlation).min() > 0
    with pytest.raises(ValueError, match="not positive definite"):
        wide_prior(["weibull"] * 5, cv=(1.5,) * 5, correlation=correlation)


def test_copula_states_need_positive_variance():
    mean = xr.DataArray([1.0, 1.0], dims="state", coords={"state": ["a", "b"]})
    with pytest.raises(ValueError, match="positive arithmetic variance"):
        CorrelatedPrior(mean, np.diag([0.0, 0.1]), marginal=mean.copy(data=np.array(["weibull", "normal"], dtype=object)))
