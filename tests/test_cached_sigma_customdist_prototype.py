"""Scientific and execution oracles for the temporary GH769 graph prototype."""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.stats import multivariate_normal
import xarray as xr

from openghg_inversions.models.fixed_ou import FixedOuLowRank
from openghg_inversions.rhime.cached_sigma import make_cached_sigma_compound_step
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from scripts.prototype_cached_sigma_customdist import build_fixture, replace_cached_potential
from test_co2_o2_fixed_ou import _dense, _kwargs, _prepared


LINKED_TAU = {"co2:A": 6.0, "o2:A": 8.0, "o2:B": 2.0}


def _build(linked=False):
    if not linked:
        return build_fixture(observed=True)
    cached = build_co2_o2_cached_sigma_model(
        **_kwargs(_prepared(), interleaved=True),
        tau_hours=LINKED_TAU,
        site_amplitude_prior_scale=0.8,
        initial_site_amplitudes=0.4,
    )
    replace_cached_potential(cached)
    return cached


def _covariance(prepared, amplitude):
    """Assemble the full Gaussian covariance independently of its solver."""
    covariance = np.diag(prepared.diagonal_variance) + prepared.factor @ prepared.factor.T
    for block in prepared.site_blocks:
        rows = block.observation_indices
        correlation = block.correlation_cholesky @ block.correlation_cholesky.T
        covariance[np.ix_(rows, rows)] += amplitude[block.site] ** 2 * correlation
    return covariance


def _covariance_oracle(cached, linked):
    if not linked:
        return lambda amplitude: _covariance(cached.target.prepared, amplitude)
    kwargs = _kwargs(_prepared(), interleaved=True)
    species = kwargs["observations"].species.values
    covariance = _dense(kwargs, np.zeros(cached.target.n_group), LINKED_TAU)
    assert np.any(covariance[np.ix_(species == "co2", species == "o2")] != 0.0)
    return lambda amplitude: _dense(kwargs, amplitude, LINKED_TAU)


@pytest.mark.parametrize("linked", [False, True])
def test_observed_distribution_has_normalized_density_and_physical_gradients(linked):
    cached = _build(linked)
    prepared = cached.target.prepared
    dense_covariance = _covariance_oracle(cached, linked)
    mean = pt.dvector("arbitrary_mean")
    amplitude = pt.dvector("arbitrary_amplitude")
    value = pt.dvector("arbitrary_observation")
    distribution = pm.CustomDist.dist(
        mean, amplitude, logp=prepared.logp, random=prepared.random,
        signature="(n),(s)->(n)",
    )
    logp = pm.logp(distribution, value)
    evaluate = pytensor.function(
        [value, mean, amplitude], [logp, pt.grad(logp, mean), pt.grad(logp, amplitude)]
    )
    rng = np.random.default_rng(769)
    for _ in range(3):
        mean_value = rng.normal(size=prepared.n_observation)
        amplitude_value = rng.uniform(0.05, 1.5, prepared.n_site)
        observations = rng.normal(size=prepared.n_observation)
        covariance = dense_covariance(amplitude_value)
        actual, mean_gradient, amplitude_gradient = evaluate(
            observations, mean_value, amplitude_value
        )
        expected = multivariate_normal.logpdf(observations, mean=mean_value, cov=covariance)
        np.testing.assert_allclose(actual, expected, atol=1e-9)
        np.testing.assert_allclose(
            mean_gradient, np.linalg.solve(covariance, observations - mean_value), atol=1e-9
        )
        for index in range(prepared.n_site):
            plus, minus = amplitude_value.copy(), amplitude_value.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            derivative = (
                multivariate_normal.logpdf(observations, mean=mean_value, cov=dense_covariance(plus))
                - multivariate_normal.logpdf(observations, mean=mean_value, cov=dense_covariance(minus))
            ) / 2e-5
            np.testing.assert_allclose(amplitude_gradient[index], derivative, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("linked", [False, True])
def test_real_graph_has_one_likelihood_and_ignores_stale_sampler_cache(linked):
    cached = _build(linked)
    model = cached.model
    dense_covariance = _covariance_oracle(cached, linked)
    assert not model.potentials
    assert [rv.name for rv in model.observed_RVs] == ["y"]
    assert "cached_fixed_ou_likelihood" not in model.named_vars
    likelihood = model.compile_logp(vars=model.observed_RVs)
    priors = model.compile_logp(vars=model.free_RVs)
    full_logp = model.compile_logp()
    gradient_variables = [*cached.states, cached.amplitude]
    gradient = model.compile_dlogp(vars=gradient_variables)
    mean_expression = model.replace_rvs_by_values([cached.modelled_mean])[0]
    mean = model.compile_fn(mean_expression, inputs=model.value_vars, on_unused_input="ignore")
    alternate_value = pt.dvector("alternate_observation")
    alternate_logp = model.replace_rvs_by_values([pm.logp(model["y"], alternate_value)])[0]
    alternate_likelihood = model.compile_fn(
        alternate_logp, inputs=[alternate_value, *model.value_vars],
        point_fn=False, on_unused_input="ignore",
    )
    point = model.initial_point()
    amplitude_name = model.rvs_to_values[cached.amplitude].name

    def oracle(candidate):
        covariance = dense_covariance(np.exp(candidate[amplitude_name]))
        return priors(candidate) + multivariate_normal.logpdf(
            cached.target.observations, mean=mean(candidate), cov=covariance
        )

    for shift in (-0.3, 0.4):
        for rv in cached.states:
            name = model.rvs_to_values[rv].name
            point[name] = np.full_like(point[name], shift)
        point[amplitude_name] = np.log(np.linspace(0.2 + shift / 10, 0.8, cached.target.n_group))
        expected = oracle(point)
        np.testing.assert_allclose(full_logp(point), expected, atol=1e-9)
        alternate = cached.target.observations + np.linspace(-0.4, 0.7, cached.target.n_obs)
        covariance = dense_covariance(np.exp(point[amplitude_name]))
        np.testing.assert_allclose(
            alternate_likelihood(alternate, *[point[var.name] for var in model.value_vars]),
            multivariate_normal.logpdf(alternate, mean=mean(point), cov=covariance),
            atol=1e-9,
        )
        before = likelihood(point)
        cached.shared_cache.update(cached.target.refresh(np.full(cached.target.n_group, 2.5)))
        np.testing.assert_allclose(likelihood(point), before, atol=1e-12)
        numerical = []
        for rv in gradient_variables:
            name = model.rvs_to_values[rv].name
            for index in range(point[name].size):
                plus = {key: value.copy() for key, value in point.items()}
                minus = {key: value.copy() for key, value in point.items()}
                plus[name].flat[index] += 1e-5
                minus[name].flat[index] -= 1e-5
                numerical.append((oracle(plus) - oracle(minus)) / 2e-5)
        np.testing.assert_allclose(gradient(point), numerical, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("observed", [False, True])
def test_state_logp_and_gradient_reveal_covariance_evaluation_cost(monkeypatch, observed):
    cached = build_fixture(observed=observed)
    original = FixedOuLowRank.evaluate
    calls = []

    def counted(prepared, residual, amplitude):
        calls.append(np.asarray(amplitude).copy())
        return original(prepared, residual, amplitude)

    monkeypatch.setattr(FixedOuLowRank, "evaluate", counted)
    logp = cached.model.compile_logp()
    gradient = cached.model.compile_dlogp(vars=list(cached.states))
    point = cached.model.initial_point()
    calls.clear()
    for _ in range(3):
        logp(point)
        gradient(point)
    assert len(calls) == (6 if observed else 0)


def test_seeded_compound_sampler_posterior_matches_cached_graph():
    """Check paired short trajectories; this is not a convergence assessment."""
    posteriors = []
    for observed in (False, True):
        cached = build_fixture(observed=observed, n_observations=8, n_states=2, rank=2)
        with cached.model:
            step = make_cached_sigma_compound_step(
                model=cached.model,
                sigma=cached.amplitude,
                states=cached.states,
                modelled_mean=cached.modelled_mean,
                target=cached.target,
                shared_cache=cached.shared_cache,
                initial_cache=cached.initial_cache,
                prior_scale=cached.site_amplitude_prior_scale,
                rng=np.random.default_rng(769),
            )
            trace = pm.sample(
                draws=12, tune=12, chains=2, cores=1, step=step, random_seed=769,
                progressbar=False, compute_convergence_checks=False,
                idata_kwargs={"log_likelihood": False},
            )
        posteriors.append(trace.posterior)
    for variable in ("flux_scaling", "ou_site_amplitude", "modelled_concentration"):
        assert posteriors[0][variable].shape[:2] == (2, 12)
        np.testing.assert_allclose(
            posteriors[0][variable], posteriors[1][variable], rtol=1e-6, atol=1e-6,
        )


def _assert_joint_gaussian_residuals(observations, means, amplitudes, dense_covariance, tolerance):
    residuals = observations - means
    whitened = np.asarray([
        np.linalg.solve(np.linalg.cholesky(dense_covariance(amplitude)), residual)
        for residual, amplitude in zip(residuals, amplitudes, strict=True)
    ])
    np.testing.assert_allclose(whitened.mean(axis=0), 0.0, atol=tolerance)
    np.testing.assert_allclose(np.cov(whitened.T), np.eye(observations.shape[1]), atol=tolerance)


@pytest.mark.parametrize("linked", [False, True])
def test_seeded_prior_and_two_chain_heterogeneous_posterior_predictions(linked):
    cached = _build(linked)
    model = cached.model
    dense_covariance = _covariance_oracle(cached, linked)
    cached.shared_cache.update(cached.target.refresh(np.full(cached.target.n_group, 2.5)))
    with model:
        prior = pm.sample_prior_predictive(draws=1200, random_seed=769)
        repeated = pm.sample_prior_predictive(draws=1200, random_seed=769)
    np.testing.assert_array_equal(prior.prior_predictive.y, repeated.prior_predictive.y)
    _assert_joint_gaussian_residuals(
        prior.prior_predictive.y.values[0], prior.prior[cached.modelled_mean.name].values[0],
        prior.prior[cached.amplitude.name].values[0], dense_covariance, 0.18,
    )
    draws = 1200
    dataset = xr.Dataset(coords={"chain": [0, 1], "draw": np.arange(draws)})
    amplitudes = np.empty((2, draws, cached.target.n_group))
    for chain in range(2):
        for half in range(2):
            section = slice(half * draws // 2, (half + 1) * draws // 2)
            amplitudes[chain, section] = np.linspace(0.2, 0.7, cached.target.n_group) + 0.3 * chain + 0.1 * half
    for rv in cached.states:
        dim = model.named_vars_to_dims[rv.name][0]
        n_state = len(model.coords[dim])
        dataset = dataset.assign_coords({dim: list(model.coords[dim])})
        latent = np.zeros((2, draws, n_state))
        latent[1] += 0.7
        latent[:, draws // 2:] -= 0.4
        dataset[rv.name] = (("chain", "draw", dim), latent)
    amplitude_dim = model.named_vars_to_dims[cached.amplitude.name][0]
    dataset = dataset.assign_coords({amplitude_dim: list(model.coords[amplitude_dim])})
    dataset[cached.amplitude.name] = (("chain", "draw", amplitude_dim), amplitudes)
    cached.shared_cache.update(cached.target.refresh(np.full(cached.target.n_group, 3.5)))
    with model:
        posterior = pm.sample_posterior_predictive(
            dataset, var_names=["y", cached.modelled_mean.name], random_seed=770, progressbar=False
        ).posterior_predictive
        repeated = pm.sample_posterior_predictive(
            dataset, var_names=["y", cached.modelled_mean.name], random_seed=770, progressbar=False
        ).posterior_predictive
    np.testing.assert_array_equal(posterior.y, repeated.y)
    assert posterior.y.shape == (2, draws, cached.target.n_obs)
    assert not np.allclose(posterior[cached.modelled_mean.name][0], posterior[cached.modelled_mean.name][1])
    mean_expression = model.replace_rvs_by_values([cached.modelled_mean])[0]
    mean = model.compile_fn(mean_expression, inputs=model.value_vars, on_unused_input="ignore")
    point = model.initial_point()
    for chain in range(2):
        for half in range(2):
            section = slice(half * draws // 2, (half + 1) * draws // 2)
            for rv in cached.states:
                point[model.rvs_to_values[rv].name] = dataset[rv.name].values[chain, section.start]
            point[model.rvs_to_values[cached.amplitude].name] = np.log(amplitudes[chain, section.start])
            expected_mean = np.broadcast_to(mean(point), (draws // 2, cached.target.n_obs))
            np.testing.assert_allclose(posterior[cached.modelled_mean.name].values[chain, section], expected_mean)
            _assert_joint_gaussian_residuals(
                posterior.y.values[chain, section], posterior[cached.modelled_mean.name].values[chain, section],
                amplitudes[chain, section], dense_covariance, 0.25,
            )
