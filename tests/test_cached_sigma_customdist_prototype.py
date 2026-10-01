"""Scientific and execution oracles for the temporary GH769 graph prototype."""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from pymc.blocking import DictToArrayBijection
from scipy.stats import multivariate_normal
import xarray as xr

from openghg_inversions.models.fixed_ou import FixedOuLowRank
from openghg_inversions.rhime.cached_sigma import make_cached_sigma_compound_step
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from scripts.prototype_cached_sigma_customdist import (
    build_fixture,
    make_observed_cached_compound_step,
    replace_cached_potential,
)
from test_co2_o2_fixed_ou import _dense, _kwargs, _prepared


LINKED_TAU = {"co2:A": 6.0, "o2:A": 8.0, "o2:B": 2.0}


def _build(linked=False, *, observed=True):
    if not linked:
        return build_fixture(observed=observed)
    cached = build_co2_o2_cached_sigma_model(
        **_kwargs(_prepared(), interleaved=True),
        tau_hours=LINKED_TAU,
        site_amplitude_prior_scale=0.8,
        initial_site_amplitudes=0.4,
    )
    if observed:
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
    for mode in ("cached", "observed", "observed-cached"):
        cached = build_fixture(observed=mode == "observed", n_observations=8, n_states=2, rank=2)
        initial_point = cached.model.initial_point()
        with cached.model:
            if mode == "observed-cached":
                step = make_observed_cached_compound_step(cached, rng=np.random.default_rng(769))
            else:
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
                draws=12, tune=12, chains=2, cores=1, step=step, random_seed=769, initvals=initial_point,
                progressbar=False, compute_convergence_checks=False,
                idata_kwargs={"log_likelihood": False},
            )
        posteriors.append(trace.posterior)
    for variable in ("flux_scaling", "ou_site_amplitude", "modelled_concentration"):
        assert posteriors[0][variable].shape[:2] == (2, 12)
        for posterior in posteriors[1:]:
            np.testing.assert_allclose(posteriors[0][variable], posterior[variable], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("linked", [False, True])
def test_compiled_cached_state_step_matches_exact_public_model_without_covariance_work(monkeypatch, linked):
    cached = _build(linked, observed=False)
    step = make_observed_cached_compound_step(cached, rng=np.random.default_rng(769))
    state_step = step.methods[1]
    public_logp = cached.model.compile_logp()
    public_gradient = cached.model.compile_dlogp(vars=list(cached.states))
    point = cached.model.initial_point()
    amplitude_name = cached.model.rvs_to_values[cached.amplitude].name
    for shift, amplitudes in (
        (-0.3, np.linspace(0.2, 0.7, cached.target.n_group)),
        (0.4, np.linspace(0.6, 1.2, cached.target.n_group)),
    ):
        for name in state_step.var_names:
            point[name] = np.full_like(point[name], shift)
        point[amplitude_name] = np.log(amplitudes)
        cached.shared_cache.update(cached.target.refresh(amplitudes))
        state_step._logp_dlogp_func.set_extra_values(point)
        state_point = DictToArrayBijection.map({name: point[name] for name in state_step.var_names})
        cached_logp, cached_gradient = state_step._logp_dlogp_func(state_point)
        np.testing.assert_allclose(cached_logp, public_logp(point), atol=1e-9)
        np.testing.assert_allclose(cached_gradient, public_gradient(point), atol=1e-9)

    def forbidden(*args, **kwargs):
        raise AssertionError("A cached state transition must not evaluate or factor covariance.")

    monkeypatch.setattr(FixedOuLowRank, "evaluate", forbidden)
    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    updated, stats = state_step.step(point)
    assert np.isfinite(updated[state_step.var_names[0]]).all()
    assert stats[0]["tree_size"] > 0


def _sample_observed_cached_spawn(*, observed=True):
    cached = build_fixture(observed=False)
    initial_points = []
    for amplitude, latent in ((0.3, -0.2), (0.6, 0.3)):
        point = cached.model.initial_point()
        point[cached.model.rvs_to_values[cached.amplitude].name] = np.full(cached.target.n_group, np.log(amplitude))
        for rv in cached.states:
            name = cached.model.rvs_to_values[rv].name
            point[name] = np.full_like(point[name], latent)
        initial_points.append(point)
    with cached.model:
        if observed:
            step = make_observed_cached_compound_step(cached, rng=np.random.default_rng(769))
        else:
            step = make_cached_sigma_compound_step(
                model=cached.model, sigma=cached.amplitude, states=cached.states,
                modelled_mean=cached.modelled_mean, target=cached.target,
                shared_cache=cached.shared_cache, initial_cache=cached.initial_cache,
                prior_scale=cached.site_amplitude_prior_scale, rng=np.random.default_rng(769),
            )
        trace = pm.sample(
            draws=4, tune=4, chains=2, cores=2, mp_ctx="spawn", step=step,
            random_seed=769, progressbar=False, compute_convergence_checks=False,
            initvals=initial_points,
            idata_kwargs={"log_likelihood": False},
        )
    return cached, trace


def test_spawned_observed_cached_chains_preserve_shared_updates_and_exact_public_density():
    cached, trace = _sample_observed_cached_spawn()
    _, repeated = _sample_observed_cached_spawn()
    _, baseline = _sample_observed_cached_spawn(observed=False)
    for variable in ("flux_scaling", "ou_site_amplitude", "modelled_concentration"):
        np.testing.assert_array_equal(trace.posterior[variable], repeated.posterior[variable])
        np.testing.assert_allclose(trace.posterior[variable], baseline.posterior[variable], rtol=1e-6, atol=1e-6)
    assert not np.array_equal(trace.posterior.ou_site_amplitude[0], trace.posterior.ou_site_amplitude[1])
    assert "cache_refreshes" in trace.sample_stats
    assert int(trace.sample_stats.cache_refreshes.sum()) > 0
    cached.shared_cache.update(cached.target.refresh(np.full(cached.target.n_group, 3.5)))
    likelihood = cached.model.compile_logp(vars=cached.model.observed_RVs)
    point = cached.model.initial_point()
    expected_log_likelihood = np.empty((2, 4))
    for chain in range(2):
        for draw in range(4):
            for rv in cached.states:
                point[cached.model.rvs_to_values[rv].name] = trace.posterior[rv.name].values[chain, draw]
            amplitude = trace.posterior[cached.amplitude.name].values[chain, draw]
            point[cached.model.rvs_to_values[cached.amplitude].name] = np.log(amplitude)
            expected = multivariate_normal.logpdf(
                cached.target.observations,
                mean=trace.posterior[cached.modelled_mean.name].values[chain, draw],
                cov=_covariance(cached.target.prepared, amplitude),
            )
            expected_log_likelihood[chain, draw] = expected
            np.testing.assert_allclose(likelihood(point), expected, atol=1e-9)
    pm.compute_log_likelihood(trace, model=cached.model, progressbar=False)
    assert trace.log_likelihood.y.shape == (2, 4)
    np.testing.assert_allclose(trace.log_likelihood.y, expected_log_likelihood, atol=1e-9)


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
