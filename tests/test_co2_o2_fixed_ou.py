"""Dense two-tracer oracle and matched linked fixed-OU sampler contracts."""

from dataclasses import replace

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.stats import multivariate_normal

from openghg_inversions.rhime.co2 import prepare_co2_o2_inputs
from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
from openghg_inversions.rhime.co2.co2_o2_fixed_ou import linked_fixed_ou_alignment
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_runner import run_rhime_co2_o2_cached_sigma_from_prepared_inputs
from openghg_inversions.rhime.co2.co2_o2_runner import run_rhime_co2_o2_from_prepared_inputs
from openghg_inversions.rhime.co2.co2_cached_sigma_runner import _sampler_for_cached_graph
from openghg_inversions.rhime.sampling import RhimeSampler
from openghg_inversions.serialization import save_inferencedata, load_inferencedata

from test_rhime_co2_o2 import _inputs, _independent_error
from test_cached_sigma_sampling import _nuts_stats


def _prepared():
    inputs = _inputs()
    inputs['o2_units'] = 'ppm'
    for channel, sites, times in [('co2', ['A', 'A'], [4, 0]), ('o2', ['A', 'B', 'A'], [4, 2, 0])]:
        obs = inputs[f'{channel}_observations']
        dim = obs.dims[0]
        inputs[f'{channel}_observations'] = obs.assign_coords(
            site=(dim, sites),
            time=(dim, np.datetime64('2021-01-01T00') + np.array(times).astype('timedelta64[h]')),
        )
    return prepare_co2_o2_inputs(**inputs)


def _kwargs(prepared, *, interleaved=False):
    observations = prepared.observations
    fixed = prepared.fixed_prior_contribution
    error = _independent_error(prepared)
    aggregation = prepared.aggregation_error
    if interleaved:
        order = [2, 0, 3, 1, 4]
        observations, fixed, error = [array.isel(observation=order) for array in (observations, fixed, error)]
        covariance = aggregation.covariance
        covariance = covariance.isel({covariance.dims[0]: order, covariance.dims[1]: order})
        aggregation = replace(aggregation, covariance=covariance, marginal_variance=aggregation.marginal_variance[order])
    return dict(
        observations=observations, fixed_prior_contribution=fixed,
        co2_sensitivity=prepared.co2_sensitivity, o2_sensitivity=prepared.o2_sensitivity,
        aggregation_error=aggregation, retained_prior=prepared.retained_prior,
        independent_error_sd=error,
    )


def _dense(kwargs, amplitudes, tau):
    obs = kwargs['observations']
    alignment = linked_fixed_ou_alignment(obs)
    result = kwargs['aggregation_error'].covariance.values.copy()
    result += np.diag(kwargs['independent_error_sd'].values ** 2)
    hours = obs.time.values.astype('datetime64[m]').astype(float) / 60
    for index, label in enumerate(alignment.site_labels.values):
        rows = np.flatnonzero(alignment.site_index.values == index)
        result[np.ix_(rows, rows)] += amplitudes[index] ** 2 * np.exp(-np.abs(hours[rows, None] - hours[None, rows]) / tau[label])
    return result


@pytest.mark.parametrize('interleaved', [False, True])
def test_linked_stock_cached_covariance_logp_and_gradients_match_dense(interleaved):
    kwargs = _kwargs(_prepared(), interleaved=interleaved)
    alignment = linked_fixed_ou_alignment(kwargs['observations'])
    tau = {'co2:A': 6., 'o2:A': 8., 'o2:B': 2.}
    sigma = np.array([.4, .6, .8])
    cached = build_co2_o2_cached_sigma_model(**kwargs, tau_hours=tau,
        site_amplitude_prior_scale=.8, initial_site_amplitudes=dict(zip(alignment.site_labels.values, sigma)))
    stock = build_co2_o2_model(**kwargs, tau_hours=tau,
        site_amplitude_prior={'pdf': 'halfnormal', 'sigma': .8})
    covariance = _dense(kwargs, sigma, tau)
    state = np.array([.9, 1.1, .8, 1.2, 1.05])
    mean = cached.target.fixed_contribution + cached.target.design @ state
    residual = kwargs['observations'].values - mean
    expected = multivariate_normal.logpdf(residual, cov=covariance)
    np.testing.assert_allclose(cached.target.prepared.solve(np.eye(5), sigma).solution, np.linalg.inv(covariance), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cached.target.refresh(sigma).log_likelihood(state), expected, rtol=1e-11)
    amplitude = pt.dvector('amplitude')
    residual_var = pt.dvector('residual')
    stock_target = cached.target.prepared
    logp = stock_target.logp(residual_var, pt.zeros(5), amplitude)
    actual, residual_gradient, amplitude_gradient = pytensor.function(
        [residual_var, amplitude], [logp, pt.grad(logp, residual_var), pt.grad(logp, amplitude)]
    )(residual, sigma)
    np.testing.assert_allclose(actual, expected, rtol=1e-11)
    np.testing.assert_allclose(residual_gradient, -np.linalg.solve(covariance, residual), rtol=1e-10)
    for index in range(len(sigma)):
        plus, minus = sigma.copy(), sigma.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        expected_gradient = (multivariate_normal.logpdf(residual, cov=_dense(kwargs, plus, tau)) - multivariate_normal.logpdf(residual, cov=_dense(kwargs, minus, tau))) / 2e-5
        np.testing.assert_allclose(amplitude_gradient[index], expected_gradient, rtol=1e-6)
    # Compare the actual observed PyMC likelihood to the cached target at the
    # same flux/amplitude point (including transforms and correlated state).
    point = stock.initial_point()
    point['ou_site_amplitude_log__'] = np.log(sigma)
    stock_mean = stock.compile_fn(stock.replace_rvs_by_values([stock['modelled_concentration']]), point_fn=True, on_unused_input='ignore')(point)[0]
    np.testing.assert_allclose(stock.compile_logp(vars=[stock['y']])(point), cached.target.log_likelihood_from_mean(stock_mean, sigma), rtol=1e-10)


def test_linked_cache_accept_reject_order_and_chain_locality(monkeypatch):
    kwargs = _kwargs(_prepared())
    first = build_co2_o2_cached_sigma_model(**kwargs, tau_hours=6., site_amplitude_prior_scale=.8)
    second = build_co2_o2_cached_sigma_model(**kwargs, tau_hours=6., site_amplitude_prior_scale=.8)
    sampler = _sampler_for_cached_graph(RhimeSampler(nuts_sampler='pymc'), cached_model=first,
        sigma_target_accept=.8, state_target_accept=.9)
    amplitude_step, state_step = sampler.sample_kwargs['step'].methods
    assert amplitude_step.target is first.target
    assert first.shared_cache.linear is not second.shared_cache.linear
    point = first.model.initial_point()
    proposed = point['ou_site_amplitude_log__'] + [.05, -.03, .1]
    monkeypatch.setattr(amplitude_step.nuts_step, 'step', lambda _: ({'ou_site_amplitude_log__': proposed.copy()}, _nuts_stats()))
    updated, stats = amplitude_step.step(point)
    assert stats[0]['cache_refreshes'] == 1
    np.testing.assert_allclose(first.shared_cache.linear.get_value(), first.target.refresh(np.exp(proposed)).linear)
    np.testing.assert_allclose(second.shared_cache.linear.get_value(), second.initial_cache.linear)
    monkeypatch.setattr(amplitude_step.nuts_step, 'step', lambda point: ({'ou_site_amplitude_log__': point['ou_site_amplitude_log__'].copy()}, _nuts_stats()))
    _, stats = amplitude_step.step(updated)
    assert stats[0]['cache_refreshes'] == 0
    monkeypatch.setattr(first.target, 'refresh', lambda _: pytest.fail('state step refreshed covariance'))
    state_step.step(updated)


@pytest.mark.parametrize('cached', [False, True])
def test_linked_ou_rejects_unsupported_backend_before_sampling(cached):
    prepared = _prepared()
    runner = run_rhime_co2_o2_cached_sigma_from_prepared_inputs if cached else run_rhime_co2_o2_from_prepared_inputs
    options = {'site_amplitude_prior_scale': .8} if cached else {'fixed_site_amplitudes': .8}
    with pytest.raises(ValueError, match="nuts_sampler='pymc'"):
        runner(prepared_inputs=prepared, independent_error_sd=_independent_error(prepared),
               tau_hours=6., sampler=RhimeSampler(nuts_sampler='numpyro'), **options)


def test_linked_ou_requires_same_units():
    observations = _prepared().observations.assign_coords(observation_units=('observation', ['ppm','ppm','per meg','per meg','per meg']))
    with pytest.raises(ValueError, match='same units'):
        linked_fixed_ou_alignment(observations)


def test_linked_cached_sampling_serializes_group_labels_and_joint_outputs(tmp_path):
    prepared = _prepared()
    trace = run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
        prepared_inputs=prepared, independent_error_sd=_independent_error(prepared),
        tau_hours={'co2:A': 6., 'o2:A': 8., 'o2:B': 2.}, site_amplitude_prior_scale=.8,
        sampler=RhimeSampler(nuts_sampler='pymc', draws=4, tune=4, chains=1, burn=0,
            sample_kwargs={'cores': 1, 'random_seed': 171, 'progressbar': False, 'compute_convergence_checks': False},
            sample_posterior_predictive=True),
    )
    path = tmp_path / 'linked.nc'
    save_inferencedata(trace, path)
    restored = load_inferencedata(path)
    assert restored.posterior.ou_site.values.tolist() == ['co2:A', 'o2:A', 'o2:B']
    assert restored.posterior.ou_species.values.tolist() == ['co2', 'o2', 'o2']
    assert restored.posterior.ou_station.values.tolist() == ['A', 'A', 'B']
    assert restored.posterior.ou_site_amplitude.attrs['units'] == 'ppm'
    assert restored.constant_data.ou_tau_hours.attrs['units'] == 'hours'
    assert restored.log_likelihood.y.dims == ('chain', 'draw')
    assert restored.log_likelihood.y.attrs['rhime_likelihood_scope'] == 'joint_observation_vector'
    assert restored.posterior_predictive.y.sizes['observation'] == 5
    assert restored.posterior_predictive.species.values.tolist() == ['co2','co2','o2','o2','o2']
