"""Centered fixed-OU likelihoods preserve absolute means, densities and gradients."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models import StateActivity
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2 import build_co2_model, build_co2_cached_sigma_model
from openghg_inversions.rhime.co2.centered_boundary import bind_boundary_operands
from test_co2_cached_sigma import _inputs
from test_co2_o2_fixed_ou import _prepared, _kwargs


@pytest.fixture(autouse=True)
def double_precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def _operands(rows, reference):
    dim = str(rows.dims[0])
    mass = xr.DataArray(np.tile([.7, .3], (rows.size, 1)), dims=(dim, "bc_region"),
                        coords={dim: rows[dim], "bc_region": ["north", "south"]}, attrs={"units": "1"})
    return xr.Dataset({
        "H_bc_mass": mass.rename({dim: "nmeasure"}),
        "H_bc_anomaly": (mass.copy(data=np.tile([.02, -.01], (rows.size, 1)))
                           .rename({dim: "nmeasure"}).assign_attrs(units="ppm")),
        "bc_reference": xr.DataArray(reference, attrs={"units": "ppm"}),
    }, attrs={"boundary_representation": "centered-v1"})


def _assert_equivalent(stock, centered, cached):
    point = stock.initial_point()
    assert set(point) == set(centered.initial_point()) == set(cached.model.initial_point())
    for name in point:
        point[name] = np.asarray(point[name]) + .013
    point["ou_site_amplitude_log__"] = np.log(np.linspace(.3, .4, point["ou_site_amplitude_log__"].size))
    cached.shared_cache.update(cached.target.refresh(np.exp(point["ou_site_amplitude_log__"])))
    def gradient(model):
        values = sorted(model.value_vars, key=lambda value: value.name)
        gradients = model.compile_fn(pt.grad(model.logp(), values), inputs=model.value_vars,
                                      on_unused_input="ignore")(point)
        return np.concatenate([np.atleast_1d(value).ravel() for value in gradients])

    means = []
    for model in (stock, centered, cached.model):
        means.append(model.compile_fn(model.replace_rvs_by_values([model["modelled_concentration"]]),
                                      inputs=model.value_vars, on_unused_input="ignore")(point)[0])
        np.testing.assert_allclose(model.compile_logp()(point), stock.compile_logp()(point), rtol=2e-10, atol=1e-8)
        np.testing.assert_allclose(gradient(model), gradient(stock), rtol=2e-10, atol=1e-8)
    np.testing.assert_allclose(means[0], means[1], rtol=1e-14)
    np.testing.assert_allclose(means[0], means[2], rtol=1e-14)
    cached_expression = cached.model.replace_rvs_by_values([cached.cached_state_logp])[0]
    cached_logp = cached.model.compile_fn(cached_expression, inputs=cached.model.value_vars,
                                        on_unused_input="ignore")(point)
    np.testing.assert_allclose(cached_logp, stock.compile_logp(vars=stock.observed_RVs)(point), rtol=2e-10, atol=1e-8)
    state_values = [cached.model.rvs_to_values[state] for state in cached.states]
    observed = cached.model.logp(vars=cached.model.observed_RVs)
    gradients = cached.model.compile_fn([*pt.grad(cached_expression, state_values), *pt.grad(observed, state_values)],
                                        inputs=cached.model.value_vars, on_unused_input="ignore")(point)
    for actual, expected in zip(gradients[:len(state_values)], gradients[len(state_values):], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-8)


@pytest.mark.parametrize("reference", [400., 209300.])
@pytest.mark.parametrize("fixed_boundary", [False, True])
def test_single_centered_fixed_ou_preserves_full_density_and_gradient(reference, fixed_boundary):
    inputs = _inputs()
    operands = _operands(inputs.mf, reference)
    bound = bind_boundary_operands(operands, inputs.mf)
    options = dict(
        retained_prior=CorrelatedLognormalPrior(inputs.alpha_prior_mean, inputs.alpha_prior_covariance,
                                               covariance_dim="region_cov"),
        fixed_prior_contribution=inputs.fixed_prior_contribution,
        observations=(inputs.mf + reference).assign_attrs(units="ppm"),
        observation_error=inputs.mf_error, aggregation_error=resolve_aggregation_error(inputs, "dense"),
        bc_prior={"pdf": "normal", "mu": 1., "sigma": .01},
        bc_state_activity=StateActivity(active=[True, False] if fixed_boundary else True, fixed_value=1.),
    )
    ou = dict(tau_hours=4., site_amplitude_prior={"pdf": "halfnormal", "sigma": .8})
    stock = build_co2_model(inputs.H, boundary_sensitivity=bound.H_bc_anomaly + reference * bound.H_bc_mass,
                            likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=ou, **options)
    centered = build_co2_model(inputs.H, boundary_operands=operands,
                               likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=ou, **options)
    cached = build_co2_cached_sigma_model(inputs.H, boundary_operands=operands, **ou, **options)
    _assert_equivalent(stock, centered, cached)


@pytest.mark.parametrize("fixed_boundary", [False, True])
def test_linked_centered_fixed_ou_preserves_joint_density_and_full_gradient(fixed_boundary):
    from openghg_inversions.rhime.co2 import build_co2_o2_model, build_co2_o2_cached_sigma_model

    kwargs = _kwargs(_prepared())
    operands, absolute = {}, {}
    for channel, reference in (("co2", 400.), ("o2", 209300.)):
        rows = kwargs["observations"].sel(species=channel)
        rows = rows.rename({rows.dims[0]: kwargs[f"{channel}_sensitivity"].dims[0]})
        operands[channel] = _operands(rows, reference)
        bound = bind_boundary_operands(operands[channel], rows)
        absolute[channel] = bound.H_bc_anomaly + reference * bound.H_bc_mass
    shift = xr.where(kwargs["observations"].species == "co2", 400., 209300.)
    kwargs["observations"] = kwargs["observations"] + shift
    kwargs.update(tau_hours=4., site_amplitude_prior={"pdf": "halfnormal", "sigma": .8},
                  bc_prior={channel: {"pdf": "normal", "mu": 1., "sigma": .01} for channel in ("co2", "o2")},
                  bc_state_activity={channel: StateActivity(active=[True, False] if fixed_boundary else True,
                                                            fixed_value=1.) for channel in ("co2", "o2")})
    stock = build_co2_o2_model(boundary_sensitivity=absolute, **kwargs)
    centered = build_co2_o2_model(boundary_operands=operands, **kwargs)
    cached = build_co2_o2_cached_sigma_model(boundary_operands=operands, **kwargs)
    _assert_equivalent(stock, centered, cached)


def test_direct_custom_likelihood_receives_centered_observations_and_mean():
    inputs = _inputs()
    operands = _operands(inputs.mf, 209300.)
    seen = {}

    def likelihood(**kwargs):
        seen.update(kwargs)
        return add_fixed_ou_gaussian_likelihood(**kwargs, tau_hours=4., fixed_site_amplitudes=.3)

    model = build_co2_model(inputs.H,
        retained_prior=CorrelatedLognormalPrior(inputs.alpha_prior_mean, inputs.alpha_prior_covariance,
                                               covariance_dim="region_cov"),
        fixed_prior_contribution=inputs.fixed_prior_contribution, observations=inputs.mf,
        observation_error=inputs.mf_error, aggregation_error=resolve_aggregation_error(inputs, "dense"),
        boundary_operands=operands, likelihood_builder=likelihood)
    xr.testing.assert_allclose(seen["observations"], inputs.mf - 209300.)
    mean, absolute = model.compile_fn(model.replace_rvs_by_values([seen["mean"], model["modelled_concentration"]]),
                                      inputs=model.value_vars, on_unused_input="ignore")(model.initial_point())
    np.testing.assert_allclose(mean + 209300., absolute, rtol=1e-14)


@pytest.mark.parametrize("reference", [400., 209300.])
def test_fixed_amplitude_absolute_and_centered_density_full_gradient(reference):
    """Fixed amplitudes need an ordinary oracle, with non-unit fixed states."""
    inputs = _inputs()
    operands = _operands(inputs.mf, reference)
    bound = bind_boundary_operands(operands, inputs.mf)
    options = dict(
        retained_prior=CorrelatedLognormalPrior(inputs.alpha_prior_mean, inputs.alpha_prior_covariance,
                                               covariance_dim="region_cov"),
        fixed_prior_contribution=inputs.fixed_prior_contribution,
        observations=(inputs.mf + reference).assign_attrs(units="ppm"),
        observation_error=inputs.mf_error, aggregation_error=resolve_aggregation_error(inputs, "dense"),
        state_activity=StateActivity(active=[True, False], fixed_value=1.1),
        bc_prior={"pdf": "normal", "mu": 1., "sigma": .01},
        bc_state_activity=StateActivity(active=[True, False], fixed_value=1.00003),
        likelihood_builder=add_fixed_ou_gaussian_likelihood,
        likelihood_kwargs={"tau_hours": {"AAA": 3., "BBB": 7.}, "fixed_site_amplitudes": {"AAA": .3, "BBB": .4}},
    )
    stock = build_co2_model(inputs.H, boundary_sensitivity=bound.H_bc_anomaly + reference * bound.H_bc_mass, **options)
    centered = build_co2_model(inputs.H, boundary_operands=operands, **options)
    point = stock.initial_point()
    assert set(point) == set(centered.initial_point())
    for name in point:
        point[name] = np.asarray(point[name]) + (.00003 if "bc" in name else .013)
    np.testing.assert_allclose(centered.compile_logp()(point), stock.compile_logp()(point), rtol=2e-10, atol=1e-8)
    gradients, means = [], []
    for model in (stock, centered):
        variables = sorted(model.value_vars, key=lambda value: value.name)
        gradients.append(model.compile_fn(pt.grad(model.logp(), variables), inputs=model.value_vars,
                                          on_unused_input="ignore")(point))
        means.append(model.compile_fn(model.replace_rvs_by_values([model["modelled_concentration"]]),
                                      inputs=model.value_vars, on_unused_input="ignore")(point)[0])
    for actual, expected in zip(gradients[1], gradients[0], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-8)
    np.testing.assert_allclose(means[1], means[0], rtol=1e-14, atol=1e-10)
