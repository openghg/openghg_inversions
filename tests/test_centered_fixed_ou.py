"""Centered fixed-OU likelihoods preserve absolute means, densities and gradients."""

from functools import wraps

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
from openghg_inversions.models.boundary import CenteredOperands, MultiplicativeChoice
from test_co2_cached_sigma import _inputs
from test_co2_o2_fixed_ou import _prepared, _kwargs
from openghg_inversions.rhime.specs import DEFAULT_BC_PRIOR
from boundary_test_utils import bound_model_arguments, bound_linked_model_arguments


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


@pytest.mark.parametrize("reference", [400.0, 209300.0])
@pytest.mark.parametrize("fixed_boundary", [False, True])
def test_single_centered_fixed_ou_preserves_full_density_and_gradient(reference, fixed_boundary):
    inputs = _inputs()
    operands = _operands(inputs.mf, reference)
    bound = CenteredOperands.from_dataset(operands).bind(inputs.mf)
    options = dict(
        retained_prior=CorrelatedLognormalPrior(
            inputs.alpha_prior_mean, inputs.alpha_prior_covariance, covariance_dim="region_cov"
        ),
        fixed_prior_contribution=inputs.fixed_prior_contribution,
        observations=(inputs.mf + reference).assign_attrs(units="ppm"),
        observation_error=inputs.mf_error,
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.01},
        bc_state_activity=StateActivity(active=[True, False] if fixed_boundary else True, fixed_value=1.0),
    )
    ou = dict(tau_hours=4.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8})
    stock = build_co2_model(
        inputs.H,
        **bound_model_arguments(
            {
                "likelihood_builder": add_fixed_ou_gaussian_likelihood,
                "likelihood_kwargs": ou,
                **options,
                "boundary_sensitivity": bound.H_bc_anomaly + reference * bound.H_bc_mass,
            }
        ),
    )
    centered = build_co2_model(
        inputs.H,
        **bound_model_arguments(
            {
                "likelihood_builder": add_fixed_ou_gaussian_likelihood,
                "likelihood_kwargs": ou,
                **options,
                "boundary_operands": operands,
            }
        ),
    )
    cached = build_co2_cached_sigma_model(
        inputs.H, **bound_model_arguments({**ou, **options, "boundary_operands": operands})
    )
    _assert_equivalent(stock, centered, cached)


@pytest.mark.parametrize("fixed_boundary", [False, True])
def test_linked_centered_fixed_ou_preserves_joint_density_and_full_gradient(fixed_boundary):
    from openghg_inversions.rhime.co2 import build_co2_o2_model, build_co2_o2_cached_sigma_model

    kwargs = _kwargs(_prepared())
    operands, absolute = {}, {}
    for channel, reference in (("co2", 400.0), ("o2", 209300.0)):
        rows = kwargs["observations"].sel(species=channel)
        rows = rows.rename({rows.dims[0]: kwargs[f"{channel}_sensitivity"].dims[0]})
        operands[channel] = _operands(rows, reference)
        bound = CenteredOperands.from_dataset(operands[channel]).bind(rows)
        absolute[channel] = bound.H_bc_anomaly + reference * bound.H_bc_mass
    shift = xr.where(kwargs["observations"].species == "co2", 400.0, 209300.0)
    kwargs["observations"] = kwargs["observations"] + shift
    kwargs.update(
        tau_hours=4.0,
        site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.8},
        bc_prior={channel: {"pdf": "normal", "mu": 1.0, "sigma": 0.01} for channel in ("co2", "o2")},
        bc_state_activity={
            channel: StateActivity(active=[True, False] if fixed_boundary else True, fixed_value=1.0)
            for channel in ("co2", "o2")
        },
    )
    stock = build_co2_o2_model(**bound_linked_model_arguments({"boundary_sensitivity": absolute, **kwargs}))
    centered = build_co2_o2_model(**bound_linked_model_arguments({"boundary_operands": operands, **kwargs}))
    cached = build_co2_o2_cached_sigma_model(
        **bound_linked_model_arguments({"boundary_operands": operands, **kwargs})
    )
    _assert_equivalent(stock, centered, cached)


@pytest.mark.parametrize("wrapped", [False, True])
def test_direct_centered_custom_likelihood_is_rejected_before_construction(monkeypatch, wrapped):
    """Reject callbacks and fixed-OU wrappers before graph construction."""
    from openghg_inversions.rhime.co2 import co2_model

    inputs = _inputs()
    operands = _operands(inputs.mf, 209300.0)
    seen = []
    original_model = co2_model.registered_model

    def model():
        """Record any premature PyMC graph construction."""
        seen.append("model")
        return original_model()

    def likelihood(**kwargs):
        """Record callbacks that must remain uncalled for centered inputs."""
        seen.append("callback")
        return add_fixed_ou_gaussian_likelihood(**kwargs, tau_hours=4.0, fixed_site_amplitudes=0.3)

    monkeypatch.setattr(co2_model, "registered_model", model)
    if wrapped:
        likelihood = wraps(add_fixed_ou_gaussian_likelihood)(likelihood)
    with pytest.raises(ValueError, match="Centered.*built-in fixed-OU.*custom"):
        build_co2_model(
            inputs.H,
            retained_prior=CorrelatedLognormalPrior(
                inputs.alpha_prior_mean, inputs.alpha_prior_covariance, covariance_dim="region_cov"
            ),
            fixed_prior_contribution=inputs.fixed_prior_contribution,
            observations=inputs.mf,
            observation_error=inputs.mf_error,
            aggregation_error=resolve_aggregation_error(inputs, "dense"),
            likelihood_builder=likelihood,
            boundary=MultiplicativeChoice(dict(DEFAULT_BC_PRIOR)).bind(inputs.mf, CenteredOperands.from_dataset(operands)),
        )
    assert seen == []


@pytest.mark.parametrize("reference", [400.0, 209300.0])
def test_fixed_amplitude_absolute_and_centered_density_full_gradient(reference):
    """Fixed amplitudes need an ordinary oracle, with non-unit fixed states."""
    inputs = _inputs()
    operands = _operands(inputs.mf, reference)
    bound = CenteredOperands.from_dataset(operands).bind(inputs.mf)
    options = dict(
        retained_prior=CorrelatedLognormalPrior(
            inputs.alpha_prior_mean, inputs.alpha_prior_covariance, covariance_dim="region_cov"
        ),
        fixed_prior_contribution=inputs.fixed_prior_contribution,
        observations=(inputs.mf + reference).assign_attrs(units="ppm"),
        observation_error=inputs.mf_error,
        aggregation_error=resolve_aggregation_error(inputs, "dense"),
        state_activity=StateActivity(active=[True, False], fixed_value=1.1),
        bc_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.01},
        bc_state_activity=StateActivity(active=[True, False], fixed_value=1.00003),
        likelihood_builder=add_fixed_ou_gaussian_likelihood,
        likelihood_kwargs={
            "tau_hours": {"AAA": 3.0, "BBB": 7.0},
            "fixed_site_amplitudes": {"AAA": 0.3, "BBB": 0.4},
        },
    )
    stock = build_co2_model(
        inputs.H,
        **bound_model_arguments(
            {**options, "boundary_sensitivity": bound.H_bc_anomaly + reference * bound.H_bc_mass}
        ),
    )
    centered = build_co2_model(inputs.H, **bound_model_arguments({**options, "boundary_operands": operands}))
    point = stock.initial_point()
    assert set(point) == set(centered.initial_point())
    for name in point:
        point[name] = np.asarray(point[name]) + (0.00003 if "bc" in name else 0.013)
    np.testing.assert_allclose(
        centered.compile_logp()(point), stock.compile_logp()(point), rtol=2e-10, atol=1e-8
    )
    gradients, means = [], []
    for model in (stock, centered):
        variables = sorted(model.value_vars, key=lambda value: value.name)
        gradients.append(
            model.compile_fn(
                pt.grad(model.logp(), variables), inputs=model.value_vars, on_unused_input="ignore"
            )(point)
        )
        means.append(
            model.compile_fn(
                model.replace_rvs_by_values([model["modelled_concentration"]]),
                inputs=model.value_vars,
                on_unused_input="ignore",
            )(point)[0]
        )
    for actual, expected in zip(gradients[1], gradients[0], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-8)
    np.testing.assert_allclose(means[1], means[0], rtol=1e-14, atol=1e-10)
