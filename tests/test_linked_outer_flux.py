"""Outer-flux option on linked CO2/O2 recipes: one state selection shared by both channels."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from openghg_inversions.correlated_state import CorrelatedPrior
from openghg_inversions.models.outer_flux import OuterFluxChoice
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model
from test_co2_o2_fixed_ou import _prepared
from test_rhime_co2_o2 import _independent_error

NOISE = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
GROUPS = ["inner", "inner", "outer", "outer", "outer"]


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64", cast_policy="numpy+floatX"):
        yield


def case(choice=OuterFluxChoice("fixed", ("outer",))):
    """Resolve one outer selection on a small linked preparation."""
    base = _prepared()
    state = base.retained_prior.state_dim
    mean = base.retained_prior.mean.assign_coords(basis_group=(state, GROUPS))
    prior = CorrelatedPrior(mean, base.retained_prior.arithmetic_covariance)
    inputs = dict(
        observations=base.observations,
        fixed_prior_contribution=base.fixed_prior_contribution,
        co2_sensitivity=base.co2_sensitivity,
        o2_sensitivity=base.o2_sensitivity,
        aggregation_error=base.aggregation_error,
        retained_prior=prior,
        independent_error_sd=_independent_error(base),
        outer_flux=choice,
    )
    return base, prior, inputs


def evaluate(model, names, point):
    function = model.compile_fn(
        model.replace_rvs_by_values([model[name] for name in names]), inputs=model.value_vars, on_unused_input="ignore",
    )
    return function(point)


def test_fixed_outer_states_are_held_in_the_joint_state():
    base, prior, inputs = case()
    model = build_co2_o2_model(**inputs, **NOISE)
    point = model.initial_point()
    assert point["flux_scaling_active_latent"].shape == (2,)
    for name in point:
        point[name] = np.linspace(-0.3, 0.4, point[name].size).reshape(point[name].shape)
    state, prediction = evaluate(model, ["flux_scaling", "modelled_concentration"], point)
    np.testing.assert_allclose(state[2:], prior.mean.values[2:])

    # Both channels see the raw forward model of the joint state.
    species = base.observations.indexes["observation"].get_level_values("species")
    raw = np.zeros(species.size)
    for channel in ("co2", "o2"):
        raw[species == channel] = getattr(base, f"{channel}_sensitivity").values @ state
    np.testing.assert_allclose(prediction, base.fixed_prior_contribution.values + raw, atol=1e-8)


def test_ordinary_and_cached_linked_models_have_the_same_density():
    _, _, inputs = case()
    ordinary = build_co2_o2_model(**inputs, **NOISE)
    cached = build_co2_o2_cached_sigma_model(**inputs, **NOISE)
    point = ordinary.initial_point()
    for name in point:
        point[name] = np.linspace(0.1, 0.3, point[name].size).reshape(point[name].shape)
    cached_point = cached.model.initial_point()
    cached_point.update(point)
    np.testing.assert_allclose(ordinary.compile_logp()(point), cached.model.compile_logp()(cached_point), atol=1e-8)
    cached.shared_cache.update(cached.target.refresh(np.exp(cached_point["ou_site_amplitude_log__"])))
    cached_likelihood = cached.model.replace_rvs_by_values([cached.cached_state_logp])[0]
    values = [cached.model.rvs_to_values[rv] for rv in cached.states]
    cached_outputs = cached.model.compile_fn(
        [cached_likelihood, *pt.grad(cached_likelihood, values)], inputs=cached.model.value_vars, on_unused_input="ignore",
    )(cached_point)
    direct_likelihood = ordinary.logp(vars=ordinary.observed_RVs)
    direct_values = [ordinary.rvs_to_values[ordinary[rv.name]] for rv in cached.states]
    direct_outputs = ordinary.compile_fn(
        [direct_likelihood, *pt.grad(direct_likelihood, direct_values)], inputs=ordinary.value_vars,
        on_unused_input="ignore",
    )(point)
    for actual, expected in zip(cached_outputs, direct_outputs):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-7)


def test_fixed_sources_hold_only_matching_outer_states():
    _, prior, _ = case()
    sources = prior.mean["source"].values
    held = (np.array(GROUPS) == "outer") & (sources == sources[-1])
    _, prior, inputs = case(OuterFluxChoice("fixed", ("outer",), (str(sources[-1]),)))
    np.testing.assert_array_equal(inputs["outer_flux"].resolve_states(prior).fixed, held)
    model = build_co2_o2_model(**inputs, **NOISE)
    assert model.initial_point()["flux_scaling_active_latent"].shape == (int((~held).sum()),)


@pytest.mark.parametrize("cached", [False, True])
def test_linked_builders_resolve_the_choice_on_the_shared_prior(cached):
    """A choice valid on another prior is checked again on the one the model samples."""
    _, prior, inputs = case()
    covariance = prior.arithmetic_covariance.to_numpy().copy()
    covariance[0, -1] = covariance[-1, 0] = 0.01 * np.sqrt(covariance[0, 0] * covariance[-1, -1])
    correlated = CorrelatedPrior(prior.mean, prior.arithmetic_covariance.copy(data=covariance))
    inputs["outer_flux"].resolve_states(prior)
    with pytest.raises(ValueError, match="zero prior covariance"):
        if cached:
            build_co2_o2_cached_sigma_model(**{**inputs, "retained_prior": correlated}, **NOISE)
        else:
            build_co2_o2_model(**{**inputs, "retained_prior": correlated}, **NOISE)


@pytest.mark.parametrize("cached", [False, True])
def test_linked_builders_reject_an_explicit_fix_on_a_jointly_pruned_held_state(cached):
    """Joint exact-zero pruning does not hide a conflicting explicit fixed value."""
    from openghg_inversions.models.state_activity import StateActivity

    _, prior, _ = case()
    state = prior.state_dim
    last = prior.mean[state].values[-1]
    _, _, inputs = case(OuterFluxChoice("fixed", ("outer",), (str(prior.mean["source"].values[-1]),)))
    for channel in ("co2_sensitivity", "o2_sensitivity"):
        inputs[channel] = inputs[channel].where(inputs[channel][state] != last, 0.0)
    policy = StateActivity(active=prior.mean[state] != last, fixed_value=2.0)
    build = build_co2_o2_cached_sigma_model if cached else build_co2_o2_model
    with pytest.raises(ValueError, match="at values other than their prior mean"):
        build(**inputs, state_activity=policy, **NOISE)
