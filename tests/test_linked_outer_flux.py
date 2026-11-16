"""Outer-flux option on linked CO2/O2 recipes: one state selection, per-channel centring."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

from boundary_test_utils import centered_transport
from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.boundary import AnomalyScalingChoice
from openghg_inversions.models.outer_flux import OuterFluxChoice
from openghg_inversions.rhime.co2.co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_o2_model import bind_linked_outer_flux, build_co2_o2_model
from test_co2_o2_fixed_ou import _prepared
from test_rhime_co2_o2 import _independent_error

MEAN_SHIFT = {"pdf": "normal", "mu": 0.0, "sigma": 2.0}
NOISE = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
GROUPS = ["inner", "inner", "outer", "outer", "outer"]


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64", cast_policy="numpy+floatX"):
        yield


def native(base, channel):
    sensitivity = getattr(base, f"{channel}_sensitivity")
    return select_gathered_data_array(
        base.observations, key=channel, key_dim="species", ragged_dim="channel_observation", stack_dim="observation",
    ).rename(observation=str(sensitivity.dims[0])).assign_attrs(units="ppm")


def boundary(obs, reference, mean_prior=MEAN_SHIFT):
    row = str(obs.dims[0])
    mass = xr.DataArray(
        np.resize([[0.6, 0.4], [0.2, 0.8], [0.5, 0.5]], (obs.size, 2)), dims=(row, "bc_region"),
        coords={row: obs[row], "bc_region": ["N", "S"]}, attrs={"units": "1"},
    )
    anomaly = mass.copy(data=np.arange(2 * obs.size).reshape(-1, 2) * 0.01 - 0.02).assign_attrs(units="ppm")
    transport = centered_transport(
        anomaly.rename({row: "nmeasure"}), mass.rename({row: "nmeasure"}),
        xr.DataArray(reference, attrs={"units": "ppm"}),
    )
    return AnomalyScalingChoice(mean_prior, 0.3).bind(obs.rename({row: "nmeasure"}), transport)


def case(choice=None, mean_prior=MEAN_SHIFT):
    """Bind one outer selection to both channels of a small linked preparation."""
    base = _prepared()
    mean = base.retained_prior.mean.assign_coords(basis_group=(base.retained_prior.state_dim, GROUPS))
    prior = CorrelatedLognormalPrior(mean, base.retained_prior.arithmetic_covariance)
    choice = choice or OuterFluxChoice("multiplicative", ("outer",), ("ocean",))
    sensitivities = {
        channel: getattr(base, f"{channel}_sensitivity").assign_coords(site=native(base, channel).site)
        for channel in ("co2", "o2")
    }
    outer = bind_linked_outer_flux(
        choice, prior, observations=base.observations,
        co2_sensitivity=sensitivities["co2"], o2_sensitivity=sensitivities["o2"],
    )
    bounds = {
        channel: boundary(native(base, channel), reference, mean_prior)
        for channel, reference in (("co2", 400.0), ("o2", 200000.0))
    }
    inputs = dict(
        observations=base.observations,
        fixed_prior_contribution=base.fixed_prior_contribution,
        co2_sensitivity=base.co2_sensitivity,
        o2_sensitivity=base.o2_sensitivity,
        aggregation_error=base.aggregation_error,
        retained_prior=prior,
        independent_error_sd=_independent_error(base),
        boundary=bounds,
        outer_flux=outer,
    )
    return base, prior, inputs


def evaluate(model, names, point):
    function = model.compile_fn(
        model.replace_rvs_by_values([model[name] for name in names]), inputs=model.value_vars, on_unused_input="ignore",
    )
    return function(point)


def test_each_channel_centres_only_its_own_rows():
    base, prior, inputs = case()
    model = build_co2_o2_model(**inputs, **NOISE)
    point = model.initial_point()
    for name in point:
        point[name] = np.linspace(-0.3, 0.4, point[name].size).reshape(point[name].shape)
    state, prediction, co2_bc, o2_bc = evaluate(
        model, ["flux_scaling", "modelled_concentration", "co2_mu_bc", "o2_mu_bc"], point,
    )
    np.testing.assert_allclose(state[[3, 4]], prior.mean.values[[3, 4]])  # ocean held at its prior mean

    species = base.observations.indexes["observation"].get_level_values("species")
    removed = np.zeros(species.size)
    raw = np.zeros(species.size)
    for channel in ("co2", "o2"):
        rows = species == channel
        mu = inputs["outer_flux"][channel].data.outer_column_mean.values
        removed[rows] = mu @ (state - prior.mean.values)
        raw[rows] = getattr(base, f"{channel}_sensitivity").values @ state
    expected = base.fixed_prior_contribution.values + raw + co2_bc + o2_bc - removed
    np.testing.assert_allclose(prediction, expected, atol=1e-8)
    assert {"baseline_delta", "co2_baseline_delta"}.isdisjoint(model.named_vars)


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


def test_linked_option_needs_one_selection_both_channels_and_mean_shifts():
    _, _, inputs = case()
    fixed = case(OuterFluxChoice("fixed", ("outer",)))[2]["outer_flux"]
    with pytest.raises(ValueError, match="one outer state selection"):
        build_co2_o2_model(**{**inputs, "outer_flux": {"co2": inputs["outer_flux"]["co2"], "o2": fixed["o2"]}})
    with pytest.raises(ValueError, match="exactly 'co2' and 'o2'"):
        build_co2_o2_model(**{**inputs, "outer_flux": {"co2": inputs["outer_flux"]["co2"]}})
    _, _, no_shift = case(mean_prior=None)
    with pytest.raises(ValueError, match="mean shift prior"):
        build_co2_o2_model(**no_shift)


def test_linked_option_rejects_sampled_ocean():
    with pytest.raises(ValueError, match="ocean"):
        _, _, inputs = case(OuterFluxChoice("multiplicative", ("outer",)))
        build_co2_o2_model(**inputs)
