"""Outer-region flux option: state selection, fixed and marginalised outer states in the CO2 recipes."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

from boundary_test_utils import centered_transport
from openghg_inversions.correlated_state import CorrelatedPrior
from openghg_inversions.models.boundary import AnomalyScalingChoice
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.models.outer_flux import OuterFluxChoice
from openghg_inversions.models.state_activity import (
    StateActivity,
    prepare_linear_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.observation_error import AggregationError, resolve_aggregation_error
from openghg_inversions.rhime.co2.co2_cached_sigma_model import build_co2_cached_sigma_model
from openghg_inversions.rhime.co2.co2_model import build_co2_model

MEAN_SHIFT = {"pdf": "normal", "mu": 0.0, "sigma": 5.0}
NOISE = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
FIXED = OuterFluxChoice("fixed", ("outer",))
FIXED_OCEAN = OuterFluxChoice("fixed", ("outer",), ("ocean",))
MARGINAL = OuterFluxChoice("marginalise", ("outer",))
MARGINAL_OCEAN = OuterFluxChoice("marginalise", ("outer",), ("ocean",))


@pytest.fixture(autouse=True)
def precision():
    # Centered boundaries and scalar prior constants need float64.
    with pytensor.config.change_flags(floatX="float64", cast_policy="numpy+floatX"):
        yield


def observations():
    return xr.DataArray(
        [410.0, 411.0, 412.0, 413.0],
        dims="nmeasure",
        coords={
            "nmeasure": np.arange(4),
            "site": ("nmeasure", ["A", "A", "A", "B"]),
            "time": ("nmeasure", np.datetime64("2021-01-01") + np.arange(4).astype("timedelta64[D]")),
        },
        attrs={"units": "ppm"},
    )


def prior(cross=0.0):
    """Two inner and three outer states; r1 has no response."""
    mean = xr.DataArray(
        [1.0, 1.0, 1.0, 1.2, 0.9],
        dims="region",
        coords={
            "region": [f"r{i}" for i in range(5)],
            "basis_group": ("region", ["inner", "inner", "outer", "outer", "outer"]),
            "source": ("region", ["GPP", "TER", "GPP", "TER", "ocean"]),
        },
    )
    covariance = np.diag([0.08, 0.07, 0.06, 0.05, 0.04])
    covariance[0, 1] = covariance[1, 0] = 0.01
    covariance[0, 4] = covariance[4, 0] = cross
    return CorrelatedPrior(mean, covariance, covariance_dim="region_cov")


def sensitivity(obs, mean):
    values = np.array([
        [0.3, 0.0, 0.2, 0.4, 0.1],
        [0.5, 0.0, 0.6, 0.4, 0.3],
        [0.1, 0.0, 0.9, 0.4, 0.2],
        [0.7, 0.0, 0.3, 0.4, 0.5],
    ])
    return xr.DataArray(
        values, dims=("nmeasure", "region"), coords={**dict(obs.coords), **dict(mean.coords)}, attrs={"units": "ppm"},
    )


def boundary(obs):
    mass = xr.DataArray(
        [[0.6, 0.4], [0.2, 0.8], [0.5, 0.5], [0.7, 0.3]], dims=("nmeasure", "bc_region"),
        coords={"nmeasure": obs.nmeasure, "bc_region": ["N", "S"]}, attrs={"units": "1"},
    )
    anomaly = mass.copy(data=np.arange(8).reshape(4, 2) * 0.01 - 0.025).assign_attrs(units="ppm")
    transport = centered_transport(anomaly, mass, xr.DataArray(400.0, attrs={"units": "ppm"}))
    choice = AnomalyScalingChoice(MEAN_SHIFT, 0.7, frequency="2D", anchor_time="2021-01-01")
    return choice.bind(obs, transport)


def resolved(choice=FIXED, cross=0.0):
    obs = observations()
    flux_prior = prior(cross)
    h = sensitivity(obs, flux_prior.mean)
    return obs, flux_prior, h, choice.resolve_states(flux_prior)


def inputs(cross=0.0):
    obs = observations()
    flux_prior = prior(cross)
    return obs, flux_prior, sensitivity(obs, flux_prior.mean)


def model_inputs(obs, flux_prior):
    return dict(
        retained_prior=flux_prior,
        fixed_prior_contribution=xr.full_like(obs, 0.2).rename("fixed_prior_contribution"),
        observations=obs,
        observation_error=xr.full_like(obs, 0.2),
        aggregation_error=resolve_aggregation_error(obs.to_dataset(name="mf"), "none"),
    )


def test_states_are_selected_by_basis_group_and_source():
    np.testing.assert_array_equal(FIXED.resolve_states(prior()).fixed, [False, False, True, True, True])
    np.testing.assert_array_equal(FIXED_OCEAN.resolve_states(prior()).fixed, [False, False, False, False, True])


def test_only_the_fixed_and_marginalise_modes_are_available():
    for mode in ("multiplicative", "additive"):
        with pytest.raises(ValueError, match="Unknown outer-flux mode"):
            OuterFluxChoice(mode, ("outer",))  # type: ignore[arg-type]


def test_selectors_need_their_coordinates_and_matching_outer_states():
    unlabelled = CorrelatedPrior(
        prior().mean.drop_vars("source"), prior().arithmetic_covariance, covariance_dim="region_cov",
    )
    with pytest.raises(ValueError, match="'source' coordinate"):
        FIXED_OCEAN.resolve_states(unlabelled)
    # Without fixed_sources the source labels are not needed.
    FIXED.resolve_states(unlabelled)
    with pytest.raises(ValueError, match="No flux state"):
        OuterFluxChoice("fixed", ("missing",)).resolve_states(prior())
    with pytest.raises(ValueError, match="No outer flux state"):
        OuterFluxChoice("fixed", ("outer",), ("FF",)).resolve_states(prior())
    # A partly matching selector names every unmatched label.
    with pytest.raises(ValueError, match=r"\['missing', 'other'\]"):
        OuterFluxChoice("fixed", ("outer", "missing", "other")).resolve_states(prior())
    with pytest.raises(ValueError, match=r"source \['Ggp'\]"):
        OuterFluxChoice("fixed", ("outer",), ("ocean", "Ggp")).resolve_states(prior())
    with pytest.raises(ValueError, match="at least one outer basis group"):
        OuterFluxChoice("fixed", ())


def test_selectors_accept_lazy_labels():
    pytest.importorskip("dask")
    eager = prior()
    state = eager.mean.dims[0]
    mean = eager.mean.assign_coords(
        basis_group=(state, eager.mean.basis_group.data.astype(object)),
        source=(state, eager.mean.source.data.astype(object)),
    )
    mean = mean.assign_coords(basis_group=mean.basis_group.chunk(), source=mean.source.chunk())
    lazy = CorrelatedPrior(mean, eager.arithmetic_covariance, covariance_dim="region_cov")
    np.testing.assert_array_equal(FIXED_OCEAN.resolve_states(lazy).fixed, [False, False, False, False, True])
    with pytest.raises(ValueError, match=r"source \['Ggp'\]"):
        OuterFluxChoice("fixed", ("outer",), ("ocean", "Ggp")).resolve_states(lazy)


def test_fixed_outer_states_must_be_uncorrelated_with_the_rest():
    with pytest.raises(ValueError, match="zero prior covariance"):
        FIXED_OCEAN.resolve_states(prior(cross=0.01))
    # Marginalising conditions F on the other states instead.
    states = MARGINAL_OCEAN.resolve_states(prior(cross=0.01))
    assert states.regression.shape == (1, 4)
    assert MARGINAL_OCEAN.resolve_states(prior()).regression is None
    # Correlation among states that stay sampled is kept.
    FIXED_OCEAN.resolve_states(prior())


def test_fix_states_holds_fixed_outer_states_at_their_prior_mean():
    _, _, h, states = resolved(FIXED_OCEAN)
    prepared = prepare_linear_sensitivity(h, output_dim="nmeasure")
    policy = StateActivity(active=h.region != "r0", fixed_value=1.0)
    activity = states.fix_states(resolve_state_activity(prepared.removed, policy))
    np.testing.assert_array_equal(activity.active, [False, False, True, True, False])
    np.testing.assert_allclose(activity.fixed_value, [1.0, 1.0, 1.0, 1.0, 0.9])


def test_fix_states_rejects_conflicting_explicit_fixed_values():
    _, _, h, states = resolved(FIXED_OCEAN)
    prepared = prepare_linear_sensitivity(h, output_dim="nmeasure")
    explicit_ocean = h.region != "r4"
    with pytest.raises(ValueError, match=r"\['r4'\] at values other than their prior mean"):
        states.fix_states(resolve_state_activity(prepared.removed, StateActivity(explicit_ocean, fixed_value=2.0)))
    # An explicit fix at the prior mean agrees with the option.
    matching = xr.where(h.region == "r4", 0.9, 1.0)
    activity = states.fix_states(resolve_state_activity(prepared.removed, StateActivity(explicit_ocean, matching)))
    np.testing.assert_allclose(activity.fixed_value, [1.0, 1.0, 1.0, 1.0, 0.9])
    # A held state inactive only through exact-zero pruning is not an explicit fix.
    pruned = prepare_linear_sensitivity(h.where(h.region != "r4", 0.0), output_dim="nmeasure")
    activity = states.fix_states(resolve_state_activity(pruned.removed, StateActivity(fixed_value=1.0)))
    np.testing.assert_allclose(activity.fixed_value, [1.0, 1.0, 1.0, 1.0, 0.9])
    # An explicit fix still conflicts when the state's column is also zero.
    with pytest.raises(ValueError, match=r"\['r4'\] at values other than their prior mean"):
        states.fix_states(resolve_state_activity(pruned.removed, StateActivity(explicit_ocean, fixed_value=2.0)))


def evaluate(model, names, point):
    function = model.compile_fn(
        model.replace_rvs_by_values([model[name] for name in names]), inputs=model.value_vars, on_unused_input="ignore",
    )
    return function(point)


@pytest.mark.parametrize("with_boundary", [True, False])
def test_fixed_outer_states_keep_the_raw_forward_model(with_boundary):
    obs, flux_prior, h = inputs()
    bc = boundary(obs) if with_boundary else None
    model = build_co2_model(h, **model_inputs(obs, flux_prior), boundary=bc, outer_flux=FIXED)

    # Only r0 is sampled: r1 has no response and r2..r4 are fixed outer states.
    point = model.initial_point()
    assert point["flux_scaling_active_latent"].shape == (1,)
    point["flux_scaling_active_latent"] = np.array([0.3])
    state, prediction = evaluate(model, ["flux_scaling", "modelled_concentration"], point)
    np.testing.assert_allclose(state[1:], [1.0, 1.0, 1.2, 0.9])
    mu_bc = evaluate(model, ["mu_bc"], point)[0] if with_boundary else 0.0
    np.testing.assert_allclose(prediction, h.values @ state + 0.2 + mu_bc, atol=1e-10)


@pytest.mark.parametrize(
    "choice, cross", [(FIXED_OCEAN, 0.0), (MARGINAL_OCEAN, 0.0), (MARGINAL_OCEAN, 0.01)],
    ids=["fixed", "marginalise", "marginalise-correlated"],
)
def test_ordinary_and_cached_models_have_the_same_density(choice, cross):
    obs, flux_prior, h = inputs(cross)
    common = model_inputs(obs, flux_prior)
    ordinary = build_co2_model(
        h, **common, boundary=boundary(obs), outer_flux=choice,
        likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=NOISE,
    )
    cached = build_co2_cached_sigma_model(h, **common, boundary=boundary(obs), outer_flux=choice, **NOISE)
    point = ordinary.initial_point()
    # r0, r2, r3; r1 too when the held ocean is correlated with the inner states.
    assert point["flux_scaling_active_latent"].shape == (4 if cross else 3,)
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


@pytest.mark.parametrize("cached", [False, True])
def test_builders_resolve_the_choice_on_the_prior_they_sample(cached):
    # The choice is valid on an uncorrelated prior but not on this one, so
    # the builder must check it against the prior it is given.
    obs, flux_prior, h = inputs(cross=0.01)
    FIXED_OCEAN.resolve_states(prior())
    with pytest.raises(ValueError, match="zero prior covariance"):
        if cached:
            build_co2_cached_sigma_model(h, **model_inputs(obs, flux_prior), outer_flux=FIXED_OCEAN, **NOISE)
        else:
            build_co2_model(h, **model_inputs(obs, flux_prior), outer_flux=FIXED_OCEAN)


def test_outputs_record_the_mode_and_selectors():
    outputs = FIXED_OCEAN.outputs()
    assert FIXED_OCEAN.resolve_states(prior()).outputs() == outputs
    assert outputs.roles == {} and outputs.units == {}
    assert outputs.metadata == {
        "outer_flux_mode": "fixed", "outer_groups": ["outer"], "outer_fixed_sources": ["ocean"],
    }


def aggregation_forms(obs):
    """The same 4x4 aggregation covariance as none, diagonal, low-rank and dense inputs."""
    rows = obs.nmeasure
    factor = np.array([[0.2], [0.1], [-0.1], [0.3]])
    diagonal = np.array([0.01, 0.02, 0.01, 0.03])
    dense = factor @ factor.T + np.diag(diagonal)
    data = obs.to_dataset(name="mf").assign(
        aggregation_error_sd=("nmeasure", np.sqrt(np.diag(dense))),
        low_rank_factor=(("nmeasure", "agg_rank"), factor),
        diagonal_residual_variance=("nmeasure", diagonal),
        aggregation_error_covariance=(("nmeasure", "nmeasure_cov"), dense),
    ).assign_coords(nmeasure_cov=rows.values)
    return {
        "none": (resolve_aggregation_error(obs.to_dataset(name="mf"), "none"), np.zeros((4, 4))),
        "diagonal": (resolve_aggregation_error(data, "diagonal"), np.diag(np.diag(dense))),
        "low_rank": (resolve_aggregation_error(data, "low_rank"), dense),
        "dense": (resolve_aggregation_error(data, "dense"), dense),
    }


def covariance_of(error: AggregationError) -> np.ndarray:
    if error.mode == "dense":
        return error.covariance.values
    if error.mode == "low_rank":
        return error.factor.values @ error.factor.values.T + np.diag(error.diagonal_variance.values)
    if error.mode == "diagonal":
        return np.diag(error.diagonal_variance.values)
    return np.zeros((error.marginal_variance.size,) * 2)


def test_marginalise_adds_the_outer_prior_covariance_to_every_error_form():
    obs, flux_prior, h, states = resolved(MARGINAL)
    outer = slice(2, 5)
    added = h.values[:, outer] @ flux_prior.arithmetic_covariance.values[outer, outer] @ h.values[:, outer].T
    assert states.prior_factor.shape == (3, 3)
    for form, (error, covariance) in aggregation_forms(obs).items():
        result = states.add_marginal_error(h, error, output_dim="nmeasure")
        assert result.mode == ("dense" if form == "dense" else "low_rank"), form
        np.testing.assert_allclose(covariance_of(result), covariance + added, atol=1e-14, err_msg=form)
        np.testing.assert_allclose(result.marginal_variance, np.diag(covariance + added), atol=1e-14, err_msg=form)
    # Fixed mode adds nothing.
    error = aggregation_forms(obs)["low_rank"][0]
    assert FIXED.resolve_states(flux_prior).add_marginal_error(h, error, output_dim="nmeasure") is error


def gaussian_posterior(design, offset, observations, noise, mean, covariance):
    """Exact posterior mean and covariance of x ~ N(mean, covariance), y ~ N(design x + offset, noise)."""
    gain = covariance @ design.T @ np.linalg.inv(design @ covariance @ design.T + noise)
    return mean + gain @ (observations - offset - design @ mean), covariance - gain @ design @ covariance


@pytest.mark.parametrize(
    "choice, cross", [(MARGINAL, 0.0), (MARGINAL_OCEAN, 0.0), (MARGINAL, 0.01), (MARGINAL_OCEAN, 0.01)],
    ids=["outer", "ocean", "outer-correlated", "ocean-correlated"],
)
def test_marginalised_posterior_equals_the_jointly_sampled_inner_marginal(choice, cross):
    """Linear Gaussian check: integrating O out analytically equals sampling it uncentred."""
    obs, flux_prior, h, states = resolved(choice, cross)
    error, aggregation = aggregation_forms(obs)["low_rank"]
    noise = aggregation + np.diag(np.full(4, 0.2**2))
    design, mean, covariance = h.values, flux_prior.mean.values, flux_prior.arithmetic_covariance.values
    y = obs.values - 410.0
    fixed = xr.full_like(obs, 0.2)
    joint_mean, joint_covariance = gaussian_posterior(design, fixed.values, y, noise, mean, covariance)

    held = states.fixed.values
    kept = ~held
    conditioned, offset = states.condition_sensitivity(h, fixed, output_dim="nmeasure")
    assert (conditioned is h) == (cross == 0.0)
    marginal = states.add_marginal_error(conditioned, error, output_dim="nmeasure")
    marginal_mean, marginal_covariance = gaussian_posterior(
        conditioned.values[:, kept], offset.values + design[:, held] @ mean[held], y,
        covariance_of(marginal) + np.diag(np.full(4, 0.2**2)), mean[kept], covariance[np.ix_(kept, kept)],
    )
    np.testing.assert_allclose(marginal_mean, joint_mean[kept], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(marginal_covariance, joint_covariance[np.ix_(kept, kept)], rtol=1e-12, atol=1e-13)
    if cross:
        return
    # Holding uncorrelated O with zero uncertainty is not this marginal: it is tighter.
    _, fixed_covariance = gaussian_posterior(
        design[:, kept], fixed.values + design[:, held] @ mean[held], y, noise, mean[kept], covariance[np.ix_(kept, kept)],
    )
    assert fixed_covariance[0, 0] < joint_covariance[0, 0] - 1e-4


@pytest.mark.parametrize("cross", [0.0, 0.01], ids=["uncorrelated", "correlated"])
def test_marginalised_model_likelihood_uses_the_widened_error(cross):
    """The likelihood is p(y | x_I) with x_O integrated out under its conditional Gaussian prior."""
    obs, flux_prior, h, _ = resolved(MARGINAL, cross)
    inputs = model_inputs(obs, flux_prior)
    inputs["aggregation_error"], aggregation = aggregation_forms(obs)["low_rank"]
    model = build_co2_model(
        h, **inputs, outer_flux=MARGINAL, likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=NOISE,
    )
    point = model.initial_point()
    # r1 has no direct response, but with r0 correlated to the outer ocean it predicts it, so it is sampled.
    active = 2 if cross else 1
    assert point["flux_scaling_active_latent"].shape == (active,)
    point["flux_scaling_active_latent"] = np.array([0.3, -0.2][:active])
    # A negligible OU amplitude leaves R + A + H_O Sigma_OO H_O^T.
    point["ou_site_amplitude_log__"] = np.full(point["ou_site_amplitude_log__"].shape, np.log(1e-7))
    (state,) = evaluate(model, ["flux_scaling"], point)
    np.testing.assert_allclose(state[2:], flux_prior.mean.values[2:])
    # Independent reference: x_O | x_I ~ N(m_O + S_OI S_II^-1 (x_I - m_I), S_OO - S_OI S_II^-1 S_IO).
    mean, cov = flux_prior.mean.values, flux_prior.arithmetic_covariance.values
    inner, outer = slice(0, 2), slice(2, 5)
    gain = np.linalg.solve(cov[inner, inner], cov[inner, outer]).T
    outer_mean = mean[outer] + gain @ (state[inner] - mean[inner])
    added = h.values[:, outer] @ (cov[outer, outer] - gain @ cov[inner, outer]) @ h.values[:, outer].T
    covariance = aggregation + added + np.diag(np.full(4, 0.2**2))
    residual = obs.values - (h.values[:, inner] @ state[inner] + h.values[:, outer] @ outer_mean + 0.2)
    expected = -0.5 * (residual @ np.linalg.solve(covariance, residual) + np.linalg.slogdet(2 * np.pi * covariance)[1])
    likelihood = model.compile_fn(
        model.logp(vars=model.observed_RVs), inputs=model.value_vars, on_unused_input="ignore",
    )(point)
    np.testing.assert_allclose(likelihood, expected, rtol=1e-9)


def test_marginalised_outputs_record_the_added_rank():
    outputs = MARGINAL_OCEAN.resolve_states(prior()).outputs()
    assert outputs.metadata["outer_flux_mode"] == "marginalise"
    assert outputs.metadata["outer_marginal_rank"] == 1
