"""Outer-region flux option: state selection, row centring and CO2 recipe rules."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

from boundary_test_utils import centered_transport
from openghg_inversions.correlated_state import CorrelatedPrior
from openghg_inversions.models.boundary import AdditiveChoice, AnomalyScalingChoice
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.models.outer_flux import OuterFluxChoice
from openghg_inversions.models.state_activity import (
    StateActivity,
    prepare_linear_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2.co2_cached_sigma_model import build_co2_cached_sigma_model
from openghg_inversions.rhime.co2.atmospheric_baseline import reconstruct_atmospheric_baseline
from openghg_inversions.rhime.co2.co2_model import build_co2_model

MEAN_SHIFT = {"pdf": "normal", "mu": 0.0, "sigma": 5.0}
NOISE = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
MULTIPLICATIVE = OuterFluxChoice("multiplicative", ("outer",), ("ocean",))


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
    """Two inner and three outer states; r1 has no response, r3 a constant one."""
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


def boundary(obs, mean_prior=MEAN_SHIFT):
    mass = xr.DataArray(
        [[0.6, 0.4], [0.2, 0.8], [0.5, 0.5], [0.7, 0.3]], dims=("nmeasure", "bc_region"),
        coords={"nmeasure": obs.nmeasure, "bc_region": ["N", "S"]}, attrs={"units": "1"},
    )
    anomaly = mass.copy(data=np.arange(8).reshape(4, 2) * 0.01 - 0.025).assign_attrs(units="ppm")
    transport = centered_transport(anomaly, mass, xr.DataArray(400.0, attrs={"units": "ppm"}))
    choice = AnomalyScalingChoice(mean_prior, 0.7, frequency="2D", anchor_time="2021-01-01")
    return choice.bind(obs, transport)


def bound(choice=MULTIPLICATIVE, cross=0.0, **additive):
    obs = observations()
    flux_prior = prior(cross)
    h = sensitivity(obs, flux_prior.mean)
    outer = choice.resolve_states(flux_prior).bind_rows(obs, h, **additive)
    return obs, flux_prior, h, outer


def additive_inputs(obs):
    patterns = ["bio_0", "bio_1", "FF_0"]
    design = xr.DataArray(
        [[0.1, 0.3, 0.2], [0.2, 0.3, 0.1], [0.4, 0.3, 0.3], [0.1, 0.3, 0.6]],
        dims=("nmeasure", "outer_pattern"),
        coords={
            "nmeasure": obs.nmeasure,
            "outer_pattern": patterns,
            "outer_source": ("outer_pattern", ["net_bio", "net_bio", "FF"]),
        },
        attrs={"units": "ppm"},
    )
    density_sd = xr.DataArray([2.0, 2.0, 1.0], dims="outer_pattern", coords={"outer_pattern": patterns},
                              attrs={"units": "mol m-2 s-1"})
    return dict(additive_design=design, additive_density_sd=density_sd)


def model_inputs(obs, flux_prior, h):
    return dict(
        retained_prior=flux_prior,
        fixed_prior_contribution=xr.full_like(obs, 0.2).rename("fixed_prior_contribution"),
        observations=obs,
        observation_error=xr.full_like(obs, 0.2),
        aggregation_error=resolve_aggregation_error(obs.to_dataset(name="mf"), "none"),
    )


def test_states_are_selected_by_basis_group_and_source():
    states = MULTIPLICATIVE.resolve_states(prior())
    np.testing.assert_array_equal(states.sampled, [False, False, True, True, False])
    np.testing.assert_array_equal(states.fixed, [False, False, False, False, True])

    for mode in ("fixed", "additive"):
        states = OuterFluxChoice(mode, ("outer",)).resolve_states(prior())
        np.testing.assert_array_equal(states.fixed, [False, False, True, True, True])
        assert not bool(states.sampled.any())


def test_selectors_need_their_coordinates_and_at_least_one_outer_state():
    unlabelled = CorrelatedPrior(
        prior().mean.drop_vars("source"), prior().arithmetic_covariance, covariance_dim="region_cov",
    )
    with pytest.raises(ValueError, match="'source' coordinate"):
        MULTIPLICATIVE.resolve_states(unlabelled)
    with pytest.raises(ValueError, match="No flux state"):
        OuterFluxChoice("fixed", ("missing",)).resolve_states(prior())
    with pytest.raises(ValueError, match="only to multiplicative"):
        OuterFluxChoice("fixed", ("outer",), ("ocean",))


def test_fixed_outer_states_must_be_uncorrelated_with_the_rest():
    with pytest.raises(ValueError, match="zero prior covariance"):
        MULTIPLICATIVE.resolve_states(prior(cross=0.01))
    # Correlation between sampled states is kept.
    MULTIPLICATIVE.resolve_states(prior())


def test_equal_site_weights_and_level_compensation():
    _, _, h, outer = bound()
    data = outer.data.compute()
    np.testing.assert_allclose(data.centering_weights, [1 / 6, 1 / 6, 1 / 6, 1 / 2])
    mu = data.centering_weights.values @ h.values
    np.testing.assert_allclose(data.outer_column_mean, np.where([0, 0, 1, 1, 0], mu, 0.0))
    np.testing.assert_allclose(data.level_compensation, mu[2] * 1.0 + mu[3] * 1.2)
    assert data.attrs["units"] == "ppm"


def test_centring_after_pruning_preserves_the_forward_model():
    _, flux_prior, h, outer = bound()
    prepared = prepare_linear_sensitivity(h, output_dim="nmeasure")
    assert prepared.sensitivity.dims == ("nmeasure", "region_retained")  # r1 was pruned
    centred = outer.centre(prepared)
    np.testing.assert_array_equal(centred.removed, prepared.removed)

    data = outer.data.compute()
    x = np.array([1.3, 0.8, 0.7, 1.6, 0.9])
    m = flux_prior.mean.values
    mu = data.outer_column_mean.values
    retained = centred.retained_indices
    lhs = centred.sensitivity.values @ x[retained] + 0.2 + float(data.level_compensation) + mu @ (x - m)
    np.testing.assert_allclose(lhs, h.values @ x + 0.2, atol=1e-14)

    # Sampled outer columns are centred; the constant one becomes zero but stays retained.
    weights = data.centering_weights.values
    sampled_retained = np.flatnonzero(outer.states.sampled.values[retained])
    np.testing.assert_allclose(weights @ centred.sensitivity.values[:, sampled_retained], 0.0, atol=1e-15)
    np.testing.assert_array_equal(centred.sensitivity.sel(region_retained="r3"), 0.0)


def test_centring_without_pruning_and_on_selected_rows():
    _, _, h, outer = bound()
    h = h.where(h.region != "r1", 0.25)
    prepared = prepare_linear_sensitivity(h, output_dim="nmeasure")
    assert prepared.sensitivity.dims == ("nmeasure", "region")
    rows = xr.DataArray([True, False, True, False], dims="nmeasure", coords={"nmeasure": h.nmeasure})
    centred = outer.centre(prepared, rows=rows).sensitivity
    mu = outer.data.outer_column_mean.values
    np.testing.assert_allclose(centred.values, h.values - np.outer(rows.values, mu))


def test_fix_states_holds_fixed_outer_states_at_their_prior_mean():
    _, _, h, outer = bound()
    prepared = prepare_linear_sensitivity(h, output_dim="nmeasure")
    policy = StateActivity(active=h.region != "r0", fixed_value=1.0)
    activity = outer.states.fix_states(resolve_state_activity(prepared.removed, policy))
    np.testing.assert_array_equal(activity.active, [False, False, True, True, False])
    np.testing.assert_allclose(activity.fixed_value, [1.0, 1.0, 1.0, 1.0, 0.9])


@pytest.mark.parametrize("mean_prior", [None, "no boundary"])
def test_recipes_require_a_boundary_mean_shift(mean_prior):
    obs, flux_prior, h, outer = bound()
    bc = None if mean_prior == "no boundary" else boundary(obs, mean_prior=None)
    with pytest.raises(ValueError, match="mean shift prior"):
        build_co2_model(h, **model_inputs(obs, flux_prior, h), boundary=bc, outer_flux=outer)
    with pytest.raises(ValueError, match="mean shift prior"):
        build_co2_cached_sigma_model(h, **model_inputs(obs, flux_prior, h), boundary=bc, outer_flux=outer, **NOISE)


def evaluate(model, names, point):
    function = model.compile_fn(
        model.replace_rvs_by_values([model[name] for name in names]), inputs=model.value_vars, on_unused_input="ignore",
    )
    return function(point)


def test_model_keeps_the_full_state_with_one_level():
    obs, flux_prior, h, outer = bound()
    model = build_co2_model(h, **model_inputs(obs, flux_prior, h), boundary=boundary(obs), outer_flux=outer)
    assert {"baseline_delta", "delta_sd"}.isdisjoint(model.named_vars)
    assert [rv.name for rv in model.free_RVs if "mean_shift" in rv.name] == ["bc_mean_shift"]

    # r0, r2 and r3 are sampled: r1 has no response, r4 is a fixed outer state.
    assert model.initial_point()["flux_scaling_active_latent"].shape == (3,)
    point = model.initial_point()
    point["flux_scaling_active_latent"] = np.array([0.3, -0.2, 0.5])
    state, prediction = evaluate(model, ["flux_scaling", "modelled_concentration"], point)
    assert state.shape == (5,)
    np.testing.assert_allclose(state[[1, 4]], [1.0, 0.9])

    # The absolute prediction is the raw forward model plus the boundary,
    # less the outer level the mean shift absorbs.
    mu_bc, = evaluate(model, ["mu_bc"], point)
    data = outer.data.compute()
    removed_level = data.outer_column_mean.values @ (state - flux_prior.mean.values)
    np.testing.assert_allclose(prediction, h.values @ state + 0.2 + mu_bc - removed_level, atol=1e-10)


def test_ordinary_and_cached_models_have_the_same_density():
    obs, flux_prior, h, outer = bound()
    inputs = model_inputs(obs, flux_prior, h)
    ordinary = build_co2_model(
        h, **inputs, boundary=boundary(obs), outer_flux=outer,
        likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=NOISE,
    )
    cached = build_co2_cached_sigma_model(h, **inputs, boundary=boundary(obs), outer_flux=outer, **NOISE)
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


def test_additive_mode_fixes_outer_states_and_keeps_every_pattern():
    obs = observations()
    obs_inputs = additive_inputs(obs)
    _, flux_prior, h, outer = bound(OuterFluxChoice("additive", ("outer",)), **obs_inputs)
    data = outer.data.compute()
    np.testing.assert_allclose(data.centering_weights.values @ data.additive_centered.values, 0.0, atol=1e-15)
    np.testing.assert_array_equal(data.additive_centered.sel(outer_pattern="bio_1"), 0.0)

    inputs = model_inputs(obs, flux_prior, h)
    for model in (
        build_co2_model(h, **inputs, boundary=boundary(obs), outer_flux=outer,
                        likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=NOISE),
        build_co2_cached_sigma_model(h, **inputs, boundary=boundary(obs), outer_flux=outer, **NOISE).model,
    ):
        assert model.initial_point()["outer_pattern_coefficient"].shape == (3,)
        assert model.initial_point()["flux_scaling_active_latent"].shape == (1,)  # only r0
    assert outer.outputs().roles == {
        "outer_flux_pattern_coefficient": "outer_pattern_coefficient",
        "outer_flux_pattern_concentration": "outer_pattern_contribution",
    }


@pytest.mark.parametrize("source", ["GPP", "ocean"])
def test_additive_patterns_reject_gross_and_ocean_sources(source):
    obs = observations()
    inputs = additive_inputs(obs)
    inputs["additive_design"] = inputs["additive_design"].assign_coords(
        outer_source=("outer_pattern", [source, "net_bio", "FF"]),
    )
    with pytest.raises(ValueError, match="outer_source"):
        bound(OuterFluxChoice("additive", ("outer",)), **inputs)


def test_additive_inputs_belong_to_additive_mode_only():
    obs = observations()
    with pytest.raises(ValueError, match="other modes take neither"):
        bound(MULTIPLICATIVE, **additive_inputs(obs))
    with pytest.raises(ValueError, match="other modes take neither"):
        bound(OuterFluxChoice("additive", ("outer",)))


def test_outputs_record_the_level_interpretation():
    metadata = bound()[3].outputs().metadata
    assert metadata["outer_flux_mode"] == "multiplicative"
    assert metadata["outer_fixed_sources"] == ["ocean"]
    assert metadata["boundary_mean_shift_interpretation"] == "unattributed atmospheric level"


def test_additive_level_only_boundary_works_with_outer_flux():
    obs, flux_prior, h, outer = bound()
    mass = xr.DataArray(
        [[0.6, 0.4], [0.2, 0.8], [0.5, 0.5], [0.7, 0.3]], dims=("nmeasure", "bc_region"),
        coords={"nmeasure": obs.nmeasure, "bc_region": ["N", "S"]}, attrs={"units": "1"},
    )
    anomaly = mass.copy(data=np.zeros((4, 2))).assign_attrs(units="ppm")
    transport = centered_transport(anomaly, mass, xr.DataArray(400.0, attrs={"units": "ppm"}))
    level_only = AdditiveChoice(MEAN_SHIFT, 0.0).bind(obs, transport)
    model = build_co2_model(h, **model_inputs(obs, flux_prior, h), boundary=level_only, outer_flux=outer)
    assert "bc_mean_shift" in model.named_vars


@pytest.mark.parametrize("mode", ["multiplicative", "additive"])
def test_atmospheric_baseline_is_the_prediction_less_inner_fluxes(mode):
    obs = observations()
    choice = MULTIPLICATIVE if mode == "multiplicative" else OuterFluxChoice("additive", ("outer",))
    _, flux_prior, h, outer = bound(choice, **(additive_inputs(obs) if mode == "additive" else {}))
    model = build_co2_model(h, **model_inputs(obs, flux_prior, h), boundary=boundary(obs), outer_flux=outer)
    point = model.initial_point()
    for name in point:
        point[name] = np.linspace(-0.4, 0.6, point[name].size).reshape(point[name].shape)
    names = ["flux_scaling", "mu_bc", "modelled_concentration"]
    if mode == "additive":
        names.append("outer_pattern_coefficient")
    values = dict(zip(names, evaluate(model, names, point)))

    posterior = xr.Dataset({
        "flux_scaling": ("region", values["flux_scaling"]),
        "mu_bc": ("nmeasure", values["mu_bc"]),
    }, coords={"region": h.region, "nmeasure": h.nmeasure})
    if mode == "additive":
        posterior["outer_pattern_coefficient"] = xr.DataArray(
            values["outer_pattern_coefficient"], dims="outer_pattern",
            coords={"outer_pattern": outer.data.outer_pattern},
        )
    result = reconstruct_atmospheric_baseline(
        outer.data, flux_sensitivity=h, fixed_prior_contribution=xr.full_like(obs, 0.2),
        prior_mean=flux_prior.mean, posterior=posterior,
    ).compute()

    inner = h.values[:, :2] @ values["flux_scaling"][:2]
    np.testing.assert_allclose(result.atmospheric_baseline, values["modelled_concentration"] - inner, atol=1e-10)
    np.testing.assert_allclose(result.outer_flux_contribution, h.values[:, 2:] @ values["flux_scaling"][2:], atol=1e-12)
    assert result.atmospheric_baseline.attrs["units"] == "ppm"


def test_binding_rejects_sensitivity_rows_that_differ_from_the_observations():
    obs = observations()
    flux_prior = prior()
    h = sensitivity(obs, flux_prior.mean)
    states = MULTIPLICATIVE.resolve_states(flux_prior)
    with pytest.raises(ValueError, match="exact"):
        states.bind_rows(obs, h.isel(nmeasure=slice(1, None)))


@pytest.mark.parametrize("change", ["rows", "missing_width", "extra_width"])
def test_additive_binding_needs_matching_rows_and_one_width_per_pattern(change):
    obs = observations()
    inputs = additive_inputs(obs)
    if change == "rows":
        inputs["additive_design"] = inputs["additive_design"].isel(nmeasure=slice(1, None))
    elif change == "missing_width":
        inputs["additive_density_sd"] = inputs["additive_density_sd"].isel(outer_pattern=[0, 1])
    else:
        extra = inputs["additive_density_sd"].isel(outer_pattern=[0]).assign_coords(outer_pattern=["bio_9"])
        inputs["additive_density_sd"] = xr.concat([inputs["additive_density_sd"], extra], "outer_pattern")
    with pytest.raises(ValueError, match="exact"):
        bound(OuterFluxChoice("additive", ("outer",)), **inputs)
