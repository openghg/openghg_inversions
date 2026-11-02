"""Independent shared-baseline equations and actual cached sampler oracle."""

import numpy as np
import pymc as pm
import pytest
import pytensor
import pytensor.tensor as pt
from scipy.stats import multivariate_normal
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.biosphere_prior import calibrate_biosphere_prior
from openghg_inversions.models.boundary_corrections import prepare_boundary_correction
from openghg_inversions.models.linked_shared_baseline import prepare_linked_shared_baseline
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2.co2_o2_fixed_ou import linked_fixed_ou_alignment
from openghg_inversions.rhime.co2.linked_shared_baseline_model import (
    build_shared_co2_o2_model,
    build_shared_co2_o2_cached_sigma_model,
    _linked_inputs,
    annotate_linked_shared_baseline_trace,
    reconstruct_linked_shared_baseline,
)
from openghg_inversions.rhime.co2.shared_baseline_model import build_shared_co2_model
from openghg_inversions.serialization import (
    encode_multiindexes_for_storage, restore_serialisation_multiindexes, save_trace, load_trace,
)
from test_boundary_correction_priors import operands
from test_co2_o2_fixed_ou import _kwargs, _prepared


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def review_case(mode="multiplicative", method="A"):
    """Unequal supports, native references, gated BCs and two fixed oceans."""
    base = _prepared()
    kwargs = _kwargs(base, interleaved=True)
    observations = kwargs["observations"]
    observations = observations.copy(data=np.where(
        observations.species.values == "co2", 400.0, 209300.0
    ) + np.array([0.013, 0.2, -0.014, -0.1, 0.025]))
    kwargs["observations"] = observations
    prior = base.retained_prior
    outer = prior.mean.copy(data=np.array([True, True, False, True, True]))
    sampled = prior.mean.copy(data=np.array([True, True, False, False, False]))
    raw, fixed, corrections = {}, {}, {}
    for channel, reference in (("co2", 400.0), ("o2", 209300.0)):
        h = getattr(base, f"{channel}_sensitivity")
        row = str(h.dims[0])
        native = select_gathered_data_array(
            observations, key=channel, key_dim="species", ragged_dim="channel_observation",
            stack_dim="observation",
        ).rename(observation=row).assign_attrs(units="ppm")
        raw[channel] = h.assign_coords(site=native.site, time=native.time).assign_attrs(units="ppm")
        fixed[channel] = xr.full_like(native, 0.0123).assign_attrs(units="ppm")
        bc = operands(native, reference)
        # Deliberately close to unit mass. Forming c*g and subtracting c loses
        # the tiny anomaly, whereas A*1+c*(g-1) preserves its native precision.
        mass = np.column_stack((np.full(native.size, 0.6), np.full(native.size, 0.4)))
        mass[:, 1] += np.arange(native.size) * 1e-12
        bc["H_bc_mass"] = bc.H_bc_mass.copy(data=mass)
        bc["H_bc_anomaly"] = bc.H_bc_anomaly.copy(
            data=np.arange(native.size * 2).reshape(native.size, 2) * 1e-10 + 1e-11
        )
        options = {} if method == "level" else {"pattern_sd": 0.37}
        corrections[channel] = prepare_boundary_correction(
            bc, native, method=method, level_sd=2.0, frequency="2h",
            anchor_time="2021-01-01", **options,
        )
    calibration = None
    if mode == "multiplicative_shrunk":
        weights = xr.DataArray(
            [[-2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0, 0.0]],
            dims=("process", prior.state_dim),
            coords={"process": ["GPP", "TER"], **dict(prior.mean.coords)}, attrs={"units": "kg/year"},
        )
        calibration = calibrate_biosphere_prior(
            prior, biosphere=sampled, weights=weights,
            reference=xr.DataArray(
                [-2.0, 3.0], dims="process", coords={"process": ["GPP", "TER"]}, attrs={"units": "kg/year"},
            ),
            residual_sd_multiplier=0.5,
        )
    prepared = prepare_linked_shared_baseline(
        co2_sensitivity=raw["co2"], o2_sensitivity=raw["o2"], retained_prior=prior,
        fixed_prior_contribution=fixed, boundary_correction=corrections,
        outer=outer, sampled_outer=sampled, mode=mode, delta_sd={"co2": 0.4, "o2": 0.003},
        calibrated_prior=calibration,
    )
    return prepared, kwargs, raw, fixed, corrections, outer, sampled


def _evaluate(model, expressions, point):
    return model.compile_fn(
        model.replace_rvs_by_values(expressions), inputs=model.value_vars, on_unused_input="ignore",
    )(point)


@pytest.mark.parametrize("mode", ["fixed", "multiplicative", "multiplicative_shrunk"])
def test_independent_mean_density_and_cached_state_gradients(mode):
    p, kwargs, raw, fixed, bc, outer, sampled = review_case(mode)
    data = {name: kwargs[name] for name in ("observations", "independent_error_sd", "aggregation_error")}
    ou = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    ordinary = build_shared_co2_o2_model(p, **data, **ou)
    cached = build_shared_co2_o2_cached_sigma_model(p, **data, **ou)
    point = ordinary.initial_point()
    for name, value in point.items():
        point[name] = np.linspace(0.00011, 0.00031, value.size).reshape(value.shape)
    cp = cached.model.initial_point()
    cp.update(point)
    assert sum(rv.name.startswith("flux_scaling") for rv in ordinary.free_RVs) == 1
    expressions = []
    for channel in ("co2", "o2"):
        h = raw[channel].values
        weights = np.array([0.5, 0.5]) if channel == "co2" else np.array([0.25, 0.5, 0.25])
        np.testing.assert_array_equal(p[channel].centering_weights, weights)
        state = ordinary.flux_scaling
        multi = mode.startswith("multiplicative")
        keep = (~outer | sampled).values if multi else (~outer).values
        inner = (~outer).values
        # The oracle starts from physical inputs, not the transformed model design.
        inner_h = h[:, inner]
        inner_state = state[np.flatnonzero(inner[keep])]
        mean = fixed[channel].values + inner_h @ inner_state
        mean = mean + h[:, outer.values] @ np.ones(outer.values.sum())
        if multi:
            selected = sampled.values
            centered = h[:, selected] - weights @ h[:, selected]
            mean = mean + centered @ (state[np.flatnonzero(selected[keep])] - 1.0)
        anomaly = bc[channel].bc_prescribed_centered.values
        mass = bc[channel].bc_mass_response.values
        reference = float(bc[channel].bc_reference)
        mean = mean + anomaly + reference * (mass - 1.0)
        design = (bc[channel].bc_pattern_operator * bc[channel].bc_pattern_sd).values
        centered = design - weights @ design
        mean = mean + ordinary[f"{channel}_bc_level"][0]
        mean = mean + centered @ ordinary[f"{channel}_bc_pattern"]
        expressions.append(mean)
    observation = data["observations"]
    # The joint order is deliberately interleaved. Reassemble from native labels.
    channel_rows = {name: np.flatnonzero(observation.species.values == name) for name in ("co2", "o2")}
    means = pt.zeros(observation.size)
    for channel, expression in zip(("co2", "o2"), expressions):
        means = pt.set_subtensor(means[channel_rows[channel]], expression)
    actual = _evaluate(ordinary, [ordinary.modelled_concentration, means], point)
    references = np.where(observation.species.values == "co2", 400.0, 209300.0)
    np.testing.assert_allclose(actual[0], actual[1] + references, rtol=0, atol=3e-11)
    amplitude = np.exp(point["ou_site_amplitude_log__"])
    covariance = data["aggregation_error"].covariance.values.copy()
    covariance += np.diag(data["independent_error_sd"].values ** 2)
    alignment = linked_fixed_ou_alignment(observation)
    hours = observation.time.values.astype("datetime64[m]").astype(float) / 60
    for group, sigma in enumerate(amplitude):
        rows = np.flatnonzero(alignment.site_index.values == group)
        covariance[np.ix_(rows, rows)] += sigma ** 2 * np.exp(
            -np.abs(hours[rows, None] - hours[None, rows]) / 5.0
        )
    assert np.any(covariance[np.ix_(channel_rows["co2"], channel_rows["o2"])] != 0)
    centered_observations = observation.values - references
    residual = pt.as_tensor_variable(centered_observations) - means
    oracle_logp = -0.5 * (
        observation.size * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1]
        + residual @ np.linalg.solve(covariance, np.eye(observation.size)) @ residual
    )
    likelihood = ordinary.logp(vars=ordinary.observed_RVs)
    oracle_logp = ordinary.replace_rvs_by_values([oracle_logp])[0]
    state_values = [ordinary.rvs_to_values[ordinary[rv.name]] for rv in cached.states]
    ordinary_out = ordinary.compile_fn(
        [likelihood, *pt.grad(likelihood, state_values)], inputs=ordinary.value_vars,
        on_unused_input="ignore",
    )(point)
    expected = ordinary.compile_fn(
        [oracle_logp, *pt.grad(oracle_logp, state_values)], inputs=ordinary.value_vars,
        on_unused_input="ignore",
    )(point)
    np.testing.assert_allclose(expected[0], multivariate_normal.logpdf(
        centered_observations - actual[1], cov=covariance
    ), atol=1e-10)
    cached.shared_cache.update(cached.target.refresh(amplitude))
    cached_logp = cached.model.replace_rvs_by_values([cached.cached_state_logp])[0]
    cached_values = [cached.model.rvs_to_values[rv] for rv in cached.states]
    cached_out = cached.model.compile_fn(
        [cached_logp, *pt.grad(cached_logp, cached_values)], inputs=cached.model.value_vars,
        on_unused_input="ignore",
    )(cp)
    for actuals in (ordinary_out, cached_out):
        for actual, desired in zip(actuals, expected):
            np.testing.assert_allclose(actual, desired, rtol=1e-10, atol=2e-8)


@pytest.mark.parametrize("builder", [build_shared_co2_o2_model, build_shared_co2_o2_cached_sigma_model])
def test_valid_level_only_boundary_does_not_require_pattern_state(builder):
    p, kwargs, *_ = review_case(method="level")
    model = builder(
        p, **{name: kwargs[name] for name in ("observations", "independent_error_sd", "aggregation_error")},
        tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
    )
    graph = getattr(model, "model", model)
    assert "co2_bc_pattern" not in graph and "o2_bc_pattern" not in graph


def test_gross_biosphere_direction_is_visible_with_different_oxidative_ratios():
    """Equal GPP/TER changes cancel in CO2 but remain visible to O2."""
    co2 = np.array([[-1.0, 1.0], [-2.0, 2.0]])
    oxygen = co2 * np.array([-1.1, -0.9])
    gross = np.array([1.0, 1.0])
    np.testing.assert_array_equal(co2 @ gross, 0.0)
    assert np.linalg.norm(oxygen @ gross) > 0.1
    assert np.linalg.matrix_rank(np.vstack((co2, oxygen))) == 2
    with pytest.raises(ValueError, match="GPP/TER"):
        review_case(mode="additive")


def test_native_o2_anomaly_survives_reference_and_nonunit_mass():
    prepared, kwargs, _, _, raw, *_ = review_case()
    _, _, corrections, *_ = _linked_inputs(prepared, kwargs["observations"])
    for channel in ("co2", "o2"):
        expected = raw[channel].bc_prescribed_centered + raw[channel].bc_reference * (
            raw[channel].bc_mass_response - 1.0
        )
        np.testing.assert_array_equal(corrections[channel].bc_prescribed_centered, expected)
        assert float(corrections[channel].bc_level_sd) == {"co2": 0.4, "o2": 0.003}[channel]
    with pytest.raises(ValueError, match="native centered"):
        _linked_inputs(
            {name: p.drop_vars("bc_prescribed_centered") for name, p in prepared.items()},
            kwargs["observations"],
        )


def test_serialized_subset_uses_the_frozen_full_support_operator(tmp_path):
    prepared, kwargs, *_ = review_case()
    noise = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    data = {name: kwargs[name] for name in ("observations", "independent_error_sd", "aggregation_error")}
    full = build_shared_co2_o2_cached_sigma_model(prepared, **data, **noise)
    loaded = {}
    for channel, p in prepared.items():
        path = tmp_path / f"{channel}.nc"
        encode_multiindexes_for_storage(p).to_netcdf(path)
        with xr.open_dataset(path) as opened:
            loaded[channel] = restore_serialisation_multiindexes(opened.load())
        xr.testing.assert_equal(loaded[channel], p)
    restored = build_shared_co2_o2_cached_sigma_model(loaded, **data, **noise)
    np.testing.assert_array_equal(restored.target.fixed_contribution, full.target.fixed_contribution)
    np.testing.assert_array_equal(restored.target.design, full.target.design)
    # Drop the last O2 row, leaving its original full-window weights and means.
    rows = np.arange(4)
    obs = data["observations"].isel(observation=rows)
    subset = {}
    for channel, p in loaded.items():
        native = select_gathered_data_array(
            obs, key=channel, key_dim="species", ragged_dim="channel_observation", stack_dim="observation"
        )
        row = p.attrs["native_observation_dim"]
        subset[channel] = p.sel({row: native.observation.values})
    from dataclasses import replace

    covariance = data["aggregation_error"].covariance
    aggregation = replace(
        data["aggregation_error"],
        covariance=covariance.isel({covariance.dims[0]: rows, covariance.dims[1]: rows}),
        marginal_variance=data["aggregation_error"].marginal_variance[rows],
    )
    rebuilt = build_shared_co2_o2_cached_sigma_model(
        subset, observations=obs, independent_error_sd=data["independent_error_sd"].isel(observation=rows),
        aggregation_error=aggregation, **noise,
    )
    np.testing.assert_array_equal(rebuilt.target.fixed_contribution, full.target.fixed_contribution[rows])
    np.testing.assert_array_equal(rebuilt.target.design, full.target.design[rows])


def test_shared_dask_transport_graph_executes_once_for_both_channels():
    import dask.array as da
    from dask import delayed

    prepared, kwargs, *_ = review_case()
    calls = []

    @delayed
    def shared_payload():
        calls.append(1)
        return 1.0

    shared = da.from_delayed(shared_payload(), shape=(), dtype=float)
    lazy = {
        channel: p.assign(flux_sensitivity=p.flux_sensitivity.copy(data=p.flux_sensitivity.data * shared))
        for channel, p in prepared.items()
    }
    _linked_inputs(lazy, kwargs["observations"])
    assert calls == [1]
    assert all(isinstance(p.flux_sensitivity.data, da.Array) for p in lazy.values())


def test_inner_oceans_remain_in_the_joint_sampled_flux_state():
    _, kwargs, raw, fixed, corrections, outer, sampled = review_case()
    prior = kwargs["retained_prior"]
    inner_oceans = outer.copy(data=np.array([True, True, False, False, False]))
    frozen = prepare_linked_shared_baseline(
        co2_sensitivity=raw["co2"], o2_sensitivity=raw["o2"], retained_prior=prior,
        fixed_prior_contribution=fixed, boundary_correction=corrections,
        outer=inner_oceans, sampled_outer=sampled, mode="fixed", delta_sd={"co2": 0.4, "o2": 0.003},
    )
    for channel in ("co2", "o2"):
        np.testing.assert_array_equal(frozen[channel].retained_mean.state, prior.mean.state.values[2:])
        np.testing.assert_array_equal(frozen[channel].flux_sensitivity, raw[channel].values[:, 2:])


def test_actual_restored_trace_preserves_pattern_identity_and_reconstruction(tmp_path):
    prepared, kwargs, *_ = review_case()
    data = {name: kwargs[name] for name in ("observations", "independent_error_sd", "aggregation_error")}
    model = build_shared_co2_o2_model(prepared, **data)
    point = model.initial_point()
    for name, value in point.items():
        point[name] = np.linspace(0.0001, 0.0007, value.size).reshape(value.shape)
    # A real PyMC backend records the exact nonzero point and all deterministics;
    # sampling randomness is unnecessary for exercising its persisted labels.
    backend = pm.backends.ndarray.NDArray(model=model)
    backend.setup(draws=1, chain=0)
    backend.record(point, in_warmup=False)
    backend.close()
    trace = pm.to_inference_data(trace=pm.backends.base.MultiTrace([backend]), model=model)
    trace = restore_inferencedata_coords(trace, get_coord_registry(model))
    trace = annotate_linked_shared_baseline_trace(trace, model)
    path = tmp_path / "actual-trace.nc"
    save_trace(trace, path)
    reopened = load_trace(path)
    frozen = {channel: reopened[f"shared_baseline_{channel}"].to_dataset() for channel in ("co2", "o2")}
    posterior = reopened["posterior"].to_dataset()
    reconstructed = reconstruct_linked_shared_baseline(frozen, posterior)
    for channel in ("co2", "o2"):
        mask = kwargs["observations"].species.values == channel
        expected = posterior[f"{channel}_modelled_concentration"].isel(observation=mask).values.squeeze()
        np.testing.assert_allclose(reconstructed[channel].modelled_concentration.values.squeeze(), expected, atol=3e-11)
    # A saved posterior with reversed labelled pattern support must be selected
    # by identity or rejected, never assigned the original labels by position.
    pattern = posterior.co2_bc_pattern
    dim = str(pattern.dims[-1])
    wrong = posterior.drop_dims(dim)
    wrong["co2_bc_pattern"] = pattern.isel({dim: slice(None, None, -1)}).assign_coords(
        {dim: [f"wrong:{i}" for i in range(pattern.sizes[dim])]}
    )
    with pytest.raises((ValueError, KeyError), match="pattern|label|align|coordinate"):
        reconstruct_linked_shared_baseline(frozen, wrong)


def test_co2_channel_marginal_matches_existing_co2_shared_recipe():
    """Dropping the O2 observations gives the existing CO2 mean and target."""
    prepared, kwargs, *_ = review_case()
    noise = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    data = {name: kwargs[name] for name in ("observations", "independent_error_sd", "aggregation_error")}
    linked = build_shared_co2_o2_model(prepared, **data, **noise)
    selected = np.flatnonzero(data["observations"].species.values == "co2")
    native = select_gathered_data_array(
        data["observations"], key="co2", key_dim="species", ragged_dim="channel_observation",
        stack_dim="observation",
    ).rename(observation="nmeasure").assign_attrs(units="ppm")
    error = data["independent_error_sd"].isel(observation=selected).values
    covariance = data["aggregation_error"].covariance.values[np.ix_(selected, selected)]
    dense = xr.DataArray(
        covariance, dims=("nmeasure", "nmeasure_cov"),
        coords={"nmeasure": native.nmeasure, "nmeasure_cov": native.nmeasure.values},
    )
    aggregation = resolve_aggregation_error(
        xr.Dataset({"mf": native, "aggregation_error_covariance": dense}), "dense"
    )
    co2 = build_shared_co2_model(
        prepared["co2"].rename(co2_measure="nmeasure"), observations=native,
        observation_error=native.copy(data=error), aggregation_error=aggregation,
        likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=noise,
    )
    linked_point = linked.initial_point()
    for name, value in linked_point.items():
        linked_point[name] = np.linspace(0.001, 0.007, value.size).reshape(value.shape)
    co2_point = co2.initial_point()
    for target, source in (("flux_scaling_latent", "flux_scaling_latent"),
                           ("bc_level", "co2_bc_level"), ("bc_pattern", "co2_bc_pattern")):
        co2_point[target] = linked_point[source]
    co2_point["ou_site_amplitude_log__"] = linked_point["ou_site_amplitude_log__"][:1]
    # The marginal covariance is the CO2 block of the complete linked target;
    # cross-channel aggregation covariance disappears only with marginalization.
    hours = native.time.values.astype("datetime64[m]").astype(float) / 60
    sigma = np.exp(co2_point["ou_site_amplitude_log__"])[0]
    marginal_covariance = covariance + np.diag(error ** 2) + sigma ** 2 * np.exp(
        -np.abs(hours[:, None] - hours[None, :]) / 5.0
    )
    linked_likelihood = pm.logp(
        pm.MvNormal.dist(mu=linked.co2_modelled_concentration[selected], cov=marginal_covariance),
        native.values,
    )
    linked_likelihood = linked.replace_rvs_by_values([linked_likelihood])[0]
    co2_likelihood = co2.logp(vars=co2.observed_RVs)
    linked_values = [linked.rvs_to_values[linked[name]] for name in (
        "flux_scaling_latent", "co2_bc_level", "co2_bc_pattern"
    )]
    co2_values = [co2.rvs_to_values[co2[name]] for name in ("flux_scaling_latent", "bc_level", "bc_pattern")]
    actual = linked.compile_fn(
        [linked_likelihood, *pt.grad(linked_likelihood, linked_values)], inputs=linked.value_vars,
        on_unused_input="ignore",
    )(linked_point)
    expected = co2.compile_fn(
        [co2_likelihood, *pt.grad(co2_likelihood, co2_values)], inputs=co2.value_vars,
        on_unused_input="ignore",
    )(co2_point)
    for value, target in zip(actual, expected):
        np.testing.assert_allclose(value, target, rtol=1e-10, atol=2e-10)
