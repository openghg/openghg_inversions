"""Independent shared-baseline equations and actual cached sampler oracle."""

from dataclasses import replace

import numpy as np
import pymc as pm
import pytest
import pytensor
import pytensor.tensor as pt
from scipy.stats import multivariate_normal
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.biosphere_prior import calibrate_biosphere_prior
from openghg_inversions.models.boundary import AdditiveChoice, AnomalyScalingChoice
from openghg_inversions.models.linked_shared_baseline import bind_linked_shared_baseline
from boundary_test_utils import centered_transport
from openghg_inversions.models.shared_baseline import SharedBaselineChoice
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
from openghg_inversions.rhime.co2.shared_baseline_model import build_shared_co2_model, build_shared_co2_cached_sigma_model
from openghg_inversions.serialization import save_trace, load_trace
from test_co2_o2_fixed_ou import _kwargs, _prepared

DELTA_SD = {"co2": 0.4, "o2": 0.003}
PERIODS = dict(frequency="2h", anchor_time="2021-01-01")
BOUNDARIES = {
    "level": None,
    "A": AnomalyScalingChoice(None, 0.37, **PERIODS),
    "curtain_contrasts": AdditiveChoice(None, 0.37, anomalies="period_contrasts", **PERIODS),
}


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64", cast_policy="numpy+floatX"):
        yield


def review_case(mode="multiplicative", method="A", *, partition_size=None, boundary_inputs=None):
    """Unequal supports, native references, gated BCs and two fixed oceans.

    Returns the bound baselines, joint model inputs, raw native sensitivities,
    fixed contributions, independent native boundary terms, outer selectors,
    and a rebinding function for the same native inputs.
    """
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
    boundary = BOUNDARIES[method]
    raw, fixed, natives, transports, corrections = {}, {}, {}, {}, {}
    for channel, reference in (("co2", 400.0), ("o2", 209300.0)):
        h = getattr(base, f"{channel}_sensitivity")
        row = str(h.dims[0])
        native = select_gathered_data_array(
            observations, key=channel, key_dim="species", ragged_dim="channel_observation",
            stack_dim="observation",
        ).rename(observation=row).assign_attrs(units="ppm")
        natives[channel] = native
        raw[channel] = h.assign_coords(site=native.site, time=native.time).assign_attrs(units="ppm")
        fixed[channel] = xr.full_like(native, 0.0123).assign_attrs(units="ppm")
        # Normalized surface transport retains tiny native O2 anomalies.
        coords = {"nmeasure": native[row].values, "bc_region": ["N", "S"]}
        mass = np.column_stack((np.full(native.size, 0.6), np.full(native.size, 0.4)))
        anomaly = np.arange(native.size * 2).reshape(native.size, 2) * 1e-10 + 1e-11
        if partition_size is not None:
            # Normalize native transport once, then project an ordinary complete
            # partition. Summed columns retain natural float64 roundoff.
            native_weights = np.random.default_rng(7).uniform(0.1, 2, (3, 72))[:native.size]
            native_weights /= native_weights.sum(axis=1, keepdims=True)
            regions = np.arange(72) % partition_size
            mass = np.stack([native_weights[:, regions == j].sum(axis=1)
                             for j in range(partition_size)], axis=1)
            coords["bc_region"] = np.arange(partition_size)
            anomaly = mass * np.linspace(-1e-9, 2e-9, partition_size)
        transport = centered_transport(
            xr.DataArray(anomaly, dims=("nmeasure", "bc_region"), coords=coords, attrs={"units": "ppm"}),
            xr.DataArray(mass, dims=("nmeasure", "bc_region"), coords=coords, attrs={"units": "1"}),
            xr.DataArray(reference, attrs={"units": "ppm"}),
        )
        transports[channel] = transport
        if boundary_inputs is not None:
            boundary_inputs[channel] = transport
        # Independent native boundary terms: the prescribed anomaly sum and the
        # family's own standard-normal anomaly design on the native rows.
        rows = native.rename({row: "nmeasure"})
        factor = None if boundary is None else boundary.bind(rows, transport).anomaly_factor()
        corrections[channel] = xr.Dataset({
            "bc_prescribed_centered": transport.H_bc_anomaly.sum("bc_region"),
            "bc_pattern_design": (
                xr.DataArray(np.empty((native.size, 0)), dims=("nmeasure", "bc_pattern_term"))
                if factor is None else factor.transpose("nmeasure", ...).drop_vars(
                    [name for name in factor.coords if name not in ("nmeasure",)]
                ).rename({factor.dims[-1]: "bc_pattern_term"})
            ),
        })
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

    def rebind(mode=mode, **overrides):
        choices = {channel: SharedBaselineChoice(boundary, sd, mode) for channel, sd in DELTA_SD.items()}
        inputs = dict(
            co2_sensitivity=raw["co2"], o2_sensitivity=raw["o2"], retained_prior=prior,
            fixed_prior_contribution=fixed, boundary_transport=transports,
            outer=outer, sampled_outer=sampled, calibrated_prior=calibration,
        )
        return bind_linked_shared_baseline(choices, natives, **{**inputs, **overrides})

    prepared = rebind()
    return prepared, kwargs, raw, fixed, corrections, outer, sampled, rebind


def _evaluate(model, expressions, point):
    return model.compile_fn(
        model.replace_rvs_by_values(expressions), inputs=model.value_vars, on_unused_input="ignore",
    )(point)


@pytest.mark.parametrize("mode", ["fixed", "multiplicative", "multiplicative_shrunk"])
def test_independent_mean_density_and_cached_state_gradients(mode, partition_size=None):
    native_bc = {}
    baseline, kwargs, raw, fixed, bc, outer, sampled, _ = review_case(
        mode, partition_size=partition_size, boundary_inputs=native_bc,
    )
    p = {channel: bound.data for channel, bound in baseline.items()}
    data = {name: kwargs[name] for name in ("observations", "independent_error_sd", "aggregation_error")}
    ou = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    ordinary = build_shared_co2_o2_model(baseline, **data, **ou)
    cached = build_shared_co2_o2_cached_sigma_model(baseline, **data, **ou)
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
        mean = mean + anomaly
        if partition_size is not None:
            mean = mean + float(native_bc[channel].bc_reference) * (native_bc[channel].H_bc_mass.sum("bc_region").values - 1)
        design = bc[channel].bc_pattern_design.values
        centered = design - weights @ design
        mean = mean + ordinary[f"{channel}_bc_mean_shift"]
        mean = mean + centered @ ordinary[f"{channel}_bc_pattern_standardized"]
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


@pytest.mark.parametrize("mode", ["fixed", "multiplicative", "multiplicative_shrunk"])
def test_independent_whole_target_all_value_and_actual_cached_state_gradients(mode):
    from pymc.blocking import DictToArrayBijection
    from test_co2_cached_sigma import _compound

    baseline, kwargs, raw, fixed, corrections, outer, sampled, _ = review_case(mode)
    prepared = {channel: bound.data for channel, bound in baseline.items()}
    inputs = {name: kwargs[name] for name in ("observations", "independent_error_sd", "aggregation_error")}
    noise = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    ordinary = build_shared_co2_o2_model(baseline, **inputs, **noise)
    cached = build_shared_co2_o2_cached_sigma_model(baseline, **inputs, **noise)
    point = ordinary.initial_point()
    for name, value in point.items():
        scale = 0.001 if name == "o2_bc_mean_shift" else 1.0
        point[name] = np.asarray(scale * np.linspace(0.031, 0.137, value.size).reshape(value.shape))

    frozen = prepared["co2"]
    dim = frozen.attrs["state_dim"]
    mean = frozen.retained_mean.values
    covariance = np.log1p(frozen.retained_covariance.values / np.outer(mean, mean))
    location = np.log(mean) - np.diag(covariance) / 2
    cholesky = np.linalg.cholesky(covariance)
    original_mean = kwargs["retained_prior"].mean
    original_index = original_mean.get_index(dim)
    selected_index = frozen.retained_mean.get_index(dim)
    inner_positions = selected_index.get_indexer(original_index[~outer.values])
    sampled_positions = selected_index.get_indexer(original_index[sampled.values])
    assert (inner_positions >= 0).all()
    if mode.startswith("multiplicative"):
        assert (sampled_positions >= 0).all()

    observation = inputs["observations"]
    species = observation.species.values
    groups = list(dict.fromkeys(zip(species, observation.site.values)))
    group_index = np.array([groups.index(pair) for pair in zip(species, observation.site.values)])
    hours = (observation.time.values - observation.time.values[0]) / np.timedelta64(1, "h")
    correlation = np.exp(-abs(hours[:, None] - hours[None, :]) / 5.0)
    correlation *= group_index[:, None] == group_index[None, :]
    base_covariance = inputs["aggregation_error"].covariance.values.copy()
    base_covariance += np.diag(inputs["independent_error_sd"].values ** 2)
    references = np.where(species == "co2", 400.0, 209300.0)
    observations = observation.values - references
    normal_constant = np.log(2 * np.pi)
    native = {}
    for channel, weights in (("co2", np.array([0.5, 0.5])), ("o2", np.array([0.25, 0.5, 0.25]))):
        rows = np.flatnonzero(species == channel)
        # Resolve native rows by their explicit labels, independently of model gathering.
        row = raw[channel].dims[0]
        positions = raw[channel].get_index(row).get_indexer(observation.channel_observation.values[rows])
        assert (positions >= 0).all()
        h = raw[channel].values
        patterns = corrections[channel].bc_pattern_design.values
        native[channel] = (
            rows, positions, h, weights, patterns,
            float(prepared[channel].delta_sd), corrections[channel].bc_prescribed_centered.values,
        )

    def density(values):
        flux = np.exp(location + cholesky @ values["flux_scaling_latent"])
        small_mean = np.empty(observation.size)
        for channel, (rows, positions, h, weights, patterns, _, prescribed) in native.items():
            contribution = fixed[channel].values + h[:, ~outer.values] @ flux[inner_positions]
            contribution += h[:, outer.values] @ original_mean.values[outer.values]
            if mode.startswith("multiplicative"):
                sampled_h = h[:, sampled.values]
                contribution += (sampled_h - weights @ sampled_h) @ (
                    flux[sampled_positions] - mean[sampled_positions]
                )
            contribution += prescribed + values[f"{channel}_bc_mean_shift"]
            contribution += (patterns - weights @ patterns) @ values[f"{channel}_bc_pattern_standardized"]
            small_mean[rows] = contribution[positions]
        eta = values["ou_site_amplitude_log__"]
        amplitude = np.exp(eta)
        row_amplitude = amplitude[group_index]
        covariance = base_covariance + np.outer(row_amplitude, row_amplitude) * correlation
        residual = observations - small_mean
        result = -0.5 * (
            observation.size * normal_constant + np.linalg.slogdet(covariance)[1]
            + residual @ np.linalg.solve(covariance, residual)
        )
        for name in ("flux_scaling_latent", "co2_bc_pattern_standardized", "o2_bc_pattern_standardized"):
            z = values[name]
            result -= 0.5 * (z.size * normal_constant + z @ z)
        for channel, (_, _, _, _, _, sd, _) in native.items():
            result -= np.log(sd) + 0.5 * (normal_constant + (values[f"{channel}_bc_mean_shift"] / sd) ** 2)
        result += np.sum(np.log(2) - np.log(0.75) - 0.5 * normal_constant - 0.5 * (amplitude / 0.75) ** 2 + eta)
        return float(result)

    expected = density(point)
    gradients = {}
    for name, value in point.items():
        gradient = np.empty(value.size)
        for i in range(value.size):
            plus = {key: item.copy() for key, item in point.items()}
            minus = {key: item.copy() for key, item in point.items()}
            plus[name].flat[i] += 1e-5
            minus[name].flat[i] -= 1e-5
            gradient[i] = (density(plus) - density(minus)) / 2e-5
        gradients[name] = gradient
    for model in (ordinary, cached.model):
        expected_gradient = np.concatenate([gradients[value.name] for value in model.value_vars])
        actual = model.compile_logp()(point)
        actual_gradient = model.compile_dlogp()(point)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-7, atol=2e-7)
        print(f"{mode} full target: density error={abs(actual - expected):.3g}, "
              f"all-value gradient max error={np.max(abs(actual_gradient - expected_gradient)):.3g}")

    cached.shared_cache.update(cached.target.refresh(np.exp(point["ou_site_amplitude_log__"])))
    state_step = _compound(cached).methods[1]
    state_step._logp_dlogp_func.set_extra_values(point)
    state_point = DictToArrayBijection.map({name: point[name] for name in state_step.var_names})
    actual, actual_gradient = state_step._logp_dlogp_func(state_point)
    expected_gradient = np.concatenate([gradients[name] for name in state_step.var_names])
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-7, atol=2e-7)
    print(f"{mode} actual cached state: density error={abs(actual - expected):.3g}, "
          f"state gradient max error={np.max(abs(actual_gradient - expected_gradient)):.3g}")


@pytest.mark.parametrize("builder", [build_shared_co2_o2_model, build_shared_co2_o2_cached_sigma_model])
def test_valid_level_only_boundary_does_not_require_pattern_state(builder):
    p, kwargs, *_ = review_case(method="level")
    model = builder(
        p, **{name: kwargs[name] for name in ("observations", "independent_error_sd", "aggregation_error")},
        tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75},
    )
    graph = getattr(model, "model", model)
    assert "co2_bc_pattern_standardized" not in graph and "o2_bc_pattern_standardized" not in graph


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


def test_native_o2_anomaly_survives_scalar_reference():
    prepared, kwargs, _, _, raw, *_ = review_case()
    bound, *_ = _linked_inputs(
        prepared, kwargs["observations"], kwargs["independent_error_sd"], aggregation_error=kwargs["aggregation_error"],
    )
    for channel in ("co2", "o2"):
        expected = raw[channel].bc_prescribed_centered
        np.testing.assert_array_equal(bound[channel].bc_prescribed_centered, expected)
        assert float(bound[channel].delta_sd) == {"co2": 0.4, "o2": 0.003}[channel]
    with pytest.raises(ValueError, match="native centered"):
        _linked_inputs(
            {name: replace(p, data=p.data.drop_vars("bc_prescribed_centered")) for name, p in prepared.items()},
            kwargs["observations"], kwargs["independent_error_sd"],
            aggregation_error=kwargs["aggregation_error"],
        )


def test_bound_subset_uses_the_full_support_operator():
    prepared, kwargs, *_ = review_case()
    noise = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    data = {name: kwargs[name] for name in ("observations", "independent_error_sd", "aggregation_error")}
    full = build_shared_co2_o2_cached_sigma_model(prepared, **data, **noise)
    # Drop the last O2 row, leaving its original full-window weights and means.
    rows = np.arange(4)
    obs = data["observations"].isel(observation=rows)
    subset = {}
    for channel, bound in prepared.items():
        native = select_gathered_data_array(
            obs, key=channel, key_dim="species", ragged_dim="channel_observation", stack_dim="observation"
        )
        row = bound.data.attrs["native_observation_dim"]
        subset[channel] = replace(bound, data=bound.data.sel({row: native.observation.values}))
    assert subset["o2"].data.sizes["o2_measure"] < prepared["o2"].data.sizes["o2_measure"]
    xr.testing.assert_equal(subset["o2"].data.bc_pattern_mean, prepared["o2"].data.bc_pattern_mean)

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
        channel: replace(p, data=p.data.assign(
            flux_sensitivity=p.data.flux_sensitivity.copy(data=p.data.flux_sensitivity.data * shared),
        ))
        for channel, p in prepared.items()
    }
    _linked_inputs(lazy, kwargs["observations"], kwargs["independent_error_sd"], aggregation_error=kwargs["aggregation_error"])
    assert calls == [1]
    assert all(isinstance(p.data.flux_sensitivity.data, da.Array) for p in lazy.values())


def test_inner_oceans_remain_in_the_joint_sampled_flux_state():
    _, kwargs, raw, _, _, outer, sampled, rebind = review_case()
    prior = kwargs["retained_prior"]
    inner_oceans = outer.copy(data=np.array([True, True, False, False, False]))
    bound = rebind("fixed", outer=inner_oceans, sampled_outer=sampled)
    for channel in ("co2", "o2"):
        np.testing.assert_array_equal(bound[channel].data.retained_mean.state, prior.mean.state.values[2:])
        np.testing.assert_array_equal(bound[channel].data.flux_sensitivity, raw[channel].values[:, 2:])


@pytest.mark.parametrize("method", ["A", "curtain_contrasts"])
def test_actual_restored_trace_preserves_pattern_identity_and_reconstruction(tmp_path, method):
    prepared, kwargs, _, _, corrections, *_ = review_case(method=method)
    for channel, original in corrections.items():
        assert original.sizes["bc_pattern_term"] == prepared[channel].data.sizes["bc_pattern_term"] > 0
        assert "method" not in prepared[channel].data.bc_pattern_term.attrs
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
    trace = annotate_linked_shared_baseline_trace(trace, prepared)
    trace = annotate_linked_shared_baseline_trace(trace, prepared)
    for channel in ("co2", "o2"):
        coefficient = trace.posterior[f"{channel}_bc_pattern_standardized"]
        assert coefficient.attrs["units"] == "1"
        assert "method" not in coefficient.coords[coefficient.dims[-1]].attrs
        assert f"{channel}_boundary_correction_schema" not in trace.attrs
    path = tmp_path / "actual-trace.nc"
    save_trace(trace, path)
    reopened = load_trace(path)
    frozen = {channel: reopened[f"shared_baseline_{channel}"].to_dataset() for channel in ("co2", "o2")}
    posterior = reopened["posterior"].to_dataset()
    reconstructed = reconstruct_linked_shared_baseline(frozen, posterior)
    for channel in ("co2", "o2"):
        mask = kwargs["observations"].species.values == channel
        expected = posterior["modelled_concentration"].isel(observation=mask).values.squeeze()
        np.testing.assert_allclose(reconstructed[channel].modelled_concentration.values.squeeze(), expected, atol=3e-11)
    # A saved posterior with reversed labelled pattern support must be selected
    # by identity or rejected, never assigned the original labels by position.
    pattern = posterior.co2_bc_pattern_standardized
    dim = str(pattern.dims[-1])
    wrong = posterior.drop_dims(dim)
    wrong["co2_bc_pattern_standardized"] = pattern.isel({dim: slice(None, None, -1)}).assign_coords(
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
        replace(prepared["co2"], data=prepared["co2"].data.rename(co2_measure="nmeasure")), observations=native,
        observation_error=native.copy(data=error), aggregation_error=aggregation,
        likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=noise,
    )
    linked_point = linked.initial_point()
    for name, value in linked_point.items():
        linked_point[name] = np.linspace(0.001, 0.007, value.size).reshape(value.shape)
    co2_point = co2.initial_point()
    for target, source in (("flux_scaling_latent", "flux_scaling_latent"),
                           ("bc_mean_shift", "co2_bc_mean_shift"), ("bc_pattern_standardized", "co2_bc_pattern_standardized")):
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
        pm.MvNormal.dist(mu=linked.modelled_concentration[selected], cov=marginal_covariance),
        native.values,
    )
    linked_likelihood = linked.replace_rvs_by_values([linked_likelihood])[0]
    co2_likelihood = co2.logp(vars=co2.observed_RVs)
    linked_values = [linked.rvs_to_values[linked[name]] for name in (
        "flux_scaling_latent", "co2_bc_mean_shift", "co2_bc_pattern_standardized"
    )]
    co2_values = [co2.rvs_to_values[co2[name]] for name in ("flux_scaling_latent", "bc_mean_shift", "bc_pattern_standardized")]
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


@pytest.mark.parametrize("partition_size", [4, 12])
def test_normalized_native_partition_roundoff_preserves_o2_oracle(partition_size):
    native = {}
    review_case(partition_size=partition_size, boundary_inputs=native)
    roundoff = native["o2"].H_bc_mass.sum("bc_region").values - 1
    assert np.any(roundoff != 0)
    assert np.max(abs(roundoff)) <= np.finfo(float).eps
    # At 209300 ppm this native partition has a 2.324e-11 ppm closure residual.
    # Retain the existing mean/density/full-gradient tolerances in the oracle.
    assert 0 < np.max(abs(209300 * roundoff)) < 3e-11
    test_independent_mean_density_and_cached_state_gradients("multiplicative", partition_size)


@pytest.mark.parametrize("linked,cached", [(False, False), (False, True), (True, False), (True, True)])
def test_shared_public_builders_reject_float32_before_materialization(linked, cached):
    prepared, _, *_ = review_case()
    import dask.array as da
    from dask import delayed

    @delayed
    def forbidden():
        pytest.fail("Float32 graph executed a payload")

    prepared = {
        channel: replace(bound, data=bound.data.assign(flux_sensitivity=bound.data.flux_sensitivity.copy(
            data=da.from_delayed(forbidden(), shape=bound.data.flux_sensitivity.shape, dtype=float),
        )))
        for channel, bound in prepared.items()
    }
    if linked:
        builder = build_shared_co2_o2_cached_sigma_model if cached else build_shared_co2_o2_model
        inputs = dict(observations=None, independent_error_sd=None, aggregation_error=None)
    else:
        builder = build_shared_co2_cached_sigma_model if cached else build_shared_co2_model
        prepared = prepared["co2"]
        inputs = dict(observations=None, observation_error=None, aggregation_error=None)
    if cached:
        inputs.update(tau_hours=5., site_amplitude_prior={"pdf": "halfnormal", "sigma": .75})
    with pytensor.config.change_flags(floatX="float32"):
        with pytest.raises(ValueError, match="floatX=float64"):
            builder(prepared, **inputs)


@pytest.mark.parametrize("cached", [False, True])
def test_shared_transport_and_independent_error_materialize_once(cached):
    import dask.array as da
    from dask import delayed

    prepared, kwargs, *_ = review_case()
    calls = []

    @delayed
    def payload():
        calls.append(1)
        return 1.0

    shared = da.from_delayed(payload(), shape=(), dtype=float)
    prepared = {
        channel: replace(bound, data=bound.data.assign(flux_sensitivity=bound.data.flux_sensitivity * shared))
        for channel, bound in prepared.items()
    }
    inputs = {name: kwargs[name] for name in (
        "observations", "independent_error_sd", "aggregation_error",
    )}
    inputs["independent_error_sd"] = inputs["independent_error_sd"] * shared
    inputs["observations"] = inputs["observations"] * shared
    aggregation = inputs["aggregation_error"]
    inputs["aggregation_error"] = replace(aggregation, covariance=aggregation.covariance * shared)
    builder = build_shared_co2_o2_cached_sigma_model if cached else build_shared_co2_o2_model
    builder(prepared, **inputs, tau_hours=5., site_amplitude_prior={"pdf": "halfnormal", "sigma": xr.DataArray(.75 * shared)})
    assert calls == [1]


@pytest.mark.parametrize("cached", [False, True])
def test_observation_error_auxiliary_coordinates_stay_lazy(cached):
    import dask.array as da
    from dask import delayed

    prepared, kwargs, *_ = review_case()

    @delayed
    def forbidden():
        pytest.fail("Unused auxiliary coordinate executed")

    inputs = {name: kwargs[name] for name in ("observations", "independent_error_sd", "aggregation_error")}
    for name in ("observations", "independent_error_sd"):
        inputs[name] = inputs[name].assign_coords(unused=("observation", da.from_delayed(
            forbidden(), shape=(inputs[name].size,), dtype=float,
        )))
    for channel, bound in prepared.items():
        row = bound.data.attrs["native_observation_dim"]
        prepared[channel] = replace(bound, data=bound.data.assign_coords(
            unused_native=(row, da.from_delayed(forbidden(), shape=(bound.data.sizes[row],), dtype=float)),
        ))
    inputs["independent_error_sd"] = inputs["independent_error_sd"].assign_coords(
        observation_units=("observation", da.from_delayed(forbidden(), shape=(inputs["observations"].size,), dtype=str)),
    )
    builder = build_shared_co2_o2_cached_sigma_model if cached else build_shared_co2_o2_model
    builder(prepared, **inputs, tau_hours=5., site_amplitude_prior={"pdf": "halfnormal", "sigma": .75})
    assert all(isinstance(inputs[name].unused.data, da.Array) for name in ("observations", "independent_error_sd"))


def test_zero_centered_patterns_keep_raw_uncertainty_and_canonical_levels():
    from test_linked_shared_baseline import _assert_linked_density_and_gradient_parity

    baseline, kwargs, *_ = review_case()
    for channel, bound in baseline.items():
        frozen = bound.data.copy()
        row = frozen.attrs["native_observation_dim"]
        raw = xr.ones_like(frozen.bc_pattern_raw) * 0.25
        frozen["bc_pattern_raw"] = raw
        frozen["bc_pattern_mean"] = raw.isel({row: 0}, drop=True)
        baseline[channel] = replace(bound, data=frozen)
    prepared = {channel: bound.data for channel, bound in baseline.items()}
    inputs = {name: kwargs[name] for name in ("observations", "independent_error_sd", "aggregation_error")}
    noise = dict(tau_hours=5., site_amplitude_prior={"pdf": "halfnormal", "sigma": .75})
    ordinary = build_shared_co2_o2_model(baseline, **inputs, **noise)
    cached = build_shared_co2_o2_cached_sigma_model(baseline, **inputs, **noise)
    point = _assert_linked_density_and_gradient_parity(ordinary, cached)
    values = _evaluate(ordinary, [ordinary.flux_scaling], point)[0]
    posterior = xr.Dataset({"flux_scaling": xr.DataArray(
        values, dims="state", coords={"state": prepared["co2"].state},
    )})
    for channel, frozen in prepared.items():
        assert f"{channel}_baseline_delta" not in ordinary
        count = frozen.sizes["bc_pattern_term"]
        assert point[f"{channel}_bc_pattern_standardized"].size == count
        # The level responds with one on its own channel's rows and zero elsewhere.
        response = inputs["observations"].species.values == channel
        level, coefficient = _evaluate(ordinary, [
            ordinary[f"{channel}_mu_bc_mean_shift"], ordinary[f"{channel}_bc_mean_shift"],
        ], point)
        np.testing.assert_array_equal(level, response * coefficient)
        posterior[f"{channel}_bc_mean_shift"] = xr.DataArray(.3)
        posterior[f"{channel}_bc_pattern_standardized"] = xr.DataArray(
            np.ones(count), dims=f"{channel}_pattern",
            coords={f"{channel}_pattern": frozen.bc_pattern_term.values},
        )
    replay = reconstruct_linked_shared_baseline(prepared, posterior)
    for channel, frozen in prepared.items():
        np.testing.assert_array_equal(replay[channel].bc_centered_correction, 0.)
        np.testing.assert_allclose(replay[channel].bc_raw_correction, .25 * frozen.sizes["bc_pattern_term"])
