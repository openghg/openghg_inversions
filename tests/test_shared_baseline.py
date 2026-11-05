"""Shared baseline algebra, subset replay, and graph parity."""

from dataclasses import replace

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pytest
import pytensor
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.biosphere_prior import calibrate_biosphere_prior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.boundary import AdditiveChoice, AnomalyScalingChoice, CenteredOperands
from openghg_inversions.models.shared_baseline import SharedBaselineChoice
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2.shared_baseline_model import (
    build_shared_co2_model,
    build_shared_co2_cached_sigma_model,
    reconstruct_shared_baseline,
)

SCALING = AnomalyScalingChoice(None, 0.7, frequency="2D", anchor_time="2021-01-01")
NOISE = dict(tau_hours=5, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})


def operands(obs, c=400.0):
    """Normalised two-curtain centered transport on the observation rows."""
    n = obs.size
    mass = xr.DataArray(
        np.resize([[0.6, 0.3], [0.2, 0.9], [0.5, 0.3], [0.8, 0.4]], (n, 2)), dims=("nmeasure", "bc_region"),
        coords={"nmeasure": obs.nmeasure, "bc_region": ["N", "S"]}, attrs={"units": "1"},
    )
    mass = (mass / mass.sum("bc_region")).assign_attrs(units="1")
    anomaly = mass.copy(data=np.arange(2 * n).reshape(n, 2) * 0.01 - 0.025).assign_attrs(units="ppm")
    return CenteredOperands(anomaly, mass, xr.DataArray(c, attrs={"units": "ppm"}))


def bind(obs, h, prior, fixed, *, mode="multiplicative", boundary=SCALING, transport=None, **inputs):
    inputs.setdefault("outer", prior.mean.copy(data=[False, True, True, True][:prior.mean.size]))
    inputs.setdefault("sampled_outer", prior.mean.copy(data=[False, True, True, False][:prior.mean.size]))
    return SharedBaselineChoice(boundary, 5.0, mode).bind(
        obs, flux_sensitivity=h, retained_prior=prior, fixed_prior_contribution=fixed,
        boundary_transport=operands(obs) if transport is None else transport, **inputs,
    )


def errors(obs):
    return dict(observations=obs, observation_error=xr.full_like(obs, 0.2),
                aggregation_error=resolve_aggregation_error(obs.to_dataset(name="mf"), "none"))


def case(mode="multiplicative", cross=0.0, lazy_payload_calls=None):
    obs = xr.DataArray(
        [410.0, 411.0, 412.0, 413.0],
        dims="nmeasure",
        coords={
            "nmeasure": np.arange(4),
            "site": ("nmeasure", ["A", "A", "A", "B"]),
            "time": ("nmeasure", np.datetime64("2021-01-01") + np.arange(4).astype("timedelta64[D]")),
        },
        attrs={"units": "ppm"},
    )
    index = pd.MultiIndex.from_tuples(
        [("GPP", 0), ("GPP", 1), ("TER", 1), ("ocean", 1)], names=["source", "region_in_source"]
    )
    mean = xr.DataArray(
        np.ones(4), dims="region", coords=xr.Coordinates.from_pandas_multiindex(index, "region")
    )
    covariance = np.diag([0.08, 0.07, 0.06, 0.05])
    covariance[0, 1] = covariance[1, 0] = cross
    prior = CorrelatedLognormalPrior(mean, covariance, covariance_dim="region_cov")
    h = xr.DataArray(
        np.arange(16).reshape(4, 4) * 0.03,
        dims=("nmeasure", "region"),
        coords={**dict(obs.coords), **dict(mean.coords)},
        attrs={"units": "ppm"},
    )
    sampled = mean.copy(data=[False, True, True, False])
    transport = operands(obs)
    kwargs = {}
    if mode == "multiplicative_shrunk":
        weights = xr.DataArray(
            [[0.0, -2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]],
            dims=("process", "region"),
            coords={"process": ["GPP", "TER"], **dict(mean.coords)},
            attrs={"units": "kg/year"},
        )
        reference = xr.DataArray(
            [-2.0, 3.0], dims="process", coords={"process": ["GPP", "TER"]}, attrs={"units": "kg/year"}
        )
        kwargs["calibrated_prior"] = calibrate_biosphere_prior(
            prior, biosphere=sampled, weights=weights, reference=reference, residual_sd_multiplier=0.5
        )
    if mode == "additive":
        coords = {
            "nmeasure": obs.nmeasure,
            "outer_pattern": [f"{x}_{j}" for x in ("bio", "FF") for j in range(6)],
            "outer_source": ("outer_pattern", ["net_bio"] * 6 + ["FF"] * 6),
        }
        kwargs["additive_design"] = xr.DataArray(
            np.arange(48).reshape(4, 12) * 0.01,
            dims=("nmeasure", "outer_pattern"),
            coords=coords,
            attrs={"units": "ppm"},
        )
        kwargs["additive_density_sd"] = xr.DataArray(
            np.ones(12) * 2,
            dims="outer_pattern",
            coords={"outer_pattern": coords["outer_pattern"]},
            attrs={"units": "mol m-2 s-1"},
        )
    fixed = xr.full_like(obs, 0.2)
    if lazy_payload_calls is not None:
        import dask.array as da
        from dask import delayed

        def borrowed(value):
            data = value.values

            @delayed
            def load():
                lazy_payload_calls.append(value.name)
                return data

            return value.copy(data=da.from_delayed(load(), shape=data.shape, dtype=data.dtype)).chunk(
                {"nmeasure": 2}
            )

        h = borrowed(h)
        fixed = borrowed(fixed)
        transport = CenteredOperands(
            borrowed(transport.anomaly.rename("anomaly")), borrowed(transport.mass.rename("mass")), transport.reference,
        )
    baseline = bind(obs, h, prior, fixed, mode=mode, transport=transport, **kwargs)
    return obs, h, prior, baseline


def reconstructed_point(model, point, baseline):
    prepared = baseline.data
    names = ["flux_scaling", "bc_mean_shift"]
    if prepared.sizes["bc_pattern_term"]:
        names.append("bc_pattern_standardized")
    if "additive_centered" in prepared:
        names.append("outer_pattern_coefficient")
    values = model.compile_fn(model.replace_rvs_by_values([model[n] for n in names]),
                              inputs=model.value_vars, on_unused_input="ignore")(point)
    dimensions = {"flux_scaling": prepared.attrs["state_dim"], "bc_pattern_standardized": "bc_pattern_term",
                  "outer_pattern_coefficient": "outer_pattern"}
    samples = xr.Dataset({name: xr.DataArray(value, dims=dimensions[name],
                         coords={dimensions[name]: prepared[dimensions[name]]}) if name in dimensions
                         else xr.DataArray(value) for name, value in zip(names, values)})
    return reconstruct_shared_baseline(prepared, samples), values[0]


@pytest.fixture(autouse=True)
def precision():
    # Scalar prior parameters must retain float64 normalization constants too.
    with pytensor.config.change_flags(floatX="float64", cast_policy="numpy+floatX"):
        yield


@pytest.mark.parametrize("mode", ["fixed", "additive", "multiplicative", "multiplicative_shrunk"])
def test_centring_closure_and_bound_subsets(mode):
    _obs, h, prior, baseline = case(mode)
    p = baseline.data.compute()
    np.testing.assert_allclose(p.centering_weights, [1 / 6, 1 / 6, 1 / 6, 1 / 2])
    np.testing.assert_allclose(p.centering_weights @ (p.bc_pattern_raw - p.bc_pattern_mean), 0.0, atol=1e-15)
    assert "selected_outer_raw" not in p and "selected_outer_centered" not in p
    if mode.startswith("multiplicative"):
        np.testing.assert_allclose(
            p.centering_weights @ p.flux_sensitivity.where(p.sampled_outer, 0), 0.0, atol=1e-15,
        )
    baseline = p.fixed_prior_contribution + p.flux_sensitivity @ p.retained_mean + p.physical_bc_prior
    np.testing.assert_allclose(baseline, 0.2 + h @ prior.mean + p.physical_bc_prior)
    sub = p.isel(nmeasure=[0, 3])
    np.testing.assert_allclose((sub.bc_pattern_raw - sub.bc_pattern_mean), sub.bc_pattern_raw - p.bc_pattern_mean)
    assert np.max(abs(sub.centering_weights @ (sub.bc_pattern_raw - sub.bc_pattern_mean))) > 0
    if mode == "additive":
        assert p.sizes["region"] == 1
        np.testing.assert_allclose(p.additive_unit_operator * p.additive_density_sd, p.additive_raw)


def test_nonzero_cross_covariance_preserved_or_rejected():
    _, _, prior, baseline = case(cross=0.01)
    np.testing.assert_allclose(baseline.data.retained_covariance, prior.arithmetic_covariance.values[:3, :3])
    for mode in ("fixed", "additive"):
        with pytest.raises(ValueError, match="cross covariance"):
            case(mode, cross=0.01)


@pytest.mark.parametrize("mode", ["fixed", "additive", "multiplicative", "multiplicative_shrunk"])
def test_ordinary_cached_nonzero_density_gradient_and_absolute_closure(mode):
    obs, h, _prior, p = case(mode)
    aggregation = resolve_aggregation_error(obs.to_dataset(name="mf"), "none")
    kwargs = dict(observations=obs, observation_error=xr.full_like(obs, 0.2), aggregation_error=aggregation)
    noise = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    ordinary = build_shared_co2_model(
        p, **kwargs, likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=noise
    )
    cached = build_shared_co2_cached_sigma_model(p, **kwargs, **noise)
    point = _assert_density_and_gradient_parity(ordinary, cached)
    _assert_independent_density_and_all_gradients(ordinary, cached, point, h, p)

    reconstructed, state = reconstructed_point(ordinary, point, p)
    p = p.data
    pred, delta, bc, outer, raw, removed = [reconstructed[name].values for name in (
        "modelled_concentration", "baseline_delta", "bc_centered_correction",
        "outer_centered_correction", "outer_raw_correction", "outer_removed_mean",
    )]
    graph_prediction = ordinary.compile_fn(ordinary.replace_rvs_by_values([ordinary.modelled_concentration]),
        inputs=ordinary.value_vars, on_unused_input="ignore")(point)[0]
    np.testing.assert_allclose(pred, graph_prediction, atol=1e-10)
    inner = h.values[:, :1] @ state[:1]
    expected = 0.2 + p.physical_bc_prior.values + p.physical_outer_prior.values + inner + delta + bc + outer
    np.testing.assert_allclose(pred, expected, atol=1e-10)
    np.testing.assert_allclose(raw - removed, outer, atol=1e-14)


def test_lazy_preparation_never_executes_borrowed_transport_payloads():
    import dask.array as da

    calls = []
    _, _, _, baseline = case(lazy_payload_calls=calls)
    prepared = baseline.data
    assert calls == []
    assert isinstance(prepared.flux_sensitivity.data, da.Array)
    assert isinstance((prepared.bc_pattern_raw - prepared.bc_pattern_mean).data, da.Array)
    assert isinstance(prepared.fixed_prior_contribution.data, da.Array)


def test_additive_net_and_independent_gross_designs_have_identical_covariance():
    prepared = case("additive")[3].data
    unit = prepared.additive_unit_operator.values[:, :6]
    # Independent gross SDs 1.2 and 1.6 have net SD 2.0 on the same support.
    gross = np.column_stack((unit * 1.2, unit * 1.6))
    net = prepared.additive_raw.values[:, :6]
    weights = prepared.centering_weights.values
    gross -= weights @ gross
    net = net - weights @ net
    np.testing.assert_allclose(gross @ gross.T, net @ net.T, atol=1e-14)


def _assert_density_and_gradient_parity(ordinary, cached):
    point = ordinary.initial_point()
    for name in point:
        point[name] = np.linspace(0.1, 0.3, point[name].size).reshape(point[name].shape)
    cp = cached.model.initial_point()
    cp.update(point)
    np.testing.assert_allclose(ordinary.compile_logp()(point), cached.model.compile_logp()(cp), atol=1e-8)
    np.testing.assert_allclose(ordinary.compile_dlogp()(point), cached.model.compile_dlogp()(cp), atol=1e-7)
    cached.shared_cache.update(cached.target.refresh(np.exp(cp["ou_site_amplitude_log__"])))
    cached_likelihood = cached.model.replace_rvs_by_values([cached.cached_state_logp])[0]
    cached_values = [cached.model.rvs_to_values[rv] for rv in cached.states]
    cached_outputs = cached.model.compile_fn(
        [cached_likelihood, *pt.grad(cached_likelihood, cached_values)],
        inputs=cached.model.value_vars,
        on_unused_input="ignore",
    )(cp)
    ordinary_likelihood = ordinary.logp(vars=ordinary.observed_RVs)
    ordinary_values = [ordinary.rvs_to_values[ordinary[rv.name]] for rv in cached.states]
    direct_outputs = ordinary.compile_fn(
        [ordinary_likelihood, *pt.grad(ordinary_likelihood, ordinary_values)],
        inputs=ordinary.value_vars,
        on_unused_input="ignore",
    )(point)
    for actual, expected in zip(cached_outputs, direct_outputs):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-7)

    return point


def test_constant_outer_column_keeps_joint_physical_prior_and_cached_gradients():
    obs, _, _, _ = case()
    obs = obs.isel(nmeasure=[0, 1])
    mean = xr.DataArray([1.0, 1.5], dims="region", coords={"region": ["inner", "outer"]})
    prior = CorrelatedLognormalPrior(mean, [[0.08, 0.03], [0.03, 0.12]], covariance_dim="region_cov")
    h = xr.DataArray(
        [[2.0, 4.0], [6.0, 4.0]], dims=("nmeasure", "region"),
        coords={**dict(obs.coords), "region": mean.region},
        attrs={"units": "ppm"},
    )
    prepared = bind(obs, h, prior, xr.full_like(obs, 0.2), boundary=None,
                    outer=mean.copy(data=[False, True]), sampled_outer=mean.copy(data=[False, True]))
    np.testing.assert_array_equal(prepared.data.flux_sensitivity.sel(region="outer"), 0.0)
    aggregation = resolve_aggregation_error(obs.to_dataset(name="mf"), "none")
    kwargs = dict(observations=obs, observation_error=xr.full_like(obs, 0.2), aggregation_error=aggregation)
    noise = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    ordinary = build_shared_co2_model(
        prepared, **kwargs, likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=noise,
    )
    cached = build_shared_co2_cached_sigma_model(prepared, **kwargs, **noise)
    point = _assert_density_and_gradient_parity(ordinary, cached)
    for model in (ordinary, cached.model):
        # Recover the exact Gaussian map from zero and unit latent points, avoiding sampling error.
        state_point = dict(point)
        latent = "flux_scaling_latent"
        assert state_point[latent].shape == (2,)
        log_states = []
        for z in (np.zeros(2), np.array([1., 0.]), np.array([0., 1.])):
            state_point[latent] = z
            reconstructed, state = reconstructed_point(model, state_point, prepared)
            physical, raw, centered, prediction, delta = [reconstructed[name].values for name in (
                "outer_physical_contribution", "outer_raw_correction", "outer_centered_correction",
                "modelled_concentration", "baseline_delta",
            )]
            log_states.append(np.log(state))
            np.testing.assert_allclose(raw, 4 * (state[1] - 1.5), atol=1e-14)
            np.testing.assert_allclose(physical, 4 * state[1], atol=1e-14)
            np.testing.assert_array_equal(centered, 0.0)
            np.testing.assert_allclose(
                prediction, 0.2 + prepared.data.physical_bc_prior + 6.0 + h[:, 0] * state[0] + delta,
                atol=1e-12,
            )
        location = log_states[0]
        factor = np.column_stack([value - location for value in log_states[1:]])
        latent_covariance = factor @ factor.T
        arithmetic_mean = np.exp(location + np.diag(latent_covariance) / 2)
        arithmetic_covariance = np.outer(arithmetic_mean, arithmetic_mean) * np.expm1(latent_covariance)
        np.testing.assert_allclose(arithmetic_mean, prior.mean, atol=1e-14)
        np.testing.assert_allclose(arithmetic_covariance, prior.arithmetic_covariance, atol=1e-14)
        assert arithmetic_covariance[1, 1] > 0
        assert arithmetic_covariance[0, 1] > 0


@pytest.mark.parametrize("boundary", [
    AdditiveChoice({"pdf": "normal", "mu": 0.0, "sigma": 5.0}, 0.7, anomalies="period_contrasts"),
    AnomalyScalingChoice({"pdf": "normal", "mu": 0.0, "sigma": 5.0}, 0.7),
    AdditiveChoice({"pdf": "normal", "mu": 0.0, "sigma": 5.0}, 0.7),
])
def test_shared_level_replaces_no_boundary_mean_silently(boundary):
    with pytest.raises(ValueError, match="mean prior|period-contrast"):
        SharedBaselineChoice(boundary, 5.0, "multiplicative")


@pytest.mark.parametrize("boundary", [
    None, SCALING, AdditiveChoice(None, 0.7, anomalies="period_contrasts", frequency="2D", anchor_time="2021-01-01"),
    AdditiveChoice(None, 0.7, anomalies="independent"),
])
def test_boundary_patterns_are_the_family_anomaly_factor(boundary):
    obs, h, prior, _ = case()
    data = bind(obs, h, prior, xr.full_like(obs, 0.2), boundary=boundary).data.compute()
    if boundary is None:
        assert data.sizes["bc_pattern_term"] == 0
        return
    factor = boundary.bind(obs, operands(obs)).anomaly_factor()
    np.testing.assert_allclose(data.bc_pattern_raw, factor.transpose("nmeasure", ...), rtol=0, atol=1e-15)


def test_shared_model_keeps_scientific_states_without_duplicate_diagnostics():
    obs, _, _, baseline = case("additive")
    model = build_shared_co2_model(
        baseline, observations=obs, observation_error=xr.full_like(obs, 0.2),
        aggregation_error=resolve_aggregation_error(obs.to_dataset(name="mf"), "none"),
        fixed_model_mismatch=0.0,
    )
    assert {"bc_mean_shift", "bc_pattern_standardized", "outer_pattern_coefficient"} <= set(model.named_vars)
    assert not {"baseline_delta", "bc_level", "bc_level_operator", "bc_mass_response", "outer_raw_correction",
                "outer_removed_mean", "shared_physical_prior_baseline"} & set(model.named_vars)
    assert not hasattr(model, "shared_baseline")


def test_additive_preparation_does_not_compute_secondary_observation_coordinates():
    obs, h, prior, baseline = case("additive")
    prepared = baseline.data
    additive = (
        prepared.additive_raw.assign_coords(site=h.site, time=h.time)
        .chunk({"nmeasure": 2})
        .assign_coords(outer_source=prepared.additive_raw.outer_source)
    )
    tasks = []
    with Callback(posttask=lambda key, *_: tasks.append(key)):
        result = bind(
            obs, h.copy(data=da.from_array(h.data, chunks=(2, 4))), prior, xr.full_like(obs, 0.2), mode="additive",
            additive_design=additive, additive_density_sd=prepared.additive_density_sd,
        ).data
    assert tasks == []
    assert isinstance(result.additive_centered.data, da.Array)
    assert isinstance(additive.site.data, da.Array)
    xr.testing.assert_allclose(result.additive_centered.reset_coords(drop=True), prepared.additive_centered.reset_coords(drop=True))


def test_explicit_additive_design_accepts_small_labelled_basis_and_equivalent_units():
    obs, h, prior, baseline = case("additive")
    baseline = baseline.data
    design = baseline.additive_raw.isel(outer_pattern=[0, 6])
    kwargs = dict(
        mode="additive", boundary=None,
        additive_design=(design * 1000).assign_attrs(units="ppb"),
        additive_density_sd=baseline.additive_density_sd.isel(outer_pattern=[0, 6]),
    )
    result = bind(
        obs, (h * 1000).assign_attrs(units="ppb"), prior, xr.full_like(obs, 200).assign_attrs(units="ppb"), **kwargs,
    ).data
    np.testing.assert_allclose(result.additive_raw, design)
    np.testing.assert_allclose(result.fixed_prior_contribution, baseline.fixed_prior_contribution)
    with pytest.raises(ValueError, match="declared concentration units"):
        bind(obs, h.assign_attrs(units=None), prior, xr.full_like(obs, .2), **kwargs)


def test_model_boundary_leaves_unused_frozen_diagnostics_lazy():
    from dask import delayed
    from openghg_inversions.rhime.co2.shared_baseline_model import shared_model_payload

    obs, _, _, baseline = case()
    prepared = baseline.data

    @delayed
    def unused():
        raise AssertionError("Unused reconstruction diagnostic was computed")

    prepared["complete_prior_concentration"] = xr.DataArray(
        da.from_delayed(unused(), shape=(4,), dtype=float), dims="nmeasure", coords={"nmeasure": obs.nmeasure},
    )
    selected = shared_model_payload(prepared).compute()
    assert "complete_prior_concentration" not in selected
    assert isinstance(prepared.complete_prior_concentration.data, da.Array)


@pytest.mark.parametrize("source", ["bc", "outer"])
def test_constant_pattern_keeps_raw_uncertainty_and_cached_gradients(source):
    obs, _, _, baseline = case("additive" if source == "outer" else "multiplicative")
    prepared = baseline.data.copy()
    raw, mean, dimension, variable = (
        ("bc_pattern_raw", "bc_pattern_mean", "bc_pattern_term", "bc_pattern_standardized")
        if source == "bc" else
        ("additive_raw", "additive_mean", "outer_pattern", "outer_pattern_coefficient")
    )
    constant = xr.full_like(prepared[raw], 0.7)
    prepared[raw] = constant
    prepared[mean] = xr.dot(prepared.centering_weights, constant, dim="nmeasure")
    if source == "outer":
        prepared["additive_centered"] = constant - prepared[mean]
    kwargs = dict(
        observations=obs, observation_error=xr.full_like(obs, 0.2),
        aggregation_error=resolve_aggregation_error(obs.to_dataset(name="mf"), "none"),
    )
    noise = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    prepared = replace(baseline, data=prepared)
    ordinary = build_shared_co2_model(
        prepared, **kwargs, likelihood_builder=add_fixed_ou_gaussian_likelihood, likelihood_kwargs=noise,
    )
    cached = build_shared_co2_cached_sigma_model(prepared, **kwargs, **noise)
    point = _assert_density_and_gradient_parity(ordinary, cached)
    assert variable in {rv.name for rv in cached.states}
    assert point[variable].size == prepared.data.sizes[dimension]
    replay, _ = reconstructed_point(ordinary, point, prepared)
    np.testing.assert_allclose(replay[f"{source}_centered_correction"], 0, atol=1e-15)
    np.testing.assert_allclose(replay[f"{source}_raw_correction"], 0.7 * point[variable].sum(), atol=1e-14)
    assert float(abs(replay[f"{source}_raw_correction"]).max()) > 0


def test_shared_level_uses_additive_mean_shift_names_roles_and_prior():
    from openghg_inversions.models.coords import registered_model

    obs, _, _, baseline = case()
    shared = build_shared_co2_cached_sigma_model(baseline, **errors(obs), **NOISE)
    additive = AdditiveChoice({"pdf": "normal", "mu": 0.0, "sigma": 5.0}, 0.0, anomalies="period_contrasts")
    bound = additive.bind(obs, operands(obs))
    with registered_model() as other:
        bound.add()
    for role in ("boundary_mean_shift", "boundary_mean_shift_concentration"):
        assert shared.boundary_outputs.roles[role] == bound.outputs().roles[role]
    for model in (shared.model, other):
        assert "baseline_delta" not in model
        point = model.initial_point()
        point["bc_mean_shift"] = np.array(-0.42)
        level = model.compile_fn(
            model.replace_rvs_by_values([model["mu_bc_mean_shift"]]), inputs=model.value_vars, on_unused_input="ignore",
        )(point)[0]
        np.testing.assert_allclose(level, -0.42, rtol=0, atol=1e-15)  # Normalised transport: M1 = 1.
        logp = model.compile_logp(vars=[model["bc_mean_shift"]])(point)
        np.testing.assert_allclose(logp, -np.log(5 * np.sqrt(2 * np.pi)) - 0.5 * (-0.42 / 5) ** 2)


@pytest.mark.parametrize("cached", [False, True])
def test_shared_transport_and_independent_error_materialize_once(cached):
    from dask import delayed

    obs, h, prior, _ = case()
    calls = []
    native_h = h.data

    @delayed
    def load():
        calls.append("shared")
        return np.column_stack((native_h, np.full(obs.size, 0.2)))

    values = da.from_delayed(load(), shape=(obs.size, 5), dtype=float)
    h = h.copy(data=values[:, :4])
    error = obs.copy(data=values[:, 4])
    baseline = bind(obs, h, prior, xr.full_like(obs, 0.2), boundary=None)
    prepared = baseline.data

    @delayed
    def unused():
        raise AssertionError("Unused diagnostics must remain lazy")

    prepared["complete_prior_concentration"] = obs.copy(data=da.from_delayed(unused(), shape=obs.shape, dtype=float))
    prepared = prepared.assign_coords(unused_auxiliary=("nmeasure", da.from_delayed(unused(), shape=obs.shape, dtype=float)))
    aggregation_error = resolve_aggregation_error(obs.to_dataset(name="mf"), "none")
    obs = obs.assign_coords(unused_observation_auxiliary=(
        "nmeasure", da.from_delayed(unused(), shape=obs.shape, dtype=float),
    ))
    error = error.assign_coords(unused_error_auxiliary=(
        "nmeasure", da.from_delayed(unused(), shape=error.shape, dtype=float),
    ))
    builder = build_shared_co2_cached_sigma_model if cached else build_shared_co2_model
    kwargs = dict(tau_hours=5, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75}) if cached else {}
    builder(replace(baseline, data=prepared), observations=obs, observation_error=error,
            aggregation_error=aggregation_error, **kwargs)
    assert calls == ["shared"]
    assert isinstance(prepared.complete_prior_concentration.data, da.Array)
    assert isinstance(obs.unused_observation_auxiliary.data, da.Array)
    assert isinstance(error.unused_error_auxiliary.data, da.Array)


@pytest.mark.parametrize("cached", [False, True])
def test_shared_float32_rejected_before_transport_materialization(cached):
    from dask import delayed

    obs, _, _, baseline = case()

    @delayed
    def forbidden():
        raise AssertionError("float32 rejection must precede materialization")

    prepared = baseline.data.assign(flux_sensitivity=baseline.data.flux_sensitivity.copy(
        data=da.from_delayed(forbidden(), shape=baseline.data.flux_sensitivity.shape, dtype=float),
    ))
    builder = build_shared_co2_cached_sigma_model if cached else build_shared_co2_model
    kwargs = dict(tau_hours=5, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75}) if cached else {}
    with pytensor.config.change_flags(floatX="float32"), pytest.raises(ValueError, match="float64"):
        builder(replace(baseline, data=prepared), **errors(obs), **kwargs)


def test_shared_trace_roundtrip_canonical_roles_units_and_graph_free_replay(tmp_path):
    import pymc as pm
    from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
    from openghg_inversions.rhime.co2.boundary_output import decode_boundary_facts, reconstruct_boundary, restore_absolute_predictives
    from openghg_inversions.rhime.co2.shared_baseline_model import annotate_shared_baseline_trace
    from openghg_inversions.serialization import load_trace, save_trace

    obs, _, _, baseline = case("additive")
    model = build_shared_co2_model(
        baseline, **errors(obs),
    )
    with model:
        trace = pm.sample_prior_predictive(draws=2, random_seed=117)
    trace = restore_inferencedata_coords(trace, get_coord_registry(model))
    trace = restore_absolute_predictives(trace)
    np.testing.assert_allclose(trace.observed_data.y, obs)
    trace = annotate_shared_baseline_trace(trace, baseline)
    save_trace(trace, tmp_path / "shared.nc")
    restored = load_trace(tmp_path / "shared.nc")
    assert "shared_atmospheric" in restored.attrs["rhime_model_metadata"]
    assert restored.prior.bc_mean_shift.attrs["units"] == "ppm"
    assert restored.prior.bc_pattern_standardized.attrs["units"] == "1"
    assert restored.prior.bc_pattern_term.attrs["units"] == "1"
    facts = decode_boundary_facts(restored)[""]
    scalar = reconstruct_boundary(facts, restored.prior.to_dataset())
    xr.testing.assert_equal(scalar["mean_shift"], restored.prior.bc_mean_shift)
    replay = reconstruct_shared_baseline(restored.shared_baseline.to_dataset(), restored.prior.to_dataset())
    xr.testing.assert_allclose(
        replay.modelled_concentration.transpose(*restored.prior.modelled_concentration.dims).reset_coords(drop=True),
        restored.prior.modelled_concentration.reset_coords(drop=True),
    )
    xr.testing.assert_allclose(replay.baseline_delta, restored.prior.bc_mean_shift)
    assert "baseline_delta" not in restored.prior


@pytest.mark.parametrize("reference", [400.0, 209300.0])
def test_shared_likelihood_constructs_small_mean_without_reference_cancellation(reference):
    obs, h, prior, _ = case()
    obs = (obs - 400 + reference).assign_attrs(units="ppm")
    transport = operands(obs, reference)
    prepared = bind(obs, h, prior, xr.full_like(obs, 0.2).assign_attrs(units="ppm"), transport=transport)
    noise = dict(tau_hours=5, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    kwargs = dict(
        observations=obs, observation_error=xr.full_like(obs, 0.2),
        aggregation_error=resolve_aggregation_error(obs.to_dataset(name="mf"), "none"),
    )
    import pymc as pm

    def likelihood(*, mean, **inputs):
        pm.Deterministic("shared_likelihood_mean", mean, dims="nmeasure")
        return add_fixed_ou_gaussian_likelihood(mean=mean, **inputs)

    ordinary = build_shared_co2_model(
        prepared, **kwargs, likelihood_builder=likelihood, likelihood_kwargs=noise,
    )
    cached = build_shared_co2_cached_sigma_model(prepared, **kwargs, **noise)
    for model in (ordinary, cached.model):
        point = model.initial_point()
        point["bc_mean_shift"] = np.array(0.000000003)
        point["bc_pattern_standardized"] = np.linspace(0.1, 0.3, point["bc_pattern_standardized"].size)
        names = ["co2_flux_contribution", "bc_pattern_contribution", "bc_mean_shift"]
        flux, pattern, level = model.compile_fn(
            model.replace_rvs_by_values([model[name] for name in names]), inputs=model.value_vars, on_unused_input="ignore",
        )(point)
        expected = flux + prepared.data.bc_prescribed_centered.data + level + pattern
        mean = model["shared_likelihood_mean"] if model is ordinary else cached.modelled_mean
        actual = model.compile_fn(model.replace_rvs_by_values([mean]), inputs=model.value_vars, on_unused_input="ignore")(point)[0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(model["Y"].get_value(), obs.data - reference)


@pytest.mark.parametrize("source", ["GPP", "ocean"])
def test_additive_outer_design_rejects_gross_and_ocean_labels(source):
    obs, h, prior, baseline = case("additive")
    prepared = baseline.data
    with pytest.raises(ValueError, match="net_bio/FF"):
        bind(
            obs, h, prior, xr.full_like(obs, 0.2), mode="additive", boundary=None,
            additive_design=prepared.additive_raw.assign_coords(outer_source=("outer_pattern", [source] * 12)),
            additive_density_sd=prepared.additive_density_sd,
        )


def _assert_independent_density_and_all_gradients(ordinary, cached, point, h, baseline):
    """Dense NumPy forward/prior/OU target, independent of PyMC and cache code."""
    prepared = baseline.data.compute()
    names = [value.name for value in ordinary.value_vars]
    shapes = {name: point[name].shape for name in names}
    sizes = {name: point[name].size for name in names}
    vector = np.concatenate([point[name].reshape(-1) for name in names])
    mean = prepared.retained_mean.data
    covariance = np.log1p(prepared.retained_covariance.data / np.outer(mean, mean))
    location = np.log(mean) - np.diag(covariance) / 2
    cholesky = np.linalg.cholesky(covariance)
    weights = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 2])
    original_mean = prepared.original_prior_mean.data
    sampled = prepared.sampled_outer.data
    time = (h.time.data - h.time.data[0]) / np.timedelta64(1, "h")
    site = np.array([0, 0, 0, 1])
    correlation = np.exp(-abs(time[:, None] - time[None, :]) / 5) * (site[:, None] == site[None, :])
    observed = np.array([410, 411, 412, 413]) - prepared.bc_reference_shift.data
    normal_constant = np.log(2 * np.pi)

    def density(vector):
        values, offset = {}, 0
        for name in names:
            values[name] = vector[offset:offset + sizes[name]].reshape(shapes[name])
            offset += sizes[name]
        flux = np.exp(location + cholesky @ values["flux_scaling_latent"])
        small_mean = 0.2 + h.data[:, 0] * flux[0] + h.data[:, 1:] @ original_mean[1:]
        if prepared.attrs["mode"].startswith("multiplicative"):
            outer_h = h.data[:, 1:3]
            small_mean += (outer_h - weights @ outer_h) @ (flux[sampled] - mean[sampled])
        bc = prepared.bc_pattern_raw.data
        small_mean += prepared.bc_prescribed_centered.data + values["bc_mean_shift"]
        small_mean += (bc - weights @ bc) @ values["bc_pattern_standardized"]
        if "additive_raw" in prepared:
            outer = prepared.additive_raw.data
            small_mean += (outer - weights @ outer) @ values["outer_pattern_coefficient"]
        eta = values["ou_site_amplitude_log__"]
        amplitude = np.exp(eta)
        row_amplitude = amplitude[site]
        noise = np.diag(np.full(4, 0.2**2)) + np.outer(row_amplitude, row_amplitude) * correlation
        residual = observed - small_mean
        logp = -0.5 * (4 * normal_constant + np.linalg.slogdet(noise)[1] + residual @ np.linalg.solve(noise, residual))
        for name in ("flux_scaling_latent", "bc_pattern_standardized", "outer_pattern_coefficient"):
            if name in values:
                z = values[name]
                logp -= 0.5 * (z.size * normal_constant + z @ z)
        logp -= np.log(5) + 0.5 * (normal_constant + (values["bc_mean_shift"] / 5)**2)
        logp += np.sum(np.log(2) - np.log(0.75) - 0.5 * normal_constant - 0.5 * (amplitude / 0.75)**2 + eta)
        return float(logp)

    expected = density(vector)
    gradient = np.empty_like(vector)
    for i in range(vector.size):
        step = np.zeros_like(vector)
        step[i] = 1e-5
        gradient[i] = (density(vector + step) - density(vector - step)) / (2e-5)
    for model in (ordinary, cached.model):
        np.testing.assert_allclose(model.compile_logp()(point), expected, rtol=0, atol=1e-10)
        np.testing.assert_allclose(model.compile_dlogp()(point), gradient, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("cached", [False, True])
def test_single_shared_nonunit_mass_restores_absolute_predictives(cached):
    import pymc as pm
    from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
    from openghg_inversions.rhime.co2.boundary_output import restore_absolute_predictives
    from openghg_inversions.rhime.co2.shared_baseline_model import annotate_shared_baseline_trace

    obs, h, prior, _ = case()
    transport = operands(obs)
    scale = xr.DataArray([.5, 2., 1.5, .75], dims="nmeasure", coords={"nmeasure": obs.nmeasure})
    transport = replace(transport, mass=(transport.mass * scale).assign_attrs(units="1"))
    bound = bind(obs, h, prior, xr.full_like(obs, .2).assign_attrs(units="ppm"), transport=transport)
    builder = build_shared_co2_cached_sigma_model if cached else build_shared_co2_model
    model = builder(bound, **errors(obs), **(NOISE if cached else {}))
    model = model.model if cached else model
    with model:
        trace = pm.sample_prior_predictive(2, random_seed=138)
    trace = restore_inferencedata_coords(trace, get_coord_registry(model))
    raw = trace.prior_predictive.y.copy(deep=True)
    mean = trace.prior.modelled_concentration.copy(deep=True)
    shift = transport.reference * transport.mass.sum("bc_region")
    trace = restore_absolute_predictives(trace)
    trace = annotate_shared_baseline_trace(trace, bound)
    np.testing.assert_allclose(trace.observed_data.y, obs)
    np.testing.assert_allclose(trace.prior_predictive.y, raw + shift)
    np.testing.assert_array_equal(trace.prior.modelled_concentration, mean)
    once = trace.copy(deep=True)
    restore_absolute_predictives(trace)
    annotate_shared_baseline_trace(trace, bound)
    xr.testing.assert_identical(trace, once)
