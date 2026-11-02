"""Shared baseline algebra, frozen subset replay, and graph parity."""

import numpy as np
import pandas as pd
import pytest
import pytensor
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.biosphere_prior import calibrate_biosphere_prior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.shared_baseline import prepare_shared_baseline
from openghg_inversions.models.boundary_corrections import prepare_boundary_correction
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.rhime.co2.shared_baseline_model import (
    build_shared_co2_model,
    build_shared_co2_cached_sigma_model,
)
from openghg_inversions.serialization import (
    encode_multiindexes_for_storage,
    restore_serialisation_multiindexes,
)
from test_boundary_correction_priors import operands


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
    )
    outer = mean.copy(data=[False, True, True, True])
    sampled = mean.copy(data=[False, True, True, False])
    bc = prepare_boundary_correction(
        operands(obs), obs, method="A", level_sd=5.0, pattern_sd=0.7, frequency="2D", anchor_time="2021-01-01"
    )
    kwargs = dict(outer=outer, sampled_outer=sampled, mode=mode, delta_sd=5.0)
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
        bc = bc.copy()
        for name, value in bc.data_vars.items():
            if "nmeasure" in value.dims:
                bc[name] = borrowed(value)
        bc = bc.chunk({"nmeasure": 2})
    prepared = prepare_shared_baseline(h, prior, fixed, bc, **kwargs)
    return obs, h, prior, prepared


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


@pytest.mark.parametrize("mode", ["fixed", "additive", "multiplicative", "multiplicative_shrunk"])
def test_frozen_means_closure_and_reload(mode, tmp_path):
    _obs, h, prior, p = case(mode)
    np.testing.assert_allclose(p.centering_weights, [1 / 6, 1 / 6, 1 / 6, 1 / 2])
    np.testing.assert_allclose(p.centering_weights @ p.bc_pattern_centered, 0.0, atol=1e-15)
    np.testing.assert_allclose(p.centering_weights @ p.selected_outer_centered, 0.0, atol=1e-15)
    baseline = p.fixed_prior_contribution + p.flux_sensitivity @ p.retained_mean + p.physical_bc_prior
    np.testing.assert_allclose(baseline, 0.2 + h @ prior.mean + p.physical_bc_prior)
    sub = p.isel(nmeasure=[0, 3])
    np.testing.assert_allclose(sub.bc_pattern_centered, sub.bc_pattern_raw - p.bc_pattern_mean)
    assert np.max(abs(sub.centering_weights @ sub.bc_pattern_centered)) > 0
    path = tmp_path / "prepared.nc"
    encode_multiindexes_for_storage(p).to_netcdf(path)
    with xr.open_dataset(path) as opened:
        restored = restore_serialisation_multiindexes(opened.load(), strict=True)
    xr.testing.assert_equal(restored, p)
    if mode == "additive":
        assert p.sizes["region"] == 1
        np.testing.assert_allclose(p.additive_unit_operator * p.additive_density_sd, p.additive_raw)


def test_nonzero_cross_covariance_preserved_or_rejected():
    _, _, prior, p = case(cross=0.01)
    np.testing.assert_allclose(p.retained_covariance, prior.arithmetic_covariance.values[:3, :3])
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

    names = [
        "modelled_concentration",
        "flux_scaling",
        "baseline_delta",
        "bc_centered_correction",
        "outer_centered_correction",
        "outer_raw_correction",
        "outer_removed_mean",
    ]
    pred, state, delta, bc, outer, raw, removed = ordinary.compile_fn(
        ordinary.replace_rvs_by_values([ordinary[n] for n in names]),
        inputs=ordinary.value_vars,
        on_unused_input="ignore",
    )(point)
    inner = h.values[:, :1] @ state[:1]
    expected = 0.2 + p.physical_bc_prior.values + p.physical_outer_prior.values + inner + delta + bc + outer
    np.testing.assert_allclose(pred, expected, atol=1e-10)
    np.testing.assert_allclose(raw - removed, outer, atol=1e-14)


def test_lazy_preparation_never_executes_borrowed_transport_payloads():
    import dask.array as da

    calls = []
    _, _, _, prepared = case(lazy_payload_calls=calls)
    assert calls == []
    assert isinstance(prepared.flux_sensitivity.data, da.Array)
    assert isinstance(prepared.bc_pattern_centered.data, da.Array)
    assert isinstance(prepared.fixed_prior_contribution.data, da.Array)


def test_additive_net_and_independent_gross_designs_have_identical_covariance():
    _, _, _, prepared = case("additive")
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
    )
    boundary = prepare_boundary_correction(operands(obs), obs, method="level", level_sd=5.0)
    prepared = prepare_shared_baseline(
        h, prior, xr.full_like(obs, 0.2), boundary,
        outer=mean.copy(data=[False, True]), sampled_outer=mean.copy(data=[False, True]),
        mode="multiplicative", delta_sd=5.0,
    )
    np.testing.assert_array_equal(prepared.flux_sensitivity.sel(region="outer"), 0.0)
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
        names = ["flux_scaling", "outer_physical_contribution", "outer_raw_correction",
                 "outer_centered_correction", "modelled_concentration", "baseline_delta"]
        evaluate = model.compile_fn(
            model.replace_rvs_by_values([model[name] for name in names]),
            inputs=model.value_vars, on_unused_input="ignore",
        )
        state_point = dict(point)
        latent = "flux_scaling_latent"
        assert state_point[latent].shape == (2,)
        log_states = []
        for z in (np.zeros(2), np.array([1., 0.]), np.array([0., 1.])):
            state_point[latent] = z
            state, physical, raw, centered, prediction, delta = evaluate(state_point)
            log_states.append(np.log(state))
            np.testing.assert_allclose(raw, 4 * (state[1] - 1.5), atol=1e-14)
            np.testing.assert_allclose(physical, 4 * state[1], atol=1e-14)
            np.testing.assert_array_equal(centered, 0.0)
            np.testing.assert_allclose(
                prediction, 0.2 + prepared.physical_bc_prior + 6.0 + h[:, 0] * state[0] + delta,
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
