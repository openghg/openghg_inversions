"""Frozen unequal tracer supports and matched linked shared-baseline graphs."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.models.boundary_corrections import prepare_boundary_correction
from openghg_inversions.models.linked_shared_baseline import prepare_linked_shared_baseline
from openghg_inversions.rhime.co2.linked_shared_baseline_model import (
    build_shared_co2_o2_model, build_shared_co2_o2_cached_sigma_model,
)
from test_boundary_correction_priors import operands
from test_co2_o2_fixed_ou import _prepared
from test_rhime_co2_o2 import _independent_error


def case(mode="multiplicative", cross=0.0, constant_outer=False):
    base = _prepared()
    prior = base.retained_prior
    if cross or constant_outer:
        from openghg_inversions.correlated_state import CorrelatedLognormalPrior
        covariance = prior.arithmetic_covariance.copy()
        covariance.values[0, 2] = covariance.values[2, 0] = cross
        mean = prior.mean
        if constant_outer:
            values = mean.values.copy()
            values[2] = 1.5
            mean = mean.copy(data=values)
        prior = CorrelatedLognormalPrior(mean, covariance)
    fixed, corrections, raw = {}, {}, {}
    for channel in ("co2", "o2"):
        h = getattr(base, f"{channel}_sensitivity")
        row = str(h.dims[0])
        obs = select_gathered_data_array(
            base.observations, key=channel, key_dim="species", ragged_dim="channel_observation",
            stack_dim="observation",
        ).rename(observation=row).assign_attrs(units="ppm")
        h = h.assign_coords(site=obs.site, time=obs.time).assign_attrs(units="ppm")
        if constant_outer:
            values = h.values.copy()
            values[:, 2] = 4.0 if channel == "co2" else -3.0
            h = h.copy(data=values)
        raw[channel] = h
        fixed[channel] = xr.zeros_like(obs).assign_attrs(units="ppm")
        corrections[channel] = prepare_boundary_correction(
            operands(obs, c=400.0 if channel == "co2" else 200000.0), obs,
            method="A", level_sd=2.0, pattern_sd=0.3,
        )
    outer = prior.mean.copy(data=[False, False, True, True, True])
    sampled = prior.mean.copy(data=[False, False, True, False, False])
    calibration = None
    if mode == "multiplicative_shrunk":
        from openghg_inversions.biosphere_prior import calibrate_biosphere_prior
        outer = prior.mean.copy(data=[True, True, False, True, True])
        sampled = prior.mean.copy(data=[True, True, False, False, False])
        weights = xr.DataArray(
            [[-1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]],
            dims=("process", "state"), coords={"process": ["GPP", "TER"], **dict(prior.mean.coords)},
            attrs={"units": "kg/year"},
        )
        reference = xr.DataArray([-1., 1.], dims="process", coords={"process": ["GPP", "TER"]},
                                 attrs={"units": "kg/year"})
        calibration = calibrate_biosphere_prior(prior, biosphere=sampled, weights=weights,
                                               reference=reference, residual_sd_multiplier=0.5)
    p = prepare_linked_shared_baseline(
        co2_sensitivity=raw["co2"], o2_sensitivity=raw["o2"], retained_prior=prior,
        fixed_prior_contribution=fixed, boundary_correction=corrections,
        outer=outer, sampled_outer=sampled, mode=mode, delta_sd={"co2": 2.0, "o2": 0.5},
        calibrated_prior=calibration,
    )
    return p, base, raw


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


@pytest.mark.parametrize("mode", ["fixed", "multiplicative", "multiplicative_shrunk"])
def test_frozen_unequal_rows_prior_and_absolute_closure(mode):
    p, base, raw = case(mode, cross=0.001 if mode == "multiplicative" else 0.0)
    assert p["co2"].sizes["co2_measure"] == 2
    assert p["o2"].sizes["o2_measure"] == 3
    np.testing.assert_allclose(p["o2"].centering_weights, [0.25, 0.5, 0.25])
    for channel, prepared in p.items():
        row = prepared.attrs["native_observation_dim"]
        np.testing.assert_allclose(prepared.centering_weights @ prepared.bc_pattern_centered, 0, atol=1e-15)
        subset = prepared.isel({row: [0]})
        xr.testing.assert_equal(subset.bc_pattern_mean, prepared.bc_pattern_mean)
        np.testing.assert_allclose(
            prepared.fixed_prior_contribution + prepared.flux_sensitivity @ prepared.retained_mean,
            raw[channel] @ base.retained_prior.mean,
        )
    xr.testing.assert_equal(p["co2"].retained_covariance, p["o2"].retained_covariance)
    if mode == "multiplicative":
        assert p["co2"].retained_covariance.values[0, 2] == 0.001


def test_reject_additive_and_removed_covariance():
    with pytest.raises(ValueError, match="oxidative allocation"):
        case("additive")
    with pytest.raises(ValueError, match="cross covariance"):
        case("fixed", cross=0.001)


@pytest.mark.parametrize("mode", ["fixed", "multiplicative", "multiplicative_shrunk"])
def test_linked_cached_density_gradient_and_refreshed_cache(mode):
    p, base, _ = case(mode)
    kwargs = dict(observations=base.observations, independent_error_sd=_independent_error(base),
                  aggregation_error=base.aggregation_error)
    noise = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    ordinary = build_shared_co2_o2_model(p, **kwargs, **noise)
    cached = build_shared_co2_o2_cached_sigma_model(p, **kwargs, **noise)
    _assert_linked_density_and_gradient_parity(ordinary, cached)


def test_saved_preparation_and_posterior_reconstruct_without_model(tmp_path):
    from openghg_inversions.rhime.co2.linked_shared_baseline_model import reconstruct_linked_shared_baseline
    from openghg_inversions.serialization import encode_multiindexes_for_storage, restore_serialisation_multiindexes

    prepared, base, _ = case()
    model = build_shared_co2_o2_model(
        prepared, observations=base.observations, independent_error_sd=_independent_error(base),
        aggregation_error=base.aggregation_error,
    )
    point = model.initial_point()
    for name in point:
        point[name] = np.full_like(point[name], 0.25)
    names = ["flux_scaling", "co2_bc_level", "o2_bc_level", "co2_bc_pattern", "o2_bc_pattern",
             "co2_modelled_concentration", "o2_modelled_concentration"]
    values = model.compile_fn(model.replace_rvs_by_values([model[n] for n in names]),
                              inputs=model.value_vars, on_unused_input="ignore")(point)
    p = prepared["co2"]
    posterior = xr.Dataset({"flux_scaling": xr.DataArray(values[0], dims="state", coords={"state": p.state})})
    for channel, offset in (("co2", 1), ("o2", 2)):
        posterior[f"{channel}_bc_level"] = xr.DataArray(values[offset], dims=f"{channel}_level")
        posterior[f"{channel}_bc_pattern"] = xr.DataArray(
            values[offset + 2], dims=f"{channel}_pattern",
            coords={f"{channel}_pattern": prepared[channel].bc_pattern_term.values},
        )
    restored = {}
    for channel, frozen in prepared.items():
        path = tmp_path / f"{channel}.nc"
        encode_multiindexes_for_storage(frozen).to_netcdf(path)
        with xr.open_dataset(path) as saved:
            restored[channel] = restore_serialisation_multiindexes(saved.load())
    path = tmp_path / "posterior.nc"
    encode_multiindexes_for_storage(posterior).to_netcdf(path)
    with xr.open_dataset(path) as saved:
        posterior = restore_serialisation_multiindexes(saved.load())
    reconstructed = reconstruct_linked_shared_baseline(restored, posterior)
    for channel, expected in zip(("co2", "o2"), values[-2:]):
        mask = base.observations.species.values == channel
        np.testing.assert_allclose(reconstructed[channel].modelled_concentration, expected[mask], atol=1e-10)
        np.testing.assert_allclose(
            reconstructed[channel].outer_raw_correction - reconstructed[channel].outer_removed_mean,
            reconstructed[channel].outer_centered_correction, atol=1e-15,
        )


@pytest.mark.parametrize("ocean_source", ["OCEAN", "CO2_OCEAN", "O2_OCEAN"])
def test_uppercase_outer_ocean_cannot_be_sampled(ocean_source):
    from openghg_inversions.correlated_state import CorrelatedLognormalPrior

    prepared, base, raw = case()
    mean = base.retained_prior.mean.assign_coords(
        source=("state", ["GPP", "TER", "FF", ocean_source, ocean_source])
    )
    prior = CorrelatedLognormalPrior(mean, base.retained_prior.arithmetic_covariance)
    outer = mean.copy(data=[False, False, True, True, True])
    sampled = mean.copy(data=[False, False, True, True, False])
    # The ocean policy owns this rejection before binding channel operands.
    with pytest.raises(ValueError, match="Outer ocean must remain fixed"):
        prepare_linked_shared_baseline(
            co2_sensitivity=raw["co2"], o2_sensitivity=raw["o2"], retained_prior=prior,
            fixed_prior_contribution={c: p.fixed_prior_contribution for c, p in prepared.items()},
            boundary_correction=prepared, outer=outer, sampled_outer=sampled,
            mode="multiplicative", delta_sd={"co2": 2.0, "o2": 0.5},
        )


def _assert_linked_density_and_gradient_parity(ordinary, cached):
    point = ordinary.initial_point()
    for name in point:
        point[name] = np.linspace(0.1, 0.3, point[name].size).reshape(point[name].shape)
    cp = cached.model.initial_point()
    cp.update(point)
    assert "co2_bc_level" in point and "o2_bc_level" in point
    assert sum(v.name == "flux_scaling_latent" for v in ordinary.free_RVs) == 1
    np.testing.assert_allclose(ordinary.compile_logp()(point), cached.model.compile_logp()(cp), atol=1e-7)
    np.testing.assert_allclose(ordinary.compile_dlogp()(point), cached.model.compile_dlogp()(cp), atol=1e-6)
    cached.shared_cache.update(cached.target.refresh(np.exp(cp["ou_site_amplitude_log__"])))
    cached_likelihood = cached.model.replace_rvs_by_values([cached.cached_state_logp])[0]
    values = [cached.model.rvs_to_values[rv] for rv in cached.states]
    actual = cached.model.compile_fn(
        [cached_likelihood, *pt.grad(cached_likelihood, values)], inputs=cached.model.value_vars,
        on_unused_input="ignore",
    )(cp)
    likelihood = ordinary.logp(vars=ordinary.observed_RVs)
    values = [ordinary.rvs_to_values[ordinary[rv.name]] for rv in cached.states]
    expected = ordinary.compile_fn(
        [likelihood, *pt.grad(likelihood, values)], inputs=ordinary.value_vars, on_unused_input="ignore",
    )(point)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=1e-9, atol=1e-6)

    return point


def test_constant_outer_columns_preserve_one_joint_physical_draw_in_both_channels():
    prepared, base, _ = case(cross=0.001, constant_outer=True)
    for p in prepared.values():
        np.testing.assert_array_equal(p.flux_sensitivity.isel(state=2), 0.0)
    kwargs = dict(observations=base.observations, independent_error_sd=_independent_error(base),
                  aggregation_error=base.aggregation_error)
    noise = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    ordinary = build_shared_co2_o2_model(prepared, **kwargs, **noise)
    cached = build_shared_co2_o2_cached_sigma_model(prepared, **kwargs, **noise)
    point = _assert_linked_density_and_gradient_parity(ordinary, cached)
    for model in (ordinary, cached.model):
        assert sum(rv.name == "flux_scaling_latent" for rv in model.free_RVs) == 1
        names = ["flux_scaling"]
        for channel in ("co2", "o2"):
            names.extend(f"{channel}_{name}" for name in (
                "outer_raw_correction", "outer_physical_contribution", "outer_centered_correction",
            ))
        evaluate = model.compile_fn(
            model.replace_rvs_by_values([model[name] for name in names]),
            inputs=model.value_vars, on_unused_input="ignore",
        )
        state_point = dict(point)
        assert state_point["flux_scaling_latent"].shape == (3,)
        log_states = []
        for z in (np.zeros(3), *np.eye(3)):
            state_point["flux_scaling_latent"] = z
            state, *outputs = evaluate(state_point)
            log_states.append(np.log(state))
            for index, (channel, coefficient) in enumerate((("co2", 4.0), ("o2", -3.0))):
                mask = base.observations.species.values == channel
                raw, physical, centered = outputs[index * 3:index * 3 + 3]
                np.testing.assert_allclose(raw[mask], coefficient * (state[2] - 1.5), atol=1e-14)
                np.testing.assert_array_equal(raw[~mask], 0.0)
                np.testing.assert_allclose(
                    physical[mask], prepared[channel].physical_outer_prior + raw[mask], atol=1e-14,
                )
                np.testing.assert_array_equal(centered, 0.0)
        # Derive exact moments from the sampled graph's Gaussian map, without Monte Carlo error.
        location = log_states[0]
        factor = np.column_stack([value - location for value in log_states[1:]])
        covariance = factor @ factor.T
        mean = np.exp(location + np.diag(covariance) / 2)
        arithmetic_covariance = np.outer(mean, mean) * np.expm1(covariance)
        np.testing.assert_allclose(mean, prepared["co2"].retained_mean, atol=1e-14)
        np.testing.assert_allclose(arithmetic_covariance, prepared["co2"].retained_covariance, atol=1e-14)
        assert mean[2] == pytest.approx(1.5)
        assert arithmetic_covariance[2, 2] > 0
        assert arithmetic_covariance[0, 2] > 0
