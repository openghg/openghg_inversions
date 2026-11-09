"""Bound unequal tracer supports and matched linked shared-baseline graphs."""

from dataclasses import replace

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.models.boundary import AnomalyScalingChoice
from openghg_inversions.models.linked_shared_baseline import bind_linked_shared_baseline
from openghg_inversions.models.shared_baseline import SharedBaselineChoice
from openghg_inversions.rhime.co2.linked_shared_baseline_model import (
    build_shared_co2_o2_model, build_shared_co2_o2_cached_sigma_model,
    reconstruct_linked_shared_baseline,
)
from test_co2_o2_fixed_ou import _prepared
from test_rhime_co2_o2 import _independent_error
from test_shared_baseline import operands

DELTA_SD = {"co2": 2.0, "o2": 0.5}


def choices(mode, boundary=AnomalyScalingChoice(None, 0.3)):
    return {channel: SharedBaselineChoice(boundary, sd, mode) for channel, sd in DELTA_SD.items()}


def case(mode="multiplicative", cross=0.0, constant_outer=False, inputs=None):
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
    fixed, transport, raw, native = {}, {}, {}, {}
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
        native[channel] = obs
        fixed[channel] = xr.zeros_like(obs).assign_attrs(units="ppm")
        transport[channel] = operands(obs.rename({row: "nmeasure"}), c=400.0 if channel == "co2" else 200000.0)
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
    if inputs is not None:
        inputs.update(observations=native, fixed_prior_contribution=fixed, boundary_sensitivity=transport)
    p = bind_linked_shared_baseline(
        choices(mode), native,
        co2_sensitivity=raw["co2"], o2_sensitivity=raw["o2"], retained_prior=prior,
        fixed_prior_contribution=fixed, boundary_sensitivity=transport,
        outer=outer, sampled_outer=sampled, calibrated_prior=calibration,
    )
    return p, base, raw


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64", cast_policy="numpy+floatX"):
        yield


@pytest.mark.parametrize("mode", ["fixed", "multiplicative", "multiplicative_shrunk"])
def test_bound_unequal_rows_prior_and_absolute_closure(mode):
    bound, base, raw = case(mode, cross=0.001 if mode == "multiplicative" else 0.0)
    p = {channel: baseline.data for channel, baseline in bound.items()}
    assert p["co2"].sizes["co2_measure"] == 2
    assert p["o2"].sizes["o2_measure"] == 3
    np.testing.assert_allclose(p["o2"].centering_weights, [0.25, 0.5, 0.25])
    for channel, prepared in p.items():
        row = prepared.attrs["native_observation_dim"]
        np.testing.assert_allclose(prepared.centering_weights @ (prepared.bc_pattern_raw - prepared.bc_pattern_mean), 0, atol=1e-15)
        subset = replace(bound[channel], data=prepared.isel({row: [0]})).data
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


def test_posterior_reconstructs_from_recorded_data_without_model(tmp_path, monkeypatch):
    from openghg_inversions.serialization import encode_multiindexes_for_storage, restore_serialisation_multiindexes

    baseline, base, _ = case()
    model = build_shared_co2_o2_model(
        baseline, observations=base.observations, independent_error_sd=_independent_error(base),
        aggregation_error=base.aggregation_error,
    )
    point = model.initial_point()
    for name in point:
        point[name] = np.full_like(point[name], 0.25)
    names = ["flux_scaling", "co2_bc_mean_shift", "o2_bc_mean_shift", "co2_bc_pattern_standardized", "o2_bc_pattern_standardized",
             "modelled_concentration", "modelled_concentration"]
    values = model.compile_fn(model.replace_rvs_by_values([model[n] for n in names]),
                              inputs=model.value_vars, on_unused_input="ignore")(point)
    recorded = {channel: bound.data for channel, bound in baseline.items()}
    posterior = xr.Dataset({"flux_scaling": xr.DataArray(values[0], dims="state", coords={"state": recorded["co2"].state})})
    for channel, offset in (("co2", 1), ("o2", 2)):
        posterior[f"{channel}_bc_mean_shift"] = xr.DataArray(values[offset])
        # Saved samples carry the boundary's labelled (curtain, period) pattern index.
        index = recorded[channel].get_index("bc_pattern_term")
        index = index.set_names([f"{channel}_{name}" for name in index.names])
        posterior[f"{channel}_bc_pattern_standardized"] = xr.DataArray(
            values[offset + 2], dims=f"{channel}_pattern",
            coords=xr.Coordinates.from_pandas_multiindex(index, f"{channel}_pattern"),
        )
    path = tmp_path / "posterior.nc"
    encode_multiindexes_for_storage(posterior).to_netcdf(path)
    with xr.open_dataset(path) as saved:
        posterior = restore_serialisation_multiindexes(saved.load())

    def forbidden(*args, **kwargs):
        pytest.fail("Replay attempted graph construction or binding")

    import openghg_inversions.rhime.co2.linked_shared_baseline_model as recipe
    monkeypatch.setattr(recipe, "registered_model", forbidden)
    monkeypatch.setattr(SharedBaselineChoice, "bind", forbidden)
    reconstructed = reconstruct_linked_shared_baseline(recorded, posterior)
    for channel, expected in zip(("co2", "o2"), values[-2:]):
        mask = base.observations.species.values == channel
        np.testing.assert_allclose(reconstructed[channel].modelled_concentration, expected[mask], atol=1e-10)
        np.testing.assert_allclose(
            reconstructed[channel].outer_raw_correction - reconstructed[channel].outer_removed_mean,
            reconstructed[channel].outer_centered_correction, atol=1e-15,
        )


@pytest.mark.parametrize("ocean_source", ["ocean", "OCEAN"])
def test_outer_ocean_cannot_be_sampled(ocean_source):
    from openghg_inversions.correlated_state import CorrelatedLognormalPrior

    inputs = {}
    _, base, raw = case(inputs=inputs)
    mean = base.retained_prior.mean.assign_coords(
        source=("state", ["GPP", "TER", "FF", ocean_source, ocean_source])
    )
    prior = CorrelatedLognormalPrior(mean, base.retained_prior.arithmetic_covariance)
    outer = mean.copy(data=[False, False, True, True, True])
    sampled = mean.copy(data=[False, False, True, True, False])
    # The ocean policy owns this rejection before binding channel operands.
    with pytest.raises(ValueError, match="Outer ocean must remain fixed"):
        bind_linked_shared_baseline(
            choices("multiplicative"), co2_sensitivity=raw["co2"], o2_sensitivity=raw["o2"], retained_prior=prior,
            outer=outer, sampled_outer=sampled, **inputs,
        )


def _assert_linked_density_and_gradient_parity(ordinary, cached):
    point = ordinary.initial_point()
    for name in point:
        point[name] = np.linspace(0.1, 0.3, point[name].size).reshape(point[name].shape)
    cp = cached.model.initial_point()
    cp.update(point)
    assert "co2_bc_mean_shift" in point and "o2_bc_mean_shift" in point
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
    baseline, base, _ = case(cross=0.001, constant_outer=True)
    prepared = {channel: bound.data for channel, bound in baseline.items()}
    for p in prepared.values():
        np.testing.assert_array_equal(p.flux_sensitivity.isel(state=2), 0.0)
    kwargs = dict(observations=base.observations, independent_error_sd=_independent_error(base),
                  aggregation_error=base.aggregation_error)
    noise = dict(tau_hours=5.0, site_amplitude_prior={"pdf": "halfnormal", "sigma": 0.75})
    ordinary = build_shared_co2_o2_model(baseline, **kwargs, **noise)
    cached = build_shared_co2_o2_cached_sigma_model(baseline, **kwargs, **noise)
    point = _assert_linked_density_and_gradient_parity(ordinary, cached)
    for model in (ordinary, cached.model):
        assert sum(rv.name == "flux_scaling_latent" for rv in model.free_RVs) == 1
        evaluate = model.compile_fn(
            model.replace_rvs_by_values([model.flux_scaling]),
            inputs=model.value_vars, on_unused_input="ignore",
        )
        state_point = dict(point)
        assert state_point["flux_scaling_latent"].shape == (3,)
        log_states = []
        for z in (np.zeros(3), *np.eye(3)):
            state_point["flux_scaling_latent"] = z
            state = evaluate(state_point)[0]
            samples = xr.Dataset({"flux_scaling": xr.DataArray(state, dims="state", coords={"state": prepared["co2"].state})})
            for channel, frozen in prepared.items():
                samples[f"{channel}_bc_mean_shift"] = xr.DataArray(0.)
                dimension = f"{channel}_pattern"
                samples[f"{channel}_bc_pattern_standardized"] = xr.DataArray(
                    np.zeros(frozen.sizes["bc_pattern_term"]), dims=dimension,
                    coords={dimension: frozen.bc_pattern_term.values},
                )
            components = reconstruct_linked_shared_baseline(prepared, samples)
            log_states.append(np.log(state))
            for index, (channel, coefficient) in enumerate((("co2", 4.0), ("o2", -3.0))):
                raw, physical, centered = [components[channel][name].values for name in (
                    "outer_raw_correction", "outer_physical_contribution", "outer_centered_correction",
                )]
                np.testing.assert_allclose(raw, coefficient * (state[2] - 1.5), atol=1e-14)
                np.testing.assert_allclose(
                    physical, prepared[channel].physical_outer_prior + raw, atol=1e-14,
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


def test_linked_prior_identity_validation_survives_optimized_python():
    import subprocess
    import sys
    from pathlib import Path

    program = """
import sys
from dataclasses import replace
sys.path.insert(0, "tests")
from test_linked_shared_baseline import case
from test_rhime_co2_o2 import _independent_error
from openghg_inversions.rhime.co2.linked_shared_baseline_model import _linked_inputs
for name in ("original_prior_mean", "original_prior_covariance", "retained_mean", "retained_covariance",
             "sampled_outer", "original_outer_mean", "calibration_attrs", "mode", "calibration_reference"):
    frozen, base, _ = case("multiplicative_shrunk")
    data = frozen["o2"].data.copy(deep=True)
    if name in ("calibration_attrs", "mode"):
        data.attrs[name] = "different"
    elif name == "calibration_reference":
        data[name] = data.delta_sd
    else:
        value = data[name]
        data[name] = (~value if name == "sampled_outer" else value * 1.5)
    frozen["o2"] = replace(frozen["o2"], data=data)
    try:
        _linked_inputs(frozen, base.observations, _independent_error(base), aggregation_error=base.aggregation_error)
    except ValueError:
        pass
    else:
        raise RuntimeError("Mismatched linked prior was accepted: " + name)
"""
    subprocess.run([sys.executable, "-O", "-c", program], cwd=Path(__file__).parents[1], check=True)


@pytest.mark.parametrize("name", [
    "original_prior_mean", "original_prior_covariance", "retained_mean", "retained_covariance",
    "sampled_outer", "original_outer_mean", "calibration_attrs", "mode", "calibration_reference",
])
def test_all_linked_identities_reject_before_prior_construction(monkeypatch, name):
    import openghg_inversions.rhime.co2.linked_shared_baseline_model as recipe

    frozen, base, _ = case("multiplicative_shrunk")
    data = frozen["o2"].data.copy(deep=True)
    if name in ("calibration_attrs", "mode"):
        data.attrs[name] = "different"
    elif name == "calibration_reference":
        data[name] = xr.DataArray(1.0)
    else:
        value = data[name]
        data[name] = ~value if value.dtype == bool else value * 1.5
    frozen["o2"] = replace(frozen["o2"], data=data)

    def unexpected(*args, **kwargs):
        pytest.fail("Prior constructed before channel identities were compared")

    monkeypatch.setattr(recipe, "CorrelatedLognormalPrior", unexpected)
    with pytest.raises(ValueError, match="identical|identity|same resolved"):
        recipe._linked_inputs(frozen, base.observations, _independent_error(base), aggregation_error=base.aggregation_error)


def test_linked_inputs_construct_exactly_one_prior(monkeypatch):
    import openghg_inversions.rhime.co2.linked_shared_baseline_model as recipe

    frozen, base, _ = case()
    constructor = recipe.CorrelatedLognormalPrior
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return constructor(*args, **kwargs)

    monkeypatch.setattr(recipe, "CorrelatedLognormalPrior", counted)
    recipe._linked_inputs(frozen, base.observations, _independent_error(base), aggregation_error=base.aggregation_error)
    assert calls == [1]


@pytest.mark.parametrize("dimension", ["state", "original_state", "state_cov"])
def test_linked_prior_label_order_is_part_of_identity(dimension):
    from openghg_inversions.rhime.co2.linked_shared_baseline_model import _linked_inputs

    frozen, base, _ = case()
    # The covariance column dimension is carried explicitly by the bound data.
    if dimension == "state_cov":
        dimension = frozen["o2"].data.attrs["covariance_dim"]
    frozen["o2"] = replace(frozen["o2"], data=frozen["o2"].data.isel({dimension: slice(None, None, -1)}))
    with pytest.raises(ValueError, match="identical"):
        _linked_inputs(frozen, base.observations, _independent_error(base), aggregation_error=base.aggregation_error)


def test_linked_inputs_require_row_observation_units():
    """A missing per-row unit coordinate is a validation error, not an AttributeError."""
    import openghg_inversions.rhime.co2.linked_shared_baseline_model as recipe

    frozen, base, _ = case()
    observations = base.observations.drop_vars("observation_units")
    with pytest.raises(ValueError, match="observation_units"):
        recipe._linked_inputs(frozen, observations, _independent_error(base), aggregation_error=base.aggregation_error)
