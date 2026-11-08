"""Configured shared graph construction, annotation and saved-stage dispatch."""

from dataclasses import replace

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.rhime.co2.co2_o2_configuration import Co2O2ModelConfig, load_co2_o2_params, resolve_co2_o2_sampler
from openghg_inversions.rhime.co2.shared_preparation import prepare_linked_shared_inputs
from openghg_inversions.rhime.co2.shared_workflow import build_linked_shared_workflow, annotate_linked_shared_workflow
from test_linked_shared_outputs import saved  # noqa: F401
from test_linked_shared_preparation import gathered_native, native, options, precision  # noqa: F401, F811


def request(**kwargs):
    return dict(recipe="co2_o2_shared_baseline", channels={name: {"units": "ppm", "independent_error_sd": .1} for name in ("co2", "o2")}, **kwargs)


@pytest.mark.parametrize("mode", ["fixed", "multiplicative"])
def test_configured_shared_graph_and_full_physical_states(native, mode):  # noqa: F811
    prepared = prepare_linked_shared_inputs(native, **options(native, mode))
    config = Co2O2ModelConfig.from_params(load_co2_o2_params(params=request()))
    built, cached, prepared = build_linked_shared_workflow(prepared, **config.bind(prepared))
    assert cached is None
    assert "co2_bc_mean_shift" in built.model and "o2_bc_mean_shift" in built.model
    with built.model:
        trace = pm.sample_prior_predictive(2, random_seed=12)
    registry = get_coord_registry(built.model)
    trace = restore_inferencedata_coords(trace, registry)
    trace = annotate_linked_shared_workflow(trace, built=built, prepared=prepared)
    assert trace.attrs["rhime_recipe"] == "co2_o2_shared_baseline"
    from openghg_inversions.rhime.co2.shared_outputs import physical_flux_scaling

    assert "full_physical_flux_scaling" not in trace.prior
    assert "physical_state_dim" not in trace.attrs
    full = physical_flux_scaling(trace, "prior")
    assert full.sizes["state"] == prepared.retained_prior.mean.size
    xr.testing.assert_identical(full.state, prepared.retained_prior.mean.state)
    for channel in ("co2", "o2"):
        xr.testing.assert_identical(trace[f"shared_baseline_{channel}"].to_dataset(), prepared.shared_baseline[channel].data)
    removed = prepared.retained_prior.mean.state.values[~prepared.retained_prior.mean.state.isin(trace["prior"].state)]
    if len(removed):
        np.testing.assert_allclose(full.sel(state=removed).mean(("prior_chain", "prior_draw")).values,
                                   prepared.retained_prior.mean.sel(state=removed).values)



def test_recipe_mismatch_and_frozen_overrides_rejected(native):  # noqa: F811
    shared = prepare_linked_shared_inputs(native, **options(native))
    shared_config = Co2O2ModelConfig.from_params(request())
    with pytest.raises(ValueError, match="recipe"):
        shared_config.bind(native)
    ordinary = replace(shared_config, recipe="co2_o2")
    with pytest.raises(ValueError, match="recipe"):
        ordinary.bind(shared)
    for section in ({"biosphere": {"path": "unused.nc"}}, {"channels": {"co2": {"units": "ppm", "boundary": {"enabled": True}}, "o2": {"units": "ppm"}}}):
        with pytest.raises(ValueError, match="frozen preparation"):
            Co2O2ModelConfig.from_params({**request(), **section})


def test_shared_inferred_ou_configuration_binds_and_builds_cached_target(native):  # noqa: F811
    configured = Co2O2ModelConfig.from_params(request(likelihood={
        "kind": "fixed_ou", "tau_hours": 2., "site_amplitude_prior": {"pdf": "halfnormal", "sigma": .2},
        "initial_site_amplitudes": .1,
    }))
    assert configured.initial_site_amplitudes == .1
    assert configured.site_amplitude_prior["sigma"] == .2
    prepared = prepare_linked_shared_inputs(native, **options(native))
    bound = configured.bind(prepared)
    assert bound["site_amplitude_prior"] == configured.site_amplitude_prior
    assert bound["initial_site_amplitudes"] == .1
    built, cached, prepared = build_linked_shared_workflow(prepared, **bound)
    assert cached is not None
    assert built.model is cached.model
    # The positive amplitude transform performs an exp/log round trip.
    np.testing.assert_allclose(cached.initial_cache.sigma, .1, rtol=4 * np.finfo(float).eps, atol=0)


def test_shared_inferred_ou_resolves_a_valid_cached_sampler():
    params = request(likelihood={"kind": "fixed_ou", "tau_hours": 2.,
                                  "site_amplitude_prior": {"pdf": "halfnormal", "sigma": .2}})
    config = Co2O2ModelConfig.from_params(params)
    sampler = resolve_co2_o2_sampler(params, model=config)
    assert sampler.sample_kwargs["target_accept"] == .95
    tuned = resolve_co2_o2_sampler({**params, "sampling": {"target_accept": .9}}, model=config)
    assert tuned.sample_kwargs["target_accept"] == .9


@pytest.mark.parametrize("shared_preparation", [False, True])
def test_save_samples_rejects_cross_recipe_before_writing(saved, tmp_path, shared_preparation):  # noqa: F811
    from openghg_inversions.rhime.co2.co2_o2_handoffs import save_preparation, load_preparation, save_samples
    prepared, trace = saved
    if not shared_preparation:
        prepared = replace(prepared, shared_baseline={})
    else:
        trace = trace.copy(deep=True)
        trace.attrs["rhime_recipe"] = "co2_o2"
    manifest = save_preparation(prepared, output_dir=tmp_path / "prepare", start_date="2020-01-01",
                                end_date="2020-01-02", domain="EUROPE")
    loaded = load_preparation(manifest)
    destination = tmp_path / "sample"
    with pytest.raises(ValueError, match="recipe"):
        save_samples(manifest, trace, actual_inference={}, preparation_generation=loaded.generation,
                     preparation_inputs_id=loaded.inputs_id, output_dir=destination)
    assert not destination.exists()


@pytest.mark.parametrize("extension", ["nc", "zarr"])
@pytest.mark.parametrize("all_retained", [False, True])
def test_gathered_annotation_and_graph_free_product_replay(gathered_native, tmp_path, monkeypatch, extension, all_retained):  # noqa: F811
    from openghg_inversions.rhime.co2.co2_o2_outputs import (
        Co2O2OutputOptions, calculate_co2_o2_products, make_co2_o2_output,
    )
    from openghg_inversions.rhime.co2.shared_outputs import physical_flux_scaling
    from openghg_inversions.serialization import load_trace, save_trace

    kwargs = options(gathered_native)
    if all_retained:
        kwargs.update(outer=xr.zeros_like(kwargs["outer"]), sampled_outer=xr.zeros_like(kwargs["sampled_outer"]))
    prepared = prepare_linked_shared_inputs(gathered_native, **kwargs)
    config = Co2O2ModelConfig.from_params(request())
    built, _, prepared = build_linked_shared_workflow(prepared, **config.bind(prepared))
    with built.model:
        trace = pm.sample_prior_predictive(2, random_seed=12)
    trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
    # Label-only replay fixture: duplicate prior draws as a synthetic posterior.
    trace["posterior"] = xr.DataTree(trace.prior.to_dataset())
    trace = annotate_linked_shared_workflow(trace, built=built, prepared=prepared)
    expected = prepared.retained_prior.mean
    full = physical_flux_scaling(trace, "prior")
    xr.testing.assert_identical(full.state, expected.state)
    selected = trace.prior.flux_scaling.rename(chain="prior_chain", draw="prior_draw")
    xr.testing.assert_allclose(full.sel(state=selected.state), selected)
    removed = ~expected.state.isin(selected.state)
    xr.testing.assert_allclose(full.isel(state=removed), expected.isel(state=removed).broadcast_like(full.isel(state=removed)))
    path = tmp_path / f"samples.{extension}"
    save_trace(trace, path)
    restored = load_trace(path)
    monkeypatch.setattr(pm.Model, "__init__", lambda *a, **kw: pytest.fail("replay constructed a model"))
    monkeypatch.setattr(pm, "sample", lambda *a, **kw: pytest.fail("replay sampled"))
    products = []
    for samples in (trace, restored):
        result = make_co2_o2_output(prepared=prepared, trace=samples, start_date="2020-01-01",
                                   end_date="2020-01-02", domain="EUROPE")
        calculate_co2_o2_products(result=result, output=Co2O2OutputOptions())
        products.append(result.outputs)
        xr.testing.assert_identical(result.outputs["state_scaling"].state, expected.state)
        for channel in ("co2", "o2"):
            values = result.outputs[f"{channel}_concentration_components"]
            xr.testing.assert_allclose(values.modelled_concentration_prior,
                                       values.physical_atmospheric_total_prior - values.removed_mean_correction_prior)
    assert products[0].keys() == products[1].keys()
    for name in products[0]:
        xr.testing.assert_allclose(products[0][name], products[1][name])


@pytest.mark.parametrize("kind", ["additive", "fixed_gaussian", "unknown"])
def test_shared_configuration_rejects_unsupported_likelihoods(kind):
    with pytest.raises(ValueError, match="fixed_independent.*fixed_ou"):
        Co2O2ModelConfig.from_params(request(likelihood={"kind": kind}))


def test_shared_workflow_rejects_float32_before_materialization(native):  # noqa: F811
    import pytensor

    prepared = prepare_linked_shared_inputs(native, **options(native))
    with pytensor.config.change_flags(floatX="float32"):
        with pytest.raises(ValueError, match="floatX=float64"):
            build_linked_shared_workflow(prepared)


@pytest.mark.parametrize("entry", ["builder", "ordinary_runner", "cached_runner"])
def test_ordinary_linked_builder_rejects_shared_handoff_before_materialization(native, entry):  # noqa: F811
    from dask.callbacks import Callback
    from openghg_inversions.rhime.co2.co2_o2_runner import (
        build_rhime_co2_o2, run_rhime_co2_o2_from_prepared_inputs,
    )
    from openghg_inversions.rhime.co2.co2_o2_cached_sigma_runner import run_rhime_co2_o2_cached_sigma_from_prepared_inputs

    prepared = prepare_linked_shared_inputs(native, **options(native))
    prepared = replace(prepared, co2_sensitivity=prepared.co2_sensitivity.chunk(),
                       o2_sensitivity=prepared.o2_sensitivity.chunk())
    with Callback(pretask=lambda *a: pytest.fail("ordinary builder computed shared preparation")):
        with pytest.raises(ValueError, match="sample_linked_shared_workflow"):
            if entry == "builder":
                build_rhime_co2_o2(prepared_inputs=prepared)
            elif entry == "ordinary_runner":
                run_rhime_co2_o2_from_prepared_inputs(prepared_inputs=prepared)
            else:
                run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
                    prepared_inputs=prepared, tau_hours=2.,
                    site_amplitude_prior={"pdf": "halfnormal", "sigma": .2},
                )


@pytest.mark.parametrize("recipe,likelihood", [
    ("co2_o2", {"kind": "fixed_ou", "tau_hours": 2., "fixed_site_amplitudes": .1, "initial_site_amplitudes": .2}),
    ("co2_o2_shared_baseline", {"kind": "fixed_independent", "initial_site_amplitudes": .2}),
    ("co2_o2_shared_baseline", {"kind": "fixed_ou", "tau_hours": 2., "fixed_site_amplitudes": .1, "initial_site_amplitudes": .2}),
    ("co2_o2_shared_baseline", {"kind": "fixed_ou", "tau_hours": 2., "site_amplitude_prior_scale": .2}),
])
def test_likelihood_owner_rejects_unused_amplitude_options(recipe, likelihood):
    with pytest.raises(ValueError, match="unsupported|unused"):
        Co2O2ModelConfig.from_params({**request(likelihood=likelihood), "recipe": recipe})


def test_configured_target_accept_reaches_both_ordered_cached_steps(native, monkeypatch):  # noqa: F811
    from openghg_inversions.inference import cached_sigma
    from openghg_inversions.rhime.co2 import shared_workflow

    prepared = prepare_linked_shared_inputs(native, **options(native))
    params = request(likelihood={"kind": "fixed_ou", "tau_hours": 2.,
                                "site_amplitude_prior": {"pdf": "halfnormal", "sigma": .2}},
                     sampling={"target_accept": .87})
    configured = Co2O2ModelConfig.from_params(params)
    sampler = resolve_co2_o2_sampler(params, model=configured)
    calls = []
    state_step = cached_sigma.pm.NUTS

    def state(*args, **kwargs):
        names = [variable.name for variable in kwargs["vars"]]
        kind = "sigma" if len(names) == 1 and names[0].startswith("ou_site_amplitude") else "state"
        calls.append((kind, kwargs["target_accept"]))
        return state_step(*args, **kwargs)

    def sampled(built, active, *, recompute_deterministics):
        assert recompute_deterministics is True
        assert "target_accept" not in active.sample_kwargs
        assert len(active.sample_kwargs["step"].methods) == 2
        return xr.DataTree()

    monkeypatch.setattr(cached_sigma.pm, "NUTS", state)
    monkeypatch.setattr(shared_workflow, "sample_rhime_model", sampled)
    monkeypatch.setattr(shared_workflow, "annotate_linked_shared_workflow", lambda trace, **kw: trace)
    shared_workflow.sample_linked_shared_workflow(prepared, sampler=sampler, **configured.bind(prepared))
    assert calls == [("sigma", .87), ("state", .87)]
    assert sampler.sample_kwargs["target_accept"] == .87


@pytest.mark.parametrize("cached", [False, True])
def test_joint_source_is_reused_through_build_annotation_and_serialization(native, tmp_path, monkeypatch, cached):  # noqa: F811
    import dask.array as da
    from dask import delayed
    from openghg_inversions.serialization import load_trace, save_trace

    prepared = prepare_linked_shared_inputs(native, **options(native))
    prepared = replace(prepared, independent_error_sd=xr.full_like(prepared.observations, .1))
    calls = []

    @delayed
    def source():
        calls.append("source")
        return 1.

    @delayed
    def unused():
        pytest.fail("unused auxiliary diagnostic executed")

    token = da.from_delayed(source(), shape=(), dtype=float)

    def lazy(array):
        data = da.from_array(array.data)
        data = da.where(token == 1, data, False) if array.dtype == bool else data * token
        return array.copy(data=data)

    def frozen(dataset):
        return dataset.assign({name: lazy(array) for name, array in dataset.data_vars.items()})

    observations = lazy(prepared.observations).assign_coords(
        unused_diagnostic=("observation", da.from_delayed(unused(), shape=prepared.observations.shape, dtype=float)),
        observation_units=prepared.observations.observation_units.copy(
            data=da.where(token == 1, prepared.observations.observation_units.data, "bad")),
    )
    lazy_prepared = replace(
        prepared, observations=observations,
        fixed_prior_contribution=lazy(prepared.fixed_prior_contribution),
        co2_sensitivity=lazy(prepared.co2_sensitivity), o2_sensitivity=lazy(prepared.o2_sensitivity),
        boundary_operands={name: frozen(value) for name, value in prepared.boundary_operands.items()},
        shared_baseline={name: replace(value, data=frozen(value.data)) for name, value in prepared.shared_baseline.items()},
        aggregation_error=replace(prepared.aggregation_error, covariance=lazy(prepared.aggregation_error.covariance)),
        independent_error_sd=lazy(prepared.independent_error_sd),
    )
    error = lazy(prepared.independent_error_sd)  # distinct explicit and saved operands
    kwargs = dict(tau_hours=2., site_amplitude_prior={"pdf": "halfnormal", "sigma": .2}) if cached else {}
    from openghg_inversions.rhime.co2 import shared_workflow
    from openghg_inversions.rhime.sampling import RhimeSampler
    captured = {}
    build = shared_workflow.build_linked_shared_workflow

    def resolved(*args, **kwargs):
        built, target, materialized = build(*args, **kwargs)
        assert (target is not None) == cached
        captured["prepared"] = materialized
        return built, target, materialized

    def sampled(built, active, *, recompute_deterministics):
        assert recompute_deterministics
        with built.model:
            trace = pm.sample_prior_predictive(2, random_seed=18)
        return restore_inferencedata_coords(trace, get_coord_registry(built.model))

    monkeypatch.setattr(shared_workflow, "build_linked_shared_workflow", resolved)
    monkeypatch.setattr(shared_workflow, "sample_rhime_model", sampled)
    trace = shared_workflow.sample_linked_shared_workflow(
        lazy_prepared, sampler=RhimeSampler(nuts_sampler="pymc"), independent_error_sd=error, **kwargs,
    )
    materialized = captured["prepared"]
    assert calls == ["source"]
    assert isinstance(lazy_prepared.observations.data, da.Array)
    assert "unused_diagnostic" in lazy_prepared.observations.coords
    assert not isinstance(materialized.observations.data, da.Array)
    path = tmp_path / "resolved-trace.nc"
    save_trace(trace, path)
    restored = load_trace(path)
    assert calls == ["source"]
    for channel in ("co2", "o2"):
        xr.testing.assert_identical(restored[f"shared_baseline_{channel}"].to_dataset(), materialized.shared_baseline[channel].data)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("fault", ["values", "units"])
def test_error_override_identity_before_graph_construction(native, monkeypatch, cached, fault):  # noqa: F811
    from openghg_inversions.rhime.co2 import shared_workflow
    prepared = prepare_linked_shared_inputs(native, **options(native))
    prepared = replace(prepared, independent_error_sd=xr.full_like(prepared.observations, .1))
    error = prepared.independent_error_sd.chunk()
    error = (error * 2 if fault == "values" else error.assign_coords(
        observation_units=xr.full_like(error.observation_units, "ppb")))
    monkeypatch.setattr(shared_workflow, "build_shared_co2_o2_model", lambda *a, **kw: pytest.fail("built invalid graph"))
    monkeypatch.setattr(shared_workflow, "build_shared_co2_o2_cached_sigma_model", lambda *a, **kw: pytest.fail("built invalid graph"))
    kwargs = dict(tau_hours=2., site_amplitude_prior={"pdf": "halfnormal", "sigma": .2}) if cached else {}
    with pytest.raises(ValueError, match="values|observation_units"):
        build_linked_shared_workflow(prepared, independent_error_sd=error, **kwargs)


@pytest.mark.parametrize("mode", ["fixed", "multiplicative", "multiplicative_shrunk"])
def test_loaded_row_subsets_keep_frozen_science_and_construct_one_selected_prior(native, tmp_path, monkeypatch, mode):  # noqa: F811
    from openghg_inversions.rhime.co2.co2_o2_preparation import Co2O2PreparedInputs
    from openghg_inversions.rhime.co2 import linked_shared_baseline_model as recipe
    prepared = prepare_linked_shared_inputs(native, **options(native, mode))
    path = tmp_path / "prepared.nc"
    prepared.save(path)
    restored = Co2O2PreparedInputs.load(path)
    rows = [0, 2]  # first native receptor in each channel
    covariance = restored.aggregation_error.covariance.isel(observation=rows, observation_cov=rows)
    subset = replace(
        restored, observations=restored.observations.isel(observation=rows),
        fixed_prior_contribution=restored.fixed_prior_contribution.isel(observation=rows),
        co2_sensitivity=restored.co2_sensitivity.isel(co2_measure=[0]),
        o2_sensitivity=restored.o2_sensitivity.isel(o2_measure=[0]),
        boundary_operands={channel: value.isel(nmeasure=[0]) for channel, value in restored.boundary_operands.items()},
        shared_baseline={channel: replace(value, data=value.data.isel({value.data.attrs["native_observation_dim"]: [0]}))
                         for channel, value in restored.shared_baseline.items()},
        aggregation_error=replace(restored.aggregation_error, covariance=covariance,
                                  marginal_variance=restored.aggregation_error.marginal_variance[rows]),
        independent_error_sd=xr.full_like(restored.observations.isel(observation=rows), .1),
    )
    calls = []
    prior = recipe.CorrelatedLognormalPrior

    def selected(*args, **kwargs):
        calls.append(args[0].size)
        return prior(*args, **kwargs)

    monkeypatch.setattr(recipe, "CorrelatedLognormalPrior", selected)
    monkeypatch.setattr("openghg_inversions.biosphere_prior.calibrate_biosphere_prior", lambda *a, **kw: pytest.fail("recalibrated"))
    monkeypatch.setattr("openghg_inversions.models.shared_baseline.SharedBaselineChoice.bind", lambda *a, **kw: pytest.fail("prepared calendar again"))
    built, _, eager = build_linked_shared_workflow(subset)
    assert calls == [subset.shared_baseline["co2"].data.sizes["state"]]
    assert "co2_bc_mean_shift" in built.model and "o2_bc_mean_shift" in built.model
    assert not any("baseline_delta" in name for name in built.model.named_vars)
    for channel in ("co2", "o2"):
        for name in ("reference", "centering_weights", "bc_pattern_mean", "outer_column_mean"):
            expected = subset.shared_baseline[channel].data[name]
            xr.testing.assert_identical(eager.shared_baseline[channel].data[name], expected)


@pytest.mark.parametrize("cached", [False, True])
def test_nonunit_mass_absolute_predictives_and_snapshot_replay(native, tmp_path, cached):  # noqa: F811
    from openghg_inversions.rhime.co2.co2_o2_preparation import Co2O2PreparedInputs
    from openghg_inversions.rhime.co2.co2_o2_model import _gather_co2_o2_sensitivity
    from openghg_inversions.postprocessing.boundary_outputs import restore_absolute_predictives
    from openghg_inversions.serialization import load_trace, save_trace

    raw = {}
    for channel, value in native.boundary_operands.items():
        scale = xr.DataArray([.5, 2., 1.5][:value.sizes["nmeasure"]], dims="nmeasure",
                             coords={"nmeasure": value.nmeasure})
        raw[channel] = value.assign(H_bc_mass=(value.H_bc_mass * scale).assign_attrs(units="1"))
    order = [2, 0, 3, 1, 4]
    covariance = native.aggregation_error.covariance.isel(observation=order, observation_cov=order)
    native = replace(native, boundary_operands=raw,
                     observations=native.observations.isel(observation=order),
                     fixed_prior_contribution=native.fixed_prior_contribution.isel(observation=order),
                     aggregation_error=replace(native.aggregation_error, covariance=covariance,
                                               marginal_variance=native.aggregation_error.marginal_variance[order]))
    prepared = prepare_linked_shared_inputs(native, **options(native))
    snapshot = tmp_path / "nonunit.nc"
    prepared.save(snapshot)
    prepared = Co2O2PreparedInputs.load(snapshot)
    expected_shifts = []
    for channel in ("co2", "o2"):
        value = raw[channel]
        row = str(getattr(native, f"{channel}_sensitivity").dims[0])
        expected = (value.bc_reference * value.H_bc_mass.sum("bc_region")).rename(nmeasure=row)
        np.testing.assert_allclose(prepared.shared_baseline[channel].data.bc_reference_shift, expected)
        expected_shifts.append(expected)
    expected_shift = _gather_co2_o2_sensitivity(*expected_shifts).sel(observation=native.observations.observation)
    kwargs = dict(tau_hours=2., site_amplitude_prior={"pdf": "halfnormal", "sigma": .2}) if cached else {}
    built, _, materialized = build_linked_shared_workflow(prepared, independent_error_sd=xr.full_like(prepared.observations, .1), **kwargs)
    with built.model:
        trace = pm.sample_prior_predictive(3, random_seed=128)
    trace = restore_inferencedata_coords(trace, get_coord_registry(built.model))
    raw_y = trace.prior_predictive.y.copy(deep=True)
    trace["posterior_predictive"] = xr.DataTree(xr.Dataset({"y": raw_y.isel(draw=[0, 1]).assign_coords(draw=[7, 9])}))
    prediction = raw_y.isel(draw=[2], observation=[3, 0]).assign_coords(draw=[41])
    trace["predictions"] = xr.DataTree(xr.Dataset({"y": prediction}))
    prediction_shift = (expected_shift.isel(observation=[3, 0]) * 2).rename("boundary_reference_shift")
    trace["predictions_constant_data"] = xr.DataTree(prediction_shift.to_dataset())
    trace["log_likelihood"] = xr.DataTree(xr.Dataset({"y": xr.DataArray([[-3., -4.]], dims=("chain", "draw"))}))
    trace["log_prior"] = xr.DataTree(xr.Dataset({"joint": xr.DataArray([[-1., -2.]], dims=("chain", "draw"))}))
    raw_observed = trace.observed_data.y.copy(deep=True)
    absolute_mean = trace.prior.modelled_concentration.copy(deep=True)
    likelihood = trace.log_likelihood.to_dataset().copy(deep=True)
    prior_density = trace.log_prior.to_dataset().copy(deep=True)
    trace = annotate_linked_shared_workflow(trace, built=built, prepared=materialized)
    np.testing.assert_allclose(trace.observed_data.y, native.observations)
    np.testing.assert_allclose(raw_observed + expected_shift, native.observations)
    np.testing.assert_allclose(trace.prior_predictive.y, raw_y + expected_shift)
    np.testing.assert_allclose(trace.posterior_predictive.y, raw_y.isel(draw=[0, 1]).data + expected_shift.data)
    np.testing.assert_allclose(trace.predictions.y, prediction + prediction_shift)
    xr.testing.assert_identical(trace.prior.modelled_concentration, absolute_mean.assign_attrs(trace.prior.modelled_concentration.attrs))
    xr.testing.assert_equal(trace.log_prior.to_dataset(), prior_density)
    np.testing.assert_array_equal(trace.log_likelihood.y, likelihood.y)
    for group in ("observed_data", "prior_predictive", "posterior_predictive", "predictions"):
        assert trace[group].y.attrs["concentration_representation"] == "absolute-v1"
    once = trace.copy(deep=True)
    trace = annotate_linked_shared_workflow(trace, built=built, prepared=materialized)
    xr.testing.assert_identical(trace, once)
    path = tmp_path / "absolute.nc"
    save_trace(trace, path)
    reopened = load_trace(path)
    restore_absolute_predictives(reopened)
    for group in ("observed_data", "prior_predictive", "posterior_predictive", "predictions"):
        xr.testing.assert_identical(reopened[group].y, once[group].y)
    xr.testing.assert_identical(prepared.observations, native.observations)
