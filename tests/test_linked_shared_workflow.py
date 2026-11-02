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
    built, cached = build_linked_shared_workflow(prepared, **config.bind(prepared))
    assert cached is None
    assert "co2_bc_level" in built.model and "o2_bc_level" in built.model
    with built.model:
        trace = pm.sample_prior_predictive(2, random_seed=12)
    registry = get_coord_registry(built.model)
    trace = restore_inferencedata_coords(trace, registry)
    trace = annotate_linked_shared_workflow(trace, built=built, prepared=prepared)
    assert trace.attrs["rhime_recipe"] == "co2_o2_shared_baseline"
    full = trace["prior"].full_physical_flux_scaling
    assert full.sizes["physical_state"] == prepared.retained_prior.mean.size
    for channel in ("co2", "o2"):
        xr.testing.assert_identical(trace[f"shared_baseline_{channel}"].to_dataset(), built.model.linked_shared_baseline[channel])
    removed = prepared.retained_prior.mean.state.values[~prepared.retained_prior.mean.state.isin(trace["prior"].state)]
    if len(removed):
        np.testing.assert_allclose(full.sel(physical_state=removed).mean(("chain", "draw")).values,
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


def test_shared_inferred_ou_configuration():
    configured = Co2O2ModelConfig.from_params(request(likelihood={
        "kind": "fixed_ou", "tau_hours": 2., "site_amplitude_prior": {"pdf": "halfnormal", "sigma": .2},
        "initial_site_amplitudes": .1,
    }))
    assert configured.initial_site_amplitudes == .1
    assert configured.site_amplitude_prior["sigma"] == .2


def test_shared_inferred_ou_resolves_a_valid_cached_sampler():
    from openghg_inversions.rhime.co2.co2_cached_sigma_runner import validate_cached_co2_sampler
    params = request(likelihood={"kind": "fixed_ou", "tau_hours": 2.,
                                  "site_amplitude_prior": {"pdf": "halfnormal", "sigma": .2}})
    config = Co2O2ModelConfig.from_params(params)
    sampler = resolve_co2_o2_sampler(params, model=config)
    validate_cached_co2_sampler(sampler)
    with pytest.raises(ValueError, match="omit sampling.target_accept"):
        resolve_co2_o2_sampler({**params, "sampling": {"target_accept": .9}}, model=config)


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
    built, _ = build_linked_shared_workflow(prepared, **config.bind(prepared))
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
