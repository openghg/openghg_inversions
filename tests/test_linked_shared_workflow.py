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
from test_linked_shared_preparation import native, options, precision  # noqa: F401, F811


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
