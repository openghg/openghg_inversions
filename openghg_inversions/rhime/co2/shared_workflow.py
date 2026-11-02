"""Configured execution of the frozen linked atmospheric baseline recipe."""

from __future__ import annotations

import json

from dask import compute
import numpy as np
import xarray as xr

from openghg_inversions.array_ops import to_dense

from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.sampling import sample_rhime_model
from .co2_cached_sigma_runner import _sampler_for_cached_graph
from .co2_o2_runner import (
    _CO2_O2_VARIABLE_ROLES, _co2_o2_metadata, _validate_independent_error_labels,
    _validate_independent_error_values,
    annotate_co2_o2_trace,
)
from .linked_shared_baseline_model import (
    annotate_linked_shared_baseline_trace, build_shared_co2_o2_model,
    build_shared_co2_o2_cached_sigma_model,
)


def build_linked_shared_workflow(prepared, *, independent_error_sd=None, **likelihood):
    """Build the reviewed shared recipe from its frozen preparation.

    Related observation/error arrays cross the ordinary linked materialization
    boundary together. Inferred OU amplitudes select the maintained cache target.
    Returns the ordinary build result and an optional cached sampling target.
    """
    if not prepared.shared_baseline:
        raise ValueError("Linked shared execution requires frozen shared baseline inputs.")
    saved = prepared.independent_error_sd
    error = saved if independent_error_sd is None else independent_error_sd
    if error is None:
        raise ValueError("Supply independent_error_sd explicitly or in the prepared artifact.")
    errors = [error] if saved is None or saved is error else [error, saved]
    for value in errors:
        _validate_independent_error_labels(prepared.observations, value)
    # Frozen operators and observations may share a large lazy transport graph.
    # Compute them together once at this explicit PyMC construction boundary.
    observations, dense_errors, frozen = compute(
        to_dense(prepared.observations), [to_dense(value) for value in errors],
        {name: value.map(to_dense, keep_attrs=True) for name, value in prepared.shared_baseline.items()},
    )
    for value in dense_errors:
        _validate_independent_error_values(value)
        if not np.array_equal(value.observation_units.values, observations.observation_units.values):
            raise ValueError("independent_error_sd observation_units must match prepared observations.")
    error = dense_errors[0]
    if len(dense_errors) == 2 and not np.array_equal(error.values, dense_errors[1].values):
        raise ValueError("Explicit independent_error_sd must match the prepared artifact values.")
    kwargs = dict(observations=observations, independent_error_sd=error,
                  aggregation_error=prepared.aggregation_error, **likelihood)
    cached = None
    if likelihood.get("site_amplitude_prior") is not None:
        cached = build_shared_co2_o2_cached_sigma_model(frozen, **kwargs)
        model = cached.model
    else:
        model = build_shared_co2_o2_model(frozen, **kwargs)
    roles = dict(_CO2_O2_VARIABLE_ROLES)
    # Full physical states are added after sampling; selected flux_scaling remains
    # available unchanged to the reviewed pure atmospheric reconstructor.
    roles["flux_scale"] = "full_physical_flux_scaling"
    if likelihood.get("tau_hours") is not None:
        roles.update(independent_error="error", fixed_ou_timescale="ou_tau_hours",
                     fixed_ou_site_amplitude="ou_site_amplitude",
                     observation_to_fixed_ou_site_index="ou_site_index")
    metadata = _co2_o2_metadata(prepared, observations=observations)
    metadata.update(recipe="co2_o2_shared_baseline", outer_mode=prepared.shared_baseline["co2"].attrs["mode"],
                    baseline="independent physical CO2/O2 atmospheric deltas; frozen centered outer and BC patterns")
    if likelihood.get("tau_hours") is not None:
        metadata["likelihood"] = "joint Gaussian with fixed OU timescales by species/site"
        metadata["site_amplitudes"] = "inferred with matched cache" if cached is not None else "fixed"
    built = RhimeModelBuildResult(model, OutputContract(variable_roles=roles, metadata=metadata))
    return built, cached


def annotate_linked_shared_workflow(trace, *, built, prepared):
    """Restore normal linked labels/predictions and persist complete frozen context."""
    trace = annotate_co2_o2_trace(trace, built=built)
    trace = annotate_linked_shared_baseline_trace(trace, built.model)
    mean = prepared.retained_prior.mean
    state_dim = prepared.retained_prior.state_dim
    physical_dim = "full_physical_state" if state_dim == "physical_state" else "physical_state"
    trace.attrs["physical_state_dim"] = physical_dim
    rename = {name: physical_dim if name == state_dim else f"physical_{name}"
              for name, coord in mean.coords.items() if state_dim in coord.dims}
    rename[state_dim] = physical_dim
    for name in ("prior", "posterior"):
        if name not in trace.children:
            continue
        group = trace[name].to_dataset()
        selected = group.flux_scaling
        full = selected.reindex({state_dim: mean[state_dim]})
        # Removed outer states are fixed at their declared arithmetic prior mean.
        # Use bare mean coordinates here to avoid aligning selected auxiliaries.
        full = xr.where(mean[state_dim].isin(selected[state_dim]), full,
                        xr.DataArray(mean.data, dims=mean.dims, coords={state_dim: mean[state_dim]})).transpose(*selected.dims)
        full = full.drop_vars([key for key in full.coords if key in mean.coords and key != state_dim])
        full = full.assign_coords({key: coord for key, coord in mean.coords.items()}).rename(rename)
        full.attrs.update(units="dimensionless flux scale", fixed_outer_policy="original arithmetic prior mean")
        group["full_physical_flux_scaling"] = full
        trace[name] = xr.DataTree(group)
    trace.attrs["rhime_recipe"] = "co2_o2_shared_baseline"
    trace.attrs["rhime_variable_roles"] = json.dumps(dict(built.output_contract.variable_roles), sort_keys=True)
    for name, group in trace.children.items():
        if not name.startswith("shared_baseline_"):
            group.attrs["rhime_recipe"] = "co2_o2_shared_baseline"
    return trace


def sample_linked_shared_workflow(prepared, *, sampler, **model_options):
    """Sample with maintained compact deterministic reconstruction and OU cache."""
    built, cached = build_linked_shared_workflow(prepared, **model_options)
    active = sampler if cached is None else _sampler_for_cached_graph(
        sampler, cached_model=cached, sigma_target_accept=0.95, state_target_accept=0.95,
    )
    trace = sample_rhime_model(built, active, recompute_deterministics=True)
    return annotate_linked_shared_workflow(trace, built=built, prepared=prepared)
