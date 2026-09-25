"""Run the linked fixed-OU graph with the existing CO2 CompoundStep."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import arviz as az
import numpy as np
import xarray as xr

from openghg_inversions.models import StateActivity
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model

from .co2_cached_sigma_runner import (
    _append_joint_outputs,
    _posterior_predictive_requested,
    _predictive_seed,
    _sampler_for_cached_graph,
)
from .co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from .co2_o2_preparation import Co2O2PreparedInputs
from .co2_o2_runner import (
    _CO2_O2_VARIABLE_ROLES,
    _annotate_co2_o2_trace,
    _annotate_linked_fixed_ou_trace,
    _co2_o2_metadata,
    _materialize_co2_o2_pymc_inputs,
    _validate_independent_error_labels,
    _validate_independent_error_values,
)


def run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
    *,
    prepared_inputs: Co2O2PreparedInputs,
    independent_error_sd: xr.DataArray,
    tau_hours: float | Mapping[str, float],
    site_amplitude_prior_scale: float,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
    state_activity: StateActivity | None = None,
    use_bc: Mapping[str, bool] | None = None,
    bc_prior: Mapping[str, PriorArgs] | None = None,
    bc_state_activity: Mapping[str, StateActivity] | None = None,
    offset_prior: Mapping[str, PriorArgs] | None = None,
    offset_args: Mapping[str, Mapping[str, Any]] | None = None,
    sampler: RhimeSampler | None = None,
    sigma_target_accept: float = 0.8,
    state_target_accept: float = 0.9,
) -> az.InferenceData:
    """Sample linked species/site OU amplitudes followed by affine states.

    Uses the prepared-input, independent-error and channel baseline contracts
    of :func:`run_rhime_co2_o2_from_prepared_inputs`. Both channels must use the
    same concentration units. Fixed ``tau_hours`` and optional positive initial
    amplitudes accept scalars or mappings keyed by ``co2:SITE``/``o2:SITE``.
    The HalfNormal ``site_amplitude_prior_scale`` is in concentration units.
    The two target-accept settings tune the amplitude and state NUTS steps.

    Only the PyMC backend is supported. The runner owns its CompoundStep and
    attaches normalized joint log likelihoods and optional joint predictive
    vectors from the same numerical target used during sampling. Predictive
    keyword arguments support only ``random_seed``.
    """
    requested_sampler = sampler or RhimeSampler(nuts_sampler="pymc")
    if requested_sampler.nuts_sampler != "pymc":
        raise ValueError("Linked cached fixed-OU requires nuts_sampler='pymc'.")
    prepared = prepared_inputs
    _validate_independent_error_labels(prepared.observations, independent_error_sd)
    boundaries = dict(getattr(prepared, "boundary_sensitivity", {}))
    if use_bc is not None:
        if not isinstance(use_bc, Mapping) or set(use_bc) - {"co2", "o2"} or any(
            not isinstance(value, bool) for value in use_bc.values()
        ):
            raise ValueError("use_bc must map co2/o2 to booleans.")
        for channel, enabled in use_bc.items():
            if enabled and channel not in boundaries:
                raise ValueError(f"{channel} use_bc requires prepared boundary_sensitivity.")
            if not enabled:
                boundaries.pop(channel, None)
    materialized = _materialize_co2_o2_pymc_inputs(
        prepared.observations, prepared.fixed_prior_contribution,
        prepared.co2_sensitivity, prepared.o2_sensitivity, independent_error_sd,
        *boundaries.values(),
    )
    observations, fixed, co2, o2, error = materialized[:5]
    boundaries = dict(zip(boundaries, materialized[5:], strict=True))
    if not np.array_equal(error.observation_units.values, observations.observation_units.values):
        raise ValueError("independent_error_sd observation_units must match the prepared observations.")
    _validate_independent_error_values(error)
    cached = build_co2_o2_cached_sigma_model(
        observations=observations, fixed_prior_contribution=fixed,
        co2_sensitivity=co2, o2_sensitivity=o2,
        aggregation_error=prepared.aggregation_error, retained_prior=prepared.retained_prior,
        independent_error_sd=error, tau_hours=tau_hours,
        site_amplitude_prior_scale=site_amplitude_prior_scale,
        initial_site_amplitudes=initial_site_amplitudes, state_activity=state_activity,
        boundary_sensitivity=boundaries, bc_prior=bc_prior,
        bc_state_activity=bc_state_activity, offset_prior=offset_prior, offset_args=offset_args,
    )
    metadata = _co2_o2_metadata(prepared, observations=observations)
    metadata.update(recipe="co2_o2_cached_sigma_fixed_ou",
                    likelihood="joint Gaussian with fixed OU blocks by (species, site)")
    roles = dict(_CO2_O2_VARIABLE_ROLES)
    roles.update(independent_error="error", fixed_ou_site_amplitude="ou_site_amplitude",
                 observation_to_fixed_ou_site_index="ou_site_index", fixed_ou_timescale="ou_tau_hours")
    for channel in ("co2", "o2"):
        if channel in boundaries:
            roles.update({
                f"{channel}_boundary_concentration": f"{channel}_mu_bc",
                f"{channel}_boundary_scale": f"{channel}_bc",
                f"{channel}_boundary_sensitivity": f"{channel}_hbc",
            })
        if channel in (offset_prior or {}):
            roles[f"{channel}_offset_concentration"] = f"{channel}_offset"
    if boundaries:
        roles["boundary_concentration"] = "boundary_concentration"
    if offset_prior:
        roles["offset_concentration"] = "offset_concentration"
    if boundaries or offset_prior:
        roles["baseline_concentration"] = "baseline_concentration"
    built = RhimeModelBuildResult(model=cached.model, variable_roles=roles, metadata=metadata)
    sampling_sampler = _sampler_for_cached_graph(
        requested_sampler, cached_model=cached,
        sigma_target_accept=sigma_target_accept, state_target_accept=state_target_accept,
    )
    trace = sample_rhime_model(built, sampling_sampler)
    trace = _append_joint_outputs(
        trace, cached_model=cached, observations=observations,
        posterior_predictive=_posterior_predictive_requested(requested_sampler),
        random_seed=_predictive_seed(requested_sampler),
    )
    trace = _annotate_co2_o2_trace(trace, built=built)
    trace = _annotate_linked_fixed_ou_trace(trace, observations=observations)
    trace.attrs["rhime_recipe"] = "co2_o2_cached_sigma_fixed_ou"
    for group_name in trace.groups():
        getattr(trace, group_name).attrs["rhime_recipe"] = "co2_o2_cached_sigma_fixed_ou"
    return trace
