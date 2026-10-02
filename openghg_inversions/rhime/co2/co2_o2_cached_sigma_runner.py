"""Run the linked fixed-OU graph with the existing CO2 CompoundStep."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
    _materialize_co2_o2_replay_inputs,
)


def run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
    *,
    prepared_inputs: Co2O2PreparedInputs,
    independent_error_sd: xr.DataArray | None = None,
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
) -> xr.DataTree:
    """Sample linked species/site OU amplitudes followed by affine states.

    One joint likelihood retains the prepared cross-channel aggregation
    covariance and adds independent errors and OU blocks grouped by species
    and site. Both channels must use the same concentration units. Inputs
    remain borrowed; related payloads are materialized together before graph
    construction. The runner constructs its sigma-then-state CompoundStep,
    refreshing the accepted amplitude cache before each affine-state update.

    Args:
        prepared_inputs: Linked handoff containing joint labelled observations,
            native-channel sensitivities, affine contribution, retained prior,
            joint aggregation covariance, ratio provenance, and optional
            channel boundary sensitivities. Observations require aligned
            species, site, time, and observation_units coordinates.
        independent_error_sd: Optional finite positive fixed standard deviations
            on the joint observation axis, in its concentration units. Labels
            and observation_units must match the prepared observations. When
            omitted, use the artifact's saved error vector. Explicit and saved
            vectors must agree in labels, units, and values when both exist.
        tau_hours: Finite positive fixed OU timescale in hours, applied to each
            species/site group. A mapping uses co2:SITE and o2:SITE keys and
            must cover every observed group.
        site_amplitude_prior_scale: Finite positive HalfNormal scale for the
            independent species/site OU amplitudes, in concentration units.
        initial_site_amplitudes: Optional finite positive starting amplitudes,
            in concentration units. A scalar applies to every group; mappings
            use the same keys as tau_hours. Defaults to the prior scale.
        state_activity: Optional labelled active/fixed retained-state policy.
            Omitted states follow the model's structural activity policy.
        use_bc: Optional co2/o2 boolean mapping selecting prepared boundary
            sensitivities. Omitted entries include available boundaries;
            selecting a missing channel boundary fails.
        bc_prior: Channel-keyed dimensionless boundary-scale priors. Enabled
            channels use the ordinary CO2 boundary prior when omitted.
        bc_state_activity: Optional channel-keyed labelled active/fixed
            boundary-state policies. Requires the selected channel boundary.
        offset_prior: Channel-keyed offset priors in concentration units.
            Channels omitted from this mapping have no offset.
        offset_args: Per-channel per_site, offset_freq, and drop_first options
            with the ordinary CO2 offset meanings. Requires an offset prior
            for each configured channel; the default is one offset per site.
        sampler: Optional sampling configuration, defaulting to PyMC. Only the
            PyMC backend is supported. Caller-supplied step or target_accept
            controls are rejected because the runner owns both NUTS steps.
            Joint predictive output supports only random_seed in predictive
            keyword arguments and is drawn from the same numerical target.
        sigma_target_accept: Target acceptance probability for the amplitude
            NUTS step, defaulting to 0.8.
        state_target_accept: Target acceptance probability for the affine-state
            NUTS step, defaulting to 0.9.

    Returns:
        Restored DataTree with posterior samples, labelled observed data and
        constants, scientific roles, concentration units, and JSON model
        metadata. The log_likelihood group contains one normalized joint value
        per chain/draw, not pointwise likelihoods. Prior groups and joint
        posterior_predictive observation vectors are included when requested
        by the sampler. No artifact is written to disk.

    Raises:
        ValueError: If independent errors are missing, invalid, mislabelled,
            or conflict with saved errors; channel units or grouping
            coordinates are invalid; selected boundary/offset settings are
            inconsistent; OU scales or amplitudes are invalid; or the sampler
            requests an unsupported backend, step, tuning control, or
            predictive keyword. Label, prior, covariance, and activity errors
            from model construction also propagate.
        TypeError: If channel offset frequency or boolean options have
            unsupported types.
    """
    requested_sampler = sampler or RhimeSampler(nuts_sampler="pymc")
    if requested_sampler.nuts_sampler != "pymc":
        raise ValueError("Linked cached fixed-OU requires nuts_sampler='pymc'.")
    prepared = prepared_inputs
    boundaries = dict(getattr(prepared, "boundary_sensitivity", {}))
    if use_bc is not None:
        if (
            not isinstance(use_bc, Mapping)
            or set(use_bc) - {"co2", "o2"}
            or any(not isinstance(value, bool) for value in use_bc.values())
        ):
            raise ValueError("use_bc must map co2/o2 to booleans.")
        for channel, enabled in use_bc.items():
            if enabled and channel not in boundaries:
                raise ValueError(f"{channel} use_bc requires prepared boundary_sensitivity.")
            if not enabled:
                boundaries.pop(channel, None)
    materialized, boundaries = _materialize_co2_o2_replay_inputs(
        prepared, independent_error_sd, boundaries
    )
    observations, fixed, co2, o2, error = materialized
    cached = build_co2_o2_cached_sigma_model(
        observations=observations,
        fixed_prior_contribution=fixed,
        co2_sensitivity=co2,
        o2_sensitivity=o2,
        aggregation_error=prepared.aggregation_error,
        retained_prior=prepared.retained_prior,
        independent_error_sd=error,
        tau_hours=tau_hours,
        site_amplitude_prior_scale=site_amplitude_prior_scale,
        initial_site_amplitudes=initial_site_amplitudes,
        state_activity=state_activity,
        boundary_sensitivity=boundaries,
        bc_prior=bc_prior,
        bc_state_activity=bc_state_activity,
        offset_prior=offset_prior,
        offset_args=offset_args,
    )
    metadata = _co2_o2_metadata(prepared, observations=observations)
    metadata.update(
        recipe="co2_o2_cached_sigma_fixed_ou",
        likelihood="joint Gaussian with fixed OU blocks by (species, site)",
    )
    roles = dict(_CO2_O2_VARIABLE_ROLES)
    roles.update(
        independent_error="error",
        fixed_ou_site_amplitude="ou_site_amplitude",
        observation_to_fixed_ou_site_index="ou_site_index",
        fixed_ou_timescale="ou_tau_hours",
    )
    for channel in ("co2", "o2"):
        if channel in boundaries:
            roles.update(
                {
                    f"{channel}_boundary_concentration": f"{channel}_mu_bc",
                    f"{channel}_boundary_scale": f"{channel}_bc",
                    f"{channel}_boundary_sensitivity": f"{channel}_hbc",
                }
            )
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
        requested_sampler,
        cached_model=cached,
        sigma_target_accept=sigma_target_accept,
        state_target_accept=state_target_accept,
    )
    trace = sample_rhime_model(built, sampling_sampler)
    trace = _append_joint_outputs(
        trace,
        cached_model=cached,
        observations=observations,
        posterior_predictive=_posterior_predictive_requested(requested_sampler),
        random_seed=_predictive_seed(requested_sampler),
    )
    trace = _annotate_co2_o2_trace(trace, built=built)
    trace = _annotate_linked_fixed_ou_trace(trace, observations=observations)
    trace.attrs["rhime_recipe"] = "co2_o2_cached_sigma_fixed_ou"
    for group in trace.children.values():
        group.attrs["rhime_recipe"] = "co2_o2_cached_sigma_fixed_ou"
    return trace
