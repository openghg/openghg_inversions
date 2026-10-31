"""Run the linked fixed-OU graph with the existing CO2 CompoundStep."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import xarray as xr

from openghg_inversions.postprocessing.contracts import OutputContract

from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models import StateActivity
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.inference.sampling import resolve_sampler
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model

from .co2_cached_sigma_runner import (
    _sampler_for_cached_graph,
)
from openghg_inversions.models.boundary_corrections import _boundary_correction_roles
from openghg_inversions.models.boundary import require_centered_precision
from .co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from .co2_o2_model import _validate_channel_baseline_options
from .co2_o2_preparation import Co2O2PreparedInputs
from .co2_o2_runner import (
    _CO2_O2_VARIABLE_ROLES,
    _add_additive_boundary_roles,
    annotate_co2_o2_trace,
    _co2_o2_metadata,
    _materialize_co2_o2_replay_inputs,
)


def run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
    *,
    prepared_inputs: Co2O2PreparedInputs,
    independent_error_sd: xr.DataArray | None = None,
    tau_hours: float | Mapping[str, float],
    site_amplitude_prior: PriorArgs,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
    state_activity: StateActivity | None = None,
    use_bc: Mapping[str, bool] | None = None,
    boundary_correction: Mapping[str, xr.Dataset] | None = None,
    bc_mean_shift_prior_co2: PriorArgs | None = None,
    bc_mean_shift_prior_o2: PriorArgs | None = None,
    bc_anomaly_scale_co2: float | None = None,
    bc_anomaly_scale_o2: float | None = None,
    boundary_correction_sensitivity_co2: xr.DataArray | None = None,
    boundary_correction_sensitivity_o2: xr.DataArray | None = None,
    bc_centering_weights_co2: xr.DataArray | None = None,
    bc_centering_weights_o2: xr.DataArray | None = None,
    bc_prior: Mapping[str, PriorArgs] | None = None,
    bc_state_activity: Mapping[str, StateActivity] | None = None,
    bc_freq: str | None = None,
    bc_anchor_time: DatetimeLike | None = None,
    offset_prior: Mapping[str, PriorArgs] | None = None,
    offset_args: Mapping[str, Mapping[str, Any]] | None = None,
    sampler: RhimeSampler | Literal["default"] = "default",
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

    Selected centered boundary operands currently require PyTensor
    ``floatX=float64``. This is a restriction of the current implementation.
    Observations, predictions, and boundary contributions are restored to
    absolute concentrations; see the
    :ref:`centered boundary contract <surface-centered-boundary-contract>`.

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
        site_amplitude_prior: Independent positive prior specification for the
            independent species/site OU amplitudes, in concentration units.
        initial_site_amplitudes: Optional finite positive starting amplitudes,
            in concentration units. A scalar applies to every group; mappings
            use the same keys as tau_hours. Defaults to the prior support point.
        state_activity: Optional labelled active/fixed retained-state policy.
            Omitted states follow the model's structural activity policy.
        use_bc: Optional co2/o2 boolean mapping selecting prepared boundary
            sensitivities. Omitted entries include available boundaries;
            selecting a missing channel boundary fails.
        bc_prior: Channel-keyed dimensionless boundary-scale priors. Enabled
            channels use the ordinary CO2 boundary prior when omitted.
        boundary_correction: Mapping of channel to independently prepared
            physical level/pattern correction Dataset. Replaces the saved
            boundary on selected channels; omit use_bc on those channels.
            References, widths and periods remain channel-specific.
        bc_mean_shift_prior_co2: Optional scalar CO2 additive mean-shift prior,
            in its observation concentration units. Requires bc_anomaly_scale_co2
            and replaces that channel's multiplicative prior/activity.
        bc_mean_shift_prior_o2: Corresponding independent O2 mean-shift prior.
        bc_anomaly_scale_co2: Finite, nonnegative CO2 Gaussian scale before
            weighted centring, in concentration units, including every supplied
            curtain/period state. Zero retains only the common mean without
            sampled contrasts.
        bc_anomaly_scale_o2: Corresponding independent O2 anomaly scale; zero
            likewise omits O2 contrasts.
        boundary_correction_sensitivity_co2: Dimensionless CO2 unit transport on
            its native observation and boundary-state axes. Centered operands
            default to their explicitly supplied H_bc_mass; no division by the
            concentration reference is used.
        boundary_correction_sensitivity_o2: Corresponding O2 unit transport.
        bc_centering_weights_co2: Positive dimensionless weights with exactly the
            resolved CO2 curtain/period labels, normalized within the model.
        bc_centering_weights_o2: Corresponding O2 centring weights. Prepared-input
            runners use saved correction sensitivities and weights when omitted.
        bc_state_activity: Optional channel-keyed labelled active/fixed
            boundary-state policies. Requires the selected channel boundary.
        bc_freq: Optional period for raw ordinary or centered channel boundaries.
        bc_anchor_time: Optional origin for fixed-duration boundary periods.
        offset_prior: Channel-keyed offset priors in concentration units.
            Channels omitted from this mapping have no offset.
        offset_args: Per-channel per_site, offset_freq, and drop_first options
            with the ordinary CO2 offset meanings. Requires an offset prior
            for each configured channel; the default is one offset per site.
        sampler: Sampler configuration or ``"default"`` for PyMC; ``None`` is invalid. Only the
            PyMC backend is supported. Caller-supplied step or target_accept
            controls are rejected because the runner owns both NUTS steps.
            Joint predictive output supports only random_seed in predictive
            keyword arguments and uses the exact observed PyMC distribution.
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
            inconsistent; selected centered boundaries use a floatX other
            than float64; OU scales or amplitudes are invalid; or the sampler
            requests an unsupported backend, step, tuning control, or
            predictive keyword. Label, prior, covariance, and activity errors
            from model construction also propagate.
        TypeError: If channel offset frequency or boolean options have
            unsupported types, or the sampler is neither a configuration nor
            ``"default"``.
    """
    requested_sampler = resolve_sampler(sampler, nuts_sampler="pymc")
    if requested_sampler.nuts_sampler != "pymc":
        raise ValueError("Linked cached fixed-OU requires nuts_sampler='pymc'.")
    prepared = prepared_inputs
    boundaries = dict(getattr(prepared, "boundary_sensitivity", {}))
    operands = dict(prepared.boundary_operands)
    for channel, response, weights in (
        ("co2", boundary_correction_sensitivity_co2, bc_centering_weights_co2),
        ("o2", boundary_correction_sensitivity_o2, bc_centering_weights_o2),
    ):
        if channel in (boundary_correction or {}) and (response is not None or weights is not None):
            raise ValueError("Independent corrections replace weighted additive response/weights.")
    for channel in (boundary_correction or {}):
        if (use_bc or {}).get(channel, False):
            raise ValueError("boundary_correction supplies physical BC; omit use_bc on that channel.")
        boundaries.pop(channel, None)
        operands.pop(channel, None)
    if use_bc is not None:
        if (
            not isinstance(use_bc, Mapping)
            or set(use_bc) - {"co2", "o2"}
            or any(not isinstance(value, bool) for value in use_bc.values())
        ):
            raise ValueError("use_bc must map co2/o2 to booleans.")
    if use_bc is not None:
        for channel, enabled in use_bc.items():
            if enabled and channel not in boundaries and channel not in operands:
                raise ValueError(f"{channel} use_bc requires prepared boundary_sensitivity.")
            if not enabled:
                boundaries.pop(channel, None)
                operands.pop(channel, None)
    _validate_channel_baseline_options(
        boundaries,
        bc_prior,
        bc_state_activity,
        offset_prior,
        offset_args,
        operands,
        boundary_correction=boundary_correction,
        bc_mean_shift_prior_co2=bc_mean_shift_prior_co2,
        bc_mean_shift_prior_o2=bc_mean_shift_prior_o2,
        bc_anomaly_scale_co2=bc_anomaly_scale_co2,
        bc_anomaly_scale_o2=bc_anomaly_scale_o2,
    )
    correction_inputs = {}
    prepared_corrections = getattr(prepared, "boundary_correction_sensitivity", {})
    prepared_weights = getattr(prepared, "bc_centering_weights", {})
    for channel, mean_prior, explicit_correction, explicit_weights in (
        ("co2", bc_mean_shift_prior_co2, boundary_correction_sensitivity_co2, bc_centering_weights_co2),
        ("o2", bc_mean_shift_prior_o2, boundary_correction_sensitivity_o2, bc_centering_weights_o2),
    ):
        if mean_prior is None:
            continue
        correction = (
            explicit_correction if explicit_correction is not None else prepared_corrections.get(channel)
        )
        weights = explicit_weights if explicit_weights is not None else prepared_weights.get(channel)
        if correction is not None:
            correction_inputs[f"{channel}_correction"] = correction
        if weights is not None:
            correction_inputs[f"{channel}_weights"] = weights
    for channel, correction_prior in (boundary_correction or {}).items():
        correction_inputs.update({f"{channel}_prior_{name}": value for name, value in correction_prior.data_vars.items()})
    if operands:
        require_centered_precision()
    materialized, boundaries, operands = _materialize_co2_o2_replay_inputs(
        prepared, independent_error_sd, {**boundaries, **correction_inputs}, operands
    )
    boundary_correction_sensitivity_co2 = boundaries.pop("co2_correction", None)
    boundary_correction_sensitivity_o2 = boundaries.pop("o2_correction", None)
    bc_centering_weights_co2 = boundaries.pop("co2_weights", None)
    bc_centering_weights_o2 = boundaries.pop("o2_weights", None)
    if boundary_correction is not None:
        boundary_correction = {
            channel: xr.Dataset(
                {name: boundaries.pop(f"{channel}_prior_{name}") for name in raw.data_vars}, attrs=raw.attrs,
            ) for channel, raw in boundary_correction.items()
        }
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
        site_amplitude_prior=site_amplitude_prior,
        initial_site_amplitudes=initial_site_amplitudes,
        state_activity=state_activity,
        boundary_sensitivity=boundaries,
        boundary_operands=operands,
        bc_freq=bc_freq,
        bc_anchor_time=bc_anchor_time,
        boundary_correction=boundary_correction,
        bc_mean_shift_prior_co2=bc_mean_shift_prior_co2,
        bc_mean_shift_prior_o2=bc_mean_shift_prior_o2,
        bc_anomaly_scale_co2=bc_anomaly_scale_co2,
        bc_anomaly_scale_o2=bc_anomaly_scale_o2,
        boundary_correction_sensitivity_co2=boundary_correction_sensitivity_co2,
        boundary_correction_sensitivity_o2=boundary_correction_sensitivity_o2,
        bc_centering_weights_co2=bc_centering_weights_co2,
        bc_centering_weights_o2=bc_centering_weights_o2,
        bc_prior=bc_prior,
        bc_state_activity=bc_state_activity,
        offset_prior=offset_prior,
        offset_args=offset_args,
    )
    metadata = _co2_o2_metadata(prepared, observations=observations)
    if boundary_correction:
        metadata.update(boundary_representation="centered-v1", independent_boundary_channels=list(boundary_correction))
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
        if channel in boundaries or channel in operands:
            roles.update(
                {
                    f"{channel}_boundary_concentration": f"{channel}_mu_bc",
                    f"{channel}_boundary_scale": f"{channel}_bc",
                    f"{channel}_boundary_sensitivity": f"{channel}_hbc",
                }
            )
        if channel in (offset_prior or {}):
            roles[f"{channel}_offset_concentration"] = f"{channel}_offset"
    for channel in operands:
        roles.pop(f"{channel}_boundary_sensitivity", None)
        roles[f"{channel}_boundary_anomaly"] = f"{channel}_H_bc_anomaly"
        roles[f"{channel}_boundary_mass"] = f"{channel}_H_bc_mass"
        roles[f"{channel}_boundary_reference"] = f"{channel}_bc_reference"
    if operands:
        metadata.update(
            boundary_representation="centered-v1",
            bc_frequency=bc_freq,
            bc_anchor_time=None if bc_anchor_time is None else str(bc_anchor_time),
        )
    if boundaries or operands or boundary_correction:
        roles["boundary_concentration"] = "boundary_concentration"
    if offset_prior:
        roles["offset_concentration"] = "offset_concentration"
    if boundaries or operands or boundary_correction or offset_prior:
        roles["baseline_concentration"] = "baseline_concentration"
    for channel in (boundary_correction or {}):
        roles.update(_boundary_correction_roles(cached.model, prefix=f"{channel}_"))
    _add_additive_boundary_roles(
        roles,
        metadata,
        co2_prior=bc_mean_shift_prior_co2,
        o2_prior=bc_mean_shift_prior_o2,
    )
    built = RhimeModelBuildResult(
        model=cached.model,
        output_contract=OutputContract(
            variable_roles=roles,
            metadata=metadata,
        ),
    )
    sampling_sampler = _sampler_for_cached_graph(
        requested_sampler,
        cached_model=cached,
        sigma_target_accept=sigma_target_accept,
        state_target_accept=state_target_accept,
    )
    trace = sample_rhime_model(built, sampling_sampler)
    trace = annotate_co2_o2_trace(trace, built=built)
    trace.attrs["rhime_recipe"] = "co2_o2_cached_sigma_fixed_ou"
    for group in trace.children.values():
        group.attrs["rhime_recipe"] = "co2_o2_cached_sigma_fixed_ou"
    return trace
