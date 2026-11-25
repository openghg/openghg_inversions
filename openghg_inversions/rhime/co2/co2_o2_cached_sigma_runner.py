"""Run the linked fixed-OU graph with the existing CO2 CompoundStep.

Inferred amplitudes use the sigma-then-state sweep. Fixed amplitudes are its
specialisation: the state quadratic is built once and only the state NUTS step
runs. The ordinary linked runner and configured ``sample`` stage send fixed-OU
fits here too, keeping their ``co2_o2`` recipe annotations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

import xarray as xr

from openghg_inversions.inference.sampling import resolve_sampler
from openghg_inversions.models import StateActivity
from openghg_inversions.models.boundary import Choice
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model

from .co2_cached_sigma_runner import (
    _sampler_for_cached_graph,
    resolve_cached_target_accept,
    validate_cached_co2_sampler,
)
from .co2_o2_cached_sigma_model import _build_co2_o2_cached_sigma_model
from .co2_o2_model import _validate_channel_offset_options
from .co2_o2_preparation import Co2O2PreparedInputs
from .co2_o2_runner import (
    _CO2_O2_VARIABLE_ROLES,
    _bind_co2_o2_boundaries,
    _co2_o2_metadata,
    _linked_boundary_contract,
    _materialize_co2_o2_replay_inputs,
    annotate_co2_o2_trace,
)


CACHED_FIXED_OU_RECIPE = "co2_o2_cached_sigma_fixed_ou"


def _validate_linked_cached_request(
    sampler: RhimeSampler,
    *,
    site_amplitude_prior: PriorArgs | None,
    fixed_site_amplitudes: float | Mapping[str, float] | None,
    sigma_target_accept: float | None,
    state_target_accept: float | None,
) -> None:
    """Reject inconsistent amplitude and sampler choices before loading arrays."""
    if sampler.nuts_sampler != "pymc":
        raise ValueError("Linked cached fixed-OU requires nuts_sampler='pymc'.")
    if (site_amplitude_prior is None) == (fixed_site_amplitudes is None):
        raise ValueError("Pass exactly one of `site_amplitude_prior` or `fixed_site_amplitudes`.")
    fixed = fixed_site_amplitudes is not None
    validate_cached_co2_sampler(sampler, fixed_amplitudes=fixed)
    resolve_cached_target_accept(
        sampler,
        fixed_amplitudes=fixed,
        sigma_target_accept=sigma_target_accept,
        state_target_accept=state_target_accept,
    )


def _sample_linked_cached(
    *,
    prepared_inputs: Co2O2PreparedInputs,
    sampler: RhimeSampler,
    tau_hours: float | Mapping[str, float],
    independent_error_sd: xr.DataArray | None = None,
    site_amplitude_prior: PriorArgs | None = None,
    fixed_site_amplitudes: float | Mapping[str, float] | None = None,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
    state_activity: StateActivity | None = None,
    boundary: Mapping[str, Choice | None] | None = None,
    offset_prior: Mapping[str, PriorArgs] | None = None,
    offset_args: Mapping[str, Mapping[str, Any]] | None = None,
    sigma_target_accept: float | None = None,
    state_target_accept: float | None = None,
    recipe: str = CACHED_FIXED_OU_RECIPE,
) -> tuple[xr.DataTree, dict[str, Any]]:
    """Build the linked cached graph, sample its matched CompoundStep and annotate it.

    ``recipe`` names the output recipe: the dedicated cached recipe, or
    ``"co2_o2"`` when the ordinary linked fixed-OU route delegates here, which
    keeps that route's annotations. ``sampler`` must already be resolved.

    Returns:
        The annotated trace and the actual-inference description (sampler
        settings, algorithm and step targets), also stored as the trace's
        ``rhime_actual_inference`` attribute.
    """
    _validate_linked_cached_request(
        sampler,
        site_amplitude_prior=site_amplitude_prior,
        fixed_site_amplitudes=fixed_site_amplitudes,
        sigma_target_accept=sigma_target_accept,
        state_target_accept=state_target_accept,
    )
    fixed = fixed_site_amplitudes is not None
    prepared = prepared_inputs
    _validate_channel_offset_options(offset_prior, offset_args)
    materialized, bound, eager_priors = _materialize_co2_o2_replay_inputs(
        prepared, independent_error_sd, _bind_co2_o2_boundaries(prepared, boundary),
        prior_parameters={
            "offset": offset_prior or {},
            **({"amplitude": site_amplitude_prior} if site_amplitude_prior is not None else {}),
        },
    )
    offset_prior = eager_priors["offset"]
    site_amplitude_prior = eager_priors.get("amplitude")
    observations, fixed_contribution, co2, o2, error = materialized
    cached = _build_co2_o2_cached_sigma_model(
        observations=observations,
        fixed_prior_contribution=fixed_contribution,
        co2_sensitivity=co2,
        o2_sensitivity=o2,
        aggregation_error=prepared.aggregation_error,
        retained_prior=prepared.retained_prior,
        independent_error_sd=error,
        tau_hours=tau_hours,
        site_amplitude_prior=site_amplitude_prior,
        initial_site_amplitudes=initial_site_amplitudes,
        fixed_site_amplitudes=fixed_site_amplitudes,
        state_activity=state_activity,
        boundary=bound,
        offset_prior=offset_prior,
        offset_args=offset_args,
    )
    metadata = _co2_o2_metadata(prepared, observations=observations)
    metadata.update(likelihood="joint Gaussian with fixed OU blocks by (species, site)")
    if recipe != "co2_o2":
        metadata["recipe"] = recipe
    roles = dict(_CO2_O2_VARIABLE_ROLES)
    roles.update(
        independent_error="error",
        fixed_ou_site_amplitude="ou_site_amplitude",
        observation_to_fixed_ou_site_index="ou_site_index",
        fixed_ou_timescale="ou_tau_hours",
    )
    boundary_roles, boundary_metadata = _linked_boundary_contract(bound)
    roles.update(boundary_roles)
    metadata.update(boundary_metadata)
    for channel in (offset_prior or {}):
        roles[f"{channel}_offset_concentration"] = f"{channel}_offset"
    if offset_prior:
        roles["offset_concentration"] = "offset_concentration"
    if bound or offset_prior:
        roles["baseline_concentration"] = "baseline_concentration"
    built = RhimeModelBuildResult(
        model=cached.model,
        output_contract=OutputContract(
            variable_roles=roles,
            metadata=metadata,
        ),
    )
    sigma_target, state_target = resolve_cached_target_accept(
        sampler,
        fixed_amplitudes=fixed,
        sigma_target_accept=sigma_target_accept,
        state_target_accept=state_target_accept,
    )
    sampling_sampler = _sampler_for_cached_graph(
        sampler,
        cached_model=cached,
        sigma_target_accept=sigma_target_accept,
        state_target_accept=state_target_accept,
    )
    actual: dict[str, Any] = dict(sampler.describe())
    if fixed:
        actual.update(algorithm="cached_fixed_ou_state_nuts", state_target_accept=state_target)
    else:
        actual.update(algorithm="cached_ou_sigma_nuts", sigma_target_accept=sigma_target)
        if cached.states:
            actual.update(algorithm="cached_ou_sigma_then_state_nuts", state_target_accept=state_target)
    trace = sample_rhime_model(built, sampling_sampler)
    trace = annotate_co2_o2_trace(trace, built=built)
    if recipe != "co2_o2":
        trace.attrs["rhime_recipe"] = recipe
        for group in trace.children.values():
            group.attrs["rhime_recipe"] = recipe
    trace.attrs["rhime_actual_inference"] = json.dumps(actual)
    return trace, actual


def run_rhime_co2_o2_cached_sigma_from_prepared_inputs(
    *,
    prepared_inputs: Co2O2PreparedInputs,
    independent_error_sd: xr.DataArray | None = None,
    tau_hours: float | Mapping[str, float],
    site_amplitude_prior: PriorArgs | None = None,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
    fixed_site_amplitudes: float | Mapping[str, float] | None = None,
    state_activity: StateActivity | None = None,
    boundary: Mapping[str, Choice | None] | None = None,
    offset_prior: Mapping[str, PriorArgs] | None = None,
    offset_args: Mapping[str, Mapping[str, Any]] | None = None,
    sampler: RhimeSampler | Literal["default"] = "default",
    sigma_target_accept: float | None = None,
    state_target_accept: float | None = None,
) -> xr.DataTree:
    """Sample linked species/site OU amplitudes followed by affine states.

    One joint likelihood retains the prepared cross-channel aggregation
    covariance and adds independent errors and OU blocks grouped by species
    and site. Inputs remain borrowed; related payloads are materialized
    together before graph construction. The runner constructs its
    sigma-then-state CompoundStep, refreshing the accepted amplitude cache
    before each affine-state update. With ``fixed_site_amplitudes`` there is
    no amplitude step: the state quadratic is built once, never refreshed,
    and the CompoundStep holds only the state NUTS step.

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
            Requires identical CO2 and O2 units. Exactly one of this or
            ``fixed_site_amplitudes`` is required.
        initial_site_amplitudes: Optional finite positive starting amplitudes,
            in concentration units. A scalar applies to every group; mappings
            use the same keys as tau_hours. Defaults to the prior support point.
            Only valid with ``site_amplitude_prior``.
        fixed_site_amplitudes: Fixed non-negative species/site amplitudes in
            each group's channel units, as a scalar or a mapping with the
            same keys as tau_hours. With mixed channel units (for example ppm
            CO2 and per-meg O2) a nonzero value must be a mapping. Stored as
            ``constant_data["ou_site_amplitude"]``.
        state_activity: Optional labelled active/fixed retained-state policy.
            Omitted states follow the model's structural activity policy.
        boundary: Optional boundary choices keyed by ``co2``/``o2``, each bound
            once to that channel's prepared absolute sensitivity or centered
            operands. ``None`` disables a channel's boundary; omitted channels
            have no boundary term, with a warning when the preparation has
            their transport.
        offset_prior: Channel-keyed offset priors in concentration units.
            Channels omitted from this mapping have no offset.
        offset_args: Per-channel per_site, offset_freq, and drop_first options
            with the ordinary CO2 offset meanings. Requires an offset prior
            for each configured channel; the default is one offset per site.
        sampler: Sampler configuration or ``"default"`` for PyMC; ``None`` is invalid. Only the
            PyMC backend is supported. Caller-supplied steps are rejected
            because the runner owns the NUTS steps. With inferred amplitudes
            ``sample_kwargs['target_accept']`` is rejected; with fixed
            amplitudes it sets the state step's target. Fixed amplitudes also
            accept the usual nested ``sample_kwargs['nuts']`` options and PyMC
            prediction choices. With inferred amplitudes, predictive keyword
            arguments are limited to random_seed.
        sigma_target_accept: Target acceptance probability for the amplitude
            NUTS step, defaulting to 0.8. Rejected with fixed amplitudes.
        state_target_accept: Target acceptance probability for the affine-state
            NUTS step. Defaults to 0.9 with inferred amplitudes; with fixed
            amplitudes, to the top-level or nested NUTS ``target_accept``
            option if supplied, or else 0.8.

    Returns:
        Restored DataTree with posterior samples, labelled observed data and
        constants, scientific roles, concentration units, and JSON model
        metadata. The log_likelihood group contains one normalized joint value
        per chain/draw, not pointwise likelihoods. Prior groups and joint
        posterior_predictive observation vectors are included when requested
        by the sampler. ``rhime_actual_inference`` records the algorithm and
        step targets. No artifact is written to disk.

    Raises:
        ValueError: If independent errors are missing, invalid, mislabelled,
            or conflict with saved errors; channel units or grouping
            coordinates are invalid; selected boundary/offset settings are
            inconsistent; selected centered boundaries use a floatX other
            than float64; not exactly one amplitude option is given; OU scales
            or amplitudes are invalid; fixed amplitudes leave no active state;
            or the sampler requests an unsupported backend, step, tuning
            control, or predictive keyword. Label, prior, covariance, and
            activity errors from model construction also propagate.
        TypeError: If channel offset frequency or boolean options have
            unsupported types, or the sampler is neither a configuration nor
            ``"default"``.
    """
    requested_sampler = resolve_sampler(sampler, nuts_sampler="pymc")
    trace, _ = _sample_linked_cached(
        prepared_inputs=prepared_inputs,
        sampler=requested_sampler,
        independent_error_sd=independent_error_sd,
        tau_hours=tau_hours,
        site_amplitude_prior=site_amplitude_prior,
        fixed_site_amplitudes=fixed_site_amplitudes,
        initial_site_amplitudes=initial_site_amplitudes,
        state_activity=state_activity,
        boundary=boundary,
        offset_prior=offset_prior,
        offset_args=offset_args,
        sigma_target_accept=sigma_target_accept,
        state_target_accept=state_target_accept,
    )
    return trace
