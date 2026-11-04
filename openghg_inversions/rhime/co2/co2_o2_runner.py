"""Advanced replay seam for prepared CO2/O2 scientific inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.inference.sampling import resolve_sampler
from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models import StateActivity
from openghg_inversions.models.boundary import (
    BoundBoundary,
    boundary_output_metadata,
    boundary_output_roles,
    boundary_output_units,
    prepare_boundary,
    require_centered_precision,
    restore_absolute_predictives,
)
from openghg_inversions.models.priors import PriorArgs, prior_records
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model
from openghg_inversions.rhime.specs import DEFAULT_BC_PRIOR

from .co2_model import _materialize_model_boundaries, _normalise_offset_args
from .co2_runner import _validate_bound_boundary_options
from .co2_o2_model import build_co2_o2_model
from .co2_o2_preparation import Co2O2PreparedInputs

_CO2_O2_VARIABLE_ROLES = {
    "observation": "y",
    "concentration": "y",
    "modelled_concentration": "modelled_concentration",
    "flux_scale": "flux_scaling",
    "emissions_sensitivity": "co2_o2_sensitivity",
    "flux_contribution": "co2_o2_flux_contribution",
    "coherent_prior_contribution": "fixed_prior_contribution",
    "independent_error": "fixed_independent_error_sd",
    "total_marginal_error": "epsilon",
}


def _validate_independent_error_labels(
    observations: xr.DataArray,
    independent_error_sd: xr.DataArray,
) -> None:
    """Require the external error vector to use the joint row structure."""
    observation_dim = str(observations.dims[0])
    if (
        independent_error_sd.dims != (observation_dim,)
        or observation_dim not in independent_error_sd.indexes
        or not independent_error_sd.indexes[observation_dim].equals(observations.indexes[observation_dim])
        or independent_error_sd.indexes[observation_dim].names != observations.indexes[observation_dim].names
    ):
        raise ValueError("independent_error_sd must use the prepared observation dimension and labels.")
    if "observation_units" not in independent_error_sd.coords or independent_error_sd[
        "observation_units"
    ].dims != (observation_dim,):
        raise ValueError(
            "independent_error_sd requires observation_units on the prepared observation dimension."
        )


def _validate_independent_error_values(array: xr.DataArray) -> None:
    """Validate the materialized external independent-error payload."""
    values = np.asarray(array.data)
    valid = (
        np.issubdtype(values.dtype, np.number)
        and not np.issubdtype(values.dtype, np.complexfloating)
        and np.isfinite(values).all()
        and (values > 0).all()
    )
    if not valid:
        raise ValueError("independent_error_sd must contain only finite positive real numeric values.")


def _materialize_co2_o2_replay_inputs(
    prepared: Co2O2PreparedInputs, independent_error_sd: xr.DataArray | None,
    boundary: Mapping[str, BoundBoundary], *, prior_parameters: Mapping[str, Any],
) -> tuple[tuple[xr.DataArray, ...], dict[str, BoundBoundary], dict[str, Any]]:
    """Compute core, bound boundary and selected priors together for PyMC."""
    saved_error = getattr(prepared, "independent_error_sd", None)
    if independent_error_sd is None:
        independent_error_sd = saved_error
    if independent_error_sd is None:
        raise ValueError("Supply independent_error_sd explicitly or in the prepared artifact.")
    errors = [independent_error_sd]
    if saved_error is not None and saved_error is not independent_error_sd:
        errors.append(saved_error)
    for error in errors:
        _validate_independent_error_labels(prepared.observations, error)
    materialized, boundary, priors = _materialize_model_boundaries(
        (prepared.observations, prepared.fixed_prior_contribution,
         prepared.co2_sensitivity, prepared.o2_sensitivity, *errors),
        boundary, priors=prior_parameters,
    )
    observations = materialized[0]
    dense_errors = materialized[4:]
    for error in dense_errors:
        if not np.array_equal(error["observation_units"].data, observations["observation_units"].data):
            raise ValueError("independent_error_sd observation_units must match the prepared observations.")
        _validate_independent_error_values(error)
    if len(dense_errors) == 2 and not np.array_equal(dense_errors[0].data, dense_errors[1].data):
        raise ValueError("Explicit independent_error_sd must match the prepared artifact values.")
    return (*materialized[:4], dense_errors[0]), boundary, priors


def _co2_o2_metadata(
    prepared: Co2O2PreparedInputs,
    *,
    observations: xr.DataArray,
) -> dict[str, object]:
    """Return JSON-safe scientific identity for the sampled result."""
    channel_units = {
        species: str(
            observations["observation_units"].where(observations["species"] == species, drop=True).data[0]
        )
        for species in ("co2", "o2")
    }
    ratio_provenance = json.loads(prepared.o2_sensitivity.attrs["oxidation_ratio_provenance"])
    return {
        "recipe": "co2_o2",
        "prior": "correlated arithmetic-moment lognormal",
        "likelihood": "joint Gaussian with fixed independent channel error",
        "independent_error": "fixed labelled standard deviation resolved from prepared or explicit inputs",
        "o2_sensitivity_ratio": {
            "convention": "embedded_signed_o2_per_co2",
            "application": "embedded in the O2 rows of the supplied joint sensitivity; no model multiplier",
            "scope": "shared GPP/TER/FF states; O2 ocean applied directly",
            **ratio_provenance,
        },
        "observation_units": channel_units,
        "provenance": dict(prepared.provenance),
    }


def _annotate_co2_o2_trace(
    trace: xr.DataTree,
    *,
    built: RhimeModelBuildResult,
) -> xr.DataTree:
    """Persist scientific roles, units, and provenance after coord restoration."""
    trace.attrs["rhime_priors"] = json.dumps(prior_records(built.model))
    trace.attrs["rhime_recipe"] = "co2_o2"
    trace.attrs["rhime_variable_roles"] = json.dumps(
        dict(built.output_contract.variable_roles),
        sort_keys=True,
    )
    trace.attrs["rhime_model_metadata"] = json.dumps(dict(built.output_contract.metadata), sort_keys=True)
    roles_by_variable: dict[str, list[str]] = {}
    for role, variable in built.output_contract.variable_roles.items():
        roles_by_variable.setdefault(variable, []).append(role)

    concentration_variables = {
        "boundary_reference_shift",
        "y",
        "observed_concentration",
        "modelled_concentration",
        "co2_o2_flux_contribution",
        "fixed_prior_contribution",
        "fixed_independent_error_sd",
        "epsilon",
    }
    concentration_variables.update(
        variable
        for role, variable in built.output_contract.variable_roles.items()
        if role.endswith("concentration")
    )
    state_variables = {
        "flux_scaling",
        "flux_scaling_fixed_value",
    }
    for group in trace.children.values():
        for variable, roles in roles_by_variable.items():
            if variable in group:
                group[variable].attrs["rhime_scientific_roles"] = json.dumps(sorted(roles))
        for variable in concentration_variables & set(group.variables):
            unit_coordinate = group[variable].coords.get("observation_units")
            units = (
                np.unique(unit_coordinate.values.astype(str))
                if unit_coordinate is not None
                else np.array([], dtype=str)
            )
            group[variable].attrs["units"] = (
                str(units[0]) if units.size == 1 else "mixed; see observation_units coordinate"
            )
        for variable in state_variables & set(group.variables):
            group[variable].attrs["units"] = "dimensionless flux scale"
        if "co2_o2_sensitivity" in group:
            group["co2_o2_sensitivity"].attrs["units"] = "observation_units per dimensionless flux scale"
        if "aggregation_error_covariance" in group:
            group["aggregation_error_covariance"].attrs["units"] = "observation_units * observation_units_cov"
        for name, unit in built.output_contract.metadata.get("boundary_units", {}).items():
            if name in group:
                group[name].attrs["units"] = unit
                group[name].attrs["tracer"] = name.split("_", 1)[0]
        for channel in ("co2", "o2"):
            if f"{channel}_offset" in group:
                group[f"{channel}_offset"].attrs["tracer"] = channel
        group.attrs["rhime_recipe"] = "co2_o2"
    return trace


def _add_linked_boundary_metadata(roles: Mapping[str, str], metadata: dict[str, Any]) -> None:
    """Retain the scientific channel choices declared by the components."""
    selected = [channel for channel in ("co2", "o2") if f"{channel}_boundary_mean_shift" in roles]
    if selected:
        metadata["additive_boundary_channels"] = selected
        metadata["boundary_prior"] = "common mean shift plus weighted-centered Gaussian anomalies"


def _validate_channel_baseline_options(
    boundary_sensitivity: Mapping[str, xr.DataArray] | None,
    bc_prior: Mapping[str, PriorArgs] | None,
    bc_state_activity: Mapping[str, StateActivity] | None,
    offset_prior: Mapping[str, PriorArgs] | None,
    offset_args: Mapping[str, Mapping[str, Any]] | None,
    boundary_operands: Mapping[str, xr.Dataset] | None = None,
    *,
    bc_mean_shift_prior: Mapping[str, PriorArgs] | None = None,
    bc_anomaly_scale: Mapping[str, float] | None = None,
    bc_centering_weights: Mapping[str, xr.DataArray] | None = None,
) -> None:
    """Reject unused channel options before constructing or sampling a model."""
    for name, values in (
        ("boundary_sensitivity", boundary_sensitivity),
        ("boundary_operands", boundary_operands),
        ("bc_prior", bc_prior),
        ("bc_state_activity", bc_state_activity),
        ("offset_prior", offset_prior),
        ("offset_args", offset_args),
        ("bc_mean_shift_prior", bc_mean_shift_prior),
        ("bc_anomaly_scale", bc_anomaly_scale),
        ("bc_centering_weights", bc_centering_weights),
    ):
        if values is not None and (not isinstance(values, Mapping) or set(values) - {"co2", "o2"}):
            raise ValueError(f"{name} must be a mapping keyed only by 'co2' and 'o2'.")
    for channel in ("co2", "o2"):
        if channel in (offset_args or {}) and channel not in (offset_prior or {}):
            raise ValueError(f"{channel} offset_args require offset_prior.")
        _normalise_offset_args((offset_args or {}).get(channel))




def _prepare_co2_o2_boundaries(
    prepared: Co2O2PreparedInputs, boundaries: Mapping[str, xr.DataArray],
    operands: Mapping[str, xr.Dataset], weights: Mapping[str, xr.DataArray], *,
    bc_freq: str | None, bc_anchor_time: DatetimeLike | None,
    bc_prior: Mapping[str, PriorArgs] | None, bc_state_activity: Mapping[str, StateActivity] | None,
    bc_mean_shift_prior: Mapping[str, PriorArgs] | None, bc_anomaly_scale: Mapping[str, float] | None,
) -> dict[str, BoundBoundary]:
    """Bind each selected native channel once before payload materialization."""
    result = {}
    observations = prepared.observations
    output_dim = str(observations.dims[0])
    for channel, sensitivity in (("co2", prepared.co2_sensitivity), ("o2", prepared.o2_sensitivity)):
        native = observations
        if channel in boundaries or channel in operands:
            native = select_gathered_data_array(
                observations, key=channel, key_dim="species", ragged_dim="channel_observation", stack_dim=output_dim,
            ).rename({output_dim: str(sensitivity.dims[0])})
            # Native channel rows have one physical unit despite the joint mixed-unit label.
            native = native.assign_attrs(units=str(native.observation_units.values[0]))
        prior = (bc_prior or {}).get(channel)
        additive = (bc_mean_shift_prior or {}).get(channel) is not None or (bc_anomaly_scale or {}).get(channel) is not None
        if prior is None and (channel in boundaries or channel in operands) and not additive:
            prior = dict(DEFAULT_BC_PRIOR)
        bound = prepare_boundary(
            native, boundary_sensitivity=boundaries.get(channel), boundary_operands=operands.get(channel),
            bc_frequency=bc_freq if channel in boundaries or channel in operands else None,
            bc_anchor_time=bc_anchor_time if channel in boundaries or channel in operands else None,
            bc_prior=prior,
            bc_state_activity=(bc_state_activity or {}).get(channel),
            bc_mean_shift_prior=(bc_mean_shift_prior or {}).get(channel),
            bc_anomaly_scale=(bc_anomaly_scale or {}).get(channel), bc_centering_weights=weights.get(channel),
        )
        if bound is not None:
            result[channel] = bound
    if not result and (bc_freq is not None or bc_anchor_time is not None):
        raise ValueError("Boundary period choices require boundary inputs.")
    return result


def _linked_boundary_output_metadata(
    boundary: Mapping[str, BoundBoundary], *, raw_choices: bool,
    bc_freq: str | None, bc_anchor_time: DatetimeLike | None,
) -> dict[str, Any]:
    """Preserve requested CLI metadata or describe supplied fitted records."""
    facts = {channel: boundary_output_metadata(bound) for channel, bound in boundary.items()}
    metadata: dict[str, Any] = {} if raw_choices else {"boundary_metadata": facts}
    if any("boundary_representation" in values for values in facts.values()):
        metadata["boundary_representation"] = "centered-v1"
        if raw_choices:
            metadata.update(bc_frequency=bc_freq,
                            bc_anchor_time=None if bc_anchor_time is None else str(bc_anchor_time))
    return metadata


def _resolve_co2_o2_boundary(
    prepared: Co2O2PreparedInputs, *, boundary: Mapping[str, BoundBoundary] | None,
    use_bc: Mapping[str, bool] | None, bc_prior: Mapping[str, PriorArgs] | None,
    bc_state_activity: Mapping[str, StateActivity] | None,
    bc_mean_shift_prior: Mapping[str, PriorArgs] | None,
    bc_anomaly_scale: Mapping[str, float] | None,
    bc_centering_weights: Mapping[str, xr.DataArray] | None,
    bc_freq: str | None, bc_anchor_time: DatetimeLike | None,
    offset_prior: Mapping[str, PriorArgs] | None,
    offset_args: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, BoundBoundary]:
    """Use supplied channel records or bind configured raw channels once."""
    _validate_bound_boundary_options(
        boundary, use_bc=use_bc, bc_prior=bc_prior, bc_state_activity=bc_state_activity,
        bc_mean_shift_prior=bc_mean_shift_prior, bc_anomaly_scale=bc_anomaly_scale,
        bc_centering_weights=bc_centering_weights, bc_freq=bc_freq, bc_anchor_time=bc_anchor_time,
    )
    if boundary is not None:
        if not isinstance(boundary, Mapping) or set(boundary) - {"co2", "o2"}:
            raise ValueError("boundary must be a mapping keyed only by 'co2' and 'o2'.")
        _validate_channel_baseline_options(None, None, None, offset_prior, offset_args)
        for bound in boundary.values():
            if boundary_output_metadata(bound).get("boundary_representation") == "centered-v1":
                require_centered_precision()
        return dict(boundary)
    boundaries = dict(getattr(prepared, "boundary_sensitivity", {}))
    operands = dict(getattr(prepared, "boundary_operands", {}))
    if use_bc is not None:
        if (
            not isinstance(use_bc, Mapping)
            or set(use_bc) - {"co2", "o2"}
            or any(not isinstance(value, bool) for value in use_bc.values())
        ):
            raise ValueError("use_bc must map co2/o2 to booleans.")
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
        bc_mean_shift_prior=bc_mean_shift_prior,
        bc_anomaly_scale=bc_anomaly_scale,
        bc_centering_weights=bc_centering_weights,
    )
    selected_weights = {}
    prepared_weights = prepared.bc_centering_weights
    for channel in ("co2", "o2"):
        if (bc_mean_shift_prior or {}).get(channel) is None:
            continue
        weights = (bc_centering_weights or {}).get(channel, prepared_weights.get(channel))
        if weights is not None:
            selected_weights[channel] = weights
    return _prepare_co2_o2_boundaries(
        prepared, boundaries, operands, selected_weights,
        bc_freq=bc_freq, bc_anchor_time=bc_anchor_time, bc_prior=bc_prior,
        bc_state_activity=bc_state_activity, bc_mean_shift_prior=bc_mean_shift_prior,
        bc_anomaly_scale=bc_anomaly_scale,
    )


def build_rhime_co2_o2(
    *,
    prepared_inputs: Co2O2PreparedInputs,
    independent_error_sd: xr.DataArray | None = None,
    state_activity: StateActivity | None = None,
    use_bc: Mapping[str, bool] | None = None,
    boundary: Mapping[str, BoundBoundary] | None = None,
    bc_mean_shift_prior: Mapping[str, PriorArgs] | None = None,
    bc_anomaly_scale: Mapping[str, float] | None = None,
    bc_centering_weights: Mapping[str, xr.DataArray] | None = None,
    bc_prior: Mapping[str, PriorArgs] | None = None,
    bc_state_activity: Mapping[str, StateActivity] | None = None,
    bc_freq: str | None = None,
    bc_anchor_time: DatetimeLike | None = None,
    offset_prior: Mapping[str, PriorArgs] | None = None,
    offset_args: Mapping[str, Mapping[str, Any]] | None = None,
    tau_hours: float | Mapping[str, float] | None = None,
    fixed_site_amplitudes: float | Mapping[str, float] | None = None,
    site_amplitude_prior: Mapping[str, Any] | None = None,
) -> RhimeModelBuildResult:
    """Materialize borrowed linked inputs and build their joint scientific graph.

    The scientific arguments have the meanings documented by
    :func:`run_rhime_co2_o2_from_prepared_inputs`. Saved and explicit independent
    errors must agree. Related sensitivities, errors and auxiliary coordinates
    are computed together at this PyMC boundary without mutating the inputs.

    Returns:
        The concrete model with scientific roles, units and preparation
        provenance, ready for native prior prediction or sampling.

    Raises:
        ValueError: If input labels, errors, channel baseline choices, state
            activity or covariance are inconsistent with the linked model.
    """
    prepared = prepared_inputs
    raw_boundary = boundary is None
    boundary = _resolve_co2_o2_boundary(
        prepared, boundary=boundary, use_bc=use_bc, bc_prior=bc_prior,
        bc_state_activity=bc_state_activity, bc_mean_shift_prior=bc_mean_shift_prior,
        bc_anomaly_scale=bc_anomaly_scale, bc_centering_weights=bc_centering_weights,
        bc_freq=bc_freq, bc_anchor_time=bc_anchor_time,
        offset_prior=offset_prior, offset_args=offset_args,
    )
    materialized, boundary, eager_priors = _materialize_co2_o2_replay_inputs(
        prepared, independent_error_sd, boundary,
        prior_parameters={"offset": offset_prior or {}, "amplitude": site_amplitude_prior or {}},
    )
    offset_prior = eager_priors["offset"]
    site_amplitude_prior = eager_priors["amplitude"] or None
    observations, fixed_prior_contribution, co2_sensitivity, o2_sensitivity, independent_error_sd = (
        materialized
    )
    model = build_co2_o2_model(
        observations=observations,
        fixed_prior_contribution=fixed_prior_contribution,
        co2_sensitivity=co2_sensitivity,
        o2_sensitivity=o2_sensitivity,
        aggregation_error=prepared.aggregation_error,
        retained_prior=prepared.retained_prior,
        independent_error_sd=independent_error_sd,
        state_activity=state_activity,
        tau_hours=tau_hours,
        fixed_site_amplitudes=fixed_site_amplitudes,
        site_amplitude_prior=site_amplitude_prior,
        boundary=boundary,
        offset_prior=offset_prior,
        offset_args=offset_args,
    )
    variable_roles = dict(_CO2_O2_VARIABLE_ROLES)
    for channel, bound in boundary.items():
        variable_roles.update(boundary_output_roles(bound, prefix=f"{channel}_"))
    for channel in (offset_prior or {}):
        variable_roles[f"{channel}_offset_concentration"] = f"{channel}_offset"
    if boundary:
        variable_roles["boundary_concentration"] = "boundary_concentration"
    if offset_prior:
        variable_roles["offset_concentration"] = "offset_concentration"
    if boundary or offset_prior:
        variable_roles["baseline_concentration"] = "baseline_concentration"
    metadata = _co2_o2_metadata(prepared, observations=observations)
    metadata.update(_linked_boundary_output_metadata(
        boundary, raw_choices=raw_boundary, bc_freq=bc_freq, bc_anchor_time=bc_anchor_time,
    ))
    if tau_hours is not None:
        variable_roles.update(
            {
                "independent_error": "error",
                "fixed_ou_site_amplitude": "ou_site_amplitude",
                "observation_to_fixed_ou_site_index": "ou_site_index",
                "fixed_ou_timescale": "ou_tau_hours",
            }
        )
        metadata["likelihood"] = "joint Gaussian with fixed OU blocks by (species, site)"
    metadata["boundary_units"] = {
        name: unit for channel, bound in boundary.items()
        for name, unit in boundary_output_units(bound, prefix=f"{channel}_").items()
    }
    if raw_boundary:
        _add_linked_boundary_metadata(variable_roles, metadata)
    return RhimeModelBuildResult(
        model=model,
        output_contract=OutputContract(
            variable_roles=variable_roles,
            metadata=metadata,
        ),
    )


def run_rhime_co2_o2_from_prepared_inputs(
    *,
    prepared_inputs: Co2O2PreparedInputs,
    independent_error_sd: xr.DataArray | None = None,
    state_activity: StateActivity | None = None,
    use_bc: Mapping[str, bool] | None = None,
    boundary: Mapping[str, BoundBoundary] | None = None,
    bc_mean_shift_prior: Mapping[str, PriorArgs] | None = None,
    bc_anomaly_scale: Mapping[str, float] | None = None,
    bc_centering_weights: Mapping[str, xr.DataArray] | None = None,
    bc_prior: Mapping[str, PriorArgs] | None = None,
    bc_state_activity: Mapping[str, StateActivity] | None = None,
    bc_freq: str | None = None,
    bc_anchor_time: DatetimeLike | None = None,
    offset_prior: Mapping[str, PriorArgs] | None = None,
    offset_args: Mapping[str, Mapping[str, Any]] | None = None,
    sampler: RhimeSampler | Literal["default"] = "default",
    tau_hours: float | Mapping[str, float] | None = None,
    fixed_site_amplitudes: float | Mapping[str, float] | None = None,
    site_amplitude_prior: Mapping[str, Any] | None = None,
) -> xr.DataTree:
    """Build and sample the CO2/O2 model from prepared scientific inputs.

    This advanced replay seam begins after channel preparation. The public
    ``run_rhime_co2_o2`` name is reserved for the future complete production
    recipe, including acquisition, preparation, materialization, and outputs.

    The fixed independent channel error is required and remains labelled. It
    may be stored in the prepared artifact or supplied explicitly; when both
    are present their labels, units, and values must agree. Run
    policy, such as the UOB prototype's one ppm value for both channels, belongs
    to the caller rather than the model API.

    Args:
        prepared_inputs: Validated preparation handoff containing joint
            observations and affine intercept on ``("observation",)``, separate
            channel sensitivities on their native observation axes and the retained
            state axis, a dense joint aggregation covariance, retained prior,
            ratio provenance, units, labels, and scientific provenance.
        independent_error_sd: Optional positive finite fixed standard deviations on
            ``("observation",)``. Labels and ``observation_units`` must match
            ``prepared_inputs.observations`` exactly. These values remain fixed
            data and are not an inferred mismatch amplitude. Defaults to the
            artifact's saved error vector. An explicit vector must agree with
            saved errors when the artifact contains them.
        state_activity: Optional labelled active/fixed policy on the retained
            state dimension. Omitted states use the model's structural activity
            policy.
        boundary: Already bound channel records, replacing raw boundary choices.
            An empty mapping disables boundaries; None selects configured raw inputs.
        use_bc: Optional co2/o2 boolean mapping selecting prepared boundary
            sensitivities. Omitted entries include available channel boundaries.
        bc_prior: Channel-keyed independent boundary-scale priors; the CO2
            default is used for each enabled channel when omitted.
        bc_mean_shift_prior: Channel-keyed scalar additive mean priors in
            concentration units, requiring native centered boundary operands.
        bc_anomaly_scale: Matching channel-keyed nonnegative Gaussian scales
            before weighted centering. Zero retains only the scalar mean.
        bc_centering_weights: Positive dimensionless weights over all resolved
            region/period states, keyed by channel; never inferred from sensitivity.
        bc_state_activity: Channel-keyed labelled fixed/active boundary policies.
        bc_freq: Optional model-owned frequency for raw boundary periods.
        bc_anchor_time: Optional timestamp anchoring raw boundary periods.
        offset_prior: Channel-keyed offset priors. Omitted channels have no offset.
        offset_args: Per-channel offset_freq, drop_first, and per_site options,
            with the same meanings as the CO2 offset component. Baseline terms
            require identical channel units and are zero on the other channel.
        tau_hours: Optional fixed OU timescale, scalar or mapping keyed by
            ``co2:SITE``/``o2:SITE``, in hours. Requires the same channel units.
        fixed_site_amplitudes: Fixed additive species/site standard deviations
            in concentration units, scalar or mapping using the same keys.
        site_amplitude_prior: Prior for inferred species/site amplitudes,
            mutually exclusive with fixed amplitudes.
        sampler: Sampler configuration or ``"default"``; explicit ``None`` is invalid. Fixed OU requires and
            defaults to PyMC; the fixed-error default uses NumPyro.

    Returns:
        Restored DataTree with observed concentrations in
        ``observed_data["y"]``, fixed independent standard deviations in
        ``constant_data["fixed_independent_error_sd"]`` (``error`` for fixed
        OU), labelled coordinates,
        data-dependent concentration units, scientific-role annotations, and
        JSON model metadata.

    Raises:
        TypeError: If the sampler is neither a configuration nor ``"default"``.
        ValueError: If the fixed independent standard deviation is not a finite
            positive real numeric vector or fails the model's
            observation-label/unit contract. Label, state, covariance, and
            activity errors from model construction are also propagated.
    """
    sampler = resolve_sampler(
        sampler,
        nuts_sampler="pymc" if tau_hours is not None else "numpyro",
        sample_kwargs={"target_accept": 0.95},
    )
    if tau_hours is not None and sampler.nuts_sampler != "pymc":
        raise ValueError("Linked fixed-OU requires nuts_sampler='pymc'.")
    built = build_rhime_co2_o2(
        prepared_inputs=prepared_inputs,
        independent_error_sd=independent_error_sd,
        state_activity=state_activity,
        use_bc=use_bc,
        boundary=boundary,
        bc_mean_shift_prior=bc_mean_shift_prior,
        bc_anomaly_scale=bc_anomaly_scale,
        bc_centering_weights=bc_centering_weights,
        bc_prior=bc_prior,
        bc_state_activity=bc_state_activity,
        bc_freq=bc_freq,
        bc_anchor_time=bc_anchor_time,
        offset_prior=offset_prior,
        offset_args=offset_args,
        tau_hours=tau_hours,
        fixed_site_amplitudes=fixed_site_amplitudes,
        site_amplitude_prior=site_amplitude_prior,
    )
    trace = sample_rhime_model(built, sampler)
    return annotate_co2_o2_trace(trace, built=built)


def _annotate_linked_fixed_ou_trace(
    trace: xr.DataTree,
    *,
    observations: xr.DataArray,
) -> xr.DataTree:
    """Retain OU group identity, units, and joint likelihood semantics."""
    units = str(observations.observation_units.values[0])
    for group_name, node in trace.children.items():
        group = node.to_dataset()
        if "ou_site" in group.coords:
            labels = group.ou_site.values.astype(str)
            group = group.assign_coords(
                ou_species=("ou_site", [label.split(":", 1)[0] for label in labels]),
                ou_station=("ou_site", [label.split(":", 1)[1] for label in labels]),
            )
        for name in ("ou_site_amplitude", "Y", "error"):
            if name in group:
                group[name].attrs["units"] = units
        if "ou_tau_hours" in group:
            group.ou_tau_hours.attrs.update(units="hours", dtype="timedelta64[ns]")
        if group_name == "log_likelihood" and "y" in group:
            group.y.attrs.pop("units", None)
            group.y.attrs.update(
                rhime_scientific_roles=json.dumps(["joint_log_likelihood"]),
                rhime_likelihood_scope="joint_observation_vector",
                rhime_normalized_log_likelihood=1,
            )
        if group_name in {"prior_predictive", "posterior_predictive"} and "y" in group:
            group.y.attrs["rhime_predictive_scope"] = "joint_observation_vector"
        trace[group_name] = group
    return trace


def annotate_co2_o2_trace(
    trace: xr.DataTree,
    *,
    built: RhimeModelBuildResult,
) -> xr.DataTree:
    """Attach the built joint model's scientific meaning after coordinate restoration.

    For centered models, updates numerical ``y`` values in ``observed_data``,
    ``prior_predictive``, ``posterior_predictive`` and ``predictions`` in place
    by adding the retained boundary reference shift. Values marked
    ``concentration_representation="absolute-v1"`` are left unchanged, so
    repeated annotation does not add the shift twice. See
    :ref:`surface-centered-boundary-contract`.

    Both posterior and native prior-predictive traces retain roles, units,
    preparation provenance and used priors. Fixed-OU traces additionally retain
    species/site grouping and joint likelihood/predictive scope. The trace is
    updated and returned; no graph is constructed or inference performed.
    """
    trace.attrs["rhime_boundary_format_version"] = 1
    restore_absolute_predictives(trace)
    trace = _annotate_co2_o2_trace(trace, built=built)
    if "log_likelihood" in trace.children and "y" in trace["log_likelihood"]:
        likelihood = trace["log_likelihood"]["y"]
        likelihood.attrs.pop("units", None)
        likelihood.attrs.update(
            rhime_scientific_roles=json.dumps(["joint_log_likelihood"]),
            rhime_likelihood_scope="joint_observation_vector",
            rhime_normalized_log_likelihood=1,
        )
    for name in ("prior_predictive", "posterior_predictive", "predictions"):
        if name in trace.children and "y" in trace[name]:
            trace[name]["y"].attrs["rhime_predictive_scope"] = "joint_observation_vector"
    if "fixed_ou_timescale" in built.output_contract.variable_roles:
        trace = _annotate_linked_fixed_ou_trace(trace, observations=trace["observed_data"].dataset["y"])
    return trace
