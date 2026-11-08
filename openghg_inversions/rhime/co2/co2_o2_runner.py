"""Advanced replay seam for prepared CO2/O2 scientific inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import xarray as xr

from dask import compute as dask_compute

from openghg_inversions.array_ops import select_gathered_data_array, to_dense_tree
from openghg_inversions.inference.sampling import resolve_sampler
from openghg_inversions.models import StateActivity
from openghg_inversions.models.boundary import Boundary, BoundaryChoice, MultiplicativeChoice, RawCenteredTransport
from openghg_inversions.models.priors import PriorArgs, prior_records
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model

from .boundary_output import BOUNDARY_FORMAT_VERSION, restore_absolute_predictives
from .co2_o2_model import _validate_channel_offset_options, build_co2_o2_model
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
    boundary: Mapping[str, Boundary], *, prior_parameters: Mapping[str, Any],
) -> tuple[tuple[xr.DataArray, ...], dict[str, Boundary], dict[str, Any]]:
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
    materialized: tuple[xr.DataArray, ...]
    bound: dict[str, Boundary]
    priors: dict[str, Any]
    materialized, bound, priors = dask_compute(*to_dense_tree((
        (prepared.observations, prepared.fixed_prior_contribution,
         prepared.co2_sensitivity, prepared.o2_sensitivity, *errors),
        dict(boundary), dict(prior_parameters),
    )))
    observations = materialized[0]
    dense_errors = materialized[4:]
    for error in dense_errors:
        if not np.array_equal(error["observation_units"].data, observations["observation_units"].data):
            raise ValueError("independent_error_sd observation_units must match the prepared observations.")
        _validate_independent_error_values(error)
    if len(dense_errors) == 2 and not np.array_equal(dense_errors[0].data, dense_errors[1].data):
        raise ValueError("Explicit independent_error_sd must match the prepared artifact values.")
    return (*materialized[:4], dense_errors[0]), bound, priors


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




def _bind_co2_o2_boundaries(
    prepared: Co2O2PreparedInputs, boundary: Mapping[str, BoundaryChoice[RawCenteredTransport]] | None,
) -> dict[str, Boundary]:
    """Bind each channel's boundary choice once to its native prepared transport."""
    if boundary is not None and (not isinstance(boundary, Mapping) or set(boundary) - {"co2", "o2"}):
        raise ValueError("boundary must be a mapping keyed only by 'co2' and 'o2'.")
    result = {}
    for channel, sensitivity in (("co2", prepared.co2_sensitivity), ("o2", prepared.o2_sensitivity)):
        choice = (boundary or {}).get(channel)
        if choice is None:
            continue
        native = select_gathered_data_array(
            prepared.observations, key=channel, key_dim="species", ragged_dim="channel_observation",
            stack_dim=str(prepared.observations.dims[0]),
        ).rename({str(prepared.observations.dims[0]): str(sensitivity.dims[0])})
        if channel in prepared.boundary_operands:
            operands = prepared.boundary_operands[channel]
            if channel in prepared.bc_centering_weights:
                operands = RawCenteredTransport.with_centering_weights(
                    operands, prepared.bc_centering_weights[channel],
                )
            result[channel] = choice.bind(native, operands)
        elif channel not in prepared.boundary_sensitivity:
            raise ValueError(f"The {channel} boundary choice requires prepared {channel} boundary inputs.")
        elif isinstance(choice, MultiplicativeChoice):
            result[channel] = choice.bind(native, prepared.boundary_sensitivity[channel])
        else:
            raise ValueError(
                f"The {channel} {type(choice).__name__} requires centered boundary operands; prepare them again."
            )
    return result


def _linked_boundary_contract(boundary: Mapping[str, Boundary]) -> tuple[dict[str, str], dict[str, Any]]:
    """Collect channel-prefixed boundary roles, units and metadata."""
    roles: dict[str, str] = {}
    metadata: dict[str, Any] = {"boundary_units": {}, "boundary_metadata": {}}
    for channel, bound in boundary.items():
        outputs = bound.outputs(prefix=f"{channel}_")
        roles.update(outputs.roles)
        metadata["boundary_units"].update(outputs.units)
        metadata["boundary_metadata"][channel] = outputs.metadata
    if boundary:
        roles["boundary_concentration"] = "boundary_concentration"
    return roles, metadata


def build_rhime_co2_o2(
    *,
    prepared_inputs: Co2O2PreparedInputs,
    independent_error_sd: xr.DataArray | None = None,
    state_activity: StateActivity | None = None,
    boundary: Mapping[str, BoundaryChoice[RawCenteredTransport]] | None = None,
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
    _validate_channel_offset_options(offset_prior, offset_args)
    materialized, bound, eager_priors = _materialize_co2_o2_replay_inputs(
        prepared, independent_error_sd, _bind_co2_o2_boundaries(prepared, boundary),
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
        boundary=bound,
        offset_prior=offset_prior,
        offset_args=offset_args,
    )
    boundary_roles, boundary_metadata = _linked_boundary_contract(bound)
    variable_roles = {**_CO2_O2_VARIABLE_ROLES, **boundary_roles}
    for channel in (offset_prior or {}):
        variable_roles[f"{channel}_offset_concentration"] = f"{channel}_offset"
    if offset_prior:
        variable_roles["offset_concentration"] = "offset_concentration"
    if bound or offset_prior:
        variable_roles["baseline_concentration"] = "baseline_concentration"
    metadata = {**_co2_o2_metadata(prepared, observations=observations), **boundary_metadata}
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
    boundary: Mapping[str, BoundaryChoice[RawCenteredTransport]] | None = None,
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
        boundary: Optional boundary choices keyed by ``co2``/``o2``. Each is
            bound once to that channel's prepared absolute sensitivity or
            centered operands (with any prepared centering weights). Omitted
            channels have no boundary term.
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
        boundary=boundary,
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
    trace.attrs["rhime_boundary_format_version"] = BOUNDARY_FORMAT_VERSION
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
