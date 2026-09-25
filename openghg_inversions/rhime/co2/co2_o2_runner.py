"""Advanced replay seam for prepared CO2/O2 scientific inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import arviz as az
from dask import compute as dask_compute
from dask.array import Array as DaskArray
import numpy as np
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.models import StateActivity
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model

from .co2_o2_model import build_co2_o2_model, _validate_channel_baseline_options
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


def _materialize_co2_o2_pymc_inputs(
    *arrays: xr.DataArray,
) -> tuple[xr.DataArray, ...]:
    """Materialize related labelled arrays together without changing inputs."""
    lazy_coordinates = [
        (array_index, name, coordinate)
        for array_index, array in enumerate(arrays)
        for name, coordinate in array.coords.items()
        if isinstance(coordinate.data, DaskArray)
    ]
    computed = dask_compute(
        *(to_dense(array).data for array in arrays),
        *(coordinate.data for _, _, coordinate in lazy_coordinates),
    )
    dense_arrays = [
        array.copy(deep=False, data=data) for array, data in zip(arrays, computed[: len(arrays)], strict=True)
    ]
    for (array_index, name, coordinate), data in zip(
        lazy_coordinates,
        computed[len(arrays) :],
        strict=True,
    ):
        dense_arrays[array_index] = dense_arrays[array_index].assign_coords(
            {name: coordinate.copy(deep=False, data=data)}
        )
    return tuple(dense_arrays)


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
        "independent_error": "fixed labelled standard deviation supplied by caller",
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
    trace: az.InferenceData,
    *,
    built: RhimeModelBuildResult,
) -> az.InferenceData:
    """Persist scientific roles, units, and provenance after coord restoration."""
    trace.attrs["rhime_recipe"] = "co2_o2"
    trace.attrs["rhime_variable_roles"] = json.dumps(
        dict(built.variable_roles),
        sort_keys=True,
    )
    trace.attrs["rhime_model_metadata"] = json.dumps(dict(built.metadata), sort_keys=True)
    roles_by_variable: dict[str, list[str]] = {}
    for role, variable in built.variable_roles.items():
        roles_by_variable.setdefault(variable, []).append(role)

    concentration_variables = {
        "y",
        "observed_concentration",
        "modelled_concentration",
        "co2_o2_flux_contribution",
        "fixed_prior_contribution",
        "fixed_independent_error_sd",
        "epsilon",
    }
    concentration_variables.update(
        variable for role, variable in built.variable_roles.items() if role.endswith("concentration")
    )
    state_variables = {
        "flux_scaling",
        "flux_scaling_fixed_value",
    }
    for group_name in trace.groups():
        group = getattr(trace, group_name)
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
        for channel in ("co2", "o2"):
            for variable in (f"{channel}_bc", f"{channel}_bc_fixed_value"):
                if variable in group:
                    group[variable].attrs["units"] = "dimensionless boundary scale"
                    group[variable].attrs["tracer"] = channel
            for variable in (f"{channel}_mu_bc", f"{channel}_offset"):
                if variable in group:
                    group[variable].attrs["tracer"] = channel
            if f"{channel}_hbc" in group:
                group[f"{channel}_hbc"].attrs["units"] = "observation_units per dimensionless boundary scale"
        group.attrs["rhime_recipe"] = "co2_o2"
    return trace


def run_rhime_co2_o2_from_prepared_inputs(
    *,
    prepared_inputs: Co2O2PreparedInputs,
    independent_error_sd: xr.DataArray,
    state_activity: StateActivity | None = None,
    use_bc: Mapping[str, bool] | None = None,
    bc_prior: Mapping[str, PriorArgs] | None = None,
    bc_state_activity: Mapping[str, StateActivity] | None = None,
    offset_prior: Mapping[str, PriorArgs] | None = None,
    offset_args: Mapping[str, Mapping[str, Any]] | None = None,
    sampler: RhimeSampler | None = None,
    tau_hours: float | Mapping[str, float] | None = None,
    fixed_site_amplitudes: float | Mapping[str, float] | None = None,
    site_amplitude_prior: Mapping[str, Any] | None = None,
) -> az.InferenceData:
    """Build and sample the CO2/O2 model from prepared scientific inputs.

    This advanced replay seam begins after channel preparation. The public
    ``run_rhime_co2_o2`` name is reserved for the future complete production
    recipe, including acquisition, preparation, materialization, and outputs.

    The fixed independent channel error is required and remains labelled. Run
    policy, such as the UOB prototype's one ppm value for both channels, belongs
    to the caller rather than the model API.

    Args:
        prepared_inputs: Validated preparation handoff containing joint
            observations and affine intercept on ``("observation",)``, separate
            channel sensitivities on their native observation axes and the retained
            state axis, a dense joint aggregation covariance, retained prior,
            ratio provenance, units, labels, and scientific provenance.
        independent_error_sd: Positive finite fixed standard deviations on
            ``("observation",)``. Labels and ``observation_units`` must match
            ``prepared_inputs.observations`` exactly. These values remain fixed
            data and are not an inferred mismatch amplitude.
        state_activity: Optional labelled active/fixed policy on the retained
            state dimension. Omitted states use the model's structural activity
            policy.
        use_bc: Optional co2/o2 boolean mapping selecting prepared boundary
            sensitivities. Omitted entries include available channel boundaries.
        bc_prior: Channel-keyed independent boundary-scale priors; the CO2
            default is used for each enabled channel when omitted.
        bc_state_activity: Channel-keyed labelled fixed/active boundary policies.
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
        sampler: Optional RHIME sampler configuration. The accepted CO2/O2
            NumPyro defaults are used when omitted.

    Returns:
        Restored inference data with observed concentrations in
        ``observed_data["y"]``, fixed independent standard deviations in
        ``constant_data["fixed_independent_error_sd"]``, labelled coordinates,
        data-dependent concentration units, scientific-role annotations, and
        JSON model metadata.

    Raises:
        ValueError: If the fixed independent standard deviation is not a finite
            positive real numeric vector or fails the model's
            observation-label/unit contract. Label, state, covariance, and
            activity errors from model construction are also propagated.
    """
    if tau_hours is not None and sampler is not None and sampler.nuts_sampler != "pymc":
        raise ValueError("Linked fixed-OU requires nuts_sampler='pymc'.")
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
    _validate_channel_baseline_options(boundaries, bc_prior, bc_state_activity, offset_prior, offset_args)
    _validate_independent_error_labels(
        prepared.observations,
        independent_error_sd,
    )
    (
        observations,
        fixed_prior_contribution,
        co2_sensitivity,
        o2_sensitivity,
        independent_error_sd,
        *boundary_arrays,
    ) = _materialize_co2_o2_pymc_inputs(
        prepared.observations,
        prepared.fixed_prior_contribution,
        prepared.co2_sensitivity,
        prepared.o2_sensitivity,
        independent_error_sd,
        *boundaries.values(),
    )
    boundaries = dict(zip(boundaries, boundary_arrays, strict=True))
    if not np.array_equal(
        independent_error_sd["observation_units"].data,
        observations["observation_units"].data,
    ):
        raise ValueError("independent_error_sd observation_units must match the prepared observations.")
    _validate_independent_error_values(independent_error_sd)
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
        boundary_sensitivity=boundaries,
        bc_prior=bc_prior,
        bc_state_activity=bc_state_activity,
        offset_prior=offset_prior,
        offset_args=offset_args,
    )
    variable_roles = dict(_CO2_O2_VARIABLE_ROLES)
    for channel in ("co2", "o2"):
        if channel in boundaries:
            variable_roles.update(
                {
                    f"{channel}_boundary_concentration": f"{channel}_mu_bc",
                    f"{channel}_boundary_scale": f"{channel}_bc",
                    f"{channel}_boundary_sensitivity": f"{channel}_hbc",
                }
            )
        if channel in (offset_prior or {}):
            variable_roles[f"{channel}_offset_concentration"] = f"{channel}_offset"
    if boundaries:
        variable_roles["boundary_concentration"] = "boundary_concentration"
    if offset_prior:
        variable_roles["offset_concentration"] = "offset_concentration"
    if boundaries or offset_prior:
        variable_roles["baseline_concentration"] = "baseline_concentration"
    metadata = _co2_o2_metadata(prepared, observations=observations)
    if tau_hours is not None:
        variable_roles.update({"independent_error": "error", "fixed_ou_site_amplitude": "ou_site_amplitude",
                      "observation_to_fixed_ou_site_index": "ou_site_index",
                      "fixed_ou_timescale": "ou_tau_hours"})
        metadata["likelihood"] = "joint Gaussian with fixed OU blocks by (species, site)"
    built = RhimeModelBuildResult(
        model=model,
        variable_roles=variable_roles,
        metadata=metadata,
    )
    sampler = sampler or RhimeSampler(nuts_sampler="pymc" if tau_hours is not None else "numpyro", sample_kwargs={"target_accept": 0.95})
    trace = sample_rhime_model(built, sampler)
    trace = _annotate_co2_o2_trace(trace, built=built)
    if tau_hours is not None:
        trace = _annotate_linked_fixed_ou_trace(trace, observations=observations)
    return trace


def _annotate_linked_fixed_ou_trace(
    trace: az.InferenceData,
    *,
    observations: xr.DataArray,
) -> az.InferenceData:
    """Retain OU group identity, units, and joint likelihood semantics."""
    units = str(observations.observation_units.values[0])
    for group_name in trace.groups():
        group = getattr(trace, group_name)
        if "ou_site" in group.coords:
            labels = group.ou_site.values.astype(str)
            group = group.assign_coords(
                ou_species=("ou_site", [label.split(":", 1)[0] for label in labels]),
                ou_station=("ou_site", [label.split(":", 1)[1] for label in labels]),
            )
            setattr(trace, group_name, group)
        for name in ("ou_site_amplitude", "Y", "error"):
            if name in group:
                group[name].attrs["units"] = units
        if "ou_tau_hours" in group:
            group.ou_tau_hours.attrs["units"] = "hours"
        if group_name == "log_likelihood" and "y" in group:
            group.y.attrs.pop("units", None)
            group.y.attrs.update(
                rhime_scientific_roles=json.dumps(["joint_log_likelihood"]),
                rhime_likelihood_scope="joint_observation_vector",
                rhime_normalized_log_likelihood=True,
            )
    return trace
