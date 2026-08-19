"""Sampling seam for validated linked CO2/O2 scientific inputs."""

from __future__ import annotations

import json

import arviz as az
import xarray as xr

from openghg_inversions.models import StateActivity
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model

from .linked_model import build_linked_co2_o2_model
from .linked_preparation import Co2O2PreparedInputs


_LINKED_VARIABLE_ROLES = {
    "observation": "observed_concentration",
    "concentration": "modelled_concentration",
    "modelled_concentration": "modelled_concentration",
    "pollution_concentration": "modelled_concentration",
    "flux_scale": "flux_scaling",
    "flux_scaling": "flux_scaling",
    "emissions_sensitivity": "effective_observation_operator",
    "prior_forward_concentration": "prior_forward_concentration",
    "independent_error": "fixed_independent_error_sd",
}


def _default_linked_sampler() -> RhimeSampler:
    """Return the sampler settings used by the accepted UOB prototype."""
    return RhimeSampler(
        nuts_sampler="numpyro",
        sample_kwargs={"target_accept": 0.95},
    )


def _linked_metadata(prepared: Co2O2PreparedInputs) -> dict[str, object]:
    """Return JSON-safe scientific identity for the sampled result."""
    channel_units = {
        tracer: str(
            prepared.observations["observation_units"]
            .where(prepared.observations["tracer"] == tracer, drop=True)
            .values[0]
        )
        for tracer in ("co2", "o2")
    }
    return {
        "recipe": "co2_o2",
        "prior": "correlated arithmetic-moment lognormal",
        "likelihood": "joint Gaussian with fixed independent channel error",
        "independent_error": "fixed labelled standard deviation supplied by caller",
        "observation_units": channel_units,
        "provenance": dict(prepared.provenance),
    }


def _annotate_linked_trace(
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
        "observed_concentration",
        "modelled_concentration",
        "prior_forward_concentration",
        "fixed_independent_error_sd",
    }
    state_variables = {
        "flux_scaling",
        "prior_flux_scaling",
        "flux_scaling_fixed_value",
    }
    for group_name in trace.groups():
        group = getattr(trace, group_name)
        for variable, roles in roles_by_variable.items():
            if variable in group:
                group[variable].attrs["rhime_scientific_roles"] = json.dumps(sorted(roles))
        for variable in concentration_variables & set(group.variables):
            group[variable].attrs["units"] = "mixed; see observation_units coordinate"
        for variable in state_variables & set(group.variables):
            group[variable].attrs["units"] = "dimensionless flux scale"
        if "effective_observation_operator" in group:
            group["effective_observation_operator"].attrs["units"] = (
                "observation_units per dimensionless flux scale"
            )
        if "aggregation_error_covariance" in group:
            group["aggregation_error_covariance"].attrs["units"] = "observation_units * observation_units_cov"
        group.attrs["rhime_recipe"] = "co2_o2"
        setattr(trace, group_name, group)
    return trace


def run_rhime_co2_o2(
    *,
    prepared_inputs: Co2O2PreparedInputs,
    independent_error_sd: xr.DataArray,
    state_activity: StateActivity | None = None,
    sampler: RhimeSampler | None = None,
) -> az.InferenceData:
    """Build and sample the linked model from validated scientific inputs.

    The fixed independent channel error is required and remains labelled. Run
    policy, such as the UOB prototype's one ppm value for both channels, belongs
    to the caller rather than the model API.
    """
    prepared = prepared_inputs
    model = build_linked_co2_o2_model(
        observations=prepared.observations,
        prior_forward_mean=prepared.prior_forward_mean,
        effective_observation_operator=prepared.effective_observation_operator,
        aggregation_error=prepared.aggregation_error,
        retained_prior=prepared.retained_prior,
        independent_error_sd=independent_error_sd,
        state_activity=state_activity,
    )
    built = RhimeModelBuildResult(
        model=model,
        variable_roles=_LINKED_VARIABLE_ROLES,
        metadata=_linked_metadata(prepared),
    )
    trace = sample_rhime_model(
        built,
        _default_linked_sampler() if sampler is None else sampler,
    )
    return _annotate_linked_trace(trace, built=built)
