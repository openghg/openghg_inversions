"""Advanced replay seam for prepared CO2/O2 scientific inputs."""

from __future__ import annotations

import json

import arviz as az
from dask import compute as dask_compute
from dask.array import Array as DaskArray
import numpy as np
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.models import StateActivity
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model

from .co2_o2_model import build_co2_o2_model
from .co2_o2_preparation import Co2O2PreparedInputs


_CO2_O2_VARIABLE_ROLES = {
    "observation": "observed_concentration",
    "concentration": "y",
    "modelled_concentration": "modelled_concentration",
    "pollution_concentration": "modelled_concentration",
    "flux_scale": "flux_scaling",
    "flux_scaling": "flux_scaling",
    "co2_emissions_sensitivity": "co2_operator",
    "o2_emissions_sensitivity": "o2_operator",
    "prior_forward_concentration": "prior_forward_concentration",
    "independent_error": "fixed_independent_error_sd",
}


def _default_co2_o2_sampler() -> RhimeSampler:
    """Return the sampler settings used by the accepted UOB prototype."""
    return RhimeSampler(
        nuts_sampler="numpyro",
        sample_kwargs={"target_accept": 0.95},
    )


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
        array.copy(deep=False, data=data)
        for array, data in zip(arrays, computed[: len(arrays)], strict=True)
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


def _co2_o2_metadata(prepared: Co2O2PreparedInputs) -> dict[str, object]:
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
        "o2_operator_ratio": {
            "convention": prepared.o2_operator_ratio_convention,
            "application": "embedded in the supplied O2 operator; no model multiplier",
            "reference_ratio": "source/spatially resolved in the paired native O2 flux",
            "direction": "O2 flux per CO2 flux",
            "sign": "signed; positive CO2 flux has negative O2 loading",
            "scope": "shared GPP/TER/FF states; O2 ocean applied directly",
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
        for variable in {"co2_operator", "o2_operator"} & set(group.variables):
            group[variable].attrs["units"] = "observation_units per dimensionless flux scale"
        if "aggregation_error_covariance" in group:
            group["aggregation_error_covariance"].attrs["units"] = "observation_units * observation_units_cov"
        group.attrs["rhime_recipe"] = "co2_o2"
        setattr(trace, group_name, group)
    return trace


def run_rhime_co2_o2_from_prepared_inputs(
    *,
    prepared_inputs: Co2O2PreparedInputs,
    independent_error_sd: xr.DataArray,
    state_activity: StateActivity | None = None,
    sampler: RhimeSampler | None = None,
) -> az.InferenceData:
    """Build and sample the CO2/O2 model from prepared scientific inputs.

    This advanced replay seam begins after channel preparation. The public
    ``run_rhime_co2_o2`` name is reserved for the future complete production
    recipe, including acquisition, preparation, materialization, and outputs.

    The fixed independent channel error is required and remains labelled. Run
    policy, such as the UOB prototype's one ppm value for both channels, belongs
    to the caller rather than the model API.
    """
    prepared = prepared_inputs
    (
        observations,
        prior_forward_mean,
        co2_operator,
        o2_operator,
        independent_error_sd,
    ) = _materialize_co2_o2_pymc_inputs(
        prepared.observations,
        prepared.prior_forward_mean,
        prepared.co2_operator,
        prepared.o2_operator,
        independent_error_sd,
    )
    if not np.isfinite(independent_error_sd.data).all() or (
        independent_error_sd.data <= 0
    ).any():
        raise ValueError("independent_error_sd must contain only finite positive values.")
    model = build_co2_o2_model(
        observations=observations,
        prior_forward_mean=prior_forward_mean,
        co2_operator=co2_operator,
        o2_operator=o2_operator,
        aggregation_error=prepared.aggregation_error,
        retained_prior=prepared.retained_prior,
        independent_error_sd=independent_error_sd,
        state_activity=state_activity,
    )
    built = RhimeModelBuildResult(
        model=model,
        variable_roles=_CO2_O2_VARIABLE_ROLES,
        metadata=_co2_o2_metadata(prepared),
    )
    trace = sample_rhime_model(
        built,
        _default_co2_o2_sampler() if sampler is None else sampler,
    )
    return _annotate_co2_o2_trace(trace, built=built)
