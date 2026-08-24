"""Readable prepared-input runner for the CO2 RHIME recipe."""

from __future__ import annotations

from collections import defaultdict
import json

import arviz as az
import xarray as xr

from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import (
    AggregationErrorMode,
    OBSERVATION_ERROR_INPUT_NAMES,
    aggregation_error_input_names,
    resolve_aggregation_error,
)
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.materialization import materialize_pymc_inputs
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model
from openghg_inversions.sigma import SigmaAlignment

from .co2_model import build_co2_model


_CO2_SCIENTIFIC_INPUT_NAMES = (
    "H",
    "alpha_prior_mean",
    "alpha_prior_covariance",
    "fixed_prior_contribution",
    *OBSERVATION_ERROR_INPUT_NAMES,
)


def _annotate_co2_trace(
    trace: az.InferenceData,
    built: RhimeModelBuildResult,
    *,
    concentration_units: str | None,
) -> az.InferenceData:
    """Persist the CO2 scientific manifest on a sampled trace."""
    roles = dict(built.variable_roles)
    metadata = dict(built.metadata)
    trace.attrs["rhime_variable_roles"] = json.dumps(roles, sort_keys=True)
    trace.attrs["rhime_model_metadata"] = json.dumps(metadata, sort_keys=True)

    roles_by_variable: dict[str, list[str]] = defaultdict(list)
    for role, variable_name in roles.items():
        roles_by_variable[variable_name].append(role)
    # PyMC data names differ from prepared-array role names. Keep those
    # concrete scientific identities visible on stored constants too.
    concrete_roles = {
        "error": ["observation_error"],
        "min_error": ["minimum_error"],
        "fixed_model_mismatch": ["fixed_model_mismatch"],
        "fixed_prior_contribution": ["coherent_prior_contribution"],
        "co2_sensitivity": ["emissions_sensitivity"],
        "modelled_concentration": ["model_mean"],
        "co2_flux_contribution": ["pollution_concentration"],
        "flux_scaling": ["flux_scale"],
        "flux_scaling_active": ["active_flux_scale"],
        "outer_sensitivity": ["outer_emissions_sensitivity"],
        "outer_flux_contribution": ["outer_pollution_concentration"],
        "outer_flux_scaling": ["outer_flux_scale"],
        "outer_flux_scaling_active": ["active_outer_flux_scale"],
        "hbc": ["boundary_sensitivity"],
        "bc": ["boundary_scale"],
        "mu_bc": ["boundary_concentration"],
        "epsilon": ["model_error"],
        "y": ["concentration"],
    }
    for name, scientific_roles in concrete_roles.items():
        roles_by_variable[name].extend(scientific_roles)

    concentration_variables = {
        "error",
        "min_error",
        "fixed_model_mismatch",
        "fixed_prior_contribution",
        "modelled_concentration",
        "co2_flux_contribution",
        "outer_flux_contribution",
        "mu_bc",
        "epsilon",
        "y",
    }
    for group_name in trace.groups():
        group = getattr(trace, group_name)
        if not isinstance(group, xr.Dataset):
            continue
        group.attrs["rhime_recipe"] = "co2"
        for name, variable in group.data_vars.items():
            scientific_roles = sorted(set(roles_by_variable.get(name, ())))
            if scientific_roles:
                variable.attrs["rhime_scientific_roles"] = json.dumps(scientific_roles)
            if name in {
                "bc",
                "flux_scaling",
                "flux_scaling_active",
                "outer_flux_scaling",
                "outer_flux_scaling_active",
            }:
                variable.attrs["units"] = "1"
            elif concentration_units is not None and name in concentration_variables:
                variable.attrs["units"] = concentration_units
    return trace


def _state_activity_from_inputs(model_inputs: xr.Dataset) -> StateActivity | None:
    """Return an exact prepared activity policy when one is present."""
    if "state_is_active" not in model_inputs:
        return None
    fixed_value: float | xr.DataArray = 1.0
    if "state_fixed_value" in model_inputs:
        fixed_value = model_inputs["state_fixed_value"]
    return StateActivity(
        active=model_inputs["state_is_active"],
        fixed_value=fixed_value,
    )


def co2_model_input_names(
    prepared_inputs: RhimePreparedInputs,
    *,
    aggregation_error_mode: AggregationErrorMode,
    preserve_prepared_fixed_mismatch: bool,
    derive_sigma_alignment: bool = False,
) -> tuple[str, ...]:
    """Declare prepared arrays consumed by the selected CO2 components.

    Args:
        prepared_inputs: Prepared RHIME artifact containing the candidate
            inversion inputs.
        aggregation_error_mode: Aggregation-error representation selected for
            the likelihood.
        preserve_prepared_fixed_mismatch: Include a prepared fixed mismatch
            field when present.
        derive_sigma_alignment: Include the site indicator needed to derive
            the default mismatch-error grouping.

    Returns:
        Names of the arrays to materialize for model construction.

    Raises:
        ValueError: If a required prepared input is absent.
    """
    inputs = prepared_inputs.inv_inputs
    names = list(_CO2_SCIENTIFIC_INPUT_NAMES)
    names.extend(aggregation_error_input_names(inputs, aggregation_error_mode))
    if derive_sigma_alignment:
        names.append("site_indicator")
    if "state_is_active" in inputs:
        names.append("state_is_active")
        if "state_fixed_value" in inputs:
            names.append("state_fixed_value")
    if preserve_prepared_fixed_mismatch and "fixed_model_mismatch" in inputs:
        names.append("fixed_model_mismatch")
    missing = [name for name in names if name not in inputs]
    if missing:
        raise ValueError(f"CO2 prepared inputs are missing required variable(s): {missing!r}.")
    return tuple(names)


def run_rhime_co2(
    *,
    prepared_inputs: RhimePreparedInputs,
    sigma_alignment: SigmaAlignment | None = None,
    sigma_prior: PriorArgs | None = None,
    fixed_model_mismatch: float | xr.DataArray | None = None,
    sampler: RhimeSampler | None = None,
    aggregation_error_mode: AggregationErrorMode = "dense",
    no_model_error: bool = False,
) -> az.InferenceData:
    """Materialize, build, and sample the CO2 coherent-reduction model.

    This callable is the public production replay seam for an already
    validated :class:`RhimePreparedInputs` artifact. It alone unpacks the
    prepared dataset; the model builder receives named scientific arrays.

    ``fixed_model_mismatch=None`` preserves a prepared fixed-mismatch field if
    present, otherwise omits the term. An explicit scalar or labelled vector
    overrides prepared data. By default, inferred model error varies by site
    over one shared time period. An explicit ``sigma_alignment`` overrides
    that alignment; ``no_model_error=True`` disables inferred model error.
    The Verification Games fixed-likelihood harness passes 1 ppm and disables
    inferred model error.

    Args:
        prepared_inputs: Validated coherent-reduction inputs for the CO2
            recipe.
        sigma_alignment: Optional grouping policy for inferred additive model
            error. The default is derived from the prepared site indicator.
        sigma_prior: Optional prior arguments for inferred additive model
            error.
        fixed_model_mismatch: Optional known scalar or labelled mismatch
            standard deviation. When omitted, a prepared value is preserved.
        sampler: Optional RHIME sampler configuration.
        aggregation_error_mode: Prepared aggregation-error representation to
            use in the likelihood.
        no_model_error: If true, omit inferred additive model error.

    Returns:
        Sampled inference data annotated with the CO2 variable-role and model
        manifests.

    Raises:
        ValueError: If prepared inputs are missing, inconsistent, or fail
            model construction.
    """
    prepared = prepared_inputs.validated()
    names = co2_model_input_names(
        prepared,
        aggregation_error_mode=aggregation_error_mode,
        preserve_prepared_fixed_mismatch=fixed_model_mismatch is None,
        derive_sigma_alignment=not no_model_error and sigma_alignment is None,
    )
    model_inputs = materialize_pymc_inputs(prepared, variable_names=names)
    if not no_model_error and sigma_alignment is None:
        sigma_alignment = SigmaAlignment.from_frequency(model_inputs["site_indicator"])
    aggregation_error = resolve_aggregation_error(
        model_inputs,
        aggregation_error_mode,
    )
    prepared_mismatch = (
        model_inputs.get("fixed_model_mismatch") if fixed_model_mismatch is None else fixed_model_mismatch
    )
    model = build_co2_model(
        model_inputs["H"],
        prior_mean=model_inputs["alpha_prior_mean"],
        prior_covariance=model_inputs["alpha_prior_covariance"],
        fixed_prior_contribution=model_inputs["fixed_prior_contribution"],
        observations=model_inputs["mf"],
        observation_error=model_inputs["mf_error"],
        minimum_error=model_inputs["min_error"],
        aggregation_error=aggregation_error,
        sigma_alignment=sigma_alignment,
        sigma_prior=sigma_prior,
        fixed_model_mismatch=prepared_mismatch,
        state_activity=_state_activity_from_inputs(model_inputs),
        no_model_error=no_model_error,
    )
    built = RhimeModelBuildResult(
        model=model,
        variable_roles={
            "observation": "y",
            "observation_error": "mf_error",
            "minimum_error": "min_error",
            "concentration": "y",
            "model_error": "epsilon",
            "model_mean": "modelled_concentration",
            "pollution_concentration": "co2_flux_contribution",
            "flux_scale": "flux_scaling",
            "emissions_sensitivity": "co2_sensitivity",
            "coherent_prior_contribution": "fixed_prior_contribution",
        },
        metadata={
            "recipe": "co2",
            "kind": "builtin",
            "prior": "correlated arithmetic-moment lognormal",
            "basis_artifact_source": getattr(prepared, "basis_artifact_source", "unknown"),
            "basis_artifact_path": getattr(prepared, "basis_artifact_path", None),
        },
    )
    trace = sample_rhime_model(built, RhimeSampler() if sampler is None else sampler)
    return _annotate_co2_trace(
        trace,
        built,
        concentration_units=model_inputs["mf"].attrs.get("units"),
    )


__all__ = ["co2_model_input_names", "run_rhime_co2"]
