"""Readable prepared-input runner for the CO2 RHIME recipe."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import arviz as az
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.scalar_sigma import (
    ScalarSigmaEigenbasis,
    add_scalar_sigma_eigen_likelihood,
    load_scalar_sigma_eigenbasis,
    prepare_scalar_sigma_eigenbasis,
)
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import (
    AggregationError,
    aggregation_error_input_names,
    resolve_aggregation_error,
)
from openghg_inversions.rhime.builders import (
    RhimeLikelihoodBuilder,
    RhimeModelBuildResult,
    callable_metadata,
)
from openghg_inversions.rhime.materialization import materialize_pymc_inputs
from openghg_inversions.rhime.outputs import annotate_likelihood_trace
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model
from openghg_inversions.sigma import SigmaAlignment

from .co2_model import build_co2_model
from .co2_preparation import Co2PreparedInputs


_CO2_SCIENTIFIC_INPUT_NAMES = (
    "H",
    "alpha_prior_mean",
    "alpha_prior_covariance",
    "fixed_prior_contribution",
    "mf",
    "mf_error",
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
        "fixed_model_mismatch": ["fixed_model_mismatch"],
        "fixed_prior_contribution": ["coherent_prior_contribution"],
        "co2_sensitivity": ["emissions_sensitivity"],
        "modelled_concentration": ["model_mean"],
        "co2_flux_contribution": ["pollution_concentration"],
        "flux_scaling": ["flux_scale"],
        "flux_scaling_active": ["active_flux_scale"],
        "hbc": ["boundary_sensitivity"],
        "bc": ["boundary_scale"],
        "mu_bc": ["boundary_concentration"],
        "offset_latent": ["offset_coefficient"],
        "offset": ["offset_concentration"],
        "sigma_global": ["global_iid_mismatch_standard_deviation"],
        "epsilon": ["model_error"],
        "y": ["concentration"],
    }
    for name, scientific_roles in concrete_roles.items():
        roles_by_variable[name].extend(scientific_roles)

    concentration_variables = {
        "error",
        "fixed_model_mismatch",
        "fixed_prior_contribution",
        "modelled_concentration",
        "co2_flux_contribution",
        "mu_bc",
        "offset_latent",
        "offset",
        "sigma_global",
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
    prepared_inputs: Co2PreparedInputs,
    *,
    preserve_prepared_fixed_mismatch: bool,
    use_bc: bool = False,
) -> tuple[str, ...]:
    """Declare prepared arrays consumed by the selected CO2 components.

    Args:
        prepared_inputs: Prepared RHIME artifact containing the candidate
            inversion inputs.
        preserve_prepared_fixed_mismatch: Include a prepared fixed mismatch
            field when present.
        use_bc: Include the prepared boundary-condition sensitivity.

    Returns:
        Names of the arrays to materialize for model construction.

    Raises:
        ValueError: If a required prepared input is absent.
    """
    inputs = prepared_inputs.inv_inputs
    names = list(_CO2_SCIENTIFIC_INPUT_NAMES)
    if use_bc:
        names.append("H_bc")
    names.extend(
        aggregation_error_input_names(inputs, prepared_inputs.aggregation_error_mode)
    )
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


def prepare_co2_scalar_sigma_eigenbasis(
    prepared_inputs: Co2PreparedInputs,
) -> ScalarSigmaEigenbasis:
    """Prepare the scalar-sigma cache value from labelled CO2 inputs.

    This is the visible eager boundary for the cache. The observations,
    reported error, and selected aggregation-error representation are
    materialized together before the dense eigendecomposition.

    Args:
        prepared_inputs: Validated prepared inputs for the CO2-only recipe.

    Returns:
        Labelled eigenbasis ready to save or pass to
        :func:`~openghg_inversions.models.scalar_sigma.add_scalar_sigma_eigen_likelihood`.

    Raises:
        ValueError: If required CO2 inputs are missing, misaligned, or cannot
            form a valid scalar-sigma base covariance.
    """
    prepared = prepared_inputs.validated()
    names = (
        "mf",
        "mf_error",
        *aggregation_error_input_names(
            prepared.inv_inputs,
            prepared.aggregation_error_mode,
        ),
    )
    model_inputs = materialize_pymc_inputs(prepared.rhime_inputs, variable_names=names)
    aggregation_error = resolve_aggregation_error(
        model_inputs,
        prepared.aggregation_error_mode,
    )
    return prepare_scalar_sigma_eigenbasis(
        observations=model_inputs["mf"],
        observation_error=model_inputs["mf_error"],
        aggregation_error=aggregation_error,
    )


def _resolve_co2_likelihood_kwargs(
    likelihood_builder: RhimeLikelihoodBuilder | None,
    likelihood_kwargs: Mapping[str, Any] | None,
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
) -> Mapping[str, Any] | None:
    """Resolve file-backed scalar-sigma options before model construction."""
    if likelihood_builder is not add_scalar_sigma_eigen_likelihood:
        return likelihood_kwargs
    options = dict(likelihood_kwargs or {})
    allowed = {"eigenbasis_path", "sigma_prior"}
    if unexpected := sorted(options.keys() - allowed):
        raise ValueError(f"Unexpected scalar-sigma likelihood option(s): {unexpected!r}.")
    if "eigenbasis_path" not in options or "sigma_prior" not in options:
        raise ValueError(
            "Scalar-sigma likelihood selection requires `eigenbasis_path` and `sigma_prior`."
        )
    if not isinstance(options["eigenbasis_path"], str | Path):
        raise TypeError("Scalar-sigma `eigenbasis_path` must be a string or Path.")
    return {
        "eigenbasis": load_scalar_sigma_eigenbasis(
            options["eigenbasis_path"],
            observations=observations,
            observation_error=observation_error,
            aggregation_error=aggregation_error,
        ),
        "sigma_prior": options["sigma_prior"],
    }


def run_rhime_co2(
    *,
    prepared_inputs: Co2PreparedInputs,
    sigma_alignment: SigmaAlignment | None = None,
    sigma_prior: PriorArgs | None = None,
    fixed_model_mismatch: float | xr.DataArray | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    sampler: RhimeSampler | None = None,
    no_model_error: bool = False,
    use_bc: bool = False,
    bc_prior: PriorArgs | None = None,
    bc_state_activity: StateActivity | None = None,
    offset_prior: PriorArgs | None = None,
    offset_args: Mapping[str, Any] | None = None,
) -> az.InferenceData:
    """Materialize, build, and sample the CO2 coherent-reduction model.

    This callable is the public production replay seam for an already
    validated :class:`Co2PreparedInputs` artifact. It alone unpacks the
    prepared dataset and constructs the complete retained prior; the model
    builder receives named scientific values.

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
        likelihood_builder: Optional ordinary likelihood component replacing
            the default additive-sigma likelihood.
        likelihood_kwargs: Options passed only to the selected likelihood. For
            the scalar-sigma eigen likelihood, pass an ``eigenbasis_path`` and
            ``sigma_prior``; the cache is loaded before model construction.
        sampler: Optional RHIME sampler configuration.
        no_model_error: If true, omit inferred additive model error.
        use_bc: Whether to include prepared ``H_bc`` boundary sensitivity.
        bc_prior: Optional prior for boundary-condition scaling.
        bc_state_activity: Optional active/fixed boundary-state policy.
        offset_prior: Optional prior for an offset component. When omitted, no
            offset is added.
        offset_args: Optional offset settings: ``offset_freq``, ``drop_first``,
            and ``per_site``.

    Returns:
        Sampled inference data annotated with the CO2 variable-role and model
        manifests.

    Raises:
        ValueError: If model-error options contradict ``no_model_error``, or
            prepared inputs are missing, inconsistent, or fail model
            construction.
    """
    if no_model_error and (sigma_alignment is not None or sigma_prior is not None):
        raise ValueError(
            "`no_model_error=True` cannot be combined with `sigma_alignment` or `sigma_prior`."
        )
    if likelihood_builder is None and likelihood_kwargs:
        raise ValueError("likelihood_kwargs require likelihood_builder.")
    if not use_bc and (bc_prior is not None or bc_state_activity is not None):
        raise ValueError("bc_prior and bc_state_activity require use_bc=True.")
    if offset_prior is None and offset_args:
        raise ValueError("offset_args require offset_prior.")
    if likelihood_builder is not None and (
        no_model_error
        or sigma_alignment is not None
        or sigma_prior is not None
        or fixed_model_mismatch is not None
    ):
        raise ValueError(
            "likelihood_builder replaces the default model-error options; "
            "pass its options in likelihood_kwargs."
        )
    prepared = prepared_inputs.validated()
    names = co2_model_input_names(
        prepared,
        preserve_prepared_fixed_mismatch=(likelihood_builder is None and fixed_model_mismatch is None),
        use_bc=use_bc,
    )
    model_inputs = materialize_pymc_inputs(prepared.rhime_inputs, variable_names=names)
    if likelihood_builder is None and not no_model_error and sigma_alignment is None:
        sigma_alignment = SigmaAlignment.from_observations(model_inputs["mf"])
    aggregation_error = resolve_aggregation_error(
        model_inputs,
        prepared.aggregation_error_mode,
    )
    model_likelihood_kwargs = _resolve_co2_likelihood_kwargs(
        likelihood_builder,
        likelihood_kwargs,
        observations=model_inputs["mf"],
        observation_error=model_inputs["mf_error"],
        aggregation_error=aggregation_error,
    )
    prepared_mismatch = (
        model_inputs.get("fixed_model_mismatch")
        if likelihood_builder is None and fixed_model_mismatch is None
        else fixed_model_mismatch
    )
    prior_covariance = model_inputs["alpha_prior_covariance"]
    retained_prior = CorrelatedLognormalPrior(
        model_inputs["alpha_prior_mean"],
        prior_covariance,
        covariance_dim=str(prior_covariance.dims[-1]),
    )
    model = build_co2_model(
        model_inputs["H"],
        retained_prior=retained_prior,
        fixed_prior_contribution=model_inputs["fixed_prior_contribution"],
        observations=model_inputs["mf"],
        observation_error=model_inputs["mf_error"],
        aggregation_error=aggregation_error,
        likelihood_builder=likelihood_builder,
        likelihood_kwargs=model_likelihood_kwargs,
        sigma_alignment=sigma_alignment,
        sigma_prior=sigma_prior,
        fixed_model_mismatch=prepared_mismatch,
        state_activity=_state_activity_from_inputs(model_inputs),
        boundary_sensitivity=model_inputs.get("H_bc") if use_bc else None,
        bc_prior=bc_prior,
        bc_state_activity=bc_state_activity,
        offset_prior=offset_prior,
        offset_args=offset_args,
    )
    variable_roles = {
        "observation": "y",
        "concentration": "y",
        "model_error": "epsilon",
        "model_mean": "modelled_concentration",
        "pollution_concentration": "co2_flux_contribution",
        "flux_scale": "flux_scaling",
        "emissions_sensitivity": "co2_sensitivity",
        "coherent_prior_contribution": "fixed_prior_contribution",
    }
    if "error" in model.named_vars:
        variable_roles["observation_error"] = "error"
    if use_bc:
        variable_roles.update(
            {
                "boundary_concentration": "mu_bc",
                "boundary_scale": "bc",
                "boundary_sensitivity": "hbc",
            }
        )
    if offset_prior is not None:
        variable_roles["offset_concentration"] = "offset"
    built = RhimeModelBuildResult(
        model=model,
        variable_roles=variable_roles,
        metadata={
            "recipe": "co2",
            "kind": "builtin",
            "prior": "correlated arithmetic-moment lognormal",
            "basis_artifact_source": getattr(prepared, "basis_artifact_source", "unknown"),
            "basis_artifact_path": getattr(prepared, "basis_artifact_path", None),
        },
    )
    trace = sample_rhime_model(built, RhimeSampler() if sampler is None else sampler)
    trace = _annotate_co2_trace(
        trace,
        built,
        concentration_units=model_inputs["mf"].attrs.get("units"),
    )
    if likelihood_builder is not None:
        annotate_likelihood_trace(
            trace,
            builder_identity=callable_metadata(likelihood_builder),
            likelihood_kwargs=likelihood_kwargs,
        )
    return trace


__all__ = [
    "co2_model_input_names",
    "prepare_co2_scalar_sigma_eigenbasis",
    "run_rhime_co2",
]
