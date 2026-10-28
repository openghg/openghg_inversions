"""Materialize, construct and annotate the concrete CO2 scientific model.

These explicit operations share numerical meaning between configured full
execution and saved inference. Saved product replay does not construct a model.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import xarray as xr
from dask.array import Array
from dask.base import unpack_collections

from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions._serialization_codecs import _json_value
from openghg_inversions.postprocessing.contracts import OutputContract

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.priors import PriorArgs, prior_records
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.models.site_sigma import add_site_sigma_gaussian_likelihood
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
from openghg_inversions.rhime.materialization import (
    materialize_pymc_inputs,
    _materialize_pymc_inputs_and_priors,
)
from openghg_inversions.rhime.outputs import annotate_likelihood_trace
from openghg_inversions.sigma import SigmaAlignment

from .co2_model import _normalise_offset_args, build_co2_model
from .centered_boundary import OPERAND_NAMES, require_centered_precision, restore_absolute_predictives
from .co2_preparation import Co2PreparedInputs


_CO2_SCIENTIFIC_INPUT_NAMES = (
    "H",
    "alpha_prior_mean",
    "alpha_prior_covariance",
    "fixed_prior_contribution",
    "mf",
    "mf_error",
)


def annotate_co2_trace(
    trace: xr.DataTree,
    built: RhimeModelBuildResult,
    *,
    concentration_units: str | None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
) -> xr.DataTree:
    """Restore absolute CO2 observations/predictions and persist roles and units.

    For centered models, updates numerical ``y`` values in ``observed_data``,
    ``prior_predictive``, ``posterior_predictive`` and ``predictions`` in place
    by adding the retained boundary reference shift. Values marked
    ``concentration_representation="absolute-v1"`` are left unchanged, so
    repeated annotation does not add the shift twice. See
    :ref:`surface-centered-boundary-contract`.

    Args:
        trace: PyMC sample groups represented as a DataTree, annotated in place.
        built: Concrete model with the CO2 variable-role and provenance mapping.
        concentration_units: Observation concentration units, or ``None`` to
            omit concentration-unit attributes.
        likelihood_builder: Optional component whose identity is descriptive
            provenance, never executable replay instructions.
        likelihood_kwargs: Component options described without computing arrays.

    Returns:
        The annotated trace.
    """
    restore_absolute_predictives(trace)
    trace.attrs["rhime_priors"] = json.dumps(prior_records(built.model))
    roles = dict(built.output_contract.variable_roles)
    metadata = dict(built.output_contract.metadata)
    trace.attrs["rhime_variable_roles"] = json.dumps(roles, sort_keys=True)
    trace.attrs["rhime_model_metadata"] = json.dumps(metadata, sort_keys=True)
    if likelihood_builder is not None:
        annotate_likelihood_trace(
            trace, builder_identity=callable_metadata(likelihood_builder),
            likelihood_kwargs=describe_co2_likelihood_kwargs(likelihood_kwargs),
        )

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
        "H_bc_anomaly", "bc_reference", "bc_reference_shift", "boundary_reference_shift",
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
    for group in trace.children.values():
        group.attrs["rhime_recipe"] = "co2"
        for name, variable in group.data_vars.items():
            scientific_roles = sorted(set(roles_by_variable.get(name, ())))
            if scientific_roles:
                variable.attrs["rhime_scientific_roles"] = json.dumps(scientific_roles)
            if name in {
                "bc",
                "H_bc_mass",
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
        names.extend(OPERAND_NAMES if "H_bc_anomaly" in inputs else ("H_bc",))
    names.extend(aggregation_error_input_names(inputs, prepared_inputs.aggregation_error_mode))
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
    prepared = prepared_inputs
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
        raise ValueError("Scalar-sigma likelihood selection requires `eigenbasis_path` and `sigma_prior`.")
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


def describe_co2_likelihood_kwargs(value: Any) -> Any:
    """Describe likelihood options without executing borrowed numerical arrays.

    Labelled arrays record their name, dimensions, shape and units; bare Dask
    arrays record shape and dtype. Consumed numerical values are retained by
    their scientific component; descriptive-only arrays are not materialized here.
    """
    if isinstance(value, xr.DataArray):
        return {
            "name": value.name,
            "dims": list(value.dims),
            "shape": list(value.shape),
            "units": value.attrs.get("units"),
        }
    if isinstance(value, Array):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): describe_co2_likelihood_kwargs(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [describe_co2_likelihood_kwargs(item) for item in value]
    return _json_value(value)


def materialize_co2_inputs(
    *,
    prepared_inputs: Co2PreparedInputs,
    sigma_alignment: SigmaAlignment | None = None,
    sigma_prior: PriorArgs | None = None,
    fixed_model_mismatch: float | xr.DataArray | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    no_model_error: bool = False,
    use_bc: bool = False,
    bc_prior: PriorArgs | None = None,
    bc_state_activity: StateActivity | None = None,
    offset_prior: PriorArgs | None = None,
    offset_args: Mapping[str, Any] | None = None,
) -> tuple[xr.Dataset, dict[str, PriorArgs]]:
    """Jointly materialize selected CO2 inputs, enabled priors and fixed mismatch.

    Return selected scientific arrays and prior parameters keyed by their graph
    variables: ``sigma``, ``bc``, ``offset_latent``, ``sigma_site``,
    ``ou_site_amplitude`` or ``sigma_global``. Explicit labelled fixed mismatch
    is installed in the returned Dataset. Construction consumes these numerical
    values separately from the unchanged model choices. Unselected arrays and
    arbitrary custom likelihood options remain borrowed and uncomputed.

    Args:
        prepared_inputs: Borrowed coherent-reduction preparation.
        sigma_alignment: Additive-error grouping; omitted grouping is derived
            from the retained observation sites during construction.
        sigma_prior: Optional additive-error prior; omitted parameters use the
            component defaults.
        fixed_model_mismatch: Known concentration standard deviation, either a
            scalar or an exactly aligned observation vector. Omission retains
            a prepared mismatch field when present.
        likelihood_builder: Ordinary component replacing additive-error options.
        likelihood_kwargs: Selected component options. Built-in numerical
            priors participate in this boundary; custom components own any
            additional numerical option materialization.
        no_model_error: Omit inferred additive error; cannot accompany a sigma
            prior, grouping or replacement likelihood.
        use_bc: Include prepared boundary sensitivity.
        bc_prior: Optional boundary-scale prior, enabled only with ``use_bc``.
        bc_state_activity: Optional boundary active/fixed-state policy.
        offset_prior: Optional prior enabling concentration offsets.
        offset_args: Offset frequency, ``per_site`` and ``drop_first`` settings.

    Returns:
        Borrowed Dataset copy with selected arrays eager, plus eager prior
        parameters. The preparation and supplied parameter mappings are unchanged.

    Raises:
        ValueError: If model choices conflict, required prepared arrays are
            absent, or a labelled mismatch does not align exactly.
    """
    if no_model_error and (sigma_alignment is not None or sigma_prior is not None):
        raise ValueError("`no_model_error=True` cannot be combined with `sigma_alignment` or `sigma_prior`.")
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
    prepared = prepared_inputs
    if use_bc and "H_bc_anomaly" in prepared.inv_inputs:
        if likelihood_builder is not None:
            raise ValueError("Centered boundaries do not support custom or fixed-OU likelihoods.")
        require_centered_precision()
    names = co2_model_input_names(
        prepared,
        preserve_prepared_fixed_mismatch=(likelihood_builder is None and fixed_model_mismatch is None),
        use_bc=use_bc,
    )
    backend_inputs = prepared.rhime_inputs
    if isinstance(fixed_model_mismatch, xr.DataArray):
        fixed_model_mismatch, _ = xr.align(fixed_model_mismatch, prepared.inv_inputs["mf"], join="exact")
        backend_inputs = replace(backend_inputs, inv_inputs=prepared.inv_inputs.assign(
            fixed_model_mismatch=fixed_model_mismatch,
        ))
        names = (*names, "fixed_model_mismatch")
    priors = {}
    if likelihood_builder is None and not no_model_error and sigma_prior is not None:
        priors["sigma"] = sigma_prior
    if use_bc and bc_prior is not None:
        priors["bc"] = bc_prior
    if offset_prior is not None:
        priors["offset_latent"] = offset_prior
    likelihood_options = likelihood_kwargs or {}
    if likelihood_builder is add_scalar_sigma_eigen_likelihood and "sigma_prior" in likelihood_options:
        priors["sigma_global"] = likelihood_options["sigma_prior"]
    elif likelihood_builder is add_site_sigma_gaussian_likelihood and "site_amplitude_prior" in likelihood_options:
        priors["sigma_site"] = likelihood_options["site_amplitude_prior"]
    elif likelihood_builder is add_fixed_ou_gaussian_likelihood and "site_amplitude_prior" in likelihood_options:
        priors["ou_site_amplitude"] = likelihood_options["site_amplitude_prior"]
    return _materialize_pymc_inputs_and_priors(
        backend_inputs, variable_names=names, prior_parameters=priors,
    )


def build_co2_from_materialized(
    model_inputs: xr.Dataset,
    *,
    prepared_inputs: Co2PreparedInputs,
    priors: Mapping[str, PriorArgs],
    sigma_alignment: SigmaAlignment | None = None,
    sigma_prior: PriorArgs | None = None,
    fixed_model_mismatch: float | xr.DataArray | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    no_model_error: bool = False,
    use_bc: bool = False,
    bc_prior: PriorArgs | None = None,
    bc_state_activity: StateActivity | None = None,
    bc_freq: str | None = None,
    bc_anchor_time: DatetimeLike | None = None,
    offset_prior: PriorArgs | None = None,
    offset_args: Mapping[str, Any] | None = None,
) -> RhimeModelBuildResult:
    """Construct the CO2 model from explicitly materialized scientific inputs.

    ``prepared_inputs`` supplies aggregation mode and descriptive provenance.
    ``priors`` supplies selected eager numerical parameters under the graph names
    returned by :func:`~openghg_inversions.rhime.co2.co2_runner.materialize_co2_inputs`. Prior arguments in the model
    choices select components, but their borrowed numerical values are not used.
    This operation performs construction only; saved replay never calls it.
    Remaining model choices have the semantics documented by
    :func:`~openghg_inversions.rhime.co2.co2_runner.materialize_co2_inputs`. A configured scalar-sigma eigenbasis is
    loaded and checked against observations, error and aggregation covariance
    before its likelihood is constructed.

    Args:
        model_inputs: Selected scientific arrays already materialized together.
        prepared_inputs: Preparation supplying aggregation mode and provenance.
        priors: Materialized prior parameters under their concrete graph names.
        bc_freq: Optional model-owned frequency for raw boundary periods.
        bc_anchor_time: Optional timestamp anchoring raw boundary periods.

    Returns:
        Constructed model with scientific variable roles and bounded metadata.

    Raises:
        ValueError: If selected inputs or numerical priors remain lazy, or
            scientific components reject the supplied choices.
    """
    prepared = prepared_inputs
    if use_bc and "H_bc_anomaly" in prepared.inv_inputs:
        if likelihood_builder is not None:
            raise ValueError("Centered boundaries do not support custom or fixed-OU likelihoods.")
        require_centered_precision()
    names = co2_model_input_names(
        prepared, use_bc=use_bc,
        preserve_prepared_fixed_mismatch=likelihood_builder is None and fixed_model_mismatch is None,
    )
    prepared_mismatch = (
        model_inputs.get("fixed_model_mismatch")
        if likelihood_builder is None and (fixed_model_mismatch is None or isinstance(fixed_model_mismatch, xr.DataArray))
        else fixed_model_mismatch
    )
    lazy, _ = unpack_collections((
        *(model_inputs[name] for name in names), priors, prepared_mismatch,
    ))
    if lazy:
        raise ValueError("CO2 model inputs and enabled priors must be materialized before building.")
    if likelihood_builder is None and not no_model_error and sigma_alignment is None:
        sigma_alignment = SigmaAlignment.from_observations(model_inputs["mf"])
    aggregation_error = resolve_aggregation_error(
        model_inputs,
        prepared.aggregation_error_mode,
    )
    options = dict(likelihood_kwargs or {})
    if likelihood_builder is add_scalar_sigma_eigen_likelihood and "sigma_prior" in options:
        options["sigma_prior"] = priors["sigma_global"]
    elif likelihood_builder is add_site_sigma_gaussian_likelihood and "site_amplitude_prior" in options:
        options["site_amplitude_prior"] = priors["sigma_site"]
    elif likelihood_builder is add_fixed_ou_gaussian_likelihood and "site_amplitude_prior" in options:
        options["site_amplitude_prior"] = priors["ou_site_amplitude"]
    model_likelihood_kwargs = _resolve_co2_likelihood_kwargs(
        likelihood_builder,
        options if likelihood_kwargs is not None else None,
        observations=model_inputs["mf"],
        observation_error=model_inputs["mf_error"],
        aggregation_error=aggregation_error,
    )
    prior_covariance = model_inputs["alpha_prior_covariance"]
    retained_prior = CorrelatedLognormalPrior(
        model_inputs["alpha_prior_mean"],
        prior_covariance,
        covariance_dim=str(prior_covariance.dims[-1]),
    )
    boundary_operands = model_inputs[list(OPERAND_NAMES)] if use_bc and "H_bc_anomaly" in model_inputs else None
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
        sigma_prior=priors["sigma"] if sigma_prior is not None else None,
        fixed_model_mismatch=prepared_mismatch,
        state_activity=_state_activity_from_inputs(model_inputs),
        boundary_sensitivity=model_inputs.get("H_bc") if use_bc and boundary_operands is None else None,
        boundary_operands=boundary_operands,
        bc_freq=bc_freq,
        bc_anchor_time=bc_anchor_time,
        bc_prior=priors["bc"] if use_bc and bc_prior is not None else None,
        bc_state_activity=bc_state_activity,
        offset_prior=priors["offset_latent"] if offset_prior is not None else None,
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
    if use_bc and "H_bc_anomaly" in model_inputs:
        variable_roles.pop("boundary_sensitivity", None)
        variable_roles.update(boundary_anomaly="H_bc_anomaly", boundary_mass="H_bc_mass", boundary_reference="bc_reference")
    if offset_prior is not None:
        variable_roles["offset_concentration"] = "offset"
    for role, name in {
        "active_flux_scale": "flux_scaling_active",
        "offset_coefficient": "offset_latent",
    }.items():
        if name in model.named_vars:
            variable_roles[role] = name
    resolved_offset_args = None
    if offset_prior is not None:
        frequency, drop_first, per_site = _normalise_offset_args(offset_args)
        resolved_offset_args = {"offset_freq": frequency, "drop_first": drop_first, "per_site": per_site}
    built = RhimeModelBuildResult(
        model=model,
        output_contract=OutputContract(
            variable_roles=variable_roles,
            metadata={
                "recipe": "co2",
                "kind": "builtin",
                "prior": "correlated arithmetic-moment lognormal",
                "no_model_error": no_model_error,
                "use_bc": use_bc,
                **({
                    "boundary_representation": "centered-v1",
                    "bc_frequency": bc_freq,
                    "bc_anchor_time": None if bc_anchor_time is None else str(bc_anchor_time),
                } if boundary_operands is not None else {}),
                "add_offset": offset_prior is not None,
                "offset_args": resolved_offset_args,
                "likelihood": (
                    "additive_sigma" if likelihood_builder is None else
                    "site_sigma" if likelihood_builder is add_site_sigma_gaussian_likelihood else
                    "fixed_ou" if likelihood_builder is add_fixed_ou_gaussian_likelihood else
                    "scalar_sigma" if likelihood_builder is add_scalar_sigma_eigen_likelihood else "custom"
                ),
                "fixed_model_mismatch": "fixed_model_mismatch" if "fixed_model_mismatch" in model.named_vars else None,
                "basis_artifact_source": getattr(prepared, "basis_artifact_source", "unknown"),
                "basis_artifact_path": getattr(prepared, "basis_artifact_path", None),
            },
        ),
    )
    return built



__all__ = [
    "annotate_co2_trace",
    "build_co2_from_materialized",
    "materialize_co2_inputs",
    "co2_model_input_names",
    "prepare_co2_scalar_sigma_eigenbasis",
]
