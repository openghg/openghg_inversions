"""Materialize, construct and annotate the concrete CO2 scientific model.

These explicit operations share numerical meaning between configured full
execution and saved inference. Saved product replay does not construct a model.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import xarray as xr
from dask.array import Array
from dask import compute as dask_compute
from dask.base import unpack_collections

from openghg_inversions.biosphere_prior import BiospherePrior
from openghg_inversions.models.biosphere import annotate_biosphere_prior
from openghg_inversions._serialization_codecs import _json_value
from openghg_inversions.array_ops import to_dense_tree
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.inversion_data import PreparedInputs
from openghg_inversions.models.boundary import Boundary, BoundaryChoice, MultiplicativeChoice, RawCenteredTransport
from openghg_inversions.models.fixed_ou import add_fixed_ou_gaussian_likelihood
from openghg_inversions.models.priors import PriorArgs, prior_records
from openghg_inversions.models.scalar_sigma import (
    ScalarSigmaEigenbasis,
    add_scalar_sigma_eigen_likelihood,
    load_scalar_sigma_eigenbasis,
    prepare_scalar_sigma_eigenbasis,
)
from openghg_inversions.models.site_sigma import add_site_sigma_gaussian_likelihood
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import (
    AggregationError,
    aggregation_error_input_names,
    resolve_aggregation_error,
)
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime.builders import (
    RhimeLikelihoodBuilder,
    RhimeModelBuildResult,
    callable_metadata,
)
from openghg_inversions.rhime.materialization import (
    materialize_pymc_inputs,
)
from openghg_inversions.rhime.outputs import annotate_likelihood_trace
from openghg_inversions.sigma import SigmaAlignment

from .boundary_output import BOUNDARY_FORMAT_VERSION, restore_absolute_predictives
from .co2_model import _normalise_offset_args, build_co2_model
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
    biosphere_prior: BiospherePrior | None = None,
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
        biosphere_prior: Optional resolved calibration to persist with the trace.
        concentration_units: Observation concentration units, or ``None`` to
            omit concentration-unit attributes.
        likelihood_builder: Optional component whose identity is descriptive
            provenance, never executable replay instructions.
        likelihood_kwargs: Component options described without computing arrays.

    Returns:
        The annotated trace.
    """
    trace.attrs["rhime_boundary_format_version"] = BOUNDARY_FORMAT_VERSION
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
        "offset_latent": ["offset_coefficient"],
        "offset": ["offset_concentration"],
        "sigma_global": ["global_iid_mismatch_standard_deviation"],
        "epsilon": ["model_error"],
        "y": ["concentration"],
    }
    for name, scientific_roles in concrete_roles.items():
        roles_by_variable[name].extend(scientific_roles)

    concentration_variables = {
        "boundary_reference_shift",
        "error",
        "fixed_model_mismatch",
        "fixed_prior_contribution",
        "modelled_concentration",
        "co2_flux_contribution",
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
                "flux_scaling",
                "flux_scaling_active",
            }:
                variable.attrs["units"] = "1"
            elif concentration_units is not None and name in concentration_variables:
                variable.attrs["units"] = concentration_units
            if name in built.output_contract.metadata.get("boundary_units", {}):
                variable.attrs["units"] = built.output_contract.metadata["boundary_units"][name]
    return annotate_biosphere_prior(trace, biosphere_prior)


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
) -> tuple[str, ...]:
    """Declare prepared non-boundary arrays consumed by the CO2 components.

    Boundary transport is materialized with its bound boundary instead.

    Args:
        prepared_inputs: Prepared RHIME artifact containing the candidate
            inversion inputs.
        preserve_prepared_fixed_mismatch: Include a prepared fixed mismatch
            field when present.

    Returns:
        Names of the arrays to materialize for model construction.

    Raises:
        ValueError: If a required prepared input is absent.
    """
    inputs = prepared_inputs.inv_inputs
    names = list(_CO2_SCIENTIFIC_INPUT_NAMES)
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


def _bind_co2_boundary(
    inputs: xr.Dataset, boundary: BoundaryChoice | None,
) -> Boundary | None:
    """Bind a boundary choice once to the prepared absolute or centered transport.

    The prepared dataset identifies its representation here; absolute ``H_bc``
    is bound only by choices that accept it. CO2 prepared inputs establish
    :class:`RawCenteredTransport` for centered inputs when constructed or
    loaded, so the dataset is annotated here without validating it again.
    """
    if boundary is None:
        return None
    if "H_bc_anomaly" in inputs:
        return boundary.bind(inputs["mf"], cast(RawCenteredTransport, inputs))
    if "H_bc" not in inputs:
        raise ValueError("A boundary choice requires prepared H_bc or centered boundary operands.")
    if not isinstance(boundary, MultiplicativeChoice):
        raise ValueError(f"{type(boundary).__name__} requires centered boundary operands; prepare them again.")
    return boundary.bind(inputs["mf"], inputs["H_bc"])


def _materialize_co2_inputs(
    prepared: PreparedInputs, names: Collection[str], priors: Mapping[str, PriorArgs],
    boundary: Boundary | None,
) -> tuple[xr.Dataset, dict[str, PriorArgs], Boundary | None]:
    """Compute selected arrays, the bound boundary and prior parameters together."""
    selected = prepared.inv_inputs[list(names)]
    data, eager_priors, boundary = dask_compute(*to_dense_tree((selected, dict(priors), boundary)))
    variables = dict(prepared.inv_inputs.variables)
    variables.update(data.variables)
    return prepared.inv_inputs._replace(variables=variables), eager_priors, boundary


def materialize_co2_inputs(
    *,
    prepared_inputs: Co2PreparedInputs,
    sigma_alignment: SigmaAlignment | None = None,
    sigma_prior: PriorArgs | None = None,
    fixed_model_mismatch: float | xr.DataArray | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    no_model_error: bool = False,
    boundary: BoundaryChoice | None = None,
    offset_prior: PriorArgs | None = None,
    offset_args: Mapping[str, Any] | None = None,
) -> tuple[xr.Dataset, dict[str, PriorArgs], Boundary | None]:
    """Jointly materialize selected CO2 inputs, enabled priors and fixed mismatch.

    Nonboundary prior parameters are keyed by graph variables: ``sigma``,
    ``offset_latent``, ``sigma_site``, ``ou_site_amplitude`` or ``sigma_global``.
    The boundary choice is bound once to the prepared transport and returned
    as an eager bound boundary.
    Explicit labelled fixed mismatch is installed in the returned Dataset. Construction consumes these numerical
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
        boundary: Optional boundary choice, bound to the prepared ``H_bc`` or
            centered operands. ``None`` omits the boundary.
        offset_prior: Optional prior enabling concentration offsets.
        offset_args: Offset frequency, ``per_site`` and ``drop_first`` settings.

    Returns:
        tuple: A shallow Dataset copy with selected arrays eager, eager prior
            parameters, and the jointly materialized bound boundary (or None).
            Unselected arrays remain borrowed; the preparation and supplied
            parameter mappings are unchanged.

    Raises:
        ValueError: If model choices conflict, required prepared arrays are
            absent, a centered boundary uses a custom likelihood callback, or a
            labelled mismatch does not align exactly.
    """
    if no_model_error and (sigma_alignment is not None or sigma_prior is not None):
        raise ValueError("`no_model_error=True` cannot be combined with `sigma_alignment` or `sigma_prior`.")
    if likelihood_builder is None and likelihood_kwargs:
        raise ValueError("likelihood_kwargs require likelihood_builder.")
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
    if (boundary is not None and "H_bc_anomaly" in prepared.inv_inputs
            and likelihood_builder not in (None, add_fixed_ou_gaussian_likelihood)):
        raise ValueError("Centered runner replay supports the built-in fixed-OU likelihood, not custom callbacks.")
    names = co2_model_input_names(
        prepared,
        preserve_prepared_fixed_mismatch=(likelihood_builder is None and fixed_model_mismatch is None),
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
    if offset_prior is not None:
        priors["offset_latent"] = offset_prior
    likelihood_options = likelihood_kwargs or {}
    if likelihood_builder is add_scalar_sigma_eigen_likelihood and "sigma_prior" in likelihood_options:
        priors["sigma_global"] = likelihood_options["sigma_prior"]
    elif likelihood_builder is add_site_sigma_gaussian_likelihood and "site_amplitude_prior" in likelihood_options:
        priors["sigma_site"] = likelihood_options["site_amplitude_prior"]
    elif likelihood_builder is add_fixed_ou_gaussian_likelihood and "site_amplitude_prior" in likelihood_options:
        priors["ou_site_amplitude"] = likelihood_options["site_amplitude_prior"]
    # Boundary labels are checked before any selected numerical payload is computed.
    bound = _bind_co2_boundary(prepared.inv_inputs, boundary)
    return _materialize_co2_inputs(backend_inputs, names, priors, bound)



def build_co2_from_materialized(
    model_inputs: xr.Dataset,
    *,
    prepared_inputs: Co2PreparedInputs,
    biosphere_prior: BiospherePrior | None = None,
    priors: Mapping[str, PriorArgs],
    boundary: Boundary | None = None,
    sigma_alignment: SigmaAlignment | None = None,
    sigma_prior: PriorArgs | None = None,
    fixed_model_mismatch: float | xr.DataArray | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    no_model_error: bool = False,
    offset_prior: PriorArgs | None = None,
    offset_args: Mapping[str, Any] | None = None,
) -> RhimeModelBuildResult:
    """Construct the CO2 model from explicitly materialized scientific inputs.

    ``prepared_inputs`` supplies aggregation mode and descriptive provenance.
    ``priors`` supplies selected eager numerical parameters under the graph names
    returned by :func:`~openghg_inversions.rhime.co2.co2_runner.materialize_co2_inputs`. Prior arguments in the model
    choices select components, but their borrowed numerical values are not used.
    ``boundary`` carries the resolved scientific boundary returned by that same
    materialization; construction forwards it without rebinding periods or priors.
    This operation performs construction only; saved replay never calls it.
    Remaining model choices have the semantics documented by
    :func:`~openghg_inversions.rhime.co2.co2_runner.materialize_co2_inputs`. A configured scalar-sigma eigenbasis is
    loaded and checked against observations, error and aggregation covariance
    before its likelihood is constructed.

    Args:
        biosphere_prior: Optional explicit calibration of the retained state prior.
        model_inputs: Selected scientific arrays already materialized together.
        prepared_inputs: Preparation supplying aggregation mode and provenance.
        priors: Materialized prior parameters under their concrete graph names.
        boundary: Jointly materialized bound boundary, or None when disabled.

    Returns:
        Constructed PyMC model with component-declared scientific roles, units
        and bounded metadata. Numerical arrays remain fixed for this graph.

    Raises:
        ValueError: If selected inputs or numerical priors remain lazy, or
            scientific components reject the supplied choices.
    """
    prepared = prepared_inputs
    names = co2_model_input_names(
        prepared,
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
    model = build_co2_model(
        model_inputs["H"],
        retained_prior=retained_prior,
        biosphere_prior=biosphere_prior,
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
        boundary=boundary,
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
    boundary_outputs = boundary.outputs() if boundary is not None else None
    if boundary_outputs is not None:
        variable_roles.update(boundary_outputs.roles)
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
                "prior": biosphere_prior.data.attrs["family"] if biosphere_prior is not None else "correlated arithmetic-moment lognormal",
                "no_model_error": no_model_error,
                "use_bc": boundary is not None,
                **(boundary_outputs.metadata if boundary_outputs is not None else {"boundary_formulation": None}),
                "boundary_units": boundary_outputs.units if boundary_outputs is not None else {},
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
