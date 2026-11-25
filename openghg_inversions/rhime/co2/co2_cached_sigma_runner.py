"""Run the cached fixed-OU CO2 recipe from dedicated prepared inputs.

This boundary validates the named input inventory, materializes related
arrays together, restores the cached covariance basis during model building,
and coordinates the compound sampler for site amplitudes and model states.
Fixed site amplitudes use the same graph and sampler without the amplitude
step: the state quadratic is built once and never refreshed.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

import xarray as xr
from pymc.initial_point import convert_str_to_rv_dict, make_initial_point_fn  # type: ignore[import-untyped]

from openghg_inversions.correlated_state import CorrelatedPrior
from openghg_inversions.inference.cached_sigma import make_cached_sigma_compound_step
from openghg_inversions.models.boundary import Choice
from openghg_inversions.models.outer_flux import OuterFluxChoice
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import (
    aggregation_error_input_names,
    resolve_aggregation_error,
)
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.sampling import RhimeSampler

from .co2_cached_sigma_model import (
    OU_SITE_AMPLITUDE,
    OU_SITE_INDEX,
    Co2CachedSigmaModel,
    _build_co2_cached_sigma_model,
)
from .co2_model import _normalise_offset_args
from .co2_preparation import Co2PreparedInputs
from .co2_runner import (
    _bind_co2_boundary,
    _materialize_co2_inputs,
    _state_activity_from_inputs,
    annotate_co2_trace,
)

_CO2_CACHED_SIGMA_INPUT_NAMES = (
    "H",
    "alpha_prior_mean",
    "alpha_prior_covariance",
    "fixed_prior_contribution",
    "mf",
    "mf_error",
)


def co2_cached_sigma_input_names(
    prepared_inputs: Co2PreparedInputs,
) -> tuple[str, ...]:
    """Declare non-boundary arrays consumed by the named cached fixed-OU recipe.

    Boundary transport is materialized with its bound boundary instead.

    Args:
        prepared_inputs: Validated CO2 artifact to inspect.

    Returns:
        Ordered input-variable names required by the recipe.

    Raises:
        ValueError: If a selected input is missing from the artifact.
    """
    inputs = prepared_inputs.inv_inputs
    names = list(_CO2_CACHED_SIGMA_INPUT_NAMES)
    names.extend(
        aggregation_error_input_names(inputs, prepared_inputs.aggregation_error_mode)
    )
    if "state_is_active" in inputs:
        names.append("state_is_active")
        if "state_fixed_value" in inputs:
            names.append("state_fixed_value")
    missing = [name for name in names if name not in inputs]
    if missing:
        raise ValueError(
            "CO2 inputs for inferred fixed-OU amplitudes are missing required variable(s): "
            f"{missing!r}."
        )
    return tuple(names)


FIXED_AMPLITUDE_STATE_TARGET_ACCEPT = 0.8
"""State NUTS target for fixed amplitudes when none is given: PyMC's NUTS default.

This is the value the previous plain NUTS fixed-amplitude route used when no
``target_accept`` was supplied.
"""


def validate_cached_co2_sampler(sampler: RhimeSampler, *, fixed_amplitudes: bool = False) -> None:
    """Reject incompatible inference choices before preparing any arrays.

    With ``fixed_amplitudes`` the sampler has only the state NUTS step, so a
    ``sample_kwargs['target_accept']`` is accepted and applies to that step.
    """
    if sampler.nuts_sampler != "pymc":
        raise ValueError(
            "Fixed-amplitude fixed-OU sampling requires nuts_sampler='pymc'."
            if fixed_amplitudes
            else "Inferred fixed-OU amplitudes require nuts_sampler='pymc'."
        )
    sample_kwargs = dict(sampler.sample_kwargs or {})
    if "target_accept" in sample_kwargs and not fixed_amplitudes:
        raise ValueError(
            "Inferred fixed-OU amplitudes use separate NUTS tuning controls; pass "
            "separate `sigma_target_accept` and `state_target_accept` controls instead of "
            "sampler.sample_kwargs['target_accept']."
        )
    if sample_kwargs.get("step") is not None:
        raise ValueError(
            "The fixed-OU amplitude sampler constructs its required sigma-then-state "
            "CompoundStep; do not pass sample_kwargs['step']."
        )
    if fixed_amplitudes:
        return
    requested = sampler.sample_posterior_predictive
    if not isinstance(requested, bool):
        unsupported_names = set(requested) - {"y", "concentration"}
        if unsupported_names:
            raise ValueError(
                "Fixed-OU amplitude prediction supports only y or concentration; "
                f"got {sorted(unsupported_names)!r}."
            )
    unsupported = set(sampler.posterior_predictive_kwargs or {}) - {"random_seed"}
    if unsupported:
        raise ValueError(
            "Fixed-OU amplitude prediction supports only random_seed; "
            f"got {sorted(unsupported)!r}."
        )


def resolve_cached_target_accept(
    sampler: RhimeSampler,
    *,
    fixed_amplitudes: bool,
    sigma_target_accept: float | None = None,
    state_target_accept: float | None = None,
) -> tuple[float | None, float]:
    """Resolve the step targets of the cached fixed-OU sampler.

    Inferred amplitudes default to 0.8 (amplitude step) and 0.9 (state step).
    Fixed amplitudes have no amplitude step, so ``sigma_target_accept`` is
    rejected; the state step uses ``state_target_accept``, else the sampler's
    ``sample_kwargs['target_accept']`` or ``sample_kwargs['nuts']['target_accept']``
    (the controls of the previous plain NUTS route), else
    :data:`FIXED_AMPLITUDE_STATE_TARGET_ACCEPT`.

    Returns:
        The amplitude-step target (``None`` for fixed amplitudes) and the
        state-step target.

    Raises:
        ValueError: If fixed amplitudes receive ``sigma_target_accept`` or more
            than one state-step target.
        TypeError: If fixed amplitudes receive a non-mapping NUTS option.
    """
    if not fixed_amplitudes:
        return (
            0.8 if sigma_target_accept is None else sigma_target_accept,
            0.9 if state_target_accept is None else state_target_accept,
        )
    if sigma_target_accept is not None:
        raise ValueError(
            "sigma_target_accept tunes the amplitude step of inferred fixed-OU amplitudes; "
            "fixed amplitudes have no amplitude step."
        )
    sample_kwargs = sampler.sample_kwargs or {}
    nuts_kwargs = sample_kwargs.get("nuts", {})
    if not isinstance(nuts_kwargs, Mapping):
        raise TypeError("Fixed-OU sample_kwargs['nuts'] must be a mapping.")
    targets = [section["target_accept"] for section in (sample_kwargs, nuts_kwargs) if "target_accept" in section]
    if state_target_accept is not None:
        targets.append(state_target_accept)
    if len(targets) > 1:
        raise ValueError(
            "Pass only one target for the fixed-OU state step: state_target_accept, "
            "sample_kwargs['target_accept'], or sample_kwargs['nuts']['target_accept']."
        )
    return None, float(targets[0]) if targets else FIXED_AMPLITUDE_STATE_TARGET_ACCEPT


def _sampler_for_cached_graph(
    sampler: RhimeSampler,
    *,
    cached_model: Co2CachedSigmaModel,
    sigma_target_accept: float | None = None,
    state_target_accept: float | None = None,
) -> RhimeSampler:
    """Return a sampler copy whose ``step`` is the matched cached CompoundStep.

    Inferred amplitudes get the sigma-then-state sweep. Fixed amplitudes get
    the state step alone, with targets resolved by
    :func:`resolve_cached_target_accept`. Fixed amplitudes forward the
    caller's nested ``sample_kwargs['nuts']`` options onto that step.
    """
    fixed = bool(getattr(cached_model, "fixed_amplitudes", False))
    validate_cached_co2_sampler(sampler, fixed_amplitudes=fixed)
    sigma_target_accept, state_target_accept = resolve_cached_target_accept(
        sampler,
        fixed_amplitudes=fixed,
        sigma_target_accept=sigma_target_accept,
        state_target_accept=state_target_accept,
    )
    sample_kwargs = dict(sampler.sample_kwargs or {})
    sample_kwargs.pop("target_accept", None)
    nuts_kwargs = dict(sample_kwargs.pop("nuts", {})) if fixed else {}
    nuts_kwargs.pop("target_accept", None)
    model = cached_model.model
    defaults = convert_str_to_rv_dict(model, cached_model.initial_values)

    def merge_initial_values(overrides: Mapping[str, Any] | None) -> dict[Any, Any]:
        # Normalize separately so transformed caller keys override physical
        # recipe defaults for the same RV, regardless of mapping order.
        return {**defaults, **convert_str_to_rv_dict(model, overrides or {})}

    overrides = sample_kwargs.get("initvals")
    if isinstance(overrides, Sequence) and not isinstance(overrides, (str, bytes)):
        sample_kwargs["initvals"] = [merge_initial_values(item) for item in overrides]
    else:
        sample_kwargs["initvals"] = merge_initial_values(cast(Mapping[str, Any] | None, overrides))
    baseline = make_initial_point_fn(
        model=model, overrides=defaults, return_transformed=True,
    )(seed=0)
    with model:
        sample_kwargs["step"] = make_cached_sigma_compound_step(
            model=model,
            sigma=None if fixed else cached_model.amplitude,
            states=cached_model.states,
            modelled_mean=cached_model.modelled_mean,
            target=cached_model.target,
            shared_cache=cached_model.shared_cache,
            initial_cache=cached_model.initial_cache,
            cached_state_logp=cached_model.cached_state_logp,
            initial_point=baseline,
            sigma_target_accept=0.8 if sigma_target_accept is None else sigma_target_accept,
            state_target_accept=state_target_accept,
            state_nuts_kwargs=nuts_kwargs,
        )
    sample_kwargs.setdefault("mp_ctx", "spawn")
    configured = copy.copy(sampler)
    configured.sample_kwargs = sample_kwargs
    return configured


def _annotate_cached_co2_trace(
    trace: xr.DataTree,
    built: RhimeModelBuildResult,
    *,
    concentration_units: str | None,
) -> xr.DataTree:
    """Add output semantics that belong only to the cached fixed-OU recipe."""
    trace = annotate_co2_trace(
        trace,
        built,
        concentration_units=concentration_units,
    )
    for group in trace.children.values():
        group.attrs["rhime_recipe"] = built.output_contract.metadata["recipe"]

    if "log_likelihood" in trace.children and "y" in trace["log_likelihood"]:
        likelihood = trace["log_likelihood"]["y"]
        likelihood.attrs["rhime_scientific_roles"] = json.dumps(["joint_log_likelihood"])
        likelihood.attrs.pop("units", None)
        likelihood.attrs.update(
            rhime_likelihood_scope="joint_observation_vector",
            rhime_normalized_log_likelihood=1,
        )
    for name in ("prior_predictive", "posterior_predictive"):
        if name in trace.children and "y" in trace[name]:
            predictive = trace[name]["y"]
            predictive.attrs.update(
                rhime_scientific_roles=json.dumps(["concentration"]),
                rhime_predictive_scope="joint_observation_vector",
            )
            if concentration_units is not None:
                predictive.attrs["units"] = concentration_units
    if "observed_data" in trace.children and "y" in trace["observed_data"]:
        observed = trace["observed_data"]["y"]
        observed.attrs["rhime_scientific_roles"] = json.dumps(["observation"])
        if concentration_units is not None:
            observed.attrs["units"] = concentration_units
    for group in trace.children.values():
        if concentration_units is not None and OU_SITE_AMPLITUDE in group:
            group[OU_SITE_AMPLITUDE].attrs["units"] = concentration_units
    if "constant_data" in trace.children:
        constant_data = trace["constant_data"]
        if "ou_tau_hours" in constant_data:
            constant_data["ou_tau_hours"].attrs["units"] = "hours"
        for name in ("Y", "error"):
            if concentration_units is not None and name in constant_data:
                constant_data[name].attrs["units"] = concentration_units
    return trace


def build_rhime_co2_cached_sigma(
    *,
    prepared_inputs: Co2PreparedInputs,
    tau_hours: float | Mapping[str, float],
    site_amplitude_prior: PriorArgs | None = None,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
    fixed_site_amplitudes: float | Mapping[str, float] | None = None,
    boundary: Choice | None = None,
    outer_flux: OuterFluxChoice | None = None,
    offset_prior: PriorArgs | None = None,
    offset_args: Mapping[str, Any] | None = None,
) -> Co2CachedSigmaModel:
    """Materialize and build the matched cached fixed-OU graph without sampling.

    Args:
        prepared_inputs: Validated CO2 inputs with observation-aligned site and
            time coordinates and the selected aggregation-error representation.
        tau_hours: Fixed OU decorrelation time in hours, as a scalar or a
            mapping covering all observed sites.
        site_amplitude_prior: Independent positive site-amplitude prior in
            observation concentration units. Exactly one of this or
            ``fixed_site_amplitudes`` is required.
        initial_site_amplitudes: Positive scalar or site mapping in the same
            units; defaults to the prior support point. Only valid with
            ``site_amplitude_prior``.
        fixed_site_amplitudes: Fixed non-negative scalar or site mapping in the
            same units. The graph then holds them as ``ou_site_amplitude``
            model data and the matched sampler has only the state step.
        boundary: Optional boundary choice, bound once to the prepared
            ``H_bc`` or centered operands. ``None`` omits the boundary.
        outer_flux: Optional outer-flux choice, which the model builder
            resolves on the retained flux prior.
        offset_prior: Optional prior for an offset component. When omitted, no
            offset is added.
        offset_args: Optional offset settings: ``offset_freq``, ``drop_first``,
            and ``per_site``.

    Returns:
        Concrete graph and numerical target required by the matched sampler.

    Raises:
        ValueError: If component options contradict one another, prepared
            arrays are invalid, or fixed OU mappings or parameters are invalid.
    """
    if offset_prior is None and offset_args:
        raise ValueError("offset_args require offset_prior.")
    if (site_amplitude_prior is None) == (fixed_site_amplitudes is None):
        raise ValueError("Pass exactly one of `site_amplitude_prior` or `fixed_site_amplitudes`.")
    offset_freq, offset_drop_first, offset_per_site = _normalise_offset_args(offset_args)
    prepared = prepared_inputs
    names = co2_cached_sigma_input_names(prepared)
    model_inputs, eager_priors, bound = _materialize_co2_inputs(
        prepared.rhime_inputs, names,
        {
            **({"amplitude": site_amplitude_prior} if site_amplitude_prior is not None else {}),
            **({"offset": offset_prior} if offset_prior is not None else {}),
        },
        _bind_co2_boundary(prepared.inv_inputs, boundary),
    )
    site_amplitude_prior = eager_priors.get("amplitude")
    offset_prior = eager_priors.get("offset")
    aggregation_error = resolve_aggregation_error(
        model_inputs,
        prepared.aggregation_error_mode,
    )
    prior_covariance = model_inputs["alpha_prior_covariance"]
    retained_prior = CorrelatedPrior(
        model_inputs["alpha_prior_mean"],
        prior_covariance,
        covariance_dim=str(prior_covariance.dims[-1]),
    )
    cached_model = _build_co2_cached_sigma_model(
        model_inputs["H"],
        retained_prior=retained_prior,
        fixed_prior_contribution=model_inputs["fixed_prior_contribution"],
        observations=model_inputs["mf"],
        observation_error=model_inputs["mf_error"],
        aggregation_error=aggregation_error,
        tau_hours=tau_hours,
        site_amplitude_prior=site_amplitude_prior,
        initial_site_amplitudes=initial_site_amplitudes,
        fixed_site_amplitudes=fixed_site_amplitudes,
        state_activity=cast(StateActivity | None, _state_activity_from_inputs(model_inputs)),
        boundary=bound,
        outer_flux=outer_flux,
        offset_prior=offset_prior,
        offset_freq=offset_freq,
        offset_drop_first=offset_drop_first,
        offset_per_site=offset_per_site,
    )
    return cached_model


def co2_cached_sigma_build_result(
    cached_model: Co2CachedSigmaModel,
    prepared_inputs: Co2PreparedInputs,
) -> RhimeModelBuildResult:
    """Describe a built cached graph for sampling and stored output metadata.

    Args:
        cached_model: Graph returned by :func:`build_rhime_co2_cached_sigma`.
        prepared_inputs: Prepared inputs used to build the graph, providing
            basis provenance.

    Returns:
        Model build result with scientific roles and serializable provenance.
    """
    metadata = {
        "recipe": "co2",
        "kind": "builtin",
        "prior": "correlated arithmetic-moment lognormal",
        "mismatch_component": "fixed_within_site_ou",
        "basis_artifact_source": getattr(
            prepared_inputs,
            "basis_artifact_source",
            "unknown",
        ),
        "basis_artifact_path": getattr(prepared_inputs, "basis_artifact_path", None),
    }
    variable_roles = {
        "observation": "Y",
        "observation_error": "error",
        "concentration": "y",
        "model_error": "epsilon",
        "model_mean": "modelled_concentration",
        "pollution_concentration": "co2_flux_contribution",
        "flux_scale": "flux_scaling",
        "fixed_ou_site_amplitude": OU_SITE_AMPLITUDE,
        "observation_to_fixed_ou_site_index": OU_SITE_INDEX,
        "fixed_ou_timescale": "ou_tau_hours",
        "emissions_sensitivity": "co2_sensitivity",
        "coherent_prior_contribution": "fixed_prior_contribution",
    }
    if cached_model.boundary_outputs is not None:
        outputs = cached_model.boundary_outputs
        variable_roles.update(outputs.roles)
        metadata.update(outputs.metadata, boundary_units=outputs.units)
    else:
        metadata["boundary_units"] = {}
    if cached_model.outer_flux_outputs is not None:
        variable_roles.update(cached_model.outer_flux_outputs.roles)
        metadata.update(cached_model.outer_flux_outputs.metadata)
    if metadata.get("boundary_representation") == "centered-v1":
        variable_roles["observation"] = "y"
    if "offset" in cached_model.model.named_vars:
        variable_roles["offset_concentration"] = "offset"
    for role, name in {
        "active_flux_scale": "flux_scaling_active",
        "offset_coefficient": "offset_latent",
    }.items():
        if name in cached_model.model.named_vars:
            variable_roles[role] = name
    built = RhimeModelBuildResult(
        model=cached_model.model,
        output_contract=OutputContract(
            variable_roles=variable_roles,
            metadata=metadata,
        ),
    )
    return built


__all__ = [
    "FIXED_AMPLITUDE_STATE_TARGET_ACCEPT",
    "build_rhime_co2_cached_sigma",
    "co2_cached_sigma_build_result",
    "co2_cached_sigma_input_names",
    "resolve_cached_target_accept",
]
