"""Run the cached fixed-OU CO2 recipe from dedicated prepared inputs.

This boundary validates the named input inventory, materializes related
arrays together, restores the cached covariance basis during model building,
and coordinates the compound sampler for site amplitudes and model states.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
from typing import Any, cast

import pymc as pm
from pymc.initial_point import convert_str_to_rv_dict, make_initial_point_fn  # type: ignore[import-untyped]
import xarray as xr

from openghg_inversions.postprocessing.contracts import OutputContract

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import (
    aggregation_error_input_names,
    resolve_aggregation_error,
)
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.inference.cached_sigma import make_cached_sigma_compound_step
from openghg_inversions.rhime.materialization import materialize_pymc_inputs
from openghg_inversions.rhime.sampling import RhimeSampler

from .co2_cached_sigma_model import (
    OU_SITE_AMPLITUDE,
    OU_SITE_INDEX,
    Co2CachedSigmaModel,
    build_co2_cached_sigma_model,
)
from .co2_model import _normalise_offset_args
from .co2_preparation import Co2PreparedInputs
from .co2_runner import (
    annotate_co2_trace,
    _state_activity_from_inputs,
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
    *,
    use_bc: bool = False,
) -> tuple[str, ...]:
    """Declare arrays consumed by the named cached fixed-OU recipe.

    Args:
        prepared_inputs: Validated CO2 artifact to inspect.
        use_bc: Include the boundary-condition sensitivity ``H_bc``.

    Returns:
        Ordered input-variable names required by the recipe.

    Raises:
        ValueError: If a selected input is missing from the artifact.
    """
    inputs = prepared_inputs.inv_inputs
    names = list(_CO2_CACHED_SIGMA_INPUT_NAMES)
    if use_bc:
        names.append("H_bc")
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
            "Cached-sigma CO2 prepared inputs are missing required variable(s): "
            f"{missing!r}."
        )
    return tuple(names)


def validate_cached_co2_sampler(sampler: RhimeSampler) -> None:
    """Reject incompatible inference choices before preparing any arrays."""
    if sampler.nuts_sampler != "pymc":
        raise ValueError("The cached-sigma CO2 recipe requires nuts_sampler='pymc'.")
    sample_kwargs = dict(sampler.sample_kwargs or {})
    if "target_accept" in sample_kwargs:
        raise ValueError(
            "The cached-sigma CO2 recipe owns two NUTS tuning controls; pass "
            "separate `sigma_target_accept` and `state_target_accept` controls instead of "
            "sampler.sample_kwargs['target_accept']."
        )
    if sample_kwargs.get("step") is not None:
        raise ValueError(
            "The cached-sigma CO2 recipe constructs its required sigma-then-state "
            "CompoundStep; do not pass sample_kwargs['step']."
        )
    requested = sampler.sample_posterior_predictive
    if not isinstance(requested, bool):
        unsupported_names = set(requested) - {"y", "concentration"}
        if unsupported_names:
            raise ValueError(
                "Cached-sigma prediction supports only y or concentration; "
                f"got {sorted(unsupported_names)!r}."
            )
    unsupported = set(sampler.posterior_predictive_kwargs or {}) - {"random_seed"}
    if unsupported:
        raise ValueError(
            "Cached-sigma prediction supports only random_seed; "
            f"got {sorted(unsupported)!r}."
        )


def _sampler_for_cached_graph(
    sampler: RhimeSampler,
    *,
    cached_model: Co2CachedSigmaModel,
    sigma_target_accept: float,
    state_target_accept: float,
) -> RhimeSampler:
    validate_cached_co2_sampler(sampler)
    sample_kwargs = dict(sampler.sample_kwargs or {})
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
            sigma=cached_model.amplitude,
            states=cached_model.states,
            modelled_mean=cached_model.modelled_mean,
            target=cached_model.target,
            shared_cache=cached_model.shared_cache,
            initial_cache=cached_model.initial_cache,
            cached_state_logp=cached_model.cached_state_logp,
            initial_point=baseline,
            sigma_target_accept=sigma_target_accept,
            state_target_accept=state_target_accept,
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
    site_amplitude_prior: PriorArgs,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
    use_bc: bool = False,
    bc_prior: PriorArgs | None = None,
    bc_state_activity: StateActivity | None = None,
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
            observation concentration units.
        initial_site_amplitudes: Positive scalar or site mapping in the same
            units; defaults to the prior support point.
        use_bc: Include prepared boundary sensitivity ``H_bc``.
        bc_prior: Optional boundary-scaling prior.
        bc_state_activity: Optional active/fixed boundary-state policy.
        offset_prior: Optional offset prior; omit to disable the component.
        offset_args: Offset settings ``offset_freq``, ``drop_first``, and
            ``per_site``.

    Returns:
        Concrete graph and numerical target required by the matched sampler.

    Raises:
        ValueError: If component options contradict one another, prepared
            arrays are invalid, or fixed OU mappings or parameters are invalid.
    """
    if not use_bc and (bc_prior is not None or bc_state_activity is not None):
        raise ValueError("bc_prior and bc_state_activity require use_bc=True.")
    if offset_prior is None and offset_args:
        raise ValueError("offset_args require offset_prior.")
    offset_freq, offset_drop_first, offset_per_site = _normalise_offset_args(offset_args)
    prepared = prepared_inputs
    names = co2_cached_sigma_input_names(
        prepared,
        use_bc=use_bc,
    )
    model_inputs = materialize_pymc_inputs(prepared.rhime_inputs, variable_names=names)
    aggregation_error = resolve_aggregation_error(
        model_inputs,
        prepared.aggregation_error_mode,
    )
    prior_covariance = model_inputs["alpha_prior_covariance"]
    retained_prior = CorrelatedLognormalPrior(
        model_inputs["alpha_prior_mean"],
        prior_covariance,
        covariance_dim=str(prior_covariance.dims[-1]),
    )
    cached_model = build_co2_cached_sigma_model(
        model_inputs["H"],
        retained_prior=retained_prior,
        fixed_prior_contribution=model_inputs["fixed_prior_contribution"],
        observations=model_inputs["mf"],
        observation_error=model_inputs["mf_error"],
        aggregation_error=aggregation_error,
        tau_hours=tau_hours,
        site_amplitude_prior=site_amplitude_prior,
        initial_site_amplitudes=initial_site_amplitudes,
        state_activity=cast(StateActivity | None, _state_activity_from_inputs(model_inputs)),
        boundary_sensitivity=model_inputs.get("H_bc") if use_bc else None,
        bc_prior=bc_prior,
        bc_state_activity=bc_state_activity,
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
    if "hbc" in cached_model.model.named_vars:
        variable_roles.update(
            {
                "boundary_concentration": "mu_bc",
                "boundary_scale": "bc",
                "boundary_sensitivity": "hbc",
            }
        )
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


def sample_co2_cached_prior_predictive(
    cached_model: Co2CachedSigmaModel,
    prepared_inputs: Co2PreparedInputs,
    *,
    draws: int = 500,
    random_seed: int | None = None,
) -> xr.DataTree:
    """Draw cached CO2 priors and exact correlated observation replicates.

    Native PyMC prediction uses each draw's model mean and site amplitudes,
    retaining the full covariance independently of the mutable sampler cache.

    Args:
        cached_model: Graph built from the supplied prepared inputs.
        prepared_inputs: Prepared observations and basis provenance.
        draws: Number of prior draws.
        random_seed: Optional reproducible seed for priors and joint replicates.

    Returns:
        Annotated prior, prior-predictive, observed-data, and constant groups.
    """
    with cached_model.model:
        trace = pm.sample_prior_predictive(draws=draws, random_seed=random_seed)
    registry = get_coord_registry(cached_model.model)
    if registry is not None:
        trace = restore_inferencedata_coords(trace, registry)
    return _annotate_cached_co2_trace(
        trace,
        co2_cached_sigma_build_result(cached_model, prepared_inputs),
        concentration_units=prepared_inputs.inv_inputs["mf"].attrs.get("units"),
    )


__all__ = [
    "build_rhime_co2_cached_sigma",
    "co2_cached_sigma_build_result",
    "co2_cached_sigma_input_names",
    "sample_co2_cached_prior_predictive",
]
