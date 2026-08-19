"""Small validation helpers shared by concrete RHIME model recipes."""

from __future__ import annotations

from typing import Any, cast

import pymc as pm

from openghg_inversions.models.likelihoods import add_gaussian_observation_likelihood
from openghg_inversions.models.pollution_event import build_pollution_event_error

from .builders import (
    RhimeLikelihoodBuilder,
    RhimeLikelihoodContext,
    RhimeLikelihoodResult,
    RhimeModelBuilder,
    RhimeModelBuilderContext,
    RhimeModelBuildResult,
    validate_model_build_result,
)
from .specs import RhimeModelSpec


_LIKELIHOOD_RESULT_ATTR = "_openghg_rhime_likelihood_result"


def _build_builtin_likelihood(context: RhimeLikelihoodContext) -> RhimeLikelihoodResult:
    """Add the historical Gaussian observation model for a RHIME recipe."""
    state = build_pollution_event_error(
        context.data,
        pollution_mean=context.pollution_mean,
        pollution_event_baseline=context.pollution_event_baseline,
        sigma_alignment=context.sigma_alignment,
        sigma_prior=context.sigma_prior,
        power=context.power,
        pollution_events_from_obs=context.pollution_events_from_obs,
        no_model_error=context.no_model_error,
        retain_unused_sigma=context.retain_unused_sigma,
        aggregation_error_mode=context.aggregation_error_mode,
        output_dim=context.output_dim,
    )
    likelihood = add_gaussian_observation_likelihood(
        observed=state.observed,
        mean=context.mean,
        independent_variance=state.independent_variance,
        aggregation_error=state.aggregation_error,
        output_dim=context.output_dim,
    )
    return RhimeLikelihoodResult(
        likelihood=likelihood,
        error_scale=state.error_scale,
        variable_roles={"concentration": "y", "model_error": "epsilon"},
        supported_output_formats=("none", "inv_out", "basic", "paris", "legacy"),
        metadata={
            "family": "pollution_event_gaussian",
            "sigma_interpretation": "pollution_event_scaled",
        },
    )


def build_and_attach_rhime_likelihood(
    model: pm.Model,
    context: RhimeLikelihoodContext,
    likelihood_builder: RhimeLikelihoodBuilder | None,
) -> RhimeLikelihoodResult:
    """Build, validate, and attach a recipe's observation component."""
    result = (
        _build_builtin_likelihood(context)
        if likelihood_builder is None
        else likelihood_builder(context)
    )
    if not isinstance(result, RhimeLikelihoodResult):
        raise TypeError(
            "A RHIME likelihood builder must return `RhimeLikelihoodResult`; "
            f"got {type(result).__name__}."
        )
    missing_names = sorted(set(result.variable_roles.values()) - set(model.named_vars))
    if missing_names:
        raise ValueError(
            "RHIME likelihood roles refer to variables absent from the active PyMC model: "
            f"{missing_names!r}."
        )
    setattr(model, _LIKELIHOOD_RESULT_ATTR, result)
    return result


def get_rhime_likelihood_result(model: pm.Model) -> RhimeLikelihoodResult:
    """Return the explicit likelihood roles attached by a RHIME recipe."""
    try:
        return cast(RhimeLikelihoodResult, getattr(model, _LIKELIHOOD_RESULT_ATTR))
    except AttributeError as exc:
        raise ValueError(
            "The PyMC model has no RHIME likelihood result. Build it with a public RHIME model "
            "builder or return explicit roles from a complete `RhimeModelBuilder`."
        ) from exc


def builtin_model_build_result(
    model: pm.Model,
    *,
    model_spec: RhimeModelSpec,
    multisector: bool,
) -> RhimeModelBuildResult:
    """Describe a built-in standard or multisector graph through public roles."""
    try:
        likelihood_result = get_rhime_likelihood_result(model)
        likelihood_roles = dict(likelihood_result.variable_roles)
        supported_output_formats = likelihood_result.supported_output_formats
        likelihood_metadata = dict(likelihood_result.metadata)
    except ValueError:
        # Preserve historical test doubles and wrappers. Production built-in
        # graphs always carry their explicit likelihood result.
        likelihood_roles = {"concentration": "y", "model_error": "epsilon"}
        supported_output_formats = ("none", "inv_out", "basic", "paris", "legacy")
        likelihood_metadata = {}

    roles = {
        "observation": "mf",
        "observation_error": "mf_error",
        "minimum_error": "min_error",
        **likelihood_roles,
    }
    if multisector:
        for sector in model_spec.sectors:
            roles[f"flux_scale:{sector.name}"] = f"x_{sector.variable_suffix}"
            roles[f"flux_contribution:{sector.name}"] = f"mu_{sector.variable_suffix}"
            roles[f"emissions_sensitivity:{sector.name}"] = f"hx_{sector.variable_suffix}"
    else:
        roles.update({"flux_scale": "x", "flux_contribution": "mu", "emissions_sensitivity": "hx"})
    if model_spec.use_bc:
        roles.update({"baseline": "mu_bc", "baseline_scale": "bc", "baseline_sensitivity": "hbc"})
    if model_spec.add_offset:
        roles["offset"] = "offset"

    metadata: dict[str, Any] = {"kind": "builtin"}
    if likelihood_metadata:
        metadata["likelihood"] = likelihood_metadata
    return RhimeModelBuildResult(
        model=model,
        variable_roles=roles,
        supported_output_formats=cast(tuple[Any, ...], supported_output_formats),
        metadata=metadata,
    )


def validated_custom_model_build(
    model_builder: RhimeModelBuilder,
    *,
    context: RhimeModelBuilderContext,
) -> RhimeModelBuildResult:
    """Call and validate an advanced complete-model builder."""
    result = model_builder(context)
    if not isinstance(result, RhimeModelBuildResult):
        raise TypeError(
            "A RHIME model builder must return `RhimeModelBuildResult`; "
            f"got {type(result).__name__}."
        )
    validate_model_build_result(result, context=context)
    return result


def validate_likelihood_builder(likelihood_builder: object | None) -> None:
    """Reject non-callable likelihood builders at a public recipe boundary."""
    if likelihood_builder is not None and not callable(likelihood_builder):
        raise TypeError(
            f"`likelihood_builder` must be callable or None; got {type(likelihood_builder).__name__}."
        )
