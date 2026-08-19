"""Small validation helpers shared by concrete RHIME model recipes."""

from __future__ import annotations

from typing import Any, cast

import pymc as pm

from .builders import (
    RhimeModelBuilder,
    RhimeModelBuilderContext,
    RhimeModelBuildResult,
    validate_model_build_result,
)
from .likelihood import get_rhime_likelihood_result
from .specs import RhimeModelSpec


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
