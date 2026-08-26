"""Small validation helpers shared by concrete RHIME model recipes."""

from __future__ import annotations

from collections.abc import Collection

import pymc as pm

from .builders import (
    RhimeModelBuilder,
    RhimeModelBuilderContext,
    RhimeModelBuildResult,
)
from .specs import RhimeModelSpec


def builtin_model_build_result(
    model: pm.Model,
    *,
    model_spec: RhimeModelSpec,
    multisector: bool,
    input_names: Collection[str],
    preserve_legacy_baseline: bool = False,
) -> RhimeModelBuildResult:
    """Describe a built-in graph through its whole-model output roles.

    Args:
        model: Concrete built-in PyMC graph.
        model_spec: Scientific options used to build the graph.
        multisector: Whether sector-specific flux roles are required.
        input_names: Names available from retained canonical inputs.
        preserve_legacy_baseline: Whether the compatibility graph exposes the
            historical boundary-only baseline role.

    Returns:
        Concrete graph plus semantic roles and supported output formats.
    """
    roles = {
        "observation": "mf",
        "observation_error": "mf_error",
        "minimum_error": "min_error",
        "concentration": "y",
        "model_error": "epsilon",
    }
    if "mf_repeatability" in input_names:
        roles["observation_repeatability"] = "mf_repeatability"
    if "mf_variability" in input_names:
        roles["observation_variability"] = "mf_variability"
    if multisector:
        for sector in model_spec.sectors:
            roles[f"flux_scale:{sector.name}"] = f"x_{sector.variable_suffix}"
            roles[f"flux_contribution:{sector.name}"] = f"mu_{sector.variable_suffix}"
            roles[f"emissions_sensitivity:{sector.name}"] = f"hx_{sector.variable_suffix}"
    else:
        roles.update({"flux_scale": "x", "flux_contribution": "mu", "emissions_sensitivity": "hx"})
    if model_spec.use_bc:
        roles.update(
            {
                "boundary": "mu_bc",
                "baseline_scale": "bc",
                "baseline_sensitivity": "hbc",
            }
        )
    if model_spec.add_offset:
        roles["offset"] = "offset"
    if preserve_legacy_baseline and model_spec.use_bc:
        roles["baseline"] = "mu_bc"
    elif model_spec.use_bc and not model_spec.add_offset:
        roles["baseline"] = "mu_bc"
    elif model_spec.add_offset and not model_spec.use_bc:
        roles["baseline"] = "offset"

    supported_output_formats = (
        ("none", "inv_out", "paris")
        if multisector
        else ("none", "inv_out", "basic", "paris", "legacy")
    )
    return RhimeModelBuildResult(
        model=model,
        variable_roles=roles,
        supported_output_formats=supported_output_formats,
        metadata={"kind": "builtin"},
    )


def validated_custom_model_build(
    model_builder: RhimeModelBuilder,
    *,
    context: RhimeModelBuilderContext,
) -> RhimeModelBuildResult:
    """Call an advanced complete-model builder and validate its return type.

    Args:
        model_builder: User-owned complete-model callable.
        context: Validated prepared inputs and run settings.

    Returns:
        Concrete custom model build result.

    Raises:
        TypeError: If the callable returns the wrong result type.
    """
    result = model_builder(context)
    if not isinstance(result, RhimeModelBuildResult):
        raise TypeError(
            "A RHIME model builder must return `RhimeModelBuildResult`; "
            f"got {type(result).__name__}."
        )
    return result
