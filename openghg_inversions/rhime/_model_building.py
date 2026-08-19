"""Small validation helpers shared by concrete RHIME model recipes."""

from __future__ import annotations

from collections.abc import Collection

import pymc as pm
from pytensor.tensor.variable import TensorVariable

from .builders import (
    RhimeModelBuilder,
    RhimeModelBuilderContext,
    RhimeModelBuildResult,
)
from .specs import RhimeModelSpec


def validate_built_rhime_likelihood(
    model: pm.Model,
    likelihood: object,
) -> TensorVariable:
    """Validate the canonical result of an ordinary likelihood component.

    Args:
        model: Active model after likelihood construction.
        likelihood: Value returned by the likelihood callable.

    Returns:
        Validated observed concentration variable named ``y``.

    Raises:
        TypeError: If the callable did not return a PyTensor variable.
        ValueError: If canonical ``y`` or ``epsilon`` variables are absent.
    """
    if not isinstance(likelihood, TensorVariable):
        raise TypeError(
            "A RHIME likelihood builder must return a PyTensor variable; "
            f"got {type(likelihood).__name__}."
        )
    if likelihood.name != "y":
        raise ValueError(
            "A RHIME likelihood builder must name its observed concentration variable `y`; "
            f"got {likelihood.name!r}."
        )
    missing_names = sorted({"y", "epsilon"} - set(model.named_vars))
    if missing_names:
        raise ValueError(
            "A RHIME likelihood builder did not create the canonical variables required by "
            "sampling and outputs: "
            f"{missing_names!r}."
        )
    return likelihood


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
    elif model_spec.use_bc and model_spec.add_offset:
        roles["baseline"] = "mu_baseline"
    elif model_spec.use_bc:
        roles["baseline"] = "mu_bc"
    elif model_spec.add_offset:
        roles["baseline"] = "offset"

    return RhimeModelBuildResult(
        model=model,
        variable_roles=roles,
        supported_output_formats=("none", "inv_out", "basic", "paris", "legacy"),
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


def validate_likelihood_builder(likelihood_builder: object | None) -> None:
    """Validate an optional ordinary likelihood extension point.

    Args:
        likelihood_builder: Candidate likelihood callable or ``None``.

    Raises:
        TypeError: If a non-callable value is supplied.
    """
    if likelihood_builder is not None and not callable(likelihood_builder):
        raise TypeError(
            f"`likelihood_builder` must be callable or None; got {type(likelihood_builder).__name__}."
        )
