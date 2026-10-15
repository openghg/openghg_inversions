"""Public whole-model build contracts and validation for RHIME customizations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import pymc as pm
from pytensor.tensor.variable import TensorVariable
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.models.coords import get_coord_registry
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime.specs import RhimeRunSpec


RhimeLikelihoodBuilder = Callable[..., TensorVariable]


@dataclass(frozen=True)
class RhimeModelBuilderContext:
    """Advanced compatibility input supplied only to a complete model builder.

    Ordinary in-tree recipes and components use explicit named scientific
    inputs. This context remains solely for user-owned complete models invoked
    through the multisector ``run_rhime_from_prepared_inputs`` route; those builders own validation
    and materialization of any lazy arrays they consume.

    Args:
        prepared_inputs: Validated canonical inputs, retained basis functions,
            and preparation metadata.
        run_spec: Model, output, and run settings for this execution. The
            callable is deliberately kept outside this serializable spec.
        multisector: Whether the validated prepared layout and model spec are
            sector resolved.
    """

    prepared_inputs: RhimePreparedInputs
    run_spec: RhimeRunSpec
    multisector: bool


@dataclass(frozen=True)
class RhimeModelBuildResult:
    """Concrete graph and its validated, serializable scientific output contract.

    Args:
        model: Concrete PyMC model for sampling.
        output_contract: Semantic variable roles, supported output formats and
            construction metadata. Custom models default to sampling-only
            capability when constructing an ordinary ``OutputContract``.
    """

    model: pm.Model
    output_contract: OutputContract


class RhimeModelBuilder(Protocol):
    """Advanced callable contract for a complete user-owned model factory."""

    def __call__(self, context: RhimeModelBuilderContext, /) -> RhimeModelBuildResult:
        """Build a concrete model from validated prepared inputs and settings."""
        ...


def validate_model_build_result(
    result: RhimeModelBuildResult,
    *,
    context: RhimeModelBuilderContext,
) -> None:
    """Validate a custom build result before sampling or postprocessing.

    Args:
        result: Complete model build result to validate.
        context: Prepared inputs and run settings for the active model build.
    Raises:
        ValueError: If the requested output is unsupported or a declared
            variable role refers to an absent variable.
    """
    if get_coord_registry(result.model) is None:
        raise ValueError(
            "A custom RHIME model must carry a `CoordRegistry`; construct it "
            "with `registered_model()`."
        )

    output_format = context.run_spec.output.output_format
    result.output_contract.validate_requested_output(output_format)

    available_names = set(result.model.named_vars) | set(context.prepared_inputs.inv_inputs.variables)
    missing = {role: name for role, name in result.output_contract.variable_roles.items() if name not in available_names}
    if missing:
        details = ", ".join(f"{role}={name!r}" for role, name in sorted(missing.items()))
        raise ValueError(
            "Custom RHIME model variable roles refer to names absent from both the PyMC model and "
            f"prepared inversion inputs: {details}."
        )


def callable_metadata(builder: Callable[..., Any]) -> dict[str, str]:
    """Return stable, serializable direct-Python callable identity metadata.

    Args:
        builder: Callable whose import module and qualified name identify the
            runtime customization.

    Returns:
        JSON-serializable module and qualified-name fields.
    """
    return {
        "module": getattr(builder, "__module__", type(builder).__module__),
        "qualname": getattr(builder, "__qualname__", type(builder).__qualname__),
    }
