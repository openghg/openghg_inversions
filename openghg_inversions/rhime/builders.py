"""Public extension contracts for complete RHIME model builders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from typing import Any, Protocol

import pymc as pm

from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.rhime.specs import OutputFormat, RhimeRunSpec


_OUTPUT_FORMATS: frozenset[OutputFormat] = frozenset(
    {"none", "inv_out", "basic", "paris", "legacy"}
)


@dataclass(frozen=True)
class RhimeModelBuilderContext:
    """Labelled inputs supplied to a complete model builder.

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
    """Concrete model and serializable metadata returned by a model builder.

    Custom builders default to supporting sampling-only runs
    (``output_format="none"``). A builder must explicitly declare additional
    formats after ensuring that its role manifest and trace satisfy those
    postprocessing contracts.

    Args:
        model: Concrete PyMC model for :class:`RhimeSampler`.
        variable_roles: Semantic role to concrete input/model variable name.
            Roles such as ``concentration``, ``model_error``, ``flux_scale``,
            and ``baseline`` let sampling and outputs avoid name inference.
            Components that do not exist, such as model error in a fixed-error
            model, should be omitted.
        supported_output_formats: Output formats the builder declares safe.
            ``"none"`` always means sampling without RHIME postprocessing.
        metadata: Additional JSON-serializable builder/provenance metadata.
    """

    model: pm.Model
    variable_roles: Mapping[str, str]
    supported_output_formats: tuple[OutputFormat, ...] = ("none",)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy and validate the serializable portion of the result."""
        if not isinstance(self.model, pm.Model):
            raise TypeError(
                "`RhimeModelBuildResult.model` must be a `pymc.Model`; "
                f"got {type(self.model).__name__}."
            )

        roles = {str(role): str(name) for role, name in self.variable_roles.items()}
        invalid_roles = [role for role, name in roles.items() if not role or not name]
        if invalid_roles:
            raise ValueError(
                "`RhimeModelBuildResult.variable_roles` requires non-empty role and variable names; "
                f"invalid roles: {invalid_roles!r}."
            )
        if not roles:
            raise ValueError("`RhimeModelBuildResult.variable_roles` must not be empty.")

        output_formats = tuple(dict.fromkeys(self.supported_output_formats))
        invalid_formats = sorted(set(output_formats) - _OUTPUT_FORMATS)
        if invalid_formats:
            raise ValueError(
                "`RhimeModelBuildResult.supported_output_formats` contains unsupported values: "
                f"{invalid_formats!r}."
            )
        if "none" not in output_formats:
            raise ValueError(
                "`RhimeModelBuildResult.supported_output_formats` must include 'none' so the "
                "model remains usable without RHIME postprocessing."
            )

        metadata = dict(self.metadata)
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("`RhimeModelBuildResult.metadata` must be JSON serializable.") from exc

        object.__setattr__(self, "variable_roles", roles)
        object.__setattr__(self, "supported_output_formats", output_formats)
        object.__setattr__(self, "metadata", metadata)


class RhimeModelBuilder(Protocol):
    """Callable contract for a complete user-owned RHIME model factory."""

    def __call__(self, context: RhimeModelBuilderContext, /) -> RhimeModelBuildResult:
        """Build a concrete model from validated prepared inputs and settings."""
        ...


def validate_model_build_result(
    result: RhimeModelBuildResult,
    *,
    context: RhimeModelBuilderContext,
) -> None:
    """Validate a custom build result before sampling or postprocessing."""
    output_format = context.run_spec.output.output_format
    if output_format not in result.supported_output_formats:
        raise ValueError(
            f"Custom RHIME model builder does not declare output_format={output_format!r} compatible. "
            f"Declared formats: {list(result.supported_output_formats)!r}. Use output_format='none' or "
            "return a build result that explicitly supports the requested RHIME output contract."
        )

    available_names = set(result.model.named_vars) | set(context.prepared_inputs.inv_inputs.variables)
    missing = {
        role: name for role, name in result.variable_roles.items() if name not in available_names
    }
    if missing:
        details = ", ".join(f"{role}={name!r}" for role, name in sorted(missing.items()))
        raise ValueError(
            "Custom RHIME model variable roles refer to names absent from both the PyMC model and "
            f"prepared inversion inputs: {details}."
        )

    if "concentration" not in result.variable_roles:
        raise ValueError(
            "Custom RHIME model variable roles must declare `concentration` so predictive sampling "
            "and concentration outputs do not infer the observed-variable name."
        )


def callable_metadata(builder: RhimeModelBuilder) -> dict[str, str]:
    """Return stable, serializable direct-Python callable identity metadata."""
    return {
        "module": getattr(builder, "__module__", type(builder).__module__),
        "qualname": getattr(builder, "__qualname__", type(builder).__qualname__),
    }
