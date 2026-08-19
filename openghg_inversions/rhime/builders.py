"""Public model-build contracts and validation for RHIME customizations.

The validation applies both to complete-model builders and to built-in model
results that wrap a custom likelihood builder.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from typing import Any, Protocol

import pymc as pm
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.observation_error import AggregationError
from openghg_inversions.rhime.specs import OutputFormat, RhimeRunSpec
from openghg_inversions.sigma import SigmaAlignment


_OUTPUT_FORMATS: frozenset[OutputFormat] = frozenset({"none", "inv_out", "basic", "paris", "legacy"})


class RhimeLikelihoodBuilder(Protocol):
    """Explicit callable contract for a complete RHIME likelihood component."""

    def __call__(
        self,
        *,
        observations: xr.DataArray,
        observation_error: xr.DataArray,
        minimum_error: xr.DataArray,
        aggregation_error: AggregationError,
        mean: TensorVariable,
        pollution_mean: TensorVariable,
        pollution_event_baseline: TensorVariable | None,
        sigma_alignment: SigmaAlignment,
        sigma_prior: Mapping[str, Any],
        power: Mapping[str, Any] | float,
        pollution_events_from_obs: bool,
        no_model_error: bool,
        output_dim: str,
    ) -> TensorVariable:
        """Add canonical ``y`` and ``epsilon`` variables to the active model."""
        ...


@dataclass(frozen=True)
class RhimeModelBuilderContext:
    """Advanced compatibility input supplied only to a complete model builder.

    Ordinary in-tree recipes and components use explicit named scientific
    inputs. This context remains solely for user-owned complete models invoked
    through ``run_rhime_from_prepared_inputs``; those builders own validation
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
                f"`RhimeModelBuildResult.model` must be a `pymc.Model`; got {type(self.model).__name__}."
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
        ValueError: If the requested output is unsupported or declared
            variable roles are incomplete or refer to absent variables.
    """
    output_format = context.run_spec.output.output_format
    if output_format not in result.supported_output_formats:
        raise ValueError(
            f"Custom RHIME model builder does not declare output_format={output_format!r} compatible. "
            f"Declared formats: {list(result.supported_output_formats)!r}. Use output_format='none' or "
            "return a build result that explicitly supports the requested RHIME output contract."
        )

    available_names = set(result.model.named_vars) | set(context.prepared_inputs.inv_inputs.variables)
    missing = {role: name for role, name in result.variable_roles.items() if name not in available_names}
    if missing:
        details = ", ".join(f"{role}={name!r}" for role, name in sorted(missing.items()))
        raise ValueError(
            "Custom RHIME model variable roles refer to names absent from both the PyMC model and "
            f"prepared inversion inputs: {details}."
        )

    required_roles = {"concentration"}
    if output_format not in ("none", "inv_out"):
        required_roles.update(
            {
                "observation",
                "observation_error",
                "observation_repeatability",
                "observation_variability",
                "minimum_error",
                "model_error",
            }
        )
        if context.multisector:
            for sector in context.run_spec.model.sectors:
                required_roles.update(
                    {
                        f"flux_scale:{sector.name}",
                        f"flux_contribution:{sector.name}",
                        f"emissions_sensitivity:{sector.name}",
                    }
                )
        else:
            required_roles.update(
                {"flux_scale", "flux_contribution", "emissions_sensitivity"}
            )
        if context.run_spec.model.use_bc:
            required_roles.update(
                {"baseline_scale", "baseline_sensitivity", "boundary"}
            )
        if context.run_spec.model.add_offset:
            required_roles.add("offset")

    missing_roles = sorted(required_roles - set(result.variable_roles))
    if missing_roles:
        raise ValueError(
            f"RHIME output_format={output_format!r} requires variable roles absent from the "
            f"complete model build result: {missing_roles!r}."
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
