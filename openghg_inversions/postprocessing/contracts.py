"""Serializable scientific output information independent of a model graph."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Field, StringConstraints, field_validator

if TYPE_CHECKING:
    import xarray as xr


from openghg_inversions._record_schema import JsonValue, Record, SchemaVersion


_SCHEMA = "openghg_inversions.output_contract"
NonemptyString = Annotated[str, StringConstraints(pattern=r"\S")]
OutputFormat = Literal["none", "inv_out", "basic", "paris", "legacy"]


class OutputContract(Record):
    """Bind scientific roles and a selected state axis to durable model data.

    All fields are required when validating a saved contract. Use :meth:`create`
    for a new contract with runtime defaults. Metadata owns its ordinary JSON
    containers; this value carries no numerical arrays or callable objects.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    schema_name: Literal["openghg_inversions.output_contract"] = Field(alias="schema")
    schema_version: SchemaVersion
    variable_roles: Annotated[
        dict[NonemptyString, NonemptyString],
        Field(min_length=1, json_schema_extra={"propertyNames": {"pattern": r"\S"}}),
    ]
    supported_output_formats: Annotated[
        list[OutputFormat], Field(min_length=1, json_schema_extra={"contains": {"const": "none"}}),
    ]
    metadata: dict[str, JsonValue]
    state_dimension_mapping: Annotated[
        dict[Literal["trace", "basis"], NonemptyString],
        Field(json_schema_extra={"oneOf": [{"maxProperties": 0}, {"required": ["trace", "basis"]}]}),
    ]

    @field_validator("supported_output_formats")
    @classmethod
    def _require_none(cls, values: list[OutputFormat]) -> list[OutputFormat]:
        if "none" not in values:
            raise ValueError("supported_output_formats must include 'none'")
        return list(dict.fromkeys(values))

    @field_validator("state_dimension_mapping")
    @classmethod
    def _require_both_dimensions(cls, values: dict[str, str]) -> dict[str, str]:
        if values and set(values) != {"trace", "basis"}:
            raise ValueError("State-dimension mappings require both 'trace' and 'basis'")
        return values

    @classmethod
    def create(
        cls, *, variable_roles: Mapping[str, str],
        supported_output_formats: tuple[OutputFormat, ...] | list[OutputFormat] = ("none",),
        metadata: Mapping[str, JsonValue] | None = None,
        state_dimension_mapping: Mapping[str, str] | None = None,
    ) -> OutputContract:
        """Create new output metadata; saved contracts must supply every field."""
        return cls(
            schema=_SCHEMA, schema_version=1, variable_roles=dict(variable_roles),
            supported_output_formats=(list(supported_output_formats)
                                      if isinstance(supported_output_formats, tuple) else supported_output_formats),
            metadata={} if metadata is None else dict(metadata),
            state_dimension_mapping={} if state_dimension_mapping is None else dict(state_dimension_mapping),
        )

    def with_updates(self, **changes: Any) -> OutputContract:
        """Validate a new output view and own its JSON containers."""
        current = self.model_dump(mode="python", by_alias=True, warnings=False)
        return type(self).model_validate(current | changes)

    def validate_requested_output(self, output_format: str) -> None:
        """Reject a writer that the bound scientific contract does not support."""
        if output_format not in self.supported_output_formats:
            raise ValueError(
                f"RHIME model does not declare output_format={output_format!r} compatible. "
                f"Declared formats: {list(self.supported_output_formats)!r}. Use output_format='none' or "
                "select a model that explicitly supports the requested RHIME output contract."
            )

    def validate_trace(self, trace: xr.DataTree) -> None:
        """Require saved reconstruction states without materializing trace arrays.

        Predictive evidence is optional. A contract advertising products must
        retain its declared flux scales, boundary scales and observation offsets
        in the posterior, even when the current invocation requests no products.
        """
        if set(self.supported_output_formats) == {"none"}:
            return
        posterior = trace.children.get("posterior")
        available = set(posterior.data_vars) if posterior is not None else set()
        missing = {
            role: name for role, name in self.variable_roles.items()
            if role.split(":", 1)[0] in {"flux_scale", "baseline_scale", "offset"} and name not in available
        }
        if missing:
            raise ValueError(f"Trace is missing output-contract reconstruction state(s): {missing!r}.")

    def to_dict(self) -> dict[str, Any]:
        """Validate current metadata before projecting it into a versioned JSON value."""
        current = self.model_dump(mode="python", by_alias=True, warnings=False)
        return type(self).model_validate(current).model_dump(mode="json", by_alias=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OutputContract:
        """Load the exact supported schema, rejecting incomplete or unknown data."""
        return cls.model_validate(value)


DEFAULT_VARIABLE_ROLES: dict[str, str] = {
    "observation": "mf",
    "observation_error": "mf_error",
    "observation_prior_factor": "mf_prior_factor",
    "observation_prior_upper_level_factor": "mf_prior_upper_level_factor",
    "observation_repeatability": "mf_repeatability",
    "observation_variability": "mf_variability",
    "flux_scale": "x",
    "model_error": "epsilon",
    "concentration": "y",
    "baseline": "mu_bc",
    "boundary": "mu_bc",
    "offset": "offset",
    "emissions_sensitivity": "hx",
    "baseline_sensitivity": "hbc",
    "minimum_error": "min_error",
}


def resolve_predictive_names(
    names: list[str], *, available: set[str], variable_roles: Mapping[str, str] | None = None,
) -> list[str]:
    """Resolve predictive roles with the same concentration convention as outputs.

    Concrete names remain valid. The historical default concentration name is
    redirected only when absent and an explicit concentration role is supplied.
    """
    roles = dict(variable_roles or {})
    concentration = DEFAULT_VARIABLE_ROLES["concentration"]
    if concentration not in available and "concentration" in roles:
        roles.setdefault(concentration, roles["concentration"])
    resolved = [roles.get(name, name) for name in names]
    missing = sorted(set(resolved) - available)
    if missing:
        raise ValueError(
            "RhimeSampler posterior-predictive variables are absent from the model: "
            f"{missing!r}. Use concrete model names or semantic keys from `variable_roles`."
        )
    return list(dict.fromkeys(resolved))
