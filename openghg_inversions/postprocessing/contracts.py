"""Serializable scientific output information independent of a model graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, StringConstraints, field_validator

if TYPE_CHECKING:
    import xarray as xr


from openghg_inversions._record_schema import JsonValue, Record, SchemaVersion


_SCHEMA = "openghg_inversions.output_contract"
NonemptyString = Annotated[str, StringConstraints(pattern=r"\S")]
OutputFormat = Literal["none", "inv_out", "basic", "paris", "legacy"]


class OutputContractRecord(Record):
    """Version-1 output metadata, with a generated JSON Schema."""

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


@dataclass(frozen=True)
class OutputContract:
    """Bind scientific roles and a selected state axis to durable model data.

    Args:
        variable_roles: Nonempty mapping of nonempty scientific role strings to
            nonempty variable names in samples or prepared data.
        supported_output_formats: Supported writer names from ``none``, ``inv_out``,
            ``basic``, ``paris`` and ``legacy``. Must include ``none``.
        metadata: Finite JSON-compatible builder identity and scientific provenance.
        state_dimension_mapping: Optional explicit ``trace`` and ``basis`` state
            dimensions for one output view over the shared posterior. Must be empty
            or contain exactly these two keys with nonempty string values.

    Raises:
        ValueError: A role, format, metadata value or dimension mapping violates
            the constraints above.
    """

    variable_roles: Mapping[str, str]
    supported_output_formats: tuple[str, ...] = ("none",)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    state_dimension_mapping: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and copy metadata without touching numerical arrays."""
        record = self._record()
        object.__setattr__(self, "variable_roles", record.variable_roles)
        object.__setattr__(self, "supported_output_formats", tuple(record.supported_output_formats))
        object.__setattr__(self, "metadata", record.metadata)
        object.__setattr__(self, "state_dimension_mapping", record.state_dimension_mapping)

    def _record(self) -> OutputContractRecord:
        return OutputContractRecord(
            schema=_SCHEMA,
            schema_version=1,
            variable_roles=self.variable_roles,
            supported_output_formats=(
                list(self.supported_output_formats)
                if isinstance(self.supported_output_formats, tuple) else self.supported_output_formats
            ),
            metadata=self.metadata,
            state_dimension_mapping=self.state_dimension_mapping,
        )

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
        """Return a versioned JSON value with no numerical payload or callable."""
        return self._record().model_dump(mode="json", by_alias=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OutputContract:
        """Load the exact supported schema, rejecting incomplete or unknown data."""
        record = OutputContractRecord.model_validate(value)
        return cls(
            variable_roles=record.variable_roles,
            supported_output_formats=tuple(record.supported_output_formats),
            metadata=record.metadata,
            state_dimension_mapping=record.state_dimension_mapping,
        )
