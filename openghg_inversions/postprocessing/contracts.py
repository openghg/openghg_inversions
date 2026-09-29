"""Serializable scientific output information independent of a model graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from typing import Any


_SCHEMA = "openghg_inversions.output_contract"
_FORMATS = frozenset({"none", "inv_out", "basic", "paris", "legacy"})


@dataclass(frozen=True)
class OutputContract:
    """Bind scientific roles and a selected state axis to durable model data.

    Args:
        variable_roles: Scientific role to variable name in samples or prepared data.
        supported_output_formats: Formats whose existing writer contracts are met.
        metadata: JSON-compatible builder identity and scientific provenance.
        state_dimension_mapping: Optional explicit ``trace`` and ``basis`` state
            dimensions for one output view over the shared posterior.
    """

    variable_roles: Mapping[str, str]
    supported_output_formats: tuple[str, ...] = ("none",)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    state_dimension_mapping: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and copy metadata without touching numerical arrays."""
        if not isinstance(self.variable_roles, Mapping) or not self.variable_roles:
            raise ValueError("OutputContract.variable_roles must be a non-empty mapping.")
        roles = dict(self.variable_roles)
        if any(not isinstance(value, str) or not value.strip() for pair in roles.items() for value in pair):
            raise ValueError("OutputContract.variable_roles requires non-empty string roles and names.")
        if not isinstance(self.supported_output_formats, (tuple, list)) or any(
            not isinstance(value, str) or value not in _FORMATS for value in self.supported_output_formats
        ):
            raise ValueError("OutputContract.supported_output_formats contains unsupported values.")
        formats = tuple(dict.fromkeys(self.supported_output_formats))
        if "none" not in formats:
            raise ValueError("OutputContract.supported_output_formats must include 'none'.")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("OutputContract.metadata must be a JSON-compatible mapping.")
        try:
            metadata = json.loads(json.dumps(dict(self.metadata), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("OutputContract.metadata must be finite JSON-compatible data.") from exc
        if not isinstance(self.state_dimension_mapping, Mapping):
            raise ValueError("OutputContract.state_dimension_mapping must be a mapping.")
        dimensions = dict(self.state_dimension_mapping)
        if dimensions and (
            set(dimensions) != {"trace", "basis"}
            or any(not isinstance(value, str) or not value.strip() for value in dimensions.values())
        ):
            raise ValueError(
                "State-dimension mappings require exactly the non-empty keys 'trace' and 'basis'."
            )
        object.__setattr__(self, "variable_roles", roles)
        object.__setattr__(self, "supported_output_formats", formats)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "state_dimension_mapping", dimensions)

    def validate_requested_output(self, output_format: str) -> None:
        """Reject a writer that the bound scientific contract does not support."""
        if output_format not in self.supported_output_formats:
            raise ValueError(
                f"RHIME model does not declare output_format={output_format!r} compatible. "
                f"Declared formats: {list(self.supported_output_formats)!r}. Use output_format='none' or "
                "select a model that explicitly supports the requested RHIME output contract."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON value with no numerical payload or callable."""
        return {
            "schema": _SCHEMA,
            "schema_version": 1,
            "variable_roles": dict(self.variable_roles),
            "supported_output_formats": list(self.supported_output_formats),
            "metadata": dict(self.metadata),
            "state_dimension_mapping": dict(self.state_dimension_mapping),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OutputContract:
        """Load the exact supported schema, rejecting incomplete or unknown data."""
        expected = {
            "schema",
            "schema_version",
            "variable_roles",
            "supported_output_formats",
            "metadata",
            "state_dimension_mapping",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("Output contract has missing or unexpected fields.")
        if (
            value["schema"] != _SCHEMA
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
        ):
            raise ValueError("Output contract has an unsupported schema or schema_version.")
        return cls(
            variable_roles=value["variable_roles"],
            supported_output_formats=value["supported_output_formats"],
            metadata=value["metadata"],
            state_dimension_mapping=value["state_dimension_mapping"],
        )
