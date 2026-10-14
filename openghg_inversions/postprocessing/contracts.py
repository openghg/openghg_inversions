"""Serializable scientific output information independent of a model graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xarray as xr


from openghg_inversions._serialization_codecs import _copy_json_mapping


_SCHEMA = "openghg_inversions.output_contract"
_FORMATS = frozenset({"none", "inv_out", "basic", "paris", "legacy"})


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
        metadata = _copy_json_mapping(self.metadata, "OutputContract.metadata")
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

    def validate_trace(self, trace: xr.DataTree) -> None:
        """Require saved reconstruction states without materializing trace arrays.

        Predictive evidence is optional. A contract advertising products must
        retain its declared flux and boundary scales in the posterior, even
        when the current invocation does not request products.
        """
        if set(self.supported_output_formats) == {"none"}:
            return
        posterior = trace.children.get("posterior")
        available = set(posterior.data_vars) if posterior is not None else set()
        missing = {
            role: name for role, name in self.variable_roles.items()
            if role.split(":", 1)[0] in {"flux_scale", "baseline_scale"} and name not in available
        }
        if missing:
            raise ValueError(f"Trace is missing output-contract reconstruction state(s): {missing!r}.")

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
