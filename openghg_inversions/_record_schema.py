"""Validation primitives for configuration and finite JSON handoffs."""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, JsonValue


def _integer_version(value: object) -> object:
    if type(value) is not int:
        raise ValueError("schema_version must be an integer")
    return value


SchemaVersion = Annotated[Literal[1], BeforeValidator(_integer_version)]


def _json_number(value: object) -> object:
    """Reject scalar conversions that could execute a borrowed Dask graph."""
    if type(value) not in (int, float):
        raise ValueError("expected a JSON number, without array or object conversion")
    return value


JsonFloat = Annotated[float, BeforeValidator(_json_number)]
RuntimeConfig = ConfigDict(
    strict=True, extra="forbid", arbitrary_types_allowed=True,
    allow_inf_nan=False, validate_default=True,
)


def validate_json_keys(value: object) -> None:
    """Reject lossy mapping keys before an explicit JSON projection.

    Numerical values remain borrowed and are never inspected here. Projection
    of paths, dates or arrays belongs to the public writer that chooses it.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            validate_json_keys(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            validate_json_keys(item)


class Record(BaseModel):
    """Strict finite JSON metadata, without coercion or numerical execution."""

    model_config = ConfigDict(
        strict=True, extra="forbid", allow_inf_nan=False,
        validate_default=True, frozen=True,
    )


__all__ = ["JsonFloat", "JsonValue", "Record", "RuntimeConfig", "SchemaVersion", "validate_json_keys"]
