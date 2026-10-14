"""Strict, finite-JSON records for the serialization-library prototype."""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, JsonValue


def _integer_version(value: object) -> object:
    """A JSON Boolean is not an integer artifact version."""
    if type(value) is not int:
        raise ValueError("schema_version must be an integer")
    return value


SchemaVersion = Annotated[Literal[1], BeforeValidator(_integer_version)]


class Record(BaseModel):
    """Validate external metadata without coercion or numerical execution."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        frozen=True,
    )


__all__ = ["JsonValue", "Record", "SchemaVersion"]
