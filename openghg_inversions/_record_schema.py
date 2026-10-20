"""Validation primitives for configuration and finite JSON handoffs."""

from collections.abc import Mapping
import math
from typing import Annotated, Literal

import numpy as np
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


def normalize_runtime_scalar(value: object) -> object:
    """Unbox only known eager NumPy numbers at a runtime configuration boundary."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _runtime_number(value: object) -> object:
    return _json_number(normalize_runtime_scalar(value))


RuntimeFloat = Annotated[float, BeforeValidator(_runtime_number)]
RuntimeInt = Annotated[int, BeforeValidator(normalize_runtime_scalar)]
RuntimeConfig = ConfigDict(
    strict=True, extra="forbid", arbitrary_types_allowed=True,
    allow_inf_nan=False, validate_default=True,
)


def portable_json(value: object) -> JsonValue:
    """Copy native finite JSON values without coercing runtime objects.

    Configuration may borrow arrays, distributions, or sampler objects while
    running. Exporting those objects requires an explicit product-specific
    representation; this boundary never calls their conversion methods.
    """
    ancestors: set[int] = set()

    def visit(item: object, path: str) -> JsonValue:
        if item is None or type(item) in (str, bool, int):
            return item  # type: ignore[return-value]
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError(f"{path}: JSON numbers must be finite")
            return item
        if type(item) not in (dict, list):
            raise ValueError(f"{path}: {type(item).__name__} has no portable JSON representation")
        identity = id(item)
        if identity in ancestors:
            raise ValueError(f"{path}: cyclic containers have no portable JSON representation")
        ancestors.add(identity)
        try:
            if isinstance(item, dict):
                result: dict[str, JsonValue] = {}
                for key, child in item.items():
                    if type(key) is not str:
                        raise ValueError(f"{path}: JSON object keys must be strings")
                    result[key] = visit(child, f"{path}[{key!r}]")
                return result
            return [visit(child, f"{path}[{index}]") for index, child in enumerate(item)]
        finally:
            ancestors.remove(identity)

    return visit(value, "$")


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


__all__ = [
    "JsonFloat", "JsonValue", "Record", "RuntimeConfig", "RuntimeFloat", "RuntimeInt", "SchemaVersion",
    "normalize_runtime_scalar", "portable_json", "validate_json_keys",
]
