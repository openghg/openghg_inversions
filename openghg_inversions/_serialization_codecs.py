"""Lightweight private codecs shared by versioned artifact schemas."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from typing import Any

import numpy as np


_TAGGED_JSON_VALUE_ENCODING = "tagged_json_v1"


def _encode_tagged_json_value(value: object) -> str:
    """Encode a supported scalar or nested tuple as compact tagged JSON.

    Args:
        value: String, Boolean, integer, float, NumPy scalar, or nested tuple
            composed from those types.

    Returns:
        Tagged compact JSON representation of ``value``.

    Raises:
        TypeError: If ``value`` has an unsupported type.
    """
    if isinstance(value, np.generic):
        value = value.item()
    payload: list[object]
    if isinstance(value, tuple):
        payload = ["tuple", [_encode_tagged_json_value(item) for item in value]]
    elif isinstance(value, bool):
        payload = ["bool", value]
    elif isinstance(value, int):
        payload = ["int", value]
    elif isinstance(value, float):
        payload = ["float", value]
    elif isinstance(value, str):
        payload = ["str", value]
    else:
        raise TypeError(f"Unsupported tagged JSON value type: {type(value).__name__}")
    return json.dumps(payload, separators=(",", ":"))


def _decode_tagged_json_value(encoded: str) -> object:
    """Strictly decode a value produced by :func:`_encode_tagged_json_value`.

    Args:
        encoded: Tagged compact JSON value.

    Returns:
        Restored supported Python scalar or tuple value.

    Raises:
        ValueError: If the JSON, tag, or tagged payload is malformed.
    """
    if not isinstance(encoded, str):
        raise ValueError("Encoded tagged JSON value must be a string")
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Encoded tagged JSON value is not valid JSON") from error
    if not isinstance(payload, list) or len(payload) != 2 or not isinstance(payload[0], str):
        raise ValueError("Encoded tagged JSON value must be a two-item tagged array")

    kind, value = payload
    if kind == "tuple":
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("Encoded tagged JSON tuple must contain encoded string elements")
        return tuple(_decode_tagged_json_value(item) for item in value)
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError("Encoded tagged JSON bool must contain a Boolean")
        return value
    if kind == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("Encoded tagged JSON int must contain an integer")
        return value
    if kind == "float":
        if not isinstance(value, float):
            raise ValueError("Encoded tagged JSON float must contain a float")
        return value
    if kind == "str":
        if not isinstance(value, str):
            raise ValueError("Encoded tagged JSON str must contain a string")
        return value
    raise ValueError(f"Unknown encoded tagged JSON value kind {kind!r}")


def _numpy_scalar_json_default(value: object) -> object:
    """Convert a NumPy scalar to its JSON-compatible Python scalar."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _decode_serialized_bool(value: object, name: str) -> bool:
    """Decode a serialized Boolean represented only by Boolean or integer 0/1."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"Serialized {name} must be Boolean or integer 0/1")


def _json_value(value: Any, *, materialize_dataarrays: bool = False) -> Any:
    """Convert supported metadata to JSON values without implicit lazy execution.

    NumPy dates and durations retain their string meaning at every precision;
    arrays become nested lists. Paths, dates, times and durations become strings,
    and slices retain their bounds in a mapping. Unsupported objects raise
    ``TypeError`` instead of being stringified. Bare Dask arrays are unsupported.

    ``materialize_dataarrays=True`` opts into an eager serialization boundary:
    DataArrays are computed and encoded with dimensions, dimension coordinates
    and values, without mutating the borrowed array.
    """

    def convert(item: Any) -> Any:
        return _json_value(item, materialize_dataarrays=materialize_dataarrays)

    if isinstance(value, np.datetime64 | np.timedelta64):
        return str(value)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, Path | timedelta):
        return str(value)
    if isinstance(value, np.ndarray):
        return convert(value[()]) if value.ndim == 0 else [convert(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, slice):
        return {
            "type": "slice",
            "start": convert(value.start),
            "stop": convert(value.stop),
            "step": convert(value.step),
        }
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            converted_key = convert(key)
            if converted_key is not None and not isinstance(converted_key, str | bool | int | float):
                raise TypeError(f"Unsupported JSON mapping key type: {type(key).__name__}")
            result[str(converted_key)] = convert(item)
        return result
    if isinstance(value, tuple | list):
        return [convert(item) for item in value]
    if materialize_dataarrays:
        import xarray as xr

        if isinstance(value, xr.DataArray):
            materialized = value.compute()
            return {
                "dims": [str(dim) for dim in materialized.dims],
                "coords": {
                    str(dim): convert(materialized.coords[dim].to_numpy())
                    for dim in materialized.dims
                    if dim in materialized.coords
                },
                "values": convert(materialized.to_numpy()),
            }
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _copy_json_mapping(value: Mapping[str, Any], name: str, *, sort_keys: bool = False) -> dict[str, Any]:
    """Copy finite JSON metadata without extending the JSON encoder's input types."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False, sort_keys=sort_keys))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON-compatible data.") from exc


def _decode_json_mapping(raw: object, name: str) -> dict[str, Any]:
    """Decode a JSON object string, rejecting malformed or nonfinite metadata."""
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a JSON object string.")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} contains invalid JSON.") from exc
    return _copy_json_mapping(value, name)
