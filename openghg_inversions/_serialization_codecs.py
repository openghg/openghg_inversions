"""Lightweight private codecs shared by versioned artifact schemas."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


_TAGGED_JSON_VALUE_ENCODING = "tagged_json_v1"


def _encode_tagged_json_value(value: object) -> str:
    """Encode a supported scalar or nested tuple as compact tagged JSON.

    Args:
        value: String, Boolean, integer, float, NumPy scalar, timestamp, or
            nested tuple composed from those types.

    Returns:
        Tagged compact JSON representation of ``value``.

    Raises:
        TypeError: If ``value`` has an unsupported type.
    """
    if isinstance(value, np.datetime64):
        payload = ["datetime64", [str(value.dtype), int(value.astype(np.int64))]]
    elif isinstance(value, pd.Timestamp):
        payload = ["timestamp", value.isoformat()]
    elif isinstance(value, np.generic):
        value = value.item()
        return _encode_tagged_json_value(value)
    else:
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
    if kind == "datetime64":
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[0], str)
            or not isinstance(value[1], int)
            or isinstance(value[1], bool)
        ):
            raise ValueError("Encoded tagged JSON datetime64 must contain a dtype and integer value")
        try:
            dtype = np.dtype(value[0])
            if dtype.kind != "M":
                raise TypeError
            return np.asarray(value[1], dtype=dtype)[()]
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("Encoded tagged JSON datetime64 has an invalid dtype or value") from error
    if kind == "timestamp":
        if not isinstance(value, str):
            raise ValueError("Encoded tagged JSON timestamp must contain a string")
        try:
            return pd.Timestamp(value)
        except ValueError as error:
            raise ValueError("Encoded tagged JSON timestamp is invalid") from error
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
