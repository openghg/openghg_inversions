"""Shared PyMC/PyTensor runtime configuration."""

from __future__ import annotations

import os
import sys
from typing import Any, cast


def _pytensor_flag_is_explicit(name: str) -> bool:
    """Return whether the process supplied one named ``PYTENSOR_FLAGS`` value."""
    for setting in os.environ.get("PYTENSOR_FLAGS", "").split(","):
        key, separator, _ = setting.partition("=")
        if separator and key.strip() == name:
            return True
    return False


_TRUE_ENVIRONMENT_VALUES = frozenset({"1", "true", "yes", "on"})


def jax_x64_requested() -> bool:
    """Return whether the process asked JAX for 64-bit arrays with ``JAX_ENABLE_X64``."""
    return os.environ.get("JAX_ENABLE_X64", "").strip().lower() in _TRUE_ENVIRONMENT_VALUES


def configure_pytensor() -> None:
    """Apply memory-conscious defaults only to a fresh PyTensor runtime.

    Process owners can select another graph precision with ``PYTENSOR_FLAGS``
    before importing PyTensor, PyMC, or OpenGHG Inversions. ``JAX_ENABLE_X64``
    set to a true value also selects ``floatX=float64`` unless ``PYTENSOR_FLAGS``
    names ``floatX``: PyTensor's JAX backend sets JAX precision from ``floatX``
    when it is first imported, so a float32 default would silently turn the
    request off. An already-imported runtime is left untouched because changing
    process-wide tensor defaults after graph libraries have initialized is
    ambiguous and unsafe.
    """
    pytensor_was_loaded = "pytensor" in sys.modules
    import pytensor

    config = cast(Any, pytensor).config
    if pytensor_was_loaded:
        return
    if not _pytensor_flag_is_explicit("floatX"):
        config.floatX = "float64" if jax_x64_requested() else "float32"
    if config.floatX == "float32" and not _pytensor_flag_is_explicit("warn_float64"):
        config.warn_float64 = "warn"
