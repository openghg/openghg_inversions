"""Shared PyMC/PyTensor runtime configuration."""

from __future__ import annotations

from typing import Any, cast


def configure_pytensor() -> None:
    """Apply OpenGHG Inversions PyTensor defaults before importing PyMC."""
    import pytensor

    config = cast(Any, pytensor).config
    config.floatX = "float32"
    config.warn_float64 = "warn"
