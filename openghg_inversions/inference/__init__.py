"""Inference mechanics shared by explicit scientific recipes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sampling import RhimeSampler

__all__ = ["RhimeSampler"]


def __getattr__(name: str) -> type["RhimeSampler"]:
    """Load the sampler only when requested, keeping diagnostics backend-neutral."""
    if name == "RhimeSampler":
        from .sampling import RhimeSampler

        return RhimeSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
