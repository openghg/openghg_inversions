"""Declared scientific outputs of one bound boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Outputs:
    """Roles, units and metadata a bound boundary declares for its outputs.

    Every value is derived from resolved choices and binding facts, not from a
    built graph, so recipes can record them before sampling.

    Attributes:
        roles: Scientific role name to graph variable name.
        units: Graph variable name to physical units.
        metadata: Bounded JSON-safe facts: ``boundary_formulation``,
            ``bc_frequency``, ``bc_anchor_time`` and, for centered sensitivity,
            ``boundary_representation``.
    """

    roles: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def prefixed(self, prefix: str) -> Outputs:
        """Return the same declarations for graph names built with ``prefix``."""
        return Outputs(
            {prefix + role: prefix + name for role, name in self.roles.items()},
            {prefix + name: unit for name, unit in self.units.items()},
            dict(self.metadata),
        )


__all__ = ["Outputs"]
