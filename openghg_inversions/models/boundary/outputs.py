"""Graph pieces and declared scientific outputs of one bound boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pytensor.tensor.variable import TensorVariable

from openghg_inversions.models.components import AffineBlock


@dataclass(frozen=True)
class BoundaryCorrection:
    """Graph pieces of one sampled boundary correction, such as a mean shift or anomalies.

    Attributes:
        state: Correction on every bound boundary state, in concentration units.
        signal: Observation response ``M state``.
        block: Affine block: the response of ``M`` to each sampled coefficient,
            with those physical coefficients.
        latent: Sampled variable behind the correction, after any
            reparameterization.
    """

    state: TensorVariable
    signal: TensorVariable
    block: AffineBlock
    latent: TensorVariable


@dataclass(frozen=True)
class BoundaryOutputs:
    """Roles, units and metadata a bound boundary declares for its outputs.

    Every value is derived from resolved choices and binding facts, not from a
    built graph, so recipes can record them before sampling.

    Attributes:
        roles: Scientific role name to graph variable name.
        units: Graph variable name to physical units.
        metadata: Bounded JSON-safe facts: ``boundary_formulation``,
            ``bc_frequency``, ``bc_anchor_time`` and, for centered transport,
            ``boundary_representation``.
    """

    roles: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def prefixed(self, prefix: str) -> BoundaryOutputs:
        """Return the same declarations for graph names built with ``prefix``."""
        return BoundaryOutputs(
            {prefix + role: prefix + name for role, name in self.roles.items()},
            {prefix + name: unit for name, unit in self.units.items()},
            dict(self.metadata),
        )


__all__ = ["BoundaryCorrection", "BoundaryOutputs"]
