"""What preparation must produce for a boundary choice.

Each choice family states which prepared sensitivity representations it can
bind and which extra prepared inputs it needs, so preparation can resolve an
omitted ``boundary_representation`` and reject an unusable explicit one
before any data are acquired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, get_args

from openghg_inversions.boundary_sensitivity import BoundaryRepresentation

if TYPE_CHECKING:
    from openghg_inversions.models.boundary import Choice


@dataclass(frozen=True)
class Preparation:
    """Prepared boundary inputs required by one boundary choice.

    Attributes:
        representation: ``"ordinary"`` for absolute ``H_bc`` or ``"centered"``
            for anomaly, mass and reference operands.
        centering_weights: Whether preparation must store
            ``bc_centering_weights``.
    """

    representation: BoundaryRepresentation
    centering_weights: bool = False

    def __post_init__(self) -> None:
        """Reject an unknown representation."""
        check_representation(self.representation)


def check_representation(representation: object) -> BoundaryRepresentation:
    """Return a known boundary representation, or raise ``ValueError``."""
    if representation not in get_args(BoundaryRepresentation):
        raise ValueError(
            f"boundary_representation must be one of {get_args(BoundaryRepresentation)!r}, not {representation!r}."
        )
    return representation  # type: ignore[return-value]


def boundary_preparation(
    choice: Choice | None, representation: BoundaryRepresentation | None = None,
) -> Preparation:
    """Resolve the prepared boundary inputs for a choice, or for no boundary.

    Args:
        choice: The configured boundary choice, or ``None`` when no boundary
            term is configured.
        representation: Explicitly requested representation, or ``None`` to
            use the choice's default.

    Returns:
        The required preparation. Without a choice, the requested
        representation or ``"ordinary"``.

    Raises:
        ValueError: If the representation is unknown or the choice cannot
            bind it.
    """
    if representation is not None:
        check_representation(representation)
    if choice is None:
        return Preparation(representation or "ordinary")
    return choice.preparation(representation)
