"""Compatibility imports for shared additive boundary components.

New callers should use :mod:`openghg_inversions.models.additive_boundary`.
"""

from openghg_inversions.models.additive_boundary import (
    AdditiveBoundaryResult,
    CenteredBoundary,
    add_centered_boundary,
    prepare_centered_boundary,
)

__all__ = [
    "AdditiveBoundaryResult",
    "CenteredBoundary",
    "add_centered_boundary",
    "prepare_centered_boundary",
]
