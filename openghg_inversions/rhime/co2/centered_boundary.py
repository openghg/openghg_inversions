"""Compatibility imports for shared numerical boundary components.

New callers should use :mod:`openghg_inversions.models.boundary`.
"""

from openghg_inversions.models.boundary import (
    OPERAND_NAMES,
    CenteredBoundaryMap,
    CenteredBoundaryResult,
    add_centered_boundary,
    bind_boundary_operands,
    require_centered_precision,
    restore_absolute_predictives,
)

__all__ = [
    "OPERAND_NAMES",
    "CenteredBoundaryMap",
    "CenteredBoundaryResult",
    "add_centered_boundary",
    "bind_boundary_operands",
    "require_centered_precision",
    "restore_absolute_predictives",
]
