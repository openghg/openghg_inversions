"""Deprecated keyword resolution for independently supported basis adapters."""

import warnings
from enum import Enum
from typing import Any


class _Unset(Enum):
    TOKEN = 0


UNSET = _Unset.TOKEN


def resolve_deprecated_keyword(
    old: str, new: str, old_value: Any, new_value: Any, *, default: Any = None
) -> Any:
    """Resolve a modern keyword, rejecting simultaneous old/new spellings."""
    if old_value is not UNSET:
        if new_value is not UNSET:
            raise ValueError(f"Supply only `{new}`; `{old}` and `{new}` cannot be supplied together.")
        warnings.warn(
            f"`{old}` is deprecated; use `{new}` instead. `{old}` will be removed in 0.9.",
            DeprecationWarning,
            stacklevel=3,
        )
        return old_value
    return default if new_value is UNSET else new_value
