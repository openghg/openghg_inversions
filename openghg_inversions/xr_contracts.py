"""Declaration machinery for static xarray data contracts.

A contract is a phantom subtype of ``xr.DataArray``, ``xr.Dataset`` or
``xr.DataTree`` that names the guarantees a handoff establishes. Runtime
values stay ordinary xarray objects: a contract's class-level ``validate``
checks structure and returns the identical object with the contract's static
type, and trusted producers declare the contract as their return type. See
:doc:`/development/validation_and_xarray` for the pattern.

Concrete contracts live in ``contracts.py`` modules beside their producers and
are declared with :func:`xr_contract`. This module only holds the decorator
and the registry; shared conformance tests discover ``contracts.py`` modules
by filename and compare the registry with each contract's examples.
"""

from __future__ import annotations

import inspect
from typing import Any

import xarray as xr


# Values are declared contract classes; their validators have contract-specific signatures.
_REGISTRY: dict[str, type[Any]] = {}
_BASES = (xr.DataArray, xr.Dataset, xr.DataTree)


def xr_contract[T: xr.DataArray | xr.Dataset | xr.DataTree](cls: type[T]) -> type[T]:
    """Declare an xarray contract type and add it to the registry.

    Args:
        cls: Phantom subtype of one supported xarray container, defined in a
            ``contracts`` module, with its own staticmethod or classmethod
            ``validate``.

    Returns:
        The same class, unchanged.

    Raises:
        TypeError: If ``cls`` does not subclass a supported container, lacks
            its own class-level ``validate``, or is defined outside a
            ``contracts`` module.
    """
    if not (isinstance(cls, type) and issubclass(cls, _BASES)):
        raise TypeError(f"{cls!r} must subclass xr.DataArray, xr.Dataset or xr.DataTree.")
    validator = inspect.getattr_static(cls, "validate", None)
    if "validate" not in vars(cls) or not isinstance(validator, staticmethod | classmethod):
        raise TypeError(f"{cls.__qualname__} must define its own staticmethod or classmethod validate.")
    if cls.__module__.rsplit(".", 1)[-1] != "contracts":
        raise TypeError(f"{cls.__qualname__} must be declared in a 'contracts' module, not {cls.__module__!r}.")
    _REGISTRY[f"{cls.__module__}.{cls.__qualname__}"] = cls
    return cls


def registered_contracts() -> dict[str, type[Any]]:
    """Return the declared contracts imported so far, keyed by qualified name.

    Returns:
        A copy of the registry. Only contracts whose modules have been
        imported are present; conformance tests import every discovered
        ``contracts`` module first.
    """
    return dict(_REGISTRY)


__all__ = ["registered_contracts", "xr_contract"]
