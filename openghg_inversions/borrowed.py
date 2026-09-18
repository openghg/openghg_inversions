"""Experimental static markers for borrowed NumPy and xarray arrays.

The :func:`borrow` helper is an identity function at runtime.  Its overloads
cast supported arrays to phantom subclasses whose common mutation entry points
are marked with PEP 702 ``deprecated`` diagnostics.  The diagnostic means
"mutation is unsupported through this borrowed reference"; the underlying
object is neither copied nor made read-only.

This is an experimental module without a stability guarantee.  Its deliberately
narrow public surface consists only of :class:`BorrowedNDArray`,
:class:`BorrowedDataArray`, and :func:`borrow`.

It deliberately provides a warning aid rather than an immutability claim.
Static diagnostics can be bypassed by an untyped alias, ``cast``,
``Any``, widening to an ordinary ``ndarray`` or ``DataArray`` parameter,
mutating helper APIs (including NumPy ``out=``), NumPy functions whose stubs
erase subclasses, or xarray escape hatches such as ``coords``, ``variables``,
``data_vars``, ``attrs``, and ``loc``.
Mutation through aliases created before :func:`borrow` is also invisible.  The
``DataArray.data`` marker preserves arbitrary duck-array use through
``__getattr__``, so slices of that data are currently typed as ``Any`` and are
another false negative.  NumPy view-producing APIs are only protected when
their existing annotations preserve ``Self``; not every NumPy operation does.
Conversely, ``values`` or ``to_numpy`` may materialize an independent buffer
from a lazy backend, but the experiment conservatively retains the borrowed
marker because the backend is not represented in the ``DataArray`` type.  A
deep ``DataArray.copy()`` relinquishes the marker as an explicit mutation-rights
escape; for lazy and duck-array backends this says nothing about independent
memory, graph isolation, or computation.

PEP 702 diagnostics are checker configuration, not runtime warnings.  Pyright
reports explicit calls through ``reportDeprecated``, but version 1.1.408 does
not report assignment syntax that implicitly calls a deprecated
``__setitem__``.  The mutation operands therefore also use ``Never`` to make
direct assignments fail in both Pyright and Mypy.  Mypy recognizes the PEP 702
annotations but requires the ``deprecated`` optional error code to be enabled.
The project does not yet publish a package-wide PEP 561 ``py.typed`` marker;
this experimental contract is enforced in the repository checker environment
until its coverage and first production uses have been evaluated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, overload

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from typing_extensions import Never, deprecated
else:
    Never = Any

    def deprecated(message: str, *, category: None = None) -> Any:
        """Return a runtime no-op for a type-checker-only PEP 702 marker."""

        def decorator(value: Any) -> Any:
            return value

        return decorator


__all__ = ["BorrowedDataArray", "BorrowedNDArray", "borrow"]


class _BorrowedArrayData(Protocol):
    """Static facade for the arbitrary duck array returned by ``DataArray.data``.

    Attribute access remains ``Any`` so that backend-specific observational
    APIs stay usable.  Only direct indexed assignment is marked; aliases
    returned by indexed reads are not tracked.
    """

    def __getitem__(self, key: Any) -> Any:
        """Return an item or untracked alias from the underlying duck array."""
        ...

    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def __setitem__(self, key: Any, mutation_requires_mutable_copy: Never) -> None:
        """Mark direct indexed assignment as unsupported."""
        ...

    def __getattr__(self, name: str) -> Any:
        """Preserve access to backend-specific observational attributes."""
        ...


class BorrowedNDArray(np.ndarray[Any, Any]):
    """Type-only NumPy subtype marking common in-place operations.

    Instances returned by :func:`borrow` are ordinary ``numpy.ndarray``
    objects at runtime.  Do not construct this class or use it for runtime
    ``isinstance`` checks.
    """

    __slots__ = ()

    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def __setitem__(  # type: ignore[override]
        self, key: Any, mutation_requires_mutable_copy: Never
    ) -> None:
        """Mark indexed assignment as unsupported."""
        super().__setitem__(key, mutation_requires_mutable_copy)

    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def fill(self, mutation_requires_mutable_copy: Never) -> None:  # type: ignore[override]
        """Mark in-place filling as unsupported."""
        super().fill(mutation_requires_mutable_copy)

    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def __iadd__(  # type: ignore[override,misc]
        self, mutation_requires_mutable_copy: Never
    ) -> BorrowedNDArray:
        """Mark in-place addition as unsupported."""
        return cast(BorrowedNDArray, super().__iadd__(mutation_requires_mutable_copy))

    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def partition(self, *args: Any, **kwargs: Any) -> None:
        """Mark in-place partitioning as unsupported."""
        super().partition(*args, **kwargs)

    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def put(self, *args: Any, **kwargs: Any) -> None:
        """Mark indexed in-place insertion as unsupported."""
        super().put(*args, **kwargs)

    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def resize(self, *args: Any, **kwargs: Any) -> None:
        """Mark in-place resizing as unsupported."""
        super().resize(*args, **kwargs)

    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def setfield(self, *args: Any, **kwargs: Any) -> None:
        """Mark in-place field assignment as unsupported."""
        super().setfield(*args, **kwargs)

    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def setflags(self, *args: Any, **kwargs: Any) -> None:
        """Mark changes to array flags as unsupported."""
        super().setflags(*args, **kwargs)

    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def sort(self, *args: Any, **kwargs: Any) -> None:
        """Mark in-place sorting as unsupported."""
        super().sort(*args, **kwargs)

    def copy(self, order: Any = "C") -> np.ndarray[Any, Any]:  # type: ignore[override]
        """Return a mutable independent NumPy array.

        Args:
            order: Memory-layout order passed to ``numpy.ndarray.copy``.

        Returns:
            An ordinary mutable NumPy array with independent storage.
        """
        return super().copy(order)


class BorrowedDataArray(xr.DataArray):
    """Type-only xarray ``DataArray`` subtype marking common mutations.

    ``values`` and ``to_numpy`` return :class:`BorrowedNDArray` statically.
    ``data`` returns a narrow duck-array facade that marks direct indexed
    assignment without assuming a NumPy backend.
    """

    __slots__ = ()

    @property
    def data(self) -> _BorrowedArrayData:
        """Return the exact borrowed NumPy, Dask, or other duck-array backend."""
        return cast(_BorrowedArrayData, super().data)

    @data.setter
    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def data(self, value: Never) -> None:
        """Mark replacement of underlying duck-array data as unsupported."""
        self.variable.data = value

    @property  # type: ignore[override]
    def values(self) -> BorrowedNDArray:
        """Return NumPy values with a conservative borrowed static marker.

        Xarray may compute or materialize the backend during conversion.
        """
        return cast(BorrowedNDArray, super().values)

    @values.setter
    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def values(self, value: Never) -> None:
        """Mark replacement of NumPy values as unsupported."""
        self.variable.values = value

    def to_numpy(self) -> BorrowedNDArray:
        """Convert to NumPy while conservatively retaining the borrowed marker.

        Conversion may compute or materialize the backend. Xarray does not
        promise that the returned NumPy array is an independent copy.
        """
        return cast(BorrowedNDArray, super().to_numpy())

    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def __setitem__(  # type: ignore[override]
        self, key: Any, mutation_requires_mutable_copy: Never
    ) -> None:
        """Mark indexed and coordinate assignment as unsupported."""
        super().__setitem__(key, mutation_requires_mutable_copy)

    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def __delitem__(  # type: ignore[override]
        self, mutation_requires_mutable_copy: Never
    ) -> None:
        """Mark coordinate deletion as unsupported."""
        super().__delitem__(mutation_requires_mutable_copy)

    @deprecated("Mutation through a borrowed array reference is unsupported", category=None)
    def __iadd__(  # type: ignore[override,misc]
        self, mutation_requires_mutable_copy: Never
    ) -> BorrowedDataArray:
        """Mark in-place addition as unsupported."""
        return cast(BorrowedDataArray, super().__iadd__(mutation_requires_mutable_copy))

    @overload  # type: ignore[override]
    def copy(self, deep: Literal[True] = True, data: Any = None) -> xr.DataArray: ...

    @overload
    def copy(self, deep: Literal[False], data: Any = None) -> BorrowedDataArray: ...

    @overload
    def copy(self, deep: bool, data: Any = None) -> xr.DataArray | BorrowedDataArray: ...

    def copy(  # type: ignore[override]
        self, deep: bool = True, data: Any = None
    ) -> xr.DataArray | BorrowedDataArray:
        """Relinquish borrowing after a deep copy; retain it if shallow.

        Args:
            deep: Whether xarray should copy the wrapped data and coordinates.
            data: Optional replacement data passed through to xarray.

        Returns:
            An ordinary mutable ``DataArray`` static type when ``deep`` is
            true, otherwise another borrowed static reference.

        Note:
        A deep copy returns an ordinary mutable static type but does not promise
        backend-independent memory or graph isolation for lazy or duck arrays.
        """
        return cast(xr.DataArray | BorrowedDataArray, super().copy(deep=deep, data=data))


@overload
def borrow(value: np.ndarray[Any, Any]) -> BorrowedNDArray: ...


@overload
def borrow(value: xr.DataArray) -> BorrowedDataArray: ...


def borrow(value: np.ndarray[Any, Any] | xr.DataArray) -> BorrowedNDArray | BorrowedDataArray:
    """Mark a NumPy or xarray array as borrowed for static checking.

    Args:
        value: NumPy array or xarray ``DataArray`` whose existing object
            identity and lazy backend must be preserved.

    Returns:
        The exact input object with a borrowed static type.  No runtime type is
        changed.

    Note:
        This function performs no copy, validation, runtime wrapping, or
        write-protection.  PEP 702 diagnostics only affect supported static
        type checkers.
    """
    return cast(BorrowedNDArray | BorrowedDataArray, value)
