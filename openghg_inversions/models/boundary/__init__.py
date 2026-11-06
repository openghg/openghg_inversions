"""Boundary-condition components for RHIME recipes.

A boundary family is two small records in its own module:

* a *choice* (:class:`MultiplicativeChoice`, :class:`AdditiveChoice`): the
  prior specification before binding, holding the resolved scientific
  options (priors, activity, anomaly scale and the boundary period calendar)
  without arrays. Configuration adapters produce choices; they never touch
  arrays.
* a *bound boundary* (:class:`MultiplicativeBoundary`, :class:`AdditiveBoundary`)
  returned by ``choice.bind(observations, transport)``. It pairs the choice
  with borrowed labelled transport aligned to one fit's observation rows and
  expanded to its boundary periods.

A recipe binds each boundary once, materializes the bound record together
with its other inputs (``dask.compute`` traverses these dataclasses, including
choice priors; see :func:`~openghg_inversions.array_ops.to_dense_tree` and the
joint-computation example in :doc:`/development/rhime_model_development`), then calls
``boundary.add(...)`` inside its registered model. ``add`` returns an
:class:`~openghg_inversions.models.components.AffineConcentrationTerm`, which
ordinary and cached likelihoods consume without knowing the family, and
``boundary.outputs()`` declares the roles, units and metadata recorded with
the trace. :class:`BoundaryChoice` and :class:`Boundary` are structural
protocols, so adding a centered boundary family means adding one module with
a choice and a bound record, plus its configuration adapter and tests;
nothing here enumerates the families.

Transport is either absolute ``H_bc`` or :class:`CenteredOperands`
``A``, ``M`` and ``c`` with ``H = A + c M``. :class:`BoundaryChoice` is
generic in the transport its ``bind`` accepts: every family accepts centered
operands, and only :class:`MultiplicativeChoice` also accepts absolute
``H_bc``, so a ``BoundaryChoice[CenteredOperands]`` is any family. The
transport parameter is erased at runtime, so the recipe input helpers that
identify a prepared dataset's representation admit absolute transport only
for :class:`MultiplicativeChoice`; a new family that accepts absolute
transport must also be admitted there. Centered transport keeps small
anomalies after cancellation: likelihoods use observations shifted by the
fixed ``c M 1``, while absolute outputs add it back. Centered graphs need a
float64 PyTensor process (:func:`require_centered_precision`), checked once
when centered transport is bound.

Validation happens where data enters: preparation and artifact loading own
units, labels and layout; binding owns calendar and row alignment; ``add``
checks values that only become available once lazy payloads are materialized.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

import xarray as xr

from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models.boundary.additive import AdditiveBoundary, AdditiveChoice
from openghg_inversions.models.boundary.multiplicative import MultiplicativeBoundary, MultiplicativeChoice
from openghg_inversions.models.boundary.operands import (
    CenteredOperands,
    centered_boundary_expression,
    require_centered_precision,
)
from openghg_inversions.models.boundary.outputs import BoundaryOutputs
from openghg_inversions.models.components import AffineConcentrationTerm


TransportT_contra = TypeVar("TransportT_contra", contravariant=True)


@runtime_checkable
class Boundary(Protocol):
    """A boundary choice bound to one fit's borrowed labelled transport.

    Implementations are frozen dataclasses whose ``data`` holds every
    observation-aligned array, so recipes can pad or subset rows with
    ``dataclasses.replace`` without knowing the family.
    """

    __dataclass_fields__: ClassVar[dict[str, Any]]

    @property
    def data(self) -> xr.Dataset: ...

    def add(
        self, *, output_dim: str = ..., prefix: str = ..., owner: str | None = ...,
    ) -> AffineConcentrationTerm: ...

    def outputs(self, prefix: str = ...) -> BoundaryOutputs: ...


@runtime_checkable
class BoundaryChoice(Protocol[TransportT_contra]):
    """Prior specification for one boundary family, before binding.

    The type parameter is the transport representation ``bind`` accepts.
    It is contravariant, so a choice accepting ``xr.DataArray |
    CenteredOperands`` is also a ``BoundaryChoice[CenteredOperands]``.
    """

    __dataclass_fields__: ClassVar[dict[str, Any]]

    @property
    def frequency(self) -> str | None: ...

    @property
    def anchor_time(self) -> DatetimeLike | None: ...

    def bind(self, observations: xr.DataArray, transport: TransportT_contra) -> Boundary: ...


__all__ = [
    "AdditiveBoundary",
    "AdditiveChoice",
    "Boundary",
    "BoundaryChoice",
    "BoundaryOutputs",
    "CenteredOperands",
    "MultiplicativeBoundary",
    "MultiplicativeChoice",
    "centered_boundary_expression",
    "require_centered_precision",
]
