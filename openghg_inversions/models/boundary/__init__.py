"""Boundary-condition components for RHIME recipes.

Names here are short because the package supplies the context: a
:class:`Choice` is a boundary choice and :class:`Outputs` are boundary
outputs. Runners and builders conventionally name their parameter
``boundary``, which shadows the package, so they import :class:`Choice` and
:class:`Bound` directly (``boundary: Bound | None``).

The package has four layers, followed in this order by one boundary from
configuration to a saved trace:

.. list-table::
   :header-rows: 1

   * - Layer
     - Records
     - Modules
   * - Sensitivity contracts: what the arrays guarantee
     - :class:`RawCenteredSensitivity`, :class:`BoundCenteredSensitivity`,
       :class:`BoundAdditiveSensitivity` (static types on ``xr.Dataset``)
     - ``contracts``; binding uses
       :func:`~openghg_inversions.boundary_sensitivity.bind_period_states`
   * - Families: the scientific prior
     - a *choice* and a *bound* record per family, typed by the
       :class:`Choice` and :class:`Bound` protocols
     - ``multiplicative``, ``additive``, ``anomaly_scaling``
   * - Shared centered pieces: the term inside the model
     - :class:`~.centered.CenteredBoundary`, its
       :class:`~.centered.RegisteredSensitivity`, :class:`~.centered.MeanShift`,
       :class:`~.centered.Correction`, and the additive
       :class:`~.additive_anomalies.Anomalies`
     - ``centered``, ``additive_anomalies``
   * - Handoffs: what recipes and outputs receive
     - :class:`~openghg_inversions.models.components.AffineConcentrationTerm`
       from ``add``, :class:`Outputs` from ``outputs``
     - ``openghg_inversions.models.components``, ``outputs``

A *choice* (:class:`MultiplicativeChoice`, :class:`AdditiveChoice`,
:class:`AnomalyScalingChoice`) is one resolved selection of the user's
boundary options (priors, activity, anomaly structure and scale, reference
and the boundary period calendar) before any arrays are attached.
Configuration adapters produce choices. ``choice.bind(observations,
sensitivity)`` returns the *bound* record (:class:`MultiplicativeBoundary` or
:class:`CenteredMultiplicativeBoundary`, :class:`AdditiveBoundary`,
:class:`AnomalyScalingBoundary`), which pairs the
choice with borrowed labelled sensitivity aligned to one fit's observation
rows and expanded to its boundary periods.

A recipe binds each boundary once, materializes the bound record together
with its other inputs (``dask.compute`` traverses these dataclasses, including
choice priors; see :func:`~openghg_inversions.array_ops.to_dense_tree`), then calls
``boundary.add(...)`` inside its registered model. ``add`` is the affine
counterpart of :func:`~openghg_inversions.models.components.add_linear_component`:
it registers the sensitivity, adds the sampled corrections and returns an
:class:`~openghg_inversions.models.components.AffineConcentrationTerm`, which
ordinary and cached likelihoods consume without knowing the family, and
``boundary.outputs()`` declares the roles, units and metadata recorded with
the trace. :class:`Choice` and :class:`Bound` are structural protocols, so
adding a centered boundary family means adding one module with a choice and
a bound record inheriting :class:`~.centered.CenteredBoundary`, plus its
configuration adapter and tests; nothing here enumerates the families.

Sensitivity is either absolute ``H_bc`` or centered ``A``, ``M`` and ``c``
with ``H = A + c M``. Preparation and loading establish
:class:`RawCenteredSensitivity`; :meth:`BoundCenteredSensitivity.bind`
produces :class:`BoundCenteredSensitivity`, which every centered family
consumes; and weighted additive anomalies extend it to
:class:`BoundAdditiveSensitivity`. Every :class:`Choice` binds raw centered
sensitivity; only :class:`MultiplicativeChoice` also accepts absolute
``H_bc``, binding it to :class:`MultiplicativeBoundary`, the one bound
record that does not inherit :class:`~.centered.CenteredBoundary`. Centered
sensitivity keeps small anomalies after cancellation: likelihoods use
observations shifted by the fixed ``c M 1``, while absolute outputs add it
back. Centered graphs need a float64 PyTensor process
(:func:`require_centered_precision`), checked once when centered sensitivity
is bound.

Validation happens where data enters: preparation and artifact loading
establish the raw contract's units, labels and layout; binding owns calendar
and row alignment and establishes the bound contract by construction; ``add``
checks values that only become available once lazy payloads are materialized.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

import xarray as xr

from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models.boundary.additive import AdditiveBoundary, AdditiveChoice
from openghg_inversions.models.boundary.anomaly_scaling import AnomalyScalingBoundary, AnomalyScalingChoice
from openghg_inversions.models.boundary.multiplicative import (
    CenteredMultiplicativeBoundary,
    MultiplicativeBoundary,
    MultiplicativeChoice,
    centered_boundary_expression,
)
from openghg_inversions.models.boundary.contracts import (
    BoundAdditiveSensitivity,
    BoundCenteredSensitivity,
    RawCenteredSensitivity,
    normalize_centering_weights,
    require_centered_precision,
    validate_centering_weights,
)
from openghg_inversions.models.boundary.outputs import Outputs
from openghg_inversions.models.components import AffineConcentrationTerm


@runtime_checkable
class Bound(Protocol):
    """A boundary choice bound to one fit's borrowed labelled sensitivity.

    Implementations are frozen dataclasses whose ``data`` holds every
    observation-aligned array, so recipes can pad or subset rows with
    ``dataclasses.replace`` without knowing the family. Shared guarantees:
    every observation-aligned variable has :attr:`observation_dim` as its
    first dimension, ordered by the bound observation labels; variables
    without it describe states or scalars and must not be treated as rows.
    Centered families hold :class:`BoundCenteredSensitivity`; absolute
    multiplicative sensitivity has no contract. Binding establishes the data;
    records and recipes trust it and preserve it when replacing rows.
    """

    __dataclass_fields__: ClassVar[dict[str, Any]]

    @property
    def data(self) -> xr.Dataset:
        """Borrowed sensitivity bound to this fit's observation rows and states."""
        ...

    @property
    def observation_dim(self) -> str:
        """Observation dimension leading every observation-aligned variable."""
        ...

    def add(
        self, *, output_dim: str = ..., prefix: str = ..., owner: str | None = ...,
    ) -> AffineConcentrationTerm:
        """Add the boundary's states and concentration to the current PyMC model.

        Args:
            output_dim: Observation dimension registered by the model.
            prefix: Prefix for graph variable names.
            owner: Namespace for component-owned state coordinates, or ``None``
                for local names.

        Returns:
            The boundary's concentration term.
        """
        ...

    def outputs(self, prefix: str = ...) -> Outputs:
        """Declare roles, units and metadata of the graph ``add`` builds.

        Args:
            prefix: Graph-name prefix used for ``add``.

        Returns:
            The declarations, available before sampling.
        """
        ...


@runtime_checkable
class Choice(Protocol):
    """One resolved selection of a boundary family's options, before binding."""

    __dataclass_fields__: ClassVar[dict[str, Any]]

    @property
    def frequency(self) -> str | None:
        """Boundary-period frequency (``"monthly"`` or a fixed duration), or ``None`` for one period."""
        ...

    @property
    def anchor_time(self) -> DatetimeLike | None:
        """Optional start of the fixed-duration period grid."""
        ...

    def bind(self, observations: xr.DataArray, sensitivity: RawCenteredSensitivity) -> Bound:
        """Bind prepared sensitivity to one fit's observations and periods.

        Args:
            observations: Observations of the fit, supplying row labels and
                times.
            sensitivity: Raw centered sensitivity; only
                :class:`MultiplicativeChoice` also accepts absolute ``H_bc``.

        Returns:
            The bound boundary.

        Raises:
            ValueError: If the sensitivity or period settings are invalid.
        """
        ...


__all__ = [
    "AdditiveBoundary",
    "AdditiveChoice",
    "AnomalyScalingBoundary",
    "AnomalyScalingChoice",
    "BoundAdditiveSensitivity",
    "Bound",
    "BoundCenteredSensitivity",
    "CenteredMultiplicativeBoundary",
    "Choice",
    "Outputs",
    "MultiplicativeBoundary",
    "MultiplicativeChoice",
    "RawCenteredSensitivity",
    "centered_boundary_expression",
    "normalize_centering_weights",
    "require_centered_precision",
    "validate_centering_weights",
]
