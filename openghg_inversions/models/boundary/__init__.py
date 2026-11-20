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
and the boundary period calendar) before any arrays are attached. Each choice
class names its family in ``kind``, which configuration selects explicitly
with ``type`` and outputs record as ``boundary_formulation``.
Configuration adapters produce choices; :func:`bind_prepared` binds one to
the sensitivity representation that preparation produced. ``choice.bind(observations,
sensitivity)`` binds centered sensitivity, and an :class:`AbsoluteChoice`'s
``bind_absolute`` binds absolute ``H_bc``. Either returns the *bound* record
(:class:`CenteredMultiplicativeBoundary` or, from ``bind_absolute``,
:class:`MultiplicativeBoundary`; :class:`AdditiveBoundary`;
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
sensitivity; only :class:`MultiplicativeChoice` is also an
:class:`AbsoluteChoice`, whose ``bind_absolute`` binds absolute ``H_bc`` to
:class:`MultiplicativeBoundary`, the one bound record that does not inherit
:class:`~.centered.CenteredBoundary`. :func:`bind_prepared` is the one place
that checks a choice against the representation preparation produced; before
preparation, ``choice.preparation(representation)`` (or
:func:`boundary_preparation`, which also covers no boundary) states which
representation and prepared inputs a choice needs. Centered
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
from openghg_inversions.models.boundary.preparation import Preparation, boundary_preparation
from openghg_inversions.boundary_sensitivity import BoundaryRepresentation
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
    """One resolved selection of a boundary family's options, before binding.

    ``kind`` names the family, as selected by the configuration ``type`` and
    recorded as ``boundary_formulation``.
    """

    __dataclass_fields__: ClassVar[dict[str, Any]]
    kind: ClassVar[str]

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
            sensitivity: Raw centered sensitivity. Absolute ``H_bc`` is bound
                by :meth:`AbsoluteChoice.bind_absolute` instead.

        Returns:
            The bound boundary.

        Raises:
            ValueError: If the sensitivity or period settings are invalid.
        """
        ...

    def preparation(self, representation: BoundaryRepresentation | None = None) -> Preparation:
        """Return the prepared inputs this choice needs.

        Args:
            representation: Requested boundary representation, or ``None`` for
                the family's default.

        Returns:
            The representation and prepared inputs this choice binds.

        Raises:
            ValueError: If the family cannot bind the requested representation.
        """
        ...


@runtime_checkable
class AbsoluteChoice(Choice, Protocol):
    """A choice that can also bind absolute sensitivity ``H_bc``.

    Only :class:`MultiplicativeChoice` implements it; recipes reach it through
    :func:`bind_prepared` rather than testing the family themselves.
    """

    def bind_absolute(self, observations: xr.DataArray, sensitivity: xr.DataArray) -> Bound:
        """Bind prepared absolute sensitivity to one fit's observations and periods.

        Args:
            observations: Observations of the fit with their observation axis
                first, supplying row labels and times.
            sensitivity: Absolute ``H_bc`` in concentration units on the raw
                ``nmeasure`` or ``time`` axis, or already expanded to periods
                when the choice requests no calendar.

        Returns:
            The bound boundary. Numerical payloads stay borrowed and lazy.

        Raises:
            ValueError: If the sensitivity or period settings are invalid.
        """
        ...


def bind_prepared(
    choice: Choice,
    observations: xr.DataArray,
    *,
    centered: RawCenteredSensitivity | None = None,
    absolute: xr.DataArray | None = None,
    label: str = "boundary",
) -> Bound:
    """Bind a choice to the one sensitivity representation that preparation produced.

    Recipes pass whichever representation their prepared inputs record; this
    is the single check that the choice's family supports it.

    Args:
        choice: The resolved boundary choice.
        observations: One channel's observations, as for ``choice.bind``.
        centered: Raw centered sensitivity, when preparation produced it.
        absolute: Absolute ``H_bc``, when preparation produced only that.
        label: Name of the boundary in error messages, such as ``"o2 boundary"``.

    Returns:
        The bound boundary.

    Raises:
        ValueError: If preparation produced neither or both representations,
            or absolute sensitivity is given to a family that requires
            centered sensitivity.
    """
    if centered is None and absolute is None:
        raise ValueError(f"The {label} requires prepared H_bc or centered boundary operands.")
    if centered is not None and absolute is not None:
        raise ValueError(f"The {label} received both prepared H_bc and centered boundary operands; prepare one.")
    if centered is not None:
        return choice.bind(observations, centered)
    assert absolute is not None
    if not isinstance(choice, AbsoluteChoice):
        raise ValueError(
            f"The {choice.kind} {label} requires centered boundary operands; "
            "prepare again with boundary_representation='centered'."
        )
    return choice.bind_absolute(observations, absolute)


__all__ = [
    "AbsoluteChoice",
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
    "Preparation",
    "boundary_preparation",
    "MultiplicativeChoice",
    "RawCenteredSensitivity",
    "bind_prepared",
    "centered_boundary_expression",
    "normalize_centering_weights",
    "require_centered_precision",
    "validate_centering_weights",
]
