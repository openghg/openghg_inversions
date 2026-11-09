"""Multiplicative boundary scaling, from absolute or centered sensitivity.

The boundary contribution is ``H b`` for dimensionless scaling states ``b``.
:meth:`MultiplicativeChoice.bind` returns one bound record per sensitivity
representation:

- :class:`MultiplicativeBoundary` holds absolute sensitivity ``H``
  (``H_bc``) and adds ``H b`` as an ordinary linear component.
- :class:`CenteredMultiplicativeBoundary` holds centered sensitivity
  ``H = A + c M`` (:class:`~.contracts.BoundCenteredSensitivity`; see
  :doc:`/usage/centered_boundary_conditions`). Like the other centered
  families it writes the contribution as ``A 1 + c M 1 + (A + c M)(b - 1)``,
  and the likelihood omits the fixed ``c M 1``, which keeps small anomalies
  after cancellation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytensor.tensor as pt
import xarray as xr
from dask import is_dask_collection

from openghg_inversions.boundary_sensitivity import bind_boundary_sensitivity, validate_centered_boundary_periods
from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models.components import (
    AffineBlock,
    AffineConcentrationTerm,
    StateVectorResult,
    add_linear_component,
    add_state_vector,
)
from openghg_inversions.models.state_activity import StateActivity, prepare_linear_sensitivity, resolve_state_activity

from .centered import CenteredBoundary, Correction, MeanShift
from .contracts import BoundCenteredSensitivity, RawCenteredSensitivity
from .outputs import Outputs

def centered_boundary_expression(anomaly, mass, reference, state):
    """Evaluate ``A b + c M (b - 1)`` without subtracting a large reference.

    Args:
        anomaly: Concentration anomaly sensitivity A, on observations by states.
        mass: Dimensionless mass M on the same axes.
        reference: Scalar concentration reference c.
        state: Dimensionless physical boundary scaling vector b.

    Returns:
        Centered concentration, using NumPy or PyTensor operands supplied by
        the caller. Alignment and materialization belong to the caller.
    """
    return anomaly @ state + reference * (mass @ (state - 1.0))


@dataclass(frozen=True)
class MultiplicativeChoice:
    """Prior specification for multiplicative boundary scaling, before binding.

    A choice holds the scientific options (scaling prior, state activity and
    boundary period calendar) without bound observations or sensitivity arrays.
    :meth:`bind` pairs it with one fit's observations and sensitivity.

    Attributes:
        prior: Prior for the dimensionless scaling states ``bc``, as passed to
            :func:`~openghg_inversions.models.components.add_state_vector`.
            Labelled parameters remain borrowed and are materialized with the
            bound record.
        activity: Active/fixed state policy. ``None`` makes every state with
            nonzero sensitivity active and fixes structurally zero states at
            one.
        frequency: Boundary period frequency, with the syntax and calendar of
            ``bc_freq``: ``None`` for one period over the window,
            ``"monthly"`` for calendar months, or a positive fixed-duration
            string such as ``"7D"``.
        anchor_time: Origin of fixed-duration periods; only valid with a
            fixed-duration ``frequency``. ``None`` uses the earliest selected
            observation of the fit. The RHIME parameter adapter instead
            defaults ``bc_freq_anchor`` to the configured start date.

    Raises:
        ValueError: On construction, for an invalid frequency or anchor.
    """

    prior: Mapping[str, Any]
    activity: StateActivity | None = None
    frequency: str | None = None
    anchor_time: DatetimeLike | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prior", dict(self.prior))
        validate_centered_boundary_periods(self.frequency, self.anchor_time)

    @property
    def mean_shift(self) -> MeanShift:
        """No mean shift: scaling the whole sensitivity already moves the boundary mean."""
        return MeanShift(None)

    def bind(
        self, observations: xr.DataArray, sensitivity: xr.DataArray | RawCenteredSensitivity,
    ) -> MultiplicativeBoundary | CenteredMultiplicativeBoundary:
        """Bind prepared absolute or centered sensitivity to one fit's rows and periods.

        Args:
            observations: One channel's observations with its observation axis
                first. Its labels select and order the sensitivity rows; its
                ``time`` values assign rows to boundary periods.
            sensitivity: Absolute ``H_bc`` in concentration units, or raw
                centered sensitivity, on the raw ``nmeasure`` or ``time`` axis.
                A Dataset is centered sensitivity; a DataArray is absolute. Absolute
                sensitivity already expanded to periods is accepted only without
                a ``frequency`` or ``anchor_time``.

        Returns:
            :class:`MultiplicativeBoundary` holding ``H_bc``, or
            :class:`CenteredMultiplicativeBoundary` holding ``H_bc_anomaly``,
            ``H_bc_mass`` and the scalar ``bc_reference``, on
            ``(observation, state)``. Raw spatial sensitivity and any explicit
            calendar give a ``bc_region`` state index with
            ``(bc_curtain, bc_period)`` levels; unmarked absolute sensitivity
            without a calendar keeps its own state labels. Numerical payloads
            stay borrowed and lazy.

        Raises:
            ValueError: If centered sensitivity is bound without float64
                PyTensor, periods or indexed timestamps conflict with the
                observations, or a calendar override is requested for
                already-expanded absolute sensitivity.
        """
        if isinstance(sensitivity, xr.Dataset):
            return CenteredMultiplicativeBoundary(self, BoundCenteredSensitivity.bind(
                sensitivity, observations, frequency=self.frequency, anchor_time=self.anchor_time,
            ))
        absolute = bind_boundary_sensitivity(
            sensitivity, observations, frequency=self.frequency, anchor_time=self.anchor_time,
        )
        return MultiplicativeBoundary(self, absolute.rename("H_bc").to_dataset())


def _add_scaling(choice: MultiplicativeChoice, sensitivity: xr.DataArray, *, output_dim: str, prefix: str,
                 owner: str | None) -> StateVectorResult:
    """Add the scaling states ``bc`` for the absolute sensitivity ``H``.

    Exactly-zero columns of ``H`` are fixed at one unless the activity policy
    says otherwise, as for any linear component.
    """
    activity = resolve_state_activity(prepare_linear_sensitivity(sensitivity, output_dim=output_dim).removed, choice.activity)
    return add_state_vector(activity, prior_args=dict(choice.prior), var_name=f"{prefix}bc", coordinate_owner=owner)


def _multiplicative_roles() -> tuple[dict[str, str], dict[str, str]]:
    """Roles and units of the scaling states, shared by both representations."""
    units = {name: "1" for name in ("bc", "bc_active", "bc_fixed_value", "bc_latent", "bc_active_latent")}
    return {"boundary_scale": "bc"}, units


@dataclass(frozen=True)
class MultiplicativeBoundary:
    """Multiplicative choice bound to one fit's borrowed absolute sensitivity.

    Attributes:
        choice: The resolved multiplicative choice.
        data: Absolute ``H_bc`` (concentration units) on ``(observation,
            state)``. It keeps its own supported state layouts and has no
            contract. The array is borrowed and may be Dask-backed.
            Materialize it jointly with the other model inputs and the choice
            prior before :meth:`add`; rows may be padded or subset with
            ``dataclasses.replace``, keeping that layout.

    :meth:`MultiplicativeChoice.bind` establishes the layout by construction,
    so the record does not validate it again.
    """

    choice: MultiplicativeChoice
    data: xr.Dataset

    @property
    def observation_dim(self) -> str:
        """Observation dimension of the bound sensitivity, its first dimension."""
        return str(self.data.H_bc.dims[0])

    def add(self, *, output_dim: str = "nmeasure", prefix: str = "", owner: str | None = "bc") -> AffineConcentrationTerm:
        """Add the scaling states and the boundary concentration ``H b`` to the current model.

        Must be called inside an active PyMC model that has the coordinate
        registry, after ``data`` has been materialized jointly with the other
        model inputs. Like any linear component it registers the retained
        columns of ``H`` as ``hbc``, the scaling state vector ``bc`` and the
        deterministic ``mu_bc``.

        Args:
            output_dim: Observation dimension registered by the model.
            prefix: Prefix for graph variable names.
            owner: Namespace for component-owned boundary-state coordinates;
                ``None`` keeps the local names. Graph names are controlled
                separately by ``prefix``.

        Returns:
            The boundary term, entering the likelihood absolutely: the fixed
            contribution is ``H`` over inactive states times their fixed
            values; when active states exist, the one affine block is ``H``
            over those states, with the state vector's latent as its sampled
            variable.

        Raises:
            ValueError: If ``output_dim`` is not the bound observation
                dimension, ``data`` is still lazy, or the activity policy
                cannot be resolved against the states.
            RuntimeError: If the model lacks the coordinate registry.
        """
        if output_dim != self.observation_dim:
            raise ValueError(f"output_dim {output_dim!r} is not the bound observation dimension {self.observation_dim!r}.")
        if is_dask_collection(self.data.H_bc):
            raise ValueError("Materialize boundary data jointly with model inputs before add().")
        sensitivity = self.data.H_bc
        component = add_linear_component(
            prepare_linear_sensitivity(sensitivity, output_dim=output_dim),
            data_name=f"{prefix}hbc", prior_args=dict(self.choice.prior), var_name=f"{prefix}bc",
            output_name=f"{prefix}mu_bc", state_activity=self.choice.activity, output_dim=output_dim,
            coordinate_owner=owner,
        )
        activity = component.activity
        state_dim, active, inactive = activity.state_dim, activity.active_indices, activity.fixed_indices
        fixed = xr.dot(sensitivity.isel({state_dim: inactive}), activity.fixed_value.isel({state_dim: inactive}), dim=state_dim)
        return AffineConcentrationTerm.absolute(
            component.output,
            fixed_contribution=fixed.assign_attrs(sensitivity.attrs),
            blocks=(AffineBlock(sensitivity.isel({state_dim: active}), component.state[active]),) if active.size else (),
            sampled_rvs=(component.latent,) if component.latent is not None else (),
        )

    def outputs(self, prefix: str = "") -> Outputs:
        """Declare the roles, units and metadata of this boundary's graph.

        Declarations depend only on the choice and the bound data, so they
        are available before :meth:`add` or sampling.

        Args:
            prefix: Graph-name prefix used for :meth:`add`.

        Returns:
            Scientific roles mapped to graph variable names, graph variable
            units, and configuration and binding metadata. ``prefix`` is
            applied to both role keys and graph names; metadata is unprefixed.
        """
        roles, units = _multiplicative_roles()
        roles.update(boundary_concentration="mu_bc", boundary_sensitivity="hbc")
        concentration_units = self.data.H_bc.attrs.get("units")
        if concentration_units is not None:
            units.update(mu_bc=concentration_units, hbc=concentration_units)
        metadata = {
            "boundary_formulation": "multiplicative",
            "bc_frequency": self.choice.frequency,
            "bc_anchor_time": None if self.choice.anchor_time is None else str(self.choice.anchor_time),
        }
        return Outputs(roles, units, metadata).prefixed(prefix)


@dataclass(frozen=True)
class CenteredMultiplicativeBoundary(CenteredBoundary):
    """Multiplicative choice bound to one fit's borrowed centered sensitivity.

    Attributes:
        choice: The resolved multiplicative choice.
        data: :class:`~.contracts.BoundCenteredSensitivity`: ``H_bc_anomaly``
            ``A`` (concentration units) and ``H_bc_mass`` ``M``
            (dimensionless) on ``(observation, bc_region)`` and scalar
            ``bc_reference`` ``c``, so the absolute sensitivity is
            ``A + c M``. The arrays are borrowed and may be Dask-backed.
            Materialize them jointly with the other model inputs and the
            choice prior before :meth:`add`; rows may be padded or subset
            with ``dataclasses.replace``, keeping that layout.

    :meth:`MultiplicativeChoice.bind` establishes ``data`` by construction, so
    the record trusts that contract rather than validating it again.
    """

    choice: MultiplicativeChoice
    data: BoundCenteredSensitivity

    def add(self, *, output_dim: str = "nmeasure", prefix: str = "", owner: str | None = "bc") -> AffineConcentrationTerm:
        """Add the scaling states and the boundary concentration to the current model.

        Must be called inside an active PyMC model that has the coordinate
        registry, after ``data`` has been materialized jointly with the other
        model inputs. It registers the sensitivity and reference as model
        data, plus the scaling state vector ``bc`` and the deterministic
        ``mu_bc``.

        Args:
            output_dim: Observation dimension registered by the model.
            prefix: Prefix for graph variable names.
            owner: Namespace for component-owned boundary-state coordinates;
                ``None`` keeps the local names. Graph names are controlled
                separately by ``prefix``.

        Returns:
            The boundary term: ``output`` is the absolute ``mu_bc``; the
            likelihood contribution omits the fixed ``c M 1``, which is the
            reference shift; the fixed contribution is ``A 1`` plus
            ``(A + c M)`` over inactive states times their fixed values minus
            one; when active states exist, the one affine block is
            ``A + c M`` over those states with coefficients ``b - 1``, and the
            state vector's latent is the sampled variable.

        Raises:
            ValueError: If ``output_dim`` is not the bound observation
                dimension, ``data`` is still lazy, materialized operands are
                invalid, or the activity policy cannot be resolved against the
                states.
            RuntimeError: If the model lacks the coordinate registry.
        """
        registered = self._register_sensitivity(output_dim=output_dim, prefix=prefix, owner=owner)
        data = self.data
        sensitivity = (data.H_bc_anomaly + data.bc_reference * data.H_bc_mass).assign_attrs(data.H_bc_anomaly.attrs)
        vector = _add_scaling(self.choice, sensitivity, output_dim=output_dim, prefix=prefix, owner=owner)
        activity = vector.activity
        active, inactive = activity.active_indices, activity.fixed_indices

        # Response (A + c M)(b - 1) of the scaling, evaluated without forming A + c M.
        deviation = vector.state - 1.0
        signal = pt.dot(registered.anomaly, deviation) + registered.reference * pt.dot(registered.mass, deviation)
        inactive_values = activity.fixed_value.isel(bc_region=inactive) - 1
        fixed = registered.fixed + xr.dot(sensitivity.isel(bc_region=inactive), inactive_values, dim="bc_region")
        corrections = []
        if vector.latent is not None:
            block = AffineBlock(sensitivity.isel(bc_region=active), deviation[active])
            corrections.append(Correction(deviation, signal, block, vector.latent))
        return self._term(registered, corrections, signal, fixed=fixed.assign_attrs(sensitivity.attrs))

    def outputs(self, prefix: str = "") -> Outputs:
        """Declare the roles, units and metadata of this boundary's graph.

        Declarations depend only on the choice and the bound data, so they
        are available before :meth:`add` or sampling.

        Args:
            prefix: Graph-name prefix used for :meth:`add`.

        Returns:
            Scientific roles mapped to graph variable names, graph variable
            units, and configuration and binding metadata. ``prefix`` is
            applied to both role keys and graph names; metadata is unprefixed.
        """
        roles, units = _multiplicative_roles()
        return self._outputs("multiplicative", prefix, roles=roles, units=units, metadata={})


__all__ = [
    "CenteredMultiplicativeBoundary", "MultiplicativeBoundary", "MultiplicativeChoice",
    "centered_boundary_expression",
]
