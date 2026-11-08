"""Multiplicative boundary scaling, from absolute or centered transport.

The boundary contribution is ``H b`` for dimensionless scaling states ``b``.
With absolute transport ``H`` (``H_bc``) the graph evaluates that product
directly. With centered transport ``H = A + c M``
(:class:`~.contracts.RawCenteredTransport`; see
:doc:`/usage/centered_boundary_conditions`) the likelihood instead uses
``A b + c M (b - 1)`` against observations shifted by ``c M 1``, which keeps
small anomalies after cancellation; the absolute output still adds the shift.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from dask import is_dask_collection

from openghg_inversions.boundary_sensitivity import bind_boundary_sensitivity, validate_centered_boundary_periods
from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models.components import AffineBlock, AffineConcentrationTerm, add_state_vector
from openghg_inversions.models.coords import add_model_data, component_coords
from openghg_inversions.models.state_activity import StateActivity, prepare_linear_sensitivity, resolve_state_activity

from .contracts import BoundCenteredTransport, RawCenteredTransport
from .operands import centered_boundary_expression
from .outputs import BoundaryOutputs


@dataclass(frozen=True)
class MultiplicativeChoice:
    """Prior specification for multiplicative boundary scaling, before binding.

    A choice holds the scientific options (scaling prior, state activity and
    boundary period calendar) without bound observations or transport arrays.
    :meth:`bind` pairs it with one fit's observations and transport.

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

    def bind(
        self, observations: xr.DataArray, transport: xr.DataArray | RawCenteredTransport,
    ) -> MultiplicativeBoundary:
        """Bind prepared absolute or centered transport to one fit's rows and periods.

        Args:
            observations: One channel's observations with its observation axis
                first. Its labels select and order the transport rows; its
                ``time`` values assign rows to boundary periods.
            transport: Absolute ``H_bc`` in concentration units, or raw
                centered transport, on the raw ``nmeasure`` or ``time`` axis.
                A Dataset is centered transport; a DataArray is absolute. Absolute
                transport already expanded to periods is accepted only without
                a ``frequency`` or ``anchor_time``.

        Returns:
            The bound boundary. ``data`` holds ``H_bc``, or ``H_bc_anomaly``,
            ``H_bc_mass`` and the scalar ``bc_reference``, on
            ``(observation, state)``. Raw spatial transport and any explicit
            calendar give a ``bc_region`` state index with
            ``(bc_curtain, bc_period)`` levels; unmarked absolute transport
            without a calendar keeps its own state labels. Numerical payloads
            stay borrowed and lazy.

        Raises:
            ValueError: If centered transport is bound without float64
                PyTensor, periods or indexed timestamps conflict with the
                observations, or a calendar override is requested for
                already-expanded absolute transport.
        """
        data: xr.Dataset
        if isinstance(transport, xr.Dataset):
            data = BoundCenteredTransport.bind(
                transport, observations, frequency=self.frequency, anchor_time=self.anchor_time,
            )
        else:
            data = bind_boundary_sensitivity(
                transport, observations, frequency=self.frequency, anchor_time=self.anchor_time,
            ).rename("H_bc").to_dataset()
        return MultiplicativeBoundary(self, data)


@dataclass(frozen=True)
class MultiplicativeBoundary:
    """Multiplicative choice bound to one fit's borrowed labelled transport.

    Attributes:
        choice: The resolved multiplicative choice.
        data: :class:`~.contracts.BoundCenteredTransport`, or absolute
            ``H_bc`` (concentration units) on ``(observation, state)``.
            Absolute transport keeps its own supported state layouts and has
            no contract. The arrays are borrowed and may be Dask-backed.
            Materialize them jointly with the other model inputs and the
            choice prior before :meth:`add`; rows may be padded or subset
            with ``dataclasses.replace``, keeping that layout.

    :meth:`MultiplicativeChoice.bind` establishes either layout by
    construction, so the record does not validate it again.
    """

    choice: MultiplicativeChoice
    data: BoundCenteredTransport | xr.Dataset

    @property
    def observation_dim(self) -> str:
        """Observation dimension of the bound transport, the first transport dimension."""
        return str(self.data["H_bc_anomaly" if self.centered else "H_bc"].dims[0])

    @property
    def centered(self) -> bool:
        """Whether the bound transport uses the centered representation."""
        return "H_bc_anomaly" in self.data

    def add(self, *, output_dim: str = "nmeasure", prefix: str = "", owner: str | None = "bc") -> AffineConcentrationTerm:
        """Add the scaling states and the boundary concentration to the current model.

        Must be called inside an active PyMC model that has the coordinate
        registry, after ``data`` has been materialized jointly with the other
        model inputs. It registers the transport (``hbc``, or
        ``H_bc_anomaly``, ``H_bc_mass`` and ``bc_reference``) as model data,
        plus the scaling state vector ``bc`` and the deterministic ``mu_bc``.

        The affine representation is ``fixed + D_active (b_active - k)``,
        where ``D`` is the absolute design (``A + cM`` when centered),
        ``k = 1`` for centered and ``0`` for absolute transport, and the fixed
        part retains inactive states and, when centered, ``A 1``.

        Args:
            output_dim: Observation dimension registered by the model.
            prefix: Prefix for graph variable names.
            owner: Namespace for component-owned boundary-state coordinates;
                ``None`` keeps the local names. Graph names are controlled
                separately by ``prefix``.

        Returns:
            The boundary term: ``output`` is the absolute boundary
            concentration ``mu_bc``; when centered, the likelihood
            contribution omits the reference shift ``c M 1``; the fixed
            contribution follows the representation above. When active states
            exist, the term includes one affine block over those states and
            the state vector's latent as its sampled variable. Otherwise it
            has no affine blocks or sampled variables.

        Raises:
            ValueError: If ``output_dim`` is not the bound observation
                dimension, ``data`` is still lazy, materialized centered
                operands are non-finite or have invalid mass, the reference
                shift is non-finite, or the activity policy cannot be resolved
                against the states.
            RuntimeError: If the model lacks the coordinate registry.
        """
        if output_dim != self.observation_dim:
            raise ValueError(f"output_dim {output_dim!r} is not the bound observation dimension {self.observation_dim!r}.")
        if any(is_dask_collection(value) for value in self.data.data_vars.values()):
            raise ValueError("Materialize boundary data jointly with model inputs before add().")

        shift: xr.DataArray | None = None
        if self.centered:
            anomaly = self.data.H_bc_anomaly
            mass = self.data.H_bc_mass
            reference = self.data.bc_reference
            shift = BoundCenteredTransport.reference_shift(self.data).rename(f"{prefix}bc_reference_shift")
            design = (anomaly + reference * mass).assign_attrs(anomaly.attrs)
        else:
            design = self.data.H_bc
        prepared = prepare_linear_sensitivity(design, output_dim=output_dim)
        activity = resolve_state_activity(prepared.removed, self.choice.activity)
        vector = add_state_vector(activity, prior_args=dict(self.choice.prior), var_name=f"{prefix}bc", coordinate_owner=owner)
        state_dim = activity.state_dim
        active, inactive = activity.active_indices, activity.fixed_indices
        inactive_design = design.isel({state_dim: inactive})
        inactive_values = activity.fixed_value.isel({state_dim: inactive})
        if shift is not None:
            a = add_model_data(component_coords(anomaly, owner, observation_dim=output_dim), f"{prefix}H_bc_anomaly")
            m = add_model_data(component_coords(mass, owner, observation_dim=output_dim), f"{prefix}H_bc_mass")
            c = add_model_data(reference, f"{prefix}bc_reference")
            likelihood = centered_boundary_expression(a, m, c, vector.state)
            output = pm.Deterministic(f"{prefix}mu_bc", likelihood + pt.as_tensor_variable(shift.values), dims=output_dim)
            fixed = anomaly.sum(state_dim, skipna=False) + xr.dot(inactive_design, inactive_values - 1, dim=state_dim)
            coefficients = vector.state[active] - 1
        else:
            h = add_model_data(component_coords(design, owner, observation_dim=output_dim), f"{prefix}hbc")
            output = pm.Deterministic(f"{prefix}mu_bc", pt.dot(h, vector.state), dims=output_dim)
            likelihood = output
            fixed = xr.dot(inactive_design, inactive_values, dim=state_dim)
            coefficients = vector.state[active]
        return AffineConcentrationTerm(
            output, likelihood, shift,
            fixed_contribution=fixed.assign_attrs(design.attrs),
            blocks=(AffineBlock(design.isel({state_dim: active}), coefficients),) if active.size else (),
            sampled_rvs=(vector.latent,) if vector.latent is not None else (),
        )

    def outputs(self, prefix: str = "") -> BoundaryOutputs:
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
        roles = {"boundary_concentration": "mu_bc", "boundary_scale": "bc"}
        units = {name: "1" for name in ("bc", "bc_active", "bc_fixed_value", "bc_latent", "bc_active_latent")}
        metadata = {
            "boundary_formulation": "multiplicative",
            "bc_frequency": self.choice.frequency,
            "bc_anchor_time": None if self.choice.anchor_time is None else str(self.choice.anchor_time),
        }
        if self.centered:
            concentration = self.data.bc_reference.attrs["units"]
            roles.update(boundary_anomaly="H_bc_anomaly", boundary_mass="H_bc_mass", boundary_reference="bc_reference")
            units.update({name: concentration for name in (
                "mu_bc", "H_bc_anomaly", "bc_reference",
            )})
            units["H_bc_mass"] = "1"
            metadata["boundary_representation"] = "centered-v1"
        else:
            roles["boundary_sensitivity"] = "hbc"
            concentration = self.data.H_bc.attrs.get("units")
            if concentration is not None:
                units.update(mu_bc=concentration, hbc=concentration)
        return BoundaryOutputs(roles, units, metadata).prefixed(prefix)


__all__ = ["MultiplicativeBoundary", "MultiplicativeChoice"]
