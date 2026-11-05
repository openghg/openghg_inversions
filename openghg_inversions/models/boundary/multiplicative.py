"""Multiplicative boundary scaling, from absolute or centered transport.

The boundary contribution is ``H b`` for dimensionless scaling states ``b``.
With absolute transport ``H`` (``H_bc``) the graph evaluates that product
directly. With centered transport ``H = A + c M`` the likelihood instead uses
``A b + c M (b - 1)`` against observations shifted by ``c M 1``, which keeps
small anomalies after cancellation; the absolute output still adds the shift.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from dask import is_dask_collection

from openghg_inversions.boundary_sensitivity import bind_boundary_sensitivity, validate_centered_boundary_periods
from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models.components import AffineBlock, AffineConcentrationTerm, add_state_vector
from openghg_inversions.models.coords import add_model_data, component_coords
from openghg_inversions.models.state_activity import StateActivity, prepare_linear_sensitivity, resolve_state_activity

from .operands import CenteredOperands, centered_boundary_expression
from .outputs import BoundaryOutputs


@dataclass(frozen=True)
class MultiplicativeChoice:
    """Data-free multiplicative boundary choice.

    Attributes:
        prior: Prior for the dimensionless scaling states ``bc``.
        activity: Optional active/fixed state policy.
        frequency: Optional boundary period frequency.
        anchor_time: Optional origin for fixed-duration periods.
    """

    prior: Mapping[str, Any]
    activity: StateActivity | None = None
    frequency: str | None = None
    anchor_time: DatetimeLike | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prior", dict(self.prior))
        validate_centered_boundary_periods(self.frequency, self.anchor_time)

    def bind(
        self, observations: xr.DataArray, transport: xr.DataArray | CenteredOperands,
    ) -> MultiplicativeBoundary:
        """Bind prepared absolute or centered transport to one fit's rows and periods."""
        if isinstance(transport, CenteredOperands):
            data = transport.bind(observations, frequency=self.frequency, anchor_time=self.anchor_time)
        else:
            data = bind_boundary_sensitivity(
                transport, observations, frequency=self.frequency, anchor_time=self.anchor_time,
            ).rename("H_bc").to_dataset()
        return MultiplicativeBoundary(self, data)


@dataclass(frozen=True)
class MultiplicativeBoundary:
    """Multiplicative choice bound to one fit's borrowed labelled transport.

    ``data`` holds ``H_bc``, or centered ``H_bc_anomaly``, ``H_bc_mass`` and
    ``bc_reference``, on ``(observation, state)``. Materialize it with the
    other model inputs before :meth:`add`.
    """

    choice: MultiplicativeChoice
    data: xr.Dataset

    @property
    def centered(self) -> bool:
        """Whether the bound transport uses the centered representation."""
        return "H_bc_anomaly" in self.data

    def add(self, *, output_dim: str = "nmeasure", prefix: str = "", owner: str | None = "bc") -> AffineConcentrationTerm:
        """Add the scaling states and the boundary concentration to the current model.

        The affine representation is ``fixed + D_active (b_active - k)``,
        where ``D`` is the absolute design (``A + cM`` when centered),
        ``k = 1`` for centered and ``0`` for absolute transport, and the fixed
        part retains inactive states and, when centered, ``A 1``.

        Args:
            output_dim: Observation dimension registered by the model.
            prefix: Prefix for graph variable names.
            owner: Namespace for labelled boundary-state coordinates.

        Raises:
            ValueError: If ``data`` is still lazy, or materialized centered
                operands are non-finite or have invalid mass.
        """
        if any(is_dask_collection(value) for value in self.data.data_vars.values()):
            raise ValueError("Materialize boundary data jointly with model inputs before add().")
        if self.centered:
            anomaly = self.data.H_bc_anomaly.transpose(output_dim, "bc_region")
            mass = self.data.H_bc_mass.transpose(output_dim, "bc_region")
            reference = self.data.bc_reference
            if not np.isfinite(reference.values).all():
                raise ValueError("Centered bc_reference must be finite.")
            if not np.isfinite(mass.values).all() or (mass.values < 0).any() or not (mass.values > 0).any():
                raise ValueError("H_bc_mass must be finite, non-negative and not all zero.")
            design = (anomaly + reference * mass).assign_attrs(anomaly.attrs)
        else:
            design = self.data.H_bc.transpose(output_dim, ...)
        prepared = prepare_linear_sensitivity(design, output_dim=output_dim)
        activity = resolve_state_activity(prepared.removed, self.choice.activity)
        vector = add_state_vector(activity, prior_args=dict(self.choice.prior), var_name=f"{prefix}bc", coordinate_owner=owner)
        state_dim = activity.state_dim
        active, inactive = activity.active_indices, activity.fixed_indices
        inactive_design = design.isel({state_dim: inactive})
        inactive_values = activity.fixed_value.isel({state_dim: inactive})
        if self.centered:
            a = add_model_data(component_coords(anomaly, owner, observation_dim=output_dim), f"{prefix}H_bc_anomaly")
            m = add_model_data(component_coords(mass, owner, observation_dim=output_dim), f"{prefix}H_bc_mass")
            c = add_model_data(reference, f"{prefix}bc_reference")
            shift = (reference * mass.sum(state_dim, skipna=False)).rename(
                f"{prefix}bc_reference_shift"
            ).assign_attrs(reference.attrs)
            if not np.isfinite(shift.values).all():
                raise ValueError("Boundary reference shift must be finite.")
            likelihood = centered_boundary_expression(a, m, c, vector.state)
            output = pm.Deterministic(f"{prefix}mu_bc", likelihood + pt.as_tensor_variable(shift.values), dims=output_dim)
            fixed = anomaly.sum(state_dim, skipna=False) + xr.dot(inactive_design, inactive_values - 1, dim=state_dim)
            coefficients = vector.state[active] - 1
        else:
            h = add_model_data(component_coords(design, owner, observation_dim=output_dim), f"{prefix}hbc")
            output = pm.Deterministic(f"{prefix}mu_bc", pt.dot(h, vector.state), dims=output_dim)
            likelihood, shift = output, None
            fixed = xr.dot(inactive_design, inactive_values, dim=state_dim)
            coefficients = vector.state[active]
        return AffineConcentrationTerm(
            output, likelihood, shift,
            fixed_contribution=fixed.assign_attrs(design.attrs),
            blocks=(AffineBlock(design.isel({state_dim: active}), coefficients),) if active.size else (),
            sampled_rvs=(vector.latent,) if vector.latent is not None else (),
        )

    def outputs(self, prefix: str = "") -> BoundaryOutputs:
        """Declare roles, units and metadata for the graph built by :meth:`add`."""
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
                "mu_bc", "H_bc_anomaly", "bc_reference", "bc_reference_shift", "boundary_reference_shift",
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
