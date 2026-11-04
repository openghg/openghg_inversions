"""Boundary corrections that scale the prescribed concentration anomaly.

Centered transport gives the prescribed boundary as ``A 1 + c M 1``. Choosing
a reference ``c'`` re-expresses the same prescribed boundary with the anomaly
``A' = A + (c - c') M``, so ``A' 1 + c' M 1 = A 1 + c M 1``. This family adds

    m M 1 + A' v

a mean shift ``m`` in concentration units, responding through ``M`` as in the
additive family, and dimensionless independent zero-mean Gaussian scalings
``v`` of the anomaly on each ``(bc_curtain, bc_period)`` state. The prescribed
boundary is unchanged by the reference, but the covariance of ``A' v`` is not:
the reference is part of the prior. It is the only boundary family where the
reference has scientific meaning.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from dask import is_dask_collection
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.boundary_sensitivity import validate_centered_boundary_periods
from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models.components import AffineBlock, AffineConcentrationTerm, get_model_latent
from openghg_inversions.models.coords import add_model_data, component_coords
from openghg_inversions.models.priors import parse_prior

from .operands import CenteredOperands
from .outputs import BoundaryOutputs

ReferencePolicy = float | Literal["stored", "whole_window_transport_mean"]


@dataclass(frozen=True)
class AnomalyScalingChoice:
    """Data-free anomaly-scaling boundary choice.

    Attributes:
        mean_prior: Prior for the mean shift in observation concentration
            units, scalar or one shift per period with ``mean_by_period``; or
            ``None`` for no mean shift.
        scale_sd: Finite positive standard deviation of the dimensionless
            anomaly scalings.
        reference: ``"stored"`` keeps the prepared reference; a finite number
            in observation concentration units replaces it; and
            ``"whole_window_transport_mean"`` chooses the reference for which
            the fit's recentred anomaly sums to zero over all rows,
            ``c + sum(A) / sum(M)``.
        mean_by_period: Fit one independent mean shift per boundary period.
        frequency: Optional boundary period frequency.
        anchor_time: Optional origin for fixed-duration periods.
    """

    mean_prior: Mapping[str, Any] | None
    scale_sd: float
    reference: ReferencePolicy = "stored"
    mean_by_period: bool = False
    frequency: str | None = None
    anchor_time: DatetimeLike | None = None

    def __post_init__(self) -> None:
        if self.mean_prior is not None:
            object.__setattr__(self, "mean_prior", dict(self.mean_prior))
        validate_centered_boundary_periods(self.frequency, self.anchor_time)
        if not np.isfinite(self.scale_sd) or self.scale_sd <= 0:
            raise ValueError("bc_scale_sd must be finite and positive.")
        if isinstance(self.reference, str):
            if self.reference not in ("stored", "whole_window_transport_mean"):
                raise ValueError("bc_reference must be 'stored', 'whole_window_transport_mean' or a finite number.")
        elif not np.isfinite(self.reference):
            raise ValueError("bc_reference must be 'stored', 'whole_window_transport_mean' or a finite number.")
        if self.mean_by_period and (self.mean_prior is None or self.frequency is None):
            raise ValueError("bc_mean_by_period requires a mean shift prior and a boundary frequency.")

    def bind(
        self, observations: xr.DataArray, transport: xr.DataArray | CenteredOperands,
    ) -> AnomalyScalingBoundary:
        """Bind centered transport to one fit and re-express it about the chosen reference.

        The recentred anomaly and any whole-window reference stay lazy; they
        are computed with the other model inputs.
        """
        if not isinstance(transport, CenteredOperands):
            raise ValueError("Anomaly-scaling boundaries require centered operands.")
        data = transport.bind(observations, frequency=self.frequency, anchor_time=self.anchor_time)
        if self.reference == "stored":
            return AnomalyScalingBoundary(self, data)
        anomaly, mass, stored = data.H_bc_anomaly, data.H_bc_mass, data.bc_reference
        if self.reference == "whole_window_transport_mean":
            shift = anomaly.sum() / mass.sum()
        else:
            shift = float(self.reference) - stored
        return AnomalyScalingBoundary(self, data.assign(
            H_bc_anomaly=(anomaly - shift * mass).assign_attrs(anomaly.attrs),
            bc_reference=(stored + shift).assign_attrs(stored.attrs),
        ))


@dataclass(frozen=True)
class AnomalyScalingBoundary:
    """Anomaly-scaling choice bound to one fit's borrowed centered operands.

    ``data`` holds the recentred ``H_bc_anomaly``, ``H_bc_mass`` and the chosen
    ``bc_reference`` on the bound ``bc_region`` states. Materialize it with
    the other model inputs before :meth:`add`.
    """

    choice: AnomalyScalingChoice
    data: xr.Dataset

    def anomaly_factor(self) -> xr.DataArray:
        """Recentred anomaly ``s A'`` on ``(observation, bc_region)``, lazy with the transport.

        The anomaly-scaling concentration is this factor times independent
        standard normals, in concentration units.
        """
        return (self.data.H_bc_anomaly * self.choice.scale_sd).assign_attrs(self.data.H_bc_anomaly.attrs)

    def add(self, *, output_dim: str = "nmeasure", prefix: str = "", owner: str | None = "bc") -> AffineConcentrationTerm:
        """Add the mean shift, anomaly scalings and boundary concentration to the current model.

        Its affine blocks are the mean response through ``M`` (first) and the
        recentred anomaly ``A'`` for the scalings.

        Args:
            output_dim: Observation dimension registered by the model.
            prefix: Prefix for graph variable names.
            owner: Namespace for labelled boundary-state coordinates.

        Raises:
            ValueError: If ``data`` is still lazy, materialized operands are
                invalid, or a whole-window mean prior is not scalar.
        """
        if any(is_dask_collection(value) for value in self.data.data_vars.values()):
            raise ValueError("Materialize boundary data jointly with model inputs before add().")
        choice = self.choice
        anomaly = self.data.H_bc_anomaly.transpose(output_dim, "bc_region")
        mass = self.data.H_bc_mass.transpose(output_dim, "bc_region")
        reference = self.data.bc_reference
        if not np.isfinite(reference.values).all() or not np.isfinite(anomaly.values).all():
            raise ValueError("Centered bc_reference and boundary anomaly must be finite.")
        if not np.isfinite(mass.values).all() or (mass.values < 0).any() or not (mass.values > 0).any():
            raise ValueError("H_bc_mass must be finite, non-negative and not all zero.")
        reference_shift = (reference * mass.sum("bc_region", skipna=False)).rename(
            f"{prefix}bc_reference_shift"
        ).assign_attrs(reference.attrs)
        fixed = anomaly.sum("bc_region", skipna=False).assign_attrs(anomaly.attrs)
        count = mass.sizes["bc_region"]
        periods = mass.bc_period

        def register(array: xr.DataArray, name: str) -> TensorVariable:
            return add_model_data(component_coords(array, owner, observation_dim=output_dim), f"{prefix}{name}")

        a = register(anomaly, "H_bc_anomaly")
        m = register(mass, "H_bc_mass")
        register(reference, "bc_reference")
        register(xr.DataArray(choice.scale_sd), "bc_scale_sd")
        state_dim = str(component_coords(mass, owner, observation_dim=output_dim).dims[1])
        latents: tuple[TensorVariable, ...] = ()
        blocks: list[AffineBlock] = []
        mean_signal = pt.zeros((mass.sizes[output_dim],))
        if choice.mean_prior is not None:
            if choice.mean_by_period:
                labels = pd.unique(periods.values)
                indicator = (periods == xr.DataArray(
                    labels, dims="bc_mean_period", coords={"bc_mean_period": labels},
                )).astype(float).assign_attrs(units="1")
                period_dim = str(component_coords(indicator, owner).dims[1])
                states = register(indicator, "bc_mean_indicator")
                mean = parse_prior(f"{prefix}bc_mean_shift", dict(choice.mean_prior), dims=period_dim)
                mean_state = pt.dot(states, mean)
                coefficients = mean
            else:
                indicator = xr.ones_like(periods, dtype=float).expand_dims(bc_mean=[0], axis=1)
                mean = parse_prior(f"{prefix}bc_mean_shift", dict(choice.mean_prior))
                if mean.ndim != 0:
                    raise ValueError("bc_mean_shift_prior must define one scalar over the inversion window.")
                mean_state = pt.ones((count,)) * mean
                coefficients = pt.atleast_1d(mean)
            mean_signal = pm.Deterministic(f"{prefix}mu_bc_mean_shift", pt.dot(m, mean_state), dims=output_dim)
            latents += (get_model_latent(mean, f"{prefix}bc_mean_shift"),)
            blocks.append(AffineBlock(xr.dot(mass, indicator, dim="bc_region").assign_attrs(units="1"), coefficients))
        scaling = pm.Normal(f"{prefix}bc_scaling", mu=0.0, sigma=choice.scale_sd, dims=state_dim)
        scaling_signal = pm.Deterministic(f"{prefix}mu_bc_scaling", pt.dot(a, scaling), dims=output_dim)
        absolute_reference = register(
            (fixed + reference_shift).assign_attrs(anomaly.attrs), "mu_bc_reference",
        )
        output = pm.Deterministic(
            f"{prefix}mu_bc", absolute_reference + mean_signal + scaling_signal, dims=output_dim,
        )
        blocks.append(AffineBlock(anomaly, scaling))
        return AffineConcentrationTerm(
            output, pt.sum(a, axis=1) + mean_signal + scaling_signal, reference_shift,
            fixed_contribution=fixed, blocks=tuple(blocks), sampled_rvs=(*latents, scaling),
        )

    def outputs(self, prefix: str = "") -> BoundaryOutputs:
        """Declare roles, units and metadata for the graph built by :meth:`add`."""
        choice = self.choice
        concentration = self.data.bc_reference.attrs["units"]
        roles = {
            "boundary_concentration": "mu_bc",
            "boundary_anomaly": "H_bc_anomaly",
            "boundary_mass": "H_bc_mass",
            "boundary_reference": "bc_reference",
            "boundary_reference_concentration": "mu_bc_reference",
            "boundary_anomaly_scaling": "bc_scaling",
            "boundary_anomaly_scaling_concentration": "mu_bc_scaling",
        }
        units = {name: concentration for name in (
            "mu_bc", "H_bc_anomaly", "bc_reference", "bc_reference_shift", "boundary_reference_shift",
            "mu_bc_reference", "mu_bc_scaling",
        )}
        units.update(H_bc_mass="1", bc_scaling="1", bc_scale_sd="1")
        prior = choice.mean_prior
        if prior is not None:
            roles.update(boundary_mean_shift="bc_mean_shift", boundary_mean_shift_concentration="mu_bc_mean_shift")
            units.update(bc_mean_shift=concentration, mu_bc_mean_shift=concentration)
            if prior.get("reparameterise") and str(prior.get("pdf", "")).lower() == "lognormal":
                units["bc_mean_shift_latent"] = "1"
            if choice.mean_by_period:
                units["bc_mean_indicator"] = "1"
        reference = choice.reference
        metadata = {
            "boundary_formulation": "anomaly_scaling",
            "boundary_representation": "centered-v1",
            "boundary_mean": "none" if prior is None else "by_period" if choice.mean_by_period else "whole_window",
            "bc_reference_policy": reference if isinstance(reference, str) else "explicit",
            "bc_frequency": choice.frequency,
            "bc_anchor_time": None if choice.anchor_time is None else str(choice.anchor_time),
        }
        return BoundaryOutputs(roles, units, metadata).prefixed(prefix)


__all__ = ["AnomalyScalingBoundary", "AnomalyScalingChoice"]
