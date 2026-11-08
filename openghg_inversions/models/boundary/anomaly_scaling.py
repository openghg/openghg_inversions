"""Boundary corrections that scale the prescribed concentration anomaly.

Centered transport (:class:`~.contracts.BoundCenteredTransport`; see
:doc:`/usage/centered_boundary_conditions`) gives the prescribed boundary as
``A 1 + c M 1``. This
family adds two corrections: a mean shift ``m`` in concentration units,
transported through ``M`` as in the additive family, and dimensionless
independent zero-mean Gaussian scalings ``v`` of the reference-adjusted
anomaly ``A'`` on each ``(bc_curtain, bc_period)`` state:

    m M 1 + A' v

``A'`` is the anomaly about a chosen reference ``c'``. Changing the reference
leaves the prescribed concentration unchanged but changes ``A'``, and hence
the covariance of ``A' v``: the reference is part of this prior, and this is
the only boundary family where it has scientific meaning. Explicitly,
``A' = A + (c - c') M``, so ``A' 1 + c' M 1 = A 1 + c M 1``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from dask import is_dask_collection
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.boundary_sensitivity import validate_centered_boundary_periods
from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models.components import AffineBlock, AffineConcentrationTerm
from openghg_inversions.models.coords import add_model_data, component_coords

from .mean_shift import add_boundary_mean_shift, mean_shift_outputs
from .contracts import BoundCenteredTransport, RawCenteredTransport
from .outputs import BoundaryOutputs

ReferencePolicy = float | Literal["stored", "whole_window_transport_mean"]


@dataclass(frozen=True)
class AnomalyScalingChoice:
    """Prior specification for anomaly-scaling boundary corrections, before binding.

    A choice holds the scientific options (mean shift prior, scaling width,
    reference policy and boundary period calendar) without bound
    observations or transport arrays.
    :meth:`bind` pairs it with one fit's observations and centered transport.

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
        mean_by_period: Fit one independent mean shift per boundary period;
            requires a mean prior and a ``frequency``.
        frequency: Boundary period frequency, with the syntax and calendar of
            ``bc_freq``; see :class:`~openghg_inversions.models.boundary.AdditiveChoice`.
        anchor_time: Origin of fixed-duration periods; ``None`` uses the
            earliest selected observation of the fit. See
            :class:`~openghg_inversions.models.boundary.AdditiveChoice`.

    Raises:
        ValueError: On construction, for a non-positive or non-finite scale,
            an unknown or non-finite reference, ``mean_by_period`` without a
            mean prior or frequency, or an invalid frequency or anchor.
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

    def bind(self, observations: xr.DataArray, transport: RawCenteredTransport) -> AnomalyScalingBoundary:
        """Bind centered transport to one fit and re-express it about the chosen reference.

        Args:
            observations: One channel's observations with its observation axis
                first. Its labels select and order the transport rows; its
                ``time`` values assign rows to boundary periods.
            transport: Raw centered transport on its ``nmeasure`` or ``time``
                axis. Centering weights are ignored.

        Returns:
            The bound boundary. ``H_bc_anomaly`` is the recentred ``A'`` and
            ``H_bc_mass`` is ``M``, both on ``(observation, bc_region)`` with
            a ``(bc_curtain, bc_period)`` state index; ``bc_reference`` is the
            chosen scalar ``c'``. A whole-window reference is computed from
            the rows of this fit. Payloads, the recentred anomaly and any
            computed reference stay lazy until joint materialization.

        Raises:
            ValueError: If PyTensor is not configured for float64, or periods
                or indexed timestamps conflict with the observations.
        """
        data = BoundCenteredTransport.bind(
            transport, observations, frequency=self.frequency, anchor_time=self.anchor_time,
        )
        if self.reference == "stored":
            return AnomalyScalingBoundary(self, data)
        anomaly, mass, stored = data.H_bc_anomaly, data.H_bc_mass, data.bc_reference
        if self.reference == "whole_window_transport_mean":
            shift = anomaly.sum() / mass.sum()
        else:
            shift = float(self.reference) - stored
        # Recentring keeps the bound layout and units: the shift is a scalar.
        recentred = data.assign(
            H_bc_anomaly=(anomaly - shift * mass).assign_attrs(anomaly.attrs),
            bc_reference=(stored + shift).assign_attrs(stored.attrs),
        )
        return AnomalyScalingBoundary(self, cast(BoundCenteredTransport, recentred))


@dataclass(frozen=True)
class AnomalyScalingBoundary:
    """Anomaly-scaling choice bound to one fit's borrowed centered operands.

    Attributes:
        choice: The resolved anomaly-scaling choice.
        data: The recentred ``H_bc_anomaly`` ``A'`` (concentration units) and
            ``H_bc_mass`` (dimensionless) on ``(observation, bc_region)``,
            where ``bc_region`` indexes ``(bc_curtain, bc_period)`` states,
            and the chosen scalar ``bc_reference`` ``c'`` in the anomaly's
            units. The arrays are borrowed and may be Dask-backed.
            Materialize them jointly with the other model inputs before
            :meth:`add`; rows may be padded or subset with
            ``dataclasses.replace``, keeping that layout.

    :meth:`AnomalyScalingChoice.bind` establishes ``data`` by construction,
    so the record trusts that contract rather than validating it again.
    """

    choice: AnomalyScalingChoice
    data: BoundCenteredTransport

    @property
    def observation_dim(self) -> str:
        """Observation dimension of the bound transport."""
        return BoundCenteredTransport.observation_dim(self.data)

    def anomaly_factor(self) -> xr.DataArray:
        """Observation-space anomaly design scaled to standard-normal coordinates.

        The design stays lazy with the transport, so it can be combined with
        other designs before joint materialization.

        Returns:
            ``s A'`` on ``(observation, bc_region)`` in concentration units.
            Multiplying it by independent standard normals gives the
            anomaly-scaling concentration.
        """
        return (self.data.H_bc_anomaly * self.choice.scale_sd).assign_attrs(self.data.H_bc_anomaly.attrs)

    def add(self, *, output_dim: str = "nmeasure", prefix: str = "", owner: str | None = "bc") -> AffineConcentrationTerm:
        """Add the mean shift, anomaly scalings and boundary concentration to the current model.

        Must be called inside an active PyMC model that has the coordinate
        registry, after ``data`` has been materialized jointly with the other
        model inputs. It registers the transport, reference, scale and any
        period indicators as model data, plus the sampled variables and
        their deterministic responses.

        Args:
            output_dim: Observation dimension registered by the model.
            prefix: Prefix for graph variable names.
            owner: Namespace for component-owned boundary-state coordinates;
                ``None`` keeps the local names such as ``bc_region``. Graph
                names are controlled separately by ``prefix``.

        Returns:
            The boundary term: ``output`` is the absolute boundary
            concentration ``mu_bc``; the likelihood contribution omits the
            fixed ``c' M 1``, which is the reference shift; the fixed
            contribution is ``A' 1``; the affine blocks are the mean response
            through ``M`` (when sampled) followed by ``A'`` for the scalings
            ``v``; and the sampled variables are the mean latent and ``v``.

        Raises:
            ValueError: If ``output_dim`` is not the bound observation
                dimension, ``data`` is still lazy, materialized operands are
                invalid, or a whole-window mean prior is not scalar.
            RuntimeError: If the model lacks the coordinate registry.
        """
        if output_dim != self.observation_dim:
            raise ValueError(f"output_dim {output_dim!r} is not the bound observation dimension {self.observation_dim!r}.")
        if any(is_dask_collection(value) for value in self.data.data_vars.values()):
            raise ValueError("Materialize boundary data jointly with model inputs before add().")

        # Values become checkable only after joint materialization.
        choice = self.choice
        anomaly = self.data.H_bc_anomaly
        mass = self.data.H_bc_mass
        reference = self.data.bc_reference
        reference_shift = BoundCenteredTransport.reference_shift(self.data).rename(f"{prefix}bc_reference_shift")
        fixed = anomaly.sum("bc_region", skipna=False).assign_attrs(anomaly.attrs)

        # Owner-qualified coordinates plus model data, sharing the observation axis.
        def register_boundary_data(array: xr.DataArray, name: str) -> TensorVariable:
            """Register a boundary array with the recipe's coordinate ownership."""
            return add_model_data(component_coords(array, owner, observation_dim=output_dim), f"{prefix}{name}")

        a = register_boundary_data(anomaly, "H_bc_anomaly")
        m = register_boundary_data(mass, "H_bc_mass")
        register_boundary_data(reference, "bc_reference")
        # Save the resolved width for provenance; the prior uses the scalar directly.
        register_boundary_data(xr.DataArray(choice.scale_sd), "bc_scale_sd")
        state_dim = str(component_coords(mass, owner, observation_dim=output_dim).dims[1])
        latents: tuple[TensorVariable, ...] = ()
        blocks: list[AffineBlock] = []

        # Mean shift, transported through M.
        mean_signal = pt.zeros((mass.sizes[output_dim],))
        if choice.mean_prior is not None:
            mean = add_boundary_mean_shift(
                choice.mean_prior, mass, m, by_period=choice.mean_by_period,
                output_dim=output_dim, prefix=prefix, owner=owner,
            )
            mean_signal = mean.signal
            latents += (mean.latent,)
            blocks.append(mean.block)

        # Scalings v of the recentred anomaly, contributing A' v.
        scaling = pm.Normal(f"{prefix}bc_scaling", mu=0.0, sigma=choice.scale_sd, dims=state_dim)
        scaling_signal = pm.Deterministic(f"{prefix}mu_bc_scaling", pt.dot(a, scaling), dims=output_dim)
        latents += (scaling,)
        blocks.append(AffineBlock(anomaly, scaling))

        # Complete boundary contribution A' 1 + c' M 1 + m M 1 + A' v.
        absolute_reference = register_boundary_data(
            (fixed + reference_shift).assign_attrs(anomaly.attrs), "mu_bc_reference",
        )
        output = pm.Deterministic(
            f"{prefix}mu_bc", absolute_reference + mean_signal + scaling_signal, dims=output_dim,
        )
        return AffineConcentrationTerm(
            output, pt.sum(a, axis=1) + mean_signal + scaling_signal, reference_shift,
            fixed_contribution=fixed, blocks=tuple(blocks), sampled_rvs=latents,
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
        choice = self.choice
        concentration = self.data.bc_reference.attrs["units"]
        roles = {
            "boundary_concentration": "mu_bc",
            "boundary_anomaly": "H_bc_anomaly",
            "boundary_mass": "H_bc_mass",
            "boundary_reference": "bc_reference",
            "boundary_reference_concentration": "mu_bc_reference",
            # The shared anomaly roles carry the dimensionless scalings v and
            # their contribution A' v through boundary reconstruction.
            "boundary_correction_anomaly": "bc_scaling",
            "boundary_anomaly_concentration": "mu_bc_scaling",
        }
        units = {name: concentration for name in (
            "mu_bc", "H_bc_anomaly", "bc_reference", "mu_bc_reference", "mu_bc_scaling",
        )}
        units.update(H_bc_mass="1", bc_scaling="1", bc_scale_sd="1")

        mean = mean_shift_outputs(choice.mean_prior, by_period=choice.mean_by_period, concentration=concentration)
        roles.update(mean.roles)
        units.update(mean.units)

        reference = choice.reference
        metadata = {
            "boundary_formulation": "anomaly_scaling",
            "boundary_representation": "centered-v1",
            **mean.metadata,
            "bc_reference_policy": reference if isinstance(reference, str) else "explicit",
            "bc_frequency": choice.frequency,
            "bc_anchor_time": None if choice.anchor_time is None else str(choice.anchor_time),
        }
        return BoundaryOutputs(roles, units, metadata).prefixed(prefix)


__all__ = ["AnomalyScalingBoundary", "AnomalyScalingChoice"]
