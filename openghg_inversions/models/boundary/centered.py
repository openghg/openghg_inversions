"""Shared base and graph pieces of the centered boundary families.

Centered sensitivity represents the absolute boundary sensitivity as
``A + c M`` (:class:`~.contracts.BoundCenteredSensitivity`). The centered
multiplicative, additive and anomaly-scaling families inherit
:class:`CenteredBoundary` and build their boundary term in the same order:

1. :meth:`CenteredBoundary._register_sensitivity` checks the materialized
   sensitivity once and registers ``A``, ``M`` and ``c``, returning a
   :class:`RegisteredSensitivity`;
2. the family adds its corrections in scientific order, each a
   :class:`Correction`; the additive and anomaly-scaling families start with
   the shared :class:`MeanShift`;
3. :meth:`CenteredBoundary._term` adds the absolute ``mu_bc`` and returns the
   :class:`~openghg_inversions.models.components.AffineConcentrationTerm`.

:meth:`CenteredBoundary._outputs` declares the matching shared outputs. The
base defines no ``add``, so each family's recipe stays explicit. Absolute
multiplicative sensitivity has no centered representation and does not use
these pieces.

The additive and anomaly-scaling families share one mean equation: a shift ``m`` in observation
concentration units on every bound boundary state, either one scalar over the
inversion window or one independent value per boundary period, contributing
``M m`` to the observations. Under supported surface preparation ``M 1 = 1``;
otherwise the response is the propagated shift rather than a constant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.array_ops import get_xr_dummies
from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models.components import AffineBlock, AffineConcentrationTerm, get_model_latent
from openghg_inversions.models.coords import add_component_data, add_model_data, component_coords
from openghg_inversions.models.priors import parse_prior

from .contracts import BoundCenteredSensitivity
from .outputs import Outputs


@dataclass(frozen=True)
class Correction:
    """Graph pieces of one sampled boundary correction, such as a mean shift or anomalies.

    Attributes:
        state: Correction on every bound boundary state. Its units depend on
            the formulation: concentration units for mean shifts and additive
            anomalies, which respond through ``M``; dimensionless for anomaly
            scalings ``v`` and multiplicative deviations ``b - 1``, which
            respond through ``A'`` and ``A + c M``. Combine states only within
            one formulation.
        signal: Observation response of the correction, in concentration
            units.
        block: Affine block: the response to each sampled coefficient, with
            those physical coefficients.
        latent: Sampled variable behind the correction, after any
            reparameterization.
    """

    state: TensorVariable
    signal: TensorVariable
    block: AffineBlock
    latent: TensorVariable


@dataclass(frozen=True)
class RegisteredSensitivity:
    """Bound centered sensitivity registered in the current model.

    Attributes:
        data: The materialized bound sensitivity.
        anomaly: ``A`` registered as model data.
        mass: ``M`` registered as model data.
        reference: The scalar ``c`` registered as model data.
        reference_shift: The fixed ``c M 1``.
        fixed: The fixed contribution ``A 1``.
        output_dim: Observation dimension registered by the model.
        state_dim: Owner-qualified boundary-state dimension.
        prefix: Prefix for graph variable names.
        owner: Namespace for component-owned boundary-state coordinates.
    """

    data: BoundCenteredSensitivity
    anomaly: TensorVariable
    mass: TensorVariable
    reference: TensorVariable
    reference_shift: xr.DataArray
    fixed: xr.DataArray
    output_dim: str
    state_dim: str
    prefix: str
    owner: str | None

    def register(self, array: xr.DataArray, name: str) -> TensorVariable:
        """Register a boundary array with this term's prefix and coordinate owner.

        Args:
            array: Materialized array on observation and/or boundary-state axes.
            name: Unprefixed graph name.

        Returns:
            The registered model data.
        """
        return add_component_data(array, f"{self.prefix}{name}", self.owner, observation_dim=self.output_dim)


@dataclass(frozen=True)
class MeanShift:
    """A shift ``m`` on every bound boundary state, in observation concentration units.

    Attributes:
        prior: Mean shift prior, as passed to
            :func:`~openghg_inversions.models.priors.parse_prior`, or ``None``
            for no shift. With ``by_period``, labelled parameters may lie on
            ``bc_mean_period`` with the fit's period-start labels; they are
            qualified with the coordinate owner like the period indicators.
        by_period: Sample one independent shift per boundary period instead
            of one scalar over the inversion window.
    """

    prior: Mapping[str, Any] | None
    by_period: bool = False

    def add(self, registered: RegisteredSensitivity) -> list[Correction]:
        """Add the shift and its response ``mu_bc_mean_shift`` through ``M``.

        Args:
            registered: The family's registered sensitivity.

        Returns:
            No corrections without a prior. Otherwise one: the shift on every
            state, its response, an affine block whose sensitivity is ``M 1`` (or
            ``M`` summed over each period's states) and the sampled latent.

        Raises:
            ValueError: If a whole-window prior does not define one scalar.
        """
        if self.prior is None:
            return []
        prefix, owner = registered.prefix, registered.owner
        mass = registered.data.H_bc_mass
        periods = mass.bc_period
        if self.by_period:
            indicator = get_xr_dummies(
                periods, categories=np.unique(periods.values), cat_dim="bc_mean_period", return_sparse=False,
            ).astype(float).assign_attrs(units="1")
            owned = component_coords(indicator, owner)
            states = add_model_data(owned, f"{prefix}bc_mean_indicator")
            # Labelled parameters share the indicator's owner-qualified period axis.
            owned_prior = {
                name: component_coords(value, owner) if isinstance(value, xr.DataArray) else value
                for name, value in self.prior.items()
            }
            mean = parse_prior(f"{prefix}bc_mean_shift", owned_prior, dims=str(owned.dims[1]))
            state = pt.dot(states, mean)
            coefficients = mean
        else:
            indicator = xr.ones_like(periods, dtype=float).expand_dims(bc_mean=[0], axis=1)
            mean = parse_prior(f"{prefix}bc_mean_shift", dict(self.prior))
            if mean.ndim != 0:
                raise ValueError("bc_mean_shift_prior must define one scalar over the inversion window.")
            state = pt.ones((mass.sizes["bc_region"],)) * mean
            coefficients = pt.atleast_1d(mean)
        signal = pm.Deterministic(
            f"{prefix}mu_bc_mean_shift", pt.dot(registered.mass, state), dims=registered.output_dim,
        )
        response = xr.dot(mass, indicator, dim="bc_region").assign_attrs(units="1")
        latent = get_model_latent(mean, f"{prefix}bc_mean_shift")
        return [Correction(state, signal, AffineBlock(response, coefficients), latent)]

    def outputs(self, concentration_units: str) -> Outputs:
        """Declare the roles, units and metadata of the shift.

        Args:
            concentration_units: Observation concentration units.

        Returns:
            Unprefixed declarations; ``boundary_mean`` metadata is ``"none"``,
            ``"whole_window"`` or ``"by_period"``.
        """
        if self.prior is None:
            return Outputs(metadata={"boundary_mean": "none"})
        roles = {"boundary_mean_shift": "bc_mean_shift", "boundary_mean_shift_concentration": "mu_bc_mean_shift"}
        units = {"bc_mean_shift": concentration_units, "mu_bc_mean_shift": concentration_units}
        if self.prior.get("reparameterise") and str(self.prior.get("pdf", "")).lower() == "lognormal":
            units["bc_mean_shift_latent"] = "1"
        if self.by_period:
            units["bc_mean_indicator"] = "1"
        return Outputs(roles, units, {"boundary_mean": "by_period" if self.by_period else "whole_window"})


class CenteredChoice(Protocol):
    """Options every centered family's choice provides to :class:`CenteredBoundary`."""

    @property
    def mean_shift(self) -> MeanShift:
        """Resolved mean shift of the family."""
        ...

    @property
    def frequency(self) -> str | None:
        """Boundary-period frequency, or ``None`` for one period."""
        ...

    @property
    def anchor_time(self) -> DatetimeLike | None:
        """Optional start of the fixed-duration period grid."""
        ...


class CenteredBoundary:
    """Base of the bound centered families: a choice and borrowed centered ``data``.

    Subclasses are frozen dataclasses that declare ``choice`` and ``data``.
    Their ``add`` registers the sensitivity, adds corrections in scientific
    order and returns :meth:`_term`; their ``outputs`` extends
    :meth:`_outputs` with the family's own declarations.
    """

    choice: CenteredChoice
    data: BoundCenteredSensitivity

    @property
    def observation_dim(self) -> str:
        """Observation dimension of the bound sensitivity."""
        return BoundCenteredSensitivity.observation_dim(self.data)

    @property
    def concentration_units(self) -> str:
        """Observation concentration units of the bound sensitivity."""
        return str(self.data.bc_reference.attrs["units"])

    def _register_sensitivity(self, *, output_dim: str, prefix: str, owner: str | None) -> RegisteredSensitivity:
        """Check the materialized sensitivity once and register ``A``, ``M`` and ``c``.

        Raises:
            ValueError: If ``output_dim`` is not the bound observation dimension,
                or the operands are lazy or numerically invalid.
        """
        if output_dim != self.observation_dim:
            raise ValueError(f"output_dim {output_dim!r} is not the bound observation dimension {self.observation_dim!r}.")
        data = self.data
        # The class-owned numerical operation rejects lazy values and checks them once.
        shift = BoundCenteredSensitivity.reference_shift(data).rename(f"{prefix}bc_reference_shift")
        anomaly, mass = data.H_bc_anomaly, data.H_bc_mass
        a = add_component_data(anomaly, f"{prefix}H_bc_anomaly", owner, observation_dim=output_dim)
        m = add_component_data(mass, f"{prefix}H_bc_mass", owner, observation_dim=output_dim)
        c = add_component_data(data.bc_reference, f"{prefix}bc_reference", owner, observation_dim=output_dim)
        return RegisteredSensitivity(
            data, a, m, c, shift,
            fixed=anomaly.sum("bc_region", skipna=False).assign_attrs(anomaly.attrs),
            output_dim=output_dim,
            state_dim=str(component_coords(mass, owner, observation_dim=output_dim).dims[1]),
            prefix=prefix,
            owner=owner,
        )

    def _term(
        self,
        registered: RegisteredSensitivity,
        corrections: Sequence[Correction],
        signal: TensorVariable,
        *,
        fixed: xr.DataArray | None = None,
    ) -> AffineConcentrationTerm:
        """Add ``mu_bc`` and return the boundary term.

        Args:
            registered: The registered sensitivity.
            corrections: Corrections in scientific order; their blocks and
                latents form the affine representation.
            signal: Total observation response of the corrections, including
                any part from states that are not sampled.
            fixed: Fixed contribution of the affine representation, when
                unsampled states respond as well as ``A 1``; defaults to
                ``A 1``.

        Returns:
            The term: ``output`` is the absolute ``mu_bc``; the likelihood
            contribution omits the fixed ``c M 1``, which is the reference
            shift; the fixed contribution is ``fixed``.
        """
        absolute_reference = registered.register(
            (registered.fixed + registered.reference_shift).assign_attrs(self.data.H_bc_anomaly.attrs),
            "mu_bc_reference",
        )
        output = pm.Deterministic(
            f"{registered.prefix}mu_bc", absolute_reference + signal, dims=registered.output_dim,
        )
        return AffineConcentrationTerm(
            output, pt.sum(registered.anomaly, axis=1) + signal, registered.reference_shift,
            fixed_contribution=registered.fixed if fixed is None else fixed, blocks=tuple(item.block for item in corrections),
            sampled_rvs=tuple(item.latent for item in corrections),
        )

    def _outputs(
        self,
        formulation: str,
        prefix: str,
        *,
        roles: Mapping[str, str],
        units: Mapping[str, str],
        metadata: Mapping[str, Any],
    ) -> Outputs:
        """Declare the shared centered outputs extended by one family's own.

        Returns:
            The sensitivity, reference and ``mu_bc`` roles and units, the mean
            shift's, then ``roles``, ``units`` and ``metadata``, with
            ``prefix`` applied to roles and graph names.
        """
        concentration_units = self.concentration_units
        mean = self.choice.mean_shift.outputs(concentration_units)
        names = {
            "boundary_concentration": "mu_bc",
            "boundary_anomaly": "H_bc_anomaly",
            "boundary_mass": "H_bc_mass",
            "boundary_reference": "bc_reference",
            "boundary_reference_concentration": "mu_bc_reference",
        }
        anchor_time = self.choice.anchor_time
        return Outputs(
            {**names, **mean.roles, **roles},
            {**{name: concentration_units for name in names.values()}, "H_bc_mass": "1", **mean.units, **units},
            {
                "boundary_formulation": formulation,
                "boundary_representation": "centered-v1",
                **metadata,
                **mean.metadata,
                "bc_frequency": self.choice.frequency,
                "bc_anchor_time": None if anchor_time is None else str(anchor_time),
            },
        ).prefixed(prefix)


__all__ = ["CenteredBoundary", "CenteredChoice", "Correction", "MeanShift", "RegisteredSensitivity"]
