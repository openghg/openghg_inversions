"""Graph and output pieces shared by the centered additive and anomaly-scaling families.

Both families register their bound transport with :func:`add_centered_transport`,
add their corrections in scientific order, and complete the absolute boundary
concentration with :meth:`CenteredGraph.term`. The multiplicative family scales
the whole design instead, so it does not use these pieces.

The families share one mean equation: a shift ``m`` in observation
concentration units on every bound boundary state, either one scalar over the
inversion window or one independent value per boundary period, contributing
``M m`` to the observations. Under supported surface preparation ``M 1 = 1``;
otherwise the response is the transported shift rather than a constant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models.components import AffineBlock, AffineConcentrationTerm, get_model_latent
from openghg_inversions.models.coords import add_model_data, component_coords
from openghg_inversions.models.priors import parse_prior

from .contracts import BoundCenteredTransport
from .outputs import BoundaryCorrection, BoundaryOutputs


def add_boundary_mean_shift(
    prior: Mapping[str, Any],
    mass: xr.DataArray,
    mass_data: TensorVariable,
    *,
    by_period: bool,
    output_dim: str,
    prefix: str,
    owner: str | None,
) -> BoundaryCorrection:
    """Add a transported boundary mean shift to the current model.

    Args:
        prior: Mean shift prior in observation concentration units, as passed
            to :func:`~openghg_inversions.models.priors.parse_prior`. With
            ``by_period``, labelled parameters may lie on ``bc_mean_period``
            with the fit's period-start labels; they are qualified with
            ``owner`` like the period indicators.
        mass: Materialized ``M`` on ``(output_dim, bc_region)`` with a
            ``bc_period`` state level.
        mass_data: The same ``M`` already registered in the model.
        by_period: Sample one independent shift per boundary period instead
            of one scalar over the inversion window.
        output_dim: Observation dimension registered by the model.
        prefix: Prefix for graph variable names.
        owner: Namespace for component-owned coordinates.

    Returns:
        The state shift, its response ``mu_bc_mean_shift``, an affine block
        whose design is ``M 1`` (or ``M`` summed over each period's states)
        and the sampled shift latent.

    Raises:
        ValueError: If a whole-window prior does not define one scalar.
    """
    periods = mass.bc_period
    if by_period:
        labels = pd.unique(periods.values)
        indicator = (periods == xr.DataArray(
            labels, dims="bc_mean_period", coords={"bc_mean_period": labels},
        )).astype(float).assign_attrs(units="1")
        registered = component_coords(indicator, owner)
        states = add_model_data(registered, f"{prefix}bc_mean_indicator")
        # Labelled parameters share the indicator's owner-qualified period axis.
        owned_prior = {
            name: component_coords(value, owner) if isinstance(value, xr.DataArray) else value
            for name, value in prior.items()
        }
        mean = parse_prior(f"{prefix}bc_mean_shift", owned_prior, dims=str(registered.dims[1]))
        state = pt.dot(states, mean)
        coefficients = mean
    else:
        indicator = xr.ones_like(periods, dtype=float).expand_dims(bc_mean=[0], axis=1)
        mean = parse_prior(f"{prefix}bc_mean_shift", dict(prior))
        if mean.ndim != 0:
            raise ValueError("bc_mean_shift_prior must define one scalar over the inversion window.")
        state = pt.ones((mass.sizes["bc_region"],)) * mean
        coefficients = pt.atleast_1d(mean)
    signal = pm.Deterministic(f"{prefix}mu_bc_mean_shift", pt.dot(mass_data, state), dims=output_dim)
    response = xr.dot(mass, indicator, dim="bc_region").assign_attrs(units="1")
    return BoundaryCorrection(
        state, signal, AffineBlock(response, coefficients), get_model_latent(mean, f"{prefix}bc_mean_shift"),
    )


def mean_shift_outputs(prior: Mapping[str, Any] | None, *, by_period: bool, concentration: str) -> BoundaryOutputs:
    """Declare the roles, units and metadata of a boundary mean shift.

    Args:
        prior: Mean shift prior, or ``None`` when the family samples no shift.
        by_period: Whether one shift is sampled per boundary period.
        concentration: Observation concentration units.

    Returns:
        Unprefixed declarations; ``boundary_mean`` metadata is ``"none"``,
        ``"whole_window"`` or ``"by_period"``.
    """
    if prior is None:
        return BoundaryOutputs(metadata={"boundary_mean": "none"})
    roles = {"boundary_mean_shift": "bc_mean_shift", "boundary_mean_shift_concentration": "mu_bc_mean_shift"}
    units = {"bc_mean_shift": concentration, "mu_bc_mean_shift": concentration}
    if prior.get("reparameterise") and str(prior.get("pdf", "")).lower() == "lognormal":
        units["bc_mean_shift_latent"] = "1"
    if by_period:
        units["bc_mean_indicator"] = "1"
    return BoundaryOutputs(roles, units, {"boundary_mean": "by_period" if by_period else "whole_window"})


@dataclass(frozen=True)
class CenteredGraph:
    """Bound centered transport registered in the current model.

    Attributes:
        data: The materialized bound transport.
        output_dim: Observation dimension registered by the model.
        prefix: Prefix for graph variable names.
        owner: Namespace for component-owned boundary-state coordinates.
        anomaly: ``A`` registered as model data.
        mass: ``M`` registered as model data.
        reference_shift: The fixed ``c M 1``.
        fixed: The fixed contribution ``A 1``.
        state_dim: Owner-qualified boundary-state dimension.
    """

    data: BoundCenteredTransport
    output_dim: str
    prefix: str
    owner: str | None
    anomaly: TensorVariable
    mass: TensorVariable
    reference_shift: xr.DataArray
    fixed: xr.DataArray
    state_dim: str

    def register(self, array: xr.DataArray, name: str) -> TensorVariable:
        """Register a boundary array with the recipe's coordinate ownership."""
        return _register(array, name, self.output_dim, self.prefix, self.owner)

    def mean_shift(self, prior: Mapping[str, Any] | None, *, by_period: bool) -> list[BoundaryCorrection]:
        """Add the optional mean shift, transported through ``M``."""
        if prior is None:
            return []
        return [add_boundary_mean_shift(
            prior, self.data.H_bc_mass, self.mass, by_period=by_period,
            output_dim=self.output_dim, prefix=self.prefix, owner=self.owner,
        )]

    def term(self, corrections: Sequence[BoundaryCorrection]) -> AffineConcentrationTerm:
        """Add ``mu_bc`` and return the boundary term with the corrections' blocks and latents."""
        signal = sum((item.signal for item in corrections), start=pt.zeros((self.data.sizes[self.output_dim],)))
        absolute_reference = self.register(
            (self.fixed + self.reference_shift).assign_attrs(self.data.H_bc_anomaly.attrs), "mu_bc_reference",
        )
        output = pm.Deterministic(f"{self.prefix}mu_bc", absolute_reference + signal, dims=self.output_dim)
        return AffineConcentrationTerm(
            output, pt.sum(self.anomaly, axis=1) + signal, self.reference_shift,
            fixed_contribution=self.fixed, blocks=tuple(item.block for item in corrections),
            sampled_rvs=tuple(item.latent for item in corrections),
        )


def _register(array: xr.DataArray, name: str, output_dim: str, prefix: str, owner: str | None) -> TensorVariable:
    return add_model_data(component_coords(array, owner, observation_dim=output_dim), f"{prefix}{name}")


def add_centered_transport(
    data: BoundCenteredTransport, *, output_dim: str, prefix: str, owner: str | None,
) -> CenteredGraph:
    """Check materialized transport once and register ``A``, ``M`` and ``c``.

    Raises:
        ValueError: If ``output_dim`` is not the bound observation dimension,
            or the operands are lazy or numerically invalid.
    """
    observation_dim = BoundCenteredTransport.observation_dim(data)
    if output_dim != observation_dim:
        raise ValueError(f"output_dim {output_dim!r} is not the bound observation dimension {observation_dim!r}.")
    # The class-owned numerical operation rejects lazy values and checks them once.
    shift = BoundCenteredTransport.reference_shift(data).rename(f"{prefix}bc_reference_shift")
    anomaly, mass = data.H_bc_anomaly, data.H_bc_mass
    a = _register(anomaly, "H_bc_anomaly", output_dim, prefix, owner)
    m = _register(mass, "H_bc_mass", output_dim, prefix, owner)
    _register(data.bc_reference, "bc_reference", output_dim, prefix, owner)
    return CenteredGraph(
        data, output_dim, prefix, owner, a, m, shift,
        fixed=anomaly.sum("bc_region", skipna=False).assign_attrs(anomaly.attrs),
        state_dim=str(component_coords(mass, owner, observation_dim=output_dim).dims[1]),
    )


def centered_outputs(
    data: BoundCenteredTransport,
    formulation: str,
    *,
    mean_prior: Mapping[str, Any] | None,
    mean_by_period: bool,
    frequency: str | None,
    anchor_time: DatetimeLike | None,
    roles: Mapping[str, str],
    units: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> BoundaryOutputs:
    """Declare the shared centered outputs extended by one family's own.

    Returns:
        Unprefixed declarations: the transport, reference and ``mu_bc`` roles
        and units, the mean shift's, then ``roles``, ``units`` and ``metadata``.
    """
    concentration = data.bc_reference.attrs["units"]
    mean = mean_shift_outputs(mean_prior, by_period=mean_by_period, concentration=concentration)
    names = {
        "boundary_concentration": "mu_bc",
        "boundary_anomaly": "H_bc_anomaly",
        "boundary_mass": "H_bc_mass",
        "boundary_reference": "bc_reference",
        "boundary_reference_concentration": "mu_bc_reference",
    }
    return BoundaryOutputs(
        {**names, **mean.roles, **roles},
        {**{name: concentration for name in names.values()}, "H_bc_mass": "1", **mean.units, **units},
        {
            "boundary_formulation": formulation,
            "boundary_representation": "centered-v1",
            **metadata,
            **mean.metadata,
            "bc_frequency": frequency,
            "bc_anchor_time": None if anchor_time is None else str(anchor_time),
        },
    )


__all__ = [
    "CenteredGraph",
    "add_boundary_mean_shift",
    "add_centered_transport",
    "centered_outputs",
    "mean_shift_outputs",
]
