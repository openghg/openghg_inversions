"""Boundary mean shift transported through the dimensionless mass ``M``.

The additive and anomaly-scaling families share one mean equation: a shift
``m`` in observation concentration units on every bound boundary state,
either one scalar over the inversion window or one independent value per
boundary period, contributing ``M m`` to the observations. Under supported
surface preparation ``M 1 = 1``; otherwise the response is the transported
shift rather than a constant.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.models.components import AffineBlock, get_model_latent
from openghg_inversions.models.coords import add_model_data, component_coords
from openghg_inversions.models.priors import parse_prior

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


__all__ = ["add_boundary_mean_shift", "mean_shift_outputs"]
