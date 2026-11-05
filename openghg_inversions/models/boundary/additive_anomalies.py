"""Additive anomaly formulations in boundary-state concentration units.

Weighted and per-period anomalies use distinct contrast bases ``B`` and
contribute ``M B z``. Independent anomalies sample the states directly and
contribute ``M z``. The basis functions also supply the lazy affine designs;
the graph constructors receive the current materialized transport explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from dask import is_dask_collection
from pytensor.tensor.variable import TensorVariable
from scipy.linalg import helmert

from openghg_inversions.models.components import AffineBlock
from openghg_inversions.models.coords import add_model_data, component_coords

from .outputs import BoundaryOutputs

AnomalyStructure = Literal["weighted", "period_contrasts", "independent"]


@dataclass(frozen=True)
class BoundaryAnomalies:
    """Graph pieces of one additive boundary anomaly contribution.

    Attributes:
        state: Concentration correction on each boundary state, ``bc_anomaly``.
        signal: Transported correction ``M state``, named ``mu_bc_anomaly``.
        block: Observation-space design and physical anomaly coefficients.
        latent: Sampled state or contrast coefficients, in concentration units.
    """

    state: TensorVariable
    signal: TensorVariable
    block: AffineBlock
    latent: TensorVariable


def normalized_centering_weights(weights: xr.DataArray) -> xr.DataArray:
    """Validate materialized centering weights and normalize their sum to one.

    Args:
        weights: Borrowed positive weights on the bound ``bc_region`` states.

    Returns:
        Dimensionless weights with the same state coordinates. Scaling by the
        maximum first avoids overflow when summing large finite weights.

    Raises:
        ValueError: If weights are lazy, empty, non-finite or not positive.
    """
    if is_dask_collection(weights.data):
        raise ValueError("Materialize bc_centering_weights before forming a weighted anomaly design.")
    if not weights.size or not np.isfinite(weights.values).all() or not (weights.values > 0).all():
        raise ValueError("bc_centering_weights must be finite and strictly positive.")
    scaled = weights / weights.max()
    return (scaled / scaled.sum()).assign_attrs(units="1")


def weighted_anomaly_basis(weights: xr.DataArray) -> xr.DataArray:
    """Build the globally weighted contrast basis ``Q - 1 wᵀQ``.

    Args:
        weights: Materialized, normalized weights on ``bc_region`` with the
            bound curtain/period state coordinates.

    Returns:
        Dimensionless basis on ``(bc_region, bc_contrast)`` with ``N - 1``
        columns for ``N`` states. Its weighted state mean is zero.
    """
    q = helmert(weights.size, full=False).T
    q = q - np.ones((weights.size, 1)) @ (weights.values @ q)[None, :]
    return xr.DataArray(
        q, dims=("bc_region", "bc_contrast"), coords={"bc_contrast": np.arange(q.shape[1])},
        attrs={"units": "1"},
    ).assign_coords(weights.bc_region.coords)


def period_contrast_basis(states: xr.DataArray) -> xr.DataArray:
    """Build separate equal-weight curtain contrasts within each period.

    Args:
        states: Bound ``bc_region`` coordinate with a ``bc_period`` level.

    Returns:
        Dimensionless basis on ``(bc_region, bc_contrast)`` with ``N - P``
        columns for ``N`` states across ``P`` periods. Each column sums to zero
        within every period; periods have separate contrast coordinates.
    """
    periods = states.bc_period.values
    columns = []
    for label in pd.unique(periods):
        members = np.flatnonzero(periods == label)
        block = np.zeros((states.size, members.size - 1))
        block[members] = helmert(members.size, full=False).T
        columns.append(block)
    q = np.hstack(columns)
    return xr.DataArray(
        q, dims=("bc_region", "bc_contrast"), coords={"bc_contrast": np.arange(q.shape[1])},
        attrs={"units": "1"},
    ).assign_coords(states.coords)


def add_weighted_anomalies(
    mass: xr.DataArray, mass_data: TensorVariable, weights: xr.DataArray, scale: float,
    *, output_dim: str, prefix: str, owner: str | None,
) -> BoundaryAnomalies:
    """Sample globally weighted zero-mean anomalies and transport them through ``M``.

    Args:
        mass: Materialized dimensionless ``M`` on ``(output_dim, bc_region)``.
        mass_data: The same ``M`` registered in the active model.
        weights: Normalized centering weights aligned with the bound states.
        scale: Positive Gaussian width of the contrast coefficients in
            observation concentration units.
        output_dim: Registered observation dimension.
        prefix: Graph variable prefix.
        owner: Namespace for boundary-state and contrast coordinates.

    Returns:
        State anomalies, their response, affine block and sampled contrasts.
        Call inside a registered model when at least two states are present.
    """
    basis = weighted_anomaly_basis(weights)
    return _add_contrast_anomalies(mass, mass_data, basis, scale, output_dim=output_dim, prefix=prefix, owner=owner)


def add_period_contrast_anomalies(
    mass: xr.DataArray, mass_data: TensorVariable, scale: float,
    *, output_dim: str, prefix: str, owner: str | None,
) -> BoundaryAnomalies:
    """Sample independent curtain contrasts for each boundary period.

    Args:
        mass: Materialized dimensionless ``M`` on ``(output_dim, bc_region)``
            with bound curtain/period state coordinates.
        mass_data: The same ``M`` registered in the active model.
        scale: Positive Gaussian width of the contrast coefficients in
            observation concentration units.
        output_dim: Registered observation dimension.
        prefix: Graph variable prefix.
        owner: Namespace for boundary-state and contrast coordinates.

    Returns:
        State anomalies, their response, affine block and sampled contrasts.
        Call inside a registered model with at least one curtain contrast.
    """
    basis = period_contrast_basis(mass.bc_region)
    return _add_contrast_anomalies(mass, mass_data, basis, scale, output_dim=output_dim, prefix=prefix, owner=owner)


def _add_contrast_anomalies(
    mass: xr.DataArray, mass_data: TensorVariable, basis: xr.DataArray, scale: float,
    *, output_dim: str, prefix: str, owner: str | None,
) -> BoundaryAnomalies:
    """Sample physical contrast coefficients and construct ``B z`` and ``M B z``.

    Args:
        mass: Materialized dimensionless transport on ``(output_dim, bc_region)``.
        mass_data: Registered tensor for ``mass`` in the active model.
        basis: Dimensionless contrast-to-state matrix, aligned with ``mass``.
        scale: Gaussian width in observation concentration units.
        output_dim: Registered observation dimension.
        prefix: Graph variable prefix.
        owner: Namespace for the state and contrast coordinates.

    Returns:
        State anomalies, their response, affine block and sampled contrasts.
    """
    owned_basis = component_coords(basis, owner)
    basis_data = add_model_data(owned_basis, f"{prefix}bc_anomaly_basis")
    state_dim, contrast_dim = map(str, owned_basis.dims)
    coordinates = pm.Normal(f"{prefix}bc_anomaly_latent", mu=0.0, sigma=scale, dims=contrast_dim)
    state = pm.Deterministic(f"{prefix}bc_anomaly", pt.dot(basis_data, coordinates), dims=state_dim)
    signal = pm.Deterministic(f"{prefix}mu_bc_anomaly", pt.dot(mass_data, state), dims=output_dim)
    response = xr.dot(mass, basis, dim="bc_region").transpose(output_dim, "bc_contrast").assign_attrs(units="1")
    return BoundaryAnomalies(state, signal, AffineBlock(response, coordinates), coordinates)


def add_independent_anomalies(
    mass: xr.DataArray, mass_data: TensorVariable, scale: float,
    *, output_dim: str, prefix: str, owner: str | None,
) -> BoundaryAnomalies:
    """Sample anomalies directly on the boundary states, contributing ``M z``.

    Args:
        mass: Materialized dimensionless ``M`` on ``(output_dim, bc_region)``.
        mass_data: The same ``M`` registered in the active model.
        scale: Positive Gaussian width of each state anomaly in observation
            concentration units.
        output_dim: Registered observation dimension.
        prefix: Graph variable prefix.
        owner: Namespace for boundary-state coordinates.

    Returns:
        State anomalies, their response, affine block and sampled states.
        Call inside a registered model. No basis or additional latent is built.
    """
    state_dim = str(component_coords(mass, owner, observation_dim=output_dim).dims[1])
    state = pm.Normal(f"{prefix}bc_anomaly", mu=0.0, sigma=scale, dims=state_dim)
    signal = pm.Deterministic(f"{prefix}mu_bc_anomaly", pt.dot(mass_data, state), dims=output_dim)
    return BoundaryAnomalies(state, signal, AffineBlock(mass.assign_attrs(units="1"), state), state)


def anomaly_outputs(structure: AnomalyStructure, *, sampled: bool, concentration: str) -> BoundaryOutputs:
    """Declare additive anomaly outputs before constructing a graph.

    Args:
        structure: Selected additive anomaly formulation.
        sampled: Whether the scale and number of coordinates permit sampling.
        concentration: Observation concentration units.

    Returns:
        Unprefixed roles and units for the selected formulation. Weighted
        anomalies retain scale/weight roles even when unsampled so trace
        reconstruction can identify intentionally zero anomalies.
    """
    roles = {}
    units = {"bc_anomaly_scale": concentration}
    if structure == "weighted":
        roles.update(boundary_anomaly_scale="bc_anomaly_scale", boundary_centering_weights="bc_centering_weights")
        units["bc_centering_weights"] = "1"
    if sampled:
        roles.update(boundary_correction_anomaly="bc_anomaly", boundary_anomaly_concentration="mu_bc_anomaly")
        units.update(bc_anomaly=concentration, mu_bc_anomaly=concentration)
        if structure != "independent":
            units.update(bc_anomaly_latent=concentration, bc_anomaly_basis="1")
    return BoundaryOutputs(roles, units)
