"""Additive anomaly formulations in boundary-state concentration units.

Weighted and per-period anomalies use distinct contrast bases ``B`` and
contribute ``M B z``. Independent anomalies sample the states directly and
contribute ``M z``. :func:`anomaly_basis` is the one place that maps a
structure to its basis; the lazy observation-space factor and the graph both
use it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.tensor.variable import TensorVariable
from scipy.linalg import helmert

from openghg_inversions.models.components import AffineBlock
from openghg_inversions.models.coords import add_model_data, component_coords

from .outputs import BoundaryCorrection, BoundaryOutputs

AnomalyStructure = Literal["weighted", "period_contrasts", "independent"]


def weighted_anomaly_basis(weights: xr.DataArray) -> xr.DataArray:
    """Build the globally weighted contrast basis ``Q - 1 wᵀQ``.

    Args:
        weights: Materialized, normalized weights on ``bc_region`` with the
            bound curtain/period state coordinates.

    Returns:
        Dimensionless basis on ``(bc_region, bc_contrast)`` with ``N - 1``
        columns for ``N`` states. Its weighted state mean is zero.
    """
    helmert_basis = helmert(weights.size, full=False).T
    q = helmert_basis - np.ones((weights.size, 1)) @ (weights.values @ helmert_basis)[None, :]
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


def anomaly_basis(
    structure: AnomalyStructure, states: xr.DataArray, weights: xr.DataArray | None,
) -> xr.DataArray | None:
    """Select the contrast-to-state basis ``B`` of an anomaly structure.

    Args:
        structure: Selected additive anomaly formulation.
        states: Bound ``bc_region`` coordinate with a ``bc_period`` level.
        weights: Normalized centering weights from
            :meth:`~.contracts.BoundAdditiveTransport.normalized_weights`; required for ``"weighted"``
            and ignored otherwise.

    Returns:
        :func:`weighted_anomaly_basis` or :func:`period_contrast_basis`, or
        ``None`` for independent anomalies, which are sampled on the states.

    Raises:
        ValueError: If weighted anomalies have no weights.
    """
    if structure == "independent":
        return None
    if structure == "period_contrasts":
        return period_contrast_basis(states)
    if weights is None:
        raise ValueError("Weighted anomalies require normalized bc_centering_weights.")
    return weighted_anomaly_basis(weights)


def add_anomalies(
    mass: xr.DataArray, mass_data: TensorVariable, basis: xr.DataArray | None, scale: float,
    *, output_dim: str, prefix: str, owner: str | None,
) -> BoundaryCorrection:
    """Sample zero-mean Gaussian anomalies and transport them through ``M``.

    With a basis, contrast coefficients ``z`` are sampled as
    ``bc_anomaly_latent``, the state anomalies are ``bc_anomaly = B z`` and
    the block design is ``M B``. Without one, ``bc_anomaly`` is sampled
    directly on the states and the block design is ``M``; no basis or extra
    latent is registered. Must be called inside a registered model, with a
    positive ``scale`` and at least one coefficient; :class:`~openghg_inversions.models.boundary.AdditiveBoundary`
    calls it only when ``has_anomalies`` establishes both.

    Args:
        mass: Materialized dimensionless ``M`` on ``(output_dim, bc_region)``.
        mass_data: The same ``M`` registered in the active model.
        basis: Basis from :func:`anomaly_basis`, aligned with ``mass``.
        scale: Gaussian width of each coefficient in observation
            concentration units.
        output_dim: Registered observation dimension.
        prefix: Graph variable prefix.
        owner: Namespace for boundary-state and contrast coordinates.

    Returns:
        State anomalies, their response ``mu_bc_anomaly``, the affine block
        and the sampled coefficients.
    """
    if basis is None:
        state_dim = str(component_coords(mass, owner, observation_dim=output_dim).dims[1])
        state = pm.Normal(f"{prefix}bc_anomaly", mu=0.0, sigma=scale, dims=state_dim)
        signal = pm.Deterministic(f"{prefix}mu_bc_anomaly", pt.dot(mass_data, state), dims=output_dim)
        return BoundaryCorrection(state, signal, AffineBlock(mass.assign_attrs(units="1"), state), state)
    owned_basis = component_coords(basis, owner)
    basis_data = add_model_data(owned_basis, f"{prefix}bc_anomaly_basis")
    state_dim, contrast_dim = map(str, owned_basis.dims)
    coefficients = pm.Normal(f"{prefix}bc_anomaly_latent", mu=0.0, sigma=scale, dims=contrast_dim)
    state = pm.Deterministic(f"{prefix}bc_anomaly", pt.dot(basis_data, coefficients), dims=state_dim)
    signal = pm.Deterministic(f"{prefix}mu_bc_anomaly", pt.dot(mass_data, state), dims=output_dim)
    response = xr.dot(mass, basis, dim="bc_region").transpose(output_dim, "bc_contrast").assign_attrs(units="1")
    return BoundaryCorrection(state, signal, AffineBlock(response, coefficients), coefficients)


def anomaly_outputs(structure: AnomalyStructure, *, sampled: bool, concentration: str) -> BoundaryOutputs:
    """Declare additive anomaly outputs before constructing a graph.

    Args:
        structure: Selected additive anomaly formulation.
        sampled: Whether the scale and number of coordinates permit sampling.
        concentration: Observation concentration units.

    Returns:
        Unprefixed roles and units. The anomaly state and contribution are
        always declared, as sampled values or recorded zeros.
    """
    roles = {"boundary_correction_anomaly": "bc_anomaly", "boundary_anomaly_concentration": "mu_bc_anomaly"}
    units = {"bc_anomaly_scale": concentration, "bc_anomaly": concentration, "mu_bc_anomaly": concentration}
    if structure == "weighted":
        units["bc_centering_weights"] = "1"
    if sampled and structure != "independent":
        units.update(bc_anomaly_latent=concentration, bc_anomaly_basis="1")
    return BoundaryOutputs(roles, units)
