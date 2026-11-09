"""Additive anomaly formulations in boundary-state concentration units.

Weighted and per-period anomalies use distinct contrast bases ``B`` and
contribute ``M B z``. Independent anomalies sample the states directly and
contribute ``M z``. :func:`anomaly_basis` is the one place that maps a
structure to its basis; the lazy observation-space factor and the graph both
use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from scipy.linalg import helmert

from openghg_inversions.models.components import AffineBlock
from openghg_inversions.models.coords import add_model_data, component_coords

from .centered import Correction, RegisteredSensitivity
from .contracts import normalize_centering_weights
from .outputs import Outputs

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
            :func:`~.contracts.normalize_centering_weights`; required for ``"weighted"``
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


@dataclass(frozen=True)
class Anomalies:
    """Zero-mean Gaussian anomalies of one bound additive boundary.

    With a basis, contrast coefficients ``z`` are sampled as
    ``bc_anomaly_latent``, the state anomalies are ``bc_anomaly = B z`` and
    the block sensitivity is ``M B``. Independent anomalies are sampled directly
    on the states, with block sensitivity ``M``. When nothing can be sampled,
    zero anomalies are registered once as model data instead of recorded with
    every draw; output readers broadcast them over the draws, so replay needs
    no policy.

    Attributes:
        structure: Selected additive anomaly formulation.
        scale: Gaussian width of each coordinate in observation concentration
            units.
        sampled: Whether the scale is positive and the structure has at least
            one coordinate on the bound states; see
            :attr:`~openghg_inversions.models.boundary.AdditiveBoundary.has_anomalies`.
    """

    structure: AnomalyStructure
    scale: float
    sampled: bool

    def weights(self, data: xr.Dataset) -> xr.DataArray | None:
        """Normalize the centering weights that weighted anomalies use.

        Args:
            data: Materialized bound sensitivity.

        Returns:
            Normalized weights for the ``"weighted"`` structure, else ``None``.
        """
        if self.structure != "weighted":
            return None
        return normalize_centering_weights(data.bc_centering_weights)

    def add(self, registered: RegisteredSensitivity) -> list[Correction]:
        """Register the width and any weights, then sample anomalies or record zeros.

        Args:
            registered: The family's registered sensitivity.

        Returns:
            No corrections when nothing is sampled; the zero anomalies are then
            model data. Otherwise one: the state
            anomalies, their response ``mu_bc_anomaly``, the affine block and
            the sampled coefficients.

        Raises:
            ValueError: If weighted-centering weights are invalid.
        """
        prefix, output_dim, state_dim = registered.prefix, registered.output_dim, registered.state_dim
        data = registered.data
        weights = self.weights(data)
        # Save the resolved width for replay; the priors use the scalar directly.
        registered.register(xr.DataArray(self.scale), "bc_anomaly_scale")
        if weights is not None:
            registered.register(weights, "bc_centering_weights")
        mass = data.H_bc_mass
        if not self.sampled:
            # Constant zeros: the same roles as sampled anomalies, stored once rather than per draw.
            units = registered.data.bc_reference.attrs["units"]
            registered.register(xr.zeros_like(mass.sum(output_dim)).assign_attrs(units=units), "bc_anomaly")
            registered.register(xr.zeros_like(mass.sum("bc_region")).assign_attrs(units=units), "mu_bc_anomaly")
            return []
        basis = anomaly_basis(self.structure, mass.bc_region, weights)
        if basis is None:
            state = pm.Normal(f"{prefix}bc_anomaly", mu=0.0, sigma=self.scale, dims=state_dim)
            signal = pm.Deterministic(f"{prefix}mu_bc_anomaly", pt.dot(registered.mass, state), dims=output_dim)
            return [Correction(state, signal, AffineBlock(mass.assign_attrs(units="1"), state), state)]
        owned_basis = component_coords(basis, registered.owner)
        basis_data = add_model_data(owned_basis, f"{prefix}bc_anomaly_basis")
        coefficients = pm.Normal(f"{prefix}bc_anomaly_latent", mu=0.0, sigma=self.scale, dims=str(owned_basis.dims[1]))
        state = pm.Deterministic(f"{prefix}bc_anomaly", pt.dot(basis_data, coefficients), dims=state_dim)
        signal = pm.Deterministic(f"{prefix}mu_bc_anomaly", pt.dot(registered.mass, state), dims=output_dim)
        response = xr.dot(mass, basis, dim="bc_region").transpose(output_dim, "bc_contrast").assign_attrs(units="1")
        return [Correction(state, signal, AffineBlock(response, coefficients), coefficients)]

    def outputs(self, concentration_units: str) -> Outputs:
        """Declare anomaly outputs before constructing a graph.

        Args:
            concentration_units: Observation concentration units.

        Returns:
            Unprefixed roles and units. The anomaly state and contribution are
            always declared, as sampled values or constant zeros.
        """
        roles = {"boundary_correction_anomaly": "bc_anomaly", "boundary_anomaly_concentration": "mu_bc_anomaly"}
        units = {name: concentration_units for name in ("bc_anomaly_scale", "bc_anomaly", "mu_bc_anomaly")}
        if self.structure == "weighted":
            units["bc_centering_weights"] = "1"
        if self.sampled and self.structure != "independent":
            units.update(bc_anomaly_latent=concentration_units, bc_anomaly_basis="1")
        return Outputs(roles, units)


__all__ = ["Anomalies", "AnomalyStructure", "anomaly_basis", "period_contrast_basis", "weighted_anomaly_basis"]
