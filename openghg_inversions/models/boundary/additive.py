"""Additive boundary corrections with weighted-centred anomalies.

The additive boundary replaces fitted multiplicative scales. With centered
transport ``A``, ``M`` and ``c`` its absolute contribution is
``A 1 + c M 1 + M (m 1 + d)``: a scalar common mean shift ``m`` plus anomalies
``d`` with zero weighted mean, both in observation concentration units. The
likelihood omits the fixed ``c M 1``. Anomalies use the non-redundant normal
coordinates ``d = (Q - 1 wᵀQ) z`` for a Helmert basis ``Q``, so they never
change the weighted mean; see
https://mc-stan.org/docs/stan-users-guide/regression.html#parameterizing-centered-vectors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from dask import is_dask_collection
from pytensor.tensor.variable import TensorVariable
from scipy.linalg import helmert

from openghg_inversions.boundary_sensitivity import validate_centered_boundary_periods
from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models.components import AffineBlock, AffineConcentrationTerm, get_model_latent
from openghg_inversions.models.coords import add_model_data, component_coords
from openghg_inversions.models.priors import parse_prior

from .operands import CenteredOperands
from .outputs import BoundaryOutputs


@dataclass(frozen=True)
class AdditiveChoice:
    """Data-free additive boundary choice.

    Attributes:
        mean_prior: Scalar prior for the common mean shift, in observation
            concentration units.
        anomaly_scale: Finite non-negative Gaussian scale of the anomalies
            before weighted centring. Zero, or a single boundary state, keeps
            only the common mean.
        frequency: Optional boundary period frequency.
        anchor_time: Optional origin for fixed-duration periods.
    """

    mean_prior: Mapping[str, Any]
    anomaly_scale: float
    frequency: str | None = None
    anchor_time: DatetimeLike | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean_prior", dict(self.mean_prior))
        validate_centered_boundary_periods(self.frequency, self.anchor_time)
        if not np.isfinite(self.anomaly_scale) or self.anomaly_scale < 0:
            raise ValueError("bc_anomaly_scale must be finite and non-negative.")

    def bind(self, observations: xr.DataArray, transport: xr.DataArray | CenteredOperands) -> AdditiveBoundary:
        """Bind centered transport and its centering weights to one fit.

        Weights on ``(bc_curtain, bc_period)`` are stacked onto the bound
        ``bc_region`` states; their labels must match exactly, so a weight
        calendar that differs from the fit's periods is rejected.
        """
        if not isinstance(transport, CenteredOperands) or transport.weights is None:
            raise ValueError("Additive boundaries require centered operands with bc_centering_weights.")
        data = transport.bind(observations, frequency=self.frequency, anchor_time=self.anchor_time)
        weights = transport.weights
        periods = weights.indexes["bc_period"]
        if isinstance(periods, pd.DatetimeIndex):
            # Period starts from the fit calendar are nanosecond timestamps.
            weights = weights.assign_coords(bc_period=periods.as_unit("ns"))
        weights = weights.stack(bc_region=("bc_curtain", "bc_period"))
        xr.align(data.H_bc_anomaly, weights, join="exact", copy=False, exclude={str(observations.dims[0])})
        # Labels match exactly, so the bound state index is reused for the weights.
        return AdditiveBoundary(self, data.assign(bc_centering_weights=weights.variable))


@dataclass(frozen=True)
class AdditiveBoundary:
    """Additive choice bound to one fit's borrowed centered operands and weights.

    ``data`` holds ``H_bc_anomaly``, ``H_bc_mass``, ``bc_reference`` and
    ``bc_centering_weights`` on the bound ``bc_region`` states. Materialize it
    with the other model inputs before :meth:`add`.
    """

    choice: AdditiveChoice
    data: xr.Dataset

    @property
    def has_anomalies(self) -> bool:
        """Whether anomalies are sampled (positive scale and several states)."""
        return self.choice.anomaly_scale > 0 and self.data.sizes["bc_region"] > 1

    def add(self, *, output_dim: str = "nmeasure", prefix: str = "", owner: str | None = "bc") -> AffineConcentrationTerm:
        """Add the mean shift, anomalies and boundary concentration to the current model.

        The corrections multiply only the dimensionless transport ``M``; they
        are never multiplied by ``c`` or ``A``. Its affine blocks are the mean
        response ``M 1`` (first) and, with anomalies, ``M (Q - 1 wᵀQ)``.

        Args:
            output_dim: Observation dimension registered by the model.
            prefix: Prefix for graph variable names.
            owner: Namespace for labelled boundary-state coordinates.

        Raises:
            ValueError: If ``data`` is still lazy, materialized operands or
                weights are invalid, or the mean prior is not scalar.
        """
        if any(is_dask_collection(value) for value in self.data.data_vars.values()):
            raise ValueError("Materialize boundary data jointly with model inputs before add().")
        anomaly = self.data.H_bc_anomaly.transpose(output_dim, "bc_region")
        mass = self.data.H_bc_mass.transpose(output_dim, "bc_region")
        reference = self.data.bc_reference
        weights = self.data.bc_centering_weights
        if not np.isfinite(reference.values).all() or not np.isfinite(anomaly.values).all():
            raise ValueError("Centered bc_reference and boundary anomaly must be finite.")
        if not np.isfinite(mass.values).all() or (mass.values < 0).any() or not (mass.values > 0).any():
            raise ValueError("H_bc_mass must be finite, non-negative and not all zero.")
        if not weights.size or not np.isfinite(weights.values).all() or not (weights.values > 0).all():
            raise ValueError("bc_centering_weights must be finite and strictly positive.")
        weights = weights / weights.max()
        weights = (weights / weights.sum()).assign_attrs(units="1")
        count = weights.size
        q = helmert(count, full=False).T if self.has_anomalies else np.empty((count, 0))
        basis = xr.DataArray(
            q - np.ones((count, 1)) @ (weights.values @ q)[None, :], dims=("bc_region", "bc_contrast"),
            coords={"bc_region": weights.bc_region, "bc_contrast": np.arange(q.shape[1])}, attrs={"units": "1"},
        )
        reference_shift = (reference * mass.sum("bc_region", skipna=False)).rename(
            f"{prefix}bc_reference_shift"
        ).assign_attrs(reference.attrs)
        if not np.isfinite(reference_shift.values).all():
            raise ValueError("Boundary reference shift must be finite.")
        fixed = anomaly.sum("bc_region", skipna=False).assign_attrs(anomaly.attrs)

        def register(array: xr.DataArray, name: str) -> TensorVariable:
            return add_model_data(component_coords(array, owner, observation_dim=output_dim), f"{prefix}{name}")

        a = register(anomaly, "H_bc_anomaly")
        m = register(mass, "H_bc_mass")
        register(reference, "bc_reference")
        register(weights, "bc_centering_weights")
        register(xr.DataArray(self.choice.anomaly_scale), "bc_anomaly_scale")
        state_dim = str(component_coords(weights, owner).dims[0])
        mean = parse_prior(f"{prefix}bc_mean_shift", dict(self.choice.mean_prior))
        if mean.ndim != 0:
            raise ValueError("bc_mean_shift_prior must define one scalar over the inversion window.")
        latents: tuple[TensorVariable, ...] = (get_model_latent(mean, f"{prefix}bc_mean_shift"),)
        blocks = [AffineBlock(
            mass.sum("bc_region", skipna=False).expand_dims(bc_mean=[0], axis=1).assign_attrs(units="1"),
            pt.atleast_1d(mean),
        )]
        if self.has_anomalies:
            # Registering the basis also registers its contrast coordinate.
            basis_data = register(basis, "bc_anomaly_basis")
            contrast_dim = str(component_coords(basis, owner).dims[1])
            contrasts = pm.Normal(
                f"{prefix}bc_anomaly_latent", mu=0.0, sigma=self.choice.anomaly_scale, dims=contrast_dim,
            )
            latents += (contrasts,)
            corrections = pm.Deterministic(f"{prefix}bc_anomaly", pt.dot(basis_data, contrasts), dims=state_dim)
            blocks.append(AffineBlock(
                xr.dot(mass, basis, dim="bc_region").transpose(output_dim, "bc_contrast").assign_attrs(units="1"),
                contrasts,
            ))
        else:
            corrections = pt.zeros((count,))
        pm.Deterministic(f"{prefix}bc_correction", mean + corrections, dims=state_dim)
        absolute_reference = register(
            (anomaly.sum("bc_region") + reference_shift).assign_attrs(anomaly.attrs), "mu_bc_reference",
        )
        mean_signal = pm.Deterministic(f"{prefix}mu_bc_mean_shift", pt.sum(m, axis=1) * mean, dims=output_dim)
        anomaly_signal = pt.dot(m, corrections)
        if self.has_anomalies:
            anomaly_signal = pm.Deterministic(f"{prefix}mu_bc_anomaly", anomaly_signal, dims=output_dim)
        output = pm.Deterministic(
            f"{prefix}mu_bc", absolute_reference + mean_signal + anomaly_signal, dims=output_dim,
        )
        return AffineConcentrationTerm(
            output, pt.sum(a, axis=1) + mean_signal + anomaly_signal, reference_shift,
            fixed_contribution=fixed, blocks=tuple(blocks), sampled_rvs=latents,
        )

    def outputs(self, prefix: str = "") -> BoundaryOutputs:
        """Declare roles, units and metadata for the graph built by :meth:`add`.

        A positive scale with several states always declares anomaly outputs,
        so missing anomaly draws remain errors. Otherwise the zero anomalies
        are reconstructed from the saved scale and weights.
        """
        concentration = self.data.bc_reference.attrs["units"]
        roles = {
            "boundary_concentration": "mu_bc",
            "boundary_anomaly": "H_bc_anomaly",
            "boundary_mass": "H_bc_mass",
            "boundary_reference": "bc_reference",
            "boundary_mean_shift": "bc_mean_shift",
            "boundary_anomaly_scale": "bc_anomaly_scale",
            "boundary_centering_weights": "bc_centering_weights",
            "boundary_correction": "bc_correction",
            "boundary_reference_concentration": "mu_bc_reference",
            "boundary_mean_shift_concentration": "mu_bc_mean_shift",
        }
        units = {name: concentration for name in (
            "mu_bc", "H_bc_anomaly", "bc_reference", "bc_reference_shift", "boundary_reference_shift",
            "bc_mean_shift", "bc_anomaly_scale", "bc_correction", "mu_bc_reference", "mu_bc_mean_shift",
        )}
        units.update(H_bc_mass="1", bc_centering_weights="1")
        prior = self.choice.mean_prior
        if prior.get("reparameterise") and str(prior.get("pdf", "")).lower() == "lognormal":
            units["bc_mean_shift_latent"] = "1"
        if self.has_anomalies:
            roles.update(boundary_correction_anomaly="bc_anomaly", boundary_anomaly_concentration="mu_bc_anomaly")
            units.update({name: concentration for name in ("bc_anomaly", "bc_anomaly_latent", "mu_bc_anomaly")})
            units["bc_anomaly_basis"] = "1"
        metadata = {
            "boundary_formulation": "additive",
            "boundary_representation": "centered-v1",
            "bc_frequency": self.choice.frequency,
            "bc_anchor_time": None if self.choice.anchor_time is None else str(self.choice.anchor_time),
        }
        return BoundaryOutputs(roles, units, metadata).prefixed(prefix)


__all__ = ["AdditiveBoundary", "AdditiveChoice"]
