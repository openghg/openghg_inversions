"""Additive boundary corrections in observation concentration units.

With centered transport ``A``, ``M`` and ``c`` the absolute boundary
contribution is ``A 1 + c M 1 + M d``. The corrections ``d`` live on the bound
``(bc_curtain, bc_period)`` states and are transported only by the
dimensionless ``M``; they are never multiplied by ``c`` or ``A``. The
likelihood omits the fixed ``c M 1``.

``d`` is a mean shift plus anomalies, both in observation concentration units.
The mean shift is one scalar ``m 1`` or, with ``mean_by_period``, one shift per
boundary period; either responds through ``M``, so a whole-window shift
contributes ``m M 1``. The anomaly structure is part of the prior:

``"weighted"``
    Anomalies with zero weighted mean over every state,
    ``d = (Q - 1 wᵀQ) z`` for a Helmert basis ``Q`` (see
    https://mc-stan.org/docs/stan-users-guide/regression.html#parameterizing-centered-vectors).
``"period_contrasts"``
    Orthonormal Helmert contrasts of the curtains within each period, so every
    period's equal-weight curtain mean is left to the mean shift. With ``K``
    curtains and ``P`` periods there are ``P (K - 1)`` coordinates rather
    than the ``K P - 1`` of weighted centering, and a different covariance.
``"independent"``
    One independent anomaly per state. Without a mean shift, curtain
    contrasts with scale ``s`` plus a mean with scale ``s / sqrt(K)`` give
    exactly the same covariance.

The anomaly coordinates ``z`` are independent zero-mean Gaussians with the
explicit ``anomaly_scale``; nothing is calibrated from the data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

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

AnomalyStructure = Literal["weighted", "period_contrasts", "independent"]


@dataclass(frozen=True)
class AdditiveChoice:
    """Data-free additive boundary choice.

    Attributes:
        mean_prior: Prior for the mean shift, in observation concentration
            units, or ``None`` for no mean shift (sampled period contrasts or
            independent anomalies only).
            It is scalar, or one shift per period with ``mean_by_period``.
        anomaly_scale: Finite non-negative Gaussian scale of the anomaly
            coordinates. Zero keeps only the mean shift.
        anomalies: Anomaly structure: ``"weighted"`` (requires prepared
            centering weights), ``"period_contrasts"`` or ``"independent"``.
        mean_by_period: Fit one independent mean shift per boundary period.
        frequency: Optional boundary period frequency.
        anchor_time: Optional origin for fixed-duration periods.
    """

    mean_prior: Mapping[str, Any] | None
    anomaly_scale: float
    anomalies: AnomalyStructure = "weighted"
    mean_by_period: bool = False
    frequency: str | None = None
    anchor_time: DatetimeLike | None = None

    def __post_init__(self) -> None:
        if self.mean_prior is not None:
            object.__setattr__(self, "mean_prior", dict(self.mean_prior))
        validate_centered_boundary_periods(self.frequency, self.anchor_time)
        if self.anomalies not in get_args(AnomalyStructure):
            raise ValueError(f"Unknown boundary anomaly structure {self.anomalies!r}.")
        if not np.isfinite(self.anomaly_scale) or self.anomaly_scale < 0:
            raise ValueError("bc_anomaly_scale must be finite and non-negative.")
        if self.mean_prior is None and (self.anomalies == "weighted" or self.anomaly_scale == 0):
            raise ValueError("Only sampled period-contrast or independent anomalies may omit the mean shift prior.")
        if self.mean_by_period and (self.mean_prior is None or self.frequency is None):
            raise ValueError("bc_mean_by_period requires a mean shift prior and a boundary frequency.")

    def bind(self, observations: xr.DataArray, transport: xr.DataArray | CenteredOperands) -> AdditiveBoundary:
        """Bind centered transport (and any centering weights) to one fit.

        Weighted anomalies stack the prepared weights on ``(bc_curtain,
        bc_period)`` onto the bound ``bc_region`` states; their labels must
        match exactly, so a weight calendar that differs from the fit's
        periods is rejected. Other structures ignore prepared weights.
        """
        if not isinstance(transport, CenteredOperands):
            raise ValueError("Additive boundaries require centered operands.")
        data = transport.bind(observations, frequency=self.frequency, anchor_time=self.anchor_time)
        if self.anomalies != "weighted":
            return AdditiveBoundary(self, data)
        if transport.weights is None:
            raise ValueError("Weighted additive boundary anomalies require bc_centering_weights.")
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
    """Additive choice bound to one fit's borrowed centered operands.

    ``data`` holds ``H_bc_anomaly``, ``H_bc_mass`` and ``bc_reference`` on the
    bound ``bc_region`` states, plus ``bc_centering_weights`` for weighted
    anomalies. Materialize it with the other model inputs before :meth:`add`.
    """

    choice: AdditiveChoice
    data: xr.Dataset

    @property
    def has_anomalies(self) -> bool:
        """Whether anomalies are sampled: a positive scale and some coordinates."""
        states = self.data.indexes["bc_region"]
        count = {
            "weighted": len(states) - 1,
            "period_contrasts": len(states) - states.get_level_values("bc_period").nunique(),
            "independent": len(states),
        }[self.choice.anomalies]
        return self.choice.anomaly_scale > 0 and count > 0

    def _anomaly_basis(self, weights: np.ndarray | None) -> xr.DataArray:
        """Basis ``B`` mapping anomaly coordinates to every bound state.

        ``weights`` are the normalized centering weights of weighted anomalies.
        """
        periods = self.data.bc_period
        count = periods.size
        q: np.ndarray
        if self.choice.anomalies == "weighted":
            assert weights is not None
            q = helmert(count, full=False).T
            q = q - np.ones((count, 1)) @ (weights @ q)[None, :]
        elif self.choice.anomalies == "period_contrasts":
            columns = []
            for label in pd.unique(periods.values):
                members = np.flatnonzero(periods.values == label)
                block = np.zeros((count, members.size - 1))
                block[members] = helmert(members.size, full=False).T
                columns.append(block)
            q = np.hstack(columns)
        else:
            q = np.eye(count)
        return xr.DataArray(
            q, dims=("bc_region", "bc_contrast"), coords={"bc_contrast": np.arange(q.shape[1])},
            attrs={"units": "1"},
        ).assign_coords(self.data.bc_region.coords)

    def anomaly_factor(self) -> xr.DataArray | None:
        """Observation-space anomaly design scaled to standard-normal coordinates.

        Returns ``s M B`` on ``(observation, bc_contrast)`` in concentration
        units, so the anomaly concentration is this factor times independent
        standard normals; ``None`` without anomalies. The factor stays lazy
        with the transport. Weighted anomalies require materialized weights.
        """
        if not self.has_anomalies:
            return None
        weights = None
        if self.choice.anomalies == "weighted":
            values = self.data.bc_centering_weights
            if is_dask_collection(values.data):
                raise ValueError("Materialize bc_centering_weights before forming a weighted anomaly design.")
            weights = (values / values.sum()).values
        design = xr.dot(self.data.H_bc_mass, self._anomaly_basis(weights), dim="bc_region")
        return (design * self.choice.anomaly_scale).assign_attrs(units=self.data.bc_reference.attrs["units"])

    def add(self, *, output_dim: str = "nmeasure", prefix: str = "", owner: str | None = "bc") -> AffineConcentrationTerm:
        """Add the mean shift, anomalies and boundary concentration to the current model.

        Its affine blocks are the mean response ``M 1`` (or ``M`` summed over
        each period's states) and, with anomalies, ``M B`` for the anomaly
        basis ``B``.

        Args:
            output_dim: Observation dimension registered by the model.
            prefix: Prefix for graph variable names.
            owner: Namespace for labelled boundary-state coordinates.

        Raises:
            ValueError: If ``data`` is still lazy, materialized operands or
                weights are invalid, or a whole-window mean prior is not scalar.
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
        if not np.isfinite(reference_shift.values).all():
            raise ValueError("Boundary reference shift must be finite.")
        fixed = anomaly.sum("bc_region", skipna=False).assign_attrs(anomaly.attrs)
        count = mass.sizes["bc_region"]
        periods = mass.bc_period

        def register(array: xr.DataArray, name: str) -> TensorVariable:
            return add_model_data(component_coords(array, owner, observation_dim=output_dim), f"{prefix}{name}")

        a = register(anomaly, "H_bc_anomaly")
        m = register(mass, "H_bc_mass")
        register(reference, "bc_reference")
        register(xr.DataArray(choice.anomaly_scale), "bc_anomaly_scale")
        state_dim = str(component_coords(mass, owner, observation_dim=output_dim).dims[1])
        latents: tuple[TensorVariable, ...] = ()
        blocks = []
        mean_state = pt.zeros((count,))
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
            latents += (get_model_latent(mean, f"{prefix}bc_mean_shift"),)
            response = xr.dot(mass, indicator, dim="bc_region").assign_attrs(units="1")
            blocks.append(AffineBlock(response, coefficients))
            mean_signal = pm.Deterministic(f"{prefix}mu_bc_mean_shift", pt.dot(m, mean_state), dims=output_dim)
        anomaly_state = pt.zeros((count,))
        anomaly_signal = pt.zeros((mass.sizes[output_dim],))
        if choice.anomalies == "weighted":
            weights = self.data.bc_centering_weights
            if not weights.size or not np.isfinite(weights.values).all() or not (weights.values > 0).all():
                raise ValueError("bc_centering_weights must be finite and strictly positive.")
            weights = weights / weights.max()
            weights = (weights / weights.sum()).assign_attrs(units="1")
            register(weights, "bc_centering_weights")
        if self.has_anomalies:
            basis = self._anomaly_basis(weights.values if choice.anomalies == "weighted" else None)
            # Registering the basis also registers its contrast coordinate.
            basis_data = register(basis, "bc_anomaly_basis")
            contrast_dim = str(component_coords(basis, owner).dims[1])
            contrasts = pm.Normal(f"{prefix}bc_anomaly_latent", mu=0.0, sigma=choice.anomaly_scale, dims=contrast_dim)
            latents += (contrasts,)
            anomaly_state = pm.Deterministic(f"{prefix}bc_anomaly", pt.dot(basis_data, contrasts), dims=state_dim)
            anomaly_signal = pm.Deterministic(f"{prefix}mu_bc_anomaly", pt.dot(m, anomaly_state), dims=output_dim)
            blocks.append(AffineBlock(
                xr.dot(mass, basis, dim="bc_region").transpose(output_dim, "bc_contrast").assign_attrs(units="1"),
                contrasts,
            ))
        pm.Deterministic(f"{prefix}bc_correction", mean_state + anomaly_state, dims=state_dim)
        absolute_reference = register(
            (anomaly.sum("bc_region") + reference_shift).assign_attrs(anomaly.attrs), "mu_bc_reference",
        )
        output = pm.Deterministic(
            f"{prefix}mu_bc", absolute_reference + mean_signal + anomaly_signal, dims=output_dim,
        )
        return AffineConcentrationTerm(
            output, pt.sum(a, axis=1) + mean_signal + anomaly_signal, reference_shift,
            fixed_contribution=fixed, blocks=tuple(blocks), sampled_rvs=latents,
        )

    def outputs(self, prefix: str = "") -> BoundaryOutputs:
        """Declare roles, units and metadata for the graph built by :meth:`add`.

        Weighted anomalies with a positive scale and several states always
        declare anomaly outputs, so missing anomaly draws remain errors.
        Otherwise their zero anomalies are reconstructed from the saved scale
        and weights.
        """
        choice = self.choice
        concentration = self.data.bc_reference.attrs["units"]
        roles = {
            "boundary_concentration": "mu_bc",
            "boundary_anomaly": "H_bc_anomaly",
            "boundary_mass": "H_bc_mass",
            "boundary_reference": "bc_reference",
            "boundary_correction": "bc_correction",
            "boundary_reference_concentration": "mu_bc_reference",
        }
        units = {name: concentration for name in (
            "mu_bc", "H_bc_anomaly", "bc_reference", "bc_reference_shift", "boundary_reference_shift",
            "bc_anomaly_scale", "bc_correction", "mu_bc_reference",
        )}
        units["H_bc_mass"] = "1"
        prior = choice.mean_prior
        if prior is not None:
            roles.update(boundary_mean_shift="bc_mean_shift", boundary_mean_shift_concentration="mu_bc_mean_shift")
            units.update(bc_mean_shift=concentration, mu_bc_mean_shift=concentration)
            if prior.get("reparameterise") and str(prior.get("pdf", "")).lower() == "lognormal":
                units["bc_mean_shift_latent"] = "1"
            if choice.mean_by_period:
                units["bc_mean_indicator"] = "1"
        if choice.anomalies == "weighted":
            roles.update(boundary_anomaly_scale="bc_anomaly_scale", boundary_centering_weights="bc_centering_weights")
            units["bc_centering_weights"] = "1"
        if self.has_anomalies:
            roles.update(boundary_correction_anomaly="bc_anomaly", boundary_anomaly_concentration="mu_bc_anomaly")
            units.update({name: concentration for name in ("bc_anomaly", "bc_anomaly_latent", "mu_bc_anomaly")})
            units["bc_anomaly_basis"] = "1"
        metadata = {
            "boundary_formulation": "additive",
            "boundary_representation": "centered-v1",
            "boundary_anomalies": choice.anomalies,
            "boundary_mean": "none" if prior is None else "by_period" if choice.mean_by_period else "whole_window",
            "bc_frequency": choice.frequency,
            "bc_anchor_time": None if choice.anchor_time is None else str(choice.anchor_time),
        }
        return BoundaryOutputs(roles, units, metadata).prefixed(prefix)


__all__ = ["AdditiveBoundary", "AdditiveChoice", "AnomalyStructure"]
