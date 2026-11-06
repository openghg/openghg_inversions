"""Additive boundary corrections with weighted-centred anomalies.

The additive boundary replaces fitted multiplicative scales. With centered
transport ``A``, ``M`` and ``c`` (:class:`~.operands.CenteredOperands`; see
:doc:`/usage/centered_boundary_conditions`) its absolute contribution is
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

from .operands import CenteredOperands, check_bound_transport, materialized_reference_shift
from .outputs import BoundaryOutputs


@dataclass(frozen=True)
class AdditiveChoice:
    """Prior specification for additive boundary corrections, before binding.

    A choice holds the scientific options (mean shift prior, anomaly width
    and boundary period calendar) without bound observations or transport
    arrays. :meth:`bind` pairs it with one fit's observations and centered
    transport.

    Attributes:
        mean_prior: Scalar prior for the common mean shift over the inversion
            window, in observation concentration units.
        anomaly_scale: Finite non-negative Gaussian standard deviation of the
            anomaly coordinates before weighted centring, in concentration
            units. Zero, or a single boundary state, keeps only the mean shift.
        frequency: Boundary period frequency, with the syntax and calendar of
            ``bc_freq``: ``None`` for one period over the window,
            ``"monthly"`` for calendar months, or a positive fixed-duration
            string such as ``"7D"``.
        anchor_time: Origin of fixed-duration periods; only valid with a
            fixed-duration ``frequency``. ``None`` uses the earliest selected
            observation of the fit. The RHIME parameter adapter instead
            defaults ``bc_freq_anchor`` to the configured start date.

    Raises:
        ValueError: On construction, for a negative or non-finite scale, or an
            invalid frequency or anchor.
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

    def bind(self, observations: xr.DataArray, transport: CenteredOperands) -> AdditiveBoundary:
        """Bind centered transport and its centering weights to one fit.

        Args:
            observations: One channel's observations with its observation axis
                first. Its labels select and order the transport rows; its
                ``time`` values assign rows to boundary periods.
            transport: Prepared centered operands on their raw ``nmeasure``
                or ``time`` axis, with ``weights`` on
                ``(bc_curtain, bc_period)``.

        Returns:
            The bound boundary. ``H_bc_anomaly`` and ``H_bc_mass`` lie on
            ``(observation, bc_region)`` with a ``(bc_curtain, bc_period)``
            state index, beside the stored scalar ``bc_reference`` and
            ``bc_centering_weights`` on ``bc_region``. Numerical payloads stay
            borrowed and lazy.

        Raises:
            ValueError: If PyTensor is not configured for float64, periods or
                indexed timestamps conflict with the observations, there are
                no weights, or the weights' curtain and period labels differ
                from the bound states (for example because they were prepared
                with another calendar).
        """
        if transport.weights is None:
            raise ValueError("Additive boundaries require bc_centering_weights.")
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

    Attributes:
        choice: The resolved additive choice.
        data: ``H_bc_anomaly`` (concentration units) and ``H_bc_mass``
            (dimensionless) on ``(observation, bc_region)``, where
            ``bc_region`` indexes ``(bc_curtain, bc_period)`` states; scalar
            ``bc_reference`` in the anomaly's units; and positive
            ``bc_centering_weights`` on ``bc_region``. The arrays are borrowed
            and may be Dask-backed. Materialize them jointly with the other
            model inputs before :meth:`add`; rows may be padded or subset with
            ``dataclasses.replace``, keeping that layout.

    Raises:
        ValueError: On construction, if ``data`` lacks a variable or does not
            have that layout. Only dimensions are checked.
    """

    choice: AdditiveChoice
    data: xr.Dataset

    def __post_init__(self) -> None:
        check_bound_transport(self.data, ("H_bc_anomaly", "H_bc_mass"), state_dim="bc_region")
        if "bc_centering_weights" not in self.data or self.data.bc_centering_weights.dims != ("bc_region",):
            raise ValueError("Bound additive data needs bc_centering_weights on bc_region; bind the choice again.")

    @property
    def observation_dim(self) -> str:
        """Observation dimension of the bound transport."""
        return str(self.data.H_bc_anomaly.dims[0])

    @property
    def has_anomalies(self) -> bool:
        """Whether anomalies are sampled (positive scale and several states)."""
        return self.choice.anomaly_scale > 0 and self.data.sizes["bc_region"] > 1

    def add(self, *, output_dim: str = "nmeasure", prefix: str = "", owner: str | None = "bc") -> AffineConcentrationTerm:
        """Add the mean shift, anomalies and boundary concentration to the current model.

        Must be called inside an active PyMC model that has the coordinate
        registry, after ``data`` has been materialized jointly with the other
        model inputs. It registers the transport, reference, scale, weights
        and any anomaly basis as model data, plus the sampled variables and
        their deterministic responses. The corrections multiply only the
        dimensionless transport ``M``; they are never multiplied by ``c`` or
        ``A``.

        Args:
            output_dim: Observation dimension registered by the model.
            prefix: Prefix for graph variable names.
            owner: Namespace for component-owned boundary-state coordinates;
                ``None`` keeps the local names such as ``bc_region``. Graph
                names are controlled separately by ``prefix``.

        Returns:
            The boundary term: ``output`` is the absolute boundary
            concentration ``mu_bc``; the likelihood contribution omits the
            fixed ``c M 1``, which is the reference shift; the fixed
            contribution is ``A 1``; the affine blocks are the mean response
            ``M 1`` followed, with anomalies, by ``M (Q - 1 wᵀQ)`` with their
            physical coefficients; and the sampled variables are the mean
            latent and anomaly coordinates.

        Raises:
            ValueError: If ``output_dim`` is not the bound observation
                dimension, ``data`` is still lazy, materialized operands or
                weights are invalid, or the mean prior is not scalar.
            RuntimeError: If the model lacks the coordinate registry.
        """
        if output_dim != self.observation_dim:
            raise ValueError(f"output_dim {output_dim!r} is not the bound observation dimension {self.observation_dim!r}.")
        if any(is_dask_collection(value) for value in self.data.data_vars.values()):
            raise ValueError("Materialize boundary data jointly with model inputs before add().")

        # Values become checkable only after joint materialization.
        anomaly = self.data.H_bc_anomaly
        mass = self.data.H_bc_mass
        reference = self.data.bc_reference
        weights = self.data.bc_centering_weights
        reference_shift = materialized_reference_shift(anomaly, mass, reference).rename(f"{prefix}bc_reference_shift")
        if not weights.size or not np.isfinite(weights.values).all() or not (weights.values > 0).all():
            raise ValueError("bc_centering_weights must be finite and strictly positive.")
        weights = weights / weights.max()
        weights = (weights / weights.sum()).assign_attrs(units="1")

        # Weighted-centred contrast basis (Q - 1 wᵀQ) for a Helmert basis Q.
        count = weights.size
        q = helmert(count, full=False).T if self.has_anomalies else np.empty((count, 0))
        basis = xr.DataArray(
            q - np.ones((count, 1)) @ (weights.values @ q)[None, :], dims=("bc_region", "bc_contrast"),
            coords={"bc_region": weights.bc_region, "bc_contrast": np.arange(q.shape[1])}, attrs={"units": "1"},
        )
        fixed = anomaly.sum("bc_region", skipna=False).assign_attrs(anomaly.attrs)

        def register(array: xr.DataArray, name: str) -> TensorVariable:
            return add_model_data(component_coords(array, owner, observation_dim=output_dim), f"{prefix}{name}")

        a = register(anomaly, "H_bc_anomaly")
        m = register(mass, "H_bc_mass")
        register(reference, "bc_reference")
        register(weights, "bc_centering_weights")
        register(xr.DataArray(self.choice.anomaly_scale), "bc_anomaly_scale")
        state_dim = str(component_coords(weights, owner).dims[0])

        # Sampled mean shift and, with anomalies, contrast coordinates.
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

        # Concentration responses: absolute output and centered likelihood term.
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
        """Declare the roles, units and metadata of this boundary's graph.

        Declarations depend only on the choice and the bound data, so they
        are available before :meth:`add` or sampling. A positive scale with
        several states always declares anomaly outputs, so missing anomaly
        draws remain errors. Otherwise the zero anomalies are reconstructed
        from the saved scale and weights.

        Args:
            prefix: Graph-name prefix used for :meth:`add`.

        Returns:
            Scientific roles mapped to graph variable names, graph variable
            units, and configuration and binding metadata. ``prefix`` is
            applied to both role keys and graph names; metadata is unprefixed.
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
            "mu_bc", "H_bc_anomaly", "bc_reference", "bc_mean_shift", "bc_anomaly_scale", "bc_correction",
            "mu_bc_reference", "mu_bc_mean_shift",
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
