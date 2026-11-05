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
    One independent anomaly per state. Curtain contrasts with scale ``s``
    plus one mean shift per period with scale ``s / sqrt(K)``
    (``mean_by_period``) give exactly the same covariance. A single
    whole-window mean does so only with one period; otherwise it correlates
    periods that independent anomalies leave uncorrelated.

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
from openghg_inversions.models.components import AffineBlock, AffineConcentrationTerm
from openghg_inversions.models.coords import add_model_data, component_coords

from .mean_shift import add_boundary_mean_shift, mean_shift_outputs
from .operands import CenteredOperands, materialized_reference_shift
from .outputs import BoundaryOutputs

AnomalyStructure = Literal["weighted", "period_contrasts", "independent"]


@dataclass(frozen=True)
class AdditiveChoice:
    """Prior specification for additive boundary corrections, before binding.

    A choice holds the scientific options (mean shift prior, anomaly
    structure and width, and boundary period calendar) without any arrays.
    :meth:`bind` pairs it with one fit's observations and centered transport.

    Attributes:
        mean_prior: Prior for the mean shift in observation concentration
            units, scalar or one shift per period with ``mean_by_period``; or
            ``None`` for no mean shift. ``None`` is allowed only for sampled
            period contrasts or independent anomalies.
        anomaly_scale: Finite non-negative Gaussian standard deviation of the
            anomaly coordinates, in concentration units. Zero keeps only the
            mean shift.
        anomalies: Anomaly structure: ``"weighted"`` (requires prepared
            centering weights), ``"period_contrasts"`` or ``"independent"``.
        mean_by_period: Fit one independent mean shift per boundary period;
            requires a mean prior and a ``frequency``.
        frequency: Boundary period frequency, with the syntax and calendar of
            ``bc_freq``: ``None`` for one period over the window,
            ``"monthly"`` for calendar months, or a positive fixed-duration
            string such as ``"7D"``.
        anchor_time: Origin of fixed-duration periods; only valid with a
            fixed-duration ``frequency``. ``None`` uses the earliest selected
            observation of the fit. The RHIME parameter adapter instead
            defaults ``bc_freq_anchor`` to the configured start date.

    Raises:
        ValueError: On construction, for an unknown anomaly structure, a
            negative or non-finite scale, a missing mean prior where one is
            required, ``mean_by_period`` without a mean prior or frequency,
            or an invalid frequency or anchor.
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

    def bind(self, observations: xr.DataArray, transport: CenteredOperands) -> AdditiveBoundary:
        """Bind centered transport, and any centering weights, to one fit.

        Args:
            observations: One channel's observations with its observation axis
                first. Its labels select and order the transport rows; its
                ``time`` values assign rows to boundary periods.
            transport: Prepared centered operands on their raw ``nmeasure``
                or ``time`` axis. Weighted anomalies also need its
                ``weights`` on ``(bc_curtain, bc_period)``; other structures
                ignore them.

        Returns:
            The bound boundary. ``H_bc_anomaly`` and ``H_bc_mass`` lie on
            ``(observation, bc_region)`` with a ``(bc_curtain, bc_period)``
            state index, beside the stored scalar ``bc_reference`` and, for
            weighted anomalies, ``bc_centering_weights`` on ``bc_region``.
            Numerical payloads stay borrowed and lazy.

        Raises:
            ValueError: If PyTensor is not configured for float64, periods or
                indexed timestamps conflict with the observations, weighted
                anomalies have no weights, or the weights' curtain and period
                labels differ from the bound states (for example because they
                were prepared with another calendar).
        """
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

    Attributes:
        choice: The resolved additive choice.
        data: ``H_bc_anomaly`` (concentration units) and ``H_bc_mass``
            (dimensionless) on ``(observation, bc_region)``, where
            ``bc_region`` indexes ``(bc_curtain, bc_period)`` states; scalar
            ``bc_reference`` in the anomaly's units; and, for weighted
            anomalies, positive ``bc_centering_weights`` on ``bc_region``.
            The arrays are borrowed and may be Dask-backed. Materialize them
            jointly with the other model inputs before :meth:`add`; rows may
            be padded or subset with ``dataclasses.replace``.
    """

    choice: AdditiveChoice
    data: xr.Dataset

    @property
    def has_anomalies(self) -> bool:
        """Whether anomalies are sampled.

        True when ``anomaly_scale`` is positive and the structure has at least
        one coordinate: ``N - 1`` weighted, ``N - P`` period contrasts or
        ``N`` independent, for ``N`` bound states in ``P`` periods.
        """
        states = self.data.indexes["bc_region"]
        count = {
            "weighted": len(states) - 1,
            "period_contrasts": len(states) - states.get_level_values("bc_period").nunique(),
            "independent": len(states),
        }[self.choice.anomalies]
        return self.choice.anomaly_scale > 0 and count > 0

    def _anomaly_basis(self, weights: np.ndarray | None) -> xr.DataArray | None:
        """Basis ``B`` mapping anomaly coordinates to every bound state.

        ``weights`` are the normalized centering weights of weighted anomalies.
        Independent anomalies need no basis (``B = I``) and return ``None``.
        """
        if self.choice.anomalies == "independent":
            return None
        periods = self.data.bc_period
        count = periods.size
        q: np.ndarray
        if self.choice.anomalies == "weighted":
            assert weights is not None
            q = helmert(count, full=False).T
            q = q - np.ones((count, 1)) @ (weights @ q)[None, :]
        else:
            columns = []
            for label in pd.unique(periods.values):
                members = np.flatnonzero(periods.values == label)
                block = np.zeros((count, members.size - 1))
                block[members] = helmert(members.size, full=False).T
                columns.append(block)
            q = np.hstack(columns)
        return xr.DataArray(
            q, dims=("bc_region", "bc_contrast"), coords={"bc_contrast": np.arange(q.shape[1])},
            attrs={"units": "1"},
        ).assign_coords(self.data.bc_region.coords)

    def anomaly_factor(self) -> xr.DataArray | None:
        """Observation-space anomaly design scaled to standard-normal coordinates.

        The design stays lazy with the transport, so it can be combined with
        other designs before joint materialization.

        Returns:
            ``s M B`` in concentration units: on ``(observation,
            bc_contrast)`` for weighted anomalies and period contrasts, and
            ``s M`` on ``(observation, bc_region)`` for independent anomalies.
            Multiplying it by independent standard normals gives the anomaly
            concentration. ``None`` when :attr:`has_anomalies` is false.

        Raises:
            ValueError: If weighted-centering weights are still lazy.
        """
        if not self.has_anomalies:
            return None
        weights = None
        if self.choice.anomalies == "weighted":
            values = self.data.bc_centering_weights
            if is_dask_collection(values.data):
                raise ValueError("Materialize bc_centering_weights before forming a weighted anomaly design.")
            weights = (values / values.sum()).values
        basis = self._anomaly_basis(weights)
        design = self.data.H_bc_mass if basis is None else xr.dot(self.data.H_bc_mass, basis, dim="bc_region")
        return (design * self.choice.anomaly_scale).assign_attrs(units=self.data.bc_reference.attrs["units"])

    def add(self, *, output_dim: str = "nmeasure", prefix: str = "", owner: str | None = "bc") -> AffineConcentrationTerm:
        """Add the mean shift, anomalies and boundary concentration to the current model.

        Must be called inside an active PyMC model that has the coordinate
        registry, after ``data`` has been materialized jointly with the other
        model inputs. It registers the transport, reference, scale and any
        weights, basis or indicators as model data, plus the sampled
        variables and their deterministic responses.

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
            (``M 1``, or ``M`` summed over each period's states) followed,
            with anomalies, by ``M B`` (``M`` itself for independent
            anomalies) with their physical coefficients; and the sampled
            variables are the mean latent and anomaly coordinates.

        Raises:
            ValueError: If ``data`` is still lazy, materialized operands or
                weights are invalid, or a whole-window mean prior is not scalar.
            RuntimeError: If the model lacks the coordinate registry.
        """
        if any(is_dask_collection(value) for value in self.data.data_vars.values()):
            raise ValueError("Materialize boundary data jointly with model inputs before add().")
        choice = self.choice
        anomaly = self.data.H_bc_anomaly.transpose(output_dim, "bc_region")
        mass = self.data.H_bc_mass.transpose(output_dim, "bc_region")
        reference = self.data.bc_reference
        reference_shift = materialized_reference_shift(anomaly, mass, reference).rename(f"{prefix}bc_reference_shift")
        fixed = anomaly.sum("bc_region", skipna=False).assign_attrs(anomaly.attrs)
        weights = None
        if choice.anomalies == "weighted":
            weights = self.data.bc_centering_weights
            if not weights.size or not np.isfinite(weights.values).all() or not (weights.values > 0).all():
                raise ValueError("bc_centering_weights must be finite and strictly positive.")
            weights = weights / weights.max()
            weights = (weights / weights.sum()).assign_attrs(units="1")
        basis = self._anomaly_basis(None if weights is None else weights.values) if self.has_anomalies else None

        # Owner-qualified coordinates plus model data, sharing the observation axis.
        def register_boundary_data(array: xr.DataArray, name: str) -> TensorVariable:
            return add_model_data(component_coords(array, owner, observation_dim=output_dim), f"{prefix}{name}")

        a = register_boundary_data(anomaly, "H_bc_anomaly")
        m = register_boundary_data(mass, "H_bc_mass")
        register_boundary_data(reference, "bc_reference")
        register_boundary_data(xr.DataArray(choice.anomaly_scale), "bc_anomaly_scale")
        if weights is not None:
            register_boundary_data(weights, "bc_centering_weights")
        state_dim = str(component_coords(mass, owner, observation_dim=output_dim).dims[1])
        n_states, n_rows = mass.sizes["bc_region"], mass.sizes[output_dim]
        latents: tuple[TensorVariable, ...] = ()
        blocks: list[AffineBlock] = []

        # Mean shift, transported through M.
        mean_state, mean_signal = pt.zeros((n_states,)), pt.zeros((n_rows,))
        if choice.mean_prior is not None:
            mean = add_boundary_mean_shift(
                choice.mean_prior, mass, m, by_period=choice.mean_by_period,
                output_dim=output_dim, prefix=prefix, owner=owner,
            )
            mean_state, mean_signal = mean.state, mean.signal
            latents += (mean.latent,)
            blocks.append(mean.block)

        # Anomaly correction B z on the states, transported through M.
        anomaly_state, anomaly_signal = pt.zeros((n_states,)), pt.zeros((n_rows,))
        if self.has_anomalies:
            if basis is None:
                anomaly_state = pm.Normal(f"{prefix}bc_anomaly", mu=0.0, sigma=choice.anomaly_scale, dims=state_dim)
                coordinates, response = anomaly_state, mass
            else:
                # Registering the basis also registers its contrast coordinate.
                basis_data = register_boundary_data(basis, "bc_anomaly_basis")
                contrast_dim = str(component_coords(basis, owner).dims[1])
                coordinates = pm.Normal(
                    f"{prefix}bc_anomaly_latent", mu=0.0, sigma=choice.anomaly_scale, dims=contrast_dim,
                )
                anomaly_state = pm.Deterministic(f"{prefix}bc_anomaly", pt.dot(basis_data, coordinates), dims=state_dim)
                response = xr.dot(mass, basis, dim="bc_region").transpose(output_dim, "bc_contrast")
            anomaly_signal = pm.Deterministic(f"{prefix}mu_bc_anomaly", pt.dot(m, anomaly_state), dims=output_dim)
            latents += (coordinates,)
            blocks.append(AffineBlock(response.assign_attrs(units="1"), coordinates))

        # Complete boundary contribution A 1 + c M 1 + M d.
        pm.Deterministic(f"{prefix}bc_correction", mean_state + anomaly_state, dims=state_dim)
        absolute_reference = register_boundary_data(
            (fixed + reference_shift).assign_attrs(anomaly.attrs), "mu_bc_reference",
        )
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
        are available before :meth:`add` or sampling. Weighted anomalies with
        a positive scale and several states always declare anomaly outputs,
        so missing anomaly draws remain errors. Otherwise their zero
        anomalies are reconstructed from the saved scale and weights.

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
            "boundary_correction": "bc_correction",
            "boundary_reference_concentration": "mu_bc_reference",
        }
        units = {name: concentration for name in (
            "mu_bc", "H_bc_anomaly", "bc_reference", "bc_reference_shift", "boundary_reference_shift",
            "bc_anomaly_scale", "bc_correction", "mu_bc_reference",
        )}
        units["H_bc_mass"] = "1"

        mean = mean_shift_outputs(choice.mean_prior, by_period=choice.mean_by_period, concentration=concentration)
        roles.update(mean.roles)
        units.update(mean.units)

        if choice.anomalies == "weighted":
            roles.update(boundary_anomaly_scale="bc_anomaly_scale", boundary_centering_weights="bc_centering_weights")
            units["bc_centering_weights"] = "1"

        if self.has_anomalies:
            roles.update(boundary_correction_anomaly="bc_anomaly", boundary_anomaly_concentration="mu_bc_anomaly")
            units.update(bc_anomaly=concentration, mu_bc_anomaly=concentration)
            if choice.anomalies != "independent":
                units.update(bc_anomaly_latent=concentration, bc_anomaly_basis="1")

        metadata = {
            "boundary_formulation": "additive",
            "boundary_representation": "centered-v1",
            "boundary_anomalies": choice.anomalies,
            **mean.metadata,
            "bc_frequency": choice.frequency,
            "bc_anchor_time": None if choice.anchor_time is None else str(choice.anchor_time),
        }
        return BoundaryOutputs(roles, units, metadata).prefixed(prefix)


__all__ = ["AdditiveBoundary", "AdditiveChoice", "AnomalyStructure"]
