"""One shared atmospheric baseline level with centred boundary and outer corrections.

A :class:`SharedBaselineChoice` is data-free: the boundary anomaly structure,
the atmospheric level width and the outer-flux mode. Binding it to one fit's
prepared transport, retained prior and observation rows returns a
:class:`SharedBaseline` whose ``data`` holds the transformed, still lazy
operators the recipe consumes. Nothing is prepared or saved separately: the
recipe binds at construction, and the trace records what the model used.

Equal-site centring is computed from the bound rows. To fit a subset of rows
with the centring of a larger support, bind on the larger support and subset
the bound record with ``dataclasses.replace(bound, data=bound.data.isel(...))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import xarray as xr

from openghg_inversions.biosphere_prior import BiospherePrior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.boundary import AdditiveChoice, AnomalyScalingChoice, BoundaryOutputs, CenteredOperands

OuterMode = Literal["fixed", "additive", "multiplicative", "multiplicative_shrunk"]


@dataclass(frozen=True)
class SharedBaselineChoice:
    """Data-free shared atmospheric baseline options.

    Attributes:
        boundary: Boundary anomaly structure, or ``None`` for the prescribed
            boundary only. Its mean shift must be omitted: the atmospheric
            level replaces it. Weighted additive anomalies are unsupported.
        delta_sd: Positive scalar atmospheric level SD in observation
            concentration units.
        mode: Fixed, additive, multiplicative, or calibrated multiplicative
            outer-flux prior.
    """

    boundary: AdditiveChoice | AnomalyScalingChoice | None
    delta_sd: float
    mode: OuterMode

    def __post_init__(self) -> None:
        if self.mode not in get_args(OuterMode):
            raise ValueError("Unknown shared-baseline outer mode.")
        if not np.isfinite(self.delta_sd) or self.delta_sd <= 0:
            raise ValueError("delta_sd must be explicitly finite and positive.")
        if self.boundary is not None and self.boundary.mean_prior is not None:
            raise ValueError("The shared atmospheric level replaces the boundary mean shift; omit its mean prior.")
        if isinstance(self.boundary, AdditiveChoice) and self.boundary.anomalies == "weighted":
            raise ValueError("Shared baselines support period-contrast or independent boundary anomalies.")

    def bind(
        self,
        observations: xr.DataArray,
        *,
        flux_sensitivity: xr.DataArray,
        retained_prior: CorrelatedLognormalPrior,
        fixed_prior_contribution: xr.DataArray,
        boundary_transport: CenteredOperands,
        outer: xr.DataArray,
        sampled_outer: xr.DataArray,
        additive_design: xr.DataArray | None = None,
        additive_density_sd: xr.DataArray | None = None,
        calibrated_prior: BiospherePrior | None = None,
    ) -> SharedBaseline:
        """Centre corrections on one fit's rows and assemble its baseline.

        Each retained site gets equal total weight, spread uniformly over that
        site's rows; transported correction operators lose their weighted
        column means. Fixed/additive modes remove all outer states and require
        zero cross covariance with inner states. Multiplicative modes retain
        their full joint marginal covariance. No coherent operator or
        aggregation error is refitted.

        Args:
            observations: Native ``nmeasure`` rows with ``site`` and ``time``.
            flux_sensitivity: Borrowed observation-by-retained-state operator.
            retained_prior: Complete labelled joint physical arithmetic moments.
            fixed_prior_contribution: Original observation-aligned affine term.
            boundary_transport: Prepared centered boundary operands.
            outer: Boolean outer-state selector on the complete retained axis.
            sampled_outer: Boolean adjustable-outer subset, excluding fixed ocean.
            additive_design: Additive mode only: width-scaled
                ``(nmeasure, outer_pattern)`` net-biosphere/fossil responses with
                an ``outer_source`` label per column.
            additive_density_sd: Additive mode only: physical density SDs on
                ``outer_pattern``, convertible to ``mol m-2 s-1``.
            calibrated_prior: ``multiplicative_shrunk`` only: a resolved positive
                calibration of this retained prior.

        Returns:
            Bound baseline whose ``data`` stays lazy; only site labels and the
            small retained prior are read here.
        """
        mode = self.mode
        if (mode == "additive") != (additive_density_sd is not None):
            raise ValueError("Only additive mode requires physical additive_density_sd.")
        if (mode == "additive") != (additive_design is not None):
            raise ValueError("Only additive mode requires additive_design.")
        if (mode == "multiplicative_shrunk") != (calibrated_prior is not None):
            raise ValueError("Only multiplicative_shrunk requires calibrated_prior.")
        row = "nmeasure"
        if observations.dims != (row,):
            raise ValueError("Shared baselines bind native nmeasure observation rows.")
        dim, cov_dim = retained_prior.state_dim, retained_prior.covariance_dim
        mean = retained_prior.mean
        flux_sensitivity, mean, outer, sampled_outer = xr.align(
            flux_sensitivity, mean, outer, sampled_outer, join="exact", copy=False
        )
        if outer.dims != (dim,) or sampled_outer.dims != (dim,):
            raise ValueError("Outer selectors must use the retained-state axis.")
        if outer.dtype != bool or sampled_outer.dtype != bool or bool((sampled_outer & ~outer).any()):
            raise ValueError("sampled_outer must be a boolean subset of outer.")
        if self.boundary is None:
            bc_data, bc_factor = boundary_transport.bind(observations), None
        else:
            bound = self.boundary.bind(observations, boundary_transport)
            bc_data, bc_factor = bound.data, bound.anomaly_factor()
        units = bc_data.bc_reference.attrs["units"]
        anomaly = bc_data.H_bc_anomaly.transpose(row, "bc_region")
        mass = bc_data.H_bc_mass.transpose(row, "bc_region")
        prescribed = anomaly.sum("bc_region", skipna=False).assign_attrs(units=units)
        shift = (bc_data.bc_reference * mass.sum("bc_region", skipna=False)).assign_attrs(units=units)
        if bc_factor is None:
            bc = xr.DataArray(np.empty((observations.size, 0)), dims=(row, "bc_pattern_term"),
                              coords={row: observations[row], "bc_pattern_term": np.arange(0)})
        else:
            bc = bc_factor.rename({bc_factor.dims[-1]: "bc_pattern_term"}).transpose(row, "bc_pattern_term")
        h = flux_sensitivity.transpose(row, dim)
        if not h.get_index(row).is_unique or not mean.get_index(dim).is_unique:
            raise ValueError("Observation and state labels must be unique.")

        # One observation-coordinate owner avoids eager comparisons of borrowed
        # auxiliary coordinates when the numerical operators remain Dask-backed.
        def without_observation_auxiliary(value):
            names = [
                name
                for name, coord in value.coords.items()
                if name != row and row in coord.dims and name not in value.indexes
            ]
            return value.drop_vars(names)

        h, fixed_prior_contribution, prescribed, shift, bc = xr.align(
            *(without_observation_auxiliary(value) for value in (h, fixed_prior_contribution, prescribed, shift, bc)),
            join="exact", copy=False,
        )
        for name, array in (("flux_sensitivity", h), ("fixed_prior_contribution", fixed_prior_contribution)):
            if not array.attrs.get("units"):
                raise ValueError(f"{name} requires declared concentration units.")
        h = h.pint.quantify().pint.to(units).pint.dequantify().assign_attrs(units=units)
        fixed_prior_contribution = fixed_prior_contribution.pint.quantify().pint.to(units).pint.dequantify().assign_attrs(units=units)
        if "site" not in observations.coords or observations.site.dims != (row,):
            raise ValueError("Equal-site centring requires observation-aligned site labels.")
        sites, inverse, counts = np.unique(observations.site.values, return_inverse=True, return_counts=True)
        weights = xr.DataArray(1 / (len(sites) * counts[inverse]), dims=row, coords={row: h[row]})
        covariance = retained_prior.arithmetic_covariance
        outer_positions = np.flatnonzero(outer.values)
        selected_covariance = covariance
        if calibrated_prior is not None:
            if not (calibrated_prior.baseline.mean.equals(retained_prior.mean)
                    and calibrated_prior.baseline.arithmetic_covariance.equals(covariance)):
                raise ValueError("Calibration baseline must match the retained physical prior.")
            if calibrated_prior.positive_prior is None:
                raise ValueError("Multiplicative shrinkage requires a positive lognormal calibration.")
            if bool((calibrated_prior.data.biosphere & ~sampled_outer).any()):
                raise ValueError("Shared outer calibration may only change sampled outer biosphere states.")
            selected_covariance = calibrated_prior.positive_prior.arithmetic_covariance
        multi = mode.startswith("multiplicative")
        keep = ~outer | sampled_outer if multi else ~outer
        positions = np.flatnonzero(keep.values)
        removed = np.flatnonzero(~keep.values)
        if np.any(selected_covariance.values[np.ix_(positions, removed)] != 0):
            raise ValueError("Removing fixed outer states requires zero cross covariance with retained sampled states.")
        column_mean = xr.dot(weights, h, dim=row)
        centered_h = h - column_mean
        transformed_h = (xr.where(sampled_outer, centered_h, h).transpose(row, dim) if multi else h).assign_attrs(units=units)
        physical_outer = xr.dot(h.where(outer, 0), mean, dim=dim)
        # Compensate at the prior mean: affine + H_new @ m_kept = fixed_old + H_old @ m_full.
        affine = fixed_prior_contribution + physical_outer
        if multi:
            affine = affine - xr.dot(centered_h.where(sampled_outer, 0), mean, dim=dim)
        bc = bc.assign_attrs(units=units)
        bc.bc_pattern_term.attrs = {"units": "1", "prior": "independent standard normal"}

        def renamed_state(array, new_dim):
            state_axes = {dim, cov_dim} & set(array.dims)
            names = {
                name: new_dim if name == dim else f"{new_dim}_cov" if name == cov_dim else f"{new_dim}_{name}"
                for name, value in array.coords.items()
                if state_axes.intersection(value.dims)
            }
            names.update({axis: new_dim if axis == dim else f"{new_dim}_cov" for axis in state_axes})
            return array.rename(names)

        physical_bc = (prescribed + shift).assign_attrs(units=units)
        result = xr.Dataset(
            {
                "centering_weights": weights,
                "outer_column_mean": renamed_state(column_mean, "original_state"),
                "original_prior_mean": renamed_state(mean, "original_state"),
                "original_prior_covariance": renamed_state(covariance, "original_state"),
                "physical_outer_prior": physical_outer,
                "physical_prior_baseline": fixed_prior_contribution + physical_bc + physical_outer,
                "complete_prior_concentration": fixed_prior_contribution + physical_bc + xr.dot(h, mean, dim=dim),
                "original_outer_operator": renamed_state(h.isel({dim: outer_positions}), "physical_outer_state"),
                "original_outer_mean": renamed_state(mean.isel({dim: outer_positions}), "physical_outer_state"),
                "flux_sensitivity": transformed_h.isel({dim: positions}),
                "retained_mean": mean.isel({dim: positions}),
                "retained_covariance": selected_covariance.isel({dim: positions, cov_dim: positions}),
                "fixed_prior_contribution": affine.assign_attrs(units=units),
                "bc_pattern_raw": bc,
                "bc_pattern_mean": xr.dot(weights, bc, dim=row),
                "bc_prescribed_centered": prescribed,
                "bc_reference_shift": shift,
                "physical_bc_prior": physical_bc,
                "reference": bc_data.bc_reference,
                "delta_sd": xr.DataArray(self.delta_sd, attrs={"units": units}),
                "sampled_outer": sampled_outer.isel({dim: positions}),
            },
            attrs={
                "mode": mode,
                "state_dim": dim,
                "covariance_dim": cov_dim,
                "units": units,
                "boundary_level_policy": "independent atmospheric delta replaces any boundary mean shift",
                "centering_policy": "equal-site totals; uniform rows within site over the bound rows",
            },
        )
        if additive_design is not None and additive_density_sd is not None:
            if not additive_design.attrs.get("units"):
                raise ValueError("Additive design requires declared concentration units.")
            additive_design = additive_design.pint.quantify().pint.to(units).pint.dequantify().assign_attrs(units=units)
            additive_design, _ = xr.align(additive_design, h, join="exact", copy=False)
            if additive_design.dims != (row, "outer_pattern") or not additive_design.sizes["outer_pattern"]:
                raise ValueError("Additive design requires nonempty net-bio/FF columns on outer_pattern.")
            if not additive_design.get_index("outer_pattern").is_unique:
                raise ValueError("Additive outer pattern labels must be unique.")
            if additive_density_sd.dims != ("outer_pattern",) or not additive_density_sd.attrs.get("units"):
                raise ValueError("Physical additive density SDs require outer_pattern labels and declared units.")
            additive_density_sd = additive_density_sd.pint.quantify().pint.to("mol m-2 s-1").pint.dequantify().assign_attrs(units="mol m-2 s-1")
            additive_density_sd, _ = xr.align(additive_density_sd, additive_design, join="exact", copy=False)
            if not np.isfinite(additive_density_sd.values).all() or bool((additive_density_sd <= 0).any()):
                raise ValueError("Physical additive density SDs must be finite and positive.")
            source = additive_design.coords.get("outer_source")
            if (
                source is None
                or source.dims != ("outer_pattern",)
                or not set(source.values.tolist()) <= {"FF", "net_bio"}
            ):
                raise ValueError(
                    "Additive patterns require outer_source labels for net_bio/FF corrections."
                )
            result["additive_density_sd"] = additive_density_sd
            result["additive_unit_operator"] = additive_design / additive_density_sd
            result["additive_raw"] = additive_design
            result["additive_mean"] = xr.dot(weights, additive_design, dim=row)
            result["additive_centered"] = (additive_design - result.additive_mean).assign_attrs(units=units)
        if calibrated_prior is not None:
            calibration = calibrated_prior.to_dataset()
            rename = {name: f"calibration_{name}" for name in set(calibration.variables) | set(calibration.dims)}
            result = xr.merge([result, calibration.rename(rename)], join="exact", compat="no_conflicts")
        return SharedBaseline(self, result)


@dataclass(frozen=True)
class SharedBaseline:
    """Shared baseline choice bound to one fit's borrowed, still lazy operators."""

    choice: SharedBaselineChoice
    data: xr.Dataset

    def outputs(self, prefix: str = "") -> BoundaryOutputs:
        """Declare the atmospheric level's roles and units and the bound choice's metadata."""
        boundary = self.choice.boundary
        units = self.data.attrs["units"]
        roles = {"boundary_mean_shift": "bc_mean_shift", "boundary_mean_shift_concentration": "mu_bc_mean_shift"}
        metadata = {
            "boundary_formulation": "shared_atmospheric",
            "boundary_representation": "centered-v1",
            "shared_baseline_mode": self.choice.mode,
            "boundary_anomalies": (
                None if boundary is None
                else "anomaly_scaling" if isinstance(boundary, AnomalyScalingChoice) else boundary.anomalies
            ),
            "bc_frequency": None if boundary is None else boundary.frequency,
            "bc_anchor_time": None if boundary is None or boundary.anchor_time is None else str(boundary.anchor_time),
        }
        return BoundaryOutputs(roles, {"bc_mean_shift": units, "mu_bc_mean_shift": units}, metadata).prefixed(prefix)


__all__ = ["OuterMode", "SharedBaseline", "SharedBaselineChoice"]
