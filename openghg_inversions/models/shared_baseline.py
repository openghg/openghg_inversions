"""Frozen, labelled operators for one shared atmospheric baseline.

Preparation borrows lazy transport arrays. Freezing equal-site weights reads
site labels eagerly; retained covariance checks use the existing small eager
prior. Model construction is the numerical transport execution boundary.
"""

from __future__ import annotations

import json

import numpy as np
import xarray as xr

from openghg_inversions.biosphere_prior import BiospherePrior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior

SCHEMA = "shared-atmospheric-baseline-v1"


def prepare_shared_baseline(
    flux_sensitivity: xr.DataArray,
    retained_prior: CorrelatedLognormalPrior,
    fixed_prior_contribution: xr.DataArray,
    boundary_correction: xr.Dataset,
    *,
    outer: xr.DataArray,
    sampled_outer: xr.DataArray,
    mode: str,
    delta_sd: float,
    additive_design: xr.DataArray | None = None,
    calibrated_prior: BiospherePrior | None = None,
    additive_density_sd: xr.DataArray | None = None,
) -> xr.Dataset:
    """Freeze equal-site centring and a complete baseline for four outer choices.

    ``boundary_correction`` is already period-gated by
    ``prepare_boundary_correction``. Its level is replaced by one atmospheric
    delta with response exactly one, and its patterns are centred afterwards.
    ``outer`` selects all outer states; ``sampled_outer`` excludes fixed ocean.
    Fixed/additive modes remove all outer states and require zero cross
    covariance with inner states. Multiplicative modes retain their full joint
    marginal covariance. No coherent operator or aggregation error is refitted.

    Additive design has axes ``(nmeasure, outer_pattern)`` and exactly twelve
    unique labelled columns: six net-biosphere followed by six fossil-fuel
    unit-density responses multiplied by explicit density SDs. It replaces,
    rather than supplements, the original outer states. Ocean stays fixed.
    Shrinkage requires an already resolved positive #996 calibration of this
    exact baseline. Widths and calibration are never chosen here.

    Save the returned Dataset with the standard MultiIndex codec. Subset its
    observation rows directly; stored column means and scalar reference are
    frozen and are never recomputed for a subset.
    """
    if mode not in {"fixed", "additive", "multiplicative", "multiplicative_shrunk"}:
        raise ValueError("Unknown shared-baseline outer mode.")
    if not np.isfinite(delta_sd) or delta_sd <= 0:
        raise ValueError("delta_sd must be explicitly finite and positive.")
    if (mode == "additive") != (additive_density_sd is not None):
        raise ValueError("Only additive mode requires physical additive_density_sd.")
    if (mode == "additive") != (additive_design is not None):
        raise ValueError("Only additive mode requires additive_design.")
    if (mode == "multiplicative_shrunk") != (calibrated_prior is not None):
        raise ValueError("Only multiplicative_shrunk requires calibrated_prior.")
    dim, cov_dim = retained_prior.state_dim, retained_prior.covariance_dim
    mean = retained_prior.mean
    flux_sensitivity, mean, outer, sampled_outer = xr.align(
        flux_sensitivity, mean, outer, sampled_outer, join="exact", copy=False
    )
    if outer.dims != (dim,) or sampled_outer.dims != (dim,):
        raise ValueError("Outer selectors must use the retained-state axis.")
    if outer.dtype != bool or sampled_outer.dtype != bool or bool((sampled_outer & ~outer).any()):
        raise ValueError("sampled_outer must be a boolean subset of outer.")
    row = "nmeasure"
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

    fixed_prior_contribution = without_observation_auxiliary(fixed_prior_contribution)
    boundary_correction = without_observation_auxiliary(boundary_correction)
    h, fixed_prior_contribution, boundary_correction = xr.align(
        h, fixed_prior_contribution, boundary_correction, join="exact", copy=False
    )
    if boundary_correction.attrs.get("boundary_correction_schema") != "independent-boundary-correction-v1":
        raise ValueError("Expected a prepared independent boundary correction.")
    units = boundary_correction.bc_reference.attrs["units"]
    for name, array in (("flux_sensitivity", h), ("fixed_prior_contribution", fixed_prior_contribution)):
        if array.attrs.get("units", units) != units:
            raise ValueError(f"{name} must use the boundary correction concentration units.")
    if boundary_correction.bc_reference.ndim != 0:
        raise ValueError("Shared baseline requires one scalar reference.")
    if "site" not in h.coords or h.site.dims != (row,):
        raise ValueError("Equal-site centring requires observation-aligned site labels.")
    sites, inverse, counts = np.unique(h.site.values, return_inverse=True, return_counts=True)
    weights = xr.DataArray(1 / (len(sites) * counts[inverse]), dims=row, coords={row: h[row]})
    covariance = retained_prior.arithmetic_covariance
    outer_positions = np.flatnonzero(outer.values)
    inner_positions = np.flatnonzero(~outer.values)
    if mode in {"fixed", "additive"} and np.any(
        covariance.values[np.ix_(inner_positions, outer_positions)] != 0
    ):
        raise ValueError("Fixed/additive outer removal requires zero inner/outer cross covariance.")
    selected_covariance = covariance
    if calibrated_prior is not None:
        xr.testing.assert_equal(calibrated_prior.baseline.mean, retained_prior.mean)
        xr.testing.assert_equal(calibrated_prior.baseline.arithmetic_covariance, covariance)
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
        raise ValueError("Removing fixed outer states requires zero covariance with retained sampled states.")
    outer_mean = xr.dot(weights, h, dim=row)
    centered_h = h - outer_mean
    transformed_h = xr.where(sampled_outer, centered_h, h) if multi else h
    physical_outer = xr.dot(h.where(outer, 0), mean, dim=dim)
    affine = fixed_prior_contribution + physical_outer
    if multi:
        affine = affine - xr.dot(centered_h.where(sampled_outer, 0), mean, dim=dim)
    bc = boundary_correction.bc_pattern_operator * boundary_correction.bc_pattern_sd
    bc_mean = xr.dot(weights, bc, dim=row)

    def renamed_state(array, new_dim):
        state_axes = {dim, cov_dim} & set(array.dims)
        names = {
            name: new_dim if name == dim else f"{new_dim}_cov" if name == cov_dim else f"{new_dim}_{name}"
            for name, value in array.coords.items()
            if state_axes.intersection(value.dims)
        }
        names.update({axis: new_dim if axis == dim else f"{new_dim}_cov" for axis in state_axes})
        return array.rename(names)

    result = xr.Dataset(
        {
            "centering_weights": weights,
            "outer_column_mean": renamed_state(outer_mean, "original_state"),
            "original_prior_mean": renamed_state(mean, "original_state"),
            "original_prior_covariance": renamed_state(covariance, "original_state"),
            "physical_outer_prior": physical_outer,
            "physical_prior_baseline": fixed_prior_contribution
            + boundary_correction.bc_prescribed_centered
            + boundary_correction.bc_reference_shift
            + physical_outer,
            "complete_prior_concentration": fixed_prior_contribution
            + boundary_correction.bc_prescribed_centered
            + boundary_correction.bc_reference_shift
            + xr.dot(h, mean, dim=dim),
            "original_outer_operator": renamed_state(h.isel({dim: outer_positions}), "physical_outer_state"),
            "original_outer_mean": renamed_state(mean.isel({dim: outer_positions}), "physical_outer_state"),
            "flux_sensitivity": transformed_h.isel({dim: positions}),
            "retained_mean": mean.isel({dim: positions}),
            "retained_covariance": selected_covariance.isel({dim: positions, cov_dim: positions}),
            "fixed_prior_contribution": affine,
            "bc_pattern_raw": bc,
            "original_bc_pattern_sd": boundary_correction.bc_pattern_sd,
            "bc_period_index": boundary_correction.bc_period_index,
            "bc_pattern_mean": bc_mean,
            "bc_pattern_centered": bc - bc_mean,
            "physical_bc_prior": boundary_correction.bc_prescribed_centered
            + boundary_correction.bc_reference_shift,
            "reference": boundary_correction.bc_reference,
            "delta_sd": xr.DataArray(delta_sd),
            "sampled_outer": sampled_outer.isel({dim: positions}),
            "selected_outer_raw": h.isel({dim: positions}).where(sampled_outer.isel({dim: positions}), 0),
            "selected_outer_centered": centered_h.isel({dim: positions}).where(
                sampled_outer.isel({dim: positions}), 0
            ),
        },
        attrs={
            "shared_baseline_schema": SCHEMA,
            "mode": mode,
            "state_dim": dim,
            "covariance_dim": cov_dim,
            "units": boundary_correction.bc_reference.attrs["units"],
            "boundary_correction_attrs": json.dumps(boundary_correction.attrs),
            "centering_policy": "equal-site totals; uniform rows within site; frozen before subsets",
        },
    )
    if additive_design is not None:
        if additive_design.attrs.get("units") != units:
            raise ValueError("Additive design must declare the boundary correction concentration units.")
        additive_design, _ = xr.align(additive_design, h, join="exact", copy=False)
        if additive_design.dims != (row, "outer_pattern") or additive_design.sizes["outer_pattern"] != 12:
            raise ValueError("Additive design requires six net-bio and six FF columns on outer_pattern.")
        if not additive_design.get_index("outer_pattern").is_unique:
            raise ValueError("Additive outer pattern labels must be unique.")
        if additive_density_sd.dims != ("outer_pattern",) or not additive_density_sd.attrs.get("units"):
            raise ValueError("Physical additive density SDs require outer_pattern labels and declared units.")
        additive_density_sd, _ = xr.align(additive_density_sd, additive_design, join="exact", copy=False)
        if not np.isfinite(additive_density_sd.values).all() or bool((additive_density_sd <= 0).any()):
            raise ValueError("Physical additive density SDs must be finite and positive.")
        source = additive_design.coords.get("outer_source")
        if (
            source is None
            or source.dims != ("outer_pattern",)
            or sorted(source.values.tolist()) != ["FF"] * 6 + ["net_bio"] * 6
        ):
            raise ValueError(
                "Additive patterns require outer_source labels for six net_bio and six FF corrections."
            )
        result["additive_density_sd"] = additive_density_sd
        result["additive_unit_operator"] = additive_design / additive_density_sd
        result["additive_raw"] = additive_design
        result["additive_mean"] = xr.dot(weights, additive_design, dim=row)
        result["additive_centered"] = additive_design - result.additive_mean
    if calibrated_prior is not None:
        calibration = calibrated_prior.to_dataset()
        rename = {name: f"calibration_{name}" for name in set(calibration.variables) | set(calibration.dims)}
        result = xr.merge([result, calibration.rename(rename)], join="exact", compat="no_conflicts")
        result.attrs["calibration_attrs"] = json.dumps(calibration.attrs)
    return result
