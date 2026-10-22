"""Concrete centered boundary arithmetic and absolute predictive replay."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.boundary_sensitivity import align_boundary_operands, validate_boundary_operands
from openghg_inversions.models.components import LinearComponentResult, add_state_vector
from openghg_inversions.models.coords import add_model_data, component_coords
from openghg_inversions.models.state_activity import prepare_linear_sensitivity, resolve_state_activity

OPERAND_NAMES = ("H_bc_anomaly", "H_bc_mass", "bc_reference")


def require_centered_precision() -> None:
    """Require explicitly selected double precision without changing global policy."""
    if pytensor.config.floatX != "float64":
        raise ValueError("Centered boundaries require explicitly configured PyTensor floatX=float64.")


def bind_boundary_operands(operands, observations, *, frequency=None, anchor_time=None):
    """Align one channel's raw spatial operands before any joint padding."""
    require_centered_precision()
    if not validate_boundary_operands(operands):
        raise ValueError("Expected centered boundary operands.")
    units = observations.attrs.get("units")
    if "observation_units" in observations.coords:
        unique_units = np.unique(observations.observation_units.data)
        units = str(unique_units[0]) if unique_units.size == 1 else None
    if units is not None and operands.bc_reference.attrs.get("units") != units:
        raise ValueError("Centered boundary reference units must match observation units.")
    native_dim = str(observations.dims[0])
    raw_dim = "time" if "time" in operands.H_bc_anomaly.dims else "nmeasure"
    labels = observations.rename({native_dim: raw_dim}) if native_dim != raw_dim else observations
    aligned = align_boundary_operands(
        operands,
        frequency=frequency,
        anchor_time=anchor_time,
        observation_labels=labels[raw_dim],
    )
    if raw_dim != native_dim:
        aligned = aligned.rename({raw_dim: native_dim})
    return aligned


def add_centered_boundary(operands, *, prior, activity=None, prefix="", output_dim="nmeasure", owner="bc"):
    """Keep the original BC prior/state and evaluate A b + c M (b - 1)."""
    anomaly = operands.H_bc_anomaly.transpose(output_dim, "bc_region")
    mass = operands.H_bc_mass.transpose(output_dim, "bc_region")
    reference = operands.bc_reference
    if not np.isfinite(reference.data):
        raise ValueError("Centered bc_reference must be finite.")
    # Activity belongs to the complete absolute design, including constant curtains.
    prepared = prepare_linear_sensitivity(anomaly + reference * mass, output_dim=output_dim)
    resolved = resolve_state_activity(prepared.removed, activity)
    vector = add_state_vector(resolved, prior_args=prior, var_name=f"{prefix}bc", coordinate_owner=owner)
    a = add_model_data(component_coords(anomaly, owner, observation_dim=output_dim), f"{prefix}H_bc_anomaly")
    m = add_model_data(component_coords(mass, owner, observation_dim=output_dim), f"{prefix}H_bc_mass")
    c = add_model_data(reference, f"{prefix}bc_reference")
    centered = pt.dot(a, vector.state) + c * pt.dot(m, vector.state - 1)
    shift = (reference * mass.sum("bc_region")).rename(f"{prefix}bc_reference_shift")
    shift_data = add_model_data(shift, shift.name)
    absolute = pm.Deterministic(f"{prefix}mu_bc", centered + shift_data, dims=output_dim)
    return LinearComponentResult(a, vector.latent, vector.state, absolute, resolved), centered, shift


def restore_absolute_predictives(trace: xr.DataTree) -> xr.DataTree:
    """Restore centered y once, using only retained trace constants and labels."""
    for name in ("observed_data", "prior_predictive", "posterior_predictive", "predictions"):
        if name not in trace.children or "y" not in trace[name]:
            continue
        constants = ("predictions_constant_data", "constant_data") if name == "predictions" else ("constant_data",)
        shift = next(
            (trace[group]["boundary_reference_shift"] for group in constants
             if group in trace.children and "boundary_reference_shift" in trace[group]),
            None,
        )
        if shift is None:
            continue
        trace.attrs["boundary_representation"] = "centered-v1"
        y = trace[name]["y"]
        if y.attrs.get("concentration_representation") == "absolute-v1":
            continue
        shift = shift.sel({dim: y[dim] for dim in shift.dims})
        y, aligned_shift = xr.align(y, shift, join="exact", copy=False)
        restored = y + aligned_shift
        restored.attrs = dict(y.attrs, concentration_representation="absolute-v1")
        trace[name]["y"] = restored.variable
    return trace
