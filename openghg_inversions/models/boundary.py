"""Shared centered boundary arithmetic and absolute predictive replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.boundary_sensitivity import _select_native_boundary, align_boundary_operands, bind_boundary_sensitivity
from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.components import AffineBlock, AffineConcentrationTerm, ConcentrationTerm, add_state_vector
from openghg_inversions.models.coords import add_model_data, component_coords
from openghg_inversions.models.state_activity import StateActivity, prepare_linear_sensitivity, resolve_state_activity
from openghg_inversions.models.additive_boundary import AdditiveBoundary, add_additive_boundary, prepare_additive_boundary
from openghg.util import cf_ureg

OPERAND_NAMES = ("H_bc_anomaly", "H_bc_mass", "bc_reference")


def require_centered_precision() -> None:
    """Require explicitly selected double precision without changing global policy."""
    if pytensor.config.floatX != "float64":
        raise ValueError("Centered boundaries require explicitly configured PyTensor floatX=float64.")


def bind_boundary_operands(
    operands: xr.Dataset,
    observations: xr.DataArray,
    *,
    frequency: str | None = None,
    anchor_time: DatetimeLike | None = None,
) -> xr.Dataset:
    """Align and expand raw boundary operands for one observation channel.

    Observation dimension labels own the row order; their auxiliary ``time``
    values own period membership when supplied, replacing auxiliary boundary
    timestamps. Indexed boundary times must match; this binding check may compute
    a lazy timestamp coordinate, while numerical operands remain borrowed. This calls
    :func:`~openghg_inversions.boundary_sensitivity.align_boundary_operands`
    before linked-channel padding, retaining its raw-operand validation and
    lazy payloads. Reference units must match observation ``units`` (or their
    single ``observation_units`` coordinate value) when supplied.

    Args:
        operands: Raw centered anomaly and mass on the spatial ``bc_region``
            and ``time`` or ``nmeasure`` axes, plus scalar reference.
        observations: One native channel with its observation axis first.
        frequency: Optional frequency expanding spatial states by period.
        anchor_time: Optional origin for that period alignment.

    Returns:
        Aligned operands on the native observation axis and the complete
        ``bc_region`` state axis, including any requested period expansion.
        These are ready for joint padding or eager PyMC materialization.

    Raises:
        ValueError: If double precision is not configured, raw operand or
            period validation fails, or reference and observation units differ.
    """
    require_centered_precision()
    native_dim = str(observations.dims[0])
    raw_dim = "time" if "H_bc_anomaly" in operands and "time" in operands.H_bc_anomaly.dims else "nmeasure"
    aligned = align_boundary_operands(
        _select_native_boundary(operands, observations, raw_dim),
        frequency=frequency,
        anchor_time=anchor_time,
    )
    units = observations.attrs.get("units")
    if "observation_units" in observations.coords:
        unique_units = np.unique(observations.observation_units.data)
        units = str(unique_units[0]) if unique_units.size == 1 else None
    if units is not None and aligned.bc_reference.attrs.get("units") != units:
        raise ValueError("Centered boundary reference units must match observation units.")
    if raw_dim != native_dim:
        # Native time has already determined periods; the restored dimension
        # index owns those same labels, so its auxiliary copy is redundant.
        if native_dim == "time" and "time" in aligned.coords:
            aligned = aligned.drop_vars("time")
        aligned = aligned.rename({raw_dim: native_dim})
    return aligned


def centered_boundary_expression(anomaly, mass, reference, state):
    """Evaluate ``A b + c M (b - 1)`` without subtracting a large reference.

    Args:
        anomaly: Concentration anomaly design A, on observations by states.
        mass: Dimensionless transport M on the same axes.
        reference: Scalar concentration reference c.
        state: Dimensionless physical boundary scaling vector b.

    Returns:
        Centered concentration, using NumPy or PyTensor operands supplied by
        the caller. Alignment and materialization belong to the caller.
    """
    return anomaly @ state + reference * (mass @ (state - 1.0))


@dataclass(frozen=True)
class MultiplicativeBoundary:
    """Borrowed labelled design and resolved dimensionless prior/activity policy.

    Structural activity is inspected at the eager graph boundary. ``centered``
    selects native anomaly/mass operands; otherwise ``data`` contains ``H_bc``.
    """

    data: xr.Dataset
    prior: PriorArgs
    activity: StateActivity | None
    centered: bool


BoundBoundary = MultiplicativeBoundary | AdditiveBoundary


def boundary_payload(bound: BoundBoundary) -> dict[str, Any]:
    """Extract borrowed numerical inputs for joint model materialization.

    The prior mapping includes its scalar/symbolic configuration unchanged.
    Every array-valued prior or activity input travels beside ``data`` so shared
    Dask graphs can be materialized once by the calling recipe. No copying,
    densification or computation occurs here.
    """
    if isinstance(bound, AdditiveBoundary):
        return {"data": bound.data, "prior": bound.mean_prior}
    payload: dict[str, Any] = {"data": bound.data, "prior": bound.prior}
    if bound.activity is not None:
        payload.update(activity_active=bound.activity.active, activity_fixed_value=bound.activity.fixed_value)
    return payload


def replace_boundary_payload(bound: BoundBoundary, payload: Mapping[str, Any]) -> BoundBoundary:
    """Restore jointly materialized inputs, retaining resolved scientific choices."""
    if isinstance(bound, AdditiveBoundary):
        return replace(bound, data=payload["data"], mean_prior=payload["prior"])
    activity = bound.activity
    if activity is not None:
        activity = replace(activity, active=payload["activity_active"], fixed_value=payload["activity_fixed_value"])
    return replace(bound, data=payload["data"], prior=payload["prior"], activity=activity)


def boundary_output_roles(bound: BoundBoundary, *, prefix: str = "") -> dict[str, str]:
    """Declare finite boundary scientific roles from resolved inputs, not a graph.

    Prefixes apply to both role and variable names. A positive anomaly scale
    with multiple states always declares anomaly outputs; missing active draws
    therefore remain errors. Zero scale or one state declares a mean-only
    component whose zero anomalies can be reconstructed from its saved facts.
    """
    roles = {"boundary_concentration": "mu_bc"}
    if isinstance(bound, AdditiveBoundary) or bound.centered:
        roles.update(boundary_anomaly="H_bc_anomaly", boundary_mass="H_bc_mass", boundary_reference="bc_reference")
    else:
        roles["boundary_sensitivity"] = "hbc"
    if isinstance(bound, AdditiveBoundary):
        roles.update(
            boundary_mean_shift="bc_mean_shift", boundary_anomaly_scale="bc_anomaly_scale",
            boundary_centering_weights="bc_centering_weights", boundary_correction="bc_correction",
            boundary_reference_concentration="mu_bc_reference",
            boundary_mean_shift_concentration="mu_bc_mean_shift",
        )
        if bound.anomaly_scale > 0 and bound.data.sizes["bc_region"] > 1:
            roles.update(boundary_correction_anomaly="bc_anomaly", boundary_anomaly_concentration="mu_bc_anomaly")
    else:
        roles["boundary_scale"] = "bc"
    return {prefix + role: prefix + name for role, name in roles.items()}


def boundary_output_units(bound: BoundBoundary, *, prefix: str = "") -> dict[str, str]:
    """Declare physical units for boundary graph quantities and saved constants.

    The result maps concrete variable names to units for output annotation.
    Names describe the finite component outputs, independently of which draw
    groups contain them. Coefficient expressions and reparameterized sampled
    variables have distinct physical meanings.
    """
    centered = isinstance(bound, AdditiveBoundary) or bound.centered
    concentration = bound.data.bc_reference.attrs["units"] if centered else bound.data.H_bc.attrs.get("units")
    units = {"mu_bc": concentration} if concentration is not None else {}
    if centered:
        units.update({name: concentration for name in (
            "H_bc_anomaly", "bc_reference", "bc_reference_shift", "boundary_reference_shift",
        )})
        units["H_bc_mass"] = "1"
    elif concentration is not None:
        units["hbc"] = concentration
    if isinstance(bound, AdditiveBoundary):
        units.update({name: concentration for name in (
            "bc_mean_shift", "bc_anomaly_scale", "bc_correction", "mu_bc_reference", "mu_bc_mean_shift",
        )})
        if bound.mean_prior.get("reparameterise") and str(bound.mean_prior.get("pdf", "")).lower() == "lognormal":
            units["bc_mean_shift_latent"] = "1"
        units["bc_centering_weights"] = "1"
        if bound.anomaly_scale > 0 and bound.data.sizes["bc_region"] > 1:
            units.update({name: concentration for name in ("bc_anomaly", "bc_anomaly_latent", "mu_bc_anomaly")})
            units["bc_anomaly_basis"] = "1"
    else:
        units.update({name: "1" for name in ("bc", "bc_active", "bc_fixed_value", "bc_latent", "bc_active_latent")})
    return {prefix + name: unit for name, unit in units.items()}


def _convert_units(array: xr.DataArray, units: str) -> xr.DataArray:
    """Normalize scientific units without evaluating the borrowed payload."""
    source = array.attrs.get("units")
    if source is None:
        raise ValueError(f"Boundary {array.name} requires units compatible with {units!r}.")
    try:
        factor = float(cf_ureg.Quantity(1, str(source)).to(units).magnitude)
    except Exception as exc:
        raise ValueError(f"Boundary {array.name} requires units compatible with {units!r}.") from exc
    converted = array if factor == 1.0 else array * factor
    return converted.assign_attrs(dict(array.attrs, units=units))


def prepare_boundary(
    observations: xr.DataArray,
    *,
    boundary_sensitivity: xr.DataArray | None = None,
    boundary_operands: xr.Dataset | None = None,
    bc_prior: PriorArgs | None = None,
    bc_state_activity: StateActivity | None = None,
    bc_frequency: str | None = None,
    bc_anchor_time: DatetimeLike | None = None,
    bc_mean_shift_prior: PriorArgs | None = None,
    bc_centering_weights: xr.DataArray | None = None,
    bc_anomaly_scale: float | None = None,
) -> BoundBoundary | None:
    """Resolve one boundary's scientific options and bind its labels once.

    Pass raw spatial sensitivities or native centered operands. Complete
    additive choices replace multiplicative prior/activity; otherwise a prior
    selects multiplicative boundaries. Passive inputs are not evaluated when
    no boundary is selected. Numerical arrays remain borrowed and lazy, ready
    for the recipe's joint materialization and ``dataclasses.replace``.
    """
    additive = bc_mean_shift_prior is not None or bc_anomaly_scale is not None
    if additive:
        if bc_prior is not None or bc_state_activity is not None:
            raise ValueError("Centred additive boundaries replace bc_prior and bc_state_activity; omit both.")
        if any(value is None for value in (
            bc_mean_shift_prior, bc_anomaly_scale, bc_centering_weights, boundary_operands,
        )):
            raise ValueError("Additive boundaries require mean prior, anomaly scale, weights and centered operands together.")
    elif bc_centering_weights is not None:
        raise ValueError("Boundary centering weights require an additive mean prior and anomaly scale.")
    elif bc_prior is None:
        if bc_state_activity is not None or bc_frequency is not None or bc_anchor_time is not None:
            raise ValueError("Boundary activity and periods require a boundary prior.")
        return None
    if boundary_operands is not None and boundary_sensitivity is not None:
        raise ValueError("Select boundary_sensitivity or boundary_operands, not both.")
    if boundary_operands is not None:
        units = observations.attrs.get("units")
        if "observation_units" in observations.coords:
            distinct = np.unique(observations.observation_units.data)
            units = str(distinct[0]) if distinct.size == 1 else None
        units = units or boundary_operands.bc_reference.attrs.get("units")
        if units is None:
            raise ValueError("Centered boundary concentrations require units.")
        boundary_operands = boundary_operands.assign(
            H_bc_anomaly=_convert_units(boundary_operands.H_bc_anomaly, units),
            H_bc_mass=_convert_units(boundary_operands.H_bc_mass, "1"),
            bc_reference=_convert_units(boundary_operands.bc_reference, units),
        )
        data = bind_boundary_operands(
            boundary_operands, observations, frequency=bc_frequency, anchor_time=bc_anchor_time,
        )
        if additive:
            return prepare_additive_boundary(
                data, bc_centering_weights, observations, bc_mean_shift_prior, bc_anomaly_scale,
                output_dim=str(observations.dims[0]),
            )
        assert bc_prior is not None
        return MultiplicativeBoundary(data, dict(bc_prior), bc_state_activity, True)
    if boundary_sensitivity is None:
        raise ValueError("A multiplicative boundary prior requires boundary_sensitivity or centered operands.")
    data = bind_boundary_sensitivity(
        boundary_sensitivity, observations, frequency=bc_frequency, anchor_time=bc_anchor_time,
    )
    units = observations.attrs.get("units")
    if units is not None and data.attrs.get("units") is not None:
        data = _convert_units(data, units)
    assert bc_prior is not None
    return MultiplicativeBoundary(data.rename("H_bc").to_dataset(), dict(bc_prior), bc_state_activity, False)


def add_boundary(
    bound: BoundBoundary,
    *,
    output_dim: str = "nmeasure",
    prefix: str = "",
    owner: str | None = None,
) -> AffineConcentrationTerm:
    """Construct a complete boundary graph and its labelled affine equation.

    Call after joint materialization of ``bound.data`` inside a registered model.
    The centered likelihood is evaluated directly as ``A b + c M (b-1)``.
    Its affine representation is ``A1 + D_inactive (b_fixed-1)`` plus
    ``D_active (b_active-1)``, with ``D=A+cM``. Absolute multiplicative
    boundaries analogously retain inactive-state effects. Sampled variables
    remain separate from these physical coefficient expressions. Multiplicative
    boundaries default to coordinate owner ``bc``; additive boundaries retain
    local coordinates unless an explicit owner is supplied.
    """
    if isinstance(bound, AdditiveBoundary):
        return add_additive_boundary(bound, output_dim=output_dim, prefix=prefix, owner=owner)
    if owner is None:
        owner = "bc"
    if any(hasattr(value.data, "__dask_graph__") for value in bound.data.data_vars.values()):
        raise ValueError("Materialize boundary.data jointly with model inputs before add_boundary.")
    if bound.centered:
        require_centered_precision()
        anomaly = bound.data.H_bc_anomaly.transpose(output_dim, "bc_region")
        mass = bound.data.H_bc_mass.transpose(output_dim, "bc_region")
        reference = bound.data.bc_reference
        if not np.isfinite(reference.values).all():
            raise ValueError("Centered bc_reference must be finite.")
        if not np.isfinite(mass.values).all() or (mass.values < 0).any() or not (mass.values > 0).any():
            raise ValueError("H_bc_mass must be finite, non-negative and not all zero.")
        design = (anomaly + reference * mass).assign_attrs(anomaly.attrs)
    else:
        design = bound.data.H_bc.transpose(output_dim, ...)
    prepared = prepare_linear_sensitivity(design, output_dim=output_dim)
    activity = resolve_state_activity(prepared.removed, bound.activity)
    vector = add_state_vector(activity, prior_args=bound.prior, var_name=f"{prefix}bc", coordinate_owner=owner)
    state_dim = activity.state_dim
    active, inactive = activity.active_indices, activity.fixed_indices
    if bound.centered:
        a = add_model_data(component_coords(anomaly, owner, observation_dim=output_dim), f"{prefix}H_bc_anomaly")
        m = add_model_data(component_coords(mass, owner, observation_dim=output_dim), f"{prefix}H_bc_mass")
        c = add_model_data(reference, f"{prefix}bc_reference")
        shift = (reference * mass.sum(state_dim, skipna=False)).rename(
            f"{prefix}bc_reference_shift"
        ).assign_attrs(reference.attrs)
        if not np.isfinite(shift.values).all():
            raise ValueError("Boundary reference shift must be finite.")
        likelihood = centered_boundary_expression(a, m, c, vector.state)
        absolute = pm.Deterministic(f"{prefix}mu_bc", likelihood + pt.as_tensor_variable(shift.values), dims=output_dim)
        fixed = anomaly.sum(state_dim, skipna=False) + xr.dot(
            design.isel({state_dim: inactive}), activity.fixed_value.isel({state_dim: inactive}) - 1,
            dim=state_dim,
        )
        coefficient = vector.state[active] - 1
        term = ConcentrationTerm(absolute, likelihood, shift)
    else:
        h = add_model_data(component_coords(design, owner, observation_dim=output_dim), f"{prefix}hbc")
        absolute = pm.Deterministic(f"{prefix}mu_bc", pt.dot(h, vector.state), dims=output_dim)
        fixed = xr.dot(design.isel({state_dim: inactive}), activity.fixed_value.isel({state_dim: inactive}), dim=state_dim)
        coefficient = vector.state[active]
        term = ConcentrationTerm.absolute(absolute)
    fixed = fixed.assign_attrs(design.attrs)
    blocks = (AffineBlock(design.isel({state_dim: active}), coefficient),) if active.size else ()
    sampled = (vector.latent,) if vector.latent is not None else ()
    return AffineConcentrationTerm(term, fixed, blocks, sampled)


def restore_absolute_predictives(trace: xr.DataTree) -> xr.DataTree:
    """Restore centered y once, using only retained trace constants and labels."""
    for name in ("observed_data", "prior_predictive", "posterior_predictive", "predictions"):
        if name not in trace.children or "y" not in trace[name]:
            continue
        constants = ("predictions_constant_data", "constant_data") if name == "predictions" else ("constant_data",)
        shift = next(
            (trace[group].dataset["boundary_reference_shift"] for group in constants
             if group in trace.children and "boundary_reference_shift" in trace[group]),
            None,
        )
        if shift is None:
            continue
        trace.attrs["boundary_representation"] = "centered-v1"
        y = trace[name].dataset["y"]
        if y.attrs.get("concentration_representation") == "absolute-v1":
            continue
        shift = shift.sel({dim: y[dim] for dim in shift.dims})
        y, aligned_shift = xr.align(y, shift, join="exact", copy=False)
        restored = y + aligned_shift
        restored.attrs = dict(y.attrs, concentration_representation="absolute-v1")
        trace[name]["y"] = restored.variable
    return trace
