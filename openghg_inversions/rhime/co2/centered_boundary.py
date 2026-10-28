"""Concrete centered boundary arithmetic and absolute predictive replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import pymc as pm
import pytensor
import xarray as xr
from numpy.typing import NDArray
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.boundary_sensitivity import _select_native_boundary, align_boundary_operands
from openghg_inversions.inversion_inputs import DatetimeLike
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.components import AffineTerm, LinearComponentResult, add_state_vector
from openghg_inversions.models.coords import add_model_data, component_coords
from openghg_inversions.models.state_activity import StateActivity, prepare_linear_sensitivity, resolve_state_activity

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


ArrayT = TypeVar("ArrayT", TensorVariable, NDArray[np.float64])


@dataclass(frozen=True, slots=True, eq=False)
class CenteredBoundaryMap(Generic[ArrayT]):
    """Absolute and centered concentrations of one physical boundary state.

    The absolute contribution is affine in the boundary scaling ``b``:
    ``concentration(b) = (A + c M) b``. The centered representation moves the
    deterministic reference ``q = c M 1`` from the mean onto the observations,
    so ``centered_concentration(b) = concentration(b) - q``. Both describe the
    same state and the same physical model; only the likelihood representation
    differs.

    Keep ``A``, ``M`` and ``c`` separate: subtracting ``q`` from a formed
    absolute contribution loses small anomalies against a large reference,
    such as O2 at about 209,300 ppm. The graph builder uses registered PyTensor
    operands; the cached likelihood projection uses the same equations on
    dense NumPy operands. Alignment, materialization, precision and state
    activity belong to the caller.

    Attributes:
        anomaly: Anomaly design ``A`` on observation by boundary-state axes.
        mass: Dimensionless mass design ``M`` on those same axes.
        reference: Scalar reference concentration ``c``.
        reference_shift: Observation-aligned concentration ``q = c M 1``.
            Mass rows need not sum to one.
    """

    anomaly: ArrayT
    mass: ArrayT
    reference: ArrayT | float
    reference_shift: ArrayT

    def centered_concentration(self, state: ArrayT) -> ArrayT:
        """Evaluate ``A b + c M (b - 1)``, the likelihood contribution against ``y - q``."""
        return self.anomaly @ state + self.reference * (self.mass @ (state - 1.0))

    def concentration(self, state: ArrayT) -> ArrayT:
        """Evaluate the absolute boundary concentration ``A b + c M b``."""
        return self.centered_concentration(state) + self.reference_shift


@dataclass(frozen=True)
class CenteredBoundaryResult:
    """Boundary graph with its state, affine term and retained operands.

    Attributes:
        component: Boundary state, activity, anomaly design tensor and absolute
            concentration output ``A b + c M b``.
        term: Absolute output, centered likelihood contribution
            ``A b + c M (b - 1)`` and observation shift ``q = c M 1``, which
            is also restored to predictive draws.
        operands: The aligned anomaly, mass and reference used to build the
            graph, borrowed for a cached likelihood's dense lowering.
    """

    component: LinearComponentResult
    term: AffineTerm
    operands: xr.Dataset


def add_centered_boundary(
    operands: xr.Dataset,
    *,
    prior: PriorArgs,
    activity: StateActivity | None = None,
    prefix: str = "",
    output_dim: str = "nmeasure",
    owner: str = "bc",
) -> CenteredBoundaryResult:
    """Construct absolute and centered concentrations from one boundary state.

    Call inside an active
    :func:`~openghg_inversions.models.coords.registered_model` context. Operands must be
    aligned, period-expanded values from :func:`bind_boundary_operands`, with
    ``H_bc_anomaly`` and ``H_bc_mass`` on ``(output_dim, bc_region)`` (either
    input order is accepted), and scalar ``bc_reference``. Anomaly and reference
    use the observations' concentration units; mass and the boundary scaling
    state are dimensionless. Registration owns coordinate compatibility with
    the existing observation axis and the namespaced boundary-state axis.

    This is an eager PyMC construction boundary: materialize related borrowed
    arrays together before calling. No input arrays are modified. Double
    precision must already be selected. Activity is resolved from the absolute
    design ``A + c M``, preserving constant-curtain states and the supplied prior.

    Args:
        operands: Already aligned and expanded anomaly, mass and reference.
        prior: Prior arguments for the dimensionless boundary scaling state.
        activity: Optional fixed/active boundary-state policy.
        prefix: Prefix for registered graph variable names.
        output_dim: Registered observation dimension.
        owner: Namespace owning the boundary-state coordinates.

    Returns:
        The boundary component, its affine term and the borrowed operands. The
        component's ``output`` is ``A b + c M b``; its ``data`` is the anomaly
        design, not the complete absolute design. Compose the term with
        :func:`~openghg_inversions.models.components.likelihood_representation`;
        add the shift back to observed and predictive ``y`` for absolute output.

    Raises:
        ValueError: If the scalar reference is nonfinite, activity cannot be
            resolved, or registered coordinates conflict with the active model.
    """
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
    shift = (reference * mass.sum("bc_region")).rename(f"{prefix}bc_reference_shift")
    shift_data = add_model_data(shift, shift.name)
    boundary_map = CenteredBoundaryMap(a, m, c, shift_data)
    centered = boundary_map.centered_concentration(vector.state)
    absolute = pm.Deterministic(f"{prefix}mu_bc", boundary_map.concentration(vector.state), dims=output_dim)
    return CenteredBoundaryResult(
        LinearComponentResult(a, vector.latent, vector.state, absolute, resolved),
        AffineTerm(absolute, centered, shift),
        operands,
    )


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
