"""Centered boundary transport binding and numerical helpers.

Centered preparation stores the boundary contribution as three operands:
the concentration anomaly ``A`` (``H_bc_anomaly``), the dimensionless transport
mass ``M`` (``H_bc_mass``) and a scalar concentration reference ``c``
(``bc_reference``). Absolute concentrations are ``A + c M``. Preparation and
artifact loading establish :class:`~.contracts.RawCenteredTransport`; this
module trusts that contract and binds the raw spatial operands to one fit's
observation rows and boundary periods, producing
:class:`~.contracts.BoundCenteredTransport` by construction.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytensor
import xarray as xr

from openghg_inversions.boundary_sensitivity import _expand_boundary_periods, _select_native_boundary
from openghg_inversions.inversion_inputs import DatetimeLike

from .contracts import STATE_LEVELS, BoundAdditiveTransport, BoundCenteredTransport, RawCenteredTransport


def require_centered_precision() -> None:
    """Require a process-wide PyTensor ``floatX`` of float64 for centered graphs.

    ``pm.Data`` casts registered arrays to the global ``floatX``. Centered
    operands rely on float64 to keep small anomalies after cancellation, so a
    float32 process would silently round them. This only reads the global
    configuration; set ``PYTENSOR_FLAGS=floatX=float64`` before starting Python.
    """
    if pytensor.config.floatX != "float64":
        raise ValueError("Centered boundaries require explicitly configured PyTensor floatX=float64.")


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


def check_bound_transport(data: xr.Dataset, names: tuple[str, ...], *, state_dim: str | None = None) -> str:
    """Check the layout of bound transport and return its observation dimension.

    Bound records store transport on ``(observation, state)``, the order that
    binding produces and row padding preserves, so ``add`` reads the arrays
    as stored. Only dimensions are inspected; payloads stay borrowed and lazy.

    Args:
        data: The bound record's data.
        names: Transport variables, which must share the same two dimensions.
        state_dim: Required state dimension, or ``None`` to accept the bound
            state labels.

    Returns:
        The shared observation dimension.

    Raises:
        ValueError: If a variable is missing or the variables do not share
            one ``(observation, state)`` layout ending in ``state_dim``, or if
            a centered ``bc_reference`` is present but not scalar.
    """
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"Bound boundary data lacks {', '.join(missing)}; bind the choice again.")
    dims = {data[name].dims for name in names}
    layout = next(iter(dims))
    if len(dims) != 1 or len(layout) != 2 or (state_dim is not None and layout[1] != state_dim):
        expected = f"(observation, {state_dim or 'state'})"
        raise ValueError(f"Bound boundary transport must share one {expected} layout; got {sorted(dims)}.")
    if "bc_reference" in data and data["bc_reference"].ndim:
        raise ValueError("Bound bc_reference must be scalar.")
    return str(layout[0])


def materialized_reference_shift(
    anomaly: xr.DataArray, mass: xr.DataArray, reference: xr.DataArray, *, state_dim: str = "bc_region",
) -> xr.DataArray:
    """Check materialized centered values and return the fixed shift ``c M 1``.

    Preparation validates layout, labels and units without computing borrowed
    payloads, and recentring at binding can derive new values, so finite
    values are first checkable here, after the joint materialization that
    must precede graph construction. Every centered family calls this once at
    the start of ``add``.

    Args:
        anomaly: Materialized ``A`` on observation-by-state axes.
        mass: Materialized dimensionless ``M`` on the same axes.
        reference: Materialized scalar ``c`` in concentration units.
        state_dim: Boundary-state dimension summed by ``M 1``.

    Returns:
        ``c M 1`` on the observation axis, with the reference's attributes.

    Raises:
        ValueError: If ``A`` or ``c`` is non-finite, ``M`` is non-finite,
            negative or all zero, or the shift is non-finite.
    """
    if not np.isfinite(reference.values).all() or not np.isfinite(anomaly.values).all():
        raise ValueError("Centered bc_reference and boundary anomaly must be finite.")
    if not np.isfinite(mass.values).all() or (mass.values < 0).any() or not (mass.values > 0).any():
        raise ValueError("H_bc_mass must be finite, non-negative and not all zero.")
    shift = (reference * mass.sum(state_dim, skipna=False)).assign_attrs(reference.attrs)
    if not np.isfinite(shift.values).all():
        raise ValueError("Boundary reference shift must be finite.")
    return shift


def bind_centered_transport(
    transport: RawCenteredTransport,
    observations: xr.DataArray,
    *,
    frequency: str | None = None,
    anchor_time: DatetimeLike | None = None,
) -> BoundCenteredTransport:
    """Select one fit's observation rows and expand spatial states by period.

    Observation dimension labels own the row order; their auxiliary ``time``
    values own period membership when supplied. Indexed boundary times must
    match those timestamps; this check may compute a lazy timestamp
    coordinate, while numerical operands remain borrowed and lazy.

    The result is established by construction: row selection by unique
    observation labels and the shared period mask produce the
    ``(observation, bc_region)`` layout with a ``(bc_curtain, bc_period)``
    state index for every curtain in every period, and the scalar reference
    is carried unchanged. Raw centering weights are not included; see
    :func:`bind_centering_weights`.

    Args:
        transport: Raw centered transport on its ``time`` or ``nmeasure`` axis.
        observations: One native channel with its observation axis first.
        frequency: Optional frequency expanding spatial states by period.
        anchor_time: Optional origin for fixed-duration periods.

    Returns:
        ``H_bc_anomaly`` and ``H_bc_mass`` on ``(observation, bc_region)`` with
        the bound state index, plus the scalar ``bc_reference``.

    Raises:
        ValueError: If PyTensor is not configured for float64, or periods or
            indexed timestamps conflict with the observations.
    """
    require_centered_precision()
    native_dim = str(observations.dims[0])
    raw_dim = "time" if "time" in transport.H_bc_anomaly.dims else "nmeasure"
    raw = _select_native_boundary(transport[["H_bc_anomaly", "H_bc_mass"]], observations, raw_dim)
    bound = _expand_boundary_periods(
        raw, observation_dim=raw_dim, frequency=frequency, anchor_time=anchor_time, observation_labels=None,
    )
    if raw_dim != native_dim:
        # Native time has already determined periods; the restored dimension
        # index owns those same labels, so its auxiliary copy is redundant.
        if native_dim == "time" and "time" in bound.coords:
            bound = bound.drop_vars("time")
        bound = bound.rename({raw_dim: native_dim})
    bound = bound.transpose(native_dim, "bc_region").assign(bc_reference=transport.bc_reference)
    return cast(BoundCenteredTransport, bound)


def bind_centering_weights(bound: BoundCenteredTransport, weights: xr.DataArray) -> BoundAdditiveTransport:
    """Align raw centering weights to bound states, giving the additive extension.

    Weights must label exactly the bound curtains and periods, so a calendar
    mismatch is reported here rather than silently reindexed. The bound state
    index is reused for the stacked weights; their values stay borrowed.

    Args:
        bound: Bound centered transport.
        weights: Raw weights on ``(bc_curtain, bc_period)``, structurally
            checked with the raw transport.

    Returns:
        ``bound`` with ``bc_centering_weights`` on ``bc_region``.

    Raises:
        ValueError: If the weights' curtain and period labels differ from the
            bound states.
    """
    periods = weights.indexes["bc_period"]
    if isinstance(periods, pd.DatetimeIndex):
        # Period starts from the fit calendar are nanosecond timestamps.
        weights = weights.assign_coords(bc_period=periods.as_unit("ns"))
    stacked = weights.stack(bc_region=STATE_LEVELS)
    observation_dim = BoundCenteredTransport.observation_dim(bound)
    xr.align(bound.H_bc_anomaly, stacked, join="exact", copy=False, exclude={observation_dim})
    # Labels match exactly, so the bound state index is reused for the weights.
    return cast(BoundAdditiveTransport, bound.assign(bc_centering_weights=stacked.variable))


__all__ = [
    "bind_centered_transport",
    "bind_centering_weights",
    "check_bound_transport",
    "centered_boundary_expression",
    "materialized_reference_shift",
    "require_centered_precision",
]
