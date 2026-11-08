"""Shared precision requirement and centered multiplicative expression.

Transport schemas, binding and numerical preparation belong to the domain
classes in :mod:`.contracts`.
"""

from __future__ import annotations

import pytensor
import xarray as xr


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


__all__ = [
    "check_bound_transport",
    "centered_boundary_expression",
    "require_centered_precision",
]
