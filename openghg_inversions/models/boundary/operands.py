"""Shared precision requirement and centered multiplicative expression.

Transport schemas, binding and numerical preparation belong to the domain
classes in :mod:`.contracts`.
"""

from __future__ import annotations

import pytensor


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


__all__ = [
    "centered_boundary_expression",
    "require_centered_precision",
]
