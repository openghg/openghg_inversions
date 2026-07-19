"""Experimental xarray conversion for retained spatial RJMCMC states.

The converter in this module represents only retained sampler states. Proposal
diagnostics describe every attempted transition, so they deliberately remain
outside the draw-indexed dataset until a separate transition-diagnostics
output contract is defined.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from openghg_inversions.experimental.rjmcmc.sampling import SamplingTrace


def _integer_array(values: object, *, name: str, ndim: int) -> NDArray[np.int64]:
    """Return an owned integer array with the required rank."""
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ValueError(f"trace.{name} must be {ndim}-dimensional.")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"trace.{name} must contain integers.")
    return np.array(array, dtype=np.int64, copy=True)


def _float_array(values: object, *, name: str, ndim: int) -> NDArray[np.float64]:
    """Return an owned real-valued array with the required rank."""
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ValueError(f"trace.{name} must be {ndim}-dimensional.")
    if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
        raise ValueError(f"trace.{name} must contain real numeric values.")
    return np.array(array, dtype=np.float64, copy=True)


def _validate_diagnostics(trace: SamplingTrace) -> None:
    """Validate diagnostic array ranks and their shared segment length."""
    moves = np.asarray(trace.moves)
    accepted = np.asarray(trace.accepted)
    log_acceptance_ratio = np.asarray(trace.log_acceptance_ratio)
    if moves.ndim != 1:
        raise ValueError("trace.moves must be one-dimensional.")
    if accepted.ndim != 1 or not np.issubdtype(accepted.dtype, np.bool_):
        raise ValueError("trace.accepted must be a one-dimensional boolean array.")
    if log_acceptance_ratio.ndim != 1 or not (
        np.issubdtype(log_acceptance_ratio.dtype, np.floating)
        or np.issubdtype(log_acceptance_ratio.dtype, np.integer)
    ):
        raise ValueError("trace.log_acceptance_ratio must be a one-dimensional real array.")
    if not (moves.size == accepted.size == log_acceptance_ratio.size):
        raise ValueError("trace transition diagnostics must have the same length.")
    if np.any(np.isnan(np.asarray(log_acceptance_ratio, dtype=np.float64))):
        raise ValueError("trace.log_acceptance_ratio must not contain NaN values.")


def _validated_retained_arrays(
    trace: SamplingTrace,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.bool_],
]:
    """Validate retained-state shapes, supports, and fixed-capacity padding."""
    k = _integer_array(trace.k, name="k", ndim=1)
    nuclei = _integer_array(trace.nuclei, name="nuclei", ndim=2)
    coefficients = _float_array(trace.coefficients, name="coefficients", ndim=2)
    fixed_coefficients = _float_array(
        trace.fixed_coefficients,
        name="fixed_coefficients",
        ndim=2,
    )
    log_target = _float_array(trace.log_target, name="log_target", ndim=1)
    state_transition = _integer_array(
        trace.state_transition,
        name="state_transition",
        ndim=1,
    )
    n_draws = k.size

    if nuclei.shape[0] != n_draws:
        raise ValueError("trace.nuclei must have one row per retained state.")
    if coefficients.shape != nuclei.shape:
        raise ValueError("trace.coefficients must have the same shape as trace.nuclei.")
    if fixed_coefficients.shape[0] != n_draws:
        raise ValueError("trace.fixed_coefficients must have one row per retained state.")
    if log_target.shape != (n_draws,):
        raise ValueError("trace.log_target must have one value per retained state.")
    if state_transition.shape != (n_draws,):
        raise ValueError("trace.state_transition must have one value per retained state.")
    if np.any(state_transition < 0) or np.any(np.diff(state_transition) <= 0):
        raise ValueError("trace.state_transition must be non-negative and strictly increasing.")
    if not np.all(np.isfinite(log_target)):
        raise ValueError("trace.log_target must contain only finite values.")

    capacity = nuclei.shape[1]
    if np.any((k < 1) | (k > capacity)):
        raise ValueError("trace.k must lie within the represented region-slot capacity.")
    active = np.arange(capacity, dtype=np.int64)[np.newaxis, :] < k[:, np.newaxis]
    if np.any(nuclei[active] < 0):
        raise ValueError("Active trace.nuclei entries must be non-negative.")
    if np.any(nuclei[~active] != -1):
        raise ValueError("Inactive trace.nuclei padding must equal -1.")
    if np.any(coefficients[~active] != 0.0):
        raise ValueError("Inactive trace.coefficients padding must equal zero.")
    if np.any(~np.isfinite(coefficients[active])) or np.any(coefficients[active] <= 0.0):
        raise ValueError("Active trace.coefficients entries must be finite and positive.")
    if np.any(~np.isfinite(fixed_coefficients)) or np.any(fixed_coefficients <= 0.0):
        raise ValueError("trace.fixed_coefficients must contain finite positive values.")

    for row, row_k in enumerate(k):
        active_nuclei = nuclei[row, : int(row_k)]
        if active_nuclei.size > 1 and np.any(np.diff(active_nuclei) <= 0):
            raise ValueError("Active trace.nuclei entries must be strictly increasing.")

    return (
        k,
        nuclei,
        coefficients,
        fixed_coefficients,
        log_target,
        state_transition,
        active,
    )


def sampling_trace_to_dataset(
    trace: SamplingTrace,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Convert retained RJMCMC states to an experimental xarray dataset.

    The ``draw`` coordinate is a dense local row number. The associated
    ``state_transition`` coordinate records the global number of completed
    transitions, so collection-time warmup and thinning remain explicit.
    Dynamic-region values retain their fixed-capacity padding along
    ``region_slot``. Always-active coefficients use a separate
    ``fixed_parameter`` dimension, including a zero-width dimension for traces
    without a fixed block.

    Args:
        trace: Retained fixed-capacity states and segment diagnostics.
        metadata: Optional caller-supplied dataset attributes. No run metadata
            is inferred by this converter.

    Returns:
        Dataset containing ``k``, padded ``nuclei`` and ``coefficients``, an
        ``active`` mask, ``fixed_coefficients``, and ``log_target``.

    Raises:
        TypeError: If ``trace`` or ``metadata`` has the wrong type.
        ValueError: If retained arrays, padding, supports, transition numbers,
            or diagnostic shapes are malformed.
    """
    if not isinstance(trace, SamplingTrace):
        raise TypeError("trace must be a SamplingTrace instance.")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None.")

    _validate_diagnostics(trace)
    (
        k,
        nuclei,
        coefficients,
        fixed_coefficients,
        log_target,
        state_transition,
        active,
    ) = _validated_retained_arrays(trace)

    n_draws, capacity = nuclei.shape
    n_fixed = fixed_coefficients.shape[1]
    return xr.Dataset(
        data_vars={
            "k": ("draw", k),
            "nuclei": (("draw", "region_slot"), nuclei),
            "coefficients": (("draw", "region_slot"), coefficients),
            "active": (("draw", "region_slot"), active),
            "fixed_coefficients": (
                ("draw", "fixed_parameter"),
                fixed_coefficients,
            ),
            "log_target": ("draw", log_target),
        },
        coords={
            "draw": np.arange(n_draws, dtype=np.int64),
            "region_slot": np.arange(capacity, dtype=np.int64),
            "fixed_parameter": np.arange(n_fixed, dtype=np.int64),
            "state_transition": ("draw", state_transition),
        },
        attrs={} if metadata is None else dict(metadata),
    )


__all__ = ["sampling_trace_to_dataset"]
