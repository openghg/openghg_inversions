"""Experimental xarray conversion for retained spatial RJMCMC states.

The converter preserves fixed-capacity spatial state, fixed coefficients,
optional inferred-OU parameters, and optional shared coefficient-hierarchy
coordinates on labelled dimensions. Proposal diagnostics describe every
attempted transition, so they deliberately remain outside the draw-indexed
dataset until a separate transition-diagnostics output contract is defined.
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


def _validated_optional_target_arrays(
    trace: SamplingTrace,
    *,
    n_draws: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    bool,
]:
    """Validate retained OU parameters and shared-hierarchy coordinates.

    Args:
        trace: Sampling trace containing optional retained target parameters.
        n_draws: Expected number of retained-state rows.

    Returns:
        Owned mismatch amplitudes, correlation timescales, eta and zeta
        arrays, followed by the explicit hierarchy-activation flag.

    Raises:
        ValueError: If an array has the wrong rank or retained-row count, an
            OU parameter is nonfinite or nonpositive, or hierarchy coordinates
            are inconsistent with the activation flag.
    """
    mismatch_sd = _float_array(trace.mismatch_sd, name="mismatch_sd", ndim=2)
    correlation_timescale = _float_array(
        trace.correlation_timescale,
        name="correlation_timescale",
        ndim=2,
    )
    if mismatch_sd.shape[0] != n_draws:
        raise ValueError("trace.mismatch_sd must have one row per retained state.")
    if correlation_timescale.shape[0] != n_draws:
        raise ValueError("trace.correlation_timescale must have one row per retained state.")
    if np.any(~np.isfinite(mismatch_sd)) or np.any(mismatch_sd <= 0.0):
        raise ValueError("trace.mismatch_sd must contain finite positive values.")
    if np.any(~np.isfinite(correlation_timescale)) or np.any(correlation_timescale <= 0.0):
        raise ValueError("trace.correlation_timescale must contain finite positive values.")

    eta = _float_array(trace.eta, name="eta", ndim=1)
    zeta = _float_array(trace.zeta, name="zeta", ndim=1)
    if eta.shape != (n_draws,):
        raise ValueError("trace.eta must have one value per retained state.")
    if zeta.shape != (n_draws,):
        raise ValueError("trace.zeta must have one value per retained state.")
    hierarchy_active = bool(trace.coefficient_hierarchy_active)
    if hierarchy_active:
        if not np.all(np.isfinite(eta)):
            raise ValueError("trace.eta must be finite when the coefficient hierarchy is active.")
        if not np.all(np.isfinite(zeta)):
            raise ValueError("trace.zeta must be finite when the coefficient hierarchy is active.")
    elif not np.all(np.isnan(eta)) or not np.all(np.isnan(zeta)):
        raise ValueError("trace.eta and trace.zeta must contain only NaN when the hierarchy is inactive.")
    return mismatch_sd, correlation_timescale, eta, zeta, hierarchy_active


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
    without a fixed block. Inferred OU parameters similarly use labelled
    ``mismatch_group`` and ``timescale_parameter`` dimensions, which have zero
    width for an independent-error trace. Shared coefficient-prior hierarchy
    coordinates are retained as ``eta = log(M)`` and ``zeta = log(S)``, where
    ``M`` and ``S`` are arithmetic moments; derived arithmetic moments are
    included for convenience.

    Args:
        trace: Retained fixed-capacity states and segment diagnostics.
        metadata: Optional caller-supplied dataset attributes. No run metadata
            is inferred by this converter.

    Returns:
        Dataset containing the spatial state, fixed coefficients, optional OU
        and shared-hierarchy parameters, and ``log_target``.

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
    mismatch_sd, correlation_timescale, eta, zeta, hierarchy_active = _validated_optional_target_arrays(
        trace, n_draws=n_draws
    )
    with np.errstate(over="ignore"):
        coefficient_prior_mean = np.exp(eta)
        coefficient_prior_sd = np.exp(zeta)
    if hierarchy_active and (
        np.any(~np.isfinite(coefficient_prior_mean))
        or np.any(coefficient_prior_mean <= 0.0)
        or np.any(~np.isfinite(coefficient_prior_sd))
        or np.any(coefficient_prior_sd <= 0.0)
    ):
        raise ValueError(
            "trace.eta and trace.zeta must imply finite positive arithmetic coefficient-prior moments."
        )

    dataset = xr.Dataset(
        data_vars={
            "k": ("draw", k),
            "nuclei": (("draw", "region_slot"), nuclei),
            "coefficients": (("draw", "region_slot"), coefficients),
            "active": (("draw", "region_slot"), active),
            "fixed_coefficients": (
                ("draw", "fixed_parameter"),
                fixed_coefficients,
            ),
            "mismatch_sd": (("draw", "mismatch_group"), mismatch_sd),
            "correlation_timescale": (
                ("draw", "timescale_parameter"),
                correlation_timescale,
            ),
            "eta": ("draw", eta),
            "zeta": ("draw", zeta),
            "coefficient_prior_mean": ("draw", coefficient_prior_mean),
            "coefficient_prior_sd": ("draw", coefficient_prior_sd),
            "coefficient_hierarchy_active": hierarchy_active,
            "log_target": ("draw", log_target),
        },
        coords={
            "draw": np.arange(n_draws, dtype=np.int64),
            "region_slot": np.arange(capacity, dtype=np.int64),
            "fixed_parameter": np.arange(n_fixed, dtype=np.int64),
            "mismatch_group": np.arange(mismatch_sd.shape[1], dtype=np.int64),
            "timescale_parameter": np.arange(
                correlation_timescale.shape[1],
                dtype=np.int64,
            ),
            "state_transition": ("draw", state_transition),
        },
        attrs={} if metadata is None else dict(metadata),
    )
    dataset["mismatch_sd"].attrs = {
        "long_name": "OU model-data mismatch standard deviation",
        "description": "Amplitude of the latent unit-variance OU process for each mismatch group.",
    }
    dataset["correlation_timescale"].attrs = {
        "long_name": "OU correlation timescale",
        "description": "OU correlation timescale in the units of the likelihood observation time.",
    }
    dataset["eta"].attrs = {
        "long_name": "log arithmetic coefficient-prior mean",
        "description": "eta = log(M), where M is the shared lognormal prior's arithmetic mean.",
    }
    dataset["zeta"].attrs = {
        "long_name": "log arithmetic coefficient-prior standard deviation",
        "description": "zeta = log(S), where S is the shared lognormal prior's arithmetic SD.",
    }
    dataset["coefficient_prior_mean"].attrs = {
        "long_name": "arithmetic coefficient-prior mean",
        "description": "M = exp(eta) for the shared dynamic-coefficient prior.",
    }
    dataset["coefficient_prior_sd"].attrs = {
        "long_name": "arithmetic coefficient-prior standard deviation",
        "description": "S = exp(zeta) for the shared dynamic-coefficient prior.",
    }
    dataset["coefficient_hierarchy_active"].attrs = {
        "long_name": "shared coefficient hierarchy active",
        "description": "Whether eta and zeta are inferred shared-hierarchy coordinates.",
    }
    return dataset


__all__ = ["sampling_trace_to_dataset"]
