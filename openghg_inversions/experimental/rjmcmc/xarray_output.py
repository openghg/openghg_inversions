"""Experimental xarray conversion for spatial RJMCMC output.

The converter preserves fixed-capacity spatial state, fixed coefficients,
optional inferred-OU parameters, and optional shared coefficient-hierarchy
coordinates on labelled dimensions. Structural proposal diagnostics use a
separate transition-indexed dataset because they describe attempted candidates
rather than retained posterior draws.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from openghg_inversions.experimental.rjmcmc.mixing_diagnostics import (
    STRUCTURAL_INVALID_REASON_LABELS,
    StructuralDiagnostics,
    StructuralDiagnosticsProvenance,
)
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
    dataset["mismatch_group"].attrs = {
        "long_name": "ordinal mismatch-amplitude group",
        "description": (
            "Zero-based internal group index; scientific group identities must be supplied "
            "by the run manifest or frozen-input metadata."
        ),
    }
    dataset["timescale_parameter"].attrs = {
        "long_name": "ordinal OU timescale parameter",
        "description": (
            "Zero-based internal parameter index; site mapping and time units must be supplied "
            "by the run manifest or frozen-input metadata."
        ),
    }
    return dataset


def structural_diagnostics_to_dataset(
    diagnostics: StructuralDiagnostics,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """Convert structural proposal diagnostics to a labelled dataset.

    The ``structural_transition`` coordinate is the global completed atomic
    transition number for each structural attempt. It is intentionally
    independent of the retained-state ``draw`` dimension produced by
    :func:`sampling_trace_to_dataset`. Save one dataset per segment; for
    chain-wide derivations, restore each with
    :func:`structural_diagnostics_from_dataset` and concatenate the diagnostic
    objects. Direct xarray concatenation would broadcast the variable-length
    segment endpoint nucleus sets.

    Args:
        diagnostics: Immutable proposal-level structural diagnostics.
        metadata: Optional caller-supplied attributes. Reserved schema and
            problem-shape attributes cannot be overridden.

    Returns:
        Dataset containing candidate, acceptance, target-accounting, geometry,
        and prediction-space metrics plus segment endpoint nucleus sets.

    Raises:
        TypeError: If ``diagnostics`` or ``metadata`` has the wrong type.
        ValueError: If caller metadata conflicts with a reserved attribute.
    """
    if not isinstance(diagnostics, StructuralDiagnostics):
        raise TypeError("diagnostics must be a StructuralDiagnostics instance.")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None.")

    attrs = {} if metadata is None else dict(metadata)
    reserved_attrs: dict[str, Any] = {
        "schema": "openghg_inversions_rjmcmc_structural_diagnostics_v1",
        "n_grid_cells": diagnostics.n_grid_cells,
        "n_observations": diagnostics.n_observations,
        "segment_transition_start": diagnostics.segment_transition_start,
        "segment_transition_end": diagnostics.segment_transition_end,
        "chain_id": diagnostics.provenance.chain_id,
        "problem_fingerprint": diagnostics.provenance.problem_fingerprint,
        "transition_coordinate": ("global completed atomic-transition number after the structural proposal"),
        "prediction_standardization": (
            "elementwise observation_sd; complete covariance whitening only "
            "for the fixed diagonal error model"
        ),
        "coefficient_contrast": (
            "signed log ratio of the event coefficient to its comparison-region coefficient"
        ),
    }
    conflicts = {name for name, value in reserved_attrs.items() if name in attrs and attrs[name] != value}
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"metadata conflicts with reserved attributes: {names}.")
    attrs.update(reserved_attrs)

    endpoint_names = {"initial_nuclei", "final_nuclei"}
    metadata_names = {
        "n_grid_cells",
        "n_observations",
        "segment_transition_start",
        "segment_transition_end",
        "provenance",
    }
    data_vars: dict[str, Any] = {}
    for diagnostic_field in fields(StructuralDiagnostics):
        name = diagnostic_field.name
        if name in endpoint_names or name in metadata_names or name == "transition":
            continue
        data_vars[name] = ("structural_transition", getattr(diagnostics, name))
    data_vars["initial_nuclei"] = ("initial_region", diagnostics.initial_nuclei)
    data_vars["final_nuclei"] = ("final_region", diagnostics.final_nuclei)

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "structural_transition": diagnostics.transition,
            "initial_region": np.arange(diagnostics.initial_nuclei.size, dtype=np.int64),
            "final_region": np.arange(diagnostics.final_nuclei.size, dtype=np.int64),
        },
        attrs=attrs,
    )
    dataset["structural_transition"].attrs = {
        "long_name": "global completed atomic-transition number",
    }
    dataset["owner_changed_cell_count"].attrs = {
        "long_name": "native cells with a changed owner nucleus identity",
    }
    dataset["owner_changed_cell_fraction"].attrs = {
        "long_name": "fraction of native cells with a changed owner nucleus identity",
    }
    dataset["invalid_reason_code"].attrs = {
        "long_name": "structural proposal invalidity code",
        "description": "; ".join(
            [
                *(
                    f"{code}={label or 'valid'}"
                    for code, label in sorted(STRUCTURAL_INVALID_REASON_LABELS.items())
                ),
            ]
        ),
    }
    dataset["observation_error_standardized_prediction_change_l2"].attrs = {
        "long_name": "observation-error-standardized candidate prediction change",
        "description": (
            "Euclidean norm after elementwise division by observation_sd; not full OU covariance whitening."
        ),
    }
    dataset["coefficient_contrast"].attrs = {
        "long_name": "event-region log coefficient ratio",
    }
    return dataset


def structural_diagnostics_from_dataset(
    dataset: xr.Dataset,
    *,
    required_metadata: Mapping[str, Any] | None = None,
) -> StructuralDiagnostics:
    """Restore validated structural diagnostics from their xarray contract.

    Args:
        dataset: Dataset produced by
            :func:`structural_diagnostics_to_dataset`.
        required_metadata: Optional run/chain/problem attributes that must
            match exactly before the diagnostic object is restored.

    Returns:
        Immutable structural diagnostics suitable for chronological
        concatenation and derived mixing summaries.

    Raises:
        TypeError: If ``dataset`` is not an xarray dataset or
            ``required_metadata`` is not a mapping or ``None``.
        ValueError: If the schema, dimensions, variables, or problem metadata
            are missing or malformed.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("dataset must be an xarray Dataset.")
    if required_metadata is not None and not isinstance(required_metadata, Mapping):
        raise TypeError("required_metadata must be a mapping or None.")
    if required_metadata is not None:
        mismatches = {name for name, value in required_metadata.items() if dataset.attrs.get(name) != value}
        if mismatches:
            names = ", ".join(sorted(mismatches))
            raise ValueError(f"dataset does not match required metadata: {names}.")
    expected_schema = "openghg_inversions_rjmcmc_structural_diagnostics_v1"
    if dataset.attrs.get("schema") != expected_schema:
        raise ValueError(f"dataset schema must equal {expected_schema!r}.")
    metadata_names = {
        "n_grid_cells",
        "n_observations",
        "segment_transition_start",
        "segment_transition_end",
        "chain_id",
        "problem_fingerprint",
    }
    for name in metadata_names:
        if name not in dataset.attrs:
            raise ValueError(f"dataset is missing required attribute {name!r}.")
    if "structural_transition" not in dataset.coords:
        raise ValueError("dataset is missing the structural_transition coordinate.")

    endpoint_dims = {
        "initial_nuclei": ("initial_region",),
        "final_nuclei": ("final_region",),
    }
    values: dict[str, Any] = {
        "transition": np.asarray(dataset.coords["structural_transition"].values),
    }
    for diagnostic_field in fields(StructuralDiagnostics):
        name = diagnostic_field.name
        if name == "transition" or name in metadata_names or name == "provenance":
            continue
        if name not in dataset:
            raise ValueError(f"dataset is missing required variable {name!r}.")
        expected_dims = endpoint_dims.get(name, ("structural_transition",))
        if dataset[name].dims != expected_dims:
            raise ValueError(f"dataset variable {name!r} must have dimensions {expected_dims!r}.")
        values[name] = np.asarray(dataset[name].values)
    for name in metadata_names - {"chain_id", "problem_fingerprint"}:
        values[name] = dataset.attrs[name]
    values["provenance"] = StructuralDiagnosticsProvenance(
        chain_id=dataset.attrs["chain_id"],
        problem_fingerprint=dataset.attrs["problem_fingerprint"],
    )
    return StructuralDiagnostics(**values)


__all__ = [
    "sampling_trace_to_dataset",
    "structural_diagnostics_from_dataset",
    "structural_diagnostics_to_dataset",
]
