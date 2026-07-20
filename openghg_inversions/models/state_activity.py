"""Label-aware active/fixed state-vector policies for linear inversion models.

This module separates state selection from PyMC graph construction. A
``StateActivity`` policy can combine exact-zero sensitivity pruning with an
explicit labelled activity mask and fixed ``basis_group`` labels. Detection
follows the state coordinate order on the supplied sensitivity matrix, and
resolution follows the resulting labelled mask; integer label ranges are never
inferred.

Inactive states remain part of the public state vector and use labelled or
scalar fixed values. The default fixed value is one, which preserves the prior
forward-model contribution of a multiplicative flux-scaling state.

``detect_zero_sensitivity`` validates a linear design and reduces it to a
labelled state mask. ``resolve_state_activity`` combines that mask with a
policy to produce the canonical ``ResolvedStateActivity``;
``active_prior_args`` then aligns and subsets state-valued prior parameters.
Finite checks, exact-zero reductions, and state-vector alignment are explicit
eager-compute boundaries for lazy or Dask-backed inputs during model building.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr


#: Strictly boolean scalar, positional mask, or labelled mask.
ActivityValue = bool | np.ndarray | xr.DataArray
#: Numeric scalar, positional vector, or labelled vector for inactive states.
FixedValue = float | int | np.ndarray | xr.DataArray


@dataclass(frozen=True)
class StateActivity:
    """Describe which states are sampled and how inactive states are fixed.

    Args:
        active: Boolean scalar, positional one-dimensional mask in canonical
            state order, or labelled one-dimensional mask. A labelled mask is
            aligned to the canonical state coordinate carried by the
            zero-sensitivity mask, so its input order need not match the
            canonical order. Set this to ``False`` to freeze a whole component
            or sector.
        fixed_value: Scalar or one-dimensional state-aligned values used for
            inactive states. Multiplicative scaling states default to one.
        fixed_groups: ``basis_group`` labels to freeze. Group selection is by
            coordinate value, never by state-number ranges.
        group_coord: Name of the state coordinate containing group labels.
        prune_zero: Whether states marked exactly zero by prior design
            inspection are made inactive. No tolerance is applied; every
            nonzero finite column remains active by default.

    Explicit ``active`` masks, fixed groups, and exact-zero pruning are
    combined with logical AND. This permits one policy to represent zero-H
    pruning, frozen states, frozen groups, or an entirely frozen sector.
    """

    active: ActivityValue = True
    fixed_value: FixedValue = 1.0
    fixed_groups: tuple[str, ...] = ()
    group_coord: str = "basis_group"
    prune_zero: bool = True


@dataclass(frozen=True)
class ResolvedStateActivity:
    """A state-activity policy aligned to one detected linear design.

    Attributes:
        state_dim: Canonical state dimension name from the detection mask.
        active: Boolean mask in canonical state order.
        fixed_value: Fixed values in canonical state order.
        zero_sensitivity: Labelled mask identifying exactly-zero design
            columns and carrying the state coordinates used for graph
            construction.
    """

    state_dim: str
    active: xr.DataArray
    fixed_value: xr.DataArray
    zero_sensitivity: xr.DataArray

    @property
    def n_state(self) -> int:
        """Return the full state-vector length."""
        return int(self.active.sizes[self.state_dim])

    @property
    def n_active(self) -> int:
        """Return the number of sampled states."""
        return int(self.active.sum().compute().item())

    @property
    def active_indices(self) -> np.ndarray:
        """Return positional indices of active states in canonical order."""
        return np.flatnonzero(_materialize_1d(self.active, name="active", dtype=bool))

    @property
    def fixed_indices(self) -> np.ndarray:
        """Return positional indices of inactive states in canonical order."""
        return np.flatnonzero(~_materialize_1d(self.active, name="active", dtype=bool))


def _state_dim(sensitivity: xr.DataArray, output_dim: str) -> str:
    """Return the sole non-output dimension of a linear sensitivity matrix.

    Args:
        sensitivity: Candidate linear sensitivity matrix.
        output_dim: Name of its observation/output dimension.

    Returns:
        The single state-dimension name.

    Raises:
        ValueError: If the matrix does not have exactly one non-output dimension.
    """
    state_dims = [str(dim) for dim in sensitivity.dims if dim != output_dim]
    if len(state_dims) != 1:
        raise ValueError(
            "State activity requires a linear sensitivity matrix with exactly "
            f"one non-output dimension; found {state_dims!r}."
        )
    return state_dims[0]


def detect_zero_sensitivity(
    sensitivity: xr.DataArray,
    *,
    output_dim: str = "nmeasure",
) -> xr.DataArray:
    """Return the labelled mask of exactly-zero sensitivity columns.

    Args:
        sensitivity: Finite two-dimensional linear design containing
            ``output_dim`` and one uniquely labelled state dimension.
        output_dim: Name of the observation/output dimension.

    Returns:
        A materialized boolean mask over the state dimension. State labels and
        auxiliary state coordinates are retained from ``sensitivity``.

    Raises:
        ValueError: If ``sensitivity`` is not a finite two-dimensional design
            with the required output dimension and unique state labels.

    Notes:
        Finite validation and exact-zero reduction materialize lazy design data
        during model construction.
    """
    output_dim = str(output_dim)
    if sensitivity.ndim != 2 or output_dim not in sensitivity.dims:
        raise ValueError(
            "State activity requires a two-dimensional linear sensitivity matrix "
            f"containing output dimension {output_dim!r}; found dimensions {sensitivity.dims!r}."
        )

    state_dim = _state_dim(sensitivity, output_dim)
    matrix = sensitivity.transpose(output_dim, state_dim)
    _require_unique_state_coord(matrix, state_dim)

    try:
        finite_mask = cast(xr.DataArray, np.isfinite(matrix))
    except TypeError as exc:
        raise ValueError("Sensitivity must contain numeric values.") from exc
    if not bool(finite_mask.all().compute().item()):
        raise ValueError("Sensitivity must contain only finite values.")

    return (matrix == 0).all(dim=output_dim).compute().rename("zero_sensitivity")


def _materialize_1d(
    value: xr.DataArray,
    *,
    name: str,
    dtype: Any = None,
) -> np.ndarray:
    """Compute a one-dimensional array and return its NumPy representation.

    Args:
        value: Scalar or one-dimensional xarray value to materialize.
        name: User-facing value name included in validation errors.
        dtype: Optional NumPy dtype for the result.

    Returns:
        A materialized one-dimensional NumPy array. Scalar input is represented
        by a length-one array.

    Raises:
        ValueError: If ``value`` has more than one dimension.
    """
    if value.ndim > 1:
        raise ValueError(f"{name} must be scalar or one-dimensional; found shape {value.shape!r}.")
    return np.asarray(value.compute().to_numpy(), dtype=dtype).reshape(-1)


def _require_unique_state_coord(sensitivity: xr.DataArray, state_dim: str) -> pd.Index:
    """Return a present, one-dimensional, unique canonical state index.

    Args:
        sensitivity: Labelled array containing the state coordinate.
        state_dim: Canonical state-dimension name.

    Returns:
        The validated state index.

    Raises:
        ValueError: If the coordinate is absent, not state-only, or not unique.
    """
    if state_dim not in sensitivity.coords:
        raise ValueError(f"Sensitivity must have labelled {state_dim!r} coordinates.")
    state_coord = sensitivity.coords[state_dim]
    if state_coord.dims != (state_dim,):
        raise ValueError(f"State coordinate {state_dim!r} must be indexed only by {state_dim!r}.")
    state_index = state_coord.to_index()
    if not state_index.is_unique:
        raise ValueError(f"Sensitivity {state_dim!r} coordinates must be unique.")
    return state_index


def _require_same_state_labels(
    value: xr.DataArray,
    *,
    canonical_index: pd.Index,
    state_dim: str,
    name: str,
) -> None:
    """Validate that a labelled value defines exactly the canonical states.

    Args:
        value: Labelled state value to validate.
        canonical_index: Required state labels.
        state_dim: Canonical state-dimension name.
        name: User-facing value name for validation errors.

    Raises:
        ValueError: If supplied labels are duplicate, missing, or additional.
    """
    value_index = value.coords[state_dim].to_index()
    if not value_index.is_unique:
        raise ValueError(f"{name} {state_dim!r} coordinates must be unique.")
    same_labels = (
        len(value_index) == len(canonical_index)
        and bool(canonical_index.isin(value_index).all())
        and bool(value_index.isin(canonical_index).all())
    )
    if not same_labels:
        raise ValueError(f"{name} labels must match the canonical {state_dim!r} coordinate exactly.")


def _require_finite(values: np.ndarray, *, name: str) -> None:
    """Reject non-numeric, NaN, or infinite array values.

    Args:
        values: Numeric array to validate.
        name: User-facing value name for validation errors.

    Raises:
        ValueError: If values are non-numeric or non-finite.
    """
    try:
        finite = np.isfinite(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if not bool(finite.all()):
        raise ValueError(f"{name} must contain only finite values.")


def _require_boolean(values: xr.DataArray, *, name: str) -> None:
    """Require a strictly boolean aligned state value.

    Args:
        values: Aligned value to validate without coercion.
        name: User-facing value name for validation errors.

    Raises:
        ValueError: If the value dtype is not boolean.
    """
    if np.asarray(values.to_numpy()).dtype != np.dtype(bool):
        raise ValueError(f"{name} must contain only boolean values.")


def _align_state_value(
    value: ActivityValue | FixedValue,
    *,
    sensitivity: xr.DataArray,
    state_dim: str,
    name: str,
    dtype: Any,
) -> xr.DataArray:
    """Broadcast or label-align and materialize a canonical state value.

    Args:
        value: Scalar, positional vector, or labelled state vector.
        sensitivity: Labelled array providing canonical state labels and
            length.
        state_dim: Canonical state-dimension name.
        name: User-facing value name for validation errors.
        dtype: Optional NumPy dtype used after alignment.

    Returns:
        Materialized one-dimensional data in canonical state order.

    Raises:
        ValueError: If shape, dimensions, or labels do not match the state.
    """
    state_coord = sensitivity.coords[state_dim]
    canonical_index = _require_unique_state_coord(sensitivity, state_dim)
    state_size = sensitivity.sizes[state_dim]

    if isinstance(value, xr.DataArray):
        if value.ndim == 0:
            scalar = value.compute().item()
            values = np.full(state_size, scalar, dtype=dtype)
        else:
            if value.dims != (state_dim,):
                raise ValueError(
                    f"{name} must be scalar or have only the {state_dim!r} dimension; "
                    f"found dimensions {value.dims!r}."
                )
            if state_dim not in value.coords:
                raise ValueError(f"{name} must have labelled {state_dim!r} coordinates.")
            _require_same_state_labels(
                value,
                canonical_index=canonical_index,
                state_dim=state_dim,
                name=name,
            )
            aligned = value.sel({state_dim: state_coord})
            values = _materialize_1d(aligned, name=name, dtype=dtype)
    else:
        array = np.asarray(value)
        if array.ndim == 0:
            values = np.full(state_size, array.item(), dtype=dtype)
        elif array.ndim == 1 and array.size == state_size:
            values = np.asarray(array, dtype=dtype)
        else:
            raise ValueError(
                f"{name} must be scalar or a one-dimensional array of length {state_size}; "
                f"found shape {array.shape!r}."
            )

    return xr.DataArray(
        values,
        dims=(state_dim,),
        coords={state_dim: state_coord},
        name=name,
    )


def resolve_state_activity(
    zero_sensitivity: xr.DataArray,
    policy: StateActivity | None = None,
) -> ResolvedStateActivity:
    """Resolve an active/fixed policy against a labelled zero-state mask.

    Args:
        zero_sensitivity: One-dimensional, strictly boolean mask identifying
            exactly-zero design columns. It must have a unique labelled state
            coordinate and may carry auxiliary state coordinates used by the
            policy.
        policy: Optional activity policy. When omitted, exact-zero columns are
            pruned and all other states are active with inactive value one.

    Returns:
        A policy aligned to the mask's canonical state coordinate.

    Raises:
        ValueError: If the zero mask is not one-dimensional, boolean, and
            uniquely labelled; supplied arrays cannot be aligned; or requested
            group metadata is absent or not state-aligned.

    Notes:
        Policy vectors are materialized during model construction. Use
        ``detect_zero_sensitivity`` to validate and reduce a linear design
        before calling this function.
    """
    policy = policy or StateActivity()
    if zero_sensitivity.ndim != 1:
        raise ValueError(
            f"zero_sensitivity must be one-dimensional; found dimensions {zero_sensitivity.dims!r}."
        )
    state_dim = str(zero_sensitivity.dims[0])
    _require_unique_state_coord(zero_sensitivity, state_dim)
    _require_boolean(zero_sensitivity, name="zero_sensitivity")
    zero_sensitivity = zero_sensitivity.compute().rename("zero_sensitivity")

    explicit_active = _align_state_value(
        policy.active,
        sensitivity=zero_sensitivity,
        state_dim=state_dim,
        name="active",
        dtype=None,
    )
    _require_boolean(explicit_active, name="active")
    fixed_value = _align_state_value(
        policy.fixed_value,
        sensitivity=zero_sensitivity,
        state_dim=state_dim,
        name="fixed_value",
        dtype=float,
    )
    _require_finite(_materialize_1d(fixed_value, name="fixed_value"), name="fixed_value")

    active = explicit_active
    if policy.prune_zero:
        active = active & ~zero_sensitivity

    if policy.fixed_groups:
        if policy.group_coord not in zero_sensitivity.coords:
            raise ValueError(f"Cannot freeze groups: sensitivity has no {policy.group_coord!r} coordinate.")
        groups = zero_sensitivity.coords[policy.group_coord]
        if groups.dims != (state_dim,):
            raise ValueError(
                f"Group coordinate {policy.group_coord!r} must be indexed only by {state_dim!r}."
            )
        group_values = _materialize_1d(groups, name=policy.group_coord)
        available_groups = {str(value) for value in group_values.tolist()}
        missing_groups = sorted(set(policy.fixed_groups) - available_groups)
        if missing_groups:
            raise ValueError(
                f"Fixed group label(s) {missing_groups!r} are absent from {policy.group_coord!r}."
            )
        group_is_fixed = xr.DataArray(
            np.isin(group_values.astype(str), policy.fixed_groups),
            dims=(state_dim,),
            coords={state_dim: zero_sensitivity.coords[state_dim]},
        )
        active = active & ~group_is_fixed

    return ResolvedStateActivity(
        state_dim=state_dim,
        active=active.rename("active"),
        fixed_value=fixed_value,
        zero_sensitivity=zero_sensitivity,
    )


def active_prior_args(
    prior_args: dict[str, Any],
    activity: ResolvedStateActivity,
) -> dict[str, Any]:
    """Return prior arguments sliced to active states in canonical order.

    Scalar parameters are retained. One-dimensional NumPy state parameters are
    interpreted in canonical order and must describe the full state vector.
    Labelled xarray parameters are first aligned by the canonical state
    coordinate and then subset, allowing input label order to differ safely.
    Array-backed parameters are materialized at this model-building boundary.
    The support arrays of an ``Interpolated`` prior are preserved rather than
    interpreted as state-valued parameters.

    Args:
        prior_args: PyMC prior specification.
        activity: Resolved activity contract in canonical state order.

    Returns:
        A copy of ``prior_args`` containing scalar or active-state parameters.

    Raises:
        ValueError: If an array-valued distribution parameter is not scalar or
            full-state one-dimensional data.
    """
    result: dict[str, Any] = {}
    active_indices = activity.active_indices
    canonical_coord = activity.active.coords[activity.state_dim]
    pdf = str(prior_args.get("pdf", "")).lower()

    for name, value in prior_args.items():
        if name in {"pdf", "reparameterise"}:
            result[name] = value
            continue
        if pdf == "interpolated" and name in {"x_points", "pdf_points"}:
            support = value.compute().to_numpy() if isinstance(value, xr.DataArray) else np.asarray(value)
            _require_finite(support, name=f"Prior parameter {name!r}")
            result[name] = support
            continue
        if isinstance(value, xr.DataArray):
            if value.ndim == 0:
                scalar = value.compute().item()
                scalar_array = np.asarray(scalar)
                if np.issubdtype(scalar_array.dtype, np.number):
                    _require_finite(scalar_array, name=f"Prior parameter {name!r}")
                result[name] = scalar
                continue
            aligned = _align_state_value(
                value,
                sensitivity=xr.DataArray(
                    np.empty(activity.n_state),
                    dims=(activity.state_dim,),
                    coords={activity.state_dim: canonical_coord},
                ).expand_dims(nmeasure=[0]),
                state_dim=activity.state_dim,
                name=f"prior parameter {name!r}",
                dtype=None,
            )
            aligned_values = _materialize_1d(aligned, name=f"Prior parameter {name!r}")
            _require_finite(aligned_values, name=f"Prior parameter {name!r}")
            result[name] = aligned_values[active_indices]
            continue

        array = np.asarray(value)
        if array.ndim == 0:
            result[name] = value
        elif array.ndim == 1 and array.size == activity.n_state:
            _require_finite(array, name=f"Prior parameter {name!r}")
            result[name] = array[active_indices]
        else:
            raise ValueError(
                f"Prior parameter {name!r} must be scalar or a one-dimensional full-state "
                f"array of length {activity.n_state}; found shape {array.shape!r}."
            )

    return result
