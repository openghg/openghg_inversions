"""Label-aware active/fixed state-vector policies for linear inversion models.

This module separates state selection from PyMC graph construction. A
``StateActivity`` policy can combine exact-zero sensitivity pruning with an
explicit labelled activity mask and fixed ``basis_group`` labels. Resolution
always follows the state coordinate order on the supplied sensitivity matrix;
integer label ranges are never inferred.

Inactive states remain part of the public state vector and use labelled or
scalar fixed values. The default fixed value is one, which preserves the prior
forward-model contribution of a multiplicative flux-scaling state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr


StateValue = bool | float | int | np.ndarray | xr.DataArray


@dataclass(frozen=True)
class StateActivity:
    """Describe which states are sampled and how inactive states are fixed.

    Args:
        active: Scalar or one-dimensional labelled mask. A labelled mask is
            aligned to the sensitivity matrix by its state coordinate, so its
            input order need not match the matrix order. Set this to ``False``
            to freeze a whole component or sector.
        fixed_value: Scalar or one-dimensional state-aligned values used for
            inactive states. Multiplicative scaling states default to one.
        fixed_groups: ``basis_group`` labels to freeze. Group selection is by
            coordinate value, never by state-number ranges.
        group_coord: Name of the state coordinate containing group labels.
        prune_zero: Whether states whose full sensitivity column is exactly
            zero are made inactive. No tolerance is applied; every nonzero
            finite value remains active by default.

    Explicit ``active`` masks, fixed groups, and exact-zero pruning are
    combined with logical AND. This permits one policy to represent zero-H
    pruning, frozen states, frozen groups, or an entirely frozen sector.
    """

    active: StateValue = True
    fixed_value: StateValue = 1.0
    fixed_groups: tuple[str, ...] = ()
    group_coord: str = "basis_group"
    prune_zero: bool = True


@dataclass(frozen=True)
class ResolvedStateActivity:
    """A state-activity policy aligned to one sensitivity matrix.

    Attributes:
        state_dim: Canonical state dimension name from the matrix.
        active: Boolean mask in canonical state order.
        fixed_value: Fixed values in canonical state order.
        zero_sensitivity: Mask identifying exactly-zero sensitivity columns.
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
        return int(self.active.sum().item())

    @property
    def active_indices(self) -> np.ndarray:
        """Return positional indices of active states in canonical order."""
        return np.flatnonzero(np.asarray(self.active.values, dtype=bool))

    @property
    def fixed_indices(self) -> np.ndarray:
        """Return positional indices of inactive states in canonical order."""
        return np.flatnonzero(~np.asarray(self.active.values, dtype=bool))


def _state_dim(sensitivity: xr.DataArray, output_dim: str) -> str:
    """Return the sole non-output dimension of a linear sensitivity matrix."""
    state_dims = [str(dim) for dim in sensitivity.dims if dim != output_dim]
    if len(state_dims) != 1:
        raise ValueError(
            "State activity requires a linear sensitivity matrix with exactly "
            f"one non-output dimension; found {state_dims!r}."
        )
    return state_dims[0]


def _align_state_value(
    value: StateValue,
    *,
    sensitivity: xr.DataArray,
    state_dim: str,
    name: str,
    dtype: Any,
) -> xr.DataArray:
    """Broadcast or label-align a state value to the sensitivity matrix."""
    state_coord = sensitivity.coords[state_dim]
    state_size = sensitivity.sizes[state_dim]

    if isinstance(value, xr.DataArray):
        if value.ndim == 0:
            values = np.full(state_size, value.item(), dtype=dtype)
        else:
            if value.dims != (state_dim,):
                raise ValueError(
                    f"{name} must be scalar or have only the {state_dim!r} dimension; "
                    f"found dimensions {value.dims!r}."
                )
            if state_dim not in value.coords:
                raise ValueError(f"{name} must have labelled {state_dim!r} coordinates.")
            try:
                aligned = value.sel({state_dim: state_coord})
            except KeyError as exc:
                raise ValueError(
                    f"{name} does not define every label in the canonical {state_dim!r} coordinate."
                ) from exc
            values = np.asarray(aligned.values, dtype=dtype)
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
    sensitivity: xr.DataArray,
    policy: StateActivity | None = None,
    *,
    output_dim: str = "nmeasure",
) -> ResolvedStateActivity:
    """Resolve an active/fixed state policy against a sensitivity matrix.

    Args:
        sensitivity: Labelled linear sensitivity matrix.
        policy: Optional activity policy. When omitted, exact-zero columns are
            pruned and all other states are active with inactive value one.
        output_dim: Name of the observation/output dimension.

    Returns:
        A policy aligned to the matrix's canonical state coordinate.

    Raises:
        ValueError: If the matrix is not two-dimensional in state/output terms,
            supplied arrays cannot be aligned, or requested group metadata is
            absent or not state-aligned.
    """
    policy = policy or StateActivity()
    state_dim = _state_dim(sensitivity, output_dim)
    matrix = sensitivity.transpose(output_dim, state_dim)

    explicit_active = _align_state_value(
        policy.active,
        sensitivity=matrix,
        state_dim=state_dim,
        name="active",
        dtype=bool,
    )
    fixed_value = _align_state_value(
        policy.fixed_value,
        sensitivity=matrix,
        state_dim=state_dim,
        name="fixed_value",
        dtype=float,
    )
    zero_sensitivity = ~(matrix != 0).any(dim=output_dim)
    zero_sensitivity = zero_sensitivity.rename("zero_sensitivity")

    active = explicit_active
    if policy.prune_zero:
        active = active & ~zero_sensitivity

    if policy.fixed_groups:
        if policy.group_coord not in matrix.coords:
            raise ValueError(f"Cannot freeze groups: sensitivity has no {policy.group_coord!r} coordinate.")
        groups = matrix.coords[policy.group_coord]
        if groups.dims != (state_dim,):
            raise ValueError(
                f"Group coordinate {policy.group_coord!r} must be indexed only by {state_dim!r}."
            )
        available_groups = {str(value) for value in groups.values.tolist()}
        missing_groups = sorted(set(policy.fixed_groups) - available_groups)
        if missing_groups:
            raise ValueError(
                f"Fixed group label(s) {missing_groups!r} are absent from {policy.group_coord!r}."
            )
        active = active & ~groups.astype(str).isin(policy.fixed_groups)

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

    Scalar parameters are retained. One-dimensional NumPy parameters must
    describe the full state vector. Labelled xarray parameters are first
    aligned by the canonical state coordinate and then subset, allowing input
    label order to differ safely.

    Args:
        prior_args: PyMC prior specification.
        activity: Resolved activity contract for the sensitivity matrix.

    Returns:
        A copy of ``prior_args`` containing scalar or active-state parameters.

    Raises:
        ValueError: If an array-valued distribution parameter is not scalar or
            full-state one-dimensional data.
    """
    result: dict[str, Any] = {}
    active_indices = activity.active_indices
    canonical_coord = activity.active.coords[activity.state_dim]

    for name, value in prior_args.items():
        if name in {"pdf", "reparameterise"}:
            result[name] = value
            continue
        if isinstance(value, xr.DataArray):
            aligned = _align_state_value(
                value,
                sensitivity=xr.DataArray(
                    np.empty(activity.n_state),
                    dims=(activity.state_dim,),
                    coords={activity.state_dim: canonical_coord},
                ).expand_dims(nmeasure=[0]),
                state_dim=activity.state_dim,
                name=f"prior parameter {name!r}",
                dtype=float,
            )
            result[name] = aligned.values[active_indices]
            continue

        array = np.asarray(value)
        if array.ndim == 0:
            result[name] = value
        elif array.ndim == 1 and array.size == activity.n_state:
            result[name] = array[active_indices]
        else:
            raise ValueError(
                f"Prior parameter {name!r} must be scalar or a one-dimensional full-state "
                f"array of length {activity.n_state}; found shape {array.shape!r}."
            )

    return result
