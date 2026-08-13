"""Labelled in-memory native-covariance products for retained scaling states.

This module is an eager numerical kernel. Callers supply a canonical, eager
native sensitivity ``H`` and basis prolongation ``U``; basis operators own any
source expansion and the pipeline owns materialization. Custom restrictions
may remain sparse or Dask-backed until the named projection boundary, where
they are materialized once and reused by all retained-state RHS blocks.

For native covariance ``B`` and retained restriction ``Pi``, the kernel returns
``C_alpha = Pi B Pi.T``, ``H U_*``, ``H B Pi.T``, and ``H B H.T`` (or its
diagonal). The default strategy preserves bucket prolongation by deriving
``Pi_U = (U.T B^-1 U)^-1 U.T B^-1``. Durable artifact identity and persistence
are intentionally deferred to the artifact-I/O layer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
import xarray as xr
from xarray.namedarray.pycompat import is_chunked_array

from openghg_inversions._labelled_matrices import (
    matrix_column_dim,
    renamed_column_coordinates,
    to_column_axis,
    with_square_matrix_diagnostics,
)
from openghg_inversions.array_ops import to_dense
from openghg_inversions.native_covariance import InvertibleNativeCovarianceAction

MAX_DENSE_EIGEN_DIAGNOSTIC_SIZE = 512


@dataclass(frozen=True, slots=True)
class RetainedProjection:
    """A labelled restriction/prolongation pair selected by one strategy.

    Attributes:
        restriction: ``Pi`` with dimensions ``(state_dim, *native_dims)``.
        prolongation: Covariance-natural ``U_*`` with dimensions
            ``(*native_dims, state_dim)``.
        strategy: Stable identifier for the scientific projection choice.
        state_covariance: Optional cached ``C_alpha``. The kernel independently
            recomputes and validates caches supplied by external strategies.
        covariance_restriction_transpose: Optional cached ``B Pi.T``. External
            cached values are independently recomputed and validated.
    """

    restriction: xr.DataArray
    prolongation: xr.DataArray
    strategy: str
    state_covariance: xr.DataArray | None = None
    covariance_restriction_transpose: xr.DataArray | None = None


@dataclass(frozen=True, slots=True)
class _PreserveBucketProjection(RetainedProjection):
    """Private built-in projection with independently derived trusted products."""


class RetainedProjectionStrategy(Protocol):
    """Extension seam for choosing retained coefficients and their lift."""

    def projection(
        self,
        covariance: InvertibleNativeCovarianceAction,
        basis_prolongation: xr.DataArray,
        *,
        native_dims: tuple[str, ...],
        state_dim: str,
    ) -> RetainedProjection:
        """Return a compatible labelled restriction and prolongation."""
        ...


@dataclass(frozen=True, slots=True)
class PreserveBucketProlongation:
    """Derive ``Pi_U`` so covariance-weighted prolongation equals ``U_bucket``."""

    name: str = "preserve_bucket_prolongation"

    def __post_init__(self) -> None:
        """Require a stable non-empty strategy identifier."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Projection strategy name must be a non-empty string")

    def projection(
        self,
        covariance: InvertibleNativeCovarianceAction,
        basis_prolongation: xr.DataArray,
        *,
        native_dims: tuple[str, ...],
        state_dim: str,
    ) -> RetainedProjection:
        """Construct the prior-precision-compatible restriction.

        Args:
            covariance: Invertible labelled action for native covariance ``B``.
            basis_prolongation: Eager canonical bucket prolongation ``U``.
            native_dims: Ordered native dimensions.
            state_dim: Retained-state dimension.

        Returns:
            The coherent ``(Pi_U, U)`` pair and algebraically known products.

        Raises:
            ValueError: If ``U`` is invalid or lacks full column rank.
        """
        prolongation = _validated_prolongation(
            basis_prolongation, native_dims=native_dims, state_dim=state_dim
        )
        state_column_dim = matrix_column_dim(
            state_dim,
            {str(name) for name in (*prolongation.dims, *prolongation.coords)},
        )
        precision_prolongation = to_column_axis(
            covariance.solve(prolongation),
            row_dim=state_dim,
            column_dim=state_column_dim,
            leading_dims=native_dims,
        )
        gram = xr.dot(prolongation, precision_prolongation, dim=list(native_dims)).transpose(
            state_dim, state_column_dim
        )
        gram_values = np.asarray(gram.values, dtype=np.float64)
        _require_symmetric(gram_values, "U.T B^-1 U")
        try:
            factor = cho_factor(gram_values, lower=True, check_finite=True)
            covariance_values = cho_solve(factor, np.eye(gram_values.shape[0]), check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "The bucket prolongation contains redundant retained states; "
                "U.T B^-1 U must be positive definite"
            ) from exc
        state_covariance = xr.DataArray(
            covariance_values,
            dims=(state_dim, state_column_dim),
            attrs={"units": "1"},
        )
        state_index = prolongation.get_index(state_dim)
        if isinstance(state_index, pd.MultiIndex):
            state_covariance = state_covariance.assign_coords(
                xr.Coordinates.from_pandas_multiindex(state_index, state_dim)
            )
        else:
            state_covariance = state_covariance.assign_coords({state_dim: prolongation.coords[state_dim]})
        state_covariance = state_covariance.assign_coords(
            renamed_column_coordinates(
                prolongation,
                row_dim=state_dim,
                column_dim=state_column_dim,
            )
        )
        state_covariance = with_square_matrix_diagnostics(
            state_covariance,
            mathematical_name="C_alpha",
            require_positive_definite=True,
        )
        restriction = xr.dot(state_covariance, precision_prolongation, dim=state_column_dim).transpose(
            state_dim, *native_dims
        )
        restriction = restriction.rename("restriction").assign_attrs(
            mathematical_name="Pi_U",
            definition="(U.T B^-1 U)^-1 U.T B^-1",
            strategy=self.name,
            units="1",
        )
        prolongation = prolongation.rename("prolongation").assign_attrs(
            mathematical_name="U_bucket", strategy=self.name, units="1"
        )
        b_pi_t = xr.dot(prolongation, state_covariance, dim=state_dim).transpose(
            *native_dims, state_column_dim
        )
        return _PreserveBucketProjection(
            restriction=restriction,
            prolongation=prolongation,
            strategy=self.name,
            state_covariance=state_covariance,
            covariance_restriction_transpose=b_pi_t,
        )


@dataclass(frozen=True, slots=True)
class NativeCovarianceProducts:
    """Labelled in-memory product blocks induced by one coherent projection."""

    restriction: xr.DataArray
    prolongation: xr.DataArray
    state_covariance: xr.DataArray
    effective_observation_operator: xr.DataArray
    observation_state_cross_covariance: xr.DataArray
    native_observation_covariance: xr.DataArray
    strategy: str
    observation_covariance_view: Literal["dense", "diagonal"]


def project_native_covariance(
    *,
    covariance: InvertibleNativeCovarianceAction,
    basis_prolongation: xr.DataArray,
    state_dim: str,
    native_sensitivity: xr.DataArray,
    observation_dim: str,
    observation_covariance: Literal["dense", "diagonal"] = "dense",
    observation_batch_size: int = 64,
    strategy: RetainedProjectionStrategy | None = None,
) -> NativeCovarianceProducts:
    """Compute coherent labelled native-covariance products eagerly.

    Args:
        covariance: Labelled native covariance action with a compatible solve.
        basis_prolongation: Canonical eager ``U`` from the basis-side native
            expansion boundary.
        state_dim: Retained-state dimension shared by ``U`` and ``Pi``.
        native_sensitivity: Canonical eager native sensitivity ``H``.
        observation_dim: Observation dimension in ``H``.
        observation_covariance: Return dense ``H B H.T`` or its diagonal.
        observation_batch_size: Positive integral number of covariance
            right-hand sides per eager block.
        strategy: Projection choice; defaults to bucket preservation.

    Returns:
        Frozen in-memory labelled product value object. Scaling covariance,
        restriction, and prolongation have units ``"1"``. Products linear in
        ``H`` inherit its units; observation covariance uses squared ``H``
        units when supplied.

    Raises:
        TypeError: If ``observation_batch_size`` is not an integral non-Boolean.
        ValueError: If arrays are lazy, labels or dimensions disagree, values
            are invalid, or an option is unsupported.
    """
    native_dims = tuple(covariance.native_dims)
    if len({*native_dims, state_dim, observation_dim}) != len(native_dims) + 2:
        raise ValueError("Native, retained-state, and observation dimension names must be distinct")
    if isinstance(observation_batch_size, bool) or not isinstance(observation_batch_size, Integral):
        raise TypeError("observation_batch_size must be an integral non-Boolean value")
    batch_size = int(observation_batch_size)
    if batch_size <= 0:
        raise ValueError("observation_batch_size must be positive")
    if observation_covariance not in {"dense", "diagonal"}:
        raise ValueError("observation_covariance must be 'dense' or 'diagonal'")

    sensitivity = _validated_sensitivity(
        native_sensitivity,
        native_dims=native_dims,
        observation_dim=observation_dim,
    )
    prolongation = _validated_prolongation(basis_prolongation, native_dims=native_dims, state_dim=state_dim)
    _validate_exact_native_coordinates(
        prolongation, sensitivity, native_dims=native_dims, role="prolongation"
    )
    projection_strategy = strategy if strategy is not None else PreserveBucketProlongation()
    projection = projection_strategy.projection(
        covariance,
        prolongation,
        native_dims=native_dims,
        state_dim=state_dim,
    )
    projection = _validated_projection(
        projection,
        native_reference=sensitivity,
        native_dims=native_dims,
        state_dim=state_dim,
    )
    occupied_names = {
        str(name)
        for array in (sensitivity, projection.restriction, projection.prolongation)
        for name in (*array.dims, *array.coords)
    }
    state_column_dim = matrix_column_dim(state_dim, occupied_names)
    trusted_projection = (
        _reaxis_builtin_projection(
            projection,
            native_dims=native_dims,
            state_dim=state_dim,
            state_column_dim=state_column_dim,
        )
        if type(projection_strategy) is PreserveBucketProlongation
        and isinstance(projection, _PreserveBucketProjection)
        else None
    )
    if trusted_projection is None:
        b_pi_t = _apply_restriction_blocks(
            covariance,
            projection.restriction,
            native_dims=native_dims,
            state_dim=state_dim,
            column_dim=state_column_dim,
            rhs_block_size=batch_size,
        )
    else:
        assert trusted_projection.covariance_restriction_transpose is not None
        b_pi_t = trusted_projection.covariance_restriction_transpose

    if trusted_projection is None:
        state_covariance = _restriction_product_blocks(
            projection.restriction,
            b_pi_t,
            native_dims=native_dims,
            state_dim=state_dim,
            column_dim=state_column_dim,
            rhs_block_size=batch_size,
        )
        state_covariance = with_square_matrix_diagnostics(
            state_covariance.rename("state_covariance").assign_attrs(units="1"),
            mathematical_name="C_alpha",
            require_positive_definite=True,
        )
        _validate_external_cached_products(
            projection,
            computed_b_pi_t=b_pi_t,
            computed_state_covariance=state_covariance,
        )
    else:
        assert trusted_projection.state_covariance is not None
        state_covariance = trusted_projection.state_covariance.rename("state_covariance")

    if trusted_projection is None:
        _validate_projection_invariant(
            projection,
            state_covariance,
            b_pi_t,
            native_dims=native_dims,
            state_dim=state_dim,
            state_column_dim=state_column_dim,
        )
    h_units = sensitivity.attrs.get("units")
    linear_units = {"units": h_units} if h_units is not None else {}
    effective_operator = xr.dot(sensitivity, projection.prolongation, dim=list(native_dims)).transpose(
        observation_dim, state_dim
    )
    effective_operator = effective_operator.rename("effective_observation_operator").assign_attrs(
        mathematical_name="H_alpha", definition="H U_*", **linear_units
    )
    cross_covariance = xr.dot(sensitivity, b_pi_t, dim=list(native_dims)).transpose(
        observation_dim, state_column_dim
    )
    cross_covariance = cross_covariance.rename("observation_state_cross_covariance").assign_attrs(
        mathematical_name="H B Pi.T", **linear_units
    )
    native_observation_covariance = _observation_covariance(
        covariance,
        sensitivity,
        native_dims=native_dims,
        observation_dim=observation_dim,
        output=observation_covariance,
        batch_size=batch_size,
    )
    return NativeCovarianceProducts(
        restriction=projection.restriction,
        prolongation=projection.prolongation,
        state_covariance=state_covariance,
        effective_observation_operator=effective_operator,
        observation_state_cross_covariance=cross_covariance,
        native_observation_covariance=native_observation_covariance,
        strategy=projection.strategy,
        observation_covariance_view=observation_covariance,
    )


def _apply_restriction_blocks(
    covariance: InvertibleNativeCovarianceAction,
    restriction: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    state_dim: str,
    column_dim: str,
    rhs_block_size: int,
) -> xr.DataArray:
    """Apply covariance to explicit dense retained-state RHS blocks."""
    blocks: list[xr.DataArray] = []
    for start in range(0, restriction.sizes[state_dim], rhs_block_size):
        sliced = restriction.isel({state_dim: slice(start, start + rhs_block_size)})
        rhs = to_column_axis(
            sliced,
            row_dim=state_dim,
            column_dim=column_dim,
            leading_dims=native_dims,
        )
        rhs = _materialize_dense_preserving_coordinates(rhs)
        blocks.append(covariance.apply(rhs))
    return xr.concat(blocks, dim=column_dim).transpose(*native_dims, column_dim)


def _restriction_product_blocks(
    restriction: xr.DataArray,
    b_pi_t: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    state_dim: str,
    column_dim: str,
    rhs_block_size: int,
) -> xr.DataArray:
    """Form ``Pi B Pi.T`` while materializing explicit restriction row blocks."""
    blocks: list[xr.DataArray] = []
    for start in range(0, restriction.sizes[state_dim], rhs_block_size):
        rows = _materialize_dense_preserving_coordinates(
            restriction.isel({state_dim: slice(start, start + rhs_block_size)})
        )
        blocks.append(xr.dot(rows, b_pi_t, dim=list(native_dims)))
    return xr.concat(blocks, dim=state_dim).transpose(state_dim, column_dim)


def _observation_covariance(
    covariance: InvertibleNativeCovarianceAction,
    sensitivity: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    observation_dim: str,
    output: Literal["dense", "diagonal"],
    batch_size: int,
) -> xr.DataArray:
    """Compute ``H B H.T`` in eager observation RHS blocks."""
    column_dim = matrix_column_dim(
        observation_dim,
        {str(name) for name in (*sensitivity.dims, *sensitivity.coords)},
    )
    count = sensitivity.sizes[observation_dim]
    values = np.empty(
        (count, count) if output == "dense" else count,
        dtype=np.result_type(sensitivity.dtype, np.float64),
    )
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        rhs = to_column_axis(
            sensitivity.isel({observation_dim: slice(start, stop)}),
            row_dim=observation_dim,
            column_dim=column_dim,
            leading_dims=native_dims,
        )
        b_rhs = covariance.apply(rhs)
        if output == "dense":
            block = xr.dot(sensitivity, b_rhs, dim=list(native_dims)).transpose(observation_dim, column_dim)
            values[:, start:stop] = np.asarray(block.values)
        else:
            values[start:stop] = np.asarray((rhs * b_rhs).sum(dim=list(native_dims)).values)
    units = sensitivity.attrs.get("units")
    unit_attrs = {"units": f"({units})^2"} if units is not None else {}
    row_coords = {
        str(name): coordinate
        for name, coordinate in sensitivity.coords.items()
        if set(coordinate.dims).issubset({observation_dim})
    }
    if output == "dense":
        result = xr.DataArray(
            values,
            dims=(observation_dim, column_dim),
            coords={
                **row_coords,
                **renamed_column_coordinates(sensitivity, row_dim=observation_dim, column_dim=column_dim),
            },
            attrs=unit_attrs,
            name="native_observation_covariance",
        )
        return with_square_matrix_diagnostics(
            result,
            mathematical_name="H B H.T",
            maximum_eigen_diagnostic_size=MAX_DENSE_EIGEN_DIAGNOSTIC_SIZE,
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("diag(H B H.T) must contain only finite real values")
    diagonal_scale = max(1.0, float(np.max(np.abs(values))) if values.size else 1.0)
    diagonal_tolerance = 1e-10 * diagonal_scale
    if np.any(values < -diagonal_tolerance):
        raise ValueError("diag(H B H.T) contains materially negative covariance values")
    result = xr.DataArray(
        values,
        dims=observation_dim,
        coords=row_coords,
        attrs={
            **unit_attrs,
            "mathematical_name": "diag(H B H.T)",
            "minimum_diagonal": float(np.min(values)),
            "diagonal_nonnegative": bool(np.all(values >= -diagonal_tolerance)),
            "diagnostic_tolerance": diagonal_tolerance,
        },
        name="native_observation_covariance",
    )
    return result


def _validated_projection(
    projection: RetainedProjection,
    *,
    native_reference: xr.DataArray,
    native_dims: tuple[str, ...],
    state_dim: str,
) -> RetainedProjection:
    """Validate labels and materialize a custom restriction exactly once."""
    if not isinstance(projection.strategy, str) or not projection.strategy:
        raise ValueError("Projection strategy must return a non-empty strategy identifier")
    restriction = projection.restriction.transpose(state_dim, *native_dims).assign_attrs(
        {**projection.restriction.attrs, "units": "1"}
    )
    expected = {state_dim, *native_dims}
    if set(restriction.dims) != expected or len(restriction.dims) != len(expected):
        raise ValueError("Projection strategy restriction has invalid labelled dimensions")
    prolongation = _validated_prolongation(
        projection.prolongation, native_dims=native_dims, state_dim=state_dim
    ).assign_attrs({**projection.prolongation.attrs, "units": "1"})
    for role, array in (("restriction", restriction), ("prolongation", prolongation)):
        _validate_exact_native_coordinates(array, native_reference, native_dims=native_dims, role=role)
    if state_dim not in restriction.coords or state_dim not in prolongation.coords:
        raise ValueError("Projection strategy restriction and prolongation require state labels")
    if not restriction.get_index(state_dim).equals(prolongation.get_index(state_dim)):
        raise ValueError("Projection strategy restriction/prolongation state labels differ")
    _require_compatible_axis_coordinates(
        restriction,
        prolongation,
        dim=state_dim,
        role="retained-state",
    )
    for role, array in (("restriction", restriction), ("prolongation", prolongation)):
        _require_reference_native_coordinates(
            array,
            native_reference,
            native_dims=native_dims,
            role=role,
        )
    if not isinstance(projection, _PreserveBucketProjection):
        restriction = _materialize_dense_preserving_coordinates(restriction)
        restriction_values = np.asarray(restriction.values)
        if np.iscomplexobj(restriction_values) or not np.all(np.isfinite(restriction_values)):
            raise ValueError("Projection strategy restriction must contain only finite real values")
    return replace(projection, restriction=restriction, prolongation=prolongation)


def _reaxis_builtin_projection(
    projection: _PreserveBucketProjection,
    *,
    native_dims: tuple[str, ...],
    state_dim: str,
    state_column_dim: str,
) -> _PreserveBucketProjection:
    """Rebuild trusted built-in product columns on the globally safe axis."""
    assert projection.state_covariance is not None
    assert projection.covariance_restriction_transpose is not None
    old_column_dim = next(dim for dim in projection.state_covariance.dims if dim != state_dim)
    if old_column_dim == state_column_dim:
        return projection
    state_covariance = xr.DataArray(
        projection.state_covariance.data,
        dims=(state_dim, state_column_dim),
        coords={
            str(name): coordinate
            for name, coordinate in projection.state_covariance.coords.items()
            if set(coordinate.dims).issubset({state_dim})
        },
        attrs=projection.state_covariance.attrs,
        name=projection.state_covariance.name,
    ).assign_coords(
        renamed_column_coordinates(
            projection.prolongation,
            row_dim=state_dim,
            column_dim=state_column_dim,
        )
    )
    b_pi_t = xr.DataArray(
        projection.covariance_restriction_transpose.data,
        dims=(*native_dims, state_column_dim),
        coords={
            str(name): coordinate
            for name, coordinate in projection.covariance_restriction_transpose.coords.items()
            if set(coordinate.dims).issubset(native_dims)
        },
        attrs=projection.covariance_restriction_transpose.attrs,
        name=projection.covariance_restriction_transpose.name,
    ).assign_coords(
        renamed_column_coordinates(
            projection.prolongation,
            row_dim=state_dim,
            column_dim=state_column_dim,
        )
    )
    return replace(
        projection,
        state_covariance=state_covariance,
        covariance_restriction_transpose=b_pi_t,
    )


def _validate_external_cached_products(
    projection: RetainedProjection,
    *,
    computed_b_pi_t: xr.DataArray,
    computed_state_covariance: xr.DataArray,
) -> None:
    """Independently validate optional products supplied by external strategies."""
    candidates = (
        (
            "covariance_restriction_transpose",
            projection.covariance_restriction_transpose,
            computed_b_pi_t,
        ),
        ("state_covariance", projection.state_covariance, computed_state_covariance),
    )
    for name, cached, computed in candidates:
        if cached is None:
            continue
        cached_values = np.asarray(to_dense(cached).compute().values)
        computed_values = np.asarray(computed.values)
        if (
            cached_values.shape != computed_values.shape
            or np.iscomplexobj(cached_values)
            or not np.all(np.isfinite(cached_values))
            or not np.allclose(cached_values, computed_values, rtol=1e-10, atol=1e-12)
        ):
            raise ValueError(
                f"Projection strategy supplied inconsistent cached {name}; "
                "it does not match the independently computed covariance product"
            )


def _require_compatible_axis_coordinates(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    dim: str,
    role: str,
) -> None:
    """Require identical semantic coordinates attached to one shared axis."""
    left_coords = {
        str(name): coordinate
        for name, coordinate in left.coords.items()
        if dim in coordinate.dims and str(name) != dim
    }
    right_coords = {
        str(name): coordinate
        for name, coordinate in right.coords.items()
        if dim in coordinate.dims and str(name) != dim
    }
    if set(left_coords) != set(right_coords):
        raise ValueError(f"Projection {role} auxiliary coordinate names differ between Pi and U")
    for name, coordinate in left_coords.items():
        other = right_coords[name]
        if coordinate.dims != other.dims or not coordinate.equals(other):
            raise ValueError(f"Projection {role} auxiliary coordinate {name!r} differs between Pi and U")


def _require_reference_native_coordinates(
    array: xr.DataArray,
    reference: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    role: str,
) -> None:
    """Require all reference native-only and scalar coordinates on a projection array."""
    native_set = set(native_dims)
    expected = {
        str(name): coordinate
        for name, coordinate in reference.coords.items()
        if set(coordinate.dims).issubset(native_set)
    }
    for name, coordinate in expected.items():
        if name not in array.coords:
            raise ValueError(f"Projection {role} is missing compatible native coordinate {name!r}")
        actual = array.coords[name]
        if actual.dims != coordinate.dims or not actual.equals(coordinate):
            raise ValueError(f"Projection {role} native coordinate {name!r} is incompatible")


def _validate_exact_native_coordinates(
    array: xr.DataArray,
    reference: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    role: str,
) -> None:
    """Require exact ordered native indexes before labelled contractions."""
    for dim in native_dims:
        if dim not in array.coords:
            raise ValueError(f"Projection {role} is missing native coordinate {dim!r}")
        if not array.get_index(dim).equals(reference.get_index(dim)):
            raise ValueError(
                f"Projection {role} native coordinate {dim!r} must exactly match the covariance grid"
            )


def _validate_projection_invariant(
    projection: RetainedProjection,
    state_covariance: xr.DataArray,
    b_pi_t: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    state_dim: str,
    state_column_dim: str,
) -> None:
    """Require ``B Pi.T = U_* C_alpha`` for a custom strategy."""
    right = xr.dot(projection.prolongation, state_covariance, dim=state_dim).transpose(
        *native_dims, state_column_dim
    )
    left_values = np.asarray(b_pi_t.values, dtype=np.float64)
    right_values = np.asarray(right.values, dtype=np.float64)
    scale = max(
        float(np.max(np.abs(left_values))) if left_values.size else 0.0,
        float(np.max(np.abs(right_values))) if right_values.size else 0.0,
        np.finfo(np.float64).tiny,
    )
    error = float(np.max(np.abs(left_values - right_values))) if left_values.size else 0.0
    if error > 1e-9 * scale + 10.0 * np.finfo(np.float64).tiny:
        raise ValueError("Projection strategy is incoherent: expected B Pi.T = U_* C_alpha")


def _validated_prolongation(
    prolongation: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    state_dim: str,
) -> xr.DataArray:
    """Validate an eager labelled native-by-retained prolongation."""
    expected = {*native_dims, state_dim}
    if set(prolongation.dims) != expected or len(prolongation.dims) != len(expected):
        raise ValueError(
            f"prolongation must have exactly native and retained-state dimensions; got {prolongation.dims!r}"
        )
    if is_chunked_array(prolongation.data):
        raise ValueError(
            "basis_prolongation is lazy; materialize canonical U explicitly at the upstream "
            "pipeline boundary before calling project_native_covariance"
        )
    result = _materialize_dense_preserving_coordinates(prolongation).transpose(*native_dims, state_dim)
    values = np.asarray(result.values)
    if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
        raise ValueError("prolongation must contain only finite real values")
    if result.sizes[state_dim] == 0 or np.any(np.all(values == 0.0, axis=tuple(range(len(native_dims))))):
        raise ValueError("prolongation contains an empty retained state")
    return result


def _validated_sensitivity(
    sensitivity: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    observation_dim: str,
) -> xr.DataArray:
    """Validate canonical eager sensitivity without a disposable B apply."""
    expected = {*native_dims, observation_dim}
    if set(sensitivity.dims) != expected or len(sensitivity.dims) != len(expected):
        raise ValueError(
            "native_sensitivity must have exactly observation and native dimensions; "
            f"got {sensitivity.dims!r}"
        )
    if observation_dim not in sensitivity.coords:
        raise ValueError(f"native_sensitivity is missing observation coordinate {observation_dim!r}")
    if is_chunked_array(sensitivity.data):
        raise ValueError(
            "native_sensitivity is lazy; materialize canonical H explicitly at the upstream "
            "pipeline boundary before calling project_native_covariance"
        )
    result = _materialize_dense_preserving_coordinates(sensitivity).transpose(observation_dim, *native_dims)
    if result.sizes[observation_dim] == 0:
        raise ValueError("native_sensitivity must contain at least one observation")
    values = np.asarray(result.values)
    if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
        raise ValueError("native_sensitivity must contain only finite real values")
    return result


def _require_symmetric(values: np.ndarray, name: str, tolerance: float = 1e-10) -> None:
    """Require a square numeric array to be symmetric within tolerance."""
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be square")
    asymmetry = float(np.max(np.abs(values - values.T))) if values.size else 0.0
    scale = max(1.0, float(np.max(np.abs(values))) if values.size else 1.0)
    if asymmetry > tolerance * scale:
        raise ValueError(f"{name} is not symmetric within tolerance {tolerance:g}")


def _materialize_dense_preserving_coordinates(array: xr.DataArray) -> xr.DataArray:
    """Materialize dense payloads without reconstructing typed xarray indexes."""
    materialized = array.compute()
    data = materialized.data
    if hasattr(data, "todense"):
        data = data.todense()
    return materialized.copy(data=np.asarray(data))
