"""Labelled in-memory native-covariance products for retained scaling states.

This module is an eager numerical kernel. Callers supply a canonical native
sensitivity ``H`` and basis prolongation ``U``; basis operators own any source
expansion. Their payloads may remain sparse or Dask-backed until the named
projection boundary, where related arrays are materialized together. Custom
restrictions are likewise materialized once and reused by all retained-state
RHS blocks.

For native covariance ``B`` and retained restriction ``Pi``, the kernel returns
``C_alpha = Pi B Pi.T``, ``H U_*``, ``H B Pi.T``, and ``H B H.T`` (or its
diagonal). Strategies return authoritative ``Pi``; the kernel derives
``B Pi.T``, ``C_alpha``, and ``U_* = B Pi.T C_alpha^-1``. The default strategy
preserves bucket prolongation by deriving
``Pi_U = (U.T B^-1 U)^-1 U.T B^-1``. The kernel trusts the covariance action's
real self-adjoint positive-definite semantics and compatible inverse rather
than globally certifying a matrix-free operator. Durable artifact identity and
persistence are intentionally deferred to the artifact-I/O layer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import sys
from typing import Literal, NoReturn, Protocol

if sys.version_info >= (3, 11):
    from typing import assert_never
else:

    def assert_never(value: NoReturn) -> NoReturn:
        """Backport typing.assert_never until Python 3.10 support is removed."""
        raise AssertionError(f"Expected an unreachable value, got {value!r}")

from dask.base import compute
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.lapack import dpocon
import xarray as xr

from openghg_inversions._labelled_matrices import (
    matrix_column_dim,
    renamed_column_coordinates,
    to_column_axis,
)
from openghg_inversions.array_ops import to_dense
from openghg_inversions.native_covariance import InvertibleNativeCovarianceAction


MIN_RETAINED_RECIPROCAL_CONDITION = float(np.sqrt(np.finfo(np.float64).eps))


@dataclass(frozen=True, slots=True, eq=False)
class RetainedProjection:
    """A labelled retained restriction selected by one strategy.

    Attributes:
        restriction: Dimensionless ``Pi`` with dimensions
            ``(state_dim, *native_dims)``. The frozen dataclass does not freeze
            this mutable DataArray.
        strategy: Stable identifier for the scientific projection choice.
    """

    restriction: xr.DataArray
    strategy: str


class RetainedProjectionStrategy(Protocol):
    """Extension seam for choosing authoritative retained coefficients."""

    def projection(
        self,
        covariance: InvertibleNativeCovarianceAction,
        basis_prolongation: xr.DataArray,
        *,
        native_dims: tuple[str, ...],
        state_dim: str,
    ) -> RetainedProjection:
        """Return the authoritative labelled retained restriction ``Pi``.

        Args:
            covariance: Labelled native covariance action.
            basis_prolongation: Dimensionless basis ``U`` with dimensions
                ``(*native_dims, state_dim)``.
            native_dims: Ordered native covariance dimensions.
            state_dim: Retained-state dimension.

        Returns:
            A dimensionless restriction with dimensions
            ``(state_dim, *native_dims)``. The kernel derives ``B Pi.T``,
            ``C_alpha``, and ``U_*`` from it.
        """
        ...


@dataclass(frozen=True, slots=True)
class PreserveBucketProlongation:
    """Derive ``Pi_U`` so covariance-weighted prolongation equals ``U_bucket``."""

    name: str = "preserve_bucket_prolongation"

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
            The authoritative bucket-compatible restriction ``Pi_U``.

        Raises:
            ValueError: If ``U`` is invalid or lacks full column rank.
        """
        prolongation = basis_prolongation
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
        gram_values = (gram_values + gram_values.T) * 0.5
        try:
            factor = cho_factor(gram_values, lower=True, check_finite=True)
            covariance_values = cho_solve(factor, np.eye(gram_values.shape[0]), check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "The bucket prolongation contains redundant retained states; "
                "U.T B^-1 U must be positive definite"
            ) from exc
        state_coordinates = {
            str(name): coordinate
            for name, coordinate in prolongation.coords.items()
            if set(coordinate.dims).issubset({state_dim})
        }
        state_covariance = xr.DataArray(
            covariance_values,
            dims=(state_dim, state_column_dim),
            coords={
                **state_coordinates,
                **renamed_column_coordinates(
                    prolongation,
                    row_dim=state_dim,
                    column_dim=state_column_dim,
                ),
            },
            attrs={"units": "1"},
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
        return RetainedProjection(
            restriction=restriction,
            strategy=self.name,
        )


@dataclass(frozen=True, slots=True, eq=False)
class NativeCovarianceProducts:
    """Labelled in-memory product blocks induced by one retained restriction.

    Attributes:
        restriction: Dimensionless ``Pi`` with dimensions
            ``(state_dim, *native_dims)``.
        prolongation: Derived dimensionless covariance-natural ``U_*`` with
            dimensions ``(*native_dims, state_dim)``.
        state_covariance: Positive-definite dimensionless ``C_alpha`` with
            distinct typed row/column state dimensions.
        effective_observation_operator: ``H U_*`` with dimensions
            ``(observation_dim, state_dim)`` and sensitivity units.
        observation_state_cross_covariance: ``H B Pi.T`` with dimensions
            ``(observation_dim, state_column_dim)`` and sensitivity units.
        native_observation_covariance: Dense positive-semidefinite (and
            possibly singular) ``H B H.T`` on distinct typed observation axes,
            or its nonnegative observation-axis diagonal. Units are squared
            sensitivity units.
        strategy: Identifier of the strategy that selected ``Pi``.
        observation_covariance_view: Whether the observation covariance field
            contains the dense matrix or its diagonal.

    Notes:
        Freezing the dataclass prevents field reassignment, but the contained
        DataArrays remain mutable. Product attributes are construction-time
        snapshots, not live views of input attribute mappings.
    """

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
        basis_prolongation: Canonical labelled ``U`` from the basis-side native
            expansion boundary.
        state_dim: Retained-state dimension shared by ``U`` and ``Pi``.
        native_sensitivity: Canonical labelled native sensitivity ``H``.
        observation_dim: Observation dimension in ``H``.
        observation_covariance: Return dense ``H B H.T`` or its diagonal.
        observation_batch_size: Positive integral number of covariance
            right-hand sides per eager block.
        strategy: Authoritative restriction choice; defaults to the
            bucket-preserving restriction.

    Returns:
        Frozen in-memory labelled product value object. Scaling covariance,
        restriction, and prolongation have units ``"1"``. Products linear in
        ``H`` inherit its units; observation covariance uses squared ``H``
        units when supplied.

        The kernel trusts the covariance action's declared real,
        self-adjoint, positive-definite semantics. It does not globally
        certify a matrix-free ``B`` or revalidate products constructed here.

    Raises:
        ValueError: If labelled inputs cannot be exactly aligned, the batch
            size is not positive, or a required Cholesky factorization fails.
    """
    native_dims = tuple(covariance.native_dims)
    batch_size = observation_batch_size
    if batch_size <= 0:
        raise ValueError("observation_batch_size must be positive")

    sensitivity = native_sensitivity.transpose(observation_dim, *native_dims)
    prolongation = basis_prolongation.transpose(*native_dims, state_dim)
    # xr.dot otherwise uses an inner join and could silently discard native
    # cells. Establish the shared grid once before the eager product boundary.
    sensitivity, prolongation = xr.align(
        sensitivity,
        prolongation,
        join="exact",
        copy=False,
    )
    dense_sensitivity = to_dense(sensitivity)
    dense_prolongation = to_dense(prolongation)
    sensitivity_data, prolongation_data = compute(
        dense_sensitivity.data,
        dense_prolongation.data,
    )
    sensitivity = sensitivity.copy(data=sensitivity_data)
    prolongation = prolongation.copy(data=prolongation_data)

    projection_strategy = strategy if strategy is not None else PreserveBucketProlongation()
    projection = projection_strategy.projection(
        covariance,
        prolongation,
        native_dims=native_dims,
        state_dim=state_dim,
    )
    projection = _prepare_projection(
        projection,
        native_reference=sensitivity,
        native_dims=native_dims,
        state_dim=state_dim,
    )
    occupied_names = {
        str(name) for array in (sensitivity, projection.restriction) for name in (*array.dims, *array.coords)
    }
    state_column_dim = matrix_column_dim(state_dim, occupied_names)
    b_pi_t = _apply_restriction_blocks(
        covariance,
        projection.restriction,
        native_dims=native_dims,
        state_dim=state_dim,
        column_dim=state_column_dim,
        rhs_block_size=batch_size,
    )
    state_covariance = _restriction_product_blocks(
        projection.restriction,
        b_pi_t,
        native_dims=native_dims,
        state_dim=state_dim,
        column_dim=state_column_dim,
        rhs_block_size=batch_size,
    )
    state_covariance_values = np.asarray(state_covariance.values, dtype=np.float64)
    state_covariance = state_covariance.copy(
        data=(state_covariance_values + state_covariance_values.T) * 0.5
    ).rename("state_covariance").assign_attrs(
        mathematical_name="C_alpha",
        units="1",
    )
    prolongation, reciprocal_condition = _derive_covariance_natural_prolongation(
        projection.restriction,
        state_covariance,
        b_pi_t,
        native_dims=native_dims,
        state_dim=state_dim,
        state_column_dim=state_column_dim,
    )
    state_covariance = state_covariance.assign_attrs(
        estimated_reciprocal_condition_number=reciprocal_condition,
        condition_number_norm="1",
    )
    h_units = sensitivity.attrs.get("units")
    linear_units = {"units": h_units} if h_units is not None else {}
    effective_operator = xr.dot(sensitivity, prolongation, dim=list(native_dims)).transpose(
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
        prolongation=prolongation,
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
) -> tuple[xr.DataArray, float]:
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
    """Form ``Pi B Pi.T`` from the already materialized restriction."""
    blocks: list[xr.DataArray] = []
    for start in range(0, restriction.sizes[state_dim], rhs_block_size):
        rows = restriction.isel({state_dim: slice(start, start + rhs_block_size)})
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
    if output == "dense":
        dense_output = True
    elif output == "diagonal":
        dense_output = False
    else:
        assert_never(output)
    column_dim = matrix_column_dim(
        observation_dim,
        {str(name) for name in (*sensitivity.dims, *sensitivity.coords)},
    )
    count = sensitivity.sizes[observation_dim]
    values = np.empty(
        (count, count) if dense_output else count,
        dtype=np.result_type(sensitivity.dtype, np.float64),
    )
    diagonal_error_bounds = None if dense_output else np.empty(count, dtype=np.float64)
    native_size = int(np.prod([sensitivity.sizes[dim] for dim in native_dims]))
    machine_epsilon = np.finfo(np.dtype(values.dtype)).eps
    contraction_error_factor = 8.0 * native_size * machine_epsilon
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        rhs = to_column_axis(
            sensitivity.isel({observation_dim: slice(start, stop)}),
            row_dim=observation_dim,
            column_dim=column_dim,
            leading_dims=native_dims,
        )
        b_rhs = covariance.apply(rhs)
        if dense_output:
            block = xr.dot(sensitivity, b_rhs, dim=list(native_dims)).transpose(observation_dim, column_dim)
            values[:, start:stop] = np.asarray(block.values)
        else:
            terms = rhs * b_rhs
            values[start:stop] = np.asarray(terms.sum(dim=list(native_dims)).values)
            assert diagonal_error_bounds is not None
            contraction_magnitudes = np.asarray(
                abs(terms).sum(dim=list(native_dims)).values,
                dtype=np.float64,
            )
            diagonal_error_bounds[start:stop] = contraction_error_factor * np.maximum(
                contraction_magnitudes,
                0.0,
            )
    units = sensitivity.attrs.get("units")
    unit_attrs = {"units": f"({units})^2"} if units is not None else {}
    row_coords = {
        str(name): coordinate
        for name, coordinate in sensitivity.coords.items()
        if set(coordinate.dims).issubset({observation_dim})
    }
    if dense_output:
        values = (values + values.T) * 0.5
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
        return result.assign_attrs(mathematical_name="H B H.T")
    assert diagonal_error_bounds is not None
    if np.any(values < -diagonal_error_bounds):
        raise ValueError("diag(H B H.T) contains materially negative covariance values")
    values = np.where(values < 0.0, 0.0, values)
    result = xr.DataArray(
        values,
        dims=observation_dim,
        coords=row_coords,
        attrs={
            **unit_attrs,
            "mathematical_name": "diag(H B H.T)",
            "minimum_diagonal": float(np.min(values)),
            "diagonal_nonnegative": True,
            "maximum_contraction_error_bound": float(np.max(diagonal_error_bounds)),
            "diagnostic_tolerance": "per_observation_contraction_error_bound",
        },
        name="native_observation_covariance",
    )
    return result


def _prepare_projection(
    projection: RetainedProjection,
    *,
    native_reference: xr.DataArray,
    native_dims: tuple[str, ...],
    state_dim: str,
) -> RetainedProjection:
    """Canonicalize and materialize a strategy result at the extension seam."""
    restriction = projection.restriction.transpose(state_dim, *native_dims).assign_attrs(
        {**projection.restriction.attrs, "units": "1"}
    )
    # A custom strategy is an extension boundary. Exact alignment prevents its
    # restriction from silently contracting over only part of the native grid.
    restriction, _ = xr.align(
        restriction,
        native_reference,
        join="exact",
        copy=False,
    )
    dense_restriction = to_dense(restriction)
    (restriction_data,) = compute(dense_restriction.data)
    restriction = restriction.copy(data=restriction_data)
    values = np.asarray(restriction.values)
    if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
        raise ValueError("Projection strategy restriction must contain finite real values")
    return replace(projection, restriction=restriction)


def _derive_covariance_natural_prolongation(
    restriction: xr.DataArray,
    state_covariance: xr.DataArray,
    b_pi_t: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    state_dim: str,
    state_column_dim: str,
) -> xr.DataArray:
    """Derive labelled ``U_* = B Pi.T C_alpha^-1`` from authoritative ``Pi``.

    The result has dimensions ``(*native_dims, state_dim)`` and preserves
    native coordinates from ``B Pi.T`` and state coordinates from ``Pi``.

    Args:
        restriction: Authoritative labelled ``Pi``.
        state_covariance: Positive-definite ``C_alpha``.
        b_pi_t: Labelled ``B Pi.T``.
        native_dims: Ordered native covariance dimensions.
        state_dim: Retained-state row dimension.
        state_column_dim: Distinct retained-state column dimension.

    Returns:
        The dimensionless covariance-natural prolongation ``U_*`` and the
        estimated reciprocal 1-norm condition number of ``C_alpha``.

    Raises:
        ValueError: If ``C_alpha`` cannot be factored as positive definite.
    """
    covariance_values = np.asarray(state_covariance.values, dtype=np.float64)
    try:
        factor = cho_factor(covariance_values, lower=True, check_finite=True)
        reciprocal_condition, condition_info = dpocon(
            factor[0],
            np.linalg.norm(covariance_values, ord=1),
            uplo="L",
        )
        if condition_info != 0:
            raise np.linalg.LinAlgError(
                f"LAPACK condition estimation failed with info={condition_info}"
            )
        if reciprocal_condition < MIN_RETAINED_RECIPROCAL_CONDITION:
            raise ValueError(
                "C_alpha is too ill-conditioned for a stable retained solve "
                f"(estimated reciprocal 1-norm condition {reciprocal_condition:.3e}; "
                f"minimum {MIN_RETAINED_RECIPROCAL_CONDITION:.3e}); use a "
                "non-redundant retained restriction"
            )
        b_pi_values = np.asarray(b_pi_t.transpose(*native_dims, state_column_dim).values)
        native_shape = tuple(b_pi_t.sizes[dim] for dim in native_dims)
        solved = cho_solve(
            factor,
            b_pi_values.reshape((-1, b_pi_t.sizes[state_column_dim])).T,
            check_finite=True,
        ).T.reshape((*native_shape, restriction.sizes[state_dim]))
    except np.linalg.LinAlgError as exc:
        raise ValueError("C_alpha must be positive definite to derive U_*") from exc
    coords = {
        str(name): coordinate
        for name, coordinate in b_pi_t.coords.items()
        if set(coordinate.dims).issubset(native_dims)
    }
    coords.update(
        {
            str(name): coordinate
            for name, coordinate in restriction.coords.items()
            if set(coordinate.dims).issubset({state_dim})
        }
    )
    return xr.DataArray(
        solved,
        dims=(*native_dims, state_dim),
        coords=coords,
        attrs={
            "mathematical_name": "U_*",
            "definition": "B Pi.T C_alpha^-1",
            "units": "1",
        },
        name="prolongation",
    ), float(reciprocal_condition)
