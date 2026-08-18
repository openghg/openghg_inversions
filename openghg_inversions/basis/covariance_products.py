"""Labelled in-memory native-covariance products for retained scaling states.

This module is an eager numerical kernel. Callers supply a canonical, eager
native sensitivity ``H`` and basis prolongation ``U``; basis operators own any
source expansion and the pipeline owns materialization. Custom restrictions
may remain sparse or Dask-backed until the named projection boundary, where
they are materialized once and reused by all retained-state RHS blocks.

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
from openghg_inversions.native_covariance import InvertibleNativeCovarianceAction

MAX_DENSE_EIGEN_DIAGNOSTIC_SIZE = 512


@dataclass(frozen=True, slots=True)
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
            The authoritative bucket-compatible restriction ``Pi_U``.

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
            cholesky_diagonal = np.diag(factor[0])
            largest_pivot = float(np.max(np.abs(cholesky_diagonal)))
            if (
                not cholesky_diagonal.size
                or largest_pivot <= 0.0
                or float(np.min(np.abs(cholesky_diagonal))) <= 1e-6 * largest_pivot
            ):
                raise np.linalg.LinAlgError("numerically rank-deficient Gram matrix")
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


@dataclass(frozen=True, slots=True)
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
        basis_prolongation: Canonical eager ``U`` from the basis-side native
            expansion boundary.
        state_dim: Retained-state dimension shared by ``U`` and ``Pi``.
        native_sensitivity: Canonical eager native sensitivity ``H``.
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

        The kernel trusts the covariance action's declared self-adjoint
        positive-definite semantics; it does not globally certify a
        matrix-free ``B``. Runtime checks diagnose only the requested products
        and covariance-action outputs encountered here.

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
    state_covariance = with_square_matrix_diagnostics(
        state_covariance.rename("state_covariance").assign_attrs(units="1"),
        mathematical_name="C_alpha",
        require_positive_definite=True,
    )
    prolongation = _derive_covariance_natural_prolongation(
        projection.restriction,
        state_covariance,
        b_pi_t,
        native_dims=native_dims,
        state_dim=state_dim,
        state_column_dim=state_column_dim,
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
        blocks.append(
            _validated_covariance_apply(
                covariance,
                rhs,
                context="B Pi.T",
            )
        )
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


def _validated_covariance_apply(
    covariance: InvertibleNativeCovarianceAction,
    rhs: xr.DataArray,
    *,
    context: str,
) -> xr.DataArray:
    """Apply ``B`` and reject invalid action output.

    Args:
        covariance: Native covariance action.
        rhs: Labelled right-hand sides.
        context: Product name included in validation errors.

    Returns:
        The labelled finite-real action result.

    Raises:
        ValueError: If the result is not a DataArray or contains complex or
            non-finite values.
    """
    result = covariance.apply(rhs)
    if not isinstance(result, xr.DataArray):
        raise ValueError(
            f"covariance.apply action-contract violation while computing {context}: "
            "expected an xarray.DataArray"
        )
    values = np.asarray(result.values)
    if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"covariance.apply action-contract violation while computing {context}: "
            "output must contain only finite real values"
        )
    return result


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
    diagonal_error_bounds = np.empty(count, dtype=np.float64) if output == "diagonal" else None
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
        b_rhs = _validated_covariance_apply(
            covariance,
            rhs,
            context="H B H.T",
        )
        if output == "dense":
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
    assert diagonal_error_bounds is not None
    if not np.all(np.isfinite(diagonal_error_bounds)):
        raise ValueError("diag(H B H.T) numerical error bounds must be finite")
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
    _validate_exact_native_coordinates(
        restriction,
        native_reference,
        native_dims=native_dims,
        role="restriction",
    )
    if state_dim not in restriction.coords:
        raise ValueError("Projection strategy restriction requires state labels")
    _require_reference_native_coordinates(
        restriction,
        native_reference,
        native_dims=native_dims,
        role="restriction",
    )
    restriction = _materialize_dense_preserving_coordinates(restriction)
    restriction_values = np.asarray(restriction.values)
    if np.iscomplexobj(restriction_values) or not np.all(np.isfinite(restriction_values)):
        raise ValueError("Projection strategy restriction must contain only finite real values")
    return replace(projection, restriction=restriction)


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
        The dimensionless covariance-natural prolongation ``U_*``.

    Raises:
        ValueError: If ``C_alpha`` cannot be factored as positive definite.
    """
    covariance_values = np.asarray(state_covariance.values, dtype=np.float64)
    try:
        factor = cho_factor(covariance_values, lower=True, check_finite=True)
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
    )


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
