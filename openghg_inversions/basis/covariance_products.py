"""Labelled in-memory native-covariance products for retained scaling states.

This module is an eager numerical kernel. Callers supply a canonical native
sensitivity ``H`` and a retained projection strategy. Each strategy owns the
inputs that define its restriction ``Pi``: :class:`PreserveBucketProlongation`
holds a basis prolongation ``U`` and :class:`SuppliedRestriction` holds a
physical ``Pi``. Sensitivity payloads may remain sparse or Dask-backed until the
named projection boundary. Restrictions are materialized once and reused by
all retained-state RHS blocks; a sparse ``Pi`` stays sparse, and only bounded
retained-state RHS blocks are densified.

For native covariance ``B`` and retained restriction ``Pi``, the kernel returns
``C_alpha = Pi B Pi.T``, ``H U_*``, ``H B Pi.T``, and ``H B H.T`` (or its
diagonal). Strategies return authoritative ``Pi``; the kernel derives
``B Pi.T``, ``C_alpha``, and ``U_* = B Pi.T C_alpha^-1``.
:class:`PreserveBucketProlongation` preserves bucket prolongation by deriving
``Pi_U = (U.T B^-1 U)^-1 U.T B^-1``. :class:`SuppliedRestriction` instead
uses a caller-defined physical ``Pi`` chosen independently of ``B``, such as
one built by :func:`absolute_weighted_regional_mean_restriction`. The kernel
trusts the covariance action's real self-adjoint positive-definite semantics
and compatible inverse rather than globally certifying a matrix-free operator.
Durable artifact identity and persistence are intentionally deferred to the
artifact-I/O layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import logging
from typing import assert_never, Literal, Protocol

from dask.base import compute
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sparse import COO, SparseArray  # type: ignore[import-untyped]
# scipy-stubs 1.15.3 omits the runtime LAPACK wrappers, including dpocon.
from scipy.linalg.lapack import dpocon  # type: ignore[attr-defined]
import xarray as xr

from openghg_inversions._labelled_matrices import (
    coordinates_on_dims,
    matrix_column_dim,
    square_matrix,
    to_column_axis,
)
from openghg_inversions.array_ops import materialize_together, require_unique_index, to_dense
from openghg_inversions.borrowed import BorrowedDataArray, borrow
from openghg_inversions.native_covariance import InvertibleNativeCovarianceAction


logger = logging.getLogger(__name__)

MIN_RETAINED_RECIPROCAL_CONDITION = float(np.sqrt(np.finfo(np.float64).eps))
ROW_NORMALIZATION_TOLERANCE = 1e-12


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
    """Extension seam for choosing authoritative retained coefficients.

    A strategy owns whatever inputs define its restriction; the kernel passes
    only the covariance action and the dimension names.
    """

    def projection(
        self,
        covariance: InvertibleNativeCovarianceAction,
        *,
        native_dims: tuple[str, ...],
        state_dim: str,
    ) -> RetainedProjection:
        """Return the authoritative labelled retained restriction ``Pi``.

        Args:
            covariance: Labelled native covariance action.
            native_dims: Ordered native covariance dimensions.
            state_dim: Retained-state dimension.

        Returns:
            A dimensionless restriction with dimensions
            ``(state_dim, *native_dims)`` whose retained labels label every
            retained product. Its payload may be sparse or lazy. The kernel
            derives ``B Pi.T``, ``C_alpha``, and ``U_*`` from it; the
            covariance action checks its native grid when applied.
        """
        ...


class PreserveBucketProlongation:
    """Derive ``Pi_U`` so covariance-weighted prolongation equals ``U_bucket``.

    Args:
        basis_prolongation: Canonical labelled bucket prolongation ``U`` with
            dimensions ``(*native_dims, state_dim)``, as returned by a basis
            operator's native expansion. It is borrowed, not copied; its
            payload may be sparse or Dask-backed and is materialized when the
            projection is computed.
        name: Stable strategy identifier recorded with the products.
    """

    __slots__ = ("_basis_prolongation", "_name")

    def __init__(
        self,
        basis_prolongation: xr.DataArray,
        *,
        name: str = "preserve_bucket_prolongation",
    ) -> None:
        """Hold the borrowed prolongation and identifier."""
        self._basis_prolongation = basis_prolongation
        self._name = name

    @property
    def basis_prolongation(self) -> BorrowedDataArray:
        """Return the borrowed bucket prolongation ``U``."""
        return borrow(self._basis_prolongation)

    @property
    def name(self) -> str:
        """Return the strategy identifier."""
        return self._name

    def projection(
        self,
        covariance: InvertibleNativeCovarianceAction,
        *,
        native_dims: tuple[str, ...],
        state_dim: str,
    ) -> RetainedProjection:
        """Construct the prior-precision-compatible restriction.

        Args:
            covariance: Invertible labelled action for native covariance ``B``.
            native_dims: Ordered native dimensions.
            state_dim: Retained-state dimension.

        Returns:
            The authoritative bucket-compatible restriction ``Pi_U``.

        Raises:
            ValueError: If ``U`` is invalid or lacks full column rank.
        """
        # With the Gram matrix G = U.T B^-1 U, the restriction is
        #   Pi_U = G^-1 (B^-1 U).T.
        # Then C_alpha = Pi_U B Pi_U.T = G^-1, and the kernel's derived lift
        # U_* = B Pi_U.T C_alpha^-1 = B (B^-1 U G^-1) G = U recovers the buckets.
        prolongation = self._basis_prolongation.transpose(*native_dims, state_dim)
        (prolongation,) = materialize_together(prolongation, purpose="bucket prolongation U")

        # B^-1 U, with its state axis renamed to a distinct column dimension so
        # that the Gram product contracts only the native axes.
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

        # G = U.T B^-1 U, averaged with its transpose to remove rounding asymmetry.
        gram = xr.dot(prolongation, precision_prolongation, dim=list(native_dims)).transpose(
            state_dim, state_column_dim
        )
        gram_values = np.asarray(gram.values, dtype=np.float64)
        gram_values = (gram_values + gram_values.T) * 0.5

        # Buckets that are linearly dependent under B^-1 make G singular.
        # Cholesky alone does not detect that: Apple Accelerate can accept an
        # exactly singular G, and the noisy G^-1 then gives a Pi with
        # independent garbage rows that later checks on C_alpha accept. The
        # condition estimate from the same factor rejects it. Using the
        # kernel's threshold rejects essentially nothing the kernel would
        # accept: a valid Pi_U has C_alpha = G^-1, whose condition number
        # equals G's.
        if not np.isfinite(gram_values).all():
            raise ValueError("The bucket prolongation Gram matrix must contain only finite values")
        try:
            factor, reciprocal_condition = _cholesky_with_condition(gram_values)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "The bucket prolongation contains redundant retained states; "
                "U.T B^-1 U must be positive definite"
            ) from exc
        if reciprocal_condition < MIN_RETAINED_RECIPROCAL_CONDITION:
            raise ValueError(
                "The bucket prolongation contains redundant or nearly redundant retained states; "
                f"U.T B^-1 U has estimated reciprocal 1-norm condition {reciprocal_condition:.3e} "
                f"(minimum {MIN_RETAINED_RECIPROCAL_CONDITION:.3e})"
            )
        inverse_gram_values = cho_solve(factor, np.eye(gram_values.shape[0]), check_finite=False)

        # Pi_U = G^-1 (B^-1 U).T, contracting G^-1's column axis with B^-1 U's.
        inverse_gram = square_matrix(
            inverse_gram_values,
            prolongation,
            row_dim=state_dim,
            column_dim=state_column_dim,
            attrs={"units": "1"},
        )
        restriction = xr.dot(inverse_gram, precision_prolongation, dim=state_column_dim).transpose(
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


class SuppliedRestriction:
    """Use a caller-supplied physical restriction ``Pi`` as retained coordinates.

    The restriction defines ``alpha = Pi x`` before, and independently of, the
    native covariance ``B``. The product kernel then derives
    ``C_alpha = Pi B Pi.T``, the covariance-natural lift
    ``U_* = B Pi.T C_alpha^-1``, and ``H_alpha = H U_*``. The strategy is
    policy-neutral: absolute-flux-weighted regional means (see
    :func:`absolute_weighted_regional_mean_restriction`), signed flux-total
    restrictions, and other physical restrictions are all accepted. The
    restriction's state coordinate labels every retained product.

    ``alpha = Pi x`` is positive for every positive native scaling ``x``
    exactly when every entry of ``Pi`` is nonnegative and every row has
    positive weight; call such a ``Pi`` positivity preserving. A LogNormal
    retained state needs this, so pass ``require_positivity_preserving=True``
    for one. The derived lift ``U_*`` may still be signed, because it is a
    linear conditional mean, not a positive scaling map. Row normalization is
    recorded, not required, because preserving a mean-one scaling is a policy
    choice of the caller.

    Args:
        restriction: Dimensionless ``Pi`` with dimensions exactly
            ``(state_dim, *native_dims)`` in any order and a unique indexed
            state coordinate. It is borrowed, not copied, and is read when the
            projection is computed. It may be NumPy, ``sparse`` or
            Dask-backed; a sparse payload must have zero fill value and is
            kept sparse.
        require_positivity_preserving: Refuse a restriction with a negative
            entry or a row without positive weight. Without it, any finite
            real full-row-rank restriction is accepted.
        name: Stable strategy identifier recorded with the products.
    """

    __slots__ = ("_name", "_require_positivity_preserving", "_restriction")

    def __init__(
        self,
        restriction: xr.DataArray,
        *,
        require_positivity_preserving: bool = False,
        name: str = "supplied_restriction",
    ) -> None:
        """Hold the borrowed restriction and its positivity requirement."""
        self._restriction = restriction
        self._require_positivity_preserving = require_positivity_preserving
        self._name = name

    @property
    def restriction(self) -> BorrowedDataArray:
        """Return the borrowed supplied restriction ``Pi``."""
        return borrow(self._restriction)

    @property
    def require_positivity_preserving(self) -> bool:
        """Return whether a signed restriction or a row without weight is refused."""
        return self._require_positivity_preserving

    @property
    def name(self) -> str:
        """Return the strategy identifier."""
        return self._name

    def projection(
        self,
        covariance: InvertibleNativeCovarianceAction,
        *,
        native_dims: tuple[str, ...],
        state_dim: str,
    ) -> RetainedProjection:
        """Validate and return the supplied restriction with provenance.

        Full row rank is established by the kernel's positive-definite
        factorization of ``C_alpha``, which is equivalent for positive-definite
        ``B``. Exact native coordinates are checked by the covariance action
        when the kernel applies ``B`` to ``Pi``.

        Args:
            covariance: Native covariance action; unused because ``Pi`` is
                chosen independently of ``B``.
            native_dims: Ordered native dimensions.
            state_dim: Retained-state dimension.

        Returns:
            The materialized restriction, with sparse payloads kept sparse and
            attributes recording its supplied provenance and positivity
            diagnostics.

        Raises:
            ValueError: If dimensions, labels, or values are invalid, or
                positivity preservation is required and ``Pi`` lacks it.
        """
        del covariance

        # transpose rejects missing or extra dimensions.
        restriction = self._restriction.transpose(state_dim, *native_dims)
        state_index = require_unique_index(restriction, state_dim, name="Supplied restriction")
        restriction = _materialize_restriction(restriction)

        most_negative_weights, row_sums = _restriction_row_diagnostics(restriction, state_dim=state_dim)
        nonnegative = bool(np.all(most_negative_weights >= 0.0))
        positive_row_sums = bool(np.all(row_sums > 0.0))
        row_normalized = bool(np.all(np.abs(row_sums - 1.0) <= ROW_NORMALIZATION_TOLERANCE))

        if self._require_positivity_preserving:
            if not nonnegative:
                row = int(np.argmin(most_negative_weights))
                raise ValueError(
                    "A positivity-preserving restriction must be nonnegative; "
                    f"row {_label(state_index, row)!r} has weight {most_negative_weights[row]:.3e}. A signed "
                    "Pi can map positive native scalings to nonpositive retained scalings"
                )
            if not positive_row_sums:
                row = int(np.argmin(row_sums))
                raise ValueError(
                    "A positivity-preserving restriction needs positive weight in every "
                    f"restriction row; row {_label(state_index, row)!r} sums to {row_sums[row]:.3e}"
                )

        # Keep any caller-recorded definition, such as the builder's weighting
        # formula; restriction_source records that Pi was supplied.
        restriction = restriction.rename("restriction").assign_attrs(
            mathematical_name="Pi",
            strategy=self._name,
            restriction_source="supplied",
            positivity_preserving_required=self._require_positivity_preserving,
            nonnegative=nonnegative,
            positive_row_sums=positive_row_sums,
            row_normalized=row_normalized,
            positivity_preserving=nonnegative and positive_row_sums,
            units="1",
        )
        return RetainedProjection(restriction=restriction, strategy=self._name)


def absolute_weighted_regional_mean_restriction(
    regions: Mapping[str, xr.DataArray],
    weights: xr.DataArray,
    *,
    state_dim: str = "state",
    native_source_dim: str = "native_source",
    source_dim: str = "source",
    region_dim: str = "region_in_source",
) -> xr.DataArray:
    """Build a source-blocked regional mean weighted by absolute cell totals.

    This is one restriction policy for :class:`SuppliedRestriction`. For
    source ``s``, region ``r`` and native cell ``i`` with signed weight
    ``q_si``, the restriction is

    ``Pi_(s,r),(s',i) = 1[s = s'] 1[i in r] |q_si| / sum_(j in r) |q_sj|``.

    Each row is therefore exactly nonnegative, sums to one, and averages
    native scaling factors within one source and one region, so the result is
    positivity preserving, as a LogNormal retained state needs. One choice of
    weights is ``q_si = mean(prior flux)_si * cell_area_i``, which weights each
    regional mean by its cells' prior emissions.

    Inputs are materialized together at this eager boundary. The result is
    sparse, with at most one stored weight per active native cell; cells with
    zero weight store nothing.

    Args:
        regions: Non-empty insertion-ordered mapping from source label to an
            integer region-label array on the spatial native grid. Zero and
            positive labels are active regions, so zero-based labels such as
            those from ``range`` work directly; negative labels mark inactive
            cells, which receive no weight.
        weights: Signed per-cell weights ``q`` with dimensions
            ``(native_source_dim, *spatial_dims)``. Its source coordinate must
            list the ``regions`` keys in the same order, and its spatial
            coordinates must match each region array exactly. Only absolute
            values are used, so units cancel.
        state_dim: Retained-state dimension of the result.
        native_source_dim: Native source dimension in ``weights``.
        source_dim: Source level of the retained-state MultiIndex.
        region_dim: Region level of the retained-state MultiIndex. The
            defaults match
            :class:`~openghg_inversions.basis.operators.MultiSourceBucketBasisOperator`.

    Returns:
        Dimensionless sparse ``Pi`` with dimensions
        ``(state_dim, native_source_dim, *spatial_dims)``. The state
        coordinate is a ``(source_dim, region_dim)`` MultiIndex in source order.
        Each source's active regions are numbered ``0, 1, ...`` in ascending
        label order; the map's label values carry no meaning in the result.
        For maps without inactive cells, this numbering matches
        :class:`~openghg_inversions.basis.operators.MultiSourceBucketBasisOperator`.

    Raises:
        ValueError: If no sources are given, source or spatial labels do not
            match, region labels are not integers, weights are not finite, or
            an active region has zero total absolute weight.
    """
    # Check the source axis of the weights against the region maps.
    if not regions:
        raise ValueError("regions must contain at least one source")
    sources = list(regions)
    if native_source_dim not in weights.indexes:
        raise ValueError(
            f"weights must have an indexed native source coordinate {native_source_dim!r} "
            "listing the regions keys"
        )
    if weights.indexes[native_source_dim].tolist() != sources:
        raise ValueError(
            f"weights {native_source_dim!r} labels must equal the regions keys in order; "
            f"got {weights.indexes[native_source_dim].tolist()!r}, expected {sources!r}"
        )
    spatial_dims = tuple(str(dim) for dim in weights.dims if dim != native_source_dim)
    weights = weights.transpose(native_source_dim, *spatial_dims)

    # Check each region map, align it exactly with the weights' spatial grid,
    # then materialize all maps and weights together.
    aligned_regions = []
    for source in sources:
        source_regions = regions[source]
        if set(source_regions.dims) != set(spatial_dims):
            raise ValueError(
                f"regions[{source!r}] must have dimensions {spatial_dims!r}; got {source_regions.dims!r}"
            )
        if not np.issubdtype(source_regions.dtype, np.integer):
            raise ValueError(
                f"regions[{source!r}] must contain integer region labels; got {source_regions.dtype}"
            )
        # Exact alignment rejects missing, extra, or reordered native cells.
        source_regions, _ = xr.align(
            source_regions.transpose(*spatial_dims),
            weights,
            join="exact",
            copy=False,
            exclude=[native_source_dim],
        )
        aligned_regions.append(source_regions)
    *materialized_regions, weights = materialize_together(
        *aligned_regions, weights, purpose="regional-mean region maps and weights"
    )
    weight_values = np.asarray(weights.data)
    if np.iscomplexobj(weight_values) or not np.all(np.isfinite(weight_values)):
        raise ValueError("weights must contain finite real values")

    # Build the sparse entries one source block at a time. Each source's rows
    # follow the rows of the sources before it, starting at row_offset.
    spatial_shape = weight_values.shape[1:]
    state_labels: list[tuple[str, int]] = []
    row_offset = 0
    row_blocks, source_blocks, cell_blocks, value_blocks = [], [], [], []
    for source_index, (source, source_regions) in enumerate(zip(sources, materialized_regions, strict=True)):
        flat_labels = np.asarray(source_regions.data).reshape(-1)
        absolute = np.abs(weight_values[source_index].reshape(-1))
        active_cells = np.flatnonzero(flat_labels >= 0)

        # The inverse maps each active cell's arbitrary region label to its
        # consecutive local row 0, 1, ... in ascending label order. bincount
        # then sums |q| per local row, giving each region's denominator.
        source_region_labels, local_rows = np.unique(flat_labels[active_cells], return_inverse=True)
        region_count = source_region_labels.size
        totals = np.bincount(local_rows, weights=absolute[active_cells], minlength=region_count)
        if np.any(totals <= 0.0):
            region = int(source_region_labels[np.argmax(totals <= 0.0)])
            raise ValueError(f"Region {region} of source {source!r} has zero total absolute weight")

        row_blocks.append(row_offset + local_rows)
        source_blocks.append(np.full(active_cells.size, source_index))
        cell_blocks.append(active_cells)
        value_blocks.append(absolute[active_cells] / totals[local_rows])
        state_labels.extend((source, position) for position in range(region_count))
        row_offset += region_count
    if not state_labels:
        raise ValueError("regions must contain at least one active (nonnegative) region label")

    # Assemble Pi. COO coordinates are (state row, native source, *spatial
    # indices); unravel_index turns flattened active cells back into grid
    # indices. An active cell with zero weight contributes nothing to its
    # regional mean, so it is not stored and the COO holds no explicit zeros.
    cells = np.concatenate(cell_blocks)
    coordinates = np.vstack(
        [
            np.concatenate(row_blocks),
            np.concatenate(source_blocks),
            *np.unravel_index(cells, spatial_shape),
        ]
    )
    values = np.concatenate(value_blocks)
    stored = values > 0.0
    data = COO(
        coordinates[:, stored],
        values[stored],
        shape=(len(state_labels), len(sources), *spatial_shape),
    )

    # Label Pi: native coordinates come from the weights, and the state is a
    # (source, region) MultiIndex in row order.
    state_index = pd.MultiIndex.from_tuples(state_labels, names=(source_dim, region_dim))
    return xr.DataArray(
        data,
        dims=(state_dim, native_source_dim, *spatial_dims),
        coords=coordinates_on_dims(weights, {native_source_dim, *spatial_dims}),
        name="restriction",
        attrs={
            "mathematical_name": "Pi",
            "definition": "|q_si| / sum_(j in r) |q_sj| within one source and region",
            "policy": "absolute-weighted regional mean",
            "units": "1",
        },
    ).assign_coords(xr.Coordinates.from_pandas_multiindex(state_index, state_dim))


@dataclass(frozen=True, slots=True, eq=False)
class NativeCovarianceProducts:
    """Labelled in-memory product blocks induced by one retained restriction.

    Attributes:
        restriction: Dimensionless ``Pi`` with dimensions
            ``(state_dim, *native_dims)``. It is eager; a sparse payload
            selected by the strategy remains sparse.
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
    strategy: RetainedProjectionStrategy,
    state_dim: str,
    native_sensitivity: xr.DataArray,
    observation_dim: str,
    observation_covariance: Literal["dense", "diagonal"] = "dense",
    observation_batch_size: int = 64,
) -> NativeCovarianceProducts:
    """Compute coherent labelled native-covariance products eagerly.

    Args:
        covariance: Labelled native covariance action with a compatible solve.
        strategy: Authoritative restriction choice, owning the inputs that
            define ``Pi``. Use :class:`PreserveBucketProlongation` with a
            basis prolongation, or :class:`SuppliedRestriction` with a
            physical restriction.
        state_dim: Retained-state dimension of ``Pi``.
        native_sensitivity: Canonical labelled native sensitivity ``H``.
        observation_dim: Observation dimension in ``H``.
        observation_covariance: Return dense ``H B H.T`` or its diagonal.
        observation_batch_size: Positive integral number of covariance
            right-hand sides per eager block. The same size bounds the
            retained-state blocks of ``Pi`` to which ``B`` is applied.

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
    if observation_batch_size <= 0:
        raise ValueError("observation_batch_size must be positive")

    # Resolve Pi and materialize H. The covariance action owns the native grid:
    # each apply or solve aligns its right-hand side exactly, so Pi, U and H
    # are checked where they meet B.
    projection = _prepare_projection(
        strategy.projection(covariance, native_dims=native_dims, state_dim=state_dim),
        native_dims=native_dims,
        state_dim=state_dim,
    )
    sensitivity = native_sensitivity.transpose(observation_dim, *native_dims)
    (sensitivity,) = materialize_together(sensitivity, purpose="native sensitivity H")

    # Retained-state products. state_column_dim is a fresh name for the
    # second state axis of B Pi.T and C_alpha, so xarray never contracts both
    # state axes at once.
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
        rhs_block_size=observation_batch_size,
    )
    # C_alpha = Pi B Pi.T. A sparse Pi contracts over its stored weights only,
    # with no dense blocks; the result is a dense d x d matrix for d retained
    # states. Averaging with its transpose removes rounding asymmetry.
    state_covariance = xr.dot(projection.restriction, b_pi_t, dim=list(native_dims)).transpose(
        state_dim, state_column_dim
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

    # Observation-space products, which carry the units of H.
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
        batch_size=observation_batch_size,
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
    """Apply covariance to retained-state RHS blocks, densifying one block at a time."""
    blocks: list[xr.DataArray] = []
    for start in range(0, restriction.sizes[state_dim], rhs_block_size):
        sliced = restriction.isel({state_dim: slice(start, start + rhs_block_size)})
        rhs = to_column_axis(
            to_dense(sliced),
            row_dim=state_dim,
            column_dim=column_dim,
            leading_dims=native_dims,
        )
        blocks.append(covariance.apply(rhs))
    return xr.concat(blocks, dim=column_dim).transpose(*native_dims, column_dim)


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

    # Each block applies B to batch_size columns of H.T, renamed to a distinct
    # column dimension, then contracts with H over the native axes. `values`
    # holds the full matrix, or only its diagonal.
    column_dim = matrix_column_dim(
        observation_dim,
        {str(name) for name in (*sensitivity.dims, *sensitivity.coords)},
    )
    count = sensitivity.sizes[observation_dim]
    values = np.empty(
        (count, count) if dense_output else count,
        dtype=np.result_type(sensitivity.dtype, np.float64),
    )
    # The rounding error of a sum of n terms is bounded by about
    # n * eps * sum(|terms|); the factor 8 adds headroom above that bound.
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
            # diag(H B H.T)_k = sum_i H_ki (B H.T)_ik, one native sum per observation.
            terms = rhs * b_rhs
            values[start:stop] = np.asarray(terms.sum(dim=list(native_dims)).values)
            assert diagonal_error_bounds is not None
            contraction_magnitudes = np.asarray(abs(terms).sum(dim=list(native_dims)).values, dtype=np.float64)
            diagonal_error_bounds[start:stop] = contraction_error_factor * contraction_magnitudes

    units = sensitivity.attrs.get("units")
    unit_attrs = {"units": f"({units})^2"} if units is not None else {}
    if dense_output:
        # Averaging with the transpose removes rounding asymmetry between blocks.
        return square_matrix(
            (values + values.T) * 0.5,
            sensitivity,
            row_dim=observation_dim,
            column_dim=column_dim,
            name="native_observation_covariance",
            attrs={**unit_attrs, "mathematical_name": "H B H.T"},
        )

    # A diagonal entry is a variance. Negative values within the rounding bound
    # are clipped to zero; larger ones mean B is not positive semidefinite.
    assert diagonal_error_bounds is not None
    if np.any(values < -diagonal_error_bounds):
        raise ValueError("diag(H B H.T) contains materially negative covariance values")
    values = np.where(values < 0.0, 0.0, values)
    result = xr.DataArray(
        values,
        dims=observation_dim,
        coords=coordinates_on_dims(sensitivity, {observation_dim}),
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
    native_dims: tuple[str, ...],
    state_dim: str,
) -> RetainedProjection:
    """Canonicalize and materialize a strategy result at the extension seam."""
    restriction = projection.restriction.transpose(state_dim, *native_dims).assign_attrs(
        {**projection.restriction.attrs, "units": "1"}
    )
    restriction = _materialize_restriction(restriction)
    return replace(projection, restriction=restriction)


def _materialize_restriction(restriction: xr.DataArray) -> xr.DataArray:
    """Compute ``Pi`` once, keep sparse storage sparse, and require finite real values.

    Args:
        restriction: Labelled restriction whose payload may be NumPy, sparse,
            or Dask-backed with either chunk type.

    Returns:
        An eager restriction whose payload is a NumPy array or a zero-fill
        :class:`sparse.COO` array.

    Raises:
        ValueError: If a sparse payload has nonzero fill value or any stored
            value is complex or non-finite.
    """
    # Unlike array_ops.materialize_together, this boundary keeps sparse Pi sparse.
    logger.debug("Materializing retained restriction Pi%s, keeping sparse storage", tuple(restriction.shape))
    (data,) = compute(restriction.data)
    if isinstance(data, SparseArray):
        data = COO(data)
        if data.fill_value != 0:
            raise ValueError("A sparse restriction must use a zero fill value")
        stored = data.data
    else:
        data = np.asarray(data)
        stored = data
    if np.iscomplexobj(stored) or not np.all(np.isfinite(stored)):
        raise ValueError("Projection strategy restriction must contain finite real values")
    return restriction.copy(data=data)


def _restriction_row_diagnostics(
    restriction: xr.DataArray,
    *,
    state_dim: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's most negative weight and its sum for eager ``Pi``, without densifying.

    The most negative weight is ``min(0, row minimum)``: zero for a row with no
    negative entry, otherwise its smallest entry. Callers only need to find
    negative entries and report one, so starting from zero avoids scanning the
    implicit zeros of a sparse row.

    Args:
        restriction: Eager ``Pi`` with ``state_dim`` first, as returned by
            :func:`_materialize_restriction` after canonical transposition.
        state_dim: Retained-state dimension.

    Returns:
        The most negative weight and the sum of each row, as float arrays.
    """
    data = restriction.data
    row_count = restriction.sizes[state_dim]
    native_axes = tuple(axis for axis, dim in enumerate(restriction.dims) if dim != state_dim)
    most_negative_weights = np.zeros(row_count, dtype=np.float64)
    if isinstance(data, COO):
        # coords[0] is each stored entry's row, because state_dim is axis 0.
        np.minimum.at(most_negative_weights, data.coords[0], data.data)
        row_sums = data.sum(axis=native_axes).todense()
    else:
        values = np.asarray(data).reshape(row_count, -1)
        if values.shape[1]:  # min() fails on rows with no native cells
            most_negative_weights = np.minimum(most_negative_weights, values.min(axis=1))
        row_sums = values.sum(axis=1)
    return most_negative_weights, np.asarray(row_sums, dtype=np.float64)


def _label(index: pd.Index, position: int) -> object:
    """Return one index label with NumPy scalars converted to Python values for messages."""
    label = index[position]
    if isinstance(label, tuple):
        return tuple(value.item() if isinstance(value, np.generic) else value for value in label)
    return label.item() if isinstance(label, np.generic) else label


def _cholesky_with_condition(matrix: np.ndarray) -> tuple[tuple[np.ndarray, bool], float]:
    """Cholesky-factor a symmetric matrix and estimate its reciprocal condition.

    A successful Cholesky factorization does not prove that a matrix is
    numerically nonsingular: some LAPACK builds, such as Apple Accelerate,
    accept an exactly singular matrix with a tiny positive last pivot. The
    LAPACK ``dpocon`` estimate reuses the factor, so callers can reject such a
    matrix without an eigendecomposition. See
    ``docs/development/validation_and_xarray.rst``.

    Args:
        matrix: Real symmetric matrix expected to be positive definite.

    Returns:
        The lower Cholesky factor in :func:`scipy.linalg.cho_factor` form,
        and the estimated reciprocal 1-norm condition number.

    Raises:
        numpy.linalg.LinAlgError: If the factorization or condition estimate
            fails.
    """
    factor = cho_factor(matrix, lower=True, check_finite=True)
    reciprocal_condition, condition_info = dpocon(factor[0], np.linalg.norm(matrix, ord=1), uplo="L")
    if condition_info != 0:
        raise np.linalg.LinAlgError(f"LAPACK condition estimation failed with info={condition_info}")
    return factor, float(reciprocal_condition)


def _derive_covariance_natural_prolongation(
    restriction: xr.DataArray,
    state_covariance: xr.DataArray,
    b_pi_t: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    state_dim: str,
    state_column_dim: str,
) -> tuple[xr.DataArray, float]:
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
        factor, reciprocal_condition = _cholesky_with_condition(covariance_values)
        if reciprocal_condition < MIN_RETAINED_RECIPROCAL_CONDITION:
            raise ValueError(
                "C_alpha is too ill-conditioned for a stable retained solve "
                f"(estimated reciprocal 1-norm condition {reciprocal_condition:.3e}; "
                f"minimum {MIN_RETAINED_RECIPROCAL_CONDITION:.3e}); use a "
                "non-redundant retained restriction"
            )
        b_pi_values = np.asarray(b_pi_t.transpose(*native_dims, state_column_dim).values)
        native_shape = tuple(b_pi_t.sizes[dim] for dim in native_dims)
        # U_* C_alpha = B Pi.T and C_alpha is symmetric, so solve
        # C_alpha U_*.T = (B Pi.T).T with each flattened native cell as one
        # right-hand-side column, then restore the native shape.
        solved = cho_solve(
            factor,
            b_pi_values.reshape((-1, b_pi_t.sizes[state_column_dim])).T,
            check_finite=True,
        ).T.reshape((*native_shape, restriction.sizes[state_dim]))
    except np.linalg.LinAlgError as exc:
        raise ValueError("C_alpha must be positive definite to derive U_*") from exc

    # U_* takes its native labels from B Pi.T and its state labels from Pi.
    return xr.DataArray(
        solved,
        dims=(*native_dims, state_dim),
        coords={**coordinates_on_dims(b_pi_t, native_dims), **coordinates_on_dims(restriction, {state_dim})},
        attrs={
            "mathematical_name": "U_*",
            "definition": "B Pi.T C_alpha^-1",
            "units": "1",
        },
        name="prolongation",
    ), float(reciprocal_condition)
