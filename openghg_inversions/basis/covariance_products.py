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
    matrix_column_dim,
    renamed_column_coordinates,
    to_column_axis,
)
from openghg_inversions.array_ops import require_unique_index, to_dense
from openghg_inversions.borrowed import BorrowedDataArray, borrow
from openghg_inversions.native_covariance import InvertibleNativeCovarianceAction


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
            checks its native grid against ``H`` and derives ``B Pi.T``,
            ``C_alpha``, and ``U_*`` from it.
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
        prolongation = self._basis_prolongation.transpose(*native_dims, state_dim)
        (prolongation_data,) = compute(to_dense(prolongation).data)
        prolongation = prolongation.copy(data=prolongation_data)
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
        if not np.isfinite(gram_values).all():
            raise ValueError("The bucket prolongation Gram matrix must contain only finite values")
        eigenvalues = np.linalg.eigvalsh(gram_values)
        if eigenvalues.size and eigenvalues[0] <= len(eigenvalues) * np.finfo(np.float64).eps * eigenvalues[-1]:
            raise ValueError("The bucket prolongation contains redundant retained states")
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

    A LogNormal retained scaling ``alpha = Pi x`` is positive for every
    positive native scaling ``x`` only when ``Pi`` is nonnegative with a
    positive weight in each row. ``retained_distribution="lognormal"``
    therefore refuses a signed restriction or an all-zero row; the derived lift
    ``U_*`` may still be signed because it is a linear conditional mean, not a
    positive scaling map. Row normalization is recorded, not required, because
    preserving a mean-one scaling is a policy choice of the caller.

    Args:
        restriction: Dimensionless ``Pi`` with dimensions exactly
            ``(state_dim, *native_dims)`` in any order and a unique indexed
            state coordinate. It is borrowed, not copied, and is read when the
            projection is computed. It may be NumPy, ``sparse`` or
            Dask-backed; a sparse payload must have zero fill value and is
            kept sparse.
        retained_distribution: Retained-state prior family used to decide
            admissibility. ``"gaussian"`` accepts any finite real full-row-rank
            restriction.
        name: Stable strategy identifier recorded with the products.
    """

    __slots__ = ("_name", "_restriction", "_retained_distribution")

    def __init__(
        self,
        restriction: xr.DataArray,
        *,
        retained_distribution: Literal["gaussian", "lognormal"] = "gaussian",
        name: str = "supplied_restriction",
    ) -> None:
        """Hold the borrowed restriction and its admissibility setting."""
        self._restriction = restriction
        self._retained_distribution = retained_distribution
        self._name = name

    @property
    def restriction(self) -> BorrowedDataArray:
        """Return the borrowed supplied restriction ``Pi``."""
        return borrow(self._restriction)

    @property
    def retained_distribution(self) -> Literal["gaussian", "lognormal"]:
        """Return the retained-state prior family used for admissibility."""
        return self._retained_distribution

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
        ``B``. Exact native coordinates are checked against ``H`` by the
        kernel.

        Args:
            covariance: Native covariance action; unused because ``Pi`` is
                chosen independently of ``B``.
            native_dims: Ordered native dimensions.
            state_dim: Retained-state dimension.

        Returns:
            The materialized restriction, with sparse payloads kept sparse and
            attributes recording its supplied provenance and admissibility.

        Raises:
            ValueError: If dimensions, labels, or values are invalid, or the
                restriction is not admissible for a LogNormal retained state.
        """
        del covariance
        expected_dims = {state_dim, *native_dims}
        supplied = self._restriction
        if set(supplied.dims) != expected_dims or supplied.ndim != len(expected_dims):
            raise ValueError(
                f"Supplied restriction must have dimensions {(state_dim, *native_dims)!r}; "
                f"got {supplied.dims!r}"
            )
        restriction = supplied.transpose(state_dim, *native_dims)
        state_index = require_unique_index(restriction, state_dim, name="Supplied restriction")
        restriction = _materialize_restriction(restriction)
        row_minima, row_sums = _restriction_row_diagnostics(restriction, state_dim=state_dim)
        nonnegative = bool(np.all(row_minima >= 0.0))
        positive_rows = bool(np.all(row_sums > 0.0))
        if self._retained_distribution == "lognormal":
            if not nonnegative:
                row = int(np.argmin(row_minima))
                raise ValueError(
                    "A LogNormal retained state requires a nonnegative restriction; "
                    f"row {_label(state_index, row)!r} has weight {row_minima[row]:.3e}. A signed "
                    "Pi can map positive native scalings to nonpositive retained scalings"
                )
            if not positive_rows:
                row = int(np.argmin(row_sums))
                raise ValueError(
                    "A LogNormal retained state requires positive weight in every "
                    f"restriction row; row {_label(state_index, row)!r} sums to {row_sums[row]:.3e}"
                )
        elif self._retained_distribution != "gaussian":
            assert_never(self._retained_distribution)
        row_normalized = bool(np.all(np.abs(row_sums - 1.0) <= ROW_NORMALIZATION_TOLERANCE))
        restriction = restriction.rename("restriction").assign_attrs(
            mathematical_name="Pi",
            definition="supplied",
            strategy=self._name,
            restriction_source="supplied",
            retained_distribution=self._retained_distribution,
            nonnegative=nonnegative,
            row_normalized=row_normalized,
            lognormal_admissible=nonnegative and positive_rows,
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

    For source ``s``, region ``r`` and native cell ``i`` with signed weight
    ``q_si``, the restriction is

    ``Pi_(s,r),(s',i) = 1[s = s'] 1[i in r] |q_si| / sum_(j in r) |q_sj|``.

    Each row is therefore exactly nonnegative, sums to one, and averages
    native scaling factors within one source and one region, so the result is
    admissible for a LogNormal retained state. The verification-games 14-site
    production runs use ``q_si = mean(prior flux)_si * cell_area_i``. This is
    one supplied policy for :class:`SuppliedRestriction`, not a default.

    Inputs are materialized together at this eager boundary. The result is
    sparse, with one stored weight per active native cell.

    Args:
        regions: Non-empty insertion-ordered mapping from source label to an
            integer region-label array on the spatial native grid. Positive
            labels are active regions; zero or negative labels mark inactive
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
        As in :class:`~openghg_inversions.basis.operators.MultiSourceBucketBasisOperator`,
        each source's active regions are numbered ``0, 1, ...`` in ascending
        label order; the map's label values carry no meaning in the result.

    Raises:
        ValueError: If no sources are given, source or spatial labels do not
            match, region labels are not integers, weights are not finite, or
            an active region has zero total absolute weight.
    """
    if not regions:
        raise ValueError("regions must contain at least one source")
    sources = list(regions)
    if native_source_dim not in weights.dims:
        raise ValueError(f"weights must have native source dimension {native_source_dim!r}")
    if weights.indexes[native_source_dim].tolist() != sources:
        raise ValueError(
            f"weights {native_source_dim!r} labels must equal the regions keys in order; "
            f"got {weights.indexes[native_source_dim].tolist()!r}, expected {sources!r}"
        )
    spatial_dims = tuple(str(dim) for dim in weights.dims if dim != native_source_dim)
    weights = weights.transpose(native_source_dim, *spatial_dims)
    labels = []
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
        labels.append(to_dense(source_regions).data)
    label_values, weight_values = compute(labels, to_dense(weights).data)
    weight_values = np.asarray(weight_values)
    if np.iscomplexobj(weight_values) or not np.all(np.isfinite(weight_values)):
        raise ValueError("weights must contain finite real values")

    spatial_shape = weight_values.shape[1:]
    state_labels: list[tuple[str, int]] = []
    row_blocks, source_blocks, cell_blocks, value_blocks = [], [], [], []
    for source_index, (source, source_labels) in enumerate(zip(sources, label_values, strict=True)):
        flat_labels = np.asarray(source_labels).reshape(-1)
        absolute = np.abs(weight_values[source_index].reshape(-1))
        active_cells = np.flatnonzero(flat_labels > 0)
        source_region_labels, local_rows = np.unique(flat_labels[active_cells], return_inverse=True)
        totals = np.bincount(local_rows, weights=absolute[active_cells], minlength=source_region_labels.size)
        if np.any(totals <= 0.0):
            region = int(source_region_labels[np.argmax(totals <= 0.0)])
            raise ValueError(f"Region {region} of source {source!r} has zero total absolute weight")
        row_blocks.append(len(state_labels) + local_rows)
        source_blocks.append(np.full(active_cells.size, source_index))
        cell_blocks.append(active_cells)
        value_blocks.append(absolute[active_cells] / totals[local_rows])
        state_labels.extend((source, position) for position in range(source_region_labels.size))
    if not state_labels:
        raise ValueError("regions must contain at least one active (positive) region label")

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
    native_coords = {
        str(name): coordinate
        for name, coordinate in weights.coords.items()
        if set(coordinate.dims).issubset({native_source_dim, *spatial_dims})
    }
    state_index = pd.MultiIndex.from_tuples(state_labels, names=(source_dim, region_dim))
    return xr.DataArray(
        data,
        dims=(state_dim, native_source_dim, *spatial_dims),
        coords=native_coords,
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
    sensitivity = native_sensitivity.transpose(observation_dim, *native_dims)
    projection = _resolve_projection(
        strategy,
        covariance,
        native_reference=sensitivity,
        native_dims=native_dims,
        state_dim=state_dim,
    )
    (sensitivity_data,) = compute(to_dense(sensitivity).data)
    return _native_covariance_products(
        covariance=covariance,
        projection=projection,
        sensitivity=sensitivity.copy(data=sensitivity_data),
        state_dim=state_dim,
        observation_dim=observation_dim,
        observation_covariance=observation_covariance,
        batch_size=observation_batch_size,
    )


def _resolve_projection(
    strategy: RetainedProjectionStrategy,
    covariance: InvertibleNativeCovarianceAction,
    *,
    native_reference: xr.DataArray,
    native_dims: tuple[str, ...],
    state_dim: str,
) -> RetainedProjection:
    """Select ``Pi`` and check it against the possibly lazy native grid of ``H``.

    Exact alignment uses indexes only, so a mismatched grid fails before any
    lazy sensitivity payload is computed.
    """
    projection = strategy.projection(covariance, native_dims=native_dims, state_dim=state_dim)
    return _prepare_projection(
        projection,
        native_reference=native_reference,
        native_dims=native_dims,
        state_dim=state_dim,
    )


def _native_covariance_products(
    *,
    covariance: InvertibleNativeCovarianceAction,
    projection: RetainedProjection,
    sensitivity: xr.DataArray,
    state_dim: str,
    observation_dim: str,
    observation_covariance: Literal["dense", "diagonal"],
    batch_size: int,
) -> NativeCovarianceProducts:
    """Form all products from a prepared ``Pi`` and eager canonical ``H``."""
    native_dims = tuple(covariance.native_dims)
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
        rows = to_dense(restriction.isel({state_dim: slice(start, start + rhs_block_size)}))
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
    """Return each row's minimum (capped at zero) and sum for eager ``Pi``, without densifying."""
    data = restriction.data
    row_count = restriction.sizes[state_dim]
    native_axes = tuple(axis for axis, dim in enumerate(restriction.dims) if dim != state_dim)
    row_minima = np.zeros(row_count, dtype=np.float64)
    if isinstance(data, COO):
        np.minimum.at(row_minima, data.coords[0], data.data)
        row_sums = data.sum(axis=native_axes).todense()
    else:
        values = np.asarray(data).reshape(row_count, -1)
        if values.shape[1]:
            row_minima = np.minimum(row_minima, values.min(axis=1))
        row_sums = values.sum(axis=1)
    return row_minima, np.asarray(row_sums, dtype=np.float64)


def _label(index: pd.Index, position: int) -> object:
    """Return one index label with NumPy scalars converted to Python values for messages."""
    label = index[position]
    if isinstance(label, tuple):
        return tuple(value.item() if isinstance(value, np.generic) else value for value in label)
    return label.item() if isinstance(label, np.generic) else label


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
