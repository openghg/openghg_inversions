"""Coherent native-covariance products for retained bucket scaling states.

OpenGHG Inversions constructs a native observation operator ``H`` from
footprint times prior flux, so the native vector ``x`` and retained vector
``alpha = Pi x`` are multiplicative scalings. Their covariance algebra applies
to the centred perturbations ``x - m`` and ``alpha - Pi m``. The existing
:attr:`~openghg_inversions.basis.operators.BasisOperator.basis_matrix` supplies
the single-source bucket prolongation ``U_bucket`` (native by retained), or the
gathered spatial template expanded to source-native ``U_bucket`` here. It is
not ``Pi.T``.

This module's initial strategy preserves that established coefficient meaning.
For native covariance ``B`` it derives the compatible restriction

``Pi_U = (U.T B^-1 U)^-1 U.T B^-1``.

Consequently ``Pi_U U = I``, ``C_alpha = Pi_U B Pi_U.T`` and the
covariance-weighted prolongation
``U_* = B Pi_U.T C_alpha^-1`` is exactly ``U_bucket``.  The retained
observation operator is therefore ``H_alpha = H U_bucket``, matching the
current bucket sensitivity.  The centred residual
``r = (x - m) - U_bucket (alpha - Pi_U m)`` satisfies
``Cov(r, alpha) = 0``. Under a joint Gaussian model this also gives the affine
conditional mean ``m + U_bucket (alpha - Pi_U m)``.

The public strategy protocol deliberately keeps ``Pi`` separate from basis
geometry so a future strategy can instead define physical retained
functionals and derive its covariance-weighted prolongation.  This foundation
computes labelled ``C_alpha``, ``H B Pi.T``, and ``H B H.T`` (or its diagonal)
without constructing dense native ``B``. Operations eagerly materialize
``U``, ``H``, and the requested product matrices; dense ``H B H.T`` therefore
uses quadratic observation-space storage. Constructing unresolved covariance
and the coherent reduced likelihood belongs to OPE-18.

The components have deliberately separate roles:

1. :class:`~openghg_inversions.basis.operators.BasisOperator` supplies the
   bucket geometry ``U_bucket``.
2. :class:`~openghg_inversions.native_covariance.InvertibleNativeCovarianceAction`
   supplies labelled applications of ``B`` and ``B^-1``; its protocol is
   imported rather than redefined here.
3. :class:`RetainedProjectionStrategy` chooses a coherent ``(Pi, U_*)`` pair
   and returns it as the :class:`RetainedProjection` value object.
   :class:`PreserveBucketProlongation` is the initial concrete structural
   implementation of that strategy protocol.
4. :func:`project_native_covariance` combines those inputs into the frozen,
   serializable :class:`NativeCovarianceProducts` dataclass. Its contained
   xarray objects remain mutable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Hashable, Literal, Protocol, cast

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import xarray as xr

from openghg_inversions.basis.operators import BasisOperator
from openghg_inversions.native_covariance import InvertibleNativeCovarianceAction

MAX_DENSE_EIGEN_DIAGNOSTIC_SIZE = 512


@dataclass(frozen=True, slots=True)
class RetainedProjection:
    """A labelled restriction/prolongation pair selected by one strategy.

    Attributes:
        restriction: ``Pi`` with dimensions ``(state_dim, *native_dims)``;
            retained coefficients satisfy ``alpha = Pi x``.
        prolongation: Covariance-compatible ``U_*`` with dimensions
            ``(*native_dims, state_dim)`` and ``x_hat = U_* alpha`` in centred
            coordinates.
        strategy: Stable identifier for the scientific projection choice.
    """

    restriction: xr.DataArray
    prolongation: xr.DataArray
    strategy: str


class RetainedProjectionStrategy(Protocol):
    """Extension seam for choosing retained coefficients and their lift.

    Implementations must return a full-rank ``Pi`` and the covariance-natural
    ``U_* = B Pi.T (Pi B Pi.T)^-1``. This invariant is what makes the returned
    residual uncorrelated with the retained coefficients.
    """

    def projection(
        self,
        covariance: InvertibleNativeCovarianceAction,
        basis_prolongation: xr.DataArray,
        *,
        native_dims: tuple[str, ...],
        state_dim: str,
    ) -> RetainedProjection:
        """Return a compatible labelled restriction and prolongation.

        Args:
            covariance: Invertible labelled action for the native covariance
                ``B``.
            basis_prolongation: Candidate bucket prolongation with dimensions
                ``(*native_dims, state_dim)``.
            native_dims: Ordered dimensions of the native state.
            state_dim: Name of the retained-state dimension.

        Returns:
            A labelled :class:`RetainedProjection` whose restriction is
            full-rank and whose prolongation satisfies
            ``B Pi.T = U_* (Pi B Pi.T)``.

        Raises:
            ValueError: If the supplied arrays are invalid or a coherent,
                full-rank projection cannot be constructed.
        """
        ...


class _Digest(Protocol):
    """Minimal hashlib digest surface used by identity helpers."""

    def update(self, data: bytes, /) -> None:
        """Add bytes to the digest."""
        ...


@dataclass(frozen=True, slots=True)
class PreserveBucketProlongation:
    """Derive ``Pi_U`` so covariance-weighted prolongation equals ``U_bucket``.

    This class is a concrete structural implementation of
    :class:`RetainedProjectionStrategy`. :class:`RetainedProjection` is the
    value object returned by :meth:`projection`, not the protocol being
    implemented.

    Attributes:
        name: Stable strategy identifier stored with projected products.

    ``B`` must be positive definite and ``U_bucket`` must have full column rank.
    """

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
            covariance: Invertible labelled action for the native covariance ``B``.
            basis_prolongation: Current bucket prolongation ``U_bucket``.
            native_dims: Ordered native dimensions.
            state_dim: Retained-state dimension.

        Returns:
            A :class:`RetainedProjection` containing the labelled pair
            ``(Pi_U, U_bucket)``.

        Raises:
            ValueError: If the prolongation is invalid or has redundant columns.
        """
        prolongation = _validated_prolongation(
            basis_prolongation, native_dims=native_dims, state_dim=state_dim
        )
        state_column_dim = _matrix_column_dim(state_dim, prolongation.dims)
        precision_prolongation = _to_plain_column_axis(
            covariance.solve(prolongation),
            row_dim=state_dim,
            column_dim=state_column_dim,
            leading_dims=native_dims,
        )
        gram = xr.dot(
            prolongation,
            precision_prolongation,
            dim=list(native_dims),
        ).transpose(state_dim, state_column_dim)
        gram_values = np.asarray(gram.values, dtype=np.float64)
        _validate_symmetric(gram_values, "U.T B^-1 U")
        eigenvalues = np.linalg.eigvalsh((gram_values + gram_values.T) * 0.5)
        largest_eigenvalue = float(eigenvalues[-1]) if eigenvalues.size else 0.0
        if (
            not eigenvalues.size
            or largest_eigenvalue <= 0.0
            or float(eigenvalues[0]) <= 1e-12 * largest_eigenvalue
        ):
            raise ValueError(
                "The bucket prolongation contains redundant or numerically unsupported retained states; "
                "U.T B^-1 U must be positive definite and full rank"
            )
        try:
            factor = cho_factor(gram_values, lower=True, check_finite=True)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "The bucket prolongation contains redundant or unsupported retained states; "
                "U.T B^-1 U must be positive definite"
            ) from exc
        covariance_values = cho_solve(factor, np.eye(gram_values.shape[0]), check_finite=False)
        covariance_coords: dict[str, xr.DataArray] = {
            state_dim: prolongation.coords[state_dim],
        }
        for name, coordinate in prolongation.coords.items():
            if name != state_dim and tuple(coordinate.dims) == (state_dim,):
                covariance_coords[str(name)] = coordinate
        covariance_coords.update(
            _column_coordinates(prolongation, row_dim=state_dim, column_dim=state_column_dim)
        )
        state_covariance = xr.DataArray(
            covariance_values,
            dims=(state_dim, state_column_dim),
            coords=covariance_coords,
        )
        restriction = xr.dot(
            state_covariance,
            precision_prolongation,
            dim=state_column_dim,
        ).transpose(state_dim, *native_dims)
        restriction = restriction.rename("restriction").assign_attrs(
            mathematical_name="Pi_U",
            definition="(U.T B^-1 U)^-1 U.T B^-1",
            strategy=self.name,
        )
        prolongation = prolongation.rename("prolongation").assign_attrs(
            mathematical_name="U_bucket",
            strategy=self.name,
        )
        return RetainedProjection(
            restriction=restriction,
            prolongation=prolongation,
            strategy=self.name,
        )


@dataclass(frozen=True, slots=True)
class NativeCovarianceProducts:
    """Labelled product blocks induced by one coherent ``(B, H, Pi, U)`` set.

    Attributes:
        restriction: Retained restriction ``Pi`` with dimensions retained by native.
        prolongation: Covariance-compatible ``U_*`` with dimensions native by retained.
        state_covariance: ``C_alpha = Pi B Pi.T``.
        effective_observation_operator: ``H_alpha = H U_*``.
        observation_state_cross_covariance: ``H B Pi.T``.
        native_observation_covariance: Dense ``H B H.T`` with a collision-safe
            observation column dimension, or its labelled observation diagonal.
        strategy: Stable retained-projection strategy name.
        content_identity: SHA-256 identity binding inputs and products.
        covariance_configuration: Reproducible serialized kernel/source configuration.
        basis_provenance: Stable basis implementation and dimension metadata.
    """

    restriction: xr.DataArray
    prolongation: xr.DataArray
    state_covariance: xr.DataArray
    effective_observation_operator: xr.DataArray
    observation_state_cross_covariance: xr.DataArray
    native_observation_covariance: xr.DataArray
    strategy: str
    content_identity: str
    covariance_configuration: xr.Dataset | None = None
    basis_provenance: dict[str, str] = field(default_factory=dict)

    schema = "openghg_inversions.native_covariance_products"
    schema_version = 1

    def to_dataset(self) -> xr.Dataset:
        """Serialize all labelled product blocks and their shared identity.

        Returns:
            A dataset containing every product array plus schema, strategy,
            identity, and basis-provenance metadata. Native covariance
            configuration is intentionally excluded because it occupies a
            separate child node in :meth:`to_datatree`.
        """
        return xr.Dataset(
            {
                "restriction": self.restriction,
                "prolongation": self.prolongation,
                "state_covariance": self.state_covariance,
                "effective_observation_operator": self.effective_observation_operator,
                "observation_state_cross_covariance": self.observation_state_cross_covariance,
                "native_observation_covariance": self.native_observation_covariance,
            },
            attrs={
                "schema": self.schema,
                "schema_version": self.schema_version,
                "strategy": self.strategy,
                "content_identity": self.content_identity,
                "basis_provenance": json.dumps(self.basis_provenance, sort_keys=True),
            },
        )

    def to_datatree(self) -> xr.DataTree:
        """Serialize products and reproducible covariance configuration.

        Returns:
            A tree whose root contains :meth:`to_dataset` output and whose
            optional ``covariance_configuration`` child preserves the native
            covariance constructor data.
        """
        from openghg_inversions.serialization import reset_serialisation_multiindexes

        tree = xr.DataTree(reset_serialisation_multiindexes(self.to_dataset()))
        if self.covariance_configuration is not None:
            tree["covariance_configuration"] = xr.DataTree(
                _encode_configuration_dataset(self.covariance_configuration)
            )
        return tree

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> NativeCovarianceProducts:
        """Restore and validate :meth:`to_dataset` output.

        Args:
            dataset: Versioned product-block dataset.

        Returns:
            Restored frozen product dataclass. Its contained arrays remain
            mutable.

        Raises:
            ValueError: If the schema version, required variables, shared
                content identity, projection strategy, or JSON-encoded basis
                provenance is invalid.
        """
        if dataset.attrs.get("schema") != cls.schema:
            raise ValueError(f"Expected product schema {cls.schema!r}")
        if dataset.attrs.get("schema_version") != cls.schema_version:
            raise ValueError(f"Unsupported product schema version {dataset.attrs.get('schema_version')!r}")
        required = (
            "restriction",
            "prolongation",
            "state_covariance",
            "effective_observation_operator",
            "observation_state_cross_covariance",
            "native_observation_covariance",
        )
        missing = [name for name in required if name not in dataset]
        if missing:
            raise ValueError(f"Serialized covariance products are missing variables {missing}")
        content_identity = str(dataset.attrs.get("content_identity", ""))
        strategy = str(dataset.attrs.get("strategy", ""))
        if not content_identity or not strategy:
            raise ValueError("Serialized covariance products are missing identity or strategy metadata")
        for name in required:
            if dataset[name].attrs.get("content_identity") != content_identity:
                raise ValueError(f"Serialized product {name!r} does not share the root content identity")
            if dataset[name].attrs.get("projection_strategy") != strategy:
                raise ValueError(f"Serialized product {name!r} does not share the root projection strategy")
        return cls(
            restriction=dataset["restriction"],
            prolongation=dataset["prolongation"],
            state_covariance=dataset["state_covariance"],
            effective_observation_operator=dataset["effective_observation_operator"],
            observation_state_cross_covariance=dataset["observation_state_cross_covariance"],
            native_observation_covariance=dataset["native_observation_covariance"],
            strategy=strategy,
            content_identity=content_identity,
            basis_provenance=json.loads(str(dataset.attrs.get("basis_provenance", "{}"))),
        )

    @classmethod
    def from_datatree(cls, tree: xr.DataTree) -> NativeCovarianceProducts:
        """Restore products and covariance configuration from a data tree.

        Args:
            tree: Tree produced by :meth:`to_datatree`.

        Returns:
            Restored frozen product dataclass, including covariance
            configuration when the child node is present. Its contained
            arrays and mappings remain mutable.

        Raises:
            ValueError: If the root product schema, shared identity, strategy,
                MultiIndex metadata, or covariance configuration is invalid.
        """
        from openghg_inversions.serialization import (
            MULTIINDEX_DIMS_ATTR,
            restore_serialisation_multiindexes,
        )

        root_dataset = tree.to_dataset(inherit=False)
        if MULTIINDEX_DIMS_ATTR in root_dataset.attrs:
            root_dataset = restore_serialisation_multiindexes(root_dataset, strict=True)
        products = cls.from_dataset(root_dataset)
        if "covariance_configuration" not in tree:
            return products
        return replace(
            products,
            covariance_configuration=_decode_configuration_dataset(
                cast(xr.DataTree, tree["covariance_configuration"]).to_dataset(inherit=False)
            ),
        )


def project_native_covariance(
    *,
    covariance: InvertibleNativeCovarianceAction,
    basis_operator: BasisOperator,
    native_sensitivity: xr.DataArray,
    observation_dim: str,
    observation_covariance: Literal["dense", "diagonal"] = "dense",
    observation_batch_size: int = 64,
    strategy: RetainedProjectionStrategy | None = None,
) -> NativeCovarianceProducts:
    """Compute coherent labelled native-covariance product blocks.

    Args:
        covariance: Labelled native covariance action with a compatible solve.
        basis_operator: Basis whose matrix supplies the initial bucket prolongation.
        native_sensitivity: Native ``H`` containing footprint times prior flux,
            with dimensions ``(observation_dim, *native_dims)`` in any order.
        observation_dim: Name of the observation dimension in ``native_sensitivity``.
        observation_covariance: Return dense ``H B H.T`` or only its diagonal.
        observation_batch_size: Positive number of observation right-hand sides
            applied to ``B`` in each eager batch. This is independent of Dask
            array chunking: each batch produces a result block and the blocks
            are concatenated. Dense ``H B H.T`` is still fully materialized.
        strategy: Retained projection choice. The default preserves bucket scalings.

    Returns:
        Frozen product dataclass containing labelled blocks tied by one content
        identity. Its contained arrays remain mutable. Inputs and products are
        eagerly materialized; dense observation covariance has
        quadratic observation-space storage. Full PSD eigendiagnostics are skipped
        above 512 rows to avoid adding cubic observation-space work. Input attrs/units are not propagated:
        product attrs are replaced by mathematical diagnostics because this API
        does not implement unit algebra.

    Raises:
        ValueError: If labels, dimensions, values, or options are invalid.
    """
    native_dims = tuple(covariance.native_dims)
    state_dim = basis_operator.meta.state_dim
    if len({*native_dims, state_dim, observation_dim}) != len(native_dims) + 2:
        raise ValueError("Native, retained-state, and observation dimension names must be distinct")
    basis_grid_dims = tuple(basis_operator.meta.grid_dims)
    if basis_grid_dims != native_dims and not (
        len(native_dims) == len(basis_grid_dims) + 1 and native_dims[1:] == basis_grid_dims
    ):
        raise ValueError(
            "Basis grid dimensions must equal the covariance spatial dimensions; "
            f"{basis_operator.meta.grid_dims!r} is incompatible with {native_dims!r}"
        )
    sensitivity = _validated_sensitivity(
        native_sensitivity,
        native_dims=native_dims,
        observation_dim=observation_dim,
        covariance=covariance,
    )
    batch_size = int(observation_batch_size)
    if batch_size <= 0:
        raise ValueError("observation_batch_size must be positive")
    if observation_covariance not in {"dense", "diagonal"}:
        raise ValueError("observation_covariance must be 'dense' or 'diagonal'")

    basis_prolongation = _native_basis_prolongation(
        basis_operator,
        sensitivity,
        native_dims=native_dims,
        state_dim=state_dim,
    )
    projection_strategy = strategy or PreserveBucketProlongation()
    projection = projection_strategy.projection(
        covariance,
        basis_prolongation,
        native_dims=native_dims,
        state_dim=state_dim,
    )
    state_column_dim = _matrix_column_dim(state_dim, projection.restriction.dims)
    restriction_transpose = _to_plain_column_axis(
        projection.restriction,
        row_dim=state_dim,
        column_dim=state_column_dim,
        leading_dims=native_dims,
    )
    b_restriction_transpose = covariance.apply(restriction_transpose)

    state_covariance = xr.dot(
        projection.restriction,
        b_restriction_transpose,
        dim=list(native_dims),
    ).transpose(state_dim, state_column_dim)
    _validate_positive_definite(np.asarray(state_covariance.values), "C_alpha")
    state_covariance = _with_matrix_diagnostics(
        state_covariance.rename("state_covariance"), mathematical_name="C_alpha"
    )
    _validate_projection_invariant(
        projection,
        state_covariance,
        b_restriction_transpose,
        native_dims=native_dims,
        state_dim=state_dim,
        state_column_dim=state_column_dim,
    )

    effective_operator = xr.dot(
        sensitivity,
        projection.prolongation,
        dim=list(native_dims),
    ).transpose(observation_dim, state_dim)
    effective_operator = effective_operator.rename("effective_observation_operator").assign_attrs(
        mathematical_name="H_alpha",
        definition="H U_*",
    )
    cross_covariance = xr.dot(
        sensitivity,
        b_restriction_transpose,
        dim=list(native_dims),
    ).transpose(observation_dim, state_column_dim)
    cross_covariance = cross_covariance.rename("observation_state_cross_covariance").assign_attrs(
        mathematical_name="H B Pi.T"
    )

    native_observation_covariance = _observation_covariance(
        covariance,
        sensitivity,
        native_dims=native_dims,
        observation_dim=observation_dim,
        output=observation_covariance,
        batch_size=batch_size,
    )
    content_identity = _content_identity(
        covariance,
        projection.restriction,
        projection.prolongation,
        sensitivity,
        strategy=projection.strategy,
        observation_covariance=observation_covariance,
    )
    covariance_to_dataset = getattr(covariance, "to_dataset", None)
    covariance_configuration = (
        cast(xr.Dataset, covariance_to_dataset()).copy(deep=True) if callable(covariance_to_dataset) else None
    )
    basis_provenance = {
        "operator_type": f"{type(basis_operator).__module__}.{type(basis_operator).__qualname__}",
        "operator_kind": str(getattr(basis_operator, "kind", "unknown")),
        "grid_dims": repr(tuple(basis_operator.meta.grid_dims)),
        "state_dim": state_dim,
    }
    shared_attrs = {
        "projection_strategy": projection.strategy,
        "content_identity": content_identity,
    }
    arrays = (
        projection.restriction,
        projection.prolongation,
        state_covariance,
        effective_operator,
        cross_covariance,
        native_observation_covariance,
    )
    for array in arrays:
        array.attrs.update(shared_attrs)

    return NativeCovarianceProducts(
        restriction=projection.restriction,
        prolongation=projection.prolongation,
        state_covariance=state_covariance,
        effective_observation_operator=effective_operator,
        observation_state_cross_covariance=cross_covariance,
        native_observation_covariance=native_observation_covariance,
        strategy=projection.strategy,
        content_identity=content_identity,
        covariance_configuration=covariance_configuration,
        basis_provenance=basis_provenance,
    )


def _observation_covariance(
    covariance: InvertibleNativeCovarianceAction,
    sensitivity: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    observation_dim: str,
    output: Literal["dense", "diagonal"],
    batch_size: int,
) -> xr.DataArray:
    """Compute ``H B H.T`` in eager observation right-hand-side batches.

    Batching limits the size of the temporary native-space ``B H.T`` block;
    it is not Dask chunking. Result blocks are concatenated along a distinct
    observation-column dimension. Dense output therefore still materializes
    the complete quadratic observation covariance, while diagonal output
    materializes only its labelled diagonal.

    Args:
        covariance: Labelled action implementing multiplication by ``B``.
        sensitivity: Eager native sensitivity ``H`` with observation followed
            by native dimensions.
        native_dims: Ordered native dimensions contracted in the products.
        observation_dim: Name of the observation row dimension.
        output: Whether to return the dense covariance or only its diagonal.
        batch_size: Positive number of observation columns processed per
            covariance application.

    Returns:
        Labelled dense ``H B H.T`` or ``diag(H B H.T)``.

    Raises:
        ValueError: If covariance labels are incompatible or a dense result
            fails the symmetry diagnostic.
    """
    observation_column_dim = _matrix_column_dim(observation_dim, sensitivity.dims)
    blocks: list[xr.DataArray] = []
    for start in range(0, sensitivity.sizes[observation_dim], batch_size):
        stop = min(start + batch_size, sensitivity.sizes[observation_dim])
        rhs = _to_plain_column_axis(
            sensitivity.isel({observation_dim: slice(start, stop)}),
            row_dim=observation_dim,
            column_dim=observation_column_dim,
            leading_dims=native_dims,
        )
        b_rhs = covariance.apply(rhs)
        if output == "dense":
            block = xr.dot(sensitivity, b_rhs, dim=list(native_dims)).transpose(
                observation_dim, observation_column_dim
            )
        else:
            matching = _to_plain_column_axis(
                sensitivity.isel({observation_dim: slice(start, stop)}),
                row_dim=observation_dim,
                column_dim=observation_column_dim,
                leading_dims=native_dims,
            )
            block = (matching * b_rhs).sum(dim=list(native_dims))
        blocks.append(block)
    combined = xr.concat(blocks, dim=observation_column_dim)
    if output == "dense":
        combined = _with_matrix_diagnostics(
            combined.rename("native_observation_covariance"), mathematical_name="H B H.T"
        )
    else:
        combined = combined.rename({observation_column_dim: observation_dim})
        combined = combined.assign_coords({observation_dim: sensitivity.coords[observation_dim]})
        combined = combined.rename("native_observation_covariance").assign_attrs(
            mathematical_name="diag(H B H.T)",
            minimum_diagonal=float(combined.min().item()),
            diagonal_nonnegative=bool((combined >= -1e-10).all().item()),
            diagnostic_tolerance=1e-10,
        )
    return combined


def _validate_projection_invariant(
    projection: RetainedProjection,
    state_covariance: xr.DataArray,
    b_restriction_transpose: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    state_dim: str,
    state_column_dim: str,
) -> None:
    """Require a strategy to return its covariance-natural prolongation.

    Args:
        projection: Restriction/prolongation pair returned by the strategy.
        state_covariance: Labelled ``C_alpha = Pi B Pi.T``.
        b_restriction_transpose: Labelled ``B Pi.T``.
        native_dims: Ordered native dimensions.
        state_dim: Retained-state row dimension.
        state_column_dim: Collision-safe retained-state column dimension.

    Raises:
        ValueError: If dimensions, state labels, or values are invalid, or if
            ``B Pi.T`` and ``U_* C_alpha`` disagree beyond tolerance.
    """
    prolongation = _validated_prolongation(
        projection.prolongation,
        native_dims=native_dims,
        state_dim=state_dim,
    )
    restriction = projection.restriction
    expected_dims = {state_dim, *native_dims}
    if set(restriction.dims) != expected_dims or len(restriction.dims) != len(expected_dims):
        raise ValueError("Projection strategy restriction has invalid labelled dimensions")
    if not np.all(np.isfinite(np.asarray(restriction.values))):
        raise ValueError("Projection strategy restriction must contain only finite values")
    if not np.array_equal(restriction.coords[state_dim].values, prolongation.coords[state_dim].values):
        raise ValueError("Projection strategy restriction/prolongation state labels differ")
    u_c = xr.dot(prolongation, state_covariance, dim=state_dim).transpose(*native_dims, state_column_dim)
    left = np.asarray(b_restriction_transpose.values, dtype=np.float64)
    right = np.asarray(u_c.values, dtype=np.float64)
    scale = max(
        float(np.max(np.abs(left))) if left.size else 0.0,
        float(np.max(np.abs(right))) if right.size else 0.0,
        np.finfo(np.float64).tiny,
    )
    absolute_error = float(np.max(np.abs(left - right))) if left.size else 0.0
    if absolute_error > 1e-9 * scale + 10.0 * np.finfo(np.float64).tiny:
        raise ValueError("Projection strategy is incoherent: expected B Pi.T = U_* C_alpha")


def _native_basis_prolongation(
    basis_operator: BasisOperator,
    sensitivity: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    state_dim: str,
) -> xr.DataArray:
    """Return ``U_bucket`` on the full native space, including source blocks.

    A single-source basis already spans every native dimension. For a gathered
    multisource basis, the operator carries source identity on the ragged state
    coordinate rather than as a matrix dimension. This helper expands it onto
    the covariance action's explicit leading source dimension, setting all
    cross-source state columns to zero while retaining canonical state order.

    Args:
        basis_operator: Spatial basis supplying the bucket matrix and metadata.
        sensitivity: Native sensitivity whose leading source coordinate defines
            source ordering for multisource inputs.
        native_dims: Ordered native covariance dimensions.
        state_dim: Retained-state dimension of the basis matrix.

    Returns:
        Eager ``U_bucket`` spanning ``(*native_dims, state_dim)``.

    Raises:
        ValueError: If a multisource basis lacks source labels or names a source
            absent from the native sensitivity.
    """
    basis_grid_dims = tuple(basis_operator.meta.grid_dims)
    basis_matrix = _densify(basis_operator.basis_matrix)
    if native_dims == basis_grid_dims:
        return basis_matrix

    native_source_dim = native_dims[0]
    basis_source_dim = getattr(basis_operator, "source_dim", None)
    if not isinstance(basis_source_dim, str) or basis_source_dim not in basis_matrix.coords:
        raise ValueError(
            "A covariance with a leading native source dimension requires a gathered "
            "multisource basis carrying source labels on its state coordinate"
        )
    state_sources = np.asarray(basis_matrix.coords[basis_source_dim].values)
    native_sources = np.asarray(sensitivity.coords[native_source_dim].values)
    missing = [label for label in state_sources if label not in set(native_sources.tolist())]
    if missing:
        raise ValueError(f"Basis state sources are absent from native sensitivity: {missing!r}")
    base = basis_matrix.transpose(*basis_grid_dims, state_dim)
    values = np.asarray(base.values)
    source_mask = native_sources[:, np.newaxis] == state_sources[np.newaxis, :]
    expanded = values[np.newaxis, ...] * source_mask.reshape(
        (native_sources.size, *([1] * len(basis_grid_dims)), state_sources.size)
    )
    coords: dict[str, xr.DataArray] = {
        native_source_dim: sensitivity.coords[native_source_dim],
        **{dim: base.coords[dim] for dim in basis_grid_dims},
        state_dim: base.coords[state_dim],
    }
    for name, coordinate in base.coords.items():
        if name not in coords and set(coordinate.dims).issubset({state_dim}):
            coords[str(name)] = coordinate
    return xr.DataArray(
        expanded,
        dims=(*native_dims, state_dim),
        coords=coords,
        name="prolongation",
        attrs={**basis_matrix.attrs, "mathematical_name": "U_bucket"},
    )


def _to_plain_column_axis(
    array: xr.DataArray,
    *,
    row_dim: str,
    column_dim: str,
    leading_dims: tuple[str, ...],
) -> xr.DataArray:
    """Replace a rich row axis by a collision-safe plain column axis.

    Args:
        array: Labelled array containing the row and leading dimensions.
        row_dim: Existing row dimension, possibly backed by a MultiIndex.
        column_dim: New collision-safe plain dimension name.
        leading_dims: Dimensions to retain before the new column dimension.

    Returns:
        An array sharing the input data and attributes, with row coordinates
        copied onto the new plain column axis.
    """
    ordered = array.transpose(*leading_dims, row_dim)
    coords = {dim: ordered.coords[dim] for dim in leading_dims}
    coords.update(_column_coordinates(ordered, row_dim=row_dim, column_dim=column_dim))
    return xr.DataArray(
        ordered.data,
        dims=(*leading_dims, column_dim),
        coords=coords,
        name=ordered.name,
        attrs=ordered.attrs,
    )


def _validated_prolongation(
    prolongation: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    state_dim: str,
) -> xr.DataArray:
    """Materialize and validate a labelled native-by-retained prolongation.

    Args:
        prolongation: Candidate ``U_*`` or ``U_bucket`` array.
        native_dims: Ordered native dimensions required on the array.
        state_dim: Required retained-state dimension.

    Returns:
        A finite, eager array ordered as ``(*native_dims, state_dim)``.

    Raises:
        ValueError: If dimensions are missing or duplicated, values are not
            finite, or a retained-state column is empty.
    """
    expected_dims = set((*native_dims, state_dim))
    if set(prolongation.dims) != expected_dims or len(prolongation.dims) != len(expected_dims):
        raise ValueError(
            "prolongation must have exactly the native grid and retained-state dimensions; "
            f"got {prolongation.dims!r}"
        )
    result = _densify(prolongation).transpose(*native_dims, state_dim)
    values = np.asarray(result.values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("prolongation must contain only finite values")
    if result.sizes[state_dim] == 0 or np.any(np.all(values == 0.0, axis=tuple(range(len(native_dims))))):
        raise ValueError("prolongation contains an empty retained state")
    return result


def _validated_sensitivity(
    sensitivity: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    observation_dim: str,
    covariance: InvertibleNativeCovarianceAction,
) -> xr.DataArray:
    """Materialize and validate the labelled native sensitivity ``H``.

    Args:
        sensitivity: Candidate observation-by-native sensitivity array.
        native_dims: Ordered native dimensions required on the array.
        observation_dim: Required observation dimension.
        covariance: Covariance action used to validate native grid labels.

    Returns:
        A finite, eager array ordered as ``(observation_dim, *native_dims)``.

    Raises:
        ValueError: If dimensions or observation labels are invalid, the array
            is empty or non-finite, or covariance grid labels do not match.
    """
    expected_dims = set((*native_dims, observation_dim))
    if set(sensitivity.dims) != expected_dims or len(sensitivity.dims) != len(expected_dims):
        raise ValueError(
            "native_sensitivity must have exactly one observation dimension and the native grid "
            f"dimensions; got {sensitivity.dims!r}"
        )
    if observation_dim not in sensitivity.coords:
        raise ValueError(f"native_sensitivity is missing observation coordinate {observation_dim!r}")
    result = _densify(sensitivity).transpose(observation_dim, *native_dims)
    # Applying B to one labelled column performs the covariance-grid label validation.
    if result.sizes[observation_dim] == 0:
        raise ValueError("native_sensitivity must contain at least one observation")
    covariance.apply(result.isel({observation_dim: slice(0, 1)}))
    values = np.asarray(result.values)
    if not np.all(np.isfinite(values)):
        raise ValueError("native_sensitivity must contain only finite values")
    return result


def _densify(array: xr.DataArray) -> xr.DataArray:
    """Eagerly materialize an array and convert sparse storage to NumPy."""
    materialized = array.compute()
    data = materialized.data
    if hasattr(data, "todense"):
        data = data.todense()
    return materialized.copy(data=np.asarray(data))


def _matrix_column_dim(primary_dim: str, occupied_dims: tuple[Hashable, ...]) -> str:
    """Return an unoccupied column-dimension name derived from a row name."""
    candidate = f"{primary_dim}_cov"
    index = 2
    while candidate in occupied_dims:
        candidate = f"{primary_dim}_cov_{index}"
        index += 1
    return candidate


def _column_coordinate(coordinate: xr.DataArray, column_dim: str) -> xr.DataArray:
    """Copy row labels onto a plain column coordinate.

    Tuple-valued labels are encoded as compact JSON strings so xarray does not
    reinterpret them as a two-dimensional coordinate.

    Args:
        coordinate: One-dimensional row coordinate to copy.
        column_dim: Destination column dimension.

    Returns:
        A plain one-dimensional column coordinate preserving attributes.
    """
    source_values = list(coordinate.values)
    if any(isinstance(value, tuple) for value in source_values):
        values = np.asarray(
            [json.dumps(list(value), separators=(",", ":")) for value in source_values],
            dtype=str,
        )
    else:
        values = np.asarray(source_values)
    return xr.DataArray(values, dims=column_dim, attrs=coordinate.attrs)


def _column_coordinates(
    array: xr.DataArray,
    *,
    row_dim: str,
    column_dim: str,
) -> dict[str, xr.DataArray]:
    """Build a plain column index and copies of row-axis metadata.

    Args:
        array: Source array containing the row coordinate and auxiliary labels.
        row_dim: Row dimension whose coordinates are copied.
        column_dim: Collision-safe column dimension receiving the copies.

    Returns:
        Coordinates for the new column axis. Auxiliary coordinate names gain a
        ``_cov`` suffix, with numeric suffixes added to avoid collisions.
    """
    result = {column_dim: _column_coordinate(array.coords[row_dim], column_dim)}
    for name, coordinate in array.coords.items():
        if name == row_dim or tuple(coordinate.dims) != (row_dim,):
            continue
        column_name = f"{name}_cov"
        suffix = 2
        while column_name in array.coords or column_name in result:
            column_name = f"{name}_cov_{suffix}"
            suffix += 1
        result[column_name] = xr.DataArray(
            np.asarray(coordinate.values),
            dims=column_dim,
            attrs=coordinate.attrs,
        )
    return result


def _validate_symmetric(values: np.ndarray, name: str, *, tolerance: float = 1e-10) -> None:
    """Validate matrix symmetry relative to its largest absolute value.

    Args:
        values: Matrix values to validate.
        name: Mathematical name used in an error message.
        tolerance: Relative symmetry tolerance, with unit absolute scale floor.

    Raises:
        ValueError: If the maximum asymmetry exceeds the scaled tolerance.
    """
    asymmetry = float(np.max(np.abs(values - values.T))) if values.size else 0.0
    scale = max(1.0, float(np.max(np.abs(values))) if values.size else 1.0)
    if asymmetry > tolerance * scale:
        raise ValueError(f"{name} is not symmetric within tolerance {tolerance:g}")


def _validate_positive_definite(values: np.ndarray, name: str) -> None:
    """Reject singular or numerically redundant covariance coordinates.

    Args:
        values: Symmetric covariance matrix to validate.
        name: Mathematical name used in an error message.

    Raises:
        ValueError: If the matrix is asymmetric, empty, non-positive, or has a
            smallest eigenvalue no larger than ``1e-12`` times the largest.
    """
    _validate_symmetric(values, name)
    eigenvalues = np.linalg.eigvalsh((values + values.T) * 0.5)
    largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    if not eigenvalues.size or largest <= 0.0 or float(eigenvalues[0]) <= 1e-12 * largest:
        raise ValueError(f"{name} must be positive definite and full rank")


def _with_matrix_diagnostics(array: xr.DataArray, *, mathematical_name: str) -> xr.DataArray:
    """Attach symmetry and bounded-cost PSD diagnostics to a dense matrix.

    Args:
        array: Dense square labelled matrix.
        mathematical_name: Mathematical label stored in the result attributes.

    Returns:
        The array with diagnostic attributes. Full eigenvalue diagnostics are
        skipped when the matrix exceeds :data:`MAX_DENSE_EIGEN_DIAGNOSTIC_SIZE`.

    Raises:
        ValueError: If the matrix is not symmetric within tolerance.
    """
    values = np.asarray(array.values, dtype=np.float64)
    _validate_symmetric(values, mathematical_name)
    attrs: dict[str, object] = {
        "mathematical_name": mathematical_name,
        "symmetry_absolute_error": float(np.max(np.abs(values - values.T))) if values.size else 0.0,
        "diagnostic_tolerance": 1e-10,
    }
    if values.shape[0] <= MAX_DENSE_EIGEN_DIAGNOSTIC_SIZE:
        eigenvalues = np.linalg.eigvalsh((values + values.T) * 0.5)
        attrs.update(
            minimum_eigenvalue=float(eigenvalues.min()) if eigenvalues.size else 0.0,
            psd_diagnostic="full_eigendecomposition",
        )
    else:
        attrs.update(
            minimum_eigenvalue=np.nan,
            psd_diagnostic=(f"skipped_full_eigendecomposition_above_{MAX_DENSE_EIGEN_DIAGNOSTIC_SIZE}"),
        )
    return array.assign_attrs(attrs)


def _encode_configuration_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Namespace configuration dimensions before DataTree persistence.

    Args:
        dataset: Native covariance configuration to encode.

    Returns:
        A copy whose dimensions have a ``covariance_configuration__`` prefix.
        The reversible original-to-encoded mapping is stored in the
        ``openghg_inversions:configuration_dims`` attribute so parent DataTree
        dimensions cannot absorb the child dimensions.
    """
    rename = {str(dim): f"covariance_configuration__{dim}" for dim in dataset.dims}
    encoded = dataset.rename(rename).copy()
    encoded.attrs = dict(encoded.attrs)
    encoded.attrs["openghg_inversions:configuration_dims"] = json.dumps(rename, sort_keys=True)
    return encoded


def _decode_configuration_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Restore namespaced configuration dimensions after persistence.

    Args:
        dataset: Encoded covariance configuration child dataset.

    Returns:
        A dataset with original dimension names and encoding metadata removed.

    Raises:
        ValueError: If the dimension mapping metadata is absent or invalid.
    """
    encoded_dims = dataset.attrs.get("openghg_inversions:configuration_dims")
    if not isinstance(encoded_dims, str):
        raise ValueError("Serialized covariance configuration is missing dimension metadata")
    rename = json.loads(encoded_dims)
    if not isinstance(rename, dict):
        raise ValueError("Serialized covariance configuration dimension metadata is invalid")
    restored = dataset.rename({str(encoded): str(original) for original, encoded in rename.items()})
    restored.attrs = dict(restored.attrs)
    del restored.attrs["openghg_inversions:configuration_dims"]
    return restored


def _content_identity(
    covariance: InvertibleNativeCovarianceAction,
    *arrays: xr.DataArray,
    strategy: str,
    observation_covariance: str,
) -> str:
    """Hash scientific inputs and configuration into a stable identity.

    Execution batching is deliberately excluded, because changing
    ``observation_batch_size`` does not change the requested scientific
    product. Computed product values are also excluded to avoid binding the
    identity to batch-dependent floating-point roundoff.

    Args:
        covariance: Native covariance action, preferably with serializable
            constructor configuration.
        *arrays: Scientific input arrays to bind to the identity.
        strategy: Stable retained-projection strategy name.
        observation_covariance: Requested dense or diagonal output form.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    digest.update(strategy.encode("utf-8"))
    digest.update(observation_covariance.encode("utf-8"))
    to_dataset = getattr(covariance, "to_dataset", None)
    if callable(to_dataset):
        covariance_dataset = cast(xr.Dataset, to_dataset())
        digest.update(repr(sorted(covariance_dataset.attrs.items())).encode("utf-8"))
        for name in sorted(str(name) for name in covariance_dataset.coords):
            _update_array_digest(digest, covariance_dataset.coords[name])
        for name in sorted(str(name) for name in covariance_dataset.data_vars):
            _update_array_digest(digest, covariance_dataset[name])
    else:
        covariance_type = f"{type(covariance).__module__}.{type(covariance).__qualname__}"
        digest.update(covariance_type.encode("utf-8"))
        digest.update(repr(covariance).encode("utf-8"))
    for array in arrays:
        _update_array_digest(digest, array)
    return digest.hexdigest()


def _update_array_digest(digest: _Digest, array: xr.DataArray) -> None:
    """Update a content digest with array data, dimensions, and coordinates."""
    digest.update(str(array.name).encode("utf-8"))
    digest.update(repr(tuple(array.dims)).encode("utf-8"))
    values = np.ascontiguousarray(array.values)
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(repr(values.shape).encode("ascii"))
    if values.dtype.hasobject or values.dtype.kind in {"U", "S"}:
        digest.update(repr(values.tolist()).encode("utf-8"))
    else:
        digest.update(values.view(np.uint8).tobytes())
    for name in sorted(str(name) for name in array.coords):
        coordinate = array.coords[name]
        labels = np.asarray(coordinate.values)
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(coordinate.dims)).encode("utf-8"))
        if labels.dtype.hasobject or labels.dtype.kind in {"U", "S"}:
            digest.update(repr(labels.tolist()).encode("utf-8"))
        else:
            digest.update(np.ascontiguousarray(labels).view(np.uint8).tobytes())
