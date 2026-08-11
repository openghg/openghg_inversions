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

Projected artifacts carry a source identity derived from covariance
configuration, projection strategy, and labelled ``Pi``, ``U``, and ``H``
inputs, plus a view identity derived from that source identity and the requested
dense/diagonal representation. These identities bind artifact metadata; they
are not checksums of computed output values. Observation batching fills one
preallocated eager result but does not reduce final dense quadratic storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any, Hashable, Literal, Protocol, cast

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import xarray as xr

from openghg_inversions.basis.operators import BasisOperator
from openghg_inversions.native_covariance import InvertibleNativeCovarianceAction

MAX_DENSE_EIGEN_DIAGNOSTIC_SIZE = 512
_PRODUCT_COORDINATE_NAMESPACES_ATTR = "openghg_inversions:product_coordinate_namespaces"
_PRODUCT_MULTIINDEX_DIMS_ATTR = "openghg_inversions:product_multiindex_dims"


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
    residual uncorrelated with the retained coefficients. Both arrays must use
    native labels exactly matching the covariance grid in the same order, and
    their retained-state labels must agree.
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
        source_content_identity: SHA-256 identity of the projection strategy,
            covariance configuration or fallback representation, and labelled
            ``Pi/U/H`` values, dimensions, and coordinates. It is independent
            of the dense or diagonal numerical view.
        view_identity: SHA-256 identity derived from the source identity and
            requested observation-covariance view.
        observation_covariance_view: Numerical view represented by
            ``native_observation_covariance``.
        covariance_configuration_digest: Digest declaring the covariance
            configuration required to reproduce the source artifact. The
            configuration content itself is stored separately in a DataTree
            child.
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
    source_content_identity: str
    view_identity: str
    observation_covariance_view: Literal["dense", "diagonal"]
    covariance_configuration_digest: str = ""
    covariance_configuration: xr.Dataset | None = None
    basis_provenance: dict[str, str] = field(default_factory=dict)

    schema = "openghg_inversions.native_covariance_products"
    schema_version = 2

    def to_dataset(self) -> xr.Dataset:
        """Serialize labelled product blocks and their source/view identities.

        Returns:
            A dataset containing every product array plus schema, strategy,
            source/view identities, and basis-provenance metadata. Native
            covariance configuration is intentionally excluded because it
            occupies a separate child node in :meth:`to_datatree`.

            The identities bind declared source and view metadata across the
            artifact. They are not integrity checksums of the numerical output
            variables, whose values are deliberately not re-hashed here.
        """
        configuration_digest = self._validated_configuration_digest()
        arrays, coordinate_namespaces = _namespace_product_coordinates(
            {
                "restriction": self.restriction,
                "prolongation": self.prolongation,
                "state_covariance": self.state_covariance,
                "effective_observation_operator": self.effective_observation_operator,
                "observation_state_cross_covariance": self.observation_state_cross_covariance,
                "native_observation_covariance": self.native_observation_covariance,
            }
        )
        dataset = xr.Dataset(
            arrays,
            attrs={
                "schema": self.schema,
                "schema_version": self.schema_version,
                "strategy": self.strategy,
                "source_content_identity": self.source_content_identity,
                "view_identity": self.view_identity,
                "observation_covariance_view": self.observation_covariance_view,
                "covariance_configuration_digest": configuration_digest,
                "basis_provenance": json.dumps(self.basis_provenance, sort_keys=True),
                _PRODUCT_COORDINATE_NAMESPACES_ATTR: json.dumps(
                    coordinate_namespaces,
                    sort_keys=True,
                ),
            },
        )
        multiindex_dims = sorted(
            str(dim)
            for dim, index in dataset.indexes.items()
            if dim in dataset.dims and getattr(index, "nlevels", 1) > 1
        )
        dataset.attrs[_PRODUCT_MULTIINDEX_DIMS_ATTR] = json.dumps(multiindex_dims)
        return dataset

    def to_datatree(self) -> xr.DataTree:
        """Serialize products and reproducible covariance configuration.

        Returns:
            A tree whose root contains :meth:`to_dataset` output and whose
            optional ``covariance_configuration`` child preserves the native
            covariance constructor data, source binding, and content digest.
        """
        from openghg_inversions.serialization import reset_serialisation_multiindexes

        tree = xr.DataTree(reset_serialisation_multiindexes(self.to_dataset()))
        if self.covariance_configuration is None and self.covariance_configuration_digest:
            raise ValueError(
                "Cannot serialize covariance products as a DataTree without the declared "
                "covariance configuration content"
            )
        if self.covariance_configuration is not None:
            configuration_digest = self._validated_configuration_digest()
            tree["covariance_configuration"] = xr.DataTree(
                _encode_configuration_dataset(
                    self.covariance_configuration,
                    source_content_identity=self.source_content_identity,
                    configuration_digest=configuration_digest,
                )
            )
        return tree

    def _validated_configuration_digest(self) -> str:
        """Return the declared digest after checking any attached configuration."""
        if self.covariance_configuration is None:
            return self.covariance_configuration_digest
        actual_digest = _configuration_digest(self.covariance_configuration)
        if self.covariance_configuration_digest and self.covariance_configuration_digest != actual_digest:
            raise ValueError("Attached covariance configuration does not match its declared digest")
        return actual_digest

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
                source/view identity, projection strategy, numerical view, or
                JSON-encoded basis provenance is invalid. Numerical variable
                values are not checksummed by this metadata validation.
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
        source_content_identity = _validated_identity(
            dataset.attrs.get("source_content_identity"),
            name="source content identity",
        )
        view_identity = _validated_identity(
            dataset.attrs.get("view_identity"),
            name="view identity",
        )
        observation_covariance_view = dataset.attrs.get("observation_covariance_view")
        if observation_covariance_view not in {"dense", "diagonal"}:
            raise ValueError("Serialized covariance products have an invalid numerical view")
        expected_view_identity = _view_identity(
            source_content_identity,
            cast(Literal["dense", "diagonal"], observation_covariance_view),
        )
        if view_identity != expected_view_identity:
            raise ValueError("Serialized covariance products have an invalid derived view identity")
        configuration_digest = dataset.attrs.get("covariance_configuration_digest")
        if configuration_digest != "":
            _validated_identity(
                configuration_digest,
                name="covariance configuration digest",
            )
        strategy = str(dataset.attrs.get("strategy", ""))
        if not strategy:
            raise ValueError("Serialized covariance products are missing strategy metadata")
        for name in required:
            if dataset[name].attrs.get("source_content_identity") != source_content_identity:
                raise ValueError(f"Serialized product {name!r} does not share the root source identity")
            if dataset[name].attrs.get("view_identity") != view_identity:
                raise ValueError(f"Serialized product {name!r} does not share the root view identity")
            if dataset[name].attrs.get("projection_strategy") != strategy:
                raise ValueError(f"Serialized product {name!r} does not share the root projection strategy")
        coordinate_namespaces = _decode_product_coordinate_namespaces(
            dataset.attrs.get(_PRODUCT_COORDINATE_NAMESPACES_ATTR)
        )
        if set(coordinate_namespaces) != set(required):
            raise ValueError("Serialized covariance products have invalid coordinate namespace owners")
        for owner, mapping in coordinate_namespaces.items():
            prefix = f"__product__{owner}__"
            if any(not encoded.startswith(prefix) for encoded in mapping):
                raise ValueError(
                    f"Serialized covariance product {owner!r} has invalid coordinate namespace names"
                )
        encoded_coordinate_names = {
            coordinate for mapping in coordinate_namespaces.values() for coordinate in mapping
        }
        arrays: dict[str, xr.DataArray] = {}
        for name in required:
            mapping = coordinate_namespaces.get(name, {})
            foreign_coordinates = [
                coordinate
                for coordinate in encoded_coordinate_names - mapping.keys()
                if coordinate in dataset[name].coords
            ]
            array = dataset[name].drop_vars(foreign_coordinates)
            arrays[name] = _restore_product_coordinates(array, mapping)
        basis_provenance = _decode_basis_provenance(dataset.attrs.get("basis_provenance"))
        _validate_serialized_product_arrays(
            arrays,
            observation_covariance_view=cast(Literal["dense", "diagonal"], observation_covariance_view),
            basis_provenance=basis_provenance,
            declared_multiindex_dims=_decode_product_multiindex_dims(
                dataset.attrs.get(_PRODUCT_MULTIINDEX_DIMS_ATTR)
            ),
        )
        return cls(
            restriction=arrays["restriction"],
            prolongation=arrays["prolongation"],
            state_covariance=arrays["state_covariance"],
            effective_observation_operator=arrays["effective_observation_operator"],
            observation_state_cross_covariance=arrays["observation_state_cross_covariance"],
            native_observation_covariance=arrays["native_observation_covariance"],
            strategy=strategy,
            source_content_identity=source_content_identity,
            view_identity=view_identity,
            observation_covariance_view=cast(Literal["dense", "diagonal"], observation_covariance_view),
            covariance_configuration_digest=str(configuration_digest),
            basis_provenance=basis_provenance,
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
            ValueError: If the root product schema, source/view identities,
                strategy, MultiIndex metadata, or the covariance
                configuration's source binding or content digest is invalid.
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
            if root_dataset.attrs.get("covariance_configuration_digest"):
                raise ValueError("Serialized covariance products are missing covariance configuration")
            return products
        configuration_digest = _validated_identity(
            root_dataset.attrs.get("covariance_configuration_digest"),
            name="covariance configuration digest",
        )
        return replace(
            products,
            covariance_configuration=_decode_configuration_dataset(
                cast(xr.DataTree, tree["covariance_configuration"]).to_dataset(inherit=False),
                expected_source_content_identity=products.source_content_identity,
                expected_configuration_digest=configuration_digest,
            ),
        )


def _namespace_product_coordinates(
    arrays: dict[str, xr.DataArray],
) -> tuple[dict[str, xr.DataArray], dict[str, dict[str, str]]]:
    """Namespace auxiliary coordinates so independent axes cannot collide in a dataset."""
    canonical_multiindexes: dict[str, Any] = {}
    normalized: dict[str, xr.DataArray] = {}
    for variable_name, original in arrays.items():
        array = original
        for dim, index in tuple(array.indexes.items()):
            dim = str(dim)
            if dim not in array.dims or getattr(index, "nlevels", 1) <= 1:
                continue
            canonical = canonical_multiindexes.setdefault(dim, index)
            if not index.equals(canonical) or tuple(index.names) != tuple(canonical.names):
                raise ValueError(f"Product arrays carry inconsistent MultiIndex labels for {dim!r}")
            if index is canonical:
                continue
            index_coordinate_names = [dim, *(str(name) for name in index.names)]
            array = array.drop_indexes(index_coordinate_names).drop_vars(index_coordinate_names)
            array = array.assign_coords(xr.Coordinates.from_pandas_multiindex(canonical, dim))
        normalized[variable_name] = array
    arrays = normalized

    occupied = {str(name) for array in arrays.values() for name in (*array.dims, *array.coords)}
    usages: dict[str, dict[tuple[str, ...], tuple[bool, bool, int]]] = {}
    for array in arrays.values():
        has_multiindex = any(
            dim in array.dims and getattr(index, "nlevels", 1) > 1 for dim, index in array.indexes.items()
        )
        for coordinate_name in (str(name) for name in array.coords):
            if coordinate_name in array.dims:
                continue
            coordinate_dims = tuple(str(dim) for dim in array.coords[coordinate_name].dims)
            previous = usages.setdefault(coordinate_name, {}).get(
                coordinate_dims,
                (False, False, 0),
            )
            usages[coordinate_name][coordinate_dims] = (
                previous[0] or coordinate_name in array.xindexes,
                previous[1] or has_multiindex,
                previous[2] + 1,
            )
    canonical_signatures = {
        name: max(signatures, key=lambda dims: signatures[dims]) for name, signatures in usages.items()
    }

    encoded: dict[str, xr.DataArray] = {}
    namespaces: dict[str, dict[str, str]] = {}
    for variable_name, array in arrays.items():
        rename: dict[str, str] = {}
        for coordinate_name in (str(name) for name in array.coords):
            if coordinate_name in array.dims:
                continue
            coordinate_dims = tuple(str(dim) for dim in array.coords[coordinate_name].dims)
            if coordinate_name in array.xindexes or coordinate_dims == canonical_signatures[coordinate_name]:
                continue
            candidate = f"__product__{variable_name}__{coordinate_name}"
            suffix = 2
            while candidate in occupied:
                candidate = f"__product__{variable_name}__{coordinate_name}_{suffix}"
                suffix += 1
            occupied.add(candidate)
            rename[coordinate_name] = candidate
        encoded[variable_name] = array.rename(rename) if rename else array
        namespaces[variable_name] = {encoded_name: original for original, encoded_name in rename.items()}
    return encoded, namespaces


def _decode_product_coordinate_namespaces(value: object) -> dict[str, dict[str, str]]:
    """Decode and validate reversible per-product coordinate namespaces."""
    if not isinstance(value, str):
        raise ValueError("Serialized covariance products are missing coordinate namespace metadata")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Serialized covariance products have invalid coordinate namespace metadata"
        ) from error
    if not isinstance(decoded, dict):
        raise ValueError("Serialized covariance products have invalid coordinate namespace metadata")
    result: dict[str, dict[str, str]] = {}
    for variable_name, mapping in decoded.items():
        if (
            not isinstance(variable_name, str)
            or not isinstance(mapping, dict)
            or any(
                not isinstance(encoded, str) or not isinstance(original, str)
                for encoded, original in mapping.items()
            )
        ):
            raise ValueError("Serialized covariance products have invalid coordinate namespace metadata")
        result[variable_name] = mapping
    return result


def _restore_product_coordinates(array: xr.DataArray, mapping: dict[str, str]) -> xr.DataArray:
    """Restore one product array's original auxiliary coordinate names."""
    missing = [name for name in mapping if name not in array.coords]
    if missing:
        raise ValueError(f"Serialized product is missing namespaced coordinates {missing}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Serialized product coordinate namespace metadata is not one-to-one")
    return array.rename(mapping) if mapping else array


def _decode_product_multiindex_dims(value: object) -> tuple[str, ...]:
    """Decode dimensions declared to require restored MultiIndex semantics."""
    if not isinstance(value, str):
        raise ValueError("Serialized covariance products are missing MultiIndex declarations")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("Serialized covariance products have invalid MultiIndex declarations") from error
    if not isinstance(decoded, list) or any(not isinstance(dim, str) for dim in decoded):
        raise ValueError("Serialized covariance products have invalid MultiIndex declarations")
    return tuple(decoded)


def _validate_serialized_product_arrays(
    arrays: dict[str, xr.DataArray],
    *,
    observation_covariance_view: Literal["dense", "diagonal"],
    basis_provenance: dict[str, str],
    declared_multiindex_dims: tuple[str, ...],
) -> None:
    """Validate dimensional, label, view, and restored-index invariants."""
    state_dim = basis_provenance.get("state_dim", "")
    if not state_dim:
        raise ValueError("Serialized covariance products are missing retained-state provenance")
    restriction = arrays["restriction"]
    prolongation = arrays["prolongation"]
    if not restriction.dims or restriction.dims[0] != state_dim:
        raise ValueError("Serialized restriction has invalid retained-state dimensions")
    native_dims = tuple(str(dim) for dim in restriction.dims[1:])
    if not native_dims or prolongation.dims != (*native_dims, state_dim):
        raise ValueError("Serialized restriction/prolongation dimensions are inconsistent")
    if restriction.sizes[state_dim] != prolongation.sizes[state_dim]:
        raise ValueError("Serialized restriction/prolongation retained-state sizes differ")

    state_covariance = arrays["state_covariance"]
    if state_covariance.ndim != 2 or state_covariance.dims[0] != state_dim:
        raise ValueError("Serialized state covariance has invalid matrix dimensions")
    state_column_dim = str(state_covariance.dims[1])
    expected_state_column_dim = _matrix_column_dim(state_dim, restriction.dims)
    if state_column_dim != expected_state_column_dim:
        raise ValueError("Serialized state covariance has an invalid column dimension name")
    state_size = restriction.sizes[state_dim]
    if state_covariance.shape != (state_size, state_size):
        raise ValueError("Serialized state covariance shape does not match retained state")

    effective = arrays["effective_observation_operator"]
    cross = arrays["observation_state_cross_covariance"]
    if effective.ndim != 2 or effective.dims[1] != state_dim:
        raise ValueError("Serialized effective observation operator has invalid dimensions")
    observation_dim = str(effective.dims[0])
    if cross.dims != (observation_dim, state_column_dim) or cross.shape != (
        effective.sizes[observation_dim],
        state_size,
    ):
        raise ValueError("Serialized observation/state products have inconsistent dimensions")

    observation_covariance = arrays["native_observation_covariance"]
    observation_size = effective.sizes[observation_dim]
    expected_shape = (observation_size, observation_size)
    if observation_covariance_view == "diagonal":
        if observation_covariance.dims != (observation_dim,) or observation_covariance.shape != (
            observation_size,
        ):
            raise ValueError("Serialized diagonal observation covariance has invalid dimensions")
    elif (
        observation_covariance.ndim != 2
        or observation_covariance.dims[0] != observation_dim
        or (observation_covariance.shape != expected_shape)
    ):
        raise ValueError("Serialized dense observation covariance has invalid dimensions")

    for name, array in arrays.items():
        values = np.asarray(array.values)
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError(f"Serialized product {name!r} must contain only finite real values")

    for dim in declared_multiindex_dims:
        owners = [array for array in arrays.values() if dim in array.dims]
        if not owners or any(
            (index := array.indexes.get(dim)) is None or getattr(index, "nlevels", 1) <= 1 for array in owners
        ):
            raise ValueError(f"Serialized covariance products did not restore MultiIndex dimension {dim!r}")

    _require_matching_coordinate(restriction, prolongation, state_dim, "retained state")
    _require_matching_coordinate(restriction, state_covariance, state_dim, "retained state")
    _require_column_coordinates(
        restriction,
        state_covariance,
        row_dim=state_dim,
        column_dim=state_column_dim,
        role="retained-state column",
    )
    _require_matching_coordinate(state_covariance, cross, state_column_dim, "retained-state column")
    _require_column_coordinates(
        restriction,
        cross,
        row_dim=state_dim,
        column_dim=state_column_dim,
        role="retained-state cross-covariance column",
    )
    _require_matching_coordinate(effective, cross, observation_dim, "observation")
    _require_matching_coordinate(effective, observation_covariance, observation_dim, "observation")
    if observation_covariance_view == "dense":
        observation_column_dim = str(observation_covariance.dims[1])
        if observation_column_dim != _matrix_column_dim(
            observation_dim,
            (observation_dim, *native_dims),
        ):
            raise ValueError("Serialized dense observation covariance has an invalid column dimension name")
        _require_column_coordinates(
            effective,
            observation_covariance,
            row_dim=observation_dim,
            column_dim=observation_column_dim,
            role="observation-covariance column",
        )


def _require_matching_coordinate(
    left: xr.DataArray,
    right: xr.DataArray,
    dim: str,
    role: str,
) -> None:
    """Require two arrays to carry identical labels on a shared dimension."""
    if dim not in left.coords or dim not in right.coords:
        raise ValueError(f"Serialized {role} products are missing coordinate {dim!r}")
    if not np.array_equal(left.coords[dim].values, right.coords[dim].values):
        raise ValueError(f"Serialized {role} product labels are inconsistent")


def _require_column_coordinates(
    row_array: xr.DataArray,
    column_array: xr.DataArray,
    *,
    row_dim: str,
    column_dim: str,
    role: str,
) -> None:
    """Require serialized matrix columns to mirror all row-axis coordinates."""
    expected = _column_coordinates(row_array, row_dim=row_dim, column_dim=column_dim)
    for name, coordinate in expected.items():
        if name not in column_array.coords:
            raise ValueError(f"Serialized {role} product is missing coordinate {name!r}")
        if not np.array_equal(column_array.coords[name].values, coordinate.values):
            raise ValueError(f"Serialized {role} product labels are inconsistent")


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
            array chunking: each batch fills a slice of one preallocated result.
            Dense ``H B H.T`` is still fully materialized.
        strategy: Retained projection choice. The default preserves bucket scalings.

    Returns:
        Frozen product dataclass whose arrays carry a shared source identity
        and a view identity derived from the requested dense/diagonal
        representation. These identities bind metadata and are not checksums
        of computed output values. Its contained arrays remain mutable. Inputs
        and products are eagerly materialized; dense observation covariance
        has quadratic observation-space storage. Full PSD eigendiagnostics are
        skipped above 512 rows to avoid adding cubic observation-space work.
        Input attrs/units are not propagated: product attrs are replaced by
        mathematical diagnostics because this API does not implement unit
        algebra.

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
    projection_strategy = strategy if strategy is not None else PreserveBucketProlongation()
    projection = projection_strategy.projection(
        covariance,
        basis_prolongation,
        native_dims=native_dims,
        state_dim=state_dim,
    )
    projection = _validated_projection(
        projection,
        native_reference=sensitivity,
        native_dims=native_dims,
        state_dim=state_dim,
    )
    state_column_dim = _matrix_column_dim(state_dim, projection.restriction.dims)
    state_column_coordinates = _column_coordinates(
        projection.restriction,
        row_dim=state_dim,
        column_dim=state_column_dim,
    )
    sensitivity = _namespace_axis_coordinate_collisions(
        sensitivity,
        axis_dim=observation_dim,
        occupied={str(name) for name in projection.prolongation.coords} | set(state_column_coordinates),
    )
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
    source_content_identity = _source_content_identity(
        covariance,
        projection.restriction,
        projection.prolongation,
        sensitivity,
        strategy=projection.strategy,
    )
    view_identity = _view_identity(source_content_identity, observation_covariance)
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
        "source_content_identity": source_content_identity,
        "view_identity": view_identity,
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
        source_content_identity=source_content_identity,
        view_identity=view_identity,
        observation_covariance_view=observation_covariance,
        covariance_configuration_digest=(
            _configuration_digest(covariance_configuration) if covariance_configuration is not None else ""
        ),
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
    it is not Dask chunking. The result is allocated once and filled by
    observation-column batch. Dense output therefore still materializes the
    complete quadratic observation covariance, while diagonal output
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
        Labelled dense ``H B H.T`` with dimensions ``(observation_dim,
        collision-safe column dimension)``, or ``diag(H B H.T)`` with dimension
        ``(observation_dim,)``. Observation-axis coordinates are preserved.

    Raises:
        ValueError: If covariance labels are incompatible or a dense result
            fails the symmetry diagnostic.
    """
    observation_column_dim = _matrix_column_dim(observation_dim, sensitivity.dims)
    observation_count = sensitivity.sizes[observation_dim]
    result_values = np.empty(
        (observation_count, observation_count) if output == "dense" else observation_count,
        dtype=np.result_type(sensitivity.dtype, np.float64),
    )
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
            result_values[:, start:stop] = np.asarray(block.values)
        else:
            block = (rhs * b_rhs).sum(dim=list(native_dims))
            result_values[start:stop] = np.asarray(block.values)
    if output == "dense":
        coords = {
            str(name): coordinate
            for name, coordinate in sensitivity.coords.items()
            if set(coordinate.dims).issubset({observation_dim})
        }
        coords.update(
            _column_coordinates(
                sensitivity,
                row_dim=observation_dim,
                column_dim=observation_column_dim,
            )
        )
        combined = xr.DataArray(
            result_values,
            dims=(observation_dim, observation_column_dim),
            coords=coords,
        )
        combined = _with_matrix_diagnostics(
            combined.rename("native_observation_covariance"), mathematical_name="H B H.T"
        )
    else:
        coords = {
            str(name): coordinate
            for name, coordinate in sensitivity.coords.items()
            if set(coordinate.dims).issubset({observation_dim})
        }
        combined = xr.DataArray(result_values, dims=observation_dim, coords=coords)
        combined = combined.rename("native_observation_covariance").assign_attrs(
            mathematical_name="diag(H B H.T)",
            minimum_diagonal=float(combined.min().item()),
            diagonal_nonnegative=bool((combined >= -1e-10).all().item()),
            diagnostic_tolerance=1e-10,
        )
    return combined


def _validated_projection(
    projection: RetainedProjection,
    *,
    native_reference: xr.DataArray,
    native_dims: tuple[str, ...],
    state_dim: str,
) -> RetainedProjection:
    """Validate custom strategy labels before any labelled contractions.

    Args:
        projection: Restriction/prolongation pair returned by a strategy.
        native_reference: Validated sensitivity carrying canonical native
            coordinates.
        native_dims: Ordered native dimensions.
        state_dim: Retained-state dimension.

    Returns:
        A projection with eager arrays in canonical dimension order.

    Raises:
        ValueError: If either array has invalid dimensions, values, or labels.
            Native coordinates must exactly equal the reference coordinates;
            xarray reordering or intersection is never used here.
    """
    restriction = projection.restriction
    if not isinstance(projection.strategy, str) or not projection.strategy:
        raise ValueError("Projection strategy must return a non-empty strategy identifier")
    expected_restriction_dims = {state_dim, *native_dims}
    if set(restriction.dims) != expected_restriction_dims or len(restriction.dims) != len(
        expected_restriction_dims
    ):
        raise ValueError("Projection strategy restriction has invalid labelled dimensions")
    restriction = _densify(restriction).transpose(state_dim, *native_dims)
    restriction_values = np.asarray(restriction.values)
    if np.iscomplexobj(restriction_values) or not np.all(np.isfinite(restriction_values)):
        raise ValueError("Projection strategy restriction must contain only finite real values")

    prolongation = _validated_prolongation(
        projection.prolongation,
        native_dims=native_dims,
        state_dim=state_dim,
    )
    for role, array in (("restriction", restriction), ("prolongation", prolongation)):
        _validate_exact_native_coordinates(
            array,
            native_reference,
            native_dims=native_dims,
            role=role,
        )
    if state_dim not in restriction.coords or state_dim not in prolongation.coords:
        raise ValueError("Projection strategy restriction and prolongation require state labels")
    if not np.array_equal(
        restriction.coords[state_dim].values,
        prolongation.coords[state_dim].values,
    ):
        raise ValueError("Projection strategy restriction/prolongation state labels differ")
    return replace(projection, restriction=restriction, prolongation=prolongation)


def _validate_exact_native_coordinates(
    array: xr.DataArray,
    reference: xr.DataArray,
    *,
    native_dims: tuple[str, ...],
    role: str,
) -> None:
    """Require exact native labels before product alignment or contraction.

    Args:
        array: Strategy-produced restriction or prolongation.
        reference: Validated array carrying canonical native coordinates.
        native_dims: Ordered native dimensions to compare.
        role: Array role used in validation errors.

    Raises:
        ValueError: If a native coordinate is absent, reordered, or different.
    """
    for dim in native_dims:
        if dim not in array.coords:
            raise ValueError(f"Projection strategy {role} is missing native coordinate {dim!r}")
        actual = np.asarray(array.coords[dim].values)
        expected = np.asarray(reference.coords[dim].values)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(
                f"Projection strategy {role} native coordinate {dim!r} must exactly match "
                "the covariance grid in the same order"
            )


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
    prolongation = projection.prolongation
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


def _namespace_axis_coordinate_collisions(
    array: xr.DataArray,
    *,
    axis_dim: str,
    occupied: set[str],
) -> xr.DataArray:
    """Give colliding auxiliary/index-level names a stable axis prefix."""
    rename: dict[str, str] = {}
    reserved = occupied | {str(name) for name in array.coords} | {str(dim) for dim in array.dims}
    for name, coordinate in array.coords.items():
        name = str(name)
        if name == axis_dim or axis_dim not in coordinate.dims or name not in occupied:
            continue
        candidate = f"{axis_dim}_{name}"
        suffix = 2
        while candidate in reserved:
            candidate = f"{axis_dim}_{name}_{suffix}"
            suffix += 1
        reserved.add(candidate)
        rename[name] = candidate
    return array.rename(rename) if rename else array


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
    raw_values = np.asarray(result.values)
    if np.iscomplexobj(raw_values) or not np.all(np.isfinite(raw_values)):
        raise ValueError("prolongation must contain only finite real values")
    values = np.asarray(raw_values, dtype=np.float64)
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
    if result.sizes[observation_dim] == 0:
        raise ValueError("native_sensitivity must contain at least one observation")
    values = np.asarray(result.values)
    if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
        raise ValueError("native_sensitivity must contain only finite real values")
    # Applying B to one labelled column performs the covariance-grid label validation.
    covariance.apply(result.isel({observation_dim: slice(0, 1)}))
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
            [
                json.dumps(
                    list(value),
                    separators=(",", ":"),
                    default=_json_label_default,
                )
                for value in source_values
            ],
            dtype=str,
        )
    else:
        values = np.asarray(source_values)
    return xr.DataArray(values, dims=column_dim, attrs=coordinate.attrs)


def _json_label_default(value: object) -> str:
    """Encode a rich tuple-label scalar as a stable string.

    NumPy scalars are first converted to their Python equivalent. Datetime-like
    objects use ``isoformat()``, while other objects use ``str()``.

    Args:
        value: Non-JSON-native coordinate-label value.

    Returns:
        Stable string suitable for a JSON tuple token.
    """
    if isinstance(value, np.generic):
        value = value.item()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


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
    raw_values = np.asarray(array.values)
    if np.iscomplexobj(raw_values):
        raise ValueError(f"{mathematical_name} must contain real values")
    values = np.asarray(raw_values, dtype=np.float64)
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


def _encode_configuration_dataset(
    dataset: xr.Dataset,
    *,
    source_content_identity: str,
    configuration_digest: str,
) -> xr.Dataset:
    """Namespace configuration dimensions before DataTree persistence.

    Args:
        dataset: Native covariance configuration to encode.
        source_content_identity: Identity of the source content whose
            covariance configuration this dataset describes.
        configuration_digest: Digest of the decoded configuration content.

    Returns:
        A copy whose dimensions have a ``covariance_configuration__`` prefix.
        The reversible original-to-encoded mapping is stored in the
        ``openghg_inversions:configuration_dims`` attribute so parent DataTree
        dimensions cannot absorb the child dimensions. The child also carries
        the source identity so configurations from different artifacts cannot
        be mixed silently.
    """
    rename = {str(dim): f"covariance_configuration__{dim}" for dim in dataset.dims}
    encoded = dataset.rename(rename).copy()
    encoded.attrs = dict(encoded.attrs)
    encoded.attrs["openghg_inversions:configuration_dims"] = json.dumps(rename, sort_keys=True)
    encoded.attrs["openghg_inversions:source_content_identity"] = source_content_identity
    encoded.attrs["openghg_inversions:configuration_digest"] = configuration_digest
    return encoded


def _decode_configuration_dataset(
    dataset: xr.Dataset,
    *,
    expected_source_content_identity: str,
    expected_configuration_digest: str,
) -> xr.Dataset:
    """Restore namespaced configuration dimensions after persistence.

    Args:
        dataset: Encoded covariance configuration child dataset.
        expected_source_content_identity: Source identity declared by the root
            product dataset.
        expected_configuration_digest: Configuration digest declared by the
            root product dataset.

    Returns:
        A dataset with original dimension names and encoding metadata removed.

    Raises:
        ValueError: If the source binding or dimension mapping metadata is
            absent or invalid.
    """
    child_source_identity = dataset.attrs.get("openghg_inversions:source_content_identity")
    if child_source_identity != expected_source_content_identity:
        raise ValueError("Serialized covariance configuration does not share the root source identity")
    child_configuration_digest = dataset.attrs.get("openghg_inversions:configuration_digest")
    if child_configuration_digest != expected_configuration_digest:
        raise ValueError("Serialized covariance configuration does not share the root digest")
    encoded_dims = dataset.attrs.get("openghg_inversions:configuration_dims")
    if not isinstance(encoded_dims, str):
        raise ValueError("Serialized covariance configuration is missing dimension metadata")
    rename = json.loads(encoded_dims)
    if not isinstance(rename, dict):
        raise ValueError("Serialized covariance configuration dimension metadata is invalid")
    restored = dataset.rename({str(encoded): str(original) for original, encoded in rename.items()})
    restored.attrs = dict(restored.attrs)
    del restored.attrs["openghg_inversions:configuration_dims"]
    del restored.attrs["openghg_inversions:source_content_identity"]
    del restored.attrs["openghg_inversions:configuration_digest"]
    if _configuration_digest(restored) != expected_configuration_digest:
        raise ValueError("Serialized covariance configuration content does not match its digest")
    return restored


def _configuration_digest(dataset: xr.Dataset) -> str:
    """Hash decoded covariance configuration content into a stable digest.

    Args:
        dataset: Configuration in its decoded, original-dimension form.

    Returns:
        Lowercase hexadecimal SHA-256 digest stable across supported NetCDF
        scalar and string coercions.
    """
    digest = hashlib.sha256()
    digest.update(b"openghg_inversions.native_covariance_products.configuration.v1\0")
    digest.update(_canonical_json(dataset.attrs).encode("utf-8"))
    for name in sorted(str(name) for name in dataset.coords):
        _update_configuration_array_digest(digest, dataset.coords[name])
    for name in sorted(str(name) for name in dataset.data_vars):
        _update_configuration_array_digest(digest, dataset[name])
    return digest.hexdigest()


def _update_configuration_array_digest(digest: _Digest, array: xr.DataArray) -> None:
    """Hash one configuration array across NetCDF scalar/string coercions.

    Args:
        digest: Digest receiving canonical array metadata and values.
        array: Configuration coordinate or data variable to hash.
    """
    digest.update(str(array.name).encode("utf-8"))
    digest.update(_canonical_json(tuple(str(dim) for dim in array.dims)).encode("utf-8"))
    digest.update(_canonical_json(array.attrs).encode("utf-8"))
    values = np.ascontiguousarray(array.values)
    digest.update(_canonical_json(values.shape).encode("utf-8"))
    if values.dtype.hasobject or values.dtype.kind in {"U", "S"}:
        digest.update(_canonical_json(values.tolist()).encode("utf-8"))
    else:
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(values.view(np.uint8).tobytes())


def _canonical_json(value: object) -> str:
    """Encode serialization-compatible metadata with normalized scalar types.

    Args:
        value: Metadata value accepted by JSON or
            :func:`_canonical_json_default`.

    Returns:
        Compact, key-sorted JSON text.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_canonical_json_default,
    )


def _canonical_json_default(value: object) -> object:
    """Normalize NumPy, byte, array, and datetime-like values for JSON.

    Args:
        value: Value rejected by the standard JSON encoder.

    Returns:
        A JSON-native scalar, list, or string.

    Raises:
        TypeError: If the value has no supported canonical representation.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    raise TypeError(f"Cannot encode {type(value).__name__} in a configuration digest")


def _source_content_identity(
    covariance: InvertibleNativeCovarianceAction,
    *arrays: xr.DataArray,
    strategy: str,
) -> str:
    """Hash shared scientific inputs and configuration into a stable identity.

    Execution batching is deliberately excluded, because changing
    ``observation_batch_size`` does not change the requested scientific
    product. The dense/diagonal numerical view and computed product values are
    also excluded. This identity binds source content, not stored-byte
    integrity, and therefore does not detect numerical output tampering. A
    covariance with ``to_dataset()`` contributes its configuration metadata,
    coordinates, and variables; the custom-action fallback is stable only when
    that object's ``repr`` is stable.

    Args:
        covariance: Native covariance action, preferably with serializable
            constructor configuration.
        *arrays: Labelled restriction, prolongation, and sensitivity arrays;
            their values, dimensions, and all coordinates are hashed.
        strategy: Stable retained-projection strategy name.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    digest.update(strategy.encode("utf-8"))
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


def _view_identity(
    source_content_identity: str,
    observation_covariance_view: Literal["dense", "diagonal"],
) -> str:
    """Derive a numerical-view identity from source content and view kind.

    Args:
        source_content_identity: Identity shared by all numerical views of the
            same ``B/H/Pi/U`` source.
        observation_covariance_view: Requested dense or diagonal view.

    Returns:
        Lowercase hexadecimal SHA-256 digest for this derived view.
    """
    digest = hashlib.sha256()
    digest.update(b"openghg_inversions.native_covariance_products.view.v1\0")
    digest.update(source_content_identity.encode("ascii"))
    digest.update(b"\0")
    digest.update(observation_covariance_view.encode("ascii"))
    return digest.hexdigest()


def _validated_identity(value: object, *, name: str) -> str:
    """Validate a lowercase hexadecimal SHA-256 identity.

    Args:
        value: Serialized identity candidate.
        name: Human-readable identity name used in an error.

    Returns:
        The validated identity string.

    Raises:
        ValueError: If the value is not a 64-character lowercase hex digest.
    """
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Serialized covariance products have an invalid {name}")
    return value


def _decode_basis_provenance(value: object) -> dict[str, str]:
    """Decode and validate the serialized string-to-string provenance map.

    Args:
        value: JSON text from root dataset metadata.

    Returns:
        Decoded provenance mapping.

    Raises:
        ValueError: If the value is not valid JSON encoding an object with
            string keys and string values.
    """
    if not isinstance(value, str):
        raise ValueError("Serialized covariance products have invalid basis provenance JSON")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Serialized covariance products have invalid basis provenance JSON") from exc
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in decoded.items()
    ):
        raise ValueError("Serialized covariance products basis provenance must be a string mapping")
    return decoded


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
        digest.update(labels.dtype.str.encode("ascii"))
        digest.update(repr(labels.shape).encode("ascii"))
        if labels.dtype.hasobject or labels.dtype.kind in {"U", "S"}:
            digest.update(repr(labels.tolist()).encode("utf-8"))
        else:
            digest.update(np.ascontiguousarray(labels).view(np.uint8).tobytes())
