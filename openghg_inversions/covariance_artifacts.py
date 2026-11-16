"""Versioned persistence for labelled covariance products and coherent reductions.

Two artifacts share one layout and one identity convention:

* a **products artifact** persists
  :class:`~openghg_inversions.basis.covariance_products.NativeCovarianceProducts`
  in either observation-covariance view, optionally without the dense
  prolongation ``U_*``;
* a **coherent artifact** persists
  :class:`~openghg_inversions.coherent_reduction.CoherentGaussianReduction`.
  It omits ``U_*``, ``H B Pi.T`` and ``H B H.T``, which its stored blocks
  already determine.

Layout
------
Each artifact is a :class:`xarray.DataTree`::

    /               header attributes only
    /covariance     the covariance action's own ``to_dataset()`` configuration
    /<axes node>    every logical-axis coordinate, stored once
      /<product>    one float64 variable named after the group, without
                    coordinates of its own

The axes node is ``products`` or ``reduction``. Product groups inherit its
coordinates, so xarray aligns them on build and on load. A coordinate shared
by several products must have equal labels in each. It is stored once, with
the union of the attributes the products give it; conflicting attribute
values are refused. MultiIndexes are
expanded with :func:`~openghg_inversions.serialization.encode_multiindexes_for_storage`.

Logical axes
------------
The ``openghg_inversions:axes`` attribute of the axes node declares the
logical axes ``state``, ``observation`` and ``native`` and the column axes
``state_column`` and, for a dense observation covariance,
``observation_column``. A column axis records the row axis it mirrors and an
explicit map from each row coordinate name (index, MultiIndex levels and
auxiliary coordinates) to its column counterpart. The map is read from the
column naming of :mod:`openghg_inversions._labelled_matrices` when saving and
used, never re-derived, when loading. Both save and load require each column
axis to be an exact copy of its row axis.

Values and attributes
---------------------
Products are stored as float64, except that ``U_*`` may be stored as float32
(``prolongation_dtype="float32"``). ``U_*`` only lifts retained values into
native space, and ``H_alpha`` is stored separately. ``C_alpha`` is factorized,
``A`` involves cancellation and the observation-side products feed ``A``, so
they stay float64, as does the sparse ``Pi``. Loading checks each product's
dtype against this rule. A sparse restriction ``Pi`` is stored with
CF compression by gathering: a ``restriction_entry`` dimension, an integer
coordinate of flat indices with a ``compress`` attribute, and the stored
values. It loads back as a zero-fill :class:`sparse.COO` without being
densified. A product's ``units`` stay a plain CF attribute. Its other
attributes are kept as one JSON string, so Booleans and floats round-trip
as JSON values; tuples become lists.

Identities
----------
``source_id`` identifies the inputs: the covariance configuration, the
restriction including its labels and provenance, the caller's ``inputs_id``
for the sensitivity ``H`` and native mean ``m``, and the state and
observation dimensions. It does not depend on the observation view or on
whether ``U_*`` is stored. Dense and diagonal products and the coherent
artifact from the same inputs therefore share it.

``content_id`` is a SHA-256 digest of everything stored except itself. It is
the one root identity that binds the constituent products. Both identities
are computed from current content, never cached: editing a loaded object and
saving it gives a new ``content_id``. Loading recomputes both and fails if
the file was altered. The digest detects accidental modification, truncation
and groups mixed from different artifacts. It is not a signature, and
loading does not re-derive numerical relationships between products, such as
``H_alpha C_alpha = H B Pi.T`` or positive semidefiniteness of ``A``. Those
hold by construction in the kernel. Loading checks only that values are
finite, ``C_alpha`` is symmetric and positive definite, and a diagonal view
is nonnegative.

Schema evolution
----------------
Readers accept only the schema versions they know. Any change to the layout,
required content, meaning or digest canonicalization increments
``schema_version``. Strict validation makes even additive metadata a version
change. A reader for the previous version is kept, at least until the
following minor release, and decodes into the current in-memory types. The
embedded covariance configuration and the MultiIndex metadata keep their own
independent versions.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Literal, get_args

from dask.base import compute
import numpy as np
import pandas as pd
from sparse import COO, SparseArray  # type: ignore[import-untyped]
import xarray as xr

from openghg_inversions._serialization_codecs import _copy_json_mapping, _json_value
from openghg_inversions.array_ops import materialize_together
from openghg_inversions.basis.covariance_products import (
    NativeCovarianceProducts,
    covariance_natural_prolongation,
)
from openghg_inversions.coherent_reduction import CoherentGaussianReduction
from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.serialization import (
    MULTIINDEX_DIMS_ATTR,
    encode_multiindexes_for_storage,
    open_datatree_loaded,
    restore_declared_multiindexes,
    save_datatree,
)
from openghg_inversions.source_covariance import IndependentSourceCovariance

__all__ = [
    "COHERENT_SCHEMA",
    "CoherentReductionArtifact",
    "NativeCovarianceProductsArtifact",
    "PRODUCTS_SCHEMA",
    "SCHEMA_VERSION",
    "coherent_from_datatree",
    "coherent_to_datatree",
    "content_id",
    "load_coherent_reduction",
    "load_products",
    "products_from_datatree",
    "products_to_datatree",
    "save_coherent_reduction",
    "save_products",
    "source_id",
    "with_prolongation",
]

PRODUCTS_SCHEMA = "openghg_inversions.native_covariance_products"
COHERENT_SCHEMA = "openghg_inversions.coherent_gaussian_reduction"
SCHEMA_VERSION = 1

AXES_ATTR = "openghg_inversions:axes"
ATTRS_ATTR = "openghg_inversions:attrs"
STORAGE_ATTR = "openghg_inversions:storage"
GATHERED_DIM = "restriction_entry"

ProlongationDtype = Literal["float64", "float32"]
# Storage dtypes each product may use. Only U_* may be float32: it is read back
# only to lift retained values into native space, while H_alpha is stored
# separately. C_alpha is factorized, A involves cancellation, and the
# observation-side products feed A, so they and the sparse Pi stay float64.
_DEFAULT_STORAGE_DTYPES = frozenset({"float64"})
_STORAGE_DTYPES = {"prolongation": frozenset(get_args(ProlongationDtype))}

SerializableCovariance = SeparableExponentialCovariance | IndependentSourceCovariance
_COVARIANCE_TYPES: dict[str, type[SeparableExponentialCovariance] | type[IndependentSourceCovariance]] = {
    SeparableExponentialCovariance.schema: SeparableExponentialCovariance,
    IndependentSourceCovariance.schema: IndependentSourceCovariance,
}

# Logical axes of each product, in dimension order. "native" expands to the
# ordered native dimensions.
_PRODUCT_AXES: dict[str, tuple[str, ...]] = {
    "restriction": ("state", "native"),
    "prolongation": ("native", "state"),
    "state_covariance": ("state", "state_column"),
    "effective_observation_operator": ("observation", "state"),
    "observation_state_cross_covariance": ("observation", "state_column"),
    "native_observation_covariance": ("observation", "observation_column"),
}
_DIAGONAL_OBSERVATION_AXES = ("observation",)
_COHERENT_AXES: dict[str, tuple[str, ...]] = {
    "restriction": ("state", "native"),
    "retained_mean": ("state",),
    "retained_covariance": ("state", "state_column"),
    "effective_observation_operator": ("observation", "state"),
    "native_observation_mean": ("observation",),
    "observation_intercept": ("observation",),
    "unresolved_observation_covariance": ("observation", "observation_column"),
}
_OPTIONAL_PRODUCTS = {"prolongation"}

# Products in "1"; products linear in H share its units; quadratic ones use (u)^2.
_DIMENSIONLESS = {"restriction", "prolongation", "state_covariance", "retained_mean", "retained_covariance"}
_LINEAR_IN_H = {
    "effective_observation_operator",
    "observation_state_cross_covariance",
    "native_observation_mean",
    "observation_intercept",
}
_QUADRATIC_IN_H = {"native_observation_covariance", "unresolved_observation_covariance"}

_PRODUCTS_ROOT_ATTRS = {
    "schema",
    "schema_version",
    "content_id",
    "source_id",
    "inputs_id",
    "provenance",
    "strategy",
    "observation_covariance_view",
}
_COHERENT_ROOT_ATTRS = {
    "schema",
    "schema_version",
    "content_id",
    "source_id",
    "inputs_id",
    "provenance",
    "strategy",
    "native_family",
    "reduction",
}
NATIVE_FAMILY = "gaussian"
REDUCTION_KIND = "exact_gaussian_conditional"

_RESERVED_NAMES = {
    *_PRODUCT_AXES,
    *_COHERENT_AXES,
    GATHERED_DIM,
    "covariance",
    "products",
    "reduction",
}


@dataclass(frozen=True, slots=True, eq=False)
class NativeCovarianceProductsArtifact:
    """Native-covariance products with the metadata that binds their inputs.

    Attributes:
        products: The labelled product blocks. ``products.prolongation`` is
            ``None`` when the artifact was saved without ``U_*``.
        covariance: The native covariance action ``B`` that defined the
            products. Its configuration is embedded in the artifact.
        inputs_id: Non-empty identifier of the source of the sensitivity
            ``H``, for example a prepared-inputs content identifier. ``H`` is
            not persisted, so this is how the artifact names it.
        provenance: JSON object of basis or projection provenance, frozen
            read-only.

    Notes:
        Identities are not stored on the object; :func:`source_id` and
        :func:`content_id` compute them from current content. The contained
        DataArrays are borrowed and remain mutable.
    """

    products: NativeCovarianceProducts
    covariance: SerializableCovariance
    inputs_id: str
    provenance: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        """Validate binding metadata and freeze the provenance mapping."""
        _check_binding_fields(self.covariance, self.inputs_id)
        if not isinstance(self.products, NativeCovarianceProducts):
            raise TypeError("products must be NativeCovarianceProducts")
        object.__setattr__(self, "provenance", _frozen_json_object(self.provenance, "provenance"))


@dataclass(frozen=True, slots=True, eq=False)
class CoherentReductionArtifact:
    """A coherent Gaussian reduction with the metadata that binds its inputs.

    Attributes:
        reduction: The labelled coherent reduction, including ``Pi``.
        covariance: The native covariance action ``B`` used by the reduction.
            Its configuration is embedded in the artifact.
        inputs_id: Non-empty identifier of the source of the sensitivity
            ``H`` and native mean ``m``.
        provenance: JSON object of basis or projection provenance, frozen
            read-only.

    Notes:
        As for :class:`NativeCovarianceProductsArtifact`, identities are
        computed from content rather than stored.
    """

    reduction: CoherentGaussianReduction
    covariance: SerializableCovariance
    inputs_id: str
    provenance: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        """Validate binding metadata and freeze the provenance mapping."""
        _check_binding_fields(self.covariance, self.inputs_id)
        if not isinstance(self.reduction, CoherentGaussianReduction):
            raise TypeError("reduction must be a CoherentGaussianReduction")
        object.__setattr__(self, "provenance", _frozen_json_object(self.provenance, "provenance"))


Artifact = NativeCovarianceProductsArtifact | CoherentReductionArtifact


# --- Public API ---------------------------------------------------------------------------


def products_to_datatree(
    artifact: NativeCovarianceProductsArtifact,
    *,
    include_prolongation: bool = True,
    prolongation_dtype: ProlongationDtype = "float64",
) -> xr.DataTree:
    """Encode a products artifact as a versioned DataTree.

    This is a named serialization boundary. Lazy payloads are computed once;
    a sparse restriction stays sparse. The artifact is not mutated.

    Args:
        artifact: Products and binding metadata to encode.
        include_prolongation: Store the dense ``U_*``. When false, it can be
            rebuilt after loading with :func:`with_prolongation`.
        prolongation_dtype: Storage precision of ``U_*``. ``"float32"``
            halves its size; ``U_*`` only lifts retained values into native
            space, so its rounding does not reach ``H_alpha`` or ``A``. Every
            other product is stored as float64.

    Returns:
        The encoded tree, with ``source_id`` and ``content_id`` root
        attributes.

    Raises:
        ValueError: If labels, units, names or dtypes cannot be stored under
            the schema, or ``U_*`` is requested but absent.
    """
    tree = _encode_products(
        artifact, include_prolongation=include_prolongation, prolongation_dtype=prolongation_dtype
    )
    tree.attrs["content_id"] = _digest_tree(tree)
    return tree


def coherent_to_datatree(artifact: CoherentReductionArtifact) -> xr.DataTree:
    """Encode a coherent-reduction artifact as a versioned DataTree.

    Args:
        artifact: Coherent reduction and binding metadata to encode.

    Returns:
        The encoded tree, with ``source_id`` and ``content_id`` root
        attributes.

    Raises:
        ValueError: If labels, units, names or dtypes cannot be stored under
            the schema.
    """
    tree = _encode_coherent(artifact)
    tree.attrs["content_id"] = _digest_tree(tree)
    return tree


def save_products(
    artifact: NativeCovarianceProductsArtifact,
    path: str | Path,
    *,
    include_prolongation: bool = True,
    prolongation_dtype: ProlongationDtype = "float64",
) -> None:
    """Save a products artifact to NetCDF or Zarr.

    Args:
        artifact: Products and binding metadata to save.
        path: Destination ending in ``.nc`` or ``.zarr``.
        include_prolongation: Store the dense ``U_*``.
        prolongation_dtype: Storage precision of ``U_*``, ``"float64"`` or
            ``"float32"``. Every other product is stored as float64.
    """
    tree = products_to_datatree(
        artifact, include_prolongation=include_prolongation, prolongation_dtype=prolongation_dtype
    )
    save_datatree(tree, path)


def save_coherent_reduction(artifact: CoherentReductionArtifact, path: str | Path) -> None:
    """Save a coherent-reduction artifact to NetCDF or Zarr.

    Args:
        artifact: Coherent reduction and binding metadata to save.
        path: Destination ending in ``.nc`` or ``.zarr``.
    """
    save_datatree(coherent_to_datatree(artifact), path)


def load_products(path: str | Path) -> NativeCovarianceProductsArtifact:
    """Load and validate a products artifact.

    Args:
        path: NetCDF or Zarr artifact written by :func:`save_products`.

    Returns:
        The decoded artifact; ``products.prolongation`` is ``None`` when the
        file does not store ``U_*``, and keeps its stored float32 or float64
        precision otherwise.

    Raises:
        ValueError: If the schema, labels, values or identities are invalid.
    """
    return products_from_datatree(open_datatree_loaded(path))


def load_coherent_reduction(path: str | Path) -> CoherentReductionArtifact:
    """Load and validate a coherent-reduction artifact.

    Args:
        path: NetCDF or Zarr artifact written by
            :func:`save_coherent_reduction`.

    Returns:
        The decoded artifact.

    Raises:
        ValueError: If the schema, labels, values or identities are invalid.
    """
    return coherent_from_datatree(open_datatree_loaded(path))


def products_from_datatree(tree: xr.DataTree) -> NativeCovarianceProductsArtifact:
    """Validate a products tree completely, then decode it.

    Args:
        tree: Eagerly loaded tree in the products schema.

    Returns:
        The decoded artifact.

    Raises:
        ValueError: If the schema, labels, values or identities are invalid.
    """
    _check_root(tree, PRODUCTS_SCHEMA, _PRODUCTS_ROOT_ATTRS, axes_node="products")
    view = tree.attrs["observation_covariance_view"]
    if view not in ("dense", "diagonal"):
        raise ValueError(f"Unsupported observation_covariance_view {view!r}")
    schema = _products_schema(view)
    covariance = _decode_covariance(tree.children["covariance"])
    arrays = _decode_axes_node(tree.children["products"], schema, optional=_OPTIONAL_PRODUCTS)
    _check_covariance_binding(covariance, arrays["restriction"])
    _check_units(arrays)
    diagonal = "native_observation_covariance" if view == "diagonal" else None
    _check_numerics(arrays, square="state_covariance", diagonal=diagonal)
    products = NativeCovarianceProducts(
        restriction=arrays["restriction"],
        prolongation=arrays.get("prolongation"),
        state_covariance=arrays["state_covariance"],
        effective_observation_operator=arrays["effective_observation_operator"],
        observation_state_cross_covariance=arrays["observation_state_cross_covariance"],
        native_observation_covariance=arrays["native_observation_covariance"],
        strategy=_nonempty_string(tree.attrs["strategy"], "strategy"),
        observation_covariance_view=view,
    )
    artifact = NativeCovarianceProductsArtifact(
        products=products,
        covariance=covariance,
        inputs_id=_nonempty_string(tree.attrs["inputs_id"], "inputs_id"),
        provenance=_decode_json_object(tree.attrs["provenance"], "provenance"),
    )
    stored_prolongation = arrays.get("prolongation")
    single_precision = stored_prolongation is not None and stored_prolongation.dtype == np.float32
    _check_identities(
        tree,
        artifact,
        include_prolongation=stored_prolongation is not None,
        prolongation_dtype="float32" if single_precision else "float64",
    )
    return artifact


def coherent_from_datatree(tree: xr.DataTree) -> CoherentReductionArtifact:
    """Validate a coherent-reduction tree completely, then decode it.

    Args:
        tree: Eagerly loaded tree in the coherent-reduction schema.

    Returns:
        The decoded artifact.

    Raises:
        ValueError: If the schema, labels, values or identities are invalid.
    """
    _check_root(tree, COHERENT_SCHEMA, _COHERENT_ROOT_ATTRS, axes_node="reduction")
    if tree.attrs["native_family"] != NATIVE_FAMILY or tree.attrs["reduction"] != REDUCTION_KIND:
        raise ValueError("Unsupported native_family or reduction kind")
    covariance = _decode_covariance(tree.children["covariance"])
    arrays = _decode_axes_node(tree.children["reduction"], _COHERENT_AXES, optional=set())
    _check_covariance_binding(covariance, arrays["restriction"])
    _check_units(arrays)
    _check_numerics(arrays, square="retained_covariance", diagonal=None)
    reduction = CoherentGaussianReduction(
        restriction=arrays["restriction"],
        retained_mean=arrays["retained_mean"],
        retained_covariance=arrays["retained_covariance"],
        effective_observation_operator=arrays["effective_observation_operator"],
        native_observation_mean=arrays["native_observation_mean"],
        observation_intercept=arrays["observation_intercept"],
        unresolved_observation_covariance=arrays["unresolved_observation_covariance"],
        projection_strategy=_nonempty_string(tree.attrs["strategy"], "strategy"),
    )
    artifact = CoherentReductionArtifact(
        reduction=reduction,
        covariance=covariance,
        inputs_id=_nonempty_string(tree.attrs["inputs_id"], "inputs_id"),
        provenance=_decode_json_object(tree.attrs["provenance"], "provenance"),
    )
    _check_identities(tree, artifact, include_prolongation=False, prolongation_dtype="float64")
    return artifact


def source_id(artifact: Artifact) -> str:
    """Return the identity of an artifact's inputs.

    The identity covers the covariance configuration, the restriction's
    values, label values and own attributes (its projection provenance),
    ``inputs_id`` and the state and observation dimensions. Coordinate
    attributes such as units are shared with the other products and are
    covered by :func:`content_id` instead. It ignores the observation-covariance view and whether
    ``U_*`` is stored, so dense, diagonal and coherent artifacts built from
    the same inputs share it.

    Args:
        artifact: A products or coherent-reduction artifact.

    Returns:
        ``"sha256:"`` followed by a hexadecimal digest.
    """
    restriction = (
        artifact.products.restriction
        if isinstance(artifact, NativeCovarianceProductsArtifact)
        else artifact.reduction.restriction
    )
    observation_dim = _observation_dim(artifact)
    restriction_group = _product_dataset("restriction", restriction)
    # Label values identify Pi; coordinate attributes such as units are
    # shared with other products on the axes node and are covered by content_id.
    encoded_labels = encode_multiindexes_for_storage(xr.Dataset(coords=restriction.coords))
    restriction_labels = xr.Dataset(
        {str(name): xr.Variable(variable.dims, variable.values) for name, variable in encoded_labels.variables.items()},
        attrs=encoded_labels.attrs,
    )
    header = {
        "covariance": _digest_datasets([("covariance", artifact.covariance.to_dataset())]),
        "restriction": _digest_datasets([("labels", restriction_labels), ("values", restriction_group)]),
        "inputs_id": artifact.inputs_id,
        "state_dim": str(restriction.dims[0]),
        "observation_dim": observation_dim,
    }
    return "sha256:" + sha256(json.dumps(header, sort_keys=True).encode()).hexdigest()


def content_id(
    artifact: Artifact,
    *,
    include_prolongation: bool = True,
    prolongation_dtype: ProlongationDtype = "float64",
) -> str:
    """Return the content identity an artifact would be saved with.

    Args:
        artifact: A products or coherent-reduction artifact.
        include_prolongation: For a products artifact, whether ``U_*`` would
            be stored. Ignored for a coherent artifact.
        prolongation_dtype: For a products artifact, the precision ``U_*``
            would be stored with. Ignored for a coherent artifact.

    Returns:
        ``"sha256:"`` followed by a hexadecimal digest of every stored value,
        label and attribute except ``content_id`` itself.
    """
    if isinstance(artifact, NativeCovarianceProductsArtifact):
        return _digest_tree(
            _encode_products(
                artifact, include_prolongation=include_prolongation, prolongation_dtype=prolongation_dtype
            )
        )
    return _digest_tree(_encode_coherent(artifact))


def with_prolongation(
    artifact: NativeCovarianceProductsArtifact,
    *,
    rhs_block_size: int = 64,
) -> NativeCovarianceProductsArtifact:
    """Return a products artifact whose ``U_*`` is present, rebuilding it if needed.

    ``U_* = B Pi.T C_alpha^-1`` is rebuilt from the embedded covariance
    configuration, the stored ``Pi`` and ``C_alpha`` with
    :func:`~openghg_inversions.basis.covariance_products.covariance_natural_prolongation`.
    The result matches the stored ``U_*`` up to floating-point roundoff. The
    input artifact is not mutated.

    Args:
        artifact: A products artifact, typically loaded without ``U_*``.
        rhs_block_size: Retained states per covariance application.

    Returns:
        ``artifact`` itself if ``U_*`` is present, otherwise a new artifact
        with the rebuilt ``U_*``. ``source_id`` is unchanged.
    """
    if artifact.products.prolongation is not None:
        return artifact
    prolongation = covariance_natural_prolongation(
        artifact.covariance,
        artifact.products.restriction,
        artifact.products.state_covariance,
        rhs_block_size=rhs_block_size,
    )
    return replace(artifact, products=replace(artifact.products, prolongation=prolongation))


# --- Encoding -----------------------------------------------------------------------------


def _encode_products(
    artifact: NativeCovarianceProductsArtifact,
    *,
    include_prolongation: bool,
    prolongation_dtype: str,
) -> xr.DataTree:
    """Build the products tree without ``content_id``."""
    if prolongation_dtype not in _STORAGE_DTYPES["prolongation"]:
        raise ValueError(f"prolongation_dtype must be 'float64' or 'float32'; got {prolongation_dtype!r}")
    products = artifact.products
    arrays = {
        "restriction": products.restriction,
        "state_covariance": products.state_covariance,
        "effective_observation_operator": products.effective_observation_operator,
        "observation_state_cross_covariance": products.observation_state_cross_covariance,
        "native_observation_covariance": products.native_observation_covariance,
    }
    if include_prolongation:
        if products.prolongation is None:
            raise ValueError("include_prolongation=True needs products with U_*; use with_prolongation first")
        arrays["prolongation"] = products.prolongation
    schema = _products_schema(products.observation_covariance_view)
    axes_node = _encode_axes_node(arrays, schema, dtypes={"prolongation": prolongation_dtype})
    return _assemble(
        artifact,
        axes_name="products",
        axes_node=axes_node,
        attrs={
            "schema": PRODUCTS_SCHEMA,
            "strategy": products.strategy,
            "observation_covariance_view": products.observation_covariance_view,
        },
    )


def _encode_coherent(artifact: CoherentReductionArtifact) -> xr.DataTree:
    """Build the coherent-reduction tree without ``content_id``."""
    reduction = artifact.reduction
    arrays = {name: getattr(reduction, name) for name in _COHERENT_AXES}
    axes_node = _encode_axes_node(arrays, _COHERENT_AXES)
    return _assemble(
        artifact,
        axes_name="reduction",
        axes_node=axes_node,
        attrs={
            "schema": COHERENT_SCHEMA,
            "strategy": reduction.projection_strategy,
            "native_family": NATIVE_FAMILY,
            "reduction": REDUCTION_KIND,
        },
    )


def _assemble(
    artifact: Artifact,
    *,
    axes_name: str,
    axes_node: dict[str, xr.Dataset],
    attrs: dict[str, object],
) -> xr.DataTree:
    """Combine the covariance configuration, axes node and header attributes."""
    tree = xr.DataTree.from_dict(
        {
            "/covariance": artifact.covariance.to_dataset(),
            **{f"/{axes_name}{path}": dataset for path, dataset in axes_node.items()},
        }
    )
    tree.attrs.update(
        attrs,
        schema_version=SCHEMA_VERSION,
        source_id=source_id(artifact),
        inputs_id=artifact.inputs_id,
        provenance=json.dumps(dict(artifact.provenance), sort_keys=True),
    )
    return tree


def _products_schema(view: str) -> dict[str, tuple[str, ...]]:
    """Return the logical axes of each product for one observation view."""
    if view == "diagonal":
        return {**_PRODUCT_AXES, "native_observation_covariance": _DIAGONAL_OBSERVATION_AXES}
    if view == "dense":
        return dict(_PRODUCT_AXES)
    raise ValueError(f"Unsupported observation_covariance_view {view!r}")


def _encode_axes_node(
    arrays: Mapping[str, xr.DataArray],
    schema: Mapping[str, tuple[str, ...]],
    *,
    dtypes: Mapping[str, str] | None = None,
) -> dict[str, xr.Dataset]:
    """Return the axes dataset at ``""`` and one dataset per product group.

    Products are stored as float64 unless ``dtypes`` names another storage
    dtype for them.
    """
    _check_units(arrays)
    registry = _axes_registry(arrays, schema)
    for name, array in arrays.items():
        expected = _expected_dims(schema[name], registry)
        if tuple(str(dim) for dim in array.dims) != expected:
            raise ValueError(f"{name} must have dimensions {expected!r}; got {array.dims!r}")

    # Coordinates on logical axes are stored once; scalar coordinates stay
    # with their product.
    try:
        axes = xr.merge(
            [array.drop_vars(_scalar_coordinates(array)).coords.to_dataset() for array in arrays.values()],
            compat="equals",
            join="exact",
            combine_attrs="no_conflicts",
        )
    except ValueError as exc:
        raise ValueError(
            "Products carry inconsistent labels, or conflicting coordinate attributes, on a shared logical axis"
        ) from exc
    reserved = _RESERVED_NAMES.intersection(str(name) for name in (*axes.dims, *axes.coords))
    if reserved:
        raise ValueError(f"Axis name {sorted(reserved)[0]!r} is reserved by the covariance artifact schema")
    unlabelled = [dim for dim in axes.dims if dim not in axes.indexes]
    if unlabelled:
        raise ValueError(f"Axis dimension {unlabelled[0]!r} must have an index coordinate")
    _check_label_types(axes)
    _check_mirrors(axes, registry)

    encoded_axes = encode_multiindexes_for_storage(axes)
    encoded_axes.attrs = {**encoded_axes.attrs, AXES_ATTR: json.dumps(registry, sort_keys=True)}
    result = {"": encoded_axes}
    for name, array in arrays.items():
        result[f"/{name}"] = _product_dataset(name, array, dtype=(dtypes or {}).get(name, "float64"))
    return result


def _product_dataset(name: str, array: xr.DataArray, *, dtype: str = "float64") -> xr.Dataset:
    """Return one product group: a ``dtype`` variable plus any scalar coordinates."""
    if name == "restriction":
        (data,) = compute(array.data)
    else:
        (dense,) = materialize_together(array, purpose=f"{name} serialization")
        data = dense.data
    if isinstance(data, SparseArray):
        data = COO(data)
        if data.fill_value != 0:
            raise ValueError(f"A sparse {name} must use a zero fill value")
    values = data.data if isinstance(data, COO) else np.asarray(data)
    if not (np.issubdtype(values.dtype, np.floating) or np.issubdtype(values.dtype, np.integer)):
        raise ValueError(f"{name} must contain real numbers; got dtype {values.dtype}")

    attrs = {key: value for key, value in array.attrs.items() if key != "units"}
    try:
        encoded_attrs = json.dumps(_json_value(attrs), sort_keys=True)
    except TypeError as exc:
        raise ValueError(f"{name} attributes must be JSON-serializable: {exc}") from exc
    variable_attrs: dict[str, object] = {ATTRS_ATTR: encoded_attrs}
    if "units" in array.attrs:
        variable_attrs["units"] = str(array.attrs["units"])

    scalars = {str(key): array.coords[key].variable for key in _scalar_coordinates(array)}
    if isinstance(data, COO):
        flat = np.ravel_multi_index(data.coords, data.shape).astype(np.int64)
        return xr.Dataset(
            {name: xr.Variable(GATHERED_DIM, values.astype(dtype), attrs={**variable_attrs, STORAGE_ATTR: "gathered"})},
            coords={
                GATHERED_DIM: xr.Variable(GATHERED_DIM, flat, attrs={"compress": " ".join(str(d) for d in array.dims)}),
                **scalars,
            },
        )
    return xr.Dataset(
        {name: xr.Variable(array.dims, values.astype(dtype), attrs={**variable_attrs, STORAGE_ATTR: "dense"})},
        coords=scalars,
    )


def _scalar_coordinates(array: xr.DataArray) -> list[Hashable]:
    """Return the names of zero-dimensional coordinates."""
    return [name for name, coordinate in array.coords.items() if coordinate.ndim == 0]


def _axes_registry(arrays: Mapping[str, xr.DataArray], schema: Mapping[str, tuple[str, ...]]) -> dict[str, Any]:
    """Read logical axis dimensions and column-axis name maps from the products."""
    dims_by_axis: dict[str, Any] = {}
    for name, array in arrays.items():
        logical = schema[name]
        if "native" in logical:
            position = logical.index("native")
            native = [str(dim) for dim in array.dims[position : array.ndim - (len(logical) - 1 - position)]]
            _set_axis(dims_by_axis, "native", native)
            positions = [*range(position), *range(position + len(native), array.ndim)]
            others = [axis for axis in logical if axis != "native"]
        else:
            positions = list(range(array.ndim))
            others = list(logical)
        if len(positions) != len(others):
            raise ValueError(f"{name} does not have the axes {logical!r}")
        for axis, position in zip(others, positions, strict=True):
            _set_axis(dims_by_axis, axis, str(array.dims[position]))

    registry: dict[str, Any] = {
        "native": {"dims": dims_by_axis["native"]},
        "state": {"dim": dims_by_axis["state"]},
        "observation": {"dim": dims_by_axis["observation"]},
    }
    for column, row in (("state_column", "state"), ("observation_column", "observation")):
        if column not in dims_by_axis:
            continue
        source = next(array for name, array in arrays.items() if column in schema[name])
        registry[column] = {
            "dim": dims_by_axis[column],
            "mirrors": row,
            "names": _mirror_names(source, row_dim=dims_by_axis[row], column_dim=dims_by_axis[column]),
        }
    return registry


def _set_axis(dims_by_axis: dict[str, Any], axis: str, dims: Any) -> None:
    """Record an axis's dimensions, requiring every product to agree."""
    if dims_by_axis.setdefault(axis, dims) != dims:
        raise ValueError(f"Products disagree on the {axis} axis: {dims_by_axis[axis]!r} and {dims!r}")


def _mirror_names(array: xr.DataArray, *, row_dim: str, column_dim: str) -> dict[str, str]:
    """Map row-axis coordinate names to column-axis names by the labelled-matrix convention.

    MultiIndex levels map by position. Auxiliary coordinate ``name`` maps to
    ``name_cov`` or ``name_cov_<n>``, the names
    :func:`~openghg_inversions._labelled_matrices.renamed_column_coordinates`
    produces.
    """
    names = {row_dim: column_dim}
    row_index = array.get_index(row_dim) if row_dim in array.indexes else None
    column_index = array.get_index(column_dim) if column_dim in array.indexes else None
    row_levels = list(row_index.names) if isinstance(row_index, pd.MultiIndex) else []
    column_levels = list(column_index.names) if isinstance(column_index, pd.MultiIndex) else []
    if len(row_levels) != len(column_levels):
        raise ValueError(f"Column axis {column_dim!r} does not mirror the index of {row_dim!r}")
    names.update(zip((str(level) for level in row_levels), (str(level) for level in column_levels), strict=True))

    def auxiliary(dim: str, levels: list[Any]) -> list[str]:
        """Return the names of 1-D non-index coordinates on ``dim``."""
        return [
            str(name)
            for name, coordinate in array.coords.items()
            if coordinate.dims == (dim,) and name != dim and name not in levels
        ]

    unmatched = set(auxiliary(column_dim, column_levels))
    for name in auxiliary(row_dim, row_levels):
        pattern = re.compile(re.escape(f"{name}_cov") + r"(_\d+)?")
        candidates = sorted(candidate for candidate in unmatched if pattern.fullmatch(candidate))
        if len(candidates) != 1:
            raise ValueError(f"Column axis {column_dim!r} has no unique mirror of coordinate {name!r}")
        names[name] = candidates[0]
        unmatched.discard(candidates[0])
    if unmatched:
        raise ValueError(f"Column axis {column_dim!r} has unmirrored coordinate {sorted(unmatched)[0]!r}")
    return names


def _expected_dims(logical: tuple[str, ...], registry: Mapping[str, Any]) -> tuple[str, ...]:
    """Expand logical axes to dimension names."""
    dims: list[str] = []
    for axis in logical:
        if axis == "native":
            dims.extend(registry["native"]["dims"])
        elif axis not in registry:
            raise ValueError(f"Logical axis {axis!r} is not declared")
        else:
            dims.append(registry[axis]["dim"])
    return tuple(dims)


def _check_mirrors(axes: xr.Dataset, registry: Mapping[str, Any]) -> None:
    """Require each column axis to copy its row axis's labels, order and attributes."""
    for column in ("state_column", "observation_column"):
        if column not in registry:
            continue
        entry = registry[column]
        row_dim = registry[entry["mirrors"]]["dim"]
        column_dim = entry["dim"]
        names = entry["names"]
        if names.get(row_dim) != column_dim:
            raise ValueError(f"Column axis {column_dim!r} must map {row_dim!r} to itself")
        row_names = {
            str(name) for name, coordinate in axes.coords.items() if coordinate.dims == (row_dim,)
        }
        column_names = {
            str(name) for name, coordinate in axes.coords.items() if coordinate.dims == (column_dim,)
        }
        if set(names) != row_names or set(names.values()) != column_names:
            raise ValueError(f"Column axis {column_dim!r} does not mirror every coordinate of {row_dim!r}")
        for row_name, column_name in names.items():
            row, mirror = axes.coords[row_name], axes.coords[column_name]
            if not np.array_equal(row.values, mirror.values) or row.attrs != mirror.attrs:
                raise ValueError(f"Column coordinate {column_name!r} is not an exact copy of {row_name!r}")


def _check_label_types(axes: xr.Dataset) -> None:
    """Reject labels that NetCDF cannot store with their type."""
    for name, variable in axes.variables.items():
        if variable.dtype.kind == "O" and not isinstance(axes.indexes.get(name), pd.MultiIndex):
            if not all(isinstance(value, str) for value in np.ravel(variable.values)):
                raise ValueError(f"Coordinate {name!r} must contain typed labels; object labels must be strings")


# --- Decoding -----------------------------------------------------------------------------


def _check_root(tree: xr.DataTree, schema: str, attrs: set[str], *, axes_node: str) -> None:
    """Validate the root header and children before decoding anything."""
    if tree.attrs.get("schema") != schema:
        raise ValueError(f"Expected schema {schema!r}; got {tree.attrs.get('schema')!r}")
    if tree.attrs.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported {schema} schema_version {tree.attrs.get('schema_version')!r}")
    _require_keys(set(tree.attrs), attrs, "root attribute")
    _require_keys(set(tree.children), {"covariance", axes_node}, "root child")
    if tree.to_dataset(inherit=False).variables:
        raise ValueError("The artifact root must not contain variables")


def _decode_covariance(node: xr.DataTree) -> SerializableCovariance:
    """Restore the embedded covariance action through its own schema."""
    if node.children:
        raise ValueError("Unexpected covariance child group")
    dataset = node.to_dataset(inherit=False)
    covariance_type = _COVARIANCE_TYPES.get(str(dataset.attrs.get("schema")))
    if covariance_type is None:
        raise ValueError(f"Unsupported covariance configuration schema {dataset.attrs.get('schema')!r}")
    return covariance_type.from_dataset(dataset)


def _decode_axes_node(
    node: xr.DataTree,
    schema: Mapping[str, tuple[str, ...]],
    *,
    optional: set[str],
) -> dict[str, xr.DataArray]:
    """Validate the axes node and its product groups, returning labelled products."""
    raw_axes = node.to_dataset(inherit=False)
    _require_keys(set(raw_axes.attrs), {AXES_ATTR} | ({MULTIINDEX_DIMS_ATTR} & set(raw_axes.attrs)), "axes attribute")
    if raw_axes.data_vars:
        raise ValueError("The axes node must not contain data variables")
    registry = _decode_json_object(raw_axes.attrs[AXES_ATTR], AXES_ATTR)
    axes = restore_declared_multiindexes(raw_axes, strict=MULTIINDEX_DIMS_ATTR in raw_axes.attrs)
    axes.attrs = {}
    _check_registry(registry, axes)
    _check_mirrors(axes, registry)

    present = set(node.children)
    _require_keys(present, set(schema) - (optional - present), "product group")
    arrays = {}
    for name in present:
        expected_dims = _expected_dims(schema[name], registry)
        arrays[name] = _decode_product(node.children[name], name, expected_dims, axes)
    return arrays


def _check_registry(registry: Mapping[str, Any], axes: xr.Dataset) -> None:
    """Validate the logical-axis declarations against the stored axes."""
    required = {"native", "state", "observation"}
    _require_keys(set(registry), required | ({"state_column", "observation_column"} & set(registry)), "logical axis")
    try:
        native = registry["native"]["dims"]
        dims = [*native, *(registry[axis]["dim"] for axis in registry if axis != "native")]
        for column in ("state_column", "observation_column"):
            if column in registry and registry[column]["mirrors"] != column.removesuffix("_column"):
                raise ValueError(f"{column} must mirror {column.removesuffix('_column')}")
            if column in registry and not isinstance(registry[column]["names"], dict):
                raise ValueError(f"{column} names must be an object")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed {AXES_ATTR} declaration") from exc
    if not all(isinstance(dim, str) and dim for dim in dims) or len(set(dims)) != len(dims):
        raise ValueError(f"{AXES_ATTR} must declare distinct non-empty dimension names")
    _require_keys(set(map(str, axes.dims)), set(dims), "axis dimension")
    for dim in dims:
        if dim not in axes.indexes:
            raise ValueError(f"Axis dimension {dim!r} has no index coordinate")


def _decode_product(node: xr.DataTree, name: str, dims: tuple[str, ...], axes: xr.Dataset) -> xr.DataArray:
    """Validate one product group and return its labelled DataArray."""
    if node.children:
        raise ValueError(f"Unexpected child group in {name!r}")
    dataset = node.to_dataset(inherit=False)
    if dataset.attrs:
        raise ValueError(f"Unexpected attribute on product group {name!r}")
    _require_keys(set(dataset.data_vars), {name}, f"{name} variable")
    variable = dataset[name].variable
    allowed_attrs = {ATTRS_ATTR, STORAGE_ATTR} | ({"units"} & set(variable.attrs))
    _require_keys(set(variable.attrs), allowed_attrs, f"{name} attribute")
    allowed_dtypes = _STORAGE_DTYPES.get(name, _DEFAULT_STORAGE_DTYPES)
    if variable.dtype.name not in allowed_dtypes:
        raise ValueError(f"{name} must be stored as {' or '.join(sorted(allowed_dtypes))}; got {variable.dtype}")
    attrs = dict(_decode_json_object(variable.attrs[ATTRS_ATTR], f"{name} attributes"))
    if "units" in variable.attrs:
        attrs["units"] = str(variable.attrs["units"])

    storage = variable.attrs[STORAGE_ATTR]
    coordinate_names = {str(key) for key in dataset.coords}
    scalars = {key for key in coordinate_names if dataset.coords[key].ndim == 0}
    shape = tuple(axes.sizes[dim] for dim in dims)
    if storage == "gathered":
        if name != "restriction":
            raise ValueError(f"Only the restriction may use gathered storage; {name!r} does")
        _require_keys(coordinate_names - scalars, {GATHERED_DIM}, f"{name} coordinate")
        if variable.dims != (GATHERED_DIM,):
            raise ValueError(f"Gathered {name} must lie on {GATHERED_DIM!r}")
        indices = dataset.coords[GATHERED_DIM]
        if indices.attrs != {"compress": " ".join(dims)}:
            raise ValueError(f"Gathered {name} must compress dimensions {' '.join(dims)!r}")
        flat = np.asarray(indices.values)
        size = int(np.prod(shape))
        if flat.dtype.kind not in "iu" or (flat.size and (flat[0] < 0 or flat[-1] >= size)):
            raise ValueError(f"Gathered {name} indices must be integers within the restriction shape")
        if flat.size > 1 and not np.all(np.diff(flat) > 0):
            raise ValueError(f"Gathered {name} indices must be unique and increasing")
        data: Any = COO(np.vstack(np.unravel_index(flat, shape)), np.asarray(variable.values), shape=shape)
    elif storage == "dense":
        if coordinate_names - scalars:
            raise ValueError(f"Product group {name!r} must not carry axis coordinates")
        if variable.dims != dims:
            raise ValueError(f"{name} must have dimensions {dims!r}; got {variable.dims!r}")
        data = np.asarray(variable.values)
    else:
        raise ValueError(f"Unsupported {name} storage {storage!r}")

    on_dims = axes.drop_vars([key for key, coordinate in axes.coords.items() if not set(coordinate.dims) <= set(dims)])
    array = xr.DataArray(data, dims=dims, coords=on_dims.coords, name=name, attrs=attrs)
    return array.assign_coords({key: dataset.coords[key] for key in scalars})


def _check_covariance_binding(covariance: SerializableCovariance, restriction: xr.DataArray) -> None:
    """Require the embedded covariance to act on exactly the stored native axes."""
    native_dims = tuple(str(dim) for dim in restriction.dims[1:])
    if tuple(covariance.native_dims) != native_dims:
        raise ValueError(
            f"Embedded covariance acts on {tuple(covariance.native_dims)!r}, not the stored native axes {native_dims!r}"
        )
    configured = covariance.to_dataset()
    for dim in native_dims:
        if dim not in configured.indexes or not configured.indexes[dim].equals(restriction.get_index(dim)):
            raise ValueError(f"Embedded covariance labels on {dim!r} differ from the stored native axis")


def _check_units(arrays: Mapping[str, xr.DataArray]) -> None:
    """Require dimensionless scaling products and units consistent with ``H``."""
    for name in _DIMENSIONLESS & set(arrays):
        if arrays[name].attrs.get("units") != "1":
            raise ValueError(f"{name} must have units '1'")
    linear = {arrays[name].attrs.get("units") for name in _LINEAR_IN_H & set(arrays)}
    if len(linear) != 1:
        raise ValueError("Products linear in H must share one units attribute (or all omit it)")
    (units,) = linear
    expected = f"({units})^2" if units is not None else None
    for name in _QUADRATIC_IN_H & set(arrays):
        if arrays[name].attrs.get("units") != expected:
            raise ValueError(f"{name} must have units {expected!r}")


def _check_numerics(arrays: Mapping[str, xr.DataArray], *, square: str, diagonal: str | None) -> None:
    """Check finite values, a symmetric positive-definite ``C_alpha`` and a nonnegative diagonal."""
    for name, array in arrays.items():
        stored = array.data.data if isinstance(array.data, COO) else np.asarray(array.data)
        if not np.all(np.isfinite(stored)):
            raise ValueError(f"{name} must contain only finite values")
    matrix = np.asarray(arrays[square].values)
    scale = max(1.0, float(np.max(np.abs(matrix))) if matrix.size else 1.0)
    if matrix.size and float(np.max(np.abs(matrix - matrix.T))) > 1e-10 * scale:
        raise ValueError(f"{square} must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{square} must be positive definite") from exc
    if diagonal is not None and np.any(np.asarray(arrays[diagonal].values) < 0.0):
        raise ValueError(f"{diagonal} must be nonnegative")


def _check_identities(
    tree: xr.DataTree,
    artifact: Artifact,
    *,
    include_prolongation: bool,
    prolongation_dtype: ProlongationDtype,
) -> None:
    """Require the stored identities to match the decoded content."""
    if tree.attrs["source_id"] != source_id(artifact):
        raise ValueError("Stored source_id does not match the artifact's covariance, restriction and inputs_id")
    stored_content = content_id(
        artifact, include_prolongation=include_prolongation, prolongation_dtype=prolongation_dtype
    )
    if tree.attrs["content_id"] != stored_content:
        raise ValueError("Stored content_id does not match the artifact content; the file was altered")


# --- Digests and small helpers ------------------------------------------------------------


def _digest_tree(tree: xr.DataTree) -> str:
    """Digest every node of a tree except the root ``content_id``."""
    datasets = []
    for node in sorted(tree.subtree, key=lambda item: item.path):
        dataset = node.to_dataset(inherit=False)
        if node.is_root:
            dataset.attrs = {key: value for key, value in dataset.attrs.items() if key != "content_id"}
        datasets.append((node.path, dataset))
    return "sha256:" + _digest_datasets(datasets)


def _digest_datasets(datasets: Iterable[tuple[str, xr.Dataset]]) -> str:
    """Return a canonical hexadecimal SHA-256 digest of named datasets.

    Each variable contributes its name, dimensions, shape, a type tag, its
    values normalized by kind and its attributes as canonical JSON. Floats
    keep their width, integers use little-endian 64-bit bytes, dates use
    int64 nanoseconds and strings use length-prefixed UTF-8, so integer
    widths and string representations that differ only in spelling hash
    alike.
    """
    digest = sha256()

    def update(text: str) -> None:
        encoded = text.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    for path, dataset in datasets:
        update(path)
        update(json.dumps(_json_value(dict(dataset.attrs)), sort_keys=True))
        for name in sorted(dataset.variables, key=str):
            variable = dataset.variables[name]
            update(str(name))
            update(json.dumps([[str(dim) for dim in variable.dims], list(variable.shape)]))
            update(json.dumps(_json_value(dict(variable.attrs)), sort_keys=True))
            tag, payload = _normalized_values(np.asarray(variable.values), str(name))
            update(tag)
            digest.update(payload)
    return digest.hexdigest()


def _normalized_values(values: np.ndarray, name: str) -> tuple[str, bytes]:
    """Return a type tag and canonical bytes for an array's values."""
    kind = values.dtype.kind
    if kind == "f":
        # Keep the width: float32 and float64 storage of U_* are different content.
        width = 4 if values.dtype.itemsize <= 4 else 8
        return f"float{8 * width}", values.astype(f"<f{width}").tobytes()
    if kind in "iu":
        return "int", values.astype("<i8").tobytes()
    if kind == "b":
        return "bool", values.astype("u1").tobytes()
    if kind == "M":
        return "datetime", values.astype("datetime64[ns]").astype("<i8").tobytes()
    if kind == "m":
        return "timedelta", values.astype("timedelta64[ns]").astype("<i8").tobytes()
    # "T" is NumPy's variable-width StringDType, which newer Zarr and xarray
    # versions return for string labels.
    if kind in "USOT":
        chunks = []
        for item in np.ravel(values):
            if isinstance(item, bytes):
                item = item.decode()
            if not isinstance(item, str):
                raise ValueError(f"Variable {name!r} holds a {type(item).__name__} label; labels must be typed")
            encoded = item.encode()
            chunks.append(len(encoded).to_bytes(8, "little") + encoded)
        return "str", b"".join(chunks)
    raise ValueError(f"Variable {name!r} has unsupported dtype {values.dtype}")


def _observation_dim(artifact: Artifact) -> str:
    """Return the observation dimension of an artifact."""
    operator = (
        artifact.products.effective_observation_operator
        if isinstance(artifact, NativeCovarianceProductsArtifact)
        else artifact.reduction.effective_observation_operator
    )
    return str(operator.dims[0])


def _check_binding_fields(covariance: object, inputs_id: object) -> None:
    """Validate the covariance type and ``inputs_id`` of an artifact."""
    if not isinstance(covariance, SeparableExponentialCovariance | IndependentSourceCovariance):
        raise TypeError(
            "covariance must be a SeparableExponentialCovariance or IndependentSourceCovariance, "
            "whose configuration can be embedded"
        )
    _nonempty_string(inputs_id, "inputs_id")


def _nonempty_string(value: object, name: str) -> str:
    """Return ``value`` if it is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _frozen_json_object(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return a read-only JSON-object copy of ``value``."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return MappingProxyType(_copy_json_mapping(value, name, sort_keys=True))


def _decode_json_object(raw: object, name: str) -> Mapping[str, Any]:
    """Decode a JSON object attribute."""
    try:
        value = json.loads(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_keys(actual: set[Any], expected: set[str], context: str) -> None:
    """Require exactly the expected keys, naming the first difference."""
    actual = {str(key) for key in actual}
    extra = actual - expected
    missing = expected - actual
    if extra:
        raise ValueError(f"Unexpected {context} {sorted(extra)[0]!r}")
    if missing:
        raise ValueError(f"Missing {context} {sorted(missing)[0]!r}")

