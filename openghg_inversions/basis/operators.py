"""Basis operator classes (new API; does not yet replace legacy basis_functions.py).

Design goals
------------
1) Separate the *partition/aggregation operator* (basis functions) from any *flux weighting*.
   - The basis operator represents a linear map from a grid (lat/lon) to a reduced state space.
   - Flux weighting (multiplying by flux on the grid, interpolation to maps, covariance transforms)
     is handled by a separate wrapper class (planned: FluxWeightedBasis) so that:
       * sensitivity(fp_x_flux) does not require flux (since fp_x_flux is already precomputed),
       * but flux-aware operations remain available when needed.

2) Canonical "state" dimension.
   - Operators expose a single state dimension (default name: "state").
   - In multisource/multisector cases with ragged per-source region counts, the state coordinate
     becomes a MultiIndex over (source, region_in_source). This avoids padding with zeros.

3) Minimal metadata (BasisMeta).
   - We only need to know which dims to dot over (grid_dims) and the state_dim name.
   - Any special alignment hacks are implemented in concrete subclasses rather than inferred from metadata.

4) Serialization via xarray.DataTree.
   - BasisOperator.to_datatree() returns a self-describing DataTree with schema/kind/version attrs.
   - BasisOperator.decode_datatree(dt) dispatches to the correct registered subclass based on dt.attrs["kind"].
   - For multisource operators, the canonical serialized representation stores per-source flat basis arrays
     under dt["basis_flat"][<source>], keeping storage compact and natural.

How to use
----------
- Construct a basis operator:
    op = BucketBasisOperator(basis_flat)                         # single-sector
    op = MultiSourceBucketBasisOperator({"a": bf_a, "b": bf_b})   # ragged multisource

- Compute sensitivities:
    H = op.sensitivity(fp_x_flux)

  where fp_x_flux is an xarray.DataArray with at least the grid dims (lat, lon), and typically time.
  In multisource workflows, fp_x_flux often has a separate dimension "source". The multisource operator
  implements an alignment/broadcast hack so that the fp_x_flux source dimension can be matched against
  the MultiIndex level "source" stored on the state coordinate.

- Serialize/deserialize:
    dt = op.to_datatree()
    op2 = BasisOperator.decode_datatree(dt)

Notes:
-----
- This module is an initial step in refactoring. It is intended to coexist with the legacy
  basis_functions.py until the old APIs are reimplemented using these operators and tests are migrated.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal
from typing_extensions import Self

import numpy as np
import xarray as xr

from openghg_inversions.array_ops import (
    align_to_multi_index_level_values,
    concat_gather_data_arrays,
    force_align,
    get_xr_dummies,
)

# ----------------------------
# Registry
# ----------------------------

_BASIS_OPERATOR_REGISTRY: dict[str, type[BasisOperator]] = {}


def register_basis_operator(kind: str):
    """Registers a `BasisOperator` subclass for DataTree deserialisation.

    This decorator builds a small module-level registry mapping a stable `kind`
    string (stored in `dt.attrs["kind"]`) to a concrete `BasisOperator` subclass.

    Args:
        kind: Stable key identifying the operator type on disk. This is written to
            `dt.attrs["kind"]` by `BasisOperator.to_datatree()` and used by
            `BasisOperator.decode_datatree()`.

    Returns:
        A class decorator that registers the decorated class under `kind`.

    Raises:
        ValueError: If `kind` is already registered to a different class.
    """

    def _decorator(cls: type[BasisOperator]) -> type[BasisOperator]:
        if kind in _BASIS_OPERATOR_REGISTRY and _BASIS_OPERATOR_REGISTRY[kind] is not cls:
            raise ValueError(
                f"BasisOperator kind '{kind}' already registered with {_BASIS_OPERATOR_REGISTRY[kind]}"
            )
        _BASIS_OPERATOR_REGISTRY[kind] = cls
        cls.kind = kind  # type: ignore[attr-defined]
        return cls

    return _decorator


def get_basis_operator_class(kind: str) -> type[BasisOperator]:
    """Looks up a registered `BasisOperator` subclass.

    Args:
        kind: Registry key for the operator type (e.g. `"bucket"`).

    Returns:
        The registered `BasisOperator` subclass.

    Raises:
        KeyError: If `kind` is not registered.
    """
    try:
        return _BASIS_OPERATOR_REGISTRY[kind]
    except KeyError as e:
        raise KeyError(
            f"Unknown BasisOperator kind '{kind}'. " f"Known kinds: {sorted(_BASIS_OPERATOR_REGISTRY)}"
        ) from e


# ----------------------------
# Metadata
# ----------------------------


@dataclass(frozen=True)
class BasisMeta:
    """Metadata describing how to apply a basis operator.

    The intent is to keep this minimal: we only store what is needed for the
    default implementations of `BasisOperator.sensitivity` and
    `BasisOperator.interpolate`.

    Attributes:
        grid_dims: Dimensions to dot over when reducing a gridded quantity to the
            reduced state (typically `("lat", "lon")`).
        state_dim: Canonical dimension name for the reduced state axis.
    """

    grid_dims: tuple[str, ...] = ("lat", "lon")
    state_dim: str = "state"


# ----------------------------
# Base class
# ----------------------------


class BasisOperator(ABC):
    """Abstract basis operator.

    Concrete subclasses must define:
    - meta (grid dims + state dim)
    - basis_matrix: one-hot/dummy matrix with dims (*grid_dims, state_dim)
    - to_datatree / from_datatree

    The default sensitivity implementation assumes fp_x_flux has the grid dims and
    any extra dims (e.g. time, source) are preserved.
    """

    # stable kind string used for serialization dispatch
    kind: ClassVar[str]

    schema: ClassVar[str] = "openghg_inversions.basis_operator"
    schema_version: ClassVar[int] = 1

    @property
    @abstractmethod
    def meta(self) -> BasisMeta:
        """Operator metadata (must be provided by subclasses)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def basis_matrix(self) -> xr.DataArray:
        """Dummy matrix mapping grid -> state.

        Expected dims: (*grid_dims, state_dim)
        (state_dim may be a MultiIndex coordinate)
        """
        raise NotImplementedError

    def sensitivity(self, fp_x_flux: xr.DataArray, fillna: bool = True) -> xr.DataArray:
        """Computes the sensitivity matrix ("H") by dotting over the grid.

        This implements the common bucket-basis reduction:

        - `fp_x_flux` is a gridded quantity with dimensions that include
          `meta.grid_dims` (typically `lat` and `lon`) and usually a `time` dimension.
        - `basis_matrix` is a one-hot/dummy matrix that maps each grid cell to exactly
          one basis region/state.

        The returned array keeps all non-grid dimensions from `fp_x_flux` and includes
        the reduced state dimension `meta.state_dim`.

        Args:
            fp_x_flux: Footprint x flux array to reduce. Must contain all
                `meta.grid_dims`.
            fillna: if True, fill NaNs in fp_x_flux with 0.0.

        Returns:
            Sensitivity matrix with dimension `meta.state_dim` and any remaining
            non-grid dimensions (e.g. `time`).
        """
        mat_aligned = force_align(self.basis_matrix, fp_x_flux, dims=list(self.meta.grid_dims))
        mat_aligned = mat_aligned.transpose(*self.meta.grid_dims, ...)

        # xr.dot keeps non-dot dims from both arguments
        if fillna:
            h = xr.dot(fp_x_flux.fillna(0.0), mat_aligned, dim=list(self.meta.grid_dims)).as_numpy()
        else:
            h = xr.dot(fp_x_flux, mat_aligned, dim=list(self.meta.grid_dims)).as_numpy()

        # canonical ordering (state_dim first if present)
        if self.meta.state_dim in h.dims:
            if "time" in h.dims:
                h = h.transpose(self.meta.state_dim, "time", ...)
            else:
                h = h.transpose(self.meta.state_dim, ...)
        return h

    def interpolate(self, state: xr.DataArray, weights: xr.DataArray | None = None) -> xr.DataArray:
        """Interpolates/reconstructs a gridded field from a state vector.

        This maps from the reduced basis space back to the grid by multiplying the
        basis dummy matrix by a state vector.

        If `weights` is provided (e.g. a flux field on the grid), it is multiplied
        elementwise with the basis matrix before interpolation. This corresponds to
        using a flux-weighted interpolation operator.

        Args:
            state: State vector with dimension `meta.state_dim`.
            weights: Optional gridded weights with dimensions matching `meta.grid_dims`
                (and broadcastable to `basis_matrix`).

        Returns:
            Reconstructed gridded field with dimensions including `meta.grid_dims`.

        Raises:
            ValueError: If `meta.state_dim` is not a dimension of `state`.
        """
        if self.meta.state_dim not in state.dims:
            raise ValueError(f"State dim '{self.meta.state_dim}' missing from state dims {state.dims}")
        mat = self.basis_matrix
        if weights is not None:
            weights_aligned = force_align(weights, mat, dims=list(self.meta.grid_dims))
            mat = mat * weights_aligned
        return xr.dot(mat, state, dim=self.meta.state_dim)

    # ---- DataTree IO ----

    @abstractmethod
    def to_datatree(self) -> xr.DataTree:
        """Serialises this operator to an `xarray.DataTree`.

        The returned DataTree is intended to be self-describing. It must include enough
        information to allow round-tripping via `BasisOperator.decode_datatree()`.

        Returns:
            DataTree representation of the operator.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_datatree(cls, dt: xr.DataTree) -> Self:
        """Constructs an operator instance from an `xarray.DataTree`.

        Concrete subclasses implement this to load whatever canonical representation
        they write in `to_datatree()`.

        Args:
            dt: DataTree created by `to_datatree()`.

        Returns:
            An instance of the operator.
        """
        raise NotImplementedError

    @classmethod
    def decode_datatree(cls, dt: xr.DataTree) -> BasisOperator:
        """Dispatches a DataTree to the correct registered operator subclass.

        Args:
            dt: DataTree representation of a basis operator.

        Returns:
            A concrete `BasisOperator` instance.

        Raises:
            ValueError: If the schema or schema version is unsupported.
            KeyError: If the `kind` is not registered.
        """
        schema = dt.attrs.get("schema")
        if schema != cls.schema:
            raise ValueError(f"Unexpected schema '{schema}', expected '{cls.schema}'")

        version = int(dt.attrs.get("schema_version", -1))
        if version != cls.schema_version:
            raise ValueError(f"Unsupported schema_version {version}; expected {cls.schema_version}")

        kind = dt.attrs.get("kind")
        if not kind:
            raise ValueError("Missing dt.attrs['kind'] for BasisOperator dispatch.")

        op_cls = get_basis_operator_class(str(kind))
        return op_cls.from_datatree(dt)


# ----------------------------
# Helper functions
# ----------------------------

def drop_singleton_time(da: xr.DataArray, *, name: str = "basis_flat") -> xr.DataArray:
    """Drop a singleton ``time`` dimension if present; otherwise raise.

    This is a strict helper intended for basis operators that assume a 2D basis over
    the grid dims. It avoids silently discarding time-varying basis information.

    Args:
        da: Input DataArray which may or may not have a ``time`` dimension.
        name: Label used in error messages to identify what is being checked.

    Returns:
        ``da`` with ``time`` removed if it exists and has length 1, otherwise ``da``
        unchanged.

    Raises:
        ValueError: If ``time`` exists and has length not equal to 1.
    """
    if "time" not in da.dims:
        return da

    time_size = da.sizes.get("time")
    if time_size != 1:
        raise ValueError(
            f"{name} has a non-singleton 'time' dimension (size={time_size}); "
            "cannot drop time without losing time-varying basis information. "
            "Please pass a 2D basis without 'time'."
        )

    # Use squeeze to remove the dim (and drop the coordinate variable if it becomes scalar).
    return da.squeeze("time", drop=True)


# ----------------------------
# Concrete operators
# ----------------------------

# By convention, basis region numbering starts at 1, but some code had
# a region coordinate started at 0. You can choose a range index starting
# at 0 (i.e. 0, 1, ..., N-1) or at 1 (i.e. 1, 2, ..., N), or "basis_values"
# which will use whatever values are in the flat basis array values.
RegionLabels = Literal["range0", "range1", "basis_values"]


@register_basis_operator("bucket")
class BucketBasisOperator(BasisOperator):
    """Single flat bucket basis: basis_flat(lat, lon) with integer region labels.

    Stores basis_flat and constructs basis_matrix via get_xr_dummies.
    """

    def __init__(
        self,
        basis_flat: xr.DataArray,
        *,
        meta: BasisMeta | None = None,
        state_dim: str | None = None,
        region_labels: RegionLabels = "range0",
        chunks: dict[str, int] | None = None,
    ) -> None:
        """Creates a single-source bucket basis operator.

        Args:
            basis_flat: Integer-labelled basis array on the grid (typically `(lat, lon)`).
                If a singleton `time` dimension is present, it is dropped.
            meta: Metadata describing grid and state dimension names.
            state_dim: Optional override of `meta.state_dim`.
            region_labels: Policy for the output state coordinate labels:
                - `"range0"`: `0..N-1` (legacy-friendly)
                - `"range1"`: `1..N`
                - `"basis_values"`: use the unique positive labels found in `basis_flat`.
            chunks: Optional chunking to apply to the basis matrix.
        """
        meta = meta or BasisMeta()
        if state_dim is not None:
            meta = BasisMeta(grid_dims=meta.grid_dims, state_dim=state_dim)
        self._meta = meta

        # store canonical 2D basis
        self.basis_flat = drop_singleton_time(basis_flat, name="basis_flat")
        self.basis_flat = self.basis_flat.rename("basis_flat")

        self.region_labels = region_labels

        # create dummy matrix (grid -> state)
        # cat_dim name must match meta.state_dim
        mat = get_xr_dummies(self.basis_flat, cat_dim=self.meta.state_dim)

        # optionally override state coordinate policy
        mat = self._apply_region_labels_policy(mat)

        # chunking
        mat = mat.chunk(chunks) if chunks is not None else mat.chunk()

        self._basis_matrix = mat

    @property
    def meta(self) -> BasisMeta:
        """Basis metadata."""
        return self._meta

    @property
    def basis_matrix(self) -> xr.DataArray:
        """Basis matrix."""
        return self._basis_matrix

    def _apply_region_labels_policy(self, mat: xr.DataArray) -> xr.DataArray:
        """Applies the configured `region_labels` policy to the state coordinate.

        Args:
            mat: Dummy matrix returned by `get_xr_dummies`.

        Returns:
            `mat` with an updated state coordinate.

        Notes:
            This assumes `get_xr_dummies` orders categories in ascending order of the
            unique positive labels in `basis_flat`.
        """
        # Determine basis label values present (assume positive ints, often 1..N)
        labels = np.unique(self.basis_flat.values.astype(int))
        labels = labels[labels > 0]
        n = len(labels)

        if self.region_labels == "range0":
            coord = np.arange(n, dtype=int)
        elif self.region_labels == "range1":
            coord = np.arange(1, n + 1, dtype=int)
        elif self.region_labels == "basis_values":
            coord = labels.astype(int)
        else:
            raise ValueError(f"Unknown region_labels policy: {self.region_labels}")

        # Assign coordinate. Important: this assumes get_xr_dummies created state columns
        # ordered by sorted unique labels > 0. If that assumption changes, we must reindex.
        return mat.assign_coords({self.meta.state_dim: coord})

    # ---- DataTree IO ----

    def to_datatree(self) -> xr.DataTree:
        """Serialises the operator to a DataTree.

        Returns:
            A DataTree with a dataset containing `basis_flat` and attributes sufficient
            to reconstruct the operator.
        """
        ds = xr.Dataset({"basis_flat": self.basis_flat})
        dt = xr.DataTree(ds)
        dt.attrs.update(
            {
                "schema": self.schema,
                "schema_version": self.schema_version,
                "kind": self.kind,
                "grid_dims": self.meta.grid_dims,
                "state_dim": self.meta.state_dim,
                "region_labels": self.region_labels,
            }
        )
        return dt

    @classmethod
    def from_datatree(cls, dt: xr.DataTree) -> Self:
        """Deserialises a `BucketBasisOperator` from a DataTree.

        Args:
            dt: DataTree produced by `BucketBasisOperator.to_datatree()`.

        Returns:
            A reconstructed `BucketBasisOperator`.
        """
        ds = dt.to_dataset()

        basis_flat = ds["basis_flat"]
        meta = BasisMeta(
            grid_dims=tuple(dt.attrs.get("grid_dims", ("lat", "lon"))),
            state_dim=str(dt.attrs.get("state_dim", "state")),
        )
        region_labels = str(dt.attrs.get("region_labels", "range0"))

        return cls(
            basis_flat=basis_flat,
            meta=meta,
            region_labels=region_labels,  # type: ignore[arg-type]
        )


@register_basis_operator("multisource_bucket")
class MultiSourceBucketBasisOperator(BasisOperator):
    """Multiple flat bases keyed by source, with potentially ragged region counts.

    Canonical state_dim is a gathered MultiIndex over (source, region_in_source).
    """

    def __init__(
        self,
        basis_flat: dict[str, xr.DataArray],
        *,
        meta: BasisMeta | None = None,
        source_dim: str = "source",
        region_in_source_dim: str = "region_in_source",
        state_dim: str | None = None,
        chunks: dict[str, int] | None = None,
    ) -> None:
        """Creates a multisource bucket basis operator with ragged per-source regions.

        The canonical state dimension is a gathered MultiIndex over
        `(source, region_in_source)`, stored on the single dimension `meta.state_dim`.

        Args:
            basis_flat: Mapping from source name to a 2D integer-labelled basis array
                (typically `(lat, lon)`).
            meta: Metadata describing grid and state dimension names.
            source_dim: Name of the source dimension/level.
            region_in_source_dim: Name for the per-source region index level.
            state_dim: Optional override of `meta.state_dim`.
            chunks: Optional chunking to apply to the gathered basis matrix.

        Raises:
            ValueError: If `basis_flat` is empty.
        """
        meta = meta or BasisMeta()
        if state_dim is not None:
            meta = BasisMeta(grid_dims=meta.grid_dims, state_dim=state_dim)
        self._meta = meta

        if not basis_flat:
            raise ValueError("basis_flat dict is empty.")

        self.source_dim = source_dim
        self.region_in_source_dim = region_in_source_dim

        # Canonicalise: 2D, consistent lat/lon assumed for now.
        self.basis_flat = {k: drop_singleton_time(v, name=f"basis_flat[{k!r}]") for k, v in basis_flat.items()}
        self.basis_flat = {k: v.rename("basis_flat") for k, v in self.basis_flat.items()}

        # Build per-source dummy matrices with ragged region_in_source dim
        mats: dict[str, xr.DataArray] = {}
        for src, bf in self.basis_flat.items():
            # use region_in_source_dim so we can gather it
            mats[src] = get_xr_dummies(bf, cat_dim=self.region_in_source_dim)

        # Gather concat over source + region_in_source_dim into state_dim
        # Result has dims (*grid_dims, state_dim) and state_dim is a MultiIndex
        mat = concat_gather_data_arrays(
            mats,
            key_dim=self.source_dim,
            ragged_dim=self.region_in_source_dim,
            stack_dim=self.meta.state_dim,
        )

        # chunking
        mat = mat.chunk(chunks) if chunks is not None else mat.chunk()

        self._basis_matrix = mat

    @property
    def meta(self) -> BasisMeta:
        """Basis metadata."""
        return self._meta

    @property
    def basis_matrix(self) -> xr.DataArray:
        """Basis matrix."""
        return self._basis_matrix

    def _align_source_like_state(self, other: xr.DataArray) -> xr.DataArray:
        """Broadcast `other(source, ...)` onto `state` using the state MultiIndex level `source`.

        This implements the "source alignment hack" used in the earlier prototype:
        - basis_matrix has dim `state` with MultiIndex levels (source, region_in_source)
        - fp_x_flux has dim `source`
        - to multiply fp_x_flux by basis_matrix we need to index fp_x_flux by the state.source order

        Returns an array indexed by `state` and no longer carrying the original `source` coord var.
        """
        if self.source_dim not in other.dims:
            return other

        state_index = self.basis_matrix[self.meta.state_dim]

        return align_to_multi_index_level_values(
            other,
            multi_index=state_index,
            multi_dim=self.meta.state_dim,
            level=self.source_dim,
            other_dim=self.source_dim,
        )

    def sensitivity(self, fp_x_flux: xr.DataArray, fillna: bool = True) -> xr.DataArray:
        """Compute sensitivity for multisource fp_x_flux.

        Overrides base method to broadcast the fp_x_flux `source` dim onto the gathered `state` dim.
        """
        mat_aligned = force_align(self.basis_matrix, fp_x_flux, dims=list(self.meta.grid_dims))
        mat_aligned = mat_aligned.transpose(*self.meta.grid_dims, ...)

        fp_on_state = self._align_source_like_state(fp_x_flux)

        if fillna:
            h = xr.dot(fp_on_state.fillna(0.0), mat_aligned, dim=list(self.meta.grid_dims)).as_numpy()
        else:
            h = xr.dot(fp_on_state, mat_aligned, dim=list(self.meta.grid_dims)).as_numpy()

        if self.meta.state_dim in h.dims:
            if "time" in h.dims:
                h = h.transpose(self.meta.state_dim, "time", ...)
            else:
                h = h.transpose(self.meta.state_dim, ...)
        return h

    def interpolate(self, state: xr.DataArray, weights: xr.DataArray | None = None) -> xr.DataArray:
        """Interpolate/reconstruct a gridded field from a state vector.

        For MultiSourceBucketBasisOperator, `weights` may include a `source_dim` that is also a
        level name in the gathered MultiIndex on `meta.state_dim`. In that case we broadcast
        weights along the gathered state axis (repeating per-source weights across all regions
        within that source) by replacing `source_dim` with `meta.state_dim`.

        The state vector itself is expected to be defined on `meta.state_dim` and should not
        include a separate coordinate named like a MultiIndex level (e.g. `source_dim`).

        Args:
            state_array: State vector defined on `meta.state_dim`.
            weights: Optional gridded weights (e.g. prior fluxes) on `meta.grid_dims`. May
                optionally include `source_dim` for per-source weights.

        Returns:
            Gridded reconstructed field on `meta.grid_dims`.
        """
        if self.meta.state_dim not in state.dims:
            raise ValueError(
                f"Expected state_array to have dim '{self.meta.state_dim}', got dims {state.dims}"
            )

        mat = self.basis_matrix

        if weights is not None:
            weights_aligned = force_align(weights, mat, dims=list(self.meta.grid_dims))

            # broadcast source if necessary
            weights_aligned = self._align_source_like_state(weights_aligned)

            mat = mat * weights_aligned

        out = xr.dot(mat, state, dim=self.meta.state_dim).as_numpy()
        out = out.transpose(*self.meta.grid_dims, ...)
        return out

    # ---- DataTree IO ----

    def to_datatree(self) -> xr.DataTree:
        """Serialises the multisource operator to a DataTree.

        The returned DataTree stores per-source `basis_flat` arrays as children under
        `dt["basis_flat"][<source>]`.

        Returns:
            DataTree representation of the operator.
        """
        # root: metadata only (empty dataset OK)
        dt = xr.DataTree()
        dt.attrs.update(
            {
                "schema": self.schema,
                "schema_version": self.schema_version,
                "kind": self.kind,
                "grid_dims": self.meta.grid_dims,
                "state_dim": self.meta.state_dim,
                "source_dim": self.source_dim,
                "region_in_source_dim": self.region_in_source_dim,
            }
        )

        # store basis_flat per source as children
        dt["basis_flat"] = xr.DataTree.from_dict(
            {src: xr.Dataset({"basis_flat": bf}) for src, bf in self.basis_flat.items()}
        )

        return dt

    @classmethod
    def from_datatree(cls, dt: xr.DataTree) -> Self:
        """Deserialises a `MultiSourceBucketBasisOperator` from a DataTree.

        Args:
            dt: DataTree produced by `MultiSourceBucketBasisOperator.to_datatree()`.

        Returns:
            A reconstructed `MultiSourceBucketBasisOperator`.
        """
        meta = BasisMeta(
            grid_dims=tuple(dt.attrs.get("grid_dims", ("lat", "lon"))),
            state_dim=str(dt.attrs.get("state_dim", "state")),
        )
        source_dim = str(dt.attrs.get("source_dim", "source"))
        region_in_source_dim = str(dt.attrs.get("region_in_source_dim", "region_in_source"))

        if "basis_flat" not in dt:
            raise ValueError("Expected child node 'basis_flat' in DataTree.")

        basis_node = dt["basis_flat"]
        basis_flat: dict[str, xr.DataArray] = {}
        for src, child in basis_node.items():
            ds = child.to_dataset()
            if "basis_flat" not in ds:
                raise ValueError(f"Missing 'basis_flat' variable for source '{src}'.")
            basis_flat[str(src)] = ds["basis_flat"]

        return cls(
            basis_flat=basis_flat,
            meta=meta,
            source_dim=source_dim,
            region_in_source_dim=region_in_source_dim,
        )
