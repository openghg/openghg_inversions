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
from typing import ClassVar, Literal, Self

import numpy as np
import xarray as xr

from openghg_inversions.array_ops import (
    concat_gather_data_arrays,
    force_align,
    get_xr_dummies,
)

# ----------------------------
# Registry
# ----------------------------

_BASIS_OPERATOR_REGISTRY: dict[str, type[BasisOperator]] = {}


def register_basis_operator(kind: str):
    """Decorator to register BasisOperator subclasses for DataTree deserialisation."""

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
    """Minimal metadata needed to apply the operator.

    grid_dims are the dims to dot over when computing sensitivities.
    state_dim is the canonical output dim for the reduced state vector.
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

    def sensitivity(self, fp_x_flux: xr.DataArray) -> xr.DataArray:
        """Compute H = fp_x_flux dot basis_matrix over grid dims."""
        B = force_align(self.basis_matrix, fp_x_flux, dims=list(self.meta.grid_dims))
        B = B.transpose(*self.meta.grid_dims, ...)

        # xr.dot keeps non-dot dims from both arguments
        H = xr.dot(fp_x_flux, B, dim=list(self.meta.grid_dims)).as_numpy()

        # canonical ordering (state_dim first if present)
        if self.meta.state_dim in H.dims:
            if "time" in H.dims:
                H = H.transpose(self.meta.state_dim, "time", ...)
            else:
                H = H.transpose(self.meta.state_dim, ...)
        return H

    def interpolate(self, state: xr.DataArray, weights: xr.DataArray | None = None) -> xr.DataArray:
        """Map from state -> grid by multiplying the dummy matrix by state.

        If weights is provided (e.g. flux), use (basis_matrix * weights) as the interpolation matrix.
        """
        if self.meta.state_dim not in state.dims:
            raise ValueError(f"State dim '{self.meta.state_dim}' missing from state dims {state.dims}")
        M = self.basis_matrix
        if weights is not None:
            weights_aligned = force_align(weights, M, dims=list(self.meta.grid_dims))
            M = M * weights_aligned
        return xr.dot(M, state, dim=self.meta.state_dim)

    # ---- DataTree IO ----

    @abstractmethod
    def to_datatree(self) -> xr.DataTree:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_datatree(cls, dt: xr.DataTree) -> Self:
        raise NotImplementedError

    @classmethod
    def decode_datatree(cls, dt: xr.DataTree) -> BasisOperator:
        """Generic dispatcher from DataTree -> concrete BasisOperator."""
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
# Concrete operators
# ----------------------------

# By convention, basis region numbering starts at 1, but for a while the
# region coordinate started at 0;
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
    ):
        meta = meta or BasisMeta()
        if state_dim is not None:
            meta = BasisMeta(grid_dims=meta.grid_dims, state_dim=state_dim)
        self._meta = meta

        # store canonical 2D basis
        self.basis_flat = basis_flat.isel(time=0, drop=True) if "time" in basis_flat.dims else basis_flat
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
        return self._meta

    @property
    def basis_matrix(self) -> xr.DataArray:
        return self._basis_matrix

    def _apply_region_labels_policy(self, mat: xr.DataArray) -> xr.DataArray:
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
    ):
        meta = meta or BasisMeta()
        if state_dim is not None:
            meta = BasisMeta(grid_dims=meta.grid_dims, state_dim=state_dim)
        self._meta = meta

        if not basis_flat:
            raise ValueError("basis_flat dict is empty.")

        self.source_dim = source_dim
        self.region_in_source_dim = region_in_source_dim

        # Canonicalise: 2D, consistent lat/lon assumed for now.
        self.basis_flat = {
            k: (v.isel(time=0, drop=True) if "time" in v.dims else v) for k, v in basis_flat.items()
        }

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
        return self._meta

    @property
    def basis_matrix(self) -> xr.DataArray:
        return self._basis_matrix

    # ---- DataTree IO ----

    def to_datatree(self) -> xr.DataTree:
        # root: metadata only (empty dataset OK)
        dt = xr.DataTree(xr.Dataset())
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
        basis_group = xr.DataTree(xr.Dataset())
        dt["basis_flat"] = basis_group

        for src, bf in self.basis_flat.items():
            basis_group[src] = xr.DataTree(xr.Dataset({"basis_flat": bf}))

        return dt

    @classmethod
    def from_datatree(cls, dt: xr.DataTree) -> Self:
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
