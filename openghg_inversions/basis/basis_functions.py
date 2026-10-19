"""Flux-weighted basis objects and their durable representation.

``FluxWeightedBasis`` (also exported as ``BasisFunctions``) pairs a retained
flux field with a basis operator. The operator owns the grid-to-state mapping,
including region and optional source ordering; the flux remains separate so
the same object can project sensitivities and reconstruct gridded fluxes.

NetCDF and Zarr persistence uses a versioned DataTree with separate ``basis``
and ``flux`` children. Only namespaced OpenGHG Inversions metadata is retained,
encoded in the root ``basis_metadata_json`` attribute. Legacy raw namespaced
root attributes remain readable; a present JSON attribute must be valid.
Loading reconstructs the operator and preserves its grid, state, region, and
source coordinates. A shared flux may omit the source dimension; when source
labels are present, they must exactly match the operator's labels.
Source-specific flux arrays are stacked only when their native time indexes
match exactly; callers must explicitly resample heterogeneous frequencies
first.

Example:
    Project a cached footprint-times-flux field into state space::

        def apply_basis_functions(ds: xr.Dataset, bf: BasisFunctions) -> xr.Dataset:
            if "fp_x_flux" not in ds:
                return ds
            return bf.sensitivity(ds.fp_x_flux).rename("H").to_dataset()
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, cast

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from openghg_inversions._serialization_codecs import _numpy_scalar_json_default
from openghg_inversions.array_ops import (
    force_align,
    require_unique_index,
    same_index,
)
from openghg_inversions.basis.operators import (
    BasisOperator,
    BucketBasisOperator,
    MultiSourceBucketBasisOperator,
    RegionLabels,
)
from openghg_inversions.flux_sanitization import sanitize_flux_nonfinite
from openghg_inversions.serialization import open_datatree_loaded, save_datatree
from openghg_inversions.native_flux import _stack_flux_sources_with_alignment

BASIS_METADATA_ATTR_PREFIX = "openghg_inversions:"
_BASIS_METADATA_JSON_ATTR = "basis_metadata_json"
BASIS_ARTIFACT_SOURCE_ATTR = f"{BASIS_METADATA_ATTR_PREFIX}basis_artifact_source"
BASIS_ARTIFACT_PATH_ATTR = f"{BASIS_METADATA_ATTR_PREFIX}basis_artifact_path"
_BASIS_ARTIFACT_SOURCE_FALLBACK_ATTR = "basis_artifact_source"
_BASIS_ARTIFACT_PATH_FALLBACK_ATTR = "basis_artifact_path"


@dataclass(frozen=True, slots=True)
class FluxWeightedBasis:
    """Pair a grid-to-state operator with a native prior flux field.

    Flux retains its inventory time axis independently of observation-aligned
    responses passed to :meth:`sensitivity`. Arrays are borrowed; construction
    preserves lazy flux payloads and does not make independent snapshots.
    Source-specific operators own source order: indexed flux labels must match
    their labels and are reordered accordingly.

    Attributes:
        operator: Basis operator that owns the state/grid mapping.
        flux: Native prior flux on the operator grid. Callers must supply
            compatible spatial coordinates. May include ``time`` and ``source``.
        metadata: Namespaced metadata carried through DataTree serialization.
            Values may be JSON scalars (including ``None`` and finite floats),
            nested lists, or dictionaries with string keys. NumPy counterparts
            of supported JSON scalars are normalized to Python scalars and
            tuples to lists when saved.
            Other values and non-string dictionary keys are rejected.

    Notes:
        - Namespaced metadata is stored in the ``basis_metadata_json`` root
          attribute. Both loading routes also accept legacy raw namespaced
          attributes; malformed JSON metadata is rejected when present.
        - A flux with a `source` dimension can still be paired with a `BucketBasisOperator`; in that case
          the same basis is applied to each source when broadcasting occurs in downstream operations.
        - Spatial alignment and broadcasting are enforced by the operations
          that combine flux, operator geometry, and state.
    """

    operator: BasisOperator
    flux: xr.DataArray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate flux source labels and normalize source-specific order."""
        if "source" not in self.flux.dims:
            return
        _validated_flux_source_labels(self.flux)
        if isinstance(self.operator, MultiSourceBucketBasisOperator):
            object.__setattr__(
                self,
                "flux",
                _reorder_flux_to_operator_sources(self.flux, self.operator.source_labels),
            )

    @property
    def source_labels(self) -> tuple[str, ...] | None:
        """Return the authoritative source labels when this basis is source-aware.

        Source-specific operators own their source order. A shared operator may
        instead obtain source labels from its retained flux. A completely
        source-independent basis returns ``None``.

        Returns:
            Ordered source labels, or ``None`` when neither the operator nor
            flux is source-indexed.
        """
        if isinstance(self.operator, MultiSourceBucketBasisOperator):
            return self.operator.source_labels
        if "source" in self.flux.dims:
            return _validated_flux_source_labels(self.flux)
        return None

    @classmethod
    def from_flat_basis(
        cls,
        basis_flat: xr.DataArray,
        flux: xr.DataArray,
        *,
        region_labels: RegionLabels = "range0",
        operator_kwargs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FluxWeightedBasis:
        """Construct from a single-source (standard) flattened basis array.

        Args:
            basis_flat: Flattened basis labels on the inversion grid, e.g. dims like (lat, lon)
                (or whatever grid dims the operator expects). Values should label regions.
            flux: Flux on the same grid dims as `basis_flat`. May optionally contain extra dims such
                as `time` or `source`; these will be carried along by downstream operations.
            region_labels: Policy for the output state coordinate labels:
                - `"range0"`: `0..N-1` (legacy-friendly)
                - `"range1"`: `1..N`
                - `"basis_values"`: use the unique positive labels found in `basis_flat`.
            operator_kwargs: Optional kwargs forwarded to `BucketBasisOperator`.
            metadata: Optional namespaced metadata to carry on the basis object.

        Returns:
            A FluxWeightedBasis pairing a BucketBasisOperator with the provided flux.
        """
        kwargs: dict[str, Any] = dict(operator_kwargs or {})
        if region_labels is not None:
            kwargs["region_labels"] = region_labels

        operator_cls = cast(Any, BucketBasisOperator)
        operator = operator_cls(basis_flat=basis_flat, **kwargs)
        flux = sanitize_flux_nonfinite(flux, context="retained basis construction")
        return cls(operator=operator, flux=flux, metadata=dict(metadata or {}))

    @classmethod
    def from_multi_source_flat_basis(
        cls,
        basis_flat: Mapping[str, xr.DataArray],
        flux: xr.DataArray | Mapping[str, xr.DataArray],
        *,
        operator_kwargs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FluxWeightedBasis:
        """Construct from a multi-source flattened basis mapping.

        Args:
            basis_flat: Mapping from source name to that source's flattened basis labels on the grid.
                Each value should be a DataArray on the inversion grid (e.g. dims like (lat, lon)).
            flux: Flux on the inversion grid. A shared DataArray may omit the
                ``source`` dimension. When ``source`` is present, its unique
                labels must match ``basis_flat``. A mapping is stacked along
                ``source`` after requiring identical time indexes.
            operator_kwargs: Optional kwargs forwarded to `MultiSourceBucketBasisOperator`.
            metadata: Optional namespaced metadata to carry on the basis object.

        Returns:
            A FluxWeightedBasis pairing a MultiSourceBucketBasisOperator with the provided flux.

        Raises:
            ValueError: If present flux source labels are duplicated or do not
                match the basis operator, or mapped fluxes have incompatible
                time indexes.
        """
        kwargs: dict[str, Any] = dict(operator_kwargs or {})

        operator_cls = cast(Any, MultiSourceBucketBasisOperator)
        operator = operator_cls(basis_flat=dict(basis_flat), **kwargs)

        if isinstance(flux, xr.DataArray):
            trust_attrs = True
        else:
            flux = _stack_flux_sources_with_alignment(flux)
            trust_attrs = False

        flux = sanitize_flux_nonfinite(
            flux,
            context="retained multisource basis construction",
            trust_attrs=trust_attrs,
        )
        return cls(operator=operator, flux=flux, metadata=dict(metadata or {}))

    def with_flux(self, flux: xr.DataArray) -> Self:
        """Return a copy with the same operator and metadata but a different flux."""
        flux = sanitize_flux_nonfinite(flux, context="retained basis flux replacement")
        return type(self)(operator=self.operator, flux=flux, metadata=dict(self.metadata))

    def with_metadata(self, metadata: Mapping[str, Any]) -> Self:
        """Return a copy with additional metadata."""
        return type(self)(
            operator=self.operator,
            flux=self.flux,
            metadata={**self.metadata, **dict(metadata)},
        )

    def validated(self) -> Self:
        """Return a copy that revalidates mutable xarray-backed state.

        Returns:
            A basis object reconstructed from the current operator, flux, and
            metadata values.
        """
        return type(self)(
            operator=self.operator,
            flux=self.flux,
            metadata=dict(self.metadata),
        )

    def select_sources(self, sources: Sequence[str]) -> Self:
        """Return a copy restricted to source-specific basis functions.

        This is used when a retained artifact contains more sources than the
        current run requested. Single-source/shared-basis operators are kept as
        is, with only flux restricted when it has a ``source`` dimension.

        Args:
            sources: Source names and order to keep.

        Returns:
            A basis object restricted to ``sources``.

        Raises:
            ValueError: If a source-specific basis is missing a requested
                source.
        """
        source_order = list(sources)
        flux = self.flux
        if "source" in flux.dims:
            flux = flux.sel(source=source_order)
        return type(self)(
            operator=_select_operator_sources(self.operator, source_order),
            flux=flux,
            metadata=dict(self.metadata),
        )

    @property
    def basis_artifact_source(self) -> str | None:
        """Return the source used to create/load this basis, when recorded."""
        source = self.metadata.get(
            BASIS_ARTIFACT_SOURCE_ATTR, self.metadata.get(_BASIS_ARTIFACT_SOURCE_FALLBACK_ATTR)
        )
        return str(source) if source is not None else None

    @property
    def basis_artifact_path(self) -> str | None:
        """Return the basis artifact path used to create/load this basis, when recorded."""
        path = self.metadata.get(
            BASIS_ARTIFACT_PATH_ATTR, self.metadata.get(_BASIS_ARTIFACT_PATH_FALLBACK_ATTR)
        )
        return str(path) if path is not None else None

    def for_source(self, source: str, *, state_dim: str | None = None) -> Self:
        """Return a source-specific basis object for postprocessing.

        Shared-basis operators are reused with the selected source flux. Operators
        that own source-specific basis maps can expose an ``operator_for_source``
        method to return a single-source operator without going through the
        legacy flat-basis view.

        Args:
            source: Flux source label to select from source-specific flux and
                basis operators.
            state_dim: Optional state dimension for the selected source
                operator. When omitted, the operator chooses its default.

        Returns:
            A basis object with the selected source flux and a compatible
            operator.

        Raises:
            ValueError: If the underlying source-specific operator does not
                contain ``source``.
        """
        flux = self.flux.sel(source=source, drop=True) if "source" in self.flux.dims else self.flux
        operator_for_source = getattr(self.operator, "operator_for_source", None)
        if callable(operator_for_source):
            operator = cast(BasisOperator, operator_for_source(source, state_dim=state_dim))
            return type(self)(operator=operator, flux=flux, metadata=dict(self.metadata))
        return type(self)(operator=self.operator, flux=flux, metadata=dict(self.metadata))

    def flat_basis(self) -> xr.DataArray | dict[str, xr.DataArray]:
        """Return the flattened basis view used by legacy basis artifacts."""
        basis_flat = getattr(self.operator, "basis_flat", None)
        if isinstance(basis_flat, xr.DataArray):
            return basis_flat.rename("basis")
        if isinstance(basis_flat, dict):
            return {key: value.rename("basis") for key, value in basis_flat.items()}
        raise AttributeError("Operator does not expose `basis_flat`.")

    def to_datatree(self) -> xr.DataTree:
        """Serialise to a DataTree with `basis` and `flux` groups.

        Root namespaced metadata is encoded as JSON according to the class's
        metadata contract, normalizing supported NumPy scalars and tuples.
        Array payloads remain borrowed and lazy.

        Returns:
            A DataTree with:
              - `basis`: BasisOperator DataTree (via operator.to_datatree()).
              - `flux`: a Dataset containing the flux DataArray as variable `flux`.

        Raises:
            TypeError: If namespaced metadata contains unsupported values or
                non-string dictionary keys.
            ValueError: If namespaced metadata contains non-finite floats.
        """
        basis_functions = self.validated()
        dt_basis = basis_functions.operator.to_datatree()

        # Store flux as a dataset to preserve name/attrs cleanly.
        flux = basis_functions.flux.rename("flux")
        dt_flux = xr.DataTree(xr.Dataset({"flux": flux}))

        dt_dict = {"basis": dt_basis, "flux": dt_flux}

        dt = xr.DataTree.from_dict(dt_dict)
        dt.attrs.update(
            {
                "schema": "openghg_inversions.flux_weighted_basis",
                "schema_version": 1,
            }
        )
        dt.attrs.update(_serialisable_basis_metadata(basis_functions.metadata))
        return dt

    @classmethod
    def from_datatree(cls, dt: xr.DataTree) -> FluxWeightedBasis:
        """Deserialise from a DataTree produced by `to_datatree`.

        Args:
            dt: DataTree with `basis` and `flux` groups.

        Returns:
            A FluxWeightedBasis instance.

        Raises:
            KeyError: If required groups are missing.
            ValueError: If the schema/version is unsupported, the flux
                variable is missing, serialized metadata is invalid, or stored
                multi-source flux labels do not match the basis operator.
        """
        metadata = _basis_artifact_metadata(dt)
        operator = BasisOperator.decode_datatree(cast(xr.DataTree, dt["basis"]))
        ds_flux = dt["flux"].to_dataset()
        if "flux" not in ds_flux:
            raise ValueError("Missing variable 'flux' in dt['flux'] dataset.")
        flux = ds_flux["flux"]
        return cls(operator=operator, flux=flux, metadata=metadata)

    def sensitivity(self, fp_x_flux: xr.DataArray, fillna: bool = True) -> xr.DataArray:
        """Compute sensitivity (grid -> state) via the underlying operator.

        This is typically used to form the reduced Jacobian:
            H = operator.sensitivity(fp_x_flux)

        Args:
            fp_x_flux: Footprints multiplied by flux, on the inversion grid dims.
                May contain extra dims (e.g. time, site, source).
            fillna: if True, fill NaNs in fp_x_flux with 0.0.

        Returns:
            Sensitivity with a `state`-like dimension as defined by the operator.
        """
        return self.operator.sensitivity(fp_x_flux, fillna=fillna)

    def save(self, output_file: str | Path) -> None:
        """Save the basis object as a DataTree artifact.

        Args:
            output_file: Path to a ``.nc`` or ``.zarr`` artifact. NetCDF is used
                for ``.nc`` paths and Zarr is used for ``.zarr`` paths. Suffixes
                are case-insensitive and the requested path is preserved.
                Zarr storage lazily regularizes chunks without changing borrowed
                arrays. Metadata follows the class's JSON value contract.

        Raises:
            TypeError: If retained metadata contains unsupported values or
                non-string dictionary keys.
            ValueError: If ``output_file`` does not end with ``.nc`` or ``.zarr``,
                or retained metadata contains non-finite floats.
        """
        output_file = Path(output_file)
        suffix = output_file.suffix.lower()
        if suffix not in (".nc", ".zarr"):
            raise ValueError(f"BasisFunctions artifact path {output_file} must end in '.nc' or '.zarr'.")
        save_datatree(self.to_datatree(), output_file)

    @classmethod
    def load(cls: type[Self], file_path: str | Path) -> Self:
        """Load a basis object saved with ``BasisFunctions.save``.

        Both the operator and saved flux payloads are loaded eagerly while the
        xarray file context is open, so the returned object retains no closed
        file handles. Use :meth:`reuse_operator` to bind a current native prior
        without reading the saved flux payloads.

        Args:
            file_path: Path to a ``BasisFunctions`` DataTree artifact.

        Returns:
            The reconstructed operator paired with its saved native prior,
            preserving the saved state, grid, and source coordinates.

        Raises:
            OSError: If the artifact cannot be opened.
            KeyError: If a required DataTree group or basis variable is
                missing.
            ValueError: If the serialized schema, metadata, labels, or
                operator state is invalid.
        """
        return cast(Self, cls.from_datatree(open_datatree_loaded(file_path)))

    @classmethod
    def reuse_operator(cls: type[Self], file_path: str | Path, *, flux: xr.DataArray) -> Self:
        """Load a saved operator and bind current native prior flux.

        Only the operator subtree is materialized. Saved flux payloads are not
        read. Runtime sources select and order source-specific partitions;
        missing sources and incompatible grids fail at this binding boundary.
        Use :meth:`load` for reconstruction with the saved retained prior.

        Args:
            file_path: Retained ``.nc`` or ``.zarr`` basis artifact.
            flux: Borrowed current native prior on the operator grid, optionally
                with ``time`` and ``source``. Source labels must be unique
                strings. A source-specific operator must contain each runtime
                source; unused saved sources are omitted. Spatial coordinates
                are matched within
                :func:`~openghg_inversions.array_ops.force_align` tolerances.
                Dask-backed flux payloads remain lazy.

        Returns:
            Saved operator and metadata paired with current flux, with
            non-finite flux values replaced by zero. Runtime source order is
            retained without changing the operator's scientific dimension names.

        Raises:
            OSError: If the artifact cannot be opened.
            KeyError: If a required DataTree group or basis variable is missing.
            ValueError: If artifact schema, metadata, or operator is invalid, its flux
                variable is missing, or current flux source labels or spatial
                coordinates are incompatible.
        """
        engine = "zarr" if Path(file_path).suffix.lower() == ".zarr" else None
        with xr.open_datatree(file_path, engine=engine) as dt:
            metadata = _basis_artifact_metadata(dt)
            if "flux" not in dt["flux"].to_dataset():
                raise ValueError("Missing variable 'flux' in dt['flux'] dataset.")
            dt["basis"].load()
            operator = BasisOperator.decode_datatree(cast(xr.DataTree, dt["basis"]))
        if "source" in flux.dims:
            operator = _select_operator_sources(operator, _validated_flux_source_labels(flux))
        flux = force_align(
            flux,
            reference=operator.basis_matrix.reset_coords(drop=True),
            dims=operator.meta.grid_dims,
        )
        flux = sanitize_flux_nonfinite(flux, context="retained basis runtime flux")
        return cls(operator=operator, flux=flux, metadata=metadata)

    def state_to_native(self, state: xr.DataArray) -> xr.DataArray:
        """Reconstruct linear native scaling with the coarse-to-fine map (prolongation).

        A source-specific basis retains an ordered ``native_source`` dimension
        (or ``native_{source_dim}`` when its source dimension has another name).
        Sum that dimension explicitly when a total scaling grid is needed.
        """
        return self.operator.state_to_native(state)

    def state_to_flux(self, state: xr.DataArray) -> xr.DataArray:
        """Reconstruct signed flux from the retained state and this basis's flux.

        A source-specific basis retains an ordered ``native_source`` dimension
        (or ``native_{source_dim}`` when renamed). A shared basis with
        source-resolved retained flux instead returns a ``source`` dimension.
        Sum the source dimension explicitly when a total flux grid is needed.
        """
        flux = self.flux
        state_dim = self.operator.meta.state_dim
        for grid_dim in self.operator.meta.grid_dims:
            expected = require_unique_index(self.operator.basis_matrix, grid_dim, name="basis")
            if not same_index(require_unique_index(flux, grid_dim, name="flux"), expected):
                raise ValueError(f"flux {grid_dim!r} labels must exactly match the basis.")

        native_source_dim = None
        if isinstance(self.operator, MultiSourceBucketBasisOperator) and self.operator.source_dim in flux.dims:
            native_source_dim = f"native_{self.operator.source_dim}"
            if list(require_unique_index(flux, self.operator.source_dim, name="flux")) != list(
                self.operator.source_labels
            ):
                raise ValueError("flux source labels must exactly match the basis source order.")
            flux = flux.rename({self.operator.source_dim: native_source_dim})

        occupied = set(state.dims) | set(state.coords) | set(flux.dims) | set(flux.coords)
        renames = {}
        for dim in state.dims:
            if dim == state_dim or dim not in flux.dims:
                continue
            base = dim if dim.startswith("state_") else f"state_{dim}"
            candidate = base
            suffix = 2
            while candidate in occupied:
                candidate = f"{base}_{suffix}"
                suffix += 1
            renames[dim] = candidate
            occupied.add(candidate)
        native = self.state_to_native(state.rename(renames))
        if native_source_dim is not None and native_source_dim not in native.dims:
            raise ValueError("source-specific flux requires a source-specific native basis.")
        native, flux = xr.align(native, flux, join="exact", copy=False)
        return (native * flux).rename("flux").assign_attrs(flux.attrs)

    def interpolate(self, state: xr.DataArray, *, flux: bool = False) -> xr.DataArray:
        """Interpolate from state vector to the grid.

        Args:
            state: State vector values with dim matching the operator's state dim (usually `state`).
            flux: If True, apply flux-weighting using `self.flux` as weights. If False, returns the
                unweighted basis interpolation.

        Returns:
            Interpolated gridded array on the operator grid dims, with any non-dot dims preserved.
        """
        warnings.warn(
            "BasisFunctions.interpolate is deprecated; use state_to_native or state_to_flux, "
            "then sum source axes explicitly for total fields.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not flux:
            return self.operator.interpolate(state)

        # TODO: test if _align_source_like_state is needed here
        return self.operator.interpolate(state, weights=self.flux)

    def plot(self, *, shuffle: bool = False, **plot_kwargs: Any) -> Any:
        """Plot basis labels.

        - For single-source basis, returns the result of xarray's plot call.
        - For multi-source basis, creates one subplot per source and returns (fig, axes).

        Args:
            shuffle: If True, randomly permute region labels for visual separation.
            **plot_kwargs: Forwarded to xarray's `.plot()`.

        Returns:
            Plotting object(s). For multi-source, returns (fig, axes).
        """
        basis_flat = getattr(self.operator, "basis_flat", None)
        if basis_flat is None:
            raise AttributeError("Operator does not expose `basis_flat`; cannot plot.")

        if isinstance(basis_flat, xr.DataArray):
            data = basis_flat
            if shuffle:
                data = _shuffle_region_labels(data)
            return data.plot(**plot_kwargs)

        if isinstance(basis_flat, dict):
            sources = sorted(list(basis_flat.keys()))  # sort for stable plotting order
            n = len(sources)
            if n == 0:
                raise ValueError("Empty multi-source basis_flat; nothing to plot.")

            # Simple layout: one row per source.
            fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(7, 3.0 * n), squeeze=False)
            axes_flat = axes[:, 0]

            for ax, source in zip(axes_flat, sources, strict=True):
                data = basis_flat[source]
                if shuffle:
                    data = _shuffle_region_labels(data)
                data.plot(ax=ax, **plot_kwargs)
                ax.set_title(str(source))

            fig.tight_layout()
            return fig, axes_flat

        raise TypeError(f"Unexpected type for operator.basis_flat: {type(basis_flat)!r}")


def basis_functions_from_flat_basis(
    *,
    flux: xr.DataArray,
    basis_flat: xr.DataArray | Mapping[str, xr.DataArray],
    metadata: Mapping[str, Any] | None = None,
) -> FluxWeightedBasis:
    """Attach resolved native flux to shared or source-specific flat labels.

    Source-specific labels are selected in the flux source order. The caller
    owns source combination, native time alignment, and spatial compatibility.
    Constructors validate source labels. Inputs remain borrowed and flux
    payloads may remain lazy; partition geometry is materialized by the operator.
    Prepared basis state coordinates use ``region``.

    Args:
        flux: Resolved native prior on the partition grid, optionally with
            ``time`` and ``source``. Non-finite values are replaced with zero.
        basis_flat: Shared spatial region labels, or source-keyed partitions.
            For source-indexed flux, every runtime source must have a partition;
            extra partitions are omitted and runtime source order is retained.
            A flux without ``source`` is shared across all supplied partitions.
        metadata: Optional namespaced metadata retained with the basis object.

    Returns:
        Operator paired with the current native prior, ready for projection
        of observation-aligned responses and gridded flux reconstruction.

    Raises:
        ValueError: If source labels are invalid, a runtime source partition is
            missing, or partition geometry cannot form a valid operator.
    """
    if isinstance(basis_flat, Mapping):
        selected_basis = dict(basis_flat)
        if "source" in flux.dims:
            selected_sources = _validated_flux_source_labels(flux)
            missing_sources = [source for source in selected_sources if source not in selected_basis]
            if missing_sources:
                raise ValueError(f"Multi-source basis is missing runtime flux sources: {missing_sources!r}.")
            selected_basis = {source: selected_basis[source] for source in selected_sources}
        return FluxWeightedBasis.from_multi_source_flat_basis(
            basis_flat=selected_basis,
            flux=flux,
            operator_kwargs={"state_dim": "region"},
            metadata=metadata,
        )

    return FluxWeightedBasis.from_flat_basis(
        basis_flat=basis_flat,
        flux=flux,
        operator_kwargs={"state_dim": "region"},
        metadata=metadata,
    )


def _serialisable_basis_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Encode namespaced JSON metadata in one backend-safe root attribute."""

    def require_string_keys(value: Any) -> None:
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("Basis metadata dictionaries must have string keys.")
            for item in value.values():
                require_string_keys(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                require_string_keys(item)

    retained = {
        key: value
        for key, value in metadata.items()
        if isinstance(key, str) and key.startswith(BASIS_METADATA_ATTR_PREFIX)
    }
    require_string_keys(retained)
    return {
        _BASIS_METADATA_JSON_ATTR: json.dumps(
            retained, default=_numpy_scalar_json_default, allow_nan=False
        )
    }


def _validated_flux_source_labels(flux: xr.DataArray) -> tuple[str, ...]:
    """Return unique string source labels from a source-indexed flux.

    Args:
        flux: Flux expected to have a labelled ``source`` dimension.

    Returns:
        Source coordinate labels in stored order.

    Raises:
        ValueError: If the source dimension is absent or its labels are not
            unique strings.
    """
    if "source" not in flux.dims:
        raise ValueError("Multi-source flux must have a 'source' dimension.")

    source_labels = tuple(flux["source"].values.tolist())
    if not all(isinstance(source, str) for source in source_labels):
        raise ValueError("Multi-source flux labels must all be strings.")
    if len(set(source_labels)) != len(source_labels):
        raise ValueError("Multi-source flux labels must be unique.")
    return source_labels


def _reorder_flux_to_operator_sources(flux: xr.DataArray, operator_sources: tuple[str, ...]) -> xr.DataArray:
    """Validate source membership and reorder flux to canonical operator order.

    Args:
        flux: Source-indexed flux array.
        operator_sources: Canonical ordered source labels from the basis
            operator.

    Returns:
        Flux selected in ``operator_sources`` order.

    Raises:
        ValueError: If flux source labels do not exactly match the operator
            source set.
    """
    flux_sources = _validated_flux_source_labels(flux)
    if set(flux_sources) != set(operator_sources):
        raise ValueError(
            "Multi-source flux labels must exactly match basis source labels; "
            f"got flux={flux_sources!r}, basis={operator_sources!r}."
        )
    return flux.sel(source=list(operator_sources))


def _shuffle_region_labels(basis_flat: xr.DataArray) -> xr.DataArray:
    """Shuffle integer-like region labels for plotting clarity.

    This preserves NaNs/masked values and only permutes the set of unique finite labels.
    """
    values = basis_flat.values
    finite = np.isfinite(values)
    if not np.any(finite):
        return basis_flat

    labels = np.unique(values[finite])
    # Only shuffle if we have at least 2 labels.
    if labels.size < 2:
        return basis_flat

    perm = labels.copy()
    rng = np.random.default_rng()
    rng.shuffle(perm)
    mapping = dict(zip(labels.tolist(), perm.tolist(), strict=True))

    shuffled = values.copy()
    for old, new in mapping.items():
        shuffled[values == old] = new

    return basis_flat.copy(data=shuffled)


# Friendly alias
BasisFunctions = FluxWeightedBasis


def _basis_artifact_metadata(dt: xr.DataTree) -> dict[str, Any]:
    """Validate the retained schema and restore namespaced metadata."""
    schema = dt.attrs.get("schema", None)
    version = dt.attrs.get("schema_version", None)
    if schema is not None and schema != "openghg_inversions.flux_weighted_basis":
        raise ValueError(f"Unexpected schema: {schema!r}")
    if version is not None and int(version) != 1:
        raise ValueError(f"Unexpected schema_version: {version!r}")

    if "basis" not in dt:
        raise KeyError("Missing 'basis' group in DataTree.")
    if "flux" not in dt:
        raise KeyError("Missing 'flux' group in DataTree.")

    if not isinstance(dt["basis"], xr.DataTree):
        raise ValueError("'basis' is not an xr.DataTree.")

    if _BASIS_METADATA_JSON_ATTR in dt.attrs:
        try:
            decoded_metadata = json.loads(dt.attrs[_BASIS_METADATA_JSON_ATTR])
            if not isinstance(decoded_metadata, dict) or not all(
                key.startswith(BASIS_METADATA_ATTR_PREFIX) for key in decoded_metadata
            ):
                raise ValueError("Expected a dictionary of namespaced basis metadata.")
            # Reject non-finite JSON numbers accepted by Python's decoder.
            json.dumps(decoded_metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid serialized basis metadata JSON.") from error
        return decoded_metadata

    metadata: dict[str, Any] = {}
    for key, value in dt.attrs.items():
        key = str(key)
        if key.startswith(BASIS_METADATA_ATTR_PREFIX):
            metadata[key] = value

    return metadata


def _select_operator_sources(operator: BasisOperator, sources: Sequence[str]) -> BasisOperator:
    """Select source-specific partitions while preserving scientific dimensions."""
    if not isinstance(operator, MultiSourceBucketBasisOperator):
        return operator
    missing = [source for source in sources if source not in operator.basis_flat]
    if missing:
        raise ValueError(f"BasisFunctions object is missing sources required by this run: {missing}")
    return MultiSourceBucketBasisOperator(
        basis_flat={source: operator.basis_flat[source] for source in sources},
        meta=operator.meta,
        source_dim=operator.source_dim,
        region_in_source_dim=operator.region_in_source_dim,
    )
