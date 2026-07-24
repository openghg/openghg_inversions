"""Flux-weighted basis objects and their durable representation.

``FluxWeightedBasis`` (also exported as ``BasisFunctions``) pairs a retained
flux field with a basis operator. The operator owns the grid-to-state mapping,
including region and optional source ordering; the flux remains separate so
the same object can project sensitivities and reconstruct gridded fluxes.

NetCDF and Zarr persistence uses a versioned DataTree with separate ``basis``
and ``flux`` children. Only namespaced OpenGHG Inversions metadata is retained.
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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from openghg.analyse._utils import match_dataset_dims, stack_datasets
from typing_extensions import Self

from openghg_inversions.array_ops import (
    force_align,
)
from openghg_inversions.basis.operators import (
    BasisOperator,
    BucketBasisOperator,
    MultiSourceBucketBasisOperator,
    RegionLabels,
)
from openghg_inversions.flux_sanitization import sanitize_flux_nonfinite

BASIS_METADATA_ATTR_PREFIX = "openghg_inversions:"
BASIS_ARTIFACT_SOURCE_ATTR = f"{BASIS_METADATA_ATTR_PREFIX}basis_artifact_source"
BASIS_ARTIFACT_PATH_ATTR = f"{BASIS_METADATA_ATTR_PREFIX}basis_artifact_path"
_BASIS_ARTIFACT_SOURCE_FALLBACK_ATTR = "basis_artifact_source"
_BASIS_ARTIFACT_PATH_FALLBACK_ATTR = "basis_artifact_path"


@dataclass(frozen=True, slots=True)
class FluxWeightedBasis:
    """A thin wrapper pairing a BasisOperator with a flux field.

    This class is intentionally lightweight: it stores the operator and the flux, and provides
    constructors that build the appropriate operator from a basis representation.

    Attributes:
        operator: Basis operator that owns the state/grid mapping.
        flux: Representative flux field associated with the basis.
        metadata: Namespaced metadata carried through DataTree serialization.

    Notes:
        - A flux with a `source` dimension can still be paired with a `BucketBasisOperator`; in that case
          the same basis is applied to each source when broadcasting occurs in downstream operations.
        - Alignment/broadcasting details are currently delegated to the operator implementations.
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
        basis_flat = self.flat_basis()
        flux = self.flux
        if "source" in flux.dims:
            flux = flux.sel(source=source_order)

        if not isinstance(basis_flat, dict):
            return self.with_flux(flux)

        missing = [source for source in source_order if source not in basis_flat]
        if missing:
            raise ValueError(f"BasisFunctions object is missing sources required by this run: {missing}")

        operator_kwargs: dict[str, Any] = {"state_dim": self.operator.meta.state_dim}
        source_dim = getattr(self.operator, "source_dim", None)
        region_in_source_dim = getattr(self.operator, "region_in_source_dim", None)
        if source_dim is not None:
            operator_kwargs["source_dim"] = source_dim
        if region_in_source_dim is not None:
            operator_kwargs["region_in_source_dim"] = region_in_source_dim

        return cast(
            Self,
            type(self).from_multi_source_flat_basis(
                basis_flat={source: basis_flat[source] for source in source_order},
                flux=flux,
                operator_kwargs=operator_kwargs,
                metadata=self.metadata,
            ),
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
        """Return the legacy flattened basis view used by ``fp_sensitivity``."""
        basis_flat = getattr(self.operator, "basis_flat", None)
        if isinstance(basis_flat, xr.DataArray):
            return basis_flat.rename("basis")
        if isinstance(basis_flat, dict):
            return {key: value.rename("basis") for key, value in basis_flat.items()}
        raise AttributeError("Operator does not expose `basis_flat`.")

    def to_datatree(self) -> xr.DataTree:
        """Serialise to a DataTree with `basis` and `flux` groups.

        Returns:
            A DataTree with:
              - `basis`: BasisOperator DataTree (via operator.to_datatree()).
              - `flux`: a Dataset containing the flux DataArray as variable `flux`.

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
                variable is missing, or stored multi-source flux labels do not
                match the basis operator.
        """
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

        operator = BasisOperator.decode_datatree(dt["basis"])  # type: ignore [arg-type]

        ds_flux = dt["flux"].to_dataset()
        if "flux" not in ds_flux:
            raise ValueError("Missing variable 'flux' in dt['flux'] dataset.")
        flux = ds_flux["flux"]
        metadata: dict[str, Any] = {}
        for key, value in dt.attrs.items():
            key = str(key)
            if key.startswith(BASIS_METADATA_ATTR_PREFIX):
                metadata[key] = value

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
                for ``.nc`` paths and Zarr is used for ``.zarr`` paths.

        Raises:
            ValueError: If ``output_file`` does not end with ``.nc`` or ``.zarr``.
        """
        output_file = Path(output_file)
        suffix = output_file.suffix.lower()
        if suffix == ".nc":
            self.to_datatree().to_netcdf(output_file)
        elif suffix == ".zarr":
            self.to_datatree().to_zarr(output_file)
        else:
            raise ValueError(f"BasisFunctions artifact path {output_file} must end in '.nc' or '.zarr'.")

    @classmethod
    def load(cls: type[Self], file_path: str | Path) -> Self:
        """Load a basis object saved with ``BasisFunctions.save``.

        The DataTree is loaded while the xarray file context is open, so the
        returned ``BasisFunctions`` object does not retain references to closed
        file handles.

        Args:
            file_path: Path to a ``BasisFunctions`` DataTree artifact.

        Returns:
            The reconstructed basis object.

        Raises:
            OSError: If the artifact cannot be opened.
            KeyError: If a required DataTree group or basis variable is
                missing.
            ValueError: If the serialized schema, metadata, labels, or
                operator state is invalid.
        """
        with xr.open_datatree(file_path) as dt:
            return cast(Self, cls.from_datatree(dt.load()))

    def interpolate(self, state: xr.DataArray, *, flux: bool = False) -> xr.DataArray:
        """Interpolate from state vector to the grid.

        Args:
            state: State vector values with dim matching the operator's state dim (usually `state`).
            flux: If True, apply flux-weighting using `self.flux` as weights. If False, returns the
                unweighted basis interpolation.

        Returns:
            Interpolated gridded array on the operator grid dims, with any non-dot dims preserved.
        """
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


def basis_functions_from_fp_all_flat_basis(
    *,
    fp_all: dict,
    basis_flat: xr.DataArray | Mapping[str, xr.DataArray],
    metadata: Mapping[str, Any] | None = None,
) -> FluxWeightedBasis:
    """Compatibility adapter that constructs a basis object from flat basis data.

    Legacy flat basis artifacts remain readable for a transition period, but
    modern preparation and postprocessing should consume the returned
    ``BasisFunctions`` object rather than the flat representation directly.

    Args:
        fp_all: Legacy merged-data dictionary containing a ``".flux"`` side
            channel and optional ``".split_by_sectors"`` flag.
        basis_flat: Flat basis array, or source-keyed flat basis arrays, from
            the legacy basis-generation/loading path.
        metadata: Optional namespaced metadata to carry on the basis object.

    Returns:
        A retained ``BasisFunctions`` object with flux reconstructed from
        ``fp_all``.
    """
    flux = flux_from_fp_all(fp_all)
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


def flux_from_fp_all(fp_all: dict) -> xr.DataArray:
    """Legacy adapter that builds representative flux from ``fp_all``.

    This is a temporary compatibility helper for the current ``fp_sensitivity``
    path. Remove it when issue #429 makes ``BasisFunctions.sensitivity`` the
    source of prepared sensitivities.

    Args:
        fp_all: Legacy merged-data dictionary containing ``fp_all[".flux"]`` and
            optional ``fp_all[".split_by_sectors"]`` metadata.

    Returns:
        A combined flux array for non-sectoral workflows, or a source-stacked
        flux array for sectoral workflows. Source-stacked output follows the
        insertion order of ``fp_all[".flux"]``.

    Raises:
        ValueError: If ``fp_all[".flux"]`` is missing or empty, or a sectoral
            source mapping mixes timed and timeless arrays or contains unequal,
            missing, or duplicate native time coordinates.
        TypeError: If a flux entry cannot be converted to a ``DataArray``.
    """
    if ".flux" not in fp_all or not fp_all[".flux"]:
        raise ValueError("Cannot construct BasisFunctions object: fp_all['.flux'] is missing or empty.")

    flux_entries = fp_all[".flux"]
    flux_arrays = {key: _extract_flux_dataarray(value, flux_key=key) for key, value in flux_entries.items()}

    if _is_multi_source_workflow(fp_all):
        flux = _stack_flux_sources_with_alignment(flux_arrays)
    else:
        flux = _combine_flux_sources_like_modelscenario(flux_arrays)

    return sanitize_flux_nonfinite(
        flux,
        context="retained basis flux from fp_all",
        trust_attrs=len(flux_arrays) == 1,
    )


def _serialisable_basis_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only namespaced metadata that can be stored as DataTree root attributes."""
    return {key: value for key, value in metadata.items() if str(key).startswith(BASIS_METADATA_ATTR_PREFIX)}


def _is_multi_source_workflow(fp_all: dict) -> bool:
    """Interpret legacy ``fp_all`` metadata to determine multi-source mode."""
    split_by_sectors = fp_all.get(".split_by_sectors")
    if split_by_sectors is not None:
        return bool(split_by_sectors)

    flux_entries = fp_all.get(".flux")
    return isinstance(flux_entries, dict) and len(flux_entries) > 1


def _extract_flux_dataarray(flux_entry: object, flux_key: str) -> xr.DataArray:
    """Extract flux from legacy ``fp_all[".flux"]`` entry containers.

    Args:
        flux_entry: Legacy flux entry, xarray ``Dataset``, or ``DataArray``.
        flux_key: Source key used only for error reporting.

    Returns:
        The ``flux`` DataArray.

    Raises:
        TypeError: If the entry does not expose a flux DataArray.
    """
    flux_entry_data = getattr(flux_entry, "data", None)
    if isinstance(flux_entry_data, xr.Dataset) and "flux" in flux_entry_data:
        return flux_entry_data["flux"]
    if isinstance(flux_entry, xr.Dataset) and "flux" in flux_entry:
        return flux_entry["flux"]
    if isinstance(flux_entry, xr.DataArray):
        return flux_entry

    raise TypeError(
        "Could not extract a flux DataArray from fp_all['.flux']. "
        f"Got type {type(flux_entry)!r} for flux entry {flux_key!r}."
    )


def _combine_flux_sources_like_modelscenario(flux_arrays: dict[str, xr.DataArray]) -> xr.DataArray:
    """Legacy adapter that reproduces ``ModelScenario`` flux combination.

    Args:
        flux_arrays: Flux arrays keyed by legacy ``fp_all[".flux"]`` source name.

    Returns:
        A single flux array with non-time grid dimensions aligned and multiple
        non-sectoral sources summed or time-stacked as ``ModelScenario`` did.
    """
    flux_datasets = [
        arr.rename("flux").to_dataset() if arr.name != "flux" else arr.to_dataset()
        for arr in flux_arrays.values()
    ]

    if len(flux_datasets) == 1:
        return flux_datasets[0]["flux"]

    dims = [dim for dim in flux_datasets[0].dims if dim != "time"]
    flux_datasets = match_dataset_dims(flux_datasets, dims=dims)
    if "time" in flux_datasets[0].dims:
        flux_stacked = stack_datasets(flux_datasets, dim="time", method="ffill")
    else:
        flux_stacked = flux_datasets[0]
        for flux_dataset in flux_datasets[1:]:
            flux_stacked = flux_stacked + flux_dataset

    return flux_stacked["flux"]


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


def _require_matching_source_time_indexes(
    flux_arrays: Mapping[str, xr.DataArray],
) -> None:
    """Reject source fluxes whose native time axes cannot be stacked safely.

    Sources may all be timeless or may all use the same exact time index.
    Mixed timed/timeless inputs and unequal indexes require an explicit
    temporal resampling policy and are therefore rejected here.

    Args:
        flux_arrays: Source-labelled flux arrays.

    Raises:
        ValueError: If the mapping is empty, source labels are invalid, or
            time dimensions/indexes are incompatible.
    """
    if not flux_arrays:
        raise ValueError("Cannot stack an empty source-flux mapping.")

    source_labels = tuple(flux_arrays)
    if not all(isinstance(source, str) for source in source_labels):
        raise ValueError("Source-flux mapping labels must all be strings.")
    if len(set(source_labels)) != len(source_labels):
        raise ValueError("Source-flux mapping labels must be unique.")

    timed_sources = {source: "time" in flux.dims for source, flux in flux_arrays.items()}
    if len(set(timed_sources.values())) > 1:
        raise ValueError(
            "Cannot stack timed and timeless source fluxes; explicitly align or "
            "resample every source to a common time index first."
        )
    if not any(timed_sources.values()):
        return

    reference_source = source_labels[0]
    for source in source_labels:
        flux = flux_arrays[source]
        if "time" not in flux.coords or flux["time"].dims != ("time",):
            raise ValueError(
                "Cannot stack source fluxes without an explicit one-dimensional time coordinate; "
                f"source {source!r} has only a positional time dimension."
            )
        if not flux.get_index("time").is_unique:
            raise ValueError(
                "Cannot stack source fluxes with duplicate time coordinates; "
                f"source {source!r} must be made unique first."
            )

    reference_index = flux_arrays[reference_source].get_index("time")
    unequal_sources = [
        source
        for source in source_labels[1:]
        if not flux_arrays[source].get_index("time").equals(reference_index)
    ]
    if unequal_sources:
        raise ValueError(
            "Cannot stack source fluxes with unequal time indexes; explicitly "
            "align or resample every source to a common time index first. "
            f"Reference source {reference_source!r}; unequal sources {unequal_sources!r}."
        )


def _stack_flux_sources_with_alignment(
    flux_arrays: Mapping[str, xr.DataArray],
) -> xr.DataArray:
    """Stack source fluxes after exact temporal and dimensional validation.

    Args:
        flux_arrays: Source-labeled arrays in the output source order. Arrays
            must have identical dimension sets. Timed arrays must all carry the
            same explicit, unique ``time`` index. Non-time grid coordinates are
            aligned to the first source without interpolation.

    Returns:
        Flux with a leading ``source`` dimension in mapping insertion order and
        all remaining dimensions in the first source's order.

    Raises:
        ValueError: If the mapping is empty, labels are invalid, dimension sets
            differ, time coordinates are missing or duplicated, or indexes
            cannot be aligned exactly.
    """
    _require_matching_source_time_indexes(flux_arrays)
    first_key = next(iter(flux_arrays))
    reference = flux_arrays[first_key]
    reference_dims = set(reference.dims)
    dims_to_align = [dim for dim in reference.dims if dim != "time"]

    aligned_flux = {}
    for key, arr in flux_arrays.items():
        if set(arr.dims) != reference_dims:
            raise ValueError(
                "Cannot stack source fluxes with different dimensions; "
                f"reference source {first_key!r} has {reference.dims!r}, "
                f"source {key!r} has {arr.dims!r}."
            )
        aligned_flux[key] = force_align(
            arr.transpose(*reference.dims),
            reference=reference,
            dims=dims_to_align,
        )

    return xr.concat(
        [arr.expand_dims({"source": [key]}) for key, arr in aligned_flux.items()],
        dim="source",
        join="exact",
    )


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
