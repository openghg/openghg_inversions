"""BasisFunctions object to encapsulate representation of basis.

Example usage:

>> def apply_basis_functions(ds: xr.Dataset, bf: BasisFunctions) -> xr.Dataset:
>>     if "fp_x_flux" not in ds:
>>         return ds
>>     return bf.sensitivity(ds.fp_x_flux).rename("H").to_dataset()

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from openghg.analyse._utils import match_dataset_dims, stack_datasets

from openghg_inversions.array_ops import (
    concat_data_arrays,
    force_align,
)
from openghg_inversions.basis.operators import (
    BasisOperator,
    BucketBasisOperator,
    MultiSourceBucketBasisOperator,
    RegionLabels,
)

BASIS_METADATA_ATTR_PREFIX = "openghg_inversions:"
BASIS_ARTIFACT_SOURCE_ATTR = f"{BASIS_METADATA_ATTR_PREFIX}basis_artifact_source"


@dataclass(frozen=True, slots=True)
class FluxWeightedBasis:
    """A thin wrapper pairing a BasisOperator with a flux field.

    This class is intentionally lightweight: it stores the operator and the flux, and provides
    constructors that build the appropriate operator from a basis representation.

    Notes:
        - A flux with a `source` dimension can still be paired with a `BucketBasisOperator`; in that case
          the same basis is applied to each source when broadcasting occurs in downstream operations.
        - Alignment/broadcasting details are currently delegated to the operator implementations.
    """

    operator: BasisOperator
    flux: xr.DataArray
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_basis_flat(
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

        Returns:
            A FluxWeightedBasis pairing a BucketBasisOperator with the provided flux.
        """
        kwargs: dict[str, Any] = dict(operator_kwargs or {})
        if region_labels is not None:
            kwargs["region_labels"] = region_labels

        operator_cls = cast(Any, BucketBasisOperator)
        operator = operator_cls(basis_flat=basis_flat, **kwargs)
        return cls(operator=operator, flux=flux, metadata=dict(metadata or {}))

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
        """Construct from a legacy flat basis array."""
        return cls.from_basis_flat(
            basis_flat=basis_flat,
            flux=flux,
            region_labels=region_labels,
            operator_kwargs=operator_kwargs,
            metadata=metadata,
        )

    @classmethod
    def from_multi_source_basis_flat(
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
            flux: Flux on the inversion grid. Commonly has a `source` dimension coordinate matching
                the keys of `basis_flat`, but this is not required at construction time.
            operator_kwargs: Optional kwargs forwarded to `MultiSourceBucketBasisOperator`.

        Returns:
            A FluxWeightedBasis pairing a MultiSourceBucketBasisOperator with the provided flux.
        """
        kwargs: dict[str, Any] = dict(operator_kwargs or {})

        operator_cls = cast(Any, MultiSourceBucketBasisOperator)
        operator = operator_cls(basis_flat=dict(basis_flat), **kwargs)

        if not isinstance(flux, xr.DataArray):
            flux = concat_data_arrays(flux, key_dim="source")

        return cls(operator=operator, flux=flux, metadata=dict(metadata or {}))

    @classmethod
    def from_fp_all_flat_basis(
        cls,
        *,
        fp_all: dict,
        basis_flat: xr.DataArray | Mapping[str, xr.DataArray],
        metadata: Mapping[str, Any] | None = None,
    ) -> FluxWeightedBasis:
        """Construct a basis object from legacy wrapper inputs."""
        flux = cls.flux_from_fp_all(fp_all)
        if isinstance(basis_flat, Mapping):
            return cls.from_multi_source_basis_flat(
                basis_flat=basis_flat,
                flux=flux,
                operator_kwargs={"state_dim": "region"},
                metadata=metadata,
            )

        return cls.from_basis_flat(
            basis_flat=basis_flat,
            flux=flux,
            operator_kwargs={"state_dim": "region"},
            metadata=metadata,
        )

    @classmethod
    def flux_from_fp_all(cls, fp_all: dict) -> xr.DataArray:
        """Build the representative flux field from a legacy ``fp_all`` object."""
        if ".flux" not in fp_all or not fp_all[".flux"]:
            raise ValueError("Cannot construct BasisFunctions object: fp_all['.flux'] is missing or empty.")

        flux_entries = fp_all[".flux"]
        flux_arrays = {
            key: _extract_flux_dataarray(value, flux_key=key) for key, value in flux_entries.items()
        }

        if _is_multi_source_workflow(fp_all):
            flux = _stack_flux_sources_with_alignment(flux_arrays)
        else:
            flux = _combine_flux_sources_like_modelscenario(flux_arrays)

        return flux

    def with_flux(self, flux: xr.DataArray) -> FluxWeightedBasis:
        """Return a copy with the same operator and metadata but a different flux."""
        return type(self)(operator=self.operator, flux=flux, metadata=dict(self.metadata))

    def with_metadata(self, metadata: Mapping[str, Any]) -> FluxWeightedBasis:
        """Return a copy with additional metadata."""
        return type(self)(
            operator=self.operator,
            flux=self.flux,
            metadata={**self.metadata, **dict(metadata)},
        )

    @property
    def basis_artifact_source(self) -> str | None:
        """Return the source used to create/load this basis, when recorded."""
        source = self.metadata.get(BASIS_ARTIFACT_SOURCE_ATTR)
        return str(source) if source is not None else None

    def flat_basis(self) -> xr.DataArray | dict[str, xr.DataArray]:
        """Return the flattened basis representation used by legacy H construction."""
        basis_flat = getattr(self.operator, "basis_flat", None)
        if isinstance(basis_flat, xr.DataArray):
            return basis_flat.rename("basis")
        if isinstance(basis_flat, dict):
            return {key: value.rename("basis") for key, value in basis_flat.items()}
        raise AttributeError("Operator does not expose `basis_flat`.")

    def basis_matrix(
        self,
        *,
        state_dim: str | None = None,
        state_coord: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """Return the basis matrix, optionally translated to a requested state dimension."""
        basis = self.operator.basis_matrix
        current_state_dim = self.operator.meta.state_dim
        if state_dim is not None and state_dim != current_state_dim:
            basis = basis.rename({current_state_dim: state_dim})
            current_state_dim = state_dim
        if state_coord is not None:
            if state_coord.name is None:
                raise ValueError("state_coord must be named so it can be used for reindexing.")
            basis = basis.reindex({state_coord.name: state_coord})
        return basis

    def to_datatree(self) -> xr.DataTree:
        """Serialise to a DataTree with `basis` and `flux` groups.

        Returns:
            A DataTree with:
              - `basis`: BasisOperator DataTree (via operator.to_datatree()).
              - `flux`: a Dataset containing the flux DataArray as variable `flux`.

        Raises:
            KeyError: If serialisation would overwrite an existing group name.
        """
        dt_basis = self.operator.to_datatree()

        # Store flux as a dataset to preserve name/attrs cleanly.
        flux = self.flux.rename("flux")
        dt_flux = xr.DataTree(xr.Dataset({"flux": flux}))

        dt_dict = {"basis": dt_basis, "flux": dt_flux}

        dt = xr.DataTree.from_dict(dt_dict)
        dt.attrs.update(
            {
                "schema": "openghg_inversions.flux_weighted_basis",
                "schema_version": 1,
            }
        )
        dt.attrs.update(_serialisable_basis_metadata(self.metadata))
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
            ValueError: If schema/version mismatch or flux variable missing.
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


def _serialisable_basis_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only namespaced metadata that can be stored as DataTree root attributes."""
    return {key: value for key, value in metadata.items() if str(key).startswith(BASIS_METADATA_ATTR_PREFIX)}


def _is_multi_source_workflow(fp_all: dict) -> bool:
    """Determine multi-source/sector mode from fp_all metadata."""
    split_by_sectors = fp_all.get(".split_by_sectors")
    if split_by_sectors is not None:
        return bool(split_by_sectors)

    flux_entries = fp_all.get(".flux")
    return isinstance(flux_entries, dict) and len(flux_entries) > 1


def _extract_flux_dataarray(flux_entry: object, flux_key: str) -> xr.DataArray:
    """Extract a DataArray named ``flux`` from supported flux entry containers."""
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
    """Combine fluxes as in ModelScenario.combine_flux_sources."""
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


def _stack_flux_sources_with_alignment(flux_arrays: dict[str, xr.DataArray]) -> xr.DataArray:
    """Stack fluxes along `source`, validating structural coordinate alignment."""
    first_key = next(iter(flux_arrays))
    reference = flux_arrays[first_key]
    dims_to_align = [dim for dim in reference.dims if dim != "time"]

    aligned_flux = {}
    for key, arr in flux_arrays.items():
        aligned_flux[key] = force_align(arr, reference=reference, dims=dims_to_align)

    return xr.concat(
        [arr.expand_dims({"source": [key]}) for key, arr in aligned_flux.items()],
        dim="source",
        join="outer",
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
