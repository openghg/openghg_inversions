"""BasisFunctions object to encapsulate representation of basis.

Example usage:

>> def apply_basis_functions(ds: xr.Dataset, bf: BasisFunctions) -> xr.Dataset:
>>     if "fp_x_flux" not in ds:
>>         return ds
>>     return bf.sensitivity(ds.fp_x_flux).rename("H").to_dataset()

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from openghg_inversions.array_ops import (
    concat_data_arrays,
)
from openghg_inversions.basis.operators import (
    BasisOperator,
    BucketBasisOperator,
    MultiSourceBucketBasisOperator,
)
from openghg_inversions.config.paths import Paths

openghginv_path = Paths.openghginv


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

    @classmethod
    def from_basis_flat(
        cls,
        basis_flat: xr.DataArray,
        flux: xr.DataArray,
        *,
        region_labels: xr.DataArray | None = None,
        operator_kwargs: Mapping[str, Any] | None = None,
    ) -> FluxWeightedBasis:
        """Construct from a single-source (standard) flattened basis array.

        Args:
            basis_flat: Flattened basis labels on the inversion grid, e.g. dims like (lat, lon)
                (or whatever grid dims the operator expects). Values should label regions.
            flux: Flux on the same grid dims as `basis_flat`. May optionally contain extra dims such
                as `time` or `source`; these will be carried along by downstream operations.
            region_labels: Optional labels corresponding to region IDs. If not provided, the operator
                will infer labels (typically from unique values of `basis_flat`).
            operator_kwargs: Optional kwargs forwarded to `BucketBasisOperator`.

        Returns:
            A FluxWeightedBasis pairing a BucketBasisOperator with the provided flux.
        """
        kwargs: dict[str, Any] = dict(operator_kwargs or {})
        if region_labels is not None:
            kwargs["region_labels"] = region_labels

        operator = BucketBasisOperator(basis_flat=basis_flat, **kwargs)
        return cls(operator=operator, flux=flux)

    @classmethod
    def from_multi_source_basis_flat(
        cls,
        basis_flat: Mapping[str, xr.DataArray],
        flux: xr.DataArray | Mapping[str, xr.DataArray],
        *,
        region_labels: Mapping[str, xr.DataArray] | None = None,
        operator_kwargs: Mapping[str, Any] | None = None,
    ) -> FluxWeightedBasis:
        """Construct from a multi-source flattened basis mapping.

        Args:
            basis_flat: Mapping from source name to that source's flattened basis labels on the grid.
                Each value should be a DataArray on the inversion grid (e.g. dims like (lat, lon)).
            flux: Flux on the inversion grid. Commonly has a `source` dimension coordinate matching
                the keys of `basis_flat`, but this is not required at construction time.
            region_labels: Optional mapping from source name to region labels for that source's basis.
                If not provided, the operator will infer labels per source.
            operator_kwargs: Optional kwargs forwarded to `MultiSourceBucketBasisOperator`.

        Returns:
            A FluxWeightedBasis pairing a MultiSourceBucketBasisOperator with the provided flux.
        """
        kwargs: dict[str, Any] = dict(operator_kwargs or {})
        if region_labels is not None:
            kwargs["region_labels"] = dict(region_labels)

        operator = MultiSourceBucketBasisOperator(basis_flat=dict(basis_flat), **kwargs)

        if not isinstance(flux, xr.DataArray):
            flux = concat_data_arrays(flux, key_dim="source")

        return cls(operator=operator, flux=flux)

    def to_datatree(self) -> xr.DataTree:
        """Serialise to a DataTree with `basis` and `flux` groups.

        Returns:
            A DataTree with:
              - `basis`: BasisOperator DataTree (via operator.to_datatree()).
              - `flux`: a Dataset containing the flux DataArray as variable `flux`.

        Raises:
            KeyError: If serialisation would overwrite an existing group name.
        """
        dt = xr.DataTree()
        dt.attrs.update(
            {
                "schema": "openghg_inversions.flux_weighted_basis",
                "schema_version": 1,
            }
        )

        dt_basis = self.operator.to_datatree()

        # Store flux as a dataset to preserve name/attrs cleanly.
        flux = self.flux.rename("flux")

        dt_flux = xr.DataTree(xr.Dataset({"flux": flux}))

        dt["basis"] = dt_basis
        dt["flux"] = dt_flux
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

        operator = BasisOperator.decode_datatree(dt["basis"])

        ds_flux = dt["flux"].to_dataset()
        if "flux" not in ds_flux:
            raise ValueError("Missing variable 'flux' in dt['flux'] dataset.")
        flux = ds_flux["flux"]

        return cls(operator=operator, flux=flux)

    def sensitivity(self, fp_x_flux: xr.DataArray) -> xr.DataArray:
        """Compute sensitivity (grid -> state) via the underlying operator.

        This is typically used to form the reduced Jacobian:
            H = operator.sensitivity(fp_x_flux)

        Args:
            fp_x_flux: Footprints multiplied by flux, on the inversion grid dims.
                May contain extra dims (e.g. time, site, source).

        Returns:
            Sensitivity with a `state`-like dimension as defined by the operator.
        """
        return self.operator.sensitivity(fp_x_flux)

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
            sources = list(basis_flat.keys())
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
