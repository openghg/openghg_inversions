"""Compose labelled native covariance actions into independent source blocks.

The multisource covariance represented here is block diagonal by source. Each
configured source owns a
:class:`~openghg_inversions.native_covariance.SeparableExponentialCovariance`
on the same labelled spatial grid, while amplitudes, correlation lengths, and
optional class masks may differ by source. Applying or solving the composite
action dispatches to each source block independently, preserves all labelled
right-hand-side dimensions, and never constructs a dense cross-source matrix.

Source labels are non-empty strings whose insertion order defines the canonical
block order. Input arrays must carry exactly those string labels in that order;
values are not coerced between types. The leading native source dimension
defaults to ``"native_source"``. This is intentionally distinct from the
``"source"`` level on OGI's gathered retained-state MultiIndex because xarray
cannot represent a dimension and a MultiIndex level with the same name in one
prolongation array.

:class:`IndependentSourceCovariance` can serialize its complete reproducible
configuration, including typed class labels, to an xarray dataset. Restoration
validates the schema, source labels, required variables, and spatial coordinate
metadata before reconstructing the component actions.

Source dispatch intentionally uses eager per-source ``sel``, component
operator, and ``concat`` calls. A source-only ``apply_ufunc`` or source
chunking is not equivalent: each component consumes the full labelled spatial
grid and owns an explicit eager NumPy boundary, rather than acting pointwise
along a source vector.

``IndependentSourceCovariance`` is an ordinary slotted, identity-based action.
It copies the source mapping once and exposes it through a read-only proxy;
component actions and their borrowed coordinate properties are not copied on
ordinary access.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Literal, Mapping, SupportsFloat

import numpy as np
import xarray as xr

from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions._serialization_codecs import (
    _TAGGED_JSON_VALUE_ENCODING,
    _decode_serialized_bool,
    _decode_tagged_json_value,
    _encode_tagged_json_value,
    _numpy_scalar_json_default,
)

__all__ = ["IndependentSourceCovariance"]

_SERIALIZED_VARIABLE_NAMES = {
    "sigma",
    "correlation_length",
    "latitude_correlation_length",
    "longitude_correlation_length",
    "latitude_correlation_length_explicit",
    "longitude_correlation_length_explicit",
    "class_blocked",
    "class_label_encoded",
    "class_label_name",
    "class_label_attrs",
}


class IndependentSourceCovariance:
    """Block-diagonal native covariance over an ordered source mapping.

    Args:
        source_covariances: Non-empty insertion-ordered mapping from source
            labels to separable spatial covariance actions on the same grid.
        source_dim: Explicit native source dimension. It should not collide
            with a level name on the retained state MultiIndex.

    Raises:
        TypeError: If a source block is not a
            :class:`SeparableExponentialCovariance` or the source mapping cannot
            be copied.
        ValueError: If the mapping is empty, source labels or ``source_dim``
            are invalid, or block spatial dimensions or grids differ.

    Notes:
        The source mapping and source-dimension properties are read-only.
        Changed configuration requires explicit reconstruction.
    """

    __slots__ = ("_source_covariances", "_source_dim", "_source_labels")

    schema = "openghg_inversions.independent_source_covariance"
    schema_version = 1

    def __init__(
        self,
        source_covariances: Mapping[str, SeparableExponentialCovariance],
        source_dim: str = "native_source",
    ) -> None:
        """Validate source blocks and retain their canonical insertion order.

        The complete block mapping is type-checked before any block attributes
        are inspected. The validated copy is exposed through a read-only
        mapping proxy so later mutations of the caller's mapping cannot change
        the covariance configuration.

        Raises:
            TypeError: If ``source_covariances`` cannot be copied into a
                mapping or any block is not a
                :class:`SeparableExponentialCovariance`.
            ValueError: If no blocks are supplied, source labels or
                ``source_dim`` are invalid, or blocks do not share identical
                labelled spatial grids.
        """
        covariances = dict(source_covariances)
        if not covariances:
            raise ValueError("source_covariances must contain at least one source block")
        if not isinstance(source_dim, str) or not source_dim:
            raise ValueError("source_dim must be a non-empty string")
        if source_dim in _SERIALIZED_VARIABLE_NAMES:
            raise ValueError(f"source_dim {source_dim!r} is reserved by the serialized schema")
        if any(not isinstance(label, str) or not label for label in covariances):
            raise ValueError("source covariance labels must be non-empty strings")
        for label, covariance in covariances.items():
            if not isinstance(covariance, SeparableExponentialCovariance):
                raise TypeError(f"Source {label!r} must use SeparableExponentialCovariance")

        reference = next(iter(covariances.values()))
        if source_dim in reference.native_dims:
            raise ValueError("source_dim must be distinct from the spatial native dimensions")
        reserved_spatial_dims = _SERIALIZED_VARIABLE_NAMES.intersection(reference.native_dims)
        if reserved_spatial_dims:
            reserved = sorted(reserved_spatial_dims)[0]
            raise ValueError(f"spatial dimension {reserved!r} is reserved by the serialized schema")
        for label, covariance in covariances.items():
            if covariance.native_dims != reference.native_dims:
                raise ValueError("All source covariance blocks must use the same spatial dimensions")
            for dim, expected, actual in zip(
                reference.native_dims,
                (reference.latitude, reference.longitude),
                (covariance.latitude, covariance.longitude),
                strict=True,
            ):
                if not expected.identical(actual):
                    raise ValueError(f"Source {label!r} covariance grid differs on {dim!r}")
        self._source_covariances = MappingProxyType(covariances)
        self._source_dim = source_dim
        self._source_labels = tuple(covariances)

    @property
    def source_covariances(self) -> Mapping[str, SeparableExponentialCovariance]:
        """Return the read-only source-to-covariance mapping."""
        return self._source_covariances

    @property
    def source_dim(self) -> str:
        """Return the read-only native source dimension name."""
        return self._source_dim

    @property
    def source_labels(self) -> tuple[str, ...]:
        """Return configured source labels in canonical insertion order.

        Returns:
            Source labels defining the block order.
        """
        return self._source_labels

    @property
    def native_dims(self) -> tuple[str, ...]:
        """Return the source dimension followed by common spatial dimensions.

        Returns:
            Native dimensions in vectorisation order.
        """
        first = self.source_covariances[self.source_labels[0]]
        return (self.source_dim, *first.native_dims)

    def apply(self, rhs: xr.DataArray) -> xr.DataArray:
        """Apply independent source covariance blocks to labelled RHS arrays.

        Args:
            rhs: Array containing the exact source and spatial native labels;
                all other right-hand-side dimensions are preserved.

        Returns:
            Blockwise ``B rhs`` in the original dimension order.

        Raises:
            TypeError: If ``rhs`` is not an xarray data array.
            ValueError: If source or spatial dimensions, coordinates, labels,
                or numerical values are missing or invalid.
        """
        return self._operate(rhs, operation="apply")

    def solve(self, rhs: xr.DataArray) -> xr.DataArray:
        """Solve independent source covariance blocks for labelled RHS arrays.

        Args:
            rhs: Array containing the exact source and spatial native labels;
                all other right-hand-side dimensions are preserved.

        Returns:
            Blockwise ``B^-1 rhs`` in the original dimension order.

        Raises:
            TypeError: If ``rhs`` is not an xarray data array.
            ValueError: If source or spatial dimensions, coordinates, labels,
                or numerical values are missing or invalid.
            numpy.linalg.LinAlgError: If a class-blocked component solve does
                not converge.
        """
        return self._operate(rhs, operation="solve")

    def to_dataset(self) -> xr.Dataset:
        """Serialize source order and reproducible component configuration.

        Returns:
            Versioned dataset containing source-specific covariance parameters,
            independent eager copies of the common spatial coordinates, and
            any encoded class labels. The returned dataset may be mutated
            without changing component actions.

        Raises:
            TypeError: If a class label or class-label attribute cannot be
                represented by the tagged JSON encoding.
        """
        first = self.source_covariances[self.source_labels[0]]
        latitude = first.latitude.copy(deep=True)
        longitude = first.longitude.copy(deep=True)
        source = xr.IndexVariable(self.source_dim, list(self.source_labels))
        sigma = xr.DataArray(
            [self.source_covariances[label].sigma for label in self.source_labels],
            dims=self.source_dim,
            coords={self.source_dim: source},
        )
        correlation_length = xr.DataArray(
            [self.source_covariances[label].correlation_length for label in self.source_labels],
            dims=self.source_dim,
            coords={self.source_dim: source},
        )
        latitude_correlation_length = xr.DataArray(
            [self.source_covariances[label].latitude_correlation_length for label in self.source_labels],
            dims=self.source_dim,
            coords={self.source_dim: source},
        )
        longitude_correlation_length = xr.DataArray(
            [self.source_covariances[label].longitude_correlation_length for label in self.source_labels],
            dims=self.source_dim,
            coords={self.source_dim: source},
        )
        latitude_correlation_length_explicit = xr.DataArray(
            [
                self.source_covariances[label].latitude_correlation_length_explicit
                for label in self.source_labels
            ],
            dims=self.source_dim,
            coords={self.source_dim: source},
        )
        longitude_correlation_length_explicit = xr.DataArray(
            [
                self.source_covariances[label].longitude_correlation_length_explicit
                for label in self.source_labels
            ],
            dims=self.source_dim,
            coords={self.source_dim: source},
        )
        blocked = xr.DataArray(
            [self.source_covariances[label].class_labels is not None for label in self.source_labels],
            dims=self.source_dim,
            coords={self.source_dim: source},
        )
        label_values = np.full(
            (len(self.source_labels), first.latitude.size, first.longitude.size),
            "",
            dtype=object,
        )
        label_names: list[str] = []
        label_attrs: list[str] = []
        for index, label in enumerate(self.source_labels):
            class_labels = self.source_covariances[label].class_labels
            if class_labels is not None:
                values = class_labels.transpose(*first.native_dims).values
                label_values[index] = np.vectorize(_encode_tagged_json_value, otypes=[object])(values)
                label_names.append(str(class_labels.name) if class_labels.name is not None else "")
                label_attrs.append(
                    json.dumps(
                        class_labels.attrs,
                        sort_keys=True,
                        default=_numpy_scalar_json_default,
                    )
                )
            else:
                label_names.append("")
                label_attrs.append("{}")
        return xr.Dataset(
            {
                "sigma": sigma,
                "correlation_length": correlation_length,
                "latitude_correlation_length": latitude_correlation_length,
                "longitude_correlation_length": longitude_correlation_length,
                "latitude_correlation_length_explicit": latitude_correlation_length_explicit,
                "longitude_correlation_length_explicit": longitude_correlation_length_explicit,
                "class_blocked": blocked,
                "class_label_encoded": xr.DataArray(
                    label_values,
                    dims=(self.source_dim, *first.native_dims),
                    coords={
                        self.source_dim: source,
                        first.native_dims[0]: latitude,
                        first.native_dims[1]: longitude,
                    },
                ),
                "class_label_name": xr.DataArray(
                    label_names,
                    dims=self.source_dim,
                    coords={self.source_dim: source},
                ),
                "class_label_attrs": xr.DataArray(
                    label_attrs,
                    dims=self.source_dim,
                    coords={self.source_dim: source},
                ),
            },
            attrs={
                "schema": self.schema,
                "schema_version": self.schema_version,
                "source_dim": self.source_dim,
                "latitude_dim": first.native_dims[0],
                "longitude_dim": first.native_dims[1],
                "class_label_encoding": _TAGGED_JSON_VALUE_ENCODING,
            },
        )

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> IndependentSourceCovariance:
        """Restore source blocks from :meth:`to_dataset` output.

        Args:
            dataset: Versioned independent-source covariance dataset.

        Returns:
            Reconstructed source-block covariance action with class-label
            values, name, and JSON-compatible attributes restored.

        Raises:
            ValueError: If schema metadata, source or spatial coordinates, or
                required variables are absent or invalid, or encoded metadata
                cannot be decoded.
        """
        if dataset.attrs.get("schema") != cls.schema:
            raise ValueError(f"Expected source covariance schema {cls.schema!r}")
        if dataset.attrs.get("schema_version") != cls.schema_version:
            raise ValueError("Unsupported independent-source covariance schema version")
        if dataset.attrs.get("class_label_encoding") != _TAGGED_JSON_VALUE_ENCODING:
            raise ValueError("Unsupported independent-source class-label encoding")
        source_dim = str(dataset.attrs.get("source_dim", ""))
        latitude_dim = str(dataset.attrs.get("latitude_dim", ""))
        longitude_dim = str(dataset.attrs.get("longitude_dim", ""))
        required = {
            "sigma",
            "correlation_length",
            "latitude_correlation_length",
            "longitude_correlation_length",
            "latitude_correlation_length_explicit",
            "longitude_correlation_length_explicit",
            "class_blocked",
            "class_label_encoded",
            "class_label_name",
            "class_label_attrs",
        }
        missing = required.difference(dataset.data_vars)
        spatial_coordinates_valid = (
            bool(latitude_dim)
            and bool(longitude_dim)
            and latitude_dim != longitude_dim
            and latitude_dim in dataset.coords
            and longitude_dim in dataset.coords
        )
        if not source_dim or source_dim not in dataset.coords or missing:
            raise ValueError(f"Serialized source covariance is missing labels or variables {sorted(missing)}")
        if not spatial_coordinates_valid:
            raise ValueError("Serialized source covariance is missing latitude or longitude coordinates")
        raw_source_labels = dataset.coords[source_dim].values.tolist()
        if any(not isinstance(label, str) or not label for label in raw_source_labels):
            raise ValueError("Serialized source covariance labels must be non-empty strings")
        covariances: dict[str, SeparableExponentialCovariance] = {}
        for label in raw_source_labels:
            labels = None
            blocked = _decode_serialized_bool(
                dataset["class_blocked"].sel({source_dim: label}).item(),
                f"class_blocked flag for source {label!r}",
            )
            encoded = dataset["class_label_encoded"].sel({source_dim: label}, drop=True)
            encoded_values = np.asarray(encoded.values, dtype=str)
            has_encoded_labels = bool(np.any(encoded_values != ""))
            if blocked != has_encoded_labels:
                raise ValueError(
                    f"Serialized class_blocked flag for source {label!r} contradicts encoded labels"
                )
            if blocked:
                decoded = np.vectorize(_decode_tagged_json_value, otypes=[object])(encoded.values)
                name = str(dataset["class_label_name"].sel({source_dim: label}).item()) or None
                attrs = json.loads(str(dataset["class_label_attrs"].sel({source_dim: label}).item()))
                labels = encoded.copy(data=decoded).rename(name).assign_attrs(attrs)
            correlation_length = _positive_finite(
                dataset["correlation_length"].sel({source_dim: label}).item(),
                f"correlation length for source {label!r}",
            )
            latitude_length = _positive_finite(
                dataset["latitude_correlation_length"].sel({source_dim: label}).item(),
                f"latitude correlation length for source {label!r}",
            )
            longitude_length = _positive_finite(
                dataset["longitude_correlation_length"].sel({source_dim: label}).item(),
                f"longitude correlation length for source {label!r}",
            )
            latitude_explicit = _decode_serialized_bool(
                dataset["latitude_correlation_length_explicit"].sel({source_dim: label}).item(),
                f"latitude correlation-length flag for source {label!r}",
            )
            longitude_explicit = _decode_serialized_bool(
                dataset["longitude_correlation_length_explicit"].sel({source_dim: label}).item(),
                f"longitude correlation-length flag for source {label!r}",
            )
            if not latitude_explicit and latitude_length != correlation_length:
                raise ValueError(
                    f"Implicit latitude correlation length for source {label!r} contradicts the fallback"
                )
            if not longitude_explicit and longitude_length != correlation_length:
                raise ValueError(
                    f"Implicit longitude correlation length for source {label!r} contradicts the fallback"
                )
            covariances[label] = SeparableExponentialCovariance(
                latitude=dataset.coords[latitude_dim],
                longitude=dataset.coords[longitude_dim],
                sigma=float(dataset["sigma"].sel({source_dim: label}).item()),
                correlation_length=correlation_length,
                latitude_correlation_length=latitude_length if latitude_explicit else None,
                longitude_correlation_length=longitude_length if longitude_explicit else None,
                class_labels=labels,
            )
        return cls(covariances, source_dim=source_dim)

    def _operate(
        self,
        rhs: xr.DataArray,
        *,
        operation: Literal["apply", "solve"],
    ) -> xr.DataArray:
        """Validate labels and dispatch one covariance operation per source.

        Args:
            rhs: Candidate labelled right-hand side.
            operation: Component method to invoke for every source block.

        Returns:
            Concatenated block results transposed to the input dimension order.

        Raises:
            TypeError: If ``rhs`` is not an xarray data array.
            ValueError: If source or spatial labels are missing or do not
                exactly match the configured labels, or values are invalid.
            numpy.linalg.LinAlgError: If a requested class-blocked solve does
                not converge.
        """
        if not isinstance(rhs, xr.DataArray):
            raise TypeError("rhs must be an xarray.DataArray")
        if self.source_dim not in rhs.dims or self.source_dim not in rhs.coords:
            raise ValueError(f"rhs must contain labelled source dimension coordinate {self.source_dim!r}")
        expected = xr.DataArray(
            list(self.source_labels),
            dims=self.source_dim,
            coords={self.source_dim: list(self.source_labels)},
        )
        try:
            aligned_rhs, _ = xr.align(rhs, expected, join="exact", copy=False)
        except xr.AlignmentError as error:
            raise ValueError("rhs source labels/order do not match covariance configuration") from error

        original_dims = tuple(str(dim) for dim in rhs.dims)
        results: list[xr.DataArray] = []
        for label in self.source_labels:
            source_rhs = aligned_rhs.sel({self.source_dim: label})
            action = self.source_covariances[label]
            results.append(getattr(action, operation)(source_rhs))
        combined = xr.concat(
            results,
            dim=aligned_rhs.coords[self.source_dim],
            coords="minimal",
            compat="override",
        )
        restored_data = combined.transpose(*original_dims, transpose_coords=False).data
        return rhs.copy(data=restored_data, deep=False)


def _positive_finite(value: SupportsFloat, name: str) -> float:
    """Decode one serialized positive finite covariance parameter."""
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"Serialized {name} must be finite and strictly positive")
    return resolved
