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
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np
import xarray as xr

from openghg_inversions.native_covariance import SeparableExponentialCovariance

__all__ = ["IndependentSourceCovariance"]


@dataclass(frozen=True, slots=True)
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
    """

    source_covariances: Mapping[str, SeparableExponentialCovariance]
    source_dim: str = "native_source"
    _source_labels: tuple[str, ...] = field(init=False, repr=False)

    schema = "openghg_inversions.independent_source_covariance"
    schema_version = 1

    def __post_init__(self) -> None:
        """Validate source blocks and freeze their canonical insertion order.

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
        covariances = dict(self.source_covariances)
        if not covariances:
            raise ValueError("source_covariances must contain at least one source block")
        if not isinstance(self.source_dim, str) or not self.source_dim:
            raise ValueError("source_dim must be a non-empty string")
        if any(not isinstance(label, str) or not label for label in covariances):
            raise ValueError("source covariance labels must be non-empty strings")
        for label, covariance in covariances.items():
            if not isinstance(covariance, SeparableExponentialCovariance):
                raise TypeError(f"Source {label!r} must use SeparableExponentialCovariance")

        reference = next(iter(covariances.values()))
        if self.source_dim in reference.native_dims:
            raise ValueError("source_dim must be distinct from the spatial native dimensions")
        for label, covariance in covariances.items():
            if covariance.native_dims != reference.native_dims:
                raise ValueError("All source covariance blocks must use the same spatial dimensions")
            for dim, expected, actual in zip(
                reference.native_dims,
                (reference.latitude, reference.longitude),
                (covariance.latitude, covariance.longitude),
                strict=True,
            ):
                if not np.array_equal(expected.values, actual.values):
                    raise ValueError(f"Source {label!r} covariance grid differs on {dim!r}")
        object.__setattr__(self, "source_covariances", MappingProxyType(covariances))
        object.__setattr__(self, "_source_labels", tuple(covariances))

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
            the common spatial coordinates, and any encoded class labels.

        Raises:
            TypeError: If a class label or class-label attribute cannot be
                represented by the tagged JSON encoding.
        """
        first = self.source_covariances[self.source_labels[0]]
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
                label_values[index] = np.vectorize(_encode_class_label, otypes=[object])(values)
                label_names.append(str(class_labels.name) if class_labels.name is not None else "")
                label_attrs.append(json.dumps(class_labels.attrs, sort_keys=True, default=_json_default))
            else:
                label_names.append("")
                label_attrs.append("{}")
        return xr.Dataset(
            {
                "sigma": sigma,
                "correlation_length": correlation_length,
                "latitude_correlation_length": latitude_correlation_length,
                "longitude_correlation_length": longitude_correlation_length,
                "class_blocked": blocked,
                "class_label_encoded": xr.DataArray(
                    label_values,
                    dims=(self.source_dim, *first.native_dims),
                    coords={
                        self.source_dim: source,
                        first.native_dims[0]: first.latitude,
                        first.native_dims[1]: first.longitude,
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
                "class_label_encoding": "tagged_json_v1",
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
        source_dim = str(dataset.attrs.get("source_dim", ""))
        latitude_dim = str(dataset.attrs.get("latitude_dim", ""))
        longitude_dim = str(dataset.attrs.get("longitude_dim", ""))
        required = {
            "sigma",
            "correlation_length",
            "latitude_correlation_length",
            "longitude_correlation_length",
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
            if bool(dataset["class_blocked"].sel({source_dim: label}).item()):
                encoded = dataset["class_label_encoded"].sel({source_dim: label}, drop=True)
                decoded = np.vectorize(_decode_class_label, otypes=[object])(encoded.values)
                name = str(dataset["class_label_name"].sel({source_dim: label}).item()) or None
                attrs = json.loads(str(dataset["class_label_attrs"].sel({source_dim: label}).item()))
                labels = encoded.copy(data=decoded).rename(name).assign_attrs(attrs)
            covariances[label] = SeparableExponentialCovariance(
                latitude=dataset.coords[latitude_dim],
                longitude=dataset.coords[longitude_dim],
                sigma=float(dataset["sigma"].sel({source_dim: label}).item()),
                correlation_length=float(dataset["correlation_length"].sel({source_dim: label}).item()),
                latitude_correlation_length=float(
                    dataset["latitude_correlation_length"].sel({source_dim: label}).item()
                ),
                longitude_correlation_length=float(
                    dataset["longitude_correlation_length"].sel({source_dim: label}).item()
                ),
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
            raise ValueError(f"rhs must contain labelled source dimension {self.source_dim!r}")
        actual_labels = tuple(rhs.coords[self.source_dim].values.tolist())
        if any(not isinstance(label, str) for label in actual_labels) or actual_labels != self.source_labels:
            raise ValueError(
                "rhs source labels/order do not match covariance configuration: "
                f"{actual_labels!r} != {self.source_labels!r}"
            )
        original_dims = tuple(str(dim) for dim in rhs.dims)
        results: list[xr.DataArray] = []
        for label in self.source_labels:
            source_rhs = rhs.sel({self.source_dim: label}, drop=True)
            action = self.source_covariances[label]
            results.append(getattr(action, operation)(source_rhs))
        source_coordinate = xr.DataArray(
            list(self.source_labels),
            dims=self.source_dim,
            coords={self.source_dim: list(self.source_labels)},
            attrs=rhs.coords[self.source_dim].attrs,
        )
        combined = xr.concat(results, dim=source_coordinate)
        combined.name = rhs.name
        combined.attrs = rhs.attrs
        return combined.transpose(*original_dims)


def _encode_class_label(value: object) -> str:
    """Encode a class label without losing its supported Python scalar type.

    Args:
        value: String, Boolean, integer, float, NumPy scalar, or nested tuple
            composed from those types.

    Returns:
        Tagged compact JSON representation of ``value``.

    Raises:
        TypeError: If ``value`` has an unsupported type.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, tuple):
        payload = ["tuple", [_encode_class_label(item) for item in value]]
    elif isinstance(value, bool):
        payload = ["bool", value]
    elif isinstance(value, int):
        payload = ["int", value]
    elif isinstance(value, float):
        payload = ["float", value]
    elif isinstance(value, str):
        payload = ["str", value]
    else:
        raise TypeError(f"Unsupported class-label type for serialization: {type(value).__name__}")
    return json.dumps(payload, separators=(",", ":"))


def _decode_class_label(encoded: str) -> object:
    """Decode one tagged JSON class label.

    Args:
        encoded: Tagged JSON produced by :func:`_encode_class_label`.

    Returns:
        Restored supported Python scalar or tuple value.

    Raises:
        ValueError: If the JSON is malformed or its type tag is unknown.
    """
    kind, value = json.loads(str(encoded))
    if kind == "tuple":
        return tuple(_decode_class_label(item) for item in value)
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "str":
        return str(value)
    raise ValueError(f"Unknown encoded class-label kind {kind!r}")


def _json_default(value: object) -> object:
    """Convert a NumPy scalar attribute to a JSON-compatible scalar.

    Args:
        value: Attribute value rejected by JSON's standard encoder.

    Returns:
        Equivalent built-in Python scalar.

    Raises:
        TypeError: If ``value`` is not a NumPy scalar.
    """
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Class-label attr {value!r} is not JSON serializable")
