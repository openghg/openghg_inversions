"""Independent labelled source blocks for native covariance actions.

The first multisource covariance contract is block diagonal by source.  Each
configured source owns a :class:`~openghg_inversions.native_covariance.SeparableExponentialCovariance`
on the same spatial grid, so source-specific amplitudes, correlation lengths,
and optional class masks remain explicit.  The action applies or solves each
block independently and preserves the configured non-lexical source order.

The leading native source dimension defaults to ``"native_source"``.  This is
intentionally distinct from the ``"source"`` level on OGI's gathered retained
state MultiIndex: xarray cannot represent a dimension and a MultiIndex level
with the same name in one prolongation array.  Callers can rename a native
``source`` dimension at this preparation boundary without changing its labels.
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
        ValueError: If source labels, spatial dimensions, or grids differ.
    """

    source_covariances: Mapping[str, SeparableExponentialCovariance]
    source_dim: str = "native_source"
    _source_labels: tuple[str, ...] = field(init=False, repr=False)

    schema = "openghg_inversions.independent_source_covariance"
    schema_version = 1

    def __post_init__(self) -> None:
        covariances = dict(self.source_covariances)
        if not covariances:
            raise ValueError("source_covariances must contain at least one source block")
        if not isinstance(self.source_dim, str) or not self.source_dim:
            raise ValueError("source_dim must be a non-empty string")
        if self.source_dim in next(iter(covariances.values())).native_dims:
            raise ValueError("source_dim must be distinct from the spatial native dimensions")
        if any(not isinstance(label, str) or not label for label in covariances):
            raise ValueError("source covariance labels must be non-empty strings")

        reference = next(iter(covariances.values()))
        for label, covariance in covariances.items():
            if not isinstance(covariance, SeparableExponentialCovariance):
                raise TypeError(f"Source {label!r} must use SeparableExponentialCovariance")
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
        """Configured source labels in canonical insertion order."""
        return self._source_labels

    @property
    def native_dims(self) -> tuple[str, ...]:
        """Explicit source dimension followed by the common spatial dimensions."""
        first = self.source_covariances[self.source_labels[0]]
        return (self.source_dim, *first.native_dims)

    def apply(self, rhs: xr.DataArray) -> xr.DataArray:
        """Apply independent source covariance blocks to labelled RHS arrays.

        Args:
            rhs: Array containing the exact source and spatial native labels;
                all other right-hand-side dimensions are preserved.

        Returns:
            Blockwise ``B rhs`` in the original dimension order.
        """
        return self._operate(rhs, operation="apply")

    def solve(self, rhs: xr.DataArray) -> xr.DataArray:
        """Solve independent source covariance blocks for labelled RHS arrays.

        Args:
            rhs: Array containing the exact source and spatial native labels;
                all other right-hand-side dimensions are preserved.

        Returns:
            Blockwise ``B^-1 rhs`` in the original dimension order.
        """
        return self._operate(rhs, operation="solve")

    def to_dataset(self) -> xr.Dataset:
        """Serialize source order and every reproducible component configuration."""
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
            ValueError: If schema metadata or required variables are absent.
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
        if not source_dim or source_dim not in dataset.coords or missing:
            raise ValueError(f"Serialized source covariance is missing labels or variables {sorted(missing)}")
        covariances: dict[str, SeparableExponentialCovariance] = {}
        for raw_label in dataset.coords[source_dim].values:
            label = str(raw_label)
            labels = None
            if bool(dataset["class_blocked"].sel({source_dim: raw_label}).item()):
                encoded = dataset["class_label_encoded"].sel({source_dim: raw_label}, drop=True)
                decoded = np.vectorize(_decode_class_label, otypes=[object])(encoded.values)
                name = str(dataset["class_label_name"].sel({source_dim: raw_label}).item()) or None
                attrs = json.loads(str(dataset["class_label_attrs"].sel({source_dim: raw_label}).item()))
                labels = encoded.copy(data=decoded).rename(name).assign_attrs(attrs)
            covariances[label] = SeparableExponentialCovariance(
                latitude=dataset.coords[latitude_dim],
                longitude=dataset.coords[longitude_dim],
                sigma=float(dataset["sigma"].sel({source_dim: raw_label}).item()),
                correlation_length=float(dataset["correlation_length"].sel({source_dim: raw_label}).item()),
                latitude_correlation_length=float(
                    dataset["latitude_correlation_length"].sel({source_dim: raw_label}).item()
                ),
                longitude_correlation_length=float(
                    dataset["longitude_correlation_length"].sel({source_dim: raw_label}).item()
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
        """Validate source labels and dispatch one covariance operation per block."""
        if self.source_dim not in rhs.dims or self.source_dim not in rhs.coords:
            raise ValueError(f"rhs must contain labelled source dimension {self.source_dim!r}")
        actual_labels = tuple(str(label) for label in rhs.coords[self.source_dim].values)
        if actual_labels != self.source_labels:
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
    """Encode common scalar/tuple class labels without losing their Python type."""
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
    """Decode one tagged JSON class label produced by :func:`_encode_class_label`."""
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
    """Convert NumPy scalar attrs to JSON-compatible Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Class-label attr {value!r} is not JSON serializable")
