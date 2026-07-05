"""Helpers for combining partition-local basis labels into one state axis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

BASIS_LABEL_DIM = "basis_label"
BASIS_GROUP_COORD = "basis_group"
BASIS_PARTITION_COORD = "basis_partition"
REGION_IN_PARTITION_COORD = "region_in_partition"


@dataclass(frozen=True, slots=True)
class BasisPartition:
    """Partition-local basis labels and their semantic group name.

    Positive integer values in ``labels`` are treated as local region labels.
    Non-positive values and NaNs are treated as cells outside this partition.
    """

    name: str
    labels: xr.DataArray
    group: str
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BasisLayoutResult:
    """Combined flat basis labels plus raw-label keyed state metadata."""

    basis_flat: xr.DataArray
    state_metadata: xr.Dataset


@dataclass(frozen=True, slots=True)
class BasisLayout:
    """Combine disjoint basis partitions into one flat label map.

    This first implementation is intentionally eager and in-memory: partition
    label arrays are materialized as NumPy arrays while building the combined
    map. File loading, lazy execution, and large-grid chunking policy remain
    outside this small metadata/layout boundary.
    """

    partitions: Sequence[BasisPartition]
    state_dim: str = "state"

    def to_flat_basis(self, *, name: str = "basis") -> BasisLayoutResult:
        """Return a positive-label flat basis and metadata keyed by raw labels.

        This method materializes partition labels and the combined map in
        memory.

        Returns:
            A combined basis map plus a metadata dataset indexed by
            ``basis_label``. ``BucketBasisOperator`` maps that metadata onto the
            final state coordinate after its configured label policy is applied.

        Raises:
            ValueError: If partitions are empty, overlap, contain no mapped
                cells, or leave grid cells unmapped.
            TypeError: If partition labels are not numeric.
        """
        partitions = tuple(self.partitions)
        if not partitions:
            raise ValueError("BasisLayout requires at least one partition.")

        template = partitions[0].labels
        combined = np.zeros(template.shape, dtype=int)
        covered = np.zeros(template.shape, dtype=bool)
        basis_labels: list[int] = []
        groups: list[str] = []
        partition_names: list[str] = []
        region_in_partition: list[int] = []
        next_label = 1

        for partition in partitions:
            labels = _align_partition_labels(partition.labels, template)
            values = _numeric_label_values(labels, partition.name)
            mapped = np.isfinite(values) & (values > 0)

            if not mapped.any():
                raise ValueError(f"Basis partition {partition.name!r} has no positive labels.")
            if np.any(covered & mapped):
                raise ValueError(f"Basis partition {partition.name!r} overlaps an earlier partition.")

            local_labels = _positive_integer_labels(values[mapped], partition.name)
            for local_label in local_labels:
                combined[mapped & (values == local_label)] = next_label
                basis_labels.append(next_label)
                groups.append(partition.group)
                partition_names.append(partition.name)
                region_in_partition.append(int(local_label))
                next_label += 1

            covered |= mapped

        if not covered.all():
            missing = int(np.size(covered) - np.count_nonzero(covered))
            raise ValueError(f"BasisLayout partitions leave {missing} grid cells unmapped.")

        basis_flat = xr.DataArray(
            combined,
            dims=template.dims,
            coords=template.coords,
            name=name,
            attrs=dict(template.attrs),
        )
        state_metadata = xr.Dataset(
            data_vars={
                BASIS_GROUP_COORD: (BASIS_LABEL_DIM, np.asarray(groups, dtype=object)),
                BASIS_PARTITION_COORD: (BASIS_LABEL_DIM, np.asarray(partition_names, dtype=object)),
                REGION_IN_PARTITION_COORD: (BASIS_LABEL_DIM, np.asarray(region_in_partition, dtype=int)),
            },
            coords={BASIS_LABEL_DIM: np.asarray(basis_labels, dtype=int)},
            attrs={"state_dim": self.state_dim},
        )
        return BasisLayoutResult(basis_flat=basis_flat, state_metadata=state_metadata)


def _align_partition_labels(labels: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    """Return labels transposed to the template grid after exact coordinate checks."""
    if set(labels.dims) != set(template.dims):
        raise ValueError(
            f"Basis partition dims {labels.dims!r} do not match template dims {template.dims!r}."
        )
    labels = labels.transpose(*template.dims)
    xr.align(template, labels, join="exact")
    return labels


def _numeric_label_values(labels: xr.DataArray, partition_name: str) -> np.ndarray:
    """Return label values as float for NaN-aware validation."""
    values = np.asarray(labels.values)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"Basis partition {partition_name!r} labels must be numeric.")
    return values.astype(float)


def _positive_integer_labels(values: np.ndarray, partition_name: str) -> np.ndarray:
    """Return sorted unique positive integer labels, validating integrality."""
    if not np.all(values == np.floor(values)):
        raise ValueError(f"Basis partition {partition_name!r} labels must be integer-valued.")
    return np.unique(values.astype(int))
