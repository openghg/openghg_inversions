"""Grouped basis layout helpers.

This module contains the small internal representation used to combine
partition-local basis labels into one flat basis map while preserving semantic
state metadata. It is intentionally separate from basis-generation algorithms:
algorithms decide how to split cells inside a mask or region class, while
``BasisLayout`` records how those partition-local outputs are assembled into the
single state axis consumed by ``BucketBasisOperator``.

``BasisLayout`` does not infer a catch-all remainder region. Callers that need
one should pass it as an explicit ``BasisPartition`` so the semantic group and
local region label are visible in the resulting state metadata.

The current implementation is eager and in-memory. It is intended for small
metadata/layout assembly steps after region masks and basis labels have already
been loaded or generated. File loading, lazy execution, large-grid chunking, and
public RHIME configuration are handled outside this module.
"""

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
STATE_METADATA_COORDS = (BASIS_GROUP_COORD, BASIS_PARTITION_COORD, REGION_IN_PARTITION_COORD)


@dataclass(frozen=True, slots=True)
class BasisPartition:
    """Partition-local basis labels and their semantic group name.

    Positive integer values in ``labels`` are treated as local region labels.
    Non-positive values and NaNs are treated as cells outside this partition.

    Attributes:
        name: Stable partition name used in ``basis_partition`` metadata.
        labels: Partition-local label array. Positive integer values identify
            regions inside this partition.
        group: Semantic group name used in ``basis_group`` metadata.
        attrs: Optional partition metadata reserved for future callers. These
            attrs are not currently serialized into the layout result.
    """

    name: str
    labels: xr.DataArray
    group: str
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BasisLayoutResult:
    """Combined flat basis labels plus raw-label keyed state metadata.

    Attributes:
        basis_flat: Positive integer flat basis map covering every grid cell.
        state_metadata: Dataset indexed by ``basis_label`` with
            ``basis_group``, ``basis_partition``, and ``region_in_partition``.
    """

    basis_flat: xr.DataArray
    state_metadata: xr.Dataset


@dataclass(frozen=True, slots=True)
class BasisStateMetadata:
    """Semantic coordinates describing a basis operator state axis.

    The metadata may start indexed by raw ``basis_label`` values from a flat
    basis layout or by the final operator state dimension. This value object
    owns validation and alignment so operator classes do not need to know the
    dataset schema beyond their state coordinate and raw basis-label order.

    Attributes:
        dataset: Dataset containing ``basis_group``, ``basis_partition``, and
            ``region_in_partition`` variables indexed by either ``basis_label``
            or a final state dimension.
    """

    dataset: xr.Dataset

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset | BasisStateMetadata) -> BasisStateMetadata:
        """Return ``dataset`` as a ``BasisStateMetadata`` instance.

        Args:
            dataset: Metadata dataset or an existing ``BasisStateMetadata``.

        Returns:
            ``dataset`` wrapped as ``BasisStateMetadata``.
        """
        if isinstance(dataset, BasisStateMetadata):
            return dataset
        return cls(dataset=dataset)

    @classmethod
    def from_matrix(cls, mat: xr.DataArray, *, state_dim: str) -> BasisStateMetadata | None:
        """Extract complete basis state metadata from matrix coordinates.

        Args:
            mat: Basis matrix that may carry grouped metadata coordinates on the
                state dimension.
            state_dim: Name of the state dimension on ``mat``.

        Returns:
            Metadata indexed by ``state_dim`` if all grouped metadata
            coordinates are present, otherwise ``None``.
        """
        if not all(name in mat.coords for name in STATE_METADATA_COORDS):
            return None
        dataset = xr.Dataset(
            data_vars={
                name: (state_dim, np.asarray(mat.coords[name].values))
                for name in STATE_METADATA_COORDS
            },
            coords={state_dim: np.asarray(mat[state_dim].values)},
        )
        return cls(dataset=dataset)

    def to_dataset(self) -> xr.Dataset:
        """Return the metadata as an xarray dataset."""
        return self.dataset

    def on_state_dim(
        self,
        *,
        state_dim: str,
        state_coord: xr.DataArray,
        basis_value_labels: Sequence[int] | np.ndarray,
    ) -> BasisStateMetadata:
        """Return metadata indexed by the final operator state dimension.

        Args:
            state_dim: Final state dimension name.
            state_coord: Final state coordinate after the operator label policy
                has been applied.
            basis_value_labels: Raw basis labels ordered to match the operator
                state columns before final relabeling.

        Returns:
            Metadata aligned to ``state_coord`` and indexed by ``state_dim``.

        Raises:
            ValueError: If metadata labels do not match basis labels, if raw
                labels are duplicated, if final-state metadata has the wrong
                length, or if required variables are missing.
        """
        dataset = self.dataset
        self._validate_required_variables(dataset)

        if BASIS_LABEL_DIM in dataset.dims:
            metadata_labels = _unique_integer_values(
                dataset[BASIS_LABEL_DIM].values,
                name=BASIS_LABEL_DIM,
            )
            expected_labels = _unique_integer_values(basis_value_labels, name="basis labels")
            if set(metadata_labels.tolist()) != set(expected_labels.tolist()):
                raise ValueError(
                    "State metadata basis_label values must match the basis labels; "
                    f"got {metadata_labels.tolist()} and expected {expected_labels.tolist()}."
                )
            metadata = dataset.sel({BASIS_LABEL_DIM: expected_labels})
            metadata = metadata.rename({BASIS_LABEL_DIM: state_dim})
        elif state_dim in dataset.dims:
            metadata = dataset.copy()
            if metadata.sizes[state_dim] != state_coord.sizes[state_dim]:
                raise ValueError(
                    f"State metadata has {metadata.sizes[state_dim]} entries on "
                    f"{state_dim!r}; expected {state_coord.sizes[state_dim]}."
                )
            if state_dim in metadata.coords and not np.array_equal(
                metadata[state_dim].values,
                state_coord.values,
            ):
                raise ValueError(
                    f"State metadata coordinate {state_dim!r} does not match "
                    "the final operator state coordinate."
                )
        else:
            raise ValueError(
                f"State metadata must be indexed by {BASIS_LABEL_DIM!r} or {state_dim!r}."
            )

        metadata = metadata.assign_coords({state_dim: state_coord})
        for name in STATE_METADATA_COORDS:
            variable = metadata[name]
            if variable.dims != (state_dim,):
                raise ValueError(
                    f"State metadata variable {name!r} must have only dimension "
                    f"{state_dim!r}; got {variable.dims!r}."
                )
        return BasisStateMetadata(dataset=metadata)

    def assign_to_matrix(self, mat: xr.DataArray, *, state_dim: str) -> xr.DataArray:
        """Attach metadata variables as coordinates on a basis matrix.

        Args:
            mat: Basis matrix whose state coordinate matches this metadata.
            state_dim: State dimension used by ``mat`` and this metadata.

        Returns:
            ``mat`` with grouped metadata coordinates attached.

        Raises:
            ValueError: If this metadata is not already indexed by
                ``state_dim`` or required variables are malformed.
        """
        self._validate_required_variables(self.dataset)
        if state_dim not in self.dataset.dims:
            raise ValueError(f"State metadata must be indexed by {state_dim!r}.")

        coords: dict[str, tuple[str, np.ndarray]] = {}
        for name in STATE_METADATA_COORDS:
            variable = self.dataset[name]
            if variable.dims != (state_dim,):
                raise ValueError(
                    f"State metadata variable {name!r} must have only dimension "
                    f"{state_dim!r}; got {variable.dims!r}."
                )
            coords[name] = (state_dim, np.asarray(variable.values))
        return mat.assign_coords(coords)

    @staticmethod
    def _validate_required_variables(dataset: xr.Dataset) -> None:
        """Raise if the standard grouped metadata variables are incomplete."""
        for name in STATE_METADATA_COORDS:
            if name not in dataset:
                raise ValueError(f"State metadata is missing required variable {name!r}.")


@dataclass(frozen=True, slots=True)
class BasisLayout:
    """Combine disjoint basis partitions into one flat label map.

    This first implementation is intentionally eager and in-memory: partition
    label arrays are materialized as NumPy arrays while building the combined
    map. File loading, lazy execution, and large-grid chunking policy remain
    outside this small metadata/layout boundary.

    Attributes:
        partitions: Ordered partition definitions to combine. Partitions must
            cover disjoint grid cells and the combined layout must cover the
            full grid.
        state_dim: Name of the downstream state dimension that will eventually
            receive the metadata coordinates.
    """

    partitions: Sequence[BasisPartition]
    state_dim: str = "state"

    def to_flat_basis(self, *, name: str = "basis") -> BasisLayoutResult:
        """Return a positive-label flat basis and metadata keyed by raw labels.

        This method materializes partition labels and the combined map in
        memory.

        Args:
            name: Name to assign to the returned flat-basis DataArray.

        Returns:
            A combined basis map plus metadata indexed by ``basis_label``.
            ``BucketBasisOperator`` maps that metadata onto the final state
            coordinate after its configured label policy is applied.

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
    """Return labels transposed to the template grid.

    Args:
        labels: Partition label array to validate and transpose.
        template: First partition label array whose dimensions and coordinates
            define the layout grid.

    Returns:
        ``labels`` transposed into ``template`` dimension order.

    Raises:
        ValueError: If dimensions differ or coordinates are not exactly aligned.
    """
    if set(labels.dims) != set(template.dims):
        raise ValueError(
            f"Basis partition dims {labels.dims!r} do not match template dims {template.dims!r}."
        )
    labels = labels.transpose(*template.dims)
    xr.align(template, labels, join="exact")
    return labels


def _numeric_label_values(labels: xr.DataArray, partition_name: str) -> np.ndarray:
    """Return numeric label values as floats for NaN-aware validation.

    Args:
        labels: Partition label array to materialize.
        partition_name: Partition name used in validation error messages.

    Returns:
        Label values as a floating-point NumPy array.

    Raises:
        TypeError: If ``labels`` is not numeric.
    """
    values = np.asarray(labels.values)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"Basis partition {partition_name!r} labels must be numeric.")
    return values.astype(float)


def _positive_integer_labels(values: np.ndarray, partition_name: str) -> np.ndarray:
    """Return sorted unique positive integer labels.

    Args:
        values: Positive mapped label values for one partition.
        partition_name: Partition name used in validation error messages.

    Returns:
        Sorted unique integer label values.

    Raises:
        ValueError: If any mapped label value is not integer-valued.
    """
    if not np.all(values == np.floor(values)):
        raise ValueError(f"Basis partition {partition_name!r} labels must be integer-valued.")
    return np.unique(values.astype(int))


def _unique_integer_values(values: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    """Return integer values after checking uniqueness.

    Args:
        values: Values expected to be integer-valued.
        name: Name used in validation error messages.

    Returns:
        Integer values in their original order.

    Raises:
        ValueError: If values are not finite integers or contain duplicates.
    """
    raw_values = np.asarray(values)
    if not np.issubdtype(raw_values.dtype, np.number):
        raise ValueError(f"{name} values must be numeric integer values.")
    try:
        numeric_values = raw_values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be integer-valued.") from exc

    if not np.all(np.isfinite(numeric_values)) or not np.all(
        numeric_values == np.floor(numeric_values)
    ):
        raise ValueError(f"{name} values must be integer-valued.")

    integer_values = numeric_values.astype(int)
    if len(np.unique(integer_values)) != len(integer_values):
        raise ValueError(f"{name} values must be unique; got {integer_values.tolist()}.")
    return integer_values
