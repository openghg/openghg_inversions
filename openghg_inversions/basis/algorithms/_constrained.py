"""Mask- and region-constrained basis generation helpers.

The helpers in this module are intentionally pure: callers pass an already
computed 2D weight field and an already loaded 2D class mask. File loading,
footprint/flux reduction, and domain-specific country lookup stay at the caller
boundary.

The main public entry point is :func:`region_constrained_basis`, with
:func:`allocate_nbasis_by_class` handling class-local target allocation. The
default implementation uses :class:`GreedyAxisParallelSplitStrategy` with
:class:`AxisParallelSplitStep`; callers can supply other partition steps such as
:class:`InertialSplitStep`.

Greedy split steps can optionally use a geometry object for split-shape
decisions. Geometry affects only coordinate comparisons such as long-axis
choice and inertial projection; contribution weights, label allocation, and
split-stopping policies remain separate.

Output labels preserve the input weight dimensions and coordinates. Positive
labels are globally unique, unmapped cells remain ``0``, and region-class
boundaries are never crossed.

Layered region-class helpers are also pure. They combine already loaded masks
into composite class labels so callers can express intersections such as
land/sea by inner/outer without baking file loading or runner configuration into
the algorithm.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Literal, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy import ndimage

from ._weighted import bucket_value_split

# Allocation modes control how an integer ``nbasis`` is distributed across
# region classes before class-local splitting is applied.
AllocationMode: TypeAlias = Literal["weight", "area"]
NbasisAllocation: TypeAlias = int | Mapping[Hashable, int]
GridNode: TypeAlias = tuple[int, int]
GridPartition: TypeAlias = list[GridNode]
_INERTIAL_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class _CoordinatePCA:
    """Principal-component fit for partition coordinates."""

    centroid: npt.NDArray[np.float64]
    centered: npt.NDArray[np.float64]
    eigenvalues: npt.NDArray[np.float64]
    eigenvectors: npt.NDArray[np.float64]


class SplitStrategy(Protocol):
    """Strategy protocol for class-local basis splitting."""

    def __call__(
        self,
        weights: np.ndarray,
        class_mask: np.ndarray,
        target_regions: int,
    ) -> np.ndarray:
        """Split one class into local basis labels.

        Args:
            weights: Non-negative weight field for the full grid.
            class_mask: Boolean mask selecting the cells in the class being
                split.
            target_regions: Requested number of local labels for this class.

        Returns:
            Integer label array with positive labels inside ``class_mask`` and
            zero outside it.
        """
        ...


class ComponentConsolidationPolicy(Protocol):
    """Policy protocol for optional post-construction region consolidation."""

    def __call__(
        self,
        labels: xr.DataArray,
        region_classes: xr.DataArray,
    ) -> xr.DataArray:
        """Return labels after optional class-safe region consolidation."""
        ...


class PartitionStep(Protocol):
    """Strategy protocol for splitting one partition into child partitions."""

    def __call__(self, nodes: GridPartition, weights: np.ndarray) -> list[GridPartition]:
        """Split one grid-node partition.

        Args:
            nodes: Grid nodes in the partition being split.
            weights: Non-negative weight field aligned to the source grid.

        Returns:
            Child grid-node partitions. Returning one child leaves the
            partition effectively unsplit; returning multiple children lets the
            greedy orchestrator decide how many can fit in the target count.
        """
        ...


class SplitGeometry(Protocol):
    """Geometry protocol for mapping grid nodes into physical coordinates."""

    def coordinates(
        self,
        nodes: GridPartition,
        node_weights: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64] | None:
        """Return physical coordinates for grid nodes.

        Args:
            nodes: Grid nodes in the partition being split.
            node_weights: Optional non-negative weights for the same nodes.

        Returns:
            A finite ``(nnode, 2)`` coordinate array whose first column is
            aligned with grid axis ``0`` and second column is aligned with grid
            axis ``1``. Return ``None`` when physical coordinates are
            unavailable and index-space fallback should be used.
        """
        ...


@dataclass
class LatLonGridGeometry:
    """Local tangent-plane geometry for latitude/longitude grids.

    Coordinates are computed per partition in metres using a local
    equirectangular approximation centered on the weighted latitude/longitude of
    the selected nodes. The returned coordinate columns are local north-south
    and east-west metre offsets, matching grid axes ``0`` and ``1`` for
    row/column split decisions.

    Attributes:
        latitudes: Finite two-dimensional latitude coordinate grid in degrees,
            aligned to grid node ``(row, col)`` indexing.
        longitudes: Finite two-dimensional longitude coordinate grid in
            degrees, with the same shape and node alignment as ``latitudes``.
        earth_radius_m: Earth radius used for converting angular differences to
            metres. Must be positive and finite.
    """

    latitudes: npt.NDArray[np.float64]
    longitudes: npt.NDArray[np.float64]
    earth_radius_m: float = 6_371_008.8

    @classmethod
    def from_dataarray(
        cls,
        data: xr.DataArray,
        *,
        lat_name: str = "lat",
        lon_name: str = "lon",
        earth_radius_m: float = 6_371_008.8,
    ) -> LatLonGridGeometry:
        """Create geometry from latitude and longitude coordinates.

        Args:
            data: Two-dimensional grid with latitude and longitude coordinates,
                ordered as ``(lat_name, lon_name)`` so node axes align with the
                returned north-south and east-west metre offsets.
            lat_name: Name of the latitude coordinate and first dimension.
            lon_name: Name of the longitude coordinate and second dimension.
            earth_radius_m: Earth radius used for metre scaling.

        Returns:
            Geometry aligned to ``data``.

        Raises:
            ValueError: If dimensions are not ordered as ``(lat_name,
                lon_name)`` or coordinates cannot be broadcast to the data grid.
        """
        if data.ndim != 2:
            raise ValueError("LatLonGridGeometry requires a two-dimensional grid.")
        if data.dims != (lat_name, lon_name):
            raise ValueError(
                f"LatLonGridGeometry requires grid dimensions ordered as ({lat_name!r}, {lon_name!r})."
            )
        if lat_name not in data.coords or lon_name not in data.coords:
            raise ValueError(f"Grid must define {lat_name!r} and {lon_name!r} coordinates.")

        latitudes, longitudes = xr.broadcast(data.coords[lat_name], data.coords[lon_name])
        try:
            latitudes = latitudes.broadcast_like(data).transpose(*data.dims)
            longitudes = longitudes.broadcast_like(data).transpose(*data.dims)
        except ValueError as exc:
            raise ValueError("Latitude and longitude coordinates must align to the data grid.") from exc

        return cls(
            latitudes=np.asarray(latitudes.to_numpy(), dtype=np.float64),
            longitudes=np.asarray(longitudes.to_numpy(), dtype=np.float64),
            earth_radius_m=earth_radius_m,
        )

    def __post_init__(self) -> None:
        """Validate and normalize geometry arrays.

        Raises:
            ValueError: If latitude/longitude arrays are not two-dimensional,
                aligned, finite arrays or the Earth radius is not positive and
                finite.
        """
        latitudes = np.asarray(self.latitudes, dtype=np.float64)
        longitudes = np.asarray(self.longitudes, dtype=np.float64)
        if latitudes.shape != longitudes.shape:
            raise ValueError("latitudes and longitudes must have the same shape.")
        if latitudes.ndim != 2:
            raise ValueError("latitudes and longitudes must be two-dimensional.")
        if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
            raise ValueError("latitudes and longitudes must be finite.")
        if self.earth_radius_m <= 0.0 or not np.isfinite(self.earth_radius_m):
            raise ValueError("earth_radius_m must be positive and finite.")

        self.latitudes = latitudes
        self.longitudes = longitudes

    def coordinates(
        self,
        nodes: GridPartition,
        node_weights: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64] | None:
        """Return local tangent-plane coordinates for ``nodes`` in metres.

        Args:
            nodes: Grid nodes in the selected partition.
            node_weights: Optional non-negative weights for the same nodes.
                These weights set the local projection center. Equal weights are
                used when weights are omitted or all zero.

        Returns:
            A finite ``(nnode, 2)`` array of local north-south and east-west
            metre offsets. The local center is the weighted mean latitude and
            circular weighted mean longitude for ``nodes``, so partitions near
            the antimeridian use the shorter wrapped longitude difference.
            Empty ``nodes`` returns an empty coordinate array. Invalid
            coordinates, out-of-bounds nodes, or invalid ``node_weights`` return
            ``None`` so callers can fall back to row/column index coordinates.
        """
        if not nodes:
            return np.empty((0, 2), dtype=np.float64)

        rows, cols = _node_indices(nodes)
        try:
            latitudes = self.latitudes[rows, cols].astype(np.float64)
            longitudes = self.longitudes[rows, cols].astype(np.float64)
        except IndexError:
            return None
        if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
            return None

        weights = _coordinate_weights(len(nodes), node_weights)
        if weights is None:
            return None

        lat0 = float(np.average(latitudes, weights=weights))
        lon0 = _weighted_circular_longitude(longitudes, weights)
        delta_lon = ((longitudes - lon0 + 180.0) % 360.0) - 180.0
        north_south_m = self.earth_radius_m * np.deg2rad(latitudes - lat0)
        east_west_m = self.earth_radius_m * np.deg2rad(delta_lon) * np.cos(np.deg2rad(lat0))
        coords = np.column_stack((north_south_m, east_west_m)).astype(np.float64)
        if not np.isfinite(coords).all():
            return None
        return coords


class SplitAcceptancePolicy(Protocol):
    """Policy protocol for accepting proposed child partitions."""

    def __call__(
        self,
        parent: GridPartition,
        children: list[GridPartition],
        weights: np.ndarray,
    ) -> bool:
        """Return true when proposed child partitions should be accepted.

        Args:
            parent: Parent partition selected by greedy orchestration.
            children: Non-empty child partitions proposed by a
                :class:`PartitionStep`.
            weights: Non-negative weight field aligned to the source grid.

        Returns:
            True if greedy orchestration should replace ``parent`` with
            ``children``. False freezes ``parent`` as a completed partition.
        """
        ...


class TargetSplitAcceptancePolicy(SplitAcceptancePolicy, Protocol):
    """Policy protocol for accepting splits using the class-local target count."""

    def __call__(
        self,
        parent: GridPartition,
        children: list[GridPartition],
        weights: np.ndarray,
        target_regions: int | None = None,
    ) -> bool:
        """Return true when proposed child partitions should be accepted.

        Args:
            parent: Parent partition selected by greedy orchestration.
            children: Non-empty child partitions proposed by a
                :class:`PartitionStep`.
            weights: Non-negative class/source-local weight field aligned to
                the source grid.
            target_regions: Requested class/source-local upper target count
                when available.

        Returns:
            True if greedy orchestration should replace ``parent`` with
            ``children``. False freezes ``parent`` as a completed partition.
        """
        ...

    def accept_split(
        self,
        parent: GridPartition,
        children: list[GridPartition],
        weights: np.ndarray,
        target_regions: int,
    ) -> bool:
        """Return true when proposed children should be accepted.

        Args:
            parent: Parent partition selected by greedy orchestration.
            children: Non-empty child partitions proposed by a
                :class:`PartitionStep`.
            weights: Non-negative class/source-local weight field aligned to
                the source grid.
            target_regions: Requested class/source-local upper target count.

        Returns:
            True if greedy orchestration should replace ``parent`` with
            ``children``. False freezes ``parent`` as a completed partition.
        """
        ...


SplitAcceptance: TypeAlias = SplitAcceptancePolicy | TargetSplitAcceptancePolicy


@dataclass(frozen=True)
class MinChildWeightShare:
    """Reject splits whose lightest child is below a parent-weight share.

    This is a split-balance guard. It compares children with their current
    parent partition, not with the total class/source weight being partitioned.
    """

    min_child_weight_share: float

    def __post_init__(self) -> None:
        """Validate the minimum child weight share threshold."""
        if not 0.0 <= self.min_child_weight_share <= 1.0:
            raise ValueError("min_child_weight_share must be between 0 and 1.")

    def __call__(
        self,
        parent: GridPartition,
        children: list[GridPartition],
        weights: np.ndarray,
    ) -> bool:
        """Return true when every child has enough parent weight share."""
        parent_weight = _node_weight(parent, weights)
        if parent_weight <= 0.0:
            parent_weight = float(len(parent))
            child_weights = [float(len(child)) for child in children]
        else:
            child_weights = [_node_weight(child, weights) for child in children]

        if parent_weight <= 0.0:
            return False
        return min(child_weights) / parent_weight >= self.min_child_weight_share


@dataclass(frozen=True)
class MinChildTargetWeightShare:
    """Reject splits whose lightest child is below an equal-target share.

    ``min_child_target_weight_share`` is compared with
    ``min(child_weight) / (weights.sum() / target_regions)`` for the
    class/source-local weights being partitioned. This policy stops creation of
    low-weight basis regions relative to the requested equal-weight target; it
    is not a parent-relative split-balance guard.
    """

    min_child_target_weight_share: float

    def __post_init__(self) -> None:
        """Validate the minimum child target weight share threshold."""
        if not 0.0 <= self.min_child_target_weight_share <= 1.0:
            raise ValueError("min_child_target_weight_share must be between 0 and 1.")

    def __call__(
        self,
        parent: GridPartition,
        children: list[GridPartition],
        weights: np.ndarray,
        target_regions: int | None = None,
    ) -> bool:
        """Return true when every child meets the equal-target threshold."""
        if target_regions is None:
            raise ValueError("target_regions is required for MinChildTargetWeightShare.")
        return self.accept_split(parent, children, weights, target_regions)

    def accept_split(
        self,
        parent: GridPartition,
        children: list[GridPartition],
        weights: np.ndarray,
        target_regions: int,
    ) -> bool:
        """Return true when every child is large enough to become a region.

        ``weights`` is the class/source-local field passed to greedy
        partitioning, so ``weights.sum() / target_regions`` is the equal-weight
        target region weight. If the total weight is zero, fall back to
        cell-count shares for direct policy use; the default greedy strategy
        already converts all-zero classes to an area surrogate before policies
        are evaluated.
        """
        del parent

        if target_regions < 1:
            raise ValueError("target_regions must be at least 1.")

        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            total_weight = float(weights.size)
            child_weights = [float(len(child)) for child in children]
        else:
            child_weights = [_node_weight(child, weights) for child in children]

        if total_weight <= 0.0:
            return False
        equal_target_weight = total_weight / target_regions
        if equal_target_weight <= 0.0:
            return False
        return min(child_weights) / equal_target_weight >= self.min_child_target_weight_share


@dataclass(frozen=True)
class MaxChildPCAEccentricity:
    """Reject splits that create child partitions above a PCA eccentricity limit.

    The eccentricity is computed from each child partition's unweighted node
    coordinates. If ``geometry`` is supplied, its physical coordinates are used;
    otherwise row/column index coordinates are used. Single-cell children have
    eccentricity ``1`` because they have no resolvable long axis. Multi-cell
    rank-one children have infinite eccentricity and are rejected by any finite
    threshold.
    """

    max_child_pca_eccentricity: float
    geometry: SplitGeometry | None = None
    tolerance: float = _INERTIAL_TOLERANCE

    def __post_init__(self) -> None:
        """Validate the eccentricity threshold and tolerance."""
        if self.max_child_pca_eccentricity < 1.0 or not np.isfinite(self.max_child_pca_eccentricity):
            raise ValueError("max_child_pca_eccentricity must be at least 1 and finite.")
        if self.tolerance < 0.0 or not np.isfinite(self.tolerance):
            raise ValueError("tolerance must be non-negative and finite.")

    def __call__(
        self,
        parent: GridPartition,
        children: list[GridPartition],
        weights: np.ndarray,
    ) -> bool:
        """Return true when every child is below the eccentricity threshold."""
        del parent, weights
        return all(
            _partition_pca_eccentricity(child, geometry=self.geometry, tolerance=self.tolerance)
            <= self.max_child_pca_eccentricity
            for child in children
        )


@dataclass(frozen=True, init=False)
class AllSplitAcceptancePolicies:
    """Accept a split only when every policy accepts it."""

    policies: tuple[SplitAcceptance, ...]

    def __init__(self, *policies: SplitAcceptance) -> None:
        """Create a policy that combines multiple acceptance policies."""
        if not policies:
            raise ValueError("At least one split acceptance policy is required.")
        object.__setattr__(self, "policies", tuple(policies))

    def __call__(
        self,
        parent: GridPartition,
        children: list[GridPartition],
        weights: np.ndarray,
        target_regions: int | None = None,
    ) -> bool:
        """Return true when all component policies accept the split."""
        return all(
            _split_acceptance_allows(policy, parent, children, weights, target_regions)
            for policy in self.policies
        )

    def accept_split(
        self,
        parent: GridPartition,
        children: list[GridPartition],
        weights: np.ndarray,
        target_regions: int,
    ) -> bool:
        """Return true when all component policies accept the split."""
        return self(parent, children, weights, target_regions)


@dataclass(frozen=True)
class AxisParallelSplitStep:
    """Split one partition along an axis-parallel line.

    This is a cleaned-up version of the prototype's axis-parallel split step.
    Greedy orchestration is handled separately by
    :class:`GreedyAxisParallelSplitStrategy`.

    Attributes:
        balanced: If true, choose the weighted long axis and split near half
            total node weight. If false, choose the geometric long axis and
            split by cell count.
        clean_splits: If true, keep all cells with the same selected-axis
            coordinate on the same side of the split.
        geometry: Optional geometry used to choose the split axis. The split
            itself remains a row- or column-aligned cut.
    """

    balanced: bool = True
    clean_splits: bool = False
    geometry: SplitGeometry | None = None

    def __call__(self, nodes: GridPartition, weights: np.ndarray) -> list[GridPartition]:
        """Return child partitions from one axis-parallel split.

        Args:
            nodes: Grid nodes in the partition being split.
            weights: Non-negative weight field aligned to the source grid.

        Returns:
            Two child partitions when the split succeeds, or the original
            partition when the input cannot be split without an empty side.
            When ``geometry`` is provided, it is used only to choose the split
            axis; child partitions are still produced by sorting row/column
            nodes along that axis.
        """
        left, right = _axis_parallel_split_nodes(
            nodes,
            weights,
            balanced=self.balanced,
            clean_splits=self.clean_splits,
            geometry=self.geometry,
        )
        if not left or not right:
            return [nodes]
        return [left, right]


@dataclass(frozen=True)
class InertialSplitStep:
    """Experimental split step using a weighted principal inertial axis.

    The split projects partition cells onto the principal axis of their
    weighted covariance, then cuts that one-dimensional ordering by weight or by
    count. This lets diagonal, rotated, or strongly anisotropic high-gradient
    structures split along their natural orientation instead of being forced
    through row/column cuts. The greedy class-local orchestrator still invokes
    this step independently inside each region class, so labels keep the same
    region-constrained boundary guarantees as axis-parallel splitting.

    By default the covariance uses grid-index coordinates. Pass
    ``geometry=LatLonGridGeometry.from_dataarray(...)`` to use local physical
    north-south and east-west metre offsets for each selected partition.
    Degenerate covariance, tied projections at the selected cut, and other
    numerically unstable cases fall back to an axis-parallel split.

    Attributes:
        balanced: If true, split the inertial projection near half total node
            weight. If false, split by cell count.
        geometry: Optional geometry used for the covariance and projection. The
            fallback split uses the same geometry for axis choice.
    """

    balanced: bool = True
    geometry: SplitGeometry | None = None

    def __call__(self, nodes: GridPartition, weights: np.ndarray) -> list[GridPartition]:
        """Return child partitions from one inertial-axis split.

        Args:
            nodes: Grid nodes in the partition being split.
            weights: Non-negative weight field aligned to the source grid.

        Returns:
            Two child partitions when the inertial split succeeds, or the
            original partition when neither inertial nor fallback splitting can
            produce two non-empty sides.
        """
        left, right = _inertial_split_nodes(
            nodes,
            weights,
            balanced=self.balanced,
            geometry=self.geometry,
        )
        if not left or not right:
            return [nodes]
        return [left, right]


def _connected_node_components(
    nodes: GridPartition,
    shape: tuple[int, int],
    *,
    connectivity: int,
) -> list[GridPartition]:
    """Return deterministic connected components for grid nodes."""
    if not nodes:
        return []
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 (edge) or 2 (edge and corner).")

    mask = np.zeros(shape, dtype=bool)
    rows, columns = zip(*nodes, strict=True)
    mask[np.asarray(rows), np.asarray(columns)] = True
    labels, count = ndimage.label(
        mask,
        structure=ndimage.generate_binary_structure(2, connectivity),
    )
    return [list(zip(*np.where(labels == component), strict=True)) for component in range(1, int(count) + 1)]


@dataclass(frozen=True)
class ConnectedComponentPartitionStep:
    """Make every child from another partition step spatially connected.

    This wrapper is useful for partition steps such as
    :class:`InertialSplitStep`, whose one-dimensional projection can assign
    spatially disconnected cells to the same child. Each proposed child is
    decomposed into deterministic connected components before the greedy
    orchestrator accepts it.

    Attributes:
        split_step: Partition step whose children should be made connected.
        connectivity: Two-dimensional neighbourhood definition. ``1`` uses
            edge-sharing (four-neighbour) connectivity and ``2`` additionally
            includes corner-sharing (eight-neighbour) connectivity.
    """

    split_step: PartitionStep
    connectivity: int = 1

    def __post_init__(self) -> None:
        """Validate the requested two-dimensional connectivity."""
        if self.connectivity not in (1, 2):
            raise ValueError("connectivity must be 1 (edge) or 2 (edge and corner).")

    def __call__(
        self,
        nodes: GridPartition,
        weights: np.ndarray,
    ) -> list[GridPartition]:
        """Split once, then separate disconnected pieces of every child."""
        children = self.split_step(nodes, weights)
        return [
            component
            for child in children
            for component in _connected_node_components(
                child,
                weights.shape,
                connectivity=self.connectivity,
            )
        ]


@dataclass(frozen=True)
class GreedyAxisParallelSplitStrategy:
    """Class-local greedy repeated-bisection strategy.

    This is a cleaned-up version of the prototype's axis-parallel partitioning
    algorithm. It repeatedly splits the highest-weight current part until the
    requested class-local region count is reached or no splittable parts remain.
    """

    balanced: bool = True
    clean_splits: bool = False
    split_step: PartitionStep | None = None
    split_acceptance: SplitAcceptance | None = None

    def __call__(
        self,
        weights: np.ndarray,
        class_mask: np.ndarray,
        target_regions: int,
    ) -> np.ndarray:
        """Return class-local labels using greedy repeated splitting.

        Args:
            weights: Non-negative weight field for the full grid.
            class_mask: Boolean mask selecting the cells in the class being
                split.
            target_regions: Requested upper target for local labels in this
                class. Split acceptance policies may stop before this count is
                reached.

        Returns:
            Integer label array with positive class-local labels inside
            ``class_mask`` and zero outside it.
        """
        if target_regions < 1:
            raise ValueError("target_regions must be at least 1.")
        if not class_mask.any():
            return np.zeros(weights.shape, dtype=np.int64)

        class_weights = np.where(class_mask, weights, 0.0)
        if float(class_weights.sum()) == 0.0:
            class_weights = class_mask.astype(np.float64)

        nodes = _node_list_from_mask(class_mask)
        split_step = self.split_step or AxisParallelSplitStep(
            balanced=self.balanced,
            clean_splits=self.clean_splits,
        )
        partition = _greedy_partitioning(
            [nodes],
            target_regions,
            class_weights,
            split_step=split_step,
            split_acceptance=self.split_acceptance,
        )
        return _labels_from_node_partition(partition, weights.shape)


@dataclass
class ConnectedComponentSplitStrategy:
    """Allocate and split each connected piece of a class independently.

    A disconnected class requires at least one label per connected component.
    If ``target_regions`` is below that geographic minimum, the effective
    target is raised rather than assigning one label to disconnected cells.
    Targets above the minimum are allocated across components by weight, with
    the existing area fallback for all-zero weights.

    Attributes:
        split_strategy: Class-local strategy applied separately to each
            connected component.
        connectivity: Two-dimensional neighbourhood definition. ``1`` uses
            edge-sharing (four-neighbour) connectivity and ``2`` additionally
            includes corner-sharing (eight-neighbour) connectivity.
        diagnostics: Per-call requested, minimum, effective, and actual region
            counts.
    """

    split_strategy: SplitStrategy
    connectivity: int = 1
    diagnostics: list[dict[str, int]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Validate the requested two-dimensional connectivity."""
        if self.connectivity not in (1, 2):
            raise ValueError("connectivity must be 1 (edge) or 2 (edge and corner).")

    def __call__(
        self,
        weights: np.ndarray,
        class_mask: np.ndarray,
        target_regions: int,
    ) -> np.ndarray:
        """Return connected labels, raising the target to the geographic minimum."""
        if target_regions < 1:
            raise ValueError("target_regions must be at least 1.")
        if weights.shape != class_mask.shape or weights.ndim != 2:
            raise ValueError("weights and class_mask must be aligned two-dimensional arrays.")
        if not class_mask.any():
            return np.zeros(weights.shape, dtype=np.int64)

        component_labels, component_count = ndimage.label(
            class_mask,
            structure=ndimage.generate_binary_structure(2, self.connectivity),
        )
        effective_target = min(
            max(int(target_regions), int(component_count)),
            int(class_mask.sum()),
        )
        dims = ("row", "column")
        coordinates = {
            "row": np.arange(weights.shape[0]),
            "column": np.arange(weights.shape[1]),
        }
        allocation = allocate_nbasis_by_class(
            xr.DataArray(weights, dims=dims, coords=coordinates),
            xr.DataArray(component_labels, dims=dims, coords=coordinates),
            effective_target,
            allocation="weight",
            min_regions_per_class=1,
            unmapped_values=(0,),
        )

        labels = np.zeros(weights.shape, dtype=np.int64)
        next_region = 1
        for component in range(1, int(component_count) + 1):
            component_mask = component_labels == component
            local_labels = self.split_strategy(
                weights,
                component_mask,
                allocation[component],
            )
            if not np.array_equal(local_labels > 0, component_mask):
                raise RuntimeError("Connected split strategy did not preserve class coverage.")
            for local_region in _positive_labels(local_labels):
                region_mask = local_labels == local_region
                _component_labels, region_component_count = ndimage.label(
                    region_mask,
                    structure=ndimage.generate_binary_structure(2, self.connectivity),
                )
                if int(region_component_count) != 1:
                    raise RuntimeError("Connected split strategy produced a disconnected label.")
                labels[region_mask] = next_region
                next_region += 1

        actual_regions = next_region - 1
        self.diagnostics.append(
            {
                "requested_target": int(target_regions),
                "connected_component_minimum": int(component_count),
                "effective_target": int(effective_target),
                "actual_regions": int(actual_regions),
            }
        )
        return labels


@dataclass(frozen=True)
class AxisAlignedWeightedSplitStrategy:
    """Compatibility strategy based on the existing recursive weighted basis.

    This keeps the current weighted basis shape available for comparison:
    recursively split rectangles along the longer axis until each rectangle is
    below a searched threshold. New constrained code defaults to
    :class:`GreedyAxisParallelSplitStrategy` instead.
    """

    max_iter: int = 32

    def __call__(
        self,
        weights: np.ndarray,
        class_mask: np.ndarray,
        target_regions: int,
    ) -> np.ndarray:
        """Return class-local labels using recursive weighted bucket splits.

        Args:
            weights: Non-negative weight field for the full grid.
            class_mask: Boolean mask selecting the cells in the class being
                split.
            target_regions: Requested number of local labels for this class.

        Returns:
            Integer label array with positive class-local labels inside
            ``class_mask`` and zero outside it.
        """
        if target_regions < 1:
            raise ValueError("target_regions must be at least 1.")
        if not class_mask.any():
            return np.zeros(weights.shape, dtype=np.int64)

        class_weights = np.where(class_mask, weights, 0.0)
        total_weight = float(class_weights.sum())
        if total_weight == 0.0:
            class_weights = class_mask.astype(np.float64)
            total_weight = float(class_weights.sum())

        low = 0.0
        high = total_weight
        best = _labels_for_bucket(class_weights, class_mask, bucket=high)
        best_error = abs(_count_positive_labels(best) - target_regions)

        for _ in range(self.max_iter):
            bucket = (low + high) / 2.0
            labels = _labels_for_bucket(class_weights, class_mask, bucket=bucket)
            nregions = _count_positive_labels(labels)
            error = abs(nregions - target_regions)
            if error < best_error:
                best = labels
                best_error = error
                if error == 0:
                    break

            if nregions > target_regions:
                low = bucket
            else:
                high = bucket

        return best


def region_constrained_basis(
    weights: xr.DataArray,
    region_classes: xr.DataArray,
    nbasis: NbasisAllocation,
    *,
    allocation: AllocationMode = "weight",
    min_regions_per_class: int = 1,
    split_strategy: SplitStrategy | None = None,
    component_consolidation: ComponentConsolidationPolicy | None = None,
    unmapped_values: Iterable[Hashable] = (),
) -> xr.DataArray:
    """Generate basis labels independently inside each mask/region class.

    Args:
        weights: Two-dimensional non-negative weight field.
        region_classes: Two-dimensional class field on the same grid as
            ``weights``. Each non-null value is treated as a mapped class unless
            listed in ``unmapped_values``.
        nbasis: Either a total number of basis regions to allocate across
            classes, or an explicit mapping from class value to class-local
            region target.
        allocation: Automatic allocation mode used when ``nbasis`` is an
            integer. ``"weight"`` allocates proportional to class total weight,
            falling back to area if all class weights are zero. ``"area"``
            allocates proportional to mapped cell count.
        min_regions_per_class: Minimum automatic allocation for each non-empty
            mapped class. If the requested total is smaller than this minimum
            requires, a ``ValueError`` is raised.
        split_strategy: Class-local splitting strategy. Defaults to
            :class:`GreedyAxisParallelSplitStrategy`.
        component_consolidation: Optional policy applied to the globally
            relabelled basis after class-local construction. Policies that
            deliberately combine disconnected components must preserve class
            boundaries and report that strict connectivity no longer holds.
        unmapped_values: Additional class values to leave as output label ``0``.

    Returns:
        ``xarray.DataArray`` with the same dimensions and coordinates as
        ``weights``. Mapped cells receive globally unique positive integer
        labels; unmapped cells receive ``0``.

    Notes:
        Labels are guaranteed not to cross class boundaries because each class is
        split independently and relabelled with a global offset. The default
        strategy can assign one label to disconnected pieces of the
        same class if the class mask itself is disconnected; contiguity is not
        guaranteed by this helper.
    """
    weights, region_classes = _align_2d_inputs(weights, region_classes)
    weight_values = _validate_weights(weights)
    class_values = region_classes.to_numpy()
    mapped_classes = _mapped_classes(class_values, unmapped_values)
    strategy = split_strategy or GreedyAxisParallelSplitStrategy()

    labels = np.zeros(weight_values.shape, dtype=np.int64)
    if not mapped_classes:
        return _labels_dataarray(labels, weights)

    targets = allocate_nbasis_by_class(
        weights,
        region_classes,
        nbasis,
        allocation=allocation,
        min_regions_per_class=min_regions_per_class,
        unmapped_values=unmapped_values,
    )

    next_label = 1
    for class_value in mapped_classes:
        target_regions = targets[class_value]
        class_mask = _class_value_mask(class_values, class_value)
        local_labels = strategy(weight_values, class_mask, target_regions)
        for local_label in _positive_labels(local_labels):
            labels[(local_labels == local_label) & class_mask] = next_label
            next_label += 1

    result = _labels_dataarray(labels, weights)
    if component_consolidation is not None:
        result = component_consolidation(result, region_classes)
    return result


def intersect_region_class_layers(
    layers: Mapping[Hashable, xr.DataArray],
    *,
    unmapped_values: Iterable[Hashable] = (),
    name: str = "region_classes",
) -> xr.DataArray:
    """Intersect aligned region-class layers into composite class labels.

    Args:
        layers: Ordered mapping from layer name to two-dimensional class field.
            Mapping insertion order defines the order of values in each output
            class tuple.
        unmapped_values: Layer values that should leave the output cell
            unmapped. Null values in any layer are always unmapped.
        name: Name for the returned ``DataArray``.

    Returns:
        Object-valued ``DataArray`` with the same dimensions and coordinates as
        the first layer. Mapped cells contain tuples of layer values, while
        cells that are null or explicitly unmapped in any layer contain
        ``NaN``.

    Raises:
        ValueError: If no layers are supplied, any layer is not two-dimensional,
            layer dimension names differ, or a mapped layer value is not
            hashable.
        xarray.AlignmentError: If layer coordinates do not align exactly.

    Notes:
        The tuple labels can be passed directly to
        :func:`region_constrained_basis`. This is the small lattice-style
        construction needed for layered masks such as land/sea crossed with an
        inner/outer rectangle.
    """
    layer_items = list(layers.items())
    if not layer_items:
        raise ValueError("At least one region-class layer must be supplied.")

    template = layer_items[0][1]
    if template.ndim != 2:
        raise ValueError("Region-class layers must be two-dimensional.")

    aligned_layers: list[xr.DataArray] = []
    for layer_name, layer in layer_items:
        if layer.ndim != 2:
            raise ValueError(f"Region-class layer {layer_name!r} must be two-dimensional.")
        if set(layer.dims) != set(template.dims):
            raise ValueError("Region-class layers must use the same dimensions.")
        aligned_layers.append(layer.transpose(*template.dims))

    aligned_layers = list(xr.align(*aligned_layers, join="exact"))
    layer_values = [layer.to_numpy() for layer in aligned_layers]
    unmapped = set(unmapped_values)
    class_values = np.empty(template.shape, dtype=object)
    class_values[:] = np.nan

    for index in np.ndindex(template.shape):
        values: list[Hashable] = []
        mapped = True
        for layer in layer_values:
            value = cast(Hashable, layer[index])
            if _is_unmapped_layer_value(value, unmapped):
                mapped = False
                break
            values.append(value)
        if mapped:
            class_values[index] = tuple(values)

    return xr.DataArray(
        class_values,
        dims=template.dims,
        coords=template.coords,
        name=name,
        attrs={"region_class_layers": tuple(str(layer_name) for layer_name, _ in layer_items)},
    )


def allocate_nbasis_by_class(
    weights: xr.DataArray,
    region_classes: xr.DataArray,
    nbasis: NbasisAllocation,
    *,
    allocation: AllocationMode = "weight",
    min_regions_per_class: int = 1,
    unmapped_values: Iterable[Hashable] = (),
) -> dict[Hashable, int]:
    """Allocate class-local region targets for constrained basis generation.

    Args:
        weights: Two-dimensional non-negative weight field.
        region_classes: Two-dimensional class field aligned to ``weights``.
        nbasis: Total number of regions to distribute, or an explicit mapping
            from class value to class-local region target.
        allocation: Automatic allocation mode. ``"weight"`` uses class total
            weight, falling back to area if all mapped weights are zero.
            ``"area"`` uses mapped cell count.
        min_regions_per_class: Minimum automatic allocation for each non-empty
            mapped class.
        unmapped_values: Additional class values to leave unallocated.

    Returns:
        Mapping from mapped class value to target number of local regions.

    Raises:
        ValueError: If inputs cannot be aligned, weights are invalid, or the
            requested allocation is impossible.
    """
    if min_regions_per_class < 0:
        raise ValueError("min_regions_per_class must be non-negative.")
    weights, region_classes = _align_2d_inputs(weights, region_classes)
    weight_values = _validate_weights(weights)
    class_values = region_classes.to_numpy()
    mapped_classes = _mapped_classes(class_values, unmapped_values)

    if not mapped_classes:
        if not isinstance(nbasis, Mapping) and nbasis != 0:
            raise ValueError("Cannot allocate basis regions without mapped classes.")
        return {}

    capacities = {
        class_value: int(np.count_nonzero(_class_value_mask(class_values, class_value)))
        for class_value in mapped_classes
    }

    if isinstance(nbasis, Mapping):
        return _explicit_allocation(mapped_classes, nbasis, capacities)

    if nbasis < 0:
        raise ValueError("nbasis must be non-negative.")

    minima = {
        class_value: min(min_regions_per_class, capacity) for class_value, capacity in capacities.items()
    }

    min_total = sum(minima.values())
    max_total = sum(capacities.values())
    if nbasis < min_total:
        raise ValueError(
            f"nbasis={nbasis} is smaller than the minimum {min_total} required "
            f"for {len(mapped_classes)} mapped classes."
        )
    if nbasis > max_total:
        raise ValueError(f"nbasis={nbasis} exceeds the {max_total} mapped cells available for splitting.")

    scores = _allocation_scores(weight_values, class_values, mapped_classes, allocation)
    return _distribute_regions(mapped_classes, scores, minima, capacities, nbasis)


def _align_2d_inputs(
    weights: xr.DataArray,
    region_classes: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Validate, transpose, and exactly align the 2D weight and class fields.

    Args:
        weights: Two-dimensional weight field whose dimension order defines the
            output dimension order.
        region_classes: Two-dimensional class field using the same dimension
            names and coordinates as ``weights``.

    Returns:
        ``weights`` and ``region_classes`` with identical coordinates and
        ``region_classes`` transposed to match ``weights``.

    Raises:
        ValueError: If either input is not 2D or the dimension names differ.
        xarray.AlignmentError: If coordinates do not match exactly.
    """
    if weights.ndim != 2:
        raise ValueError("weights must be two-dimensional.")
    if region_classes.ndim != 2:
        raise ValueError("region_classes must be two-dimensional.")
    if set(weights.dims) != set(region_classes.dims):
        raise ValueError("weights and region_classes must use the same dimensions.")

    region_classes = region_classes.transpose(*weights.dims)
    weights, region_classes = xr.align(weights, region_classes, join="exact")
    return weights, region_classes


def _validate_weights(weights: xr.DataArray) -> np.ndarray:
    """Return finite non-negative weights as a float NumPy array."""
    values = np.asarray(weights.to_numpy(), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite.")
    if (values < 0.0).any():
        raise ValueError("weights must be non-negative.")
    return values


def _mapped_classes(
    class_values: npt.NDArray[np.object_] | np.ndarray,
    unmapped_values: Iterable[Hashable],
) -> list[Hashable]:
    """Return unique class values that should receive basis labels.

    Args:
        class_values: Raw class array from the aligned ``region_classes`` input.
        unmapped_values: Additional class values that should remain output label
            ``0``.

    Returns:
        Unique class values in first-seen order, excluding nulls and explicitly
        unmapped values.
    """
    unmapped = set(unmapped_values)
    classes: list[Hashable] = []
    for value in pd.unique(class_values.ravel()):
        if _is_unmapped_layer_value(cast(Hashable, value), unmapped):
            continue
        classes.append(cast(Hashable, value))
    return classes


def _is_unmapped_layer_value(value: Hashable, unmapped_values: set[Hashable]) -> bool:
    """Return true when a layer value should not be mapped."""
    isna = pd.isna(value)
    if isinstance(isna, (bool, np.bool_)) and bool(isna):
        return True
    try:
        return value in unmapped_values
    except TypeError as exc:
        raise ValueError(f"Region-class layer value {value!r} is not hashable.") from exc


def _class_value_mask(class_values: np.ndarray, class_value: Hashable) -> np.ndarray:
    """Return a Boolean mask for one class value, including tuple labels."""
    if isinstance(class_value, tuple):
        return np.asarray(
            np.frompyfunc(lambda value: value == class_value, 1, 1)(class_values),
            dtype=bool,
        )

    try:
        mask = class_values == class_value
    except ValueError:
        mask = np.frompyfunc(lambda value: value == class_value, 1, 1)(class_values)
    if not isinstance(mask, np.ndarray) or mask.shape != class_values.shape:
        mask = np.frompyfunc(lambda value: value == class_value, 1, 1)(class_values)
    return np.asarray(mask, dtype=bool)


def _explicit_allocation(
    mapped_classes: list[Hashable],
    nbasis: Mapping[Hashable, int],
    capacities: Mapping[Hashable, int],
) -> dict[Hashable, int]:
    """Validate and normalize a caller-supplied per-class allocation.

    Args:
        mapped_classes: Mapped class values that need allocations.
        nbasis: Caller-supplied mapping from class value to region target.
        capacities: Maximum possible labels per class, currently the number of
            mapped cells for that class.

    Returns:
        Integer allocation for each mapped class, preserving ``mapped_classes``
        order.

    Raises:
        ValueError: If an allocation is missing, non-positive, or greater than
            class capacity.
    """
    missing = [class_value for class_value in mapped_classes if class_value not in nbasis]
    if missing:
        raise ValueError(f"Explicit nbasis allocation is missing classes: {missing!r}.")

    allocations = {class_value: int(nbasis[class_value]) for class_value in mapped_classes}
    invalid = {class_value: target for class_value, target in allocations.items() if target < 1}
    if invalid:
        raise ValueError(f"Class allocations must be positive for mapped classes: {invalid!r}.")
    over_allocated = {
        class_value: target for class_value, target in allocations.items() if target > capacities[class_value]
    }
    if over_allocated:
        raise ValueError(f"Class allocations exceed mapped cell counts: {over_allocated!r}.")
    return allocations


def _allocation_scores(
    weights: np.ndarray,
    class_values: np.ndarray,
    mapped_classes: list[Hashable],
    allocation: AllocationMode,
) -> dict[Hashable, float]:
    """Compute proportional allocation scores for each mapped class.

    Args:
        weights: Non-negative weight values aligned to ``class_values``.
        class_values: Raw class values aligned to ``weights``.
        mapped_classes: Class values eligible for allocation.
        allocation: ``"weight"`` to score by total class weight, or ``"area"``
            to score by mapped cell count.

    Returns:
        Non-negative score for each mapped class.

    Raises:
        ValueError: If ``allocation`` is not supported.
    """
    if allocation == "area":
        return {
            class_value: float(np.count_nonzero(_class_value_mask(class_values, class_value)))
            for class_value in mapped_classes
        }
    if allocation != "weight":
        raise ValueError("allocation must be 'weight' or 'area'.")

    scores = {
        class_value: float(weights[_class_value_mask(class_values, class_value)].sum())
        for class_value in mapped_classes
    }
    if sum(scores.values()) == 0.0:
        return _allocation_scores(weights, class_values, mapped_classes, "area")
    return scores


def _distribute_regions(
    mapped_classes: list[Hashable],
    scores: Mapping[Hashable, float],
    minima: Mapping[Hashable, int],
    capacities: Mapping[Hashable, int],
    nbasis: int,
) -> dict[Hashable, int]:
    """Distribute remaining regions by score while respecting minima and capacity.

    Args:
        mapped_classes: Class values in deterministic allocation order.
        scores: Proportional allocation scores per class.
        minima: Initial allocation per class.
        capacities: Maximum allocation per class.
        nbasis: Total number of regions requested across all mapped classes.

    Returns:
        Allocation whose values sum to ``nbasis`` unless capacities are already
        exhausted by validated inputs.
    """
    allocations = dict(minima)
    remaining = nbasis - sum(allocations.values())

    while remaining > 0:
        candidates = [
            class_value
            for class_value in mapped_classes
            if allocations[class_value] < capacities[class_value]
        ]
        if not candidates:
            break

        score_total = sum(scores[class_value] for class_value in candidates)
        if score_total == 0.0:
            score_total = float(len(candidates))
            candidate_scores = {class_value: 1.0 for class_value in candidates}
        else:
            candidate_scores = {class_value: scores[class_value] for class_value in candidates}

        quotas = {
            class_value: remaining * candidate_scores[class_value] / score_total for class_value in candidates
        }
        extras = {
            class_value: min(
                int(np.floor(quota)),
                capacities[class_value] - allocations[class_value],
            )
            for class_value, quota in quotas.items()
        }

        added = sum(extras.values())
        if added == 0:
            class_value = max(
                candidates,
                key=lambda value: (
                    quotas[value],
                    capacities[value] - allocations[value],
                    str(value),
                ),
            )
            extras[class_value] = 1
            added = 1

        for class_value, extra in extras.items():
            allocations[class_value] += extra
        remaining -= added

    return allocations


def _node_list_from_mask(class_mask: np.ndarray) -> GridPartition:
    """Return grid-index nodes selected by a Boolean class mask."""
    return list(zip(*np.where(class_mask)))


def _labels_from_node_partition(partition: list[GridPartition], shape: tuple[int, ...]) -> np.ndarray:
    """Convert node partitions to a dense integer label array.

    Args:
        partition: Sequence of node lists. Each node list becomes one positive
            label in output order.
        shape: Shape of the output label array.

    Returns:
        Integer label array with label ``0`` outside all partition nodes.
    """
    labels = np.zeros(shape, dtype=np.int64)
    for label, nodes in enumerate(partition, start=1):
        rows, cols = _node_indices(nodes)
        labels[rows, cols] = label
    return labels


@dataclass(order=True)
class _PrioritizedPartition:
    """Priority queue item for selecting the next partition to split."""

    priority: tuple[float, int, int]
    nodes: GridPartition = field(compare=False)


class _PartitionPriorityQueue:
    """Priority queue that pops the highest-weight partition first."""

    def __init__(self, weights: np.ndarray) -> None:
        """Create a partition priority queue ranked by weight and size.

        Args:
            weights: Non-negative weight field used to rank partitions.
        """
        self._queue: PriorityQueue[_PrioritizedPartition] = PriorityQueue()
        self._weights = weights
        self._counter = 0

    def __bool__(self) -> bool:
        """Return true when the queue contains at least one partition."""
        return not self._queue.empty()

    def push(self, nodes: GridPartition) -> None:
        """Insert a partition into the priority queue."""
        if not nodes:
            return
        priority = (-_node_weight(nodes, self._weights), -len(nodes), self._counter)
        self._counter += 1
        self._queue.put_nowait(_PrioritizedPartition(priority, nodes))

    def pop(self) -> GridPartition:
        """Remove and return the highest-priority partition."""
        return self._queue.get_nowait().nodes


def _greedy_partitioning(
    init_partition: list[GridPartition],
    target_regions: int,
    weights: np.ndarray,
    *,
    split_step: PartitionStep,
    split_acceptance: SplitAcceptance | None = None,
) -> list[GridPartition]:
    """Apply a partition step greedily until a target count is reached.

    Args:
        init_partition: Initial list of partitions to refine. For class-local
            splitting this normally contains one node list for the class.
        target_regions: Desired number of output partitions.
        weights: Non-negative weight field used for part ranking and passed to
            ``split_step``.
        split_step: Callable that splits one selected partition into child
            partitions. Returning fewer than two non-empty children marks the
            selected partition as done.
        split_acceptance: Optional policy applied after ``split_step`` proposes
            non-empty children and before those children are accepted. Rejected
            splits freeze the selected parent partition.

    Returns:
        List of output partitions. The result may contain fewer than
        ``target_regions`` entries when no active partition can be split further
        or when split acceptance rejects the remaining candidates.
    """
    active = _PartitionPriorityQueue(weights)
    done: list[GridPartition] = []
    current_regions = 0

    for nodes in init_partition:
        if not nodes:
            continue
        current_regions += 1
        if len(nodes) > 1:
            active.push(nodes)
        else:
            done.append(nodes)

    while current_regions < target_regions and active:
        nodes = active.pop()
        child_partitions = [child for child in split_step(nodes, weights) if child]

        if len(child_partitions) < 2:
            done.append(nodes)
            continue
        if current_regions - 1 + len(child_partitions) > target_regions:
            done.append(nodes)
            continue
        if split_acceptance is not None and not _split_acceptance_allows(
            split_acceptance,
            nodes,
            child_partitions,
            weights,
            target_regions,
        ):
            done.append(nodes)
            continue

        current_regions -= 1
        for subnodes in child_partitions:
            current_regions += 1
            if len(subnodes) > 1:
                active.push(subnodes)
            else:
                done.append(subnodes)

    while active:
        done.append(active.pop())

    return done


def _axis_parallel_split_nodes(
    nodes: GridPartition,
    weights: np.ndarray,
    *,
    balanced: bool,
    clean_splits: bool,
    geometry: SplitGeometry | None = None,
) -> tuple[GridPartition, GridPartition]:
    """Split nodes along an axis-parallel line.

    Args:
        nodes: Grid-index nodes in the part being split.
        weights: Non-negative weight field aligned to the source grid.
        balanced: If true, choose the weighted long axis and split near half
            total node weight. If false, choose the geometric long axis and
            split by cell count.
        clean_splits: If true, keep equal selected-axis coordinates together,
            even when that makes the split degenerate.
        geometry: Optional physical geometry used only to choose the split
            axis. Cuts still follow row/column order.

    Returns:
        Two node lists. Either side may be empty for degenerate input or a
        degenerate clean split.
    """
    if len(nodes) < 2:
        return nodes, []

    axis = (
        _long_axis_weighted(nodes, weights, geometry=geometry)
        if balanced
        else _long_axis(nodes, geometry=geometry)
    )
    ordered = sorted(nodes, key=lambda node: (node[axis], node[1 - axis]))

    if balanced:
        rows, cols = _node_indices(ordered)
        split_index = _idx_of_half_cumsum(weights[rows, cols])
    else:
        split_index = len(ordered) // 2

    split_index = min(max(split_index, 1), len(ordered) - 1)
    if clean_splits:
        threshold = ordered[split_index - 1][axis]
        left = [node for node in ordered if node[axis] <= threshold]
        right = [node for node in ordered if node[axis] > threshold]
        return left, right

    return ordered[:split_index], ordered[split_index:]


def _inertial_split_nodes(
    nodes: GridPartition,
    weights: np.ndarray,
    *,
    balanced: bool,
    geometry: SplitGeometry | None = None,
) -> tuple[GridPartition, GridPartition]:
    """Split nodes along their weighted principal inertial axis.

    The implementation treats the row/column node coordinates as point masses,
    orders cells by projection onto the dominant weighted covariance axis, and
    splits that ordering near half total weight or half cell count. Compared
    with row/column splits, this can preserve diagonal or rotated structures
    while still returning ordinary node partitions to the greedy constrained
    strategy.

    Degenerate geometry, tied projections at the selected cut, or numerically
    unstable inertial fits fall back to an axis-parallel split so callers always
    get deterministic behavior.

    Args:
        nodes: Grid-index nodes in the partition being split.
        weights: Non-negative weight field aligned to the source grid.
        balanced: If true, split near half total node weight. If false, split by
            cell count.
        geometry: Optional physical geometry used for covariance/projection and
            for fallback axis choice.

    Returns:
        Two child partitions. Either side may be empty only if both inertial and
        fallback splitting are degenerate.
    """
    fallback = _axis_parallel_split_nodes(
        nodes,
        weights,
        balanced=balanced,
        clean_splits=False,
        geometry=geometry,
    )
    if len(nodes) < 3:
        return fallback

    inertial_order = _inertial_ordered_nodes(nodes, weights, geometry=geometry)
    if inertial_order is None:
        return fallback

    ordered, projections = inertial_order
    if balanced:
        rows, cols = _node_indices(ordered)
        split_index = _idx_of_half_cumsum(weights[rows, cols])
    else:
        split_index = len(ordered) // 2

    split_index = min(max(split_index, 1), len(ordered) - 1)
    if _projection_tie_at_split(projections, split_index):
        return fallback

    left = ordered[:split_index]
    right = ordered[split_index:]
    if not left or not right:
        return fallback
    return left, right


def _inertial_ordered_nodes(
    nodes: GridPartition,
    weights: np.ndarray,
    *,
    geometry: SplitGeometry | None = None,
) -> tuple[GridPartition, npt.NDArray[np.float64]] | None:
    """Return nodes ordered by projection onto the weighted inertial axis.

    Args:
        nodes: Grid-index nodes in the partition being split.
        weights: Non-negative weight field aligned to the source grid.
        geometry: Optional physical geometry used to compute projection
            coordinates.

    Returns:
        A pair of ordered nodes and their projections, where projection values
        are aligned element-for-element with the ordered nodes. Returns ``None``
        when no stable projection ordering is available and callers should use
        fallback splitting.
    """
    rows, cols = _node_indices(nodes)
    node_weights = weights[rows, cols].astype(np.float64)
    axis_and_centroid = _weighted_inertial_axis(nodes, node_weights, geometry=geometry)
    if axis_and_centroid is None:
        return None

    axis, centroid = axis_and_centroid
    coords = _node_coordinates(nodes, geometry=geometry, node_weights=node_weights)
    projections = (coords - centroid) @ axis
    if not np.isfinite(projections).all() or float(np.ptp(projections)) <= _INERTIAL_TOLERANCE:
        return None

    order = sorted(
        range(len(nodes)),
        key=lambda index: (float(projections[index]), nodes[index][0], nodes[index][1]),
    )
    ordered_nodes = [nodes[index] for index in order]
    ordered_projections = projections[order]
    return ordered_nodes, ordered_projections


def _weighted_inertial_axis(
    nodes: GridPartition,
    node_weights: npt.NDArray[np.float64],
    *,
    geometry: SplitGeometry | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Return the principal weighted inertial axis and centroid.

    The current problem is only a 2D row/column covariance, so a closed-form
    axis formula would be possible. ``np.linalg.eigh`` is intentionally kept
    here because the covariance is symmetric, the arrays are tiny, and the
    symmetric eigensolver avoids slope/division edge cases while leaving a
    straightforward path to future higher-dimensional coordinate spaces such as
    lat/lon/time basis functions.

    Args:
        nodes: Grid-index nodes in the partition being split.
        node_weights: One non-negative weight for each node.
        geometry: Optional geometry used to map nodes into coordinates before
            fitting the weighted covariance. Without geometry, row/column index
            coordinates are used.

    Returns:
        A normalized principal-axis vector and the weighted centroid in the
        same coordinate units. Returns ``None`` when weights, geometry,
        covariance, or eigenvectors are degenerate.
    """
    if not np.isfinite(node_weights).all():
        return None

    total_weight = float(node_weights.sum())
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        return None

    coords = _node_coordinates(nodes, geometry=geometry, node_weights=node_weights)
    pca = _coordinate_pca(coords, node_weights=node_weights)
    if pca is None:
        return None

    mxy = float((node_weights * pca.centered[:, 0] * pca.centered[:, 1]).sum())
    if np.isclose(mxy, 0.0, rtol=_INERTIAL_TOLERANCE, atol=_INERTIAL_TOLERANCE):
        return None

    axis_index = int(np.argmax(pca.eigenvalues))
    if float(pca.eigenvalues[axis_index]) <= _INERTIAL_TOLERANCE:
        return None
    eigenvalue_gap = float(pca.eigenvalues[axis_index] - pca.eigenvalues[1 - axis_index])
    if eigenvalue_gap <= _INERTIAL_TOLERANCE * max(1.0, abs(float(pca.eigenvalues[axis_index]))):
        return None

    axis = pca.eigenvectors[:, axis_index].astype(np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= _INERTIAL_TOLERANCE or not np.isfinite(axis_norm):
        return None

    axis = _canonical_inertial_axis(axis / axis_norm)
    return axis, pca.centroid


def _coordinate_pca(
    coords: npt.NDArray[np.float64],
    *,
    node_weights: npt.NDArray[np.float64] | None = None,
) -> _CoordinatePCA | None:
    """Return a PCA fit for finite partition coordinates."""
    if coords.ndim != 2 or coords.shape[0] == 0 or not np.isfinite(coords).all():
        return None

    if node_weights is None:
        centroid = coords.mean(axis=0)
        centered = coords - centroid
        covariance = centered.T @ centered / len(coords)
    else:
        weights = np.asarray(node_weights, dtype=np.float64).reshape(-1)
        if weights.shape != (coords.shape[0],) or not np.isfinite(weights).all():
            return None
        total_weight = float(weights.sum())
        if total_weight <= 0.0 or not np.isfinite(total_weight):
            return None
        weight_column = weights.reshape(-1, 1)
        centroid = (coords * weight_column).sum(axis=0) / total_weight
        centered = coords - centroid
        covariance = centered.T @ (centered * weight_column) / total_weight

    if not np.isfinite(covariance).all():
        return None

    # ``eigh`` is a stable symmetric eigensolver for this tiny covariance and
    # keeps the implementation dimension-agnostic if coordinates grow past 2D.
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return None

    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        return None
    return _CoordinatePCA(
        centroid=centroid.astype(np.float64),
        centered=centered.astype(np.float64),
        eigenvalues=eigenvalues.astype(np.float64),
        eigenvectors=eigenvectors.astype(np.float64),
    )


def _canonical_inertial_axis(axis: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Flip an eigenvector to a deterministic orientation for sorting."""
    nonzero = np.flatnonzero(np.abs(axis) > _INERTIAL_TOLERANCE)
    if len(nonzero) > 0 and axis[int(nonzero[0])] < 0.0:
        return -axis
    return axis


def _projection_tie_at_split(projections: npt.NDArray[np.float64], split_index: int) -> bool:
    """Return true when an inertial split would divide equal projections."""
    return bool(
        np.isclose(
            projections[split_index - 1],
            projections[split_index],
            rtol=_INERTIAL_TOLERANCE,
            atol=_INERTIAL_TOLERANCE,
        )
    )


def _idx_of_half_cumsum(weights: npt.ArrayLike) -> int:
    """Return the split index whose prefix/suffix weight sums are closest."""
    weight_values = np.asarray(weights, dtype=np.float64).ravel()
    if len(weight_values) < 2:
        return len(weight_values)

    sum1 = 0.0
    sum2 = float(weight_values.sum())
    idx = 0
    while idx < len(weight_values) and sum1 < sum2:
        sum1 += float(weight_values[idx])
        sum2 -= float(weight_values[idx])
        idx += 1

    if idx > 0:
        old_sum1 = sum1 - float(weight_values[idx - 1])
        old_sum2 = sum2 + float(weight_values[idx - 1])
        if (sum1 - sum2) > (old_sum2 - old_sum1):
            idx -= 1

    return min(max(idx, 1), len(weight_values) - 1)


def _long_axis(nodes: GridPartition, *, geometry: SplitGeometry | None = None) -> int:
    """Return the grid axis with the largest geometric spread."""
    if not nodes:
        return 0
    coords = _node_coordinates(nodes, geometry=geometry)
    return int(np.argmax(coords.max(axis=0) - coords.min(axis=0)))


def _long_axis_weighted(
    nodes: GridPartition,
    weights: np.ndarray,
    *,
    geometry: SplitGeometry | None = None,
) -> int:
    """Return the grid axis with the largest weighted absolute spread."""
    if not nodes:
        return 0
    rows, cols = _node_indices(nodes)
    node_weights = weights[rows, cols].astype(np.float64)
    total_weight = float(node_weights.sum())
    if total_weight == 0.0:
        return _long_axis(nodes, geometry=geometry)

    coords = _node_coordinates(nodes, geometry=geometry, node_weights=node_weights)
    weight_column = node_weights.reshape(-1, 1)
    centroid = (coords * weight_column).sum(axis=0) / total_weight
    spread = (weight_column * np.abs(coords - centroid)).sum(axis=0) / total_weight
    return int(np.argmax(spread))


def _node_coordinates(
    nodes: GridPartition,
    *,
    geometry: SplitGeometry | None = None,
    node_weights: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Return physical coordinates when available, otherwise row/column indices.

    Args:
        nodes: Grid-index nodes to map into coordinate space.
        geometry: Optional geometry object. If it returns invalid or missing
            coordinates, this helper falls back to index coordinates.
        node_weights: Optional node weights forwarded to ``geometry`` for
            partition-local coordinate centering.

    Returns:
        A finite ``(len(nodes), 2)`` coordinate array. The fallback coordinate
        columns are raw row and column indices.
    """
    if geometry is not None:
        coords = geometry.coordinates(nodes, node_weights)
        if coords is not None:
            coord_values = np.asarray(coords, dtype=np.float64)
            if coord_values.shape == (len(nodes), 2) and np.isfinite(coord_values).all():
                return coord_values
    return np.asarray(nodes, dtype=np.float64)


def _partition_pca_eccentricity(
    nodes: GridPartition,
    *,
    geometry: SplitGeometry | None = None,
    tolerance: float = _INERTIAL_TOLERANCE,
) -> float:
    """Return the square-root PCA variance ratio for one partition shape."""
    if len(nodes) < 2:
        return 1.0

    coords = _node_coordinates(nodes, geometry=geometry)
    pca = _coordinate_pca(coords)
    if pca is None:
        return np.inf

    minor = float(max(pca.eigenvalues[0], 0.0))
    major = float(max(pca.eigenvalues[-1], 0.0))
    if major <= tolerance:
        return 1.0
    if minor <= tolerance:
        return np.inf
    return float(np.sqrt(major / minor))


def _coordinate_weights(
    size: int,
    node_weights: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float64] | None:
    """Return finite non-negative coordinate weights with equal-weight fallback.

    Args:
        size: Expected number of weights.
        node_weights: Optional weights to validate.

    Returns:
        A one-dimensional weight array of length ``size``. Missing or all-zero
        weights become equal weights. Invalid length, non-finite values, or
        negative values return ``None`` so coordinate generation can fail
        cleanly.
    """
    if node_weights is None:
        return np.ones(size, dtype=np.float64)

    weights = np.asarray(node_weights, dtype=np.float64).reshape(-1)
    if len(weights) != size or not np.isfinite(weights).all() or (weights < 0.0).any():
        return None
    if float(weights.sum()) <= 0.0:
        return np.ones(size, dtype=np.float64)
    return weights


def _weighted_circular_longitude(
    longitudes: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> float:
    """Return a weighted longitude mean in degrees."""
    angles = np.deg2rad(longitudes)
    sin_mean = float(np.average(np.sin(angles), weights=weights))
    cos_mean = float(np.average(np.cos(angles), weights=weights))
    if np.isclose(sin_mean, 0.0, atol=_INERTIAL_TOLERANCE) and np.isclose(
        cos_mean,
        0.0,
        atol=_INERTIAL_TOLERANCE,
    ):
        return float(np.average(longitudes, weights=weights))
    return float(((np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 180.0) % 360.0) - 180.0)


def _node_weight(nodes: GridPartition, weights: np.ndarray) -> float:
    """Return the total weight assigned to nodes."""
    rows, cols = _node_indices(nodes)
    return float(weights[rows, cols].sum())


def _split_acceptance_allows(
    policy: SplitAcceptance,
    parent: GridPartition,
    children: list[GridPartition],
    weights: np.ndarray,
    target_regions: int | None = None,
) -> bool:
    """Return true when a split acceptance policy accepts a proposed split."""
    if target_regions is not None:
        accept_split = getattr(policy, "accept_split", None)
        if accept_split is not None:
            return bool(accept_split(parent, children, weights, target_regions))
    return bool(policy(parent, children, weights))


def _node_indices(nodes: GridPartition) -> tuple[list[int], list[int]]:
    """Split grid-index nodes into row and column index lists."""
    if not nodes:
        return [], []
    rows, cols = zip(*nodes)
    return list(rows), list(cols)


def _labels_for_bucket(weights: np.ndarray, class_mask: np.ndarray, bucket: float) -> np.ndarray:
    """Apply the rectangular weighted splitter and mask labels to one class.

    Args:
        weights: Class-local weight field, with cells outside the class normally
            set to zero.
        class_mask: Boolean mask selecting cells in the current class.
        bucket: Maximum rectangular bucket weight passed to
            :func:`bucket_value_split`.

    Returns:
        Integer label array with positive labels only inside ``class_mask``.
    """
    labels = np.zeros(weights.shape, dtype=np.int64)
    label = 1
    for ymin, ymax, xmin, xmax in bucket_value_split(weights, bucket):
        region_mask = class_mask[ymin:ymax, xmin:xmax]
        if not region_mask.any():
            continue
        label_slice = labels[ymin:ymax, xmin:xmax]
        label_slice[region_mask] = label
        label += 1
    return labels


def _count_positive_labels(labels: np.ndarray) -> int:
    """Return the number of positive labels in an integer label array."""
    return len(_positive_labels(labels))


def _positive_labels(labels: np.ndarray) -> np.ndarray:
    """Return sorted unique positive labels from an integer label array."""
    unique = np.unique(labels)
    return unique[unique > 0]


def _labels_dataarray(labels: np.ndarray, weights: xr.DataArray) -> xr.DataArray:
    """Wrap a label array in a ``basis`` DataArray using weight coordinates."""
    return xr.DataArray(labels, dims=weights.dims, coords=weights.coords, name="basis")


__all__ = [
    "AllSplitAcceptancePolicies",
    "AllocationMode",
    "AxisAlignedWeightedSplitStrategy",
    "AxisParallelSplitStep",
    "ComponentConsolidationPolicy",
    "ConnectedComponentPartitionStep",
    "ConnectedComponentSplitStrategy",
    "GreedyAxisParallelSplitStrategy",
    "InertialSplitStep",
    "LatLonGridGeometry",
    "MaxChildPCAEccentricity",
    "MinChildTargetWeightShare",
    "MinChildWeightShare",
    "NbasisAllocation",
    "PartitionStep",
    "SplitAcceptance",
    "SplitAcceptancePolicy",
    "SplitGeometry",
    "SplitStrategy",
    "TargetSplitAcceptancePolicy",
    "allocate_nbasis_by_class",
    "intersect_region_class_layers",
    "region_constrained_basis",
]
