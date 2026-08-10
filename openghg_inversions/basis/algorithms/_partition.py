"""Grid partitioning contracts, geometries, policies, and split strategies.

The public :func:`greedy_partitioning` function operates on node partitions, a
weight array, a requested partition count, and caller-supplied split behavior.
It has no knowledge of region classes, global basis labels, or xarray output
composition. :class:`GreedySplitStrategy` adapts that generic machinery to one
Boolean class mask.

A :class:`PartitionStep` may mark a partition as unsplittable by returning a
single child (normally the unchanged parent). Multi-child proposals are
validated before acceptance: children must be non-empty, disjoint subsets that
exactly cover the parent. Optional split-acceptance policies may then freeze a
valid proposed split. Geometry is separate from contribution weights and is
used only for coordinate-space split decisions. Refinement is deterministic
for deterministic split behavior: partitions are prioritized by descending
total weight, then descending node count, and exact ties retain insertion
order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy import ndimage

#: One grid cell represented by its ``(row, column)`` integer index.
GridNode: TypeAlias = tuple[int, int]
#: Mutable list of grid-cell indices forming one partition.
GridPartition: TypeAlias = list[GridNode]

_INERTIAL_TOLERANCE = 1.0e-12


class PartitionStep(Protocol):
    """Strategy protocol for splitting one partition into child partitions."""

    def __call__(self, nodes: GridPartition, weights: np.ndarray) -> list[GridPartition]:
        """Split one grid-node partition.

        Args:
            nodes: Grid nodes in the partition being split.
            weights: Non-negative weight field aligned to the source grid.

        Returns:
            Child grid-node partitions. Returning one child leaves the
            partition effectively unsplit. Multiple children must be
            non-empty, pairwise disjoint subsets that exactly cover ``nodes``.
        """
        ...


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
            children: Valid child partitions proposed by a
                :class:`PartitionStep`.
            weights: Non-negative weight field aligned to the source grid.

        Returns:
            True if greedy orchestration should replace ``parent`` with
            ``children``. False freezes ``parent`` as a completed partition.
        """
        ...


class TargetSplitAcceptancePolicy(SplitAcceptancePolicy, Protocol):
    """Policy protocol for accepting splits using the target count."""

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
            children: Valid child partitions proposed by a
                :class:`PartitionStep`.
            weights: Non-negative weight field aligned to the source grid.
            target_regions: Requested upper target count when available.

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
            children: Valid child partitions proposed by a
                :class:`PartitionStep`.
            weights: Non-negative weight field aligned to the source grid.
            target_regions: Requested upper target count.

        Returns:
            True if greedy orchestration should replace ``parent`` with
            ``children``. False freezes ``parent`` as a completed partition.
        """
        ...


#: Split-acceptance callable, optionally aware of the refinement target.
SplitAcceptance: TypeAlias = SplitAcceptancePolicy | TargetSplitAcceptancePolicy


@dataclass(order=True)
class _PrioritizedPartition:
    """Priority queue item for selecting the next partition to split."""

    priority: tuple[float, int, int]
    nodes: GridPartition = field(compare=False)


class _PartitionPriorityQueue:
    """Priority queue that pops the highest-weight partition first."""

    def __init__(self, weights: np.ndarray) -> None:
        """Create a queue ranked by partition weight, size, and insertion."""
        self._queue: PriorityQueue[_PrioritizedPartition] = PriorityQueue()
        self._weights = weights
        self._counter = 0

    def __bool__(self) -> bool:
        """Return true when the queue contains at least one partition."""
        return not self._queue.empty()

    def push(self, nodes: GridPartition) -> None:
        """Insert a non-empty partition into the priority queue."""
        if not nodes:
            return
        priority = (-_node_weight(nodes, self._weights), -len(nodes), self._counter)
        self._counter += 1
        self._queue.put_nowait(_PrioritizedPartition(priority, nodes))

    def pop(self) -> GridPartition:
        """Remove and return the highest-priority partition."""
        return self._queue.get_nowait().nodes


def greedy_partitioning(
    init_partition: list[GridPartition],
    target_regions: int,
    weights: np.ndarray,
    *,
    split_step: PartitionStep,
    split_acceptance: SplitAcceptance | None = None,
) -> list[GridPartition]:
    """Apply a partition step greedily until a target count is reached.

    Args:
        init_partition: Initial partitions to refine.
        target_regions: Refinement ceiling for the number of output partitions.
            Must be at least one. The function never coarsens an initial
            partition whose count already exceeds this value.
        weights: Two-dimensional non-negative weight field used for partition
            ranking and passed unchanged to ``split_step`` and
            ``split_acceptance``. Every node must be a valid ``(row, column)``
            index into this array.
        split_step: Callable that proposes child partitions for one selected
            parent. Fewer than two returned children mark the parent as
            unsplittable.
        split_acceptance: Optional policy applied to a valid multi-child
            proposal. Rejected splits freeze the selected parent.

    Returns:
        Output partitions. Empty initial partitions are ignored. The result
        may contain fewer than
        ``target_regions`` entries when no active partition can be split, an
        accepted proposal would overshoot the target, or acceptance policies
        reject the remaining candidates.

    Raises:
        ValueError: If ``target_regions`` is less than one, or a multi-child
            proposal contains an empty child, duplicate or overlapping nodes,
            nodes outside its parent, or does not exactly cover its parent.
    """
    if target_regions < 1:
        raise ValueError("target_regions must be at least 1.")

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
        child_partitions = split_step(nodes, weights)

        if len(child_partitions) < 2:
            done.append(nodes)
            continue
        _validate_child_partitions(nodes, child_partitions)
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
        for child in child_partitions:
            current_regions += 1
            if len(child) > 1:
                active.push(child)
            else:
                done.append(child)

    while active:
        done.append(active.pop())

    return done


def _validate_child_partitions(parent: GridPartition, children: list[GridPartition]) -> None:
    """Validate that multiple children form an exact partition of a parent.

    Args:
        parent: Parent partition that the split step attempted to divide.
        children: Proposed non-empty child partitions.

    Raises:
        ValueError: If a child is empty, contains duplicate or out-of-parent
            nodes, overlaps another child, or the children do not exactly cover
            the parent.
    """
    if any(not child for child in children):
        raise ValueError("PartitionStep multi-child proposals must not contain empty children.")

    parent_nodes = set(parent)
    seen_nodes: set[GridNode] = set()
    for child in children:
        child_nodes = set(child)
        if len(child_nodes) != len(child):
            raise ValueError("PartitionStep child partitions must not contain duplicate nodes.")
        if not child_nodes <= parent_nodes:
            raise ValueError("PartitionStep child partitions may contain only nodes from their parent.")
        if seen_nodes & child_nodes:
            raise ValueError("PartitionStep child partitions must be pairwise disjoint.")
        seen_nodes.update(child_nodes)

    if seen_nodes != parent_nodes:
        raise ValueError("PartitionStep child partitions must exactly cover their parent.")


def _node_weight(nodes: GridPartition, weights: np.ndarray) -> float:
    """Return total weight for valid ``(row, column)`` indices in a 2-D array."""
    rows, columns = zip(*nodes)
    return float(weights[list(rows), list(columns)].sum())


def _split_acceptance_allows(
    policy: SplitAcceptance,
    parent: GridPartition,
    children: list[GridPartition],
    weights: np.ndarray,
    target_regions: int | None = None,
) -> bool:
    """Dispatch a valid proposed split to an acceptance policy.

    Args:
        policy: Acceptance policy to evaluate.
        parent: Parent partition selected for refinement.
        children: Valid proposed child partitions.
        weights: Two-dimensional weights passed to the policy.
        target_regions: Optional refinement ceiling. When supplied and the
            policy defines ``accept_split``, that target-aware method is used;
            otherwise the three-argument callable is used.

    Returns:
        Whether the policy accepts the proposed split.
    """
    if target_regions is not None:
        accept_split = getattr(policy, "accept_split", None)
        if accept_split is not None:
            return bool(accept_split(parent, children, weights, target_regions))
    return bool(policy(parent, children, weights))


@dataclass(frozen=True)
class _CoordinatePCA:
    """Principal-component fit for partition coordinates."""

    centroid: npt.NDArray[np.float64]
    centered: npt.NDArray[np.float64]
    eigenvalues: npt.NDArray[np.float64]
    eigenvectors: npt.NDArray[np.float64]


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

    By default, every child is subject to the eccentricity limit. Setting
    ``min_child_target_weight_share`` exempts children whose weight is below
    that share of one class/source-local equal-weight target region,
    ``weights.sum() / target_regions``. This target-aware exception is used
    only through :meth:`accept_split`; the conservative three-argument call
    remains strict.

    The exception affects split acceptance only. It does not reconnect, freeze,
    prune, or marginalize an exempt child after the split is accepted.
    """

    max_child_pca_eccentricity: float
    geometry: SplitGeometry | None = None
    tolerance: float = _INERTIAL_TOLERANCE
    min_child_target_weight_share: float = 0.0

    def __post_init__(self) -> None:
        """Validate the eccentricity, tolerance, and materiality thresholds."""
        if self.max_child_pca_eccentricity < 1.0 or not np.isfinite(self.max_child_pca_eccentricity):
            raise ValueError("max_child_pca_eccentricity must be at least 1 and finite.")
        if self.tolerance < 0.0 or not np.isfinite(self.tolerance):
            raise ValueError("tolerance must be non-negative and finite.")
        if not 0.0 <= self.min_child_target_weight_share <= 1.0:
            raise ValueError("min_child_target_weight_share must be between 0 and 1.")

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

    def accept_split(
        self,
        parent: GridPartition,
        children: list[GridPartition],
        weights: np.ndarray,
        target_regions: int,
    ) -> bool:
        """Return true when every materially weighted child meets the limit.

        ``weights`` is the class/source-local field passed to greedy
        partitioning, so ``weights.sum() / target_regions`` is the equal-weight
        target region weight. Children strictly below
        ``min_child_target_weight_share`` times that reference weight are
        exempt from the eccentricity veto.

        If the total weight is zero, cell counts provide the same direct-call
        fallback used by :class:`MinChildTargetWeightShare`; the default greedy
        strategy already converts all-zero classes to an area surrogate before
        policies are evaluated. If every child is below the materiality
        threshold, the strict guard is retained rather than accepting
        vacuously.
        """
        if target_regions < 1:
            raise ValueError("target_regions must be at least 1.")
        if self.min_child_target_weight_share == 0.0:
            return self(parent, children, weights)

        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            total_weight = float(weights.size)
            child_weights = [float(len(child)) for child in children]
        else:
            child_weights = [_node_weight(child, weights) for child in children]

        if total_weight <= 0.0:
            return False
        minimum_material_weight = self.min_child_target_weight_share * total_weight / target_regions
        material_children = [
            child
            for child, child_weight in zip(children, child_weights, strict=True)
            if child_weight >= minimum_material_weight
        ]
        if not material_children:
            material_children = children
        return self(parent, material_children, weights)


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
    :class:`GreedySplitStrategy`.

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
    labels, count = cast(
        tuple[np.ndarray, int],
        ndimage.label(
            mask,
            structure=ndimage.generate_binary_structure(2, connectivity),
        ),
    )
    return [list(zip(*np.where(labels == component), strict=True)) for component in range(1, int(count) + 1)]


def _decompose_connected_children(
    children: list[GridPartition],
    shape: tuple[int, int],
    *,
    connectivity: int,
) -> list[GridPartition]:
    """Return the connected-component decomposition of child partitions."""
    return [
        component
        for child in children
        for component in _connected_node_components(
            child,
            shape,
            connectivity=connectivity,
        )
    ]


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
        return _decompose_connected_children(
            children,
            weights.shape,
            connectivity=self.connectivity,
        )


def _component_adjacencies(
    left_components: list[GridPartition],
    right_components: list[GridPartition],
    shape: tuple[int, int],
    *,
    connectivity: int,
) -> tuple[list[set[int]], list[set[int]]]:
    """Return cross-side adjacency sets for two component collections."""
    right_labels = np.zeros(shape, dtype=np.int64)
    for component_index, component in enumerate(right_components, start=1):
        rows, columns = _node_indices(component)
        right_labels[rows, columns] = component_index

    if connectivity == 1:
        offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
    else:
        offsets = tuple(
            (row_offset, column_offset)
            for row_offset in (-1, 0, 1)
            for column_offset in (-1, 0, 1)
            if row_offset != 0 or column_offset != 0
        )

    left_adjacencies = [set() for _component in left_components]
    right_adjacencies = [set() for _component in right_components]
    nrows, ncolumns = shape
    for left_index, component in enumerate(left_components):
        for row, column in component:
            for row_offset, column_offset in offsets:
                adjacent_row = row + row_offset
                adjacent_column = column + column_offset
                if not (0 <= adjacent_row < nrows and 0 <= adjacent_column < ncolumns):
                    continue
                right_index = int(right_labels[adjacent_row, adjacent_column]) - 1
                if right_index >= 0:
                    left_adjacencies[left_index].add(right_index)
                    right_adjacencies[right_index].add(left_index)

    return left_adjacencies, right_adjacencies


def _repair_binary_connected_children(
    parent: GridPartition,
    children: list[GridPartition],
    weights: np.ndarray,
    *,
    connectivity: int,
) -> list[GridPartition] | None:
    """Return a deterministic connected binary repair when one exists."""
    if len(children) != 2 or not all(children):
        return None

    parent_nodes = set(parent)
    left_nodes = set(children[0])
    right_nodes = set(children[1])
    if (
        len(parent_nodes) != len(parent)
        or len(left_nodes) != len(children[0])
        or len(right_nodes) != len(children[1])
        or left_nodes & right_nodes
        or left_nodes | right_nodes != parent_nodes
    ):
        return None

    left_components = _connected_node_components(
        children[0],
        weights.shape,
        connectivity=connectivity,
    )
    right_components = _connected_node_components(
        children[1],
        weights.shape,
        connectivity=connectivity,
    )
    if not left_components or not right_components:
        return None

    left_adjacencies, right_adjacencies = _component_adjacencies(
        left_components,
        right_components,
        weights.shape,
        connectivity=connectivity,
    )
    left_weights = [_node_weight(component, weights) for component in left_components]
    right_weights = [_node_weight(component, weights) for component in right_components]
    total_left_weight = sum(left_weights)
    total_right_weight = sum(right_weights)

    best_key: tuple[float, float, int, int] | None = None
    best_primary_indices: tuple[int, int] | None = None
    for left_index, adjacent_right in enumerate(left_adjacencies):
        if len(right_components) - len(adjacent_right) > 1:
            continue
        for right_index, adjacent_left in enumerate(right_adjacencies):
            if len(left_components) - len(adjacent_left) > 1:
                continue
            if len(adjacent_right) - int(right_index in adjacent_right) != len(right_components) - 1:
                continue
            if len(adjacent_left) - int(left_index in adjacent_left) != len(left_components) - 1:
                continue

            moved_weight = (
                total_left_weight - left_weights[left_index] + total_right_weight - right_weights[right_index]
            )
            repaired_left_weight = left_weights[left_index] + total_right_weight - right_weights[right_index]
            repaired_right_weight = right_weights[right_index] + total_left_weight - left_weights[left_index]
            key = (
                moved_weight,
                abs(repaired_left_weight - repaired_right_weight),
                left_index,
                right_index,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_primary_indices = (left_index, right_index)

    if best_primary_indices is None:
        return None

    left_primary, right_primary = best_primary_indices
    repaired_left = sorted(
        left_components[left_primary]
        + [
            node
            for component_index, component in enumerate(right_components)
            if component_index != right_primary
            for node in component
        ]
    )
    repaired_right = sorted(
        right_components[right_primary]
        + [
            node
            for component_index, component in enumerate(left_components)
            if component_index != left_primary
            for node in component
        ]
    )
    return [repaired_left, repaired_right]


@dataclass(frozen=True)
class ConnectedBinaryPartitionStep:
    """Repair a provisional binary split into two connected child partitions.

    This opt-in wrapper preserves binary arity when disconnected cut fragments
    can be reassigned safely. It labels the connected components on both sides,
    retains one primary component on each original side, and moves every
    secondary component to the opposite side. Candidates are valid only when
    both resulting children are connected.

    Valid candidates are selected deterministically by minimum moved fitting
    weight, then minimum absolute child-weight imbalance, then row-major
    component order. If the wrapped step does not return a valid binary
    partition, or no connected binary reassignment exists, the result falls
    back to the same multi-child component decomposition as
    :class:`ConnectedComponentPartitionStep`.

    Attributes:
        split_step: Partition step whose provisional binary children should be
            repaired.
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
        """Return two repaired connected children or the component fallback."""
        children = self.split_step(nodes, weights)
        repaired = _repair_binary_connected_children(
            nodes,
            children,
            weights,
            connectivity=self.connectivity,
        )
        if repaired is not None:
            return repaired
        return _decompose_connected_children(
            children,
            weights.shape,
            connectivity=self.connectivity,
        )


@dataclass(frozen=True)
class GreedySplitStrategy:
    """Adapt generic greedy partitioning to one class mask.

    The strategy converts the selected cells to grid nodes, applies the
    explicitly supplied partition step through :func:`greedy_partitioning`,
    and converts the result to dense positive class-local labels. It contains
    no implicit choice of split geometry or algorithm.

    Attributes:
        split_step: Partition step used to propose children for the selected
            highest-priority partition.
        split_acceptance: Optional policy that may reject a valid proposed
            split and freeze its parent.
    """

    split_step: PartitionStep
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
            Integer label array with the same shape as ``weights``, positive
            class-local labels inside ``class_mask``, and zero outside it. The
            result may contain fewer labels than ``target_regions``.

        Raises:
            ValueError: If ``target_regions`` is less than one or the partition
                step proposes malformed child partitions.
        """
        if target_regions < 1:
            raise ValueError("target_regions must be at least 1.")
        if not class_mask.any():
            return np.zeros(weights.shape, dtype=np.int64)

        class_weights = np.where(class_mask, weights, 0.0)
        if float(class_weights.sum()) == 0.0:
            class_weights = class_mask.astype(np.float64)

        nodes = _node_list_from_mask(class_mask)
        partition = greedy_partitioning(
            [nodes],
            target_regions,
            class_weights,
            split_step=self.split_step,
            split_acceptance=self.split_acceptance,
        )
        return _labels_from_node_partition(partition, weights.shape)


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


def _node_indices(nodes: GridPartition) -> tuple[list[int], list[int]]:
    """Split grid-index nodes into row and column index lists."""
    if not nodes:
        return [], []
    rows, cols = zip(*nodes)
    return list(rows), list(cols)


__all__ = [
    "AllSplitAcceptancePolicies",
    "AxisParallelSplitStep",
    "ConnectedBinaryPartitionStep",
    "ConnectedComponentPartitionStep",
    "GreedySplitStrategy",
    "GridNode",
    "GridPartition",
    "InertialSplitStep",
    "LatLonGridGeometry",
    "MaxChildPCAEccentricity",
    "MinChildTargetWeightShare",
    "MinChildWeightShare",
    "PartitionStep",
    "SplitAcceptance",
    "SplitAcceptancePolicy",
    "SplitGeometry",
    "TargetSplitAcceptancePolicy",
    "greedy_partitioning",
]
