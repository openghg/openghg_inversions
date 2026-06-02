"""Mask- and region-constrained basis generation helpers.

The helpers in this module are intentionally pure: callers pass an already
computed 2D weight field and an already loaded 2D class mask. File loading,
footprint/flux reduction, and domain-specific country lookup stay at the caller
boundary.

The default implementation uses the prototype-inspired greedy axis-parallel
split shape: each class is split independently with repeated bisection, then
labels are offset globally. The class orchestration accepts a split strategy
protocol so an inertial partition, quadtree step, recursive weighted split, or
other tile generator can be substituted without changing mask alignment or
label-offset behavior.
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

from ._weighted import bucket_value_split

# Allocation modes control how an integer ``nbasis`` is distributed across
# region classes before class-local splitting is applied.
AllocationMode: TypeAlias = Literal["weight", "area"]
NbasisAllocation: TypeAlias = int | Mapping[Hashable, int]
GridNode: TypeAlias = tuple[int, int]
GridPartition: TypeAlias = list[GridNode]


class SplitStrategy(Protocol):
    """Strategy protocol for class-local basis splitting."""

    def __call__(
        self,
        weights: np.ndarray,
        class_mask: np.ndarray,
        target_regions: int,
    ) -> np.ndarray:
        """Return positive local labels for cells in ``class_mask``."""
        ...


class PartitionStep(Protocol):
    """Strategy protocol for splitting one partition into child partitions."""

    def __call__(self, nodes: GridPartition, weights: np.ndarray) -> list[GridPartition]:
        """Return child partitions for ``nodes``."""
        ...


@dataclass(frozen=True)
class AxisParallelSplitStep:
    """Split one partition along an axis-parallel line.

    This is a cleaned-up version of the prototype's axis-parallel split step.
    Greedy orchestration is handled separately by
    :class:`GreedyAxisParallelSplitStrategy`.
    """

    balanced: bool = True
    clean_splits: bool = False

    def __call__(self, nodes: GridPartition, weights: np.ndarray) -> list[GridPartition]:
        """Return child partitions from an axis-parallel split."""
        left, right = _axis_parallel_split_nodes(
            nodes,
            weights,
            balanced=self.balanced,
            clean_splits=self.clean_splits,
        )
        if not left or not right:
            return [nodes]
        return [left, right]


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

    def __call__(
        self,
        weights: np.ndarray,
        class_mask: np.ndarray,
        target_regions: int,
    ) -> np.ndarray:
        """Return class-local labels using greedy axis-parallel bisection."""
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
        )
        return _labels_from_node_partition(partition, weights.shape)


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
        """Return class-local labels using recursive weighted bucket splits."""
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
        class_mask = class_values == class_value
        local_labels = strategy(weight_values, class_mask, target_regions)
        for local_label in _positive_labels(local_labels):
            labels[(local_labels == local_label) & class_mask] = next_label
            next_label += 1

    return _labels_dataarray(labels, weights)


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
        class_value: int(np.count_nonzero(class_values == class_value)) for class_value in mapped_classes
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
        if pd.isna(value) or value in unmapped:
            continue
        classes.append(cast(Hashable, value))
    return classes


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
            class_value: float(np.count_nonzero(class_values == class_value))
            for class_value in mapped_classes
        }
    if allocation != "weight":
        raise ValueError("allocation must be 'weight' or 'area'.")

    scores = {
        class_value: float(weights[class_values == class_value].sum()) for class_value in mapped_classes
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

    Returns:
        List of output partitions. The result may contain fewer than
        ``target_regions`` entries when no active partition can be split further.
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

    Returns:
        Two node lists. Either side may be empty for degenerate input or a
        degenerate clean split.
    """
    if len(nodes) < 2:
        return nodes, []

    axis = _long_axis_weighted(nodes, weights) if balanced else _long_axis(nodes)
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


def _long_axis(nodes: GridPartition) -> int:
    """Return the grid axis with the largest geometric spread."""
    if not nodes:
        return 0
    coords = np.asarray(nodes)
    return int(np.argmax(coords.max(axis=0) - coords.min(axis=0)))


def _long_axis_weighted(nodes: GridPartition, weights: np.ndarray) -> int:
    """Return the grid axis with the largest weighted absolute spread."""
    if not nodes:
        return 0
    rows, cols = _node_indices(nodes)
    node_weights = weights[rows, cols].astype(np.float64)
    total_weight = float(node_weights.sum())
    if total_weight == 0.0:
        return _long_axis(nodes)

    coords = np.asarray(nodes, dtype=np.float64)
    weight_column = node_weights.reshape(-1, 1)
    centroid = (coords * weight_column).sum(axis=0) / total_weight
    spread = (weight_column * np.abs(coords - centroid)).sum(axis=0) / total_weight
    return int(np.argmax(spread))


def _node_weight(nodes: GridPartition, weights: np.ndarray) -> float:
    """Return the total weight assigned to nodes."""
    rows, cols = _node_indices(nodes)
    return float(weights[rows, cols].sum())


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
    "AllocationMode",
    "AxisParallelSplitStep",
    "AxisAlignedWeightedSplitStrategy",
    "GreedyAxisParallelSplitStrategy",
    "NbasisAllocation",
    "PartitionStep",
    "SplitStrategy",
    "allocate_nbasis_by_class",
    "region_constrained_basis",
]
