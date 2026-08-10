"""Generic greedy refinement of weighted grid-node partitions.

This module defines the low-level contracts and orchestration used by
constrained basis algorithms. The public :func:`greedy_partitioning` function
operates only on node partitions, a weight array, a requested partition count,
and caller-supplied split behavior. It deliberately has no knowledge of class
masks, xarray objects, global basis labels, or domain-specific geometry.

A :class:`PartitionStep` may mark a partition as unsplittable by returning a
single child (normally the unchanged parent). Multi-child proposals are
validated before acceptance: children must be non-empty, disjoint subsets that
exactly cover the parent. Optional split-acceptance policies may then freeze a
valid proposed split. Refinement is deterministic for a deterministic split
step: partitions are prioritized by descending total weight, then descending
node count, and exact ties retain insertion order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Protocol, TypeAlias

import numpy as np

#: One grid cell represented by its ``(row, column)`` integer index.
GridNode: TypeAlias = tuple[int, int]
#: Mutable list of grid-cell indices forming one partition.
GridPartition: TypeAlias = list[GridNode]


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


__all__ = [
    "GridNode",
    "GridPartition",
    "PartitionStep",
    "SplitAcceptance",
    "SplitAcceptancePolicy",
    "TargetSplitAcceptancePolicy",
    "greedy_partitioning",
]
