"""Framework-independent Metropolis updates for dyadic product spaces.

The transition in this module changes only the discrete partition. Inner
root-and-contrast coordinates and always-active outer coefficients remain in a
fixed order and retain exactly the same values. Consequently this kernel has no
dimension matching transformation and no Jacobian; active-prior and inactive
pseudo-prior terms belong in the caller-supplied augmented log density.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from .proposals import MergeMove, SplitMove, apply_move, enumerate_merge_moves, enumerate_split_moves
from .state import PartitionState
from .tree import DyadicTree

PartitionMove: TypeAlias = SplitMove | MergeMove


@dataclass(frozen=True, slots=True, eq=False)
class ProductSpaceState:
    """Discrete partition and fixed-dimensional continuous coordinates.

    Array inputs are copied and made read-only so a transition cannot mutate a
    previous state through shared NumPy storage.

    Attributes:
        partition: Active dyadic frontier.
        inner_coordinates: Permanent root-and-contrast coordinate vector.
        outer_coefficients: Always-active fixed outer-region coefficients.
    """

    partition: PartitionState
    inner_coordinates: np.ndarray
    outer_coefficients: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))

    def __post_init__(self) -> None:
        """Validate, copy, and freeze both continuous coordinate vectors."""
        if not isinstance(self.partition, PartitionState):
            raise TypeError("partition must be a PartitionState.")
        object.__setattr__(
            self,
            "inner_coordinates",
            _frozen_vector(self.inner_coordinates, name="inner_coordinates"),
        )
        object.__setattr__(
            self,
            "outer_coefficients",
            _frozen_vector(self.outer_coefficients, name="outer_coefficients"),
        )

    def __eq__(self, other: object) -> bool:
        """Return value equality across the partition and both coordinate vectors."""
        if not isinstance(other, ProductSpaceState):
            return NotImplemented
        return (
            self.partition == other.partition
            and np.array_equal(self.inner_coordinates, other.inner_coordinates)
            and np.array_equal(self.outer_coefficients, other.outer_coefficients)
        )

    __hash__ = None  # type: ignore[assignment]


LogAugmentedDensity: TypeAlias = Callable[[ProductSpaceState], float]
PartitionLogDensity: TypeAlias = Callable[[PartitionState], float]


@dataclass(frozen=True, slots=True)
class PartitionNeighbor:
    """Unique split-or-merge destination and its forward probability."""

    partition: PartitionState
    move: PartitionMove
    log_q: float


@dataclass(frozen=True, slots=True)
class ProductSpaceTransition:
    """Complete diagnostic record for one partition Metropolis update.

    Attributes:
        state: Accepted candidate or unchanged current state.
        candidate: Proposed state before the accept/reject decision.
        move: Proposed split or merge, or ``None`` for an isolated tree.
        accepted: Whether the proposed partition was accepted.
        current_log_density: Augmented target at the source state.
        candidate_log_density: Augmented target at the proposed state.
        log_q_forward: Log proposal probability from source to candidate.
        log_q_reverse: Log proposal probability from candidate to source.
        log_acceptance_ratio: Unclipped log Metropolis-Hastings ratio.
    """

    state: ProductSpaceState
    candidate: ProductSpaceState
    move: PartitionMove | None
    accepted: bool
    current_log_density: float
    candidate_log_density: float
    log_q_forward: float
    log_q_reverse: float
    log_acceptance_ratio: float


@dataclass(frozen=True, slots=True)
class CollapsedPartitionTransition:
    """Diagnostic record for one marginal partition Metropolis update.

    Attributes:
        partition: Accepted candidate or unchanged current partition.
        candidate: Proposed partition before the accept/reject decision.
        move: Proposed split or merge, or ``None`` for an isolated tree.
        accepted: Whether the proposed partition was accepted.
        current_log_density: Marginal target at the source partition.
        candidate_log_density: Marginal target at the proposed partition.
        log_q_forward: Log proposal probability from source to candidate.
        log_q_reverse: Log proposal probability from candidate to source.
        log_acceptance_ratio: Unclipped log Metropolis-Hastings ratio.
    """

    partition: PartitionState
    candidate: PartitionState
    move: PartitionMove | None
    accepted: bool
    current_log_density: float
    candidate_log_density: float
    log_q_forward: float
    log_q_reverse: float
    log_acceptance_ratio: float


def enumerate_partition_neighbors(
    tree: DyadicTree,
    partition: PartitionState,
) -> tuple[PartitionNeighbor, ...]:
    """Enumerate unique one-split or one-merge partition neighbors.

    The proposal is uniform over unique destination frontiers. Split
    destinations precede merge destinations, with both groups following the
    stable ordering from :mod:`.proposals`.

    Args:
        tree: Tree defining legal split and merge moves.
        partition: Valid source partition.

    Returns:
        Neighbor records with common ``log_q = -log(number_of_neighbors)``.
        A partition on a one-grid-cell tree has no neighbors.
    """
    partition.validate(tree)
    unique: dict[frozenset[int], tuple[PartitionState, PartitionMove]] = {}
    moves: tuple[PartitionMove, ...] = (
        *enumerate_split_moves(tree, partition),
        *enumerate_merge_moves(tree, partition),
    )
    for move in moves:
        candidate = apply_move(tree, partition, move)
        unique.setdefault(candidate.active, (candidate, move))

    if not unique:
        return ()
    log_q = -math.log(len(unique))
    return tuple(
        PartitionNeighbor(partition=candidate, move=move, log_q=log_q) for candidate, move in unique.values()
    )


def partition_metropolis_step(
    tree: DyadicTree,
    current: ProductSpaceState,
    *,
    log_density: LogAugmentedDensity,
    rng: np.random.Generator,
) -> ProductSpaceTransition:
    """Apply one exact split-or-merge Metropolis-Hastings update.

    The continuous values are held fixed. The supplied target must evaluate the
    complete normalized product-space density, including the active prior, the
    inactive pseudo-prior, the partition prior, and any always-active outer
    block.

    Args:
        tree: Tree defining the current and proposed partitions.
        current: Current fixed-dimensional product-space state.
        log_density: Callable returning the normalized augmented log density.
        rng: Caller-owned NumPy random generator.

    Returns:
        Diagnostic transition record containing the accepted state.

    Raises:
        TypeError: If ``rng`` is not a NumPy generator or ``log_density`` is not
            callable.
        ValueError: If the current target density is non-finite, the candidate
            target is NaN or positive infinity, or the reverse proposal is
            missing.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    if not callable(log_density):
        raise TypeError("log_density must be callable.")
    current.partition.validate(tree)
    current_log_density = _checked_log_density(log_density(current), current=True)
    neighbors = enumerate_partition_neighbors(tree, current.partition)

    if not neighbors:
        return ProductSpaceTransition(
            state=current,
            candidate=current,
            move=None,
            accepted=False,
            current_log_density=current_log_density,
            candidate_log_density=current_log_density,
            log_q_forward=0.0,
            log_q_reverse=0.0,
            log_acceptance_ratio=-math.inf,
        )

    neighbor = neighbors[int(rng.integers(len(neighbors)))]
    candidate = ProductSpaceState(
        partition=neighbor.partition,
        inner_coordinates=current.inner_coordinates,
        outer_coefficients=current.outer_coefficients,
    )
    candidate_log_density = _checked_log_density(log_density(candidate), current=False)
    reverse_neighbors = enumerate_partition_neighbors(tree, candidate.partition)
    reverse = next(
        (item for item in reverse_neighbors if item.partition == current.partition),
        None,
    )
    if reverse is None:  # pragma: no cover - protects future proposal extensions.
        raise ValueError("Proposed partition has no reverse split-or-merge move.")

    log_acceptance_ratio = candidate_log_density - current_log_density + reverse.log_q - neighbor.log_q
    uniform = float(rng.random())
    log_uniform = -math.inf if uniform == 0.0 else math.log(uniform)
    accepted = bool(log_uniform < min(0.0, log_acceptance_ratio))
    return ProductSpaceTransition(
        state=candidate if accepted else current,
        candidate=candidate,
        move=neighbor.move,
        accepted=accepted,
        current_log_density=current_log_density,
        candidate_log_density=candidate_log_density,
        log_q_forward=neighbor.log_q,
        log_q_reverse=reverse.log_q,
        log_acceptance_ratio=log_acceptance_ratio,
    )


def collapsed_partition_metropolis_step(
    tree: DyadicTree,
    current: PartitionState,
    *,
    log_density: PartitionLogDensity,
    rng: np.random.Generator,
) -> CollapsedPartitionTransition:
    """Apply one split-or-merge MH update to a marginal partition target.

    This transition is useful when continuous coefficients can be integrated
    exactly.  It shares the local proposal and reverse-degree correction with
    :func:`partition_metropolis_step`, but its target is a density on
    :class:`PartitionState` rather than the augmented product-space state.

    Args:
        tree: Tree defining legal source and candidate partitions.
        current: Current valid partition.
        log_density: Callable returning the normalized or relatively normalized
            marginal log density for a partition.
        rng: Caller-owned NumPy random generator.

    Returns:
        Complete diagnostic transition with the accepted partition.

    Raises:
        TypeError: If inputs have the wrong object types.
        ValueError: If the current density is non-finite, the candidate density
            is NaN or positive infinity, or a reverse move is absent.
    """
    if not isinstance(current, PartitionState):
        raise TypeError("current must be a PartitionState.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    if not callable(log_density):
        raise TypeError("log_density must be callable.")
    current.validate(tree)
    current_log_density = _checked_log_density(log_density(current), current=True)
    neighbors = enumerate_partition_neighbors(tree, current)

    if not neighbors:
        return CollapsedPartitionTransition(
            partition=current,
            candidate=current,
            move=None,
            accepted=False,
            current_log_density=current_log_density,
            candidate_log_density=current_log_density,
            log_q_forward=0.0,
            log_q_reverse=0.0,
            log_acceptance_ratio=-math.inf,
        )

    neighbor = neighbors[int(rng.integers(len(neighbors)))]
    candidate_log_density = _checked_log_density(log_density(neighbor.partition), current=False)
    reverse_neighbors = enumerate_partition_neighbors(tree, neighbor.partition)
    reverse = next(
        (item for item in reverse_neighbors if item.partition == current),
        None,
    )
    if reverse is None:  # pragma: no cover - protects future proposal extensions.
        raise ValueError("Proposed partition has no reverse split-or-merge move.")

    log_acceptance_ratio = candidate_log_density - current_log_density + reverse.log_q - neighbor.log_q
    uniform = float(rng.random())
    log_uniform = -math.inf if uniform == 0.0 else math.log(uniform)
    accepted = bool(log_uniform < min(0.0, log_acceptance_ratio))
    return CollapsedPartitionTransition(
        partition=neighbor.partition if accepted else current,
        candidate=neighbor.partition,
        move=neighbor.move,
        accepted=accepted,
        current_log_density=current_log_density,
        candidate_log_density=candidate_log_density,
        log_q_forward=neighbor.log_q,
        log_q_reverse=reverse.log_q,
        log_acceptance_ratio=log_acceptance_ratio,
    )


def _frozen_vector(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    """Return a copied, finite, read-only floating-point vector."""
    source = np.asarray(values)
    if np.iscomplexobj(source):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(source, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = array.copy()
    result.setflags(write=False)
    return result


def _checked_log_density(value: float, *, current: bool) -> float:
    """Validate one scalar log density while allowing rejected ``-inf`` proposals."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("log_density must return a real scalar.") from error
    if math.isnan(result) or result == math.inf or (current and result == -math.inf):
        if current:
            raise ValueError("current log density must be finite.")
        raise ValueError("candidate log density must be finite or negative infinity.")
    return result


__all__ = [
    "CollapsedPartitionTransition",
    "LogAugmentedDensity",
    "PartitionMove",
    "PartitionLogDensity",
    "PartitionNeighbor",
    "ProductSpaceState",
    "ProductSpaceTransition",
    "collapsed_partition_metropolis_step",
    "enumerate_partition_neighbors",
    "partition_metropolis_step",
]
