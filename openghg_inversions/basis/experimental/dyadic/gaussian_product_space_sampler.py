"""Blocked Gaussian sampler for dyadic product-space partition inference.

The sampler in this module does not enumerate the set of valid partitions.
Each structural update proposes one local split or merge while holding the
permanent root-and-contrast coordinates fixed.  It then refreshes the active
inner coefficients, fixed outer coefficients, and inactive pseudo-prior
coordinates from their exact Gaussian conditional laws.

This is a Metropolis-within-Gibbs reference implementation.  It is intended to
establish a correct, scalable baseline before replacing the exact Gaussian
continuous update with NUTS or another non-Gaussian kernel.  ``warmup`` is a
discard period only; no proposal or mass-matrix adaptation occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np

from .gaussian_product_space import GaussianProductSpaceTarget
from .product_space import ProductSpaceState, partition_metropolis_step
from .state import PartitionState


@dataclass(frozen=True, slots=True, eq=False)
class GaussianProductSpaceTrace:
    """Retained states and partition-transition diagnostics from one chain.

    Attributes:
        partitions: Active partition for each retained draw.
        inner_coordinates: Permanent inner root-and-contrast coordinates with
            shape ``(draw, inner_coordinate)``.
        outer_coefficients: Always-active outer coefficients with shape
            ``(draw, outer_region)``.
        partition_accepted: Acceptance indicators for the local structural
            updates immediately preceding each retained draw.  Its second
            dimension is ``partition_updates_per_draw``.
        partition_log_acceptance_ratio: Unclipped Metropolis-Hastings log ratios
            corresponding to :attr:`partition_accepted`.
        warmup_acceptance_rate: Fraction of structural proposals accepted in
            the discarded warmup period, or ``None`` when ``warmup`` was zero.
        thinning: Number of complete transition cycles between retained draws.
    """

    partitions: tuple[PartitionState, ...]
    inner_coordinates: np.ndarray
    outer_coefficients: np.ndarray
    partition_accepted: np.ndarray
    partition_log_acceptance_ratio: np.ndarray
    warmup_acceptance_rate: float | None
    thinning: int

    def __post_init__(self) -> None:
        """Validate dimensions and make all diagnostic arrays read-only."""
        draw_count = len(self.partitions)
        if draw_count < 1:
            raise ValueError("partitions must contain at least one retained draw.")
        if any(not isinstance(partition, PartitionState) for partition in self.partitions):
            raise TypeError("partitions must contain PartitionState values.")

        inner = _frozen_matrix(self.inner_coordinates, name="inner_coordinates")
        outer = _frozen_matrix(self.outer_coefficients, name="outer_coefficients")
        accepted = _frozen_matrix(
            self.partition_accepted,
            name="partition_accepted",
            dtype=np.bool_,
        )
        log_ratio = _frozen_matrix(
            self.partition_log_acceptance_ratio,
            name="partition_log_acceptance_ratio",
        )
        if inner.shape[0] != draw_count or outer.shape[0] != draw_count:
            raise ValueError("coordinate arrays must have one row per retained partition.")
        if accepted.shape[0] != draw_count or log_ratio.shape != accepted.shape:
            raise ValueError("partition diagnostics must align with retained partitions.")
        if accepted.shape[1] < 1:
            raise ValueError("partition diagnostics must contain at least one update per draw.")
        if not np.all(np.isfinite(log_ratio) | np.isneginf(log_ratio)):
            raise ValueError("partition_log_acceptance_ratio cannot contain NaN or positive infinity.")
        if self.warmup_acceptance_rate is not None and not 0.0 <= self.warmup_acceptance_rate <= 1.0:
            raise ValueError("warmup_acceptance_rate must lie between zero and one.")

        thinning = _positive_integer(self.thinning, name="thinning")
        object.__setattr__(self, "inner_coordinates", inner)
        object.__setattr__(self, "outer_coefficients", outer)
        object.__setattr__(self, "partition_accepted", accepted)
        object.__setattr__(self, "partition_log_acceptance_ratio", log_ratio)
        object.__setattr__(self, "thinning", thinning)

    @property
    def draw_count(self) -> int:
        """Return the number of retained product-space states."""
        return len(self.partitions)

    @property
    def region_counts(self) -> np.ndarray:
        """Return the active inner-region count for every retained draw."""
        result = np.fromiter(
            (len(partition.active) for partition in self.partitions),
            dtype=np.int64,
            count=self.draw_count,
        )
        result.setflags(write=False)
        return result

    @property
    def partition_acceptance_rate(self) -> float:
        """Return the retained structural-proposal acceptance fraction."""
        return float(np.mean(self.partition_accepted))

    def state(self, draw: int) -> ProductSpaceState:
        """Reconstruct one retained product-space state.

        Args:
            draw: Positional draw index, including standard negative indices.

        Returns:
            Immutable partition and continuous coordinates for that draw.

        Raises:
            TypeError: If ``draw`` is not an integer.
            IndexError: If ``draw`` is outside the retained trace.
        """
        draw_index = index(draw)
        return ProductSpaceState(
            partition=self.partitions[draw_index],
            inner_coordinates=self.inner_coordinates[draw_index],
            outer_coefficients=self.outer_coefficients[draw_index],
        )


def sample_gaussian_product_space(
    target: GaussianProductSpaceTarget,
    initial_partition: PartitionState,
    *,
    draws: int,
    warmup: int = 0,
    thinning: int = 1,
    partition_updates_per_draw: int = 1,
    rng: np.random.Generator,
) -> GaussianProductSpaceTrace:
    """Sample latent dyadic partitions and their Gaussian coefficients.

    One transition cycle applies ``partition_updates_per_draw`` local
    split-or-merge Metropolis updates.  Every structural proposal holds the
    permanent continuous coordinates fixed, after which all continuous values
    are redrawn from their exact law conditional on the accepted partition.
    This composition preserves the augmented product-space target without
    constructing a catalogue of partitions.

    Args:
        target: Gaussian augmented product-space target.
        initial_partition: Valid starting frontier.  A good deterministic
            partition can be supplied when starting from the root mixes poorly.
        draws: Positive number of retained states.
        warmup: Number of complete transition cycles to discard.  No adaptive
            tuning is performed during these cycles.
        thinning: Positive number of post-warmup cycles per retained state.
        partition_updates_per_draw: Positive number of structural updates and
            conditional Gaussian refreshes in each cycle.
        rng: Caller-owned NumPy random generator.

    Returns:
        Retained chain with structural transition diagnostics.

    Raises:
        TypeError: If inputs have the wrong object or scalar types.
        ValueError: If an integer control is outside its documented range or
            ``initial_partition`` is invalid for the target tree.
    """
    if not isinstance(target, GaussianProductSpaceTarget):
        raise TypeError("target must be a GaussianProductSpaceTarget.")
    if not isinstance(initial_partition, PartitionState):
        raise TypeError("initial_partition must be a PartitionState.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    draw_count = _positive_integer(draws, name="draws")
    warmup_count = _non_negative_integer(warmup, name="warmup")
    thin = _positive_integer(thinning, name="thinning")
    update_count = _positive_integer(
        partition_updates_per_draw,
        name="partition_updates_per_draw",
    )
    initial_partition.validate(target.tree)

    state = target.draw_conditional_state(initial_partition, rng)
    retained_partitions: list[PartitionState] = []
    retained_inner: list[np.ndarray] = []
    retained_outer: list[np.ndarray] = []
    retained_accepted: list[list[bool]] = []
    retained_log_ratios: list[list[float]] = []
    warmup_acceptances = 0
    warmup_proposals = warmup_count * update_count
    total_cycles = warmup_count + draw_count * thin

    for cycle in range(total_cycles):
        cycle_accepted: list[bool] = []
        cycle_log_ratios: list[float] = []
        for _ in range(update_count):
            transition = partition_metropolis_step(
                target.tree,
                state,
                log_density=target.log_density,
                rng=rng,
            )
            cycle_accepted.append(transition.accepted)
            cycle_log_ratios.append(transition.log_acceptance_ratio)
            state = target.draw_conditional_state(transition.state.partition, rng)

        if cycle < warmup_count:
            warmup_acceptances += sum(cycle_accepted)
            continue
        if (cycle - warmup_count) % thin != thin - 1:
            continue

        retained_partitions.append(state.partition)
        retained_inner.append(state.inner_coordinates)
        retained_outer.append(state.outer_coefficients)
        retained_accepted.append(cycle_accepted)
        retained_log_ratios.append(cycle_log_ratios)

    warmup_rate = None if warmup_proposals == 0 else warmup_acceptances / warmup_proposals
    return GaussianProductSpaceTrace(
        partitions=tuple(retained_partitions),
        inner_coordinates=np.asarray(retained_inner, dtype=float),
        outer_coefficients=np.asarray(retained_outer, dtype=float),
        partition_accepted=np.asarray(retained_accepted, dtype=bool),
        partition_log_acceptance_ratio=np.asarray(retained_log_ratios, dtype=float),
        warmup_acceptance_rate=warmup_rate,
        thinning=thin,
    )


def _positive_integer(value: int, *, name: str) -> int:
    """Return a strictly positive integer control value."""
    result = _integer(value, name=name)
    if result < 1:
        raise ValueError(f"{name} must be at least 1.")
    return result


def _non_negative_integer(value: int, *, name: str) -> int:
    """Return a non-negative integer control value."""
    result = _integer(value, name=name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _integer(value: int, *, name: str) -> int:
    """Return an integer control while rejecting booleans and coercions."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    try:
        return index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer.") from error


def _frozen_matrix(
    values: np.ndarray,
    *,
    name: str,
    dtype: np.dtype[np.generic] | type[np.generic] = np.float64,
) -> np.ndarray:
    """Return a copied finite two-dimensional read-only array."""
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = array.copy()
    result.setflags(write=False)
    return result


__all__ = [
    "GaussianProductSpaceTrace",
    "sample_gaussian_product_space",
]
