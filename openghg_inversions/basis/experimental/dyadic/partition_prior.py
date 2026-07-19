"""Explicit partition priors whose probability depends only on region count.

The primary construction assigns a declared marginal probability to K and
shares it uniformly among the exact number of valid dyadic partitions with
that K.  The resulting object is both a framework-independent
``Callable[[PartitionState], float]`` and a fixed lookup table suitable for a
symbolic split-mask model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .enumeration import count_partitions_by_region
from .state import PartitionState
from .tree import DyadicTree


@dataclass(frozen=True, slots=True, eq=False)
class RegionCountPartitionPrior:
    """Normalized partition prior represented by one log probability per K.

    Index zero is always negative infinity because every valid partition has at
    least one region.  Values at positive indices are log probabilities for one
    particular partition with that K, not marginal log probabilities for K.

    Attributes:
        tree: Canonical tree defining valid partitions and exact counts.
        log_probability_by_k: Read-only vector with shape
            ``(number_of_grid_cells + 1,)``.
    """

    tree: DyadicTree
    log_probability_by_k: np.ndarray

    def __post_init__(self) -> None:
        """Validate the lookup and require total prior mass one."""
        if not isinstance(self.tree, DyadicTree):
            raise TypeError("tree must be a DyadicTree.")
        canonical = DyadicTree.from_shape(self.tree.shape)
        if self.tree != canonical:
            raise ValueError("tree must be a complete canonical DyadicTree.")
        values = np.asarray(self.log_probability_by_k, dtype=float)
        expected_shape = (len(self.tree.leaf_ids) + 1,)
        if values.shape != expected_shape:
            raise ValueError(f"log_probability_by_k must have shape {expected_shape}.")
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise ValueError("log_probability_by_k must be finite or negative infinity.")
        if values[0] != -math.inf:
            raise ValueError("log_probability_by_k[0] must be negative infinity.")

        counts = count_partitions_by_region(self.tree)
        finite = np.isfinite(values[1:])
        if not finite.any():
            raise ValueError("At least one region count must have positive prior mass.")
        log_masses = np.array(
            [values[k] + math.log(counts[k]) for k in range(1, len(values))],
            dtype=float,
        )
        maximum = float(log_masses[finite].max())
        total = float(np.exp(log_masses[finite] - maximum).sum())
        log_total = maximum + math.log(total)
        if not math.isclose(log_total, 0.0, abs_tol=1e-12):
            raise ValueError("Partition prior probabilities must sum to one.")

        frozen = values.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "log_probability_by_k", frozen)

    @classmethod
    def uniform_k(
        cls,
        tree: DyadicTree,
        *,
        minimum_regions: int = 1,
        maximum_regions: int | None = None,
    ) -> RegionCountPartitionPrior:
        """Assign uniform marginal mass to an inclusive K interval.

        Conditional on K, each valid partition receives equal probability.

        Args:
            tree: Canonical dyadic tree.
            minimum_regions: Smallest K with positive marginal mass.
            maximum_regions: Largest K with positive marginal mass. Defaults to
                the number of finest grid cells.

        Returns:
            Normalized callable partition prior with ``p(P)=p(K)/N_K``.

        Raises:
            TypeError: If either bound is not an integer.
            ValueError: If the inclusive interval lies outside the tree.
        """
        if not isinstance(tree, DyadicTree):
            raise TypeError("tree must be a DyadicTree.")
        maximum = len(tree.leaf_ids) if maximum_regions is None else maximum_regions
        minimum = _integer_bound(minimum_regions, name="minimum_regions")
        maximum = _integer_bound(maximum, name="maximum_regions")
        if not 1 <= minimum <= maximum <= len(tree.leaf_ids):
            raise ValueError("Region-count bounds must define a valid inclusive interval.")

        counts = count_partitions_by_region(tree, max_regions=maximum)
        values = np.full(len(tree.leaf_ids) + 1, -math.inf, dtype=float)
        log_k_mass = -math.log(maximum - minimum + 1)
        for region_count in range(minimum, maximum + 1):
            values[region_count] = log_k_mass - math.log(counts[region_count])
        return cls(tree=tree, log_probability_by_k=values)

    @property
    def marginal_probability_by_k(self) -> np.ndarray:
        """Return normalized marginal prior probabilities for all K."""
        counts = count_partitions_by_region(self.tree)
        result = np.zeros_like(self.log_probability_by_k)
        for region_count, partition_count in counts.items():
            log_probability = self.log_probability_by_k[region_count]
            if np.isfinite(log_probability):
                result[region_count] = math.exp(log_probability) * partition_count
        result.setflags(write=False)
        return result

    def __call__(self, partition: PartitionState) -> float:
        """Return normalized log prior probability for one partition.

        Args:
            partition: Frontier that is valid on :attr:`tree`. Partition states
                do not carry tree identity, so validation is structural.

        Returns:
            Log probability for that complete partition, or negative infinity
            when its K has zero prior mass.
        """
        if not isinstance(partition, PartitionState):
            raise TypeError("partition must be a PartitionState.")
        partition.validate(self.tree)
        return float(self.log_probability_by_k[len(partition.active)])


def _integer_bound(value: int, *, name: str) -> int:
    """Return a built-in integer bound while rejecting coercions."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


__all__ = ["RegionCountPartitionPrior"]
