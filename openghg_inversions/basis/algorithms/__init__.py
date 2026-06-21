"""Algorithms for computing basis functions."""

from ._constrained import (
    AllocationMode,
    AxisParallelSplitStep,
    AxisAlignedWeightedSplitStrategy,
    GreedyAxisParallelSplitStrategy,
    MinChildWeightShare,
    PartitionStep,
    SplitAcceptancePolicy,
    SplitStrategy,
    allocate_nbasis_by_class,
    region_constrained_basis,
)
from ._quadtree import get_quadtree_basis as quadtree_algorithm
from ._weighted import nregion_landsea_basis as weighted_algorithm

__all__ = [
    "AllocationMode",
    "AxisParallelSplitStep",
    "AxisAlignedWeightedSplitStrategy",
    "GreedyAxisParallelSplitStrategy",
    "MinChildWeightShare",
    "SplitStrategy",
    "PartitionStep",
    "SplitAcceptancePolicy",
    "allocate_nbasis_by_class",
    "quadtree_algorithm",
    "region_constrained_basis",
    "weighted_algorithm",
]
