"""Algorithms for computing basis functions."""

from ._constrained import (
    AllocationMode,
    AllSplitAcceptancePolicies,
    AxisParallelSplitStep,
    AxisAlignedWeightedSplitStrategy,
    GreedyAxisParallelSplitStrategy,
    MinChildTargetWeightShare,
    MinChildWeightShare,
    PartitionStep,
    SplitAcceptance,
    SplitAcceptancePolicy,
    SplitStrategy,
    TargetSplitAcceptancePolicy,
    allocate_nbasis_by_class,
    region_constrained_basis,
)
from ._quadtree import get_quadtree_basis as quadtree_algorithm
from ._weighted import nregion_landsea_basis as weighted_algorithm

__all__ = [
    "AllocationMode",
    "AllSplitAcceptancePolicies",
    "AxisParallelSplitStep",
    "AxisAlignedWeightedSplitStrategy",
    "GreedyAxisParallelSplitStrategy",
    "MinChildTargetWeightShare",
    "MinChildWeightShare",
    "SplitAcceptance",
    "SplitStrategy",
    "PartitionStep",
    "SplitAcceptancePolicy",
    "TargetSplitAcceptancePolicy",
    "allocate_nbasis_by_class",
    "quadtree_algorithm",
    "region_constrained_basis",
    "weighted_algorithm",
]
