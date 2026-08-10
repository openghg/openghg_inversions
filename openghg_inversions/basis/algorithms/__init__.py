"""Public algorithms for computing basis functions.

This package re-exports weighted and quadtree algorithms alongside constrained
basis generation, class composition, allocation, and tuple-safe class masking
helpers.
"""

from ._consolidation import (
    ContrastProximityComponentConsolidation,
    InactiveComponentPolicy,
)
from ._constrained import (
    AllocationMode,
    AllSplitAcceptancePolicies,
    AxisAlignedWeightedSplitStrategy,
    AxisParallelSplitStep,
    ComponentConsolidationPolicy,
    ConnectedBinaryPartitionStep,
    ConnectedComponentPartitionStep,
    ConnectedComponentSplitStrategy,
    GreedySplitStrategy,
    InertialSplitStep,
    LatLonGridGeometry,
    MaxChildPCAEccentricity,
    MinChildTargetWeightShare,
    MinChildWeightShare,
    NbasisAllocation,
    SplitGeometry,
    SplitStrategy,
    allocate_nbasis_by_class,
    combine_inner_outer_region_classes,
    intersect_region_class_layers,
    normalize_spatial_grid,
    region_class_mask,
    region_constrained_basis,
)
from ._contrast import (
    ContrastScoreSplitAcceptance,
    SplitContrastScore,
    contrast_tau_from_multiplier_cv,
    split_contrast_score,
)
from ._quadtree import get_quadtree_basis as quadtree_algorithm
from ._partition import (
    GridNode,
    GridPartition,
    PartitionStep,
    SplitAcceptance,
    SplitAcceptancePolicy,
    TargetSplitAcceptancePolicy,
    greedy_partitioning,
)
from ._weighted import nregion_landsea_basis as weighted_algorithm

__all__ = [
    "AllSplitAcceptancePolicies",
    "AllocationMode",
    "AxisAlignedWeightedSplitStrategy",
    "AxisParallelSplitStep",
    "ComponentConsolidationPolicy",
    "ConnectedBinaryPartitionStep",
    "ConnectedComponentPartitionStep",
    "ConnectedComponentSplitStrategy",
    "ContrastProximityComponentConsolidation",
    "ContrastScoreSplitAcceptance",
    "GreedySplitStrategy",
    "GridNode",
    "GridPartition",
    "InactiveComponentPolicy",
    "InertialSplitStep",
    "LatLonGridGeometry",
    "MaxChildPCAEccentricity",
    "MinChildTargetWeightShare",
    "MinChildWeightShare",
    "NbasisAllocation",
    "PartitionStep",
    "SplitAcceptance",
    "SplitAcceptancePolicy",
    "SplitContrastScore",
    "SplitGeometry",
    "SplitStrategy",
    "TargetSplitAcceptancePolicy",
    "allocate_nbasis_by_class",
    "combine_inner_outer_region_classes",
    "contrast_tau_from_multiplier_cv",
    "greedy_partitioning",
    "intersect_region_class_layers",
    "normalize_spatial_grid",
    "quadtree_algorithm",
    "region_class_mask",
    "region_constrained_basis",
    "split_contrast_score",
    "weighted_algorithm",
]
