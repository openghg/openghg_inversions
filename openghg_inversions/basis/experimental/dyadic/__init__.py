"""Experimental canonical binary partitions for rectangular grids.

This package provides a small, pure-NumPy representation of one deterministic
binary tree, immutable active frontiers, partition objectives, local proposals,
and a generic stochastic local-search runner. It is provisional and is not
re-exported from :mod:`openghg_inversions.basis`.
"""

from .initializers import InitializationResult, greedy_partition, random_partition, threshold_partition
from .demo_runner import (
    DemoSearchConfig,
    DemoSearchRun,
    VariableKSearchConfig,
    VariableKSearchRun,
    excess_region_penalty,
    run_fixed_count_dfs_search,
    run_variable_k_dfs_search,
)
from .multiscale import CoarsenedGrid, MultiscaleDesign, direct_gather, sum_coarsen_grid
from .objectives import (
    CovarianceBuilder,
    GaussianDFSObjective,
    IsotropicRegionCovariance,
    direct_observation_space_dfs,
    gaussian_dfs,
    isotropic_observation_space_dfs,
    prototype_quadratic_tile_scores,
)
from .proposals import (
    MergeMove,
    Move,
    PairedMove,
    PairedNeighbor,
    SplitMove,
    apply_move,
    enumerate_merge_moves,
    enumerate_paired_moves,
    enumerate_paired_neighbors,
    enumerate_split_moves,
    reverse_move,
)
from .search import (
    PiecewiseGeometricSchedule,
    SearchProposal,
    SearchResult,
    SearchStep,
    TemperatureSchedule,
    stochastic_local_search,
)
from .state import PartitionState
from .tree import DyadicTree, NodeId, Tile

__all__ = [
    "CoarsenedGrid",
    "CovarianceBuilder",
    "DemoSearchConfig",
    "DemoSearchRun",
    "DyadicTree",
    "GaussianDFSObjective",
    "InitializationResult",
    "IsotropicRegionCovariance",
    "MergeMove",
    "Move",
    "MultiscaleDesign",
    "NodeId",
    "PairedMove",
    "PairedNeighbor",
    "PartitionState",
    "PiecewiseGeometricSchedule",
    "SearchProposal",
    "SearchResult",
    "SearchStep",
    "SplitMove",
    "TemperatureSchedule",
    "Tile",
    "VariableKSearchConfig",
    "VariableKSearchRun",
    "apply_move",
    "direct_gather",
    "direct_observation_space_dfs",
    "enumerate_merge_moves",
    "enumerate_paired_moves",
    "enumerate_paired_neighbors",
    "enumerate_split_moves",
    "excess_region_penalty",
    "gaussian_dfs",
    "isotropic_observation_space_dfs",
    "greedy_partition",
    "prototype_quadratic_tile_scores",
    "random_partition",
    "reverse_move",
    "run_fixed_count_dfs_search",
    "run_variable_k_dfs_search",
    "stochastic_local_search",
    "sum_coarsen_grid",
    "threshold_partition",
]
