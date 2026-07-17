"""Experimental canonical binary partitions for rectangular grids.

This package provides NumPy/SciPy reference implementations for one
deterministic binary tree, immutable active frontiers, Gaussian projection and
partition objectives, local proposals, and stochastic local search. It is
provisional and is not re-exported from :mod:`openghg_inversions.basis`.
"""

from .initializers import InitializationResult, greedy_partition, random_partition, threshold_partition
from .demo_runner import (
    DemoSearchConfig,
    DemoSearchRun,
    ProjectedVariableKSearchRun,
    VariableKSearchConfig,
    VariableKSearchRun,
    excess_region_penalty,
    run_fixed_count_dfs_search,
    run_projected_variable_k_dfs_search,
    run_variable_k_dfs_search,
)
from .dynamic_programming import (
    AdditivePartitionSolution,
    additive_partition_frontier,
    optimal_additive_partition,
)
from .gaussian_projection import (
    BocquetProjection,
    GaussianPosterior,
    GaussianProjectionAnalysis,
    build_bocquet_projection,
    equation_45_objective,
    gaussian_projection_oracle,
    native_gaussian_posterior,
    projected_bayesian_information_gain,
    projected_bayesian_kl,
    projected_dfs,
    projected_fisher_aggregation_aware,
    projected_fisher_base_r,
    reduced_gaussian_posterior,
    restriction_for_prolongation,
)
from .grid_covariance import SeparableGridCovariance
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
from .partition_diagnostics import (
    GaussianPartitionDiagnostics,
    GaussianPartitionObjectives,
    build_partition_diagnostics,
    emissions_compression_quality,
    gaussian_partition_objectives,
    gaussian_posterior_mean,
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
from .rhime_gaussian import NativePosteriorMarginals, RHIMEGaussianMultiscale
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
    "AdditivePartitionSolution",
    "BocquetProjection",
    "CoarsenedGrid",
    "CovarianceBuilder",
    "DemoSearchConfig",
    "DemoSearchRun",
    "DyadicTree",
    "GaussianDFSObjective",
    "GaussianPartitionDiagnostics",
    "GaussianPartitionObjectives",
    "GaussianPosterior",
    "GaussianProjectionAnalysis",
    "InitializationResult",
    "IsotropicRegionCovariance",
    "MergeMove",
    "Move",
    "MultiscaleDesign",
    "NativePosteriorMarginals",
    "NodeId",
    "PairedMove",
    "PairedNeighbor",
    "PartitionState",
    "PiecewiseGeometricSchedule",
    "ProjectedVariableKSearchRun",
    "RHIMEGaussianMultiscale",
    "SearchProposal",
    "SearchResult",
    "SearchStep",
    "SeparableGridCovariance",
    "SplitMove",
    "TemperatureSchedule",
    "Tile",
    "VariableKSearchConfig",
    "VariableKSearchRun",
    "apply_move",
    "additive_partition_frontier",
    "build_partition_diagnostics",
    "build_bocquet_projection",
    "direct_gather",
    "direct_observation_space_dfs",
    "enumerate_merge_moves",
    "enumerate_paired_moves",
    "enumerate_paired_neighbors",
    "enumerate_split_moves",
    "emissions_compression_quality",
    "equation_45_objective",
    "excess_region_penalty",
    "gaussian_dfs",
    "gaussian_partition_objectives",
    "gaussian_projection_oracle",
    "gaussian_posterior_mean",
    "isotropic_observation_space_dfs",
    "native_gaussian_posterior",
    "greedy_partition",
    "optimal_additive_partition",
    "prototype_quadratic_tile_scores",
    "projected_bayesian_information_gain",
    "projected_bayesian_kl",
    "projected_dfs",
    "projected_fisher_aggregation_aware",
    "projected_fisher_base_r",
    "random_partition",
    "reduced_gaussian_posterior",
    "restriction_for_prolongation",
    "reverse_move",
    "run_fixed_count_dfs_search",
    "run_projected_variable_k_dfs_search",
    "run_variable_k_dfs_search",
    "stochastic_local_search",
    "sum_coarsen_grid",
    "threshold_partition",
]
