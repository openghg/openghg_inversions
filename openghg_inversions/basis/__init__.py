"""Basis construction, projection, and retained covariance-product interfaces.

The package provides basis-generation functions, flux-weighted preparation
wrappers, prior-width projection helpers, and the public OPE-17 retained
covariance-product API.

Basis operators own grid/state geometry: their bucket matrix is a prolongation
from retained scalings to the native grid, not an automatic retained
restriction. ``FluxWeightedBasis``, exported here as ``BasisFunctions``, pairs
that geometry with flux for sensitivity projection and reconstruction; it does
not own native covariance transforms. Project-owned workflows can compose the
basis algorithms and use ``basis_functions_from_fp_all_flat_basis`` to attach
current-run flux without importing private preparation functions.

Native covariance actions live in :mod:`openghg_inversions.native_covariance`
and :mod:`openghg_inversions.source_covariance`. The interfaces re-exported
here choose a compatible restriction/prolongation pair and prepare labelled
product blocks. They are a low-level input to later coherent reduction and do
not themselves construct the centred reduced likelihood, unresolved
covariance, or a complete coherent-reduction artifact.
"""

from ._functions import (
    basis_weights_from_fp_all,
    bucket_basis_from_weights,
    bucket_basis_function,
    bucketbasisfunction,
    fixed_outer_regions_basis,
    load_country_region_classes,
    load_intem_outer_regions,
    paired_abs_response_weights,
    quadtree_basis_from_weights,
    quadtree_basis_function,
    quadtreebasisfunction,
    region_constrained_basis_from_weights,
    region_constrained_fixed_outer_basis_from_weights,
    region_constrained_basis_function,
)
from ._wrapper import (
    basis_functions_wrapper,
    load_basis_functions,
    make_basis_functions,
)
from .basis_functions import BasisFunctions, basis_functions_from_fp_all_flat_basis
from .prior_uncertainty import (
    MEAN_TOTAL_TARGET_STATISTIC,
    MEDIAN_RELATIVE_TARGET_STATISTIC,
    calibrate_basis_prior_stdev,
    project_basis_prior_stdev,
)
from .covariance_products import (
    NativeCovarianceProducts,
    PreserveBucketProlongation,
    RetainedProjection,
    RetainedProjectionStrategy,
    project_native_covariance,
)

__all__ = [
    "basis_functions_wrapper",
    "basis_functions_from_fp_all_flat_basis",
    "basis_weights_from_fp_all",
    "BasisFunctions",
    "bucket_basis_from_weights",
    "bucket_basis_function",
    "bucketbasisfunction",
    "fixed_outer_regions_basis",
    "load_basis_functions",
    "load_country_region_classes",
    "load_intem_outer_regions",
    "MEAN_TOTAL_TARGET_STATISTIC",
    "MEDIAN_RELATIVE_TARGET_STATISTIC",
    "NativeCovarianceProducts",
    "PreserveBucketProlongation",
    "RetainedProjection",
    "RetainedProjectionStrategy",
    "make_basis_functions",
    "paired_abs_response_weights",
    "project_basis_prior_stdev",
    "project_native_covariance",
    "quadtree_basis_from_weights",
    "quadtree_basis_function",
    "quadtreebasisfunction",
    "region_constrained_basis_from_weights",
    "region_constrained_fixed_outer_basis_from_weights",
    "region_constrained_basis_function",
    "calibrate_basis_prior_stdev",
]
