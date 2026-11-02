"""Reusable model components, coordinates, priors, and state APIs.

Linear sensitivity preparation removes structural zero columns once before graph
construction; state activity then controls scientific fixing among the full
labelled state.

Importing this package configures PyTensor before re-exporting reusable model
primitives. RHIME-specific recipes and contracts live in ``openghg_inversions.rhime``.
"""

# ruff: noqa: E402

from openghg_inversions._pymc_config import configure_pytensor

configure_pytensor()

from openghg_inversions.models.components import (
    CorrelatedStateResult,
    LinearComponentResult,
    apply_linear_sensitivity,
    add_coherent_affine_component,
    add_linked_linear_component,
    add_linear_component,
    add_model_data,
    add_offset_component,
    add_correlated_lognormal_state,
    add_correlated_lognormal_state_with_activity,
    add_sigma_component,
)
from openghg_inversions.models.cached_sigma import (
    FixedOuCachedSigmaTarget,
    MarginalQuadraticCache,
)
from openghg_inversions.models.coords import (
    CoordRegistry,
    add_coords,
    attach_coord_registry,
    get_coord_registry,
    registered_model,
    restore_inferencedata_coords,
)
from openghg_inversions.models.priors import parse_prior
from openghg_inversions.models.scalar_sigma import (
    ScalarSigmaEigenbasis,
    add_scalar_sigma_eigen_likelihood,
    load_scalar_sigma_eigenbasis,
    prepare_scalar_sigma_eigenbasis,
    save_scalar_sigma_eigenbasis,
)
from openghg_inversions.models.site_sigma import add_site_sigma_gaussian_likelihood
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.state_activity import (
    PreparedLinearSensitivity,
    ResolvedStateActivity,
    StateActivity,
    active_prior_args,
    detect_zero_sensitivity,
    prepare_linear_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.observation_error import AggregationErrorMode

__all__ = [
    "CoordRegistry",
    "CorrelatedLognormalPrior",
    "CorrelatedStateResult",
    "LinearComponentResult",
    "AggregationErrorMode",
    "FixedOuCachedSigmaTarget",
    "MarginalQuadraticCache",
    "ResolvedStateActivity",
    "PreparedLinearSensitivity",
    "StateActivity",
    "ScalarSigmaEigenbasis",
    "add_scalar_sigma_eigen_likelihood",
    "add_coords",
    "attach_coord_registry",
    "get_coord_registry",
    "registered_model",
    "restore_inferencedata_coords",
    "parse_prior",
    "prepare_scalar_sigma_eigenbasis",
    "save_scalar_sigma_eigenbasis",
    "load_scalar_sigma_eigenbasis",
    "add_model_data",
    "add_correlated_lognormal_state",
    "add_correlated_lognormal_state_with_activity",
    "add_coherent_affine_component",
    "add_linked_linear_component",
    "add_linear_component",
    "apply_linear_sensitivity",
    "add_sigma_component",
    "add_site_sigma_gaussian_likelihood",
    "add_offset_component",
    "active_prior_args",
    "detect_zero_sensitivity",
    "prepare_linear_sensitivity",
    "resolve_state_activity",
]
