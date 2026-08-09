"""Public model builders, coordinates, priors, and state-activity APIs.

State activity separates design inspection with ``detect_zero_sensitivity``,
policy resolution with ``resolve_state_activity``, and graph construction in
the component helpers.

Importing this package configures PyTensor before re-exporting the supported
RHIME and reusable component entry points.
"""

# ruff: noqa: E402

from openghg_inversions._pymc_config import configure_pytensor

configure_pytensor()

from openghg_inversions.models.components import (
    CorrelatedStateResult,
    LinearComponentResult,
    StateLinearComponentResult,
    add_inferpymc_likelihood_component,
    add_linear_component,
    add_model_data,
    add_offset_component,
    add_correlated_lognormal_state,
    add_sigma_component,
    add_state_linear_component,
)
from openghg_inversions.models.coords import (
    CoordRegistry,
    add_coords,
    attach_coord_registry,
    get_coord_registry,
    restore_inferencedata_coords,
)
from openghg_inversions.models.priors import parse_prior
from openghg_inversions.correlated_state import (
    CorrelatedLognormalPrior,
    MarginalCorrelatedLognormalPrior,
)
from openghg_inversions.models.state_activity import (
    ResolvedStateActivity,
    StateActivity,
    active_prior_args,
    detect_zero_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.models.rhime import (
    DEFAULT_BC_PRIOR,
    DEFAULT_OFFSET_PRIOR,
    DEFAULT_SIGMA_PRIOR,
    DEFAULT_X_PRIOR,
    RhimeBuilderStrategy,
    RhimeModelSpec,
    SectorSpec,
    build_rhime_model,
    build_rhime_model_from_spec,
    build_rhime_multisector_model,
    build_rhime_multisector_model_from_spec,
    get_rhime_likelihood_result,
    safe_pymc_name,
)
from openghg_inversions.models.rhime_likelihood import (
    RhimeLikelihoodBuilder,
    RhimeLikelihoodContext,
    RhimeLikelihoodResult,
    RhimeObservationState,
    add_rhime_likelihood_component,
    build_absolute_sigma_gaussian_likelihood,
    build_gaussian_rhime_likelihood,
    build_rhime_observation_state,
)
from openghg_inversions.observation_error import AggregationErrorMode

__all__ = [
    "CoordRegistry",
    "CorrelatedLognormalPrior",
    "CorrelatedStateResult",
    "DEFAULT_BC_PRIOR",
    "DEFAULT_OFFSET_PRIOR",
    "DEFAULT_SIGMA_PRIOR",
    "DEFAULT_X_PRIOR",
    "AggregationErrorMode",
    "RhimeBuilderStrategy",
    "RhimeLikelihoodBuilder",
    "RhimeLikelihoodContext",
    "RhimeLikelihoodResult",
    "RhimeModelSpec",
    "RhimeObservationState",
    "ResolvedStateActivity",
    "SectorSpec",
    "StateActivity",
    "add_coords",
    "attach_coord_registry",
    "get_coord_registry",
    "restore_inferencedata_coords",
    "parse_prior",
    "LinearComponentResult",
    "MarginalCorrelatedLognormalPrior",
    "StateLinearComponentResult",
    "add_model_data",
    "add_correlated_lognormal_state",
    "add_linear_component",
    "add_state_linear_component",
    "add_sigma_component",
    "add_offset_component",
    "add_inferpymc_likelihood_component",
    "add_rhime_likelihood_component",
    "build_absolute_sigma_gaussian_likelihood",
    "build_gaussian_rhime_likelihood",
    "build_rhime_observation_state",
    "build_rhime_model",
    "build_rhime_model_from_spec",
    "build_rhime_multisector_model",
    "build_rhime_multisector_model_from_spec",
    "get_rhime_likelihood_result",
    "safe_pymc_name",
    "active_prior_args",
    "detect_zero_sensitivity",
    "resolve_state_activity",
]
