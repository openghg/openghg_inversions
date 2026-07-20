"""Reusable model-building helpers for OpenGHG inversions."""

# ruff: noqa: E402

from openghg_inversions._pymc_config import configure_pytensor

configure_pytensor()

from openghg_inversions.models.components import (
    LinearComponentResult,
    StateLinearComponentResult,
    add_inferpymc_likelihood_component,
    add_linear_component,
    add_model_data,
    add_offset_component,
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
from openghg_inversions.models.state_activity import (
    ResolvedStateActivity,
    StateActivity,
    active_prior_args,
    resolve_state_activity,
)
from openghg_inversions.models.rhime import (
    DEFAULT_BC_PRIOR,
    DEFAULT_OFFSET_PRIOR,
    DEFAULT_SIGMA_PRIOR,
    DEFAULT_X_PRIOR,
    RhimeModelSpec,
    SectorSpec,
    build_rhime_model,
    build_rhime_model_from_spec,
    build_rhime_multisector_model,
    build_rhime_multisector_model_from_spec,
    safe_pymc_name,
)

__all__ = [
    "CoordRegistry",
    "DEFAULT_BC_PRIOR",
    "DEFAULT_OFFSET_PRIOR",
    "DEFAULT_SIGMA_PRIOR",
    "DEFAULT_X_PRIOR",
    "RhimeModelSpec",
    "ResolvedStateActivity",
    "SectorSpec",
    "StateActivity",
    "add_coords",
    "attach_coord_registry",
    "get_coord_registry",
    "restore_inferencedata_coords",
    "parse_prior",
    "LinearComponentResult",
    "StateLinearComponentResult",
    "add_model_data",
    "add_linear_component",
    "add_state_linear_component",
    "add_sigma_component",
    "add_offset_component",
    "add_inferpymc_likelihood_component",
    "build_rhime_model",
    "build_rhime_model_from_spec",
    "build_rhime_multisector_model",
    "build_rhime_multisector_model_from_spec",
    "safe_pymc_name",
    "active_prior_args",
    "resolve_state_activity",
]
