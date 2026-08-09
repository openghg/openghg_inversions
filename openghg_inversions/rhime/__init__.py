"""Public RHIME API."""

from __future__ import annotations

from openghg_inversions.models.rhime import RhimeModelSpec, SectorSpec
from openghg_inversions.models.rhime_likelihood import (
    RhimeLikelihoodBuilder,
    RhimeLikelihoodContext,
    RhimeLikelihoodResult,
    RhimeObservationState,
    build_gaussian_rhime_likelihood,
    build_rhime_observation_state,
)

from .builders import RhimeModelBuilder, RhimeModelBuilderContext, RhimeModelBuildResult
from .params import params_from_config, resolve_flux_sources
from .runner import (
    RhimeResult,
    run_rhime,
    run_rhime_from_prepared_inputs,
    run_rhime_multisector,
)
from .sampling import RhimeSampler
from .specs import RhimeOutputSpec, RhimeRunSpec

__all__ = [
    "SectorSpec",
    "RhimeModelSpec",
    "RhimeModelBuilder",
    "RhimeModelBuilderContext",
    "RhimeModelBuildResult",
    "RhimeLikelihoodBuilder",
    "RhimeLikelihoodContext",
    "RhimeLikelihoodResult",
    "RhimeObservationState",
    "RhimeOutputSpec",
    "RhimeSampler",
    "RhimeRunSpec",
    "RhimeResult",
    "params_from_config",
    "build_gaussian_rhime_likelihood",
    "build_rhime_observation_state",
    "resolve_flux_sources",
    "run_rhime",
    "run_rhime_from_prepared_inputs",
    "run_rhime_multisector",
]
