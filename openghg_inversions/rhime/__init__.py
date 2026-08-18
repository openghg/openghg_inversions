"""Public RHIME runners, specifications, builders, and orchestration stages.

Use ``run_rhime``, ``run_rhime_multisector``, or
``run_rhime_from_prepared_inputs`` for complete runs. Copied runners may use
the supported resolve, retrieve/reload, filter, basis, sensitivity, assembly,
alignment, materialization, build, sample, and result stages directly.
Alignment is pure; acquisition may access data, model materialization crosses
the eager backend boundary, sampling executes PyMC, and result stages may
write requested products.
"""

from __future__ import annotations

from openghg_inversions.models.rhime import RhimeModelSpec, SectorSpec
from openghg_inversions.models.rhime_likelihood import (
    RhimeLikelihoodBuilder,
    RhimeLikelihoodContext,
    RhimeLikelihoodResult,
    RhimeObservationState,
    build_absolute_sigma_gaussian_likelihood,
    build_gaussian_rhime_likelihood,
    build_rhime_observation_state,
)
from openghg_inversions.observation_error import select_aggregation_error_mode

from .builders import RhimeModelBuilder, RhimeModelBuilderContext, RhimeModelBuildResult
from .materialization import materialize_pymc_inputs
from .params import params_from_config, resolve_flux_sources
from .preparation import (
    assemble_rhime_inputs,
    build_rhime_basis,
    build_rhime_sensitivities,
    filter_rhime_observations,
    retrieve_or_reload_rhime_data,
)
from .runner import (
    RhimeResult,
    build_multisector_rhime_model,
    build_standard_rhime_model,
    make_multisector_rhime_result,
    make_standard_rhime_result,
    resolve_rhime_options,
    run_rhime,
    run_rhime_from_prepared_inputs,
    run_rhime_multisector,
    sample_rhime_model,
    with_prepared_rhime_sites,
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
    "assemble_rhime_inputs",
    "build_multisector_rhime_model",
    "build_rhime_basis",
    "build_rhime_sensitivities",
    "build_standard_rhime_model",
    "filter_rhime_observations",
    "build_absolute_sigma_gaussian_likelihood",
    "build_gaussian_rhime_likelihood",
    "build_rhime_observation_state",
    "make_multisector_rhime_result",
    "make_standard_rhime_result",
    "materialize_pymc_inputs",
    "retrieve_or_reload_rhime_data",
    "resolve_flux_sources",
    "resolve_rhime_options",
    "run_rhime",
    "run_rhime_from_prepared_inputs",
    "run_rhime_multisector",
    "sample_rhime_model",
    "select_aggregation_error_mode",
    "with_prepared_rhime_sites",
]
