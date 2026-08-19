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

# ruff: noqa: E402

# PyTensor precision must be selected before importing likelihoods, recipes, or
# any other module that imports PyMC.
from openghg_inversions._pymc_config import configure_pytensor

configure_pytensor()

from openghg_inversions.observation_error import select_aggregation_error_mode

from .builders import (
    RhimeLikelihoodBuilder,
    RhimeLikelihoodContext,
    RhimeLikelihoodResult,
    RhimeModelBuilder,
    RhimeModelBuilderContext,
    RhimeModelBuildResult,
)
from .materialization import materialize_pymc_inputs
from .params import params_from_config, resolve_flux_sources, resolve_rhime_options
from .preparation import (
    assemble_rhime_inputs,
    build_rhime_basis,
    build_rhime_sensitivities,
    filter_rhime_observations,
    retrieve_or_reload_rhime_data,
    with_prepared_rhime_sites,
)
from .multisector import (
    build_multisector_rhime_model,
    build_multisector_rhime_model_result,
    make_multisector_rhime_result,
    run_rhime_multisector,
)
from .outputs import RhimeResult
from .prepared import run_rhime_from_prepared_inputs
from .sampling import RhimeSampler, sample_rhime_model
from .standard import (
    build_standard_rhime_model,
    build_standard_rhime_model_result,
    make_standard_rhime_result,
    run_rhime,
)
from .specs import RhimeModelSpec, RhimeOutputSpec, RhimeRunSpec, SectorSpec

__all__ = [
    "SectorSpec",
    "RhimeModelSpec",
    "RhimeLikelihoodBuilder",
    "RhimeLikelihoodContext",
    "RhimeLikelihoodResult",
    "RhimeModelBuilder",
    "RhimeModelBuilderContext",
    "RhimeModelBuildResult",
    "RhimeOutputSpec",
    "RhimeSampler",
    "RhimeRunSpec",
    "RhimeResult",
    "params_from_config",
    "assemble_rhime_inputs",
    "build_multisector_rhime_model",
    "build_multisector_rhime_model_result",
    "build_rhime_basis",
    "build_rhime_sensitivities",
    "build_standard_rhime_model",
    "build_standard_rhime_model_result",
    "filter_rhime_observations",
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
