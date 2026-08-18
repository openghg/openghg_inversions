"""Compatibility imports for the public RHIME runner surface.

Scientific execution now lives in the readable :mod:`.standard` and
:mod:`.multisector` recipe modules. New code may import those modules directly;
the established ``openghg_inversions.rhime.runner`` imports remain supported.
"""

from openghg_inversions.models import RhimeModelSpec, SectorSpec
from openghg_inversions.observation_error import select_aggregation_error_mode

from .builders import RhimeModelBuilder, RhimeModelBuilderContext, RhimeModelBuildResult
from .materialization import materialize_pymc_inputs
from .multisector import (
    build_multisector_rhime_model,
    make_multisector_rhime_result,
    run_rhime_multisector,
)
from .outputs import RhimeResult
from .params import params_from_config, resolve_flux_sources, resolve_rhime_options
from .preparation import (
    assemble_rhime_inputs,
    build_rhime_basis,
    build_rhime_sensitivities,
    filter_rhime_observations,
    retrieve_or_reload_rhime_data,
    with_prepared_rhime_sites,
)
from .prepared import run_rhime_from_prepared_inputs
from .sampling import RhimeSampler, sample_rhime_model
from .specs import RhimeOutputSpec, RhimeRunSpec
from .standard import build_standard_rhime_model, make_standard_rhime_result, run_rhime

__all__ = [
    "SectorSpec",
    "RhimeModelSpec",
    "RhimeModelBuilder",
    "RhimeModelBuilderContext",
    "RhimeModelBuildResult",
    "RhimeOutputSpec",
    "RhimeSampler",
    "RhimeRunSpec",
    "RhimeResult",
    "assemble_rhime_inputs",
    "build_multisector_rhime_model",
    "build_rhime_basis",
    "build_rhime_sensitivities",
    "build_standard_rhime_model",
    "filter_rhime_observations",
    "make_multisector_rhime_result",
    "make_standard_rhime_result",
    "materialize_pymc_inputs",
    "params_from_config",
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
