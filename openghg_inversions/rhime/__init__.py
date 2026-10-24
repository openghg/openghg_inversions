"""Public RHIME runners, specifications, builders, and orchestration stages.

Use ``run(..., model="standard")`` with an explicit model selection, or the
prepared-input entry points for complete runs. Copied runners may use
the supported resolve, acquire, observation-error, filter, basis, sensitivity, assembly,
alignment, materialization, build, sample, result, and output stages directly.
Alignment is pure; acquisition may access data, model materialization crosses
the eager backend boundary, sampling executes PyMC, and output stages may write
requested products.
"""

from __future__ import annotations

# ruff: noqa: E402

# PyTensor precision must be selected before importing likelihoods, recipes, or
# any other module that imports PyMC.
from openghg_inversions._pymc_config import configure_pytensor

configure_pytensor()

from .builders import (
    RhimeLikelihoodBuilder,
    RhimeModelBuildResult,
)
from .co2 import build_co2_model, co2_model_input_names, run_co2
from openghg_inversions.inversion_data import MergedData, PreparedInputs, SiteOptions

from .api import run
from .materialization import materialize_pymc_inputs
from .ini import read_rhime_ini
from .params import (
    DataConfig,
    group_params,
    RhimeConfig,
    resolve_flux_sources,
)
from .preparation import (
    assemble_inputs,
    build_sensitivities,
    filter_observations,
    prepare_observation_errors,
)
from .multisector import (
    build_multisector_model,
    prepare_multisector,
    materialize_multisector_inputs,
    multisector_model_input_names,
    run_multisector,
)
from .nested import (
    NestedRhimePreparedInputs,
    NestedOutput,
    inner_data_config,
    materialize_nested_inputs,
    make_nested_inversion_outputs,
    align_inner_merged_to_outer_observations,
    build_nested_rhime_model,
    combine_nested_rhime_inputs,
    mask_outer_merged_for_inner_domain,
    nested_model_input_names,
    prepare_nested_rhime_inputs,
    run_nested,
)
from .outputs import (
    make_model_output,
    postprocess_multisector_outputs,
    save_inversion_output,
    validate_multisector_output_resources,
    annotate_sample_trace,
    postprocess_standard_outputs,
    validate_output_resources,
    validate_standard_output_resources,
)
from .sampling import RhimeSampler, sample_rhime_model
from .stages import (
    CONVERGENCE_CHECK_NAME,
    PREPARATION_CHECK_NAME,
    diagnose_rhime_stage,
)
from .standard import (
    build_standard_model,
    materialize_standard_inputs,
    run_standard,
    standard_model_input_names,
)
from .specs import (
    AdditiveSigmaSettings,
    FixedErrorSettings,
    PollutionEventSettings,
    RhimeModelSpec,
    RhimeOutputSpec,
    SectorSpec,
)

__all__ = [
    "run",
    "run_standard",
    "run_multisector",
    "run_nested",
    "MergedData",
    "PreparedInputs",
    "assemble_inputs",
    "RhimeConfig",
    "DataConfig",
    "group_params",
    "read_rhime_ini",
    "SectorSpec",
    "AdditiveSigmaSettings",
    "FixedErrorSettings",
    "PollutionEventSettings",
    "RhimeModelSpec",
    "RhimeLikelihoodBuilder",
    "NestedRhimePreparedInputs",
    "NestedOutput",
    "inner_data_config",
    "materialize_nested_inputs",
    "make_nested_inversion_outputs",
    "RhimeModelBuildResult",
    "RhimeOutputSpec",
    "RhimeSampler",
    "CONVERGENCE_CHECK_NAME",
    "PREPARATION_CHECK_NAME",
    "align_inner_merged_to_outer_observations",
    "build_multisector_model",
    "prepare_multisector",
    "materialize_multisector_inputs",
    "make_model_output",
    "postprocess_multisector_outputs",
    "save_inversion_output",
    "validate_multisector_output_resources",
    "build_co2_model",
    "build_nested_rhime_model",
    "build_sensitivities",
    "build_standard_model",
    "materialize_standard_inputs",
    "filter_observations",
    "prepare_observation_errors",
    "diagnose_rhime_stage",
    "combine_nested_rhime_inputs",
    "mask_outer_merged_for_inner_domain",
    "nested_model_input_names",
    "prepare_nested_rhime_inputs",
    "annotate_sample_trace",
    "postprocess_standard_outputs",
    "validate_output_resources",
    "validate_standard_output_resources",
    "materialize_pymc_inputs",
    "co2_model_input_names",
    "multisector_model_input_names",
    "SiteOptions",
    "resolve_flux_sources",
    "run_co2",
    "sample_rhime_model",
    "standard_model_input_names",
]
