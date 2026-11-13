"""Type stubs for the lazily exported RHIME public API."""

from .api import (
    run as run,
)
from .standard import (
    run_standard as run_standard,
    build_standard_model as build_standard_model,
    materialize_standard_inputs as materialize_standard_inputs,
    standard_model_input_names as standard_model_input_names,
)
from .multisector import (
    run_multisector as run_multisector,
    build_multisector_model as build_multisector_model,
    prepare_multisector as prepare_multisector,
    materialize_multisector_inputs as materialize_multisector_inputs,
    multisector_model_input_names as multisector_model_input_names,
)
from .nested import (
    run_nested as run_nested,
    NestedRhimePreparedInputs as NestedRhimePreparedInputs,
    NestedOutput as NestedOutput,
    inner_data_config as inner_data_config,
    materialize_nested_inputs as materialize_nested_inputs,
    make_nested_inversion_outputs as make_nested_inversion_outputs,
    align_inner_merged_to_outer_observations as align_inner_merged_to_outer_observations,
    build_nested_rhime_model as build_nested_rhime_model,
    combine_nested_rhime_inputs as combine_nested_rhime_inputs,
    mask_outer_merged_for_inner_domain as mask_outer_merged_for_inner_domain,
    nested_model_input_names as nested_model_input_names,
    prepare_nested_rhime_inputs as prepare_nested_rhime_inputs,
)
from openghg_inversions.inversion_data import (
    MergedData as MergedData,
    PreparedInputs as PreparedInputs,
    SiteOptions as SiteOptions,
)
from .preparation import (
    assemble_inputs as assemble_inputs,
    build_sensitivities as build_sensitivities,
    filter_observations as filter_observations,
    prepare_observation_errors as prepare_observation_errors,
)
from .params import (
    RhimeConfig as RhimeConfig,
    DataConfig as DataConfig,
    group_params as group_params,
    resolve_flux_sources as resolve_flux_sources,
)
from .ini import (
    read_rhime_ini as read_rhime_ini,
)
from .specs import (
    SectorSpec as SectorSpec,
    AdditiveSigmaSettings as AdditiveSigmaSettings,
    FixedErrorSettings as FixedErrorSettings,
    PollutionEventSettings as PollutionEventSettings,
    RhimeModelSpec as RhimeModelSpec,
    RhimeOutputSpec as RhimeOutputSpec,
)
from .builders import (
    RhimeLikelihoodBuilder as RhimeLikelihoodBuilder,
    RhimeModelBuildResult as RhimeModelBuildResult,
)
from .sampling import (
    RhimeSampler as RhimeSampler,
    sample_rhime_model as sample_rhime_model,
)
from .stages import (
    CONVERGENCE_CHECK_NAME as CONVERGENCE_CHECK_NAME,
    PREPARATION_CHECK_NAME as PREPARATION_CHECK_NAME,
    diagnose_rhime_stage as diagnose_rhime_stage,
)
from .outputs import (
    make_model_output as make_model_output,
    postprocess_multisector_outputs as postprocess_multisector_outputs,
    save_inversion_output as save_inversion_output,
    validate_multisector_output_resources as validate_multisector_output_resources,
    annotate_sample_trace as annotate_sample_trace,
    postprocess_standard_outputs as postprocess_standard_outputs,
    validate_output_resources as validate_output_resources,
    validate_standard_output_resources as validate_standard_output_resources,
)
from .co2 import (
    build_co2_model as build_co2_model,
    co2_model_input_names as co2_model_input_names,
    run_co2 as run_co2,
)
from .materialization import (
    materialize_pymc_inputs as materialize_pymc_inputs,
)
from .model_inputs import (
    resolve_model_inputs as resolve_model_inputs,
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
    "resolve_model_inputs",
    "co2_model_input_names",
    "multisector_model_input_names",
    "SiteOptions",
    "resolve_flux_sources",
    "run_co2",
    "sample_rhime_model",
    "standard_model_input_names",
]
