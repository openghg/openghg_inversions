"""Public RHIME runners, specifications, builders, and orchestration stages.

Use ``run(..., model="standard")`` with an explicit model selection, or the
prepared-input entry points for complete runs. Copied runners may use
the supported resolve, acquire, observation-error, filter, basis, sensitivity, assembly,
alignment, materialization, build, sample, result, and output stages directly.
Alignment is pure; acquisition may access data, model materialization crosses
the eager backend boundary, sampling executes PyMC, and output stages may write
requested products.

Public names are imported lazily on first access, so importing this package
does not load every model family. ``__init__.pyi`` declares the same names for
type checkers and IDEs.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from importlib.util import find_spec as _find_spec
from typing import Any

# PyTensor precision must be selected before any RHIME submodule imports PyMC.
# Every submodule import runs this package initializer first.
from openghg_inversions._pymc_config import configure_pytensor as _configure_pytensor

_configure_pytensor()

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

# Public name -> module that defines it (relative to this package or absolute).
_EXPORTS: dict[str, str] = {
    "run": ".api",
    "run_standard": ".standard",
    "run_multisector": ".multisector",
    "run_nested": ".nested",
    "MergedData": "openghg_inversions.inversion_data",
    "PreparedInputs": "openghg_inversions.inversion_data",
    "assemble_inputs": ".preparation",
    "RhimeConfig": ".params",
    "DataConfig": ".params",
    "group_params": ".params",
    "read_rhime_ini": ".ini",
    "SectorSpec": ".specs",
    "AdditiveSigmaSettings": ".specs",
    "FixedErrorSettings": ".specs",
    "PollutionEventSettings": ".specs",
    "RhimeModelSpec": ".specs",
    "RhimeLikelihoodBuilder": ".builders",
    "NestedRhimePreparedInputs": ".nested",
    "NestedOutput": ".nested",
    "inner_data_config": ".nested",
    "materialize_nested_inputs": ".nested",
    "make_nested_inversion_outputs": ".nested",
    "RhimeModelBuildResult": ".builders",
    "RhimeOutputSpec": ".specs",
    "RhimeSampler": ".sampling",
    "CONVERGENCE_CHECK_NAME": ".stages",
    "PREPARATION_CHECK_NAME": ".stages",
    "align_inner_merged_to_outer_observations": ".nested",
    "build_multisector_model": ".multisector",
    "prepare_multisector": ".multisector",
    "materialize_multisector_inputs": ".multisector",
    "make_model_output": ".outputs",
    "postprocess_multisector_outputs": ".outputs",
    "save_inversion_output": ".outputs",
    "validate_multisector_output_resources": ".outputs",
    "build_co2_model": ".co2",
    "build_nested_rhime_model": ".nested",
    "build_sensitivities": ".preparation",
    "build_standard_model": ".standard",
    "materialize_standard_inputs": ".standard",
    "filter_observations": ".preparation",
    "prepare_observation_errors": ".preparation",
    "diagnose_rhime_stage": ".stages",
    "combine_nested_rhime_inputs": ".nested",
    "mask_outer_merged_for_inner_domain": ".nested",
    "nested_model_input_names": ".nested",
    "prepare_nested_rhime_inputs": ".nested",
    "annotate_sample_trace": ".outputs",
    "postprocess_standard_outputs": ".outputs",
    "validate_output_resources": ".outputs",
    "validate_standard_output_resources": ".outputs",
    "materialize_pymc_inputs": ".materialization",
    "resolve_model_inputs": ".model_inputs",
    "co2_model_input_names": ".co2",
    "multisector_model_input_names": ".multisector",
    "SiteOptions": "openghg_inversions.inversion_data",
    "resolve_flux_sources": ".params",
    "run_co2": ".co2",
    "sample_rhime_model": ".sampling",
    "standard_model_input_names": ".standard",
}


def __getattr__(name: str) -> Any:
    """Import a public RHIME name, or an unimported submodule, on first access.

    Args:
        name: Attribute requested from ``openghg_inversions.rhime``.

    Returns:
        The exported object, or the submodule named ``name``.

    Raises:
        AttributeError: If ``name`` is neither a public export nor a submodule.
    """
    module_name = _EXPORTS.get(name)
    if module_name is not None:
        value = getattr(_import_module(module_name, __name__), name)
    elif not name.startswith("_") and _find_spec(f"{__name__}.{name}") is not None:
        # Preserve ``rhime.<submodule>`` access that the eager initializer allowed.
        value = _import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the public RHIME API for interactive introspection."""
    return sorted(set(__all__) | set(globals()))
