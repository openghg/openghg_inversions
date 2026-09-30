"""Compatibility imports; implementation lives in :mod:`openghg_inversions.recipes.multisector`."""

from openghg_inversions.recipes.multisector import (
    _BASELINE_INPUT_NAMES as _BASELINE_INPUT_NAMES,
    _MULTISECTOR_FLUX_INPUT_NAMES as _MULTISECTOR_FLUX_INPUT_NAMES,
    _OBSERVATION_INPUT_NAMES as _OBSERVATION_INPUT_NAMES,
    _SectorComponent as _SectorComponent,
    _prepare_multisector_flux_components as _prepare_multisector_flux_components,
    _require_component_inputs as _require_component_inputs,
    _validate_multisector_basis_layout as _validate_multisector_basis_layout,
    build_multisector_rhime_model as build_multisector_rhime_model,
    build_multisector_rhime_model_result as build_multisector_rhime_model_result,
    make_multisector_rhime_result as make_multisector_rhime_result,
    multisector_model_input_names as multisector_model_input_names,
    run_rhime_multisector as run_rhime_multisector,
)
