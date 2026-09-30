"""Compatibility imports; implementation lives in :mod:`openghg_inversions.recipes.co2.co2_runner`."""

from openghg_inversions.recipes.co2.co2_runner import (
    _CO2_SCIENTIFIC_INPUT_NAMES as _CO2_SCIENTIFIC_INPUT_NAMES,
    _annotate_co2_trace as _annotate_co2_trace,
    _resolve_co2_likelihood_kwargs as _resolve_co2_likelihood_kwargs,
    _state_activity_from_inputs as _state_activity_from_inputs,
    co2_model_input_names as co2_model_input_names,
    prepare_co2_scalar_sigma_eigenbasis as prepare_co2_scalar_sigma_eigenbasis,
    run_rhime_co2 as run_rhime_co2,
)

__all__ = ['co2_model_input_names', 'prepare_co2_scalar_sigma_eigenbasis', 'run_rhime_co2']
