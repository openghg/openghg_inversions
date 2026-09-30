"""Compatibility imports; implementation lives in :mod:`openghg_inversions.recipes._model_building`."""

from openghg_inversions.recipes._model_building import (
    ForwardModelTerms as ForwardModelTerms,
    _add_builtin_likelihood as _add_builtin_likelihood,
    _call_custom_likelihood as _call_custom_likelihood,
    _resolve_site_additive_sigma_prior as _resolve_site_additive_sigma_prior,
    add_configured_additive_sigma_likelihood as add_configured_additive_sigma_likelihood,
    add_configured_fixed_error_likelihood as add_configured_fixed_error_likelihood,
    add_configured_pollution_event_likelihood as add_configured_pollution_event_likelihood,
    add_rhime_likelihood as add_rhime_likelihood,
    builtin_model_build_result as builtin_model_build_result,
    prepare_additive_sigma_inputs as prepare_additive_sigma_inputs,
    validated_custom_model_build as validated_custom_model_build,
)
