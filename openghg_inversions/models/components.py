"""Compatibility aliases for :mod:`openghg_inversions.model_components.components`.

Implementation and function globals live in the canonical component module.
"""

from openghg_inversions.model_components.components import (
    CorrelatedLognormalPrior as CorrelatedLognormalPrior,
    make_freq_indicator as make_freq_indicator,
    make_site_indicator as make_site_indicator,
    add_coords as add_coords,
    parse_prior as parse_prior,
    PreparedLinearSensitivity as PreparedLinearSensitivity,
    ResolvedStateActivity as ResolvedStateActivity,
    StateActivity as StateActivity,
    active_prior_args as active_prior_args,
    resolve_state_activity as resolve_state_activity,
    SigmaAlignment as SigmaAlignment,
    LinearComponentResult as LinearComponentResult,
    OffsetComponentResult as OffsetComponentResult,
    StateVectorResult as StateVectorResult,
    CorrelatedStateResult as CorrelatedStateResult,
    get_model_latent as get_model_latent,
    resolve_model_variable as resolve_model_variable,
    add_model_data as add_model_data,
    add_linear_component as add_linear_component,
    apply_linear_sensitivity as apply_linear_sensitivity,
    add_linked_linear_component as add_linked_linear_component,
    add_coherent_affine_component as add_coherent_affine_component,
    add_state_vector as add_state_vector,
    add_correlated_lognormal_state as add_correlated_lognormal_state,
    add_correlated_lognormal_state_with_activity as add_correlated_lognormal_state_with_activity,
    _add_prepared_correlated_lognormal_state_with_activity as _add_prepared_correlated_lognormal_state_with_activity,
    prepare_active_correlated_lognormal_prior as prepare_active_correlated_lognormal_prior,
    add_sigma_component as add_sigma_component,
    _add_offset_component_result as _add_offset_component_result,
    add_offset_component as add_offset_component,
)
