"""Compatibility imports; implementation lives in :mod:`openghg_inversions.recipes.co2.co2_cached_sigma_runner`."""

from openghg_inversions.recipes.co2.co2_cached_sigma_runner import (
    _CO2_CACHED_SIGMA_INPUT_NAMES as _CO2_CACHED_SIGMA_INPUT_NAMES,
    _annotate_cached_co2_trace as _annotate_cached_co2_trace,
    _append_joint_outputs as _append_joint_outputs,
    _observation_coords as _observation_coords,
    _posterior_predictive_requested as _posterior_predictive_requested,
    _predictive_seed as _predictive_seed,
    _sampler_for_cached_graph as _sampler_for_cached_graph,
    co2_cached_sigma_input_names as co2_cached_sigma_input_names,
    run_rhime_co2_cached_sigma as run_rhime_co2_cached_sigma,
)

__all__ = ['co2_cached_sigma_input_names', 'run_rhime_co2_cached_sigma']
