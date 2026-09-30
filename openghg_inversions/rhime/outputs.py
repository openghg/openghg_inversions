"""Compatibility imports; implementation lives in :mod:`openghg_inversions.recipes.outputs`."""

from openghg_inversions.recipes.outputs import (
    RhimeResult as RhimeResult,
    _define_derived_output_filename as _define_derived_output_filename,
    _define_output_filename as _define_output_filename,
    _make_inversion_output as _make_inversion_output,
    _make_multisector_flux_diagnostics as _make_multisector_flux_diagnostics,
    _resolve_output_path as _resolve_output_path,
    _sampler_metadata as _sampler_metadata,
    _save_requested_trace as _save_requested_trace,
    _structured_metadata as _structured_metadata,
    annotate_likelihood_trace as annotate_likelihood_trace,
    make_multisector_rhime_outputs as make_multisector_rhime_outputs,
    make_standard_rhime_outputs as make_standard_rhime_outputs,
)
