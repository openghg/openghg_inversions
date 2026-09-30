"""Compatibility imports; implementation lives in :mod:`openghg_inversions.recipes.co2.co2_o2_runner`."""

from openghg_inversions.recipes.co2.co2_o2_runner import (
    _CO2_O2_VARIABLE_ROLES as _CO2_O2_VARIABLE_ROLES,
    _annotate_co2_o2_trace as _annotate_co2_o2_trace,
    _annotate_linked_fixed_ou_trace as _annotate_linked_fixed_ou_trace,
    _co2_o2_metadata as _co2_o2_metadata,
    _materialize_co2_o2_pymc_inputs as _materialize_co2_o2_pymc_inputs,
    _validate_independent_error_labels as _validate_independent_error_labels,
    _validate_independent_error_values as _validate_independent_error_values,
    run_rhime_co2_o2_from_prepared_inputs as run_rhime_co2_o2_from_prepared_inputs,
)
