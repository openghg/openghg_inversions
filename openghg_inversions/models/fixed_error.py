"""Compatibility aliases for :mod:`openghg_inversions.model_components.fixed_error`.

Implementation and function globals live in the canonical component module.
"""

from openghg_inversions.model_components.fixed_error import (
    add_aggregation_error_data as add_aggregation_error_data,
    add_gaussian_observation_likelihood as add_gaussian_observation_likelihood,
    add_model_data as add_model_data,
    AggregationError as AggregationError,
    validate_observation_error_arrays as validate_observation_error_arrays,
    add_fixed_error_likelihood as add_fixed_error_likelihood,
    __all__ as __all__,
)
