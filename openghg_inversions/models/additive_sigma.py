"""Compatibility aliases for :mod:`openghg_inversions.model_components.additive_sigma`.

Implementation and function globals live in the canonical component module.
"""

from openghg_inversions.model_components.additive_sigma import (
    add_model_data as add_model_data,
    add_sigma_component as add_sigma_component,
    PriorArgs as PriorArgs,
    add_aggregation_error_data as add_aggregation_error_data,
    add_gaussian_observation_likelihood as add_gaussian_observation_likelihood,
    AggregationError as AggregationError,
    validate_observation_error_arrays as validate_observation_error_arrays,
    SigmaAlignment as SigmaAlignment,
    FIXED_MODEL_MISMATCH as FIXED_MODEL_MISMATCH,
    DEFAULT_ADDITIVE_SIGMA_PRIOR as DEFAULT_ADDITIVE_SIGMA_PRIOR,
    add_additive_sigma_likelihood as add_additive_sigma_likelihood,
    __all__ as __all__,
)
