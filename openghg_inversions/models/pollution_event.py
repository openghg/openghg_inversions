"""Compatibility aliases for :mod:`openghg_inversions.model_components.pollution_event`.

Implementation and function globals live in the canonical component module.
"""

from openghg_inversions.model_components.pollution_event import (
    add_model_data as add_model_data,
    add_sigma_component as add_sigma_component,
    RegisteredAggregationError as RegisteredAggregationError,
    add_aggregation_error_data as add_aggregation_error_data,
    add_gaussian_observation_likelihood as add_gaussian_observation_likelihood,
    parse_prior as parse_prior,
    AggregationError as AggregationError,
    validate_observation_error_arrays as validate_observation_error_arrays,
    SigmaAlignment as SigmaAlignment,
    _PollutionEventErrorState as _PollutionEventErrorState,
    _build_pollution_event_error as _build_pollution_event_error,
    add_pollution_event_likelihood as add_pollution_event_likelihood,
    __all__ as __all__,
)
