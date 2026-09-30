"""Compatibility aliases for :mod:`openghg_inversions.model_components._gaussian_observation`.

Implementation and function globals live in the canonical component module.
"""

from openghg_inversions.model_components._gaussian_observation import (
    add_model_data as add_model_data,
    AGGREGATION_ERROR_COVARIANCE as AGGREGATION_ERROR_COVARIANCE,
    DIAGONAL_RESIDUAL_VARIANCE as DIAGONAL_RESIDUAL_VARIANCE,
    LOW_RANK_FACTOR as LOW_RANK_FACTOR,
    AggregationError as AggregationError,
    RegisteredAggregationError as RegisteredAggregationError,
    add_aggregation_error_data as add_aggregation_error_data,
    _low_rank_gaussian_logp as _low_rank_gaussian_logp,
    _low_rank_gaussian_random as _low_rank_gaussian_random,
    add_gaussian_observation_likelihood as add_gaussian_observation_likelihood,
)
