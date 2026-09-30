"""Compatibility aliases for :mod:`openghg_inversions.model_components.fixed_ou`.

Implementation and function globals live in the canonical component module.
"""

from openghg_inversions.model_components.fixed_ou import (
    expand_mapping as expand_mapping,
    add_model_data as add_model_data,
    parse_prior as parse_prior,
    positive_prior_args as positive_prior_args,
    AggregationError as AggregationError,
    aggregation_error_as_low_rank as aggregation_error_as_low_rank,
    validate_observation_error_arrays as validate_observation_error_arrays,
    SigmaAlignment as SigmaAlignment,
    FloatArray as FloatArray,
    IntArray as IntArray,
    _HOURS_PER_NANOSECOND as _HOURS_PER_NANOSECOND,
    FixedOuSiteBlock as FixedOuSiteBlock,
    FixedOuLikelihoodEvaluation as FixedOuLikelihoodEvaluation,
    FixedOuCovarianceSolve as FixedOuCovarianceSolve,
    _rejection_gradient_amplitude as _rejection_gradient_amplitude,
    _FixedOuLogpOp as _FixedOuLogpOp,
    FixedOuLowRank as FixedOuLowRank,
    prepare_fixed_ou_low_rank as prepare_fixed_ou_low_rank,
    _time_hours as _time_hours,
    _vector as _vector,
    _matrix as _matrix,
    _group_index as _group_index,
    _site_labels as _site_labels,
    _tau_values as _tau_values,
    _site_values as _site_values,
    add_fixed_ou_gaussian_likelihood as add_fixed_ou_gaussian_likelihood,
)
