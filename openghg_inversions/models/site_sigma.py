"""Compatibility aliases for :mod:`openghg_inversions.model_components.site_sigma`.

Implementation and function globals live in the canonical component module.
"""

from openghg_inversions.model_components.site_sigma import (
    expand_mapping as expand_mapping,
    add_aggregation_error_data as add_aggregation_error_data,
    add_gaussian_observation_likelihood as add_gaussian_observation_likelihood,
    add_model_data as add_model_data,
    add_coords as add_coords,
    parse_prior as parse_prior,
    positive_prior_args as positive_prior_args,
    AggregationError as AggregationError,
    validate_observation_error_arrays as validate_observation_error_arrays,
    SigmaAlignment as SigmaAlignment,
    SITE_SIGMA_DIM as SITE_SIGMA_DIM,
    SITE_SIGMA as SITE_SIGMA,
    SITE_SIGMA_INDEX as SITE_SIGMA_INDEX,
    SIGMA_OBSERVATION as SIGMA_OBSERVATION,
    add_site_sigma_gaussian_likelihood as add_site_sigma_gaussian_likelihood,
    __all__ as __all__,
)
