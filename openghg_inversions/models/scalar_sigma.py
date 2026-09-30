"""Compatibility aliases for :mod:`openghg_inversions.model_components.scalar_sigma`.

Implementation and function globals live in the canonical component module.
"""

from openghg_inversions.model_components.scalar_sigma import (
    to_dense as to_dense,
    validate_covariance_coordinates as validate_covariance_coordinates,
    add_model_data as add_model_data,
    parse_prior as parse_prior,
    positive_prior_args as positive_prior_args,
    AggregationError as AggregationError,
    decode_cf_multiindexes as decode_cf_multiindexes,
    encode_cf_multiindexes as encode_cf_multiindexes,
    SCALAR_SIGMA_CACHE_SCHEMA as SCALAR_SIGMA_CACHE_SCHEMA,
    SCALAR_SIGMA_CACHE_VERSION as SCALAR_SIGMA_CACHE_VERSION,
    SCALAR_SIGMA_MODE_DIM as SCALAR_SIGMA_MODE_DIM,
    EIGENVECTORS as EIGENVECTORS,
    EIGENVALUES as EIGENVALUES,
    ScalarSigmaAggregationMode as ScalarSigmaAggregationMode,
    _align_observation_array as _align_observation_array,
    _materialize_together as _materialize_together,
    _finite_values as _finite_values,
    _scale_relative_tolerance as _scale_relative_tolerance,
    _base_covariance_sha256 as _base_covariance_sha256,
    scalar_sigma_base_covariance as scalar_sigma_base_covariance,
    ScalarSigmaEigenbasis as ScalarSigmaEigenbasis,
    prepare_scalar_sigma_eigenbasis as prepare_scalar_sigma_eigenbasis,
    save_scalar_sigma_eigenbasis as save_scalar_sigma_eigenbasis,
    load_scalar_sigma_eigenbasis as load_scalar_sigma_eigenbasis,
    add_scalar_sigma_eigen_likelihood as add_scalar_sigma_eigen_likelihood,
    __all__ as __all__,
)
