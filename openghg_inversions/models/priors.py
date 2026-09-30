"""Compatibility aliases for :mod:`openghg_inversions.model_components.priors`.

Implementation and function globals live in the canonical component module.
"""

from openghg_inversions.model_components.priors import (
    PriorArgs as PriorArgs,
    _POSITIVE_PRIOR_FAMILIES as _POSITIVE_PRIOR_FAMILIES,
    lognormal_mu_sigma as lognormal_mu_sigma,
    _update_log_normal_prior as _update_log_normal_prior,
    positive_prior_args as positive_prior_args,
    parse_prior as parse_prior,
)
