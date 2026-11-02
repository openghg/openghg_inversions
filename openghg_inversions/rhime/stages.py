"""Standard numerical handoffs and shared posterior diagnostics.

Multisector handoffs live in ``multisector_stages``; CO2 in ``co2.handoffs``.
"""

from ._standard_stages import prepare, sample, postprocess, prior_predictive, load_preparation, load_samples, save_preparation, save_samples
from ._stage_checks import (
    CHECK_SCHEMA_VERSION,
    CONVERGENCE_CHECK_NAME,
    PREPARATION_CHECK_NAME,
    diagnose_rhime_stage,
)

__all__ = [
    "prepare", "sample", "postprocess", "prior_predictive",
    "save_preparation", "load_preparation", "save_samples", "load_samples",
    "CHECK_SCHEMA_VERSION", "CONVERGENCE_CHECK_NAME", "PREPARATION_CHECK_NAME",
    "diagnose_rhime_stage",
]
