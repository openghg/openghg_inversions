"""Compatibility aliases for :mod:`openghg_inversions.model_components.cached_sigma`.

Implementation and function globals live in the canonical component module.
"""

from openghg_inversions.model_components.cached_sigma import (
    FixedOuLikelihoodEvaluation as FixedOuLikelihoodEvaluation,
    FixedOuLowRank as FixedOuLowRank,
    FloatArray as FloatArray,
    _vector as _vector,
    _matrix as _matrix,
    MarginalQuadraticCache as MarginalQuadraticCache,
    FixedOuCachedSigmaTarget as FixedOuCachedSigmaTarget,
)
