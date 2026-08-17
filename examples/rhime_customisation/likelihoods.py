"""Likelihood selected by both RHIME customization examples.

The public absolute-sigma Gaussian builder is re-exported here so a consumer
project has one plainly named module for its scientific customization. It
treats inferred sigma as an absolute observation-aligned error contribution,
rather than scaling it by the modeled pollution event.

``likelihood_builder`` is the exported runner seam. Importing this module does
not build a model, retrieve data, sample, or write outputs.
"""

from openghg_inversions.rhime import (
    RhimeLikelihoodBuilder,
    build_absolute_sigma_gaussian_likelihood,
)

likelihood_builder: RhimeLikelihoodBuilder = build_absolute_sigma_gaussian_likelihood

__all__ = ["build_absolute_sigma_gaussian_likelihood", "likelihood_builder"]
