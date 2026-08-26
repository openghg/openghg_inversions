"""Compatibility imports for the unchanged ``run_hbmcmc`` entry point.

Likelihood graph construction is owned by :mod:`openghg_inversions.models`.
This module creates no PyMC variables; it only preserves the legacy runner's
import path.
"""

from openghg_inversions.models.additive_sigma import additive_sigma_likelihood_builder


__all__ = ["additive_sigma_likelihood_builder"]
