"""Classes and functions to making self-contained parts of the RHIME model."""
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor import TensorVariable


def make_offset(site_indicator: np.ndarray, prior_args: dict, name: str = "offset", output_dim: str = "nmeasure", drop_first: bool = False) -> TensorVariable:
    """Create an offset inside a PyMC model.

    Note: this *must* be called from inside a PyMC `model` context.

    Args:
        site_indicator: array with same length as obs, with integers to indicator which site
          an observation belongs to
        prior_args: dict of prior args for offset prior
        name: name for offset in PyMC model
        output_dim: name of dimension for output
        drop_first: if True, set first site's offset to zero

    Returns:
        TensorVariable containing offset vector (to add to modelled observations).
    """
    from .inversion_pymc import parse_prior  # TODO move parse_prior into this file?

    sites = np.unique(site_indicator)

    n_sites = len(sites) - 1 if drop_first else len(sites)

    matrix = pd.get_dummies(site_indicator, drop_first=drop_first, dtype=int).values
    offset_x = parse_prior(name + "0", prior_args, shape=n_sites)

    return pm.Deterministic(name, pt.dot(matrix, offset_x), dims=output_dim)
