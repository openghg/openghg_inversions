"""Reusable prior helpers for PyMC model construction."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions import continuous
from pytensor.tensor import TensorVariable

PriorArgs: TypeAlias = dict[str, str | float | bool]


def lognormal_mu_sigma(mean: float, stdev: float) -> tuple[float, float]:
    """Return lognormal ``mu`` and ``sigma`` for the requested mean and stdev."""
    var = np.log(1 + (stdev / mean) ** 2)
    mu = np.log(mean) - 0.5 * var
    sigma = np.sqrt(var)
    return mu, sigma


def _update_log_normal_prior(prior_params: PriorArgs) -> None:
    if "stdev" not in prior_params:
        return

    stdev = float(prior_params["stdev"])
    mean = float(prior_params.get("mean", 1.0))
    mu, sigma = lognormal_mu_sigma(mean, stdev)
    prior_params["mu"] = mu
    prior_params["sigma"] = sigma
    del prior_params["stdev"]
    if "mean" in prior_params:
        del prior_params["mean"]


def parse_prior(name: str, prior_params: PriorArgs, **kwargs) -> TensorVariable:
    """Create a PyMC continuous prior from a prior parameter dictionary."""
    pdf_dict = {cd.lower(): cd for cd in continuous.__all__}

    params = prior_params.copy()
    pdf = str(params.pop("pdf")).lower()

    if pdf == "lognormal":
        _update_log_normal_prior(params)

        if params.get("reparameterise", False):
            latent = pm.Normal(f"{name}_latent", 0, 1, **kwargs)
            return pm.Deterministic(name, pt.exp(params["mu"] + params["sigma"] * latent), **kwargs)

    params.pop("reparameterise", None)

    try:
        dist = getattr(continuous, pdf_dict[pdf])
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"The distribution '{pdf}' doesn't appear to be a continuous distribution defined by PyMC."
        ) from exc

    return dist(name, **params, **kwargs)
