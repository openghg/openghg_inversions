"""Create reusable scalar or array-valued PyMC prior variables.

``parse_prior`` registers continuous distributions on the active model;
``lognormal_mu_sigma`` converts requested moments, including arrays. Optional
lognormal reparameterization exposes ``<name>_latent`` and keeps ``<name>`` as
the user-facing deterministic variable.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions import continuous
from pytensor.tensor.variable import TensorVariable

PriorArgs: TypeAlias = dict[str, Any]


def lognormal_mu_sigma(
    mean: float | np.ndarray,
    stdev: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert lognormal mean and stdev into PyMC's ``mu`` and ``sigma``.

    Args:
        mean: Requested scalar or array-valued mean of the lognormal
            distribution.
        stdev: Requested scalar or array-valued standard deviation of the
            lognormal distribution.

    Returns:
        A ``(mu, sigma)`` tuple suitable for ``pm.Lognormal``.
    """
    mean_array = np.asarray(mean)
    stdev_array = np.asarray(stdev)
    var = np.log(1 + (stdev_array / mean_array) ** 2)
    mu = np.log(mean_array) - 0.5 * var
    sigma = np.sqrt(var)
    if mu.ndim == 0 and sigma.ndim == 0:
        return float(mu), float(sigma)
    return mu, sigma


def _update_log_normal_prior(prior_params: PriorArgs) -> None:
    """Rewrite requested lognormal moments to PyMC parameters in place.

    Args:
        prior_params: Mutable mapping. When ``stdev`` is present, it and
            optional ``mean`` are replaced by ``mu`` and ``sigma``.
    """
    if "stdev" not in prior_params:
        return

    stdev = prior_params["stdev"]
    mean = prior_params.get("mean", 1.0)
    mu, sigma = lognormal_mu_sigma(mean, stdev)
    prior_params["mu"] = mu
    prior_params["sigma"] = sigma
    del prior_params["stdev"]
    if "mean" in prior_params:
        del prior_params["mean"]


def parse_prior(name: str, prior_params: PriorArgs, **kwargs) -> TensorVariable:
    """Create a continuous PyMC prior from a prior-parameter dictionary.

    Args:
        name: Name of the user-facing PyMC variable to create.
        prior_params: Prior specification including ``pdf`` and any distribution
            parameters accepted by the chosen PyMC distribution.
        **kwargs: Additional keyword arguments forwarded to the created PyMC
            variable, such as ``dims``.

    Returns:
        The created PyMC random variable or deterministic transform.

    Raises:
        ValueError: If ``prior_params["pdf"]`` does not name a supported PyMC
            continuous distribution.

    This helper must be called inside an active ``pm.Model`` context because it
    registers the created variable with the current model.
    """
    pdf_dict = {cd.lower(): cd for cd in continuous.__all__}

    params = prior_params.copy()
    pdf = str(params.pop("pdf")).lower()

    if pdf == "lognormal":
        _update_log_normal_prior(params)

        if params.get("reparameterise", False):
            latent = pm.Normal(f"{name}_latent", 0, 1, **kwargs)
            mu = pm.floatX(params["mu"])
            sigma = pm.floatX(params["sigma"])
            return pm.Deterministic(name, pt.exp(mu + sigma * latent), **kwargs)

    params.pop("reparameterise", None)

    try:
        dist = getattr(continuous, pdf_dict[pdf])
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"The distribution '{pdf}' doesn't appear to be a continuous distribution defined by PyMC."
        ) from exc

    return dist(name, **params, **kwargs)
