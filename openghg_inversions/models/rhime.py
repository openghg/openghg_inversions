from functools import reduce

import numpy as np
import pymc as pm

from .components import LinearForwardComponent, Offset, RHIMELikelihood


def build_rhime_model(
    Hx: np.ndarray,
    Y: np.ndarray,
    error: np.ndarray,
    siteindicator: np.ndarray,
    Hbc: np.ndarray | None = None,
    xprior: dict = {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
    bcprior: dict = {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
    sigprior: dict = {"pdf": "uniform", "lower": 0.1, "upper": 3.0},
    sigma_freq: str | None = None,
    y_time: np.ndarray | None = None,
    sigma_per_site: bool = True,
    offsetprior: dict = {"pdf": "normal", "mu": 0, "sigma": 1},
    add_offset: bool = False,
    min_error: np.ndarray | float = 0.0,
    reparameterise_log_normal: bool = False,
    pollution_events_from_obs: bool = False,
    no_model_error: bool = False,
) -> pm.Model:

    if reparameterise_log_normal:
        for prior in [xprior, bcprior, sigprior, offsetprior]:
            if prior["pdf"] == "lognormal":
                prior["reparameterise"] = True

    # add forward model components
    forward_model_components = []

    forward_model_components.append(LinearForwardComponent(name="flux", h_matrix=Hx.T, prior_args=xprior))

    if Hbc is not None:
        forward_model_components.append(LinearForwardComponent(name="bc", h_matrix=Hbc.T, prior_args=bcprior))

    if add_offset:
        forward_model_components.append(Offset(site_indicator=siteindicator, prior_args=offsetprior))

    # make likelihood
    likelihood = RHIMELikelihood(
        y_obs=Y,
        error=error,
        sigma_prior=sigprior,
        site_indicator=siteindicator,
        min_error=min_error,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
        sigma_per_site=sigma_per_site,
        sigma_freq=sigma_freq,
        y_time=y_time,
    )

    with pm.Model() as model:
        for component in forward_model_components:
            component.build()

        mu_total = reduce(lambda x, y: x + y, [component.model.mu for component in forward_model_components])
        pm.Deterministic("mu", mu_total)

        likelihood.build()

    return model
