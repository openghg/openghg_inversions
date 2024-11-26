import numpy as np
import pymc as pm

from openghg_inversions.models.components import (
    Flux,
    BoundaryConditions,
    Offset,
    Baseline,
    ForwardModel,
    RHIMELikelihood,
    Sigma,
)


def make_rhime_forward_component(
        Hx: np.ndarray,
        Hbc: np.ndarray | None = None,
        add_offset: bool = False,
        xprior: dict | None = None,
        bcprior: dict | None = None,
        offsetprior: dict | None = None,
        site_indicator: np.ndarray | None = None,
        reparameterise_log_normal: bool = False,
    ) -> ForwardModel:

        # set default prior params
        xprior = xprior or {"pdf": "normal", "mu": 1.0, "sigma": 1.0}
        bcprior = bcprior or {"pdf": "normal", "mu": 1.0, "sigma": 1.0}
        offsetprior = offsetprior or {"pdf": "normal", "mu": 0.0, "sigma": 1.0}

        if reparameterise_log_normal:
            for prior in [xprior, bcprior, offsetprior]:
                if prior["pdf"] == "lognormal":
                    prior["reparameterise"] = True

        # create forward model components
        flux = Flux(name="flux", h_matrix=Hx.T, prior_args=xprior)

        baseline_components = {}
        if Hbc is not None:
            baseline_components["bc"] = BoundaryConditions(name="bc", h_matrix=Hbc.T, prior_args=bcprior)

        if add_offset:
            if site_indicator is None:
                raise ValueError("Need `site_indicator` to add Offset.")
            baseline_components["offset"] = Offset(site_indicator=site_indicator, prior_args=offsetprior)

        baseline = Baseline(**baseline_components)

        return ForwardModel(flux=flux, baseline=baseline)


def rhime_model(
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

    forward_model = make_rhime_forward_component(
        Hx=Hx,
        Hbc=Hbc,
        xprior=xprior,
        bcprior=bcprior,
        offsetprior=offsetprior,
        add_offset=add_offset,
        site_indicator=siteindicator,
        reparameterise_log_normal=reparameterise_log_normal,
    )

    sigma = Sigma(sigma_prior=sigprior, site_indicator=siteindicator, sigma_freq=sigma_freq, y_time=y_time, sigma_per_site=sigma_per_site)

    likelihood = RHIMELikelihood(
        y_obs=Y,
        error=error,
        sigma=sigma,
        min_error=min_error,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
    )

    with pm.Model() as model:
        forward_model.build()

        likelihood.build(forward=forward_model)

    return model
