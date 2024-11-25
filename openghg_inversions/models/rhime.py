from functools import reduce
from typing import Iterable

import numpy as np
import pymc as pm
from pytensor.tensor.variable import TensorVariable

from openghg_inversions.models.components import (
    HasOutput,
    LinearForwardComponent,
    ModelComponent,
    Offset,
    RHIMELikelihood,
)


def sum_outputs(components: Iterable[HasOutput]) -> TensorVariable:
    """Sum the output variables of a list of components."""
    return reduce(lambda x, y: x + y, (component.output for component in components))


class RHIMEForwardModel(ModelComponent):
    def __init__(
        self,
        Hx: np.ndarray,
        Hbc: np.ndarray | None = None,
        add_offset: bool = False,
        xprior: dict | None = None,
        bcprior: dict | None = None,
        offsetprior: dict | None = None,
        site_indicator: np.ndarray | None = None,
        reparameterise_log_normal: bool = False,
    ) -> None:
        super().__init__()

        self.name = "forward"

        # set default prior params
        xprior = xprior or {"pdf": "normal", "mu": 1.0, "sigma": 1.0}
        bcprior = bcprior or {"pdf": "normal", "mu": 1.0, "sigma": 1.0}
        offsetprior = offsetprior or {"pdf": "normal", "mu": 0.0, "sigma": 1.0}

        if reparameterise_log_normal:
            for prior in [xprior, bcprior, offsetprior]:
                if prior["pdf"] == "lognormal":
                    prior["reparameterise"] = True

        # add forward model components
        components = []

        components.append(LinearForwardComponent(name="flux", h_matrix=Hx.T, prior_args=xprior))

        if Hbc is not None:
            components.append(LinearForwardComponent(name="bc", h_matrix=Hbc.T, prior_args=bcprior))

        if add_offset:
            if site_indicator is None:
                raise ValueError("Need `site_indicator` to add Offset.")
            components.append(Offset(site_indicator=site_indicator, prior_args=offsetprior))

        self.components = components

    def build(self) -> None:
        self._model = pm.Model(name=self.name)

        with self.model:
            for component in self.components:
                component.build()

            non_flux_components = [component for component in self.components if "flux" not in component.name]
            if non_flux_components:
                pm.Deterministic("baseline", sum_outputs(non_flux_components))
                
            pm.Deterministic("mu", sum_outputs(self.components))


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

    forward_model = RHIMEForwardModel(
        Hx=Hx,
        Hbc=Hbc,
        xprior=xprior,
        bcprior=bcprior,
        offsetprior=offsetprior,
        add_offset=add_offset,
        site_indicator=siteindicator,
        reparameterise_log_normal=reparameterise_log_normal,
    )

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
        forward_model.build()

        mu_flux = forward_model["flux::mu"]
        mu = forward_model["mu"]
        baseline = forward_model.get("baseline")

        likelihood.build(mu=mu, mu_flux=mu_flux, baseline=baseline)

    return model
