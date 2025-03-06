"""Functions for performing MCMC inversion.
PyMC library used for Bayesian modelling.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions import continuous
from pytensor.tensor import TensorVariable


# type alias for prior args
PriorArgs = dict[str, str | float]


def lognormal_mu_sigma(mean: float, stdev: float) -> tuple[float, float]:
    """Return the pymc `mu` and `sigma` parameters that give a log normal distribution
    with the given mean and stdev.

    Args:
        mean: desired mean of log normal
        stdev: desired standard deviation of log normal

    Returns:
        tuple (mu, sigma), where `pymc.LogNormal(mu, sigma)` has the given mean and stdev.

    Formulas for log normal mean and variance:

    mean = exp(mu + 0.5 * sigma ** 2)
    stdev ** 2 = var = exp(2*mu + sigma ** 2) * (exp(sigma ** 2) - 1)

    This gives linear equations for `mu` and `sigma ** 2`:

    mu + 0.5 * sigma ** 2 = log(mean)
    sigma ** 2 = log(1 + (stdev / mean)**2)

    So

    mu = log(mean) - 0.5 * log(1 + (stdev/mean)**2)
    sigma = sqrt(log(1 + (stdev / mean)**2))
    """
    var = np.log(1 + (stdev / mean) ** 2)
    mu = np.log(mean) - 0.5 * var
    sigma = np.sqrt(var)
    return mu, sigma


def update_log_normal_prior(prior_params):
    if "stdev" in prior_params:
        stdev = float(prior_params["stdev"])
        mean = float(prior_params.get("mean", 1.0))

        mu, sigma = lognormal_mu_sigma(mean, stdev)
        prior_params["mu"] = mu
        prior_params["sigma"] = sigma

        del prior_params["stdev"]
        if "mean" in prior_params:
            del prior_params["mean"]


def parse_prior(name: str, prior_params: PriorArgs, **kwargs) -> TensorVariable:
    """Parses all PyMC continuous distributions:
    https://docs.pymc.io/api/distributions/continuous.html.

    Args:
        name:
          name of variable in the pymc model
        prior_params:
          dict of parameters for the distribution, including 'pdf' for the distribution to use.
          The value of `prior_params["pdf"]` must match the name of a PyMC continuous
          distribution: https://docs.pymc.io/api/distributions/continuous.html
        **kwargs: for instance, `shape` or `dims`
    Returns:
        continuous PyMC distribution

    For example:
    ```
    params = {"pdf": "uniform", "lower": 0.0, "upper": 1.0}
    parse_prior("x", params, shape=(20, 20))
    ```
    will create a 20 x 20 array of uniform random variables.
    Alternatively,
    ```
    params = {"pdf": "uniform", "lower": 0.0, "upper": 1.0}
    parse_prior("x", params, dims="nmeasure"))
    ```
    will create an array of uniform random variables with the same shape
    as the dimension coordinate `nmeasure`. This can be used if `pm.Model`
    is provided with coordinates.

    Note: `parse_prior` must be called inside a `pm.Model` context (i.e. after `with pm.Model()`)
    has an important side-effect of registering the random variable with the model.
    """
    # create dict to lookup continuous PyMC distributions by name, ignoring case
    pdf_dict = {cd.lower(): cd for cd in continuous.__all__}

    params = prior_params.copy()
    pdf = str(params.pop("pdf")).lower()  # str is just for typing...

    # special processing for lognormals
    if pdf == "lognormal":
        update_log_normal_prior(params)

        if params.get("reparameterise", False):
            temp = pm.Normal(f"{name}0", 0, 1, **kwargs)
            return pm.Deterministic(name, pt.exp(params["mu"] + params["sigma"] * temp), **kwargs)

    try:
        dist = getattr(continuous, pdf_dict[pdf])
    except AttributeError:
        raise ValueError(
            f"The distribution '{pdf}' doesn't appear to be a continuous distribution defined by PyMC."
        )

    return dist(name, **params, **kwargs)
