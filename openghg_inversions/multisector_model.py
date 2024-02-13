from dataclasses import dataclass
from functools import reduce
from typing import cast, Optional, Union

import arviz as az
import numpy as np
import pymc as pm
from pymc.distributions import continuous
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable
import xarray as xr

from openghg_inversions.hbmcmc.inversionsetup import offset_matrix

# type alias for prior args
PriorArgs = dict[str, Union[str, float]]


def parse_prior(name: str, prior_params: PriorArgs, **kwargs) -> TensorVariable:
    """
    Parses all PyMC continuous distributions:
    https://docs.pymc.io/api/distributions/continuous.html

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

    Note: calling `parse_prior` inside a `pm.Model` context (i.e. after `with pm.Model()`)
    has an important side-effect of registering the random variable with the model. Typically,
    `parse_prior` will be used in a `pm.Model` context, although it could be used to construct
    random variables for other purposes.
    """
    # create dict to lookup continuous PyMC distributions by name, ignoring case
    pdf_dict = {cd.lower(): cd for cd in continuous.__all__}

    params = prior_params.copy()
    pdf = str(params.pop("pdf")).lower()  # str is just for typing...
    try:
        dist = getattr(continuous, pdf_dict[pdf])
    except AttributeError:
        raise ValueError(
            f"The distribution '{pdf}' doesn't appear to be a continuous distribution defined by PyMC."
        )
    return dist(name, **params, **kwargs)


# Data classes for inputs
#
# Note: data classes can be used like dictionaries with pre-specified types
# and default arguments. They're a good replacement for dictionaries that always
# contain an "expected" set of keys-value pairs.
#
# There are several data classes here, with variable grouped (loosely) by the stage that they
# will be used in building the model. This is to facilitate breaking the model building code
# into smaller units, if desired in the future.
@dataclass
class FluxInputs:
    """Dataclass to hold info for adding a flux sector to the inversion.

    If multiple sectors are used in an inversion, the `label` variable should be
    used to distinguish these cases.

    If `xprior` is None, then the the sensitivity from this sector is added as a constant
    (rather than as a parameter to be inferred).
    """

    Hx: np.ndarray
    label: Optional[str] = None
    xprior: Optional[PriorArgs] = None


@dataclass
class UnobservedInputs:
    """Dataclass containing unobserved priors info.

    NOTE: xprior and Hx should be in here (e.g. as `FluxInputs`), but it is
    simpler to build the model if they are separate.
    """

    Hbc: np.ndarray
    sigma_freq_index: np.ndarray
    bcprior: PriorArgs #= {"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.1, "lower": 0.0}
    sigprior: PriorArgs #= {"pdf": "uniform", "lower": 0.1, "upper": 3.0}
    offsetprior: PriorArgs #= {"pdf": "normal", "mu": 0, "sigma": 1}


@dataclass
class ObservedInputs:
    Y: np.ndarray
    error: np.ndarray


# Functions for building model
def make_rhime_model(
    unobserved_inputs: UnobservedInputs,
    flux_inputs: Union[FluxInputs, list[FluxInputs]],
    observed_inputs: ObservedInputs,
    min_model_error: float,
    siteindicator: np.ndarray,
    add_offset: bool = False,
    sigma_per_site: bool = True,
) -> pm.Model:
    # Step 1: add unobserved vars
    nbc = unobserved_inputs.Hbc.shape[0]

    # convert siteindicator into a site indexer
    if sigma_per_site:
        sites = siteindicator.astype(int)
        nsites = np.amax(sites) + 1
    else:
        sites = np.zeros_like(siteindicator).astype(int)
        nsites = 1
    nsigmas = np.amax(unobserved_inputs.sigma_freq_index) + 1

    if add_offset:
        B = offset_matrix(siteindicator)

    with pm.Model() as model:
        xbc = parse_prior("xbc", unobserved_inputs.bcprior, shape=nbc)
        mu_bc = pm.Deterministic("mu_bc", pt.dot(unobserved_inputs.Hbc.T, xbc))

        sig = parse_prior("sig", unobserved_inputs.sigprior, shape=(nsites, nsigmas))
        pollution_event_scaling = sig[sites, unobserved_inputs.sigma_freq_index]

        if add_offset:
            offset = parse_prior("offset", unobserved_inputs.offsetprior, shape=nsites - 1)
            offset_vec = pt.concatenate((np.array([0]), offset), axis=0)
            mu_offset = pm.Deterministic("mu_offset", pt.dot(B, offset_vec))

    # Step 2: add flux vars
    if not isinstance(flux_inputs, list):
        flux_inputs = [flux_inputs]

    pollution_events: list[TensorVariable] = []  # list of mu's
    with model:
        # for each FluxInputs object, add a prior var if xprior is not None
        # otherwise, x is constant.
        for fi in flux_inputs:
            nx = fi.Hx.shape[0]

            if fi.label is None:
                suffix = "_"
            else:
                suffix = "_" + fi.label

            if fi.xprior is None:
                x = pt.ones(nx)
            else:
                x = parse_prior("x" + suffix, fi.xprior, shape=nx)

            pollution_events.append(pm.Deterministic("mu" + suffix, pt.dot(fi.Hx.T, x)))

    # Step 3: create mu
    with model:
        pollution_events_total = reduce(lambda x, y: x + y, pollution_events)

        if add_offset:
            mu = pm.Deterministic("mu", pollution_events_total + mu_bc + mu_offset)
        else:
            mu = pm.Deterministic("mu", pollution_events_total + mu_bc)

    # Step 4: create epsilon
    with model:
        abs_pollution_events = [pt.abs(x) for x in pollution_events]  # type: ignore
        abs_pollution_events_total = reduce(lambda x, y: x + y, abs_pollution_events)  # type: ignore
        model_error = abs_pollution_events_total * pollution_event_scaling
        epsilon = pt.sqrt(observed_inputs.error**2 + min_model_error**2 + model_error**2)

    # Step 5: add likelihood
    with model:
        ny = len(observed_inputs.Y)
        pm.Normal("y", mu, epsilon, observed=observed_inputs.Y, shape=ny)

    return model


def sample_model(model: pm.Model, nit: int, tune: int, nchain: int, verbose: bool = True) -> az.InferenceData:
    """Sample posterior.

    NOTE: I'm using NUTS on all variables for simplicity -- BM
    """
    with model:
        idata = pm.sample(nit, tune=tune, chains=nchain, progressbar=verbose)

    return idata


def convergence_check(trace: az.InferenceData) -> str:
    """Print convergence message and return "Failed" or "Passed"."""
    x_vars = [dv for dv in trace.posterior.data_vars if str(dv).startswith("x_")]
    gelrubs = [az.rhat(trace.posterior[dv]).max() for dv in x_vars]
    gelrub = max(gelrubs)
    if gelrub > 1.05:
        print("Failed Gelman-Rubin at 1.05")
        return "Failed"
    else:
        print(f"Passed Gelman-Rubin at 1.05. Max over x_ variables: {gelrub}")
        return "Passed"


def prepare_rhime_outs(trace: az.InferenceData, burn: int) -> tuple:
    """Convert InferenceData to outputs given by inferpymc_postprocess_outs.

    NOTE: I had to transpose some of the outputs to get the dimension to line up.
    It would be easier to use xarray's named dimensions for this.
    """
    if "posterior" not in trace.groups():
        raise ValueError("`trace` must have `posterior` group.")

    posterior = cast(xr.Dataset, trace.posterior)  # type: ignore
    posterior = posterior.isel(chain=0, draw=slice(burn, None), drop=False)



    outs = {str(dv): posterior[dv] for dv in posterior.data_vars if str(dv).startswith("x_")}
    bcouts = posterior["xbc"]
    sigouts = posterior["sig"]

    if "offset" in posterior.data_vars:
        offset_outs = posterior["offset"].values.T
        YBCtrace = posterior["mu_bc"].values.T + posterior["mu_offset"].values.T
        OFFtrace = posterior["mu_offset"].values.T
    else:
        YBCtrace = posterior["mu_bc"].values.T
        offset_outs = np.zeros_like(next(iter(outs.values())))
        OFFtrace = YBCtrace * 0

    # Anita and Will agree that we should actually report `posterior["y"]`, but
    # I'm leaving this the same as the current code -- BM
    Ytrace = posterior["mu"].values.T

    return outs, bcouts, sigouts, offset_outs, Ytrace, YBCtrace, OFFtrace


def experimental_inferpymc(
    Hx,
    Hbc,
    Y,
    error,
    siteindicator,
    sigma_freq_index,
    xprior={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
    bcprior={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
    sigprior={"pdf": "uniform", "lower": 0.1, "upper": 3.0},
    nit=2e5,
    burn=1e5,
    tune=1e5,
    nchain=2,
    sigma_per_site=True,
    offsetprior={"pdf": "normal", "mu": 0, "sigma": 1},
    add_offset=False,
    verbose=False,
    min_error=0.0,
    **kwargs,
):

    # check if xprior is a dict of dicts
    if any(isinstance(vals, dict) for vals in xprior.values()):
        # raise error if Hx is not a dict with the same keys as xprior
        if not isinstance(Hx, dict) or (list(xprior.keys()) != list(Hx.keys())):
            raise ValueError(
                "If `xprior` is a dictionary of prior parameters, then `Hx` must be a dict with the same keys."
            )

        flux_inputs = []

        for k in xprior.keys():
            flux_inputs.append(FluxInputs(label=k, Hx=Hx[k], xprior=xprior[k]))
    else:
        flux_inputs = [FluxInputs(label=None, Hx=Hx, xprior=xprior)]

    unobserved_inputs = UnobservedInputs(
        Hbc=Hbc,
        sigma_freq_index=sigma_freq_index,
        bcprior=bcprior,
        sigprior=sigprior,
        offsetprior=offsetprior,
    )
    observed_inputs = ObservedInputs(Y=Y, error=error)

    model = make_rhime_model(
        unobserved_inputs=unobserved_inputs,
        flux_inputs=flux_inputs,
        observed_inputs=observed_inputs,
        min_model_error=min_error,
        siteindicator=siteindicator,
        add_offset=add_offset,
        sigma_per_site=sigma_per_site,
    )

    trace = sample_model(model, nit=int(nit), tune=int(tune), nchain=nchain, verbose=verbose)

    convergence = convergence_check(trace)

    outs, *outputs = prepare_rhime_outs(trace, burn=int(burn))

    # if only one xprior specified, return array instead of dict
    if len(flux_inputs) == 1:
        outs = next(iter(outs.values()))

    return outs, *outputs, convergence, "NUTS", "NUTS"
