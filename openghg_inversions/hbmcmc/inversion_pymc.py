"""Functions for performing MCMC inversion.
PyMC library used for Bayesian modelling.
"""

import re
import getpass
from pathlib import Path

import numpy as np

# import pytensor before pymc so we can set config values
import pytensor
pytensor.config.floatX = "float32"
pytensor.config.warn_float64 = "warn"

import pymc as pm
import pandas as pd
import xarray as xr
import pytensor.tensor as pt
import arviz as az
from scipy import stats
from pymc.distributions import continuous
from pytensor.tensor import TensorVariable

from openghg_inversions import convert
from openghg_inversions import utils
from openghg_inversions.hbmcmc.components import make_offset
from openghg_inversions.hbmcmc.hbmcmc_output import define_output_filename
from openghg_inversions.config.version import code_version


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
    try:
        dist = getattr(continuous, pdf_dict[pdf])
    except AttributeError:
        raise ValueError(
            f"The distribution '{pdf}' doesn't appear to be a continuous distribution defined by PyMC."
        )

    return dist(name, **params, **kwargs)


def _make_coords(
    Y: np.ndarray,
    Hx: np.ndarray,
    site_indicator: np.ndarray,
    sigma_freq_indices: np.ndarray,
    Hbc: np.ndarray | None = None,
    sites: list[str] | None = None,
    sigma_per_site: bool = False,
) -> dict:
    result = {
        "nmeasure": np.arange(len(Y)),
        "nx": np.arange(Hx.shape[0]),
        "sites": sites if sites is not None else np.unique(site_indicator),
        "nsigma_time": np.unique(sigma_freq_indices),
        "nsigma_site": np.unique(site_indicator) if sigma_per_site else [0],
    }
    if Hbc is not None:
        result["nbc"] = np.arange(Hbc.shape[0])
    return result


def inferpymc(
    Hx: np.ndarray,
    Y: np.ndarray,
    error: np.ndarray,
    siteindicator: np.ndarray,
    sigma_freq_index: np.ndarray,
    Hbc: np.ndarray | None = None,
    xprior: dict = {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
    bcprior: dict = {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
    sigprior: dict = {"pdf": "uniform", "lower": 0.1, "upper": 3.0},
    nuts_sampler: str = "pymc",
    nit: int = 20000,
    burn: int = 10000,
    tune: int = 10000,
    nchain: int = 4,
    sigma_per_site: bool = True,
    offsetprior: dict = {"pdf": "normal", "mu": 0, "sigma": 1},
    add_offset: bool = False,
    verbose: bool = False,
    min_error: np.ndarray | float | None = 0.0,
    use_bc: bool = True,
    reparameterise_log_normal: bool = False,
    pollution_events_from_obs: bool = False,
    no_model_error: bool = False,
    offset_args: dict | None = None,
    power: dict | float = 1.99,
    sampler_kwargs: dict | None = None,
) -> dict:
    """Uses PyMC module for Bayesian inference for emissions field, boundary
    conditions and (currently) a single model error value.
    This uses a Normal likelihood but the (hyper)prior PDFs can be selected by user.

    Args:
      Hx:
        Transpose of the sensitivity matrix to map emissions to measurement.
        This is the same as what is given from fp_data[site].H.values, where
        fp_data is the output from e.g. footprint_data_merge, but where it
        has been stacked for all sites.
      Y:
        Measurement vector containing all measurements
      error:
        Measurement error vector, containg a value for each element of Y.
      siteindicator:
        Array of indexing integers that relate each measurement to a site
      sigma_freq_index:
        Array of integer indexes that converts time into periods
      Hbc:
        Same as Hx but for boundary conditions. Only used if use_bc=True.
      xprior:
        Dictionary containing information about the prior PDF for emissions.
        The entry "pdf" is the name of the analytical PDF used, see
        https://docs.pymc.io/api/distributions/continuous.html for PDFs
        built into pymc3, although they may have to be coded into the script.
        The other entries in the dictionary should correspond to the shape
        parameters describing that PDF as the online documentation,
        e.g. N(1,1**2) would be: `xprior={pdf: "normal", "mu": 1.0, "sigma": 1.0}`.
        Note that the standard deviation should be used rather than the
        precision. Currently all variables are considered iid.
      bcprior:
        Same as xprior but for boundary conditions. Only used if use_bc=True.
      sigprior:
        Same as xprior but for model error.
      nuts_sampler:
        nuts_sampler use by pymc.sample. Options are "pymc" and "numpyro"?
      nit:
        number of samples to generate (per chain)
      burn:
        number of samples to discard (or "burn") from the beginning of each chain
      tune:
        number of tuning steps used by sampler
      nchain:
        number of chains use by sampler. You should use at least 2 chains for the convergence checks
        to work; four chains is better. Chains run in parallel, so the number of chains doesn't affect
        running time, provided the number of threads available is at least the number of chains.
      sigma_per_site (bool):
        Whether a model sigma value will be calculated for each site independantly (True) or all sites together (False).
        Default: True
      offsetprior (dict):
        Same as above but for bias offset. Only used is addoffset=True.
      add_offset (bool):
        Add an offset (intercept) to all sites but the first in the site list. Default False.
      verbose:
        When True, prints progress bar
      min_error:
        Minimum error to use during inversion. Only used if no_model_error is False.
      save_trace:
        Path where to save the trace. If None, the trace is not saved.
        Default None.
      use_bc:
        When True, use and infer boundary conditions.
      reparameterise_log_normal:
        If there are many divergences when using a log normal prior, setting this to True might help. It samples from a normal prior, then puts the normal samples through a function that converts them to log normal samples; this changes the space the sampler needs to explore.
      pollution_events_from_obs:
        When True, calculate the pollution events from obs; when false pollution events are set
        to the modeled concentration.
      no_model_error:
        When True, only use observation error in likelihood function (omitting min. model error
        and model error from scaling pollution events.)
      offset_args: optional arguments to pass to `make_offset`.
      power: power to raise pollution events to when using pollution events from obs. Default is 1.99.
        Any value (strictly) between 1 and 2 will work. If a dictionary is passed, this is used to create
        a prior for the power, making the power a hyper-parameter.

    Returns:
      Dictionary containing:
        xouts (array):
          MCMC chain for emissions scaling factors for each basis function.
        sigouts (array):
          MCMC chain for model error.
        Ytrace (array):
          MCMC chain for modelled obs..
        OFFSETtrace (array):
          MCMC chain for the offset.
        convergence (str):
          Passed/Failed convergence test as to whether mutliple chains
          have a Gelman-Rubin diagnostic value <1.05
        step1 (str):
          Type of MCMC sampler for emissions and boundary condition updates.
          Currently it's hardwired to NUTS (probably wouldn't change this
          unless you're doing something obscure).
        step2 (str):
          Type of MCMC sampler for model error updates.
          Currently it's hardwired to a slice sampler. This parameter is low
          dimensional and quite simple with a slice sampler, although could
          easily be changed.
        bcouts (array):
          MCMC chain for boundary condition scaling factors. Only if use_bc is True.
        YBCtrace (array):
          MCMC chain for modelled boundary condition Only if use_bc is True.

    TO DO:
       - Allow non-iid variables
    """
    if use_bc and Hbc is None:
        raise ValueError("If `use_bc` is True, then `Hbc` must be provided.")

    burn = int(burn)

    hx = Hx.T
    nx = hx.shape[1]

    if use_bc:
        hbc = Hbc.T
        nbc = hbc.shape[1]

    ny = len(Y)

    nit = int(nit)

    # convert siteindicator into a site indexer
    sites = siteindicator.astype(int) if sigma_per_site else np.zeros_like(siteindicator).astype(int)

    coords = _make_coords(
        Y, Hx, siteindicator, sigma_freq_index, Hbc, sigma_per_site=sigma_per_site, sites=None
    )

    if isinstance(min_error, float) or (isinstance(min_error, np.ndarray) and min_error.ndim == 0):
        min_error = min_error * np.ones_like(Y)

    with pm.Model(coords=coords) as model:
        step1_vars = []

        if reparameterise_log_normal and xprior["pdf"] == "lognormal":
            x0 = pm.Normal("x0", 0, 1, dims="nx")
            x = pm.Deterministic("x", pt.exp(xprior["mu"] + xprior["sigma"] * x0), dims="nx")
            step1_vars.append(x0)
        else:
            x = parse_prior("x", xprior, dims="nx")
            step1_vars.append(x)

        if use_bc:
            if reparameterise_log_normal and bcprior["pdf"] == "lognormal":
                bc0 = pm.Normal("bc0", 0, 1, dims="nbc")
                bc = pm.Deterministic("bc", pt.exp(bcprior["mu"] + bcprior["sigma"] * bc0), dims="nbc")
                step1_vars.append(bc0)
            else:
                bc = parse_prior("bc", bcprior, dims="nbc")
                step1_vars.append(bc)

        sigma = parse_prior("sigma", sigprior, dims=("nsigma_site", "nsigma_time"))

        hx = pm.Data("hx", hx, dims=("nmeasure", "nx"))
        mu = pm.Deterministic("mu", pt.dot(hx, x), dims="nmeasure")

        if use_bc:
            hbc = pm.Data("hbc", hbc, dims=("nmeasure", "nbc"))
            mu_bc = pm.Deterministic("mu_bc", pt.dot(hbc, bc), dims="nmeasure")
            mu += mu_bc

        if add_offset:
            offset_args = offset_args or {}
            offset = make_offset(siteindicator, offsetprior, **offset_args)
            mu += offset

        Y = pm.Data("Y", Y, dims="nmeasure")  # type: ignore
        error = pm.Data("error", error, dims="nmeasure")  # type: ignore
        min_error = pm.Data("min_error", min_error, dims="nmeasure")  # type: ignore

        if pollution_events_from_obs is True:
            if use_bc is True:
                pollution_event = pt.abs(Y - mu_bc)
            else:
                pollution_event = pt.abs(Y) + 1e-6 * pt.mean(Y)  # small non-zero term to prevent NaNs
        else:
            pollution_event = pt.abs(pt.dot(hx, x))

        pollution_event_scaled_error = pollution_event * sigma[sites, sigma_freq_index]

        if no_model_error is True:
            # need some small non-zero value to avoid sampling problems
            mean_obs = np.nanmean(Y)
            small_amount = 1e-12 * mean_obs
            eps = pt.maximum(pt.abs(error), small_amount)  # type: ignore
        else:
            power0 = parse_prior("power", power) if isinstance(power, dict) else power
            eps = pt.maximum(pt.sqrt(error**2 + pt.pow(pollution_event_scaled_error, power0)), min_error)  # type: ignore

        epsilon = pm.Deterministic("epsilon", eps, dims="nmeasure")

        pm.Normal("y", mu=mu, sigma=epsilon, observed=Y, dims="nmeasure")

        step1 = pm.NUTS(vars=step1_vars)
        step2 = pm.Slice(vars=[sigma])
        step = [step1, step2] if nuts_sampler == "pymc" else None
        sampler_kwargs = sampler_kwargs or {}
        trace = pm.sample(
            nit,
            tune=int(tune),
            chains=nchain,
            step=step,
            # progressbar=verbose,
            progressbar=False,
            cores=nchain,
            nuts_sampler=nuts_sampler,
            idata_kwargs={"log_likelihood": True},
            **sampler_kwargs,
        )

    posterior_burned = trace.posterior.isel(chain=0, draw=slice(burn, nit)).drop_vars("chain")

    xouts = posterior_burned.x

    if use_bc:
        bcouts = posterior_burned.bc

    sigouts = posterior_burned.sigma

    # Check for convergence
    gelrub = pm.rhat(trace)["x"].max()
    if gelrub > 1.05:
        print("Failed Gelman-Rubin at 1.05")
        convergence = "Failed"
    else:
        convergence = "Passed"

    if nuts_sampler != "pymc":
        divergences = np.sum(trace.sample_stats.diverging).values
        if divergences > 0:
            print(f"There were {divergences} divergences. Try increasing target accept or reparameterise.")

    if add_offset:
        OFFtrace = posterior_burned.offset
    else:
        OFFtrace = xr.zeros_like(posterior_burned.mu)

    if use_bc:
        YBCtrace = posterior_burned.mu_bc + OFFtrace
        Ytrace = posterior_burned.mu + YBCtrace
    else:
        Ytrace = posterior_burned.mu + OFFtrace


    # truncate trace and sample prior and predictive distributions
    trace = trace.isel(draw=slice(burn, None))
    ndraw = nit - burn
    trace.extend(pm.sample_prior_predictive(ndraw, model))
    trace.extend(pm.sample_posterior_predictive(trace, model=model, var_names=["y"]))



    result = {
        "xouts": xouts,
        "sigouts": sigouts,
        "Ytrace": Ytrace.values.T,
        "OFFSETtrace": OFFtrace.values.T,
        "convergence": convergence,
        "step1": step1,
        "step2": step2,
        "model": model,
        "trace": trace,
    }

    if use_bc:
        result["bcouts"] = bcouts
        result["YBCtrace"] = YBCtrace.values.T

    return result


def inferpymc_postprocessouts(
    xouts: np.ndarray,
    sigouts: np.ndarray,
    convergence: str,
    Hx: np.ndarray,
    Y: np.ndarray,
    error: np.ndarray,
    Ytrace: np.ndarray,
    OFFSETtrace: np.ndarray,
    step1: str,
    step2: str,
    xprior: dict,
    sigprior: dict,
    offsetprior: dict | None,
    Ytime: np.ndarray,
    siteindicator: np.ndarray,
    sigma_freq_index: np.ndarray,
    domain: str,
    species: str,
    sites: list,
    start_date: str,
    end_date: str,
    outputname: str,
    outputpath: str,
    country_unit_prefix: str | None,
    burn: int,
    tune: int,
    nchain: int,
    sigma_per_site: bool,
    emissions_name: str,
    bcprior: dict | None = None,
    YBCtrace: np.ndarray | None = None,
    bcouts: np.ndarray | None = None,
    Hbc: np.ndarray | None = None,
    obs_repeatability: np.ndarray | None = None,
    obs_variability: np.ndarray | None = None,
    fp_data: dict | None = None,
    country_file: str | None = None,
    add_offset: bool = False,
    rerun_file: xr.Dataset | None = None,
    use_bc: bool = False,
    min_error: float | np.ndarray = 0.0,
) -> xr.Dataset:
    r"""Takes the output from inferpymc function, along with some other input
    information, calculates statistics on them and places it all in a dataset.
    Also calculates statistics on posterior emissions for the countries in
    the inversion domain and saves all in netcdf.

    Note that the uncertainties are defined by the highest posterior
    density (HPD) region and NOT percentiles (as the tdMCMC code).
    The HPD region is defined, for probability content (1-a), as:
        1) P(x \in R | y) = (1-a)
        2) for x1 \in R and x2 \notin R, P(x1|y)>=P(x2|y)

    Args:
      xouts:
        MCMC chain for emissions scaling factors for each basis function.
      sigouts:
        MCMC chain for model error.
      convergence:
        Passed/Failed convergence test as to whether mutliple chains
        have a Gelman-Rubin diagnostic value <1.05
      Hx:
        Transpose of the sensitivity matrix to map emissions to measurement.
        This is the same as what is given from fp_data[site].H.values, where
        fp_data is the output from e.g. footprint_data_merge, but where it
        has been stacked for all sites.
      Y:
        Measurement vector containing all measurements
      error:
        Measurement error vector, containg a value for each element of Y.
      Ytrace:
        Trace of modelled y values calculated from mcmc outputs and H matrices
      OFFSETtrace:
        Trace from offsets (if used).
      step1:
        Type of MCMC sampler for emissions and boundary condition updates.
      step2:
        Type of MCMC sampler for model error updates.
      xprior:
        Dictionary containing information about the prior PDF for emissions.
        The entry "pdf" is the name of the analytical PDF used, see
        https://docs.pymc.io/api/distributions/continuous.html for PDFs
        built into pymc3, although they may have to be coded into the script.
        The other entries in the dictionary should correspond to the shape
        parameters describing that PDF as the online documentation,
        e.g. N(1,1**2) would be: xprior={pdf:"normal", "mu":1, "sigma":1}.
        Note that the standard deviation should be used rather than the
        precision. Currently all variables are considered iid.
      sigprior:
        Same as xprior but for model error.
      offsetprior:
        Same as xprior but for bias offset. Only used is add_offset=True.
      Ytime:
        Time stamp of measurements as used by the inversion.
      siteindicator:
        Numerical indicator of which site the measurements belong to,
        same length at Y.
      sigma_freq_index:
        Array of integer indexes that converts time into periods
      domain:
        Inversion spatial domain.
      species:
        Species of interest
      sites:
        List of sites in inversion
      start_date:
        Start time of inversion "YYYY-mm-dd"
      end_date:
        End time of inversion "YYYY-mm-dd"
      outputname:
        Unique identifier for output/run name.
      outputpath:
        Path to where output should be saved.
      country_unit_prefix:
        A prefix for scaling the country emissions. Current options are:
        'T' will scale to Tg, 'G' to Gg, 'M' to Mg, 'P' to Pg.
        To add additional options add to acrg_convert.prefix
        Default is none and no scaling will be applied (output in g).
      burn:
        Number of iterations burned in MCMC
      tune:
        Number of iterations used to tune step size
      nchain:
        Number of independent chains run
      sigma_per_site:
        Whether a model sigma value will be calculated for each site independantly (True)
        or all sites together (False).
      emissions_name:
        List with "source" values as used when adding emissions data to the OpenGHG object store.
      bcprior:
        Same as xrpior but for boundary conditions.
      YBCtrace:
        Trace of modelled boundary condition values calculated from mcmc outputs and Hbc matrices
      bcouts:
        MCMC chain for boundary condition scaling factors.
      Hbc:
        Same as Hx but for boundary conditions
      obs_repeatability:
        Instrument error
      obs_variability:
        Error from resampling observations
      fp_data:
        Output from footprints_data_merge + sensitivies
      country_file:
        Path of country definition file
      add_offset:
        Add an offset (intercept) to all sites but the first in the site list. Default False.
      rerun_file (xarray dataset, optional):
        An xarray dataset containing the ncdf output from a previous run of the MCMC code.
      use_bc:
        When True, use and infer boundary conditions.
      min_error:
        Minimum error to use during inversion. Only used if no_model_error is False.

    Returns:
        xarray dataset containing results from inversion

    TO DO:
        - Look at compressability options for netcdf output
        - I'm sure the number of inputs can be cut down or found elsewhere.
        - Currently it can only work out the country total emissions if
          the a priori emissions are constant over the inversion period
          or else monthly (and inversion is for less than one calendar year).
    """
    print("Post-processing output")

    # Get parameters for output file
    nit = xouts.shape[0]
    nx = Hx.shape[0]
    ny = len(Y)

    if use_bc:
        nbc = Hbc.shape[0]
        nBC = np.arange(nbc)

    nui = np.arange(2)
    steps = np.arange(nit)
    nmeasure = np.arange(ny)
    nparam = np.arange(nx)

    # OFFSET HYPERPARAMETER
    YmodmuOFF = np.mean(OFFSETtrace, axis=1)  # mean
    YmodmedOFF = np.median(OFFSETtrace, axis=1)  # median
    YmodmodeOFF = np.zeros(shape=OFFSETtrace.shape[0])  # mode

    for i in range(0, OFFSETtrace.shape[0]):
        # if sufficient no. of iterations use a KDE to calculate mode
        # else, mean value used in lieu
        if np.nanmax(OFFSETtrace[i, :]) > np.nanmin(OFFSETtrace[i, :]):
            xes_off = np.linspace(np.nanmin(OFFSETtrace[i, :]), np.nanmax(OFFSETtrace[i, :]), 200)
            kde = stats.gaussian_kde(OFFSETtrace[i, :]).evaluate(xes_off)
            YmodmodeOFF[i] = xes_off[kde.argmax()]
        else:
            YmodmodeOFF[i] = np.mean(OFFSETtrace[i, :])

    Ymod95OFF = az.hdi(OFFSETtrace.T, 0.95)
    Ymod68OFF = az.hdi(OFFSETtrace.T, 0.68)

    # Y-BC HYPERPARAMETER
    if use_bc:
        YmodmuBC = np.mean(YBCtrace, axis=1)
        YmodmedBC = np.median(YBCtrace, axis=1)
        YmodmodeBC = np.zeros(shape=YBCtrace.shape[0])

        for i in range(0, YBCtrace.shape[0]):
            # if sufficient no. of iterations use a KDE to calculate mode
            # else, mean value used in lieu
            if np.nanmax(YBCtrace[i, :]) > np.nanmin(YBCtrace[i, :]):
                xes_bc = np.linspace(np.nanmin(YBCtrace[i, :]), np.nanmax(YBCtrace[i, :]), 200)
                kde = stats.gaussian_kde(YBCtrace[i, :]).evaluate(xes_bc)
                YmodmodeBC[i] = xes_bc[kde.argmax()]
            else:
                YmodmodeBC[i] = np.mean(YBCtrace[i, :])

        Ymod95BC = az.hdi(YBCtrace.T, 0.95)
        Ymod68BC = az.hdi(YBCtrace.T, 0.68)
        YaprioriBC = np.sum(Hbc, axis=0)

    # Y-VALUES HYPERPARAMETER (XOUTS * H)
    Ymodmu = np.mean(Ytrace, axis=1)
    Ymodmed = np.median(Ytrace, axis=1)
    Ymodmode = np.zeros(shape=Ytrace.shape[0])

    for i in range(0, Ytrace.shape[0]):
        # if sufficient no. of iterations use a KDE to calculate mode
        # else, mean value used in lieu
        if np.nanmax(Ytrace[i, :]) > np.nanmin(Ytrace[i, :]):
            xes = np.arange(np.nanmin(Ytrace[i, :]), np.nanmax(Ytrace[i, :]), 0.5)
            kde = stats.gaussian_kde(Ytrace[i, :]).evaluate(xes)
            Ymodmode[i] = xes[kde.argmax()]
        else:
            Ymodmode[i] = np.mean(Ytrace[i, :])

    Ymod95 = az.hdi(Ytrace.T, 0.95)
    Ymod68 = az.hdi(Ytrace.T, 0.68)

    if use_bc:
        Yapriori = np.sum(Hx.T, axis=1) + np.sum(Hbc.T, axis=1)
    else:
        Yapriori = np.sum(Hx.T, axis=1)

    sitenum = np.arange(len(sites))

    if fp_data is None and rerun_file is not None:
        lon = rerun_file.lon.values
        lat = rerun_file.lat.values
        site_lat = rerun_file.sitelats.values
        site_lon = rerun_file.sitelons.values
        bfds = rerun_file.basisfunctions
    else:
        lon = fp_data[sites[0]].lon.values
        lat = fp_data[sites[0]].lat.values
        site_lat = np.zeros(len(sites))
        site_lon = np.zeros(len(sites))
        for si, site in enumerate(sites):
            site_lat[si] = fp_data[site].release_lat.values[0]
            site_lon[si] = fp_data[site].release_lon.values[0]
        bfds = fp_data[".basis"]

    # Calculate mean  and mode posterior scale map and flux field
    scalemap_mu = np.zeros_like(bfds.values)
    scalemap_mode = np.zeros_like(bfds.values)

    for npm in nparam:
        scalemap_mu[bfds.values == (npm + 1)] = np.mean(xouts[:, npm])
        if np.nanmax(xouts[:, npm]) > np.nanmin(xouts[:, npm]):
            xes = np.arange(np.nanmin(xouts[:, npm]), np.nanmax(xouts[:, npm]), 0.01)
            kde = stats.gaussian_kde(xouts[:, npm]).evaluate(xes)
            scalemap_mode[bfds.values == (npm + 1)] = xes[kde.argmax()]
        else:
            scalemap_mode[bfds.values == (npm + 1)] = np.mean(xouts[:, npm])

    if rerun_file is not None:
        flux_array_all = np.expand_dims(rerun_file.fluxapriori.values, 2)
    elif emissions_name is None:
        raise ValueError("Emissions name not provided.")
    else:
        emds = fp_data[".flux"][emissions_name[0]]
        flux_array_all = emds.data.flux.values

    # HACK: assume that smallest flux dim is time, then re-order flux so that
    # time is the last coordinate
    flux_dim_shape = flux_array_all.shape
    flux_dim_positions = range(len(flux_dim_shape))
    smallest_dim_position = min(list(zip(flux_dim_positions, flux_dim_shape)), key=(lambda x: x[1]))[0]

    flux_array_all = np.moveaxis(flux_array_all, smallest_dim_position, -1)
    # end HACK

    if flux_array_all.shape[2] == 1:
        print("\nAssuming flux prior is annual and extracting first index of flux array.")
        apriori_flux = flux_array_all[:, :, 0]
    else:
        print("\nAssuming flux prior is monthly.")
        print(f"Extracting weighted average flux prior from {start_date} to {end_date}")
        allmonths = pd.date_range(start_date, end_date).month[:-1].values
        allmonths -= 1  # to align with zero indexed array

        apriori_flux = np.zeros_like(flux_array_all[:, :, 0])

        # calculate the weighted average flux across the whole inversion period
        for m in np.unique(allmonths):
            apriori_flux += flux_array_all[:, :, m] * np.sum(allmonths == m) / len(allmonths)

    flux = scalemap_mode * apriori_flux

    # Basis functions to save
    bfarray = bfds.values - 1

    # Calculate country totals
    area = utils.areagrid(lat, lon)
    if not rerun_file:
        c_object = utils.get_country(domain, country_file=country_file)
        cntryds = xr.Dataset(
            {"country": (["lat", "lon"], c_object.country), "name": (["ncountries"], c_object.name)},
            coords={"lat": (c_object.lat), "lon": (c_object.lon)},
        )
        cntrynames = cntryds.name.values
        cntrygrid = cntryds.country.values
    else:
        cntrynames = rerun_file.countrynames.values
        cntrygrid = rerun_file.countrydefinition.values

    cntrymean = np.zeros(len(cntrynames))
    cntrymedian = np.zeros(len(cntrynames))
    cntrymode = np.zeros(len(cntrynames))
    cntry68 = np.zeros((len(cntrynames), len(nui)))
    cntry95 = np.zeros((len(cntrynames), len(nui)))
    cntrysd = np.zeros(len(cntrynames))
    cntryprior = np.zeros(len(cntrynames))
    molarmass = convert.molar_mass(species)

    unit_factor = convert.prefix(country_unit_prefix)
    if country_unit_prefix is None:
        country_unit_prefix = ""
    country_units = country_unit_prefix + "g"
    if rerun_file is not None:
        obs_units = rerun_file.Yobs.attrs["units"].split(" ")[0]
    else:
        obs_units = str(fp_data[".units"])

    for ci, cntry in enumerate(cntrynames):
        cntrytottrace = np.zeros(len(steps))
        cntrytotprior = 0
        for bf in range(int(np.max(bfarray)) + 1):
            bothinds = np.logical_and(cntrygrid == ci, bfarray == bf)
            cntrytottrace += (
                np.sum(area[bothinds].ravel() * apriori_flux[bothinds].ravel() * 3600 * 24 * 365 * molarmass)
                * xouts[:, bf]
                / unit_factor
            )
            cntrytotprior += (
                np.sum(area[bothinds].ravel() * apriori_flux[bothinds].ravel() * 3600 * 24 * 365 * molarmass)
                / unit_factor
            )
        cntrymean[ci] = np.mean(cntrytottrace)
        cntrymedian[ci] = np.median(cntrytottrace)

        if np.nanmax(cntrytottrace) > np.nanmin(cntrytottrace):
            xes = np.linspace(np.nanmin(cntrytottrace), np.nanmax(cntrytottrace), 200)
            kde = stats.gaussian_kde(cntrytottrace).evaluate(xes)
            cntrymode[ci] = xes[kde.argmax()]
        else:
            cntrymode[ci] = np.mean(cntrytottrace)

        cntrysd[ci] = np.std(cntrytottrace)
        cntry68[ci, :] = az.hdi(cntrytottrace.values, 0.68)
        cntry95[ci, :] = az.hdi(cntrytottrace.values, 0.95)
        cntryprior[ci] = cntrytotprior

    # make min. model error variable
    if isinstance(min_error, float) or (isinstance(min_error, np.ndarray) and min_error.ndim == 0):
        min_error = min_error * np.ones_like(Y)

    # Make output netcdf file
    data_vars = {
        "Yobs": (["nmeasure"], Y),
        "Yerror": (["nmeasure"], error),
        "Yerror_repeatability": (["nmeasure"], obs_repeatability),
        "Yerror_variability": (["nmeasure"], obs_variability),
        "min_model_error": (["nmeasure"], min_error),
        "Ytime": (["nmeasure"], Ytime),
        "Yapriori": (["nmeasure"], Yapriori),
        "Ymodmean": (["nmeasure"], Ymodmu),
        "Ymodmedian": (["nmeasure"], Ymodmed),
        "Ymodmode": (["nmeasure"], Ymodmode),
        "Ymod95": (["nmeasure", "nUI"], Ymod95),
        "Ymod68": (["nmeasure", "nUI"], Ymod68),
        "Yoffmean": (["nmeasure"], YmodmuOFF),
        "Yoffmedian": (["nmeasure"], YmodmedOFF),
        "Yoffmode": (["nmeasure"], YmodmodeOFF),
        "Yoff68": (["nmeasure", "nUI"], Ymod68OFF),
        "Yoff95": (["nmeasure", "nUI"], Ymod95OFF),
        "xtrace": (["steps", "nparam"], xouts.values),
        "sigtrace": (["steps", "nsigma_site", "nsigma_time"], sigouts.values),
        "siteindicator": (["nmeasure"], siteindicator),
        "sigmafreqindex": (["nmeasure"], sigma_freq_index),
        "sitenames": (["nsite"], sites),
        "sitelons": (["nsite"], site_lon),
        "sitelats": (["nsite"], site_lat),
        "fluxapriori": (["lat", "lon"], apriori_flux),
        "fluxmode": (["lat", "lon"], flux),
        "scalingmean": (["lat", "lon"], scalemap_mu),
        "scalingmode": (["lat", "lon"], scalemap_mode),
        "basisfunctions": (["lat", "lon"], bfarray),
        "countrymean": (["countrynames"], cntrymean),
        "countrymedian": (["countrynames"], cntrymedian),
        "countrymode": (["countrynames"], cntrymode),
        "countrysd": (["countrynames"], cntrysd),
        "country68": (["countrynames", "nUI"], cntry68),
        "country95": (["countrynames", "nUI"], cntry95),
        "countryapriori": (["countrynames"], cntryprior),
        "countrydefinition": (["lat", "lon"], cntrygrid),
        "xsensitivity": (["nmeasure", "nparam"], Hx.T),
    }

    coords = {
        "stepnum": (["steps"], steps),
        "paramnum": (["nlatent"], nparam),
        "measurenum": (["nmeasure"], nmeasure),
        "UInum": (["nUI"], nui),
        "nsites": (["nsite"], sitenum),
        "nsigma_time": (["nsigma_time"], np.unique(sigma_freq_index)),
        "nsigma_site": (["nsigma_site"], np.arange(sigouts.shape[1]).astype(int)),
        "lat": (["lat"], lat),
        "lon": (["lon"], lon),
        "countrynames": (["countrynames"], cntrynames),
    }

    if use_bc:
        data_vars.update(
            {
                "YaprioriBC": (["nmeasure"], YaprioriBC),
                "YmodmeanBC": (["nmeasure"], YmodmuBC),
                "YmodmedianBC": (["nmeasure"], YmodmedBC),
                "YmodmodeBC": (["nmeasure"], YmodmodeBC),
                "Ymod95BC": (["nmeasure", "nUI"], Ymod95BC),
                "Ymod68BC": (["nmeasure", "nUI"], Ymod68BC),
                "bctrace": (["steps", "nBC"], bcouts.values),
                "bcsensitivity": (["nmeasure", "nBC"], Hbc.T),
            }
        )
        coords["numBC"] = (["nBC"], nBC)

    outds = xr.Dataset(data_vars, coords=coords)

    outds.fluxmode.attrs["units"] = "mol/m2/s"
    outds.fluxapriori.attrs["units"] = "mol/m2/s"
    outds.Yobs.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yerror.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yerror_repeatability.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yerror_variability.attrs["units"] = obs_units + " " + "mol/mol"
    outds.min_model_error.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yapriori.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Ymodmean.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Ymodmedian.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Ymodmode.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Ymod95.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Ymod68.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yoffmean.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yoffmedian.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yoffmode.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yoff95.attrs["units"] = obs_units + " " + "mol/mol"
    outds.Yoff68.attrs["units"] = obs_units + " " + "mol/mol"
    outds.countrymean.attrs["units"] = country_units
    outds.countrymedian.attrs["units"] = country_units
    outds.countrymode.attrs["units"] = country_units
    outds.country68.attrs["units"] = country_units
    outds.country95.attrs["units"] = country_units
    outds.countrysd.attrs["units"] = country_units
    outds.countryapriori.attrs["units"] = country_units
    outds.xsensitivity.attrs["units"] = obs_units + " " + "mol/mol"
    outds.sigtrace.attrs["units"] = obs_units + " " + "mol/mol"

    outds.Yobs.attrs["longname"] = "observations"
    outds.Yerror.attrs["longname"] = "measurement error"
    outds.min_model_error.attrs["longname"] = "minimum model error"
    outds.Ytime.attrs["longname"] = "time of measurements"
    outds.Yapriori.attrs["longname"] = "a priori simulated measurements"
    outds.Ymodmean.attrs["longname"] = "mean of posterior simulated measurements"
    outds.Ymodmedian.attrs["longname"] = "median of posterior simulated measurements"
    outds.Ymodmode.attrs["longname"] = "mode of posterior simulated measurements"
    outds.Ymod68.attrs["longname"] = " 0.68 Bayesian credible interval of posterior simulated measurements"
    outds.Ymod95.attrs["longname"] = " 0.95 Bayesian credible interval of posterior simulated measurements"
    outds.Yoffmean.attrs["longname"] = "mean of posterior simulated offset between measurements"
    outds.Yoffmedian.attrs["longname"] = "median of posterior simulated offset between measurements"
    outds.Yoffmode.attrs["longname"] = "mode of posterior simulated offset between measurements"
    outds.Yoff68.attrs["longname"] = (
        " 0.68 Bayesian credible interval of posterior simulated offset between measurements"
    )
    outds.Yoff95.attrs["longname"] = (
        " 0.95 Bayesian credible interval of posterior simulated offset between measurements"
    )
    outds.xtrace.attrs["longname"] = "trace of unitless scaling factors for emissions parameters"
    outds.sigtrace.attrs["longname"] = "trace of model error parameters"
    outds.siteindicator.attrs["longname"] = "index of site of measurement corresponding to sitenames"
    outds.sigmafreqindex.attrs["longname"] = "perdiod over which the model error is estimated"
    outds.sitenames.attrs["longname"] = "site names"
    outds.sitelons.attrs["longname"] = "site longitudes corresponding to site names"
    outds.sitelats.attrs["longname"] = "site latitudes corresponding to site names"
    outds.fluxapriori.attrs["longname"] = "mean a priori flux over period"
    outds.fluxmode.attrs["longname"] = "mode posterior flux over period"
    outds.scalingmean.attrs["longname"] = "mean scaling factor field over period"
    outds.scalingmode.attrs["longname"] = "mode scaling factor field over period"
    outds.basisfunctions.attrs["longname"] = "basis function field"
    outds.countrymean.attrs["longname"] = "mean of ocean and country totals"
    outds.countrymedian.attrs["longname"] = "median of ocean and country totals"
    outds.countrymode.attrs["longname"] = "mode of ocean and country totals"
    outds.country68.attrs["longname"] = "0.68 Bayesian credible interval of ocean and country totals"
    outds.country95.attrs["longname"] = "0.95 Bayesian credible interval of ocean and country totals"
    outds.countrysd.attrs["longname"] = "standard deviation of ocean and country totals"
    outds.countryapriori.attrs["longname"] = "prior mean of ocean and country totals"
    outds.countrydefinition.attrs["longname"] = "grid definition of countries"
    outds.xsensitivity.attrs["longname"] = "emissions sensitivity timeseries"

    if use_bc:
        outds.YmodmeanBC.attrs["units"] = obs_units + " " + "mol/mol"
        outds.YmodmedianBC.attrs["units"] = obs_units + " " + "mol/mol"
        outds.YmodmodeBC.attrs["units"] = obs_units + " " + "mol/mol"
        outds.Ymod95BC.attrs["units"] = obs_units + " " + "mol/mol"
        outds.Ymod68BC.attrs["units"] = obs_units + " " + "mol/mol"
        outds.YaprioriBC.attrs["units"] = obs_units + " " + "mol/mol"
        outds.bcsensitivity.attrs["units"] = obs_units + " " + "mol/mol"

        outds.YaprioriBC.attrs["longname"] = "a priori simulated boundary conditions"
        outds.YmodmeanBC.attrs["longname"] = "mean of posterior simulated boundary conditions"
        outds.YmodmedianBC.attrs["longname"] = "median of posterior simulated boundary conditions"
        outds.YmodmodeBC.attrs["longname"] = "mode of posterior simulated boundary conditions"
        outds.Ymod68BC.attrs["longname"] = (
            " 0.68 Bayesian credible interval of posterior simulated boundary conditions"
        )
        outds.Ymod95BC.attrs["longname"] = (
            " 0.95 Bayesian credible interval of posterior simulated boundary conditions"
        )
        outds.bctrace.attrs["longname"] = (
            "trace of unitless scaling factors for boundary condition parameters"
        )
        outds.bcsensitivity.attrs["longname"] = "boundary conditions sensitivity timeseries"

    outds.attrs["Start date"] = start_date
    outds.attrs["End date"] = end_date
    outds.attrs["Latent sampler"] = str(step1)[20:33]
    outds.attrs["Hyper sampler"] = str(step2)[20:33]
    outds.attrs["Burn in"] = str(int(burn))
    outds.attrs["Tuning steps"] = str(int(tune))
    outds.attrs["Number of chains"] = str(int(nchain))
    outds.attrs["Error for each site"] = str(sigma_per_site)
    outds.attrs["Emissions Prior"] = "".join([f"{k},{v}," for k, v in xprior.items()])[:-1]
    outds.attrs["Model error Prior"] = "".join([f"{k},{v}," for k, v in sigprior.items()])[:-1]
    if use_bc:
        outds.attrs["BCs Prior"] = "".join([f"{k},{v}," for k, v in bcprior.items()])[:-1]
    if add_offset:
        outds.attrs["Offset Prior"] = "".join([f"{k},{v}," for k, v in offsetprior.items()])[:-1]
    outds.attrs["Creator"] = getpass.getuser()
    outds.attrs["Date created"] = str(pd.Timestamp("today"))
    outds.attrs["Convergence"] = convergence
    outds.attrs["Repository version"] = code_version()
    outds.attrs["min_model_error"] = (
        min_error  # TODO: remove this once PARIS formatting switches over to using min error data var
    )

    # variables with variable length data types shouldn't be compressed
    # e.g. object ("O") or unicode ("U") type
    do_not_compress = []
    dtype_pat = re.compile(r"[<>=]?[UO]")  # regex for Unicode and Object dtypes
    for dv in outds.data_vars:
        if dtype_pat.match(outds[dv].data.dtype.str):
            do_not_compress.append(dv)

    # setting compression levels for data vars in outds
    comp = dict(zlib=True, complevel=5, shuffle=True)
    encoding = {var: comp for var in outds.data_vars if var not in do_not_compress}

    output_filename = define_output_filename(outputpath, species, domain, outputname, start_date, ext=".nc")
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    outds.to_netcdf(output_filename, encoding=encoding, mode="w")

    return outds
