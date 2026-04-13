"""Functions for performing MCMC inversion with the xarray-first PyMC builder."""

import re
import getpass
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# import pytensor before pymc so we can set config values
import pytensor

pytensor.config.floatX = "float32"
pytensor.config.warn_float64 = "warn"

import pymc as pm  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
import arviz as az  # noqa: E402
from scipy import stats  # noqa: E402

from openghg_inversions import convert  # noqa: E402
from openghg_inversions import utils  # noqa: E402
from openghg_inversions.hbmcmc.hbmcmc_output import define_output_filename  # noqa: E402
from openghg_inversions.config.version import code_version  # noqa: E402
from openghg_inversions.models.components import (  # noqa: E402
    add_inferpymc_likelihood_component,
    add_linear_component,
    add_offset_component,
)
from openghg_inversions.models.coords import CoordRegistry, attach_coord_registry  # noqa: E402
from openghg_inversions.models.priors import PriorArgs  # noqa: E402


@dataclass
class InferPyMCModelSetup:
    """Container for the PyMC model and sampler configuration used by inferpymc.

    Attributes:
        model: PyMC model built for the inversion.
        step1: Step method used for emissions and boundary-condition variables.
        step2: Step method used for sigma variables.
        sample_kwargs: Extra keyword arguments forwarded to ``pm.sample``.
    """

    model: pm.Model
    step1: Any
    step2: Any
    sample_kwargs: dict[str, Any]


def _contiguous_index(index: np.ndarray) -> np.ndarray:
    """Remap integer period indices to contiguous 0..N-1 values."""
    index = np.asarray(index, dtype=int)
    if index.size == 0:
        return index

    uniq = np.unique(index)
    return np.searchsorted(uniq, index).astype(int)


def _contiguous_sigma_time_index(sigma_freq_index: np.ndarray) -> np.ndarray:
    """Remap sigma period indices to contiguous 0..N-1 values.

    Monthly period indices can legitimately have gaps when entire periods have no
    data (e.g. Jan and Mar only => [0, 2]). PyMC array indexing is positional, so
    these values must be compacted before using `sigma[..., sigma_freq_index]`.
    """
    return _contiguous_index(sigma_freq_index)


def _weighted_apriori_flux_for_months(flux_array_all: np.ndarray, month_index: np.ndarray) -> np.ndarray:
    """Compute a weighted prior flux average using compacted month positions."""
    month_index = _contiguous_index(month_index)
    apriori_flux = np.zeros_like(flux_array_all[:, :, 0])

    for month_pos in np.unique(month_index):
        apriori_flux += flux_array_all[:, :, month_pos] * np.sum(month_index == month_pos) / len(month_index)

    return apriori_flux


# ----------------------------------------
# Model building code
# ----------------------------------------

# Defaults to avoid mutable default arguments in model building functions.
DEFAULT_XPRIOR: PriorArgs = {"pdf": "normal", "mu": 1.0, "sigma": 1.0}
DEFAULT_BCPRIOR: PriorArgs = {"pdf": "normal", "mu": 1.0, "sigma": 1.0}
DEFAULT_SIGPRIOR: PriorArgs = {"pdf": "uniform", "lower": 0.1, "upper": 3.0}
DEFAULT_OFFSETPRIOR: PriorArgs = {"pdf": "normal", "mu": 0, "sigma": 1}


def _prepare_builder_priors(
    *,
    xprior: dict | None,
    bcprior: dict | None,
    sigprior: dict | None,
    offsetprior: dict | None,
    reparameterise_log_normal: bool,
) -> tuple[dict, dict, dict, dict]:
    """Copy builder priors and apply builder-level prior options.

    Args:
        xprior: Optional emissions prior overrides.
        bcprior: Optional boundary-condition prior overrides.
        sigprior: Optional sigma prior overrides.
        offsetprior: Optional offset prior overrides.
        reparameterise_log_normal: Public compatibility flag for requesting
            lognormal reparameterisation. The preferred long-term interface is
            to set ``"reparameterise": True`` directly in the relevant prior
            argument dictionary.

    Returns:
        Copies of the prior dictionaries with defaults filled in and any
        builder-level reparameterisation settings applied.
    """
    prepared_xprior = DEFAULT_XPRIOR.copy() if xprior is None else xprior.copy()
    prepared_bcprior = DEFAULT_BCPRIOR.copy() if bcprior is None else bcprior.copy()
    prepared_sigprior = DEFAULT_SIGPRIOR.copy() if sigprior is None else sigprior.copy()
    prepared_offsetprior = DEFAULT_OFFSETPRIOR.copy() if offsetprior is None else offsetprior.copy()

    if reparameterise_log_normal:
        # TODO: later prefer prior-native `reparameterise=True` over this separate public flag.
        if str(prepared_xprior.get("pdf", "")).lower() == "lognormal":
            prepared_xprior["reparameterise"] = True
        if str(prepared_bcprior.get("pdf", "")).lower() == "lognormal":
            prepared_bcprior["reparameterise"] = True

    return prepared_xprior, prepared_bcprior, prepared_sigprior, prepared_offsetprior


def _canonicalise_inferpymc_dataset(
    inv_inputs: xr.Dataset,
    /,
    use_bc: bool,
    obs_dim: str = "nmeasure",
    input_state_dim: str = "region",
    input_bc_state_dim: str = "bc_region",
    state_dim: str = "nx",
    bc_state_dim: str = "nbc",
) -> xr.Dataset:
    """Convert inversion inputs into the canonical dataset used by the builder.

    Args:
        inv_inputs: Dataset produced by ``make_inv_inputs``.
        use_bc: Whether the canonical dataset should include boundary-condition
            sensitivities.
        obs_dim: Observation dimension name used by the model components.
        input_state_dim: State dimension name used by emissions sensitivities in
            ``inv_inputs``.
        input_bc_state_dim: State dimension name used by BC sensitivities in
            ``inv_inputs``.
        state_dim: State dimension name used for emissions sensitivities.
        bc_state_dim: State dimension name used for boundary-condition
            sensitivities.

    Returns:
        An xarray dataset with observation-first sensitivities and compact sigma
        period indices.

    Raises:
        ValueError: If boundary conditions are requested but ``inv_inputs`` does
            not contain ``H_bc``.
    """
    obs_coord = inv_inputs.indexes[obs_dim] if obs_dim in inv_inputs.indexes else inv_inputs[obs_dim].values
    state_coord = inv_inputs[input_state_dim].values
    data_vars: dict[str, xr.DataArray] = {
        "H": xr.DataArray(
            inv_inputs["H"].transpose(obs_dim, input_state_dim).values,
            dims=(obs_dim, state_dim),
            coords={obs_dim: obs_coord, state_dim: np.arange(state_coord.size)},
            name="H",
        ),
        "mf": xr.DataArray(inv_inputs["mf"].values, dims=(obs_dim,), coords={obs_dim: obs_coord}, name="mf"),
        "mf_error": xr.DataArray(
            inv_inputs["mf_error"].values,
            dims=(obs_dim,),
            coords={obs_dim: obs_coord},
            name="mf_error",
        ),
        "site_indicator": xr.DataArray(
            inv_inputs["site_indicator"].values.astype(int),
            dims=(obs_dim,),
            coords={obs_dim: obs_coord},
            name="site_indicator",
        ),
        "sigma_freq_index": xr.DataArray(
            _contiguous_sigma_time_index(inv_inputs["sigma_freq_index"].values),
            dims=(obs_dim,),
            coords={obs_dim: obs_coord},
            name="sigma_freq_index",
        ),
    }
    min_error_values = inv_inputs["min_error"].values
    if np.isscalar(min_error_values) or np.ndim(min_error_values) == 0:
        min_error_values = np.full(inv_inputs.sizes[obs_dim], min_error_values)
    data_vars["min_error"] = xr.DataArray(
        min_error_values,
        dims=(obs_dim,),
        coords={obs_dim: obs_coord},
        name="min_error",
    )
    coords: dict[str, Any] = {obs_dim: obs_coord, state_dim: np.arange(state_coord.size)}

    if not isinstance(obs_coord, pd.MultiIndex):
        if "time" in inv_inputs.coords:
            coords["time"] = xr.DataArray(
                inv_inputs["time"].values,
                dims=(obs_dim,),
                coords={obs_dim: obs_coord},
                name="time",
            )

        for coord_name in ("site",):
            if coord_name in inv_inputs.coords:
                coords[coord_name] = xr.DataArray(
                    inv_inputs[coord_name].values,
                    dims=(obs_dim,),
                    coords={obs_dim: obs_coord},
                    name=coord_name,
                )

    if use_bc:
        if "H_bc" not in inv_inputs:
            raise ValueError("If `use_bc` is True, `inv_inputs` must contain `H_bc`.")
        bc_coord = inv_inputs[input_bc_state_dim].values
        coords[bc_state_dim] = np.arange(bc_coord.size)
        data_vars["H_bc"] = xr.DataArray(
            inv_inputs["H_bc"].transpose(obs_dim, input_bc_state_dim).values,
            dims=(obs_dim, bc_state_dim),
            coords={obs_dim: obs_coord, bc_state_dim: np.arange(bc_coord.size)},
            name="H_bc",
        )

    return xr.Dataset(data_vars=data_vars, coords=coords)


def build_inferpymc_model(
    xprior: dict | None = None,
    bcprior: dict | None = None,
    sigprior: dict | None = None,
    sigma_per_site: bool = True,
    offsetprior: dict | None = None,
    add_offset: bool = False,
    min_error: np.ndarray | float | None = None,
    use_bc: bool = True,
    reparameterise_log_normal: bool = False,
    pollution_events_from_obs: bool = False,
    no_model_error: bool = False,
    offset_args: dict | None = None,
    power: dict | float = 1.99,
    nuts_sampler: str = "pymc",
    inv_inputs: xr.Dataset | None = None,
) -> InferPyMCModelSetup:
    """Build the current component-based inferpymc model.

    Args:
        xprior: Prior specification for emissions scaling factors.
        bcprior: Prior specification for boundary-condition scaling factors.
        sigprior: Prior specification for model-error terms.
        sigma_per_site: Whether sigma should vary by site.
        offsetprior: Prior specification for optional offsets.
        add_offset: Whether to include an offset term in the model.
        min_error: Retained for API compatibility. The active runtime path uses
            ``inv_inputs["min_error"]``.
        use_bc: Whether to include boundary-condition terms in the model.
        reparameterise_log_normal: Whether to request lognormal
            reparameterisation for supported priors.
        pollution_events_from_obs: Whether to derive pollution-event scaling
            from observations rather than modelled concentrations.
        no_model_error: Whether to suppress the explicit model-error term.
        offset_args: Extra keyword arguments forwarded to
            ``add_offset_component``.
        power: Exponent or prior specification used in the likelihood error
            scaling.
        nuts_sampler: Sampler backend name passed through to the model setup.
        inv_inputs: Dataset produced by ``make_inv_inputs``.

    Returns:
        ``InferPyMCModelSetup`` containing the built model and sampler
        configuration.

    Raises:
        ValueError: If ``inv_inputs`` is not provided.
    """
    if inv_inputs is None:
        raise ValueError("`inferpymc` now expects `inv_inputs` from `make_inv_inputs`.")

    canonical_ds = _canonicalise_inferpymc_dataset(inv_inputs, use_bc=use_bc)
    xprior, bcprior, sigprior, offsetprior = _prepare_builder_priors(
        xprior=xprior,
        bcprior=bcprior,
        sigprior=sigprior,
        offsetprior=offsetprior,
        reparameterise_log_normal=reparameterise_log_normal,
    )

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        step1_vars = []

        flux_component = add_linear_component(
            canonical_ds["H"],
            data_name="hx",
            prior_args=xprior,
            var_name="x",
            output_name="mu",
            output_dim="nmeasure",
            compute_deterministic=True,
        )
        step1_vars.append(flux_component.latent)

        mu_bc = None
        if use_bc and "H_bc" in canonical_ds:
            bc_component = add_linear_component(
                canonical_ds["H_bc"],
                data_name="hbc",
                prior_args=bcprior,
                var_name="bc",
                output_name="mu_bc",
                output_dim="nmeasure",
                compute_deterministic=True,
            )
            mu_bc = bc_component.output
            step1_vars.append(bc_component.latent)

        offset = None
        if add_offset:
            offset_args = offset_args or {}
            offset = add_offset_component(
                canonical_ds["site_indicator"],
                prior_args=offsetprior,
                output_name="offset",
                output_dim="nmeasure",
                **offset_args,
            )

        add_inferpymc_likelihood_component(
            canonical_ds,
            mu=flux_component.output,
            mu_bc=mu_bc,
            offset=offset,
            sigprior=sigprior,
            power=power,
            pollution_events_from_obs=pollution_events_from_obs,
            no_model_error=no_model_error,
            sigma_per_site=sigma_per_site,
            output_dim="nmeasure",
        )

        sigma = model.named_vars["sigma"]
        step1 = pm.NUTS(vars=step1_vars)
        step2 = pm.Slice(vars=[sigma])

    return InferPyMCModelSetup(
        model=model,
        step1=step1,
        step2=step2,
        sample_kwargs={"step": [step1, step2] if nuts_sampler == "pymc" else None},
    )


# ----------------------------------------
# Build/run model
# ----------------------------------------


def inferpymc(
    inv_inputs: xr.Dataset | None = None,
    xprior: dict | None = None,
    bcprior: dict | None = None,
    sigprior: dict | None = None,
    nuts_sampler: str = "pymc",
    nit: int = 20000,
    burn: int = 10000,
    tune: int = 10000,
    nchain: int = 4,
    sigma_per_site: bool = True,
    offsetprior: dict | None = None,
    add_offset: bool = False,
    verbose: bool = False,
    use_bc: bool = True,
    reparameterise_log_normal: bool = False,
    pollution_events_from_obs: bool = False,
    no_model_error: bool = False,
    offset_args: dict | None = None,
    power: dict | float = 1.99,
    sampler_kwargs: dict | None = None,
) -> dict:
    """Perform Bayesian inference with PyMC for emissions, BCs, and model error.

    This routine builds the current component-based PyMC model from
    ``make_inv_inputs`` output, runs sampling, and returns legacy-compatible
    result keys used by downstream postprocessing.

    Args:
        inv_inputs: xarray.Dataset produced by ``make_inv_inputs``.
        xprior: Dictionary describing the prior PDF for emissions. The entry "pdf"
            is the name of the analytical PDF used; other entries are shape
            parameters (e.g., {'pdf': 'lognormal', 'stdev': 1.0}).
        bcprior: Prior specification for boundary conditions. Only used if use_bc is True.
            A common choice is {'pdf': 'truncatednormal', 'lower': 0.0, 'mu': 1.0, 'sigma': 0.1}.
        sigprior: Prior specification for the model-error parameter(s).
        nuts_sampler: Name of the NUTS sampler used by pymc.sample (e.g., "pymc" or "numpyro").
        nit: Total number of iterations (samples + tuning) to draw per chain.
        burn: Number of samples to discard as burn-in.
        tune: Number of tuning steps passed to the sampler.
        nchain: Number of MCMC chains to run.
        sigma_per_site: If True, estimate a separate sigma (model error) for each site.
        offsetprior: Prior specification for offsets applied to sites or observations.
        add_offset: If True, include an offset term in the model.
        verbose: If True, print additional diagnostic information.
        use_bc: If True, include boundary condition terms in the model.
        reparameterise_log_normal: If True, reparameterise log-normal priors for numerical stability.
        pollution_events_from_obs: If True, derive pollution event terms from observations.
        no_model_error: If True, do not include an explicit model-error term.
        offset_args: Additional arguments used when constructing offsets.
        power: Exponent used in certain weighting or prior schemes; may be a dict or float.
        sampler_kwargs: Extra keyword arguments passed to the sampler.

    Returns:
        Dictionary containing inference results, samples, and diagnostics.

        The returned dictionary uses the legacy key structure. Depending on the
        selected options, keys typically include:

        - ``"xouts"``: posterior samples for emissions / fluxes
        - ``"sigouts"``: posterior samples for sigma terms
        - ``"Ytrace"``: modelled concentrations
        - ``"bcouts"`` and ``"YBCtrace"`` when boundary conditions are used
        - ``"OFFSETtrace"`` when offsets are enabled
        - ``"trace"``, ``"model"``, and convergence metadata

        Raises:
            ValueError: If the model cannot be built from the supplied
                ``inv_inputs`` and configuration.
    """
    burn = int(burn)
    nit = int(nit)

    setup = build_inferpymc_model(
        xprior=xprior,
        bcprior=bcprior,
        sigprior=sigprior,
        sigma_per_site=sigma_per_site,
        offsetprior=offsetprior,
        add_offset=add_offset,
        use_bc=use_bc,
        reparameterise_log_normal=reparameterise_log_normal,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
        offset_args=offset_args,
        power=power,
        nuts_sampler=nuts_sampler,
        inv_inputs=inv_inputs,
    )

    sampler_kwargs = sampler_kwargs or {}
    with setup.model:
        trace = pm.sample(
            nit,
            tune=int(tune),
            chains=nchain,
            step=setup.sample_kwargs["step"],
            # progressbar=verbose,
            progressbar=False,
            cores=nchain,
            nuts_sampler=nuts_sampler,
            idata_kwargs={"log_likelihood": True},
            **sampler_kwargs,
        )

    model = setup.model
    step1 = setup.step1
    step2 = setup.step2

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
    r"""Take the output from inferpymc function along with other input information.

    Calculates statistics on them and places it all in a dataset.
    Also calculates statistics on posterior emissions for the countries in
    the inversion domain and saves all in netcdf.

    Note that the uncertainties are defined by the highest posterior
    density (HPD) region and NOT percentiles (as the tdMCMC code).
    The HPD region is defined, for probability content (1-a), as:

        1) P(x ∈ R | y) = (1-a)
        2) for x1 ∈ R and x2 ∉ R, P(x1|y)>=P(x2|y)

    Args:
        xouts: MCMC chain for emissions scaling factors for each basis function.
        sigouts: MCMC chain for model error.
        convergence: Passed/Failed convergence test as to whether multiple chains
            have a Gelman-Rubin diagnostic value <1.05.
        Hx: Transpose of the sensitivity matrix to map emissions to measurement.
            This is the same as what is given from fp_data[site].H.values, where
            fp_data is the output from e.g. footprint_data_merge, but where it
            has been stacked for all sites.
        Y: Measurement vector containing all measurements.
        error: Measurement error vector, containing a value for each element of Y.
        Ytrace: Trace of modelled y values calculated from mcmc outputs and H matrices.
        OFFSETtrace: Trace from offsets (if used).
        step1: Type of MCMC sampler for emissions and boundary condition updates.
        step2: Type of MCMC sampler for model error updates.
        xprior: Dictionary containing information about the prior PDF for emissions.
            The entry "pdf" is the name of the analytical PDF used, see
            https://docs.pymc.io/api/distributions/continuous.html for PDFs
            built into pymc3, although they may have to be coded into the script.
            The other entries in the dictionary should correspond to the shape
            parameters describing that PDF as the online documentation,
            e.g. N(1,1**2) would be: xprior={pdf:"normal", "mu":1, "sigma":1}.
            Note that the standard deviation should be used rather than the
            precision. Currently all variables are considered iid.
        sigprior: Same as xprior but for model error.
        offsetprior: Same as xprior but for bias offset. Only used is add_offset=True.
        Ytime: Time stamp of measurements as used by the inversion.
        siteindicator: Numerical indicator of which site the measurements belong to,
            same length at Y.
        sigma_freq_index: Array of integer indexes that converts time into periods.
        domain: Inversion spatial domain.
        species: Species of interest.
        sites: List of sites in inversion.
        start_date: Start time of inversion "YYYY-mm-dd".
        end_date: End time of inversion "YYYY-mm-dd".
        outputname: Unique identifier for output/run name.
        outputpath: Path to where output should be saved.
        country_unit_prefix: A prefix for scaling the country emissions. Current options are:
            'T' will scale to Tg, 'G' to Gg, 'M' to Mg, 'P' to Pg.
            To add additional options add to acrg_convert.prefix
            Default is none and no scaling will be applied (output in g).
        burn: Number of iterations burned in MCMC
        tune: Number of iterations used to tune step size
        nchain: Number of independent chains run
        sigma_per_site: Whether a model sigma value will be calculated for each site independantly (True)
            or all sites together (False).
        emissions_name: List with "source" values as used when adding emissions data to the OpenGHG object store.
        bcprior: Same as xrpior but for boundary conditions.
        YBCtrace: Trace of modelled boundary condition values calculated from mcmc outputs and Hbc matrices
        bcouts: MCMC chain for boundary condition scaling factors.
        Hbc: Same as Hx but for boundary conditions
        obs_repeatability: Instrument error
        obs_variability: Error from resampling observations
        fp_data: Output from footprints_data_merge + sensitivies
        country_file: Path of country definition file
        add_offset: Add an offset (intercept) to all sites but the first in the site list. Default False.
        rerun_file (xarray dataset, optional): An xarray dataset containing the ncdf output from a previous run of the MCMC code.
        use_bc: When True, use and infer boundary conditions.
        min_error: Minimum error to use during inversion. Only used if no_model_error is False.

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
        flux_time_values = None
    elif emissions_name is None:
        raise ValueError("Emissions name not provided.")
    else:
        emds = fp_data[".flux"][emissions_name[0]]
        flux_array_all = emds.data.flux.values
        if "time" in emds.data.flux.coords:
            flux_time_values = emds.data.flux["time"].values
        elif "flux_time" in emds.data.flux.coords:
            flux_time_values = emds.data.flux["flux_time"].values
        else:
            flux_time_values = None

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
        if flux_time_values is None:
            raise ValueError("Time-varying flux prior requires time coordinates on the flux data.")
        flux_period = utils._infer_flux_period(
            flux_time_values,
            getattr(emds.data.flux, "attrs", {}).get("time_period") if rerun_file is None else None,
        )
        print(f"\nAssuming flux prior is {flux_period}.")
        print(f"Extracting weighted average flux prior from {start_date} to {end_date}")
        month_index = utils._map_times_to_available_period_positions(Ytime, flux_time_values, flux_period)
        apriori_flux = _weighted_apriori_flux_for_months(flux_array_all, month_index)

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
