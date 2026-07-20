"""Functions for performing MCMC inversion with the xarray-first PyMC builder."""

import re
import getpass
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np

# Configure PyTensor before importing PyMC.
from openghg_inversions._pymc_config import configure_pytensor

configure_pytensor()

import pymc as pm  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
import arviz as az  # noqa: E402
from scipy import stats  # noqa: E402

from openghg_inversions import convert  # noqa: E402
from openghg_inversions import utils  # noqa: E402
from openghg_inversions._sampling import _reset_retained_draws  # noqa: E402
from openghg_inversions.hbmcmc.hbmcmc_output import define_output_filename  # noqa: E402
from openghg_inversions.config.version import code_version  # noqa: E402
from openghg_inversions.models import build_rhime_model  # noqa: E402
from openghg_inversions.models.components import resolve_model_variable  # noqa: E402
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords  # noqa: E402
from openghg_inversions.models.priors import PriorArgs  # noqa: E402
from openghg_inversions.inversion_inputs import _compact_integer_index  # noqa: E402
from openghg_inversions.sigma import SigmaAlignment  # noqa: E402

# ----------------------------------------
# Model building code
# ----------------------------------------

# Defaults to avoid mutable default arguments in model building functions.
DEFAULT_XPRIOR: PriorArgs = {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0, "reparameterise": True}
DEFAULT_BCPRIOR: PriorArgs = {"pdf": "truncatednormal", "mu": 1.0, "sigma": 0.05, "lower": 0.0}
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
        warnings.warn(
            "`reparameterise_log_normal` is deprecated. Set `reparameterise=True` in the relevant prior args instead.",
            FutureWarning,
            stacklevel=2,
        )
        if str(prepared_xprior.get("pdf", "")).lower() == "lognormal":
            prepared_xprior["reparameterise"] = True
        if str(prepared_bcprior.get("pdf", "")).lower() == "lognormal":
            prepared_bcprior["reparameterise"] = True

    return prepared_xprior, prepared_bcprior, prepared_sigprior, prepared_offsetprior


def build_inferpymc_model(
    inv_inputs: xr.Dataset,
    *,
    xprior: dict | None = None,
    bcprior: dict | None = None,
    sigprior: dict | None = None,
    sigma_per_site: bool = True,
    offsetprior: dict | None = None,
    add_offset: bool = False,
    use_bc: bool = True,
    reparameterise_log_normal: bool = False,
    pollution_events_from_obs: bool = False,
    no_model_error: bool = False,
    offset_args: dict | None = None,
    power: dict | float = 1.99,
) -> pm.Model:
    """Compatibility adapter for the standard RHIME model builder.

    Args:
        inv_inputs: Legacy dataset produced by
            ``prepare_fixedbasis_inversion_data`` or an equivalent adapter.
            It must contain the observation and model variables required by
            the component-based model, including at minimum ``H``, ``mf``,
            ``mf_error``, ``site_indicator``, ``sigma_freq_index``, and
            ``min_error``. When ``use_bc`` is true, it must also contain
            ``H_bc``.
        xprior: Prior specification for emissions scaling factors.
        bcprior: Prior specification for boundary-condition scaling factors.
        sigprior: Prior specification for model-error terms.
        sigma_per_site: Whether sigma should vary by site.
        offsetprior: Prior specification for optional offsets.
        add_offset: Whether to include an offset term in the model.
        use_bc: Whether to include boundary-condition terms in the model.
        reparameterise_log_normal: Deprecated compatibility flag for lognormal
            reparameterisation. Set ``reparameterise=True`` in the relevant
            prior mapping instead.
        pollution_events_from_obs: Whether to derive pollution-event scaling
            from observations rather than modelled concentrations.
        no_model_error: Whether to suppress the explicit model-error term.
        offset_args: Extra keyword arguments forwarded to
            ``add_offset_component``.
        power: Exponent or prior specification used in the likelihood error
            scaling.

    Returns:
        Built PyMC model for the current inferpymc compatibility path.

    Raises:
        ValueError: If modern ``fixed_baseline`` data are supplied to this
            legacy compatibility builder.

    Warns:
        FutureWarning: If ``reparameterise_log_normal`` is enabled.
    """
    if "fixed_baseline" in inv_inputs:
        raise ValueError(
            "The legacy HBMCMC model builder does not support fixed_baseline; "
            "use the modern RHIME model builder instead."
        )

    xprior, bcprior, sigprior, offsetprior = _prepare_builder_priors(
        xprior=xprior,
        bcprior=bcprior,
        sigprior=sigprior,
        offsetprior=offsetprior,
        reparameterise_log_normal=reparameterise_log_normal,
    )
    sigma_alignment = SigmaAlignment.from_indices(
        inv_inputs["site_indicator"],
        inv_inputs["sigma_freq_index"],
        per_site=sigma_per_site,
    )

    return build_rhime_model(
        inv_inputs,
        sigma_alignment=sigma_alignment,
        x_prior=xprior,
        bc_prior=bcprior,
        sigma_prior=sigprior,
        offset_prior=offsetprior,
        add_offset=add_offset,
        use_bc=use_bc,
        pollution_events_from_obs=pollution_events_from_obs,
        no_model_error=no_model_error,
        offset_args=offset_args,
        power=power,
    )


# ----------------------------------------
# Build/run model
# ----------------------------------------


def extend_inferencedata_predictive(
    trace: az.InferenceData,
    *,
    model: pm.Model,
    sample_prior_predictive: bool | int = False,
    sample_posterior_predictive: bool | list[str] = False,
) -> az.InferenceData:
    """Extend an InferenceData trace with optional predictive groups.

    Updates InferenceData in-place with requested groups and returns the
    result for convenience.

    Args:
        trace: Posterior trace to extend with predictive groups.
        model: Built PyMC model used for predictive sampling.
        sample_prior_predictive: If truthy, sample prior predictive draws and
            append ``prior`` and ``prior_predictive`` groups. If an integer,
            use that many draws; if ``True``, reuse the posterior draw count.
        sample_posterior_predictive: If truthy, sample posterior predictive
            draws and append ``posterior_predictive``. If a list, restrict
            posterior predictive sampling to those variable names.

    Returns:
        Input ``trace`` extended with the requested predictive groups.
    """
    if sample_prior_predictive:
        prior_draws = (
            trace.posterior.sizes["draw"] if sample_prior_predictive is True else int(sample_prior_predictive)
        )
        with model:
            trace.extend(pm.sample_prior_predictive(prior_draws, model))

    if sample_posterior_predictive:
        posterior_var_names = (
            None if sample_posterior_predictive is True else list(sample_posterior_predictive)
        )
        with model:
            trace.extend(pm.sample_posterior_predictive(trace, model=model, var_names=posterior_var_names))

    return trace


def sample(
    model: pm.Model,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    burn: int = 0,  # TODO: add sensible defaults
    sample_prior_predictive: bool | int = False,
    sample_posterior_predictive: bool | list[str] = False,
    **kwargs: Any,
) -> az.InferenceData:
    """Sample from a built inferpymc model.

    Args:
        model: Built PyMC model to sample from.
        draws: Number of posterior draws requested per chain before burn
            slicing.
        tune: Number of tuning draws passed to ``pm.sample``.
        chains: Number of MCMC chains to run.
        burn: Number of posterior draws to discard from the returned
            ``InferenceData``.
        sample_prior_predictive: Optional prior predictive sampling request.
            If an integer, use that many draws; if ``True``, reuse the
            posterior draw count.
        sample_posterior_predictive: Optional posterior predictive sampling
            request. If a list, restrict sampling to those variable names.
        **kwargs: Additional keyword arguments forwarded to ``pm.sample``.
            ``return_inferencedata`` is always forced to ``True`` and
            ``idata_kwargs["log_likelihood"]`` is always enabled.

    Returns:
        Burn-sliced ``InferenceData`` for the requested model, optionally
        extended with predictive groups. Retained draw coordinates are reset
        to consecutive zero-based integers, and ``burn`` is stored on the root
        and draw-bearing group attributes.
    """
    sample_kwargs = dict(kwargs)
    sample_kwargs.pop("return_inferencedata", None)
    idata_kwargs = dict(sample_kwargs.pop("idata_kwargs", {}))
    idata_kwargs["log_likelihood"] = True

    with model:
        raw_trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            return_inferencedata=True,
            idata_kwargs=idata_kwargs,
            **sample_kwargs,
        )

    burned_trace = raw_trace.isel(draw=slice(burn, None))
    burned_trace = _reset_retained_draws(cast(az.InferenceData, burned_trace), burn=burn)
    burned_trace = extend_inferencedata_predictive(
        burned_trace,
        model=model,
        sample_prior_predictive=sample_prior_predictive,
        sample_posterior_predictive=sample_posterior_predictive,
    )
    registry = get_coord_registry(model)
    if registry is not None:
        burned_trace = restore_inferencedata_coords(burned_trace, registry)

    nuts_sampler = sample_kwargs.get("nuts_sampler", "pymc")
    if nuts_sampler != "pymc" and sample_kwargs.get("compute_convergence_checks", True):
        if "sample_stats" in burned_trace and "diverging" in burned_trace.sample_stats:
            divergences = np.sum(burned_trace.sample_stats.diverging).values
            if divergences > 0:
                warnings.warn(
                    f"There were {divergences} divergences. Try increasing target accept or reparameterise.",
                    UserWarning,
                    stacklevel=2,
                )

    return burned_trace


# ------------------------------------------------------------
# Legacy compatibility helpers
# ------------------------------------------------------------


def _rename_trace_for_legacy_inferpymc(trace: az.InferenceData) -> az.InferenceData:
    """Return a legacy-compatible trace view with inferpymc dim names.

    Note:
        Legacy adapter code. This helper converts canonical modern trace
        dimension names into the legacy inferpymc naming expected by
        downstream compatibility code.

    Args:
        trace: Canonical ``InferenceData`` returned by the modern sampling
            path.

    Returns:
        A copied ``InferenceData`` whose groups use the legacy inferpymc
        dimension names where required. Root and group attributes are
        preserved.
    """
    rename_map = {"region": "nx", "bc_region": "nbc"}
    renamed_groups: dict[str, xr.Dataset] = {}

    for group in trace.groups():
        ds = trace[group]
        applicable = {old: new for old, new in rename_map.items() if old in ds.dims or old in ds.coords}
        renamed_groups[group] = ds.rename(applicable) if applicable else ds.copy()

    return cast(Any, az.InferenceData)(attrs=dict(trace.attrs), **renamed_groups)


def _adapt_legacy_inferpymc_results(
    *,
    trace: az.InferenceData,
    model: pm.Model,
    use_bc: bool,
    add_offset: bool,
    sample_kwargs: dict[str, Any],
) -> dict:
    """Adapt modern sampling outputs into the legacy inferpymc return structure.

    Note:
        Legacy adapter code. This helper is the compatibility boundary between
        the modern ``InferenceData``-first sampling path and the legacy
        inferpymc dict-of-arrays return contract.

    Args:
        trace: Canonical ``InferenceData`` returned by the modern sampling
            path.
        model: Built PyMC model used for sampling.
        use_bc: Whether boundary-condition terms are enabled.
        add_offset: Whether offset terms are enabled.
        sample_kwargs: Sampling keyword arguments actually used by the legacy
            compatibility run.

    Returns:
        Dictionary matching the legacy inferpymc return contract.
    """
    legacy_trace = _rename_trace_for_legacy_inferpymc(trace)
    posterior = legacy_trace.posterior.isel(chain=0, drop=True)

    xouts = posterior.x
    sigouts = posterior.sigma

    if use_bc:
        bcouts = posterior.bc

    gelrub = pm.rhat(legacy_trace)["x"].max()
    if gelrub > 1.05:
        print("Failed Gelman-Rubin at 1.05")
        convergence = "Failed"
    else:
        convergence = "Passed"

    if add_offset:
        offset_trace = posterior.offset
    else:
        offset_trace = xr.zeros_like(posterior.mu)

    if use_bc:
        ybc_trace = posterior.mu_bc + offset_trace
        y_trace = posterior.mu + ybc_trace
    else:
        y_trace = posterior.mu + offset_trace

    step1, step2 = sample_kwargs.get("step", (None, None))

    result = {
        "xouts": xouts,
        "sigouts": sigouts,
        "Ytrace": y_trace.values.T,
        "OFFSETtrace": offset_trace.values.T,
        "convergence": convergence,
        "step1": step1,
        "step2": step2,
        "model": model,
        "trace": legacy_trace,
    }

    if use_bc:
        result["bcouts"] = bcouts
        result["YBCtrace"] = ybc_trace.values.T

    return result


def inferpymc(
    inv_inputs: xr.Dataset,
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

    This routine is the compatibility entrypoint for the current PyMC path.
    It builds the component-based model from a legacy inversion-input dataset,
    runs sampling, and adapts the result into the return structure used by
    downstream fixedbasis postprocessing. The input must include
    ``sigma_freq_index``; modern RHIME inputs intentionally do not.

    Args:
        inv_inputs: Legacy fixedbasis inversion inputs including an
            observation-aligned ``sigma_freq_index``.
        xprior: Dictionary describing the prior PDF for emissions. The entry "pdf"
            is the name of the analytical PDF used; other entries are shape
            parameters (e.g., {'pdf': 'lognormal', 'stdev': 1.0}).
        bcprior: Prior specification for boundary conditions. Only used if use_bc is True.
            A common choice is {'pdf': 'truncatednormal', 'lower': 0.0, 'mu': 1.0, 'sigma': 0.1}.
        sigprior: Prior specification for the model-error parameter(s).
        nuts_sampler: Name of the NUTS sampler used by pymc.sample (e.g., "pymc" or "numpyro").
        nit: Number of posterior draws to keep per chain. Tuning draws are
            controlled separately via ``tune``.
        burn: Number of samples to discard as burn-in.
        tune: Number of tuning steps passed to the sampler.
        nchain: Number of MCMC chains to run.
        sigma_per_site: If True, estimate a separate sigma (model error) for each site.
        offsetprior: Prior specification for offsets applied to sites or observations.
        add_offset: If True, include an offset term in the model.
        verbose: If True, print additional diagnostic information.
        use_bc: If True, include boundary condition terms in the model.
        reparameterise_log_normal: Deprecated compatibility flag for lognormal
            reparameterisation. Set ``reparameterise=True`` in the relevant
            prior mapping instead.
        pollution_events_from_obs: If True, derive pollution event terms from observations.
        no_model_error: If True, do not include an explicit model-error term.
        offset_args: Additional arguments used when constructing offsets.
        power: Exponent used in certain weighting or prior schemes; may be a dict or float.
        sampler_kwargs: Extra keyword arguments passed to the sampler.

    Returns:
        Dictionary containing inference results, samples, and diagnostics in
        the legacy ``inferpymc`` key structure. Depending on the selected
        options, keys typically include:

        - ``"xouts"``: posterior samples for emissions / fluxes
        - ``"sigouts"``: posterior samples for sigma terms
        - ``"Ytrace"``: modelled concentrations
        - ``"bcouts"`` and ``"YBCtrace"`` when boundary conditions are used
        - ``"OFFSETtrace"`` when offsets are enabled
        - ``"trace"``, ``"model"``, and convergence metadata

    Raises:
        ValueError: If the model cannot be built from the supplied
            ``inv_inputs`` and configuration.

    Warns:
        FutureWarning: If ``reparameterise_log_normal`` is enabled.
    """
    burn = int(burn)
    nit = int(nit)

    model = build_inferpymc_model(
        inv_inputs,
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
    )

    sampler_kwargs = sampler_kwargs or {}

    # add steps for pymc sampler
    if nuts_sampler == "pymc":
        with model:
            latent_vars = tuple(
                variable
                for variable in (resolve_model_variable(model, "x"), resolve_model_variable(model, "bc"))
                if variable is not None
            )
            sampler_kwargs["step"] = [
                pm.NUTS(latent_vars),
                pm.Slice([resolve_model_variable(model, "sigma")]),
            ]

    sampler_kwargs.setdefault("progressbar", False)
    sampler_kwargs.setdefault("cores", nchain)

    trace = sample(
        model,
        draws=nit,
        burn=burn,
        tune=int(tune),
        chains=nchain,
        sample_prior_predictive=True,
        sample_posterior_predictive=["y"],
        nuts_sampler=nuts_sampler,
        **sampler_kwargs,
    )

    return _adapt_legacy_inferpymc_results(
        trace=trace,
        model=model,
        use_bc=use_bc,
        add_offset=add_offset,
        sample_kwargs=sampler_kwargs,
    )


# ------------------------------------------------------------
# Legacy post-processing
# ------------------------------------------------------------


def _weighted_apriori_flux_for_months(flux_array_all: np.ndarray, month_index: np.ndarray) -> np.ndarray:
    """Compute a weighted prior flux average using compacted month positions."""
    month_index = _compact_integer_index(month_index)
    apriori_flux = np.zeros_like(flux_array_all[:, :, 0])

    for month_pos in np.unique(month_index):
        apriori_flux += flux_array_all[:, :, month_pos] * np.sum(month_index == month_pos) / len(month_index)

    return apriori_flux


def _hdi_for_parameter_traces(trace: np.ndarray, hdi_prob: float) -> np.ndarray:
    """Return HDI intervals for traces shaped as ``(parameter, draw)``."""
    return az.hdi(trace.T[np.newaxis, :, :], hdi_prob=hdi_prob)


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
    emissions_name: list[str] | None,
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

    Ymod95OFF = _hdi_for_parameter_traces(OFFSETtrace, hdi_prob=0.95)
    Ymod68OFF = _hdi_for_parameter_traces(OFFSETtrace, hdi_prob=0.68)

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

        Ymod95BC = _hdi_for_parameter_traces(YBCtrace, hdi_prob=0.95)
        Ymod68BC = _hdi_for_parameter_traces(YBCtrace, hdi_prob=0.68)
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

    Ymod95 = _hdi_for_parameter_traces(Ytrace, hdi_prob=0.95)
    Ymod68 = _hdi_for_parameter_traces(Ytrace, hdi_prob=0.68)

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
    scalemap_mu = np.zeros_like(bfds.values, dtype=float)
    scalemap_mode = np.zeros_like(bfds.values, dtype=float)

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
        cntry68[ci, :] = az.hdi(cntrytottrace.values, hdi_prob=0.68)
        cntry95[ci, :] = az.hdi(cntrytottrace.values, hdi_prob=0.95)
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
