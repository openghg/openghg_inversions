from collections import namedtuple
from collections.abc import Callable

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.utils import add_suffix, get_parameters

Diagnostic = namedtuple("Diagnostic", ["func", "params"])

# this dictionary will be populated by using the decorator `register_stat`
diagnostics: dict[str, Diagnostic] = {}


def register_diagnostic(diagnostic: Callable) -> Callable:
    """Decorator function to register diagnostics functions.

    Args:
        diagnostic: diagnostics function to register

    Returns:
        diagnostic, the input function (no modifications made)
    """
    diagnostics[diagnostic.__name__] = Diagnostic(diagnostic, get_parameters(diagnostic))
    return diagnostic


@register_diagnostic
@add_suffix("trace")
def summary(inv_out: InversionOutput) -> xr.Dataset:
    """Return diagnostics summary computed by arviz.

    Diagnostics reported:
    - `mcse_mean`: mean Monte Carlo standard error
    - `mcse_sd`: standard deviation of Monte Carlo standard error
    - `ess_bulk`: effective sample size (see e.g. Gelman et. al.
      "Bayesian Data Analysis", equation (11.8)) after "rank normalising"
    - `ess_tail`: minimum effective sample size for 5% and 95% quantiles.
    - `r_hat`: the "potential scale reduction", which compares variance within
      chains to pooled variance across chains. If all chains have converged,
      these will be the same and r_hat will be 1. Otherwise, r_hat will be
      greater than 1. Ideally, all r_hat values should be below 1.01

    Args:
        inv_out: InversionOutput to summarise.

    Returns:
        xr.Dataset with diagnostic summary
    """
    return az.summary(inv_out.trace, kind="diagnostics", fmt="xarray")  # type: ignore


def _r2_by_site(ds: xr.Dataset, report_prior: bool = False) -> xr.Dataset:
    """Helper function for computing Bayesian R2 scores."""

    def az_r2_func(arr1: np.ndarray, arr2: np.ndarray) -> pd.Series:
        """Compute r2 values.

        `az.r2_score` will fail if there is no data, so we return NaNs in this case.
        """
        if len(arr1) == 0:
            return pd.Series([np.nan, np.nan])
        return az.r2_score(arr1, arr2)

    def func(ds: xr.Dataset) -> xr.Dataset:
        """Calculate r2 for one site."""
        site = ds.site.values
        ds = ds.squeeze("site", drop=True).dropna("time")

        y_true = ds.y_obs
        y_post_pred = ds.y_posterior_predictive.transpose("draw", "time")

        post_result = xr.apply_ufunc(
            az_r2_func,
            y_true,
            y_post_pred,
            input_core_dims=[["time"], ["draw", "time"]],
            output_core_dims=[["new"]],
        )

        if report_prior:
            y_prior_pred = ds.y_prior_predictive.transpose("draw", "time")

            prior_result = xr.apply_ufunc(
                az_r2_func,
                y_true,
                y_prior_pred,
                input_core_dims=[["time"], ["draw", "time"]],
                output_core_dims=[["new"]],
            )

            result = xr.concat(
                [prior_result.expand_dims(when=["prior"]), post_result.expand_dims(when=["post"])], dim="when"
            )
        else:
            result = post_result

        return result.expand_dims(site=site)

    return ds.groupby("site").map(func).to_dataset("new").rename({0: "r2_bayes", 1: "r2_bayes_std"})


@register_diagnostic
def bayes_r2_by_site(inv_out: InversionOutput, report_prior: bool = False) -> xr.Dataset:
    """Compute Bayesian R2 scores grouped by site.

    Scores are computed for posterior predictive traces (compared
    against true obs).

    Prior R2 scores also be computed, but they can not necessarily be
    compared with the posterior scores, since Bayesian R2 scores are
    normalised to always fall between 0 and 1.

    Args:
        inv_out: InversionOutput object containing obs and trace
        report_prior: if True, return prior R2 in addition to posterior R2

    Returns:
        xr.Dataset containing posterior (and optionally, prior) Bayesian R2 values,
          with uncertainties.

    """
    y_true = inv_out.obs.unstack("nmeasure")
    y_pred = inv_out.get_trace_dataset(var_names="y").unstack("nmeasure")
    ds = xr.merge([y_true, y_pred])

    return _r2_by_site(ds, report_prior=report_prior)


@register_diagnostic
def bayes_r2_by_site_resample(
    inv_out: InversionOutput, freq: str = "MS", report_prior: bool = False
) -> xr.Dataset:
    """Compute Bayesian R2 scores grouped by site and time.

    Scores are computed for posterior predictive traces (compared
    against true obs).

    Prior R2 scores also be computed, but they can not necessarily be
    compared with the posterior scores, since Bayesian R2 scores are
    normalised to always fall between 0 and 1.

    Args:
        inv_out: InversionOutput object containing obs and trace
        freq: frequency to resample to (should be a pandas freq. str that
          can be passed to `xr.Dataset.resample`)
        report_prior: if True, return prior R2 in addition to posterior R2

    Returns:
        xr.Dataset containing posterior (and optionally, prior) Bayesian R2 values,
          with uncertainties.

    """
    y_true = inv_out.obs.unstack("nmeasure")
    y_pred = inv_out.get_trace_dataset(var_names="y").unstack("nmeasure")
    ds = xr.merge([y_true, y_pred])

    results = []
    for time, sub_ds in ds.resample(time=freq):
        results.append(_r2_by_site(sub_ds, report_prior=report_prior).expand_dims(time=[time]))

    return xr.concat(results, dim="time")
