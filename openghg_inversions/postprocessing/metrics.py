"""Scientific scores for observation-aligned posterior and prior predictions."""

import numpy as np
import xarray as xr

from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import observation_inputs_for_outputs
from openghg_inversions.postprocessing.stats import combine_chain_draw


def _r2_by_site(ds: xr.Dataset, report_prior: bool = False) -> xr.Dataset:
    """Helper function for computing Bayesian R2 scores."""

    def bayesian_r2(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Compute the former ArviZ ``r2_score`` mean and standard deviation."""
        if len(arr1) == 0:
            return np.array([np.nan, np.nan])
        variance_estimate = np.var(arr2, axis=1)
        variance_residual = np.var(arr1 - arr2, axis=1)
        samples = variance_estimate / (variance_estimate + variance_residual)
        return np.array([np.mean(samples), np.std(samples)])

    def func(ds: xr.Dataset) -> xr.Dataset:
        """Calculate r2 for one site."""
        site = ds.site.values
        ds = ds.squeeze("site", drop=True).dropna("time")

        y_true = ds.y_obs
        y_post_pred = ds.y_posterior_predictive.dropna("draw", how="all").transpose("draw", "time")

        post_result = xr.apply_ufunc(
            bayesian_r2,
            y_true,
            y_post_pred,
            input_core_dims=[["time"], ["draw", "time"]],
            output_core_dims=[["new"]],
        )

        if report_prior:
            y_prior_pred = ds.y_prior_predictive.dropna("draw", how="all").transpose("draw", "time")

            prior_result = xr.apply_ufunc(
                bayesian_r2,
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


def _concentration_trace(inv_out: InversionOutput) -> xr.Dataset:
    """Return concentration traces using diagnostic product names."""
    trace = inv_out.trace_dataset(var_roles="concentration")
    concentration_name = inv_out.variable_name("concentration")
    trace = trace.rename(
        {
            data_var: str(data_var).replace(f"{concentration_name}_", "y_", 1)
            for data_var in trace.data_vars
            if str(data_var).startswith(f"{concentration_name}_")
        }
    )
    trace, sample_dim = combine_chain_draw(trace)
    if sample_dim != "draw":
        trace = trace.rename({sample_dim: "draw"})
    return trace


def bayes_r2_by_site(inv_out: InversionOutput, report_prior: bool = False) -> xr.Dataset:
    """Compute Bayesian R2 scores grouped by site.

    Scores are computed for posterior predictive traces (compared
    against true obs).

    Prior R2 scores also be computed, but they can not necessarily be
    compared with the posterior scores, since Bayesian R2 scores are
    normalised to always fall between 0 and 1.

    Args:
        inv_out: Inversion output containing obs and trace
        report_prior: if True, return prior R2 in addition to posterior R2

    Returns:
        xr.Dataset: containing posterior (and optionally, prior) Bayesian R2 values,
            with uncertainties.

    """
    y_true = observation_inputs_for_outputs(inv_out)["y_obs"].unstack("nmeasure")
    y_pred = _concentration_trace(inv_out).unstack("nmeasure")
    ds = xr.merge([y_true, y_pred])

    return _r2_by_site(ds, report_prior=report_prior)


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
        inv_out: Inversion output containing obs and trace
        freq: frequency to resample to (should be a pandas freq. str that
          can be passed to `xr.Dataset.resample`)
        report_prior: if True, return prior R2 in addition to posterior R2

    Returns:
        xr.Dataset: containing posterior (and optionally, prior) Bayesian R2 values,
            with uncertainties.

    """
    y_true = observation_inputs_for_outputs(inv_out)["y_obs"].unstack("nmeasure")
    y_pred = _concentration_trace(inv_out).unstack("nmeasure")
    ds = xr.merge([y_true, y_pred])

    results = []
    for time, sub_ds in ds.resample(time=freq):
        results.append(_r2_by_site(sub_ds, report_prior=report_prior).expand_dims(time=[time]))

    return xr.concat(results, dim="time")
