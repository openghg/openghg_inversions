from collections import namedtuple
from typing import Callable

import arviz as az
import numpy as np
import xarray as xr

from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.utils import add_suffix, get_parameters

Diagnostic = namedtuple("Diagnostic", ["func", "params"])

# this dictionary will be populated by using the decorator `register_stat`
diagnostics: dict[str, Diagnostic] = {}


def register_diagnostic(diagnostic: Callable) -> Callable:
    """Decorator function to register stats functions.

    Args:
        stat: stats function to register

    Returns:
        stat, the input function (no modifications made)
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

    Args:
        inv_out: InversionOutput to summarise.

    Returns:
        xr.Dataset with diagnostic summary
    """
    return az.summary(inv_out.trace, kind="diagnostics", fmt="xarray")  # type; ignore


@register_diagnostic
def r2_bayes(inv_out: InversionOutput) -> xr.Dataset:
    """Calculate Bayesian r2 score by site."""
    def func(y_true, y_pred):
        scores_by_site = []
        for i in range(y_true.shape[0]):
            # select site i
            y_true_i = y_true[i,...]
            y_pred_i = y_pred[i,...].T

            # remove NaNs (usually caused by unstacking `nmeasure`)
            filt = np.isfinite(y_true_i)
            y_true_i = np.where(filt, y_true_i, 0)
            y_pred_i = np.where(filt, y_pred_i, 0)

            # calculate r2 score
            r2 = az.r2_score(y_true_i, y_pred_i)
            scores_by_site.append(r2)
        return np.vstack(scores_by_site)

    y_pred = inv_out.get_trace_dataset(var_names="y").y_posterior_predictive
    result = xr.apply_ufunc(func, inv_out.get_obs(), y_pred, input_core_dims=[["time"], ["time", "draw"]], output_core_dims=[["new"]])

    # unstack new dim added by az.r2_score to two data variables
    result = result.to_dataset("new").rename({0: "r2_bayes", 1: "r2_bayes_std"})
    return result
