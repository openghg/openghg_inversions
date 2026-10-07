"""Compatibility diagnostic registry and output-facing convergence summary."""

from collections import namedtuple
from collections.abc import Callable

import xarray as xr

from openghg_inversions.inference.diagnostics import posterior_summary
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.metrics import (
    _concentration_trace as _concentration_trace,
    _r2_by_site as _r2_by_site,
    bayes_r2_by_site,
    bayes_r2_by_site_resample,
)
from openghg_inversions.postprocessing.utils import add_suffix, get_parameters

Diagnostic = namedtuple("Diagnostic", ["func", "params"])

# Populated by the compatibility registration decorator below.
diagnostics: dict[str, Diagnostic] = {}


def register_diagnostic(diagnostic: Callable) -> Callable:
    """Decorator function to register diagnostics functions.

    Args:
        diagnostic: diagnostics function to register

    Returns:
        Callable: diagnostic, the input function (no modifications made)
    """
    diagnostics[diagnostic.__name__] = Diagnostic(diagnostic, get_parameters(diagnostic))
    return diagnostic


@register_diagnostic
@add_suffix("trace")
def summary(inv_out: InversionOutput) -> xr.Dataset:
    """Return diagnostics summary computed by arviz.

    Diagnostics reported:
        - mcse_mean: mean Monte Carlo standard error
        - mcse_sd: standard deviation of Monte Carlo standard error
        - ess_bulk: effective sample size (see e.g. Gelman et. al.
          "Bayesian Data Analysis", equation (11.8)) after "rank normalising"
        - ess_tail: minimum effective sample size for 5% and 95% quantiles.
        - r_hat: the "potential scale reduction", which compares variance within
          chains to pooled variance across chains. If all chains have converged,
          these will be the same and r_hat will be 1. Otherwise, r_hat will be
          greater than 1. Ideally, all r_hat values should be below 1.01

    Args:
        inv_out: Inversion output to summarise.

    Returns:
        xr.Dataset: Dataset with diagnostic summary.
    """
    result = posterior_summary(inv_out.trace_group("posterior"))
    metrics = ["mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat"]
    return result.sel(metric=metrics)


# Preserve the existing registry and import names; scientific scores live in metrics.
register_diagnostic(bayes_r2_by_site)
register_diagnostic(bayes_r2_by_site_resample)
