from collections import namedtuple
from typing import Callable

import arviz as az
import xarray as xr

from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.utils import add_suffix, get_parameters

Diagnostic = namedtuple("StatsFunction", ["func", "params"])

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

    Args:
        inv_out: InversionOutput to summarise.

    Returns:
        xr.Dataset
    """
    return az.summary(inv_out.trace, kind="diagnostics", fmt="xarray")  # type; ignore
