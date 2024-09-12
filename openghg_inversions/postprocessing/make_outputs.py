from pathlib import Path

import xarray as xr

from .countries import Countries
from .inversion_output import InversionOutput
from ..utils import get_country_file_path


def sample_predictive_distributions(inv_out: InversionOutput) -> None:
    """Sample prior and posterior predictive distributions.

    Prior distributions are sampled as a side effect.

    Updates `inv_out` in place. If these distributions are already present in
    the inversion output, then they are *not* sampled again.
    """
    # TODO: this might be something that is done in the sampling step in `inferpymc` in the future
    dists = ["prior_predictive", "posterior_predictive", "prior"]
    if any(not dist in inv_out.trace for dist in dists):
        inv_out.sample_predictive_distributions()


def make_country_traces(inv_out: InversionOutput, species: str, country_file: str | Path | None = None, domain: str | None = None):
    # TODO: add regions that are aggregates of several countries?
    sample_predictive_distributions(inv_out)  # make sure we have prior distributions

    country_file_path = get_country_file_path(country_file=country_file, domain=domain)
    countries = Countries(xr.open_dataset(country_file_path))

    country_trace = countries.get_country_trace(species=species, inv_out=inv_out)

    return country_trace
