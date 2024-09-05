from pathlib import Path

import xarray as xr

from .countries import Countries
from .inversion_output import InversionOutput
from ..utils import get_country_file_path

def make_rhime_outputs(inv_out: InversionOutput, species: str, country_file: str | Path | None = None, domain: str | None = None):
    # sample predictive distributions
    ndraw = inv_out.trace.posterior.sizes["draw"]  # type: ignore
    inv_out.sample_predictive_distributions(ndraw=ndraw)

    country_file_path = get_country_file_path(country_file=country_file, domain=domain)
    countries = Countries(xr.open_dataset(country_file_path))

    country_trace = countries.get_country_trace(species=species, inv_out=inv_out)
