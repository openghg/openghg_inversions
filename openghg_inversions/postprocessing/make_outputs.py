from pathlib import Path

import xarray as xr

from openghg_inversions.array_ops import sparse_xr_dot
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.stats import calculate_stats
from openghg_inversions.utils import get_country_file_path


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


def make_replace_names_dict(names: list[str], old: str, new: str) -> dict[str, str]:
    return {name: name.replace(old, new) for name in names}


def rename_by_replacement(ds: xr.Dataset, old: str, new: str) -> xr.Dataset:
    rename_dict = make_replace_names_dict(list(ds.data_vars), old, new)
    return ds.rename(rename_dict)


def make_flux_outputs(inv_out: InversionOutput, stats_args: dict | None = None):
    """Return dataset of stats for fluxes and scaling factors."""

    trace = inv_out.get_trace_dataset(convert_nmeasure=False, var_names="x")

    if stats_args is None:
        stats_args = {}
    stats_args["chunk_dim"] = "nx"
    stats_ds = calculate_stats(trace, **stats_args)

    flux_stats = sparse_xr_dot((inv_out.flux * inv_out.basis), stats_ds)
    flux_stats = rename_by_replacement(flux_stats, "x", "flux")

    scale_factor_stats = sparse_xr_dot(inv_out.basis, stats_ds)
    scale_factor_stats = rename_by_replacement(scale_factor_stats, "x", "flux_scaling")

    return xr.merge([flux_stats, scale_factor_stats])
