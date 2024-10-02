from pathlib import Path

import xarray as xr

from openghg_inversions.array_ops import sparse_xr_dot
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.stats import calculate_stats
from openghg_inversions.postprocessing.utils import rename_by_replacement
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
    scale_factor_stats = rename_by_replacement(scale_factor_stats, "x", "scaling")

    return xr.merge([flux_stats, scale_factor_stats])


def flatten_post_prior(ds: xr.Dataset) -> xr.Dataset:
    """Add a dimension `when` that is either "post" or "prior".

    This reduces the number of data variables in the outputs.

    Note: do this before flattening suffixes.
    """
    ds_list = []
    dvs_list = []
    for coord, when in [("post", "posterior"), ("prior", "prior")]:
        dvs = [str(dv) for dv in ds.data_vars if when in dv]
        dvs_list.extend(dvs)
        # select either "posterior" or "prior" vars, remove those from the variable names
        # then add "post" or "prior" as a coordinate for the dimension "when"
        ds_list.append(rename_by_replacement(ds[dvs], f"{when}_", "").expand_dims({"when": [coord]}))

    result_ds = xr.concat(ds_list, dim="when")

    # check if any data vars have been left out, and add them to the result
    other_dvs = [str(dv) for dv in ds.data_vars if str(dv) not in dvs_list]
    if other_dvs:
        result_ds = xr.merge([result_ds, ds[other_dvs]])

    return result_ds


def convert_suffixes_to_dim(ds: xr.Dataset, suffixes: list[str], new_dim: str) -> xr.Dataset:
    ds_list = []
    dvs_list = []
    for suff in suffixes:
        dvs = [str(dv) for dv in ds.data_vars if str(dv).endswith(suff)]
        dvs_list.extend(dvs)
        # select either "posterior" or "prior" vars, remove those from the variable names
        # then add "post" or "prior" as a coordinate for the dimension "when"
        ds_list.append(rename_by_replacement(ds[dvs], f"_{suff}", "").expand_dims({new_dim: [suff]}))

    result_ds = xr.concat(ds_list, dim=new_dim)

    # check if any data vars have been left out, and add them to the result
    other_dvs = [str(dv) for dv in ds.data_vars if str(dv) not in dvs_list]
    if other_dvs:
        result_ds = xr.merge([result_ds, ds[other_dvs]])

    return result_ds


def sort_data_vars(ds: xr.Dataset) -> xr.Dataset:
    """Sort data variables by variable name, then suffix."""

    def sort_key(s: str):
        s_split = s.rsplit("_", maxsplit=1)
        if len(s_split) == 1:
            s_split.append("")
        return tuple(s_split)

    dv_sorted = sorted(list(ds.data_vars), key=sort_key)
    return ds[dv_sorted]  # type: ignore


def make_concentration_outputs(inv_out: InversionOutput, stats_args: dict | None = None):
    conc_vars = ["y", "mu_bc"] if "mu_bc" in inv_out.trace.posterior else ["y"]
    trace = inv_out.get_trace_dataset(var_names=conc_vars)

    if stats_args is None:
        stats_args = {}

    stats_args["chunk_dim"] = "nmeasure"
    stats_args["chunk_size"] = 1

    conc_stats = calculate_stats(trace, **stats_args)

    return conc_stats


def get_obs_and_errors(inv_out: InversionOutput) -> xr.Dataset:
    to_merge = [
        inv_out.get_obs(),
        inv_out.get_obs_err(),
        inv_out.get_obs_repeatability(),
        inv_out.get_obs_variability(),
        inv_out.get_model_err(),
        inv_out.get_total_err(),
    ]
    return xr.merge(to_merge)
