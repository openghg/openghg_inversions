from pathlib import Path
from typing import Literal

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


def make_country_traces(
    inv_out: InversionOutput,
    species: str,
    country_file: str | Path | None = None,
    domain: str | None = None,
    country_regions: dict[str, list[str]] | Path | None = None,
    country_code: Literal["alpha2", "alpha3"] | None = None,
):
    # TODO: species and domain should be available to InversionOutput?
    # TODO: add regions that are aggregates of several countries?
    sample_predictive_distributions(inv_out)  # make sure we have prior distributions

    country_file_path = get_country_file_path(country_file=country_file, domain=domain)
    countries = Countries(xr.open_dataset(country_file_path), country_code=country_code)

    country_trace = countries.get_country_trace(
        species=species, inv_out=inv_out, country_regions=country_regions
    )

    return country_trace


def make_flux_outputs(inv_out: InversionOutput, stats_args: dict | None = None):
    """Return dataset of stats for fluxes and scaling factors."""

    trace = inv_out.get_trace_dataset(unstack_nmeasure=False, var_names="x")

    if stats_args is None:
        stats_args = {}
    stats_args["chunk_dim"] = "nx"
    stats_ds = calculate_stats(trace, **stats_args)

    flux_stats = sparse_xr_dot((inv_out.flux * inv_out.basis), stats_ds)

    for dv in flux_stats.data_vars:
        if dv in stats_ds.data_vars:
            flux_stats[dv].attrs = stats_ds[dv].attrs
            flux_stats[dv].attrs["long_name"] = flux_stats[dv].attrs["long_name"].replace("trace_of_flux_scaling_factor", "flux")
            flux_stats[dv].attrs["units"] = inv_out.flux.attrs.get("units", "mol/m2/s")

    flux_stats = rename_by_replacement(flux_stats, "x", "flux")

    scale_factor_stats = sparse_xr_dot(inv_out.basis, stats_ds)

    for dv in scale_factor_stats.data_vars:
        if dv in stats_ds.data_vars:
            scale_factor_stats[dv].attrs = stats_ds[dv].attrs
            scale_factor_stats[dv].attrs["long_name"] = scale_factor_stats[dv].attrs["long_name"].replace("trace_of_", "")

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

    # TODO: this doesn't always work, e.g. for hdi_68
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
    # TODO: some of these variables could just be stored in a dataset in InversionOutput,
    # rather than in separate data arrays
    to_merge = [
        inv_out.get_obs(),
        inv_out.get_obs_err(),
        inv_out.get_obs_repeatability(),
        inv_out.get_obs_variability(),
        inv_out.get_model_err(),
        inv_out.get_total_err(),
    ]
    result = xr.merge(to_merge)
    result.attrs = {}
    return result


paris_regions_dict = {
    "BELUX": ["BEL", "LUX"],
    "BENELUX": ["BEL", "LUX", "NLD"],
    "CW_EU": [
        "AUT",
        "BEL",
        "CHE",
        "CZE",
        "DEU",
        "ESP",
        "FRA",
        "GBR",
        "HRV",
        "HUN",
        "IRL",
        "ITA",
        "LUX",
        "NLD",
        "POL",
        "PRT",
        "SVK",
        "SVN",
    ],
    "EU_GRP2": ["AUT", "BEL", "CHE", "DEU", "DNK", "FRA", "GBR", "IRL", "ITA", "LUX", "NLD"],
    "NW_EU": ["BEL", "DEU", "DNK", "FRA", "GBR", "IRL", "LUX", "NLD"],
    "NW_EU2": ["BEL", "DEU", "FRA", "GBR", "IRL", "LUX", "NLD"],
    "NW_EU_CONTINENT": ["BEL", "DEU", "FRA", "LUX", "NLD"],
}


def basic_output(
    inv_out: InversionOutput, species: str, country_file: str | Path | None = None, domain: str | None = None
) -> xr.Dataset:
    obs_and_errs = get_obs_and_errors(inv_out)
    conc_outs = make_concentration_outputs(inv_out)
    flux_outs = make_flux_outputs(inv_out)

    country_traces = make_country_traces(
        inv_out,
        species=species,
        country_file=country_file,
        domain=domain,
        country_code="alpha3",
        country_regions=paris_regions_dict,
    )
    country_outs = calculate_stats(country_traces)

    model_data = inv_out.get_model_data(var_names=["hx", "hbc", "min_error"]).rename(
        {"hx": "Hx", "hbc": "Hbc", "min_error": "min_model_error"}
    )

    result = xr.merge([obs_and_errs, conc_outs, flux_outs, country_outs, model_data, inv_out.get_flat_basis()])

    for dv in result.data_vars:
        for k, v in result[dv].attrs.items():
            result[dv].attrs[k] = v.replace("_", " ")

    result.attrs["description"] = "RHIME inversion outputs."

    return result
