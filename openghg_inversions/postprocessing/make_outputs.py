from pathlib import Path
from typing import Literal

import xarray as xr

from openghg_inversions.array_ops import sparse_xr_dot
from openghg_inversions.postprocessing.countries import Countries, paris_regions_dict
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.stats import calculate_stats
from openghg_inversions.postprocessing.utils import rename_by_replacement


def make_flux_outputs(
    inv_out: InversionOutput,
    stats: list[str] | None = None,
    stats_args: dict | None = None,
    include_scale_factors: bool = True,
    report_flux_on_inversion_grid: bool = True,
) -> xr.Dataset:
    """Return dataset of stats for fluxes and scaling factors.

    Args:
        inv_out: InversionOutput containing MCMC traces.
        stats: list of stats to use. If `None`, the default for
          `calculate_stats` is used, which is "mean" and "quantiles". See the
          `postprocessing.stats` submodule for more options.
        stats_args: dict of arguments to be passed to stats functions. If a key
          in this dict is the name of an argument for a stats function, then the
          value for this key will be passed to the stats function. To pass an
          option to a specific stats function, write the key in the form `<stat
          function name>__<key>`, with a double underscore.
            For instance, `stats_args = {"mode_kde__chunk_size": 20}` would pass the
          argument `chunk_size = 20` to the stat function `mode_kde`, and no others.
        include_scale_factors: If True, report stats for scale factors, in
          addition to stats for fluxes (which are calculated by transforming the
          scale factor stats to the lat/lon grid using the prior flux).
        report_flux_on_inversion_grid: If True, report fluxes by basis function,
          without incorporating the prior flux. Note: we do not actually optimise
          for this quantity, since the prior flux is used in the forward model.

    Returns:
        xr.Dataset with computed flux stats.

    """
    trace = inv_out.get_trace_dataset(var_names="x")

    if stats_args is None:
        stats_args = {}

    if stats is not None:
        stats_args["stats"] = stats

    stats_args["chunk_dim"] = "nx"
    stats_ds = calculate_stats(trace, **stats_args)

    if report_flux_on_inversion_grid:
        agg_flux = (
            (inv_out.basis * inv_out.flux).sum(["lat", "lon"]) / inv_out.basis.sum(["lat", "lon"])
        ).fillna(0.0)
        flux_stats = sparse_xr_dot(inv_out.basis, agg_flux * stats_ds)
    else:
        flux_stats = sparse_xr_dot((inv_out.flux * inv_out.basis), stats_ds)

    for dv in flux_stats.data_vars:
        if dv in stats_ds.data_vars:
            flux_stats[dv].attrs = stats_ds[dv].attrs
            flux_stats[dv].attrs["long_name"] = (
                flux_stats[dv].attrs["long_name"].replace("trace_of_flux_scaling_factor", "flux")
            )
            flux_stats[dv].attrs["units"] = inv_out.flux.attrs.get("units", "mol/m2/s")

    flux_stats = rename_by_replacement(flux_stats, "x", "flux")

    if include_scale_factors:
        scale_factor_stats = sparse_xr_dot(inv_out.basis, stats_ds)

        for dv in scale_factor_stats.data_vars:
            if dv in stats_ds.data_vars:
                scale_factor_stats[dv].attrs = stats_ds[dv].attrs
                scale_factor_stats[dv].attrs["long_name"] = (
                    scale_factor_stats[dv].attrs["long_name"].replace("trace_of_", "")
                )

        scale_factor_stats = rename_by_replacement(scale_factor_stats, "x", "scaling")

        flux_stats = xr.merge([flux_stats, scale_factor_stats])

    return flux_stats.as_numpy()


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


def make_concentration_outputs(
    inv_out: InversionOutput,
    stats: list[str] | None = None,
    stats_args: dict | None = None,
) -> xr.Dataset:
    """Return dataset of stats for concentrations.

    Args:
        inv_out: InversionOutput containing MCMC traces.
        stats: list of stats to use. If `None`, the default for
          `calculate_stats` is used, which is "mean" and "quantiles". See the
          `postprocessing.stats` submodule for more options.
        stats_args: dict of arguments to be passed to stats functions. If a key
          in this dict is the name of an argument for a stats function, then the
          value for this key will be passed to the stats function. To pass an
          option to a specific stats function, write the key in the form `<stat
          function name>__<key>`, with a double underscore.
            For instance, `stats_args = {"mode_kde__chunk_size": 20}` would pass the
          argument `chunk_size = 20` to the stat function `mode_kde`, and no others.

    Returns:
        xr.Dataset with computed flux stats.

    """
    conc_vars = ["y", "mu_bc"] if "mu_bc" in inv_out.trace.posterior else ["y"]
    trace = inv_out.get_trace_dataset(var_names=conc_vars)

    if stats_args is None:
        stats_args = {}

    if stats is not None:
        stats_args["stats"] = stats

    stats_args["chunk_dim"] = "nmeasure"
    stats_args["chunk_size"] = 1

    conc_stats = calculate_stats(trace, **stats_args)

    return conc_stats


def make_country_outputs(
    inv_out: InversionOutput,
    country_file: str | Path | None = None,
    country_regions: str | Path | dict[str, list[str]] | Literal["paris"] | None = None,
    stats: list[str] | None = None,
    stats_args: dict | None = None,
    country_code: Literal["alpha2", "alpha3"] | None = "alpha3"
) -> xr.Dataset:
    """Calculate country emission stats.

    Args:
        inv_out: InversionOutput containing MCMC traces
        country_file: path to country definition file. If `None`, the default
          country file location and the domain of the InversionOutput will be used
          to try to find a suitable country file.
        country_regions: dict mapping country region names (e.g. "BENELUX") to a
          list of (country codes) of the countries comprising that regions (e.g.
          `["BEL", "NLD", "LUX"]`).
        stats: list of stats to use. If `None`, the default for
          `calculate_stats` is used, which is "mean" and "quantiles". See the
          `postprocessing.stats` submodule for more options.
        stats_args: dict of arguments to be passed to stats functions. If a key
          in this dict is the name of an argument for a stats function, then the
          value for this key will be passed to the stats function. To pass an
          option to a specific stats function, write the key in the form `<stat
          function name>__<key>`, with a double underscore.
            For instance, `stats_args = {"mode_kde__chunk_size": 20}` would pass the
          argument `chunk_size = 20` to the stat function `mode_kde`, and no others.
        country_code: If set to "alpha2" or "alpha3", country names will be
          converted to two or three digit country codes, respectively. Country
          region definitions should use the same type of code as specified here.

    Returns:
        xr.Dataset containing statistics for the specified countries and regions.

    """
    if country_regions == "paris":
        country_regions = paris_regions_dict[inv_out.domain.lower()]
    elif isinstance(country_regions, str):
        country_regions = Path(country_regions)

    countries = Countries.from_file(
        country_file=country_file, country_code=country_code, country_regions=country_regions, domain=inv_out.domain
    )
    country_traces = countries.get_country_trace(inv_out=inv_out)

    if stats_args is None:
        stats_args = {}
    if stats is not None:
        stats_args["stats"] = stats

    country_stats = calculate_stats(country_traces, **stats_args)

    return country_stats.as_numpy()


def basic_output(
    inv_out: InversionOutput,
    country_file: str | Path | None = None,
    country_regions: str | Path | dict[str, list[str]] | Literal["paris"] | None = None,
    stats: list[str] | None = None,
    stats_args: dict | None = None
) -> xr.Dataset:
    """Create basic output with concentrations, flux totals, and country totals.

    The dataset returned also contains the basis functions, and other data used
    to create the model, like "H matrices" and the flux used.

    Args:
        inv_out: InversionOutput to process
        country_file: path to country file
        country_regions: optional country regions to use. If "paris" is passed,
        then the PARIS regions will be used.
        stats: list of stats to use; if `None`, defaults to ["mean",
        "quantile"].
        stats_args: optional arguments to pass to the stats functions.

    Returns:
        xr.Dataset containing statistics for concentrations, fluxes, and country
        totals.

    """
    obs_and_errs = inv_out.get_obs_and_errors()
    conc_outs = make_concentration_outputs(inv_out, stats=stats, stats_args=stats_args)
    flux_outs = make_flux_outputs(inv_out, stats=stats, stats_args=stats_args)
    country_outs = make_country_outputs(
        inv_out,
        country_file=country_file,
        country_regions=country_regions,
        stats=stats,
        stats_args=stats_args
    )

    model_data = inv_out.get_model_data(var_names=["hx", "hbc", "min_error"]).rename(
        {"hx": "Hx", "hbc": "Hbc", "min_error": "min_model_error"}
    )

    result = xr.merge(
        [obs_and_errs, conc_outs, flux_outs, country_outs, model_data, inv_out.get_flat_basis()]
    )

    for dv in result.data_vars:
        for k, v in result[dv].attrs.items():
            result[dv].attrs[k] = v.replace("_", " ")

    result.attrs["description"] = "RHIME inversion outputs."

    return result
