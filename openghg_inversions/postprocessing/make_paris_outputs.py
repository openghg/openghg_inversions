from pathlib import Path
import re
from typing import Any, Literal

import pandas as pd
import xarray as xr

from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import (
    get_obs_and_errors,
    make_concentration_outputs,
    make_flux_outputs,
    make_country_outputs,
)
from openghg_inversions.postprocessing.stats import stats_functions
from openghg_inversions.utils import get_country_file_path


# path to `paris_formatting` submodule
paris_formatting_path = Path(__file__).parent

# paths to template files
conc_template_path = paris_formatting_path / "PARIS_Lagrangian_inversion_concentration_EUROPE_v03.cdl"
flux_template_path = paris_formatting_path / "PARIS_Lagrangian_inversion_flux_EUROPE.cdl"


var_pat = re.compile(r"\s*[a-z]+ ([a-zA-Z_]+)\(.*\)")
attr_pat = re.compile(r"\s+([a-zA-Z_]+):([a-zA-Z_]+)\s*=\s*([^;]+)")


def get_data_var_attrs(template_file: str | Path, species: str | None = None) -> dict[str, dict[str, Any]]:
    """Extract data variable attributes from template file."""
    attr_dict: dict[str, Any] = {}

    with open(template_file, "r") as f:
        in_vars = False
        for line in f.readlines():
            if line.startswith("variables"):
                in_vars = True
            if in_vars:
                if m := var_pat.match(line):
                    attr_dict[m.group(1)] = {}
                if (m := attr_pat.match(line)) is not None and "FillValue" not in m.group(2):
                    val = m.group(3).strip().strip('"')

                    if species is not None:
                        val = val.replace("<species>", species)

                    attr_dict[m.group(1)][m.group(2)] = val

    return attr_dict


def make_global_attrs(
    output_type: Literal["flux", "conc"],
    author: str = "OpenGHG",
    species: str = "inert",
    domain: str = "EUROPE",
    apriori_description: str = "EDGAR 8.0",
    history: str | None = None,
    comment: str | None = None,
) -> dict[str, str]:
    global_attrs = {}
    global_attrs["title"] = (
        "Observed and simulated atmospheric concentrations"
        if output_type == "conc"
        else "Flux estimates: spatially-resolved and by country"
    )
    global_attrs.update(
        author=author,
        source="processed NAME(8.0) model output",
        transport_model="NAME",
        transport_model_version="NAME III (version 8.0)",
        met_model="UKV",
        species=species,
        domain=domain,
        inversion_method="RHIME",
        apriori_description=apriori_description,
        publication_acknowledgements="Please acknowledge ACRG, University of Bristol, in any publication that uses this data.",
    )
    global_attrs["history"] = history if history is not None else ""
    global_attrs["comment"] = comment if comment is not None else ""

    return global_attrs


def add_variable_attrs(
    ds: xr.Dataset, attrs: dict[str, dict[str, Any]], units: float | None = None
) -> xr.Dataset:
    """Update data variables and coordinates of Dataset based on attributes dictionary.

    If `units` provided, data variables with "units" attribute will be rescaled by `units`. This is to convert e.g.
    from 1e-9 mol/mol to mol/mol.
    """
    for k, v in attrs.items():
        if k in ds.data_vars:
            if units is not None and "units" in v and v["units"].count("mol") == 2:
                ds[k] = units * ds[k]
            ds[k].attrs = v
        elif k in ds.coords:
            ds.coords[k].attrs = v

    return ds


def convert_time_to_unix_epoch(x: xr.Dataset, units: str = "1s") -> xr.Dataset:
    """Convert `time` coordinate of xarray Dataset or DataArray to number of "units" since
    1 Jan 1970 (the "UNIX epoch").
    """
    time_converted = (pd.DatetimeIndex(x.time) - pd.Timestamp("1970-01-01")) / pd.Timedelta(units)

    return x.assign_coords(time=time_converted)


def shift_measurement_time_to_midpoint(ds: xr.Dataset, period: str = "4h") -> xr.Dataset:
    """Adjust `time` coordinate of concentrations to represent half averaging "period"."""
    time_shifted = pd.to_datetime(ds["time"].astype("datetime64[ns]").values) + pd.to_timedelta(period) / 2
    ds = ds.assign_coords(time=time_shifted)
    return ds


def paris_concentration_outputs(
    inv_out: InversionOutput, report_mode: bool = False, obs_avg_period: str = "4h"
) -> xr.Dataset:
    """Create PARIS concentration outputs.

    TODO: add offset
    """
    if report_mode:
        stats = ["kde_mode", "quantiles"]
    else:
        stats = ["mean", "quantiles"]

    stats_args = {"quantiles__quantiles": [0.159, 0.841]}

    obs_and_errs = (
        get_obs_and_errors(inv_out, unstack_nmeasure=True)
        .rename(
            {
                "y_obs": "Yobs",
                "y_obs_repeatability": "uYobs_repeatability",
                "y_obs_variability": "uYobs_variability",
                "model_error": "uYmod",
                "total_error": "uYtotal",
            }
        )
        .drop_vars("y_obs_error")
    )

    conc_outputs = make_concentration_outputs(inv_out, stats, stats_args, unstack_nmeasure=True)

    # rename to match PARIS concentrations template
    def renamer(name: str) -> str:
        when = "apost" if "posterior" in name else "apriori"
        suffix = "BC" if "bc" in name else ""
        prefix = "qY" if "quantile" in name else "Y"
        return prefix + when + suffix

    rename_dict = {"quantile": "percentile"}
    for dv in conc_outputs.data_vars:
        rename_dict[str(dv)] = renamer(str(dv))

    conc_outputs = conc_outputs.rename(rename_dict)

    # We produce these, but they aren't in the template
    if "qYapostBC" in conc_outputs.data_vars:
        conc_outputs = conc_outputs.drop_vars(["qYapostBC", "qYaprioriBC"])

    conc_attrs = get_data_var_attrs(conc_template_path)

    units = float(inv_out.obs.attrs["units"].split(" ")[0])  # e.g. get 1e-12 from "1e-12 mol/mol"

    common_rename_dict = {"site": "nsite"}

    result = (
        xr.merge([obs_and_errs, conc_outputs])
        .pipe(shift_measurement_time_to_midpoint, obs_avg_period)
        .pipe(convert_time_to_unix_epoch, "1d")
        .rename(common_rename_dict)
        .pipe(add_variable_attrs, conc_attrs, units)
        .transpose("time", "percentile", "nsite")
        .rename_vars(nsite="sitenames")
    )

    result.sitenames.attrs["long_name"] = "identifier of site"

    result.attrs = make_global_attrs("conc")

    return result


def paris_flux_output(
    inv_out: InversionOutput,
    country_file: str | Path | None = None,
    time_point: Literal["start", "midpoint"] = "midpoint",
    report_mode: bool = False,
    inversion_grid: bool = False,
    flux_frequency: Literal["monthly", "yearly"] | str = "yearly",
) -> xr.Dataset:
    if report_mode:
        stats = ["kde_mode", "quantiles"]
    else:
        stats = ["mean", "quantiles"]

    stats_args = {"quantiles__quantiles": [0.159, 0.841]}

    flux_outs = make_flux_outputs(
        inv_out,
        stats=stats,
        stats_args=stats_args,
        report_flux_on_inversion_grid=False,
        include_scale_factors=False,
    )

    emissions_attrs = get_data_var_attrs(flux_template_path, inv_out.species)

    country_outs = make_country_outputs(
        inv_out,
        country_file=country_file,
        country_regions="paris",
        stats=stats,
        stats_args=stats_args,
    )
    country_outs = country_outs * 1e-3  # convert g/yr to kg/yr

    # add country mask
    country_path = get_country_file_path(country_file)
    countries = Countries(xr.open_dataset(country_path))

    country_fraction = countries.matrix

    # rename to match PARIS flux template
    def renamer(name: str) -> str:
        """Rename variables to match PARIS flux template.

        NOTE: this won't work correctly if HDI is used instead of quantiles.
        """
        if "country" in name:
            name = name.replace("country", "country_flux_total")
        elif "flux" in name:
            name = name.replace("flux", "flux_total")

        if "quantile" in name:
            name = "percentile_" + name.replace("_quantile", "")

        for stats_func_name in stats_functions:
            if name.endswith(f"_{stats_func_name}"):
                name = name.removesuffix(f"_{stats_func_name}")

        return name

    flux_rename_dict = {str(dv): renamer(str(dv)) for dv in flux_outs.data_vars}
    country_rename_dict = {str(dv): renamer(str(dv)) for dv in country_outs.data_vars}
    rename_dict = {**flux_rename_dict, **country_rename_dict}

    dim_rename_dict = {"quantile": "percentile", "flux_time": "time"}

    if "lat" in flux_outs.dims:
        dim_rename_dict["lat"] = "latitude"
    if "lon" in flux_outs.dims:
        dim_rename_dict["lon"] = "longitude"


    if time_point == "midpoint":
        if flux_frequency == "monthly":
            offset = pd.DateOffset(weeks=2)
        elif flux_frequency == "yearly":
            offset = pd.DateOffset(months=6)
        else:
            offset = pd.to_timedelta(flux_frequency) / 2

        time_func = lambda ds: ds.assign_coords(time=(pd.to_datetime(ds.time.values) + offset))
    else:
        time_func = lambda ds: ds

    result = (
        xr.merge([flux_outs, country_outs, country_fraction])
        .rename(dim_rename_dict)
        .pipe(time_func)
        .pipe(convert_time_to_unix_epoch, "1d")
        .rename(rename_dict)
        .pipe(add_variable_attrs, emissions_attrs)
    )

    if inversion_grid:
        inversion_grid_flux_rename_dict = {v: f"{v}_inversion_grid" for v in flux_rename_dict.values()}
        inversion_grid_flux_outs = (
            make_flux_outputs(
                inv_out,
                stats=stats,
                stats_args=stats_args,
                report_flux_on_inversion_grid=True,
                include_scale_factors=False,
            )
            .rename(dim_rename_dict)
            .pipe(time_func)
            .pipe(convert_time_to_unix_epoch, "1d")
            .rename(flux_rename_dict)
            .pipe(add_variable_attrs, emissions_attrs)
            .rename(inversion_grid_flux_rename_dict)
        )
        result = result.merge(inversion_grid_flux_outs)


    result = result.transpose(
        "time", "percentile", "country", "latitude", "longitude"
    )

    # use "countrynumber" instead of "country"
    result = result.rename_dims(country="countrynumber")

    result.attrs = make_global_attrs("flux")

    return result


def make_paris_outputs(
    inv_out: InversionOutput,
    country_file: str | Path | None = None,
    time_point: Literal["start", "midpoint"] = "midpoint",
    report_mode: bool = False,
    inversion_grid: bool = False,
    obs_avg_period: str = "4h",
) -> tuple[xr.Dataset, xr.Dataset]:
    # infer flux frequency
    # NOTE: this only works in certain cases
    if "time_period" in inv_out.flux.attrs:
        time_period = inv_out.flux.attrs["time_period"]
        if "year" in time_period:
            flux_frequency = "yearly"
        elif "month" in time_period:
            flux_frequency = "monthly"
        else:
            # hopefully this can be parsed by pd.to_timedelta
            flux_frequency = time_period
    else:
        # take most frequent gap between times
        try:
            flux_frequency_delta = pd.Series(inv_out.flux.flux_time.values).diff().mode()[0]
        except KeyError:
            # only one time value
            flux_frequency = "yearly"
        else:
            flux_frequency = pd.tseries.frequencies.to_offset(flux_frequency_delta).freqstr  # type: ignore

            # "1 days" will be converted to "D" by the previous two lines, so we need to add a "1" in front
            if not flux_frequency[0].isdigit():
                flux_frequency = "1" + flux_frequency

    conc_outs = paris_concentration_outputs(inv_out, report_mode=report_mode, obs_avg_period=obs_avg_period)
    flux_outs = paris_flux_output(
        inv_out,
        report_mode=report_mode,
        country_file=country_file,
        inversion_grid=inversion_grid,
        time_point=time_point,
        flux_frequency=flux_frequency,
    )

    return flux_outs, conc_outs
