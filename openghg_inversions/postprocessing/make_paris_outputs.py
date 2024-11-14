from pathlib import Path
import re
from typing import Any, Literal

import pandas as pd
import xarray as xr

from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import get_obs_and_errors, make_concentration_outputs


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
    if units == "1s":
        time_converted = (pd.DatetimeIndex(x.time) - pd.Timestamp("1970-01-01")) // pd.Timedelta(units) / (24 * 3600)
    else:
        time_converted = (pd.DatetimeIndex(x.time) - pd.Timestamp("1970-01-01")) // pd.Timedelta(units)

    return x.assign_coords(time=time_converted)


def shift_measurement_time_to_midpoint(ds: xr.Dataset, period: str = "4h") -> xr.Dataset:
    """Adjust `time` coordinate of concentrations to represent half averaging "period"."""
    time_shifted = pd.to_datetime(ds["time"].astype("datetime64[ns]").values) + pd.to_timedelta(period) / 2
    ds = ds.assign_coords(time=time_shifted)
    return ds


def paris_concentration_outputs(
    inv_out: InversionOutput, report_mode: bool = False, obs_avg_period: str = "4h"
) -> xr.Dataset:
    if report_mode:
        stats = ["kde_mode", "quantiles"]
    else:
        stats = ["mean", "quantiles"]

    stats_args = {"quantiles__quantiles": [0.159, 0.841]}

    obs_and_errs = get_obs_and_errors(inv_out).rename({
        "y_obs": "Yobs",
        "y_obs_repeatability": "uYobs_repeatability",
        "y_obs_variability": "uYobs_variability",
        "model_error": "uYmod",
        "total_error": "uYtotal",
    }).drop_vars("y_obs_error")

    conc_outputs = make_concentration_outputs(inv_out, stats, stats_args)

    def renamer(name: str) -> str:
        when = "apost" if "posterior" in name else "apriori"
        suffix = "BC" if "bc" in name else ""
        prefix = "qY" if "quantile" in name else "Y"
        return prefix + when + suffix

    rename_dict = {"quantile": "percentile"}
    for dv in conc_outputs.data_vars:
        rename_dict[str(dv)] = renamer(str(dv))

    conc_outputs = conc_outputs.rename(rename_dict)

    if "qYapostBC" in conc_outputs.data_vars:
        conc_outputs = conc_outputs.drop_vars(["qYapostBC", "qYaprioriBC"])

    conc_attrs = get_data_var_attrs(conc_template_path)

    units = float(inv_out.obs.attrs["units"].split(" ")[0])  # e.g. get 1e-12 from "1e-12 mol/mol"

    common_rename_dict = {"site": "nsite"}

    result = (xr.merge([obs_and_errs, conc_outputs])
              .pipe(shift_measurement_time_to_midpoint, obs_avg_period)
              .pipe(convert_time_to_unix_epoch, "1s")
              .rename(common_rename_dict)
              .pipe(add_variable_attrs, conc_attrs, units)
              .transpose("time", "percentile", "nsite")
              .rename_vars(nsite="sitenames")
              )

    result.sitenames.attrs["long_name"] = "identifier of site"

    result.attrs = make_global_attrs("conc")

    return result
