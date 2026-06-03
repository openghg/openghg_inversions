from pathlib import Path
import getpass
import json
import math
import re
from collections.abc import Hashable
import warnings
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import pandas as pd
import xarray as xr

from openghg.util import timestamp_now  # pyright: ignore[reportPrivateImportUsage]
from openghg_inversions import convert
from openghg_inversions.array_ops import align_sparse_lat_lon, sparse_xr_dot
from openghg_inversions.config.version import code_version
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import (
    make_concentration_outputs,
    make_flux_outputs,
    make_country_outputs,
    make_multisector_flux_trace_outputs,
    observation_and_error_outputs,
)
from openghg_inversions.postprocessing.stats import calculate_stats, stats_functions


# path to `paris_formatting` submodule
paris_formatting_path = Path(__file__).parent

ParisTemplateVersion = Literal["legacy", "latest"]


class ParisTemplateFiles(NamedTuple):
    """CDL templates used for one PARIS output schema version."""

    concentration: Path
    concentration_version: str
    flux: Path
    flux_version: str


PARIS_LATEST_COUNTRIES = (
    "AUT",
    "BEL",
    "CHE",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "FIN",
    "FRA",
    "GBR",
    "HRV",
    "HUN",
    "IRL",
    "ITA",
    "LUX",
    "NLD",
    "NOR",
    "POL",
    "PRT",
    "SVK",
    "SVN",
    "SWE",
)


DEFAULT_PARIS_TEMPLATE_VERSION: ParisTemplateVersion = "legacy"
PARIS_TEMPLATE_FILES: dict[ParisTemplateVersion, ParisTemplateFiles] = {
    "legacy": ParisTemplateFiles(
        concentration=paris_formatting_path / "PARIS_Lagrangian_inversion_concentration_EUROPE_v03.cdl",
        concentration_version="v03",
        flux=paris_formatting_path / "PARIS_Lagrangian_inversion_flux_EUROPE.cdl",
        flux_version="legacy",
    ),
    "latest": ParisTemplateFiles(
        concentration=paris_formatting_path / "PARIS_Lagrangian_inversion_concentration_EUROPE_v04.cdl",
        concentration_version="v04",
        flux=paris_formatting_path / "PARIS_Lagrangian_inversion_flux_EUROPE_v03.cdl",
        flux_version="v03",
    ),
}


def paris_template_files(template_version: ParisTemplateVersion) -> ParisTemplateFiles:
    """Return CDL template paths for a PARIS output schema version."""
    try:
        return PARIS_TEMPLATE_FILES[template_version]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported PARIS template version {template_version!r}; "
            f"expected one of {sorted(PARIS_TEMPLATE_FILES)!r}."
        ) from exc


var_pat = re.compile(r"\s*[a-z]+ ([a-zA-Z_]+)\(.*\)")
attr_pat = re.compile(r"\s+([a-zA-Z_]+):([a-zA-Z_]+)\s*=\s*([^;]+)")


def _require_paris_metadata(inv_out: InversionOutput, *, allow_multisector: bool = False) -> tuple[str, str]:
    """Return metadata required by current PARIS products."""
    if inv_out.is_multisector and not allow_multisector:
        raise ValueError("PARIS postprocessing supports only single-sector RHIME outputs.")

    species = inv_out.species
    domain = inv_out.domain
    if species is None or domain is None:
        raise ValueError(
            "PARIS postprocessing requires InversionOutput metadata fields 'species' and 'domain'."
        )
    return species, domain


def get_data_var_attrs(template_file: str | Path, species: str | None = None) -> dict[str, dict[str, Any]]:
    """Extract data variable attributes from template file."""
    attr_dict: dict[str, Any] = {}

    with open(template_file) as f:
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


def _site_info_for_site(site: str) -> dict[str, Any]:
    """Return site metadata from ``openghg_defs`` when available."""
    try:
        from openghg_defs import site_info_file
    except ImportError:
        return {}

    try:
        site_info = json.loads(Path(site_info_file).read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    provider_info = site_info.get(site.upper(), {})
    if not isinstance(provider_info, dict) or not provider_info:
        return {}

    first_provider = next(iter(provider_info.values()))
    return first_provider if isinstance(first_provider, dict) else {}


def _as_float_or_nan(value: object) -> float:
    """Best-effort conversion of scalar metadata values to float."""
    if value is None:
        return np.nan
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+", value)
        if match is None:
            return np.nan
        value = match.group(0)
    else:
        value = str(value)
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return np.nan
    return converted if math.isfinite(converted) else np.nan


def _measurement_values(inv_out: InversionOutput, names: tuple[str, ...], size: int) -> np.ndarray | None:
    """Return a 1-D measurement-aligned variable from inversion inputs."""
    for name in names:
        if name not in inv_out.inv_inputs:
            continue
        data = inv_out.inv_inputs[name]
        if "nmeasure" not in data.dims or data.sizes["nmeasure"] != size:
            continue
        return np.asarray(data.values)
    return None


def _numeric_measurement_values(
    inv_out: InversionOutput,
    names: tuple[str, ...],
    size: int,
) -> np.ndarray | None:
    """Return measurement-aligned numeric metadata values from inversion inputs."""
    values = _measurement_values(inv_out, names, size)
    if values is None:
        return None
    return np.asarray([_as_float_or_nan(value) for value in values], dtype="float32")


def _datetime_values_to_epoch_days(values: object) -> np.ndarray:
    """Convert datetime-like values to days since the Unix epoch."""
    array = np.asarray(values)
    flat = pd.to_datetime(array.reshape(-1))
    converted = (flat - pd.Timestamp("1970-01-01")) / pd.Timedelta("1d")
    return np.asarray(converted, dtype="float64").reshape(array.shape)


def _time_period_to_timedelta(period: str | None) -> pd.Timedelta:
    """Parse a period string, falling back to zero for unsupported calendar periods."""
    if period is None:
        return cast(pd.Timedelta, pd.Timedelta(0))
    try:
        delta = pd.to_timedelta(period)
    except ValueError:
        return cast(pd.Timedelta, pd.Timedelta(0))
    return delta if isinstance(delta, pd.Timedelta) else cast(pd.Timedelta, pd.Timedelta(0))


def _add_observation_time_bounds(ds: xr.Dataset, obs_avg_period: str | None) -> xr.Dataset:
    """Assign midpoint observation times and start/end bounds on latest PARIS concentration output."""
    period = _time_period_to_timedelta(obs_avg_period)
    start_times = pd.to_datetime(ds["time"].values)
    end_times = start_times + period
    midpoint_times = start_times + period / 2
    time_bnds = np.stack([start_times, end_times], axis=1)
    return ds.assign_coords(time=("index", midpoint_times)).assign(time_bnds=(("index", "nbnds"), time_bnds))


def _convert_time_and_bounds_to_epoch_days(ds: xr.Dataset) -> xr.Dataset:
    """Convert ``time`` and optional ``time_bnds`` to days since 1970-01-01."""
    result = ds.assign_coords(time=("index", _datetime_values_to_epoch_days(ds["time"].values)))
    if "time_bnds" in result:
        result["time_bnds"] = (
            result["time_bnds"].dims,
            _datetime_values_to_epoch_days(result.time_bnds.values),
        )
    return result


def _platform_metadata(inv_out: InversionOutput, site_values: np.ndarray, size: int) -> xr.Dataset:
    """Build platform identifiers and sample location fields for latest concentration output."""
    raw_height_values = _measurement_values(inv_out, ("inlet_height", "inlet"), size)
    intake_height = _numeric_measurement_values(inv_out, ("inlet_height", "inlet"), size)

    platform_labels: list[str] = []
    platform_index: list[int] = []
    label_to_index: dict[str, int] = {}

    for i, site in enumerate(site_values):
        site_label = str(site)
        height_label = None
        if raw_height_values is not None:
            raw_height = raw_height_values[i]
            if raw_height is not None and str(raw_height).lower() != "nan":
                height_label = str(raw_height)
        label = f"{site_label}-{height_label}" if height_label else site_label
        if label not in label_to_index:
            label_to_index[label] = len(platform_labels)
            platform_labels.append(label)
        platform_index.append(label_to_index[label])

    site_metadata = [_site_info_for_site(str(site)) for site in site_values]
    longitude = _numeric_measurement_values(inv_out, ("longitude", "lon"), size)
    latitude = _numeric_measurement_values(inv_out, ("latitude", "lat"), size)
    altitude = _numeric_measurement_values(inv_out, ("altitude", "height_station_masl"), size)

    if longitude is None:
        longitude = np.asarray(
            [_as_float_or_nan(metadata.get("longitude")) for metadata in site_metadata],
            dtype="float64",
        )
    if latitude is None:
        latitude = np.asarray(
            [_as_float_or_nan(metadata.get("latitude")) for metadata in site_metadata],
            dtype="float64",
        )
    if altitude is None:
        altitude = np.asarray(
            [_as_float_or_nan(metadata.get("height_station_masl")) for metadata in site_metadata],
            dtype="float32",
        )
    if intake_height is None:
        intake_height = np.full(size, np.nan, dtype="float32")

    altitude_model = _numeric_measurement_values(inv_out, ("altitude_model",), size)
    if altitude_model is None:
        altitude_model = altitude.astype("float32", copy=True)

    intake_height_model = _numeric_measurement_values(inv_out, ("intake_height_model", "fp_height"), size)
    if intake_height_model is None:
        intake_height_model = intake_height.astype("float32", copy=True)

    return xr.Dataset(
        {
            "longitude": ("index", longitude),
            "latitude": ("index", latitude),
            "altitude": ("index", altitude),
            "intake_height": ("index", intake_height),
            "altitude_model": ("index", altitude_model),
            "intake_height_model": ("index", intake_height_model),
            "number_of_identifier": ("index", np.asarray(platform_index, dtype="int16")),
            "assimilation_flag": ("index", np.ones(size, dtype="int16")),
        },
        coords={"platform": ("platform", np.asarray(platform_labels, dtype=object))},
    )


def make_global_attrs(
    output_type: Literal["flux", "conc"],
    author: str | None = None,
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
        institution="ACRG, University of Bristol, UK",
        author=author or getpass.getuser(),
        inversion_system="RHIME",
        inversion_system_version=code_version(),
        apriori_description=apriori_description,
        transport_model="NAME",
        transport_model_version="NAME III (version 8.0)",
        met_model="UKV",
        domain=domain,
        species=species,
        project="Process Attribution of Regional emISsions (PARIS)",
        references="Ganesan, et.al., 2014, doi: 10.5194/acp-14-3855-2014",
        acknowledgements="Please acknowledge ACRG, University of Bristol, in any publication that uses this data.",
    )
    default_history = f"RHIME results processed at: {timestamp_now()}"
    global_attrs["history"] = history or default_history

    if comment is not None:
        global_attrs["comment"] = comment

    global_attrs["conventions"] = "CF-1.8"
    global_attrs["license"] = "CC-BY-4.0"

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


def _cast_float_data_vars_to_float32(ds: xr.Dataset) -> xr.Dataset:
    """Cast floating PARIS data variables to float32 at the product boundary."""
    updates: dict[Hashable, xr.DataArray] = {}
    for name in ds.data_vars:
        if np.issubdtype(ds[name].dtype, np.floating):
            updates[name] = ds[name].astype("float32")

    return ds.assign(updates) if updates else ds


def convert_time_to_unix_epoch(x: xr.Dataset, units: str = "1s") -> xr.Dataset:
    """Convert `time` coordinate of xarray Dataset or DataArray to number of "units" since 1 Jan 1970 (the "UNIX epoch")."""
    time_converted = (pd.DatetimeIndex(x.time) - pd.Timestamp("1970-01-01")) / pd.Timedelta(units)

    return x.assign_coords(time=time_converted)


def shift_measurement_time_to_midpoint(ds: xr.Dataset, period: str = "4h") -> xr.Dataset:
    """Adjust `time` coordinate of concentrations to represent half averaging "period"."""
    time_shifted = pd.to_datetime(ds["time"].astype("datetime64[ns]").values) + pd.to_timedelta(period) / 2
    ds = ds.assign_coords(time=time_shifted)
    return ds


def paris_concentration_outputs(
    inv_out: InversionOutput,
    report_mode: bool = False,
    obs_avg_period: str = "4h",
    template_version: ParisTemplateVersion = DEFAULT_PARIS_TEMPLATE_VERSION,
) -> xr.Dataset:
    """Create PARIS concentration outputs.

    TODO: add offset
    """
    if template_version == "latest":
        return paris_concentration_outputs_latest(
            inv_out,
            report_mode=report_mode,
            obs_avg_period=obs_avg_period,
        )

    stats = ["kde_mode", "quantiles"] if report_mode else ["mean", "quantiles"]

    stats_args = {"quantiles__quantiles": [0.159, 0.841]}

    obs_and_errs_raw = observation_and_error_outputs(inv_out).unstack("nmeasure")
    existing_vars = set(obs_and_errs_raw.data_vars)
    rename_map = {
        "y_obs": "Yobs",
        "y_obs_prior_factor": "Yobs_prior_factor",
        "y_obs_prior_upper_level_factor": "Yobs_prior_upper_level_factor",
        "y_obs_repeatability": "uYobs_repeatability",
        "y_obs_variability": "uYobs_variability",
        "model_error": "uYmod",
        "total_error": "uYtotal",
    }
    filtered_rename_map = {k: v for k, v in rename_map.items() if k in existing_vars}
    obs_and_errs = obs_and_errs_raw.rename(filtered_rename_map).drop_vars("y_obs_error")

    conc_outputs = make_concentration_outputs(inv_out, stats, stats_args, combine_bc_and_offset=True).unstack(
        "nmeasure"
    )

    # rename to match PARIS concentrations template
    def renamer(name: str) -> str:
        when = "apost" if "posterior" in name else "apriori"

        if "bc" in name:
            suffix = "BC"
        elif "offset" in name:
            suffix = "_bias"
        else:
            suffix = ""

        prefix = "qY" if "quantile" in name else "Y"
        return prefix + when + suffix

    rename_dict = {"quantile": "percentile"}
    for dv in conc_outputs.data_vars:
        rename_dict[str(dv)] = renamer(str(dv))

    conc_outputs = conc_outputs.rename(rename_dict)

    # We produce these, but they aren't in the template
    if "qYapostBC" in conc_outputs.data_vars:
        conc_outputs = conc_outputs.drop_vars(["qYapostBC", "qYaprioriBC"])

    if "qYapost_bias" in conc_outputs.data_vars:
        conc_outputs = conc_outputs.drop_vars(["qYapost_bias", "qYapriori_bias"])

    template_files = paris_template_files(template_version)
    conc_attrs = get_data_var_attrs(template_files.concentration)

    units = float(obs_and_errs_raw["y_obs"].attrs["units"].split(" ")[0])

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

    if "Yobs_prior_factor" in result.data_vars and "Yobs_prior_upper_level_factor" in result.data_vars:
        with xr.set_options(keep_attrs="default"):
            factor = result["Yobs_prior_factor"] + result["Yobs_prior_upper_level_factor"]
        result["Yobs"] += factor
        result["Yapost"] += factor
        result["Yapriori"] += factor
        result["qYapost"] += factor
        result["qYapriori"] += factor
        if "YapostBC" in result.data_vars:
            result["YapostBC"] += factor
            result["YaprioriBC"] += factor

    result.sitenames.attrs["long_name"] = "identifier of site"

    result.attrs = make_global_attrs("conc")
    result.attrs["paris_concentration_template_version"] = template_files.concentration_version

    return _cast_float_data_vars_to_float32(result)


def paris_concentration_outputs_latest(
    inv_out: InversionOutput,
    report_mode: bool = False,
    obs_avg_period: str = "4h",
) -> xr.Dataset:
    """Create single-sector PARIS concentration outputs for the latest CDL template."""
    species, domain = _require_paris_metadata(inv_out)
    stats = ["kde_mode", "stdev", "quantiles"] if report_mode else ["mean", "stdev", "quantiles"]
    stats_args = {"quantiles__quantiles": [0.159, 0.841]}

    obs_and_errs_raw = observation_and_error_outputs(inv_out)
    existing_vars = set(obs_and_errs_raw.data_vars)
    rename_map = {
        "y_obs": "mf_observed",
        "y_obs_repeatability": "stdev_mf_observed_repeatability",
        "y_obs_variability": "stdev_mf_observed_variability",
        "model_error": "stdev_mf_model",
        "total_error": "stdev_mf_total",
    }
    obs_and_errs = obs_and_errs_raw.rename(
        {name: output_name for name, output_name in rename_map.items() if name in existing_vars}
    ).drop_vars(
        [
            name
            for name in (
                "y_obs_error",
                "y_obs_prior_factor",
                "y_obs_prior_upper_level_factor",
            )
            if name in obs_and_errs_raw
        ]
    )

    conc_outputs = make_concentration_outputs(
        inv_out,
        stats=stats,
        stats_args=stats_args,
        combine_bc_and_offset=True,
    )

    def renamer(name: str) -> str | None:
        when = "posterior" if "posterior" in name else "prior"
        if "quantile" in name:
            prefix = "percentile_"
        elif "stdev" in name:
            prefix = "stdev_"
        else:
            prefix = ""

        if name.startswith("y_"):
            return f"{prefix}mf_{when}"
        if name.startswith("mu_bc_") and not prefix:
            return f"mf_bc_{when}"
        if name.startswith("offset_") and not prefix:
            return f"mf_bias_{when}"
        return None

    rename_dict = {"quantile": "percentile"}
    keep_vars = []
    for data_var in conc_outputs.data_vars:
        output_name = renamer(str(data_var))
        if output_name is not None:
            rename_dict[str(data_var)] = output_name
            keep_vars.append(data_var)

    conc_outputs = conc_outputs[keep_vars].rename(rename_dict)
    result = xr.merge([obs_and_errs, conc_outputs])

    if "y_obs_prior_factor" in obs_and_errs_raw and "y_obs_prior_upper_level_factor" in obs_and_errs_raw:
        with xr.set_options(keep_attrs="default"):
            factor = (
                obs_and_errs_raw["y_obs_prior_factor"] + obs_and_errs_raw["y_obs_prior_upper_level_factor"]
            )
        for name in (
            "mf_observed",
            "mf_prior",
            "mf_posterior",
            "percentile_mf_prior",
            "percentile_mf_posterior",
            "mf_bc_prior",
            "mf_bc_posterior",
        ):
            if name in result:
                result[name] = result[name] + factor

    if "nmeasure" in result.dims:
        result = result.reset_index("nmeasure").rename({"nmeasure": "index"})
    else:
        result = result.rename({"time": "index"})

    size = result.sizes["index"]
    site_values = np.asarray(result["site"].values) if "site" in result else np.asarray(["unknown"] * size)
    platform_metadata = _platform_metadata(inv_out, site_values, size)

    template_files = paris_template_files("latest")
    conc_attrs = get_data_var_attrs(template_files.concentration, species)
    units = float(obs_and_errs_raw["y_obs"].attrs["units"].split(" ")[0])

    result = (
        xr.merge([result, platform_metadata])
        .pipe(_add_observation_time_bounds, obs_avg_period)
        .pipe(_convert_time_and_bounds_to_epoch_days)
        .pipe(add_variable_attrs, conc_attrs, units)
        .transpose("index", "percentile", "platform", "nbnds", missing_dims="ignore")
    )

    result.attrs = make_global_attrs("conc", species=species, domain=domain)
    result.attrs["paris_concentration_template_version"] = template_files.concentration_version

    return result.as_numpy()


def _flux_frequency_to_offset(flux_frequency: str) -> pd.DateOffset | pd.Timedelta:
    """Convert a flux frequency string to a calendar-aware pandas offset."""
    if flux_frequency == "monthly":
        return pd.DateOffset(months=1)
    elif flux_frequency == "yearly":
        return pd.DateOffset(years=1)
    else:
        return pd.to_timedelta(flux_frequency)


def _flux_interval_midpoints(
    flux_times: list[pd.Timestamp],
    flux_period: pd.DateOffset | pd.Timedelta,
    inv_start: pd.Timestamp,
    inv_end: pd.Timestamp,
) -> tuple[list[pd.Timestamp], list[int]]:
    """Compute output timestamps as midpoints of each flux interval clipped to the inversion period.

    For each flux interval [ft, ft + flux_period], the output timestamp is the
    midpoint of the overlap with [inv_start, inv_end]. This correctly handles
    cases where the flux period and inversion period differ in length, e.g.:

    - 3-monthly inversion on yearly fluxes: the yearly flux interval is clipped
      to the inversion period, so the midpoint is within the inversion period.
    - 2-yearly inversion on yearly fluxes: each yearly flux interval lies fully
      within the inversion period, so the midpoint is mid-year as expected.

    Args:
        flux_times: Start timestamps of each flux interval.
        flux_period: Duration of a single flux interval.
        inv_start: Start of the inversion period.
        inv_end: End of the inversion period.

    Returns:
        Tuple of (midpoint_timestamps, valid_time_indices). Both lists contain
        one entry per flux interval that overlaps the inversion period.
    """
    midpoints = []
    valid_indices = []
    for i, ft in enumerate(flux_times):
        overlap_start = max(ft, inv_start)
        overlap_end = min(ft + flux_period, inv_end)
        # Only include if there is valid overlap
        if overlap_end > overlap_start:
            midpoint = overlap_start + (overlap_end - overlap_start) / 2
            midpoints.append(midpoint)
            valid_indices.append(i)
    return midpoints, valid_indices


def _flux_interval_midpoints_and_bounds(
    flux_times: list[pd.Timestamp],
    flux_period: pd.DateOffset | pd.Timedelta,
    inv_start: pd.Timestamp,
    inv_end: pd.Timestamp,
) -> tuple[list[pd.Timestamp], list[tuple[pd.Timestamp, pd.Timestamp]], list[int]]:
    """Compute clipped flux interval midpoints, bounds, and retained indices."""
    midpoints = []
    bounds = []
    valid_indices = []
    for i, ft in enumerate(flux_times):
        overlap_start = max(ft, inv_start)
        overlap_end = min(ft + flux_period, inv_end)
        if overlap_end > overlap_start:
            midpoints.append(overlap_start + (overlap_end - overlap_start) / 2)
            bounds.append((overlap_start, overlap_end))
            valid_indices.append(i)
    return midpoints, bounds, valid_indices


def _assign_flux_time_bounds(
    ds: xr.Dataset,
    flux_frequency: Literal["monthly", "yearly"] | str,
    inv_start: pd.Timestamp,
    inv_end: pd.Timestamp,
) -> xr.Dataset:
    """Assign midpoint flux times and clipped interval bounds for latest PARIS flux output."""
    flux_period = _flux_frequency_to_offset(flux_frequency)
    flux_times = list(pd.to_datetime(ds.time.values))
    midpoints, bounds, valid_indices = _flux_interval_midpoints_and_bounds(
        flux_times,
        flux_period,
        inv_start,
        inv_end,
    )
    time_bnds = np.asarray(bounds, dtype="datetime64[ns]")
    return (
        ds.isel(time=valid_indices)
        .assign_coords(time=midpoints)
        .assign(time_bnds=(("time", "nbnds"), time_bnds))
    )


def _convert_flux_time_and_bounds_to_epoch_days(ds: xr.Dataset) -> xr.Dataset:
    """Convert flux ``time`` and ``time_bnds`` to days since 1970-01-01."""
    result = ds.assign_coords(time=("time", _datetime_values_to_epoch_days(ds["time"].values)))
    if "time_bnds" in result:
        result["time_bnds"] = (
            result["time_bnds"].dims,
            _datetime_values_to_epoch_days(result.time_bnds.values),
        )
    return result


def _latest_paris_countries(country_file: str | Path | None, domain: str) -> Countries:
    """Return country metadata for the latest PARIS CDL country list."""
    countries = Countries.from_file(
        country_file=country_file,
        country_code="alpha3",
        country_selections=list(PARIS_LATEST_COUNTRIES),
        domain=domain,
    )
    countries.matrix = countries.matrix.reindex(country=list(PARIS_LATEST_COUNTRIES))
    return countries


def _multisector_country_trace_kg(inv_out: InversionOutput, countries: Countries, species: str) -> xr.Dataset:
    """Return multisector total country flux traces in kg/yr from reconstructed total flux."""
    total_flux_trace = make_multisector_flux_trace_outputs(
        inv_out,
        report_flux_on_inversion_grid=False,
    )[["flux_total_prior", "flux_total_posterior"]].rename(
        {
            "flux_total_prior": "country_prior",
            "flux_total_posterior": "country_posterior",
        }
    )
    country_weights = countries.matrix.as_numpy() * countries.area_grid.as_numpy()
    country_weights = align_sparse_lat_lon(country_weights, total_flux_trace["country_posterior"])
    country_trace = sparse_xr_dot(country_weights, total_flux_trace, dim=["lat", "lon"])
    country_trace = country_trace * 365 * 24 * 3600 * convert.molar_mass(species) * 1e-3
    return country_trace.reindex(country=list(PARIS_LATEST_COUNTRIES))


def _latest_country_outputs(
    inv_out: InversionOutput,
    countries: Countries,
    species: str,
    stats: list[str],
    stats_args: dict[str, Any],
    country_file: str | Path | None,
) -> xr.Dataset:
    """Return latest PARIS country statistics for single- or multisector outputs."""
    if inv_out.is_multisector:
        country_trace = _multisector_country_trace_kg(inv_out, countries, species)
        country_stats_args = dict(stats_args)
        country_stats_args["stats"] = stats
        return calculate_stats(country_trace, **country_stats_args)

    country_outs = make_country_outputs(
        inv_out,
        country_file=country_file,
        country_selections=list(PARIS_LATEST_COUNTRIES),
        stats=stats,
        stats_args=stats_args,
        country_code="alpha3",
    )
    return (country_outs * 1e-3).reindex(country=list(PARIS_LATEST_COUNTRIES))


def _country_posterior_covariance_kg(
    inv_out: InversionOutput,
    countries: Countries,
    species: str,
    flux_frequency: Literal["monthly", "yearly"] | str,
) -> np.ndarray:
    """Return posterior country-total covariance in kg2 yr-2."""
    if inv_out.is_multisector:
        country_trace = _multisector_country_trace_kg(inv_out, countries, species)
        posterior = country_trace["country_posterior"]
    else:
        country_trace = countries.get_country_trace(inv_out=inv_out)
        posterior = country_trace["country_posterior"] * 1e-3
    flux_period = _flux_frequency_to_offset(flux_frequency)
    flux_times = list(pd.to_datetime(posterior.flux_time.values))
    _, _, valid_indices = _flux_interval_midpoints_and_bounds(
        flux_times,
        flux_period,
        inv_out.start_time,
        inv_out.end_time,
    )

    values = posterior.isel(flux_time=valid_indices).transpose("flux_time", "country", "draw").values
    if values.shape[2] < 2:
        return np.full((values.shape[0], values.shape[1], values.shape[1]), np.nan, dtype="float32")

    centered = values - values.mean(axis=2, keepdims=True)
    return np.einsum("tcd,ted->tce", centered, centered) / (values.shape[2] - 1)


def _add_country_covariance(result: xr.Dataset, covariance: np.ndarray, attrs: dict[str, Any]) -> xr.Dataset:
    """Add latest PARIS country covariance with the duplicate country dimension required by the template."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result["covariance_flux_total_posterior_country"] = (
            ("time", "country", "country"),
            covariance,
        )
    result["covariance_flux_total_posterior_country"].attrs = attrs
    return result


def paris_flux_output(
    inv_out: InversionOutput,
    country_file: str | Path | None = None,
    time_point: Literal["start", "midpoint"] = "midpoint",
    report_mode: bool = False,
    inversion_grid: bool = True,
    flux_frequency: Literal["monthly", "yearly"] | str = "yearly",
    template_version: ParisTemplateVersion = DEFAULT_PARIS_TEMPLATE_VERSION,
) -> xr.Dataset:
    if template_version == "latest":
        return paris_flux_output_latest(
            inv_out,
            country_file=country_file,
            time_point=time_point,
            report_mode=report_mode,
            inversion_grid=inversion_grid,
            flux_frequency=flux_frequency,
        )

    species, domain = _require_paris_metadata(inv_out)
    stats = ["kde_mode", "quantiles"] if report_mode else ["mean", "quantiles"]

    stats_args = {"quantiles__quantiles": [0.159, 0.841]}

    flux_outs = make_flux_outputs(
        inv_out,
        stats=stats,
        stats_args=stats_args,
        report_flux_on_inversion_grid=False,
        include_scale_factors=False,
    )

    template_files = paris_template_files(template_version)
    emissions_attrs = get_data_var_attrs(template_files.flux, species)
    country_outs = make_country_outputs(
        inv_out,
        country_file=country_file,
        country_regions="paris",
        stats=stats,
        stats_args=stats_args,
        country_code="alpha3",
    )
    country_outs = country_outs * 1e-3  # convert g/yr to kg/yr

    # add country mask
    countries = Countries.from_file(country_file=country_file, country_code="alpha3", domain=domain)

    country_fraction = countries.matrix.as_numpy().rename("country_fraction")

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
        flux_period = _flux_frequency_to_offset(flux_frequency)
        inv_start = inv_out.start_time
        inv_end = inv_out.end_time

        def time_func(ds):
            flux_times = pd.to_datetime(ds.time.values)
            midpoints, valid_indices = _flux_interval_midpoints(flux_times, flux_period, inv_start, inv_end)
            return ds.isel(time=valid_indices).assign_coords(time=midpoints)
    else:

        def time_func(ds):
            return ds

    result = (
        xr.merge([flux_outs, country_outs, country_fraction.reindex_like(flux_outs)], join="outer")
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

    result = result.transpose("time", "percentile", "country", "latitude", "longitude")

    result.attrs = make_global_attrs("flux")
    result.attrs["paris_flux_template_version"] = template_files.flux_version

    return _cast_float_data_vars_to_float32(result).as_numpy()


def paris_flux_output_latest(
    inv_out: InversionOutput,
    country_file: str | Path | None = None,
    time_point: Literal["start", "midpoint"] = "midpoint",
    report_mode: bool = False,
    inversion_grid: bool = True,
    flux_frequency: Literal["monthly", "yearly"] | str = "yearly",
) -> xr.Dataset:
    """Create single-sector PARIS flux outputs for the latest CDL template."""
    if time_point != "midpoint":
        raise ValueError("Latest PARIS flux output requires midpoint time coordinates and time bounds.")

    species, domain = _require_paris_metadata(inv_out, allow_multisector=True)
    stats = ["kde_mode", "stdev", "quantiles"] if report_mode else ["mean", "stdev", "quantiles"]
    stats_args = {"quantiles__quantiles": [0.159, 0.841]}

    flux_outs = make_flux_outputs(
        inv_out,
        stats=stats,
        stats_args=stats_args,
        report_flux_on_inversion_grid=False,
        include_scale_factors=False,
    )
    countries = _latest_paris_countries(country_file=country_file, domain=domain)
    if inv_out.is_multisector:
        flux_outs = flux_outs[
            [data_var for data_var in flux_outs.data_vars if str(data_var).startswith("flux_total_")]
        ]
    country_outs = _latest_country_outputs(
        inv_out,
        countries=countries,
        species=species,
        stats=stats,
        stats_args=stats_args,
        country_file=country_file,
    )
    country_fraction = countries.matrix.as_numpy().rename("country_fraction")
    cell_area = countries.area_grid.as_numpy().rename("cell_area")
    country_covariance = _country_posterior_covariance_kg(
        inv_out,
        countries=countries,
        species=species,
        flux_frequency=flux_frequency,
    )

    def flux_renamer(name: str) -> str:
        if not name.startswith("flux_total_"):
            name = name.replace("flux_", "flux_total_", 1)
        if "quantile" in name:
            name = "percentile_" + name.replace("_quantile", "")
        elif name.endswith("_stdev"):
            name = "stdev_" + name.removesuffix("_stdev")
        for stats_func_name in stats_functions:
            if name.endswith(f"_{stats_func_name}"):
                name = name.removesuffix(f"_{stats_func_name}")
        return name

    def country_renamer(name: str) -> str:
        suffix = name.removeprefix("country_")
        if "quantile" in suffix:
            suffix = suffix.replace("_quantile", "")
            return f"percentile_flux_total_{suffix}_country"
        if suffix.endswith("_stdev"):
            suffix = suffix.removesuffix("_stdev")
            return f"stdev_flux_total_{suffix}_country"
        for stats_func_name in stats_functions:
            if suffix.endswith(f"_{stats_func_name}"):
                suffix = suffix.removesuffix(f"_{stats_func_name}")
        return f"flux_total_{suffix}_country"

    flux_rename_dict = {str(dv): flux_renamer(str(dv)) for dv in flux_outs.data_vars}
    country_rename_dict = {str(dv): country_renamer(str(dv)) for dv in country_outs.data_vars}

    dim_rename_dict = {"quantile": "percentile", "flux_time": "time"}
    if "lat" in flux_outs.dims:
        dim_rename_dict["lat"] = "latitude"
    if "lon" in flux_outs.dims:
        dim_rename_dict["lon"] = "longitude"

    template_files = paris_template_files("latest")
    emissions_attrs = get_data_var_attrs(template_files.flux, species)

    result = (
        xr.merge(
            [
                flux_outs,
                country_outs,
                country_fraction.reindex_like(flux_outs),
                cell_area.reindex_like(flux_outs),
            ],
            join="outer",
        )
        .rename(dim_rename_dict)
        .pipe(_assign_flux_time_bounds, flux_frequency, inv_out.start_time, inv_out.end_time)
        .pipe(_convert_flux_time_and_bounds_to_epoch_days)
        .rename({**flux_rename_dict, **country_rename_dict})
        .pipe(add_variable_attrs, emissions_attrs)
    )

    if inversion_grid:
        inversion_grid_flux_rename_dict = {v: f"{v}_inversion_grid" for v in flux_rename_dict.values()}
        inversion_grid_flux_outs_raw = make_flux_outputs(
            inv_out,
            stats=stats,
            stats_args=stats_args,
            report_flux_on_inversion_grid=True,
            include_scale_factors=False,
        )
        if inv_out.is_multisector:
            inversion_grid_flux_outs_raw = inversion_grid_flux_outs_raw[
                [
                    data_var
                    for data_var in inversion_grid_flux_outs_raw.data_vars
                    if str(data_var).startswith("flux_total_")
                ]
            ]
        inversion_grid_flux_outs = (
            inversion_grid_flux_outs_raw.rename(dim_rename_dict)
            .pipe(_assign_flux_time_bounds, flux_frequency, inv_out.start_time, inv_out.end_time)
            .pipe(_convert_flux_time_and_bounds_to_epoch_days)
            .rename(flux_rename_dict)
            .pipe(add_variable_attrs, emissions_attrs)
            .rename(inversion_grid_flux_rename_dict)
        )
        result = result.merge(inversion_grid_flux_outs)

    result = result.transpose("time", "percentile", "country", "latitude", "longitude", "nbnds").as_numpy()
    result = _add_country_covariance(
        result,
        country_covariance,
        emissions_attrs.get("covariance_flux_total_posterior_country", {}),
    )
    result.attrs = make_global_attrs("flux", species=species, domain=domain)
    result.attrs["paris_flux_template_version"] = template_files.flux_version

    return result


def infer_flux_frequency(flux: xr.DataArray) -> str:
    """Attempt to infer flux frequency.

    This does not work in all cases. If the flux has a "time_period" attribute,
    then that will be used. Otherwise, we try to infer the period by looking at
    the differences between timestamps. If only one timestamp is found, then a
    default value of "yearly" is returned.

    Args:
        flux: flux DataArray

    Returns:
        frequency string that can be parsed by pd.to_timedelta, or is "yearly" or "monthly"

    Raises:
        ValueError: if inferred frequency is not "yearly" or "monthly", and cannot be parsed by pd.to_timedelta

    """
    if "time_period" in flux.attrs:
        time_period = flux.attrs["time_period"]
        if "year" in time_period:
            return "yearly"
        if "month" in time_period:
            return "monthly"

        # check if the result can be parsed by pd.to_timedelta
        try:
            pd.to_timedelta(time_period)
        except ValueError as e:
            raise ValueError(
                f"Flux frequency {time_period} from flux.attrs['time_period'] cannot be parsed by pd.to_timedelta."
            ) from e
        else:
            return time_period

    else:
        # take most frequent gap between times
        try:
            flux_frequency_delta = pd.Series(flux.flux_time.values).diff().mode()[0]
        except KeyError:
            # only one time value
            return "yearly"
        else:
            flux_frequency = pd.tseries.frequencies.to_offset(flux_frequency_delta).freqstr  # type: ignore

            # "1 days" will be converted to "D" by the previous two lines, so we need to add a "1" in front
            if not flux_frequency[0].isdigit():
                flux_frequency = "1" + flux_frequency

            # check if the result can be parsed
            try:
                pd.to_timedelta(flux_frequency)
            except ValueError as e:
                raise ValueError(
                    f"Flux frequency {flux_frequency} inferred from gaps in flux.time cannot be parsed by pd.to_timedelta"
                    "(and flux.attrs['time_period'] is not set)."
                ) from e
            else:
                return flux_frequency


def make_paris_outputs(
    inv_out: InversionOutput,
    country_file: str | Path | None = None,
    time_point: Literal["start", "midpoint"] = "midpoint",
    report_mode: bool = False,
    inversion_grid: bool = True,
    obs_avg_period: str = "4h",
    domain: str | None = None,
    template_version: ParisTemplateVersion = DEFAULT_PARIS_TEMPLATE_VERSION,
) -> tuple[xr.Dataset, xr.Dataset]:
    # infer flux frequency
    flux_frequency = infer_flux_frequency(inv_out.flux)
    conc_outs = paris_concentration_outputs(
        inv_out,
        report_mode=report_mode,
        obs_avg_period=obs_avg_period,
        template_version=template_version,
    )
    flux_outs = paris_flux_output(
        inv_out,
        report_mode=report_mode,
        country_file=country_file,
        inversion_grid=inversion_grid,
        time_point=time_point,
        flux_frequency=flux_frequency,
        template_version=template_version,
    )

    return flux_outs, conc_outs
