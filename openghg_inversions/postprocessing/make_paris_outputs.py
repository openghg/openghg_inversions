import getpass
import json
import math
import re
from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import pandas as pd
import xarray as xr
from openghg.util import timestamp_now  # pyright: ignore[reportPrivateImportUsage]

from openghg_inversions import utils
from openghg_inversions.array_ops import align_sparse_lat_lon
from openghg_inversions.config.version import code_version
from openghg_inversions.flux_sanitization import copy_flux_nonfinite_attrs
from openghg_inversions.inversion_data._units import mole_fraction_unit_scale
from openghg_inversions.postprocessing._basis_products import add_basis_reconstruction_metadata
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.postprocessing.make_outputs import (
    OutputSector,
    make_concentration_outputs,
    make_country_outputs,
    make_flux_outputs,
    make_multisector_country_trace_outputs,
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


var_pat = re.compile(r"\s*[a-z]+ ([a-zA-Z_][a-zA-Z0-9_]*)\(.*\)")
var_type_pat = re.compile(r"\s*([a-z]+) ([a-zA-Z_][a-zA-Z0-9_]*)\(.*\)")
attr_pat = re.compile(r"\s+([a-zA-Z_][a-zA-Z0-9_]*):([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^;]+)")
sector_name_pat = re.compile(r"[^a-z0-9]+")
NETCDF_TO_NUMPY_DTYPE = {
    "byte": "int8",
    "short": "int16",
    "int": "int32",
    "float": "float32",
    "double": "float64",
}


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


def get_data_var_dtypes(template_file: str | Path) -> dict[str, str]:
    """Extract numeric variable dtypes from a CDL template file."""
    dtype_dict: dict[str, str] = {}

    with open(template_file) as f:
        in_vars = False
        for line in f.readlines():
            if line.startswith("variables"):
                in_vars = True
            if not in_vars:
                continue
            if (m := var_type_pat.match(line)) is None:
                continue
            netcdf_type, var_name = m.groups()
            if netcdf_type in NETCDF_TO_NUMPY_DTYPE:
                dtype_dict[var_name] = NETCDF_TO_NUMPY_DTYPE[netcdf_type]

    return dtype_dict


def _replace_sector_placeholder(value: Any, sector_name: str) -> Any:
    """Replace PARIS sector placeholders in string template values."""
    if not isinstance(value, str):
        return value
    return value.replace("<sector_name>", sector_name).replace("sector_name", sector_name)


def _expand_sector_template_mapping(
    mapping: Mapping[str, Any],
    sector_names: Iterable[str],
) -> dict[str, Any]:
    """Expand CDL ``sector_name`` placeholders for concrete sector variable names."""
    expanded = dict(mapping)
    for template_name, template_value in mapping.items():
        if "sector_name" not in template_name:
            continue
        for sector_name in sector_names:
            output_name = template_name.replace("sector_name", sector_name)
            if isinstance(template_value, Mapping):
                expanded[output_name] = {
                    key: _replace_sector_placeholder(value, sector_name)
                    for key, value in template_value.items()
                }
            else:
                expanded[output_name] = _replace_sector_placeholder(template_value, sector_name)
    return expanded


def _paris_output_sectors(inv_out: InversionOutput) -> list[OutputSector]:
    """Return multisector metadata in the local output-sector shape."""
    raw_sectors = inv_out.model_metadata.get("sectors")
    if not raw_sectors:
        raise ValueError("Multisector PARIS output requires model_metadata['sectors'].")

    sectors: list[OutputSector] = []
    for raw_sector in raw_sectors:
        if isinstance(raw_sector, Mapping):
            name = raw_sector.get("name")
            flux_source = raw_sector.get("flux_source")
            variable_suffix = raw_sector.get("variable_suffix")
        else:
            name = getattr(raw_sector, "name", None)
            flux_source = getattr(raw_sector, "flux_source", None)
            variable_suffix = getattr(raw_sector, "variable_suffix", None)
        if name is None or flux_source is None or variable_suffix is None:
            raise ValueError("Sector metadata must include 'name', 'flux_source', and 'variable_suffix'.")
        sectors.append(OutputSector(str(name), str(flux_source), str(variable_suffix)))

    return sectors


def _paris_sector_name_by_suffix(inv_out: InversionOutput) -> dict[str, str]:
    """Return PARIS sector variable names keyed by RHIME variable suffix."""
    if not inv_out.is_multisector:
        return {}

    sector_name_by_suffix = {}
    used_names = set()
    for sector in _paris_output_sectors(inv_out):
        sector_name = sector_name_pat.sub("", sector.variable_suffix.lower())
        if not sector_name:
            raise ValueError(f"Could not derive a PARIS sector name from {sector.variable_suffix!r}.")
        if sector_name == "total":
            raise ValueError("PARIS sector name 'total' is reserved for summed flux variables.")
        if sector_name in used_names:
            raise ValueError(
                "PARIS sector names must be unique after removing separator characters; "
                f"duplicate sector name {sector_name!r}."
            )
        used_names.add(sector_name)
        sector_name_by_suffix[sector.variable_suffix] = sector_name

    return sector_name_by_suffix


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
    """Build global attributes shared by PARIS output products.

    Args:
        output_type: PARIS product type, either flux or concentration.
        author: Output author. Defaults to the current user.
        species: Species represented by the output.
        domain: Spatial inversion domain.
        apriori_description: Description of the prior emissions product.
        history: Optional processing history. Defaults to a timestamped RHIME
            processing entry.
        comment: Optional dataset comment. A descriptive nonempty default is
            used when this is omitted or empty.

    Returns:
        CF-oriented global attributes for a PARIS dataset.
    """
    global_attrs = {}
    global_attrs["title"] = (
        "Observed and simulated atmospheric concentrations"
        if output_type == "conc"
        else "Flux estimates: spatially-resolved and by country"
    )

    global_attrs.update(
        institution="ACRG, University of Bristol, UK",
        author=author or getpass.getuser(),
        source=(
            "Trace gas concentrations from observations and transport simulations / inverse estimation."
            if output_type == "conc"
            else "Estimated flux from trace gas observations, transport simulations, and inversion code."
        ),
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
    global_attrs["comment"] = comment or "Inverse modelling output generated by RHIME for the PARIS project."

    global_attrs["Conventions"] = "CF-1.8"
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


def _astype_data_array(da: xr.DataArray, dtype: str) -> xr.DataArray:
    """Cast a DataArray, including template-required duplicate-dimension variables."""
    if len(set(da.dims)) == len(da.dims):
        return da.astype(dtype)
    return da.copy(data=da.data.astype(dtype))


def _cast_data_vars_to_template_dtypes(
    ds: xr.Dataset,
    template_file: str | Path,
    *,
    sector_names: Iterable[str] = (),
) -> xr.Dataset:
    """Cast PARIS variables and coordinates to numeric dtypes declared by a CDL template."""
    data_var_updates: dict[Hashable, xr.DataArray] = {}
    coordinate_updates: dict[Hashable, tuple[tuple[Hashable, ...], Any, dict[Hashable, Any]]] = {}
    dtype_mapping = cast(
        dict[str, str],
        _expand_sector_template_mapping(get_data_var_dtypes(template_file), sector_names),
    )
    for name, dtype in dtype_mapping.items():
        if name not in ds.variables or ds[name].dtype == np.dtype(dtype):
            continue
        if name in ds.coords:
            coordinate = ds[name]
            coordinate_updates[name] = (
                coordinate.dims,
                coordinate.data.astype(dtype),
                dict(coordinate.attrs),
            )
        else:
            data_var_updates[name] = _astype_data_array(ds[name], dtype)

    result = ds.assign(data_var_updates) if data_var_updates else ds
    return result.assign_coords(coordinate_updates) if coordinate_updates else result


def _prepare_latest_paris_netcdf_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Prevent xarray from adding attributes excluded by the latest PARIS templates."""
    result = ds.copy(deep=False)
    for coordinate in result.coords.values():
        coordinate.encoding = {**coordinate.encoding, "_FillValue": None}
    result["time_bnds"].encoding = {
        **result["time_bnds"].encoding,
        "_FillValue": None,
        "coordinates": None,
    }
    return result


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

    units = mole_fraction_unit_scale(
        obs_and_errs_raw["y_obs"].attrs["units"],
        context="PARIS observation units",
    )

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
        # Column prior factors are part of the total column mole fraction, not the BC-only term.
        result["Yobs"] += factor
        result["Yapost"] += factor
        result["Yapriori"] += factor
        result["qYapost"] += factor
        result["qYapriori"] += factor

    result.sitenames.attrs["long_name"] = "identifier of site"

    result.attrs = make_global_attrs("conc")
    result.attrs["paris_concentration_template_version"] = template_files.concentration_version

    return _cast_float_data_vars_to_float32(result)


def paris_concentration_outputs_latest(
    inv_out: InversionOutput,
    report_mode: bool = False,
    obs_avg_period: str = "4h",
) -> xr.Dataset:
    """Create total PARIS concentration outputs for the latest CDL template."""
    species, domain = _require_paris_metadata(inv_out, allow_multisector=True)
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
        # Column prior factors are part of the total column mole fraction, not the BC-only term.
        for name in (
            "mf_observed",
            "mf_prior",
            "mf_posterior",
            "percentile_mf_prior",
            "percentile_mf_posterior",
        ):
            if name in result:
                result[name] = result[name] + factor

    if "nmeasure" in result.dims:
        result = result.reset_index("nmeasure").rename({"nmeasure": "index"})
    else:
        result = result.rename({"time": "index"})

    # The latest template requires these fields even when no baseline was solved.
    # No-BC runs currently have no baseline time series to report, so record it as missing.
    reference_mf = next(
        result[name] for name in ("mf_prior", "mf_posterior", "mf_observed") if name in result
    )
    for name in ("mf_bc_prior", "mf_bc_posterior"):
        if name not in result:
            result[name] = xr.full_like(reference_mf, np.nan, dtype="float32")

    size = result.sizes["index"]
    site_values = np.asarray(result["site"].values) if "site" in result else np.asarray(["unknown"] * size)
    platform_metadata = _platform_metadata(inv_out, site_values, size)
    result = result.drop_vars("site", errors="ignore")

    template_files = paris_template_files("latest")
    conc_attrs = get_data_var_attrs(template_files.concentration, species)
    units = mole_fraction_unit_scale(
        obs_and_errs_raw["y_obs"].attrs["units"],
        context="PARIS observation units",
    )

    result = (
        xr.merge([result, platform_metadata])
        .pipe(_add_observation_time_bounds, obs_avg_period)
        .pipe(_convert_time_and_bounds_to_epoch_days)
        .pipe(add_variable_attrs, conc_attrs, units)
        .transpose("index", "percentile", "platform", "nbnds", missing_dims="ignore")
    )

    result.attrs = make_global_attrs("conc", species=species, domain=domain)
    result.attrs["paris_concentration_template_version"] = template_files.concentration_version

    result = _cast_data_vars_to_template_dtypes(result, template_files.concentration).as_numpy()
    return _prepare_latest_paris_netcdf_encoding(result)


def _flux_frequency_to_offset(flux_frequency: str) -> pd.DateOffset | pd.Timedelta:
    """Convert a flux frequency string to a calendar-aware pandas offset."""
    normalized_period = utils._normalize_flux_period(flux_frequency)
    if normalized_period == "monthly":
        return pd.DateOffset(months=1)
    if normalized_period == "yearly":
        return pd.DateOffset(years=1)
    if normalized_period is None:
        raise ValueError(
            f"Flux period {flux_frequency!r} is not a recognized calendar period "
            "or a positive fixed duration."
        )
    return pd.to_timedelta(normalized_period)


def _flux_interval_midpoints(
    flux_times: list[pd.Timestamp],
    flux_period: pd.DateOffset | pd.Timedelta,
    inv_start: pd.Timestamp,
    inv_end: pd.Timestamp,
    flux_frequency: str | None = None,
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
        flux_frequency: User-facing frequency name used in overlap errors.

    Returns:
        Tuple of (midpoint_timestamps, valid_time_indices). Both lists contain
        one entry per flux interval that overlaps the inversion period.

    Raises:
        ValueError: If no flux interval overlaps the inversion period.
    """
    midpoints, _, valid_indices = _flux_interval_midpoints_and_bounds(
        flux_times,
        flux_period,
        inv_start,
        inv_end,
        flux_frequency,
    )
    return midpoints, valid_indices


def _flux_interval_midpoints_and_bounds(
    flux_times: list[pd.Timestamp],
    flux_period: pd.DateOffset | pd.Timedelta,
    inv_start: pd.Timestamp,
    inv_end: pd.Timestamp,
    flux_frequency: str | None = None,
) -> tuple[list[pd.Timestamp], list[tuple[pd.Timestamp, pd.Timestamp]], list[int]]:
    """Compute clipped flux interval midpoints, bounds, and retained indices.

    Args:
        flux_times: Start timestamps for available flux periods.
        flux_period: Duration of each flux period.
        inv_start: Start of the inversion period.
        inv_end: End of the inversion period.
        flux_frequency: User-facing frequency name used in overlap errors.

    Returns:
        Midpoints, clipped bounds, and source indices for overlapping periods.

    Raises:
        ValueError: If no flux interval overlaps the inversion period.
    """
    frequency = flux_frequency or str(flux_period)
    if len(flux_times) == 0:
        raise ValueError(
            f"No flux interval overlaps the inversion period for frequency {frequency!r}: "
            f"no flux timestamps are available; inversion start={inv_start}, end={inv_end}."
        )

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

    if not valid_indices:
        flux_start = min(flux_times)
        flux_end = max(flux_time + flux_period for flux_time in flux_times)
        raise ValueError(
            f"No flux interval overlaps the inversion period for frequency {frequency!r}: "
            f"flux interval start={flux_start}, end={flux_end}; "
            f"inversion start={inv_start}, end={inv_end}."
        )

    return midpoints, bounds, valid_indices


def _assign_flux_time_bounds(
    ds: xr.Dataset,
    flux_frequency: Literal["monthly", "yearly"] | str,
    inv_start: pd.Timestamp,
    inv_end: pd.Timestamp,
) -> xr.Dataset:
    """Assign midpoint times and clipped bounds to overlapping flux periods.

    Args:
        ds: Flux output with period starts on its ``time`` coordinate.
        flux_frequency: Calendar alias or positive fixed duration for each
            source period.
        inv_start: Start of the inversion period.
        inv_end: End of the inversion period.

    Returns:
        The subset of flux periods overlapping the inversion, with midpoint
        ``time`` values and a ``time_bnds`` variable.

    Raises:
        ValueError: If the flux coordinate is empty or no period overlaps the
            inversion.
    """
    flux_period = _flux_frequency_to_offset(flux_frequency)
    flux_times = list(pd.to_datetime(ds.time.values))
    midpoints, bounds, valid_indices = _flux_interval_midpoints_and_bounds(
        flux_times,
        flux_period,
        inv_start,
        inv_end,
        flux_frequency,
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


def _latest_paris_countries(
    country_file: str | Path | None,
    domain: str,
    country_selections: Iterable[str] | None,
) -> Countries:
    """Return country metadata for a latest-template PARIS output.

    Args:
        country_file: Optional country-definition file for the inversion domain.
        domain: Inversion domain used to resolve a default country file.
        country_selections: Optional country names or codes to include. ``None``
            preserves all countries and their order from the selected country
            file, allowing the PARIS format to be used outside Europe.

    Returns:
        Country masks and area grid using alpha-3 labels. Explicit selections
        are ordered as requested.
    """
    countries = Countries.from_file(
        country_file=country_file,
        country_code="alpha3",
        domain=domain,
    )
    if country_selections is not None:
        selections = countries.country_labels.select_by_country_info(country_selections)
        countries.matrix = countries.matrix.sel(country=list(selections))
        countries.country_selections = selections
    return countries


def _multisector_country_trace_kg(
    inv_out: InversionOutput,
    countries: Countries,
    sector_name_by_suffix: Mapping[str, str],
) -> xr.Dataset:
    """Return PARIS-labelled multisector country traces in kilograms per year.

    Args:
        inv_out: Multisector inversion output containing retained sector basis
            functions and scaling traces.
        countries: Country masks and cell areas used to project basis regions
            directly to country totals.
        sector_name_by_suffix: Mapping from each RHIME trace-variable suffix to
            its normalized PARIS sector name. For example, ``{"total_ff":
            "totalff"}`` maps ``country_total_ff_posterior`` to
            ``country_totalff_posterior`` without confusing it with the summed
            total variable.

    Returns:
        Lazy or sparse-compatible total and per-sector country traces with
        dimensions ``(country, flux_time, draw)`` and units of kg yr-1. Summed
        variables use ``country_prior`` and ``country_posterior``; sector
        variables use ``country_<sector>_<when>``.

    Raises:
        ValueError: If required multisector metadata is missing or retained
            basis, flux, and country grids cannot be aligned.
    """
    projected_trace = make_multisector_country_trace_outputs(inv_out, countries)
    rename = {}
    for variable_suffix, sector_name in sector_name_by_suffix.items():
        for when in ("prior", "posterior"):
            source_name = f"country_{variable_suffix}_{when}"
            if source_name in projected_trace:
                rename[source_name] = f"country_{sector_name}_{when}"

    # Float32 aggregation can incur significant round-off error in high-magnitude
    # country totals and uncertainty statistics, so promote the projected country
    # samples before calculating them. Keeping the promotion at country level lets
    # these calculations share one aligned float64 sector intermediate while the
    # much larger spatial draw arrays retain their original dtype.
    sector_trace = projected_trace[list(rename)].rename(rename).astype(np.float64) * 1e-3
    result = sector_trace.copy(deep=False)
    for when in ("prior", "posterior"):
        sector_variables = [
            f"country_{sector_name}_{when}"
            for sector_name in sector_name_by_suffix.values()
            if f"country_{sector_name}_{when}" in sector_trace
        ]
        result[f"country_{when}"] = xr.concat(
            [sector_trace[name] for name in sector_variables],
            dim="sector",
        ).sum("sector", min_count=len(sector_name_by_suffix))

    result = result[["country_prior", "country_posterior", *sector_trace.data_vars]]
    result.attrs = dict(projected_trace.attrs)
    for name in result.data_vars:
        result[name].attrs["units"] = "kg/yr"
    return result


def _single_sector_country_trace_kg(
    inv_out: InversionOutput,
    countries: Countries,
) -> xr.Dataset:
    """Return a float64 single-sector country trace in kilograms per year.

    Args:
        inv_out: Single-sector inversion output.
        countries: Country masks and cell areas used for country projection.

    Returns:
        Projected prior and posterior country traces promoted to float64 before
        conversion from grams to kilograms per year. Dimensions, coordinates,
        and dataset attributes are preserved, and every output variable has
        ``units="kg/yr"``; variables typically have dimensions
        ``(flux_time, country, draw)``.
    """
    projected_trace = countries.get_country_trace(inv_out=inv_out)
    result = projected_trace.astype(np.float64) * 1e-3
    result.attrs = dict(projected_trace.attrs)
    for name in result.data_vars:
        result[name].attrs["units"] = "kg/yr"
    return result


def _latest_country_outputs(
    inv_out: InversionOutput,
    countries: Countries,
    stats: list[str],
    stats_args: dict[str, Any],
    sector_name_by_suffix: Mapping[str, str] | None = None,
    multisector_country_trace: xr.Dataset | None = None,
) -> xr.Dataset:
    """Return latest PARIS country statistics for single- or multisector outputs.

    Args:
        inv_out: Single- or multisector inversion output.
        countries: Country masks and cell areas used for country projection.
        stats: Statistics to calculate from the country traces.
        stats_args: Additional arguments passed to ``calculate_stats``.
        sector_name_by_suffix: Optional mapping from RHIME sector suffixes to
            normalized PARIS sector names.
        multisector_country_trace: Optional precomputed single- or multisector
            country trace in kilograms per year. The historical parameter name
            is retained for compatibility.

    Returns:
        Country statistics in kilograms per year, with ``country`` and
        ``flux_time`` dimensions and total plus per-sector variables where
        applicable.
    """
    country_trace = multisector_country_trace
    if country_trace is None:
        country_trace = (
            _multisector_country_trace_kg(
                inv_out,
                countries,
                sector_name_by_suffix or _paris_sector_name_by_suffix(inv_out),
            )
            if inv_out.is_multisector
            else _single_sector_country_trace_kg(inv_out, countries)
        )
    country_stats_args = dict(stats_args)
    country_stats_args["stats"] = stats
    return calculate_stats(country_trace, **country_stats_args)


def _country_posterior_covariance_kg(
    inv_out: InversionOutput,
    countries: Countries,
    flux_frequency: Literal["monthly", "yearly"] | str,
    multisector_country_trace: xr.Dataset | None = None,
) -> np.ndarray:
    """Calculate total posterior covariance between countries.

    This calculation applies to both single-sector and multisector outputs. In
    the single-sector case, country totals are mapped directly from basis-region
    scaling traces by ``Countries.get_country_trace``. Multisector callers can
    supply the already projected and summed country trace to avoid repeating
    that work.

    Args:
        inv_out: Single- or multisector inversion output.
        countries: Country masks and cell areas used for the country projection.
        flux_frequency: Frequency used to select output flux intervals.
        multisector_country_trace: Optional precomputed single- or multisector
            country trace in kilograms per year. The historical parameter name
            is retained for compatibility.

    Returns:
        Population covariance for each flux interval with dimensions ordered as
        time, first country, and second country, in kg2 yr-2.
    """
    country_trace = multisector_country_trace
    if country_trace is None:
        country_trace = (
            _multisector_country_trace_kg(
                inv_out,
                countries,
                _paris_sector_name_by_suffix(inv_out),
            )
            if inv_out.is_multisector
            else _single_sector_country_trace_kg(inv_out, countries)
        )
    posterior = country_trace["country_posterior"]
    flux_period = _flux_frequency_to_offset(flux_frequency)
    flux_times = list(pd.to_datetime(posterior.flux_time.values))
    _, _, valid_indices = _flux_interval_midpoints_and_bounds(
        flux_times,
        flux_period,
        inv_out.start_time,
        inv_out.end_time,
        flux_frequency,
    )

    posterior = posterior.isel(flux_time=valid_indices).dropna("draw", how="all")
    values = np.asarray(
        posterior.transpose("flux_time", "country", "draw").values,
        dtype=np.float64,
    )
    if values.shape[2] == 0:
        return np.full((values.shape[0], values.shape[1], values.shape[1]), np.nan, dtype=np.float64)

    centered = values - values.mean(axis=2, keepdims=True)
    return np.einsum("tcd,ted->tce", centered, centered) / values.shape[2]


def _sector_country_posterior_covariances_kg(
    inv_out: InversionOutput,
    countries: Countries,
    flux_frequency: Literal["monthly", "yearly"] | str,
    sector_name_by_suffix: Mapping[str, str],
    multisector_country_trace: xr.Dataset | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Calculate within-sector and between-sector country covariances.

    Args:
        inv_out: Multisector inversion output.
        countries: Country masks and cell areas used for country projection.
        flux_frequency: Frequency used to select output flux intervals.
        sector_name_by_suffix: Mapping from RHIME trace suffixes to normalized
            PARIS sector names.
        multisector_country_trace: Optional precomputed total and sector country
            traces in kilograms per year.

    Returns:
        A mapping of sector name to population covariance arrays with shape
        ``(flux_time, country, country)`` and an array of covariance between
        sectors within each country with shape
        ``(flux_time, country, sector, sector)``. Both are in kg2 yr-2. The
        second result is ``None`` when no sectors are supplied.
    """
    if not sector_name_by_suffix:
        return {}, None

    sector_trace = multisector_country_trace
    if sector_trace is None:
        sector_trace = _multisector_country_trace_kg(
            inv_out,
            countries,
            sector_name_by_suffix,
        )
    sector_names = list(sector_name_by_suffix.values())
    first_posterior = sector_trace[f"country_{sector_names[0]}_posterior"]
    flux_period = _flux_frequency_to_offset(flux_frequency)
    flux_times = list(pd.to_datetime(first_posterior.flux_time.values))
    _, _, valid_indices = _flux_interval_midpoints_and_bounds(
        flux_times,
        flux_period,
        inv_out.start_time,
        inv_out.end_time,
        flux_frequency,
    )

    sector_posteriors = [
        sector_trace[f"country_{sector_name}_posterior"]
        .isel(flux_time=valid_indices)
        .dropna("draw", how="all")
        .transpose("flux_time", "country", "draw")
        for sector_name in sector_names
    ]
    sector_covariances = {}
    for sector_name, posterior in zip(sector_names, sector_posteriors, strict=True):
        values = np.asarray(posterior.values, dtype=np.float64)
        if values.shape[2] == 0:
            covariance = np.full(
                (values.shape[0], values.shape[1], values.shape[1]),
                np.nan,
                dtype=np.float64,
            )
        else:
            centered = values - values.mean(axis=2, keepdims=True)
            covariance = np.einsum("tcd,ted->tce", centered, centered) / values.shape[2]
        sector_covariances[sector_name] = covariance

    posterior_by_sector = xr.concat(
        [
            posterior.expand_dims(sector=[sector_name])
            for sector_name, posterior in zip(sector_names, sector_posteriors, strict=True)
        ],
        dim="sector",
    )
    values = np.asarray(
        posterior_by_sector.transpose("flux_time", "country", "sector", "draw").values,
        dtype=np.float64,
    )
    if values.shape[3] == 0:
        sector_cross_covariance = np.full(
            (values.shape[0], values.shape[1], values.shape[2], values.shape[2]),
            np.nan,
            dtype=np.float64,
        )
    else:
        centered = values - values.mean(axis=3, keepdims=True)
        sector_cross_covariance = np.einsum("tcsd,tced->tcse", centered, centered) / values.shape[3]

    return sector_covariances, sector_cross_covariance


def _add_country_covariance(result: xr.Dataset, covariance: np.ndarray, attrs: dict[str, Any]) -> xr.Dataset:
    """Add latest PARIS country covariance using distinct country axes.

    Args:
        result: Dataset to modify in place.
        covariance: Time-major covariance with dimensions ``(time, country,
            country)``.
        attrs: Attributes for the emitted covariance variable.

    Returns:
        The same dataset, with covariance ordered as ``(country, country_2,
        time)``.
    """
    result["covariance_flux_total_posterior_country"] = (
        ("country", "country_2", "time"),
        covariance.transpose(1, 2, 0),
    )
    result["covariance_flux_total_posterior_country"].attrs = attrs
    return result


def _add_sector_country_covariances(
    result: xr.Dataset,
    sector_covariances: Mapping[str, np.ndarray],
    sector_cross_covariance: np.ndarray | None,
    attrs: Mapping[str, dict[str, Any]],
) -> xr.Dataset:
    """Add latest PARIS sector covariance variables using distinct axes.

    Args:
        result: Dataset to modify in place.
        sector_covariances: Per-sector covariance arrays with dimensions
            ``(time, country, country)``.
        sector_cross_covariance: Optional cross-sector covariance with
            dimensions ``(time, country, sector, sector)``.
        attrs: Attributes keyed by emitted covariance variable name.

    Returns:
        The same dataset. Country covariances are ordered as ``(country,
        country_2, time)`` and cross-sector covariance as ``(sector_2, sector,
        country, time)``.
    """
    for sector_name, covariance in sector_covariances.items():
        variable_name = f"covariance_flux_{sector_name}_posterior_country"
        result[variable_name] = (
            ("country", "country_2", "time"),
            covariance.transpose(1, 2, 0),
        )
        result[variable_name].attrs = attrs.get(variable_name, {})

    if sector_cross_covariance is not None:
        result["covariance_flux_sectors_posterior_country"] = (
            ("sector_2", "sector", "country", "time"),
            sector_cross_covariance.transpose(3, 2, 1, 0),
        )
        result["covariance_flux_sectors_posterior_country"].attrs = attrs.get(
            "covariance_flux_sectors_posterior_country",
            {},
        )

    return result


def paris_flux_output(
    inv_out: InversionOutput,
    country_file: str | Path | None = None,
    time_point: Literal["start", "midpoint"] = "midpoint",
    report_mode: bool = False,
    inversion_grid: bool = True,
    flux_frequency: Literal["monthly", "yearly"] | str = "yearly",
    template_version: ParisTemplateVersion = DEFAULT_PARIS_TEMPLATE_VERSION,
    country_selections: Iterable[str] | None = PARIS_LATEST_COUNTRIES,
) -> xr.Dataset:
    """Create a flux product using a selected PARIS template version.

    Args:
        inv_out: Inversion output with retained basis functions and flux traces.
        country_file: Optional country-definition NetCDF file.
        time_point: Flux timestamp convention.
        report_mode: If true, report KDE modes instead of means as central
            estimates.
        inversion_grid: If true, include reduced inversion-grid variables.
        flux_frequency: Frequency used to construct output flux intervals.
        template_version: PARIS template version to emit. The default preserves
            the legacy output contract; ``"latest"`` selects flux v03.
        country_selections: Optional country names or codes for the latest
            template. The default emits the canonical 22-country EUROPE v03
            schema; pass ``None`` to include all countries from another domain
            file. The legacy template ignores this option.

    Returns:
        PARIS flux dataset using the selected template's variable names,
        dimensions, dtypes, units, and attributes.

    Raises:
        ValueError: If required inversion metadata is missing or a latest-only
            time or frequency option is invalid.
    """
    if template_version == "latest":
        return paris_flux_output_latest(
            inv_out,
            country_file=country_file,
            country_selections=country_selections,
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
            midpoints, valid_indices = _flux_interval_midpoints(
                flux_times,
                flux_period,
                inv_start,
                inv_end,
                flux_frequency,
            )
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
    result = copy_flux_nonfinite_attrs(result, flux_outs)

    result = _cast_float_data_vars_to_float32(result).as_numpy()
    return add_basis_reconstruction_metadata(result, inv_out.basis_functions)


def paris_flux_output_latest(
    inv_out: InversionOutput,
    country_file: str | Path | None = None,
    time_point: Literal["start", "midpoint"] = "midpoint",
    report_mode: bool = False,
    inversion_grid: bool = True,
    flux_frequency: Literal["monthly", "yearly"] | str = "yearly",
    country_selections: Iterable[str] | None = PARIS_LATEST_COUNTRIES,
) -> xr.Dataset:
    """Create single- or multisector flux output using the latest PARIS template.

    Multisector output contains total and per-sector spatial and country
    statistics. Sector fluxes are reconstructed with their own prior fluxes,
    while total flux is calculated by summing those reconstructed traces.
    Country covariance variables use distinct country and sector axes required
    by the supplied CDL schema.

    Args:
        inv_out: Inversion output with retained basis functions and flux traces.
        country_file: Optional country-definition NetCDF file. If omitted, the
            country helper resolves its configured default for the inversion
            domain.
        time_point: Flux timestamp convention. The latest template supports only
            ``"midpoint"`` because it also reports interval bounds.
        report_mode: If true, report KDE modes instead of means as central flux
            estimates.
        inversion_grid: If true, include the optional reduced inversion-grid
            variables.
        flux_frequency: Period used to construct output intervals. Calendar
            values ``"monthly"`` and ``"yearly"`` and positive fixed pandas
            duration strings are supported.
        country_selections: Optional country names or codes to include. The
            default emits the canonical 22-country EUROPE v03 product; pass
            ``None`` to include every country from another domain file.

    Returns:
        NumPy-backed latest-template PARIS flux dataset with template dtypes
        and attributes. Country covariances use ``(country, country_2, time)``;
        cross-sector covariance uses ``(sector_2, sector, country, time)``.
        Secondary coordinates duplicate their primary-axis labels and
        attributes.

    Raises:
        ValueError: If midpoint times are not requested, required inversion or
            sector metadata is missing or invalid, the flux frequency is not
            supported, or no valid flux interval falls within the inversion.
    """
    if time_point != "midpoint":
        raise ValueError("Latest PARIS flux output requires midpoint time coordinates and time bounds.")

    species, domain = _require_paris_metadata(inv_out, allow_multisector=True)
    stats = ["kde_mode", "stdev", "quantiles"] if report_mode else ["mean", "stdev", "quantiles"]
    stats_args = {"quantiles__quantiles": [0.159, 0.841]}
    sector_name_by_suffix = _paris_sector_name_by_suffix(inv_out)
    sector_names = list(sector_name_by_suffix.values())

    spatial_flux_trace = (
        make_multisector_flux_trace_outputs(
            inv_out,
            report_flux_on_inversion_grid=False,
            materialize=False,
        )
        if inv_out.is_multisector
        else None
    )
    flux_outs = (
        calculate_stats(spatial_flux_trace, stats=stats, **stats_args)
        if spatial_flux_trace is not None
        else make_flux_outputs(
            inv_out,
            stats=stats,
            stats_args=stats_args,
            report_flux_on_inversion_grid=False,
            include_scale_factors=False,
        )
    )
    countries = _latest_paris_countries(
        country_file=country_file,
        domain=domain,
        country_selections=country_selections,
    )
    country_trace = (
        _multisector_country_trace_kg(
            inv_out,
            countries,
            sector_name_by_suffix,
        )
        if inv_out.is_multisector
        else _single_sector_country_trace_kg(inv_out, countries)
    )
    country_outs = _latest_country_outputs(
        inv_out,
        countries=countries,
        stats=stats,
        stats_args=stats_args,
        sector_name_by_suffix=sector_name_by_suffix,
        multisector_country_trace=country_trace,
    )
    country_fraction = countries.matrix.as_numpy().rename("country_fraction")
    cell_area = countries.area_grid.as_numpy().rename("cell_area")
    country_covariance = _country_posterior_covariance_kg(
        inv_out,
        countries=countries,
        flux_frequency=flux_frequency,
        multisector_country_trace=country_trace,
    )
    sector_covariances, sector_cross_covariance = (
        _sector_country_posterior_covariances_kg(
            inv_out,
            countries=countries,
            flux_frequency=flux_frequency,
            sector_name_by_suffix=sector_name_by_suffix,
            multisector_country_trace=country_trace,
        )
        if inv_out.is_multisector
        else ({}, None)
    )

    def flux_renamer(name: str) -> str:
        for variable_suffix, sector_name in sorted(
            sector_name_by_suffix.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            sector_prefix = f"flux_{variable_suffix}_"
            remainder = name.removeprefix(sector_prefix)
            if remainder != name and remainder.startswith(("prior_", "posterior_")):
                name = f"flux_{sector_name}_{remainder}"
                break
        else:
            if name.startswith("flux_") and not name.startswith("flux_total_"):
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
        flux_label = "total"
        for sector_name in sorted(sector_names, key=len, reverse=True):
            sector_prefix = f"{sector_name}_"
            remainder = suffix.removeprefix(sector_prefix)
            if remainder != suffix and remainder.startswith(("prior_", "posterior_")):
                flux_label = sector_name
                suffix = remainder
                break

        if "quantile" in suffix:
            suffix = suffix.replace("_quantile", "")
            return f"percentile_flux_{flux_label}_{suffix}_country"
        if suffix.endswith("_stdev"):
            suffix = suffix.removesuffix("_stdev")
            return f"stdev_flux_{flux_label}_{suffix}_country"
        for stats_func_name in stats_functions:
            if suffix.endswith(f"_{stats_func_name}"):
                suffix = suffix.removesuffix(f"_{stats_func_name}")
        return f"flux_{flux_label}_{suffix}_country"

    flux_rename_dict = {str(dv): flux_renamer(str(dv)) for dv in flux_outs.data_vars}
    country_rename_dict = {str(dv): country_renamer(str(dv)) for dv in country_outs.data_vars}

    dim_rename_dict = {"quantile": "percentile", "flux_time": "time"}
    if "lat" in flux_outs.dims:
        dim_rename_dict["lat"] = "latitude"
    if "lon" in flux_outs.dims:
        dim_rename_dict["lon"] = "longitude"

    template_files = paris_template_files("latest")
    emissions_attrs = cast(
        dict[str, dict[str, Any]],
        _expand_sector_template_mapping(get_data_var_attrs(template_files.flux, species), sector_names),
    )

    result = (
        xr.merge(
            [
                flux_outs,
                country_outs,
                align_sparse_lat_lon(country_fraction, flux_outs),
                align_sparse_lat_lon(cell_area, flux_outs),
            ],
            join="outer",
        )
        .rename(dim_rename_dict)
        .pipe(_assign_flux_time_bounds, flux_frequency, inv_out.start_time, inv_out.end_time)
        .pipe(_convert_flux_time_and_bounds_to_epoch_days)
        .rename({**flux_rename_dict, **country_rename_dict})
    )
    result = result.assign_coords(country_2=("country_2", np.asarray(result["country"].values, dtype=object)))
    if sector_names:
        sector_values = np.asarray(sector_names, dtype=object)
        result = result.assign_coords(
            sector=("sector", sector_values),
            sector_2=("sector_2", sector_values.copy()),
        )
    result = result.pipe(add_variable_attrs, emissions_attrs)

    if inversion_grid:
        inversion_grid_flux_rename_dict = {v: f"{v}_inversion_grid" for v in flux_rename_dict.values()}
        inversion_grid_flux_outs_raw = make_flux_outputs(
            inv_out,
            stats=stats,
            stats_args=stats_args,
            report_flux_on_inversion_grid=True,
            include_scale_factors=False,
        )
        inversion_grid_flux_outs = (
            inversion_grid_flux_outs_raw.rename(dim_rename_dict)
            .pipe(_assign_flux_time_bounds, flux_frequency, inv_out.start_time, inv_out.end_time)
            .pipe(_convert_flux_time_and_bounds_to_epoch_days)
            .rename(flux_rename_dict)
            .rename(inversion_grid_flux_rename_dict)
            .pipe(add_variable_attrs, emissions_attrs)
        )
        result = result.merge(inversion_grid_flux_outs)

    result = result.transpose(
        "percentile",
        "sector_2",
        "sector",
        "country",
        "country_2",
        "time",
        "latitude",
        "longitude",
        "nbnds",
        missing_dims="ignore",
    ).as_numpy()
    result = _add_country_covariance(
        result,
        country_covariance,
        emissions_attrs.get("covariance_flux_total_posterior_country", {}),
    )
    result = _add_sector_country_covariances(
        result,
        sector_covariances,
        sector_cross_covariance,
        emissions_attrs,
    )
    result.attrs = make_global_attrs("flux", species=species, domain=domain)
    result.attrs["paris_flux_template_version"] = template_files.flux_version
    result = copy_flux_nonfinite_attrs(
        result,
        spatial_flux_trace if spatial_flux_trace is not None else flux_outs,
    )

    result = _cast_data_vars_to_template_dtypes(result, template_files.flux, sector_names=sector_names)
    result = add_basis_reconstruction_metadata(result, inv_out.basis_functions)
    return _prepare_latest_paris_netcdf_encoding(result)


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
        ValueError: If source metadata is neither a recognized calendar period
            nor a positive fixed duration.

    """
    if "time_period" in flux.attrs:
        time_period = flux.attrs["time_period"]
        normalized_period = utils._normalize_flux_period(time_period)
        if normalized_period is not None:
            return normalized_period
        if not utils._flux_period_is_missing(time_period):
            raise ValueError(
                f"Flux period {time_period!r} from flux.attrs['time_period'] is not a recognized "
                "calendar period or a positive fixed duration."
            )

    calendar_period = utils._infer_calendar_flux_period(flux.flux_time.values)
    if calendar_period is not None:
        return calendar_period

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
    country_selections: Iterable[str] | None = PARIS_LATEST_COUNTRIES,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Create matching PARIS flux and concentration products.

    Args:
        inv_out: Inversion output containing observations, model traces,
            retained basis functions, and prior fluxes.
        country_file: Optional country-definition NetCDF file.
        time_point: Flux timestamp convention.
        report_mode: If true, report KDE modes instead of means as central
            estimates.
        inversion_grid: If true, include reduced inversion-grid flux variables.
        obs_avg_period: Averaging period recorded in concentration metadata.
        domain: Optional domain override for concentration metadata.
        template_version: PARIS template version to emit. ``"latest"`` selects
            concentration v04 and flux v03.
        country_selections: Optional latest-template country names or codes.
            The default emits the canonical 22-country EUROPE v03 schema; pass
            ``None`` to include all countries from another domain file.

    Returns:
        A tuple containing the flux dataset followed by the concentration
        dataset.

    Raises:
        ValueError: If required inversion metadata is missing or template
            constraints are not satisfied.
    """
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
        country_selections=country_selections,
        inversion_grid=inversion_grid,
        time_point=time_point,
        flux_frequency=flux_frequency,
        template_version=template_version,
    )

    return flux_outs, conc_outs
