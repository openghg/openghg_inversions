from pathlib import Path
import getpass
import re
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from openghg.util import timestamp_now
from openghg_inversions import convert
from openghg_inversions.config.version import code_version
from openghg_inversions.postprocessing.countries import Countries, paris_regions_dict
from openghg_inversions.postprocessing.inversion_output import (
    InversionOutput,
    make_inv_out_from_rhime_outputs,
)
from openghg_inversions.postprocessing.make_outputs import (
    make_concentration_outputs,
    make_flux_outputs,
)
from openghg_inversions.postprocessing.stats import stats_functions, calculate_stats
from openghg_inversions.postprocessing.utils import rename_by_replacement
from openghg_inversions.array_ops import sparse_xr_dot
from openghg_inversions.utils import get_country_file_path


# path to `paris_formatting` submodule
paris_formatting_path = Path(__file__).parent

# paths to template files
conc_template_path = paris_formatting_path / "PARIS_Lagrangian_inversion_concentration_EUROPE_v03.cdl"
flux_template_path = paris_formatting_path / "PARIS_Lagrangian_inversion_flux_EUROPE.cdl"


var_pat = re.compile(r"\s*[a-z]+ ([a-zA-Z_]+)\(.*\)")
attr_pat = re.compile(r"\s+([a-zA-Z_]+):([a-zA-Z_]+)\s*=\s*([^;]+)")


def nested_inner_domain_label(domain: str, inner_domain: str | None) -> str:
    """Return the normalized nested-domain label used in inner PARIS outputs."""
    if inner_domain is None:
        return domain

    domain_label = str(domain).lower()
    inner_label = str(inner_domain).lower()
    if domain_label.endswith(f"-{inner_label}"):
        return domain_label
    return f"{domain_label}-{inner_label}"


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


def _require_nonempty_flux_output(ds: xr.Dataset, *, label: str) -> None:
    """Fail before writing an empty PARIS flux file."""
    dims = {dim: int(size) for dim, size in ds.sizes.items()}
    flux_vars = [str(name) for name in ds.data_vars if str(name).startswith("flux")]
    print(
        "DIAGNOSTIC paris_flux_output | "
        f"label={label} dims={dims} flux_vars={flux_vars}",
        flush=True,
    )
    if not flux_vars:
        raise ValueError(f"PARIS {label} flux output has no flux variables.")

    empty_dims = {dim: size for dim, size in dims.items() if size == 0}
    if empty_dims:
        raise ValueError(
            f"PARIS {label} flux output is empty along dimensions {empty_dims}. "
            "Check basis/flux time alignment before writing outputs."
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
    """Update data variables and coordinates of Dataset based on attributes dictionary."""
    for k, v in attrs.items():
        if k in ds.data_vars:
            if units is not None and "units" in v and v["units"].count("mol") == 2:
                ds[k] = units * ds[k]
            ds[k].attrs = v
        elif k in ds.coords:
            ds.coords[k].attrs = v

    return ds


def convert_time_to_unix_epoch(x: xr.Dataset, units: str = "1s") -> xr.Dataset:
    """Convert `time` coordinate to number of 'units' since 1 Jan 1970."""
    time_converted = (pd.DatetimeIndex(x.time) - pd.Timestamp("1970-01-01")) / pd.Timedelta(units)
    return x.assign_coords(time=time_converted)


def shift_measurement_time_to_midpoint(ds: xr.Dataset, period: str = "4h") -> xr.Dataset:
    """Adjust `time` coordinate of concentrations to represent half averaging 'period'."""
    time_shifted = pd.to_datetime(ds["time"].astype("datetime64[ns]").values) + pd.to_timedelta(period) / 2
    ds = ds.assign_coords(time=time_shifted)
    return ds

def _densify_dataarray(da: xr.DataArray) -> xr.DataArray:
    """Return a copy of da with sparse backing replaced by dense numpy array."""
    data = da.data
    if hasattr(data, "todense"):
        data = np.asarray(data.todense())
    return da.copy(data=data)

def _densify(da: xr.DataArray) -> np.ndarray:
    """Extract numpy array from DataArray, densifying sparse backing if necessary."""
    data = da.data
    if hasattr(data, "todense"):
        data = data.todense()
    return np.asarray(data)


def _fine_coord_over_outer_extent(outer_coord: xr.DataArray, inner_coord: xr.DataArray) -> xr.DataArray:
    """Create a coordinate at inner resolution over the full outer extent."""
    if inner_coord.size < 2:
        return outer_coord

    step = float(abs(inner_coord.diff(inner_coord.dims[0]).median()))
    if step == 0.0:
        return outer_coord

    outer_values = np.asarray(outer_coord.values, dtype=float)
    ascending = outer_values[-1] >= outer_values[0]
    start = float(outer_values.min())
    stop = float(outer_values.max())
    values = np.arange(start, stop + step * 0.5, step)
    if not ascending:
        values = values[::-1]

    return xr.DataArray(values, dims=outer_coord.dims, name=outer_coord.name)


def _interp_to_grid_nearest(ds: xr.Dataset | xr.DataArray, lat: xr.DataArray, lon: xr.DataArray):
    """Interpolate to target lat/lon using nearest values without zero-filling."""
    return ds.interp(lat=lat, lon=lon, method="nearest", kwargs={"fill_value": "extrapolate"})


def _inner_extent_mask_on_grid(inner_lat: xr.DataArray, inner_lon: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray):
    """Return target-grid mask for the inner-domain lat/lon extent."""
    lat_mask = (lat >= float(inner_lat.min())) & (lat <= float(inner_lat.max()))
    lon_mask = (lon >= float(inner_lon.min())) & (lon <= float(inner_lon.max()))
    target = xr.DataArray(
        np.zeros((lat.size, lon.size), dtype=bool),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
    )
    return (lat_mask & lon_mask).broadcast_like(target)


def _reindex_flux_time_nearest(ds: xr.Dataset, target_flux_time: xr.DataArray) -> xr.Dataset:
    """Align flux_time to target using nearest values where needed."""
    if "flux_time" not in ds.dims or "flux_time" not in target_flux_time.dims:
        return ds
    if np.array_equal(ds.flux_time.values, target_flux_time.values):
        return ds
    return ds.reindex(flux_time=target_flux_time, method="nearest")


def _percentile_value(da: xr.DataArray, value: float) -> xr.DataArray:
    if "percentile" not in da.coords:
        raise ValueError(f"{da.name or 'DataArray'} has no percentile coordinate.")

    percentile_values = np.asarray(da["percentile"].values, dtype=float)
    index = int(np.argmin(np.abs(percentile_values - value)))
    if abs(percentile_values[index] - value) > 1e-6:
        raise ValueError(
            f"Could not find percentile {value}; available percentiles are {percentile_values}."
        )
    return da.isel(percentile=index)


def _finite_stats(da: xr.DataArray) -> tuple[float, float]:
    values = np.asarray(da.compute().values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(np.nanmedian(values)), float(np.nanpercentile(values, 90))


def _finite_corr(a: xr.DataArray, b: xr.DataArray) -> float:
    a_values = np.asarray(a.compute().values, dtype=float).ravel()
    b_values = np.asarray(b.compute().values, dtype=float).ravel()
    finite = np.isfinite(a_values) & np.isfinite(b_values)
    if finite.sum() < 2:
        return np.nan
    return float(np.corrcoef(a_values[finite], b_values[finite])[0, 1])


def _country_flux_uncertainty_diagnostics(ds: xr.Dataset, label: str) -> xr.Dataset:
    """Print and attach compact diagnostics for country flux uncertainty."""
    required = [
        "country_flux_total_posterior",
        "country_flux_total_prior",
        "percentile_country_flux_total_posterior",
        "percentile_country_flux_total_prior",
    ]
    if any(name not in ds.data_vars for name in required):
        return ds

    def width_and_relative(mean_name: str, percentile_name: str) -> tuple[xr.DataArray, xr.DataArray, int, int, int]:
        mean = ds[mean_name]
        percentiles = ds[percentile_name]
        lower = _percentile_value(percentiles, 0.159)
        upper = _percentile_value(percentiles, 0.841)
        width = upper - lower
        valid_mean = abs(mean) > 1e-30
        rel_width = width.where(valid_mean) / abs(mean.where(valid_mean))
        percentile_order_violations = (upper < lower).fillna(False)
        mean_outside_interval = ((mean < lower) | (mean > upper)).fillna(False)
        nonzero_mean_count = valid_mean.fillna(False)
        return (
            width,
            rel_width,
            int(percentile_order_violations.sum().compute()),
            int(mean_outside_interval.sum().compute()),
            int(nonzero_mean_count.sum().compute()),
        )

    (
        posterior_width,
        posterior_rel_width,
        posterior_order_violations,
        posterior_mean_outside,
        posterior_nonzero_mean_count,
    ) = width_and_relative(
        "country_flux_total_posterior",
        "percentile_country_flux_total_posterior",
    )
    (
        prior_width,
        prior_rel_width,
        prior_order_violations,
        prior_mean_outside,
        prior_nonzero_mean_count,
    ) = width_and_relative(
        "country_flux_total_prior",
        "percentile_country_flux_total_prior",
    )

    rel_inflation = posterior_rel_width / prior_rel_width.where(abs(prior_rel_width) > 1e-30)
    posterior_rel_median, posterior_rel_p90 = _finite_stats(posterior_rel_width)
    prior_rel_median, prior_rel_p90 = _finite_stats(prior_rel_width)
    inflation_median, inflation_p90 = _finite_stats(rel_inflation)
    width_flux_corr = _finite_corr(abs(ds["country_flux_total_posterior"]), posterior_width)

    diagnostic = (
        f"label={label} posterior_rel_width_median={posterior_rel_median:.3f} "
        f"posterior_rel_width_p90={posterior_rel_p90:.3f} "
        f"prior_rel_width_median={prior_rel_median:.3f} prior_rel_width_p90={prior_rel_p90:.3f} "
        f"posterior_prior_rel_width_ratio_median={inflation_median:.3f} "
        f"posterior_prior_rel_width_ratio_p90={inflation_p90:.3f} "
        f"posterior_width_abs_flux_corr={width_flux_corr:.3f} "
        f"percentile_order_violations={posterior_order_violations + prior_order_violations} "
        f"mean_outside_68pct_interval={posterior_mean_outside + prior_mean_outside} "
        f"posterior_nonzero_mean_count={posterior_nonzero_mean_count} "
        f"prior_nonzero_mean_count={prior_nonzero_mean_count}"
    )
    print(f"DIAGNOSTIC country_flux_uncertainty | {diagnostic}", flush=True)

    ds = ds.copy()
    ds.attrs[f"{label}_country_flux_uncertainty_diagnostic"] = diagnostic
    return ds


def _paris_country_regions(domain: str) -> dict[str, list[str]] | None:
    return paris_regions_dict.get(domain.lower())


def _paris_countries(country_file: str | Path | None, domain: str) -> Countries:
    return Countries.from_file(
        country_file=country_file,
        country_code="alpha3",
        country_regions=_paris_country_regions(domain),
        domain=domain,
    )

def make_inner_domain_country_outputs(
    flux_stats: xr.Dataset,
    countries: Countries,
    species: str,
) -> xr.Dataset:
    inner_lat = flux_stats.lat
    inner_lon = flux_stats.lon

    # Densify before interp — scipy cannot handle sparse-backed arrays
    fine_country_matrix = _densify_dataarray(countries.matrix).interp(
        lat=inner_lat,
        lon=inner_lon,
        method="nearest",
        kwargs={"fill_value": 0.0},
    ).fillna(0.0)

    dlat_deg = float(abs(inner_lat.diff("lat").median()))
    dlon_deg = float(abs(inner_lon.diff("lon").median()))
    R = 6.371e6
    lat_rad = np.deg2rad(inner_lat)
    cell_area = (
        np.cos(lat_rad)
        * np.deg2rad(dlat_deg)
        * np.deg2rad(dlon_deg)
        * R ** 2
    )

    country_stats = sparse_xr_dot(fine_country_matrix, (cell_area * flux_stats).fillna(0.0))

    seconds_per_year = 365 * 24 * 3600
    country_stats = country_stats * seconds_per_year * convert.molar_mass(species) * 1e-3
    country_stats = rename_by_replacement(country_stats, "flux", "country")

    for dv in country_stats.data_vars:
        suffix = str(dv).removeprefix("country_")
        country_stats[dv].attrs["units"] = "kg a-1"
        country_stats[dv].attrs["long_name"] = (
            f"inner-domain country-total {suffix} {species} fluxes at 6 km resolution"
        )

    return country_stats.as_numpy()


def _get_inner_trace_dataset(inv_out: InversionOutput, n_inner_nx: int) -> xr.Dataset:
    """Return inner-domain x traces with variables named like the standard x trace."""
    trace_ds = inv_out.get_trace_dataset()
    inner_vars = [str(dv) for dv in trace_ds.data_vars if str(dv).startswith("x_inner_")]

    if inner_vars:
        inner_trace = trace_ds[inner_vars].rename_vars(
            {name: name.replace("x_inner_", "x_", 1) for name in inner_vars}
        )
        if "nx_inner" in inner_trace.dims:
            inner_trace = inner_trace.rename({"nx_inner": "nx"})
    else:
        combined_trace = inv_out.get_trace_dataset(var_names="x")
        if "nx" not in combined_trace.dims:
            raise ValueError("Cannot create inner PARIS flux output: no nx dimension found in x trace.")
        if combined_trace.sizes["nx"] < n_inner_nx:
            raise ValueError(
                "Cannot create inner PARIS flux output: inner basis has "
                f"{n_inner_nx} regions but the available x trace has only "
                f"{combined_trace.sizes['nx']} regions. Expected an x_inner trace "
                "or a combined x trace containing the inner block first."
            )
        inner_trace = combined_trace.isel(nx=slice(0, n_inner_nx))

    if inner_trace.sizes.get("nx") != n_inner_nx:
        raise ValueError(
            "Cannot create inner PARIS flux output: inner trace has "
            f"{inner_trace.sizes.get('nx')} regions but inner basis has {n_inner_nx}."
        )

    return inner_trace.assign_coords(nx=inv_out.inner_basis.nx.values)


def paris_concentration_outputs(
    inv_out: InversionOutput, report_mode: bool = False, obs_avg_period: str = "4h"
) -> xr.Dataset:
    """Create PARIS concentration outputs."""
    stats = ["kde_mode", "quantiles"] if report_mode else ["mean", "quantiles"]
    stats_args = {"quantiles__quantiles": [0.159, 0.841]}

    obs_and_errs_raw = inv_out.get_obs_and_errors().unstack("nmeasure")
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
    obs_and_errs = (
        obs_and_errs_raw
        .rename(filtered_rename_map)
        .drop_vars("y_obs_error")
    )

    conc_outputs = make_concentration_outputs(
        inv_out, stats, stats_args, combine_bc_and_offset=True, concentration_variable="mu"
    ).unstack("nmeasure")

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

    if "qYapostBC" in conc_outputs.data_vars:
        conc_outputs = conc_outputs.drop_vars(["qYapostBC", "qYaprioriBC"])
    if "qYapost_bias" in conc_outputs.data_vars:
        conc_outputs = conc_outputs.drop_vars(["qYapost_bias", "qYapriori_bias"])

    conc_attrs = get_data_var_attrs(conc_template_path)
    units = float(inv_out.obs.attrs["units"].split(" ")[0])
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
    result.attrs = make_global_attrs("conc", species=inv_out.species, domain=inv_out.domain)
    result.attrs["prior_factor_adjustment_applied"] = "false"

    if "Yobs_prior_factor" in result.data_vars and "Yobs_prior_upper_level_factor" in result.data_vars:
        result.attrs["prior_factor_adjustment_applied"] = "true"

    return result


def paris_flux_output(
    inv_out: InversionOutput,
    country_file: str | Path | None = None,
    time_point: Literal["start", "midpoint"] = "midpoint",
    report_mode: bool = False,
    inversion_grid: bool = True,
    flux_frequency: Literal["monthly", "yearly"] | str = "yearly",
    inner_domain: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """Create PARIS flux outputs.

    Returns:
        Tuple of (outer_flux_dataset, inner_flux_dataset).
        inner_flux_dataset is None when no inner domain is present.
        The inner dataset is at the native fine-grid resolution (e.g. 6 km).
        The outer dataset remains on the standard-domain grid and includes
        inner-domain flux values regridded onto that grid.
    """
    stats = ["kde_mode", "quantiles"] if report_mode else ["mean", "quantiles"]
    stats_args = {"quantiles__quantiles": [0.159, 0.841]}

    # ------------------------------------------------------------------ #
    # Outer domain                                                         #
    # ------------------------------------------------------------------ #
    flux_outs = make_flux_outputs(
        inv_out,
        stats=stats,
        stats_args=stats_args,
        report_flux_on_inversion_grid=False,
        include_scale_factors=False,
    ).fillna(0.0)
    _require_nonempty_flux_output(flux_outs, label="outer_initial")

    countries = _paris_countries(country_file=country_file, domain=inv_out.domain)
    emissions_attrs = get_data_var_attrs(flux_template_path, inv_out.species)

    inner_flux_stats: xr.Dataset | None = None
    inner_stats_ds: xr.Dataset | None = None

    if inv_out.inner_basis is not None and inv_out.inner_flux is not None:
        n_inner_nx = len(inv_out.inner_basis.nx)
        inner_trace_ds = _get_inner_trace_dataset(inv_out, n_inner_nx)
        inner_stats_args = {**stats_args, "stats": stats, "chunk_dim": "nx"}
        inner_stats_ds = calculate_stats(inner_trace_ds, **inner_stats_args)
        inner_flux = inv_out.inner_flux.fillna(0.0)
        inner_flux_stats = sparse_xr_dot(inner_flux * inv_out.inner_basis, inner_stats_ds)
        inner_flux_stats = rename_by_replacement(inner_flux_stats, "x", "flux").fillna(0.0)

        inner_on_standard = _interp_to_grid_nearest(inner_flux_stats, lat=flux_outs.lat, lon=flux_outs.lon)
        inner_on_standard = _reindex_flux_time_nearest(inner_on_standard, flux_outs.flux_time)
        inner_extent_on_standard = _inner_extent_mask_on_grid(
            inv_out.inner_flux.lat,
            inv_out.inner_flux.lon,
            flux_outs.lat,
            flux_outs.lon,
        )

        combined_vars = {}
        for name in flux_outs.data_vars:
            if name in inner_on_standard:
                combined_vars[name] = xr.where(inner_extent_on_standard, inner_on_standard[name], flux_outs[name])
        flux_outs = flux_outs.assign(combined_vars)
        flux_outs.attrs["nested_output_grid"] = (
            "standard-domain grid with inner-domain flux regridded by nearest neighbour over inner extent"
        )
        _require_nonempty_flux_output(flux_outs, label="outer_output_grid")

    output_lat = flux_outs.lat
    output_lon = flux_outs.lon

    country_outs = make_inner_domain_country_outputs(
        flux_outs,
        countries,
        inv_out.species,
    )

    country_fraction = (
        _densify_dataarray(countries.matrix)
        .interp(lat=output_lat, lon=output_lon, method="nearest", kwargs={"fill_value": "extrapolate"})
        .fillna(0.0)
        .as_numpy()
        .rename("country_fraction")
    )

    def renamer(name: str) -> str:
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

        def time_func(ds: xr.Dataset) -> xr.Dataset:
            return ds.assign_coords(time=(pd.to_datetime(ds.time.values) + offset))
    else:
        def time_func(ds: xr.Dataset) -> xr.Dataset:
            return ds

    result = (
        xr.merge([flux_outs, country_outs, country_fraction], join="outer")
        .rename(dim_rename_dict)
        .pipe(time_func)
        .pipe(convert_time_to_unix_epoch, "1d")
        .rename(rename_dict)
        .pipe(add_variable_attrs, emissions_attrs)
    )

    if inversion_grid:
        inversion_grid_flux_rename_dict = {v: f"{v}_inversion_grid" for v in flux_rename_dict.values()}
        inversion_grid_flux_outs = (
            flux_outs.copy()
            .rename(dim_rename_dict)
            .pipe(time_func)
            .pipe(convert_time_to_unix_epoch, "1d")
            .rename(flux_rename_dict)
            .pipe(add_variable_attrs, emissions_attrs)
            .rename(inversion_grid_flux_rename_dict)
        )
        result = result.merge(inversion_grid_flux_outs)

    result = result.transpose("time", "percentile", "country", "latitude", "longitude")
    result.attrs = make_global_attrs("flux", species=inv_out.species, domain=inv_out.domain)
    result = _country_flux_uncertainty_diagnostics(result, label="outer")

    # ------------------------------------------------------------------ #
    # Inner domain (fine grid, e.g. 6 km) — separate output              #
    # ------------------------------------------------------------------ #
    if inv_out.inner_basis is None or inv_out.inner_flux is None or inner_flux_stats is None or inner_stats_ds is None:
        return result.as_numpy(), None

    # Reconstruct flux at native 6 km resolution: sum_k( x[k] * basis[k] * flux )
    # Result has dims (flux_time, lat_inner, lon_inner, quantile) — full spatial detail preserved
    inner_country_outs = make_inner_domain_country_outputs(
            inner_flux_stats, countries, inv_out.species
        )

    # Keep native prior-flux texture in the PARIS inversion-grid variables.
    # Fluxie reads these names for plot_inversion_flux_grid, so using region
    # average flux here hides the 6 km detail carried by inner_flux_stats.
    inner_inversion_grid_flux_raw = inner_flux_stats.copy()

    def paris_renamer(name: str) -> str:
        """Convert raw stat names to PARIS convention.

            Input names look like:
              flux_posterior_mean       -> flux_total_posterior
              flux_prior_mean           -> flux_total_prior
              flux_posterior_quantile   -> percentile_flux_total_posterior
              flux_prior_quantile       -> percentile_flux_total_prior
              country_posterior_mean    -> country_flux_total_posterior
              country_prior_mean        -> country_flux_total_prior
              country_posterior_quantile -> percentile_country_flux_total_posterior
              country_prior_quantile    -> percentile_country_flux_total_prior
        """
        is_country = name.startswith("country_")
        is_percentile = "quantile" in name

        # Determine posterior/prior
        if "posterior" in name:
            when = "posterior"
        elif "prior" in name:
            when = "prior"
        else:
            # fallback: keep as-is for unrecognised names
            return name

        if is_country:
            base = "country_flux_total"
        else:
            base = "flux_total"

        if is_percentile:
            return f"percentile_{base}_{when}"
        else:
            return f"{base}_{when}"

        # Build rename dicts using paris_renamer on the raw variable names
    inner_flux_rename = {
        str(dv): paris_renamer(str(dv))
        for dv in inner_flux_stats.data_vars
    }
    inner_country_rename = {
        str(dv): paris_renamer(str(dv))
        for dv in inner_country_outs.data_vars
    }
    inner_inv_grid_rename = {
        str(dv): paris_renamer(str(dv)) + "_inversion_grid"
        for dv in inner_inversion_grid_flux_raw.data_vars
    }

    # Dim rename: use same names as outer so the inner file is self-consistent
    inner_dim_rename: dict[str, str] = {"quantile": "percentile", "flux_time": "time"}
    inner_dim_rename["lat"] = "latitude"
    inner_dim_rename["lon"] = "longitude"

    inner_flux_out = (
        inner_flux_stats
        .rename(inner_dim_rename)
        .rename(inner_flux_rename)
        .pipe(time_func)
        .pipe(convert_time_to_unix_epoch, "1d")
        .pipe(add_variable_attrs, emissions_attrs)
    )

    inner_country_out = (
        inner_country_outs
        .rename({"quantile": "percentile", "flux_time": "time"})
        .rename(inner_country_rename)
        .pipe(time_func)
        .pipe(convert_time_to_unix_epoch, "1d")
        .pipe(add_variable_attrs, emissions_attrs)
        .transpose("time", "percentile", "country")
    )

    inner_inversion_grid_out = (
        inner_inversion_grid_flux_raw
        .rename(inner_dim_rename)
        .rename(inner_inv_grid_rename)
        .pipe(time_func)
        .pipe(convert_time_to_unix_epoch, "1d")
        .pipe(add_variable_attrs, emissions_attrs)
    )

    # Fine-grid country fraction mask
    inner_country_fraction = _densify_dataarray(countries.matrix).interp(
        lat=inv_out.inner_flux.lat,
        lon=inv_out.inner_flux.lon,
        method="nearest",
        kwargs={"fill_value": 0.0},
    ).fillna(0.0).as_numpy().rename({"lat": "latitude", "lon": "longitude"}).rename("country_fraction") 

    # Set attrs manually — template won't match after rename
    inner_country_fraction.attrs = {
            "units": "1",
            "long_name": "fraction of grid cell associated to country",
        }
    # Transpose inner outputs to match outer dim ordering: (time, [percentile,] lat, lon)
    # and manually copy attributes for inversion_grid vars which don't match template keys
    def _fix_inner_inv_grid_attrs_and_order(ds: xr.Dataset) -> xr.Dataset:
        """Transpose dims and copy attrs from base var to _inversion_grid variant."""
        result_vars = {}
        for dv in ds.data_vars:
            da = ds[dv]
            dv_str = str(dv)

            # Copy attrs from the matching non-inversion_grid var in emissions_attrs
            # e.g. "flux_total_posterior_inversion_grid" -> look up "flux_total_posterior"
            if not da.attrs and dv_str.endswith("_inversion_grid"):
                base_name = dv_str.removesuffix("_inversion_grid")
                if base_name in emissions_attrs:
                    da = da.copy()
                    da.attrs = dict(emissions_attrs[base_name])

            # Reorder dims to (time, percentile, latitude, longitude) where present
            target_dim_order = [
                d for d in ["time", "percentile", "latitude", "longitude"]
                if d in da.dims
            ]
            # append any unexpected extra dims at the end
            target_dim_order += [d for d in da.dims if d not in target_dim_order]
            da = da.transpose(*target_dim_order)

            result_vars[dv_str] = da

        return ds.assign(result_vars)

    inner_flux_out = _fix_inner_inv_grid_attrs_and_order(inner_flux_out)
    inner_inversion_grid_out = _fix_inner_inv_grid_attrs_and_order(inner_inversion_grid_out)

    inner_result = xr.merge(
        [inner_flux_out, inner_country_out, inner_inversion_grid_out, inner_country_fraction],
        join="outer",
    )
    inner_domain_label = nested_inner_domain_label(inv_out.domain, inner_domain)
    inner_result.attrs = make_global_attrs("flux", species=inv_out.species, domain=inner_domain_label)
    inner_result.attrs["inner_domain"] = inner_domain_label
    inner_result.attrs["spatial_resolution"] = inner_domain_label
    inner_result = _country_flux_uncertainty_diagnostics(inner_result, label="inner")

    inner_result = inner_result.astype({v: "float32" for v in inner_result.data_vars})

    return result.as_numpy(), inner_result.as_numpy()


def infer_flux_frequency(flux: xr.DataArray) -> str:
    """Attempt to infer flux frequency."""
    if "time_period" in flux.attrs:
        time_period = flux.attrs["time_period"]
        if "year" in time_period:
            return "yearly"
        if "month" in time_period:
            return "monthly"
        try:
            pd.to_timedelta(time_period)
        except ValueError as e:
            raise ValueError(
                f"Flux frequency {time_period} from flux.attrs['time_period'] cannot be parsed by pd.to_timedelta."
            ) from e
        else:
            return time_period
    else:
        try:
            flux_frequency_delta = pd.Series(flux.flux_time.values).diff().mode()[0]
        except KeyError:
            return "yearly"
        else:
            flux_frequency = pd.tseries.frequencies.to_offset(flux_frequency_delta).freqstr  # type: ignore
            if not flux_frequency[0].isdigit():
                flux_frequency = "1" + flux_frequency
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
    inner_domain: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset | None, xr.Dataset]:
    """Create all PARIS outputs.

    Returns:
        Tuple of (flux_outs, inner_flux_outs, conc_outs).
        inner_flux_outs is None when no inner domain is present.
        When present, inner_flux_outs is a separate dataset at the native
        fine-grid resolution (e.g. 6 km), suitable for saving as a separate
        netCDF file using the standard PARIS filename builder for the nested domain.
        If `inner_domain` is provided, the inner output domain metadata is
        normalized as "<outer-domain>-<inner-domain>" (for example, "europe-6km").
    """
    flux_frequency = infer_flux_frequency(inv_out.flux)
    conc_outs = paris_concentration_outputs(inv_out, report_mode=report_mode, obs_avg_period=obs_avg_period)
    flux_outs, inner_flux_outs = paris_flux_output(
        inv_out,
        report_mode=report_mode,
        country_file=country_file,
        inversion_grid=inversion_grid,
        time_point=time_point,
        flux_frequency=flux_frequency,
        inner_domain=inner_domain,
    )

    return flux_outs, inner_flux_outs, conc_outs


def make_paris_flux_outputs_from_rhime(
    rhime_outputs: xr.Dataset,
    species: str,
    domain: str,
    country_file: str | Path | None = None,
    time_point: Literal["start", "midpoint"] = "midpoint",
    report_mode: bool = False,
    inversion_grid: bool = True,
    flux_frequency: Literal["monthly", "yearly"] | str = "yearly",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    inv_out = make_inv_out_from_rhime_outputs(
        rhime_outputs, species=species, domain=domain, start_date=start_date, end_date=end_date
    )
    flux_outputs, inner_flux_outs = paris_flux_output(
        inv_out, country_file, time_point, report_mode, inversion_grid, flux_frequency
    )
    return flux_outputs, inner_flux_outs
