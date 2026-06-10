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
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.postprocessing.inversion_output import (
    InversionOutput,
    make_inv_out_from_rhime_outputs,
)
from openghg_inversions.postprocessing.make_outputs import (
    make_concentration_outputs,
    make_flux_outputs,
    make_country_outputs,
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

    country_stats = sparse_xr_dot(fine_country_matrix, cell_area * flux_stats)

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
        inv_out, stats, stats_args, combine_bc_and_offset=True
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
    result.attrs = make_global_attrs("conc")
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
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """Create PARIS flux outputs.

    Returns:
        Tuple of (outer_flux_dataset, inner_flux_dataset).
        inner_flux_dataset is None when no inner domain is present.
        The inner dataset is at the native fine-grid resolution (e.g. 6 km),
        stitched with the outer domain interpolated to that grid wherever the
        inner basis has no coverage, so there are no holes.
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
    )

    emissions_attrs = get_data_var_attrs(flux_template_path, inv_out.species)
    country_outs = make_country_outputs(
        inv_out,
        country_file=country_file,
        country_regions="paris",
        stats=stats,
        stats_args=stats_args,
        country_code="alpha3",
    )
    country_outs = country_outs * 1e-3  # g/yr -> kg/yr

    countries = Countries.from_file(
        country_file=country_file, country_code="alpha3", domain=inv_out.domain,
        
    )
    country_fraction = countries.matrix.as_numpy().rename("country_fraction")

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

    # ------------------------------------------------------------------ #
    # Inner domain (fine grid, e.g. 6 km) — separate output              #
    # ------------------------------------------------------------------ #
    if inv_out.inner_basis is None or inv_out.inner_flux is None:
        return result.as_numpy(), None

    n_inner_nx = len(inv_out.inner_basis.nx)  # recompute here to be safe
    full_trace_ds = inv_out.get_trace_dataset(var_names="x")

    inner_trace_ds = (
        full_trace_ds
        .isel(nx=slice(0, n_inner_nx))
        .assign_coords(nx=inv_out.inner_basis.nx.values)
    )

    # Verify alignment before proceeding
    assert inner_trace_ds.sizes["nx"] == n_inner_nx, (
        f"Inner trace nx size {inner_trace_ds.sizes['nx']} != n_inner_nx {n_inner_nx}"
    )
    assert np.array_equal(inner_trace_ds.nx.values, inv_out.inner_basis.nx.values), (
        f"nx mismatch: trace={inner_trace_ds.nx.values[:5]}... basis={inv_out.inner_basis.nx.values[:5]}..."
    )

    inner_stats_args = {**stats_args, "stats": stats, "chunk_dim": "nx"}
    inner_stats_ds = calculate_stats(inner_trace_ds, **inner_stats_args)

    # Reconstruct flux at native 6 km resolution: sum_k( x[k] * basis[k] * flux )
    # Result has dims (flux_time, lat_inner, lon_inner, quantile) — full spatial detail preserved
    inner_flux_stats = sparse_xr_dot(inv_out.inner_flux * inv_out.inner_basis, inner_stats_ds)
    inner_flux_stats = rename_by_replacement(inner_flux_stats, "x", "flux")

    # --- 2. Build coverage mask (where inner basis has non-zero coverage) ---
    # inner_basis.sum("nx") > 0 is sparse-backed; densify before boolean ops
    inner_basis_sum = inv_out.inner_basis.sum("nx")
    inner_basis_sum_data = _densify(inner_basis_sum)
    inner_coverage_mask_np = inner_basis_sum_data > 0  # (lat, lon) or (flux_time, lat, lon)

    # --- 3. Interpolate outer flux_outs to the inner fine grid ---
    # flux_outs is on the EUROPE grid; interp to inner lat/lon so we can
    # fill gaps in the inner domain with outer values (no holes in output).
    outer_on_inner = flux_outs.interp(
        lat=inv_out.inner_flux.lat,
        lon=inv_out.inner_flux.lon,
        method="linear",  # linear is fine for gap-filling; inner domain overrides where covered
    ).fillna(0.0)

    # --- 4. Stitch: use inner flux where covered, outer interpolated elsewhere ---
    stitched_pieces = {}
    for dv in inner_flux_stats.data_vars:
        inner_da = inner_flux_stats[dv]
        inner_np = _densify(inner_da)  # shape matches inner_da.dims exactly

        # Build outer fallback aligned to inner_da.dims by name
        dv_str = str(dv)
        if dv_str in outer_on_inner.data_vars:
            outer_da = outer_on_inner[dv_str]
            # Transpose outer_da to match inner_da.dims, inserting missing dims as size-1
            outer_dims = list(outer_da.dims)
            inner_dims = list(inner_da.dims)

            # Add any dims present in inner but absent in outer as size-1 axes
            for dim in inner_dims:
                if dim not in outer_dims:
                    outer_da = outer_da.expand_dims({dim: 1})
                    outer_dims = list(outer_da.dims)

            # Reorder to match inner_da.dims exactly
            outer_da = outer_da.transpose(*inner_dims)
            outer_np = np.broadcast_to(_densify(outer_da), inner_np.shape).copy()
        else:
            outer_np = np.zeros_like(inner_np)

        # Build mask aligned to inner_da.dims by name
        # inner_coverage_mask_np has dims (lat, lon) or (flux_time, lat, lon)
        if inner_coverage_mask_np.ndim == 2:
            mask_dims = ["lat", "lon"]
        else:
            mask_dims = ["flux_time", "lat", "lon"]

        inner_dims = list(inner_da.dims)
        mask = inner_coverage_mask_np
        for i, dim in enumerate(inner_dims):
            if dim not in mask_dims:
                mask = np.expand_dims(mask, axis=i)

        mask = np.broadcast_to(mask, inner_np.shape)

        stitched_np = np.where(mask, inner_np, outer_np)
        stitched_pieces[dv_str] = xr.DataArray(
            stitched_np,
            dims=inner_da.dims,
            coords=inner_da.coords,
            attrs=inner_da.attrs,
        )

    stitched_flux = xr.Dataset(stitched_pieces).fillna(0.0)
    # Diagnostic: check how much of the domain is covered by inner basis
    inner_coverage_fraction = float(inner_coverage_mask_np.mean())
    inner_nonzero = int(inner_coverage_mask_np.sum())
    total_cells = int(inner_coverage_mask_np.size)
    print(
        f"DEBUGOUT inner domain coverage: {inner_nonzero}/{total_cells} cells "
        f"({100*inner_coverage_fraction:.1f}%) covered by inner basis",
        flush=True,
    )

    # Check spatial detail in stitched_flux
    for dv in list(stitched_flux.data_vars)[:2]:
        da = stitched_flux[dv]
        inner_np = _densify(da)

        # Build mask with axes matching da.dims by name (same logic as stitch block)
        if inner_coverage_mask_np.ndim == 2:
            mask_dims = ["lat", "lon"]
        else:
            mask_dims = ["flux_time", "lat", "lon"]

        mask = inner_coverage_mask_np
        for i, dim in enumerate(list(da.dims)):
            if dim not in mask_dims:
                mask = np.expand_dims(mask, axis=i)

        mask_broadcast = np.broadcast_to(mask, inner_np.shape)

        inner_vals = inner_np[mask_broadcast]
        outer_vals = inner_np[~mask_broadcast]
        print(
            f"DEBUGOUT stitched {dv}: "
            f"inner region mean={np.nanmean(inner_vals):.3e} std={np.nanstd(inner_vals):.3e} | "
            f"outer region mean={np.nanmean(outer_vals):.3e} std={np.nanstd(outer_vals):.3e}",
            flush=True,
        )
    # --- 5. Country totals at fine resolution using fine-grid country mask ---
    inner_country_outs = make_inner_domain_country_outputs(
            stitched_flux, countries, inv_out.species
        )
    print(f"DEBUGOUT countries.matrix country dim: {list(countries.matrix.country.values)}", flush=True)

        # --- 6. Inversion-grid flux for inner domain ---
        # One uniform value per basis region painted back onto the fine grid.
        # Mirrors the outer inversion_grid output.
    inner_agg_flux = (
            (inv_out.inner_basis * inv_out.inner_flux).sum(["lat", "lon"])
            / inv_out.inner_basis.sum(["lat", "lon"])
        ).fillna(0.0)

    inner_inversion_grid_flux_raw = sparse_xr_dot(
            inv_out.inner_basis,
            inner_agg_flux * inner_stats_ds,
        )
    inner_inversion_grid_flux_raw = rename_by_replacement(
            inner_inversion_grid_flux_raw, "x", "flux"
        )

        # --- 7. Apply PARIS naming convention ---
        # stitched_flux vars are already named e.g. "flux_posterior_mean", "flux_prior_mean"
        # after rename_by_replacement(..., "x", "flux") in step 1.
        # inner_country_outs vars come from make_inner_domain_country_outputs which calls
        # rename_by_replacement(..., "flux", "country"), giving "country_posterior_mean" etc.
        # inner_inversion_grid_flux_raw vars are "flux_posterior_mean" etc. (same as stitched_flux)
        #
        # We apply renamer() ONCE to each set, using the raw names as input.
        # renamer() must NOT be applied to already-renamed names.

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
        for dv in stitched_flux.data_vars
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
        stitched_flux
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
    inner_result = xr.merge(
        [inner_flux_out, inner_country_out, inner_inversion_grid_out, inner_country_fraction],
        join="outer",
    )
    inner_result.attrs = make_global_attrs("flux")
    inner_result.attrs["inner_domain"] = "true"
    inner_result.attrs["spatial_resolution"] = "6km (native inner domain)"

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
) -> tuple[xr.Dataset, xr.Dataset | None, xr.Dataset]:
    """Create all PARIS outputs.

    Returns:
        Tuple of (flux_outs, inner_flux_outs, conc_outs).
        inner_flux_outs is None when no inner domain is present.
        When present, inner_flux_outs is a separate dataset at the native
        fine-grid resolution (e.g. 6 km), suitable for saving as a separate
        netCDF file with '_inner_domain' in the filename.
    """
    def _pick_mean(ds: xr.Dataset, candidates: list[str]) -> tuple[str | None, float | None]:
        for var in candidates:
            if var in ds.data_vars:
                return var, float(ds[var].mean().values)
        return None, None

    def _find_coord_label(coord: xr.DataArray, candidates: list[str]) -> str | None:
        coord_vals = [str(v) for v in coord.values]
        for candidate in candidates:
            if candidate in coord_vals:
                return candidate
        return None

    def _format_time_values(time_coord: xr.DataArray) -> str:
        timestamps = [
            str((pd.Timestamp("1970-01-01") + pd.to_timedelta(float(t), unit="D")).date())
            for t in np.asarray(time_coord.values)
        ]
        return "[" + ", ".join(timestamps) + "]"

    def _format_values(da: xr.DataArray) -> str:
        return "[" + ", ".join(f"{float(v):.3f}" for v in np.asarray(da.values)) + "]"

    flux_frequency = infer_flux_frequency(inv_out.flux)
    conc_outs = paris_concentration_outputs(inv_out, report_mode=report_mode, obs_avg_period=obs_avg_period)
    flux_outs, inner_flux_outs = paris_flux_output(
        inv_out,
        report_mode=report_mode,
        country_file=country_file,
        inversion_grid=inversion_grid,
        time_point=time_point,
        flux_frequency=flux_frequency,
    )

    # Consistency diagnostics
    conc_prior_var, conc_prior_mean = _pick_mean(
        conc_outs, ["Yapriori", "qYapriori", "Yapriori_modeled", "qYapriori_modeled"]
    )
    conc_post_var, conc_post_mean = _pick_mean(
        conc_outs, ["Yapost", "qYapost", "Yapost_modeled", "qYapost_modeled"]
    )
    flux_prior_var, flux_prior_mean = _pick_mean(
        flux_outs, ["flux_total_prior", "flux_total_apriori", "percentile_flux_total_prior", "percentile_flux_total_apriori"]
    )
    flux_post_var, flux_post_mean = _pick_mean(
        flux_outs, ["flux_total_posterior", "flux_total_apost", "percentile_flux_total_posterior", "percentile_flux_total_apost"]
    )

    if None not in [conc_prior_mean, conc_post_mean, flux_prior_mean, flux_post_mean]:
        conc_delta = conc_post_mean - conc_prior_mean
        flux_delta = flux_post_mean - flux_prior_mean
        conc_ratio = np.nan if abs(conc_prior_mean) < 1e-30 else conc_post_mean / conc_prior_mean
        flux_ratio = np.nan if abs(flux_prior_mean) < 1e-30 else flux_post_mean / flux_prior_mean
        print(
            "DEBUGOUT: PARIS consistency | "
            f"conc({conc_post_var}-{conc_prior_var})={conc_delta:.6e}, ratio={conc_ratio:.6f} | "
            f"flux({flux_post_var}-{flux_prior_var})={flux_delta:.6e}, ratio={flux_ratio:.6f}",
            flush=True,
        )
    else:
        print(
            "DEBUGOUT: PARIS consistency | unable to compute side-by-side prior/posterior check "
            f"(conc vars: {conc_prior_var}, {conc_post_var}; flux vars: {flux_prior_var}, {flux_post_var})",
            flush=True,
        )

    flux_region_candidates = {
        "NW EUROPE": ["NW_EU", "NW_EU2"],
        "GERMANY": ["DEU", "GERMANY"],
        "UK": ["GBR", "UK"],
        "BENELUX": ["BENELUX", "BELUX"],
        "FRANCE": ["FRA", "FRANCE"],
        "IRELAND": ["IRL", "IRELAND"],
    }
    flux_prior_name = next((v for v in ["country_flux_total_prior", "country_flux_total_apriori"] if v in flux_outs), None)
    flux_post_name = next((v for v in ["country_flux_total_posterior", "country_flux_total_apost"] if v in flux_outs), None)

    if flux_prior_name and flux_post_name and "country" in flux_outs.coords and "time" in flux_outs.coords:
        flux_time_str = _format_time_values(flux_outs.time)
        for display_name, candidates in flux_region_candidates.items():
            region_label = _find_coord_label(flux_outs.country, candidates)
            if region_label is None:
                continue
            prior_series = flux_outs[flux_prior_name].sel(country=region_label).compute()
            post_series = flux_outs[flux_post_name].sel(country=region_label).compute()
            print(
                f"DEBUGOUT: PARIS flux values {display_name} ({region_label}) | time={flux_time_str} | "
                f"prior={_format_values(prior_series)} | posterior={_format_values(post_series)}",
                flush=True,
            )

    conc_prior_name = next((v for v in ["Yapriori", "qYapriori"] if v in conc_outs), None)
    conc_post_name = next((v for v in ["Yapost", "qYapost"] if v in conc_outs), None)
    site_coord_name = (
        "sitenames" if "sitenames" in conc_outs.coords
        else ("nsite" if "nsite" in conc_outs.coords else None)
    )
    if conc_prior_name and conc_post_name and site_coord_name is not None and "time" in conc_outs.coords:
        mhd_label = _find_coord_label(conc_outs[site_coord_name], ["MHD"])
        if mhd_label is not None:
            conc_time_str = _format_time_values(conc_outs.time)
            prior_series = conc_outs[conc_prior_name].sel({site_coord_name: mhd_label}).compute()
            post_series = conc_outs[conc_post_name].sel({site_coord_name: mhd_label}).compute()
            print(
                f"DEBUGOUT: PARIS conc values MHD | time={conc_time_str} | "
                f"prior={_format_values(prior_series)} | posterior={_format_values(post_series)}",
                flush=True,
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