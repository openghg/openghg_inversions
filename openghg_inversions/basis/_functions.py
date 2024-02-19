from pathlib import Path
from typing import cast, Optional

import numpy as np
import xarray as xr


def _flux_fp_from_fp_all(
    fp_all: dict, emissions_name: Optional[list[str]] = None
) -> tuple[xr.DataArray, list[xr.DataArray]]:
    """Get flux and list of footprints from `fp_all` dictionary and optional list of emissions names."""
    if emissions_name is not None:
        flux = fp_all[".flux"][emissions_name[0]].data.flux
    else:
        first_flux = next(iter(fp_all[".flux"].values()))
        flux = first_flux.data.flux

    flux = cast(xr.DataArray, flux)

    footprints: list[xr.DataArray] = [v.fp for k, v in fp_all.items() if not k.startswith(".")]

    return flux, footprints


def _mean_fp_times_mean_flux(
    flux: xr.DataArray,
    footprints: list[xr.DataArray],
    abs_flux: bool = False,
    mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Multiply mean flux by mean of footprints, optionally restricted to a Boolean mask."""
    if abs_flux is True:
        print("Using absolute value of flux array.")
        flux = abs(flux)

    mean_flux = flux.mean("time")

    fp_total = sum(footprints)  # this seems to be faster than concatentating and summing over new axis
    n_measure = sum(len(fp.time) for fp in footprints)

    fp_total = cast(xr.DataArray, fp_total)  # otherwise mypy complains about the next line
    mean_fp = fp_total.sum("time") / n_measure

    if mask is not None:
        # align to footprint lat/lon
        mean_fp, mean_flux, mask = xr.align(mean_fp, mean_flux, mask, join="override")
        return (mean_fp * mean_flux).where(mask, drop=True)

    mean_fp, mean_flux = xr.align(mean_fp, mean_flux, join="override")
    return mean_fp * mean_flux


def _save_basis(
    basis: xr.Dataset,
    basis_algorithm: str,
    output_dir: str,
    domain: str,
    species: str,
    output_name: Optional[str] = None,
) -> None:
    """Save basis functions to netCDF.

    Args:
        basis: basis dataset to save
        basis_algorithm: name of basis algorithm (e.g. "quadtree" or "weighted")
        output_dir: root directory to save basis functions
        domain: domain of inversion; basis is saved in a "domain" directory inside `output_dir`
        species: species of inversion
        output_name
    """
    basis_out_path = Path(output_dir, domain.upper())

    if not basis_out_path.exists():
        basis_out_path.mkdir(parents=True)

    start_date = str(basis.time.min().values)[:7]  # year and month

    if output_name is None:
        output_name = f"{basis_algorithm}_{species}_{domain}_{start_date}.nc"
    else:
        output_name = f"{basis_algorithm}_{species}-{output_name}_{domain}_{start_date}.nc"

    basis.to_netcdf(basis_out_path / output_name, mode="w")
