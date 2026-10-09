"""Prepare reported observation errors before temporal filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING
import logging

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from openghg_inversions.inversion_data.acquisition import RhimeMergedData

logger = logging.getLogger(__name__)


def prepare_observation_errors(
    merged: RhimeMergedData, *, averaging_error: bool = True,
) -> RhimeMergedData:
    """Derive missing ``mf_error`` on owned datasets before temporal filtering.

    Combine repeatability and variability in quadrature when ``averaging_error``
    is true. Otherwise use repeatability, falling back to variability if absent.
    Missing components become zero; both absent is an error when ``mf_error``
    is absent. The historical
    zero-error replacement uses the larger of the nonzero error median and the
    concentration standard deviation over the complete acquired population.
    NaN errors keep the existing missing-value policy.

    Args:
        merged: Borrowed acquired data before filtering
            or temporal aggregation. Existing custom ``mf_error`` is retained
            unchanged, including its labels and metadata. Missing diagnostic
            components are supplied as zero arrays for gathered assembly.
        averaging_error: Whether variability contributes to derived errors.

    Returns:
        A merged handoff. Datasets needing derived errors or missing diagnostic
        components use shallow owned containers. All numerical inputs remain
        borrowed; complete existing-error datasets are returned unchanged.
        Error diagnostics and zero replacement may compute derived arrays.

    Raises:
        ValueError: If a site lacking ``mf_error`` has neither uncertainty component.
    """
    # TODO: do we want to fill missing values in repeatability or variability?
    result = {}
    for site, borrowed in merged.site_data.items():
        if "mf_error" in borrowed:
            missing_components = [name for name in ("mf_repeatability", "mf_variability") if name not in borrowed]
            if missing_components:
                ds = borrowed.copy(deep=False)
                for name in missing_components:
                    ds[name] = xr.zeros_like(borrowed["mf_error"])
                    ds[name].attrs = {"long_name": name, "units": borrowed.mf_error.attrs.get("units")}
                result[site] = ds
            else:
                result[site] = borrowed
            continue
        ds = borrowed.copy(deep=False)
        result[site] = ds
        mf_long_name = ds.mf.attrs.get("long_name", "")
        mf_units = ds.mf.attrs.get("units", None)

        variability_missing = False
        if "mf_variability" not in ds:
            ds["mf_variability"] = xr.zeros_like(ds.mf)
            variability_missing = True
        ds["mf_variability"].attrs["long_name"] = mf_long_name + "_variability"
        ds["mf_variability"].attrs["units"] = mf_units

        if "mf_repeatability" not in ds:
            if variability_missing:
                raise ValueError(f"Obs data for site {site} is missing both repeatability and variability.")

            ds["mf_repeatability"] = xr.zeros_like(ds.mf_variability)

            ds["mf_error"] = ds["mf_variability"]

            if averaging_error:
                logger.info(
                    "`mf_repeatability` not present; using `mf_variability` for `mf_error` at site %s", site
                )

        elif averaging_error:
            # Fill with zeros so that if one of repeatability and variability is not NaN, then mf_error will not be NaN.
            ds["mf_error"] = np.sqrt(
                ds["mf_repeatability"].fillna(0) ** 2 + ds["mf_variability"].fillna(0) ** 2
            )
        else:
            ds["mf_error"] = ds["mf_repeatability"]

        ds["mf_repeatability"].attrs["long_name"] = mf_long_name + "_repeatability"
        ds["mf_repeatability"].attrs["units"] = mf_units
        ds["mf_error"].attrs["long_name"] = mf_long_name + "_error"
        ds["mf_error"].attrs["units"] = mf_units

        # warnings/info for debugging
        err0 = (ds["mf_error"] == 0) | (
            ds["mf_error"].isnull()
        )  # might have NaN if averaging_error is False

        if err0.any():
            percent0 = 100 * err0.mean()
            logger.warning(
                (
                    "`mf_error` is zero/nan for %.2f percent of times at site %s;"
                    "filling with max(median(mf_error), std(mf))."
                ),
                percent0,
                site,
            )

            mf_err_da = ds["mf_error"].as_numpy()  # load into memory to avoid Dask issues
            fill_value = np.nanmax(
                [
                    mf_err_da.where(mf_err_da != 0).dropna(dim="time").median(),
                    ds["mf"].std(dim="time"),
                ]
            )
            ds["mf_error"] = mf_err_da.where(mf_err_da != 0, fill_value)
            info_msg = (
                "If `averaging_period` matches the frequency of the obs data, then `mf_variability` "
                "will be zero. Try setting `averaging_period = None`."
            )
            logger.info(info_msg)

    if all(result[site] is dataset for site, dataset in merged.site_data.items()):
        return merged
    return merged.with_site_data(result, context="Observation error preparation")
