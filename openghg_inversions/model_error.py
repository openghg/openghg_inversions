"""Functions for computing estimates of model error.
"""


import numpy as np
import xarray as xr


def residual_error_method(
    ds_dict: dict[str, xr.Dataset],
    robust: bool = False,
    by_site: bool = False,
) -> np.ndarray:
    """Compute estimate of model error using residual error method.

    This method is explained in "Modeling of Atmospheric Chemistry" by Brasseur
    and Jacobs in Box 11.2 on p.499-500, following "Comparative inverse analysis of satellitle (MOPITT)
    and aircraft (TRACE-P) observations to estimate Asian sources of carbon monoxide", by Heald, Jacob,
    Jones, et.al. (Journal of Geophysical Research, vol. 109, 2004).

    Roughly, we assume that the observations y are equal to the modelled observations y_mod (mf_mod + bc_mod),
    plus a bias term b, and instrument, representation, and model error:

    y = y_mod + b + err_I + err_R + err_M

    Assuming the errors are mean zero, we have

    (y - y_mod) - mean(y - y_mod) = err_I + err_R + err_M  (*)

    where the mean is taken over all observations.

    Calculating the RMS of the LHS of (*) gives us an estimate for

    sqrt(sigma_I^2 + sigma_R^2 +  sigma_M^2),

    where sigma_I is the standard deviation of err_I, and so on.

    Thus a rough estimate for sigma_M is the RMS of the LHS of (*), possibly with the RMS of
    the instrument/observation and averaging errors removed (this isn't implemented here).

    Note: in the "non-robust" case, we are computing the standard deviation of y - y_mod. The mean on the LHS
    of equation (*) could be taken over a subset of the observation, in which case the value calculated is not
    a standard deviation. We wrote the derivation this way to match Brasseur and Jacobs.

    Args:
        ds_dict: dictionary of combined scenario datasets, keyed by site codes.
        robust: if True, use the "median absolute deviation" (https://en.wikipedia.org/wiki/Median_absolute_deviation)
            instead of the standard deviation. MAD is a measure of spread, similar to standard deviation, but
            is more robust to outliers.
        by_site: if True, return array with one mininum error value per site

    Returns:
        np.ndarray: estimated value(s) for model error.
    """
    # if "bc_mod" is present, we need to add it to "mf_mod"
    if all("bc_mod" in v for k, v in ds_dict.items() if not k.startswith(".")):
        ds = xr.concat(
            [
                v[["mf", "bc_mod", "mf_mod"]].expand_dims({"site": [k]})
                for k, v in ds_dict.items()
                if not k.startswith(".")
            ],
            dim="site",
        )

        scaling_factor = float(ds.mf.units) / float(ds.bc_mod.units)
        ds["modelled_obs"] = ds.mf_mod + ds.bc_mod / scaling_factor
    else:
        ds = xr.concat(
            [
                v[["mf", "mf_mod"]].expand_dims({"site": [k]})
                for k, v in ds_dict.items()
                if not k.startswith(".")
            ],
            dim="site",
        )
        ds["modelled_obs"] = ds.mf_mod

    if robust is True:
        # call `.as_numpy` because dask arrays throw an error when we try to compute a median
        if by_site is True:
            avg = (ds.mf - ds.modelled_obs).as_numpy().groupby("site").median(dim="time")
            res_err = np.abs(ds.mf - ds.modelled_obs - avg).as_numpy().groupby("site").median(dim="time")
        else:
            avg = (ds.mf - ds.modelled_obs).as_numpy().median(dim=["time", "site"])
            res_err = np.abs(ds.mf - ds.modelled_obs - avg).as_numpy().median(dim=["site", "time"])
    elif by_site is True:
        avg = (ds.mf - ds.modelled_obs).groupby("site").mean(dim="time")
        res_err = np.sqrt(((ds.mf - ds.modelled_obs - avg) ** 2).groupby("site").mean("time"))
    else:
        avg = (ds.mf - ds.modelled_obs).mean()
        res_err = np.sqrt(((ds.mf - ds.modelled_obs - avg) ** 2).mean())

    return res_err.values


def percentile_error_method(ds_dict: dict[str, xr.Dataset]) -> np.ndarray:
    """Compute estimate of minimum model error using percentile error method

    This is a simple method to estimate the minimum model error (i.e. the model error used at baseline
    points). For each site. it takes the monthly median measured mf and subtracts the monthly 5th
    percentile measured mf, then calculates the annual mean of these monthly values. The thinking behind
    this is that transport error might result in modelled enhancements at the baseline points, even with
    an accurate flux map. So this provides a rough calculation for the likely impact of such an event.

    Args:
        ds_dict: dictionary of combined scenario datasets, keyed by site codes.

    Returns:
        np.ndarray: estimated value(s) for model error.
    """
    # Combine mf data from each site into a single dataset with a site dimension
    ds = xr.concat(
        [v[["mf"]].expand_dims({"site": [k]}) for k, v in ds_dict.items() if not k.startswith(".")],
        dim="site",
    )

    # Calculate monthly percentiles, then take the annual mean difference for each site
    monthly_50pc = ds.mf.as_numpy().resample(time="MS").quantile(0.5)
    monthly_5pc = ds.mf.as_numpy().resample(time="MS").quantile(0.05)
    res_err = (monthly_50pc - monthly_5pc).groupby("site").mean(dim="time")

    return res_err.values


def setup_min_error(min_error: np.ndarray, siteindicator: np.ndarray) -> np.ndarray:
    """Given min_error vector with same length as number of sites, create a vector
    aligned with obs stacked by site.
    """
    # need the same number of min_error values as distinct values in siteindicator
    assert np.max(siteindicator) == len(min_error) - 1

    return min_error[siteindicator.astype(int)]
