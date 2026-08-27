"""Normalize, calculate, and align minimum model-error values."""

import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr


@dataclass(frozen=True, eq=False, slots=True)
class MinimumError:
    """Observation-aligned minimum model error.

    Attributes:
        values: Minimum error labelled by observation.
        method: Source form used to obtain the values.
        by_site: Whether the source values varied by site.
        sites: Site order used for per-site alignment.
    """

    values: xr.DataArray
    method: str
    by_site: bool
    sites: tuple[str, ...]

    @classmethod
    def prepare(
        cls,
        observations: xr.Dataset,
        fp_data: Mapping[str, xr.Dataset],
        value: str | Mapping[str, float] | int | float = 0.0,
        *,
        by_site: bool = True,
    ) -> "MinimumError":
        """Prepare minimum error from a scalar, site mapping, or named method.

        Args:
            observations: Gathered observations containing ``mf`` and a labelled
                ``site`` coordinate for per-site values.
            fp_data: Per-site scientific datasets used by calculated methods.
            value: Scalar, per-site mapping, ``"residual"``, or ``"percentile"``.
            by_site: Whether residual values are calculated separately by site.

        Returns:
            A validated, observation-aligned minimum error value.

        Raises:
            ValueError: If the configuration, site labels, or resulting values
                are invalid.
        """
        if "mf" not in observations:
            raise ValueError("Minimum-error preparation requires an 'mf' observation array.")
        if not isinstance(by_site, bool):
            raise ValueError(f"Minimum-error 'by_site' must be a boolean, got {type(by_site).__name__}.")

        site_coord: xr.DataArray | None = None
        sites: tuple[str, ...] = ()
        method: str
        varies_by_site = False

        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            source = np.asarray(float(value))
            method = "scalar"
        elif isinstance(value, np.ndarray) and value.ndim == 0:
            source = value
            method = "scalar"
        elif isinstance(value, Mapping):
            site_coord = observations.coords.get("site")
            if site_coord is None:
                raise ValueError("Per-site min_error values require a labelled site coordinate.")
            sites = tuple(dict.fromkeys(str(site) for site in site_coord.values))
            missing = [site for site in sites if site not in value]
            if missing:
                raise ValueError(f"min_error mapping is missing values for site(s): {missing}")
            source = np.asarray([value[site] for site in sites], dtype=float)
            method = "per_site"
            varies_by_site = True
        elif value in ("residual", "percentile"):
            site_coord = observations.coords.get("site")
            if site_coord is None:
                raise ValueError("Calculated min_error values require a labelled site coordinate.")
            sites = tuple(dict.fromkeys(str(site) for site in site_coord.values))
            selected = {site: fp_data[site] for site in sites if site in fp_data}
            missing_data = [site for site in sites if site not in selected]
            if missing_data:
                raise ValueError(f"Minimum-error calculation is missing fp_data for site(s): {missing_data}")
            method = value
            varies_by_site = value == "percentile" or by_site
            source = (
                residual_error_method(selected, by_site=by_site)
                if value == "residual"
                else percentile_error_method(selected)
            )
        else:
            raise ValueError(f"Option '{value}' is not valid.")

        source = np.asarray(source, dtype=float)
        if not np.isfinite(source).all():
            raise ValueError("Minimum-error values contain NaN or infinite values.")
        if (source < 0).any():
            raise ValueError("Minimum-error values must be non-negative.")

        if varies_by_site:
            if site_coord is None:
                raise ValueError("Per-site min_error values require a labelled site coordinate.")
            if source.ndim > 1 or source.size != len(sites):
                raise ValueError(f"Minimum error has {source.size} site values; expected {len(sites)}.")
            per_site = xr.DataArray(
                source.reshape(-1),
                coords={"site": np.asarray(sites)},
                dims="site",
            )
            data = per_site.sel(site=site_coord)
        else:
            if source.size != 1:
                raise ValueError("A non-site minimum error must contain exactly one value.")
            data = xr.full_like(observations.mf, float(source.reshape(-1)[0]), dtype=float)

        data = data.rename("min_error")
        data.attrs = {
            "units": observations.mf.attrs.get("units", ""),
            "minimum_error_method": method,
            "minimum_error_by_site": int(varies_by_site),
            "minimum_error_sites": ",".join(sites),
        }
        return cls(data, method, varies_by_site, sites)


def normalise_min_error_options(options: Mapping[str, Any] | None) -> dict[str, bool]:
    """Validate options supported by calculated minimum-error methods.

    Args:
        options: Optional minimum-error configuration mapping.

    Returns:
        A normalized mapping containing a boolean ``by_site`` value.

    Raises:
        ValueError: If the value is not a mapping, contains unsupported keys,
            or supplies a non-boolean ``by_site`` value.
    """
    if options is None:
        return {"by_site": False}
    if not isinstance(options, Mapping):
        raise ValueError(f"`min_error_options` must be a mapping/dict or None, got {type(options).__name__}.")

    unsupported = sorted(str(key) for key in options if key != "by_site")
    if unsupported:
        raise ValueError(
            "`min_error_options` contains unsupported option(s): "
            f"{unsupported!r}. The only supported option is `by_site`."
        )

    by_site = options.get("by_site", False)
    if not isinstance(by_site, bool):
        raise ValueError(f"`min_error_options['by_site']` must be a boolean, got {type(by_site).__name__}.")
    return {"by_site": by_site}


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
    """Compute estimate of minimum model error using percentile error method.

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
    result = []
    for site, dataset in ds_dict.items():
        if site.startswith("."):
            continue
        mf = dataset.mf.as_numpy()
        monthly_50pc = mf.resample(time="MS").quantile(0.5)
        monthly_5pc = mf.resample(time="MS").quantile(0.05)
        result.append((monthly_50pc - monthly_5pc).mean().item())

    return np.asarray(result)
