from functools import wraps
from typing import Callable, Optional, Sequence

import arviz as az
import numpy as np
import scipy
import xarray as xr


# this dictionary will be populated by using the decorator `register_stat`
stats_functions = {}


def register_stat(stat: Callable) -> Callable:
    """Decorator function to register stats functions.

    Args:
        stat: stats function to register

    Returns:
        stat, the input function (no modifications made)
    """
    stats_functions[stat.__name__] = stat
    return stat


def add_suffix(suffix: str):
    """Decorator to add suffix to variable names of dataset returned by decorated function.

    For example:

    @add_suffix("abc")
    def some_func():
       ...

    will add "_abc" to the end of all the data variables in the dataset returned by
    `some_func`. (So this only works is the output of `some_func` is an xr.Dataset or xr.DataArray.)

    Note: technically, `add_suffix` creates a new decorator
    each time it is called. This is the `decorate` function
    that is returned. Then the actual "decoration" is done by
    the `decorate` function.

    So

    @add_suffix("mean")
    def calc_mean():
        pass

    is the same as

    temp = add_suffix("mean")  # get `decorate`

    @temp
    def calc_mean():
        pass
    """
    def decorate(func):
        @wraps(func)
        def call(*args, **kwargs):
            result = func(*args, **kwargs)
            rename_dict = {dv: str(dv) + "_" + suffix for dv in result.data_vars}
            return result.rename(rename_dict)
        return call
    return decorate


@register_stat
@add_suffix("quantile")
def quantiles(ds: xr.Dataset, q: Sequence[float] = [0.159, 0.841], sample_dim: str = "draw"):
    return ds.quantile(q=q, dim=sample_dim)


@register_stat
@add_suffix("mode")
def mode(data, sample_dim="draw", thin: int = 1):
    """Approximate the mode by the midpoint of the shorted interval containing k samples.

    The slowest step is sorting. Still, this is over 30x faster than computing the KDE.
    (Unless you parallelise the KDE version by chunking the input.)

    Thinning by some integer factor will produce a corresponding speed up. For instance,
    if `thin = 2` is passed, then the running time will be roughly half.
    """
    def mode_of_arr(arr, k):
        arr = np.sort(arr, axis=-1)
        id_med = np.argmin(arr[..., k:] - arr[..., :-k], axis=-1, keepdims=True)
        mid = (np.take_along_axis(arr, id_med, axis=-1) + np.take_along_axis(arr, id_med + k, axis=-1)) / 2
        return mid.squeeze(axis=-1)

    if thin > 1:
        data = data.isel({sample_dim: slice(None, None, int(thin))})
        k = int((data.sizes[sample_dim] // thin) ** 0.8)  # k = (# draws)^{4/5}
    else:
        k = int(data.sizes[sample_dim] ** 0.8)  # k = (# draws)^{4/5}

    return xr.apply_ufunc(mode_of_arr, data, input_core_dims=[[sample_dim]], kwargs={"k": k})


@register_stat
@add_suffix("mode")
def mode_kde(da: xr.DataArray, sample_dim="draw") -> xr.DataArray:
    """Calculate the (KDE smoothed) mode of a data array containing MCMC
    samples.

    This can be parallelized if you chunk the DataArray first, e.g.

    da_chunked = da.chunk({"basis_region": 10})
    """

    def mode_of_row(row):
        if np.all(np.isnan(row)):
            return np.nan

        if np.nanmax(row) > np.nanmin(row):
            xvals = np.linspace(np.nanmin(row), np.nanmax(row), 200)
            kde = scipy.stats.gaussian_kde(row).evaluate(xvals)
            return xvals[kde.argmax()]

        return np.nanmean(row)

    def func(arr):
        return np.apply_along_axis(func1d=mode_of_row, axis=-1, arr=arr)

    return xr.apply_ufunc(func, da, input_core_dims=[[sample_dim]], dask="parallelized")


@register_stat
def hdi(data, hdi_prob=0.68, sample_dim="draw"):
    # wrap here to get hdi prob in suffix
    @add_suffix(f"hdi_{int(100 * hdi_prob)}")
    def calc(data, hdi_prob):
        return az.hdi(data, hdi_prob=hdi_prob)

    if not "chain" in data.dims:
        data = data.expand_dims({"chain": [0]})

    if sample_dim != "draw":
        data = data.rename({sample_dim: "draw"})

    return calc(data, hdi_prob)


@register_stat
@add_suffix("stdev")
def stdev(data, sample_dim="draw"):
    return data.std(dim=sample_dim)


@register_stat
@add_suffix("mean")
def mean(data, sample_dim="draw"):
    return data.mean(dim=sample_dim)


@register_stat
@add_suffix("median")
def median(data, sample_dim="draw"):
    return data.median(dim=sample_dim)


def calculate_stats(
    ds: xr.Dataset,
    name: str,
    chunk_dim: str,
    chunk_size: int = 10,
    var_names: Optional[list[str]] = None,
    report_mode: bool = False,
    add_bc_suffix: bool = False,
) -> list[xr.Dataset]:
    output = []
    if var_names is None:
        var_names = list(ds.data_vars)
    for var_name in var_names:
        suffix = "apost" if "posterior" in var_name else "apriori"
        if add_bc_suffix:
            suffix += "BC"

        if report_mode:
            stats = [
                calc_mode_kde(ds[var_name].dropna(dim="draw").chunk({chunk_dim: chunk_size}), sample_dim="draw")
                .compute()
                .rename(f"{name}{suffix}"),
            ]
        else:
            stats = [
                ds[var_name].mean("draw").rename(f"{name}{suffix}"),
                # calc_mode(ds[var_name].dropna(dim="draw").chunk({chunk_dim: chunk_size}), sample_dim="draw")
                # .compute()
                # .rename(f"{name}{suffix}_mode"),
            ]
        stats.append(
            make_quantiles(ds[var_name].dropna(dim="draw"), sample_dim="draw").rename(f"q{name}{suffix}")
        )

        output.extend(stats)
    return output
