"""Functions for computing statistics on datasets.

Statistics that call ``_to_dense_dataset`` need dense-compatible chunks for the
operation itself. Statistics that can operate on sparse-backed arrays should not
densify their inputs here; product code can still densify final outputs before
NetCDF serialisation.
"""

from collections import namedtuple
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TypeVar, cast

import arviz as az
import dask
import numpy as np
import scipy
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.postprocessing.utils import add_suffix, get_parameters, update_attrs

StatsFunction = namedtuple("StatsFunction", ["name", "func", "params"])
"""Tuple holding a stats function and the parameters it accepts.

Storing the parameters the stats function accepts helps with passing
optional arguments to the stats functions without having to specify
extra parameters for each stats function separately.
"""

# this dictionary will be populated by using the decorator `register_stat`
stats_functions: dict[str, StatsFunction] = {}


def register_stat(stat: Callable) -> Callable:
    """Decorator function to register stats functions.

    Args:
        stat: stats function to register

    Returns:
        stat, the input function (no modifications made)
    """
    stats_functions[stat.__name__] = StatsFunction(stat.__name__, stat, get_parameters(stat))
    return stat


def _to_dense_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Convert sparse-backed variables to dense chunks when a statistic cannot handle sparse arrays."""
    return xr.Dataset(
        {name: to_dense(data) for name, data in ds.data_vars.items()},
        coords=ds.coords,
        attrs=ds.attrs,
    )


_STATS_MAX_POINTS = 1024
_STATS_CHUNKS = {"lat": 32, "lon": 32, "latitude": 32, "longitude": 32, "flux_time": 1}


def _bounded_chunk_sizes(
    data: xr.Dataset | xr.DataArray, sample_dims: Iterable[str] = ("draw", "chain")
) -> dict[str, int]:
    """Limit non-sample blocks to 1024 points, with spatial sides at most 32.

    Apply these chunks before constructing expensive producers; slicing their
    already-built outputs cannot bound the producers' original task payloads.
    Sample dimensions retain their existing chunks and are never tiled.
    """
    remaining = _STATS_MAX_POINTS
    chunks = {}
    for dim, size in data.sizes.items():
        if dim not in sample_dims:
            chunks[dim] = max(1, min(size, _STATS_CHUNKS.get(dim, 128), remaining))
            remaining = max(1, remaining // chunks[dim])
    return chunks


_XarrayT = TypeVar("_XarrayT", xr.DataArray, xr.Dataset)


def _chunk_non_sample_dims(data: _XarrayT, sample_dims: Iterable[str] = ("draw", "chain")) -> _XarrayT:
    """Bound producer chunks while retaining compatible, already-bounded layouts."""
    chunks: dict = _bounded_chunk_sizes(data, sample_dims)
    variables = [data] if isinstance(data, xr.DataArray) else list(data.data_vars.values())
    for dim, limit in chunks.items():
        layouts = {value.variable.chunksizes[dim] for value in variables if value.chunks and dim in value.dims}
        if len(layouts) == 1:
            layout = next(iter(layouts))
            if max(layout) <= limit:
                chunks[dim] = layout
    return data.chunk(chunks)


def _consolidate_sample_dimension(ds: xr.Dataset, sample_dim: str) -> xr.Dataset:
    """Rechunk a Dask-backed sample dimension into a single core chunk."""
    if any(data.chunks is not None and sample_dim in data.dims for data in ds.data_vars.values()):
        return ds.chunk({sample_dim: -1})
    return ds


@register_stat
@add_suffix("quantile")
@update_attrs("quantile_of")
def quantiles(
    ds: xr.Dataset,
    quantiles: Sequence[float] = [0.159, 0.841],
    sample_dim: str = "draw",
) -> xr.Dataset:
    """Compute quantiles.

    Args:
        ds: input dataset; must have dimension specified by `sample_dim`
          (default is "draw")
        quantiles: sequence of quantiles to compute; default values correspond
          to mean +/- 1 stdev for a normally distributed sequence.
        sample_dim: dimension to compute quantiles over; defaults to "draw",
          which is the default sample dimension for PyMC outputs.

    Returns:
        xr.Dataset of specified quantiles, with a new `quantile` dimension.

    """
    ds = _consolidate_sample_dimension(_to_dense_dataset(ds), sample_dim)
    # cast to float32 since quantiles involve interpolation, and this always converts to
    # float64 in numpy and scipy interpolation routines
    return ds.quantile(q=quantiles, dim=sample_dim).astype("float32")


@register_stat
@add_suffix("mode")
@update_attrs("mode_of")
def mode(ds: xr.Dataset, sample_dim: str = "draw", thin: int = 1) -> xr.Dataset:
    """Approximate the mode using the shortest interval containing ``k`` samples.

    The slowest step is sorting. Still, this is over 30x faster than computing the KDE.
    (Unless you parallelise the KDE version by chunking the input.)

    Thinning by some integer factor will produce a corresponding speed up. For instance,
    if `thin = 2` is passed, then the running time will be roughly half.

    Args:
        ds: Dataset containing samples.
        sample_dim: Dimension containing the samples.
        thin: Integer stride used to thin samples before estimating the mode.

    Returns:
        Dataset containing the approximate mode for each input variable.
    """

    def mode_of_arr(arr, k):
        arr = np.sort(arr, axis=-1)
        id_med = np.argmin(arr[..., k:] - arr[..., :-k], axis=-1, keepdims=True)
        mid = (np.take_along_axis(arr, id_med, axis=-1) + np.take_along_axis(arr, id_med + k, axis=-1)) / 2
        return mid.squeeze(axis=-1)

    if thin > 1:
        ds = ds.isel({sample_dim: slice(None, None, int(thin))})
        k = int((ds.sizes[sample_dim] // thin) ** 0.8)  # k = (# draws)^{4/5}
    else:
        k = int(ds.sizes[sample_dim] ** 0.8)  # k = (# draws)^{4/5}

    ds = _consolidate_sample_dimension(_to_dense_dataset(ds), sample_dim)
    return xr.apply_ufunc(
        mode_of_arr,
        ds,
        input_core_dims=[[sample_dim]],
        kwargs={"k": k},
        dask="parallelized",
        output_dtypes=[float],
    )


@register_stat
@add_suffix("mode")
@update_attrs("mode_of")
def mode_kde(
    ds: xr.Dataset,
    sample_dim: str = "draw",
    chunk_dim: str | None = None,
    chunk_size: int = 10,
) -> xr.Dataset:
    """Calculate the (KDE smoothed) mode of a data array containing MCMC samples.

    This can be parallelized if you chunk the DataArray first, e.g.
    >>> da_chunked = da.chunk({"basis_region": 10})

    Args:
        ds: Dataset containing samples.
        sample_dim: Dimension containing the samples.
        chunk_dim: Optional non-sample dimension to chunk for parallel execution.
        chunk_size: Chunk size to use along ``chunk_dim``.

    Returns:
        Dataset containing the KDE-smoothed mode for each input variable.
    """

    def mode_of_row(row):
        row = row[np.isfinite(row)]
        if row.size == 0:
            return np.nan

        if row.size == 1:
            return float(row[0])

        if np.max(row) > np.min(row):
            xvals = np.linspace(np.min(row), np.max(row), 200)
            kde = scipy.stats.gaussian_kde(row).evaluate(xvals)
            return xvals[kde.argmax()]

        return float(np.mean(row))

    def func(arr):
        if arr.shape[-1] == 0:
            return np.full(arr.shape[:-1], np.nan, dtype=float)
        return np.apply_along_axis(func1d=mode_of_row, axis=-1, arr=arr)

    ds = _consolidate_sample_dimension(_to_dense_dataset(ds), sample_dim)
    if chunk_dim is not None and chunk_dim != sample_dim:
        return xr.apply_ufunc(
            func,
            ds.chunk({chunk_dim: chunk_size}),
            input_core_dims=[[sample_dim]],
            dask="parallelized",
            output_dtypes=[float],
        )
    return xr.apply_ufunc(
        func,
        ds,
        input_core_dims=[[sample_dim]],
        dask="parallelized",
        output_dtypes=[float],
    )


@register_stat
def hdi(
    ds: xr.Dataset,
    hdi_prob: float | Iterable[float] = 0.68,
    sample_dim: str = "draw",
) -> xr.Dataset:
    """Compute exact highest density intervals with bounded non-sample tasks.

    Dask inputs remain lazy; each kernel receives all chains and samples for
    its non-sample block. Large eager inputs are chunked without changing them.
    Existing upstream producers must already have bounded task payloads.

    Args:
        ds: Dataset containing samples.
        hdi_prob: Probability or probabilities contained by each interval.
        sample_dim: Dimension containing the samples.

    Returns:
        Dataset containing the highest density interval bounds.
    """
    chunks = _bounded_chunk_sizes(ds, (sample_dim, "chain"))
    if any(data.chunks is not None for data in ds.data_vars.values()) or any(
        ds.sizes[dim] > size for dim, size in chunks.items()
    ):
        ds = _chunk_non_sample_dims(ds, (sample_dim, "chain"))
    ds = _consolidate_sample_dimension(_to_dense_dataset(ds), sample_dim)
    # handle case of multiple hdi_probs
    if isinstance(hdi_prob, Iterable):
        return xr.merge([hdi(ds, hdi_prob=prob, sample_dim=sample_dim) for prob in hdi_prob])

    # wrap here to get hdi prob in suffix
    @add_suffix(f"hdi_{int(100 * hdi_prob)}")
    @update_attrs(f"hdi_{int(100 * hdi_prob)}_of")
    def calc(data: xr.Dataset, probability: float) -> xr.Dataset:
        """Call ArviZ and narrow its union return type for Dataset input."""
        return cast(
            xr.Dataset,
            az.hdi(
                data,
                hdi_prob=probability,
                dask_kwargs={
                    "dask": "parallelized",
                    "output_dtypes": [float],
                    "dask_gufunc_kwargs": {"output_sizes": {"hdi": 2}},
                },
            ),
        )

    if "chain" not in ds.dims:
        ds = ds.expand_dims({"chain": [0]})

    if sample_dim != "draw":
        ds = ds.rename({sample_dim: "draw"})

    ds = _consolidate_sample_dimension(ds, "chain")
    return calc(ds, hdi_prob)


@register_stat
@add_suffix("stdev")
@update_attrs("standard_deviation_of")
def stdev(ds: xr.Dataset, sample_dim="draw"):
    """Compute sample standard deviation."""
    return ds.std(dim=sample_dim)


@register_stat
@add_suffix("mean")
@update_attrs("mean_of")
def mean(ds: xr.Dataset, sample_dim="draw"):
    """Compute sample mean."""
    return ds.mean(dim=sample_dim)


@register_stat
@add_suffix("median")
@update_attrs("median_of")
def median(ds: xr.Dataset, sample_dim: str = "draw") -> xr.Dataset:
    """Compute the median along the sample dimension.

    Args:
        ds: Dataset containing samples.
        sample_dim: Dimension containing the samples.

    Returns:
        Dataset containing the median for each input variable.
    """
    ds = _consolidate_sample_dimension(_to_dense_dataset(ds), sample_dim)
    return ds.median(dim=sample_dim)


_BUILTIN_STAT_FUNCTIONS = frozenset((mean, stdev, median, quantiles, mode, mode_kde, hdi))


def calculate_stats(
    ds: xr.Dataset,
    stats: list[str] = ["mean", "quantiles"],
    *,
    _prepare_block: Callable[[xr.Dataset], xr.Dataset] | None = None,
    **kwargs,
) -> xr.Dataset:
    """Calculate statistics, completing one bounded non-sample block at a time.

    Related variables with the same non-sample dimensions and all requested
    statistics share a computation per block. Sample chunks retain their
    original reduction order; exact interval kernels receive complete samples.
    Results are eager (and may retain sparse storage). Borrowed inputs are not
    changed. Upstream producers must already use bounded non-sample chunks;
    this boundary limits workspace, not resident inputs or final output size.
    Compatible bounded source chunks are retained. Inconsistent chunk layouts
    across related variables may require a source chunk in adjacent blocks.
    Mixed eager/lazy groups may also split a source block beside a singleton
    edge to preserve the eager sample-reduction layout.
    Calls containing a custom registered kernel use the complete dataset so
    its defaults and cross-variable operands remain intact; that compatibility
    fallback does not provide the built-in non-sample workspace bound.

    Args:
        ds: dataset to calculate stats on.
        stats: list of stats to calculate.
        _prepare_block: Internal arithmetic applied after blocking and before
          statistic kernels. Related operands must have matching non-sample
          dimensions. Preserve their NumPy or Dask backend and borrowed inputs.
        **kwargs: arguments to pass to stats functions. If a parameter can be passed to a stats function,
          it will be passed. To pass to a specific stats function, use `<stats func name>__<key> = <value>`.
          Note: that is a double underscore. For instance `mode_kde__chunk_dim="country"` would specify `chunk_dim`
          only for the stats function "mode_kde".

    Returns:
        dataset containing all stats calculated on all variables in input dataset.

    Raises:
        ValueError: if a statistic in `stats` is not found in the registry.

    """
    if not stats:
        return xr.Dataset()

    # A stat-specific probability overrides the generic argument, including
    # when both refer to the same one-shot iterable.
    key = "hdi__hdi_prob" if "hdi__hdi_prob" in kwargs else "hdi_prob"
    if "hdi" in stats and isinstance(kwargs.get(key), Iterable):
        kwargs[key] = tuple(kwargs[key])

    for stat in stats:
        if stat not in stats_functions:
            raise ValueError(f"Statistic {stat} not available.")

    # Normalize only effective arguments; aliases of one iterator share its
    # tuple, and a completely overridden generic iterator remains untouched.
    reduction_iterators = {}
    for stat in stats:
        key = f"{stat}__sample_dim" if f"{stat}__sample_dim" in kwargs else "sample_dim"
        dimensions = kwargs.get(key)
        if isinstance(dimensions, Iterator) and (
            key != "sample_dim" or "sample_dim" in stats_functions[stat].params
        ):
            identity = id(dimensions)
            if identity not in reduction_iterators:
                reduction_iterators[identity] = tuple(dimensions)
            kwargs[key] = reduction_iterators[identity]

    if any(
        all(stats_functions[stat].func is not builtin for builtin in _BUILTIN_STAT_FUNCTIONS)
        for stat in stats
    ):
        with dask.config.set(scheduler="single-threaded"):
            data = _prepare_block(ds) if _prepare_block is not None else ds
            return _calculate_stats(data, stats, **kwargs).compute()

    sample_dims = {"chain"}
    for stat in stats:
        dimensions = kwargs.get(f"{stat}__sample_dim", kwargs.get("sample_dim", "draw"))
        if dimensions is None or dimensions is Ellipsis:
            sample_dims.update(ds.dims)
        elif isinstance(dimensions, str):
            sample_dims.add(dimensions)
        else:
            sample_dims.update(dimensions)
    groups: dict[tuple, list] = {}
    for name, data in ds.data_vars.items():
        dimensions = tuple(dim for dim in ds.dims if dim in data.dims and dim not in sample_dims)
        groups.setdefault(dimensions, []).append(name)

    def reduce_blocks(data: xr.Dataset, dimensions: tuple, chunks: dict, markers: list) -> xr.Dataset:
        if dimensions:
            dim, *remaining = dimensions
            if data.sizes[dim] > chunks[dim]:
                layouts = {value.variable.chunksizes[dim] for value in data.data_vars.values() if value.chunks}
                layout = next(iter(layouts)) if len(layouts) == 1 else ()
                # Keep already-bounded shared source chunks intact to avoid
                # reloading their payload across neighbouring output blocks.
                if layout and max(layout) <= chunks[dim]:
                    boundaries = np.cumsum((0, *layout)).tolist()
                    if any(value.chunks is None for value in data.data_vars.values()):
                        # Avoid singleton eager blocks anywhere in the layout.
                        # Prefer whole-source joins; if a full neighbour must be
                        # split, both slices retain at least two points so its
                        # Dask reduction keeps the original sample layout too.
                        index = 0
                        while index < len(boundaries) - 1:
                            if boundaries[index + 1] - boundaries[index] == 1:
                                left = boundaries[index] - boundaries[index - 1] if index else 0
                                right = (
                                    boundaries[index + 2] - boundaries[index + 1]
                                    if index + 2 < len(boundaries) else 0
                                )
                                if left and left + 1 <= chunks[dim]:
                                    del boundaries[index]
                                    index -= 1
                                elif right and right + 1 <= chunks[dim]:
                                    del boundaries[index + 1]
                                elif left > 3:
                                    boundaries[index] -= 2
                                elif right > 3:
                                    boundaries[index + 1] += 2
                            index += 1
                else:
                    boundaries = [*range(0, data.sizes[dim], chunks[dim]), data.sizes[dim]]
                    # A singleton edge can turn a strided NumPy sample reduction
                    # into a contiguous one. Keep at least two points when the
                    # policy allows, without overlapping or enlarging blocks.
                    if (
                        len(boundaries) > 2
                        and boundaries[-1] - boundaries[-2] == 1
                        and chunks[dim] > 2
                    ):
                        boundaries[-2] -= 1
                return xr.concat(
                    [
                        reduce_blocks(data.isel({dim: slice(start, stop)}), tuple(remaining), chunks, markers)
                        for start, stop in zip(boundaries[:-1], boundaries[1:])
                    ],
                    dim=dim,
                    combine_attrs="override",
                )
            return reduce_blocks(data, tuple(remaining), chunks, markers)
        if _prepare_block is not None:
            data = _prepare_block(data)
        # Joint computation preserves sharing and the Dask sample reduction order.
        return _calculate_stats(data, stats, **kwargs).drop_vars(markers, errors="ignore").compute()

    with dask.config.set(scheduler="single-threaded"):
        results = []
        for dimensions, names in groups.items():
            data = ds[names]
            # Scalar variables still belong to a dataset with these reduction
            # dimensions; dropping all of them would make Dataset.mean/std fail.
            markers = []
            for dim in sample_dims:
                if dim in ds.dims and dim not in data.dims:
                    if dim in ds.coords:
                        data = data.assign_coords(ds[[dim]].coords)
                    else:
                        # An auxiliary carrier supplies only the dimension size,
                        # not axis labels or an index. Its scalar broadcast view
                        # allocates no sample-sized payload and is removed before
                        # computation. Unique keys cannot shadow borrowed names.
                        marker = object()
                        markers.append(marker)
                        data = data.assign_coords(
                            {marker: (dim, np.broadcast_to(np.uint8(0), (ds.sizes[dim],)))}
                        )
            results.append(reduce_blocks(data, dimensions, _bounded_chunk_sizes(data, sample_dims), markers))
    return xr.merge(results)


def _calculate_stats(ds: xr.Dataset, stats: list[str], **kwargs) -> xr.Dataset:
    """Build the unchanged statistic kernels for one bounded block."""
    stats_datasets = []

    for stat in stats:
        if stat not in stats_functions:
            raise ValueError(f"Statistic {stat} not available.")
        sf = stats_functions[stat]

        # get any kwargs that could be passed to this stat
        sf_kwargs = {k: v for k, v in kwargs.items() if k in sf.params}

        # add "specific" kwargs of the form `<stat name>__<key>`
        for k, v in kwargs.items():
            if k.startswith(f"{stat}__"):
                sf_kwargs[k.removeprefix(f"{stat}__")] = v

        sf_result = sf.func(ds, **sf_kwargs)
        stats_datasets.append(sf_result)

    return xr.merge(stats_datasets)
