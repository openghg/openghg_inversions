"""Numerical and execution regressions for shared bounded statistics."""

from collections import Counter

import arviz as az
import dask.array as da
from dask import delayed
from dask.callbacks import Callback
import numpy as np
import pytest
import sparse
import xarray as xr

from openghg_inversions.postprocessing import stats as module


@pytest.mark.parametrize("eager_companion", [False, True], ids=["dask-only", "mixed"])
@pytest.mark.parametrize(
    "layout",
    [
        (100, 1, 100), (1, 100, 100), (127, 1, 128),
        (2, 1, 128), (100, 1, 100, 100, 1),
        (1, 128), (128, 1, 128), (128, 1, 1, 128),
    ],
)
def test_singleton_source_blocks_keep_eager_and_lazy_reduction_layouts(layout, eager_companion):
    """Leading, internal and adjacent singleton chunks preserve native kernels."""
    values = np.random.default_rng(31).normal(1950, 200, (1000, sum(layout))).astype("float32")
    boundaries = np.cumsum((0, *layout))
    calls = Counter()

    def source(start, stop):
        calls[start] += 1
        return -values[:, start:stop]

    lazy = da.concatenate(
        [
            da.from_delayed(delayed(source)(start, stop), shape=(1000, stop - start), dtype=values.dtype)
            for start, stop in zip(boundaries[:-1], boundaries[1:])
        ],
        axis=1,
    )
    data = xr.Dataset(
        {
            "companion": (("draw", "point"), values if eager_companion else -lazy),
            "lazy": (("draw", "point"), lazy),
        },
        coords={"point": np.arange(sum(layout))[::-1]},
    )
    borrowed = data.lazy.data
    graph = borrowed.__dask_graph__()
    statistics = ["mean", "stdev", "quantiles", "hdi"]
    expected = module._calculate_stats(data, statistics, hdi_prob=[0.68, 0.95]).compute()
    calls.clear()
    payloads = []

    def observe(key, value, *args):
        if isinstance(value, np.ndarray) and 1000 in value.shape:
            payloads.append(value.size)

    with Callback(posttask=observe):
        actual = module.calculate_stats(data, statistics, hdi_prob=[0.68, 0.95])
    xr.testing.assert_identical(actual, expected)
    expected_calls = {start: 1 for start in boundaries[:-1]}
    if eager_companion and layout == (1, 128):
        expected_calls[1] = 2
    elif eager_companion and layout == (128, 1, 128):
        expected_calls[0] = 2
    assert calls == expected_calls
    assert payloads and max(payloads) <= 1000 * 128
    assert data.lazy.data is borrowed
    assert data.lazy.data.__dask_graph__() is graph


def test_statistic_chunk_policy_ignores_lazy_auxiliary_coordinate_layout():
    """Inspect numerical chunks without forcing or rechunking borrowed coordinates."""
    values = np.arange(20 * 259, dtype="float32").reshape(20, 259)
    data = xr.Dataset(
        {"x": (("draw", "point"), da.from_array(values, chunks=(20, 100)))},
        coords={"aux": ("point", da.arange(259, chunks=128))},
    )
    source = data.x.data
    source_graph = source.__dask_graph__()
    auxiliary = data.aux.data
    auxiliary_graph = auxiliary.__dask_graph__()
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        direct = module.hdi(data)
    assert not tasks
    expected = module._calculate_stats(data.compute(), ["mean", "hdi"]).compute()
    actual = module.calculate_stats(data, ["mean", "hdi"])
    xr.testing.assert_identical(actual, expected)
    xr.testing.assert_identical(direct.compute(), expected[["x_hdi_68"]])
    assert data.x.data is source
    assert data.x.data.__dask_graph__() is source_graph
    assert data.aux.data is auxiliary
    assert data.aux.data.__dask_graph__() is auxiliary_graph
    assert source.chunks == ((20,), (100, 100, 59))
    assert auxiliary.chunks == ((128, 128, 3),)


@pytest.mark.parametrize("sparse_payload", [False, True])
@pytest.mark.parametrize("source_size", [100, 128])
def test_statistics_reuse_bounded_sources_and_preserve_borrowed_graphs(sparse_payload, source_size):
    """Each source feeds both variables, summaries and HDI probabilities once."""
    calls = Counter()
    values = np.random.default_rng(7).uniform(0, 2, (19, 259)).astype("float32")
    values[12:, 5] = np.nan
    values[:, 17] = 0

    def source(start, stop):
        calls[start] += 1
        block = values[:, start:stop]
        return sparse.COO.from_numpy(block) if sparse_payload else block

    chunks = [
        da.from_delayed(
            delayed(source)(start, min(start + source_size, 259)),
            shape=(19, min(source_size, 259 - start)),
            dtype="float32",
            meta=sparse.COO.from_numpy(np.empty((0, 0))) if sparse_payload else None,
        )
        for start in range(0, 259, source_size)
    ]
    original = da.concatenate(chunks, axis=1)
    data = xr.Dataset(
        {"y": (("draw", "nmeasure"), original), "related": (("draw", "nmeasure"), original * 2)},
        coords={"draw": np.arange(19), "nmeasure": np.arange(259)[::-1]},
    )
    graph = data.y.data.__dask_graph__()
    original_chunks = data.y.chunks
    payloads = []

    def observe(key, value, dsk, state, worker):
        if isinstance(value, (np.ndarray, sparse.COO)):
            payloads.append(value.shape)

    with Callback(posttask=observe):
        actual = module.calculate_stats(
            data, stats=["mean", "stdev", "quantiles", "hdi"], hdi_prob=(p for p in [0.68, 0.95])
        )

    assert calls == {start: 1 for start in range(0, 259, source_size)}
    assert data.y.data is original
    assert data.y.data.__dask_graph__() is graph
    assert data.y.chunks == original_chunks
    assert payloads and max(np.prod(shape) for shape in payloads) <= 19 * 128
    assert all(value.chunks is None for value in actual.data_vars.values())
    expected = module._calculate_stats(
        data, ["mean", "stdev", "quantiles", "hdi"], hdi_prob=[0.68, 0.95]
    ).compute()
    xr.testing.assert_identical(module._to_dense_dataset(actual), module._to_dense_dataset(expected))
    for name in expected:
        if "hdi" in name or "quantile" in name:
            xr.testing.assert_identical(actual[name], expected[name])


@pytest.mark.parametrize("sample_dim", ["draw", "samples"])
@pytest.mark.parametrize("option", ["hdi_prob", "hdi__hdi_prob", "shared"])
def test_hdi_generators_and_complete_sample_cores(sample_dim, option):
    """Complete chain/sample cores survive NaNs, edge blocks and generator aliasing."""
    values = np.random.default_rng(14).normal(size=(2, 17, 131)).astype("float32")
    values[0, -3:, 9] = np.nan
    data = xr.Dataset({"x": (("chain", sample_dim, "state"), values)}).chunk(
        {"chain": 1, sample_dim: 5, "state": 128}
    )
    probability = (p for p in (0.68, 0.95))
    options = (
        {option: probability}
        if option != "shared"
        else {
            "hdi_prob": probability,
            "hdi__hdi_prob": probability,
        }
    )
    result = module.calculate_stats(data, stats=["hdi"], sample_dim=sample_dim, **options)
    reference = data.compute()
    for prob in (0.68, 0.95):
        expected = az.hdi(reference, prob=prob, dim=("chain", sample_dim), skipna=True).rename({"ci_bound": "hdi"})
        xr.testing.assert_equal(
            result[f"x_hdi_{int(prob * 100)}"], expected.x.rename(f"x_hdi_{int(prob * 100)}")
        )
    assert result.hdi.attrs == {"ci_kind": "hdi", "ci_prob": 0.68}


@pytest.mark.parametrize("sparse_payload", [False, True])
def test_pooled_chain_statistics_share_bounded_sources(sparse_payload):
    """Pool complete chains inside blocks and retain devel's exact reference kernels."""
    values = np.random.default_rng(17).normal(size=(2, 17, 259)).astype("float32")
    values[1] += 20  # A first-chain-only reduction cannot pass this comparison.
    values[0, -3:, 9] = np.nan
    values[1, :, 19] = np.nan
    calls = Counter()

    def source(start, stop):
        calls[start] += 1
        block = values[:, :, start:stop]
        return sparse.COO.from_numpy(block) if sparse_payload else block

    samples = da.concatenate(
        [
            da.from_delayed(
                delayed(source)(start, min(start + 100, 259)),
                shape=(2, 17, min(100, 259 - start)),
                dtype="float32",
                meta=sparse.COO.from_numpy(np.empty((0, 0, 0))) if sparse_payload else None,
            )
            for start in range(0, 259, 100)
        ],
        axis=2,
    ).rechunk({0: 1, 1: 5})
    data = xr.Dataset(
        {"posterior": (("chain", "draw", "point"), samples), "related": (("chain", "draw", "point"), samples * 2)},
        coords={"chain": [4, 8], "draw": np.arange(17) + 10, "point": np.arange(259)[::-1]},
    )
    borrowed_graph = data.posterior.data.__dask_graph__()
    pooled, sample_dim = module.combine_chain_draw(data)
    # Sparse Dask std with NaNs is unsupported by sparse's mixed arithmetic.
    stats = ["mean", "quantiles", "hdi"] if sparse_payload else ["mean", "stdev", "quantiles", "hdi"]
    expected = xr.merge(
        [
            *[module.stats_functions[stat].func(pooled, sample_dim=sample_dim) for stat in stats if stat != "hdi"],
            module.hdi(module._to_dense_dataset(pooled).compute(), sample_dim=sample_dim, hdi_prob=[0.68, 0.95]),
        ]
    ).compute()
    calls.clear()
    payloads = []
    with Callback(
        posttask=lambda key, value, *args: payloads.append(value.size)
        if isinstance(value, (np.ndarray, sparse.COO)) else None
    ):
        actual = module.calculate_stats(data, stats, hdi_prob=[0.68, 0.95])
    xr.testing.assert_identical(module._to_dense_dataset(actual), module._to_dense_dataset(expected))
    assert "chain" not in actual.dims and "draw" not in actual.dims and "sample" not in actual.dims
    assert calls == {0: 1, 100: 1, 200: 1}
    assert max(payloads) <= 2 * 17 * 128
    assert data.posterior.data.__dask_graph__() is borrowed_graph


def test_generated_sample_override_retains_complete_chain_draw_axes():
    """An override may name devel's pooled sample axis before it exists."""
    values = np.random.default_rng(19).normal(size=(2, 259, 129)).astype("float32")
    values[1] += 20
    data = xr.Dataset({"x": (("chain", "draw", "point"), values)}).chunk(
        {"chain": 1, "draw": 64, "point": 64}
    )
    pooled, sample_dim = module.combine_chain_draw(data)
    expected = module.mean(pooled, sample_dim=sample_dim).compute()
    payloads = []
    with Callback(
        posttask=lambda key, value, *args: payloads.append(value.size)
        if isinstance(value, (np.ndarray, sparse.COO)) else None
    ):
        actual = module.calculate_stats(data, ["mean"], mean__sample_dim="sample")
    xr.testing.assert_identical(actual, expected)
    assert actual.sizes == {"point": 129}
    assert max(payloads) <= 2 * 259 * 128


def test_direct_hdi_stays_lazy_and_bounds_non_sample_chunks():
    """Split an oversized measurement chunk before complete-sample kernels execute."""
    values = da.from_array(np.arange(23 * 267, dtype="float32").reshape(23, 267), chunks=(6, 267))
    data = xr.Dataset({"y": (("draw", "nmeasure"), values)})
    seen = []
    with Callback(posttask=lambda *args: seen.append(args[1])):
        result = module.hdi(data, hdi_prob=[0.68, 0.95])
    assert not seen
    assert max(result.y_hdi_68.chunks[0]) <= 128
    xr.testing.assert_identical(result.compute(), module.hdi(data.compute(), hdi_prob=[0.68, 0.95]).compute())


def test_block_assembly_keeps_independent_state_axes_and_metadata():
    """Grouping variables never broadcasts unequal labelled state spaces together."""
    data = xr.Dataset(
        {
            "outer": (("draw", "outer_state"), np.ones((12, 137), dtype="float32"), {"units": "1"}),
            "inner": (("draw", "inner_state"), np.ones((12, 9), dtype="float32") * 2, {"domain": "inner"}),
        },
        coords={"outer_state": np.arange(137)[::-1], "inner_state": [9, 1, 7, 3, 5, 2, 6, 4, 8]},
    )
    expected = module._calculate_stats(data, ["mean", "quantiles"]).compute()
    actual = module.calculate_stats(data, stats=["mean", "quantiles"])
    xr.testing.assert_identical(actual, expected)


def test_effective_hdi_probability_does_not_consume_overridden_generator():
    """An ignored generic iterable remains untouched by a stat-specific override."""
    generic = (p for p in (0.5,))
    data = xr.Dataset({"x": (("draw", "state"), np.ones((20, 129)))})
    result = module.calculate_stats(
        data, stats=["hdi"], hdi_prob=generic, hdi__hdi_prob=(p for p in (0.68, 0.95))
    )
    assert list(generic) == [0.5]
    assert set(result.data_vars) == {"x_hdi_68", "x_hdi_95"}


def test_multiple_flux_times_edge_blocks_and_exact_companion_kernels(monkeypatch):
    """All sample-dependent companions use the same bounded spatial/time slices."""
    monkeypatch.setattr(module, "_STATS_CHUNKS", {"lat": 2, "lon": 2, "flux_time": 1})
    values = np.random.default_rng(21).normal(size=(25, 5, 7, 3)).astype("float32")
    values[:, 0, 0, :] = 0
    data = xr.Dataset(
        {"flux": (("draw", "lat", "lon", "flux_time"), values, {"units": "mol m-2 s-1"})},
        coords={"lat": [4, 3, 2, 1, 0], "lon": range(7), "flux_time": [10, 20, 30]},
    ).chunk({"draw": 5, "lat": 2, "lon": 2, "flux_time": 1})
    original_attrs = data.flux.attrs.copy()
    options = {"stats": ["mean", "stdev", "median", "mode", "quantiles", "hdi"], "hdi_prob": [0.68, 0.95]}
    sizes = []
    with Callback(
        posttask=lambda key, value, *args: sizes.append(value.size) if isinstance(value, np.ndarray) else None
    ):
        actual = module.calculate_stats(data, **options)
    expected = module._calculate_stats(data, **options).compute()
    xr.testing.assert_identical(actual, expected)
    assert data.flux.attrs == original_attrs
    assert max(sizes) <= 25 * 2 * 2


@pytest.mark.parametrize("sample_dim", [None, ["draw", "state"]])
@pytest.mark.parametrize("specific", [False, True])
def test_reduction_over_multiple_or_all_dimensions(sample_dim, specific):
    """Global/list reductions remain scalar rather than becoming per-tile summaries."""
    data = xr.Dataset({"x": (("draw", "state"), np.arange(2 * 129, dtype="float64").reshape(2, 129))}).chunk(
        {"draw": 1, "state": 100}
    )
    options = {"mean__sample_dim" if specific else "sample_dim": sample_dim}
    actual = module.calculate_stats(data, stats=["mean"], **options)
    expected = module._calculate_stats(data, ["mean"], **options).compute()
    xr.testing.assert_identical(actual, expected)
    assert actual.x_mean.ndim == 0


@pytest.mark.parametrize("argument", ["sample_dim", "mean__sample_dim"])
def test_iterator_reduction_dimensions_are_reused_by_the_kernel(argument):
    data = xr.Dataset({"x": (("draw", "state"), np.arange(258, dtype="float32").reshape(2, 129))})
    actual = module.calculate_stats(data, ["mean"], **{argument: iter(("draw", "state"))})
    expected = module._calculate_stats(data, ["mean"], **{argument: ("draw", "state")})
    xr.testing.assert_identical(actual, expected)
    assert actual.x_mean.item() == 128.5


@pytest.mark.parametrize("generic", [False, True])
def test_shared_reduction_iterator_is_consumed_once(generic):
    data = xr.Dataset({"x": (("draw", "state"), np.arange(258, dtype="float32").reshape(2, 129))})
    dimensions = iter(("draw", "state"))
    options = {"mean__sample_dim": dimensions, "sample_dim" if generic else "stdev__sample_dim": dimensions}
    actual = module.calculate_stats(data, ["mean", "stdev"], **options)
    expected = module._calculate_stats(
        data, ["mean", "stdev"], **{key: ("draw", "state") for key in options}
    )
    xr.testing.assert_identical(actual, expected)


def test_unused_reduction_iterator_is_not_consumed():
    data = xr.Dataset({"x": (("draw", "state"), np.arange(258, dtype="float32").reshape(2, 129))})
    generic = iter(("draw",))
    actual = module.calculate_stats(data, ["mean"], sample_dim=generic, mean__sample_dim=iter(("state",)))
    expected = module._calculate_stats(data, ["mean"], mean__sample_dim="state")
    xr.testing.assert_identical(actual, expected)
    assert tuple(generic) == ("draw",)


@pytest.mark.parametrize("registered_name", ["custom_mean", "mean"])
@pytest.mark.parametrize("callable_instance", [False, True])
@pytest.mark.parametrize("explicit_dimension", [False, True])
def test_registered_reduction_keeps_devel_arguments_and_complete_dataset(
    monkeypatch, registered_name, callable_instance, explicit_dimension
):
    monkeypatch.setattr(module, "stats_functions", dict(module.stats_functions))
    calls = []

    def custom_mean(data, sample_dim="state"):
        calls.append(data.sizes)
        return data.mean(sample_dim)

    custom_mean.__name__ = registered_name
    if callable_instance:
        class CustomMean:
            __hash__ = None
            __name__ = registered_name

            def __call__(self, data, sample_dim="state"):
                return custom_mean(data, sample_dim)

        module.register_stat(CustomMean())
    else:
        module.register_stat(custom_mean)
    data = xr.Dataset({"x": (("draw", "state"), np.arange(258, dtype="float32").reshape(2, 129))})
    options = {"sample_dim": "state"} if explicit_dimension else {}
    actual = module.calculate_stats(data, [registered_name], **options)
    # Devel resolves the generic default to draw before invoking custom kernels.
    xr.testing.assert_identical(actual, data.mean("state" if explicit_dimension else "draw"))
    assert len(calls) == 1
    assert calls[0]["state"] == 129


def test_whole_dataset_statistic_receives_all_operands(monkeypatch):
    monkeypatch.setattr(module, "stats_functions", dict(module.stats_functions))
    calls = []

    @module.register_stat(whole_dataset=True)
    def cross_moment(data):
        calls.append(tuple(data.data_vars))
        return (data.x * data.y).mean("draw").to_dataset(name="cross_moment")

    data = xr.Dataset(
        {
            "x": (("draw", "state"), np.arange(258, dtype="float32").reshape(2, 129)),
            "y": ("draw", [2.0, 3.0]),
        }
    )
    expected = module._calculate_stats(data, ["mean", "cross_moment"])
    calls.clear()
    actual = module.calculate_stats(data, ["mean", "cross_moment"])
    xr.testing.assert_identical(actual, expected)
    assert calls == [("x", "y")]


@pytest.mark.parametrize("coordinate_state", ["absent", "indexed", "unindexed"])
@pytest.mark.parametrize("scalar_value", [7.0, True])
def test_scalar_groups_preserve_original_dimension_indexes(coordinate_state, scalar_value):
    data = xr.Dataset(
        {
            "x": (("draw", "state", "point"), np.arange(3 * 2 * 129).reshape(3, 2, 129)),
            "scalar": ((), scalar_value),
        },
        coords={"draw": [6, 4, 2], "state": [9, 3]} if coordinate_state != "absent" else None,
    )
    if coordinate_state == "unindexed":
        data = data.drop_indexes(["draw", "state"])
    original = data.copy(deep=True)
    options = {"mean__sample_dim": "state", "quantiles__sample_dim": "draw"}
    statistics = ["mean", "stdev"] if isinstance(scalar_value, bool) else ["mean", "stdev", "quantiles"]
    expected = module._calculate_stats(data, statistics, **options)
    actual = module.calculate_stats(data, statistics, **options)
    xr.testing.assert_identical(actual, expected)
    assert set(actual.indexes) == set(expected.indexes)
    assert actual.scalar_mean.item() == scalar_value
    assert actual.scalar_stdev.item() == 0.0
    xr.testing.assert_identical(data, original)


def test_custom_kernel_fallback_pools_devel_chains(monkeypatch):
    """Whole-dataset custom kernels receive devel's pooled sample axis and all operands."""
    monkeypatch.setattr(module, "stats_functions", module.stats_functions.copy())

    @module.register_stat(whole_dataset=True)
    def custom_difference(data, sample_dim="draw"):
        return (data.x - data.y).mean(sample_dim).to_dataset(name="difference")

    data = xr.Dataset(
        {
            "x": (("chain", "draw", "point"), np.arange(24).reshape(2, 3, 4)),
            "y": (("chain", "draw", "point"), np.ones((2, 3, 4))),
        }
    )
    pooled, sample_dim = module.combine_chain_draw(data)
    xr.testing.assert_identical(
        module.calculate_stats(data, ["custom_difference"]),
        custom_difference(pooled, sample_dim=sample_dim),
    )


def test_specific_reduction_override_follows_generic_chain_pooling():
    """A non-sample override reduces its axis after generic chain/draw pooling."""
    data = xr.Dataset({"x": (("chain", "draw", "point"), np.arange(24).reshape(2, 3, 4))})
    pooled, _ = module.combine_chain_draw(data)
    expected = module.mean(pooled, sample_dim="point")
    actual = module.calculate_stats(data, ["mean"], mean__sample_dim="point")
    xr.testing.assert_identical(actual, expected)
    assert actual.sizes == {"sample": 6}


def test_custom_statistic_is_called_per_variable_by_default(monkeypatch):
    """Devel invoked registered statistics once for each single-variable dataset."""
    monkeypatch.setattr(module, "stats_functions", dict(module.stats_functions))
    calls = []

    @module.register_stat
    def data_range(data, sample_dim="draw"):
        calls.append(tuple(data.data_vars))
        (name,) = data.data_vars
        values = data[name]
        return (values.max(sample_dim) - values.min(sample_dim)).to_dataset(name=f"{name}_range")

    data = xr.Dataset(
        {
            "x": (("draw", "point"), np.arange(12.0).reshape(3, 4)),
            "y": ("draw", [2.0, 5.0, 3.0]),
        }
    )
    actual = module.calculate_stats(data, ["mean", "data_range"])
    assert sorted(calls) == [("x",), ("y",)]
    xr.testing.assert_equal(actual.x_range, data.x.max("draw") - data.x.min("draw"))
    assert actual.y_range.item() == 3.0
    xr.testing.assert_equal(actual.x_mean, data.x.mean("draw"))


def test_pointwise_custom_statistic_keeps_bounded_blocks(monkeypatch):
    """Declaring per-point independence keeps the built-in block bound."""
    monkeypatch.setattr(module, "stats_functions", dict(module.stats_functions))
    sizes = []

    @module.register_stat(pointwise=True)
    def rms(data, sample_dim="draw"):
        sizes.append(dict(data.sizes))
        return np.sqrt((data**2).mean(sample_dim)).rename({name: f"{name}_rms" for name in data.data_vars})

    data = xr.Dataset({"x": (("draw", "point"), np.arange(3 * 300, dtype=float).reshape(3, 300))})
    actual = module.calculate_stats(data, ["mean", "rms"])
    assert len(sizes) > 1
    assert all(size["point"] <= module._bounded_chunk_sizes(data)["point"] for size in sizes)
    xr.testing.assert_allclose(actual.x_rms, np.sqrt((data.x**2).mean("draw")))


def test_undeclared_custom_statistic_uses_complete_dataset(monkeypatch):
    """Custom statistics without a pointwise declaration keep the unrestricted fallback."""
    monkeypatch.setattr(module, "stats_functions", dict(module.stats_functions))
    sizes = []

    @module.register_stat
    def rms(data, sample_dim="draw"):
        sizes.append(dict(data.sizes))
        return np.sqrt((data**2).mean(sample_dim)).rename({name: f"{name}_rms" for name in data.data_vars})

    data = xr.Dataset({"x": (("draw", "point"), np.arange(3 * 300, dtype=float).reshape(3, 300))})
    module.calculate_stats(data, ["mean", "rms"])
    assert sizes == [{"draw": 3, "point": 300}]
