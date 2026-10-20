"""Execution regressions for shared concentration and country products."""

from dask.callbacks import Callback
import numpy as np
import pytest
import sparse
import xarray as xr

from openghg_inversions.array_ops import sparse_xr_dot
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.postprocessing.make_outputs import make_concentration_outputs
from openghg_inversions.postprocessing.stats import _calculate_stats, calculate_stats


@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_concentration_combination_bounds_producers(multisector_postprocessing_inv_out, monkeypatch, backend):
    """Boundary/offset arithmetic is blocked before shared uncertainty execution."""
    inv = multisector_postprocessing_inv_out()
    values = np.arange(20 * 257, dtype="float64").reshape(20, 257) % 11
    posterior = xr.Dataset(
        {
            name: (("chain", "draw", "nmeasure"), (values * factor)[None])
            for name, factor in (("y", 1), ("mu_bc", 2), ("offset", -1))
        },
        coords={"chain": [0], "draw": np.arange(20), "nmeasure": np.arange(257)},
    )
    if backend == "dask":
        posterior = posterior.chunk({"draw": 5})
    prior = posterior.isel(draw=slice(0, 16))
    inv.trace = xr.DataTree.from_dict({"posterior": posterior, "prior": prior})
    trace = inv.trace_dataset(var_roles=["concentration", "boundary", "offset"])
    for group in ("prior", "posterior"):
        trace[f"mu_bc_{group}"] = trace[f"mu_bc_{group}"] + trace[f"offset_{group}"]
        trace[f"mu_bc_{group}"].attrs["long_name"] = f"{group}_modelled_baseline_including_offset"
    options = {"hdi_prob": (0.68, 0.95)}
    statistics = ["mean", "stdev", "quantiles", "hdi"]
    expected = _calculate_stats(trace, statistics, **options).compute()
    payloads = []
    additions = []
    original_add = xr.DataArray.__add__

    def add(left, right):
        result = original_add(left, right)
        if str(left.name).startswith("mu_bc_"):
            additions.append(result.size)
        return result

    monkeypatch.setattr(xr.DataArray, "__add__", add)

    def record(key, value, *args):
        if isinstance(value, (np.ndarray, sparse.COO)) and 20 in value.shape:
            payloads.append(value.size)

    with Callback(posttask=record):
        actual = make_concentration_outputs(inv, statistics, options, combine_bc_and_offset=True)
    xr.testing.assert_identical(actual, expected)
    assert additions and max(additions) <= 20 * 128
    if backend == "dask":
        assert payloads and max(payloads) <= 20 * 128
    assert options == {"hdi_prob": (0.68, 0.95)}
    assert inv.trace.posterior.y.data is posterior.y.data


def test_country_projection_bounds_upstream_draw_tasks():
    """Projection bounds country/time work before constructing draw-wise totals."""
    weights = xr.DataArray(
        (np.arange(3 * 385 * 3, dtype="float64").reshape(3, 385, 3) % 7) - 3,
        dims=("flux_time", "country", "region"),
        coords={"flux_time": [0, 1, 2], "country": np.arange(385), "region": [2, 0, 1]},
    )
    samples = xr.Dataset(
        {"x": (("region", "draw"), np.arange(3 * 20, dtype="float64").reshape(3, 20) % 5)},
        coords={"region": [2, 0, 1], "draw": np.arange(20)},
    )
    original_weights = weights.data
    payloads = []

    def record(key, value, *args):
        if isinstance(value, (np.ndarray, sparse.COO)) and 20 in value.shape:
            payloads.append(value.size)

    with Callback(posttask=record):
        trace = Countries._get_country_trace("ch4", samples, weights)
        assert not payloads
        actual = calculate_stats(trace, stats=["mean", "stdev", "hdi"], hdi_prob=[0.68, 0.95])
    from openghg_inversions import convert

    expected_trace = sparse_xr_dot(weights, samples) * 365 * 24 * 3600 * convert.molar_mass("ch4")
    expected = _calculate_stats(expected_trace, ["mean", "stdev", "hdi"], hdi_prob=[0.68, 0.95]).compute()
    # Dask's unchanged mean kernel can round one float64 ULP differently from
    # the eager mean after this projection becomes a bounded lazy producer.
    np.testing.assert_array_max_ulp(actual.x_mean.values, expected.x_mean.values, maxulp=1)
    xr.testing.assert_identical(actual.assign(x_mean=expected.x_mean), expected)
    xr.testing.assert_identical(actual.x_mean.copy(data=expected.x_mean.data), expected.x_mean)
    assert payloads and max(payloads) <= 20 * 128
    assert weights.data is original_weights


def test_concentration_preserves_existing_source_blocks(multisector_postprocessing_inv_out):
    """Already-bounded loaders execute once across related variables and summaries."""
    from collections import Counter

    import dask.array as da
    from dask import delayed

    calls = Counter()
    values = np.arange(20 * 259, dtype="float32").reshape(1, 20, 259) % 7

    def source(start, stop):
        calls[start] += 1
        return values[:, :, start:stop]

    data = da.concatenate(
        [
            da.from_delayed(
                delayed(source)(start, min(start + 100, 259)),
                shape=(1, 20, min(100, 259 - start)),
                dtype=values.dtype,
            )
            for start in range(0, 259, 100)
        ],
        axis=2,
    )
    posterior = xr.Dataset(
        {
            name: (("chain", "draw", "nmeasure"), data * factor)
            for name, factor in (("y", 1), ("mu_bc", 2), ("offset", -1))
        }
    )
    inv = multisector_postprocessing_inv_out()
    inv.trace = xr.DataTree.from_dict({"posterior": posterior})
    graph = posterior.y.data.__dask_graph__()
    actual = make_concentration_outputs(
        inv, stats=["mean", "hdi"], stats_args={"hdi_prob": [0.68, 0.95]}, combine_bc_and_offset=True
    )
    assert calls == {0: 1, 100: 1, 200: 1}
    assert posterior.y.chunks == ((1,), (20,), (100, 100, 59))
    assert posterior.y.data.__dask_graph__() is graph
    np.testing.assert_array_equal(actual.mu_bc_posterior_mean, actual.y_posterior_mean)


def test_multisector_tiles_reuse_compact_sources_for_flux_and_scaling(multisector_postprocessing_inv_out):
    """Persist selected compact samples once across tiles, sectors and summaries."""
    from collections import Counter
    from dataclasses import replace

    import dask.array as da
    from dask import delayed

    from openghg_inversions.basis.basis_functions import BasisFunctions
    from openghg_inversions.postprocessing.make_outputs import make_flux_outputs

    calls = Counter()
    values = np.arange(20, dtype="float32").reshape(1, 20, 1) / 7

    def source(start):
        calls[start] += 1
        return values[:, start : start + 5]

    def unrequested():
        raise AssertionError("Flux products must not materialize concentration samples")

    samples = da.concatenate(
        [da.from_delayed(delayed(source)(start), (1, 5, 1), dtype="float32") for start in range(0, 20, 5)],
        axis=1,
    )
    posterior = xr.Dataset(
        {
            "x_ff": (("chain", "draw", "region"), samples + 1),
            "x_ocean": (("chain", "draw", "region"), samples * 0.5 + 2),
            "y": (
                ("chain", "draw", "nmeasure"),
                da.from_delayed(delayed(unrequested)(), (1, 20, 10000), dtype="float32"),
            ),
        },
        coords={"chain": [0], "draw": np.arange(20), "region": [0]},
    )
    prior = posterior.isel(draw=slice(0, 16))
    labels = xr.DataArray(
        np.ones((33, 3), dtype=int), dims=("lat", "lon"), coords={"lat": np.arange(33), "lon": [2, 0, 1]}
    )
    flux = xr.concat([labels.astype("float32"), labels.astype("float32") * 3], dim="source").assign_coords(
        source=["ff-inventory", "ocean-inventory"]
    )
    basis = BasisFunctions.from_flat_basis(labels, flux, operator_kwargs={"state_dim": "region"})
    inv = multisector_postprocessing_inv_out(basis)
    inv.trace = xr.DataTree.from_dict({"prior": prior, "posterior": posterior})
    original_graph = posterior.x_ff.data.__dask_graph__()
    original_chunks = posterior.x_ff.chunks
    options = {"stats": ["mean", "stdev", "quantiles", "hdi"], "stats_args": {"hdi_prob": [0.68, 0.95]}}
    actual = make_flux_outputs(
        inv, **options, include_scale_factors=True, report_flux_on_inversion_grid=False
    )

    assert calls == {start: 1 for start in range(0, 20, 5)}
    assert posterior.x_ff.data.__dask_graph__() is original_graph
    assert posterior.x_ff.chunks == original_chunks
    assert inv.trace.posterior.y.data is posterior.y.data
    assert actual.sizes["lat"] == 33
    assert "scaling_ff_posterior_hdi_95" in actual
    assert "flux_total_posterior_hdi_68" in actual

    # Retain the same sample chunks in the reference to preserve reduction order.
    eager_posterior = posterior[["x_ff", "x_ocean"]].compute().chunk({"draw": 5})
    eager_prior = eager_posterior.isel(draw=slice(0, 16))
    reference = replace(inv, trace=xr.DataTree.from_dict({"prior": eager_prior, "posterior": eager_posterior}))
    expected = make_flux_outputs(
        reference, **options, include_scale_factors=True, report_flux_on_inversion_grid=False
    )
    xr.testing.assert_identical(actual, expected)


def test_concentration_retains_eager_sample_layout(multisector_postprocessing_inv_out):
    """Column selection must not change float32 reduction order through a Dask copy."""
    inv = multisector_postprocessing_inv_out()
    values = np.random.default_rng(31).normal(1950, 200, (1000, 260)).astype("float32")
    values = values[:, np.arange(513) % 260]
    assert values.flags.f_contiguous
    posterior = xr.Dataset({"y": (("chain", "draw", "nmeasure"), values[None])})
    inv.trace = xr.DataTree.from_dict({"posterior": posterior})
    selected = inv.trace_dataset(var_roles="concentration")
    statistics = ["mean", "stdev", "quantiles", "hdi"]
    expected = _calculate_stats(selected, statistics, hdi_prob=[0.68, 0.95]).compute()
    actual = make_concentration_outputs(inv, stats=statistics, stats_args={"hdi_prob": [0.68, 0.95]})
    xr.testing.assert_identical(actual, expected)
    assert inv.trace.posterior.y.data is posterior.y.data
    assert posterior.y.data.flags.f_contiguous
    assert not np.array_equal(np.nanmean(values, axis=0), np.nanmean(np.ascontiguousarray(values), axis=0))


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("backend", ["numpy", "dask", "mixed"])
def test_combined_baseline_retains_native_reduction_kernels(
    multisector_postprocessing_inv_out, monkeypatch, order, backend
):
    """Bound addition without changing eager strides or existing Dask reductions."""
    inv = multisector_postprocessing_inv_out()
    values = np.random.default_rng(31).normal(1950, 200, (1000, 260)).astype("float32")
    values = np.array(values[:, np.arange(513) % 260], order=order)
    posterior = xr.Dataset(
        {
            name: (("chain", "draw", "nmeasure"), (values * factor)[None])
            for name, factor in (("y", 1), ("mu_bc", 2), ("offset", -1))
        },
        coords={"nmeasure": np.arange(513)[::-1]},
    )
    if backend == "dask":
        posterior = posterior.chunk({"draw": 250, "nmeasure": 100})
    elif backend == "mixed":
        posterior["offset"] = posterior.offset.chunk({"draw": 250, "nmeasure": 100})
    inv.trace = xr.DataTree.from_dict({"posterior": posterior})
    borrowed = {name: data.data for name, data in posterior.data_vars.items()}
    expected_trace = inv.trace_dataset(var_roles=["concentration", "boundary", "offset"])
    expected_trace["mu_bc_posterior"] = expected_trace.mu_bc_posterior + expected_trace.offset_posterior
    expected_trace.mu_bc_posterior.attrs["long_name"] = "posterior_modelled_baseline_including_offset"
    statistics = ["mean", "stdev", "quantiles", "hdi"]
    expected = _calculate_stats(expected_trace, statistics, hdi_prob=[0.68, 0.95]).compute()
    additions = []
    original_add = xr.DataArray.__add__

    def add(left, right):
        result = original_add(left, right)
        if left.name == "mu_bc_posterior":
            additions.append(result.sizes["nmeasure"])
        return result

    monkeypatch.setattr(xr.DataArray, "__add__", add)
    actual = make_concentration_outputs(
        inv, stats=statistics, stats_args={"hdi_prob": [0.68, 0.95]}, combine_bc_and_offset=True
    )
    xr.testing.assert_identical(actual, expected)
    assert additions and max(additions) <= 128
    assert min(additions) < max(additions)  # Includes the edge block.
    for name, data in posterior.data_vars.items():
        assert data.data is borrowed[name]


@pytest.mark.parametrize("offset_dims", [(), ("draw",)])
def test_combined_baseline_keeps_unbroadcast_offset_dimensions(
    multisector_postprocessing_inv_out, offset_dims
):
    """Temporary broadcast operands must not enlarge the reported offset product."""
    inv = multisector_postprocessing_inv_out()
    values = np.random.default_rng(31).normal(size=(20, 259)).astype("float32")
    posterior = xr.Dataset(
        {
            "y": (("chain", "draw", "nmeasure"), values[None]),
            "mu_bc": (("chain", "draw", "nmeasure"), (values * 2)[None]),
            "offset": (offset_dims, np.arange(20, dtype="float32") if offset_dims else np.float32(2)),
        },
        coords={"nmeasure": np.arange(259)[::-1], "draw": np.arange(20)},
    )
    inv.trace = xr.DataTree.from_dict({"posterior": posterior})
    trace = inv.trace_dataset(var_roles=["concentration", "boundary", "offset"])
    trace["mu_bc_posterior"] = trace.mu_bc_posterior + trace.offset_posterior
    trace.mu_bc_posterior.attrs["long_name"] = "posterior_modelled_baseline_including_offset"
    expected = _calculate_stats(trace, ["mean", "stdev"]).compute()
    actual = make_concentration_outputs(inv, stats=["mean", "stdev"], combine_bc_and_offset=True)
    xr.testing.assert_identical(actual, expected)
    assert actual.offset_posterior_mean.dims == ()
    assert posterior.offset.dims == offset_dims


@pytest.mark.parametrize("source_size, points", [(100, 401), (128, 513)])
def test_mixed_concentration_preserves_eager_singleton_edge(
    multisector_postprocessing_inv_out, source_size, points
):
    """Coalesce an edge when possible; otherwise only its neighbour is reloaded."""
    from collections import Counter

    import dask.array as da
    from dask import delayed

    values = np.random.default_rng(31).normal(1950, 200, (1000, points)).astype("float32")
    calls = Counter()

    def load(start, stop):
        calls[start] += 1
        return -values[:, start:stop]

    offsets = da.concatenate(
        [
            da.from_delayed(
                delayed(load)(start, min(start + source_size, points)),
                shape=(1000, min(source_size, points - start)),
                dtype=values.dtype,
            )
            for start in range(0, points, source_size)
        ],
        axis=1,
    )
    inv = multisector_postprocessing_inv_out()
    inv.trace = xr.DataTree.from_dict(
        {"posterior": xr.Dataset(
            {
                "y": (("draw", "nmeasure"), values),
                "mu_bc": (("draw", "nmeasure"), values * 2),
                "offset": (("draw", "nmeasure"), offsets),
            }
        )}
    )
    trace = inv.trace_dataset(var_roles=["concentration", "boundary", "offset"])
    trace["mu_bc_posterior"] = trace.mu_bc_posterior + trace.offset_posterior
    trace.mu_bc_posterior.attrs["long_name"] = "posterior_modelled_baseline_including_offset"
    statistics = ["mean", "stdev", "quantiles", "hdi"]
    expected = _calculate_stats(trace, statistics, hdi_prob=[0.68, 0.95]).compute()
    calls.clear()
    actual = make_concentration_outputs(
        inv, stats=statistics, stats_args={"hdi_prob": [0.68, 0.95]}, combine_bc_and_offset=True
    )
    xr.testing.assert_identical(actual, expected)
    expected_calls = {start: 1 for start in range(0, points, source_size)}
    if source_size == 128:
        expected_calls[384] = 2
    assert calls == expected_calls


def test_small_country_projection_retains_original_contraction():
    """An already-bounded matrix keeps its native kernel and float32 accumulation."""
    from openghg_inversions import convert

    rng = np.random.default_rng(31)
    weights = xr.DataArray(
        rng.uniform(0.01, 1, (3, 3, 256)).astype("float32"),
        dims=("flux_time", "country", "region"),
        coords={"flux_time": [0, 1, 2], "country": ["GBR", "FRA", "DEU"], "region": np.arange(256)[::-1]},
    )
    samples = xr.Dataset(
        {"x": (("region", "draw"), rng.uniform(0.01, 1, (256, 1000)).astype("float32"))},
        coords={"region": weights.region, "draw": np.arange(1000)},
    )
    original_weights = weights.data
    original_samples = samples.x.data
    expected = sparse_xr_dot(weights, samples) * 365 * 24 * 3600 * convert.molar_mass("ch4")
    actual = Countries._get_country_trace("ch4", samples, weights)
    assert actual.x.chunks is None
    xr.testing.assert_identical(actual, expected)
    assert weights.data is original_weights
    assert samples.x.data is original_samples
