"""Shared reconstruction bounds, including independently retained native grids."""

from __future__ import annotations

from dataclasses import replace

import arviz as az
from dask.callbacks import Callback
import numpy as np
import pytest
import sparse
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.postprocessing import make_outputs
from openghg_inversions.postprocessing import _basis_products
from openghg_inversions.postprocessing._basis_products import (
    reconstruct_flux_stats,
    reconstruct_scale_factor_stats,
)


def _domain_basis(domain, *, grouped=False, operator_chunks=None):
    """Independent outer/inner fixtures, with a pre-applied outer overlap mask."""
    nlat, nlon, nstate = (5, 7, 3) if domain == "outer" else (7, 4, 2)
    labels = np.array([2, 7, 11][:nstate])
    flat = xr.DataArray(
        labels[np.indices((nlat, nlon)).sum(axis=0) % nstate],
        dims=("lat", "lon"),
        coords={"lat": np.arange(nlat)[::-1] / 10, "lon": np.arange(nlon) / 10},
    )
    values = (np.arange(nlat * nlon).reshape(nlat, nlon) % 5 - 2).astype("float32")
    overlap = np.zeros_like(values, dtype=bool)
    if domain == "outer":
        overlap[1:3, 2:5] = True
        values[overlap] = 0
    flux = xr.DataArray(
        values[..., None] * np.array([1, 2, -1], dtype="float32"),
        dims=("lat", "lon", "flux_time"),
        coords={**flat.coords, "flux_time": [0, 1, 2], "overlap_mask": (("lat", "lon"), overlap)},
        attrs={"units": "mol/m2/s", "domain": domain},
    ).chunk({"lat": 2, "lon": 3, "flux_time": 3})
    metadata = None
    if grouped:
        metadata = xr.Dataset(
            {
                "basis_group": ("basis_label", ["west", "east", "west"][:nstate]),
                "basis_partition": ("basis_label", ["coastal", "inland", "coastal"][:nstate]),
                "region_in_partition": ("basis_label", [1, 1, 2][:nstate]),
            },
            coords={"basis_label": labels},
        )
    return BasisFunctions.from_flat_basis(
        flat,
        flux,
        region_labels="basis_values",
        operator_kwargs={"state_dim": "region", "state_metadata": metadata, "chunks": operator_chunks},
        metadata={"openghg_inversions:domain": domain},
    )


@pytest.mark.parametrize("domain", ["outer", "inner"])
@pytest.mark.parametrize("grouped", [False, True], ids=["ungrouped", "grouped"])
@pytest.mark.parametrize("inversion_grid", [False, True], ids=["native", "region-mean"])
@pytest.mark.parametrize("sparse_state", [False, True], ids=["numpy-state", "sparse-state"])
def test_reconstruction_keeps_labelled_native_domains_and_borrowed_inputs(
    domain, grouped, inversion_grid, sparse_state
):
    """Grouping is state metadata, not a sector or a second spatial domain."""
    basis = _domain_basis(domain, grouped=grouped)
    region = basis.operator.basis_matrix.region.values[::-1]
    values = np.arange(len(region) * 2, dtype="float32").reshape(len(region), 2) - 2
    state = xr.Dataset(
        {"summary": (("region", "quantile"), sparse.COO.from_numpy(values) if sparse_state else values)},
        coords={"region": region, "quantile": [0.25, 0.75]},
        attrs={"contract": "independent-native-grid"},
    ).chunk({"region": -1})
    state.summary.attrs["units"] = "1"
    flux_data = basis.flux.data
    flux_graph = flux_data.__dask_graph__()
    matrix_data = basis.operator.basis_matrix.data
    state_data = state.summary.data
    state_graph = state_data.__dask_graph__()
    before_metadata = basis.operator.state_metadata
    expected = (
        reconstruct_flux_stats(basis, basis.flux, state, report_flux_on_inversion_grid=inversion_grid)
        .map(to_dense, keep_attrs=True)
        .compute(scheduler="single-threaded")
    )
    actual = reconstruct_flux_stats(
        basis,
        basis.flux,
        state,
        report_flux_on_inversion_grid=inversion_grid,
        _chunks={"lat": 2, "lon": 3, "flux_time": 1},
    ).compute(scheduler="single-threaded")
    xr.testing.assert_identical(actual, expected)
    xr.testing.assert_identical(actual.lat, basis.flux.lat)
    xr.testing.assert_identical(actual.lon, basis.flux.lon)
    if not inversion_grid:
        xr.testing.assert_identical(actual.overlap_mask, basis.flux.overlap_mask)
    assert basis.metadata["openghg_inversions:domain"] == domain
    assert basis.flux.attrs["domain"] == domain
    assert basis.flux.data is flux_data
    assert basis.flux.data.__dask_graph__() is flux_graph
    assert basis.operator.basis_matrix.data is matrix_data
    assert state.summary.data is state_data
    assert state.summary.data.__dask_graph__() is state_graph
    if grouped:
        xr.testing.assert_identical(basis.operator.state_metadata, before_metadata)


@pytest.mark.parametrize("keep_attrs", [False, True])
def test_bounded_conversions_preserve_metadata_independently_of_xarray_defaults(keep_attrs):
    """Payload conversion must not depend on the installed xarray attr default."""
    basis = _domain_basis("outer")
    state = xr.Dataset(
        {"summary": ("region", np.ones(3, dtype="float32"), {"units": "1"})},
        coords={"region": [11, 2, 7]},
        attrs={"contract": "retained-summary-metadata"},
    )
    with xr.set_options(keep_attrs=keep_attrs):
        expected_flux = (
            reconstruct_flux_stats(basis, basis.flux, state, report_flux_on_inversion_grid=False)
            .map(to_dense, keep_attrs=True)
            .compute(scheduler="single-threaded")
        )
        actual_flux = reconstruct_flux_stats(
            basis,
            basis.flux,
            state,
            report_flux_on_inversion_grid=False,
            _chunks={"lat": 2, "lon": 3, "flux_time": 1},
        ).compute(scheduler="single-threaded")
        expected_scale = (
            reconstruct_scale_factor_stats(basis, state)
            .map(to_dense, keep_attrs=True)
            .compute(scheduler="single-threaded")
        )
        actual_scale = reconstruct_scale_factor_stats(basis, state, _chunks={"lat": 2, "lon": 3}).compute(
            scheduler="single-threaded"
        )
    xr.testing.assert_identical(actual_flux, expected_flux)
    xr.testing.assert_identical(actual_scale, expected_scale)
    for product in (expected_flux, actual_flux, expected_scale, actual_scale):
        assert product.attrs["contract"] == state.attrs["contract"]
        assert product.summary.attrs == state.summary.attrs == {"units": "1"}
    assert "openghg_inversions:non_finite_flux" in actual_flux.attrs


@pytest.mark.parametrize("inversion_grid", [False, True])
@pytest.mark.parametrize("summary_count", [1, 2], ids=["scalar-summary", "two-quantiles"])
def test_reconstruction_bounds_numpy_and_sparse_upstream_tasks(inversion_grid, summary_count):
    """Measure producer payloads, not only the final tile layout."""
    basis = _domain_basis("outer", grouped=True)
    state = xr.Dataset(
        {"summary": (("region", "quantile"), sparse.COO.from_numpy(np.ones((3, 2), dtype="float32")))},
        coords={"region": [11, 2, 7], "quantile": [0.25, 0.75]},
    ).chunk({"region": -1})
    if summary_count == 1:
        state = state.isel(quantile=0, drop=True)
    outputs = []
    with Callback(posttask=lambda key, value, *args: outputs.append((key, value))):
        result = reconstruct_flux_stats(
            basis,
            basis.flux,
            state,
            report_flux_on_inversion_grid=inversion_grid,
            _chunks={"lat": 2, "lon": 3, "flux_time": 1},
        )
    if not inversion_grid:
        assert not outputs
    # Region means use the original spatial reduction; interpolation has its
    # own bounded graph and never expands compact statistics to grid-by-draw.
    outputs.clear()
    with Callback(posttask=lambda key, value, *args: outputs.append((key, value))):
        # Observe individual producers even when dense conversion can fuse them.
        materialized = result.compute(scheduler="single-threaded", optimize_graph=False)
    assert isinstance(materialized.summary.data, np.ndarray)
    sparse_payloads = [value for _, value in outputs if isinstance(value, sparse.COO)]
    numpy_payloads = [value for _, value in outputs if isinstance(value, np.ndarray)]
    assert sparse_payloads and numpy_payloads
    assert max(value.size for value in sparse_payloads) <= 2 * 3 * 3
    assert max(value.size for value in numpy_payloads) <= result.summary.size
    # Inputs to the final assembly may be complete outputs, but contraction
    # producers themselves have at most one spatial/time tile of two summaries.
    contractions = [value for key, value in outputs if "tensordot" in str(key) or "einsum" in str(key)]
    assert contractions
    assert all(isinstance(value, (np.ndarray, sparse.COO)) for value in contractions)
    assert max(value.size for value in contractions) <= 2 * 3 * summary_count
    assert isinstance(result.summary.data._meta, np.ndarray)
    assert max(result.summary.chunksizes["lat"]) <= 2
    assert max(result.summary.chunksizes["lon"]) <= 3
    assert max(result.summary.chunksizes["flux_time"]) == 1


def test_scale_reconstruction_chunks_geometry_without_computing_borrowed_states():
    basis = _domain_basis("inner", grouped=True)
    state = xr.Dataset({"mean": ("region", np.array([1, 2], dtype="float32"))}, coords={"region": [7, 2]})
    state = state.chunk({"region": -1})
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        result = reconstruct_scale_factor_stats(basis, state, _chunks={"lat": 2, "lon": 3})
    assert not tasks
    expected = reconstruct_scale_factor_stats(basis, state).compute().map(to_dense, keep_attrs=True)
    xr.testing.assert_identical(result.compute(), expected)
    assert max(result["mean"].chunksizes["lat"]) == 2
    assert max(result["mean"].chunksizes["lon"]) == 3


def test_chunked_geometry_reuses_constructed_sparse_payload_without_reloading_flat():
    """The constructor owns geometry materialization; reconstruction only borrows it."""
    retained = _domain_basis("outer", grouped=True)
    loads = []

    def load(block):
        loads.append(block.shape)
        return block

    flat = retained.operator.basis_flat.chunk({"lat": 3, "lon": 4})
    flat = flat.copy(data=flat.data.map_blocks(load, meta=np.empty((0, 0), dtype=flat.dtype)))
    original_flat_graph = flat.data.__dask_graph__()
    basis = BasisFunctions.from_flat_basis(
        flat,
        retained.flux,
        region_labels="basis_values",
        operator_kwargs={
            "state_dim": "region",
            "chunks": {"region": 1},
            "state_metadata": retained.operator.state_metadata,
        },
    )
    assert loads
    loads.clear()
    original_matrix = basis.operator.basis_matrix.data
    original_graph = original_matrix.__dask_graph__()
    state = xr.Dataset(
        {"mean": ("region", np.array([2, 1, 3], dtype="float32"))},
        coords={"region": [11, 2, 7]},
    )
    for limits in ({"lat": 2, "lon": 3, "flux_time": 1}, {"lat": 3, "lon": 2, "flux_time": 1}):
        tasks = []
        with Callback(pretask=lambda key, *args: tasks.append(key)):
            bounded = _basis_products._chunk_basis_functions(basis, limits)
            scale = reconstruct_scale_factor_stats(basis, state, _chunks=limits)
            flux = reconstruct_flux_stats(
                basis, basis.flux, state, report_flux_on_inversion_grid=False, _chunks=limits
            )
        assert not tasks
        assert bounded.operator._unchunked_basis_matrix is basis.operator._unchunked_basis_matrix
        assert bounded.operator.basis_matrix.variable.chunksizes["region"] == (1, 1, 1)
        scale.compute(scheduler="single-threaded")
        flux.compute(scheduler="single-threaded")
    assert not loads
    assert flat.data.__dask_graph__() is original_flat_graph
    assert basis.operator.basis_matrix.data is original_matrix
    assert basis.operator.basis_matrix.data.__dask_graph__() is original_graph


@pytest.mark.parametrize("inversion_grid", [False, True])
def test_related_reconstruction_outputs_reuse_the_borrowed_flux_source(inversion_grid):
    basis = _domain_basis("outer")
    loads = []

    def load(block):
        loads.append(block.shape)
        return block

    flux = basis.flux.copy(data=basis.flux.data.map_blocks(load, meta=np.empty((0, 0, 0), dtype="float32")))
    basis = replace(basis, flux=flux)
    borrowed = flux.data
    graph = borrowed.__dask_graph__()
    state = xr.Dataset(
        {
            "first": ("region", np.ones(3, dtype="float32")),
            "second": ("region", np.full(3, 2, dtype="float32")),
        },
        coords={"region": [11, 2, 7]},
    )
    output = reconstruct_flux_stats(
        basis,
        flux,
        state,
        report_flux_on_inversion_grid=inversion_grid,
        _chunks={"lat": 2, "lon": 3, "flux_time": 1},
    ).compute(scheduler="single-threaded")
    assert len(loads) == np.prod(borrowed.numblocks)
    np.testing.assert_array_equal(to_dense(output.second), to_dense(output["first"]) * 2)
    assert flux.data is borrowed
    assert flux.data.__dask_graph__() is graph


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sparse_state", [False, True])
@pytest.mark.parametrize("missing", ["partial", "all"])
def test_weighted_tiles_preserve_original_zero_nan_masks_and_reuse_sources(
    monkeypatch, dtype, sparse_state, missing
):
    """Refine unequal original chunks without changing sparse zero pruning."""
    basis = _domain_basis("outer", grouped=True, operator_chunks={"lat": 3, "lon": 4})
    loads = []

    def source(block):
        loads.append(block.shape)
        return block

    flux = basis.flux.astype(dtype)
    flux = flux.copy(data=flux.data.map_blocks(source, meta=np.empty((0, 0, 0), dtype=dtype)))
    basis = replace(basis, flux=flux)
    values = np.array([[2, 3], [0, 1], [-1, 2]], dtype=dtype)
    values[: 1 if missing == "partial" else 3, 1] = np.nan
    state = xr.Dataset(
        {"first": (("region", "hdi"), sparse.COO.from_numpy(values) if sparse_state else values)},
        coords={"region": [11, 2, 7], "hdi": ["lower", "higher"]},
    ).chunk({"region": -1})
    state["second"] = state["first"] * 2
    expected = (
        reconstruct_flux_stats(basis, flux, state, report_flux_on_inversion_grid=False)
        .map(to_dense, keep_attrs=True)
        .compute(scheduler="single-threaded")
    )
    loads.clear()
    flux_data, flux_graph = flux.data, flux.data.__dask_graph__()
    matrix_data = basis.operator.basis_matrix.data
    state_data = state["first"].data
    payloads = []
    multiply = _basis_products._weighted_basis_block

    def observe(*args, **kwargs):
        result = multiply(*args, **kwargs)
        payloads.append((result.size, result.nbytes, result.nnz))
        return result

    monkeypatch.setattr(_basis_products, "_weighted_basis_block", observe)
    actual = reconstruct_flux_stats(
        basis,
        flux,
        state,
        report_flux_on_inversion_grid=False,
        _chunks={"lat": 2, "lon": 2, "flux_time": 1},
    ).compute(scheduler="single-threaded")
    xr.testing.assert_identical(actual, expected)
    assert payloads and max(size for size, _, _ in payloads) <= 2 * 2 * 3
    assert len(loads) == np.prod(flux_data.numblocks)
    assert flux.data is flux_data and flux.data.__dask_graph__() is flux_graph
    assert basis.operator.basis_matrix.data is matrix_data
    assert state["first"].data is state_data


@pytest.mark.parametrize("prior_draws", [16, 20])
def test_public_padded_hdis_keep_signed_zero_cells(
    multisector_postprocessing_inv_out, monkeypatch, prior_draws
):
    """Four padded prior draws leave an upper HDI endpoint missing in 0.7."""
    flat = xr.DataArray([[1], [1]], dims=("lat", "lon"), coords={"lat": [0, 1], "lon": [0]})
    flux = xr.DataArray(
        np.array([[[-1, 1]], [[-0.0, 0.0]]], dtype="float32"),
        dims=("lat", "lon", "flux_time"),
        coords={**flat.coords, "flux_time": [0, 1]},
    )
    output = multisector_postprocessing_inv_out(
        BasisFunctions.from_flat_basis(flat, flux, operator_kwargs={"state_dim": "region"})
    )
    output.run_metadata["split_by_sectors"] = False
    output.trace = az.from_dict(
        posterior={"x": np.arange(1, 21, dtype="float32").reshape(1, 20, 1)},
        prior={"x": np.arange(1, prior_draws + 1, dtype="float32").reshape(1, prior_draws, 1)},
        coords={"region": [0]},
        dims={"x": ["region"]},
    )

    def product():
        probabilities = (value for value in (0.68, 0.95))
        return make_outputs.make_flux_outputs(
            output,
            stats=["mean", "stdev", "quantiles", "hdi"],
            stats_args={"hdi_prob": probabilities, "hdi__hdi_prob": probabilities},
            report_flux_on_inversion_grid=False,
        )

    original = make_outputs.reconstruct_flux_stats

    def unbounded(*args, **kwargs):
        kwargs.pop("_chunks", None)
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(make_outputs, "reconstruct_flux_stats", unbounded)
        expected = product()
    actual = product()
    xr.testing.assert_identical(actual, expected)
    for probability in (68, 95):
        endpoint = actual[f"flux_prior_hdi_{probability}"].sel(flux_time=1, lat=1, lon=0, hdi="higher")
        assert bool(endpoint.isnull()) == (prior_draws == 16)
        assert not bool(actual[f"flux_posterior_hdi_{probability}"].isnull().any())


@pytest.mark.parametrize("grouped", [False, True])
def test_standard_common_postprocessing_keeps_outer_and_inner_separate(
    multisector_postprocessing_inv_out, monkeypatch, grouped
):
    """Common views only: this does not backport the devel nested runner."""
    monkeypatch.setattr(make_outputs, "_FLUX_STATS_CHUNKS", {"lat": 2, "lon": 3, "flux_time": 1})
    products = {}
    for domain in ("outer", "inner"):
        basis = _domain_basis(domain, grouped=grouped)
        output = multisector_postprocessing_inv_out(basis)
        output.run_metadata["split_by_sectors"] = False
        output.model_metadata = {"species": "ch4", "domain": domain}
        region = basis.operator.basis_matrix.region.values[::-1]
        draws = np.arange(19 * len(region), dtype="float32").reshape(1, 19, len(region)) / 8
        output.trace = az.from_dict(
            posterior={"x": np.concatenate([draws[:, :13], draws[:, :13] + 100], axis=0)},
            prior={"x": draws},
            coords={"region": region},
            dims={"x": ["region"]},
        )
        recorded = []
        original = make_outputs.reconstruct_flux_stats

        def record_reconstruction(basis, flux, stats, **kwargs):
            recorded.append((stats, kwargs.get("_chunks")))
            return original(basis, flux, stats, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(make_outputs, "reconstruct_flux_stats", record_reconstruction)
            products[domain] = make_outputs.make_flux_outputs(
                output, stats=["mean", "stdev", "quantiles"], report_flux_on_inversion_grid=False
            )
        assert len(recorded) == 1
        assert "draw" not in recorded[0][0].dims
        assert "lat" not in recorded[0][0].dims
        assert recorded[0][1] == {"lat": 2, "lon": 3, "flux_time": 1}
        xr.testing.assert_identical(products[domain].lat, basis.flux.lat)
        xr.testing.assert_identical(products[domain].lon, basis.flux.lon)
        assert output.domain == domain
        mean = draws[0, :13].mean(axis=0)
        expected = (
            reconstruct_flux_stats(
                basis,
                basis.flux,
                xr.Dataset({"expected": ("region", mean)}, coords={"region": region}),
                report_flux_on_inversion_grid=False,
            )
            .compute()
            .map(to_dense, keep_attrs=True)
        )
        np.testing.assert_array_equal(products[domain].flux_posterior_mean, expected.expected)
        if domain == "outer":
            masked = products[domain].flux_posterior_mean.where(basis.flux.overlap_mask)
            assert bool((masked.fillna(0) == 0).all())
    assert products["outer"].sizes["lat"] != products["inner"].sizes["lat"]
    assert products["outer"].sizes["lon"] != products["inner"].sizes["lon"]


def test_future_multisector_nested_shape_uses_independent_common_views(multisector_postprocessing_inv_out):
    """Synthetic future combination only; there is no multisector nested runner."""
    for domain in ("outer", "inner"):
        native = _domain_basis(domain)
        flat = native.operator.basis_flat
        counts = len(native.operator.basis_matrix.region), 1
        flux = xr.concat(
            [native.flux, native.flux], dim=xr.IndexVariable("source", ["ocean-inventory", "ff-inventory"])
        )
        basis = BasisFunctions.from_multi_source_flat_basis(
            {"ff-inventory": flat, "ocean-inventory": xr.ones_like(flat)},
            flux,
            operator_kwargs={"state_dim": "region"},
        )
        output = multisector_postprocessing_inv_out(basis)
        output.model_metadata["domain"] = domain
        draws = np.arange(19, dtype="float32")
        x_ff = np.broadcast_to(draws[:, None], (19, counts[0])).copy()
        x_ocean = (20 - draws)[:, None]
        output.trace = az.from_dict(
            posterior={"x_ff": x_ff[None], "x_ocean": x_ocean[None]},
            prior={"x_ff": x_ff[None, :13], "x_ocean": x_ocean[None, :13]},
            coords={"state_ff": np.arange(counts[0])[::-1], "state_ocean": [0]},
            dims={"x_ff": ["state_ff"], "x_ocean": ["state_ocean"]},
        )
        products = make_outputs.make_flux_outputs(
            output, stats=["mean", "stdev", "quantiles"], report_flux_on_inversion_grid=False
        )
        xr.testing.assert_identical(products.lat, native.flux.lat)
        xr.testing.assert_identical(products.lon, native.flux.lon)
        assert output.domain == domain
        np.testing.assert_array_equal(products.flux_total_posterior_stdev, 0)
        expected = (native.flux * np.float32(20)).transpose(*products.flux_total_posterior_mean.dims)
        np.testing.assert_array_equal(products.flux_total_posterior_mean, expected.compute())
        assert float(products.flux_ff_posterior_stdev.max()) > 0
