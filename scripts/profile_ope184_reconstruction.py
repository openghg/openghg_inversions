#!/usr/bin/env python3
"""Compare source-preserving retained-state reconstruction strategies."""

from __future__ import annotations

import argparse
import json
import resource
import time

import dask
import numpy as np
import xarray as xr
from sparse import SparseArray

from openghg_inversions.basis.operators import BucketBasisOperator, MultiSourceBucketBasisOperator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("single", "shared", "ragged"))
    parser.add_argument("strategy", choices=("slices", "per_source", "expanded", "baseline"))
    args = parser.parse_args()

    grid = {"lat": np.arange(48), "lon": np.arange(64)}
    cells = np.arange(48 * 64).reshape(48, 64)
    labels = ("zeta", "alpha", "mu")
    if args.case == "ragged":
        op = MultiSourceBucketBasisOperator(
            {source: xr.DataArray((np.roll(cells, i, axis=1) % count) + 1,
                                  dims=("lat", "lon"), coords=grid)
             for i, (source, count) in enumerate(zip(labels, (7, 13, 5), strict=True))},
            chunks={"lat": 24, "lon": 32},
        )
    else:
        op = BucketBasisOperator(
            xr.DataArray(cells % 9 + 1, dims=("lat", "lon"), coords=grid),
            chunks={"lat": 24, "lon": 32},
        )
    state = xr.DataArray(
        np.arange(op.basis_matrix.sizes["state"] * 2 * 20, dtype=np.float32).reshape(-1, 2, 20) / 100,
        dims=("state", "chain", "draw"),
        coords={"state": op.basis_matrix.state, "chain": range(2), "draw": range(20)},
    ).chunk({"state": -1, "chain": 1, "draw": 10})
    flux = xr.DataArray(
        np.ones((3, 48, 64), dtype=np.float32),
        dims=("source", "lat", "lon"), coords={"source": list(labels), **grid},
    ).chunk({"source": 1, "lat": 24, "lon": 32})
    matrix = op.basis_matrix

    def calculate() -> xr.DataArray:
        if args.case != "ragged":
            native = xr.dot(matrix, state, dim="state")
            return native if args.case == "single" else native * flux
        assert isinstance(op, MultiSourceBucketBasisOperator)
        if args.strategy == "baseline":
            return op.interpolate(state, weights=flux)
        if args.strategy == "expanded":
            layout = flux.rename(source="native_source")
            expanded = op._native_prolongation(layout, native_dims=("native_source", "lat", "lon"))
            return xr.dot(expanded, state, dim="state") * layout
        pieces = []
        for source in labels:
            positions = np.flatnonzero(matrix.source.values == source)
            if args.strategy == "slices":
                part = xr.dot(matrix.isel(state=positions), state.isel(state=positions), dim="state")
            else:
                local = op.operator_for_source(source, state_dim="state")
                local_state = state.isel(state=positions).reset_index("state", drop=True).assign_coords(
                    state=local.basis_matrix.state
                )
                part = xr.dot(local.basis_matrix, local_state, dim="state")
            pieces.append(part * flux.sel(source=source, drop=True))
        return xr.concat(pieces, dim=xr.IndexVariable("native_source", list(labels)))

    count = 0
    largest_dense = largest_sparse = 0

    def on_task(_key, result, _graph, _state, _worker_id):
        nonlocal largest_dense, largest_sparse
        if isinstance(result, np.ndarray):
            largest_dense = max(largest_dense, result.nbytes)
        elif isinstance(result, SparseArray):
            largest_sparse = max(largest_sparse, result.nbytes)

    def before_task(_key, _graph, _state):
        nonlocal count
        count += 1

    callbacks = dask.callbacks.Callback(pretask=before_task, posttask=on_task)
    if args.case == "ragged" and args.strategy == "baseline":
        graph_seconds = None  # Legacy interpolation computes during the call.
        start = time.perf_counter()
        with callbacks:
            output = calculate()
    else:
        start = time.perf_counter()
        output_graph = calculate()
        graph_seconds = time.perf_counter() - start
        start = time.perf_counter()
        with callbacks:
            output = output_graph.compute()
    if "native_source" in output.dims:
        total = output.sum("native_source")
    elif "source" in output.dims:
        total = output.sum("source")
    else:
        total = output
    execution_seconds = time.perf_counter() - start
    baseline = (
        output if args.strategy == "baseline" else op.interpolate(state, weights=flux)
    ) if args.case == "ragged" else None
    if baseline is not None:
        xr.testing.assert_allclose(total.transpose(*baseline.dims), baseline)
    print(json.dumps({
        "case": args.case, "strategy": args.strategy,
        "graph_seconds": None if graph_seconds is None else round(graph_seconds, 3),
        "execution_seconds": round(execution_seconds, 3),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "dask_tasks": count,
        "largest_dense_chunk_mib": round(largest_dense / 2**20, 3),
        "largest_sparse_chunk_mib": round(largest_sparse / 2**20, 3),
        "shape": dict(output.sizes),
    }))


if __name__ == "__main__":
    main()
