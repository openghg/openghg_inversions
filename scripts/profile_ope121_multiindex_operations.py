#!/usr/bin/env python3
"""Profile the gathered MultiIndex operations investigated by OPE-121."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import dask
import numpy as np
import xarray as xr

from openghg_inversions.array_ops import (
    align_to_multi_index_level_values,
    force_align,
    iter_multi_index_level_slices,
    to_dense,
)
from openghg_inversions.basis.operators import MultiSourceBucketBasisOperator


def make_fixture(args: argparse.Namespace) -> tuple[MultiSourceBucketBasisOperator, xr.DataArray, xr.DataArray, xr.DataArray]:
    rng = np.random.default_rng(args.seed)
    counts = tuple(int(value) for value in args.states.split(","))
    sources = [f"source-{i}" for i in range(len(counts))]
    coords = {"lat": np.arange(args.lat), "lon": np.arange(args.lon)}
    cells = np.arange(args.lat * args.lon).reshape(args.lat, args.lon)
    bases = {
        source: xr.DataArray(
            (np.roll(cells, i, axis=1) % count) + 1,
            dims=("lat", "lon"),
            coords=coords,
        )
        for i, (source, count) in enumerate(zip(sources, counts, strict=True))
    }
    chunks = None if args.execution == "eager" else {"lat": args.lat_chunk, "lon": args.lon_chunk}
    operator = MultiSourceBucketBasisOperator(bases, state_dim="state", chunks=chunks)
    if args.execution == "eager":
        operator._basis_matrix = to_dense(operator.basis_matrix).compute()  # noqa: SLF001

    fp_x_flux = xr.DataArray(
        rng.standard_normal((len(sources), args.time, args.lat, args.lon), dtype=np.float32),
        dims=("source", "time", "lat", "lon"),
        coords={"source": sources, "time": np.arange(args.time), **coords},
    )
    weights = xr.DataArray(
        rng.random((len(sources), args.lat, args.lon), dtype=np.float32),
        dims=("source", "lat", "lon"),
        coords={"source": sources, **coords},
    )
    state = xr.DataArray(
        rng.standard_normal(
            (operator.basis_matrix.sizes["state"], args.chain, args.draw), dtype=np.float32
        ),
        dims=("state", "chain", "draw"),
        coords={"state": operator.basis_matrix["state"], "chain": np.arange(args.chain), "draw": np.arange(args.draw)},
    )
    if args.execution == "dask":
        fp_x_flux = fp_x_flux.chunk(
            {"source": 1, "time": args.time_chunk, "lat": args.lat_chunk, "lon": args.lon_chunk}
        )
        weights = weights.chunk({"source": 1, "lat": args.lat_chunk, "lon": args.lon_chunk})
        state = state.chunk({"state": -1, "chain": 1, "draw": args.draw_chunk})
    return operator, fp_x_flux, weights, state


def sensitivity_broadcast(operator: MultiSourceBucketBasisOperator, fp_x_flux: xr.DataArray) -> xr.DataArray:
    matrix = force_align(operator.basis_matrix, fp_x_flux, dims=list(operator.meta.grid_dims))
    matrix = matrix.transpose(*operator.meta.grid_dims, ...)
    fp_on_state = align_to_multi_index_level_values(
        fp_x_flux,
        multi_index=matrix[operator.meta.state_dim],
        multi_dim=operator.meta.state_dim,
        level=operator.source_dim,
        other_dim=operator.source_dim,
    )
    result = xr.dot(fp_on_state.fillna(0.0), matrix, dim=list(operator.meta.grid_dims)).as_numpy()
    return result.transpose(operator.meta.state_dim, "time", ...)


def sensitivity_pr651(operator: MultiSourceBucketBasisOperator, fp_x_flux: xr.DataArray) -> xr.DataArray:
    matrix = force_align(operator.basis_matrix, fp_x_flux, dims=list(operator.meta.grid_dims))
    matrix = matrix.transpose(*operator.meta.grid_dims, ...)
    state_sources = np.asarray(matrix[operator.source_dim].values)
    pieces = [
        xr.dot(
            fp_x_flux.sel({operator.source_dim: source}, drop=True).fillna(0.0),
            matrix.isel({operator.meta.state_dim: np.flatnonzero(state_sources == source)}),
            dim=list(operator.meta.grid_dims),
        )
        for source in operator.source_labels
    ]
    result = xr.concat(pieces, dim=operator.meta.state_dim).as_numpy()
    return result.transpose(operator.meta.state_dim, "time", ...)


def interpolation_sourcewise(
    operator: MultiSourceBucketBasisOperator,
    state: xr.DataArray,
    weights: xr.DataArray,
) -> xr.DataArray:
    weights = force_align(weights, operator.basis_matrix, dims=list(operator.meta.grid_dims))
    contributions = []
    for _, positions, source_weights in iter_multi_index_level_slices(
        weights,
        multi_index=operator.basis_matrix[operator.meta.state_dim],
        multi_dim=operator.meta.state_dim,
        level=operator.source_dim,
        array_dim=operator.source_dim,
    ):
        source_field = xr.dot(
            operator.basis_matrix.isel({operator.meta.state_dim: positions}),
            state.isel({operator.meta.state_dim: positions}),
            dim=operator.meta.state_dim,
        )
        contributions.append(source_field * source_weights)
    return sum(contributions[1:], contributions[0]).as_numpy().transpose(*operator.meta.grid_dims, ...)


def compare(actual: xr.DataArray, expected: xr.DataArray) -> dict[str, object]:
    expected = expected.transpose(*actual.dims)
    difference = np.abs(np.asarray(actual) - np.asarray(expected))
    denominator = np.maximum(np.abs(np.asarray(expected)), np.finfo(expected.dtype).tiny)
    try:
        xr.testing.assert_identical(actual.coords.to_dataset(), expected.coords.to_dataset())
        coordinates_identical = True
    except AssertionError:
        coordinates_identical = False
    return {
        "coordinates_identical": coordinates_identical,
        "max_absolute_difference": float(difference.max(initial=0.0)),
        "max_relative_difference": float((difference / denominator).max(initial=0.0)),
    }


def revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=("sensitivity-broadcast", "sensitivity-pr651", "sensitivity-helper", "interpolation-current", "interpolation-sourcewise", "sensitivity-shared", "interpolation-shared"),
    )
    parser.add_argument("--execution", choices=("eager", "dask"), default="eager")
    parser.add_argument("--states", default="12,15,18,8")
    parser.add_argument("--time", type=int, default=24)
    parser.add_argument("--lat", type=int, default=80)
    parser.add_argument("--lon", type=int, default=100)
    parser.add_argument("--chain", type=int, default=2)
    parser.add_argument("--draw", type=int, default=40)
    parser.add_argument("--lat-chunk", type=int, default=40)
    parser.add_argument("--lon-chunk", type=int, default=50)
    parser.add_argument("--time-chunk", type=int, default=8)
    parser.add_argument("--draw-chunk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    operator, fp_x_flux, weights, state = make_fixture(args)
    tasks: set[object] = set()
    callbacks = dask.callbacks.Callback(pretask=lambda key, _graph, _state: tasks.add(key))
    operations: dict[str, Callable[[], xr.DataArray]] = {
        "sensitivity-broadcast": lambda: sensitivity_broadcast(operator, fp_x_flux),
        "sensitivity-pr651": lambda: sensitivity_pr651(operator, fp_x_flux),
        "sensitivity-helper": lambda: operator.sensitivity(fp_x_flux),
        "interpolation-current": lambda: operator.interpolate(state, weights=weights),
        "interpolation-sourcewise": lambda: interpolation_sourcewise(operator, state, weights),
        "sensitivity-shared": lambda: operator.sensitivity(fp_x_flux.sum("source")),
        "interpolation-shared": lambda: operator.interpolate(state, weights=weights.sum("source")),
    }
    started = time.perf_counter()
    with callbacks:
        output = operations[args.operation]()
    wall_seconds = time.perf_counter() - started
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if args.operation in {"sensitivity-shared", "interpolation-shared"}:
        reference = output
    elif args.operation.startswith("sensitivity") and args.operation != "sensitivity-broadcast":
        reference = sensitivity_broadcast(operator, fp_x_flux)
    elif args.operation == "sensitivity-broadcast":
        reference = operator.sensitivity(fp_x_flux)
    elif args.operation == "interpolation-sourcewise":
        reference = operator.interpolate(state, weights=weights)
    elif args.operation == "interpolation-current":
        reference = interpolation_sourcewise(operator, state, weights)
    else:
        reference = output

    print(
        json.dumps(
            {
                "ogi_revision": revision(),
                "script_sha256": script_sha256(),
                "operation": args.operation,
                "execution": args.execution,
                "dtype": str(fp_x_flux.dtype),
                "dimensions": {
                    "source": fp_x_flux.sizes.get("source", 0),
                    "states_per_source": args.states,
                    "state": operator.basis_matrix.sizes["state"],
                    "time": args.time,
                    "lat": args.lat,
                    "lon": args.lon,
                    "chain": args.chain,
                    "draw": args.draw,
                },
                "chunks": {name: getattr(value.data, "chunks", None) for name, value in {"basis": operator.basis_matrix, "fp_x_flux": fp_x_flux, "weights": weights, "state": state}.items()},
                "wall_seconds": wall_seconds,
                "peak_rss_mib": peak_rss_kib / 1024,
                "dask_tasks": len(tasks),
                "broadcast_intermediate_shape": [
                    operator.basis_matrix.sizes["state"],
                    args.time,
                    args.lat,
                    args.lon,
                ],
                "broadcast_intermediate_size_mib": (
                    operator.basis_matrix.sizes["state"]
                    * args.time
                    * args.lat
                    * args.lon
                    * fp_x_flux.dtype.itemsize
                    / 1024**2
                ),
                "output_shape": output.shape,
                "output_size_mib": output.nbytes / 1024**2,
                "comparison": compare(output, reference),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
