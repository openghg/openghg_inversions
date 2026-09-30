#!/usr/bin/env python3
"""Reproduce OPE-169 artifact sizes and process peak memory on a labelled fixture."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import xarray as xr

from openghg_inversions.basis import AffineFluxMap
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact, load, save
from openghg_inversions.basis.operators import BucketBasisOperator


def fixture(representation: str) -> AffineFluxMapArtifact:
    """Build a 72 x 96 grid, 96 states, two times, and 256 posterior samples."""
    lat = np.linspace(40, 60, 72)
    lon = np.linspace(-12, 16, 96)
    coords = {"lat": lat, "lon": lon}
    cells = np.arange(72 * 96).reshape(72, 96)
    basis = xr.DataArray(cells % 96, dims=("lat", "lon"), coords=coords)
    operator = BucketBasisOperator(basis, state_dim="state", chunks={"lat": 24, "lon": 24})
    native_mean = xr.DataArray(
        1 + 0.1 * np.sin(cells / 53),
        dims=("lat", "lon"),
        coords=coords,
        attrs={"units": "1"},
    ).chunk({"lat": 24, "lon": 24})
    flux = xr.DataArray(
        np.stack([1 + 0.2 * np.cos(cells / 37), -0.8 - 0.1 * np.sin(cells / 41)]),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], **coords},
        attrs={"units": "mol m-2 s-1"},
    ).chunk({"time": 1, "lat": 24, "lon": 24})
    if representation == "bucket":
        prolongation = operator
    else:
        # Explicit supplied-restriction fixture: genuinely dense, not a bucket alias.
        states = np.arange(96)
        values = (cells[..., None] % 96 == states).astype(np.float32)
        values += 0.001 * np.sin((cells[..., None] + states) / 17)
        prolongation = xr.DataArray(
            values,
            dims=("lat", "lon", "state"),
            coords={**coords, "state": operator.basis_matrix.state},
            attrs={"units": "1"},
        ).chunk({"lat": 24, "lon": 24, "state": 96})
    assert isinstance(native_mean.data, da.Array)
    assert isinstance(flux.data, da.Array)
    assert isinstance(
        (prolongation if isinstance(prolongation, xr.DataArray) else prolongation.basis_matrix).data,
        da.Array,
    )
    return AffineFluxMapArtifact(
        AffineFluxMap(native_mean, flux, prolongation, "state"),
        prepared_inputs_id="ope169-memory-fixture-v1",
        projection_provenance={"fixture": "bucket" if representation == "bucket" else "supplied"},
        reconstruction_provenance={"seed": 169, "grid": [72, 96]},
    )


def samples(affine_map: AffineFluxMap) -> tuple[xr.DataArray, xr.DataArray]:
    state = (
        affine_map.prolongation
        if isinstance(affine_map.prolongation, xr.DataArray)
        else affine_map.prolongation.basis_matrix
    ).state
    reference = xr.DataArray(
        0.8 + 0.1 * np.sin(np.arange(96)),
        dims="state",
        coords={"state": state},
        attrs={"units": "1"},
    )
    draws = xr.DataArray(
        1 + 0.2 * np.random.default_rng(169).standard_normal((2, 128, 96)),
        dims=("chain", "draw", "state"),
        coords={"chain": [0, 1], "draw": np.arange(128), "state": state},
        attrs={"units": "1"},
    ).chunk({"chain": 1, "draw": 32, "state": 96})
    return reference, draws


def aggregate(affine_map: AffineFluxMap) -> xr.DataArray:
    """Contract four country-like functionals before applying sample axes."""
    lat = affine_map.native_mean.lat
    lon = affine_map.native_mean.lon
    membership = xr.DataArray(
        np.stack(
            [
                ((np.arange(72)[:, None] >= 36) == (i >= 2))
                & ((np.arange(96)[None, :] >= 48) == (i % 2 == 1))
                for i in range(4)
            ]
        ).astype(float),
        dims=("country", "lat", "lon"),
        coords={"country": ["NW", "NE", "SW", "SE"], "lat": lat, "lon": lon},
    )
    area = xr.DataArray(np.cos(np.deg2rad(lat)), dims="lat", coords={"lat": lat})
    weights = membership * area
    native_dims = ("lat", "lon")
    prolongation = (
        affine_map.prolongation
        if isinstance(affine_map.prolongation, xr.DataArray)
        else affine_map.prolongation.basis_matrix
    )
    reference_country = (weights * affine_map.flux * affine_map.native_mean).sum(native_dims)
    action = xr.dot(weights * affine_map.flux, prolongation, dim=native_dims)
    assert set(action.dims) == {"country", "time", "state"}
    # Named compact-action boundary: materialize only country/time/state products.
    reference_country, action = dask.compute(reference_country, action)
    reference, draws = samples(affine_map)
    # Named aggregate-output boundary; no native-grid-by-sample intermediate.
    aggregate_output = reference_country + xr.dot(action, draws - reference, dim="state")
    assert isinstance(aggregate_output.data, da.Array)
    return aggregate_output.compute()


def worker(path: Path, operation: str) -> None:
    # Each operation has a fresh fork; ru_maxrss starts at its current RSS.
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    artifact = load(path)  # Named load boundary in the persistence API.
    if operation == "load":
        result = artifact.affine_map.native_mean
    elif operation == "grid_apply":
        reference, draws = samples(artifact.affine_map)
        # Named requested-grid-output boundary.
        grid_output = artifact.affine_map.state_to_flux(draws, reference_state=reference)
        assert isinstance(grid_output.data, da.Array)
        result = grid_output.compute()
    else:
        result = aggregate(artifact.affine_map)
    assert result.size > 0 and np.isfinite(result.data).all()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "operation": operation,
                "baseline_rss_mib": round(before / 1024, 1),
                "peak_rss_mib": round(peak / 1024, 1),
                "peak_above_baseline_mib": round((peak - before) / 1024, 1),
                "output_shape": list(result.shape),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("nc", "zarr"), default="nc")
    parser.add_argument("--worker", nargs=2, metavar=("PATH", "OPERATION"))
    args = parser.parse_args()
    if args.worker:
        pid = os.fork()
        if pid:
            _, status = os.waitpid(pid, 0)
            if status:
                raise RuntimeError(f"Benchmark worker exited with status {status}.")
        else:
            worker(Path(args.worker[0]), args.worker[1])
        return
    for representation in ("bucket", "explicit"):
        with tempfile.TemporaryDirectory(prefix="ope169-memory-") as directory:
            path = Path(directory) / f"{representation}.{args.format}"
            save(fixture(representation), path)  # Named serialization boundary.
            artifact_bytes = (
                sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                if path.is_dir()
                else path.stat().st_size
            )
            measurements = []
            for operation in ("load", "grid_apply", "aggregate_contraction"):
                proc = subprocess.run(
                    [sys.executable, __file__, "--worker", str(path), operation],
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                )
                measurements.append(json.loads(proc.stdout))
            print(
                json.dumps(
                    {
                        "representation": representation,
                        "format": args.format,
                        "artifact_bytes": artifact_bytes,
                        "measurements": measurements,
                    }
                )
            )


if __name__ == "__main__":
    main()
