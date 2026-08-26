#!/usr/bin/env python3
"""Instrument the OPE-87 WUR25 producer's real multisource sensitivity calls."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import runpy
import subprocess
import time
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from sparse import SparseArray

from openghg_inversions.array_ops import force_align
from openghg_inversions.basis.basis_functions import BasisFunctions


class FirstSensitivityComplete(Exception):
    """Stop the production builder after its first instrumented sensitivity."""


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _chunks(array: xr.DataArray) -> tuple[tuple[int, ...], ...] | None:
    return getattr(array.data, "chunks", None)


def _sensitivity_pr651(
    basis_functions: BasisFunctions, fp_x_flux: xr.DataArray, *, fillna: bool
) -> xr.DataArray:
    """Run PR 651's exact source-wise implementation as a profile reference."""
    operator = basis_functions.operator
    matrix = force_align(operator.basis_matrix, fp_x_flux, dims=list(operator.meta.grid_dims))
    matrix = matrix.transpose(*operator.meta.grid_dims, ...)
    state_sources = np.asarray(matrix[operator.source_dim].values)
    pieces = []
    for source in operator.source_labels:
        source_flux = fp_x_flux.sel({operator.source_dim: source}, drop=True)
        if fillna:
            source_flux = source_flux.fillna(0.0)
        pieces.append(
            xr.dot(
                source_flux,
                matrix.isel(
                    {operator.meta.state_dim: np.flatnonzero(state_sources == source)}
                ),
                dim=list(operator.meta.grid_dims),
            )
        )
    result = xr.concat(pieces, dim=operator.meta.state_dim).as_numpy()
    return result.transpose(operator.meta.state_dim, "time", ...)


def _sensitivity_native(
    basis_functions: BasisFunctions, fp_x_flux: xr.DataArray, *, fillna: bool
) -> xr.DataArray:
    """Contract source-native input with the sparse block-diagonal prolongation."""
    operator = basis_functions.operator
    native_source_dim = "native_source"
    fp_native = fp_x_flux.rename({operator.source_dim: native_source_dim})
    prolongation = operator.native_prolongation(
        fp_native,
        native_dims=(native_source_dim, *operator.meta.grid_dims),
    )
    if fillna:
        fp_native = fp_native.fillna(0.0)
    result = xr.dot(
        fp_native,
        prolongation,
        dim=(native_source_dim, *operator.meta.grid_dims),
    ).as_numpy()
    return result.transpose(operator.meta.state_dim, "time", ...)


def _compare(actual: xr.DataArray, reference: xr.DataArray) -> dict[str, object]:
    reference = reference.transpose(*actual.dims)
    difference = np.abs(np.asarray(actual) - np.asarray(reference))
    denominator = np.maximum(
        np.abs(np.asarray(reference)), np.finfo(reference.dtype).tiny
    )
    try:
        xr.testing.assert_identical(
            actual.coords.to_dataset(), reference.coords.to_dataset()
        )
        coordinates_identical = True
    except AssertionError:
        coordinates_identical = False
    return {
        "coordinates_identical": coordinates_identical,
        "max_absolute_difference": float(difference.max(initial=0.0)),
        "max_relative_difference": float(
            (difference / denominator).max(initial=0.0)
        ),
    }


def main() -> None:
    original = BasisFunctions.sensitivity
    call_count = 0

    def profiled_sensitivity(
        basis_functions: BasisFunctions,
        fp_x_flux: xr.DataArray,
        fillna: bool = True,
    ) -> xr.DataArray:
        nonlocal call_count
        call_count += 1
        tasks: set[object] = set()
        started = time.perf_counter()
        with dask.callbacks.Callback(pretask=lambda key, _graph, _state: tasks.add(key)):
            output = original(basis_functions, fp_x_flux, fillna=fillna)
        wall_seconds = time.perf_counter() - started

        comparison: dict[str, object] | None = None
        if os.environ.get("OPE121_COMPARE_PR651") == "1":
            reference = _sensitivity_pr651(
                basis_functions, fp_x_flux, fillna=fillna
            )
            comparison = _compare(output, reference)

        native_candidate: dict[str, object] | None = None
        if os.environ.get("OPE121_PROFILE_NATIVE") == "1":
            native_tasks: set[object] = set()
            observed_chunk_bytes = {"dense": 0, "sparse": 0}

            def record_chunk(_key, result, _graph, _state, _worker_id) -> None:
                if isinstance(result, np.ndarray):
                    observed_chunk_bytes["dense"] = max(
                        observed_chunk_bytes["dense"], result.nbytes
                    )
                elif isinstance(result, SparseArray):
                    observed_chunk_bytes["sparse"] = max(
                        observed_chunk_bytes["sparse"], result.nbytes
                    )

            native_started = time.perf_counter()
            with dask.callbacks.Callback(
                pretask=lambda key, _graph, _state: native_tasks.add(key),
                posttask=record_chunk,
            ):
                native_output = _sensitivity_native(
                    basis_functions, fp_x_flux, fillna=fillna
                )
            native_candidate = {
                "wall_seconds": time.perf_counter() - native_started,
                "dask_tasks": len(native_tasks),
                "max_observed_dense_chunk_mib": observed_chunk_bytes["dense"] / 1024**2,
                "max_observed_sparse_chunk_mib": observed_chunk_bytes["sparse"] / 1024**2,
                "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "comparison_to_helper": _compare(native_output, output),
            }

        payload = {
            "call": call_count,
            "ogi_revision": os.environ.get("OGI_REVISION") or _revision(),
            "vg_revision": os.environ.get("VG_REVISION"),
            "script_sha256": _script_sha256(),
            "fp_x_flux_dims": fp_x_flux.dims,
            "fp_x_flux_shape": fp_x_flux.shape,
            "fp_x_flux_dtype": str(fp_x_flux.dtype),
            "fp_x_flux_chunks": _chunks(fp_x_flux),
            "basis_dims": basis_functions.operator.basis_matrix.dims,
            "basis_shape": basis_functions.operator.basis_matrix.shape,
            "basis_chunks": _chunks(basis_functions.operator.basis_matrix),
            "source_labels": list(basis_functions.operator.source_labels),
            "output_dims": output.dims,
            "output_shape": output.shape,
            "output_size_mib": output.nbytes / 1024**2,
            "dask_tasks": len(tasks),
            "wall_seconds": wall_seconds,
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "comparison_to_pr651": comparison,
            "native_candidate": native_candidate,
        }
        print("OPE121_SENSITIVITY_PROFILE=" + json.dumps(payload, sort_keys=True), flush=True)
        if os.environ.get("OPE121_STOP_AFTER_FIRST") == "1":
            raise FirstSensitivityComplete
        return output

    BasisFunctions.sensitivity = profiled_sensitivity
    try:
        runpy.run_module("scripts.build_ope87_wur25_full_mf_prepared_input", run_name="__main__")
    except FirstSensitivityComplete:
        print("OPE121_SINGLE_SITE_GATE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
