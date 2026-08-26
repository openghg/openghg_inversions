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

from openghg_inversions.basis.basis_functions import BasisFunctions
from scripts.profile_ope121_multiindex_operations import sensitivity_pr651


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
            reference = sensitivity_pr651(basis_functions.operator, fp_x_flux)
            reference = reference.transpose(*output.dims)
            difference = np.abs(np.asarray(output) - np.asarray(reference))
            denominator = np.maximum(
                np.abs(np.asarray(reference)), np.finfo(reference.dtype).tiny
            )
            try:
                xr.testing.assert_identical(
                    output.coords.to_dataset(), reference.coords.to_dataset()
                )
                coordinates_identical = True
            except AssertionError:
                coordinates_identical = False
            comparison = {
                "coordinates_identical": coordinates_identical,
                "max_absolute_difference": float(difference.max(initial=0.0)),
                "max_relative_difference": float(
                    (difference / denominator).max(initial=0.0)
                ),
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
