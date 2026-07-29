"""Compile-only canary for the score-regularized conditional-flow loss.

This script exercises one exact 64-row score-loss value-and-parameter-gradient
evaluation for either frozen tiny-domain flow branch.  It does not fit a
model or evaluate scientific thresholds.  Its purpose is to detect CPU/XLA
mixed-derivative compilation failures before launching the complete N1 array.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from typing import Any

CPU_XLA_FLAGS = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 "
    "--xla_cpu_parallel_codegen_split_count=1"
)

import equinox as eqx  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import paramax  # noqa: E402

from examples.rjmcmc import score_regularized_flow_tiny_screen as screen  # noqa: E402
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_training import (  # noqa: E402
    RawLogMassScoreLoss,
    make_score_regularized_conditional_flow,
)

SCHEMA = "rjmcmc-score-regularized-flow-compile-canary-v1"
DIMENSIONS = (1, 3)
MICROBATCH_SIZE = 64


def _canonical_json(payload: object) -> str:
    """Return strict canonical JSON."""
    return json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(payload: object) -> str:
    """Return SHA-256 over strict canonical JSON bytes."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _write_exclusive(path: Path, payload: object) -> None:
    """Create one canonical JSON file without replacing existing evidence."""
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _script_sha256() -> str:
    """Return the byte identity of this committed canary driver."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _runtime_versions() -> dict[str, str]:
    """Return the runtime package identity relevant to compilation."""
    return {
        package: importlib.metadata.version(package)
        for package in ("equinox", "flowjax", "jax", "jaxlib", "paramax")
    }


def _finite_gradient_summary(gradients: Any) -> tuple[int, int, float]:
    """Validate gradient leaves and return leaf/element/max-absolute summaries."""
    leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(gradients)
        if eqx.is_inexact_array(leaf)
    ]
    if not leaves:
        raise RuntimeError("the compile canary produced no differentiable gradients.")
    if any(leaf.dtype != jnp.dtype(jnp.float64) for leaf in leaves):
        raise RuntimeError("the compile canary gradients are not all float64.")
    if any(not bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves):
        raise RuntimeError("the compile canary produced a non-finite gradient.")
    element_count = sum(int(leaf.size) for leaf in leaves)
    maximum = max(float(jnp.max(jnp.abs(leaf))) for leaf in leaves)
    return len(leaves), element_count, maximum


def run_canary(
    *,
    dimension: int,
    source_git_revision: str,
) -> dict[str, Any]:
    """Compile and execute one frozen-microbatch composite gradient."""
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}.")
    if len(source_git_revision) != 40 or any(
        character not in "0123456789abcdef"
        for character in source_git_revision
    ):
        raise ValueError("source_git_revision must be a lowercase full Git SHA.")
    if screen.CPU_XLA_FLAGS != CPU_XLA_FLAGS:
        raise RuntimeError("the canary and development CPU controls disagree.")
    screen._validate_development_protocol()

    flow = make_score_regularized_conditional_flow(
        dimension,
        source_seed=8_291 + dimension,
    )
    params, static = eqx.partition(
        flow,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    projected = jnp.linspace(
        -1.75,
        1.75,
        MICROBATCH_SIZE * dimension,
        dtype=jnp.float64,
    ).reshape(MICROBATCH_SIZE, dimension)
    raw_log_mass = jnp.linspace(
        -0.7,
        0.7,
        MICROBATCH_SIZE,
        dtype=jnp.float64,
    )
    target_score = jnp.sin(raw_log_mass) - 0.1 * jnp.sum(projected, axis=1)
    loss = RawLogMassScoreLoss(
        condition_center=0.0,
        condition_scale=1.0,
    )
    loss_and_gradient = eqx.filter_jit(eqx.filter_value_and_grad(loss))

    started = time.perf_counter()
    value, gradients = loss_and_gradient(
        params,
        static,
        projected,
        raw_log_mass,
        target_score,
        key=jr.key(97 + dimension),
    )
    value, gradients = jax.block_until_ready((value, gradients))
    elapsed = time.perf_counter() - started
    loss_value = float(value)
    if not math.isfinite(loss_value):
        raise RuntimeError("the compile canary loss is not finite.")
    gradient_leaves, gradient_elements, gradient_maximum = (
        _finite_gradient_summary(gradients)
    )
    return {
        "schema": SCHEMA,
        "source_git_revision": source_git_revision,
        "driver_sha256": _script_sha256(),
        "development_protocol_sha256": screen.DEVELOPMENT_PROTOCOL_SHA256,
        "cpu_xla_flags": CPU_XLA_FLAGS,
        "runtime": _runtime_versions(),
        "dimension": dimension,
        "flow_branch": (
            "masked-autoregressive" if dimension == 1 else "coupling"
        ),
        "score_microbatch_size": MICROBATCH_SIZE,
        "loss": loss_value,
        "gradient_leaf_count": gradient_leaves,
        "gradient_element_count": gradient_elements,
        "maximum_absolute_gradient": gradient_maximum,
        "compile_and_execute_seconds": elapsed,
        "compile_pass": True,
        "scientific_thresholds_evaluated": False,
    }


def _publish(
    output_directory: Path,
    payload: dict[str, Any],
) -> tuple[Path, Path]:
    """Publish the report and then its completion marker create-only."""
    output_directory.mkdir(parents=True, exist_ok=True)
    report = output_directory / f"q{payload['dimension']}.report.json"
    marker = output_directory / f"q{payload['dimension']}.complete.json"
    if report.exists() or marker.exists():
        raise FileExistsError("refusing to replace existing compile-canary evidence.")
    envelope = {
        "payload": payload,
        "sha256": _sha256_json(payload),
    }
    _write_exclusive(report, envelope)
    marker_payload = {
        "schema": "rjmcmc-score-regularized-flow-compile-canary-complete-v1",
        "report": report.name,
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }
    _write_exclusive(marker, marker_payload)
    return report, marker


def main() -> None:
    """Run the requested compile canary and publish its evidence."""
    if os.environ.get("XLA_FLAGS") != CPU_XLA_FLAGS:
        raise RuntimeError(
            "XLA_FLAGS does not match the frozen CPU compilation control."
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, choices=DIMENSIONS, required=True)
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    payload = run_canary(
        dimension=arguments.dimension,
        source_git_revision=arguments.source_git_revision,
    )
    report, marker = _publish(arguments.output_directory, payload)
    print(
        _canonical_json(
            {
                "report": str(report),
                "completion_marker": str(marker),
                "payload": payload,
            }
        )
    )


if __name__ == "__main__":
    main()
