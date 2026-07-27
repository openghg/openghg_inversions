#!/usr/bin/env python
"""Authenticate and operationally replay one native-quadrature artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openghg_inversions.experimental.rjmcmc.aggregation_error_native_quadrature import (
    ConditionalNativeQuadrature,
)


def main() -> None:
    """Replay bytes, likelihood, gradient, and deterministic sampling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    serialized = args.artifact.read_bytes()
    if hashlib.sha256(serialized).hexdigest() != args.expected_sha256:
        raise ValueError("artifact file digest does not match expected SHA-256")
    artifact = ConditionalNativeQuadrature.from_bytes(
        serialized,
        expected_sha256=args.expected_sha256,
    )
    masses = np.ones(artifact.region_count, dtype=np.float64)
    observation = np.zeros(
        artifact.context.observation_count,
        dtype=np.float64,
    )
    value, gradient = artifact.log_likelihood_and_mass_gradient(
        observation,
        masses,
    )
    first, first_indices = artifact.sample_with_component_indices(
        masses,
        sample_count=64,
        rng=np.random.default_rng(2_147),
    )
    second, second_indices = artifact.sample_with_component_indices(
        masses,
        sample_count=64,
        rng=np.random.default_rng(2_147),
    )
    if not np.array_equal(first, second) or not np.array_equal(
        first_indices,
        second_indices,
    ):
        raise RuntimeError("artifact simulator did not replay deterministically")
    print(
        json.dumps(
            {
                "artifact_sha256": artifact.artifact_sha256,
                "canonical_replay": artifact.to_bytes() == serialized,
                "gradient": gradient.tolist(),
                "log_likelihood": value,
                "sample_sha256": hashlib.sha256(
                    np.ascontiguousarray(first, dtype="<f8").tobytes(order="C")
                ).hexdigest(),
                "component_indices_sha256": hashlib.sha256(
                    np.ascontiguousarray(
                        first_indices,
                        dtype="<i8",
                    ).tobytes(order="C")
                ).hexdigest(),
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
