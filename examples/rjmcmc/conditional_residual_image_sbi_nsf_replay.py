#!/usr/bin/env python
"""Authenticate and operationally replay one conditional sbi-NSF artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_sbi_nsf import (
    ConditionalResidualImageSbiNsf,
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
    artifact = ConditionalResidualImageSbiNsf.from_bytes(
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
    condition = torch.as_tensor(
        artifact.conditioner(masses)[None, :],
        dtype=torch.float64,
    )
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(1_019)
            standardized, sampled_log_prob = artifact.model.sample_and_log_prob(
                torch.Size([64]),
                condition=condition,
            )
        with torch.no_grad():
            separate_log_prob = artifact.model.log_prob(
                standardized,
                condition=condition,
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    round_trip_error = float(
        torch.max(torch.abs(sampled_log_prob - separate_log_prob))
    )
    if round_trip_error > 1.0e-6:
        raise RuntimeError("NSF sample/log-probability round trip exceeded tolerance")
    first = artifact.sample_observation(
        masses,
        sample_count=8,
        source_seed=2_147,
    )
    second = artifact.sample_observation(
        masses,
        sample_count=8,
        source_seed=2_147,
    )
    if not np.array_equal(first, second):
        raise RuntimeError("artifact simulator did not replay deterministically")
    print(
        json.dumps(
            {
                "artifact_sha256": artifact.artifact_sha256,
                "canonical_replay": artifact.to_bytes() == serialized,
                "gradient": gradient.tolist(),
                "log_likelihood": value,
                "sample_log_prob_round_trip_maximum_error": round_trip_error,
                "sample_sha256": hashlib.sha256(
                    np.ascontiguousarray(first, dtype="<f8").tobytes(order="C")
                ).hexdigest(),
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
