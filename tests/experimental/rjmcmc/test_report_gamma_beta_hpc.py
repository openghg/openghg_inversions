"""Focused tests for the Gamma--Beta Stage-3 HPC postprocessor."""

from __future__ import annotations

import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.experimental.rjmcmc.gamma_beta_io import (
    GAMMA_BETA_TRACE_SCHEMA_ID,
)

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "rjmcmc"
    / "report_gamma_beta_hpc.py"
)


@pytest.fixture(scope="module")
def report_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "report_gamma_beta_hpc_example",
        EXAMPLE_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load the Gamma-Beta Stage-3 postprocessor.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _trace(problem_sha: str = "p" * 64) -> xr.Dataset:
    attempts = 1_400
    draws = 2
    move = np.resize(
        np.asarray(
            (
                "split",
                "merge",
                "root_refresh",
                "fraction_refresh",
                *(("fixed_coefficient",) * 6),
            )
        ),
        attempts,
    )
    coefficient = np.full(attempts, -1, dtype=np.int64)
    fixed_positions = np.flatnonzero(move == "fixed_coefficient")
    coefficient[fixed_positions] = np.arange(fixed_positions.size) % 6
    return xr.Dataset(
        {
            "k": ("draw", np.asarray([5, 5], dtype=np.int64)),
            "frontier_node_id": (
                ("draw", "region_slot"),
                np.asarray([[0], [0]], dtype=np.int64),
            ),
            "frontier_active": (
                ("draw", "region_slot"),
                np.asarray([[True], [True]]),
            ),
            "split_node_id": (
                ("draw", "split_slot"),
                np.asarray([[-1], [-1]], dtype=np.int64),
            ),
            "split_fraction": (
                ("draw", "split_slot"),
                np.asarray([[np.nan], [np.nan]]),
            ),
            "split_active": (
                ("draw", "split_slot"),
                np.asarray([[False], [False]]),
            ),
            "root_total": ("draw", np.ones(draws)),
            "fixed_coefficients": (
                ("draw", "fixed_parameter"),
                np.ones((draws, 6)),
            ),
            "log_gaussian_likelihood": ("draw", np.full(draws, -10.0)),
            "log_likelihood": ("draw", np.full(draws, -10.0)),
            "log_root_prior": ("draw", np.full(draws, -1.0)),
            "log_fraction_prior": ("draw", np.full(draws, -1.0)),
            "log_partition_prior": ("draw", np.full(draws, -1.0)),
            "log_fixed_coefficient_prior": ("draw", np.full(draws, -1.0)),
            "log_target": ("draw", np.full(draws, -14.0)),
            "slot": ("attempt", np.full(attempts, "structural")),
            "move": ("attempt", move),
            "valid": ("attempt", np.ones(attempts, dtype=np.bool_)),
            "accepted": ("attempt", np.zeros(attempts, dtype=np.bool_)),
            "node_id": ("attempt", np.full(attempts, -1, dtype=np.int64)),
            "coefficient_id": ("attempt", coefficient),
            "k_before": ("attempt", np.full(attempts, 5, dtype=np.int64)),
            "k_after": ("attempt", np.full(attempts, 5, dtype=np.int64)),
            "log_acceptance_ratio": ("attempt", np.zeros(attempts)),
        },
        coords={
            "draw": np.arange(draws),
            "state_transition": ("draw", np.asarray([2_800, 2_814])),
            "attempt": np.arange(attempts),
            "global_transition": ("attempt", np.arange(1, attempts + 1)),
            "region_slot": [0],
            "split_slot": [0],
            "fixed_parameter": [f"outer_{position}" for position in range(6)],
        },
        attrs={
            "schema_id": GAMMA_BETA_TRACE_SCHEMA_ID,
            "problem_sha256": problem_sha,
        },
    )


def test_completion_hash_validation_rejects_changed_artifact(
    tmp_path: Path,
    report_module: ModuleType,
) -> None:
    directory = tmp_path / "segment"
    directory.mkdir()
    hashes = {}
    for name in report_module.ARTIFACT_NAMES:
        payload = f"payload:{name}".encode()
        (directory / name).write_bytes(payload)
        hashes[name] = _digest(payload)
    (directory / "complete.json").write_text(
        json.dumps(
            {
                "checkpoint": "checkpoint.npz",
                "sha256": hashes,
            }
        ),
        encoding="utf-8",
    )

    assert report_module._validate_completion(directory)["sha256"] == hashes
    (directory / "trace.nc").write_bytes(b"changed")
    with pytest.raises(ValueError, match="trace.nc"):
        report_module._validate_completion(directory)


def test_exact_stage3_layout_requires_four_by_ten(
    tmp_path: Path,
    report_module: ModuleType,
) -> None:
    for chain in range(4):
        for segment in range(10):
            directory = tmp_path / f"chain_{chain}" / f"segment_{segment:03d}"
            directory.mkdir(parents=True)
            (directory / "complete.json").write_text("{}", encoding="utf-8")

    report_module._validate_exact_layout(tmp_path)
    (tmp_path / "chain_3" / "segment_009" / "complete.json").unlink()
    with pytest.raises(ValueError, match="four chains x ten segments"):
        report_module._validate_exact_layout(tmp_path)


def test_trace_validation_and_attempt_counts(
    report_module: ModuleType,
) -> None:
    trace = _trace()
    manifest = {
        "problem_sha256": "p" * 64,
        "likelihood": {"power": 1.0},
    }

    report_module._validate_trace(trace, chain=0, segment=0, manifest=manifest)
    summary = report_module._attempt_summary(trace)
    assert sum(value["attempts"] for value in summary["moves"].values()) == 1_400
    assert sum(
        value["attempts"] for value in summary["fixed_coefficients"].values()
    ) == int(np.count_nonzero(trace["move"].values == "fixed_coefficient"))

    broken = trace.copy(deep=True)
    broken["accepted"].values[0] = True
    broken["valid"].values[0] = False
    with pytest.raises(ValueError, match="accepted invalid"):
        report_module._validate_trace(
            broken,
            chain=0,
            segment=0,
            manifest=manifest,
        )


def test_edge_flow_reversal_and_first_passage_definitions(
    report_module: ModuleType,
) -> None:
    attempts = {
        "global_transition": np.asarray([1, 2, 3, 4]),
        "move": np.asarray(["split", "merge", "split", "split"]),
        "valid": np.asarray([True, True, True, True]),
        "accepted": np.asarray([True, True, True, True]),
        "node_id": np.asarray([7, 7, 9, 10]),
        "coefficient_id": np.full(4, -1),
        "k_before": np.asarray([5, 6, 5, 6]),
        "k_after": np.asarray([6, 5, 6, 7]),
    }

    flow = report_module._edge_flow(attempts)
    assert flow[0]["realized_bidirectional_flow"] == 1
    reversals = report_module._immediate_reversals(attempts)
    assert reversals["adjacent_atomic_opposite_direction_reversals"] == 2
    assert reversals["exact_node_reversals"] == 1
    passage = report_module._first_passage(attempts, 5)
    assert passage["first_transition_by_absolute_k_distance"]["1"] == 1
    assert passage["first_transition_by_absolute_k_distance"]["5"] is None
    assert passage["maximum_absolute_k_distance"] == 2
    assert passage["net_k_displacement"] == 2


def test_rank_convergence_metrics_are_strict_json_compatible(
    report_module: ModuleType,
) -> None:
    generator = np.random.default_rng(9)
    values = generator.normal(size=(4, 200))
    result = report_module._convergence_metric(values)

    assert result["rank_normalized_split_rhat"] is not None
    assert result["bulk_ess"] is not None
    assert result["tail_ess"] is not None
    assert result["mcse_over_sd"] is not None
    json.dumps(result, allow_nan=False)


def test_flux_auxiliary_contract_and_units(
    report_module: ModuleType,
) -> None:
    dataset = xr.Dataset(
        {
            "prior_flux": (
                ("lat", "lon"),
                np.ones((2, 3)),
                {"units": "mol m-2 s-1"},
            ),
            "grid_cell_area": (("lat", "lon"), np.full((2, 3), 2.0)),
            "country_fraction": (
                ("country", "lat", "lon"),
                np.full((2, 2, 3), 0.5),
            ),
        },
        coords={
            "lat": [50.0, 51.0],
            "lon": [-2.0, -1.0, 0.0],
            "country": ["GBR", "IRL"],
        },
    )
    prior, area, fractions = report_module._validate_flux_auxiliaries(dataset)
    assert prior.shape == (6,)
    assert area.tolist() == [2.0] * 6
    assert set(fractions) == {"GBR", "IRL"}

    dataset["prior_flux"].attrs["units"] = "unknown"
    with pytest.raises(ValueError, match="mol m-2 s-1"):
        report_module._validate_flux_auxiliaries(dataset)


def test_alternating_start_chain_identity_contract(
    report_module: ModuleType,
) -> None:
    chain_ids = ("chain-0", "chain-1", "chain-2", "chain-3")
    initial_k = (50, 250, 50, 250)
    initial_hashes = ("low-hash", "high-hash", "low-hash", "high-hash")

    report_module._validate_chain_identities(
        chain_ids,
        initial_k,
        initial_hashes,
    )

    with pytest.raises(ValueError, match="distinct chain IDs"):
        report_module._validate_chain_identities(
            ("chain-0", "chain-1", "chain-2", "chain-2"),
            initial_k,
            initial_hashes,
        )
    with pytest.raises(ValueError, match="Repeated Stage-3 initial K"):
        report_module._validate_chain_identities(
            chain_ids,
            initial_k,
            ("low-a", "high-hash", "low-b", "high-hash"),
        )
    with pytest.raises(ValueError, match="Distinct Stage-3 initial K"):
        report_module._validate_chain_identities(
            chain_ids,
            initial_k,
            ("same-hash", "same-hash", "same-hash", "same-hash"),
        )
