"""Frozen identities for the atmospheric-like MH-guided S1 experiment."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.mh_local_search_nuts_reference import (
    prepare_s0_nuts_reference,
    reference_seeds,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_synthetic import (
    DEFINITION_SCHEMA,
    build_stage_definition,
    canonical_json,
    file_sha256,
    frozen_local_reference_seeds,
    frozen_stage_budgets,
    json_sha256,
    materialize_replicate,
    stage_operators,
    write_envelope,
)
from openghg_inversions.experimental.rjmcmc.mh_local_search_retry_authorization import (
    validate_retry_authorization_token,
)


def _driver() -> ModuleType:
    path = Path(__file__).parents[3] / "examples/rjmcmc/mh_local_search_synthetic.py"
    spec = importlib.util.spec_from_file_location("mh_local_search_s1_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s1_definition_has_exact_golden_identity(tmp_path: Path) -> None:
    definition = build_stage_definition("s1")
    assert json_sha256(definition) == "24099340c5e192bbd258e32270e61247bbad33769da277c39d143f15702f819d"
    assert (
        definition["training_operator_sha256"]
        == "382df6a86723ae6988c336ef6825a6b9476d3ad3817e30c425a20932fa58cada"
    )
    assert (
        definition["heldout_operator_sha256"]
        == "3a789ca21a2d6aea0b541dd344ec97c2e1cd163ec21367bf23902ce25d5a4c56"
    )
    assert definition["p0_sha256"] == "e38df6bdef46ea93b77debfa3b2c4c4efa44cbc816128c8f4376fd5facdd23a1"
    scenarios = cast(dict[str, Any], definition["scenarios"])
    assert scenarios["edge-one"]["pstar_sha256"] == (
        "2a12c303c176e120f8937910800128f1465cf4414b9cf29c2b4102745c8b51c2"
    )
    assert scenarios["edge-one"]["witness"]["path_index"] == 0
    assert scenarios["relocation-one"]["pstar_sha256"] == (
        "203177d3a2a8ec6e0b8f6bb749a0fef98e96323bf2c16ddf1d246906c96fdce2"
    )
    assert scenarios["relocation-one"]["witness"]["path_index"] == 0
    envelope = tmp_path / "definition.json"
    write_envelope(envelope, DEFINITION_SCHEMA, definition)
    assert file_sha256(envelope) == "d90c89431d2b19d83084bb51ddd556f3457cf4f53498dd4772b5de9dc6755d60"


def test_s1_reference_budgets_and_seeds_are_frozen() -> None:
    assert frozen_stage_budgets("s1") == (10_000, 50_000, 5)
    assert reference_seeds("s1") == ((None, 74101, 74102, 74103), 74100)
    assert frozen_local_reference_seeds("s1") == (74201, 74202, 74203, 74204)


def test_s1_training_and_heldout_operators_have_no_identical_rows() -> None:
    training, _, heldout, _ = stage_operators("s1")
    assert not np.any(np.all(training[:, None, :] == heldout[None, :, :], axis=2))


def test_s1_nuts_reference_uses_only_replicate_zero_and_selected_topology() -> None:
    definition = build_stage_definition("s1")
    training, evaluation = materialize_replicate(
        definition,
        scenario="edge-one",
        replicate=0,
    )
    p0 = prepare_s0_nuts_reference(training, evaluation, topology_role="p0")
    pstar = prepare_s0_nuts_reference(training, evaluation, topology_role="pstar")
    assert p0.stage == pstar.stage == "s1"
    assert p0.cell_name == "edge-one-p0"
    assert pstar.cell_name == "edge-one-pstar"
    assert tuple(start.seed for start in p0.starts) == (None, 74101, 74102, 74103)
    assert p0.data.rectangle_bounds.tolist() != pstar.data.rectangle_bounds.tolist()


def test_s1_gate_thresholds_do_not_use_equal_wall_or_s0_aligned_cap() -> None:
    driver = _driver()
    aligned = {
        "median_mobile_over_fixed_heldout": 1.09,
        "mobile_over_fixed_heldout": [0.9, 1.0, 1.18, 1.40],
    }
    assert driver._stage_utility_passes("s1", "aligned", aligned)
    assert not driver._stage_utility_passes("s0", "aligned", aligned)
    misaligned = {
        "median_mobile_over_fixed_heldout": 0.95,
        "mobile_over_fixed_heldout": [0.8, 0.9, 1.0, 1.20],
        "mobile_heldout_below_one_count": 3,
        "median_mobile_over_fixed_native": 0.98,
        "mobile_over_fixed_equal_wall_heldout": [2.0, 2.0, 2.0, 2.0],
    }
    assert driver._stage_utility_passes("s1", "edge-one", misaligned)
    misaligned["mobile_over_fixed_heldout"] = [0.8, 0.9, 1.0, 1.2000000000001]
    assert not driver._stage_utility_passes("s1", "edge-one", misaligned)


def test_factor4_token_scope_is_bound_to_s1(tmp_path: Path) -> None:
    token = {
        "schema": "openghg_inversions.mh_local_search_retry_authorization_token.v1",
        "source_revision": "a" * 40,
        "definition_sha256": "b" * 64,
        "scope": "s1-homogeneous-factor4-branch-matrix-v1",
        "authorized_branch_profile": "factor4",
        "primary_conditional_reference_completion_sha256": "c" * 64,
        "primary_nuts_completion_sha256": "d" * 64,
        "primary_local_completion_sha256": "e" * 64,
    }
    path = tmp_path / "token.json"
    path.write_text(canonical_json(token) + "\n")
    assert (
        len(
            validate_retry_authorization_token(
                path,
                source_revision="a" * 40,
                definition_sha256="b" * 64,
                stage="s1",
            )
        )
        == 64
    )
    with pytest.raises(ValueError, match="identity"):
        validate_retry_authorization_token(
            path,
            source_revision="a" * 40,
            definition_sha256="b" * 64,
            stage="s0",
        )


def test_s1_training_is_published_before_sealed_evaluation(tmp_path: Path) -> None:
    driver = _driver()
    definition_path = tmp_path / "definition.json"
    write_envelope(
        definition_path,
        DEFINITION_SCHEMA,
        build_stage_definition("s1"),
    )
    training_path = tmp_path / "training.json"
    evaluation_path = tmp_path / "evaluation.json"
    driver.command_materialize_training(
        argparse.Namespace(
            definition=definition_path,
            scenario="edge-one",
            replicate=0,
            training_output=training_path,
        )
    )
    assert training_path.is_file()
    assert not evaluation_path.exists()
    driver.command_materialize_evaluation(
        argparse.Namespace(
            definition=definition_path,
            training=training_path,
            scenario="edge-one",
            replicate=0,
            evaluation_output=evaluation_path,
        )
    )
    assert evaluation_path.is_file()
