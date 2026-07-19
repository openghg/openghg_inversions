"""Tests for the TAC/MHD InTEM Gamma--Beta recovery experiment."""

from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:You are creating a TensorVariable with float64 dtype:UserWarning",
    "ignore:numpy.ndarray size changed, may indicate binary incompatibility:RuntimeWarning",
)


@pytest.fixture(scope="module")
def example_module() -> ModuleType:
    """Load the root example with its sibling calibration module available."""
    repository_root = Path(__file__).resolve().parents[3]
    examples_directory = repository_root / "examples" / "basis"
    script = examples_directory / "dyadic_gamma_beta_intem_product_space_recovery.py"
    module_name = "_test_dyadic_gamma_beta_intem_product_space_recovery"
    sys.path.insert(0, str(examples_directory))
    specification = util.spec_from_file_location(module_name, script)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load example module from {script}.")
    module = util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recovery_case(example_module: ModuleType) -> Any:
    """Build the calibrated 32-inner-region case once."""
    return example_module.build_case(
        data_directory=Path("tests/data"),
        inner_regions=32,
        data_seed=1701,
        k_continuation_probability=0.5,
    )


def test_case_uses_real_design_and_closes_declared_truth(recovery_case: Any) -> None:
    """The synthetic target should preserve real design rows and exact accounting."""
    case = recovery_case
    layout = case.partition_layout

    assert case.data.G.shape == (47, 293, 391)
    assert case.train_indices.size == 32
    assert case.holdout_indices.size == 15
    assert layout.minimum_regions == 11
    assert layout.maximum_regions == 38
    assert layout.split_count == 27
    assert layout.region_count(case.truth_split_mask) == 12
    assert case.k_continuation_probability == 0.5
    assert case.latent_prior.marginal_probability_by_k[11] > case.latent_prior.marginal_probability_by_k[12]
    assert case.coordinate_layout.forest is layout.forest
    assert case.train_target.coordinate_layout.forest is layout.forest
    assert case.holdout_target.coordinate_layout.forest is layout.forest
    assert np.all(np.isfinite(case.truth_field))
    assert np.unique(case.truth_field).size > 2
    np.testing.assert_allclose(
        case.train_noiseless,
        np.einsum(
            "oij,ij->o",
            case.data.G[case.train_indices],
            case.truth_field,
            optimize=True,
        ),
    )


def test_case_keeps_outer_groups_fixed_and_inner_groups_refinable(recovery_case: Any) -> None:
    """InTEM outer roots should be terminal while land/ocean supply split nodes."""
    case = recovery_case
    forest = case.coordinate_layout.forest
    outer_groups = tuple(group for group in forest.groups if group.name.startswith("intem_outer_"))
    inner_groups = tuple(group for group in forest.groups if group.name.startswith("inner_"))

    assert len(outer_groups) == 6
    assert {group.name for group in inner_groups} == {"inner_land", "inner_ocean"}
    for root_id in forest.root_ids:
        node = forest.nodes[root_id]
        group = forest.groups[node.group_index]
        if group.name.startswith("intem_outer_"):
            assert not node.child_ids
    assert any(
        node.child_ids and forest.groups[node.group_index].name == "inner_land"
        for node in forest.nodes
    )
    assert any(
        forest.groups[forest.nodes[root_id].group_index].name == "inner_ocean"
        for root_id in forest.root_ids
    )
    assert not any(
        node.child_ids and forest.groups[node.group_index].name == "inner_ocean"
        for node in forest.nodes
    )


@pytest.mark.slow
def test_short_latent_chain_runs_on_data_backed_forest(
    example_module: ModuleType,
    recovery_case: Any,
) -> None:
    """A short realistic chain should construct, move, and retain finite diagnostics."""
    benchmark = example_module.run_benchmark(
        recovery_case,
        draws=100,
        tune=100,
        sampling_seed=20260719,
        target_accept=0.95,
    )

    assert benchmark.observation_count == 47
    assert benchmark.possible_split_count == 27
    for fit in (benchmark.latent, benchmark.fixed_true, benchmark.fixed_underfit):
        assert np.isfinite(fit.holdout_prediction_rmse)
        assert np.isfinite(fit.inner_land_field_rmse)
        assert np.isfinite(fit.holdout_log_predictive_density)
        assert fit.divergence_count >= 0
