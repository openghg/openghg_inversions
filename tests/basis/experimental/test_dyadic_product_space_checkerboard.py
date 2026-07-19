"""Tests for the latent-versus-fixed dyadic checkerboard benchmark."""

from collections.abc import Iterator
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree


@pytest.fixture(scope="module")
def benchmark_module() -> Iterator[ModuleType]:
    """Load the executable benchmark as an isolated module."""
    repository_root = Path(__file__).resolve().parents[3]
    script = repository_root / "examples/basis/dyadic_product_space_checkerboard.py"
    module_name = "_test_dyadic_product_space_checkerboard"
    specification = importlib.util.spec_from_file_location(module_name, script)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load benchmark module from {script}.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    yield module
    sys.modules.pop(module_name, None)


def test_checkerboard_truth_has_declared_regular_blocks(benchmark_module: Any) -> None:
    """The benchmark truth should alternate 0.5/1.5 on 2 by 2 regions."""
    truth = benchmark_module.checkerboard_truth()

    assert truth.shape == (8, 8)
    assert set(np.unique(truth)) == {0.5, 1.5}
    np.testing.assert_array_equal(truth[:2, :2], np.full((2, 2), 0.5))
    np.testing.assert_array_equal(truth[:2, 2:4], np.full((2, 2), 1.5))


def test_regular_depth_partitions_match_coarse_and_truth_counts(benchmark_module: Any) -> None:
    """Balanced 8 by 8 depth frontiers should contain 8 and 16 regions."""
    tree = DyadicTree.from_shape((8, 8))

    coarse = benchmark_module.regular_depth_partition(tree, 3)
    truth = benchmark_module.regular_depth_partition(tree, 4)

    assert len(coarse.active) == 8
    assert len(truth.active) == 16
    assert np.all(np.bincount(truth.to_labels(tree).ravel())[1:] == 4)


def test_checkerboard_case_uses_disjoint_train_and_holdout_rows(benchmark_module: Any) -> None:
    """The target and holdout arrays should have their documented dimensions."""
    case = benchmark_module.build_checkerboard_case()

    assert case.target.observations.shape == (64,)
    assert case.train_design.shape == (64, 8, 8)
    assert case.holdout_design.shape == (32, 8, 8)
    assert case.holdout_observations.shape == (32,)
    assert case.holdout_noiseless.shape == (32,)
    assert len(case.wrong_partition.active) == len(case.truth_partition.active) == 16
    assert case.wrong_partition != case.truth_partition


def test_short_benchmark_exercises_latent_k_and_p(benchmark_module: Any) -> None:
    """A short smoke run should move away from the fixed starting partition."""
    result = benchmark_module.run_benchmark(draws=30, warmup=20, seed=19)

    assert result.fixed_truth.mean_regions == 16.0
    assert result.fixed_wrong.mean_regions == 16.0
    assert result.fixed_coarse.mean_regions == 8.0
    assert result.latent_unique_partitions > 1
    assert result.latent.minimum_regions >= 8
    assert result.latent.maximum_regions <= 28
    assert 0.0 <= result.latent_partition_acceptance_rate <= 1.0
    assert result.latent_warmup_acceptance_rate is not None


@pytest.mark.slow
def test_seeded_latent_benchmark_beats_predeclared_fixed_partitions(
    benchmark_module: Any,
) -> None:
    """The declared long run should beat fixed wrong-P and underfit baselines."""
    result = benchmark_module.run_benchmark(draws=4_000, warmup=2_000, seed=481)

    assert result.latent_beats_wrong_prediction_rmse
    assert result.latent_beats_coarse_prediction_rmse
    assert result.latent.predictive_log_density > result.fixed_wrong.predictive_log_density
    assert result.latent.recovered_checkerboard_contrast > 0.8
    assert result.latent_unique_partitions > 500
