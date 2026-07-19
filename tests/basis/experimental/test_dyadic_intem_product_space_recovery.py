"""Focused tests for the exact 4 by 4 InTEM recovery benchmark."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.multiscale import MultiscaleDesign


@pytest.fixture(scope="module")
def benchmark_module() -> Iterator[ModuleType]:
    """Load the executable benchmark as an isolated module."""
    repository_root = Path(__file__).resolve().parents[3]
    script = repository_root / "examples/basis/dyadic_intem_product_space_recovery.py"
    module_name = "_test_dyadic_intem_product_space_recovery"
    specification = importlib.util.spec_from_file_location(module_name, script)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load recovery benchmark module from {script}.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    yield module
    sys.modules.pop(module_name, None)


@pytest.fixture(scope="module")
def recovery_case(benchmark_module: ModuleType) -> Any:
    """Build the deterministic matched-target recovery case once."""
    return benchmark_module.build_recovery_case()


def test_truth_is_representable_by_regular_k4_quadrants(
    benchmark_module: ModuleType,
    recovery_case: Any,
) -> None:
    """Four moderate active contrasts should exactly generate the quadrant truth."""
    case = recovery_case
    active_indices = case.target.contrast_layout.active_coordinate_indices(case.truth_partition)
    reconstructed = benchmark_module.decode_inner_field(
        case.target.contrast_layout,
        case.truth_partition,
        case.truth_coordinates,
    )
    bounds = [case.tree.tile(node_id).bounds for node_id in case.truth_partition.ordered_active()]

    assert len(case.truth_partition.active) == 4
    assert bounds == [
        (0, 2, 0, 2),
        (0, 2, 2, 4),
        (2, 4, 0, 2),
        (2, 4, 2, 4),
    ]
    assert active_indices == (0, 1, 2, 9)
    assert np.count_nonzero(case.truth_coordinates) == 4
    assert np.max(np.abs(case.truth_coordinates[list(active_indices)])) <= 0.5
    assert np.unique(case.inner_truth).size == 4
    np.testing.assert_allclose(reconstructed, case.inner_truth, rtol=0.0, atol=1e-15)


def test_training_target_excludes_all_holdout_rows(recovery_case: Any) -> None:
    """Only rows zero through 31 should condition the shared Gaussian target."""
    case = recovery_case
    rebuilt_training_design = MultiscaleDesign.from_grid(case.train_inner_design, case.tree)

    np.testing.assert_array_equal(case.train_row_indices, np.arange(32))
    np.testing.assert_array_equal(case.holdout_row_indices, np.arange(32, 48))
    assert not np.intersect1d(case.train_row_indices, case.holdout_row_indices).size
    assert case.train_inner_design.shape == (32, 4, 4)
    assert case.holdout_inner_design.shape == (16, 4, 4)
    assert case.train_outer_design.shape == (32, 7)
    assert case.holdout_outer_design.shape == (16, 7)
    assert case.target.observations.shape == (32,)
    assert case.holdout_observations.shape == (16,)
    np.testing.assert_array_equal(case.target.observations, case.train_observations)
    np.testing.assert_array_equal(case.target.observation_mean, case.train_observation_mean)
    np.testing.assert_array_equal(case.target.inner_design.values, rebuilt_training_design.values)
    np.testing.assert_array_equal(case.target.outer_design, case.train_outer_design)
    assert not np.shares_memory(case.target.observations, case.holdout_observations)


def test_predeclared_wrong_partition_has_same_k_but_different_p(recovery_case: Any) -> None:
    """The wrong geometry should be a valid K=4 frontier distinct from truth."""
    case = recovery_case
    case.truth_partition.validate(case.tree)
    case.wrong_partition.validate(case.tree)

    assert len(case.truth_partition.active) == len(case.wrong_partition.active) == 4
    assert case.truth_partition != case.wrong_partition
    assert not np.array_equal(
        case.truth_partition.to_labels(case.tree),
        case.wrong_partition.to_labels(case.tree),
    )
    assert len(case.underfit_partition.active) == 2


def test_exact_pk_over_nk_prior_is_normalized(recovery_case: Any) -> None:
    """Exact enumeration should give unit prior mass and uniform mass over K."""
    case = recovery_case
    region_counts = np.array([len(partition.active) for partition in case.partitions])
    prior = np.exp(np.array([case.target.partition_log_prior(partition) for partition in case.partitions]))
    expected_counts = {
        1: 1,
        2: 1,
        3: 2,
        4: 5,
        5: 14,
        6: 26,
        7: 44,
        8: 69,
        9: 94,
        10: 114,
        11: 116,
        12: 94,
        13: 60,
        14: 28,
        15: 8,
        16: 1,
    }

    assert len(case.partitions) == 677
    assert case.partition_counts == expected_counts
    assert prior.sum() == pytest.approx(1.0, abs=1e-14)
    for region_count, partition_count in expected_counts.items():
        selected = prior[region_counts == region_count]
        assert selected.size == partition_count
        assert selected.sum() == pytest.approx(1.0 / 16.0, abs=1e-14)
        np.testing.assert_allclose(selected, np.full(partition_count, 1.0 / (16 * partition_count)))


def test_exact_results_are_deterministic_finite_and_recover_truth(
    benchmark_module: ModuleType,
    recovery_case: Any,
) -> None:
    """Repeated analytic evaluation should be finite and retain the pinned recovery result."""
    first = benchmark_module.evaluate_recovery_case(recovery_case)
    second = benchmark_module.evaluate_recovery_case(recovery_case)

    assert first.as_dict() == second.as_dict()
    for value in _floating_values(asdict(first)):
        assert math.isfinite(value)

    assert first.diagnostics.partition_count == 677
    assert first.diagnostics.posterior_map_partition_index == first.diagnostics.truth_partition_index
    assert first.diagnostics.posterior_map_k == 4
    assert first.diagnostics.prior_mass_total == pytest.approx(1.0, abs=1e-14)
    assert first.diagnostics.truth_partition_probability == pytest.approx(
        0.5680998827076419,
        rel=1e-10,
    )
    assert (
        first.fixed_truth.holdout_log_predictive_density
        > first.fixed_wrong_k4.holdout_log_predictive_density
        > first.fixed_underfit.holdout_log_predictive_density
    )
    assert first.fixed_truth.field_rmse < first.fixed_wrong_k4.field_rmse
    assert first.fixed_truth.field_rmse < first.fixed_underfit.field_rmse


@pytest.mark.parametrize("sampler", ["augmented", "collapsed"])
def test_short_local_chain_uses_non_enumerating_partition_updates(
    benchmark_module: ModuleType,
    recovery_case: Any,
    sampler: str,
) -> None:
    """Both local kernels should move and return finite oracle comparisons."""
    result = benchmark_module.sample_recovery_case(
        recovery_case,
        draws=80,
        warmup=40,
        sampler=sampler,
        seed=91,
    )

    assert result.sampled.sampler == sampler
    assert result.sampled.draws == 80
    assert result.sampled.unique_partitions > 1
    assert 0.0 <= result.sampled.partition_acceptance_rate <= 1.0
    assert math.isfinite(result.sampled.sampled_mixture.holdout_log_predictive_density)
    assert result.sampled.exact_truth_probability == pytest.approx(
        result.exact.diagnostics.truth_partition_probability
    )


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ["augmented", "collapsed"])
def test_long_local_chain_matches_exact_k_and_predictive_oracle(
    benchmark_module: ModuleType,
    recovery_case: Any,
    sampler: str,
) -> None:
    """Declared local chains should reproduce exact K mass and predictions."""
    result = benchmark_module.sample_recovery_case(
        recovery_case,
        draws=20_000,
        warmup=2_000,
        sampler=sampler,
        seed=481,
    )

    assert result.sampled.k_total_variation_distance < 0.05
    assert abs(result.sampled.sampled_truth_probability - result.sampled.exact_truth_probability) < 0.05
    assert abs(
        result.sampled.sampled_mixture.holdout_log_predictive_density
        - result.exact.latent_677_partition_mixture.holdout_log_predictive_density
    ) < 0.5


def _floating_values(value: object) -> Iterator[float]:
    """Yield every floating-point scalar nested in a benchmark dictionary."""
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _floating_values(item)
