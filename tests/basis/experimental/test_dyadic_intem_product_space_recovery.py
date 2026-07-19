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

pytestmark = pytest.mark.filterwarnings(
    "ignore:You are creating a TensorVariable with float64 dtype:UserWarning"
)


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


def test_depth_first_leaf_values_are_scattered_to_grid_order(
    benchmark_module: ModuleType,
    recovery_case: Any,
) -> None:
    """Leaf-order vectors should not be reshaped as row-major grid values."""
    tree = recovery_case.tree
    leaf_values = np.asarray([tree.leaf_ids], dtype=float)
    expected = np.empty((1, *tree.shape), dtype=float)
    for leaf_index, node_id in enumerate(tree.leaf_ids):
        tile = tree.tile(node_id)
        expected[:, tile.row_start, tile.col_start] = leaf_values[:, leaf_index]

    actual = benchmark_module._leaf_values_to_grid(tree, leaf_values)

    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual, leaf_values.reshape(1, *tree.shape))


def test_monte_carlo_holdout_density_matches_explicit_gaussian_mixtures(
    benchmark_module: ModuleType,
    recovery_case: Any,
) -> None:
    """One- and two-component holdout mixtures should match direct logpdfs."""
    case = recovery_case
    first = case.holdout_observations - 0.01
    second = case.holdout_observations + 0.02
    first_logp = benchmark_module._multivariate_normal_logpdf(
        case.holdout_observations,
        first,
        case.holdout_observation_covariance,
    )
    second_logp = benchmark_module._multivariate_normal_logpdf(
        case.holdout_observations,
        second,
        case.holdout_observation_covariance,
    )
    maximum = max(first_logp, second_logp)
    expected_mixture = maximum + math.log(
        (math.exp(first_logp - maximum) + math.exp(second_logp - maximum)) / 2.0
    )

    assert benchmark_module._monte_carlo_holdout_log_density(
        case,
        first[None, :],
    ) == pytest.approx(first_logp)
    assert benchmark_module._monte_carlo_holdout_log_density(
        case,
        np.stack((first, second)),
    ) == pytest.approx(expected_mixture)


def test_conditional_acceptance_rate_returns_none_for_absent_move_type(
    benchmark_module: ModuleType,
) -> None:
    """Missing split or merge proposals should not serialize as NaN."""
    accepted = np.array([True, False, True])

    assert benchmark_module._conditional_acceptance_rate(
        accepted,
        np.array([True, True, False]),
    ) == pytest.approx(0.5)
    assert benchmark_module._conditional_acceptance_rate(
        accepted,
        np.zeros(3, dtype=bool),
    ) is None


def test_effective_sample_helpers_normalize_arviz_dataset_results(
    benchmark_module: ModuleType,
) -> None:
    """ESS and MCSE helpers should return finite Python scalars."""
    draws = np.random.default_rng(4).normal(size=1_000)
    ess, mcse = benchmark_module._bulk_ess_and_mean_mcse(draws)
    minimum_ess = benchmark_module._minimum_bulk_ess(
        np.column_stack((draws, np.roll(draws, 1)))
    )

    assert math.isfinite(ess) and ess > 1.0
    assert math.isfinite(mcse) and mcse > 0.0
    assert math.isfinite(minimum_ess) and minimum_ess > 1.0


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


@pytest.mark.slow
def test_pymc_split_mask_and_nuts_chain_matches_fixed_and_latent_oracles(
    benchmark_module: ModuleType,
    recovery_case: Any,
) -> None:
    """The full PyMC chain should recover P and direct posterior predictions."""
    result = benchmark_module.sample_pymc_recovery_case(
        recovery_case,
        draws=20_000,
        warmup=3_000,
        seed=481,
        target_accept=0.95,
    )

    assert result.sampled.divergence_count / result.sampled.draws < 0.005
    assert result.sampled.k_total_variation_distance < 0.08
    assert result.sampled.partition_total_variation_distance < 0.12
    assert abs(
        result.sampled.sampled_truth_probability
        - result.sampled.exact_truth_probability
    ) < 0.08
    assert result.sampled.beats_wrong_fixed_k4
    assert result.sampled.beats_underfit_fixed_k2
    assert result.sampled.oracle_log_score_noninferior
    assert result.sampled.basis_region_count_bulk_ess > 1.0
    assert result.sampled.basis_region_count_mcse > 0.0
    assert result.sampled.truth_partition_indicator_bulk_ess > 1.0
    assert result.sampled.truth_partition_probability_mcse > 0.0
    assert result.sampled.minimum_inner_coordinate_bulk_ess > 1.0
    assert abs(
        result.sampled.sampled_posterior.holdout_log_predictive_density
        - result.exact.latent_677_partition_mixture.holdout_log_predictive_density
    ) < 0.5
    assert abs(
        result.sampled.sampled_posterior.noiseless_holdout_rmse
        - result.exact.latent_677_partition_mixture.noiseless_holdout_rmse
    ) < 0.02
    assert abs(
        result.sampled.sampled_posterior.field_rmse
        - result.exact.latent_677_partition_mixture.field_rmse
    ) < 0.02


def _floating_values(value: object) -> Iterator[float]:
    """Yield every floating-point scalar nested in a benchmark dictionary."""
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _floating_values(item)
