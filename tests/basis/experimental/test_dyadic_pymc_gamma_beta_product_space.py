"""Tests for the non-enumerating PyMC Gamma--Beta product space."""

from __future__ import annotations

from itertools import product
import math
from typing import Any

import numpy as np
import pymc as pm
import pytest

from openghg_inversions.basis.experimental.dyadic.gamma_beta import (
    DepthKappaStrategy,
    GammaBetaForest,
    GammaBetaGroupSpec,
)
from openghg_inversions.basis.experimental.dyadic.gamma_beta_partition import (
    GammaBetaPartitionLayout,
    GammaBetaRegionCountPrior,
)
from openghg_inversions.basis.experimental.dyadic.gamma_beta_product_space import (
    GammaBetaProductSpaceTarget,
)
from openghg_inversions.basis.experimental.dyadic.pymc_gamma_beta_product_space import (
    GammaBetaSplitMaskStep,
    build_pymc_gamma_beta_product_space_model,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:You are creating a TensorVariable with float64 dtype:UserWarning"
)


def _case(
    *,
    correlated_observation_errors: bool = True,
) -> tuple[
    GammaBetaProductSpaceTarget,
    GammaBetaPartitionLayout,
    GammaBetaRegionCountPrior,
]:
    """Build one stochastic two-grid-cell group and one fixed outer group."""
    forest = GammaBetaForest.from_groups(
        np.array([[1.0, 1.0, 2.0]]),
        [
            GammaBetaGroupSpec(
                "inner",
                np.array([[True, True, False]]),
                root_variance=0.25,
                max_depth=1,
            ),
            GammaBetaGroupSpec(
                "outer",
                np.array([[False, False, True]]),
                root_variance=0.0,
                max_depth=0,
            ),
        ],
        require_full_coverage=True,
    )
    covariance = (
        np.array([[0.25, 0.05], [0.05, 0.5]])
        if correlated_observation_errors
        else np.diag([0.25, 0.5])
    )
    target = GammaBetaProductSpaceTarget.from_grid(
        observations=np.array([3.0, -0.5]),
        observation_mean=np.array([0.2, -0.1]),
        finest_grid_design=np.array(
            [
                [[1.0, 2.0, 0.5]],
                [[-0.5, 1.0, 2.0]],
            ]
        ),
        forest=forest,
        kappa_strategy=DepthKappaStrategy(base_kappa=4.0),
        observation_covariance=covariance,
    )
    layout = GammaBetaPartitionLayout.from_forest(forest)
    prior = GammaBetaRegionCountPrior.uniform_k(layout)
    return target, layout, prior


def _logit(values: np.ndarray) -> np.ndarray:
    """Return elementwise log odds for values strictly inside zero and one."""
    return np.log(values) - np.log1p(-values)


def test_model_has_fixed_positive_coordinates_and_translates_state() -> None:
    """The graph should retain all Gamma/Beta variables under both partitions."""
    target, layout, prior = _case()
    adapter = build_pymc_gamma_beta_product_space_model(target, prior)
    point = adapter.model.initial_point()

    assert point["split_mask"].shape == (1,)
    assert point["stochastic_group_root_scalings_log__"].shape == (1,)
    assert point["split_fractions_logodds__"].shape == (1,)

    active, roots, node_scalings = adapter.physical_state(
        np.array([1]),
        np.array([1.2]),
        np.array([0.35]),
    )
    assert active == layout.active_node_ids(np.array([1]))
    np.testing.assert_array_equal(roots, [1.2, 1.0])
    np.testing.assert_allclose(
        node_scalings,
        target.coordinate_layout.node_scalings(roots, np.array([0.35])),
    )


def test_symbolic_prediction_and_likelihood_match_numpy_for_each_partition() -> None:
    """The fixed graph should reproduce the NumPy target for split and unsplit P."""
    target, layout, prior = _case()
    adapter = build_pymc_gamma_beta_product_space_model(target, prior)
    likelihood = adapter.model.compile_logp(vars=adapter.model.observed_RVs)
    roots = np.array([1.2])
    fractions = np.array([0.35])

    for split in (0, 1):
        mask = np.array([split], dtype=np.int8)
        point = adapter.model.initial_point()
        point["split_mask"] = mask.astype(point["split_mask"].dtype)
        point["stochastic_group_root_scalings_log__"] = np.log(roots).astype(
            point["stochastic_group_root_scalings_log__"].dtype
        )
        point["split_fractions_logodds__"] = _logit(fractions).astype(
            point["split_fractions_logodds__"].dtype
        )
        active = layout.active_node_ids(mask)
        group_roots = np.array([roots[0], 1.0])

        assert float(likelihood(point)) == pytest.approx(
            target.log_likelihood(active, group_roots, fractions),
            abs=2.0e-5,
        )


def test_symbolic_diagonal_likelihood_matches_numpy_oracle() -> None:
    """Independent Normal observations should retain the normalized density."""
    target, layout, prior = _case(correlated_observation_errors=False)
    adapter = build_pymc_gamma_beta_product_space_model(target, prior)
    likelihood = adapter.model.compile_logp(vars=adapter.model.observed_RVs)
    roots = np.array([1.2])
    fractions = np.array([0.35])
    point = adapter.model.initial_point()
    point["split_mask"] = np.array(
        [1],
        dtype=point["split_mask"].dtype,
    )
    point["stochastic_group_root_scalings_log__"] = np.log(roots).astype(
        point["stochastic_group_root_scalings_log__"].dtype
    )
    point["split_fractions_logodds__"] = _logit(fractions).astype(
        point["split_fractions_logodds__"].dtype
    )

    assert float(likelihood(point)) == pytest.approx(
        target.log_likelihood(
            layout.active_node_ids(np.array([1])),
            np.array([roots[0], 1.0]),
            fractions,
        ),
        abs=2.0e-5,
    )


def test_symbolic_canonical_gate_matches_forest_codec() -> None:
    """The PyTensor ancestry gate should accept exactly canonical deep masks."""
    forest = GammaBetaForest.from_groups(
        np.ones((1, 4)),
        [GammaBetaGroupSpec("inner", np.ones((1, 4), dtype=bool), max_depth=2)],
        require_full_coverage=True,
    )
    target = GammaBetaProductSpaceTarget.from_grid(
        observations=np.zeros(1),
        finest_grid_design=np.ones((1, 1, 4)),
        forest=forest,
        kappa_strategy=DepthKappaStrategy(),
    )
    layout = GammaBetaPartitionLayout.from_forest(forest)
    prior = GammaBetaRegionCountPrior.uniform_k(layout)
    adapter = build_pymc_gamma_beta_product_space_model(target, prior)
    logp = adapter.model.compile_logp()
    point = adapter.model.initial_point()

    for bits in product((0, 1), repeat=layout.split_count):
        mask = np.asarray(bits, dtype=np.int8)
        point["split_mask"] = mask.astype(point["split_mask"].dtype)
        try:
            layout.canonical_split_mask(mask)
        except ValueError:
            assert float(logp(point)) == -np.inf
        else:
            assert np.isfinite(float(logp(point)))


def test_symbolic_partition_density_has_normalized_absolute_measure() -> None:
    """Bernoulli base density plus potentials should equal the declared p(P)."""
    target, layout, prior = _case()
    adapter = build_pymc_gamma_beta_product_space_model(target, prior)
    full_logp = adapter.model.compile_logp()
    continuous_variables = [adapter.split_fractions, *adapter.model.observed_RVs]
    if adapter.stochastic_group_root_scalings is not None:
        continuous_variables.insert(0, adapter.stochastic_group_root_scalings)
    continuous_logp = adapter.model.compile_logp(vars=continuous_variables)
    point = adapter.model.initial_point()
    continuous_value = float(continuous_logp(point))

    recovered_mass = 0.0
    for split in (0, 1):
        mask = np.array([split], dtype=point["split_mask"].dtype)
        point["split_mask"] = mask
        recovered_log_prior = float(full_logp(point)) - continuous_value
        assert recovered_log_prior == pytest.approx(prior(mask), abs=2.0e-6)
        recovered_mass += math.exp(recovered_log_prior)

    assert recovered_mass == pytest.approx(1.0, abs=2.0e-6)


def test_custom_step_changes_only_the_partition_mask() -> None:
    """A structural update must preserve every transformed continuous value."""
    target, layout, prior = _case()
    adapter = build_pymc_gamma_beta_product_space_model(target, prior)
    with adapter.model:
        step = GammaBetaSplitMaskStep(
            adapter.split_mask,
            layout=layout,
            model=adapter.model,
            rng=7,
        )
    point = adapter.model.initial_point()
    point["stochastic_group_root_scalings_log__"][:] = np.log([1.2])
    point["split_fractions_logodds__"][:] = _logit(np.array([0.35]))
    source = {name: values.copy() for name, values in point.items()}

    updated, stats = step.step(point)

    np.testing.assert_array_equal(
        updated["stochastic_group_root_scalings_log__"],
        source["stochastic_group_root_scalings_log__"],
    )
    np.testing.assert_array_equal(
        updated["split_fractions_logodds__"],
        source["split_fractions_logodds__"],
    )
    layout.canonical_split_mask(updated["split_mask"])
    assert set(stats[0]) == {
        "accepted",
        "log_acceptance_ratio",
        "partition_regions",
        "proposal_degree",
        "proposed_split",
        "proposed_swap",
        "reverse_degree",
        "tune",
    }


def test_custom_step_uses_asymmetric_neighbor_hastings_correction() -> None:
    """A uniform-P target should expose only the reverse/source degree ratio."""
    forest = GammaBetaForest.from_groups(
        np.ones((1, 4)),
        [
            GammaBetaGroupSpec(
                "inner",
                np.ones((1, 4), dtype=bool),
                root_variance=0.25,
                max_depth=2,
            )
        ],
        require_full_coverage=True,
    )
    target = GammaBetaProductSpaceTarget.from_grid(
        observations=np.zeros(1),
        finest_grid_design=np.zeros((1, 1, 4)),
        forest=forest,
        kappa_strategy=DepthKappaStrategy(),
    )
    layout = GammaBetaPartitionLayout.from_forest(forest)
    prior = GammaBetaRegionCountPrior.from_marginal_probabilities(
        layout,
        {
            region_count: partition_count
            for region_count, partition_count in enumerate(layout.partition_counts_by_k)
            if partition_count
        },
    )
    source_mask = layout.initial_split_mask(2)
    adapter = build_pymc_gamma_beta_product_space_model(
        target,
        prior,
        initial_split_mask=source_mask,
    )
    with adapter.model:
        step = GammaBetaSplitMaskStep(
            adapter.split_mask,
            layout=layout,
            model=adapter.model,
            rng=3,
        )
    point = adapter.model.initial_point()

    updated, stats = step.step(point)

    assert bool(stats[0]["accepted"])
    assert int(stats[0]["proposal_degree"]) == len(layout.neighbors(source_mask)) == 3
    reverse_degree = len(layout.neighbors(updated["split_mask"]))
    assert int(stats[0]["reverse_degree"]) == reverse_degree
    assert float(stats[0]["log_acceptance_ratio"]) == pytest.approx(
        math.log(3.0 / reverse_degree),
        abs=2.0e-6,
    )


@pytest.mark.slow
def test_partition_step_recovers_exact_prior_only_distribution() -> None:
    """A long local chain should recover the exact five-partition prior."""
    forest = GammaBetaForest.from_groups(
        np.ones((1, 4)),
        [
            GammaBetaGroupSpec(
                "inner",
                np.ones((1, 4), dtype=bool),
                root_variance=0.25,
                max_depth=2,
            )
        ],
        require_full_coverage=True,
    )
    target = GammaBetaProductSpaceTarget.from_grid(
        observations=np.zeros(1),
        finest_grid_design=np.zeros((1, 1, 4)),
        forest=forest,
        kappa_strategy=DepthKappaStrategy(),
    )
    layout = GammaBetaPartitionLayout.from_forest(forest)
    prior = GammaBetaRegionCountPrior.from_marginal_probabilities(
        layout,
        {
            region_count: partition_count
            for region_count, partition_count in enumerate(layout.partition_counts_by_k)
            if partition_count
        },
    )
    adapter = build_pymc_gamma_beta_product_space_model(target, prior)
    with adapter.model:
        step = GammaBetaSplitMaskStep(
            adapter.split_mask,
            layout=layout,
            model=adapter.model,
            rng=20260719,
        )
    point = adapter.model.initial_point()
    canonical_masks: dict[tuple[int, ...], np.ndarray] = {}
    for bits in product((0, 1), repeat=layout.split_count):
        try:
            mask = layout.canonical_split_mask(bits)
        except ValueError:
            continue
        canonical_masks[tuple(int(value) for value in mask)] = mask
    counts = {mask: 0 for mask in canonical_masks}

    for draw in range(12_000):
        point, _ = step.step(point)
        if draw >= 2_000:
            counts[tuple(int(value) for value in point["split_mask"])] += 1

    assert len(canonical_masks) == 5
    for key, mask in canonical_masks.items():
        sampled_probability = counts[key] / 10_000
        assert sampled_probability == pytest.approx(
            math.exp(prior(mask)),
            abs=0.025,
        )


def test_step_factory_assigns_mask_before_native_nuts() -> None:
    """Compound steps should own disjoint variables in the intended order."""
    target, _, prior = _case()
    adapter = build_pymc_gamma_beta_product_space_model(target, prior)
    partition_step, nuts_step = adapter.step_methods(
        partition_rng=5,
        nuts_kwargs={"target_accept": 0.9},
    )
    compound = pm.CompoundStep([partition_step, nuts_step])

    assert compound.methods == [partition_step, nuts_step]
    partition_names = {variable.name for variable in partition_step.vars}
    nuts_names = {variable.name for variable in nuts_step.vars}
    assert partition_names == {"split_mask"}
    assert nuts_names == {
        "stochastic_group_root_scalings_log__",
        "split_fractions_logodds__",
    }
    assert partition_names.isdisjoint(nuts_names)


def test_tiny_chain_prefers_identifiable_split_truth() -> None:
    """The compound sampler should recover a strongly identified two-region truth."""
    observation_count = 40
    design = np.zeros((observation_count, 1, 2), dtype=float)
    design[: observation_count // 2, 0, 0] = 1.0
    design[observation_count // 2 :, 0, 1] = 1.0
    forest = GammaBetaForest.from_groups(
        np.ones((1, 2)),
        [
            GammaBetaGroupSpec(
                "inner",
                np.ones((1, 2), dtype=bool),
                root_variance=0.25,
                max_depth=1,
            )
        ],
        require_full_coverage=True,
    )
    truth = np.array([0.5, 1.5])
    observations = np.einsum("oij,ij->o", design, truth.reshape(1, 2))
    observations += np.random.default_rng(1701).normal(0.0, 0.05, observation_count)
    target = GammaBetaProductSpaceTarget.from_grid(
        observations=observations,
        finest_grid_design=design,
        forest=forest,
        kappa_strategy=DepthKappaStrategy(base_kappa=2.0),
        observation_sd=0.05,
    )
    layout = GammaBetaPartitionLayout.from_forest(forest)
    prior = GammaBetaRegionCountPrior.uniform_k(layout)
    adapter = build_pymc_gamma_beta_product_space_model(target, prior)
    steps = adapter.step_methods(
        partition_rng=20260719,
        nuts_kwargs={"target_accept": 0.9},
    )

    with adapter.model:
        trace: Any = pm.sample(
            draws=1_000,
            tune=1_000,
            chains=1,
            cores=1,
            step=list(steps),
            random_seed=20260719,
            progressbar=False,
            compute_convergence_checks=False,
            return_inferencedata=True,
        )

    split_probability = float(np.asarray(trace.posterior["split_mask"]).mean())
    predictions = np.asarray(trace.posterior["observation_prediction"]).reshape(
        -1,
        observation_count,
    )
    posterior_mean = predictions.mean(axis=0)
    prior_prediction = np.ones(observation_count)
    posterior_rmse = float(np.sqrt(np.mean(np.square(posterior_mean - observations))))
    prior_rmse = float(np.sqrt(np.mean(np.square(prior_prediction - observations))))

    assert split_probability > 0.9
    assert posterior_rmse < 0.25 * prior_rmse
    assert int(np.asarray(trace.sample_stats["diverging"]).sum()) == 0
