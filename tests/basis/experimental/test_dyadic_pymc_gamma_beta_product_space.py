"""Tests for the non-enumerating PyMC Gamma--Beta product space."""

from __future__ import annotations

from itertools import product
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


def _case() -> tuple[
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
        observation_covariance=np.array([[0.25, 0.05], [0.05, 0.5]]),
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
        "reverse_degree",
        "tune",
    }


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
