"""Tests for the non-enumerating PyMC dyadic split-mask adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm
import pytest

from openghg_inversions.basis.experimental.dyadic.enumeration import enumerate_partitions
from openghg_inversions.basis.experimental.dyadic.gaussian_product_space import (
    GaussianProductSpaceTarget,
)
from openghg_inversions.basis.experimental.dyadic.partition_prior import (
    RegionCountPartitionPrior,
)
from openghg_inversions.basis.experimental.dyadic.product_space import ProductSpaceState
from openghg_inversions.basis.experimental.dyadic.pymc_split_mask_product_space import (
    DyadicSplitMaskStep,
    build_pymc_split_mask_product_space_model,
)
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree

pytestmark = pytest.mark.filterwarnings(
    "ignore:You are creating a TensorVariable with float64 dtype:UserWarning"
)


def _target(
    *,
    observation: float = 1.2,
) -> tuple[GaussianProductSpaceTarget, RegionCountPartitionPrior, tuple[PartitionState, ...]]:
    """Build a three-partition Gaussian target with a symbolic prior."""
    tree = DyadicTree.from_shape((1, 3))
    partitions = enumerate_partitions(tree)
    prior = RegionCountPartitionPrior.uniform_k(tree)
    target = GaussianProductSpaceTarget.from_grid(
        observations=np.array([observation]),
        inner_grid_design=np.array([[[1.0, 2.0, 3.0]]]),
        tree=tree,
        observation_covariance=np.array([[1.0]]),
        inner_prior_scale=1.0,
        inactive_pseudo_prior_scale=1.0,
        outer_design=np.array([[2.0]]),
        outer_prior_covariance=np.array([[0.25]]),
        partition_log_prior=prior,
    )
    return target, prior, partitions


def test_model_uses_fixed_split_mask_and_translates_physical_state() -> None:
    """The model should expose one permanent mask and continuous state."""
    target, _, partitions = _target()
    adapter = build_pymc_split_mask_product_space_model(target)
    point = adapter.model.initial_point()

    assert point["split_mask"].shape == (2,)
    assert point["split_mask"].dtype.kind in "iu"
    assert point["inner_whitened"].shape == (3,)
    assert point["outer_whitened"].shape == (1,)

    mask = target.contrast_layout.split_mask(partitions[1])
    partition, inner, outer = adapter.physical_state(
        mask,
        np.array([0.2, -0.4, 0.7]),
        np.array([0.5]),
    )
    assert partition == partitions[1]
    np.testing.assert_allclose(
        inner,
        np.sqrt(target.inner_prior_variances) * np.array([0.2, -0.4, 0.7]),
    )
    np.testing.assert_allclose(outer, [0.25])


def test_model_logp_matches_numpy_target_for_every_partition() -> None:
    """The static symbolic design should equal every gathered NumPy design."""
    target, _, partitions = _target(observation=-0.4)
    adapter = build_pymc_split_mask_product_space_model(target)
    logp = adapter.model.compile_logp()
    differences: list[float] = []

    for inner_raw, outer_raw in (
        (np.array([0.0, 0.0, 0.0]), np.array([0.0])),
        (np.array([0.3, -0.8, 1.1]), np.array([-0.4])),
    ):
        for partition in partitions:
            mask = target.contrast_layout.split_mask(partition)
            point = adapter.model.initial_point()
            point["split_mask"] = mask.astype(point["split_mask"].dtype)
            point["inner_whitened"] = inner_raw.astype(point["inner_whitened"].dtype)
            point["outer_whitened"] = outer_raw.astype(point["outer_whitened"].dtype)
            _, inner, outer = adapter.physical_state(mask, inner_raw, outer_raw)
            reference = target.log_density(ProductSpaceState(partition, inner, outer))
            differences.append(float(logp(point)) - reference)

    np.testing.assert_allclose(differences, differences[0], atol=2e-5)


def test_model_rejects_noncanonical_mask_in_symbolic_density() -> None:
    """A descendant split below an unsplit root should have zero density."""
    target, _, _ = _target()
    adapter = build_pymc_split_mask_product_space_model(target)
    logp = adapter.model.compile_logp()
    point = adapter.model.initial_point()
    point["split_mask"] = np.array([0, 1], dtype=point["split_mask"].dtype)

    assert float(logp(point)) == -np.inf


def test_split_mask_step_changes_only_partition_and_preserves_canonical_mask() -> None:
    """The custom step should leave both continuous arrays unchanged."""
    target, _, _ = _target()
    adapter = build_pymc_split_mask_product_space_model(target)
    with adapter.model:
        step = DyadicSplitMaskStep(
            adapter.split_mask,
            layout=target.contrast_layout,
            model=adapter.model,
            rng=9,
        )
    point = adapter.model.initial_point()
    point["inner_whitened"][:] = np.array([0.2, -0.4, 0.7])
    point["outer_whitened"][:] = np.array([0.3])
    source = {name: values.copy() for name, values in point.items()}

    updated, stats = step.step(point)

    np.testing.assert_array_equal(updated["inner_whitened"], source["inner_whitened"])
    np.testing.assert_array_equal(updated["outer_whitened"], source["outer_whitened"])
    mask = np.asarray(updated["split_mask"], dtype=np.bool_)
    target.contrast_layout.partition_from_split_mask(mask)
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
    """The compound steps should own disjoint variables in declared order."""
    target, _, _ = _target()
    adapter = build_pymc_split_mask_product_space_model(target)
    partition_step, nuts_step = adapter.step_methods(
        partition_rng=5,
        nuts_kwargs={"target_accept": 0.85},
    )
    compound = pm.CompoundStep([partition_step, nuts_step])

    assert compound.methods == [partition_step, nuts_step]
    partition_names = {variable.name for variable in partition_step.vars}
    nuts_names = {variable.name for variable in nuts_step.vars}
    assert partition_names == {"split_mask"}
    assert nuts_names == {"inner_whitened", "outer_whitened"}
    assert partition_names.isdisjoint(nuts_names)


def test_native_nuts_chain_matches_exact_tiny_partition_oracle() -> None:
    """A split-mask and NUTS chain should recover the exact partition marginal."""
    target, _, partitions = _target(observation=1.8)
    expected = target.partition_probabilities(partitions)
    adapter = build_pymc_split_mask_product_space_model(target)
    steps = adapter.step_methods(
        partition_rng=20260719,
        nuts_kwargs={"target_accept": 0.85},
    )

    with adapter.model:
        trace: Any = pm.sample(
            draws=2_000,
            tune=1_000,
            chains=1,
            cores=1,
            step=list(steps),
            random_seed=20260719,
            progressbar=False,
            compute_convergence_checks=False,
            return_inferencedata=True,
        )

    masks = np.asarray(trace.posterior["split_mask"]).reshape(-1, 2)
    observed = np.zeros(len(partitions), dtype=float)
    partition_indices = {partition: index for index, partition in enumerate(partitions)}
    for mask in masks:
        partition = target.contrast_layout.partition_from_split_mask(mask.astype(np.bool_))
        observed[partition_indices[partition]] += 1.0
    observed /= masks.shape[0]

    np.testing.assert_allclose(observed, list(expected.values()), atol=0.06)
    assert "basis_region_count" in trace.posterior
    assert trace.posterior["inner_whitened"].shape[-1] == 3
    assert any(name.endswith("accepted") for name in trace.sample_stats)
    assert "step_size" in trace.sample_stats


def test_builder_scales_to_eight_by_eight_without_partition_catalogue() -> None:
    """An 8x8 model should allocate only tree-sized mask and design arrays."""
    tree = DyadicTree.from_shape((8, 8))
    prior = RegionCountPartitionPrior.uniform_k(
        tree,
        minimum_regions=8,
        maximum_regions=28,
    )
    target = GaussianProductSpaceTarget.from_grid(
        observations=np.zeros(3),
        inner_grid_design=np.ones((3, 8, 8)),
        tree=tree,
        observation_covariance=np.eye(3),
        partition_log_prior=prior,
    )

    adapter = build_pymc_split_mask_product_space_model(target)
    point = adapter.model.initial_point()

    assert point["split_mask"].shape == (63,)
    assert point["inner_whitened"].shape == (64,)
    assert int(point["split_mask"].sum()) + 1 == 8
    assert "partition_index" not in point


def test_builder_rejects_unsymbolic_prior_and_partition_dependent_whitening() -> None:
    """The scalable builder should reject callbacks and unequal pseudo-priors."""
    target, prior, _ = _target()
    callback_target = GaussianProductSpaceTarget.from_grid(
        observations=target.observations,
        observation_mean=target.observation_mean,
        inner_grid_design=np.array([[[1.0, 2.0, 3.0]]]),
        tree=target.tree,
        observation_covariance=target.observation_covariance,
        partition_log_prior=lambda partition: prior(partition),
    )
    with pytest.raises(TypeError, match="RegionCountPartitionPrior"):
        build_pymc_split_mask_product_space_model(callback_target)

    unequal_target = GaussianProductSpaceTarget.from_grid(
        observations=np.array([1.2]),
        inner_grid_design=np.array([[[1.0, 2.0, 3.0]]]),
        tree=target.tree,
        observation_covariance=np.array([[1.0]]),
        inactive_pseudo_prior_scale=2.0,
        partition_log_prior=prior,
    )
    with pytest.raises(ValueError, match="pseudo-prior variances"):
        build_pymc_split_mask_product_space_model(unequal_target)
