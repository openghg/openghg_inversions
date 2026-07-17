"""Tests for the native PyMC adapter to the dyadic Gaussian product space."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pymc as pm
import pytest
from pymc.blocking import DictToArrayBijection

from openghg_inversions.basis.experimental.dyadic.enumeration import enumerate_partitions
from openghg_inversions.basis.experimental.dyadic.gaussian_product_space import (
    GaussianProductSpaceTarget,
)
from openghg_inversions.basis.experimental.dyadic.product_space import ProductSpaceState
from openghg_inversions.basis.experimental.dyadic.pymc_product_space import (
    DyadicPartitionStep,
    build_pymc_product_space_model,
)
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree

pytestmark = pytest.mark.filterwarnings(
    "ignore:You are creating a TensorVariable with float64 dtype:UserWarning"
)


def _target(
    *,
    observation: float = 1.2,
    pseudo_prior_scale: float = 1.0,
    partition_log_weights: tuple[float, ...] | None = None,
) -> tuple[GaussianProductSpaceTarget, tuple[PartitionState, ...]]:
    """Build the three-partition Gaussian reference target used by adapter tests."""
    tree = DyadicTree.from_shape((1, 3))
    partitions = enumerate_partitions(tree)
    weights = partition_log_weights or tuple(-math.log(len(partitions)) for _ in partitions)
    if len(weights) != len(partitions):
        raise ValueError("partition_log_weights must match the enumerated partitions.")

    def partition_prior(partition: PartitionState) -> float:
        """Return the configured log weight for one explicit partition."""
        assert partition in partitions
        return weights[partitions.index(partition)]

    target = GaussianProductSpaceTarget.from_grid(
        observations=np.array([observation]),
        inner_grid_design=np.array([[[1.0, 2.0, 3.0]]]),
        tree=tree,
        observation_covariance=np.array([[1.0]]),
        inner_prior_scale=1.0,
        inactive_pseudo_prior_scale=pseudo_prior_scale,
        outer_design=np.array([[2.0]]),
        outer_prior_covariance=np.array([[0.25]]),
        partition_log_prior=partition_prior,
    )
    return target, partitions


def test_model_has_fixed_shape_variables_and_translates_physical_state() -> None:
    """The model should preserve permanent inner and fixed outer dimensions."""
    target, partitions = _target()
    adapter = build_pymc_product_space_model(target, partitions)

    point = adapter.model.initial_point()

    assert np.asarray(point["partition_index"]).shape == ()
    assert np.asarray(point["partition_index"]).dtype.kind in "iu"
    assert point["inner_whitened"].shape == (3,)
    assert point["outer_whitened"].shape == (1,)
    partition, inner, outer = adapter.physical_state(
        1,
        np.array([0.2, -0.4, 0.7]),
        np.array([0.5]),
    )
    assert partition == partitions[1]
    np.testing.assert_allclose(
        inner,
        np.sqrt(target.inner_prior_variances) * np.array([0.2, -0.4, 0.7]),
    )
    np.testing.assert_allclose(outer, [0.25])


def test_model_logp_matches_numpy_target_up_to_whitening_jacobian() -> None:
    """PyTensor and NumPy augmented densities should differ by one constant."""
    target, partitions = _target(observation=-0.4)
    adapter = build_pymc_product_space_model(target, partitions)
    logp = adapter.model.compile_logp()
    differences: list[float] = []

    for inner_raw, outer_raw in (
        (np.array([0.0, 0.0, 0.0]), np.array([0.0])),
        (np.array([0.3, -0.8, 1.1]), np.array([-0.4])),
    ):
        for index, partition in enumerate(partitions):
            point = adapter.model.initial_point()
            point["partition_index"] = np.asarray(index, dtype=point["partition_index"].dtype)
            point["inner_whitened"] = inner_raw.astype(point["inner_whitened"].dtype)
            point["outer_whitened"] = outer_raw.astype(point["outer_whitened"].dtype)
            _, inner, outer = adapter.physical_state(index, inner_raw, outer_raw)
            reference = target.log_density(ProductSpaceState(partition, inner, outer))
            differences.append(float(logp(point)) - reference)

    np.testing.assert_allclose(differences, differences[0], atol=2e-5)


def test_partition_step_changes_only_partition_and_tracks_tuning() -> None:
    """The custom step should preserve continuous point arrays and tuning state."""
    target, partitions = _target()
    adapter = build_pymc_product_space_model(target, partitions)
    with adapter.model:
        step = DyadicPartitionStep(
            adapter.partition_index,
            tree=target.tree,
            partitions=partitions,
            model=adapter.model,
            rng=9,
        )
    point = adapter.model.initial_point()
    point["inner_whitened"][:] = np.array([0.2, -0.4, 0.7])
    point["outer_whitened"][:] = np.array([0.3])
    source = {name: values.copy() for name, values in point.items()}

    updated, stats = step.step(point)

    np.testing.assert_array_equal(point["inner_whitened"], source["inner_whitened"])
    np.testing.assert_array_equal(point["outer_whitened"], source["outer_whitened"])
    np.testing.assert_array_equal(updated["inner_whitened"], source["inner_whitened"])
    np.testing.assert_array_equal(updated["outer_whitened"], source["outer_whitened"])
    assert np.asarray(updated["partition_index"]).dtype == source["partition_index"].dtype
    assert stats[0]["tune"] is True
    assert set(stats[0]) == {
        "accepted",
        "log_acceptance_ratio",
        "partition_regions",
        "tune",
    }

    step.stop_tuning()
    _, stopped_stats = step.step(updated)
    assert stopped_stats[0]["tune"] is False
    step.reset_tuning()
    _, reset_stats = step.step(updated)
    assert reset_stats[0]["tune"] is True


def test_partition_step_applies_asymmetric_hastings_correction() -> None:
    """Root-to-middle and middle-to-root moves should use opposite log-two terms."""
    target, partitions = _target()
    adapter = build_pymc_product_space_model(target, partitions)

    with adapter.model:
        root_step = cast(
            DyadicPartitionStep,
            DyadicPartitionStep(
                adapter.partition_index,
                tree=target.tree,
                partitions=partitions,
                model=adapter.model,
                rng=0,
            ),
        )
        middle_step = cast(
            DyadicPartitionStep,
            DyadicPartitionStep(
                adapter.partition_index,
                tree=target.tree,
                partitions=partitions,
                model=adapter.model,
                rng=0,
            ),
        )

    root = DictToArrayBijection.map({"partition_index": np.array(0, dtype=np.int64)})
    middle = DictToArrayBijection.map({"partition_index": np.array(1, dtype=np.int64)})

    def logp(point: Any) -> float:
        """Return the partition index as a controlled test log density."""
        return float(point.data[0])

    _, root_stats = root_step.astep(root, logp)
    _, middle_stats = middle_step.astep(middle, logp)

    assert root_stats[0]["log_acceptance_ratio"] == pytest.approx(1.0 - math.log(2.0))
    assert middle_stats[0]["log_acceptance_ratio"] == pytest.approx(-1.0 + math.log(2.0))


def test_step_factory_assigns_disjoint_variables_in_partition_first_order() -> None:
    """The custom step should own only the index and precede native NUTS."""
    target, partitions = _target()
    adapter = build_pymc_product_space_model(target, partitions)

    partition_step, nuts_step = adapter.step_methods(
        partition_rng=5,
        nuts_kwargs={"target_accept": 0.85},
    )
    compound = pm.CompoundStep([partition_step, nuts_step])

    assert compound.methods == [partition_step, nuts_step]
    partition_names = {variable.name for variable in partition_step.vars}
    nuts_names = {variable.name for variable in nuts_step.vars}
    assert partition_names == {"partition_index"}
    assert nuts_names == {"inner_whitened", "outer_whitened"}
    assert partition_names.isdisjoint(nuts_names)


def test_native_nuts_compound_sampler_matches_exact_partition_oracle() -> None:
    """A tuned native-NUTS run should recover the exact tiny partition marginal."""
    target, partitions = _target(observation=1.8)
    expected = target.partition_probabilities(partitions)
    adapter = build_pymc_product_space_model(target, partitions)
    steps = adapter.step_methods(
        partition_rng=20260717,
        nuts_kwargs={"target_accept": 0.85},
    )

    with adapter.model:
        trace: Any = pm.sample(
            draws=2_000,
            tune=1_000,
            chains=1,
            cores=1,
            step=list(steps),
            random_seed=20260717,
            progressbar=False,
            compute_convergence_checks=False,
            return_inferencedata=True,
        )

    values = np.asarray(trace.posterior["partition_index"]).reshape(-1)
    observed = np.bincount(values, minlength=len(partitions)) / values.size
    np.testing.assert_allclose(observed, list(expected.values()), atol=0.06)
    assert trace.posterior["inner_whitened"].shape[-1] == 3
    assert trace.posterior["outer_whitened"].shape[-1] == 1
    assert any(name.endswith("accepted") for name in trace.sample_stats)
    assert "step_size" in trace.sample_stats


def test_builder_rejects_partition_dependent_whitening_and_open_subsets() -> None:
    """The first adapter should require equal pseudo-priors and a closed state space."""
    unequal_target, partitions = _target(pseudo_prior_scale=2.0)

    with pytest.raises(ValueError, match="pseudo-prior variances"):
        build_pymc_product_space_model(unequal_target, partitions)

    target, partitions = _target()
    with pytest.raises(ValueError, match="closed"):
        build_pymc_product_space_model(target, partitions[:2])


def test_partition_prior_keeps_tiny_positive_mass_distinct_from_zero() -> None:
    """Finite log weights should not underflow into zero-mass partitions."""
    target, partitions = _target(partition_log_weights=(0.0, -200.0, -math.inf))
    adapter = build_pymc_product_space_model(target, partitions)
    logp = adapter.model.compile_logp()
    point = adapter.model.initial_point()

    values = []
    for index in range(len(partitions)):
        point["partition_index"] = np.asarray(index, dtype=point["partition_index"].dtype)
        values.append(float(logp(point)))

    assert math.isfinite(values[0])
    assert math.isfinite(values[1])
    assert values[1] - values[0] == pytest.approx(-200.0, abs=2e-5)
    assert values[2] == -math.inf


def test_builder_rejects_unrepresentable_finite_log_prior_spread() -> None:
    """Extreme finite log weights must not silently become zero prior mass."""
    target, partitions = _target(partition_log_weights=(1e308, -1e308, -math.inf))

    with pytest.raises(ValueError, match="model dtype range"):
        build_pymc_product_space_model(target, partitions)


def test_builder_omits_outer_variable_for_empty_outer_block() -> None:
    """A target without outer regions should not construct a shape-zero random variable."""
    tree = DyadicTree.from_shape((1, 2))
    partitions = enumerate_partitions(tree)
    target = GaussianProductSpaceTarget.from_grid(
        observations=[0.2],
        inner_grid_design=[[[1.0, 2.0]]],
        tree=tree,
        observation_covariance=[[1.0]],
    )

    adapter = build_pymc_product_space_model(target, partitions)

    assert adapter.outer_whitened is None
    assert "outer_whitened" not in adapter.model.initial_point()
    partition, inner, outer = adapter.physical_state(0, [0.0, 0.0])
    assert partition == partitions[0]
    assert inner.shape == (2,)
    assert outer.shape == (0,)
