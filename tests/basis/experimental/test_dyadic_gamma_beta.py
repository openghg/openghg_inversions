"""Tests for grouped positive Gamma--Beta priors on masked dyadic trees."""

from __future__ import annotations

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.gamma_beta import (
    DepthKappaStrategy,
    GammaBetaForest,
    GammaBetaGroupSpec,
    GammaBetaSplitContext,
    MomentSplitConstraint,
    gamma_beta_child_moments,
)


class _RecordingKappaStrategy:
    """Record split contexts while returning one fixed concentration."""

    def __init__(self, kappa: float) -> None:
        """Initialize the strategy with a concentration to return."""
        self.kappa = kappa
        self.contexts: list[GammaBetaSplitContext] = []

    def __call__(self, context: GammaBetaSplitContext) -> float:
        """Record ``context`` and return the configured concentration."""
        self.contexts.append(context)
        return self.kappa


class _FailIfCalledKappaStrategy:
    """Fail when a fixed group incorrectly requests a split concentration."""

    def __call__(self, context: GammaBetaSplitContext) -> float:
        """Raise because fixed groups have no stochastic Beta splits."""
        raise AssertionError(f"Kappa strategy was called for fixed node {context.node_id}.")


def _single_group_forest(
    expected_mass: np.ndarray,
    *,
    root_variance: float = 0.0,
    max_depth: int = 1,
    target_regions: int | None = None,
    partition_weight: np.ndarray | None = None,
) -> GammaBetaForest:
    """Return a fully covered forest containing one hard group."""
    group = GammaBetaGroupSpec(
        name="group",
        mask=np.ones(expected_mass.shape, dtype=bool),
        root_variance=root_variance,
        max_depth=max_depth,
        target_regions=target_regions,
    )
    return GammaBetaForest.from_groups(
        expected_mass,
        [group],
        partition_weight=partition_weight,
        require_full_coverage=True,
    )


def test_depth_kappa_strategy_uses_effective_split_depth() -> None:
    """Compiled concentrations grow geometrically with effective tree depth."""
    forest = _single_group_forest(np.ones((1, 4)), max_depth=2)
    strategy = DepthKappaStrategy(base_kappa=2.0, depth_multiplier=3.0)

    samples = forest.sample(3, kappa_strategy=strategy, rng=1)

    internal_nodes = [node for node in forest.nodes if node.child_ids]
    assert {node.depth for node in internal_nodes} == {0, 1}
    for node in internal_nodes:
        assert samples.kappa_by_node[node.node_id] == pytest.approx(2.0 * 3.0**node.depth)
    assert np.isnan(samples.kappa_by_node[list(forest.leaf_ids)]).all()


def test_depth_kappa_cap_is_applied_before_exponentiation_overflow() -> None:
    """A finite cap handles extreme depths without first overflowing."""
    context = GammaBetaSplitContext(
        node_id=0,
        group_name="group",
        depth=10_000,
        geometric_depth=10_000,
        parent_expected_mass=2.0,
        child_expected_masses=(1.0, 1.0),
        child_grid_cell_counts=(1, 1),
    )

    assert DepthKappaStrategy(base_kappa=2.0, depth_multiplier=2.0, max_kappa=128.0)(context) == 128.0
    with pytest.raises(ValueError, match="overflowed"):
        DepthKappaStrategy(base_kappa=2.0, depth_multiplier=2.0)(context)


def test_root_gamma_and_unequal_child_moments_match_analytic_values() -> None:
    """Root and unequal-mass child draws reproduce their analytic moments."""
    parent_variance = 0.25
    kappa = 6.0
    first_fraction = 0.25
    forest = _single_group_forest(
        np.array([[1.0, 3.0]]),
        root_variance=parent_variance,
        max_depth=1,
    )

    samples = forest.sample(
        40_000,
        kappa_strategy=DepthKappaStrategy(base_kappa=kappa, depth_multiplier=1.0),
        rng=20260718,
    )
    root_id = forest.root_ids[0]
    first_id, second_id = forest.nodes[root_id].child_ids
    root = samples.node_scalings[:, root_id]
    first = samples.node_scalings[:, first_id]
    second = samples.node_scalings[:, second_id]
    analytic = gamma_beta_child_moments(
        parent_variance=parent_variance,
        first_expected_fraction=first_fraction,
        kappa=kappa,
    )
    analytic_covariance = samples.analytic_leaf_covariance()

    assert np.mean(root) == pytest.approx(1.0, abs=0.01)
    assert np.var(root, ddof=1) == pytest.approx(parent_variance, abs=0.012)
    np.testing.assert_allclose([np.mean(first), np.mean(second)], 1.0, atol=0.02)
    np.testing.assert_allclose(
        [np.var(first, ddof=1), np.var(second, ddof=1)],
        [analytic.first_variance, analytic.second_variance],
        rtol=0.06,
        atol=0.015,
    )
    assert np.cov(first, second, ddof=1)[0, 1] == pytest.approx(analytic.covariance, abs=0.015)
    np.testing.assert_allclose(
        analytic_covariance,
        np.array(
            [
                [analytic.first_variance, analytic.covariance],
                [analytic.covariance, analytic.second_variance],
            ]
        ),
    )


def test_equal_grid_distance_does_not_imply_equal_tree_covariance() -> None:
    """Dyadic ancestry distinguishes equally separated neighbouring regions."""
    forest = _single_group_forest(
        np.ones((1, 4)),
        root_variance=0.25,
        max_depth=2,
    )
    samples = forest.sample(
        2,
        kappa_strategy=DepthKappaStrategy(base_kappa=4.0, depth_multiplier=1.0),
        rng=1,
    )

    covariance = samples.analytic_leaf_covariance()

    assert covariance[0, 1] != pytest.approx(covariance[1, 2])


def test_sampled_splits_conserve_additive_mass_exactly() -> None:
    """Every sampled parent total equals the sum of its two child totals."""
    expected_mass = np.arange(1.0, 13.0).reshape(3, 4)
    forest = _single_group_forest(expected_mass, root_variance=0.5, max_depth=3)

    samples = forest.sample(
        257,
        kappa_strategy=DepthKappaStrategy(base_kappa=2.5, depth_multiplier=1.7),
        rng=7,
    )

    assert samples.maximum_conservation_error() < 1.0e-12


def test_multilevel_analytic_leaf_covariance_matches_prior_simulation() -> None:
    """Exact tree covariance agrees with a multilevel Monte Carlo estimate."""
    forest = _single_group_forest(
        np.array([[1.0, 2.0, 3.0, 5.0]]),
        root_variance=0.4,
        max_depth=2,
    )
    samples = forest.sample(
        80_000,
        kappa_strategy=DepthKappaStrategy(base_kappa=3.0, depth_multiplier=2.0),
        rng=219,
    )

    empirical = np.cov(samples.node_scalings[:, forest.leaf_ids], rowvar=False, ddof=1)

    np.testing.assert_allclose(
        empirical,
        samples.analytic_leaf_covariance(),
        rtol=0.045,
        atol=0.025,
    )


def test_disconnected_components_share_root_scaling_without_shared_beta_split() -> None:
    """Disconnected supports share a group scale but have separate split roots."""
    mask = np.array([[True, True, False, True, True]])
    group = GammaBetaGroupSpec(
        name="islands",
        mask=mask,
        root_variance=0.4,
        max_depth=1,
    )
    forest = GammaBetaForest.from_groups(np.ones(mask.shape), [group])
    strategy = _RecordingKappaStrategy(4.0)

    samples = forest.sample(512, kappa_strategy=strategy, rng=81)

    assert len(forest.root_ids) == 2
    first_root, second_root = forest.root_ids
    assert forest.nodes[first_root].parent_id is None
    assert forest.nodes[second_root].parent_id is None
    assert set(forest.nodes[first_root].flat_indices) == {0, 1}
    assert set(forest.nodes[second_root].flat_indices) == {3, 4}
    np.testing.assert_array_equal(
        samples.node_scalings[:, first_root],
        samples.node_scalings[:, second_root],
    )
    assert {context.node_id for context in strategy.contexts} == {first_root, second_root}
    assert all(context.child_expected_masses == (1.0, 1.0) for context in strategy.contexts)
    assert not np.array_equal(
        samples.split_fractions[:, first_root],
        samples.split_fractions[:, second_root],
    )


def test_disconnected_terminal_components_have_shared_root_covariance() -> None:
    """Unsplit components in one semantic group have perfectly shared root variation."""
    mask = np.array([[True, False, True]])
    group = GammaBetaGroupSpec(
        name="islands",
        mask=mask,
        root_variance=0.4,
        max_depth=1,
        target_regions=2,
    )
    forest = GammaBetaForest.from_groups(np.ones(mask.shape), [group])

    samples = forest.sample(4, kappa_strategy=_FailIfCalledKappaStrategy(), rng=2)

    np.testing.assert_allclose(
        samples.analytic_leaf_covariance(),
        np.full((2, 2), 0.4),
    )


def test_independent_group_roots_have_zero_cross_group_covariance() -> None:
    """Leaf scalings in separate semantic groups have zero prior covariance."""
    first_mask = np.array([[True, False]])
    second_mask = ~first_mask
    groups = [
        GammaBetaGroupSpec("first", first_mask, root_variance=0.5),
        GammaBetaGroupSpec("second", second_mask, root_variance=0.2),
    ]
    forest = GammaBetaForest.from_groups(
        np.ones((1, 2)),
        groups,
        require_full_coverage=True,
    )

    samples = forest.sample(4, kappa_strategy=_FailIfCalledKappaStrategy(), rng=3)

    np.testing.assert_allclose(
        samples.analytic_leaf_covariance(),
        np.diag([0.5, 0.2]),
    )


def test_fixed_groups_do_not_invoke_kappa_strategy() -> None:
    """A fixed group samples only its root and never asks for a kappa value."""
    forest = _single_group_forest(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        root_variance=0.2,
        max_depth=0,
    )

    samples = forest.sample(16, kappa_strategy=_FailIfCalledKappaStrategy(), rng=12)

    assert forest.root_ids == forest.leaf_ids
    assert np.isnan(samples.kappa_by_node).all()
    assert np.isnan(samples.split_fractions).all()
    assert np.isfinite(samples.node_scalings).all()


def test_zero_mass_child_stops_that_split_without_losing_support() -> None:
    """A zero-mass child leaves its parent terminal and avoids invalid Beta shapes."""
    expected_mass = np.array([[0.0, 1.0, 2.0, 3.0]])
    forest = _single_group_forest(expected_mass, root_variance=0.5, max_depth=2)
    strategy = _RecordingKappaStrategy(3.0)

    samples = forest.sample(32, kappa_strategy=strategy, rng=13)

    labels = forest.leaf_labels()
    assert np.all(labels > 0)
    assert np.isfinite(samples.node_scalings).all()
    assert samples.maximum_conservation_error() < 1.0e-12
    assert len(strategy.contexts) == 2


def test_hard_group_labels_cover_grid_without_crossing_boundaries() -> None:
    """Fixed outer and refinable land/ocean leaves label each grid point once."""
    outer = np.zeros((3, 4), dtype=bool)
    outer[0, :] = True
    inner_land = np.zeros_like(outer)
    inner_land[1:, :2] = True
    inner_ocean = np.zeros_like(outer)
    inner_ocean[1:, 2:] = True
    groups = [
        GammaBetaGroupSpec("outer", outer, max_depth=0),
        GammaBetaGroupSpec("inner_land", inner_land, max_depth=2),
        GammaBetaGroupSpec("inner_ocean", inner_ocean, max_depth=2),
    ]
    forest = GammaBetaForest.from_groups(
        np.arange(1.0, 13.0).reshape(3, 4),
        groups,
        require_full_coverage=True,
    )

    labels = forest.leaf_labels()
    expected_group = np.full(forest.shape, -1, dtype=np.int64)
    for group_index, group in enumerate(groups):
        expected_group[group.mask] = group_index

    assert set(np.unique(labels)) == set(range(1, len(forest.leaf_ids) + 1))
    assert np.all(labels > 0)
    outer_leaves = [node_id for node_id in forest.leaf_ids if forest.nodes[node_id].group_index == 0]
    assert len(outer_leaves) == 1
    for label, node_id in enumerate(forest.leaf_ids, start=1):
        node = forest.nodes[node_id]
        label_groups = expected_group.reshape(-1)[labels.reshape(-1) == label]
        assert set(label_groups) == {node.group_index}
        np.testing.assert_array_equal(
            np.flatnonzero(labels.reshape(-1) == label),
            node.flat_indices,
        )


def test_weighted_region_budget_reaches_exact_target_and_refines_high_weight_branch() -> None:
    """Best-first refinement spends an exact budget on the heavier branch."""
    expected_mass = np.ones((4, 4))
    partition_weight = np.ones((4, 4))
    partition_weight[:2, :] = 10.0
    forest = _single_group_forest(
        expected_mass,
        max_depth=2,
        target_regions=3,
        partition_weight=partition_weight,
    )

    labels = forest.leaf_labels()

    assert len(forest.leaf_ids) == 3
    assert np.unique(labels[:2, :]).size == 2
    assert np.unique(labels[2:, :]).size == 1
    assert forest.partition_weight.flags.writeable is False


def test_moment_constraint_skips_invalid_high_priority_split() -> None:
    """The priority queue continues after rejecting an unstable refinement."""
    expected_mass = np.array([[0.001, 0.999, 1.0, 1.0]])
    partition_weight = np.array([[100.0, 100.0, 1.0, 1.0]])
    group = GammaBetaGroupSpec(
        "group",
        np.ones(expected_mass.shape, dtype=bool),
        root_variance=0.0,
        max_depth=2,
        target_regions=3,
    )
    strategy = DepthKappaStrategy(base_kappa=2.0, depth_multiplier=1.0)

    forest = GammaBetaForest.from_groups(
        expected_mass,
        [group],
        partition_weight=partition_weight,
        kappa_strategy=strategy,
        split_constraint=MomentSplitConstraint(
            min_beta_shape=0.1,
            max_child_variance=None,
        ),
        require_full_coverage=True,
    )

    labels = forest.leaf_labels()
    assert len(forest.leaf_ids) == 3
    assert np.unique(labels[:, :2]).size == 1
    assert np.unique(labels[:, 2:]).size == 2


def test_moment_constraint_can_stop_below_requested_region_budget() -> None:
    """An admissibility failure can leave a stable topology below target K."""
    expected_mass = np.array([[1.0, 99.0]])
    group = GammaBetaGroupSpec(
        "group",
        np.ones(expected_mass.shape, dtype=bool),
        root_variance=0.0,
        max_depth=1,
        target_regions=2,
    )
    strategy = DepthKappaStrategy(base_kappa=2.0, depth_multiplier=1.0)

    forest = GammaBetaForest.from_groups(
        expected_mass,
        [group],
        kappa_strategy=strategy,
        split_constraint=MomentSplitConstraint(
            min_beta_shape=None,
            max_child_variance=9.0,
        ),
        require_full_coverage=True,
    )

    assert len(forest.leaf_ids) == 1

    with pytest.raises(ValueError, match="only 1 satisfy the split constraints"):
        GammaBetaForest.from_groups(
            expected_mass,
            [group],
            kappa_strategy=strategy,
            split_constraint=MomentSplitConstraint(
                min_beta_shape=None,
                max_child_variance=9.0,
                allow_fewer_regions=False,
            ),
            require_full_coverage=True,
        )


def test_moment_constraint_propagates_parent_variance_to_grandchildren() -> None:
    """Grandchild admissibility uses exact variance inherited from its parent."""
    group = GammaBetaGroupSpec(
        "group",
        np.ones((1, 4), dtype=bool),
        root_variance=0.0,
        max_depth=2,
        target_regions=4,
    )
    strategy = DepthKappaStrategy(base_kappa=4.0, depth_multiplier=1.0)

    forest = GammaBetaForest.from_groups(
        np.ones((1, 4)),
        [group],
        kappa_strategy=strategy,
        split_constraint=MomentSplitConstraint(
            min_beta_shape=None,
            max_child_variance=0.3,
        ),
        require_full_coverage=True,
    )

    # The root children have variance 0.2 and pass. Their children would have
    # variance 0.44, so both second-level refinements must stop.
    assert len(forest.leaf_ids) == 2


def test_moment_constraint_requires_concentration_policy() -> None:
    """Forest construction cannot evaluate moment limits without kappa."""
    group = GammaBetaGroupSpec(
        "group",
        np.ones((1, 2), dtype=bool),
        max_depth=1,
        target_regions=2,
    )

    with pytest.raises(ValueError, match="kappa_strategy is required"):
        GammaBetaForest.from_groups(
            np.ones((1, 2)),
            [group],
            split_constraint=MomentSplitConstraint(),
        )


def test_region_budget_rejects_targets_outside_component_and_depth_limits() -> None:
    """An exact budget must fit both disconnected roots and candidate leaves."""
    disconnected = np.array([[True, False, True]])
    too_few = GammaBetaGroupSpec(
        "islands",
        disconnected,
        max_depth=1,
        target_regions=1,
    )
    with pytest.raises(ValueError, match="at least 2 regions"):
        GammaBetaForest.from_groups(np.ones(disconnected.shape), [too_few])

    too_many = GammaBetaGroupSpec(
        "group",
        np.ones((2, 2), dtype=bool),
        max_depth=1,
        target_regions=3,
    )
    with pytest.raises(ValueError, match="max_depth permits only 2"):
        GammaBetaForest.from_groups(np.ones((2, 2)), [too_many])


@pytest.mark.parametrize("target_regions", [0, -1, 1.5, True])
def test_group_rejects_invalid_region_budget(target_regions: object) -> None:
    """Group specifications require a positive integer region budget."""
    error = TypeError if isinstance(target_regions, (float, bool)) else ValueError
    with pytest.raises(error, match="target_regions"):
        GammaBetaGroupSpec(
            "group",
            np.ones((1, 2), dtype=bool),
            max_depth=1,
            target_regions=target_regions,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_kappa", [0.0, -1.0, np.nan, np.inf])
def test_sampling_rejects_invalid_kappa_strategy_results(invalid_kappa: float) -> None:
    """Sampling rejects every non-positive or non-finite strategy result."""
    forest = _single_group_forest(np.ones((1, 2)), max_depth=1)

    def invalid_strategy(context: GammaBetaSplitContext) -> float:
        """Return the invalid concentration supplied by the test case."""
        del context
        return invalid_kappa

    with pytest.raises(ValueError, match="Kappa strategy returned invalid value"):
        forest.sample(1, kappa_strategy=invalid_strategy, rng=0)


@pytest.mark.parametrize("invalid_mass", [-1.0, np.nan, np.inf])
def test_forest_rejects_negative_or_nonfinite_expected_mass(invalid_mass: float) -> None:
    """Forest construction rejects negative and non-finite additive mass."""
    expected_mass = np.ones((2, 2))
    expected_mass[0, 0] = invalid_mass
    group = GammaBetaGroupSpec("group", np.ones(expected_mass.shape, dtype=bool))

    with pytest.raises(ValueError, match="expected_mass must be finite and non-negative"):
        GammaBetaForest.from_groups(expected_mass, [group])


def test_forest_rejects_non_matrix_expected_mass() -> None:
    """Forest construction requires expected mass to be two-dimensional."""
    group = GammaBetaGroupSpec("group", np.ones((1, 2), dtype=bool))

    with pytest.raises(ValueError, match="expected_mass must be two-dimensional"):
        GammaBetaForest.from_groups(np.ones(2), [group])


@pytest.mark.parametrize("invalid_weight", [-1.0, np.nan, np.inf])
def test_forest_rejects_invalid_partition_weight(invalid_weight: float) -> None:
    """Topology-selection weights must be finite and non-negative."""
    partition_weight = np.ones((2, 2))
    partition_weight[0, 0] = invalid_weight
    group = GammaBetaGroupSpec("group", np.ones((2, 2), dtype=bool))

    with pytest.raises(ValueError, match="partition_weight must be finite and non-negative"):
        GammaBetaForest.from_groups(
            np.ones((2, 2)),
            [group],
            partition_weight=partition_weight,
        )


def test_forest_rejects_aggregate_mass_overflow() -> None:
    """Finite elements whose group sum overflows receive a contextual error."""
    expected_mass = np.full((1, 2), np.finfo(np.float64).max)
    group = GammaBetaGroupSpec("group", np.ones(expected_mass.shape, dtype=bool))

    with pytest.raises(ValueError, match="Additive sum overflowed"):
        GammaBetaForest.from_groups(expected_mass, [group])


def test_sampling_rejects_unrepresentable_child_mass_fraction() -> None:
    """An extreme positive mass ratio cannot reach NumPy as a zero Beta shape."""
    forest = _single_group_forest(
        np.array([[np.nextafter(0.0, 1.0), 1.0e308]]),
        max_depth=1,
    )

    with pytest.raises(ValueError, match="cannot be represented as positive finite Beta shapes"):
        forest.sample(
            2,
            kappa_strategy=DepthKappaStrategy(base_kappa=2.0),
            rng=3,
        )


def test_experimental_package_reexports_gamma_beta_api() -> None:
    """The provisional package exposes the documented Gamma-Beta entry points."""
    from openghg_inversions.basis.experimental import dyadic

    assert dyadic.GammaBetaForest is GammaBetaForest
    assert dyadic.DepthKappaStrategy is DepthKappaStrategy
