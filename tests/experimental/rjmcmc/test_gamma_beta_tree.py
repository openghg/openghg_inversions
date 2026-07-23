"""Tests for the immutable fixed-tree Gamma--Beta reference model."""

from __future__ import annotations

from math import log

import numpy as np
import pytest
from scipy import stats

from openghg_inversions.experimental.rjmcmc.dyadic_tree import (
    CanonicalDyadicTree,
    DyadicFrontier,
    enumerate_frontiers,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_tree import (
    GammaBetaTreePrior,
    GammaBetaTreeProblem,
    TreePartitionPrior,
    build_gamma_beta_tree_state,
    render_cell_mass,
)


def _two_cell_problem(*, likelihood_power: float = 1.0) -> GammaBetaTreeProblem:
    """Return an identity-sensitivity problem on one two-cell tree."""
    tree = CanonicalDyadicTree.from_shape((1, 2))
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        np.ones(2),
        concentration=2.0,
        root_mean=1.0,
        root_variance=0.25,
    )
    return GammaBetaTreeProblem(
        observations=np.zeros(2),
        observation_sd=np.array([0.5, 2.0]),
        sensitivity=np.eye(2),
        prior=prior,
        partition_prior=TreePartitionPrior.uniform_k(tree),
        likelihood_power=likelihood_power,
    )


def test_normalized_gamma_and_beta_densities_match_scipy() -> None:
    """Root shape-rate Gamma and split Beta densities include normalizers."""
    problem = _two_cell_problem()
    prior = problem.prior
    root_total = 1.7
    fraction = 0.25

    assert prior.root_shape == pytest.approx(4.0)
    assert prior.root_rate == pytest.approx(4.0)
    assert prior.beta_parameters(problem.tree.root_id) == pytest.approx((1.0, 1.0))
    assert prior.log_root_density(root_total) == pytest.approx(
        stats.gamma.logpdf(root_total, a=4.0, scale=0.25)
    )
    assert prior.log_fraction_density(problem.tree.root_id, fraction) == pytest.approx(
        stats.beta.logpdf(fraction, a=1.0, b=1.0)
    )


def test_constant_concentration_uses_nominal_child_fraction() -> None:
    """Constant concentration multiplies each nominal child mass fraction."""
    tree = CanonicalDyadicTree.from_shape((1, 2))
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        [1.0, 3.0],
        kappa=8.0,
        root_shape=2.0,
        root_rate=5.0,
    )

    assert prior.beta_parameters(tree.root_id) == pytest.approx((2.0, 6.0))
    assert prior.log_root_density(1.0) == pytest.approx(stats.gamma.logpdf(1.0, a=2.0, scale=0.2))


def test_additive_gamma_beta_identity_in_root_fraction_coordinates() -> None:
    """Additive tree density equals cell-Gamma density times its Jacobian."""
    tree = CanonicalDyadicTree.from_shape((1, 2))
    alpha = np.array([2.0, 3.5])
    rate = 4.0
    prior = GammaBetaTreePrior.additive_cell_alpha(
        tree,
        nominal_cell_mass=[1.0, 1.0],
        cell_alpha=alpha,
        root_rate=rate,
    )
    cell_mass = np.array([0.7, 1.4])
    root_total = float(cell_mass.sum())
    fraction = float(cell_mass[0] / root_total)

    tree_log_density = prior.log_root_density(root_total) + prior.log_fraction_density(
        tree.root_id,
        fraction,
    )
    cell_log_density = float(stats.gamma.logpdf(cell_mass, a=alpha, scale=1.0 / rate).sum())
    assert prior.root_shape == pytest.approx(float(alpha.sum()))
    assert prior.beta_parameters(tree.root_id) == pytest.approx(tuple(alpha))
    assert tree_log_density == pytest.approx(cell_log_density + log(root_total))


def test_uniform_k_prior_is_normalized_over_all_two_by_two_frontiers() -> None:
    """Uniform K mass is divided by the exact number of frontiers at each K."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    prior = TreePartitionPrior.uniform_k(tree)
    frontiers = enumerate_frontiers(tree)

    assert prior.partition_counts == (0, 1, 1, 2, 1)
    assert prior.p_k == pytest.approx([0.0, 0.25, 0.25, 0.25, 0.25])
    probabilities = np.array([np.exp(prior.log_probability(frontier)) for frontier in frontiers])
    assert probabilities.sum() == pytest.approx(1.0)
    assert {
        len(frontier): np.exp(prior.log_probability(frontier)) for frontier in frontiers
    } == pytest.approx({1: 0.25, 2: 0.25, 3: 0.125, 4: 0.25})


def test_bounded_partition_prior_truncates_the_count_table() -> None:
    """A moderate tree computes and stores counts only through maximum K."""
    tree = CanonicalDyadicTree.from_shape((16, 16))
    prior = TreePartitionPrior.uniform_k(tree, minimum_k=2, maximum_k=5)

    assert len(prior.partition_counts) == 6
    assert prior.p_k.shape == (6,)
    assert prior.p_k[:2].tolist() == [0.0, 0.0]
    assert prior.p_k[2:].tolist() == pytest.approx([0.25] * 4)


def test_geometric_k_prior_has_normalized_successive_mass_ratio() -> None:
    """Truncated geometric K probabilities have the declared decay ratio."""
    tree = CanonicalDyadicTree.from_shape((2, 2))
    prior = TreePartitionPrior.geometric(tree, continuation_probability=0.25)

    assert prior.p_k.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(prior.p_k[2:] / prior.p_k[1:-1], 0.25)
    with pytest.raises(ValueError, match="strictly between"):
        TreePartitionPrior.geometric(tree, ratio=1.0)


def test_state_propagates_mass_and_predicts_coarse_and_split_models() -> None:
    """Two-cell coarse and split states conserve mass and give exact predictions."""
    problem = _two_cell_problem()
    coarse = build_gamma_beta_tree_state(
        problem,
        frontier=DyadicFrontier.root(problem.tree),
        root_total=4.0,
        active_fractions=[],
    )
    split_frontier = coarse.frontier.split(problem.tree, problem.tree.root_id)
    split = build_gamma_beta_tree_state(
        problem,
        frontier=split_frontier,
        root_total=4.0,
        active_fractions=[0.25],
    )

    np.testing.assert_allclose(coarse.prediction, [2.0, 2.0])
    np.testing.assert_allclose(render_cell_mass(problem, coarse), [2.0, 2.0])
    np.testing.assert_allclose(split.active_node_masses, [1.0, 3.0])
    np.testing.assert_allclose(split.prediction, [1.0, 3.0])
    np.testing.assert_allclose(render_cell_mass(problem, split), [1.0, 3.0])
    assert split.active_node_masses.sum() == pytest.approx(split.root_total)
    assert split.log_target == pytest.approx(
        split.log_likelihood + split.log_root_prior + split.log_fraction_prior + split.log_partition_prior
    )


def test_mass_sensitivity_conversion_matches_scaled_rhime_columns() -> None:
    """Dividing fp_x_flux by nominal mass recovers the scaling forward model."""
    tree = CanonicalDyadicTree.from_shape((1, 2))
    nominal_mass = np.array([1.0, 3.0])
    fp_x_flux = np.array([[2.0, 9.0]])
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        nominal_mass,
        concentration=2.0,
        root_mean=4.0,
        root_variance=1.0,
    )
    problem = GammaBetaTreeProblem(
        observations=np.zeros(1),
        observation_sd=np.ones(1),
        sensitivity=fp_x_flux / nominal_mass,
        prior=prior,
        partition_prior=TreePartitionPrior.uniform_k(tree),
    )
    scale = 2.0
    state = build_gamma_beta_tree_state(
        problem,
        frontier=DyadicFrontier.root(tree),
        root_total=scale * nominal_mass.sum(),
        active_fractions=[],
    )

    np.testing.assert_allclose(state.prediction, scale * fp_x_flux.sum(axis=1))


def test_problem_state_and_rendered_arrays_are_owned_and_read_only() -> None:
    """Public numerical arrays are copied and protected from mutation."""
    observations = np.zeros(2)
    observation_sd = np.ones(2)
    sensitivity = np.eye(2)
    nominal = np.ones(2)
    tree = CanonicalDyadicTree.from_shape((1, 2))
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        nominal,
        concentration=2.0,
        root_shape=4.0,
        root_rate=4.0,
    )
    problem = GammaBetaTreeProblem(
        observations=observations,
        observation_sd=observation_sd,
        sensitivity=sensitivity,
        prior=prior,
        partition_prior=TreePartitionPrior.uniform_k(tree),
    )
    observations[:] = 9.0
    observation_sd[:] = 9.0
    sensitivity[:] = 9.0
    nominal[:] = 9.0
    state = build_gamma_beta_tree_state(
        problem,
        frontier=DyadicFrontier.root(problem.tree),
        root_total=1.0,
        active_fractions=[],
    )
    rendered = render_cell_mass(problem, state)

    np.testing.assert_array_equal(problem.observations, [0.0, 0.0])
    np.testing.assert_array_equal(problem.observation_sd, [1.0, 1.0])
    np.testing.assert_array_equal(problem.sensitivity, np.eye(2))
    np.testing.assert_array_equal(problem.prior.nominal_cell_mass, [1.0, 1.0])
    arrays = (
        problem.observations,
        problem.observation_sd,
        problem.sensitivity,
        problem.node_nominal_mass,
        problem.node_design,
        problem.prior.nominal_cell_mass,
        problem.prior.beta_shape_by_node,
        problem.partition_prior.p_k,
        state.active_fractions,
        state.active_node_masses,
        state.prediction,
        state.residual,
        rendered,
    )
    assert all(not array.flags.writeable for array in arrays)


def test_likelihood_power_scales_only_the_gaussian_target_component() -> None:
    """Likelihood power zero preserves the raw density but removes its target term."""
    problem = _two_cell_problem(likelihood_power=0.0)
    state = build_gamma_beta_tree_state(
        problem,
        frontier=DyadicFrontier.root(problem.tree),
        root_total=4.0,
        active_fractions=[],
    )

    assert state.log_gaussian_likelihood == pytest.approx(
        stats.norm.logpdf(state.residual, loc=0.0, scale=problem.observation_sd).sum()
    )
    assert state.log_likelihood == 0.0


@pytest.mark.parametrize(
    ("root_total", "active_fractions", "match"),
    [
        (0.0, [0.5], "root_total"),
        (1.0, [], "shape"),
        (1.0, [0.0], "strictly between"),
        (1.0, [1.0], "strictly between"),
        (1.0, [np.nan], "finite"),
    ],
)
def test_state_builder_rejects_malformed_support(
    root_total: float,
    active_fractions: list[float],
    match: str,
) -> None:
    """State construction rejects root and active-fraction support violations."""
    problem = _two_cell_problem()
    split_frontier = DyadicFrontier.root(problem.tree).split(problem.tree, problem.tree.root_id)

    with pytest.raises(ValueError, match=match):
        build_gamma_beta_tree_state(
            problem,
            frontier=split_frontier,
            root_total=root_total,
            active_fractions=active_fractions,
        )


def test_problem_and_prior_reject_malformed_shapes_and_parameters() -> None:
    """Declared priors and observation models reject invalid static inputs."""
    tree = CanonicalDyadicTree.from_shape((1, 2))
    with pytest.raises(ValueError, match="strictly positive"):
        GammaBetaTreePrior.constant_concentration(
            tree,
            [1.0, 0.0],
            concentration=2.0,
            root_shape=1.0,
            root_rate=1.0,
        )
    with pytest.raises(ValueError, match="must be supplied together"):
        GammaBetaTreePrior.constant_concentration(
            tree,
            [1.0, 1.0],
            concentration=2.0,
            root_mean=1.0,
        )

    problem = _two_cell_problem()
    with pytest.raises(ValueError, match="sensitivity"):
        GammaBetaTreeProblem(
            observations=np.array([0.0, 0.0]),
            observation_sd=np.array([1.0, 1.0]),
            sensitivity=np.ones((2, 3)),
            prior=problem.prior,
            partition_prior=problem.partition_prior,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        GammaBetaTreeProblem(
            observations=np.array([0.0, 0.0]),
            observation_sd=np.array([1.0, 0.0]),
            sensitivity=np.eye(2),
            prior=problem.prior,
            partition_prior=problem.partition_prior,
        )


def test_density_helpers_return_negative_infinity_outside_coordinate_support() -> None:
    """Root Gamma and split Beta density helpers use negative infinity off support."""
    prior = _two_cell_problem().prior

    assert prior.log_root_density(0.0) == -np.inf
    assert prior.log_root_density(np.inf) == -np.inf
    assert prior.log_fraction_density(prior.tree.root_id, 0.0) == -np.inf
    assert prior.log_fraction_density(prior.tree.root_id, 1.0) == -np.inf
