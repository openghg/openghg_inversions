"""Tests for the immutable fixed-tree Gamma--Beta reference model."""

from __future__ import annotations

from math import log

import numpy as np
import pytest
from scipy import stats

from openghg_inversions.experimental.rjmcmc.core import FixedDesignBlock
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
    assert split.fixed_coefficients.shape == (0,)
    np.testing.assert_array_equal(split.dynamic_prediction, split.prediction)
    np.testing.assert_array_equal(split.fixed_prediction, np.zeros(2))
    assert split.log_fixed_coefficient_prior == 0.0
    assert split.log_target == pytest.approx(
        split.log_likelihood + split.log_root_prior + split.log_fraction_prior + split.log_partition_prior
    )


def test_active_target_matches_closed_form_product_space_oracle() -> None:
    """Mass coordinates match the archived scaling oracle after its Jacobian."""
    tree = CanonicalDyadicTree.from_shape((1, 2))
    nominal_mass = np.array([1.0, 3.0])
    expected_total = 4.0
    root_scaling_variance = 0.25
    root_scaling = 1.2
    root_total = expected_total * root_scaling
    split_fraction = 0.35
    per_mass_design = np.array([[0.4, 1.1], [-0.7, 0.3]])
    observations = np.array([4.2, -0.1])
    observation_sd = np.array([0.8, 1.3])
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        nominal_mass,
        kappa=4.0,
        root_mean=expected_total,
        root_variance=expected_total**2 * root_scaling_variance,
    )
    problem = GammaBetaTreeProblem(
        observations=observations,
        observation_sd=observation_sd,
        sensitivity=per_mass_design,
        prior=prior,
        partition_prior=TreePartitionPrior.uniform_k(tree),
    )
    coarse = build_gamma_beta_tree_state(
        problem,
        frontier=DyadicFrontier.root(tree),
        root_total=root_total,
        active_fractions=[],
    )
    split = build_gamma_beta_tree_state(
        problem,
        frontier=coarse.frontier.split(tree, tree.root_id),
        root_total=root_total,
        active_fractions=[split_fraction],
    )

    finest_scaling_design = per_mass_design * nominal_mass[np.newaxis, :]
    expected_fraction = nominal_mass[0] / expected_total
    archived_coarse_prediction = root_scaling * finest_scaling_design.sum(axis=1)
    archived_split_prediction = finest_scaling_design @ np.array(
        [
            root_scaling * split_fraction / expected_fraction,
            root_scaling * (1.0 - split_fraction) / (1.0 - expected_fraction),
        ]
    )
    np.testing.assert_allclose(coarse.prediction, archived_coarse_prediction)
    np.testing.assert_allclose(split.prediction, archived_split_prediction)

    archived_root_log_density = stats.gamma.logpdf(
        root_scaling,
        a=1.0 / root_scaling_variance,
        scale=root_scaling_variance,
    )
    mass_coordinate_root_log_density = archived_root_log_density - log(expected_total)
    inactive_beta_log_marginal = log(stats.beta.cdf(1.0, a=1.0, b=3.0) - stats.beta.cdf(0.0, a=1.0, b=3.0))
    coarse_oracle = (
        stats.norm.logpdf(
            observations,
            loc=archived_coarse_prediction,
            scale=observation_sd,
        ).sum()
        + mass_coordinate_root_log_density
        + log(0.5)
        + inactive_beta_log_marginal
    )
    split_oracle = (
        stats.norm.logpdf(
            observations,
            loc=archived_split_prediction,
            scale=observation_sd,
        ).sum()
        + mass_coordinate_root_log_density
        + stats.beta.logpdf(split_fraction, a=1.0, b=3.0)
        + log(0.5)
    )
    assert coarse.log_root_prior == pytest.approx(mass_coordinate_root_log_density)
    assert coarse.log_target == pytest.approx(coarse_oracle)
    assert split.log_target == pytest.approx(split_oracle)


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


def test_fixed_offset_and_outer_block_close_total_prediction_and_target() -> None:
    """Fixed offset and inferred outer coefficients close prediction and prior."""
    base = _two_cell_problem()
    fixed_offset = np.array([0.2, -0.1])
    fixed_design = np.array([[1.0, 2.0], [-1.0, 0.5]])
    fixed_prior_mean = np.array([1.0, 2.0])
    fixed_prior_sd = np.array([0.5, 0.25])
    fixed_coefficients = np.array([0.8, 1.2])
    fixed_block = FixedDesignBlock(
        design=fixed_design,
        coefficient_prior_mean=fixed_prior_mean,
        coefficient_prior_sd=fixed_prior_sd,
    )
    problem = GammaBetaTreeProblem(
        observations=base.observations,
        observation_sd=base.observation_sd,
        sensitivity=base.sensitivity,
        prior=base.prior,
        partition_prior=base.partition_prior,
        fixed_offset=fixed_offset,
        fixed_block=fixed_block,
    )
    state = build_gamma_beta_tree_state(
        problem,
        frontier=DyadicFrontier.root(problem.tree),
        root_total=4.0,
        active_fractions=[],
        fixed_coefficients=fixed_coefficients,
    )

    expected_dynamic = np.array([2.0, 2.0])
    expected_fixed = fixed_offset.copy() + fixed_design @ fixed_coefficients
    expected_log_prior = 0.0
    for coefficient, mean, standard_deviation in zip(
        fixed_coefficients,
        fixed_prior_mean,
        fixed_prior_sd,
        strict=True,
    ):
        sigma = np.sqrt(np.log1p((standard_deviation / mean) ** 2))
        mu = np.log(mean) - 0.5 * sigma**2
        expected_log_prior += stats.lognorm.logpdf(coefficient, s=sigma, scale=np.exp(mu))

    fixed_offset[:] = 99.0
    fixed_design[:] = 99.0
    fixed_prior_mean[:] = 99.0
    fixed_prior_sd[:] = 99.0
    fixed_coefficients[:] = 99.0
    np.testing.assert_allclose(state.dynamic_prediction, expected_dynamic)
    np.testing.assert_allclose(state.fixed_prediction, expected_fixed)
    np.testing.assert_allclose(state.prediction, expected_dynamic + expected_fixed)
    np.testing.assert_allclose(state.residual, state.prediction - problem.observations)
    assert problem.n_fixed_coefficients == 2
    assert state.log_fixed_coefficient_prior == pytest.approx(expected_log_prior)
    assert problem.fixed_offset is not None
    for array in (
        problem.fixed_offset,
        fixed_block.design,
        fixed_block.coefficient_prior_mean,
        fixed_block.coefficient_prior_sd,
        state.fixed_coefficients,
        state.dynamic_prediction,
        state.fixed_prediction,
        state.prediction,
        state.residual,
    ):
        assert not array.flags.writeable
    assert state.log_target == pytest.approx(
        state.log_likelihood
        + state.log_root_prior
        + state.log_fraction_prior
        + state.log_partition_prior
        + state.log_fixed_coefficient_prior
    )


def test_node_design_bottom_up_aggregation_matches_direct_descendant_scan() -> None:
    """Bottom-up node columns equal direct nominal-weighted descendant sums."""
    rng = np.random.default_rng(812)
    tree = CanonicalDyadicTree.from_shape((3, 5))
    nominal = rng.lognormal(size=15)
    sensitivity = rng.normal(size=(4, 15))
    prior = GammaBetaTreePrior.constant_concentration(
        tree,
        nominal,
        concentration=3.0,
        root_shape=2.0,
        root_rate=1.0,
    )
    problem = GammaBetaTreeProblem(
        observations=np.zeros(4),
        observation_sd=np.ones(4),
        sensitivity=sensitivity,
        prior=prior,
        partition_prior=TreePartitionPrior.uniform_k(tree, maximum_k=5),
    )

    for node in tree.nodes:
        indices = np.asarray(node.cell_indices, dtype=np.int64)
        direct_mass = float(nominal[indices].sum())
        direct_design = sensitivity[:, indices] @ (nominal[indices] / direct_mass)
        assert problem.node_nominal_mass[node.node_id] == pytest.approx(direct_mass)
        np.testing.assert_allclose(
            problem.node_design[:, node.node_id],
            direct_design,
            rtol=2.0e-15,
            atol=2.0e-15,
        )


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
    assert problem.fixed_offset is not None
    arrays = (
        problem.observations,
        problem.observation_sd,
        problem.sensitivity,
        problem.node_nominal_mass,
        problem.node_design,
        problem.fixed_offset,
        problem.prior.nominal_cell_mass,
        problem.prior.beta_shape_by_node,
        problem.partition_prior.p_k,
        state.active_fractions,
        state.active_node_masses,
        state.fixed_coefficients,
        state.dynamic_prediction,
        state.fixed_prediction,
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
    with pytest.raises(ValueError, match="fixed_offset"):
        GammaBetaTreeProblem(
            observations=np.array([0.0, 0.0]),
            observation_sd=np.array([1.0, 1.0]),
            sensitivity=np.eye(2),
            prior=problem.prior,
            partition_prior=problem.partition_prior,
            fixed_offset=np.zeros(3),
        )
    with pytest.raises(ValueError, match="one row per observation"):
        GammaBetaTreeProblem(
            observations=np.array([0.0, 0.0]),
            observation_sd=np.array([1.0, 1.0]),
            sensitivity=np.eye(2),
            prior=problem.prior,
            partition_prior=problem.partition_prior,
            fixed_block=FixedDesignBlock(
                design=np.ones((3, 1)),
                coefficient_prior_mean=np.ones(1),
                coefficient_prior_sd=np.ones(1),
            ),
        )


def test_fixed_coefficient_builder_contract_rejects_malformed_state() -> None:
    """Fixed coefficients are required exactly when a fixed block is present."""
    base = _two_cell_problem()
    fixed_problem = GammaBetaTreeProblem(
        observations=base.observations,
        observation_sd=base.observation_sd,
        sensitivity=base.sensitivity,
        prior=base.prior,
        partition_prior=base.partition_prior,
        fixed_block=FixedDesignBlock(
            design=np.ones((2, 2)),
            coefficient_prior_mean=np.ones(2),
            coefficient_prior_sd=np.ones(2),
        ),
    )
    frontier = DyadicFrontier.root(base.tree)

    with pytest.raises(ValueError, match="required"):
        build_gamma_beta_tree_state(
            fixed_problem,
            frontier=frontier,
            root_total=1.0,
            active_fractions=[],
        )
    with pytest.raises(ValueError, match="shape"):
        build_gamma_beta_tree_state(
            fixed_problem,
            frontier=frontier,
            root_total=1.0,
            active_fractions=[],
            fixed_coefficients=[1.0],
        )
    with pytest.raises(ValueError, match="strictly positive"):
        build_gamma_beta_tree_state(
            fixed_problem,
            frontier=frontier,
            root_total=1.0,
            active_fractions=[],
            fixed_coefficients=[1.0, 0.0],
        )
    with pytest.raises(ValueError, match="configured fixed_block"):
        build_gamma_beta_tree_state(
            base,
            frontier=frontier,
            root_total=1.0,
            active_fractions=[],
            fixed_coefficients=[1.0],
        )


def test_density_helpers_return_negative_infinity_outside_coordinate_support() -> None:
    """Root Gamma and split Beta density helpers use negative infinity off support."""
    prior = _two_cell_problem().prior

    assert prior.log_root_density(0.0) == -np.inf
    assert prior.log_root_density(np.inf) == -np.inf
    assert prior.log_fraction_density(prior.tree.root_id, 0.0) == -np.inf
    assert prior.log_fraction_density(prior.tree.root_id, 1.0) == -np.inf
