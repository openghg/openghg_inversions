"""Tests for transported fixed-dimensional dyadic product-space updates."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.contrast import TreeContrastLayout
from openghg_inversions.basis.experimental.dyadic.enumeration import enumerate_partitions
from openghg_inversions.basis.experimental.dyadic.gaussian_product_space import (
    GaussianProductSpaceTarget,
)
from openghg_inversions.basis.experimental.dyadic.product_space import ProductSpaceState
from openghg_inversions.basis.experimental.dyadic.proposals import MergeMove, PairedMove, SplitMove
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.transformed_product_space import (
    AdditiveCoefficientTransform,
    GaussianContrastProposal,
    swap_contrast_coordinate,
    transported_partition_metropolis_step,
)
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree


def _one_by_three_partitions() -> tuple[DyadicTree, tuple[PartitionState, ...]]:
    """Return the asymmetric three-partition path for a one-by-three grid."""
    tree = DyadicTree.from_shape((1, 3))
    partitions = enumerate_partitions(tree)
    assert len(partitions) == 3
    return tree, partitions


def _gaussian_target(
    pseudo_prior_scale: float,
) -> tuple[GaussianProductSpaceTarget, tuple[PartitionState, ...]]:
    """Build a tiny Gaussian target with exactly enumerable partition masses."""
    tree, partitions = _one_by_three_partitions()
    log_uniform = -math.log(len(partitions))

    def partition_log_prior(partition: PartitionState) -> float:
        """Assign normalized uniform mass to each valid tiny-tree partition."""
        assert partition in partitions
        return log_uniform

    target = GaussianProductSpaceTarget.from_grid(
        observations=[1.8],
        inner_grid_design=[[[1.0, 2.0, 3.0]]],
        tree=tree,
        observation_covariance=[[1.0]],
        inactive_pseudo_prior_scale=pseudo_prior_scale,
        outer_design=[[2.0]],
        outer_prior_covariance=[[0.25]],
        partition_log_prior=partition_log_prior,
    )
    return target, partitions


def _finite_difference_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    step: float = 1e-6,
) -> np.ndarray:
    """Estimate a square Jacobian with centered finite differences."""
    columns = []
    for index in range(point.size):
        offset = np.zeros_like(point)
        offset[index] = step
        columns.append((function(point + offset) - function(point - offset)) / (2.0 * step))
    return np.column_stack(columns)


def test_unequal_mass_additive_transform_round_trips_and_preserves_mass() -> None:
    """An unequal-child split should preserve weighted mass and invert exactly."""
    transform = AdditiveCoefficientTransform(left_mass=1.0, right_mass=2.0)

    left, right = transform.split(parent=2.5, contrast=-1.2)
    recovered_parent, recovered_contrast = transform.merge(left, right)

    assert (left, right) == pytest.approx((1.7, 2.9))
    assert recovered_parent == pytest.approx(2.5)
    assert recovered_contrast == pytest.approx(-1.2)
    assert transform.left_mass * left + transform.right_mass * right == pytest.approx(
        transform.total_mass * 2.5
    )


def test_additive_transform_has_unit_finite_difference_jacobian() -> None:
    """The parent/contrast-to-children map should have unit absolute determinant."""
    transform = AdditiveCoefficientTransform(left_mass=1.25, right_mass=3.75)

    def split(values: np.ndarray) -> np.ndarray:
        """Return both child coefficients for finite-difference evaluation."""
        return np.asarray(transform.split(float(values[0]), float(values[1])))

    jacobian = _finite_difference_jacobian(split, np.array([0.8, -1.1]))

    assert transform.log_abs_jacobian == 0.0
    assert abs(np.linalg.det(jacobian)) == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError, match="Combined child mass"):
        AdditiveCoefficientTransform(left_mass=1e308, right_mass=1e308)


def test_coordinate_swap_round_trips_without_touching_other_coordinates() -> None:
    """A second swap with the reverse auxiliary should restore the source vector."""
    coordinates = np.array([0.3, -0.4, 1.7, 2.2])

    forward = swap_contrast_coordinate(coordinates, 2, -3.5)
    reverse = swap_contrast_coordinate(forward.coordinates, 2, forward.reverse_auxiliary)

    np.testing.assert_array_equal(forward.coordinates, [0.3, -0.4, -3.5, 2.2])
    np.testing.assert_array_equal(reverse.coordinates, coordinates)
    assert forward.reverse_auxiliary == 1.7
    assert reverse.reverse_auxiliary == -3.5
    assert forward.log_abs_jacobian == reverse.log_abs_jacobian == 0.0
    assert not forward.coordinates.flags.writeable
    coordinates[:] = 99.0
    np.testing.assert_array_equal(reverse.coordinates, [0.3, -0.4, 1.7, 2.2])


def test_gaussian_proposal_uses_destination_active_and_inactive_laws() -> None:
    """Splits should use active parameters while merges use inactive parameters."""
    tree, (root, middle, _) = _one_by_three_partitions()
    state = ProductSpaceState(root, np.zeros(3))
    proposal = GaussianContrastProposal(
        active_means=np.array([0.0, 1.5, 0.0]),
        active_variances=np.array([1.0, 4.0, 1.0]),
        inactive_means=np.array([0.0, -2.0, 0.0]),
        inactive_variances=np.array([1.0, 0.25, 1.0]),
    )
    split = SplitMove(tree.root_id)
    merge = MergeMove(tree.root_id)

    split_draw = proposal.draw(state, split, 1, np.random.default_rng(12))
    merge_draw = proposal.draw(
        ProductSpaceState(middle, np.zeros(3)),
        merge,
        1,
        np.random.default_rng(12),
    )

    expected_standard_normal = np.random.default_rng(12).normal()
    assert split_draw == pytest.approx(1.5 + 2.0 * expected_standard_normal)
    assert merge_draw == pytest.approx(-2.0 + 0.5 * expected_standard_normal)
    assert proposal.log_density(1.5, state, split, 1) == pytest.approx(-0.5 * math.log(2.0 * math.pi * 4.0))
    assert proposal.log_density(-2.0, state, merge, 1) == pytest.approx(-0.5 * math.log(2.0 * math.pi * 0.25))


def test_gaussian_proposal_copies_parameters_and_validates_its_interface() -> None:
    """Proposal parameters should be immutable and invalid calls should fail early."""
    active = np.ones(3)
    proposal = GaussianContrastProposal.centered(active, np.full(3, 2.0))
    active[:] = 9.0

    np.testing.assert_array_equal(proposal.active_variances, np.ones(3))
    assert not proposal.active_variances.flags.writeable
    with pytest.raises(ValueError, match="positive"):
        GaussianContrastProposal.centered([1.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="same shape"):
        GaussianContrastProposal.centered([1.0], [1.0, 2.0])
    with pytest.raises(IndexError, match="outside"):
        proposal.log_density(0.0, _dummy_state(), SplitMove(0), 3)
    with pytest.raises(TypeError, match="SplitMove or MergeMove"):
        proposal.log_density(0.0, _dummy_state(), PairedMove(0, 1), 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        proposal.draw(_dummy_state(), SplitMove(0), 1, object())  # type: ignore[arg-type]


def _dummy_state() -> ProductSpaceState:
    """Return a valid two-coordinate state for proposal interface checks."""
    tree = DyadicTree.from_shape((1, 2))
    return ProductSpaceState(PartitionState.root(tree), np.zeros(2))


@pytest.mark.parametrize(
    ("source_index", "seed"),
    [(0, 7), (1, 0)],
    ids=["split", "merge"],
)
def test_transported_split_merge_has_exact_mh_accounting(
    source_index: int,
    seed: int,
) -> None:
    """Both move directions should report exact target, partition, and auxiliary terms."""
    tree, partitions = _one_by_three_partitions()
    layout = TreeContrastLayout.from_tree(tree)
    source = partitions[source_index]
    current = ProductSpaceState(
        source,
        np.array([0.2, -0.8, 1.4]),
        np.array([5.0, -2.0]),
    )
    proposal = GaussianContrastProposal(
        active_means=np.array([0.0, 0.4, -0.2]),
        active_variances=np.array([1.0, 1.7, 0.8]),
        inactive_means=np.array([0.0, -0.6, 0.3]),
        inactive_variances=np.array([1.0, 0.5, 2.1]),
    )
    target_logs = {partitions[0]: 0.0, partitions[1]: 4.0, partitions[2]: -3.0}

    transition = transported_partition_metropolis_step(
        layout,
        current,
        log_density=lambda state: target_logs[state.partition],
        auxiliary_proposal=proposal,
        rng=np.random.default_rng(seed),
    )

    assert transition.coordinate_index == 1
    assert transition.forward_auxiliary is not None
    assert transition.reverse_auxiliary is not None
    assert transition.reverse_auxiliary == -0.8
    assert isinstance(transition.move, (SplitMove, MergeMove))
    reverse_move = transition.move.reverse()
    forward_auxiliary_log_density = proposal.log_density(
        transition.forward_auxiliary,
        current,
        transition.move,
        transition.coordinate_index,
    )
    reverse_auxiliary_log_density = proposal.log_density(
        transition.reverse_auxiliary,
        transition.candidate,
        reverse_move,
        transition.coordinate_index,
    )
    expected = (
        target_logs[transition.candidate.partition]
        - target_logs[source]
        + transition.log_partition_q_reverse
        - transition.log_partition_q_forward
        + reverse_auxiliary_log_density
        - forward_auxiliary_log_density
    )
    assert transition.current_log_density == target_logs[source]
    assert transition.candidate_log_density == target_logs[transition.candidate.partition]
    assert transition.log_acceptance_ratio == pytest.approx(expected)
    assert {transition.log_partition_q_forward, transition.log_partition_q_reverse} == {
        0.0,
        -math.log(2.0),
    }
    assert transition.log_auxiliary_q_forward == pytest.approx(forward_auxiliary_log_density)
    assert transition.log_auxiliary_q_reverse == pytest.approx(reverse_auxiliary_log_density)
    assert transition.log_abs_jacobian == 0.0
    np.testing.assert_array_equal(transition.candidate.inner_coordinates[[0, 2]], [0.2, 1.4])
    np.testing.assert_array_equal(transition.candidate.outer_coefficients, [5.0, -2.0])
    assert not transition.candidate.inner_coordinates.flags.writeable
    assert not transition.candidate.outer_coefficients.flags.writeable


def test_transported_move_obeys_pointwise_detailed_balance() -> None:
    """Paired source and destination fluxes should agree before integration."""
    tree, partitions = _one_by_three_partitions()
    layout = TreeContrastLayout.from_tree(tree)
    current = ProductSpaceState(partitions[0], np.array([0.2, -0.8, 1.4]))
    proposal = GaussianContrastProposal(
        active_means=np.array([0.0, 0.4, -0.2]),
        active_variances=np.array([1.0, 1.7, 0.8]),
        inactive_means=np.array([0.0, -0.6, 0.3]),
        inactive_variances=np.array([1.0, 0.5, 2.1]),
    )
    target_logs = {partitions[0]: -0.7, partitions[1]: 1.2, partitions[2]: -2.0}

    transition = transported_partition_metropolis_step(
        layout,
        current,
        log_density=lambda state: target_logs[state.partition],
        auxiliary_proposal=proposal,
        rng=np.random.default_rng(7),
    )

    forward_acceptance = min(1.0, math.exp(transition.log_acceptance_ratio))
    reverse_acceptance = min(1.0, math.exp(-transition.log_acceptance_ratio))
    forward_flux = (
        math.exp(
            transition.current_log_density
            + transition.log_partition_q_forward
            + transition.log_auxiliary_q_forward
        )
        * forward_acceptance
    )
    reverse_flux = (
        math.exp(
            transition.candidate_log_density
            + transition.log_partition_q_reverse
            + transition.log_auxiliary_q_reverse
        )
        * reverse_acceptance
    )

    assert forward_flux == pytest.approx(reverse_flux, rel=1e-12)


def test_matched_gaussian_auxiliary_terms_cancel_changed_target_factors() -> None:
    """Prior-matched split and merge proposals should cancel affected densities."""
    target, partitions = _gaussian_target(pseudo_prior_scale=2.5)
    proposal = GaussianContrastProposal.centered(
        target.inner_prior_variances,
        target.inactive_pseudo_prior_variances,
    )
    coordinate_index = target.contrast_layout.contrast_index(target.tree.root_id)
    inactive_value = -0.8
    active_value = 0.35
    root_state = ProductSpaceState(partitions[0], np.array([0.2, inactive_value, 1.4]))
    middle_state = ProductSpaceState(partitions[1], np.array([0.2, active_value, 1.4]))
    split = SplitMove(target.tree.root_id)
    merge = split.reverse()

    active_proposal_log_density = proposal.log_density(
        active_value,
        root_state,
        split,
        coordinate_index,
    )
    inactive_proposal_log_density = proposal.log_density(
        inactive_value,
        middle_state,
        merge,
        coordinate_index,
    )
    active_variance = float(target.inner_prior_variances[coordinate_index])
    inactive_variance = float(target.inactive_pseudo_prior_variances[coordinate_index])
    active_target_log_density = -0.5 * (
        math.log(2.0 * math.pi * active_variance) + active_value**2 / active_variance
    )
    inactive_target_log_density = -0.5 * (
        math.log(2.0 * math.pi * inactive_variance) + inactive_value**2 / inactive_variance
    )
    changed_target_factor = active_target_log_density - inactive_target_log_density
    auxiliary_correction = inactive_proposal_log_density - active_proposal_log_density

    assert changed_target_factor + auxiliary_correction == pytest.approx(0.0, abs=1e-15)


def test_transported_step_returns_an_isolated_tree_without_using_auxiliary() -> None:
    """A one-cell tree should return unchanged without drawing an auxiliary."""
    tree = DyadicTree.from_shape((1, 1))
    layout = TreeContrastLayout.from_tree(tree)
    current = ProductSpaceState(
        PartitionState.root(tree),
        np.array([0.2]),
        np.array([1.0]),
    )

    transition = transported_partition_metropolis_step(
        layout,
        current,
        log_density=lambda state: -1.25,
        auxiliary_proposal=GaussianContrastProposal.centered([1.0], [2.0]),
        rng=np.random.default_rng(4),
    )

    assert transition.state is current
    assert transition.candidate is current
    assert transition.move is None
    assert not transition.accepted
    assert transition.coordinate_index is None
    assert transition.forward_auxiliary is None
    assert transition.reverse_auxiliary is None
    assert transition.log_acceptance_ratio == -math.inf


def test_transported_step_rejects_invalid_public_interfaces() -> None:
    """Kernel interface and dimensional errors should have explicit failures."""
    tree = DyadicTree.from_shape((1, 2))
    layout = TreeContrastLayout.from_tree(tree)
    current = ProductSpaceState(PartitionState.root(tree), np.zeros(2))
    proposal = GaussianContrastProposal.centered(np.ones(2), np.ones(2))
    valid_arguments = {
        "layout": layout,
        "current": current,
        "log_density": lambda state: 0.0,
        "auxiliary_proposal": proposal,
        "rng": np.random.default_rng(3),
    }

    for keyword, invalid, message in [
        ("layout", object(), "layout"),
        ("current", object(), "current"),
        ("log_density", 1.0, "callable"),
        ("auxiliary_proposal", object(), "draw and log_density"),
        ("rng", object(), "numpy.random.Generator"),
    ]:
        arguments = dict(valid_arguments)
        arguments[keyword] = invalid
        with pytest.raises(TypeError, match=message):
            transported_partition_metropolis_step(**arguments)  # type: ignore[arg-type]

    wrong_size = ProductSpaceState(PartitionState.root(tree), np.zeros(1))
    with pytest.raises(ValueError, match="do not match"):
        transported_partition_metropolis_step(
            layout,
            wrong_size,
            log_density=lambda state: 0.0,
            auxiliary_proposal=proposal,
            rng=np.random.default_rng(3),
        )


@pytest.mark.parametrize("pseudo_prior_scale", [0.45, 2.5])
def test_conditional_and_transported_chain_matches_exact_partition_probabilities(
    pseudo_prior_scale: float,
) -> None:
    """The transported tiny Gaussian chain should recover exact partition masses."""
    target, partitions = _gaussian_target(pseudo_prior_scale)
    expected = target.partition_probabilities(partitions)
    proposal = GaussianContrastProposal.centered(
        target.inner_prior_variances,
        target.inactive_pseudo_prior_variances,
    )
    rng = np.random.default_rng(20260718)
    state = target.draw_conditional_state(partitions[0], rng)
    counts = {partition: 0 for partition in partitions}
    draws = 10_000
    burn_in = 1_000

    for draw in range(draws):
        state = target.draw_conditional_state(state.partition, rng)
        state = transported_partition_metropolis_step(
            target.contrast_layout,
            state,
            log_density=target.log_density,
            auxiliary_proposal=proposal,
            rng=rng,
        ).state
        if draw >= burn_in:
            counts[state.partition] += 1

    observed = np.array([counts[partition] for partition in partitions], dtype=float)
    observed /= observed.sum()
    np.testing.assert_allclose(observed, list(expected.values()), atol=0.05)
