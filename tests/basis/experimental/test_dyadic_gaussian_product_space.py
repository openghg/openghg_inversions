"""Tests for the analytic Gaussian dyadic product-space target."""

import math

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.enumeration import enumerate_partitions
from openghg_inversions.basis.experimental.dyadic.gaussian_product_space import (
    GaussianProductSpaceTarget,
)
from openghg_inversions.basis.experimental.dyadic.gaussian_product_space_sampler import (
    sample_gaussian_product_space,
)
from openghg_inversions.basis.experimental.dyadic.product_space import (
    ProductSpaceState,
)
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree


def _target(
    *,
    observation: float = 0.0,
    pseudo_prior_scale: float = 1.0,
    outer_prior_scale: float = 0.5,
) -> tuple[GaussianProductSpaceTarget, tuple[PartitionState, ...]]:
    """Build a one-observation target with three exactly enumerable partitions."""
    tree = DyadicTree.from_shape((1, 3))
    partitions = enumerate_partitions(tree)
    log_uniform = -math.log(len(partitions))

    def prior(partition: PartitionState) -> float:
        """Assign normalized uniform mass to each enumerated partition."""
        assert partition in partitions
        return log_uniform

    target = GaussianProductSpaceTarget.from_grid(
        observations=np.array([observation]),
        inner_grid_design=np.array([[[1.0, 2.0, 3.0]]]),
        tree=tree,
        observation_covariance=np.array([[1.0]]),
        inner_prior_scale=1.0,
        inactive_pseudo_prior_scale=pseudo_prior_scale,
        outer_design=np.array([[2.0]]),
        outer_prior_covariance=np.array([[outer_prior_scale**2]]),
        partition_log_prior=prior,
    )
    return target, partitions


def test_exact_partition_probabilities_match_scalar_gaussian_oracle() -> None:
    """Analytic probabilities should match independently derived predictive variances."""
    target, partitions = _target(observation=0.0)

    probabilities = target.partition_probabilities(partitions)

    # R contributes 1 and the outer block contributes 2**2 * 0.5**2 = 1.
    # Inner variances are 6**2/3, 1**2 + 5**2/2, and 1**2+2**2+3**2.
    predictive_variances = np.array([14.0, 15.5, 16.0])
    expected = 1.0 / np.sqrt(predictive_variances)
    expected /= expected.sum()
    np.testing.assert_allclose(list(probabilities.values()), expected, atol=1e-12)


def test_exact_partition_probabilities_do_not_depend_on_pseudo_prior() -> None:
    """Normalized inactive pseudo-priors should integrate out of model probabilities."""
    narrow, partitions = _target(observation=1.3, pseudo_prior_scale=0.4)
    broad, _ = _target(observation=1.3, pseudo_prior_scale=3.0)

    narrow_probabilities = narrow.partition_probabilities(partitions)
    broad_probabilities = broad.partition_probabilities(partitions)

    np.testing.assert_allclose(
        list(narrow_probabilities.values()),
        list(broad_probabilities.values()),
        atol=1e-14,
    )


def test_augmented_density_includes_normalized_inactive_pseudo_priors() -> None:
    """Changing pseudo-prior scale should retain its partition-dependent normalizer."""
    unit, partitions = _target(pseudo_prior_scale=1.0)
    broad, _ = _target(pseudo_prior_scale=2.0)
    root_state = ProductSpaceState(partitions[0], np.zeros(3), np.zeros(1))
    finest_state = ProductSpaceState(partitions[-1], np.zeros(3), np.zeros(1))

    root_difference = broad.log_density(root_state) - unit.log_density(root_state)
    finest_difference = broad.log_density(finest_state) - unit.log_density(finest_state)

    assert root_difference == pytest.approx(-2.0 * math.log(2.0))
    assert finest_difference == pytest.approx(0.0)


def test_conditional_posterior_matches_direct_information_form() -> None:
    """The active inner and outer block should match a direct Gaussian calculation."""
    target, partitions = _target(observation=1.2)

    conditional = target.conditional_posterior(partitions[0])

    design = np.array([[6.0, 2.0]])
    prior_covariance = np.diag([1.0 / 3.0, 0.25])
    expected_covariance = np.linalg.inv(np.linalg.inv(prior_covariance) + design.T @ design)
    expected_mean = expected_covariance @ design.T @ np.array([1.2])
    assert conditional.active_inner_indices == (0,)
    np.testing.assert_allclose(conditional.mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(conditional.covariance, expected_covariance, atol=1e-12)
    same = target.conditional_posterior(partitions[0])
    assert conditional == same
    with pytest.raises(TypeError, match="unhashable"):
        hash(conditional)


def test_refined_conditional_uses_documented_signed_contrast_design() -> None:
    """A nonzero-data posterior should retain the declared left-minus-right sign."""
    target, partitions = _target(observation=1.2)

    conditional = target.conditional_posterior(partitions[1])

    # Regional columns are 1 and 5. The unequal-child decoder gives contrast
    # column 1*(2/3) + 5*(-1/3) = -1 for delta = left - right.
    design = np.array([[6.0, -1.0, 2.0]])
    prior_covariance = np.diag([1.0 / 3.0, 1.5, 0.25])
    expected_covariance = np.linalg.inv(np.linalg.inv(prior_covariance) + design.T @ design)
    expected_mean = expected_covariance @ design.T @ np.array([1.2])

    np.testing.assert_allclose(conditional.mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(conditional.covariance, expected_covariance, atol=1e-12)


def test_conditional_draw_has_permanent_inner_and_fixed_outer_dimensions() -> None:
    """A Gibbs draw should refresh inactive contrasts without changing dimensions."""
    target, partitions = _target(observation=-0.7, pseudo_prior_scale=1.8)

    state = target.draw_conditional_state(partitions[0], np.random.default_rng(8))

    assert state.partition == partitions[0]
    assert state.inner_coordinates.shape == (3,)
    assert state.outer_coefficients.shape == (1,)
    assert np.all(np.isfinite(state.inner_coordinates))
    assert np.all(np.isfinite(state.outer_coefficients))


def test_outer_prior_is_a_separate_always_active_factor() -> None:
    """Changing only the outer prior should change its normalized density factor."""
    narrow, partitions = _target(outer_prior_scale=0.5)
    broad, _ = _target(outer_prior_scale=1.0)
    state = ProductSpaceState(partitions[1], np.zeros(3), np.array([0.0]))

    difference = broad.log_density(state) - narrow.log_density(state)

    assert difference == pytest.approx(-math.log(2.0))


@pytest.mark.parametrize("pseudo_prior_scale", [0.6, 2.0])
def test_gibbs_and_partition_mh_frequencies_match_exact_oracle(
    pseudo_prior_scale: float,
) -> None:
    """The complete tiny sampler should recover exact partition probabilities."""
    target, partitions = _target(observation=1.8, pseudo_prior_scale=pseudo_prior_scale)
    expected = target.partition_probabilities(partitions)
    rng = np.random.default_rng(20260717)
    trace = sample_gaussian_product_space(
        target,
        partitions[0],
        draws=11_000,
        warmup=1_000,
        rng=rng,
    )
    observed = np.array([trace.partitions.count(partition) for partition in partitions], dtype=float)
    observed /= observed.sum()
    np.testing.assert_allclose(observed, list(expected.values()), atol=0.05)


def test_blocked_sampler_returns_fixed_coordinate_trace_and_diagnostics() -> None:
    """The reusable chain should retain fixed dimensions and local MH diagnostics."""
    target, partitions = _target(observation=0.4)

    trace = sample_gaussian_product_space(
        target,
        partitions[1],
        draws=7,
        warmup=4,
        thinning=2,
        partition_updates_per_draw=3,
        rng=np.random.default_rng(41),
    )

    assert trace.draw_count == 7
    assert trace.inner_coordinates.shape == (7, 3)
    assert trace.outer_coefficients.shape == (7, 1)
    assert trace.partition_accepted.shape == (7, 3)
    assert trace.partition_log_acceptance_ratio.shape == (7, 3)
    assert trace.region_counts.shape == (7,)
    assert 0.0 <= trace.partition_acceptance_rate <= 1.0
    assert trace.warmup_acceptance_rate is not None
    assert 0.0 <= trace.warmup_acceptance_rate <= 1.0
    assert trace.thinning == 2
    np.testing.assert_array_equal(trace.state(-1).inner_coordinates, trace.inner_coordinates[-1])
    assert not trace.inner_coordinates.flags.writeable
    assert not trace.partition_accepted.flags.writeable


@pytest.mark.parametrize(
    ("keyword", "value", "exception", "message"),
    [
        ("draws", 0, ValueError, "draws"),
        ("warmup", -1, ValueError, "warmup"),
        ("thinning", True, TypeError, "thinning"),
        ("partition_updates_per_draw", 0, ValueError, "partition_updates_per_draw"),
    ],
)
def test_blocked_sampler_rejects_invalid_controls(
    keyword: str,
    value: object,
    exception: type[Exception],
    message: str,
) -> None:
    """Invalid draw, warmup, thinning, and update controls should fail early."""
    target, partitions = _target()
    arguments: dict[str, object] = {
        "draws": 2,
        "warmup": 0,
        "thinning": 1,
        "partition_updates_per_draw": 1,
        "rng": np.random.default_rng(1),
    }
    arguments[keyword] = value

    with pytest.raises(exception, match=message):
        sample_gaussian_product_space(target, partitions[0], **arguments)  # type: ignore[arg-type]


def test_zero_mass_partition_prior_is_supported() -> None:
    """Hard partition constraints should give zero density instead of aborting."""
    target, partitions = _target()

    def root_only_prior(partition: PartitionState) -> float:
        """Assign all prior mass to the root partition."""
        return 0.0 if partition == partitions[0] else -math.inf

    constrained = GaussianProductSpaceTarget.from_grid(
        observations=target.observations,
        inner_grid_design=np.array([[[1.0, 2.0, 3.0]]]),
        tree=target.tree,
        observation_covariance=target.observation_covariance,
        outer_design=target.outer_design,
        outer_prior_covariance=target.outer_prior_covariance,
        partition_log_prior=root_only_prior,
    )
    candidate = ProductSpaceState(partitions[1], np.zeros(3), np.zeros(1))

    assert constrained.log_density(candidate) == -math.inf
    assert constrained.partition_probabilities(partitions) == {
        partitions[0]: 1.0,
        partitions[1]: 0.0,
        partitions[2]: 0.0,
    }


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("inactive_pseudo_prior_scale", 0.0, "inactive_pseudo_prior_scale"),
        ("outer_prior_covariance", None, "outer_prior_covariance"),
    ],
)
def test_target_rejects_invalid_prior_configuration(
    keyword: str,
    value: object,
    message: str,
) -> None:
    """Invalid pseudo-prior and fixed-outer prior settings should fail early."""
    tree = DyadicTree.from_shape((1, 2))
    arguments: dict[str, object] = {
        "observations": [0.0],
        "inner_grid_design": [[[1.0, 2.0]]],
        "tree": tree,
        "observation_covariance": [[1.0]],
        "outer_design": [[1.0]],
        "outer_prior_covariance": [[0.25]],
    }
    arguments[keyword] = value

    with pytest.raises(ValueError, match=message):
        GaussianProductSpaceTarget.from_grid(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("observations", np.array([1.0 + 1.0j])),
        ("observation_mean", np.array([1.0 + 1.0j])),
        ("observation_covariance", np.array([[1.0 + 1.0j]])),
        ("outer_design", np.array([[1.0 + 1.0j]])),
        ("outer_prior_covariance", np.array([[1.0 + 1.0j]])),
    ],
)
def test_target_rejects_complex_arrays(keyword: str, value: np.ndarray) -> None:
    """Complex model inputs should fail instead of being silently truncated."""
    tree = DyadicTree.from_shape((1, 2))
    arguments: dict[str, object] = {
        "observations": [0.0],
        "observation_mean": [0.0],
        "inner_grid_design": [[[1.0, 2.0]]],
        "tree": tree,
        "observation_covariance": [[1.0]],
        "outer_design": [[1.0]],
        "outer_prior_covariance": [[0.25]],
    }
    arguments[keyword] = value

    with pytest.raises(ValueError, match="real-valued"):
        GaussianProductSpaceTarget.from_grid(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("keyword", ["inner_prior_scale", "inactive_pseudo_prior_scale"])
@pytest.mark.parametrize("scale", [1e154, 1e308])
def test_target_rejects_scales_that_overflow_variances(keyword: str, scale: float) -> None:
    """Extreme finite scales should produce a documented validation error."""
    tree = DyadicTree.from_shape((1, 2))
    arguments: dict[str, object] = {
        "observations": [0.0],
        "inner_grid_design": [[[1.0, 2.0]]],
        "tree": tree,
        "observation_covariance": [[1.0]],
        keyword: scale,
    }

    with pytest.raises(ValueError, match="variances"):
        GaussianProductSpaceTarget.from_grid(**arguments)  # type: ignore[arg-type]
