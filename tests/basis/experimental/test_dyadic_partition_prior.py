"""Tests for explicit region-count dyadic partition priors."""

import math

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.enumeration import enumerate_partitions
from openghg_inversions.basis.experimental.dyadic.partition_prior import RegionCountPartitionPrior
from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree


def test_uniform_k_prior_normalizes_over_every_four_by_four_partition() -> None:
    """Uniform marginal K mass should divide exactly among all 677 frontiers."""
    tree = DyadicTree.from_shape((4, 4))
    partitions = enumerate_partitions(tree)

    prior = RegionCountPartitionPrior.uniform_k(
        tree,
        minimum_regions=3,
        maximum_regions=8,
    )
    probabilities = np.exp([prior(partition) for partition in partitions])
    counts = np.array([len(partition.active) for partition in partitions])

    assert probabilities.sum() == pytest.approx(1.0, abs=1e-14)
    for region_count in range(3, 9):
        assert probabilities[counts == region_count].sum() == pytest.approx(1.0 / 6.0)
    assert np.all(probabilities[(counts < 3) | (counts > 8)] == 0.0)
    np.testing.assert_allclose(prior.marginal_probability_by_k[3:9], np.full(6, 1.0 / 6.0))
    assert not prior.log_probability_by_k.flags.writeable
    assert not prior.marginal_probability_by_k.flags.writeable


def test_partition_lookup_matches_k_table_and_rejects_invalid_frontier() -> None:
    """The callable should use its fixed table only for valid frontiers."""
    tree = DyadicTree.from_shape((2, 2))
    prior = RegionCountPartitionPrior.uniform_k(tree)
    partition = enumerate_partitions(tree, region_count=3)[0]

    assert prior(partition) == prior.log_probability_by_k[3]

    with pytest.raises(ValueError, match="not in the tree"):
        prior(PartitionState(active=frozenset({999})))


@pytest.mark.parametrize(
    ("minimum", "maximum", "exception"),
    [
        (0, 2, ValueError),
        (3, 2, ValueError),
        (1, 5, ValueError),
        (True, 2, TypeError),
        (1, 2.5, TypeError),
    ],
)
def test_uniform_k_prior_rejects_invalid_bounds(
    minimum: object,
    maximum: object,
    exception: type[Exception],
) -> None:
    """Region-count intervals should reject invalid values without coercion."""
    tree = DyadicTree.from_shape((2, 2))

    with pytest.raises(exception):
        RegionCountPartitionPrior.uniform_k(  # type: ignore[arg-type]
            tree,
            minimum_regions=minimum,
            maximum_regions=maximum,
        )


def test_direct_prior_requires_normalized_partition_mass() -> None:
    """Direct lookup construction should reject unnormalized or empty mass."""
    tree = DyadicTree.from_shape((1, 2))

    with pytest.raises(ValueError, match="sum to one"):
        RegionCountPartitionPrior(tree, np.array([-math.inf, 0.0, 0.0]))
    with pytest.raises(ValueError, match="positive prior mass"):
        RegionCountPartitionPrior(tree, np.full(3, -math.inf))
    with pytest.raises(ValueError, match=r"\[0\]"):
        RegionCountPartitionPrior(tree, np.array([0.0, -math.log(2.0), -math.log(2.0)]))
