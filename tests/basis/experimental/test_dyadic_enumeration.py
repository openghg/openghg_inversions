"""Tests for exhaustive enumeration of tiny dyadic partition frontiers."""

from __future__ import annotations

import numpy as np
import pytest

from openghg_inversions.basis.experimental.dyadic.enumeration import enumerate_partitions
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree


@pytest.mark.parametrize(
    ("shape", "expected_active"),
    [
        ((1, 1), ((0,),)),
        ((1, 2), ((0,), (1, 2))),
        ((1, 3), ((0,), (1, 2), (1, 3, 4))),
        (
            (2, 2),
            (
                (0,),
                (1, 4),
                (1, 5, 6),
                (2, 3, 4),
                (2, 3, 5, 6),
            ),
        ),
    ],
)
def test_enumeration_has_expected_counts_and_recursive_order(
    shape: tuple[int, int],
    expected_active: tuple[tuple[int, ...], ...],
) -> None:
    """Tiny trees should expose every frontier in stable recursive order."""
    tree = DyadicTree.from_shape(shape)

    states = enumerate_partitions(tree)

    assert tuple(state.ordered_active() for state in states) == expected_active
    for state in states:
        state.validate(tree)


@pytest.mark.parametrize(
    ("region_count", "expected_active"),
    [
        (1, ((0,),)),
        (2, ((1, 4),)),
        (3, ((1, 5, 6), (2, 3, 4))),
        (4, ((2, 3, 5, 6),)),
    ],
)
def test_exact_region_count_filter_preserves_order(
    region_count: int,
    expected_active: tuple[tuple[int, ...], ...],
) -> None:
    """An exact region filter should retain matching states without reordering."""
    tree = DyadicTree.from_shape((2, 2))

    states = enumerate_partitions(tree, region_count=region_count)

    assert tuple(state.ordered_active() for state in states) == expected_active


def test_integer_like_region_count_is_accepted() -> None:
    """NumPy integer scalars should satisfy the integer-like count contract."""
    tree = DyadicTree.from_shape((1, 2))

    states = enumerate_partitions(tree, region_count=np.int64(2))  # type: ignore[arg-type]

    assert tuple(state.ordered_active() for state in states) == ((1, 2),)


def test_four_by_four_reference_tree_has_677_partitions() -> None:
    """The planned Gaussian reference tree should retain its exact oracle size."""
    tree = DyadicTree.from_shape((4, 4))

    states = enumerate_partitions(tree)

    assert len(states) == 677
    assert sum(len(enumerate_partitions(tree, region_count=count)) for count in range(1, 17)) == 677


@pytest.mark.parametrize("region_count", [0, -1, 5])
def test_region_count_rejects_values_outside_grid_cell_range(region_count: int) -> None:
    """Counts below one or above the grid-cell leaf count should be rejected."""
    tree = DyadicTree.from_shape((2, 2))

    with pytest.raises(ValueError, match="region_count"):
        enumerate_partitions(tree, region_count=region_count)


@pytest.mark.parametrize("region_count", [True, 1.5, "2"])
def test_region_count_rejects_boolean_and_nonintegers(region_count: object) -> None:
    """Boolean and non-integer counts should fail rather than be coerced."""
    tree = DyadicTree.from_shape((2, 2))

    with pytest.raises(TypeError, match="region_count"):
        enumerate_partitions(tree, region_count=region_count)  # type: ignore[arg-type]


def test_enumeration_rejects_non_tree_input() -> None:
    """Enumeration should reject inputs that do not satisfy the tree contract."""
    with pytest.raises(TypeError, match="DyadicTree"):
        enumerate_partitions(object())  # type: ignore[arg-type]
