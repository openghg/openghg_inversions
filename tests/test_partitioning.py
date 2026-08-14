"""Tests for the generic greedy partitioning engine."""

import numpy as np
import pytest

from openghg_inversions.basis.algorithms import greedy_partitioning


def test_greedy_partitioning_prioritizes_weight_size_and_insertion_order():
    """Partitions are selected by weight, then size, then stable insertion order."""
    weights = np.array([[3.0, 3.0, 2.0, 2.0, 1.0, 2.5, 2.5, 2.5, 2.5]])
    highest_weight = [(0, 0), (0, 1)]
    largest_tied_weight = [(0, 2), (0, 3), (0, 4)]
    first_equal_partition = [(0, 5), (0, 6)]
    second_equal_partition = [(0, 7), (0, 8)]
    partitions = [
        first_equal_partition,
        largest_tied_weight,
        highest_weight,
        second_equal_partition,
    ]

    class RecordUnsplittable:
        """Record selection order while declining every proposed split."""

        def __init__(self) -> None:
            self.calls: list[list[tuple[int, int]]] = []

        def __call__(
            self,
            nodes: list[tuple[int, int]],
            weights: np.ndarray,
        ) -> list[list[tuple[int, int]]]:
            del weights
            self.calls.append(nodes)
            return [nodes]

    split_step = RecordUnsplittable()

    greedy_partitioning(
        partitions,
        target_regions=5,
        weights=weights,
        split_step=split_step,
    )

    assert split_step.calls == [
        highest_weight,
        largest_tied_weight,
        first_equal_partition,
        second_equal_partition,
    ]


def test_greedy_partitioning_returns_unsplittable_parent():
    """An unsplittable parent remains intact in the result."""
    parent = [(0, 0), (0, 1), (0, 2)]

    result = greedy_partitioning(
        [parent],
        target_regions=2,
        weights=np.ones((1, 3)),
        split_step=lambda nodes, weights: [nodes],
    )

    assert result == [parent]


def test_greedy_partitioning_skips_nary_split_that_would_overshoot():
    """An N-ary proposal is skipped when its children exceed the target count."""
    parent = [(0, 0), (0, 1), (0, 2)]

    result = greedy_partitioning(
        [parent],
        target_regions=2,
        weights=np.ones((1, 3)),
        split_step=lambda nodes, weights: [[node] for node in nodes],
    )

    assert result == [parent]


@pytest.mark.parametrize(
    "children",
    [
        pytest.param(
            [[(0, 0), (0, 1)], [(0, 1), (0, 2)]],
            id="overlapping-children",
        ),
        pytest.param(
            [[(0, 0)], [(0, 1)]],
            id="omitted-parent-node",
        ),
        pytest.param(
            [[(0, 0), (0, 0)], [(0, 1), (0, 2)]],
            id="duplicate-node",
        ),
        pytest.param(
            [[(0, 0)], [(0, 1), (0, 2), (0, 3)]],
            id="outside-parent-node",
        ),
        pytest.param(
            [[(0, 0), (0, 1), (0, 2)], []],
            id="empty-child",
        ),
    ],
)
def test_greedy_partitioning_rejects_malformed_partition_step_output(
    children: list[list[tuple[int, int]]],
):
    """Partition steps must return an exact, disjoint partition of the parent."""
    parent = [(0, 0), (0, 1), (0, 2)]

    with pytest.raises(ValueError):
        greedy_partitioning(
            [parent],
            target_regions=2,
            weights=np.ones((1, 4)),
            split_step=lambda nodes, weights: children,
        )
