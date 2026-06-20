import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.algorithms import (
    AxisParallelSplitStep,
    GreedyAxisParallelSplitStrategy,
    allocate_nbasis_by_class,
    region_constrained_basis,
)
from openghg_inversions.basis.algorithms._constrained import InertialSplitStep


def _class_values_for_labels(labels: xr.DataArray, classes: xr.DataArray) -> dict[int, set]:
    """Collect class values covered by each positive basis label."""
    result = {}
    for label in np.unique(labels.values):
        if label == 0:
            continue
        class_values = classes.values[labels.values == label]
        result[int(label)] = {value for value in class_values if value == value}
    return result


def _partition_weight(nodes: list[tuple[int, int]], weights: np.ndarray) -> float:
    """Return total weight for a node partition."""
    if not nodes:
        return 0.0
    rows, cols = zip(*nodes)
    return float(weights[list(rows), list(cols)].sum())


def test_region_constrained_basis_labels_do_not_cross_classes():
    """Generated labels are globally unique and stay inside one class."""
    weights = xr.DataArray(
        np.array(
            [
                [8.0, 7.0, 1.0, 1.0],
                [6.0, 5.0, 1.0, 1.0],
                [1.0, 1.0, 6.0, 7.0],
                [np.nan, np.nan, 8.0, 9.0],
            ]
        ),
        dims=("lat", "lon"),
        coords={"lat": [10.0, 20.0, 30.0, 40.0], "lon": [1.0, 2.0, 3.0, 4.0]},
    ).fillna(0.0)
    classes = xr.DataArray(
        np.array(
            [
                ["land", "land", "sea", "sea"],
                ["land", "land", "sea", "sea"],
                ["land", "land", "sea", "sea"],
                [np.nan, np.nan, "sea", "sea"],
            ],
            dtype=object,
        ),
        dims=weights.dims,
        coords=weights.coords,
    )

    labels = region_constrained_basis(weights, classes, nbasis=4)

    assert labels.name == "basis"
    assert labels.dims == weights.dims
    assert labels.sel(lat=40.0, lon=1.0) == 0
    assert labels.sel(lat=40.0, lon=2.0) == 0
    assert set(np.unique(labels.values)) == {0, 1, 2, 3, 4}
    assert all(len(class_values) == 1 for class_values in _class_values_for_labels(labels, classes).values())


def test_allocate_nbasis_by_class_weighted_with_minimum():
    """Automatic allocation keeps a minimum and favours higher-weight classes."""
    weights = xr.DataArray(
        np.array([[10.0, 10.0, 1.0, 1.0], [10.0, 10.0, 1.0, 1.0]]),
        dims=("lat", "lon"),
    )
    classes = xr.DataArray(
        np.array([["high", "high", "low", "low"], ["high", "high", "low", "low"]]),
        dims=weights.dims,
    )

    allocation = allocate_nbasis_by_class(weights, classes, nbasis=5)

    assert allocation["high"] > allocation["low"]
    assert allocation["low"] >= 1
    assert sum(allocation.values()) == 5


def test_region_constrained_basis_uses_explicit_allocation():
    """Explicit per-class allocation controls class-local region targets."""
    weights = xr.DataArray(np.ones((4, 4)), dims=("lat", "lon"))
    classes = xr.DataArray(
        np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
            ]
        ),
        dims=weights.dims,
    )

    labels = region_constrained_basis(weights, classes, nbasis={1: 2, 2: 1})

    class_1_labels = set(np.unique(labels.where(classes == 1, 0))) - {0}
    class_2_labels = set(np.unique(labels.where(classes == 2, 0))) - {0}
    assert len(class_1_labels) == 2
    assert len(class_2_labels) == 1


def test_region_constrained_basis_splits_zero_weight_classes_by_area():
    """All-zero weights should fall back to area allocation and splitting."""
    weights = xr.DataArray(np.zeros((2, 4)), dims=("lat", "lon"))
    classes = xr.DataArray(
        np.array([["left", "left", "right", "right"], ["left", "left", "right", "right"]]),
        dims=weights.dims,
    )

    labels = region_constrained_basis(weights, classes, nbasis=4)

    assert set(np.unique(labels.values)) == {1, 2, 3, 4}
    assert all(len(class_values) == 1 for class_values in _class_values_for_labels(labels, classes).values())


def test_greedy_axis_parallel_strategy_hits_target_region_count():
    """Greedy axis-parallel splitting reaches the requested count when cells permit."""
    weights = np.array(
        [
            [8.0, 7.0, 1.0, 1.0],
            [6.0, 5.0, 1.0, 1.0],
            [1.0, 1.0, 6.0, 7.0],
            [1.0, 1.0, 8.0, 9.0],
        ]
    )
    class_mask = np.ones(weights.shape, dtype=bool)

    labels = GreedyAxisParallelSplitStrategy()(weights, class_mask, target_regions=5)

    assert set(np.unique(labels)) == {1, 2, 3, 4, 5}


def test_inertial_split_produces_two_non_empty_child_partitions():
    """Non-degenerate inertial splits produce two child node partitions."""
    nodes = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    weights = np.zeros((5, 5))
    for node in nodes:
        weights[node] = 1.0

    children = InertialSplitStep()(nodes, weights)

    assert len(children) == 2
    assert all(children)
    assert sorted(children[0] + children[1]) == nodes


def test_inertial_split_degenerate_geometry_falls_back_deterministically():
    """Axis-aligned geometry with mxy=0 falls back to the axis-parallel step."""
    nodes = [(0, 1), (1, 1), (2, 1), (3, 1)]
    weights = np.ones((4, 3))

    inertial_children = InertialSplitStep(balanced=False)(nodes, weights)
    fallback_children = AxisParallelSplitStep(balanced=False)(nodes, weights)

    assert inertial_children == fallback_children
    assert len(inertial_children) == 2


def test_inertial_split_projection_tie_falls_back_deterministically():
    """A split boundary through equal projections uses the fallback splitter."""
    nodes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    weights = np.array([[1.0, 100.0], [100.0, 1.0]])

    inertial_children = InertialSplitStep(balanced=True)(nodes, weights)
    fallback_children = AxisParallelSplitStep(balanced=True)(nodes, weights)

    assert inertial_children == fallback_children
    assert len(inertial_children) == 2


def test_inertial_split_unsplittable_partition_returns_original_nodes():
    """Unsplittable partitions return one non-empty child and no empty child."""
    nodes = [(0, 0)]
    weights = np.ones((1, 1))

    children = InertialSplitStep()(nodes, weights)

    assert children == [nodes]


@pytest.mark.parametrize("fill_value", [0.0, 1.0])
def test_inertial_split_handles_zero_and_equal_weights(fill_value: float):
    """All-zero and equal weights do not crash or produce empty children."""
    nodes = [(0, 0), (1, 1), (2, 2), (3, 3)]
    weights = np.full((4, 4), fill_value)

    children = InertialSplitStep()(nodes, weights)

    assert len(children) == 2
    assert all(children)
    assert sorted(children[0] + children[1]) == nodes


def test_inertial_split_balanced_approximates_half_weight_split():
    """Balanced inertial splitting chooses the split nearest half total weight."""
    nodes = [(0, 0), (1, 1), (2, 2), (3, 3)]
    weights = np.zeros((4, 4))
    for node, value in zip(nodes, [1.0, 1.0, 8.0, 10.0]):
        weights[node] = value

    children = InertialSplitStep(balanced=True)(nodes, weights)
    child_weights = sorted(_partition_weight(child, weights) for child in children)

    assert child_weights == [10.0, 10.0]


def test_inertial_split_unbalanced_uses_count_based_split():
    """Unbalanced inertial splitting divides ordered nodes by count."""
    nodes = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    weights = np.zeros((5, 5))
    for node, value in zip(nodes, [100.0, 1.0, 1.0, 1.0, 1.0]):
        weights[node] = value

    children = InertialSplitStep(balanced=False)(nodes, weights)

    assert sorted(len(child) for child in children) == [2, 3]
    assert sorted(_partition_weight(child, weights) for child in children) == [3.0, 101.0]


def test_region_constrained_basis_with_inertial_step_keeps_class_boundaries():
    """Inertial split steps still run independently inside region classes."""
    weights = xr.DataArray(np.ones((4, 4)), dims=("lat", "lon"))
    class_values = np.full((4, 4), np.nan, dtype=object)
    for index in range(4):
        class_values[index, index] = "main"
        class_values[index, 3 - index] = "anti"
    classes = xr.DataArray(class_values, dims=weights.dims)

    labels = region_constrained_basis(
        weights,
        classes,
        nbasis=4,
        split_strategy=GreedyAxisParallelSplitStrategy(split_step=InertialSplitStep()),
    )

    assert set(np.unique(labels.values)) == {0, 1, 2, 3, 4}
    assert all(len(class_values) == 1 for class_values in _class_values_for_labels(labels, classes).values())


def test_greedy_strategy_accepts_partition_step_returning_multiple_regions():
    """Greedy splitting accepts partition steps that return more than two children."""
    weights = np.ones((2, 3))
    class_mask = np.ones(weights.shape, dtype=bool)

    class SplitByColumn:
        """Custom partition step that groups nodes by column."""

        def __call__(self, nodes: list[tuple[int, int]], weights: np.ndarray) -> list[list[tuple[int, int]]]:
            partitions = [[], [], []]
            for row, col in nodes:
                partitions[col].append((row, col))
            return partitions

    labels = GreedyAxisParallelSplitStrategy(split_step=SplitByColumn())(
        weights,
        class_mask,
        target_regions=3,
    )

    assert set(np.unique(labels)) == {1, 2, 3}


def test_greedy_strategy_does_not_overshoot_target_with_multi_region_step():
    """Multi-region partition steps are skipped when they would exceed target."""
    weights = np.ones((2, 3))
    class_mask = np.ones(weights.shape, dtype=bool)

    class SplitByColumn:
        """Custom partition step that groups nodes by column."""

        def __call__(self, nodes: list[tuple[int, int]], weights: np.ndarray) -> list[list[tuple[int, int]]]:
            partitions = [[], [], []]
            for row, col in nodes:
                partitions[col].append((row, col))
            return partitions

    labels = GreedyAxisParallelSplitStrategy(split_step=SplitByColumn())(
        weights,
        class_mask,
        target_regions=2,
    )

    assert set(np.unique(labels)) == {1}


def test_region_constrained_basis_rejects_explicit_over_allocation():
    """Explicit class allocations cannot request more labels than mapped cells."""
    weights = xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"))
    classes = xr.DataArray(np.array([["a", "a"], ["a", "a"]]), dims=weights.dims)

    with pytest.raises(ValueError, match="exceed mapped cell counts"):
        region_constrained_basis(weights, classes, nbasis={"a": 5})


def test_region_constrained_basis_accepts_custom_split_strategy():
    """The strategy boundary allows future inertial or quadtree-style splitters."""
    weights = xr.DataArray(np.ones((2, 4)), dims=("lat", "lon"))
    classes = xr.DataArray(
        np.array([["left", "left", "right", "right"], ["left", "left", "right", "right"]]),
        dims=weights.dims,
    )

    class OneRegionPerClass:
        """Custom test splitter that ignores requested region count."""

        def __call__(
            self,
            weights: np.ndarray,
            class_mask: np.ndarray,
            target_regions: int,
        ) -> np.ndarray:
            labels = np.zeros(weights.shape, dtype=np.int64)
            labels[class_mask] = 1
            return labels

    labels = region_constrained_basis(
        weights,
        classes,
        nbasis=4,
        split_strategy=OneRegionPerClass(),
    )

    assert set(np.unique(labels.values)) == {1, 2}
    assert all(len(class_values) == 1 for class_values in _class_values_for_labels(labels, classes).values())
