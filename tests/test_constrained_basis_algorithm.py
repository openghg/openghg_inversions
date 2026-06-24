import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.algorithms import (
    AllSplitAcceptancePolicies,
    AxisParallelSplitStep,
    GreedyAxisParallelSplitStrategy,
    InertialSplitStep,
    LatLonGridGeometry,
    MinChildTargetWeightShare,
    MinChildWeightShare,
    allocate_nbasis_by_class,
    region_constrained_basis,
)


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


def test_axis_parallel_split_uses_lat_lon_geometry_for_axis_choice():
    """Physical geometry can choose latitude over high-latitude index width."""
    weights = np.ones((2, 6))
    grid = xr.DataArray(
        weights,
        dims=("lat", "lon"),
        coords={"lat": [80.0, 81.0], "lon": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]},
    )
    geometry = LatLonGridGeometry.from_dataarray(grid)
    nodes = [(row, col) for row in range(2) for col in range(6)]

    index_children = AxisParallelSplitStep(balanced=False, clean_splits=True)(nodes, weights)
    physical_children = AxisParallelSplitStep(
        balanced=False,
        clean_splits=True,
        geometry=geometry,
    )(nodes, weights)

    assert {frozenset(child) for child in index_children} == {
        frozenset((row, col) for row in range(2) for col in range(3)),
        frozenset((row, col) for row in range(2) for col in range(3, 6)),
    }
    assert {frozenset(child) for child in physical_children} == {
        frozenset((0, col) for col in range(6)),
        frozenset((1, col) for col in range(6)),
    }


def test_axis_parallel_balanced_split_uses_lat_lon_geometry_for_axis_choice():
    """Default balanced axis selection also uses physical geometry when provided."""
    weights = np.ones((2, 6))
    grid = xr.DataArray(
        weights,
        dims=("lat", "lon"),
        coords={"lat": [80.0, 81.0], "lon": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]},
    )
    geometry = LatLonGridGeometry.from_dataarray(grid)
    nodes = [(row, col) for row in range(2) for col in range(6)]

    children = AxisParallelSplitStep(clean_splits=True, geometry=geometry)(nodes, weights)

    assert {frozenset(child) for child in children} == {
        frozenset((0, col) for col in range(6)),
        frozenset((1, col) for col in range(6)),
    }


def test_lat_lon_geometry_requires_lat_lon_dimension_order():
    """Lat/lon geometry must keep node axis zero aligned to latitude."""
    grid = xr.DataArray(
        np.ones((6, 2)),
        dims=("lon", "lat"),
        coords={"lat": [80.0, 81.0], "lon": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]},
    )

    with pytest.raises(ValueError, match="dimensions ordered"):
        LatLonGridGeometry.from_dataarray(grid)


@pytest.mark.parametrize(
    "geometry_result",
    [
        None,
        np.full((12, 2), np.nan),
        np.zeros((12, 1)),
    ],
)
def test_axis_parallel_split_falls_back_when_geometry_is_unavailable(geometry_result):
    """Invalid split geometry falls back to row/column axis choice."""

    class InvalidGeometry:
        """Geometry that cannot provide usable physical coordinates."""

        def coordinates(self, nodes, node_weights=None):
            return geometry_result

    weights = np.ones((2, 6))
    nodes = [(row, col) for row in range(2) for col in range(6)]

    children = AxisParallelSplitStep(
        balanced=False,
        clean_splits=True,
        geometry=InvalidGeometry(),
    )(nodes, weights)
    fallback_children = AxisParallelSplitStep(balanced=False, clean_splits=True)(nodes, weights)

    assert children == fallback_children


def test_axis_parallel_split_falls_back_when_lat_lon_geometry_shape_mismatches():
    """Lat/lon geometry outside the node bounds falls back to row/column indices."""
    weights = np.ones((2, 6))
    nodes = [(row, col) for row in range(2) for col in range(6)]
    geometry = LatLonGridGeometry(
        latitudes=np.array([[80.0]]),
        longitudes=np.array([[0.0]]),
    )

    children = AxisParallelSplitStep(
        balanced=False,
        clean_splits=True,
        geometry=geometry,
    )(nodes, weights)
    fallback_children = AxisParallelSplitStep(balanced=False, clean_splits=True)(nodes, weights)

    assert children == fallback_children


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


def test_inertial_split_can_differ_from_axis_parallel_split():
    """Inertial ordering can split anisotropic shapes away from row/column cuts."""
    nodes = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
    weights = np.zeros((2, 4))
    for node in nodes:
        weights[node] = 1.0

    inertial_children = InertialSplitStep(balanced=False)(nodes, weights)
    axis_parallel_children = AxisParallelSplitStep(balanced=False)(nodes, weights)

    inertial_sets = {frozenset(child) for child in inertial_children}
    axis_parallel_sets = {frozenset(child) for child in axis_parallel_children}
    assert inertial_sets != axis_parallel_sets
    assert inertial_sets == {
        frozenset({(0, 3), (0, 2)}),
        frozenset({(0, 1), (0, 0), (1, 0)}),
    }


def test_inertial_split_uses_lat_lon_geometry_for_projection_order():
    """Physical geometry can change inertial PCA ordering at high latitude."""
    weights = np.ones((3, 10))
    grid = xr.DataArray(
        weights,
        dims=("lat", "lon"),
        coords={"lat": [80.0, 84.0, 88.0], "lon": np.arange(10.0)},
    )
    geometry = LatLonGridGeometry.from_dataarray(grid)
    nodes = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2)]

    index_children = InertialSplitStep(balanced=False)(nodes, weights)
    physical_children = InertialSplitStep(balanced=False, geometry=geometry)(nodes, weights)

    assert {frozenset(child) for child in index_children} == {
        frozenset({(0, 0), (1, 0)}),
        frozenset({(0, 1), (0, 2), (1, 2)}),
    }
    assert {frozenset(child) for child in physical_children} == {
        frozenset({(0, 0), (0, 1)}),
        frozenset({(0, 2), (1, 0), (1, 2)}),
    }


@pytest.mark.parametrize(
    "geometry_result",
    [
        None,
        np.full((5, 2), np.nan),
        np.zeros((5, 1)),
    ],
)
def test_inertial_split_falls_back_when_geometry_is_unavailable(geometry_result):
    """Invalid split geometry falls back to row/column coordinate behavior."""

    class InvalidGeometry:
        """Geometry that cannot provide usable physical coordinates."""

        def coordinates(self, nodes, node_weights=None):
            return geometry_result

    nodes = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
    weights = np.zeros((2, 4))
    for node in nodes:
        weights[node] = 1.0

    children = InertialSplitStep(balanced=False, geometry=InvalidGeometry())(nodes, weights)
    fallback_children = InertialSplitStep(balanced=False)(nodes, weights)

    assert children == fallback_children


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


def test_greedy_strategy_rejects_low_weight_child_split():
    """Splits producing a low-weight child are rejected."""
    weights = np.array([[100.0, 1.0, 1.0, 1.0]])
    class_mask = np.ones(weights.shape, dtype=bool)

    labels = GreedyAxisParallelSplitStrategy(
        split_acceptance=MinChildWeightShare(min_child_weight_share=0.05),
    )(weights, class_mask, target_regions=2)

    assert set(np.unique(labels)) == {1}


def test_greedy_strategy_accepts_split_above_min_child_weight_share():
    """Splits are accepted when all children meet the minimum weight share."""
    weights = np.ones((1, 4))
    class_mask = np.ones(weights.shape, dtype=bool)

    labels = GreedyAxisParallelSplitStrategy(
        split_acceptance=MinChildWeightShare(min_child_weight_share=0.25),
    )(weights, class_mask, target_regions=2)

    assert set(np.unique(labels)) == {1, 2}


def test_greedy_strategy_split_stopping_can_return_fewer_regions_than_requested():
    """Greedy stopping treats requested regions as an upper target."""
    weights = np.array([[50.0, 50.0, 1.0, 1.0]])
    class_mask = np.ones(weights.shape, dtype=bool)

    labels = GreedyAxisParallelSplitStrategy(
        split_acceptance=MinChildWeightShare(min_child_weight_share=0.1),
    )(weights, class_mask, target_regions=3)

    assert set(np.unique(labels)) == {1, 2}


def test_greedy_strategy_split_stopping_freezes_rejected_partition():
    """Rejected partitions are frozen instead of being requeued."""
    weights = np.array([[100.0, 1.0, 1.0, 1.0]])
    class_mask = np.ones(weights.shape, dtype=bool)

    class LowWeightTailSplit:
        """Custom splitter that repeatedly proposes the same poor split."""

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, nodes: list[tuple[int, int]], weights: np.ndarray) -> list[list[tuple[int, int]]]:
            self.calls += 1
            return [nodes[:1], nodes[1:]]

    split_step = LowWeightTailSplit()
    labels = GreedyAxisParallelSplitStrategy(
        split_step=split_step,
        split_acceptance=MinChildWeightShare(min_child_weight_share=0.05),
    )(weights, class_mask, target_regions=3)

    assert set(np.unique(labels)) == {1}
    assert split_step.calls == 1


def test_child_target_weight_share_rejects_small_balanced_children():
    """Target-weight stopping rejects children below the equal-region target."""
    weights = np.array([[100.0, 100.0, 1.0, 1.0]])
    parent = [(0, 2), (0, 3)]
    children = [[(0, 2)], [(0, 3)]]

    assert MinChildWeightShare(min_child_weight_share=0.1)(parent, children, weights)
    assert not MinChildTargetWeightShare(min_child_target_weight_share=0.1)(
        parent,
        children,
        weights,
        target_regions=3,
    )


def test_child_target_weight_share_can_accept_parent_imbalanced_children():
    """Target-weight stopping is not a parent-relative balance guard."""
    weights = np.array([[100.0, 1.0, 1.0, 1.0]])
    parent = [(0, 0), (0, 1), (0, 2), (0, 3)]
    children = [[(0, 0)], [(0, 1), (0, 2), (0, 3)]]

    assert MinChildTargetWeightShare(min_child_target_weight_share=0.05)(
        parent,
        children,
        weights,
        target_regions=2,
    )
    assert not MinChildWeightShare(min_child_weight_share=0.05)(parent, children, weights)


def test_child_target_weight_share_rejects_split_that_creates_small_child():
    """Target-weight stopping rejects a split that would create a small region."""
    weights = np.array([[100.0, 1.0, 1.0, 1.0]])
    class_mask = np.ones(weights.shape, dtype=bool)

    class LowWeightTailSplit:
        """Split one heavy cell from the low-weight tail."""

        def __call__(self, nodes: list[tuple[int, int]], weights: np.ndarray) -> list[list[tuple[int, int]]]:
            return [nodes[:1], nodes[1:]]

    labels = GreedyAxisParallelSplitStrategy(
        split_step=LowWeightTailSplit(),
        split_acceptance=MinChildTargetWeightShare(min_child_target_weight_share=0.1),
    )(weights, class_mask, target_regions=2)

    assert set(np.unique(labels)) == {1}


def test_child_target_weight_share_accepts_normal_split():
    """Target-weight stopping accepts children above the equal-region threshold."""
    weights = np.array([[100.0, 10.0, 10.0]])
    class_mask = np.ones(weights.shape, dtype=bool)

    class HeavyThenTailSplit:
        """Split one heavy cell from an acceptable tail."""

        def __call__(self, nodes: list[tuple[int, int]], weights: np.ndarray) -> list[list[tuple[int, int]]]:
            return [nodes[:1], nodes[1:]]

    labels = GreedyAxisParallelSplitStrategy(
        split_step=HeavyThenTailSplit(),
        split_acceptance=MinChildTargetWeightShare(min_child_target_weight_share=0.1),
    )(weights, class_mask, target_regions=2)

    assert set(np.unique(labels)) == {1, 2}


def test_min_child_target_weight_share_zero_weight_falls_back_to_area_target():
    """Zero total weight uses cell counts for direct policy calls."""
    weights = np.zeros((1, 4))
    parent = [(0, 0), (0, 1), (0, 2), (0, 3)]
    children = [[(0, 0)], [(0, 1)]]

    assert MinChildTargetWeightShare(min_child_target_weight_share=0.5)(
        parent,
        children,
        weights,
        target_regions=2,
    )
    assert not MinChildTargetWeightShare(min_child_target_weight_share=0.75)(
        parent,
        children,
        weights,
        target_regions=2,
    )


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_min_child_target_weight_share_validates_threshold(threshold: float):
    """Target weight share thresholds must be between zero and one."""
    with pytest.raises(ValueError, match="min_child_target_weight_share must be between 0 and 1"):
        MinChildTargetWeightShare(min_child_target_weight_share=threshold)


def test_all_split_acceptance_policies_requires_every_policy_to_accept():
    """Split acceptance policies can be composed with all-of semantics."""
    weights = np.array([[100.0, 1.0, 1.0, 1.0]])
    parent = [(0, 0), (0, 1), (0, 2), (0, 3)]
    children = [[(0, 0)], [(0, 1), (0, 2), (0, 3)]]
    policy = AllSplitAcceptancePolicies(
        MinChildTargetWeightShare(min_child_target_weight_share=0.05),
        MinChildWeightShare(min_child_weight_share=0.05),
    )

    assert not policy(parent, children, weights, target_regions=2)


def test_greedy_strategy_composes_target_and_balance_policies():
    """Greedy orchestration passes target counts into composed policies."""
    weights = np.array([[100.0, 1.0, 1.0, 1.0]])
    class_mask = np.ones(weights.shape, dtype=bool)

    class LowWeightTailSplit:
        """Split one heavy cell from the low-weight tail."""

        def __call__(self, nodes: list[tuple[int, int]], weights: np.ndarray) -> list[list[tuple[int, int]]]:
            return [nodes[:1], nodes[1:]]

    labels = GreedyAxisParallelSplitStrategy(
        split_step=LowWeightTailSplit(),
        split_acceptance=AllSplitAcceptancePolicies(
            MinChildTargetWeightShare(min_child_target_weight_share=0.05),
            MinChildWeightShare(min_child_weight_share=0.05),
        ),
    )(weights, class_mask, target_regions=2)

    assert set(np.unique(labels)) == {1}


def test_region_constrained_basis_child_target_stopping_uses_class_local_total():
    """Target-weight stopping uses each class total as the denominator."""
    weights = xr.DataArray(
        np.array([[100.0, 100.0], [1.0, 1.0]]),
        dims=("lat", "lon"),
    )
    classes = xr.DataArray(
        np.array([["high", "high"], ["low", "low"]]),
        dims=weights.dims,
    )

    labels = region_constrained_basis(
        weights,
        classes,
        nbasis={"high": 1, "low": 2},
        split_strategy=GreedyAxisParallelSplitStrategy(
            split_acceptance=MinChildTargetWeightShare(min_child_target_weight_share=0.5),
        ),
    )

    assert len(set(np.unique(labels.values)) - {0}) == 3
    assert all(len(class_values) == 1 for class_values in _class_values_for_labels(labels, classes).values())


def test_region_constrained_basis_split_stopping_keeps_class_boundaries():
    """Weight-share stopping still partitions each region class independently."""
    weights = xr.DataArray(
        np.array([[50.0, 50.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
        dims=("lat", "lon"),
    )
    classes = xr.DataArray(
        np.array([["high", "high", "high", "high"], ["even", "even", "even", "even"]]),
        dims=weights.dims,
    )

    labels = region_constrained_basis(
        weights,
        classes,
        nbasis={"high": 3, "even": 2},
        split_strategy=GreedyAxisParallelSplitStrategy(
            split_acceptance=MinChildWeightShare(min_child_weight_share=0.1),
        ),
    )

    assert len(set(np.unique(labels.values)) - {0}) == 4
    assert all(len(class_values) == 1 for class_values in _class_values_for_labels(labels, classes).values())


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
