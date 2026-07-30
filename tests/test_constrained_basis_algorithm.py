import json

import numpy as np
import pytest
import xarray as xr
from scipy import ndimage

from openghg_inversions.basis.algorithms import (
    AllSplitAcceptancePolicies,
    AxisParallelSplitStep,
    ConnectedBinaryPartitionStep,
    ConnectedComponentPartitionStep,
    ConnectedComponentSplitStrategy,
    ContrastProximityComponentConsolidation,
    ContrastScoreSplitAcceptance,
    GreedyAxisParallelSplitStrategy,
    InertialSplitStep,
    LatLonGridGeometry,
    MaxChildPCAEccentricity,
    MinChildTargetWeightShare,
    MinChildWeightShare,
    allocate_nbasis_by_class,
    contrast_tau_from_multiplier_cv,
    intersect_region_class_layers,
    region_constrained_basis,
    split_contrast_score,
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


def _mask_from_nodes(nodes: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    """Return a boolean grid mask for a node partition."""
    mask = np.zeros(shape, dtype=bool)
    if nodes:
        rows, columns = zip(*nodes, strict=True)
        mask[list(rows), list(columns)] = True
    return mask


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


def test_intersect_region_class_layers_creates_composite_classes():
    """Layered masks can be crossed into class labels for constrained splitting."""
    weights = xr.DataArray(
        np.ones((3, 3)),
        dims=("lat", "lon"),
        coords={"lat": [1.0, 2.0, 3.0], "lon": [10.0, 20.0, 30.0]},
    )
    landsea = xr.DataArray(
        np.array(
            [
                ["land", "land", "sea"],
                ["land", "sea", "sea"],
                ["unknown", "sea", "sea"],
            ],
            dtype=object,
        ),
        dims=weights.dims,
        coords=weights.coords,
    )
    inner_outer = xr.DataArray(
        np.array(
            [
                ["inner", "outer", "outer"],
                ["inner", "inner", "outer"],
                ["inner", "outer", "outer"],
            ],
            dtype=object,
        ),
        dims=weights.dims,
        coords=weights.coords,
    )

    classes = intersect_region_class_layers(
        {"surface": landsea, "window": inner_outer},
        unmapped_values={"unknown"},
    )
    labels = region_constrained_basis(weights, classes, nbasis=4, allocation="area")

    assert classes.name == "region_classes"
    assert classes.attrs["region_class_layers"] == ("surface", "window")
    assert classes.sel(lat=1.0, lon=10.0).item() == ("land", "inner")
    assert classes.sel(lat=1.0, lon=30.0).item() == ("sea", "outer")
    assert np.isnan(classes.sel(lat=3.0, lon=10.0).item())
    assert set(np.unique(labels.values)) == {0, 1, 2, 3, 4}
    assert all(len(class_values) == 1 for class_values in _class_values_for_labels(labels, classes).values())


def test_intersect_region_class_layers_aligns_transposed_layers():
    """Layer intersections preserve the first layer's dimension order."""
    landsea = xr.DataArray(
        np.array([["land", "sea"], ["land", "sea"]], dtype=object),
        dims=("lat", "lon"),
        coords={"lat": [1.0, 2.0], "lon": [10.0, 20.0]},
    )
    inner_outer = xr.DataArray(
        np.array([["inner", "outer"], ["outer", "outer"]], dtype=object).T,
        dims=("lon", "lat"),
        coords={"lon": [10.0, 20.0], "lat": [1.0, 2.0]},
    )

    classes = intersect_region_class_layers({"surface": landsea, "window": inner_outer})

    assert classes.dims == landsea.dims
    assert classes.sel(lat=1.0, lon=10.0).item() == ("land", "inner")
    assert classes.sel(lat=2.0, lon=10.0).item() == ("land", "outer")


def test_tuple_class_masks_work_when_last_dimension_matches_tuple_length():
    """Tuple class labels should not be compared with NumPy broadcasting."""
    weights = xr.DataArray(
        np.ones((2, 2)),
        dims=("lat", "lon"),
        coords={"lat": [1.0, 2.0], "lon": [10.0, 20.0]},
    )
    landsea = xr.DataArray(
        np.array([["land", "sea"], ["land", "sea"]], dtype=object),
        dims=weights.dims,
        coords=weights.coords,
    )
    inner_outer = xr.DataArray(
        np.array([["inner", "outer"], ["outer", "inner"]], dtype=object),
        dims=weights.dims,
        coords=weights.coords,
    )
    classes = intersect_region_class_layers({"surface": landsea, "window": inner_outer})

    allocation = allocate_nbasis_by_class(
        weights,
        classes,
        nbasis={
            ("land", "inner"): 1,
            ("sea", "outer"): 1,
            ("land", "outer"): 1,
            ("sea", "inner"): 1,
        },
    )
    labels = region_constrained_basis(weights, classes, nbasis=allocation)

    assert allocation == {
        ("land", "inner"): 1,
        ("sea", "outer"): 1,
        ("land", "outer"): 1,
        ("sea", "inner"): 1,
    }
    assert set(np.unique(labels.values)) == {1, 2, 3, 4}
    assert all(len(class_values) == 1 for class_values in _class_values_for_labels(labels, classes).values())


def test_intersect_region_class_layers_unmaps_nulls_in_later_layers():
    """A null in any layer should leave the composite class unmapped."""
    landsea = xr.DataArray(
        np.array([["land", "sea"], ["land", "sea"]], dtype=object),
        dims=("lat", "lon"),
    )
    inner_outer = xr.DataArray(
        np.array([["inner", np.nan], ["outer", "inner"]], dtype=object),
        dims=landsea.dims,
    )

    classes = intersect_region_class_layers({"surface": landsea, "window": inner_outer})

    assert np.isnan(classes.isel(lat=0, lon=1).item())
    assert classes.isel(lat=1, lon=1).item() == ("sea", "inner")


def test_intersect_region_class_layers_rejects_unhashable_values():
    """Composite class labels require hashable layer values."""
    bad_values = np.empty((1, 1), dtype=object)
    bad_values[0, 0] = ["not", "hashable"]
    bad_layer = xr.DataArray(bad_values, dims=("lat", "lon"))

    with pytest.raises(ValueError, match="not hashable"):
        intersect_region_class_layers({"bad": bad_layer})


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


class CheckerboardSplitStep:
    """Split nodes into parity groups that are disconnected by edges."""

    def __call__(
        self,
        nodes: list[tuple[int, int]],
        _weights: np.ndarray,
    ) -> list[list[tuple[int, int]]]:
        even = [node for node in nodes if sum(node) % 2 == 0]
        odd = [node for node in nodes if sum(node) % 2 == 1]
        return [even, odd]


def _coastline_like_binary_case() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Return a connected mask whose balanced cut has a small detached fragment."""
    weights = np.zeros((5, 8), dtype=float)
    class_mask = np.zeros(weights.shape, dtype=bool)
    class_mask[:, :2] = True
    class_mask[2:, 2:] = True
    class_mask[0, 2:4] = True

    weights[:, :2] = 1.0
    weights[2:, 2:] = 9.8 / 18.0
    weights[0, 2:4] = 0.1
    nodes = [(int(row), int(column)) for row, column in zip(*np.where(class_mask), strict=True)]
    return weights, class_mask, nodes


def test_connected_component_partition_step_splits_disconnected_children():
    """Partition-step children are decomposed into deterministic components."""
    nodes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    step = ConnectedComponentPartitionStep(
        CheckerboardSplitStep(),
        connectivity=1,
    )

    children = step(nodes, np.ones((2, 2), dtype=float))

    assert children == [[(0, 0)], [(1, 1)], [(0, 1)], [(1, 0)]]


def test_connected_binary_partition_step_repairs_coastline_fragment():
    """A detached cut fragment moves across the cut to preserve binary arity."""
    weights, _class_mask, nodes = _coastline_like_binary_case()
    split_step = ConnectedBinaryPartitionStep(
        AxisParallelSplitStep(balanced=True, clean_splits=True),
        connectivity=1,
    )

    children = split_step(nodes, weights)

    assert children == split_step(nodes, weights)
    assert [len(child) for child in children] == [12, 18]
    assert len(children) == 2
    assert set(children[0]).isdisjoint(children[1])
    assert set(children[0]) | set(children[1]) == set(nodes)
    assert all(ndimage.label(_mask_from_nodes(child, weights.shape))[1] == 1 for child in children)
    child_weights = [_partition_weight(child, weights) for child in children]
    assert abs(child_weights[0] - child_weights[1]) / sum(child_weights) < 0.03


def test_connected_binary_partition_step_uses_stable_component_tie_break():
    """Equal repair objectives retain the first row-major component on each side."""
    nodes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    split_step = ConnectedBinaryPartitionStep(
        CheckerboardSplitStep(),
        connectivity=1,
    )

    children = split_step(nodes, np.ones((2, 2), dtype=float))

    assert children == [
        [(0, 0), (1, 0)],
        [(0, 1), (1, 1)],
    ]


def test_connected_binary_partition_step_breaks_moved_weight_tie_by_balance():
    """Child-weight balance breaks a tie between minimum-moved-weight repairs."""
    nodes = [(row, column) for row in range(2) for column in range(3)]
    weights = np.asarray(
        [
            [5.0, 1.5, 4.0],
            [2.0, 0.5, 1.0],
        ]
    )
    split_step = ConnectedBinaryPartitionStep(
        CheckerboardSplitStep(),
        connectivity=1,
    )

    children = split_step(nodes, weights)

    assert children == [
        [(0, 1), (0, 2), (1, 2)],
        [(0, 0), (1, 0), (1, 1)],
    ]
    assert [_partition_weight(child, weights) for child in children] == [6.5, 7.5]


def test_connected_binary_partition_step_falls_back_when_repair_is_impossible():
    """An unrepairable binary split uses the historical component decomposition."""
    nodes = [(0, column) for column in range(8)]
    weights = np.ones((1, 8), dtype=float)
    historical_step = ConnectedComponentPartitionStep(CheckerboardSplitStep(), connectivity=1)
    binary_step = ConnectedBinaryPartitionStep(CheckerboardSplitStep(), connectivity=1)

    children = binary_step(nodes, weights)

    assert children == historical_step(nodes, weights)
    assert children == [[(0, column)] for column in (0, 2, 4, 6, 1, 3, 5, 7)]


def test_connected_binary_repair_avoids_eccentricity_rejection_and_target_overshoot():
    """Repaired material children pass the guard and fit a binary target."""
    weights, class_mask, nodes = _coastline_like_binary_case()
    axis_split = AxisParallelSplitStep(balanced=True, clean_splits=True)
    historical_step = ConnectedComponentPartitionStep(axis_split, connectivity=1)
    binary_step = ConnectedBinaryPartitionStep(axis_split, connectivity=1)
    eccentricity_guard = MaxChildPCAEccentricity(max_child_pca_eccentricity=10.0)

    historical_children = historical_step(nodes, weights)
    repaired_children = binary_step(nodes, weights)
    historical_labels = GreedyAxisParallelSplitStrategy(
        split_step=historical_step,
        split_acceptance=eccentricity_guard,
    )(weights, class_mask, target_regions=3)
    repaired_labels = GreedyAxisParallelSplitStrategy(
        split_step=binary_step,
        split_acceptance=eccentricity_guard,
    )(weights, class_mask, target_regions=2)

    assert len(historical_children) == 3
    assert not eccentricity_guard(nodes, historical_children, weights)
    assert eccentricity_guard(nodes, repaired_children, weights)
    assert set(np.unique(historical_labels)) == {0, 1}
    assert set(np.unique(repaired_labels)) == {0, 1, 2}
    assert all(ndimage.label(repaired_labels == label)[1] == 1 for label in (1, 2))


def test_connected_strategy_raises_target_to_disconnected_class_minimum():
    """Disconnected class pieces receive distinct labels below the minimum target."""
    weights = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ]
    )
    class_mask = weights > 0
    strategy = ConnectedComponentSplitStrategy(
        GreedyAxisParallelSplitStrategy(),
        connectivity=1,
    )

    labels = strategy(weights, class_mask, target_regions=1)

    np.testing.assert_array_equal(labels > 0, class_mask)
    assert set(np.unique(labels)) == {0, 1, 2}
    assert strategy.diagnostics == [
        {
            "requested_target": 1,
            "connected_component_minimum": 2,
            "effective_target": 2,
            "actual_regions": 2,
        }
    ]


def test_connected_strategy_four_and_eight_neighbour_diagonal_adjacency():
    """Connectivity controls whether diagonally adjacent cells share a label."""
    weights = np.eye(2, dtype=float)
    class_mask = weights > 0
    four_neighbour = ConnectedComponentSplitStrategy(
        GreedyAxisParallelSplitStrategy(),
        connectivity=1,
    )
    eight_neighbour = ConnectedComponentSplitStrategy(
        GreedyAxisParallelSplitStrategy(),
        connectivity=2,
    )

    four_labels = four_neighbour(weights, class_mask, target_regions=1)
    eight_labels = eight_neighbour(weights, class_mask, target_regions=1)

    assert int(four_labels.max()) == 2
    assert int(eight_labels.max()) == 1


def test_connected_strategy_zero_weight_components_use_area_fallback():
    """All-zero component weights still receive the requested allocation."""
    weights = np.zeros((2, 4), dtype=float)
    class_mask = np.asarray(
        [
            [True, True, False, True],
            [True, True, False, True],
        ]
    )
    strategy = ConnectedComponentSplitStrategy(
        GreedyAxisParallelSplitStrategy(),
        connectivity=1,
    )

    labels = strategy(weights, class_mask, target_regions=3)

    np.testing.assert_array_equal(labels > 0, class_mask)
    assert np.array_equal(np.unique(labels), np.asarray([0, 1, 2, 3]))
    for region in (1, 2, 3):
        assert ndimage.label(labels == region)[1] == 1


def _component_consolidation_policy(
    labels: xr.DataArray,
    *,
    contribution: np.ndarray,
    cell_weight: np.ndarray | None = None,
    source_classes: xr.DataArray | None = None,
    max_merge_distance_km: float = 100.0,
    max_merge_lambda: float = 0.0,
    inactive_component_policy: str = "keep",
    min_regions: int | None = None,
) -> ContrastProximityComponentConsolidation:
    """Return a consolidation policy aligned to a small test grid."""
    return ContrastProximityComponentConsolidation(
        contribution=contribution,
        cell_weight=np.ones(labels.shape) if cell_weight is None else cell_weight,
        geometry=LatLonGridGeometry.from_dataarray(labels),
        max_merge_distance_km=max_merge_distance_km,
        max_merge_lambda=max_merge_lambda,
        source_classes=source_classes,
        inactive_component_policy=inactive_component_policy,
        min_regions=min_regions,
    )


def test_component_consolidation_merges_whole_secondary_not_refined_singleton():
    """A refined one-cell region in the primary component is not a forced island."""
    labels = xr.DataArray(
        np.array([[1, 2, 2, 0, 3]]),
        dims=("lat", "lon"),
        coords={"lat": [50.0], "lon": [0.0, 0.1, 0.2, 0.3, 0.4]},
    )
    classes = xr.full_like(labels, "land", dtype=object)
    contribution = np.array([[[10.0, 1.0, 1.0, 0.0, 1.0]]])
    policy = _component_consolidation_policy(
        labels,
        contribution=contribution,
        max_merge_distance_km=30.0,
    )

    consolidated = policy(labels, classes)

    np.testing.assert_array_equal(
        consolidated.values,
        np.array([[1, 2, 2, 0, 2]]),
    )
    diagnostics = policy.diagnostics[-1]
    assert diagnostics["component_forced_original_labels"] == [3]
    assert diagnostics["resulting_regions"] == 2
    assert diagnostics["strict_connected_input"] is True
    assert diagnostics["strict_connected_output"] is False
    assert diagnostics["deliberately_disconnected_regions"] == [
        {
            "label": 2,
            "original_labels": [2, 3],
            "component_count": 2,
        }
    ]
    assert json.loads(consolidated.attrs["component_consolidation_diagnostics"]) == diagnostics


def test_component_consolidation_rejects_high_contrast_nearby_component():
    """Proximity alone cannot merge a component with a strong reverse-split score."""
    labels = xr.DataArray(
        np.array([[1, 1, 0, 2]]),
        dims=("lat", "lon"),
        coords={"lat": [50.0], "lon": [0.0, 0.1, 0.2, 0.3]},
    )
    classes = xr.full_like(labels, "land", dtype=object)
    policy = _component_consolidation_policy(
        labels,
        contribution=np.array([[[0.0, 0.0, 0.0, 10.0]]]),
    )

    consolidated = policy(labels, classes)

    np.testing.assert_array_equal(consolidated.values, labels.values)
    assert policy.diagnostics[-1]["merges"] == []
    assert policy.diagnostics[-1]["unmerged_component_forced_original_labels"] == [2]
    [evaluation] = policy.diagnostics[-1]["initial_candidate_evaluations"]
    assert evaluation["first_original_label"] == 1
    assert evaluation["second_original_label"] == 2
    assert evaluation["source_class"] == "'__all_sources__'"
    assert evaluation["region_class"] == "'land'"
    assert evaluation["distance_km"] == pytest.approx(14.29496)
    assert evaluation["first_mass"] == 2.0
    assert evaluation["second_mass"] == 1.0
    assert evaluation["lambda"] == pytest.approx(400.0 / 9.0)
    assert evaluation["delta_dfs"] == pytest.approx((400.0 / 9.0) / (1.0 + 400.0 / 9.0))
    assert evaluation["delta_eig"] == pytest.approx(0.5 * np.log1p(400.0 / 9.0))
    assert evaluation["accepted"] is False
    assert evaluation["reason"] == "contrast_threshold"


@pytest.mark.parametrize("boundary", ["class", "source"])
def test_component_consolidation_never_crosses_class_or_source_boundary(boundary: str):
    """A closer region in another class or source is never a candidate target."""
    labels = xr.DataArray(
        np.array([[1, 0, 0, 2, 3]]),
        dims=("lat", "lon"),
        coords={"lat": [50.0], "lon": [0.0, 0.1, 0.2, 0.3, 0.4]},
    )
    classes = xr.DataArray(
        np.array([["a", "a", "a", "b", "a"]], dtype=object),
        dims=labels.dims,
        coords=labels.coords,
    )
    sources = None
    if boundary == "source":
        classes = xr.full_like(labels, "a", dtype=object)
        sources = xr.DataArray(
            np.array([["one", "one", "one", "two", "one"]], dtype=object),
            dims=labels.dims,
            coords=labels.coords,
        )
    policy = _component_consolidation_policy(
        labels,
        contribution=np.zeros((1, *labels.shape)),
        source_classes=sources,
        max_merge_distance_km=20.0,
    )

    consolidated = policy(labels, classes)

    np.testing.assert_array_equal(consolidated.values, labels.values)
    assert policy.diagnostics[-1]["candidate_edges"] == 0


def test_component_consolidation_zero_mass_policy_is_explicit():
    """Inactive components are either retained or proximity-merged by policy."""
    labels = xr.DataArray(
        np.array([[1, 1, 0, 2]]),
        dims=("lat", "lon"),
        coords={"lat": [50.0], "lon": [0.0, 0.1, 0.2, 0.3]},
    )
    classes = xr.full_like(labels, "land", dtype=object)
    cell_weight = np.array([[1.0, 1.0, 0.0, 0.0]])
    contribution = np.zeros((1, *labels.shape))
    keep = _component_consolidation_policy(
        labels,
        contribution=contribution,
        cell_weight=cell_weight,
        inactive_component_policy="keep",
    )
    merge = _component_consolidation_policy(
        labels,
        contribution=contribution,
        cell_weight=cell_weight,
        inactive_component_policy="merge_nearest",
    )

    kept = keep(labels, classes)
    merged = merge(labels, classes)

    assert int(kept.max()) == 2
    assert int(merged.max()) == 1
    assert keep.diagnostics[-1]["unmerged_component_forced_original_labels"] == [2]
    assert keep.diagnostics[-1]["initial_candidate_evaluations"][0]["reason"] == "inactive_kept"
    assert keep.diagnostics[-1]["initial_candidate_evaluations"][0]["accepted"] is False
    assert merge.diagnostics[-1]["merges"][0]["reason"] == "inactive_nearest"


def test_component_consolidation_is_deterministic():
    """Tied candidate scores and distances produce repeatable labels and provenance."""
    labels = xr.DataArray(
        np.array([[1, 0, 2, 0, 3]]),
        dims=("lat", "lon"),
        coords={"lat": [50.0], "lon": [0.0, 0.1, 0.2, 0.3, 0.4]},
    )
    classes = xr.full_like(labels, "land", dtype=object)

    first_policy = _component_consolidation_policy(
        labels,
        contribution=np.zeros((1, *labels.shape)),
    )
    second_policy = _component_consolidation_policy(
        labels,
        contribution=np.zeros((1, *labels.shape)),
    )

    first = first_policy(labels, classes)
    second = second_policy(labels, classes)

    xr.testing.assert_identical(first, second)
    assert first_policy.diagnostics == second_policy.diagnostics


def test_component_consolidation_respects_region_floor():
    """A configured region floor limits otherwise acceptable merges."""
    labels = xr.DataArray(
        np.array([[1, 0, 2, 0, 3]]),
        dims=("lat", "lon"),
        coords={"lat": [50.0], "lon": [0.0, 0.1, 0.2, 0.3, 0.4]},
    )
    classes = xr.full_like(labels, "land", dtype=object)
    policy = _component_consolidation_policy(
        labels,
        contribution=np.zeros((1, *labels.shape)),
        min_regions=2,
    )

    consolidated = policy(labels, classes)

    assert len(set(np.unique(consolidated.values)) - {0}) == 2
    assert len(policy.diagnostics[-1]["merges"]) == 1


def test_component_consolidation_requires_connected_input_labels():
    """The policy cannot silently reinterpret an already disconnected label."""
    labels = xr.DataArray(
        np.array([[1, 0, 1]]),
        dims=("lat", "lon"),
        coords={"lat": [50.0], "lon": [0.0, 0.1, 0.2]},
    )
    classes = xr.full_like(labels, "land", dtype=object)
    policy = _component_consolidation_policy(
        labels,
        contribution=np.zeros((1, *labels.shape)),
    )

    with pytest.raises(ValueError, match="requires strictly connected input labels"):
        policy(labels, classes)


def test_region_constrained_basis_applies_component_consolidation_policy():
    """The low-level basis API can opt into post-construction consolidation."""
    weights = xr.DataArray(
        np.ones((1, 5)),
        dims=("lat", "lon"),
        coords={"lat": [50.0], "lon": [0.0, 0.1, 0.2, 0.3, 0.4]},
    )
    classes = xr.DataArray(
        np.array([["land", "land", "land", np.nan, "land"]], dtype=object),
        dims=weights.dims,
        coords=weights.coords,
    )
    policy = _component_consolidation_policy(
        weights,
        contribution=np.ones((1, *weights.shape)),
        max_merge_distance_km=30.0,
    )

    consolidated = region_constrained_basis(
        weights,
        classes,
        nbasis=1,
        split_strategy=ConnectedComponentSplitStrategy(
            GreedyAxisParallelSplitStrategy(),
        ),
        component_consolidation=policy,
    )

    assert int(consolidated.max()) == 1
    assert policy.diagnostics[-1]["original_regions"] == 2
    assert policy.diagnostics[-1]["resulting_regions"] == 1


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


def test_max_child_pca_eccentricity_rejects_rank_one_child():
    """Rank-one multi-cell children have infinite PCA eccentricity."""
    weights = np.ones((3, 3))
    parent = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
    children = [[(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (2, 0)]]

    assert not MaxChildPCAEccentricity(max_child_pca_eccentricity=10.0)(parent, children, weights)


def test_max_child_pca_eccentricity_accepts_compact_and_single_cell_children():
    """Compact child shapes and single-cell children are accepted."""
    weights = np.ones((3, 3))
    parent = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
    children = [[(0, 0), (0, 1), (1, 0), (1, 1)], [(2, 2)]]

    assert MaxChildPCAEccentricity(max_child_pca_eccentricity=2.0)(parent, children, weights)


def test_greedy_strategy_pca_eccentricity_stopping_freezes_rejected_partition():
    """Shape stopping can reject an otherwise valid split."""
    weights = np.ones((1, 4))
    class_mask = np.ones(weights.shape, dtype=bool)

    labels = GreedyAxisParallelSplitStrategy(
        split_acceptance=MaxChildPCAEccentricity(max_child_pca_eccentricity=10.0),
    )(weights, class_mask, target_regions=2)

    assert set(np.unique(labels)) == {1}


def test_max_child_pca_eccentricity_can_use_custom_geometry():
    """Custom geometry can change the eccentricity decision."""

    class LineGeometry:
        """Map every node onto one physical line."""

        def coordinates(self, nodes, node_weights=None):
            return np.array([[row + col, row + col] for row, col in nodes], dtype=np.float64)

    weights = np.ones((2, 2))
    parent = [(row, col) for row in range(2) for col in range(2)]
    children = [parent]

    assert MaxChildPCAEccentricity(max_child_pca_eccentricity=2.0)(parent, children, weights)
    assert not MaxChildPCAEccentricity(max_child_pca_eccentricity=2.0, geometry=LineGeometry())(
        parent,
        children,
        weights,
    )


def test_max_child_pca_eccentricity_tolerance_controls_rank_one_detection():
    """Tolerance controls when small minor variance is treated as rank one."""
    weights = np.ones((2, 4))
    parent = [(0, 0), (0, 3), (1, 0), (1, 3)]
    children = [parent]

    assert MaxChildPCAEccentricity(max_child_pca_eccentricity=3.0, tolerance=0.0)(parent, children, weights)
    assert not MaxChildPCAEccentricity(max_child_pca_eccentricity=3.0, tolerance=0.25)(
        parent,
        children,
        weights,
    )


def test_max_child_pca_eccentricity_composes_with_target_weight_policy():
    """PCA shape stopping composes with target-aware policies."""
    weights = np.ones((1, 4))
    class_mask = np.ones(weights.shape, dtype=bool)

    labels = GreedyAxisParallelSplitStrategy(
        split_acceptance=AllSplitAcceptancePolicies(
            MinChildTargetWeightShare(min_child_target_weight_share=0.1),
            MaxChildPCAEccentricity(max_child_pca_eccentricity=10.0),
        ),
    )(weights, class_mask, target_regions=2)

    assert set(np.unique(labels)) == {1}


@pytest.mark.parametrize("threshold", [0.0, 0.999, -1.0, np.inf, np.nan])
def test_max_child_pca_eccentricity_validates_threshold(threshold: float):
    """PCA eccentricity thresholds must be positive and finite."""
    with pytest.raises(ValueError, match="max_child_pca_eccentricity must be at least 1 and finite"):
        MaxChildPCAEccentricity(max_child_pca_eccentricity=threshold)


@pytest.mark.parametrize("tolerance", [-1.0, np.inf, np.nan])
def test_max_child_pca_eccentricity_validates_tolerance(tolerance: float):
    """PCA eccentricity tolerance must be non-negative and finite."""
    with pytest.raises(ValueError, match="tolerance must be non-negative and finite"):
        MaxChildPCAEccentricity(max_child_pca_eccentricity=10.0, tolerance=tolerance)


def test_split_contrast_score_is_zero_for_equal_mass_weighted_mean_contribution():
    """Equal child mean contribution vectors add no observation-space direction."""
    contribution = np.array([[[1.0, 1.0]], [[2.0, 2.0]]])
    cell_weight = np.array([[2.0, 3.0]])

    score = split_contrast_score(
        contribution=contribution,
        cell_weight=cell_weight,
        child_a=[(0, 0)],
        child_b=[(0, 1)],
    )

    assert np.allclose(score.contrast, [0.0, 0.0])
    assert score.lambda_value == pytest.approx(0.0)
    assert score.delta_dfs == pytest.approx(0.0)
    assert score.delta_eig == pytest.approx(0.0)
    assert score.uncalibrated


def test_split_contrast_score_child_swap_changes_sign_not_information_scores():
    """Child order changes the contrast sign but not lambda, DFS, or EIG."""
    contribution = np.array([[[1.0, 0.0]], [[0.0, 2.0]]])
    cell_weight = np.array([[2.0, 3.0]])

    score_ab = split_contrast_score(
        contribution=contribution,
        cell_weight=cell_weight,
        child_a=[(0, 0)],
        child_b=[(0, 1)],
    )
    score_ba = split_contrast_score(
        contribution=contribution,
        cell_weight=cell_weight,
        child_a=[(0, 1)],
        child_b=[(0, 0)],
    )

    assert np.allclose(score_ab.contrast, -score_ba.contrast)
    assert score_ab.lambda_value == pytest.approx(score_ba.lambda_value)
    assert score_ab.delta_dfs == pytest.approx(score_ba.delta_dfs)
    assert score_ab.delta_eig == pytest.approx(score_ba.delta_eig)


def test_split_contrast_additive_multiplier_parameterisation_preserves_mass():
    """The split contrast coefficient changes child multipliers without changing regional mass."""
    mu_a = 2.0
    mu_b = 3.0
    mu_g = mu_a + mu_b
    alpha_0 = 1.2
    delta = 0.4

    alpha_a = alpha_0 + delta * mu_b / mu_g
    alpha_b = alpha_0 - delta * mu_a / mu_g

    assert mu_a * alpha_a + mu_b * alpha_b == pytest.approx(mu_g * alpha_0)


def test_split_contrast_lambda_scales_as_tau_squared():
    """Tau is the prior SD of the split contrast coefficient."""
    contribution = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    cell_weight = np.ones((1, 2))

    score_tau_1 = split_contrast_score(
        contribution=contribution,
        cell_weight=cell_weight,
        child_a=[(0, 0)],
        child_b=[(0, 1)],
        contrast_tau=1.0,
    )
    score_tau_3 = split_contrast_score(
        contribution=contribution,
        cell_weight=cell_weight,
        child_a=[(0, 0)],
        child_b=[(0, 1)],
        contrast_tau=3.0,
    )

    assert score_tau_3.lambda_value == pytest.approx(9.0 * score_tau_1.lambda_value)


def test_split_contrast_diagonal_design_covariance_uses_weighted_norm():
    """Diagonal S entries are variances in the design-observation row space."""
    contribution = np.array([[[1.0, 0.0]], [[0.0, 2.0]]])
    cell_weight = np.array([[2.0, 3.0]])
    s_diag = np.array([2.0, 8.0])

    score = split_contrast_score(
        contribution=contribution,
        cell_weight=cell_weight,
        child_a=[(0, 0)],
        child_b=[(0, 1)],
        contrast_tau=2.0,
        contrast_s_diag=s_diag,
    )

    expected_contrast = np.array([1.2, -2.4])
    expected_lambda = 4.0 * np.sum(expected_contrast**2 / s_diag)
    assert np.allclose(score.contrast, expected_contrast)
    assert score.lambda_value == pytest.approx(expected_lambda)


def test_split_contrast_xarray_diagonal_covariance_aligns_design_dims():
    """Xarray diagonal S is aligned to contribution design dimensions."""
    contribution = xr.DataArray(
        np.array([[[[1.0, 0.0]], [[0.0, 2.0]]]]),
        dims=("site", "time", "lat", "lon"),
        coords={"site": ["TAC"], "time": [0, 1], "lat": [0], "lon": [0, 1]},
    )
    cell_weight = xr.DataArray(
        np.array([[2.0, 3.0]]),
        dims=("lat", "lon"),
        coords={"lat": [0], "lon": [0, 1]},
    )
    s_diag = xr.DataArray(
        np.array([[2.0], [8.0]]),
        dims=("time", "site"),
        coords={"site": ["TAC"], "time": [0, 1]},
    )

    score = split_contrast_score(
        contribution=contribution,
        cell_weight=cell_weight,
        child_a=[(0, 0)],
        child_b=[(0, 1)],
        contrast_tau=2.0,
        contrast_s_diag=s_diag,
    )

    expected_contrast = np.array([1.2, -2.4])
    expected_lambda = 4.0 * np.sum(expected_contrast**2 / np.array([2.0, 8.0]))
    assert np.allclose(score.contrast, expected_contrast)
    assert score.lambda_value == pytest.approx(expected_lambda)


def test_contrast_tau_from_multiplier_cv_helpers():
    """Multiplier-CV helpers are optional approximations for contrast tau."""
    cv = 0.5

    assert contrast_tau_from_multiplier_cv(cv, approximation="additive") == pytest.approx(np.sqrt(2.0) * cv)
    assert contrast_tau_from_multiplier_cv(cv, approximation="log") == pytest.approx(
        np.sqrt(2.0) * np.sqrt(np.log1p(cv**2))
    )


def test_contrast_score_acceptance_rejects_low_contrast_split():
    """A high threshold freezes a low-contrast proposed split."""
    weights = np.ones((1, 2))
    class_mask = np.ones(weights.shape, dtype=bool)
    contribution = np.array([[[1.0, 1.0]], [[2.0, 2.0]]])

    labels = GreedyAxisParallelSplitStrategy(
        split_acceptance=ContrastScoreSplitAcceptance(
            contribution=contribution,
            min_contrast_lambda=0.1,
        ),
    )(weights, class_mask, target_regions=2)

    assert set(np.unique(labels)) == {1}


def test_contrast_score_acceptance_accepts_high_contrast_split():
    """A low threshold accepts a high-contrast proposed split."""
    weights = np.ones((1, 2))
    class_mask = np.ones(weights.shape, dtype=bool)
    contribution = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])

    labels = GreedyAxisParallelSplitStrategy(
        split_acceptance=ContrastScoreSplitAcceptance(
            contribution=contribution,
            min_contrast_lambda=0.1,
        ),
    )(weights, class_mask, target_regions=2)

    assert set(np.unique(labels)) == {1, 2}


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
