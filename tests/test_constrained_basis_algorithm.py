import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.algorithms import (
    allocate_nbasis_by_class,
    region_constrained_basis,
)


def _class_values_for_labels(labels: xr.DataArray, classes: xr.DataArray) -> dict[int, set]:
    result = {}
    for label in np.unique(labels.values):
        if label == 0:
            continue
        class_values = classes.values[labels.values == label]
        result[int(label)] = {value for value in class_values if value == value}
    return result


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
