"""Mask- and region-constrained basis generation helpers.

The helpers in this module are intentionally pure: callers pass an already
computed 2D weight field and an already loaded 2D class mask. File loading,
footprint/flux reduction, and domain-specific country lookup stay at the caller
boundary.

The first implementation uses the existing axis-aligned weighted split shape:
each class is split independently with rectangular bucket splits, then labels
are offset globally. The class orchestration accepts a split strategy protocol so
an inertial partition, quadtree step, or other tile generator can be substituted
without changing mask alignment or label-offset behavior.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from ._weighted import bucket_value_split

AllocationMode = Literal["weight", "area"]
NbasisAllocation = int | Mapping[Hashable, int]


class SplitStrategy(Protocol):
    """Strategy protocol for class-local basis splitting."""

    def __call__(
        self,
        weights: np.ndarray,
        class_mask: np.ndarray,
        target_regions: int,
    ) -> np.ndarray:
        """Return positive local labels for cells in ``class_mask``."""
        ...


@dataclass(frozen=True)
class AxisAlignedWeightedSplitStrategy:
    """Class-local axis-aligned weighted split strategy.

    This is the mask-aware core of the existing weighted basis shape: recursively
    split rectangles along the longer axis until each rectangle is below a
    threshold. The threshold is searched to approximate ``target_regions``.
    """

    max_iter: int = 32

    def __call__(
        self,
        weights: np.ndarray,
        class_mask: np.ndarray,
        target_regions: int,
    ) -> np.ndarray:
        """Return class-local labels using weighted rectangular bucket splits."""
        if target_regions < 1:
            raise ValueError("target_regions must be at least 1.")
        if not class_mask.any():
            return np.zeros(weights.shape, dtype=np.int64)

        class_weights = np.where(class_mask, weights, 0.0)
        total_weight = float(class_weights.sum())
        if total_weight == 0.0:
            return _labels_for_bucket(class_weights, class_mask, bucket=0.0)

        low = 0.0
        high = total_weight
        best = _labels_for_bucket(class_weights, class_mask, bucket=high)
        best_error = abs(_count_positive_labels(best) - target_regions)

        for _ in range(self.max_iter):
            bucket = (low + high) / 2.0
            labels = _labels_for_bucket(class_weights, class_mask, bucket=bucket)
            nregions = _count_positive_labels(labels)
            error = abs(nregions - target_regions)
            if error < best_error:
                best = labels
                best_error = error
                if error == 0:
                    break

            if nregions > target_regions:
                low = bucket
            else:
                high = bucket

        return best


def region_constrained_basis(
    weights: xr.DataArray,
    region_classes: xr.DataArray,
    nbasis: NbasisAllocation,
    *,
    allocation: AllocationMode = "weight",
    min_regions_per_class: int = 1,
    split_strategy: SplitStrategy | None = None,
    unmapped_values: Iterable[Hashable] = (),
) -> xr.DataArray:
    """Generate basis labels independently inside each mask/region class.

    Args:
        weights: Two-dimensional non-negative weight field.
        region_classes: Two-dimensional class field on the same grid as
            ``weights``. Each non-null value is treated as a mapped class unless
            listed in ``unmapped_values``.
        nbasis: Either a total number of basis regions to allocate across
            classes, or an explicit mapping from class value to class-local
            region target.
        allocation: Automatic allocation mode used when ``nbasis`` is an
            integer. ``"weight"`` allocates proportional to class total weight,
            falling back to area if all class weights are zero. ``"area"``
            allocates proportional to mapped cell count.
        min_regions_per_class: Minimum automatic allocation for each non-empty
            mapped class. If the requested total is smaller than this minimum
            requires, a ``ValueError`` is raised.
        split_strategy: Class-local splitting strategy. Defaults to
            :class:`AxisAlignedWeightedSplitStrategy`.
        unmapped_values: Additional class values to leave as output label ``0``.

    Returns:
        ``xarray.DataArray`` with the same dimensions and coordinates as
        ``weights``. Mapped cells receive globally unique positive integer
        labels; unmapped cells receive ``0``.

    Notes:
        Labels are guaranteed not to cross class boundaries because each class is
        split independently and relabelled with a global offset. The default
        rectangular strategy can assign one label to disconnected pieces of the
        same class if the class mask itself is disconnected; contiguity is not
        guaranteed by this helper.
    """
    weights, region_classes = _align_2d_inputs(weights, region_classes)
    weight_values = _validate_weights(weights)
    class_values = region_classes.to_numpy()
    mapped_classes = _mapped_classes(class_values, unmapped_values)
    strategy = split_strategy or AxisAlignedWeightedSplitStrategy()

    labels = np.zeros(weight_values.shape, dtype=np.int64)
    if not mapped_classes:
        return _labels_dataarray(labels, weights)

    targets = allocate_nbasis_by_class(
        weights,
        region_classes,
        nbasis,
        allocation=allocation,
        min_regions_per_class=min_regions_per_class,
        unmapped_values=unmapped_values,
    )

    next_label = 1
    for class_value in mapped_classes:
        target_regions = targets[class_value]
        class_mask = class_values == class_value
        local_labels = strategy(weight_values, class_mask, target_regions)
        for local_label in _positive_labels(local_labels):
            labels[(local_labels == local_label) & class_mask] = next_label
            next_label += 1

    return _labels_dataarray(labels, weights)


def allocate_nbasis_by_class(
    weights: xr.DataArray,
    region_classes: xr.DataArray,
    nbasis: NbasisAllocation,
    *,
    allocation: AllocationMode = "weight",
    min_regions_per_class: int = 1,
    unmapped_values: Iterable[Hashable] = (),
) -> dict[Hashable, int]:
    """Allocate class-local region targets for constrained basis generation."""
    if min_regions_per_class < 0:
        raise ValueError("min_regions_per_class must be non-negative.")
    weights, region_classes = _align_2d_inputs(weights, region_classes)
    weight_values = _validate_weights(weights)
    class_values = region_classes.to_numpy()
    mapped_classes = _mapped_classes(class_values, unmapped_values)

    if isinstance(nbasis, Mapping):
        return _explicit_allocation(mapped_classes, nbasis)

    if nbasis < 0:
        raise ValueError("nbasis must be non-negative.")
    if not mapped_classes:
        if nbasis != 0:
            raise ValueError("Cannot allocate basis regions without mapped classes.")
        return {}

    capacities = {
        class_value: int(np.count_nonzero(class_values == class_value)) for class_value in mapped_classes
    }
    minima = {
        class_value: min(min_regions_per_class, capacity) for class_value, capacity in capacities.items()
    }

    min_total = sum(minima.values())
    max_total = sum(capacities.values())
    if nbasis < min_total:
        raise ValueError(
            f"nbasis={nbasis} is smaller than the minimum {min_total} required "
            f"for {len(mapped_classes)} mapped classes."
        )
    if nbasis > max_total:
        raise ValueError(f"nbasis={nbasis} exceeds the {max_total} mapped cells available for splitting.")

    scores = _allocation_scores(weight_values, class_values, mapped_classes, allocation)
    return _distribute_regions(mapped_classes, scores, minima, capacities, nbasis)


def _align_2d_inputs(
    weights: xr.DataArray,
    region_classes: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    if weights.ndim != 2:
        raise ValueError("weights must be two-dimensional.")
    if region_classes.ndim != 2:
        raise ValueError("region_classes must be two-dimensional.")
    if set(weights.dims) != set(region_classes.dims):
        raise ValueError("weights and region_classes must use the same dimensions.")

    region_classes = region_classes.transpose(*weights.dims)
    weights, region_classes = xr.align(weights, region_classes, join="exact")
    return weights, region_classes


def _validate_weights(weights: xr.DataArray) -> np.ndarray:
    values = np.asarray(weights.to_numpy(), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite.")
    if (values < 0.0).any():
        raise ValueError("weights must be non-negative.")
    return values


def _mapped_classes(
    class_values: npt.NDArray[np.object_] | np.ndarray,
    unmapped_values: Iterable[Hashable],
) -> list[Hashable]:
    unmapped = set(unmapped_values)
    classes: list[Hashable] = []
    for value in pd.unique(class_values.ravel()):
        if pd.isna(value) or value in unmapped:
            continue
        classes.append(cast(Hashable, value))
    return classes


def _explicit_allocation(
    mapped_classes: list[Hashable],
    nbasis: Mapping[Hashable, int],
) -> dict[Hashable, int]:
    missing = [class_value for class_value in mapped_classes if class_value not in nbasis]
    if missing:
        raise ValueError(f"Explicit nbasis allocation is missing classes: {missing!r}.")

    allocations = {class_value: int(nbasis[class_value]) for class_value in mapped_classes}
    invalid = {class_value: target for class_value, target in allocations.items() if target < 1}
    if invalid:
        raise ValueError(f"Class allocations must be positive for mapped classes: {invalid!r}.")
    return allocations


def _allocation_scores(
    weights: np.ndarray,
    class_values: np.ndarray,
    mapped_classes: list[Hashable],
    allocation: AllocationMode,
) -> dict[Hashable, float]:
    if allocation == "area":
        return {
            class_value: float(np.count_nonzero(class_values == class_value))
            for class_value in mapped_classes
        }
    if allocation != "weight":
        raise ValueError("allocation must be 'weight' or 'area'.")

    scores = {
        class_value: float(weights[class_values == class_value].sum()) for class_value in mapped_classes
    }
    if sum(scores.values()) == 0.0:
        return _allocation_scores(weights, class_values, mapped_classes, "area")
    return scores


def _distribute_regions(
    mapped_classes: list[Hashable],
    scores: Mapping[Hashable, float],
    minima: Mapping[Hashable, int],
    capacities: Mapping[Hashable, int],
    nbasis: int,
) -> dict[Hashable, int]:
    allocations = dict(minima)
    remaining = nbasis - sum(allocations.values())

    while remaining > 0:
        candidates = [
            class_value
            for class_value in mapped_classes
            if allocations[class_value] < capacities[class_value]
        ]
        if not candidates:
            break

        score_total = sum(scores[class_value] for class_value in candidates)
        if score_total == 0.0:
            score_total = float(len(candidates))
            candidate_scores = {class_value: 1.0 for class_value in candidates}
        else:
            candidate_scores = {class_value: scores[class_value] for class_value in candidates}

        quotas = {
            class_value: remaining * candidate_scores[class_value] / score_total for class_value in candidates
        }
        extras = {
            class_value: min(
                int(np.floor(quota)),
                capacities[class_value] - allocations[class_value],
            )
            for class_value, quota in quotas.items()
        }

        added = sum(extras.values())
        if added == 0:
            class_value = max(
                candidates,
                key=lambda value: (
                    quotas[value],
                    capacities[value] - allocations[value],
                    str(value),
                ),
            )
            extras[class_value] = 1
            added = 1

        for class_value, extra in extras.items():
            allocations[class_value] += extra
        remaining -= added

    return allocations


def _labels_for_bucket(weights: np.ndarray, class_mask: np.ndarray, bucket: float) -> np.ndarray:
    labels = np.zeros(weights.shape, dtype=np.int64)
    label = 1
    for ymin, ymax, xmin, xmax in bucket_value_split(weights, bucket):
        region_mask = class_mask[ymin:ymax, xmin:xmax]
        if not region_mask.any():
            continue
        label_slice = labels[ymin:ymax, xmin:xmax]
        label_slice[region_mask] = label
        label += 1
    return labels


def _count_positive_labels(labels: np.ndarray) -> int:
    return len(_positive_labels(labels))


def _positive_labels(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    return unique[unique > 0]


def _labels_dataarray(labels: np.ndarray, weights: xr.DataArray) -> xr.DataArray:
    return xr.DataArray(labels, dims=weights.dims, coords=weights.coords, name="basis")


__all__ = [
    "AllocationMode",
    "AxisAlignedWeightedSplitStrategy",
    "NbasisAllocation",
    "SplitStrategy",
    "allocate_nbasis_by_class",
    "region_constrained_basis",
]
