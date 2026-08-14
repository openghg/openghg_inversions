"""Contrast-aware consolidation of component-forced basis regions."""

from __future__ import annotations

import json
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy import ndimage
from scipy.spatial import cKDTree

from ._contrast import (
    SplitContrastScore,
    _design_by_grid_values,
    _s_diag_values,
)

if TYPE_CHECKING:
    from ._partition import GridNode, LatLonGridGeometry

InactiveComponentPolicy = Literal["keep", "merge_nearest"]


@dataclass
class _ActiveRegion:
    """Aggregated state for one current consolidation region."""

    original_labels: tuple[int, ...]
    nodes: tuple[GridNode, ...]
    group_key: tuple[Hashable, Hashable]
    forced_only: bool
    mass: float
    contribution: npt.NDArray[np.float64]


@dataclass
class ContrastProximityComponentConsolidation:
    """Merge weak, nearby regions created solely by disconnected components.

    The input labels must already satisfy the requested strict connectivity.
    For each source/class group, the largest connected component is treated as
    the primary component. A secondary component is eligible for consolidation
    only when it remains one whole basis region. This deliberately excludes
    one-cell regions created by useful refinement inside the primary component.

    Candidate pairs never cross the supplied ``region_classes`` or
    ``source_classes``. Positive-mass candidates are accepted only when every
    configured reverse-split contrast score is at or below its merge threshold.
    A zero-mass endpoint is either kept or merged to its nearest eligible
    neighbour according to ``inactive_component_policy``.

    Attributes:
        contribution: Fixed design contribution array with one or more
            design-observation dimensions followed by the two spatial
            dimensions.
        cell_weight: Non-negative prior mass field used by contrast scoring.
        geometry: Latitude/longitude geometry aligned to the label grid.
        max_merge_distance_km: Maximum nearest-cell separation for a candidate.
        max_merge_delta_eig: Optional maximum reverse-split ``delta_eig``.
        max_merge_lambda: Optional maximum reverse-split ``lambda``.
        contrast_tau: Prior standard deviation of the split contrast
            coefficient. ``None`` uses the uncalibrated value ``1``.
        contrast_sigma_design: Optional scalar design standard deviation.
        contrast_s_diag: Optional diagonal design covariance.
        source_classes: Optional source field. When omitted, all cells are
            treated as belonging to one source.
        inactive_component_policy: ``"keep"`` or ``"merge_nearest"``.
        connectivity: ``1`` for four-neighbour or ``2`` for eight-neighbour
            input components.
        min_regions: Optional global region-count floor.
        spatial_dims: Optional contribution spatial dimensions.
        diagnostics: Per-call JSON-compatible consolidation diagnostics.
    """

    contribution: xr.DataArray | npt.ArrayLike
    cell_weight: xr.DataArray | npt.ArrayLike
    geometry: LatLonGridGeometry
    max_merge_distance_km: float
    max_merge_delta_eig: float | None = None
    max_merge_lambda: float | None = None
    contrast_tau: float | None = None
    contrast_sigma_design: float | None = None
    contrast_s_diag: xr.DataArray | npt.ArrayLike | None = None
    source_classes: xr.DataArray | npt.ArrayLike | None = None
    inactive_component_policy: InactiveComponentPolicy = "keep"
    connectivity: int = 1
    min_regions: int | None = None
    spatial_dims: tuple[Hashable, Hashable] | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Validate consolidation configuration."""
        if self.max_merge_distance_km < 0.0 or not np.isfinite(self.max_merge_distance_km):
            raise ValueError("max_merge_distance_km must be non-negative and finite.")
        if self.max_merge_delta_eig is None and self.max_merge_lambda is None:
            raise ValueError("Specify max_merge_delta_eig and/or max_merge_lambda.")
        _validate_optional_non_negative(
            "max_merge_delta_eig",
            self.max_merge_delta_eig,
        )
        _validate_optional_non_negative("max_merge_lambda", self.max_merge_lambda)
        _validate_optional_non_negative("contrast_tau", self.contrast_tau)
        if self.contrast_sigma_design is not None and self.contrast_s_diag is not None:
            raise ValueError("Specify only one of contrast_sigma_design or contrast_s_diag.")
        if self.contrast_sigma_design is not None and (
            self.contrast_sigma_design <= 0.0 or not np.isfinite(self.contrast_sigma_design)
        ):
            raise ValueError("contrast_sigma_design must be positive and finite.")
        if self.inactive_component_policy not in ("keep", "merge_nearest"):
            raise ValueError("inactive_component_policy must be 'keep' or 'merge_nearest'.")
        if self.connectivity not in (1, 2):
            raise ValueError("connectivity must be 1 (edge) or 2 (edge and corner).")
        if self.min_regions is not None and self.min_regions < 1:
            raise ValueError("min_regions must be at least 1 when supplied.")

    def __call__(
        self,
        labels: xr.DataArray,
        region_classes: xr.DataArray,
    ) -> xr.DataArray:
        """Return labels after deterministic contrast/proximity consolidation."""
        labels, region_classes = _align_label_inputs(labels, region_classes)
        label_values = _validate_labels(labels)
        source_values = _aligned_source_values(self.source_classes, labels)
        class_values = region_classes.to_numpy()
        group_by_label = _label_groups(
            label_values,
            class_values,
            source_values,
        )
        structure = ndimage.generate_binary_structure(2, self.connectivity)
        _validate_connected_labels(label_values, structure)

        cell_weight = _aligned_cell_weight(self.cell_weight, labels)
        (
            contribution_values,
            weight_values,
            design_shape,
            design_dims,
        ) = _design_by_grid_values(
            self.contribution,
            cell_weight,
            spatial_dims=self.spatial_dims,
        )
        if weight_values.shape != label_values.shape:
            raise ValueError("cell_weight and labels must have the same spatial shape.")
        if (weight_values < 0.0).any():
            raise ValueError("cell_weight must be non-negative.")
        if self.geometry.latitudes.shape != label_values.shape:
            raise ValueError("geometry and labels must have the same spatial shape.")

        original_regions = _initial_regions(
            label_values,
            group_by_label,
            weight_values,
            contribution_values,
        )
        forced_labels = _component_forced_labels(
            label_values,
            group_by_label,
            weight_values,
            structure,
        )
        for label, region in original_regions.items():
            region.forced_only = label in forced_labels

        candidate_distances = _candidate_region_distances(
            label_values,
            group_by_label,
            forced_labels,
            self.geometry,
            self.max_merge_distance_km,
        )
        inverse_variance = _inverse_design_variance(
            design_shape=design_shape,
            design_dims=design_dims,
            sigma_design=self.contrast_sigma_design,
            s_diag=self.contrast_s_diag,
        )
        initial_candidate_evaluations = self._initial_candidate_evaluations(
            original_regions,
            candidate_distances,
            inverse_variance,
        )

        active = dict(original_regions)
        original_to_active = {label: label for label in original_regions}
        merges: list[dict[str, Any]] = []
        while True:
            if self.min_regions is not None and len(active) <= self.min_regions:
                break
            candidates = self._accepted_candidates(
                active,
                original_to_active,
                candidate_distances,
                inverse_variance,
            )
            if not candidates:
                break

            _priority, first_id, second_id, distance_km, score, reason = min(
                candidates,
                key=lambda candidate: candidate[0],
            )
            first = active.pop(first_id)
            second = active.pop(second_id)
            source, target = _ordered_merge_endpoints(first, second)
            merged_region = _merge_active_regions(first, second)
            result_id = min(merged_region.original_labels)
            active[result_id] = merged_region
            for original_label in merged_region.original_labels:
                original_to_active[original_label] = result_id

            merges.append(
                {
                    "source_original_labels": list(source.original_labels),
                    "target_original_labels": list(target.original_labels),
                    "result_original_labels": list(merged_region.original_labels),
                    "source_class": repr(source.group_key[0]),
                    "region_class": repr(source.group_key[1]),
                    "distance_km": distance_km,
                    "lambda": None if score is None else score.lambda_value,
                    "delta_dfs": None if score is None else score.delta_dfs,
                    "delta_eig": None if score is None else score.delta_eig,
                    "source_mass": source.mass,
                    "target_mass": target.mass,
                    "reason": reason,
                }
            )

        consolidated_values, final_groups = _labels_from_active_regions(
            active,
            label_values.shape,
        )
        deliberately_disconnected = _deliberately_disconnected_regions(
            consolidated_values,
            final_groups,
            structure,
        )
        merged_original_labels = {
            original_label
            for region in active.values()
            if len(region.original_labels) > 1
            for original_label in region.original_labels
        }
        call_diagnostics: dict[str, Any] = {
            "policy": "contrast_proximity",
            "connectivity": self.connectivity,
            "max_merge_distance_km": self.max_merge_distance_km,
            "max_merge_delta_eig": self.max_merge_delta_eig,
            "max_merge_lambda": self.max_merge_lambda,
            "inactive_component_policy": self.inactive_component_policy,
            "strict_connected_input": True,
            "strict_connected_output": not deliberately_disconnected,
            "original_regions": len(original_regions),
            "resulting_regions": len(active),
            "component_forced_original_labels": sorted(forced_labels),
            "component_forced_regions": [
                {
                    "original_label": label,
                    "source_class": repr(original_regions[label].group_key[0]),
                    "region_class": repr(original_regions[label].group_key[1]),
                    "cell_count": len(original_regions[label].nodes),
                    "mass": original_regions[label].mass,
                }
                for label in sorted(forced_labels)
            ],
            "unmerged_component_forced_original_labels": sorted(forced_labels - merged_original_labels),
            "candidate_edges": len(candidate_distances),
            "initial_candidate_evaluations": initial_candidate_evaluations,
            "merges": merges,
            "deliberately_disconnected_regions": deliberately_disconnected,
            "score_uncalibrated": self.contrast_tau is None
            or (self.contrast_sigma_design is None and self.contrast_s_diag is None),
        }
        self.diagnostics.append(call_diagnostics)

        output = labels.copy(data=consolidated_values)
        output.attrs = dict(labels.attrs)
        output.attrs["component_consolidation"] = "contrast_proximity"
        output.attrs["component_consolidation_diagnostics"] = json.dumps(
            call_diagnostics,
            sort_keys=True,
        )
        output.attrs["strict_connected_labels"] = not deliberately_disconnected
        return output

    def _initial_candidate_evaluations(
        self,
        original_regions: dict[int, _ActiveRegion],
        candidate_distances: dict[tuple[int, int], float],
        inverse_variance: float | npt.NDArray[np.float64],
    ) -> list[dict[str, Any]]:
        """Describe every initial proximity edge and its contrast decision."""
        result = []
        for (first_id, second_id), distance_km in sorted(candidate_distances.items()):
            first = original_regions[first_id]
            second = original_regions[second_id]
            score, reason, accepted = self._candidate_decision(
                first,
                second,
                inverse_variance,
            )
            result.append(
                {
                    "first_original_label": first_id,
                    "second_original_label": second_id,
                    "source_class": repr(first.group_key[0]),
                    "region_class": repr(first.group_key[1]),
                    "distance_km": distance_km,
                    "first_mass": first.mass,
                    "second_mass": second.mass,
                    "lambda": None if score is None else score.lambda_value,
                    "delta_dfs": None if score is None else score.delta_dfs,
                    "delta_eig": None if score is None else score.delta_eig,
                    "accepted": accepted,
                    "reason": reason,
                }
            )
        return result

    def _accepted_candidates(
        self,
        active: dict[int, _ActiveRegion],
        original_to_active: dict[int, int],
        candidate_distances: dict[tuple[int, int], float],
        inverse_variance: float | npt.NDArray[np.float64],
    ) -> list[
        tuple[
            tuple[Any, ...],
            int,
            int,
            float,
            SplitContrastScore | None,
            str,
        ]
    ]:
        """Return accepted, deterministically prioritized current candidates."""
        active_distances: dict[tuple[int, int], float] = {}
        for (first_original, second_original), distance in candidate_distances.items():
            first_id = original_to_active[first_original]
            second_id = original_to_active[second_original]
            if first_id == second_id:
                continue
            pair = cast(tuple[int, int], tuple(sorted((first_id, second_id))))
            active_distances[pair] = min(distance, active_distances.get(pair, np.inf))

        accepted = []
        for (first_id, second_id), distance in active_distances.items():
            first = active[first_id]
            second = active[second_id]
            if first.group_key != second.group_key:
                continue
            if not first.forced_only and not second.forced_only:
                continue

            score, reason, accepted_by_policy = self._candidate_decision(
                first,
                second,
                inverse_variance,
            )
            if not accepted_by_policy:
                continue
            if score is None:
                priority = (
                    -1.0,
                    distance,
                    first.original_labels,
                    second.original_labels,
                )
            else:
                priority = (
                    score.delta_eig,
                    distance,
                    first.original_labels,
                    second.original_labels,
                )
            accepted.append(
                (
                    priority,
                    first_id,
                    second_id,
                    distance,
                    score,
                    reason,
                )
            )
        return accepted

    def _candidate_decision(
        self,
        first: _ActiveRegion,
        second: _ActiveRegion,
        inverse_variance: float | npt.NDArray[np.float64],
    ) -> tuple[SplitContrastScore | None, str, bool]:
        """Return the score, reason, and acceptance for one candidate pair."""
        if first.mass <= 0.0 or second.mass <= 0.0:
            accepted = self.inactive_component_policy == "merge_nearest"
            reason = "inactive_nearest" if accepted else "inactive_kept"
            return None, reason, accepted

        score = _aggregate_contrast_score(
            first,
            second,
            tau=self.contrast_tau,
            inverse_variance=inverse_variance,
            uncalibrated=self.contrast_tau is None
            or (self.contrast_sigma_design is None and self.contrast_s_diag is None),
        )
        accepted = self._accepts_score(score)
        return score, "weak_contrast" if accepted else "contrast_threshold", accepted

    def _accepts_score(self, score: SplitContrastScore) -> bool:
        """Return true when all configured merge thresholds accept ``score``."""
        if self.max_merge_lambda is not None and score.lambda_value > self.max_merge_lambda:
            return False
        return not (self.max_merge_delta_eig is not None and score.delta_eig > self.max_merge_delta_eig)


def _align_label_inputs(
    labels: xr.DataArray,
    region_classes: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return aligned two-dimensional label and class arrays."""
    if labels.ndim != 2 or region_classes.ndim != 2:
        raise ValueError("labels and region_classes must be two-dimensional.")
    if set(labels.dims) != set(region_classes.dims):
        raise ValueError("labels and region_classes must use the same dimensions.")
    region_classes = region_classes.transpose(*labels.dims)
    return cast(
        tuple[xr.DataArray, xr.DataArray],
        tuple(xr.align(labels, region_classes, join="exact")),
    )


def _validate_labels(labels: xr.DataArray) -> npt.NDArray[np.int64]:
    """Return validated non-negative integer label values."""
    values = np.asarray(labels.to_numpy())
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must contain integers.")
    values = values.astype(np.int64)
    if (values < 0).any():
        raise ValueError("labels must be non-negative.")
    return values


def _aligned_source_values(
    source_classes: xr.DataArray | npt.ArrayLike | None,
    labels: xr.DataArray,
) -> np.ndarray:
    """Return a source field aligned to ``labels``."""
    if source_classes is None:
        result = np.empty(labels.shape, dtype=object)
        result[:] = "__all_sources__"
        return result
    if isinstance(source_classes, xr.DataArray):
        if source_classes.ndim != 2 or set(source_classes.dims) != set(labels.dims):
            raise ValueError("source_classes must use the label dimensions.")
        source_classes = source_classes.transpose(*labels.dims)
        _labels, source_classes = xr.align(labels, source_classes, join="exact")
        return source_classes.to_numpy()
    values = np.asarray(source_classes)
    if values.shape != labels.shape:
        raise ValueError("source_classes and labels must have the same shape.")
    return values


def _aligned_cell_weight(
    cell_weight: xr.DataArray | npt.ArrayLike,
    labels: xr.DataArray,
) -> xr.DataArray | npt.ArrayLike:
    """Require xarray cell weights to align exactly with the label grid."""
    if not isinstance(cell_weight, xr.DataArray):
        return cell_weight
    if cell_weight.ndim != 2 or set(cell_weight.dims) != set(labels.dims):
        raise ValueError("cell_weight must use the label dimensions.")
    cell_weight = cell_weight.transpose(*labels.dims)
    _labels, cell_weight = xr.align(labels, cell_weight, join="exact")
    return cell_weight


def _label_groups(
    labels: npt.NDArray[np.int64],
    region_classes: np.ndarray,
    source_classes: np.ndarray,
) -> dict[int, tuple[Hashable, Hashable]]:
    """Return the unique source/class group occupied by each label."""
    groups: dict[int, tuple[Hashable, Hashable]] = {}
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        group_values = {
            (
                _as_hashable(source_classes[index], name="source class"),
                _as_hashable(region_classes[index], name="region class"),
            )
            for index in zip(*np.where(labels == label), strict=True)
        }
        if len(group_values) != 1:
            raise ValueError(f"Basis label {label} crosses a source or region-class boundary.")
        groups[label] = group_values.pop()
    return groups


def _as_hashable(value: Any, *, name: str) -> Hashable:
    """Return a non-null hashable scalar or tuple."""
    is_null = pd.isna(value)
    if isinstance(is_null, (bool, np.bool_)) and bool(is_null):
        raise ValueError(f"A positive basis label occupies a null {name}.")
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"{name} value {value!r} is not hashable.") from exc
    return cast(Hashable, value)


def _validate_connected_labels(
    labels: npt.NDArray[np.int64],
    structure: npt.NDArray[np.bool_],
) -> None:
    """Require every positive input label to be strictly connected."""
    for label in (int(value) for value in np.unique(labels) if value > 0):
        if int(ndimage.label(labels == label, structure=structure)[1]) != 1:
            raise ValueError(
                "ContrastProximityComponentConsolidation requires strictly "
                f"connected input labels; label {label} is disconnected."
            )


def _initial_regions(
    labels: npt.NDArray[np.int64],
    group_by_label: dict[int, tuple[Hashable, Hashable]],
    weights: npt.NDArray[np.float64],
    contribution: npt.NDArray[np.float64],
) -> dict[int, _ActiveRegion]:
    """Aggregate nodes, mass, and design contribution by original label."""
    result = {}
    for label, group_key in group_by_label.items():
        rows, columns = np.where(labels == label)
        nodes = tuple(zip(rows.tolist(), columns.tolist(), strict=True))
        selected_weights = weights[rows, columns]
        result[label] = _ActiveRegion(
            original_labels=(label,),
            nodes=nodes,
            group_key=group_key,
            forced_only=False,
            mass=float(selected_weights.sum()),
            contribution=(contribution[:, rows, columns] * selected_weights.reshape(1, -1)).sum(axis=1),
        )
    return result


def _component_forced_labels(
    labels: npt.NDArray[np.int64],
    group_by_label: dict[int, tuple[Hashable, Hashable]],
    weights: npt.NDArray[np.float64],
    structure: npt.NDArray[np.bool_],
) -> set[int]:
    """Identify whole secondary components represented by exactly one region."""
    grouped_labels: dict[tuple[Hashable, Hashable], list[int]] = {}
    for label, group_key in group_by_label.items():
        grouped_labels.setdefault(group_key, []).append(label)

    forced: set[int] = set()
    for group_labels in grouped_labels.values():
        group_mask = np.isin(labels, group_labels)
        components, component_count = ndimage.label(group_mask, structure=structure)
        if component_count <= 1:
            continue
        component_ids = range(1, int(component_count) + 1)
        primary_component = max(
            component_ids,
            key=lambda component: (
                int(np.count_nonzero(components == component)),
                float(weights[components == component].sum()),
                -int(np.flatnonzero(components == component)[0]),
            ),
        )
        for component in component_ids:
            if component == primary_component:
                continue
            component_mask = components == component
            component_labels = {int(value) for value in np.unique(labels[component_mask]) if value > 0}
            if len(component_labels) != 1:
                continue
            label = component_labels.pop()
            if np.array_equal(labels == label, component_mask):
                forced.add(label)
    return forced


def _candidate_region_distances(
    labels: npt.NDArray[np.int64],
    group_by_label: dict[int, tuple[Hashable, Hashable]],
    forced_labels: set[int],
    geometry: LatLonGridGeometry,
    max_distance_km: float,
) -> dict[tuple[int, int], float]:
    """Return minimum cell-centre distances for eligible original-label pairs."""
    if not forced_labels:
        return {}
    mapped_rows, mapped_columns = np.where(labels > 0)
    mapped_labels = labels[mapped_rows, mapped_columns]
    mapped_points = _unit_sphere_points(
        geometry.latitudes[mapped_rows, mapped_columns],
        geometry.longitudes[mapped_rows, mapped_columns],
    )
    tree = cKDTree(mapped_points)
    max_angle = max_distance_km / (geometry.earth_radius_m / 1_000.0)
    max_chord = 2.0 * np.sin(min(max_angle, np.pi) / 2.0)

    distances: dict[tuple[int, int], float] = {}
    for forced_label in sorted(forced_labels):
        forced_mask = mapped_labels == forced_label
        forced_points = mapped_points[forced_mask]
        for point, neighbour_indices in zip(
            forced_points,
            tree.query_ball_point(forced_points, max_chord),
            strict=True,
        ):
            if not neighbour_indices:
                continue
            neighbour_points = mapped_points[neighbour_indices]
            neighbour_labels = mapped_labels[neighbour_indices]
            chord_lengths = np.linalg.norm(neighbour_points - point, axis=1)
            angles = 2.0 * np.arcsin(np.clip(chord_lengths / 2.0, 0.0, 1.0))
            distances_km = angles * geometry.earth_radius_m / 1_000.0
            for neighbour_label, distance_km in zip(
                neighbour_labels,
                distances_km,
                strict=True,
            ):
                neighbour_label = int(neighbour_label)
                if neighbour_label == forced_label:
                    continue
                if group_by_label[neighbour_label] != group_by_label[forced_label]:
                    continue
                pair = cast(
                    tuple[int, int],
                    tuple(sorted((forced_label, neighbour_label))),
                )
                distances[pair] = min(
                    float(distance_km),
                    distances.get(pair, np.inf),
                )
    return distances


def _unit_sphere_points(
    latitudes: npt.ArrayLike,
    longitudes: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return Cartesian unit-sphere points for latitude/longitude coordinates."""
    latitude_radians = np.deg2rad(np.asarray(latitudes, dtype=np.float64))
    longitude_radians = np.deg2rad(np.asarray(longitudes, dtype=np.float64))
    cos_latitude = np.cos(latitude_radians)
    return np.column_stack(
        (
            cos_latitude * np.cos(longitude_radians),
            cos_latitude * np.sin(longitude_radians),
            np.sin(latitude_radians),
        )
    )


def _inverse_design_variance(
    *,
    design_shape: tuple[int, ...],
    design_dims: tuple[Hashable, ...],
    sigma_design: float | None,
    s_diag: xr.DataArray | npt.ArrayLike | None,
) -> float | npt.NDArray[np.float64]:
    """Return scalar or per-observation inverse design variance."""
    if sigma_design is not None:
        return 1.0 / sigma_design**2
    if s_diag is None:
        return 1.0
    variance = _s_diag_values(
        s_diag,
        design_shape=design_shape,
        design_dims=design_dims,
    )
    if (variance <= 0.0).any() or not np.isfinite(variance).all():
        raise ValueError("contrast_s_diag entries must be positive finite variances.")
    return 1.0 / variance


def _aggregate_contrast_score(
    first: _ActiveRegion,
    second: _ActiveRegion,
    *,
    tau: float | None,
    inverse_variance: float | npt.NDArray[np.float64],
    uncalibrated: bool,
) -> SplitContrastScore:
    """Return the reverse-split score from cached regional aggregates."""
    total_mass = first.mass + second.mass
    contrast = (second.mass / total_mass) * first.contribution - (
        first.mass / total_mass
    ) * second.contribution
    tau_value = 1.0 if tau is None else float(tau)
    norm = float(np.sum(contrast**2 * inverse_variance))
    lambda_value = float(tau_value**2 * norm)
    return SplitContrastScore(
        contrast=contrast,
        lambda_value=lambda_value,
        delta_dfs=float(lambda_value / (1.0 + lambda_value)),
        delta_eig=float(0.5 * np.log1p(lambda_value)),
        mu_a=first.mass,
        mu_b=second.mass,
        tau=tau_value,
        uncalibrated=uncalibrated,
    )


def _ordered_merge_endpoints(
    first: _ActiveRegion,
    second: _ActiveRegion,
) -> tuple[_ActiveRegion, _ActiveRegion]:
    """Return a deterministic forced source and merge target."""
    if first.forced_only != second.forced_only:
        return (first, second) if first.forced_only else (second, first)
    return (first, second) if first.original_labels < second.original_labels else (second, first)


def _merge_active_regions(
    first: _ActiveRegion,
    second: _ActiveRegion,
) -> _ActiveRegion:
    """Return cached aggregate state for a merged pair."""
    return _ActiveRegion(
        original_labels=tuple(sorted((*first.original_labels, *second.original_labels))),
        nodes=(*first.nodes, *second.nodes),
        group_key=first.group_key,
        forced_only=first.forced_only and second.forced_only,
        mass=first.mass + second.mass,
        contribution=first.contribution + second.contribution,
    )


def _labels_from_active_regions(
    active: dict[int, _ActiveRegion],
    shape: tuple[int, int],
) -> tuple[npt.NDArray[np.int64], dict[int, _ActiveRegion]]:
    """Return compact labels and their active-region provenance."""
    values: npt.NDArray[np.int64] = np.zeros(shape, dtype=np.int64)
    final_groups: dict[int, _ActiveRegion] = {}
    for final_label, region in enumerate(
        sorted(active.values(), key=lambda value: value.original_labels),
        start=1,
    ):
        rows, columns = zip(*region.nodes, strict=True)
        values[np.asarray(rows), np.asarray(columns)] = final_label
        final_groups[final_label] = region
    return values, final_groups


def _deliberately_disconnected_regions(
    labels: npt.NDArray[np.int64],
    final_groups: dict[int, _ActiveRegion],
    structure: npt.NDArray[np.bool_],
) -> list[dict[str, Any]]:
    """Describe every deliberately disconnected consolidated output label."""
    result = []
    for label, region in final_groups.items():
        component_count = int(ndimage.label(labels == label, structure=structure)[1])
        if component_count > 1:
            result.append(
                {
                    "label": label,
                    "original_labels": list(region.original_labels),
                    "component_count": component_count,
                }
            )
    return result


def _validate_optional_non_negative(name: str, value: float | None) -> None:
    """Validate an optional non-negative finite scalar."""
    if value is not None and (value < 0.0 or not np.isfinite(value)):
        raise ValueError(f"{name} must be non-negative and finite.")


__all__ = [
    "ContrastProximityComponentConsolidation",
    "InactiveComponentPolicy",
]
