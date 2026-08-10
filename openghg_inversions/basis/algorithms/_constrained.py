"""Mask- and region-constrained basis generation helpers.

The helpers in this module are intentionally pure: callers pass an already
computed 2D weight field and an already loaded 2D class mask. File loading,
footprint/flux reduction, and domain-specific country lookup stay at the caller
boundary.

The main public entry point is :func:`region_constrained_basis`, with
:func:`allocate_nbasis_by_class` handling class-local target allocation. The
default implementation uses
:class:`~openghg_inversions.basis.algorithms.GreedySplitStrategy` configured
with :class:`~openghg_inversions.basis.algorithms.AxisParallelSplitStep`.
Callers can configure other group algorithms, including a greedy strategy using
:class:`~openghg_inversions.basis.algorithms.InertialSplitStep`.

Greedy split steps can optionally use a geometry object for split-shape
decisions. Geometry affects only coordinate comparisons such as long-axis
choice and inertial projection; contribution weights, label allocation, and
split-stopping policies remain separate.

Output labels preserve the input weight dimensions and coordinates. Positive
labels are globally unique, unmapped cells remain ``0``, and region-class
boundaries are never crossed.

Layered region-class helpers are also pure. They combine already loaded masks
into composite class labels so callers can express intersections such as
land/sea by inner/outer without baking file loading or runner configuration into
the algorithm. Use :func:`region_class_mask` to select their tuple-valued class
labels without NumPy treating a tuple as an array-like comparison operand.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from openghg.util import cf_ureg  # pyright: ignore[reportPrivateImportUsage]
import pandas as pd
from pint.errors import PintError
import xarray as xr
from scipy import ndimage

from ._partition import AxisParallelSplitStep, GreedySplitStrategy

# Allocation modes control how an integer ``nbasis`` is distributed across
# region classes before class-local splitting is applied.
AllocationMode: TypeAlias = Literal["weight", "area"]
NbasisAllocation: TypeAlias = int | Mapping[Hashable, int]
_GRID_COORDINATE_DEGREE_ATOL = 2.0e-5
_GRID_METADATA_KEYS = ("units", "calendar", "axis", "standard_name", "positive")
_DESCRIPTIVE_COORDINATE_ATTRS = {"comment", "history", "long_name", "longname", "source"}
_CRS_ATTRIBUTE_MARKERS = {"crs_wkt", "grid_mapping_name", "spatial_ref"}
_SPATIAL_COORDINATE_NAMES = {"lat", "latitude", "lon", "longitude", "x", "y"}
_SPATIAL_STANDARD_NAMES = {
    "latitude",
    "longitude",
    "projection_x_coordinate",
    "projection_y_coordinate",
}


class SplitStrategy(Protocol):
    """Strategy protocol for class-local basis splitting."""

    def __call__(
        self,
        weights: np.ndarray,
        class_mask: np.ndarray,
        target_regions: int,
    ) -> np.ndarray:
        """Split one class into local basis labels.

        Args:
            weights: Two-dimensional non-negative weight field for the full
                grid.
            class_mask: Two-dimensional Boolean mask with the same shape as
                ``weights``, selecting the class cells to split. When selected
                weights sum to zero, equal per-cell weights are used.
            target_regions: Requested number of local labels for this class.

        Returns:
            Integer, non-Boolean array with the same shape as ``class_mask``.
            Every selected cell has a positive label and every unselected cell
            is zero. Positive label values need not be contiguous, and the
            result may contain fewer regions than ``target_regions``.
        """
        ...


class ComponentConsolidationPolicy(Protocol):
    """Policy protocol for optional post-construction region consolidation."""

    def __call__(
        self,
        labels: xr.DataArray,
        region_classes: xr.DataArray,
    ) -> xr.DataArray:
        """Return labels after optional class-safe region consolidation."""
        ...


@dataclass
class ConnectedComponentSplitStrategy:
    """Allocate and split each connected piece of a class independently.

    A disconnected class requires at least one label per connected component.
    If ``target_regions`` is below that geographic minimum, the effective
    target is raised rather than assigning one label to disconnected cells.
    Targets above the minimum are allocated across components by weight, with
    the existing area fallback for all-zero weights.

    Attributes:
        split_strategy: Class-local strategy applied separately to each
            connected component.
        connectivity: Two-dimensional neighbourhood definition. ``1`` uses
            edge-sharing (four-neighbour) connectivity and ``2`` additionally
            includes corner-sharing (eight-neighbour) connectivity.
        diagnostics: Per-call requested, minimum, effective, and actual region
            counts.
    """

    split_strategy: SplitStrategy
    connectivity: int = 1
    diagnostics: list[dict[str, int]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Validate the requested two-dimensional connectivity."""
        if self.connectivity not in (1, 2):
            raise ValueError("connectivity must be 1 (edge) or 2 (edge and corner).")

    def __call__(
        self,
        weights: np.ndarray,
        class_mask: np.ndarray,
        target_regions: int,
    ) -> np.ndarray:
        """Return connected labels, raising the target to the geographic minimum."""
        if target_regions < 1:
            raise ValueError("target_regions must be at least 1.")
        if weights.shape != class_mask.shape or weights.ndim != 2:
            raise ValueError("weights and class_mask must be aligned two-dimensional arrays.")
        if not class_mask.any():
            return np.zeros(weights.shape, dtype=np.int64)

        component_labels, component_count = cast(
            tuple[np.ndarray, int],
            ndimage.label(
                class_mask,
                structure=ndimage.generate_binary_structure(2, self.connectivity),
            ),
        )
        effective_target = min(
            max(int(target_regions), int(component_count)),
            int(class_mask.sum()),
        )
        dims = ("row", "column")
        coordinates = {
            "row": np.arange(weights.shape[0]),
            "column": np.arange(weights.shape[1]),
        }
        allocation = allocate_nbasis_by_class(
            xr.DataArray(weights, dims=dims, coords=coordinates),
            xr.DataArray(component_labels, dims=dims, coords=coordinates),
            effective_target,
            allocation="weight",
            min_regions_per_class=1,
            unmapped_values=(0,),
        )

        labels = np.zeros(weights.shape, dtype=np.int64)
        next_region = 1
        for component in range(1, int(component_count) + 1):
            component_mask = component_labels == component
            local_labels = self.split_strategy(
                weights,
                component_mask,
                allocation[component],
            )
            if not np.array_equal(local_labels > 0, component_mask):
                raise RuntimeError("Connected split strategy did not preserve class coverage.")
            for local_region in _positive_labels(local_labels):
                region_mask = local_labels == local_region
                _component_labels, region_component_count = cast(
                    tuple[np.ndarray, int],
                    ndimage.label(
                        region_mask,
                        structure=ndimage.generate_binary_structure(2, self.connectivity),
                    ),
                )
                if int(region_component_count) != 1:
                    raise RuntimeError("Connected split strategy produced a disconnected label.")
                labels[region_mask] = next_region
                next_region += 1

        actual_regions = next_region - 1
        self.diagnostics.append(
            {
                "requested_target": int(target_regions),
                "connected_component_minimum": int(component_count),
                "effective_target": int(effective_target),
                "actual_regions": int(actual_regions),
            }
        )
        return labels


def region_constrained_basis(
    weights: xr.DataArray,
    region_classes: xr.DataArray,
    nbasis: NbasisAllocation,
    *,
    allocation: AllocationMode = "weight",
    min_regions_per_class: int = 1,
    split_strategy: SplitStrategy | None = None,
    component_consolidation: ComponentConsolidationPolicy | None = None,
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
        split_strategy: Class-local splitting strategy. Defaults to an explicit
            :class:`GreedySplitStrategy` using
            :class:`AxisParallelSplitStep`.
        component_consolidation: Optional policy applied to the globally
            relabelled basis after class-local construction. Policies that
            deliberately combine disconnected components must preserve class
            boundaries and report that strict connectivity no longer holds.
        unmapped_values: Additional class values to leave as output label ``0``.

    Returns:
        ``xarray.DataArray`` with the same dimensions and coordinates as
        ``weights``. Mapped cells receive globally unique positive integer
        labels; unmapped cells receive ``0``.

    Raises:
        ValueError: If either input is not two-dimensional, the dimension names
            differ, weights are invalid, or the requested allocation is
            impossible, or a split strategy returns labels with the wrong shape
            or dtype, non-positive labels inside its class mask, or nonzero
            labels outside its class mask.
        xarray.AlignmentError: If the inputs do not describe physically
            compatible spatial grids after transposition.

    Notes:
        Labels are guaranteed not to cross class boundaries because each class is
        split independently and relabelled with a global offset. The default
        strategy can assign one label to disconnected pieces of the
        same class if the class mask itself is disconnected; contiguity is not
        guaranteed by this helper.
    """
    weights, region_classes = _align_2d_inputs(
        weights,
        region_classes,
        reference_name="weights",
        candidate_name="region_classes",
    )
    weight_values = _validate_weights(weights)
    class_values = region_classes.to_numpy()
    mapped_classes, class_codes = _factorize_mapped_classes(class_values, unmapped_values)
    strategy = (
        split_strategy
        if split_strategy is not None
        else GreedySplitStrategy(split_step=AxisParallelSplitStep())
    )

    labels = np.zeros(weight_values.shape, dtype=np.int64)
    if not mapped_classes:
        return _labels_dataarray(labels, weights)

    targets = _allocate_nbasis_by_code(
        weight_values,
        mapped_classes,
        class_codes,
        nbasis,
        allocation=allocation,
        min_regions_per_class=min_regions_per_class,
    )

    next_label = 1
    for class_code, class_value in enumerate(mapped_classes):
        target_regions = targets[class_value]
        class_mask = class_codes == class_code
        local_labels = _validate_split_strategy_labels(
            strategy(weight_values, class_mask, target_regions),
            class_mask,
        )
        for local_label in _positive_labels(local_labels):
            labels[(local_labels == local_label) & class_mask] = next_label
            next_label += 1

    result = _labels_dataarray(labels, weights)
    if component_consolidation is not None:
        result = component_consolidation(result, region_classes)
    return result


def combine_inner_outer_region_classes(
    inner_mask: xr.DataArray,
    inner_classes: xr.DataArray,
    outer_classes: xr.DataArray,
    *,
    unmapped_values: Iterable[Hashable] = (),
    name: str = "region_classes",
) -> xr.DataArray:
    """Select and tag aligned inner- and outer-domain region classes.

    Args:
        inner_mask: Two-dimensional Boolean field selecting cells from
            ``inner_classes``. False cells select from ``outer_classes``.
        inner_classes: Two-dimensional class field for inner-domain cells.
        outer_classes: Two-dimensional class field for outer-domain cells.
        unmapped_values: Selected class values that should leave the output
            cell unmapped. Selected null values are always unmapped.
        name: Name for the returned ``DataArray``.

    Returns:
        Object-valued ``DataArray`` with the same dimensions and coordinates as
        ``inner_mask``. Mapped cells contain ``("inner", value)`` or
        ``("outer", value)`` tuples. Unmapped cells contain ``NaN``.

    Raises:
        ValueError: If any input is not two-dimensional, dimension-name sets
            differ, ``inner_mask`` is not Boolean, or a selected class value is
            not hashable.
        xarray.AlignmentError: If, after transposition to ``inner_mask``
            dimension order, an input does not describe a physically compatible
            spatial grid.

    Notes:
        Values on the unselected side do not affect the result, including null
        values. Domain tags prevent equal inner and outer class values from
        colliding when passed to :func:`region_constrained_basis`.

        This source-neutral composition helper advances `issue #449
        <https://github.com/openghg/openghg_inversions/issues/449>`_.
    """
    inner_mask, inner_classes = _align_2d_inputs(
        inner_mask,
        inner_classes,
        reference_name="inner_mask",
        candidate_name="inner_classes",
    )
    inner_mask, outer_classes = _align_2d_inputs(
        inner_mask,
        outer_classes,
        reference_name="inner_mask",
        candidate_name="outer_classes",
    )
    if not np.issubdtype(inner_mask.dtype, np.bool_):
        raise ValueError("inner_mask must be Boolean.")

    mask_values = inner_mask.to_numpy()
    inner_values = inner_classes.to_numpy()
    outer_values = outer_classes.to_numpy()
    unmapped = set(unmapped_values)
    class_values = np.empty(inner_mask.shape, dtype=object)
    class_values[:] = np.nan
    retained_labels: dict[tuple[Hashable, Hashable], tuple[Hashable, Hashable]] = {}

    for index in np.ndindex(inner_mask.shape):
        is_inner = bool(mask_values[index])
        value = cast(Hashable, inner_values[index] if is_inner else outer_values[index])
        if _is_unmapped_layer_value(value, unmapped):
            continue
        label = ("inner" if is_inner else "outer", value)
        class_values[index] = retained_labels.setdefault(label, label)

    return xr.DataArray(
        class_values,
        dims=inner_mask.dims,
        coords=inner_mask.coords,
        name=name,
    )


def intersect_region_class_layers(
    layers: Mapping[Hashable, xr.DataArray],
    *,
    unmapped_values: Iterable[Hashable] = (),
    name: str = "region_classes",
) -> xr.DataArray:
    """Intersect aligned region-class layers into composite class labels.

    Args:
        layers: Ordered mapping from layer name to two-dimensional class field.
            Mapping insertion order defines the order of values in each output
            class tuple.
        unmapped_values: Layer values that should leave the output cell
            unmapped. Null values in any layer are always unmapped.
        name: Name for the returned ``DataArray``.

    Returns:
        Object-valued ``DataArray`` with the same dimensions and coordinates as
        the first layer. Mapped cells contain tuples of layer values, while
        cells that are null or explicitly unmapped in any layer contain
        ``NaN``. Its ``region_class_layers`` attribute records the string form
        of each layer name in mapping insertion order.

    Raises:
        ValueError: If no layers are supplied, any layer is not two-dimensional,
            layer dimension names differ, or a mapped layer value is not
            hashable.
        xarray.AlignmentError: If, after transposition to the first layer's
            dimension order, any layer does not describe a physically
            compatible spatial grid.

    Notes:
        The tuple labels can be passed directly to
        :func:`region_constrained_basis`. This is the small lattice-style
        construction needed for layered masks such as land/sea crossed with an
        inner/outer rectangle. Use :func:`region_class_mask` rather than raw
        xarray equality when selecting a tuple label.
    """
    layer_items = list(layers.items())
    if not layer_items:
        raise ValueError("At least one region-class layer must be supplied.")

    template = layer_items[0][1]
    if template.ndim != 2:
        raise ValueError("Region-class layers must be two-dimensional.")

    template_name = f"region-class layer {layer_items[0][0]!r}"
    aligned_layers: list[xr.DataArray] = []
    for layer_name, layer in layer_items:
        _, aligned_layer = _align_2d_inputs(
            template,
            layer,
            reference_name=template_name,
            candidate_name=f"region-class layer {layer_name!r}",
        )
        aligned_layers.append(aligned_layer)

    layer_values = [layer.to_numpy() for layer in aligned_layers]
    unmapped = set(unmapped_values)
    class_values = np.empty(template.shape, dtype=object)
    class_values[:] = np.nan
    retained_labels: dict[tuple[Hashable, ...], tuple[Hashable, ...]] = {}

    for index in np.ndindex(template.shape):
        values: list[Hashable] = []
        mapped = True
        for layer in layer_values:
            value = cast(Hashable, layer[index])
            if _is_unmapped_layer_value(value, unmapped):
                mapped = False
                break
            values.append(value)
        if mapped:
            label = tuple(values)
            class_values[index] = retained_labels.setdefault(label, label)

    return xr.DataArray(
        class_values,
        dims=template.dims,
        coords=template.coords,
        name=name,
        attrs={"region_class_layers": tuple(str(layer_name) for layer_name, _ in layer_items)},
    )


def region_class_mask(
    region_classes: xr.DataArray,
    class_value: Hashable,
    *,
    name: str = "region_class_mask",
) -> xr.DataArray:
    """Select one scalar or tuple-valued region class reliably.

    Args:
        region_classes: Region-class values to compare with ``class_value``.
            Dimensions and all attached coordinates are preserved on the
            returned mask.
        class_value: Scalar or tuple-valued class label to select.
        name: Name for the returned Boolean ``DataArray``.

    Returns:
        Boolean ``DataArray`` that is true exactly where ``region_classes``
        contains ``class_value``.

    Notes:
        Raw expressions such as ``region_classes == ("land", "inner")`` are
        unreliable for object arrays because NumPy may treat the tuple as an
        array-like operand and broadcast its elements instead of comparing the
        tuple as one value. This helper performs elementwise tuple comparison
        through the same path used by constrained allocation and splitting.
        Object values are materialized eagerly; dask chunks are not preserved
        in the returned mask.
    """
    mask = _class_value_mask(region_classes.to_numpy(), class_value)
    return xr.DataArray(mask, dims=region_classes.dims, coords=region_classes.coords, name=name)


def allocate_nbasis_by_class(
    weights: xr.DataArray,
    region_classes: xr.DataArray,
    nbasis: NbasisAllocation,
    *,
    allocation: AllocationMode = "weight",
    min_regions_per_class: int = 1,
    unmapped_values: Iterable[Hashable] = (),
) -> dict[Hashable, int]:
    """Allocate class-local region targets for constrained basis generation.

    Args:
        weights: Two-dimensional non-negative weight field.
        region_classes: Two-dimensional class field aligned to ``weights``.
        nbasis: Total number of regions to distribute, or an explicit mapping
            from class value to class-local region target.
        allocation: Automatic allocation mode. ``"weight"`` uses class total
            weight, falling back to area if all mapped weights are zero.
            ``"area"`` uses mapped cell count.
        min_regions_per_class: Minimum automatic allocation for each non-empty
            mapped class.
        unmapped_values: Additional class values to leave unallocated.

    Returns:
        Mapping from mapped class value to target number of local regions.

    Raises:
        ValueError: If either input is not two-dimensional, the dimension names
            differ, weights are invalid, or the requested allocation is
            impossible.
        xarray.AlignmentError: If the inputs do not describe physically
            compatible spatial grids after transposition.
    """
    if min_regions_per_class < 0:
        raise ValueError("min_regions_per_class must be non-negative.")
    weights, region_classes = _align_2d_inputs(
        weights,
        region_classes,
        reference_name="weights",
        candidate_name="region_classes",
    )
    weight_values = _validate_weights(weights)
    mapped_classes, class_codes = _factorize_mapped_classes(
        region_classes.to_numpy(),
        unmapped_values,
    )
    return _allocate_nbasis_by_code(
        weight_values,
        mapped_classes,
        class_codes,
        nbasis,
        allocation=allocation,
        min_regions_per_class=min_regions_per_class,
    )


def _align_2d_inputs(
    reference: xr.DataArray,
    candidate: xr.DataArray,
    *,
    reference_name: str,
    candidate_name: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Validate and normalize two physically compatible 2D grids.

    Args:
        reference: Two-dimensional field whose dimension order and attached
            coordinates define the output grid.
        candidate: Two-dimensional field using the same dimension names and a
            physically compatible grid after transposition.
        reference_name: Caller-specific name for ``reference`` in errors.
        candidate_name: Caller-specific name for ``candidate`` in errors.

    Returns:
        ``reference`` and ``candidate`` with ``candidate`` transposed and
        normalized to the authoritative coordinates from ``reference``.

    Raises:
        ValueError: If either input is not two-dimensional or their dimension
            names differ.
        xarray.AlignmentError: If spatial coordinate names, dimensions, values,
            grid-defining metadata, or CRS definitions are incompatible.
    """
    if reference.ndim != 2:
        raise ValueError(f"{reference_name} must be two-dimensional.")
    if candidate.ndim != 2:
        raise ValueError(f"{candidate_name} must be two-dimensional.")
    if set(reference.dims) != set(candidate.dims):
        raise ValueError(f"{reference_name} and {candidate_name} must use the same dimensions.")

    candidate = candidate.transpose(*reference.dims)
    reference, candidate = _normalize_matching_grid_coordinates(
        reference,
        candidate,
        reference_name=reference_name,
        candidate_name=candidate_name,
    )
    try:
        reference, candidate = xr.align(reference, candidate, join="exact")
    except xr.AlignmentError as exc:
        raise xr.AlignmentError(
            f"{candidate_name} cannot be aligned exactly with {reference_name}: {exc}"
        ) from exc
    return reference, candidate


def normalize_spatial_grid(
    reference: xr.DataArray,
    candidate: xr.DataArray,
    *,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
) -> xr.DataArray:
    """Normalize a physically compatible two-dimensional field to a reference grid.

    Args:
        reference: Two-dimensional field whose dimension order, coordinate
            values, and nonconflicting metadata define the output grid.
        candidate: Two-dimensional field to validate and normalize.
        reference_name: Name used for ``reference`` in validation errors.
        candidate_name: Name used for ``candidate`` in validation errors.

    Returns:
        ``candidate`` transposed to the reference dimension order and assigned
        the reference grid coordinates, with nonconflicting coordinate metadata
        retained from both fields.

    Raises:
        ValueError: If either field is not two-dimensional or their dimension
            names differ.
        xarray.AlignmentError: If grid coordinates, grid-defining metadata, or
            CRS definitions are physically incompatible.

    Notes:
        This aligns to an arbitrary reference grid. OpenGHG's
        ``openghg.util.align_lat_lon`` instead canonicalizes one field against
        a named OpenGHG domain and does not validate arbitrary curvilinear grids,
        units, or CRS metadata.
    """
    _, normalized_candidate = _align_2d_inputs(
        reference,
        candidate,
        reference_name=reference_name,
        candidate_name=candidate_name,
    )
    return normalized_candidate


def _normalize_matching_grid_coordinates(
    reference: xr.DataArray,
    candidate: xr.DataArray,
    *,
    reference_name: str,
    candidate_name: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Validate a physical grid and adopt the reference coordinates.

    Args:
        reference: Grid whose spatial coordinates are authoritative.
        candidate: Transposed grid whose physical coordinates must be
            compatible with ``reference``.
        reference_name: Caller-specific name for ``reference`` in errors.
        candidate_name: Caller-specific name for ``candidate`` in errors.

    Raises:
        xarray.AlignmentError: If spatial coordinate names, dimensions, values,
            grid-defining metadata, or scalar CRS definitions conflict.

    Returns:
        Both arrays normalized to the spatial and CRS coordinate values from
        ``reference``, with nonconflicting metadata from either input retained.
        Descriptive coordinate attributes, numeric storage dtype, and unrelated
        scalar provenance coordinates do not define the physical grid.
    """
    reference_coordinates = _spatial_coordinate_names(reference)
    candidate_coordinates = _spatial_coordinate_names(candidate)
    if reference_coordinates != candidate_coordinates:
        missing = sorted(reference_coordinates - candidate_coordinates, key=str)
        unexpected = sorted(candidate_coordinates - reference_coordinates, key=str)
        raise xr.AlignmentError(
            f"{candidate_name} must define the same spatial grid coordinates as {reference_name}; "
            f"missing {missing!r}, unexpected {unexpected!r}."
        )

    for coordinate_name in sorted(reference_coordinates, key=str):
        reference_coordinate = reference.coords[coordinate_name]
        candidate_coordinate = candidate.coords[coordinate_name]
        if set(reference_coordinate.dims) != set(candidate_coordinate.dims):
            raise xr.AlignmentError(
                f"Coordinate {coordinate_name!r} on {candidate_name} has dimensions "
                f"{candidate_coordinate.dims!r}, which are incompatible with "
                f"{reference_coordinate.dims!r} on {reference_name}."
            )
        candidate_coordinate = candidate_coordinate.transpose(*reference_coordinate.dims)
        if not _coordinate_values_compatible(reference_coordinate, candidate_coordinate):
            raise xr.AlignmentError(
                f"Coordinate {coordinate_name!r} on {candidate_name} is not physically "
                f"compatible with {reference_name}."
            )
        _validate_grid_metadata(
            reference_coordinate,
            candidate_coordinate,
            coordinate_name=coordinate_name,
            reference_name=reference_name,
            candidate_name=candidate_name,
        )

    reference_crs = _crs_coordinate_names(reference, array_name=reference_name)
    candidate_crs = _crs_coordinate_names(candidate, array_name=candidate_name)
    if reference_crs != candidate_crs:
        raise xr.AlignmentError(
            f"{candidate_name} must define the same CRS coordinates as {reference_name}; "
            f"missing {sorted(reference_crs - candidate_crs, key=str)!r}, "
            f"unexpected {sorted(candidate_crs - reference_crs, key=str)!r}."
        )
    for coordinate_name in sorted(reference_crs, key=str):
        reference_coordinate = reference.coords[coordinate_name]
        candidate_coordinate = candidate.coords[coordinate_name]
        if not _coordinate_values_compatible(reference_coordinate, candidate_coordinate):
            raise xr.AlignmentError(
                f"CRS coordinate {coordinate_name!r} on {candidate_name} has a different value "
                f"from {reference_name}."
            )
        reference_attrs = {
            key: value
            for key, value in reference_coordinate.attrs.items()
            if key not in _DESCRIPTIVE_COORDINATE_ATTRS
        }
        candidate_attrs = {
            key: value
            for key, value in candidate_coordinate.attrs.items()
            if key not in _DESCRIPTIVE_COORDINATE_ATTRS
        }
        if reference_attrs.keys() != candidate_attrs.keys() or any(
            not _metadata_values_equal(reference_attrs[key], candidate_attrs[key]) for key in reference_attrs
        ):
            raise xr.AlignmentError(
                f"CRS coordinate {coordinate_name!r} on {candidate_name} conflicts with {reference_name}."
            )

    authoritative_coordinates = {
        name: _coordinate_with_merged_metadata(reference.coords[name], candidate.coords[name])
        for name in reference_coordinates | reference_crs
    }
    return (
        reference.assign_coords(authoritative_coordinates),
        candidate.assign_coords(authoritative_coordinates),
    )


def _coordinate_with_merged_metadata(
    reference: xr.DataArray,
    candidate: xr.DataArray,
) -> xr.DataArray:
    """Keep reference values while retaining nonconflicting metadata from both inputs."""
    merged_attributes = dict(candidate.attrs)
    merged_attributes.update(reference.attrs)
    coordinate = reference.copy(deep=False)
    coordinate.attrs = merged_attributes
    return coordinate


def _spatial_coordinate_names(array: xr.DataArray) -> set[Hashable]:
    """Return dimension indexes and recognized CF horizontal coordinates."""
    grid_dimensions = set(array.dims)
    return {
        name
        for name, coordinate in array.coords.items()
        if coordinate.dims
        and set(coordinate.dims).issubset(grid_dimensions)
        and _is_spatial_coordinate(name, coordinate, grid_dimensions)
    }


def _is_spatial_coordinate(
    name: Hashable,
    coordinate: xr.DataArray,
    grid_dimensions: set[Hashable],
) -> bool:
    """Return whether a coordinate participates in horizontal grid identity.

    Args:
        name: Coordinate name.
        coordinate: Coordinate array whose dimensions and CF metadata are
            inspected.
        grid_dimensions: Dimension names defining the horizontal grid.

    Returns:
        True when the coordinate is a dimension index or is identified as a
        horizontal coordinate by its name or CF metadata.
    """
    if name in grid_dimensions and coordinate.dims == (name,):
        return True
    if str(name).lower() in _SPATIAL_COORDINATE_NAMES:
        return True
    if str(coordinate.attrs.get("standard_name", "")).lower() in _SPATIAL_STANDARD_NAMES:
        return True
    return str(coordinate.attrs.get("axis", "")).upper() in {"X", "Y"}


def _crs_coordinate_names(array: xr.DataArray, *, array_name: str) -> set[Hashable]:
    """Return recognized CRS coordinates and reject unresolved mappings."""
    names = {
        name
        for name, coordinate in array.coords.items()
        if not coordinate.dims and _CRS_ATTRIBUTE_MARKERS.intersection(coordinate.attrs)
    }
    grid_mapping = array.attrs.get("grid_mapping")
    if isinstance(grid_mapping, str):
        referenced_name = grid_mapping.split(":", maxsplit=1)[0].strip()
        if referenced_name not in array.coords:
            raise xr.AlignmentError(
                f"{array_name} references grid mapping {referenced_name!r}, but it is not an "
                "attached coordinate; load with decode_coords='all' or attach it before alignment."
            )
        names.add(referenced_name)
    return names


def _coordinate_values_compatible(reference: xr.DataArray, candidate: xr.DataArray) -> bool:
    """Return whether coordinate values describe the same physical locations."""
    if reference.shape != candidate.shape:
        return False
    reference_values = reference.to_numpy()
    candidate_values = candidate.to_numpy()
    if np.issubdtype(reference_values.dtype, np.number) and np.issubdtype(candidate_values.dtype, np.number):
        return bool(
            np.allclose(
                reference_values,
                candidate_values,
                rtol=0.0,
                atol=_coordinate_absolute_tolerance(reference, candidate),
                equal_nan=True,
            )
        )
    if np.array_equal(reference_values, candidate_values):
        return True
    reference_null = pd.isna(reference_values)
    candidate_null = pd.isna(candidate_values)
    return bool(
        np.array_equal(reference_null, candidate_null)
        and np.all(reference_null | (reference_values == candidate_values))
    )


def _coordinate_absolute_tolerance(reference: xr.DataArray, candidate: xr.DataArray) -> float:
    """Return a unit- and storage-precision-aware absolute tolerance."""
    if reference.ndim == 0:
        return 0.0
    values = np.concatenate(
        [
            np.asarray(reference.to_numpy(), dtype=np.float64).ravel(),
            np.asarray(candidate.to_numpy(), dtype=np.float64).ravel(),
        ]
    )
    finite_values = values[np.isfinite(values)]
    scale = max(1.0, float(np.max(np.abs(finite_values))) if finite_values.size else 1.0)
    floating_dtypes = [
        dtype for dtype in (reference.dtype, candidate.dtype) if np.issubdtype(dtype, np.floating)
    ]
    precision_tolerance = max(
        (8.0 * np.finfo(dtype).eps * scale for dtype in floating_dtypes),
        default=0.0,
    )

    units = str(reference.attrs.get("units") or candidate.attrs.get("units") or "").lower()
    if "radian" in units or units == "rad":
        tolerance = max(precision_tolerance, float(np.deg2rad(_GRID_COORDINATE_DEGREE_ATOL)))
    elif "degree" in units:
        tolerance = max(precision_tolerance, _GRID_COORDINATE_DEGREE_ATOL)
    else:
        tolerance = precision_tolerance

    grid_spacing = _representative_grid_spacing(reference)
    if grid_spacing is not None:
        tolerance = min(tolerance, grid_spacing * 1.0e-3)
    return float(tolerance)


def _representative_grid_spacing(coordinate: xr.DataArray) -> float | None:
    """Return the smallest median nonzero step across coordinate dimensions."""
    values = np.asarray(coordinate.to_numpy(), dtype=np.float64)
    axis_spacings: list[float] = []
    for axis in range(values.ndim):
        differences = np.abs(np.diff(values, axis=axis))
        finite_nonzero = differences[np.isfinite(differences) & (differences > 0.0)]
        if finite_nonzero.size:
            axis_spacings.append(float(np.median(finite_nonzero)))
    return min(axis_spacings) if axis_spacings else None


def _metadata_values_equal(reference: object, candidate: object) -> bool:
    """Compare scalar or array-valued metadata without ambiguous truth values."""
    reference_values = np.asarray(reference)
    candidate_values = np.asarray(candidate)
    if reference_values.shape != candidate_values.shape:
        return False
    if np.issubdtype(reference_values.dtype, np.number) and np.issubdtype(candidate_values.dtype, np.number):
        return bool(np.allclose(reference_values, candidate_values, rtol=0.0, atol=0.0, equal_nan=True))
    return bool(np.array_equal(reference_values, candidate_values))


def _validate_grid_metadata(
    reference: xr.DataArray,
    candidate: xr.DataArray,
    *,
    coordinate_name: Hashable,
    reference_name: str,
    candidate_name: str,
) -> None:
    """Reject conflicting grid-defining metadata when both sides provide it."""
    for attribute_name in _GRID_METADATA_KEYS:
        reference_value = reference.attrs.get(attribute_name)
        candidate_value = candidate.attrs.get(attribute_name)
        if (
            reference_value is not None
            and candidate_value is not None
            and not _grid_metadata_values_equal(attribute_name, reference_value, candidate_value)
        ):
            raise xr.AlignmentError(
                f"Coordinate {coordinate_name!r} on {candidate_name} has conflicting "
                f"{attribute_name!r} metadata from {reference_name}."
            )


def _grid_metadata_values_equal(attribute_name: str, reference: object, candidate: object) -> bool:
    """Return whether grid metadata values are physically equivalent."""
    if attribute_name == "units" and isinstance(reference, str) and isinstance(candidate, str):
        try:
            return cf_ureg.parse_units(reference) == cf_ureg.parse_units(candidate)
        except (PintError, TypeError, ValueError):
            # Preserve the historical exact-string behavior for unit labels
            # outside OpenGHG's CF-aware registry.
            return reference == candidate
    return _metadata_values_equal(reference, candidate)


def _validate_weights(weights: xr.DataArray) -> np.ndarray:
    """Return finite non-negative weights as a float NumPy array."""
    values = np.asarray(weights.to_numpy(), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite.")
    if (values < 0.0).any():
        raise ValueError("weights must be non-negative.")
    return values


def _factorize_mapped_classes(
    class_values: npt.NDArray[np.object_] | np.ndarray,
    unmapped_values: Iterable[Hashable],
) -> tuple[list[Hashable], np.ndarray]:
    """Factor mapped class values once into first-seen integer codes.

    Args:
        class_values: Raw class array from the aligned ``region_classes`` input.
        unmapped_values: Additional class values that should remain output label
            ``0``.

    Returns:
        Unique mapped class values in first-seen order and a same-shaped integer
        code array. Mapped codes are contiguous from zero; null and explicitly
        unmapped cells use ``-1``.

    Raises:
        ValueError: If a mapped class value is not hashable.
    """
    unmapped = set(unmapped_values)
    try:
        raw_codes, unique_values = pd.factorize(
            class_values.ravel(),
            sort=False,
            use_na_sentinel=True,
        )
    except TypeError as exc:
        raise ValueError("Region-class values must be hashable.") from exc

    classes: list[Hashable] = []
    raw_to_mapped = np.full(len(unique_values), -1, dtype=np.intp)
    for raw_code, value in enumerate(unique_values):
        class_value = cast(Hashable, value)
        if _is_unmapped_layer_value(class_value, unmapped):
            continue
        raw_to_mapped[raw_code] = len(classes)
        classes.append(class_value)

    class_codes = np.full(raw_codes.shape, -1, dtype=np.intp)
    present = raw_codes >= 0
    class_codes[present] = raw_to_mapped[raw_codes[present]]
    return classes, class_codes.reshape(class_values.shape)


def _allocate_nbasis_by_code(
    weight_values: np.ndarray,
    mapped_classes: list[Hashable],
    class_codes: np.ndarray,
    nbasis: NbasisAllocation,
    *,
    allocation: AllocationMode,
    min_regions_per_class: int,
) -> dict[Hashable, int]:
    """Allocate class targets using already-factorized integer class codes.

    Args:
        weight_values: Non-negative spatial weights aligned with
            ``class_codes``.
        mapped_classes: Class values in the order represented by non-negative
            codes.
        class_codes: Integer class codes aligned with ``weight_values``.
            Mapped codes are contiguous from zero; ``-1`` marks unmapped cells.
        nbasis: Total requested region count or an explicit per-class
            allocation.
        allocation: ``"weight"`` to allocate by summed weight or ``"area"``
            to allocate by mapped cell count.
        min_regions_per_class: Minimum automatic allocation for each non-empty
            mapped class.

    Returns:
        Mapping from each mapped class to its allocated region count.

    Raises:
        ValueError: If the minimum or requested total is negative, allocation
            is requested without mapped classes, the request exceeds mapped
            capacity or required minima, the explicit allocation is invalid,
            or ``allocation`` is unsupported.
    """
    if min_regions_per_class < 0:
        raise ValueError("min_regions_per_class must be non-negative.")
    if not mapped_classes:
        if not isinstance(nbasis, Mapping) and nbasis != 0:
            raise ValueError("Cannot allocate basis regions without mapped classes.")
        return {}

    present_codes = class_codes[class_codes >= 0]
    capacity_values = np.bincount(present_codes, minlength=len(mapped_classes))
    capacities = {
        class_value: int(capacity_values[class_code]) for class_code, class_value in enumerate(mapped_classes)
    }
    if isinstance(nbasis, Mapping):
        return _explicit_allocation(mapped_classes, nbasis, capacities)
    if nbasis < 0:
        raise ValueError("nbasis must be non-negative.")

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

    scores = _allocation_scores(weight_values, class_codes, mapped_classes, allocation)
    return _distribute_regions(mapped_classes, scores, minima, capacities, nbasis)


def _is_unmapped_layer_value(value: Hashable, unmapped_values: set[Hashable]) -> bool:
    """Return true when a layer value should not be mapped."""
    isna = pd.isna(value)
    if isinstance(isna, (bool, np.bool_)) and bool(isna):
        return True
    try:
        return value in unmapped_values
    except TypeError as exc:
        raise ValueError(f"Region-class layer value {value!r} is not hashable.") from exc


def _class_value_mask(class_values: np.ndarray, class_value: Hashable) -> np.ndarray:
    """Return a Boolean mask for one class value, including tuple labels."""
    if isinstance(class_value, tuple) or class_values.dtype == object:
        return np.asarray(
            np.frompyfunc(lambda value: _class_values_equal(value, class_value), 1, 1)(class_values),
            dtype=bool,
        )

    try:
        mask = class_values == class_value
    except ValueError:
        mask = np.frompyfunc(lambda value: _class_values_equal(value, class_value), 1, 1)(class_values)
    if not isinstance(mask, np.ndarray) or mask.shape != class_values.shape:
        mask = np.frompyfunc(lambda value: _class_values_equal(value, class_value), 1, 1)(class_values)
    return np.asarray(mask, dtype=bool)


def _class_values_equal(value: object, class_value: Hashable) -> bool:
    """Compare one class value without treating missing values as truthy."""
    try:
        result = value == class_value
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _explicit_allocation(
    mapped_classes: list[Hashable],
    nbasis: Mapping[Hashable, int],
    capacities: Mapping[Hashable, int],
) -> dict[Hashable, int]:
    """Validate and normalize a caller-supplied per-class allocation.

    Args:
        mapped_classes: Mapped class values that need allocations.
        nbasis: Caller-supplied mapping from class value to region target.
        capacities: Maximum possible labels per class, currently the number of
            mapped cells for that class.

    Returns:
        Integer allocation for each mapped class, preserving ``mapped_classes``
        order.

    Raises:
        ValueError: If an allocation is missing, non-positive, or greater than
            class capacity.
    """
    missing = [class_value for class_value in mapped_classes if class_value not in nbasis]
    if missing:
        raise ValueError(f"Explicit nbasis allocation is missing classes: {missing!r}.")

    allocations = {class_value: int(nbasis[class_value]) for class_value in mapped_classes}
    invalid = {class_value: target for class_value, target in allocations.items() if target < 1}
    if invalid:
        raise ValueError(f"Class allocations must be positive for mapped classes: {invalid!r}.")
    over_allocated = {
        class_value: target for class_value, target in allocations.items() if target > capacities[class_value]
    }
    if over_allocated:
        raise ValueError(f"Class allocations exceed mapped cell counts: {over_allocated!r}.")
    return allocations


def _allocation_scores(
    weights: np.ndarray,
    class_codes: np.ndarray,
    mapped_classes: list[Hashable],
    allocation: AllocationMode,
) -> dict[Hashable, float]:
    """Compute proportional allocation scores for each mapped class.

    Args:
        weights: Non-negative weight values aligned to ``class_codes``.
        class_codes: Factorized integer class codes aligned to ``weights``.
        mapped_classes: Class values eligible for allocation.
        allocation: ``"weight"`` to score by total class weight, or ``"area"``
            to score by mapped cell count.

    Returns:
        Non-negative score for each mapped class.

    Raises:
        ValueError: If ``allocation`` is not supported.
    """
    if allocation == "area":
        counts = np.bincount(class_codes[class_codes >= 0], minlength=len(mapped_classes))
        return {
            class_value: float(counts[class_code]) for class_code, class_value in enumerate(mapped_classes)
        }
    if allocation != "weight":
        raise ValueError("allocation must be 'weight' or 'area'.")

    present = class_codes >= 0
    weight_sums = np.bincount(
        class_codes[present],
        weights=weights[present],
        minlength=len(mapped_classes),
    )
    scores = {
        class_value: float(weight_sums[class_code]) for class_code, class_value in enumerate(mapped_classes)
    }
    if sum(scores.values()) == 0.0:
        return _allocation_scores(weights, class_codes, mapped_classes, "area")
    return scores


def _distribute_regions(
    mapped_classes: list[Hashable],
    scores: Mapping[Hashable, float],
    minima: Mapping[Hashable, int],
    capacities: Mapping[Hashable, int],
    nbasis: int,
) -> dict[Hashable, int]:
    """Distribute remaining regions by score while respecting minima and capacity.

    Args:
        mapped_classes: Class values in deterministic allocation order.
        scores: Proportional allocation scores per class.
        minima: Initial allocation per class.
        capacities: Maximum allocation per class.
        nbasis: Total number of regions requested across all mapped classes.

    Returns:
        Allocation whose values sum to ``nbasis`` unless capacities are already
        exhausted by validated inputs.
    """
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


def _validate_split_strategy_labels(labels: np.ndarray, class_mask: np.ndarray) -> np.ndarray:
    """Validate a class-local split-strategy result against its class mask.

    Args:
        labels: Strategy result to convert to a NumPy array and validate.
        class_mask: Boolean mask for the class passed to the strategy.

    Returns:
        NumPy-converted integer label array.

    Raises:
        ValueError: If the shape differs from ``class_mask``, the dtype is not
            a non-Boolean integer dtype, any class cell is non-positive, or any
            cell outside the class is nonzero.
    """
    label_values = np.asarray(labels)
    if label_values.shape != class_mask.shape:
        raise ValueError(
            "Split strategy labels must have the same shape as the class grid; "
            f"got {label_values.shape}, expected {class_mask.shape}."
        )
    if np.issubdtype(label_values.dtype, np.bool_) or not np.issubdtype(label_values.dtype, np.integer):
        raise ValueError("Split strategy labels must have an integer, non-boolean dtype.")
    if not np.all(label_values[class_mask] > 0):
        raise ValueError("Split strategy labels must be strictly positive on every class-mask cell.")
    if not np.all(label_values[~class_mask] == 0):
        raise ValueError("Split strategy labels must be exactly zero outside the class mask.")
    return label_values


def _positive_labels(labels: np.ndarray) -> np.ndarray:
    """Return sorted unique positive labels from an integer label array."""
    unique = np.unique(labels)
    return unique[unique > 0]


def _labels_dataarray(labels: np.ndarray, weights: xr.DataArray) -> xr.DataArray:
    """Wrap a label array in a ``basis`` DataArray using weight coordinates."""
    return xr.DataArray(labels, dims=weights.dims, coords=weights.coords, name="basis")


__all__ = [
    "AllocationMode",
    "ComponentConsolidationPolicy",
    "ConnectedComponentSplitStrategy",
    "NbasisAllocation",
    "SplitStrategy",
    "allocate_nbasis_by_class",
    "combine_inner_outer_region_classes",
    "intersect_region_class_layers",
    "normalize_spatial_grid",
    "region_class_mask",
    "region_constrained_basis",
]
