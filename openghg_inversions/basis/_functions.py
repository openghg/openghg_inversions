"""Load and generate basis functions from files, fluxes, and footprints."""

import getpass
import logging
from collections.abc import Hashable, Mapping
from functools import partial
from numbers import Integral
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import xarray as xr

from .algorithms import (
    AllocationMode,
    AxisParallelSplitStep,
    ContrastScoreSplitAcceptance,
    GreedySplitStrategy,
    NbasisAllocation,
    SplitStrategy,
    allocate_nbasis_by_class,
    combine_inner_outer_region_classes,
    normalize_spatial_grid,
    region_constrained_basis,
)
from .algorithms import quadtree_algorithm, weighted_algorithm

from openghg_inversions.array_ops import to_dense
from openghg_inversions.config.paths import Paths

logger = logging.getLogger(__name__)


openghginv_path = Paths.openghginv
_INNER_REGION_LABEL_ATTR = "inner_region_label"


BasisWeightRule = Literal["abs_of_sum", "sum_of_abs"]
"""How sector responses reduce to one weight field.

``"abs_of_sum"`` is ``mean_o(abs(sum_s R_s(o)))`` and ``"sum_of_abs"`` is
``sum_s mean_o(abs(R_s(o)))``. Both take the absolute value per observation.
"""


def _resolve_weight_rule(weight_rule: BasisWeightRule | None, *, multisector: bool) -> BasisWeightRule:
    """Return the explicit rule, or the state-layout default, rejecting unknown or unusable rules."""
    if weight_rule is None:
        return "sum_of_abs" if multisector else "abs_of_sum"
    if weight_rule not in ("abs_of_sum", "sum_of_abs"):
        raise ValueError("weight_rule must be 'abs_of_sum', 'sum_of_abs' or None.")
    if weight_rule == "sum_of_abs" and not multisector:
        raise ValueError(
            "weight_rule='sum_of_abs' needs source-resolved responses; use it with multisector=True."
        )
    return weight_rule


def _summed_response_magnitude(
    response: xr.DataArray, weight_rule: BasisWeightRule, *, sector_dim: str | None,
) -> xr.DataArray:
    """Sum absolute observation responses over ``time``, reducing sectors by ``weight_rule``.

    A grouped basis can call this with its group's sectors selected from the
    source-resolved response and its own rule.

    Args:
        response: One site's response with ``time``, spatial dimensions and,
            when ``sector_dim`` is given, a sector dimension.
        weight_rule: Sum sectors before (``"abs_of_sum"``) or after
            (``"sum_of_abs"``) the per-observation absolute value.
        sector_dim: Sector dimension of ``response``, or None for a total response.

    Returns:
        Lazy spatial field of summed per-observation magnitudes.
    """
    if sector_dim is not None and weight_rule == "abs_of_sum":
        response = response.sum(sector_dim)
    magnitude = abs(response)
    if sector_dim is not None and weight_rule == "sum_of_abs":
        magnitude = magnitude.sum(sector_dim)
    return magnitude.sum("time")


def basis_weights_from_data(
    site_data: Mapping[str, xr.Dataset],
    *,
    multisector: bool = False,
    weight_rule: BasisWeightRule | None = None,
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Materialize response-magnitude weights for generated basis fitting.

    Weights come from the observation-indexed responses already formed by the
    transport provider, so raw footprints (``fp`` or a history-resolved
    ``fp_time_resolved``) and native flux are not read. The absolute value is
    taken for each retained observation before averaging, so temporal sign
    changes cannot cancel. Observations are pooled across sites by count: each
    retained observation contributes once, and sites are not given equal
    weight.

    Two rules reduce sector responses ``R_s``:

    * ``"abs_of_sum"``: ``W = mean_o(abs(sum_s R_s))``. Sectors are summed
      before the absolute value, so opposing sector responses cancel and
      resolution follows the net signal.
    * ``"sum_of_abs"``: ``W = sum_s mean_o(abs(R_s))``. Each sector contributes
      its own magnitude, so cancellation between sectors does not remove
      resolution.

    By default the state layout selects the rule. Sectors sharing one scaling
    state (standard RHIME) use ``"abs_of_sum"`` on the selected-source total
    ``fp_x_flux``. Sectors with separate scaling states sharing one partition
    (multisector) use ``"sum_of_abs"`` on ``fp_x_flux_sectoral``. A
    multisector caller may choose ``"abs_of_sum"`` instead, for example when
    positively correlated sector scalings (such as GPP and TER) make the net
    response the informative signal.

    This named eager algorithm boundary computes the reduced two-dimensional
    field once, densifying sparse payloads. Borrowed inputs are unchanged.

    Args:
        site_data: Non-empty mapping of filtered site datasets. Each contains
            ``fp_x_flux`` (or ``fp_x_flux_sectoral`` with a ``source``
            dimension when ``multisector``) with ``time`` and the same spatial
            coordinates at every site. Sites may have no retained observations,
            provided at least one observation remains overall.
        multisector: Weight source-resolved responses for sectors with separate
            scaling states instead of the shared total response.
        weight_rule: ``"abs_of_sum"`` or ``"sum_of_abs"``; None selects the
            state-layout default. ``"sum_of_abs"`` requires ``multisector``.
        mask: Optional Boolean spatial mask with the weights' spatial sizes;
            its coordinates are overridden by the response grid. Cells outside
            it are dropped.

    Returns:
        Eager spatial weights on the response grid, in ``lat``/``lon`` order
        when those dimensions are present.

    Raises:
        ValueError: If the rule is invalid for the layout, no site or retained
            observation is supplied, a response is missing, or site response
            grids differ.
    """
    weight_rule = _resolve_weight_rule(weight_rule, multisector=multisector)
    name = "fp_x_flux_sectoral" if multisector else "fp_x_flux"
    if not site_data:
        raise ValueError("Generated basis weights require at least one site dataset.")
    magnitude_sums: list[xr.DataArray] = []
    n_observations = 0
    for site, dataset in site_data.items():
        if name not in dataset:
            raise ValueError(f"site_data[{site!r}] requires {name!r} for generated basis weights.")
        response = dataset[name]
        magnitude_sums.append(
            _summed_response_magnitude(response, weight_rule, sector_dim="source" if multisector else None)
        )
        n_observations += response.sizes["time"]
    if n_observations == 0:
        raise ValueError("Generated basis weights require at least one retained observation.")

    try:
        magnitude_sums = list(xr.align(*magnitude_sums, join="exact"))
    except ValueError as error:
        raise ValueError(f"Site {name!r} responses must share exact spatial coordinates.") from error
    weights = cast(xr.DataArray, sum(magnitude_sums)) / n_observations
    weights = weights.transpose("lat", "lon", ..., missing_dims="ignore")
    weights = to_dense(weights).compute()

    if mask is not None:
        weights, mask = xr.align(weights, mask.transpose(*weights.dims), join="override")
        weights = weights.where(mask, drop=True)
    return weights


def _validate_basis_algorithm(basis_algorithm: str) -> None:
    """Reject unsupported generated algorithms before fitting inputs are materialized."""
    if basis_algorithm not in ("quadtree", "weighted", "region_constrained"):
        raise ValueError(
            "Basis algorithm not recognised. Please use 'quadtree', 'weighted', "
            "'region_constrained', or input a basis function file"
        )


def basis_from_weights(
    weights: xr.DataArray,
    start_date: str,
    domain: str,
    basis_algorithm: str,
    nbasis: int,
    *,
    landsea_classes: xr.DataArray | None = None,
    landsea_indices: np.ndarray | None = None,
    **region_kwargs: Any,
) -> xr.DataArray:
    """Dispatch resolved spatial weights to one eager basis-fitting algorithm.

    ``basis_algorithm`` selects ``quadtree``, ``weighted`` or
    ``region_constrained``. Weighted fitting requires ``landsea_classes`` or
    a pre-aligned dense ``landsea_indices`` array; region-constrained fitting
    requires ``region_classes`` in ``region_kwargs``. Other region keywords
    follow :func:`region_constrained_basis_from_weights`.

    Inputs are borrowed; the selected fitting boundary materializes weights.
    Returns integer region labels on the spatial grid with a singleton time
    coordinate from ``start_date`` and domain metadata. Invalid algorithms or
    missing required maps raise ValueError; no resource discovery occurs here.
    """
    _validate_basis_algorithm(basis_algorithm)
    if basis_algorithm == "quadtree":
        return quadtree_basis_from_weights(weights, start_date, domain, nbasis=nbasis)
    if basis_algorithm == "weighted":
        return bucket_basis_from_weights(
            weights, start_date, domain, nbasis=nbasis,
            landsea_classes=landsea_classes, landsea_indices=landsea_indices,
        )
    if region_kwargs.get("region_classes") is None:
        raise ValueError("region_classes must be supplied for the region_constrained basis algorithm.")
    return region_constrained_basis_from_weights(
        weights, start_date, domain, nbasis=nbasis, **region_kwargs,
    )


def _fixed_outer_inner_region_label(regions: xr.DataArray) -> int:
    """Return the explicitly marked inner label, with legacy max-label fallback."""
    configured_label = regions.attrs.get(_INNER_REGION_LABEL_ATTR)
    if configured_label is None:
        return int(regions.max().item())
    if isinstance(configured_label, bool) or not isinstance(configured_label, Integral):
        raise ValueError(
            f"Fixed outer-region map attribute {_INNER_REGION_LABEL_ATTR!r} must be an integer."
        )
    inner_label = int(configured_label)
    if not bool((regions == inner_label).any().item()):
        raise ValueError(
            f"Fixed outer-region map marks inner label {inner_label}, but that label is absent."
        )
    return inner_label


def _sanitize_generated_basis_weights(
    weights: xr.DataArray,
    *,
    algorithm: str,
    require_nonzero: bool = False,
) -> xr.DataArray:
    """Materialize weights, replace non-finite cells with zero, and reject empty fields.

    Args:
        weights: Generated-basis spatial weight field.
        algorithm: Name included in validation errors.
        require_nonzero: Also reject a field with no non-zero finite weights.

    Returns:
        Eager weight field with non-finite values replaced by zero.

    Raises:
        ValueError: If no finite values remain, or ``require_nonzero`` is true
            and all finite values are zero.
    """
    weights = weights.as_numpy()
    finite = xr.apply_ufunc(np.isfinite, weights)
    if not bool(finite.any().item()):
        raise ValueError(f"{algorithm} generated-basis weights contain no finite values.")

    sanitized = weights.where(finite, 0.0)

    if require_nonzero and not bool((sanitized != 0.0).any().item()):
        raise ValueError(
            f"{algorithm} generated-basis weights contain no non-zero finite values "
            "after replacing non-finite values with zero."
        )

    return sanitized


def _normalise_weights_by_max(weights: xr.DataArray) -> xr.DataArray:
    """Return weights scaled by their maximum when the maximum is positive."""
    max_weight = float(weights.max())
    if max_weight > 0:
        return weights / max_weight
    return weights


def _normalise_weights_by_nonzero_max(weights: xr.DataArray) -> xr.DataArray:
    """Return weights scaled by their finite non-zero maximum."""
    max_weight = float(weights.max())
    if not np.isfinite(max_weight) or max_weight == 0.0:
        raise ValueError("generated-basis weights have no finite non-zero maximum.")
    return weights / max_weight


def _finalise_generated_basis(
    basis_field: xr.DataArray,
    *,
    start_date: str,
    domain: str,
) -> xr.DataArray:
    """Attach the legacy generated-basis dimensions, name, and metadata."""
    basis_field = basis_field.expand_dims({"time": [pd.to_datetime(start_date)]}, axis=-1)
    basis_field = basis_field.rename("basis")
    basis_field.attrs["creator"] = getpass.getuser()
    basis_field.attrs["date created"] = str(pd.Timestamp.today())
    basis_field.attrs["domain"] = domain
    return basis_field


def quadtree_basis_from_weights(
    weights: xr.DataArray,
    start_date: str,
    domain: str,
    *,
    nbasis: int = 100,
    seed: int | None = None,
) -> xr.DataArray:
    """Create a quadtree basis field from precomputed 2D weights.

    Args:
        weights: Two-dimensional basis weight field.
        start_date: Start date of the inversion period.
        domain: Domain across which to calculate basis functions.
        nbasis: Desired number of basis regions.
        seed: Optional seed passed to ``scipy.optimize.dual_annealing``.

    Returns:
        Basis field with ``lat``/``lon`` dimensions, a singleton ``time``
        dimension, and integer region labels.
    """
    weights = _sanitize_generated_basis_weights(weights, algorithm="quadtree", require_nonzero=True)
    func = partial(quadtree_algorithm, nbasis=nbasis, seed=seed)
    quad_basis = xr.apply_ufunc(func, weights)
    return _finalise_generated_basis(quad_basis, start_date=start_date, domain=domain)


def bucket_basis_from_weights(
    weights: xr.DataArray,
    start_date: str,
    domain: str,
    *,
    nbasis: int = 100,
    landsea_classes: xr.DataArray | None = None,
    landsea_indices: np.ndarray | None = None,
) -> xr.DataArray:
    """Create a legacy weighted bucket basis field from precomputed 2D weights.

    Weights are normalized by their nonzero maximum before fitting. This eager
    fitting boundary materializes a supplied class map; inputs are borrowed.
    Resolve map files before calling. One of ``landsea_classes`` or
    ``landsea_indices`` must be supplied, and labelled classes take precedence.

    Args:
        weights: Two-dimensional basis weight field.
        start_date: Start date of the inversion period.
        domain: Domain across which to calculate basis functions.
        nbasis: Desired number of basis regions.
        landsea_classes: Borrowed map on the weights grid. Coordinates are
            normalized to that grid before fitting. Classes 0 and 1 are split
            separately; other values remain label 0.
        landsea_indices: Pre-aligned dense class array with the same shape and
            axis order as ``weights``, used when no labelled map is supplied.

    Returns:
        Basis field with ``lat``/``lon`` dimensions, a singleton ``time``
        dimension, and integer region labels.
    """
    if landsea_classes is not None:
        landsea_classes = normalize_spatial_grid(
            weights, landsea_classes, reference_name="weights", candidate_name="land/sea class map",
        )
        landsea_indices = landsea_classes.transpose(*weights.dims).to_numpy()
    if landsea_indices is None:
        raise ValueError("landsea_classes must be resolved before weighted basis construction.")
    weights = _sanitize_generated_basis_weights(weights, algorithm="weighted bucket", require_nonzero=True)
    weights = _normalise_weights_by_nonzero_max(weights)
    algorithm_kwargs: dict[str, Any] = {
        "nregion": nbasis,
        "bucket": 1,
        "domain": domain,
    }
    if landsea_indices is not None:
        algorithm_kwargs["landsea_indices"] = landsea_indices
    func = partial(weighted_algorithm, **algorithm_kwargs)
    bucket_basis = xr.apply_ufunc(func, weights)
    return _finalise_generated_basis(bucket_basis, start_date=start_date, domain=domain)


def region_constrained_basis_from_weights(
    weights: xr.DataArray,
    start_date: str,
    domain: str,
    *,
    region_classes: xr.DataArray,
    nbasis: NbasisAllocation = 100,
    allocation: AllocationMode = "weight",
    min_regions_per_class: int = 1,
    split_strategy: SplitStrategy | None = None,
    split_acceptance: Literal["none", "contrast_score"] = "none",
    contrast_contribution: xr.DataArray | None = None,
    contrast_cell_weight: xr.DataArray | None = None,
    min_contrast_delta_eig: float | None = None,
    min_contrast_lambda: float | None = None,
    contrast_tau: float | None = None,
    contrast_sigma_design: float | None = None,
    contrast_s_diag: xr.DataArray | None = None,
) -> xr.DataArray:
    """Create constrained basis labels from weights and region classes.

    This is the weight-first adapter for the current ``region_constrained``
    basis algorithm. Labels are generated independently within each non-null
    region class.

    Args:
        weights: Two-dimensional basis weight field.
        start_date: Start date of the inversion period.
        domain: Domain across which to calculate basis functions.
        region_classes: Two-dimensional class field on the same spatial grid as
            ``weights``. A full-domain field may be supplied when ``weights``
            are a rectangular crop of that grid.
        nbasis: Total number of basis regions, or class-local allocation
            accepted by ``region_constrained_basis``.
        allocation: Automatic allocation mode used when ``nbasis`` is an
            integer. ``"weight"`` allocates by class total weight; ``"area"``
            allocates by mapped cell count.
        min_regions_per_class: Minimum automatic allocation for each non-empty
            mapped class.
        split_strategy: Optional class-local label generator. When omitted, the
            default greedy generator is used, optionally configured by
            ``split_acceptance``.
        split_acceptance: Optional split-acceptance criterion. The default
            ``"none"`` preserves existing behavior. ``"contrast_score"`` uses
            a mass-preserving observation-space contrast gate.
        contrast_contribution: Design contribution array for contrast scoring.
        contrast_cell_weight: Optional prior flux or split-mass field used as
            ``mu`` in contrast scoring. When omitted, the unnormalised input
            weights are used as a split-mass proxy.
        min_contrast_delta_eig: Optional minimum ``delta_eig`` threshold.
        min_contrast_lambda: Optional minimum ``lambda`` threshold.
        contrast_tau: Prior standard deviation of the split contrast
            coefficient ``delta = alpha_A - alpha_B``.
        contrast_sigma_design: Optional scalar design standard deviation,
            equivalent to ``S = contrast_sigma_design**2 I``.
        contrast_s_diag: Optional diagonal design covariance entries in the
            same row space as ``contrast_contribution``.

    Returns:
        Basis field with globally unique integer labels that do not cross
        ``region_classes`` values.

    Raises:
        ValueError: If both ``split_strategy`` and a non-default
            ``split_acceptance`` are supplied.
    """
    raw_weights = _sanitize_generated_basis_weights(weights, algorithm="region-constrained")
    weights = _normalise_weights_by_max(raw_weights)
    region_classes = _spatial_field_on_weights_grid(
        weights,
        region_classes,
        candidate_name="region_classes",
    )
    if split_strategy is not None and split_acceptance != "none":
        raise ValueError("split_strategy cannot be combined with a non-default split_acceptance.")
    if split_strategy is None:
        split_strategy = _region_constrained_split_strategy(
            split_acceptance=split_acceptance,
            contrast_contribution=contrast_contribution,
            contrast_cell_weight=contrast_cell_weight if contrast_cell_weight is not None else raw_weights,
            min_contrast_delta_eig=min_contrast_delta_eig,
            min_contrast_lambda=min_contrast_lambda,
            contrast_tau=contrast_tau,
            contrast_sigma_design=contrast_sigma_design,
            contrast_s_diag=contrast_s_diag,
        )

    constrained_basis = region_constrained_basis(
        weights,
        region_classes,
        nbasis,
        allocation=allocation,
        min_regions_per_class=min_regions_per_class,
        split_strategy=split_strategy,
    )
    return _finalise_generated_basis(constrained_basis, start_date=start_date, domain=domain)


def _spatial_field_on_weights_grid(
    weights: xr.DataArray,
    field: xr.DataArray,
    *,
    candidate_name: str,
) -> xr.DataArray:
    """Subset or reorder a spatial field before strict grid normalization.

    When both inputs have matching one-dimensional dimension coordinates and
    the candidate grid is at least as large, each weights coordinate selects its
    unique nearest numeric candidate coordinate or unique exact nonnumeric match.
    Physical-coordinate tolerances, metadata, and CRS compatibility are then
    validated by :func:`normalize_spatial_grid`.

    Args:
        weights: Two-dimensional weight field defining the target grid.
        field: Two-dimensional candidate field on the target grid or a
            larger grid containing it.
        candidate_name: Semantic field name used in alignment errors.

    Returns:
        Candidate field subset, transposed, and normalized to the weights grid.

    Raises:
        ValueError: If either input is not two-dimensional or their dimension
            names differ.
        xarray.AlignmentError: If the candidate field cannot be normalized to the
            physical weights grid.
    """
    if weights.ndim != 2 or field.ndim != 2 or set(weights.dims) != set(field.dims):
        return normalize_spatial_grid(
            weights,
            field,
            reference_name="weights",
            candidate_name=candidate_name,
        )

    field = field.transpose(*weights.dims)
    indexers: dict[Hashable, np.ndarray] = {}
    for dimension in weights.dims:
        if weights.sizes[dimension] > field.sizes[dimension]:
            break
        if dimension not in weights.coords or dimension not in field.coords:
            break

        weight_coordinate = weights.coords[dimension]
        field_coordinate = field.coords[dimension]
        if weight_coordinate.dims != (dimension,) or field_coordinate.dims != (dimension,):
            break

        weight_values = weight_coordinate.to_numpy()
        field_values = field_coordinate.to_numpy()
        if np.issubdtype(weight_values.dtype, np.number) and np.issubdtype(field_values.dtype, np.number):
            distances = np.abs(
                np.asarray(field_values, dtype=np.float64)[:, np.newaxis]
                - np.asarray(weight_values, dtype=np.float64)[np.newaxis, :]
            )
            nearest = np.argmin(distances, axis=0)
        else:
            nearest_matches: list[int] = []
            for value in weight_values:
                matches = np.flatnonzero(field_values == value)
                if matches.size != 1:
                    break
                nearest_matches.append(int(matches[0]))
            if len(nearest_matches) != weight_values.size:
                break
            nearest = np.asarray(nearest_matches, dtype=np.intp)

        if np.unique(nearest).size != nearest.size:
            break
        indexers[dimension] = nearest
    else:
        field = field.isel(indexers)

    return normalize_spatial_grid(
        weights,
        field,
        reference_name="weights",
        candidate_name=candidate_name,
    )


class _FixedOuterSplitStrategy:
    """Keep outer class maps fixed while dispatching an inner generator."""

    def __init__(
        self,
        inner_mask: np.ndarray,
        inner_strategy: SplitStrategy | None,
    ) -> None:
        """Initialize fixed-outer dispatch.

        Args:
            inner_mask: Boolean mask selecting cells handled by the inner
                splitting strategy.
            inner_strategy: Optional strategy for inner cells. When omitted,
                ``GreedySplitStrategy`` configured with
                ``AxisParallelSplitStep`` is used.
        """
        self.inner_mask = np.asarray(inner_mask, dtype=bool)
        self.inner_strategy = (
            inner_strategy
            if inner_strategy is not None
            else GreedySplitStrategy(split_step=AxisParallelSplitStep())
        )

    def __call__(
        self,
        weights: np.ndarray,
        class_mask: np.ndarray,
        target_regions: int,
    ) -> np.ndarray:
        """Return one fixed label for outer IDs or dispatch an inner class.

        Args:
            weights: Non-negative spatial weights passed to the inner strategy.
            class_mask: Boolean mask selecting the current class.
            target_regions: Requested label count for the current class.

        Returns:
            Integer labels with the same shape as ``class_mask``, positive
            inside the selected class and zero outside. Outer classes receive
            one fixed label; inner classes use the configured strategy.

        Raises:
            ValueError: Propagated if the inner strategy rejects the request.
        """
        if np.any(class_mask & ~self.inner_mask):
            labels = np.zeros(class_mask.shape, dtype=np.int64)
            labels[class_mask] = 1
            return labels
        return self.inner_strategy(weights, class_mask, target_regions)


def region_constrained_fixed_outer_basis_from_weights(
    weights: xr.DataArray,
    start_date: str,
    domain: str,
    *,
    nbasis: int | np.integer = 100,
    outer_regions: xr.DataArray,
    region_classes: xr.DataArray,
    allocation: AllocationMode = "weight",
    min_regions_per_class: int = 1,
    split_strategy: SplitStrategy | None = None,
) -> xr.DataArray:
    """Create a constrained fixed-outer basis from precomputed weights.

    The largest value in the InTEM outer-region map selects the bounded inner
    domain. ``nbasis`` is distributed only among the inner ``region_classes``;
    each distinct non-null outer-region value outside that maximum-valued inner
    domain receives one fixed basis label. Inner and outer class values are
    tagged before constrained splitting, so their final positive integer labels
    are globally disjoint.

    Args:
        weights: Non-negative, two-dimensional whole-domain basis weight field
            whose grid defines the output coordinates. Non-finite cells are
            replaced by zero, but at least one finite cell is required. Unlike
            :func:`region_constrained_basis_from_weights`, this fixed-outer
            adapter does not accept cropped weights because it emits the outer
            states as well as the bounded inner states.
        start_date: Start date of the inversion period.
        domain: Domain used for output metadata.
        nbasis: Total number of requested inner-region basis labels.
        outer_regions: Borrowed resolved InTEM outer-region map.
        region_classes: Borrowed inner classification, such as a binary
            land/sea or multi-country integer map. Each distinct non-null
            selected value is a class. Resolve maps before numerical work.
        allocation: Mode used to distribute ``nbasis`` across selected inner
            classes.
        min_regions_per_class: Minimum automatic allocation for each non-empty
            selected inner class.
        split_strategy: Optional class-local label generator applied only to
            bounded inner classes after layout composition. Outer IDs remain
            fixed maps even when they are disconnected. The inner default is
            ``GreedySplitStrategy`` configured with
            ``AxisParallelSplitStep``.

    Returns:
        Basis field on the weights grid with a singleton ``time`` dimension,
        standard generated-basis metadata, one target per non-null outer class,
        outside the maximum-valued inner domain, and ``nbasis`` targets
        allocated across the inner classes. Null outer cells remain label
        ``0``.

    Raises:
        TypeError: If ``nbasis`` is not a non-Boolean integral value.
        ValueError: If weights, dimensions, dimension names, outer-map values,
            inner classes, allocation, or split-strategy labels are invalid.
        xarray.AlignmentError: If the supplied fields are not on
            physically compatible spatial grids.
    """
    if isinstance(nbasis, (bool, np.bool_)) or not isinstance(nbasis, (Integral, np.integer)):
        raise TypeError("nbasis must be an integer inner-region target.")
    nbasis = int(nbasis)

    loaded_outer_regions = outer_regions
    loaded_region_classes = region_classes

    loaded_outer_regions = _spatial_field_on_weights_grid(
        weights,
        loaded_outer_regions,
        candidate_name="outer_regions",
    )
    loaded_region_classes = _spatial_field_on_weights_grid(
        weights,
        loaded_region_classes,
        candidate_name="region_classes",
    )

    try:
        inner_region_value = loaded_outer_regions.max(skipna=True).compute().item()
    except (TypeError, ValueError) as exc:
        raise ValueError("outer_regions must contain a finite maximum region value.") from exc
    if not isinstance(inner_region_value, (int, float, np.integer, np.floating)) or not np.isfinite(
        inner_region_value
    ):
        raise ValueError("outer_regions must contain a finite maximum region value.")

    inner_mask = loaded_outer_regions == inner_region_value
    inner_classes = loaded_region_classes.where(inner_mask)
    raw_weights = _sanitize_generated_basis_weights(weights, algorithm="fixed-outer region-constrained")
    raw_inner_targets = allocate_nbasis_by_class(
        raw_weights,
        inner_classes,
        nbasis,
        allocation=allocation,
        min_regions_per_class=min_regions_per_class,
    )
    if not raw_inner_targets:
        raise ValueError("The maximum-valued outer region contains no mapped inner classes.")

    targets: dict[Hashable, int] = {}
    outer_values = loaded_outer_regions.where(~inner_mask).to_numpy().ravel()
    for value in pd.unique(outer_values):
        if bool(pd.isna(value)):
            continue
        targets[("outer", cast(Hashable, value))] = 1
    targets.update({("inner", class_value): target for class_value, target in raw_inner_targets.items()})

    composed_classes = combine_inner_outer_region_classes(
        inner_mask,
        loaded_region_classes,
        loaded_outer_regions,
    )
    fixed_outer_strategy = _FixedOuterSplitStrategy(inner_mask.to_numpy(), split_strategy)
    return region_constrained_basis_from_weights(
        raw_weights,
        start_date,
        domain,
        region_classes=composed_classes,
        nbasis=targets,
        allocation=allocation,
        min_regions_per_class=min_regions_per_class,
        split_strategy=fixed_outer_strategy,
    )


def _region_constrained_split_strategy(
    *,
    split_acceptance: Literal["none", "contrast_score"],
    contrast_contribution: xr.DataArray | None,
    contrast_cell_weight: xr.DataArray,
    min_contrast_delta_eig: float | None,
    min_contrast_lambda: float | None,
    contrast_tau: float | None,
    contrast_sigma_design: float | None,
    contrast_s_diag: xr.DataArray | None,
):
    """Build the greedy split strategy with an optional contrast acceptance gate.

    Args:
        split_acceptance: ``"none"`` or ``"contrast_score"``.
        contrast_contribution: Design contribution array required for contrast
            scoring.
        contrast_cell_weight: Spatial weight field used as contrast split mass.
        min_contrast_delta_eig: Optional minimum ``delta_eig`` score.
        min_contrast_lambda: Optional minimum ``lambda`` score.
        contrast_tau: Optional prior standard deviation of the split contrast.
        contrast_sigma_design: Optional scalar design standard deviation.
        contrast_s_diag: Optional diagonal design covariance entries.

    Returns:
        Greedy split strategy configured for the requested acceptance mode.

    Raises:
        ValueError: If the acceptance mode is unknown or contrast scoring lacks
            a design contribution array.
    """
    if split_acceptance == "none":
        return GreedySplitStrategy(split_step=AxisParallelSplitStep())
    if split_acceptance != "contrast_score":
        raise ValueError("split_acceptance must be 'none' or 'contrast_score'.")
    if contrast_contribution is None:
        raise ValueError("contrast_contribution is required when split_acceptance='contrast_score'.")
    return GreedySplitStrategy(
        split_step=AxisParallelSplitStep(),
        split_acceptance=ContrastScoreSplitAcceptance(
            contribution=contrast_contribution,
            cell_weight=contrast_cell_weight,
            min_contrast_delta_eig=min_contrast_delta_eig,
            min_contrast_lambda=min_contrast_lambda,
            contrast_tau=contrast_tau,
            contrast_sigma_design=contrast_sigma_design,
            contrast_s_diag=contrast_s_diag,
        ),
    )


def fixed_outer_regions_basis_from_data(
    site_data: Mapping[str, xr.Dataset],
    flux: xr.DataArray,
    start_date: str,
    basis_algorithm: str,
    domain: str,
    nbasis: int = 100,
    *,
    multisector: bool = False,
    weight_rule: BasisWeightRule | None = None,
    outer_regions: xr.DataArray,
    landsea_classes: xr.DataArray | None = None,
    region_classes: xr.DataArray | None = None,
    region_allocation: AllocationMode = "weight",
    min_regions_per_class: int = 1,
    split_acceptance: Literal["none", "contrast_score"] = "none",
    contrast_contribution: xr.DataArray | None = None,
    contrast_cell_weight: xr.DataArray | None = None,
    min_contrast_delta_eig: float | None = None,
    min_contrast_lambda: float | None = None,
    contrast_tau: float | None = None,
    contrast_sigma_design: float | None = None,
    contrast_s_diag: xr.DataArray | None = None,
    allow_empty_inner_region: bool = False,
) -> xr.DataArray:
    """Use fixed InTEM outer regions and fit inner regions with an algorithm.

    The InTEM outer-region file defines known outer labels. Its optional
    ``inner_region_label`` attribute identifies the inner inversion region;
    legacy files without that metadata continue to use their largest label.
    This inner mask is passed to ``basis_algorithm`` and then inserted back
    into the fixed outer map. This fitting boundary materializes a dense local
    geometry map after coordinate alignment; borrowed Dask/sparse inputs remain
    unchanged. Inner weights come from :func:`basis_weights_from_data`, with
    its eager fitting boundary and state-layout weighting rule.

    For nested outer-domain preparation, ``allow_empty_inner_region=True``
    keeps the inner label unsplit when its masked weights contain finite values
    but all are zero. Otherwise the selected basis algorithm handles the inner
    weights normally.

    Args:
        site_data: Borrowed filtered site datasets containing ``fp_x_flux``,
            or ``fp_x_flux_sectoral`` when ``multisector``.
        flux: Borrowed native flux whose spatial grid is authoritative for the
            outer-region map. Its values are not used for weighting.
        start_date: Start date of the inversion period.
        basis_algorithm: Algorithm used to fit the inner region. Supported
            values are ``"quadtree"``, ``"weighted"``, and
            ``"region_constrained"``.
        domain: Domain across which to calculate basis functions.
        nbasis: Desired number of inner-region basis labels.
        multisector: Weight source-resolved responses for sectors with
            separate scaling states instead of the shared total response.
        weight_rule: ``"abs_of_sum"`` or ``"sum_of_abs"`` sector reduction;
            None selects the state-layout default. See
            :func:`basis_weights_from_data`.
        outer_regions: Borrowed resolved fixed outer-region map.
        landsea_classes: Borrowed resolved land/sea classes for weighted fitting.
        region_classes: Region or country class field used only with
            ``basis_algorithm="region_constrained"``. File loading should
            happen before calling this helper.
        region_allocation: Allocation mode for ``region_constrained``. One of
            ``"weight"`` or ``"area"``.
        min_regions_per_class: Minimum automatic allocation for each non-empty
            mapped class when using ``region_constrained``.
        split_acceptance: Optional split-acceptance criterion for
            ``region_constrained`` inner-region splitting.
        contrast_contribution: Design contribution array used only when
            ``split_acceptance="contrast_score"``.
        contrast_cell_weight: Optional prior flux or split-mass field used for
            contrast scoring.
        min_contrast_delta_eig: Optional minimum contrast ``delta_eig``.
        min_contrast_lambda: Optional minimum contrast ``lambda``.
        contrast_tau: Prior standard deviation of the split contrast
            coefficient. If omitted, ``tau=1`` is uncalibrated.
        contrast_sigma_design: Optional scalar design standard deviation.
        contrast_s_diag: Optional diagonal design covariance entries.
        allow_empty_inner_region: If true, keep the inner label unsplit when
            its masked weights contain finite values but all are zero. Intended
            for nested outer-domain preparation.

    Returns:
        Basis field with fixed outer labels and generated inner labels. When
        ``allow_empty_inner_region`` is true and finite inner weights are all
        zero, its label is kept unsplit.
    """
    _validate_basis_algorithm(basis_algorithm)
    _resolve_weight_rule(weight_rule, multisector=multisector)
    intem_regions = outer_regions

    # Validate the physical grid before adopting the authoritative flux coordinates.
    flux_grid = flux.isel(
        {dimension: 0 for dimension in flux.dims if dimension not in intem_regions.dims},
        drop=True,
    )
    intem_regions = normalize_spatial_grid(
        flux_grid,
        intem_regions,
        reference_name="flux",
        candidate_name="fixed outer-region map",
    )

    # Fixed-outer fitting needs concrete geometry for label discovery, cropped
    # masks and insertion. Materialize it once here; the caller's map is borrowed.
    intem_regions = to_dense(intem_regions).compute()
    inner_index = _fixed_outer_inner_region_label(intem_regions)

    mask = intem_regions == inner_index

    weights = basis_weights_from_data(site_data, multisector=multisector, weight_rule=weight_rule, mask=mask)
    if allow_empty_inner_region:
        finite_inner_weights = weights.to_numpy()
        finite_inner_weights = finite_inner_weights[np.isfinite(finite_inner_weights)]
        if finite_inner_weights.size and not bool((finite_inner_weights != 0.0).any()):
            logger.warning(
                f"Fixed outer-region map's inner label {inner_index} has no non-zero footprint*flux "
                "response; keeping it as a single fixed region instead of subdividing it further "
                "(allow_empty_inner_region=True)."
            )
            basis = intem_regions.copy().rename("basis")
            basis += 1  # intem_region_definitions.nc regions start at 0, not 1
            basis = basis.expand_dims({"time": [pd.to_datetime(start_date)]})
            return basis

    algorithm_kwargs: dict[str, Any] = {}
    if basis_algorithm == "weighted":
        if landsea_classes is None:
            raise ValueError("landsea_classes must be resolved before weighted basis construction.")
        landsea_classes = normalize_spatial_grid(
            intem_regions,
            landsea_classes,
            reference_name="fixed outer-region map",
            candidate_name="land/sea class map",
        )
        inner_landsea = landsea_classes.where(mask, drop=True)
        algorithm_kwargs["landsea_indices"] = inner_landsea.to_numpy()
    if basis_algorithm == "region_constrained":
        algorithm_kwargs.update(
            {
                "region_classes": region_classes,
                "allocation": region_allocation,
                "min_regions_per_class": min_regions_per_class,
                "split_acceptance": split_acceptance,
                "contrast_contribution": contrast_contribution,
                "contrast_cell_weight": contrast_cell_weight,
                "min_contrast_delta_eig": min_contrast_delta_eig,
                "min_contrast_lambda": min_contrast_lambda,
                "contrast_tau": contrast_tau,
                "contrast_sigma_design": contrast_sigma_design,
                "contrast_s_diag": contrast_s_diag,
            }
        )
    inner_region = basis_from_weights(
        weights, start_date, domain, basis_algorithm, nbasis, **algorithm_kwargs,
    )

    basis = intem_regions.copy().rename("basis")

    fixed_outer_values = intem_regions.where(~mask).to_numpy()
    finite_outer_values = fixed_outer_values[np.isfinite(fixed_outer_values)]
    if finite_outer_values.size == 0:
        raise ValueError("Fixed outer-region map must retain at least one label outside the inner region.")
    max_fixed_outer_label = int(finite_outer_values.max())

    loc_dict = {
        "lat": slice(inner_region.lat.min(), inner_region.lat.max() + 0.1),
        "lon": slice(inner_region.lon.min(), inner_region.lon.max() + 0.1),
    }
    # Generated algorithms use positive labels starting at one. Place those
    # labels above every retained fixed-outer label before converting the
    # complete field from zero-based to one-based labels below. Offsetting
    # from ``inner_index`` is only safe for legacy maps whose inner label is
    # already the maximum; EUHROB deliberately marks a non-maximum label.
    basis.loc[loc_dict] = (inner_region + max_fixed_outer_label).squeeze().values

    basis += 1  # intem_region_definitions.nc regions start at 0, not 1

    basis = basis.expand_dims({"time": [pd.to_datetime(start_date)]})

    return basis
