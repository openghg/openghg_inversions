"""Calibrate native prior covariance amplitudes against functional SD targets.

The native state is a grid of dimensionless flux scaling perturbations ``x``
for each source ``s``, with prior covariance

``B_s = sum_c a_{s,c}**2 M_c K M_c``,

where ``K`` is the unit-amplitude separable exponential kernel of
:class:`~openghg_inversions.native_covariance.SeparableExponentialCovariance`
with correlation length ``L``, ``M_c`` is the indicator of native class ``c``
and ``a_{s,c}`` is the native grid-cell standard deviation.  Without class
labels there is one class covering the grid.  Sources are independent, as in
:class:`~openghg_inversions.source_covariance.IndependentSourceCovariance`, so
a cross-source correlation (for example between GPP and TER) is not modelled.

A declared functional ``r`` (for example a country or regional mask multiplied
by cell area) and a source flux ``f_s`` define the native total-flux weights
``q_{s,r} = f_s * w_r``.  The implied prior standard deviation of that total is

``SD_{s,r} = sqrt(q_{s,r}^T B_s q_{s,r}) = sqrt(sum_c a_{s,c}**2 v_{s,r,c})``,

with ``v_{s,r,c} = (M_c q)^T K (M_c q)``.  The unit variances ``v`` are
evaluated through the matrix-free covariance action, so the dense native
covariance is never formed.

Relative targets refer to the cancellation-stabilized scale
``max(abs(sum q), cancellation_floor * sum abs(q))``, so a mixed-sign total
that nearly cancels does not demand a near-zero SD.

Because ``SD`` is linear in a shared amplitude, calibration is simple:

* one amplitude per source: with ``statistic="geometric-mean"`` (default),
  ``log a`` is the mean of ``log(target_sd / unit_sd)``, the least-squares
  fit in log SD; with ``statistic="median"``, ``a = 1 / median(unit_sd /
  target_sd)``, which makes the median achieved-to-target ratio one and, for a
  shared relative target, equals the ``target / median(unit relative SD)`` rule
  of :func:`~openghg_inversions.basis.prior_uncertainty.calibrate_basis_prior_stdev`
  and the verification-games prototype.  Both are exact for one target;
* one amplitude per source and class: a bounded least-squares fit in log SD,
  which requires the targeted functionals to identify every class.  A class
  that the targets push towards zero stops at a floor and is reported.

Every target is classified as ``met`` or ``unmet`` against ``target_rtol``,
or ``at_bound`` when it depends on a class at the floor.

Every combination of the flux's remaining dimensions (for example ``time``) is
calibrated independently, so per-month calibration is the default.  A frozen
policy, such as calibrating one month and scaling it by a factor for every
month, is expressed by passing those amplitudes to
:func:`evaluate_native_b_amplitudes` with ``allow_broadcast=True``.

Native amplitudes calibrated here define ``B`` before any coherent reduction.
A retained-prior recalibration applied afterwards to the same sources would
calibrate their uncertainty twice; use one or the other for a source.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr
import scipy.optimize

from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.source_covariance import IndependentSourceCovariance

__all__ = [
    "NativeBAmplitudeCalibration",
    "calibrate_native_b_amplitudes",
    "calibrated_source_covariance",
    "calibrated_source_covariances",
    "evaluate_native_b_amplitudes",
]

AmplitudeStatistic = Literal["geometric-mean", "median"]

STATUS_MET = "met"
STATUS_UNMET = "unmet"
STATUS_AT_BOUND = "at_bound"
STATUS_DIAGNOSTIC = "diagnostic"
STATUS_ZERO = "zero"
TOTAL_OK = "ok"
TOTAL_CANCELLATION = "cancellation"
TOTAL_ZERO = "zero"

# Per-class amplitudes may not fall below this fraction of the common
# single-amplitude solution; reaching it is reported as ``at_bound``.
CLASS_AMPLITUDE_FLOOR = 1.0e-6


@dataclass(frozen=True)
class NativeBAmplitudeCalibration:
    """Calibrated native grid-cell standard deviations and their diagnostics.

    Attributes:
        amplitudes: Native grid-cell standard deviations with dimensions
            ``(source, *slice_dims)``, plus the class dimension when amplitudes
            were calibrated per class.  A class absent from a source's labels
            has a missing amplitude; a class at the floor has the floor value.
        diagnostics: Dataset over ``(source, functional, *slice_dims)`` with
            signed and absolute totals, the target scale and ``total_status``,
            unit-amplitude SDs, requested and achieved SDs, their ratio and log
            residual, and the per-functional ``status``.  ``to_dataframe()``
            gives a table.
        calibration_slices: Labels of every slice dimension that was
            calibrated, in order.  ``amplitudes`` carries the same labels.
        class_at_bound: Boolean ``(source, *slice_dims, class)`` mask of
            per-class amplitudes at the floor, or ``None`` for one amplitude
            per source.
        statistic: Statistic combining several targets of one source.
        correlation_length: Shared correlation length in degrees.
        latitude_correlation_length: Resolved latitude correlation length.
        longitude_correlation_length: Resolved longitude correlation length.
        per_class: Whether amplitudes were calibrated per class.
    """

    amplitudes: xr.DataArray
    diagnostics: xr.Dataset
    calibration_slices: Mapping[str, tuple[object, ...]]
    class_at_bound: xr.DataArray | None
    statistic: str
    correlation_length: float
    latitude_correlation_length: float
    longitude_correlation_length: float
    per_class: bool


def calibrate_native_b_amplitudes(
    flux: xr.DataArray,
    functional_weights: xr.DataArray,
    *,
    correlation_length: float,
    target_relative_sd: xr.DataArray | float | None = None,
    target_sd: xr.DataArray | float | None = None,
    latitude_correlation_length: float | None = None,
    longitude_correlation_length: float | None = None,
    class_labels: xr.DataArray | None = None,
    per_class: bool = False,
    statistic: AmplitudeStatistic = "geometric-mean",
    cancellation_floor: float = 0.1,
    target_rtol: float = 0.05,
    strict: bool = False,
    source_dim: str = "source",
    functional_dim: str = "functional",
    latitude_dim: str = "lat",
    longitude_dim: str = "lon",
    class_dim: str = "class",
) -> NativeBAmplitudeCalibration:
    """Calibrate native ``B`` amplitudes so functional SDs match targets.

    Each source, and each combination of the flux's remaining dimensions, is
    calibrated independently.  Functionals with a missing target are reported
    as diagnostics only.  Relative targets are converted to absolute SD
    targets with the cancellation-stabilized scale described in the module
    docstring.  Functionals with no weighted flux or zero unit variance cannot
    carry a target.

    The default geometric mean is the least-squares fit in log SD, so a source
    whose targets disagree is over-dispersed symmetrically in log space.  The
    median statistic gives parity with the median rules used elsewhere and is
    less sensitive to one outlying region.  Neither can meet inconsistent
    targets: such targets are reported ``unmet``.

    Args:
        flux: Native flux with the source and native grid dimensions.  Any
            other dimension (for example ``time``) is a slice dimension and is
            calibrated independently.
        functional_weights: Functional weights, such as region masks times
            cell area, with the functional and native grid dimensions.  An
            optional source dimension gives per-source weights.
        correlation_length: Correlation length ``L`` in degrees of the
            separable exponential kernel.
        target_relative_sd: Requested SD relative to the target scale, a
            scalar or labelled array broadcastable over source, functional and
            slice dimensions.  Missing values are diagnostic-only functionals.
        target_sd: Requested absolute SD in the units of ``flux *
            functional_weights`` summed over the grid.  Supply exactly one of
            ``target_relative_sd`` and ``target_sd``.
        latitude_correlation_length: Optional latitude-specific length.
        longitude_correlation_length: Optional longitude-specific length.
        class_labels: Optional native class labels, with an optional source
            dimension.  Cells in different classes are uncorrelated.
        per_class: Calibrate one amplitude per source and class.  Only the
            geometric-mean statistic applies.
        statistic: ``"geometric-mean"`` or ``"median"`` for one amplitude per
            source.
        cancellation_floor: Fraction of the absolute weighted total below
            which a signed total is treated as cancelling.
        target_rtol: Relative tolerance on achieved over target SD for a
            ``met`` status.
        strict: Raise if any target is ``unmet`` or ``at_bound``.
        source_dim: Source dimension name.
        functional_dim: Functional dimension name.
        latitude_dim: Latitude dimension name (degrees).
        longitude_dim: Longitude dimension name (degrees).
        class_dim: Name of the class dimension of per-class amplitudes.

    Returns:
        Amplitudes, slice provenance and achieved-versus-target diagnostics.

    Raises:
        ValueError: If inputs are misaligned or non-finite, the targets are
            ambiguous or invalid, a source has no usable target, per-class
            amplitudes are not identified, or ``strict`` and a target is not
            met.
    """
    if (target_relative_sd is None) == (target_sd is None):
        raise ValueError("Supply exactly one of target_relative_sd and target_sd")
    if per_class and class_labels is None:
        raise ValueError("per_class=True requires class_labels")
    if statistic not in ("geometric-mean", "median"):
        raise ValueError("statistic must be 'geometric-mean' or 'median'")
    if per_class and statistic != "geometric-mean":
        raise ValueError("per_class=True supports only statistic='geometric-mean'")
    _check_tolerances(cancellation_floor, target_rtol)
    kernel, components = _unit_components(
        flux,
        functional_weights,
        correlation_length=correlation_length,
        latitude_correlation_length=latitude_correlation_length,
        longitude_correlation_length=longitude_correlation_length,
        class_labels=class_labels,
        source_dim=source_dim,
        functional_dim=functional_dim,
        native_dims=(latitude_dim, longitude_dim),
        class_dim=class_dim,
        cancellation_floor=cancellation_floor,
    )
    targets = _absolute_targets(components, target_relative_sd=target_relative_sd, target_sd=target_sd)
    totals = components["native_signed_total"]
    unit_variance = components["unit_class_variance"]
    slice_dims = [str(dim) for dim in totals.dims if dim not in (source_dim, functional_dim)]
    class_values = unit_variance[class_dim].values
    amplitude_dims = [source_dim, *slice_dims] + ([class_dim] if per_class else [])
    amplitudes = xr.DataArray(
        np.full([unit_variance.sizes[dim] for dim in amplitude_dims], np.nan),
        dims=amplitude_dims,
        coords={dim: unit_variance[dim] for dim in amplitude_dims},
        name="native_grid_cell_stdev",
    )
    at_bound = xr.zeros_like(components["class_present"]) if per_class else None

    for index in _slice_indices(totals, [source_dim, *slice_dims]):
        variance = unit_variance.isel(index).transpose(functional_dim, class_dim).values
        target = targets.isel(index).transpose(functional_dim).values
        present = components["class_present"].isel(index).transpose(class_dim).values
        usable = np.isfinite(target) & (variance.sum(axis=1) > 0.0)
        where = ", ".join(f"{dim}={totals[dim].values[i]}" for dim, i in index.items())
        if not usable.any():
            raise ValueError(f"No usable target functional for {where}")
        if per_class and at_bound is not None:
            values, bound = _per_class_amplitudes(variance[usable][:, present], target[usable], where=where)
            class_amplitudes = np.full(class_values.size, np.nan)
            class_amplitudes[present] = values
            class_bound = np.zeros(class_values.size, dtype=bool)
            class_bound[present] = bound
            amplitudes[index] = class_amplitudes
            at_bound[index] = class_bound
        else:
            ratio = np.sqrt(variance[usable].sum(axis=1)) / target[usable]
            if statistic == "median":
                amplitudes[index] = float(1.0 / np.median(ratio))
            else:
                amplitudes[index] = float(np.exp(-np.mean(np.log(ratio))))

    diagnostics = _diagnostics(
        components,
        amplitudes,
        targets=targets,
        class_at_bound=at_bound,
        target_rtol=target_rtol,
        class_dim=class_dim,
    )
    if strict:
        _raise_if_not_met(diagnostics)
    return NativeBAmplitudeCalibration(
        amplitudes=amplitudes,
        diagnostics=diagnostics,
        calibration_slices={dim: tuple(totals[dim].values) for dim in slice_dims},
        class_at_bound=None if at_bound is None else at_bound.astype(bool),
        statistic=statistic,
        correlation_length=kernel.correlation_length,
        latitude_correlation_length=kernel.latitude_correlation_length,
        longitude_correlation_length=kernel.longitude_correlation_length,
        per_class=per_class,
    )


def evaluate_native_b_amplitudes(
    flux: xr.DataArray,
    functional_weights: xr.DataArray,
    amplitudes: xr.DataArray,
    *,
    correlation_length: float,
    target_relative_sd: xr.DataArray | float | None = None,
    target_sd: xr.DataArray | float | None = None,
    allow_broadcast: bool = False,
    latitude_correlation_length: float | None = None,
    longitude_correlation_length: float | None = None,
    class_labels: xr.DataArray | None = None,
    cancellation_floor: float = 0.1,
    target_rtol: float = 0.05,
    source_dim: str = "source",
    functional_dim: str = "functional",
    latitude_dim: str = "lat",
    longitude_dim: str = "lon",
    class_dim: str = "class",
) -> xr.Dataset:
    """Report the functional SDs implied by supplied native amplitudes.

    This is the diagnostic half of :func:`calibrate_native_b_amplitudes` and
    expresses frozen policies.  For example, calibrating January and scaling
    it by 1.7 for every month is::

        january = calibrate_native_b_amplitudes(flux.sel(time=["2021-01"]), ...)
        frozen = january.amplitudes.isel(time=0) * 1.7
        evaluate_native_b_amplitudes(flux, weights, frozen, allow_broadcast=True, ...)

    Amplitudes calibrated for one slice must not silently stand in for
    another: applying amplitudes that lack a slice dimension of ``flux`` (or
    carry a different scalar label for it) to several or different slices
    requires ``allow_broadcast=True``.  Targets are optional here and only
    fill the requested, ratio and status columns.

    Args:
        flux: Native flux, as in :func:`calibrate_native_b_amplitudes`.
        functional_weights: Functional weights, as in
            :func:`calibrate_native_b_amplitudes`.
        amplitudes: Positive native grid-cell standard deviations over the
            source dimension, optionally with slice and class dimensions.
        correlation_length: Correlation length ``L`` in degrees.
        target_relative_sd: Optional relative SD references.
        target_sd: Optional absolute SD references.
        allow_broadcast: Permit amplitudes from one slice to be applied to
            other slices.
        latitude_correlation_length: Optional latitude-specific length.
        longitude_correlation_length: Optional longitude-specific length.
        class_labels: Optional native class labels.
        cancellation_floor: As in :func:`calibrate_native_b_amplitudes`.
        target_rtol: As in :func:`calibrate_native_b_amplitudes`.
        source_dim: Source dimension name.
        functional_dim: Functional dimension name.
        latitude_dim: Latitude dimension name.
        longitude_dim: Longitude dimension name.
        class_dim: Class dimension name of per-class amplitudes.

    Returns:
        Diagnostics dataset with the layout described in
        :class:`NativeBAmplitudeCalibration`.

    Raises:
        ValueError: If inputs are misaligned, amplitudes would be broadcast
            across slices without ``allow_broadcast``, or amplitudes are
            missing, non-positive, or non-finite for a source or class in use.
    """
    if target_relative_sd is not None and target_sd is not None:
        raise ValueError("Supply at most one of target_relative_sd and target_sd")
    _check_tolerances(cancellation_floor, target_rtol)
    _, components = _unit_components(
        flux,
        functional_weights,
        correlation_length=correlation_length,
        latitude_correlation_length=latitude_correlation_length,
        longitude_correlation_length=longitude_correlation_length,
        class_labels=class_labels,
        source_dim=source_dim,
        functional_dim=functional_dim,
        native_dims=(latitude_dim, longitude_dim),
        class_dim=class_dim,
        cancellation_floor=cancellation_floor,
    )
    if target_relative_sd is None and target_sd is None:
        targets = xr.full_like(components["native_signed_total"], np.nan)
    else:
        targets = _absolute_targets(components, target_relative_sd=target_relative_sd, target_sd=target_sd)
    in_use = components["class_present"]
    if source_dim not in amplitudes.dims:
        raise ValueError(f"amplitudes must have the {source_dim!r} dimension")
    if class_dim in amplitudes.dims and class_labels is None:
        raise ValueError("per-class amplitudes require class_labels")
    slice_dims = [dim for dim in in_use.dims if dim not in (source_dim, class_dim)]
    amplitudes = _checked_slice_provenance(amplitudes, in_use, slice_dims, allow_broadcast=allow_broadcast)
    # A per-source amplitude broadcast over classes gives the same SD, so one
    # per-class layout serves both cases.
    amplitudes = _labelled_on(amplitudes, in_use, name="amplitudes")
    values = np.asarray(amplitudes.where(in_use, 1.0).values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("amplitudes must be positive and finite for every source, slice and class in use")
    return _diagnostics(
        components,
        amplitudes,
        targets=targets,
        class_at_bound=None,
        target_rtol=target_rtol,
        class_dim=class_dim,
    )


def calibrated_source_covariance(
    amplitudes: xr.DataArray,
    *,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    correlation_length: float,
    latitude_correlation_length: float | None = None,
    longitude_correlation_length: float | None = None,
    class_labels: xr.DataArray | Mapping[str, xr.DataArray] | None = None,
    source_dim: str = "source",
) -> IndependentSourceCovariance:
    """Build the independent per-source native covariance for one slice.

    This is the hand-off from calibration to native preparation: each source
    gets a :class:`SeparableExponentialCovariance` with its calibrated
    ``sigma`` and the same kernel and class blocking used to calibrate it.
    Use :func:`calibrated_source_covariances` for per-month amplitudes.

    Args:
        amplitudes: One positive amplitude per source; select a slice first.
            Per-class amplitudes are not representable by
            :class:`SeparableExponentialCovariance` and are rejected.
        latitude: Native latitude coordinate in degrees.
        longitude: Native longitude coordinate in degrees.
        correlation_length: Correlation length ``L`` in degrees.
        latitude_correlation_length: Optional latitude-specific length.
        longitude_correlation_length: Optional longitude-specific length.
        class_labels: Optional class labels shared by every source, labels
            with a source dimension, or a source-to-labels mapping.
        source_dim: Source dimension of ``amplitudes``; also used as the
            native source dimension of the returned covariance.

    Returns:
        Block-diagonal native covariance over sources in amplitude order.

    Raises:
        ValueError: If ``amplitudes`` has any dimension besides the source.
    """
    if amplitudes.dims != (source_dim,):
        raise ValueError(
            f"amplitudes must have only the {source_dim!r} dimension; use "
            "calibrated_source_covariances for slices, and note per-class amplitudes are not supported"
        )
    blocks = {}
    for source in amplitudes[source_dim].values:
        labels: xr.DataArray | None
        if isinstance(class_labels, xr.DataArray) and source_dim in class_labels.dims:
            labels = class_labels.sel({source_dim: source}, drop=True)
        elif isinstance(class_labels, Mapping):
            labels = class_labels[str(source)]
        else:
            labels = class_labels
        blocks[str(source)] = SeparableExponentialCovariance(
            latitude=latitude,
            longitude=longitude,
            sigma=float(amplitudes.sel({source_dim: source})),
            correlation_length=correlation_length,
            latitude_correlation_length=latitude_correlation_length,
            longitude_correlation_length=longitude_correlation_length,
            class_labels=labels,
        )
    return IndependentSourceCovariance(blocks, source_dim=source_dim)


def calibrated_source_covariances(
    amplitudes: xr.DataArray,
    *,
    slice_dim: str,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    correlation_length: float,
    latitude_correlation_length: float | None = None,
    longitude_correlation_length: float | None = None,
    class_labels: xr.DataArray | Mapping[str, xr.DataArray] | None = None,
    source_dim: str = "source",
) -> dict[Hashable, IndependentSourceCovariance]:
    """Build one independent per-source native covariance per slice.

    Args:
        amplitudes: Amplitudes over exactly the source and ``slice_dim``
            dimensions, for example ``calibration.amplitudes`` from a
            per-month calibration.
        slice_dim: Slice dimension, for example ``"time"``.
        latitude: Native latitude coordinate in degrees.
        longitude: Native longitude coordinate in degrees.
        correlation_length: Correlation length ``L`` in degrees.
        latitude_correlation_length: Optional latitude-specific length.
        longitude_correlation_length: Optional longitude-specific length.
        class_labels: As in :func:`calibrated_source_covariance`.
        source_dim: Source dimension name.

    Returns:
        Mapping from each ``slice_dim`` label to its covariance, in order.

    Raises:
        ValueError: If ``amplitudes`` has other dimensions.
    """
    if set(amplitudes.dims) != {source_dim, slice_dim}:
        raise ValueError(f"amplitudes must have exactly the {source_dim!r} and {slice_dim!r} dimensions")
    return {
        label: calibrated_source_covariance(
            amplitudes.sel({slice_dim: label}, drop=True),
            latitude=latitude,
            longitude=longitude,
            correlation_length=correlation_length,
            latitude_correlation_length=latitude_correlation_length,
            longitude_correlation_length=longitude_correlation_length,
            class_labels=class_labels,
            source_dim=source_dim,
        )
        for label in amplitudes[slice_dim].values
    }


def _check_tolerances(cancellation_floor: float, target_rtol: float) -> None:
    """Validate the cancellation floor and target tolerance."""
    if not np.isfinite(cancellation_floor) or not 0.0 <= cancellation_floor <= 1.0:
        raise ValueError("cancellation_floor must lie between zero and one")
    if not np.isfinite(target_rtol) or target_rtol <= 0.0:
        raise ValueError("target_rtol must be positive and finite")


def _unit_components(
    flux: xr.DataArray,
    functional_weights: xr.DataArray,
    *,
    correlation_length: float,
    latitude_correlation_length: float | None,
    longitude_correlation_length: float | None,
    class_labels: xr.DataArray | None,
    source_dim: str,
    functional_dim: str,
    native_dims: tuple[str, str],
    class_dim: str,
    cancellation_floor: float,
) -> tuple[SeparableExponentialCovariance, xr.Dataset]:
    """Return totals and unit-amplitude class variances of every functional.

    Args:
        flux: Native flux over source, grid and slice dimensions.
        functional_weights: Functional weights over functional and grid
            dimensions, optionally per source.
        correlation_length: Shared correlation length.
        latitude_correlation_length: Optional latitude override.
        longitude_correlation_length: Optional longitude override.
        class_labels: Optional class labels, optionally per source.
        source_dim: Source dimension name.
        functional_dim: Functional dimension name.
        native_dims: Latitude and longitude dimension names.
        class_dim: Class dimension name.
        cancellation_floor: Fraction of the absolute total below which the
            signed total is treated as cancelling.

    Returns:
        The unblocked unit kernel and a dataset with ``native_signed_total``,
        ``native_absolute_total``, ``target_scale`` and ``total_status`` over
        ``(source, functional, *slice)``, ``unit_class_variance`` with an
        added class dimension, and the Boolean ``class_present`` over
        ``(source, *slice, class)``.

    Notes:
        Work is done one slice at a time; peak memory is a few arrays of
        ``n_source * n_functional * n_lat * n_lon`` values.
    """
    for name, array, required in (
        ("flux", flux, (source_dim, *native_dims)),
        ("functional_weights", functional_weights, (functional_dim, *native_dims)),
    ):
        if not isinstance(array, xr.DataArray):
            raise ValueError(f"{name} must be an xarray.DataArray")
        missing = [dim for dim in required if dim not in array.dims]
        if missing:
            raise ValueError(f"{name} is missing dimensions {missing}")
    if functional_dim in flux.dims:
        raise ValueError(f"flux must not have the functional dimension {functional_dim!r}")
    extra = set(functional_weights.dims) - set(flux.dims) - {functional_dim}
    if extra:
        raise ValueError(f"functional_weights has unexpected dimensions {sorted(map(str, extra))}")
    flux = flux.astype(float)
    functional_weights = functional_weights.astype(float)
    try:
        flux, functional_weights = xr.align(flux, functional_weights, join="exact")
    except ValueError as error:
        raise ValueError("flux and functional_weights coordinates must align exactly") from error
    for name, array in (("flux", flux), ("functional_weights", functional_weights)):
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"{name} must be finite")

    kernel = SeparableExponentialCovariance(
        latitude=flux[native_dims[0]],
        longitude=flux[native_dims[1]],
        sigma=1.0,
        correlation_length=correlation_length,
        latitude_correlation_length=latitude_correlation_length,
        longitude_correlation_length=longitude_correlation_length,
    )
    slice_dims = [dim for dim in flux.dims if dim not in (source_dim, *native_dims)]
    if class_labels is None:
        labels = xr.zeros_like(flux.isel({dim: 0 for dim in slice_dims}, drop=True), dtype=int)
        class_values = np.array([0])
    else:
        allowed = ({*native_dims}, {source_dim, *native_dims})
        if set(class_labels.dims) not in allowed:
            raise ValueError(f"class_labels must have dimensions {native_dims} and optionally {source_dim!r}")
        try:
            labels = xr.align(class_labels, flux, join="exact")[0]
        except ValueError as error:
            raise ValueError("class_labels coordinates must align with flux exactly") from error
        if bool(labels.isnull().any()):
            raise ValueError("class_labels must assign every native cell to a class")
        class_values = np.unique(np.asarray(labels.values).reshape(-1))

    totals = []
    absolute = []
    variances = []
    for index in _slice_indices(flux, slice_dims):
        weights = flux.isel(index) * functional_weights.isel(
            {dim: i for dim, i in index.items() if dim in functional_weights.dims}
        )
        totals.append(weights.sum(native_dims))
        absolute.append(abs(weights).sum(native_dims))
        per_class = []
        for value in class_values:
            masked = weights.where(labels == value, 0.0)
            per_class.append((masked * kernel.apply(masked)).sum(native_dims))
        variances.append(xr.concat(per_class, dim=class_dim))
    class_coord = {class_dim: class_values}
    order = (source_dim, functional_dim, *slice_dims)
    total = _restack(totals, flux, slice_dims).transpose(*order)
    absolute_total = _restack(absolute, flux, slice_dims).transpose(*order)
    variance = _restack(variances, flux, slice_dims).assign_coords(class_coord).transpose(*order, class_dim)
    if bool((variance < -1.0e-12 * np.abs(variance).max()).any()):
        raise ValueError("native B produced a materially negative functional variance")
    present = xr.concat([(labels == value).any(native_dims) for value in class_values], dim=class_dim)
    present = present.assign_coords(class_coord).broadcast_like(total.isel({functional_dim: 0}, drop=True))
    floor = cancellation_floor * absolute_total
    total_status = xr.where(
        absolute_total <= 0.0,
        TOTAL_ZERO,
        xr.where(abs(total) <= floor, TOTAL_CANCELLATION, TOTAL_OK),
    )
    components = xr.Dataset(
        {
            "native_signed_total": total,
            "native_absolute_total": absolute_total,
            "target_scale": xr.where(abs(total) > floor, abs(total), floor),
            "total_status": total_status.astype(str),
            "unit_class_variance": variance.clip(min=0.0),
            "class_present": present.transpose(source_dim, *slice_dims, class_dim),
        }
    )
    return kernel, components


def _slice_indices(array: xr.DataArray, dims: Sequence[Hashable]) -> list[dict[Hashable, int]]:
    """Return positional index dictionaries over every combination of ``dims``."""
    return [
        dict(zip(dims, combination, strict=True))
        for combination in itertools.product(*(range(array.sizes[dim]) for dim in dims))
    ]


def _restack(pieces: list[xr.DataArray], flux: xr.DataArray, slice_dims: Sequence[Hashable]) -> xr.DataArray:
    """Reassemble per-slice results, in slice order, along the slice dimensions.

    The scalar slice coordinates left by positional selection are dropped
    first, so that slice dimensions of any size, including one, are rebuilt
    from ``flux``'s own coordinates.
    """
    if not slice_dims:
        return pieces[0]
    stacked = xr.concat([piece.reset_coords(drop=True) for piece in pieces], dim="_slice")
    index = np.arange(stacked.sizes["_slice"]).reshape(tuple(flux.sizes[dim] for dim in slice_dims))
    result = stacked.isel(_slice=xr.DataArray(index, dims=list(slice_dims)))
    return result.assign_coords({dim: flux[dim].values for dim in slice_dims})


def _absolute_targets(
    components: xr.Dataset,
    *,
    target_relative_sd: xr.DataArray | float | None,
    target_sd: xr.DataArray | float | None,
) -> xr.DataArray:
    """Broadcast targets to the functional layout as absolute SDs.

    Relative targets become ``relative * target_scale``; they are missing
    where the functional has no weighted flux.
    """
    totals = components["native_signed_total"]
    supplied = target_relative_sd if target_sd is None else target_sd
    if supplied is None:
        raise ValueError("Supply target_relative_sd or target_sd")
    requested = _labelled_on(supplied, totals, name="targets")
    finite = requested.values[np.isfinite(requested.values)]
    if np.any(finite <= 0.0):
        raise ValueError("targets must be positive where supplied")
    if target_sd is not None:
        return requested
    scale = components["target_scale"]
    return (requested * scale).where(scale > 0.0)


def _labelled_on(
    value: xr.DataArray | float, layout: xr.DataArray, *, name: str, missing: float = np.nan
) -> xr.DataArray:
    """Select labelled ``value`` onto ``layout`` and broadcast it.

    Every dimension of ``value`` must be a labelled dimension of ``layout``.
    Labels absent from ``value`` are filled with ``missing``; labels absent
    from ``layout`` raise, so a misspelt source or functional is not ignored.
    """
    array = value if isinstance(value, xr.DataArray) else xr.DataArray(value)
    for dim in array.dims:
        if dim not in layout.dims or dim not in array.indexes:
            raise ValueError(f"{name} dimension {dim!r} must be a labelled dimension of {list(layout.dims)}")
        unknown = set(array[dim].values.tolist()) - set(layout[dim].values.tolist())
        if unknown:
            raise ValueError(f"{name} has unknown {dim!r} labels {sorted(map(str, unknown))}")
    selected = array.astype(float).reindex(
        {dim: layout[dim].values for dim in array.dims}, fill_value=missing
    )
    return selected.reset_coords(drop=True).broadcast_like(layout)


def _checked_slice_provenance(
    amplitudes: xr.DataArray,
    layout: xr.DataArray,
    slice_dims: Sequence[Hashable],
    *,
    allow_broadcast: bool,
) -> xr.DataArray:
    """Refuse to apply one slice's amplitudes to other slices unless allowed.

    Args:
        amplitudes: Supplied amplitudes, possibly with scalar slice labels
            recording the slice they were calibrated for.
        layout: Array carrying the flux slice dimensions.
        slice_dims: Slice dimensions of the flux.
        allow_broadcast: Permit broadcasting across slices.

    Returns:
        ``amplitudes`` without scalar slice coordinates.
    """
    for dim in slice_dims:
        labels = layout.indexes[dim]
        if dim in amplitudes.dims:
            missing = labels.difference(amplitudes.indexes[dim])
            if len(missing):
                raise ValueError(f"amplitudes have no {dim!r} slice for {list(map(str, missing))}")
            continue
        origin = amplitudes.coords[dim].values[()] if dim in amplitudes.coords else None
        same_slice = len(labels) == 1 and (origin is None or bool(labels[0] == origin))
        if not same_slice and not allow_broadcast:
            source = "an unrecorded slice" if origin is None else f"{dim}={origin}"
            raise ValueError(
                f"amplitudes from {source} would be applied to {dim} {list(map(str, labels))}; "
                "pass allow_broadcast=True to evaluate a frozen-slice policy"
            )
    scalar = [dim for dim in slice_dims if dim in amplitudes.coords and dim not in amplitudes.dims]
    return amplitudes.drop_vars(scalar)


def _per_class_amplitudes(
    variance: np.ndarray, target: np.ndarray, *, where: str
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-class amplitudes by bounded least squares in log SD.

    Args:
        variance: Unit variances, targets by classes present for the source.
        target: Absolute SD targets.
        where: Slice and source description for error messages.

    Returns:
        Amplitudes of the present classes and a mask of those within a factor
        of ten of the floor,
        ``CLASS_AMPLITUDE_FLOOR`` times the common single-amplitude solution.
    """
    scaled = variance / target[:, None] ** 2
    if np.linalg.matrix_rank(scaled) < scaled.shape[1]:
        raise ValueError(
            f"Per-class amplitudes are not identified for {where}: the targeted "
            "functionals do not separate every class"
        )

    def residual(log_amplitude: np.ndarray) -> np.ndarray:
        return 0.5 * np.log(scaled @ np.exp(2.0 * log_amplitude))

    common = -0.5 * np.log(np.mean(scaled.sum(axis=1)))
    lower = common + np.log(CLASS_AMPLITUDE_FLOOR)
    start = np.full(scaled.shape[1], common)
    # scipy-stubs resolves least_squares to its private module of that name.
    fit = scipy.optimize.least_squares(  # type: ignore[operator]
        residual,
        start,
        bounds=(np.full_like(start, lower), np.full_like(start, np.inf)),
        xtol=1.0e-14,
        ftol=1.0e-14,
        gtol=1.0e-14,
    )
    if fit.status <= 0 or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"Per-class amplitude fit failed for {where}: {fit.message}")
    # A class within a factor of ten of the floor has collapsed.
    return np.exp(fit.x), fit.x <= lower + np.log(10.0)


def _diagnostics(
    components: xr.Dataset,
    amplitudes: xr.DataArray,
    *,
    targets: xr.DataArray,
    class_at_bound: xr.DataArray | None,
    target_rtol: float,
    class_dim: str,
) -> xr.Dataset:
    """Return achieved-versus-target SDs and statuses for every functional."""
    totals = components["native_signed_total"]
    unit_variance = components["unit_class_variance"]
    scale = components["target_scale"]
    positive_scale = scale.where(scale > 0.0)
    unit_sd = unit_variance.sum(class_dim) ** 0.5
    if class_dim in amplitudes.dims:
        achieved = (amplitudes.fillna(0.0) ** 2 * unit_variance).sum(class_dim) ** 0.5
    else:
        achieved = amplitudes * unit_sd
    achieved = achieved.transpose(*totals.dims)
    used = targets.notnull() & (unit_sd > 0.0)
    ratio = (achieved / targets).where(used)
    log_ratio = xr.apply_ufunc(np.log, ratio)
    within = abs(log_ratio) <= np.log1p(target_rtol)
    if class_at_bound is None:
        touches_bound = xr.zeros_like(used)
    else:
        touches_bound = ((unit_variance > 0.0) & class_at_bound).any(class_dim)
    status = xr.where(
        unit_sd <= 0.0,
        STATUS_ZERO,
        xr.where(
            ~used,
            STATUS_DIAGNOSTIC,
            xr.where(touches_bound, STATUS_AT_BOUND, xr.where(within, STATUS_MET, STATUS_UNMET)),
        ),
    )
    return xr.Dataset(
        {
            "native_signed_total": totals,
            "native_absolute_total": components["native_absolute_total"],
            "target_scale": scale,
            "total_status": components["total_status"],
            "unit_sd": unit_sd,
            "unit_relative_sd": unit_sd / positive_scale,
            "target_relative_sd": targets / positive_scale,
            "target_sd": targets,
            "achieved_sd": achieved,
            "achieved_relative_sd": achieved / positive_scale,
            "achieved_cv": achieved / abs(totals).where(totals != 0.0),
            "achieved_to_target_ratio": ratio,
            "log_sd_residual": log_ratio,
            "used_for_calibration": used,
            "status": status.astype(str),
        }
    )


def _raise_if_not_met(diagnostics: xr.Dataset) -> None:
    """Raise when any target is unmet or depends on a class at the floor."""
    failed = diagnostics["status"].isin([STATUS_UNMET, STATUS_AT_BOUND])
    if bool(failed.any()):
        table = diagnostics[["status", "achieved_to_target_ratio"]].where(failed).to_dataframe().dropna()
        raise ValueError(f"{len(table)} calibration targets were not met:\n{table.head(10)}")
