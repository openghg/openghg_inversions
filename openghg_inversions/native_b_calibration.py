"""Calibrate native prior covariance amplitudes against functional SD targets.

The native state is a grid of dimensionless flux scaling perturbations ``x``
for each source ``s``, with prior covariance

``B_s = sum_c a_{s,c}**2 M_c K M_c``,

where ``K`` is the unit-amplitude separable exponential kernel of
:class:`~openghg_inversions.native_covariance.SeparableExponentialCovariance`
with correlation length ``L``, ``M_c`` is the indicator of native class ``c``
and ``a_{s,c}`` is the native grid-cell standard deviation.  Without class
labels there is one class covering the grid.  Sources are independent, as in
:class:`~openghg_inversions.source_covariance.IndependentSourceCovariance`.

A declared functional ``r`` (for example a country or regional mask multiplied
by cell area) and a source flux ``f_s`` define the native total-flux weights
``q_{s,r} = f_s * w_r``.  The implied prior standard deviation of that total is

``SD_{s,r} = sqrt(q_{s,r}^T B_s q_{s,r}) = sqrt(sum_c a_{s,c}**2 v_{s,r,c})``,

with ``v_{s,r,c} = (M_c q)^T K (M_c q)``.  The unit variances ``v`` are
evaluated through the matrix-free covariance action, so the dense native
covariance is never formed.  Because ``SD`` is linear in a shared amplitude,
calibration is simple:

* one amplitude per source: ``log a`` is the mean of
  ``log(target_sd) - log(unit_sd)`` over the source's targeted functionals,
  which is the least-squares solution in log SD and is exact for one target;
* one amplitude per source and class: a small nonlinear least-squares fit in
  log SD over the source's targets, which requires the targeted functionals to
  identify every class.

Every combination of the flux's remaining dimensions (for example ``time``) is
calibrated independently, so per-month calibration is the default.  A frozen
policy, such as calibrating one month and scaling it by a factor for every
month, is expressed by passing those amplitudes to
:func:`evaluate_native_b_amplitudes`.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
import scipy.optimize

from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.source_covariance import IndependentSourceCovariance

__all__ = [
    "NativeBAmplitudeCalibration",
    "calibrate_native_b_amplitudes",
    "calibrated_source_covariance",
    "evaluate_native_b_amplitudes",
]

STATUS_TARGET = "target"
STATUS_DIAGNOSTIC = "diagnostic"
STATUS_NEAR_ZERO_TOTAL = "near-zero-total"
STATUS_ZERO_VARIANCE = "zero-variance"


@dataclass(frozen=True)
class NativeBAmplitudeCalibration:
    """Calibrated native grid-cell standard deviations and their diagnostics.

    Attributes:
        amplitudes: Native grid-cell standard deviations with dimensions
            ``(source, *slice_dims)``, plus the class dimension when amplitudes
            were calibrated per class.  A class absent from a source's labels
            has a missing amplitude.
        diagnostics: Dataset over ``(source, functional, *slice_dims)`` with
            signed totals, unit-amplitude SDs, requested and achieved SDs, and
            a per-functional ``status``.  ``to_dataframe()`` gives a table.
        correlation_length: Shared correlation length in degrees.
        latitude_correlation_length: Resolved latitude correlation length.
        longitude_correlation_length: Resolved longitude correlation length.
        per_class: Whether amplitudes were calibrated per class.
    """

    amplitudes: xr.DataArray
    diagnostics: xr.Dataset
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
    source_dim: str = "source",
    functional_dim: str = "functional",
    latitude_dim: str = "lat",
    longitude_dim: str = "lon",
    class_dim: str = "class",
    near_zero_total: float = 1.0e-30,
) -> NativeBAmplitudeCalibration:
    """Calibrate native ``B`` amplitudes so functional SDs match targets.

    Each source, and each combination of the flux's remaining dimensions, is
    calibrated independently.  Functionals with a missing target are reported
    as diagnostics only.  Relative targets are converted to absolute SD
    targets with ``abs(native signed total)``; functionals whose total is
    below ``near_zero_total`` cannot carry a relative target.

    With one amplitude per source the solution is the least-squares fit in
    log SD, ``log a = mean(log(target_sd / unit_sd))``, which matches every
    target exactly when a source has one target.  With ``per_class=True`` the
    targets of each source must identify all of its classes, otherwise a
    ``ValueError`` names the slice and source.

    Args:
        flux: Native flux with the source and native grid dimensions.  Any
            other dimension (for example ``time``) is a slice dimension and is
            calibrated independently.
        functional_weights: Functional weights, such as region masks times
            cell area, with the functional and native grid dimensions.  An
            optional source dimension gives per-source weights.
        correlation_length: Correlation length ``L`` in degrees of the
            separable exponential kernel.
        target_relative_sd: Requested SD relative to ``abs(total)``, a scalar
            or array broadcastable over source, functional and slice
            dimensions.  Missing values are diagnostic-only functionals.
        target_sd: Requested absolute SD in the units of ``flux *
            functional_weights`` summed over the grid.  Supply exactly one of
            ``target_relative_sd`` and ``target_sd``.
        latitude_correlation_length: Optional latitude-specific length.
        longitude_correlation_length: Optional longitude-specific length.
        class_labels: Optional native class labels, with an optional source
            dimension.  Cells in different classes are uncorrelated.
        per_class: Calibrate one amplitude per source and class.
        source_dim: Source dimension name.
        functional_dim: Functional dimension name.
        latitude_dim: Latitude dimension name (degrees).
        longitude_dim: Longitude dimension name (degrees).
        class_dim: Name of the class dimension of per-class amplitudes.
        near_zero_total: Absolute total below which a functional has no
            relative SD.

    Returns:
        Amplitudes and achieved-versus-target diagnostics.

    Raises:
        ValueError: If inputs are misaligned or non-finite, the targets are
            ambiguous or invalid, a source has no usable target, or per-class
            amplitudes are not identified.
    """
    if (target_relative_sd is None) == (target_sd is None):
        raise ValueError("Supply exactly one of target_relative_sd and target_sd")
    if per_class and class_labels is None:
        raise ValueError("per_class=True requires class_labels")
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
    )
    targets = _absolute_targets(
        components,
        target_relative_sd=target_relative_sd,
        target_sd=target_sd,
        near_zero_total=near_zero_total,
    )
    totals = components["native_signed_total"]
    unit_variance = components["unit_class_variance"]
    slice_dims = [dim for dim in totals.dims if dim not in (source_dim, functional_dim)]
    class_values = unit_variance[class_dim].values
    amplitude_dims = [source_dim, *slice_dims] + ([class_dim] if per_class else [])
    amplitudes = xr.DataArray(
        np.full([unit_variance.sizes[dim] for dim in amplitude_dims], np.nan),
        dims=amplitude_dims,
        coords={dim: unit_variance[dim] for dim in amplitude_dims},
        name="native_grid_cell_stdev",
    )

    for index in _slice_indices(totals, [source_dim, *slice_dims]):
        variance = unit_variance.isel(index).transpose(functional_dim, class_dim).values
        target = targets.isel(index).transpose(functional_dim).values
        present = components["class_present"].isel(index).transpose(class_dim).values
        usable = np.isfinite(target) & (variance.sum(axis=1) > 0.0)
        where = ", ".join(f"{dim}={totals[dim].values[i]}" for dim, i in index.items())
        if not usable.any():
            raise ValueError(f"No usable target functional for {where}")
        if per_class:
            amplitudes[index] = _per_class_amplitudes(
                variance[usable][:, present],
                target[usable],
                n_classes=class_values.size,
                present=present,
                where=where,
            )
        else:
            unit_sd = np.sqrt(variance[usable].sum(axis=1))
            amplitudes[index] = float(np.exp(np.mean(np.log(target[usable] / unit_sd))))

    diagnostics = _diagnostics(
        components,
        amplitudes,
        targets=targets,
        near_zero_total=near_zero_total,
        class_dim=class_dim,
    )
    return NativeBAmplitudeCalibration(
        amplitudes=amplitudes,
        diagnostics=diagnostics,
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
    latitude_correlation_length: float | None = None,
    longitude_correlation_length: float | None = None,
    class_labels: xr.DataArray | None = None,
    source_dim: str = "source",
    functional_dim: str = "functional",
    latitude_dim: str = "lat",
    longitude_dim: str = "lon",
    class_dim: str = "class",
    near_zero_total: float = 1.0e-30,
) -> xr.Dataset:
    """Report the functional SDs implied by supplied native amplitudes.

    This is the diagnostic half of :func:`calibrate_native_b_amplitudes` and
    expresses frozen policies.  For example, calibrating January and scaling
    it by 1.7 for every month is::

        january = calibrate_native_b_amplitudes(flux.sel(time=["2021-01"]), ...)
        frozen = january.amplitudes.isel(time=0, drop=True) * 1.7
        evaluate_native_b_amplitudes(flux, weights, frozen, ...)

    Amplitudes broadcast over any slice dimension they lack.  Targets are
    optional here and only fill the requested and residual columns.

    Args:
        flux: Native flux, as in :func:`calibrate_native_b_amplitudes`.
        functional_weights: Functional weights, as in
            :func:`calibrate_native_b_amplitudes`.
        amplitudes: Positive native grid-cell standard deviations over the
            source dimension, optionally with slice and class dimensions.
        correlation_length: Correlation length ``L`` in degrees.
        target_relative_sd: Optional relative SD references.
        target_sd: Optional absolute SD references.
        latitude_correlation_length: Optional latitude-specific length.
        longitude_correlation_length: Optional longitude-specific length.
        class_labels: Optional native class labels.
        source_dim: Source dimension name.
        functional_dim: Functional dimension name.
        latitude_dim: Latitude dimension name.
        longitude_dim: Longitude dimension name.
        class_dim: Class dimension name of per-class amplitudes.
        near_zero_total: Absolute total below which a functional has no
            relative SD.

    Returns:
        Diagnostics dataset with the layout described in
        :class:`NativeBAmplitudeCalibration`.

    Raises:
        ValueError: If inputs are misaligned or amplitudes are missing,
            non-positive, or non-finite for a source or class in use.
    """
    if target_relative_sd is not None and target_sd is not None:
        raise ValueError("Supply at most one of target_relative_sd and target_sd")
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
    )
    if target_relative_sd is None and target_sd is None:
        targets = xr.full_like(components["native_signed_total"], np.nan)
    else:
        targets = _absolute_targets(
            components,
            target_relative_sd=target_relative_sd,
            target_sd=target_sd,
            near_zero_total=near_zero_total,
        )
    in_use = components["class_present"]
    if source_dim not in amplitudes.dims:
        raise ValueError(f"amplitudes must have the {source_dim!r} dimension")
    if class_dim in amplitudes.dims and class_labels is None:
        raise ValueError("per-class amplitudes require class_labels")
    # A per-source amplitude broadcast over classes gives the same SD, so one
    # per-class layout serves both cases.
    amplitudes = _labelled_on(amplitudes, in_use, name="amplitudes")
    checked = amplitudes.where(in_use, 1.0)
    values = np.asarray(checked.values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("amplitudes must be positive and finite for every source and class in use")
    return _diagnostics(
        components,
        amplitudes,
        targets=targets,
        near_zero_total=near_zero_total,
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
            f"amplitudes must have only the {source_dim!r} dimension; "
            "select one slice, and note per-class amplitudes are not supported"
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

    Returns:
        The unblocked unit kernel and a dataset with ``native_signed_total``
        over ``(source, functional, *slice)``, ``unit_class_variance`` with an
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
    variances = []
    for index in _slice_indices(flux, slice_dims):
        weights = flux.isel(index) * functional_weights.isel(
            {dim: i for dim, i in index.items() if dim in functional_weights.dims}
        )
        totals.append(weights.sum(native_dims))
        per_class = []
        for value in class_values:
            masked = weights.where(labels == value, 0.0)
            per_class.append((masked * kernel.apply(masked)).sum(native_dims))
        variances.append(xr.concat(per_class, dim=class_dim))
    class_coord = {class_dim: class_values}
    total = _restack(totals, flux, slice_dims).transpose(source_dim, functional_dim, *slice_dims)
    variance = (
        _restack(variances, flux, slice_dims)
        .assign_coords(class_coord)
        .transpose(source_dim, functional_dim, *slice_dims, class_dim)
    )
    if bool((variance < -1.0e-12 * np.abs(variance).max()).any()):
        raise ValueError("native B produced a materially negative functional variance")
    present = xr.concat([(labels == value).any(native_dims) for value in class_values], dim=class_dim)
    present = present.assign_coords(class_coord).broadcast_like(total.isel({functional_dim: 0}, drop=True))
    components = xr.Dataset(
        {
            "native_signed_total": total,
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
    """Reassemble per-slice results, in slice order, along the slice dimensions."""
    if not slice_dims:
        return pieces[0]
    stacked = xr.concat(pieces, dim="_slice")
    shape = {dim: flux.sizes[dim] for dim in slice_dims}
    index = np.arange(stacked.sizes["_slice"]).reshape(tuple(shape.values()))
    result = stacked.isel(_slice=xr.DataArray(index, dims=list(slice_dims)))
    return result.drop_vars("_slice", errors="ignore").assign_coords({dim: flux[dim] for dim in slice_dims})


def _absolute_targets(
    components: xr.Dataset,
    *,
    target_relative_sd: xr.DataArray | float | None,
    target_sd: xr.DataArray | float | None,
    near_zero_total: float,
) -> xr.DataArray:
    """Broadcast targets to the functional layout as absolute SDs.

    Relative targets become ``relative * abs(total)``; they are dropped (set
    missing) where the total is below ``near_zero_total``.
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
    magnitude = abs(totals)
    return (requested * magnitude).where(magnitude > near_zero_total)


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
    return selected.broadcast_like(layout)


def _per_class_amplitudes(
    variance: np.ndarray,
    target: np.ndarray,
    *,
    n_classes: int,
    present: np.ndarray,
    where: str,
) -> np.ndarray:
    """Fit per-class amplitudes by least squares in log SD.

    Args:
        variance: Unit variances, targets by present classes.
        target: Absolute SD targets.
        n_classes: Total number of classes in the output.
        present: Boolean mask of classes present for this source.
        where: Slice and source description for error messages.

    Returns:
        Amplitudes over all classes, missing where a class is absent.
    """
    scaled = variance / target[:, None] ** 2
    if np.linalg.matrix_rank(scaled) < scaled.shape[1]:
        raise ValueError(
            f"Per-class amplitudes are not identified for {where}: the targeted "
            "functionals do not separate every class"
        )

    def residual(log_amplitude: np.ndarray) -> np.ndarray:
        return 0.5 * np.log(scaled @ np.exp(2.0 * log_amplitude))

    start = np.full(scaled.shape[1], -0.5 * np.log(np.mean(scaled.sum(axis=1))))
    # scipy-stubs resolves least_squares to its private module of that name.
    fit = scipy.optimize.least_squares(  # type: ignore[operator]
        residual, start, xtol=1.0e-14, ftol=1.0e-14, gtol=1.0e-14
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"Per-class amplitude fit failed for {where}: {fit.message}")
    result = np.full(n_classes, np.nan)
    result[present] = np.exp(fit.x)
    return result


def _diagnostics(
    components: xr.Dataset,
    amplitudes: xr.DataArray,
    *,
    targets: xr.DataArray,
    near_zero_total: float,
    class_dim: str,
) -> xr.Dataset:
    """Return achieved-versus-target SDs for every source and functional."""
    totals = components["native_signed_total"]
    unit_variance = components["unit_class_variance"]
    magnitude = abs(totals)
    nonzero = magnitude > near_zero_total
    unit_sd = unit_variance.sum(class_dim) ** 0.5
    if class_dim in amplitudes.dims:
        achieved = (amplitudes.fillna(0.0) ** 2 * unit_variance).sum(class_dim) ** 0.5
    else:
        achieved = amplitudes * unit_sd
    status = xr.where(
        ~nonzero,
        STATUS_NEAR_ZERO_TOTAL,
        xr.where(
            unit_sd <= 0.0,
            STATUS_ZERO_VARIANCE,
            xr.where(targets.notnull(), STATUS_TARGET, STATUS_DIAGNOSTIC),
        ),
    )
    used = targets.notnull() & (unit_sd > 0.0)
    achieved = achieved.transpose(*totals.dims)
    return xr.Dataset(
        {
            "native_signed_total": totals,
            "unit_sd": unit_sd,
            "unit_relative_sd": (unit_sd / magnitude).where(nonzero),
            "target_relative_sd": (targets / magnitude).where(nonzero),
            "target_sd": targets,
            "achieved_sd": achieved,
            "achieved_relative_sd": (achieved / magnitude).where(nonzero),
            "log_sd_residual": xr.apply_ufunc(np.log, achieved / targets).where(used),
            "used_for_calibration": used,
            "status": status.astype(str),
        }
    )
