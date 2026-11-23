"""Accurate centered boundary operands for stable CO2/O2 surface inversions."""

from collections.abc import Mapping
from typing import Literal, cast, get_args

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from pint.errors import PintError

from openghg_inversions.array_ops import to_dense
from openghg_inversions.boundary_sensitivity import BoundaryRepresentation

from ._units import mole_fraction_unit_scale

#: How boundary concentrations are timed for each native release and curtain cell.
#: ``release_time`` selects the field valid at the release; ``mean_age`` selects it at
#: the release time minus the cell's stored mean particle age.
BoundaryTimeMethod = Literal["release_time", "mean_age"]
BOUNDARY_TIME_METHODS: tuple[str, ...] = get_args(BoundaryTimeMethod)

_NANOSECONDS_PER_HOUR = 3.6e12


# ---------------------------------------------------------------------------
# VENDORED MEAN-AGE LAG (temporary).
#
# Adapted from openghg/openghg#1806 (issue openghg/openghg#1805), branch
# ``codex/1805-mean-age-boundary-lag`` at f91cc96e1c37, file
# ``openghg/analyse/_modelled_baseline.py``: ``_age_hours``,
# ``_boundary_intervals`` and ``_lagged_rows``, and the gather in
# ``_mean_age_boundary``. The pinned OpenGHG revision predates that PR.
# Adaptations: no climatology mapping (surface acquisition disables it),
# exact spatial alignment instead of spatial forward filling, ages lagged in
# float64, and OpenGHG's interval-start ``time_period`` convention only.
# Remove this block and call OpenGHG's helper once #1806 is merged and the
# pinned OpenGHG revision includes it.
# ---------------------------------------------------------------------------


def _age_hours(fp: xr.Dataset, direction: str) -> xr.DataArray:
    """Read one curtain's stored mean particle ages as numeric hours.

    Positive-weight cells need ages with duration units. A curtain whose
    weights are all zero may lack usable ages; zeros are returned for it
    (checking this computes one scalar reduction of the weights).

    Raises:
        ValueError: If ``mean_age_particles_<direction>`` is missing or lacks
            duration units while the curtain has positive weight.
    """
    name = f"mean_age_particles_{direction}"
    try:
        if name not in fp:
            raise ValueError(f"Missing {name}; mean-age boundary timing requires particle ages.")
        age = fp[name]
        if not age.attrs.get("units"):
            raise ValueError(f"{name} must declare duration units, such as hours.")
        return cast(xr.DataArray, age.pint.quantify().pint.to("hour").pint.dequantify())
    except (PintError, ValueError, TypeError) as exc:
        weights = fp[f"particle_locations_{direction}"]
        if not bool((weights != 0).any().compute()):
            return xr.zeros_like(weights, dtype=float)
        raise ValueError(f"{name} must be present with duration units convertible to hours.") from exc


def _boundary_intervals(bc: xr.Dataset, period: str) -> tuple[np.ndarray, np.ndarray]:
    """Return start-labelled native BC intervals as sorted int64 nanoseconds.

    Each stored ``time`` label starts an interval of width ``period`` (OpenGHG
    period syntax). Intervals must be positive and nonoverlapping; internal
    gaps are allowed and rejected later only where a positive-weight cell
    needs them. ``bc.time`` must already be unique and increasing.
    """
    from openghg.util import relative_time_offset

    source_times = pd.DatetimeIndex(bc.indexes["time"])
    try:
        offset = relative_time_offset(period=period)
        starts = source_times.to_numpy(dtype="datetime64[ns]")
        ends = (source_times + offset).to_numpy(dtype="datetime64[ns]")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid boundary interval period {period!r}.") from exc
    if np.isnat(ends).any() or (ends <= starts).any() or (ends[:-1] > starts[1:]).any():
        raise ValueError("Boundary intervals must have finite, positive, nonoverlapping support.")
    return starts.astype("int64"), ends.astype("int64")


def _lagged_rows(
    age: np.ndarray, weight: np.ndarray, release: np.ndarray, *, starts: np.ndarray, ends: np.ndarray,
) -> np.ndarray:
    """Locate each active cell's lagged time ``release - age`` in sorted BC intervals.

    Args:
        age: Mean ages in numeric hours for one curtain block.
        weight: Particle weights, broadcastable with ``age``.
        release: Native datetime64 release times, broadcastable to the block.
        starts: Sorted interval starts as int64 epoch nanoseconds.
        ends: Matching exclusive interval ends.

    Returns:
        Interval positions with the broadcast block shape. Zero-weight cells
        get position zero as a placeholder; they need no age or BC support.
        Ages keep their fractional value to nanosecond precision.

    Raises:
        ValueError: If weights are negative or nonfinite, positive-weight ages
            are negative or nonfinite, or a positive-weight lagged time lies
            outside the intervals (before the first, after the last, or in a gap).
    """
    if (~np.isfinite(weight) | (weight < 0)).any():
        raise ValueError("Particle weights must be finite and nonnegative.")
    active = weight > 0
    if (active & (~np.isfinite(age) | (age < 0))).any():
        raise ValueError("Positive-weight particle ages must be finite and nonnegative.")
    hours = np.where(active, age, 0).astype(np.float64)
    if (hours >= np.iinfo(np.int64).max / _NANOSECONDS_PER_HOUR).any():
        raise ValueError("Particle ages exceed the supported datetime range.")
    release_ns = release.astype("datetime64[ns]").astype("int64")
    targets = release_ns - np.rint(hours * _NANOSECONDS_PER_HOUR).astype("int64")
    if (active & (targets > release_ns)).any():
        raise ValueError("Particle ages exceed the supported datetime range.")
    targets = np.where(active, targets, starts[0])
    positions = np.searchsorted(starts, targets, side="right") - 1
    interval_indices = np.maximum(positions, 0)
    unsupported = active & ((positions < 0) | (targets >= ends[interval_indices]))
    if unsupported.any():
        missing = np.asarray(targets)[unsupported].flat[0].astype("datetime64[ns]")
        raise ValueError(
            f"Boundary conditions do not cover positive-weight lagged time {missing}; "
            "acquire earlier boundary history covering the largest particle age."
        )
    return cast(np.ndarray, interval_indices)


def _mask_inactive(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Zero inactive cells so placeholders cannot carry NaNs into products."""
    return np.where(weights > 0, values, 0)


def _mean_age_concentration(
    concentration: xr.DataArray, weight: xr.DataArray, age: xr.DataArray,
    *, starts: np.ndarray, ends: np.ndarray,
) -> xr.DataArray:
    """Select ``C_q(t_i - a_iq)`` for every native release ``i`` and curtain cell ``q``.

    ``concentration`` must already share ``weight``'s spatial indexes exactly.
    The result has ``weight``'s dimensions and coordinates and the source
    dtype; zero-weight cells are zero. One scalar reduction of ages and
    weights bounds the BC rows read; the gather itself stays lazy for Dask inputs.
    """
    age = age.transpose(*weight.dims)
    interval_indices = xr.apply_ufunc(
        _lagged_rows, age, weight, weight.time,
        kwargs={"starts": starts, "ends": ends}, dask="parallelized", output_dtypes=[np.int64],
    )
    spatial_dims = [d for d in weight.dims if d != "time"]
    spatial_shape = tuple(weight.sizes[d] for d in spatial_dims)
    cell_count = int(np.prod(spatial_shape))
    cell_indices = xr.DataArray(np.arange(cell_count).reshape(spatial_shape), dims=spatial_dims)

    # Bound the BC rows read before pairing source chunks with lazy index chunks.
    limits = xr.Dataset({
        "first": interval_indices.where(weight > 0, len(starts)).min(),
        "last": interval_indices.where(weight > 0, 0).max(),
    }).compute()
    first, last = int(limits["first"]), int(limits["last"])
    if first > last:
        return xr.zeros_like(weight, dtype=concentration.dtype).assign_attrs(concentration.attrs)
    # Reset inactive placeholders to this window so they cannot pull in older history.
    interval_indices = interval_indices.where(weight > 0, first) - first
    # Flatten (interval, cell) as interval * cell_count + cell, preserving cell identity.
    flat = (interval_indices * cell_count + cell_indices).transpose("time", *spatial_dims)
    source = concentration.isel(time=slice(first, last + 1)).transpose("time", *spatial_dims)
    if source.chunks is not None or flat.chunks is not None:
        values = da.take(da.asarray(source.data).ravel(), da.asarray(flat.data).ravel())
    else:
        values = np.take(source.values.ravel(), flat.values.ravel())
    selected = xr.DataArray(
        values.reshape(flat.shape), dims=flat.dims, coords=weight.coords, attrs=concentration.attrs,
    ).transpose(*weight.dims)
    return xr.apply_ufunc(
        _mask_inactive, selected, weight, dask="parallelized", keep_attrs=True,
        output_dtypes=[concentration.dtype],
    )


# --------------------------- end of vendored block ---------------------------


def mean_age_history_start(fp: xr.Dataset) -> pd.Timestamp:
    """Return the earliest lagged time ``t_i - a_iq`` over positive-weight curtain cells.

    Boundary retrieval uses this to reach back by the largest particle age
    rather than one BC period. Computes one scalar reduction per curtain.

    Args:
        fp: Native releases with ``particle_locations_n/e/s/w`` and
            ``mean_age_particles_n/e/s/w`` (duration units).

    Returns:
        The earliest positive-weight lagged time; the first release time when
        every curtain has zero weight.

    Raises:
        ValueError: If ages are missing, unitless or invalid at positive weights.
    """
    release = fp.indexes["time"]
    earliest = release.min()
    for direction in "nesw":
        weight = fp[f"particle_locations_{direction}"]
        age = _age_hours(fp, direction).transpose(*weight.dims)
        # Releases without active cells lag by zero, i.e. need support at release.
        maximum = age.where(weight > 0).max([d for d in weight.dims if d != "time"]).fillna(0)
        lag = fp.time - (maximum.astype("float64") * _NANOSECONDS_PER_HOUR).astype("timedelta64[ns]")
        earliest = min(earliest, pd.Timestamp(lag.min().compute().values))
    return earliest


def baseline_reference_sensitivities(
    bc: xr.Dataset,
    fp: xr.Dataset,
    *,
    species: str,
    reference: float,
    reference_units: str,
    mass_tolerance: float,
    bc_time_method: BoundaryTimeMethod = "release_time",
) -> xr.Dataset:
    """Compute stable CO2/O2 boundary anomaly and mass operands.

    Each native curtain retains its dimensions. All four curtains are normalized
    together per observation, then ``anomaly = mass * (vmr - reference)``.
    Reconstruct absolute contributions as ``anomaly + bc_reference * mass``.
    Arithmetic uses the north curtain's scale. Output units retain its units,
    with any explicit numerical factor absorbed into the output values.

    BC timestamps mark interval starts, with validity given by the dataset's
    ``time_period`` attribute (OpenGHG period syntax, e.g. ``1 month``). Forward
    filling is allowed only inside those intervals, including for the final BC
    timestamp. Shared spatial indexes must match exactly; no spatial filling is
    performed. Inputs and their metadata are borrowed without mutation.

    With ``bc_time_method="mean_age"``, each native release ``i`` and curtain
    cell ``q`` uses ``C_q(t_i - a_iq)``, where ``a_iq`` is the stored fractional
    ``mean_age_particles_<d>`` (no rounding, no per-release age), looked up in
    the same interval-start BC intervals. Positive-weight lagged times must be
    covered; zero-weight cells need no age or BC support. Normalization,
    anomaly and mass checks are unchanged, so ordinary and centered operands
    both carry the lag. Bounding the BC rows read computes one scalar
    reduction of each curtain's ages and weights.

    Arithmetic uses float64 but cannot recover precision absent in the inputs.
    Dask data validation runs when the operands are materialized; coordinates,
    units, reference and tolerance are validated immediately. No chunk
    alignment or full-array materialization is requested here.

    Args:
        bc: Dataset containing ``vmr_n/e/s/w`` with concentration units.
        fp: Dataset containing dimensionless ``particle_locations_n/e/s/w``.
        species: Tracer label. The caller owns the supported-species policy.
        reference: Finite arithmetic reference, before unit conversion.
        reference_units: Concentration units of the reference.
        mass_tolerance: Explicit absolute tolerance for raw mass versus unity;
            must be finite and satisfy ``0 <= mass_tolerance < 1``.
        bc_time_method: ``"release_time"`` (default) or ``"mean_age"``. Mean
            age requires ``mean_age_particles_n/e/s/w`` with duration units.

    Returns:
        Dataset with ``bc_anomaly_n/e/s/w``, dimensionless ``bc_mass_n/e/s/w``
        and scalar ``bc_reference``. All values are float64.

    Raises:
        ValueError: For invalid data, units, coordinates,
            coverage, reference or mass tolerance.
    """
    import pandas as pd

    from openghg.util import relative_time_offset
    from openghg.util._units import cf_ureg

    if not np.isfinite(reference):
        raise ValueError("Boundary reference must be finite.")
    if not np.isfinite(mass_tolerance) or not 0 <= mass_tolerance < 1:
        raise ValueError("mass_tolerance must be finite and satisfy 0 <= tolerance < 1.")
    if bc_time_method not in BOUNDARY_TIME_METHODS:
        raise ValueError(f"bc_time_method must be one of {BOUNDARY_TIME_METHODS!r}.")

    for dataset in (bc, fp):
        if "time" not in dataset.indexes:
            raise ValueError("Boundary and footprint time must be indexed coordinates.")
        index = dataset.indexes["time"]
        if (
            not isinstance(index, pd.DatetimeIndex)
            or not len(index)
            or not index.is_unique
            or not index.is_monotonic_increasing
            or index.hasnans
        ):
            raise ValueError("Boundary and footprint time must be nonempty, unique and increasing.")
    if "time_period" not in bc.attrs:
        raise ValueError("BC time_period must declare forward-fill coverage.")
    period = relative_time_offset(period=bc.attrs["time_period"])
    bc_time = bc.indexes["time"]
    fp_time = fp.indexes["time"]
    if bc_time[0] + period <= bc_time[0]:
        raise ValueError("BC time_period must be positive.")
    if bc_time_method == "mean_age":
        starts, ends = _boundary_intervals(bc, bc.attrs["time_period"])
    else:
        positions = bc_time.get_indexer(fp_time, method="ffill")
        if (positions < 0).any() or any(t >= bc_time[int(i)] + period for t, i in zip(fp_time, positions)):
            raise ValueError("BC time_period coverage does not cover every footprint observation.")

    def conversion(source: str, target: str) -> float:
        try:
            source_scale = mole_fraction_unit_scale(source, context="boundary")
            target_scale = mole_fraction_unit_scale(target, context="boundary")
            if not all(np.isfinite(scale) and scale > 0 for scale in (source_scale, target_scale)):
                raise ValueError("Boundary unit scales must be finite and positive.")
            return source_scale / target_scale
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid boundary units: {source!r} -> {target!r}.") from exc

    units = bc["vmr_n"].attrs.get("units")
    if not units or not reference_units:
        raise ValueError("Boundary concentration and reference units must be explicit.")
    converted_reference = np.float64(reference) * conversion(reference_units, units)
    if not np.isfinite(converted_reference):
        raise ValueError("Converted boundary reference must be finite.")
    # Pint array quantification needs a unit, rather than a scaled expression.
    expression = cf_ureg.parse_expression(units)
    output_scale = float(getattr(expression, "magnitude", expression))
    output_units = units if output_scale == 1 else str(getattr(expression, "units", "1"))

    def checked_values(values: np.ndarray, *, nonnegative: bool = False) -> np.ndarray:
        if not np.isfinite(values).all() or (nonnegative and (values < 0).any()):
            raise ValueError("Boundary concentrations/weights must be finite; weights must be nonnegative.")
        return values

    def checked_total(values: np.ndarray) -> np.ndarray:
        if (
            not np.isfinite(values).all()
            or (values <= 0).any()
            or (np.abs(values - 1) > mass_tolerance).any()
        ):
            raise ValueError("Total boundary mass must be positive and within mass_tolerance of unity.")
        return values

    weights = {}
    concentrations = {}
    for direction in "nesw":
        weight = fp[f"particle_locations_{direction}"]
        concentration = bc[f"vmr_{direction}"]
        # Exact alignment cannot check dimensions lacking indexed coordinates.
        # Spatial coordinates come from the BC field in either timing method.
        for dim in weight.dims:
            if dim not in weight.indexes or dim not in concentration.indexes:
                raise ValueError(f"Boundary curtain coordinate {dim!r} must be indexed.")
            bc_unit, fp_unit = concentration[dim].attrs.get("units"), weight[dim].attrs.get("units")
            if dim != "time" and bc_unit and fp_unit and cf_ureg.Unit(bc_unit) != cf_ureg.Unit(fp_unit):
                raise ValueError(f"Boundary curtain coordinate units for {dim!r} must match exactly.")
        try:
            if set(concentration.dims) != set(weight.dims):
                raise ValueError("dimension mismatch")
            concentration = concentration.transpose(*weight.dims)
            xr.align(concentration, weight, join="exact", copy=False, exclude=["time"])
        except ValueError as exc:
            raise ValueError("Boundary curtain dimensions and indexed coordinates must match exactly.") from exc
        if bc_time_method == "mean_age":
            selected = _mean_age_concentration(
                concentration, weight, _age_hours(fp, direction), starts=starts, ends=ends,
            )
        else:
            selected = concentration.reindex(time=fp.time, method="ffill", copy=False)
        selected, weight = xr.align(selected, weight, join="exact", copy=False)
        concentration_units = concentration.attrs.get("units")
        if not concentration_units:
            raise ValueError("Boundary concentrations must have explicit units.")
        scale = conversion(concentration_units, units)
        weight_units = weight.attrs.get("units")
        if not weight_units or conversion(weight_units, "1") != 1:
            raise ValueError("Boundary weights must have unscaled dimensionless units.")
        weights[direction] = xr.apply_ufunc(
            checked_values,
            weight.astype("float64", copy=False),
            kwargs={"nonnegative": True},
            dask="parallelized",
            output_dtypes=[np.float64],
        )
        selected = selected.astype("float64", copy=False)
        if scale != 1:
            selected = selected * scale
        concentrations[direction] = xr.apply_ufunc(
            checked_values,
            selected,
            dask="parallelized",
            output_dtypes=[np.float64],
        )

    total = sum(w.sum([dim for dim in w.dims if dim != "time"], skipna=False) for w in weights.values())
    total = xr.apply_ufunc(checked_total, total, dask="parallelized", output_dtypes=[np.float64])
    # A sparse denominator's implicit zero would make the quotient's fill value NaN.
    total = to_dense(total)
    result = {}
    for direction in "nesw":
        mass = weights[direction] / total
        anomaly = mass * (concentrations[direction] - converted_reference)
        if output_scale != 1:
            anomaly = anomaly * output_scale
        result[f"bc_mass_{direction}"] = mass.assign_attrs(units="1")
        result[f"bc_anomaly_{direction}"] = anomaly.assign_attrs(units=output_units)
    result["bc_reference"] = xr.DataArray(converted_reference * output_scale, attrs={"units": output_units})
    return xr.Dataset(result, attrs={"bc_time_method": bc_time_method})


def select_boundary_representation(
    site_data: Mapping[str, xr.Dataset], representation: BoundaryRepresentation,
) -> dict[str, xr.Dataset]:
    """Give every site dataset the requested boundary representation.

    Surface acquisition keeps centered operands, so one merged dataset can feed
    either preparation. Ordinary preparation collapses them exactly to
    ``bc_<d> = bc_anomaly_<d> + bc_reference * bc_mass_<d>`` in float64,
    in the anomaly units; the reference cancels. Centered preparation keeps
    them unchanged. Input datasets are borrowed and never modified. Sites
    already in the requested representation, including sites without
    boundaries, map to the same dataset objects. Only centered sites requested
    as ordinary map to new datasets, which share their remaining variables
    with the input. Lazy payloads stay lazy.

    Args:
        site_data: Per-site merged datasets.
        representation: ``"ordinary"`` for absolute ``bc_n/e/s/w``, or
            ``"centered"`` for ``bc_anomaly``/``bc_mass``/``bc_reference``.

    Returns:
        A new mapping of site datasets in the requested representation.

    Raises:
        ValueError: If the representation is unknown, or centered operands are
            requested from a site that has only absolute boundary sensitivities.
    """
    if representation not in get_args(BoundaryRepresentation):
        raise ValueError(f"boundary_representation must be one of {get_args(BoundaryRepresentation)!r}.")
    result = {}
    for site, dataset in site_data.items():
        centered = dataset.attrs.get("boundary_representation") == "centered-v1"
        if representation == "centered" and not centered and "bc_n" in dataset:
            raise ValueError(
                f"Merged data for site {site!r} has only absolute boundary sensitivities and cannot "
                "produce centered operands; acquire it through surface transport."
            )
        if representation == "centered" or not centered:
            result[site] = dataset
            continue
        reference = dataset["bc_reference"]
        absolute = {
            f"bc_{d}": (dataset[f"bc_anomaly_{d}"] + reference * dataset[f"bc_mass_{d}"]).assign_attrs(
                units=dataset[f"bc_anomaly_{d}"].attrs["units"]
            )
            for d in "nesw"
        }
        operands = [f"{prefix}_{d}" for prefix in ("bc_anomaly", "bc_mass") for d in "nesw"]
        collapsed = dataset.drop_vars([*operands, "bc_reference"]).assign(absolute)
        collapsed.attrs = {key: value for key, value in dataset.attrs.items() if key != "boundary_representation"}
        result[site] = collapsed
    return result
