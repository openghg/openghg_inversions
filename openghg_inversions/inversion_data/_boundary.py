"""Accurate centered boundary operands for stable CO2/O2 surface inversions."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import numpy as np
import xarray as xr

from openghg_inversions.array_ops import to_dense

from ._units import mole_fraction_unit_scale


@dataclass(frozen=True)
class BoundaryTemporalPolicy:
    """Explicit native-field temporal contract; hold preserves archived behavior.

    Nearest/linear require point samples and a positive ``max_gap`` duration.
    For nearest this limits distance to the selected sample (ties choose later);
    for linear it limits the enclosing sample interval. Neither extrapolates.
    Interval selection requires declared interval means and ``bounds`` naming
    eager datetime metadata of shape (time, 2); intervals are left closed and
    right open. It uses a constant representative mean inside each interval,
    without claiming to preserve transported means for varying particle weights.
    Fresh retrieval additionally requires positive ``retrieval_padding`` for
    opt-in policies, explicitly widening the search to adjacent files.
    """

    method: Literal["hold", "nearest", "linear", "interval"] = "hold"
    input_semantics: Literal["point", "interval_mean"] | None = None
    max_gap: str | pd.Timedelta | None = None
    bounds: str | None = None
    retrieval_padding: str | pd.Timedelta | None = None

    def __post_init__(self) -> None:
        if self.method not in {"hold", "nearest", "linear", "interval"}:
            raise ValueError("Unknown boundary temporal policy.")
        if self.method in {"nearest", "linear"}:
            if self.input_semantics != "point" or self.max_gap is None:
                raise ValueError("Nearest/linear boundary policies require point semantics and max_gap.")
        elif self.method == "interval":
            if self.input_semantics != "interval_mean" or not self.bounds:
                raise ValueError("Interval boundary policy requires interval_mean semantics and bounds.")
        elif any(value is not None for value in (self.input_semantics, self.max_gap, self.bounds, self.retrieval_padding)):
            raise ValueError("Hold uses archived time_period semantics; do not supply another input contract.")
        if self.method in {"nearest", "linear"} and self.bounds is not None:
            raise ValueError("Point-sample policies do not accept interval bounds.")
        if self.method == "interval" and self.max_gap is not None:
            raise ValueError("Interval coverage is defined by explicit bounds, not max_gap.")
        for name in ("max_gap", "retrieval_padding"):
            value = getattr(self, name)
            if value is not None:
                duration = pd.Timedelta(value)
                if pd.isna(duration) or duration <= pd.Timedelta(0):
                    raise ValueError(f"Boundary {name} must be a positive finite duration.")


def select_boundary_fields(
    bc: xr.Dataset, times: xr.DataArray, *, policy: BoundaryTemporalPolicy,
) -> xr.Dataset:
    """Select/interpolate borrowed native curtains before particle contraction.

    Only indexed time and explicit eager interval metadata are inspected here.
    Scientific field arrays remain lazy and inputs are never modified.
    """
    from openghg.util import relative_time_offset

    if "time" not in bc.indexes or times.dims != ("time",):
        raise ValueError("Boundary time must be indexed and targets must use the time dimension.")
    for time in (bc.time, times):
        index = time.to_index()
        if (not isinstance(index, pd.DatetimeIndex) or not len(index) or index.hasnans
                or not index.is_unique or not index.is_monotonic_increasing):
            raise ValueError("Boundary and target time must be nonempty, unique and increasing.")
    source, target = bc.indexes["time"], times.to_index()
    fields = bc[[f"vmr_{d}" for d in "nesw"]]
    if policy.method == "hold":
        if "time_period" not in bc.attrs:
            raise ValueError("BC time_period must declare forward-fill coverage.")
        period = relative_time_offset(period=bc.attrs["time_period"])
        if source[0] + period <= source[0]:
            raise ValueError("BC time_period must be positive.")
        positions = source.get_indexer(target, method="ffill")
        if (positions < 0).any() or any(t >= source[int(i)] + period for t, i in zip(target, positions)):
            raise ValueError("BC time_period coverage does not cover every footprint observation.")
    elif policy.method == "interval":
        if policy.bounds not in bc:
            raise ValueError("Declared boundary bounds variable is missing.")
        bounds = bc[policy.bounds]
        if bounds.ndim != 2 or bounds.dims[0] != "time" or bounds.shape != (len(source), 2):
            raise ValueError("Boundary bounds must have shape (time, 2).")
        if hasattr(bounds.data, "__dask_graph__"):
            raise ValueError("Explicit boundary bounds must be eager datetime metadata.")
        if not np.issubdtype(bounds.dtype, np.datetime64):
            raise ValueError("Boundary bounds must contain datetime64 metadata.")
        starts, ends = pd.DatetimeIndex(bounds.data[:, 0]), pd.DatetimeIndex(bounds.data[:, 1])
        if (starts.hasnans or ends.hasnans or not starts.is_monotonic_increasing
                or not starts.is_unique or (ends <= starts).any()
                or (starts[1:] < ends[:-1]).any()
                or ((source < starts) | (source > ends)).any()):
            raise ValueError("Boundary intervals must be ordered, nonoverlapping and contain their labels.")
        positions = starts.get_indexer(target, method="ffill")
        if (positions < 0).any() or (target >= ends.take(positions)).any():
            raise ValueError("Boundary interval bounds do not cover every footprint observation.")
    else:
        if target[0] < source[0] or target[-1] > source[-1]:
            raise ValueError("Point-sample boundary policy cannot extrapolate beyond source labels.")
        maximum = pd.Timedelta(policy.max_gap)
        if policy.method == "nearest":
            positions = source.get_indexer(target, method="nearest")
            if (abs(source.take(positions) - target) > maximum).any():
                raise ValueError("Nearest boundary sample exceeds max_gap.")
        else:
            left = source.get_indexer(target, method="ffill")
            right = source.get_indexer(target, method="bfill")
            if (source.take(right) - source.take(left) > maximum).any():
                raise ValueError("Linear boundary support interval exceeds max_gap.")
            # Explicit linear weights avoid backend interpolation and keep sparse/Dask fields lazy.
            span = (source.take(right) - source.take(left)).as_unit("ns").asi8
            alpha = np.divide((target - source.take(left)).as_unit("ns").asi8, span,
                              out=np.zeros(len(target), dtype=float), where=span != 0)
            before = fields.isel(time=xr.DataArray(left, dims="time")).astype("float64").assign_coords(time=times)
            after = fields.isel(time=xr.DataArray(right, dims="time")).astype("float64").assign_coords(time=times)
            weight = xr.DataArray(alpha, dims="time", coords={"time": times})
            return xr.Dataset({name: (before[name] + weight * (after[name] - before[name])).assign_attrs(fields[name].attrs)
                               for name in fields}, attrs=bc.attrs)
    return fields.isel(time=xr.DataArray(positions, dims="time")).assign_coords(time=times)


def baseline_reference_sensitivities(
    bc: xr.Dataset,
    fp: xr.Dataset,
    *,
    species: str,
    reference: float,
    reference_units: str,
    mass_tolerance: float,
    temporal_policy: BoundaryTemporalPolicy | None = None,
) -> xr.Dataset:
    """Compute stable CO2/O2 boundary anomaly and mass operands.

    Each native curtain retains its dimensions. All four curtains are normalized
    together per observation, then ``anomaly = mass * (vmr - reference)``.
    Reconstruct absolute contributions as ``anomaly + bc_reference * mass``.
    Arithmetic uses the north curtain's scale. Output units retain its units,
    with any explicit numerical factor absorbed into the output values.

    By default BC timestamps mark interval starts, with validity given by the dataset's
    ``time_period`` attribute (OpenGHG period syntax, e.g. ``1 month``). Forward
    filling is allowed only inside those intervals, including for the final BC
    timestamp. Shared spatial indexes must match exactly; no spatial filling is
    performed. Inputs and their metadata are borrowed without mutation.

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
        temporal_policy: Explicit native-field sampling contract. None retains
            archived hold behavior; selection precedes particle contraction.

    Returns:
        Dataset with ``bc_anomaly_n/e/s/w``, dimensionless ``bc_mass_n/e/s/w``
        and scalar ``bc_reference``. All values are float64.

    Raises:
        ValueError: For invalid data, units, coordinates,
            coverage, reference or mass tolerance.
    """
    from openghg.util._units import cf_ureg

    if not np.isfinite(reference):
        raise ValueError("Boundary reference must be finite.")
    if not np.isfinite(mass_tolerance) or not 0 <= mass_tolerance < 1:
        raise ValueError("mass_tolerance must be finite and satisfy 0 <= tolerance < 1.")

    if "time" not in fp.indexes:
        raise ValueError("Boundary and footprint time must be indexed coordinates.")
    selected_fields = select_boundary_fields(bc, fp.time, policy=temporal_policy or BoundaryTemporalPolicy())

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
        selected = selected_fields[f"vmr_{direction}"]
        # Exact alignment cannot check dimensions lacking indexed coordinates.
        for dim in weight.dims:
            if dim not in weight.indexes or dim not in selected.indexes:
                raise ValueError(f"Boundary curtain coordinate {dim!r} must be indexed.")
            bc_unit, fp_unit = selected[dim].attrs.get("units"), weight[dim].attrs.get("units")
            if bc_unit and fp_unit and cf_ureg.Unit(bc_unit) != cf_ureg.Unit(fp_unit):
                raise ValueError(f"Boundary curtain coordinate units for {dim!r} must match exactly.")
        try:
            selected = selected.transpose(*weight.dims)
            selected, weight = xr.align(selected, weight, join="exact", copy=False)
        except ValueError as exc:
            raise ValueError("Boundary curtain dimensions and indexed coordinates must match exactly.") from exc
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
    return xr.Dataset(result)
