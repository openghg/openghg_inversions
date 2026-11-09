"""Compose native linked inputs with explicitly resolved atmospheric baselines."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import replace
import json
from typing import Any, TypeVar, cast

from dask import compute
from dask.array import Array as DaskArray
import numpy as np
from openghg.util import cf_ureg
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array, to_dense, to_dense_tree
from openghg_inversions.biosphere_prior import BiospherePrior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.boundary import AdditiveChoice, AnomalyScalingChoice
from openghg_inversions.models.linked_shared_baseline import bind_linked_shared_baseline
from openghg_inversions.models.shared_baseline import SharedBaseline, SharedBaselineChoice
from .co2_o2_preparation import Co2O2PreparedInputs


SHARED_SNAPSHOT_SCHEMA = "openghg_inversions.linked_shared_snapshot/v1"


def _choice_record(choice: SharedBaselineChoice) -> dict[str, Any]:
    """Encode only the finite data-free fields owned by the current choice."""
    result: dict[str, Any] = dict(mode=choice.mode, delta_sd=choice.delta_sd)
    boundary = choice.boundary
    if boundary is not None:
        options: dict[str, Any] = dict(mean_by_period=boundary.mean_by_period)
        if boundary.frequency is not None:
            options["frequency"] = boundary.frequency
        if boundary.anchor_time is not None:
            options["anchor_time"] = str(boundary.anchor_time)
        if isinstance(boundary, AdditiveChoice):
            options.update(anomaly_scale=boundary.anomaly_scale, anomalies=boundary.anomalies)
        elif isinstance(boundary, AnomalyScalingChoice):
            options.update(scale_sd=boundary.scale_sd, reference=boundary.reference)
        else:
            raise ValueError("Unsupported shared baseline boundary choice.")
        result["boundary"] = options
    return result


def _snapshot_bound(data: xr.Dataset) -> SharedBaseline:
    """Restore factual operators directly; never bind or recenter a saved fit."""
    from .prior_configuration import shared_baseline_choice

    try:
        choice = shared_baseline_choice(json.loads(data.attrs["shared_choice_json"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid saved shared baseline choice; prepare again.") from exc
    return SharedBaseline(choice, data)


def _native_channel(array: xr.DataArray, channel: str, row: str) -> xr.DataArray:
    return select_gathered_data_array(
        array, key=channel, key_dim="species", ragged_dim="channel_observation", stack_dim="observation",
    ).rename(observation=row)


def _channel_units(observations: xr.DataArray) -> str:
    units = np.unique(observations.observation_units.values)
    if len(units) != 1 or not isinstance(units[0], str) or not units[0].strip():
        raise ValueError("Each native channel requires one explicit concentration unit.")
    units = str(units[0])
    cf_ureg.Quantity(1, units).to("mol/mol")
    return units


def prepare_linked_shared_inputs(
    prepared: Co2O2PreparedInputs,
    *,
    choices: Mapping[str, SharedBaselineChoice],
    outer: xr.DataArray,
    sampled_outer: xr.DataArray,
    calibrated_prior: BiospherePrior | None = None,
) -> Co2O2PreparedInputs:
    """Freeze a linked baseline on complete retained native channel support.

    Both channels require raw centered boundary operands, explicit boundary
    choices whose positive scalar atmospheric widths use native concentration units. Outer
    selectors use the original joint prior; O2 responses are already signed.
    A supplied calibration is reused without fitting new moments. Preparation
    borrows transport arrays and returns a new handoff; later subsets select
    frozen rows rather than repeating this operation.

    Args:
        prepared: Native linked handoff with raw centered BC operands.
        choices: Channel-keyed data-free shared baseline choices.
        outer: Boolean selector on the original joint physical state axis.
        sampled_outer: Adjustable outer subset excluding outer ocean.
        calibrated_prior: Optional already resolved positive calibration.

    Returns:
        New borrowed handoff carrying frozen native-row baselines. Transport stays
        lazy; independently mutable input payloads are not overwritten.
    """
    channels = {"co2", "o2"}
    if prepared.shared_baseline:
        raise ValueError("Shared baselines are already frozen; subset their rows without preparing again.")
    if any(set(mapping) != channels for mapping in (choices, prepared.boundary_operands)):
        raise ValueError("Supply co2 and o2 shared choices and raw boundary operands.")
    # Small receptor metadata share one boundary; transport stays borrowed.
    metadata, fixed_units = compute(
        {name: prepared.observations[name] for name in ("site", "time", "observation_units")
         if name in prepared.observations.coords},
        prepared.fixed_prior_contribution.observation_units,
    )
    row_observations = prepared.observations.assign_coords(metadata)
    row_fixed = prepared.fixed_prior_contribution.assign_coords(observation_units=fixed_units)
    sensitivities, fixed, observations = {}, {}, {}
    for channel in ("co2", "o2"):
        h = getattr(prepared, f"{channel}_sensitivity")
        row = str(h.dims[0])
        obs = _native_channel(row_observations, channel, row)
        units = _channel_units(obs)
        if h.attrs.get("units") not in {units, f"{units} per dimensionless flux scale"}:
            raise ValueError(f"{channel} sensitivity units must match native observations.")
        h, obs = xr.align(h, obs, join="exact", copy=False)
        native_fixed = _native_channel(row_fixed, channel, row)
        if _channel_units(native_fixed) != units:
            raise ValueError(f"{channel} fixed contribution units must match native observations.")
        obs = obs.assign_attrs(units=units)
        # Observations own row metadata, while the supplied transport remains borrowed.
        row_coords = {name: coord for name, coord in obs.coords.items() if coord.dims == (row,)}
        state_coords = {
            name: coord for name, coord in prepared.retained_prior.mean.coords.items()
            if prepared.retained_prior.state_dim in coord.dims and name not in h.indexes
        }
        # The validated retained prior owns state meanings; reuse its coordinates
        # instead of asking xarray to compare lazy auxiliary copies in arithmetic.
        sensitivities[channel] = h.assign_coords({**row_coords, **state_coords}).assign_attrs(units=units)
        fixed[channel] = native_fixed.assign_attrs(units=units)
        observations[channel] = obs
    frozen = bind_linked_shared_baseline(
        choices, observations,
        co2_sensitivity=sensitivities["co2"], o2_sensitivity=sensitivities["o2"],
        retained_prior=prepared.retained_prior, fixed_prior_contribution=fixed,
        boundary_sensitivity=prepared.boundary_operands,
        outer=outer, sampled_outer=sampled_outer, calibrated_prior=calibrated_prior,
    )
    def original_selector(selector: xr.DataArray) -> xr.DataArray:
        return selector.rename({
            name: "original_state" if name == prepared.retained_prior.state_dim else f"original_state_{name}"
            for name in selector.coords
        })

    for channel, bound in frozen.items():
        value = bound.data.assign_attrs(shared_snapshot_schema=SHARED_SNAPSHOT_SCHEMA,
                                        shared_choice_json=json.dumps(_choice_record(bound.choice), sort_keys=True))
        if calibrated_prior is not None:
            value.attrs["calibration_attrs"] = json.dumps(calibrated_prior.to_dataset().attrs, sort_keys=True)
        row = value.attrs["native_observation_dim"]
        value = value.assign_coords({name: coord for name, coord in observations[channel].coords.items()
                                     if coord.dims == (row,)})
        raw = _native_raw_boundary(prepared.boundary_operands[channel], row, observations[channel])
        for name in ("H_bc_anomaly", "H_bc_mass", "bc_reference"):
            value[f"native_{name}"] = raw[name]
        value.attrs["native_ratio_provenance"] = prepared.o2_sensitivity.attrs["oxidation_ratio_provenance"]
        value.attrs["native_sensitivity_units"] = getattr(prepared, f"{channel}_sensitivity").attrs.get("units", "")
        value["original_outer"] = original_selector(outer)
        value["original_sampled_outer"] = original_selector(sampled_outer)
        value.delta_sd.attrs["units"] = value.attrs["units"]
        frozen[channel] = replace(bound, data=value)
    return replace(prepared, shared_baseline=frozen)


def _native_raw_boundary(raw: xr.Dataset, row: str, observations: xr.DataArray) -> xr.Dataset:
    raw_row = next(str(dim) for dim in raw.H_bc_anomaly.dims if dim != "bc_region")
    if raw_row == "time":
        raw = raw.sel(time=observations.time)
        if "time" not in raw.dims:
            raw_row = row
        raw = raw.assign_coords({row: observations[row]})
    rename = {"bc_region": "native_bc_region"}
    if raw_row != row:
        rename[raw_row] = row
    return raw.rename(rename)


def _same_array(
    left: xr.DataArray, right: xr.DataArray, name: Hashable, *, roundoff: tuple[xr.DataArray, int] | None = None,
) -> None:
    """Require equal labels and values; reduced operators may differ by summation roundoff.

    Args:
        left: Saved array.
        right: Recomputed array.
        name: Name used in error messages.
        roundoff: For reduced operators, ``(scale, terms)``: the sum of the
            absolute values of the summed terms on ``right``'s axes, and how
            many terms each element sums. ``None`` requires exact equality.

    Raises:
        ValueError: If labels, coordinates or values differ beyond the allowed
            roundoff.
    """
    if left.dims != right.dims or left.sizes != right.sizes or any(
        (dim in left.indexes) != (dim in right.indexes)
        or (dim in left.indexes and not left.indexes[dim].equals(right.indexes[dim])) for dim in left.dims
    ):
        raise ValueError(f"Shared baseline {name} labels must match.")
    if not isinstance(left.data, DaskArray) and not isinstance(right.data, DaskArray):
        if not left.variable.equals(right.variable):
            # Dask and NumPy reductions may accumulate the same operands in a
            # different order; two orders of an n-term sum differ by at most about
            # 2 n eps times the sum of absolute terms. Direct input identities remain exact.
            if roundoff is None:
                raise ValueError(f"Shared baseline {name} values must match.")
            scale, terms = roundoff
            eps = np.finfo(np.float64).eps
            allowed = 2 * max(terms, 1) * eps * np.asarray(scale.transpose(*right.dims).data) + 32 * eps * np.abs(right.data)
            if not np.all(np.abs(np.asarray(left.data) - np.asarray(right.data)) <= allowed):
                raise ValueError(f"Shared baseline {name} values must match.")
    for coordinate in right.coords:
        if coordinate not in left.coords:
            raise ValueError(f"Shared baseline {name} coordinate {coordinate} must match.")
        a, b = left[coordinate], right[coordinate]
        if not isinstance(a.data, DaskArray) and not isinstance(b.data, DaskArray) and not a.variable.equals(b.variable):
            raise ValueError(f"Shared baseline {name} coordinate {coordinate} must match.")


def _restore_state(array: xr.DataArray, prefix: str, prior: CorrelatedLognormalPrior) -> xr.DataArray:
    return array.rename({
        name: (prior.state_dim if name == prefix else prior.covariance_dim if name == f"{prefix}_cov"
               else name.removeprefix(f"{prefix}_"))
        for name in set(array.coords) | set(array.dims) if name.startswith(prefix)
    })


def _validate_operator_identity(
    value: xr.Dataset, h: xr.DataArray, fixed: xr.DataArray, raw_bc: xr.Dataset,
    observations: xr.DataArray, prior: CorrelatedLognormalPrior,
) -> None:
    """Bind frozen operators to their saved scientific inputs; never refit means."""
    raw = _native_raw_boundary(raw_bc, str(h.dims[0]), observations)
    for name in ("H_bc_anomaly", "H_bc_mass", "bc_reference"):
        _same_array(value[f"native_{name}"], raw[name], f"native {name}")
        if value[f"native_{name}"].attrs.get("units") != raw[name].attrs.get("units"):
            raise ValueError(f"Shared baseline native {name} units must match.")
    if h.attrs.get("units", "") != value.attrs["native_sensitivity_units"]:
        raise ValueError("Shared baseline native sensitivity units must match.")
    dim = prior.state_dim
    outer = _restore_state(value.original_outer, "original_state", prior)
    sampled = _restore_state(value.original_sampled_outer, "original_state", prior)
    means = _restore_state(value.outer_column_mean, "original_state", prior)
    multi = value.attrs["mode"].startswith("multiplicative")
    keep = ~outer | sampled if multi else ~outer
    positions, outer_positions = np.flatnonzero(keep.data), np.flatnonzero(outer.data)
    centered = h - means
    transformed = xr.where(sampled, centered, h).transpose(*h.dims) if multi else h
    physical_outer = xr.dot(h.where(outer, 0), prior.mean, dim=dim)
    expected_fixed = fixed + physical_outer
    if multi:
        expected_fixed = expected_fixed - xr.dot(centered.where(sampled, 0), prior.mean, dim=dim)
    expected = {
        "flux_sensitivity": transformed.isel({dim: positions}),
        "physical_outer_prior": physical_outer,
        "fixed_prior_contribution": expected_fixed,
    }
    _same_array(_restore_state(value.original_outer_operator, "physical_outer_state", prior),
                h.isel({dim: outer_positions}), "original outer operator")
    bound = raw.rename({"native_bc_region": "bc_region"})
    mass = bound.H_bc_mass.sum("bc_region", skipna=False)
    anomaly = (bound.H_bc_anomaly + (bound.bc_reference - value.reference) * bound.H_bc_mass).sum("bc_region", skipna=False)
    shift = value.reference * mass
    expected.update(bc_prescribed_centered=anomaly, bc_reference_shift=shift,
                    physical_bc_prior=anomaly + shift)
    expected["physical_prior_baseline"] = fixed + anomaly + shift + physical_outer
    expected["complete_prior_concentration"] = fixed + anomaly + shift + xr.dot(h, prior.mean, dim=dim)
    # Absolute-term sums bound the roundoff of each reduced operator.
    outer_scale = xr.dot(abs(h.where(outer, 0)), abs(prior.mean), dim=dim)
    fixed_scale = abs(fixed) + outer_scale
    if multi:
        fixed_scale = fixed_scale + xr.dot(abs(centered.where(sampled, 0)), abs(prior.mean), dim=dim)
    bc_scale = (abs(bound.H_bc_anomaly) + abs(bound.bc_reference - value.reference) * abs(bound.H_bc_mass)
                + abs(value.reference) * abs(bound.H_bc_mass)).sum("bc_region", skipna=False)
    shift_scale = abs(shift)
    states, regions = h.sizes[dim], bound.sizes["bc_region"]
    roundoff = {
        "physical_outer_prior": (outer_scale, states),
        "fixed_prior_contribution": (fixed_scale, states + 1),
        "bc_prescribed_centered": (bc_scale, 2 * regions),
        "bc_reference_shift": (shift_scale, regions),
        "physical_bc_prior": (bc_scale, 2 * regions),
        "physical_prior_baseline": (abs(fixed) + bc_scale + outer_scale, states + 2 * regions + 1),
        "complete_prior_concentration": (
            abs(fixed) + bc_scale + xr.dot(abs(h), abs(prior.mean), dim=dim), states + 2 * regions + 1,
        ),
    }
    for name, array in expected.items():
        _same_array(value[name], array, name, roundoff=roundoff.get(name))


def _validate_shared_baseline(prepared: Co2O2PreparedInputs, *, numerical: bool) -> None:
    """Validate frozen identities at preparation/storage boundaries, without refitting."""
    shared = prepared.shared_baseline
    if not shared:
        return
    if set(shared) != {"co2", "o2"} or set(prepared.boundary_operands) != {"co2", "o2"}:
        raise ValueError("Shared baselines require both channels and their raw boundary operands.")
    modes, channel_units = set(), set()
    required = {
        "original_prior_mean", "original_prior_covariance", "retained_mean", "retained_covariance",
        "original_outer", "original_sampled_outer", "sampled_outer", "original_outer_mean",
        "bc_prescribed_centered", "bc_reference_shift", "reference", "delta_sd",
        "centering_weights", "outer_column_mean", "bc_pattern_mean", "bc_pattern_raw", "flux_sensitivity",
        "original_outer_operator", "physical_outer_prior", "fixed_prior_contribution",
        "physical_bc_prior", "native_H_bc_anomaly", "native_H_bc_mass", "native_bc_reference",
        "physical_prior_baseline", "complete_prior_concentration",
    }
    for channel, bound in shared.items():
        value = bound.data
        if value.attrs.get("shared_snapshot_schema") != SHARED_SNAPSHOT_SCHEMA or required - set(value):
            raise ValueError(f"{channel} shared baseline is missing frozen scientific operands; prepare again.")
        if _choice_record(bound.choice) != json.loads(value.attrs["shared_choice_json"]):
            raise ValueError("Shared baseline choice identity must match its snapshot.")
        if bound.choice.mode != value.attrs.get("mode") or (numerical and bound.choice.delta_sd != float(value.delta_sd)):
            raise ValueError("Shared baseline choice and numerical operands must match.")
        if value.attrs.get("mode") not in {"fixed", "multiplicative", "multiplicative_shrunk"}:
            raise ValueError("Shared channels require a supported outer mode.")
        if (value.attrs["mode"] == "multiplicative_shrunk") != ("calibration_attrs" in value.attrs):
            raise ValueError("Only multiplicative_shrunk requires a resolved frozen calibration.")
        if value.attrs.get("native_ratio_provenance") != prepared.o2_sensitivity.attrs.get("oxidation_ratio_provenance"):
            raise ValueError("Shared baseline signed-ratio provenance must match.")
        row = str(getattr(prepared, f"{channel}_sensitivity").dims[0])
        if value.attrs.get("linked_channel") != channel or value.attrs.get("native_observation_dim") != row:
            raise ValueError(f"{channel} shared baseline native channel identity must match.")
        obs = _native_channel(prepared.observations, channel, row)
        units = (value.attrs.get("units")
                 if not numerical and isinstance(obs.observation_units.data, DaskArray) else _channel_units(obs))
        if value.attrs.get("units") != units or value.reference.attrs.get("units") != units:
            raise ValueError(f"{channel} shared baseline units must match observations.")
        if value.delta_sd.attrs.get("units") != units:
            raise ValueError(f"{channel} delta width units must match observations.")
        if row not in value.indexes or not value.indexes[row].equals(obs.indexes[row]):
            raise ValueError(f"{channel} shared baseline rows must match observations.")
        if value.reference.ndim or value.delta_sd.ndim:
            raise ValueError("Shared baseline references and delta widths must be scalar.")
        if numerical:
            original_h = getattr(prepared, f"{channel}_sensitivity")
            original_fixed = _native_channel(prepared.fixed_prior_contribution, channel, row)
            dense, raw_bc = value, prepared.boundary_operands[channel]
            if any(not np.isfinite(array.data).all() for array in dense.data_vars.values()
                   if np.issubdtype(array.dtype, np.number)):
                raise ValueError("Shared baseline numerical operands must be finite.")
            if float(dense.delta_sd) <= 0:
                raise ValueError("Shared baseline delta widths must be positive.")
            boundary = bound.choice.boundary
            if boundary is None or isinstance(boundary, AdditiveChoice) or (
                isinstance(boundary, AnomalyScalingChoice) and boundary.reference == "stored"
            ):
                _same_array(dense.reference, raw_bc.bc_reference, "stored reference")
            _validate_operator_identity(
                dense, original_h, original_fixed, raw_bc,
                obs.assign_attrs(units=units), prepared.retained_prior,
            )
        for name in ("original_outer", "original_sampled_outer", "sampled_outer"):
            if value[name].dtype != bool:
                raise ValueError("Shared baseline outer selectors must be boolean.")
        channel_units.add(units)
        modes.add(value.attrs.get("mode"))
        prior = prepared.retained_prior
        if value.attrs.get("state_dim") != prior.state_dim or value.attrs.get("covariance_dim") != prior.covariance_dim:
            raise ValueError("Shared baseline state dimensions must match the original prior.")
        for selector in (value.original_outer, value.original_sampled_outer):
            if selector.dims != ("original_state",) or not selector.get_index("original_state").equals(
                value.original_prior_mean.get_index("original_state")
            ):
                raise ValueError("Shared baseline original selectors must match the original prior.")
        for name, expected in (("original_prior_mean", prior.mean),
                               ("original_prior_covariance", prior.arithmetic_covariance)):
            original = value[name]
            rename = {coord: (prior.state_dim if coord == "original_state" else
                              prior.covariance_dim if coord == "original_state_cov" else
                              coord.removeprefix("original_state_"))
                      for coord in set(original.coords) | set(original.dims) if coord.startswith("original_state")}
            _same_array(original.rename(rename), expected, name)
    if len(modes) != 1 or not modes <= {"fixed", "multiplicative", "multiplicative_shrunk"}:
        raise ValueError("Both shared channels must use the same supported mode.")
    if len(channel_units) != 1:
        raise ValueError("Linked shared baselines require common channel units.")
    left, right = shared["co2"].data, shared["o2"].data
    calibration_names = {name for name in left if name.startswith("calibration_")}
    if calibration_names != {name for name in right if name.startswith("calibration_")}:
        raise ValueError("Both shared channels must retain the same calibration identity.")
    if left.attrs.get("calibration_attrs") != right.attrs.get("calibration_attrs"):
        raise ValueError("Both shared channels must retain the same calibration identity.")
    for field_name in {"retained_mean", "retained_covariance", "original_outer", "original_sampled_outer",
                       "sampled_outer", "original_outer_mean"} | calibration_names:
        _same_array(left[field_name], right[field_name], field_name)

    if numerical:
        from .co2_o2_preparation import _ratio_record
        ratio = prepared.o2_co2_flux_ratio
        ratio_record = _ratio_record(
            ratio, prepared.o2_co2_flux_ratio_unavailable_reason,
            np.asarray(ratio.data) if ratio is not None else np.empty(0),
        )
        if json.loads(left.attrs["native_ratio_provenance"]) != ratio_record:
            raise ValueError("Shared baseline signed-ratio values and provenance must match.")
        value = left
        selected_prior = prior
        if value.attrs["mode"] == "multiplicative_shrunk":
            calibration = value[[name for name in value if name.startswith("calibration_")]]
            calibration = calibration.rename({
                name: name.removeprefix("calibration_") for name in set(calibration.variables) | set(calibration.dims)
            }).assign_attrs(json.loads(value.attrs["calibration_attrs"]))
            if (calibration.attrs.get("biosphere_prior_schema") != "openghg_inversions.biosphere_prior/v1"
                    or calibration.attrs.get("family") != "lognormal"):
                raise ValueError("Shared shrinkage requires a resolved positive prior.")
            _same_array(calibration.baseline_mean, prior.mean, "calibration baseline mean")
            _same_array(calibration.baseline_covariance, prior.arithmetic_covariance,
                        "calibration baseline covariance")
            selected_prior = BiospherePrior.from_dataset(calibration).positive_prior
            if selected_prior is None:
                raise ValueError("Shared shrinkage requires a resolved positive prior.")
        outer, sampled = value.original_outer, value.original_sampled_outer
        if bool((sampled & ~outer).any()):
            raise ValueError("Shared sampled_outer must be a subset of outer.")
        keep = ~outer | sampled if value.attrs["mode"].startswith("multiplicative") else ~outer
        positions = np.flatnonzero(keep.data)
        _same_array(value.retained_mean, selected_prior.mean.isel({prior.state_dim: positions}),
                    "selected prior mean")
        _same_array(value.retained_covariance, selected_prior.arithmetic_covariance.isel(
            {prior.state_dim: positions, prior.covariance_dim: positions}),
            "selected prior covariance")


_Array = TypeVar("_Array", xr.DataArray, xr.Dataset)


def _consumed_coordinates(value: _Array, *, native_rows: tuple[str, ...] = ()) -> _Array:
    """Keep only the row coordinates the recipe consumes and saves, without computing."""
    coordinates = {"site", "time", "observation_units", "observation_units_cov"}
    rows = {"observation", "observation_cov", "nmeasure", "time",
            value.attrs.get("native_observation_dim"), *native_rows}
    return value.drop_vars([
        name for name, coordinate in value.coords.items()
        if name not in value.indexes and name not in coordinates and rows.intersection(coordinate.dims)
    ])


def _dense_consumed(value: _Array, *, native_rows: tuple[str, ...] = ()) -> _Array:
    """Select scientific coordinates before the named eager boundary."""
    value = _consumed_coordinates(value, native_rows=native_rows)
    return cast(_Array, value.map(to_dense, keep_attrs=True) if isinstance(value, xr.Dataset) else to_dense(value))


def materialize_linked_shared_inputs(
    prepared: Co2O2PreparedInputs, *, independent_error_sd: xr.DataArray | None = None,
    likelihood: Mapping[str, Any] | None = None,
) -> tuple[Co2O2PreparedInputs, xr.DataArray, dict[str, Any]]:
    """Jointly resolve frozen identities and graph operands, validate and reuse them.

    The returned record continues through model construction, annotation and saving.
    Borrowed source records and unused auxiliary diagnostics remain untouched.
    """
    from .co2_o2_runner import _validate_independent_error_labels, _validate_independent_error_values
    from openghg_inversions.models.boundary import require_centered_precision

    require_centered_precision()
    _validate_shared_baseline(prepared, numerical=False)
    saved = prepared.independent_error_sd
    error = saved if independent_error_sd is None else independent_error_sd
    if error is None:
        raise ValueError("Supply independent_error_sd explicitly or in the prepared artifact.")
    errors = [error] if saved is None or saved is error else [error, saved]
    for value in errors:
        _validate_independent_error_labels(prepared.observations, value)
    native_rows = (str(prepared.co2_sensitivity.dims[0]), str(prepared.o2_sensitivity.dims[0]))
    arrays, frozen, raw, ratio, aggregation, dense_errors, resolved_likelihood = compute(
        {name: _dense_consumed(getattr(prepared, name), native_rows=native_rows) for name in (
            "observations", "fixed_prior_contribution", "co2_sensitivity", "o2_sensitivity")},
        {name: _dense_consumed(value.data, native_rows=native_rows) for name, value in prepared.shared_baseline.items()},
        {name: _dense_consumed(value, native_rows=native_rows) for name, value in prepared.boundary_operands.items()},
        None if prepared.o2_co2_flux_ratio is None else _dense_consumed(prepared.o2_co2_flux_ratio),
        {name: (None if getattr(prepared.aggregation_error, name) is None
                else _dense_consumed(getattr(prepared.aggregation_error, name), native_rows=native_rows))
         for name in ("covariance", "factor", "diagonal_variance")},
        [_dense_consumed(value, native_rows=native_rows) for value in errors], to_dense_tree(dict(likelihood or {})),
        optimize_graph=False,
    )
    for value in dense_errors:
        _validate_independent_error_values(value)
        if not value.observation_units.variable.equals(arrays["observations"].observation_units.variable):
            raise ValueError("independent_error_sd observation_units must match prepared observations.")
    error = dense_errors[0]
    if len(dense_errors) == 2 and not error.variable.equals(dense_errors[1].variable):
        raise ValueError("Explicit independent_error_sd must match the prepared artifact values.")
    materialized = replace(
        prepared, **arrays, shared_baseline={name: replace(prepared.shared_baseline[name], data=value)
                                          for name, value in frozen.items()}, boundary_operands=raw,
        o2_co2_flux_ratio=ratio,
        aggregation_error=replace(prepared.aggregation_error, **aggregation),
        independent_error_sd=dense_errors[-1],
    )
    from .co2_o2_preparation import _validate_prepared_inputs
    _validate_prepared_inputs(materialized)
    return materialized, error, resolved_likelihood
