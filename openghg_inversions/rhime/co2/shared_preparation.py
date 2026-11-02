"""Compose native linked inputs with explicitly resolved atmospheric baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json

from dask import compute
from dask.array import Array as DaskArray
import numpy as np
from openghg.util import cf_ureg
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array, to_dense
from openghg_inversions.biosphere_prior import BiospherePrior
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.boundary import bind_boundary_operands
from openghg_inversions.models.linked_shared_baseline import prepare_linked_shared_baseline
from openghg_inversions.models.shared_baseline import SCHEMA
from .co2_o2_preparation import Co2O2PreparedInputs
from .prior_configuration import BoundaryCorrectionConfig


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


def _boundary_row_metadata(observations: xr.DataArray) -> xr.DataArray:
    """Resolve receptor metadata together; keep concentration/transport borrowed."""
    coordinates = {
        name: observations[name] for name in ("site", "time")
        if name in observations.coords and isinstance(observations[name].data, DaskArray)
    }
    resolved = compute(coordinates)[0] if coordinates else {}
    # Boundary preparation consumes row labels, times and units only. Eager
    # placeholders keep its zero-period vector independent of lazy observations.
    return observations.assign_coords(resolved).copy(data=np.zeros(observations.shape, dtype=np.float64))


def prepare_linked_shared_inputs(
    prepared: Co2O2PreparedInputs,
    *,
    boundary: Mapping[str, BoundaryCorrectionConfig],
    outer: xr.DataArray,
    sampled_outer: xr.DataArray,
    mode: str,
    delta_sd: Mapping[str, xr.DataArray],
    calibrated_prior: BiospherePrior | None = None,
) -> Co2O2PreparedInputs:
    """Freeze a linked baseline on complete retained native channel support.

    Both channels require raw centered boundary operands, explicit boundary
    configuration and positive scalar delta widths with physical units. Outer
    selectors use the original joint prior; O2 responses are already signed.
    A supplied calibration is reused without fitting new moments. Preparation
    borrows transport arrays and returns a new handoff; later subsets select
    frozen rows rather than repeating this operation.
    """
    channels = {"co2", "o2"}
    if prepared.shared_baseline:
        raise ValueError("Shared baselines are already frozen; subset their rows without preparing again.")
    if any(set(mapping) != channels for mapping in (boundary, delta_sd, prepared.boundary_operands)):
        raise ValueError("Supply co2 and o2 boundary configs, raw boundary operands and delta widths.")
    sensitivities, fixed, corrections, widths = {}, {}, {}, {}
    for channel in ("co2", "o2"):
        h = getattr(prepared, f"{channel}_sensitivity")
        row = str(h.dims[0])
        obs = _native_channel(prepared.observations, channel, row)
        units = _channel_units(obs)
        if h.attrs.get("units") not in {units, f"{units} per dimensionless flux scale"}:
            raise ValueError(f"{channel} sensitivity units must match native observations.")
        xr.align(h, obs, join="exact", copy=False)
        native_fixed = _native_channel(prepared.fixed_prior_contribution, channel, row)
        if _channel_units(native_fixed) != units:
            raise ValueError(f"{channel} fixed contribution units must match native observations.")
        obs = _boundary_row_metadata(obs.assign_attrs(units=units))
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
        corrections[channel] = boundary[channel].prepare(prepared.boundary_operands[channel], obs)
        width = delta_sd[channel]
        if not isinstance(width, xr.DataArray) or width.ndim != 0 or not width.attrs.get("units"):
            raise ValueError(f"{channel} delta_sd requires a scalar DataArray with physical units.")
        widths[channel] = float(cf_ureg.Quantity(width.item(), width.attrs["units"]).to(units).magnitude)
    if len({correction.bc_reference.attrs["units"] for correction in corrections.values()}) != 1:
        raise ValueError("Linked shared baselines require common channel concentration units.")
    frozen = prepare_linked_shared_baseline(
        co2_sensitivity=sensitivities["co2"], o2_sensitivity=sensitivities["o2"],
        retained_prior=prepared.retained_prior, fixed_prior_contribution=fixed,
        boundary_correction=corrections, outer=outer, sampled_outer=sampled_outer,
        mode=mode, delta_sd=widths, calibrated_prior=calibrated_prior,
    )
    for value in frozen.values():
        def original_selector(selector):
            return selector.rename({
                name: "original_state" if name == prepared.retained_prior.state_dim else f"original_state_{name}"
                for name in selector.coords
            })
        value["original_outer"] = original_selector(outer)
        value["original_sampled_outer"] = original_selector(sampled_outer)
        value.delta_sd.attrs["units"] = value.attrs["units"]
    result = replace(prepared, shared_baseline=frozen)
    _validate_shared_baseline(result, materialize=False)
    return result


def _same_array(left: xr.DataArray, right: xr.DataArray, name: str, *, materialize: bool) -> None:
    if left.dims != right.dims or left.sizes != right.sizes or any(
        (dim in left.indexes) != (dim in right.indexes)
        or (dim in left.indexes and not left.indexes[dim].equals(right.indexes[dim])) for dim in left.dims
    ):
        raise ValueError(f"Shared baseline {name} labels must match.")
    if materialize:
        left, right = compute(left, right)
    if not isinstance(left.data, DaskArray) and not isinstance(right.data, DaskArray):
        if not left.variable.equals(right.variable):
            raise ValueError(f"Shared baseline {name} values must match.")
    for coordinate in right.coords:
        if coordinate not in left.coords:
            raise ValueError(f"Shared baseline {name} coordinate {coordinate} must match.")
        a, b = left[coordinate], right[coordinate]
        if materialize:
            a, b = compute(a, b)
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
        "selected_outer_raw": h.isel({dim: positions}).where(sampled.isel({dim: positions}), 0),
        "selected_outer_centered": centered.isel({dim: positions}).where(sampled.isel({dim: positions}), 0),
    }
    _same_array(_restore_state(value.original_outer_operator, "physical_outer_state", prior),
                h.isel({dim: outer_positions}), "original outer operator", materialize=False)
    bound = bind_boundary_operands(raw_bc, observations)
    native_mass, native_anomaly = bound.H_bc_mass.astype(np.float64), bound.H_bc_anomaly.astype(np.float64)
    mass = native_mass.sum("bc_region", skipna=False)
    anomaly = (native_anomaly + (bound.bc_reference.astype(np.float64) - value.reference) * native_mass).sum(
        "bc_region", skipna=False,
    )
    expected.update(bc_mass_response=mass, bc_prescribed_centered=anomaly,
                    bc_reference_shift=value.reference * mass,
                    physical_bc_prior=anomaly + value.reference * mass)
    policy = json.loads(value.attrs["boundary_correction_attrs"])["reference_policy"]
    if policy == "stored_scalar":
        _same_array(value.reference, bound.bc_reference, "stored reference", materialize=False)
    expected["physical_prior_baseline"] = fixed + anomaly + value.reference * mass + physical_outer
    expected["complete_prior_concentration"] = fixed + anomaly + value.reference * mass + xr.dot(h, prior.mean, dim=dim)
    for name, array in expected.items():
        _same_array(value[name], array, name, materialize=False)


def _validate_shared_baseline(prepared: Co2O2PreparedInputs, *, materialize: bool) -> None:
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
        "bc_prescribed_centered", "bc_mass_response", "bc_reference_shift", "reference", "delta_sd",
        "centering_weights", "outer_column_mean", "bc_pattern_mean", "flux_sensitivity",
        "original_outer_operator", "physical_outer_prior", "fixed_prior_contribution",
        "selected_outer_raw", "selected_outer_centered", "physical_bc_prior",
        "physical_prior_baseline", "complete_prior_concentration",
    }
    for channel, value in shared.items():
        if value.attrs.get("shared_baseline_schema") != SCHEMA or required - set(value):
            raise ValueError(f"{channel} shared baseline is missing frozen scientific operands.")
        if value.attrs.get("mode") not in {"fixed", "multiplicative", "multiplicative_shrunk"}:
            raise ValueError("Shared channels require a supported outer mode.")
        if (value.attrs["mode"] == "multiplicative_shrunk") != ("calibration_attrs" in value.attrs):
            raise ValueError("Only multiplicative_shrunk requires a resolved frozen calibration.")
        row = str(getattr(prepared, f"{channel}_sensitivity").dims[0])
        if value.attrs.get("linked_channel") != channel or value.attrs.get("native_observation_dim") != row:
            raise ValueError(f"{channel} shared baseline native channel identity must match.")
        obs = _native_channel(prepared.observations, channel, row)
        units = (value.attrs.get("units")
                 if not materialize and isinstance(obs.observation_units.data, DaskArray) else _channel_units(obs))
        if value.attrs.get("units") != units or value.reference.attrs.get("units") != units:
            raise ValueError(f"{channel} shared baseline units must match observations.")
        if value.delta_sd.attrs.get("units") != units:
            raise ValueError(f"{channel} delta width units must match observations.")
        if row not in value.indexes or not value.indexes[row].equals(obs.indexes[row]):
            raise ValueError(f"{channel} shared baseline rows must match observations.")
        if value.reference.ndim or value.delta_sd.ndim:
            raise ValueError("Shared baseline references and delta widths must be scalar.")
        if materialize:
            original_h = getattr(prepared, f"{channel}_sensitivity")
            original_fixed = _native_channel(prepared.fixed_prior_contribution, channel, row)
            dense, original_h, original_fixed, raw_bc = compute(
                value.map(to_dense, keep_attrs=True), to_dense(original_h), to_dense(original_fixed),
                prepared.boundary_operands[channel].map(to_dense, keep_attrs=True),
            )
            if any(not np.isfinite(array.data).all() for array in dense.data_vars.values()
                   if np.issubdtype(array.dtype, np.number)):
                raise ValueError("Shared baseline numerical operands must be finite.")
            if float(dense.delta_sd) <= 0:
                raise ValueError("Shared baseline delta widths must be positive.")
            _validate_operator_identity(
                dense, original_h, original_fixed, raw_bc,
                _boundary_row_metadata(obs).assign_attrs(units=units), prepared.retained_prior,
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
            _same_array(original.rename(rename), expected, name, materialize=materialize)
        if materialize:
            selected_prior = prior
            if value.attrs["mode"] == "multiplicative_shrunk":
                calibration = value[[name for name in value if name.startswith("calibration_")]]
                calibration = calibration.rename({
                    name: name.removeprefix("calibration_") for name in set(calibration.variables) | set(calibration.dims)
                }).assign_attrs(json.loads(value.attrs["calibration_attrs"]))
                if (calibration.attrs.get("biosphere_prior_schema") != "openghg_inversions.biosphere_prior/v1"
                        or calibration.attrs.get("family") != "lognormal"):
                    raise ValueError("Shared shrinkage requires a resolved positive prior.")
                _same_array(calibration.baseline_mean, prior.mean, "calibration baseline mean", materialize=True)
                _same_array(calibration.baseline_covariance, prior.arithmetic_covariance,
                            "calibration baseline covariance", materialize=True)
                # Validate stored moments directly; the calibration codec also
                # recomputes its policy, which is outside this frozen handoff.
                selected_prior = CorrelatedLognormalPrior(
                    calibration.baseline_mean, calibration.covariance, covariance_dim=prior.covariance_dim,
                )
            outer, sampled = compute(value.original_outer, value.original_sampled_outer)
            if bool((sampled & ~outer).any()):
                raise ValueError("Shared sampled_outer must be a subset of outer.")
            keep = ~outer | sampled if value.attrs["mode"].startswith("multiplicative") else ~outer
            positions = np.flatnonzero(keep.data)
            _same_array(value.retained_mean, selected_prior.mean.isel({prior.state_dim: positions}),
                        "selected prior mean", materialize=True)
            _same_array(value.retained_covariance, selected_prior.arithmetic_covariance.isel(
                {prior.state_dim: positions, prior.covariance_dim: positions}),
                "selected prior covariance", materialize=True)
    if len(modes) != 1 or not modes <= {"fixed", "multiplicative", "multiplicative_shrunk"}:
        raise ValueError("Both shared channels must use the same supported mode.")
    if len(channel_units) != 1:
        raise ValueError("Linked shared baselines require common channel units.")
    left, right = shared["co2"], shared["o2"]
    calibration = {name for name in left if name.startswith("calibration_")}
    if calibration != {name for name in right if name.startswith("calibration_")}:
        raise ValueError("Both shared channels must retain the same calibration identity.")
    if left.attrs.get("calibration_attrs") != right.attrs.get("calibration_attrs"):
        raise ValueError("Both shared channels must retain the same calibration identity.")
    for name in {"retained_mean", "retained_covariance", "original_outer", "original_sampled_outer",
                 "sampled_outer", "original_outer_mean"} | calibration:
        _same_array(left[name], right[name], name, materialize=materialize)
