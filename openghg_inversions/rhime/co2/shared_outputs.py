"""Graph-free native-channel products for frozen linked shared baselines."""

from __future__ import annotations

import json

import numpy as np
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array
from openghg_inversions.postprocessing.inversion_output import trace_group
from openghg_inversions.postprocessing.linked_paris_outputs import _role

from .co2_o2_preparation import Co2O2PreparedInputs
from .linked_shared_baseline_model import reconstruct_linked_shared_baseline


SHARED_RECIPE = "co2_o2_shared_baseline"


def physical_flux_scaling(trace: xr.DataTree, group: str) -> xr.DataArray:
    """Restore the original physical state labels beside selected sampled states."""
    value = _role(trace, "flux_scale", group)
    state_dim = trace["shared_baseline_co2"].attrs["state_dim"]
    physical_dim = trace.attrs.get("physical_state_dim", "physical_state")
    rename = {name: name.removeprefix("physical_") for name, coord in value.coords.items()
              if physical_dim in coord.dims and name.startswith("physical_")}
    rename[physical_dim] = state_dim
    return value.rename(rename)


def validate_shared_baseline_trace(trace: xr.DataTree, prepared: Co2O2PreparedInputs | None) -> None:
    """Reject missing frozen operands and preparation/trace identity mismatches."""
    for channel in ("co2", "o2"):
        name = f"shared_baseline_{channel}"
        if name not in trace.children:
            raise ValueError(f"Linked shared outputs require saved {name} operands.")
        if prepared is not None:
            if channel not in prepared.shared_baseline:
                raise ValueError("Linked shared outputs require a shared-baseline preparation.")
            try:
                xr.testing.assert_identical(trace[name].to_dataset(inherit=False), prepared.shared_baseline[channel])
            except AssertionError as exc:
                raise ValueError(f"Saved {channel} shared baseline differs from the preparation.") from exc
        frozen = trace[name].to_dataset(inherit=False)
        for group in ("prior", "posterior"):
            if group not in trace.children:
                continue
            samples = trace_group(trace, group)
            state_dim = frozen.attrs["state_dim"]
            if "flux_scaling" not in samples or set(samples.flux_scaling.dims) != {"chain", "draw", state_dim}:
                raise ValueError("Linked shared outputs require selected flux_scaling draws with chain/draw/state axes.")
            xr.align(samples.flux_scaling, frozen.retained_mean, join="exact", copy=False)
            if f"{channel}_bc_level" not in samples:
                raise ValueError(f"Linked shared outputs require {channel} physical baseline delta draws.")
            if frozen.sizes["bc_pattern_term"] and f"{channel}_bc_pattern" not in samples:
                raise ValueError(f"Linked shared outputs require {channel} boundary pattern draws.")


def reconstruct_shared_concentrations(
    trace: xr.DataTree, prepared: Co2O2PreparedInputs,
) -> dict[str, xr.Dataset]:
    """Retain atmospheric accounting separately from physical source flux effects.

    The reviewed pure reconstructor owns centered atmospheric components. Original
    sensitivities and full physical scales produce per-source physical effects;
    those effects exclude the removed-mean atmospheric corrections. All products
    preserve native channel rows and the shared posterior draw labels.
    """
    validate_shared_baseline_trace(trace, prepared)
    result = {}
    for group in ("prior", "posterior"):
        if group not in trace.children:
            continue
        components = reconstruct_linked_shared_baseline(prepared.shared_baseline, trace_group(trace, group))
        scales = physical_flux_scaling(trace, group)
        state_dim = prepared.retained_prior.state_dim
        for channel in ("co2", "o2"):
            frozen = prepared.shared_baseline[channel]
            row = frozen.attrs["native_observation_dim"]
            values = components[channel]
            if group == "prior":
                values = values.rename({dim: f"prior_{dim}" for dim in ("chain", "draw") if dim in values.dims})
            values = values.rename({name: f"{name}_{group}" for name in values.data_vars})
            observed = select_gathered_data_array(
                prepared.observations, key=channel, key_dim="species", ragged_dim="channel_observation",
                stack_dim="observation",
            ).rename(observation=row)
            values["observed"] = observed
            values["affine_physical"] = select_gathered_data_array(
                prepared.fixed_prior_contribution, key=channel, key_dim="species",
                ragged_dim="channel_observation", stack_dim="observation",
            ).rename(observation=row)
            values[f"residual_{group}"] = observed - values[f"modelled_concentration_{group}"]
            sensitivity = getattr(prepared, f"{channel}_sensitivity")
            sensitivity, scales = xr.align(sensitivity, scales, join="exact", copy=False)
            selected = prepared.retained_prior.mean.tracer_scope.str.lower().isin(["shared", channel])
            sources = np.unique(prepared.retained_prior.mean.source.values[selected.values])
            physical_total = values.affine_physical
            for source in sources:
                states = np.flatnonzero((selected & (prepared.retained_prior.mean.source == source)).values)
                contribution = xr.dot(
                    sensitivity.isel({state_dim: states}), scales.isel({state_dim: states}), dim=state_dim,
                )
                values[f"physical_{str(source).lower()}_{group}"] = contribution
                physical_total = physical_total + contribution
            values[f"physical_flux_contribution_{group}"] = physical_total
            values[f"physical_bc_prior_{group}"] = frozen.physical_bc_prior
            values[f"physical_atmospheric_total_{group}"] = (
                physical_total + frozen.physical_bc_prior + values[f"baseline_delta_{group}"]
                + values[f"bc_raw_correction_{group}"]
            )
            values[f"removed_mean_correction_{group}"] = (
                values[f"bc_removed_mean_{group}"] + values[f"outer_removed_mean_{group}"]
            )
            prediction_group = f"{group}_predictive"
            roles = json.loads(trace.attrs["rhime_variable_roles"])
            if prediction_group in trace.children and "concentration" in roles:
                predictive = select_gathered_data_array(
                    _role(trace, "concentration", prediction_group), key=channel, key_dim="species",
                    ragged_dim="channel_observation", stack_dim="observation",
                ).rename(observation=row)
                values[f"predictive_{group}"] = predictive.rename({
                    dim: f"{prediction_group}_{dim}" for dim in ("chain", "draw") if dim in predictive.dims
                })
            if group == "posterior":
                values["total_error"] = select_gathered_data_array(
                    _role(trace, "total_marginal_error", group), key=channel, key_dim="species",
                    ragged_dim="channel_observation", stack_dim="observation",
                ).rename(observation=row)
            values.attrs.update(
                species=channel, units=frozen.attrs["units"],
                source_components=json.dumps([str(source).lower() for source in sources]),
                component_accounting=(
                    "modelled_concentration = physical_atmospheric_total - removed_mean_correction; "
                    "physical source contributions use original sensitivities and full physical states"
                ),
            )
            if channel in result:
                result[channel] = xr.merge([result[channel], values], join="exact", compat="equals")
            else:
                result[channel] = values
    return result
