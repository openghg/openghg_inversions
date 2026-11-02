"""Explicit linked recipes with a separate shared BC/outer delta per tracer."""

from collections.abc import Mapping

from dask import compute

import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.array_ops import select_gathered_data_array, to_dense
from .co2_o2_model import build_co2_o2_model, _gather_co2_o2_sensitivity, _pad_channel_design
from .co2_o2_cached_sigma_model import build_co2_o2_cached_sigma_model
from .shared_baseline_model import _model_inputs


def _linked_inputs(prepared, observations):
    if set(prepared) != {"co2", "o2"}:
        raise ValueError("Expected co2 and o2 frozen shared baselines.")
    if len({p.attrs["units"] for p in prepared.values()}) != 1:
        raise ValueError("Linked shared baselines require common channel units; convert before preparation.")
    prepared, observations = compute(
        {channel: value.map(to_dense, keep_attrs=True) for channel, value in prepared.items()},
        to_dense(observations),
    )
    bound, corrections, priors = {}, {}, {}
    for channel in ("co2", "o2"):
        p = prepared[channel]
        if not {"bc_prescribed_centered", "bc_mass_response", "bc_reference_shift"} <= set(p):
            raise ValueError("Linked replay requires native centered BC operands; rebuild from original BC inputs.")
        if p.attrs["mode"] == "additive":
            raise ValueError("Linked additive net mode requires an explicit GPP/TER oxidative allocation.")
        row = p.attrs["native_observation_dim"]
        native = select_gathered_data_array(
            observations, key=channel, key_dim="species", ragged_dim="channel_observation",
            stack_dim=str(observations.dims[0]),
        ).rename({observations.dims[0]: "nmeasure"}).assign_attrs(units=p.attrs["units"])
        units = observations.observation_units.sel(
            {observations.dims[0]: observations[observations.dims[0]][observations.species == channel]}
        )
        if any(value != p.attrs["units"] for value in units.values):
            raise ValueError("Observation row units must match each frozen baseline.")
        rename = {row: "nmeasure"} if row != "nmeasure" else {}
        b, prior, correction = _model_inputs(p.rename(rename), native)
        undo = {"nmeasure": row} if row != "nmeasure" else {}
        bound[channel], corrections[channel], priors[channel] = b.rename(undo), correction.rename(undo), prior
    for name in ("original_prior_mean", "original_prior_covariance", "sampled_outer", "original_outer_mean"):
        xr.testing.assert_identical(bound["co2"][name], bound["o2"][name])
    xr.testing.assert_identical(priors["co2"].mean, priors["o2"].mean)
    xr.testing.assert_identical(priors["co2"].arithmetic_covariance, priors["o2"].arithmetic_covariance)
    calibration = {name for name in bound["co2"] if name.startswith("calibration_")}
    if calibration != {name for name in bound["o2"] if name.startswith("calibration_")}:
        raise ValueError("Both channels must retain the same resolved calibration identity.")
    for name in calibration:
        xr.testing.assert_identical(bound["co2"][name], bound["o2"][name])
    if bound["co2"].attrs.get("calibration_attrs") != bound["o2"].attrs.get("calibration_attrs"):
        raise ValueError("Both channels must retain the same resolved calibration identity.")
    if bound["co2"].attrs["mode"] != bound["o2"].attrs["mode"]:
        raise ValueError("Both channels must use the same resolved outer mode.")
    fixed = _gather_co2_o2_sensitivity(
        bound["co2"].fixed_prior_contribution, bound["o2"].fixed_prior_contribution,
        output_dim=str(observations.dims[0]),
    ).sel({observations.dims[0]: observations[observations.dims[0]]})
    return bound, priors["co2"], corrections, fixed, observations


def _diagnostics(model, prepared, observations):
    row = str(observations.dims[0])
    with model:
        model.linked_shared_baseline = prepared
        for channel, p in prepared.items():
            coefficients = model[f"{channel}_bc_pattern"] if p.sizes["bc_pattern_term"] else pt.zeros((0,))
            def padded(value):
                return _pad_channel_design(value, observations, channel).values
            pm.Deterministic(f"{channel}_baseline_delta", model[f"{channel}_bc_level"][0])
            raw_bc = padded(p.bc_pattern_raw) @ coefficients
            centered_bc = padded(p.bc_pattern_centered) @ coefficients
            pm.Deterministic(f"{channel}_bc_raw_correction", raw_bc, dims=row)
            pm.Deterministic(f"{channel}_bc_centered_correction", centered_bc, dims=row)
            pm.Deterministic(f"{channel}_bc_removed_mean", raw_bc - centered_bc, dims=row)
            departure = model.flux_scaling - p.retained_mean.values
            raw = padded(p.selected_outer_raw) @ departure
            centered = padded(p.selected_outer_centered) @ departure
            pm.Deterministic(f"{channel}_outer_raw_correction", raw, dims=row)
            pm.Deterministic(f"{channel}_outer_centered_correction", centered, dims=row)
            pm.Deterministic(f"{channel}_outer_removed_mean", raw - centered, dims=row)
            def vector(value):
                return padded(value.expand_dims(scalar=[0], axis=1))[:, 0]
            mask = vector(xr.ones_like(p.physical_outer_prior))
            delta = model[f"{channel}_baseline_delta"] * mask
            pm.Deterministic(
                f"{channel}_shared_baseline_contribution",
                vector(p.physical_prior_baseline) + delta + centered_bc + centered, dims=row,
            )
            pm.Deterministic(
                f"{channel}_modelled_concentration",
                vector(p.fixed_prior_contribution + p.physical_bc_prior)
                + padded(p.flux_sensitivity) @ model.flux_scaling + delta + centered_bc, dims=row,
            )
            physical = vector(p.physical_outer_prior)
            pm.Deterministic(f"{channel}_outer_physical_contribution", physical + raw, dims=row)
    return model


def build_shared_co2_o2_model(
    prepared: Mapping[str, xr.Dataset], *, observations, independent_error_sd,
    aggregation_error, tau_hours=None, fixed_site_amplitudes=None, site_amplitude_prior=None,
):
    """Build one linked flux state with independent physical atmospheric deltas.

    Prepared sensitivities embed oxidative ratios. The complete joint retained
    prior and aggregation covariance pass unchanged to the existing builder.
    Boundary and total outputs are absolute; the likelihood carries the small
    native centered BC expression plus reference*(mass_response-1), retaining
    precision without constructing and subtracting an O2 background.

    Start Python with ``PYTENSOR_FLAGS=floatX=float64`` before imports,
    preserving any other configured flags. Float64 inputs alone are insufficient;
    see :ref:`linked-shared-baseline-precision`.
    """
    bound, prior, corrections, fixed, observations = _linked_inputs(prepared, observations)
    model = build_co2_o2_model(
        observations=observations, independent_error_sd=independent_error_sd,
        aggregation_error=aggregation_error, retained_prior=prior,
        co2_sensitivity=bound["co2"].flux_sensitivity, o2_sensitivity=bound["o2"].flux_sensitivity,
        fixed_prior_contribution=fixed, boundary_correction=corrections,
        tau_hours=tau_hours, fixed_site_amplitudes=fixed_site_amplitudes,
        site_amplitude_prior=site_amplitude_prior,
    )
    return _diagnostics(model, bound, observations)


def build_shared_co2_o2_cached_sigma_model(
    prepared: Mapping[str, xr.Dataset], *, observations, independent_error_sd,
    aggregation_error, tau_hours, site_amplitude_prior, initial_site_amplitudes=None,
):
    """Build the same linked mean with the existing matched OU cache target.

    Start Python with ``PYTENSOR_FLAGS=floatX=float64`` before imports,
    preserving any other configured flags. Float64 inputs alone are insufficient;
    see :ref:`linked-shared-baseline-precision`.
    """
    bound, prior, corrections, fixed, observations = _linked_inputs(prepared, observations)
    cached = build_co2_o2_cached_sigma_model(
        observations=observations, independent_error_sd=independent_error_sd,
        aggregation_error=aggregation_error, retained_prior=prior,
        co2_sensitivity=bound["co2"].flux_sensitivity, o2_sensitivity=bound["o2"].flux_sensitivity,
        fixed_prior_contribution=fixed, boundary_correction=corrections,
        tau_hours=tau_hours, site_amplitude_prior=site_amplitude_prior,
        initial_site_amplitudes=initial_site_amplitudes,
    )
    _diagnostics(cached.model, bound, observations)
    return cached


def annotate_linked_shared_baseline_trace(trace, model):
    """Persist both complete frozen native preparations beside restored samples."""
    for channel, prepared in model.linked_shared_baseline.items():
        if isinstance(trace, xr.DataTree):
            trace[f"shared_baseline_{channel}"] = xr.DataTree(prepared)
        else:
            trace.add_groups({f"shared_baseline_{channel}": prepared})
    return trace


def reconstruct_linked_shared_baseline(prepared, posterior):
    """Reconstruct native absolute concentrations from restored labelled samples.

    This pure operation consumes saved frozen preparations and posterior state,
    level and pattern coefficients. It builds no model and recalibrates nothing.
    Coordinates must first be restored with the model's normal sample codec.
    """
    result = {}
    for channel, p in prepared.items():
        state_dim = p.attrs["state_dim"]
        state = posterior.flux_scaling.sel({state_dim: p[state_dim]})
        delta = posterior[f"{channel}_bc_level"]
        delta = delta.isel({delta.dims[-1]: 0}, drop=True)
        if p.sizes["bc_pattern_term"]:
            coefficient = posterior[f"{channel}_bc_pattern"]
            coefficient_dim = coefficient.dims[-1]
            if not coefficient.get_index(coefficient_dim).equals(p.get_index("bc_pattern_term")):
                raise ValueError(f"{channel} posterior pattern labels must match the frozen preparation.")
            coefficient = coefficient.rename({coefficient_dim: "bc_pattern_term"})
            coefficient = coefficient.assign_coords(bc_pattern_term=p.bc_pattern_term)
            bc = xr.dot(p.bc_pattern_centered, coefficient, dim="bc_pattern_term")
            raw_bc = xr.dot(p.bc_pattern_raw, coefficient, dim="bc_pattern_term")
        else:
            bc = xr.zeros_like(p.physical_bc_prior)
            raw_bc = bc
        departure = state - p.retained_mean
        outer = xr.dot(p.selected_outer_centered, departure, dim=state_dim)
        raw = xr.dot(p.selected_outer_raw, departure, dim=state_dim)
        result[channel] = xr.Dataset({
            "baseline_delta": delta,
            "bc_raw_correction": raw_bc,
            "bc_centered_correction": bc,
            "bc_removed_mean": raw_bc - bc,
            "outer_raw_correction": raw,
            "outer_centered_correction": outer,
            "outer_removed_mean": raw - outer,
            "outer_physical_contribution": p.physical_outer_prior + raw,
            "shared_baseline_contribution": p.physical_prior_baseline + delta + bc + outer,
            "modelled_concentration": p.fixed_prior_contribution + p.physical_bc_prior + delta + bc
            + xr.dot(p.flux_sensitivity, state, dim=state_dim),
        }, attrs={"units": p.attrs["units"]})
    return result
