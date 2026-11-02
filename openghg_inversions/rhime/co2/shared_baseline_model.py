"""Explicit ordinary and cached CO2 recipes with a shared atmospheric delta."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from dask import compute

from openghg_inversions.array_ops import to_dense
from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.coords import add_model_data, component_coords
from openghg_inversions.models.shared_baseline import SCHEMA
from openghg_inversions.rhime.co2.co2_model import build_co2_model
from openghg_inversions.rhime.co2.co2_cached_sigma_model import build_co2_cached_sigma_model


def _model_inputs(prepared, observations):
    """Materialize related frozen arrays together at the PyMC boundary."""
    if prepared.attrs.get("shared_baseline_schema") != SCHEMA:
        raise ValueError("Unsupported shared baseline schema.")
    prepared, observations = xr.align(prepared, observations, join="exact", copy=False)
    prepared = compute(prepared.map(to_dense, keep_attrs=True))[0]
    if observations.attrs.get("units") != prepared.attrs["units"]:
        raise ValueError("Shared baseline and observations must use the same declared concentration units.")
    if any(
        not np.isfinite(v.values).all()
        for v in prepared.data_vars.values()
        if np.issubdtype(v.dtype, np.number)
    ):
        raise ValueError("Shared baseline numerical operands must be finite.")
    prior = CorrelatedLognormalPrior(
        prepared.retained_mean,
        prepared.retained_covariance,
        covariance_dim=prepared.attrs["covariance_dim"],
    )
    patterns = prepared.bc_pattern_centered.reset_coords(drop=True)
    if "additive_centered" in prepared:
        additive = prepared.additive_centered.reset_coords(drop=True).rename(outer_pattern="bc_pattern_term")
        additive = additive.assign_coords(
            bc_pattern_term=[f"outer:{x}" for x in additive.bc_pattern_term.values]
        )
        patterns = xr.concat([patterns, additive], dim="bc_pattern_term", coords="minimal", compat="override")
    units = observations.attrs["units"]
    one = xr.ones_like(observations).assign_attrs(units="1")
    reference_shift = (one * prepared.reference).assign_attrs(units=units)
    # Older CO2 v1 artifacts only stored physical BC. Their replay retains
    # its historical precision; linked O2 explicitly requires native operands.
    centered = (
        prepared.bc_prescribed_centered + prepared.reference * (prepared.bc_mass_response - 1.0)
        if {"bc_prescribed_centered", "bc_mass_response"} <= set(prepared)
        else prepared.physical_bc_prior - reference_shift
    )
    correction = xr.Dataset(
        {
            "bc_prescribed_centered": centered.assign_attrs(units=units),
            "bc_reference_shift": reference_shift,
            "bc_reference": prepared.reference.assign_attrs(units=units),
            "bc_mass_response": one,
            "bc_level_operator": one.expand_dims(bc_level_term=[0], axis=1),
            "bc_pattern_operator": patterns.assign_attrs(units=units),
            "bc_level_sd": prepared.delta_sd.assign_attrs(units=units),
            "bc_pattern_sd": xr.DataArray(1.0, attrs={"units": "1"}),
            "bc_period_index": xr.zeros_like(observations, dtype=int),
        },
        attrs={
            "boundary_correction_schema": "independent-boundary-correction-v1",
            "method": "A",
            "level_support": "global",
        },
    )
    return prepared, prior, correction


def _diagnostics(model, prepared):
    """Expose physical raw, removed mean and centred corrections separately."""
    with model:
        model.shared_baseline = prepared
        for name, value in prepared.data_vars.items():
            add_model_data(component_coords(value, "shared"), f"shared_{name}")
        pm.Deterministic("baseline_delta", model.bc_level[0])
        nbc = prepared.sizes["bc_pattern_term"]
        bc = model.bc_pattern[:nbc] if nbc else pt.zeros((0,))
        pm.Deterministic("bc_raw_correction", prepared.bc_pattern_raw.values @ bc, dims="nmeasure")
        pm.Deterministic("bc_removed_mean", prepared.bc_pattern_mean.values @ bc)
        pm.Deterministic("bc_centered_correction", prepared.bc_pattern_centered.values @ bc, dims="nmeasure")
        if prepared.attrs["mode"].startswith("multiplicative"):
            departure = model.flux_scaling - prepared.retained_mean.values
            raw = prepared.selected_outer_raw.values @ departure
            centered = prepared.selected_outer_centered.values @ departure
        elif "additive_raw" in prepared:
            coefficient = model.bc_pattern[nbc:]
            raw = prepared.additive_raw.values @ coefficient
            centered = prepared.additive_centered.values @ coefficient
        else:
            raw = pt.zeros(prepared.sizes["nmeasure"])
            centered = raw
        pm.Deterministic(
            "shared_baseline_contribution",
            prepared.physical_prior_baseline.values
            + model.baseline_delta
            + model.bc_centered_correction
            + centered,
            dims="nmeasure",
        )
        pm.Deterministic(
            "physical_bc_contribution",
            prepared.physical_bc_prior.values + model.bc_raw_correction,
            dims="nmeasure",
        )
        if "additive_density_sd" in prepared:
            pm.Deterministic(
                "outer_density_correction",
                model.bc_pattern[nbc:] * prepared.additive_density_sd.values,
                dims="shared_outer_pattern",
            )
        pm.Deterministic("outer_raw_correction", raw, dims="nmeasure")
        pm.Deterministic("outer_centered_correction", centered, dims="nmeasure")
        pm.Deterministic("outer_removed_mean", raw - centered, dims="nmeasure")
        pm.Deterministic(
            "outer_physical_contribution", prepared.physical_outer_prior.values + raw, dims="nmeasure"
        )
    return model


def build_shared_co2_model(
    prepared: xr.Dataset,
    *,
    observations,
    observation_error,
    aggregation_error,
    likelihood_builder=None,
    likelihood_kwargs=None,
    sigma_alignment=None,
    sigma_prior=None,
    fixed_model_mismatch=None,
):
    """Build ordinary CO2 with frozen shared baseline and existing noise policy.

    The scalar reference centres only the likelihood; named predictions retain
    absolute concentration. The physical source of delta remains unidentified.
    Linked CO2/O2 channels are not supported by this separate recipe.
    """
    prepared, prior, correction = _model_inputs(prepared, observations)
    model = build_co2_model(
        flux_sensitivity=prepared.flux_sensitivity,
        retained_prior=prior,
        prune_zero_flux_columns=False,
        fixed_prior_contribution=prepared.fixed_prior_contribution,
        boundary_correction=correction,
        observations=observations,
        observation_error=observation_error,
        aggregation_error=aggregation_error,
        likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs,
        sigma_alignment=sigma_alignment,
        sigma_prior=sigma_prior,
        fixed_model_mismatch=fixed_model_mismatch,
    )
    return _diagnostics(model, prepared)


def build_shared_co2_cached_sigma_model(
    prepared: xr.Dataset,
    *,
    observations,
    observation_error,
    aggregation_error,
    tau_hours,
    site_amplitude_prior,
    initial_site_amplitudes=None,
):
    """Build the same shared CO2 mean with the existing cached fixed-OU target."""
    prepared, prior, correction = _model_inputs(prepared, observations)
    cached = build_co2_cached_sigma_model(
        flux_sensitivity=prepared.flux_sensitivity,
        retained_prior=prior,
        prune_zero_flux_columns=False,
        fixed_prior_contribution=prepared.fixed_prior_contribution,
        boundary_correction=correction,
        observations=observations,
        observation_error=observation_error,
        aggregation_error=aggregation_error,
        tau_hours=tau_hours,
        site_amplitude_prior=site_amplitude_prior,
        initial_site_amplitudes=initial_site_amplitudes,
    )
    _diagnostics(cached.model, prepared)
    return cached


def annotate_shared_baseline_trace(trace, model):
    """Persist the complete frozen preparation beside sampled output.

    Call after the ordinary coordinate restoration and absolute-predictive
    restoration used by the CO2 recipe. This group owns centring, density
    widths, retained prior identity and any resolved #996 calibration.
    """
    prepared = model.shared_baseline
    if isinstance(trace, xr.DataTree):
        trace["shared_baseline"] = xr.DataTree(prepared)
    else:
        trace.add_groups({"shared_baseline": prepared})
    trace.attrs["shared_baseline_schema"] = SCHEMA
    return trace
