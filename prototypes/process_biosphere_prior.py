"""Small process-prior experiment over a complete labelled biosphere state.

GPP, autotrophic respiration (Ra), and heterotrophic respiration (Rh) have
positive, mean-one multipliers. Signs belong to the supplied sensitivity.
The linked prior correlates log multipliers for GPP/Ra in matching groups;
Rh and different groups are conditionally independent. The hierarchy shares
one HalfNormal log-scale across every coefficient, with the same fixed
conditional correlation. It pools width, not process means or correlations.

Inputs are already materialized and use ppm for concentration quantities.
There is no unresolved state or aggregation error in this prototype.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.models import (
    CorrelatedLognormalPrior,
    add_coherent_affine_component,
    add_coords,
    add_correlated_lognormal_state,
    apply_linear_sensitivity,
    prepare_linear_sensitivity,
    registered_model,
)
from openghg_inversions.models.additive_sigma import add_additive_sigma_likelihood
from openghg_inversions.observation_error import resolve_aggregation_error


BASE_LOG_SD = float(np.sqrt(np.log1p(0.2**2)))
MODES = ("independent", "linked", "hierarchical")


def _log_correlation(sensitivity: xr.DataArray, correlation: float) -> np.ndarray:
    """Validate labelled process controls and pair GPP/Ra by group identity."""
    if set(sensitivity.dims) != {"nmeasure", "flux_state"}:
        raise ValueError("Sensitivity must have dimensions nmeasure and flux_state.")
    if not sensitivity.sizes["flux_state"] or not sensitivity.sizes["nmeasure"]:
        raise ValueError("Sensitivity must contain observations and process states.")
    if "flux_state" not in sensitivity.indexes or not sensitivity.indexes["flux_state"].is_unique:
        raise ValueError("flux_state must have unique indexed labels.")
    for name in ("source", "group"):
        if name not in sensitivity.coords or sensitivity[name].dims != ("flux_state",):
            raise ValueError(f"{name} must be a coordinate over flux_state.")
        if pd.isna(sensitivity[name].values).any():
            raise ValueError(f"{name} labels must not be missing.")
    source, group = sensitivity.source.values, sensitivity.group.values
    if not set(source).issubset({"GPP", "Ra", "Rh"}):
        raise ValueError("source labels must be GPP, Ra, or Rh; TER cannot substitute for Ra or Rh.")
    pairs = pd.MultiIndex.from_arrays([source, group])
    if not pairs.is_unique:
        raise ValueError("Each (source, group) pair must be unique.")
    if not np.isfinite(correlation) or not 0 <= correlation < 1:
        raise ValueError("log_correlation must be finite and in [0, 1).")
    result = np.eye(len(source))
    positions = dict(zip(pairs, range(len(source))))
    for (name, label), i in positions.items():
        j = positions.get(("Ra", label))
        if name == "GPP" and j is not None:
            result[i, j] = result[j, i] = correlation
    return result


def _resolve_hyper_scale(log_sd: float, hyper_scale: float | None) -> float:
    """Match multiplier variance by default, with finite second moments."""
    if not np.isfinite(log_sd) or log_sd <= 0:
        raise ValueError("log_sd must be finite and positive.")
    if hyper_scale is None:
        hyper_scale = float(np.sqrt(-np.expm1(-2 * log_sd**2) / 2))
    if not np.isfinite(hyper_scale) or not 0 < hyper_scale < 1 / np.sqrt(2):
        raise ValueError("hyper_scale must be positive and below 1/sqrt(2) for finite prior variance.")
    return hyper_scale


def process_prior_covariance(
    flux_sensitivity: xr.DataArray,
    *,
    mode: str = "linked",
    log_sd: float = BASE_LOG_SD,
    log_correlation: float = 0.6,
    hyper_scale: float | None = None,
) -> np.ndarray:
    """Return arithmetic multiplier covariance in the supplied state order.

    ``log_correlation`` is a latent correlation, not an arithmetic correlation.
    In the hierarchy, conditional variance is exp(s**2)-1, where s is
    HalfNormal(hyper_scale). Its marginal covariance is
    (1 - 2 * hyper_scale**2 * R)**(-1/2) - 1, elementwise. The default
    hyper-scale makes every marginal multiplier SD equal to that of log_sd.
    Different groups have zero covariance but share higher-order dependence.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}.")
    scale = _resolve_hyper_scale(log_sd, hyper_scale)
    correlation = _log_correlation(flux_sensitivity, log_correlation)
    if mode == "independent":
        correlation = np.eye(len(correlation))
    if mode == "hierarchical":
        return np.expm1(-0.5 * np.log1p(-2 * scale**2 * correlation))
    return np.expm1(log_sd**2 * correlation)


def build_process_biosphere_model(
    flux_sensitivity: xr.DataArray,
    *,
    observations: xr.DataArray,
    fixed_prior_contribution: xr.DataArray,
    mode: str = "independent",
    log_sd: float = BASE_LOG_SD,
    log_correlation: float = 0.6,
    hyper_scale: float | None = None,
    observation_sd: float = 1.0,
) -> tuple[pm.Model, xr.Dataset]:
    """Build one process-prior arm and return its labelled state activity.

    ``fixed_prior_contribution`` is an affine intercept, not a complete prior
    prediction. All supplied states must have nonzero measurement sensitivity:
    removing invisible correlated states would require conditional prior
    reconstruction to preserve uncertainty in full-domain flux aggregates.
    ``observation_sd`` is the fixed total additive Gaussian SD in ppm.
    """
    covariance = process_prior_covariance(
        flux_sensitivity, mode=mode, log_sd=log_sd,
        log_correlation=log_correlation, hyper_scale=hyper_scale,
    )
    if not np.isfinite(observation_sd) or observation_sd <= 0:
        raise ValueError("observation_sd must be finite and positive.")
    prepared = prepare_linear_sensitivity(flux_sensitivity, output_dim="nmeasure")
    if prepared.removed.any().item():
        raise ValueError(
            "Exact-zero sensitivity columns are unsupported: conditional prior reconstruction "
            "is required before removing states from full-domain flux uncertainty."
        )
    mean = xr.ones_like(prepared.removed, dtype=float).rename("prior_mean")
    activity = xr.Dataset({
        "active": ~prepared.removed,
        "fixed_value": mean,
        "structurally_removed": prepared.removed,
    })
    with registered_model() as model:
        if mode == "hierarchical":
            add_coords(mean.coords, model_dims=("flux_state",))
            scale = pm.HalfNormal("shared_log_sd", sigma=_resolve_hyper_scale(log_sd, hyper_scale))
            latent = pm.Normal("flux_scaling_latent", 0.0, 1.0, dims="flux_state")
            correlation = _log_correlation(flux_sensitivity, log_correlation)
            state = pm.Deterministic(
                "flux_scaling",
                pt.exp(-0.5 * scale**2 + scale * pt.dot(np.linalg.cholesky(correlation), latent)),
                dims="flux_state",
            )
        else:
            state = add_correlated_lognormal_state(
                CorrelatedLognormalPrior(mean, covariance), var_name="flux_scaling"
            ).state
        signal = apply_linear_sensitivity(
            prepared, state, data_name="co2_sensitivity", output_name="scaled_flux_contribution",
        )
        prediction = add_coherent_affine_component(
            fixed_prior_contribution.rename("fixed_prior_contribution"), signal,
            output_name="modelled_concentration",
        )
        add_additive_sigma_likelihood(
            observations=observations,
            observation_error=xr.full_like(observations, observation_sd, dtype=float),
            aggregation_error=resolve_aggregation_error(observations.to_dataset(name="mf"), "none"),
            mean=prediction,
        )
    return model, activity
