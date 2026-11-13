"""Configured execution of the frozen linked atmospheric baseline recipe."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import xarray as xr

from openghg_inversions.models.boundary import require_centered_precision

from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.rhime.builders import RhimeModelBuildResult
from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model
from .co2_cached_sigma_runner import _sampler_for_cached_graph
from .co2_o2_runner import (
    _CO2_O2_VARIABLE_ROLES, _co2_o2_metadata,
    annotate_co2_o2_trace,
)
from .shared_preparation import materialize_linked_shared_inputs
from .linked_shared_baseline_model import (
    annotate_linked_shared_baseline_trace, build_shared_co2_o2_model,
    build_shared_co2_o2_cached_sigma_model,
)


from .co2_cached_sigma_model import Co2CachedSigmaModel
from .co2_o2_preparation import Co2O2PreparedInputs


def build_linked_shared_workflow(
    prepared: Co2O2PreparedInputs, *, independent_error_sd: xr.DataArray | None = None,
    tau_hours: float | Mapping[str, float] | None = None,
    fixed_site_amplitudes: float | Mapping[str, float] | None = None,
    site_amplitude_prior: Mapping[str, Any] | None = None,
    initial_site_amplitudes: float | Mapping[str, float] | None = None,
) -> tuple[RhimeModelBuildResult, Co2CachedSigmaModel | None, Co2O2PreparedInputs]:
    """Resolve frozen science once and select the existing ordinary or cached recipe.

    The returned materialized preparation must also be used for annotation.
    Inferred fixed-OU amplitudes select the maintained matched cache target.

    Args:
        prepared: Linked handoff containing frozen native shared baselines.
        independent_error_sd: Error SDs on the joint observation axis, or None
            to use saved errors. Explicit and saved errors must agree in labels,
            observation units and values when both are supplied.
        tau_hours: Fixed OU timescales, scalar or keyed by species/site. None
            selects independent Gaussian errors.
        fixed_site_amplitudes: Fixed OU amplitudes, scalar or keyed by species/site.
            Mutually exclusive with site_amplitude_prior.
        site_amplitude_prior: Prior for inferred OU amplitudes; selects the
            matched cached recipe and requires tau_hours.
        initial_site_amplitudes: Initial inferred OU amplitudes, scalar or keyed
            by species/site. Requires site_amplitude_prior.

    Returns:
        tuple[openghg_inversions.rhime.builders.RhimeModelBuildResult, openghg_inversions.rhime.co2.co2_cached_sigma_model.Co2CachedSigmaModel | None, openghg_inversions.rhime.co2.co2_o2_preparation.Co2O2PreparedInputs]:
            Concrete build with scientific output roles, optional cached sampling
            target, and the eager preparation reused for annotation and saving.
    """
    require_centered_precision()
    if not prepared.shared_baseline:
        raise ValueError("Linked shared execution requires frozen shared baseline inputs.")
    if tau_hours is None and (site_amplitude_prior is not None or fixed_site_amplitudes is not None):
        raise ValueError("OU amplitudes require fixed tau_hours.")
    if site_amplitude_prior is not None and fixed_site_amplitudes is not None:
        raise ValueError("Select fixed or inferred site amplitudes, not both.")
    if site_amplitude_prior is None and initial_site_amplitudes is not None:
        raise ValueError("initial_site_amplitudes is unused without inferred OU amplitudes.")
    prepared, error, likelihood = materialize_linked_shared_inputs(
        prepared, independent_error_sd=independent_error_sd,
        likelihood=dict(tau_hours=tau_hours, fixed_site_amplitudes=fixed_site_amplitudes,
                        site_amplitude_prior=site_amplitude_prior, initial_site_amplitudes=initial_site_amplitudes),
    )
    observations = prepared.observations
    cached = None
    if site_amplitude_prior is not None:
        cached = build_shared_co2_o2_cached_sigma_model(
            prepared.shared_baseline, observations=observations, independent_error_sd=error,
            aggregation_error=prepared.aggregation_error, tau_hours=likelihood["tau_hours"],
            site_amplitude_prior=likelihood["site_amplitude_prior"],
            initial_site_amplitudes=likelihood["initial_site_amplitudes"],
        )
        model = cached.model
    else:
        model = build_shared_co2_o2_model(
            prepared.shared_baseline, observations=observations, independent_error_sd=error,
            aggregation_error=prepared.aggregation_error, tau_hours=likelihood["tau_hours"],
            fixed_site_amplitudes=likelihood["fixed_site_amplitudes"],
        )
    roles = dict(_CO2_O2_VARIABLE_ROLES)
    for channel in ("co2", "o2"):
        roles.update(prepared.shared_baseline[channel].outputs(prefix=f"{channel}_").roles)
        roles[f"{channel}_atmospheric_level"] = f"{channel}_bc_mean_shift"
        if prepared.shared_baseline[channel].data.sizes["bc_pattern_term"]:
            roles[f"{channel}_boundary_pattern_coefficient"] = f"{channel}_bc_pattern_standardized"
    if likelihood.get("tau_hours") is not None:
        roles.update(independent_error="error", fixed_ou_timescale="ou_tau_hours",
                     fixed_ou_site_amplitude="ou_site_amplitude",
                     observation_to_fixed_ou_site_index="ou_site_index")
    metadata = _co2_o2_metadata(prepared, observations=observations)
    metadata.update(recipe="co2_o2_shared_baseline", outer_mode=prepared.shared_baseline["co2"].data.attrs["mode"],
                    baseline="independent physical CO2/O2 atmospheric deltas; frozen centered outer and BC patterns")
    if likelihood.get("tau_hours") is not None:
        metadata["likelihood"] = "joint Gaussian with fixed OU timescales by species/site"
        metadata["site_amplitudes"] = "inferred with matched cache" if cached is not None else "fixed"
    built = RhimeModelBuildResult(model, OutputContract(variable_roles=roles, metadata=metadata))
    return built, cached, prepared


def annotate_linked_shared_workflow(
    trace: xr.DataTree, *, built: RhimeModelBuildResult, prepared: Co2O2PreparedInputs,
) -> xr.DataTree:
    """Restore normal linked labels/predictions and persist complete frozen context.

    Args:
        trace: Sampled linked trace with restored observation labels.
        built:
            Concrete graph and its scientific output roles.
        prepared:
            Materialized handoff returned by build_linked_shared_workflow,
            containing the frozen native shared baselines.

    Returns:
        xarray.DataTree: Annotated trace carrying absolute predictions, scientific
            roles and frozen channel baselines for later reconstruction.
    """
    trace = annotate_co2_o2_trace(trace, built=built)
    trace = annotate_linked_shared_baseline_trace(trace, prepared.shared_baseline)
    trace.attrs["rhime_recipe"] = "co2_o2_shared_baseline"
    for name, group in trace.children.items():
        if not name.startswith("shared_baseline_"):
            group.attrs["rhime_recipe"] = "co2_o2_shared_baseline"
    return trace


def sample_linked_shared_workflow(
    prepared: Co2O2PreparedInputs, *, sampler: RhimeSampler, **model_options: Any,
) -> xr.DataTree:
    """Sample with maintained compact deterministic reconstruction and OU cache.

    Args:
        prepared:
            Linked handoff with already frozen native shared baselines.
        sampler:
            Resolved sampling controls; cached OU selects its maintained step order.
        **model_options:
            Explicit linked likelihood options accepted by the workflow builder.

    Returns:
        xarray.DataTree: Annotated trace with restored labels, absolute predictives,
            selected sampled states and frozen preparations. Physical diagnostics
            are reconstructed later.
    """
    if model_options.get("tau_hours") is not None and sampler.nuts_sampler != "pymc":
        raise ValueError("Linked fixed-OU requires nuts_sampler='pymc'.")
    built, cached, prepared = build_linked_shared_workflow(prepared, **model_options)
    active = sampler
    if cached is not None:
        cached_sampler = copy.copy(sampler)
        cached_sampler.sample_kwargs = dict(sampler.sample_kwargs or {})
        target_accept = cached_sampler.sample_kwargs.pop("target_accept", .95)
        active = _sampler_for_cached_graph(
            cached_sampler, cached_model=cached,
            sigma_target_accept=target_accept, state_target_accept=target_accept,
        )
    trace = sample_rhime_model(built, active, recompute_deterministics=True)
    return annotate_linked_shared_workflow(trace, built=built, prepared=prepared)
