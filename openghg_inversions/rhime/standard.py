"""Readable scientific recipe for a standard single-sector RHIME inversion.

The public recipe deliberately spells out its complete execution order. Small
amounts of forwarding shared with the multisector recipe are retained so a
scientist can read or copy this module without reconstructing a pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import arviz as az
import pymc as pm
import xarray as xr

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.models.components import add_state_linear_component
from openghg_inversions.models.coords import CoordRegistry, attach_coord_registry
from openghg_inversions.models.rhime import (
    _LIKELIHOOD_RESULT_ATTR,
    _add_rhime_observation_components,
    _prepare_builder_priors,
    RhimeModelSpec,
)
from openghg_inversions.models.rhime_likelihood import RhimeLikelihoodBuilder
from openghg_inversions.models.state_activity import StateActivity
from openghg_inversions.observation_error import AggregationErrorMode
from openghg_inversions.sigma import SigmaAlignment

from ._model_building import (
    builtin_model_build_result,
    validate_likelihood_builder,
    validated_custom_model_build,
)
from .builders import (
    RhimeModelBuilder,
    RhimeModelBuilderContext,
    RhimeModelBuildResult,
    callable_metadata,
    validate_model_build_result,
)
from .materialization import materialize_pymc_inputs
from .outputs import RhimeResult, apply_output_bundle, make_standard_output_bundle
from .params import params_from_config, resolve_rhime_options
from .preparation import (
    assemble_rhime_inputs,
    build_rhime_basis,
    build_rhime_sensitivities,
    filter_rhime_observations,
    retrieve_or_reload_rhime_data,
    with_prepared_rhime_sites,
)
from .sampling import RhimeSampler, sample_rhime_model
from .specs import RhimeRunSpec


def build_rhime_model(
    inv_inputs: xr.Dataset,
    *,
    sigma_alignment: SigmaAlignment,
    x_prior: dict | None = None,
    bc_prior: dict | None = None,
    sigma_prior: dict | None = None,
    offset_prior: dict | None = None,
    add_offset: bool = False,
    use_bc: bool = True,
    pollution_events_from_obs: bool = False,
    no_model_error: bool = False,
    aggregation_error_mode: AggregationErrorMode = "auto",
    offset_args: dict | None = None,
    power: dict | float = 1.99,
    state_activity: StateActivity | None = None,
    bc_state_activity: StateActivity | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> pm.Model:
    """Build the concrete standard single-sector RHIME model.

    The recipe constructs the flux contribution first, then adds the shared
    baseline, offset, model-data mismatch, aggregation-error, and likelihood
    components in scientific order.
    """
    x_prior, bc_prior, sigma_prior, offset_prior = _prepare_builder_priors(
        x_prior=x_prior,
        bc_prior=bc_prior,
        sigma_prior=sigma_prior,
        offset_prior=offset_prior,
    )

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        flux_component = add_state_linear_component(
            inv_inputs["H"],
            data_name="hx",
            prior_args=x_prior,
            var_name="x",
            output_name="mu",
            output_dim="nmeasure",
            compute_deterministic=True,
            state_activity=state_activity,
        )
        likelihood_result = _add_rhime_observation_components(
            inv_inputs,
            mu=flux_component.output,
            sigma_alignment=sigma_alignment,
            bc_prior=bc_prior,
            sigma_prior=sigma_prior,
            offset_prior=offset_prior,
            add_offset=add_offset,
            use_bc=use_bc,
            bc_state_activity=bc_state_activity,
            pollution_events_from_obs=pollution_events_from_obs,
            no_model_error=no_model_error,
            aggregation_error_mode=aggregation_error_mode,
            offset_args=offset_args,
            power=power,
            likelihood_builder=likelihood_builder,
        )
        setattr(model, _LIKELIHOOD_RESULT_ATTR, likelihood_result)

    return model


def build_rhime_model_from_spec(
    inv_inputs: xr.Dataset,
    model_spec: RhimeModelSpec,
    *,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> pm.Model:
    """Build the standard model from a normalized RHIME model specification."""
    if len(model_spec.sectors) != 1:
        raise ValueError("Standard RHIME model specs must include exactly one sector.")

    sector = model_spec.sectors[0]
    sigma_alignment = SigmaAlignment.from_frequency(
        inv_inputs["site_indicator"],
        frequency=model_spec.sigma_freq,
        per_site=model_spec.sigma_per_site,
        anchor_time=model_spec.sigma_freq_anchor,
    )
    state_activity = model_spec.state_activity
    if model_spec.sector_state_activities is not None:
        state_activity = model_spec.sector_state_activities.get(sector.name, state_activity)
    if sector.state_activity is not None:
        state_activity = sector.state_activity
    return build_rhime_model(
        inv_inputs,
        sigma_alignment=sigma_alignment,
        x_prior=dict(sector.x_prior),
        state_activity=state_activity,
        bc_prior=model_spec.bc_prior,
        bc_state_activity=model_spec.bc_state_activity,
        sigma_prior=model_spec.sigma_prior,
        offset_prior=model_spec.offset_prior,
        add_offset=model_spec.add_offset,
        use_bc=model_spec.use_bc,
        pollution_events_from_obs=model_spec.pollution_events_from_obs,
        no_model_error=model_spec.no_model_error,
        aggregation_error_mode=model_spec.aggregation_error_mode,
        offset_args=model_spec.offset_args,
        power=model_spec.power,
        likelihood_builder=likelihood_builder,
    )


def build_standard_rhime_model(
    *,
    prepared: RhimePreparedInputs,
    model_inputs: xr.Dataset,
    run_spec: RhimeRunSpec,
    model_builder: RhimeModelBuilder | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> RhimeModelBuildResult:
    """Build the concrete single-sector graph and describe its output roles."""
    if model_builder is not None and likelihood_builder is not None:
        raise ValueError("Pass either `model_builder` or `likelihood_builder`, not both.")
    timing_start = timer_start()
    builder_context = RhimeModelBuilderContext(
        prepared_inputs=prepared,
        run_spec=run_spec,
        multisector=False,
    )
    if model_builder is not None:
        result = validated_custom_model_build(model_builder, context=builder_context)
    else:
        model = build_rhime_model_from_spec(
            model_inputs,
            run_spec.model,
            **({} if likelihood_builder is None else {"likelihood_builder": likelihood_builder}),
        )
        result = builtin_model_build_result(model, model_spec=run_spec.model, multisector=False)
    if likelihood_builder is not None:
        validate_model_build_result(result, context=builder_context, builder_kind="likelihood")
    log_timing("rhime.model_build", timer_seconds(timing_start), multisector=False)
    return result


def make_standard_rhime_result(
    *,
    prepared: RhimePreparedInputs,
    run_spec: RhimeRunSpec,
    sampler: RhimeSampler,
    model_build_result: RhimeModelBuildResult,
    idata: az.InferenceData,
    build_and_sample_seconds: float,
    model_builder: RhimeModelBuilder | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> RhimeResult:
    """Construct a standard result and write only its requested products."""
    result = RhimeResult(
        run_spec=run_spec,
        model_spec=run_spec.model,
        output_spec=run_spec.output,
        inv_inputs=prepared.inv_inputs,
        idata=idata,
        sampler=sampler,
        model=model_build_result.model,
        basis_functions=prepared.basis_functions,
        model_build_result=model_build_result,
        output_metadata={"build_and_sample_seconds": build_and_sample_seconds},
    )
    builder_metadata = dict(model_build_result.metadata)
    if model_builder is not None:
        identity = callable_metadata(model_builder)
        result.output_metadata["model_builder"] = identity
        builder_metadata["model_builder"] = identity
    if likelihood_builder is not None:
        identity = callable_metadata(likelihood_builder)
        result.output_metadata["likelihood_builder"] = identity
        builder_metadata["likelihood_builder"] = identity

    timing_start = timer_start()
    output_bundle = make_standard_output_bundle(
        output_spec=run_spec.output,
        run_spec=run_spec,
        model_spec=run_spec.model,
        idata=idata,
        prepared=prepared,
        country_file=run_spec.output.country_file,
        sampler=sampler,
        variable_roles=model_build_result.variable_roles,
        builder_metadata=builder_metadata,
    )
    log_timing(
        "rhime.output_bundle_total",
        timer_seconds(timing_start),
        multisector=False,
        output_format=run_spec.output.output_format,
    )
    apply_output_bundle(result, output_bundle)
    return result


def run_rhime(
    *,
    config_file: str | Path | None = None,
    merged_data: RhimeMergedData | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    **kwargs: Any,
) -> RhimeResult:
    """Run a standard single-sector RHIME inversion.

    The visible process is resolve → retrieve/reload → filter → basis →
    sensitivities → assemble → materialize → build → sample → result/output.

    Args:
        config_file: Optional INI configuration file. Values in ``kwargs``
            override values read from this file.
        merged_data: Optional externally supplied merged scientific data.
            Passing this borrowed handoff bypasses OpenGHG acquisition and
            merged-cache I/O, then resumes at the visible filtering stage.
            The retrieval stage checks its sector layout without mutating it.
        likelihood_builder: Optional Python-only callable invoked with a
            ``RhimeLikelihoodContext`` in the active PyMC model and returning
            ``RhimeLikelihoodResult``. The result declares semantic variable
            roles, supported output formats, and JSON-compatible metadata;
            roles drive predictive selection and output compatibility is
            validated before sampling. The callable is never read from
            configuration or stored in run/model specifications.
        **kwargs: RHIME run parameters using snake-case names, such as
            ``output_path``, ``output_name``, ``flux_sources``, and
            ``x_prior``. ``species`` names the primary gas or tracer used for
            object-store lookup and output naming. ``flux_sources`` contains
            OpenGHG flux ``source`` values. Legacy ``emissions_name`` is
            accepted only as a compatibility alias when ``flux_sources`` is
            absent.

    Returns:
        Modern RHIME result containing canonical inputs, InferenceData, specs,
        output metadata, and generated outputs.

    Raises:
        TypeError: If a likelihood builder is not callable or returns the
            wrong result type.
        ValueError: If required parameters are missing, unsupported parameters
            are supplied, the flux-source count is invalid, or likelihood
            roles, metadata, or requested-output compatibility are invalid.

    Notes:
        A non-callable likelihood builder is rejected before configuration is
        parsed or data is acquired, prepared, or materialized.
    """
    validate_likelihood_builder(likelihood_builder)
    params = (
        params_from_config(config_file, extra_kwargs=kwargs, normalise=False)
        if config_file is not None
        else dict(kwargs)
    )
    setup = resolve_rhime_options(params=params, multisector=False)

    preparation_start = timer_start()
    merged = retrieve_or_reload_rhime_data(
        setup.data_args,
        multisector=False,
        merged_data=merged_data,
    )
    filtered = filter_rhime_observations(merged, setup.data_args)
    basis_functions = build_rhime_basis(filtered, setup.data_args)
    site_data = build_rhime_sensitivities(
        filtered,
        basis_functions,
        setup.data_args,
        multisector=False,
    )
    prepared = assemble_rhime_inputs(filtered, basis_functions, site_data, setup.data_args)
    log_timing(
        "rhime.prepare_inputs",
        timer_seconds(preparation_start),
        multisector=False,
        nmeasure=prepared.inv_inputs.sizes.get("nmeasure"),
        sites=len(prepared.sites),
        regions=prepared.inv_inputs.sizes.get("region"),
        sources=prepared.inv_inputs.sizes.get("source"),
        basis_source=prepared.basis_artifact_source,
    )
    run_spec = with_prepared_rhime_sites(setup.run_spec, prepared)

    model_inputs = materialize_pymc_inputs(
        prepared,
        aggregation_error_mode=run_spec.model.aggregation_error_mode,
    )
    build_and_sample_start = timer_start()
    model_build_result = build_standard_rhime_model(
        prepared=prepared,
        model_inputs=model_inputs,
        run_spec=run_spec,
        likelihood_builder=likelihood_builder,
    )
    idata = sample_rhime_model(
        model_build_result,
        setup.sampler,
        use_variable_roles=likelihood_builder is not None,
    )
    return make_standard_rhime_result(
        prepared=prepared,
        run_spec=run_spec,
        sampler=setup.sampler,
        model_build_result=model_build_result,
        idata=idata,
        build_and_sample_seconds=timer_seconds(build_and_sample_start),
        likelihood_builder=likelihood_builder,
    )
