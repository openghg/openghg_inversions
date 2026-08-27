"""Readable scientific recipe for a standard single-sector RHIME inversion.

The public recipe deliberately spells out its complete execution order. Small
amounts of forwarding shared with the multisector recipe are retained so a
scientist can read or copy this module without reconstructing a pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import arviz as az
import pymc as pm
import xarray as xr

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.models.components import (
    add_linear_component,
    add_offset_component,
)
from openghg_inversions.models.coords import registered_model
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models.state_activity import StateActivity, prepare_linear_sensitivity
from openghg_inversions.observation_error import (
    AggregationError,
    aggregation_error_input_names,
    resolve_aggregation_error,
)
from openghg_inversions.sigma import SigmaAlignment

from ._model_building import (
    ForwardModelTerms,
    add_rhime_likelihood,
    builtin_model_build_result,
    validated_custom_model_build,
)
from .builders import (
    RhimeLikelihoodBuilder,
    RhimeModelBuilder,
    RhimeModelBuilderContext,
    RhimeModelBuildResult,
    callable_metadata,
    validate_model_build_result,
)
from .materialization import materialize_pymc_inputs
from .outputs import RhimeResult, make_standard_rhime_outputs
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
from .specs import (
    DEFAULT_BC_PRIOR,
    DEFAULT_OFFSET_PRIOR,
    DEFAULT_X_PRIOR,
    LikelihoodSettings,
    PollutionEventSettings,
    RhimeModelSpec,
    RhimeRunSpec,
)


_STANDARD_FLUX_INPUT_NAMES = ("H",)
_OBSERVATION_INPUT_NAMES = ("mf", "mf_error")
_BASELINE_INPUT_NAMES = ("H_bc",)


def _require_component_inputs(
    prepared: RhimePreparedInputs,
    names: tuple[str, ...],
    *,
    owner: str,
) -> None:
    """Fail before materialization with the selected component named."""
    missing = [name for name in names if name not in prepared.inv_inputs]
    if missing:
        raise ValueError(f"{owner} requires prepared input(s) {missing!r}.")


def standard_model_input_names(
    prepared: RhimePreparedInputs,
    model_spec: RhimeModelSpec,
) -> tuple[str, ...]:
    """Declare arrays required by selected standard-model components.

    Args:
        prepared: Backend-neutral prepared inputs.
        model_spec: Resolved standard-model component options.

    Returns:
        Prepared variable names selected for coordinated materialization.

    Raises:
        ValueError: If a selected component's required input is absent or its
            aggregation-error representation is ambiguous.
    """
    _require_component_inputs(
        prepared,
        _STANDARD_FLUX_INPUT_NAMES,
        owner="Standard flux component",
    )
    _require_component_inputs(
        prepared,
        _OBSERVATION_INPUT_NAMES,
        owner="Observation-error component",
    )
    names = [
        *_STANDARD_FLUX_INPUT_NAMES,
        *_OBSERVATION_INPUT_NAMES,
    ]
    likelihood_inputs = (
        () if model_spec.likelihood is None else model_spec.likelihood.required_prepared_inputs
    )
    if likelihood_inputs:
        _require_component_inputs(
            prepared,
            likelihood_inputs,
            owner="Selected built-in likelihood",
        )
        names.extend(likelihood_inputs)
    if model_spec.use_bc:
        _require_component_inputs(
            prepared,
            _BASELINE_INPUT_NAMES,
            owner="Standard baseline component selected by `use_bc=True`",
        )
        names.extend(_BASELINE_INPUT_NAMES)
    aggregation_names = aggregation_error_input_names(
        prepared.inv_inputs,
        model_spec.aggregation_error_mode,
    )
    _require_component_inputs(
        prepared,
        aggregation_names,
        owner=(
            "Aggregation-error component selected by "
            f"`aggregation_error_mode={model_spec.aggregation_error_mode!r}`"
        ),
    )
    names.extend(aggregation_names)
    return tuple(names)


def build_standard_rhime_model(
    flux_sensitivity: xr.DataArray,
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    minimum_error: xr.DataArray | None = None,
    likelihood_settings: LikelihoodSettings | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    boundary_sensitivity: xr.DataArray | None = None,
    x_prior: PriorArgs | None = None,
    bc_prior: PriorArgs | None = None,
    offset_prior: PriorArgs | None = None,
    add_offset: bool = False,
    use_bc: bool = True,
    offset_args: dict | None = None,
    state_activity: StateActivity | None = None,
    bc_state_activity: StateActivity | None = None,
    preserve_legacy_likelihood: bool = False,
    sigma_alignment: SigmaAlignment | None = None,
    legacy_unused_sigma_settings: PollutionEventSettings | None = None,
) -> pm.Model:
    """Build the concrete standard single-sector RHIME model.

    Args:
        flux_sensitivity: Labelled flux sensitivity matrix.
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        aggregation_error: Validated fixed aggregation-error representation.
        minimum_error: Optional prepared minimum total-error floor.
        likelihood_settings: Resolved built-in likelihood settings.
        likelihood_builder: Optional Python-only custom likelihood.
        likelihood_kwargs: Options for the custom likelihood.
        boundary_sensitivity: Optional labelled boundary sensitivity matrix.
        x_prior: Prior specification for flux scaling factors.
        bc_prior: Prior specification for boundary-condition scaling factors.
        offset_prior: Prior specification for optional offsets.
        add_offset: Whether to include an offset term.
        use_bc: Whether to include boundary-condition terms.
        offset_args: Extra keyword arguments for the offset component.
        state_activity: Optional active/fixed flux-state policy.
        bc_state_activity: Optional active/fixed boundary-state policy.
        preserve_legacy_likelihood: Whether to preserve ``run_hbmcmc``'s
            boundary-only pollution event and unused ``sigma`` variable.
        sigma_alignment: Optional precomputed mismatch alignment. Ordinary
            runners derive it from observations.
        legacy_unused_sigma_settings: Private ``run_hbmcmc`` settings for its
            historical disconnected sigma variable.

    Returns:
        Built PyMC model.

    Raises:
        KeyError: If required sensitivity inputs are absent.
        ValueError: If labels, state policies, priors, or canonical likelihood
            variables are invalid.
        TypeError: If the likelihood returns the wrong result type.
    """
    x_prior = dict(DEFAULT_X_PRIOR if x_prior is None else x_prior)
    bc_prior = dict(DEFAULT_BC_PRIOR if bc_prior is None else bc_prior)
    offset_prior = dict(DEFAULT_OFFSET_PRIOR if offset_prior is None else offset_prior)

    prepared_flux = prepare_linear_sensitivity(flux_sensitivity)
    prepared_boundary = (
        prepare_linear_sensitivity(boundary_sensitivity)
        if use_bc and boundary_sensitivity is not None
        else None
    )

    with registered_model() as model:
        flux_component = add_linear_component(
            prepared_flux,
            data_name="hx",
            prior_args=x_prior,
            var_name="x",
            output_name="mu",
            output_dim="nmeasure",
            compute_deterministic=True,
            state_activity=state_activity,
        )
        pollution_mean = flux_component.output

        boundary_mean = None
        if use_bc:
            if boundary_sensitivity is None:
                raise ValueError(
                    "Standard baseline component requires `boundary_sensitivity` when `use_bc` is true."
                )
            assert prepared_boundary is not None
            boundary_mean = add_linear_component(
                prepared_boundary,
                data_name="hbc",
                prior_args=bc_prior,
                var_name="bc",
                output_name="mu_bc",
                output_dim="nmeasure",
                compute_deterministic=True,
                state_activity=bc_state_activity,
            ).output

        offset = None
        if add_offset:
            offset = add_offset_component(
                observations,
                prior_args=offset_prior,
                output_name="offset",
                output_dim="nmeasure",
                **(offset_args or {}),
            )

        baseline_mean = boundary_mean
        if offset is not None:
            baseline_mean = offset if baseline_mean is None else baseline_mean + offset
        modelled_mean = pollution_mean if baseline_mean is None else pollution_mean + baseline_mean

        forward = ForwardModelTerms(
            total=modelled_mean,
            pollution=pollution_mean,
            baseline=baseline_mean,
        )
        add_rhime_likelihood(
            settings=likelihood_settings,
            likelihood_builder=likelihood_builder,
            likelihood_kwargs=likelihood_kwargs,
            forward=forward,
            observations=observations,
            observation_error=observation_error,
            aggregation_error=aggregation_error,
            minimum_error=minimum_error,
            output_dim="nmeasure",
            sigma_alignment=sigma_alignment,
            legacy_pollution_event_baseline=boundary_mean,
            preserve_legacy_pollution_event=preserve_legacy_likelihood,
            legacy_unused_sigma_settings=legacy_unused_sigma_settings,
        )
    return model


def build_standard_rhime_model_result(
    *,
    prepared: RhimePreparedInputs,
    model_inputs: xr.Dataset,
    run_spec: RhimeRunSpec,
    model_builder: RhimeModelBuilder | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    preserve_legacy_likelihood: bool = False,
    legacy_unused_sigma_settings: PollutionEventSettings | None = None,
) -> RhimeModelBuildResult:
    """Build the standard graph and describe its output roles.

    Args:
        prepared: Retained prepared-input artifact used by custom builders.
        model_inputs: Eager canonical arrays for the built-in PyMC graph.
        run_spec: Resolved model, sampling, and output specification.
        model_builder: Optional complete-model builder for advanced prepared-
            input workflows.
        likelihood_builder: Optional observation-error and distribution builder
            used with the built-in graph.
        likelihood_kwargs: Options expanded only into the custom likelihood.
        preserve_legacy_likelihood: Whether to preserve the historical
            ``run_hbmcmc`` likelihood graph and pollution-event definition.
        legacy_unused_sigma_settings: Private compatibility settings for a
            disconnected historical sigma variable.

    Returns:
        Model plus variable roles, supported outputs, and build metadata.

    Raises:
        ValueError: If both builder extension points are supplied or the built
            result is inconsistent with the run specification.
    """
    if model_builder is not None and likelihood_builder is not None:
        raise ValueError("Pass either `model_builder` or `likelihood_builder`, not both.")
    if likelihood_kwargs and likelihood_builder is None:
        raise ValueError("Non-empty `likelihood_kwargs` require an active `likelihood_builder`.")
    timing_start = timer_start()
    if model_builder is not None:
        builder_context = RhimeModelBuilderContext(
            prepared_inputs=prepared,
            run_spec=run_spec,
            multisector=False,
        )
        result = validated_custom_model_build(model_builder, context=builder_context)
        validate_model_build_result(result, context=builder_context)
    else:
        model_spec = run_spec.model
        if len(model_spec.sectors) != 1:
            raise ValueError("Standard RHIME model specs must include exactly one sector.")
        sector = model_spec.sectors[0]
        if likelihood_builder is not None and model_spec.likelihood is not None:
            raise ValueError("A custom likelihood cannot be combined with built-in likelihood settings.")
        aggregation_error = resolve_aggregation_error(
            model_inputs,
            model_spec.aggregation_error_mode,
        )
        state_activity = (
            sector.state_activity if sector.state_activity is not None else model_spec.state_activity
        )
        model = build_standard_rhime_model(
            model_inputs["H"],
            observations=model_inputs["mf"],
            observation_error=model_inputs["mf_error"],
            aggregation_error=aggregation_error,
            minimum_error=model_inputs.get("min_error"),
            likelihood_settings=model_spec.likelihood,
            likelihood_builder=likelihood_builder,
            likelihood_kwargs=likelihood_kwargs,
            boundary_sensitivity=model_inputs.get("H_bc"),
            x_prior=dict(sector.x_prior),
            state_activity=state_activity,
            bc_prior=model_spec.bc_prior,
            bc_state_activity=model_spec.bc_state_activity,
            offset_prior=model_spec.offset_prior,
            add_offset=model_spec.add_offset,
            use_bc=model_spec.use_bc,
            offset_args=model_spec.offset_args,
            preserve_legacy_likelihood=preserve_legacy_likelihood,
            legacy_unused_sigma_settings=legacy_unused_sigma_settings,
        )
        result = builtin_model_build_result(
            model,
            model_spec=run_spec.model,
            multisector=False,
            input_names=tuple(str(name) for name in model_inputs.data_vars),
            preserve_legacy_baseline=preserve_legacy_likelihood,
        )
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
    likelihood_kwargs: Mapping[str, Any] | None = None,
    _compatibility_likelihood_provenance: Mapping[str, Any] | None = None,
) -> RhimeResult:
    """Construct a sampled standard result before output side effects.

    Args:
        prepared: Retained canonical inputs and basis functions.
        run_spec: Resolved model, output, and run settings.
        sampler: Sampler configuration used for the trace.
        model_build_result: Concrete graph and semantic variable roles.
        idata: Sampled posterior and predictive groups.
        build_and_sample_seconds: Combined graph-build and sampling duration.
        model_builder: Optional complete-model callable used for provenance.
        likelihood_builder: Optional likelihood callable used for provenance.
        likelihood_kwargs: Serializable options owned by the likelihood.
        _compatibility_likelihood_provenance: Pre-resolved private compatibility provenance.

    Returns:
        Standard-run result ready for requested output construction.
    """
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
    if model_builder is not None:
        identity = callable_metadata(model_builder)
        result.output_metadata["model_builder"] = identity
    if likelihood_builder is not None:
        identity = callable_metadata(likelihood_builder)
        result.output_metadata["likelihood_builder"] = identity
    if likelihood_kwargs is not None:
        result.output_metadata["likelihood_kwargs"] = likelihood_kwargs
    if _compatibility_likelihood_provenance is not None:
        result.output_metadata.update(dict(_compatibility_likelihood_provenance))
    return result


def run_rhime(
    *,
    config_file: str | Path | None = None,
    merged_data: RhimeMergedData | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    preserve_legacy_likelihood: bool = False,
    _compatibility_likelihood_provenance: Mapping[str, Any] | None = None,
    _compatibility_unused_sigma_settings: PollutionEventSettings | None = None,
    **kwargs: Any,
) -> RhimeResult:
    """Run a standard single-sector RHIME inversion.

    The visible process is resolve → retrieve/reload → filter → basis →
    sensitivities → assemble → materialize → build → sample → result →
    requested outputs.

    Args:
        config_file: Optional INI configuration file. Values in ``kwargs``
            override values read from this file.
        merged_data: Optional externally supplied merged scientific data.
            Passing this borrowed handoff bypasses OpenGHG acquisition and
            merged-cache I/O, then resumes at the visible filtering stage.
            The retrieval stage checks its sector layout without mutating it.
        likelihood_builder: Optional Python-only callable invoked with a
            completed forward-model mean and explicit error-model inputs in
            the active PyMC model. It must return the canonical observed
            variable ``y`` and create the canonical error scale ``epsilon``.
            The callable is never read from configuration or stored in
            run/model specifications.
        likelihood_kwargs: Options specific to the custom likelihood. Common
            scientific arrays are passed explicitly by the recipe.
        preserve_legacy_likelihood: Private ``run_hbmcmc`` compatibility
            switch. Ordinary RHIME callers should leave it false.
        _compatibility_likelihood_provenance: Private ``run_hbmcmc`` record of
            the historical additive callback spelling and options.
        _compatibility_unused_sigma_settings: Private ``run_hbmcmc`` settings
            for its historical disconnected sigma variable.
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
            variables or requested-output compatibility are invalid.

    """
    if likelihood_kwargs and likelihood_builder is None:
        raise ValueError("Non-empty `likelihood_kwargs` require an active `likelihood_builder`.")
    if _compatibility_likelihood_provenance is not None and likelihood_builder is not None:
        raise ValueError("Compatibility likelihood provenance cannot accompany a custom likelihood builder.")
    params = (
        params_from_config(config_file, extra_kwargs=kwargs, normalise=False)
        if config_file is not None
        else dict(kwargs)
    )
    if likelihood_builder is not None:
        if params.get("mismatch_model") is not None:
            raise ValueError("A custom likelihood cannot be combined with a built-in mismatch model.")
        params["mismatch_model"] = None
    setup = resolve_rhime_options(params=params, multisector=False)
    if likelihood_builder is None and setup.run_spec.model.likelihood is None:
        raise ValueError("A standard RHIME run requires a built-in or custom likelihood.")

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
        variable_names=standard_model_input_names(
            prepared,
            run_spec.model,
        ),
    )
    build_and_sample_start = timer_start()
    model_build_result = build_standard_rhime_model_result(
        prepared=prepared,
        model_inputs=model_inputs,
        run_spec=run_spec,
        likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs,
        preserve_legacy_likelihood=preserve_legacy_likelihood,
        legacy_unused_sigma_settings=_compatibility_unused_sigma_settings,
    )
    idata = sample_rhime_model(
        model_build_result,
        setup.sampler,
    )
    result = make_standard_rhime_result(
        prepared=prepared,
        run_spec=run_spec,
        sampler=setup.sampler,
        model_build_result=model_build_result,
        idata=idata,
        build_and_sample_seconds=timer_seconds(build_and_sample_start),
        likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs,
        _compatibility_likelihood_provenance=_compatibility_likelihood_provenance,
    )
    output_start = timer_start()
    make_standard_rhime_outputs(result=result, prepared=prepared)
    log_timing(
        "rhime.output_total",
        timer_seconds(output_start),
        multisector=False,
        output_format=run_spec.output.output_format,
    )
    return result
