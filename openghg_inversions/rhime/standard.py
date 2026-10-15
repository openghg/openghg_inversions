"""Readable scientific recipe for a standard single-sector RHIME inversion.

The public recipe deliberately spells out its complete execution order. Small
amounts of forwarding shared with the multisector recipe are retained so a
scientist can read or copy this module without reconstructing a pipeline.
"""

from __future__ import annotations

from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.basis.support import load_bc_basis
from openghg_inversions.inference.sampling import resolve_sampler
from openghg_inversions.basis import prepare_basis
from openghg_inversions._provenance import installed_ogi_provenance
from openghg_inversions.rhime.preparation import acquire_merged_data, export_basis, resolve_basis_inputs

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pymc as pm
import xarray as xr

from openghg_inversions._timing import log_timing, timed, timer_seconds, timer_start
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.models.components import (
    add_linear_component,
    add_offset_component,
)
from openghg_inversions.models.coords import registered_model
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.postprocessing.contracts import OutputContract
from openghg_inversions.models.state_activity import StateActivity, prepare_linear_sensitivity
from openghg_inversions.observation_error import (
    AggregationError,
    aggregation_error_input_names,
    resolve_aggregation_error,
)
from openghg_inversions.sigma import SigmaAlignment
from openghg_inversions.model_error import MinErrorConfig

from ._model_building import (
    ForwardModelTerms,
    add_rhime_likelihood,
    builtin_model_build_result,
)
from .builders import (
    RhimeLikelihoodBuilder,
    RhimeModelBuildResult,
    callable_metadata,
)

from .materialization import (
    _materialize_pymc_inputs_and_priors,
    materialize_pymc_inputs as materialize_pymc_inputs,
)
from .outputs import (
    RhimeResult,
    annotate_likelihood_trace,
    make_standard_rhime_outputs,
    validate_output_resources,
)
from .params import RhimeConfig
from .requests import load_rhime_params
from .preparation import (
    assemble_rhime_inputs,
    build_sensitivities,
    filter_observations,
    prepare_observation_errors,
)
from .sampling import RhimeSampler, sample_rhime_model
from .specs import (
    DEFAULT_BC_PRIOR,
    DEFAULT_OFFSET_PRIOR,
    DEFAULT_X_PRIOR,
    AdditiveSigmaSettings,
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
    sigma_alignment: SigmaAlignment | None = None,
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
        sigma_alignment: Optional precomputed mismatch alignment. Ordinary
            runners derive it from observations.

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
        )
    return model


def build_standard_rhime_model_result(
    *,
    model_inputs: xr.Dataset,
    model_spec: RhimeModelSpec,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
) -> RhimeModelBuildResult:
    """Build the standard graph and describe its output roles.

    Args:
        model_inputs: Eager canonical arrays for the built-in PyMC graph.
        model_spec: Resolved scientific model choices.
        likelihood_builder: Optional observation-error and distribution builder
            used with the built-in graph.
        likelihood_kwargs: Options expanded only into the custom likelihood.

    Returns:
        Model plus variable roles, supported outputs, and build metadata.

    Raises:
        ValueError: If likelihood options conflict, lack a custom builder, or
            the model specification does not contain exactly one sector.
    """
    if likelihood_kwargs and likelihood_builder is None:
        raise ValueError("Non-empty `likelihood_kwargs` require an active `likelihood_builder`.")
    timing_start = timer_start()
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
    )
    result = builtin_model_build_result(
        model,
        model_spec=model_spec,
        multisector=False,
        input_names=tuple(str(name) for name in model_inputs.data_vars),
    )
    log_timing("rhime.model_build", timer_seconds(timing_start), multisector=False)
    return result


def build_standard(
    prepared: RhimePreparedInputs,
    model_spec: RhimeModelSpec,
    *,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
) -> RhimeModelBuildResult:
    """Select and materialize the standard model inputs, then build the graph.

    Every standard route (full, prepared-input and saved) builds through this
    function. Scientific inputs and enabled priors cross the named PyMC boundary
    together; output settings are not part of model construction.

    Args:
        prepared: Prepared inputs from preparation or a saved preparation.
        model_spec: Resolved scientific model choices.
        likelihood_builder: Optional custom likelihood for the built-in graph.
        likelihood_kwargs: Options for ``likelihood_builder``.

    Returns:
        Model plus variable roles, supported outputs and build metadata.

    Raises:
        ValueError: If no likelihood is selected or a selected component's
            prepared input is missing.
    """
    if likelihood_builder is None and model_spec.likelihood is None:
        raise ValueError("A standard RHIME run requires a built-in or custom likelihood.")
    names = list(standard_model_input_names(prepared, model_spec))
    if len(model_spec.sectors) != 1:
        raise ValueError("Standard RHIME model specs must include exactly one sector.")
    priors = {"x": model_spec.sectors[0].x_prior}
    if model_spec.use_bc and model_spec.bc_prior is not None:
        priors["bc"] = model_spec.bc_prior
    if model_spec.add_offset and model_spec.offset_prior is not None:
        priors["offset"] = model_spec.offset_prior
    likelihood = model_spec.likelihood
    if isinstance(likelihood, (PollutionEventSettings, AdditiveSigmaSettings)) and likelihood.sigma_prior is not None:
        priors["sigma"] = likelihood.sigma_prior
    if isinstance(likelihood, PollutionEventSettings) and isinstance(likelihood.power, Mapping):
        priors["power"] = likelihood.power
    model_inputs, priors = _materialize_pymc_inputs_and_priors(
        prepared, variable_names=tuple(dict.fromkeys(names)), prior_parameters=priors,
    )
    if likelihood is not None:
        likelihood = replace(likelihood, **{field: priors[key] for field, key in
                             (("sigma_prior", "sigma"), ("power", "power")) if key in priors})
    model_spec = replace(
        model_spec, sectors=(replace(model_spec.sectors[0], x_prior=priors["x"]),),
        bc_prior=priors.get("bc", model_spec.bc_prior),
        offset_prior=priors.get("offset", model_spec.offset_prior), likelihood=likelihood,
    )
    return build_standard_rhime_model_result(
        model_inputs=model_inputs,
        model_spec=model_spec,
        likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs,
    )


def make_standard_rhime_result(
    *,
    prepared: RhimePreparedInputs,
    run_spec: RhimeRunSpec,
    sampler: RhimeSampler | None,
    model_build_result: RhimeModelBuildResult | None = None,
    output_contract: OutputContract | None = None,
    idata: xr.DataTree,
    build_and_sample_seconds: float,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    sampler_metadata: Any = None,
    provenance: Mapping[str, Any] | None = None,
) -> RhimeResult:
    """Construct a sampled standard result before output side effects.

    Args:
        prepared: Retained canonical inputs and basis functions.
        run_spec: Resolved model, output, and run settings.
        sampler: Live sampler used for the trace, or ``None`` during saved replay.
        model_build_result: Concrete graph and roles, when retained in memory.
        output_contract: Durable output information for graph-free reconstruction.
            Defaults to the live build result's contract when supplied.
        idata: Borrowed posterior and predictive groups. Trace metadata is
            annotated in place when ``likelihood_builder`` is supplied.
        build_and_sample_seconds: Combined graph-build and sampling duration.
        likelihood_builder: Optional likelihood callable used for provenance.
        likelihood_kwargs: Serializable options owned by the likelihood.
        sampler_metadata: Recorded sampler description for saved replay, kept as
            recorded. A live ``sampler`` replaces it with its own description.
        provenance: Installed-package provenance recorded with the samples.

    Returns:
        Standard-run result ready for requested output construction.
    """
    output_metadata: dict[str, Any] = {
        "build_and_sample_seconds": build_and_sample_seconds,
        "sampler": sampler_metadata,
    }
    if provenance is not None:
        output_metadata["provenance"] = dict(provenance)
    if likelihood_builder is not None:
        identity = callable_metadata(likelihood_builder)
        output_metadata["likelihood_builder"] = identity
        annotate_likelihood_trace(
            idata,
            builder_identity=identity,
            likelihood_kwargs=likelihood_kwargs,
        )
    if likelihood_kwargs is not None:
        output_metadata["likelihood_kwargs"] = likelihood_kwargs
    return RhimeResult(
        run_spec=run_spec,
        model_spec=run_spec.model,
        output_spec=run_spec.output,
        inv_inputs=prepared.inv_inputs,
        idata=idata,
        sampler=sampler,
        model=model_build_result.model if model_build_result is not None else None,
        basis_functions=prepared.basis_functions,
        model_build_result=model_build_result,
        output_contract=output_contract,
        output_metadata=output_metadata,
    )


def prepare_standard(
    merged_data: RhimeMergedData, *,
    domain: str, start_date: str, flux_sources: tuple[str, ...],
    averaging_error: bool, filters: Any, basis_algorithm: str | None, nbasis: int,
    fix_basis_outer_regions: bool, use_bc: bool, bc_freq: str | None,
    min_error: MinErrorConfig, min_error_options: Mapping[str, bool],
    basis_inputs: Mapping[str, Any], bc_basis: xr.Dataset | None = None,
) -> RhimePreparedInputs:
    """Prepare observations and sensitivities from supplied scientific resources.

    Args:
        domain: Domain label for prepared inputs and basis metadata.
        start_date: Requested start anchoring boundary-condition periods.
        flux_sources: Ordered source identities; the first weights basis design.
        averaging_error: Include variability when deriving observation errors.
        filters: Observation-filter choices interpreted by the filtering owner.
        basis_algorithm: Selected basis-generation algorithm.
        nbasis: Requested generated basis size.
        fix_basis_outer_regions: Keep the supplied outer partition fixed.
        use_bc: Construct boundary-condition sensitivities.
        bc_freq: Baked boundary-condition scaling period.
        min_error: Minimum-error policy interpreted during assembly.
        min_error_options: Options for the minimum-error calculation.
        merged_data: Borrowed acquired observations, transport and current flux.
        basis_inputs: Selected operator path or loaded partition and support maps.
        bc_basis: Loaded boundary-condition basis, when selected.

    Returns:
        Retained backend-neutral inputs and their current native-flux binding.
    """
    preparation_start = timer_start()
    # Preparation works on per-site datasets; ``merged_data`` stays acquired data.
    with_errors = prepare_observation_errors(merged_data.site_data, averaging_error=averaging_error)
    filtered, site_options = filter_observations(with_errors, merged_data.site_options, filters=filters)
    runtime_flux = resolve_runtime_flux(merged_data.flux_data, split_by_sectors=merged_data.split_by_sectors)
    with timed("rhime.prepare_inputs.basis_build", basis_algorithm=basis_algorithm):
        basis_functions = prepare_basis(
            site_data=filtered,
            runtime_flux=runtime_flux,
            weighting_flux=merged_data.flux_data[flux_sources[0]]["flux"],
            domain=domain, start_date=start_date, basis_algorithm=basis_algorithm, nbasis=nbasis,
            fix_outer_regions=fix_basis_outer_regions,
            **basis_inputs,
        )
    sensitivities = build_sensitivities(
        filtered,
        basis_functions,
        flux_sources=flux_sources, use_bc=use_bc,
        bc_basis=bc_basis,
        multisector=False,
    )
    prepared = assemble_rhime_inputs(
        sensitivities,
        basis_functions,
        site_options=site_options,
        domain=domain, start_date=start_date, bc_freq=bc_freq, min_error=min_error,
        min_error_options=min_error_options, use_bc=use_bc,
    )
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
    return prepared


def run_standard(
    *,
    config_file: str | Path | None = None,
    config: RhimeConfig | None = None,
    merged_data: RhimeMergedData | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> RhimeResult:
    """Run a standard single-sector RHIME inversion.

    The visible process is resolve → acquire or accept supplied data → observation
    errors → filter → basis → sensitivities → assemble → materialize → build → sample → result →
    requested outputs.

    Selected basis/support files and reporting country files are checked before
    acquisition, including when merged data is supplied. Basis partitions and
    support maps are loaded at that boundary. Saved bases are rebound to current
    flux; ``basis_output_path`` is an exact retained-artifact destination.

    Without supplied merged data, this runner acquires fresh data. Persisted
    prepared inputs can be reused with
    :func:`openghg_inversions.rhime.run_rhime_from_prepared_inputs`.

    Args:
        config_file: Optional INI configuration file. Values in ``kwargs``
            override values read from this file.
        config: Complete request from ``RhimeConfig.from_params``.
            It is used without resolving again and cannot
            be combined with ``config_file`` or raw run parameters in ``kwargs``.
        merged_data: Optional externally supplied merged scientific data.
            Passing this borrowed in-memory handoff bypasses OpenGHG acquisition
            and resumes at observation-error preparation. This may derive error arrays
            and compute zero-error fallback diagnostics.
            The acquisition owner checks recorded species, domain, time window,
            and sector layout without mutating it.
        likelihood_builder: Optional Python-only callable invoked with a
            completed forward-model mean and explicit error-model inputs in
            the active PyMC model. It must return the canonical observed
            variable ``y`` and create the canonical error scale ``epsilon``.
            The callable is never read from configuration or stored in
            run/model specifications.
        likelihood_kwargs: Options specific to the custom likelihood. Common
            scientific arrays are passed explicitly by the recipe.
        **kwargs: RHIME run parameters using snake-case names, such as
            ``output_path``, ``output_name``, ``flux_sources``, and
            ``x_prior``. ``species`` names the primary gas or tracer used for
            object-store lookup and output naming. ``flux_sources`` contains
            OpenGHG flux ``source`` values. Legacy ``emissions_name`` is
            accepted only as a compatibility alias when ``flux_sources`` is
            absent.

    Returns:
        Modern RHIME result containing canonical inputs, a sampled DataTree, specs,
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
    setup_start = timer_start()
    if config is not None:
        if config_file is not None or kwargs:
            raise ValueError("Pass either resolved `config` or `config_file`/run parameters, not both.")
        if config.data.split_by_sectors:
            raise ValueError("`run_rhime` received configuration for a different sector layout.")
    else:
        params = load_rhime_params(config_file=config_file, overrides=kwargs)
        if likelihood_builder is not None and params["model"].get("mismatch_model") is not None:
            raise ValueError("A custom likelihood cannot be combined with a built-in mismatch model.")
        config = RhimeConfig.from_params(params, multisector=False)
    if likelihood_builder is not None and config.model.likelihood is not None:
        raise ValueError("A custom likelihood cannot be combined with a built-in mismatch model.")
    sampler = resolve_sampler(config.sampler)
    log_timing("rhime.runner_setup", timer_seconds(setup_start), multisector=False)
    if likelihood_builder is None and config.model.likelihood is None:
        raise ValueError("A standard RHIME run requires a built-in or custom likelihood.")

    # Read selected scientific resources before expensive data acquisition.
    validate_output_resources(config.output)
    basis_inputs = resolve_basis_inputs(config.data)
    bc_basis = load_bc_basis(config.data.bc_basis_paths) if config.data.use_bc and config.data.bc_basis_paths is not None else None
    # 1. Acquire, or check supplied merged data.
    merged = acquire_merged_data(config.data, merged_data)
    # 2. Scientific preparation, then the runner-owned optional basis export.
    prepared = prepare_standard(
        merged, basis_inputs=basis_inputs, bc_basis=bc_basis,
        **config.data.select(
            "domain", "start_date", "flux_sources", "averaging_error", "filters",
            "basis_algorithm", "nbasis", "fix_basis_outer_regions", "use_bc", "bc_freq",
            "min_error", "min_error_options",
        ),
    )
    if config.data.basis_output_path is not None:
        prepared = replace(prepared, basis_functions=export_basis(prepared.basis_functions, config.data.basis_output_path))
    run_spec = config.retained_run_spec(prepared)
    # 3. Build the graph from the prepared inputs, then sample it.
    build_and_sample_start = timer_start()
    model_build_result = build_standard(
        prepared,
        run_spec.model,
        likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs,
    )
    idata = sample_rhime_model(model_build_result, sampler)
    # 4. Bind the samples to durable output information, then make requested products.
    result = make_standard_rhime_result(
        prepared=prepared,
        run_spec=run_spec,
        sampler=sampler,
        model_build_result=model_build_result,
        idata=idata,
        build_and_sample_seconds=timer_seconds(build_and_sample_start),
        likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs,
        provenance=installed_ogi_provenance(),
    )
    output_start = timer_start()
    result = make_standard_rhime_outputs(
        result=result, prepared=prepared,
    )
    log_timing(
        "rhime.output_total",
        timer_seconds(output_start),
        multisector=False,
        output_format=run_spec.output.output_format,
    )
    return result


# Established public import retained through the 0.9 compatibility cycle.
run_rhime = run_standard
