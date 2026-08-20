"""Readable scientific recipe for a source-resolved multisector RHIME inversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import arviz as az
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.models.components import (
    add_linear_component,
    add_offset_component,
)
from openghg_inversions.models.coords import registered_model
from openghg_inversions.models.pollution_event import build_pollution_event_gaussian_likelihood
from openghg_inversions.models.priors import PriorArgs
from openghg_inversions.models._flux import (
    _namespace_sector_state_coords,
    _prepared_sources,
    _select_sector_design,
)
from openghg_inversions.models.state_activity import (
    PreparedLinearSensitivity,
    StateActivity,
    active_prior_args,
    prepare_linear_sensitivity,
    resolve_state_activity,
)
from openghg_inversions.observation_error import (
    AggregationError,
    OBSERVATION_ERROR_INPUT_NAMES,
    aggregation_error_input_names,
    resolve_aggregation_error,
)
from openghg_inversions.sigma import SigmaAlignment

from .specs import (
    DEFAULT_BC_PRIOR,
    DEFAULT_OFFSET_PRIOR,
    DEFAULT_SIGMA_PRIOR,
    RhimeModelSpec,
    RhimeRunSpec,
    SectorSpec,
)

from ._model_building import (
    builtin_model_build_result,
    validate_custom_likelihood_result,
    validate_likelihood_kwargs,
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
from .outputs import RhimeResult, make_multisector_rhime_outputs
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


_SectorComponent = tuple[SectorSpec, PreparedLinearSensitivity, PriorArgs, StateActivity]
_MULTISECTOR_FLUX_INPUT_NAMES = ("H",)
_MODEL_ERROR_ALIGNMENT_INPUT_NAMES = ("site_indicator",)
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


def multisector_model_input_names(
    prepared: RhimePreparedInputs,
    model_spec: RhimeModelSpec,
    *,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> tuple[str, ...]:
    """Declare arrays required by selected multisector-model components.

    Args:
        prepared: Backend-neutral prepared inputs.
        model_spec: Resolved multisector component options.
        likelihood_builder: Custom likelihood which owns any additional error
            inputs itself.

    Returns:
        Prepared variable names selected for coordinated materialization.

    Raises:
        ValueError: If a selected component's required input is absent or its
            aggregation-error representation is ambiguous.
    """
    _require_component_inputs(
        prepared,
        _MULTISECTOR_FLUX_INPUT_NAMES,
        owner="Multisector flux component",
    )
    _require_component_inputs(
        prepared,
        OBSERVATION_ERROR_INPUT_NAMES,
        owner="Observation-error component",
    )
    names = [
        *_MULTISECTOR_FLUX_INPUT_NAMES,
        *OBSERVATION_ERROR_INPUT_NAMES,
    ]
    if likelihood_builder is None and not model_spec.no_model_error:
        _require_component_inputs(
            prepared,
            _MODEL_ERROR_ALIGNMENT_INPUT_NAMES,
            owner="Model-error alignment component",
        )
        names.extend(_MODEL_ERROR_ALIGNMENT_INPUT_NAMES)
    elif model_spec.add_offset:
        _require_component_inputs(
            prepared,
            _MODEL_ERROR_ALIGNMENT_INPUT_NAMES,
            owner="Multisector offset component selected by `add_offset=True`",
        )
        names.extend(_MODEL_ERROR_ALIGNMENT_INPUT_NAMES)
    if model_spec.use_bc:
        _require_component_inputs(
            prepared,
            _BASELINE_INPUT_NAMES,
            owner="Multisector baseline component selected by `use_bc=True`",
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


def _prepare_multisector_flux_components(
    flux_sensitivity: xr.DataArray,
    sectors: Sequence[SectorSpec],
    *,
    state_activity: StateActivity | None,
) -> tuple[_SectorComponent, ...]:
    """Validate and select the flux inputs used by the visible model loop.

    Args:
        flux_sensitivity: Canonical source-resolved sensitivity input.
        sectors: Ordered scientific sector specifications.
        state_activity: Shared active/fixed state policy used when a sector has
            no explicit override.

    Returns:
        Sector specification, selected design, normalized prior, and resolved
        activity policy for each sector, in the requested order.

    Raises:
        ValueError: If sector names, sources, suffixes, layouts, priors, or
            activity policies are inconsistent.
    """
    sectors = tuple(sectors)
    if len(sectors) < 2:
        raise ValueError("Multi-sector RHIME requires at least two sectors.")

    sector_names = [sector.name for sector in sectors]
    duplicate_names = list(dict.fromkeys(name for name in sector_names if sector_names.count(name) > 1))
    if duplicate_names:
        raise ValueError(
            f"Multi-sector RHIME requires unique sector names; duplicate sector {duplicate_names[0]!r}."
        )
    if any(not name.strip() for name in sector_names):
        raise ValueError("Multi-sector RHIME requires non-empty sector names.")

    source_sectors: dict[str, list[str]] = {}
    for sector in sectors:
        source_sectors.setdefault(sector.flux_source, []).append(sector.name)
    duplicate_sources = {source: names for source, names in source_sectors.items() if len(names) > 1}
    if duplicate_sources:
        details = ", ".join(
            f"source {source!r} is mapped by sectors {names!r}" for source, names in duplicate_sources.items()
        )
        raise ValueError(
            "Multi-sector RHIME requires a distinct source for each current sector; " + details + "."
        )

    suffixes = [sector.variable_suffix for sector in sectors]
    if any(not suffix.strip() for suffix in suffixes):
        raise ValueError("Multi-sector RHIME requires non-empty PyMC variable suffixes.")
    duplicate_suffixes = list(dict.fromkeys(suffix for suffix in suffixes if suffixes.count(suffix) > 1))
    if duplicate_suffixes:
        raise ValueError(
            "Multi-sector RHIME requires unique PyMC variable suffixes; "
            f"duplicate suffix {duplicate_suffixes[0]!r}."
        )

    available_sources = _prepared_sources(flux_sensitivity)
    missing_sources = [
        (sector.name, sector.flux_source) for sector in sectors if sector.flux_source not in available_sources
    ]
    if missing_sources:
        details = ", ".join(f"sector {name!r} -> source {source!r}" for name, source in missing_sources)
        raise ValueError(
            f"Source data required by {details} is not present in `flux_sensitivity.source`; "
            f"available source(s): {available_sources!r}."
        )

    gathered_layout = "source" not in flux_sensitivity.dims
    components: list[_SectorComponent] = []
    for sector in sectors:
        design = _select_sector_design(
            flux_sensitivity,
            sector=sector.name,
            source=sector.flux_source,
            variable_suffix=sector.variable_suffix,
            namespace_state_dim=False,
        )
        sector_policy = sector.state_activity if sector.state_activity is not None else state_activity
        prepared_sensitivity = prepare_linear_sensitivity(design)
        resolved_activity = resolve_state_activity(prepared_sensitivity.removed, sector_policy)
        all_active = replace(
            resolved_activity,
            active=xr.ones_like(resolved_activity.active, dtype=bool),
        )
        prior = active_prior_args(dict(sector.x_prior), all_active)
        design_state_dim = next(
            str(dim)
            for dim in prepared_sensitivity.sensitivity.dims
            if dim != prepared_sensitivity.output_dim
        )
        backend_design = _namespace_sector_state_coords(
            prepared_sensitivity.sensitivity,
            variable_suffix=sector.variable_suffix,
            namespace_state_dim=gathered_layout or design_state_dim != resolved_activity.state_dim,
        )
        semantic_state_dim = resolved_activity.state_dim
        backend_state_dim = (
            f"{semantic_state_dim}_{sector.variable_suffix}"
            if gathered_layout
            else semantic_state_dim
        )
        rename_state = (
            {semantic_state_dim: backend_state_dim} if semantic_state_dim != backend_state_dim else {}
        )
        backend_removed = _namespace_sector_state_coords(
            prepared_sensitivity.removed,
            variable_suffix=sector.variable_suffix,
            namespace_state_dim=gathered_layout,
        )
        backend_prepared = PreparedLinearSensitivity(
            sensitivity=backend_design,
            removed=backend_removed,
            output_dim=prepared_sensitivity.output_dim,
        )
        backend_activity = StateActivity(
            active=resolved_activity.active.rename(rename_state),
            fixed_value=resolved_activity.fixed_value.rename(rename_state),
        )
        components.append((sector, backend_prepared, prior, backend_activity))
    return tuple(components)


def build_multisector_rhime_model(
    flux_sensitivity: xr.DataArray,
    *,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    minimum_error: xr.DataArray,
    aggregation_error: AggregationError,
    sigma_alignment: SigmaAlignment | None = None,
    sectors: Sequence[SectorSpec],
    boundary_sensitivity: xr.DataArray | None = None,
    site_indicator: xr.DataArray | None = None,
    bc_prior: PriorArgs | None = None,
    sigma_prior: PriorArgs | None = None,
    offset_prior: PriorArgs | None = None,
    add_offset: bool = False,
    use_bc: bool = True,
    pollution_events_from_obs: bool = False,
    no_model_error: bool = False,
    offset_args: dict | None = None,
    power: PriorArgs | float = 1.99,
    state_activity: StateActivity | None = None,
    bc_state_activity: StateActivity | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
) -> pm.Model:
    """Build the concrete shared-basis multi-sector RHIME model.

    Each sector receives its own state vector ``x_<sector>`` and forward-model
    contribution ``mu_<sector>``. The recipe visibly sums those contributions,
    adds the baseline and optional offset, then passes the completed mean to
    the likelihood.

    Args:
        flux_sensitivity: Labelled source-resolved flux sensitivity,
            either shared-basis or gathered source-specific state layout.
        observations: Observed mole fractions.
        observation_error: Reported observation-error standard deviations.
        minimum_error: Minimum total-error standard deviations.
        aggregation_error: Validated fixed aggregation-error representation.
        sigma_alignment: Observation alignment for mismatch parameters when
            model error is enabled.
        sectors: Ordered sector specifications containing each scientific name,
            OpenGHG source, PyMC suffix, prior, and optional activity override.
        boundary_sensitivity: Optional labelled boundary sensitivity matrix.
        site_indicator: Optional observation-to-site index used by offsets.
        bc_prior: Prior for boundary-condition scaling factors.
        sigma_prior: Prior for mismatch-error parameters.
        offset_prior: Prior for optional offsets.
        add_offset: Whether to include an offset term.
        use_bc: Whether to include boundary-condition terms.
        pollution_events_from_obs: Whether mismatch scaling uses observed
            rather than modelled pollution enhancements.
        no_model_error: Whether to suppress inferred mismatch error.
        offset_args: Extra keyword arguments for the offset component.
        power: Exponent or prior used in mismatch-error scaling.
        state_activity: State policy shared by sectors without an override.
        bc_state_activity: Optional active/fixed boundary-state policy.
        likelihood_builder: Optional observation-error and distribution builder.
            It receives the completed model mean from this recipe.
        likelihood_kwargs: Options specific to a custom likelihood. Common
            scientific arrays remain explicit and are not included here.

    Returns:
        Built PyMC model.

    Raises:
        KeyError: If required sensitivity inputs are absent.
        ValueError: If sector labels, sources, suffixes, state policies, or
            canonical likelihood variables are invalid.
        TypeError: If a custom likelihood returns the wrong result type.
    """
    likelihood_kwargs = validate_likelihood_kwargs(likelihood_builder, likelihood_kwargs)
    sector_components = _prepare_multisector_flux_components(
        flux_sensitivity,
        sectors,
        state_activity=state_activity,
    )
    bc_prior = dict(DEFAULT_BC_PRIOR if bc_prior is None else bc_prior)
    sigma_prior = dict(DEFAULT_SIGMA_PRIOR if sigma_prior is None else sigma_prior)
    offset_prior = dict(DEFAULT_OFFSET_PRIOR if offset_prior is None else offset_prior)
    prepared_boundary = (
        prepare_linear_sensitivity(boundary_sensitivity)
        if use_bc and boundary_sensitivity is not None
        else None
    )
    with registered_model() as model:
        sector_outputs = []
        for sector, design, prior, sector_policy in sector_components:
            sector_outputs.append(
                add_linear_component(
                    design,
                    data_name=f"hx_{sector.variable_suffix}",
                    prior_args=prior,
                    var_name=f"x_{sector.variable_suffix}",
                    output_name=f"mu_{sector.variable_suffix}",
                    output_dim="nmeasure",
                    compute_deterministic=True,
                    state_activity=sector_policy,
                ).output
            )

        pollution_mean = cast(Any, pt.stack(sector_outputs, axis=0)).sum(axis=0)

        boundary_mean = None
        if use_bc:
            if boundary_sensitivity is None:
                raise ValueError(
                    "Multisector baseline component requires `boundary_sensitivity` when `use_bc` is true."
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
            if site_indicator is None:
                raise ValueError(
                    "Multisector offset component requires `site_indicator` when `add_offset` is true."
                )
            offset = add_offset_component(
                site_indicator,
                prior_args=offset_prior,
                output_name="offset",
                output_dim="nmeasure",
                **(offset_args or {}),
            )

        baseline_mean = boundary_mean
        if offset is not None:
            baseline_mean = offset if baseline_mean is None else baseline_mean + offset
        modelled_mean = pollution_mean if baseline_mean is None else pollution_mean + baseline_mean

        if likelihood_builder is None:
            likelihood = build_pollution_event_gaussian_likelihood(
                observations=observations,
                observation_error=observation_error,
                minimum_error=minimum_error,
                aggregation_error=aggregation_error,
                mean=modelled_mean,
                pollution_mean=pollution_mean,
                pollution_event_baseline=baseline_mean,
                sigma_alignment=sigma_alignment,
                sigma_prior=sigma_prior,
                power=power,
                pollution_events_from_obs=pollution_events_from_obs,
                no_model_error=no_model_error,
                output_dim="nmeasure",
            )
        else:
            likelihood = likelihood_builder(
                observations=observations,
                observation_error=observation_error,
                minimum_error=minimum_error,
                aggregation_error=aggregation_error,
                mean=modelled_mean,
                pollution_mean=pollution_mean,
                pollution_event_baseline=baseline_mean,
                output_dim="nmeasure",
                **(likelihood_kwargs or {}),
            )
            validate_custom_likelihood_result(model, likelihood)

    return model


def _validate_multisector_basis_layout(
    basis_functions: BasisFunctions,
    model_spec: RhimeModelSpec,
    inv_inputs: xr.Dataset,
) -> None:
    """Require each retained source basis to match its prepared sector design.

    Args:
        basis_functions: Retained source-specific basis operators.
        model_spec: Sector declarations selecting those sources.
        inv_inputs: Prepared source-resolved sensitivity arrays.

    Raises:
        ValueError: If a source basis is missing or its state coordinate does
            not match the corresponding prepared sensitivity design.
    """
    design = inv_inputs["H"]
    region_layouts: list[tuple[str, str, int, int, bool]] = []
    for sector in model_spec.sectors:
        try:
            source_basis = basis_functions.for_source(sector.flux_source)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Sector {sector.name!r} requires source {sector.flux_source!r}, "
                "but the retained basis has no matching source-specific basis."
            ) from exc
        state_dim = source_basis.operator.meta.state_dim
        basis_index = source_basis.operator.basis_matrix.get_index(state_dim)
        sector_design = _select_sector_design(
            design,
            sector=sector.name,
            source=sector.flux_source,
            variable_suffix=sector.variable_suffix,
        )
        prepared_state_dims = [str(dim) for dim in sector_design.dims if dim != "nmeasure"]
        if len(prepared_state_dims) != 1:
            raise ValueError(
                f"Sector {sector.name!r} -> source {sector.flux_source!r} must have exactly "
                f"one prepared state dimension; found {prepared_state_dims!r}."
            )
        prepared_index = sector_design.get_index(prepared_state_dims[0])
        region_layouts.append(
            (
                sector.name,
                sector.flux_source,
                len(basis_index),
                len(prepared_index),
                basis_index.equals(prepared_index),
            )
        )
    if any(not matches for _, _, _, _, matches in region_layouts):
        details = ", ".join(
            f"sector {sector!r} -> source {source!r}: basis has {basis_count} regions, "
            f"prepared H has {prepared_count}" + ("" if matches else " with different coordinates")
            for sector, source, basis_count, prepared_count, matches in region_layouts
        )
        raise ValueError(
            "Retained source-specific basis state coordinates do not match prepared H; " + details + "."
        )


def build_multisector_rhime_model_result(
    *,
    prepared: RhimePreparedInputs,
    model_inputs: xr.Dataset,
    run_spec: RhimeRunSpec,
    model_builder: RhimeModelBuilder | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
) -> RhimeModelBuildResult:
    """Validate source-specific bases and build the multisector graph result.

    Args:
        prepared: Retained source-specific prepared-input artifact.
        model_inputs: Eager canonical arrays for the built-in PyMC graph.
        run_spec: Resolved model, sampling, and output specification.
        model_builder: Optional complete-model builder for advanced prepared-
            input workflows.
        likelihood_builder: Optional observation-error and distribution builder
            used with the built-in graph.
        likelihood_kwargs: Options expanded only into the custom likelihood.

    Returns:
        Model plus variable roles, supported outputs, and build metadata.

    Raises:
        ValueError: If the basis layout is incompatible, both extension points
            are supplied, or the result conflicts with the run specification.
    """
    if run_spec.output.output_format in ("basic", "legacy"):
        raise ValueError(
            f"RHIME output_format {run_spec.output.output_format!r} supports only single-sector runs."
        )
    likelihood_kwargs = validate_likelihood_kwargs(likelihood_builder, likelihood_kwargs)
    if model_builder is not None and likelihood_builder is not None:
        raise ValueError("Pass either `model_builder` or `likelihood_builder`, not both.")
    timing_start = timer_start()
    _validate_multisector_basis_layout(prepared.basis_functions, run_spec.model, prepared.inv_inputs)
    if model_builder is not None:
        builder_context = RhimeModelBuilderContext(
            prepared_inputs=prepared,
            run_spec=run_spec,
            multisector=True,
        )
        result = validated_custom_model_build(model_builder, context=builder_context)
        validate_model_build_result(result, context=builder_context)
    else:
        model_spec = run_spec.model
        sigma_alignment = (
            SigmaAlignment.from_frequency(
                model_inputs["site_indicator"],
                frequency=model_spec.sigma_freq,
                per_site=model_spec.sigma_per_site,
                anchor_time=model_spec.sigma_freq_anchor,
            )
            if likelihood_builder is None and not model_spec.no_model_error
            else None
        )
        aggregation_error = resolve_aggregation_error(
            model_inputs,
            model_spec.aggregation_error_mode,
        )
        model = build_multisector_rhime_model(
            model_inputs["H"],
            observations=model_inputs["mf"],
            observation_error=model_inputs["mf_error"],
            minimum_error=model_inputs["min_error"],
            aggregation_error=aggregation_error,
            sigma_alignment=sigma_alignment,
            sectors=model_spec.sectors,
            boundary_sensitivity=model_inputs.get("H_bc"),
            site_indicator=model_inputs.get("site_indicator"),
            bc_prior=model_spec.bc_prior,
            bc_state_activity=model_spec.bc_state_activity,
            sigma_prior=model_spec.sigma_prior,
            offset_prior=model_spec.offset_prior,
            add_offset=model_spec.add_offset,
            use_bc=model_spec.use_bc,
            pollution_events_from_obs=model_spec.pollution_events_from_obs,
            no_model_error=model_spec.no_model_error,
            offset_args=model_spec.offset_args,
            power=model_spec.power,
            state_activity=model_spec.state_activity,
            likelihood_builder=likelihood_builder,
            likelihood_kwargs=likelihood_kwargs,
        )
        result = builtin_model_build_result(
            model,
            model_spec=run_spec.model,
            multisector=True,
            input_names=tuple(str(name) for name in prepared.inv_inputs.data_vars),
        )
    log_timing("rhime.model_build", timer_seconds(timing_start), multisector=True)
    return result


def make_multisector_rhime_result(
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
) -> RhimeResult:
    """Construct a sampled multisector result before output side effects.

    Args:
        prepared: Retained source-resolved inputs and basis functions.
        run_spec: Resolved model, output, and run settings.
        sampler: Sampler configuration used for the trace.
        model_build_result: Concrete graph and semantic variable roles.
        idata: Sampled posterior and predictive groups.
        build_and_sample_seconds: Combined graph-build and sampling duration.
        model_builder: Optional complete-model callable used for provenance.
        likelihood_builder: Optional likelihood callable used for provenance.
        likelihood_kwargs: Serializable options owned by the likelihood.

    Returns:
        Multisector result ready for requested output construction.
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
    return result


def run_rhime_multisector(
    *,
    config_file: str | Path | None = None,
    merged_data: RhimeMergedData | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> RhimeResult:
    """Run a shared-basis multi-sector RHIME inversion.

    The visible process is resolve → retrieve/reload → filter → basis →
    sensitivities → assemble → materialize → build → sample → result →
    requested outputs.
    This module keeps source layout validation and sector-aware outputs beside
    that process instead of hiding them behind standard/multisector branching.

    Args:
        config_file: Optional INI configuration file. Values in ``kwargs``
            override values read from this file.
        merged_data: Optional externally supplied source-resolved merged
            scientific data. Passing it bypasses OpenGHG acquisition and
            merged-cache I/O, then resumes at filtering after validation.
        likelihood_builder: Optional Python-only callable invoked with a
            completed forward-model mean and explicit error-model inputs in
            the active PyMC model. It must return the canonical observed
            variable ``y`` and create the canonical error scale ``epsilon``.
            The callable is never read from configuration or stored in
            run/model specifications.
        likelihood_kwargs: Options specific to the custom likelihood. Common
            scientific arrays are passed explicitly by the recipe.
        **kwargs: RHIME run parameters using snake-case names. Multi-sector
            runs require at least two ``flux_sources`` and may include a
            complete ``sector_priors`` mapping keyed by sector name. When model
            sector labels differ from OpenGHG source values, pass
            ``sector_sources`` as a one-to-one mapping from sector name to one
            unique value in ``flux_sources``. Legacy ``emissions_name`` is
            accepted only as a compatibility alias when ``flux_sources`` is
            absent.

    Returns:
        Modern RHIME result containing canonical inputs, InferenceData, specs,
        output metadata, and sector diagnostics.

    Raises:
        TypeError: If a likelihood builder is not callable or returns the
            wrong result type.
        ValueError: If required parameters are missing, unsupported parameters
            are supplied, fewer than two flux sources are provided, or
            likelihood variables or requested-output compatibility are invalid.

    Notes:
        A non-callable likelihood builder is rejected before configuration is
        parsed or data is acquired, prepared, or materialized.
    """
    likelihood_kwargs = validate_likelihood_kwargs(likelihood_builder, likelihood_kwargs)
    params = (
        params_from_config(config_file, extra_kwargs=kwargs, normalise=False)
        if config_file is not None
        else dict(kwargs)
    )
    setup = resolve_rhime_options(params=params, multisector=True)

    preparation_start = timer_start()
    merged = retrieve_or_reload_rhime_data(
        setup.data_args,
        multisector=True,
        merged_data=merged_data,
    )
    filtered = filter_rhime_observations(merged, setup.data_args)
    basis_functions = build_rhime_basis(filtered, setup.data_args)
    site_data = build_rhime_sensitivities(
        filtered,
        basis_functions,
        setup.data_args,
        multisector=True,
    )
    prepared = assemble_rhime_inputs(filtered, basis_functions, site_data, setup.data_args)
    log_timing(
        "rhime.prepare_inputs",
        timer_seconds(preparation_start),
        multisector=True,
        nmeasure=prepared.inv_inputs.sizes.get("nmeasure"),
        sites=len(prepared.sites),
        regions=prepared.inv_inputs.sizes.get("region"),
        sources=prepared.inv_inputs.sizes.get("source"),
        basis_source=prepared.basis_artifact_source,
    )
    run_spec = with_prepared_rhime_sites(setup.run_spec, prepared)

    model_inputs = materialize_pymc_inputs(
        prepared,
        variable_names=multisector_model_input_names(
            prepared,
            run_spec.model,
            likelihood_builder=likelihood_builder,
        ),
    )
    build_and_sample_start = timer_start()
    model_build_result = build_multisector_rhime_model_result(
        prepared=prepared,
        model_inputs=model_inputs,
        run_spec=run_spec,
        likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs,
    )
    idata = sample_rhime_model(
        model_build_result,
        setup.sampler,
    )
    result = make_multisector_rhime_result(
        prepared=prepared,
        run_spec=run_spec,
        sampler=setup.sampler,
        model_build_result=model_build_result,
        idata=idata,
        build_and_sample_seconds=timer_seconds(build_and_sample_start),
        likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs,
    )
    output_start = timer_start()
    make_multisector_rhime_outputs(result=result, prepared=prepared)
    log_timing(
        "rhime.output_total",
        timer_seconds(output_start),
        multisector=True,
        output_format=run_spec.output.output_format,
    )
    return result
