"""Readable scientific recipe for a source-resolved multisector RHIME inversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import arviz as az
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.inversion_data import RhimeMergedData, RhimePreparedInputs
from openghg_inversions.models._rhime_flux import (
    _resolve_multisector_components,
    _resolve_sector_bindings,
    _select_sector_design,
)
from openghg_inversions.models.components import add_state_linear_component
from openghg_inversions.models.coords import CoordRegistry, attach_coord_registry
from openghg_inversions.models.rhime import (
    DEFAULT_BC_PRIOR,
    DEFAULT_OFFSET_PRIOR,
    DEFAULT_SIGMA_PRIOR,
    DEFAULT_X_PRIOR,
    RhimeModelSpec,
    _add_rhime_observation_components,
    _LIKELIHOOD_RESULT_ATTR,
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
from .outputs import RhimeResult, apply_output_bundle, make_multisector_output_bundle
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


def build_rhime_multisector_model(
    inv_inputs: xr.Dataset,
    *,
    sigma_alignment: SigmaAlignment,
    sectors: Sequence[str] | None = None,
    sector_sources: Mapping[str, str] | None = None,
    sector_variable_suffixes: Mapping[str, str] | None = None,
    sector_priors: Mapping[str, dict] | None = None,
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
    sector_state_activities: Mapping[str, StateActivity] | None = None,
    bc_state_activity: StateActivity | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> pm.Model:
    """Build the concrete shared-basis multi-sector RHIME model.

    Each sector receives its own state vector ``x_<sector>`` and forward-model
    contribution ``mu_<sector>``. The total ``mu`` is the sum of sector
    contributions and is passed to the standard RHIME likelihood.
    """
    sector_bindings = _resolve_sector_bindings(
        inv_inputs,
        sectors,
        sector_sources=sector_sources,
        sector_variable_suffixes=sector_variable_suffixes,
    )
    sector_components = _resolve_multisector_components(
        inv_inputs,
        sector_bindings,
        sector_priors=sector_priors,
        x_prior=x_prior,
        default_x_prior=DEFAULT_X_PRIOR,
        state_activity=state_activity,
        sector_state_activities=sector_state_activities,
    )
    bc_prior = dict(DEFAULT_BC_PRIOR if bc_prior is None else bc_prior)
    sigma_prior = dict(DEFAULT_SIGMA_PRIOR if sigma_prior is None else sigma_prior)
    offset_prior = dict(DEFAULT_OFFSET_PRIOR if offset_prior is None else offset_prior)
    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        sector_outputs = []
        for component in sector_components:
            linear_component = add_state_linear_component(
                component.design,
                data_name=f"hx_{component.variable_suffix}",
                prior_args=dict(component.prior_args),
                var_name=f"x_{component.variable_suffix}",
                output_name=f"mu_{component.variable_suffix}",
                output_dim="nmeasure",
                compute_deterministic=True,
                state_activity=component.state_activity,
            )
            sector_outputs.append(linear_component.output)

        total_mu = pm.Deterministic(
            "mu",
            cast(Any, pt.stack(sector_outputs, axis=0)).sum(axis=0),
            dims="nmeasure",
        )
        likelihood_result = _add_rhime_observation_components(
            inv_inputs,
            mu=total_mu,
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


def build_rhime_multisector_model_from_spec(
    inv_inputs: xr.Dataset,
    model_spec: RhimeModelSpec,
    *,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> pm.Model:
    """Build the concrete multi-sector RHIME model from a model spec."""
    sigma_alignment = SigmaAlignment.from_frequency(
        inv_inputs["site_indicator"],
        frequency=model_spec.sigma_freq,
        per_site=model_spec.sigma_per_site,
        anchor_time=model_spec.sigma_freq_anchor,
    )
    sector_state_activities = dict(model_spec.sector_state_activities or {})
    sector_state_activities.update(
        {
            sector.name: sector.state_activity
            for sector in model_spec.sectors
            if sector.state_activity is not None
        }
    )
    return build_rhime_multisector_model(
        inv_inputs,
        sigma_alignment=sigma_alignment,
        sectors=[sector.name for sector in model_spec.sectors],
        sector_sources={sector.name: sector.flux_source for sector in model_spec.sectors},
        sector_variable_suffixes={sector.name: sector.variable_suffix for sector in model_spec.sectors},
        sector_priors={sector.name: dict(sector.x_prior) for sector in model_spec.sectors},
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
        state_activity=model_spec.state_activity,
        sector_state_activities=sector_state_activities or None,
        likelihood_builder=likelihood_builder,
    )


def _validate_multisector_basis_layout(
    basis_functions: BasisFunctions,
    model_spec: RhimeModelSpec,
    inv_inputs: xr.Dataset,
) -> None:
    """Require each retained source basis to match its prepared sector design."""
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


def build_multisector_rhime_model(
    *,
    prepared: RhimePreparedInputs,
    model_inputs: xr.Dataset,
    run_spec: RhimeRunSpec,
    model_builder: RhimeModelBuilder | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> RhimeModelBuildResult:
    """Validate source-specific bases and build the concrete multisector graph."""
    if model_builder is not None and likelihood_builder is not None:
        raise ValueError("Pass either `model_builder` or `likelihood_builder`, not both.")
    timing_start = timer_start()
    _validate_multisector_basis_layout(prepared.basis_functions, run_spec.model, prepared.inv_inputs)
    builder_context = RhimeModelBuilderContext(
        prepared_inputs=prepared,
        run_spec=run_spec,
        multisector=True,
    )
    if model_builder is not None:
        result = validated_custom_model_build(model_builder, context=builder_context)
    else:
        model = build_rhime_multisector_model_from_spec(
            model_inputs,
            run_spec.model,
            **({} if likelihood_builder is None else {"likelihood_builder": likelihood_builder}),
        )
        result = builtin_model_build_result(model, model_spec=run_spec.model, multisector=True)
    if likelihood_builder is not None:
        validate_model_build_result(result, context=builder_context, builder_kind="likelihood")
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
) -> RhimeResult:
    """Construct a multisector result with its sector-aware output products."""
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
    output_bundle = make_multisector_output_bundle(
        output_spec=run_spec.output,
        run_spec=run_spec,
        model_spec=run_spec.model,
        idata=idata,
        prepared=prepared,
        country_file=run_spec.output.country_file,
        variable_roles=model_build_result.variable_roles,
        builder_metadata=builder_metadata,
    )
    log_timing(
        "rhime.output_bundle_total",
        timer_seconds(timing_start),
        multisector=True,
        output_format=run_spec.output.output_format,
    )
    apply_output_bundle(result, output_bundle)
    return result


def run_rhime_multisector(
    *,
    config_file: str | Path | None = None,
    merged_data: RhimeMergedData | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    **kwargs: Any,
) -> RhimeResult:
    """Run a shared-basis multi-sector RHIME inversion.

    The visible process is resolve → retrieve/reload → filter → basis →
    sensitivities → assemble → materialize → build → sample → result/output.
    This module keeps source layout validation and sector-aware outputs beside
    that process instead of hiding them behind standard/multisector branching.

    Args:
        config_file: Optional INI configuration file. Values in ``kwargs``
            override values read from this file.
        merged_data: Optional externally supplied source-resolved merged
            scientific data. Passing it bypasses OpenGHG acquisition and
            merged-cache I/O, then resumes at filtering after validation.
        likelihood_builder: Optional Python-only callable invoked with a
            ``RhimeLikelihoodContext`` in the active PyMC model and returning
            ``RhimeLikelihoodResult``. The result declares semantic variable
            roles, supported output formats, and JSON-compatible metadata;
            roles drive predictive selection and output compatibility is
            validated before sampling. The callable is never read from
            configuration or stored in run/model specifications.
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
            likelihood roles, metadata, or requested-output compatibility are
            invalid.

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
        aggregation_error_mode=run_spec.model.aggregation_error_mode,
    )
    build_and_sample_start = timer_start()
    model_build_result = build_multisector_rhime_model(
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
    return make_multisector_rhime_result(
        prepared=prepared,
        run_spec=run_spec,
        sampler=setup.sampler,
        model_build_result=model_build_result,
        idata=idata,
        build_and_sample_seconds=timer_seconds(build_and_sample_start),
        likelihood_builder=likelihood_builder,
    )
