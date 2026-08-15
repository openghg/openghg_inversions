"""Modern public RHIME runner implementation.

This module accepts Python keyword arguments or RHIME ``.ini`` files,
normalizes legacy spelling into the modern spec vocabulary, prepares inversion
inputs, builds a PyMC model, samples it, and writes requested outputs.
``run_rhime_from_prepared_inputs`` starts at the same post-preparation boundary
for callers that already hold canonical inputs and retained basis metadata.
The public stage sequence is resolve, prepare, align retained sites,
materialize model inputs, build, sample, and construct the result. Canonical
xarray and Dask inputs remain borrowed until the named materialization
boundary; preparation, sampling, and result stages may perform documented I/O.

Terminology used by the RHIME API:

- ``species`` is the primary gas or tracer name used for object-store lookup
  and output naming.
- ``source`` is the OpenGHG metadata key used to retrieve flux data. In
  sector-resolved inputs it is also the xarray coordinate on flux and
  sensitivity data.
- ``flux_sources`` is the RHIME config/API field containing requested OpenGHG
  flux ``source`` values.
- ``sector_sources`` optionally maps model sector names to OpenGHG flux
  ``source`` values when those labels differ.
- ``sector`` is a model component optimized separately in a multi-sector RHIME
  run, usually backed by one flux ``source``.
- ``tracer`` is an additional species used to constrain the primary species,
  normally with linked forward models. The current RHIME preparation path does
  not support tracer inversions.
- ``emissions_name`` is accepted only as a legacy compatibility spelling when
  ``flux_sources`` is absent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import arviz as az
import pandas as pd
import pymc as pm
import xarray as xr

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
from openghg_inversions.array_ops import to_dense
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.rhime.outputs import (
    RhimeOutputBundle,
    make_multisector_output_bundle,
    make_standard_output_bundle,
)
from . import params as rhime_params
from .params import params_from_config, resolve_flux_sources
from .sampling import RhimeSampler
from .specs import (
    RhimeOutputSpec,
    RhimeRunSpec,
    validate_output_filename_convention,
    validate_output_format,
    validate_output_path_settings,
)
from openghg_inversions.inversion_data import (
    RhimePreparedInputs,
    prepare_rhime_inputs,
)
from openghg_inversions.models import (
    RhimeLikelihoodBuilder,
    RhimeModelSpec,
    SectorSpec,
    build_rhime_model_from_spec,
    build_rhime_multisector_model_from_spec,
    get_rhime_likelihood_result,
)
from openghg_inversions.models._rhime_flux import _select_sector_design
from openghg_inversions.observation_error import resolve_aggregation_error
from openghg_inversions.observation_error import (
    AGGREGATION_ERROR_COVARIANCE,
    AGGREGATION_ERROR_SD,
    DIAGONAL_RESIDUAL_VARIANCE,
    LOW_RANK_FACTOR,
)
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from .builders import (
    RhimeModelBuilder,
    RhimeModelBuilderContext,
    RhimeModelBuildResult,
    callable_metadata,
    validate_model_build_result,
)

__all__ = [
    "SectorSpec",
    "RhimeModelSpec",
    "RhimeModelBuilder",
    "RhimeModelBuilderContext",
    "RhimeModelBuildResult",
    "RhimeOutputSpec",
    "RhimeSampler",
    "RhimeRunSpec",
    "RhimeResult",
    "build_multisector_rhime_model",
    "build_standard_rhime_model",
    "make_multisector_rhime_result",
    "make_standard_rhime_result",
    "materialize_rhime_model_inputs",
    "params_from_config",
    "prepare_rhime_run_inputs",
    "resolve_flux_sources",
    "resolve_rhime_options",
    "run_rhime",
    "run_rhime_from_prepared_inputs",
    "run_rhime_multisector",
    "sample_rhime_model",
    "with_prepared_rhime_sites",
]


@dataclass
class RhimeResult:
    """Modern RHIME run result.

    Args:
        run_spec: Top-level run metadata.
        model_spec: Model specification used to build the PyMC model.
        output_spec: Output settings used by the run.
        inv_inputs: Canonical xarray inversion inputs consumed by the model.
        idata: ArviZ ``InferenceData`` returned by sampling.
        output_metadata: Paths and notes for generated outputs.
        outputs: In-memory derived outputs keyed by output kind.
        model: Built PyMC model.
        inv_out: Modern inversion output object when created.
        basis_functions: Retained flux basis functions from preparation.
        sampler: Sampler settings used by the run.
        model_build_result: Concrete model and explicit role/output manifest.
    """

    run_spec: RhimeRunSpec
    model_spec: RhimeModelSpec
    output_spec: RhimeOutputSpec
    inv_inputs: xr.Dataset
    idata: az.InferenceData
    output_metadata: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    basis_functions: BasisFunctions | None = None
    model: pm.Model | None = None
    inv_out: InversionOutput | None = None
    sampler: RhimeSampler = field(default_factory=RhimeSampler)
    model_build_result: RhimeModelBuildResult | None = None


def _builtin_model_build_result(
    model: pm.Model,
    *,
    model_spec: RhimeModelSpec,
    multisector: bool,
) -> RhimeModelBuildResult:
    """Describe built-in standard and multisector models through the public contract."""
    try:
        likelihood_result = get_rhime_likelihood_result(model)
        likelihood_roles = dict(likelihood_result.variable_roles)
        supported_output_formats = likelihood_result.supported_output_formats
        likelihood_metadata = dict(likelihood_result.metadata)
    except ValueError:
        # Keep test doubles and third-party wrappers around the historical
        # built-in functions backward compatible. Real built-in models always
        # carry the explicit likelihood result.
        likelihood_roles = {"concentration": "y", "model_error": "epsilon"}
        supported_output_formats = ("none", "inv_out", "basic", "paris", "legacy")
        likelihood_metadata = {}

    roles = {
        "observation": "mf",
        "observation_error": "mf_error",
        "minimum_error": "min_error",
        **likelihood_roles,
    }
    if multisector:
        for sector in model_spec.sectors:
            roles[f"flux_scale:{sector.name}"] = f"x_{sector.variable_suffix}"
            roles[f"flux_contribution:{sector.name}"] = f"mu_{sector.variable_suffix}"
            roles[f"emissions_sensitivity:{sector.name}"] = f"hx_{sector.variable_suffix}"
    else:
        roles.update({"flux_scale": "x", "flux_contribution": "mu", "emissions_sensitivity": "hx"})
    if model_spec.use_bc:
        roles.update({"baseline": "mu_bc", "baseline_scale": "bc", "baseline_sensitivity": "hbc"})
    if model_spec.add_offset:
        roles["offset"] = "offset"

    metadata: dict[str, Any] = {
        "kind": "builtin",
        "strategy": model_spec.builder_strategy,
    }
    if likelihood_metadata:
        metadata["likelihood"] = likelihood_metadata

    return RhimeModelBuildResult(
        model=model,
        variable_roles=roles,
        supported_output_formats=cast(tuple[Any, ...], supported_output_formats),
        metadata=metadata,
    )


def _apply_output_bundle(result: RhimeResult, bundle: RhimeOutputBundle) -> None:
    """Apply output helper results to a mutable RHIME result object."""
    if bundle.inv_out is not None:
        result.inv_out = bundle.inv_out
    result.outputs.update(bundle.outputs)
    result.output_metadata.update(bundle.output_metadata)


def resolve_rhime_options(
    *,
    params: Mapping[str, Any],
    multisector: bool,
) -> rhime_params.RhimeRunnerSetup:
    """Resolve raw RHIME options into preparation, model, sampling, and output settings.

    This supported orchestration stage owns normalization and validation only;
    it performs no data access and retains no caller-owned mappings. The
    ``multisector`` choice selects the two intentionally different public
    recipes. The returned setup is the current W2 copy-and-modify handoff and
    may gain fields through an explicitly documented migration.

    Args:
        params: Direct Python or configuration-derived RHIME options.
        multisector: Whether to resolve the multi-sector recipe.

    Returns:
        Normalized run specification, sampler, and preparation arguments.

    Raises:
        ValueError: If required, structured, or mode-specific options are
            invalid.
    """
    timing_start = timer_start()
    setup = rhime_params.make_rhime_runner_setup(params=params, multisector=multisector)
    log_timing("rhime.runner_setup", timer_seconds(timing_start), multisector=multisector)
    return setup


def prepare_rhime_run_inputs(
    data_args: Mapping[str, Any],
    *,
    multisector: bool,
) -> RhimePreparedInputs:
    """Retrieve, filter, basis-transform, and assemble labelled RHIME inputs.

    This supported coarse preparation stage delegates to
    :func:`prepare_rhime_inputs`, whose implementation visibly owns, in order,
    merged-data retrieval or reload, observation filtering with retained-site
    alignment, basis loading or fitting, sensitivity construction, and
    labelled input assembly. It may read OpenGHG stores or local artifacts,
    write requested merged-data and basis artifacts, print progress, emit
    warnings, and eagerly check sensitivity arrays for NaNs. It does not build
    a PyMC model and does not mutate the supplied mapping. Deeper preparation
    stages remain private until their ownership contracts are stabilized.

    Args:
        data_args: Preparation arguments returned by
            :func:`resolve_rhime_options`.
        multisector: Recipe identity recorded in timing metadata.

    Returns:
        Canonical labelled inversion inputs and retained basis metadata.
    """
    timing_start = timer_start()
    prepared = prepare_rhime_inputs(**dict(data_args))
    log_timing(
        "rhime.prepare_inputs",
        timer_seconds(timing_start),
        multisector=multisector,
        nmeasure=prepared.inv_inputs.sizes.get("nmeasure"),
        sites=len(prepared.sites),
        regions=prepared.inv_inputs.sizes.get("region"),
        sources=prepared.inv_inputs.sizes.get("source"),
        basis_source=prepared.basis_artifact_source,
    )
    return prepared


def with_prepared_rhime_sites(
    run_spec: RhimeRunSpec,
    prepared: RhimePreparedInputs,
) -> RhimeRunSpec:
    """Return run metadata aligned to sites retained by preparation.

    This supported pure stage owns the handoff from filtered observations to
    run provenance. It creates a replacement specification, does not mutate
    either argument, and preserves all non-site settings. It exists for the W2
    copied-runner surface and will be migrated explicitly if site ownership
    later moves into a preparation result.

    Args:
        run_spec: Resolved run metadata before data preparation.
        prepared: Prepared inputs with authoritative retained-site metadata.

    Returns:
        A run specification whose sites and averaging periods match
        ``prepared``.
    """
    return replace(
        run_spec,
        sites=tuple(prepared.sites),
        averaging_period=tuple(prepared.averaging_period),
    )


def _validate_multisector_basis_layout(
    basis_functions: BasisFunctions,
    model_spec: RhimeModelSpec,
    inv_inputs: xr.Dataset,
) -> None:
    """Require retained basis indexes to match each prepared sector design."""
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
        source_operator = source_basis.operator
        state_dim = source_operator.meta.state_dim
        basis_index = source_operator.basis_matrix.get_index(state_dim)
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

    if any(not coordinates_match for _, _, _, _, coordinates_match in region_layouts):
        details = ", ".join(
            f"sector {sector!r} -> source {source!r}: basis has {basis_count} regions, "
            f"prepared H has {prepared_count}" + ("" if coordinates_match else " with different coordinates")
            for sector, source, basis_count, prepared_count, coordinates_match in region_layouts
        )
        raise ValueError(
            "Retained source-specific basis state coordinates do not match prepared H; " + details + "."
        )


_MODEL_INPUT_VARIABLES = (
    "H",
    "H_bc",
    "mf",
    "mf_error",
    "min_error",
    "site_indicator",
    AGGREGATION_ERROR_SD,
    AGGREGATION_ERROR_COVARIANCE,
    LOW_RANK_FACTOR,
    DIAGONAL_RESIDUAL_VARIANCE,
)


def materialize_rhime_model_inputs(prepared: RhimePreparedInputs) -> xr.Dataset:
    """Materialize related PyMC input arrays together without mutating preparation.

    This supported backend-boundary stage converts sparse chunk payloads with
    :func:`to_dense`, computes the model-owned arrays in one xarray operation,
    and places them in a shallow dataset copy. The canonical prepared dataset,
    its Dask arrays, and their chunking remain caller-owned and unchanged.
    Coordinates and non-model variables retain their existing eager or lazy
    state. This W2 stage is intentionally PyMC-specific; any later ownership
    migration will preserve a documented compatibility path.

    Args:
        prepared: Canonical RHIME inputs borrowed from preparation or replay.

    Returns:
        A dataset copy whose available model input variables are dense and
        eager, with all other variables shared from ``prepared``.
    """
    timing_start = timer_start()
    inv_inputs = prepared.inv_inputs
    names = [name for name in _MODEL_INPUT_VARIABLES if name in inv_inputs]
    dense_inputs = xr.Dataset({name: to_dense(inv_inputs[name]) for name in names}).compute()
    model_inputs = inv_inputs.copy(
        deep=False,
        data={
            name: dense_inputs[name].data if name in dense_inputs else variable.data
            for name, variable in inv_inputs.data_vars.items()
        },
    )
    log_timing(
        "rhime.model_inputs_materialize",
        timer_seconds(timing_start),
        variables=names,
    )
    return model_inputs


def _validated_custom_model_build(
    model_builder: RhimeModelBuilder,
    *,
    context: RhimeModelBuilderContext,
) -> RhimeModelBuildResult:
    """Call and validate an advanced complete-model builder."""
    model_build_result = model_builder(context)
    if not isinstance(model_build_result, RhimeModelBuildResult):
        raise TypeError(
            "A RHIME model builder must return `RhimeModelBuildResult`; "
            f"got {type(model_build_result).__name__}."
        )
    validate_model_build_result(model_build_result, context=context)
    return model_build_result


def build_standard_rhime_model(
    *,
    prepared: RhimePreparedInputs,
    model_inputs: xr.Dataset,
    run_spec: RhimeRunSpec,
    model_builder: RhimeModelBuilder | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> RhimeModelBuildResult:
    """Build and describe a standard single-sector RHIME PyMC model.

    This supported stage owns the standard graph choice and its output-role
    description. Built-in construction consumes the eager dataset returned by
    :func:`materialize_rhime_model_inputs`; an advanced complete-model builder
    retains the historical context containing canonical ``prepared`` inputs.
    It creates a PyMC graph but performs no sampling or output writes. The
    returned manifest is stage-owned, so copied runners need not construct one.

    Args:
        prepared: Canonical prepared inputs retained for advanced builders and
            later outputs.
        model_inputs: Materialized PyMC inputs.
        run_spec: Retained-site-aligned run specification.
        model_builder: Optional advanced complete-model builder.
        likelihood_builder: Optional likelihood builder used by the built-in
            model.

    Returns:
        Built model with validated roles and output capabilities.

    Raises:
        TypeError: If a complete-model builder returns the wrong type.
        ValueError: If builder roles or output capabilities are invalid.
    """
    timing_start = timer_start()
    builder_context = RhimeModelBuilderContext(
        prepared_inputs=prepared,
        run_spec=run_spec,
        multisector=False,
    )
    if model_builder is not None:
        model_build_result = _validated_custom_model_build(
            model_builder,
            context=builder_context,
        )
    else:
        model = build_rhime_model_from_spec(
            model_inputs,
            run_spec.model,
            **({} if likelihood_builder is None else {"likelihood_builder": likelihood_builder}),
        )
        model_build_result = _builtin_model_build_result(
            model,
            model_spec=run_spec.model,
            multisector=False,
        )
    if likelihood_builder is not None:
        validate_model_build_result(model_build_result, context=builder_context)
    log_timing("rhime.model_build", timer_seconds(timing_start), multisector=False)
    return model_build_result


def build_multisector_rhime_model(
    *,
    prepared: RhimePreparedInputs,
    model_inputs: xr.Dataset,
    run_spec: RhimeRunSpec,
    model_builder: RhimeModelBuilder | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> RhimeModelBuildResult:
    """Build and describe a source-resolved multi-sector RHIME PyMC model.

    This supported stage visibly owns the multi-sector-only retained-basis
    layout check and graph builder. Built-in construction consumes materialized
    inputs; an advanced complete-model builder retains the historical canonical
    prepared-input context. It creates no samples or files, and returns the
    role/output manifest needed by later supported stages so callers never
    assemble that manifest themselves. Migration of this W2 surface will be
    explicit and documented.

    Args:
        prepared: Canonical prepared inputs and source-specific basis metadata.
        model_inputs: Materialized PyMC inputs.
        run_spec: Retained-site-aligned multi-sector run specification.
        model_builder: Optional advanced complete-model builder.
        likelihood_builder: Optional likelihood builder used by the built-in
            model.

    Returns:
        Built multi-sector model with validated roles and capabilities.

    Raises:
        TypeError: If a complete-model builder returns the wrong type.
        ValueError: If basis state coordinates or builder contracts disagree.
    """
    timing_start = timer_start()
    _validate_multisector_basis_layout(
        prepared.basis_functions,
        run_spec.model,
        prepared.inv_inputs,
    )
    builder_context = RhimeModelBuilderContext(
        prepared_inputs=prepared,
        run_spec=run_spec,
        multisector=True,
    )
    if model_builder is not None:
        model_build_result = _validated_custom_model_build(
            model_builder,
            context=builder_context,
        )
    else:
        model = build_rhime_multisector_model_from_spec(
            model_inputs,
            run_spec.model,
            **({} if likelihood_builder is None else {"likelihood_builder": likelihood_builder}),
        )
        model_build_result = _builtin_model_build_result(
            model,
            model_spec=run_spec.model,
            multisector=True,
        )
    if likelihood_builder is not None:
        validate_model_build_result(model_build_result, context=builder_context)
    log_timing("rhime.model_build", timer_seconds(timing_start), multisector=True)
    return model_build_result


def sample_rhime_model(
    model_build_result: RhimeModelBuildResult,
    sampler: RhimeSampler,
    *,
    use_variable_roles: bool = False,
) -> az.InferenceData:
    """Sample one built RHIME model using supported sampler mechanics.

    This supported stage owns PyMC execution and predictive sampling side
    effects. ``use_variable_roles`` preserves the advanced builder/likelihood
    behavior while ordinary built-in models retain the historical sampler call.
    It does not mutate prepared inputs or write RHIME products. This W2 stage
    will retain a migration path if sampling ownership later changes.

    Args:
        model_build_result: Model and stage-owned semantic roles.
        sampler: Sampling and predictive settings.
        use_variable_roles: Whether predictive names should resolve through
            the build result's role mapping.

    Returns:
        Sampled ArviZ inference data.
    """
    timing_start = timer_start()
    if use_variable_roles:
        idata = sampler.sample(
            model_build_result.model,
            variable_roles=model_build_result.variable_roles,
        )
    else:
        idata = sampler.sample(model_build_result.model)

    log_timing(
        "rhime.sampler_total",
        timer_seconds(timing_start),
        draws=sampler.draws,
        burn=sampler.burn,
        tune=sampler.tune,
        chains=sampler.chains,
        nuts_sampler=sampler.nuts_sampler,
    )
    return idata


def _persisted_builder_metadata(
    model_build_result: RhimeModelBuildResult,
    *,
    model_builder: RhimeModelBuilder | None,
    likelihood_builder: RhimeLikelihoodBuilder | None,
) -> dict[str, Any]:
    """Return serializable builder provenance for output helpers."""
    metadata = dict(model_build_result.metadata)
    if model_builder is not None:
        metadata["model_builder"] = callable_metadata(model_builder)
    if likelihood_builder is not None:
        metadata["likelihood_builder"] = callable_metadata(likelihood_builder)
    return metadata


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
    """Construct and write requested standard RHIME outputs.

    This supported terminal stage owns ``RhimeResult`` construction and the
    standard output bundle, including trace and product filesystem writes. It
    preserves canonical prepared inputs in the result and records safe builder
    provenance. Callers supply stage products but never construct role or
    output manifests. The W2 signature is supported with explicit migration if
    output ownership later moves.

    Args:
        prepared: Canonical prepared inputs retained in outputs.
        run_spec: Retained-site-aligned standard run specification.
        sampler: Sampler settings used for the run.
        model_build_result: Built model and stage-owned output roles.
        idata: Sampled inference data.
        build_and_sample_seconds: Combined build and sampling duration.
        model_builder: Optional advanced builder, used only for provenance.
        likelihood_builder: Optional likelihood builder, used only for
            provenance.

    Returns:
        Complete standard RHIME result and requested products.
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
        result.output_metadata["model_builder"] = callable_metadata(model_builder)
    if likelihood_builder is not None:
        result.output_metadata["likelihood_builder"] = callable_metadata(likelihood_builder)

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
        builder_metadata=_persisted_builder_metadata(
            model_build_result,
            model_builder=model_builder,
            likelihood_builder=likelihood_builder,
        ),
    )
    log_timing(
        "rhime.output_bundle_total",
        timer_seconds(timing_start),
        multisector=False,
        output_format=run_spec.output.output_format,
    )
    _apply_output_bundle(result, output_bundle)
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
    """Construct and write requested multi-sector RHIME outputs.

    This supported terminal stage owns multi-sector diagnostics and output
    differences; unlike the standard stage it does not pass sampler metadata
    into the output bundle. It may write requested products and mutates only
    the newly created result. Callers consume the build-stage manifest without
    constructing role or output mappings. This W2 API will migrate explicitly
    if result ownership changes.

    Args:
        prepared: Canonical multi-sector inputs and retained basis metadata.
        run_spec: Retained-site-aligned multi-sector run specification.
        sampler: Sampler settings used for the run.
        model_build_result: Built model and stage-owned output roles.
        idata: Sampled inference data.
        build_and_sample_seconds: Combined build and sampling duration.
        model_builder: Optional advanced builder, used only for provenance.
        likelihood_builder: Optional likelihood builder, used only for
            provenance.

    Returns:
        Complete multi-sector result, products, and sector diagnostics.
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
        result.output_metadata["model_builder"] = callable_metadata(model_builder)
    if likelihood_builder is not None:
        result.output_metadata["likelihood_builder"] = callable_metadata(likelihood_builder)

    timing_start = timer_start()
    output_bundle = make_multisector_output_bundle(
        output_spec=run_spec.output,
        run_spec=run_spec,
        model_spec=run_spec.model,
        idata=idata,
        prepared=prepared,
        country_file=run_spec.output.country_file,
        variable_roles=model_build_result.variable_roles,
        builder_metadata=_persisted_builder_metadata(
            model_build_result,
            model_builder=model_builder,
            likelihood_builder=likelihood_builder,
        ),
    )
    log_timing(
        "rhime.output_bundle_total",
        timer_seconds(timing_start),
        multisector=True,
        output_format=run_spec.output.output_format,
    )
    _apply_output_bundle(result, output_bundle)

    return result


def run_rhime_from_prepared_inputs(
    *,
    prepared_inputs: RhimePreparedInputs,
    run_spec: RhimeRunSpec,
    sampler: RhimeSampler | None = None,
    model_builder: RhimeModelBuilder | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> RhimeResult:
    """Run RHIME directly from previously prepared inversion inputs.

    This entry point bypasses RHIME configuration normalization and OpenGHG
    data preparation. A model with one sector uses the standard builder; a
    model with two or more sectors uses the multi-sector builder. The prepared
    ``H`` source layout, run-spec layout flag, model sector count, and output
    settings are validated before model construction or sampling.

    Args:
        prepared_inputs: Canonical inversion inputs and retained preparation
            metadata to consume directly.
        run_spec: Model, output, and run metadata for the inversion. Retained
            sites and averaging periods are replaced with values from
            ``prepared_inputs``.
        sampler: Sampling settings to use. Defaults to a new ``RhimeSampler``.
        model_builder: Optional complete direct-Python model factory. It
            receives :class:`RhimeModelBuilderContext` and returns a model,
            explicit variable roles, compatible output formats, and metadata.
        likelihood_builder: Optional complete observation-component builder
            used inside the built-in standard or multisector model. It owns
            both error construction and the observed distribution.

    Returns:
        Modern RHIME result containing the built model, sampled trace, and
        requested outputs.

    Raises:
        ValueError: If both builder seams are supplied, the model specification
            contains no sectors, the sector
            count, prepared ``H`` layout, and prepared-data layout flag
            disagree, or output settings are invalid.
    """
    if model_builder is not None and likelihood_builder is not None:
        raise ValueError("Pass either `model_builder` or `likelihood_builder`, not both.")
    prepared_inputs = prepared_inputs.validated()
    sector_count = len(run_spec.model.sectors)
    if sector_count < 1:
        raise ValueError(f"`run_spec.model.sectors` must contain at least one sector; found {sector_count}.")
    multisector = sector_count > 1
    sensitivity = prepared_inputs.inv_inputs["H"]
    source_coord = sensitivity.coords.get("source")
    state_dims = [str(dim) for dim in sensitivity.dims if dim != "nmeasure"]
    gathered_source = False
    if source_coord is not None and len(state_dims) == 1 and source_coord.dims == (state_dims[0],):
        source_index = sensitivity.indexes.get(state_dims[0])
        gathered_source = isinstance(source_index, pd.MultiIndex) and "source" in source_index.names
    prepared_is_multisector = "source" in sensitivity.dims or gathered_source
    if run_spec.split_by_sectors is not prepared_is_multisector:
        raise ValueError(
            "`run_spec.split_by_sectors` must agree with the prepared `H` layout: "
            f"split_by_sectors={run_spec.split_by_sectors}, "
            f"multisector source layout present={prepared_is_multisector}."
        )
    if run_spec.split_by_sectors is not multisector:
        raise ValueError(
            "`run_spec.split_by_sectors` must agree with the model sector count: "
            f"found {sector_count} sector(s) and split_by_sectors={run_spec.split_by_sectors}."
        )

    output_spec = run_spec.output
    validate_output_format(output_spec.output_format)
    aggregation_error = resolve_aggregation_error(
        prepared_inputs.inv_inputs,
        run_spec.model.aggregation_error_mode,
    )
    if aggregation_error.mode != "none" and output_spec.output_format in {
        "basic",
        "paris",
        "legacy",
    }:
        raise ValueError(
            "RHIME aggregation-error covariance is not yet supported by derived "
            f"output_format={output_spec.output_format!r}; use 'inv_out' or 'none' until "
            "the postprocessing reconstruction follow-up lands."
        )
    validate_output_filename_convention(output_spec.output_filename_convention)
    validate_output_path_settings(
        output_format=output_spec.output_format,
        output_path=output_spec.output_path,
        save_trace=output_spec.save_trace,
        save_inversion_output=output_spec.save_inversion_output,
        multisector=multisector,
    )

    run_spec = with_prepared_rhime_sites(run_spec, prepared_inputs)
    # Complete-model builders retain their historical ownership of canonical,
    # potentially lazy prepared inputs.  Only the built-in model crosses the
    # named PyMC materialization boundary here.
    model_inputs = (
        prepared_inputs.inv_inputs
        if model_builder is not None
        else materialize_rhime_model_inputs(prepared_inputs)
    )
    active_sampler = RhimeSampler() if sampler is None else sampler
    build_and_sample_start = timer_start()

    if multisector:
        model_build_result = build_multisector_rhime_model(
            prepared=prepared_inputs,
            model_inputs=model_inputs,
            run_spec=run_spec,
            model_builder=model_builder,
            likelihood_builder=likelihood_builder,
        )
    else:
        model_build_result = build_standard_rhime_model(
            prepared=prepared_inputs,
            model_inputs=model_inputs,
            run_spec=run_spec,
            model_builder=model_builder,
            likelihood_builder=likelihood_builder,
        )

    idata = sample_rhime_model(
        model_build_result,
        active_sampler,
        use_variable_roles=model_builder is not None or likelihood_builder is not None,
    )
    build_and_sample_seconds = timer_seconds(build_and_sample_start)

    if multisector:
        return make_multisector_rhime_result(
            prepared=prepared_inputs,
            run_spec=run_spec,
            sampler=active_sampler,
            model_build_result=model_build_result,
            idata=idata,
            build_and_sample_seconds=build_and_sample_seconds,
            model_builder=model_builder,
            likelihood_builder=likelihood_builder,
        )
    return make_standard_rhime_result(
        prepared=prepared_inputs,
        run_spec=run_spec,
        sampler=active_sampler,
        model_build_result=model_build_result,
        idata=idata,
        build_and_sample_seconds=build_and_sample_seconds,
        model_builder=model_builder,
        likelihood_builder=likelihood_builder,
    )


def run_rhime(
    *,
    config_file: str | Path | None = None,
    **kwargs: Any,
) -> RhimeResult:
    """Run a standard single-sector RHIME inversion.

    Args:
        config_file: Optional INI configuration file. Values in ``kwargs``
            override values read from this file.
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
        ValueError: If required parameters are missing, unsupported parameters
            are supplied, or the flux-source count is invalid.
    """
    params = (
        params_from_config(config_file, extra_kwargs=kwargs, normalise=False)
        if config_file is not None
        else dict(kwargs)
    )
    setup = resolve_rhime_options(params=params, multisector=False)

    # Preparation visibly follows the scientific order implemented by the
    # supported coarse stage: retrieve/reload merged data; filter observations
    # while retaining aligned site metadata; load/fit the basis; construct
    # sensitivities; and assemble labelled inversion inputs.
    prepared = prepare_rhime_run_inputs(setup.data_args, multisector=False)
    run_spec = with_prepared_rhime_sites(setup.run_spec, prepared)

    # Cross the explicit eager PyMC boundary without changing canonical inputs.
    model_inputs = materialize_rhime_model_inputs(prepared)
    build_and_sample_start = timer_start()
    model_build_result = build_standard_rhime_model(
        prepared=prepared,
        model_inputs=model_inputs,
        run_spec=run_spec,
    )
    idata = sample_rhime_model(model_build_result, setup.sampler)

    return make_standard_rhime_result(
        prepared=prepared,
        run_spec=run_spec,
        sampler=setup.sampler,
        model_build_result=model_build_result,
        idata=idata,
        build_and_sample_seconds=timer_seconds(build_and_sample_start),
    )


def run_rhime_multisector(
    *,
    config_file: str | Path | None = None,
    **kwargs: Any,
) -> RhimeResult:
    """Run a shared-basis multi-sector RHIME inversion.

    Args:
        config_file: Optional INI configuration file. Values in ``kwargs``
            override values read from this file.
        **kwargs: RHIME run parameters using snake-case names. Multi-sector
            runs require at least two ``flux_sources`` and may include
            a complete ``sector_priors`` mapping keyed by sector name. When
            model sector labels differ from OpenGHG source values, pass
            ``sector_sources`` as a one-to-one mapping from sector name to one
            unique value in ``flux_sources``. Legacy ``emissions_name`` is
            accepted only as a compatibility alias when ``flux_sources`` is
            absent.

    Returns:
        Modern RHIME result containing canonical inputs, InferenceData, specs,
        output metadata, and sector diagnostics.

    Raises:
        ValueError: If required parameters are missing, unsupported parameters
            are supplied, or fewer than two flux sources are provided.
    """
    params = (
        params_from_config(config_file, extra_kwargs=kwargs, normalise=False)
        if config_file is not None
        else dict(kwargs)
    )
    setup = resolve_rhime_options(params=params, multisector=True)

    # Multi-sector preparation retains source-resolved flux sensitivities after
    # the same visible retrieve/reload, filter/site alignment, basis, and
    # labelled-assembly sequence as the standard recipe.
    prepared = prepare_rhime_run_inputs(setup.data_args, multisector=True)
    run_spec = with_prepared_rhime_sites(setup.run_spec, prepared)

    model_inputs = materialize_rhime_model_inputs(prepared)
    build_and_sample_start = timer_start()
    model_build_result = build_multisector_rhime_model(
        prepared=prepared,
        model_inputs=model_inputs,
        run_spec=run_spec,
    )
    idata = sample_rhime_model(model_build_result, setup.sampler)

    # Multi-sector output owns sector diagnostics and its distinct format
    # constraints; it is intentionally not hidden behind the standard stage.
    return make_multisector_rhime_result(
        prepared=prepared,
        run_spec=run_spec,
        sampler=setup.sampler,
        model_build_result=model_build_result,
        idata=idata,
        build_and_sample_seconds=timer_seconds(build_and_sample_start),
    )
