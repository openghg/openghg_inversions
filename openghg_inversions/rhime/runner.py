"""Modern public RHIME runner implementation.

This module accepts Python keyword arguments or RHIME ``.ini`` files,
normalizes legacy spelling into the modern spec vocabulary, prepares inversion
inputs, builds a PyMC model, samples it, and writes requested outputs.
``run_rhime_from_prepared_inputs`` starts at the same post-preparation boundary
for callers that already hold canonical inputs and retained basis metadata.

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

from dataclasses import dataclass, field, replace
import inspect
from pathlib import Path
from typing import Any, cast

import arviz as az
import pandas as pd
import pymc as pm
import xarray as xr

from openghg_inversions._timing import log_timing, timer_seconds, timer_start
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
    "params_from_config",
    "resolve_flux_sources",
    "run_rhime",
    "run_rhime_from_prepared_inputs",
    "run_rhime_multisector",
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


def _make_rhime_runner_setup(
    *,
    params: dict[str, Any],
    multisector: bool,
) -> rhime_params.RhimeRunnerSetup:
    """Normalize raw RHIME parameters into specs and preparation arguments."""
    return rhime_params.make_rhime_runner_setup(
        params=params,
        multisector=multisector,
        data_param_names=set(inspect.signature(prepare_rhime_inputs).parameters),
    )


def _run_spec_with_prepared_inputs(
    run_spec: RhimeRunSpec,
    prepared: RhimePreparedInputs,
) -> RhimeRunSpec:
    """Update run metadata with retained sites from prepared RHIME inputs."""
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
            f"prepared H has {prepared_count}"
            + ("" if coordinates_match else " with different coordinates")
            for sector, source, basis_count, prepared_count, coordinates_match in region_layouts
        )
        raise ValueError(
            "Retained source-specific basis state coordinates do not match prepared H; "
            + details
            + "."
        )


def _execute_prepared_rhime(
    *,
    prepared: RhimePreparedInputs,
    run_spec: RhimeRunSpec,
    sampler: RhimeSampler,
    multisector: bool,
    model_builder: RhimeModelBuilder | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> RhimeResult:
    """Build, sample, and produce outputs from prepared RHIME inputs."""
    run_spec = _run_spec_with_prepared_inputs(run_spec, prepared)

    build_and_sample_start = timer_start()
    timing_start = timer_start()
    if multisector:
        _validate_multisector_basis_layout(
            prepared.basis_functions,
            run_spec.model,
            prepared.inv_inputs,
        )
    builder_context = RhimeModelBuilderContext(
        prepared_inputs=prepared,
        run_spec=run_spec,
        multisector=multisector,
    )
    if model_builder is not None:
        model_build_result = model_builder(builder_context)
        if not isinstance(model_build_result, RhimeModelBuildResult):
            raise TypeError(
                "A RHIME model builder must return `RhimeModelBuildResult`; "
                f"got {type(model_build_result).__name__}."
            )
        validate_model_build_result(model_build_result, context=builder_context)
        model = model_build_result.model
    elif multisector:
        model = build_rhime_multisector_model_from_spec(
            prepared.inv_inputs,
            run_spec.model,
            **({} if likelihood_builder is None else {"likelihood_builder": likelihood_builder}),
        )
        model_build_result = _builtin_model_build_result(
            model,
            model_spec=run_spec.model,
            multisector=True,
        )
    else:
        model = build_rhime_model_from_spec(
            prepared.inv_inputs,
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
    persisted_builder_metadata = dict(model_build_result.metadata)
    if model_builder is not None:
        persisted_builder_metadata["model_builder"] = callable_metadata(model_builder)
    if likelihood_builder is not None:
        persisted_builder_metadata["likelihood_builder"] = callable_metadata(likelihood_builder)
    log_timing("rhime.model_build", timer_seconds(timing_start), multisector=multisector)

    timing_start = timer_start()
    if model_builder is None and likelihood_builder is None:
        idata = sampler.sample(model)
    else:
        idata = sampler.sample(model, variable_roles=model_build_result.variable_roles)
    log_timing(
        "rhime.sampler_total",
        timer_seconds(timing_start),
        draws=sampler.draws,
        burn=sampler.burn,
        tune=sampler.tune,
        chains=sampler.chains,
        nuts_sampler=sampler.nuts_sampler,
    )
    result = RhimeResult(
        run_spec=run_spec,
        model_spec=run_spec.model,
        output_spec=run_spec.output,
        inv_inputs=prepared.inv_inputs,
        idata=idata,
        sampler=sampler,
        model=model,
        basis_functions=prepared.basis_functions,
        model_build_result=model_build_result,
        output_metadata={"build_and_sample_seconds": timer_seconds(build_and_sample_start)},
    )
    if model_builder is not None:
        result.output_metadata["model_builder"] = callable_metadata(model_builder)
    if likelihood_builder is not None:
        result.output_metadata["likelihood_builder"] = callable_metadata(likelihood_builder)

    timing_start = timer_start()
    if multisector:
        output_bundle = make_multisector_output_bundle(
            output_spec=run_spec.output,
            run_spec=run_spec,
            model_spec=run_spec.model,
            idata=idata,
            prepared=prepared,
            country_file=run_spec.output.country_file,
            variable_roles=model_build_result.variable_roles,
            builder_metadata=persisted_builder_metadata,
        )
    else:
        output_bundle = make_standard_output_bundle(
            output_spec=run_spec.output,
            run_spec=run_spec,
            model_spec=run_spec.model,
            idata=idata,
            prepared=prepared,
            country_file=run_spec.output.country_file,
            sampler=sampler,
            variable_roles=model_build_result.variable_roles,
            builder_metadata=persisted_builder_metadata,
        )
    log_timing(
        "rhime.output_bundle_total",
        timer_seconds(timing_start),
        multisector=multisector,
        output_format=run_spec.output.output_format,
    )
    _apply_output_bundle(result, output_bundle)

    return result


def _run_common(
    *,
    multisector: bool,
    params: dict[str, Any],
) -> RhimeResult:
    """Run the shared RHIME pipeline after public wrapper/config normalization."""
    timing_start = timer_start()
    setup = _make_rhime_runner_setup(params=params, multisector=multisector)
    log_timing("rhime.runner_setup", timer_seconds(timing_start), multisector=multisector)

    timing_start = timer_start()
    prepared = prepare_rhime_inputs(**setup.data_args)
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
    return _execute_prepared_rhime(
        prepared=prepared,
        run_spec=setup.run_spec,
        sampler=setup.sampler,
        multisector=multisector,
    )


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

    return _execute_prepared_rhime(
        prepared=prepared_inputs,
        run_spec=run_spec,
        sampler=RhimeSampler() if sampler is None else sampler,
        multisector=multisector,
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
    params = params_from_config(config_file, extra_kwargs=kwargs) if config_file is not None else dict(kwargs)
    return _run_common(multisector=False, params=params)


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
    params = params_from_config(config_file, extra_kwargs=kwargs) if config_file is not None else dict(kwargs)
    return _run_common(multisector=True, params=params)
