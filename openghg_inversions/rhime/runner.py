"""Modern public RHIME runner implementation.

This module accepts Python keyword arguments or RHIME ``.ini`` files,
normalizes legacy spelling into the modern spec vocabulary, prepares inversion
inputs, builds a PyMC model, samples it, and writes requested outputs.

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
from typing import Any

import arviz as az
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
from .specs import RhimeOutputSpec, RhimeRunSpec
from openghg_inversions.inversion_data import (
    RhimePreparedInputs,
    prepare_rhime_inputs,
)
from openghg_inversions.models import (
    RhimeModelSpec,
    SectorSpec,
    build_rhime_model_from_spec,
    build_rhime_multisector_model_from_spec,
)
from openghg_inversions.postprocessing.inversion_output import InversionOutput

__all__ = [
    "SectorSpec",
    "RhimeModelSpec",
    "RhimeOutputSpec",
    "RhimeSampler",
    "RhimeRunSpec",
    "RhimeResult",
    "params_from_config",
    "resolve_flux_sources",
    "run_rhime",
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
    run_spec = _run_spec_with_prepared_inputs(setup.run_spec, prepared)

    build_and_sample_start = timer_start()
    timing_start = timer_start()
    if multisector:
        model = build_rhime_multisector_model_from_spec(prepared.inv_inputs, run_spec.model)
    else:
        model = build_rhime_model_from_spec(prepared.inv_inputs, run_spec.model)
    log_timing("rhime.model_build", timer_seconds(timing_start), multisector=multisector)

    timing_start = timer_start()
    idata = setup.sampler.sample(model)
    log_timing(
        "rhime.sampler_total",
        timer_seconds(timing_start),
        draws=setup.sampler.draws,
        burn=setup.sampler.burn,
        tune=setup.sampler.tune,
        chains=setup.sampler.chains,
        nuts_sampler=setup.sampler.nuts_sampler,
    )
    result = RhimeResult(
        run_spec=run_spec,
        model_spec=run_spec.model,
        output_spec=run_spec.output,
        inv_inputs=prepared.inv_inputs,
        idata=idata,
        sampler=setup.sampler,
        model=model,
        basis_functions=prepared.basis_functions,
        output_metadata={"build_and_sample_seconds": timer_seconds(build_and_sample_start)},
    )

    timing_start = timer_start()
    if multisector:
        output_bundle = make_multisector_output_bundle(
            output_spec=run_spec.output,
            run_spec=run_spec,
            model_spec=run_spec.model,
            idata=idata,
            prepared=prepared,
            country_file=run_spec.output.country_file,
        )
    else:
        output_bundle = make_standard_output_bundle(
            output_spec=run_spec.output,
            run_spec=run_spec,
            model_spec=run_spec.model,
            idata=idata,
            prepared=prepared,
            country_file=run_spec.output.country_file,
            sampler=setup.sampler,
        )
    log_timing(
        "rhime.output_bundle_total",
        timer_seconds(timing_start),
        multisector=multisector,
        output_format=run_spec.output.output_format,
    )
    _apply_output_bundle(result, output_bundle)

    return result


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
            ``sector_priors`` keyed by sector name. When model sector labels
            differ from OpenGHG source values, pass ``sector_sources`` as a
            mapping from sector name to one value in ``flux_sources``. Legacy
            ``emissions_name`` is accepted only as a compatibility alias when
            ``flux_sources`` is absent.

    Returns:
        Modern RHIME result containing canonical inputs, InferenceData, specs,
        output metadata, and sector diagnostics.

    Raises:
        ValueError: If required parameters are missing, unsupported parameters
            are supplied, or fewer than two flux sources are provided.
    """
    params = params_from_config(config_file, extra_kwargs=kwargs) if config_file is not None else dict(kwargs)
    return _run_common(multisector=True, params=params)
