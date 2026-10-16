"""Retained multisector entry point for already prepared scientific inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd

from openghg_inversions._timing import timer_seconds, timer_start
from openghg_inversions.inversion_data import RhimePreparedInputs

from .builders import RhimeLikelihoodBuilder, RhimeModelBuilder
from .materialization import materialize_pymc_inputs
from .multisector import build_multisector_rhime_model_result, make_multisector_rhime_result
from .multisector import multisector_model_input_names
from .outputs import (
    RhimeResult,
    make_multisector_rhime_outputs,
    annotate_sample_trace,
    validate_output_resources,
)
from .preparation import with_prepared_rhime_sites
from openghg_inversions.inference.sampling import resolve_sampler
from .sampling import RhimeSampler, sample_rhime_model
from .specs import (
    RhimeRunSpec,
    validate_output_filename_convention,
    validate_output_format,
    validate_output_path_settings,
)


def _require_prepared_layout(prepared: RhimePreparedInputs, *, multisector: bool) -> None:
    """Check externally supplied sensitivity layout against the selected recipe."""
    sensitivity = prepared.inv_inputs["H"]
    source = sensitivity.coords.get("source")
    state_dims = [dim for dim in sensitivity.dims if dim != "nmeasure"]
    gathered = (
        source is not None and len(state_dims) == 1 and source.dims == (state_dims[0],)
        and isinstance(sensitivity.indexes.get(state_dims[0]), pd.MultiIndex)
        and "source" in sensitivity.indexes[state_dims[0]].names
    )
    if ("source" in sensitivity.dims or gathered) != multisector:
        raise ValueError("The prepared H layout conflicts with the selected standard/multisector recipe.")


def run_rhime_from_prepared_inputs(
    *,
    prepared_inputs: RhimePreparedInputs,
    run_spec: RhimeRunSpec,
    sampler: RhimeSampler | Literal["default"] = "default",
    model_builder: RhimeModelBuilder | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
) -> RhimeResult:
    """Run a retained multisector recipe from prepared scientific inputs.

    Standard callers use ``run_standard(prepared=..., config=...)`` so prepared
    and acquired standard execution share the same scientific pipeline.

    Args:
        prepared_inputs: Borrowed canonical inputs and retained basis functions.
        run_spec: Multisector model, output and run settings.
        sampler: Sampler configuration or ``"default"`` for fresh defaults.
        model_builder: Optional complete multisector model factory.
        likelihood_builder: Optional custom likelihood for the built-in graph.
        likelihood_kwargs: Options specific to the custom likelihood.

    Returns:
        Retained multisector result envelope.

    Raises:
        ValueError: If the layout, extension points or output settings conflict.
        TypeError: If the sampler or model-builder result has an invalid type.
        FileNotFoundError: If a selected reporting resource is missing.
    """
    active_sampler = resolve_sampler(sampler)
    if not run_spec.split_by_sectors:
        raise ValueError("Standard prepared execution uses run_standard(prepared=..., config=...).")
    if model_builder is not None and likelihood_builder is not None:
        raise ValueError("Pass either `model_builder` or `likelihood_builder`, not both.")
    if likelihood_kwargs and likelihood_builder is None:
        raise ValueError("Non-empty `likelihood_kwargs` require an active `likelihood_builder`.")
    if likelihood_builder is not None and run_spec.model.likelihood is not None:
        raise ValueError("A custom likelihood cannot be combined with built-in likelihood settings.")
    prepared_inputs = prepared_inputs.validated()
    sector_count = len(run_spec.model.sectors)
    if sector_count < 1:
        raise ValueError(f"`run_spec.model.sectors` must contain at least one sector; found {sector_count}.")
    multisector = sector_count > 1

    _require_prepared_layout(prepared_inputs, multisector=run_spec.split_by_sectors)
    if run_spec.split_by_sectors is not multisector:
        raise ValueError(
            "`run_spec.split_by_sectors` must agree with the model sector count: "
            f"found {sector_count} sector(s) and split_by_sectors={run_spec.split_by_sectors}."
        )

    output_spec = run_spec.output
    validate_output_format(output_spec.output_format)
    validate_output_filename_convention(output_spec.output_filename_convention)
    validate_output_path_settings(
        output_format=output_spec.output_format,
        output_path=output_spec.output_path,
        save_trace=output_spec.save_trace,
        save_inversion_output=output_spec.save_inversion_output,
        multisector=multisector,
    )
    validate_output_resources(output_spec)
    if model_builder is None and likelihood_builder is None and run_spec.model.likelihood is None:
        raise ValueError("A prepared RHIME run requires a model builder or likelihood.")

    run_spec = with_prepared_rhime_sites(run_spec, prepared_inputs)
    build_and_sample_start = timer_start()
    # Complete-model builders historically own canonical, potentially lazy
    # inputs. Only built-in graphs cross the named eager PyMC boundary here.
    model_inputs = (
        prepared_inputs.inv_inputs
        if model_builder is not None
        else materialize_pymc_inputs(
            prepared_inputs,
            variable_names=multisector_model_input_names(prepared_inputs, run_spec.model),
        )
    )
    model_build_result = build_multisector_rhime_model_result(
        prepared=prepared_inputs,
        model_inputs=model_inputs,
        run_spec=run_spec,
        model_builder=model_builder,
        likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs,
    )
    if model_builder is None:
        model_build_result.output_contract.validate_requested_output(output_spec.output_format)
    idata = sample_rhime_model(
        model_build_result,
        active_sampler,
    )
    annotate_sample_trace(model_build_result, idata)
    build_and_sample_seconds = timer_seconds(build_and_sample_start)

    result = make_multisector_rhime_result(
        prepared=prepared_inputs,
        run_spec=run_spec,
        sampler=active_sampler,
        model_build_result=model_build_result,
        idata=idata,
        build_and_sample_seconds=build_and_sample_seconds,
        model_builder=model_builder,
        likelihood_builder=likelihood_builder,
        likelihood_kwargs=likelihood_kwargs,
    )
    make_multisector_rhime_outputs(result=result, prepared=prepared_inputs)
    return result
