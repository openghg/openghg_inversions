"""Advanced RHIME entry point for already prepared scientific inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd

from openghg_inversions._timing import timer_seconds, timer_start, timed
from openghg_inversions._provenance import installed_ogi_provenance
from openghg_inversions.serialization import save_trace
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from contextlib import nullcontext
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.observation_error import (
    select_aggregation_error_mode,
)

from .builders import RhimeLikelihoodBuilder, RhimeModelBuilder
from .materialization import materialize_pymc_inputs
from .multisector import build_multisector_rhime_model_result, make_multisector_rhime_result
from .multisector import multisector_model_input_names
from .outputs import (
    RhimeResult,
    make_multisector_rhime_outputs,
    make_standard_output,
    write_standard_outputs,
    standard_artifact_paths,
    annotate_sample_trace,
    validate_output_resources,
)
from .preparation import with_prepared_rhime_sites
from openghg_inversions.inference.sampling import resolve_sampler
from .sampling import RhimeSampler, sample_rhime_model
from .specs import (
    RhimeRunSpec,
    RhimeModelSpec,
    RhimeOutputSpec,
    validate_output_filename_convention,
    validate_output_format,
    validate_output_path_settings,
)
from .standard import (
    build_standard_model,
    materialize_standard_inputs,
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
    model_spec: RhimeModelSpec | None = None,
    output_spec: RhimeOutputSpec | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    run_spec: RhimeRunSpec | None = None,
    sampler: RhimeSampler | Literal["default"] = "default",
    model_builder: RhimeModelBuilder | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
    likelihood_kwargs: Mapping[str, Any] | None = None,
) -> InversionOutput | RhimeResult:
    """Build, sample, construct a result, and output a prepared RHIME run.

    This advanced route intentionally starts after retrieval, filtering, basis
    construction, sensitivity construction, and labelled-input assembly. It
    validates the retained scientific layout before crossing the PyMC
    materialization boundary. Selected reporting resources are checked before
    materialization or sampling.

    Args:
        prepared_inputs: Validated canonical inversion inputs and retained
            basis functions.
        model_spec: Standard scientific model choices.
        output_spec: Standard output policy, separate from model construction.
        start_date: Requested standard inversion start.
        end_date: Requested standard inversion end.
        run_spec: Retained multisector compatibility input only; standard callers
            supply the four explicit arguments above.
        sampler: Use ``"default"`` to request a fresh ``RhimeSampler()``.
            Explicit ``None`` is invalid; omit the argument to select defaults.
        model_builder: Complete-model callable for multisector graphs only.
            Standard runs use ``likelihood_builder`` for likelihood customization.
        likelihood_builder: Optional ordinary likelihood callable used with
            the built-in graph.
        likelihood_kwargs: Options specific to the custom likelihood.

    Returns:
        Standard scientific output, or the retained multisector result envelope.

    Raises:
        ValueError: If layouts, extension points, aggregation error, or output
            settings are inconsistent.
        TypeError: If an extension point has an invalid callable contract, or
            the sampler is neither a configuration nor ``"default"``.
        FileNotFoundError: If a selected country file is missing or is a directory.
    """
    active_sampler = resolve_sampler(sampler)
    if run_spec is None:
        if model_spec is None or output_spec is None or start_date is None or end_date is None:
            raise ValueError("Standard prepared execution requires model_spec, output_spec, start_date and end_date.")
        if len(model_spec.sectors) != 1:
            raise ValueError("Standard prepared execution requires exactly one model sector.")
        if model_builder is not None:
            raise ValueError("Standard prepared execution rejects model_builder; use likelihood_builder.")
        if likelihood_kwargs and likelihood_builder is None:
            raise ValueError("Non-empty `likelihood_kwargs` require an active `likelihood_builder`.")
        if likelihood_builder is not None and model_spec.likelihood is not None:
            raise ValueError("A custom likelihood cannot be combined with built-in likelihood settings.")
        aggregation_mode = select_aggregation_error_mode(prepared_inputs.inv_inputs, model_spec.aggregation_error_mode)
        if aggregation_mode != "none" and output_spec.output_format in {"basic", "paris", "legacy"}:
            raise ValueError(
                "RHIME aggregation-error covariance is not yet supported by derived "
                f"output_format={output_spec.output_format!r}; use 'inv_out' or 'none' until "
                "the postprocessing reconstruction follow-up lands."
            )
        validate_output_format(output_spec.output_format)
        validate_output_filename_convention(output_spec.output_filename_convention)
        validate_output_path_settings(
            output_format=output_spec.output_format, output_path=output_spec.output_path,
            save_trace=output_spec.save_trace, save_inversion_output=output_spec.save_inversion_output,
            multisector=False,
        )
        validate_output_resources(output_spec)
        prepared = prepared_inputs.validated()
        _require_prepared_layout(prepared, multisector=False)
        inputs, priors = materialize_standard_inputs(prepared, model_spec)
        built = build_standard_model(
            inputs, model_spec, priors=priors,
            likelihood_builder=likelihood_builder, likelihood_kwargs=likelihood_kwargs,
        )
        built.output_contract.validate_requested_output(output_spec.output_format)
        posterior = sample_rhime_model(built, active_sampler)
        annotate_sample_trace(
            built, posterior, likelihood_builder=likelihood_builder, likelihood_kwargs=likelihood_kwargs,
        )
        timing = (timed("rhime.output.inversion_output_create", output_format=output_spec.output_format)
                  if output_spec.output_format != "none" else nullcontext())
        with timing:
            inv_out = make_standard_output(
                prepared=prepared, model_spec=model_spec, posterior=posterior,
                output_contract=built.output_contract, start_date=start_date, end_date=end_date,
                actual_inference=active_sampler.describe(), provenance=installed_ogi_provenance(),
            )
        write_standard_outputs(inv_out, output_spec)
        output_path, trace_path = standard_artifact_paths(inv_out, output_spec)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with timed("rhime.output.inversion_output_save", path=output_path):
                inv_out.save(output_path)
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with timed("rhime.output.trace_save", path=trace_path):
                save_trace(posterior, trace_path)
        return inv_out
    if any(value is not None for value in (model_spec, output_spec, start_date, end_date)):
        raise ValueError("Pass standard model/output/date arguments or multisector run_spec, not both.")
    if not run_spec.split_by_sectors:
        raise ValueError("Standard prepared execution uses model_spec, output_spec, start_date and end_date instead of run_spec.")
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
    if model_builder is not None and not multisector:
        raise ValueError(
            "Standard prepared runs no longer accept `model_builder`; use `likelihood_builder` "
            "or build and sample a custom PyMC model directly."
        )

    _require_prepared_layout(prepared_inputs, multisector=run_spec.split_by_sectors)
    if run_spec.split_by_sectors is not multisector:
        raise ValueError(
            "`run_spec.split_by_sectors` must agree with the model sector count: "
            f"found {sector_count} sector(s) and split_by_sectors={run_spec.split_by_sectors}."
        )

    output_spec = run_spec.output
    validate_output_format(output_spec.output_format)
    if model_builder is None:
        aggregation_error_mode = select_aggregation_error_mode(
            prepared_inputs.inv_inputs,
            run_spec.model.aggregation_error_mode,
        )
        if aggregation_error_mode != "none" and output_spec.output_format in {
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
