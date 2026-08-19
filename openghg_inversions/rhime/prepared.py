"""Advanced RHIME entry point for already prepared scientific inputs."""

from __future__ import annotations

import pandas as pd

from openghg_inversions._timing import timer_seconds, timer_start
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.observation_error import (
    resolve_aggregation_error,
    select_aggregation_error_mode,
)

from ._model_building import validate_likelihood_builder_argument
from .builders import RhimeLikelihoodBuilder, RhimeModelBuilder
from .materialization import materialize_pymc_inputs
from .multisector import build_multisector_rhime_model_result, make_multisector_rhime_result
from .outputs import RhimeResult
from .preparation import with_prepared_rhime_sites
from .sampling import RhimeSampler, sample_rhime_model
from .specs import (
    RhimeRunSpec,
    validate_output_filename_convention,
    validate_output_format,
    validate_output_path_settings,
)
from .standard import build_standard_rhime_model_result, make_standard_rhime_result


def run_rhime_from_prepared_inputs(
    *,
    prepared_inputs: RhimePreparedInputs,
    run_spec: RhimeRunSpec,
    sampler: RhimeSampler | None = None,
    model_builder: RhimeModelBuilder | None = None,
    likelihood_builder: RhimeLikelihoodBuilder | None = None,
) -> RhimeResult:
    """Build, sample, and output RHIME from a canonical prepared handoff.

    This advanced route intentionally starts after retrieval, filtering, basis
    construction, sensitivity construction, and labelled-input assembly. It
    validates the retained scientific layout before crossing the PyMC
    materialization boundary.

    Args:
        prepared_inputs: Validated canonical inversion inputs and retained
            basis functions.
        run_spec: Resolved model, output, and run settings.
        sampler: Optional sampler configuration; defaults to ``RhimeSampler``.
        model_builder: Optional complete-model callable for advanced graphs.
        likelihood_builder: Optional ordinary likelihood callable used with
            the built-in graph.

    Returns:
        Sampled result with any requested output products attached.

    Raises:
        ValueError: If layouts, extension points, aggregation error, or output
            settings are inconsistent.
        TypeError: If an extension point has an invalid callable contract.
    """
    validate_likelihood_builder_argument(likelihood_builder)
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
    if model_builder is not None:
        aggregation_error_mode = resolve_aggregation_error(
            prepared_inputs.inv_inputs,
            run_spec.model.aggregation_error_mode,
        ).mode
    else:
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

    run_spec = with_prepared_rhime_sites(run_spec, prepared_inputs)
    # Complete-model builders historically own canonical, potentially lazy
    # inputs. Only built-in graphs cross the named eager PyMC boundary here.
    model_inputs = (
        prepared_inputs.inv_inputs
        if model_builder is not None
        else materialize_pymc_inputs(
            prepared_inputs,
            aggregation_error_mode=run_spec.model.aggregation_error_mode,
        )
    )
    active_sampler = RhimeSampler() if sampler is None else sampler
    build_and_sample_start = timer_start()

    if multisector:
        model_build_result = build_multisector_rhime_model_result(
            prepared=prepared_inputs,
            model_inputs=model_inputs,
            run_spec=run_spec,
            model_builder=model_builder,
            likelihood_builder=likelihood_builder,
        )
    else:
        model_build_result = build_standard_rhime_model_result(
            prepared=prepared_inputs,
            model_inputs=model_inputs,
            run_spec=run_spec,
            model_builder=model_builder,
            likelihood_builder=likelihood_builder,
        )
    idata = sample_rhime_model(
        model_build_result,
        active_sampler,
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
