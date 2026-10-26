"""Advanced copy-and-modify runner for a standard RHIME inversion.

This example deliberately replaces RHIME's default likelihood with a
project-owned Student-t builder. The orchestration is intentionally copied so
that a project can make deeper scientific changes while continuing to reuse
the supported acquisition, preparation, model, sampling, and output stages.
Use :func:`run_custom_rhime` from Python or :func:`main` from the command line.
The standard single-sector stages are preserved, model inputs materialize only
at the explicit PyMC boundary. This copied runner always acquires fresh data,
samples a model, and writes its configured outputs. To reuse persisted prepared
inputs, use :func:`openghg_inversions.rhime.run_standard` with ``prepared=...``.
"""

from __future__ import annotations

from openghg_inversions.basis import prepare_basis
from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.rhime.preparation import resolve_bc_basis, acquire_merged_data, resolve_basis_inputs

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from openghg_inversions.rhime import (
    RhimeConfig,
    make_model_output,
    postprocess_standard_outputs,
    save_inversion_output,
    annotate_sample_trace,
    assemble_inputs,
    build_sensitivities,
    filter_observations,
    prepare_observation_errors,
    validate_standard_output_resources,
    read_rhime_ini,
    sample_rhime_model,
)
from openghg_inversions.rhime.standard import build_standard_model, materialize_standard_inputs
from openghg_inversions.rhime.requests import load_params
from openghg_inversions.postprocessing.inversion_output import InversionOutput

from .likelihoods import likelihood_builder


def run_custom_rhime(
    *,
    config_file: str | Path | None = None,
    **kwargs: Any,
) -> InversionOutput:
    """Run standard RHIME with a project-owned Student-t likelihood.

    Args:
        config_file: Optional RHIME INI configuration file.
        **kwargs: Standard RHIME option names that override values from
            ``config_file``.

    Returns:
        The sampled scientific view; requested files are written before return.

    Raises:
        TypeError: If the likelihood builder returns an invalid result type.
        ValueError: If required options are missing, aggregation or output is
            unsupported, or likelihood roles or metadata are invalid.

    Notes:
        This workflow always acquires fresh data, materializes related model
        arrays together at the named PyMC boundary without mutating canonical
        prepared inputs, runs sampling, and writes outputs requested by the
        resolved RHIME options. Supported outputs are checked against the built
        model contract before posterior sampling.
    """
    params = read_rhime_ini(config_file) if config_file is not None else {}
    params.update(kwargs)
    params["mismatch_model"] = None
    config = RhimeConfig.from_params(load_params(overrides=params), multisector=False)
    validate_standard_output_resources(config.output)

    # Resolve selected support files before acquiring observations or flux.
    basis_inputs = resolve_basis_inputs(config.data)
    bc_basis = resolve_bc_basis(config.data)
    merged = acquire_merged_data(config.data)

    # Keep the scientific preparation order visible; ``merged`` stays acquired data.
    filtered, site_options = filter_observations(merged.site_data, merged.site_options, filters=config.data.filters)
    filtered = prepare_observation_errors(filtered, averaging_error=config.data.averaging_error)

    runtime_flux = resolve_runtime_flux(
        merged.flux_data, split_by_sectors=merged.split_by_sectors,
    )
    weighting_source = config.data.flux_sources[0] if config.data.flux_sources else next(iter(merged.flux_data))
    weighting_flux = merged.flux_data[weighting_source]["flux"]
    basis_functions = prepare_basis(
        site_data=filtered,
        runtime_flux=runtime_flux,
        weighting_flux=weighting_flux,
        **config.data.select(
            "domain", "start_date", "basis_algorithm", "nbasis",
        ),
        fix_outer_regions=config.data.fix_basis_outer_regions,
        **basis_inputs,
    )
    if config.data.basis_output_path is not None:
        Path(config.data.basis_output_path).parent.mkdir(parents=True, exist_ok=True)
        basis_functions.save(config.data.basis_output_path)

    sensitivities = build_sensitivities(
        filtered,
        basis_functions,
        **config.data.select(
            "flux_sources", "use_bc",
        ),
        bc_basis=bc_basis,
        multisector=False,
    )

    prepared = assemble_inputs(
        sensitivities,
        basis_functions,
        site_options=site_options,
        **config.data.select(
            "domain", "start_date", "use_bc",
        ),
    )

    model_inputs, priors = materialize_standard_inputs(prepared, config.model)
    model_build_result = build_standard_model(
        model_inputs, config.model, priors=priors,
        # This is the deliberate scientific replacement in the copied runner.
        likelihood_builder=likelihood_builder,
    )
    model_build_result.output_contract.validate_requested_output(config.output.output_format)

    idata = sample_rhime_model(model_build_result, config.sampler)

    annotate_sample_trace(model_build_result, idata, likelihood_builder=likelihood_builder)
    inv_out = make_model_output(
        prepared=prepared,
        model_spec=config.model,
        posterior=idata,
        output_contract=model_build_result.output_contract,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        actual_inference=config.sampler.describe(),
        split_by_sectors=config.data.split_by_sectors,
    )
    postprocess_standard_outputs(inv_out, config.output)
    save_inversion_output(inv_out, config.output)
    return inv_out


def _json_object(value: str) -> dict[str, Any]:
    """Parse one command-line JSON object containing additional RHIME options.

    Args:
        value: JSON text to parse.

    Returns:
        The decoded JSON object.

    Raises:
        argparse.ArgumentTypeError: If the text is invalid JSON or decodes to
            a non-object value.
    """
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--kwargs must decode to a JSON object")
    return parsed


def main(argv: Sequence[str] | None = None) -> InversionOutput:
    """Parse command-line options and run the customized RHIME workflow.

    This performs the same acquisition, materialization, sampling, and output
    side effects as :func:`run_custom_rhime`.

    Args:
        argv: Command-line tokens excluding the program name. ``None`` reads
            tokens from :data:`sys.argv`.

    Returns:
        The RHIME result returned by :func:`run_custom_rhime`.

    Raises:
        SystemExit: If argument parsing fails or help is requested.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", nargs="?", type=Path, help="RHIME INI configuration file")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--output-name")
    parser.add_argument("--draws", type=int)
    parser.add_argument("--tune", type=int)
    parser.add_argument("--chains", type=int)
    parser.add_argument(
        "--kwargs",
        type=_json_object,
        default={},
        metavar="JSON",
        help="additional RHIME options as a JSON object",
    )
    args = parser.parse_args(argv)

    overrides = dict(args.kwargs)
    for name in ("start_date", "end_date", "output_path", "output_name", "draws", "tune", "chains"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    return run_custom_rhime(config_file=args.config_file, **overrides)


if __name__ == "__main__":
    main()
