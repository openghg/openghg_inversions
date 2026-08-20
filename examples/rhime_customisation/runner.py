"""Advanced copy-and-modify runner for a standard RHIME inversion.

This example deliberately replaces RHIME's default likelihood with a
project-owned Student-t builder. The orchestration is intentionally copied so
that a project can make deeper scientific changes while continuing to reuse
the supported acquisition, preparation, model, sampling, and output stages.
Use :func:`run_custom_rhime` from Python or :func:`main` from the command line.
The standard single-sector stages are preserved, model inputs materialize only
at the explicit PyMC boundary, and a run may acquire or reload data, sample a
model, and write its configured outputs.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from openghg_inversions.rhime import (
    RhimeResult,
    assemble_rhime_inputs,
    build_rhime_basis,
    build_rhime_sensitivities,
    build_standard_rhime_model_result,
    filter_rhime_observations,
    make_standard_rhime_result,
    make_standard_rhime_outputs,
    materialize_pymc_inputs,
    params_from_config,
    resolve_rhime_options,
    retrieve_or_reload_rhime_data,
    sample_rhime_model,
    standard_model_input_names,
    with_prepared_rhime_sites,
)

from .likelihoods import likelihood_builder


def run_custom_rhime(
    *,
    config_file: str | Path | None = None,
    **kwargs: Any,
) -> RhimeResult:
    """Run standard RHIME with a project-owned Student-t likelihood.

    Args:
        config_file: Optional RHIME INI configuration file.
        **kwargs: Standard RHIME option names that override values from
            ``config_file``.

    Returns:
        The sampled result and any outputs requested by the RHIME options.

    Raises:
        TypeError: If the likelihood builder returns an invalid result type.
        ValueError: If required options are missing, aggregation or output is
            unsupported, or likelihood roles or metadata are invalid.

    Notes:
        This workflow may retrieve or reload data, materializes related model
        arrays together at the named PyMC boundary without mutating canonical
        prepared inputs, runs sampling, and writes outputs requested by the
        resolved RHIME options.
    """
    params = (
        params_from_config(config_file, extra_kwargs=kwargs, normalise=False)
        if config_file is not None
        else dict(kwargs)
    )
    setup = resolve_rhime_options(params=params, multisector=False)

    merged = retrieve_or_reload_rhime_data(setup.data_args, multisector=False)
    filtered = filter_rhime_observations(merged, setup.data_args)
    basis_functions = build_rhime_basis(filtered, setup.data_args)
    site_data = build_rhime_sensitivities(
        filtered,
        basis_functions,
        setup.data_args,
        multisector=False,
    )
    prepared = assemble_rhime_inputs(filtered, basis_functions, site_data, setup.data_args)
    run_spec = with_prepared_rhime_sites(setup.run_spec, prepared)

    model_inputs = materialize_pymc_inputs(
        prepared,
        variable_names=standard_model_input_names(prepared, run_spec.model),
    )
    build_and_sample_start = perf_counter()
    model_build_result = build_standard_rhime_model_result(
        prepared=prepared,
        model_inputs=model_inputs,
        run_spec=run_spec,
        # This is the deliberate scientific replacement in the copied runner.
        likelihood_builder=likelihood_builder,
    )
    idata = sample_rhime_model(model_build_result, setup.sampler)

    result = make_standard_rhime_result(
        prepared=prepared,
        run_spec=run_spec,
        sampler=setup.sampler,
        model_build_result=model_build_result,
        idata=idata,
        build_and_sample_seconds=perf_counter() - build_and_sample_start,
        likelihood_builder=likelihood_builder,
    )
    make_standard_rhime_outputs(result=result, prepared=prepared)
    return result


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


def main(argv: Sequence[str] | None = None) -> RhimeResult:
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
