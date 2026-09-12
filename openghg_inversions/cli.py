"""Command line interface for OpenGHG inversions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("start", help="Start date string of the format YYYY-MM-DD", nargs="?")
    parser.add_argument("end", help="End date string of the format YYYY-MM-DD", nargs="?")
    parser.add_argument("-c", "--config", help="Name including path of configuration file", required=True)
    parser.add_argument(
        "--kwargs",
        type=json.loads,
        help="Pass keyword arguments to the RHIME function, e.g. '{\"draws\": 10}'.",
    )
    parser.add_argument("--output-path", help="Path to write results to.")


def _command_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Create keyword overrides from parsed CLI arguments."""
    kwargs: dict[str, Any] = {}
    if args.start:
        kwargs["start_date"] = args.start
    if args.end:
        kwargs["end_date"] = args.end
    if args.output_path:
        kwargs["output_path"] = args.output_path
    if args.kwargs:
        kwargs.update(args.kwargs)
    return kwargs


def _run_rhime_command(args: argparse.Namespace) -> None:
    """Run the standard RHIME command with lazy imports for fast help output."""
    from openghg_inversions.rhime import run_rhime

    run_rhime(config_file=args.config, **_command_kwargs(args))


def _run_rhime_multisector_command(args: argparse.Namespace) -> None:
    """Run the multi-sector RHIME command with lazy imports for fast help output."""
    from openghg_inversions.rhime import run_rhime_multisector

    run_rhime_multisector(config_file=args.config, **_command_kwargs(args))


def _merge_paris_outputs_command(args: argparse.Namespace) -> None:
    """Merge sequential PARIS output files with lazy imports for fast help output."""
    from openghg_inversions.postprocessing.merge_paris_outputs import merge_paris_outputs

    output_type = "concentration" if args.type == "conc" else args.type
    merge_paris_outputs(args.input_files, args.output, output_type=output_type)


def _add_stage_config_args(parser: argparse.ArgumentParser) -> None:
    """Add explicit scientific configuration arguments for staged commands."""
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-c", "--config", help="Explicit RHIME INI configuration path")
    source.add_argument("--params-file", help="Explicit JSON file containing RHIME parameter names")
    parser.add_argument(
        "--model",
        choices=("standard", "multisector"),
        required=True,
        help="RHIME model recipe; never inferred from the gas",
    )
    parser.add_argument("--kwargs", type=json.loads, help="JSON object overriding RHIME parameters")


def _add_output_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR"),
        help="Explicit output directory (default: OGR OUTPUT_DIR when set)",
    )


def _stage_setup(args: argparse.Namespace):
    """Load and resolve one staged command's existing RHIME configuration."""
    from openghg_inversions.rhime.stages import load_stage_params, resolve_stage_setup

    params = load_stage_params(
        config_file=args.config,
        params_file=args.params_file,
        overrides=args.kwargs,
    )
    return resolve_stage_setup(params, model=args.model)


def _stage_output_dir(args: argparse.Namespace) -> Path:
    """Require an output directory without falling back to the current directory."""
    if args.output_dir is None:
        raise ValueError("Pass `--output-dir` or set OGR's `OUTPUT_DIR`.")
    return Path(args.output_dir).resolve()


def _prepare_command(args: argparse.Namespace) -> None:
    from openghg_inversions.rhime.stages import prepare_rhime_stage

    manifest = prepare_rhime_stage(
        setup=_stage_setup(args),
        model=args.model,
        output_dir=_stage_output_dir(args),
    )
    print(manifest["manifest_path"])


def _prior_predictive_command(args: argparse.Namespace) -> None:
    from openghg_inversions.rhime.stages import prior_predictive_stage

    result = prior_predictive_stage(
        setup=_stage_setup(args),
        model=args.model,
        prepared_inputs=args.prepared_inputs,
        preparation_manifest=args.preparation_manifest,
        output_dir=_stage_output_dir(args),
        check_output=args.check_output,
        draws=args.draws,
        stage=args.check_stage,
    )
    print(json.dumps(result, sort_keys=True))
    if args.strict and result["status"] == "fail":
        raise SystemExit(1)


def _sample_command(args: argparse.Namespace) -> None:
    from openghg_inversions.rhime.stages import sample_rhime_stage

    result = sample_rhime_stage(
        setup=_stage_setup(args),
        model=args.model,
        prepared_inputs=args.prepared_inputs,
        preparation_manifest=args.preparation_manifest,
        output_dir=_stage_output_dir(args),
    )
    print(result["artifacts"]["posterior"])


def _diagnose_command(args: argparse.Namespace) -> None:
    from openghg_inversions.rhime.stages import diagnose_rhime_stage

    result = diagnose_rhime_stage(
        posterior=args.posterior,
        output_dir=_stage_output_dir(args),
        check_output=args.check_output,
        max_rhat=args.max_rhat,
        min_bulk_ess=args.min_bulk_ess,
        min_tail_ess=args.min_tail_ess,
        max_divergences=args.max_divergences,
        stage=args.check_stage,
    )
    print(json.dumps(result, sort_keys=True))
    if args.strict and result["status"] == "fail":
        raise SystemExit(1)


def _postprocess_command(args: argparse.Namespace) -> None:
    from openghg_inversions.rhime.stages import postprocess_rhime_stage

    postprocess_rhime_stage(
        setup=_stage_setup(args),
        model=args.model,
        prepared_inputs=args.prepared_inputs,
        preparation_manifest=args.preparation_manifest,
        posterior=args.posterior,
        output_dir=_stage_output_dir(args),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the OpenGHG inversions CLI argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(prog="openghg-inversions", description="OpenGHG inversions CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-rhime", help="Run a standard RHIME inversion")
    _add_run_args(run_parser)
    run_parser.set_defaults(func=_run_rhime_command)

    run_multi_parser = subparsers.add_parser(
        "run-rhime-multisector", help="Run a shared-basis multi-sector RHIME inversion"
    )
    _add_run_args(run_multi_parser)
    run_multi_parser.set_defaults(func=_run_rhime_multisector_command)

    merge_parser = subparsers.add_parser(
        "merge-paris-outputs",
        help="Merge sequential PARIS flux or concentration NetCDF outputs",
    )
    merge_parser.add_argument("input_files", nargs="+", help="PARIS NetCDF files to merge")
    merge_parser.add_argument("-o", "--output", required=True, help="Merged NetCDF output path")
    merge_parser.add_argument(
        "--type",
        choices=("flux", "concentration", "conc"),
        help="Output type to select (auto-detected when omitted; 'conc' is an alias)",
    )
    merge_parser.set_defaults(func=_merge_paris_outputs_command)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare and persist reusable RHIME inputs")
    _add_stage_config_args(prepare_parser)
    _add_output_dir(prepare_parser)
    prepare_parser.set_defaults(func=_prepare_command)

    prior_parser = subparsers.add_parser(
        "prior-predictive", help="Validate a configured model with prepared inputs"
    )
    _add_stage_config_args(prior_parser)
    _add_output_dir(prior_parser)
    prior_parser.add_argument("--prepared-inputs", required=True)
    prior_parser.add_argument("--preparation-manifest")
    prior_parser.add_argument("--check-output")
    prior_parser.add_argument("--draws", type=int, default=100)
    prior_parser.add_argument("--check-stage", default=os.environ.get("STAGE", "prior-predictive"))
    prior_parser.add_argument("--strict", action="store_true", help="Exit nonzero when the check fails")
    prior_parser.set_defaults(func=_prior_predictive_command)

    sample_parser = subparsers.add_parser("sample", help="Sample explicitly supplied RHIME inputs")
    _add_stage_config_args(sample_parser)
    _add_output_dir(sample_parser)
    sample_parser.add_argument("--prepared-inputs", required=True)
    sample_parser.add_argument("--preparation-manifest")
    sample_parser.set_defaults(func=_sample_command)

    diagnose_parser = subparsers.add_parser("diagnose", help="Calculate posterior convergence checks")
    _add_output_dir(diagnose_parser)
    diagnose_parser.add_argument("--posterior", required=True)
    diagnose_parser.add_argument("--check-output")
    diagnose_parser.add_argument("--max-rhat", type=float, default=1.01)
    diagnose_parser.add_argument("--min-bulk-ess", type=float, default=400)
    diagnose_parser.add_argument("--min-tail-ess", type=float, default=400)
    diagnose_parser.add_argument("--max-divergences", type=int, default=0)
    diagnose_parser.add_argument("--check-stage", default=os.environ.get("STAGE", "posterior"))
    diagnose_parser.add_argument("--strict", action="store_true", help="Exit nonzero when the check fails")
    diagnose_parser.set_defaults(func=_diagnose_command)

    postprocess_parser = subparsers.add_parser(
        "postprocess", help="Create configured outputs from a saved posterior"
    )
    _add_stage_config_args(postprocess_parser)
    _add_output_dir(postprocess_parser)
    postprocess_parser.add_argument("--prepared-inputs", required=True)
    postprocess_parser.add_argument("--preparation-manifest")
    postprocess_parser.add_argument("--posterior", required=True)
    postprocess_parser.set_defaults(func=_postprocess_command)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the OpenGHG inversions CLI.

    Args:
        argv: Optional argument vector. Defaults to ``sys.argv`` when omitted.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
