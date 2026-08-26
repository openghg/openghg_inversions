"""Command line interface for OpenGHG inversions."""

from __future__ import annotations

import argparse
import json
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

    merge_paris_outputs(args.input_files, args.output, output_type=args.type)


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
        choices=("flux", "concentration"),
        help="Expected output type (auto-detected when omitted)",
    )
    merge_parser.set_defaults(func=_merge_paris_outputs_command)

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
