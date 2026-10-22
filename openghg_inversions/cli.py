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
        help="JSON configuration overrides; CO2 uses model/likelihood/sampling/outputs sections.",
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


def _run_command(args: argparse.Namespace) -> None:
    """Run the selected configuration-driven recipe."""
    if args.model == "co2":
        _run_co2_command(args)
        return
    from openghg_inversions.rhime import run

    run(model=args.model, config_file=args.config, **_command_kwargs(args))


def _run_rhime_command(args: argparse.Namespace) -> None:
    """Run the standard RHIME command with lazy imports for fast help output."""
    from openghg_inversions.rhime import run_standard

    run_standard(config_file=args.config, **_command_kwargs(args))


def _run_rhime_multisector_command(args: argparse.Namespace) -> None:
    """Run the multi-sector RHIME command with lazy imports for fast help output."""
    from openghg_inversions.rhime import run_rhime_multisector

    run_rhime_multisector(config_file=args.config, **_command_kwargs(args))


def _run_co2_command(args: argparse.Namespace) -> None:
    """Run configured CO2 inference and requested products."""
    from openghg_inversions.rhime import run

    run(
        model="co2",
        config_file=args.config, start_date=args.start, end_date=args.end,
        output_path=args.output_path, overrides=args.kwargs,
    )


def _merge_paris_outputs_command(args: argparse.Namespace) -> None:
    """Merge sequential PARIS output files with lazy imports for fast help output."""
    from openghg_inversions.postprocessing.merge_paris_outputs import merge_paris_outputs

    output_type = "concentration" if args.type == "conc" else args.type
    merge_paris_outputs(args.input_files, args.output, output_type=output_type)


def _add_stage_config_args(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add explicit scientific configuration arguments for staged commands."""
    source = parser.add_mutually_exclusive_group(required=required)
    source.add_argument("-c", "--config", help="Explicit RHIME INI or CO2 TOML configuration path")
    source.add_argument("--params-file", help="Explicit JSON file containing RHIME parameter names")
    parser.add_argument(
        "--model",
        choices=("standard", "multisector", "co2", "co2_o2"),
        default="standard",
        help="RHIME model recipe (default: standard); checked against saved handoffs",
    )
    parser.add_argument("--kwargs", type=json.loads, help="JSON object overriding RHIME parameters")


def _add_output_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR"),
        help="Explicit output directory (default: OGR OUTPUT_DIR when set)",
    )


def _stage_output_dir(args: argparse.Namespace) -> Path:
    """Require an output directory without falling back to the current directory."""
    if args.output_dir is None:
        raise ValueError("Pass `--output-dir` or set OGR's `OUTPUT_DIR`.")
    return Path(args.output_dir).resolve()


def _rhime_phase_options(args: argparse.Namespace, *, phase: str) -> dict[str, dict[str, Any]]:
    """Load only the owner groups consumed by a standard or multisector phase."""
    from openghg_inversions.rhime.requests import load_params

    return load_params(
        config_file=args.config, params_file=args.params_file, overrides=args.kwargs,
        phase_groups={"prepare": {"data"}, "sample": {"model", "sampler"},
                      "prior": {"model"}, "postprocess": {"output"}}[phase],
    )


def _co2_params(args: argparse.Namespace) -> dict[str, Any]:
    """Read a CO2 request; concrete owners resolve their own choices."""
    from openghg_inversions.rhime.co2.execution import load_co2_params

    params = load_co2_params(config_file=args.config, params_file=args.params_file,
                             params={} if args.config is None and args.params_file is None else None,
                             overrides=args.kwargs)
    return params


def _co2_o2_params(args: argparse.Namespace) -> dict[str, Any]:
    """Read linked choices without resolving unused phase sections."""
    from openghg_inversions.rhime.co2.co2_o2_configuration import load_co2_o2_params

    return load_co2_o2_params(
        config_file=args.config, params_file=args.params_file,
        params={} if args.config is None and args.params_file is None else None,
        overrides=args.kwargs,
    )


def _prepare_command(args: argparse.Namespace) -> None:
    """Run the prepare phase for the selected family."""
    from openghg_inversions.rhime.stages import prepare
    from openghg_inversions.rhime.params import DataConfig

    if args.model in {"standard", "multisector"}:
        if args.model == "multisector":
            from openghg_inversions.rhime.multisector_stages import prepare
        choices = _rhime_phase_options(args, phase="prepare")
        manifest = prepare(DataConfig.from_params(choices["data"], multisector=args.model == "multisector"),
                           output_dir=_stage_output_dir(args))
        print(manifest)
    elif args.model == "co2_o2":
        from openghg_inversions.rhime.co2.co2_o2_configuration import Co2O2PreparationConfig
        from openghg_inversions.rhime.co2.co2_o2_handoffs import prepare as prepare_linked

        print(prepare_linked(
            **Co2O2PreparationConfig.from_params(_co2_o2_params(args), require_path=True).preparation_arguments(),
            output_dir=_stage_output_dir(args),
        ))
    elif args.model == "co2":
        from openghg_inversions.rhime.co2.execution import load_co2_params
        from openghg_inversions.rhime.co2.configuration import Co2PreparationConfig
        from openghg_inversions.rhime.co2.handoffs import prepare as prepare_co2

        params = load_co2_params(config_file=args.config, params_file=args.params_file, overrides=args.kwargs)
        print(prepare_co2(**Co2PreparationConfig.from_params(params, require_path=True).preparation_arguments(), output_dir=_stage_output_dir(args)))


def _check_saved_family(args: argparse.Namespace) -> dict[str, Any] | None:
    """Reject an explicit selector that contradicts a current saved handoff."""
    manifest = args.sample_manifest if args.command == "postprocess" else args.preparation_manifest
    if not Path(manifest).exists():
        if args.model == "co2":
            raise FileNotFoundError(2, "Saved manifest does not exist", str(manifest))
        return None
    record = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if not isinstance(record, dict) and args.model != "co2":
        return None
    if not isinstance(record, dict):
        raise argparse.ArgumentError(None, "Saved manifest must contain one JSON object.")
    schema = record.get("schema", "")
    if not isinstance(schema, str) and args.model != "co2":
        return record
    if not isinstance(schema, str):
        raise argparse.ArgumentError(None, "Saved manifest schema must be a string.")
    for family in ("standard", "multisector", "co2_o2"):
        if schema.startswith(f"openghg_inversions.{family}.") and args.model != family:
            raise ValueError(f"Saved {family} checkpoint does not match --model {args.model!r}.")
    if schema.startswith("openghg_inversions.co2.") and args.model != "co2":
        raise ValueError(f"Saved CO2 checkpoint does not match --model {args.model!r}.")
    if args.model == "co2":
        stage = "sample" if args.command == "postprocess" else "prepare"
        if schema != f"openghg_inversions.co2.{stage}":
            raise argparse.ArgumentError(None, f"Unsupported CO2 manifest schema: {schema!r}.")
    return record


def _prior_predictive_command(args: argparse.Namespace) -> None:
    """Run the prior-predictive readiness check for the selected family."""
    from openghg_inversions.rhime.stages import prior_predictive

    common = dict(output_dir=_stage_output_dir(args), check_output=args.check_output,
                  draws=args.draws, stage=args.check_stage)
    _check_saved_family(args)
    if args.model == "co2_o2":
        from openghg_inversions.rhime.co2.co2_o2_handoffs import prior_predictive as prior_linked
        from openghg_inversions.rhime.co2.co2_o2_configuration import resolve_co2_o2_prior_seed

        params = _co2_o2_params(args)
        result = prior_linked(
            args.preparation_manifest, model=params,
            random_seed=resolve_co2_o2_prior_seed(params), **common,
        )
    elif args.model == "co2":
        from openghg_inversions.rhime.co2.handoffs import prior_predictive as prior_co2

        from openghg_inversions.rhime.co2.configuration import resolve_co2_prior_seed

        params = _co2_params(args)
        choices: dict[str, Any] = {name: params[name] for name in ("model", "likelihood") if name in params}
        result = prior_co2(args.preparation_manifest, model=choices,
                           random_seed=resolve_co2_prior_seed(params), **common)
    elif args.model in {"standard", "multisector"}:
        if args.model == "multisector":
            from openghg_inversions.rhime.multisector_stages import prior_predictive
        choices = _rhime_phase_options(args, phase="prior")
        result = prior_predictive(args.preparation_manifest, model=choices["model"], **common)
    print(json.dumps(result, sort_keys=True))
    if args.strict and result["status"] == "fail":
        raise SystemExit(1)


def _sample_command(args: argparse.Namespace) -> None:
    """Run the sample phase for the selected family."""
    from openghg_inversions.rhime.stages import sample
    from openghg_inversions.rhime.params import resolve_sampler_options

    _check_saved_family(args)
    if args.model == "co2_o2":
        from openghg_inversions.rhime.co2.co2_o2_handoffs import sample as sample_linked
        from openghg_inversions.rhime.co2.co2_o2_configuration import Co2O2ModelConfig, resolve_co2_o2_sampler

        params = _co2_o2_params(args)
        model = Co2O2ModelConfig.from_params(params)
        result = sample_linked(
            args.preparation_manifest, model=model, sampler=resolve_co2_o2_sampler(params, model=model),
            output_dir=_stage_output_dir(args),
        )
        print(result)
    elif args.model == "co2":
        from openghg_inversions.rhime.co2.handoffs import sample as sample_co2


        params = _co2_params(args)
        choices: dict[str, Any] = {name: params[name] for name in ("model", "likelihood") if name in params}
        result = sample_co2(args.preparation_manifest, model=choices,
                            sampling=params.get("sampling", {}), output_dir=_stage_output_dir(args))
        print(result)
    elif args.model in {"standard", "multisector"}:
        if args.model == "multisector":
            from openghg_inversions.rhime.multisector_stages import sample
        choices = _rhime_phase_options(args, phase="sample")
        result = sample(args.preparation_manifest, model=choices["model"],
                        sampler=resolve_sampler_options(choices["sampler"]),
                        output_dir=_stage_output_dir(args))
        print(result)


def _diagnose_command(args: argparse.Namespace) -> None:
    from openghg_inversions.rhime.stages import diagnose_rhime_stage

    result = diagnose_rhime_stage(
        posterior=args.posterior,
        sample_manifest=args.sample_manifest,
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
    """Run the postprocess phase for the selected family."""
    from openghg_inversions.rhime.stages import postprocess
    from openghg_inversions.rhime.specs import RhimeOutputSpec

    _check_saved_family(args)
    if args.model == "co2_o2":
        from openghg_inversions.rhime.co2.co2_o2_handoffs import postprocess as postprocess_linked
        from openghg_inversions.rhime.co2.co2_o2_outputs import resolve_co2_o2_output

        params = _co2_o2_params(args)
        params["outputs"] = {**params.get("outputs", {}), "output_path": _stage_output_dir(args)}
        postprocess_linked(args.sample_manifest, **resolve_co2_o2_output(params))
    elif args.model == "co2":
        from openghg_inversions.rhime.co2.handoffs import postprocess as postprocess_co2

        from openghg_inversions.rhime.co2.execution import resolve_co2_output

        params = _co2_params(args)
        params["outputs"] = {**params.get("outputs", {}), "output_path": _stage_output_dir(args)}
        postprocess_co2(args.sample_manifest, **resolve_co2_output(params))
    elif args.model in {"standard", "multisector"}:
        if args.model == "multisector":
            from openghg_inversions.rhime.multisector_stages import postprocess
        choices = _rhime_phase_options(args, phase="postprocess")["output"]
        choices["output_path"] = str(_stage_output_dir(args))
        postprocess(args.sample_manifest, output=RhimeOutputSpec.from_params(choices, multisector=args.model == "multisector"))


def _run_rhime_nested_command(args: argparse.Namespace) -> None:
    """Run the nested-domain RHIME command with lazy imports for fast help output."""
    from openghg_inversions.rhime import run_rhime_nested

    run_rhime_nested(config_file=args.config, **_command_kwargs(args))


def build_parser(*, prog: str = "openghg-inversions") -> argparse.ArgumentParser:
    """Build the OpenGHG inversions CLI argument parser.

    Args:
        prog: Executable name displayed in help.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(prog=prog, description="RHIME atmospheric inversions")
    subparsers = parser.add_subparsers(dest="command", required=True)

    selected_run = subparsers.add_parser("run", help="Run a selected RHIME model")
    _add_run_args(selected_run)
    selected_run.add_argument(
        "--model", choices=("standard", "multisector", "nested", "co2"), default="standard",
        help="Configuration-driven model recipe (default: standard)",
    )
    selected_run.set_defaults(func=_run_command)

    run_parser = subparsers.add_parser("run-rhime", help="Run a standard RHIME inversion")
    _add_run_args(run_parser)
    run_parser.set_defaults(func=_run_rhime_command)

    run_multi_parser = subparsers.add_parser(
        "run-rhime-multisector", help="Run a shared-basis multi-sector RHIME inversion"
    )
    _add_run_args(run_multi_parser)
    run_multi_parser.set_defaults(func=_run_rhime_multisector_command)

    co2_parser = subparsers.add_parser("run-co2", help="Alias for run --model co2")
    _add_run_args(co2_parser)
    co2_parser.set_defaults(func=_run_co2_command)

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
    _add_stage_config_args(prior_parser, required=False)
    _add_output_dir(prior_parser)
    prior_parser.add_argument("--preparation-manifest", required=True)
    prior_parser.add_argument("--check-output")
    prior_parser.add_argument("--draws", type=int, default=100)
    prior_parser.add_argument("--check-stage", default=os.environ.get("STAGE", "prior-predictive"))
    prior_parser.add_argument("--strict", action="store_true", help="Exit nonzero when the check fails")
    prior_parser.set_defaults(func=_prior_predictive_command)

    sample_parser = subparsers.add_parser("sample", help="Sample explicitly supplied RHIME inputs")
    _add_stage_config_args(sample_parser, required=False)
    _add_output_dir(sample_parser)
    sample_parser.add_argument("--preparation-manifest", required=True)
    sample_parser.set_defaults(func=_sample_command)

    diagnose_parser = subparsers.add_parser("diagnose", help="Calculate posterior convergence checks")
    _add_output_dir(diagnose_parser)
    diagnose_parser.add_argument("--posterior", help="Posterior path; omit to use a current sample manifest")
    diagnose_parser.add_argument(
        "--sample-manifest",
        help="Optional saved sampling context",
    )
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
    _add_stage_config_args(postprocess_parser, required=False)
    _add_output_dir(postprocess_parser)
    postprocess_parser.add_argument("--sample-manifest", required=True)
    postprocess_parser.set_defaults(func=_postprocess_command)

    run_nested_parser = subparsers.add_parser(
        "run-rhime-nested", help="Run a two-grid nested-domain RHIME inversion"
    )
    _add_run_args(run_nested_parser)
    run_nested_parser.set_defaults(func=_run_rhime_nested_command)

    return parser


def main(argv: list[str] | None = None, *, prog: str = "openghg-inversions") -> None:
    """Run the OpenGHG inversions CLI.

    Args:
        argv: Optional argument vector. Defaults to ``sys.argv`` when omitted.
        prog: Executable name displayed in help.
    """
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except argparse.ArgumentError as exc:
        parser.error(str(exc))
    except FileNotFoundError as exc:
        manifests = [getattr(args, name, None) for name in ("preparation_manifest", "sample_manifest")]
        if exc.filename is not None and any(
            name is not None and Path(name).resolve() == Path(exc.filename).resolve() for name in manifests
        ):
            parser.error(f"Saved manifest does not exist: {exc.filename}")
        raise


def rhime_main(argv: list[str] | None = None) -> None:
    """Run the ``rhime`` CLI with the same commands as ``openghg-inversions``.

    Args:
        argv: Optional argument vector; reads ``sys.argv`` when omitted.
    """
    main(argv, prog="rhime")


if __name__ == "__main__":
    main()
