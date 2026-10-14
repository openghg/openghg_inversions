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


def _run_command(args: argparse.Namespace) -> None:
    """Run the selected configuration-driven recipe."""
    from openghg_inversions.rhime import run

    run(model=args.model, config_file=args.config, **_command_kwargs(args))


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


def _add_stage_config_args(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add explicit scientific configuration arguments for staged commands."""
    source = parser.add_mutually_exclusive_group(required=required)
    source.add_argument("-c", "--config", help="Explicit RHIME INI or CO2 TOML configuration path")
    source.add_argument("--params-file", help="Explicit JSON file containing RHIME parameter names")
    parser.add_argument(
        "--model",
        choices=("standard", "multisector", "co2"),
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


def _stage_setup(args: argparse.Namespace):
    """Load and resolve one staged command's existing RHIME configuration."""
    from openghg_inversions.rhime.stages import load_stage_params, resolve_stage_setup

    params = load_stage_params(
        config_file=args.config,
        params_file=args.params_file,
        overrides=args.kwargs,
        model=args.model,
    )
    return resolve_stage_setup(params, model=args.model)


def _stage_output_dir(args: argparse.Namespace) -> Path:
    """Require an output directory without falling back to the current directory."""
    if args.output_dir is None:
        raise ValueError("Pass `--output-dir` or set OGR's `OUTPUT_DIR`.")
    return Path(args.output_dir).resolve()


def _standard_phase_options(args: argparse.Namespace, *, phase: str) -> dict[str, dict[str, Any]]:
    """Select active owner mappings before resolving their values."""
    from collections.abc import Mapping
    from typing import get_type_hints

    from openghg_inversions.rhime._resources import resolve_resource_paths
    from openghg_inversions.rhime._runner_compat import _ALIASES, translate_rhime_aliases
    from openghg_inversions.rhime.ini import read_rhime_ini
    from openghg_inversions.rhime.params import DataConfig, rhime_option_groups
    from openghg_inversions.rhime.specs import RhimeOutputSpec

    owners = rhime_option_groups()
    active = {"prepare": {"data"}, "sample": {"model", "sampler"},
              "prior": {"model"}, "postprocess": {"output"}}[phase]
    claims = {"use_bc", "bc_freq", "species", "domain", "flux_sources", "start_date", "end_date", "split_by_sectors"}
    aliases = _ALIASES | {"emissions_store": "flux_store", "emissions_domain": "flux_domain"}
    source = Path(args.config or args.params_file).resolve() if args.config or args.params_file else None
    params = (read_rhime_ini(source) if args.config else json.loads(source.read_text())) if source else {}
    groups: dict[str, dict[str, Any]] = {group: {} for group in owners}
    override_names: dict[str, set[str]] = {group: set() for group in owners}
    unsuitable = []
    # Classify names only: inactive shared-file values must not be resolved.
    for values, is_override in ((params, False), (args.kwargs or {}, True)):
        if not isinstance(values, Mapping):
            raise ValueError("RHIME parameters must be a mapping.")
        for key, value in values.items():
            entries = value.items() if key in owners and isinstance(value, Mapping) else [(key, value)]
            if key in owners and not isinstance(value, Mapping):
                raise ValueError(f"`{key}` must be a mapping.")
            for name, item in entries:
                canonical = aliases.get(name, name)
                owner = key if key in owners else next(
                    (group for group, names in owners.items() if canonical in names), None
                )
                if name == "use_tracer" and key not in owners:
                    owner = "data"
                if phase in {"sample", "prior"} and key not in owners and canonical in claims:
                    owner = "model"
                known = owner is not None and (canonical in owners[owner] or name == "use_tracer")
                if not known or (is_override and owner not in active):
                    unsuitable.append(f"{key}.{name}" if key in owners else name)
                    continue
                groups[owner][name] = item
                if is_override:
                    override_names[owner].add(name)
    if unsuitable:
        raise ValueError(f"Unsupported options or overrides that do not belong to this phase: {sorted(unsuitable)!r}")
    annotations = get_type_hints(DataConfig) | get_type_hints(RhimeOutputSpec)
    result = {group: {} for group in owners}
    for group in active:
        winning = groups[group]
        translated = translate_rhime_aliases(winning)
        invocation_names = {
            aliases.get(name, name) for name in override_names[group]
            if name not in aliases or aliases[name] not in winning
        }
        result[group] = resolve_resource_paths(
            {name: value for name, value in translated.items() if name not in invocation_names},
            annotations=annotations, base_dir=source.parent if source else Path.cwd(),
        ) | resolve_resource_paths(
            {name: value for name, value in translated.items() if name in invocation_names},
            annotations=annotations, base_dir=Path.cwd(),
        )
    return result


def _prepare_command(args: argparse.Namespace) -> None:
    from openghg_inversions.rhime.stages import prepare, prepare_rhime_stage
    from openghg_inversions.rhime.params import DataConfig

    if args.model == "standard":
        choices = _standard_phase_options(args, phase="prepare")
        manifest = prepare(DataConfig.from_params(choices["data"]),
                           output_dir=_stage_output_dir(args))
        print(manifest)
    else:
        manifest = prepare_rhime_stage(setup=_stage_setup(args), model=args.model,
                                       output_dir=_stage_output_dir(args))
        print(manifest["manifest_path"])



def _check_saved_family(args: argparse.Namespace) -> None:
    """Reject an explicit selector that contradicts a current saved handoff."""
    manifest = args.sample_manifest if args.command == "postprocess" else args.preparation_manifest
    record = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if isinstance(record, dict) and record.get("family") not in (None, args.model):
        raise ValueError(f"Saved family {record['family']!r} does not match --model {args.model!r}.")

def _prior_predictive_command(args: argparse.Namespace) -> None:
    from openghg_inversions.rhime.stages import prior_predictive, prior_predictive_stage

    common = dict(output_dir=_stage_output_dir(args), check_output=args.check_output,
                  draws=args.draws, stage=args.check_stage)
    if args.model == "standard":
        _reject_standard_artifact_args(args)
        choices = _standard_phase_options(args, phase="prior")
        result = prior_predictive(args.preparation_manifest, model=choices["model"], **common)
    else:
        _check_saved_family(args)
        result = prior_predictive_stage(setup=_stage_setup(args), model=args.model,
                                        prepared_inputs=args.prepared_inputs,
                                        preparation_manifest=args.preparation_manifest, **common)
    print(json.dumps(result, sort_keys=True))
    if args.strict and result["status"] == "fail":
        raise SystemExit(1)


def _reject_standard_artifact_args(args: argparse.Namespace) -> None:
    if getattr(args, "prepared_inputs", None) or getattr(args, "posterior", None):
        raise ValueError("Standard saved phases take one manifest; remove --prepared-inputs and --posterior.")
    if args.command == "postprocess" and args.preparation_manifest:
        raise ValueError("Standard postprocess takes only --sample-manifest.")


def _sample_command(args: argparse.Namespace) -> None:
    from openghg_inversions.rhime.stages import sample, sample_rhime_stage
    from openghg_inversions.rhime.params import resolve_sampler_options

    if args.model == "standard":
        _reject_standard_artifact_args(args)
        choices = _standard_phase_options(args, phase="sample")
        result = sample(args.preparation_manifest, model=choices["model"],
                        sampler=resolve_sampler_options(choices["sampler"]),
                        output_dir=_stage_output_dir(args))
        print(result)
    else:
        _check_saved_family(args)
        result = sample_rhime_stage(setup=_stage_setup(args), model=args.model,
                                   prepared_inputs=args.prepared_inputs,
                                   preparation_manifest=args.preparation_manifest,
                                   output_dir=_stage_output_dir(args))
        print(result["artifacts"]["posterior"])


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
    from openghg_inversions.rhime.stages import postprocess, postprocess_rhime_stage
    from openghg_inversions.rhime.specs import RhimeOutputSpec

    if args.model == "standard":
        _reject_standard_artifact_args(args)
        choices = _standard_phase_options(args, phase="postprocess")["output"]
        choices["output_path"] = str(_stage_output_dir(args))
        postprocess(args.sample_manifest, output=RhimeOutputSpec.from_params(choices, multisector=False))
    else:
        _check_saved_family(args)
        postprocess_rhime_stage(setup=_stage_setup(args), model=args.model,
                               prepared_inputs=args.prepared_inputs,
                               preparation_manifest=args.preparation_manifest,
                               sample_manifest=args.sample_manifest, posterior=args.posterior,
                               output_dir=_stage_output_dir(args))


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
        "--model", choices=("standard", "multisector", "nested"), default="standard",
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
    prior_parser.add_argument("--prepared-inputs", help="Deferred multisector/CO2 route only")
    prior_parser.add_argument("--preparation-manifest", required=True)
    prior_parser.add_argument("--check-output")
    prior_parser.add_argument("--draws", type=int, default=100)
    prior_parser.add_argument("--check-stage", default=os.environ.get("STAGE", "prior-predictive"))
    prior_parser.add_argument("--strict", action="store_true", help="Exit nonzero when the check fails")
    prior_parser.set_defaults(func=_prior_predictive_command)

    sample_parser = subparsers.add_parser("sample", help="Sample explicitly supplied RHIME inputs")
    _add_stage_config_args(sample_parser, required=False)
    _add_output_dir(sample_parser)
    sample_parser.add_argument("--prepared-inputs", help="Deferred multisector/CO2 route only")
    sample_parser.add_argument("--preparation-manifest", required=True)
    sample_parser.set_defaults(func=_sample_command)

    diagnose_parser = subparsers.add_parser("diagnose", help="Calculate posterior convergence checks")
    _add_output_dir(diagnose_parser)
    diagnose_parser.add_argument("--posterior", required=True)
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
    postprocess_parser.add_argument("--prepared-inputs", help="Deferred multisector/CO2 route only")
    postprocess_parser.add_argument("--preparation-manifest", help="Deferred multisector/CO2 route only")
    postprocess_parser.add_argument("--sample-manifest", required=True)
    postprocess_parser.add_argument("--posterior", help="Deferred multisector/CO2 route only")
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
    args.func(args)


def rhime_main(argv: list[str] | None = None) -> None:
    """Run the ``rhime`` CLI with the same commands as ``openghg-inversions``.

    Args:
        argv: Optional argument vector; reads ``sys.argv`` when omitted.
    """
    main(argv, prog="rhime")


if __name__ == "__main__":
    main()
