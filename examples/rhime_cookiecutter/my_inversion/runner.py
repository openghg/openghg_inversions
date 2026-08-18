"""Run RHIME from a cookiecutter-generated project package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from openghg_inversions.rhime import RhimeResult, run_rhime

from .likelihoods import likelihood_builder


def run(
    *,
    config_file: str | Path | None = None,
    **kwargs: Any,
) -> RhimeResult:
    """Run ordinary RHIME with this project's likelihood.

    Args:
        config_file: Optional RHIME INI configuration file.
        **kwargs: Standard RHIME option names overriding the configuration.

    Returns:
        The library-owned RHIME result and any requested supported output.
    """
    return run_rhime(
        config_file=config_file,
        likelihood_builder=likelihood_builder,
        **kwargs,
    )


def _json_object(value: str) -> dict[str, Any]:
    """Parse JSON command-line overrides as an object."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--kwargs must decode to a JSON object")
    return parsed


def main(argv: Sequence[str] | None = None) -> RhimeResult:
    """Parse project CLI arguments and run the customized inversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", type=Path, help="RHIME INI configuration file")
    parser.add_argument(
        "--kwargs",
        type=_json_object,
        default={},
        metavar="JSON",
        help="additional RHIME options as a JSON object",
    )
    args = parser.parse_args(argv)
    return run(config_file=args.config_file, **args.kwargs)


if __name__ == "__main__":
    main()
