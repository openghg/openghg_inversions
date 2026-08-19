"""Run ordinary RHIME with a project-owned Student-t likelihood.

This is the preferred low-ceremony customization route. It keeps RHIME's
complete acquisition-to-output pipeline and changes only the direct-Python
likelihood callable passed to :func:`openghg_inversions.rhime.run_rhime`.
Use :func:`run_with_likelihood` from Python or :func:`main` from the command
line with a normal RHIME configuration. The workflow may retrieve or reload
data, materializes related model arrays together at the named PyMC boundary,
samples, and writes requested outputs while canonical xarray/Dask inputs remain
borrowed.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from openghg_inversions.rhime import RhimeResult, run_rhime

from .likelihoods import likelihood_builder


def run_with_likelihood(
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
        TypeError: If the likelihood builder does not return a PyTensor
            variable.
        ValueError: If required options are missing, aggregation error is
            unsupported, or the likelihood omits canonical ``y`` or
            ``epsilon`` variables.

    Notes:
        This workflow may retrieve or reload data, materializes related model
        arrays together at the named PyMC boundary without mutating canonical
        prepared inputs, runs sampling, and writes configured outputs.
    """
    return run_rhime(
        config_file=config_file,
        likelihood_builder=likelihood_builder,
        **kwargs,
    )


def _json_object(value: str) -> dict[str, Any]:
    """Parse one command-line JSON object containing RHIME option overrides.

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

    Args:
        argv: Command-line tokens excluding the program name. ``None`` reads
            tokens from :data:`sys.argv`.

    Returns:
        The RHIME result returned by :func:`run_with_likelihood`.

    Raises:
        SystemExit: If argument parsing fails or help is requested.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", nargs="?", type=Path, help="RHIME INI configuration file")
    parser.add_argument(
        "--kwargs",
        type=_json_object,
        default={},
        metavar="JSON",
        help="additional RHIME options as a JSON object",
    )
    args = parser.parse_args(argv)
    return run_with_likelihood(config_file=args.config_file, **args.kwargs)


if __name__ == "__main__":
    main()
