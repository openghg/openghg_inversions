"""Build and locally preview the project's Sphinx documentation.

The ``preview`` command runs the repository's ``tox -e docs`` environment in
a temporary directory before serving the generated HTML on the IPv4 loopback
interface. The ``clean`` command removes legacy ``docs/_build`` output. On
macOS the preview opens in Safari unless browser opening is disabled. The
server runs until interrupted and does not expose the preview to other
machines on the network. Call :func:`main` or run the script directly; the
default port is 8765, ``--port`` overrides it, and ``--no-open`` suppresses
Safari.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


_DEFAULT_PORT = 8765
_HOST = "127.0.0.1"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DOCS_DIRECTORY = _REPOSITORY_ROOT / "docs"
_BUILD_DIRECTORY = _REPOSITORY_ROOT / "docs" / "_build"


def _port(value: str) -> int:
    """Parse and validate a TCP port supplied on the command line."""
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments for the documentation preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    preview_parser = subparsers.add_parser("preview", help="build and serve the documentation")
    preview_parser.add_argument(
        "--port",
        type=_port,
        default=_DEFAULT_PORT,
        help=f"loopback port for the preview server (default: {_DEFAULT_PORT})",
    )
    preview_parser.add_argument(
        "--no-open",
        action="store_true",
        help="do not open the preview in Safari on macOS",
    )
    subparsers.add_parser("clean", help="remove generated documentation output")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0].startswith("-"):
        arguments.insert(0, "preview")
    return parser.parse_args(arguments)


def _clean_docs() -> None:
    """Remove the documentation build output without touching source files."""
    shutil.rmtree(_BUILD_DIRECTORY, ignore_errors=True)


def _build_docs(source_directory: Path, build_directory: Path) -> int:
    """Build the HTML documentation and return the tox process exit code."""
    print("Building documentation with tox -e docs ...", flush=True)
    environment = {
        **os.environ,
        "DOCS_SOURCE_DIR": str(source_directory),
        "DOCS_BUILD_DIR": str(build_directory),
    }
    result = subprocess.run(
        ["tox", "-e", "docs"], cwd=_REPOSITORY_ROOT, check=False, env=environment
    )
    return result.returncode


def _open_safari(url: str) -> None:
    """Open a preview URL in Safari, reporting rather than raising on failure."""
    try:
        result = subprocess.run(["open", "-a", "Safari", url], check=False)
    except OSError:
        result = None
    if result is None or result.returncode:
        print(f"Could not open Safari automatically; visit {url}", file=sys.stderr)


def _serve_docs(port: int, *, open_safari: bool, build_directory: Path | None = None) -> int:
    """Serve the built documentation until interrupted.

    Args:
        port: TCP port to bind on the IPv4 loopback interface.
        open_safari: Whether to open the preview URL in Safari after binding.

    Returns:
        Zero when serving ends normally or after a keyboard interrupt, or one
        when the server cannot bind to the requested port.
    """
    handler = partial(
        SimpleHTTPRequestHandler, directory=str(build_directory or _BUILD_DIRECTORY)
    )
    try:
        server = ThreadingHTTPServer((_HOST, port), handler)
    except OSError as error:
        print(f"Could not serve documentation on {_HOST}:{port}: {error}", file=sys.stderr)
        return 1

    url = f"http://{_HOST}:{port}/"
    print(f"Serving documentation at {url} (press Ctrl-C to stop)", flush=True)
    if open_safari:
        _open_safari(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping documentation server.")
    finally:
        server.server_close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Build and serve the project documentation until interrupted.

    With the default ``preview`` command, this writes the Sphinx output to a
    temporary directory, prints the preview URL, and may open Safari on macOS.
    The ``clean`` command removes legacy output in ``docs/_build`` instead.

    Args:
        argv: Optional command-line arguments. When omitted, arguments are read
            from :data:`sys.argv`.

    Returns:
        The docs build exit code, or the local server exit code after a
        successful build.
    """
    args = _parse_args(argv)
    if args.command == "clean":
        _clean_docs()
        return 0

    with tempfile.TemporaryDirectory(prefix="openghg-inversions-docs-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        source_directory = temporary_path / "source"
        build_directory = temporary_path / "build"
        shutil.copytree(_DOCS_DIRECTORY, source_directory, ignore=shutil.ignore_patterns("_build"))

        build_status = _build_docs(source_directory, build_directory)
        if build_status:
            print("Documentation build failed; preview server was not started.", file=sys.stderr)
            return build_status

        return _serve_docs(
            args.port,
            open_safari=sys.platform == "darwin" and not args.no_open,
            build_directory=build_directory,
        )


if __name__ == "__main__":
    raise SystemExit(main())
