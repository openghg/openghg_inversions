"""Execute the RHIME tutorial notebooks and refresh their committed outputs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterator, Sequence

import nbformat


_ROOT = Path(__file__).resolve().parents[1]
_DATA_REPOSITORY = "git@github.com:openghg/openghg_inversions_tutorial_data.git"
_DATA_TAG = "v1.0.0"
_DEFAULT_DATA_DIRECTORY = _ROOT / "build" / f"tutorial-data-{_DATA_TAG}"
_NOTEBOOK_DIRECTORY = _ROOT / "docs" / "_build" / "jupyter_execute" / "usage"
_RUN_DIRECTORY = _ROOT / "docs" / "_build" / "tutorial-runs"
_RECORDER_HOME = _RUN_DIRECTORY / "home"
_TUTORIALS = {
    "rhime_standard_tutorial": _ROOT / "docs" / "usage" / "rhime_standard_tutorial.rst",
    "rhime_multisector_tutorial": _ROOT / "docs" / "usage" / "rhime_multisector_tutorial.rst",
}


@dataclass(frozen=True)
class _Directive:
    """One line-oriented reStructuredText directive."""

    start: int
    end: int
    body: str


def _run(command: Sequence[str], *, cwd: Path = _ROOT, env: dict[str, str] | None = None) -> str:
    """Run one required command and return stripped standard output."""
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode:
        details = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{details}")
    return result.stdout.strip()


def _directives(lines: list[str], name: str) -> list[_Directive]:
    """Return all simple indented directives named ``name``."""
    marker = f".. {name}::"
    blocks: list[_Directive] = []
    index = 0
    while index < len(lines):
        if lines[index] != marker:
            index += 1
            continue
        start = index
        index += 1
        body_lines: list[str] = []
        while index < len(lines) and (not lines[index] or lines[index].startswith("   ")):
            body_lines.append(lines[index][3:] if lines[index].startswith("   ") else "")
            index += 1
        while body_lines and not body_lines[0]:
            body_lines.pop(0)
        while body_lines and not body_lines[-1]:
            body_lines.pop()
        blocks.append(_Directive(start=start, end=index, body="\n".join(body_lines)))
    return blocks


def _replace_outputs(document: str, outputs: Sequence[str]) -> str:
    """Replace paired ``jupyter-output`` bodies without changing inputs."""
    lines = document.splitlines()
    inputs = _directives(lines, "jupyter-input")
    existing_outputs = _directives(lines, "jupyter-output")
    if len(inputs) != len(existing_outputs) or len(inputs) != len(outputs):
        raise ValueError(
            "Tutorial input, committed output, and recorded output counts must match: "
            f"{len(inputs)}, {len(existing_outputs)}, {len(outputs)}"
        )
    for input_block, output_block in zip(inputs, existing_outputs):
        between = lines[input_block.end : output_block.start]
        if any(line.strip() for line in between):
            raise ValueError("Each jupyter-output must immediately follow its jupyter-input.")

    for block, output in reversed(list(zip(existing_outputs, outputs))):
        rendered = [".. jupyter-output::", ""]
        rendered.extend(f"   {line}" if line else "" for line in output.rstrip().splitlines())
        rendered.append("")
        lines[block.start : block.end] = rendered
    return "\n".join(lines).rstrip() + "\n"


def _recorded_outputs(notebook: nbformat.NotebookNode) -> list[str]:
    """Extract stable text outputs from executed code cells."""
    recorded: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        text_outputs: list[str] = []
        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "error":
                raise RuntimeError("\n".join(output.get("traceback", [])))
            if output_type == "stream":
                continue
            data = output.get("data", {})
            if output_type == "execute_result" and "text/plain" in data:
                text_outputs.append(str(data["text/plain"]).rstrip())
        if not text_outputs:
            raise RuntimeError("Every tutorial input cell must produce a text/plain result.")
        recorded.append("\n".join(text_outputs))
    return recorded


def _require_clean_checkout() -> str:
    """Return the exact source commit, refusing uncommitted tracked changes."""
    if _run(["git", "status", "--porcelain", "--untracked-files=no"]):
        raise RuntimeError("Commit or restore tracked changes before recording tutorial outputs.")
    return _run(["git", "rev-parse", "HEAD"])


def _prepare_data(directory: Path) -> None:
    """Obtain, verify, and populate the pinned companion-data release."""
    if not directory.exists():
        _run(
            [
                "git",
                "clone",
                "--branch",
                _DATA_TAG,
                "--depth",
                "1",
                _DATA_REPOSITORY,
                str(directory),
            ]
        )
    head = _run(["git", "rev-parse", "HEAD"], cwd=directory)
    tag = _run(["git", "rev-list", "-n", "1", _DATA_TAG], cwd=directory)
    if head != tag:
        raise RuntimeError(f"{directory} is not checked out at {_DATA_TAG} ({tag}).")
    _run(["git", "lfs", "pull"], cwd=directory)
    _RECORDER_HOME.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HOME": str(_RECORDER_HOME)}
    _run([sys.executable, "scripts/populate_store.py"], cwd=directory, env=env)


@contextmanager
def _recording_environment(code_ref: str, output_path: Path) -> Iterator[None]:
    """Expose stable recording metadata and an isolated output directory."""
    values = {
        "HOME": str(_RECORDER_HOME),
        "OPENGHG_TUTORIAL_CODE_REF": code_ref,
        "OPENGHG_TUTORIAL_DATA_TAG": _DATA_TAG,
        "OPENGHG_TUTORIAL_OUTPUT_PATH": str(output_path),
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _execute_notebook(name: str, code_ref: str) -> list[str]:
    """Execute one generated notebook and return its stable text results."""
    from nbconvert.preprocessors import ExecutePreprocessor

    notebook_path = _NOTEBOOK_DIRECTORY / f"{name}.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    run_directory = _RUN_DIRECTORY / name
    run_directory.mkdir(parents=True, exist_ok=True)
    output_path = run_directory / "outputs"
    output_path.mkdir(exist_ok=True)
    executor = ExecutePreprocessor(timeout=7200, kernel_name="python3")
    with _recording_environment(code_ref, output_path):
        executor.preprocess(notebook, {"metadata": {"path": str(run_directory)}})
    return _recorded_outputs(notebook)


def _build_docs() -> None:
    """Build manual-cell notebooks and rendered pages without executing inputs."""
    subprocess.run(
        ["sphinx-build", "-M", "html", "docs", "docs/_build", "--keep-going"],
        cwd=_ROOT,
        check=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit recorder command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=_DEFAULT_DATA_DIRECTORY)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Refresh both tutorials from one clean code commit and data release."""
    args = parse_args(argv)
    code_ref = _require_clean_checkout()
    _prepare_data(args.data_directory.resolve())
    _build_docs()
    refreshed: dict[Path, str] = {}
    for name, document_path in _TUTORIALS.items():
        document = document_path.read_text(encoding="utf-8")
        refreshed[document_path] = _replace_outputs(
            document,
            _execute_notebook(name, code_ref),
        )
    for path, content in refreshed.items():
        path.write_text(content, encoding="utf-8")
    _build_docs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
