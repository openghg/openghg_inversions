"""Static-checker regression tests for borrowed NumPy and xarray arrays."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[2]
POSITIVE_PROBE = Path(__file__).with_name("borrowed_positive.py.txt")
NEGATIVE_PROBE = Path(__file__).with_name("borrowed_negative.py.txt")
EXPECTED_MARKERS = {
    "ndarray-item": re.compile(r"Never|never|incompatible type", re.IGNORECASE),
    "ndarray-method": re.compile(r"deprecated|Never|incompatible type", re.IGNORECASE),
    "ndarray-deprecated-method": re.compile(r"deprecated", re.IGNORECASE),
    "dataarray-item": re.compile(r"Never|never|incompatible type", re.IGNORECASE),
    "dataarray-values": re.compile(r"Never|never|incompatible type|read-only", re.IGNORECASE),
    "dataarray-values-item": re.compile(r"Never|never|incompatible type", re.IGNORECASE),
    "dataarray-data-item": re.compile(r"Never|never|incompatible type", re.IGNORECASE),
    "dataarray-numpy-item": re.compile(r"Never|never|incompatible type", re.IGNORECASE),
    "shallow-copy-item": re.compile(r"Never|never|incompatible type", re.IGNORECASE),
}
CheckerResult = tuple[int, dict[int, list[str]], str]
CheckerRunner = Callable[[Path], CheckerResult]


def _materialize_probe(source: Path, destination: Path) -> dict[int, str]:
    """Copy a non-Python probe to a temporary module and return error markers."""
    text = source.read_text(encoding="utf-8")
    destination.write_text(text, encoding="utf-8")
    return {
        line_number: line.rsplit("# expect-error: ", 1)[1].strip()
        for line_number, line in enumerate(text.splitlines(), start=1)
        if "# expect-error: " in line
    }


def _checker_path(name: str) -> str:
    """Return an installed checker executable or skip outside the dev environment."""
    environment_executable = Path(sys.executable).with_name(name)
    if environment_executable.is_file():
        return str(environment_executable)
    pytest.skip(f"{name} is not installed in the active environment")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a static checker from the repository root with captured diagnostics."""
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _pyright_diagnostics(probe: Path) -> tuple[int, dict[int, list[str]], str]:
    """Run Pyright and group error messages by one-based source line."""
    config = probe.with_name("pyrightconfig.json")
    config.write_text(
        json.dumps(
            {
                "include": [str(probe)],
                "extraPaths": [str(PROJECT_ROOT)],
                "reportDeprecated": "error",
            }
        ),
        encoding="utf-8",
    )
    result = _run(
        [
            _checker_path("pyright"),
            "--outputjson",
            "--project",
            str(config),
            "--pythonpath",
            sys.executable,
            str(probe),
        ]
    )
    report = json.loads(result.stdout)
    diagnostics: dict[int, list[str]] = {}
    for diagnostic in report.get("generalDiagnostics", []):
        if diagnostic.get("severity") != "error":
            continue
        line = diagnostic["range"]["start"]["line"] + 1
        diagnostics.setdefault(line, []).append(diagnostic["message"])
    return result.returncode, diagnostics, result.stdout + result.stderr


def _mypy_diagnostics(probe: Path) -> tuple[int, dict[int, list[str]], str]:
    """Run Mypy and group error messages by one-based source line."""
    result = _run(
        [
            _checker_path("mypy"),
            "--no-incremental",
            "--cache-dir=/dev/null",
            "--no-error-summary",
            "--no-pretty",
            "--show-error-codes",
            "--enable-error-code=deprecated",
            str(probe),
        ]
    )
    diagnostic_pattern = re.compile(rf"^{re.escape(str(probe))}:(\d+): error: (.*)$")
    diagnostics: dict[int, list[str]] = {}
    for output_line in result.stdout.splitlines():
        match = diagnostic_pattern.match(output_line)
        if match:
            diagnostics.setdefault(int(match.group(1)), []).append(match.group(2))
    return result.returncode, diagnostics, result.stdout + result.stderr


@pytest.mark.parametrize("runner", [_pyright_diagnostics, _mypy_diagnostics], ids=["pyright", "mypy"])
def test_borrowed_positive_type_probe(runner: CheckerRunner, tmp_path: Path) -> None:
    """Both checkers accept reads and mutation of explicitly owned copies."""
    probe = tmp_path / "borrowed_positive.py"
    _materialize_probe(POSITIVE_PROBE, probe)

    returncode, diagnostics, output = runner(probe)

    assert returncode == 0, output
    assert diagnostics == {}, output


@pytest.mark.parametrize("runner", [_pyright_diagnostics, _mypy_diagnostics], ids=["pyright", "mypy"])
def test_borrowed_negative_type_probe(runner: CheckerRunner, tmp_path: Path) -> None:
    """Both checkers reject each marked mutation through a borrowed reference."""
    probe = tmp_path / "borrowed_negative.py"
    markers = _materialize_probe(NEGATIVE_PROBE, probe)

    returncode, diagnostics, output = runner(probe)

    assert returncode != 0, output
    assert set(diagnostics) == set(markers), output
    for line, marker in markers.items():
        messages = "\n".join(diagnostics[line])
        assert EXPECTED_MARKERS[marker].search(messages), (
            f"unexpected diagnostic for {marker!r} on line {line}:\n{messages}\n\n{output}"
        )
