"""Static-checker probes for centered boundary transport contracts."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from test_borrowed_typing import CheckerRunner, _checker_path, _materialize_probe, _pyright_diagnostics, _run


POSITIVE_PROBE = Path(__file__).with_name("contracts_positive.py.txt")
NEGATIVE_PROBE = Path(__file__).with_name("contracts_negative.py.txt")
INCOMPATIBLE = re.compile(r"incompatible type|not assignable|cannot be assigned", re.IGNORECASE)


def _mypy_diagnostics(probe: Path) -> tuple[int, dict[int, list[str]], str]:
    """Run Mypy on the probe alone, silencing diagnostics from imported package modules."""
    result = _run([
        _checker_path("mypy"), "--no-incremental", "--cache-dir=/dev/null", "--no-error-summary", "--no-pretty",
        "--show-absolute-path", "--show-error-codes", "--follow-imports=silent", str(probe),
    ])
    pattern = re.compile(rf"^{re.escape(str(probe))}:(\d+): error: (.*)$")
    diagnostics: dict[int, list[str]] = {}
    for line in result.stdout.splitlines():
        if match := pattern.match(line):
            diagnostics.setdefault(int(match.group(1)), []).append(match.group(2))
    return result.returncode, diagnostics, result.stdout + result.stderr


@pytest.mark.parametrize("runner", [_pyright_diagnostics, _mypy_diagnostics], ids=["pyright", "mypy"])
def test_contract_positive_type_probe(runner: CheckerRunner, tmp_path: Path) -> None:
    """Validator and producer returns are accepted as contracts and as base Datasets."""
    probe = tmp_path / "contracts_positive.py"
    _materialize_probe(POSITIVE_PROBE, probe)

    returncode, diagnostics, output = runner(probe)

    assert returncode == 0, output
    assert diagnostics == {}, output


@pytest.mark.parametrize("runner", [_pyright_diagnostics, _mypy_diagnostics], ids=["pyright", "mypy"])
def test_contract_negative_type_probe(runner: CheckerRunner, tmp_path: Path) -> None:
    """Plain Datasets and weaker contracts are rejected at typed handoffs."""
    probe = tmp_path / "contracts_negative.py"
    markers = _materialize_probe(NEGATIVE_PROBE, probe)

    returncode, diagnostics, output = runner(probe)

    assert returncode != 0, output
    assert set(diagnostics) == set(markers), output
    for line in markers:
        assert INCOMPATIBLE.search("\n".join(diagnostics[line])), output
