"""Canonical resource comparisons; original #960 source cases remain unchanged.

These tests do not execute the retired historical selector or claim historical
artifact compatibility. Independent basis reader/writer coverage remains with
its existing tests and #951.
"""

import json
import shutil
from pathlib import Path

import pytest

from openghg_inversions.rhime.requests import load_params

FIXTURES = Path(__file__).parent / "fixtures" / "migration_960" / "resources"
CASES = json.loads((FIXTURES / "cases.json").read_text())


def _plain(value, root):
    if isinstance(value, Path):
        return value.relative_to(root).as_posix()
    if isinstance(value, dict):
        return {key: _plain(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, root) for item in value]
    return value


@pytest.mark.parametrize("case", [case for case in CASES if "expected" in case], ids=lambda case: case["id"])
def test_recorded_exact_resource_selections_use_canonical_inputs(case, tmp_path, monkeypatch):
    """The selected path meaning is preserved; no discovery/converter is run."""
    monkeypatch.chdir(tmp_path)
    grouped = load_params(overrides=case["expected"])
    actual = {name: value for group in grouped.values() for name, value in group.items()}
    assert _plain(actual, tmp_path) == case["expected"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_original_historical_resource_selectors_are_rejected_before_discovery(case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def forbidden_glob(*args, **kwargs):
        pytest.fail("retired selections must fail before discovery")

    monkeypatch.setattr(Path, "glob", forbidden_glob)
    with pytest.raises(ValueError):
        load_params(overrides={**case["params"], **case.get("overrides", {})})


def test_characterize_entrypoint_paths_are_invocation_relative(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    source = config_dir / "invocation.ini"
    shutil.copyfile(FIXTURES / "invocation.ini", source)
    monkeypatch.chdir(tmp_path)
    expected = json.loads((FIXTURES / "invocation.expected.json").read_text())
    grouped = load_params(config_file=source)
    assert _plain({name: value for group in grouped.values() for name, value in group.items()}, tmp_path) == expected
