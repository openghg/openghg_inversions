"""Modern production code must remain independent of the retiring package."""

import ast
from importlib.util import resolve_name
from pathlib import Path
import subprocess
import sys

import pytest


LEGACY = "openghg_inversions.hbmcmc"


def _legacy_imports(source, package):
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            module = resolve_name(module, package) if node.level else module
            names = [module, *(f"{module}.{alias.name}" for alias in node.names)]
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", None)
            if name in {"import_module", "__import__"} and isinstance(node.args[0], ast.Constant):
                value = node.args[0].value
                if isinstance(value, str):
                    names = [resolve_name(value, package) if value.startswith(".") else value]
        if any(name == LEGACY or name.startswith(LEGACY + ".") for name in names):
            yield node.lineno


def test_production_has_no_hbmcmc_imports():
    root = Path(__file__).resolve().parents[1]
    violations = []
    for source_root in (root / "openghg_inversions", root / "examples"):
        for path in source_root.rglob("*.py"):
            relative = path.relative_to(root)
            if relative.parts[:2] == ("openghg_inversions", "hbmcmc"):
                continue
            package = ".".join(relative.parts[:-1])
            violations.extend(f"{relative}:{line}" for line in _legacy_imports(path.read_text(), package))
    assert not violations, "Forbidden legacy imports: " + ", ".join(violations)


@pytest.mark.parametrize("source", [
    "import openghg_inversions.hbmcmc.compatibility",
    "from openghg_inversions import hbmcmc",
    "from ..hbmcmc import compatibility as reexport",
    "importlib.import_module('openghg_inversions.hbmcmc.compatibility')",
])
def test_import_boundary_detects_forbidden_dependencies(source):
    assert list(_legacy_imports(source, "openghg_inversions.rhime")) == [1]


def test_modern_entry_routes_work_with_legacy_imports_blocked():
    script = '''
import importlib.abc
import sys
import tempfile

class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "openghg_inversions.hbmcmc" or fullname.startswith("openghg_inversions.hbmcmc."):
            raise AssertionError("Modern route attempted legacy import: " + fullname)

sys.meta_path.insert(0, BlockLegacy())
from openghg_inversions.rhime import RhimeConfig
from openghg_inversions.rhime.specs import make_output_spec
from openghg_inversions.rhime.stages import prepare, resolve_stage_setup
from openghg_inversions.rhime.standard import run_standard
from openghg_inversions.rhime.multisector import run_rhime_multisector
from openghg_inversions.rhime.nested import run_nested
from openghg_inversions.inversion_data import RhimeMergedData
from tests.test_staged_workflow import _params

class Acquired(Exception):
    pass

def acquire(**kwargs):
    raise Acquired

RhimeMergedData.from_options = acquire
options = _params(output_format="none", save_inversion_output=False, basis_algorithm="quadtree")
config = RhimeConfig.from_params(options, multisector=False)
resolve_stage_setup({**options, "flux_sources": ["a", "b"]}, model="multisector")
make_output_spec(output_format="none", save_inversion_output=False, multisector=False)
for call in (
    lambda: run_standard(config=config),
    lambda: prepare(config.data, output_dir=tempfile.mkdtemp() + "/prepared"),
    lambda: run_standard(**options),
    lambda: run_rhime_multisector(**{**options, "flux_sources": ["a", "b"]}),
    lambda: run_nested(**options, inner_domain="inner"),
):
    try:
        call()
    except Acquired:
        pass
    else:
        raise AssertionError("Runner did not reach acquisition")
assert not any(name == "openghg_inversions.hbmcmc" or name.startswith("openghg_inversions.hbmcmc.") for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
