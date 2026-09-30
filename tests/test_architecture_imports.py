"""Canonical owners preserve import identities and installed resource access."""

from importlib import import_module
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "legacy, canonical",
    [
        ("models", "model_components"),
        ("models.components", "model_components.components"),
    ],
)
def test_legacy_exports_are_canonical_objects(legacy: str, canonical: str) -> None:
    old = import_module(f"openghg_inversions.{legacy}")
    current = import_module(f"openghg_inversions.{canonical}")
    names = getattr(old, "__all__", [name for name in vars(old) if not name.startswith("_")])
    assert names
    for name in names:
        assert getattr(old, name) is getattr(current, name), name


def test_canonical_implementations_do_not_import_legacy_owners() -> None:
    import ast

    package = Path(__file__).parents[1] / "openghg_inversions"
    for directory in ("model_components",):
        for path in (package / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                modules = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else ([alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                )
                assert not any(
                    name == f"openghg_inversions.{legacy}" or name.startswith(f"openghg_inversions.{legacy}.")
                    for name in modules
                    for legacy in ("rhime", "models", "forward", "workflow")
                ), str(path)
