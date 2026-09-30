"""Canonical owners preserve import identities and installed resource access."""

from importlib import import_module, resources
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "legacy, canonical",
    [
        ("rhime", "recipes"),
        ("rhime.standard", "recipes.standard"),
        ("rhime.multisector", "recipes.multisector"),
        ("rhime.nested", "recipes.nested"),
        ("rhime.prepared", "recipes.from_prepared"),
        ("rhime.preparation", "recipes.preparation_adapters"),
        ("rhime.cached_sigma", "inference.cached_sigma"),
        ("rhime.co2", "recipes.co2"),
        ("models", "model_components"),
        ("models.components", "model_components.components"),
        ("inversion_data.prepared", "inversion_data.prepared_inputs"),
        ("postprocessing.reconstruction", "postprocessing.output_views"),
        ("postprocessing.linked_paris_outputs", "recipes.co2.outputs"),
        ("forward.domain_support", "recipes._domain_support"),
        ("workflow.artifacts", "recipes._stage_artifacts"),
    ],
)
def test_legacy_exports_are_canonical_objects(legacy: str, canonical: str) -> None:
    old = import_module(f"openghg_inversions.{legacy}")
    current = import_module(f"openghg_inversions.{canonical}")
    names = getattr(old, "__all__", [name for name in vars(old) if not name.startswith("_")])
    assert names
    for name in names:
        assert getattr(old, name) is getattr(current, name), name


def test_recipe_config_resources_match_legacy_locations() -> None:
    old = resources.files("openghg_inversions.rhime").joinpath("config")
    current = resources.files("openghg_inversions.recipes").joinpath("config")
    old_names = {file.name for file in old.iterdir() if file.is_file()}
    assert old_names == {file.name for file in current.iterdir() if file.is_file()}
    for name in old_names:
        assert old.joinpath(name).read_bytes() == current.joinpath(name).read_bytes()


def test_canonical_implementations_do_not_import_legacy_owners() -> None:
    import ast

    package = Path(__file__).parents[1] / "openghg_inversions"
    for directory in ("recipes", "model_components", "inference"):
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
