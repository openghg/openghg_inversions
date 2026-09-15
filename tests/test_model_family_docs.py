"""Verify the reader paths and support boundaries in the model-family docs."""

from pathlib import Path


DOCS = Path(__file__).parents[1] / "docs" / "usage"


def _source(name: str) -> str:
    """Read one user-guide source file.

    Args:
        name: Documentation filename relative to the user-guide directory.

    Returns:
        The UTF-8 reStructuredText source.
    """
    return (DOCS / name).read_text(encoding="utf-8")


def _toctree_entries(name: str) -> list[str]:
    """Return the document names in every toctree on a page.

    Args:
        name: Documentation filename relative to the user-guide directory.

    Returns:
        The unindented document names in source order.
    """
    lines = _source(name).splitlines()
    entries: list[str] = []
    in_toctree = False
    for line in lines:
        if line == ".. toctree::":
            in_toctree = True
        elif in_toctree and line.startswith("   ") and not line.strip().startswith(":"):
            if line.strip():
                entries.append(line.strip())
        elif in_toctree and line and not line.startswith("   "):
            in_toctree = False
    return entries


def test_standard_reader_path_avoids_detailed_co2_material() -> None:
    """An ordinary reader reaches both complete recipes without CO₂ internals."""
    conceptual = _source("conceptual_inversion.rst")
    chooser = _source("model_recipes.rst")
    standard = _source("standard_model_family.rst")
    concrete = _source("concrete_rhime_model.rst")
    customising = _source("customising_rhime.rst")

    assert ":doc:`model_recipes`" in conceptual
    assert "rhime_standard_tutorial" in chooser
    assert "rhime_multisector_tutorial" in chooser
    assert "rhime_standard_tutorial" in standard
    assert "rhime_multisector_tutorial" in standard
    assert _toctree_entries("usage.rst")[:3] == [
        "installation",
        "conceptual_inversion",
        "model_recipes",
    ]
    assert "standard_model_family" in _toctree_entries("usage.rst")
    assert "rhime_standard_tutorial" in _toctree_entries("standard_model_family.rst")
    assert "rhime_multisector_tutorial" in _toctree_entries("standard_model_family.rst")
    assert "CO2 coherent-reduction model" not in concrete
    assert "Run the production cached-sigma CO2 recipe" not in customising


def test_advanced_co2_reader_path_exposes_current_boundaries() -> None:
    """A CO₂ reader can identify both recipes and every unavailable stage."""
    chooser = _source("model_recipes.rst")
    family = _source("co2_model_family.rst")
    recipes = _source("co2_models.rst")

    assert "CO₂-only" in chooser
    assert "Linked CO₂/O₂" in chooser
    assert "run_rhime_co2_o2`` workflow" in chooser
    assert "run_rhime_co2" in family
    assert "run_rhime_co2_o2_from_prepared_inputs" in family
    assert "co2_model_family" in _toctree_entries("usage.rst")
    assert _toctree_entries("co2_model_family.rst") == ["co2_models"]
    assert "Not supported by the staged CLI" in family
    assert "no complete built-in" in family
    assert "CO₂ configuration workflow" in family
    assert "scientist acceptance" in family
    assert "remain future work" in family
    assert "CO2 coherent-reduction model" in recipes
    assert "CO2/O2 shared-state model" in recipes
    assert ".. _cached-sigma-co2-recipe:" in recipes
