"""Verify the reader paths and support boundaries in the model-family docs."""

import importlib
import re
from pathlib import Path

from openghg_inversions.rhime.params import resolve_rhime_options
from openghg_inversions.rhime.specs import PollutionEventSettings


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
    assert _toctree_entries("usage.rst")[:4] == [
        "installation",
        "getting_started",
        "conceptual_inversion",
        "model_recipes",
    ]
    assert "standard_model_family" in _toctree_entries("usage.rst")
    assert "rhime_standard_tutorial" in _toctree_entries("standard_model_family.rst")
    assert "rhime_multisector_tutorial" in _toctree_entries("standard_model_family.rst")
    assert _toctree_entries("shared_scientific_concepts.rst") == [
        "grouped_basis_layout",
        "native_covariance",
        "retained_state_reconstruction",
        "affine_flux_map",
        "coherent_reduction",
    ]
    assert "CO2 coherent-reduction model" not in concrete
    assert "Run the production cached-sigma CO2 recipe" not in customising
    assert "scalar-sigma-eigenbasis.nc" not in customising


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
    assert "packaged TOML template" in family
    assert "configuration boundary for the existing Python runners" in family
    assert "scientist acceptance" in family
    assert "remain future work" in family
    assert "CO2 coherent-reduction model" in recipes
    assert "CO2/O2 shared-state model" in recipes
    assert ".. _co2-scalar-sigma-recipe:" in recipes
    assert "aggregation_error_covariance" in recipes
    assert "Unit conversion is caller-owned" in recipes
    assert ".. _co2-cached-sigma-recipe:" in recipes
    assert "Run the ordinary prepared-input CO2 runner" in recipes
    assert "Configure prepared-input replay from TOML" in recipes
    assert "load_co2_family_config" in recipes
    assert "resolve_co2_family_config" in recipes
    assert "co2_cached_sigma.toml" in recipes
    assert "Relative values are interpreted from the process" in recipes
    assert "Ordinary CO2 likelihood configuration" in recipes
    assert "Exactly one of ``fixed_site_amplitudes``" in recipes
    assert "Sampling configuration" in recipes
    assert "Non-negative integer strictly less than ``draws``" in recipes
    assert "the linked recipe overrides the\ntwo defaults" in recipes
    assert "``[channels.co2]`` and ``[channels.o2]``" in recipes
    assert "standard deviations are 1 ppm\nand 2 ppm" in recipes
    assert "The resolver does not convert these values" in recipes
    assert "prepare_co2_o2_inputs(" in recipes
    assert 'co2_units=setup.preparation_kwargs["co2_units"]' in recipes
    assert "A runnable CO2 configuration and resolver are\ntracked" not in recipes
    assert "``co2_o2``" in recipes
    assert "heterogeneous ppm/per-meg" in recipes
    assert "OPE-86" in recipes
    assert "idata = run_rhime_co2(" in recipes
    assert "idata = run_rhime_co2_cached_sigma(" in recipes
    assert recipes.count("use_bc=True") >= 2
    assert "Channel-labelled boundary sensitivities, priors and activity" in family
    assert "independent global, site, or site-by-period offsets" in family
    assert "= \\mathtt{co2\\_flux\\_contribution}" in recipes
    assert "validate_complete_observation_covariance" not in recipes


def test_nested_reader_path_exposes_transport_grid_boundary() -> None:
    """A nested reader reaches one canonical guide with accurate support."""
    chooser = _source("model_recipes.rst")
    family = _source("nested_domain_model_family.rst")
    guide = _source("nested_domains.rst")
    readme = (DOCS.parents[1] / "README.md").read_text(encoding="utf-8")
    usage = _toctree_entries("usage.rst")

    assert "Nested-domain" in chooser
    family_start = usage.index("standard_model_family")
    assert usage[family_start : family_start + 3] == [
        "standard_model_family",
        "nested_domain_model_family",
        "co2_model_family",
    ]
    assert _toctree_entries("nested_domain_model_family.rst") == ["nested_domains"]
    assert "run_rhime_nested" in family
    assert "run_rhime_nested_from_prepared_inputs" in family
    assert "generic staged commands do" in family
    assert 'output_format="none"' in family
    assert 'output_format="paris"' in family
    assert "grouped basis regions" in guide
    assert "mu_outer" in guide
    assert "inner_x_prior" in guide
    assert "inner_fp_basis_case" in guide
    assert "inner_basis_directory" in guide
    assert "inner_basis_output_path" in guide
    assert guide.count('mismatch_model="pollution_event"') == 2
    assert "outer_overlap_masked=True" in guide
    assert "rather than appending a duplicate section" in guide
    assert "source country file and exact\n  target coordinates" in guide
    assert "sampling-only" not in readme


def test_nested_documented_examples_resolve_supported_likelihoods() -> None:
    """Both documented runner shapes include a resolvable likelihood."""
    examples = (
        {
            "species": "ch4",
            "sites": ["TAC", "MHD"],
            "averaging_period": ["1h", "1h"],
            "domain": "EUROPE",
            "start_date": "2019-01-01",
            "end_date": "2019-02-01",
            "flux_sources": ["total-ukghg-edgar7"],
            "mismatch_model": "pollution_event",
            "output_name": "europe_nested_6km",
            "output_format": "none",
        },
        {
            "species": "ch4",
            "sites": ["GOSAT-BRAZIL"],
            "averaging_period": ["1D"],
            "platform": ["satellite"],
            "inlet": ["column"],
            "fp_height": ["column"],
            "max_level": [3],
            "domain": "BRAZIL",
            "start_date": "2019-01-01",
            "end_date": "2019-02-01",
            "flux_sources": ["total-inventory"],
            "mismatch_model": "pollution_event",
            "output_name": "gosat_brazil_nested",
            "output_format": "none",
        },
    )

    for params in examples:
        setup = resolve_rhime_options(params=params, multisector=False)
        assert isinstance(setup.run_spec.model.likelihood, PollutionEventSettings)


def test_moved_recipe_sections_preserve_legacy_fragment_targets() -> None:
    """Moved CO₂ sections retain their deployed fragment identifiers."""
    concrete = _source("concrete_rhime_model.rst")
    customising = _source("customising_rhime.rst")
    recipes = _source("co2_models.rst")

    assert ".. _co2-coherent-reduction-model:" in concrete
    assert "<co2-only-model>" in concrete
    assert ".. _co2-only-model:" in recipes
    assert ".. _co2-grouped-inner-and-outer-states:" in concrete
    assert "<co2-grouped-states>" in concrete
    assert ".. _co2-grouped-states:" in recipes
    assert ".. _co2-o2-shared-state-model:" in concrete
    assert "<linked-co2-o2-model>" in concrete
    assert ".. _linked-co2-o2-model:" in recipes
    assert ".. _cached-sigma-co2-recipe:" in customising
    assert ".. _run-the-production-cached-sigma-co2-recipe:" in customising
    assert "<co2-cached-sigma-recipe>" in customising


def test_co2_python_domain_references_resolve() -> None:
    """Every fully qualified CO₂ function and class reference exists."""
    targets: list[str] = []
    for name in ("co2_model_family.rst", "co2_models.rst"):
        targets.extend(re.findall(r":(?:func|class):`~?([^`]+)`", _source(name)))

    assert targets
    for target in targets:
        module_name, attribute = target.rsplit(".", maxsplit=1)
        module = importlib.import_module(module_name)
        assert hasattr(module, attribute), target
