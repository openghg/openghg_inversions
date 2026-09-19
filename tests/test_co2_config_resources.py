"""Check that the CO2-family TOML examples ship as readable resources."""

from importlib.resources import files
from pathlib import Path
import tomllib


_CONFIG_DIRECTORY = files("openghg_inversions.rhime").joinpath("config")
_EXPECTED_TEMPLATES = {
    "co2.toml": ("co2", "ordinary"),
    "co2_cached_sigma.toml": ("co2", "cached_fixed_ou"),
    "co2_o2.toml": ("co2_o2", "linked"),
}


def test_co2_family_templates_are_discoverable_and_parseable() -> None:
    """Installed resources expose one valid TOML example for each supported route."""
    resources = {
        resource.name: resource
        for resource in _CONFIG_DIRECTORY.iterdir()
        if resource.name.endswith(".toml")
    }

    assert resources.keys() == _EXPECTED_TEMPLATES.keys()
    for name, (recipe, variant) in _EXPECTED_TEMPLATES.items():
        with resources[name].open("rb") as stream:
            config = tomllib.load(stream)
        assert config["format_version"] == 1
        assert config["recipe"] == recipe
        assert config["variant"] == variant


def test_co2_family_templates_are_declared_as_package_data() -> None:
    """The setuptools wheel configuration includes the discovered TOML resources."""
    project_root = Path(__file__).parents[1]
    with (project_root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    assert "rhime/config/*.toml" in project["tool"]["setuptools"]["package-data"][
        "openghg_inversions"
    ]
    assert not any(dependency.startswith("tomli") for dependency in project["project"]["dependencies"])
