"""Version presentation works in current and appended historical Sphinx configs."""

import runpy
from pathlib import Path

import pytest


CONFIG = Path(__file__).resolve().parents[1] / "scripts" / "docs_version_config.py"


def test_local_docs_default_to_devel(monkeypatch):
    monkeypatch.delenv("DOCS_VERSION", raising=False)
    monkeypatch.delenv("DOCS_BASE_URL", raising=False)

    config = runpy.run_path(str(CONFIG.parent.parent / "docs" / "conf.py"))

    assert config["release"] == config["version"] == "devel"
    assert config["html_title"] == "OpenGHG Inversions"
    options = config["html_theme_options"]
    assert options["switcher"] == {
        "json_url": "https://openghg.github.io/openghg_inversions/versions.json",
        "version_match": "devel",
    }
    assert options["navigation_depth"] == 4
    assert options["navbar_start"] == ["navbar-logo"]
    assert options["navbar_end"] == ["theme-switcher", "navbar-icon-links", "version-switcher"]
    assert options["check_switcher"] is False
    assert "Unreleased development documentation" in options["announcement"]


def test_appended_release_config_preserves_theme_options(monkeypatch):
    monkeypatch.setenv("DOCS_VERSION", "v0.7.4")
    monkeypatch.setenv("DOCS_BASE_URL", "https://example.org/docs/")
    namespace = {}
    historical_config = 'html_theme_options = {"navbar_start": ["navbar-logo", "custom"], "navigation_depth": 6}\n'

    exec(compile(historical_config + CONFIG.read_text(), "historical-conf.py", "exec"), namespace)

    assert namespace["release"] == "0.7.4"
    assert namespace["version"] == "0.7"
    assert namespace["html_title"] == "OpenGHG Inversions"
    options = namespace["html_theme_options"]
    assert options["switcher"] == {
        "json_url": "https://example.org/docs/versions.json",
        "version_match": "v0.7.4",
    }
    assert options["navbar_start"] == ["navbar-logo", "custom"]
    assert options["navbar_end"] == ["theme-switcher", "navbar-icon-links", "version-switcher"]
    assert options["navigation_depth"] == 6
    assert "Documentation for release v0.7.4" in options["announcement"]
    assert 'href="https://example.org/docs/stable/"' in options["announcement"]
    assert 'href="https://example.org/docs/versions/"' in options["announcement"]


def test_version_configuration_is_idempotent(monkeypatch):
    monkeypatch.setenv("DOCS_VERSION", "devel")
    options = {"navbar_start": "navbar-logo, version-switcher", "navbar_end": ["custom"]}
    config = runpy.run_path(str(CONFIG), init_globals={"html_theme_options": options})
    repeated = runpy.run_path(str(CONFIG), init_globals=config)

    assert repeated["html_theme_options"] == config["html_theme_options"]
    assert repeated["html_theme_options"]["navbar_start"] == ["navbar-logo"]
    assert repeated["html_theme_options"]["navbar_end"] == ["custom", "version-switcher"]
    assert options == {"navbar_start": "navbar-logo, version-switcher", "navbar_end": ["custom"]}


@pytest.mark.parametrize("docs_version", ["stable", "0.7.4", "v0.7", "v0.7.4rc1", "v00.7.4", "../../devel"])
def test_rejects_unsupported_release_identifiers(monkeypatch, docs_version):
    monkeypatch.setenv("DOCS_VERSION", docs_version)
    with pytest.raises(ValueError, match="DOCS_VERSION"):
        runpy.run_path(str(CONFIG))
