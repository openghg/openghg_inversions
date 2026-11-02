"""Exercise archive publication using small stand-in Sphinx builds."""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.publish_docs import publish


BASE_URL = "https://openghg.github.io/openghg_inversions"


def _build(path: Path, pages: tuple[str, ...] = ("index.html",)) -> Path:
    for page in pages:
        target = path / page
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"Built {page}", encoding="utf-8")
    return path


def test_republication_replaces_only_one_version(tmp_path: Path) -> None:
    site = tmp_path / "site"
    old = _build(site / "v0.7.4", ("index.html", "removed.html"))
    _build(site / "v0.6.0", ("index.html", "old.html"))
    _build(site / "devel", ("index.html", "future.html"))
    (site / "CNAME").write_text("docs.example.org")
    _build(site / ".git", ("metadata.html",))
    build = _build(tmp_path / "build", ("index.html", "guide/topic.html", "_static/style.css"))

    publish(site, build, "v0.7.4", BASE_URL)

    assert not (old / "removed.html").exists()
    assert (old / "guide/topic.html").read_text() == "Built guide/topic.html"
    assert (old / "_static/style.css").is_file()
    assert (site / "v0.6.0/old.html").read_text() == "Built old.html"
    assert (site / "devel/future.html").read_text() == "Built future.html"
    assert (site / "CNAME").read_text() == "docs.example.org"
    assert (site / ".git/metadata.html").read_text() == "Built metadata.html"
    assert (site / ".nojekyll").is_file()


@pytest.mark.parametrize("version", ["v0.7.5", "devel", "v0.6.0"])
def test_backfills_and_devel_cannot_regress_stable(tmp_path: Path, version: str) -> None:
    site = tmp_path / "site"
    _build(site / "v0.7.9")
    _build(site / "v0.7.10", ("index.html", "guide.html"))
    build = _build(tmp_path / "build", ("index.html", "new.html"))

    publish(site, build, version, BASE_URL)

    assert f"{BASE_URL}/v0.7.10/index.html" in (site / "stable/index.html").read_text()
    assert (site / "stable/guide.html").is_file()
    assert not (site / "stable/new.html").exists()
    entries = json.loads((site / "versions.json").read_text())
    assert entries[0] == {
        "name": "0.7.10 (stable)",
        "version": "v0.7.10",
        "url": f"{BASE_URL}/v0.7.10/",
        "preferred": True,
    }
    assert entries[1]["version"] == "v0.7.9"
    assert any(entry["version"] == version for entry in entries)
    assert "Documentation versions" in (site / "versions/index.html").read_text()


def test_legacy_and_stable_deep_links_follow_current_release(tmp_path: Path) -> None:
    site = _build(tmp_path / "site", ("index.html", "guide/topic.html", "removed.html"))
    build = _build(tmp_path / "build", ("index.html", "guide/topic.html"))

    publish(site, build, "v0.7.4", BASE_URL)

    assert f"{BASE_URL}/stable/" in (site / "index.html").read_text()
    assert f"{BASE_URL}/stable/guide/topic.html" in (site / "guide/topic.html").read_text()
    assert f"{BASE_URL}/v0.7.4/guide/topic.html" in (site / "stable/guide/topic.html").read_text()
    assert "location.hash" in (site / "guide/topic.html").read_text()
    assert f"{BASE_URL}/stable/index.html" in (site / "removed.html").read_text()
    assert (site / "v0.7.4/guide/topic.html").read_text() == "Built guide/topic.html"


def test_devel_is_default_until_first_release(tmp_path: Path) -> None:
    site = tmp_path / "site"
    build = _build(tmp_path / "build")

    publish(site, build, "devel", BASE_URL + "/")

    assert f"{BASE_URL}/devel/" in (site / "index.html").read_text()
    assert json.loads((site / "versions.json").read_text()) == [
        {"name": "devel (unreleased)", "version": "devel", "url": f"{BASE_URL}/devel/"}
    ]
    assert (site / "stable/index.html").is_file()


def test_missing_index_leaves_site_untouched(tmp_path: Path) -> None:
    site = _build(tmp_path / "site", ("index.html", "v0.7.4/index.html"))
    before = {path.relative_to(site): path.read_bytes() for path in site.rglob("*") if path.is_file()}
    build = _build(tmp_path / "build", ("broken.html",))

    with pytest.raises(ValueError, match="index.html"):
        publish(site, build, "v0.7.4", BASE_URL)

    assert {path.relative_to(site): path.read_bytes() for path in site.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("version", ["../other", "v0.7", "v0.7.4rc1", "v0.7.4-rc1", "v00.7.4"])
def test_invalid_version_is_rejected_before_mutation(tmp_path: Path, version: str) -> None:
    site = tmp_path / "site"
    build = _build(tmp_path / "build")

    with pytest.raises(ValueError, match="version must"):
        publish(site, build, version, BASE_URL)

    assert not site.exists()


@pytest.mark.parametrize("build_inside_site", [False, True])
def test_overlapping_trees_are_rejected(tmp_path: Path, build_inside_site: bool) -> None:
    site = tmp_path / "site" if build_inside_site else tmp_path / "build/site"
    build = _build(site / "build" if build_inside_site else tmp_path / "build")

    with pytest.raises(ValueError, match="must not overlap"):
        publish(site, build, "devel", BASE_URL)

    assert (build / "index.html").is_file()


def test_symlink_build_is_rejected_before_replacing_release(tmp_path: Path) -> None:
    site = _build(tmp_path / "site", ("v0.7.4/index.html",))
    build = _build(tmp_path / "build")
    (build / "linked.html").symlink_to(site / "v0.7.4/index.html")

    with pytest.raises(ValueError, match="symlinks"):
        publish(site, build, "v0.7.4", BASE_URL)

    assert (site / "v0.7.4/index.html").read_text() == "Built v0.7.4/index.html"


def test_build_pages_cannot_overwrite_reserved_archive_paths(tmp_path: Path) -> None:
    site = _build(tmp_path / "site", ("v0.6.0/index.html",))
    build = _build(tmp_path / "build", ("index.html", "v0.6.0/index.html", "versions/index.html"))

    publish(site, build, "v0.7.4", BASE_URL)

    assert (site / "v0.6.0/index.html").read_text() == "Built v0.6.0/index.html"
    assert "Documentation versions" in (site / "versions/index.html").read_text()
    assert (site / "v0.7.4/versions/index.html").read_text() == "Built versions/index.html"


@pytest.mark.parametrize("versions", [("devel",), ("devel", "v0.7.4", "v0.6.0")])
def test_preview_workflow_assembles_download_layouts(tmp_path: Path, versions: tuple[str, ...]) -> None:
    """Exercise the actual shell step with single and multiple artifact layouts."""
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/docs-pr.yaml").read_text()
    step = workflow.split("      - name: Assemble preview and browsing instructions\n", 1)[1]
    script = textwrap.dedent(step.split("        run: |\n", 1)[1].split("\n      - name:", 1)[0])
    for version in versions:
        build = tmp_path / "builds"
        if len(versions) > 1:
            build /= f"docs-preview-build-{version}"
        _build(build, ("index.html", "guide.html", "_static/style.css"))
    (tmp_path / "scripts").mkdir()
    shutil.copy(root / "scripts/publish_docs.py", tmp_path / "scripts/publish_docs.py")
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"},
        check=True,
        capture_output=True,
        text=True,
    )
    preview = tmp_path / "preview"
    entries = json.loads((preview / "versions.json").read_text())
    assert {entry["version"] for entry in entries} == set(versions)
    for version in versions:
        assert (preview / version / "guide.html").is_file()
        assert (preview / version / "_static/style.css").is_file()
    assert (preview / "README.txt").is_file()
    assert (preview / "index.html").is_file()
