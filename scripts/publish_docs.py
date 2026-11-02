"""Publish one Sphinx HTML build into a checked-out GitHub Pages archive.

The caller owns checkout, build and deployment. This command replaces only the
requested version, then regenerates the switcher, archive and redirect pages.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import re
import shutil
from urllib.parse import quote, urlsplit


_RELEASE = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def _redirect(path: Path, target: str) -> None:
    """Write a redirect with a no-JavaScript fallback and preserved anchors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Documentation moved</title>"
        f'<meta http-equiv="refresh" content="0; url={escape(target, quote=True)}">'
        f'<link rel="canonical" href="{escape(target, quote=True)}">'
        f"<script>location.replace({json.dumps(target)} + location.search + location.hash);</script>"
        f'<p><a href="{escape(target, quote=True)}">Continue to the documentation</a></p></html>\n',
        encoding="utf-8",
    )


def publish(site_dir: Path, build_dir: Path, version: str, base_url: str) -> None:
    """Replace one version and regenerate aliases without losing other releases.

    Args:
        site_dir: Checked-out GitHub Pages tree, or a new empty directory.
        build_dir: Complete Sphinx HTML build, including ``index.html``.
        version: ``devel`` or a release tag such as ``v0.7.4``.
        base_url: Public site URL, including the repository path.

    Raises:
        ValueError: If inputs overlap, the build is incomplete, or a path or
            version is unsafe. Validation completes before modifying the site.
        OSError: If copying or writing the site fails.
    """
    if version != "devel" and not _RELEASE.fullmatch(version):
        raise ValueError("version must be devel or a full release tag such as v0.7.4")
    url = urlsplit(base_url)
    if (
        url.scheme not in {"https", "http"}
        or not url.netloc
        or url.query
        or url.fragment
        or any(character.isspace() or character in '<>"' for character in base_url)
    ):
        raise ValueError("base-url must be an HTTP(S) site URL without query or fragment")
    base_url = base_url.rstrip("/")
    if site_dir.is_symlink() or build_dir.is_symlink():
        raise ValueError("documentation trees must not contain symlinks")
    site = site_dir.resolve()
    build = build_dir.resolve()
    if site == build or site in build.parents or build in site.parents:
        raise ValueError("site-dir and build-dir must not overlap")
    if not (build / "index.html").is_file():
        raise ValueError("build-dir must contain index.html")
    if site.exists() and not site.is_dir():
        raise ValueError("site-dir must be a directory")
    if (build / ".git").exists():
        raise ValueError("build-dir must not contain a Git checkout")
    for directory in (build, site):
        if any(
            path.is_symlink()
            for path in directory.rglob("*")
            if directory == build or ".git" not in path.relative_to(site).parts
        ):
            raise ValueError("documentation trees must not contain symlinks")
    for name in (version, "stable", "versions"):
        if (site / name).exists() and not (site / name).is_dir():
            raise ValueError(f"publication directory is occupied by a file: {name}")

    reserved = {".git", "stable", "versions", "devel"}
    legacy_pages = {
        path.relative_to(site)
        for path in site.rglob("*.html")
        if path.relative_to(site).parts[0] not in reserved
        and not _RELEASE.fullmatch(path.relative_to(site).parts[0])
    }
    destination = site / version
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(build, destination)

    releases = sorted(
        (
            path.name
            for path in site.iterdir()
            if _RELEASE.fullmatch(path.name) and (path / "index.html").is_file()
        ),
        key=lambda name: tuple(map(int, name[1:].split("."))),
        reverse=True,
    )
    default = releases[0] if releases else "devel"
    entries = [
        {
            "name": name[1:] + (" (stable)" if name == default else ""),
            "version": name,
            "url": f"{base_url}/{name}/",
            **({"preferred": True} if name == default else {}),
        }
        for name in releases
    ]
    if (site / "devel" / "index.html").is_file():
        entries.append({"name": "devel (unreleased)", "version": "devel", "url": f"{base_url}/devel/"})
    (site / "versions.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    archive = site / "versions"
    archive.mkdir(exist_ok=True)
    links = "\n".join(
        f'<li><a href="{escape(entry["url"], quote=True)}">{escape(entry["name"])}</a></li>'
        for entry in entries
    )
    (archive / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Documentation versions</title><h1>Documentation versions</h1>"
        f"<ul>{links}</ul></html>\n",
        encoding="utf-8",
    )
    stable = site / "stable"
    if stable.exists():
        shutil.rmtree(stable)
    current_pages = {path.relative_to(site / default) for path in (site / default).rglob("*.html")}
    for page in current_pages:
        _redirect(stable / page, f"{base_url}/{default}/{quote(page.as_posix())}")
    for page in legacy_pages | current_pages:
        if page.parts[0] in reserved or _RELEASE.fullmatch(page.parts[0]):
            continue
        target_page = page if page in current_pages else Path("index.html")
        _redirect(site / page, f"{base_url}/stable/{quote(target_page.as_posix())}")
    _redirect(site / "index.html", f"{base_url}/{'stable' if releases else 'devel'}/")
    (site / ".nojekyll").touch()


def main() -> None:
    """Parse publication arguments and update the local Pages checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    try:
        publish(args.site_dir, args.build_dir, args.version, args.base_url)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Documentation publication failed: {error}\n")


if __name__ == "__main__":
    main()
