"""Sphinx version presentation, also appended to historical release configurations."""

import os as _os
import re as _re
from html import escape as _escape

_docs_version = _os.environ.get("DOCS_VERSION", "devel")
if _docs_version != "devel" and not _re.fullmatch(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", _docs_version):
    raise ValueError("DOCS_VERSION must be 'devel' or a release tag such as 'v0.7.4'")

release = _docs_version.removeprefix("v")
version = ".".join(release.split(".")[:2])
_docs_base_url = _os.environ.get("DOCS_BASE_URL", "https://openghg.github.io/openghg_inversions").rstrip("/")
_stable_url = _escape(f"{_docs_base_url}/stable/", quote=True)
_archive_url = _escape(f"{_docs_base_url}/versions/", quote=True)

html_theme_options = dict(globals().get("html_theme_options", {}))
html_theme_options["switcher"] = {
    "json_url": f"{_docs_base_url}/versions.json",
    "version_match": _docs_version,
}
# The shared index may not exist yet when building a new release or a local preview.
html_theme_options["check_switcher"] = False
_navbar_start = html_theme_options.get("navbar_start", ["navbar-logo"])
if isinstance(_navbar_start, str):
    _navbar_start = [item.strip() for item in _navbar_start.split(",") if item.strip()]
html_theme_options["navbar_start"] = list(_navbar_start)
if "version-switcher" not in html_theme_options["navbar_start"]:
    html_theme_options["navbar_start"].append("version-switcher")

if _docs_version == "devel":
    _announcement = "Unreleased development documentation; "
else:
    _announcement = f"Documentation for release {_docs_version}; "
html_theme_options["announcement"] = (
    f'{_announcement}use <a href="{_stable_url}">stable</a> for supported releases. '
    f'<a href="{_archive_url}">All documentation versions</a>.'
)
