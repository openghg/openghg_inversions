"""Installed-package provenance collection independent of scientific recipes."""

from importlib import metadata
import json
from pathlib import Path
import subprocess
from typing import Any


def installed_ogi_provenance() -> dict[str, Any]:
    """Collect installed version and available Git revision for stage sidecars.

    Editable checkouts also report tracked-file changes. A wheel without VCS
    provenance returns ``revision=None``; each recipe owns its revision policy.
    Provenance is descriptive: an uninstalled package or unavailable Git gives
    ``None`` values rather than failing execution.
    """
    try:
        distribution = metadata.distribution("openghg-inversions")
    except metadata.PackageNotFoundError:
        return {"version": None, "revision": None, "dirty": None}
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    revision = direct.get("vcs_info", {}).get("commit_id")
    dirty = None
    source = Path(__file__).resolve().parents[1]
    if revision is None and (source / ".git").exists():
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain", "--untracked-files=no"],
                    cwd=source,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
        except (OSError, subprocess.CalledProcessError):
            revision = dirty = None
    return {"version": distribution.version, "revision": revision, "dirty": dirty}
