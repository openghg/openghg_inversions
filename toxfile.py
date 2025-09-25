"""tox plugins

The plugin defined here adds the penultimate minor version of OpenGHG
to the tox config (defined in `tox.ini`).
"""
import re
from collections import namedtuple

from tox.plugin import impl

version_pat = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
Version = namedtuple("Version", "major minor patch")

@impl
def tox_add_core_config(core_conf, state):
    import urllib.request
    import json

    def get_version_tags(org: str, repo: str):
        url = f"https://api.github.com/repos/{org}/{repo}/tags"
        with urllib.request.urlopen(url) as response:
            tags = json.load(response)
        version_strings = [x["name"] for x in tags if version_pat.match(x["name"])]
        versions = [Version(*map(int, vstr.split("."))) for vstr in version_strings]
        return versions

    openghg_versions = get_version_tags("openghg", "openghg")
    _, current_minor, _ = openghg_versions[0]
    prev_minor = current_minor - 1
    prev_openghg_release_version = next(filter(lambda v: v.minor == prev_minor, openghg_versions))
    prev_openghg_release = ".".join(map(str, prev_openghg_release_version))
    core_conf.add_constant(
        "openghg_prev_minor",
        "penultimate minor release of OpenGHG",
        f"openghg=={prev_openghg_release}"
    )
