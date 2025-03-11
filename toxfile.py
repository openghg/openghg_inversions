"""tox plugins

The plugin defined here adds the penultimate minor version of OpenGHG
to the tox config (defined in `tox.ini`).
"""
import re
from collections import namedtuple

import requests

from tox.config.sets import ConfigSet
from tox.session.state import State
from tox.plugin import impl


version_pat = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
Version = namedtuple("Version", "major minor patch")


def get_version_tags(org: str, repo: str) -> list[Version]:
    """Get list of Versions from github repo."""
    r = requests.get(f"https://api.github.com/repos/{org}/{repo}/tags")
    version_strings = [x["name"] for x in r.json() if version_pat.match(x["name"])]
    versions = [Version(*map(int, vstr.split("."))) for vstr in version_strings]
    return versions


# get previous release version of OpenGHG
openghg_versions = get_version_tags("openghg", "openghg")
current_major, current_minor, _ = openghg_versions[0]
prev_minor = current_minor - 1
prev_openghg_release_version = next(filter(lambda v: v.minor == prev_minor, openghg_versions))
prev_openghg_release = ".".join(map(str, prev_openghg_release_version))

@impl
def tox_add_core_config(core_conf: ConfigSet, state: State) -> None:
    core_conf.add_constant("openghg_prev_minor", "penultimate minor release of OpenGHG", f"openghg=={prev_openghg_release}")
